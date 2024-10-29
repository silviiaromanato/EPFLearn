import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score

from evaluate import save_hf_model
from reward_dataset import *

os.environ["NO_DEPRECATION_WARNING"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
#os.environ["WANDB_DISABLED"] = "true"

TEAM_NAME = 'syntax-sorcerers'
BASE_MODEL_NAME = 'roberta-base'
DATASET_NAMES = ['StackOverflow', 'EPFL']


# Create directories for saving models and runs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DATA_DIR = os.path.join(DATA_DIR, "reward_model/")
REWARD_MODEL_DIR = os.path.join(BASE_DIR, 'reward_model')
RUNS_DIR = os.path.join(REWARD_MODEL_DIR, 'runs')
LOGS_DIR = os.path.join(REWARD_MODEL_DIR, 'logs')
EVAL_DIR = os.path.join(RUNS_DIR, 'eval')



# Set up the training arguments
TRAINING_ARGS = TrainingArguments(   
    output_dir=RUNS_DIR,                # Output directory (overwrite if exists)
    overwrite_output_dir=True,          
    evaluation_strategy="epoch",        # Run evaluation on validation set every epoch
    save_strategy="epoch",              # Save checkpoint every epoch      
    logging_strategy="epoch",           # Log loss every epoch     
    num_train_epochs=20,                # Number of training epochs per dataset
    optim="adamw_torch",                # Optimizer: AdamW     
    load_best_model_at_end=True,        # Load the best model when finished training     
    metric_for_best_model="loss",       # Best model minimizes weighted CE loss     
    #fp16=True,                          # Mixed precision training for faster training, only works on GPUs     
    report_to='wandb', 
    seed=7,
    gradient_accumulation_steps=4,       # Accumulate steps before backward pass

    # Play with training hyperparameters:
    learning_rate=1e-5,                 # Initial learning rate
    weight_decay=0.1,                   # L2 regularization strength 
    per_device_train_batch_size=16,     # Number of QA per batch (default=8)
    per_device_eval_batch_size=16,      # Number of QA per batch (default=8)    
)
    

# ----------------- Experiment 2: Classifier Reward Model ----------------- #

def load_dataset(dataset_name, tokenizer):
    ''' Load the dataset for the given dataset name. '''
    dataset_path = os.path.join(
        DATA_DIR, f'm2_reward_dataset_{TEAM_NAME}_{dataset_name}.json')
    df = pd.read_json(dataset_path)
    dataset = create_dataset(df, tokenizer)
    return dataset

def train_classifier(): 
    ''' Train a classifier to predict correctness of Question-Answer pairs. '''

    # Load the pre-trained model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # Move the model to the GPU if available
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    model.to(device)

    # Load the datasets
    datasets = [load_dataset(dataset_name, tokenizer) for dataset_name in DATASET_NAMES]

    # Train the model on each dataset sequentially
    for dataset_idx, (dataset_name, dataset) in enumerate(zip(DATASET_NAMES, datasets)):

        # Adapt training arguments
        TRAINING_ARGS.logging_dir = os.path.join(LOGS_DIR, f'training_logs_{dataset_name}')
        TRAINING_ARGS.run_name = f'run_{dataset_name}'
        if dataset_idx > 0: 
            # Decrease learning rate for subsequent datasets
            TRAINING_ARGS.learning_rate = 1e-6

        # Create the Trainer and run training
        trainer = WeightedTrainer(
            model=model,
            args=TRAINING_ARGS,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            compute_metrics=compute_metrics,       
        )
        print(f'\nTraining on dataset [{dataset_idx+1}/{len(datasets)}] ({dataset_name}).')
        trainer.train() # Resume from last checkpoint

        # Evaluate the model on test sets and save results
        for i in range(len(datasets)):
            print(f'\nEvaluating on dataset [{i+1}/{len(datasets)}] ({DATASET_NAMES[i]}).')
            eval_res = trainer.evaluate(datasets[i]['test'])
            print('Evaluation result:\n', eval_res)

        # Save the best model in RUNS_DIR
        final_run_path = os.path.join(RUNS_DIR, dataset_name)
        if not os.path.exists(final_run_path): 
            os.makedirs(final_run_path)
        print(f'\nSaving the best model in {final_run_path}.')
        save_hf_model(model, final_run_path)

def eval_classifier(): 

    # Load model from checkpoint
    checkpoint_path = os.path.join(RUNS_DIR, 'checkpoint-344')
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # Load the datasets
    datasets = [load_dataset(dataset_name, tokenizer) for dataset_name in DATASET_NAMES]

    # Evaluate the model on test sets and save results
    TRAINING_ARGS.run_name = 'evaluation'
    TRAINING_ARGS.output_dir = EVAL_DIR
    TRAINING_ARGS.report_to = 'none'

    for i in range(len(datasets)):
        print(f'\nEvaluating on dataset [{i+1}/{len(datasets)}] ({DATASET_NAMES[i]}).')
        trainer = WeightedTrainer(
            model=model,
            args=TRAINING_ARGS,
            train_dataset=datasets[i]['train'],
            eval_dataset=datasets[i]['val'],
            compute_metrics=compute_metrics,       
        )
        eval_res = trainer.evaluate(datasets[i]['test'])
        print('Evaluation result:\n', eval_res)


# ----------------- Trainer with Weighted Cross-Entropy Loss ----------------- #

class WeightedTrainer(Trainer):
    ''' CustomTrainer with weighted cross entropy loss'''
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass through the model
        labels = inputs.get("labels").squeeze(1)
        input_ids = inputs.get("input_ids").squeeze(1).to(torch.int64)
        attention_mask = inputs.get("attention_mask").squeeze(1).to(torch.int64)
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.get("logits").to(torch.float64)

        # Compute weighted cross entropy loss
        incorrect_weight = labels.shape[0] / torch.sum(labels == 0)
        correct_weight = labels.shape[0] / torch.sum(labels == 1)
        class_weights = torch.tensor([incorrect_weight, correct_weight],
                                      dtype=torch.float64, 
                                      device=labels.device)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')    
        label_matrix = torch.stack([1 - labels, labels], dim=1).to(torch.float64)
        weightedCE = torch.mean(ce_loss(logits, label_matrix))              
        return (weightedCE, outputs) if return_outputs else weightedCE
    
# ----------------- Model evaluation ----------------- #

def compute_metrics(eval_preds):
    ''' Compute evaluation metrics of predictions compared to labels. 
        
        Accuracy: proportion of correct predictions
        Recall: proportion of true positives correctly identified (TPR, sensitivity)
        Precision: proportion of predicted positives that are true positives
        TNR: true negative rate (specificity)
        ROC AUC: area under the ROC curve
        F1: harmonic mean of precision and recall 
    '''

    # Compute predicted labels
    logits, labels = eval_preds
    labels = labels.flatten()                                   
    predictions = np.argmax(logits, axis=-1)  

    # Compute metrics
    TNR = round(np.sum((predictions == 0) & (labels == 0)) / np.sum(labels == 0), 4)
    accuracy = round(accuracy_score(labels, predictions), 4)
    recall = round(recall_score(labels, predictions, zero_division=0.), 4)                # Recall = TP / (TP + FN)
    precision = round(precision_score(labels, predictions, zero_division=0.), 4)          # Precision = TP / (TP + FP)
    roc_auc = round(roc_auc_score(labels, logits[:, 1]), 4)
    f1 = round(f1_score(labels, predictions, zero_division=0.), 4)                        # F1 = 2 * (precision * recall) / (precision + recall)
    metrics = {'accuracy': accuracy, 'recall': recall, 'precision': precision, 'roc_auc': roc_auc, 'f1': f1, 'TNR': TNR}
    return metrics


if __name__ == '__main__': 

    if not os.path.exists(RUNS_DIR): 
        os.makedirs(RUNS_DIR)
    if not os.path.exists(LOGS_DIR): 
        os.makedirs(LOGS_DIR)
    if not os.path.exists(EVAL_DIR):
        os.makedirs(EVAL_DIR)

    eval_classifier()
