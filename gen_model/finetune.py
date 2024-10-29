''' Fine-tuning a generative language model on specialized content. '''

TEAM_NAME = 'syntax-sorcerers'

import os
import numpy as np
import torch
import evaluate
from transformers import (
    EarlyStoppingCallback, 
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    GenerationConfig
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from load_data import *

# ----------------- Paths ----------------- #

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_DIR = os.path.join(BASE_DIR, 'gen_model')
DATA_DIR = os.path.join(BASE_DIR, 'data')
GEN_DATA_DIR = os.path.join(DATA_DIR, 'gen_model')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')

for dir in [GEN_DIR, DATA_DIR, GEN_DATA_DIR, CHECKPOINTS_DIR]:
    if not os.path.exists(dir):
        print('Creating directory:', dir)
        os.makedirs(dir)

FINAL_CHATBOT_DIR = os.path.join(CHECKPOINTS_DIR, 'finalChatbot')
MID_CHATBOT_DIR = os.path.join(CHECKPOINTS_DIR, 'midChatbot')

# ----------------- Environment variables ----------------- #

os.environ["NO_DEPRECATION_WARNING"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # One visible GPU (non-parallel training)

# ----------------- Training hyperparameters ----------------- #

BASE_MODEL_NAME = 't5-large'

BATCH_SIZE_EVAL = 8
MAX_LENGTH = 128
SEED = 0

# ------------- Custom QA Dataset -------------- #

def load_dataset(df, tokenizer, max_length=256, val_size=0.1, test_size=0.1, seed=0): 
    ''' 
    Split the data into train, validation and test sets (80/10/10 split). 
    This function makes sure the same questions are kept in the same set. 
    '''
    indexes = df['question_id'].values
    temp_ids, test_ids = train_test_split(
        indexes, test_size=test_size, shuffle=True, random_state=seed)
    train_ids, val_ids = train_test_split(
        temp_ids, test_size=val_size/(1-test_size), shuffle=True, random_state=seed)

    train_set = df[df['question_id'].isin(train_ids)]
    val_set = df[df['question_id'].isin(val_ids)]
    test_set = df[df['question_id'].isin(test_ids)]

    dataset = {
        'train': QADataset(train_set, tokenizer, max_length=max_length),
        'val': QADataset(val_set, tokenizer, max_length=max_length),
        'test': QADataset(test_set, tokenizer, max_length=max_length)
    }
    return dataset

class QADataset(Dataset):
    ''' Custom Dataset class for fine-tuning. '''
    
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        ''' 
        Returns a sample of tokenized question-answer pair. 
        Tokenizes --> [EOS] Question [SEP] Answer [SEP]
        '''
        q, a = self.df.iloc[index][['question', 'answer']].values

        question = self.tokenizer(
            q,
            max_length=self.max_length,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True
        )
        answer = self.tokenizer(
            a,
            max_length=self.max_length,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            return_tensors='pt', 
        )

        return {
            'input_ids': question['input_ids'].flatten(),
            'attention_mask': question['attention_mask'].flatten(),
            'labels': answer['input_ids'].flatten() 
        }
    

# ----------------- Helper functions for training ----------------- #

def preprocess_logits_for_metrics(logits, labels):
    ''' Preprocess logits for computing metrics. '''
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)

def create_compute_metrics(tokenizer):
    ''' Returns a function that computes perplexity, BLEU and ROUGE scores. '''

    def compute_metrics(pred):
        ''' Compute perplexity, BLEU and ROUGE scores. '''
        nonlocal tokenizer

        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        # Replace -100 in the labels as we can't decode them.
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        references = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        # Compute average length of generated texts
        avg_length = np.mean([len(t.split()) for t in predictions])
            
        # Compute BLEU score
        bleu_score = evaluate.load('bleu')
        results_bleu = bleu_score.compute(
            predictions=predictions, references=references)

        # Compute ROUGE scores
        rouge_score = evaluate.load('rouge')
        results_rouge = rouge_score.compute(
            predictions=predictions, references=references)

        scores = {
            'bleu_score': results_bleu['bleu']*100,
            'rouge1_score': results_rouge['rouge1']*100,
            'rouge2_score': results_rouge['rouge2']*100,
            'rougeL_score': results_rouge['rougeL']*100, 
            'avg_length' : avg_length
        }

        return scores

    return compute_metrics


def create_trainer(
        model, 
        tokenizer,
        dataset,
        dataset_name = 'EPFL',
        learning_rate = 1e-4,
        num_epochs = 10,
        batch_size_train = 8,
        eval_only = False,
        generation_config = None,
    ): 

    # Define training arguments
    run_name = f'Finetune_{dataset_name}_{str(learning_rate)}'
    output_dir = os.path.join(CHECKPOINTS_DIR, f'{dataset_name}_{str(learning_rate)}')
    if dataset_name == 'EPFL':
        eval_strategy, eval_steps = 'epoch', None
        save_strategy, save_steps = 'epoch', None
    else:
        eval_strategy, eval_steps = 'steps', 0.25
        save_strategy, save_steps = 'steps', 0.5

    training_args = Seq2SeqTrainingArguments(
        # General arguments
        run_name=run_name,                          # Wandb run name      
        output_dir=output_dir,                      # Directory where checkpoints and logs will be saved
        overwrite_output_dir=True,
        evaluation_strategy=eval_strategy,          
        save_strategy=save_strategy,                
        save_steps=save_steps,                                              
        eval_steps=eval_steps,               
        logging_strategy='steps', 
        logging_steps=250,                          # Log metrics every 100 steps
        load_best_model_at_end=True,                # Load the best model after training
        metric_for_best_model="eval_loss",          # Metric to choose best model and early stopping
        greater_is_better=False,                    # Best model is the one with the lowest loss
        report_to='wandb',                          # Use Weights & Biases to track metrics
        per_device_eval_batch_size=BATCH_SIZE_EVAL, # Batch size per GPU for evaluation
        save_total_limit=3,                         # Maximum number of checkpoints to save
        generation_config=generation_config,        # Config for text generation

        # Memory optimization
        #fp16=(torch.cuda.is_available()),           # Use mixed precision training (disabled for t5-large)
        gradient_checkpointing=True,                # Use gradient checkpointing to save memory
        gradient_accumulation_steps=4,              # Memory optimization

        # Training hyperparameters
        learning_rate=learning_rate,                # Initial learning rate
        per_device_train_batch_size=batch_size_train,# Batch size per GPU
        num_train_epochs=num_epochs,                # Number of training epochs
        seed=SEED,
    )

    early_stopping = EarlyStoppingCallback(early_stopping_patience=5)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'], 
        eval_dataset=dataset['val'],
        callbacks=[early_stopping],
        compute_metrics=create_compute_metrics(tokenizer), 
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Train model on training set if not in eval_only mode
    if not eval_only:
        train_results = trainer.train()
        print('Training results:\n', train_results)

        # Save best trained model 
        checkpoint_path = os.path.join(GEN_DIR, f'checkpoint_{dataset_name}_{str(learning_rate)}')
        trainer.save_model(checkpoint_path)      
        print('\nSaving checkpoint to {}.'.format(checkpoint_path))

    return trainer


def finetune(
        checkpoint : str = BASE_MODEL_NAME, 
        eval_only : bool = False,
        seed : int = SEED, 
        ):
    ''' 
    Fine-tune generative model on EPFL dataset. 
    If checkpoints are already present, load them, skip training and evaluate.
    Args: 
        seed : int, random seed for reproducibility
        eval_only : bool, if True, skip training and evaluate model on test set
        checkpoint : str, name of the initial checkpoint to load
    '''

    # Set manual seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load model, tokenizer, generation configurations
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, model_max_length=MAX_LENGTH)
    generation_config = GenerationConfig.from_pretrained(BASE_MODEL_NAME)       # REPLACE BY CHECKPOINT
    tokenizer.pad_token = tokenizer.eos_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == 'cuda':
        model.to(device)
    print(f'Moving model to device: {device}.')

    # Load datasets
    stack_df = load_stack_data()
    stack_df = stack_df.sample(frac=0.5, random_state=seed)   # Use only 50% of StackOverflow dataset
    stack_dataset = load_dataset(stack_df, tokenizer, max_length=MAX_LENGTH, seed=seed)
    EPFL_df = load_EPFL_data(english_only=False)
    EPFL_dataset = load_dataset(EPFL_df, tokenizer, max_length=MAX_LENGTH, seed=seed)
    eval_datasets = {
        'StackExchange': stack_dataset['test'],
        'EPFL': EPFL_dataset['test']
    }

    # Evaluating or Training on the StackExchange dataset
    stack_trainer = create_trainer(
        model,
        tokenizer, 
        stack_dataset, 
        dataset_name='StackExchange', 
        learning_rate=2e-4, 
        num_epochs=2, 
        batch_size_train=8,
        generation_config=generation_config,
        eval_only= os.path.exists(MID_CHATBOT_DIR)
        )
    if not eval_only:
        stack_results = stack_trainer.train()
        print('StackOverflow training results:\n', stack_results)
    
    # Evaluate trained model on test set
    for dataset_name, eval_dataset in eval_datasets.items():
        print(f'\nEvaluating StackOverflow model on {dataset_name} test set.')
        eval_results = stack_trainer.evaluate(eval_dataset)
        for key, value in eval_results.items():
            print(f'{key}: {value}')

    # Training and Evaluating on EPFL dataset
    epfl_trainer = create_trainer(
        model, 
        tokenizer, 
        EPFL_dataset, 
        dataset_name='EPFL', 
        learning_rate=5e-5, 
        num_epochs=10, 
        batch_size_train=8,
        eval_only=os.path.exists(FINAL_CHATBOT_DIR), 
        generation_config=generation_config,
        )
    if not eval_only:
        epfl_results = epfl_trainer.train()
        print('EPFL training results:\n', epfl_results)

    # Evaluate trained model on test set
    for dataset_name, eval_dataset in eval_datasets.items():
        print(f'\nEvaluating EPFL trained model on {dataset_name} test set.')
        eval_results = epfl_trainer.evaluate(eval_dataset)
        for key, value in eval_results.items():
            print(f'{key}: {value}')


if __name__ == '__main__':
    finetune(eval_only=True, checkpoint = 'google/t5-large-ssm-nq')

