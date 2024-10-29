import torch
import os
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoint')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    RobertaConfig,
    AutoModel,
    AutoModelForSequenceClassification
)

# ========================================================
# Below is an example of how you can implement your own 
# custom HuggingFace model and use it in the evaluation.
# 
# This is where you should implement your model and the
# get_rewards() function. 
# 
# If you want to extend an existing
# HuggingFace model, you can also add additional layers or 
# heads in this class.
# ========================================================


class ClassifierRewardModelConfig(RobertaConfig):
    """
    This is an example config class for a custom HuggingFace model.
    
    - Inherit from the HuggingFace config class that is most similar to your base model.

    - You should specify the model_type as your model's class name.
    """
    model_type = "ClassifierRewardModel"
    base_model_name = 'roberta-base'
    checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')

    # If you have additional parameters to the model class,
    # you can add them inside the config class as well.
    # For example, with "def __init__(self, config, reward_dim=1):",
    # you can specify "reward_dim = 1" here in the config class.
    # Then, you can acess the reward_dim parameter in the model class 
    # by calling "self.config.reward_dim".
    
class ClassifierRewardModel(PreTrainedModel):
    """
    This is an example regression model built on top of the OpenAssistant Dberta model.
    You are not expected to follow this example, but you can use it as a reference point.
    You should have the freedom to construct your model however you want.
    
    !IMPORTANT!: You need to implement the get_rewards() function, which takes in a list of demonstrations
                and returns a list of rewards. See more details in the fuction below.
    !IMPORTANT!: You should implement your model class such that 
                it can be saved as a HuggingFace PreTrainedModel.
                This menas you also need to implement the CustomHFConfig class 
                and specify the model_type as your model's class name.
    """

    # Set the config class to your custom config class
    config_class = ClassifierRewardModelConfig

    def __init__(self, config):
        super().__init__(config)
        hf_pretrained_model_name = 'roberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained(hf_pretrained_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR, num_labels=2)
        self.config = config


    def forward(self, encoded):
        ''' Compute rewards for a single demonstration. '''
        input_ids = encoded.get("input_ids").squeeze(1).to(torch.int64)
        attention_mask = encoded.get("attention_mask").squeeze(1).to(torch.int64)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.get("logits")              # Get the logits from the model output
        predictions = torch.argmax(logits, dim=-1)  # Predicted label
        rewards = F.softmax(logits, dim=-1)[:, 1]   # Probability of positive label
        return rewards, predictions


    def get_rewards(self, demonstrations):
        """
        Get the rewards for the demonstrations.
        Args:
            demonstrations: list of dicts in the format of
            {'chat': str, 'label': int}
        Return:
            rewards: list of dicts in the format of
            {'chat': str, 'label': int, 'reward': float, 'prediction': int}
        """
        # Encode the batch of chats
        batch_chats = [data['chat'] for data in demonstrations]
        encoded = self.tokenizer.batch_encode_plus(
            batch_chats,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors='pt'
        )
        # Move the encoded batch to the GPU
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}

        # Pass the encoded batch through the model
        with torch.no_grad():
            reward_scores, predictions = self.forward(encoded)

        # Clear intermediate variables
        torch.cuda.empty_cache()

        rewards = []
        for idx, data in enumerate(demonstrations):
            prediction = predictions[idx].item()
            pred_label = 'positive' if prediction == 1 else 'negative'
            rewards.append({
                'chat': data['chat'],
                'label': data['label'],
                'reward': reward_scores[idx].item(),
                'prediction': pred_label 
            })
        return rewards