import argparse
import json
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification
)

# Here we load your CustomRewardModelConfig and CustomRewardModel classes, 
# so we have the implementation of your get_rewards function
# and load the weights from your saved model.

from model import ClassifierRewardModelConfig, ClassifierRewardModel

def load_json(filename):
    """Load json file"""
    with open(filename, 'r') as read_file:
        data = json.load(read_file)
    return data


def save_dictlist_to_json(mydictlist, filename):
    """Save a list of dictionaries to json file"""
    f = open(filename, 'w', encoding='utf-8')
    json.dump(mydictlist, f, ensure_ascii=False, indent=4) 
    f.close()


class TestDataset(Dataset):
    """Simple dataset module for testing the reward model"""
    def __init__(self, test_ds):
        self.ds = test_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ix):
        return self.ds[ix]


class Reward(torch.nn.Module):
    """
    Wrapper class for the reward model, 
    which handles loading the model and tokenizers, 
    and the forward pass for final predictions
    """
    def __init__(self, model_path):
        super().__init__()

        # Load student-defined reward model and its associated config
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, config=self.config)

    def check_reward_type(self, rewards):
        return isinstance(rewards, list) and all(isinstance(r, dict) for r in rewards)

    def forward(self, demonstrations):
        """
        Get the rewards for the demonstrations.
        Args:
            demonstrations: list of dicts in the format of
            {'chat': str, 'label': int}
        Return:
            rewards: list of dicts in the format of
            {'chat': str, 'label': int, 'reward': float, 'prediction': int}
        """
        # ===== Get the rewards from student's reward model =====
        rewards = self.model.get_rewards(demonstrations)

        # ===== Check the reward format =====
        assert self.check_reward_type(rewards), "The rewards must be a list of dicts"
        assert len(rewards) == len(demonstrations), "The number of rewards must match the number of demonstration pairs"
        return rewards
    

class Evaluator:
    def __init__(self, model_path, ds_test):
        # Load the model and dataset
        self.load_model(model_path)
        self.ds_test = ds_test
        self.dataset = TestDataset(ds_test)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=16, 
            shuffle=False,
            collate_fn=lambda x: x)

    def load_model(self, model_path):
        """Load the reward model from the specified path"""
        self.model = Reward(model_path)

        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
    
    def evaluate(self):
        """Evaluate the model on the test dataset"""
        rewards = []
        for batch in tqdm(self.dataloader): 
           rewards.extend(self.model(batch))

        # ===== Check the rewards by doing pair-wise ranking =====
        num_correct = sum(reward['label'] == reward['prediction'] for reward in rewards)
        acc = num_correct / len(self.ds_test)
        print(f"Evaluation Complete. Accuracy: {acc}.")
        return rewards

def save_hf_model(hf_model, model_path):
    """Save the model and tokenizer to the specified path"""
    hf_model.save_pretrained(model_path)
    hf_model.config.save_pretrained(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/reward-model",
        help="Path to the model")
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="m2_reward_dataset_example.json",
        help="Path to the test dataset")
    args = parser.parse_args()

    hf_pretrained_model_name = "roberta-base"

    # Save model and model config to model_path directory
    model_config = ClassifierRewardModelConfig.from_pretrained(hf_pretrained_model_name)
    model_config.problem_type = 'single_label_classification'
    model = ClassifierRewardModel(model_config)
    save_hf_model(model, args.model_path)
    
    # Load the dataset
    reward_dataset = load_json(args.data_path)

    # NOTE: Example of how we will load your model
    # Here we need to register the custom reward model and its config
    # to the AutoModel and AutoConfig classes, so that we can load
    # the model using the AutoModel and AutoConfig classes
    AutoConfig.register('ClassifierRewardModel', ClassifierRewardModelConfig)
    AutoModel.register(ClassifierRewardModelConfig, ClassifierRewardModel)

    # Load config and model from model_path directory
    evaluator = Evaluator(args.model_path, reward_dataset)

    # Evaluate the model on the test dataset
    rewards_json = evaluator.evaluate()

    # Save the rewards to a json file
    save_dictlist_to_json(rewards_json, 'm2_reward_dataset_syntax-sorcerers_labelled.json')