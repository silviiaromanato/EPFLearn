
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch

class RewardDataset(Dataset):
    ''' 
    Custom dataset for question-answer pairs for reward model training. 
    Loads one batch per question, where each batch contains all answers to the same question.
    Each QA pair is concatenated into a single chat, which is then tokenized.
    Each QA pair is labelled {1 : correct answer, 0 : incorrect answer}.
    '''
    def __init__(self, df, tokenizer):
        '''
        Inputs: 
            - df: pandas dataframe containing the question-answer pairs
                with columns: [QuestionId, AnswerId, chat, label]
            - tokenizer: RoBERTa tokenizer
        '''
        self.df = df
        self.tokenizer = tokenizer
        self.label_map = {'positive' : 1, 'negative' : 0}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        ''' 
        Get the tokenized input IDs and attention masks for all answers to the (index)-th question.
        One batch corresponds to all answers to the same question.
        Returns a batch of input_ids and attention_masks for all answers to the same question.
        '''

        # Batch different questions together: 
        chat, label = self.df.iloc[index][['chat', 'label']].values
        label = self.label_map[label]
        tokenized = tokenize(chat, self.tokenizer)
        item = {
            'input_ids': tokenized['input_ids'],       # (1 * D)
            'attention_mask': tokenized['attention_mask'], # (1 * D)
            'labels': torch.tensor([label])            # (1 * 1)
        }
        return item

def tokenize(chat, tokenizer, max_length=512):
    '''
    Tokenize a single chat, truncate to max_length tokens, pad to max_length tokens.  
    Inputs:
        - chat: string of concatenated question and answer
    Outputs:
        - input_ids: torch tensor of the tokenized chat
        - attention_masks: torch tensor of the attention masks
    '''

    # Tokenize the chat, truncate to 512 tokens, pad to 512 tokens
    tokenized = tokenizer.encode_plus(
        chat,                           # Chat to encode    
        add_special_tokens=True,        # Add '[CLS]' and '[SEP]'
        padding='max_length',
        truncation=True, 
        max_length=max_length,          # Truncate/pad to max_length tokens
        return_attention_mask=True,     # Return attention masks
        return_tensors='pt'             # Return PyTorch tensors
    )
    return tokenized


def create_dataset(df, tokenizer, seed=1): 
    ''' Split the data into train, validation and test sets (70/10/20 split). '''
    indexes = df.index.values
    temp_ids, test_ids = train_test_split(indexes, test_size=0.2, random_state=seed)
    train_ids, val_ids = train_test_split(temp_ids, test_size=0.125, random_state=seed)

    dataset = {
        'train': RewardDataset(df.iloc[train_ids], tokenizer),
        'val': RewardDataset(df.iloc[val_ids], tokenizer),
        'test': RewardDataset(df.iloc[test_ids], tokenizer)
    }
    return dataset