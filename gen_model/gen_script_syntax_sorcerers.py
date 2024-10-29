'''
This script allows anyone to easily access your generative model to produce answers for the testing questions. 
The answers from this script should match exactly to your submitted answers. It’s very important for 
this script to be reproducible so we can grade you, therefore please also prepare a requirements.txt 
file specifying the package versions, and a python.txt file to specify the python version. 
Make sure that your script is reproducible by testing from an environment from scratch.
'''

import os
import pandas as pd
import torch

from chatbot import *

TEAM_NAME = 'syntax-sorcerers'

os.environ["NO_DEPRECATION_WARNING"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
FINAL_CHATBOT_DIR = os.path.join(CHECKPOINTS_DIR, 'finalChatbot')
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROMPTS_PATH = os.path.join(DATA_DIR, 'gen_model', 'prompts.json')
ANSWERS_PATH = os.path.join(DATA_DIR, 'gen_model', f'answers_{TEAM_NAME}.json')

# ---------------- Generative model (for final model) ----------------- #

def generate_answers(
        prompts_path = PROMPTS_PATH,    
        answers_path = ANSWERS_PATH, 
        checkpoint = FINAL_CHATBOT_DIR,   
        seed = 0
        ): 
    ''' 
    Reads the prompts file, generates answers using the 
    fine-tuned chatbot and saves model answers.
        prompts_path: path to prompts.json file
        answers_path: path to save model answers file
        checkpoint: path to fine-tuned chatbot checkpoint
    '''

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize chatbot
    chatbot = Chatbot(checkpoint)

    # Load QA prompts
    if not os.path.exists(prompts_path):
        raise Exception(f'Prompts file not found at {prompts_path}.')
    with open(prompts_path, 'r') as f:
        prompts = pd.read_json(f)

    # Format question and answers
    prompts = prompts.replace({np.nan: None})
    prompts['question'] = prompts.apply(lambda x: Q_from_solutions(x['question'], x['choices']), axis=1)
    prompts['answer'] = prompts.apply(lambda x: A_from_solutions(x['answer'], x['explanation']), axis=1)
    
    responses = []
    for i, (question, answer) in enumerate(zip(prompts['question'], prompts['answer'])):
        print('\n\n' + '-' * 40, f'Prompt [{i}]', '-' * 40, '\n', question, '\n')
        print("-" * 40 + " Solution " + "-" * 40 + '\n', answer, '\n')
        print("-" * 40 + "Chatbot's generated answer" + "-" * 40 + '\n')
        response = chatbot.ask(question)
        print(response, '\n')
        responses.append(response)

    # Save model answers to answers_path
    print(f'Saving generated answers to {answers_path}.')
    answers = pd.DataFrame()
    answers['guid'] = prompts['guid']
    answers['model_answer'] = responses
    answers.to_json(answers_path, orient='records', indent=4)


if __name__ == '__main__':
    generate_answers()