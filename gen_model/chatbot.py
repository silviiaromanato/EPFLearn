''' Generating answers with fine-tuned chatbot. '''

import os
import torch
from transformers import AutoTokenizer, GenerationConfig, AutoModelForSeq2SeqLM

from finetune import *

TEAM_NAME = 'syntax-sorcerers'

os.environ["NO_DEPRECATION_WARNING"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BASE_MODEL_NAME = 't5-large'

BASELINE_MODEL_NAME = "google/t5-large-ssm-nq"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
DATA_DIR = os.path.join(BASE_DIR, 'data')
FINAL_CHATBOT_DIR = os.path.join(CHECKPOINTS_DIR, 'finalChatbot')
MID_CHATBOT_DIR = os.path.join(CHECKPOINTS_DIR, 'midChatbot')

# ---------------- Generative model (for final model) ----------------- #


class Chatbot(torch.nn.Module):

    def __init__(self, checkpoint=BASE_MODEL_NAME):
        ''' Initialize the model and tokenizer from provided checkpoint. '''
        super(Chatbot, self).__init__()

        # Load model and tokenizer
        if os.path.exists(checkpoint) or checkpoint==BASE_MODEL_NAME or checkpoint==BASELINE_MODEL_NAME:
            print('Initializing chatbot from checkpoint: ', checkpoint)
        else: 
            print(f"Couldn't find checkpoint: {checkpoint}.\nInitializing chatbot from {BASE_MODEL_NAME}.")
            checkpoint = BASE_MODEL_NAME
        if checkpoint == BASELINE_MODEL_NAME:
            self.generation_config = GenerationConfig.from_pretrained(FINAL_CHATBOT_DIR)
        else: 
            self.generation_config = GenerationConfig.from_pretrained(checkpoint)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME, model_max_length=self.generation_config.max_length)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.prefix = 'Answer the following question: '         # Prefix for generation
        self.suffix = '\nAnswer: '                              # Suffix for generation

    def ask(self, prompt):
        ''' Given a prompt, generate an answer. '''

        # 1. Encode the prompt
        prompt = self.prefix + prompt + self.suffix
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 2. Generate the answer
        response = self.model.generate(
            **encoded, 
            return_dict_in_generate=True, 
            output_scores=True, 
            pad_token_id=self.tokenizer.eos_token_id,
            generation_config=self.generation_config
            ).sequences[0]

        # 3. Decode the generated answer
        response = self.tokenizer.decode(
            response, skip_special_tokens=True)

        return response

def compare_chatbots(): 
    ''' 
    Compare generations from different models: 
    t5-base, mid-chatbot (StackOverflow), final-chatbot (StackOverflow + EPFL)
    For generation hyperparameter tuning. 
    '''

    checkpoints = [MID_CHATBOT_DIR, FINAL_CHATBOT_DIR, BASELINE_MODEL_NAME]
    chatbots = [Chatbot(checkpoint) for checkpoint in checkpoints]

    prompts = [
        #'What is the structure of an atom?', 
        #'What does a for loop do in programming?', 
        #'What is the value of the gravitational constant?',
        #'What is the difference between a cpu and gpu?', 
        #'What is the 3rd root of 27?', 
        #'What is the difference between an acid and a base?', 
        #'Hello how are you?',
        "Which light should be used to activate the PA-GFP?\nSelect the correct answer(s):\n 1. deep-UV\n 2. green\n 3. violet\n 4. blue\n",
        "How does the eigenfrequency of a resonator change when decreasing it'sdimensions?\nSelect the correct answer(s):\n 1. It decreases\n 2. It increases\n 3. Stays the same\n 4. Depends on the resonator\n",
        "When computing PageRank iteratively, the computation ends when:\nSelect the correct answer(s):\n 1. The norm of the difference of rank vectors of two subsequent iterations falls below a predefined threshold\n 2. The difference among the eigenvalues of two subsequent iterations falls below a predefined threshold\n 3. All nodes of the graph have been visited at least once\n 4. The probability of visiting an unseen node falls below a predefined threshold\n",
        "What is the benefit of LDA over LSI?\nSelect the correct answer(s):\n 1. LSI is sensitive to the ordering of the words in a document, whereas LDA is not\n 2. LDA has better theoretical explanation, and its empirical results are in general better than LSI\u2019s\n 3. LSI is based on a model of how documents are generated, whereas LDA is not\n 4. LDA represents semantic dimensions (topics, concepts) as weighted combinations of terms, whereas LSI does not\n",
        "Is Support Vector Machines a supervised or unsupervised method?",
        "What information does a Doppler shift provide?\n\n",
        "Assume that you are part of a team developing a mobile app using Scrum. One of your colleagues suggests that your team should organize daily Scrum meetings to discuss the progress of the tasks and how to implement complex features. He especially wants to discuss the implementation of a feature that will allow users to scan a QR code to get a discount, and would like some input from the team. What are your thoughts on this?",
        "Assume that we have a data matrix $\\mathbf{X}$ of dimension $D \\times N$ as usual. Suppose that its SVD is of the from $\\mathbf{X}=\\mathbf{U S V}^{\\top}$, where $\\mathbf{S}$ is a diagonal matrix with $s_{1}=N$ and $s_{2}=s_{3}=\\cdots=s_{D}=1$. Assume that we want to compress the data from $D$ to 1 dimensions via a linear transform represented by a $1 \\times D$ matrix $\\mathbf{C}$ and reconstruct then via $D \\times 1$ matrix $R$. Let $\\hat{\\mathbf{X}}=\\mathbf{R} \\mathbf{C X}$ be the reconstruction. What is the smallest value we can achieve for $\\|\\mathbf{X}-\\hat{\\mathbf{X}}\\|_{F}^{2}$ ? \n Select one of the following answers:\n (a) $D$ \n(b) $D-1$ \n(c) $N-D$ \n(d) $N-D+1$ \n(e) $N-D-1$ \n(f) $N-1$ \n(g) $N$",
        "What is the difference between genetics and epigenetics? \nSelect one of the following choices: \n 1. Genetic characteristics are heritable, but not epigenetic one \n2. Epigenetic characteristics vary between cell types, but genetic characteristics do not \n3. Differences in epigenetics do not influence gene expression whereas genetic variability does \n4. All of the above.",
        "What is the main difference between PET and SPECT? \nSelect one of the following choices: \n 1. The detection ring used in PET and SPECT are different \n2. PET scan begins with the administration of a radiopharmaceutical \n3. PET scan requires a longer time for data collection \n4. PET scan uses positron-emitting radioisotope instead of single-photon emitters"
        ]
    
    for i, prompt in enumerate(prompts):
        print('\n\n' + '-' * 40)
        print(f'Prompt [{i}]:', prompt)
        print('-' * 40)
        for checkpoint, chatbot in zip(checkpoints, chatbots):
            check_name = checkpoint.split('/')[-1]
            print("-" * 20 + f' {check_name} ' + "-" * 20 + '\n')
            response = chatbot.ask(prompt)
            print(response)
            print("-" * 40 + '\n')


if __name__ == '__main__':
    compare_chatbots()