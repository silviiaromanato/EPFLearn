''' Data pre-processing. '''

TEAM_NAME = 'syntax-sorcerers'

import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from langdetect import detect
import LaTexAccents as TeX

# Create directories for saving models and runs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
STACK_DIR = os.path.join(DATA_DIR, 'stackexchange')
GEN_DATA_DIR = os.path.join(DATA_DIR, 'gen_model')

# Generative model datasets paths
STACK_PATH = os.path.join(GEN_DATA_DIR, f'gen_dataset_{TEAM_NAME}_StackOverflow.json')
EPFL_PATH = os.path.join(GEN_DATA_DIR, f'gen_dataset_{TEAM_NAME}_EPFL.json')
QA_PATH = os.path.join(DATA_DIR, 'gen_model', f'gen_dataset_{TEAM_NAME}.json')

# ------------- Fixed data filtering parameters -------------- #

TOPICS = ['cs', 'cstheory', 'ds', 'physics', 'chem', 'eng', 'mech', 'softeng', 'quant', 'maths'] 

# Minimum number of upvotes for an answer to be included in the StackOverflow dataset
MIN_UPVOTES = zip(TOPICS, [3, 3, 3, 5, 5, 3, 3, 5, 3, 12])

# Minimum confidence level to be included in the EPFL dataset
CONFIDENCE_THRESHOLD = 4

# --------------- StackOverflow dataset --------------- # 

def load_stack_data(min_upvotes = MIN_UPVOTES, min_length = None):
    ''' Loads the StackOverflow data. '''

    # If already pre-processed, read from json
    if os.path.exists(STACK_PATH):
        print("Loading pre-processed StackOverflow data.")
        stack_data = pd.read_json(STACK_PATH)
        return stack_data
    
    # Otherwise, load from xml files
    stack_data = pd.DataFrame()
    print("Loading StackExchange data from xml files...")
    if not os.path.exists(STACK_DIR):
        print("StackExchange data not found. Please download the data and place it in the data/stackexchange directory.")

    # Load the data for each topic
    for topic in TOPICS: 
        min_upvotes_topic = min_upvotes[topic]
        topic_data = load_topic_data(topic, min_upvotes=min_upvotes_topic, min_length=min_length)
        print('\tLoading {} QA samples from topic {}.'.format(len(topic_data), topic))
        stack_data = pd.concat([stack_data, topic_data])
    
    # Reset entry index for the whole dataset
    stack_data['entry_id'] = stack_data.index

    # Save the StackOverflow data in json format
    stack_path = os.path.join(GEN_DATA_DIR, f'gen_dataset_{TEAM_NAME}_StackOverflow.json') 
    stack_data.to_json(stack_path, orient='records', indent=4)
    
    return stack_data

def load_topic_data(topic, min_upvotes=MIN_UPVOTES, min_length=None):
    ''' Preprocesses the StackOverflow data for a given topic. '''

    # Remove rows with empty body, keep useful columns
    posts = parse_xml(topic)
    posts = posts[['Body', 'PostTypeId', 'Title', 'Id', 'AcceptedAnswerId', 'AnswerCount', 'Score']]
    posts = posts[posts['Body'].notna()]

    # Split the data into questions and answers
    questions = posts[posts['PostTypeId'] == '1']
    answers = posts[posts['PostTypeId'] == '2']

    # Keep only questions with accepted answer
    questions = questions[questions['AcceptedAnswerId'].notna()]
    questions = questions.rename(columns={'Id': 'QuestionId'})
    answers = answers.rename(
        columns={'Score': 'AnswerScore', 'Body': 'AcceptedAnswerBody', 'Id' : 'AnswerId'}
        )[['AnswerId', 'AcceptedAnswerBody', 'AnswerScore']]
    
    # Filter questions with accepted answer
    QA = questions.merge(answers, left_on='AcceptedAnswerId', right_on='AnswerId', how='left')

    # Keep only questions with exactly 1 accepted answer
    QA = QA[QA.groupby('QuestionId')['QuestionId'].transform('size') == 1]

    # Remove <p> and </p> tags from accepted answers body
    QA['Body'] = QA['Body'].str.replace('<p>', '').str.replace('</p>', '')
    QA['AcceptedAnswerBody'] = QA['AcceptedAnswerBody'].str.replace('<p>', '').str.replace('</p>', '')

    # Format questions and answers
    QA['question'] = QA['Title'] + ' ' + QA['Body']
    QA.rename(columns={'AcceptedAnswerBody': 'answer', 'QuestionId' : 'question_id'}, inplace=True)
    #QA['topic'] = topic

    # Keep only answers with >= (min_upvotes) upvotes
    QA['AnswerScore'] = QA['AnswerScore'].astype(int)
    QA = QA[QA['AnswerScore'] >= min_upvotes]

    # Keep only answers with >= (min_length) characters
    if min_length is not None: 
        print('\tSelecting answers with at least {} characters.'.format(min_length))
        QA = QA[QA['question'].str.len() >= min_length]

    # Concatenate topic and question id to create unique id
    QA['question_id'] = topic + '_' + QA['question_id'].astype(str)
    QA = QA[['question_id', 'question', 'answer']]

    return QA

def parse_xml(topic): 
    ''' Parse xml file and store in dataframe. '''
    path = os.path.join(STACK_DIR, topic, 'Posts.xml')
    tree = ET.parse(path)
    root = tree.getroot()
    data = []
    for row in root.findall('row'):
        attributes = row.attrib
        data.append(attributes)
    return pd.DataFrame(data)


# --------------- EPFL interactions dataset --------------- #

def Q_from_solutions(question, choices=None): 
    """ Create a chatbot interaction from a question, answer, and optionally choices and explanation. """
    q = f"{str(question)}"
    if choices is not None: 
        q += f"\nSelect the correct answer(s):\n"
        q +=  " " + " ".join([f"{i+1}. {choice}\n" for i, choice in enumerate(choices)])
    return q

def A_from_solutions(answer, explanation=None):
    """ Create a chatbot answer from an answer and optionally an explanation. """
    a = str(answer)
    if explanation is not None:
        a += f"\n{explanation}"
    return a

def Q_from_interactions(interactions): 
    """ Create a chatbot interaction from an interaction. """
    q = ''
    # Add the first interaction whose key is user
    for interaction in interactions:
        if interaction['role'] == 'user': 
            q += f"{interaction['content']}\n\n"
            break
    return q

def A_from_interactions(interactions): 
    """ Create a chatbot interaction from an interaction. """
    # Add the first interaction whose key is assistant
    q = ''
    for interaction in interactions:
        if interaction['role'] == 'assistant': 
            q += f"{interaction['content']}\n\n"
            break
    return q

def detect_language(string):
    try:
        return detect(string)
    except: 
        return None
    

def load_EPFL_data(confidence_threshold=CONFIDENCE_THRESHOLD, english_only=True): 
    ''' Loads the EPFL data. '''
    
    # If already pre-processed, read from json
    if os.path.exists(EPFL_PATH):
        print("Loading pre-processed EPFL data.")
        epfl_data = pd.read_json(EPFL_PATH)
        return epfl_data

    # Otherwise, load from json files
    print(f"Creating the {EPFL_PATH} dataset ... ")
    print("Loading EPFL data from json files...")
    # Load interactions and solutions
    solutions_path = os.path.join(DATA_DIR, 'manual_chatgpt', 'solutions_v1.json')
    interactions_path = os.path.join(DATA_DIR, 'manual_chatgpt', 'interactions_v1.json')
    solutions = pd.read_json(solutions_path)
    interactions = pd.read_json(interactions_path)

    # ------ Process solutions ------ #
    solutions = solutions.replace({np.nan: None})
    solutions['question'] = solutions.apply(lambda x: Q_from_solutions(x['question'], x['choices']), axis=1)
    solutions['answer'] = solutions.apply(lambda x: A_from_solutions(x['answer'], x['explanation']), axis=1)
    solutions = solutions[['sol_id', 'question', 'answer']]

    # ------ Process interactions ------ #
    # Remove interactions with no content or with confidence < 4 or few-shot interactions
    print('\tRemoving interactions with confidence < {}.'.format(confidence_threshold))
    interactions = interactions[interactions['interaction'].isna() == False]
    interactions = interactions[interactions['confidence'] >= confidence_threshold]
    def count_interactions(interactions):
        return sum([1 for d in interactions if d['role'] == 'user'])
    interactions = interactions[interactions['interaction'].apply(count_interactions) == 1]

    # Format question and answer
    interactions['question'] = interactions[['interaction']].apply(lambda x: Q_from_interactions(x['interaction']), axis=1)
    interactions['answer'] = interactions[['interaction']].apply(lambda x: A_from_interactions(x['interaction']), axis=1)

    if english_only: # Only keep English interactions
        print('\tOnly keeping English interactions.')
        interactions['language'] = interactions['question'].apply(detect_language)
        interactions = interactions[interactions['language'] == 'en']
    interactions = interactions[['sol_id', 'question', 'answer']]

    # ------ Merge solutions and interactions ------ #
    EPFL_data = pd.concat([solutions, interactions])
    EPFL_data['answer_id'] = EPFL_data.index
    EPFL_data.columns = ['question_id', 'question', 'answer', 'entry_id'] 

    if not english_only:
        # -------- Add the accents to the text -------- #
        print('\tAdding accents to the text.')
        converter = TeX.AccentConverter()
        EPFL_data['question'] = EPFL_data['question'].apply(lambda x: converter.decode_Tex_Accents(x, utf8_or_ascii=1))
        EPFL_data['answer'] = EPFL_data['answer'].apply(lambda x: converter.decode_Tex_Accents(x, utf8_or_ascii=1))

    # Save the EPFL data in json format
    EPFL_path = os.path.join(DATA_DIR, 'gen_model', f'gen_dataset_{TEAM_NAME}_EPFL.json') 
    EPFL_data.to_json(EPFL_path, orient='records', indent=4)

    return EPFL_data

if __name__ == '__main__':

    load_EPFL_data(english_only=False)


    

