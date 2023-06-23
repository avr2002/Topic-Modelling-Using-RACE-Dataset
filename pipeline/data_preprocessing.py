import os
import re  
import yaml
# import nltk
import logging
import contractions
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tqdm import tqdm
tqdm.pandas(desc="Progress Bar!")

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


CONFIG_PATH = "./config/config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    data_config = config['data_config']



def train_test_split(df:pd.DataFrame)->tuple:
    """
    Performing 90-10 train-test split.
    """
    # df = get_and_read_data()
    train_documents = df[:int(0.9*len(df))]
    test_documents = df[int(0.9*len(df)):]

    return train_documents, test_documents


stop_word = stopwords.words('english')
le = WordNetLemmatizer()

def text_preprocessing(text:str) -> str:
    """
    Returns cleaned text in string and list format.
        (cleaned_text:str, tokens:list)
    """
    try:
        if pd.isnull(text):
            return text
        
        # lower-case everything
        text = text.lower()
        
        # expand all the short-form words
        text = contractions.fix(text)
        
        # remove any special chars
        text = re.sub(r'http\S+|www\S+|https\S+', '', text) # Remove URLs
        text = re.sub(r'\S+@\S+', '', text) # Remove emails
        text = re.sub(r'\b\d{1,3}[-./]?\d{1,3}[-./]?\d{1,4}\b', '', text) # Remove phone numbers
        text = re.sub(r'[^a-zA-Z]', ' ', text) # Remove other non-alphanumeric characters 
        
        # tokenization
        word_tokens = word_tokenize(text)
        
        # remove stop-word and lemmatize
        tokens = [le.lemmatize(w) for w in word_tokens if w not in stop_word and len(w)>1] #  and len(w)>3
        
        cleaned_text = " ".join(tokens)
        return cleaned_text
    except Exception as e:
        logger.error(f"\nError occured while cleaning the text document:\n{e}\n", exc_info=True)



def clean_text_data(df:pd.DataFrame):
    try:
        if 'document' in df.columns:
            df['clean_document'] = df['document'].progress_apply(lambda text: text_preprocessing(str(text)))
            #df['clean_tokens'] = df['document'].progress_apply(lambda text: text_preprocessing(str(text))[1])

            return df
        else:
            raise Exception(f"\nRequired Feature: ['document'] is not available in the given DataFrame: [{df.columns}]")
    except Exception as e:
        logger.error(f"\nError Occured while Cleaning the text DataFrame:\n{e}\n", exc_info=True)


def get_cleaned_train_test_data(df:pd.DataFrame):
    """
    Takes in the raw data in csv format and splits it into train-test set
    And returns cleaned (train, test) set
    """
    try:
        ingested_data_path = data_config['ingested_data']
        os.makedirs(ingested_data_path, exist_ok=True)

        test_file_name = data_config['cleaned_test_data_name']
        train_file_name = data_config['cleaned_train_data_name']

        test_file_path = os.path.join(ingested_data_path, test_file_name)
        train_file_path = os.path.join(ingested_data_path, train_file_name)

        if os.path.isfile(test_file_path) and os.path.isfile(train_file_path):
            cleaned_train_data = pd.read_csv(train_file_path)
            cleaned_test_data = pd.read_csv(test_file_path)

            print(f"Cleaned Train-Test Set is alread Present in [{ingested_data_path}]\nLoading the Train-Test Set...\n")

            return cleaned_train_data, cleaned_test_data
        else:
            print("Splitting the Text Data into Train-Test Set...")
            train_documents, test_documents = train_test_split(df)

            print("Cleaning the Train-Test Documents...")
            cleaned_train_data = clean_text_data(train_documents)
            cleaned_test_data = clean_text_data(test_documents)

            cleaned_train_data.to_csv(train_file_path, index=False)
            cleaned_test_data.to_csv(test_file_path, index=False)

            print("Text Preprocessing Done!\n")
            return cleaned_train_data, cleaned_test_data
    except Exception as e:
        logger.error(f"\nError Occured while executing [get_cleaned_train_test_data()] function:\n{e}\n", exc_info=True)