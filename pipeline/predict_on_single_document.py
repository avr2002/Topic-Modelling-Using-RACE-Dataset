import os
import sys
import yaml
import joblib
import logging
import pandas as pd
import numpy as np
from pipeline.data_preprocessing import text_preprocessing
from pipeline.utils import validate_files


logger = logging.getLogger(__name__)


CONFIG_PATH = "./config/config.yaml"

with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)



def read_test_data()->pd.DataFrame:
    try:
        ingested_data_path = config['data_config']['ingested_data']
        test_file_name = config['data_config']['cleaned_test_data_name']
        test_data_path = os.path.join(ingested_data_path, test_file_name)
        
        data = pd.read_csv(test_data_path)
        return data
    except Exception as e:
        logger.error(f"\nError Occured in read_test_data() in predict_on_single_document.py\n{e}\n", exc_info=True)
        sys.exit()


def vectorize_text_data(cleaned_text:list, vectorizer_name:str): 
    try:
        vectorizers_save_path = config['vectorizers_save_path']
        if vectorizer_name=='count':
            vectorizer_path = os.path.join(vectorizers_save_path, 'count_vectorizer.pkl')
            vectorizer = joblib.load(vectorizer_path)
            return vectorizer.transform(cleaned_text)
        else:
            vectorizer_path = os.path.join(vectorizers_save_path, 'tfidf_vectorizer.pkl')
            vectorizer = joblib.load(vectorizer_path)
            return vectorizer.transform(cleaned_text)
    except Exception as e:
        logger.error(f"\nError Occured in vectorize_text_data() in predict_on_single_document.py\n{e}\n", exc_info=True)
        sys.exit()


def load_model(model_name:str):
    try:
        model_save_path = config['model_save_path']
        model_names = config['model_names']
        model_path = os.path.join(model_save_path, model_names[model_name.lower()])

        model = joblib.load(model_path)
        return model
    except Exception as e:
        logger.error(f"\nError Occured in load_model() in predict_on_single_document.py\n{e}\n", exc_info=True)
        sys.exit()
    

def get_model_topic_keywords(model_name:str, topic_num:int):
    try:
        model_output_save_path = config['model_output_save_path']
        model_output_folder_names = config['model_output_folder_names']
        file_name = f'topic_keywords_{model_name}.csv'
        file_path = os.path.join(model_output_save_path, model_output_folder_names[model_name], file_name)

        topic_keywords_df = pd.read_csv(file_path)
        topic = list(topic_keywords_df.iloc[topic_num, 1:])

        return topic
    except Exception as e:
        logger.error(f"\nError Occured in load_model_topics() in predict_on_single_document.py\n{e}\n", exc_info=True)
        sys.exit()


def predict_using_single_model(model_name:str):
    """
    Predicts Topics on a single text document from test-set with the given model
        model_name: ['lsa', 'lda', 'nmf']
    """
    # user_input_text="Do you wanna enter your own text?: Press 0\nOR\nPredict on random text document?: Press 1\n"
    # user_input = input(user_input_text)
    try:
        model_name = model_name.lower()
        if model_name not in ['lda', 'lsa', 'nmf']:
            sys.exit("Wrong Model Name!\nPlease Enter the Correct Name from ['lda', 'lsa', 'nmf']")

        # Get the text data
        data = read_test_data()

        # Get random text
        random_index = np.random.randint(0, len(data))
        text = data['document'][random_index]

        # Text Cleaning
        cleaned_text = [text_preprocessing(text)]

        # Vectorized Text
        if model_name.lower() == 'lda':
            vectorized_text = vectorize_text_data(cleaned_text, 'count') 
        else:
            vectorized_text = vectorize_text_data(cleaned_text, 'tfidf')
        
        # Load model & predict
        model = load_model(model_name=model_name)
        model_output = model.transform(vectorized_text)
        predicted_topic_num = np.argmax(model_output, axis=1)[0]

        topic = get_model_topic_keywords(model_name, predicted_topic_num)
        
        df = pd.DataFrame({'Text':text, 'Topic':[topic], 'Model Name':model_name.upper()}, index=[0])
        return df
    except Exception as e:
        logger.error(f"\nError Occured in predict() in predict_on_single_document.py\n{e}\n", exc_info=True)
        sys.exit()



def predict_using_all_models()->pd.DataFrame:
    """
    Predicts Topics on a single text document from test-set with the all models
    """
    # user_input_text="Do you wanna enter your own text?: Press 0\nOR\nPredict on random text document?: Press 1\n"
    # user_input = input(user_input_text)
    try:
        lsa_output = predict_using_single_model('lsa')
        lda_output = predict_using_single_model('lda')
        nmf_output = predict_using_single_model('nmf')

        df = pd.DataFrame({'Text': lsa_output.Text.values[0], 'LSA Predicted Topic':lsa_output['Topic'],\
                           'LDA Predicted Topic':lda_output['Topic'], 'NMF Predicted Topic':nmf_output['Topic']})
        return df
    except Exception as e:
        logger.error(f"\nError Occured in predict_with_all_models() in predict_on_single_document.py\n{e}\n", exc_info=True)
        sys.exit()



def predict_topics_on_a_single_document(model_name: str):
    try:
        if validate_files():
            print("\nPredicting on a random text document...\n")

            if model_name.lower()=='all':
                predictions = predict_using_all_models()
                print(f"TEXT:\n{predictions.Text.values[0]}\n")
                print(f"LSA Predicted Topic:\n{predictions['LSA Predicted Topic'].values[0]}\n")
                print(f"LDA Predicted Topic:\n{predictions['LDA Predicted Topic'].values[0]}\n")
                print(f"NMF Predicted Topic:\n{predictions['NMF Predicted Topic'].values[0]}")
                return predictions
            
            predictions = predict_using_single_model(model_name)
            print(f"Model Name: {predictions['Model Name'].values[0]}\n")
            print(f"TEXT:\n{predictions['Text'].values[0]}\n")
            print(f"Predicted Topic:\n{predictions['Topic'].values[0]}")
            return predictions
    except Exception as e:
        logger.error(f"\nSomething Went Wrong During Prediction on Text Document:\n{e}\n", exc_info=True)
        sys.exit()

