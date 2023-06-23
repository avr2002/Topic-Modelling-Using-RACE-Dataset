import yaml
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

CONFIG_PATH = "./config/config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    data_trans_config = config['data_transformation_config']
    tfidf_params = data_trans_config['tfidf_vect_params']
    count_params = data_trans_config['count_vect_params']


def vectorize_text_data(data:pd.DataFrame, column:str, vectorizer_type:str)->tuple:
    """
    Return the vectorizer object and vectorized test
    Args:
        data: pd.DataFrame: Cleaned text document
        column: str: Column_name/feature of the data
        vectorizer_type: str: Tfidf or Count
    Output:
        object of vectorizer, transformed dataset
    """
    try:
        if vectorizer_type.lower() == "tfidf":
            # tfidf_vect = TfidfVectorizer(min_df=3, max_df=0.85,
            #                              max_features=4000, ngram_range=(1,2),
            #                              preprocessor=" ".join)
            tfidf_params['ngram_range'] = (1,2)
            #tfidf_params['preprocessor'] = " ".join
            
            tokenized_data = data[column].apply(lambda x: x.split())

            tfidf_vect = TfidfVectorizer(**tfidf_params, preprocessor=" ".join)
            vectorized_text = tfidf_vect.fit_transform(tokenized_data)

            print("Text Vectorization Successful using TfidfVectorizer\n")
            return (tfidf_vect, vectorized_text)
        elif vectorizer_type.lower() == "count":
            # count_vect = CountVectorizer(analyzer="word",
            #                              min_df = 10, # Consider a word for the vocab. only if it occurs >= min_dif in whole corpus
            #                              stop_words="english",
            #                              lowercase=True)
            count_vect = CountVectorizer(**count_params)
            vectorized_txt = count_vect.fit_transform(data[column])

            print("Text Vectorization Successful using CountVectorizer\n")
            return (count_vect, vectorized_txt)
        else:
            raise Exception(f"\nWrong Vectorizer Type Passed: [{vectorizer_type}].\nAllowed Vectorized Type: ['tfidf', 'count']")
    except Exception as e:
        logger.error(f"\nError occured in vectorizing the text data:\n{e}\n", exc_info=True)


