import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

CONFIG_PATH = "./config/config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    model_output_save_path = config['model_output_save_path']
    model_output_folder_names = config['model_output_folder_names']

os.makedirs(model_output_save_path, exist_ok=True)

def document_topic(model_output, data:pd.DataFrame, 
                   data_type:str, model_name:str, topic_keywords_df:pd.DataFrame):
    '''
    Returns a dataframe for each document having topic weightages
    and the dominant topic for each document.
    
    :args:
        model_output: Transformed data by the model
        # n_topics: int - Number of topics extracted from the model
        data: pd.DataFrame - Test/Train Data
        data_type: str - 'train' or 'test'
        model_name: str - Name of the model - ['lsa', 'lda', 'nmf']
    '''
    try:
        model_name = model_name.lower()

        # Creating the folder where the csv file will be saved
        model_output_path = os.path.join(model_output_save_path, model_output_folder_names[model_name])
        os.makedirs(model_output_path, exist_ok=True)
        
        file_name = data_type + "_document_topic_" + model_name + ".csv"
        file_save_path = os.path.join(model_output_path, file_name)
        
        # Make the pandas dataframe
        df_document_topic = pd.DataFrame()
        
        # Get dominant topic for each document
        dominant_topic = np.argmax(model_output, axis=1)

        df_document_topic['Document'] = data.document.values
        df_document_topic["Dominant_Topic"] = dominant_topic

        # Get the keywords for dominant topic
        topic_keywords = []
        for topic_num in dominant_topic:
            topic_keywords.append(list(topic_keywords_df.iloc[topic_num, 1:]))

        df_document_topic['Topic_Keywords'] = topic_keywords
        
        df_document_topic.to_csv(file_save_path, index=False)

        print(f"{model_name.upper()} Model Output: Predicted Document-Topic DataFrame for {data_type.upper()}-Set Saved to [{file_save_path}]")
        return df_document_topic
    except Exception as e:
        logger.error(f"\nError while creating the topic-document frame for the model: [{model_name}]\n {e}", exc_info=True)



# Let's views top-n key-words for each topic
def show_topic_keywords(vectorizer, model, top_n_words:int, model_name:str):
    """
    Gives top n key-words for each topic in the model
    
    :params:
        vectorizer: text-vectorizer object
        model: lsa/lda/nmf model object
        top_n_words: int
        model_name: str - Name of the model - ['lsa', 'lda', 'nmf']
        data_type: str - 'train' or 'test'
    """
    try:
        model_name = model_name.lower()

        # Creating the folder where the csv file will be saved
        model_output_path = os.path.join(model_output_save_path, model_output_folder_names[model_name])
        os.makedirs(model_output_path, exist_ok=True)
        
        file_name = "topic_keywords_" + model_name + ".csv"
        file_save_path = os.path.join(model_output_path, file_name)

        keywords = np.array(vectorizer.get_feature_names_out()) # vocabulary of words built by the vectorizer
        
        topic_keywords = []
        
        for topic_weight in model.components_:
            top_keyword_locs = (-topic_weight).argsort()[:top_n_words]
            
            # -topic_weight, negative is applied so that we get index of words 
            # in desecending order of their prob. of occuranace in a topic
            
            topic_keywords.append(keywords.take(top_keyword_locs))


        topic_keywords_df = pd.DataFrame(topic_keywords)

        topic_keywords_df.columns = ['Word '+ str(i) for i in range(topic_keywords_df.shape[1])]
        topic_keywords_df.index = ['Topic '+ str(i) for i in range(topic_keywords_df.shape[0])]

        topic_keywords_df.to_csv(file_save_path)

        print(f"{model_name.upper()} Model Output: Topic-Keywords DataFrame for Saved to [{file_save_path}]")
        return topic_keywords_df
    except Exception as e:
        logger.error(f"\nError while creating the topic-keywords frame for the model: [{model_name}]\n {e}", exc_info=True)



def get_model_predictions(vectorizer, model, model_name:str,
                          model_output, data:pd.DataFrame, 
                          data_type:str, top_n_words:int=30):
    """
    Returns Topic Predictions DataFrame for the passed data and Topic KeyWords DataFrame.
        (document_topic_df, topic_keywords_df)
    :Args:
        vectorizer: Vectorizer Used on text Data: TfidfVectorizer, CountVectorizer
        model: Model Object 
        model_name: Model Name ['lsa', 'lda', 'nmf'] 
        model_output: transformed model output on data using model.transform() or model.fit_transform()
        data: pd.DataFrame
        data_type:str: type of data: 'train' or 'test'
        top_n_words:int: Top n keywords you want in the topic (default = 30)
    """
    model_name = model_name.lower()
    topic_keywords_df = show_topic_keywords(vectorizer=vectorizer, model=model, 
                                            top_n_words=top_n_words, model_name=model_name)
    
    document_topic_df = document_topic(model_output=model_output, data=data, 
                                       model_name=model_name, data_type=data_type, 
                                       topic_keywords_df=topic_keywords_df)
    
    return document_topic_df, topic_keywords_df
    