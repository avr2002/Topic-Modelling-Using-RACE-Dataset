import os
import sys
import yaml
from pipeline.get_data import get_and_read_data
from pipeline.data_preprocessing import get_cleaned_train_test_data
from pipeline.vectorize_data import vectorize_text_data
from pipeline.topic_modeling import modeling
from pipeline.show_topics import document_topic, show_topic_keywords, get_model_predictions
from pipeline.predict_topics import predict_topics_on_test_data


CONFIG_PATH = "./config/config.yaml"

def run():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        model_save_path = config['model_save_path']
        model_output_save_path = config['model_output_save_path']
        top_n_keywords_in_a_topic = config['top_n_keywords_in_a_topic']

    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(model_output_save_path, exist_ok=True)


    # Get the Data
    data = get_and_read_data()

    # Get Train-Test and Cleaned Data 
    train_documents, test_documents = get_cleaned_train_test_data(df=data)

    # TF-IDF Vectorizer
    tfidf_vect, tfidf_vectorized_text_train = vectorize_text_data(data=train_documents, 
                                                                column='clean_document', 
                                                                vectorizer_type='tfidf') 

    tfidf_vectorized_text_test = tfidf_vect.transform(test_documents['clean_document'])

    # Count Vectorizer
    count_vect, count_vectorized_text_train = vectorize_text_data(data=train_documents, 
                                                                column='clean_document', 
                                                                vectorizer_type='count') 

    count_vectorized_text_test = count_vect.transform(test_documents['clean_document'])



    # Topic Modeling

    ## LSA
    print("------------------------LSA Starts------------------------\n")
    lsa_model, lsa_top = modeling(vectorized_data=tfidf_vectorized_text_train,
                                model_name="lsa",
                                model_save_path=model_save_path)

    
    lsa_document_topic_df, lsa_topic_keywords_df = get_model_predictions(vectorizer=tfidf_vect, model=lsa_model, 
                                                                         model_name='lsa', model_output=lsa_top, 
                                                                         data = train_documents, data_type='train', 
                                                                         top_n_words=top_n_keywords_in_a_topic)


    ### Test-Predictions
    lsa_document_topic_test_df, lsa_topic_keyword_test_df = predict_topics_on_test_data(model=lsa_model,
                                                                                        vectorizer=tfidf_vect,
                                                                                        test_data=test_documents,
                                                                                        vectorized_test_data=tfidf_vectorized_text_test,
                                                                                        model_name='lsa')

    print("\n------------------------LSA Ends--------------------------\n")



    ## LDA
    print("------------------------LDA Starts------------------------\n")
    lda_model, lda_output = modeling(vectorized_data=count_vectorized_text_train,
                                    model_name="lda",
                                    model_save_path=model_save_path)
    

    lda_document_topic_df, lda_topic_keywords_df = get_model_predictions(vectorizer=count_vect, model=lda_model, 
                                                                         model_name='lda', model_output=lda_output, 
                                                                         data = train_documents, data_type='train', 
                                                                         top_n_words=top_n_keywords_in_a_topic)


    ### Test-Predictions
    lda_document_topic_test_df, lda_topic_keyword_test_df = predict_topics_on_test_data(model=lda_model,
                                                                                        vectorizer=count_vect,
                                                                                        test_data=test_documents,
                                                                                        vectorized_test_data=count_vectorized_text_test,
                                                                                        model_name='lda')


    print("\n------------------------LDA Ends--------------------------\n")


    ## NMF
    print("------------------------NMF Starts------------------------\n")
    nmf_model, nmf_output = modeling(vectorized_data=tfidf_vectorized_text_train,
                                    model_name="nmf",
                                    model_save_path=model_save_path)
    
    nmf_document_topic_df, nmf_topic_keywords_df = get_model_predictions(vectorizer=tfidf_vect, model=nmf_model, 
                                                                         model_name='nmf', model_output=nmf_output, 
                                                                         data = train_documents, data_type='train', 
                                                                         top_n_words=top_n_keywords_in_a_topic)

    ### Test-Predictions
    nmf_document_topic_test_df, nmf_topic_keyword_test_df = predict_topics_on_test_data(model=nmf_model,
                                                                                        vectorizer=tfidf_vect,
                                                                                        test_data=test_documents,
                                                                                        vectorized_test_data=tfidf_vectorized_text_test,
                                                                                        model_name='nmf')

    print("\n------------------------NMF Ends--------------------------\n")
    return 