import logging
from pipeline.show_topics import document_topic, show_topic_keywords


logger = logging.getLogger(__name__)


def predict_topics_on_test_data(model, vectorizer, test_data, vectorized_test_data, model_name):
    """
    Predicts topics on test data.
    Returns Topic-Document Weights DataFrame and Topic-Keyword DataFrame for Test Data

    :args:
        model: model object
        test_data: pd.DataFame - Test DataFrame, 
        vectorized_test_data: Vectorized Test Data, 
        vectorizer: Tfidf/Count Vectorizer Object, 
        model_name: str - ['lda', 'lsa', 'nmf']
    """
    try:
        print(f"\n{model_name.upper()} Model Predictions on Test Data...")
        model_output = model.transform(vectorized_test_data)

        test_document_topic_df = document_topic(model_output=model_output,
                                                n_topics=model.get_params()['n_components'],
                                                data=test_data,
                                                model_name=model_name,
                                                data_type='test')
        
        test_topic_keyword_df = show_topic_keywords(vectorizer=vectorizer,
                                                    model=model, 
                                                    top_n_words=15,
                                                    model_name=model_name,
                                                    data_type='test')
        
        return test_document_topic_df, test_topic_keyword_df
    except Exception as e:
        logger.error(f"\nError Occured during Model Prediction:\n{e}", exc_info=True)