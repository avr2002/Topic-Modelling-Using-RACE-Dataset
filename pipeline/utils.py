import sys
import yaml
import os.path
import logging
from collections import Counter
from pipeline.run_pipeline import run

logger = logging.getLogger(__name__)

CONFIG_PATH = './config/check.yaml'

def validate_files():
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            files = config['files']

        check = True
        for file in files:
            if not os.path.isfile(file):
                print("All Required Files Are Not Present For Model Predictions!")
                check = False
                break
        if check:
            return check
        else:
            text = "Would you like to re-run the Pipeline?: Press 1\nElse Press 0 to Exit\n"
            user_input= int(input(text))
            if user_input == 0:
                sys.exit()
            else:
                run()
    except Exception as e:
        logger.error(f"\nError Occured While Validating Files for Model Prediction!\n{e}\n", exc_info=True)
        sys.exit()





# Define helper functions
def get_keys(topic_matrix):
    '''
    Returns an integer list of predicted topic categories for a given topic matrix
    '''
    keys = topic_matrix.argmax(axis=1).tolist()
    return keys

def keys_to_counts(keys):
    '''
    returns a tuple of topic categories and their accompanying magnitudes for a given list of keys
    '''
    count_pairs = Counter(keys).items()
    
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)