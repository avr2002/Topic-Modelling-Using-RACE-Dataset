import numpy as np
from collections import Counter

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