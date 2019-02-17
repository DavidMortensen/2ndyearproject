from collections import Counter

import pandas as pd
import numpy as np
import keras
import copy

# deliverable 1.1
def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''

    text = text.split()
    counter = Counter()

    for word in text:
        counter[word] += 1
    return(counter)

# deliverable 1.2
def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''

    counter = Counter()

    for word in bags_of_words:
        for key, value in word.items():
            counter[key] += value
    return(counter)


# deliverable 1.3
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    
    bow1 = set(bow1)
    bow2 = set(bow2)

    return(bow1 - bow2)


# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    
    vocab = []
    #target_data_copy = new_target_data
    new_target_data = copy.deepcopy(target_data)
 
    for key, value in enumerate(target_data):
        for i in value:
            if training_counts[i] < min_counts:
                del new_target_data[key][i]
            if training_counts[i] >= min_counts:
                vocab.append(i)
    
    return new_target_data, set(vocab)
   

# deliverable 4.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''
    vocab = sorted(vocab)

    #constructing empty array, to insert into
    np_array = np.zeros((len(bags_of_words), len(vocab)))

    for index, instance in enumerate(bags_of_words):
        for word in instance.keys():
            if word in vocab:
                np_array[index, vocab.index(word)] += instance[word]
    return np_array


# deliverable 4.2
def convert_categ_label_to_vec(Ys):
    """
    convert categorical label to vector
    :param Ys:
    :return:
    """
    label_set = sorted(set(Ys))

    y_num = np.array([label_set.index(i) for i in Ys])
    return (keras.utils.to_categorical(y_num, num_classes = len(np.unique(Ys))))


### helper code

def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())
