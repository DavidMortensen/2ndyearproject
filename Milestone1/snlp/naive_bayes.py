from snlp.constants import OFFSET
from snlp import clf_base, evaluation

import numpy as np
from collections import defaultdict, Counter

# deliverable 3.1
def get_corpus_counts(x,y,label):
    # type: (object, object, object) -> object
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """

    counter = Counter()

    for index, instance in enumerate(x):
        if y[index] == label:
            for word in instance:
                counter[word] += instance[word]

    return counter

# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word
    '''
    #For some reason re-using the prev method, caused accuracy problems..
    #edit: Obviously the prev method doesn't take full vocab into consideration.
    #log_probs = get_corpus_counts(x, y, label)

    # computes P(x|y=label) for a specific label
    log_probs = defaultdict(float)
    for instance, y in zip(x,y):
        if y == label:
            for word in vocab:
                log_probs[word] += instance[word]

    prob_c = sum(log_probs.values()) + len(vocab) * smoothing
    for word in log_probs:
        log_probs[word] = np.log((log_probs[word] + smoothing)/prob_c)

    return log_probs

# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    weights = defaultdict(float)
    vocab = set([word for document in x for word in document])
    class_count = Counter(y)
    class_pxys = defaultdict()

    for c in set(y):
        weights[(c, OFFSET)] = np.log(class_count[c] / len(y))
        class_pxys[c] = estimate_pxy(x, y, c, smoothing, vocab)

        for c in class_pxys:
            for word in class_pxys[c]:
                weights[(c, word)] = class_pxys[c][word]

    return weights


## helper code

# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    Hint: use np.argmax over dev accuracies

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''

    dev_accuracy = []

    for value in smoothers:
        theta_nb = estimate_nb(x_tr, y_tr, value)
        y_hat = clf_base.predict_all(x_dv, theta_nb, y_dv)
        accuracy = evaluation.acc(y_hat, y_dv)
        dev_accuracy.append(accuracy)

    return (np.argmax(dev_accuracy), dev_accuracy)