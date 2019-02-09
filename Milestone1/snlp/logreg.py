from snlp import evaluation

import keras
from keras.layers import Dense, Activation


import matplotlib.pyplot as plt

# deliverable 4.3
def build_linear(X_tr, Y_tr, loss, optimizer='sgd', metrics=['accuracy']):
    '''
    Build a linear model in torch

    :param X_tr: the set of training documents
    :param Y_tr: the set of training labels
    :returns: PyTorch linear model
    :rtype: PyTorch model
    '''
    #change these
    size1 = 99
    size2 = 9

    raise NotImplementedError


def forward(model, X):
    """
    run the forward pass
    :param model:
    :param X:
    :return:
    """
    return model.predict(X)




######################### helper code

def plot_results(losses, accuracies):
    fig,ax = plt.subplots(1,2,figsize=[12,2])
    ax[0].plot(losses)
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('iteration');
    ax[1].plot(accuracies);
    ax[1].set_ylabel('dev set accuracy')
    ax[1].set_xlabel('iteration');
