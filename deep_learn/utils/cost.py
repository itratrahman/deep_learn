# import numpy
import numpy as np


def logistic_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- logistic cost
    """

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost


def logloss(Y, AL, eps=1e-15):

    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- one hot coded matrix, shape (number of classes, number of examples)

    Returns:
    cost -- logloss
    """
    y_pred = AL.T
    y_true = Y.T

    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = np.average(-(y_true * np.log(y_pred)).sum(axis=1))

    return loss
