"""
helper functions for Logistic Regression.
"""

# Author: Yang Dai <daiy@mit.edu>


import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z.

    Parameters:
    ----------
    z -- A scalar or numpy array of any size.

    Return:
    ------
    s -- sigmoid(z)
    """

    s = 1 / (1 + np.exp(-z))
    return s


def reshape_X(X):
    """
    The original shape of X is (n_samples, n_features). Since logistic \
        regression can also be applied to neural network, I am following the \
        deep learning convention to change the shape of X to \
        (n_features, n_samples)
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1).T


def reshape_y(y):
    return y.reshape(1, -1)


def initialize_with_zeros(n_features):
    """
    This function creates a vector of zeros of shape (1, n_features) for w \
        and initializes b to 0.

    Parameters:
    ----------
    n_features : size of the w vector we want (or number of parameters in \
        this case)

    Returns:
    -------
    w : initialized vector of shape (1, n_features)
    b : initialized scalar (corresponds to the bias)
    """
    w = np.zeros((1, n_features))
    b = 0

    assert(w.shape == (1, n_features))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b
