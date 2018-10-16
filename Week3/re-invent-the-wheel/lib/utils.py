"""
helper functions for Logistic Regression.
"""

# Author: Yang Dai <daiy@mit.edu>

import os

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


def load_data(fname, delimiter='\t'):
    """
    Load data from a text file, which must be located in data/.

    No headers.
    Each row in the text file must have the same number of values.

    In production, this function should be replaced by numpy.loadtxt or \
        pd.read_csv

    Parameters
    ----------
    fname: str or path-like object (absolute or relative to the current \
        working dir)

    Returns
    -------
    result: numpy array.
    X : feature array with the shape (n_samples, n_features)
    y : label array with the shape (n_samples,)
        Features and labels.
    """

    folder = os.path.abspath(os.getcwd())
    filepath = os.path.join(folder, 'data', fname)
    with open(filepath, 'r') as fin:
        raw_data = fin.readlines()
        n_rows = len(raw_data)
        n_cols = len(raw_data[0].split(delimiter))

        X = np.empty((n_rows, n_cols - 1))
        y = []
        for i, row in enumerate(raw_data):
            X[i, :] = row.split(delimiter)[:-1]
            y.append(row.split(delimiter)[-1].strip())

        y = np.array(y)

    return X, y


def split_train_test(X, y, ratio=0.2):
    """
    Create new training and test data sets based on the provided ratio

    Parameters:
    ----------
    X : array.
        Feature array with the shape (n_samples, n_features)
    y : array.
        Label array with the shape (n_samples,)
    ratio: float
        The ratio of n_training_samples and n_test_samples.

    Returns
    -------
    X_train : array.
        Training feature array.
    y_train : array.
        Training labels.
    X_test : array.
        Test feature array.
    y_test : array.
        Test labels.
    """

    shuffled_indices = np.random.permutation(len(X))
    n_test_samples = int(len(X) * ratio)
    test_indices = shuffled_indices[:n_test_samples]
    train_indices = shuffled_indices[n_test_samples:]
    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_test = X[test_indices, :]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test
