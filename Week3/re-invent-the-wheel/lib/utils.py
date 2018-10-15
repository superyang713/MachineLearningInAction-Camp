"""
helper functions for Logistic Regression.
"""

# Author: Yang Dai <daiy@mit.edu>


import numpy as np
import h5py
import matplotlib.pyplot as plt


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


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape(train_set_y_orig.shape[0])
    test_set_y_orig = test_set_y_orig.reshape(test_set_y_orig.shape[0])

    return (train_set_x_orig,
            train_set_y_orig,
            test_set_x_orig,
            test_set_y_orig,
            classes)


def plot_image(X, y, index):
    """
    Visualize an example image in the dataset.
    """
    plt.imshow(X[index])
    plt.axis('off')
    plt.show()


def describe_data(X_train_orig, y_train, X_test_orig, y_test):
    n_samples_train = X_train_orig.shape[0]
    n_samples_test = X_test_orig.shape[0]
    print('-' * 50)
    print("Number of training samples: {}".format(n_samples_train))
    print("Number of testing samples: {}".format(n_samples_test))
    print("X_train_orig shape: {}".format(X_train_orig.shape))
    print("y_train shape: {}".format(y_train.shape))
    print("X_test_orig shape: {}".format(X_test_orig.shape))
    print("y_test shape: {}".format(y_test.shape))
    print('-' * 50)
