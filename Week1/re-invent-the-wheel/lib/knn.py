"""
Nearest Neighbor Classification
The structure is inspired by scikit-learn python library.
"""

# Author: Yang Dai <daiy@mit.edu>


from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from lib import utils


class KNeighborsClassifier:
    """Classifier implementing the k-nearest neighbors vote.

    Euclidean distance is chosen as the metric.

    Parameters
    ----------
    n_neighbors : int, optional (default = 3)
        Number of neighbors used by default for kneighbors' queries.
    """

    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fit KNN model.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : 1-d numpy array (vector) of shape [n_samples].
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """

        self.X, self.y = utils.check_X_y(X, y)
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        Returns
        -------
        y_predict : 1-d numpy array (vector) of shape [n_samples].
            Class labels for each data sample.
        """

        X = utils.check_X(X)
        n_samples = X.shape[0]

        # Use 2d array instead of for loop can dramatically improve
        # performance.
        dists = np.sqrt(
            -2 * np.dot(X, self.X.T) + np.sum(self.X**2, axis=1) + \
            np.sum(X**2, axis=1).reshape(n_samples, -1)
        )

        y_predict = []
        for dist in dists:
            k_nearest_indx = dist.argsort()[:self.n_neighbors]
            nearest_vectors = self.y[k_nearest_indx]
            c = Counter(nearest_vectors)
            y_predict.append(c.most_common(1)[0][0])

        y_predict = np.array(y_predict)
        return y_predict
