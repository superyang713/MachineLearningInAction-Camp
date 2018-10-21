"""
Logistic Regression
The structure is inspired by scikit-learn python library.

Important notes:
    1. X shape from the data source is (n_samples, n_features)
    2. y shape from the data source is (n_samples,)
    3. Internally, X is converted to (n_features, n_samples), and y is \
        converted to (1, n_samples)
    4. y_predict shape is (n_samples,)
    5. w shape is (1, n_features)
    6. b is scalar.
    7. Therefore, internally,  z = wX + b
    8. The implementation above is used to follow deep learning convention.
    9. Optimization is done by batch gradient descent.

"""

# Author: Yang Dai <daiy@mit.edu>


import numpy as np
from .utils import sigmoid, reshape_X, reshape_y, initialize_with_zeros


class LogisticRegression:
    """
    Classifier implementing the k-nearest neighbors vote.

    Optimization is based on Gradient Descent Algorithm.

    Parameters
    ----------
    learning_rate: float, optional (default = 0.1)
    """

    def __init__(self, learning_rate=0.1, max_iter=1000):
        # Hyperparameter
        self.learning_rate = learning_rate
        self.max_iter = max_iter    # Number of iterations to converge.

        # Important Attributes
        self.costs = []

    def fit(self, X, y):
        """
        Train the regressor with training data.

        Parameters:
        ----------
        X : data, a numpy array of size (n_samples, n_features)
        y : label vecotr, a numpy array of size (n_samples)

        Return:
        ------
        self.params : dictionary containing the optimized w and b
        """
        X = reshape_X(X)
        y = reshape_y(y)
        n_features, n_samples = X.shape
        w, b = initialize_with_zeros(n_features)

        for i in range(self.max_iter):
            _, cost = self._forward_propagate(w, b, X, y)
            grads = self._backward_progagate(w, b, X, y)
            dw = grads['dw']
            db = grads['db']

            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db

            self.costs.append(cost)

        self.params = {"w": w, "b": b}

    @staticmethod
    def _forward_propagate(w, b, X, y):
        """
        Implement the cost function for the propagation.

        Parameters:
        ----------
        w : weights vector, a numpy array of size (1, n_features)
        b : bias, a scalar
        X : data, a numpy array of size (n_features, n_samples)
        y : label vecotr, a numpy array of size (1, n_samples)

        Return:
        -------
        A : activation, a numpy array of size (1, n_samples)
        cost : negative log-likelihood cost for logistic regression
            For logistic regression, the gradient does not depend on cost \
            value, but it is still good to keep it.
        """
        n_samples = X.shape[1]
        A = sigmoid(np.dot(w, X) + b)
        cost = -1 / n_samples * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

        return A, cost

    def _backward_progagate(self, w, b, X, y):
        """
        Find the gradient of the cost function for the backward propagation.

        Parameters:
        ----------
        w : weights vector, a numpy array of size (1, n_features)
        b : bias, a scalar
        X : data, a numpy array of size (n_features, n_samples)
        y : label vecotr, a numpy array of size (1, n_samples)

        Return:
        -------
        grads : dict
            A dictionary containing the gradient of the loss with respect to \
            w and b.
        """

        n_features, n_samples = X.shape
        A, cost = self._forward_propagate(w, b, X, y)
        dw = 1 / n_samples * np.dot(A - y, X.T)
        db = 1 / n_samples * np.sum(A - y)

        assert(dw.shape == (1, n_features))
        assert(db.dtype == float)

        grads = {'dw': dw, 'db': db}

        return grads

    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression \
            parameters (w, b)

        Parameters:
        ----------
        X : data, a numpy array of size (n_features, n_samples)

        Returns:
        -------
        y_predict : a numpy array (vector) containing all predictions.
        '''
        X = reshape_X(X)
        n_samples = X.shape[1]
        w = self.params['w']
        b = self.params['b']
        A = sigmoid(np.dot(w, X) + b)

        y_predict = np.rint(A)
        y_predict = y_predict.reshape(n_samples)
        assert(y_predict.shape == (n_samples,))

        return y_predict

    def get_accuracy(self, X, y):
        y_predict = self.predict(X)
        n_samples = X.shape[0]
        accuracy = np.sum(y_predict == y) / n_samples
        return accuracy
