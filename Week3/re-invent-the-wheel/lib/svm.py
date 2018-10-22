"""
Two methods to optimize Support Vector Machine:
    1. The Primal form:
    This formulation can be solved using existing quadratic programming
    solvers. However, there is another formulation of the problem that can
    be solved without using quadratic programming techniques.

    2. The dual form:
    Using Lagrange duality, the optimization problem can be re-formulated
    into the dual form.

In this script, the dual form optimization method will be used. More
specifically, I will be using Sequential Minimal Optimization (SMO) algorithm
to solve the optimizatio problem. The details about this optimization algorithm
can be found by the original paper:

https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf

SMO works by breaking down the dual form of the SVM optimization problem into
many smaller optimization problems that are more easily solvable. In the
nutshell the algorithm works like this:
    1. Two multiplier values (αi and αj) are selected out and their values are
    optimized while holding all other α values constant.

    2. Once these two are optimized, another two are chosen and optimized over.

    3. Choosing and optimizing repeats until the convergence, which is
    determined based on the problem constraints. Heuristics are used to select
    two α values to optimize over, helping to speed up convergence. The
    heuristics are based on error cache that is stored while training the
    model.

What we're looking for:
    What we want out of the algorithm is a vector of  α  values that are mostly
    zeros, except where the corresponding training example is closest to the
    decision boundary. These examples are our support vectors and should lie
    near the decision boundary. We should end up with a few of them once our
    algorithm has converged. What this implies is that the resultant decision
    boundary will only depend on the training examples closest to it. If we
    were to add more examples to our training set that were far from the
    decision boundary, the support vectors would not change. However, labeled
    examples closer to the decision boundary can exert greater influence on the
    solution, subject to the degree of regularization. In other words,
    non-regularized (hard-margin) SVMs can be sensitive to outliers, while
    regularized (soft-margin) models are not.

Goal:
    Implement a binary-class SVC that is able to make use of the kernal trick.
    Use SMO to solve the SVM optimization problem.
"""


import numpy as np
import matplotlib.pyplot as plt


class SVMClassifier:
    """
    Parameters::
    ----------
    kernel : string, optional (default='linear')
        Specified the kernel type to be used in the algorithm. It can be
        'linear' or 'gaussian'.

    C : float, optional (default = 1.0)
        Penalty (Regularization) parameter of the error term.

    epsilon : float (default = 0.1)
        It specifies the epsilon-tube within which no penalty is associated
        in the training loss function with points predicted within a distance
        epsilon from the actual value.

    tol : float, optional (default = 1e-3)
        Tolerance for stopping criterion.

    """
    def __init__(self, kernel='linear', C=1.0, epsilon=0.1, tol=1e-3):
        # Hyperparameters
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.tol = tol

        # lagrange multiplier vector
        self.alphas = None

        # scalar bias term for linear kernal
        self.b = 0
        # width parameter for gaussian kernal
        self.sigma = 1

        # Error cache
        self.errors = None
        # Record objective function value
        self._obj = []

        # training data needs to be passed around for kernal trick
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Train the classifier with training data.

        Parameters:
        ----------
        X : data, a numpy array of size (n_samples, n_features)
        y : label vecotr, a numpy array of size (n_samples)

        Return:
        ------
        self : object
        """

        self.X = X
        self.y = y

        self._init_alpha()
        self._init_error(self.X)

        num_changed = 0
        examine_all = 1

        while(num_changed > 0) or (examine_all):
            num_changed = 0
            if examine_all:
                # loop over all training examples
                for i in range(self.alphas.shape[0]):
                    examine_result = self._examine_example(i)
                    num_changed += examine_result
                    if examine_result:
                        objective = self._objective_function(self.alphas)
                        self._obj.append(objective)
            else:
                # loop over examples where alphas are not already at their
                # limits
                for i in np.where(
                    (self.alphas != 0) & (self.alphas != self.C)
                )[0]:
                    examine_result = self._examine_example(i)
                    num_changed += examine_result
                    if examine_result:
                        objective = self._objective_function(self.alphas)
                        self._obj.append(objective)

            if examine_all == 1:
                examine_all = 0

            elif num_changed == 0:
                examine_all = 1

        return self

    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 based on the result from decsion
        function.

        Parameters:
        ----------
        X : data, a numpy array of size (n_samples, n_features)

        Returns:
        -------
        y_predict : a numpy array (vector) containing all predictions.
        '''

        decision = self.decision_function(X)
        y_predict = np.sign(decision)
        y_predict[y_predict == -1] = 0

        return y_predict

    def decision_function(self, X):
        """
        Applies the SVM decision function to the data X.

        parameters:
        ----------
        X : data, a numpy array of size (n_samples, n_features)

        return:
        ----------
        result : a numpy array (vector) with the shape of (n_samples,)
        """

        if self.kernel == 'linear':
            result = np.dot(
                (self.alphas * self.y), self._linear_kernel(self.X, X)
            ) - self.b
        if self.kernel == 'gaussian':
            result = np.dot(
                (self.alphas * self.y), self._gaussian_kernel(self.X, X)
            ) - self.b

        return result

    def _init_alpha(self):
        """
        Initialize a vector of alphas.

        parameters:
        ----------
        self

        return:
        ----------
        self : object
        """

        self.alphas = np.zeros(self.X.shape[0])
        return self

    def _init_error(self, X):
        """
        Initialize a vector of errors.

        parameters:
        ----------
        X : data, a numpy array of size (n_samples, n_features)

        return:
        ----------
        self : object
        """

        self.errors = self.decision_function(X) - self.y
        return self

    def _linear_kernel(self, X1, X2):
        """
        Returns the linear combination of arrays X1 and X2 with the kernal
        scalar b.

        parameters:
        ----------
        X1 : data, a numpy array of size (n_samples, n_features)
        X2 : data, a numpy array of size (n_samples, n_features)

        returns:
        -------
        result : kernelized vector dot production in the dual form.
        """
        result = np.dot(X1, X2.T) + self.b

        return result

    def _gaussian_kernel(self, X1, X2):
        """
        Returns the gaussian similarity of arrays X1 and X2 with kernel width
        parameter sigma.

        parameters:
        ----------
        X1 : data, a numpy array of size (n_samples, n_features)
        X2 : data, a numpy array of size (n_samples, n_features)

        returns:
        -------
        result : kernelized vector dot production in the dual form.
        """

        if np.ndim(X1) == 1 and np.ndim(X2) == 1:
            result = np.exp(- np.linalg.norm(X1 - X2) / (2 * self.sigma ** 2))

        elif ((np.ndim(X1) > 1 and np.ndim(X2) == 1) or
              (np.ndim(X1) == 1 and np.ndim(X2) > 1)):
            result = np.exp(
                - np.linalg.norm(X1 - X2, axis=1) / (2 * self.sigma ** 2)
            )

        elif np.ndim(X1) > 1 and np.ndim(X2) > 1:
            result = np.exp(- np.linalg.norm(
                X1[:, np.newaxis] - X2[np.newaxis, :], axis=2
            ) / (2 * self.sigma ** 2))

        return result

    def _objective_function(self, alphas):
        """
        Objective function to optimize

        Returns:
        -------
        the SVM objective function based on the input model.
        """

        if self.kernel == 'linear':
            bulk = self.y * self.y * self._linear_kernel(self.X, self.X) * \
                alphas * alphas
        elif self.kernel == 'gaussian':
            bulk = self.y * self.y * self._gaussian_kernel(self.X, self.X) * \
                alphas * alphas

        return np.sum(alphas) - 1 / 2 * np.sum(bulk)

    def _take_step(self, index1, index2):
        # Skip if chosen alphas are the same
        if index1 == index2:
            return 0

        alpha1 = self.alphas[index1]
        alpha2 = self.alphas[index2]

        low, high = self._compute_bounds(index1, index2)
        if (low == high):
            return 0

        a2 = self._compute_a2(index1, index2, low, high)
        if self._not_optimizable(a2, alpha2):
            return 0
        a1 = self._compute_a1(index1, index2, a2)

        b_new = self._compute_b(index1, index2, a1, a2)

        # Update model object with new alphas & threshold
        self.alphas[index1] = a1
        self.alphas[index2] = a2

        # Update error cache
        self._update_error_cache(b_new, index1, index2, a1, a2, alpha1, alpha2)

        # Update model threshold
        self.b = b_new

        return 1

    def _examine_example(self, index2):
        y2 = self.y[index2]
        alpha2 = self.alphas[index2]
        error2 = self.errors[index2]
        r2 = error2 * y2

        # Proceed if error is within specified tolerance (tol)
        if ((r2 < -self.tol and
             alpha2 < self.C) or (r2 > self.tol and alpha2 > 0)):

            if (len(self.alphas[(self.alphas != 0) &
                                (self.alphas != self.C)]) > 1):
                # Use 2nd choice heuristic is choose max difference in error
                if self.errors[index2] > 0:
                    index1 = np.argmin(self.errors)
                elif self.errors[index2] <= 0:
                    index1 = np.argmax(self.errors)
                step_result = self._take_step(index1, index2)
                if step_result:
                    return 1

            # Loop through non-zero and non-C alphas, starting at a random
            # point
            for index1 in np.roll(
                np.where((self.alphas != 0) & (self.alphas != self.C))[0],
                np.random.choice(np.arange(self.X.shape[0]))
            ):
                step_result = self._take_step(index1, index2)
                if step_result:
                    return 1

            # loop through all alphas, starting at a random point
            for index1 in np.roll(
                np.arange(self.X.shape[0]),
                np.random.choice(np.arange(self.X.shape[0]))
            ):
                step_result = self._take_step(index1, index2)
                if step_result:
                    return 1

        return 0

    def _compute_bounds(self, index1, index2):
        """
        Compute lower and higher bounds on new possible alpha values
        """
        y1 = self.y[index1]
        y2 = self.y[index2]
        alpha1 = self.alphas[index1]
        alpha2 = self.alphas[index2]

        if y1 == y2:
            low = max(0, alpha1 + alpha2 - self.C)
            high = min(self.C, alpha1 + alpha2)
        else:
            low = max(0, alpha2 - alpha1)
            high = min(self.C, self.C + alpha2 - alpha1)

        return low, high

    def _compute_eta(self, index1, index2):
        """
        Compute kernel & 2nd derivative eta
        """
        x1 = self.X[index1]
        x2 = self.X[index2]

        if self.kernel == 'linear':
            k11 = self._linear_kernel(x1, x1)
            k12 = self._linear_kernel(x1, x2)
            k22 = self._linear_kernel(x2, x2)
        elif self.kernel == 'gaussian':
            k11 = self._gaussian_kernel(x1, x1)
            k12 = self._gaussian_kernel(x1, x2)
            k22 = self._gaussian_kernel(x2, x2)
        eta = 2 * k12 - k11 - k22

        return eta, k11, k12, k22

    def _not_optimizable(self, a2, alpha2):
        """
        Check if a2 can be further optimized.
        """
        if np.abs(a2 - alpha2) < self.epsilon * (a2 + alpha2 + self.epsilon):
            return True

    def _compute_a1(self, index1, index2, a2):
        """
        Compute new alpha1 (a1)
        """
        y1 = self.y[index1]
        y2 = self.y[index2]
        alpha1 = self.alphas[index1]
        alpha2 = self.alphas[index2]
        a1 = alpha1 + y1 * y2 * (alpha2 - a2)

        return a1

    def _compute_a2(self, index1, index2, low, high):
        """
        Compute new alpha2 (a2)
        """

        y2 = self.y[index2]
        alpha2 = self.alphas[index2]
        error1 = self.errors[index1]
        error2 = self.errors[index2]

        eta = self._compute_eta(index1, index2)[0]

        if eta < 0:
            a2 = alpha2 - y2 * (error1 - error2) / eta
            # Clip a2 based on bounds L & H
            a2 = max(low, min(high, a2))

        # If eta is non-negative, move new a2 to bound with greater objective
        # function value
        else:
            alphas_copy = self.alphas.copy()
            alphas_copy[index2] = low
            # objective function output with a2 = L
            low_objective = self._objective_function(alphas_copy)

            alphas_copy[index2] = high
            # objective function output with a2 = H
            high_objective = self._objective_function(alphas_copy)

            if low_objective > (high_objective + self.epsilon):
                a2 = low
            elif low_objective < (high_objective - self.epsilon):
                a2 = high
            else:
                a2 = alpha2

        # Push a2 to 0 or C if very close
        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (self.C - 1e-8):
            a2 = self.C

        return a2

    def _compute_b(self, index1, index2, a1, a2):
        """
        Update threshold b to reflect newly calculated alphas
        Calculate both possible thresholds
        """
        y1 = self.y[index1]
        y2 = self.y[index2]
        alpha1 = self.alphas[index1]
        alpha2 = self.alphas[index2]
        error1 = self.errors[index1]
        error2 = self.errors[index2]
        _, k11, k12, k22 = self._compute_eta(index1, index2)

        b1 = error1 + \
            y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.b
        b2 = error2 + \
            y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.b

        # Set new threshold based on if a1 or a2 is bounded by low and/or high
        if 0 < a1 and a1 < self.C:
            b_new = b1
        elif 0 < a2 and a2 < self.C:
            b_new = b2
        # Average thresholds if both are bound
        else:
            b_new = (b1 + b2) * 0.5

        return b_new

    def _update_error_cache(self, b_new, index1, index2,
                            a1, a2, alpha1, alpha2):
        """
        Update error cache
        Error cache for optimized alphas is set to 0 if they're unbound

        Returns:
        -------
        self : object
        """
        y1 = self.y[index1]
        y2 = self.y[index2]

        for index, alpha in zip([index1, index2], [a1, a2]):
            if 0.0 < alpha < self.C:
                self.errors[index] = 0.0

        # Set non-optimized errors based on equation 12.11 in Platt's book
        non_optimized = [n for n in range(self.X.shape[0])
                         if (n != index1 and n != index2)]
        if self.kernel == "linear":
            correction = y1 * (a1 - alpha1) * \
                self._linear_kernel(self.X[index1], self.X[non_optimized]) + \
                y2 * (a2 - alpha2) * \
                self._linear_kernel(self.X[index2], self.X[non_optimized]) + \
                self.b - b_new
        elif self.kernel == "gaussian":
            correction = y1 * (a1 - alpha1) * \
                self._gaussian_kernel(self.X[index1], self.X[non_optimized]) +\
                y2 * (a2 - alpha2) * \
                self._gaussian_kernel(self.X[index2], self.X[non_optimized]) +\
                self.b - b_new
        self.errors[non_optimized] = self.errors[non_optimized] + correction

        return self

    def plot_decision_boundary(self, ax, resolution=100):
            """
            Plots the model's decision boundary on the input axes object.
            Range of decision boundary grid is determined by the training data.
            Returns decision boundary grid and axes object (`grid`, `ax`).
            """

            colors = ('b', 'k', 'r')
            # Generate coordinate grid of shape [resolution x resolution]
            # and evaluate the model over the entire space
            xrange = np.linspace(
                self.X[:, 0].min(), self.X[:, 0].max(), resolution)
            yrange = np.linspace(
                self.X[:, 1].min(), self.X[:, 1].max(), resolution)
            grid = [[
                self.decision_function(np.array([xr, yr])) for yr in yrange
            ] for xr in xrange]
            grid = np.array(grid).reshape(len(xrange), len(yrange))

            # Plot decision contours using grid and
            # make a scatter plot of training data
            ax.contour(
                xrange, yrange, grid, (-1, 0, 1), linewidths=(1, 1, 1),
                linestyles=('--', '-', '--'), colors=colors
            )
            ax.scatter(
                self.X[:, 0], self.X[:, 1], c=self.y, cmap=plt.cm.viridis,
                lw=0, alpha=0.5
            )

            # Plot support vectors (non-zero alphas)
            # as circled points (linewidth > 0)
            mask = self.alphas != 0.0
            ax.scatter(
                self.X[:, 0][mask], self.X[:, 1][mask], c=self.y[mask],
                cmap=plt.cm.viridis
            )

            return grid, ax
