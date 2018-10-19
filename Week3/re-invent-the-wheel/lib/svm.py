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


class SVMClassifier:
    def __init__(self, errors, alphas, C=1.0, kernel='linear', b=1, sigma=1):
        """
        Hyperparameters:
            C: float, optional (default=1.)

            kernel: string, optional (default='linear')
                Specified the kernel type to be used in the algorithm. It can
                be 'linear' or 'gaussian'.
        """
        # Hyperparameter
        self.C = C
        self.kernel = kernel

        # Lagrange multiplier vector
        self.alphas = alphas
        # Scalar bias term
        self.b = b
        # Error cache
        self.errors = errors
        # Record objective function value
        self._obj = []
