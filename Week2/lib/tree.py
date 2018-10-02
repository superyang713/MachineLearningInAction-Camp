"""
Decision Tree Classification based on CART.
The structure is inspired by scikit-learn python library.
I have learned tremendously about decision tree from Google Developer YouTube
Channel.

Reference:
    https://github.com/random-forests/tutorials/blob/master/decision_tree.py

There are many places that need to be improved, such as pruning, etc. But it is
good enough for my learning purpose.
"""

# Author: Yang Dai <daiy@mit.edu>


import numpy as np


class Question:
    """
    A node is splitted based on the question asked.
    """

    def __init__(self, column_number, threshold):
        self.column_number = column_number
        self.threshold = threshold

    def ask(self, x):
        feature = x[self.column_number]
        return feature > self.threshold


class Leaf:
    """
    The end node (where no more info is gained).
    It should contain the predictions we want with only one type of label.
    """

    def __init__(self, y):
        self.prediction = self._label_counts(y)

    @staticmethod
    def _label_counts(y):
        """
        Counts the number of the samples that have the same label.
        Return:
            int: the predicted label.
        """

        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]


class Node:
    """
    A part of the tree. It is used to structure the model.
    """

    def __init__(self, question, true_node, false_node):
        self.question = question
        self.true_node = true_node
        self.false_node = false_node


class DecisionTreeClassifier:
    """
    Decision tree classifier based on CART algorithm.
    """

    def __init__(self, metric="gini"):
        # hyperparameter
        self.metric_type = metric

    def fit(self, X, y):
        """
        Build the tree, in other words, train the model, and pass it to the \
            class variable.
        Recursion should be used with multiple returns. Return should be \
            either a leaf or a node.
        """

        gain, question = self._find_best_split(X, y)

        if gain == 0:
            return Leaf(y)

        X_true_branch, y_true_branch, X_false_branch, y_false_branch = \
            self._split(X, y, question)

        true_node = self.fit(X_true_branch, y_true_branch)
        false_node = self.fit(X_false_branch, y_false_branch)
        self.tree = Node(question, true_node, false_node)
        return self.tree

    def predict(self, X):
        """
        Use the model to predict the result with the new data X.
        """
        n_samples = len(X)
        y_predict = np.empty(n_samples)

        for i, x in enumerate(X):
            y_predict[i] = self._predict(self.tree, x)

        return y_predict

    def _predict(self, tree, x):
        """
        Use the model to predict the result with the new data X.
        """

        if isinstance(tree, Leaf):
            return tree.prediction

        if tree.question.ask(x):
            return self._predict(tree.true_node, x)
        else:
            return self._predict(tree.false_node, x)

    def print_tree(self, spacing=''):
        self._print_tree(self.tree, spacing=spacing)

    def _print_tree(self, tree, spacing=''):
        """
        It is a helper method to visualize the tree if the model is small.
        Used to test model.
        """
        if isinstance(tree, Leaf):
            print(spacing + "Predict", tree.prediction)
            return

        # Print the question at this node
        print(spacing + str(tree.question))

        # Call this function recursively on the true branch
        print(spacing + '--> True:')
        self._print_tree(tree.true_node, spacing + "  ")

        # Call this function recursively on the false branch
        print(spacing + '--> False:')
        self._print_tree(tree.false_node, spacing + "  ")

    @staticmethod
    def _split(X, y, question):
        """
        Split the node into two nodes based on a question.

        Returns:
        -------
            true_branch: np array
            false_branch: np array
        """

        data = np.hstack((X, y.reshape(-1, 1)))
        true_branch = np.array([])
        false_branch = np.array([])

        for row in data:
            if question.ask(row):
                true_branch = np.vstack(
                    (true_branch, row)
                ) if true_branch.size else row.reshape(1, -1)
            else:
                false_branch = np.vstack(
                    (false_branch, row)
                ) if false_branch.size else row.reshape(1, -1)

        if len(true_branch) == 0 or len(false_branch) == 0:
            X_true_branch, y_true_branch, X_false_branch, y_false_branch = \
                [], [], [], []
        else:
            X_true_branch = true_branch[:, :-1]
            y_true_branch = true_branch[:, -1]
            X_false_branch = false_branch[:, :-1]
            y_false_branch = false_branch[:, -1]

        return X_true_branch, y_true_branch, X_false_branch, y_false_branch

    def _get_metric(self, y):
        """
        Return:
        -------
            gini or entropy: float
                a metric used to measure the information gain.
        """

        n_samples = len(y)
        _, counts = np.unique(y, return_counts=True)

        if self.metric_type == 'gini':
            impurity = 1
            for count in counts:
                probability = float(count / n_samples)
                impurity -= probability ** 2

            return impurity

        if self.metric_type == 'entropy':
            entropy = 0
            for count in counts:
                probability = float(count / n_samples)
                entropy -= probability * np.log2(probability)

            return entropy

    def _get_info_gain(self, y, y_true_branch, y_false_branch):
        """
        Calculate the info gain after a node is splitted into two more nodes.
        """

        p_true = len(y_true_branch) / len(y)
        p_false = 1 - p_true

        metric_before_split = self._get_metric(y)
        metric_true = self._get_metric(y_true_branch)
        metric_false = self._get_metric(y_false_branch)

        gain = metric_before_split - p_true * metric_true - \
            p_false * metric_false

        return gain

    def _find_best_split(self, X, y):
        """
        Return:
        -------
            best_question: an instance of Question that get the highest info \
                gain. We use it to split the tree.
            best_gain: the info gain when the best question is asked.
        """
        best_gain = 0
        best_question = None
        n_features = len(X[0])

        for column_number in range(n_features):
            thresholds = set([x[column_number] for x in X])

            for threshold in thresholds:
                question = Question(column_number, threshold)

                X_true_branch, y_true_branch, X_false_branch, y_false_branch =\
                    self._split(X, y, question)

                if len(X_true_branch) == 0 or len(X_false_branch) == 0:
                    continue

                gain = self._get_info_gain(y, y_true_branch, y_false_branch)

                if gain >= best_gain:
                    best_gain = gain
                    best_question = question

        return best_gain, best_question
