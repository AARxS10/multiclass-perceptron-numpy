"""
perceptron.py

Binary and multiclass perceptron implementations.
"""

import numpy as np


class BinaryPerceptron:
    """
    Binary perceptron classifier.

    Predicts either -1 or +1.
    """

    def __init__(self, n_features, learning_rate=1.0):
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.learning_rate = learning_rate

    def net_input(self, x):
        """
        Calculate the perceptron's raw score.
        """
        return np.dot(self.weights, x) + self.bias

    def predict(self, x):
        """
        Predict the class of a single example.
        """
        if self.net_input(x) >= 0:
            return 1
        else:
            return -1

    def update(self, x, y):
        """
        Update the weights and bias using one training example.

        y must be either -1 or +1.
        """
        prediction = self.predict(x)

        if prediction != y:
            self.weights += self.learning_rate * y * x
            self.bias += self.learning_rate * y