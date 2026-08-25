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

class MultiClassPerceptron:
    """
    Multiclass perceptron classifier.
    """

    def __init__(self, n_features, n_classes=10, learning_rate=1.0):
        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)
        self.learning_rate = learning_rate
        self.n_classes = n_classes

    def predict(self, x):
        """
        Predict the class with the highest score.
        """
        scores = self.weights @ x + self.bias
        return np.argmax(scores)

    def predict_batch(self, X):
        """
        Predict classes for a batch of examples.
        """
        scores = X @ self.weights.T + self.bias
        return np.argmax(scores, axis=1)

    def train_epoch(self, X, y):
        """
        Train on the dataset once.

        Returns:
            Number of mistakes made during the epoch.
        """
        mistakes = 0

        for x_i, y_i in zip(X, y):
            prediction = self.predict(x_i)

            if prediction != y_i:
                self.weights[y_i] += self.learning_rate * x_i
                self.bias[y_i] += self.learning_rate

                self.weights[prediction] -= self.learning_rate * x_i
                self.bias[prediction] -= self.learning_rate

                mistakes += 1

        return mistakes