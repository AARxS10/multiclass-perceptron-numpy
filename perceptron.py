import numpy as np


class BinaryPerceptron:
    """
    Binary perceptron classifier.

    Predicts either 0 or 1.
    Weights and bias are initialised to 0.1.
    """

    def __init__(self, n_features, learning_rate=0.01):
        self.n_features = n_features
        self.learning_rate = learning_rate

        self.weights = np.full(
            n_features,
            0.1,
            dtype=float
        )

        self.bias = 0.1

    def score(self, X):
        """
        Return the raw perceptron score.

        X can be:
            (D,)     for one example
            (N, D)   for multiple examples
        """
        return X @ self.weights + self.bias

    def predict(self, X):
        """
        Predict binary labels 0 or 1.
        """
        scores = self.score(X)
        return (scores >= 0).astype(int)

    def train(self, X, y):
        """
        Train for one epoch using the perceptron learning rule.

        y must contain binary labels 0 or 1.

        Returns:
            Number of mistakes during the epoch.
        """

        mistakes = 0

        for i in range(len(X)):

            x_i = X[i]
            y_i = y[i]

            prediction = self.predict(x_i)

            if prediction != y_i:

                error = y_i - prediction

                self.weights += (
                    self.learning_rate * error * x_i
                )

                self.bias += (
                    self.learning_rate * error
                )

                mistakes += 1

        return mistakes


class MultiClassPerceptron:
    """
    One-vs-rest multiclass perceptron.

    Maintains one BinaryPerceptron for each class.
    """

    def __init__(
        self,
        n_features,
        n_classes=10,
        learning_rate=0.01
    ):

        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate

        # One binary perceptron for each class
        self.perceptrons = [
            BinaryPerceptron(
                n_features=n_features,
                learning_rate=learning_rate
            )
            for _ in range(n_classes)
        ]

        # Compatibility with your test_perceptron.py
        # This lets it inspect the weights.
        self.weights = np.vstack([
            p.weights for p in self.perceptrons
        ])

    def train_epoch(self, X, y, rng=None):
        """
        Train all binary perceptrons for one epoch.

        For each class:
            1 = image belongs to this class
            0 = image belongs to another class

        The training examples are shuffled between epochs.
        """

        if rng is None:
            rng = np.random.default_rng()

        indices = np.arange(len(X))
        rng.shuffle(indices)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        total_mistakes = 0

        for class_id in range(self.n_classes):

            binary_labels = (
                y_shuffled == class_id
            ).astype(int)

            mistakes = self.perceptrons[class_id].train(
                X_shuffled,
                binary_labels
            )

            total_mistakes += mistakes

        # Keep compatibility weights up to date
        self.weights = np.vstack([
            p.weights for p in self.perceptrons
        ])

        return total_mistakes

    def score(self, X):
        """
        Return the raw score from each binary perceptron.

        Returns shape:
            (N, 10)
        """

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        scores = np.column_stack([
            p.score(X)
            for p in self.perceptrons
        ])

        return scores

    def predict(self, X):
        """
        Predict the multiclass label.

        Returns:
            Class number from 0 to 9.
        """

        X = np.asarray(X)

        scores = self.score(X)

        predictions = np.argmax(
            scores,
            axis=1
        )

        # If one image was supplied, return a single integer
        if predictions.size == 1:
            return int(predictions[0])

        return predictions

    def predict_batch(self, X):
        """
        Predict a batch of images.

        Returns:
            NumPy array containing labels 0-9.
        """

        X = np.asarray(X)

        scores = self.score(X)

        return np.argmax(
            scores,
            axis=1
        )