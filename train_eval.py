"""
train_eval.py

Training loop with configurable stopping criteria, and evaluation
utilities (metrics + confusion matrix) built on top of sklearn.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


def train_model(model, X_train, y_train, X_val, y_val,
                stopping_criterion="fixed_epochs",
                max_epochs=50, error_threshold=None, patience=None,
                random_state=None):
    """
    Train a MultiClassPerceptron using one of several stopping criteria.
    """

    train_acc_history = []
    val_acc_history = []

    rng = np.random.default_rng(random_state)

    best_val_accuracy = -np.inf
    epochs_without_improvement = 0

    stopped_reason = "max_epochs_reached"

    for epoch in range(max_epochs):

        # Shuffle training data each epoch
        indices = rng.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # Train for one epoch
        mistakes = model.train_epoch(X_shuffled, y_shuffled)

        # Calculate training accuracy
        train_predictions = model.predict_batch(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)

        # Calculate validation accuracy
        val_predictions = model.predict_batch(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)

        # Store history
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)

        # Stopping criterion 1: fixed number of epochs
        if stopping_criterion == "fixed_epochs":
            if epoch + 1 == max_epochs:
                stopped_reason = "max_epochs_reached"
                break

        # Stopping criterion 2: training error threshold
        elif stopping_criterion == "error_threshold":
            if error_threshold is None:
                raise ValueError(
                    "error_threshold must be provided for error_threshold stopping."
                )

            if mistakes <= error_threshold:
                stopped_reason = "error_threshold_reached"
                break

        # Stopping criterion 3: early stopping
        elif stopping_criterion == "early_stopping":
            if patience is None:
                raise ValueError(
                    "patience must be provided for early_stopping."
                )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                stopped_reason = "early_stopping"
                break

        else:
            raise ValueError(
                "Unknown stopping criterion: "
                + stopping_criterion
            )

    return {
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
        "epochs_run": len(train_acc_history),
        "stopped_reason": stopped_reason
    }


def evaluate_model(model, X, y, class_names=None):
    """
    Evaluate a trained MultiClassPerceptron on a dataset.
    """

    predictions = model.predict_batch(X)

    accuracy = accuracy_score(y, predictions)

    precision = precision_score(
        y,
        predictions,
        average=None,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        average=None,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        average=None,
        zero_division=0
    )

    cm = confusion_matrix(y, predictions)

    return {
        "accuracy": accuracy,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "confusion_matrix": cm
    }


def grid_search(X_train, y_train, X_val, y_val,
                learning_rates, normalisation_methods,
                n_features, n_classes=10, **train_kwargs):
    """
    Try every combination of learning_rate x normalisation_method,
    train a model for each, and record validation accuracy.
    """

    from perceptron import MultiClassPerceptron
    from data_utils import normalise

    results = []

    for normalisation_method in normalisation_methods:

        # Normalise training data
        X_train_norm, stats = normalise(
            X_train,
            method=normalisation_method
        )

        # Normalise validation data using the same training statistics
        X_val_norm, _ = normalise(
            X_val,
            method=normalisation_method,
            stats=stats
        )

        for learning_rate in learning_rates:

            model = MultiClassPerceptron(
                n_features=n_features,
                n_classes=n_classes,
                learning_rate=learning_rate
            )

            history = train_model(
                model,
                X_train_norm,
                y_train,
                X_val_norm,
                y_val,
                **train_kwargs
            )

            val_accuracy = history["val_acc_history"][-1]

            results.append({
                "learning_rate": learning_rate,
                "normalisation_method": normalisation_method,
                "val_accuracy": val_accuracy,
                "train_acc_history": history["train_acc_history"],
                "val_acc_history": history["val_acc_history"]
            })

    return results