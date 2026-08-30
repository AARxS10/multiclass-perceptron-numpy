

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_model(
    model,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    max_epochs=10,
    stopping_criterion="fixed",
    patience=3,
    random_state=42
):
    """
    Train a multiclass perceptron.

    stopping_criterion:
        "fixed"          -> train for max_epochs
        "early_stopping" -> stop when validation accuracy
                            stops improving for patience epochs
    """

    rng = np.random.default_rng(random_state)

    train_acc_history = []
    val_acc_history = []
    mistakes_history = []

    best_val_accuracy = -np.inf
    epochs_without_improvement = 0
    stopped_reason = "max_epochs"

    for epoch in range(max_epochs):

        mistakes = model.train_epoch(
            X_train,
            y_train,
            rng=rng
        )

        train_predictions = model.predict(X_train)

        train_accuracy = accuracy_score(
            y_train,
            train_predictions
        )

        train_acc_history.append(train_accuracy)
        mistakes_history.append(mistakes)

        if X_val is not None and y_val is not None:

            val_predictions = model.predict(X_val)

            val_accuracy = accuracy_score(
                y_val,
                val_predictions
            )

            val_acc_history.append(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if (
            stopping_criterion == "early_stopping"
            and X_val is not None
            and y_val is not None
            and epochs_without_improvement >= patience
        ):
            stopped_reason = "early_stopping"
            break

    history = {
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
        "mistakes_history": mistakes_history,

        "training_accuracy": train_acc_history,
        "validation_accuracy": val_acc_history,
        "mistakes": mistakes_history,

        "epochs_run": len(train_acc_history),
        "stopped_reason": stopped_reason
    }

    return history


def evaluate_model(model, X, y, class_names=None, n_classes=10):
    """
    Evaluate a trained multiclass perceptron.
    """

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        labels=list(range(n_classes)),
        average=None,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=list(range(n_classes))
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,

        # Names expected by the test/evaluation scripts
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1
    }


def grid_search(
    X_train,
    y_train,
    X_val,
    y_val,
    n_features,
    n_classes=10,
    learning_rates=None,
    normalisation_methods=None,
    max_epochs=10,
    stopping_criterion="fixed",
    patience=3,
    random_state=42
):
    """
    Grid search over learning rate and normalisation.
    """

    if learning_rates is None:
        learning_rates = [0.01, 0.1, 1.0]

    if normalisation_methods is None:
        normalisation_methods = [
            "none",
            "minmax",
            "zscore"
        ]

    from perceptron import MultiClassPerceptron
    from data_utils import normalise

    results = []

    for method in normalisation_methods:

        X_train_norm, stats = normalise(
            X_train,
            method=method
        )

        X_val_norm, _ = normalise(
            X_val,
            method=method,
            stats=stats
        )

        for learning_rate in learning_rates:

            print(
                f"Learning rate: {learning_rate} | "
                f"Normalisation: {method}"
            )

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
                max_epochs=max_epochs,
                stopping_criterion=stopping_criterion,
                patience=patience,
                random_state=random_state
            )

            evaluation = evaluate_model(
                model,
                X_val_norm,
                y_val,
                n_classes=n_classes
            )

            results.append({
                "learning_rate": learning_rate,
                "normalisation_method": method,
                "val_accuracy": evaluation["accuracy"],
                "validation_accuracy": evaluation["accuracy"],
                "history": history,
                "model": model,
                "stats": stats
            })

    return results


