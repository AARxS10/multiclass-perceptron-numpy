import numpy as np

from data_utils import (
    load_dataset,
    flatten_images,
    train_val_split
)

from train_eval import grid_search


# Load grayscale training data
X, y = load_dataset(
    "DATA/train",
    mode="L"
)

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)


# Train/validation split
X_train, y_train, X_val, y_val = train_val_split(
    X,
    y,
    val_fraction=0.2,
    random_state=42
)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)


# Flatten images
X_train = flatten_images(X_train)
X_val = flatten_images(X_val)

print("Flattened training shape:", X_train.shape)
print("Flattened validation shape:", X_val.shape)


# Grid search
results = grid_search(
    X_train,
    y_train,
    X_val,
    y_val,
    learning_rates=[0.01, 0.1, 1.0],
    normalisation_methods=["none", "minmax", "zscore"],
    n_features=X_train.shape[1],
    n_classes=10,
    stopping_criterion="fixed_epochs",
    max_epochs=10,
    random_state=42
)


# Print results
print()
print("GRAYSCALE GRID SEARCH RESULTS")
print("=============================")

for result in results:
    print(
        "Learning rate:", result["learning_rate"],
        "| Normalisation:", result["normalisation_method"],
        "| Validation accuracy:", result["val_accuracy"]
    )


# Best result
best_result = max(
    results,
    key=lambda result: result["val_accuracy"]
)

print()
print("BEST GRAYSCALE RESULT")
print("=====================")
print("Learning rate:", best_result["learning_rate"])
print("Normalisation:", best_result["normalisation_method"])
print("Validation accuracy:", best_result["val_accuracy"])