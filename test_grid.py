import numpy as np

from data_utils import (
    load_dataset,
    flatten_images,
    train_val_split
)

from train_eval import grid_search


X, y = load_dataset("DATA/rgb-train/train", mode="RGB")

rng = np.random.default_rng(42)
indices = rng.choice(len(X), size=1000, replace=False)

X = X[indices]
y = y[indices]

X_train, y_train, X_val, y_val = train_val_split(
    X,
    y,
    val_fraction=0.2,
    random_state=42
)

X_train = flatten_images(X_train)
X_val = flatten_images(X_val)

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
    max_epochs=5,
    random_state=42
)

for result in results:
    print(
        "Learning rate:", result["learning_rate"],
        "Normalisation:", result["normalisation_method"],
        "Validation accuracy:", result["val_accuracy"]
    )
    