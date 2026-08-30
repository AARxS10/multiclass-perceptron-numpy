import numpy as np

from data_utils import (
    load_dataset,
    flatten_images,
    train_val_split,
    normalise
)

from perceptron import MultiClassPerceptron


# Load data
X, y = load_dataset("DATA/rgb-train/train", mode="RGB")

# Use a small subset for a quick test
rng = np.random.default_rng(42)
indices = rng.choice(len(X), size=1000, replace=False)

X = X[indices]
y = y[indices]

# Split
X_train, y_train, X_val, y_val = train_val_split(
    X,
    y,
    val_fraction=0.2,
    random_state=42
)

# Flatten
X_train = flatten_images(X_train)
X_val = flatten_images(X_val)

# Z-score using training statistics
X_train, stats = normalise(
    X_train,
    method="zscore"
)

X_val, _ = normalise(
    X_val,
    method="zscore",
    stats=stats
)


learning_rates = [0.001, 0.01, 0.1, 1.0]

for learning_rate in learning_rates:

    model = MultiClassPerceptron(
        n_features=X_train.shape[1],
        n_classes=10,
        learning_rate=learning_rate
    )

    print()
    print("Learning rate:", learning_rate)
    print("Initial weight sum:", model.weights.sum())

    for epoch in range(5):
        mistakes = model.train_epoch(
            X_train,
            y_train
        )

        predictions = model.predict_batch(X_train)
        accuracy = np.mean(predictions == y_train)

        print(
            "Epoch:", epoch + 1,
            "Mistakes:", mistakes,
            "Training accuracy:", accuracy
        )

    val_predictions = model.predict_batch(X_val)
    val_accuracy = np.mean(val_predictions == y_val)

    print("Validation accuracy:", val_accuracy)
    print("Final weight sum:", model.weights.sum())