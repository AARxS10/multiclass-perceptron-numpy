import numpy as np

from data_utils import (
    load_dataset,
    flatten_images,
    train_val_split,
    build_augmented_trainset,
    augment_brightness,
    augment_rotation
)

from train_eval import train_model, evaluate_model
from perceptron import MultiClassPerceptron


# Load dataset
X, y = load_dataset("DATA/rgb-train/train", mode="RGB")

# Train/validation split
X_train, y_train, X_val, y_val = train_val_split(
    X,
    y,
    val_fraction=0.2,
    random_state=42
)

# Keep RAW training images for augmentation
X_train_raw = X_train.copy()

# Flatten validation images
X_val = flatten_images(X_val)


def run_experiment(name, X_train_raw, y_train):
    """
    Train one model using the best baseline settings.
    """

    print()
    print("EXPERIMENT:", name)
    print("============================")

    # Flatten training images
    X_train = flatten_images(X_train_raw)

    # Z-score normalisation using training statistics
    from data_utils import normalise

    X_train, stats = normalise(
        X_train,
        method="zscore"
    )

    X_val_norm, _ = normalise(
        X_val,
        method="zscore",
        stats=stats
    )

    # Create model
    model = MultiClassPerceptron(
        n_features=X_train.shape[1],
        n_classes=10,
        learning_rate=0.01
    )

    # Train
    history = train_model(
        model,
        X_train,
        y_train,
        X_val_norm,
        y_val,
        stopping_criterion="fixed_epochs",
        max_epochs=10,
        random_state=42
    )

    # Evaluate
    evaluation = evaluate_model(
        model,
        X_val_norm,
        y_val
    )

    print("Epochs:", history["epochs_run"])
    print("Validation accuracy:", evaluation["accuracy"])

    return evaluation["accuracy"]


# --------------------------------------------------
# 1. Baseline
# --------------------------------------------------

baseline_accuracy = run_experiment(
    "No augmentation",
    X_train_raw,
    y_train
)


# --------------------------------------------------
# 2. Brightness augmentation
# --------------------------------------------------

X_brightness, y_brightness = build_augmented_trainset(
    X_train_raw,
    y_train,
    augment_brightness,
    rng=np.random.default_rng(42)
)

brightness_accuracy = run_experiment(
    "Brightness augmentation",
    X_brightness,
    y_brightness
)


# --------------------------------------------------
# 3. Rotation augmentation
# --------------------------------------------------

X_rotation, y_rotation = build_augmented_trainset(
    X_train_raw,
    y_train,
    augment_rotation,
    rng=np.random.default_rng(42)
)

rotation_accuracy = run_experiment(
    "Rotation augmentation",
    X_rotation,
    y_rotation
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("AUGMENTATION RESULTS")
print("====================")

print("No augmentation:", baseline_accuracy)
print("Brightness:", brightness_accuracy)
print("Rotation:", rotation_accuracy)