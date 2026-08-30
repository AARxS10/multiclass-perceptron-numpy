import numpy as np

from data_utils import (
    load_dataset,
    flatten_images,
    train_val_split,
    normalise,
    augment_brightness,
    augment_rotation,
    build_augmented_trainset
)

from perceptron import MultiClassPerceptron
from train_eval import train_model, evaluate_model


# ============================================================
# SETTINGS
# ============================================================

LEARNING_RATE = 0.1
MAX_EPOCHS = 10

SEEDS = [42, 43, 44, 45, 46]

CLASS_NAMES = [
    "bart_simpson",
    "charles_montgomery_burns",
    "homer_simpson",
    "krusty_the_clown",
    "lisa_simpson",
    "marge_simpson",
    "milhouse_van_houten",
    "moe_szyslak",
    "ned_flanders",
    "principal_skinner"
]


# ============================================================
# LOAD RGB DATA
# ============================================================

print("Loading RGB training data...")

X, y = load_dataset(
    "DATA/rgb-train/train",
    mode="RGB"
)

X_train, y_train, X_val, y_val = train_val_split(
    X,
    y,
    val_fraction=0.2,
    random_state=42
)

print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)


# ============================================================
# LOAD TEST DATA
# ============================================================

X_test, y_test = load_dataset(
    "DATA/test",
    mode="RGB"
)

print("Test images:", X_test.shape)


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(condition, seed):

    print()
    print("=" * 60)
    print("Condition:", condition)
    print("Seed:", seed)
    print("=" * 60)

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Select training data
    # --------------------------------------------------------

    if condition == "none":

        X_exp = X_train.copy()
        y_exp = y_train.copy()

    elif condition == "brightness":

        X_exp, y_exp = build_augmented_trainset(
            X_train,
            y_train,
            augment_brightness,
            rng=rng
        )

    elif condition == "rotation":

        X_exp, y_exp = build_augmented_trainset(
            X_train,
            y_train,
            augment_rotation,
            rng=rng
        )

    else:
        raise ValueError("Unknown augmentation condition")


    print("Training dataset size:", X_exp.shape[0])


    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    X_exp = flatten_images(X_exp)
    X_val_flat = flatten_images(X_val)
    X_test_flat = flatten_images(X_test)


    # --------------------------------------------------------
    # Z-score normalisation
    #
    # IMPORTANT:
    # Statistics are calculated ONLY from the training data.
    # --------------------------------------------------------

    X_exp, stats = normalise(
        X_exp,
        "zscore"
    )

    X_val_flat, _ = normalise(
        X_val_flat,
        "zscore",
        stats
    )

    X_test_flat, _ = normalise(
        X_test_flat,
        "zscore",
        stats
    )


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MultiClassPerceptron(
        n_features=X_exp.shape[1],
        n_classes=10,
        learning_rate=LEARNING_RATE
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    result = train_model(
        model,
        X_exp,
        y_exp,
        X_val_flat,
        y_val,
        stopping_criterion="fixed",
        max_epochs=MAX_EPOCHS,
        random_state=seed
    )


    # --------------------------------------------------------
    # Evaluate on TEST SET
    # --------------------------------------------------------

    test_result = evaluate_model(
        model,
        X_test_flat,
        y_test,
        CLASS_NAMES
    )


    accuracy = test_result["accuracy"]

    print("Epochs:", result["epochs_run"])
    print(
          "Validation accuracy:",
          result["validation_accuracy"][-1]
    )
    print("Test accuracy:", accuracy)

    return accuracy


# ============================================================
# RUN ALL 15 EXPERIMENTS
# ============================================================

results = {
    "none": [],
    "brightness": [],
    "rotation": []
}


for condition in ["none", "brightness", "rotation"]:

    for seed in SEEDS:

        accuracy = run_experiment(
            condition,
            seed
        )

        results[condition].append(accuracy)


# ============================================================
# CALCULATE MEAN AND STANDARD DEVIATION
# ============================================================

print()
print()
print("=" * 60)
print("FINAL AUGMENTATION RESULTS")
print("=" * 60)


for condition in ["none", "brightness", "rotation"]:

    values = np.array(results[condition])

    mean = np.mean(values)
    std = np.std(values)

    print()
    print(condition.upper())
    print("Individual accuracies:", values)
    print("Mean test accuracy:", mean)
    print("Standard deviation:", std)


# ============================================================
# BEST CONDITION
# ============================================================

means = {
    condition: np.mean(results[condition])
    for condition in results
}

best_condition = max(
    means,
    key=means.get
)

print()
print("=" * 60)
print("BEST AUGMENTATION CONDITION")
print("=" * 60)

print("Condition:", best_condition)
print("Mean test accuracy:", means[best_condition])