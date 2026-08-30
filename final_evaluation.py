import numpy as np

from data_utils import (
    load_dataset,
    flatten_images,
    train_val_split,
    normalise
)

from perceptron import MultiClassPerceptron
from train_eval import train_model, evaluate_model


# ============================================================
# SETTINGS
# ============================================================

LEARNING_RATE = 0.01
MAX_EPOCHS = 10
RANDOM_STATE = 42

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
# RGB MODEL
# ============================================================

print("=" * 60)
print("FINAL RGB MODEL")
print("=" * 60)

X_rgb, y_rgb = load_dataset(
    "DATA/rgb-train/train",
    mode="RGB"
)

X_rgb_train, y_rgb_train, X_rgb_val, y_rgb_val = train_val_split(
    X_rgb,
    y_rgb,
    val_fraction=0.2,
    random_state=RANDOM_STATE
)

X_rgb_train = flatten_images(X_rgb_train)
X_rgb_val = flatten_images(X_rgb_val)

X_rgb_train, rgb_stats = normalise(
    X_rgb_train,
    "zscore"
)

X_rgb_val, _ = normalise(
    X_rgb_val,
    "zscore",
    rgb_stats
)

rgb_model = MultiClassPerceptron(
    n_features=X_rgb_train.shape[1],
    n_classes=10,
    learning_rate=LEARNING_RATE
)

rgb_result = train_model(
    rgb_model,
    X_rgb_train,
    y_rgb_train,
    X_rgb_val,
    y_rgb_val,
    stopping_criterion="fixed_epochs",
    max_epochs=MAX_EPOCHS,
    random_state=RANDOM_STATE
)

print("Training complete.")
print("Validation accuracy:",
      rgb_result["val_acc_history"][-1])


# ============================================================
# RGB TEST SET
# ============================================================

X_rgb_test, y_rgb_test = load_dataset(
    "DATA/test",
    mode="RGB"
)

X_rgb_test = flatten_images(X_rgb_test)

X_rgb_test, _ = normalise(
    X_rgb_test,
    "zscore",
    rgb_stats
)

rgb_test_result = evaluate_model(
    rgb_model,
    X_rgb_test,
    y_rgb_test,
    CLASS_NAMES
)

print()
print("RGB TEST RESULTS")
print("-" * 60)

print("Accuracy:",
      rgb_test_result["accuracy"])

print("Precision per class:")
print(rgb_test_result["precision_per_class"])

print("Recall per class:")
print(rgb_test_result["recall_per_class"])

print("F1 per class:")
print(rgb_test_result["f1_per_class"])

print("Confusion matrix:")
print(rgb_test_result["confusion_matrix"])


# ============================================================
# GRAYSCALE MODEL
# ============================================================

print()
print("=" * 60)
print("FINAL GRAYSCALE MODEL")
print("=" * 60)

X_gray, y_gray = load_dataset(
    "DATA/train",
    mode="L"
)

X_gray_train, y_gray_train, X_gray_val, y_gray_val = train_val_split(
    X_gray,
    y_gray,
    val_fraction=0.2,
    random_state=RANDOM_STATE
)

X_gray_train = flatten_images(X_gray_train)
X_gray_val = flatten_images(X_gray_val)

X_gray_train, gray_stats = normalise(
    X_gray_train,
    "zscore"
)

X_gray_val, _ = normalise(
    X_gray_val,
    "zscore",
    gray_stats
)

gray_model = MultiClassPerceptron(
    n_features=X_gray_train.shape[1],
    n_classes=10,
    learning_rate=LEARNING_RATE
)

gray_result = train_model(
    gray_model,
    X_gray_train,
    y_gray_train,
    X_gray_val,
    y_gray_val,
    stopping_criterion="fixed_epochs",
    max_epochs=MAX_EPOCHS,
    random_state=RANDOM_STATE
)

print("Training complete.")
print("Validation accuracy:",
      gray_result["val_acc_history"][-1])


# ============================================================
# GRAYSCALE TEST SET
# ============================================================

X_gray_test, y_gray_test = load_dataset(
    "DATA/test",
    mode="L"
)

X_gray_test = flatten_images(X_gray_test)

X_gray_test, _ = normalise(
    X_gray_test,
    "zscore",
    gray_stats
)

gray_test_result = evaluate_model(
    gray_model,
    X_gray_test,
    y_gray_test,
    CLASS_NAMES
)

print()
print("GRAYSCALE TEST RESULTS")
print("-" * 60)

print("Accuracy:",
      gray_test_result["accuracy"])

print("Precision per class:")
print(gray_test_result["precision_per_class"])

print("Recall per class:")
print(gray_test_result["recall_per_class"])

print("F1 per class:")
print(gray_test_result["f1_per_class"])

print("Confusion matrix:")
print(gray_test_result["confusion_matrix"])


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 60)
print("FINAL RGB VS GRAYSCALE COMPARISON")
print("=" * 60)

print(
    "RGB test accuracy:",
    rgb_test_result["accuracy"]
)

print(
    "Grayscale test accuracy:",
    gray_test_result["accuracy"]
)

difference = (
    rgb_test_result["accuracy"]
    - gray_test_result["accuracy"]
)

print(
    "Difference:",
    difference
)