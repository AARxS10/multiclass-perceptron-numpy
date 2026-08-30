import numpy as np

from data_utils import load_dataset, flatten_images, normalise, train_val_split
from perceptron import MultiClassPerceptron
from train_eval import train_model, evaluate_model


# Load the dataset
X, y = load_dataset("DATA/rgb-train/train", mode="RGB")

# Use a manageable subset for testing
rng = np.random.default_rng(42)
indices = rng.choice(len(X), size=1000, replace=False)

X = X[indices]
y = y[indices]

# Split into training and validation sets
X_train, y_train, X_val, y_val = train_val_split(
    X, y, val_fraction=0.2, random_state=42
)

# Flatten
X_train = flatten_images(X_train)
X_val = flatten_images(X_val)

# Min-max normalisation
X_train, _ = normalise(X_train, method="minmax")
X_val, _ = normalise(X_val, method="minmax")

# Create a fresh model
model = MultiClassPerceptron(
    n_features=X_train.shape[1],
    n_classes=10,
    learning_rate=1.0
)

# Train
history = train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    stopping_criterion="early_stopping",
    max_epochs=50,
    patience=5,
    random_state=42
)

print("Epochs run:", history["epochs_run"])
print("Stopped reason:", history["stopped_reason"])
print("Training accuracy:", history["train_acc_history"])
print("Validation accuracy:", history["val_acc_history"])

results = evaluate_model(model, X_val, y_val)

print("Evaluation accuracy:", results["accuracy"])
print("Precision per class:", results["precision_per_class"])
print("Recall per class:", results["recall_per_class"])
print("F1 per class:", results["f1_per_class"])
print("Confusion matrix:")
print(results["confusion_matrix"])