import numpy as np

from data_utils import load_dataset, flatten_images, normalise
from perceptron import MultiClassPerceptron


# Load 200 real Simpsons images
X, y = load_dataset("DATA/rgb-train/train", mode="RGB")

rng = np.random.default_rng(42)

indices = rng.choice(len(X), size=200, replace=False)

X_small = flatten_images(X[indices])
y_small = y[indices]

# Min-max normalisation
X_small = normalise(X_small, method="minmax")


# Create a fresh model
model = MultiClassPerceptron(
    n_features=X_small.shape[1],
    n_classes=10,
    learning_rate=1.0
)

print("Initial weight sum:", np.sum(model.weights))
print("Class distribution:", np.bincount(y_small))


# Train for 5 epochs
mistakes = []

for epoch in range(5):
    m = model.train_epoch(X_small, y_small)
    mistakes.append(m)

print("Mistakes:", mistakes)

# Evaluate training accuracy
predictions = model.predict_batch(X_small)
accuracy = np.mean(predictions == y_small)

print("Accuracy:", accuracy)