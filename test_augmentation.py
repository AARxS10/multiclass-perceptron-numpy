import numpy as np

from data_utils import (
    load_dataset,
    augment_brightness,
    augment_rotation,
    build_augmented_trainset
)


X, y = load_dataset("DATA/rgb-train/train", mode="RGB")

rng = np.random.default_rng(42)

img = X[0]

bright = augment_brightness(img, rng=rng)
rotated = augment_rotation(img, rng=rng)

print("Original shape:", img.shape)
print("Brightness shape:", bright.shape)
print("Rotation shape:", rotated.shape)

print("Original dtype:", img.dtype)
print("Brightness dtype:", bright.dtype)
print("Rotation dtype:", rotated.dtype)

print("Brightness range:", bright.min(), bright.max())
print("Rotation range:", rotated.min(), rotated.max())


X_small = X[:100]
y_small = y[:100]

X_aug, y_aug = build_augmented_trainset(
    X_small,
    y_small,
    augment_brightness,
    rng=np.random.default_rng(42)
)

print("Original dataset shape:", X_small.shape)
print("Augmented dataset shape:", X_aug.shape)
print("Original labels shape:", y_small.shape)
print("Augmented labels shape:", y_aug.shape)
