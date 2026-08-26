"""
data_utils.py

Loading, splitting, normalising, and augmenting the Simpsons-MNIST data.
"""

import numpy as np
from PIL import Image
import os


CLASS_NAMES = {
    0: "bart_simpson",
    1: "charles_montgomery_burns",
    2: "homer_simpson",
    3: "krusty_the_clown",
    4: "lisa_simpson",
    5: "marge_simpson",
    6: "milhouse_van_houten",
    7: "moe_szyslak",
    8: "ned_flanders",
    9: "principal_skinner",
}


def load_dataset(root_dir, mode="RGB"):
    """
    Load all images from a directory of class-labeled subfolders.

    Expected structure: root_dir/<class_name>/*.jpg

    Args:
        root_dir: path to e.g. "data/rgb-train" or "data/grayscale-test"
        mode: "RGB" or "L" (PIL's grayscale mode)

    Returns:
        X: np.ndarray, shape (N, H, W, 3) for RGB or (N, H, W) for grayscale
        y: np.ndarray, shape (N,) — integer labels matching CLASS_NAMES
    """
    images = []
    labels = []

    name_to_label = {name: label for label, name in CLASS_NAMES.items()}

    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        label = name_to_label[class_name]

        for filename in os.listdir(class_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(class_path, filename)

                with Image.open(image_path) as img:
                    img = img.convert(mode)
                    images.append(np.array(img))
                    labels.append(label)

    X = np.stack(images)
    y = np.array(labels, dtype=np.int64)

    return X, y


def flatten_images(X):
    """
    Flatten an (N, H, W) or (N, H, W, 3) array into (N, H*W) or (N, H*W*3).
    """
    return X.reshape(X.shape[0], -1)


def train_val_split(X, y, val_fraction=0.2, random_state=None):
    """
    Split (X, y) into train and validation sets. Should shuffle before
    splitting, and ideally stratify by class so val set has roughly
    equal representation of all 10 characters.

    Args:
        X, y: full training data
        val_fraction: fraction to hold out for validation
        random_state: seed for reproducible splitting

    Returns:
        X_train, y_train, X_val, y_val
    """
    rng = np.random.default_rng(random_state)

    train_indices = []
    val_indices = []

    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)

        n_val = int(len(indices) * val_fraction)

        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    return X_train, y_train, X_val, y_val


def normalise(X, method="none", stats=None):
    """
    Normalise a flattened image array.

    Args:
        X: array of raw pixel values (0-255), shape (N, D)
        method: one of "none", "minmax", "zscore"
        stats: previously calculated statistics for z-score
               normalisation. Should come from X_train.

    Returns:
        normalised array, stats
    """

    X = X.astype(float)

    if method == "none":
        return X, None

    elif method == "minmax":
        return X / 255.0, None

    elif method == "zscore":

        if stats is None:
            mean = X.mean(axis=0)
            std = X.std(axis=0)

            std[std == 0] = 1.0

            stats = {
                "mean": mean,
                "std": std
            }

        mean = stats["mean"]
        std = stats["std"]

        X_normalised = (X - mean) / std

        return X_normalised, stats

    else:
        raise ValueError(
            "method must be 'none', 'minmax', or 'zscore'"
        )


def augment_brightness(img, low=0.7, high=1.3, rng=None):
    """
    Apply random brightness jitter to a single RAW (unflattened,
    unnormalised) image array.

    Args:
        img: np.ndarray, raw pixel values 0-255, shape (H, W) or (H, W, 3)
        low, high: bounds for the uniform random scale factor
        rng: np.random.Generator for reproducible sampling

    Returns:
        augmented image, same shape and dtype (uint8) as input
    """
    if rng is None:
        rng = np.random.default_rng()

    b = rng.uniform(low, high)

    img_aug = np.clip(
        img.astype(float) * b,
        0,
        255
    ).astype(np.uint8)

    return img_aug


def augment_rotation(img, max_angle=15, rng=None):
    """
    Apply a small random rotation to a single RAW image array, using PIL.

    Args:
        img: np.ndarray, raw pixel values 0-255
        max_angle: rotate by a random angle in [-max_angle, max_angle]
        rng: np.random.Generator for reproducible sampling

    Returns:
        rotated image as np.ndarray, same shape/dtype as input
    """
    if rng is None:
        rng = np.random.default_rng()

    angle = rng.uniform(-max_angle, max_angle)

    rotated = Image.fromarray(img).rotate(angle)

    return np.array(rotated)


def build_augmented_trainset(X_train, y_train, augment_fn, rng=None):
    """
    Apply augment_fn to every image in X_train (raw, unflattened),
    and return the ORIGINAL + AUGMENTED data concatenated together.

    Args:
        X_train: raw training images (unflattened), shape (N, H, W, [3])
        y_train: corresponding labels, shape (N,)
        augment_fn: augment_brightness or augment_rotation
        rng: np.random.Generator for reproducible sampling

    Returns:
        X_combined: shape (2N, H, W, [3])
        y_combined: shape (2N,)
    """
    if rng is None:
        rng = np.random.default_rng()

    augmented_images = []

    for img in X_train:
        augmented_images.append(augment_fn(img, rng=rng))

    X_augmented = np.stack(augmented_images)
    y_augmented = y_train.copy()

    X_combined = np.concatenate([X_train, X_augmented], axis=0)
    y_combined = np.concatenate([y_train, y_augmented], axis=0)

    return X_combined, y_combined

    