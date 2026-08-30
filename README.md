# CSC2042S Assignment 1 — RHMABD006

Multi-class perceptron image classification on the Simpsons-MNIST
dataset, implemented from scratch using NumPy (one-vs-rest, 10 binary
perceptrons).

## Setup

1. Requires Python 3.x with the following packages:

   ```
   pip install numpy pillow scikit-learn matplotlib pandas
   ```

2. Place the dataset in a `DATA/` folder alongside the notebook and
   `.py` files, with the following structure:

   ```
   DATA/
   ├── rgb-train/train/<class folders>/*.jpg     (RGB training, 8000 images)
   ├── test/<class folders>/*.jpg                (RGB test, 2000 images)
   ├── train/<class folders>/*.jpg               (grayscale training, 8000 images)
   └── grayscale-test/test/<class folders>/*.jpg (grayscale test, 2000 images)
   ```

   Each `<class folders>` set is one of: `bart_simpson`,
   `charles_montgomery_burns`, `homer_simpson`, `krusty_the_clown`,
   `lisa_simpson`, `marge_simpson`, `milhouse_van_houten`,
   `moe_szyslak`, `ned_flanders`, `principal_skinner`.

   Data is not included in this submission, per the assignment
   instructions.

3. Open `RHMABD006_formatted.ipynb` in Jupyter and run all cells from
   top to bottom (Kernel → Restart & Run All is recommended to confirm
   a clean run).

## Files

- **`RHMABD006_formatted.ipynb`** — main notebook. Runs and documents
  all tasks (1–7): data loading and splitting, perceptron
  demonstration and sanity checks, training with two stopping
  criteria, hyperparameter grid search, final evaluation on validation
  and test sets, RGB vs grayscale analysis, and the data augmentation
  experiment.
- **`perceptron.py`** — `BinaryPerceptron` and `MultiClassPerceptron`
  classes. Implements the one-vs-rest architecture from scratch using
  only NumPy: `MultiClassPerceptron` holds 10 independent
  `BinaryPerceptron` instances (weights and bias initialised to 0.1),
  trained via per-class binary relabelling, with prediction by argmax
  over the 10 raw perceptron scores.
- **`data_utils.py`** — data loading (via Pillow), flattening,
  stratified train/validation splitting, normalisation (none,
  min-max, z-score, with train-set statistics reused on validation and
  test data), and the two augmentation functions (brightness jitter,
  small random rotation) used in Task 7.
- **`train_eval.py`** — the training loop (supporting fixed-epoch and
  early-stopping criteria), evaluation utilities (accuracy, per-class
  precision/recall/F1, and confusion matrix via scikit-learn), and the
  hyperparameter grid search function.
- **`report.pdf`** — written report (max 5 pages) covering design
  decisions, results, and discussion for all tasks, with supporting
  tables.
- **`README.md`** — this file.

## Reproducibility

Fixed random seeds are used throughout (train/validation splitting,
weight-order shuffling during training, and the augmentation
experiment's five seeds) so that results can be reproduced by
re-running the notebook against the same data.
