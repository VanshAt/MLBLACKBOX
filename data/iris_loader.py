"""
iris_loader.py — Iris dataset loader for MLBlackBox.

Loads the Iris CSV and prepares it for training:
  - 4 features (sepal length, sepal width, petal length, petal width)
  - 3 classes (Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2)
  - 150 samples

For binary classification (setosa vs. rest), use get_iris_binary().
For multi-class, use get_iris_multiclass().

Features are normalized to [0, 1] range by default.
"""

import csv
import os
import random
from typing import List, Tuple, Optional


IRIS_CSV_PATH = os.path.join(os.path.dirname(__file__), "iris.csv")

# Class label mapping
LABEL_MAP = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}


def load_iris_raw() -> Tuple[List[List[float]], List[int]]:
    """
    Load raw Iris data from CSV.

    Returns:
        X: list of 150 feature vectors (4 floats each)
        y: list of 150 integer class labels (0, 1, or 2)
    """
    if not os.path.exists(IRIS_CSV_PATH):
        raise FileNotFoundError(
            f"Iris CSV not found at '{IRIS_CSV_PATH}'. "
            "Download from: https://archive.ics.uci.edu/ml/datasets/iris"
        )

    X, y = [], []
    with open(IRIS_CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5 or row[0].strip().lower() == "sepal_length":
                continue  # skip header
            try:
                features = [float(row[i]) for i in range(4)]
                label_str = row[4].strip()
                label = LABEL_MAP.get(label_str, -1)
                if label >= 0:
                    X.append(features)
                    y.append(label)
            except (ValueError, IndexError):
                continue

    return X, y


def normalize(X: List[List[float]]) -> List[List[float]]:
    """
    Min-max normalize each feature column to [0, 1].

    Args:
        X: list of feature vectors (all same length)

    Returns:
        normalized X
    """
    if not X:
        return X

    n_features = len(X[0])
    mins = [min(row[j] for row in X) for j in range(n_features)]
    maxs = [max(row[j] for row in X) for j in range(n_features)]

    normalized = []
    for row in X:
        norm_row = []
        for j in range(n_features):
            denom = maxs[j] - mins[j]
            val = (row[j] - mins[j]) / denom if denom > 0 else 0.0
            norm_row.append(val)
        normalized.append(norm_row)

    return normalized


def one_hot(label: int, n_classes: int) -> List[float]:
    """Convert integer label to one-hot vector."""
    vec = [0.0] * n_classes
    vec[label] = 1.0
    return vec


def get_iris_binary(
    normalize_features: bool = True,
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Return Iris data as binary classification: setosa (0) vs. rest (1).
    Output neurons: 1 (use Sigmoid activation + BCE loss).

    Returns:
        X: list of normalized feature vectors
        y: list of single-element lists [[0.0], [1.0], ...]
    """
    X_raw, y_raw = load_iris_raw()

    if normalize_features:
        X_raw = normalize(X_raw)

    y_binary = [[0.0] if label == 0 else [1.0] for label in y_raw]

    if shuffle:
        combined = list(zip(X_raw, y_binary))
        if seed is not None:
            random.seed(seed)
        random.shuffle(combined)
        X_raw, y_binary = zip(*combined)
        X_raw, y_binary = list(X_raw), list(y_binary)

    return X_raw, y_binary


def get_iris_multiclass(
    normalize_features: bool = True,
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Return Iris data as 3-class one-hot classification.
    Output neurons: 3 (use Linear/Softmax on output, or compare max output).

    Returns:
        X: list of normalized feature vectors
        y: list of one-hot label vectors [[1,0,0], [0,1,0], [0,0,1], ...]
    """
    X_raw, y_raw = load_iris_raw()

    if normalize_features:
        X_raw = normalize(X_raw)

    y_onehot = [one_hot(label, 3) for label in y_raw]

    if shuffle:
        combined = list(zip(X_raw, y_onehot))
        if seed is not None:
            random.seed(seed)
        random.shuffle(combined)
        X_raw, y_onehot = zip(*combined)
        X_raw, y_onehot = list(X_raw), list(y_onehot)

    return X_raw, y_onehot


def train_test_split(
    X: List[List[float]],
    y: List[List[float]],
    test_ratio: float = 0.2,
    seed: Optional[int] = 42,
) -> Tuple[List, List, List, List]:
    """
    Split X, y into train and test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    combined = list(zip(X, y))
    if seed is not None:
        random.seed(seed)
    random.shuffle(combined)

    split = int(len(combined) * (1 - test_ratio))
    train = combined[:split]
    test = combined[split:]

    X_train = [r[0] for r in train]
    y_train = [r[1] for r in train]
    X_test = [r[0] for r in test]
    y_test = [r[1] for r in test]

    return X_train, X_test, y_train, y_test
