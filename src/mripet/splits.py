"""Subject-level, class-stratified data splits.

All scans from one subject are kept in the same split, which prevents the data
leakage that inflates accuracy when a person's scans appear in both training and
test sets.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def _one_split(y: np.ndarray, groups: np.ndarray, fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n_splits = max(2, int(round(1.0 / fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, held = next(sgkf.split(np.zeros(len(y)), y, groups))
    return rest, held


def train_val_test_split(y: np.ndarray, groups: np.ndarray, test_fraction: float = 0.15,
                         val_fraction: float = 0.15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return index arrays (train, val, test) with no subject in more than one set."""
    rest, test = _one_split(y, groups, test_fraction, seed)
    val_share = val_fraction / (1.0 - test_fraction)
    tr, va = _one_split(y[rest], groups[rest], val_share, seed + 1)
    train, val = rest[tr], rest[va]
    assert not (set(groups[train]) & set(groups[test])), "subject leakage between train and test"
    assert not (set(groups[train]) & set(groups[val])), "subject leakage between train and val"
    return train, val, test


def cv_splits(y: np.ndarray, groups: np.ndarray, folds: int = 5, val_fraction: float = 0.15,
              seed: int = 42) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (train, val, test) indices for stratified, subject-level k-fold cross-validation."""
    sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (rest, test) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        tr, va = _one_split(y[rest], groups[rest], val_fraction, seed + fold + 1)
        yield rest[tr], rest[va], test
