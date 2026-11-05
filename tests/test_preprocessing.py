import numpy as np
import pytest

from mripet.fusion import fuse_images
from mripet.preprocessing import ensure_shape, normalise
from mripet.splits import cv_splits, train_val_test_split


def test_normalisers_range():
    vol = np.random.default_rng(0).normal(200, 150, (10, 12, 10))
    for method in ("clip", "minmax", "zscore"):
        out = normalise(vol, method)
        assert out.min() >= 0 and out.max() <= 1 and out.dtype == np.float32


def test_ensure_shape():
    vol = np.zeros((45, 54, 45), dtype=np.float32)
    with pytest.raises(ValueError):
        ensure_shape(vol, (91, 109, 91))
    assert ensure_shape(vol, (91, 109, 91), resample=True).shape == (91, 109, 91)


def test_fusion_methods():
    a = np.ones((2, 8, 8, 1), np.float32)
    b = np.zeros((2, 8, 8, 1), np.float32)
    assert fuse_images(a, b, "six_channel").shape == (2, 8, 8, 6)
    assert fuse_images(a, b, "concat").shape == (2, 8, 8, 2)
    assert np.allclose(fuse_images(a, b, "sum"), 1.0)
    assert fuse_images(a, b, "channels").shape == (2, 8, 8, 3)
    assert np.allclose(fuse_images(a, b, "average"), 0.5)


def test_no_subject_leakage():
    groups = np.repeat(np.arange(40), 2)          # two scans per subject
    y = (np.arange(40) % 2).repeat(2).astype(float)
    tr, va, te = train_val_test_split(y, groups, 0.2, 0.2, seed=1)
    assert not set(groups[tr]) & set(groups[te])
    assert not set(groups[va]) & set(groups[te])
    for tr, va, te in cv_splits(y, groups, folds=4, seed=1):
        assert not set(groups[tr]) & set(groups[te])
