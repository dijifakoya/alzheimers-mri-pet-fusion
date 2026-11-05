"""Intensity normalisation and shape checks for MRI and PET volumes.

Normalisation is applied to each volume on its own, so no information leaks
between training and test data.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage


def normalise_clip(volume: np.ndarray, vmin: float = 0.0, vmax: float = 400.0) -> np.ndarray:
    """Clip intensities to [vmin, vmax] and rescale to [0, 1] (used for MRI in the thesis)."""
    v = np.clip(volume.astype(np.float32), vmin, vmax)
    return (v - vmin) / (vmax - vmin)


def normalise_minmax(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Rescale a single volume to [0, 1] using its own minimum and maximum (used for PET)."""
    v = volume.astype(np.float32)
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / max(hi - lo, eps)


def normalise_zscore(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-variance scaling over non-zero (brain) voxels, then min-max to [0, 1]."""
    v = volume.astype(np.float32)
    mask = v > 0
    mean = v[mask].mean() if mask.any() else v.mean()
    std = v[mask].std() if mask.any() else v.std()
    return normalise_minmax((v - mean) / max(std, eps))


NORMALISERS = {"clip": normalise_clip, "minmax": normalise_minmax, "zscore": normalise_zscore}


def normalise(volume: np.ndarray, method: str = "minmax", **kwargs) -> np.ndarray:
    """Dispatch to a normalisation method by name (``clip``, ``minmax`` or ``zscore``)."""
    if method not in NORMALISERS:
        raise ValueError(f"Unknown normalisation '{method}'. Choose from {list(NORMALISERS)}")
    return NORMALISERS[method](volume, **kwargs)


def ensure_shape(volume: np.ndarray, shape: Sequence[int], resample: bool = False) -> np.ndarray:
    """Check a volume has the expected shape; optionally resample it with linear interpolation.

    The thesis used volumes already registered to MNI152 2 mm space (91 x 109 x 91).
    """
    shape = tuple(shape)
    if volume.shape == shape:
        return volume
    if not resample:
        raise ValueError(
            f"Volume shape {volume.shape} differs from expected {shape}. "
            "Register scans to the template first, or set data.resample_if_needed: true."
        )
    factors = [t / s for t, s in zip(shape, volume.shape)]
    return ndimage.zoom(volume, factors, order=1).astype(np.float32)
