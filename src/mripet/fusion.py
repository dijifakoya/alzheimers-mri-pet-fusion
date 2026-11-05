"""Image-level fusion of cast MRI and PET mosaics.

Three fusion levels are supported across the package:

* ``image``    – MRI and PET mosaics are combined into one image before the CNN (this module).
* ``feature``  – each modality has its own CNN branch; features are concatenated (see models.py).
* ``decision`` – each branch predicts separately and the probabilities are averaged (see models.py).
"""
from __future__ import annotations

import numpy as np


def fuse_images(mri: np.ndarray, pet: np.ndarray, method: str = "channels", alpha: float = 0.5) -> np.ndarray:
    """Fuse MRI and PET mosaics of identical shape (..., H, W, 1).

    Methods:
        channels: stack as 3 channels [MRI, PET, mean] – keeps both signals separate.
        average:  single channel ``(MRI + PET) / 2``.
        weighted: single channel ``alpha * MRI + (1 - alpha) * PET``.
    """
    if mri.shape != pet.shape:
        raise ValueError(f"MRI {mri.shape} and PET {pet.shape} mosaics must have the same shape")
    if method == "channels":
        return np.concatenate([mri, pet, (mri + pet) / 2.0], axis=-1).astype(np.float32)
    if method == "average":
        return ((mri + pet) / 2.0).astype(np.float32)
    if method == "weighted":
        return (alpha * mri + (1.0 - alpha) * pet).astype(np.float32)
    raise ValueError("method must be 'channels', 'average' or 'weighted'")
