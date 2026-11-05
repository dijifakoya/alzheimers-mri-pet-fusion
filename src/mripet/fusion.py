"""Image-level fusion of cast MRI and PET mosaics.

Three fusion levels are supported across the package:

* ``image``    – spatial-domain (pixel-level) fusion: MRI and PET mosaics are combined into one
                 image before the CNN (this module). This is the method that gave 94% accuracy in the paper.
* ``feature``  – each modality has its own CNN branch; features are concatenated (see models.py).
* ``decision`` – each branch predicts separately and the probabilities are averaged (see models.py).
"""
from __future__ import annotations

import numpy as np


METHODS = ("six_channel", "concat", "sum", "channels", "average", "weighted")


def fuse_images(mri: np.ndarray, pet: np.ndarray, method: str = "six_channel", alpha: float = 0.5) -> np.ndarray:
    """Fuse MRI and PET mosaics of identical shape (..., H, W, 1) at pixel level.

    Methods:
        six_channel: the paper's method — the 3-channel MRI image and 3-channel PET image are
                     stacked into one 6-channel image (channels: MRI, MRI, MRI, PET, PET, PET).
        concat:      the same information in 2 channels [MRI, PET] (lighter on memory).
        sum:         single channel ``MRI + PET`` (equation 6 in the paper).
        channels:    3 channels [MRI, PET, mean].
        average:     single channel ``(MRI + PET) / 2``.
        weighted:    single channel ``alpha * MRI + (1 - alpha) * PET``.
    """
    if mri.shape != pet.shape:
        raise ValueError(f"MRI {mri.shape} and PET {pet.shape} mosaics must have the same shape")
    if method == "six_channel":
        return np.concatenate([mri, mri, mri, pet, pet, pet], axis=-1).astype(np.float32)
    if method == "concat":
        return np.concatenate([mri, pet], axis=-1).astype(np.float32)
    if method == "sum":
        return (mri + pet).astype(np.float32)
    if method == "channels":
        return np.concatenate([mri, pet, (mri + pet) / 2.0], axis=-1).astype(np.float32)
    if method == "average":
        return ((mri + pet) / 2.0).astype(np.float32)
    if method == "weighted":
        return (alpha * mri + (1.0 - alpha) * pet).astype(np.float32)
    raise ValueError(f"method must be one of {METHODS}")
