"""On-the-fly data augmentation: small random rotations of the mosaic images.

Matches the thesis augmentation (rotation by -10, -5, 5 or 10 degrees, values
clipped to [0, 1]). For two-input fusion models the same rotation is applied to
the MRI and PET images of a subject so they stay aligned.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import keras
import numpy as np
from scipy import ndimage


def to_model_input(inputs: Dict[str, np.ndarray]):
    """Single-input models take an array; two-branch models take (mri, pet)."""
    return inputs["image"] if "image" in inputs else (inputs["mri"], inputs["pet"])


class MosaicSequence(keras.utils.PyDataset):
    """Batches of (inputs, labels) with optional random rotation."""

    def __init__(self, inputs: Dict[str, np.ndarray], y: np.ndarray, batch_size: int = 8,
                 angles: Optional[Sequence[float]] = None, shuffle: bool = True, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.inputs, self.y = inputs, y
        self.batch_size, self.angles, self.shuffle = batch_size, list(angles or []), shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(y))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.y) / self.batch_size))

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.order)

    def _rotate(self, img: np.ndarray, angle: float) -> np.ndarray:
        out = ndimage.rotate(img, angle, axes=(0, 1), reshape=False, order=1)
        return np.clip(out, 0.0, 1.0)

    def __getitem__(self, i: int):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        batch = {k: v[idx].copy() for k, v in self.inputs.items()}
        if self.angles:
            for j in range(len(idx)):
                angle = float(self.rng.choice(self.angles))
                for k in batch:
                    batch[k][j] = self._rotate(batch[k][j], angle)
        return to_model_input(batch), self.y[idx]
