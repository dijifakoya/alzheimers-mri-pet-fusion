"""Image casting: turn a 3D brain volume into a single 2D grid ("mosaic") image.

An n x n grid holds n*n slices taken from one anatomical view (axial, coronal or
sagittal). Slices start at a ``top`` index and move down by ``step`` each time.
They fill the grid column by column (top to bottom, then left to right), which is
the layout used in the original thesis notebooks. The final image is transposed
to match the orientation those notebooks fed to the CNN.

Volumes are expected as NumPy arrays in SimpleITK order (z, y, x), e.g. the
91 x 109 x 91 shape of MNI152 2 mm space.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

#: Array axis that is sliced for each anatomical view (SimpleITK z, y, x order).
VIEW_AXIS: Dict[str, int] = {"axial": 0, "coronal": 1, "sagittal": 2}

#: Slice selection for each grid size, as used in the thesis: grid -> (top, step).
#: Slices are ``top, top - step, top - 2*step, ...`` (n*n slices in total).
GRID_PRESETS: Dict[int, Tuple[int, int]] = {
    1: (44, 0),
    2: (60, 10),
    3: (65, 5),
    4: (68, 3),
    5: (68, 2),
    6: (62, 1),
    7: (72, 1),
}


def slice_indices(grid: int, top: Optional[int] = None, step: Optional[int] = None) -> List[int]:
    """Return the slice indices used for an ``grid x grid`` mosaic."""
    if grid < 1:
        raise ValueError("grid must be a positive integer")
    preset_top, preset_step = GRID_PRESETS.get(grid, (None, None))
    top = preset_top if top is None else top
    step = preset_step if step is None else step
    if top is None or step is None:
        raise ValueError(f"No preset for grid={grid}; please pass top and step explicitly")
    return [top - step * k for k in range(grid * grid)]


def cast_volume(
    volume: np.ndarray,
    view: str,
    grid: int,
    top: Optional[int] = None,
    step: Optional[int] = None,
    transpose: bool = True,
) -> np.ndarray:
    """Cast a 3D volume into a 2D mosaic image.

    Args:
        volume: 3D array in (z, y, x) order.
        view: ``"axial"``, ``"coronal"`` or ``"sagittal"``.
        grid: mosaic size n (n x n slices).
        top, step: optional overrides for the slice selection.
        transpose: transpose the result to match the orientation used in the thesis.

    Returns:
        2D float32 array of shape (n*b, n*a) if ``transpose`` else (n*a, n*b),
        where (a, b) is the shape of a single slice.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")
    if view not in VIEW_AXIS:
        raise ValueError(f"view must be one of {list(VIEW_AXIS)}")

    axis = VIEW_AXIS[view]
    idx = slice_indices(grid, top, step)
    depth = volume.shape[axis]
    if min(idx) < 0 or max(idx) >= depth:
        raise IndexError(f"Slices {min(idx)}..{max(idx)} fall outside the {view} axis (size {depth})")

    slices = np.moveaxis(volume, axis, 0)[idx]          # (n*n, a, b)
    a, b = slices.shape[1:]
    # k-th slice goes to column k // n and row k % n (column-major filling).
    mosaic = slices.reshape(grid, grid, a, b).transpose(1, 2, 0, 3).reshape(grid * a, grid * b)
    if transpose:
        mosaic = mosaic.T
    return np.ascontiguousarray(mosaic, dtype=np.float32)


def cast_shape(volume_shape: Tuple[int, int, int], view: str, grid: int, transpose: bool = True) -> Tuple[int, int]:
    """Shape of the mosaic produced by :func:`cast_volume` without computing it."""
    dims = [d for i, d in enumerate(volume_shape) if i != VIEW_AXIS[view]]
    h, w = grid * dims[0], grid * dims[1]
    return (w, h) if transpose else (h, w)
