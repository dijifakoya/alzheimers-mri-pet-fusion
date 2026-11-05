"""The vectorised casting must reproduce the original thesis loop implementation exactly."""
import numpy as np
import pytest

from mripet.casting import GRID_PRESETS, VIEW_AXIS, cast_shape, cast_volume, slice_indices


def original_thesis_cast(img, view, grid, top, step):
    """Loop implementation copied from the thesis notebooks (generalised over view and grid)."""
    take = {
        "axial": lambda k: img[k, :, :],
        "coronal": lambda k: img[:, k, :],
        "sagittal": lambda k: img[:, :, k],
    }[view]
    a, b = take(0).shape
    image_2d = np.empty((a * grid, b * grid))
    cut_it, row_it, col_it = top, 0, 0
    for t in range(grid * grid):
        cut = take(cut_it)
        cut_it -= step
        if t in [grid * i for i in range(1, grid)]:
            row_it = 0
            col_it += cut.shape[1]
        for i in range(cut.shape[0]):
            for j in range(cut.shape[1]):
                image_2d[i + row_it, j + col_it] = cut[i, j]
        row_it += cut.shape[0]
    return np.repeat(image_2d[None, ...], 3, axis=0).T[..., 0]


@pytest.mark.parametrize("view", list(VIEW_AXIS))
@pytest.mark.parametrize("grid", sorted(GRID_PRESETS))
def test_matches_original(view, grid):
    vol = np.random.default_rng(grid).random((91, 109, 91)).astype(np.float32)
    top, step = GRID_PRESETS[grid]
    np.testing.assert_array_equal(cast_volume(vol, view, grid), original_thesis_cast(vol, view, grid, top, step))


def test_shapes_match_thesis_inputs():
    # Input shapes used by the thesis CNNs
    assert cast_shape((91, 109, 91), "coronal", 6) == (546, 546)
    assert cast_shape((91, 109, 91), "sagittal", 5) == (545, 455)
    assert cast_volume(np.zeros((91, 109, 91)), "axial", 3).shape == (273, 327)


def test_slice_indices_and_bounds():
    assert slice_indices(2) == [60, 50, 40, 30]
    assert len(slice_indices(7)) == 49
    with pytest.raises(IndexError):
        cast_volume(np.zeros((91, 109, 91)), "axial", 2, top=95, step=1)
