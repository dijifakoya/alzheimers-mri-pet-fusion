"""mripet: image casting and fusion of MRI and PET for predicting MCI-to-Alzheimer's conversion."""

__version__ = "1.0.0"

from .casting import GRID_PRESETS, VIEW_AXIS, cast_volume, slice_indices  # noqa: F401
from .fusion import fuse_images  # noqa: F401
from .preprocessing import normalise  # noqa: F401
