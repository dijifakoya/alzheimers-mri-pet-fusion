"""Figure showing how a 3D scan is cast into 2D mosaics for each view and grid size.

Example:
    python -m mripet.visualise --mri data/synthetic/raw/MCIC/MRI/999_S_0031_v00.nii.gz \
        --pet data/synthetic/raw/MCIC/PET/999_S_0031_v00.nii.gz --out docs/casting_example.png
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .casting import cast_volume  # noqa: E402
from .io import load_volume  # noqa: E402
from .preprocessing import normalise  # noqa: E402


def casting_figure(mri_path: str, pet_path: str, out: str, grids: Sequence[int] = (3, 5, 7),
                   views: Sequence[str] = ("axial", "coronal", "sagittal")) -> None:
    mri = normalise(load_volume(mri_path), "clip", vmin=0, vmax=400)
    pet = normalise(load_volume(pet_path), "minmax")
    rows = [(f"MRI {v}", mri, v) for v in views] + [(f"PET {v}", pet, v) for v in views]
    fig, axes = plt.subplots(len(rows), len(grids), figsize=(3.2 * len(grids), 3.2 * len(rows)))
    for r, (label, vol, view) in enumerate(rows):
        for c, g in enumerate(grids):
            ax = axes[r, c]
            ax.imshow(cast_volume(vol, view, g), cmap="gray" if label.startswith("MRI") else "hot")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{g} x {g}")
            if c == 0:
                ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot example mosaics for one subject.")
    parser.add_argument("--mri", required=True)
    parser.add_argument("--pet", required=True)
    parser.add_argument("--out", default="docs/casting_example.png")
    args = parser.parse_args(argv)
    casting_figure(args.mri, args.pet, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
