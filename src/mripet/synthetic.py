"""Generate a small synthetic MRI/PET dataset so the pipeline can be run without ADNI access.

The volumes are NOT real brain scans. They are smooth ellipsoid "brains" in the
91 x 109 x 91 template shape. For the "converter" class, intensity is reduced in
two medial-temporal regions on MRI (mimicking atrophy) and in temporo-parietal
regions on PET (mimicking hypometabolism), so a model has a real signal to learn.

Example:
    python -m mripet.synthetic --out data/synthetic/raw --n-per-class 30
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

SHAPE = (91, 109, 91)  # (z, y, x)


def _ellipsoid(shape: Sequence[int], centre: Sequence[float], radii: Sequence[float]) -> np.ndarray:
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    d = ((zz - centre[0]) / radii[0]) ** 2 + ((yy - centre[1]) / radii[1]) ** 2 + ((xx - centre[2]) / radii[2]) ** 2
    return d <= 1.0


def synthetic_pair(converter: bool, rng: np.random.Generator, effect: float = 0.5):
    """Return (mri, pet) float32 volumes for one synthetic subject."""
    c = np.array(SHAPE) / 2 + rng.normal(0, 1.5, 3)
    brain = _ellipsoid(SHAPE, c, (36, 46, 38)).astype(np.float32)
    texture = ndimage.gaussian_filter(rng.normal(0, 1, SHAPE), 3).astype(np.float32)

    mri = brain * (250 + 60 * texture) + rng.normal(0, 8, SHAPE)
    pet = brain * (1.0 + 0.15 * texture) + rng.normal(0, 0.03, SHAPE)

    severity = effect * (rng.uniform(0.6, 1.0) if converter else rng.uniform(0.0, 0.25))
    for side in (-1, 1):  # left and right regions
        hippo = _ellipsoid(SHAPE, (c[0] - 8, c[1] - 6, c[2] + side * 18), (8, 10, 6))
        parietal = _ellipsoid(SHAPE, (c[0] + 10, c[1] + 10, c[2] + side * 20), (10, 12, 8))
        mri[hippo] *= 1 - severity
        pet[parietal] *= 1 - severity
    mri = ndimage.gaussian_filter(np.clip(mri, 0, None), 0.8)
    pet = ndimage.gaussian_filter(np.clip(pet, 0, None), 1.2) * 1000.0
    return mri.astype(np.float32), pet.astype(np.float32)


def write_dataset(out_dir: str | Path, n_per_class: int = 30, scans_per_subject: int = 1, seed: int = 0) -> Path:
    """Write NIfTI files in the layout expected by :mod:`mripet.io`."""
    import SimpleITK as sitk

    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    subject = 1
    for group, converter in (("nMCIC", False), ("MCIC", True)):
        for modality in ("MRI", "PET"):
            (out_dir / group / modality).mkdir(parents=True, exist_ok=True)
        for _ in range(n_per_class):
            sid = f"999_S_{subject:04d}"
            for visit in range(scans_per_subject):
                mri, pet = synthetic_pair(converter, rng)
                for modality, vol in (("MRI", mri), ("PET", pet)):
                    img = sitk.GetImageFromArray(vol)
                    img.SetSpacing((2.0, 2.0, 2.0))
                    sitk.WriteImage(img, str(out_dir / group / modality / f"{sid}_v{visit:02d}.nii.gz"))
            subject += 1
    return out_dir


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic MRI/PET dataset for testing.")
    parser.add_argument("--out", default="data/synthetic/raw")
    parser.add_argument("--n-per-class", type=int, default=30)
    parser.add_argument("--scans-per-subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    path = write_dataset(args.out, args.n_per_class, args.scans_per_subject, args.seed)
    print(f"Synthetic dataset written to {path}")


if __name__ == "__main__":
    main()
