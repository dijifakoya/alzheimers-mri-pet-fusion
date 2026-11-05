"""Reading scans and building the subject manifest.

Expected folder layout (as used in the thesis):

    data/raw/
    ├── nMCIC/            # stable MCI (did not convert to AD within 36 months) -> label 0
    │   ├── MRI/*.nii(.gz)
    │   └── PET/*.nii(.gz)
    └── MCIC/             # progressive MCI (converted to AD within 36 months)   -> label 1
        ├── MRI/*.nii(.gz)
        └── PET/*.nii(.gz)

MRI and PET files are paired by a key extracted from the file name (by default the
ADNI subject ID, e.g. ``002_S_0413``). The manifest records the subject ID so that
all scans from one person stay in the same data split.

Usage:
    python -m mripet.io --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import load_config

log = logging.getLogger(__name__)
NIFTI_SUFFIXES = (".nii", ".nii.gz", ".img", ".mha", ".mhd")


def load_volume(path: str | Path) -> np.ndarray:
    """Load a scan as a float32 array in SimpleITK (z, y, x) order."""
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image).astype(np.float32)


def _scan_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.name.lower().endswith(NIFTI_SUFFIXES))


def _key(path: Path, pattern: Optional[str]) -> str:
    stem = path.name.split(".")[0]
    if pattern:
        match = re.search(pattern, path.name)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return stem


def build_manifest(
    raw_dir: str | Path,
    class_dirs: Dict[str, int],
    modality_dirs: Dict[str, str],
    pair_pattern: Optional[str] = r"(\d{3}_S_\d{4})",
) -> pd.DataFrame:
    """Pair MRI and PET scans per subject and return a manifest DataFrame.

    Columns: ``subject_id``, ``label``, ``group``, ``mri_path``, ``pet_path``.
    If a subject has several scans per modality, they are paired in file-name order.
    """
    raw_dir = Path(raw_dir)
    rows = []
    for group, label in class_dirs.items():
        files = {m: _scan_files(raw_dir / group / d) for m, d in modality_dirs.items()}
        keyed: Dict[str, Dict[str, List[Path]]] = {}
        for modality, paths in files.items():
            for p in paths:
                keyed.setdefault(_key(p, pair_pattern), {}).setdefault(modality, []).append(p)

        for key, by_mod in sorted(keyed.items()):
            missing = [m for m in modality_dirs if m not in by_mod]
            if missing:
                log.warning("%s/%s: no %s scan found; skipped", group, key, ", ".join(missing))
                continue
            counts = {m: len(v) for m, v in by_mod.items()}
            n = min(counts.values())
            if len(set(counts.values())) > 1:
                log.warning("%s/%s: unequal scan counts %s; pairing first %d", group, key, counts, n)
            for i in range(n):
                row = {"subject_id": key, "label": int(label), "group": group}
                for m in modality_dirs:
                    row[f"{m}_path"] = str(by_mod[m][i])
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No paired scans found under {raw_dir}. Check data/README.md for the layout.")
    log.info("Manifest: %d scans from %d subjects (%s)", len(df), df.subject_id.nunique(),
             df.groupby("group").size().to_dict())
    return df


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the MRI/PET subject manifest.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    df = build_manifest(cfg["paths"]["raw_dir"], cfg["data"]["class_dirs"],
                        cfg["data"]["modality_dirs"], cfg["data"].get("pair_pattern"))
    out = Path(cfg["paths"]["manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved manifest with {len(df)} paired scans to {out}")


if __name__ == "__main__":
    main()
