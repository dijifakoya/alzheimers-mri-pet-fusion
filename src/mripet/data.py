"""Turn the manifest into model-ready arrays (with caching)."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .casting import cast_volume
from .io import load_volume
from .preprocessing import ensure_shape, normalise

log = logging.getLogger(__name__)


def _norm_kwargs(cfg: Dict[str, Any], modality: str) -> Dict[str, Any]:
    params = dict(cfg["preprocessing"][modality])
    method = params.pop("method")
    return {"method": method, **params}


def cast_scan(path: str, modality: str, view: str, grid: int, cfg: Dict[str, Any]) -> np.ndarray:
    """Load, check, normalise and cast one scan. Returns (H, W) float32."""
    vol = load_volume(path)
    vol = ensure_shape(vol, cfg["data"]["expected_shape"], cfg["data"].get("resample_if_needed", False))
    vol = normalise(vol, **_norm_kwargs(cfg, modality))
    override = (cfg["casting"].get("overrides") or {}).get(grid, {})
    return cast_volume(vol, view, grid, top=override.get("top"), step=override.get("step"))


def _cache_path(cache_dir: Path, manifest: pd.DataFrame, modality: str, view: str, grid: int,
                cfg: Dict[str, Any]) -> Path:
    key = json.dumps({
        "files": manifest[f"{modality}_path"].tolist(),
        "pre": cfg["preprocessing"][modality],
        "cast": (cfg["casting"].get("overrides") or {}).get(grid, {}),
    }, sort_keys=True)
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    return cache_dir / f"{modality}_{view}_{grid}x{grid}_{digest}.npy"


def load_arrays(manifest: pd.DataFrame, modalities: List[str], view: str, grid: int,
                cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Return ``{modality: (N, H, W, 1) array, 'y': (N,), 'groups': (N,)}``."""
    cache_dir = Path(cfg["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, np.ndarray] = {}
    for m in modalities:
        cache = _cache_path(cache_dir, manifest, m, view, grid, cfg)
        if cache.exists() and cfg["data"].get("use_cache", True):
            log.info("Loading cached %s", cache.name)
            out[m] = np.load(cache)
            continue
        log.info("Casting %d %s scans (%s, %dx%d)...", len(manifest), m.upper(), view, grid, grid)
        arr = np.stack([cast_scan(p, m, view, grid, cfg) for p in manifest[f"{m}_path"]])[..., None]
        np.save(cache, arr)
        out[m] = arr
    out["y"] = manifest["label"].to_numpy(dtype=np.float32)
    out["groups"] = manifest["subject_id"].to_numpy()
    return out
