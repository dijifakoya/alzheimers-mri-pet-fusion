"""Configuration loading and reproducibility helpers."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config. A config may name a parent with ``base: other.yaml``."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = _merge(parent, cfg)
    return _merge(cfg, overrides or {})


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and TensorFlow so runs can be repeated."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
        if deterministic:
            import tensorflow as tf

            tf.config.experimental.enable_op_determinism()
    except ImportError:  # pragma: no cover - TensorFlow is optional for preprocessing-only use
        pass


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
