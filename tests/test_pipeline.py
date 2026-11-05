"""End-to-end smoke test on a tiny synthetic dataset (runs in about a minute on CPU)."""
import json

import pandas as pd
import pytest

from mripet.config import load_config
from mripet.io import build_manifest
from mripet.synthetic import write_dataset
from mripet.train import main as train_main


@pytest.fixture(scope="module")
def tiny_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    write_dataset(root / "raw", n_per_class=10, seed=1)
    cfg_path = root / "cfg.yaml"
    cfg = load_config("configs/synthetic.yaml")
    manifest = build_manifest(root / "raw", cfg["data"]["class_dirs"], cfg["data"]["modality_dirs"],
                              cfg["data"]["pair_pattern"])
    manifest.to_csv(root / "manifest.csv", index=False)
    import yaml
    cfg["paths"] = {"raw_dir": str(root / "raw"), "manifest": str(root / "manifest.csv"),
                    "cache_dir": str(root / "cache"), "results_dir": str(root / "results"),
                    "models_dir": str(root / "models")}
    cfg["training"].update({"epochs": 2, "patience": 2, "verbose": 0})
    cfg["model"]["filters"] = [4, 8]
    cfg["split"].update({"test_fraction": 0.25, "val_fraction": 0.25})
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path, root


def test_manifest(tiny_cfg):
    _, root = tiny_cfg
    df = pd.read_csv(root / "manifest.csv")
    assert len(df) == 20 and set(df.label) == {0, 1}


@pytest.mark.parametrize("args", [
    ["--modality", "mri"],
    ["--modality", "fusion", "--fusion-level", "image"],
    ["--modality", "fusion", "--fusion-level", "feature"],
    ["--modality", "fusion", "--fusion-level", "decision"],
])
def test_train_runs(tiny_cfg, args):
    cfg_path, root = tiny_cfg
    metrics = train_main(["--config", str(cfg_path), "--view", "coronal", "--grid", "2", *args])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    saved = list((root / "results" / "holdout").glob("*/metrics.json"))
    assert saved and "auc" in json.loads(saved[0].read_text())
