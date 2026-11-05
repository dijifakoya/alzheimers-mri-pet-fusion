"""Stratified, subject-level k-fold cross-validation for one experiment.

Example:
    python -m mripet.cross_validate --config configs/default.yaml --modality fusion --fusion-level feature \
        --view coronal --grid 6
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import load_config, save_config, set_seed
from .data import load_arrays
from .metrics import summarise_folds
from .splits import cv_splits
from .train import (add_experiment_args, experiment_name, fit_and_evaluate, make_inputs, modalities_needed,
                    resolve_experiment)

log = logging.getLogger(__name__)


def run_cv(cfg, exp) -> pd.DataFrame:
    seed = cfg["seed"]
    set_seed(seed, cfg.get("deterministic", False))
    manifest = pd.read_csv(cfg["paths"]["manifest"])
    arrays = load_arrays(manifest, modalities_needed(exp), exp["view"], exp["grid"], cfg)
    inputs = make_inputs(arrays, exp)

    base = Path(cfg["paths"]["results_dir"]) / "cv" / experiment_name(exp)
    save_config({**cfg, "experiment": exp}, base / "config_used.yaml")
    folds = []
    for k, split in enumerate(cv_splits(arrays["y"], arrays["groups"], cfg["split"]["cv_folds"],
                                        cfg["split"]["val_fraction"], seed), start=1):
        log.info("Fold %d/%d", k, cfg["split"]["cv_folds"])
        m = fit_and_evaluate(inputs, arrays["y"], split, exp, cfg, base / f"fold_{k}", seed + k)
        folds.append({"fold": k, **m})

    per_fold = pd.DataFrame(folds)
    per_fold.to_csv(base / "folds.csv", index=False)
    summary = summarise_folds(folds)
    summary.to_csv(base / "summary.csv")
    print(f"\n{experiment_name(exp)} — {cfg['split']['cv_folds']}-fold CV (mean ± sd)")
    print(summary.round(3).to_string())
    return summary


def main(argv: Optional[List[str]] = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Subject-level k-fold cross-validation.")
    add_experiment_args(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    return run_cv(cfg, resolve_experiment(cfg, args))


if __name__ == "__main__":
    main()
