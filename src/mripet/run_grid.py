"""Run the full thesis comparison: every view x grid size x modality, with cross-validation.

The experiments come from the ``grid_search`` section of the config. Results are
collected into ``results/grid_summary.csv``.

Example:
    python -m mripet.run_grid --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import load_config
from .cross_validate import run_cv
from .train import experiment_name

log = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Compare views, grid sizes and modalities.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    gs = cfg["grid_search"]

    rows = []
    for modality, view, grid in itertools.product(gs["modalities"], gs["views"], gs["grids"]):
        levels = gs["fusion_levels"] if modality == "fusion" else [None]
        for level in levels:
            exp = {**cfg["experiment"], "modality": modality, "view": view, "grid": grid}
            if level:
                exp["fusion_level"] = level
            log.info("=== %s ===", experiment_name(exp))
            summary = run_cv(cfg, exp)
            row = {"experiment": experiment_name(exp), "modality": modality, "view": view, "grid": grid,
                   "fusion_level": level or "-"}
            for metric in summary.index:
                row[f"{metric}_mean"] = summary.loc[metric, "mean"]
                row[f"{metric}_std"] = summary.loc[metric, "std"]
            rows.append(row)
            out = Path(cfg["paths"]["results_dir"]) / "grid_summary.csv"
            pd.DataFrame(rows).to_csv(out, index=False)  # saved after each run in case of interruption

    table = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
    print(table[["experiment", "accuracy_mean", "sensitivity_mean", "specificity_mean", "auc_mean"]]
          .round(3).to_string(index=False))
    return table


if __name__ == "__main__":
    main()
