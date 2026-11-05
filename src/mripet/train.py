"""Train and evaluate one experiment on a subject-level train/validation/test split.

Examples:
    # MRI–PET feature-level fusion, coronal view, 6 x 6 mosaic
    python -m mripet.train --config configs/default.yaml --modality fusion --fusion-level feature --view coronal --grid 6

    # MRI only, axial view, 5 x 5 mosaic
    python -m mripet.train --config configs/default.yaml --modality mri --view axial --grid 5
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import keras
import numpy as np
import pandas as pd

from .augment import MosaicSequence, to_model_input
from .config import load_config, save_config, set_seed
from .data import load_arrays
from .fusion import fuse_images
from .metrics import (classification_metrics, plot_confusion, plot_history, plot_roc, save_metrics,
                      youden_threshold)
from .models import build_single_input, build_two_branch, compile_model
from .splits import train_val_test_split

log = logging.getLogger(__name__)


def experiment_name(exp: Dict[str, Any]) -> str:
    name = f"{exp['modality']}_{exp['view']}_{exp['grid']}x{exp['grid']}"
    if exp["modality"] == "fusion":
        name += f"_{exp['fusion_level']}"
        if exp["fusion_level"] == "image":
            name += f"-{exp['image_fusion_method']}"
    return name


def modalities_needed(exp: Dict[str, Any]) -> List[str]:
    return ["mri", "pet"] if exp["modality"] == "fusion" else [exp["modality"]]


def make_inputs(arrays: Dict[str, np.ndarray], exp: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Model inputs keyed by the model's input names."""
    if exp["modality"] in ("mri", "pet"):
        return {"image": arrays[exp["modality"]]}
    if exp["fusion_level"] == "image":
        return {"image": fuse_images(arrays["mri"], arrays["pet"], exp["image_fusion_method"])}
    return {"mri": arrays["mri"], "pet": arrays["pet"]}


def build_model(inputs: Dict[str, np.ndarray], exp: Dict[str, Any], cfg: Dict[str, Any]) -> keras.Model:
    shape = next(iter(inputs.values())).shape[1:]
    if "image" in inputs:
        model = build_single_input(shape, cfg["model"], name=experiment_name(exp))
    else:
        model = build_two_branch(shape, cfg["model"], level=exp["fusion_level"])
    return compile_model(model, cfg["training"])


def _subset(inputs: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k: v[idx] for k, v in inputs.items()}


def fit_and_evaluate(inputs: Dict[str, np.ndarray], y: np.ndarray, split: Tuple[np.ndarray, np.ndarray, np.ndarray],
                     exp: Dict[str, Any], cfg: Dict[str, Any], out_dir: Path, seed: int) -> Dict[str, Any]:
    """Train on one split, evaluate on its test set, and save metrics and figures to ``out_dir``."""
    train, val, test = split
    tcfg = cfg["training"]
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed, cfg.get("deterministic", False))
    keras.backend.clear_session()

    model = build_model(inputs, exp, cfg)
    angles = tcfg["rotation_angles"] if tcfg.get("augment", False) else None
    train_seq = MosaicSequence(_subset(inputs, train), y[train], tcfg["batch_size"], angles, shuffle=True, seed=seed)
    val_seq = MosaicSequence(_subset(inputs, val), y[val], tcfg["batch_size"], shuffle=False)

    models_dir = Path(cfg["paths"]["models_dir"]) / out_dir.relative_to(cfg["paths"]["results_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt = models_dir / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor=tcfg["monitor"], mode="max", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor=tcfg["monitor"], mode="max", patience=tcfg["patience"],
                                      restore_best_weights=True),
    ]
    start = time.time()
    hist = model.fit(train_seq, validation_data=val_seq, epochs=tcfg["epochs"], callbacks=callbacks,
                     verbose=tcfg.get("verbose", 2))
    history = pd.DataFrame(hist.history)
    history.to_csv(out_dir / "history.csv", index_label="epoch")

    predict = lambda idx: model.predict(to_model_input(_subset(inputs, idx)),  # noqa: E731
                                        batch_size=tcfg["batch_size"], verbose=0).ravel()
    # The decision threshold is chosen on the validation set only, then applied to the test set.
    rule = cfg.get("evaluation", {}).get("threshold", "youden")
    threshold = youden_threshold(y[val], predict(val)) if rule == "youden" else float(rule)
    y_prob = predict(test)
    metrics = classification_metrics(y[test], y_prob, threshold)
    metrics.update({"experiment": experiment_name(exp), "threshold": round(threshold, 4),
                    "epochs_trained": len(history),
                    "train_n": int(len(train)), "val_n": int(len(val)), "test_n": int(len(test)),
                    "train_seconds": round(time.time() - start, 1)})
    save_metrics(metrics, out_dir / "metrics.json")
    pd.DataFrame({"y_true": y[test], "y_prob": y_prob}).to_csv(out_dir / "test_predictions.csv", index=False)
    plot_history(history, out_dir / "training_curves.png")
    if len(np.unique(y[test])) == 2:
        plot_roc(y[test], y_prob, out_dir / "roc_curve.png", experiment_name(exp))
    plot_confusion(y[test], y_prob, out_dir / "confusion_matrix.png", threshold)
    log.info("%s | acc %.3f | sens %.3f | spec %.3f | AUC %.3f", experiment_name(exp), metrics["accuracy"],
             metrics["sensitivity"], metrics["specificity"], metrics["auc"])
    return metrics


def resolve_experiment(cfg: Dict[str, Any], args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    exp = dict(cfg["experiment"])
    if args is not None:
        for key in ("modality", "view", "grid", "fusion_level", "image_fusion_method"):
            value = getattr(args, key, None)
            if value is not None:
                exp[key] = value
    return exp


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--modality", choices=["mri", "pet", "fusion"])
    parser.add_argument("--view", choices=["axial", "coronal", "sagittal"])
    parser.add_argument("--grid", type=int, choices=range(1, 8))
    parser.add_argument("--fusion-level", dest="fusion_level", choices=["image", "feature", "decision"])
    parser.add_argument("--image-fusion-method", dest="image_fusion_method",
                        choices=["six_channel", "concat", "sum", "channels", "average", "weighted"])


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description="Train one MRI/PET experiment on a held-out test split.")
    add_experiment_args(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    exp = resolve_experiment(cfg, args)
    seed = cfg["seed"]
    set_seed(seed, cfg.get("deterministic", False))

    manifest = pd.read_csv(cfg["paths"]["manifest"])
    arrays = load_arrays(manifest, modalities_needed(exp), exp["view"], exp["grid"], cfg)
    inputs = make_inputs(arrays, exp)
    split = train_val_test_split(arrays["y"], arrays["groups"], cfg["split"]["test_fraction"],
                                 cfg["split"]["val_fraction"], seed)

    out_dir = Path(cfg["paths"]["results_dir"]) / "holdout" / experiment_name(exp)
    save_config({**cfg, "experiment": exp}, out_dir / "config_used.yaml")
    metrics = fit_and_evaluate(inputs, arrays["y"], split, exp, cfg, out_dir, seed)
    print(pd.Series(metrics).to_string())
    return metrics


if __name__ == "__main__":
    main()
