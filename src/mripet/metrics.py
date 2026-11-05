"""Evaluation metrics and plots."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,  # noqa: E402
                             roc_auc_score, roc_curve)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Accuracy, sensitivity, specificity, precision, F1 and AUC (positive class = converter)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn), "n": int(len(y_true)),
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1 (choose it on validation data, never on test)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[best], 0.0, 1.0))


def save_metrics(metrics: Dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def summarise_folds(fold_metrics: list) -> pd.DataFrame:
    """Mean and standard deviation of each metric across cross-validation folds."""
    df = pd.DataFrame(fold_metrics)
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "auc"]
    return pd.DataFrame({"mean": df[keys].mean(), "std": df[keys].std(ddof=1)})


def plot_history(history: pd.DataFrame, path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history[metric], label="train")
        if f"val_{metric}" in history:
            ax.plot(history[f"val_{metric}"], label="validation")
        ax.set_title(metric.capitalize()); ax.set_xlabel("Epoch"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, path: str | Path, title: Optional[str] = None) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title or "ROC curve"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, path: str | Path, threshold: float = 0.5) -> None:
    cm = confusion_matrix(np.asarray(y_true).astype(int), (np.asarray(y_prob) >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], ["sMCI", "pMCI"]); ax.set_yticks([0, 1], ["sMCI", "pMCI"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
