"""Grad-CAM explanations: which parts of the mosaic drove a prediction?

Works with the thesis CNN (``model.architecture: cnn``) for single-input and
two-branch fusion models. For fusion models a heat map is produced for each branch.

Example:
    python -m mripet.explain --model models/holdout/fusion_coronal_6x6_feature/best_model.keras \
        --config configs/default.yaml --modality fusion --fusion-level feature --view coronal --grid 6 --n 4
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from scipy import ndimage  # noqa: E402

from .augment import to_model_input  # noqa: E402
from .config import load_config  # noqa: E402
from .data import load_arrays  # noqa: E402
from .train import add_experiment_args, experiment_name, make_inputs, modalities_needed, resolve_experiment  # noqa: E402


def last_conv_layers(model: keras.Model) -> Dict[str, str]:
    """Name of the last Conv2D layer per branch ('' for single-input models)."""
    found: Dict[str, str] = {}
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            prefix = layer.name.split("conv")[0]  # 'mri_', 'pet_' or ''
            found[prefix] = layer.name
    return found


def grad_cam(model: keras.Model, inputs: Dict[str, np.ndarray], conv_layer: str) -> np.ndarray:
    """Grad-CAM heat maps (N, H, W) in [0, 1], resized to the input size."""
    grad_model = keras.Model(model.inputs, [model.get_layer(conv_layer).output, model.output])
    feed = to_model_input({k: tf.convert_to_tensor(v) for k, v in inputs.items()})
    with tf.GradientTape() as tape:
        conv_out, pred = grad_model(feed, training=False)
        score = pred[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1)).numpy()
    h, w = next(iter(inputs.values())).shape[1:3]
    maps = []
    for c in cam:
        c = ndimage.zoom(c, (h / c.shape[0], w / c.shape[1]), order=1)
        maps.append((c - c.min()) / (c.max() - c.min() + 1e-8))
    return np.stack(maps)


def save_overlays(images: np.ndarray, cams: np.ndarray, probs: np.ndarray, labels: np.ndarray,
                  path: Path, title: str) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.4), squeeze=False)
    for ax, img, cam, p, y in zip(axes[0], images, cams, probs, labels):
        ax.imshow(img[..., 0], cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.35)
        ax.set_title(f"true={'pMCI' if y else 'sMCI'}  p={p:.2f}", fontsize=10)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM heat maps for a trained model.")
    add_experiment_args(parser)
    parser.add_argument("--model", required=True, help="Path to a saved .keras model")
    parser.add_argument("--n", type=int, default=4, help="Number of examples to explain")
    parser.add_argument("--out", default=None, help="Output folder (default: results/explain/<experiment>)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    exp = resolve_experiment(cfg, args)
    model = keras.models.load_model(args.model)
    manifest = pd.read_csv(cfg["paths"]["manifest"])
    arrays = load_arrays(manifest, modalities_needed(exp), exp["view"], exp["grid"], cfg)
    inputs = make_inputs(arrays, exp)

    rng = np.random.default_rng(cfg["seed"])
    idx = rng.choice(len(arrays["y"]), size=min(args.n, len(arrays["y"])), replace=False)
    sample = {k: v[idx] for k, v in inputs.items()}
    probs = model.predict(to_model_input(sample), verbose=0).ravel()

    out = Path(args.out or Path(cfg["paths"]["results_dir"]) / "explain" / experiment_name(exp))
    out.mkdir(parents=True, exist_ok=True)
    for prefix, layer in last_conv_layers(model).items():
        cams = grad_cam(model, sample, layer)
        key = prefix.rstrip("_") or "image"
        save_overlays(sample[key], cams, probs, arrays["y"][idx], out / f"gradcam_{key}.png",
                      f"Grad-CAM ({key.upper()}) — {experiment_name(exp)}")
        print(f"Saved {out / f'gradcam_{key}.png'}")


if __name__ == "__main__":
    main()
