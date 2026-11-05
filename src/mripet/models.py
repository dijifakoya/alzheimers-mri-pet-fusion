"""CNN models for single-modality and fused MRI/PET classification."""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import keras
from keras import layers

PRETRAINED = {
    "vgg16": keras.applications.VGG16,
    "vgg19": keras.applications.VGG19,
    "resnet50": keras.applications.ResNet50,
    "resnet101": keras.applications.ResNet101,
    "resnet50v2": keras.applications.ResNet50V2,
    "resnet101v2": keras.applications.ResNet101V2,
    "densenet121": keras.applications.DenseNet121,
    "mobilenetv2": keras.applications.MobileNetV2,
    "xception": keras.applications.Xception,
}


def conv_stack(x, filters: Sequence[int] = (16, 32, 32, 64, 64), kernel_size: int = 2, l2: float = 0.01,
               prefix: str = ""):
    """The thesis feature extractor: Conv2D -> MaxPool -> BatchNorm blocks, then Flatten."""
    reg = keras.regularizers.l2(l2)
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, kernel_size, activation="relu", kernel_regularizer=reg, name=f"{prefix}conv{i + 1}")(x)
        x = layers.MaxPooling2D(name=f"{prefix}pool{i + 1}")(x)
        x = layers.BatchNormalization(name=f"{prefix}bn{i + 1}")(x)
    return layers.Flatten(name=f"{prefix}flatten")(x)


def classifier_head(x, dense_units: int = 128, dropout: float = 0.3, l2: float = 0.01, name: str = "prediction"):
    reg = keras.regularizers.l2(l2)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name=f"{name}_dense")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return layers.Dense(1, activation="sigmoid", name=name)(x)


def _backbone_features(inp, name: str, mcfg: Dict[str, Any], prefix: str = ""):
    """Features from the thesis CNN or a pretrained ImageNet backbone."""
    if name == "cnn":
        return conv_stack(inp, mcfg["filters"], mcfg["kernel_size"], mcfg["l2"], prefix)
    if name not in PRETRAINED:
        raise ValueError(f"Unknown architecture '{name}'. Choose 'cnn' or one of {list(PRETRAINED)}")
    x = inp
    if inp.shape[-1] == 1:  # pretrained networks expect 3 channels
        x = layers.Concatenate(name=f"{prefix}to_rgb")([inp, inp, inp])
    elif inp.shape[-1] != 3:  # e.g. fused 2- or 6-channel images: learn a 1x1 mapping to 3 channels
        x = layers.Conv2D(3, 1, name=f"{prefix}to_rgb")(inp)
    x = layers.Rescaling(255.0, name=f"{prefix}to_255")(x)
    base = PRETRAINED[name](include_top=False, weights=mcfg.get("pretrained_weights", "imagenet"),
                            input_shape=tuple(x.shape[1:]))
    base.name = f"{prefix}{name}"
    base.trainable = bool(mcfg.get("trainable_backbone", False))
    return layers.GlobalAveragePooling2D(name=f"{prefix}gap")(base(x))


def build_single_input(input_shape: Tuple[int, int, int], mcfg: Dict[str, Any], name: str = "cnn") -> keras.Model:
    """One input mosaic (MRI, PET, or image-level fused) -> probability of conversion."""
    inp = keras.Input(shape=input_shape, name="image")
    feats = _backbone_features(inp, mcfg["architecture"], mcfg)
    out = classifier_head(feats, mcfg["dense_units"], mcfg["dropout"], mcfg["l2"])
    return keras.Model(inp, out, name=name)


def build_two_branch(input_shape: Tuple[int, int, int], mcfg: Dict[str, Any], level: str = "feature") -> keras.Model:
    """Separate MRI and PET branches fused at the feature or decision level."""
    mri = keras.Input(shape=input_shape, name="mri")
    pet = keras.Input(shape=input_shape, name="pet")
    # The paper's best feature-level model used a different pre-trained network per branch
    # (ResNet101V2 for MRI, ResNet101 for PET); set model.branch_architectures to do the same.
    arch_mri, arch_pet = mcfg.get("branch_architectures") or [mcfg["architecture"]] * 2
    f_mri = _backbone_features(mri, arch_mri, mcfg, prefix="mri_")
    f_pet = _backbone_features(pet, arch_pet, mcfg, prefix="pet_")
    if level == "feature":
        fused = layers.Concatenate(name="feature_fusion")([f_mri, f_pet])
        out = classifier_head(fused, mcfg["dense_units"], mcfg["dropout"], mcfg["l2"])
    elif level == "decision":
        p_mri = classifier_head(f_mri, mcfg["dense_units"], mcfg["dropout"], mcfg["l2"], name="mri_prediction")
        p_pet = classifier_head(f_pet, mcfg["dense_units"], mcfg["dropout"], mcfg["l2"], name="pet_prediction")
        out = layers.Average(name="prediction")([p_mri, p_pet])
    else:
        raise ValueError("level must be 'feature' or 'decision'")
    return keras.Model([mri, pet], out, name=f"{level}_fusion")


def compile_model(model: keras.Model, tcfg: Dict[str, Any]) -> keras.Model:
    """Adam with exponential learning-rate decay and binary cross-entropy, as in the thesis."""
    schedule = keras.optimizers.schedules.ExponentialDecay(
        tcfg["learning_rate"], decay_steps=tcfg["decay_steps"], decay_rate=tcfg["decay_rate"], staircase=True)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=schedule),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    return model
