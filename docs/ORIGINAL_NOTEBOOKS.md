# From research notebooks to this package

This repository is a clean, tested re-implementation of the code from the PhD research notebooks. The table shows where each part of the original work now lives.

| Original notebooks | What they did | Now in |
|---|---|---|
| `MRI/PET Axial, Coronal, Sagittal image reconstruction 1 - 7` | Cast 3D scans into 1 × 1 … 7 × 7 mosaics per view; train the CNN; 5-fold CV | `casting.py`, `models.py`, `train.py`, `cross_validate.py` |
| `Axial/Coronal/Sagittal 5 X 5 Conversion` | Single-view 5 × 5 experiments | `train.py --view ... --grid 5` |
| `MRI/PET ... Conversion Data Augmentation` | Random rotation (−10°, −5°, 5°, 10°) during training | `augment.py` (`training.augment`) |
| `MRI/PET Fine Tuning`, `MRI/PET and Machine Learning` | Pre-trained networks (VGG16, ResNet, DenseNet, Xception, MobileNet) | `models.py` (`model.architecture`) |
| `MRI Axial Test` | Test-set evaluation, confusion matrices, ROC curves | `metrics.py` |
| Image fusion results | MRI–PET fusion | `fusion.py`, `models.py` (image-, feature- and decision-level fusion) |

## Kept exactly as in the thesis
- Slice selection for each grid size (`GRID_PRESETS` in `casting.py`) and the column-by-column mosaic layout. The test `tests/test_casting.py` checks the new code gives identical output to the original loop implementation for every view and grid size.
- MRI normalisation (clip to 0–400, scale to 0–1) and PET min–max normalisation.
- CNN architecture, L2 regularisation, dropout, Adam optimiser with exponential learning-rate decay, rotation augmentation.

## Improved
- **Subject-level splits** (`StratifiedGroupKFold`) instead of random splits of scans, so one person's scans cannot appear in both training and test data.
- **Per-scan normalisation** (the PET notebooks normalised across the whole dataset at once).
- **MRI and PET paired by subject ID** instead of by folder order.
- **Decision threshold chosen on validation data** and reported with sensitivity, specificity, F1 and AUC, not accuracy alone.
- **Configuration files, fixed seeds and saved settings** for every run instead of hard-coded paths.
- Vectorised image casting (much faster than the nested loops).
- **Grad-CAM explanations** and a **synthetic dataset** for testing without ADNI access.
- Single-channel mosaics (the original repeated the grey image three times; this is only needed for pre-trained networks, which the code handles automatically).

## Please check
- **Fusion level:** the original fusion notebook was not available, so all three standard fusion levels are implemented. Set `experiment.fusion_level` to the one described in the paper.
- **Pre-registration steps** (software and template used to produce 91 × 109 × 91 volumes) should be added to `data/README.md`.
