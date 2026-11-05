# Data

The imaging data come from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** and **cannot be redistributed**. No patient data are stored in this repository, and the `.gitignore` file blocks scan files from being uploaded.

## 1. Get access
Apply at [adni.loni.usc.edu](https://adni.loni.usc.edu). Once approved, download **T1-weighted MRI** and **FDG-PET** scans (baseline and 6-month follow-up) for participants diagnosed with MCI.

## 2. Define the groups
- **pMCI / MCIC (label 1):** MCI at baseline, converted to Alzheimer's disease within 36 months.
- **sMCI / nMCIC (label 0):** MCI at baseline, did not convert within 36 months.

Use the ADNI diagnosis tables (e.g. `DXSUM`) to assign each participant.

## 3. Pre-process (as in the paper)

**MRI** (T1-weighted MPRAGE, 1.5T). ADNI had already corrected geometric distortion and intensity non-uniformity. Then:
1. Skull stripping with **FSL BET**.
2. Linear registration with **FSL FLIRT** to the **MNI152 2 mm** template (affine, 12 degrees of freedom).
3. This gives volumes of **91 × 109 × 91** voxels. Intensity normalisation is done by this code.

**FDG-PET.** Use the ADNI-processed images (co-registered, frame-averaged and smoothed by ADNI), converted from DICOM to NIfTI. Then apply the same skull stripping and FLIRT registration as for MRI.

Example FSL commands for one scan:
```bash
bet input.nii.gz brain.nii.gz -f 0.5 -R
flirt -in brain.nii.gz -ref $FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz \
      -out brain_mni.nii.gz -omat brain_mni.mat -dof 12
```

If your scans have a different size, set `data.resample_if_needed: true` in the config to resample them (registration is still recommended).

## 4. Arrange the files

```
data/raw/
├── nMCIC/
│   ├── MRI/   002_S_0729.nii.gz, 005_S_0222.nii.gz, ...
│   └── PET/   002_S_0729.nii.gz, 005_S_0222.nii.gz, ...
└── MCIC/
    ├── MRI/   ...
    └── PET/   ...
```

MRI and PET files are matched by the **ADNI subject ID** in the file name (pattern `123_S_4567`). If a subject has more than one scan, give the files the same suffix in both folders (e.g. `002_S_0729_bl.nii.gz`) so they pair correctly.

## 5. Build the manifest
```bash
python -m mripet.io --config configs/default.yaml
```
This creates `data/manifest.csv` listing each paired MRI/PET scan, its label and subject ID.

## Dataset used in the paper

Participants with MCI from ADNI 1, ADNI GO, ADNI 2 and ADNI 3 who had both MRI and FDG-PET. Baseline and 6-month follow-up scans were included where available.

| | Progressive MCI (pMCI) | Stable MCI (sMCI) |
|---|---|---|
| Participants | 234 | 233 |
| Age, mean (SD) | 74.4 (7.07) | 72.8 (7.03) |
| Sex | 136 M, 98 F | 141 M, 92 F |
| Education, years (SD) | 15.4 (2.8) | 16.7 (2.9) |
| Scans | 430 | 434 |

Because several participants have more than one scan, this code keeps each participant's scans in the same data split.
