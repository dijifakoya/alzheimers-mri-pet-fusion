# Data

The imaging data come from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** and **cannot be redistributed**. No patient data are stored in this repository, and the `.gitignore` file blocks scan files from being uploaded.

## 1. Get access
Apply at [adni.loni.usc.edu](https://adni.loni.usc.edu). Once approved, download baseline **T1-weighted MRI** and **FDG-PET** scans for participants diagnosed with MCI.

## 2. Define the groups
- **pMCI / MCIC (label 1):** MCI at baseline, converted to Alzheimer's disease within 36 months.
- **sMCI / nMCIC (label 0):** MCI at baseline, did not convert within 36 months.

Use the ADNI diagnosis tables (e.g. `DXSUM`) to assign each participant.

## 3. Pre-process
The code expects scans already **registered to MNI152 space at 2 mm resolution**, giving volumes of **91 × 109 × 91** voxels (for example using SPM12 or FSL). PET scans should be co-registered to the same template.
<!-- TODO: add the exact registration/pre-processing steps used in the thesis (software, template, smoothing). -->

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

## Dataset used in the thesis

| Group | Scans |
|---|---|
| sMCI (nMCIC) | 434 |
| pMCI (MCIC) | 430 |

<!-- TODO: add the number of unique subjects per group and the ADNI phases used (ADNI-1/GO/2). -->
