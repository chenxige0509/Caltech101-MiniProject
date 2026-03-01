# Caltech-101 Mini Project

A comparative study of **CNN** (transfer learning) and **HOG** (hand-crafted features) models for image classification on [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02).

**Course:** AI in Medicine (Harvard/MIT)  
**Report:** [report/Caltech101_Mini_Project_Report.pdf](report/Caltech101_Mini_Project_Report.pdf) | LaTeX: [report/report.ltx](report/report.ltx)

---

## Overview

- **Models:** ResNet-18, EfficientNet-B0, Vision Transformer (ViT-B/16) with frozen backbones + trainable head; HOG + SVM / HOG + logistic regression.
- **Ablations:** Input resolution (64 / 128 / 224), data augmentation (on/off), optimizer (SGD / Adam / AdamW).
- **Metrics:** Test accuracy, top-5 accuracy, per-class precision/recall/F1.

Best test accuracy in the report: **ViT-B/16 95.93%** at 224×224; HOG+LR baseline **66.05%** at 64×64.

---

## Setup

- **Python:** 3.12 (see [.python-version](.python-version)).
- **Package manager:** [uv](https://github.com/astral-sh/uv) (or pip).

```bash
# Clone and enter project
git clone https://github.com/chenxige0509/Caltech101-MiniProject.git
cd Caltech101-MiniProject

# Install dependencies (uv)
uv sync

# Or with pip
pip install -e .
```

---

## Data

1. **Caltech-101**  
   Download and unzip into the project root so that the folder `caltech-101/` exists (with subfolders per class, e.g. `caltech-101/airplanes/`, `caltech-101/BACKGROUND_Google/`, etc.).  
   The dataset is not in the repo (too large).

2. **Train/val/test splits**  
   The code expects `train.csv`, `val.csv`, and `test.csv` in the project root (columns: `filepath`, `label`, `label_id`).  
   These can be generated from the notebook **caltech101_data_split.ipynb** (stratified 70/15/15 split). The CSVs are gitignored; regenerate them after placing Caltech-101.

---

## Run experiments

Single entry point: run all CNN ablations, HOG training, HOG F1 evaluation, and the 24 remaining ResNet/EfficientNet ablation combos.

```bash
# Full run (default 25 epochs)
uv run python -m src.run_all

# Fewer epochs
uv run python -m src.run_all --epochs 10

# Quick smoke test (baseline only, no full ablations)
uv run python -m src.run_all --epochs 3 --quick
```

Results are written under:

- `outputs/logs/` — JSON results, training history, classification reports.
- `outputs/figures/` — plots.
- `outputs/checkpoints/` — saved CNN checkpoints (`.pt`) and HOG models (`.pkl`). Checkpoints are gitignored.

---

## Project layout

```
.
├── README.md
├── pyproject.toml
├── uv.lock
├── caltech101_data_split.ipynb   # Data split + exploration
├── train.csv / val.csv / test.csv   # Generated from notebook (not in repo)
├── src/
│   ├── config.py         # Project config, paths, seed, device
│   ├── models.py         # CNN model factory (frozen backbone + head)
│   ├── train.py          # CNN training loop
│   ├── eval.py           # CNN evaluation on test set
│   ├── train_hog.py      # HOG + SVM/LR training and evaluation
│   └── run_all.py        # Run all experiments
├── report/
│   ├── report.ltx
│   └── Caltech101_Mini_Project_Report.pdf
└── outputs/
    ├── logs/
    ├── figures/
    └── checkpoints/
```

---

## Citation

Caltech-101: Li, Andreeto, Ranzato, Perona. [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02).  
See the report [report/report.ltx](report/report.ltx) for full references.

---
