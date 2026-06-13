# ML-FA-3DPC

Source code for the paper:

**Interpretable Machine Learning for Compressive and Flexural Strength Prediction of Fly Ash Blended 3D Printed Concrete with Uncertainty Quantification**

*CMC-Computers, Materials & Continua* (2026)

---

## Repository Contents

```
ML-FA-3DPC/
├── train.py          # LOMO cross-validation benchmark (8 models, TPE hyperparameter optimisation)
├── GUI/
│   ├── fa_3dpc_gui.py    # Standalone prediction interface
│   └── dataset.csv       # Experimental dataset (126 records, 7 mix compositions)
└── README.md
```

This release covers the core model benchmarking pipeline and the graphical user interface. Additional analysis scripts (SHAP attribution, prediction interval calibration, design-space mapping, Sobol sensitivity analysis) are available from the corresponding author upon request.

---

## Dataset

126 records from seven fly ash blended 3D printed concrete mix compositions:

- FA replacement: 0, 5, 7.5, 10, 15 wt.%
- Water-to-binder ratio (W/B): 0.30, 0.34, 0.35
- Curing age: 1, 3, 7, 14, 21, 28 days
- Targets: compressive strength (CS, MPa) and flexural strength (FS, MPa)

Source: https://doi.org/10.1016/j.cscm.2025.e05682

---

## Requirements

Python 3.9 or later. Install dependencies with:

```bash
pip install numpy pandas scikit-learn optuna lightgbm xgboost catboost gpytorch customtkinter shap
```

---

## Usage

### 1. Model Benchmark (`train.py`)

Runs leave-one-mix-out (LOMO) cross-validation for eight regression algorithms (ElasticNet, SVR, Random Forest, ExtraTrees, XGBoost, LightGBM, CatBoost, GPR). Hyperparameters are tuned per fold using TPE (Optuna, 30 trials).

```bash
python train.py
```

Outputs are saved to `results/`:

- `lomo_summary.csv` — mean ± SD of R², RMSE, MAE per model and target
- `lomo_detail.csv` — per-fold metrics
- `best_info.json` — best model configuration per target
- `fig_lomo_comparison.png` — model comparison bar chart
- `fig_lomo_scatter.png` — predicted vs. measured scatter plots

The dataset path is set at the top of `train.py` and defaults to `dataset.csv` in the same directory.

### 2. Graphical User Interface (`GUI/fa_3dpc_gui.py`)

A standalone prediction interface that trains both best models (ExtraTrees for CS, ElasticNet for FS) from the dataset at startup and returns simultaneous CS and FS predictions with 90% calibrated prediction intervals.

```bash
cd GUI
python fa_3dpc_gui.py
```

Three inputs are required: FA content (wt.%), W/B ratio, and curing age (days). The interface runs fully offline with no internet connection required.

**Valid input ranges** (training data boundaries):

| Input | Range |
|-------|-------|
| FA content | 0 – 15 wt.% |
| W/B ratio | 0.30 – 0.35 |
| Curing age | 1 – 28 days |

Predictions outside these ranges are extrapolations and should be interpreted with caution.

---

## Contact

For questions about the code or additional analysis scripts, contact the co-author at: lucassivan@163.com
