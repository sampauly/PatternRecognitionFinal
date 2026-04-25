# WFH Burnout — Feature Importance Analysis
**Author:** Samuel Pauly

Identifies the key behavioral predictors of work-from-home burnout using
three machine learning classifiers, derives statistically validated
intervention thresholds, and tests simplified model performance.

---

## Files

| File | Description |
|---|---|
| `wfh_burnout_dataset.csv` | Dataset (required) |
| `main.py` | Runs the full pipeline in one script |
| `01_EDA.ipynb` | Exploratory data analysis |
| `02_Models.ipynb` | Model training and feature importance |
| `03_Threshold_Analysis.ipynb` | Threshold derivation and bootstrap robustness |
| `04_Simplified_Interactions.ipynb` | Simplified models and interaction effects |
| `requirements.txt` | Python dependencies |

---

## Setup

**1. Install dependencies**
```
pip install -r requirements.txt
```

**2. Place the dataset in the same folder as the code files**
```
wfh_burnout_dataset.csv
```

---

## How to Run

### Option A — Single script (recommended)
Runs the entire analysis from start to finish:
```
python main.py
```
Estimated runtime: **2–3 minutes** (bootstrap and permutation importance are the slowest steps).
Progress is printed to the console at each step.

### Option B — Notebooks individually
Run in order inside Jupyter:
```
01_EDA.ipynb
02_Models.ipynb
03_Threshold_Analysis.ipynb
04_Simplified_Interactions.ipynb
```
Note: notebook 3 loads CSV files saved by notebook 2, so they must be run in order.

---

## Outputs

**Figures**
- `fig_01_class_distribution.png` — burnout class bar chart and pie chart
- `fig_04_boxplots_by_class.png` — feature distributions by burnout class
- `fig_08_threshold_violins.png` — violin plots with intervention threshold lines
- `fig_09_bootstrap_importance.png` — bootstrap feature importance with 95% CI
- `fig_10_roc_comparison.png` — ROC curve, full vs simplified model

**CSVs**
- `lr_importance.csv`, `rf_importance.csv`, `svm_importance.csv` — feature importance scores
- `thresholds.csv` — derived intervention thresholds per feature
- `bootstrap_importance.csv` — bootstrap mean importance and confidence intervals
- `simplified_model_results.csv` — simplified model performance comparison
