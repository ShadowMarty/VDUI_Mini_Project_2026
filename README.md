# Asteroid Prediction – Four-model implementation

This VS Code starter project compares four classifiers on the Kaggle asteroid dataset:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

It exports model metrics, prediction confidence, feature importance and timing results for Power BI.

## 1. Dataset setup

Download the CSV from:
https://www.kaggle.com/datasets/sakhawat18/asteroid-dataset/data

Place the extracted CSV at:

`dataset/dataset.csv`

If the filename differs, either rename it to `dataset.csv` or edit `DATA_PATH` near the top of each script.

**Important:** The preprocessing script automatically looks for a target named `PHA`, `is_hazardous`, or `potentially_hazardous`. If your target has another name, set `TARGET_COLUMN` in `src/config.py`.

## 2. Setup in Windows PowerShell

Open this folder in VS Code, then run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 3. Run scripts in order

```powershell
python src/01_inspection.py
python src/02_eda.py
python src/02_preprocessing.py
python src/03_train_models.py
```

Run `01_inspection.py` first. Read `output/inspection_summary.txt` and `output/missing_values_report.csv`. Preprocessing creates the cleaned dataset needed by model training; it removes records without a recognized target and reports missing feature values without imputing them.

If the target was not detected, set it explicitly in `src/config.py`.

## 4. What gets generated

### After inspection
- `output/inspection_summary.txt`
- `output/missing_values_report.csv`
- `output/sample_records.csv`

### After EDA
- `output/pha_distribution.png`
- `output/numeric_feature_distributions.png`
- `output/correlation_heatmap.png`

### After preprocessing
- `output/asteroid_cleaned.csv`
- `output/preprocessing_missing_values.csv`

### After model training
- `output/model_comparison.csv`
- `output/confusion_matrices.csv`
- `output/test_predictions.csv`
- `output/feature_importance.csv`
- `models/*.joblib`

## 5. Experimental fairness and interpretation

- All models use the same stratified train/test split and selected input columns.
- Numeric missing values are imputed using training data only.
- Logistic Regression is scaled; tree-based models are not scaled.
- The positive class is the hazardous class, where the target encoding can be identified.
- Report precision, recall, F1 and the confusion matrix, not accuracy alone. PHA datasets can be imbalanced.
- Training and prediction time are measured separately. Timing depends on hardware, software versions and dataset size.
- `predict_proba()` outputs model probability estimates; they are not automatically calibrated probabilities of real-world correctness.
- Feature importance methods differ by model. Do not compare raw importance values as if they had identical meanings.
- This project is an educational classification experiment, not an operational asteroid-risk assessment system.

## 6. Target leakage warning

The scientific definition of a potentially hazardous asteroid is related to orbital proximity and size. Some columns can therefore encode the target definition very directly. Inspect the dataset documentation and the columns before interpreting high scores.

The default configuration excludes common identifier fields and several potentially target-defining columns (`moid_ld`, `moid`, `H`, `absolute_magnitude`). You can adjust this list in `src/config.py`, but document and justify any change. Do not use the target column itself or a duplicate label as an input.

## 7. Power BI

In Power BI Desktop, use **Get Data → Text/CSV** and import:
- `model_comparison.csv`
- `test_predictions.csv`
- `confusion_matrices.csv`
- `feature_importance.csv`

Suggested visuals:
1. Model vs prediction time / training time.
2. Model comparison of precision, recall and F1.
3. Confidence distribution and low-confidence records.
4. Execution time vs recall/F1 trade-off.

Slicers can filter exported prediction records. Static feature importance does not automatically retrain a model or recalculate importance when a slicer changes.
