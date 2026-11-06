from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = ROOT_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DATA_PATH = OUTPUT_DIR / "asteroid_cleaned.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURES = [
    "diameter", "albedo", "diameter_sigma",
    "e", "a", "q", "i", "om", "w", "ma",
    "ad", "n", "per", "per_y",
    "sigma_e", "sigma_a", "sigma_q", "sigma_i",
    "sigma_om", "sigma_w", "sigma_ma",
    "sigma_ad", "sigma_n", "sigma_per"
]

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("=" * 65)
print("ASTEROID CLASSIFICATION - MODEL TRAINING")
print("=" * 65)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_PATH}\n"
        "Run 02_preprocessing.py first."
    )

df = pd.read_csv(DATA_PATH)

target_candidates = ["pha", "target", "label"]
target_col = next(
    (col for col in target_candidates if col in df.columns),
    None
)

if target_col is None:
    raise ValueError(
        f"Could not identify target column. Available columns: {list(df.columns)}"
    )

missing_features = [col for col in FEATURES if col not in df.columns]

if missing_features:
    raise ValueError(f"Missing feature columns: {missing_features}")

X = df[FEATURES].copy()
y = pd.to_numeric(df[target_col], errors="coerce")

valid_rows = y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows].astype(int)

print(f"\nDataset records: {len(df):,}")
print(f"Usable records: {len(X):,}")
print(f"Features: {len(FEATURES)}")
print(f"Target column: {target_col}")
print("\nClass distribution:")
print(y.value_counts().sort_index().rename({
    0: "Non-PHA",
    1: "PHA"
}))

# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

positive_weight = (y_train == 0).sum() / (y_train == 1).sum()

print("\nTRAIN-TEST SPLIT")
print("-" * 45)
print(f"Training records: {len(X_train):,}")
print(f"Testing records:  {len(X_test):,}")
print(f"PHA records in training: {(y_train == 1).sum():,}")
print(f"PHA records in testing:  {(y_test == 1).sum():,}")
print(f"XGBoost positive-class weight: {positive_weight:.2f}")

# --------------------------------------------------
# 4. DEFINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=15,
        min_samples_leaf=5,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),

    "XGBoost": XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=positive_weight,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
}

# --------------------------------------------------
# 5. TRAIN AND EVALUATE
# --------------------------------------------------

results = []
confusion_results = []
importance_results = []

test_predictions = pd.DataFrame({
    "actual": y_test.to_numpy()
}, index=y_test.index)

for model_name, classifier in models.items():

    print("\n" + "=" * 65)
    print(f"TRAINING: {model_name}")
    print("=" * 65)

    # Each model learns imputation values from training data only.
    steps = [
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True
        ))
    ]

    # Scaling is important for Logistic Regression.
    if model_name == "Logistic Regression":
        steps.append(("scaler", StandardScaler()))

    steps.append(("classifier", classifier))

    pipeline = Pipeline(steps)

    # Training time
    start_train = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_time = time.perf_counter() - start_train

    # Prediction time
    start_predict = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    prediction_time = time.perf_counter() - start_predict

    # Probability estimates
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, y_probability)
    pr_auc = average_precision_score(y_test, y_probability)

    # Confidence: average maximum class probability.
    confidence = np.maximum(
        y_probability, 1 - y_probability
    ).mean()

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "PHA_Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Training_Time_sec": training_time,
        "Prediction_Time_sec": prediction_time,
        "Mean_Confidence": confidence
    })

    confusion_results.append({
        "Model": model_name,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    })

    # Save predictions for later dashboard use.
    safe_name = model_name.lower().replace(" ", "_")
    test_predictions[f"{safe_name}_prediction"] = y_pred
    test_predictions[f"{safe_name}_pha_probability"] = y_probability

    # Extract feature importance.
    imputer = pipeline.named_steps["imputer"]
    transformed_features = imputer.get_feature_names_out(FEATURES)

    if model_name == "Logistic Regression":
        importance_values = np.abs(
            pipeline.named_steps["classifier"].coef_[0]
        )
    else:
        importance_values = (
            pipeline.named_steps["classifier"].feature_importances_
        )

    for feature, importance in zip(
        transformed_features, importance_values
    ):
        importance_results.append({
            "Model": model_name,
            "Feature": feature,
            "Importance": importance
        })

    # Save trained model.
    model_path = MODEL_DIR / f"{safe_name}.joblib"
    joblib.dump(pipeline, model_path)

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"PHA Recall:        {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"PR-AUC:            {pr_auc:.4f}")
    print(f"Training time:     {training_time:.3f} seconds")
    print(f"Prediction time:   {prediction_time:.3f} seconds")
    print(f"Mean confidence:   {confidence:.4f}")
    print(f"Confusion matrix:  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Saved model:       {model_path}")

# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

metrics_df = pd.DataFrame(results).sort_values(
    by="PHA_Recall", ascending=False
)

confusion_df = pd.DataFrame(confusion_results)

importance_df = pd.DataFrame(importance_results)

importance_df = importance_df.sort_values(
    ["Model", "Importance"],
    ascending=[True, False]
)

metrics_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv", index=False
)

confusion_df.to_csv(
    OUTPUT_DIR / "confusion_matrices.csv", index=False
)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv", index=False
)

test_predictions.to_csv(
    OUTPUT_DIR / "test_predictions.csv", index=True
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(metrics_df.round(4).to_string(index=False))

print("\nSaved output files:")
print("- output/model_comparison.csv")
print("- output/confusion_matrices.csv")
print("- output/feature_importance.csv")
print("- output/test_predictions.csv")
print("- models/ (four trained model files)")

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
