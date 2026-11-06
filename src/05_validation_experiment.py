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

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
MODELS = ROOT / "models"

DATA_PATH = OUTPUT / "asteroid_cleaned.csv"

RANDOM_STATE = 42

FEATURES = [
    "diameter", "albedo", "diameter_sigma",
    "e", "a", "q", "i", "om", "w", "ma",
    "ad", "n", "per", "per_y",
    "sigma_e", "sigma_a", "sigma_q", "sigma_i",
    "sigma_om", "sigma_w", "sigma_ma",
    "sigma_ad", "sigma_n", "sigma_per"
]


def evaluate(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "PHA_Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "PR_AUC": average_precision_score(y_true, probabilities),
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    }


def main():
    OUTPUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    print("=" * 70)
    print("VALIDATION-BASED ASTEROID CLASSIFICATION EXPERIMENT")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()
    y = df["pha"].astype(int)

    # First split: 70% training, 30% temporary.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # Second split: divide temporary data equally.
    # Final proportions: 70% train, 15% validation, 15% test.
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    print("\nDATA SPLIT")
    print("-" * 50)
    print(f"Training:   {len(X_train):,} ({len(X_train)/len(X):.0%})")
    print(f"Validation: {len(X_val):,} ({len(X_val)/len(X):.0%})")
    print(f"Testing:    {len(X_test):,} ({len(X_test)/len(X):.0%})")

    print("\nPHA samples:")
    print(f"Training:   {(y_train == 1).sum():,}")
    print(f"Validation: {(y_val == 1).sum():,}")
    print(f"Testing:    {(y_test == 1).sum():,}")

    positive_weight = (
        (y_train == 0).sum() / (y_train == 1).sum()
    )

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
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

    validation_results = []
    test_results = []
    prediction_data = pd.DataFrame({
        "actual": y_test.to_numpy()
    })

    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    for name, classifier in models.items():
        print("\n" + "=" * 70)
        print(f"MODEL: {name}")
        print("=" * 70)

        steps = [
            ("imputer", SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True
            ))
        ]

        if name == "Logistic Regression":
            steps.append(("scaler", StandardScaler()))

        steps.append(("classifier", classifier))
        pipeline = Pipeline(steps)

        # Train only on training data.
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_time = time.perf_counter() - start

        # Select threshold using validation data only.
        val_probabilities = pipeline.predict_proba(X_val)[:, 1]

        threshold_scores = []

        for threshold in thresholds:
            predictions = (val_probabilities >= threshold).astype(int)

            threshold_scores.append({
                "Threshold": threshold,
                "F1_Score": f1_score(
                    y_val, predictions, zero_division=0
                )
            })

        threshold_df = pd.DataFrame(threshold_scores)
        best_row = threshold_df.loc[
            threshold_df["F1_Score"].idxmax()
        ]
        best_threshold = float(best_row["Threshold"])

        validation_results.append({
            "Model": name,
            "Selected_Threshold": best_threshold,
            "Validation_F1": best_row["F1_Score"]
        })

        # Evaluate exactly once on untouched test data.
        start = time.perf_counter()
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
        prediction_time = time.perf_counter() - start

        metrics = evaluate(
            y_test,
            test_probabilities,
            best_threshold
        )

        metrics["Model"] = name
        metrics["Training_Time_sec"] = training_time
        metrics["Prediction_Time_sec"] = prediction_time

        test_results.append(metrics)

        safe_name = name.lower().replace(" ", "_")
        prediction_data[f"{safe_name}_probability"] = test_probabilities
        prediction_data[f"{safe_name}_prediction"] = (
            test_probabilities >= best_threshold
        ).astype(int)

        joblib.dump(
            pipeline,
            MODELS / f"{safe_name}_validation.joblib"
        )

        print(f"Selected threshold: {best_threshold:.2f}")
        print(f"Validation F1:      {best_row['F1_Score']:.4f}")
        print(f"Test precision:     {metrics['Precision']:.4f}")
        print(f"Test PHA recall:    {metrics['PHA_Recall']:.4f}")
        print(f"Test F1:            {metrics['F1_Score']:.4f}")
        print(f"Test PR-AUC:        {metrics['PR_AUC']:.4f}")
        print(f"Training time:      {training_time:.3f} sec")

    validation_df = pd.DataFrame(validation_results)
    test_df = pd.DataFrame(test_results)

    validation_df.to_csv(
        OUTPUT / "validation_thresholds.csv", index=False
    )

    test_df.to_csv(
        OUTPUT / "final_test_comparison.csv", index=False
    )

    prediction_data.to_csv(
        OUTPUT / "final_test_predictions.csv", index=False
    )

    print("\n" + "=" * 70)
    print("FINAL UNTOUCHED TEST-SET COMPARISON")
    print("=" * 70)
    print(test_df.round(4).to_string(index=False))

    print("\nSaved files:")
    print("- output/validation_thresholds.csv")
    print("- output/final_test_comparison.csv")
    print("- output/final_test_predictions.csv")
    print("- models/*_validation.joblib")

    print("\nVALIDATION EXPERIMENT COMPLETED!")


if __name__ == "__main__":
    main()
