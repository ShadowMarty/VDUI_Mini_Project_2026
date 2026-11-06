from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
MODELS = ROOT / "models"

DATA_PATH = OUTPUT / "asteroid_cleaned.csv"
MODEL_PATH = MODELS / "xgboost_validation.joblib"

FEATURES = [
    "diameter", "albedo", "diameter_sigma",
    "e", "a", "q", "i", "om", "w", "ma",
    "ad", "n", "per", "per_y",
    "sigma_e", "sigma_a", "sigma_q", "sigma_i",
    "sigma_om", "sigma_w", "sigma_ma",
    "sigma_ad", "sigma_n", "sigma_per"
]

RANDOM_STATE = 42


def main():
    print("=" * 65)
    print("XGBOOST MODEL EXPLAINABILITY USING SHAP")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Validation XGBoost model not found. "
            "Run 05_validation_experiment.py first."
        )

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].copy()
    y = df["pha"].astype(int)

    # Reproduce the same 70/15/15 split used during training.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )

    print(f"\nTest-set records: {len(X_test):,}")

    # Use equal numbers of PHA and non-PHA examples.
    # This allows us to examine explanations for both classes.
    positive_indices = y_test[y_test == 1].index
    negative_indices = y_test[y_test == 0].index
    n_per_class = min(
        500,
        len(positive_indices),
        len(negative_indices)
    )
    selected_positive = X_test.loc[positive_indices].sample(
        n=n_per_class,
        random_state=RANDOM_STATE
    )
    selected_negative = X_test.loc[negative_indices].sample(
        n=n_per_class,
        random_state=RANDOM_STATE
    )
    sample = pd.concat([
        selected_positive,
        selected_negative
    ]).sample(frac=1, random_state=RANDOM_STATE)
    sample_labels = y_test.loc[sample.index]

    print(f"Balanced SHAP sample: {len(sample):,}")
    print(f"PHA examples: {(sample_labels == 1).sum():,}")
    print(f"Non-PHA examples: {(sample_labels == 0).sum():,}")

    pipeline = joblib.load(MODEL_PATH)

    imputer = pipeline.named_steps["imputer"]
    classifier = pipeline.named_steps["classifier"]

    transformed = imputer.transform(sample)

    feature_names = imputer.get_feature_names_out(FEATURES)

    transformed_df = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=sample.index
    )

    print("\nCalculating SHAP values...")

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed_df)

    # Handle different SHAP output formats.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    print("SHAP calculation completed.")

    # --------------------------------------------------
    # 1. Global feature importance
    # --------------------------------------------------

    mean_importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean_Absolute_SHAP": mean_importance
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    importance_df.to_csv(
        OUTPUT / "xgboost_shap_importance.csv",
        index=False
    )

    print("\nTOP 15 FEATURES")
    print("-" * 55)
    print(importance_df.head(15).to_string(index=False))

    # --------------------------------------------------
    # 2. SHAP summary bar plot
    # --------------------------------------------------

    shap.summary_plot(
        shap_values,
        transformed_df,
        plot_type="bar",
        max_display=15,
        show=False
    )

    plt.title("XGBoost - Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(
        OUTPUT / "shap_feature_importance.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 3. SHAP beeswarm plot
    # --------------------------------------------------

    shap.summary_plot(
        shap_values,
        transformed_df,
        max_display=15,
        show=False
    )

    plt.title("XGBoost - SHAP Feature Impact")
    plt.tight_layout()
    plt.savefig(
        OUTPUT / "shap_summary.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # --------------------------------------------------
    # 4. Individual prediction explanation data
    # --------------------------------------------------

    individual_explanations = pd.DataFrame(
        shap_values,
        columns=feature_names,
        index=sample.index
    )

    individual_explanations.insert(
        0, "Actual_PHA", sample_labels
    )

    individual_explanations.to_csv(
        OUTPUT / "shap_individual_explanations.csv",
        index=True
    )

    print("\nSaved files:")
    print("- output/xgboost_shap_importance.csv")
    print("- output/shap_feature_importance.png")
    print("- output/shap_summary.png")
    print("- output/shap_individual_explanations.csv")

    print("\nSHAP EXPLAINABILITY ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
