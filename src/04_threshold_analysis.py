from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"

PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

MODEL_COLUMNS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost"
}

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def main():
    print("=" * 65)
    print("ASTEROID CLASSIFICATION - THRESHOLD ANALYSIS")
    print("=" * 65)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "test_predictions.csv not found. Run 03_train_models.py first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    y_true = df["actual"].astype(int)
    results = []

    for model_name, prefix in MODEL_COLUMNS.items():
        probability_column = f"{prefix}_pha_probability"

        if probability_column not in df.columns:
            print(f"Skipping {model_name}: probability column missing.")
            continue

        probabilities = df[probability_column].to_numpy()

        for threshold in THRESHOLDS:
            predictions = (probabilities >= threshold).astype(int)

            precision = precision_score(
                y_true, predictions, zero_division=0
            )
            recall = recall_score(
                y_true, predictions, zero_division=0
            )
            f1 = f1_score(
                y_true, predictions, zero_division=0
            )

            tn, fp, fn, tp = confusion_matrix(
                y_true, predictions, labels=[0, 1]
            ).ravel()

            results.append({
                "Model": model_name,
                "Threshold": threshold,
                "Precision": precision,
                "PHA_Recall": recall,
                "F1_Score": f1,
                "True_Negatives": tn,
                "False_Positives": fp,
                "False_Negatives": fn,
                "True_Positives": tp
            })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "threshold_analysis.csv",
        index=False
    )

    # ------------------------------------------------
    # Plot 1: Precision and recall against threshold
    # ------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for model_name in MODEL_COLUMNS:
        model_data = results_df[
            results_df["Model"] == model_name
        ]

        axes[0].plot(
            model_data["Threshold"],
            model_data["PHA_Recall"],
            label=model_name
        )

        axes[1].plot(
            model_data["Threshold"],
            model_data["Precision"],
            label=model_name
        )

    axes[0].set_title("PHA Recall vs Classification Threshold")
    axes[0].set_xlabel("Classification Threshold")
    axes[0].set_ylabel("PHA Recall")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title("Precision vs Classification Threshold")
    axes[1].set_xlabel("Classification Threshold")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "threshold_precision_recall.png",
        dpi=200
    )
    plt.close()

    # ------------------------------------------------
    # Plot 2: F1 score against threshold
    # ------------------------------------------------

    plt.figure(figsize=(10, 6))

    for model_name in MODEL_COLUMNS:
        model_data = results_df[
            results_df["Model"] == model_name
        ]

        plt.plot(
            model_data["Threshold"],
            model_data["F1_Score"],
            label=model_name
        )

    plt.title("F1 Score vs Classification Threshold")
    plt.xlabel("Classification Threshold")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "threshold_f1_comparison.png",
        dpi=200
    )
    plt.close()

    # ------------------------------------------------
    # Print best F1 threshold for each model
    # ------------------------------------------------

    print("\nBEST F1 THRESHOLD PER MODEL")
    print("-" * 65)

    for model_name in MODEL_COLUMNS:
        model_data = results_df[
            results_df["Model"] == model_name
        ]

        if model_data.empty:
            continue

        best = model_data.loc[
            model_data["F1_Score"].idxmax()
        ]

        print(f"\n{model_name}")
        print(f"Best threshold: {best['Threshold']:.2f}")
        print(f"Precision:     {best['Precision']:.4f}")
        print(f"PHA Recall:    {best['PHA_Recall']:.4f}")
        print(f"F1 Score:      {best['F1_Score']:.4f}")
        print(f"False alarms:  {int(best['False_Positives'])}")
        print(f"Missed PHAs:   {int(best['False_Negatives'])}")

    print("\nSaved:")
    print("- output/threshold_analysis.csv")
    print("- output/threshold_precision_recall.png")
    print("- output/threshold_f1_comparison.png")

    print("\nTHRESHOLD ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
