import numpy as np
import pandas as pd

from config import DATA_PATH, OUTPUT_DIR
from common import load_target_name, ensure_dirs

# ==========================================
# 1. CONFIGURATION
# ==========================================
FEATURES = [
    "diameter",
    "albedo",
    "diameter_sigma",
    "e",
    "a",
    "q",
    "i",
    "om",
    "w",
    "ma",
    "ad",
    "n",
    "per",
    "per_y",
    "sigma_e",
    "sigma_a",
    "sigma_q",
    "sigma_i",
    "sigma_om",
    "sigma_w",
    "sigma_ma",
    "sigma_ad",
    "sigma_n",
    "sigma_per",
]

TARGET = "pha"

OUTPUT_FILE = OUTPUT_DIR / "asteroid_cleaned.csv"
MISSING_FILE = OUTPUT_DIR / "preprocessing_missing_values.csv"

CHUNK_SIZE = 100000


# ==========================================
# 2. TARGET ENCODING
# ==========================================
def encode_target(series):
    mapping = {
        "Y": 1,
        "N": 0,
        "YES": 1,
        "NO": 0,
        "TRUE": 1,
        "FALSE": 0,
        "1": 1,
        "0": 0,
    }

    values = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .map(mapping)
    )

    return values


# ==========================================
# 3. PREPROCESSING
# ==========================================
def main():
    ensure_dirs()

    print("=" * 65)
    print("ASTEROID DATA PREPROCESSING")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    header = pd.read_csv(DATA_PATH, nrows=0)

    target_column = load_target_name(header.columns)

    missing_features = [
        col for col in FEATURES
        if col not in header.columns
    ]

    if missing_features:
        raise ValueError(
            f"Features not found in dataset: {missing_features}"
        )

    selected_columns = FEATURES + [target_column]

    # Remove previous output
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total_rows = 0
    retained_rows = 0
    removed_rows = 0

    target_counts = {0: 0, 1: 0}

    missing_counts = pd.Series(
        0, index=FEATURES, dtype="int64"
    )

    first_chunk = True

    # ======================================
    # 4. PROCESS DATA IN CHUNKS
    # ======================================
    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=selected_columns,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        total_rows += len(chunk)

        chunk = chunk.rename(
            columns={target_column: TARGET}
        )

        # Convert target labels
        chunk[TARGET] = encode_target(chunk[TARGET])

        # Rows without known target cannot be supervised training records
        before = len(chunk)

        chunk = chunk.dropna(
            subset=[TARGET]
        ).copy()

        removed_rows += before - len(chunk)

        chunk[TARGET] = chunk[TARGET].astype(int)

        # Convert all selected features to numeric
        for feature in FEATURES:
            chunk[feature] = pd.to_numeric(
                chunk[feature],
                errors="coerce",
            )

        # Replace infinite values
        chunk[FEATURES] = chunk[FEATURES].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        missing_counts += chunk[FEATURES].isna().sum()

        # Record class distribution
        counts = chunk[TARGET].value_counts()

        target_counts[0] += int(counts.get(0, 0))
        target_counts[1] += int(counts.get(1, 0))

        # Save without filling missing feature values.
        # Median imputation will be fitted on training data only.
        chunk.to_csv(
            OUTPUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False

        retained_rows += len(chunk)

        print(
            f"Processed: {total_rows:,} | "
            f"Retained: {retained_rows:,}"
        )

    # ======================================
    # 5. SAVE REPORTS
    # ======================================
    missing_report = pd.DataFrame({
        "Feature": FEATURES,
        "Missing_Count": missing_counts.values,
        "Missing_Percentage": (
            missing_counts.values /
            max(retained_rows, 1) * 100
        ),
    })

    missing_report.to_csv(
        MISSING_FILE,
        index=False,
    )

    print("\nPREPROCESSING SUMMARY")
    print("-" * 45)

    print("Original records:", f"{total_rows:,}")
    print("Removed unlabeled records:", f"{removed_rows:,}")
    print("Retained records:", f"{retained_rows:,}")

    print("\nEncoded target distribution:")

    print("Non-PHA (0):", f"{target_counts[0]:,}")
    print("PHA (1):", f"{target_counts[1]:,}")

    print("\nFeature missing values:")
    print(missing_report.to_string(index=False))

    print("\nSaved cleaned dataset:")
    print(OUTPUT_FILE)

    print("\nSaved missing-value report:")
    print(MISSING_FILE)

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
