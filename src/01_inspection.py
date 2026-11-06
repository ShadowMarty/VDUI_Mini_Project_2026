import pandas as pd
from config import DATA_PATH, OUTPUT_DIR
from common import load_target_name, ensure_dirs

def main():
    ensure_dirs()
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the Kaggle CSV and place it at dataset/dataset.csv."
        )

    print("=" * 70)
    print("ASTEROID DATASET INSPECTION")
    print("=" * 70)

    header = pd.read_csv(DATA_PATH, nrows=0)
    columns = header.columns.tolist()
    target = load_target_name(columns)

    print(f"\nFile: {DATA_PATH}")
    print(f"Columns: {len(columns)}")
    print("Column names:")
    for col in columns:
        print(f" - {col}")

    sample = pd.read_csv(DATA_PATH, nrows=10, low_memory=False)
    print("\nFirst 10 rows:")
    print(sample.to_string(index=False))
    print("\nSample data types:")
    print(sample.dtypes.to_string())

    rows = 0
    missing = pd.Series(0, index=columns, dtype="int64")
    target_counts = {}
    for chunk in pd.read_csv(DATA_PATH, chunksize=100_000, low_memory=False):
        rows += len(chunk)
        missing += chunk.isna().sum()
        counts = chunk[target].astype("string").fillna("<MISSING>").value_counts()
        for label, count in counts.items():
            target_counts[str(label)] = target_counts.get(str(label), 0) + int(count)
        print(f"Scanned {rows:,} rows")

    missing_report = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_percent": (missing.values / max(rows, 1)) * 100
    }).sort_values("missing_count", ascending=False)
    missing_report.to_csv(OUTPUT_DIR / "missing_values_report.csv", index=False)
    sample.to_csv(OUTPUT_DIR / "sample_records.csv", index=False)

    with open(OUTPUT_DIR / "inspection_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Dataset file: {DATA_PATH.name}\n")
        f.write(f"Total rows: {rows}\n")
        f.write(f"Total columns: {len(columns)}\n")
        f.write(f"Target column: {target}\n")
        f.write(f"Target raw value counts: {target_counts}\n")
        f.write("\nColumns:\n" + "\n".join(map(str, columns)))
        f.write("\n\nMissing values:\n" + missing_report.to_string(index=False))

    print("\nTarget column:", target)
    print("Target raw distribution:", target_counts)
    print("\nMissing-value report saved to output/missing_values_report.csv")
    print("Inspection summary saved to output/inspection_summary.txt")

if __name__ == "__main__":
    main()
