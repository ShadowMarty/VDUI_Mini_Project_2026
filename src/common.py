import pandas as pd
from config import DATA_PATH, OUTPUT_DIR, TARGET_COLUMN

def load_target_name(columns):
    if TARGET_COLUMN:
        if TARGET_COLUMN not in columns:
            raise ValueError(
                f"TARGET_COLUMN='{TARGET_COLUMN}' was not found. "
                f"Available columns: {list(columns)}"
            )
        return TARGET_COLUMN

    lookup = {str(c).strip().lower(): c for c in columns}
    for candidate in ("pha", "is_hazardous", "potentially_hazardous"):
        if candidate in lookup:
            return lookup[candidate]

    raise ValueError(
        "Could not automatically identify the target column. "
        "Set TARGET_COLUMN in src/config.py to the exact target column name."
    )

def normalize_target(series):
    """Convert common binary labels to 0/1, with 1 meaning hazardous."""
    s = series.copy()
    if pd.api.types.is_bool_dtype(s):
        return s.astype("Int64")

    # Normalize text labels without changing the original dataframe.
    text = s.astype("string").str.strip().str.lower()
    positive = {"1", "1.0", "y", "yes", "true", "t", "hazardous", "pha"}
    negative = {"0", "0.0", "n", "no", "false", "f", "non-hazardous", "nonhazardous", "non_pha"}

    mapped = text.map(
        lambda x: 1 if x in positive else (0 if x in negative else pd.NA)
    )

    # If values are numeric but represented in a less common form, try numeric conversion.
    if mapped.notna().sum() == 0:
        numeric = pd.to_numeric(s, errors="coerce")
        unique = set(numeric.dropna().unique().tolist())
        if unique.issubset({0, 1}):
            mapped = numeric.astype("Int64")

    if mapped.notna().sum() == 0:
        raise ValueError(
            "Target labels could not be mapped to binary 0/1. "
            f"Observed values: {s.dropna().astype(str).unique()[:20].tolist()}"
        )

    return mapped.astype("Int64")

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
