from pathlib import Path

# Project root is the parent folder of src/
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "dataset" / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# Set to a column name if automatic detection does not find your target.
# Supported common names: PHA, is_hazardous.
TARGET_COLUMN = None

# Common asteroid identifiers / free-text columns that should not be model inputs.
DROP_COLUMNS = {
    "name", "neo_id", "id", "object_id", "orbit_id",
    "designation", "full_name", "sentry_object"
}

# Potentially target-defining fields. Review these against your dataset's
# documentation before changing this list.
POTENTIAL_LEAKAGE_COLUMNS = {
    "moid", "moid_ld", "h", "absolute_magnitude"
}

TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_ROWS = None  # Set e.g. 200000 for a quick development run; None uses all rows.
LOW_CONFIDENCE_THRESHOLD = 0.60
