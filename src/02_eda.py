import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PATH, OUTPUT_DIR, MAX_ROWS
from common import load_target_name, normalize_target, ensure_dirs

def main():
    ensure_dirs()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    header = pd.read_csv(DATA_PATH, nrows=0)
    target = load_target_name(header.columns)

    df = pd.read_csv(DATA_PATH, low_memory=False, nrows=MAX_ROWS)
    df[target] = normalize_target(df[target])
    df = df.dropna(subset=[target]).copy()
    df[target] = df[target].astype(int)

    sns.set_theme()

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=target)
    plt.title(f"Target distribution: {target} (0 = non-hazardous, 1 = hazardous)")
    plt.xlabel("Target class")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pha_distribution.png", dpi=160)
    plt.close()

    numeric = df.select_dtypes(include="number").copy()
    numeric = numeric.drop(columns=[target], errors="ignore")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]

    # Keep plots readable; select up to 8 numeric columns with the fewest missing values.
    selected = numeric.isna().mean().sort_values().head(8).index.tolist()
    if selected:
        plot_df = df[selected].replace([float("inf"), float("-inf")], pd.NA)
        plot_df = plot_df.apply(pd.to_numeric, errors="coerce")
        plot_df = plot_df.sample(min(len(plot_df), 20_000), random_state=42)
        axes = plot_df.hist(figsize=(14, 10), bins=35)
        plt.suptitle("Distributions of selected numeric features")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "numeric_feature_distributions.png", dpi=160)
        plt.close()

    corr_cols = selected + ([target] if target in df.columns else [])
    corr = df[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
    plt.figure(figsize=(11, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation heatmap of selected numeric features and target")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    print("EDA completed. Charts saved in output/.")
    print("Rows used:", len(df))
    print("Target distribution:\n", df[target].value_counts())

if __name__ == "__main__":
    main()
