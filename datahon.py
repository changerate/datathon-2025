from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "clean" / "step1"
SAMPLE_PATH = CLEAN_DIR / "battles_cleaned_sample.parquet"
SUMMARY_PATH = CLEAN_DIR / "battles_cleaned_summary.json"
PLOT_DIR = BASE_DIR / "outputs" / "figures"


def load_sample() -> pd.DataFrame:
    """Load the cleaned step-one sample dataset created by scripts/clean_data.py."""
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned sample not found at {SAMPLE_PATH}. "
            "Run scripts/clean_data.py to generate it."
        )
    return pd.read_parquet(SAMPLE_PATH)


def print_summary() -> None:
    """Pretty-print key aggregate metrics stored in the summary JSON."""
    if not SUMMARY_PATH.exists():
        print("Summary file not found; skipping aggregate stats.")
        return

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    keys: Iterable[str] = [
        "raw_rows",
        "clean_rows",
        "sample_rows",
        "average_startingTrophies_mean",
        "winner_elixir_mean",
        "loser_elixir_mean",
        "battle_time_min",
        "battle_time_max",
    ]
    print("\nDataset summary (from battles_cleaned_summary.json):")
    for key in keys:
        value = summary.get(key, "<missing>")
        print(f"  {key}: {value}")


def explore_sample(df: pd.DataFrame) -> None:
    """Run a handful of baseline pandas inspections on the cleaned sample."""
    print(f"Loaded cleaned sample: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nSample records:")
    print(df.head())

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        print("\nNumeric summary (first 10 numeric columns):")
        first_cols = numeric_cols[:10]
        print(df[first_cols].describe().transpose())
    else:
        print("\nNo numeric columns detected in sample.")

    print("\nMissing values (top 10 columns by count):")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing.head(10))


def main() -> None:
    """Entrypoint: load cleaned data, inspect it, and print summary stats."""
    sample_df = load_sample()
    explore_sample(sample_df)
    print_summary()


if __name__ == "__main__":
    main()