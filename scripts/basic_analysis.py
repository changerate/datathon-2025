from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_SAMPLE = BASE_DIR / "data" / "clean" / "step1" / "battles_cleaned_sample.parquet"
OUTPUT_DIR = BASE_DIR / "outputs" / "basics"
PLOTS_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"


def load_sample() -> pd.DataFrame:
    """Load the cleaned step-one sample parquet into a DataFrame."""
    if not CLEAN_SAMPLE.exists():
        raise FileNotFoundError(
            f"Sample dataset not found at {CLEAN_SAMPLE}. "
            "Run scripts/clean_data.py to regenerate Step 1 outputs."
        )
    return pd.read_parquet(CLEAN_SAMPLE)


def ensure_output_dirs() -> None:
    """Ensure the figure and table output directories exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def compute_level_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Plot and export stats about total card level differences."""
    df = df.copy()
    df["total_level_diff"] = (
        df["winner.totalcard.level"] - df["loser.totalcard.level"]
    )
    df["level_advantage"] = np.select(
        [df["total_level_diff"] > 0, df["total_level_diff"] < 0],
        ["winner_higher", "winner_lower"],
        default="level_draw",
    )

    # Histogram bins (match the plot with 40 bins)
    hist_counts, bin_edges = np.histogram(df["total_level_diff"], bins=40)
    hist_df = pd.DataFrame(
        {
            "bin_start": bin_edges[:-1],
            "bin_end": bin_edges[1:],
            "count": hist_counts,
        }
    )
    hist_df.to_csv(TABLES_DIR / "deck_level_difference_histogram.csv", index=False)

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df["total_level_diff"],
        bins=40,
        kde=False,
        color="#4C72B0",
    )
    plt.title("Deck Level Difference (Winner - Loser)")
    plt.xlabel("Total card level difference")
    plt.ylabel("Match count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "deck_level_difference_hist.png", dpi=150)
    plt.close()
    return df


def compute_elixir_stats(df: pd.DataFrame) -> None:
    """Plot and export the distribution of average elixir differences."""
    df = df.copy()
    df["elixir_diff"] = df["winner.elixir.average"] - df["loser.elixir.average"]

    # Save histogram data for the elixir difference plot
    hist_counts, bin_edges = np.histogram(df["elixir_diff"], bins=40)
    elixir_hist = pd.DataFrame(
        {
            "bin_start": bin_edges[:-1],
            "bin_end": bin_edges[1:],
            "count": hist_counts,
        }
    )
    elixir_hist.to_csv(TABLES_DIR / "elixir_difference_histogram.csv", index=False)

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(
        df["elixir_diff"],
        bins=40,
        color="#8172B2",
        kde=False,
    )
    plt.title("Elixir Difference (Winner - Loser)")
    plt.xlabel("Difference in average elixir")
    plt.ylabel("Match count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "elixir_difference_hist.png", dpi=150)
    plt.close()


def compute_rarity_mix(df: pd.DataFrame) -> None:
    """Summarize and visualize rarity count differences between decks."""
    rarity_cols = [
        "common.count",
        "rare.count",
        "epic.count",
        "legendary.count",
    ]
    records = []
    for rarity in rarity_cols:
        win_col = f"winner.{rarity}"
        lose_col = f"loser.{rarity}"
        diff_col = f"{rarity}_diff"

        if win_col in df.columns and lose_col in df.columns:
            diff = df[win_col] - df[lose_col]
            records.append(
                {
                    "rarity": rarity.replace(".count", ""),
                    "advantage_rate": (diff > 0).mean(),
                    "disadvantage_rate": (diff < 0).mean(),
                    "draw_rate": (diff == 0).mean(),
                    "mean_diff": diff.mean(),
                }
            )
    rarity_df = pd.DataFrame(records)
    rarity_df.to_csv(TABLES_DIR / "rarity_mix_summary.csv", index=False)

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=rarity_df, x="rarity", y="mean_diff", color="#4C72B0")
    plt.title("Average Rarity Difference (Winner - Loser)")
    plt.xlabel("Rarity")
    plt.ylabel("Mean card count difference")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rarity_mean_diff_bar.png", dpi=150)
    plt.close()


def compute_overlap_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute shared-card counts between decks and plot their distribution."""
    def overlap_count(row: pd.Series) -> int:
        return len(set(row["winner.cards.list"]) & set(row["loser.cards.list"]))

    df = df.copy()
    df["deck_overlap"] = df.apply(overlap_count, axis=1)
    overlap_distribution = df["deck_overlap"].value_counts().sort_index()
    overlap_distribution.to_csv(TABLES_DIR / "deck_overlap_distribution.csv")

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=overlap_distribution.index,
        y=overlap_distribution.values,
        color="#55A868",
    )
    plt.title("Deck Overlap (Shared Cards) Distribution")
    plt.xlabel("Number of shared cards (0-8)")
    plt.ylabel("Match count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "deck_overlap_distribution.png", dpi=150)
    plt.close()
    return df


def compute_trophy_gap(df: pd.DataFrame) -> None:
    """Aggregate upset rates across trophy gap bins and visualize the results."""
    df = df.copy()
    df["trophy_gap"] = df["winner.startingTrophies"] - df["loser.startingTrophies"]
    df["upset"] = df["trophy_gap"] < 0

    gap_bins = pd.cut(
        df["trophy_gap"],
        bins=[-np.inf, -200, -100, -50, 0, 50, 100, 200, np.inf],
    )
    upset_rate = df.groupby(gap_bins, observed=False)["upset"].mean().reset_index()
    upset_rate.to_csv(TABLES_DIR / "upset_rate_by_trophy_gap.csv", index=False)

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=upset_rate, x="trophy_gap", y="upset", color="#C44E52")
    plt.title("Upset Rate by Trophy Gap")
    plt.xlabel("Winner - Loser starting trophies (binned)")
    plt.ylabel("Upset rate (winner had fewer trophies)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "upset_rate_by_trophy_gap.png", dpi=150)
    plt.close()


def compute_crown_margin(df: pd.DataFrame) -> None:
    """Count crown margins and plot the distribution of win decisiveness."""
    df = df.copy()
    df["crown_margin"] = df["winner.crowns"] - df["loser.crowns"]

    margin_counts = df["crown_margin"].value_counts().sort_index()
    margin_counts.to_csv(TABLES_DIR / "crown_margin_distribution.csv")

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=margin_counts.index.astype(int),
        y=margin_counts.values,
        color="#8172B2",
    )
    plt.title("Crown Margin Distribution (Winner - Loser)")
    plt.xlabel("Crown margin")
    plt.ylabel("Match count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "crown_margin_distribution.png", dpi=150)
    plt.close()


def compute_composition_stats(df: pd.DataFrame) -> None:
    """Compare deck composition counts (troop/structure/spell) between winner and loser."""
    components = ["troop.count", "structure.count", "spell.count"]
    records = []
    for component in components:
        win_col = f"winner.{component}"
        lose_col = f"loser.{component}"
        diff = df[win_col] - df[lose_col]
        records.append(
            {
                "component": component.replace(".count", ""),
                "advantage_rate": (diff > 0).mean(),
                "disadvantage_rate": (diff < 0).mean(),
                "draw_rate": (diff == 0).mean(),
                "mean_diff": diff.mean(),
            }
        )
    composition_df = pd.DataFrame(records)
    composition_df.to_csv(TABLES_DIR / "deck_composition_summary.csv", index=False)

    sns.set_style("whitegrid")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=composition_df, x="component", y="mean_diff", color="#937860")
    plt.title("Average Deck Composition Difference (Winner - Loser)")
    plt.xlabel("Component")
    plt.ylabel("Mean count difference")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "deck_composition_mean_diff.png", dpi=150)
    plt.close()


def main() -> None:
    """Run the basic exploratory analysis pipeline and produce summary outputs."""
    ensure_output_dirs()
    df = load_sample()

    df_with_levels = compute_level_stats(df)
    compute_elixir_stats(df_with_levels)
    compute_rarity_mix(df_with_levels)
    df_with_overlap = compute_overlap_stats(df_with_levels)
    compute_trophy_gap(df_with_overlap)
    compute_crown_margin(df_with_overlap)
    compute_composition_stats(df_with_overlap)


if __name__ == "__main__":
    main()

