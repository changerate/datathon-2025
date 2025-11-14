from __future__ import annotations

from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

try:
    from .card_map import CARD_MAP  # type: ignore
except ImportError:  # pragma: no cover
    from card_map import CARD_MAP  # type: ignore

BASE_DIR = Path(__file__).resolve().parent.parent
SYNERGY_PATH = BASE_DIR / "data" / "processed" / "synergy_stats.parquet"
OUTPUT_DIR = BASE_DIR / "outputs" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "top_10_synergies.png"


def load_synergy(min_games: int) -> pd.DataFrame:
    if not SYNERGY_PATH.exists():
        raise FileNotFoundError(
            f"Synergy stats not found at {SYNERGY_PATH}. "
            "Run scripts/build_synergy_counters.py first."
        )
    df = pd.read_parquet(SYNERGY_PATH)
    df["card_a_name"] = df["card_a"].map(CARD_MAP)
    df["card_b_name"] = df["card_b"].map(CARD_MAP)
    df = df[df["total"] >= min_games].copy()
    df["pair"] = df["card_a_name"] + " + " + df["card_b_name"]
    return df


def plot_top_synergy(df: pd.DataFrame, top_n: int) -> None:
    top_df = df.sort_values("smoothed_win_rate", ascending=False).head(top_n)
    if top_df.empty:
        raise ValueError(
            "No synergy pairs met the minimum games threshold. "
            "Try lowering min_games."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=top_df,
        x="smoothed_win_rate",
        y="pair",
        color="#4C72B0",
    )
    plt.xlabel("Smoothed win rate when paired")
    plt.ylabel("Card pair")
    plt.title("Top Card Synergies")
    plt.xlim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()
    print(f"Top synergy plot saved to {OUTPUT_PATH}")


def main(top_n: int = 10, min_games: int = 500) -> None:
    df = load_synergy(min_games)
    plot_top_synergy(df, top_n)


if __name__ == "__main__":
    main()


