from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

try:
    from .card_map import CARD_MAP  # type: ignore
except ImportError:  # pragma: no cover
    from card_map import CARD_MAP  # type: ignore


BASE_DIR = Path(__file__).resolve().parent.parent
SYNERGY_PATH = BASE_DIR / "data" / "processed" / "synergy_stats.parquet"
TABLE_DIR = BASE_DIR / "outputs" / "basics" / "tables"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
WIN_WIN_TABLE = TABLE_DIR / "top_win_condition_synergies.csv"
WIN_SUPPORT_TABLE = TABLE_DIR / "win_condition_support_synergies.csv"
WIN_WIN_FIGURE = FIGURE_DIR / "top_win_condition_synergies.png"

# Generous list of win conditions provided by LightningLegs
WIN_CONDITION_IDS: set[int] = {
    26000003,
    26000004,
    26000006,
    26000007,
    26000009,
    26000014,
    26000015,
    26000016,
    26000017,
    26000018,
    26000020,
    26000021,
    26000023,
    26000024,
    26000025,
    26000026,
    26000027,
    26000028,
    26000029,
    26000032,
    26000033,
    26000035,
    26000036,
    26000043,
    26000046,
    26000047,
    26000048,
    26000050,
    26000051,
    26000054,
    26000055,
    26000056,
    26000058,
    26000059,
    26000060,
    26000063,
    26000065,
    26000067,
    26000069,
    26000072,
    26000074,
    26000077,
    26000083,
    26000085,
    26000093,
    26000095,
    26000096,
    26000097,
    26000099,
    26000101,
    26000103,
    27000002,
    27000005,
    27000006,
    27000008,
    27000013,
    28000003,
    28000004,
    28000005,
    28000006,
    28000010,
    28000012,
    28000013,
    28000025,
}


def card_role(card_id: int) -> str:
    return "win_condition" if card_id in WIN_CONDITION_IDS else "support"


def load_synergy(min_games: int) -> pd.DataFrame:
    if not SYNERGY_PATH.exists():
        raise FileNotFoundError(
            f"Synergy stats not found at {SYNERGY_PATH}. "
            "Run scripts/build_synergy_counters.py first."
        )
    df = pd.read_parquet(SYNERGY_PATH)
    df = df[df["total"] >= min_games].copy()
    df["card_a_name"] = df["card_a"].map(CARD_MAP)
    df["card_b_name"] = df["card_b"].map(CARD_MAP)
    df["card_a_role"] = df["card_a"].apply(card_role)
    df["card_b_role"] = df["card_b"].apply(card_role)
    df["pair"] = df["card_a_name"] + " + " + df["card_b_name"]
    return df


def top_win_condition_pairs(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    mask = (df["card_a_role"] == "win_condition") & (df["card_b_role"] == "win_condition")
    win_df = df[mask].copy()
    win_df = win_df.sort_values("smoothed_win_rate", ascending=False).head(top_n)
    return win_df[
        [
            "card_a",
            "card_a_name",
            "card_b",
            "card_b_name",
            "total",
            "wins",
            "smoothed_win_rate",
        ]
    ]


def win_condition_support_pairs(
    df: pd.DataFrame, top_k_support: int
) -> pd.DataFrame:
    mask = (
        (df["card_a_role"] == "win_condition") & (df["card_b_role"] == "support")
    ) | (
        (df["card_b_role"] == "win_condition") & (df["card_a_role"] == "support")
    )
    ws_df = df[mask].copy()

    def normalize_row(row: pd.Series) -> pd.Series:
        if row["card_a_role"] == "win_condition":
            win_id, win_name = row["card_a"], row["card_a_name"]
            support_id, support_name = row["card_b"], row["card_b_name"]
        else:
            win_id, win_name = row["card_b"], row["card_b_name"]
            support_id, support_name = row["card_a"], row["card_a_name"]
        row["win_condition_id"] = win_id
        row["win_condition_name"] = win_name
        row["support_id"] = support_id
        row["support_name"] = support_name
        return row

    ws_df = ws_df.apply(normalize_row, axis=1)

    top_rows: list[pd.DataFrame] = []
    grouped = ws_df.groupby("win_condition_id", sort=False)
    for win_id, group in grouped:
        top_group = group.sort_values("smoothed_win_rate", ascending=False).head(top_k_support)
        top_rows.append(top_group)
    if not top_rows:
        return pd.DataFrame(
            columns=[
                "win_condition_id",
                "win_condition_name",
                "support_id",
                "support_name",
                "total",
                "wins",
                "smoothed_win_rate",
            ]
        )
    result = pd.concat(top_rows, ignore_index=True)
    return result[
        [
            "win_condition_id",
            "win_condition_name",
            "support_id",
            "support_name",
            "total",
            "wins",
            "smoothed_win_rate",
        ]
    ]


def plot_win_condition_pairs(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("No win-condition synergies available to plot.")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df,
        x="smoothed_win_rate",
        y=df["card_a_name"] + " + " + df["card_b_name"],
        color="#4C72B0",
    )
    plt.xlabel("Smoothed win rate when paired")
    plt.ylabel("Win-condition pair")
    plt.title("Top Win-Condition Synergies")
    plt.xlim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(WIN_WIN_FIGURE, dpi=150)
    plt.close()


def main(
    min_games: int = 200, top_win_pairs: int = 10, top_support_per_win: int = 5
) -> None:
    df = load_synergy(min_games)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    win_pairs = top_win_condition_pairs(df, top_win_pairs)
    win_pairs.to_csv(WIN_WIN_TABLE, index=False)
    plot_win_condition_pairs(win_pairs)
    print(f"Saved win-condition pair table to {WIN_WIN_TABLE}")
    print(f"Saved win-condition pair plot to {WIN_WIN_FIGURE}")

    win_support = win_condition_support_pairs(df, top_support_per_win)
    win_support.to_csv(WIN_SUPPORT_TABLE, index=False)
    print(f"Saved win-condition support table to {WIN_SUPPORT_TABLE}")


if __name__ == "__main__":
    main()

