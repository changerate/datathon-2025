from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from .card_map import CARD_MAP  # type: ignore
except ImportError:
    from card_map import CARD_MAP  # type: ignore  # imported for completeness


BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_SAMPLE = BASE_DIR / "data" / "clean" / "step1" / "battles_cleaned_sample.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
SYNERGY_PATH = OUTPUT_DIR / "synergy_stats.parquet"
COUNTER_PATH = OUTPUT_DIR / "counter_stats.parquet"

SYNERGY_ALPHA = 1.0
SYNERGY_BETA = 1.0
COUNTER_ALPHA = 1.0
COUNTER_BETA = 1.0
PROGRESS_EVERY = 50_000


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sample() -> pd.DataFrame:
    if not CLEAN_SAMPLE.exists():
        raise FileNotFoundError(
            f"Sample dataset not found at {CLEAN_SAMPLE}. "
            "Run scripts/clean_data.py to regenerate Step 1 outputs."
        )
    df = pd.read_parquet(CLEAN_SAMPLE)
    return df


def to_card_list(value: object) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, np.ndarray):
        return [int(x) for x in value.tolist()]
    return []


def accumulate_stats(
    df: pd.DataFrame,
) -> tuple[Dict[Tuple[int, int], list[int]], Dict[Tuple[int, int], list[int]]]:
    synergy_counts: Dict[Tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    counter_counts: Dict[Tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])

    winner_lists = df["winner.cards.list"].tolist()
    loser_lists = df["loser.cards.list"].tolist()

    for idx, (winner_raw, loser_raw) in enumerate(zip(winner_lists, loser_lists)):
        winner_cards = to_card_list(winner_raw)
        loser_cards = to_card_list(loser_raw)

        if not winner_cards or not loser_cards:
            continue

        for combo in combinations(sorted(set(winner_cards)), 2):
            synergy_counts[combo][0] += 1
            synergy_counts[combo][1] += 1
        for combo in combinations(sorted(set(loser_cards)), 2):
            synergy_counts[combo][1] += 1

        for win_card in winner_cards:
            for lose_card in loser_cards:
                counter_counts[(win_card, lose_card)][0] += 1
                counter_counts[(win_card, lose_card)][1] += 1
                counter_counts[(lose_card, win_card)][1] += 1

        if PROGRESS_EVERY and (idx + 1) % PROGRESS_EVERY == 0:
            print(f"Processed {idx + 1:,} matches...", flush=True)

    return synergy_counts, counter_counts


def build_synergy_df(counts: Dict[Tuple[int, int], list[int]]) -> pd.DataFrame:
    records = []
    for (card_a, card_b), (wins, total) in counts.items():
        smoothed = (wins + SYNERGY_ALPHA) / (total + SYNERGY_ALPHA + SYNERGY_BETA)
        records.append(
            {
                "card_a": card_a,
                "card_b": card_b,
                "wins": wins,
                "total": total,
                "smoothed_win_rate": smoothed,
            }
        )
    df = pd.DataFrame(
        records,
        columns=["card_a", "card_b", "wins", "total", "smoothed_win_rate"],
    )
    df = df.sort_values(["card_a", "card_b"]).reset_index(drop=True)
    return df


def build_counter_df(counts: Dict[Tuple[int, int], list[int]]) -> pd.DataFrame:
    records = []
    for (card_a, card_b), (wins, total) in counts.items():
        smoothed = (wins + COUNTER_ALPHA) / (total + COUNTER_ALPHA + COUNTER_BETA)
        records.append(
            {
                "card_a": card_a,
                "card_b": card_b,
                "wins": wins,
                "total": total,
                "smoothed_win_rate": smoothed,
            }
        )
    df = pd.DataFrame(
        records,
        columns=["card_a", "card_b", "wins", "total", "smoothed_win_rate"],
    )
    df = df.sort_values(["card_a", "card_b"]).reset_index(drop=True)
    return df


def main() -> None:
    ensure_output_dir()
    df = load_sample()
    synergy_counts, counter_counts = accumulate_stats(df)

    synergy_df = build_synergy_df(synergy_counts)
    counter_df = build_counter_df(counter_counts)

    synergy_df.to_parquet(SYNERGY_PATH, index=False)
    counter_df.to_parquet(COUNTER_PATH, index=False)

    print(
        f"Synergy pairs: {len(synergy_df)} stored at {SYNERGY_PATH}\n"
        f"Counter pairs: {len(counter_df)} stored at {COUNTER_PATH}"
    )


if __name__ == "__main__":
    main()

