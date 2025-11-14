from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from .card_map import CARD_MAP  # type: ignore
except ImportError:
    from card_map import CARD_MAP  # type: ignore


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_INPUT = BASE_DIR / "data" / "processed" / "baseline_features.parquet"
SYNERGY_PATH = BASE_DIR / "data" / "processed" / "synergy_stats.parquet"
COUNTER_PATH = BASE_DIR / "data" / "processed" / "counter_stats.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "features_with_synergy.parquet"
METADATA_PATH = BASE_DIR / "data" / "processed" / "features_with_synergy_meta.json"

PRIOR = 0.5
PROGRESS_EVERY = 50_000


def load_features() -> pd.DataFrame:
    if not FEATURES_INPUT.exists():
        raise FileNotFoundError(
            f"Baseline features not found at {FEATURES_INPUT}. "
            "Run scripts/build_baseline_features.py first."
        )
    return pd.read_parquet(FEATURES_INPUT)


def load_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SYNERGY_PATH.exists() or not COUNTER_PATH.exists():
        raise FileNotFoundError(
            "Synergy / counter stats missing. Run scripts/build_synergy_counters.py first."
        )
    synergy = pd.read_parquet(SYNERGY_PATH)
    counter = pd.read_parquet(COUNTER_PATH)
    return synergy, counter


def build_lookup(
    synergy: pd.DataFrame, counter: pd.DataFrame
) -> tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    synergy_lookup = {
        (int(row.card_a), int(row.card_b)): float(row.smoothed_win_rate)
        for row in synergy.itertuples(index=False)
    }
    counter_lookup = {
        (int(row.card_a), int(row.card_b)): float(row.smoothed_win_rate)
        for row in counter.itertuples(index=False)
    }
    return synergy_lookup, counter_lookup


def cards_from_row(row: np.ndarray, card_ids: np.ndarray) -> list[int]:
    indices = np.nonzero(row)[0]
    return card_ids[indices].tolist()


def compute_deck_synergy(cards: list[int], synergy_lookup: Dict[Tuple[int, int], float]) -> float:
    if len(cards) < 2:
        return PRIOR
    scores = []
    for combo in combinations(sorted(cards), 2):
        score = synergy_lookup.get(combo, PRIOR)
        scores.append(score)
    if not scores:
        return PRIOR
    return float(np.mean(scores))


def compute_counter_score(
    cards_a: list[int],
    cards_b: list[int],
    counter_lookup: Dict[Tuple[int, int], float],
) -> float:
    if not cards_a or not cards_b:
        return PRIOR
    scores = []
    for card_a, card_b in product(cards_a, cards_b):
        score = counter_lookup.get((card_a, card_b), PRIOR)
        scores.append(score)
    if not scores:
        return PRIOR
    return float(np.mean(scores))


def augment(features: pd.DataFrame, synergy_lookup, counter_lookup) -> pd.DataFrame:
    card_ids = np.array(sorted(CARD_MAP.keys()), dtype=int)
    deck_a_cols = [f"deck_a_card_{cid}" for cid in card_ids]
    deck_b_cols = [f"deck_b_card_{cid}" for cid in card_ids]

    deck_a_matrix = features[deck_a_cols].values.astype(np.uint8)
    deck_b_matrix = features[deck_b_cols].values.astype(np.uint8)

    deck_a_synergy = np.zeros(len(features), dtype=float)
    deck_b_synergy = np.zeros(len(features), dtype=float)
    deck_a_counter = np.zeros(len(features), dtype=float)
    deck_b_counter = np.zeros(len(features), dtype=float)

    for idx in range(len(features)):
        cards_a = cards_from_row(deck_a_matrix[idx], card_ids)
        cards_b = cards_from_row(deck_b_matrix[idx], card_ids)

        deck_a_synergy[idx] = compute_deck_synergy(cards_a, synergy_lookup)
        deck_b_synergy[idx] = compute_deck_synergy(cards_b, synergy_lookup)

        deck_a_counter[idx] = compute_counter_score(cards_a, cards_b, counter_lookup)
        deck_b_counter[idx] = compute_counter_score(cards_b, cards_a, counter_lookup)

        if PROGRESS_EVERY and (idx + 1) % PROGRESS_EVERY == 0:
            print(f"Augmented {idx + 1:,} rows...", flush=True)

    features = features.copy()
    features["deck_a_synergy_mean"] = deck_a_synergy
    features["deck_b_synergy_mean"] = deck_b_synergy
    features["deck_a_counter_mean"] = deck_a_counter
    features["deck_b_counter_mean"] = deck_b_counter
    features["counter_advantage_mean"] = deck_a_counter - deck_b_counter
    features["synergy_advantage_mean"] = deck_a_synergy - deck_b_synergy
    return features


def save_features(features: pd.DataFrame) -> None:
    features.to_parquet(OUTPUT_PATH, index=False)
    metadata = {
        "rows": int(features.shape[0]),
        "columns": int(features.shape[1]),
        "source": str(FEATURES_INPUT),
        "synergy_stats": str(SYNERGY_PATH),
        "counter_stats": str(COUNTER_PATH),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    features = load_features()
    synergy_df, counter_df = load_stats()
    synergy_lookup, counter_lookup = build_lookup(synergy_df, counter_df)
    augmented = augment(features, synergy_lookup, counter_lookup)
    save_features(augmented)
    print(
        f"Augmented features saved to {OUTPUT_PATH} "
        f"(columns: {augmented.shape[1]})"
    )


if __name__ == "__main__":
    main()

