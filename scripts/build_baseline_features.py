from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from .card_map import CARD_MAP  # type: ignore
except ImportError:
    from card_map import CARD_MAP


BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_SAMPLE = BASE_DIR / "data" / "clean" / "step1" / "battles_cleaned_sample.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
FEATURES_PATH = OUTPUT_DIR / "baseline_features.parquet"
METADATA_PATH = OUTPUT_DIR / "baseline_features_meta.json"
CARD_FREQ_PATH = OUTPUT_DIR / "card_frequencies.csv"

RANDOM_SEED = 42


@dataclass(frozen=True)
class DeckSnapshot:
    cards: list[int]
    starting_trophies: float
    trophy_change: float
    crowns: float
    king_hp: float
    total_level: int
    elixir: float
    troop_count: int
    structure_count: int
    spell_count: int
    common_count: int
    rare_count: int
    epic_count: int
    legendary_count: int


def load_sample() -> pd.DataFrame:
    """Load the cleaned sample parquet into memory."""
    if not CLEAN_SAMPLE.exists():
        raise FileNotFoundError(
            f"Sample dataset not found at {CLEAN_SAMPLE}. "
            "Run scripts/clean_data.py to regenerate Step 1 outputs."
        )
    return pd.read_parquet(CLEAN_SAMPLE)


def ensure_output_dir() -> None:
    """Create the processed-data directory if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_deck(row: pd.Series, prefix: str) -> DeckSnapshot:
    """Collect core deck attributes for the given prefix ('winner' or 'loser')."""
    cards = row.get(f"{prefix}.cards.list", [])
    if isinstance(cards, (list, tuple)):
        card_ids = [int(card) for card in cards]
    elif isinstance(cards, np.ndarray):
        card_ids = [int(card) for card in cards.tolist()]
    else:
        card_ids = []
    return DeckSnapshot(
        cards=card_ids,
        starting_trophies=float(row.get(f"{prefix}.startingTrophies", 0.0)),
        trophy_change=float(row.get(f"{prefix}.trophyChange", 0.0)),
        crowns=float(row.get(f"{prefix}.crowns", 0.0)),
        king_hp=float(row.get(f"{prefix}.kingTowerHitPoints", 0.0)),
        total_level=int(row.get(f"{prefix}.totalcard.level", 0)),
        elixir=float(row.get(f"{prefix}.elixir.average", 0.0)),
        troop_count=int(row.get(f"{prefix}.troop.count", 0)),
        structure_count=int(row.get(f"{prefix}.structure.count", 0)),
        spell_count=int(row.get(f"{prefix}.spell.count", 0)),
        common_count=int(row.get(f"{prefix}.common.count", 0)),
        rare_count=int(row.get(f"{prefix}.rare.count", 0)),
        epic_count=int(row.get(f"{prefix}.epic.count", 0)),
        legendary_count=int(row.get(f"{prefix}.legendary.count", 0)),
    )


def compute_card_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-card appearance counts across both decks."""
    all_cards: list[int] = []
    for column in ("winner.cards.list", "loser.cards.list"):
        cards_series = df[column].dropna()
        for cards in cards_series:
            if isinstance(cards, (list, tuple)):
                all_cards.extend(int(card) for card in cards)
            elif isinstance(cards, np.ndarray):
                all_cards.extend(int(card) for card in cards.tolist())
    counts = pd.Series(all_cards, dtype="int64").value_counts().sort_index()
    freq_df = counts.rename_axis("card_id").reset_index(name="count")
    freq_df["card_name"] = freq_df["card_id"].map(CARD_MAP)
    return freq_df


def make_record(
    deck_a: DeckSnapshot,
    deck_b: DeckSnapshot,
    label: int,
    meta: dict[str, object],
    sorted_card_ids: list[int],
) -> dict[str, float | int | str | object]:
    deck_a_cards = set(deck_a.cards)
    deck_b_cards = set(deck_b.cards)

    record: dict[str, float | int | str | object] = {
        "battle_time": meta.get("battleTime"),
        "arena_id": meta.get("arena.id"),
        "game_mode_id": meta.get("gameMode.id"),
        "label_deck_a_wins": label,
        "total_level_diff": deck_a.total_level - deck_b.total_level,
        "trophy_diff": deck_a.starting_trophies - deck_b.starting_trophies,
        "trophy_abs_diff": abs(deck_a.starting_trophies - deck_b.starting_trophies),
        "elixir_diff": deck_a.elixir - deck_b.elixir,
        "elixir_abs_diff": abs(deck_a.elixir - deck_b.elixir),
        "elixir_mean": (deck_a.elixir + deck_b.elixir) / 2.0,
        "troop_diff": deck_a.troop_count - deck_b.troop_count,
        "structure_diff": deck_a.structure_count - deck_b.structure_count,
        "spell_diff": deck_a.spell_count - deck_b.spell_count,
        "common_diff": deck_a.common_count - deck_b.common_count,
        "rare_diff": deck_a.rare_count - deck_b.rare_count,
        "epic_diff": deck_a.epic_count - deck_b.epic_count,
        "legendary_diff": deck_a.legendary_count - deck_b.legendary_count,
        "deck_overlap": len(deck_a_cards & deck_b_cards),
    }

    for card_id in sorted_card_ids:
        record[f"deck_a_card_{card_id}"] = int(card_id in deck_a_cards)
        record[f"deck_b_card_{card_id}"] = int(card_id in deck_b_cards)

    return record


def build_feature_rows(
    df: pd.DataFrame, card_ids: Iterable[int], progress_every: int = 50_000
) -> pd.DataFrame:
    """Construct the baseline feature matrix with mirrored samples."""
    records: list[dict[str, float | int | str | object]] = []
    sorted_card_ids = sorted(card_ids)

    for idx in range(len(df)):
        row = df.iloc[idx]
        winner_deck = extract_deck(row, "winner")
        loser_deck = extract_deck(row, "loser")

        meta = {
            "battleTime": row.get("battleTime"),
            "arena.id": row.get("arena.id"),
            "gameMode.id": row.get("gameMode.id"),
        }

        records.append(
            make_record(winner_deck, loser_deck, 1, meta, sorted_card_ids)
        )
        records.append(
            make_record(loser_deck, winner_deck, 0, meta, sorted_card_ids)
        )

        if progress_every and (idx + 1) % progress_every == 0:
            print(f"Processed {idx + 1:,} matches...", flush=True)

    features = pd.DataFrame.from_records(records)
    bool_columns = [col for col in features.columns if col.startswith("deck_a_card_") or col.startswith("deck_b_card_")]
    features[bool_columns] = features[bool_columns].astype("uint8")
    return features


def save_metadata(card_frequencies: pd.DataFrame, features: pd.DataFrame) -> None:
    """Persist metadata about the generated baseline feature matrix."""
    metadata = {
        "rows": int(features.shape[0]),
        "columns": int(features.shape[1]),
        "card_count": int(len(CARD_MAP)),
        "card_feature_prefixes": {
            "deck_a": "deck_a_card_",
            "deck_b": "deck_b_card_",
        },
        "random_seed": RANDOM_SEED,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    card_frequencies.to_csv(CARD_FREQ_PATH, index=False)


def main() -> None:
    """Generate the baseline feature matrix and supporting metadata."""
    ensure_output_dir()
    df = load_sample()

    card_frequencies = compute_card_frequencies(df)
    features = build_feature_rows(df, CARD_MAP.keys())

    features.to_parquet(FEATURES_PATH, index=False)
    save_metadata(card_frequencies, features)
    print(f"Baseline features written to {FEATURES_PATH} ({features.shape[0]} rows, {features.shape[1]} columns).")


if __name__ == "__main__":
    main()

