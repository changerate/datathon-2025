from __future__ import annotations

import ast
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import logging
import numpy as np
import pandas as pd


RAW_DATA_PATH = Path("battles.csv")
OUTPUT_DIR = Path("data/clean/step1")
PARTITION_DIR = OUTPUT_DIR / "partitions"
SUMMARY_PATH = OUTPUT_DIR / "battles_cleaned_summary.json"
SAMPLE_PATH = OUTPUT_DIR / "battles_cleaned_sample.parquet"

CHUNK_SIZE = 100_000
SAMPLE_TARGET = 500_000
RANDOM_SEED = 42
LOG_EVERY_N = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("prepare_step1")

DROP_COLUMNS = [
    "Unnamed: 0",
    "tournamentTag",
    "winner.tag",
    "loser.tag",
    "winner.clan.tag",
    "winner.clan.badgeId",
    "loser.clan.tag",
    "loser.clan.badgeId",
    "winner.princessTowersHitPoints",
    "loser.princessTowersHitPoints",
]

INT_COLUMNS = [
    "arena.id",
    "gameMode.id",
    "winner.card1.id",
    "winner.card2.id",
    "winner.card3.id",
    "winner.card4.id",
    "winner.card5.id",
    "winner.card6.id",
    "winner.card7.id",
    "winner.card8.id",
    "winner.card1.level",
    "winner.card2.level",
    "winner.card3.level",
    "winner.card4.level",
    "winner.card5.level",
    "winner.card6.level",
    "winner.card7.level",
    "winner.card8.level",
    "winner.totalcard.level",
    "winner.troop.count",
    "winner.structure.count",
    "winner.spell.count",
    "winner.common.count",
    "winner.rare.count",
    "winner.epic.count",
    "winner.legendary.count",
    "loser.card1.id",
    "loser.card2.id",
    "loser.card3.id",
    "loser.card4.id",
    "loser.card5.id",
    "loser.card6.id",
    "loser.card7.id",
    "loser.card8.id",
    "loser.card1.level",
    "loser.card2.level",
    "loser.card3.level",
    "loser.card4.level",
    "loser.card5.level",
    "loser.card6.level",
    "loser.card7.level",
    "loser.card8.level",
    "loser.totalcard.level",
    "loser.troop.count",
    "loser.structure.count",
    "loser.spell.count",
    "loser.common.count",
    "loser.rare.count",
    "loser.epic.count",
    "loser.legendary.count",
]


def ensure_directories() -> None:
    """Create output directories for cleaned data and partitions."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTITION_DIR.mkdir(parents=True, exist_ok=True)


def parse_card_list(value: Any) -> list[int]:
    """Normalize the raw card list field into a list of integer card IDs."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
        if isinstance(parsed, (list, tuple)):
            return [int(x) for x in parsed]
    return []


def convert_int_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Cast selected columns to pandas nullable Int64 dtype."""
    for column in columns:
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .where(frame[column].notna(), other=pd.NA)
                .astype("Int64")
            )


def safe_min(current: datetime | None, candidate: pd.Timestamp | None) -> datetime | None:
    """Return the smaller datetime, handling missing pandas timestamps."""
    if pd.isna(candidate):
        return current
    candidate_dt = candidate.to_pydatetime()
    if current is None or candidate_dt < current:
        return candidate_dt
    return current


def safe_max(current: datetime | None, candidate: pd.Timestamp | None) -> datetime | None:
    """Return the larger datetime, handling missing pandas timestamps."""
    if pd.isna(candidate):
        return current
    candidate_dt = candidate.to_pydatetime()
    if current is None or candidate_dt > current:
        return candidate_dt
    return current


def main() -> None:
    """Stream the raw CSV, clean/filter rows, and write parquet partitions plus summary."""
    ensure_directories()

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_DATA_PATH}")

    logger.info("Starting Step 1 cleaning for %s", RAW_DATA_PATH)
    logger.info(
        "Chunk size: %s | Sample target: %s rows | Output dir: %s",
        CHUNK_SIZE,
        SAMPLE_TARGET,
        OUTPUT_DIR,
    )

    stats: dict[str, Any] = {
        "raw_rows": 0,
        "filtered_rows": 0,
        "clean_rows": 0,
        "average_startingTrophies_sum": 0.0,
        "winner_elixir_sum": 0.0,
        "loser_elixir_sum": 0.0,
        "battle_time_min": None,
        "battle_time_max": None,
        "arena_counts": Counter(),
        "game_mode_counts": Counter(),
        "partitions": [],
    }

    sample_frames: list[pd.DataFrame] = []
    sampled_rows = 0
    rng = np.random.default_rng(RANDOM_SEED)

    for part_idx, chunk in enumerate(
        pd.read_csv(RAW_DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False)
    ):
        if part_idx % LOG_EVERY_N == 0:
            logger.info("Chunk %d: loaded %d raw rows", part_idx, len(chunk))
        raw_len = len(chunk)
        stats["raw_rows"] += raw_len

        chunk = chunk.loc[chunk["average.startingTrophies"] >= 4000].copy()
        filtered_len = len(chunk)
        if filtered_len == 0:
            if part_idx % LOG_EVERY_N == 0:
                logger.info(
                    "Chunk %d: skipped (no rows with average.startingTrophies >= 4000)",
                    part_idx,
                )
            continue

        if part_idx % LOG_EVERY_N == 0:
            logger.info(
                "Chunk %d: %d rows kept after trophy filter",
                part_idx,
                filtered_len,
            )

        stats["filtered_rows"] += filtered_len

        # Parse card lists
        chunk["winner.cards.list"] = chunk["winner.cards.list"].apply(parse_card_list)
        chunk["loser.cards.list"] = chunk["loser.cards.list"].apply(parse_card_list)

        # Drop unused columns
        columns_to_drop = [col for col in DROP_COLUMNS if col in chunk.columns]
        if columns_to_drop:
            chunk = chunk.drop(columns=columns_to_drop)

        # Convert datatypes
        convert_int_columns(chunk, INT_COLUMNS)
        if "battleTime" in chunk.columns:
            chunk["battleTime"] = pd.to_datetime(
                chunk["battleTime"], errors="coerce", utc=True
            )

        # Update statistics
        stats["clean_rows"] += len(chunk)
        stats["average_startingTrophies_sum"] += chunk["average.startingTrophies"].sum()
        stats["winner_elixir_sum"] += chunk["winner.elixir.average"].sum()
        stats["loser_elixir_sum"] += chunk["loser.elixir.average"].sum()

        if "arena.id" in chunk.columns:
            stats["arena_counts"].update(
                chunk["arena.id"].dropna().astype(int).tolist()
            )
        if "gameMode.id" in chunk.columns:
            stats["game_mode_counts"].update(
                chunk["gameMode.id"].dropna().astype(int).tolist()
            )

        if "battleTime" in chunk.columns:
            stats["battle_time_min"] = safe_min(
                stats["battle_time_min"], chunk["battleTime"].min()
            )
            stats["battle_time_max"] = safe_max(
                stats["battle_time_max"], chunk["battleTime"].max()
            )

        # Write partition
        partition_path = PARTITION_DIR / f"battles_cleaned_part_{part_idx:04d}.parquet"
        chunk.to_parquet(partition_path, index=False)
        stats["partitions"].append(
            {"path": str(partition_path), "rows": len(chunk)}
        )
        if part_idx % LOG_EVERY_N == 0:
            logger.info(
                "Chunk %d: wrote partition %s (%d rows)",
                part_idx,
                partition_path.name,
                len(chunk),
            )

        # Build manageable sample (first N filtered rows)
        if sampled_rows < SAMPLE_TARGET:
            remaining = SAMPLE_TARGET - sampled_rows
            sample_slice = chunk.iloc[:remaining].copy()
            if not sample_slice.empty:
                # Shuffle within slice for a little variation
                sample_slice = sample_slice.sample(
                    frac=1.0, random_state=rng.integers(0, 2**32 - 1)
                ).reset_index(drop=True)
                sample_frames.append(sample_slice)
                sampled_rows += len(sample_slice)
                logger.info(
                    "Chunk %d: captured %d rows for sample (total sampled: %d)",
                    part_idx,
                    len(sample_slice),
                    sampled_rows,
                )

    if stats["clean_rows"] == 0:
        raise RuntimeError("No rows matched the 4000+ trophies filter.")

    # Save sample dataset
    if sample_frames:
        sample_df = pd.concat(sample_frames, ignore_index=True)
        sample_df.to_parquet(SAMPLE_PATH, index=False)
        logger.info(
            "Sample dataset saved to %s with %d rows",
            SAMPLE_PATH,
            sample_df.shape[0],
        )
    else:
        sample_df = pd.DataFrame()
        logger.warning("No rows collected for sample dataset")

    # Finalize summary
    stats["partitions_count"] = len(stats["partitions"])
    stats["sample_rows"] = int(sample_df.shape[0])
    stats["average_startingTrophies_mean"] = (
        stats["average_startingTrophies_sum"] / stats["clean_rows"]
    )
    stats["winner_elixir_mean"] = (
        stats["winner_elixir_sum"] / stats["clean_rows"]
    )
    stats["loser_elixir_mean"] = (
        stats["loser_elixir_sum"] / stats["clean_rows"]
    )

    stats["arena_top10"] = stats["arena_counts"].most_common(10)
    stats["game_mode_top10"] = stats["game_mode_counts"].most_common(10)

    # Replace datetime objects with ISO strings
    if stats["battle_time_min"] is not None:
        stats["battle_time_min"] = stats["battle_time_min"].isoformat()
    if stats["battle_time_max"] is not None:
        stats["battle_time_max"] = stats["battle_time_max"].isoformat()

    # Remove non-serializable counters
    stats.pop("arena_counts", None)
    stats.pop("game_mode_counts", None)

    with SUMMARY_PATH.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)
    logger.info("Summary stats written to %s", SUMMARY_PATH)

    logger.info(
        "Finished processing: %s clean rows across %s partitions. Sample rows: %s.",
        f"{stats['clean_rows']:,}",
        stats["partitions_count"],
        f"{stats['sample_rows']:,}",
    )


if __name__ == "__main__":
    main()

