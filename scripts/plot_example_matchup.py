from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .card_map import CARD_MAP  # type: ignore
except ImportError:  # pragma: no cover
    from card_map import CARD_MAP  # type: ignore

try:
    from .train_baseline_model import (  # type: ignore
        build_pipeline,
        get_feature_sets,
        load_features,
        split_train_test,
    )
except ImportError:  # pragma: no cover
    from train_baseline_model import (
        build_pipeline,
        get_feature_sets,
        load_features,
        split_train_test,
    )

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features_with_synergy.parquet"
OUTPUT_FIGURE = BASE_DIR / "outputs" / "figures" / "example_matchup_prediction.png"
OUTPUT_TABLE = BASE_DIR / "outputs" / "basics" / "tables" / "example_matchup_prediction.csv"


def extract_decks(row: pd.Series) -> tuple[list[str], list[str]]:
    card_ids = sorted(CARD_MAP.keys())
    deck_a_cols = [f"deck_a_card_{cid}" for cid in card_ids]
    deck_b_cols = [f"deck_b_card_{cid}" for cid in card_ids]

    def to_names(columns: list[str]) -> list[str]:
        ids = [cid for cid, col in zip(card_ids, columns) if row.get(col, 0) == 1]
        return [CARD_MAP[cid] for cid in ids]

    return to_names(deck_a_cols), to_names(deck_b_cols)


def choose_confident_example(df: pd.DataFrame, proba: np.ndarray, threshold: float = 0.8) -> pd.Series:
    df = df.copy()
    df["pred_proba"] = proba
    high_conf = df[(df["pred_proba"] >= threshold) | (df["pred_proba"] <= 1 - threshold)]
    if high_conf.empty:
        return df.sort_values("pred_proba", ascending=False).iloc[0]
    return high_conf.sort_values("pred_proba", ascending=False, key=lambda s: np.abs(s - 0.5)).iloc[-1]


def plot_example(prob_a: float, actual_label: int) -> None:
    expected = "Deck A" if prob_a >= 0.5 else "Deck B"
    confidence = max(prob_a, 1 - prob_a)
    actual = "Deck A won" if actual_label == 1 else "Deck B won"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Deck A", "Deck B"], [prob_a, 1 - prob_a], color=["#4C72B0", "#C44E52"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted win probability")
    ax.set_title(
        f"Expected winner: {expected} ({confidence:.1%} confidence)\n"
        f"Actual outcome: {actual}"
    )
    for x, val in enumerate([prob_a, 1 - prob_a]):
        ax.text(x, val + 0.02, f"{val:.1%}", ha="center")
    plt.tight_layout()
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FIGURE, dpi=150)
    plt.close()
    print(f"Saved example matchup prediction to {OUTPUT_FIGURE}")


def main() -> None:
    df = load_features(FEATURES_PATH)
    train_df, test_df = split_train_test(df)

    X_train, y_train = get_feature_sets(train_df)
    X_test, y_test = get_feature_sets(test_df)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    example = choose_confident_example(test_df, probabilities, threshold=0.8)

    prob_a = float(example["pred_proba"])
    actual_label = int(example["label_deck_a_wins"])
    deck_a, deck_b = extract_decks(example)

    plot_example(prob_a, actual_label)

    # Save tabular summary
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        {
            "deck": "Deck A",
            "predicted_probability": prob_a,
            "cards": ", ".join(deck_a),
        },
        {
            "deck": "Deck B",
            "predicted_probability": 1 - prob_a,
            "cards": ", ".join(deck_b),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df["expected_winner"] = "Deck A" if prob_a >= 0.5 else "Deck B"
    summary_df["confidence"] = max(prob_a, 1 - prob_a)
    summary_df["actual_winner"] = "Deck A" if actual_label == 1 else "Deck B"
    summary_df.to_csv(OUTPUT_TABLE, index=False)

    print(f"Saved matchup details to {OUTPUT_TABLE}")
    print("Deck A cards:", summary_rows[0]["cards"])
    print("Deck B cards:", summary_rows[1]["cards"])
    print(f"Predicted P(Deck A wins): {prob_a:.3f}")


if __name__ == "__main__":
    main()

