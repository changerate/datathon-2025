from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES_PATH = BASE_DIR / "data" / "processed" / "baseline_features.parquet"
DEFAULT_METRICS_PATH = BASE_DIR / "data" / "processed" / "baseline_model_metrics.json"
TRAIN_FRACTION = 0.8
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train logistic regression on feature matrix.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to feature parquet (default: baseline_features.parquet)",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Where to save metrics JSON",
    )
    return parser.parse_args()


def load_features(path: Path) -> pd.DataFrame:
    """Load the baseline feature matrix and ensure datetime ordering."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {path}. "
            "Generate feature parquet before training."
        )
    df = pd.read_parquet(path)
    if "battle_time" in df.columns:
        df["battle_time"] = pd.to_datetime(df["battle_time"], errors="coerce", utc=True)
        df = df.sort_values("battle_time")
    else:
        df = df.sample(frac=1.0, random_state=RANDOM_STATE)
    return df


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a temporal split (80/20 by default)."""
    split_idx = int(len(df) * TRAIN_FRACTION)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def get_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target, dropping non-feature columns."""
    X = df.drop(columns=["label_deck_a_wins"])
    y = df["label_deck_a_wins"].astype(int)
    if "battle_time" in X.columns:
        X = X.drop(columns=["battle_time"])
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Create a preprocessing + logistic regression pipeline."""
    card_cols = [
        col for col in X.columns if col.startswith("deck_a_card_") or col.startswith("deck_b_card_")
    ]
    numeric_cols = [col for col in X.columns if col not in card_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="passthrough",
    )

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return pipeline


def evaluate(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Compute core classification metrics."""
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    metrics = {
        "auc": roc_auc_score(y, proba),
        "log_loss": log_loss(y, proba),
        "brier": brier_score_loss(y, proba),
        "accuracy": (preds == y).mean(),
    }
    return metrics


def save_metrics(
    train_metrics: dict[str, float], test_metrics: dict[str, float], metrics_path: Path
) -> None:
    """Write metric summaries to disk."""
    payload = {
        "train": train_metrics,
        "test": test_metrics,
        "train_fraction": TRAIN_FRACTION,
        "random_state": RANDOM_STATE,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = load_features(args.features_path)
    train_df, test_df = split_train_test(df)

    X_train, y_train = get_feature_sets(train_df)
    X_test, y_test = get_feature_sets(test_df)

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    train_metrics = evaluate(pipeline, X_train, y_train)
    test_metrics = evaluate(pipeline, X_test, y_test)

    save_metrics(train_metrics, test_metrics, args.metrics_path)

    print("Baseline logistic regression metrics:")
    print("  Train:", train_metrics)
    print("  Test :", test_metrics)


if __name__ == "__main__":
    main()

