<div align="center">

# Royale Match Predictor
### Data Royale — November 2025

</div>

## Overview

We analyze 16.8 million Clash Royale battles (Dec 7–27, 2020) from players with 4,000+ trophies to understand what drives match outcomes.  
Our pipeline cleans the raw CSV, engineers per-deck matchup features, learns card synergy & counter relationships, and trains predictive models that output “betting odds” for any pair of decks.

Key highlights:

- **Cleaning & sampling** – stream the 9.2 GB CSV, filter to competitive ladder matches, and build a 500k-match sample (mirrored to 1M rows for modeling).
- **Baseline features** – trophy/level/elixir differences, deck composition & rarity counts, deck overlap, and card one-hot indicators.
- **Synergy & counters** – data-driven win rates for every card pair (within decks) and every cross-deck pairing, summarized as deck-level synergy/counter advantages.
- **Models** – logistic regression trained on baseline features (AUC 0.674) and on the synergy-augmented set (AUC 0.739, accuracy 67.5%).
- **Visual artifacts** – EDA plots, top synergy charts, role-based (win condition vs support) pairings, and an example matchup probability graphic.

## Repository layout

```text
├── data/
│   ├── clean/step1/
│   │   ├── battles_cleaned_sample.parquet
│   │   └── battles_cleaned_summary.json
│   └── processed/                # feature matrices, synergy stats, metrics
├── outputs/
│   ├── basics/figures/           # EDA plots used in slides
│   ├── basics/tables/            # CSV summaries + matchup odds table
│   └── figures/                  # top synergy charts, matchup prediction plot
├── scripts/                      # cleaning, feature engineering, modeling, plotting
└── datahon.py                    # quick dataset summary CLI
```

> **Large raw files are excluded.**  
> Recreate them using `scripts/clean_data.py` if you have access to the original `battles.csv`.

## Getting started

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt # create as needed
```

1. Place the raw `battles.csv` (9.2 GB) at repo root.
2. Run `python scripts/clean_data.py` to produce cleaned parquet partitions and the 500k-row sample.
3. Build baseline features: `python scripts/build_baseline_features.py`
4. Learn synergy/counter stats: `python scripts/build_synergy_counters.py`
5. Augment features: `python scripts/augment_synergy_features.py`
6. Train models (baseline or synergy): `python scripts/train_baseline_model.py --features-path ...`
7. Optional visualizations:
   - `python scripts/basic_analysis.py`
   - `python scripts/analyze_synergy_roles.py`
   - `python scripts/plot_top_synergies.py`
   - `python scripts/plot_example_matchup.py`

## Slide assets

All charts referenced in the presentation live under `outputs/`.  
Example matchup probabilities are stored as both a PNG and CSV at:

- `outputs/figures/example_matchup_prediction.png`
- `outputs/basics/tables/example_matchup_prediction.csv`

Run `python scripts/plot_example_matchup.py` to refresh them with a new high-confidence matchup.

## License

This code is provided for the Data Royale 2025 datathon.  
Battle logs belong to Supercell; do not redistribute the raw dataset.
