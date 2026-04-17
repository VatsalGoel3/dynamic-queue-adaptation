from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.preprocess import DEFAULT_PROCESSED_CATALOG_PATH, load_processed_catalog

NUMERIC_FEATURE_COLUMNS = ["energy_normalized", "tempo_normalized"]
REQUIRED_SCORING_COLUMNS = {
    "track_id",
    "genre",
    "mood",
    *NUMERIC_FEATURE_COLUMNS,
}
DEFAULT_SCORING_CATALOG_PATH = DEFAULT_PROCESSED_CATALOG_PATH
GENRE_MATCH_BONUS = 0.15
MOOD_MATCH_BONUS = 0.05


def load_scoring_catalog(
    input_path: Path = DEFAULT_SCORING_CATALOG_PATH,
) -> pd.DataFrame:
    """Load the processed catalog used by the baseline scoring module."""
    catalog = load_processed_catalog(input_path)
    missing_columns = REQUIRED_SCORING_COLUMNS.difference(catalog.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"processed catalog is missing required columns: {missing}")
    return catalog


def score_seed_candidates(
    seed_track_id: str,
    catalog: pd.DataFrame | None = None,
    catalog_path: Path = DEFAULT_SCORING_CATALOG_PATH,
) -> pd.DataFrame:
    """Rank catalog candidates by interpretable similarity to a seed track."""
    scoring_catalog = (
        load_scoring_catalog(catalog_path) if catalog is None else catalog.copy()
    )
    missing_columns = REQUIRED_SCORING_COLUMNS.difference(scoring_catalog.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"catalog is missing required columns: {missing}")

    seed_rows = scoring_catalog.loc[scoring_catalog["track_id"] == seed_track_id]
    if seed_rows.empty:
        raise ValueError(f"unknown seed track_id: {seed_track_id}")

    seed_row = seed_rows.iloc[0]
    candidates = scoring_catalog.loc[
        scoring_catalog["track_id"] != seed_track_id
    ].copy()

    numeric_distances = candidates[NUMERIC_FEATURE_COLUMNS].sub(
        seed_row[NUMERIC_FEATURE_COLUMNS], axis="columns"
    ).abs()
    candidates["numeric_similarity"] = (1.0 - numeric_distances.mean(axis=1)).round(6)
    candidates["genre_bonus"] = (
        (candidates["genre"] == seed_row["genre"]).astype(float) * GENRE_MATCH_BONUS
    ).round(6)
    candidates["mood_bonus"] = (
        (candidates["mood"] == seed_row["mood"]).astype(float) * MOOD_MATCH_BONUS
    ).round(6)
    candidates["score"] = (
        candidates["numeric_similarity"]
        + candidates["genre_bonus"]
        + candidates["mood_bonus"]
    ).round(6)

    ranked_candidates = candidates.sort_values(
        ["score", "numeric_similarity", "track_id"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return ranked_candidates.reset_index(drop=True)
