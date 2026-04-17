import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.scoring import (
    DEFAULT_SCORING_CATALOG_PATH,
    load_scoring_catalog,
    score_seed_candidates,
)


def _build_ranked_catalog_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": [
                "seed_track",
                "near_match",
                "same_genre_mid",
                "cross_genre_close",
                "far_match",
            ],
            "artist_name": [
                "Artist Seed",
                "Artist Near",
                "Artist Mid",
                "Artist Cross",
                "Artist Far",
            ],
            "genre": ["pop", "pop", "pop", "rock", "pop"],
            "mood": ["calm", "calm", "driving", "calm", "calm"],
            "energy": [0.50, 0.52, 0.55, 0.51, 0.90],
            "tempo": [120, 118, 132, 121, 178],
            "energy_normalized": [0.50, 0.52, 0.55, 0.51, 0.90],
            "tempo_normalized": [0.50, 0.48, 0.60, 0.49, 0.95],
        }
    )


def test_load_scoring_catalog_uses_processed_schema() -> None:
    catalog = load_scoring_catalog()

    assert DEFAULT_SCORING_CATALOG_PATH.is_absolute()
    assert not catalog.empty
    assert {
        "track_id",
        "artist_name",
        "genre",
        "mood",
        "energy_normalized",
        "tempo_normalized",
    }.issubset(catalog.columns)


def test_score_seed_candidates_is_deterministic_and_excludes_seed() -> None:
    catalog = load_scoring_catalog()
    seed_track_id = catalog.iloc[0]["track_id"]

    first = score_seed_candidates(seed_track_id, catalog=catalog)
    second = score_seed_candidates(seed_track_id, catalog=catalog)

    assert_frame_equal(first, second)
    assert seed_track_id not in first["track_id"].tolist()
    assert not first.empty
    assert first["score"].is_monotonic_decreasing


def test_score_seed_candidates_prefers_same_genre_and_closer_features() -> None:
    catalog = _build_ranked_catalog_fixture()

    scored = score_seed_candidates("seed_track", catalog=catalog)

    assert scored["track_id"].tolist() == [
        "near_match",
        "same_genre_mid",
        "cross_genre_close",
        "far_match",
    ]
    assert {"numeric_similarity", "genre_bonus", "mood_bonus", "score"} <= set(
        scored.columns
    )
    assert scored.iloc[0]["genre_bonus"] > scored.iloc[2]["genre_bonus"]
    assert scored.iloc[0]["numeric_similarity"] > scored.iloc[-1]["numeric_similarity"]
