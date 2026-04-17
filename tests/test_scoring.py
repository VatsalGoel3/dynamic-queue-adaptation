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


def test_score_seed_candidates_uses_track_id_as_deterministic_tie_breaker() -> None:
    catalog = pd.DataFrame(
        {
            "track_id": ["seed_track", "candidate_b", "candidate_a"],
            "artist_name": ["Artist Seed", "Artist B", "Artist A"],
            "genre": ["pop", "rock", "rock"],
            "mood": ["calm", "driving", "driving"],
            "energy_normalized": [0.50, 0.60, 0.60],
            "tempo_normalized": [0.50, 0.40, 0.40],
        }
    )

    scored = score_seed_candidates("seed_track", catalog=catalog)

    assert scored["track_id"].tolist() == ["candidate_a", "candidate_b"]
    assert scored.iloc[0]["score"] == scored.iloc[1]["score"]
    assert scored.iloc[0]["numeric_similarity"] == scored.iloc[1]["numeric_similarity"]
    assert scored.iloc[0]["genre_bonus"] == scored.iloc[1]["genre_bonus"]
    assert scored.iloc[0]["mood_bonus"] == scored.iloc[1]["mood_bonus"]


def test_score_seed_candidates_exposes_mood_bonus_contribution() -> None:
    catalog = pd.DataFrame(
        {
            "track_id": ["seed_track", "mood_match", "mood_miss"],
            "artist_name": ["Artist Seed", "Artist Match", "Artist Miss"],
            "genre": ["pop", "pop", "pop"],
            "mood": ["calm", "calm", "driving"],
            "energy_normalized": [0.50, 0.55, 0.55],
            "tempo_normalized": [0.50, 0.45, 0.45],
        }
    )

    scored = score_seed_candidates("seed_track", catalog=catalog).set_index("track_id")

    assert scored.loc["mood_match", "numeric_similarity"] == scored.loc[
        "mood_miss", "numeric_similarity"
    ]
    assert scored.loc["mood_match", "genre_bonus"] == scored.loc["mood_miss", "genre_bonus"]
    assert scored.loc["mood_match", "mood_bonus"] > scored.loc["mood_miss", "mood_bonus"]
    assert scored.loc["mood_match", "score"] > scored.loc["mood_miss", "score"]


def test_score_seed_candidates_does_not_require_artist_name_column() -> None:
    catalog = _build_ranked_catalog_fixture().drop(columns=["artist_name"])

    scored = score_seed_candidates("seed_track", catalog=catalog)

    assert not scored.empty
