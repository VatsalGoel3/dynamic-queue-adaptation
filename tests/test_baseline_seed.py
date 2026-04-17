import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_sessions import load_synthetic_sessions
from src.data.preprocess import load_processed_catalog
from src.models.baseline_seed import recommend_from_seed


def test_recommend_from_seed_returns_deterministic_top_k_without_duplicates() -> None:
    catalog = load_processed_catalog()
    seed_track_id = catalog.iloc[0]["track_id"]

    first = recommend_from_seed(seed_track_id, top_k=5, catalog=catalog)
    second = recommend_from_seed(seed_track_id, top_k=5, catalog=catalog)

    assert_frame_equal(first, second)
    assert len(first) == 5
    assert seed_track_id not in first["track_id"].tolist()
    assert first["track_id"].is_unique
    assert first["score"].is_monotonic_decreasing


def test_recommend_from_seed_prefers_same_genre_continuation_for_synthetic_catalog() -> None:
    catalog = load_processed_catalog().set_index("track_id")
    sessions = load_synthetic_sessions().set_index("scenario_type")
    same_genre_session = sessions.loc["same_genre_continuation"]
    seed_track_id = same_genre_session["seed_track_id"]
    seed_genre = catalog.loc[seed_track_id, "genre"]

    recommendations = recommend_from_seed(
        seed_track_id,
        top_k=3,
        catalog=catalog.reset_index(),
    )

    assert len(recommendations) == 3
    assert set(recommendations["genre"]) == {seed_genre}


def test_recommend_from_seed_excludes_tracks_passed_from_session_artifacts() -> None:
    catalog = load_processed_catalog()
    sessions = load_synthetic_sessions().set_index("scenario_type")
    same_genre_session = sessions.loc["same_genre_continuation"]
    excluded_track_ids = (
        same_genre_session["autoplay_candidate_track_ids"]
        + same_genre_session["manual_insertion_track_ids"]
    )

    recommendations = recommend_from_seed(
        same_genre_session["seed_track_id"],
        top_k=5,
        catalog=catalog,
        exclude_track_ids=excluded_track_ids,
    )

    assert len(recommendations) == 5
    assert set(recommendations["track_id"]).isdisjoint(excluded_track_ids)


def test_recommend_from_seed_treats_single_string_exclusion_as_one_track_id() -> None:
    catalog = load_processed_catalog()
    seed_track_id = catalog.iloc[0]["track_id"]
    excluded_track_id = recommend_from_seed(
        seed_track_id,
        top_k=1,
        catalog=catalog,
    ).iloc[0]["track_id"]

    recommendations = recommend_from_seed(
        seed_track_id,
        top_k=5,
        catalog=catalog,
        exclude_track_ids=excluded_track_id,
    )

    assert excluded_track_id not in recommendations["track_id"].tolist()


def test_recommend_from_seed_deduplicates_duplicate_candidate_track_rows() -> None:
    catalog = pd.DataFrame(
        {
            "track_id": [
                "seed_track",
                "duplicate_track",
                "duplicate_track",
                "unique_track",
                "fallback_track",
            ],
            "artist_name": [
                "Artist Seed",
                "Artist Duplicate A",
                "Artist Duplicate B",
                "Artist Unique",
                "Artist Fallback",
            ],
            "genre": ["pop", "pop", "pop", "pop", "rock"],
            "mood": ["calm", "calm", "calm", "driving", "calm"],
            "energy_normalized": [0.50, 0.52, 0.52, 0.56, 0.65],
            "tempo_normalized": [0.50, 0.51, 0.51, 0.58, 0.62],
        }
    )

    recommendations = recommend_from_seed(
        "seed_track",
        top_k=3,
        catalog=catalog,
    )

    assert recommendations["track_id"].tolist() == [
        "duplicate_track",
        "unique_track",
        "fallback_track",
    ]
    assert recommendations["track_id"].is_unique


def test_recommend_from_seed_sizes_output_to_requested_top_k() -> None:
    catalog = load_processed_catalog()
    seed_track_id = catalog.iloc[1]["track_id"]

    recommendations = recommend_from_seed(seed_track_id, top_k=7, catalog=catalog)

    assert len(recommendations) == 7
