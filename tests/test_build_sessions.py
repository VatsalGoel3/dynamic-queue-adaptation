import json
import sys
from pathlib import Path

from pandas.testing import assert_frame_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_sessions import (
    DEFAULT_SYNTHETIC_SESSIONS_PATH,
    SCENARIO_TYPES,
    build_synthetic_sessions,
    load_synthetic_sessions,
    save_synthetic_sessions,
)
from src.data.load_data import generate_synthetic_catalog
from src.data.preprocess import preprocess_catalog


def _build_processed_catalog():
    raw_catalog = generate_synthetic_catalog(num_tracks=120, seed=9)
    return preprocess_catalog(raw_catalog)


def _decode_track_ids(value: str) -> list[str]:
    track_ids = json.loads(value)
    assert isinstance(track_ids, list)
    assert all(isinstance(track_id, str) for track_id in track_ids)
    return track_ids


def test_build_synthetic_sessions_emits_required_scenarios_and_queue_events() -> None:
    catalog = _build_processed_catalog()

    sessions = build_synthetic_sessions(catalog)

    assert list(sessions["scenario_type"]) == SCENARIO_TYPES
    assert sessions["session_id"].is_unique
    assert sessions["seed_track_id"].notna().all()

    for session in sessions.to_dict(orient="records"):
        autoplay_candidates = _decode_track_ids(session["autoplay_candidate_track_ids"])
        manual_insertions = _decode_track_ids(session["manual_insertion_track_ids"])

        assert len(autoplay_candidates) >= 2
        assert len(manual_insertions) >= 1
        assert session["seed_track_id"] not in autoplay_candidates
        assert session["seed_track_id"] not in manual_insertions


def test_build_synthetic_sessions_is_deterministic() -> None:
    catalog = _build_processed_catalog()

    first = build_synthetic_sessions(catalog)
    second = build_synthetic_sessions(catalog)

    assert_frame_equal(first, second)


def test_synthetic_session_scenarios_encode_expected_genre_patterns() -> None:
    catalog = _build_processed_catalog().set_index("track_id")
    sessions = build_synthetic_sessions(catalog.reset_index()).set_index("scenario_type")

    same_genre_row = sessions.loc["same_genre_continuation"]
    same_seed_genre = catalog.loc[same_genre_row["seed_track_id"], "genre"]
    same_candidate_genres = {
        catalog.loc[track_id, "genre"]
        for track_id in _decode_track_ids(same_genre_row["autoplay_candidate_track_ids"])
    }
    same_manual_genres = {
        catalog.loc[track_id, "genre"]
        for track_id in _decode_track_ids(same_genre_row["manual_insertion_track_ids"])
    }
    assert same_candidate_genres == {same_seed_genre}
    assert same_manual_genres == {same_seed_genre}

    cross_genre_row = sessions.loc["cross_genre_shift"]
    cross_seed_genre = catalog.loc[cross_genre_row["seed_track_id"], "genre"]
    cross_manual_genres = {
        catalog.loc[track_id, "genre"]
        for track_id in _decode_track_ids(cross_genre_row["manual_insertion_track_ids"])
    }
    assert len(cross_manual_genres) == 1
    assert cross_manual_genres != {cross_seed_genre}

    outlier_row = sessions.loc["one_outlier_insertion"]
    outlier_seed_genre = catalog.loc[outlier_row["seed_track_id"], "genre"]
    outlier_candidate_genres = {
        catalog.loc[track_id, "genre"]
        for track_id in _decode_track_ids(outlier_row["autoplay_candidate_track_ids"])
    }
    outlier_manual_genres = [
        catalog.loc[track_id, "genre"]
        for track_id in _decode_track_ids(outlier_row["manual_insertion_track_ids"])
    ]
    assert outlier_candidate_genres == {outlier_seed_genre}
    assert len(outlier_manual_genres) == 1
    assert outlier_manual_genres[0] != outlier_seed_genre

    repeated_row = sessions.loc["repeated_consistent_insertions"]
    repeated_seed_genre = catalog.loc[repeated_row["seed_track_id"], "genre"]
    repeated_manual_genres = [
        catalog.loc[track_id, "genre"]
        for track_id in _decode_track_ids(repeated_row["manual_insertion_track_ids"])
    ]
    assert len(repeated_manual_genres) >= 2
    assert len(set(repeated_manual_genres)) == 1
    assert repeated_manual_genres[0] != repeated_seed_genre


def test_synthetic_sessions_round_trip_persists_csv(tmp_path: Path) -> None:
    catalog = _build_processed_catalog()
    sessions = build_synthetic_sessions(catalog)
    output_path = tmp_path / "data" / "synthetic" / "synthetic_sessions.csv"

    saved_path = save_synthetic_sessions(sessions, output_path=output_path)
    loaded = load_synthetic_sessions(saved_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert_frame_equal(loaded, sessions)


def test_default_synthetic_artifact_path_is_anchored_to_repo_root() -> None:
    assert DEFAULT_SYNTHETIC_SESSIONS_PATH.is_absolute()
    assert DEFAULT_SYNTHETIC_SESSIONS_PATH.parent.name == "synthetic"
