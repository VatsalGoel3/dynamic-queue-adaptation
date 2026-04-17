from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.load_data import generate_synthetic_catalog
from src.data.preprocess import (
    DEFAULT_PROCESSED_CATALOG_PATH,
    load_processed_catalog,
    preprocess_catalog,
    save_processed_catalog,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYNTHETIC_SESSIONS_PATH = REPO_ROOT / "data/synthetic/synthetic_sessions.csv"

SESSION_COLUMNS = [
    "session_id",
    "scenario_type",
    "seed_track_id",
    "autoplay_candidate_track_ids",
    "manual_insertion_track_ids",
]

SCENARIO_TYPES = [
    "same_genre_continuation",
    "cross_genre_shift",
    "one_outlier_insertion",
    "repeated_consistent_insertions",
]


def _sorted_tracks_by_genre(catalog: pd.DataFrame) -> dict[str, list[str]]:
    grouped = (
        catalog.loc[:, ["track_id", "genre"]]
        .sort_values(["genre", "track_id"])
        .groupby("genre")["track_id"]
        .apply(list)
    )
    return grouped.to_dict()


def _genre_priority(tracks_by_genre: dict[str, list[str]]) -> list[str]:
    return [
        genre
        for genre, _ in sorted(
            tracks_by_genre.items(), key=lambda item: (-len(item[1]), item[0])
        )
    ]


def _encode_track_ids(track_ids: list[str]) -> str:
    return json.dumps(track_ids)


def build_synthetic_sessions(catalog: pd.DataFrame) -> pd.DataFrame:
    """Build a deterministic set of synthetic queue-adaptation sessions."""
    tracks_by_genre = _sorted_tracks_by_genre(catalog)
    ordered_genres = _genre_priority(tracks_by_genre)

    if len(ordered_genres) < 2:
        raise ValueError("session generation requires at least two genres")

    primary_genre = ordered_genres[0]
    secondary_genre = ordered_genres[1]
    tertiary_genre = ordered_genres[2] if len(ordered_genres) > 2 else secondary_genre

    primary_tracks = tracks_by_genre[primary_genre]
    secondary_tracks = tracks_by_genre[secondary_genre]
    tertiary_tracks = tracks_by_genre[tertiary_genre]

    if len(primary_tracks) < 4:
        raise ValueError("session generation requires four tracks in one genre")
    if len(secondary_tracks) < 2:
        raise ValueError("session generation requires two tracks in another genre")

    sessions = [
        {
            "session_id": "session_001",
            "scenario_type": "same_genre_continuation",
            "seed_track_id": primary_tracks[0],
            "autoplay_candidate_track_ids": _encode_track_ids(primary_tracks[1:3]),
            "manual_insertion_track_ids": _encode_track_ids([primary_tracks[3]]),
        },
        {
            "session_id": "session_002",
            "scenario_type": "cross_genre_shift",
            "seed_track_id": primary_tracks[0],
            "autoplay_candidate_track_ids": _encode_track_ids(primary_tracks[1:3]),
            "manual_insertion_track_ids": _encode_track_ids([secondary_tracks[0]]),
        },
        {
            "session_id": "session_003",
            "scenario_type": "one_outlier_insertion",
            "seed_track_id": primary_tracks[1],
            "autoplay_candidate_track_ids": _encode_track_ids(
                [primary_tracks[0], primary_tracks[2]]
            ),
            "manual_insertion_track_ids": _encode_track_ids([tertiary_tracks[0]]),
        },
        {
            "session_id": "session_004",
            "scenario_type": "repeated_consistent_insertions",
            "seed_track_id": primary_tracks[2],
            "autoplay_candidate_track_ids": _encode_track_ids(primary_tracks[0:2]),
            "manual_insertion_track_ids": _encode_track_ids(
                [secondary_tracks[0], secondary_tracks[1]]
            ),
        },
    ]

    return pd.DataFrame(sessions, columns=SESSION_COLUMNS)


def save_synthetic_sessions(
    sessions: pd.DataFrame, output_path: Path = DEFAULT_SYNTHETIC_SESSIONS_PATH
) -> Path:
    """Persist synthetic session definitions under the synthetic-data directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_path, index=False)
    return output_path


def load_synthetic_sessions(
    input_path: Path = DEFAULT_SYNTHETIC_SESSIONS_PATH,
) -> pd.DataFrame:
    """Load the saved synthetic session artifact from disk."""
    return pd.read_csv(input_path)


def build_default_session_artifacts(
    processed_catalog_path: Path = DEFAULT_PROCESSED_CATALOG_PATH,
    sessions_output_path: Path = DEFAULT_SYNTHETIC_SESSIONS_PATH,
) -> tuple[Path, Path]:
    """Ensure the processed catalog exists and write the deterministic sessions."""
    if processed_catalog_path.exists():
        processed_catalog = load_processed_catalog(processed_catalog_path)
    else:
        processed_catalog = preprocess_catalog(generate_synthetic_catalog())
        save_processed_catalog(processed_catalog, output_path=processed_catalog_path)

    sessions = build_synthetic_sessions(processed_catalog)
    saved_sessions_path = save_synthetic_sessions(sessions, output_path=sessions_output_path)
    return processed_catalog_path, saved_sessions_path


if __name__ == "__main__":
    build_default_session_artifacts()
