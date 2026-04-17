import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import (
    CANONICAL_TRACK_COLUMNS,
    DEFAULT_RAW_CATALOG_PATH,
    generate_synthetic_catalog,
    load_raw_catalog,
    save_raw_catalog,
)
from src.data.preprocess import (
    DEFAULT_PROCESSED_CATALOG_PATH,
    NUMERIC_FEATURE_COLUMNS,
    load_processed_catalog,
    preprocess_catalog,
    save_processed_catalog,
)


def test_generate_synthetic_catalog_uses_canonical_track_schema() -> None:
    catalog = generate_synthetic_catalog(num_tracks=5, seed=11)

    assert list(catalog.columns) == CANONICAL_TRACK_COLUMNS
    assert catalog.shape == (5, len(CANONICAL_TRACK_COLUMNS))
    assert catalog["track_id"].is_unique


def test_generate_synthetic_catalog_is_deterministic() -> None:
    first = generate_synthetic_catalog(num_tracks=8, seed=7)
    second = generate_synthetic_catalog(num_tracks=8, seed=7)

    assert_frame_equal(first, second)


def test_raw_catalog_round_trip_persists_csv(tmp_path: Path) -> None:
    catalog = generate_synthetic_catalog(num_tracks=6, seed=3)
    output_path = tmp_path / "data" / "raw" / "catalog.csv"

    saved_path = save_raw_catalog(catalog, output_path=output_path)
    loaded = load_raw_catalog(saved_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert_frame_equal(loaded, catalog, check_exact=False, rtol=1e-9, atol=1e-9)


def test_preprocess_catalog_applies_exact_min_max_normalization(
    tmp_path: Path,
) -> None:
    raw_catalog = pd.DataFrame(
        {
            "track_id": ["track_0001", "track_0002", "track_0003"],
            "artist_name": ["Artist 001", "Artist 002", "Artist 003"],
            "genre": ["pop", "rock", "jazz"],
            "energy": [0.2, 0.5, 0.8],
            "mood": ["calm", "driving", "dreamy"],
            "tempo": [90, 120, 150],
        },
        columns=CANONICAL_TRACK_COLUMNS,
    )
    processed = preprocess_catalog(raw_catalog)
    normalized_columns = [f"{column}_normalized" for column in NUMERIC_FEATURE_COLUMNS]
    output_path = tmp_path / "data" / "processed" / "catalog_processed.csv"

    assert list(raw_catalog.columns) == CANONICAL_TRACK_COLUMNS
    assert set(normalized_columns).issubset(processed.columns)
    assert processed["energy_normalized"].tolist() == [0.0, 0.5, 1.0]
    assert processed["tempo_normalized"].tolist() == [0.0, 0.5, 1.0]
    assert pd.api.types.is_float_dtype(processed["energy_normalized"])
    assert pd.api.types.is_float_dtype(processed["tempo_normalized"])

    saved_path = save_processed_catalog(processed, output_path=output_path)
    loaded = load_processed_catalog(saved_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert_frame_equal(loaded, processed, check_exact=False, rtol=1e-9, atol=1e-9)


def test_default_artifact_paths_are_anchored_to_repo_root() -> None:
    assert DEFAULT_RAW_CATALOG_PATH.is_absolute()
    assert DEFAULT_PROCESSED_CATALOG_PATH.is_absolute()
    assert DEFAULT_RAW_CATALOG_PATH.parent.name == "raw"
    assert DEFAULT_PROCESSED_CATALOG_PATH.parent.name == "processed"
