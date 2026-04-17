from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_TRACK_COLUMNS = [
    "track_id",
    "artist_name",
    "genre",
    "energy",
    "mood",
    "tempo",
]

DEFAULT_RAW_CATALOG_PATH = Path("data/raw/synthetic_track_catalog.csv")

_GENRES = ["pop", "rock", "electronic", "hip_hop", "jazz"]
_MOODS = ["calm", "uplifting", "brooding", "driving", "dreamy"]


def generate_synthetic_catalog(num_tracks: int = 100, seed: int = 0) -> pd.DataFrame:
    """Create a deterministic synthetic track catalog for offline experiments."""
    rng = np.random.default_rng(seed)

    catalog = pd.DataFrame(
        {
            "track_id": [f"track_{index:04d}" for index in range(num_tracks)],
            "artist_name": [f"Artist {index:03d}" for index in range(num_tracks)],
            "genre": rng.choice(_GENRES, size=num_tracks),
            "energy": np.round(rng.uniform(0.1, 1.0, size=num_tracks), 6),
            "mood": rng.choice(_MOODS, size=num_tracks),
            "tempo": rng.integers(70, 181, size=num_tracks),
        },
        columns=CANONICAL_TRACK_COLUMNS,
    )

    return catalog


def save_raw_catalog(
    catalog: pd.DataFrame, output_path: Path = DEFAULT_RAW_CATALOG_PATH
) -> Path:
    """Persist the raw synthetic catalog under the raw-data directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False)
    return output_path


def load_raw_catalog(input_path: Path = DEFAULT_RAW_CATALOG_PATH) -> pd.DataFrame:
    """Load a raw synthetic catalog artifact from disk."""
    catalog = pd.read_csv(input_path)
    return catalog.loc[:, CANONICAL_TRACK_COLUMNS]
