from __future__ import annotations

from pathlib import Path

import pandas as pd

NUMERIC_FEATURE_COLUMNS = ["energy", "tempo"]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_CATALOG_PATH = REPO_ROOT / "data/processed/processed_track_catalog.csv"


def preprocess_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Add normalized numeric features for later baseline and reranking models."""
    processed = catalog.copy()

    for column in NUMERIC_FEATURE_COLUMNS:
        numeric_values = processed[column].astype(float)
        minimum = numeric_values.min()
        maximum = numeric_values.max()

        if maximum == minimum:
            normalized_values = pd.Series(0.0, index=processed.index)
        else:
            normalized_values = (numeric_values - minimum) / (maximum - minimum)

        processed[f"{column}_normalized"] = normalized_values.round(6).astype(float)

    return processed


def save_processed_catalog(
    catalog: pd.DataFrame, output_path: Path = DEFAULT_PROCESSED_CATALOG_PATH
) -> Path:
    """Persist the processed catalog under the processed-data directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False)
    return output_path


def load_processed_catalog(
    input_path: Path = DEFAULT_PROCESSED_CATALOG_PATH,
) -> pd.DataFrame:
    """Load a processed catalog artifact from disk."""
    return pd.read_csv(input_path)
