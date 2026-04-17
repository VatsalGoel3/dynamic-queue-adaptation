from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.models.scoring import DEFAULT_SCORING_CATALOG_PATH, score_seed_candidates

DEFAULT_BASELINE_CATALOG_PATH = DEFAULT_SCORING_CATALOG_PATH


def _normalize_exclusion_track_ids(
    seed_track_id: str, exclude_track_ids: Iterable[str] | None
) -> set[str]:
    excluded_track_ids = {seed_track_id}
    if exclude_track_ids is not None:
        excluded_track_ids.update(exclude_track_ids)
    return excluded_track_ids


def recommend_from_seed(
    seed_track_id: str,
    top_k: int = 10,
    catalog: pd.DataFrame | None = None,
    catalog_path: Path = DEFAULT_BASELINE_CATALOG_PATH,
    exclude_track_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a deterministic top-k baseline recommendation list for a seed track."""
    ranked_candidates = score_seed_candidates(
        seed_track_id,
        catalog=catalog,
        catalog_path=catalog_path,
    )
    if top_k <= 0:
        return ranked_candidates.head(0).reset_index(drop=True)

    excluded_track_ids = _normalize_exclusion_track_ids(
        seed_track_id, exclude_track_ids
    )
    filtered_candidates = ranked_candidates.loc[
        ~ranked_candidates["track_id"].isin(excluded_track_ids)
    ]
    deduplicated_candidates = filtered_candidates.drop_duplicates(
        subset="track_id", keep="first"
    )
    return deduplicated_candidates.head(top_k).reset_index(drop=True)
