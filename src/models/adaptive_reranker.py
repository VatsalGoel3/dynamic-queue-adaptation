from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from src.data.preprocess import DEFAULT_PROCESSED_CATALOG_PATH, load_processed_catalog
from src.models.baseline_seed import recommend_from_seed
from src.models.scoring import (
    GENRE_MATCH_BONUS,
    MOOD_MATCH_BONUS,
    NUMERIC_FEATURE_COLUMNS,
    score_seed_candidates,
)
from src.simulation.intent_update import update_intent_profile
from src.simulation.queue_state import QueueState

DEFAULT_ADAPTIVE_CATALOG_PATH = DEFAULT_PROCESSED_CATALOG_PATH
DEFAULT_CANDIDATE_POOL_SIZE = 20


def _normalize_exclusion_track_ids(
    exclude_track_ids: Iterable[str] | str | None,
) -> tuple[str, ...]:
    if exclude_track_ids is None:
        return ()
    if isinstance(exclude_track_ids, str):
        return (exclude_track_ids,)
    return tuple(exclude_track_ids)


def _resolve_candidate_pool_ids(
    queue_state: QueueState,
    ranking_catalog: pd.DataFrame,
    top_k: int,
    candidate_pool_size: int,
    exclude_track_ids: Iterable[str] | str | None,
) -> tuple[str, ...]:
    explicit_exclusions = _normalize_exclusion_track_ids(exclude_track_ids)
    if queue_state.remaining_candidate_track_ids:
        return tuple(
            track_id
            for track_id in queue_state.remaining_candidate_track_ids
            if track_id not in explicit_exclusions
        )

    baseline_candidates = recommend_from_seed(
        queue_state.seed_track_id,
        top_k=max(top_k, candidate_pool_size),
        catalog=ranking_catalog,
        exclude_track_ids=(
            *queue_state.excluded_track_ids,
            *explicit_exclusions,
        ),
    )
    return tuple(baseline_candidates["track_id"].tolist())


def rerank_remaining_candidates(
    queue_state: QueueState,
    top_k: int = 10,
    catalog: pd.DataFrame | None = None,
    catalog_path: Path = DEFAULT_ADAPTIVE_CATALOG_PATH,
    exclude_track_ids: Iterable[str] | str | None = None,
    candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
) -> pd.DataFrame:
    """Rerank future candidates by blending the baseline with queue-aware intent."""
    ranking_catalog = (
        load_processed_catalog(catalog_path) if catalog is None else catalog.copy()
    )
    if top_k <= 0:
        empty_ranking = score_seed_candidates(
            queue_state.seed_track_id,
            catalog=ranking_catalog,
        ).head(0)
        empty_ranking = empty_ranking.rename(columns={"score": "baseline_score"})
        for column in [
            "intent_numeric_similarity",
            "intent_genre_bonus",
            "intent_mood_bonus",
            "intent_score",
            "pivot_strength",
            "reranked_score",
        ]:
            empty_ranking[column] = pd.Series(dtype=float)
        return empty_ranking.reset_index(drop=True)

    candidate_pool_ids = _resolve_candidate_pool_ids(
        queue_state=queue_state,
        ranking_catalog=ranking_catalog,
        top_k=top_k,
        candidate_pool_size=candidate_pool_size,
        exclude_track_ids=exclude_track_ids,
    )
    effective_queue_state = QueueState(
        seed_track_id=queue_state.seed_track_id,
        candidate_track_ids=candidate_pool_ids,
        manual_insertion_track_ids=queue_state.manual_insertion_track_ids,
        played_track_ids=queue_state.played_track_ids,
    )
    intent_profile = update_intent_profile(
        effective_queue_state,
        catalog=ranking_catalog,
    )

    ranked_candidates = score_seed_candidates(
        queue_state.seed_track_id,
        catalog=ranking_catalog,
    )
    filtered_candidates = ranked_candidates.loc[
        ranked_candidates["track_id"].isin(intent_profile.remaining_candidate_track_ids)
    ].copy()
    ranked_candidates = filtered_candidates.drop_duplicates(
        subset="track_id",
        keep="first",
    ).rename(columns={"score": "baseline_score"})

    if ranked_candidates.empty:
        for column in [
            "intent_numeric_similarity",
            "intent_genre_bonus",
            "intent_mood_bonus",
            "intent_score",
            "pivot_strength",
            "reranked_score",
        ]:
            ranked_candidates[column] = pd.Series(dtype=float)
        return ranked_candidates.reset_index(drop=True)

    numeric_distances = ranked_candidates[NUMERIC_FEATURE_COLUMNS].sub(
        {
            "energy_normalized": intent_profile.energy_normalized,
            "tempo_normalized": intent_profile.tempo_normalized,
        },
        axis="columns",
    ).abs()
    ranked_candidates["intent_numeric_similarity"] = (
        1.0 - numeric_distances.mean(axis=1)
    ).round(6)
    ranked_candidates["intent_genre_bonus"] = (
        (
            ranked_candidates["genre"] == intent_profile.insertion_preferred_genre
        ).astype(float)
        * GENRE_MATCH_BONUS
    ).round(6)
    ranked_candidates["intent_mood_bonus"] = (
        (
            ranked_candidates["mood"] == intent_profile.insertion_preferred_mood
        ).astype(float)
        * MOOD_MATCH_BONUS
    ).round(6)
    ranked_candidates["intent_score"] = (
        ranked_candidates["intent_numeric_similarity"]
        + ranked_candidates["intent_genre_bonus"]
        + ranked_candidates["intent_mood_bonus"]
    ).round(6)
    ranked_candidates["pivot_strength"] = intent_profile.pivot_strength
    ranked_candidates["reranked_score"] = (
        ranked_candidates["baseline_score"]
        + (intent_profile.pivot_strength * ranked_candidates["intent_score"])
    ).round(6)

    reranked = ranked_candidates.sort_values(
        ["reranked_score", "intent_score", "baseline_score", "track_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return reranked.head(top_k).reset_index(drop=True)
