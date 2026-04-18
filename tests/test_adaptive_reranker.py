from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_sessions import load_synthetic_sessions
from src.data.preprocess import load_processed_catalog
from src.models.adaptive_reranker import rerank_remaining_candidates
from src.models.baseline_seed import recommend_from_seed
from src.simulation.intent_update import update_intent_profile
from src.simulation.queue_state import QueueState


def _queue_state_for_session(
    scenario_row: pd.Series,
    candidate_track_ids: list[str] | None = None,
) -> QueueState:
    return QueueState(
        seed_track_id=scenario_row["seed_track_id"],
        candidate_track_ids=(
            candidate_track_ids
            if candidate_track_ids is not None
            else scenario_row["autoplay_candidate_track_ids"]
        ),
        manual_insertion_track_ids=scenario_row["manual_insertion_track_ids"],
        played_track_ids=[scenario_row["seed_track_id"]],
    )


def _genre_count(frame: pd.DataFrame, genre: str) -> int:
    return int((frame["genre"] == genre).sum())


def _build_adaptive_catalog_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": [
                "seed_pop",
                "pop_anchor_1",
                "pop_anchor_2",
                "rock_shift_1",
                "rock_shift_2",
                "rock_shift_3",
                "rock_future_1",
                "rock_future_2",
            ],
            "artist_name": [
                "Artist Seed",
                "Artist Pop 1",
                "Artist Pop 2",
                "Artist Rock 1",
                "Artist Rock 2",
                "Artist Rock 3",
                "Artist Rock Future 1",
                "Artist Rock Future 2",
            ],
            "genre": ["pop", "pop", "pop", "rock", "rock", "rock", "rock", "rock"],
            "mood": ["calm", "calm", "uplifting", "calm", "calm", "calm", "calm", "calm"],
            "energy": [0.20, 0.24, 0.28, 0.58, 0.60, 0.62, 0.57, 0.59],
            "tempo": [100, 104, 110, 150, 151, 152, 149, 153],
            "energy_normalized": [0.20, 0.24, 0.28, 0.58, 0.60, 0.62, 0.57, 0.59],
            "tempo_normalized": [0.20, 0.24, 0.30, 0.58, 0.59, 0.60, 0.57, 0.61],
        }
    )


def test_adaptive_reranker_is_deterministic() -> None:
    catalog = load_processed_catalog()
    sessions = load_synthetic_sessions()
    scenario_row = sessions.loc[sessions["scenario_type"] == "cross_genre_shift"].iloc[0]
    queue_state = _queue_state_for_session(scenario_row, candidate_track_ids=[])

    first = rerank_remaining_candidates(queue_state, top_k=5, catalog=catalog)
    second = rerank_remaining_candidates(queue_state, top_k=5, catalog=catalog)

    assert_frame_equal(first, second)
    assert first["track_id"].is_unique
    assert first["reranked_score"].is_monotonic_decreasing


def test_same_genre_continuation_stays_close_to_baseline() -> None:
    catalog = load_processed_catalog()
    sessions = load_synthetic_sessions()
    scenario_row = sessions.loc[
        sessions["scenario_type"] == "same_genre_continuation"
    ].iloc[0]
    queue_state = _queue_state_for_session(scenario_row, candidate_track_ids=[])

    baseline = recommend_from_seed(
        queue_state.seed_track_id,
        top_k=5,
        catalog=catalog,
        exclude_track_ids=(
            *queue_state.played_track_ids,
            *queue_state.manual_insertion_track_ids,
        ),
    )
    adaptive = rerank_remaining_candidates(queue_state, top_k=5, catalog=catalog)

    assert adaptive["track_id"].tolist()[:3] == baseline["track_id"].tolist()[:3]


def test_cross_genre_shift_changes_future_ranking_meaningfully() -> None:
    catalog = load_processed_catalog()
    sessions = load_synthetic_sessions()
    scenario_row = sessions.loc[sessions["scenario_type"] == "cross_genre_shift"].iloc[0]
    queue_state = _queue_state_for_session(scenario_row, candidate_track_ids=[])
    insertion_track_id = queue_state.manual_insertion_track_ids[0]
    insertion_genre = (
        catalog.loc[catalog["track_id"] == insertion_track_id, "genre"].iloc[0]
    )

    baseline = recommend_from_seed(
        queue_state.seed_track_id,
        top_k=5,
        catalog=catalog,
        exclude_track_ids=(
            *queue_state.played_track_ids,
            *queue_state.manual_insertion_track_ids,
        ),
    )
    adaptive = rerank_remaining_candidates(queue_state, top_k=5, catalog=catalog)

    assert _genre_count(adaptive, insertion_genre) > _genre_count(baseline, insertion_genre)


def test_repeated_consistent_insertions_shift_more_than_single_insertion() -> None:
    catalog = _build_adaptive_catalog_fixture()
    candidate_pool = [
        "pop_anchor_1",
        "pop_anchor_2",
        "rock_shift_1",
        "rock_shift_2",
        "rock_shift_3",
        "rock_future_1",
        "rock_future_2",
    ]
    single_state = QueueState(
        seed_track_id="seed_pop",
        candidate_track_ids=candidate_pool,
        manual_insertion_track_ids=["rock_shift_1"],
        played_track_ids=["seed_pop"],
    )
    repeated_state = QueueState(
        seed_track_id="seed_pop",
        candidate_track_ids=candidate_pool,
        manual_insertion_track_ids=["rock_shift_1", "rock_shift_2", "rock_shift_3"],
        played_track_ids=["seed_pop"],
    )

    single_profile = update_intent_profile(single_state, catalog)
    repeated_profile = update_intent_profile(repeated_state, catalog)
    single_adaptive = rerank_remaining_candidates(single_state, top_k=5, catalog=catalog)
    repeated_adaptive = rerank_remaining_candidates(repeated_state, top_k=5, catalog=catalog)
    single_insertion_genre_scores = single_adaptive.loc[
        single_adaptive["genre"] == single_profile.insertion_preferred_genre,
        "reranked_score",
    ]
    repeated_insertion_genre_scores = repeated_adaptive.loc[
        repeated_adaptive["genre"] == repeated_profile.insertion_preferred_genre,
        "reranked_score",
    ]

    assert single_profile.pivot_strength < repeated_profile.pivot_strength
    assert not single_insertion_genre_scores.empty
    assert not repeated_insertion_genre_scores.empty
    assert repeated_insertion_genre_scores.mean() > single_insertion_genre_scores.mean()


def test_single_outlier_insertion_does_not_overreact() -> None:
    catalog = load_processed_catalog()
    sessions = load_synthetic_sessions()
    scenario_row = sessions.loc[
        sessions["scenario_type"] == "one_outlier_insertion"
    ].iloc[0]
    queue_state = _queue_state_for_session(scenario_row, candidate_track_ids=[])
    seed_track_id = queue_state.seed_track_id
    seed_genre = catalog.loc[catalog["track_id"] == seed_track_id, "genre"].iloc[0]
    baseline = recommend_from_seed(
        seed_track_id,
        top_k=5,
        catalog=catalog,
        exclude_track_ids=(
            *queue_state.played_track_ids,
            *queue_state.manual_insertion_track_ids,
        ),
    )
    adaptive = rerank_remaining_candidates(queue_state, top_k=5, catalog=catalog)

    assert _genre_count(adaptive, seed_genre) >= 3
    assert len(set(adaptive["track_id"]).intersection(baseline["track_id"])) >= 3


def test_exclusions_are_preserved() -> None:
    catalog = load_processed_catalog()
    sessions = load_synthetic_sessions()
    scenario_row = sessions.loc[sessions["scenario_type"] == "cross_genre_shift"].iloc[0]
    queue_state = _queue_state_for_session(scenario_row, candidate_track_ids=[])
    excluded_track_id = recommend_from_seed(
        queue_state.seed_track_id,
        top_k=1,
        catalog=catalog,
        exclude_track_ids=(
            *queue_state.played_track_ids,
            *queue_state.manual_insertion_track_ids,
        ),
    ).iloc[0]["track_id"]

    adaptive = rerank_remaining_candidates(
        queue_state,
        top_k=5,
        catalog=catalog,
        exclude_track_ids=excluded_track_id,
    )

    assert excluded_track_id not in adaptive["track_id"].tolist()
