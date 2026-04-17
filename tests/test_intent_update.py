from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_sessions import load_synthetic_sessions
from src.data.preprocess import load_processed_catalog
from src.simulation.intent_update import update_intent_profile
from src.simulation.queue_state import QueueState


def _build_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "track_id": [
                "seed_pop",
                "candidate_pop_1",
                "candidate_pop_2",
                "rock_shift_1",
                "rock_shift_2",
                "rock_shift_3",
                "ambient_outlier",
            ],
            "genre": [
                "pop",
                "pop",
                "pop",
                "rock",
                "rock",
                "rock",
                "ambient",
            ],
            "mood": [
                "calm",
                "calm",
                "uplifting",
                "calm",
                "calm",
                "calm",
                "dreamy",
            ],
            "energy_normalized": [0.20, 0.24, 0.28, 0.60, 0.62, 0.64, 0.98],
            "tempo_normalized": [0.20, 0.24, 0.30, 0.58, 0.56, 0.60, 0.95],
        }
    )


def _load_repo_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        load_processed_catalog().set_index("track_id"),
        load_synthetic_sessions().set_index("scenario_type"),
    )


def _queue_state_for_scenario(scenarios: pd.DataFrame, scenario_type: str) -> QueueState:
    scenario = scenarios.loc[scenario_type]
    return QueueState(
        seed_track_id=scenario["seed_track_id"],
        candidate_track_ids=scenario["autoplay_candidate_track_ids"],
        manual_insertion_track_ids=scenario["manual_insertion_track_ids"],
        played_track_ids=[scenario["seed_track_id"]],
    )


def test_queue_state_tracks_seed_manual_played_and_candidate_ids() -> None:
    state = QueueState(
        seed_track_id="seed_pop",
        candidate_track_ids=["candidate_pop_1", "candidate_pop_2"],
        manual_insertion_track_ids=["rock_shift_1", "rock_shift_2"],
        played_track_ids=["seed_pop", "candidate_pop_1"],
    )

    assert state.seed_track_id == "seed_pop"
    assert state.candidate_track_ids == ("candidate_pop_1", "candidate_pop_2")
    assert state.manual_insertion_track_ids == ("rock_shift_1", "rock_shift_2")
    assert state.played_track_ids == ("seed_pop", "candidate_pop_1")
    assert state.excluded_track_ids == (
        "seed_pop",
        "candidate_pop_1",
        "rock_shift_1",
        "rock_shift_2",
        "candidate_pop_2",
    )


def test_update_intent_profile_without_insertions_keeps_seed_profile() -> None:
    catalog, _ = _load_repo_artifacts()
    seed_track_id = catalog.index[0]
    seed_row = catalog.loc[seed_track_id]
    state = QueueState(
        seed_track_id=seed_track_id,
        candidate_track_ids=[],
        manual_insertion_track_ids=[],
        played_track_ids=[seed_track_id],
    )

    profile = update_intent_profile(state, catalog=catalog.reset_index())

    assert profile.anchor_track_id == seed_track_id
    assert profile.source_track_ids == (seed_track_id,)
    assert profile.remaining_candidate_track_ids == ()
    assert profile.insertion_preferred_genre is None
    assert profile.insertion_preferred_mood is None
    assert profile.dominant_genre == seed_row["genre"]
    assert profile.dominant_mood == seed_row["mood"]
    assert profile.energy_normalized == seed_row["energy_normalized"]
    assert profile.tempo_normalized == seed_row["tempo_normalized"]
    assert profile.pivot_strength == 0.0


def test_real_sessions_anchor_core_intent_behavior_to_repo_artifacts() -> None:
    catalog, scenarios = _load_repo_artifacts()
    same_genre_state = _queue_state_for_scenario(scenarios, "same_genre_continuation")
    cross_genre_state = _queue_state_for_scenario(scenarios, "cross_genre_shift")
    outlier_state = _queue_state_for_scenario(scenarios, "one_outlier_insertion")
    repeated_state = _queue_state_for_scenario(scenarios, "repeated_consistent_insertions")

    same_genre_profile = update_intent_profile(
        same_genre_state, catalog=catalog.reset_index()
    )
    cross_genre_profile = update_intent_profile(
        cross_genre_state, catalog=catalog.reset_index()
    )
    outlier_profile = update_intent_profile(
        outlier_state, catalog=catalog.reset_index()
    )
    repeated_profile = update_intent_profile(
        repeated_state, catalog=catalog.reset_index()
    )

    same_seed = catalog.loc[same_genre_state.seed_track_id]
    cross_seed = catalog.loc[cross_genre_state.seed_track_id]
    outlier_seed = catalog.loc[outlier_state.seed_track_id]
    cross_insertions = catalog.loc[list(cross_genre_state.manual_insertion_track_ids)]
    repeated_insertions = catalog.loc[list(repeated_state.manual_insertion_track_ids)]

    assert same_genre_profile.dominant_genre == same_seed["genre"]
    assert same_genre_profile.dominant_mood == same_seed["mood"]
    assert 0.0 < same_genre_profile.pivot_strength < 0.35

    assert cross_genre_profile.dominant_genre == cross_seed["genre"]
    assert cross_genre_profile.dominant_mood == cross_seed["mood"]
    assert cross_genre_profile.insertion_preferred_genre == cross_insertions.iloc[0]["genre"]
    assert cross_genre_profile.insertion_preferred_mood == cross_insertions.iloc[0]["mood"]

    assert outlier_profile.dominant_genre == outlier_seed["genre"]
    assert outlier_profile.dominant_mood == outlier_seed["mood"]
    assert 0.0 < outlier_profile.pivot_strength < 0.25

    assert repeated_profile.dominant_genre == repeated_insertions.iloc[0]["genre"]
    assert repeated_profile.dominant_mood == repeated_insertions.iloc[0]["mood"]
    assert same_genre_profile.pivot_strength < cross_genre_profile.pivot_strength
    assert 0.15 <= cross_genre_profile.pivot_strength < 0.45
    assert cross_genre_profile.pivot_strength < repeated_profile.pivot_strength
    assert outlier_profile.pivot_strength < cross_genre_profile.pivot_strength
    assert repeated_profile.pivot_strength > 0.55


def test_intent_profile_exposes_remaining_candidates_after_seed_played_and_inserted_tracks() -> None:
    catalog = _build_catalog()
    state = QueueState(
        seed_track_id="seed_pop",
        candidate_track_ids=["candidate_pop_1", "candidate_pop_2", "rock_shift_1"],
        manual_insertion_track_ids=["rock_shift_1"],
        played_track_ids=["seed_pop", "candidate_pop_1"],
    )

    profile = update_intent_profile(state, catalog=catalog)

    assert profile.remaining_candidate_track_ids == ("candidate_pop_2",)


def test_single_outlier_keeps_seed_categorical_intent_in_targeted_fixture() -> None:
    catalog = _build_catalog()
    state = QueueState(
        seed_track_id="seed_pop",
        candidate_track_ids=["candidate_pop_1", "candidate_pop_2"],
        manual_insertion_track_ids=["ambient_outlier"],
        played_track_ids=["seed_pop"],
    )

    profile = update_intent_profile(state, catalog=catalog)

    assert profile.dominant_genre == "pop"
    assert profile.dominant_mood == "calm"
    assert 0.20 < profile.energy_normalized < 0.98
    assert 0.20 < profile.tempo_normalized < 0.95
    assert 0.0 < profile.pivot_strength < 0.20


def test_single_non_outlier_insertion_keeps_seed_categorical_anchor_in_targeted_fixture() -> None:
    catalog = _build_catalog()
    state = QueueState(
        seed_track_id="seed_pop",
        candidate_track_ids=["candidate_pop_1", "candidate_pop_2"],
        manual_insertion_track_ids=["rock_shift_1"],
        played_track_ids=["seed_pop"],
    )

    profile = update_intent_profile(state, catalog=catalog)

    assert profile.dominant_genre == "pop"
    assert profile.dominant_mood == "calm"
    assert profile.insertion_preferred_genre == "rock"
    assert profile.insertion_preferred_mood == "calm"
    assert 0.15 <= profile.pivot_strength < 0.35


def test_repeated_consistent_insertions_increase_pivot_strength_in_targeted_fixture() -> None:
    catalog = _build_catalog()
    one = update_intent_profile(
        QueueState(
            seed_track_id="seed_pop",
            candidate_track_ids=["candidate_pop_1", "candidate_pop_2"],
            manual_insertion_track_ids=["rock_shift_1"],
            played_track_ids=["seed_pop"],
        ),
        catalog=catalog,
    )
    two = update_intent_profile(
        QueueState(
            seed_track_id="seed_pop",
            candidate_track_ids=["candidate_pop_1", "candidate_pop_2"],
            manual_insertion_track_ids=["rock_shift_1", "rock_shift_2"],
            played_track_ids=["seed_pop"],
        ),
        catalog=catalog,
    )
    three = update_intent_profile(
        QueueState(
            seed_track_id="seed_pop",
            candidate_track_ids=["candidate_pop_1", "candidate_pop_2"],
            manual_insertion_track_ids=["rock_shift_1", "rock_shift_2", "rock_shift_3"],
            played_track_ids=["seed_pop"],
        ),
        catalog=catalog,
    )

    assert one.dominant_genre == "pop"
    assert one.dominant_mood == "calm"
    assert one.insertion_preferred_genre == "rock"
    assert two.dominant_genre == "rock"
    assert three.dominant_genre == "rock"
    assert 0.35 <= two.pivot_strength < 0.65
    assert 0.55 <= three.pivot_strength <= 1.0
    assert one.pivot_strength < two.pivot_strength < three.pivot_strength
    assert one.energy_normalized < two.energy_normalized < three.energy_normalized
