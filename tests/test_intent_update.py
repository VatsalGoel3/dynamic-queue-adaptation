from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    catalog = _build_catalog()
    state = QueueState(
        seed_track_id="seed_pop",
        candidate_track_ids=["candidate_pop_1", "candidate_pop_2"],
        manual_insertion_track_ids=[],
        played_track_ids=["seed_pop"],
    )

    profile = update_intent_profile(state, catalog=catalog)

    assert profile.anchor_track_id == "seed_pop"
    assert profile.dominant_genre == "pop"
    assert profile.dominant_mood == "calm"
    assert profile.energy_normalized == 0.20
    assert profile.tempo_normalized == 0.20
    assert profile.pivot_strength == 0.0


def test_single_insertion_creates_deterministic_small_pivot() -> None:
    catalog = _build_catalog()
    state = QueueState(
        seed_track_id="seed_pop",
        candidate_track_ids=["candidate_pop_1", "candidate_pop_2"],
        manual_insertion_track_ids=["rock_shift_1"],
        played_track_ids=["seed_pop"],
    )

    first = update_intent_profile(state, catalog=catalog)
    second = update_intent_profile(state, catalog=catalog)

    assert first == second
    assert first.dominant_genre == "rock"
    assert first.dominant_mood == "calm"
    assert 0.20 < first.energy_normalized < 0.60
    assert 0.20 < first.tempo_normalized < 0.58
    assert abs(first.energy_normalized - 0.60) < abs(first.energy_normalized - 0.20)
    assert 0.20 <= first.pivot_strength < 0.45


def test_repeated_consistent_insertions_increase_pivot_strength() -> None:
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

    assert one.dominant_genre == "rock"
    assert two.dominant_genre == "rock"
    assert three.dominant_genre == "rock"
    assert 0.35 <= two.pivot_strength < 0.65
    assert 0.55 <= three.pivot_strength <= 1.0
    assert one.pivot_strength < two.pivot_strength < three.pivot_strength
    assert one.energy_normalized < two.energy_normalized < three.energy_normalized


def test_single_clear_outlier_stays_low_confidence() -> None:
    catalog = _build_catalog()
    consistent_shift = update_intent_profile(
        QueueState(
            seed_track_id="seed_pop",
            candidate_track_ids=["candidate_pop_1", "candidate_pop_2"],
            manual_insertion_track_ids=["rock_shift_1"],
            played_track_ids=["seed_pop"],
        ),
        catalog=catalog,
    )
    outlier_shift = update_intent_profile(
        QueueState(
            seed_track_id="seed_pop",
            candidate_track_ids=["candidate_pop_1", "candidate_pop_2"],
            manual_insertion_track_ids=["ambient_outlier"],
            played_track_ids=["seed_pop"],
        ),
        catalog=catalog,
    )

    assert outlier_shift.dominant_genre == "ambient"
    assert outlier_shift.dominant_mood == "dreamy"
    assert outlier_shift.pivot_strength < consistent_shift.pivot_strength
    assert 0.0 < outlier_shift.pivot_strength < 0.20
