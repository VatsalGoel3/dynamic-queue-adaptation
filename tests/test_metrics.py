from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.build_sessions import load_synthetic_sessions
from src.data.preprocess import load_processed_catalog
from src.evaluation.compare_models import run_comparison_pipeline
from src.evaluation.metrics import (
    adaptation_shift_score,
    calculate_model_metrics,
    diversity_retention,
    intent_alignment_score,
    overreaction_penalty,
)
from src.models.adaptive_reranker import rerank_remaining_candidates
from src.models.baseline_seed import recommend_from_seed
from src.simulation.intent_update import update_intent_profile
from src.simulation.queue_state import QueueState


def _session_context(scenario_type: str) -> tuple[pd.DataFrame, pd.Series, QueueState]:
    catalog = load_processed_catalog()
    session_row = load_synthetic_sessions().loc[
        lambda frame: frame["scenario_type"] == scenario_type
    ].iloc[0]
    queue_state = QueueState(
        seed_track_id=session_row["seed_track_id"],
        candidate_track_ids=(),
        manual_insertion_track_ids=session_row["manual_insertion_track_ids"],
        played_track_ids=[session_row["seed_track_id"]],
    )
    return catalog, session_row, queue_state


def _session_recommendations(
    catalog: pd.DataFrame, queue_state: QueueState, top_k: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    baseline = recommend_from_seed(
        queue_state.seed_track_id,
        top_k=top_k,
        catalog=catalog,
        exclude_track_ids=(
            *queue_state.played_track_ids,
            *queue_state.manual_insertion_track_ids,
        ),
    )
    adaptive = rerank_remaining_candidates(queue_state, top_k=top_k, catalog=catalog)
    intent_profile = update_intent_profile(queue_state, catalog)
    return baseline, adaptive, intent_profile


def test_metric_helpers_are_deterministic() -> None:
    catalog, session_row, queue_state = _session_context("cross_genre_shift")
    seed_track = catalog.loc[catalog["track_id"] == queue_state.seed_track_id].iloc[0]
    baseline, adaptive, intent_profile = _session_recommendations(catalog, queue_state)

    first = calculate_model_metrics(
        recommendations=adaptive,
        baseline_recommendations=baseline,
        seed_track=seed_track,
        intent_profile=intent_profile,
        top_k=5,
    )
    second = calculate_model_metrics(
        recommendations=adaptive,
        baseline_recommendations=baseline,
        seed_track=seed_track,
        intent_profile=intent_profile,
        top_k=5,
    )

    assert first == second
    assert intent_alignment_score(adaptive, intent_profile, top_k=5) == pytest.approx(
        intent_alignment_score(adaptive, intent_profile, top_k=5)
    )
    assert adaptation_shift_score(
        adaptive, baseline, intent_profile, top_k=5
    ) == pytest.approx(adaptation_shift_score(adaptive, baseline, intent_profile, top_k=5))
    assert overreaction_penalty(adaptive, seed_track, intent_profile, top_k=5) == pytest.approx(
        overreaction_penalty(adaptive, seed_track, intent_profile, top_k=5)
    )


def test_metric_helpers_stay_bounded_and_respect_known_invariants() -> None:
    catalog = load_processed_catalog()
    sessions = load_synthetic_sessions()

    for _, session_row in sessions.iterrows():
        queue_state = QueueState(
            seed_track_id=session_row["seed_track_id"],
            candidate_track_ids=(),
            manual_insertion_track_ids=session_row["manual_insertion_track_ids"],
            played_track_ids=[session_row["seed_track_id"]],
        )
        seed_track = catalog.loc[catalog["track_id"] == queue_state.seed_track_id].iloc[0]
        baseline, adaptive, intent_profile = _session_recommendations(catalog, queue_state)

        for recommendations in (baseline, adaptive):
            intent_score = intent_alignment_score(recommendations, intent_profile, top_k=5)
            assert 0.0 <= intent_score <= 1.0
            assert 0.0 <= overreaction_penalty(recommendations, seed_track, intent_profile, top_k=5) <= 1.0

        baseline_shift = adaptation_shift_score(baseline, baseline, intent_profile, top_k=5)
        adaptive_shift = adaptation_shift_score(adaptive, baseline, intent_profile, top_k=5)
        retention = diversity_retention(adaptive, baseline, top_k=5)

        assert baseline_shift == pytest.approx(0.5)
        assert 0.0 <= adaptive_shift <= 1.0
        assert adaptive_shift >= baseline_shift
        assert 0.0 <= retention <= 1.0


def test_compare_models_pipeline_writes_compact_results(tmp_path: Path) -> None:
    output_path = tmp_path / "results_summary.csv"

    summary = run_comparison_pipeline(output_path=output_path, top_k=5)
    persisted = pd.read_csv(output_path)

    assert output_path.exists()
    assert list(summary["scenario_type"]) == list(persisted["scenario_type"])
    assert len(summary) == 4
    assert set(summary["scenario_type"]) == {
        "same_genre_continuation",
        "cross_genre_shift",
        "one_outlier_insertion",
        "repeated_consistent_insertions",
    }
    assert {
        "baseline_intent_alignment_score",
        "adaptive_intent_alignment_score",
        "adaptive_minus_baseline_intent_alignment_score",
        "baseline_adaptation_shift_score",
        "adaptive_adaptation_shift_score",
        "baseline_overreaction_penalty",
        "adaptive_overreaction_penalty",
        "baseline_diversity_retention",
        "adaptive_diversity_retention",
    }.issubset(summary.columns)
    assert_frame_equal(summary, persisted, check_exact=False, atol=1e-6, rtol=1e-6)


def test_cross_genre_shift_adaptive_improves_alignment_over_baseline() -> None:
    catalog, _, queue_state = _session_context("cross_genre_shift")
    seed_track = catalog.loc[catalog["track_id"] == queue_state.seed_track_id].iloc[0]
    baseline, adaptive, intent_profile = _session_recommendations(catalog, queue_state)

    baseline_intent = intent_alignment_score(baseline, intent_profile, top_k=5)
    adaptive_intent = intent_alignment_score(adaptive, intent_profile, top_k=5)
    baseline_shift = adaptation_shift_score(baseline, baseline, intent_profile, top_k=5)
    adaptive_shift = adaptation_shift_score(adaptive, baseline, intent_profile, top_k=5)
    adaptive_penalty = overreaction_penalty(adaptive, seed_track, intent_profile, top_k=5)

    assert adaptive_intent > baseline_intent
    assert adaptive_shift > baseline_shift
    assert adaptive_penalty >= 0.0
