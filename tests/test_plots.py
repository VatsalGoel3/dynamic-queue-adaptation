from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import METRIC_COLUMNS
from src.evaluation.plots import _load_results_summary, _weighted_mean, generate_plots


def test_generate_plots_writes_expected_figures(tmp_path: Path) -> None:
    figures_dir = tmp_path / "figures"

    generated_paths = generate_plots(figures_dir=figures_dir)

    assert [path.name for path in generated_paths] == [
        "overall_metric_comparison.png",
        "scenario_intent_alignment.png",
        "scenario_metric_deltas.png",
    ]
    for generated_path in generated_paths:
        assert generated_path.exists()
        assert generated_path.stat().st_size > 0


def test_overall_plot_uses_session_count_weighting_from_summary() -> None:
    summary = _load_results_summary()
    expected_columns = {
        "scenario_type",
        "session_count",
        *{
            f"{prefix}_{metric_name}"
            for metric_name in METRIC_COLUMNS
            for prefix in (
                "baseline",
                "adaptive",
                "adaptive_minus_baseline",
            )
        },
    }

    assert set(summary.columns) == expected_columns
    assert list(summary["scenario_type"]) == [
        "same_genre_continuation",
        "cross_genre_shift",
        "one_outlier_insertion",
        "repeated_consistent_insertions",
    ]

    session_weights = summary["session_count"].astype(float).tolist()

    for metric_name in METRIC_COLUMNS:
        baseline_values = summary[f"baseline_{metric_name}"].astype(float).tolist()
        adaptive_values = summary[f"adaptive_{metric_name}"].astype(float).tolist()
        expected_baseline = float(
            (summary[f"baseline_{metric_name}"] * summary["session_count"]).sum()
            / summary["session_count"].sum()
        )
        expected_adaptive = float(
            (summary[f"adaptive_{metric_name}"] * summary["session_count"]).sum()
            / summary["session_count"].sum()
        )

        assert _weighted_mean(baseline_values, session_weights) == pytest.approx(expected_baseline)
        assert _weighted_mean(adaptive_values, session_weights) == pytest.approx(expected_adaptive)


def test_weighted_mean_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        _weighted_mean([1.0, 2.0], [1.0])

    with pytest.raises(ValueError):
        _weighted_mean([1.0, 2.0], [0.0, 0.0])
