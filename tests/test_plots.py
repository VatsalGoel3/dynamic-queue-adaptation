from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import METRIC_COLUMNS
from src.evaluation.plots import _load_results_summary, _weighted_mean, generate_plots
import src.evaluation.plots as plots_module


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


def test_generate_plots_uses_non_uniform_session_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary_path = tmp_path / "results_summary.csv"
    figures_dir = tmp_path / "figures"
    summary = pd.DataFrame(
        [
            {
                "scenario_type": "scenario_a",
                "session_count": 1,
                "baseline_intent_alignment_score": 0.0,
                "adaptive_intent_alignment_score": 0.2,
                "adaptive_minus_baseline_intent_alignment_score": 0.2,
                "baseline_adaptation_shift_score": 0.1,
                "adaptive_adaptation_shift_score": 0.2,
                "adaptive_minus_baseline_adaptation_shift_score": 0.1,
                "baseline_overreaction_penalty": 0.3,
                "adaptive_overreaction_penalty": 0.2,
                "adaptive_minus_baseline_overreaction_penalty": -0.1,
                "baseline_diversity_retention": 0.9,
                "adaptive_diversity_retention": 0.8,
                "adaptive_minus_baseline_diversity_retention": -0.1,
            },
            {
                "scenario_type": "scenario_b",
                "session_count": 3,
                "baseline_intent_alignment_score": 1.0,
                "adaptive_intent_alignment_score": 0.6,
                "adaptive_minus_baseline_intent_alignment_score": -0.4,
                "baseline_adaptation_shift_score": 0.5,
                "adaptive_adaptation_shift_score": 0.7,
                "adaptive_minus_baseline_adaptation_shift_score": 0.2,
                "baseline_overreaction_penalty": 0.2,
                "adaptive_overreaction_penalty": 0.1,
                "adaptive_minus_baseline_overreaction_penalty": -0.1,
                "baseline_diversity_retention": 0.7,
                "adaptive_diversity_retention": 0.5,
                "adaptive_minus_baseline_diversity_retention": -0.2,
            },
        ]
    )
    summary.to_csv(summary_path, index=False)

    captured_overall_values: dict[str, list[float]] = {}
    original_save_grouped_bar_chart = plots_module._save_grouped_bar_chart

    def capture_overall_values(
        figure_path: Path,
        labels: list[str],
        baseline_values: list[float],
        adaptive_values: list[float],
        title: str,
        ylabel: str,
    ) -> None:
        if title == "Baseline vs Adaptive Metric Means":
            captured_overall_values["labels"] = labels
            captured_overall_values["baseline_values"] = baseline_values
            captured_overall_values["adaptive_values"] = adaptive_values

    monkeypatch.setattr(plots_module, "_save_grouped_bar_chart", capture_overall_values)

    try:
        generated_paths = generate_plots(results_path=summary_path, figures_dir=figures_dir)
    finally:
        monkeypatch.setattr(plots_module, "_save_grouped_bar_chart", original_save_grouped_bar_chart)

    assert [path.name for path in generated_paths] == [
        "overall_metric_comparison.png",
        "scenario_intent_alignment.png",
        "scenario_metric_deltas.png",
    ]
    assert captured_overall_values["labels"] == [
        "Intent Alignment Score",
        "Adaptation Shift Score",
        "Overreaction Penalty",
        "Diversity Retention",
    ]

    weights = summary["session_count"].astype(float).tolist()
    for index, metric_name in enumerate(METRIC_COLUMNS):
        expected_baseline = _weighted_mean(
            summary[f"baseline_{metric_name}"].astype(float).tolist(),
            weights,
        )
        expected_adaptive = _weighted_mean(
            summary[f"adaptive_{metric_name}"].astype(float).tolist(),
            weights,
        )

        assert captured_overall_values["baseline_values"][index] == pytest.approx(expected_baseline)
        assert captured_overall_values["adaptive_values"][index] == pytest.approx(expected_adaptive)

    plain_mean = summary["baseline_intent_alignment_score"].mean()
    weighted_mean = captured_overall_values["baseline_values"][0]
    assert weighted_mean != pytest.approx(plain_mean)
