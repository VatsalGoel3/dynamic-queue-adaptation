from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.compare_models import DEFAULT_RESULTS_SUMMARY_PATH
from src.evaluation.metrics import METRIC_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURES_DIR = REPO_ROOT / "reports" / "figures"
DEFAULT_PLOT_FILENAMES = (
    "overall_metric_comparison.png",
    "scenario_intent_alignment.png",
    "scenario_metric_deltas.png",
)


def _pretty_metric_name(metric_name: str) -> str:
    return metric_name.replace("_", " ").title()


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if not values:
        return 0.0
    weight_sum = float(sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return float(sum(value * weight for value, weight in zip(values, weights, strict=True)) / weight_sum)


def _load_results_summary(results_path: Path = DEFAULT_RESULTS_SUMMARY_PATH) -> pd.DataFrame:
    summary = pd.read_csv(results_path)
    required_columns = {"scenario_type", "session_count"}
    for metric_name in METRIC_COLUMNS:
        required_columns.update(
            {
                f"baseline_{metric_name}",
                f"adaptive_{metric_name}",
                f"adaptive_minus_baseline_{metric_name}",
            }
        )
    missing_columns = required_columns.difference(summary.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"results summary is missing required columns: {missing}")
    return summary


def _save_grouped_bar_chart(
    figure_path: Path,
    labels: list[str],
    baseline_values: list[float],
    adaptive_values: list[float],
    title: str,
    ylabel: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    positions = list(range(len(labels)))
    bar_width = 0.36

    axis.bar(
        [position - bar_width / 2 for position in positions],
        baseline_values,
        width=bar_width,
        label="Baseline",
        color="#64748b",
    )
    axis.bar(
        [position + bar_width / 2 for position in positions],
        adaptive_values,
        width=bar_width,
        label="Adaptive",
        color="#0f766e",
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_heatmap(
    figure_path: Path,
    rows: list[str],
    columns: list[str],
    values: list[list[float]],
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    data = pd.DataFrame(values, index=rows, columns=columns)
    max_abs = float(data.abs().to_numpy().max() or 0.0)
    limit = max(0.01, max_abs)
    image = axis.imshow(data.to_numpy(), cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")

    axis.set_title(title)
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels(columns, rotation=20, ha="right")
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(rows)

    for row_index, row_name in enumerate(rows):
        for column_index, column_name in enumerate(columns):
            cell_value = float(data.loc[row_name, column_name])
            text_color = "white" if abs(cell_value) >= limit * 0.5 else "black"
            axis.text(
                column_index,
                row_index,
                f"{cell_value:+.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Adaptive - Baseline")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def generate_plots(
    results_path: Path = DEFAULT_RESULTS_SUMMARY_PATH,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> list[Path]:
    """Generate compact evaluation plots from the committed summary CSV."""
    summary = _load_results_summary(results_path)
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    session_weights = summary["session_count"].astype(float).tolist()
    metric_labels = [_pretty_metric_name(metric_name) for metric_name in METRIC_COLUMNS]
    baseline_means = [
        _weighted_mean(summary[f"baseline_{metric_name}"].astype(float).tolist(), session_weights)
        for metric_name in METRIC_COLUMNS
    ]
    adaptive_means = [
        _weighted_mean(summary[f"adaptive_{metric_name}"].astype(float).tolist(), session_weights)
        for metric_name in METRIC_COLUMNS
    ]
    overall_path = figures_dir / DEFAULT_PLOT_FILENAMES[0]
    _save_grouped_bar_chart(
        overall_path,
        metric_labels,
        baseline_means,
        adaptive_means,
        title="Baseline vs Adaptive Metric Means",
        ylabel="Score",
    )
    saved_paths.append(overall_path)

    scenario_labels = summary["scenario_type"].tolist()
    scenario_baseline = summary["baseline_intent_alignment_score"].astype(float).tolist()
    scenario_adaptive = summary["adaptive_intent_alignment_score"].astype(float).tolist()
    scenario_path = figures_dir / DEFAULT_PLOT_FILENAMES[1]
    _save_grouped_bar_chart(
        scenario_path,
        scenario_labels,
        scenario_baseline,
        scenario_adaptive,
        title="Scenario-wise Intent Alignment",
        ylabel="Intent alignment score",
    )
    saved_paths.append(scenario_path)

    delta_rows = [
        summary[f"adaptive_minus_baseline_{metric_name}"].astype(float).tolist()
        for metric_name in METRIC_COLUMNS
    ]
    delta_matrix = [list(values) for values in zip(*delta_rows, strict=True)]
    delta_columns = [_pretty_metric_name(metric_name) for metric_name in METRIC_COLUMNS]
    delta_path = figures_dir / DEFAULT_PLOT_FILENAMES[2]
    _save_heatmap(
        delta_path,
        rows=scenario_labels,
        columns=delta_columns,
        values=delta_matrix,
        title="Adaptive Minus Baseline Metric Deltas",
    )
    saved_paths.append(delta_path)

    return saved_paths


def main() -> None:
    generated_paths = generate_plots()
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
