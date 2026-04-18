from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plots import generate_plots


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
