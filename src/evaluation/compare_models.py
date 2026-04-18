from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.build_sessions import SCENARIO_TYPES, load_synthetic_sessions
from src.data.preprocess import load_processed_catalog
from src.evaluation.metrics import METRIC_COLUMNS, calculate_model_metrics
from src.models.adaptive_reranker import rerank_remaining_candidates
from src.models.baseline_seed import recommend_from_seed
from src.simulation.intent_update import update_intent_profile
from src.simulation.queue_state import QueueState

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_SUMMARY_PATH = REPO_ROOT / "reports/results_summary.csv"


def _queue_state_from_session(session_row: pd.Series) -> QueueState:
    return QueueState(
        seed_track_id=str(session_row["seed_track_id"]),
        candidate_track_ids=(),
        manual_insertion_track_ids=tuple(session_row["manual_insertion_track_ids"]),
        played_track_ids=(str(session_row["seed_track_id"]),),
    )


def _session_metric_records(
    catalog: pd.DataFrame,
    sessions: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, session_row in sessions.iterrows():
        queue_state = _queue_state_from_session(session_row)
        seed_track = catalog.loc[catalog["track_id"] == queue_state.seed_track_id].iloc[0]
        exclude_track_ids = (
            *queue_state.played_track_ids,
            *queue_state.manual_insertion_track_ids,
        )
        baseline_recommendations = recommend_from_seed(
            queue_state.seed_track_id,
            top_k=top_k,
            catalog=catalog,
            exclude_track_ids=exclude_track_ids,
        )
        adaptive_recommendations = rerank_remaining_candidates(
            queue_state,
            top_k=top_k,
            catalog=catalog,
        )
        intent_profile = update_intent_profile(queue_state, catalog)

        for model_name, recommendations in (
            ("baseline", baseline_recommendations),
            ("adaptive", adaptive_recommendations),
        ):
            model_metrics = calculate_model_metrics(
                recommendations=recommendations,
                baseline_recommendations=baseline_recommendations,
                seed_track=seed_track,
                intent_profile=intent_profile,
                top_k=top_k,
            )
            records.append(
                {
                    "session_id": session_row["session_id"],
                    "scenario_type": session_row["scenario_type"],
                    "model": model_name,
                    **model_metrics,
                }
            )

    return pd.DataFrame(records)


def _summarize_session_metrics(session_metrics: pd.DataFrame) -> pd.DataFrame:
    scenario_order = [scenario for scenario in SCENARIO_TYPES if scenario in set(session_metrics["scenario_type"])]
    grouped = (
        session_metrics.groupby(["scenario_type", "model"], sort=False)[list(METRIC_COLUMNS)]
        .mean()
        .reset_index()
    )

    summary_rows: list[dict[str, object]] = []
    for scenario_type in scenario_order:
        scenario_metrics = grouped.loc[grouped["scenario_type"] == scenario_type].set_index("model")
        baseline_count = int(
            session_metrics.loc[session_metrics["scenario_type"] == scenario_type, "session_id"].nunique()
        )
        row: dict[str, object] = {
            "scenario_type": scenario_type,
            "session_count": baseline_count,
        }
        for metric_name in METRIC_COLUMNS:
            baseline_value = float(scenario_metrics.loc["baseline", metric_name])
            adaptive_value = float(scenario_metrics.loc["adaptive", metric_name])
            row[f"baseline_{metric_name}"] = baseline_value
            row[f"adaptive_{metric_name}"] = adaptive_value
            row[f"adaptive_minus_baseline_{metric_name}"] = adaptive_value - baseline_value
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def run_comparison_pipeline(
    output_path: Path = DEFAULT_RESULTS_SUMMARY_PATH,
    top_k: int = 5,
) -> pd.DataFrame:
    """Compare baseline and adaptive rankings across the canonical session groups."""
    catalog = load_processed_catalog()
    sessions = load_synthetic_sessions()
    session_metrics = _session_metric_records(catalog=catalog, sessions=sessions, top_k=top_k)
    if session_metrics.empty:
        raise ValueError("comparison pipeline produced no session metrics")

    summary = _summarize_session_metrics(session_metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, float_format="%.6f")
    return summary


def main() -> None:
    run_comparison_pipeline()


if __name__ == "__main__":
    main()
