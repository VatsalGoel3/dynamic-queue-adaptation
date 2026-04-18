from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.simulation.intent_update import IntentProfile

NUMERIC_FEATURE_COLUMNS = ("energy_normalized", "tempo_normalized")
METRIC_COLUMNS = (
    "intent_alignment_score",
    "adaptation_shift_score",
    "overreaction_penalty",
    "diversity_retention",
)


def _require_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = set(required_columns).difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"recommendations are missing required columns: {missing}")


def _top_k_frame(recommendations: pd.DataFrame, top_k: int | None) -> pd.DataFrame:
    if top_k is None or top_k >= len(recommendations):
        return recommendations.reset_index(drop=True)
    return recommendations.head(top_k).reset_index(drop=True)


def _rank_weights(length: int) -> list[float]:
    if length <= 0:
        return []
    raw_weights = list(range(length, 0, -1))
    total_weight = float(sum(raw_weights))
    return [weight / total_weight for weight in raw_weights]


def _weighted_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    weights = _rank_weights(len(values))
    return float(sum(value * weight for value, weight in zip(values, weights, strict=True)))


def _clip_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def _dominant_target_labels(intent_profile: IntentProfile) -> tuple[str, str]:
    return str(intent_profile.dominant_genre), str(intent_profile.dominant_mood)


def _track_alignment_score(
    track: pd.Series,
    target_genre: str,
    target_mood: str,
    target_energy: float,
    target_tempo: float,
) -> float:
    numeric_similarity = 1.0 - (
        (
            abs(float(track["energy_normalized"]) - target_energy)
            + abs(float(track["tempo_normalized"]) - target_tempo)
        )
        / len(NUMERIC_FEATURE_COLUMNS)
    )
    numeric_similarity = max(0.0, min(1.0, numeric_similarity))
    genre_match = 1.0 if str(track["genre"]) == target_genre else 0.0
    mood_match = 1.0 if str(track["mood"]) == target_mood else 0.0
    return round(0.35 * numeric_similarity + 0.4 * genre_match + 0.25 * mood_match, 6)


def _list_alignment_score(
    recommendations: pd.DataFrame,
    target_genre: str,
    target_mood: str,
    target_energy: float,
    target_tempo: float,
    top_k: int | None,
) -> float:
    if recommendations.empty:
        return 0.0
    ranked_recommendations = _top_k_frame(recommendations, top_k)
    per_track_scores = [
        _track_alignment_score(
            track=track,
            target_genre=target_genre,
            target_mood=target_mood,
            target_energy=target_energy,
            target_tempo=target_tempo,
        )
        for _, track in ranked_recommendations.iterrows()
    ]
    return _clip_score(_weighted_mean(per_track_scores))


def intent_alignment_score(
    recommendations: pd.DataFrame,
    intent_profile: IntentProfile,
    top_k: int | None = None,
) -> float:
    """Measure how closely the ranked list matches the current session intent."""
    _require_columns(recommendations, NUMERIC_FEATURE_COLUMNS + ("genre", "mood"))
    target_genre, target_mood = _dominant_target_labels(intent_profile)
    return _list_alignment_score(
        recommendations=recommendations,
        target_genre=target_genre,
        target_mood=target_mood,
        target_energy=float(intent_profile.energy_normalized),
        target_tempo=float(intent_profile.tempo_normalized),
        top_k=top_k,
    )


def adaptation_shift_score(
    recommendations: pd.DataFrame,
    baseline_recommendations: pd.DataFrame,
    intent_profile: IntentProfile,
    top_k: int | None = None,
) -> float:
    """Measure improvement over the seed-only baseline on a 0-1 centered scale."""
    current_alignment = intent_alignment_score(
        recommendations=recommendations,
        intent_profile=intent_profile,
        top_k=top_k,
    )
    baseline_alignment = intent_alignment_score(
        recommendations=baseline_recommendations,
        intent_profile=intent_profile,
        top_k=top_k,
    )
    return _clip_score(0.5 + 0.5 * (current_alignment - baseline_alignment))


def overreaction_penalty(
    recommendations: pd.DataFrame,
    seed_track: pd.Series,
    intent_profile: IntentProfile,
    top_k: int | None = None,
) -> float:
    """Conservatively penalize weak-evidence shifts away from the seed context."""
    _require_columns(recommendations, NUMERIC_FEATURE_COLUMNS + ("genre", "mood"))
    target_genre, target_mood = _dominant_target_labels(intent_profile)
    ranked_recommendations = _top_k_frame(recommendations, top_k)
    target_signal_share = _weighted_mean(
        [
            0.5 * (1.0 if str(track["genre"]) == target_genre else 0.0)
            + 0.5 * (1.0 if str(track["mood"]) == target_mood else 0.0)
            for _, track in ranked_recommendations.iterrows()
        ]
    )
    seed_misalignment = 0.5 * (
        1.0 if target_genre != str(seed_track["genre"]) else 0.0
    ) + 0.5 * (1.0 if target_mood != str(seed_track["mood"]) else 0.0)
    excess_shift = max(0.0, target_signal_share - float(intent_profile.pivot_strength))
    return _clip_score(excess_shift * seed_misalignment)


def diversity_retention(
    recommendations: pd.DataFrame,
    baseline_recommendations: pd.DataFrame,
    top_k: int | None = None,
) -> float:
    """Measure how much list diversity survives relative to the seed-only baseline."""
    _require_columns(recommendations, ("genre", "mood"))
    _require_columns(baseline_recommendations, ("genre", "mood"))
    ranked_recommendations = _top_k_frame(recommendations, top_k)
    ranked_baseline = _top_k_frame(baseline_recommendations, top_k)
    baseline_genre_count = max(1, int(ranked_baseline["genre"].nunique()))
    baseline_mood_count = max(1, int(ranked_baseline["mood"].nunique()))
    genre_retention = min(int(ranked_recommendations["genre"].nunique()), baseline_genre_count) / baseline_genre_count
    mood_retention = min(int(ranked_recommendations["mood"].nunique()), baseline_mood_count) / baseline_mood_count
    return _clip_score((genre_retention + mood_retention) / 2.0)


def calculate_model_metrics(
    recommendations: pd.DataFrame,
    baseline_recommendations: pd.DataFrame,
    seed_track: pd.Series,
    intent_profile: IntentProfile,
    top_k: int | None = None,
) -> dict[str, float]:
    """Return all Phase 5 metrics for a single model/session pair."""
    return {
        "intent_alignment_score": intent_alignment_score(
            recommendations=recommendations,
            intent_profile=intent_profile,
            top_k=top_k,
        ),
        "adaptation_shift_score": adaptation_shift_score(
            recommendations=recommendations,
            baseline_recommendations=baseline_recommendations,
            intent_profile=intent_profile,
            top_k=top_k,
        ),
        "overreaction_penalty": overreaction_penalty(
            recommendations=recommendations,
            seed_track=seed_track,
            intent_profile=intent_profile,
            top_k=top_k,
        ),
        "diversity_retention": diversity_retention(
            recommendations=recommendations,
            baseline_recommendations=baseline_recommendations,
            top_k=top_k,
        ),
    }
