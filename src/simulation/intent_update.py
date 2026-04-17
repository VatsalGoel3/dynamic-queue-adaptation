from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from src.simulation.queue_state import QueueState

NUMERIC_FEATURE_COLUMNS = ["energy_normalized", "tempo_normalized"]
REQUIRED_INTENT_COLUMNS = {
    "track_id",
    "genre",
    "mood",
    *NUMERIC_FEATURE_COLUMNS,
}
SEED_WEIGHT = 1.0
INSERTION_WEIGHTS = (1.25, 1.5, 1.75)


@dataclass(frozen=True)
class IntentProfile:
    anchor_track_id: str
    source_track_ids: tuple[str, ...]
    dominant_genre: str
    dominant_mood: str
    energy_normalized: float
    tempo_normalized: float
    pivot_strength: float


def _validate_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_INTENT_COLUMNS.difference(catalog.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"catalog is missing required columns: {missing}")
    return catalog.copy()


def _rows_for_track_ids(catalog: pd.DataFrame, track_ids: tuple[str, ...]) -> pd.DataFrame:
    indexed_catalog = catalog.set_index("track_id", drop=False)
    missing_track_ids = [track_id for track_id in track_ids if track_id not in indexed_catalog.index]
    if missing_track_ids:
        missing = ", ".join(sorted(missing_track_ids))
        raise ValueError(f"unknown track_id values: {missing}")
    return indexed_catalog.loc[list(track_ids)].reset_index(drop=True)


def _weight_for_insertion(index: int) -> float:
    if index < len(INSERTION_WEIGHTS):
        return INSERTION_WEIGHTS[index]
    return INSERTION_WEIGHTS[-1]


def _weighted_numeric_centroid(rows: pd.DataFrame, weights: list[float]) -> dict[str, float]:
    numeric_rows = rows.loc[:, NUMERIC_FEATURE_COLUMNS]
    total_weight = float(sum(weights))
    centroid = {
        column: round(
            float((numeric_rows[column] * weights).sum() / total_weight),
            6,
        )
        for column in NUMERIC_FEATURE_COLUMNS
    }
    return centroid


def _dominant_label(
    seed_label: str, labels: list[str], weights: list[float]
) -> str:
    weighted_votes: dict[str, float] = {}
    for label, weight in zip(labels, weights, strict=True):
        weighted_votes[label] = weighted_votes.get(label, 0.0) + weight
    max_vote = max(weighted_votes.values())
    tied_labels = [label for label, vote in weighted_votes.items() if vote == max_vote]
    if seed_label in tied_labels:
        return seed_label
    return sorted(tied_labels)[0]


def _single_insertion_consistency(seed_row: pd.Series, insertion_row: pd.Series) -> float:
    numeric_distance = float(
        (
            insertion_row[NUMERIC_FEATURE_COLUMNS]
            .sub(seed_row[NUMERIC_FEATURE_COLUMNS])
            .abs()
            .mean()
        )
    )
    full_category_mismatch = (
        insertion_row["genre"] != seed_row["genre"]
        and insertion_row["mood"] != seed_row["mood"]
    )
    numeric_penalty = max(0.0, (numeric_distance - 0.45) / 0.35) * 0.45
    category_penalty = 0.25 if full_category_mismatch else 0.0
    return round(max(0.2, 0.8 - numeric_penalty - category_penalty), 6)


def _multi_insertion_consistency(insertion_rows: pd.DataFrame) -> float:
    numeric_distances = []
    for left_index, right_index in combinations(range(len(insertion_rows)), 2):
        left_row = insertion_rows.iloc[left_index]
        right_row = insertion_rows.iloc[right_index]
        numeric_distances.append(
            float(
                (
                    left_row[NUMERIC_FEATURE_COLUMNS]
                    .sub(right_row[NUMERIC_FEATURE_COLUMNS])
                    .abs()
                    .mean()
                )
            )
        )

    pairwise_similarity = 1.0 - (sum(numeric_distances) / len(numeric_distances))
    genre_consistency = (
        float(insertion_rows["genre"].nunique() == 1)
        if not insertion_rows.empty
        else 0.0
    )
    mood_consistency = (
        float(insertion_rows["mood"].nunique() == 1) if not insertion_rows.empty else 0.0
    )
    return round(
        min(
            1.0,
            0.45
            + (0.35 * pairwise_similarity)
            + (0.10 * genre_consistency)
            + (0.10 * mood_consistency),
        ),
        6,
    )


def _shift_signal(
    seed_row: pd.Series,
    dominant_genre: str,
    dominant_mood: str,
    insertion_rows: pd.DataFrame,
    insertion_weights: list[float],
) -> float:
    insertion_centroid = _weighted_numeric_centroid(insertion_rows, insertion_weights)
    numeric_shift = (
        abs(insertion_centroid["energy_normalized"] - float(seed_row["energy_normalized"]))
        + abs(insertion_centroid["tempo_normalized"] - float(seed_row["tempo_normalized"]))
    ) / len(NUMERIC_FEATURE_COLUMNS)
    genre_shift = 0.25 if dominant_genre != seed_row["genre"] else 0.0
    mood_shift = 0.15 if dominant_mood != seed_row["mood"] else 0.0
    return min(1.0, round(0.05 + numeric_shift + genre_shift + mood_shift, 6))


def _pivot_strength(
    seed_row: pd.Series,
    dominant_genre: str,
    dominant_mood: str,
    insertion_rows: pd.DataFrame,
    insertion_weights: list[float],
) -> float:
    if insertion_rows.empty:
        return 0.0

    total_insert_weight = sum(insertion_weights)
    insert_weight_share = total_insert_weight / (SEED_WEIGHT + total_insert_weight)
    if len(insertion_rows) == 1:
        consistency = _single_insertion_consistency(seed_row, insertion_rows.iloc[0])
    else:
        consistency = _multi_insertion_consistency(insertion_rows)
    shift_signal = _shift_signal(
        seed_row,
        dominant_genre,
        dominant_mood,
        insertion_rows,
        insertion_weights,
    )
    return round(insert_weight_share * consistency * shift_signal, 6)


def update_intent_profile(
    queue_state: QueueState,
    catalog: pd.DataFrame,
) -> IntentProfile:
    """Blend seed and manual insertions into a deterministic queue intent profile."""
    validated_catalog = _validate_catalog(catalog)
    seed_row = _rows_for_track_ids(validated_catalog, (queue_state.seed_track_id,)).iloc[0]

    insertion_track_ids = queue_state.manual_insertion_track_ids
    if not insertion_track_ids:
        return IntentProfile(
            anchor_track_id=queue_state.seed_track_id,
            source_track_ids=(queue_state.seed_track_id,),
            dominant_genre=str(seed_row["genre"]),
            dominant_mood=str(seed_row["mood"]),
            energy_normalized=round(float(seed_row["energy_normalized"]), 6),
            tempo_normalized=round(float(seed_row["tempo_normalized"]), 6),
            pivot_strength=0.0,
        )

    insertion_rows = _rows_for_track_ids(validated_catalog, insertion_track_ids)
    insertion_weights = [
        _weight_for_insertion(index) for index in range(len(insertion_track_ids))
    ]
    source_rows = pd.concat(
        [
            pd.DataFrame([seed_row]),
            insertion_rows,
        ],
        ignore_index=True,
    )
    source_weights = [SEED_WEIGHT, *insertion_weights]
    centroid = _weighted_numeric_centroid(source_rows, source_weights)
    dominant_genre = _dominant_label(
        str(seed_row["genre"]),
        [str(value) for value in source_rows["genre"].tolist()],
        source_weights,
    )
    dominant_mood = _dominant_label(
        str(seed_row["mood"]),
        [str(value) for value in source_rows["mood"].tolist()],
        source_weights,
    )
    pivot_strength = _pivot_strength(
        seed_row,
        dominant_genre,
        dominant_mood,
        insertion_rows,
        insertion_weights,
    )

    return IntentProfile(
        anchor_track_id=queue_state.seed_track_id,
        source_track_ids=(queue_state.seed_track_id, *insertion_track_ids),
        dominant_genre=dominant_genre,
        dominant_mood=dominant_mood,
        energy_normalized=centroid["energy_normalized"],
        tempo_normalized=centroid["tempo_normalized"],
        pivot_strength=pivot_strength,
    )
