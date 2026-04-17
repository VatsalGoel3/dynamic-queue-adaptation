from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def _normalize_track_ids(track_ids: Iterable[str] | str | None) -> tuple[str, ...]:
    if track_ids is None:
        return ()
    if isinstance(track_ids, str):
        return (track_ids,)
    return tuple(track_ids)


@dataclass(frozen=True)
class QueueState:
    """Small immutable view of the queue context used by the simulator."""

    seed_track_id: str
    candidate_track_ids: tuple[str, ...] = ()
    manual_insertion_track_ids: tuple[str, ...] = ()
    played_track_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_track_ids",
            _normalize_track_ids(self.candidate_track_ids),
        )
        object.__setattr__(
            self,
            "manual_insertion_track_ids",
            _normalize_track_ids(self.manual_insertion_track_ids),
        )
        object.__setattr__(
            self,
            "played_track_ids",
            _normalize_track_ids(self.played_track_ids),
        )

    @property
    def excluded_track_ids(self) -> tuple[str, ...]:
        """Return queue-linked track IDs in stable first-seen order."""
        ordered_track_ids = [
            self.seed_track_id,
            *self.played_track_ids,
            *self.manual_insertion_track_ids,
            *self.candidate_track_ids,
        ]
        deduplicated: list[str] = []
        seen: set[str] = set()
        for track_id in ordered_track_ids:
            if track_id in seen:
                continue
            deduplicated.append(track_id)
            seen.add(track_id)
        return tuple(deduplicated)
