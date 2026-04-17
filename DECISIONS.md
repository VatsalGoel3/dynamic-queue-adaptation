# Decisions

## Frozen Decisions

| Date | Decision | Status |
| --- | --- | --- |
| 2026-04-17 | Python project | Frozen |
| 2026-04-17 | Offline simulation only | Frozen |
| 2026-04-17 | No frontend | Frozen |
| 2026-04-17 | No external music service API integration in V1 | Frozen |
| 2026-04-17 | Baseline plus adaptive reranker only | Frozen |
| 2026-04-17 | Small scope with target of 12 to 20 focused hours | Frozen |
| 2026-04-17 | Phase 2 dataset strategy is a fully synthetic track catalog with explicit metadata and numeric features | Frozen |

## Working Rule

If a future idea conflicts with a frozen decision, cut it instead of reopening scope.

## Rationale Notes

- The synthetic catalog is the fastest practical option for Phase 2.
- Deterministic generation keeps tests and offline simulations reproducible.
- Explicit metadata plus numeric features are sufficient for the planned baseline and adaptive reranking work.
