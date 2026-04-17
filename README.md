# Dynamic Queue Adaptation

Offline prototype for intent-aware queue adaptation in music recommendation systems.

This project tests one claim: manual queue insertions are useful short-term intent signals, and using them to rerank upcoming recommendations can outperform a static seed-only queue.

## Current Status

Scaffolding and project specification are complete. Phase 2 now uses a deterministic synthetic track catalog, a processed catalog artifact, and deterministic synthetic listening sessions saved under `data/synthetic`. The seed-only baseline recommender is now implemented, while adaptive reranking and evaluation work remain ahead.

## Dataset Strategy

Phase 2 uses a fully synthetic track catalog with explicit metadata and numeric features, plus a small deterministic simulation layer for session-based evaluation setup.

- Metadata: `track_id`, `artist_name`, `genre`, and `mood`
- Numeric features: `energy` and `tempo`
- Session artifacts: seed track, autoplay continuation candidates, and manual queue insertions across four scenario types
- Rationale: fastest practical option, deterministic for tests and simulation, and sufficient for later baseline and reranking experiments

## Phase 2 Data Artifacts

- `data/processed/processed_track_catalog.csv`: normalized synthetic catalog for downstream experiments
- `data/synthetic/synthetic_sessions.csv`: deterministic session scenarios for queue-adaptation simulations
- `src/data/build_sessions.py`: session builders and artifact-writing helpers for the Phase 2 simulation layer
- `src/models/baseline_seed.py`: deterministic seed-only baseline recommendations over the processed catalog

## Scope

- Python-only offline simulation
- Seed-only baseline recommender
- Queue-adaptive reranker
- Synthetic or lightweight public-data-backed evaluation
- Concise writeup and communication artifacts

## Non-Goals

- Frontend or consumer app
- Real-time streaming integration
- Browser extension
- External music service API integration in V1
- Production deployment
- Deep learning training pipeline
