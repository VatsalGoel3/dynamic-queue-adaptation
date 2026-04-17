# Dynamic Queue Adaptation

Offline prototype for intent-aware queue adaptation in music recommendation systems.

This project tests one claim: manual queue insertions are useful short-term intent signals, and using them to rerank upcoming recommendations can outperform a static seed-only queue.

## Current Status

Scaffolding and project specification are complete. Phase 2 now uses a deterministic synthetic track catalog with raw and processed data-loading helpers. Baseline recommendation, reranking, and evaluation work have not started yet.

## Dataset Strategy

Phase 2 uses a fully synthetic track catalog with explicit metadata and numeric features.

- Metadata: `track_id`, `artist_name`, `genre`, and `mood`
- Numeric features: `energy` and `tempo`
- Rationale: fastest practical option, deterministic for tests and simulation, and sufficient for later baseline and reranking experiments

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
