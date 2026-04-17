# Dynamic Queue Adaptation

Offline prototype for intent-aware queue adaptation in music recommendation systems.

This project tests one claim: manual queue insertions are useful short-term intent signals, and using them to rerank upcoming recommendations can outperform a static seed-only queue.

## Current Status

Scaffolding and project specification are complete. Implementation has not started yet.

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
