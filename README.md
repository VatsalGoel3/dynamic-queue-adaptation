# Dynamic Queue Adaptation

Offline prototype for intent-aware queue adaptation in music recommendation systems.

## Overview

This repository tests whether manual queue insertions can act as short-term intent signals and improve the ranking of upcoming recommendations versus a static seed-only queue. The implementation is deterministic, Python-only, and designed to be reproducible from the committed artifacts.

## Problem Statement

Music queues often start from a seed track, but the user may change intent after adding or inserting tracks manually. A static recommender can miss that shift. This project asks whether a queue-aware reranker can react to the insertion signals without losing the seed context.

## Hypothesis

Manual insertions contain enough signal to improve the next recommendations for the current session. The expected outcome is a small but consistent gain in intent alignment, with limited loss in diversity or overreaction.

## Repository Structure

```text
.
├── src/
│   ├── data/           # synthetic catalog generation and session artifacts
│   ├── models/         # baseline recommender and adaptive reranker
│   ├── simulation/     # queue state and intent profile logic
│   └── evaluation/     # metrics, comparison pipeline, and plots
├── data/
│   ├── processed/      # committed processed catalog artifact
│   └── synthetic/      # committed synthetic session artifact
├── reports/
│   ├── results_summary.csv
│   ├── figures/
│   └── whitepaper.md
├── tests/
├── SPEC.md
├── MASTER_PLAN.md
├── DECISIONS.md
└── TASKS.md
```

## Setup

Requirements:

- Python 3.11 or newer
- No external API keys or services

Install dependencies from the repository root:

```bash
python -m pip install -e .
```

The editable install explicitly exposes the `src` package, so the `python -m src...` commands and `from src...` imports shown below work from outside the repository root after installation.

Optional validation from the repository root:

```bash
python -m pytest
```

## Quickstart

The quickest way to reproduce the committed state is to run the artifact pipeline in order after the editable install above. These commands can be run from any working directory:

```bash
python -m src.data.build_sessions
python -m src.evaluation.compare_models
python -m src.evaluation.plots
```

Expected outputs:

- `data/processed/processed_track_catalog.csv`
- `data/synthetic/synthetic_sessions.csv`
- `reports/results_summary.csv`
- `reports/figures/overall_metric_comparison.png`
- `reports/figures/scenario_intent_alignment.png`
- `reports/figures/scenario_metric_deltas.png`

## Baseline Recommendations

The baseline recommender is seed-only and deterministic. After `pip install -e .`, this example can be run from any working directory:

```bash
python - <<'PY'
from src.data.preprocess import load_processed_catalog
from src.models.baseline_seed import recommend_from_seed

catalog = load_processed_catalog()
recommendations = recommend_from_seed(
    seed_track_id="track_0000",
    top_k=20,
    catalog=catalog,
)
print(recommendations[["track_id", "genre", "mood"]].head(10).to_string(index=False))
PY
```

If you want a different seed track, replace `track_0000` with a valid `track_id` from `data/processed/processed_track_catalog.csv`.

## Adaptive Reranking

The adaptive reranker takes a `QueueState`, folds in manual insertion intent, and reranks the remaining candidates. After `pip install -e .`, this example can also be run from any working directory:

```bash
python - <<'PY'
from src.data.build_sessions import load_synthetic_sessions
from src.data.preprocess import load_processed_catalog
from src.models.adaptive_reranker import rerank_remaining_candidates
from src.simulation.queue_state import QueueState

catalog = load_processed_catalog()
session = load_synthetic_sessions().iloc[0]
queue_state = QueueState(
    seed_track_id=session["seed_track_id"],
    candidate_track_ids=(),
    manual_insertion_track_ids=session["manual_insertion_track_ids"],
    played_track_ids=(session["seed_track_id"],),
)
recommendations = rerank_remaining_candidates(queue_state, top_k=20, catalog=catalog)
print(recommendations[["track_id", "genre", "mood", "reranked_score"]].head(10).to_string(index=False))
PY
```

## Regenerating Results

To regenerate the evaluation table and plots from the committed artifacts:

```bash
python -m src.data.build_sessions
python -m src.evaluation.compare_models
python -m src.evaluation.plots
```

The comparison step rewrites `reports/results_summary.csv`. The plotting step reads that summary and rewrites the PNGs under `reports/figures/`.

## Key Artifacts

- `data/processed/processed_track_catalog.csv`: normalized catalog used by the baseline and reranker
- `data/synthetic/synthetic_sessions.csv`: deterministic session definitions used for evaluation
- `reports/results_summary.csv`: scenario-level comparison of baseline vs adaptive metrics
- `reports/figures/`: committed plots generated from the summary CSV
- `reports/whitepaper.md`: longer-form narrative for the prototype

## Headline Findings

The committed Phase 5 results show a small intent-alignment gain for the adaptive reranker in three of four deterministic scenarios. The only regression appears in the one-outlier insertion case. Diversity retention stays unchanged, and the overreaction penalty remains zero across the committed scenarios.

## Limitations

- The evaluation uses synthetic sessions, not live user traces.
- Each scenario is represented by a single deterministic example.
- There is no statistical significance analysis.
- The prototype is offline only and does not integrate with a streaming service.

## Future Work

- Expand the session set beyond the four deterministic scenarios.
- Validate against real interaction logs.
- Add significance testing and confidence reporting.
- Explore stronger reranking policies without changing the deterministic baseline.

## Current Status

Phase 6 repo polish is complete. The editable-install path now exposes `src` correctly, the README matches the supported usage, and the only remaining task in `TASKS.md` is post-ship communication.

## Non-Goals

- Frontend or consumer app
- Real-time streaming integration
- Browser extension
- External music service API integration in v1
- Production deployment
- Deep learning training pipeline
