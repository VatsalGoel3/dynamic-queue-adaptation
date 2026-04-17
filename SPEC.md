# Specification

## Problem

Current seed-based queues can continue the original listening trajectory even after a user manually inserts tracks that clearly signal a short-term shift in intent.

This project focuses on whether those manual "play next" actions should influence the recommendation tail that follows.

## Hypothesis

Manual queue insertions are higher-signal indicators of short-term intent than the original seed alone, so using them to rerank future recommendations should improve alignment to current session intent.

## Proposed Solution

Build a simple offline reranking prototype.

- Baseline: rank candidate tracks using similarity to the original seed only.
- Adaptive model: recompute an intent representation from the seed plus manually inserted tracks, with heavier weights on insertions and simple smoothing to reduce overreaction.

## Baselines

- Seed-only recommender
- Recent-history-only variant if it can be added later without expanding scope

## Evaluation Metrics

- Intent alignment score
- Adaptation speed
- Overreaction penalty
- Diversity retention
- Skip-risk proxy

The final evaluation should report 3 to 5 of these, depending on what is most defensible with the chosen dataset setup.

## Risks

- Overfitting too hard to one outlier insertion
- Synthetic sessions that are too unrealistic
- Claims that overstate what offline simulation can prove

## Limitations

- Offline simulation only
- No live user feedback loop
- No production infrastructure
- Results should be framed as directional evidence, not proof of production lift
