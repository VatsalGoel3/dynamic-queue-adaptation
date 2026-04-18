# Intent-Aware Queue Adaptation for Music Recommendation Systems

## Abstract

This prototype evaluates whether manual queue insertions provide a useful short-term intent signal for music recommendation. A seed-only baseline recommends from the original track context, while an adaptive reranker reorders the remaining candidates after observed insertions. The current Phase 5 work focuses on offline evaluation only.

## Problem Statement

Seed-only recommendation can miss immediate session intent once a listener starts adding tracks to a queue. The goal here is not to solve long-horizon personalization, but to measure whether a lightweight reranker can track the local shift more accurately than a static baseline without overreacting.

## Method

The system compares two ranking strategies over the same processed catalog and deterministic synthetic sessions:

- Baseline: recommend from the seed track only.
- Adaptive: rerank remaining candidates after the queue state is updated with manual insertions.

Evaluation is computed from the committed summary artifact at `reports/results_summary.csv` and visualized in `reports/figures/`.

## Experimental Setup

- Catalog: deterministic processed synthetic track catalog
- Sessions: four synthetic scenario types
- Top-k: fixed by repository config and reused by the comparison pipeline
- Comparison unit: one baseline/adaptive pair per scenario
- Output artifact: `reports/results_summary.csv`

The four scenario types are:

- `same_genre_continuation`
- `cross_genre_shift`
- `one_outlier_insertion`
- `repeated_consistent_insertions`

## Metrics

The evaluation uses four simple, bounded metrics:

- `intent_alignment_score`: how well the ranked list matches the inferred session intent
- `adaptation_shift_score`: how much the adaptive list improves over the baseline
- `overreaction_penalty`: a conservative check on excessive shifting
- `diversity_retention`: how much of the baseline diversity survives the rerank

## Results Summary

The committed results show a narrow, directional improvement in intent alignment for the adaptive reranker in three of the four scenarios. The largest gain appears in `repeated_consistent_insertions`, followed by `cross_genre_shift`; `same_genre_continuation` also improves slightly. The `one_outlier_insertion` scenario is the only case where adaptive intent alignment drops below baseline, which is the expected caution case for a conservative reranker.

Across these scenarios, the overreaction penalty remains at zero, and diversity retention stays unchanged at 1.0. That is consistent with the design goal of favoring small, bounded adjustments rather than dramatic list reshuffles.

## Limitations

This is a prototype evaluation, not production validation. The sample size is tiny, the sessions are synthetic, the scenarios are deterministic, and there is no statistical significance analysis. The figures are useful for inspecting directional behavior, but they should not be treated as evidence of live-world uplift.

## Future Work

- Expand the evaluation set beyond four synthetic scenarios
- Add statistical comparison across more sessions
- Test on real catalog and session data once available
- Revisit ranking behavior if stronger pivots create visible overreaction
