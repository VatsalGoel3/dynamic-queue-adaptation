# When a Queue Stops Listening: A Small Prototype for Intent-Aware Music Ranking

Music queues expose a simple failure mode: the first seed track is often treated as the whole story, even after the listener starts making manual changes. In practice, those changes can be the clearest signal of what the user wants next. If someone inserts a few songs that lean in a different direction, a static seed-only recommender can keep insisting on the original theme long after the user has moved on.

That was the problem this repository set out to test.

The project asked a narrow question: can manual queue insertions act as short-term intent signals, and can a lightweight reranker use that signal without causing unstable reshuffles? The hypothesis was conservative. The expectation was not a dramatic lift, but a small and consistent improvement in intent alignment, with bounded changes elsewhere.

To test that idea, the repository compares two deterministic strategies:

- a seed-only baseline that ranks candidates from the original track alone
- a queue-aware adaptive reranker that updates its view of intent from manual insertions and reranks the remaining candidates

Everything is offline, Python-only, and reproducible from committed artifacts. The evaluation uses a processed catalog, synthetic sessions, and a fixed comparison pipeline that writes the summary table and figures under `reports/`. The whitepaper and README both describe the same setup: synthetic sessions, a small scenario set, and no external API dependencies.

The committed results are modest, and that is part of the point.

In the summary table, the adaptive reranker improves intent alignment in three of four deterministic scenarios: `same_genre_continuation` (+0.010591), `cross_genre_shift` (+0.025990), and `repeated_consistent_insertions` (+0.031026). The one caution case is `one_outlier_insertion`, where the adaptive reranker underperforms the baseline by -0.009308. That is the expected edge case for a conservative system: a single noisy insertion should not cause it to overreact.

The guardrails held. `overreaction_penalty` stays at `0.0` for both strategies across the committed scenarios, and `diversity_retention` stays at `1.0`. The adaptive path shifts slightly relative to the baseline, but it does not collapse into wholesale list churn. That matters because the goal here was not to make recommendations more volatile. It was to let the queue influence ranking just enough to reflect what the listener is doing now.

There are clear limitations. The evaluation is synthetic, offline, and deterministic. Each scenario is represented by a single committed example, so the results are directional rather than statistically conclusive. There is no live user traffic, no production integration, and no significance testing. This is a prototype, not a claim of deployment readiness.

Even so, a few lessons are already visible.

First, queue behavior is a useful signal surface. The listener’s manual insertions carry more context than the seed track alone. Second, the useful response is bounded adaptation, not aggressive re-ranking. The best-looking outcome here is not a large shift; it is a small improvement without collateral damage to diversity or stability. Third, evaluation design matters. A small deterministic setup makes it easy to understand what changed, but it also makes it easy to overread the results if you forget how limited the sample is.

That is the main takeaway from this prototype: queue-aware ranking can respond to local intent without becoming chaotic, but the evidence here is still narrow and conservative by design.

Repository: https://github.com/VatsalGoel3/dynamic-queue-adaptation

For the full narrative and evaluation details, see the README and whitepaper in the repo.
