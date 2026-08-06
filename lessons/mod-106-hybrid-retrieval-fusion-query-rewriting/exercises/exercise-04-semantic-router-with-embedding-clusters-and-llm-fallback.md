# Exercise 04 — Semantic Router: Embedding-Nearest-Cluster with LLM Fallback

**Estimated effort:** ~3 hours. Python, an embedding model, an LLM API for the fallback, and a small routing labeled set.

## Objective

Build a cascade router (rule-based → embedding-nearest-cluster → LLM fallback) that routes queries across three retrievers, measure its routing accuracy and end-to-end retrieval quality, and demonstrate the fallback path handles low-confidence queries gracefully. The goal is the *pattern* — a production-shaped routing stack that spends cheap most of the time and expensive-but-reliable at the tail — and the empirical evidence that the cascade beats any single mechanism on cost-adjusted accuracy.

## Prerequisites

- Chapter 5 (semantic routing) read.
- A stack with **at least three distinct retrievers** the router can dispatch to. Pick or synthesize:
    - **Real-world:** three sub-corpora of your organization (support articles, product catalog, code / API reference).
    - **Educational:** three BEIR datasets treated as separate corpora (e.g., SciFact for scientific claims, FiQA for financial questions, NQ or MS MARCO for general web queries). Each dataset becomes a "route."
    - **Minimal:** three synthetic corpora built from a single dataset by topic-filtering. Doable but less interesting.
- A labeled routing set: at least **100 (query, correct-route)** pairs. If you're using BEIR datasets as routes, the queries in each dataset are labeled implicitly by which dataset they came from. Bootstrap 30–40 queries from each of your 3 routes into a labeled routing set of 90–120.
- An embedding model (any sentence-transformer works).
- Access to a small LLM for the fallback (Claude Haiku, GPT-4o mini, or Llama 3 8B — chapter 5 argued a big model is overkill).

## Steps

**Step 1 — Ground truth setup (20 min).** Curate the labeled routing set:

- 30–50 queries per route (aim for ~100–150 total).
- For each query, the correct route label (one of your three routes).
- Include ~10 "hard" queries per split — queries that could plausibly belong to two routes ("what does the printer error code E1234 mean?" could be code-reference OR support).
- Save as `routing_labeled.jsonl`.

Split 70/30 into train/test (seeded). Every metric below reports on the test split unless noted.

**Step 2 — Rule-based fast path (30 min).** Author 3–5 simple rules that catch the trivially-routable queries:

- Regex on URL patterns, product-code patterns (`SKU-\d+`, `Q-\d+`), function-name patterns (`camelCaseIdentifier` or `snake_case_identifier`).
- Explicit keyword-triggered routes ("cancel my subscription" → `SUPPORT`).

For each rule, record the (route, confidence) it emits. Confidence is 1.0 for regex matches; use a lower default (0.7) for keyword rules.

Run over the test split. Record:

- **Coverage:** what fraction of test queries the rules catch (fired on).
- **Precision when fired:** for the queries that fired, what fraction routed correctly.

You should see coverage in the 10–30% range and precision above 95% on the queries that fired. If precision is low, your rules are wrong; tighten them.

**Step 3 — Embedding-nearest-cluster router (45 min).** Set up the centroid router:

- Pick 5–10 "prototype utterances" per route. These are hand-written queries clearly belonging to each route. Store as `route_prototypes.json`.
- Embed each prototype. Compute the centroid per route (mean of prototype embeddings, then normalize).
- At query time: embed the query, compute cosine to each centroid, output the top-scoring route + its cosine as the confidence.

Sweep the confidence threshold: for `θ ∈ {0.4, 0.5, 0.6, 0.7, 0.8}`, run the centroid router on the training split and report:

- **Coverage:** fraction of queries where the top-1 route's cosine ≥ θ (the router "commits" to a route).
- **Precision when committing:** for those queries, fraction routed correctly.
- **Precision on the fallthrough:** for queries below θ, what a random-route baseline achieves (this is the "cost of falling through").

Pick a threshold that keeps precision on committed queries above ~92% while keeping coverage above ~60%. This is the sweet-spot chapter 5 named.

**Step 4 — LLM router fallback (30 min).** For the queries the rule-based path and centroid router both fail to commit on, use an LLM router:

- Structured prompt as in chapter 5: list of routes with descriptions; ask the model to pick one and return a `confidence` (1–5 scale).
- Use structured output (function calling / JSON schema) with a strict route-name enum to prevent hallucinated routes.

Test the LLM router on the training-split fallthrough queries. Record precision.

**Step 5 — Cascade wiring (30 min).** Wire the three mechanisms into one router:

```python
def cascade_route(query: str) -> tuple[str, float, str]:
    """Returns (route, confidence, mechanism)."""
    # Try rule-based first
    if r := rule_match(query):
        return r.route, r.confidence, "rule"
    # Fall through to centroid router
    centroid = centroid_route(query)
    if centroid.confidence >= centroid_threshold:
        return centroid.route, centroid.confidence, "centroid"
    # Final fallback: LLM router
    llm = llm_route(query)
    return llm.route, llm.confidence, "llm"
```

Run over the test split. Report per-mechanism:

- **Coverage** (fraction of queries hitting each layer).
- **Precision** (fraction routed correctly when reaching that layer).
- **Cumulative precision** (fraction routed correctly overall).
- **Cumulative latency** — measured end-to-end for the cascade. Should be dominated by the rule + centroid path on 85%+ of queries; the LLM only fires on ~10–15%.

**Step 6 — Fallback for a routing error (30 min).** Design and implement one of the fallback strategies from chapter 5. Recommended: **route-and-check.**

- After routing, run retrieval with the predicted retriever.
- If the top-1 retrieval score is below a per-route threshold (indicating the retriever isn't finding anything), fall back to a broader retriever (e.g., a global retriever over all three corpora, or re-route with the LLM router).

Measure on the test split:

- **How often the fallback fires** (fraction of queries where top-1 score is below threshold).
- **How often the fallback improves the result** — for the queries that fell back, does the alternative retriever return docs the original retriever missed?

If the fallback rarely fires or rarely improves, tighten the threshold. If it fires too often (>20%), your primary router is under-confidence-calibrated; investigate.

**Step 7 — Measure end-to-end retrieval, not just routing (20 min).** Routing accuracy is a means, not the end. Retrieval quality is the end. On the test split:

- **Baseline:** route every query to a *single* retriever (say, a merged all-corpora retriever). Measure recall@100.
- **Router-based retrieval:** use the cascade router to pick a per-query retriever. Measure recall@100 on the per-route retrievers.

The router-based retrieval should beat the single-retriever baseline. If it doesn't:

- Your routing is worse than you think (routing accuracy on the test set is not translating into retrieval wins).
- Your per-route retrievers are individually weaker than the merged retriever (which sometimes happens when a corpus is small).
- Chapter 5's "the one-corpus baseline" warning applies — investigate whether the routing is worth the complexity.

**Step 8 — Write it up (~15 min).** Author `writeup.md` containing:

- The rule-based results (coverage, precision when fired).
- The centroid-router threshold sweep with the pick.
- The LLM-router results on the fallthrough set.
- The cascade end-to-end table (per-mechanism coverage, precision, cumulative).
- End-to-end retrieval quality: router-based vs. merged-retriever baseline.
- The fallback strategy and its firing rate / improvement rate.
- A one-paragraph recommendation: would you ship this router? Under what latency budget? What would you change?

## Acceptance criteria

- Cascade router implemented with all three mechanisms (rule → centroid → LLM), in order.
- Labeled routing set with ≥ 100 queries, split into train/test (70/30, seeded).
- Centroid-threshold sweep with a defended pick.
- Per-mechanism coverage, precision, latency reported.
- Fallback strategy (route-and-check or one of the alternatives from chapter 5) implemented.
- End-to-end recall@100 for router-based vs. merged-retriever baseline reported.
- If the router does not beat the merged-retriever baseline, the writeup names why and whether the complexity is still worth it.
- All LLM outputs use structured output (function calling / JSON schema) with route-name enum enforcement; no free-text route parsing.

## Stretch goals

- **Train a supervised classifier.** Fine-tune a small transformer (DistilBERT, MiniLM) on your labeled routing set. Add it as a fourth mechanism (or replace the centroid router with it). Compare its precision and latency to the centroid router.
- **Dual-route on low confidence.** For queries where centroid confidence is between two thresholds (say, 0.55–0.7), run *both* candidate routes' retrievers and RRF-fuse their outputs. Compare to the single-route result on that subset.
- **Route the fusion algorithm itself.** From exercise 02, you know which query classes favor which fusion. Route the fusion algorithm: RRF for most queries, weighted-sum for the "large arm-quality-gap" queries. Measure recall lift over always-RRF.
- **Route the rewriting decision.** From exercise 03, you know HyDE and multi-query fanout have distinct query-class fits. Route the rewriting decision: HyDE only for short/informal queries, multi-query only for short/wide-surface queries. Measure recall vs. cost.
- **Prometheus-style routing metrics.** Instrument the cascade router to emit per-mechanism counters, timing histograms, and confidence distributions. Set up an SLO on "LLM-router fire rate" (e.g., "warn if > 25%" — that means the centroid router is regressing). Chapter 5's "cheap-first" invariant becomes a live SLO.
- **Adversarial routing test.** Craft 20 adversarial queries that look like one route but really belong to another ("acme 4402 stopped working" — support? product-catalog?). Measure per-mechanism performance on those. The LLM router should beat the centroid router on these.

## Notes

- The centroid router's precision depends heavily on how distinct your prototype utterances are per route. If two routes' prototypes are semantically similar, no threshold will save you — write more-distinctive prototypes.
- Do not route without a fallback. A routing error on a rule-based or centroid decision that goes uncorrected becomes a full retrieval failure. The fallback path is chapter 5's most important pattern.
- LLM router calls at temperature 0 are still non-deterministic across API versions and provider quirks. Log the LLM output verbatim; a "reproducibility bug" a month from now is often the LLM saying something new.
- Latency measurements are only meaningful if you measure at the p95, not the mean. A cascade with p50 = 5 ms and p95 = 900 ms (the LLM tail) is a very different animal from p50 = 5 ms, p95 = 15 ms. Report both.
