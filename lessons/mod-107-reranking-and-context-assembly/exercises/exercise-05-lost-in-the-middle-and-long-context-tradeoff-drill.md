# Exercise 05 — Lost-in-the-Middle and the Long-Context Trade-off Drill

**Estimated effort:** ~2.5 hours. Python, an LLM API with a long-context model (Claude 3+ / GPT-4-family / Gemini-1.5+ or a hosted open-weight long-context model), a first-stage retriever producing top-100 candidates, one reranker (from exercise 01), and an eval harness that can measure faithfulness / answer-quality on generated responses.

## Objective

Run chapter 8's A/B/C diagnostic experiment on your own data: (A) retrieve + rerank + tight assembly, (B) retrieve + no-rerank + truncate-to-fit, (C) retrieve + no-rerank + stuff-into-long-context. Measure quality, latency, and cost per arm, then run two ordering experiments on top of arm A to reproduce (or refute) the lost-in-the-middle effect on the exact model you deploy. Produce a defended verdict: for your model, corpus, and traffic, does long context obviate the reranker?

The deliverable is the numbers a level-30 rag-engineer needs to answer *"why do we still run a reranker now that the model has 200K / 1M tokens of context?"* — with evidence, not opinion.

## Prerequisites

- Chapter 8 read end-to-end.
- Chapter 5 read (for the funnel numbers you already picked).
- Chapter 7 read (assembler shape — you'll reuse the one from exercise 04 if you did it).
- Access to a long-context LLM API. Any of: [Anthropic Claude](https://docs.claude.com/en/docs/models-overview) (200K+ tokens), [OpenAI GPT-4-family](https://platform.openai.com/docs/models) (128K+), [Google Gemini 1.5](https://ai.google.dev/gemini-api/docs/models/gemini) (1M+), or an equivalent hosted open-weight model. A short-context model does not exercise arm C — call this out and pick a different model or skip arm C.
- A first-stage retriever from mod-106 that can return top-100 candidates per query.
- One reranker from chapters 2–4 (cross-encoder open-weight or hosted API).
- ~50–200 eval queries with at least binary relevance labels; ideally with gold-standard short answers so you can score answer correctness directly.
- An eval harness with faithfulness + answer relevance metrics. [RAGAS](https://docs.ragas.io/), [TruLens](https://www.trulens.org/), or [DeepEval](https://docs.confident-ai.com/); a simple LLM-judge you author for this exercise is also acceptable.

## Steps

**Step 1 — Fix the shared inputs (20 minutes).** Cache the first-stage top-100 candidates per eval query (same shape as exercise 01's step 1). Cache the reranked top-K per query using the reranker and K you picked in exercise 02.

All three arms below run against the *same* candidate pool. Anything else confuses "the reranker's contribution" with "the retriever's contribution."

**Step 2 — Arm A: retrieve → rerank → tight assembly (30 minutes).** The pipeline this module builds:

- Retrieve top-100.
- Rerank to top-K (use your K* from exercise 02; if you skipped, start with K=5 for factoid Q&A, K=10 for synthesis).
- Assemble with your exercise-04 assembler (or a greedy-pack with numeric anchors).
- Feed to the long-context LLM with a citation-per-claim prompt.

Measure per query:

- Faithfulness, answer relevance, and (if you have gold answers) answer correctness.
- Total input tokens (context + system prompt + question).
- End-to-end p50 / p95 latency (retrieve + rerank + assemble + generate).
- Estimated cost per query = input_tokens × model_input_price + output_tokens × model_output_price (look up current prices on the vendor's pricing page).

**Step 3 — Arm B: retrieve → no-rerank → truncate-to-fit (20 minutes).** The naive baseline that skips the reranker but respects a token budget:

- Retrieve top-100.
- Skip reranking. Keep the *first-stage* order (retriever's own score).
- Greedy-pack into the same token budget arm A used. Chunks are added in first-stage-order until budget is hit; chunks past the budget are dropped.
- Feed to the same long-context LLM with the same prompt.

Measure the same four numbers as arm A.

**Step 4 — Arm C: retrieve → no-rerank → stuff-into-long-context (30 minutes).** The "just give the LLM everything" arm:

- Retrieve top-100.
- Skip reranking. Skip the token budget. Concatenate all 100 chunks (with numeric anchors) into the context.
- Feed to the long-context LLM with the same prompt.

If 100 chunks exceed the model's context window, drop to top-50 (or whatever fits) and note the truncation in your writeup.

Measure the same four numbers. Expect input token counts on the order of 50–150K tokens; expect per-query cost 10–30× arm A.

**Step 5 — Build the arm-vs-arm table (15 minutes).**

| Arm | Description | Faithfulness | Answer correctness | Input tokens (median) | p95 latency | Cost per 1000 q |
|---|---|---|---|---|---|---|
| A | rerank + tight assembly | | | | | |
| B | no-rerank + truncate | | | | | |
| C | no-rerank + long-context stuff | | | | | |

The pattern chapter 8 predicts: A wins on quality-per-dollar; C occasionally matches A on some quality metrics but loses badly on cost and latency; B usually loses to A on quality because truncation is not relevance-informed. Confirm or refute on your data.

**Step 6 — The lost-in-the-middle ordering experiment (45 minutes).** Now run the ordering probe on arm A. Fix K at a value where the effect is visible — 10 or 15 works well; the effect is muted at K ≤ 5.

Three ordering variants using the *same* reranked top-K:

- **A-best-first** — chunks in rerank-score-descending order (baseline; this is what arm A already ran).
- **A-first-and-last** — top chunk first, then chunks 2..K in descending order, then a repeat of the top chunk at the end marked `[1, repeated] ...`. Chapter 7's mitigation for lost-in-the-middle.
- **A-worst-in-the-middle** — deliberately place the strongest chunk in position ⌈K/2⌉ (the middle), with all others around it. This is the "adversarial" variant that should perform worst if lost-in-the-middle bites on your model.

Rerun the eval on each variant. Compare answer correctness / faithfulness across the three.

Interpretation:

- If A-first-and-last ≥ A-best-first > A-worst-in-the-middle by a meaningful margin, lost-in-the-middle is real on your model and K; ship first-and-last (or best-first with awareness of the risk).
- If all three arms are within noise, lost-in-the-middle is not biting on your model + K; keep best-first for simplicity.
- If A-worst-in-the-middle actually wins, something is broken in your eval or your reranker is misordering — investigate before drawing any conclusion.

**Step 7 — The "just give the LLM everything" writeup (30 minutes).** In a `long-context-verdict.md`:

- The arm-vs-arm table with all numbers filled in.
- The ordering-experiment table (three A-variants × faithfulness / correctness).
- One paragraph on which arm wins on quality-per-dollar for your data, and why.
- One paragraph on whether lost-in-the-middle is empirically visible on your model. Cite [Liu et al. 2023](https://arxiv.org/abs/2307.03172) and — if you found the effect — the specific quality delta between first-and-last and worst-in-the-middle.
- One paragraph on the three "when to skip the reranker" cases from chapter 8 (narrow corpus with saturated retrieval, ultra-tight latency, single-long-document RAG). Does *your* stack match any of them?
- A concrete recommendation: keep the reranker (default), skip it (if you match a skip case), or run a fuller A/B in production before committing.

## Acceptance criteria

- All three arms (A/B/C) run on the same first-stage candidate pool and same LLM.
- Per-arm metrics reported: faithfulness, answer correctness (or answer relevance if no gold answers), input-token median, p95 latency, cost per 1000 queries.
- Ordering experiment run with all three variants (best-first, first-and-last, worst-in-the-middle) on the same K.
- Written verdict with a specific quality-per-dollar comparison across arms and a specific recommendation for the ordering variant.
- Long-context model called out explicitly (which vendor, which model version, which context window, which pricing tier).

## Stretch goals

- **The needle-in-a-haystack calibration.** Run [Kamradt's Needle-in-a-Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) test on your chosen long-context model. Does it score close to 100%? If yes, does that near-perfection carry over to your arm-C performance on real RAG queries? Chapter 8 argues it does not; verify empirically.
- **The K sweep at long context.** Extend exercise 02's K sweep out to K ∈ {20, 30, 50, 100} on the long-context model. Does the quality peak shift with the larger window, or does it stay at the same K it did on a short-context model?
- **Cost sensitivity.** Rerun the arm-A vs. arm-C comparison with the *cheapest* long-context model you have access to (e.g., a small open-weight 128K model). Does the cost gap shrink enough that arm C becomes viable? What quality do you sacrifice?
- **The RULER-style eval.** Read [Hsieh et al. 2024 (RULER)](https://arxiv.org/abs/2404.06654) and try one of their tasks (variable tracking, multi-key needle) on your model. Compare the model's *effective* context length to the advertised one on a task closer to real RAG.
- **Multi-hop query slice.** Split your eval into single-hop vs. multi-hop queries. Does the arm-A → arm-C gap look different on multi-hop queries (where synthesis across many sources matters) than on single-hop ones? This is where the "when long context genuinely helps" nuance lives.
- **The A/B in front of real traffic.** If you have a live RAG service, deploy arm A and arm C behind a feature flag on 5% of traffic and measure a real user metric (session length, retry rate, escalation-to-human). Public eval numbers are a rehearsal; live A/B is the real answer.

## Notes

- The long-context model has a per-token price. Arm C on 50–150K input tokens can cost cents per query; running 200 eval queries under arm C at $0.05 each is $10 — small, but budget it explicitly. Do not silently blow through an API budget in a classroom exercise.
- The lost-in-the-middle effect is easier to reproduce with weaker or older models. If you run this on a top-tier 2026 long-context model, the gap between A-first-and-last and A-worst-in-the-middle may be small. That is a real result — report it. Do not tune the experiment to "find" the effect.
- Arm C's faithfulness sometimes looks acceptable in aggregate but has a specific failure mode: the answer sticks to a fact from *one* of the 100 chunks, ignoring a contradicting stronger chunk. Manually eyeball 5 arm-C failures. What is the model actually doing?
- The "context precision" metric in the RAG triad (fraction of retrieved chunks that were actually used to support the answer) usually crashes on arm C. Report it — a context precision of 0.03 on arm C vs. 0.6 on arm A is the metric that most cleanly shows arm C is *paying* for 100 chunks and *using* 3.
- The three-arm comparison is the exact experiment chapter 8 recommends as the "when someone argues to skip the reranker" diagnostic. Once you have run it on your own data, you have the internal reference to short-circuit that debate.
- If arm C wins on your data — for a specific model on a specific corpus — that is a legitimate finding. Chapter 8 named three cases where the reranker can defensibly be skipped. Match your result to one of those cases (or add a fourth, with evidence).
- Chapter 8's cost argument stands independent of quality: even if arm A and arm C tied on quality, the cost-per-1000-queries column is a decisive factor at scale. Do not omit the cost column just because it feels unfair to arm C — that is precisely the point.
