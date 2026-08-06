# Exercise 03 — HyDE and Multi-Query Fanout: Rewriting Workshop

**Estimated effort:** ~3 hours. Python, an LLM API for rewriting, and the two-arm hybrid stack from exercises 01–02.

## Objective

Implement two of the query-rewriting patterns from chapter 4 — HyDE and multi-query fanout — end-to-end, run them against a labeled eval set, and price them against the "vanilla hybrid retriever" baseline on both quality and cost. The goal is a defended decision on which pattern (if either) is worth shipping on your corpus, at what latency budget, and for which query classes.

## Prerequisites

- Chapter 4 (query rewriting) read.
- Exercises 01 and 02 done. You have a two-arm hybrid retriever with RRF fusion and a labeled eval set with a train/test split.
- Access to an LLM for rewriting. Options in decreasing order of cost sensitivity:
    - A local small model (Llama 3.1 8B Instruct via Ollama, or similar). Free per call after setup.
    - A hosted small model (Claude Haiku, GPT-4o mini). Cheap per call but network-latent.
    - A hosted frontier model (Claude Sonnet, GPT-4o). Overkill for this exercise but instructive to compare against a small model.
- Ability to time your LLM calls (wall-clock ms per rewrite).

## Steps

**Step 1 — Baseline recall (10 min).** From exercise 01, you already have the vanilla two-arm RRF recall@100 on your test split. Record it as the baseline.

Also measure the baseline latency: end-to-end retrieve() time from query in to top-100 out, averaged over the test queries and at p95. Record.

**Step 2 — Implement HyDE (40 min).** Write a `hyde_retrieve(query)` function:

1. Prompt the LLM with a HyDE-style instruction (chapter 4 gives a prompt template). Use `temperature=0.3` to reduce variance.
2. Take the LLM's output as the "hypothetical passage."
3. Embed the hypothetical passage using the same embedder your dense arm uses.
4. Do dense retrieval against the hypothetical embedding, keeping BM25 anchored on the original query.
5. RRF-fuse the two arms, `k=60`.

Points worth naming in your code:

- **Do not HyDE the lexical arm.** The hypothetical is a passage; it will bag-of-words match the wrong things. Keep BM25 on the original query.
- **Cache the hypothetical.** For repeat queries, cache HyDE outputs — no reason to re-generate the same hypothetical.
- **Log the hypothetical.** You will want to eyeball the LLM's outputs on failure cases.

Run `hyde_retrieve` over the test split. Record recall@100 and end-to-end p95 latency (including the LLM call).

**Step 3 — Implement multi-query fanout (40 min).** Write a `multi_query_retrieve(query, k=4)` function:

1. Prompt the LLM to generate `k=4` paraphrases (chapter 4 gives a prompt template). Parse the numbered list.
2. Run the full hybrid retrieval (BM25 + dense + RRF fusion) for the original query and for each of the 4 paraphrases — that is 5 hybrid retrievals total.
3. RRF-fuse the 5 ranked outputs into one top-100 fused list, `k=60`.

Points worth naming:

- **Parallelize the 5 hybrid retrievals.** They are independent; issue them concurrently to your search engine. Serial issuance adds 5× latency; parallel adds just the slowest one.
- **Handle LLM output-parsing failures.** If the LLM returns fewer than `k` rewrites (or malformed output), fall back gracefully — use whatever rewrites parsed, plus the original.
- **Log the rewrites.** Eyeball to check for identity rewrites (LLM returned the original as one of the paraphrases) or drift (rewrites lost the entity anchor).

Run `multi_query_retrieve` over the test split. Record recall@100 and end-to-end p95 latency.

**Step 4 — Compare per-query cost (20 min).** For each of the three configurations (vanilla RRF, HyDE, multi-query fanout `k=4`), record:

- **LLM calls per query** (0 / 1 / 1).
- **LLM output tokens per query** (average across the test set).
- **Dollar cost per 1000 queries** at your model's list price.
- **Retrievals per query** (1 / 1 / 5 hybrid retrievals — count each hybrid as 1 for cost purposes, since it's 2 arms in parallel).
- **End-to-end p95 latency.**

Report as a table.

**Step 5 — Quality comparison, overall and by query class (30 min).** On the test split, compare recall@100 for the three configurations overall. Then split by query class:

- **Short queries (< 5 tokens) vs. long queries.**
- **Queries containing a rare-entity anchor** (a specific product code, error code, or SKU pattern — pick one your corpus contains) **vs. queries without.**
- **Queries where the vanilla RRF top-1 has cosine < 0.5 to the query** (proxy for "query is hard for the embedder") vs. queries with strong top-1 scores.

For each bucket, report recall@100 per configuration. Look for the specific class where each rewriting pattern wins:

- HyDE typically wins on short, informal queries against formal corpora.
- Multi-query fanout typically wins on short queries with wide-semantic-surface answers.
- Both typically lose on rare-entity queries (they dilute the anchor).

If your data shows something different, name it — it might be the specifics of your corpus, or it might be a bug in your prompt.

**Step 6 — The "chain everything" ablation (20 min).** Implement `hyde_plus_multi_query(query)` — HyDE followed by multi-query fanout on the resulting hypothetical, then RRF over all 5 rewrite retrievals plus the original. Report recall@100 and cost. Chapter 4 warned this rarely pays off; verify.

**Step 7 — Model tier ablation (30 min).** Re-run HyDE and multi-query with **two different LLM tiers** — one small (Haiku, Llama 8B, or GPT-4o mini) and one mid-size (Sonnet, GPT-4o). Report recall@100 and cost for each. The question chapter 4 raised: does the frontier model materially outperform the small model in the rewriter role, or is the small model good enough?

**Step 8 — Write it up (~15 min).** Author `writeup.md` containing:

- The cost table from step 4.
- The overall recall table (vanilla, HyDE, multi-query, chain-everything) with p95 latency.
- The per-query-class table from step 5.
- The model-tier ablation from step 7.
- A recommendation: which rewriting pattern (if any) would you ship, and under what conditions? Answer the four questions:
    - Which pattern gives the best recall-per-dollar?
    - Which pattern gives the best recall-per-ms of added latency?
    - Which query class is the rewriting pattern *for*? Would you consider routing (chapter 5) — running the pattern only on that class?
    - Is any pattern strong enough that the added complexity (LLM in the retrieval path, prompt drift over time, per-query variance from LLM sampling) is worth the operational cost?

## Acceptance criteria

- HyDE implemented with the lexical arm anchored on the original query.
- Multi-query fanout implemented with parallel dispatch of the `k+1` hybrid retrievals.
- Cost table with LLM calls, tokens, dollars/1000 queries, retrievals, and p95 latency per configuration.
- Overall recall table on the test split.
- Per-query-class analysis across at least one axis (length, rare-entity, hard-for-embedder).
- Model-tier ablation across at least two LLM tiers.
- Recommendation section answers all four questions with numeric evidence.
- If any pattern regresses recall on any query class, the writeup identifies it explicitly rather than reporting only overall averages.

## Stretch goals

- **Step-back prompting.** Implement `step_back_retrieve` per chapter 4. Compare to HyDE and multi-query on your test split. On what query class does step-back win?
- **Query decomposition.** Pick 5–10 multi-hop queries from your test set (or hand-craft them if your set is single-hop). Implement `decompose_and_retrieve` and measure the recall lift on those queries specifically. What's the latency and cost delta?
- **Learned rewriter.** Instead of prompt-engineering the rewriter, fine-tune a small model on `(query, human-written-rewrite)` pairs. Compare to the prompt-engineered version. Does the fine-tuned model produce better rewrites for the same latency? (Reference: Ma et al. 2023, [arXiv:2305.14283](https://arxiv.org/abs/2305.14283).)
- **Rewriting for the reranker.** Some rewriting patterns are used not to retrieve better but to help a downstream reranker (mod-107). Skip the recall metric; instead, run each rewriting pattern's outputs through a cross-encoder reranker and measure precision@10. Does HyDE's hypothetical passage help the reranker in ways it doesn't help the retriever?
- **Per-query prompt selection.** Implement a small routing head that decides for each query which rewriting pattern (or none) to run. Use the query-class insight from step 5 to hand-author the routing rules; measure end-to-end recall and cost with the router vs. always-on rewriting.

## Notes

- Watch your token bill during this exercise. Multi-query fanout at `k=4` with a mid-size model on 300 test queries adds up to real money.
- LLM outputs are stochastic even at temperature 0.3. Repeat the experiments 2–3 times if you want a stable read; average or median the results. The variance is instructive too — a rewriting pattern that gives recall 0.85 sometimes and 0.79 other times is a red flag for production.
- Do not judge rewriting on "does the LLM output look good?" — you cannot eyeball this reliably. Only recall@k on a labeled set counts.
- The right operating point for a rewriting pattern is often "only when the vanilla retrieval is uncertain" (top-1 score below threshold, or user re-issues a similar query). That is the natural bridge to chapter 5 (routing) and worth naming in the writeup even if you don't implement it here.
