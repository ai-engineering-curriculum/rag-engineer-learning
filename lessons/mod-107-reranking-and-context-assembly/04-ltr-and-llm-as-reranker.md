# Chapter 4 — LTR and LLM-as-Reranker: The Two Ends of the Spectrum

## Why this chapter exists

Cross-encoders (chapters 2–3) are the middle of the reranker-family spectrum: moderate latency, moderate cost, strong general-purpose quality. Two other families live at the two ends.

- **At the low-latency, high-features end: LTR rerankers.** Gradient-boosted trees (LightGBM / XGBoost) over hand-designed features — BM25 score, dense similarity, click counts, freshness, source authority. Sub-millisecond per candidate. Requires labeled data and a feature pipeline. Direct heir to the pre-deep-learning search stack.
- **At the high-quality, high-cost end: LLM-as-reranker.** Ask an LLM (GPT-4, Claude, a strong open-weight model) to score or rank the candidates. Highest quality per candidate. Slowest and most expensive. Useful for the *last mile* — the top handful of candidates on hard queries — or as a periodic gold-standard eval labeler.

This chapter covers both, when to reach for each, and how they combine with the cross-encoder middle rather than replace it.

## LTR rerankers: LightGBM, XGBoost, and the feature pipeline

Learning-to-rank is the classical IR reranking approach that predates transformer rerankers by a decade and continues to run in production at every large-scale search engine (Google, Bing, Amazon Search, Yelp, LinkedIn). Mod-105 chapter 5 introduced LTR on the lexical arm; this chapter revisits it as a *reranker over a hybrid first stage*.

### The three LTR loss families

The three standard loss families for LTR — Liu 2009's foundational survey ([*Learning to Rank for Information Retrieval*](https://www.nowpublishers.com/article/Details/INR-016)) is the reference — are:

1. **Pointwise.** Predict a relevance score per document independently; loss is regression or classification against the label. Simplest; ignores the fact that ranking is a *set* problem.
2. **Pairwise.** Predict which of two documents (per query) is more relevant. The RankNet, LambdaRank, and LambdaMART lineage (Burges 2010: [*From RankNet to LambdaRank to LambdaMART*](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)) trains on all pairs within a query. This is the standard for GBDT-based LTR.
3. **Listwise.** Predict the whole ranked list at once, optimising a listwise loss like ListNet or LambdaLoss. Higher quality when it works; harder to train; usually only worth it at very large scale.

The industry default for GBDT-based LTR is **LambdaMART** — a pairwise loss trained inside a gradient-boosted tree ensemble. Both [LightGBM](https://lightgbm.readthedocs.io/en/latest/Features.html#lambdarank) and [XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html) support LambdaMART natively with the `lambdarank` / `rank:ndcg` objective.

### The feature pipeline

The interesting thing about LTR is *not* the model — LambdaMART on 500 trees is nearly a solved problem in 2026 — but the **features**. Every feature you construct is a signal the tree ensemble can exploit; missing features is usually why an LTR reranker underperforms.

The four feature buckets:

1. **Query-document match features (the reranker's core signal).**
    - BM25 score (from the lexical arm).
    - Dense similarity (cosine of query embedding · chunk embedding).
    - TF-IDF, BM25 on title vs. body separately.
    - Cross-encoder score (yes, you can feed a cross-encoder's output as a feature into an LTR model that also has other features — this is the "cross-encoder + LTR" stack chapter 5 mentions).
    - RRF score (from chapter 2 of mod-106).
    - Query-document overlap counts (exact-match count of query tokens in chunk).

2. **Query features.**
    - Query length in tokens.
    - Presence of specific query patterns — "how", "what", "compare", "vs", boolean operators.
    - Query embedding (as a fixed-dim vector fed to the model).
    - Detected intent class (from a small classifier or an LLM classifier).

3. **Document / chunk features.**
    - Chunk length.
    - Document freshness (age of the source; `updated_at` if you have it).
    - Source authority — a per-source score you maintain (canonical docs > forum posts > user-generated content, or similar for your corpus).
    - Structural role of the chunk (title vs. section-body vs. footnote).
    - Historical click-through rate, if you have telemetry.

4. **Interaction / behavioural features.**
    - Click count on this doc for similar queries (query-similarity clustering + click aggregation).
    - Dwell time, follow-up query rate — signals mod-111 (MLOps) covers logging pipelines for.
    - CTR-per-position with position bias removed (see Joachims 2017 [*Unbiased Learning-to-Rank with Biased Feedback*](https://arxiv.org/abs/1608.04468)).

The behavioural features (bucket 4) are the ones that make LTR *outperform* an off-the-shelf cross-encoder on your data. A cross-encoder has never seen your users; an LTR model trained on your click logs has. If your product has real usage, harvesting that signal into features is usually the highest-leverage reranker improvement you can make.

### The interface — a LightGBM LambdaMART reranker

```python
import lightgbm as lgb
import numpy as np

# Training: X is a (n_examples, n_features) matrix. y is a per-example relevance label (0..4).
# group is a per-query group size: group[i] = number of candidates for query i (must sum to n_examples).

model = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    label_gain=[0, 1, 3, 7, 15],   # gain per relevance grade — the default for graded relevance
)
model.fit(X_train, y_train, group=train_groups, eval_at=[10])

# Inference at query time:
# 1. Retrieve top-N candidates.
# 2. Build a per-candidate feature vector (all the features above).
# 3. Score with model.predict(feature_matrix) — this is 100 rows × ~50 features = sub-millisecond.
# 4. Sort by score, take top-K.
scores = model.predict(feature_matrix)
```

The LightGBM reference: [*Learning to rank*](https://lightgbm.readthedocs.io/en/latest/Features.html#lambdarank) in the LightGBM docs. XGBoost: [*Learning to Rank*](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html).

### When LTR pays and when it doesn't

**LTR pays when:**
- You have real behavioural signals (clicks, dwell, feedback thumbs) that no off-the-shelf model has seen.
- You have a **very tight latency budget** at rerank time (single-digit ms) and a cross-encoder does not fit.
- You have a mature feature pipeline (mod-105 chapter 5 built the outline of one) and the operational muscle to maintain it.

**LTR does not pay when:**
- You have no labels beyond binary relevance and no behavioural data. LambdaMART is starving without either.
- Your query volume is too low for click signals to be statistically meaningful (a few thousand queries per day is a rough floor; below that, cross-encoders trained on public data are the safer bet).
- You are early-stage. An LTR pipeline is real infrastructure — feature stores, training pipelines, drift monitoring. Come back to it when you have logs.

The pattern many teams end up with is **cross-encoder + LTR**: use the cross-encoder to produce a score, feed that score plus a few dozen behavioural / metadata features into a LightGBM ranker, and rerank the top-100 in single-digit milliseconds after the cross-encoder pass. This gives you both the semantic quality of a cross-encoder and the specific-to-your-users signal of LTR.

## LLM-as-reranker

At the other end: use a strong LLM as the reranker. This is the newest reranker family in production and the one that gets the most attention in recent research (RankGPT, [Sun et al. 2023, arXiv:2304.09542](https://arxiv.org/abs/2304.09542); RankZephyr, [Pradeep et al. 2023, arXiv:2312.02724](https://arxiv.org/abs/2312.02724); RankVicuna, [Pradeep et al. 2023, arXiv:2309.15088](https://arxiv.org/abs/2309.15088)).

### Three shapes: pointwise, pairwise, listwise LLM rerankers

The same three loss shapes from classical LTR reappear as three prompt shapes:

1. **Pointwise LLM reranker.** Prompt the LLM once per candidate: "given this query and this passage, is the passage relevant? Return a score 0–10." N calls per query; simple; each call is a small prompt.

2. **Pairwise LLM reranker.** Prompt per pair of candidates: "which of these two passages is more relevant to the query?" O(N²) calls in the naive version, though smart algorithms (e.g., a tournament) reduce it to O(N log N). Higher quality than pointwise when the LLM struggles with absolute scores.

3. **Listwise LLM reranker.** Prompt once with all N candidates: "here are 20 passages, return their IDs in order of relevance." One call per query; the model handles the ranking as a set problem, which matches how humans do the task. This is the RankGPT / RankZephyr shape; on strong models (GPT-4, Claude) it produces the best reranker quality reported for pure-LLM approaches.

Listwise is where most production LLM-rerankers land today, because a single API call per query is much cheaper than N calls per query if your LLM can hold all N candidates in context.

### The listwise prompt shape

A prompt template that reflects the RankGPT paper's approach:

```
You are RankGPT, an intelligent assistant that can rank passages based on their relevance to the query.

The user query is:
Query: {query}

Here are 20 passages, numbered [1] through [20]:
[1] {passage_1_text}
[2] {passage_2_text}
...
[20] {passage_20_text}

Rank the passages above based on their relevance to the query. The passages should be listed in descending
order using identifiers. The most relevant passages should be listed first. The output format should be
[] > [] > ... > [], e.g., [3] > [7] > [1] > [12] > ...

Only respond with the ranking, do not include any other text.
```

The parsing on the client side is a regex or a simple split-on-`>` to extract the ordered IDs.

Two known caveats for listwise LLM reranking that every production deployment has to handle:

- **Position bias.** LLMs are known to prefer passages nearer the front of the list. Mitigation: randomly permute the candidates before prompting, then aggregate several permutations (2–3 is usually enough).
- **Sliding-window aggregation.** If N > what fits in context (typically N > 20 with detailed passages), rank in overlapping windows and merge. RankGPT uses a sliding-window with overlap; the paper's algorithm is worth reading directly.

### The three cases where LLM reranking is worth the cost

LLM reranking is 10–100× more expensive per query than cross-encoder reranking. That cost only pays off in specific settings:

1. **The last-mile precision problem.** A cross-encoder gets you top-10; you need the *ordering within the top-3* to be very reliable (a legal-research or medical-guidance product where the LLM answer will name the source verbatim). Use the cross-encoder to filter 100→20; use an LLM reranker to sort 20→top-3. The LLM only sees the top-20, so cost is bounded.
2. **Very hard queries.** Multi-hop, comparative, contradiction-aware. "Which of these three chunks contradicts the query claim?" is a task cross-encoders handle poorly and LLMs handle naturally.
3. **A one-time or periodic gold-standard evaluation labeler** — not as an online reranker, but as an offline judge to *train* a cheaper reranker. This is the RAG-Fusion / distillation pattern: use GPT-4 to label thousands of (query, candidate) pairs, then fine-tune a MiniLM cross-encoder on those labels for online serving. You pay LLM cost once, cross-encoder cost forever.

### The three cases where LLM reranking is the wrong pick

1. **Sustained high QPS with a rerank latency budget under ~1 s.** No LLM-as-reranker will fit. Cross-encoder is your ceiling.
2. **Budget-constrained products.** Doing an LLM rerank on every query at scale will dominate your inference bill. Only do it on a fraction of queries — the hard ones — or on the last mile as in case 1 above.
3. **Reproducibility-sensitive contexts.** LLM outputs drift as vendors update models. If your reranker's behavior must be stable for audit, deploying an LLM reranker without version-pinning (and understanding that vendor deprecations force re-eval) is risky.

## Combining families — a realistic level-30 reranker stack

The families in chapters 2–4 are not exclusive. A real stack at any serious size uses several.

The pattern that shows up repeatedly in production:

```
    first stage: hybrid retrieve → top-100      (mod-106 — recall stage)
        │
        ▼
    cross-encoder rerank → top-20                (chapter 2 — semantic-quality stage)
        │
        ▼
    LTR rerank with 30–50 behavioural/features → top-10    (chapter 4 first half — personalisation stage)
        │
        ▼
    (optional) LLM rerank of the top-10 → top-3    (chapter 4 second half — last-mile stage)
        │
        ▼
    context assembly (chapters 6–7)
```

Not every stack needs every layer. Small stacks often stop after the cross-encoder. Very-large-scale enterprise search runs all four. The point is that these are not either-or choices — each layer strips out a different failure mode.

## What you should be able to do after this chapter

- Explain the three LTR loss families (pointwise, pairwise, listwise) and name LambdaMART as the standard GBDT-based reranker.
- Enumerate the four feature buckets an LTR reranker exploits (query-doc, query, doc, interaction) and identify which buckets a cross-encoder does *not* have access to.
- Reason about when LTR is worth its ops cost (real behavioural signals, tight latency budget, mature feature pipeline) and when it is not.
- Name the three LLM-as-reranker prompt shapes (pointwise, pairwise, listwise) and identify listwise as the production default.
- Recognise the position-bias problem for listwise LLM reranking and know the mitigation (random permutation + aggregation).
- Design a multi-layer reranker stack (retrieve → cross-encoder → LTR → optional LLM) and defend which layer strips which failure mode.

## Further reading

- Liu, T.-Y. (2009). *Learning to Rank for Information Retrieval.* Foundations and Trends in Information Retrieval. [nowpublishers.com](https://www.nowpublishers.com/article/Details/INR-016) — the canonical LTR survey.
- Burges, C. J. C. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview.* Microsoft Research technical report. [msr technical report](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf) — the LambdaMART reference.
- Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z., Yin, D., and Ren, Z. (2023). *Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents.* EMNLP. [arXiv:2304.09542](https://arxiv.org/abs/2304.09542) — the RankGPT paper.
- Pradeep, R., Sharifymoghaddam, S., and Lin, J. (2023). *RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models.* [arXiv:2309.15088](https://arxiv.org/abs/2309.15088).
- Pradeep, R., Sharifymoghaddam, S., and Lin, J. (2023). *RankZephyr: Effective and Robust Zero-Shot Listwise Reranking is a Breeze!* [arXiv:2312.02724](https://arxiv.org/abs/2312.02724).
- Joachims, T., Swaminathan, A., and Schnabel, T. (2017). *Unbiased Learning-to-Rank with Biased Feedback.* WSDM. [arXiv:1608.04468](https://arxiv.org/abs/1608.04468) — the reference on debiasing click-based training data.
- LightGBM: [Learning to Rank documentation](https://lightgbm.readthedocs.io/en/latest/Features.html#lambdarank).
- XGBoost: [Learning to Rank documentation](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html).
