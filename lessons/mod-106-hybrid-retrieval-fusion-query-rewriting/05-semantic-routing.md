# Chapter 5 — Semantic Routing: Getting the Query to the Right Retriever

## Why this chapter exists

Chapters 2 and 3 tune the output fusion. Chapter 4 tunes the input query. This chapter tunes the **decision**: given a query, which retriever(s), which prompt template, which index do we send it to at all?

The core insight: a production retrieval system rarely has just one corpus and one retriever. Real stacks look like this:

- A support-articles corpus (hybrid retriever tuned for docs).
- A product-catalog corpus (BM25-heavy retriever with structured filters for SKU / price).
- A code / API-reference corpus (symbol-first BM25 with dense fallback).
- A troubleshooting knowledge base (dense retriever with paraphrase-tolerant embedder).
- A conversational-shortcut router that answers "hi" / "what can you do?" / "reset my password" without hitting retrieval at all.

Routing the query "how do I return a broken toaster?" to the support-articles retriever and the query "acme model 4402 price" to the product-catalog retriever is not a nice-to-have — the *wrong* retriever produces a nonsense result no fusion can fix. This chapter walks the three routing mechanisms — intent classification, embedding-nearest-cluster, and LLM router — and the latency-vs-reliability budget you spend on each.

## The three routing mechanisms

### 1. Intent classification (small model, sub-10 ms)

Train a small classifier (a fine-tuned small transformer, a simple linear classifier over embeddings, or even a rule-based regex system) that maps a query to one of *N* intents. Route on the predicted intent.

**Mechanism.** Collect labeled `(query, intent)` pairs from historical logs, ~500–5,000 per intent. Fine-tune a small model (DistilBERT, MiniLM, a small ALBERT — any 20–100M-parameter transformer) as a text classifier. Deploy behind a `/classify` endpoint. Each query does one forward pass; predicted class chooses the retriever.

**Latency.** 2–15 ms for a small transformer on CPU (batched or single-request); < 5 ms if you use ONNX + a distilled model + fp16.

**Cost.** Cheap. One trained model amortized across all queries; no per-query LLM tokens.

**Reliability.** Depends on the label quality. For a well-labeled 5-class problem with ~1000 examples per class, expect 90–96% top-1 accuracy. That means 4–10% of queries route wrong; the impact depends on the fallback behavior (see "Fallbacks" below).

**When it wins.**

- Stable, well-defined intent taxonomy that changes slowly.
- Enough labeled data to train a supervised model (or an inexpensive labeling loop to bootstrap one).
- Interactive latency budgets that cannot afford an LLM in the routing path.

**When it loses.**

- New intents appear frequently and the retraining loop is slow (weekly retraining or worse).
- The intent surface is ambiguous — a single query genuinely belongs to two intents.
- The labeling budget is thin (< 200 labeled examples per class); the classifier will overfit.

### 2. Embedding-nearest-cluster (few-ms, no LLM)

Cluster the corpus by embedding (or maintain per-corpus centroids); embed the query; route to the nearest cluster's retriever.

**Mechanism.** Compute embeddings for each destination corpus's documents (or a stratified sample). Cluster into *N* clusters (usually one per intended route) with k-means or a simpler nearest-centroid setup. At query time: embed the query, compute cosine distance to each cluster centroid, route to the closest cluster's retriever.

For fixed-taxonomy routing, you don't even need clustering — hand-author a small set of "prototype" utterances per route, embed them, and route on nearest-prototype. This is the [semantic-router library](https://github.com/aurelio-labs/semantic-router)'s core pattern.

**Latency.** One query embedding call (batchable, cacheable if the query repeats) + N cosine comparisons. Well under 20 ms for N < 100 routes when the embedding is already cached in the retrieval pipeline (you have to embed the query for the dense arm anyway).

**Cost.** Only the embedding-model inference cost, which you were already paying for the dense retrieval arm.

**Reliability.** Depends on how separated the routes are in embedding space. For distinct-domain routes (support, catalog, code, KB), the routes are usually well-separated and top-1 accuracy is high (95%+). For overlapping-domain routes ("how do I do X?" vs. "what is X?" both matching the docs route), the accuracy drops.

**When it wins.**

- You already run dense embeddings; adding routing is nearly free.
- The routes are semantically distinct.
- You want to add new routes by adding prototype utterances rather than retraining a classifier.

**When it loses.**

- Routes overlap semantically in the embedding space (e.g., "how to file a return" could route to the returns-policy corpus or the customer-service-scripts corpus).
- The embedder is not tuned to your domain and produces noisy neighborhoods.
- You need calibrated confidences (embedding-cluster distances are not probabilities; you can't reliably say "this route was chosen with 92% confidence").

### 3. LLM router (100–1500 ms)

Prompt an LLM to pick the route. The most flexible and the most expensive.

**Mechanism.** A structured prompt: "Given the following routes, which one best fits the query?" with per-route descriptions. The LLM outputs a route name. In modern setups this is often a **tool-use / function-calling** shape: define one function per route; let the model call the appropriate function with the query as an argument. Frameworks: [LangChain routers](https://python.langchain.com/docs/how_to/routing/), OpenAI [function calling](https://platform.openai.com/docs/guides/function-calling), Anthropic's [tool use](https://docs.claude.com/en/docs/build-with-claude/tool-use).

**Prompt shape:**

```
You are routing a user query to the best-fit retrieval system. Choose exactly one
of the following routes and output only the route name.

Routes:
- SUPPORT_ARTICLES: how-to and troubleshooting for the Acme product line.
- PRODUCT_CATALOG: SKU, pricing, availability, spec lookups.
- CODE_REFERENCE: API endpoints, function signatures, code samples.
- CONVERSATIONAL: greetings, capability questions, generic requests.

Query: {query}
Route:
```

**Latency.** LLM call latency. 100–300 ms for a small hosted model (Claude Haiku, GPT-4o mini) with structured output; 500–1500 ms for a mid-size model. Non-interactive stacks can absorb this; interactive stacks may not.

**Cost.** One LLM call per query. If your model costs $0.25/million input tokens and the router prompt is ~200 tokens plus the route names, that is ~$0.00005 per query — cheap in absolute terms, painful at 10 M queries/day.

**Reliability.** Very high on clear-cut queries (near 100%); reasonable on ambiguous queries. LLMs bring judgment that classifiers and centroids lack — they can read "reset admin password" as SUPPORT even if the query has never appeared in training data. Two failure modes to watch:

- **Hallucinated route names.** The LLM invents a route that doesn't exist. Structured output (function calling / JSON schema) with strict route enums prevents this.
- **Overconfident routing on borderline queries.** The LLM picks decisively when it should split or refuse. Address by requiring a `confidence` field in the output and falling back on low confidence.

**When it wins.**

- Complex or overlapping intent surfaces where classifier accuracy plateaus.
- Fast-changing route taxonomy (adding a route is a prompt change, not a retraining cycle).
- Multi-route queries (a query genuinely belongs to two routes; the LLM can pick both).

**When it loses.**

- Latency-critical interactive stacks. 300+ ms in the routing path is often too much.
- High QPS stacks where the cost of one LLM call per query adds up.
- Stacks where the routing decision must be deterministic and audit-trailed (LLM routing has some stochasticity even at temperature 0; classifier and centroid routing are deterministic).

## The one-page decision table

| Approach | Latency budget | LLM tokens per query | Best for | Watch out for |
|---|---|---|---|---|
| **Rule-based (regex, keyword)** | < 1 ms | 0 | The 20% of queries with clear signal (URLs, product codes, "help", "cancel"). | Missing intents you didn't anticipate. |
| **Intent classifier (small transformer)** | 2–15 ms | 0 | Stable taxonomy, ≥ 500 labels/class, interactive stacks. | Retraining cadence, drift when new intents appear. |
| **Embedding-nearest-cluster** | Reuses query embedding (marginal cost 0) + few ms | 0 | Distinct-domain routes, dense pipeline already in place. | Overlapping routes, uncalibrated confidence. |
| **LLM router** | 100–1500 ms | 100–500 | Complex / overlapping / fast-changing routes; batch stacks. | Cost at scale, non-deterministic outputs. |
| **Hybrid (rule → classifier → LLM fallback)** | Varies | 0–500 (rare) | The "cascade" pattern below. | Complexity budget of the routing stack itself. |

## The cascade pattern: rule → classifier → LLM

The pattern that works in production is almost always a cascade rather than a single mechanism:

```
Query
  │
  ▼
Rule-based fast path       ── clear match ──►  Route directly, 0 ms + 0 tokens.
  │
  │ (no match)
  ▼
Classifier / centroid      ── high confidence ──►  Route on prediction, ~5 ms.
  │
  │ (low confidence)
  ▼
LLM router                 ── expensive but robust ──►  Route on LLM output, ~500 ms.
```

The intent: **cheap first, expensive last**. The rule-based layer catches the trivially-routable queries (URLs, product codes, "hi"). The classifier catches most of the rest. The LLM router is the fallback for genuinely ambiguous queries. On a real stack, this often leaves 5–15% of queries falling through to the LLM router, which keeps the p95 latency in the classifier band while catching the tail with LLM judgment.

**Watch the confidence-threshold tuning.** The classifier's "high confidence" threshold determines what falls through to the LLM. If you set it too high, the LLM fires on most queries (cost explodes); too low, the classifier's mistakes go uncorrected. Tune on a held-out labeled set; recalibrate quarterly.

## Fallbacks: what happens when routing is wrong

Every routing system routes some queries incorrectly. The graceful fallback options:

- **Route-and-check.** Route to the predicted retriever, run retrieval, check the top-1 score / recall proxy. If the top-1 score is below a per-route threshold (indicating the retriever isn't finding anything), fall back to a broader retriever (a global retriever over all corpora, or a re-routing with the LLM router).
- **Route-and-parallel.** Route to the predicted retriever, *also* run a small-cost global retriever in parallel. Compare the top-1 scores; take the winner. Roughly doubles retrieval cost; only reach for it on high-stakes queries where the routing risk outweighs the cost.
- **Confidence-based dual routing.** If routing confidence is below threshold, run *both* candidate routes and merge results with RRF. Adds cost linearly in candidate count; useful for the "genuinely two routes" case.
- **Escalation to a human / a canned response.** The classic fallback of "I couldn't find that; here's how to contact support" — appropriate when the routing failure is total and any retrieval result would be worse than a graceful null.

The "hard fail" of returning empty results is almost always worse than a hedged fallback. Design the fallback path before you ship the primary router.

## Semantic routing beyond corpus selection

Corpus selection is the classic routing use case, but routing shows up several other places in a hybrid retrieval stack:

- **Retriever choice within a single corpus.** A single support corpus might have queries that route to BM25-only ("Q-77841 error") and queries that route to dense-heavy ("my printer is making weird noises"). Chapter 3's per-arm weights are one way to express this; explicit per-query routing is another.
- **Prompt template choice.** Different query classes need different generation prompts. "How-to" queries want step-by-step; "what-is" queries want definition-first; "why" queries want causal explanation. Route the query to a prompt template.
- **Rewriting strategy choice.** From chapter 4: HyDE fits some queries, multi-query fanout fits others, no rewriting fits still others. Route the rewriting decision itself.
- **Reranker choice** (mod-107 topic, briefly worth naming here). A cross-encoder is expensive; an LLM-as-judge is expensive-er. Route only the queries where the extra cost is likely to matter.

Every one of these is a place where a `switch` statement between mechanisms based on the query class saves cost and improves quality — if you have a routing signal to switch on.

## The latency-vs-reliability budget

The tradeoff to internalize:

- **Every ms in the routing path is a ms off the retrieval + reranking + generation budget.** If your p95 target is 500 ms and routing eats 300 ms, you have 200 ms for everything else. That may not be enough.
- **Every wrong route degrades the retrieval that follows.** Wrong-route retrievals return junk; even a perfect fusion can't recover from junk candidates.
- **The "cheapest correct route" is rarely a single mechanism.** Cascades win because they let you spend cheap most of the time and expensive-but-reliable at the tail.

The rule of thumb: **budget 5–10% of your p95 latency to routing on interactive stacks; up to 30% on batch stacks.** Any more, and routing has become the bottleneck rather than the enabler.

## The three anti-patterns to avoid

1. **LLM-routing every query when a classifier would do.** The LLM router is a fine tool for the ambiguous 10–15%; running it on the trivial 85% multiplies your cost and latency by orders of magnitude for zero quality gain.
2. **Routing without a fallback.** Every router is wrong sometimes. If your architecture assumes the router is always right, a routing failure becomes a full retrieval failure. Build the fallback (route-and-check or route-and-parallel) into the primary path.
3. **Adding routes without measuring the routing accuracy.** Each new route makes the classifier's job harder (more classes, more overlap). If you add a fifth route and top-1 classifier accuracy drops from 94% to 88%, you have made every retrieval 6 percentage points worse before the retriever even runs. Measure and defend.

## Two things to try before you ship a router

1. **The one-corpus baseline.** Before you build routing, ask whether you can merge all your corpora into one retriever with per-doc `source` metadata. If the merged corpus doesn't hurt recall (you evaluated), you save the routing complexity. This is often the right first move on stacks with 2–3 corpora.
2. **The static route table.** Before you build a classifier or an LLM router, ask whether a handful of rule-based routes (regex on the URL, keyword match, structured-query prefix like `sku:`) catches most of the traffic. The routing surface is often shorter than you'd guess; the classifier work is only justified once the rules stop catching enough.

## What is NOT in this chapter (and where it goes)

- **Filter-heavy composition (structured predicates + semantic query)** — chapter 6. Different problem from routing; the filters are inside the retrieval call, not routing which retrieval call runs.
- **The retriever interface contract the peer track consumes** — chapter 7.
- **Neural rerankers (cross-encoders, ColBERT, LLM-as-judge)** — mod-107.
- **Per-route eval methodology and statistical significance** — mod-109.
- **Training the classifier (data collection, labeling loops, active learning)** — not in this track at level 30; touched by mod-109's judgment-set methodology and by fine-tuning-engineer track for supervised training craft.

## What you should be able to do after this chapter

- Name the three mechanisms (intent classifier, embedding-nearest-cluster, LLM router) and give a latency band and a cost profile for each.
- Design a cascade router (rule → classifier → LLM fallback) with a confidence-threshold pattern and defend the thresholds on an eval set.
- Recognise when routing is not the right lever (small corpora count, well-separated queries that a merged corpus handles).
- Sketch a fallback strategy for a routing failure: route-and-check, route-and-parallel, or dual-route-RRF.
- Budget routing latency as a fraction of the end-to-end p95 target.

## Further reading

- Aurelio Labs: [semantic-router](https://github.com/aurelio-labs/semantic-router) — the reference implementation of embedding-nearest-cluster routing for LLM stacks.
- LangChain: [*How to route between sub-chains*](https://python.langchain.com/docs/how_to/routing/) — the LLM-router shape as it appears in LangChain.
- Anthropic: [*Tool use with Claude*](https://docs.claude.com/en/docs/build-with-claude/tool-use) — the function-calling shape for LLM routing where each route is a tool.
- OpenAI: [*Function calling*](https://platform.openai.com/docs/guides/function-calling) — same, for OpenAI's API.
- Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108) — the standard reference for the small-classifier model class.
- Reimers, N., and Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) — the standard reference for the embedding backbone used in nearest-centroid routing.
- Bekker, J., and Davis, J. (2020). *Learning from positive and unlabeled data: a survey.* Machine Learning 109. [arXiv:1811.04820](https://arxiv.org/abs/1811.04820) — for the reader building routers with sparse labels; the PU-learning frame is useful for one-vs-rest classifier training with only positives per class.
