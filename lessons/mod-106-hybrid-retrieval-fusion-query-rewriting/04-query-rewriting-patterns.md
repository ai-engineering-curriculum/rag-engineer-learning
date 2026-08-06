# Chapter 4 — Query Rewriting: HyDE, Multi-Query Fanout, Step-Back, and Decomposition

## Why this chapter exists

Chapters 2 and 3 tuned the *output* stage — how ranked lists combine. This chapter tunes the *input* stage — the query itself, before it reaches any retriever. Every rewriting pattern here is a way to trade LLM tokens for retrieval recall: you spend an extra LLM call (sometimes several) to produce a better query, then feed the rewritten query (or several rewritten queries) into the same hybrid stack chapters 2 and 3 built.

The four patterns that come up in enterprise RAG stacks:

1. **HyDE (Hypothetical Document Embeddings)** — the LLM writes a hypothetical answer; you embed the *answer* and dense-retrieve against it.
2. **Multi-query fanout** — the LLM generates *k* paraphrases of the query; each retrieves independently; you fuse the results.
3. **Step-back prompting** — the LLM abstracts the query one level up ("what class of thing is being asked about?"), retrieves at that level, then answers.
4. **Query decomposition** — the LLM breaks a multi-hop question into sub-questions; each sub-question retrieves independently.

Each has an origin paper, a mechanism, a specific class of failure it fixes, and an eval-cost profile you need to price. This chapter walks all four. Chapter 5 turns to routing (deciding whether to run any of these at all).

## The lever this chapter pulls, and its costs

Query rewriting fixes **query-quality failures**: the query is too short (product-search shorthand), uses different terminology than the corpus (user says "cancel subscription," corpus says "revoke recurring billing agreement"), asks a multi-part question that no single passage answers, or is so specific that it misses the semantic neighborhood a broader query would have hit.

The costs, in the order they matter:

- **Extra LLM calls per query.** HyDE, step-back, and decomposition each cost at least one LLM call before retrieval. Multi-query fanout costs one call to generate *k* rewrites, plus *k* retrievals. Latency and token spend both scale with rewrites.
- **Latency.** An LLM call before retrieval is a p95-latency killer if the model is large and hosted. For interactive stacks (< 200 ms p95), you almost cannot afford a full-size LLM in the rewriting path — use a small distilled model, an in-house cheap model, or accept the batched non-interactive path.
- **Cost-per-query.** Every rewriting pattern multiplies token spend. In a stack that already runs a reranker (mod-107) and a final generation call, rewriting can double or triple the LLM cost.
- **Quality regressions on easy queries.** A wall-of-jargon HyDE answer on a simple, well-formed query can *reduce* recall — the hypothetical passage may drift away from what the corpus contains. Always eval the "off" condition too.

Every rewriting pattern below deserves an eval-set answer to: **does this pattern add recall on my corpus, and does the recall gain justify the latency and cost?** Mod-109 covers the methodology. This chapter names the levers.

## Pattern 1: HyDE (Hypothetical Document Embeddings)

**Origin:** Gao, Ma, Lin, and Callan, 2022. *Precise Zero-Shot Dense Retrieval without Relevance Labels.* [arXiv:2212.10496](https://arxiv.org/abs/2212.10496).

**Mechanism:** Given a query `q`, prompt an LLM to write a hypothetical passage that *would* answer `q`. Embed the hypothetical passage. Dense-retrieve against the hypothetical embedding rather than the query embedding.

**Intuition:** A well-trained dense embedder maps queries and passages into overlapping but distinct regions of the vector space. A query like "what is the capital of France?" and a passage like "Paris, the capital of France, has been the political center since..." are close in cosine but not identical. HyDE trades query-embedding for hypothetical-answer-embedding, moving the retrieval anchor from query-space to passage-space. When queries are much shorter or worded more differently than passages, that shift can be worth a real recall bump.

**Prompt shape:**

```
Please write a passage that answers the following question. The passage should be
written as if it were extracted from a factual reference document (encyclopedia,
technical documentation, textbook). Do not add commentary; write only the passage.

Question: {query}
Passage:
```

**Concrete example.** Query: `how do i reset the admin password on the acme dashboard`.

- **Vanilla dense retrieval** embeds the query, retrieves the top-k. Recall depends on how the embedder handles "how do I" vs. formal documentation phrasing.
- **HyDE dense retrieval** first asks the LLM to write a hypothetical passage: *"To reset the administrator password on the Acme Dashboard, navigate to Settings → Account → Security, then click 'Reset password.' You will receive a confirmation email at the address on file..."* — then embeds that hypothetical passage and retrieves against it. The retrieval anchor now lives in "passage-space" (formal instructions, third-person imperative), which is closer to what the corpus actually contains than the "how do I" phrasing.

**When it wins.** HyDE tends to help when:

- Queries are very short or informal ("reset admin pw?" or a keyword-only search bar entry).
- The corpus is stylistically formal (docs, manuals, encyclopedic content) and queries are stylistically informal.
- The embedding model was trained more on passage-passage similarity than on query-passage similarity.
- The corpus is zero-shot for the embedder (the model has not seen this domain in fine-tuning). This is HyDE's origin claim: zero-shot dense retrieval without relevance labels.

**When it loses.**

- Queries that are already long, formal, and passage-shaped. The LLM hypothetical drifts away from the query without recovering recall.
- Queries with an entity that must appear verbatim in the retrieved passage (product code `Q-77841`, error message `E1234`). The LLM often hallucinates around or drops such tokens; the hypothetical loses the anchor.
- Queries where the corpus contains multiple valid answers with different framings; the LLM will pick one framing and miss the others.

**Eval cost.** One LLM call per query, tokens roughly `|prompt| + |hypothetical passage|` (~200–400 output tokens for a good hypothetical). Latency: the LLM call's latency, added to the retrieval latency. For a hosted mid-size model, expect 400–1500 ms; for a small on-device model, 100–300 ms.

**Reference implementation:**

```python
def hyde_retrieve(query: str, llm, dense_retriever, top_n: int = 100) -> list[str]:
    hypothetical = llm.generate(
        prompt=HYDE_PROMPT.format(query=query),
        max_tokens=300,
        temperature=0.3,
    )
    return dense_retriever.retrieve(hypothetical, top_n=top_n)
```

**Combining with lexical.** Do *not* HyDE the query and then hand the hypothetical to a BM25 retriever — the hypothetical is a passage, not a keyword-shape query, and its bag-of-words will match the wrong things. The pattern is: **HyDE for the dense arm only; keep the original query for the lexical arm; RRF the two outputs.**

## Pattern 2: Multi-query fanout

**Origin:** No single origin paper; the technique is folk-lore in the LangChain / LlamaIndex era and is closely related to pseudo-relevance feedback (PRF) in the classical IR literature ([Rocchio 1971](https://en.wikipedia.org/wiki/Relevance_feedback)). Its LLM incarnation is exposed as [LangChain's MultiQueryRetriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/) and LlamaIndex's [QueryTransformationEngine](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/advanced/query_transformations/).

**Mechanism:** Given a query `q`, prompt an LLM to produce *k* paraphrases (typically 3–5). Retrieve independently for each paraphrase. Fuse the *k+1* result lists (original + rewrites) with RRF.

**Intuition:** Any one phrasing of a query is a lucky-dip on the embedder's neighborhood. Different paraphrases hit different neighborhoods; fusing them covers more of the semantic surface than a single query does. The union recall can go up by real amounts (3–10 percentage points is typical on heterogeneous corpora).

**Prompt shape:**

```
You are helping a search system. Rewrite the following query in {k} distinct ways
that preserve the original intent but vary the vocabulary and phrasing. Number each
rewrite. Do not add explanation.

Original: {query}
Rewrites:
1.
```

**Concrete example.** Query: `refund policy for damaged goods`.

Possible LLM rewrites:

1. `how to get money back for broken items`
2. `what is the return policy for defective merchandise`
3. `damaged product refund process`

Each hits a slightly different corner of the corpus. Fusion collects the union.

**When it wins.**

- Queries are short and the semantic surface of the answer is wide (many phrasings of the answer exist in the corpus).
- The corpus vocabulary drifts across time or across sections (support articles from different eras, or documentation from different products under the same brand).
- The embedder is decent but not domain-tuned.

**When it loses.**

- Queries with rare-term anchors (`Q-77841`, `SKU-4402`). The LLM paraphrases will drop the anchor and retrieve neighborhood-wide junk.
- The corpus is small and tightly written; the rewrites all retrieve the same top-k and the fusion adds no new candidates.
- Latency-critical stacks. *k* rewrites means *k+1* retrieval calls plus the LLM generation call.

**Eval cost.** One LLM call per query to generate the rewrites (tokens ≈ `|prompt| + k · |query|` — typically 100–300 tokens output for `k = 4`). Plus *k+1* retrieval calls, which run in parallel but each adds a slot in the retrieval budget. Latency: LLM call latency + max(retrieval latency across rewrites). Cost scales linearly in `k`.

**How many rewrites?** Empirical band: `k = 3–5` is the sweet spot in most literature and framework defaults. Past `k = 5`, marginal recall gain drops fast and cost keeps growing. Do not fanout to 20; it will not help and it will hurt your token bill.

**Reference implementation:**

```python
def multi_query_retrieve(
    query: str, llm, retriever, k: int = 4, rrf_k: int = 60, top_n: int = 100
) -> list[str]:
    rewrites = llm.generate(MULTI_QUERY_PROMPT.format(query=query, k=k))
    queries = [query] + parse_numbered_list(rewrites)
    ranked_lists = [retriever.retrieve(q, top_n=top_n) for q in queries]
    fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k)
    return [doc_id for doc_id, _ in fused[:top_n]]
```

For hybrid stacks: run *each rewrite* through the full hybrid stack (dense + BM25 + fusion), then RRF the per-rewrite outputs. That is `k+1` full hybrid retrievals. Expensive; measure whether the recall lift justifies it before you ship.

## Pattern 3: Step-back prompting

**Origin:** Zheng, Mishra, Chen, Cheng, Chi, Le, and Zhou, 2023. *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models.* [arXiv:2310.06117](https://arxiv.org/abs/2310.06117).

**Mechanism:** Given a specific query, prompt the LLM to *abstract* it up one level — from a specific instance to the general concept, from a specific event to the broader context. Retrieve at the abstracted level (or at both levels). Use the retrieved abstract-level context to help the LLM answer the original specific question.

**Intuition:** A very specific query ("what year did Estella move to London in David Copperfield?") is often better answered by first retrieving context on the broader concept ("David Copperfield character biographies," "London-based plot events in Dickens") and *then* narrowing down. The step-back retrieval anchor is more likely to hit a well-populated region of the corpus than the hyper-specific one.

**Prompt shape:**

```
Given the specific question below, formulate a more general "step-back" question
that asks about the broader concept or category the specific question is about.
The step-back question should be answerable by many passages, not a single hyper-
specific one.

Original question: {query}
Step-back question:
```

**Concrete example.** Query: `is the acme model 4402 covered under the 2024 extended warranty program`.

Step-back: `what does the 2024 extended warranty program cover`.

Retrieve for the step-back; use the retrieved passages as context for the LLM to answer the original.

**When it wins.**

- Specific / multi-hop queries where the specific answer requires broader context first.
- Queries with rare specific tokens that make direct retrieval sparse.
- Multi-turn conversational retrieval where the abstracted question can be shared across turns.

**When it loses.**

- Queries already at the right level of specificity. Stepping back to too-general terrain retrieves generic content that dilutes rather than helps.
- Corpora where the specific answer *is* in a single passage and step-back retrieval brings in unrelated context that crowds out the good candidate.

**Eval cost.** One LLM call to generate the step-back question; one retrieval call at the step-back level (sometimes two if you retrieve at both levels). Comparable to HyDE in latency and cost.

## Pattern 4: Query decomposition (multi-hop)

**Origin:** Multi-hop QA literature has decomposed queries since HotpotQA (Yang et al. 2018, [arXiv:1809.09600](https://arxiv.org/abs/1809.09600)); the LLM-driven variant is described in *Measuring and Narrowing the Compositionality Gap in Language Models* (Press et al. 2022, [arXiv:2210.03350](https://arxiv.org/abs/2210.03350)) and popularized as "self-ask" and by frameworks like LangChain's [`MultiHopRetriever`](https://python.langchain.com/docs/how_to/multi_vector/) and LlamaIndex's [SubQuestionQueryEngine](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/).

**Mechanism:** Given a compound query, prompt the LLM to break it into sub-questions. Each sub-question retrieves independently. Aggregate the per-sub-question contexts into the LLM's answer prompt.

**Intuition:** "What is the population of the capital of the country that won the 2018 FIFA World Cup?" is three sub-questions: who won the 2018 World Cup (France), what is France's capital (Paris), what is Paris's population. No single passage in a general corpus answers all three; three independent retrievals each hit one sub-question. Compound queries decompose into a chain of factoid queries.

**Prompt shape:**

```
Decompose the following compound question into a numbered list of simple,
independent sub-questions. Each sub-question must be answerable from a single
passage. If the question is already simple, output "1. {original question}".

Compound question: {query}
Sub-questions:
1.
```

**Concrete example.** Query: `what was the largest funding round for a company whose CEO previously worked at Acme Corp`.

Sub-questions:

1. `who worked at Acme Corp before becoming a CEO`
2. `for each of those people, what company are they now CEO of`
3. `what was the largest funding round for each of those companies`

Each retrieves independently; the LLM composes the final answer from the aggregate context.

**When it wins.**

- True multi-hop questions where no single passage contains the answer.
- Corpora that contain the *facts* but not the *composed answer*.

**When it loses.**

- Single-hop questions the LLM mis-decomposes into unnecessary sub-questions (you have doubled retrieval cost for nothing).
- Compound questions whose sub-questions are dependent (the answer to sub-question 2 depends on the answer to sub-question 1). Decomposition requires sequential retrievals, not parallel; latency explodes.

**Eval cost.** One LLM call to decompose; one retrieval per sub-question; often a final LLM call to compose the answer from the aggregate context. For a compound question with 3 sub-questions, that is ~4 LLM calls + ~3 retrievals — 4–8× the vanilla retrieval cost.

**Dependent decompositions.** When sub-question 2's phrasing depends on the answer to sub-question 1, you cannot parallelize. The decomposition becomes a chain: retrieve for sub-1, LLM synthesizes the answer, LLM generates sub-2 using that answer, retrieve for sub-2, and so on. Latency and cost multiply through the chain. Handle sparingly.

## The eval-cost table

| Pattern | LLM calls per query | Retrievals per query | Typical extra latency | When to reach for it |
|---|---|---|---|---|
| Vanilla | 0 | 1 hybrid | 0 | Baseline. |
| HyDE | 1 | 1 hybrid (dense arm HyDE'd, lexical arm unchanged) | 200–1500 ms | Short queries, formal corpus, zero-shot embedder. |
| Multi-query fanout `k=4` | 1 | 5 hybrids (parallel) | 100–500 ms (LLM) + max retrieval | Short queries, wide-semantic-surface answers. |
| Step-back | 1 | 1–2 hybrids | 200–1500 ms | Overly-specific queries, sparse specific-token retrieval. |
| Decomposition (3-hop, parallel) | 1 + 1 (compose) | 3 hybrids (parallel) | 400–3000 ms | True multi-hop questions. |
| Decomposition (3-hop, sequential) | 1 per hop + 1 compose | 3 hybrids sequential | 1500–5000+ ms | Dependent multi-hop; expensive. |

Every row past "vanilla" is a bet: the pattern adds cost, so it must add recall or answer quality *on your corpus and your query mix*. Measure before you ship any of them.

## Small model vs. large model in the rewriter role

You do not need GPT-5 or Claude Opus to write a HyDE passage or generate paraphrases. The rewriting task is:

- Bounded output length (a couple hundred tokens).
- Fairly forgiving of quality (retrieval fixes the rest — the LLM does not have to be right, just *close enough* semantically).
- Highly parallel across queries (one rewriter serving many queries at low temperature).

A small distilled model (a 7–13B parameter model like Llama 3.1 8B Instruct, or a hosted small model like Claude Haiku or GPT-4o mini) is usually enough. On some corpora, a purpose-fine-tuned rewriter (train a small model on `(query, good-rewrite)` pairs) outperforms a large general-purpose model at a fraction of the latency. Do not default to your biggest model in the rewriter role; the cost multiplies fast.

## Two anti-patterns worth naming

1. **Chaining every rewriting pattern together.** "HyDE + step-back + multi-query fanout + decomposition on every query" — that is 4+ LLM calls and 10+ retrievals per user query. The recall gain of the fourth pattern on top of the first three is nearly zero; the cost is multiplicative. Ship one rewriting pattern, measure, add a second only if the eval shows it wins.
2. **Not measuring the "off" condition.** Rewriting patterns can *lose* recall on queries they don't fit. If you ship HyDE without measuring how it affects queries that don't benefit, you may quietly regress the median query while improving the tail. Report **per-query-class metrics** (short queries vs. long queries, formal vs. informal, single-hop vs. multi-hop) — not just averages.

## Where routing comes in

The natural response to "some queries benefit from HyDE and some don't" is: **route**. Run HyDE only when the query fits a heuristic (short, informal, no rare-entity anchor); run vanilla otherwise. That is chapter 5's topic.

## What is NOT in this chapter (and where it goes)

- **Semantic routing (intent classification, embedding clusters, LLM router)** — chapter 5.
- **Filter-heavy composition** — chapter 6.
- **The retriever interface contract** — chapter 7.
- **Small-model distillation for the rewriter role** — beyond mod-106; touched on in mod-108 (retrieval-specific prompting) and in fine-tuning-engineer track.
- **The full eval methodology (per-query-class metrics, paired significance, statistical power)** — mod-109.

## What you should be able to do after this chapter

- Explain HyDE, multi-query fanout, step-back, and decomposition in one paragraph each with a concrete example.
- Name the origin paper or definitive reference for each pattern.
- Price the eval cost (LLM calls + retrievals + latency) of each on a per-query basis.
- Match a rewriting pattern to the failure mode it fixes: HyDE for style/domain mismatch, multi-query for short queries, step-back for over-specificity, decomposition for multi-hop.
- Defend the choice of a small distilled model over a large frontier model in the rewriter role for interactive stacks.
- Recognise the "chain everything" and "no measurement of off condition" anti-patterns.

## Further reading

- Gao, L., Ma, X., Lin, J., and Callan, J. (2022). *Precise Zero-Shot Dense Retrieval without Relevance Labels.* ACL 2023. [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) — the HyDE paper.
- Zheng, C., Mishra, S., Chen, X., Cheng, H.-T., Chi, E. H., Le, Q. V., and Zhou, D. (2023). *Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models.* [arXiv:2310.06117](https://arxiv.org/abs/2310.06117) — the step-back paper.
- Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A., and Lewis, M. (2022). *Measuring and Narrowing the Compositionality Gap in Language Models.* [arXiv:2210.03350](https://arxiv.org/abs/2210.03350) — the self-ask decomposition paper.
- Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., and Manning, C. D. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP. [arXiv:1809.09600](https://arxiv.org/abs/1809.09600) — the classic multi-hop QA benchmark whose decomposition pattern LLMs now emulate.
- Rocchio, J. J. (1971). *Relevance Feedback in Information Retrieval.* Chapter 14 in Salton (ed.), *The SMART Retrieval System*. Prentice-Hall. Historical: pseudo-relevance feedback is the classical-IR ancestor of multi-query fanout; useful frame for the pattern's provenance.
- LangChain: [*MultiQueryRetriever*](https://python.langchain.com/docs/how_to/MultiQueryRetriever/) and [*Rewrite-Retrieve-Read*](https://python.langchain.com/docs/how_to/query_high_cardinality/) — the framework-side view of multi-query fanout.
- LlamaIndex: [*Query Transformations*](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/advanced/query_transformations/) — HyDE, step-back, decomposition all live under one API.
- Ma, X., Gong, Y., He, P., Zhao, H., and Duan, N. (2023). *Query Rewriting for Retrieval-Augmented Large Language Models.* EMNLP. [arXiv:2305.14283](https://arxiv.org/abs/2305.14283) — a systematic study of learned query rewriting; useful for the reader who wants the fine-tuned-rewriter alternative to prompt-engineered rewriting.
