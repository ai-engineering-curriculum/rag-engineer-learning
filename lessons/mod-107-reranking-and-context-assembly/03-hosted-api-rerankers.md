# Chapter 3 — Hosted API Rerankers

## Why this chapter exists

Chapter 2 was the self-hosted cross-encoder path — download weights, run inference on your own GPU, own the model, tune it. This chapter is the alternative: **hosted rerank APIs** where a vendor runs the model and you send it (query, candidates) over HTTPS.

The three vendors this chapter covers — [Cohere Rerank](https://docs.cohere.com/docs/rerank-overview), [Voyage rerank-2](https://docs.voyageai.com/docs/reranker), [Jina Reranker](https://jina.ai/reranker/) — offer near-identical shapes but different quality / latency / pricing points. Which one you pick (or whether you pick a hosted API at all) is a joint decision about your ops appetite, your inference budget, and the quality ceiling of what you can achieve with off-the-shelf or fine-tuned self-hosted models.

## The trade-off in one line

You are trading **capex + ops (GPUs, model updates, on-call for the rerank service)** for **opex per rerank call**. The tipping point is a function of your query volume, your acceptable latency floor, and how much ops burden the self-hosted path adds relative to your team's other commitments.

Very approximate rule of thumb the industry uses (verify with your actual pricing and your actual GPU utilisation):

- **Below ~1–5 QPS sustained** — hosted is almost always cheaper. A small GPU sitting idle 95% of the time is more expensive than a per-call API bill.
- **Above ~50–100 QPS sustained** — self-hosted is almost always cheaper *if* you have the ops capability. A well-utilised GPU running a MiniLM cross-encoder at 100 QPS costs cents per hour amortised; hosted APIs at the same volume are dollars per hour.
- **Between those** — it depends on price/spec, engineering time cost, and how much you value latency floor and predictable capacity.

Do not skip the arithmetic. Compute (queries/day × candidates/query × dollars/candidate) for the hosted case and (GPU $/hour × 24 × 30) for the self-hosted case using your actual traffic shape. Vendors update pricing regularly; always confirm on the current pricing page.

## The three vendors, one interface

All three vendors expose a near-identical HTTP interface. The shape:

```
POST /rerank
{
  "model": "<model_id>",
  "query": "<query text>",
  "documents": ["<candidate 1>", "<candidate 2>", ..., "<candidate N>"],
  "top_n": 10
}

→ {
    "results": [
      { "index": 3, "relevance_score": 0.94, ... },
      { "index": 0, "relevance_score": 0.87, ... },
      ...
    ]
  }
```

You send the query and the N candidates verbatim, get back a re-ordered list of indices with scores. All three provide official Python SDKs, all three support `top_n` server-side truncation (you get back K, not N), and all three cap the number of documents per call (typically 1000).

The three specifics that differ:

### Cohere Rerank

- **Documentation:** [docs.cohere.com/docs/rerank-overview](https://docs.cohere.com/docs/rerank-overview); API reference at [docs.cohere.com/reference/rerank](https://docs.cohere.com/reference/rerank).
- **Current model families (verify on the docs page):** `rerank-v3.5`, `rerank-english-v3.0`, `rerank-multilingual-v3.0`, plus a `light` variant tuned for latency. Cohere refreshes model versions regularly; the docs page names the current default.
- **Notable capabilities:** long-context support (chunks up to several thousand tokens are handled without a hard truncate; the vendor documents the model's effective max length), and a `rank_fields` parameter for scoring structured JSON documents by specific fields — useful when your "chunks" have title / body / metadata separation.
- **Pricing model:** per rerank query, with a per-query cost that depends on the number of documents. Confirm on the [pricing page](https://cohere.com/pricing).
- **Python SDK:**

  ```python
  import cohere

  co = cohere.ClientV2(api_key="...")
  response = co.rerank(
      model="rerank-v3.5",
      query="How do I cancel my subscription?",
      documents=candidates,        # list[str] or list[dict] with rank_fields
      top_n=10,
  )
  # response.results is list of RerankResult with .index and .relevance_score
  ```

### Voyage AI (`rerank-2`, `rerank-2-lite`)

- **Documentation:** [docs.voyageai.com/docs/reranker](https://docs.voyageai.com/docs/reranker); API reference at [docs.voyageai.com/reference/reranker-api](https://docs.voyageai.com/reference/reranker-api).
- **Current model families:** `rerank-2` (higher quality), `rerank-2-lite` (lower latency and cost). Voyage documents context lengths per model on the [models page](https://docs.voyageai.com/docs/reranker).
- **Notable capabilities:** publishes numbers on domain-specific benchmarks (finance, law, code) alongside the general benchmarks; useful if you are shipping into one of those domains.
- **Python SDK:**

  ```python
  import voyageai

  vo = voyageai.Client()  # reads VOYAGE_API_KEY
  response = vo.rerank(
      query="How do I cancel my subscription?",
      documents=candidates,
      model="rerank-2",
      top_k=10,
  )
  # response.results is list of RerankingResult with .index and .relevance_score
  ```

### Jina Reranker

- **Documentation:** [jina.ai/reranker](https://jina.ai/reranker/); API reference at [api.jina.ai/redoc](https://api.jina.ai/redoc).
- **Current model families:** `jina-reranker-v2-base-multilingual` (hosted variant of the open-weight model in chapter 2), `jina-colbert-v2` (a hosted late-interaction reranker). The hosted-and-open-weight parity is Jina's differentiator — you can prototype against the hosted API and self-host the same weights when you scale.
- **Notable capabilities:** multilingual coverage on the smaller-parameter model, and the option to bring your own OSS model to Jina's inference infrastructure via their [Jina AI Cloud](https://cloud.jina.ai/) offering.
- **Pricing model:** token-based ("Rerank API tokens"), which lets you predict cost per query more reliably than per-call pricing.
- **Python SDK / HTTP:** Jina's docs show both `requests`-based HTTP calls and their `jina` client. See [jina.ai/reranker](https://jina.ai/reranker/) for the current shape.

## Latency floors and the network hop

The single most important operational fact about a hosted API is that it has a **network-hop latency floor** you cannot squeeze past by throwing money at it.

Rough anchors (measure with your own client and region; these move):

- **Same-region API call:** typically 30–150 ms round-trip for the network + TLS + vendor's own inference time. The vendor's inference on a well-tuned model is often faster than what you'd get self-hosting a similar-size model, but the network hop adds a floor you don't have with a local GPU.
- **Cross-region:** add 50–200 ms depending on the geography. If your app is in `us-east-1` and the vendor is in `eu-west-1`, that's an eye-watering p95 latency.
- **Cold-start / rate-limited / retry storms:** p99 for hosted APIs is much wider than p95 — a 30 ms p50 can turn into a 500 ms p99 on a bad hour. The vendor's status page tells you what they commit to; your monitoring should tell you what actually happened.

Two consequences:

1. **Batch every rerank call.** Some vendors allow a single API call to carry N candidates; use that (never send N single-document calls, which multiplies network overhead by N).
2. **Deploy your app in a region the vendor serves.** All three vendors publish their supported regions; align your app's region with theirs. If your regulatory constraints prevent cross-region hops (data residency, GDPR), that alone can force self-hosting.

## The `top_n` parameter — a subtle cost knob

The API accepts `documents=[N candidates]` and `top_n=K` — you send N, you get K. Some vendors charge based on N (how many candidates you *sent*), some charge based on K (how many you *received*), and some based on total input tokens.

The distinction matters because a common bug is sending N=1000 (because "why not, we might get better results") when the vendor charges per document reranked. Every candidate past the point of diminishing returns (chapter 5 has the diminishing-returns curve) is money spent for no quality gain.

Before you go to production, always:

1. Read the current pricing page for the vendor you picked.
2. Measure the diminishing-returns curve on your own data (exercise 02 walks through this).
3. Set your N to the elbow of that curve, not to the vendor's maximum.

## Compliance, data residency, PII

The retrieved candidates *contain your data* — user queries, chunks of your documentation, sometimes PII from the corpus itself. When you send them to a rerank API you are exporting that data to a third-party processor.

The things to confirm before deployment:

- **Data-processing agreement (DPA) / GDPR posture.** All three vendors publish a DPA and privacy documentation ([Cohere trust](https://cohere.com/trust), [Voyage AI privacy](https://www.voyageai.com/privacy), [Jina AI legal](https://jina.ai/legal/)). Read them; get your legal team to sign off if you handle regulated data.
- **Training data retention.** Vendor default is usually "we do not train on your data" but the details matter. Check the current terms.
- **Region control.** If you need EU-only processing, the vendor must offer an EU endpoint. Cohere and others do; confirm on their docs.
- **Enterprise deployment options.** All three offer VPC / private deployment for enterprise customers. If self-hosting is off the table but data-residency requirements are strict, this is the path to ask the vendor about.

If any of this fails your compliance bar, you are back to self-hosted (chapter 2) or an on-prem enterprise deployment of the vendor's stack.

## The five reasons to pick a hosted rerank API

1. **You do not have the ops capability to run GPUs reliably.** A rerank service that ships broken updates and takes down retrieval quality for a week is worse than one you pay for. If your team has never operated a GPU inference service in production, start hosted.
2. **Your query volume is bursty or low.** A sustained 0.5 QPS with 100× peaks is a nightmare for self-hosting (you provision for the peak, pay for the trough) and a small monthly bill for a hosted API.
3. **You want quality tracked over time by someone else.** Vendors update their models regularly; you re-benchmark and, if the new version wins on your eval, switch over. That is much cheaper than owning the model-improvement roadmap yourself.
4. **You want fewer moving parts in the first version.** Mod-101 argued that a level-30 rag-engineer builds an *end-to-end retriever* first and iterates. A hosted rerank API is a fine choice for that end-to-end version; you can move to self-hosted later once you know it wins.
5. **The vendor has a domain-specific model you can't reproduce.** Voyage's finance / legal / code rerankers, or Cohere's multilingual variants, may be a genuine head-start on your fine-tuning cost.

## The four reasons to avoid a hosted rerank API

1. **Data residency or PII rules make it impossible.** Some corpora cannot leave your VPC. Hosted API is out; enterprise on-prem deployment is the vendor option; self-hosted is the escape hatch.
2. **You have sustained high QPS with a strict latency floor.** At scale, self-hosted usually wins on both cost and p95. The tipping point depends on your specific numbers.
3. **You have a truly domain-specific relevance definition.** No off-the-shelf model — hosted or open-weight — will match a fine-tune on your labeled data. If you have the data, self-host and fine-tune.
4. **You want reproducibility across model versions.** Hosted models can deprecate. If your eval must be reproducible on a fixed model for years (research, regulated products), self-host.

## Benchmarking hosted vs. self-hosted on your own eval

Every rerank paper reports numbers on public benchmarks (MS MARCO, BEIR, TREC-DL). Public benchmarks are useful for triaging what to try; they are not a substitute for evaluating on *your* corpus and *your* queries. The delta between a public-benchmark leader and the actual leader on your data is routinely in the double-digit-percent range on nDCG@10.

The minimum viable benchmark:

1. Take 50–200 hand-labeled (query, relevant-doc-ids) pairs on your corpus.
2. Run each candidate (Cohere, Voyage, Jina, BGE, MiniLM, no-rerank baseline) at N=100 → K=10 over the same first-stage list.
3. Compute nDCG@10 (or recall@10 and MRR@10 — mod-109 covers the metric choice) for each.
4. Also record p50/p95/p99 latency for each and cost-per-1000-queries.

That triple (quality, latency, cost) is what actually determines the right pick. Do not skip the latency and cost numbers — a 2-point nDCG@10 improvement at 3× the cost and 2× the latency is often a bad trade.

Exercise 01 in this module walks through this benchmark end-to-end.

## What you should be able to do after this chapter

- Compare the three hosted rerank APIs (Cohere Rerank, Voyage rerank-2, Jina Reranker) on interface shape, pricing model, and stated capability differences without reciting a leaderboard.
- Reason about the hosted-vs-self-hosted crossover point using your own traffic shape and pricing, not a rule of thumb.
- Name the network-hop latency floor and design your rerank call (batching, region alignment, `top_n` sizing) to keep it acceptable.
- Enumerate the compliance concerns (DPA, data residency, training retention) and know which vendors offer VPC / on-prem deployment.
- Write a minimum viable rerank benchmark (quality + latency + cost) on your own data before committing to a vendor.

## Further reading

- [Cohere Rerank documentation](https://docs.cohere.com/docs/rerank-overview) — Cohere's own overview + API reference.
- [Voyage AI Reranker documentation](https://docs.voyageai.com/docs/reranker) — Voyage's model list, docs, and pricing.
- [Jina Reranker documentation](https://jina.ai/reranker/) — Jina's hosted + open-weight offerings and how they relate.
- Reimers, N. (2023). *Rerank benchmarks on BEIR* — sentence-transformers documentation includes comparison numbers on standard benchmarks: [sbert.net/docs/pretrained-cross-encoders.html](https://www.sbert.net/docs/pretrained-cross-encoders.html).
- MS MARCO Passage Ranking [leaderboard](https://microsoft.github.io/msmarco/) — for calibrating public-benchmark expectations across model families.
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) — the reference benchmark for zero-shot cross-domain reranker performance.
