# Exercise 03 — Hosted vs. Self-Hosted: Cost, Latency, and Quality-per-Dollar

**Estimated effort:** ~3 hours.

## Objective

Chapter 4 surveyed the hosted embedding APIs; chapter 5 surveyed the open self-hosted families; chapter 6 gave the four cost surfaces and the quality-per-dollar rule. This exercise puts those chapters on the same table: pick **one** hosted model and **two** self-hosted models, measure their **query-time latency**, **per-batch throughput**, and **per-million-token cost** on identical inputs, and rank them by the "quality floor first, cost second" rule from chapter 6. Quality here is a small nDCG@10 score against a shared eval set (SciFact or the tiny domain task from exercise-02) so the ranking is not purely a cost story.

By the end you will have the same shape of comparison table a mid-level architecture doc puts in front of the reviewer for embedding-choice sign-off. Exercise-04 uses this table as a template on your real corpus.

## Prerequisites

- Chapters 4, 5, 6 read once.
- Exercise-02 complete (you already have `mteb` installed, a SciFact-shaped harness, and know how to aggregate MTEB JSON).
- Python deps: `sentence-transformers`, `mteb`, `pandas`, `httpx` (for hosted-API latency measurement), plus the SDK for the hosted provider you pick — `openai`, `cohere`, `voyageai`, or `google-cloud-aiplatform`.
- One provider API key. If you cannot get one, **skip the hosted portion and run three self-hosted models** — the methodology is the same and the acceptance criteria list the fall-back explicitly.
- A machine with either a modest CUDA GPU (T4, A10G, RTX 3060+) or a modern CPU. A GPU makes throughput numbers realistic; a CPU is enough for correctness.

## Scope decisions before you write code

Pick three models spanning the axes chapter 6 said matter:

- **One hosted.** Recommended defaults: OpenAI `text-embedding-3-small` at `dimensions=512`, Cohere `embed-english-v3.0`, or Voyage `voyage-3`. Use whichever key you have.
- **One small self-hosted.** `BAAI/bge-small-en-v1.5` (384 dim, 33M params) or `intfloat/e5-small-v2` (384 dim, 33M params).
- **One large self-hosted.** `BAAI/bge-large-en-v1.5` (1024 dim, 335M params), `Snowflake/snowflake-arctic-embed-l` (1024 dim, 335M params), or `Alibaba-NLP/gte-large-en-v1.5` (1024 dim, 8k tokens).

If you run all three self-hosted (no hosted API available), keep the three axes — small vs. large, and one model that requires an instruction prefix.

Document the choice at the top of `docs/cost-latency-sweep.md`, with a one-sentence rationale per model.

## Part A — The corpus and the eval set (20 min)

Reuse exercise-02's `TinyDomainRetrieval` task if you built one, or reuse SciFact via `mteb.get_tasks(tasks=["SciFact"])`. Either is fine — the exercise is about *measurement methodology*, not about a large benchmark.

Also assemble a **latency probe set**: 500 sentences of varying length pulled from any English corpus (Wikipedia dump, Common Crawl News, or 500 chunks from mod-102's output). Save as `data/latency_probe.jsonl`. Aim for a token-length distribution across [50, 200, 500, 1000] so the throughput number is not skewed by one length bucket.

## Part B — Adapters and per-model plumbing (30 min)

Write one adapter per model in `src/embedders/`. Each exposes:

```python
class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str], *, batch_size: int = 32, role: str = "document") -> np.ndarray: ...
```

Rules:

- Hosted adapters route the `role` field to the provider's native distinction (`input_type="search_document"`/`"search_query"` for Cohere; `input_type="document"`/`"query"` for Voyage; `task_type="RETRIEVAL_DOCUMENT"`/`"RETRIEVAL_QUERY"` for Google; no-op for OpenAI). Chapter 4 argued that skipping this is a measurable quality hit — you are not allowed to skip it in this exercise.
- Self-hosted adapters for E5 / Nomic / GTE-Qwen / Snowflake v2 apply the model card's recommended prefix. Chapter 3 covered this; chapter 5 named the specific families.
- Hosted adapters must implement retry on `429` with exponential backoff. A corpus embed job that dies at 87% because you did not handle rate limits is not a fair cost comparison.
- Cache query embeddings by hash (chapter 6 named this as a real-world lever). Turn the cache on/off with a flag so you can measure both paths.

## Part C — Latency and throughput measurement (45 min)

Write `scripts/measure_latency.py` that, per adapter, produces:

- **Per-item p50 and p95 latency** on a batch of size 1 (measured over ≥100 items from `data/latency_probe.jsonl`). This is the *interactive query-encode latency* — what the user experiences.
- **Throughput** (`tokens_per_sec`) on a batch of size 64 (measured over ≥500 items). This is the *offline corpus-embed throughput*.
- **Warm-up handling.** Drop the first N=10 calls; the first call after model load or network cold-start is unrepresentative.
- **Concurrency (hosted only).** Run the batch-64 throughput with 1, 4, and 8 concurrent client tasks. Hosted providers typically get 5–20× throughput under concurrency until the rate-limit ceiling.

Persist per-model per-batch measurements to `results/latency/<model>.jsonl`, then produce `docs/latency-table.md`:

```markdown
| Model | Provenance | p50 latency (bs=1, ms) | p95 latency (bs=1, ms) | Throughput (bs=64, tokens/s) | Best concurrency | Warm-up cost (s) |
|---|---|---|---|---|---|---|
| ... | hosted / self-hosted | ... | ... | ... | ... | ... |
```

Latency measurement pitfalls (name them in a `## Methodology notes` section):

- Network path matters. Measure from the same region you would deploy from; a laptop in Sydney hitting `api.openai.com` has ~250 ms of RTT that will not exist from an AWS us-east-1 instance.
- Batch size of 1 on hosted APIs pays the full RTT + first-token latency; batch size of 64 amortizes the RTT. Both numbers matter — pick the one your deployment path actually uses.
- Self-hosted throughput on the first batch is bogus (CUDA warmup, kernel compilation). Discard.
- Pin your GPU / CPU — a background job on the same host will double p95.

## Part D — Cost model (30 min)

Compute the four cost surfaces from chapter 6 for each model. Concrete assumptions:

- **Corpus scale:** 1M chunks × 300 tokens/chunk = 300M corpus tokens.
- **Query volume:** 10M queries/month × 30 tokens/query = 300M query tokens/month.
- **Storage horizon:** 12 months.
- **GPU rental (for self-hosted):** the cheapest current spot rate for the tier that fits the model (`bge-small` on a T4 spot, `bge-large` on an A10G spot; look up current rates on AWS / GCP / Lambda Labs and cite the source). Include only the *steady-state* rate; ignore setup and idle-headroom cost for the exercise, and note the omission.

Fill `docs/cost-model.md`:

```markdown
| Model | Offline embed $ (300M tokens) | Query embed $/month (300M query tokens) | Storage $/month (raw vectors, 1M chunks) | Notes |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |
```

Rules:

- Cite the pricing page URL you used for each hosted provider. Prices move.
- Compute storage two ways per model: float32 (raw) and int8 (chapter 8's promise). Do not model MRL truncation here — that is exercise-05's turf.
- For self-hosted, throughput from Part C converts directly to offline embed cost: `time_hours = corpus_tokens / throughput_tokens_per_sec / 3600`, `cost = time_hours × gpu_hourly_rate`.

## Part E — Quality (30 min)

Run the three adapters through the eval set from Part A using MTEB or the `ir_measures`-based harness. Persist the score.

- Report **nDCG@10** and **Recall@100** for each model.
- **BM25 baseline** — include it. Chapter 2 said it is mandatory. Use `rank_bm25` on the same corpus.
- If the hosted API cannot be wrapped for MTEB in time, run the same corpus + queries manually: embed corpus and queries with the adapter, cosine-score with numpy, compute nDCG@10 with `ir_measures`. The number is comparable to what MTEB would have written.

Extend `docs/cost-latency-sweep.md` with a `Quality` section that pulls the score column into the master comparison table.

## Part F — The master comparison and recommendation (30 min)

Combine Parts C, D, and E into one table:

```markdown
| Model | Provenance | nDCG@10 (SciFact) | p95 latency (bs=1, ms) | Throughput (tokens/s) | Offline embed $ | Query embed $/month | Storage $/month (int8) |
|---|---|---|---|---|---|---|---|
| BM25 (`rank_bm25`) | Local | ... | n/a | n/a | ~0 | ~0 | ~0 |
| ... | ... | ... | ... | ... | ... | ... | ... |
```

Then apply chapter 6's rule:

1. **Set a quality floor** at (BM25 nDCG@10 + 0.05). Any model below the floor is out.
2. **Above the floor, pick the cheapest by total monthly cost** at the assumed scale.
3. **Only pay for extra quality if the delta is measurable and business-relevant** — a 0.02 nDCG lift for a 3× cost bump is not.

Close with a one-paragraph recommendation naming a winner and one alternate. Then a `## Where this decision could flip` section: at what scale would the ranking change? (A 10× query volume, a 10× corpus scale, a stricter latency SLO, a compliance restriction.) This is the section a reviewer probes; write it before you get asked.

## Deliverables

- `src/embedders/` — three adapters implementing the `Embedder` protocol.
- `data/latency_probe.jsonl` — 500 sentences with mixed length.
- `scripts/measure_latency.py`, `scripts/run_quality_eval.py`, `scripts/build_cost_table.py`.
- `results/latency/<model>.jsonl` — raw per-call measurements.
- `docs/latency-table.md`, `docs/cost-model.md`, `docs/cost-latency-sweep.md` — the three tables and the final synthesis.

## Acceptance criteria

- All three adapters produce vectors of the documented dim for the sample input; hosted adapters correctly route the `role` distinction.
- `docs/latency-table.md` contains p50/p95/throughput for all three models, with a methodology-notes section explicitly naming warm-up handling and concurrency.
- `docs/cost-model.md` contains offline-embed and monthly-query-embed dollars for all three models at the stated corpus/query scale, with the pricing-page URL cited per row.
- `docs/cost-latency-sweep.md` contains the master table (quality + latency + cost) with the BM25 row included.
- The recommendation names one model, states the floor, and calls out the specific delta ("beats floor by 0.08 nDCG at $X/mo total").
- The "Where this decision could flip" section names at least two scenarios that would change the ranking.

## Stretch goals

- Add MRL-truncation variants (OpenAI `dimensions=256` and `dimensions=1024`) as two extra rows and confirm they cost the same offline but change storage / ANN cost. Chapter 8 covers the mechanics; this is a preview.
- Serve `bge-large` behind [Text Embeddings Inference (TEI)](https://github.com/huggingface/text-embeddings-inference) instead of in-process `sentence-transformers`. Compare bs=1 p95 latency with and without TEI — you should see a meaningful drop for online serving under any real concurrency.
- Repeat the sweep on a *non-English* eval set (an MMTEB retrieval task, or a machine-translated slice of SciFact) and check whether the ranking changes. Chapter 4 argued Cohere / Voyage / Google are multilingual-stronger than OpenAI on public benchmarks; verify with your own numbers.
- Add a Bedrock passthrough of the same Cohere or Voyage model and confirm the vector is identical (they should be — Bedrock is a hosting layer, not a retraining). Note any *latency* delta from the extra AWS hop.
