# Exercise 04 — Context Assembly With Dedup, Ordering, and Citation Anchors

**Estimated effort:** ~3 hours. Python, a reranked top-K per query from exercise 01/02, an LLM API for the generator + evaluator, and access to the target model's tokenizer.

## Objective

Build a full context assembler that implements chapter 7's four jobs (dedup, ordering, citation anchors, token budgeting), produce a proper `AssembledContext` object, and measure the impact of each job on end-to-end answer quality via an ablation.

The exercise is the mid-level "make the assembler pass a real integration test" step between "reranker returns a top-K" and "generator produces a cited answer." At the end, you have a reusable assembler and evidence that each of the four jobs earns its keep on your data.

## Prerequisites

- Chapter 7 read.
- Chapter 6 read (compression, if you enabled it in exercise 03).
- Chapter 8 read (for the ordering-affects-quality argument).
- A reranked top-K per query from exercises 01/02.
- The target LLM's tokenizer available (`tiktoken` for OpenAI, `transformers.AutoTokenizer` for open models, [Anthropic count-tokens](https://docs.claude.com/en/api/messages-count-tokens) for Claude).
- An eval harness that measures faithfulness and citation-verifiability (see exercise 03 setup).

## Steps

**Step 1 — Author the `AssembledContext` shape (30 minutes).** In `assembler.py`, define the dataclasses from chapter 7:

```python
@dataclass
class SourceRef: ...
@dataclass
class AssembledChunk: ...
@dataclass
class AssembledContext: ...
```

Match chapter 7's fields exactly. This is the interface that crosses to mod-108 (and, at scale, to `applied-ai-engineer`); the shape matters.

**Step 2 — Implement the greedy-pack assembler (30 minutes).** The base algorithm from chapter 7:

```python
def assemble(
    query: str,
    query_id: str,
    reranked: list[RerankedChunk],
    budget_tokens: int,
    tokenizer,
    anchor_style: str = "numeric",   # "numeric" for [1], [2]; "source" for [Source A]
) -> AssembledContext:
    ...
```

- Iterate reranked in order.
- For each, assign an anchor, compute token count with the header (`"[1] " + text`), skip if it would exceed budget.
- Return the `AssembledContext` with `total_tokens` = sum of actual token counts.

Do *not* implement dedup, ordering variants, or MMR yet. This is the baseline.

**Step 3 — Add deduplication (20 minutes).** Add three configurable dedup strategies:

- By `doc_id` with a cap (e.g., "at most 2 chunks per doc").
- By content-hash on normalised text.
- MMR — pick chunks that maximise `lambda * rerank_score - (1-lambda) * max_similarity_to_already_selected`. Use the same embedder as your retriever to compute similarity.

Expose as `dedup_strategy: str` and `dedup_config: dict` on the `assemble()` function.

**Step 4 — Add three ordering strategies (20 minutes).**

- `best_first`: chunks sorted by rerank score descending (baseline).
- `first_and_last`: top-2 chunks first, then chunks 3..K-1 in descending order, then a copy of chunk 1 at the end with a marker like `[1, repeated] ...` (chapter 7's variant).
- `structural`: chunks ordered by (`doc_id`, position within document) — the chunk's index inside its source document.

Structural ordering needs the position-within-document metadata. If your chunks don't have it, note the omission — mod-102's ingestion contract owns this metadata.

**Step 5 — The four-job ablation (60 minutes).** Take a set of ~30 eval queries. For each, produce five `AssembledContext` variants:

- **A0 — concat baseline.** No anchors, no dedup, no ordering (just concatenate top-K chunk texts). Feed to the generator with a generic prompt.
- **A1 — anchors only.** Baseline greedy-pack with numeric anchors, best-first ordering, no dedup.
- **A2 — anchors + dedup.** Add `doc_id` cap of 2.
- **A3 — anchors + dedup + first-and-last ordering.** For K > 5.
- **A4 — anchors + dedup + ordering + measured token budget.** Actual `total_tokens` computed with the target tokenizer; assert the LLM prompt fits the context window.

For each variant, run the generator with a citation-per-claim prompt and measure:

- Faithfulness.
- Citation-verifiability.
- Answer relevance.
- Token count.

Aggregate over the 30 queries. Which job produced the biggest quality delta? Which produced the smallest?

**Step 6 — The `AssembledContext` contract test (30 minutes).** Write an integration test that asserts the four contract clauses from chapter 9:

1. **Anchor uniqueness within the response** — every `chunks[].anchor` is unique.
2. **Anchor → SourceRef stability** — indexing into `chunks` by anchor returns the correct `SourceRef`.
3. **`total_tokens` accuracy** — re-run the tokenizer on the assembled context and compare to `total_tokens`; they must match exactly.
4. **Anchor-in-LLM-output valid** — the citation numbers in the LLM's generated answer must all correspond to real anchors in the assembled context (test on at least 5 queries).

**Step 7 — Write it up (30 minutes).** In `assembly-ablation.md`:

- The ablation table (variants × metrics).
- A one-paragraph diagnosis: which job earns its keep on your data? Is dedup useful (do you have real duplicate chunks)? Is first-and-last ordering useful (does K > 5 apply to your setting)? Is the budgeting-with-actual-tokenizer catch preventing real overflows (or is it defensive)?
- One paragraph on anti-patterns chapter 7 named — did you find any in your existing (pre-exercise) code?
- The contract-test file, checked in.

## Acceptance criteria

- `assembler.py` implements the `AssembledContext` shape from chapter 7 with dataclasses.
- Greedy-pack assembly with an actual tokenizer-based token count (not an estimate).
- Three dedup strategies (doc_id cap, content-hash, MMR) available as options.
- Three ordering strategies (best-first, first-and-last, structural) available as options.
- Ablation with at least 4 variants (baseline concat, anchors, anchors + dedup, anchors + dedup + ordering + budget) measured on ≥ 20 queries.
- Contract-test file that checks the four clauses (anchor uniqueness, anchor stability, token accuracy, LLM anchor validity).
- Writeup naming which job earned the biggest quality delta on your data.

## Stretch goals

- **Per-task assembler config.** Add a `task_type` parameter that switches ordering strategy (best-first for Q&A, structural for summarisation, first-and-last for K > 5 synthesis). Measure whether task-conditional ordering beats one-size-fits-all.
- **Verbose vs. compact anchors.** Compare numeric `[1], [2]` to verbose `[Source 1: title, retrieved 2025-...]`. Does the verbose form improve citation-verifiability, and is the extra token cost worth it?
- **Duplicate detection at the sentence level.** Instead of dedup by chunk, detect duplicated sentences across chunks and mark them for the LLM ("this sentence also appears in [3]"). Does this improve or hurt the answer?
- **Structural ordering integration with mod-102.** Update your chunking pipeline to always emit `doc_id + chunk_position` metadata; verify the structural ordering works end-to-end.
- **A/B in production.** If you have a live RAG service, deploy the assembler behind a feature flag, A/B against your existing assembler, and measure any real-user metric (session satisfaction, escalation-to-human rate, click-through-to-citation). This is the peer-track observability layer — treat this as a joint exercise with an `applied-ai-engineer`-role teammate.

## Notes

- The four jobs are not always equally important. Dedup is high-impact when the corpus is redundant; low-impact when sources are naturally distinct. Ordering is high-impact when K > 5; low-impact when K ≤ 5. Budgeting is *always* important — a silent overflow is the worst-case failure and the cheapest to prevent.
- Do not couple your assembler to a specific generator. The `AssembledContext` shape is generator-agnostic; the mod-108 prompt (or the peer track's prompt) does the model-specific rendering. Keep the layers clean.
- Log the `diagnostics` field (reranker version, K, dedup strategy, ordering strategy, per-chunk source hash). This is what mod-111 (MLOps) uses to detect drift.
- If you find that A0 (concat baseline) beats A4 on your eval set, something is very wrong. Diagnose before assuming assembly is useless — most likely candidates: your metric harness has a bug, your citation prompt is not being followed, or your K is so small (K=1 or 2) that ordering and dedup have no room to matter.
- Chapter 9's anti-patterns (peer re-ordering, peer truncation, peer re-fetching sources) will bite if the assembler crosses a team boundary. Discuss the contract with whoever consumes `AssembledContext` before shipping.
