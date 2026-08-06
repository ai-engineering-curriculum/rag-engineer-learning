# Exercise 01 — Elasticsearch vs. OpenSearch vs. Vespa vs. Solr: The Family Map

**Estimated effort:** ~2 hours. All reading, note-taking, and Markdown. No code required.

## Objective

Build a defensible family map of the four Lucene-family engines (Elasticsearch, OpenSearch, Solr, Vespa) against the shared rubric from chapter 1, then use the map to make and defend an engine recommendation for a project you specify. The point is not the table — it is the *skill* of reading a vendor's docs and slotting them into a rubric in under an hour, so vendor-selection conversations start from evidence.

## Prerequisites

- Chapter 1 read end-to-end.
- Access to each vendor's docs (all publicly indexed).
- Nothing installed. Documentation-and-writing exercise.

## The rubric — six cells per engine

For each of the four engines below, write **one paragraph (3–5 sentences) per cell**, citing the primary docs page you drew from. Do not paste vendor marketing prose; state the fact in your own words with a link.

1. **License and history.** Current license (Apache 2.0, ELv2, SSPL, AGPL v3, other), any relevant licensing history (the 2021 Elastic split, the OpenSearch fork), and the practical implication for a corporate OSS policy.
2. **Deployment shape.** Self-hosted (VMs, K8s operator name, container image), first-party managed offering, cloud-provider managed offering. Who is on call.
3. **Vector / ANN maturity.** Field type name; ANN engine(s); distance metrics; quantization options; the earliest version that had production HNSW; any distinguishing feature (e.g., multi-vector tensors, PQ, choice of ANN engine in the mapping).
4. **Learning-to-rank story.** Built-in vs. plugin; supported model formats (XGBoost, LightGBM, RankLib, native); how models are uploaded; where in the query pipeline the model runs.
5. **Query / DSL surface.** Native query language (JSON DSL, HTTP params, YQL, ranking expression); the equivalent of `multi_match` and `function_score`; the native hybrid-retrieval construct (RRF, hybrid query, ranking expression).
6. **Cluster / coordination shape.** Node roles, primary shard model (or tensor/attribute shape for Vespa), coordination store (built-in vs. ZooKeeper), operator ecosystem.

## The four engines

Do all four:

- **Elasticsearch** — [elastic.co/guide/en/elasticsearch/reference/current/](https://www.elastic.co/guide/en/elasticsearch/reference/current/)
- **OpenSearch** — [opensearch.org/docs/latest/](https://opensearch.org/docs/latest/)
- **Solr** — [solr.apache.org/guide/solr/latest/](https://solr.apache.org/guide/solr/latest/)
- **Vespa** — [docs.vespa.ai](https://docs.vespa.ai/)

## Steps

**Step 1 — Read (60 min).** Spend ~15 min in each engine's *Introduction*, *Architecture*, or *Concepts* section end-to-end before drilling into any single cell. Skim, do not read every option.

**Step 2 — Fill the matrix (40 min).** Write the six cells per engine as either a Markdown table (4 engines × 6 columns) or as four sections with subsections. The graders care about the content, not the layout. Every cell must have at least one docs link.

**Step 3 — Author the four column-level takeaways (15 min).** After the matrix, write one paragraph per question:

- *If our OSS policy forbids source-available licenses (SSPL, ELv2), which engines are still on the table?*
- *If we need multi-vector / late-interaction ranking (ColBERT-style), which engine is the strongest pick and why?*
- *If we already run Solr for another workload, is there a compelling case to add Elasticsearch or OpenSearch for the retrieval workload? Under what conditions?*
- *If we're on AWS with no existing search cluster and want a managed offering, which engine is the least-friction choice and what do we give up?*

Each answer must cite the matrix cells that support it.

**Step 4 — Author a project recommendation (25 min).** Pick a hypothetical project (or a real one from your own work). Specify:

- Corpus size and dominant document type (support articles, product catalogue, log data, etc.).
- License / OSS policy constraint (Apache 2.0 only? AGPL acceptable? SSPL acceptable?).
- Team's existing infrastructure (already running ES/OS/Solr/Vespa? nothing?).
- Vector-and-hybrid ambition (single-vector HNSW? multi-vector / ColBERT? no vectors at all?).
- LTR ambition (any labels or click log? plan to add one?).
- Cloud posture (AWS-blessed managed? on-prem K8s? Elastic Cloud OK?).

Given those constraints, recommend one primary engine and one runner-up. Defend both choices against your matrix and the four-question framework from chapter 1, in ≤ 400 words.

## Acceptance criteria

- All 4 engines × 6 cells filled with a docs link in each.
- Every claim paraphrased — no vendor blockquotes.
- All four column-level takeaways present with citations back to matrix cells.
- Recommendation section ≤ 400 words with a primary pick and a runner-up.
- License history correctly named (ES license split date, OpenSearch fork date, ES AGPL re-add year) with a citation.
- No performance claims (`p99 latency`, `billions of vectors`) copied from vendor marketing without a specific docs citation or an "I did not verify this" note.

## Stretch goals

- **Add a fifth engine** as a comparison anchor: MeiliSearch, Typesense, or Manticore Search. The point is to see what a *non-Lucene* full-text engine looks like against the same rubric.
- **Vector-benchmark cross-reference.** For each of the four engines' HNSW implementations, find the [ann-benchmarks.com](https://ann-benchmarks.com/) entry (if listed) or a vendor-published benchmark and note the recall/latency numbers. Flag any that were run by the vendor themselves vs. by an independent party.
- **Deprecation-and-support cycle.** For each engine, find the current LTS version and the support-window commitment (Elastic's [product-support policy](https://www.elastic.co/support/eol), OpenSearch's [release schedule](https://opensearch.org/releases.html), Solr's [Apache-typical policy](https://solr.apache.org/downloads.html), Vespa's [release notes](https://docs.vespa.ai/en/vespa-releases.html)). Which engines are you comfortable committing a 3-year roadmap to?
- **Migration cost estimate.** Pick two engines. Sketch what a migration from one to the other would cost in mapping-translation, analyzer-porting, and API-rewriting terms. Rank difficulty from easiest to hardest of the six possible pairwise migrations.

## Notes

- The four-question engine-selection framework from chapter 1 (license, incumbency, ANN needs, LTR ambition) is the scaffold for Step 4; use it explicitly.
- Do not skip Vespa because it looks unfamiliar — its ranking-expression story is genuinely distinct and important to understand even if you don't pick it.
- If a docs page has moved, use [web.archive.org](https://web.archive.org/) to find the version that existed when you wrote the note. The URL is the citation; keep it working.
