# Exercise 01 — Vector Database Family Comparison Matrix

**Estimated effort:** ~2 hours. All reading, note-taking, and Markdown. No code required.

## Objective

Build a defensible comparison matrix of six vector databases against a shared rubric, then use the matrix to make (and defend) a project-scoped recommendation. The point of the exercise is not to end with a table — it is to end with the *skill* of reading a new vendor's docs and slotting them into the rubric in under an hour, so your team's vendor-selection conversations start from evidence rather than folklore.

## Prerequisites

- Chapters 1 and 2 read.
- Nothing installed. This is a documentation-and-writing exercise.

## The rubric — seven cells per system

For each of the six systems below, write one paragraph (3–5 sentences) per cell, citing the primary docs page you drew from. Do not paste vendor marketing prose; state the fact in your own words with a link.

1. **Deployment shape** — library / self-hosted service / managed service / in-database.
2. **Persistence and durability** — what is on disk vs. RAM; replication semantics; backup story.
3. **Index families supported** — HNSW, IVF-family, PQ variants, disk-tiered, GPU; is the algorithm exposed to you, or hidden behind an abstraction?
4. **Filter story** — pre-filter, post-filter, or integrated; how predicates on payload are pushed into the ANN scan.
5. **Multi-tenant isolation** — collection, namespace, partition, per-tenant index; strength of isolation.
6. **Quantization / memory tiering** — int8, PQ, BQ, disk-tier; what knobs are exposed.
7. **Update model** — insert / delete / update cost; incremental vs. rebuild; what happens during a schema change.

## The six systems

Pick from this list; do all six for the required deliverable (skip Pinecone or FAISS only if you have a strong reason and document it):

- **pgvector** — [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector)
- **Pinecone** — [docs.pinecone.io](https://docs.pinecone.io/)
- **Weaviate** — [weaviate.io/developers/weaviate](https://weaviate.io/developers/weaviate)
- **Qdrant** — [qdrant.tech/documentation](https://qdrant.tech/documentation)
- **Milvus** — [milvus.io/docs](https://milvus.io/docs)
- **Chroma** — [docs.trychroma.com](https://docs.trychroma.com/)

FAISS is deliberately not required — a library-only entry breaks several of the cells. Add it if you want as a seventh entry with cells marked "N/A — library."

## Steps

**Step 1 — Read.** For each system, spend 15–20 minutes in the primary docs. Skim the *Concepts* or *Architecture* section end-to-end before drilling into any single cell.

**Step 2 — Fill the matrix.** Write the seven cells per system as a Markdown table or as one section per system with subsections per cell. Use whichever renders more legibly to you; the graders care about the content, not the layout.

**Step 3 — Add three column-level takeaways.** After the matrix, write one paragraph per row of the following three questions:

- *"If I already run Postgres, which one am I most likely to reach for and why?"*
- *"If I need to serve 500 M vectors with a 40 GB memory ceiling, which two systems are the shortlist and why?"*
- *"If my priority is per-tenant isolation for 1000+ tenants, which system's story fits and why?"*

Each answer must cite the matrix cells that support it.

**Step 4 — Author a project recommendation.** Pick a hypothetical project (or a real one from your own work). Specify:

- Corpus size in 18 months (order of magnitude is fine).
- Embedding dimensionality.
- QPS target.
- Multi-tenancy shape (single tenant / few large tenants / many small tenants).
- Latency SLO.
- Team's existing infrastructure (Postgres, Elasticsearch, Kubernetes, none).

Given those constraints, recommend one vendor and one runner-up. Defend both choices against the matrix, in ≤ 300 words.

## Acceptance criteria

- Matrix has 6 systems × 7 cells filled, with a docs link in each cell.
- Every claim is your own paraphrase — no vendor blockquotes.
- Three column-level takeaways are present with citations back to the matrix.
- Recommendation section (Step 4) is ≤ 300 words and names both a chosen vendor and a runner-up.
- Nowhere in the artifact does the phrase "sub-10 ms latency" or "billions of vectors" appear without a specific citation — every performance claim is either a docs link or an "I did not verify this."

## Stretch goals

- **Add a seventh system.** Vespa, Redis Search + `VECTOR`, LanceDB, Turbopuffer, MongoDB Atlas Vector Search, ClickHouse `usearch`, or Cassandra 5 SAI. The exercise is complete when the seventh system's cells are as strong as the mandatory six.
- **Prove the filter cell.** For any two systems, find in their docs the exact API surface you would use to filter on a tenant-ID predicate and paste the shortest working query snippet. Cite the docs page.
- **Diff two versions of one vendor.** Pick a vendor whose recent release changed the ANN story (pgvector 0.4 → 0.7, Weaviate 1.18 → 1.25, Milvus 2.3 → 2.4). Write a 200-word diff highlighting what changed in the rubric cells and why it matters.
- **Interview yourself.** Have someone (or an LLM playing a peer engineer) ask you *"why not pgvector?"* / *"why not Pinecone?"* for your recommendation. Note where your defense wobbled and go patch the matrix.
