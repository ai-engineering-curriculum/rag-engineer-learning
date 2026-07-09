# Lecture 1 — What is RAG?

## Why the technique exists

Large language models store what they know in their parameters. That works for facts that are ubiquitous in the training data and stable over time, and fails in three specific ways that RAG was invented to address:

1. **Fresh facts.** A model's weights are frozen at training time. Anything that happened afterward — a released product, a policy change, a new customer record — is invisible unless retrieved.
2. **Private facts.** Anything not in the public training corpus (your company's tickets, contracts, internal wiki) cannot be recalled from weights at all.
3. **Attributable facts.** Even when the model does recall a fact, it cannot point to the sentence it came from. That makes disagreements ("where did you get that from?") unresolvable.

Retrieval-augmented generation is the pattern of answering a query by first fetching relevant text from an external corpus and then conditioning the language model's answer on that text. The corpus is swappable, updatable, and citeable; the model's weights are not.

The name and the modern formulation come from Lewis et al. (2020), *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, published at NeurIPS 2020 (see [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)). The paper introduced two variants — RAG-Sequence and RAG-Token — that jointly train the retriever and the generator. In practice, most production systems today use a decoupled retriever plus an off-the-shelf LLM; the *architecture* has moved on, but the *idea* — "retrieve then generate" — is the one that stuck.

## Parametric vs. non-parametric memory

Lewis et al. frame the design as combining two kinds of memory:

- **Parametric memory** — knowledge encoded in the weights of a language model. Fast to access, blends across topics, hard to update, impossible to cite.
- **Non-parametric memory** — knowledge held in an external corpus (documents, database rows, code files). Slow to access, discrete, easy to update, trivially citeable because retrieval returns the exact passage used.

A RAG system is deliberately hybrid. The retriever hands the generator a small non-parametric slice for each query; the generator uses its parametric knowledge (grammar, reasoning, prose style) to compose an answer grounded in that slice.

## Anatomy of a minimal RAG pipeline

Almost every RAG system, from a fifty-line prototype to a production platform, is a specialisation of five stages:

```
    ┌──────────┐   ┌────────┐   ┌───────┐   ┌────────────┐   ┌───────────┐
    │  Ingest  │──▶│ Chunk  │──▶│ Embed │──▶│  Retrieve  │──▶│  Generate │
    └──────────┘   └────────┘   └───────┘   └────────────┘   └───────────┘
     (offline)     (offline)    (offline)     (online)          (online)
```

Only the last two stages run on the hot path of a user query. The first three are batch jobs over your corpus.

### 1. Ingest

Pull raw content out of its source (a Confluence export, a PDF drop, a Postgres table, a Git repo) and put it in a form the rest of the pipeline can iterate over. Ingest owns character-set handling, structural parsing (headings, tables, code blocks), and metadata (`source_url`, `updated_at`, `access_scope`) that later stages will attach to every chunk.

### 2. Chunk

Split each document into passages small enough for a retriever to score independently and for a generator to fit alongside other passages in its context window. Chunking policy — size, overlap, boundary rules — is one of the largest levers in a RAG system, and later modules devote significant space to it.

### 3. Embed

Convert each chunk to a vector using an embedding model. Two chunks with similar meaning should land close together in the vector space. This is what makes semantic (as opposed to lexical) retrieval possible. Vectors are written to a **vector store** — an index that supports fast approximate nearest-neighbour search.

### 4. Retrieve

At query time, embed the user's query with the same embedding model, ask the vector store for the *k* nearest chunks, and return them. Real systems combine this with lexical retrieval (BM25) and often a reranker; those come in later modules.

### 5. Generate

Assemble a prompt that contains the user's query, the retrieved chunks, and instructions to answer *from the chunks*. Send it to a language model. Return the model's answer plus references back to the retrieved chunks.

## Where RAG fails, and why the rest of this curriculum exists

A minimal pipeline hides a lot of ways to be wrong. Each of the following failure modes has an entire module (or several) devoted to it later in the track:

- **Bad chunk boundaries.** A fact is split across two chunks and neither retrieves. The answer either invents a bridge or admits ignorance.
- **Vocabulary mismatch.** The user asks about "PTO" and the source document says "annual leave"; a pure-semantic retriever may still miss.
- **Relevant but not sufficient retrieval.** The right chunk was retrieved but so were four wrong ones that dominate the prompt.
- **Unfaithful generation.** The generator produces a fluent answer that contradicts the retrieved passage. This is sometimes called *grounded hallucination*.
- **Stale index.** The document was updated in the source; the vector store still has the old version.
- **Access leakage.** The retriever surfaces a chunk the querying user was not allowed to see.

Notice that only the last two are strictly retrieval problems. Most RAG failures are actually *seam* problems between the stages listed above. Getting good at RAG is largely getting good at spotting which seam is leaking.

## Terminology check

You should be able to answer the following without looking anything up before moving on:

- What is the difference between parametric and non-parametric memory?
- Which of the five pipeline stages run offline, and which run online?
- Why is chunking not just "split every 1,000 characters"?
- Why do we embed the query with the same model we used to embed the chunks?
- Give one failure mode that is *not* the retriever's fault.

The exercise next door works these into a short classification task.

## Further reading (official / primary)

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) — the paper that named the pattern.
- Guu, K., et al. (2020). *REALM: Retrieval-Augmented Language Model Pre-Training.* ICML 2020. [arXiv:2002.08909](https://arxiv.org/abs/2002.08909) — an earlier, jointly-trained retrieval-augmented LM that motivates several design choices in RAG.
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) — dense retrievers, the workhorse of the retrieval stage.
