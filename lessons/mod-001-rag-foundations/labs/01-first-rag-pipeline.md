# Lab 1 — Your first RAG pipeline

**Time:** ~60–90 minutes if the embedding and generation APIs are new to you.

## What you will build

A five-file, single-process RAG pipeline over a tiny corpus you assemble yourself. No vector database, no framework, no reranker. The point is to touch every stage with your own hands so you can feel where each seam is.

By the end you will have:

- `corpus/` — 5–10 short plain-text documents you pick.
- `ingest.py` — reads the corpus, produces chunks with metadata.
- `embed.py` — turns chunks into vectors and writes them to a JSON file.
- `query.py` — accepts a question, retrieves the top-*k* chunks, sends them to an LLM, prints the answer with citations.
- A short `NOTES.md` where you record what surprised you.

The reference solution lives in the paired [`rag-engineer-solutions`](https://github.com/ai-engineering-curriculum/rag-engineer-solutions) repo. Do not read it until you finish the "Stop and think" section at the end.

## Prerequisites

- Python 3.10+.
- An API key for **one** embedding model and **one** chat model. Any of the following combinations work for this lab; pick the pair you already have access to:
  - OpenAI `text-embedding-3-small` + `gpt-4o-mini` — see the [OpenAI API reference](https://platform.openai.com/docs/api-reference).
  - Cohere `embed-english-v3.0` + `command-r` — see the [Cohere API reference](https://docs.cohere.com/reference/).
  - Voyage `voyage-3` embeddings + any chat model you have — see the [Voyage AI docs](https://docs.voyageai.com/).
  - <!-- needs-research: confirm current on-device / open-weights option that the curriculum wants to recommend as a zero-cost alternative -->

You do **not** need a vector database. In-memory NumPy is fine at this scale.

## Constraints for this lab

To keep the seams visible, these constraints are deliberate:

- **No frameworks.** Do not use LangChain, LlamaIndex, or Haystack. You will use them later, but if you skip past the primitives now you will not know what they hide.
- **No vector database.** Store embeddings in a JSON or `.npz` file. Loop over them at query time. Yes, this is slow — but at ten documents it is *fast enough*, and you get to see the actual distance function.
- **Chunk by fixed character count.** 800 characters with 100 characters of overlap. This is not a good policy in general; that is what a later module is for. Use it here so that the chunking policy is not what you are debugging.

## Steps

### 1. Assemble a corpus

Pick 5–10 short factual documents. Public sources with a clear owner work best — for example:

- 3–4 pages of the [Python 3 language reference](https://docs.python.org/3/reference/).
- 2–3 pages of the [PostgreSQL documentation](https://www.postgresql.org/docs/current/).
- One or two `README.md` files from open-source projects you know well.

Save each as a plain `.txt` file in `corpus/`. Keep each under ~10 KB. Record the source URL as the first line of each file, prefixed with `SOURCE:`.

### 2. Ingest

Write `ingest.py` that:

1. Walks `corpus/`, reads each file.
2. For each file, produces a list of chunks. Chunk boundary rule for this lab: 800 characters, sliding window with 100 characters of overlap.
3. Attaches `{"source": <first line after SOURCE:>, "chunk_id": <int>, "text": <string>}` to each chunk.
4. Writes the result to `chunks.jsonl`.

### 3. Embed

Write `embed.py` that:

1. Reads `chunks.jsonl`.
2. Calls your chosen embedding model in batches. Check the provider's docs for the maximum batch size; do not guess.
3. Writes `embeddings.jsonl` — same rows as `chunks.jsonl` but with an added `embedding: [float, ...]` field.

### 4. Query

Write `query.py` that:

1. Accepts a question on the command line.
2. Embeds the question with the *same* model you used for the corpus.
3. Loads `embeddings.jsonl` into memory. Computes cosine similarity between the question vector and every chunk vector. Returns the top *k* = 4.
4. Builds a prompt of the form:

    ```
    Answer the question using only the passages below. If the passages do not contain the answer, say so. Cite each claim with the source URL in square brackets.

    Passages:
    [1] {chunk 1 text}  (source: {chunk 1 source})
    [2] {chunk 2 text}  (source: {chunk 2 source})
    …

    Question: {user question}
    ```

5. Sends the prompt to your chat model.
6. Prints the model's answer and, underneath it, the four sources it was given.

### 5. Try it

Run at least six queries you invent yourself, in three deliberate categories:

- **Directly answerable** — the answer is verbatim in one chunk.
- **Requires composition** — the answer needs two chunks.
- **Not in the corpus** — the answer is genuinely absent.

Record what happens in each case in `NOTES.md`.

## Stop and think

Before you look at the reference solution, answer these in your notes. The lab is more useful than the answer:

1. What did your pipeline do on the "not in the corpus" queries? Did it refuse, hedge, or confabulate? What in your prompt (or the retrieval) drove that behaviour?
2. Pick one query that returned a wrong or thin answer. Which of the five stages was most at fault? What would you change in *only* that stage?
3. Your chunks overlap by 100 characters. Did any query benefit from the overlap? How would you tell if it did?
4. If your corpus doubled in size, which stage becomes the first bottleneck? Which becomes the first *cost* problem?

Bring these notes to the next module.

## Extension (optional)

If you finish early: add BM25 (via [`rank_bm25`](https://pypi.org/project/rank-bm25/), a small pure-Python library) as a second retriever, take the union of its top *k* and the vector retriever's top *k*, and see whether any of your test queries improve. This is a preview of the hybrid-search module later in the track — do not tune it seriously here.
