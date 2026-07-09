# Quiz 1 — Foundations check

Ten short items. Answer without looking at the lecture. Then check yourself. If you miss an item, re-read the relevant section — do not just note the correct answer.

Questions first; answers and pointers at the bottom.

## Questions

1. In the terminology of Lewis et al. (2020), what is *parametric* memory and what is *non-parametric* memory?

2. List the five stages of the minimal RAG pipeline in order.

3. Which of the five stages run offline (i.e., not on the user's query path)?

4. Why must the query be embedded with the same model that embedded the corpus?

5. Give one concrete example of a fact a language model's *parametric* memory is unlikely to hold, no matter how large the model is.

6. What is the difference between a chunk and a document, and why do we chunk?

7. Name a failure mode of RAG that is *not* the retriever's fault.

8. In the lecture, the phrase "seam problems" is used. What does it mean, and why is it a useful framing?

9. Why is citation a first-class feature of RAG rather than an afterthought?

10. The original RAG paper introduced two variants. Do modern production systems typically use those variants, or have they moved on? Briefly, to what?

---

## Answers

1. **Parametric memory** = knowledge encoded in a model's weights. **Non-parametric memory** = knowledge held in an external corpus and retrieved at query time. Lecture: *Parametric vs. non-parametric memory*.

2. Ingest → Chunk → Embed → Retrieve → Generate. Lecture: *Anatomy of a minimal RAG pipeline*.

3. Ingest, Chunk, Embed. Retrieve and Generate are online. Lecture: *Anatomy of a minimal RAG pipeline*.

4. Because retrieval works by comparing vectors in the same space. Two embedding models produce different, incompatible spaces; distances between them are meaningless. Lecture: *Anatomy* — *Embed* and *Retrieve*.

5. Any of: a fact that post-dates the model's training cutoff; a fact from a private corpus (internal policy, ticket, customer record) not in the public training data; an attribution the model needs to source back to a specific document. Lecture: *Why the technique exists*.

6. A document is the raw unit ingested (a page, a PDF, a wiki article). A chunk is a passage cut from a document, small enough for the retriever to score independently and for the generator to fit alongside other passages in its context window. Lecture: *Chunk*.

7. Any of: bad chunk boundaries, unfaithful generation ("grounded hallucination"), stale index, access leakage, ingest that dropped structural content. Lecture: *Where RAG fails*.

8. A "seam problem" is a failure that lives between two stages rather than inside one — the upstream stage produced something the downstream stage cannot handle correctly. It is a useful framing because most RAG bugs are seam problems, and knowing which seam is leaking tells you which stage to fix. Lecture: *Where RAG fails* (closing paragraph).

9. Because the retrieved chunk *is* the evidence. If you cannot show the user which chunk grounded a claim, the whole "we retrieved before answering" story is unverifiable. Lecture: *Why the technique exists* — bullet 3.

10. Modern systems typically decouple the retriever from the generator and use an off-the-shelf LLM instead of the jointly-trained RAG-Sequence / RAG-Token architectures. The idea "retrieve then generate" is what stuck; the specific architecture from the paper is not what most people run today. Lecture: *Why the technique exists*.
