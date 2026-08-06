# Exercise 01 — Map the RAG pipeline and its decision points

**Estimated effort:** ~2 hours. Paper-and-pen for the first pass; a text file for the deliverable.

## Objective

You will produce a one-page pipeline map of a production-shaped RAG system, annotate every stage with the engineering decision it forces, and use that map to diagnose a set of realistic failure scenarios. The goal is not to memorize the diagram from chapter 1 — it is to make the diagram *yours*, so that future modules feel like drilling into a picture already in your head.

## Prerequisites

- Chapter 1 (*The End-to-End RAG Pipeline and the Decision at Each Stage*) — read once.
- Chapter 3 (*Retrieval as a Feature vs. Retrieval as an Engineered System*) — skimmed for the feature/system distinction.
- No code required.

## Part A — Draw the pipeline (30 min)

On paper (or in a whiteboard tool of your choice), draw the seven-stage pipeline from chapter 1:

```
Ingest → Chunk → Embed → Retrieve → Rerank → Assemble → Generate
```

For each stage, write:

1. **One-sentence purpose.** What does this stage do? Write it as if you were explaining it to a new hire.
2. **Offline or online?** Which stages run in a batch job, which run on the user's hot path.
3. **The engineering decision.** The single most consequential decision made *at this stage* (e.g. for *Chunk*, the decision is the boundary policy; for *Retrieve*, it is the index + ANN parameters + whether it is lexical/dense/hybrid).
4. **Two knobs.** Two specific tunables you could turn to change the system's behavior. Be concrete: not "the chunker" but "chunk size" and "overlap." Not "the retriever" but "`efSearch`" and "top-*k*."
5. **Which downstream module owns depth.** Point at the mod-102 through mod-112 module that goes deep on this stage.

Add the two cross-cutting decisions (evaluation and MLOps) as boxes running *around* the seven-stage flow, not in the flow.

**Deliverable A:** a photograph or scan or diagram-tool export of your annotated map. Commit it as `pipeline-map.png` (or `.pdf`) alongside the writeup deliverable in Part C.

## Part B — Diagnose eight failure cases (45 min)

For each case below, decide (1) which stage is *most likely* at fault, (2) which specific knob you would turn first, and (3) whether the fix belongs to *this role* or to a peer / higher role (feature-shape vs. engineered-system-shape from chapter 3, or the ownership contract from chapter 4).

Some cases have more than one plausible answer. When more than one stage is defensible, pick the earliest — the earliest broken stage is usually the one worth fixing first, because bad output at stage *N* is often bad input at stage *N+1*.

1. The user asks *"how do I request parental leave?"* The corpus contains a policy titled "Parental Leave Application (Form HR-14)." The retriever returns three chunks about time-off tracking, none of them the HR-14 policy. Retrieval score for the correct chunk was 0.31; for the top-1 wrong chunk it was 0.44. BM25 was not used.
2. Same query. Now the correct HR-14 chunk is in the top-3. The generator's answer says "*You can request parental leave by emailing hr@company.com*" — a plausible sentence that does not appear anywhere in the retrieved chunks.
3. Same query. Retrieval returns the correct chunk *and* an outdated 2019 draft of the policy that was superseded but never removed from the corpus. The generator quotes the outdated draft.
4. Ingestion drops all table cells from PDF exports and keeps only the surrounding paragraphs. The user asks about salary bands; retrieval returns the *Compensation Framework* section but the answer omits the actual bands.
5. p95 latency crossed 800 ms after the corpus grew from 500 K to 5 M chunks. Recall@10 on the judgment set is unchanged. The vector-DB CPU is not saturated.
6. A user on Tenant A sees a chunk from Tenant B's private wiki in the retrieved context. The ACL check happens in the application after retrieval returns.
7. Recall@10 on the judgment set improved from 0.71 to 0.79 after the team swapped embedding models, but user complaints about "answers are subtly wrong" *increased* over the same window.
8. Every third redeploy of the retriever service breaks with `MODEL_MISMATCH: vector dimension 384 does not match index dimension 768`.

**Deliverable B:** a short table with columns `case | stage | knob to turn first | who owns it`. Two to three lines of reasoning per row is enough — do not write essays.

## Part C — Write the pipeline stanza (30 min)

Write the pipeline as a short (one-page) design-doc stanza you would drop into a real project's `README.md`. It should:

- Name the seven stages.
- For each stage, state the specific technology choice you would default to for a *new* project on a modest corpus (5 M chunks, mixed HTML + PDF, no ACL requirements yet). You do *not* need to justify the choice yet — just make it, and know that mod-102 through mod-107 will let you replace each choice with a defended one.
- End with a two-sentence "known gaps" section that names two of the cross-cutting decisions (evaluation, MLOps) as *not yet in the initial design* — so that a reviewer knows you have not forgotten them.

**Deliverable C:** `pipeline-stanza.md` in your notes repo (or paste it into an issue on your fork of this curriculum).

## Acceptance criteria

- Deliverable A includes all seven stages with all five annotations per stage, plus the two cross-cutting boxes.
- Deliverable B has all eight cases classified. At least six of the eight identify the correct stage; the remaining two must have defensible reasoning for a plausible alternative.
- Deliverable C's stanza names an actual default per stage (e.g. "pgvector HNSW" — not "a vector DB"). The two known-gaps items must be genuine (evaluation and MLOps or similar) rather than filler.

## Stretch goals

- Redo Part A for the two extra shapes mentioned in Gao et al. 2024's taxonomy: draw one *Naïve RAG* pipeline (three stages, minimal) and one *Modular RAG* pipeline (with an iterate-until-satisfied loop between retrieve and generate). Compare them side-by-side.
- Extend Part B with three new cases from a real incident you have seen (redacted). Contribute the best case back to this exercise as a PR against this repo.
