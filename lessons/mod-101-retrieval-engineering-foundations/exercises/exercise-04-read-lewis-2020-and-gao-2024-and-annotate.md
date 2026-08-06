# Exercise 04 — Read Lewis 2020 and Gao 2024 and annotate them

**Estimated effort:** ~2 hours (Lewis 2020 read + notes ~45 min, Gao 2024 mapped read + notes ~60 min, synthesis ~15 min). Reading and writing.

## Objective

You will read the two canonical framing papers for this track using the two-pass method from chapter 5, produce a structured annotation for each, and finish with a short synthesis that places a project you know (real or hypothetical) on Gao et al.'s Naïve / Advanced / Modular taxonomy. The goal is to make the papers *reference material you can retrieve from* — not one-time inspirational reads.

## Prerequisites

- Chapter 5 (*The Two Papers Every Retrieval Engineer Should Have Read*).
- PDF or arXiv HTML of the two papers open:
  - Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
  - Gao, Y., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* v5. [arXiv:2312.10997](https://arxiv.org/abs/2312.10997).

## Part A — Lewis 2020 annotation (45 min)

Read Lewis 2020 in two passes as chapter 5 suggests. Then fill in this template.

```markdown
# Annotation — Lewis et al. 2020, RAG for Knowledge-Intensive NLP

Contribution in one sentence:
  <one sentence in your own words>

RAG-Sequence vs. RAG-Token in one paragraph each:
  RAG-Sequence: <what the retriever sees per output, how the marginalization works, when this shape fits>
  RAG-Token: <same three>

The one figure I would put in a design doc:
  Figure <N>. Why: <one sentence>.

The one experimental result I would reproduce (or defend against reproducing today):
  <result and setup>. Why: <one sentence>.

The one limitation I would call out in a review:
  <one sentence>

The one open question I would follow up on:
  <one sentence>

Where each of the seven pipeline stages (from chapter 1) appears in the paper — cite section or figure:
  Ingest:    <where, if at all>
  Chunk:     <where, if at all>
  Embed:     <where, if at all>
  Retrieve:  <where, if at all>
  Rerank:    <where, if at all>
  Assemble:  <where, if at all>
  Generate:  <where, if at all>

  Which of the seven stages does the paper NOT cover, and why does that make sense given the paper's year?
    <one paragraph>
```

**Deliverable A:** `annotation-lewis-2020.md`.

## Part B — Gao 2024 mapped read (60 min)

Gao 2024 is a survey; do not attempt to read it front-to-back. Follow the section guide from chapter 5:

- Sections 1–2 (introduction + overview): read every word. This is where the Naïve / Advanced / Modular taxonomy lives.
- Sections 3–5 (retrieval, generation, augmentation process): read the section headings and skim every subsection first pass. Second pass: pick two subsections most relevant to a project you actually care about and read those in full.
- Section 6 (RAG vs. Fine-Tuning): read every word.
- Section 7 (Evaluation): read every word, aware of the retrieval-only vs. application-layer split from chapter 4.
- The remaining sections: read once quickly.

Fill in this template.

```markdown
# Annotation — Gao et al. 2024, RAG for LLMs: A Survey

The three-era taxonomy in one line each:
  Naïve RAG:     <one line>
  Advanced RAG:  <one line>
  Modular RAG:   <one line>

The single figure I would drop into a design doc:
  Figure <N>. Why: <one sentence>.

Two subsections I read in depth (name and section number):
  1. <section> — because <one sentence>. What I took away in one paragraph:
     <paragraph>
  2. <section> — because <one sentence>. What I took away in one paragraph:
     <paragraph>

The paper's split of evaluation into two layers, in your own words:
  <paragraph — with an explicit mention of the boundary to `ai-eval-engineer`>

The one claim in the paper I would push back on:
  <one sentence claim> — because <one sentence>.

Three papers cited by Gao 2024 that I would read next:
  1. <cite + why>
  2. <cite + why>
  3. <cite + why>
```

**Deliverable B:** `annotation-gao-2024.md`.

## Part C — Synthesis: place a project on the taxonomy (15 min)

Pick one project. It can be:

- A real project at your workplace (write about it at whatever level of abstraction feels appropriate — a public write-up is not required).
- A hypothetical project of your own design.
- The scenario from exercise 03 (the mid-size B2B SaaS company).

Then write a one-page synthesis:

1. **One paragraph describing the project.** What is the corpus, who are the users, what problem does the retriever solve?
2. **Where on the Naïve / Advanced / Modular spectrum does the project sit *today*?** Why?
3. **Where does it need to sit in twelve months to meet real requirements?** Which of Gao 2024's Advanced or Modular components would you add, in what order?
4. **Which of Lewis 2020's design properties (retriever-generator decoupling, updatable non-parametric memory) is load-bearing for this project?** Explain why in one paragraph.

**Deliverable C:** `synthesis.md`.

## Acceptance criteria

- Deliverables A, B, C exist and follow the templates above.
- Deliverable A's "where each stage appears" section explicitly notes at least two stages the paper does *not* cover in the depth this track's chapter 1 covers them (rerank and assemble are the obvious candidates given the paper's year).
- Deliverable B's Naïve / Advanced / Modular one-liners each fit on one line and are in your own words.
- Deliverable C places its project on the taxonomy with a stated *why*, and names at least one specific Advanced or Modular component to add and the reason.

## Stretch goals

- After finishing the annotation, read one of the three cited-next papers from your Deliverable B list. Add a short annotation for it in the same format.
- Compare Gao 2024's Naïve / Advanced / Modular framing with the classification used in another recent survey (e.g. *A Comprehensive Survey of Retrieval-Augmented Generation (RAG)* — search arXiv for the current best-cited survey when you do the exercise). Note one place where the two frameworks disagree and, in one paragraph, argue which framing you prefer.
- Present your Deliverable C synthesis to a colleague or study partner. Ask them to poke holes in the placement. Update the doc.
