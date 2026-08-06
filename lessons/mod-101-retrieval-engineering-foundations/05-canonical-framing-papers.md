# Chapter 5 — The Two Papers Every Retrieval Engineer Should Have Read

## Why this chapter exists

Two papers define the shared vocabulary of modern RAG work. Interviews, design docs, vendor pitches, and open-source project READMEs all assume you have read them.

- **Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.** [arXiv:2005.11401](https://arxiv.org/abs/2005.11401). *The paper that named the pattern.*
- **Gao, Y., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* v5, March 2024.** [arXiv:2312.10997](https://arxiv.org/abs/2312.10997). *The map of the landscape four years later.*

This chapter is a *reading guide*, not a substitute for the papers. Read the actual PDFs — twice for Lewis 2020, once with the section index open for Gao 2024. The paired exercise walks you through both with a structured annotation prompt.

## Lewis et al. 2020 — the paper that named the pattern

### What to know before you open it

The 2020 landscape it landed in:

- Open-domain question answering was being reshaped by **dense passage retrievers** (Karpukhin et al. 2020, *DPR*, [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)) — the direct predecessor for the retriever half of RAG.
- **REALM** (Guu et al. 2020, [arXiv:2002.08909](https://arxiv.org/abs/2002.08909)) had already trained a language model *jointly* with a retriever, but for masked-LM pre-training rather than seq2seq generation.
- Sequence-to-sequence LMs like **BART** (Lewis et al. 2019, [arXiv:1910.13461](https://arxiv.org/abs/1910.13461)) were the strong-general-purpose generative backbone.

The paper's contribution is *combining* those pieces — a dense passage retriever + a seq2seq generator, jointly usable at inference time — and *naming* the pattern **retrieval-augmented generation**.

### The sections that carry the load

Read in order:

1. **Section 1 (Introduction).** Frames the parametric-vs-non-parametric memory distinction that is still the cleanest one-sentence justification for RAG. Copy the phrasing into your own vocabulary.
2. **Section 2 (Methods).** Two variants:
   - **RAG-Sequence** — for a given output sequence, one retrieved document is used across all tokens. Marginalize over the top-*K* retrieved docs.
   - **RAG-Token** — a different retrieved doc can inform each output token; marginalize per-token.
   Most production systems today do neither — they retrieve once, condition the generator on the top-*K* jointly, and skip the marginalization. But the *shape* of thinking about the retriever's output as a distribution the generator conditions on carries forward into every reranker and every LLM-as-reranker design.
3. **Section 4 (Results) and Section 5 (Discussion).** Skim the tables. Note that (a) RAG beats a parametric-only baseline on open-domain QA and (b) it can be *updated* by swapping the non-parametric index without retraining — the design property that made the pattern deployable.

### The three things to underline

- **The retriever and the generator are *decoupled* enough that they can be trained separately and swapped independently.** This is what made a whole industry possible; it is why we can pin an off-the-shelf `bge-large` retriever to a Claude/GPT/Gemini generator without joint training.
- **"Non-parametric" memory can be updated without retraining.** This is the property that legal, product, and support teams will implicitly demand from your system — read the paper's phrasing so you can defend it when asked.
- **Both variants define a distribution over answers marginalized over retrieved docs.** The current production shape (retrieve → rerank → assemble → generate) is a *cheaper* approximation to the same idea. Knowing the exact form clarifies what your approximation is giving up.

### Common misreadings to avoid

- "RAG means fine-tuning the generator on retrieved passages." No — RAG-Sequence and RAG-Token do fine-tune, but the pattern named in the paper is retrieval-augmented generation *at inference*. Modern usage is closer to inference-only conditioning.
- "The paper says dense retrieval beats BM25." The paper uses a DPR-style dense retriever because that is the strong retriever of the moment; it does not claim lexical retrieval is dead. In fact, mod-106 shows that hybrid (BM25 + dense) usually beats either alone on real corpora.
- "The retriever always sees a query and returns *K* passages." This describes the *interface*, but the paper's RAG-Token variant lets each output token *reweight* the passages — a subtler shape than the interface suggests.

## Gao et al. 2024 — the survey that maps the landscape

The paper (v5, March 2024) is a taxonomy of the field four years after Lewis 2020. It is long. Read it with the section index open; you are looking for a map, not memorizing every method.

### The three-era taxonomy

Gao et al. divide RAG systems into three eras. Learn the names — every design doc you review will use them.

- **Naïve RAG.** Retrieve → generate. One shot, no rewriting, no reranking. This is the level-20 shape from `llm-application-developer` mod-005. Fine for a demo; brittle in production.
- **Advanced RAG.** Adds pre-retrieval steps (query rewriting, chunk optimization, metadata attachment) and post-retrieval steps (reranking, prompt compression, context re-ordering). This is roughly the shape of a mod-106 → mod-107 pipeline in this track.
- **Modular RAG.** Composes patterns as swappable modules — routers, memory, adapters, fusion — and often loops (retrieve → generate → re-retrieve). This is where mod-110 sits: orchestration frameworks + MCP-exposed retrievers agents can call multiple times.

The taxonomy is a *diagnostic*: given a project description, you should be able to say "this is Naïve RAG that will need the Advanced additions," or "this needs to be Modular from day one because it is agent-consumed."

### The sections worth deep-reading

Read at least the following (section numbers refer to arXiv v5, March 2024 — verify against the copy you download):

- **Section 2 (Overview).** The three-era taxonomy and the top-level pipeline diagram. Memorize both.
- **Section 3 (Retrieval).** Covers retrieval source (documents / structured / knowledge graphs), granularity (token / phrase / chunk / doc), indexing structures, embedding model families, and the query-side techniques (rewriting, expansion, transformation) that mod-106 explores. Skim once, then re-read the sub-section closest to your current corpus.
- **Section 4 (Generation).** Covers context curation (which chunks, in what order, compressed how), and LLM fine-tuning options (rarely used in practice today because closed-model APIs dominate; still relevant when you serve your own generator).
- **Section 5 (Augmentation Process).** *Iterative*, *recursive*, and *adaptive* retrieval. This is where the "retrieve → verify → re-retrieve" loops that agentic patterns depend on are formalized. Read this section carefully — it is the vocabulary for handing off to `agentic-ai-engineer` (level 40).
- **Section 6 (RAG vs. Fine-Tuning).** The clearest published statement of when to reach for retrieval vs. weight-updating. Bookmark for the design-doc argument you will have every quarter.
- **Section 7 (Evaluation).** Introduces the eval axes and popular benchmarks. Retrieval-only eval (recall@k / MRR / nDCG on judgment sets) is what mod-109 owns; application-layer eval (RAG-triad — faithfulness / answer relevance / context precision-recall) is what `ai-eval-engineer` owns. Read the section aware of that boundary.

### The three things to underline

- **RAG is a *family* of patterns, not a single algorithm.** Every design conversation should start by locating the proposal in the Naïve / Advanced / Modular taxonomy — that alone rules out most bad shapes.
- **Retrieval quality and generation quality are separable evaluations.** The survey names both; the two evaluations attribute to different owners (this track vs. `ai-eval-engineer`).
- **The Modular RAG frontier — agentic retrieval, self-routing, iterative re-retrieval — is where retrieval and agentic engineering meet.** MCP-exposed retrievers (mod-110) are the interface across that seam.

### Common misreadings to avoid

- "The survey ranks methods." It classifies and describes; it rarely benchmarks. Do not treat any method presented as "the best." Run your own domain-fit benchmark (mod-103).
- "Advanced RAG obsoletes Naïve RAG." It does not — Naïve is still the right shape for the right problem. Advanced is the right shape when the requirements from chapter 3 tip you across.
- "The survey's diagrams are prescriptive architectures." They are illustrative shapes. Your production shape will look like *some* of the diagram, not all of it, and that is normal.

## How to read papers like these on-the-job

- **Two-pass read.** First pass: title → abstract → figures → conclusion. Decide if it is worth a full read. Second pass: sections in the order given above, with a note-taking template.
- **Note-taking template** (used in the paired exercise):
  - *Contribution in one sentence.*
  - *The one figure you would put in a design doc.*
  - *The one metric or experimental setup you would reproduce.*
  - *The one limitation you would call out in a review.*
  - *The one open question you would follow up on.*
- **Trace every claim to a stage in the seven-stage pipeline (chapter 1).** If a paper cannot be located on your pipeline map, either the paper is out of scope for this role or your map is missing a stage.

## What you should be able to do after this chapter

- State Lewis 2020's contribution in one sentence and distinguish RAG-Sequence from RAG-Token.
- Name Gao 2024's three-era taxonomy (Naïve, Advanced, Modular) and place a project on the spectrum.
- Cite at least one section from each paper you would send a colleague to for a specific design question.

## Further reading (the papers themselves — read these, not just this chapter)

- Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- Gao, Y., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey.* v5. [arXiv:2312.10997](https://arxiv.org/abs/2312.10997).
- Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) — the direct predecessor Lewis 2020 builds on.
- Guu, K., et al. (2020). *REALM: Retrieval-Augmented Language Model Pre-Training.* ICML 2020. [arXiv:2002.08909](https://arxiv.org/abs/2002.08909) — the jointly-trained retrieval-augmented LM that motivates several of Lewis 2020's design choices.
