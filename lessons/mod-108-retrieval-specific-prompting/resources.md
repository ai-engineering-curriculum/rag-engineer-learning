# Resources for mod-108-retrieval-specific-prompting

Curated references for the four retrieval-specific prompt patterns (citation, refusal, contradiction, post-generation verification) and the boundary to `applied-ai-engineer` and `llm-application-developer`.

## Framing papers

- Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) — the paper that named RAG; useful for the "grounding" framing this module builds on.
- Menick, J., Trebacz, M., Mikulik, V., et al. (2022). *Teaching Language Models to Support Answers with Verified Quotes.* [arXiv:2203.11147](https://arxiv.org/abs/2203.11147) — GopherCite; the reference for the answer-with-quote pattern (chapter 3) and post-generation quote verification (chapter 6).
- Bohnet, B., Tran, V. Q., Verga, P., et al. (2022). *Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models.* [arXiv:2212.08037](https://arxiv.org/abs/2212.08037) — a systematic treatment of attribution/citation for QA answers.
- Gao, T., Yen, H., Yu, J., and Chen, D. (2023). *Enabling Large Language Models to Generate Text with Citations.* EMNLP. [arXiv:2305.14627](https://arxiv.org/abs/2305.14627) — the ALCE benchmark, useful for choosing between citation shapes.
- Rashkin, H., Nikolaev, V., Lamm, M., Aroyo, L., et al. (2023). *Measuring Attribution in Natural Language Generation Models.* Computational Linguistics 49(4). [aclanthology.org/2023.cl-4.2/](https://aclanthology.org/2023.cl-4.2/) — the AIS framework for attribution measurement.

## Refusal and abstention

- Kadavath, S., Conerly, T., Askell, A., et al. (2022). *Language Models (Mostly) Know What They Know.* [arXiv:2207.05221](https://arxiv.org/abs/2207.05221) — model self-confidence signals; relevant to chapter 4's confidence-gated refusal.
- Ren, R., Wang, Y., Qu, Y., et al. (2023). *Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation.* [arXiv:2307.11019](https://arxiv.org/abs/2307.11019) — when RAG helps refusal precision.
- Yu, W., Zhang, H., Pan, X., et al. (2024). *Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models.* [arXiv:2311.09210](https://arxiv.org/abs/2311.09210) — a "notes per chunk before answering" approach that improves refusal precision.
- Feng, S., Shi, W., Bai, Y., et al. (2024). *Don't Hallucinate, Abstain: Identifying LLM Knowledge Gaps via Multi-LLM Collaboration.* ACL. [arXiv:2402.00367](https://arxiv.org/abs/2402.00367) — abstention as a first-class design goal.

## Contradiction and conflicting evidence

- Chen, H., Xu, F. F., Arora, S., and Choi, Y. (2022). *Rich Knowledge Sources Bring Complex Knowledge Conflicts.* EMNLP. [arXiv:2210.13701](https://arxiv.org/abs/2210.13701) — a systematic treatment of knowledge conflicts in RAG.
- Xie, J., Zhang, K., Chen, J., Lou, R., and Su, Y. (2024). *Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts.* ICLR. [arXiv:2305.13300](https://arxiv.org/abs/2305.13300) — whether LLMs prefer parametric vs. retrieved on conflict; directly relevant to chapter 5.
- Longpre, S., Perisetla, K., Chen, A., et al. (2021). *Entity-Based Knowledge Conflicts in Question Answering.* EMNLP. [arXiv:2109.05052](https://arxiv.org/abs/2109.05052) — earlier reference on measuring model behavior on conflicting evidence.
- Mallen, A., Asai, A., Zhong, V., et al. (2023). *When Not to Trust Language Models.* ACL. [arXiv:2212.10511](https://arxiv.org/abs/2212.10511) — the parametric-vs-retrieved boundary.

## Post-generation verification

- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — the faithfulness / citation-precision metric that chapter 6's level-3 check operationalises.
- Yue, X., Wang, B., Chen, Z., et al. (2023). *Automatic Evaluation of Attribution by Large Language Models.* EMNLP. [arXiv:2305.06311](https://arxiv.org/abs/2305.06311) — ATTRSCORE; benchmarks LLM-based vs. NLI-based attribution evaluation.
- Manakul, P., Liusie, A., and Gales, M. J. F. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection.* EMNLP. [arXiv:2303.08896](https://arxiv.org/abs/2303.08896) — the self-check pattern's reference.

## Vendor documentation

- Anthropic: [*Long context tips*](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/long-context-tips) — placement, ordering, and prompt-cache guidance for document-heavy prompts.
- Anthropic: [*Use XML tags*](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags) — the XML delimiter recommendation chapter 2 follows.
- Anthropic: [*Reduce hallucinations*](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/reduce-hallucinations) — "allow the model to say I don't know" guidance behind chapter 4.
- Anthropic: [*Structured outputs*](https://docs.claude.com/en/docs/build-with-claude/structured-outputs) — the primitive that makes chapter 3 pattern 2 reliable on Claude.
- Anthropic: [*Prompt caching*](https://docs.claude.com/en/docs/build-with-claude/prompt-caching) — placement constraint that pushes retrieved context out of the system message.
- OpenAI: [*Prompt engineering guide*](https://platform.openai.com/docs/guides/prompt-engineering) — the level-20 baseline this module extends.
- OpenAI: [*Structured Outputs*](https://platform.openai.com/docs/guides/structured-outputs) — the primitive for chapter 3 pattern 2.
- OpenAI: [*Prompt caching*](https://platform.openai.com/docs/guides/prompt-caching) — equivalent placement constraint on OpenAI.

## Production-ready verifier and eval frameworks

- [TruLens](https://www.trulens.org/) — including the [*Groundedness*](https://www.trulens.org/trulens_eval/getting_started/core_concepts/feedback_functions/) and [*RAG Triad*](https://www.trulens.org/trulens_eval/getting_started/core_concepts/rag_triad/) feedback functions; operationalises chapter 6's level 3.
- [DeepEval](https://github.com/confident-ai/deepeval) — the [Faithfulness](https://docs.confident-ai.com/docs/metrics-faithfulness) and citation metrics for post-generation checking.
- [RAGAS](https://github.com/explodinggradients/ragas) — the Python implementation of the RAGAS paper's metrics; usable both offline (mod-109 eval) and as a runtime verifier (chapter 6).

## Adjacent-track references

- [`llm-application-developer-learning`](https://github.com/ai-engineering-curriculum/llm-application-developer-learning) — level-20 baseline prompt engineering; the foundation this module extends (chapter 7).
- [`applied-ai-engineer-learning`](https://github.com/ai-engineering-curriculum/applied-ai-engineer-learning) — the peer track this module hands its four fragments and verifier to (chapter 7).
- Mod-102 of this track — where `SourceRef.updated_at`, `authority`, `page`, and `section` metadata are populated (chapters 2, 5, 6 depend on them).
- Mod-107 of this track — where `AssembledContext` is produced (chapter 2 renders it; chapter 6 verifies against it).
- Mod-109 of this track — where the eval methodology for citation-precision, refusal-precision, and verifier-precision lives.
- Mod-110 of this track — where the retriever + prompt fragments are packaged as an MCP tool (chapter 5 pattern D resolver, chapter 7 hand-off).
- Mod-111 of this track — where the fragments, refusal thresholds, and verifier are versioned, deployed, and drift-monitored.

## Also worth a look

- Google SRE Workbook: [*Implementing SLOs*](https://sre.google/workbook/implementing-slos/) — the shape of the contract clauses in chapter 7.
- Liu, N. F., Lin, K., Hewitt, J., et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) — the reference behind placement-in-context; complementary to chapter 2's placement-in-prompt discussion.
- Anthropic Cookbook: [*Retrieval-augmented generation with citations*](https://github.com/anthropics/anthropic-cookbook/) — vendor recipe covering rendered context + citation prompt; matches this module's default pattern.
