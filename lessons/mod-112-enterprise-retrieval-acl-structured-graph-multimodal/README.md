# mod-112-enterprise-retrieval-acl-structured-graph-multimodal: Enterprise Retrieval: ACL-Aware, Structured, Graph, and Multimodal

> Scaffolded by `aicg org execute-plan`. Lecture chapters and exercise content are authored on subsequent autonomous cycles.

**Estimated effort:** 14 hours

## Learning objectives

- Design permission-aware retrieval — filter at query time, filter at index time, per-tenant index isolation, and the correctness trap of each; understand the SharePoint / Google Drive / Salesforce ACL-model shape that enterprise corpora inherit
- Implement text-to-SQL retrieval over structured data — schema-linking, few-shot SQL generation with LlamaIndex NLSQLTableQueryEngine or LangChain SQL agent, query-safety guards, and the eval loop (execution accuracy on Spider / BIRD)
- Implement Graph RAG over a property graph — Neo4j / Kùzu / Memgraph + a Property Graph Index (LlamaIndex) or Microsoft GraphRAG; understand when graph structure lifts recall on multi-hop questions and when it doesn't
- Survey multimodal retrieval — CLIP / SigLIP for image-text, ColPali / DSE for document-page retrieval, audio embeddings (Whisper + text-embed proxy or CLAP), video-clip retrieval — and know where the frontier sits
- Reason about hybrid enterprise stacks — mixing vector, lexical, structured (SQL), and graph retrievers behind a router
- Cite the boundary to `ai-risk-engineer` (peer, level 25) on content-safety / PII redaction methodology; this module covers ACL-*mechanics* only
- Cite the boundary to `ai-infra-senior-engineer` (level 40) for cross-tenant isolation at platform scale

## Structure

- `01-…md` … `0N-…md`: lecture chapters.
- `exercises/`: per-exercise prompts.
- `labs/`: long-form hands-on labs.
- `quizzes/`: knowledge checks.
- `resources.md`: external references.
