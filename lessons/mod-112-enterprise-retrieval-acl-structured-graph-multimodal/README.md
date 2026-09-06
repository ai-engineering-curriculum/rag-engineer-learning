# mod-112 · Enterprise Retrieval: ACL-Aware, Structured, Graph, and Multimodal

Twelfth module in the RAG & Retrieval Engineer track. Takes the tuned, monitored, MLOps-scaffolded retriever the earlier modules produced and puts it in front of an *enterprise* corpus — heterogeneous (text + rows + graph + images), permissioned (SharePoint / Drive / Salesforce ACLs), and ingested continuously from source systems you don't own. This is the module where a retriever grows the extra surfaces enterprise data forces on it and stops being an all-vector search.

**Estimated effort:** 14 hours.

## Learning objectives

By the end of this module a learner can:

- Design permission-aware retrieval — pick between query-time filtering, index-time filtering, and per-tenant index isolation for a given ACL shape, name the correctness trap of each, and layer them in the common production configuration (per-tenant outer + query-time inner).
- Explain the ACL model shapes enterprise corpora inherit — SharePoint's inheritance / break-inheritance, Google Drive's parents + shortcuts, Salesforce's OWD + role hierarchy + sharing rules + territory + manual/Apex shares — and design an ingestion pipeline that stores the *effective* ACL and reacts to permission-change events.
- Implement text-to-SQL retrieval — build a schema linker, drive an LLM generator via LlamaIndex `NLSQLTableQueryEngine` or a LangChain SQL agent, gate the pipeline with a `sqlglot`-based safety guard + read-only DB grant + statement timeout, and evaluate with execution accuracy against Spider / BIRD and a domain-fit set.
- Implement Graph RAG over a property graph — Neo4j / Kùzu / Memgraph as the store, LlamaIndex `PropertyGraphIndex` or Microsoft GraphRAG as the orchestration, deterministic construction + LLM extraction blended, and know when multi-hop structure actually lifts recall versus when it just adds latency.
- Survey multimodal retrieval — CLIP / SigLIP for image-text, ColPali / DSE for document-page retrieval, Whisper-transcribe + text-embed (or CLAP) for audio, keyframe + transcript for video — and pick a starting stack per modality your corpus contains.
- Compose the above behind a hybrid router — rule-based, classifier-based, or LLM-based dispatch, exclusive vs. fan-out, and a *policy contract* every retriever must honor so adding one doesn't become a leak accelerator.
- Cite the boundary to `ai-risk-engineer` (peer, level 25) on content-safety / PII redaction methodology; this module owns ACL *mechanics* only.
- Cite the boundary to `ai-infra-senior-engineer` (level 40) on cross-tenant isolation at platform scale; this module owns the *concept* and the collection-per-tenant implementation, not thousands-of-tenants operational shape.

## Prerequisites

- All prior modules in this track. In particular:
  - `mod-104-vector-databases-and-ann-tuning` — chapter 7 (metadata filtering, pre-filter vs. post-filter) is the ANN-level mechanic chapter 2's query-time enforcement rides on.
  - `mod-106-hybrid-retrieval-fusion-query-rewriting` — the RRF and weighted-sum fusion patterns are what the router (chapter 7) uses to merge results across retrievers of different modalities.
  - `mod-107-reranking-and-context-assembly` — the reranker window sits after the router's merge; the "lost in the middle" caveats transfer.
  - `mod-109-retrieval-quality-evaluation` — recall@k / MRR / nDCG on judgment sets is the base; every chapter here adds an orthogonal eval hook (ACL fuzz test, execution accuracy, multi-hop recall lift, routing accuracy).
  - `mod-110-orchestration-and-mcp-exposure` — the individual retrievers this module composes behind a router are the MCP-exposed retrievers built there.
  - `mod-111-retrieval-mlops-versioning-cicd-drift` — every eval loop in this module (ACL fuzz, execution accuracy, routing accuracy, multi-hop recall) plugs into the retriever CI gate built there.
- Comfort with SQL (CTEs, window functions, joins across a handful of tables) for chapter 4.
- Comfort with Cypher (Neo4j's query language; also accepted by Kùzu and Memgraph) for chapter 5. The [Neo4j Cypher manual — introduction](https://neo4j.com/docs/cypher-manual/current/introduction/) is enough.
- Docker or an equivalent for running Neo4j / Qdrant locally; enough disk / RAM to hold a small property graph (~100 K nodes, ~500 K relationships) and a handful of vector collections.
- API access for one hosted LLM provider (OpenAI, Anthropic, or a local coder model via Ollama) for the text-to-SQL and Graph-RAG generation calls.

## Structure

| Path | What it is |
|---|---|
| [`01-why-enterprise-retrieval-is-different.md`](01-why-enterprise-retrieval-is-different.md) | The three properties enterprise corpora violate (heterogeneous, permissioned, ingested-from-elsewhere), the two motivating failures (cross-tenant leak, wrong-retriever-shape), and the peer boundaries. |
| [`02-permission-aware-retrieval-mechanics.md`](02-permission-aware-retrieval-mechanics.md) | The three enforcement points (query-time filter, index-time filter, per-tenant isolation), correctness traps of each, effective-ACL denormalization, belt-and-braces post-check, and the four-part permission-aware eval loop. |
| [`03-enterprise-acl-model-shapes.md`](03-enterprise-acl-model-shapes.md) | SharePoint inheritance / break-inheritance, Google Drive parents + shortcuts + shared drives, Salesforce OWD + role hierarchy + sharing rules + territory + manual/Apex shares, and the general "you inherit the ACL model" pattern. |
| [`04-text-to-sql-retrieval.md`](04-text-to-sql-retrieval.md) | The four-stage text-to-SQL pipeline (schema-link → generate → guard → execute), LlamaIndex `NLSQLTableQueryEngine` and the LangChain SQL agent, `sqlglot`-based safety guards, and execution-accuracy eval on Spider / BIRD. |
| [`05-graph-rag-on-property-graphs.md`](05-graph-rag-on-property-graphs.md) | Property graphs (Neo4j / Kùzu / Memgraph), deterministic construction vs. LLM extraction, LlamaIndex `PropertyGraphIndex`, Microsoft GraphRAG, and when multi-hop structure lifts recall. |
| [`06-multimodal-retrieval-survey.md`](06-multimodal-retrieval-survey.md) | Image-text (CLIP / SigLIP), document-page (ColPali / DSE), audio (Whisper + text-embed / CLAP), video (keyframe + transcript / video-clip encoders), and the multi-index-with-router shape a cross-modal stack takes. |
| [`07-hybrid-enterprise-router.md`](07-hybrid-enterprise-router.md) | Rule-based, classifier-based, and LLM-router dispatch; exclusive vs. fan-out; the policy contract that carries the caller's ACL through every branch; router evaluation on routing accuracy, end-to-end quality, and policy compliance. |
| [`08-boundaries-to-ai-risk-and-ai-infra-senior.md`](08-boundaries-to-ai-risk-and-ai-infra-senior.md) | What this module owns vs. what `ai-risk-engineer` (PII / content safety) and `ai-infra-senior-engineer` (platform-scale isolation) own; the hand-off artifacts in both directions. |
| [`exercises/`](exercises/) | Five hands-on prompts — one per major objective. |
| [`labs/`](labs/) | Long-form hands-on labs (scaffolded separately). |
| [`quizzes/`](quizzes/) | Knowledge check (scaffolded separately). |
| [`resources.md`](resources.md) | Primary-source references for the topics covered. |

## How to work through it

1. Read chapter 01 first. It states the five surfaces the rest of the module covers and draws the two peer boundaries; every later chapter refers back to that frame.
2. Read chapters 02 and 03 back-to-back, then do `exercises/exercise-01-…`. Chapter 02's three enforcement points map to your retriever's mechanics; chapter 03's SharePoint / Drive / Salesforce shapes are what those mechanics have to survive on real data. The drill runs the three variants side-by-side and gates on a leak-rate-zero fuzz test.
3. Read chapter 04 and do `exercises/exercise-02-…`. The schema-linker → generator → guard → execute pipeline is what "answer with SQL" looks like as a retriever surface; the execution-accuracy eval is what plugs it into the mod-111 CI.
4. Read chapter 05 and do `exercises/exercise-03-…`. A small Neo4j graph, LlamaIndex's `PropertyGraphIndex`, and a multi-hop judgment set is enough to feel the recall lift where the graph earns its keep — and to see where it doesn't.
5. Read chapter 06 and do `exercises/exercise-04-…`. Three encoders on the same corpus of visually-rich pages is how you internalize why "text-only retrieval on PDFs is leaving recall on the table."
6. Read chapter 07 and do `exercises/exercise-05-…`. The router is the composition layer; wiring the previous exercises' retrievers behind it and gating on the three-axis eval is how you know the ensemble beats each retriever alone.
7. Read chapter 08 last. With direct experience of the retrieval surfaces, the boundary contract to `ai-risk-engineer` and `ai-infra-senior-engineer` reads as a checklist rather than as theory.

## Ownership boundary

This module owns the *retrieval mechanics* of enterprise corpora: ACL-aware retrieval and the three enforcement points, the ACL shapes ingested from the common SaaS systems, text-to-SQL as a retriever, Graph RAG as a retriever, multimodal survey and per-modality picks, and the router that composes them behind a single policy contract.

It does **not** own PII detection, redaction methodology, prompt-injection defense evaluation, or fairness/bias evaluation of retrieval outputs — those are [`ai-risk-engineer-learning`](https://github.com/ai-engineering-curriculum) at level 25 (Governance family). It also does **not** own platform-scale per-tenant isolation (thousands of tenants, per-tenant encryption keys, cross-region replication, tenant-aware autoscaling) — that is [`ai-infra-senior-engineer-learning`](https://github.com/ai-engineering-curriculum) at level 40 (Infra family). Chapter 08 draws both lines explicitly and lists the hand-off artifacts.

The router this module builds is a *retriever surface*. The agent that decides *when* to call it — tool-selection policy, multi-step planning, cross-tool memory — is [`agentic-ai-engineer-learning`](https://github.com/ai-engineering-curriculum) at level 40.

## Primary sources

- Microsoft Learn. *Introduction to permissions in SharePoint.* <https://learn.microsoft.com/en-us/sharepoint/introduction>
- Google Workspace Admin Help. *How Drive sharing works.* <https://support.google.com/a/answer/60781>
- Salesforce Architects. *Record-Level Security.* <https://architect.salesforce.com/decision-guides/data-sharing>
- Yu, T., et al. (2018). *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.* EMNLP. <https://arxiv.org/abs/1809.08887>
- Li, J., et al. (2023). *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs (BIRD).* NeurIPS. <https://arxiv.org/abs/2305.03111>
- Edge, D., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Microsoft Research. <https://arxiv.org/abs/2404.16130>
- Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML. <https://arxiv.org/abs/2103.00020>
- Faysse, M., et al. (2024). *ColPali: Efficient Document Retrieval with Vision Language Models.* <https://arxiv.org/abs/2407.01449>
- Anthropic Engineering. *Building Effective Agents.* <https://www.anthropic.com/engineering/building-effective-agents>
- OWASP. *Top 10 for LLM Applications.* <https://owasp.org/www-project-top-10-for-large-language-model-applications/>

Further reading — chapter-by-chapter and grouped by topic — is in [`resources.md`](resources.md).
