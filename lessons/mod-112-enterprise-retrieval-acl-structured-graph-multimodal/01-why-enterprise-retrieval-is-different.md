# Chapter 1 — Why Enterprise Retrieval Is a Different Beast

## Why this chapter exists

Modules 101–111 assumed a corpus that is *homogeneous* (all text chunks with roughly the same shape), *public within its scope* (any authenticated user of the retriever may see every chunk), and *pre-processed once* (ingested by you, into your index). Real enterprise corpora violate all three assumptions on day one:

- The **corpus is heterogeneous** — half of it is unstructured text (wiki pages, tickets, PDFs), a quarter is rows in a database (CRM, HRIS, finance), and the rest is a graph of relationships (org chart, product hierarchy, contract → clause → obligation, drug → protein → disease). No one shape of retriever answers every question against that pile.
- The **corpus is permissioned** — every doc has an owner, a share list, a group inheritance chain, and (in Salesforce and similar) a rule engine that says *this row is visible if the calling user is in this role hierarchy and this territory and the record type matches*. The retriever cannot serve the same top-k to two different users. Getting this wrong leaks data.
- The **corpus is ingested continuously from systems you don't own** — SharePoint, Google Drive, Confluence, Salesforce, ServiceNow, Slack, JIRA. Their ACL models are what your ACL model becomes; you inherit their bugs. And the schema of the structured slice is what your text-to-SQL layer must understand — the columns of somebody else's database.

This module is about the retrieval concerns those three realities create. The five prior modules on chunking, embedding, indexing, ranking, and evaluation still apply — permission-aware retrieval is still a retrieval problem, text-to-SQL is still a retrieval problem, Graph RAG is still a retrieval problem — but each has its own failure modes and its own eval loop that the general chapters did not cover.

## The five surfaces this module covers

Each of the following gets its own chapter or chapter pair:

1. **Permission-aware retrieval mechanics** (chapter 2). Three places to enforce ACLs — query-time filter, index-time filter, per-tenant index isolation — with distinct correctness / performance / freshness trade-offs and a distinct *shape* of failure when you get them wrong.
2. **The ACL model shapes enterprise corpora inherit** (chapter 3). SharePoint's inheritance-plus-break-inheritance model. Google Drive's parents-plus-shortcuts model. Salesforce's OWD-plus-role-hierarchy-plus-sharing-rules-plus-territory-plus-manual-shares stack. Each is what your retrieval layer looks like after ingestion, whether you designed it or not.
3. **Text-to-SQL retrieval** (chapter 4). The structured slice of the corpus. Schema-linking, few-shot generation with LlamaIndex `NLSQLTableQueryEngine` or LangChain's SQL agent, query-safety guards (SQL injection isn't the risk; row-level data exfiltration and 90-minute joins are), and execution-accuracy eval on Spider and BIRD.
4. **Graph RAG on a property graph** (chapter 5). The relational slice. Neo4j / Kùzu / Memgraph as the graph store. LlamaIndex's Property Graph Index. Microsoft's GraphRAG pipeline. When multi-hop graph structure lifts recall and when it doesn't (spoiler — most questions are one-hop; know why the ones that aren't matter).
5. **Multimodal retrieval survey** (chapter 6). Image, document-page, audio, and video retrieval. CLIP and SigLIP for image-text. ColPali and DSE for document-page retrieval (the page *is* the image; the vision encoder eats the whole PDF page instead of parsing it). Audio via Whisper-transcribe-then-text-embed or CLAP. Video-clip retrieval as the frontier. Survey depth — you know when to reach for each, not necessarily how to fine-tune each.

Chapter 7 composes them behind a **router** — the practical shape of an enterprise stack that has all four (vector, lexical, SQL, graph) live at once. Chapter 8 is the boundary to `ai-risk-engineer` (content safety, PII redaction *methodology*) and `ai-infra-senior-engineer` (cross-tenant isolation at platform scale) so this module stays focused on ACL *mechanics* and doesn't drift into governance-family or platform-family scope.

## Two failures that motivate the discipline

### Failure 1: the retriever leaked a document across a permission boundary

The classic form: a user in tenant B receives a top-1 hit from tenant A because the ANN filter was applied *after* top-k, so a highly-similar tenant-A chunk crowded the tenant-B result set out and the post-filter left the response empty — except when a bug in the filter path silently returned the pre-filter set. Or the SharePoint sync fired *before* the "break inheritance" event was replicated, so a doc that had been re-permissioned to a restricted list was still indexed under the parent library's ACL for the next 4 hours. Or the Salesforce sharing recompute lags a role-hierarchy edit by a batch cycle, so a demoted user could still see rows in the retriever after the org chart said they couldn't.

Chapter 2 covers the mechanics; chapter 3 covers the specific enterprise sources whose ACL model you inherit. **The eval loop that catches this is an ACL fuzz test, not a recall@k number.** A recall metric averaged across a judgment set will *not* show you that user B occasionally sees tenant A's data; you need per-user, per-doc pass/fail on adversarial cases.

### Failure 2: the retriever answered a "which" question with a semantic search when the answer was a `SELECT`

The user asks "how many open P1 tickets does the support queue have today for tier-1 customers." The retriever runs a vector search over ticket text and returns the three tickets that talk *about* P1 and tier-1 most similar to the query, produces a chunk-based answer, and reports "three." The real answer is 47. The retriever should have written a SQL query against the tickets table, filtered on `status='open' AND priority='P1' AND customer.tier=1 AND created_at::date=CURRENT_DATE`, and returned the count.

Chapter 4 is the text-to-SQL slice. Chapter 7's router is how you decide *which* retriever to call in the first place. The eval loop is execution accuracy — did the generated SQL, when actually executed, return the same rows as the reference SQL? — evaluated on datasets like Spider or BIRD, or on a held-out set of your own labeled questions.

## Where this module sits on the ladder

This module is deliberately *retrieval-mechanics* only. Two peer boundaries define what it does *not* cover, and they matter enough to state up front:

- **`ai-risk-engineer` (peer, level 25, Governance family)** owns content-safety and responsible-AI evaluation methodology — guardrail effectiveness scoring, PII-redaction methodology and metrics, prompt-injection defense evaluation. This module teaches you how to *filter* a retriever by ACL and how to *isolate* tenants; it does not teach you how to design a PII-detection pipeline or how to score its precision / recall. When you find yourself asking "what is the false-negative rate of my PII scrubber," you have crossed into `ai-risk-engineer`.
- **`ai-infra-senior-engineer` (level 40, Infra family)** owns cross-tenant isolation at platform scale — dedicated hardware per tenant, VPC-per-tenant, encryption-key-per-tenant, physical shard-per-tenant, cross-region replication of per-tenant indexes. This module teaches you the three ACL enforcement points and the three tenancy models; it does not teach you how to run 5,000 per-tenant Qdrant collections in a Kubernetes cluster with per-tenant SLOs. When you find yourself sizing per-tenant physical shards on GPUs, you have crossed into `ai-infra-senior-engineer`.

Chapter 8 is the explicit contract with both peers. Every chapter in this module leans on those boundaries — if you catch yourself writing a whole subsection on how to build a PII detector, delete it and cite the peer.

## What this module assumes you have

- **A working retrieval stack** end-to-end from modules 101–111 — an ingestion pipeline (mod-102), an embedding pass and a vector index (mod-103 / mod-104), a lexical baseline (mod-105), a fusion + rewriting layer (mod-106), a reranker (mod-107), a citation-aware prompt (mod-108), a judgment-set eval (mod-109), an orchestration layer with an MCP-exposed retriever (mod-110), and the MLOps scaffolding to version and monitor it (mod-111). You are extending, not restarting.
- **Comfort with SQL.** Chapter 4's text-to-SQL work requires you to read and write moderate SQL — CTEs, window functions, joins across three tables. If SQL is unfamiliar, spend an hour with the [PostgreSQL tutorial](https://www.postgresql.org/docs/current/tutorial.html) before starting chapter 4.
- **Comfort with graph query languages.** Chapter 5 uses Cypher (Neo4j / Memgraph) as the reference. You do not need to be an expert; the [Neo4j Cypher manual — introduction](https://neo4j.com/docs/cypher-manual/current/introduction/) is enough. Kùzu and Memgraph both accept Cypher; Kùzu also accepts a variant.
- **A Docker environment** for the chapter-5 and exercise-3 Neo4j work, and enough disk / RAM to load a small property graph (~100 K nodes, ~500 K relationships).

## What this module does *not* teach

- **PII detection and redaction methodology.** `ai-risk-engineer` (peer, level 25). Presidio, Comprehend PII, Cloud DLP as tools; false-positive/false-negative rates and per-entity precision/recall as methodology; adversarial input evaluation. Referenced here only where the retrieval-side hook (redact-before-embed, redact-before-return) plugs in.
- **The internals of the ACL systems themselves.** SharePoint's role provider, Salesforce's sharing engine, Google Drive's IAM — how they work *inside* is Microsoft's / Salesforce's / Google's problem and their docs. This module teaches the *shape you inherit* after ingesting from each.
- **How to write good SQL for analytics.** Chapter 4 is retrieval-oriented text-to-SQL — narrow, question-answering SQL. Warehouse-scale analytics, dbt models, materialized aggregates, incremental refresh — all data-engineering territory. Cited when a text-to-SQL query would benefit from hitting a pre-aggregated table instead of a raw fact table.
- **How to build a knowledge graph from scratch.** Chapter 5 assumes the graph exists or is being extracted by GraphRAG's own extraction step. Ontology design, entity resolution at web scale, schema.org modeling — orthogonal disciplines.
- **Fine-tuning CLIP / SigLIP / ColPali.** Chapter 6 is a survey; you learn where they sit, how to *use* the pretrained checkpoints, and what the eval loop is. Domain adaptation of vision encoders is `fine-tuning-engineer`'s territory.
- **Platform-scale multi-tenancy.** Per-tenant index isolation *concepts* are in chapter 2; running 5,000 of them is `ai-infra-senior-engineer`'s job (chapter 8's boundary).

## The rest of the module

- **Chapter 2** — the three ACL enforcement points (query-time filter, index-time filter, per-tenant index isolation), the correctness / cost / freshness trade-offs of each, and a decision matrix.
- **Chapter 3** — the ACL model shapes you inherit from SharePoint, Google Drive, and Salesforce, and why each specific system has a specific class of failure your ingestion has to handle.
- **Chapter 4** — text-to-SQL retrieval: schema-linking, few-shot SQL generation with LlamaIndex `NLSQLTableQueryEngine` and LangChain's SQL agent, query-safety guards, and execution-accuracy eval on Spider / BIRD.
- **Chapter 5** — Graph RAG on a property graph: Neo4j / Kùzu / Memgraph, LlamaIndex's Property Graph Index, Microsoft GraphRAG, and the multi-hop question class where graph structure actually lifts recall.
- **Chapter 6** — multimodal retrieval survey: CLIP / SigLIP for image-text, ColPali / DSE for document-page retrieval, audio via Whisper-transcribe or CLAP, and video-clip retrieval as the frontier.
- **Chapter 7** — the hybrid enterprise router: composing vector + lexical + SQL + graph retrievers behind an intent-classification / LLM-router / rule-based dispatch layer, with a policy contract that carries the caller's ACL through every branch.
- **Chapter 8** — boundaries to `ai-risk-engineer` (content safety, PII redaction methodology) and `ai-infra-senior-engineer` (cross-tenant isolation at platform scale).

## What you should be able to do after this chapter

- Name the five surfaces this module covers (permission-aware retrieval, enterprise ACL shapes, text-to-SQL, Graph RAG, multimodal) and the chapter that owns each.
- State the two motivating failures (cross-tenant leak, wrong-retriever-for-question-shape) and which chapters are the defense against each.
- Draw the boundary to `ai-risk-engineer` (PII / content-safety methodology) and to `ai-infra-senior-engineer` (platform-scale isolation) and name at least one thing this module hands to each.
- Explain why an average recall@k number will not catch a cross-tenant leak, and what the eval shape (per-user adversarial pass/fail) has to look like instead.

## Further reading

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) — the canonical framing for the whole track; this module extends it to non-text, permissioned, and structured retrieval.
- Microsoft Learn: [*Overview of managed metadata and permissions in SharePoint*](https://learn.microsoft.com/en-us/sharepoint/governance/plan-permissions-strategy-sharing-and-collaboration) — the vendor's own permissions overview; chapter 3 unpacks the corners of the inheritance model this document walks past.
- Salesforce Architects: [*Record-Level Security*](https://architect.salesforce.com/decision-guides/data-sharing) — the OWD-plus-role-hierarchy-plus-sharing-rules-plus-territory model that chapter 3 covers; skim it now to see the shape of what your retriever inherits.
- Google Workspace Admin Help: [*How Drive sharing works*](https://support.google.com/a/answer/60781) — the parents-plus-shortcuts model chapter 3 unpacks.
- Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., Ma, J., Li, I., Yao, Q., Roman, S., Zhang, Z., & Radev, D. (2018). *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.* EMNLP. [Website](https://yale-lily.github.io/spider) — chapter 4's reference eval dataset; this module refers to it enough that having the leaderboard bookmarked is useful.
- Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Microsoft Research. [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) — the Microsoft GraphRAG paper chapter 5 uses as its reference implementation for the community-summary variant of Graph RAG.
- Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., & Colombo, P. (2024). *ColPali: Efficient Document Retrieval with Vision Language Models.* [arXiv:2407.01449](https://arxiv.org/abs/2407.01449) — the ColPali paper chapter 6 uses as its reference for document-page retrieval.
