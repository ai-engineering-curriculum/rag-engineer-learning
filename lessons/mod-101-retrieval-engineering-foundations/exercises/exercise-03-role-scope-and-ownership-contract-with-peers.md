# Exercise 03 — Write the retriever ownership contract for a specific team

**Estimated effort:** ~2 hours. Writing, not coding.

## Objective

You will draft a one-page **retriever ownership contract** for a plausible team that is standing up a retrieval-engineering function alongside existing product and evaluation teams. The contract makes explicit what your role owns, what it defers to peers and higher-level roles, the interface artifact your team ships, and the escalation triggers that pull a delegated party back in. Writing the contract is the exercise that keeps you from being pulled into product-facing prompt tuning at the expense of the retriever — or from silently reinventing what your peers already own.

## Prerequisites

- Chapter 3 (*Retrieval as a Feature vs. Retrieval as an Engineered System*).
- Chapter 4 (*Where a Retrieval Engineer Sits*), including the template at the end.
- The track's [`CURRICULUM.md`](../../../CURRICULUM.md) *Ownership rule* section — skim it once.

## The scenario

You are the first retrieval engineer at a mid-size B2B SaaS company. The relevant existing teams:

- **Product team A ("Ask Docs").** Two application developers. They shipped a first "ask our documentation" feature six months ago using LangChain, OpenAI embeddings, and pgvector top-*k*. It works well on the public-docs corpus but is not permission-aware. They want to reuse it on internal Confluence and Google Drive.
- **Product team B ("Support Copilot").** Three engineers. They are building a support-agent copilot that retrieves from the customer support ticket corpus (SFDC + internal notes) and rewrites customer replies. They want their agent to be able to call *your* retriever as a tool.
- **The AI Evaluation team.** Two engineers. They are the peer `ai-eval-engineer` shape — they own LangSmith / Langfuse tracing, RAGAS, DeepEval, and the RAG-triad measurement (faithfulness, answer relevance, context precision-recall) at the application layer.
- **The ML Platform / MLOps team.** Peer `ai-infra-mlops` shape. They own the model registry, the general CI/CD-for-ML pipelines, and the model-serving infra.
- **Legal / Compliance.** They will flag ACL requirements for the internal Confluence and SFDC corpora and will insist on per-tenant isolation once external customers are exposed.
- **You.** One retrieval engineer (this role). You have been hired to consolidate the two product teams' retrievers, add a permission-aware corpus, and expose the result to the Support Copilot agent as an MCP-compatible tool.

The corpora involved:

- Public product docs (~50k pages, HTML + Markdown, no permissions).
- Internal Confluence (~200k pages, HTML + attached PDFs, per-page ACL from Confluence groups).
- Internal Google Drive (~100k documents, mixed formats, per-file ACL from Google Groups).
- SFDC ticket corpus (~2M tickets, structured with a free-text description field, per-account ACL).

## Part A — Fill in the ownership-contract template (60 min)

Use the template from chapter 4 as your starting frame. Fill in:

1. **Owned by / Consumers.** Explicit team names from the scenario.
2. **The retriever this team ships.** Interface (MCP tool schema, HTTP endpoint, or both), quality claim (`recall@k` on judgment set), latency claim (p50 / p95 under a named workload), cost claim ($ per 1M queries), freshness claim (corpus lag SLO), permission claim (per-tenant isolation mechanism).

   You do *not* have the numbers yet — this exercise is written *before* mod-104 (index tuning) and mod-109 (eval). Put a placeholder such as `recall@10 = TBD, target ≥ 0.85 on judgment-set-v1` and note the module that will produce the number. This is a real design-doc pattern; do not skip it.
3. **Not this team's scope; delegated to.** For each capability the scenario surfaces (product-surface prompt shape, application-layer RAG-triad, ingestion-DAG serving infra, embedding fine-tuning, content-safety, agent orchestration, platform-scale serving), name the specific team from the scenario and the peer role from chapter 4 it maps to.
4. **Escalation triggers.** At least four. Each trigger should name the observation, the peer team it pulls in, and the artifact you would hand them (e.g. "a recall@k regression discovered in production → the AI Eval team, handed the RAG-triad instrumentation gap that hid it").

**Deliverable A:** `retriever-ownership-contract.md`, ≤ 1.5 pages of Markdown. Cannot be more than 1.5 pages — the discipline of fitting is part of the exercise.

## Part B — Write the three "not us" letters (30 min)

For each of the three capabilities most likely to be misattributed to your role, write a two-paragraph note you would send to the relevant party naming (a) why this is not your team's scope, (b) which module in your track does cover the adjacent capability that gives context, and (c) what you *would* deliver at the boundary.

Suggested capabilities to address:

1. **Product team A asks you to own the "answer with citations" prompt shape and the UI rendering of citations.** You are delegating this to the peer `applied-ai-engineer` role — write the boundary.
2. **Product team A also asks you to add faithfulness / answer-relevance measurement to the answers coming out of Ask Docs.** You are delegating this to the peer `ai-eval-engineer` role — write the boundary. Note that mod-109 in your track *does* cover retrieval-only eval, which you will run and hand them.
3. **The Support Copilot team asks you to design the agent loop that decides when to call the retriever twice vs. once.** You are delegating this to `agentic-ai-engineer` (level 40) — write the boundary. Note that mod-110 in your track exposes the retriever as an MCP tool the agent can call.

Keep each note under 250 words. The tone is *collaborative*, not defensive: you are naming the boundary, not refusing to help.

**Deliverable B:** three short notes in one file, `not-us-notes.md`.

## Part C — Draw the delegation diagram (30 min)

Draw a diagram (paper, tldraw, Excalidraw, or Mermaid) showing:

- Your team in the middle, ringed by the three product teams and the two peer teams from the scenario.
- Arrows *outward* to each peer showing what you deliver to them and what they own that you consume.
- Arrows *upward* to `agentic-ai-engineer` and — as a future item — `ai-infra-senior-engineer`.
- Arrows *downward* to the level-20 tracks you are consuming (`llm-application-developer`, `ml-engineer`, `ai-infra-junior-engineer`) as prerequisites.

Label each arrow with the *artifact* moving across it, not just the team name (e.g. "retriever MCP tool schema" going out, "judgment set + retrieval-only eval report" going out, "RAG-triad findings" coming in).

**Deliverable C:** the diagram image (or Mermaid source) as `delegation-diagram.png` / `.md`.

## Acceptance criteria

- Deliverable A includes all four sub-sections of the ownership-contract template with concrete names from the scenario. Every claim has either a placeholder value or a *link* to the module that will produce it.
- Deliverable B's three notes each name (a) the delegated team, (b) the module in your track that owns the adjacent capability, and (c) what you deliver at the boundary. Under 250 words each.
- Deliverable C's diagram labels every arrow with an artifact, not just a team name.

## Stretch goals

- Repeat the exercise for a *smaller* team — a two-person startup where you are also the applied-AI engineer and the eval engineer. Where do the boundaries collapse, and which parts of the contract remain load-bearing? Write a short reflection.
- Locate the current version of the ownership contract in your own workplace (or the closest analogue — team RFCs, on-call rotation docs, RACI matrix). Diff it against yours. Note one place where the real-world contract is *less* explicit than yours and one place where it is *more*.
