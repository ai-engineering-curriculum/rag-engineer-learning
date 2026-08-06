# Chapter 9 — The exposure/consumption boundary

## Why this chapter exists

Every role on the ladder is defined as much by what it *doesn't* own as by what it does. The RAG & Retrieval Engineer role is a level-30 specialist; the next step up, [`agentic-ai-engineer`](https://github.com/ai-engineering-curriculum/agentic-ai-engineer-learning) at level 40, is the role that *consumes* the tools you build here. This chapter draws the line explicitly so that neither role over-reaches into the other.

If you internalise nothing else from this chapter, internalise this: **this module owns the tool. The next role owns the agent that calls the tool.** Everything below is the consequence of that.

## What this module owns

Under the "exposure" umbrella:

- **The tool schema.** Names, descriptions, argument types, defaults, validation, error taxonomy. The advertised contract.
- **The tool's implementation.** The retriever behind the tool: hybrid, rerank, per-tenant filtering, citation-carrying results.
- **The wire protocol.** MCP server (chapter 07) or vendor-native tool endpoint (chapter 08); auth on that transport; per-tenant scoping at the boundary.
- **Server-side observability of the tool.** Metrics per call — latency, k, hit rate, error rate — and a trace ID per call that lets an agent's log correlate to your logs.
- **The tool's contract versioning.** How you evolve `inputSchema` without breaking downstream agents. Connect to the versioning discipline in `mod-111-retrieval-mlops-versioning-cicd-drift`.

Under the "orchestration" umbrella:

- **Retrieval-shaped orchestration.** LlamaIndex `QueryEngine` compositions, LangChain LCEL retrieval chains, LangGraph retrieve-verify loops. All *inside* the retrieval component.
- **Retrieval-side loops.** Retrieve → verify → rewrite → retrieve. Sub-question decomposition. Multi-source escalation *within* the retriever. These are internal to the tool the agent sees as a single call.

## What the next role up (`agentic-ai-engineer`) owns

Under the "consumption" umbrella:

- **Tool-selection policy.** Given three tools — your retriever, a calculator, a web search — deciding which to call and when.
- **Multi-tool orchestration and planning.** Chaining tool calls to accomplish tasks that span more than retrieval.
- **Agent-side memory and scratchpad.** The agent's per-conversation state; how tool results are summarised into working memory; when to re-query.
- **Agent-side guardrails.** Rate-limit an agent's tool use, veto certain tool arguments, filter agent outputs.
- **Consumer-side observability.** Agent traces, planning quality, tool-selection accuracy, end-to-end task success rates.
- **Human-in-the-loop for agent decisions** — approval gates at agent decision points, not at retrieval decision points.
- **Prompt policy for the agent as a whole** — how the top-level system prompt describes the agent's mission, which tools are available in which contexts, how to handle refusals from tools.

## The seam

The seam between the two lives at the tool contract itself. The below table names, per axis, what belongs on each side.

| Axis | This module (exposure) | `agentic-ai-engineer` (consumption) |
|---|---|---|
| Tool schema | Author it | Consume it |
| Tool description | Write it | Read it |
| Tool implementation | Own end-to-end | Not their concern |
| Retrieval-side loop (retrieve-verify) | Inside the tool | Not visible to the agent |
| Agent-side loop (call tool → maybe call again → answer) | Not our concern | Own end-to-end |
| Per-tenant / per-user auth at the transport | Enforce | Present credentials |
| Metrics per tool call | Emit | Consume in agent traces |
| Rate limiting inside the tool | Enforce (defensive) | Enforce (policy) |
| Contract versioning | Publish | Track and pin |
| Deprecation of a tool | Announce, dual-run | Migrate their agent |

Notice the pattern: on any given row, the *symmetric* activity belongs to the other role.

## Why the split is worth defending

Two failure modes motivate the boundary.

**Failure mode 1: retrieval engineers drift into agent design.** You add "agent-like" features into the retriever — the retriever decides when to fall back to a web search, decides when to call an LLM to summarise results, decides when to escalate to a graph traversal. Now your "tool" is itself an agent. Testing it means running an LLM inside your tool, evaluating agent-like behaviour, and shipping tool changes on the same cadence as agent changes. Consumers now depend on your tool's decisions in ways they cannot inspect. Debugging spans an unclear boundary between "the tool did something surprising" and "the agent called it wrong."

**Failure mode 2: agent engineers drift into retrieval design.** The agent's system prompt starts to include "use `retrieve` with `k=3` unless the question is about compliance, in which case use `k=8` and filter by scope=compliance." The routing logic that belongs *inside* the retriever leaks into every agent that consumes it. Now each new agent re-implements the routing (badly), and improvements to the retriever require changes to the prompts of every downstream agent.

The boundary — clean tool schema, retrieval-side smarts *inside* the tool, agent-side smarts *outside* the tool — is what keeps both roles independently testable and upgradable.

## Practical rules for staying on your side

Six habits to internalise:

1. **A tool is a function. Model it as one.** Inputs, outputs, side effects, error taxonomy. Not a partner in a conversation; a callable.
2. **Do not import an agent runtime inside your retriever.** A retrieval tool that constructs a LangGraph agent inside its implementation has crossed the boundary. Refactor before shipping.
3. **Do not put agent-side instructions in your tool description.** The description should describe the tool. It should not say "call this last after you've tried the calculator." That is agent policy.
4. **Log by tool call, not by session.** Your logs should let a consumer correlate one of your calls to one of their agent's steps. Your logs should not try to reconstruct the whole agent conversation — you don't own it.
5. **Publish a changelog for the tool schema.** Every change to the schema — added optional field, deprecated field, tightened validation — is a note in a changelog consumers can subscribe to. `mod-111` deep-dives the discipline.
6. **Treat MCP clients as adversaries.** Assume the caller may misuse arguments, spam calls, or leak results to places you did not intend. Auth, rate limit, cap `k`, filter by ACL. This is the same discipline as any public API.

## Where this fits in the wider curriculum

- **Below (owned in this module):** everything up to and including a well-versioned, observable, permission-aware retrieval tool exposed as either MCP or vendor-native tool.
- **Above (owned by `agentic-ai-engineer`, level 40):** agent design consuming the tool — tool-selection, multi-step planning, agent memory, agent guardrails, agent traces, human-in-the-loop on agent decisions.
- **Sideways:**
  - `applied-ai-engineer-learning` (peer, level 30) — the application team that wraps the *product surface* around a retrieval-backed feature. They consume the tool differently from an agent — via a fixed application flow — but the tool contract is the same.
  - `ai-eval-engineer-learning` (peer, level 30) — the team that evaluates the agent-and-retriever together at the application layer (RAG triad, faithfulness). Retrieval-only evaluation stays in your `mod-109`; end-to-end evaluation of consumption crosses into their territory.

## A last recommendation

Once you have finished the module, do the fit-matrix exercise (`exercises/exercise-01-…`). It is deliberately placed at the end because the answers get sharper after you have built with each framework and both tool contracts. The exercise is where you decide, for your own reference, which framework you would reach for first — and where you have earned that opinion.

## Summary

- This module owns the *exposure* side: retrieval-shaped orchestration and the tool contract at the network boundary.
- `agentic-ai-engineer` (level 40) owns the *consumption* side: the agent that calls the tool.
- The boundary lives at the tool schema. Retrieval-side smarts belong inside the tool; agent-side smarts belong outside.
- Defend the boundary in both directions to keep the two roles independently testable.
- Rules: model tools as functions, log per call, version the schema, treat callers as adversaries, publish changelogs, don't cross the seam.

Reading the module chapters back-to-back, from framework-vs-bare through MCP to this chapter, gives you the full arc: a tuned retriever comes in one end, a well-versioned, observable, standards-based tool goes out the other. That is what a level-30 retrieval engineer ships.
