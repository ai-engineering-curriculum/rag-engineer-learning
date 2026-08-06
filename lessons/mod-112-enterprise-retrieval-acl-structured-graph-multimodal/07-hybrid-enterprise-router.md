# Chapter 7 — Hybrid Enterprise Stacks: Routing Across Vector, Lexical, SQL, and Graph

## Why this chapter exists

Chapters 4, 5, and 6 each argued the same thing: their retriever is the right tool for a specific *shape* of question and the wrong tool for others. Text-to-SQL wins on "how many," Graph RAG wins on multi-hop traversals, ColPali wins on visually-rich documents, vector + lexical wins on everything else. Real enterprise stacks have all of them installed at once because their users ask all of those question shapes.

The retriever component that decides which of the four to call is a **router**. This chapter is about designing that router — its dispatch strategies, the policy contract it must honor across every branch (ACLs, tenancy, freshness), and the evaluation loop that keeps it honest.

The router is the "orchestration" retrieval-engineers own; agent-side orchestration that *consumes* the router as a tool is `agentic-ai-engineer`'s (level 40) concern. mod-110 built the MCP exposure of an individual retriever; this chapter is the layer that composes several MCP-exposed retrievers behind a single query surface.

## The three dispatch strategies

Routers differ in *how* they decide. Three shapes, roughly ordered by increasing flexibility and decreasing predictability:

### 1. Rule-based dispatch

The dispatch is deterministic — a set of hand-written rules maps query features to retrievers. Common rules:

- Query contains a numeric aggregate keyword (`how many`, `count`, `total`, `sum`, `average`) → text-to-SQL.
- Query contains a filter phrase that maps to a structured column (`tier-1 customers`, `open tickets`, `EMEA region`) → text-to-SQL.
- Query mentions a specific entity plus a relational verb (`reports to`, `part of`, `supplies`, `owns`) → Graph RAG.
- Query includes a filename or asks about visual content (`figure 3`, `chart on page 5`) → ColPali.
- Everything else → vector + lexical hybrid.

Trade-offs:

- **Pro:** predictable, cheap (regex or classifier), easy to debug.
- **Con:** brittle. Rules degrade as the corpus and query distribution evolve. Coverage gaps → misrouted queries → wrong answers.

This is the right default for small, well-scoped retrievers where the query distribution is stable and the team is small.

### 2. Classifier-based dispatch

A small classifier — a fine-tuned encoder head, a logistic regression on hand-featurized queries, an embedding-nearest-cluster lookup — predicts which retriever(s) to call. Training data: labeled `(query, retriever_that_should_answer)` pairs, either hand-labeled or bootstrapped from LLM-labeled examples.

The [semantic-router](https://github.com/aurelio-labs/semantic-router) library implements the embedding-nearest-cluster variant: each retriever has a set of "utterance" examples, queries are embedded and matched to the nearest cluster.

Trade-offs:

- **Pro:** learns the boundaries the rules would need many special cases to express. Cheap at inference (embedding + argmax).
- **Con:** training data is the bottleneck. Requires ongoing curation as the query distribution shifts. Classification errors are opaque (why did this go to text-to-SQL and not vector?).

This is the right choice when you have a corpus of past queries with known-good retriever labels, or you can bootstrap them cheaply.

### 3. LLM-router dispatch

The LLM gets a description of each retriever and picks. LlamaIndex's [`RouterQueryEngine`](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/) and LangChain's [multi-retriever router](https://python.langchain.com/docs/how_to/routing/) both implement this. The LLM sees:

```
You have the following retrievers available:
- vector: answers questions about ticket content, incident notes, PDFs.
- sql: answers count/sum/aggregate questions over the tickets and customers tables.
- graph: answers multi-hop relational questions (X→Y→Z through the org chart or supply chain).
- colpali: answers questions about figures, tables, or visual layout in the finance PDFs.

Query: <user question>

Which retriever(s) should answer this? Reply as JSON: {"retrievers": ["..."], "reason": "..."}
```

Trade-offs:

- **Pro:** most flexible. Handles novel query shapes without retraining. Can pick *multiple* retrievers and merge.
- **Con:** LLM latency + cost per query. Nondeterministic (small prompt changes shift the routing distribution). Debug-unfriendly ("why did the LLM route to vector?" is a prompt-forensics problem).

This is the right choice when the query distribution is broad and unstable, and cost / latency budgets allow.

### Hybrid dispatch

Nothing prevents you from stacking these — rules first for the confident cases (an obvious `SELECT COUNT` question), classifier for the medium cases, LLM router as a fallback. Or the reverse — LLM router by default, with an override table for known problematic queries. Real production stacks do stack them; the layers help with predictability where you have it and adaptability where you don't.

## Fan-out vs. exclusive routing

Two more dimensions cross-cut the strategy:

- **Exclusive routing** — pick one retriever, use only that one. Cheapest, most predictable.
- **Fan-out with fusion** — call multiple retrievers in parallel, merge with RRF or a weighted-sum (mod-106). More expensive; higher recall on ambiguous queries.

The right choice depends on the query. "How many open P1 tickets" is squarely text-to-SQL; a fan-out to vector adds noise. "What have we said publicly about the P1 incident from last quarter" is ambiguous — vector on the text corpus + SQL on the ticket table + graph on the incident-response chain could each contribute a useful chunk. Route intelligently: exclusive when confident, fan-out when not.

## The policy contract every retriever must honor

This is the most important part of the chapter — the piece that keeps routing from becoming a permission-leak accelerator.

The router is a single query-surface for the caller. Behind it, N retrievers may exist against N different backends (a vector DB, a warehouse, a graph DB, a document-page index). Each of those backends has its own permission enforcement. The router's job is to carry the caller's *policy context* through to every retriever, unchanged, and to verify that every result honored it.

The policy context — the "who is calling and what can they see":

```json
{
  "tenant_id": "acme",
  "user_id": "u_josh",
  "principals": ["u_josh", "grp_eng", "grp_project_alpha", "role_ic"],
  "clearance": "internal",
  "region": "us-east",
  "auth_freshness_seconds": 42,
  "request_id": "req-abc-123"
}
```

Every retriever the router calls must:

1. **Receive the policy context** as a first-class argument, not a header or an ambient global.
2. **Enforce it against its own backend** — text-to-SQL restricts the DB user's grants and adds row-level filters; graph retrieval adds `WHERE` clauses on node labels; vector search applies the query-time filter from chapter 2.
3. **Return results tagged with their effective ACL** so the router can do a post-check.

The router:

1. **Refuses to call any retriever without the policy context.** No exceptions. A retriever that doesn't accept a policy context does not go behind this router.
2. **Post-checks every returned chunk** against the policy context (chapter 2's belt-and-braces pattern). A chunk whose effective ACL doesn't include the caller's principals is dropped; the drop is logged as an alert (something misfired).
3. **Merges results across retrievers** with awareness of tenancy — fan-out to per-tenant collections; never leak across.
4. **Logs the routing decision, the retrievers called, the policy context, and every drop** so an incident can reconstruct exactly what happened.

Without this contract, adding a graph retriever silently exposes tenant-A data to tenant-B queries because the graph traversal doesn't honor the ACL that the vector retriever does. Chapter 2's mechanics are per-retriever; the router's contract is per-*ensemble*.

## A concrete router shape

Pseudocode for a policy-honoring, LLM-routed, fan-out-capable retriever:

```python
class EnterpriseRouter:
    def __init__(self, retrievers, dispatcher, post_check):
        self.retrievers = retrievers  # {"vector": VectorRetriever, "sql": SQLRetriever, ...}
        self.dispatch = dispatcher    # rule/classifier/LLM
        self.post_check = post_check  # ACL post-verification

    def query(self, question, policy_ctx):
        if not policy_ctx:
            raise PolicyMissing("router refuses to serve without policy context")

        # 1. Route
        decision = self.dispatch(question, policy_ctx)
        # decision -> {"retrievers": ["vector", "sql"], "reason": "..."}
        log_routing(question, decision, policy_ctx)

        # 2. Call each retriever in parallel with the policy context
        results = {}
        for name in decision["retrievers"]:
            r = self.retrievers[name]
            try:
                results[name] = r.retrieve(question, policy_ctx=policy_ctx)
            except Exception as e:
                log_retriever_error(name, e, policy_ctx)
                results[name] = []

        # 3. Post-check every chunk against the policy context
        checked = {}
        for name, chunks in results.items():
            checked[name] = []
            for chunk in chunks:
                if self.post_check(chunk, policy_ctx):
                    checked[name].append(chunk)
                else:
                    log_acl_drop(chunk, policy_ctx, name)

        # 4. Merge (RRF or dispatcher-suggested weighting)
        merged = reciprocal_rank_fusion(checked)
        return merged
```

The three log lines — `log_routing`, `log_acl_drop`, `log_retriever_error` — are the incident-response scaffolding. An audit that samples "was this chunk allowed to be returned to this user" reads from those logs.

## Evaluating a router

Router quality has three orthogonal axes:

### 1. Routing accuracy

Given a labeled set of `(query, correct_retriever)` pairs, what fraction does the router route correctly?

- Hand-label ~100–300 queries with the retriever that ought to answer each.
- Score exclusive-choice accuracy for exclusive-routing dispatch.
- For fan-out dispatch, score whether the correct retriever(s) were *in* the fanout.
- Watch class imbalance — if 80% of your queries are vector, a "route to vector always" baseline scores 80% and hides the router's actual value.

### 2. End-to-end retrieval quality

Given a labeled `(query, gold_answer)` set, does the router-plus-retriever pipeline produce a better retrieved set than each retriever alone?

- Compare `router → merged results` vs. `vector-only`, `sql-only`, `graph-only`, `fanout-then-fusion (no router)` on recall@k / MRR / nDCG.
- The router should beat every individual retriever on the *heterogeneous* judgment set (that mixes question shapes). If it doesn't, dispatch is wrong or the merge is wrong.

### 3. Policy compliance

The chapter-2 fuzz test extended: run the adversarial ACL fuzz through the router against every retriever; assert leak rate is exactly zero across all branches; assert the post-check catches any leak the primary filter missed.

Wire all three into the mod-111 retriever CI as gates. A PR that regresses routing accuracy, end-to-end recall, or leak rate cannot merge.

## Two operational patterns worth naming

### Retriever budgeting per query

The router should have a *budget* — a wall-clock cap and a cost cap — that limits how many retrievers it fans out to per query. If the LLM router picks all four, and the query fans out to all four, and each takes 300ms, you've added 300ms of parallel latency plus 4× the cost. Cap fan-out at 2, use the LLM's ranking to pick the two, log the ones it dropped for later analysis.

### Retriever health as a routing input

If the SQL warehouse is under load and returning 30-second query times, the router should degrade — either short-circuit to vector-only for the affected queries or explicitly return "the structured retriever is degraded; here's a best-effort text answer." Health signals from mod-111's monitoring feed the router directly.

## Where this chapter's boundary sits

- **The agent-side orchestration** that consumes this router as an MCP tool is `agentic-ai-engineer` (level 40). This chapter builds the router *as* a retriever surface; the agent that reasons about when to call it, how to compose it with other tools, and how to loop on unsatisfying results is that peer's concern.
- **The classifier training** for classifier-dispatch belongs partially in `applied-ai-engineer` (peer, level 30) if the classifier is a small BERT-style model, or in `fine-tuning-engineer` (peer, level 30) if it's a fine-tuned encoder. This chapter uses those artifacts; producing them at scale is not the retrieval-engineer's core skill.
- **The MCP exposure** of the router is mod-110's territory; wrap the router in an MCP tool schema whose input schema includes the policy context and whose output schema is retrieved-chunks-plus-metadata.

## What you should be able to do after this chapter

- Enumerate the three dispatch strategies (rule-based, classifier-based, LLM-router) and name a trade-off for each.
- Distinguish exclusive routing from fan-out-with-fusion and pick the right one per query shape.
- Author a policy contract for the router — the fields in the policy context, the requirements each retriever must honor, the post-check the router runs.
- Explain why adding a graph retriever without the policy contract is a permission-leak accelerator.
- Design the three-axis router evaluation (routing accuracy, end-to-end quality, policy compliance) and wire each into the mod-111 CI.
- Name the two operational patterns (retriever budgeting, health-aware routing) and explain what they defend against.
- Cite the boundary to `agentic-ai-engineer` (agent orchestration consuming the router) and to `applied-ai-engineer` / `fine-tuning-engineer` (classifier / encoder training).

## Further reading

- LlamaIndex: [*Router Query Engine*](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/) — the reference for LLM-selector routing over multiple retrievers.
- LangChain: [*How to route between sub-chains*](https://python.langchain.com/docs/how_to/routing/) — LangChain's routing primitives.
- Aurelio Labs: [*semantic-router*](https://github.com/aurelio-labs/semantic-router) — an embedding-nearest-cluster router library that's a good reference for classifier-dispatch shapes.
- Anthropic Engineering: [*Building Effective Agents*](https://www.anthropic.com/engineering/building-effective-agents) — the "routing" pattern is one of five workflow patterns; this chapter's router is a retrieval-shaped instance.
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR. [PDF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — the fusion method mod-106 introduced; the router's merge step composes it across retrievers of different modalities.
- Model Context Protocol: [*MCP specification*](https://modelcontextprotocol.io/) — the tool interface mod-110 used to expose retrievers; a router is another MCP tool, or a collection thereof.
- OWASP: [*Broken Access Control*](https://owasp.org/Top10/A01_2021-Broken_Access_Control/) — the policy-contract discipline this chapter argues for is the retriever-router analog of the OWASP guidance for web APIs.
