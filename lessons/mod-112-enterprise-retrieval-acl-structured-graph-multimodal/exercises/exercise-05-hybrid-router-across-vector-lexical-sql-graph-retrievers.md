# Exercise 05 — Hybrid Router Across Vector, Lexical, SQL, and Graph Retrievers

**Estimated effort:** ~2 hours. Python, the retrievers you built in exercises 01–03 (or lightweight substitutes if you skipped any), an LLM you can call, and a heterogeneous 30–60 question set that mixes the four query shapes.

## Objective

Compose four retrievers — vector, lexical (BM25), text-to-SQL, and Graph RAG — behind a router that honors a *policy contract* on every branch. Implement three dispatch strategies (rule-based, embedding-classifier, LLM-router), evaluate all three on routing accuracy, end-to-end retrieval quality, and policy compliance, and ship a router that beats every retriever alone on the heterogeneous set.

By the end you have:

- `router/policy.py` — the `PolicyContext` dataclass and the post-check that every returned chunk must survive.
- `router/dispatchers/{rule,classifier,llm}.py` — three routing strategies exposing the same `dispatch(question, policy_ctx) -> {retrievers, reason}` interface.
- `router/router.py` — the orchestrator that calls the retrievers with the policy context, post-checks results, RRF-merges, and logs.
- `retrievers/registry.py` — bindings from retriever name to the actual implementations from exercises 01–03 (or the lightweight stubs below).
- `evals/routing_accuracy.py` — hand-labeled `(question, correct_retriever)` set; per-strategy accuracy.
- `evals/end_to_end_recall.py` — hand-labeled `(question, gold_docs)` set; recall@10 for router-vs-each-retriever-alone.
- `evals/policy_compliance.py` — the ACL fuzz test from exercise 01 extended through the router.
- `RESULTS.md` — the three-axis comparison table and a shipping recommendation.

## Prerequisites

- Chapters 2 and 7 read.
- Exercise 01 (permission-aware retriever) — the post-check pattern reuses that code.
- Exercises 02 and 03 (text-to-SQL, Graph RAG) — the retrievers this router composes. If you skipped either, the "Notes" section below gives a lightweight stub you can drop in.
- Python 3.10+; `pip install rank-bm25 llama-index openai` (plus whatever your exercise-02 and exercise-03 retrievers need).
- An LLM for the LLM-router dispatch. GPT-4o-mini / Claude Haiku / a local coder model are all fine.

## Steps

### Step 1 — Define the policy contract (10 minutes)

`router/policy.py`:

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class PolicyContext:
    tenant_id: str
    user_id: str
    principals: List[str]                 # union of user, roles, groups
    clearance: str = "internal"            # public | internal | confidential
    request_id: str = ""

def post_check(chunk_acl, policy_ctx: PolicyContext) -> bool:
    principals = set(policy_ctx.principals)
    if principals & set(chunk_acl.get("denied_users", []) + chunk_acl.get("denied_groups", [])):
        return False
    allowed = set(chunk_acl.get("allowed_users", []) + chunk_acl.get("allowed_groups", []))
    return bool(principals & allowed) and chunk_acl.get("tenant_id") == policy_ctx.tenant_id
```

This is the belt-and-braces backstop from chapter 2, hoisted to the router layer so every retriever's output is re-verified regardless of what its primary filter did.

### Step 2 — Wire the retriever registry (15 minutes)

`retrievers/registry.py` — bind names to callables that all accept `(question, policy_ctx)` and return `List[Chunk]` where a `Chunk` has `id`, `text`, `score`, `acl`, and any retriever-specific metadata (`sql_query`, `cypher`, `page_id`, etc.).

```python
RETRIEVERS = {
    "vector":  vector_retriever,   # from exercise 01's query-time filter build
    "lexical": bm25_retriever,     # BM25Okapi over the same corpus
    "sql":     sql_retriever,      # from exercise 02, wrapped so it returns rows-as-chunks
    "graph":   graph_retriever,    # from exercise 03's PropertyGraphIndex retriever
}
```

Every retriever must:

1. Accept `policy_ctx` as a first-class argument.
2. Return chunks whose `acl` field is populated so the post-check can re-verify.
3. Enforce its own primary filter (query-time metadata filter, DB-user grants, Cypher WHERE clause on `tenant_id`) — the post-check is defense in depth, not a replacement.

### Step 3 — Rule-based dispatcher (15 minutes)

`router/dispatchers/rule.py`:

```python
import re

SQL_TRIGGERS = re.compile(r"\b(how many|count|total|sum|average|per|by\s+\w+|top\s+\d+)\b", re.I)
GRAPH_TRIGGERS = re.compile(r"\b(reports to|manages|works (at|for)|supplies|owns|connected)\b", re.I)
LEXICAL_TRIGGERS = re.compile(r'"[^"]+"|\b(error code|ticket #|CVE-\d+)\b')

def dispatch(question, policy_ctx):
    hits = []
    if SQL_TRIGGERS.search(question): hits.append("sql")
    if GRAPH_TRIGGERS.search(question): hits.append("graph")
    if LEXICAL_TRIGGERS.search(question): hits.append("lexical")
    if not hits: hits = ["vector"]
    if len(hits) == 1 and hits[0] != "sql":
        hits.append("vector")   # vector is a cheap sanity-check on non-SQL branches
    return {"retrievers": hits[:2], "reason": "rule-based"}
```

Cheap, predictable, transparent. Deliberately imperfect — the eval will show it.

### Step 4 — Classifier dispatcher (20 minutes)

`router/dispatchers/classifier.py`: use the embedding-nearest-cluster shape from [semantic-router](https://github.com/aurelio-labs/semantic-router), or write a minimal version.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

EMB = SentenceTransformer("all-MiniLM-L6-v2")

UTTERANCES = {
    "vector":  ["what does the doc say about X", "explain how X works", "summarize X"],
    "lexical": ['find "exact phrase"', "search for ticket #4521", "grep for CVE"],
    "sql":     ["how many open tickets", "count customers by tier", "sum of revenue last week"],
    "graph":   ["who reports to X", "who at X works with Y", "supply chain from X to Y"],
}

CENTROIDS = {name: EMB.encode(examples).mean(axis=0) for name, examples in UTTERANCES.items()}

def dispatch(question, policy_ctx):
    q = EMB.encode(question)
    scored = {n: float(np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c))) for n, c in CENTROIDS.items()}
    top = sorted(scored, key=scored.get, reverse=True)
    return {"retrievers": top[:2], "reason": f"classifier scores={scored}"}
```

For a small utterance set this is a toy classifier; it's still usually better than pure regex on ambiguous queries. Chapter 7 discusses where you'd graduate to a fine-tuned encoder or a small BERT head — out of scope here; the shape is what matters.

### Step 5 — LLM-router dispatcher (15 minutes)

`router/dispatchers/llm.py`: the reference pattern from [LlamaIndex `RouterQueryEngine`](https://docs.llamaindex.ai/en/stable/module_guides/querying/router/) shape.

```python
import json
from openai import OpenAI

llm = OpenAI()

PROMPT = """You have the following retrievers available:
- vector: semantic search over unstructured document text (wiki pages, tickets, PDFs).
- lexical: exact / phrase / code-token search (BM25) — good for identifiers, quoted phrases, error codes.
- sql: answers count/sum/aggregate questions over the tickets, customers, orders, products tables.
- graph: answers relational questions (X reports to Y; who works with Z; supplier chains).

Question: {question}

Reply as JSON with keys "retrievers" (list of 1-2 names from the list above) and "reason" (one sentence)."""

def dispatch(question, policy_ctx):
    resp = llm.chat.completions.create(
        model="gpt-4o-mini", temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT.format(question=question)}],
    )
    return json.loads(resp.choices[0].message.content)
```

Slowest and most expensive of the three; the eval will show where it earns that cost.

### Step 6 — The router (20 minutes)

`router/router.py`:

```python
from concurrent.futures import ThreadPoolExecutor

class EnterpriseRouter:
    def __init__(self, retrievers, dispatcher, post_check):
        self.retrievers = retrievers
        self.dispatch = dispatcher
        self.post_check = post_check

    def query(self, question, policy_ctx, top_k=10):
        if not policy_ctx: raise ValueError("router refuses to serve without policy context")
        decision = self.dispatch(question, policy_ctx)
        log_routing(question, decision, policy_ctx)

        with ThreadPoolExecutor(max_workers=len(decision["retrievers"])) as ex:
            futures = {name: ex.submit(self.retrievers[name], question, policy_ctx)
                       for name in decision["retrievers"] if name in self.retrievers}
            results = {name: f.result() for name, f in futures.items()}

        # post-check every chunk, drop failures loudly
        checked = {}
        for name, chunks in results.items():
            checked[name] = []
            for chunk in chunks:
                if self.post_check(chunk["acl"], policy_ctx):
                    checked[name].append(chunk)
                else:
                    log_acl_drop(chunk, policy_ctx, name)

        return reciprocal_rank_fusion(checked, top_k=top_k)
```

Reuse the RRF from mod-106 — a stock `1 / (60 + rank)` weighted-sum across the checked result lists.

### Step 7 — Author the heterogeneous evals (25 minutes)

Two hand-labeled sets — same 30–60 questions in each.

`evals/routing_labels.jsonl` — `(question, correct_retrievers)`, where `correct_retrievers` is a set of the retrievers that *should* fire (permits multi-answer, since fan-out is a valid choice on ambiguous questions):

```json
{"id": "q001", "question": "how many open P1 tickets from tier-1 customers?", "correct_retrievers": ["sql"]}
{"id": "q002", "question": "what did we say publicly about the Q3 incident?", "correct_retrievers": ["vector", "lexical"]}
{"id": "q003", "question": "who does the RapidWidget product owner report to?", "correct_retrievers": ["graph"]}
```

Aim for ~10 questions per retriever "home" plus ~10 genuinely ambiguous ones. Cover the four question shapes from the chapter (aggregate/filter, semantic search, exact/phrase, multi-hop) plus a couple of adversarial ones (a question that looks like it wants SQL but the interesting answer is in prose).

`evals/gold_docs.jsonl` — same question IDs, with the gold answer document/row IDs. This is what recall@10 is computed against.

Optionally, extend `evals/fuzz.jsonl` from exercise 01 with the four-retriever plane so `evals/policy_compliance.py` can assert no branch leaks.

### Step 8 — Score routing accuracy (10 minutes)

`evals/routing_accuracy.py`:

For each dispatcher × question, compare `dispatch(question).retrievers` against `correct_retrievers`. Report:

- **Exact-match rate** — fraction where the set is exactly equal.
- **Recall of correct retrievers in fanout** — for each question, `|dispatched ∩ correct| / |correct|`.
- **Precision of dispatched retrievers** — `|dispatched ∩ correct| / |dispatched|`.

The three-strategy table already tells a story: rule-based is fast and brittle, classifier is a middle ground, LLM-router is flexible but expensive and nondeterministic.

### Step 9 — Score end-to-end recall (10 minutes)

`evals/end_to_end_recall.py`: for each dispatcher × question, run through the full router (dispatch + call + post-check + merge) and compute `recall@10` against `gold_docs`.

Also run each retriever *alone* on the same set (no router) as baselines. The router should beat every retriever alone on the heterogeneous set; that is what a router is for.

### Step 10 — Score policy compliance (10 minutes)

`evals/policy_compliance.py`: reuse the fuzz test from exercise 01, applied against the router. For each `(user, question)` pair, assert every returned chunk survives `post_check(chunk.acl, policy_ctx)`. Leak rate should be exactly 0 across all three dispatchers.

Also plant a deliberate bug in *one* retriever's primary filter (say, the graph retriever's `WHERE tenant_id = $tid` clause) and verify the router's post-check catches every leak the primary filter misses — that is the value the post-check adds and the check must run.

Fix the bug afterward; keep the fuzz case as a regression guard.

### Step 11 — Write up `RESULTS.md` (10 minutes)

Publish:

- The three-axis table: `(dispatcher × {routing accuracy, end-to-end recall@10, leak rate, avg p50 latency, LLM tokens/query})`.
- One paragraph per dispatcher on when you'd ship it — small teams with stable query distributions vs. broad heterogeneous ones vs. cost/latency-sensitive ones.
- The bug you planted and the post-check catching it (one sentence + a log excerpt).
- The stacking recommendation from chapter 7: e.g., "rules first for the confident SQL/graph cases, LLM router as fallback, classifier optional if we have labeled data" — pick one based on your numbers.

## Acceptance criteria

- Three dispatchers all conform to the same `dispatch(question, policy_ctx) -> {retrievers, reason}` interface and are swappable in the router constructor.
- The router refuses to serve without a `policy_ctx` (raises), calls retrievers concurrently, post-checks every chunk, RRF-merges, and logs the routing decision, ACL drops, and retriever errors.
- The heterogeneous eval covers all four retrievers as "home" retriever for at least 5 questions each plus at least 10 ambiguous ones.
- Routing-accuracy, end-to-end-recall, and policy-compliance evals all run and are reported per dispatcher.
- The deliberate primary-filter bug is caught by the router's post-check; leak rate after fix is exactly 0 for all three dispatchers.
- `RESULTS.md` publishes the three-axis table and a per-dispatcher shipping recommendation.

## Stretch goals

- **Stack the dispatchers.** Implement chapter 7's hybrid: rules first for confident cases, classifier for medium, LLM as fallback. Compare against each alone.
- **Retriever budgeting.** Cap fanout at 2; if the LLM router picks more, take its top-2 by reason confidence. Measure the latency/quality trade-off vs. uncapped.
- **Health-aware routing.** Wire a fake "SQL warehouse is slow" signal; verify the router degrades to vector-only on SQL questions for the duration; verify it recovers.
- **Return the routing decision in the response.** Add `router_decision` to the response envelope so downstream systems can display / audit which retrievers fired. Useful for user-facing "here's how we found this" UI patterns.
- **A/B the dispatchers.** Ship the router with 90% traffic on one dispatcher and 10% on another (feature-flag pattern); score in production against a small labeled sample. This is chapter 7's shape at real ops.
- **Add a fifth retriever.** Wire the ColPali document-page retriever from exercise 04 as `pages`. Extend the dispatchers to recognize "figure 3 on page 5" style queries. Re-run the evals.
- **Refuse-to-answer branch.** For questions that clearly lack a matching retriever (out-of-scope, out-of-permission), have the LLM router return `{"retrievers": [], "reason": "..."}` and the router return a "no retriever can serve this" response. Measure how often it fires and whether the calls are correct.

## Notes

- **Lightweight stubs if you skipped an earlier exercise.** If you didn't build exercise 02 or 03, stub `sql` as a wrapper around `sqlite3` executing a hard-coded template, and stub `graph` as an in-memory dict `{entity: [related_entities]}`. The point is to exercise the *router* mechanics; the individual retrievers can be minimal.
- **The post-check is the point.** The router's real defense against retriever bugs is the belt-and-braces post-check, not the individual retrievers' primary filters. Wiring it in from step 6 makes the step-10 bug-plant meaningful.
- **RRF is a cross-retriever primitive.** The merge here composes results from retrievers of *different modalities* (rows, chunks, graph paths). Normalize to a common `(id, score)` shape before RRF; the mod-106 pattern transfers.
- **Log everything.** Route decisions, drops, retriever errors, latencies. Every incident-response story about a routed retriever ends at "we couldn't tell which retriever ran." Don't be that story.
- **Chapter 8's boundary applies.** The router is a retrieval surface; the agent that consumes it is `agentic-ai-engineer`. Do not extend the router into tool-selection loops, cross-tool memory, or multi-step planning — that is the peer's scope.
