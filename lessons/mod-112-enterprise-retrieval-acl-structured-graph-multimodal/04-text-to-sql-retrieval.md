# Chapter 4 — Text-to-SQL Retrieval Over Structured Data

## Why this chapter exists

The corpus a real enterprise RAG serves is not all unstructured text. A meaningful fraction — sometimes the majority of the questions users actually ask — is answerable by a `SELECT` statement against a relational database: the CRM's opportunity table, the HRIS's employee table, the finance system's invoice table, the ticketing system's cases table. A vector-search retriever asked "how many open P1 tickets did we close last week for tier-1 customers" will return the three tickets that talk *about* P1 and tier-1 most similarly to the query, produce an answer grounded in those chunks, and be *entirely wrong*.

Text-to-SQL is the retrieval shape that answers those questions correctly: parse the natural-language question, generate a SQL query against a known schema, execute it, return the rows (or an aggregate over the rows) as the retrieved "chunks" the generator uses.

This chapter covers the retrieval-engineer's slice of text-to-SQL:

- **Schema-linking** — how to give the LLM enough schema context that it generates SQL against the *right* tables and columns without generating SQL against everything.
- **Few-shot SQL generation** — the LlamaIndex `NLSQLTableQueryEngine` and LangChain SQL agent as reference implementations, and what to control that they abstract over.
- **Query-safety guards** — the guards that keep a rogue query from taking down your warehouse or exfiltrating a whole table.
- **Execution-accuracy evaluation** — the Spider and BIRD datasets, execution-accuracy as the metric, and a domain-fit eval loop on your own schema.

Warehouse-scale analytics engineering, dbt models, incremental materialization — data-engineering territory. This module owns the retrieval shape (question → SQL → rows) that plugs into the rest of the RAG stack.

## The pipeline

```
    natural-language question
             │
             ▼
    ┌────────────────────────┐
    │  schema linker         │  select relevant tables + columns from a possibly-huge schema
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │  SQL generator (LLM)   │  emit a SQL query given the question + selected schema
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │  safety guard          │  reject / rewrite queries that fail policy
    └────────┬───────────────┘
             │
             ▼
    ┌────────────────────────┐
    │  execute against DB    │  run against a read-only, time-limited, row-capped connection
    └────────┬───────────────┘
             │
             ▼
    rows → format → hand to generator (or hand row-count / aggregate directly)
```

Each stage has its own failure modes; each has an eval hook.

## Schema-linking — the piece the naive tutorials skip

The obvious version: paste the whole schema in the prompt, ask the LLM to write SQL. This works for a toy schema of five tables. It fails at 500 tables, or 50 tables with 100 columns each, for three reasons:

1. **Prompt budget.** The schema does not fit.
2. **Attention dilution.** Even if it fits, the LLM's ability to identify the *right* columns degrades as irrelevant schema crowds the context (the mod-107 "lost in the middle" phenomenon transferred to schema selection).
3. **Ambiguity.** Two tables both have a `status` column; without semantic hints, the LLM picks the wrong one.

Schema-linking is the retrieval step *inside* text-to-SQL — retrieve the tables and columns relevant to the question before you ask the LLM to write SQL:

- **Embed each table + column comment** as a vector. For each column, embed a short natural-language description ("customer's tier level; 1=platinum, 2=gold, 3=silver") if you have it; the column name alone (`cust_t`) is a bad embedding target.
- **At query time**, embed the user's question and retrieve the top-N most similar table/column descriptions. This is a plain mod-104 vector search over your metadata catalog.
- **Assemble a compact schema snippet** — only the tables and columns the retrieval selected, plus their foreign-key relationships. Foreign keys matter because a question about "customers with open tickets" needs both tables *and* the join key.
- **Include enough sample values.** For low-cardinality columns (e.g., `status` ∈ {open, pending, closed}), include the distinct values in the prompt. Without them, the LLM invents plausible-but-wrong values (`'in-progress'`).

The BIRD benchmark's [supplementary materials](https://bird-bench.github.io/) show empirically that schema-linking quality dominates SQL generation quality on realistic schemas — a schema linker that returns the wrong tables poisons everything downstream.

A minimal schema-linking implementation:

```python
from sentence_transformers import SentenceTransformer
import chromadb

class SchemaLinker:
    def __init__(self, schema_desc, embed_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model)
        self.client = chromadb.Client()
        self.coll = self.client.create_collection("schema")
        for table in schema_desc:
            for col in table["columns"]:
                text = f"table {table['name']} column {col['name']}: {col['description']} (type: {col['type']})"
                self.coll.add(
                    ids=[f"{table['name']}.{col['name']}"],
                    embeddings=[self.model.encode(text).tolist()],
                    metadatas=[{"table": table["name"], "column": col["name"], "type": col["type"]}],
                    documents=[text],
                )

    def link(self, question, top_k_cols=20):
        q_emb = self.model.encode(question).tolist()
        hits = self.coll.query(query_embeddings=[q_emb], n_results=top_k_cols)
        tables = {m["table"] for m in hits["metadatas"][0]}
        return list(tables)  # then look up full schema for these tables
```

This is a naïve version — production systems (LlamaIndex's [`SQLTableRetrieverQueryEngine`](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/#part-3-sql-router-query-engine), LangChain's [SQLDatabaseToolkit](https://python.langchain.com/api_reference/community/agent_toolkits/langchain_community.agent_toolkits.sql.toolkit.SQLDatabaseToolkit.html)) do the same thing with more polish (they also embed foreign-key chains, sample values, cardinality). But the fundamental shape is: retrieve schema first, generate SQL second.

## Few-shot SQL generation with LlamaIndex and LangChain

### LlamaIndex `NLSQLTableQueryEngine`

The bare-metal shape from LlamaIndex's [SQL query-engine docs](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/):

```python
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg://readonly_user:pw@host/db")
sql_db = SQLDatabase(engine, include_tables=["customers", "tickets", "products"])
qe = NLSQLTableQueryEngine(sql_database=sql_db, tables=["customers", "tickets", "products"])
response = qe.query("how many open P1 tickets closed last week for tier-1 customers?")
print(response.response)          # natural-language answer
print(response.metadata["sql_query"])  # the generated SQL
```

For larger schemas, use `SQLTableRetrieverQueryEngine` — same shape but a schema retriever is composed in front:

```python
from llama_index.core.query_engine import SQLTableRetrieverQueryEngine
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from llama_index.core import VectorStoreIndex

table_node_mapping = SQLTableNodeMapping(sql_db)
table_schema_objs = [SQLTableSchema(table_name=t) for t in sql_db.get_usable_table_names()]
obj_index = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex)
qe = SQLTableRetrieverQueryEngine(sql_db, obj_index.as_retriever(similarity_top_k=3))
```

The retriever picks the top-3 tables per query; the SQL generator sees only those tables' schemas.

### LangChain SQL agent

LangChain's shape is agentic — it exposes the DB as a set of tools (`sql_db_list_tables`, `sql_db_schema`, `sql_db_query`, `sql_db_query_checker`) and lets the LLM plan a sequence: list tables, inspect the relevant ones, draft SQL, check it, execute it, retry on error. Reference: [LangChain SQL guide](https://python.langchain.com/docs/tutorials/sql_qa/).

```python
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

db = SQLDatabase.from_uri("postgresql+psycopg://readonly_user:pw@host/db")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_sql_agent(llm=llm, db=db, agent_type="tool-calling", verbose=True)
result = agent.invoke({"input": "how many open P1 tickets closed last week for tier-1 customers?"})
```

The agent pattern is more flexible (it can iterate on errors, ask clarifying questions, chain queries) but more expensive (multiple LLM turns) and harder to constrain. The `NLSQLTableQueryEngine` pattern is more deterministic (one LLM call, one SQL query, one execution) and cheaper.

### Few-shot examples

Both frameworks support a few-shot examples list — pairs of (question, correct SQL) that go in the prompt to bias the generator toward your dialect / conventions:

```python
few_shot_examples = [
    {"question": "how many open tickets do we have?",
     "sql": "SELECT COUNT(*) FROM tickets WHERE status = 'open'"},
    {"question": "list customers with revenue over 1M",
     "sql": "SELECT name FROM customers WHERE annual_revenue > 1000000"},
    # 5-20 examples covering your common question shapes
]
```

Empirically, few-shot examples that share the *shape* of the target question (aggregation, join, filter, ordering) lift accuracy more than examples that share the *domain*. If your users frequently ask "top N by X grouped by Y" questions, include a couple of those regardless of subject.

## Query-safety guards — what a rogue query looks like and how to block it

The threat model is not SQL injection (the LLM produces the SQL; there's no user string interpolated into it — you should still use parameter binding on any literal values you *do* substitute). The real threats:

1. **Data exfiltration** — a query that selects everything from a sensitive table. `SELECT * FROM employees` when the user's role doesn't grant that.
2. **Resource abuse** — a query that scans a 10-billion-row table without an index, or a join without a predicate, or a cartesian product. Blocks the warehouse for hours; costs thousands.
3. **Destructive statements** — `DROP`, `DELETE`, `UPDATE`, `TRUNCATE`. The LLM shouldn't emit these, but if it does, they should not execute.
4. **Cross-schema / cross-database queries** — queries that reach tables outside the whitelist. `SELECT * FROM pg_catalog.pg_stat_activity` is not what you meant.
5. **Timing / covert exfiltration** — subtler; queries that use timing side channels to leak data across pipelines. Bounded query timeouts help.

The layered defense:

- **Connect as a read-only DB user** whose grants are exactly the tables + columns your text-to-SQL layer is allowed to see. Everything below is defense in depth; the primary defense is the grant. If the user has no `SELECT` on `employees.ssn`, no LLM-generated query can retrieve it.
- **Parse the generated SQL** with a SQL parser (`sqlglot`, `pglast`) before executing. Reject if the parse tree contains anything other than a `SELECT`. Reject if there is a `SELECT *`, a subquery that references a non-whitelisted table, or a function call outside a whitelist.
- **Enforce a query timeout** — `SET statement_timeout = '5s'` (Postgres) or the equivalent on your engine. A query that takes longer is killed. Set the timeout tight; users can always ask a narrower question.
- **Enforce a row cap** — wrap the generated query in `LIMIT 10000` (or similar). If the raw query returns more, the retriever knows to summarize / paginate rather than dump.
- **Whitelist column projection.** If the LLM emits `SELECT *`, rewrite to explicit columns (or reject). This prevents accidental serialization of sensitive columns you forgot to lock down.
- **Rate-limit** the text-to-SQL endpoint per user. An adversary who can issue 1000 SQL/s can extract a table one row at a time even with row caps.

An example guard implementation with `sqlglot`:

```python
import sqlglot
from sqlglot import exp

ALLOWED_TABLES = {"customers", "tickets", "products", "orders"}
ALLOWED_FUNCTIONS = {"count", "sum", "avg", "min", "max", "date_trunc", "extract"}

def safety_check(sql: str, dialect: str = "postgres") -> str | None:
    """Return None on pass; return an error message on reject."""
    try:
        tree = sqlglot.parse_one(sql, read=dialect)
    except sqlglot.errors.ParseError as e:
        return f"parse error: {e}"

    if not isinstance(tree, exp.Select):
        return "only SELECT statements are allowed"

    for table in tree.find_all(exp.Table):
        if table.name.lower() not in ALLOWED_TABLES:
            return f"table {table.name} not in whitelist"

    for star in tree.find_all(exp.Star):
        return "SELECT * is not allowed; specify columns explicitly"

    for func in tree.find_all(exp.Anonymous):
        if func.name.lower() not in ALLOWED_FUNCTIONS:
            return f"function {func.name} not in whitelist"

    return None
```

Wrap execution in a `try/except` that logs the offending query, the caller, and the failure reason. Every reject is a signal — either a text-to-SQL bug (fix the prompt) or a hostile query (alert on it).

## Execution-accuracy evaluation on Spider and BIRD

Text-to-SQL has a well-established eval methodology thanks to the [Spider](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/) benchmarks. The single most important metric is **execution accuracy** — does the generated SQL, when actually executed against the DB, produce the *same rows* as the reference SQL?

Reference set structure:

- Each example is `(question, gold_sql, db_schema, db_snapshot)`.
- The generated SQL is run against the same `db_snapshot`.
- If the returned result set equals the gold's returned result set (order-insensitive for `SELECT` without `ORDER BY`, order-sensitive with), the example is a pass.

Related metrics:

- **Exact-match accuracy** — string-level match of generated SQL to gold. Very brittle (semantically equivalent queries can look different), rarely useful for real evaluation.
- **Component-match accuracy** — Spider's original metric; compares parse-tree components (SELECT list, WHERE clauses, etc.) and reports partial credit. Useful for diagnosis, less useful for headline numbers.
- **Valid SQL rate** — of generated queries, what fraction even parses and runs without error. A lower bound on execution accuracy.

Spider is the older, cleaner set — 200 databases across 138 domains, hand-labeled by Yale researchers. It's mostly answered now (top systems > 90% on the dev set). BIRD is the newer, harder set — 12,751 questions over 95 databases, with a focus on realistic schemas that Spider glossed (dirty column names, missing docs, huge tables that require the SQL to be efficient not just correct). BIRD reports execution accuracy *and* a "valid-efficiency-score" that rewards queries that run fast on the real data.

Running the benchmarks locally:

- **Spider evaluation script:** [`spider_eval`](https://github.com/taoyds/spider/tree/master/evaluation) on GitHub — clone, install, run with your generated SQL file against the Spider dev DBs.
- **BIRD evaluation:** [`bird-bench` evaluation code](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird) — similar shape.

**Your own domain-fit eval matters more than Spider or BIRD.** Spider/BIRD tell you if your generator has the general capability; a held-out set of *your* questions against *your* schema tells you if it works on your data. Build a 50–200 question set. Author a gold SQL for each. Compute execution accuracy on your DB. Wire it into the mod-111 retriever CI so a prompt or model change that regresses execution accuracy blocks the merge.

Domain-fit set construction tips:

- Cover the question *shapes* your users ask — aggregation, filter, join, top-N, groupby, time-window.
- Include a few adversarial ones — questions that are subtly ambiguous ("last week" — calendar week or seven days back?), questions that require a nontrivial join path, questions whose plausible-looking SQL is wrong (a column that has the right name but wrong semantics).
- Version the set with the schema — if the schema changes, some gold SQL will need updating; that is expected.

## When text-to-SQL is the wrong tool

Text-to-SQL is powerful but not always right:

- **When the question is genuinely about text in unstructured docs** — a support ticket's *content* rather than *count*. Vector + lexical retrieval wins.
- **When the SQL would need to reason about text similarity** — "find customers whose feedback sounds like X." Push the fuzzy part to a vector search; use SQL only for the structured filter.
- **When the schema is too dynamic** to trust a schema-linked LLM — event tables with hundreds of thousands of columns via wide-table patterns, JSON blob columns where every row has a different shape. Text-to-JSON-path is a related but distinct discipline.
- **When the query needs graph traversal** — "customer chains connected via referrals up to 4 hops." SQL is theoretically capable (recursive CTEs) but a property graph (chapter 5) is a better fit.

Chapter 7 is where the router decides text-to-SQL vs. vector vs. graph on a per-question basis. For now, know that text-to-SQL is one of four tools, not the tool.

## What you should be able to do after this chapter

- Describe the four-stage text-to-SQL pipeline (schema-link → generate → guard → execute) and the failure mode each stage catches.
- Explain schema-linking as the retrieval step *inside* text-to-SQL, and implement a minimal version that embeds table+column descriptions and returns the top-N relevant tables.
- Use `NLSQLTableQueryEngine` or `SQLTableRetrieverQueryEngine` from LlamaIndex, and the LangChain SQL agent, and articulate the trade-off (deterministic single-shot vs. iterative agent).
- Author query-safety guards: read-only DB user, SQL parser rejecting non-`SELECT`, table / function / `*` whitelisting, timeouts, row caps, rate limiting. Explain why the DB grant is the *primary* defense.
- Define execution accuracy as the headline metric, describe Spider and BIRD as reference benchmarks, and build a domain-fit eval set on your own schema that plugs into the mod-111 CI gate.
- Name three question shapes for which text-to-SQL is the *wrong* retriever and cite chapter 7's router as the decision layer.

## Further reading

- Yu, T., Zhang, R., Yang, K., Yasunaga, M., Wang, D., Li, Z., Ma, J., Li, I., Yao, Q., Roman, S., Zhang, Z., & Radev, D. (2018). *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.* EMNLP. [arXiv:1809.08887](https://arxiv.org/abs/1809.08887); [Spider leaderboard](https://yale-lily.github.io/spider).
- Li, J., Hui, B., Qu, G., Yang, J., Li, B., Li, B., Wang, B., Qin, B., Cao, R., Geng, R., Huo, N., Zhou, X., Ma, C., Li, G., Chang, K. C.-C., Huang, F., Cheng, R., & Li, Y. (2023). *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs.* NeurIPS. [arXiv:2305.03111](https://arxiv.org/abs/2305.03111); [BIRD leaderboard](https://bird-bench.github.io/).
- LlamaIndex docs: [*Text-to-SQL Guide*](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/) — the `NLSQLTableQueryEngine` and `SQLTableRetrieverQueryEngine` reference.
- LangChain docs: [*Build a Question/Answering system over SQL data*](https://python.langchain.com/docs/tutorials/sql_qa/) — the SQL agent and toolkit reference.
- Pourreza, M., & Rafiei, D. (2023). *DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction.* NeurIPS. [arXiv:2304.11015](https://arxiv.org/abs/2304.11015) — the schema-link → decompose → generate → self-correct pattern that dominates Spider leaderboards; worth reading for the decomposition ideas.
- Gao, D., Wang, H., Li, Y., Sun, X., Qian, Y., Ding, B., & Zhou, J. (2023). *Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation.* [arXiv:2308.15363](https://arxiv.org/abs/2308.15363) — practical evaluation of prompt patterns and few-shot selection on Spider / BIRD.
- Lei, F., Chen, J., Ye, Y., Cao, R., Shin, D., Su, H., Suo, Z., Gao, H., Hu, W., Yin, P., Zhong, V., Xiong, C., Sun, R., Liu, Q., Wang, S., & Yu, T. (2024). *Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows.* [arXiv:2411.07763](https://arxiv.org/abs/2411.07763) — the enterprise-realistic successor to Spider.
- sqlglot: [GitHub](https://github.com/tobymao/sqlglot) — the parser this chapter's safety guard uses; supports 20+ SQL dialects and re-emits normalized SQL.
- PostgreSQL: [*Statement behavior configuration parameters*](https://www.postgresql.org/docs/current/runtime-config-client.html#RUNTIME-CONFIG-CLIENT-STATEMENT) — `statement_timeout` and related knobs for the execution safety envelope.
