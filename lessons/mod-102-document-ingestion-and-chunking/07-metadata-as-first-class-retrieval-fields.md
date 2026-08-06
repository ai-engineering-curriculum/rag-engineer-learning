# Chapter 7 — Metadata as First-Class Retrieval Fields

## Why this chapter exists

Every failure mode you cannot fix later starts with a metadata field you did not attach at ingest. Cannot filter by tenant? Should have attached `permission_scope`. Cannot show the user the page number in the citation? Should have attached `page`. Cannot exclude the 2019 draft that was superseded? Should have attached `updated_at`. Retrofitting a metadata field after the corpus is embedded costs a full re-ingest and re-embed pass; getting it right at ingest is a few extra lines.

This chapter names the fields the retrieval stack downstream will *demand* from you, gives you a schema to ship, and previews the pre-filter vs. post-filter correctness trap that mod-104 will drill into (and that a metadata schema either enables you to avoid or forces you into).

## The mental model: metadata is retrieval infrastructure, not documentation

New engineers treat metadata as descriptive — "nice to have for debugging." A retrieval engineer treats it as *the substrate the retriever operates on*. Six things every downstream stage does with metadata:

- **Filter** — pre-filter to a subset of the corpus before ANN scan (`tenant_id = 'acme'`), or post-filter the returned candidates. mod-104 goes deep on which one, when.
- **Boost / demote** — up-weight fresh docs, down-weight archived docs, up-weight the primary source over reprints (mod-105/106).
- **Route** — send tabular-fact queries to a table-only sub-index, prose queries to a prose sub-index. Impossible without `element_type`.
- **Dedup / cluster** — collapse near-duplicate hits by `doc_id` or `source_uri` at assembly time (mod-107).
- **Cite** — the generator emits `[cite: doc_id#section, p.14]`. Missing `page` or `section_path` = missing citations = failed compliance audit.
- **Version** — every chunk carries `parser_version` and `chunker_version`; on model swap, only re-embed the affected slice (mod-111).

Each of these is a real requirement in a real retrieval product. Each dies quietly at ingest if the field is missing.

## The schema that pays back — a starting shape

Extends the `Passage` from chapter 1 into the on-disk `Chunk` the retriever will see:

```python
@dataclass
class Chunk:
    # --- identity ---
    chunk_id: str                    # stable ID: hash(doc_id + chunker_version + ordinal)
    doc_id: str                      # stable per-source-document ID
    source_uri: str                  # URL / S3 key / git blob SHA
    ordinal: int                     # position of chunk within doc (monotonic)

    # --- content ---
    text: str                        # what gets embedded / indexed
    element_type: str                # heading | paragraph | list_item | table | table_summary
                                     # | code | figure_caption | footnote | ...
    tokens: int                      # exact count under the embedding-model tokenizer

    # --- structural ---
    section_path: list[str]          # ["Compensation", "Salary Bands", "Level 30"]
    page: int | None                 # 1-indexed
    bbox: tuple[float, float, float, float] | None  # (x0, y0, x1, y1) for citation deep-links

    # --- provenance / governance ---
    updated_at: datetime | None
    published_at: datetime | None    # publish date, distinct from source-file mtime
    author: str | None
    permission_scope: str            # tenant / group / ACL tag; see mod-112
    language: str                    # ISO 639-1: "en", "fr", "ja"
    content_hash: str                # sha256(text), lets you detect unchanged chunks on re-ingest

    # --- versioning ---
    parser: str                      # "docling@2.4.0"
    parser_version: str              # ditto, parsed out
    chunker: str                     # "llama_index.SentenceSplitter"
    chunker_version: str             # "1.0.0" — matches chunker.yaml (chapter 5)
    ingested_at: datetime            # when this chunk landed in the current corpus snapshot

    # --- parent-child linkage (chapter 6) ---
    parent_id: str | None            # for small-to-big
    children_ids: list[str]          # for hierarchical
    window_ids: list[str] | None     # for sentence-window

    # --- retrieval assist ---
    keywords: list[str]              # optional: extracted named entities / KE tags for hybrid retrieval
    summary: str | None              # optional: LLM-generated 1-sentence summary; chapter 8's table case
```

Not every field is populated for every chunk. `page` is `None` for a Markdown chunk; `bbox` is `None` for anything that is not from a PDF; `summary` is populated only for tables and figures. Keep the field, mark it nullable, and let the retriever do the right thing when it is present.

## The six fields you will regret omitting

Of the many fields above, these six are the ones production retrieval engineers routinely wish they had attached from day one:

### 1. `doc_id` — stable per-source-document identity

Not the file path (paths move). Not the URL (URLs change). Something derived from the *content* or from the source system's own stable identifier — a Confluence page ID, a SharePoint document ID, a `sha256` of the normalized bytes if nothing else is stable. Every chunk needs to be groupable back to its document for dedup, citation, and change detection.

### 2. `section_path` — hierarchical location within the document

The list of headings the chunk sits under, from H1 down. A single "Level 30" chunk from *Compensation → Salary Bands → Level 30* is barely useful in isolation; when the retriever can filter by, order by, or cite the section path, the same chunk becomes powerful. Structural chunkers (chapter 4) give you this for free; recursive splitters need you to thread it through.

### 3. `page` (and `bbox` when available) — anchor for the citation

The user asked a question, the LLM cited chunk 8371, and the user wants to see the source. `page` is the minimum viable citation. `bbox` on a PDF chunk lets you highlight the exact rectangle. Both are trivial to attach at ingest and painful to reconstruct after.

### 4. `updated_at` and `published_at` — freshness signals

`updated_at` is the source-system last-modified. `published_at` is the document's own publication date (from a `<meta>` tag, a document header, or Confluence metadata). Both feed into two decisions the retriever will make:

- **Filter stale drafts.** `WHERE updated_at > NOW() - INTERVAL '2 years'` for corpora with high churn.
- **Break ties toward fresher content.** Same score → prefer the newer chunk. Especially important when the corpus contains multiple versions of the same policy.

### 5. `permission_scope` — the ACL tag

This module does not implement enterprise ACL (mod-112 does). It absolutely does attach the field. The value can be as coarse as `tenant_id` at this stage — a per-tenant string that the retriever will use as a **pre-filter** to guarantee cross-tenant chunks never make it into the ANN scan in the first place. Missing this field at ingest = incident later.

### 6. `content_hash` — the change-detection primitive

Re-ingesting a corpus that has 5% new/changed chunks and 95% unchanged chunks should re-embed 5%, not 100%. `content_hash = sha256(normalized_text)` gives you that check for free. On re-ingest, look up each computed hash in the existing index; skip embedding if unchanged; upsert if changed. Compound wins with `parser_version` and `chunker_version` — a version bump in either implies all hashes may have shifted.

## Pre-filter vs. post-filter — the trap this schema helps you avoid

mod-104 will name this in detail; the metadata decision starts here.

- **Pre-filter** — the vector DB scans *only* the chunks matching the filter (`tenant_id = 'acme'`). Correctness guaranteed; performance depends on how the index implements the filter under the hood. Some ANN indexes handle filtering elegantly (Weaviate, Qdrant, pgvector with pruning); others degrade recall when filters get selective.
- **Post-filter** — the vector DB scans the whole index, returns top-*N*, then filters. Correctness: fine *if you retrieve enough candidates that N-after-filter ≥ K*. Failure mode: retrieve 10, filter down to 0, return no results for a tenant that had perfectly good hits.

The metadata decision now: mark the field as **indexed / filterable** in whichever store you use. In pgvector that means a real Postgres column with an index (not a JSONB blob). In Elasticsearch / OpenSearch that means a `keyword` or `date` field type in the mapping. In Pinecone / Weaviate / Qdrant / Milvus / Chroma each has its own "which metadata fields are filterable" story — read the docs *before* landing data.

Wrong: `metadata: {tenant_id: "acme", updated_at: "2026-08-06"}` stored as an opaque blob. You cannot filter on it efficiently, so the retriever can only post-filter, so tenant isolation is a "usually works" property. Right: `tenant_id` and `updated_at` as first-class indexed columns/fields with their own DDL/mapping.

## Attaching metadata during ingest — a normalization sketch

The parsers in chapter 2 emit their own native metadata (unstructured `Element.metadata`, Docling `DoclingDocument` items, PyMuPDF bboxes). The ingestion pipeline's job is to normalize them into the `Chunk` schema:

```python
def to_chunks(passages: list[Passage], *, doc_id: str, source_uri: str,
              chunker: Chunker, permission_scope: str,
              published_at: datetime | None) -> list[Chunk]:
    updated_at = fetch_source_mtime(source_uri)
    language = detect_language(" ".join(p.text for p in passages)[:2000])

    chunks: list[Chunk] = []
    for parent in chunker.chunk(passages):        # returns list[ParentUnit]
        parent_id = stable_id(doc_id, chunker.version, parent.ordinal)
        for i, small in enumerate(parent.children):
            chunks.append(Chunk(
                chunk_id=stable_id(doc_id, chunker.version, parent.ordinal, i),
                doc_id=doc_id,
                source_uri=source_uri,
                ordinal=parent.ordinal * 1000 + i,
                text=small.text,
                element_type=small.element_type,
                tokens=count_tokens(small.text),
                section_path=small.section_path,
                page=small.page,
                bbox=small.bbox,
                updated_at=updated_at,
                published_at=published_at,
                author=small.author,
                permission_scope=permission_scope,
                language=language,
                content_hash=sha256(small.text.encode("utf-8")).hexdigest(),
                parser=small.parser,
                parser_version=small.parser_version,
                chunker=chunker.name,
                chunker_version=chunker.version,
                ingested_at=now_utc(),
                parent_id=parent_id,
                children_ids=[],
                window_ids=None,
                keywords=extract_keywords(small.text),
                summary=None,       # populated for tables/figures in chapter 8
            ))
    return chunks
```

Nothing exotic. The discipline is running this on every code path; every branch of chapter 2's parser router feeds through the same normalization.

## What LlamaIndex and LangChain give you out of the box

Both frameworks model chunks as first-class objects with a `metadata` dict. Some fields are attached automatically; the rest you attach.

- **LlamaIndex** — `TextNode.metadata` is a dict, and readers (`SimpleDirectoryReader`, `SimpleWebPageReader`, ...) attach `file_path`, `file_name`, `page_label`, `creation_date` for you. Node parsers propagate parent metadata into children — the reason `MarkdownNodeParser` carries heading path down automatically. There is an `excluded_llm_metadata_keys` / `excluded_embed_metadata_keys` distinction so you can carry a field for filtering *without* embedding it into the text-to-embed. Use this — you almost never want `tenant_id` or `ingested_at` to be part of the vector.
- **LangChain** — `Document.metadata` is a dict. Metadata carried through splitters (`RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter`) into each output chunk. `SelfQueryRetriever` uses the metadata schema to translate natural-language questions into filter predicates ([SelfQueryRetriever docs](https://python.langchain.com/docs/how_to/self_query/)).

## The two disciplines that keep this working

**Discipline 1 — every field lives in a schema, not in a `dict`.** A `Chunk` `dataclass` (or Pydantic model) forces the field set to be visible in code; a `dict` lets fields drift, fields go missing on some paths, fields get typo'd (`tenant` vs. `tenant_id`), and you find out at retrieval time.

**Discipline 2 — a field is either always populated for a shape of chunk, or explicitly nullable, and the retriever handles null.** Do not populate `page = 0` for Markdown chunks because "we needed a number." `None` is the right value; the retriever should never surface `p. 0` as a citation.

## What you should be able to do after this chapter

- List the six retrieval uses of metadata (filter, boost, route, dedup, cite, version) and give one field each.
- Author a `Chunk` schema that carries identity, content, structural, provenance, versioning, and parent-child linkage fields.
- Name the six fields (`doc_id`, `section_path`, `page`, `updated_at`, `permission_scope`, `content_hash`) that engineers most commonly regret omitting.
- Explain the pre-filter vs. post-filter distinction and the metadata configuration required to enable pre-filter cheaply.
- Route metadata correctly through both LlamaIndex `TextNode.metadata` and LangChain `Document.metadata`, and use `excluded_embed_metadata_keys` (or equivalent) to keep filter-only fields out of the vector.
- Enforce the two disciplines (schema-not-dict; nullable-not-sentinel) in ingest code.

Chapter 8 tackles the trickiest content types — tables, formulas, and figures — where naïve chunking loses the source-cell reference the generator needs for a citation.

## Further reading

- LlamaIndex, [*Customizing Documents*](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/), [*Metadata Extraction*](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_metadata_extractor/).
- LangChain, [*Self-querying retriever*](https://python.langchain.com/docs/how_to/self_query/) — end-to-end example of NL → metadata filter translation.
- Weaviate, [*Filters*](https://weaviate.io/developers/weaviate/api/graphql/filters) — how filters compose with vector search in one representative vector DB.
- pgvector, [*Filtering*](https://github.com/pgvector/pgvector#filtering) — the pre-filter / post-filter shape in Postgres.
- Elasticsearch, [*Filter DSL*](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html) — filter clauses that compose with `knn` queries.
