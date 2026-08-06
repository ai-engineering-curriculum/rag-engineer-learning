# Chapter 4 — Chunking Strategies: Fixed, Structural, and Semantic Segmentation

## Why this chapter exists

At level 20, chunking is `text.split(...)`-with-overlap. Ship the demo, move on. This chapter is where you replace that with the strategies real corpora justify: **structural** (respect the document's own headings, tables, and code blocks), **semantic** (put a boundary where meaning shifts, using the embedding model itself), and the fixed / token / character windows that remain a sensible baseline when nothing else fits.

The chunker's job is to turn the `Passage` output of chapters 1–3 into `Chunk`s the embedder (mod-103) and retriever (mod-104) can work with. Every downstream number — recall@k, MRR, faithfulness — has a ceiling set by this decision. Chapters 5, 6, and 8 refine specific dimensions of it (size trade-off; hierarchical parent-child; tables and formulas). This chapter surveys the strategy families and the two libraries (LlamaIndex and LangChain) that ship implementations of each.

## The strategy families

| Strategy | Boundary rule | When it wins | When it loses |
|---|---|---|---|
| **Fixed character / token window** | Every N characters / tokens, with M overlap | Homogeneous plain text; nothing else to work with; strong baseline for ablations | Splits mid-sentence; ignores headings, tables, code |
| **Sentence / recursive** | Prefer sentence boundaries; recurse on chars if a sentence exceeds a target | Prose, articles, chat transcripts | Same weaknesses as fixed on structural content |
| **Structural (section / table / code / Markdown-header)** | Split at the document's own structure — headings, table boundaries, function definitions | Handbooks, manuals, technical docs, code, structured Office documents | Loses when structure is missing (chat, freeform notes) or over-granular (three sentences per heading) |
| **Semantic (embedding-similarity segmenter)** | Move a sliding window over sentences; put a boundary where similarity of adjacent windows drops below a threshold | Freeform prose whose topic shifts are strong but not signposted by headings | Cost (one embedding call per sentence at chunk-time); brittle on domain-specific vocab; can silently over- or under-segment |
| **Hierarchical / parent-child** | Produce small chunks *and* their bigger parents (paragraph → section → document); retrieve small, return big | RAG where the retriever needs precise units but the LLM needs surrounding context (chapter 6) | Complexity; two writes per unit; downstream pipeline must know the shape |
| **Small-to-big / sentence-window** | Index each sentence; on retrieval, expand to a fixed window of neighbours | Question-answering over long factual documents where the "answer sentence" is small | Same as parent-child; window size becomes a knob to tune |

You will use *all* of them in the same corpus, per document type. There is no default. The chapter's exercise (`exercise-03`) has you A/B them on one corpus and defend the winner with a retrieval metric.

## Fixed and recursive: the level-20 shape, made honest

Even at level 30, fixed-window and recursive splitters remain the sensible **baseline** every ablation compares to. Two facts to internalize:

1. **Count tokens, not characters.** The embedding model has a token limit (usually 512 for small BGE / E5 models, ~8K for OpenAI `text-embedding-3-large`). A 1,000-character chunk of English is ~250 tokens; a 1,000-character chunk of code or CJK is closer to 500. Use the *embedding model's own tokenizer* to count, not `len(text)`.
2. **Overlap is not free.** Overlap of ~10–15% of chunk size preserves cross-boundary continuity (a sentence that started at the end of chunk N and continued at the start of chunk N+1) at the cost of duplicated tokens in the index. Overlap over ~25% is almost always a code smell — you are compensating for a bad boundary policy.

LangChain and LlamaIndex both ship a **recursive** splitter that tries paragraph, then sentence, then word, then character boundaries — pick the first one that keeps chunks under the target. This is the sensible default for prose when structural signals are absent.

```python
# LangChain — recursive character text splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,          # target size in *tokens* if you pass a token-counting length_function
    chunk_overlap=100,       # ~12% of chunk size
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=lambda s: len(tokenizer.encode(s)),
)
chunks = splitter.split_text(long_prose)
```

```python
# LlamaIndex — sentence splitter with token-aware sizing
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,          # token budget per chunk (uses the tokenizer you configure)
    chunk_overlap=64,
    paragraph_separator="\n\n\n",
)
nodes = splitter.get_nodes_from_documents(docs)
```

Both are fine. LlamaIndex has stronger built-in metadata propagation (headings, page numbers, refs) because the ecosystem defaults to structured `Document` and `Node` types; LangChain has a broader per-language `from_language` set for code splitting.

## Structural: split where the document already tells you to

The structural family is the "free lunch" of chunking. If the parser (chapter 2) preserved headings, tables, and code blocks, you already have your boundaries — using them beats every other strategy on that content.

### By heading (Markdown, HTML, Office)

Every well-parsed document has a heading tree. Split at heading boundaries; when a section is too big for the target chunk size, recursively split it at paragraph boundaries within that section.

```python
# LangChain — MarkdownHeaderTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

headers = [("#", "H1"), ("##", "H2"), ("###", "H3")]
h_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
sections = h_splitter.split_text(markdown_text)
# each section is a Document with metadata like {"H1": "Compensation", "H2": "Salary Bands"}

r_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = r_splitter.split_documents(sections)
# now each chunk carries the section metadata (chapter 7 depends on this)
```

```python
# LlamaIndex — MarkdownNodeParser
from llama_index.core.node_parser import MarkdownNodeParser

parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(md_documents)
# each node inherits its heading path in metadata; can be re-split with SentenceSplitter downstream
```

The unstructured library's `chunk_by_title` helper ([docs](https://docs.unstructured.io/open-source/core-functionality/chunking#chunking-strategies)) does the equivalent for its typed-element output — it keeps `Table` and `CodeSnippet` elements as their own chunks and packs adjacent narrative under a title until a size budget is hit.

### By table

Never split a table across chunks. Two shapes:

- **Small table** — one chunk containing the whole table, ideally with a one-sentence LLM-generated *summary* attached so the retriever has both the row-level detail and a high-level anchor. Chapter 8 goes deep.
- **Large table** — split by row groups (10–50 rows per chunk depending on cell width), and *repeat the header row and a "Table X, rows 41–90 of 200" caption* in each chunk so a chunk fetched in isolation still carries context.

### By code

Never split a function or class across chunks. LlamaIndex's `CodeSplitter` uses tree-sitter:

```python
from llama_index.core.node_parser import CodeSplitter

py_splitter = CodeSplitter(language="python", chunk_lines=40, chunk_lines_overlap=8)
nodes = py_splitter.get_nodes_from_documents(code_documents)
```

LangChain's equivalent is `RecursiveCharacterTextSplitter.from_language(Language.PYTHON, ...)`, which uses per-language separator lists (`\nclass `, `\ndef `, etc.) rather than a real parser — cheaper, less precise. Reach for the tree-sitter route when you care.

The rule that generalizes: whatever the document's *own* segmentation is (headings, table rows, function definitions, slide boundaries, cell boundaries in a notebook), respect it before you fall back to length-based splitting.

## Semantic: split where meaning shifts

Semantic chunking uses the embedding model itself as the boundary detector. The rough algorithm:

1. Split the text into sentences (spaCy or NLTK sentence tokenizer).
2. Embed each sentence.
3. For each adjacent pair of sentences, compute cosine similarity.
4. Put a chunk boundary where similarity drops below a threshold — the intuition is that the topic has shifted.
5. Merge tiny chunks into their neighbours until every chunk is above a minimum size.

```python
# LlamaIndex — SemanticSplitterNodeParser
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

parser = SemanticSplitterNodeParser(
    buffer_size=1,                   # how many sentences per side to compare
    breakpoint_percentile_threshold=95,  # boundary where sim-delta > 95th percentile of the doc
    embed_model=embed,
)
nodes = parser.get_nodes_from_documents(docs)
```

LangChain's [`SemanticChunker`](https://python.langchain.com/docs/how_to/semantic-chunker/) implements a close variant of the algorithm popularized by Greg Kamradt in his ["5 Levels of Text Splitting"](https://github.com/FullStackRetrieval-com/RetrievalTutorials) walkthrough.

Two honest caveats before you reach for it:

- **Cost.** Semantic chunking runs one embedding call per sentence at chunk time. For a 5 M-chunk corpus, that is 5 M+ extra embedding calls on the offline path. Fine if you self-host a small model; expensive on a hosted API.
- **No free lunch on accuracy.** Semantic chunking helps when the document has meaningful topic shifts and *no* structural signposts. On a handbook with clean headings, `MarkdownHeaderTextSplitter` will beat it, faster and cheaper. `exercise-03` puts numbers on this rather than adjudicating from vibes.

## Hierarchical, parent-child, small-to-big — pointer forward

The strategies above pick *one* boundary policy per document. The **hierarchical / parent-child** strategies pick *two or more* — small chunks for precise retrieval, big chunks for context handed to the LLM — and index the small ones while returning the big ones. Chapter 6 goes deep. The relevant preview:

- LlamaIndex `HierarchicalNodeParser` produces a tree of chunk sizes (e.g., 2048 / 512 / 128 tokens) and its `AutoMergingRetriever` returns the parent when enough of its children match.
- LangChain `ParentDocumentRetriever` indexes children, retrieves parents.
- LlamaIndex `SentenceWindowNodeParser` indexes single sentences and, on retrieval, expands to a `±k`-sentence window.

Note them here; do not implement yet.

## The library map

| Task | LlamaIndex | LangChain | unstructured |
|---|---|---|---|
| Recursive character/token | `SentenceSplitter`, `TokenTextSplitter` | `RecursiveCharacterTextSplitter` | `chunk_by_character` |
| Markdown headers | `MarkdownNodeParser` | `MarkdownHeaderTextSplitter` (+ recursive) | `chunk_by_title` on parsed Markdown |
| Section / title (typed elements) | via LlamaHub readers that return typed nodes | via `unstructured` output loaded as documents | `chunk_by_title` |
| Code | `CodeSplitter` (tree-sitter) | `RecursiveCharacterTextSplitter.from_language` | `chunk_by_character` (weak) |
| Semantic | `SemanticSplitterNodeParser` | `SemanticChunker` | not built-in |
| Parent-child / hierarchical | `HierarchicalNodeParser` + `AutoMergingRetriever` | `ParentDocumentRetriever` | not built-in |
| Sentence window | `SentenceWindowNodeParser` | assemble manually | not built-in |

Neither library is a superset. It is normal to mix and match — use LlamaIndex's `MarkdownNodeParser` for scaffolding and LangChain's `SemanticChunker` for the inner split, or vice versa. Both are covered in the exercises so you leave with working muscle memory for both.

## Chunker-side decisions worth naming

- **Deterministic output.** Every chunker is a function; run it twice on the same input, get identical output. If you use `SemanticChunker` with a hosted embedding that returns slightly different vectors between calls, you will get non-reproducible chunks — and non-reproducible indexes (mod-111 hates this). Pin the embedding model version.
- **Chunk IDs.** Emit stable IDs — `sha256(doc_id + '::' + chunker_name + '::' + chunker_version + '::' + chunk_ordinal)` — not sequential integers. Stable IDs let you diff two chunker runs and re-embed only the changed chunks.
- **Chunker version in metadata.** Chapter 7 will make this a first-class field. The chunker is a versioned artifact that participates in the "which index version does this chunk belong to?" contract.
- **Round-trip check.** Every chunker should pass a "no content silently dropped" test: `sum(len(chunk.text) for chunk in chunks) ≥ 0.95 * len(source_text)` (allowing for whitespace normalization). If a chunker is silently discarding 20% of your corpus, its recall ceiling is already 80%.

## What you should be able to do after this chapter

- Name the six strategy families and pick a first-choice for a given corpus (Markdown docs, PDF handbook, source code, chat transcripts).
- Configure and call `RecursiveCharacterTextSplitter` / `SentenceSplitter` with a token-aware length function.
- Configure and call a Markdown or unstructured structural splitter that preserves heading metadata into each chunk.
- Configure and call `SemanticSplitterNodeParser` / `SemanticChunker` and name its cost implication.
- Write a "no content silently dropped" assertion for your chunker.
- Point to the LlamaIndex or LangChain class for each shape in the library map above.

Chapter 5 quantifies the trade-off: how do you pick chunk size, and how does it interact with recall, precision, embedding context, and reranker cost?

## Further reading

- LlamaIndex, [*Node Parser Modules*](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/), [*Semantic Chunking*](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/).
- LangChain, [*Text Splitters*](https://python.langchain.com/docs/how_to/#text-splitters), [*Semantic Chunker*](https://python.langchain.com/docs/how_to/semantic-chunker/), [*Split by tokens*](https://python.langchain.com/docs/how_to/split_by_token/).
- Unstructured, [*Chunking strategies*](https://docs.unstructured.io/open-source/core-functionality/chunking).
- Kamradt, G. [*5 Levels of Text Splitting*](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb) — canonical practitioner walk-through of the strategy families, including the algorithm behind `SemanticChunker`.
- Anthropic engineering, [*Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) — includes a public case study on the effect of chunk boundary + per-chunk context on retrieval quality.
