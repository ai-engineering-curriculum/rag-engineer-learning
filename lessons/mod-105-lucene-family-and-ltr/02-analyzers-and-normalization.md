# Chapter 2 — The Analyzer Pipeline: Tokenizers, Filters, Stemmers, ICU, and Synonyms

## Why this chapter exists

The single biggest lever on lexical retrieval quality is **the analyzer** — the pipeline that turns a raw string into the terms that get stored in the inverted index. Get the analyzer right and BM25 recall improves silently across every query. Get it wrong and no amount of query tuning, boosting, or reranking recovers the recall that was never indexed to begin with. This is a lever most engineers under-invest in because it looks like plumbing; it is actually the single highest-leverage thing you tune on a lexical index.

This chapter walks the analyzer as it is implemented in the Lucene family — Elasticsearch, OpenSearch, Solr, and (with different terminology) Vespa — and gives you the vocabulary and defaults to reach for. Chapter 3 uses these analyzers as the building blocks of the field mappings that a production BM25 index actually deploys.

## The analyzer contract

An analyzer is a **character filter → tokenizer → token filter[]** pipeline that runs at both **index time** (on the field's value before it lands in the inverted index) and **query time** (on the raw query string before it is turned into a boolean of term queries). Elasticsearch and OpenSearch use these exact terms; Solr calls them `<charFilter>`, `<tokenizer>`, and `<filter>` in the schema XML; Vespa calls the same idea a `stemming` / `normalizer` / `linguistics` chain, hard-coded per field.

The [Elasticsearch analyzer docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/analyzer-anatomy.html) are the canonical reference for the pipeline shape; the same shape applies (with minor renames) to the other three engines.

Concretely, the pipeline turns `"The Quick brown foxes jumped over Ms. Möller's <b>house</b>!"` into (with the `english` analyzer): `["quick", "brown", "fox", "jump", "over", "ms", "möller", "hous"]` — lowercased, HTML-stripped, tokenised on non-letter characters, stopword-filtered, and Porter-stemmed. Every design choice in the analyzer changes that output list, and the inverted index only sees the output list.

## The `standard` analyzer is a floor, not a plan

Every one of the four engines ships a **`standard` analyzer** as the out-of-the-box default:

- **Standard tokenizer** — implements the [Unicode Text Segmentation](https://unicode.org/reports/tr29/) (UAX #29) word-break algorithm. Splits on whitespace, punctuation, and other Unicode boundaries. Correct for European languages; poor for CJK, Thai, and other unspaced scripts (see "Language-specific analyzers" below).
- **Lower-case token filter.**
- **No stemming, no stopwords, no synonyms.**

The `standard` analyzer is a floor that lets you index-anything-in-any-language and get *something* back. It is never the plan for a production lexical index. Every non-trivial retrieval workload benefits from adding at least one of {language-specific stemmer, ICU normalisation, custom synonyms, stopword handling appropriate to the domain}.

The rule of thumb: **`standard` is what you demo with; a domain- and language-appropriate custom analyzer is what you ship**.

## The five decisions you make when authoring an analyzer

The decisions below are the ones a level-30 engineer needs to defend. Each is a lever you turn deliberately, and each defaults badly if you leave the engine to pick.

### 1. Character-level normalisation (before tokenisation)

Character filters clean up the input string before the tokenizer sees it. The three you reach for most often:

- **`html_strip`** ([Elastic docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-htmlstrip-charfilter.html)) — remove HTML tags. Essential for any web-scraped corpus. Without it, tokens like `<b>foo</b>` leak angle brackets into the index.
- **`mapping`** ([Elastic docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-mapping-charfilter.html)) — a straight character-substitution table. Common uses: `":)"` → `"__happy__"`, `"C#"` → `"csharp"`, `"C++"` → `"cplusplus"`. Cheap to author, brittle if abused.
- **`pattern_replace`** ([Elastic docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-pattern-replace-charfilter.html)) — regex substitution. Useful for stripping tracking codes, redacting PII patterns, or normalising known typo classes.

The one *character-level* decision worth making explicit early: **Unicode normalisation form** (NFC vs. NFD vs. NFKC vs. NFKD). Different sources produce visually identical strings that are byte-different (`é` as `U+00E9` vs. `e` + `U+0301`). See the ICU section below — you fold Unicode normalisation into a normaliser filter.

### 2. Tokenisation strategy

The tokenizer is the pipeline's single most important choice.

- **`standard` tokenizer** — UAX #29 word-break. The default. Correct for space-delimited scripts. Wrong for CJK (each character becomes a token by CJK-specific rules or gets glued together incorrectly).
- **`whitespace` tokenizer** — split only on whitespace. Keeps punctuation attached to words (`"foo,"` stays one token). Rarely what you want except for pre-normalised streams.
- **`keyword` tokenizer** — emits the entire input as a single token. This is how you make a field exact-match-only: no tokenisation, no stemming, no analysis at all. Enumerated categorical fields (`status`, `tenant_id`, `country_code`) always want `keyword`.
- **`pattern` tokenizer** — split on a regex. Useful for structured identifiers (`"orderId=12345"` split on `=`).
- **`edge_ngram` and `ngram` tokenizers** — emit prefix or all-substring shingles. Used for search-as-you-type / autocomplete; see the [Elastic autocomplete guide](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-edgengram-tokenizer.html). Bloat the index dramatically; use for the specific field that needs it, not the whole document.
- **Language-specific tokenizers** — see the next section.

### 3. Language-specific analyzers (the stemmer / segmenter question)

For every non-trivial retrieval workload in a natural language, you replace the standard tokenizer's output filters with a **language-specific** analyzer or filter chain. Elasticsearch ships language analyzers for [30+ languages](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html); OpenSearch and Solr ship near-parallel sets. Each language analyzer bundles:

- A tokenizer (usually `standard`, sometimes language-specific like Japanese `kuromoji_tokenizer` or Chinese `smartcn_tokenizer`).
- A lowercase filter (or language-appropriate case-folding).
- A stopword filter with a curated per-language stopword list.
- A **stemmer** — the language-specific token-normalisation step that reduces word forms to a common stem.

The stemmer choice matters:

- **Porter stemmer** ([Porter, 1980](https://tartarus.org/martin/PorterStemmer/def.txt)) — the classic aggressive English stemmer. Bundled as `english` analyzer's stemmer in Elasticsearch. Chops off suffixes: `"connection"` → `"connect"`, `"argues"` → `"argu"`. Aggressive; occasionally over-stems.
- **KStem** — a lighter English stemmer that preserves more word forms; Elasticsearch's `light_english` uses it.
- **Snowball stemmers** ([snowballstem.org](https://snowballstem.org/)) — a family of language-specific stemmers Porter authored later. Covers Danish, Dutch, English, Finnish, French, German, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Turkish, and more.
- **Morphology-based analyzers** — for morphologically-rich languages (Turkish, Arabic, Finnish), Snowball's suffix-stripping is inferior to true morphological analysis. Elastic ships [`hunspell` token filter](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-hunspell-tokenfilter.html) which uses Hunspell dictionaries (the same used by LibreOffice/Firefox spell-check) — much better for languages where stem forms are hard to derive from suffixes.
- **Lemmatisation vs. stemming** — stemming reduces to a *root form* that may not be a real word (`"argu"`); lemmatisation reduces to the *dictionary form* (`"argue"`). Elastic ships no built-in lemmatiser; teams that need one plug in [spaCy](https://spacy.io/) or [Stanza](https://stanfordnlp.github.io/stanza/) upstream and index the lemmatised text.

For CJK languages the tokenizer choice is more important than the stemmer:

- **Japanese** — [`analysis-kuromoji` plugin](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-kuromoji.html) for Elasticsearch/OpenSearch; [Kuromoji](https://github.com/atilika/kuromoji) is the reference Java implementation with an IPAdic-derived dictionary.
- **Chinese** — [`analysis-smartcn` plugin](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-smartcn.html) using ICTCLAS-style HMM segmentation; alternatives include [Jieba](https://github.com/fxsjy/jieba) (Python-native) called upstream.
- **Korean** — [`analysis-nori` plugin](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-nori.html) using [mecab-ko-dic](https://bitbucket.org/eunjeon/mecab-ko-dic/).
- **Thai / Lao / Khmer** — no whitespace; the [Thai tokenizer](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-thai-tokenizer.html) uses ICU's dictionary-based word-break.

The rule of thumb: **do not use the `standard` analyzer for CJK, Thai, or Arabic corpora**; recall drops silently. Install the appropriate language plugin and use its analyzer.

### 4. ICU: Unicode normalisation, folding, and collation

The [`analysis-icu` plugin](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu.html) (Elasticsearch/OpenSearch) or [ICUCollationField and ICU token filters in Solr](https://solr.apache.org/guide/solr/latest/indexing-guide/language-analysis.html#icu-collation) exposes IBM's [International Components for Unicode](https://icu.unicode.org/) — the reference library for Unicode-correct text processing.

The three ICU components you reach for:

- **`icu_normalizer`** — Unicode normalisation (NFC, NFD, NFKC, NFKD). Fixes the `é` vs. `e` + `U+0301` byte-difference class of bugs. `nfkc_cf` is the compatibility-decomposed-then-case-folded normaliser — the recommended choice for search fields because it also applies Unicode case folding correctly (which handles Turkish dotless-i, German ß → ss, and Greek final sigma correctly, unlike ASCII lowercasing).
- **`icu_folding`** — Unicode-aware ASCII folding. Reduces accented characters to their base form: `"Möller"` → `"moller"`, `"café"` → `"cafe"`, `"naïve"` → `"naive"`. Turn this on when your users are likely to type unaccented forms of accented terms — most Western-language search UIs.
- **`icu_tokenizer`** — a UAX #29-compliant tokenizer that handles Thai/Lao/Khmer word-break (via ICU's dictionary-based break iterator) that the standard tokenizer botches.

The rule of thumb: **for any multi-lingual or non-ASCII corpus, install `analysis-icu` and add `icu_normalizer` (NFKC-CF) and `icu_folding` to your custom analyzer**. It is a five-minute install that silently improves recall on accented queries.

### 5. Synonyms

Synonym filters expand query and/or index terms so that a user query for `"laptop"` also matches documents indexed with `"notebook"`. Elasticsearch, OpenSearch, and Solr all support [SolrSynonymParser](https://solr.apache.org/guide/solr/latest/indexing-guide/filters.html#synonym-graph-filter) and [WordNet synonym files](https://wordnet.princeton.edu/).

Two placement choices, with different trade-offs:

- **Index-time synonyms** — the synonym filter runs at index time and stores all synonym forms in the index. Query time sees single terms. Simpler and query-time-cheaper; **cannot be changed without reindexing**. Also destroys IDF signal (a doc with 3 synonyms of a rare term looks 3× as common as it is).
- **Query-time synonyms** — the synonym filter runs at query time and expands one term into a boolean-OR of synonyms. Reindex-free; more expensive per query; **more correct** from an IDF perspective. The preferred choice for most real deployments.

The [Elasticsearch synonym graph filter](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-graph-tokenfilter.html) (as opposed to the older `synonym` filter) correctly handles multi-word synonyms like `"i pod" => "ipod"` — the older filter mishandles phrase-position offsets and breaks phrase queries silently. **Use `synonym_graph`, not `synonym`, on any modern deployment**.

Synonym source strategies:

- **Curated per-domain list.** The strongest signal. Domain experts write `"myocardial infarction, heart attack, MI"`. Ships with the analyzer.
- **Elasticsearch's `synonym` API** (8.10+, [Elastic docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/synonym-work-with.html)) — synonym sets managed via API rather than file-on-disk. Enables edit-without-reload.
- **WordNet** — public general-purpose synonym set; broad coverage, low domain precision. Useful as a fallback, dangerous as a primary.
- **Query-log-mined synonyms** — cluster co-clicked queries. Advanced; requires a query-log pipeline you probably don't have yet at level 30.

The rule of thumb: **prefer query-time `synonym_graph` on a curated per-domain synonym file, versioned in your repo, deployed as a synonyms API set on Elasticsearch 8.10+**.

## Two concrete analyzer definitions

### An English retrieval-quality analyzer for a technical corpus

The following custom analyzer combines the five decisions above into something you would actually ship for an English technical-support corpus. Written in the Elasticsearch DSL; adapt syntactically for OpenSearch (near-identical) and Solr (rewrite as `<analyzer>` XML).

```json
PUT /support_articles
{
  "settings": {
    "analysis": {
      "char_filter": {
        "html_strip_char": { "type": "html_strip" },
        "tech_synonyms_char": {
          "type": "mapping",
          "mappings": [
            "C# => csharp",
            "C++ => cplusplus",
            ".NET => dotnet",
            "K8s => kubernetes"
          ]
        }
      },
      "filter": {
        "english_stop": {
          "type": "stop",
          "stopwords": "_english_"
        },
        "english_stemmer": {
          "type": "stemmer",
          "language": "light_english"
        },
        "support_synonyms": {
          "type": "synonym_graph",
          "synonyms_set": "support_domain",
          "updateable": true
        }
      },
      "analyzer": {
        "support_index_analyzer": {
          "type": "custom",
          "char_filter": ["html_strip_char", "tech_synonyms_char"],
          "tokenizer": "standard",
          "filter": [
            "icu_normalizer",
            "icu_folding",
            "english_stop",
            "english_stemmer"
          ]
        },
        "support_search_analyzer": {
          "type": "custom",
          "char_filter": ["tech_synonyms_char"],
          "tokenizer": "standard",
          "filter": [
            "icu_normalizer",
            "icu_folding",
            "english_stop",
            "english_stemmer",
            "support_synonyms"
          ]
        }
      }
    }
  }
}
```

Points worth naming:

- **Separate index and search analyzers** — index-time does NOT expand synonyms (they run at query time as a `synonym_graph` filter, keeping IDF correct); search-time does. Both apply the same normalisation and stemming so terms match.
- **`light_english` stemmer** over Porter's `english` — less aggressive; keeps enough word-form distinction to avoid over-stemming class-related terms (`connect` vs. `connection` vs. `connector` are separated by KStem but merged by Porter).
- **`icu_normalizer` and `icu_folding`** on every field with user-typed input; users often paste unaccented text into accented corpora.
- **`updateable: true` on the synonym filter** — allows the synonym set to be reloaded without index recreation ([Elastic docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-with-synonyms.html#synonyms-with-search-analyzer)). Only allowed on search-time analyzers.
- **Char-filter synonyms** (`"C# => csharp"`) run *before* tokenisation and pre-fix things the standard tokenizer would otherwise butcher (`C#` would tokenise to `c`). Do NOT put these in the synonym filter — they need to happen character-side.

### A CJK-ready analyzer

For a Japanese-language corpus:

```json
PUT /jp_articles
{
  "settings": {
    "analysis": {
      "filter": {
        "kuromoji_pos_stop": {
          "type": "kuromoji_part_of_speech",
          "stoptags": [ "助詞-格助詞-一般", "助詞-終助詞" ]
        }
      },
      "analyzer": {
        "jp_index_analyzer": {
          "type": "custom",
          "tokenizer": "kuromoji_tokenizer",
          "filter": [
            "kuromoji_baseform",
            "kuromoji_pos_stop",
            "ja_stop",
            "kuromoji_stemmer",
            "lowercase",
            "icu_normalizer"
          ]
        }
      }
    }
  }
}
```

Points worth naming:

- **`kuromoji_tokenizer`** — segments Japanese into morphemes correctly. `standard` would fail.
- **`kuromoji_baseform`** — normalises verb conjugations to dictionary form (行った → 行く).
- **`kuromoji_part_of_speech`** — POS-based filtering; drops particles that are the CJK equivalent of stopwords.
- **`icu_normalizer`** — normalises full-width vs. half-width digits/latin, katakana width, and other CJK-specific normalisation artefacts.

## Testing analyzers: the `_analyze` API

Every analyzer decision needs to be verified against actual expected inputs. The [`_analyze` API](https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-analyze.html) (Elasticsearch/OpenSearch) runs the analyzer over a supplied string and returns the tokens it produces:

```
POST /support_articles/_analyze
{
  "analyzer": "support_search_analyzer",
  "text": "How do I connect a C# app to Möller's SQL Server?"
}
```

The response shows each token with its position and offsets. This is the fastest debugging loop for analyzers: paste a query, look at the tokens, fix the char-filter / tokenizer / filter that produced the wrong output.

Solr has an equivalent [`analysis-field` and `analysis-fieldname` request handler](https://solr.apache.org/guide/solr/latest/query-guide/analysis-screen.html) exposed in the Admin UI's Analysis panel. Vespa exposes analyzer inspection through `vespa-visit-target` and its debugging tools; the workflow is functionally equivalent.

The rule of thumb: **for every analyzer you deploy, keep a `_analyze` scratch file in the repo with 5–10 canonical queries and their expected tokenisation**. When someone proposes an analyzer change, run the scratch file; the diff tells you what breaks.

## Common failure modes to name

Three failure modes that appear in every non-trivial deployment eventually:

- **Mismatch between index-time and search-time analyzers.** If the index tokenises `"connections"` → `"connect"` (stemmed) but the query analyzer does not stem, the query never matches. Always define both, and only diverge deliberately (e.g., synonyms only at query time).
- **Reindexing when you change the analyzer.** Analyzer changes only apply to *newly indexed* documents. Every analyzer change requires either a reindex or a dual-index cutover ([chapter 8 of mod-104](../mod-104-vector-databases-and-ann-tuning/08-index-versioning-and-dual-index-cutover.md) — the pattern is the same for lexical indexes). Design your analyzer with reindex cost in mind.
- **Synonyms breaking phrase queries.** Multi-word synonyms with the older `synonym` filter shift phrase-position offsets and break `match_phrase` queries silently. Always use `synonym_graph`.

## What is NOT in this chapter (and where it goes)

- **Field mappings, boosting, function scoring** — chapter 3.
- **The dense-vector fields alongside these text fields** — chapter 4.
- **LTR features derived from analyzed text** — chapter 5.
- **How the analyzer choice interacts with hybrid retrieval quality** — mod-106 revisits this from the fusion side.

## What you should be able to do after this chapter

- Name the character-filter → tokenizer → token-filter pipeline shape and give an example of each component.
- Pick a language-appropriate analyzer for an English, Japanese, or Chinese corpus and defend the choice.
- Choose between index-time and query-time synonyms and know why `synonym_graph` is the correct filter to use.
- Add `icu_normalizer` and `icu_folding` to any multi-lingual or user-typed-input analyzer and explain what problem each solves.
- Test an analyzer end-to-end with `_analyze` and diagnose a broken mismatch between index-time and query-time tokenisation.

## Further reading

- Elasticsearch: [*Anatomy of an analyzer*](https://www.elastic.co/guide/en/elasticsearch/reference/current/analyzer-anatomy.html) — the reference for the pipeline shape.
- Elasticsearch: [*Language analyzers*](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html) — the full list of built-in language analyzers.
- Elasticsearch: [*ICU analysis plugin*](https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-icu.html) — Unicode-aware normalisation.
- Solr: [*Language analysis*](https://solr.apache.org/guide/solr/latest/indexing-guide/language-analysis.html) — the Solr-side list of language filters.
- Elasticsearch: [*Search with synonyms*](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-with-synonyms.html) — modern synonym-API workflow.
- Porter, M. F. (1980). *An algorithm for suffix stripping.* Program, 14(3), 130–137. The paper behind the Porter stemmer: [tartarus.org/martin/PorterStemmer](https://tartarus.org/martin/PorterStemmer/def.txt).
- Snowball stemmers: [snowballstem.org](https://snowballstem.org/) — Porter's later multi-language stemmer family.
- Unicode Text Segmentation ([UAX #29](https://unicode.org/reports/tr29/)) — the word-break specification behind the standard tokenizer.
