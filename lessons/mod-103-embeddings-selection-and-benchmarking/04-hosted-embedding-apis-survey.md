# Chapter 4 — Hosted Embedding APIs: OpenAI, Cohere, Voyage, Google, Bedrock

## Why this chapter exists

Two of the four sources you shortlist in chapter 7 will almost always be hosted APIs. The reason is not quality — the top open models are competitive — it is *operational surface area*. You do not run a GPU fleet; you do not manage a batch-inference throughput SLO; you do not decide when to upgrade CUDA. You pay per million tokens and you focus on the retrieval logic.

The trade-off is the one every hosted-service story has: cost at scale, per-call latency floor, vendor lock-in, quota and rate limits, and a compliance / data-residency story you have to check against your corpus's policy. This chapter is the honest tour of the five main hosted embedding providers you will actually be asked about in an interview or in an architecture review.

Chapter 5 is the parallel tour of self-hosted open models. Chapter 6 puts the two on the same axes (latency, cost, dimensionality, quality-per-dollar) with the methodology to compare them.

> **Note on prices and rate limits.** The pricing tiers, quotas, and model versions below change frequently. Every price cited here is a rough order of magnitude; go to the vendor pricing page linked in "Further reading" before you build a cost model for a proposal.

## OpenAI text-embedding-3

**Models.** `text-embedding-3-small` (1536 dim, but reducible via the `dimensions` parameter) and `text-embedding-3-large` (3072 dim, reducible). Both support up to 8191 input tokens. Introduced by OpenAI's *[New embedding models and API updates](https://openai.com/blog/new-embedding-models-and-api-updates)* announcement (January 2024). The prior generation (`text-embedding-ada-002`) is deprecated for new work but still callable.

**What is interesting about them.**

- **Matryoshka dimensions.** Both models were trained with Matryoshka Representation Learning (MRL — chapter 8), so you can truncate the vector to any dimension via `dimensions=<n>` and retain most of the quality. `text-embedding-3-small` truncated to 512 dims often retains most of its retrieval quality; the cost per token is unchanged (still full-length embedding), but you pay a fraction of the storage and ANN cost downstream.
- **A single-family API surface.** No `input_type` distinction between query and document — you send one string, you get one vector. Simpler wrapper than Cohere or Voyage; also less lever to tune per-role.
- **Latency and throughput.** Batched at up to 2048 inputs per call; per-batch latency on `text-embedding-3-small` is typically low-hundreds of ms on well-populated batches. For a serious embed-a-corpus run you want batch size ≥ 256.

**API shape.**

```python
from openai import OpenAI
client = OpenAI()

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=["chunk 1 text", "chunk 2 text", "..."],
    dimensions=512,                             # optional; MRL truncation
    encoding_format="float",                    # or "base64" for size savings on the wire
)
vecs = [np.asarray(e.embedding) for e in resp.data]
```

**When it fits.**

- You already run against OpenAI for generation and want one vendor's rate-limits, invoicing, and status page.
- You want MRL out of the box without picking a training recipe.
- Your corpus is mostly English (multilingual is supported but Cohere / Voyage / Google are stronger on non-English; check MTEB per-language rankings before committing).

**When it does not.**

- Your corpus cannot leave a specific region or trust boundary. OpenAI's default endpoint is US-based; the [OpenAI enterprise data-processing addendum](https://openai.com/enterprise-privacy) covers standard commercial use, but data-residency / air-gap requirements need Azure OpenAI or a self-hosted model, not the OpenAI direct API.
- You need >1M free tokens/month for a hobbyist stack — hosted APIs have a per-token cost that adds up on large corpora (see chapter 6 for the cost formulas).

**Docs.** [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings), [embeddings API reference](https://platform.openai.com/docs/api-reference/embeddings), [rate limits](https://platform.openai.com/docs/guides/rate-limits).

## Cohere Embed v3

**Models.** `embed-english-v3.0`, `embed-english-light-v3.0`, `embed-multilingual-v3.0`, `embed-multilingual-light-v3.0`. Native dim 1024 (light: 384). Max 512 tokens per input — much shorter than OpenAI's 8k or Voyage's 32k; align your chunker to it (mod-102 chapter 5's chunk-size trade-off is where you set this). Announced in Cohere's *[Introducing Embed v3](https://cohere.com/blog/introducing-embed-v3)* post (November 2023).

**What is interesting about them.**

- **Role-typed inputs (`input_type`).** Every call must specify one of `"search_document"`, `"search_query"`, `"classification"`, `"clustering"`. The model produces subtly different embeddings for each role — same text, different `input_type`, different vector. Passing `input_type="search_document"` at index time and `input_type="search_query"` at query time is what the model was trained to expect and typically gives measurably higher retrieval scores than passing the same type for both.
- **Compression-aware training.** Cohere published that Embed v3 was trained with quantization in mind, so int8 and binary versions of the vectors retain quality well (chapter 8). The API can return `embedding_types=["int8"]` or `["binary"]` alongside `"float"`.
- **Multilingual support.** `embed-multilingual-v3.0` covers 100+ languages in a shared embedding space — you can index English and query in Spanish and retrieve the English docs. Voyage and Google are the closest competitors on this; OpenAI is weaker per current public benchmarks but adequate.

**API shape.**

```python
import cohere
co = cohere.Client()

# Indexing:
resp = co.embed(
    texts=["doc chunk 1", "doc chunk 2"],
    model="embed-english-v3.0",
    input_type="search_document",
    embedding_types=["float", "int8"],           # get both — pick which one to store
)

# Querying:
q_resp = co.embed(
    texts=["natural-language question"],
    model="embed-english-v3.0",
    input_type="search_query",
    embedding_types=["float"],
)
```

**When it fits.**

- Multilingual corpus or cross-lingual query.
- You care about int8 / binary quantization on the vector store (chapter 8) and want the provider's own quantization rather than post-hoc.
- The 512-token limit matches your chunk size (or you can shape the chunker to it).

**When it does not.**

- Chunks are ≫ 512 tokens and re-chunking is unacceptable.
- You want a single-role API with no per-call routing logic.

**Docs.** [Embed API reference](https://docs.cohere.com/reference/embed), [embed guide](https://docs.cohere.com/docs/embeddings), [Embed v3 announcement](https://cohere.com/blog/introducing-embed-v3), [pricing](https://cohere.com/pricing).

## Voyage AI (voyage-3, voyage-3-lite, voyage-large-2, voyage-code-2, voyage-law-2, voyage-finance-2, voyage-multilingual-2)

**Models.** `voyage-3` (1024 dim, 32k context) is the current general-purpose recommendation as of the writing of Voyage's *[voyage-3 announcement](https://blog.voyageai.com/2024/09/18/voyage-3/)* (September 2024). A `-lite` variant trades some quality for cost. Voyage also ships **domain-specialized** models — `voyage-code-3`, `voyage-law-2`, `voyage-finance-2`, `voyage-multilingual-2` — trained on domain corpora to lift retrieval quality where those domains dominate.

**What is interesting about them.**

- **Long context window.** 32k tokens is the largest of the mainstream hosted embedders. If your chunks are unusually large (whole articles, long code files), Voyage removes a chunking constraint that OpenAI (8k) and Cohere (512) impose.
- **Domain-tuned models.** For a codebase, `voyage-code-3` will usually beat any general-purpose embedder without your having to fine-tune. Same for legal (`voyage-law-2`) or financial (`voyage-finance-2`) corpora. This is the closest a hosted vendor gets to "we did the fine-tune for you" for a specific vertical.
- **Role-typed inputs.** Like Cohere, Voyage uses `input_type` (`"query"` / `"document"`) to route to the right embedding head. Skip it at your peril.
- **Distribution via Anthropic and via Amazon Bedrock.** [Anthropic's docs](https://docs.anthropic.com/en/docs/build-with-claude/embeddings) explicitly point to Voyage as their recommended embedding provider (Anthropic does not ship first-party embeddings). Voyage models are also available on [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) — the same models under AWS's IAM, VPC, and CloudTrail surface if you need them.

**API shape (Voyage direct).**

```python
import voyageai
vo = voyageai.Client()

result = vo.embed(
    ["chunk 1", "chunk 2"],
    model="voyage-3",
    input_type="document",
    truncation=True,
)
vecs = result.embeddings
```

**API shape (Voyage via Bedrock).** Standard Bedrock InvokeModel; the exact body schema follows the model card. Anthropic's [Voyage on Bedrock announcement](https://www.anthropic.com/news/voyage-ai-embeddings-on-amazon-bedrock) is the current pointer.

**When it fits.**

- You already run in AWS and want an embedding call to inherit IAM / VPC-endpoint / CloudTrail rather than adding an outside-AWS network path.
- Your corpus is code, legal, finance, or multilingual and Voyage's domain model matches.
- Chunks are long and Cohere's 512 limit is a constraint.

**When it does not.**

- General-purpose English corpus, no AWS preference — OpenAI and Cohere are as good and often simpler to onboard.

**Docs.** [Voyage embeddings docs](https://docs.voyageai.com/docs/embeddings), [Anthropic's Voyage recommendation](https://docs.anthropic.com/en/docs/build-with-claude/embeddings), [voyage-3 blog post](https://blog.voyageai.com/2024/09/18/voyage-3/), [pricing](https://www.voyageai.com/pricing).

## Google text-embedding-005 (Vertex AI)

**Models.** `text-embedding-005` (English) and `text-multilingual-embedding-002` are the current recommended defaults on Google's Vertex AI text embeddings surface. Native output dim is 768; supports Matryoshka truncation to 256 or 128 via the `outputDimensionality` parameter. Max ~2048 input tokens.

**What is interesting about them.**

- **Task-typed via `task_type`.** Google's API takes a `task_type` (`RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`, `QUESTION_ANSWERING`, `FACT_VERIFICATION`, `CODE_RETRIEVAL_QUERY`). More task types than Cohere; the routing surface is finer.
- **Native GCP integration.** Vertex AI Vector Search, BigQuery ML, and Vertex AI Search consume these embeddings without re-shaping. If your stack is GCP-native, the wiring is short.
- **Matryoshka via `outputDimensionality`.** The `text-embedding-005` model card documents that truncating to 256 dims retains most of the retrieval quality — same MRL story as OpenAI, exposed differently.

**API shape (Vertex AI Python SDK).**

```python
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

model = TextEmbeddingModel.from_pretrained("text-embedding-005")

inputs = [
    TextEmbeddingInput(text="chunk text", task_type="RETRIEVAL_DOCUMENT", title="doc title"),
]
resp = model.get_embeddings(inputs, output_dimensionality=768)
vec = resp[0].values
```

**When it fits.**

- GCP-hosted product, want IAM / VPC-SC / audit-logs on the embedding call.
- Explicit distinct `task_type`s for retrieval vs. classification vs. question-answering matches your product surface.

**When it does not.**

- You are not on GCP — the SDK and auth story is heavier to onboard than a `pip install openai`.
- You need very long context (2k tokens vs. Voyage's 32k).

**Docs.** [Vertex AI Text embeddings API](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings), [text-embedding-005 model card](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api), [supported task_types](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#api_changes_to_models_released_on_or_after_august_2023).

## Amazon Bedrock: Titan Embed and the Voyage passthrough

Bedrock is not itself an embedding model — it is a hosting layer. What you get from Bedrock:

- **`amazon.titan-embed-text-v2:0`** — Amazon's own embedder (1024 dim default, MRL-reducible to 512 or 256, up to 8k tokens). Documented in the [Bedrock model reference](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html).
- **`amazon.titan-embed-image-v1`** — image-text embeddings for multimodal (out of scope here; mod-112 covers multimodal retrieval).
- **`cohere.embed-english-v3` / `cohere.embed-multilingual-v3`** — the same Cohere models as above, served through Bedrock's IAM / VPC surface.
- **`voyage.voyage-3` / `voyage.voyage-multilingual-2` / `voyage.voyage-code-3` / `voyage.voyage-finance-2` / `voyage.voyage-law-2`** — the Voyage models on Bedrock. See Anthropic's [*Voyage AI on Bedrock*](https://www.anthropic.com/news/voyage-ai-embeddings-on-amazon-bedrock) announcement.

**Why use Bedrock over the vendor's direct API.**

- Every call is IAM-scoped and CloudTrail-audited. In many regulated shops this is the difference between "shippable" and "does not clear infosec review."
- You can put a VPC endpoint on Bedrock and never egress to the public internet.
- One invoicing surface for Cohere + Voyage + Anthropic + Meta + Mistral — useful for procurement.

**Why not.**

- Bedrock adds one more service between your code and the model — an extra layer to debug when latency spikes.
- Pricing is not always identical to the vendor's direct API — sometimes cheaper, sometimes with different rate-limit shape. Check before assuming.
- Region availability lags the vendor's direct API for newer models.

**API shape (Bedrock InvokeModel).**

```python
import json, boto3
brt = boto3.client("bedrock-runtime")

body = json.dumps({"inputText": "chunk text", "dimensions": 512, "normalize": True})
resp = brt.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=body,
    accept="application/json",
    contentType="application/json",
)
vec = json.loads(resp["body"].read())["embedding"]
```

For Cohere and Voyage on Bedrock, the request body schema matches the vendor's original — Bedrock does not remap fields. Check the vendor's docs for the schema, wrap it in `invoke_model`, done.

## The comparison table

The one you will draw on the whiteboard when the architect asks "which hosted embedder are we using?" (numbers are order-of-magnitude and will shift; check the vendor pages before quoting):

| Provider / Model | Dim (default → MRL-min) | Max tokens | Role types | Multilingual | Deployed on |
|---|---|---|---|---|---|
| OpenAI `text-embedding-3-small` | 1536 → 256 or lower | 8191 | none | good (English is strongest) | OpenAI API, Azure OpenAI |
| OpenAI `text-embedding-3-large` | 3072 → 256 or lower | 8191 | none | good | OpenAI API, Azure OpenAI |
| Cohere `embed-english-v3.0` | 1024 (no MRL) | 512 | search_query / search_document / classification / clustering | English-only variant | Cohere API, Bedrock, Sagemaker JumpStart |
| Cohere `embed-multilingual-v3.0` | 1024 (no MRL) | 512 | as above | 100+ languages | Cohere API, Bedrock |
| Voyage `voyage-3` | 1024 (no MRL) | 32000 | query / document | limited (see `voyage-multilingual-2`) | Voyage API, Bedrock |
| Voyage `voyage-code-3` / `voyage-law-2` / `voyage-finance-2` | 1024 (no MRL) | 32000 | query / document | English + domain | Voyage API, Bedrock |
| Google `text-embedding-005` | 768 → 256 | 2048 | RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT / … (8 task types) | English (see `text-multilingual-embedding-002` for other languages) | Vertex AI |
| Amazon `titan-embed-text-v2:0` | 1024 → 512 / 256 | 8192 | none | supported | Bedrock only |

## Practical adoption rules

- **Pin the model version.** All five providers version their embedding models. When a vendor deprecates a model (or introduces a subtle behavior change), your index becomes stale. Store the exact model name and API version in the index metadata (`embedding_model_id`, `embedding_model_revision`) — mod-111 makes this a first-class field of the index-version key. See chapter 8 for the coupling with dim + quantization.
- **Do not re-embed on every query.** Cache query embeddings. Repeat queries are common; the embedding API call is the expensive step per query.
- **Batch aggressively.** Every provider's per-call latency is dominated by RTT + first-token latency, not per-input compute. Batch sizes of 64–512 typically give 5–20× throughput improvement over one-at-a-time.
- **Rate limits are real.** OpenAI's tier system, Cohere's per-minute quotas, Voyage's `Retry-After`, Google's per-region QPS ceilings — all of them will throttle you during a corpus embed. Implement exponential backoff *and* have a plan for a multi-hour embed job that respects the ceiling.
- **Compliance and data-residency.** Do not assume any of these providers is approved for your data. Check your infosec / legal team's approved-vendor list, the vendor's data-processing agreement, the region the endpoint runs in, and whether the vendor retains inputs for model training (all five let you opt out; the default varies). Do this before you embed the corpus.

## What is NOT in this chapter (and where it goes)

- **Self-hosted open models.** Chapter 5.
- **Comparing hosted vs. self-hosted on cost/latency/quality.** Chapter 6.
- **The domain-fit benchmark that would tell you which of these to actually pick.** Chapter 7.
- **Quantization on the vector store side.** Chapter 8.
- **Reranker APIs (Cohere Rerank, Voyage Rerank, Jina Rerank).** mod-107 owns the reranker slot; those APIs are shaped differently (they return scores, not vectors) and belong in that module's model survey.

## What you should be able to do after this chapter

- Name the five hosted embedders in scope for this track and give the default dim, max tokens, and role-type surface of each.
- Wrap any of them in the MTEB-compatible adapter shape from chapter 3.
- Explain why Cohere and Voyage require `input_type` and what happens if you skip it (a measurable quality drop; the model was trained on typed inputs).
- Point out where each provider is served (native API, Bedrock, Vertex, Azure) and pick the surface that matches your compliance posture.
- Name the operational levers (batch size, rate limits, pinned version, opt-out of training) you must handle before the first production corpus embed.

## Further reading

- OpenAI, [*Embeddings guide*](https://platform.openai.com/docs/guides/embeddings), [*New embedding models and API updates*](https://openai.com/blog/new-embedding-models-and-api-updates) (announcement of `text-embedding-3-*`).
- Cohere, [*Embed API reference*](https://docs.cohere.com/reference/embed), [*Embeddings guide*](https://docs.cohere.com/docs/embeddings), [*Introducing Embed v3*](https://cohere.com/blog/introducing-embed-v3).
- Voyage AI, [*Embeddings docs*](https://docs.voyageai.com/docs/embeddings), [*voyage-3 announcement*](https://blog.voyageai.com/2024/09/18/voyage-3/).
- Anthropic, [*Embeddings (recommend Voyage AI)*](https://docs.anthropic.com/en/docs/build-with-claude/embeddings), [*Voyage AI embeddings on Amazon Bedrock*](https://www.anthropic.com/news/voyage-ai-embeddings-on-amazon-bedrock).
- Google Cloud, [*Get text embeddings*](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings), [*Text embeddings API — task types*](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api).
- AWS, [*Amazon Titan Text Embeddings*](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html), [*Bedrock supported models*](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html), [*Bedrock pricing*](https://aws.amazon.com/bedrock/pricing/).
