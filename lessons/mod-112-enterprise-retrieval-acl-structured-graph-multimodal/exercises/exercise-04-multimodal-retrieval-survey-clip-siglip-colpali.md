# Exercise 04 — Multimodal Retrieval Survey: CLIP, SigLIP, and ColPali on the Same Corpus

**Estimated effort:** ~3 hours. Python, a small mixed corpus (~20–40 visually-rich PDFs plus ~50–100 stand-alone images), Qdrant or Weaviate (either supports multi-vector), a GPU is nice but not required for these model sizes, and hand-labeled judgment sets for two retrieval tasks.

## Objective

Run three multimodal retrievers side-by-side on the same corpus — an OpenCLIP `ViT-B/32` baseline, a SigLIP checkpoint, and a ColPali document-page retriever — and measure recall@k / MRR on two hand-labeled judgment sets (one for image retrieval, one for document-page retrieval). Learn where each family sits on the frontier by feel, not just by leaderboard.

By the end you have:

- `corpus/pdfs/` — 20–40 visually-rich PDFs (financial reports, slide decks, product data sheets — anything with tables, figures, and mixed layouts).
- `corpus/pages/` — the PDFs rendered to per-page PNGs at 150–200 DPI.
- `corpus/images/` — 50–100 stand-alone images (product photos, diagrams, screenshots).
- `indexers/clip_index.py` — embed images and pages with OpenCLIP.
- `indexers/siglip_index.py` — embed the same with SigLIP.
- `indexers/colpali_index.py` — embed pages with ColPali (multi-vector per page); optionally embed images too.
- `retrievers/{clip,siglip,colpali}.py` — the three retrieval surfaces.
- `evals/image_retrieval.py` — 30–60 image queries with hand-labeled gold image IDs; recall@5, recall@10, MRR.
- `evals/document_page_retrieval.py` — 30–60 document-page queries with hand-labeled gold page IDs; same metrics.
- `RESULTS.md` — the comparison table, one paragraph per encoder on where it earned its keep, and the pick-per-modality you'd deploy.

## Prerequisites

- Chapter 6 read.
- mod-104 chapter 5 (multi-vector index configs) as background — ColPali's per-page multi-vector requires an index type that supports it.
- Python 3.10+; `pip install open_clip_torch sentence-transformers colpali-engine pdf2image pillow qdrant-client`. `pdf2image` needs a system-level `poppler` install (`apt install poppler-utils` on Debian/Ubuntu, `brew install poppler` on macOS).
- Qdrant running locally (`docker run -p 6333:6333 qdrant/qdrant`) — its [multi-vector index](https://qdrant.tech/documentation/concepts/vectors/#multivectors) supports ColPali's MaxSim scoring natively. Weaviate is also fine if you prefer.
- CPU is sufficient for OpenCLIP ViT-B/32 and SigLIP base. ColPali is happier on a GPU; the paper's checkpoints work on CPU for a 20–40 PDF corpus with patience (a few minutes to index).

## Steps

### Step 1 — Assemble the corpus (25 minutes)

**PDFs.** 20–40 documents that make text-only extraction leave value on the table: SEC 10-Ks with balance sheets, technical data sheets with wiring diagrams, marketing slide decks with mixed layouts, invoices with table structure. Public sources are fine — an [SEC EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch) search turns up plenty.

**Stand-alone images.** 50–100 product photos, diagrams, screenshots, whatever. Public datasets like [Unsplash lite](https://unsplash.com/data), your own screenshots, or any collection you can license.

Render every PDF page to a PNG:

```python
from pdf2image import convert_from_path
from pathlib import Path

for pdf in Path("corpus/pdfs").glob("*.pdf"):
    for i, img in enumerate(convert_from_path(pdf, dpi=180)):
        img.save(f"corpus/pages/{pdf.stem}__p{i:03d}.png")
```

Verify by opening a couple of the rendered pages — you should see readable text, tables, and figures.

### Step 2 — Author the two judgment sets (30 minutes)

**Image retrieval set** (`evals/image_queries.jsonl`): 30–60 natural-language queries with a small set of gold image IDs each.

```json
{"id": "img-q001", "query": "a wooden desk with a laptop and a coffee mug", "gold_image_ids": ["img-0032", "img-0087"]}
```

Cover shapes CLIP-family models can and cannot do — some straightforward object queries, some spatial ("dog left of cat"), some counting ("three people"), some fine-grained ("Golden Retriever puppy"), a few negations ("laptop **without** stickers"). You want the negation / counting cases to *fail* — that's the empirical demonstration of the "bag of concepts" limitation.

**Document-page retrieval set** (`evals/page_queries.jsonl`): 30–60 natural-language queries with gold `pdf__pXXX` IDs.

```json
{"id": "page-q001", "query": "the quarterly revenue table showing FY23 vs FY22 by segment", "gold_page_ids": ["contoso_10k_2023__p042"]}
```

Cover: prose-heavy pages (which text extraction handles), table-heavy pages (where OCR often fails), figure-heavy pages (where ColPali should shine), pages with unusual layouts.

Labels take real time — plan on 20–30 minutes for this step. Small, hand-labeled sets are worth more than large, auto-labeled ones for a survey like this.

### Step 3 — CLIP baseline (25 minutes)

`indexers/clip_index.py` with [OpenCLIP](https://github.com/mlfoundations/open_clip):

```python
import open_clip
import torch
from PIL import Image
from qdrant_client import QdrantClient, models

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

def embed_image(path):
    with torch.no_grad():
        return model.encode_image(preprocess(Image.open(path).convert("RGB")).unsqueeze(0))[0].cpu().numpy()

def embed_text(text):
    with torch.no_grad():
        return model.encode_text(tokenizer([text]))[0].cpu().numpy()

client = QdrantClient("localhost", port=6333)
client.recreate_collection("clip_images",
    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE))

# Index images + pages under the same collection with a modality tag
for path in list_all_images_and_pages():
    vec = embed_image(path)
    client.upsert("clip_images", [models.PointStruct(
        id=path_to_id(path), vector=vec.tolist(),
        payload={"path": str(path), "modality": "image" if "images/" in str(path) else "page"}
    )])
```

At query time, embed the text and search with the modality filter matching the eval set.

### Step 4 — SigLIP variant (20 minutes)

Same shape, different checkpoint. SigLIP is also supported by OpenCLIP; alternatively use the Hugging Face `google/siglip-base-patch16-224` via `transformers`:

```python
from transformers import AutoModel, AutoProcessor
import torch

siglip = AutoModel.from_pretrained("google/siglip-base-patch16-224")
sig_proc = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

def sig_embed_image(path):
    inputs = sig_proc(images=Image.open(path).convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        return siglip.get_image_features(**inputs)[0].numpy()

def sig_embed_text(text):
    inputs = sig_proc(text=[text], return_tensors="pt", padding="max_length")
    with torch.no_grad():
        return siglip.get_text_features(**inputs)[0].numpy()
```

Index into a `siglip_images` collection with the same modality tag. The output embedding dim is different (768 for base); size the collection accordingly.

### Step 5 — ColPali for document pages (30 minutes)

`indexers/colpali_index.py` using the [`colpali-engine`](https://github.com/illuin-tech/colpali) library:

```python
from colpali_engine.models import ColPali, ColPaliProcessor
import torch

col_model = ColPali.from_pretrained("vidore/colpali-v1.2", torch_dtype=torch.float32).eval()
col_proc = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

def col_embed_page(path):
    inputs = col_proc.process_images([Image.open(path).convert("RGB")])
    with torch.no_grad():
        return col_model(**inputs)[0].cpu().numpy()   # shape: (n_patches, dim) — MULTI-VECTOR

def col_embed_query(text):
    inputs = col_proc.process_queries([text])
    with torch.no_grad():
        return col_model(**inputs)[0].cpu().numpy()   # shape: (n_query_tokens, dim)
```

Store in Qdrant with the [multi-vector configuration](https://qdrant.tech/documentation/concepts/vectors/#multivectors) and MaxSim comparator:

```python
client.recreate_collection("colpali_pages",
    vectors_config=models.VectorParams(
        size=col_model.config.hidden_size,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
    ))

for path in list_pdf_pages():
    vecs = col_embed_page(path)
    client.upsert("colpali_pages", [models.PointStruct(
        id=path_to_id(path), vector=vecs.tolist(), payload={"path": str(path)}
    )])
```

At query time, submit the query's multi-vector; Qdrant computes MaxSim under the hood.

If the ColPali checkpoint has moved by the time you work through this, check the [ViDoRe leaderboard on Hugging Face](https://huggingface.co/vidore) for a current recommendation (ColQwen2, ColSmol, etc.). The API shape is stable across the family.

### Step 6 — Evaluate both retrieval tasks (25 minutes)

`evals/image_retrieval.py`:

For each `image_queries.jsonl` entry, retrieve top-10 from each of CLIP, SigLIP, and (optionally) ColPali. Compute:

- **Recall@5** = `|gold ∩ top5| / |gold|`, averaged over queries.
- **Recall@10** = same at k=10.
- **MRR** = mean of `1 / rank_of_first_gold` (0 if no gold in top-10).

`evals/document_page_retrieval.py`: same shape, on `page_queries.jsonl`, against page indexes for each of CLIP (pages-as-images), SigLIP (pages-as-images), and ColPali (its natural home).

You should see roughly: on images, CLIP and SigLIP within a few points of each other with SigLIP a hair ahead on average; on pages, ColPali substantially ahead on the table-heavy and figure-heavy subsets, roughly on-par or worse than CLIP on plain-prose pages. Your specific numbers will depend on the corpus, but the shape of the delta is the survey.

### Step 7 — A small text-only baseline for the page task (15 minutes)

For the document-page task only, add one more retriever as a reality check: OCR the pages with `pytesseract` (or reuse mod-102's text-extraction pipeline), chunk to page-level, embed with a text embedder (`all-MiniLM-L6-v2`), and retrieve the same top-10. This is the "would we have been fine sticking with text?" baseline.

On visually-rich pages, ColPali should beat this. On prose-heavy pages, the text baseline should be close. Report both.

### Step 8 — Write up `RESULTS.md` (15 minutes)

Publish:

- A recall@5 / recall@10 / MRR table for `(retriever × task × subset)`. Sub-buckets for the page task: `prose-heavy`, `table-heavy`, `figure-heavy` (assign each labeled page to a bucket during step 2).
- One paragraph per encoder on where it earned its keep on your corpus.
- One paragraph on the negation / counting / spatial queries in the image set — did every encoder fail, or did any handle a case? (Do not over-generalize from small numbers; note the sample size.)
- A **starter-stack recommendation** for a corpus like yours: which encoder(s) you'd deploy, and how you'd route between them (chapter 7's router preview).
- Storage overhead — total on-disk size per index; ColPali will be substantially larger (multi-vector per page). Numbers matter for chapter 7's routing budget.

## Acceptance criteria

- Three (or four, counting the text-only page baseline) retrievers implemented and callable against the same corpus.
- Both judgment sets are ≥ 30 queries, hand-labeled with gold IDs.
- The document-page eval reports per-subset (`prose-heavy` / `table-heavy` / `figure-heavy`) numbers, not just an aggregate.
- `RESULTS.md` publishes the recall/MRR table, per-encoder qualitative summary, and a stack recommendation.
- Storage overhead per index is reported (bytes on disk or vector count × dimension).

## Stretch goals

- **DSE as a fourth encoder.** [DSE](https://arxiv.org/abs/2406.11251) is the single-vector cousin of ColPali; wire it in and see how much recall you trade for the smaller index.
- **Long-CLIP.** Swap the CLIP checkpoint for a [Long-CLIP](https://arxiv.org/abs/2403.15378) variant to see whether the 248-token context helps on paragraph-length queries.
- **Audio.** Add 20–30 short audio clips (podcasts, meeting excerpts). Whisper-transcribe them; add them to the CLIP text index as text chunks with a `modality: "audio_transcript"` tag; measure retrieval on 10 audio-search queries. Optionally add CLAP as a comparison.
- **Video.** Sample keyframes from 5–10 short videos every 2 seconds; add to the CLIP image index; add Whisper transcripts of the same videos to the text index; measure the three-pattern hybrid recommended in chapter 6.
- **Cross-encoder rerank.** After multimodal top-10, feed pairs to a vision-capable LLM (GPT-4o vision, Claude Sonnet with images) as a reranker; measure the recall@5 lift. Note the latency and cost.
- **Domain-adapted checkpoint.** If your corpus is in-domain for a specialized encoder (BioMedCLIP for medical imagery, RemoteCLIP for satellite), compare the generalist checkpoint to the specialist on that subset.

## Notes

- **This is a survey exercise, not a benchmark chase.** The goal is to feel where each encoder sits, not to publish state-of-the-art numbers. If your absolute numbers are lower than [ViDoRe leaderboard](https://huggingface.co/vidore) results, that's expected — those are averaged over a much larger, curated benchmark; your corpus is small and specific.
- **Multi-vector storage is substantial.** ColPali produces dozens of vectors per page; a 40-PDF corpus at ~20 pages each with 1024 patches × 128 dim per patch is roughly 100 MB of vectors. Fine at this scale, worth noting for production.
- **PDF-render DPI matters.** 150 DPI is a good default. Below 100 DPI the vision encoder loses fine text; above 200 DPI you're paying storage / compute for no accuracy gain in most cases.
- **Fine-tuning the encoders is out of scope.** If your corpus is far out-of-domain (medical imagery, technical schematics), the pretrained checkpoints will be a floor rather than the ceiling. Fine-tuning is `fine-tuning-engineer`'s territory; note the gap in `RESULTS.md` and cite the boundary.
- **Chapter 6's "shared embedding space is usually a bad idea in practice" is worth internalizing.** You will notice that CLIP text embeddings and SigLIP text embeddings do not cross-index well; keep one index per encoder.
