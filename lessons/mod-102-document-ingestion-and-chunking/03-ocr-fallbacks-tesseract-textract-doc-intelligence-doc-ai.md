# Chapter 3 — OCR Fallbacks: Tesseract, AWS Textract, Azure Document Intelligence, GCP Document AI

## Why this chapter exists

Chapter 2 assumed the file in front of you has an honest text layer. Real corpora violate that assumption constantly: scanned contracts, faxed medical records, scanned-then-PDF-wrapped forms, PowerPoint decks flattened to images, screenshots turned into "PDFs" by a well-meaning colleague. Send those through a native-text parser and you get empty strings, ligature line-noise, or — worst of all — a *partial* extraction that looks fine to a truthy check and quietly halves recall in production.

This chapter walks the four OCR back-ends this module targets: local **Tesseract**, and hosted **AWS Textract**, **Azure AI Document Intelligence** (formerly Form Recognizer), and **GCP Document AI**. It sets up the detection heuristic that triggers the fallback, the cost / latency / accuracy trade-off you will measure in `exercise-02`, and the write-once-run-anywhere adapter pattern that lets you swap back-ends without touching the rest of the pipeline.

## Detecting when you need OCR

Chapter 1 sketched a whole-document heuristic (`total_chars / total_pages < 50 → treat as image-only`). Real corpora need **per-page** routing, because born-digital cover pages get bolted onto scanned bodies all the time:

```python
import fitz  # PyMuPDF

def per_page_needs_ocr(path: Path, min_chars_per_page: int = 100) -> list[bool]:
    """Return one bool per page: True if the page's text layer is too thin to trust."""
    verdicts = []
    doc = fitz.open(path)
    for page in doc:
        text = page.get_text("text").strip()
        if len(text) < min_chars_per_page:
            verdicts.append(True)
            continue
        # Second signal: the "text" is present but is ligature line-noise.
        # Real prose has spaces; corrupted extractions often do not.
        ratio_alpha = sum(c.isalpha() for c in text) / max(1, len(text))
        verdicts.append(ratio_alpha < 0.5)
    doc.close()
    return verdicts
```

Two refinements you will want:

- **A rasterization pass per OCR-required page.** Once a page is marked, render it to a PNG (`page.get_pixmap(dpi=300)`) and hand *that* to the OCR back-end. Do not hand the whole PDF to a per-page API and pay to OCR the ten pages that were already fine.
- **A confidence check on the OCR output.** All four back-ends return per-token confidence; discard tokens below ~0.5 and, if a page's mean confidence is below ~0.7, escalate to a stronger back-end or flag for human review. Silent low-confidence extraction is worse than no extraction — the string embeds, the vector stores, the retriever returns it.

## The four back-ends

The four covered here span a range of price points, deployment models, and layout capabilities. There are more (ABBYY, Kofax, MathPix for math-heavy PDFs) but these are the ones the job market and the LlamaIndex / LangChain integrations converge on.

### Tesseract — free, local, run-of-the-mill

[Tesseract](https://tesseract-ocr.github.io/) (Apache-2.0, historically Google, now open-source community) is the default local OCR engine, installed as a system package (`apt install tesseract-ocr` / `brew install tesseract`) and driven from Python via [`pytesseract`](https://github.com/madmaze/pytesseract). It supports ~100+ languages via installable trained-data files (`tesseract-ocr-eng`, `-fra`, `-deu`, ...).

```python
import pytesseract
from PIL import Image

text = pytesseract.image_to_string(Image.open("page-01.png"), lang="eng")
data = pytesseract.image_to_data(
    Image.open("page-01.png"),
    lang="eng",
    output_type=pytesseract.Output.DICT,
)
# data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"]
```

Reach for it when:

- The corpus contains sensitive data that cannot leave your VPC.
- The corpus is small enough that the accuracy gap vs. hosted services will not sink recall.
- The layouts are simple (single-column, standard fonts, no fancy tables). Tesseract's LSTM engine (v4+, and the current v5) is respectable on clean text but noticeably behind hosted services on complex layouts, handwriting, and tables.

The one knob that matters: `--psm` (Page Segmentation Mode). `--psm 3` (default) auto-segments; `--psm 6` treats the input as a single uniform block of text and is more accurate for images that were already cropped to one paragraph. See [Tesseract's PSM docs](https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc#options).

### AWS Textract — hosted, table-aware, per-page pricing

[AWS Textract](https://docs.aws.amazon.com/textract/latest/dg/what-is.html) is Amazon's document-analysis service. Beyond raw text, it exposes `TABLES`, `FORMS` (key-value pairs), `SIGNATURES`, `LAYOUT`, and `QUERIES` (natural-language questions answered from the document) as opt-in feature flags. Pricing is per page, per feature — check the [Textract pricing page](https://aws.amazon.com/textract/pricing/) for current numbers; do not rely on figures a chapter memorializes.

Two API shapes:

- **Sync (`DetectDocumentText`, `AnalyzeDocument`)** — single-page or short (≤ 10 MB / few pages) documents; response is the block tree.
- **Async (`StartDocumentTextDetection` / `StartDocumentAnalysis` → `GetDocumentTextDetection`)** — multi-page PDFs from S3; poll a `JobId` for completion. Async is what you use in a real ingestion pipeline.

```python
import boto3

client = boto3.client("textract")
response = client.analyze_document(
    Document={"Bytes": Path("page-01.png").read_bytes()},
    FeatureTypes=["TABLES", "LAYOUT"],
)
for block in response["Blocks"]:
    if block["BlockType"] == "LINE":
        print(block["Text"], "|", block["Confidence"])
```

Reach for it when:

- You already have AWS credentials wired for the rest of the platform.
- Tables matter and you want them extracted as cell grids, not as flat text.
- Volume is moderate; at large scale you may need to compare the total bill against reserved GPU running Tesseract or a self-hosted OCR model.

### Azure AI Document Intelligence — hosted, prebuilt models, custom model support

[Azure AI Document Intelligence](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview) (renamed from Azure Form Recognizer in 2023) sits alongside Textract in the "hosted, table-and-form-aware" category. It ships prebuilt models for invoices, receipts, ID documents, business cards, W-2s, and 1098s, plus the general-purpose `prebuilt-layout`, `prebuilt-read`, and `prebuilt-document` models. Custom-model training is supported for domain-specific forms.

```python
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

client = DocumentIntelligenceClient(endpoint=ENDPOINT, credential=AzureKeyCredential(KEY))
poller = client.begin_analyze_document(
    model_id="prebuilt-layout",
    body=Path("page-01.pdf").read_bytes(),
    content_type="application/pdf",
)
result = poller.result()
for page in result.pages:
    for line in page.lines:
        print(line.content, "|", getattr(line, "confidence", None))
```

Reach for it when: you are an Azure shop, or the corpus has a specific form-type (invoices, receipts) matched by a prebuilt model — the domain-tuned models often beat general-purpose OCR on that slice by enough to be worth the vendor lock-in.

### GCP Document AI — hosted, processor-per-document-type

[GCP Document AI](https://cloud.google.com/document-ai) uses a "processor" abstraction: you create a processor of a given type (`OCR_PROCESSOR`, `FORM_PARSER_PROCESSOR`, `INVOICE_PROCESSOR`, `LAYOUT_PARSER_PROCESSOR`, ...) and send documents to it. Like Textract and Document Intelligence, it exposes text + tables + form fields with per-token confidence.

```python
from google.cloud import documentai

client = documentai.DocumentProcessorServiceClient()
name = client.processor_path(PROJECT, "us", PROCESSOR_ID)
raw = documentai.RawDocument(content=Path("page-01.pdf").read_bytes(),
                             mime_type="application/pdf")
result = client.process_document(request=documentai.ProcessRequest(name=name, raw_document=raw))
document = result.document
print(document.text[:400])
for entity in document.entities:
    print(entity.type_, "=", entity.mention_text, "conf:", entity.confidence)
```

Reach for it when: you are a GCP shop, you already run BigQuery / GKE / Vertex; or you want to combine the OCR output with Google's downstream `Layout Parser` for structural chunking (chapter 4 will link back to the "let the OCR service do the chunking" shape).

## Cost, latency, and accuracy — measure, do not assume

Every vendor page will say their service is fastest and most accurate. The published benchmarks are on documents that look like *their* training data. On your corpus the answers might be reversed. `exercise-02` will have you measure, on ~50 of your own sample pages, four numbers per back-end:

- **Cost per 1,000 pages.** Look up the current per-page price on the vendor page; multiply. Include per-feature surcharges when you use TABLES / FORMS.
- **Wall-clock p50 and p95 per page.** Include network round-trip for hosted services; include CPU time for Tesseract. Async services need "queue-to-result" measured, not just "call-to-response."
- **Character error rate (CER)** against a small hand-labelled ground truth. Ten pages you type out by hand are enough to detect an order-of-magnitude difference.
- **Table-extraction recall.** For a page with a known table, count cells preserved with correct row/column structure. Tesseract's table story is weak; hosted services vary.

Do the measurement before you commit an architecture. Then re-measure quarterly — vendor accuracy drifts, prices drift, and Tesseract quality drifts with new models.

## The adapter pattern that keeps the rest of the pipeline back-end-agnostic

Do not scatter Textract or Tesseract calls through your chunker. Wrap each back-end behind a common shape and let the router pick:

```python
from typing import Protocol

class OcrBackend(Protocol):
    name: str
    def ocr_page(self, image_bytes: bytes, *, hint_lang: str = "eng") -> "OcrPage": ...

class OcrPage:
    text: str
    tokens: list["OcrToken"]      # each with text, bbox, confidence
    tables: list["OcrTable"]      # each with cell grid + bbox
    mean_confidence: float

class TesseractBackend:  ...
class TextractBackend:   ...
class DocIntelligenceBackend: ...
class DocumentAIBackend:  ...
```

The rest of the pipeline sees `OcrPage`. Swapping Tesseract for Textract, or routing per document by SLA, becomes one function.

## Escalation and hybrid strategies

Very few production pipelines pick one back-end and stop. Two shapes to know:

- **Confidence-triggered escalation.** Run the cheap back-end (Tesseract) first. If page mean confidence < 0.7 or table cells were dropped, re-run the page through the expensive back-end (Textract / Doc Intelligence / Doc AI). Log the escalation rate; it is a cost-driver.
- **Feature split.** Use Docling or Textract for the whole document layout (headings, tables, reading order), then use Nougat (chapter 2) or MathPix on the *pages* the layout model flagged as containing heavy math. Chapter 8 revisits this for figure / formula content.

## Guardrails and pitfalls

- **PII and residency.** Hosted OCR sees the document. Contracts, medical records, and anything with a legal residency clause need explicit sign-off before you route them to a non-VPC service. This is a governance concern the paired `ai-risk-engineer` (peer, level 25) track owns methodology for; the ownership rule for this module is: *do not silently route sensitive corpora off-prem*.
- **Cost blow-ups on re-ingest.** A "full rebuild" over a corpus you have already OCRed is expensive. Cache OCR output keyed on `(source_uri, content_hash, backend, backend_version)` so re-runs are free until the source changes.
- **The scanned-back-into-PDF trap.** Some organizations OCR their scanned PDFs *once* (using Adobe Acrobat, for example) and store the result as a searchable PDF with a text layer stitched over the images. That text layer is *the previous OCR pass's output*. If you then run your own OCR on the same PDF, you might get worse results than trusting the pre-baked layer. Read the text layer first; only fall back if it is bad.
- **Confidence != correctness.** Even a 0.95-confidence token can be wrong when the underlying character is ambiguous (`O` vs. `0`, `l` vs. `1`, `rn` vs. `m`). Do not treat high confidence as truth for downstream key extraction — form-field values still deserve a validation regex.

## What you should be able to do after this chapter

- Route a document per-page between native-text extraction and OCR based on a text-layer probe and a mean-confidence check.
- Call Tesseract, Textract, Azure Document Intelligence, and GCP Document AI from Python and normalize their outputs into a single `OcrPage` shape.
- Design a cost / latency / accuracy measurement on 50 pages of your own corpus that produces defensible numbers, not vendor-marketing claims.
- Name a confidence-triggered escalation strategy and explain when it beats picking one back-end.
- Identify the three OCR pitfalls that will bite you in production (PII / residency, re-ingest cost, pre-OCRed scan traps) and the guardrail for each.

Chapter 4 shifts gears from *getting bytes into passages* to *turning passages into retrievable chunks* — the boundary-policy decision that dominates the second half of this module.

## Further reading

- Tesseract, [User docs](https://tesseract-ocr.github.io/tessdoc/) and [`pytesseract` README](https://github.com/madmaze/pytesseract).
- AWS Textract, [Developer Guide](https://docs.aws.amazon.com/textract/latest/dg/what-is.html), [pricing](https://aws.amazon.com/textract/pricing/).
- Azure AI Document Intelligence, [Overview](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/overview), [prebuilt models](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/model-overview).
- GCP Document AI, [Overview](https://cloud.google.com/document-ai/docs/overview), [processor list](https://cloud.google.com/document-ai/docs/processors-list).
- LlamaIndex reader integrations that wrap these: [`llama-index-readers-file`](https://docs.llamaindex.ai/en/stable/module_guides/loading/connector/), [`AmazonTextractPDFReader`](https://llamahub.ai/l/readers/llama-index-readers-file), Azure and GCP variants on LlamaHub.
