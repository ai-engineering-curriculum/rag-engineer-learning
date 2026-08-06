# Exercise 02 — OCR fallback: Tesseract vs. Textract cost / latency compare

**Estimated effort:** ~2 hours (plus AWS setup if you have not used Textract before).

## Objective

Extend the router from `exercise-01` with an OCR fallback branch and measure the *actual* cost / latency / accuracy trade-off between local Tesseract and a hosted OCR service on 50 pages of your own sample. You will finish with a defensible per-corpus recommendation ("for this corpus, Tesseract at 300 DPI is enough" or "for this corpus, escalate to Textract when Tesseract page confidence < 0.75"), backed by numbers you produced, not by vendor marketing.

This exercise mandates Tesseract (free, local) and one hosted service — **pick one** of AWS Textract, Azure AI Document Intelligence, or GCP Document AI. Use whichever cloud you can get credentials on. The mechanics are the same; the exercise text uses Textract for illustration.

## Prerequisites

- `exercise-01` complete: router accepts PDFs and routes them through the native-text branch.
- Chapter 3 (*OCR Fallbacks*) read once.
- Local: `tesseract` installed (`apt`, `brew`, or the [official install docs](https://tesseract-ocr.github.io/tessdoc/Installation.html)) plus the English trained data. Python deps: `pytesseract`, `Pillow`.
- Hosted service: **one** of AWS Textract (`boto3`, credentials wired to a low-limit IAM user), Azure Document Intelligence (`azure-ai-documentintelligence`, key + endpoint), or GCP Document AI (`google-cloud-documentai`, service-account JSON + processor ID). Keep spending under $5 by capping to 50 pages; every service publishes current per-page pricing on its pricing page.

## The corpus

Build a small OCR-required corpus under `data/ocr-corpus/`:

- **10 scanned PDFs, ~5 pages each.** Old contracts, scanned academic articles from `archive.org`, `nasa.gov` scanned reports — any legally shareable scans. Total: ~50 pages.
- **1 hand-labeled "ground truth" file.** Pick 5 pages across the corpus, transcribe them yourself (yes, by hand — 10 minutes) into `data/ocr-corpus/ground-truth/{filename}--p{page}.txt`. This is your CER reference.

Commit the ground truth. Do not commit anything you cannot legally distribute; a `download.sh` script is fine.

## Part A — Extend the router with a per-page routing probe (30 min)

Update `src/ingest/router.py`'s PDF branch to do per-page routing rather than per-document routing:

```python
def route_pdf(path: pathlib.Path) -> list[Literal["native", "ocr"]]:
    """Return one branch per page."""
    ...
```

Use the two-signal probe from chapter 3 — `chars_per_page < 100` OR `ratio_alpha < 0.5` on the page's text — to mark pages as `"ocr"`. Test that the probe correctly flags every page of your scanned corpus and correctly leaves every page of `data/corpus/handbook.pdf` from exercise-01 on the `"native"` branch. Assert this in a test.

Add a rasterization helper: `page.get_pixmap(dpi=300).tobytes("png")` gives you PNG bytes ready for either OCR back-end.

## Part B — Implement two OCR back-ends behind one adapter (30 min)

Create `src/ingest/ocr.py` with:

```python
class OcrPage(Protocol):
    text: str
    tokens: list["OcrToken"]     # each with text, bbox, confidence
    mean_confidence: float

class OcrBackend(Protocol):
    name: str
    def ocr_page(self, image_bytes: bytes, *, hint_lang: str = "eng") -> OcrPage: ...

class TesseractBackend:  ...     # uses pytesseract.image_to_data
class TextractBackend:   ...     # or DocIntelligenceBackend or DocumentAIBackend
```

Both back-ends must:

- Return `text`, `tokens` with per-token confidence, and `mean_confidence`.
- Handle errors (network, malformed page image) by raising a typed exception; do not silently return `""`.

Wire the router's OCR branch: for each page marked `"ocr"`, rasterize, call the configured back-end, and produce `Chunk`s with `element_type="paragraph"`, `page=<page_num>`, `parser=backend.name`, `ocr_used=True` (add this field to your `Chunk` schema if you did not have it — chapter 1 named it), `extras["ocr_mean_confidence"]=<float>`.

## Part C — Measure (45 min)

Write `scripts/ocr_benchmark.py` that runs both back-ends across the 50-page corpus and produces `docs/ocr-benchmark.md`. Measure four numbers per back-end:

### 1. Cost per 1,000 pages

- Tesseract: `0`. Well, machine time. Note the CPU used and assume a rough $ / CPU-hour if you want; the takeaway is "effectively free."
- Hosted: `pages × price_per_page × 1000 / pages_run`. Pull `price_per_page` from the current vendor pricing page and put a `Prices as of YYYY-MM-DD` line above the table.

### 2. Wall-clock p50 and p95 per page

Time each `backend.ocr_page(...)` call. For hosted services, include the full round-trip. For async APIs (like Textract's `StartDocumentTextDetection`), include the poll-to-completion time. Report both p50 and p95.

### 3. Character error rate (CER) on the 5 ground-truth pages

Use `jiwer` (`pip install jiwer`) or implement CER manually:

```
CER = edit_distance(hypothesis, reference) / max(1, len(reference))
```

Report CER per back-end, averaged across the 5 pages. Also report the range (min, max) — one bad page distorts the mean.

### 4. Table-extraction recall

Pick two pages in the corpus that contain a real table. Count the number of cells the back-end recovered (with correct row/column indexing where the back-end supplies it) as a fraction of the ground-truth cell count. Tesseract will do poorly here; hosted services should do much better. This is the point.

Fill in the deliverable:

```markdown
# OCR benchmark — corpus: {name}, sampled 50 pages on {date}

| back-end   | $/1000 pages | p50 latency | p95 latency | CER (mean / range) | table recall |
|------------|--------------|-------------|-------------|--------------------|--------------|
| tesseract  |              |             |             |                    |              |
| {hosted}   |              |             |             |                    |              |

## Recommendation
...
```

The `Recommendation` is a paragraph that names your policy: which back-end for the whole corpus? Or: escalate to `{hosted}` when Tesseract's page confidence is below what threshold?

## Part D — Wire the recommended policy (15 min)

Whichever policy your benchmark supports, implement it in the router. Common shapes:

- **Static.** "This corpus goes through Tesseract; no escalation."
- **Confidence-escalation.** Tesseract first. If `mean_confidence < threshold` on the resulting page, re-run through the hosted back-end and prefer its output.
- **Feature-triggered.** If the page contains a table region (detected from a layout probe), route straight to the hosted back-end even if Tesseract confidence looks fine.

Add a test that the policy is honored end-to-end on one sample page from each branch.

## Deliverables

- `src/ingest/ocr.py` with two `OcrBackend` implementations and one adapter interface.
- Updated `src/ingest/router.py` with per-page routing and the policy from Part D.
- `scripts/ocr_benchmark.py` and `docs/ocr-benchmark.md` filled in with your numbers.
- `data/ocr-corpus/ground-truth/` — 5 hand-transcribed page files (yes, really).
- Tests: the per-page routing probe test, and the policy honoring test.

## Acceptance criteria

- `pytest -q` passes including the new tests.
- `docs/ocr-benchmark.md` has all four numbers filled in for both back-ends, with a dated prices line and a recommendation paragraph that is *actually a decision*, not "it depends."
- The confidence-escalation code path exists and is exercised by a test if that is the policy you chose; otherwise the static-choice code path exists and is exercised.
- Total AWS / Azure / GCP spend for the exercise: `< $5`. Screenshot or export the console cost line into `docs/ocr-benchmark.md` as proof.

## Stretch goals

- Add a third back-end (the other hosted service you did not pick) and re-run. Real production teams often benchmark two hosted services before committing.
- Feed the Tesseract output through a spell-check post-processor (`symspellpy` or `pyspellchecker`) and re-measure CER. Discuss whether the CER improvement is worth the added latency.
- Implement a caching layer keyed on `(source_uri, content_hash, backend, backend_version)` so that re-running the benchmark on the same corpus does not re-hit the hosted API. Verify by re-running and asserting the API was called zero times on the second pass.
