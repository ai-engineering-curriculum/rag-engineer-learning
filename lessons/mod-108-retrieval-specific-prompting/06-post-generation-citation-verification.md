# Chapter 6 — Post-Generation Citation Verification

## Why this chapter exists

The citation prompt (chapter 3) tells the LLM to cite. The refusal prompt (chapter 4) tells it when to abstain. The contradiction prompt (chapter 5) tells it how to reconcile disagreement. None of these three prompts *enforces* anything — they only *instruct*. The LLM can still emit a fluent, well-formatted answer whose citations point to nowhere, whose citations point to the wrong chunk, or whose citations point to the right chunk but do not support the claim.

Post-generation citation verification is the mechanical check that catches these three failures after the fact. It runs after the LLM has emitted an answer, before the answer reaches the user. When it fires, it triggers a mitigation path — retry, redact, or fail.

This chapter is the last piece of the retrieval-specific prompting stack. Without it, the earlier prompts are hope; with it, they are enforced.

## The three verification levels

There are three levels of verification, in increasing strictness:

```
    Level 1 — Anchor resolution                Level 2 — Page / section match
    (cheap, deterministic)                     (cheap, deterministic)

    Every [n] in the answer must               Every citation with a specific page or
    correspond to a chunk in the               section pointer must match the page /
    AssembledContext.                          section of the cited chunk.

    Catches: "[7]" when only [1]..[5]          Catches: "The manual states X [1, p. 42]"
    were rendered — a fabricated                when chunk [1] came from p. 15.
    citation.


    Level 3 — Claim–evidence entailment
    (expensive; LLM or NLI model)

    Every claim in the answer must be
    entailed by the text of the cited chunk.

    Catches: "The maximum is 100 °C [1]"
    when [1] actually says 85 °C — the
    citation resolves and the page is right,
    but the claim is not supported.
```

Every RAG stack should run **level 1** unconditionally. **Level 2** is cheap enough that every stack should run it if the metadata is present. **Level 3** is expensive — an LLM or NLI call per claim — and is deployed selectively (high-stakes flows, sampling, or an offline eval loop).

Chapter 6 is how you build each level and where the trade-offs are.

## Level 1 — Anchor resolution

The simplest and most important check: every citation in the LLM's output must resolve to a rendered anchor.

### The check

```python
import re
from dataclasses import dataclass

CITATION_PATTERN = re.compile(r"\[(\d+)\]")

@dataclass
class AnchorViolation:
    kind: str            # "unresolved" | "malformed"
    citation: str        # what the LLM emitted
    detail: str

def check_anchor_resolution(answer: str, ctx: AssembledContext) -> list[AnchorViolation]:
    rendered_anchors = {c.anchor for c in ctx.chunks}
    violations = []
    for m in CITATION_PATTERN.finditer(answer):
        cite = m.group(0)                     # "[3]"
        if cite not in rendered_anchors:
            violations.append(AnchorViolation(
                kind="unresolved",
                citation=cite,
                detail=f"{cite} not in rendered anchors {sorted(rendered_anchors)}",
            ))
    return violations
```

The regex assumes numeric `[n]` anchors from chapter 2. Adjust for `[Source A]` or structured citations. For structured JSON output (chapter 3 pattern 2), iterate `citations[].anchor` directly — no parsing needed.

### What this catches

- **Fabricated anchors** — the LLM cites `[7]` when only `[1]` through `[5]` were rendered. Classic hallucination signal.
- **Off-by-one drift** — the LLM cites `[0]` when the anchors start at `[1]`.
- **Anchor scheme mismatch** — the prompt rendered `[Source A]` but the LLM emitted `[1]`. Usually a prompt bug, but the check flags it before it reaches a user.

### What this does not catch

Level 1 says the citation points to a real chunk. It says nothing about whether the chunk supports the claim. Levels 2 and 3 handle that.

### Latency and cost

Regex-plus-set-lookup. Microseconds. Run on every response, no exceptions.

## Level 2 — Page / section matching

If your citation scheme includes page or section pointers (either inline `[1, p. 42]` or in a structured JSON `{"anchor": "[1]", "page": 42}`), verify that the pointer matches the chunk's actual metadata.

### The check

```python
PAGE_CITATION = re.compile(r"\[(\d+)(?:,\s*p\.\s*(\d+))?\]")

def check_page_match(answer: str, ctx: AssembledContext) -> list[AnchorViolation]:
    chunk_by_anchor = {c.anchor.strip("[]"): c for c in ctx.chunks}
    violations = []
    for m in PAGE_CITATION.finditer(answer):
        anchor_num, page_str = m.group(1), m.group(2)
        if page_str is None: continue    # no page assertion in the citation
        chunk = chunk_by_anchor.get(anchor_num)
        if chunk is None: continue       # level 1 catches this
        cited_page = int(page_str)
        actual_page = chunk.source.page
        if actual_page is None:
            violations.append(AnchorViolation(
                kind="page_missing",
                citation=m.group(0),
                detail=f"answer asserts p. {cited_page}, chunk source has no page",
            ))
        elif actual_page != cited_page:
            violations.append(AnchorViolation(
                kind="page_mismatch",
                citation=m.group(0),
                detail=f"answer asserts p. {cited_page}, chunk is p. {actual_page}",
            ))
    return violations
```

### What this catches

- The LLM confuses page numbers between adjacent chunks.
- The LLM invents plausible-sounding pages ("p. 42" is a common false page).
- The LLM propagates a page number from a *different* citation in the answer.

### Retriever-side dependency

The `SourceRef.page` and `SourceRef.section` fields must be populated on ingest (mod-102). If your ingest pipeline does not extract these, level 2 is a no-op and the LLM's page numbers are unchecked. This is often a good reason to plumb page/section extraction all the way back to ingest.

### Latency and cost

Deterministic. Milliseconds at worst. Run on every response when the citation shape includes page/section.

## Level 3 — Claim–evidence entailment

The strictest check: for every claim in the answer, verify that the cited chunk's text *entails* the claim. This is what catches the failure mode the prompt cannot: the citation exists, points to the right page, but does not support the claim.

Two approaches to entailment:

### 3a — LLM-based entailment

A second LLM call, one per (claim, evidence) pair, that returns a yes/no verdict.

```python
ENTAILMENT_PROMPT = """You will be given a CLAIM and a piece of EVIDENCE from a document.

Your job: decide whether the EVIDENCE directly supports the CLAIM.

Return a JSON object:
  {"supported": <true|false>, "reason": "<short explanation>"}

Rules:
- "supported" is true ONLY if a reader would say "yes, the claim follows from the
  evidence." Paraphrase is fine; strict verbatim match is not required.
- "supported" is false if the evidence is on-topic but does not directly support
  the specific claim, or if the evidence contradicts the claim.
- If the claim has multiple parts, "supported" is true only if ALL parts are
  supported.

CLAIM: {claim}

EVIDENCE: {evidence}

JSON:"""
```

Pros: high fidelity; handles paraphrase and multi-sentence support well.
Cons: an LLM call per (claim, cite) pair. For a 5-claim answer with 1.5 citations per claim, that's ~8 LLM calls per user query, all on the response path.

### 3b — NLI model entailment

A dedicated natural-language-inference model (e.g., DeBERTa fine-tuned on MNLI, from Hugging Face) that returns an entailment / neutral / contradiction label per pair.

Pros: much cheaper (small model, batchable, tens of ms per pair on GPU); no LLM call.
Cons: NLI models are less robust to paraphrase and long evidence than a strong LLM; require a serving stack.

Both approaches are established in the RAG-eval literature: RAGAS ([Es et al. 2023](https://arxiv.org/abs/2309.15217)) uses an LLM-based approach; the ATTRSCORE paper ([Yue et al. 2023](https://arxiv.org/abs/2305.06311)) explicitly benchmarks both.

### The check shape

```python
@dataclass
class ClaimEvidence:
    claim: str            # the sentence or span from the answer
    citations: list[str]  # the anchors it cites
    evidences: list[str]  # the actual chunk texts for those anchors

@dataclass
class EntailmentViolation:
    kind: str             # "unsupported" | "contradicted"
    claim: str
    citations: list[str]
    verdict: dict         # from the checker

def check_entailment(
    answer: str, ctx: AssembledContext, checker,
) -> list[EntailmentViolation]:
    pairs = extract_claim_evidence_pairs(answer, ctx)
    violations = []
    for p in pairs:
        verdict = checker.check(p.claim, "\n".join(p.evidences))
        if not verdict["supported"]:
            violations.append(EntailmentViolation(
                kind="unsupported",
                claim=p.claim,
                citations=p.citations,
                verdict=verdict,
            ))
    return violations
```

The `extract_claim_evidence_pairs` function is the interesting part. For pattern 1 (inline bracketed) answers, split the answer by sentence, attach any trailing `[n]` citations to the sentence they follow. For pattern 2 (structured JSON) answers, iterate `citations[]` — the `answer_span` and `quote` are already given.

Pattern 3 (answer-with-quote) makes level 3 nearly free: the quote is already in the answer; a substring check confirms it appears in the cited chunk. If the LLM invented the quote, the substring check fails immediately. This is why pattern 3 is the choice for high-stakes flows even though it costs tokens — the verification cost drops from "an LLM call per claim" to "a string search per claim."

### When to run level 3

- **Every response** — if the flow is high-stakes (compliance, medical, legal) and you can pay the latency and cost.
- **Sampling** — 5% of responses, drawn uniformly, for continuous quality monitoring. Cheap; feeds a dashboard.
- **Offline** — batch-run on all responses in eval, not on the response path. This is the mod-109 eval infrastructure; not a runtime check.

## The mitigation loop

Catching a violation is the first half. What you do about it is the second.

Four responses are common:

### 1. Fail-closed → refuse

The response is dropped, the user gets the refusal template from chapter 4. Safe; annoying if it fires often. Appropriate for high-stakes flows where wrong is worse than absent.

### 2. Retry with a stricter prompt

Re-run generation with an additional instruction — "The previous attempt cited anchors that do not exist. Only cite the anchors shown below." Sometimes fixes; often does not; costs one extra LLM call. Cap at one retry.

### 3. Redact the offending sentence

Strip the sentence containing the violating citation from the answer. Ship the rest. Preserves helpfulness for the supported parts; leaves the answer a bit choppy. Suitable when most claims are supported and one drifted.

### 4. Escalate

Log the violation, ship the answer with a warning ("verified partially — see [source]"), and route the trace to a human review queue. Suitable for flows with a human-in-the-loop review process.

Which response fits is a product decision (peer track owns the UX). The retrieval-specific-prompting layer owns the *detection* and the *decision surface* — a structured `VerificationResult` the peer track's response handler consumes.

### The VerificationResult contract

```python
@dataclass
class VerificationResult:
    passed: bool
    violations: list[AnchorViolation | EntailmentViolation]
    level_1_passed: bool     # anchor resolution
    level_2_passed: bool     # page/section
    level_3_passed: bool | None    # entailment; None if not run
    diagnostic: dict         # per-check timing, evidence traces
```

The peer track consumes `VerificationResult` to decide the mitigation (fail-closed vs. retry vs. redact vs. escalate). The rag-engineer commits to producing accurate `VerificationResult` — chapter 7 draws the boundary.

## Building the verifier: a minimal end-to-end sketch

```python
def verify(answer: str, ctx: AssembledContext,
           entailment_checker=None) -> VerificationResult:
    l1 = check_anchor_resolution(answer, ctx)
    l2 = check_page_match(answer, ctx)
    l3 = []
    if entailment_checker is not None:
        l3 = check_entailment(answer, ctx, entailment_checker)
    all_violations = l1 + l2 + l3
    return VerificationResult(
        passed=not all_violations,
        violations=all_violations,
        level_1_passed=not l1,
        level_2_passed=not l2,
        level_3_passed=(None if entailment_checker is None else not l3),
        diagnostic={"num_chunks": len(ctx.chunks), "answer_len": len(answer)},
    )
```

Wire this in between the generator call and the response send. Log every `VerificationResult` for the observability layer.

## What the verifier does NOT do

- **Verify the retrieval itself.** If the retriever returned the wrong chunks, the verifier only knows the LLM cited them. Mod-109 (Retrieval-Quality Evaluation) is where retriever quality is measured.
- **Verify the answer is *complete*.** The verifier checks that what was said is supported. Whether the answer is missing a key point is out of scope; that is a faithfulness-and-answer-relevance metric, not a citation-verification metric.
- **Verify the answer against ground truth.** The verifier compares the answer to the *retrieved context*, not to reality. If the retrieved context is wrong, a verified-supported answer is still wrong. Corpus curation (mod-102) owns this.

## Two operational patterns worth knowing

### The eval-time verifier

Level-3 entailment is expensive on the response path. A common shape is: run level 1 + level 2 on the response path (fast); run level 3 offline as part of the mod-109 eval set. This gives you a nightly dashboard of "what percent of claims were entailed by their citations?" without the runtime latency.

### The self-check pattern

A single-model self-check — ask the same LLM to verify its own answer — is popular in the literature (self-consistency, self-critique). Empirically it catches some failures (unresolved anchors, obvious mismatches) and misses others (subtle paraphrase-vs-source drift) because the same model that produced the drift often does not see it as drift on the second look. Treat self-check as a level-3-adjacent tool useful when you cannot afford a separate NLI stack; do not treat it as a substitute for a real second-opinion check.

## Anti-patterns

### 1. Verifier without a mitigation

The check fires, logs a violation, and the answer ships anyway. The failure is now visible but not prevented. Solution: every violation should trigger *some* mitigation — even if it's "log and warn" — and the peer track handler must consume the result.

### 2. Level 3 with a lax NLI threshold

An entailment model that returns "supported" on any topical overlap. Nothing gets flagged; the check is theater. Solution: calibrate the threshold against a labeled eval set; measure precision and recall of the verifier itself.

### 3. Verifying against the wrong context

The verifier runs against the assembled context from the *current* query, but the LLM cited from a *previous turn's* context (in a multi-turn conversation where context was reused). Every citation looks unresolved. Solution: pass the exact `AssembledContext` used to produce this answer through to the verifier; never re-retrieve.

### 4. Running level 3 on every response with no budget

Level-3 entailment costs an LLM call per claim. On a 5-claim answer, that's 5 LLM calls per user query on top of the generator call. If your latency budget cannot afford this, run level 3 on a sample or offline. Do not run it synchronously without measuring the cost.

### 5. Confusing verification with evaluation

The verifier is a runtime safety net. The eval framework (mod-109) is where you measure whether your prompts and retriever are getting better. Both use similar metrics; they serve different purposes. Do not use the verifier's pass rate as your quality metric — it drifts as your prompt drifts.

## What you should be able to do after this chapter

- Author level-1, level-2, and level-3 citation verifiers over an `AssembledContext` and an LLM answer.
- Choose between LLM-based and NLI-based entailment for level 3 and defend the choice by latency, cost, and quality.
- Design a mitigation loop (fail-closed / retry / redact / escalate) matched to the product's risk profile.
- Recognise the five verifier anti-patterns and correct each.
- Explain why answer-with-quote (chapter 3 pattern 3) collapses level 3 to a substring check and when this trade-off is worth the extra tokens.

## Further reading

- Menick, J., Trebacz, M., Mikulik, V., et al. (2022). *Teaching Language Models to Support Answers with Verified Quotes.* [arXiv:2203.11147](https://arxiv.org/abs/2203.11147) — GopherCite; the answer-with-quote + substring-verification pattern.
- Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). *RAGAS.* [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — the faithfulness and citation-precision metrics this chapter's level-3 check operationalises.
- Yue, X., Wang, B., Chen, Z., Zhang, K., Su, Y., and Sun, H. (2023). *Automatic Evaluation of Attribution by Large Language Models.* EMNLP. [arXiv:2305.06311](https://arxiv.org/abs/2305.06311) — ATTRSCORE, benchmarking LLM-based vs. NLI-based attribution evaluation.
- Rashkin, H., Nikolaev, V., Lamm, M., Aroyo, L., et al. (2023). *Measuring Attribution in Natural Language Generation Models.* Computational Linguistics 49(4). [aclanthology.org/2023.cl-4.2/](https://aclanthology.org/2023.cl-4.2/) — the AIS framework for attribution measurement.
- Manakul, P., Liusie, A., and Gales, M. J. F. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* EMNLP. [arXiv:2303.08896](https://arxiv.org/abs/2303.08896) — the self-check pattern's reference.
- [TruLens `Groundedness` feedback function](https://www.trulens.org/trulens_eval/getting_started/core_concepts/feedback_functions/) and [DeepEval faithfulness metric](https://docs.confident-ai.com/docs/metrics-faithfulness) — production-ready implementations of level-3-style checks.
- Mod-109 of this track (Retrieval-Quality Evaluation) — the eval infrastructure that runs verifier-shaped checks offline over an eval set.
