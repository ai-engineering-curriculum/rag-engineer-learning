# Chapter 5 — Contradiction Handling

## Why this chapter exists

The assembled context contains K chunks. Some of those chunks disagree with each other. The retriever has no way to know that, and the reranker often has no way to prefer one over the other — both are on-topic; both look relevant.

Two examples:

- A support KB article from 2022 says "maximum temperature is 80 °C." A firmware release note from 2025 says "the maximum was raised to 85 °C in firmware 4.2." Both are correct, but only one is current.
- A vendor's official documentation says a feature is available. A community forum post says it does not work. Both were retrieved. The vendor doc is more authoritative; the forum post may be reporting a real bug.

The default LLM behavior on contradiction is:

1. Pick one silently, cite only that one, and give a confident-sounding answer. (The most common failure.)
2. Blend the two into a plausible-sounding sentence that is neither. ("The maximum temperature is between 80 and 85 °C.")
3. Refuse. (Rare, and usually wrong — a real answer exists once you know the resolution rule.)

None of those is what you want. The four patterns in this chapter — **surface**, **prefer authoritative**, **prefer recent**, **hand off to a resolver** — are the tools for handling contradiction correctly, and this chapter is how you choose between them.

## The four patterns

```
    Pattern A — Surface disagreement           Pattern B — Prefer authoritative
    "According to [1], the max is              "The max operating temperature is
    80 °C. However, [2] states that            85 °C, per the official Widget Manual
    firmware 4.2 raised it to 85 °C.           v3 [1]. An older support article
    Please verify which applies to             states 80 °C [2] but is superseded."
    your firmware version."

    Pattern C — Prefer most-recent             Pattern D — Hand off to resolver
    "The maximum operating temperature         [Two-step call]
    is 85 °C, per firmware 4.2 release          Step 1: LLM detects contradiction and
    notes dated 2025-01-15 [2]. This             emits a structured contradiction
    supersedes the earlier value of              record.
    80 °C in the 2022 KB [1]."                  Step 2: A resolver — LLM, human, or
                                                 rule — decides which source wins.
```

- **A (surface)** — the LLM presents both sources and asks the user (or downstream code) to decide.
- **B (prefer authoritative)** — the LLM ranks by a source-authority signal (encoded in `SourceRef.metadata.authority`).
- **C (prefer recent)** — the LLM ranks by `SourceRef.updated_at`.
- **D (hand off)** — the LLM emits a structured "contradiction" object; a separate step (LLM call, rule engine, human) resolves.

Pattern choice is a **product decision** (the peer track has opinions) constrained by **retriever data availability** (this track has opinions). Chapter 7 draws the joint boundary.

## What "contradiction" actually means

Before you can write a contradiction-handling prompt you need a working definition. Three flavors show up in RAG:

1. **Factual contradiction** — chunk A says "the value is X", chunk B says "the value is Y", where X ≠ Y. Genuine disagreement.
2. **Temporal supersession** — chunk A says "the value is X (as of 2022)", chunk B says "the value is Y (as of 2025)". Not a disagreement — one supersedes the other.
3. **Scoped disagreement** — chunk A says "in region 1 the value is X", chunk B says "in region 2 the value is Y". Not a disagreement — different scopes.

Prompt patterns handle these differently. Pattern C (prefer recent) handles temporal supersession by design. Pattern B (prefer authoritative) handles scoped disagreement badly — the LLM needs the scope information to route correctly, and no authority ranking helps. Pattern A (surface) handles all three by punting.

The prompt template for each pattern needs to encode *which* flavor of contradiction the product expects. Get this wrong and Pattern B will confidently pick the wrong side of a scoped disagreement because it was told to prefer "official" sources without knowing that both sources are official for different regions.

## Pattern A — Surface disagreement

The prompt: when the LLM detects disagreement, it names both sources, states each version, and does not pick.

### Prompt block

```
Handling disagreement between sources:
- If two or more documents give different values, dates, or statements for the SAME
  fact, do NOT pick one silently. Instead:
  1. State both (or all) versions.
  2. Cite each with its anchor.
  3. If possible, note WHY they may differ (e.g., different dates, different scopes).
  4. If a resolution requires information you do not have, ask the user for it.
- Do NOT blend the sources into a plausible middle value.
- Do NOT choose the source with the higher retrieval rank as authoritative.
```

### Training-shot

```
Example (for reference only):

<documents>
  <document anchor="[X1]" source="Warranty policy (2022)" updated="2022-04-01">
    Warranty on Widget model A is 2 years.
  </document>
  <document anchor="[X2]" source="Warranty policy (2025)" updated="2025-01-10">
    Warranty on Widget model A is 3 years.
  </document>
</documents>

Question: How long is the warranty on a Widget model A?

Answer: The provided sources give different values. The 2022 policy states 2 years [X1];
the 2025 policy states 3 years [X2]. If your Widget was purchased in 2025 or later,
the 3-year value applies. Otherwise, please confirm which policy your purchase falls
under.
```

### When to use pattern A

- The product tolerates "please clarify" as a user-facing answer.
- The retriever cannot reliably attach authority / recency metadata.
- The stakes are high enough that a wrong pick is worse than a "please decide" prompt.
- The user is likely to know the resolution rule (which firmware version, which region) even if the retriever does not.

## Pattern B — Prefer authoritative

The prompt: an explicit authority ordering, encoded in the rendered header (chapter 2). The LLM prefers the higher-authority source.

### Prompt block

```
Handling disagreement between sources:
- Each document may include an authority attribute: "official", "community", or
  "informal".
- If two documents disagree on the same fact, prefer the "official" source. If
  both are "official", surface the disagreement (see below).
- Do NOT prefer sources merely because they were retrieved with higher scores.
- If you prefer one source over another, state that you did so, cite both, and
  briefly explain (e.g., "per the official manual").

If two OFFICIAL sources disagree, or if no authority attribute is available, fall
back to surfacing the disagreement (Pattern A rules above).
```

### Retriever-side dependency

The authority signal must be populated on ingestion (mod-102) or on retrieval (mod-106 metadata). Recommended shape: a small controlled vocabulary — `official`, `community`, `informal`, `unknown` — attached to the chunk's `SourceRef.metadata`.

Do not let the LLM infer authority from the document text ("this looks like an official document"). It is unreliable and gives the LLM a free hand to reorder sources.

### Training-shot

```
Example (for reference only):

<documents>
  <document anchor="[X1]" source="Widget Manual v3" authority="official">
    The maximum operating temperature is 85 °C.
  </document>
  <document anchor="[X2]" source="Community forum thread" authority="community">
    Some users report the widget failing above 80 °C.
  </document>
</documents>

Question: What is the maximum operating temperature of the widget?

Answer: The maximum operating temperature is 85 °C, per the official Widget Manual v3 [X1].
A community forum thread reports failures above 80 °C [X2]; this is anecdotal and does
not override the official specification.
```

### When to use pattern B

- Your retriever indexes both official (docs, KB, policy) and unofficial (forums, tickets, blog posts) content.
- The product needs a definite answer more than it needs the user to reconcile.
- The authority metadata is *reliable* on your ingest pipeline — if authority is inferred sloppily or absent for many chunks, this pattern degrades to silent-pick-one.

## Pattern C — Prefer most-recent

The prompt: use `updated_at` (a freshness signal populated on ingest) as the tiebreaker.

### Prompt block

```
Handling disagreement between sources:
- Each document is tagged with an "updated" date.
- If two documents disagree on the same fact, prefer the source with the most
  recent "updated" date. State that you did so.
- If dates are within 30 days of each other, treat them as concurrent and
  surface the disagreement instead.
- Do NOT infer dates from the document text; use only the "updated" attribute.
```

### The 30-day window

The "within 30 days" clause is a heuristic against overweighting a marginal date difference. Two policies updated within a month of each other are usually revisions of the same policy, not a supersession event. Tune per corpus:

- Firmware release notes — narrow the window (a new release supersedes even 7-day-old notes).
- Documentation drift — widen (documentation often has multiple minor edits per week without changing meaning).
- News / rapidly-changing information — narrow further.

### Retriever-side dependency

`updated_at` must be populated on ingest (mod-102). It should be the date the *source content changed*, not the date the *retriever indexed it*. If your ingest pipeline only knows "when the crawler ran," pattern C is unreliable — surface instead of prefer.

### Training-shot

```
Example (for reference only):

<documents>
  <document anchor="[X1]" source="Widget spec" updated="2022-04-01">
    Maximum operating temperature: 80 °C.
  </document>
  <document anchor="[X2]" source="Firmware 4.2 release notes" updated="2025-01-15">
    Firmware 4.2 raised the maximum operating temperature to 85 °C.
  </document>
</documents>

Question: What is the maximum operating temperature?

Answer: The maximum operating temperature is 85 °C, per the January 2025 firmware release
notes [X2]. This supersedes the earlier specification of 80 °C from the 2022 spec [X1].
```

### When to use pattern C

- Your corpus has meaningful update timestamps that reflect content change.
- The domain has real supersession — release notes, policies, regulations, product specs.
- Older-but-relevant chunks legitimately drop out of the answer when a newer source updates them.

Pattern B and C often compose: "prefer official *and* recent; when authorities differ, prefer official; when authorities agree, prefer recent; when both are the same, surface." The prompt template is the same rule set with the fallback chain named.

## Pattern D — Hand off to a resolver

The prompt: when contradiction is detected, do not answer; emit a structured contradiction record. A separate step decides.

### Output schema

```json
{
  "status": "contradiction",
  "contradiction": {
    "topic": "maximum operating temperature",
    "positions": [
      {"anchor": "[1]", "value": "80 °C", "source": "Widget Manual v2 (2022)"},
      {"anchor": "[2]", "value": "85 °C", "source": "Firmware 4.2 notes (2025)"}
    ],
    "possible_resolution": "The 2025 firmware notes appear to update the 2022 spec, but I have not been told to prefer recency."
  }
}
```

### Where the resolver lives

Three shapes:

1. **Rule-based resolver.** Downstream code applies the recency / authority rule and produces the final answer without calling the LLM again. Cheap, fast, auditable.
2. **LLM resolver.** A second LLM call, given the contradiction record plus additional retrieval context or human-authored guidance, decides. Expensive but flexible.
3. **Human resolver.** For high-stakes decisions (legal, medical, compliance escalation), route to a human. Rare in interactive flows; common in async workflows.

### When to use pattern D

- The product cannot afford a wrong pick and cannot afford a "please decide" prompt to the user.
- Contradiction is rare enough that a two-step flow does not dominate latency.
- The resolver step has information the LLM does not (organizational policy, business rules, human judgment).

### The MCP tool shape

Pattern D fits naturally into an mod-110 MCP-exposed retriever + resolver architecture. The retrieval tool returns chunks; a resolver tool (or agent) is called explicitly on contradiction; the composition is auditable.

## Composing the patterns

Real prompts often layer them. A defensible default:

```
Handling disagreement between sources:
1. If two documents disagree, first check the "authority" attribute.
   Prefer "official" over "community" over "informal".
2. If both are the same authority, check the "updated" date.
   Prefer the more recent, IF the newer document appears to explicitly supersede
   the older (e.g., "in firmware 4.2, the maximum was raised to ...").
3. Otherwise, surface both versions, cite both, and ask the user to clarify.
4. Do NOT silently pick one, do NOT blend, do NOT prefer higher retrieval scores.
```

This is patterns B, C, A composed in a fallback chain. The training-shots for each pattern above illustrate each branch. If you go this route, include *three* training-shots — one per branch — because model behavior on a compound rule with three branches is much less reliable than on a single-branch rule.

## What contradiction handling does NOT solve

- **Detecting contradiction in the first place.** These patterns tell the LLM what to *do* when it notices contradiction. Whether it notices at all depends on the pair of chunks and the model. If the two contradicting sources use different phrasings ("Widget max: 80 °C" vs. "Widget cannot operate above 85 °C") the LLM may not connect them. There is no clean prompt fix; sometimes an explicit contradiction-detection preprocessing step (a small LLM call over the retrieved set) is warranted.
- **Deciding which source is *actually* right.** Even the strongest prompt cannot recover ground truth the retriever did not provide. If the "official" source is wrong and the "community" source is right, pattern B produces a wrong answer confidently. The mitigation is corpus curation (mod-102 ingestion decisions), not prompt engineering.
- **Handling three-way disagreements gracefully.** All four patterns degrade past 2-way. If your corpus routinely retrieves 3+ contradicting sources on a single question, revisit chunking (mod-102), retrieval scope, or corpus scoping — not the prompt.

## Anti-patterns

### 1. Silent contradiction resolution

The LLM picks the highest-ranked chunk and answers from it without saying so. The user gets a confident wrong answer for half the queries where the reranker's top-1 was superseded by a lower-ranked but more recent chunk. Solution: any of the four patterns is better than none.

### 2. Trusting the LLM's judgment of "authoritative"

The prompt says "prefer the more authoritative source" without defining what authoritative means. The LLM invents its own definition per query. Solution: encode authority as a structured field on the source (mod-102 ingestion), reference it explicitly in the prompt.

### 3. Prefer-recent without a supersession signal

The prompt uses `updated_at` as a tiebreaker in a domain where updates do not mean supersession — documentation portals where every edit changes the timestamp but rarely changes semantics. Result: recency dominates when it shouldn't. Solution: pattern A (surface) is the safer default when `updated_at` is noisy.

### 4. Pattern D without a resolver

The LLM emits a structured contradiction record. Nothing downstream handles it. The user sees `{"status": "contradiction", ...}` rendered as JSON. Solution: pattern D requires a resolver step; do not deploy the pattern until the resolver is built.

## What you should be able to do after this chapter

- Distinguish the three flavors of contradiction (factual, temporal supersession, scoped) and know which pattern each fits.
- Author the prompt block and training-shot for each of the four patterns (surface, prefer authoritative, prefer recent, hand off to resolver).
- Compose the patterns into a defensible default fallback chain.
- Reason about retriever-side dependencies for each pattern (authority metadata for B, `updated_at` for C, resolver step for D) and cite the boundary to mod-102 ingestion.
- Recognise the four contradiction anti-patterns and rewrite the prompt to avoid them.

## Further reading

- Chen, H., Xu, F. F., Arora, S., and Choi, Y. (2022). *Rich Knowledge Sources Bring Complex Knowledge Conflicts.* EMNLP. [arXiv:2210.13701](https://arxiv.org/abs/2210.13701) — a systematic treatment of knowledge conflicts in retrieval-augmented generation, including how models pick between conflicting sources.
- Xie, J., Zhang, K., Chen, J., Lou, R., and Su, Y. (2024). *Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts.* ICLR. [arXiv:2305.13300](https://arxiv.org/abs/2305.13300) — studies whether LLMs prefer their parametric knowledge or the retrieved context on conflict; directly relevant to the "over-helpful" failure this chapter (and chapter 4) address.
- Longpre, S., Perisetla, K., Chen, A., Ramesh, N., DuBois, C., and Singh, S. (2021). *Entity-Based Knowledge Conflicts in Question Answering.* EMNLP. [arXiv:2109.05052](https://arxiv.org/abs/2109.05052) — an earlier reference on measuring model behavior on conflicting evidence.
- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., and Hajishirzi, H. (2023). *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories.* ACL. [arXiv:2212.10511](https://arxiv.org/abs/2212.10511) — the boundary between "trust parametric" and "trust retrieved"; complements chapter 4's refusal.
- Mod-102 of this track (Document Ingestion and Chunking) — where the `authority` and `updated_at` metadata that patterns B and C depend on are populated.
- Mod-110 of this track (Orchestration and MCP-Exposure) — where the pattern-D resolver step lives.
