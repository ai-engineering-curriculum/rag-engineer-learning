# Exercise 1 — Classify retrieval failures

**Time:** ~15 minutes. Paper and pen. No code.

## What you are practising

Reading a failed RAG interaction and locating which pipeline stage caused the failure. This is the diagnostic skill that shows up in every future module: *before* you tune anything, you have to name what is broken.

## Setup

Assume the same minimal pipeline described in the lecture:

```
Ingest → Chunk → Embed → Retrieve → Generate
```

For each case below, decide which single stage is *most likely* responsible. If more than one stage is plausibly at fault, pick the earliest one — problems propagate downstream, and the earliest broken stage is usually the one worth fixing first.

## Cases

For each case, write down (a) the stage, and (b) one sentence of reasoning.

1. The user asks "when is our office closed for winter break?" The answer comes back "I don't have that information." The company's HR handbook has a section titled *Holiday schedule* that lists the exact dates. The section is in the corpus. The retriever returned three chunks, none of them from that section.

2. Same corpus. The user asks the same question. The retriever returned the *Holiday schedule* chunk in first place. The generator answered "I don't have that information."

3. Same corpus. The user asks "who approves purchase requests over $10k?" The retrieved chunk contains the sentence "Purchase requests above $10,000 require director approval." The generator answered "The CFO approves purchases over $10,000."

4. The PDF export step drops all table cells and keeps only headers. The user asks about pricing tiers, which live in a table. The retriever returns the *Pricing* page but the answer omits the actual numbers.

5. A wiki page was updated last week to move a policy from "quarterly" to "monthly." The retriever confidently returns the old version. Nothing in the source system points at the old version anymore.

6. The user asks "how do I request time off?" The corpus uses the phrase "annual leave application" everywhere. The retriever returns unrelated chunks about *time tracking*.

7. A chunk of the correct document was retrieved, but it starts mid-sentence: "…must submit the form to their manager within 48 hours." The generator, given only this fragment, answers about the wrong form.

8. The user is a contractor. The retriever surfaces an internal-only compensation memo and the generator quotes from it.

## Debrief

Compare your answers with someone else on the track, or with your own answers after finishing the lab. There is not always a single crisp answer — case 4 and case 7, for example, are both defensible as either an ingest failure or a chunking failure depending on where you draw the line. That ambiguity is the point: production RAG bugs live at the seams between stages, and calibrating your intuition for which side of a seam a failure sits on is the actual skill.
