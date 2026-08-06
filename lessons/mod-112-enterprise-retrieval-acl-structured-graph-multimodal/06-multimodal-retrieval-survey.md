# Chapter 6 — Multimodal Retrieval Survey

## Why this chapter exists

Prior modules assumed the retriever's unit was a text chunk. Real enterprise corpora aren't all text — they include diagrams, scanned forms, screenshots, product photos, meeting recordings, training videos, technical PDFs whose meaning lives in tables and figures more than prose. A text-only retriever indexes the *caption* of a schematic, the OCR of a scanned invoice, the *transcript* of a video — and loses everything the modality carried that words alone don't.

Multimodal retrieval is the family of techniques that keeps the non-text modality on the retrieval path. This chapter is a *survey* — you learn where each family sits on the frontier, what the pretrained checkpoints get you out of the box, and which loop evaluates each. You don't leave this chapter fine-tuning a vision encoder; that is `fine-tuning-engineer` (peer, level 30) territory. You *do* leave with enough judgment to pick a starting stack for each modality your corpus contains.

The four modalities this chapter covers:

1. **Image-text retrieval** — CLIP, SigLIP, and the family. Photo, screenshot, diagram.
2. **Document-page retrieval** — ColPali, DSE. The page *is* the image; skip OCR and prose parsing.
3. **Audio retrieval** — Whisper-transcribe-then-text-embed as the pragmatic baseline; CLAP as the frontier alternative.
4. **Video retrieval** — clip-level retrieval as the shape; frontier still moving.

## Image-text retrieval — CLIP, SigLIP, and the family

### CLIP (Radford et al., 2021)

[CLIP](https://openai.com/research/clip) trains a vision encoder and a text encoder jointly with a contrastive loss on ~400M (image, caption) pairs from the web. After training, both encoders produce vectors in a shared embedding space — an image and its matching caption end up close, non-matching pairs end up far.

Retrieval usage:

- **Text-to-image:** embed the text query with the text encoder; search a corpus of image vectors (embedded with the vision encoder) for nearest neighbors.
- **Image-to-image:** embed both sides with the vision encoder; same nearest-neighbor search.
- **Image-to-text:** embed the query image; retrieve captions embedded with the text encoder. Common for reverse-searching captioned assets.

The reference implementation is [openai/CLIP](https://github.com/openai/CLIP); the widely-used OSS variant is [OpenCLIP](https://github.com/mlfoundations/open_clip), which offers multiple model sizes (ViT-B/32 through ViT-H/14) trained on various data mixes (LAION, DataComp).

```python
# OpenCLIP retrieval sketch
import open_clip
import torch

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Embed corpus images once
def embed_image(pil_image):
    with torch.no_grad():
        return model.encode_image(preprocess(pil_image).unsqueeze(0)).cpu().numpy()

# Query with text at retrieval time
def embed_query(text):
    with torch.no_grad():
        return model.encode_text(tokenizer([text])).cpu().numpy()
```

Store the image vectors in any of the vector DBs from mod-104; retrieval is a normal ANN search over embedded images.

Limitations:

- **Bag-of-concepts.** CLIP retrieves images that *contain* the concepts in the query but is weak on spatial relations ("dog **left of** cat"), counting, and negation ("dog **without** a leash"). Recent variants (SigLIP, EVA-CLIP) improve this; none solve it.
- **Domain drift.** Trained on internet-scraped alt-text; degrades on medical imagery, technical schematics, satellite imagery unless fine-tuned. Domain adaptation is a real engineering step.
- **Long text is truncated.** CLIP's text encoder has a 77-token context; querying with a paragraph-long question wastes most of it. Long-CLIP variants exist ([arXiv:2403.15378](https://arxiv.org/abs/2403.15378)) if this matters.

### SigLIP (Zhai et al., 2023)

[SigLIP](https://arxiv.org/abs/2303.15343) swaps CLIP's softmax contrastive loss for a *sigmoid* loss, which lets training scale to larger batch sizes and doesn't require the global normalization CLIP does. Empirically, SigLIP checkpoints match or beat similarly-sized CLIP on zero-shot ImageNet classification and image-text retrieval, at the same or lower training cost. SigLIP 2 ([arXiv:2502.14786](https://arxiv.org/abs/2502.14786)) extends the recipe.

For your retriever's purposes, SigLIP is a drop-in replacement for CLIP with somewhat better numbers. If you're starting from scratch, prefer SigLIP; if you already have a CLIP-based pipeline, don't switch just to switch.

Checkpoints on Hugging Face: [`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224), [`google/siglip-large-patch16-384`](https://huggingface.co/google/siglip-large-patch16-384), [`google/siglip2-*`](https://huggingface.co/collections/google/siglip2-67b5dcef38c175486e240107).

### When image-text CLIP-style retrieval is right

- **Product catalogs** — text query → matching product images.
- **Media libraries** — screenshots, marketing assets, brand imagery.
- **Diagram retrieval** where the diagram is a photo-like image (not a technical schematic that needs symbol understanding).
- **As a component in a text+image RAG** where the retriever needs to score how well a retrieved image matches the query.

### When CLIP-style is *not* right

- **Documents.** A PDF page is not a photograph. Use ColPali (below).
- **Medical / scientific imagery.** Out-of-domain. Fine-tune or use a domain-specialized encoder (BioMedCLIP, RemoteCLIP).
- **Fine-grained relations.** Bag-of-concepts fails.

## Document-page retrieval — ColPali and DSE

The dominant enterprise-corpus modality is the PDF. The classic pipeline: extract text (native + OCR) → chunk → embed → retrieve. This drops tables, figures, page layout, and the visual hierarchy that a reader uses to skim.

**ColPali** ([Faysse et al., 2024](https://arxiv.org/abs/2407.01449)) skips text extraction entirely. It treats each *page* as an image, runs a vision-language model (PaliGemma) to produce **multiple** patch-level embeddings per page (like ColBERT's late interaction, but on image patches instead of tokens), and computes retrieval scores by maximum-similarity matching of query tokens to page patches. The paper's [ViDoRe benchmark](https://huggingface.co/vidore) shows ColPali beats traditional OCR+text-embed pipelines on visually-rich documents (financial reports, technical papers, slide decks).

Why it works:

- **No text-extraction failures.** Scanned pages, complex layouts, non-Latin scripts — all handled at the vision level.
- **Tables and figures are first-class.** The visual patches over a table encode the table's content the way the eye sees it.
- **Late interaction over patches** captures fine-grained matches — a query token about "revenue" matches the specific patches where the revenue row appears.

Trade-offs:

- **Storage.** Multi-vector per page (dozens of vectors instead of one) increases index size significantly.
- **Retrieval cost.** MaxSim scoring is more expensive than dot-product; requires an engine or library that supports it (Qdrant multi-vector, custom ANN, or LlamaIndex's ColPali retriever wrappers).
- **Freshness of the checkpoint.** ColPali is an active line of work; check for newer variants (ColQwen2, ColSmol).

**DSE (Document Screenshot Embedding)** ([Ma et al., 2024](https://arxiv.org/abs/2406.11251)) is a related-but-simpler approach: a single-vector-per-page encoder, no late interaction, but the encoder is optimized for screenshots of documents (including web pages, PDFs, slides). Trade-off: cheaper index and retrieval than ColPali, lower ceiling on visually-complex pages.

**Practical recipe** for a documents-heavy corpus:

1. Render each PDF page to a PNG at 150–200 DPI.
2. Embed with ColPali (multi-vector) or DSE (single-vector) depending on your storage / cost budget.
3. Index with a vector DB that supports multi-vector (Qdrant does; use the multi-vector index type).
4. At query time, embed the query with the same encoder's text side; MaxSim against the page multi-vectors.
5. Return top-k pages; feed page images (or extracted text) to a vision-capable generator (GPT-4o, Claude Sonnet with vision, Gemini) as the "chunks."

**When to still use text-based retrieval on documents:** high-volume corpora where the storage / compute delta doesn't justify the recall improvement; corpora that are already clean digital text (docstring / API reference / clean Markdown); pipelines that need extraction anyway for other downstream steps (analytics, translation).

## Audio retrieval — Whisper-transcribe or CLAP

### Pragmatic baseline: Whisper + text embeddings

The everyday shape:

1. **Transcribe.** [Whisper](https://github.com/openai/whisper) (OpenAI, open-source) or [distil-whisper](https://github.com/huggingface/distil-whisper) (smaller, faster) turns audio to text with timestamps. Also viable: [WhisperX](https://github.com/m-bain/whisperX) for word-level alignment; [Pyannote](https://github.com/pyannote/pyannote-audio) for speaker diarization if the corpus is multi-speaker (meetings, interviews).
2. **Chunk the transcript** with timestamps preserved (mod-102 patterns; time-based windows of ~30–60 seconds are common).
3. **Embed the chunks** with your text embedding model of choice.
4. **Retrieve** as normal text; return the source-audio timestamp with each chunk so the generator can cite "audio_id X at 04:12" or the app can play the clip.

This is the pattern that ships. It loses non-speech signal (music, tone, environmental sounds) but for enterprise corpora (meetings, earnings calls, support-line recordings, training videos), the transcript is 95% of what matters.

### Frontier alternative: CLAP

[CLAP](https://github.com/microsoft/CLAP) (Contrastive Language-Audio Pretraining) is the audio analog of CLIP — an audio encoder and a text encoder jointly trained with contrastive loss on (audio, caption) pairs. Retrieval is text-to-audio directly, without transcription: "sound of a car engine starting" retrieves audio clips of engines starting.

When CLAP wins:

- Non-speech audio matters (industrial sound-monitoring, wildlife recordings, security audio).
- The signal is in *how* something sounds, not *what* was said (music retrieval by mood, product-quality audio inspection).
- You want to query with audio (find similar-sounding clips).

For a text-only, meeting-heavy enterprise corpus, Whisper + text embeddings will almost always be the right default. CLAP is worth knowing about; keep it in your survey.

## Video retrieval — the frontier

Video is where the state of the art is still moving. The three general patterns:

1. **Keyframe extraction + CLIP** — sample keyframes at a fixed rate (or on scene change), embed each with CLIP, retrieve individual frames. Simple; loses temporal structure entirely.
2. **Video-clip encoders** — dedicated video encoders like [VideoCLIP](https://arxiv.org/abs/2109.14084), [InternVideo](https://github.com/OpenGVLab/InternVideo), [ViViT](https://arxiv.org/abs/2103.15691). Embed short clips (a few seconds each) directly, then retrieve at clip granularity. Better temporal structure at higher cost.
3. **Whisper transcript + CLIP keyframe hybrid** — treat the video as two parallel streams: an audio transcript indexed as text, keyframes indexed as images. Retrieve from either stream; return both. Pragmatic and often effective for training videos, tutorials, product demos.

For enterprise corpora (training content, product demos, all-hands recordings, customer calls with video), pattern 3 is usually the right starting point — it composes techniques you already run and returns useful results now. Pattern 2 is where the research frontier is; expect the state of the art to shift multiple times per year.

Reference benchmarks: [MSR-VTT](https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/), [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/), [MSVD](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) — all text-to-video retrieval sets you can use to compare methods on public data.

## Cross-modal retrieval — what a unified stack looks like

For a corpus with text + images + PDFs + audio + video, the retrieval layer typically has *multiple indexes* — one per modality — and a router (chapter 7) that decides which to call per query. A common shape:

- **Text index** (vector + lexical) for prose.
- **Image / document-page index** (CLIP / SigLIP for photos, ColPali for pages).
- **Audio index** (transcribed chunks in the text index, with `modality: "audio"` metadata; add CLAP if non-speech matters).
- **Video index** (transcribed audio in the text index; keyframes in the image index).

Some questions hit only one; some hit multiple. The router can fan out to N modalities in parallel and RRF-merge (mod-106's fusion patterns transfer directly).

**Shared embedding space?** Occasionally people ask whether all modalities should embed to the same space. In principle CLIP-family models do this for image+text. In practice, forcing everything into one space usually costs quality — the audio embedding trained on non-speech audio does not align well with a text embedding trained on prose. Multi-index-with-router is the operational reality for now.

## Evaluating multimodal retrievers

The mod-109 metric machinery (recall@k, MRR, nDCG on judgment sets) applies verbatim — a judgment is still `(query, relevant_asset_id)`, and the "asset" being an image or a PDF page or an audio clip changes nothing about the metric. What changes:

- **Judgment sets are more expensive to build.** Labeling image-caption pairs, or video-timestamp pairs, requires more per-item annotator time than labeling text-relevance.
- **Benchmark availability.** Public multimodal retrieval benchmarks — MS COCO Captions (image-text), Flickr30K, MSR-VTT (video-text), AudioCaps (audio-text), ViDoRe (document pages), ImageBench, [MTEB Multimodal](https://huggingface.co/spaces/mteb/leaderboard) — let you compare methods; expect their absolute numbers to translate imperfectly to your corpus.
- **Modality-specific pathologies.** OCR failures on scanned pages; speaker-diarization errors on audio; keyframe undersampling on video — evaluate the *pipeline* end-to-end, not just the embedding.

For a corpus with a specific modality, build a domain-fit judgment set of 50–200 items with a colleague who owns the corpus. That is worth more than a leaderboard result on a public set.

## Where this chapter's boundary sits

- **Fine-tuning multimodal encoders** — CLIP/SigLIP/ColPali domain adaptation, contrastive-loss data curation. `fine-tuning-engineer` (peer, level 30).
- **Building a production PDF-processing pipeline** that combines OCR + layout parsing + table extraction. mod-102 covered the text side; the vision-first alternative (ColPali) obviates most of it, but if you still need extracted tables/text for other reasons (analytics, translation), mod-102 tools stand.
- **Generation over multimodal context** — feeding retrieved images + text to a vision LLM (GPT-4o vision, Claude Sonnet with images, Gemini). Generation is `applied-ai-engineer`'s (peer, level 30) product-surface concern; retrieval is here.
- **PII detection on images** (faces, license plates, screenshots with credentials). `ai-risk-engineer` (peer, level 25).

## What you should be able to do after this chapter

- Name the four modalities (image, document page, audio, video) and the reference encoder / pipeline for each (CLIP/SigLIP, ColPali/DSE, Whisper + text-embed / CLAP, keyframe+transcript hybrid / video-clip encoder).
- Argue when CLIP-style image retrieval is right and when ColPali-style document-page retrieval is right.
- Set up a Whisper + text-embedding pipeline as the audio-retrieval baseline and articulate why it usually wins over CLAP for meeting-heavy corpora.
- Explain the three video-retrieval patterns and pick the pragmatic default (transcript + keyframe hybrid).
- Design a multi-modality retrieval stack with per-modality indexes and a router (deferring the routing detail to chapter 7).
- Build a domain-fit judgment set for a non-text modality and run mod-109's recall@k / MRR / nDCG evaluation.
- Cite the boundary to `fine-tuning-engineer` for domain adaptation and to `ai-risk-engineer` for image PII.

## Further reading

- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) — the CLIP paper.
- Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). *Sigmoid Loss for Language Image Pre-Training.* ICCV. [arXiv:2303.15343](https://arxiv.org/abs/2303.15343) — SigLIP.
- Tschannen, M., Kumar, M., Steiner, A., Zhai, X., Houlsby, N., & Beyer, L. (2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.* [arXiv:2502.14786](https://arxiv.org/abs/2502.14786).
- OpenCLIP: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) — the go-to OSS CLIP + SigLIP implementation with a broad checkpoint zoo.
- Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., & Colombo, P. (2024). *ColPali: Efficient Document Retrieval with Vision Language Models.* [arXiv:2407.01449](https://arxiv.org/abs/2407.01449); benchmark: [ViDoRe on HuggingFace](https://huggingface.co/vidore). Repo: [illuin-tech/colpali](https://github.com/illuin-tech/colpali).
- Ma, X., Lin, S. C., Li, M., Chen, W., & Lin, J. (2024). *Unifying Multimodal Retrieval via Document Screenshot Embedding.* [arXiv:2406.11251](https://arxiv.org/abs/2406.11251) — the DSE approach.
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision.* [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) — Whisper.
- Elizalde, B., Deshmukh, S., Al Ismail, M., & Wang, H. (2023). *CLAP: Learning Audio Concepts from Natural Language Supervision.* ICASSP. [arXiv:2206.04769](https://arxiv.org/abs/2206.04769); repo: [microsoft/CLAP](https://github.com/microsoft/CLAP).
- Xu, H., Ghosh, G., Huang, P.-Y., Okhonko, D., Aghajanyan, A., Metze, F., Zettlemoyer, L., & Feichtenhofer, C. (2021). *VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding.* EMNLP. [arXiv:2109.14084](https://arxiv.org/abs/2109.14084).
- Wang, Y., Li, K., Li, X., Yu, J., He, Y., Chen, G., Pei, B., Zheng, R., Xu, J., Wang, Z., Shi, Y., Jiang, T., Li, S., Zhang, H., Huang, Y., Qiao, Y., Wang, Y., & Wang, L. (2022–2024). *InternVideo* series. [InternVideo on GitHub](https://github.com/OpenGVLab/InternVideo).
- Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). *Microsoft COCO: Common Objects in Context.* ECCV. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312) — the reference image-caption benchmark.
- Xu, J., Mei, T., Yao, T., & Rui, Y. (2016). *MSR-VTT: A Large Video Description Dataset for Bridging Video and Language.* CVPR. [Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf) — the standard video-text retrieval benchmark.
