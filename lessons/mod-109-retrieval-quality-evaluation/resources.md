# Resources for mod-109-retrieval-quality-evaluation

Curated references for judgment-set construction, retrieval metrics, paired significance, ablation-grid attribution, BEIR / MTEB corpus-fit sanity, and the boundary to the two peer tracks (`ai-eval-engineer`, `model-evaluation-engineer`).

## Framing papers (the "why" of retrieval eval)

- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press. [Chapter 8, "Evaluation in Information Retrieval"](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-in-information-retrieval-1.html) — the textbook derivation of precision, recall, MAP, and nDCG. Chapter 3 of this module reads on top of chapter 8.
- Voorhees, E. M. (2001). *The Philosophy of Information Retrieval Evaluation.* CLEF 2001, LNCS 2406. [DOI:10.1007/3-540-45691-0_31](https://doi.org/10.1007/3-540-45691-0_31) — the TREC-era treatment of what test-collection evaluation promises and where "the mean is not enough" comes from. Anchor for chapter 1 + chapter 4.
- Voorhees, E. M., & Harman, D. K. (Eds.). (2005). *TREC: Experiment and Evaluation in Information Retrieval.* MIT Press. [DOI:10.7551/mitpress/6916.001.0001](https://doi.org/10.7551/mitpress/6916.001.0001) — the TREC playbook; the source of the qrels shape (chapter 2) and the ablation-grid discipline (chapter 5).
- Sanderson, M. (2010). *Test Collection Based Evaluation of Information Retrieval Systems.* Foundations and Trends in Information Retrieval 4(4). [DOI:10.1561/1500000009](https://doi.org/10.1561/1500000009) — the standard survey of test-collection methodology; useful longer read after chapter 2.

## Metrics (chapter 3's toolchain)

- Järvelin, K., & Kekäläinen, J. (2002). *Cumulated Gain-Based Evaluation of IR Techniques.* ACM TOIS 20(4). [DOI:10.1145/582415.582418](https://doi.org/10.1145/582415.582418) — the original nDCG paper. Read for the exponential-gain / log-base-2 convention chapter 3 warns about.
- Chapelle, O., Metlzer, D., Zhang, Y., & Grinspan, P. (2009). *Expected Reciprocal Rank for Graded Relevance.* CIKM. [DOI:10.1145/1645953.1646033](https://doi.org/10.1145/1645953.1646033) — ERR; the cascade-model alternative to nDCG.
- Moffat, A., & Zobel, J. (2008). *Rank-Biased Precision for Measurement of Retrieval Effectiveness.* ACM TOIS 27(1). [DOI:10.1145/1416950.1416952](https://doi.org/10.1145/1416950.1416952) — RBP; the persistence-parameter alternative that chapter 3 (exercise 02 stretch) mentions.
- MacAvaney, S., Macdonald, C., & Ounis, I. (2022). *Streamlining Evaluation with `ir-measures`.* ECIR. [DOI:10.1007/978-3-030-99739-7_38](https://doi.org/10.1007/978-3-030-99739-7_38) — the paper behind the `ir_measures` library chapter 3 uses.
- [`terrierteam/ir_measures`](https://github.com/terrierteam/ir_measures) — the library itself; the syntax chapter 3 + exercise 02 use.
- [`cvangysel/pytrec_eval`](https://github.com/cvangysel/pytrec_eval) — Python bindings to `trec_eval`. Exercise 02 uses it as the cross-library sanity check.
- [`usnistgov/trec_eval`](https://github.com/usnistgov/trec_eval) — the NIST reference implementation of TREC metrics; ground truth for the exponential-gain nDCG chapter 3 uses.
- [`AmenRa/ranx`](https://github.com/AmenRa/ranx) — a Rust-backed alternative for large-scale metric computation and fusion; another cross-library check.

## Significance testing (chapter 4's method)

- Sakai, T. (2014). *Metrics, Statistics, Tests.* In *PROMISE Winter School*, LNCS 8173. [DOI:10.1007/978-3-642-54798-0_6](https://doi.org/10.1007/978-3-642-54798-0_6) — the field's reference for per-query significance testing. Chapter 4 uses it as the anchor; chapter 5 cites its factorial-design section.
- Smucker, M. D., Allan, J., & Carterette, B. (2007). *A Comparison of Statistical Significance Tests for Information Retrieval Evaluation.* CIKM. [DOI:10.1145/1321440.1321528](https://doi.org/10.1145/1321440.1321528) — the empirical comparison of paired t-test, Wilcoxon, sign, bootstrap, and randomization tests for IR. Chapter 4's Wilcoxon default cites this.
- Sakai, T. (2016). *Statistical Significance, Power, and Sample Sizes: A Systematic Review of SIGIR and TOIS, 2006-2015.* SIGIR. [DOI:10.1145/2911451.2911492](https://doi.org/10.1145/2911451.2911492) — the meta-analysis that catalogues the "report `p` without CI / effect size" failure chapter 4 pushes back on.
- Sakai, T., & Zeng, Z. (2023). *On Estimating Variances for Topic Set Size Design.* [arXiv:2304.12063](https://arxiv.org/abs/2304.12063) — the sample-size / topic-set-size methodology exercise 03 step 5 uses.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.* Chapman & Hall/CRC. [DOI:10.1201/9780429246593](https://doi.org/10.1201/9780429246593) — the standard reference for the BCa bootstrap chapter 4 + exercise 03 use.
- Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.* JRSS B, 57(1). [DOI:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) — the FDR correction chapter 4 + chapter 5 apply across the ablation grid.
- [SciPy — `scipy.stats.wilcoxon`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) and [`scipy.stats.bootstrap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) — the implementations exercise 03 pins to.
- [`statsmodels.stats.multitest`](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html) — the FDR-correction implementation exercise 03 + exercise 04 use.

## Judgment sets and LLM-assisted labeling (chapter 2's method)

- Voorhees, E. M. (1998). *Variations in Relevance Judgments and the Measurement of Retrieval Effectiveness.* SIGIR. [DOI:10.1145/290941.291017](https://doi.org/10.1145/290941.291017) — the classic result on inter-annotator disagreement in TREC judgments; the reason chapter 2 requires the spot-check.
- Zobel, J. (1998). *How Reliable Are the Results of Large-Scale Information Retrieval Experiments?* SIGIR. [DOI:10.1145/290941.291014](https://doi.org/10.1145/290941.291014) — the companion paper on pooling stability; the justification for pooling top-k across ≥ 3 retrievers (exercise 01, step 3).
- Bonifacio, L., Abonizio, H., Fadaee, M., & Nogueira, R. (2022). *InPars: Data Augmentation for Information Retrieval using Large Language Models.* SIGIR. [arXiv:2202.05144](https://arxiv.org/abs/2202.05144) — LLM query generation for training data. Chapter 2's LLM query bootstrap cites this.
- Dai, Z., Zhao, V. Y., Ma, J., et al. (2023). *Promptagator: Few-shot Dense Retrieval From 8 Examples.* ICLR. [arXiv:2209.11755](https://arxiv.org/abs/2209.11755) — few-shot LLM-generated queries; another reference for exercise 01's synthetic-query stretch.
- Faggioli, G., Dietz, L., Clarke, C. L. A., et al. (2023). *Perspectives on Large Language Models for Relevance Judgment.* ICTIR. [arXiv:2304.09161](https://arxiv.org/abs/2304.09161) — the systematic study of LLM judges for relevance labeling; the reference chapter 2's protocol B is calibrated against.
- Thomas, P., Spielman, S., Craswell, N., & Mitra, B. (2024). *Large Language Models Can Accurately Predict Searcher Preferences.* SIGIR. [arXiv:2309.10621](https://arxiv.org/abs/2309.10621) — Bing / MSR result that supports LLM-assisted labeling with human spot-check.
- MacAvaney, S., & Soldaini, L. (2023). *One-Shot Labeling for Automatic Relevance Estimation.* SIGIR. [arXiv:2302.11266](https://arxiv.org/abs/2302.11266) — DUO-style LLM labeler; useful comparison for chapter 2's minimal prompt.
- Cohen, J. (1960). *A Coefficient of Agreement for Nominal Scales.* Educational and Psychological Measurement 20(1). [DOI:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104) — the original Cohen's κ paper; the spot-check agreement metric exercise 01 step 5 computes.

## BEIR / MTEB (chapter 6)

- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS Datasets and Benchmarks. [arXiv:2104.08663](https://arxiv.org/abs/2104.08663) — the source of truth for BEIR's 18 datasets, splits, and evaluation protocol.
- [`beir-cellar/beir`](https://github.com/beir-cellar/beir) — the canonical Python harness; corpus / qrels download scripts; chapter 6's option 1.
- Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2022). *MTEB: Massive Text Embedding Benchmark.* [arXiv:2210.07316](https://arxiv.org/abs/2210.07316) — MTEB and its retrieval subtasks; superset of BEIR.
- [`embeddings-benchmark/mteb`](https://github.com/embeddings-benchmark/mteb) — the canonical Python harness for MTEB; chapter 6's option 2.
- [Hugging Face MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — the interactive leaderboard; sort by Retrieval nDCG@10.
- Muennighoff, N., et al. (2024). *MMTEB: Massive Multilingual Text Embedding Benchmark.* [arXiv:2502.13595](https://arxiv.org/abs/2502.13595) — the multilingual expansion; exercise 05 stretch.
- Lin, J., Nogueira, R., & Yates, A. (2021). *Pretrained Transformers for Text Ranking: BERT and Beyond.* Synthesis Lectures on Human Language Technologies. [DOI:10.2200/S01123ED1V01Y202108HLT053](https://doi.org/10.2200/S01123ED1V01Y202108HLT053) — the standard neural-IR reference; chapter 6 of that book covers benchmark methodology at greater depth.

## Ablation grids (chapter 5)

- Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters: Design, Innovation, and Discovery* (2nd ed.). Wiley. — the general experimental-design reference; chapter 5's Latin-square / fractional-factorial pattern comes from here.
- Lin, J., & Yang, P. (2019). *The Impact of Score Ties on Repeatability in Document Ranking.* SIGIR (short paper). [DOI:10.1145/3331184.3331339](https://doi.org/10.1145/3331184.3331339) — the underappreciated reproducibility trap that hits grid runs (score ties break differently across runs; affects nDCG at the boundary).
- Craswell, N., Mitra, B., Yilmaz, E., & Campos, D. (2020). *Overview of the TREC 2019 Deep Learning Track.* [arXiv:2003.07820](https://arxiv.org/abs/2003.07820) — a modern ablation-heavy IR report; the shape chapter 5's grid produces.
- Ma, X., Sun, K., Pradeep, R., & Lin, J. (2021). *A Replication Study of Dense Passage Retriever.* [arXiv:2104.05740](https://arxiv.org/abs/2104.05740) — a canonical one-axis-at-a-time ablation on the DPR paper.
- MacAvaney, S., Yates, A., Feldman, S., et al. (2022). *Reproducing Personalised Search Systems: Do IR Papers Give Enough Info to Replicate Them?* SIGIR. [DOI:10.1145/3477495.3531662](https://doi.org/10.1145/3477495.3531662) — the discipline of reporting enough to enable someone else to re-run your grid.
- Karpukhin, V., Oğuz, B., Min, S., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP. [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) — DPR; the reference retriever whose replication study above is the ablation-grid template.
- Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT.* [arXiv:1901.04085](https://arxiv.org/abs/1901.04085) — the reference for reranker sweeps chapter 5 and exercise 04 include.

## Application-layer eval (boundary to `ai-eval-engineer`, chapter 7)

These are the peer track's canonical references. This module cites them to draw the boundary; the peer track owns them.

- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) — the RAG-triad reference; source of the `context_precision` / `context_recall` metrics chapter 7 explicitly separates from this module's `precision@k` / `recall@k`.
- Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2023). *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.* [arXiv:2311.09476](https://arxiv.org/abs/2311.09476) — another application-layer framework on the peer's side of the boundary.
- Zhu, K., Wang, J., Zhou, J., et al. (2024). *Benchmarking Large Language Models in Retrieval-Augmented Generation.* AAAI. [arXiv:2309.01431](https://arxiv.org/abs/2309.01431) — RGB benchmark; joint retrieval + generation quality. A canonical cross-boundary artifact.
- [TruLens](https://www.trulens.org/) — the peer track's canonical dashboard; the [RAG Triad](https://www.trulens.org/trulens_eval/getting_started/core_concepts/rag_triad/) framework.
- [`confident-ai/deepeval`](https://github.com/confident-ai/deepeval) — the peer's canonical harness; [Faithfulness metric docs](https://docs.confident-ai.com/docs/metrics-faithfulness).
- [`explodinggradients/ragas`](https://github.com/explodinggradients/ragas) — the Python implementation of the RAGAS paper's metrics.

## Adjacent-track pointers

- [`ai-eval-engineer-learning`](https://github.com/ai-engineering-curriculum/ai-eval-engineer-learning) — the eval peer's curriculum (level 30). Owns application-layer RAG eval; consumes this module's qrels + run files (chapter 7).
- [`model-evaluation-engineer-learning`](https://github.com/ai-engineering-curriculum/model-evaluation-engineer-learning) — the model-eval peer's curriculum (level 30). Owns significance-test derivation, contamination detection, and cross-benchmark curation. This module uses their tools (chapter 7).
- [`llm-application-developer-learning`](https://github.com/ai-engineering-curriculum/llm-application-developer-learning) — the level-20 baseline; useful for the "why generator-in-loop metrics are separate" framing.
- Mod-102 of this track — where chunking axes for chapter 5's ablation grid are defined.
- Mod-103 of this track — where the embedder shortlist for chapter 5 and the model card / prefix conventions for chapter 6 come from.
- Mod-104 / mod-105 / mod-106 of this track — where the retriever axis for chapter 5's grid is defined.
- Mod-107 of this track — where the reranker axis and funnel widths for chapter 5's grid come from.
- Mod-111 of this track — where chapter 4's significance-test check and chapter 5's ablation grid become the CI gate; this module's `eval/VERSION` is what mod-111 pins to.

## Also worth a look

- Amigó, E., Deldjoo, Y., Mizzaro, S., et al. (2022). *A Unifying and General Account of Fairness Measurement in Recommender Systems.* Information Processing & Management 59(4). [DOI:10.1016/j.ipm.2022.102895](https://doi.org/10.1016/j.ipm.2022.102895) — a systematic treatment of the metric-family axioms behind chapter 3's "which metric answers what" framing.
- Fuhr, N. (2018). *Some Common Mistakes In IR Evaluation, And How They Can Be Avoided.* SIGIR Forum 51(3). [DOI:10.1145/3190580.3190586](https://doi.org/10.1145/3190580.3190586) — a widely-cited pushback on common IR-eval reporting mistakes; overlaps heavily with chapter 4's "don't report a mean without a CI."
- Craswell, N. (2009). *Mean Reciprocal Rank.* In *Encyclopedia of Database Systems*. [DOI:10.1007/978-0-387-39940-9_488](https://doi.org/10.1007/978-0-387-39940-9_488) — the concise MRR reference chapter 3 links.
- Voorhees, E. M. (2000). *Variations in Relevance Judgments and the Measurement of Retrieval Effectiveness.* Information Processing & Management 36(5). [DOI:10.1016/S0306-4573(00)00010-8](https://doi.org/10.1016/S0306-4573(00)00010-8) — the longer journal version of Voorhees 1998.
- [Anserini](https://github.com/castorini/anserini) and [Pyserini](https://github.com/castorini/pyserini) — the Lucene-based reproducible-IR toolkits that many BEIR reference retrievers are built on; useful for chapter 6 option 1 and exercise 04 step 3 if you want a solid BM25 baseline.
- [`ir-datasets`](https://ir-datasets.com/) — a companion library to `ir_measures`; unified loader for TREC, BEIR, and MS MARCO shape corpora. Simplifies exercise 01's BEIR stand-in and exercise 05's dataset loading.
