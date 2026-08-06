# Chapter 3 — ANN Algorithm Families: HNSW, IVF, IVF-PQ, ScaNN, and the Recall–Latency–Memory Frontier

## Why this chapter exists

Every vector database from chapter 2 is a wrapper around one or two of the algorithms in this chapter. If you cannot name what the wrapper is doing, you cannot tune it, you cannot debug it, and you cannot defend a choice against a peer engineer who will ask exactly that. The four families this chapter covers — **HNSW**, **IVF**, **IVF-PQ**, and **ScaNN** — cover ~95% of production dense-vector retrieval. Chapters 4 and 5 turn the knobs; this chapter builds the mental model of *what each algorithm actually does*, and where it sits on the recall-vs-latency-vs-memory frontier that constrains every choice.

The story is not "one algorithm wins." The story is **three complementary shapes** — a graph, a partitioning, and a partitioning with quantization — plus one particularly good implementation of the partitioning idea (ScaNN). Which one wins in your slot depends on your corpus size, your memory budget, and your recall target.

## The problem all four solve

Given `N` vectors in ℝ^d and a query `q`, exact nearest-neighbor is `O(N·d)` per query — you compare `q` against every corpus vector. On `N = 10⁷` and `d = 1024`, that is ~40 GB of arithmetic per query on `float32`. Nobody serves that at interactive latency.

ANN algorithms trade **recall** — the fraction of true nearest neighbors you actually return — for **latency** and **memory**. Every algorithm here answers three questions:

1. **What structure do we build offline** to avoid scanning the whole corpus at query time?
2. **What subset of vectors do we score at query time** and how do we choose it?
3. **How do we approximate the distance** if we cannot afford to keep every raw vector in RAM?

HNSW answers (1) with a graph. IVF answers (1) with a partitioning. IVF-PQ answers (3) with quantized codes. ScaNN answers (2) with a better anisotropic scoring function. Read the chapter with those three questions in mind.

## Family 1 — HNSW (Hierarchical Navigable Small World)

**Canonical paper.** Malkov, Y. A., & Yashunin, D. A. (2016; TPAMI 2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* [arXiv:1603.09320](https://arxiv.org/abs/1603.09320). Read it — the paper is short, clear, and every parameter you will tune in chapter 4 is defined in it.

**Shape.** HNSW builds a **multi-layer graph** where each vector is a node and edges connect it to its `M` (approximate) nearest neighbors. Higher layers are sparse; the bottom layer contains every vector. Search starts at the top layer's entry point, greedily walks toward `q`, drops one layer, walks again, and so on to the bottom layer.

```
   layer 3   ○─────○──────○                    (few, long-range)
              \   /|
   layer 2    ○  ○ ○────○                       (moderate density)
              /\/  \    /|
   layer 1   ○─○────○──○─○─────○                (denser)
              |\ /|  /|\ |  /|
   layer 0   ○─○─○──○─○─○──○─○─○─○─○─○─○─○     (every vector; short-range)
```

At query time, layer 3 does a fast coarse zoom; layer 0 does the fine-grained neighborhood search. The search is O(log N) in the number of layers and O(`efSearch`) in the fine-grained search width.

**The three knobs (depth in chapter 4).**

- **`M`** — target degree per node at layers > 0 (a factor of `2M` at layer 0). Controls graph density; a higher `M` improves recall but grows memory and build time.
- **`efConstruction`** — search-width used *while building* the graph. Higher means better-quality neighbor lists during insertion, at a build-time cost.
- **`efSearch`** — search-width used *at query time*. This is the runtime recall/latency knob you turn per query.

**Where it lives on the frontier.**

- **Recall:** Excellent — the graph-walk shape reaches high recall @ moderate `efSearch` for most corpora.
- **Latency:** Excellent — one query is O(log N + efSearch), and the constants are small.
- **Memory:** *This is where HNSW hurts.* You store every raw vector (unless you quantize the column separately) plus the graph structure: `~2M · N` edges at 4–8 bytes each. For `d = 1024`, `N = 10⁷`, `M = 32`, `float32`: raw vectors alone are ~40 GB, graph adds ~2.5 GB. Everything lives in RAM to keep query latency in the ms range.
- **Build:** Slow. Insertion cost is O(log N · efConstruction) per vector; building 10⁷ vectors takes hours on a single machine.
- **Updates:** Incremental insertion works. Deletes are usually *soft* (marked, filtered at query) — hard delete requires graph repair. Most implementations recommend periodic full-rebuild for delete-heavy workloads.

**Distance measures.** Any distance/similarity works. Cosine and inner-product are common in retrieval; the graph doesn't care as long as it's a well-defined distance.

**Implementations.** `hnswlib` (Malkov's reference), FAISS `IndexHNSWFlat`, pgvector `USING hnsw`, Qdrant (default), Weaviate (default), Chroma (default), Milvus `HNSW`, Elasticsearch/OpenSearch `knn` with `hnsw` method.

**When to reach for it.** Almost always the first choice for RAM-resident corpora under ~10⁸ vectors, especially with modest `d` (≤ 1024). If your corpus fits in memory and you want a solid recall-latency Pareto front, HNSW is the default.

**Where it starts to hurt.** Memory. On `N > 10⁸` or a fat GPU-embedding dimension (`d = 3072`+), HNSW's RAM cost is the pain point. IVF-PQ or DiskANN start to dominate — either by quantizing the vectors themselves, or by paging structure to disk.

## Family 2 — IVF (Inverted File)

**Canonical paper for the idea.** Sivic, J., & Zisserman, A. (2003). *Video Google: A text retrieval approach to object matching in videos.* ICCV 2003. [PDF](https://www.robots.ox.ac.uk/~vgg/publications/papers/sivic03.pdf) — the "inverted file" name is borrowed from IR (mod-101 chapter 2). The specific IVF-for-ANN shape is codified in Jégou et al. 2011 (below, cited in IVF-PQ).

**Shape.** Cluster the corpus into `nlist` cells (a k-means over the training set). Store each cell as a bucket of vectors ("posting list," borrowing the IR term). At query time, pick the `nprobe` closest cells to `q` and exhaustively score the vectors in those cells only.

```
   centroids c₁ … c_{nlist}     ← trained once via k-means

     [c₁]:  v₁₁, v₁₂, v₁₃, ...    ← posting list per cell
     [c₂]:  v₂₁, v₂₂, ...
     [c₃]:  v₃₁, v₃₂, ...
      ⋮

   query q  →  find nprobe nearest centroids to q  →  score all vectors in those cells
```

**The two knobs.**

- **`nlist`** — number of centroids / cells. Larger `nlist` means each cell is smaller, so each probe is cheaper; but you need more probes to cover the true neighborhood. A common heuristic (from the FAISS wiki) is `nlist ≈ 4·√N` for small N and up to `nlist ≈ 16·√N` for large N.
- **`nprobe`** — number of cells to inspect at query time. The runtime recall/latency knob.

**Where it lives on the frontier.**

- **Recall:** Good, tunable — with enough `nprobe` (as a fraction of `nlist`) recall approaches exact. `nprobe = 1` is fast and low-recall; `nprobe = nlist` is exact.
- **Latency:** Better than exact by a factor of `nlist / nprobe`. Concretely: on `N = 10⁷` with `nlist = 8192` and `nprobe = 32`, you score ~10⁷ · 32/8192 ≈ 40k vectors per query instead of 10⁷.
- **Memory:** Comparable to raw storage (still `N · d · sizeof(float)` for the vectors themselves) plus the centroid table. This is where IVF alone is not a memory win — IVF-PQ below is.
- **Build:** k-means training is O(iterations · N · nlist · d) — noticeable but tractable. Assignments are O(N · nlist · d), typically batched.
- **Updates:** Insert is cheap — assign the new vector to the nearest centroid, append to that posting list. Deletion is a list-membership toggle. The centroid quality can drift as the distribution shifts; heavy churn eventually motivates a rebuild.

**Implementations.** FAISS `IndexIVFFlat`, Milvus `IVF_FLAT`, pgvector `USING ivfflat`. Almost every ANN library has an IVF variant.

**When to reach for it.** When HNSW is too memory-hungry and you can accept some rebuild cadence. IVF is also often used as a *training pass* whose k-means output feeds a stronger index (IVF-PQ, IVF-SQ, or a filtered variant).

**Where it starts to hurt.** As a pure IVF, on high-dimensional corpora, HNSW usually beats it on the recall-latency Pareto front for RAM-resident data. IVF's real value in modern stacks is as the scaffolding of IVF-PQ (below), where the quantization inside each cell dominates the memory story.

## Family 3 — IVF-PQ (Inverted File with Product Quantization)

**Canonical paper.** Jégou, H., Douze, M., & Schmid, C. (2011). *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI 33(1). [DOI:10.1109/TPAMI.2010.57](https://ieeexplore.ieee.org/document/5432202) or [PDF](https://inria.hal.science/inria-00514462v2/document). This is the paper the whole "quantize your vectors to fit in RAM" story descends from.

**Shape.** Same IVF scaffolding as above, but each vector inside each cell is stored as a **compact product-quantization code** rather than a raw float vector.

Product quantization splits the `d`-dim vector into `pq_m` subvectors of length `d/pq_m`. For each subvector, a k-means with `2^pq_nbits` centroids is trained (usually `pq_nbits = 8`, giving 256 centroids per sub-space). Each subvector is stored as the index of its nearest sub-centroid — one byte, if `pq_nbits = 8`. A `d = 1024`, `pq_m = 64`, `pq_nbits = 8` code is 64 bytes, versus 4096 bytes for the raw `float32`.

```
   raw vector v ∈ ℝ^1024
   split into 64 subvectors of length 16 each
   subvector v₀ ─► index of nearest sub-centroid out of 256   ─► 1 byte
   subvector v₁ ─► index of nearest sub-centroid out of 256   ─► 1 byte
      ⋮
   subvector v₆₃ ─► ...                                       ─► 1 byte
                                                                ─────
   compact code                                                = 64 bytes
```

At query time, the query vector is split into the same 64 subvectors; distances from each query subvector to each of that subspace's 256 centroids are precomputed into a small lookup table. Scoring a corpus code against `q` then becomes 64 lookups + 64 additions per vector — dramatically faster than an exact `float32` dot product, especially in cache-friendly loops.

**The four knobs (depth in chapter 5).**

- **`nlist`** — IVF cell count (as above).
- **`nprobe`** — IVF probes at query time (as above).
- **`pq_m`** — number of PQ subvectors. Must divide `d`. Higher `pq_m` = larger code = better recall.
- **`pq_nbits`** — bits per PQ code. Almost always 8 (256 centroids per subspace); some implementations support 4.

**Where it lives on the frontier.**

- **Recall:** Lower than HNSW at similar latency on RAM-resident corpora. On large corpora that would OOM under HNSW, IVF-PQ dominates because the alternative is "no index at all."
- **Latency:** Excellent — the compact codes plus the precomputed lookup table make per-vector scoring extremely fast on modern CPUs (SIMD-friendly).
- **Memory:** Excellent — this is the point. A `d = 1024`, `N = 10⁹` corpus that would need ~4 TB for `float32` vectors fits into ~64 GB with `pq_m = 64`, `pq_nbits = 8`.
- **Build:** IVF k-means + PQ training. PQ training is per-subspace k-means; usually done on a representative training sample (~10⁶ vectors, per FAISS guidance).
- **Updates:** Insert is cheap once the codebook is trained (assign to cell, encode, append). Codebook drift is the failure mode — a corpus whose distribution shifts far from the training sample degrades in recall, motivating periodic re-training.

**Related variants.**

- **IVF-SQ (scalar quantization)** — each dimension of the raw vector quantized to int8 independently. Simpler than PQ, worse compression, sometimes better recall.
- **OPQ (optimized product quantization)** — Ge et al. 2013 ([DOI:10.1109/CVPR.2013.379](https://ieeexplore.ieee.org/document/6619297)) — pre-rotate the vector so subspace boundaries land at low-variance directions. Improves recall at the same code size; almost free at query time. FAISS `IndexPreTransform(OPQ, IndexIVFPQ)`.
- **DiskANN** — Subramanya et al. 2019 ([NeurIPS 2019 paper](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html), Microsoft) — graph-based like HNSW but designed to serve most of the graph from SSD with a small RAM cache. Milvus ships `DISKANN` as an index type.

**Implementations.** FAISS `IndexIVFPQ` (the reference), Milvus `IVF_PQ`. Qdrant / Weaviate / Milvus / others also expose PQ *on top of HNSW* (chapter 2), which is a different composition — the graph structure is HNSW, the vector storage inside the graph is PQ-compressed.

**When to reach for it.** When the corpus does not fit in RAM as raw vectors, or when you have a hard memory budget (small edge node, cheap serverless tier). The go-to shape for ~10⁸ – 10¹⁰ vectors in production.

**Where it starts to hurt.** Recall degrades measurably at aggressive compression; you buy some of it back with `nprobe` (spend more query time) or with a **re-rank on the raw vectors** for the top candidates (the pattern the FAISS wiki calls `refine` / `IndexRefine`). Read chapter 5 before deploying an aggressive PQ config.

## Family 4 — ScaNN (Scalable Nearest Neighbors)

**Canonical paper.** Guo, R., et al. (2020). *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.* ICML 2020. [arXiv:1908.10396](https://arxiv.org/abs/1908.10396). Also Guo, R., et al. (2020). *Announcing ScaNN: Efficient Vector Similarity Search.* [Google AI blog](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html).

**Shape.** ScaNN is a partitioning-plus-scoring library from Google Research. Structurally it looks like IVF-PQ — partition, probe, score compact codes — but its two innovations pushed the ann-benchmarks Pareto frontier at its release:

1. **Anisotropic vector quantization.** Instead of minimizing the L2 quantization error uniformly, ScaNN minimizes error *in the direction relevant to inner-product ranking*. The intuition: if the reason you want a good quantizer is that inner-product rankings should survive quantization, you should weight quantization error by how much it distorts inner products — not by how much it changes L2 distance in irrelevant directions.
2. **SOAR (Spilling with Orthogonality-Amplified Residuals)** — Sun et al. 2023 ([arXiv:2404.00774](https://arxiv.org/abs/2404.00774)) — later addition: vectors near partition boundaries get "spilled" into a second partition using orthogonal residuals so probes catch them regardless of which boundary they landed on. Improves recall without proportionally increasing scan cost.

**Where it lives on the frontier.**

- **Recall:** On the recall–latency Pareto front, ScaNN was the leader on ann-benchmarks glove/deep1B at its release and continues to hold up on high-dim MIPS (maximum inner-product search) tasks. HNSW-family implementations have closed much of the gap on lower-dim datasets, but ScaNN remains a strong choice for MIPS at scale.
- **Latency:** Excellent; SIMD-optimized.
- **Memory:** Good — comparable to IVF-PQ in the same regime.
- **Build:** k-means-like training on a representative sample.

**Implementations.** [`google-research/google-research/scann`](https://github.com/google-research/google-research/tree/master/scann) is the reference. Milvus ships a `SCANN` index type descended from it. It is not directly bundled in most databases; if you want ScaNN, you either use it as a library or use Milvus.

**When to reach for it.** When (a) you are optimizing dense MIPS at scale, (b) you can operate a library rather than a service, or (c) you are on Milvus and MIPS on your corpus outperforms HNSW in your ann-benchmarks measurement (chapter 5). For most Postgres/Weaviate/Qdrant users, ScaNN is not the default because it is not the default in those systems — but you should recognize it in the algorithm map.

## The recall-vs-latency-vs-memory frontier — at a glance

The point of the four-family survey is that **there is no free lunch** and **the axes are not independent**. Sketch the frontier as three simultaneous trade-offs:

```
                       high recall
                            ▲
                            │
                            │
                   ┌──── HNSW (RAM-resident)
                   │
                   │       (better) ●   ScaNN (IVF-family + anisotropic)
                   │
                   │              ●   IVF-PQ + refine
                   │
                   │        ●   IVF-PQ (aggressive)
                            │
                            ▼
                       low recall

              low latency ◄──────► high latency
              low memory  ◄──────► high memory
```

The three points a level-30 engineer keeps in mind:

1. **HNSW's Pareto front is best when memory is plentiful.** As soon as memory constrains you, IVF-PQ / DiskANN dominate because HNSW cannot be squeezed enough without a separate quantization step.
2. **IVF alone is rarely the endpoint.** IVF is scaffolding — for IVF-PQ, for IVF-SQ, or as one of the algorithms an engine picks for you.
3. **ScaNN's win is on high-dim MIPS at scale**, especially when the exact-ranking version of your problem is inner-product-shaped (many hosted embedding models produce vectors trained for cosine or IP, so this is very common).

## A concrete build-and-search snippet — FAISS as the reference

FAISS is the algorithm reference; the syntax makes the shapes concrete.

```python
import numpy as np
import faiss

d = 128                          # embedding dim
N = 100_000                      # corpus size
np.random.seed(0)
xb = np.random.randn(N, d).astype("float32")
xq = np.random.randn(5, d).astype("float32")

# HNSW: no training needed
index_hnsw = faiss.IndexHNSWFlat(d, 32)      # M = 32
index_hnsw.hnsw.efConstruction = 200
index_hnsw.add(xb)
index_hnsw.hnsw.efSearch = 64                # runtime recall/latency knob
D, I = index_hnsw.search(xq, k=10)

# IVF-Flat: needs training on a sample
nlist = 256
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(xb)
index_ivf.add(xb)
index_ivf.nprobe = 8                         # runtime knob
D, I = index_ivf.search(xq, k=10)

# IVF-PQ: same scaffolding + per-vector 4-byte codes (pq_m=8, pq_nbits=8)
pq_m, pq_nbits = 8, 8
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, pq_nbits)
index_ivfpq.train(xb)
index_ivfpq.add(xb)
index_ivfpq.nprobe = 16
D, I = index_ivfpq.search(xq, k=10)
```

Three algorithms, three shapes, one library. Chapter 4 tunes `M` / `efConstruction` / `efSearch` against a recall target; chapter 5 tunes `nlist` / `nprobe` / `pq_m` / `pq_nbits` and introduces ann-benchmarks as the methodology.

## What is NOT in this chapter (and where it goes)

- **Parameter sweeps and tuning methodology.** Chapters 4 and 5.
- **Which real database exposes which of these knobs.** Chapter 2 covered the vendor mapping.
- **DiskANN and other on-disk-graph specialists** — mentioned above; deeper coverage is beyond this module's scope. Read the paper if you are shopping for it.
- **Locality-sensitive hashing (LSH), tree-based ANN (KD-tree, Annoy), and graph variants beyond HNSW** — historically important, less common in production dense retrieval; not covered.

## What you should be able to do after this chapter

- Name the four families and, for each, state the structure it builds and the query-time work.
- Explain in one sentence per family where it sits on the recall-latency-memory frontier and when to reach for it.
- Read the abstract of Malkov & Yashunin 2016 and Jégou et al. 2011 and place the paper in context.
- Point to which chapter (4 or 5) tunes each family's parameters.

The next chapter zooms into HNSW — the three knobs, how each moves recall / latency / memory, and how to tune them against a target.

## Further reading

- Malkov, Y. A., & Yashunin, D. A. (2018). *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.* IEEE TPAMI. [arXiv:1603.09320](https://arxiv.org/abs/1603.09320).
- Jégou, H., Douze, M., & Schmid, C. (2011). *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI 33(1). [PDF](https://inria.hal.science/inria-00514462v2/document).
- Ge, T., et al. (2013). *Optimized Product Quantization.* CVPR 2013. [DOI:10.1109/CVPR.2013.379](https://ieeexplore.ieee.org/document/6619297).
- Guo, R., et al. (2020). *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.* ICML 2020. [arXiv:1908.10396](https://arxiv.org/abs/1908.10396).
- Sun, P., et al. (2024). *SOAR: Improved Indexing for Approximate Nearest Neighbor Search.* [arXiv:2404.00774](https://arxiv.org/abs/2404.00774).
- Subramanya, S. J., et al. (2019). *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS 2019.
- FAISS wiki: [*Guidelines to choose an index*](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index), [*The index factory*](https://github.com/facebookresearch/faiss/wiki/The-index-factory).
- Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020). *ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms.* Information Systems 87. [arXiv:1807.05614](https://arxiv.org/abs/1807.05614). Website: [ann-benchmarks.com](https://ann-benchmarks.com/).
