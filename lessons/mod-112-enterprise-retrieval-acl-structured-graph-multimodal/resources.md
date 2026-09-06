# Resources — mod-112 · Enterprise Retrieval

Primary sources (specifications, official vendor documentation, and peer-reviewed papers) for the topics in this module. Grouped by chapter. Third-party tutorials are omitted intentionally — the tooling in this space (ACL APIs, vector DBs, graph stores, multimodal encoders) moves fast and secondary sources rot quickly.

## Chapter 01 — Why enterprise retrieval is a different beast

- Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS. <https://arxiv.org/abs/2005.11401>
- Sanderson, M., & Croft, W. B. (2012). *The History of Information Retrieval Research.* Proceedings of the IEEE 100(Special Centennial Issue). <https://doi.org/10.1109/JPROC.2012.2189916>
- OWASP. *Top 10 for Large Language Model Applications.* <https://owasp.org/www-project-top-10-for-large-language-model-applications/>
- `ai-risk-engineer-learning` track curriculum — the governance peer at level 25. <https://github.com/ai-engineering-curriculum>
- `ai-infra-senior-engineer-learning` track curriculum — the platform peer at level 40. <https://github.com/ai-engineering-curriculum>

## Chapter 02 — Permission-aware retrieval mechanics

- Qdrant. *Filtering.* <https://qdrant.tech/documentation/concepts/filtering/>
- Qdrant. *Payload.* <https://qdrant.tech/documentation/concepts/payload/>
- Pinecone. *Filter with metadata.* <https://docs.pinecone.io/guides/data/filter-with-metadata>
- Weaviate. *Filters.* <https://weaviate.io/developers/weaviate/api/graphql/filters>
- Weaviate. *Multi-tenancy.* <https://weaviate.io/developers/weaviate/manage-data/multi-tenancy>
- Milvus. *Boolean expression rules.* <https://milvus.io/docs/boolean.md>
- pgvector. *Querying.* <https://github.com/pgvector/pgvector#querying>
- Elasticsearch. *Filter search results.* <https://www.elastic.co/guide/en/elasticsearch/reference/current/filter-search-results.html>
- Elasticsearch. *kNN search.* <https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html>
- OpenSearch. *k-NN filter search.* <https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/>
- OWASP. *Top 10:2021 — A01: Broken Access Control.* <https://owasp.org/Top10/A01_2021-Broken_Access_Control/>
- mod-104 chapter 7 — metadata filtering; pre-filter vs. post-filter as the ANN-tuning mechanic this chapter's security discussion depends on.

## Chapter 03 — Enterprise ACL model shapes (SharePoint, Google Drive, Salesforce)

### SharePoint / Microsoft 365

- Microsoft Learn. *Introduction to permissions in SharePoint.* <https://learn.microsoft.com/en-us/sharepoint/introduction>
- Microsoft Learn. *Plan a permissions strategy for sharing and collaboration in SharePoint.* <https://learn.microsoft.com/en-us/sharepoint/governance/plan-permissions-strategy-sharing-and-collaboration>
- Microsoft Learn. *Understand permissions and permission inheritance in SharePoint.* <https://learn.microsoft.com/en-us/sharepoint/plan-user-content-search-and-access-file-shares>
- Microsoft Graph API. *List permissions on a driveItem.* <https://learn.microsoft.com/en-us/graph/api/driveitem-list-permissions>
- Microsoft Graph API. *Delta query for files.* <https://learn.microsoft.com/en-us/graph/delta-query-files>
- Microsoft Purview. *Sensitivity labels overview.* <https://learn.microsoft.com/en-us/purview/sensitivity-labels>

### Google Drive / Workspace

- Google Workspace Admin Help. *How Drive sharing works.* <https://support.google.com/a/answer/60781>
- Google Workspace Admin Help. *Overview of shared drives.* <https://support.google.com/a/answer/7212025>
- Google Drive API v3. *Permissions.* <https://developers.google.com/drive/api/reference/rest/v3/permissions>
- Google Drive API. *Detect changes to files.* <https://developers.google.com/drive/api/guides/changes>

### Salesforce

- Salesforce Architects. *Record-Level Security.* <https://architect.salesforce.com/decision-guides/data-sharing>
- Salesforce Architects. *Sharing architecture.* <https://architect.salesforce.com/fundamentals/data-security>
- Salesforce Developer docs. *UserRecordAccess.* <https://developer.salesforce.com/docs/atlas.en-us.object_reference.meta/object_reference/sforce_api_objects_userrecordaccess.htm>
- Salesforce Help. *Enterprise Territory Management.* <https://help.salesforce.com/s/articleView?id=sf.tm2_intro.htm>
- Salesforce Help. *Sharing sets.* <https://help.salesforce.com/s/articleView?id=sf.customer_portal_setting_light.htm>

### General

- Atlassian. *Confluence — What are page restrictions?* <https://support.atlassian.com/confluence-cloud/docs/what-are-page-restrictions/>

## Chapter 04 — Text-to-SQL retrieval

- Yu, T., Zhang, R., Yang, K., et al. (2018). *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task.* EMNLP. <https://arxiv.org/abs/1809.08887>
- Spider leaderboard. <https://yale-lily.github.io/spider>
- Li, J., Hui, B., Qu, G., et al. (2023). *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs (BIRD).* NeurIPS. <https://arxiv.org/abs/2305.03111>
- BIRD leaderboard. <https://bird-bench.github.io/>
- Lei, F., Chen, J., Ye, Y., et al. (2024). *Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows.* <https://arxiv.org/abs/2411.07763>
- Pourreza, M., & Rafiei, D. (2023). *DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction.* NeurIPS. <https://arxiv.org/abs/2304.11015>
- Gao, D., Wang, H., Li, Y., et al. (2023). *Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation.* <https://arxiv.org/abs/2308.15363>
- LlamaIndex. *Text-to-SQL Guide.* <https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/>
- LangChain. *Build a Question/Answering system over SQL data.* <https://python.langchain.com/docs/tutorials/sql_qa/>
- sqlglot. *SQL parser and transpiler.* <https://github.com/tobymao/sqlglot>
- PostgreSQL. *Statement behavior configuration parameters.* <https://www.postgresql.org/docs/current/runtime-config-client.html#RUNTIME-CONFIG-CLIENT-STATEMENT>
- PostgreSQL. *Row Security Policies.* <https://www.postgresql.org/docs/current/ddl-rowsecurity.html>
- Spider evaluation code. <https://github.com/taoyds/spider/tree/master/evaluation>

## Chapter 05 — Graph RAG on a property graph

- Robinson, I., Webber, J., & Eifrem, E. (2015). *Graph Databases* (2nd ed.). O'Reilly. <https://neo4j.com/graph-databases-book/>
- Neo4j. *Cypher Manual — Introduction.* <https://neo4j.com/docs/cypher-manual/current/introduction/>
- Neo4j. *Operations Manual — Docker.* <https://neo4j.com/docs/operations-manual/current/docker/>
- Kùzu. *Getting Started.* <https://docs.kuzudb.com/get-started/>
- Memgraph. *Documentation.* <https://memgraph.com/docs>
- Edge, D., Trinh, H., Cheng, N., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Microsoft Research. <https://arxiv.org/abs/2404.16130>
- Microsoft. *GraphRAG project.* <https://github.com/microsoft/graphrag>
- Microsoft. *GraphRAG documentation.* <https://microsoft.github.io/graphrag/>
- LlamaIndex. *Property Graph Index Guide.* <https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/>
- LlamaIndex. *Kùzu property graph store.* <https://docs.llamaindex.ai/en/stable/examples/property_graph/property_graph_kuzu/>
- Tang, Y., & Yang, Y. (2024). *MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries.* <https://arxiv.org/abs/2401.15391>. Repo: <https://github.com/yixuantt/MultiHop-RAG>
- Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2022). *MuSiQue: Multi-hop Questions via Single-hop Question Composition.* TACL. <https://arxiv.org/abs/2108.00573>
- Yang, Z., Qi, P., Zhang, S., et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP. <https://arxiv.org/abs/1809.09600>
- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). *From Louvain to Leiden: guaranteeing well-connected communities.* Nature Scientific Reports. <https://arxiv.org/abs/1810.08473>

## Chapter 06 — Multimodal retrieval survey

### Image-text (CLIP / SigLIP)

- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML. <https://arxiv.org/abs/2103.00020>
- Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). *Sigmoid Loss for Language Image Pre-Training (SigLIP).* ICCV. <https://arxiv.org/abs/2303.15343>
- Tschannen, M., Kumar, M., Steiner, A., et al. (2025). *SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features.* <https://arxiv.org/abs/2502.14786>
- OpenCLIP. *Reference OSS implementation.* <https://github.com/mlfoundations/open_clip>
- Zhang, B., Zhang, P., Dong, X., Zhang, Y., & He, C. (2024). *Long-CLIP: Unlocking the Long-Text Capability of CLIP.* <https://arxiv.org/abs/2403.15378>

### Document-page (ColPali / DSE)

- Faysse, M., Sibille, H., Wu, T., et al. (2024). *ColPali: Efficient Document Retrieval with Vision Language Models.* <https://arxiv.org/abs/2407.01449>
- ColPali reference implementation. <https://github.com/illuin-tech/colpali>
- ViDoRe (Visual Document Retrieval) benchmark. <https://huggingface.co/vidore>
- Ma, X., Lin, S. C., Li, M., Chen, W., & Lin, J. (2024). *Unifying Multimodal Retrieval via Document Screenshot Embedding (DSE).* <https://arxiv.org/abs/2406.11251>
- Qdrant. *Multi-vectors (MaxSim).* <https://qdrant.tech/documentation/concepts/vectors/#multivectors>

### Audio (Whisper / CLAP)

- Radford, A., Kim, J. W., Xu, T., et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision (Whisper).* <https://arxiv.org/abs/2212.04356>
- OpenAI. *Whisper repo.* <https://github.com/openai/whisper>
- Elizalde, B., Deshmukh, S., Al Ismail, M., & Wang, H. (2023). *CLAP: Learning Audio Concepts from Natural Language Supervision.* ICASSP. <https://arxiv.org/abs/2206.04769>
- Microsoft. *CLAP repo.* <https://github.com/microsoft/CLAP>
- pyannote-audio. *Speaker diarization toolkit.* <https://github.com/pyannote/pyannote-audio>

### Video

- Xu, H., Ghosh, G., Huang, P.-Y., et al. (2021). *VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding.* EMNLP. <https://arxiv.org/abs/2109.14084>
- Arnab, A., Dehghani, M., Heigold, G., et al. (2021). *ViViT: A Video Vision Transformer.* ICCV. <https://arxiv.org/abs/2103.15691>
- OpenGVLab. *InternVideo.* <https://github.com/OpenGVLab/InternVideo>
- Xu, J., Mei, T., Yao, T., & Rui, Y. (2016). *MSR-VTT: A Large Video Description Dataset for Bridging Video and Language.* CVPR. <https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf>

### General multimodal benchmarks

- Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). *Microsoft COCO: Common Objects in Context.* ECCV. <https://arxiv.org/abs/1405.0312>
- MTEB Multimodal leaderboard. <https://huggingface.co/spaces/mteb/leaderboard>

## Chapter 07 — Hybrid enterprise router

- LlamaIndex. *Router Query Engine.* <https://docs.llamaindex.ai/en/stable/module_guides/querying/router/>
- LangChain. *How to route between sub-chains.* <https://python.langchain.com/docs/how_to/routing/>
- Aurelio Labs. *semantic-router.* <https://github.com/aurelio-labs/semantic-router>
- Anthropic Engineering. *Building Effective Agents.* <https://www.anthropic.com/engineering/building-effective-agents>
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.* SIGIR. <https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf>
- Model Context Protocol. *Specification.* <https://modelcontextprotocol.io/specification>
- OWASP. *Top 10:2021 — A01: Broken Access Control.* <https://owasp.org/Top10/A01_2021-Broken_Access_Control/>
- mod-106 chapter on fusion (RRF and weighted-sum) — the merge step this chapter's router composes across retrievers of different modalities.

## Chapter 08 — Boundaries to `ai-risk-engineer` and `ai-infra-senior-engineer`

### PII / content safety (`ai-risk-engineer`)

- Microsoft. *Presidio — PII detection and anonymization SDK.* <https://github.com/microsoft/presidio>
- AWS. *Amazon Comprehend PII detection.* <https://docs.aws.amazon.com/comprehend/latest/dg/pii.html>
- Google Cloud. *Sensitive Data Protection (DLP).* <https://cloud.google.com/security/products/dlp>
- OWASP. *Top 10 for LLM Applications.* <https://owasp.org/www-project-top-10-for-large-language-model-applications/>
- NIST. *AI Risk Management Framework (AI RMF 1.0).* <https://www.nist.gov/itl/ai-risk-management-framework>
- Guardrails AI. *Documentation.* <https://www.guardrailsai.com/docs>
- NVIDIA. *NeMo Guardrails.* <https://github.com/NVIDIA/NeMo-Guardrails>
- Inan, H., Upasani, K., Chi, J., et al. (2023). *Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.* <https://arxiv.org/abs/2312.06674>
- `ai-risk-engineer-learning` track curriculum. <https://github.com/ai-engineering-curriculum>

### Platform-scale isolation (`ai-infra-senior-engineer`)

- Weaviate. *Multi-tenancy.* <https://weaviate.io/developers/weaviate/manage-data/multi-tenancy>
- Qdrant. *Multi-tenancy patterns.* <https://qdrant.tech/documentation/guides/multiple-partitions/>
- Google Cloud. *Best practices for enterprise organizations (multi-tenant architecture).* <https://cloud.google.com/architecture/best-practices-for-enterprise-organizations>
- Kleppmann, M. (2017). *Designing Data-Intensive Applications.* O'Reilly. <https://dataintensive.net/>
- `ai-infra-senior-engineer-learning` track curriculum. <https://github.com/ai-engineering-curriculum>

### Peer tracks referenced across the module

- `applied-ai-engineer-learning` — product-surface RAG (level 30, peer). <https://github.com/ai-engineering-curriculum>
- `ai-eval-engineer-learning` — application-layer RAG eval and tracing (level 30, peer). <https://github.com/ai-engineering-curriculum>
- `agentic-ai-engineer-learning` — agent orchestration consuming the retriever (level 40, peer). <https://github.com/ai-engineering-curriculum>
- `fine-tuning-engineer-learning` — domain adaptation of encoders (level 30, peer). <https://github.com/ai-engineering-curriculum>
