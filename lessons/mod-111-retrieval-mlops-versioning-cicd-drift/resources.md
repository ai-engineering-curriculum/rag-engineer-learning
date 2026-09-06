# Resources — mod-111 · Retrieval MLOps

Primary sources (specifications, official vendor documentation, and peer-reviewed papers) for the topics in this module. Grouped by chapter. Third-party tutorials are omitted intentionally — the tooling in this space (orchestrators, CI systems, vector DBs, tracing platforms) moves fast and secondary sources rot quickly.

## Chapter 01 — Why retrieval MLOps exists

- Sculley, D., et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS. <https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html>
- Google Cloud Architecture Center. *MLOps: Continuous delivery and automation pipelines in machine learning.* <https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning>
- Huyen, C. (2022). *Designing Machine Learning Systems.* O'Reilly. Chapters 8 (data distribution shift) and 10 (infrastructure). <https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/>
- Google. *Site Reliability Engineering* — table of contents. <https://sre.google/sre-book/table-of-contents/>
- `ai-infra-mlops` track curriculum — the foundational MLOps peer at level 25. <https://github.com/ai-engineering-curriculum>
- `ai-eval-engineer-learning` track curriculum — the eval peer at level 30. <https://github.com/ai-engineering-curriculum>

## Chapter 02 — Scripted dual-index cutover with health checks and rollback

- Sato, D., Wider, A., & Windheuser, C. (2019). *Continuous Delivery for Machine Learning.* martinfowler.com. <https://martinfowler.com/articles/cd4ml.html>
- Fowler, M. (2010). *BlueGreenDeployment.* martinfowler.com. <https://martinfowler.com/bliki/BlueGreenDeployment.html>
- Google. *Site Reliability Engineering*, Ch. 8 — *Release Engineering.* <https://sre.google/sre-book/release-engineering/>
- MLflow. *Model Registry.* <https://mlflow.org/docs/latest/model-registry.html>
- Weaviate. *Aliases.* <https://weaviate.io/developers/weaviate/manage-data/collections#alias>
- Qdrant. *Collection aliases.* <https://qdrant.tech/documentation/concepts/collections/#collection-aliases>
- Milvus. *Manage aliases.* <https://milvus.io/docs/create_alias.md>
- Elasticsearch. *Aliases.* <https://www.elastic.co/guide/en/elasticsearch/reference/current/aliases.html>
- OpenSearch. *Index aliases.* <https://opensearch.org/docs/latest/im-plugin/index-alias/>
- mod-104 chapter 8 — the tuple and manifest convention this chapter automates.

## Chapter 03 — Retriever CI with a judgment-set eval gate

- Smucker, M. D., Allan, J., & Carterette, B. (2007). *A comparison of statistical significance tests for information retrieval evaluation.* CIKM. <https://dl.acm.org/doi/10.1145/1321440.1321528>
- Sakai, T. (2014). *Metrics, Statistics, Tests.* PROMISE Winter School. <https://doi.org/10.1007/978-3-642-54798-0_6>
- GitHub Actions. *Events that trigger workflows — paths.* <https://docs.github.com/en/actions/using-workflows/triggering-a-workflow#using-filters>
- GitHub Actions. *Caching dependencies to speed up workflows.* <https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows>
- GitLab CI. *`rules:changes`.* <https://docs.gitlab.com/ee/ci/yaml/#ruleschanges>
- Buildkite. *Managing a monorepo (path-based triggers).* <https://buildkite.com/docs/pipelines/monorepo>
- `ir_measures` — the IR-metric harness the CI runs. <https://ir-measur.es/>
- Karpathy, A. (2019). *A Recipe for Training Neural Networks.* <https://karpathy.github.io/2019/04/25/recipe/>
- Continuous Delivery Foundation. *CD for Machine Learning SIG.* <https://cd.foundation/blog/2020/02/11/announcing-the-cd-foundation-ml-best-practices-sig/>

## Chapter 04 — Index-build automation with Airflow / Dagster / Prefect

- Apache Airflow. *Datasets and data-aware scheduling.* <https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/datasets.html>
- Apache Airflow. *KubernetesPodOperator.* <https://airflow.apache.org/docs/apache-airflow-providers-cncf-kubernetes/stable/operators.html>
- Dagster. *Software-defined assets.* <https://docs.dagster.io/concepts/assets/software-defined-assets>
- Dagster. *Freshness policies.* <https://docs.dagster.io/concepts/assets/asset-materialization#freshness-policies>
- Dagster. *Auto-materialize policies.* <https://docs.dagster.io/concepts/assets/auto-materialize>
- Prefect. *Flows.* <https://docs.prefect.io/latest/concepts/flows/>
- Prefect. *Deployments.* <https://docs.prefect.io/latest/concepts/deployments/>
- Kubeflow Pipelines documentation. <https://www.kubeflow.org/docs/components/pipelines/>
- Google. *Site Reliability Engineering*, Ch. 4 — *Service Level Objectives.* <https://sre.google/sre-book/service-level-objectives/>
- Google. *Site Reliability Engineering*, Ch. 25 — *Data-Processing Pipelines.* <https://sre.google/sre-book/data-processing-pipelines/>
- OpenAI. *Rate limits.* <https://platform.openai.com/docs/guides/rate-limits>
- `tenacity` — retry-with-backoff library. <https://tenacity.readthedocs.io/>
- AWS S3. *Using versioning in S3 buckets.* <https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html>

## Chapter 05 — Retrieval cost and latency optimization

- OpenAI. *API pricing.* <https://openai.com/api/pricing/>
- Voyage AI. *Pricing.* <https://docs.voyageai.com/docs/pricing>
- Cohere. *Pricing.* <https://cohere.com/pricing>
- Pinecone. *Pricing.* <https://www.pinecone.io/pricing/>
- Weaviate. *Pricing.* <https://weaviate.io/pricing>
- Pinecone. *Vector quantization.* <https://docs.pinecone.io/docs/vector-quantization>
- Weaviate. *Compression.* <https://weaviate.io/developers/weaviate/configuration/compression>
- Qdrant. *Quantization.* <https://qdrant.tech/documentation/guides/quantization/>
- Cohere. *int8 & binary embeddings.* <https://cohere.com/blog/int8-binary-embeddings>
- Jégou, H., Douze, M., & Schmid, C. (2011). *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI 33(1). <https://doi.org/10.1109/TPAMI.2010.57>
- Guo, R., Sun, P., Lindgren, E., et al. (2020). *Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN).* ICML. <https://arxiv.org/abs/1908.10396>
- Malkov, Y. A., & Yashunin, D. A. (2020). *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI 42(4). <https://doi.org/10.1109/TPAMI.2018.2889473>

## Chapter 06 — Cache tiers: query-embedding, top-k, and generation

- Redis. *Client-side caching.* <https://redis.io/docs/manual/client-side-caching/>
- Redis. *Distributed locks (Redlock).* <https://redis.io/docs/manual/patterns/distributed-locks/>
- Anthropic. *Prompt caching.* <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>
- OpenAI. *Prompt caching.* <https://platform.openai.com/docs/guides/prompt-caching>
- Nishtala, R., Fugal, H., Grimm, S., et al. (2013). *Scaling Memcache at Facebook.* NSDI. <https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final170_update.pdf>
- Karger, D., et al. (1997). *Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web.* STOC. <https://doi.org/10.1145/258533.258660>
- Go standard library. *`x/sync/singleflight`.* <https://pkg.go.dev/golang.org/x/sync/singleflight>
- `cachetools` (Python). <https://cachetools.readthedocs.io/>

## Chapter 07 — Monitoring corpus, query, and quality drift

- Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS. <https://arxiv.org/abs/1810.11953>
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). *A survey on concept drift adaptation.* ACM Computing Surveys. <https://doi.org/10.1145/2523813>
- Alibi Detect. *Drift detection methods.* <https://docs.seldon.io/projects/alibi-detect/en/latest/cd/methods.html>
- Evidently AI. *Data drift docs.* <https://docs.evidentlyai.com/user-guide/tests/data-drift>
- WhyLabs. *`whylogs`.* <https://whylogs.readthedocs.io/>
- Google. *Site Reliability Engineering*, Ch. 6 — *Monitoring Distributed Systems.* <https://sre.google/sre-book/monitoring-distributed-systems/>
- Google. *Site Reliability Engineering*, Ch. 4 — *Service Level Objectives.* <https://sre.google/sre-book/service-level-objectives/>
- LangSmith. *Observability.* <https://docs.smith.langchain.com/observability>
- Langfuse. *Documentation.* <https://langfuse.com/docs>
- Arize Phoenix. *Documentation.* <https://docs.arize.com/phoenix>
- mod-109 chapter 4 — the paired-significance test the rolling quality-drift alert reuses.

## Chapter 08 — Boundaries to `ai-infra-mlops` and `ai-eval-engineer`

- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* EACL. <https://arxiv.org/abs/2309.15217>
- Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2023). *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.* <https://arxiv.org/abs/2311.09476>
- TruLens. *Documentation.* <https://www.trulens.org/>
- DeepEval. *Repository.* <https://github.com/confident-ai/deepeval>
- LangSmith. *Documentation home.* <https://docs.smith.langchain.com/>
- Langfuse. *Documentation home.* <https://langfuse.com/docs>
- Arize Phoenix. *Documentation home.* <https://docs.arize.com/phoenix>
- MLflow. *Model Registry.* <https://mlflow.org/docs/latest/model-registry.html>
- `ai-infra-mlops` track curriculum — the foundational peer. <https://github.com/ai-engineering-curriculum>
- `ai-eval-engineer-learning` track curriculum — the eval peer. <https://github.com/ai-engineering-curriculum>
- Google. *Site Reliability Engineering*, Ch. 8 — *Release Engineering.* <https://sre.google/sre-book/release-engineering/>
- mod-109 chapter 7 — the sister-track boundary from the retrieval-eval side.
