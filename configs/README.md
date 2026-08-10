# Configuration

`config.yaml` stores default paths and runtime settings used by `SuperModel` and lightweight local runs.

Important fields:

- `storage.raw_dir`, `storage.processed_dir`, `storage.ukb_dir`: data locations.
- `storage.master_nodes`: normalized `StandardNode` file.
- `retrieval.models.encoder`: dense encoder used at runtime. This should match the encoder used to build the FAISS indexes.
- `retrieval.models.generator`: generator model name used by concrete LLM managers. The default runtime manager is currently `MockLLMManager`.
- `retrieval.max_context_tokens`: context budget for generation prompts.
- `retrieval.agent_*`: Level 3 traversal defaults used by `SuperModel` for context size, beam width, score thresholds, and expansion limits.
- `alignment.checkpoint`: generic fallback checkpoint. Dataset-specific HNM checkpoints are preferred when available.
- `partitioning.target_nodes_per_chunk`: legacy config value; current index builds use PyMETIS with an internal target of about 1,000 document/entity nodes per partition.

When rebuilding indexes, keep the encoder setting synchronized with `src/core/encoders.py` and the generated checkpoint metadata.
