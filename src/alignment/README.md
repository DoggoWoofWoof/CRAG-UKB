# `src/alignment/` — InfoNCE Query-to-Partition Alignment

Trains and runs the **MLP Bi-Encoder** that maps a query embedding to its most relevant knowledge graph partitions. This is the "thinking" component of the v4 agent — i.e., the Teleport phase.

> This module is only **trained once offline**. At query time, `crag_agent.py` runs the encoder forward pass (< 1ms) and calls `vector_store.search_partitions()`.

---

## Why This Exists

FAISS can find similar *nodes*, but partitions don't natively map to query semantics. The MLP Bi-Encoder solves this by learning a contrastive alignment: queries about "Microsoft CEO" should activate the cluster containing Microsoft-related entities, not the Cora citation cluster.

The encoder shrinks the query embedding (768-dim) down to 256-dim — dimensionally aligned with partition centroid embeddings — then scores query-centroid cosine similarity. InfoNCE loss pulls the query toward its true partition's centroid and pushes it away from all other partitions in the same batch.

---

## Files

### `mlp_encoder.py` — Bi-Encoder Architecture

```python
class MLPBiEncoder(nn.Module):
    """
    Maps a 768-dim query embedding to a 256-dim partition-aligned space.
    
    Architecture:
        Linear(768 → 512) → ReLU → Dropout(0.1)
        Linear(512 → 256) → L2-normalize
    
    Used by:
        - infonce_loss.py  during training
        - crag_agent.py    at query time (encode only)
    """
    def encode_query(self, query_emb: Tensor) -> Tensor: ...
    def encode_partition(self, centroid_emb: Tensor) -> Tensor: ...
```

---

### `infonce_loss.py` — Contrastive Training Loop

```bash
# Train the Bi-Encoder
python -m src.alignment.infonce_loss \
  --train \
  --dataset webqsp \            # Uses WebQSP (question → entities → partition) pairs
  --cora-supplement \           # Additionally trains on Cora paper→category pairs
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-4 \
  --checkpoint checkpoints/mlp_encoder.pt
```

**Training Data Sources:**

| Source | Positive Pair | Why |
|---|---|---|
| WebQSP | `(question, target entity's partition)` | Direct multi-hop supervision |
| Cora | `(paper abstract, paper's category partition)` | Validates clustering on known structure |
| SQuAD v2 | `(question, answer passage's partition)` | Semantic retrieval supervision |

**InfoNCE Objective:**

```
L = -log [ exp(sim(q, p+) / τ) / Σ exp(sim(q, pi) / τ) ]

where:
  q   = query embedding (projected to 256-dim)
  p+  = true partition centroid (positive)
  pi  = all other partition centroids in batch (negatives)
  τ   = temperature (default: 0.07)
```

**Target Metric:** Top-3 Partition Recall on WebQSP dev set **> 80%** before proceeding.

---

## Checkpoints

Trained encoder weights are saved to `checkpoints/mlp_encoder.pt`.

The `crag_agent.py` loads this checkpoint at initialization:

```python
self.encoder = MLPBiEncoder.load("checkpoints/mlp_encoder.pt")
```
