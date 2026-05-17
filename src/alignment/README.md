# `src/alignment/` — KL Divergence & InfoNCE Query-to-Partition Alignment

Trains and runs the **MLP Bi-Encoder** that maps a query embedding to its most relevant structural knowledge graph partitions. This is the "teleportation" core of the Level 1 retrieval phase.

> **Update (April 2026): Phase 1 Complete.**
> The architecture has officially locked the **Golden Configuration**: `[MLP] + [KL Divergence] + [Dataset-Locked Tau] + [Max-Quartile HNM]`. This module is trained once offline, and `crag_agent.py` runs the forward pass at query time (<0.4ms latency).

---

## 🧠 The Science of C-RAG Alignment

FAISS can find similar *individual nodes*, but partitions don't natively map to query semantics. The MLP solves this by learning a dense projection. During our Level 1 ablation sweeps, we developed two core technologies to achieve maximum Recall:

### 1. Hard Negative Mining (HNM) — "The Boundary Sharpener"
In multi-hop graphs, partitions are "entangled." Standard negative sampling (random partitions) fails because it doesn't teach the model the fine borders between similar concepts.
*   We use a **Dynamic Topological Quartile Sweep** to select the hardest negatives (partitions semantically similar to the query but lacking the answer).
*   **Result**: Executing 100% (saturated) HNM yielded massive gains in complex datasets like 2Wiki (+2.94% R@1). We discovered the "Trough of Confusion," proving that intermediate HNM introduces noise, and one must use saturated HNM for deep reasoning graphs.

### 2. KL Divergence — "The Stable Teacher"
Standard models use InfoNCE (rigid multi-label matching), which collapses when faced with our aggressive Hard Negative Mining.
*   We implemented **KL Divergence** as a Teacher-Student distillation loss. The MLP (Student) is asked to match the exact probability distribution of the text embeddings (Teacher).
*   **Result**: KL Div provides "Soft Boundaries," recognizing that a hard negative is related but incorrect, preventing gradient explosion. **KL Divergence outperformed InfoNCE in every single dataset.**

---

## Files

### `mlp_encoder.py` — Bi-Encoder Architecture

```python
class MLPBiEncoder(nn.Module):
    """
    Maps a 384-dim query embedding to a 384-dim partition-aligned space.
    
    Architecture:
        Linear(384 → 256) → ReLU → Dropout(0.3)
        Linear(256 → 384) → L2-normalize
    
    Used by:
        - train_mlp.py     during training
        - crag_agent.py    at query time (encode only)
    """
    def forward(self, x: Tensor) -> Tensor: ...
```

---

### `train_mlp.py` — The Master Training Engine

The training loop handles both loss functions `(info_nce_multi`, `kl_div)` and dynamically calculates the Dataset-Locked Temperature ($\tau$) and Hard Negative count ($hn\_k$).

```bash
# Example Training Execution for HNM Ablation
python -m src.alignment.train_mlp \
  --datasets 2wiki \
  --loss kl_div \
  --tau 0.07 \
  --hnm-k 149 \
  --epochs 100 \
  --batch-size 1024
```

**Training Data Sources:**

| Source | Ground Truth Mapping | Rationale |
|---|---|---|
| SQuAD | `1 Query → 1 Document Partition` | Single-hop exact matching |
| MuSiQue | `1 Query → N Supporting Partitions` | Multi-hop reasoning across scattered paragraphs |
| 2Wiki | `1 Query → N Article Partitions` | Bridge-entity multi-document retrieval |
| MetaQA | `1 Query → N Subgraph Entities` | Relational Triple resolution |

## Anti-Overfitting & Generalization
To prevent models from memorizing the graph topology:
1.  **Deterministic Splits**: Uses a fixed `70% Train / 20% Val / 10% Test` split (seed=42).
2.  **Early Stopping**: Monitors Validation Loss (Patience = 20 epochs) and auto-saves the absolute mathematical best checkpoint.
3.  **Regularization**: 
    *   Dropout = `0.3`
    *   Weight Decay (`AdamW`) = `1e-4`
    *   Gradient Clipping (`max_norm`) = `1.0`

---

## Checkpoints

Trained encoder weights are saved to `checkpoints/{dataset}/alignment_mlp.pth`.
The system dynamically loads the specific weights based on the active index during runtime.
