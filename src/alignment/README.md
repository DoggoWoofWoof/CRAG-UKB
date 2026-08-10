# `src/alignment/` - Query-to-Partition Alignment

Trains and runs the **MLP Bi-Encoder** that maps a query embedding to its most relevant structural knowledge graph partitions. This is the "teleportation" core of the Level 1 retrieval phase.

> **Status (April-May 2026): Level 1 is the mature paper thread.**
> The selected HNM-optimized runtime configuration is an MLP trained with KL divergence, dataset-specific temperature, and dataset-specific hard-negative count. Some pre-HNM sweeps still show InfoNCE winning raw R@1, so paper claims should frame KL as the robust HNM-stress choice rather than a universal winner.

---

## 🧠 The Science of C-RAG Alignment

FAISS can find similar *individual nodes*, but partitions don't natively map to query semantics. The MLP solves this by learning a dense projection. During our Level 1 ablation sweeps, we developed two core technologies to achieve maximum Recall:

### 1. Hard Negative Mining (HNM) — "The Boundary Sharpener"
In multi-hop graphs, partitions are "entangled." Standard negative sampling (random partitions) fails because it doesn't teach the model the fine borders between similar concepts.
*   We use a **Dynamic Topological Quartile Sweep** to select the hardest negatives (partitions semantically similar to the query but lacking the answer).
*   **Result**: Executing 100% (saturated) HNM yielded massive gains in complex datasets like 2Wiki (+2.94% R@1). We discovered the "Trough of Confusion," proving that intermediate HNM introduces noise, and one must use saturated HNM for deep reasoning graphs.

### 2. KL Divergence — "The Stable Teacher"
Standard models use InfoNCE (rigid multi-label matching), which collapses when faced with our aggressive Hard Negative Mining.
*   We implemented **KL Divergence** as a soft multi-positive distribution-matching loss.
*   **Result**: KL Div is the selected objective for the HNM-optimized models because it remains stable when hard negatives are aggressively injected. This is strongest as a robustness claim, not as a universal pre-HNM dominance claim.

### 3. Coverage-aware loss (Jigsaw FullCov transfer) — "The Weakest-Positive Sharpener"
KL and multi-positive InfoNCE spread probability mass over *all* golds, but they optimize the **average** positive, not the **weakest** one — so a query can look well-trained while one required partition falls out of the top-K, sinking multi-hop coverage. We port Jigsaw's coverage objective (`coverage_losses.py`): a **CVaR-over-positives** term (mean of the top-`ceil(ρp)` largest per-positive losses → pure *min-over-positives* for p≤4) plus a **FullCov@K top-K barrier** that pushes the weakest gold above the `(K−p+1)`-th highest negative. The training loss becomes `base + λ·coverage`:
*   `coverage_kl` — **primary**: `KL(+HNM) + λ·coverage`.
*   `coverage_infonce` — ablation: `multi-positive InfoNCE(+HNM) + λ·coverage`.
*   `coverage` — ablation: the coverage term alone.

The coverage term uses its own temperature (`cov_temperature`, default 0.05) and defaults mirroring Jigsaw's paper-final config (`target_topk=20`, `topk_weight=0.35`, `cvar_fraction=0.25`, `margin_weight=0.25`). Reference result in Jigsaw: FullCov@100 66%→82% (McNemar p=4e-10). Evaluate primarily on **2Wiki/MuSiQue** (SQuAD's ≤20 partitions make top-K routing degenerate).

```bash
# Train the primary coverage loss on top of the frozen best KL+HNM config
python -m src.alignment.train_mlp --dataset 2wiki --loss_type coverage_kl \
    --tau 0.07 --hn_k 149 --lambda_cov 0.5 --epochs 100

# Or sweep lambda + get significance vs the KL baseline (routes to Modal by default)
python experiments.py run train-coverage -- --datasets 2wiki musique --lambdas 0.1 0.25 0.5 1.0
```

---

## Files

### `mlp_encoder.py` — Bi-Encoder Architecture

```python
class MLPBiEncoder(nn.Module):
    """
    Maps a 384-dim query embedding to a 384-dim partition-aligned space.
    
    Architecture:
        Linear(input_dim -> hidden_dim) -> ReLU -> Dropout
        Linear(hidden_dim -> output_dim) -> L2-normalize
    
    Used by:
        - train_mlp.py     during training
        - src/strategies/crag.py and SuperModel at query time
    """
    def forward(self, x: Tensor) -> Tensor: ...
```

---

### `train_mlp.py` — The Master Training Engine

The training loop dispatches all loss functions (`info_nce_single`, `info_nce_multi`, `kl_div`, `bce`, and the coverage-aware `coverage_kl` / `coverage_infonce` / `coverage`) and dynamically calculates the Dataset-Locked Temperature ($\tau$) and Hard Negative count ($hn\_k$). Coverage runs additionally accept `--lambda_cov`, `--cov_temperature`, `--target_topk`, `--topk_weight`, `--cvar_fraction`, `--positive_aggregation`. Use `--limit N` for a fast end-to-end smoke run.

```bash
# Example Training Execution for HNM Ablation
python -m src.alignment.train_mlp --dataset 2wiki --loss_type kl_div --epochs 100
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

Important checkpoint families:

- Architecture sweep: `checkpoints/{dataset}/alignment_{method}.pth`
- Loss sweep: `checkpoints/{dataset}/losses_ablation/alignment_mlp_{loss}.pth`
- Temperature sweep: `checkpoints/{dataset}/temp_ablation/alignment_mlp_{loss}_tau_{tau}.pth`
- HNM sweep: `checkpoints/{dataset}/hnm_ablation/alignment_mlp_{loss}_tau_{tau}_hnm_{k}.pth`
- Coverage sweep: `checkpoints/{dataset}/hnm_ablation/alignment_mlp_{coverage_loss}_tau_{tau}_hnm_{k}_lam_{lambda}.pth`
  (limited smoke runs additionally get a `_lim{N}` suffix so they never overwrite full-data checkpoints)

`SuperModel` prefers `BEST_COVERAGE_CHECKPOINTS[dataset]` (when set + present on disk) over `BEST_HNM_CHECKPOINTS[dataset]`, and falls back to older checkpoints when needed. Fill in `BEST_COVERAGE_CHECKPOINTS` after the lambda sweep selects the winning coverage model per dataset.
