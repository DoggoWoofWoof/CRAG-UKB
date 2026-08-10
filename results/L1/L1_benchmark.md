# L1 Partition-Routing Benchmark

_Generated 2026-07-29 13:49:07. Primary metric: **eqrrf6+bestof@20** (L1 partition FullCov@20). Oracle@20 is the greedy min-cover ceiling. Higher is better._

Seeds — splits `42`, offset-init `42`, fine-tune `42`.

| dataset | MiniLM-L6 | BGE-large | BGE-large-ft | gte-Qwen2-1.5B | oracle |
|---|---|---|---|---|---|
| musique_clean | 90.68 | 93.23 | 95.89 ✓ | — | 98.10 |
| 2wiki_clean | 89.60 | 90.80 | 96.07 ✓ | — | 100.00 |
| squad_clean | 89.97 | 92.42 | — | — | 100.00 |
| metaqa | 99.88 ✓ | 99.53 ✓ | — | — | 100.00 |
| hotpotqa_clean | 94.33 | 97.41 ✓ | — | — | 99.84 |

## ⚠️ Integrity warnings
- metaqa: npart MISMATCH across encoders {40: ['MiniLM-L6'], 401: ['BGE-large']} — not the same substrate!

## Missing cells (run to complete)
- **musique_clean / gte-Qwen2-1.5B**:
  - `python experiments.py run reencode-ukb --backend modal --gpu --account <acct> -- --datasets musique_clean --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --subdir gte_qwen --batch 16`
  - `python experiments.py run l1-rerank100 --backend modal --cpu --account <acct> -- --datasets musique_clean --subdir gte_qwen`
- **2wiki_clean / gte-Qwen2-1.5B**:
  - `python experiments.py run reencode-ukb --backend modal --gpu --account <acct> -- --datasets 2wiki_clean --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --subdir gte_qwen --batch 16`
  - `python experiments.py run l1-rerank100 --backend modal --cpu --account <acct> -- --datasets 2wiki_clean --subdir gte_qwen`
- **squad_clean / BGE-large-ft**:
  - `python experiments.py run l1-finetune-encoder --backend modal --gpu --account <acct> -- --datasets squad_clean --base BAAI/bge-large-en-v1.5 --subdir ft_bge --epochs 1 --batch 16 --max-seq 256`
  - `python experiments.py run l1-rerank100 --backend modal --cpu --account <acct> -- --datasets squad_clean --subdir ft_bge`
- **squad_clean / gte-Qwen2-1.5B**:
  - `python experiments.py run reencode-ukb --backend modal --gpu --account <acct> -- --datasets squad_clean --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --subdir gte_qwen --batch 16`
  - `python experiments.py run l1-rerank100 --backend modal --cpu --account <acct> -- --datasets squad_clean --subdir gte_qwen`
- **metaqa / BGE-large-ft**:
  - `python experiments.py run l1-finetune-encoder --backend modal --gpu --account <acct> -- --datasets metaqa --base BAAI/bge-large-en-v1.5 --subdir ft_bge --epochs 1 --batch 16 --max-seq 256`
  - `python experiments.py run l1-rerank100 --backend modal --cpu --account <acct> -- --datasets metaqa --subdir ft_bge`
- **metaqa / gte-Qwen2-1.5B**:
  - `python experiments.py run reencode-ukb --backend modal --gpu --account <acct> -- --datasets metaqa --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --subdir gte_qwen --batch 16`
  - `python experiments.py run l1-rerank100 --backend modal --cpu --account <acct> -- --datasets metaqa --subdir gte_qwen`
- **hotpotqa_clean / BGE-large-ft**:
  - `python experiments.py run l1-finetune-encoder --backend modal --gpu --account <acct> -- --datasets hotpotqa_clean --base BAAI/bge-large-en-v1.5 --subdir ft_bge --epochs 1 --batch 16 --max-seq 256`
  - `python experiments.py run l1-rerank100 --backend modal --cpu --account <acct> -- --datasets hotpotqa_clean --subdir ft_bge`
- **hotpotqa_clean / gte-Qwen2-1.5B**:
  - `python experiments.py run reencode-ukb --backend modal --gpu --account <acct> -- --datasets hotpotqa_clean --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --subdir gte_qwen --batch 16`
  - `python experiments.py run l1-rerank100 --backend modal --cpu --account <acct> -- --datasets hotpotqa_clean --subdir gte_qwen`
