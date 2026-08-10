# Level 1 — Partition Routing: the complete story

_L1 takes a query and returns the top-K **partitions**; L2/L3 then work **only inside** those
partitions. Primary metric = **partition FullCov@20** (fraction of queries whose **every** gold's
partition is in the top-20 — the condition for the downstream to recover the full answer). Read this
top-to-bottom; each step points to the script that produced it. All numbers on the frozen `_clean`
substrate (5 datasets). Seeds: splits `SPLIT_SEED=42`, offset-head `INIT_SEED=42`, fine-tune `FT_SEED=42`._

Datasets (partition count): musique_clean (136), 2wiki_clean (658), squad_clean (190), metaqa (401),
hotpotqa_clean (665).

---

## Step 0 — The mechanism L1 rides on: dense + overlap-voting
Retrieve the top-100 **nodes** by dense similarity; each node **votes** (RRF weight 1/(60+rank)) for its
**overlap-membership** partitions = its own partition **+ the partitions of its 1-hop graph neighbors**;
rank partitions by votes; take the top-20. The neighbor-voting *is a relational hop done structurally*
— a gold's partition can be covered even when the gold doc itself isn't retrieved (a neighbor votes for
it). This is why "dense" is already strong and why the learned offset is redundant *here* (see Step 6).

## Step 1 — Choosing the encoder  (`l1_benchmark.py` → `L1_benchmark.md`)
Same method, three frozen encoders. `eqrrf6+bestof@20`:

| encoder | musique | 2wiki | squad | metaqa | hotpot | mean |
|---|---|---|---|---|---|---|
| MiniLM-L6 | 90.68 | 89.60 | 89.97 | 98.45 | 94.33 | 92.61 |
| BGE-large | 93.23 | 90.80 | 92.42 | 99.53 | 97.41 | 94.68 |
| gte-Qwen2-1.5B | _stronger dense floor, see Step 4_ | | | | | |

Encoders are re-encoded non-destructively into a subdir (`reencode_ukb.py`; gte-Qwen2 needed a prebuilt
flash_attn wheel + `transformers==4.44.2`). Integrity enforced: every encoder shares npart + n_test per
dataset. → **stronger encoder helps; it's the main L1 lever.**

## Step 2 — The method: ONE universal head, and which head  (`l1_universal_head.py`)
Goal = a **single model** (one frozen encoder + one offset head trained once on pooled (query→gold)
triples across all 5 datasets — no per-dataset weights). We compared candidate heads. BGE, equal fusion:

| head | standalone | dense+head (mean) | musique | 2wiki | squad | metaqa | hotpot |
|---|---|---|---|---|---|---|---|
| dense only | — | 93.50 | 85.9 | 91.9 | 91.8 | 98.9 | 99.0 |
| **base** (plain offset) | 92.00 | **93.88** | 89.1 | 89.9 | 92.7 | 99.8 | 97.9 |
| hard (offset + hard-neg) | 91.74 | 93.61 | 88.4 | 89.7 | 92.0 | 99.8 | 98.2 |
| mix = mlpT (K-mixture) | 91.73 | 93.54 | 88.8 | 88.9 | 92.6 | 99.8 | 97.7 |

**`base` wins; mlpT is last; hard-negatives don't transfer universally.** So the L1 head is a plain
learned offset — **not** mlpT. (mlpT's real home is L2 — Step 6.)

## Step 3 — Fusion: best-of, not equal-RRF  (`l1_universal_head.py`)
Equal-RRF *dilutes* where dense is strong. best-of = per-node min-rank across dense+head → keeps each
retriever's strongest votes. BGE, head=base:

| fusion | mean | ≥95 | musique | 2wiki | squad | metaqa | hotpot |
|---|---|---|---|---|---|---|---|
| equal | 93.88 | 2/5 | 89.1 | 89.9 | 92.7 | 99.8 | 97.9 |
| **best-of** | **94.88** | **3/5** | 88.0 | **95.7** | 92.3 | 99.4 | 99.0 |
| gated | 94.08 | 2/5 | 86.9 | 94.2 | 91.8 | 98.9 | 98.7 |

best-of rescued 2wiki (89.9→95.7) and lifted the mean +1.0. **best-of is the fusion.** gated (my
confidence gate) underperformed plain best-of.

## Step 4 — Best single-model config: gte-Qwen2 + best-of  (`L1_universal_head_gte_qwen.json`)
| fusion | mean | ≥95 | musique | 2wiki | squad | metaqa | hotpot |
|---|---|---|---|---|---|---|---|
| dense only | 94.92 | — | 89.5 | 93.9 | 94.3 | 97.6 | 99.4 |
| **best-of** | **95.52** | 2/5 | 91.3 | 94.5 | 94.5 | 98.0 | 99.2 |

**gte lifts the dense floor** (musique 85.9→89.5, squad 91.8→94.3). Best single-model number =
**mean 95.52 FullCov@20**, but 2wiki/squad land at 94.5 (a whisker under) and musique 91.3 is the wall.
(Note: gte is *worse* than BGE on the pure-KB metaqa, 98.0 vs 99.5.)

## Step 5 — All the metrics, not one  (`l1_recall.py`)
FullCov@20 is the strict all-golds view. The fuller picture (gte, dense+overlap-voting):

| dataset | FullCov@20 | partRecall@20 | docRecall@20 | docRecall@100 | docRecall@200 |
|---|---|---|---|---|---|
| musique | 89.57 | **95.40** | 79.4 | 88.5 | 91.8 |
| 2wiki | 93.87 | **97.75** | 70.3 | 75.2 | 77.4 |
| squad | 94.11 | **95.13** | 91.9 | 95.9 | 97.1 |
| metaqa | 98.97 | **99.14** | 40.9 | 57.5 | 64.9 |
| hotpot | 99.51 | **99.76** | 97.4 | 99.4 | 99.7 |
| MEAN | 95.21 | **97.44** | 76.0 | 83.3 | 86.2 |

- **partial partition-recall is ≥95 on all 5** — the "laggards" were an artifact of the strict all-golds metric.
- **docRecall exposes the real hole**: metaqa 40.9, 2wiki 70.3 — the gold *docs* are barely retrieved, yet
  their partitions are covered anyway (overlap-voting). The doc→partition gap = the graph doing the work.

## Step 6 — Miss analysis: what actually blocks FullCov@20  (`l1_blocker.py`)
Every FullCov@20 miss classified (gte, laggards):

| dataset | %fail | OUTRANKED | NOT_RETRIEVED | NOT_VOTED |
|---|---|---|---|---|
| musique | 10.4 | 69.8 | 30.2 | 0 |
| 2wiki | 6.1 | 65.7 | 34.3 | 0 |
| squad | 5.9 | 80.2 | 19.8 | 0 |

`NOT_VOTED=0` (overlap-voting is sound). **OUTRANKED dominates (66–80%)**: the gold partition IS voted,
just at median rank ~31 (≈half at ≤30 — *just* outside top-20). NOT_RETRIEVED (20–34%) = the retrieval
wall (deep multi-hop, gold unreachable). Splitting all misses ≈ thirds: **~1/3 just past the budget**
(→ bigger K, Step 8), **~1/3 retrieval wall** (→ L2/L3), **~1/3 ranking tail** (→ encoder, exhausted).

## Step 7 — What does NOT help (ruled out with data)
- **IDF / hub-partition down-weighting** (`l1_idf_test.py`): HURTS monotonically (mean 95.2→92.7→89.8→84.1).
  The frequent partitions are genuinely relevant → the OUTRANKED gold is weakly voted, not out-voted by noise.
- Learned rerankers (attn/mlp) < RRF; deeper voting hurts (more noise partitions); membership tuning & PPR dilute.
- **Conclusion: the L1 FullCov@20 gap is encoder-bound; the non-encoder ranking levers are exhausted.**

## Step 8 — The payoff: paging  (`l1_paging.py`)
If L2/L3 get ONLY the top-K partitions (not the full graph): **FullCov@K vs % of corpus paged in** (gte):

| dataset | npart | K=20 | K=50 | K=100 |
|---|---|---|---|---|
| musique | 136 | 89.6 / 15% | 97.1 / 37% | 99.5 / 74% |
| 2wiki | 658 | 93.9 / **3%** | 97.3 / **8%** | 98.6 / 15% |
| squad | 190 | 94.1 / 11% | 98.0 / 27% | 99.3 / 53% |
| metaqa | 401 | 99.0 / **5%** | 99.9 / **13%** | 100 / 25% |
| hotpot | 665 | 99.5 / **3%** | 99.8 / **8%** | 99.8 / 15% |
| MEAN FullCov | | 95.2 | **98.4** | 99.5 |

**L3-on-partitions-only works, near-losslessly** at K=50 (98.4% complete, ~8–13% of the graph on the
high-partition datasets). The scope win is governed by **partition granularity** — great on high-npart
sets, weak on coarse musique/squad (finer partitioning is the substrate lever there).

---

## L1 verdict
- **Method:** single universal model = frozen **gte-Qwen2** + one plain-offset head + **best-of** fusion,
  over **dense + overlap-voting** partition routing.
- **Number:** mean **FullCov@20 = 95.52** (partial partition-recall ≥95 on all 5); K=50 → 98.4 for paging.
- **Honest ceiling:** strict all-5-≥95 @K=20 is encoder-bound and not reached with a single frozen model;
  the residual is ~1/3 budget (bigger K), ~1/3 retrieval-wall (→ L2/L3), ~1/3 encoder.
- **Key finding:** the relational offset/mlpT is **redundant at L1** (overlap-voting already does the hop)
  but **irreplaceable at L2** — docRecall is the real hole (metaqa 41, 2wiki 70). **mlpT → L2 seed-finding.**

## Handoff to L2
L1 hands L2 the **top-K partitions**. L2's job = find the **seed docs** inside them, where dense doc-recall
is weak — exactly the +37 (metaqa) / +15 (musique) offset regime. **Next: the L2 seed-finding eval —
gold-doc recall@{5,20,50} within the routed partitions, dense vs dense+offset/mlpT.**

## Reproduce (in order)
```bash
# 1. encoders
python experiments.py run reencode-ukb --backend modal --gpu --account <a> -- --datasets <d> --model BAAI/bge-large-en-v1.5 --subdir bge_large
python experiments.py run reencode-ukb --backend modal --gpu --account <a> -- --datasets <d> --model Alibaba-NLP/gte-Qwen2-1.5B-instruct --subdir gte_qwen
python experiments.py run l1-benchmark                                    # -> L1_benchmark.md
# 2-4. method + fusion (one universal model, all datasets)
python experiments.py run l1-universal-head --backend modal --gpu --account <a> -- --subdir gte_qwen --heads base --tr-cap 3000 --te-cap 2000 --epochs 20
# 5-8. analysis / ablations (local, read-only)
python -m src.experiments.l1_recall  --subdir gte_qwen
python -m src.experiments.l1_blocker --subdir gte_qwen
python -m src.experiments.l1_idf_test --subdir gte_qwen
python -m src.experiments.l1_paging  --subdir gte_qwen
```

## Canonical files in this folder
- `README.md` — this document; the readable, sequential L1 story with every table (Steps 0–8)
- `L1_benchmark.{md,csv,json}` — encoder selection (Step 1)
- `L1_universal_head_bge_large.json`, `L1_universal_head_gte_qwen.json` — head × fusion comparison (Steps 2–4)
- `L1_universal_findings.md` — the mechanism writeup (offset redundant at L1 → belongs at L2)
- `L1_universal_method.md` — method + reproducibility notes

The analysis tables (Steps 5–8: recall / miss-types / IDF / paging) live inline above; regenerate the raw
numbers any time with `l1_recall.py` / `l1_blocker.py` / `l1_idf_test.py` / `l1_paging.py` (all read-only).
