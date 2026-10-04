# L2 RELATION — Canonical Correctness Phase (masked semantics + encoder-invariance bound)

**Purpose:** before Relation feeds Shapley / the soft controller, fix two correctness issues flagged after
the proxy phase — (A) sparse-expert **mask semantics** (absent candidates must ABSTAIN, not be ranked),
and (B) the **canonical gte-Qwen2 encoder** vs the MiniLM proxy. Selection split = **VAL only. TEST never
inspected. L1 untouched. No controller, no Shapley, no expert dropout, no other datasets, no L3.**

Deliverables: this file · `results/L2/L2_RELATION_CANONICAL.json`. Code: `scratchpad/l2_relation_masked.py`
(the generic `l2_lib.rrf_fuse` is left untouched; masking is a separate `masked_rrf_fuse` interface).

---

## PART A — Mask semantics + tie-order audit

### A1. The tie-order leak is REAL (`RELATION_TIE_LEAK_FOUND = YES`)

The legacy path set `score = -1` for candidates with no topic→candidate relation and then ranked the whole
P50 vector with `np.argsort(-score, kind="stable")`. Because `cand` is stored in canonical **dense-desc**
order, all the tied `-1` candidates sort **after** the eligibles in their original array order = dense order.

Audit (`audit_tie_leak`), queries with ≥5 absent candidates:

| | 2wiki val | musique val |
|---|---|---|
| queries checked | 3000 | 3987 |
| `RELATION_ABSENT_CANDIDATES_RECEIVE_RANKS` | **YES** | **YES** |
| absent ranks == dense-order positions (frac) | **1.0** | **1.0** |

`RELATION_TIE_ORDER_SOURCE` = *stable argsort of −score over the full scope; absent candidates all tie at
−1 → placed after eligibles in original array order == canonical dense-desc scope order.* So every absent
candidate received a deterministic **dense-correlated** rank and an accidental `1/(k0+rank)` RRF vote.

**⇒ `LEGACY_RELATION_RESULTS_AFFECTED = YES`.** The prior relation-only MRR, best-gold ranks, disagreement
counts, and fixed-RRF gains are **PROVISIONAL** (preserved, not deleted).

### A2. Masked interface

Relation now emits two aligned arrays, `relation_score[pair]` and `relation_mask[pair]`, where
`mask=1 ⇔ a valid topic→candidate connecting sentence exists`. `MASKED_RELATION_RRF`: a masked-in candidate
contributes `1/(k0 + rank_among_eligible)`; a **masked-out candidate contributes exactly 0 (ABSTAIN) — no
pseudo-rank.** The mask is **pure graph topology** (`cand_sent≥0`) and therefore **encoder-independent**.

### A3. Masked coverage & conditional metrics (VAL)

| | 2wiki val | musique val |
|---|---|---|
| `RELATION_QUERY_COVERAGE` | 0.744 | 0.460 |
| `ANY_GOLD_RELATION_COVERAGE` | **0.570** | **0.142** |
| `ALL_GOLD_RELATION_COVERAGE` | 0.0 | 0.0 |
| mean eligible candidates / query | **1.76** (p95 5, max 12) | **2.67** (p95 14, max 82) |
| `COND_RELATION_GOLD_PRESENT` MRR / ANY@10 | 0.850 / 1.000 | 0.837 / 0.991 |

Relation is a **near-binary eligibility signal**: it fires on ~2 candidates per query and abstains on
99.8% of the P50 scope. `ALL_GOLD_RELATION_COVERAGE=0` everywhere — it only ever reaches the *second-hop
bridge* gold, never the whole gold set.

### A4. Masked fixed-fusion (VAL, COND_ANY_GOLD_IN_SCOPE)

| config | 2wiki MRR | R@5 | ALL@10 | ALL@50 | musique MRR | R@5 | ALL@10 | ALL@50 |
|---|---|---|---|---|---|---|---|---|
| E3 (d+s+off+mix) | 0.9286 | 0.6470 | 0.5233 | 0.7670 | 0.9213 | 0.8195 | 0.7849 | 0.9518 |
| **E3 + masked relation** | 0.9099 | **0.7469** | **0.6657** | **0.8927** | 0.9042 | **0.8288** | **0.7952** | **0.9558** |
| Δ | −1.9 | **+10.0** | **+14.2** | **+12.6** | −1.7 | +0.9 | +1.0 | +0.4 |

- **2wiki:** big deep-recall / multi-gold lift; small **top-MRR cost** (equal-weight RRF perturbs the already
  rank-0 first-hop gold — the same artifact seen in E0–E3 that motivates the query-aware soft controller).
- **musique:** **NEUTRAL** (tiny positive recall, small MRR cost).

### A5. GOLD-level rescue — the correct metric for a sparse bridge expert

Query-best rescue is confounded by the *easy first-hop gold* in multi-gold queries. Isolating each
relation-**eligible gold** and comparing its E3 vs E3+relation rank:

| | 2wiki val | musique val |
|---|---|---|
| eligible golds | 2073 | 573 |
| `GOLD_HURT` (all K = 5/10/20/50) | **0 / 0 / 0 / 0** | **0 / 0 / 0 / 0** |
| eligible golds the base stack MISSED (min base rank ≥50) | **24.1%** | 2.8% |
| genuinely base-missed (≥50) golds recovered into top-50 | **323** | 14 |
| … into top-20 / top-10 | 129 / 66 | 3 / 2 |

On **2wiki**, relation recovers **323 second-hop bridge golds that all four base experts placed at rank ≥50
(median 66)** into the top-50, into top-10 for 66 of them — with **zero gold demotions at any K**. Example
(qi=331): a bridge gold at dense 288 / splade 48 / offset 71 / mixture 47 → E3 45 → **E3+relation 5**. On
**musique** the eligible golds are already easy for the base (median base rank 1); the hard multi-hop chains
are simply **not relation-eligible** (the corpus title-mention graph lacks them — the known musique
reachability ceiling), so relation has little to add.

### A6. Regime raw material fixed (non-destructive)

`results/L2/_regime_raw/{2wiki,musique}_clean_val.npz` now carries:
`best_gold_rank_relation_proxy_legacy` (renamed, kept — do **not** use for supervision),
`best_gold_rank_relation_masked` (**−1 = NO_RELATION_RANK**, i.e. relation abstains on every gold; else rank
among relation-eligible candidates), plus controller features `relation_gold_present`,
`relation_any_signal_present`, `relation_num_candidates`. The raw base-expert regime columns are untouched.

---

## PART B — Canonical gte-Qwen2 encoder

### B1. Query embeddings are reusable — `RELATION_Q_EMBED_REUSE = PASS`

`encoder_swap.eq()` encodes as `normalize(gteQwen(query_instruction + text))`. The cached
`data/ukb_storage/<ds>/gte_qwen/queries_all.npy` (model `Alibaba-NLP/gte-Qwen2-1.5B-instruct`, dim 1536,
L2-normalized, same instruction) **is exactly** the relation encoder's `E(q)`. No query re-encode is needed;
the only Qwen-side work would be the ~5,100 unique connecting **sentences** (2wiki 3,464 + musique 1,617).

### B2. The mask is encoder-independent; the encoder only reorders the ~2 eligible candidates

Since eligibility is pure topology, gte-Qwen2 and MiniLM produce the **identical eligible set** per query.
An encoder can only change the **order** of that set, and hence each eligible candidate's RRF vote by at most
`1/60 − 1/(60+m−1)` (m = eligible count). With mean m≈2, that swing is ≤ **2.7e-4**.

### B3. Encoder-invariance bound — brackets ALL encoders at once (`QWEN_MASKED_RESULT`)

We recomputed masked **E3+relation** under four orderings of the eligible set — `minilm` (actual cosine),
`tied_dense` (all eligible tied → pure eligibility, **no encoder**), `reverse` (adversarial), `random`.
gte-Qwen2's ordering is necessarily a permutation inside this bracket. **Max spread across all four:**

| metric | 2wiki spread | musique spread |
|---|---|---|
| MRR | 0.0006 | 0.0007 |
| R@5 | 0.0004 | 0.0005 |
| R@50 | 0.0011 | 0.0001 |
| ALL@10 | 0.0010 | 0.0005 |
| ALL@50 | 0.0030 | 0.0000 |

**Every masked metric moves ≤ 0.3pp (2wiki) / ≤ 0.07pp (musique) across the four TESTED orderings.**

`RELATION_ENCODER_SENSITIVITY = LOW_ON_TESTED_ORDERINGS`. **This is sensitivity evidence, NOT an exhaustive
proof over every ordering an arbitrary encoder could induce** — the four probes (actual MiniLM cosine,
tied-dense, reverse, random) sample the ordering space but do not derive an oracle-best / adversarial-worst
bound. It indicates the masked result is *unlikely* to move materially with a different encoder, which is
why the mask (topology) — not the cosine tiebreak among ~2 candidates — carries relation's value.

**This does not substitute for the canonical encoder.** gte-Qwen2 is the locked canonical dense encoder and
the Relation expert must use canonical gte-Qwen2 before entering Shapley/controller (see
`L2_RELATION_QWEN.md`); MiniLM is proxy/diagnostic only.

---

## Three-row comparison — answers "encoder vs topology vs tie-order" (2wiki, E3+relation vs E3)

| row | MRR | R@5 | ALL@10 | ALL@50 |
|---|---|---|---|---|
| E3 base | 0.929 | 0.647 | 0.523 | 0.767 |
| `RELATION_PROXY_MINILM_LEGACY_RRF` | 0.931 | 0.720 | 0.630 | 0.834 |
| `RELATION_PROXY_MINILM_MASKED_RRF` | 0.910 | **0.747** | **0.666** | **0.893** |
| `RELATION_QWEN_MASKED_RRF` | = masked ±0.0006 | ±0.0004 | ±0.0010 | ±0.0030 |

**Decomposition of the effect:**
- **Accidental tie-order (LEGACY → MASKED):** materially misleading. It manufactured a **fake musique
  degradation** (legacy ALL@10 **−5.0 / 13 hurt** → masked **+1.0 / 0 gold-hurt**) and, on 2wiki, hid the
  real trade (its dense-shaped leak propped MRR up to 0.931 while understating recall). Correcting it makes
  2wiki's recall gain *larger and cleaner* and musique *neutral, not harmful*.
- **Encoder choice (MiniLM → Qwen):** ≤ **0.3pp** (B3). Negligible.
- **Relation topology / mask:** the entire real effect — +10 R@5, +14 ALL@10, 323 base-missed bridge golds
  recovered, 0 gold-hurt on 2wiki.

---

## `RELATION_VALUE = MIXED` → **RESCUE_SPECIALIST (2wiki) / NEUTRAL (musique)**

With the leak removed, relation is a **bridge-gold RESCUE_SPECIALIST**, realized *cooperatively* (it abstains
on 99.8% of candidates, so it cannot rank a list alone — its sparse vote only lifts eligible bridge golds
inside the fusion). It recovers second-hop golds the full dense+splade+offset+mixture stack cannot reach,
with zero gold demotions, exactly where the corpus graph carries the chain (2wiki). Where the graph lacks
the chains (musique), it is NEUTRAL. This is a textbook **gateable** expert for the soft controller — turn it
ON where the query has a graph bridge, OFF otherwise; fixed equal-weight fusion is *not* how to add it
(it costs top-MRR on already-solved queries).

---

## Decision gates

| gate | verdict |
|---|---|
| `RELATION_MASK_IMPLEMENTED` | **YES** |
| `RELATION_TIE_LEAK_FOUND` | **YES** (absent→dense-order ranks, frac 1.0 both datasets) |
| `LEGACY_RELATION_RESULTS_AFFECTED` | **YES** → legacy relation-only/best-gold/disagreement/fixed-RRF = PROVISIONAL |
| `RELATION_Q_EMBED_REUSE` | **PASS** (`queries_all.npy` == `eq(query)`) |
| `MINILM_MASKED_RESULT` | 2wiki R@5 .747 (+10.0) / ALL@10 .666 (+14.2) / ALL@50 .893 (+12.6) / MRR .910 (−1.9); musique NEUTRAL; GOLD_HURT=0 both |
| `QWEN_MASKED_RESULT` | pending canonical build (`L2_RELATION_QWEN.md`); `RELATION_ENCODER_SENSITIVITY = LOW_ON_TESTED_ORDERINGS` (4 probes ≤0.3pp — evidence, not a universal proof) |
| `RELATION_VALUE` | **MIXED** — RESCUE_SPECIALIST (2wiki) / NEUTRAL (musique) |
| `KEEP_RELATION_FOR_CONTROLLER` | **YES** |
| `FINAL_EXPERT_SET_READY` | **YES** — {Dense, Exact-SPLADE, Offset, Mixture, Relation(masked)} |
| `SAFE_TO_BUILD_RELATION_TRAIN` | **YES** (VAL decision = keep) |
| `SAFE_TO_START_SHAPLEY` | **NO** — requires the TRAIN relation matrix (masked) built first; VAL-only at present |
| `SAFE_TO_START_CONTROLLER` | **NO** — waits for TRAIN relation + the Shapley target phase |

---

## STOP

Reporting canonical **masked VAL** Relation behavior and stopping, per instruction. Not started: TRAIN
relation build (now unblocked), Shapley, controller, expert dropout, Hotpot/SQuAD/MetaQA/WebQSP, L3, any
TEST inspection, any L1 change. The single genuinely-remaining artifact before Shapley is the **TRAIN**
relation matrix under masked semantics (VAL is decided and encoder-invariant).
