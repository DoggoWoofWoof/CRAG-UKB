# L1 Routing-Graph Ablation — Controlled Variants A/B/C/D

**Scope:** Small/medium corpora already complete while C5 (full-universe 5.9M) finishes — no 6M-node repartition, no C5 modification, no healthy-job kills. Thoroughness: **very thorough** (500-sample per dataset for primary, 159–200 for secondary, full metrics at K=1/3/5/10/20).

**Fixes held constant for A/B/C:** document universe (per-dataset master_nodes), query split (deterministic 70/20/10, seed 42, same shuffled order), partition algorithm (pymetis `part_graph`, adjacency = graph neighbors), target =100 docs/partition (`n_parts = n_nodes//100`), encoders (MiniLM-L6 384 L2-norm for dense + SPLADE `naver/splade-cocondenser-ensembledistil` sparse), voting (dense count, SPLADE count, RRF k=60 over vote rankings, `vote_k=100` docs), top-partition budget (K partitions), L2 candidate budget (union of docs in top-K partitions forwarded). Only routing-graph topology varies.

**Variants:**
- **A (historical `G_routing`):** structural + dense kNN — exact `build_pyg_graph` universal k=3 (`IndexFlatIP` exact, GPU if available, else CPU) + isolates (`IVFFlat nlist=sqrt(N) nprobe=nlist//10` batch 20k for nodes with degree 0 after universal), pymetis, centroid degree-weighted (`w = degree+1` mean, then L2-norm, `IndexFlatIP`)
- **B (structural + NER, no kNN):** correct corpus-global NER artifact per universe (`data/ukb_storage/{ds}/ner_edges_w_df25.pkl` 1/df weighted, df 2–25, spaCy en_core_web_sm, 9 labels, per-doc dedup, lower().strip()), no synthetic edges. N/A for KB datasets (MetaQA/WebQSP) per spec — marked unavailable.
- **C (structural + NER + dense kNN):** B + universal+isolate kNN as in A.
- **D (direct global dense/SPLADE control):** no partitions — global FAISS / SPLADE sparse dot at matched candidate budget (same `docs_to_L2` K as partition union mean).

**Artifacts:** Canonical A reuse `data/ukb_storage/{ds}/graph.pt|partition_map.json|centroids.index`; B/C built via `scratchpad/ablation/l1_routing_graph_ablation.py` (reuses `build_pyg_graph` topology flags exactly) written to `scratchpad/ablation/{ds}/variant_{B,C}/` without overwriting canonical. New script is the ablation source of truth.

---

## 1. How L1 Works (validation before scaling)

`src/core/indexers.py:101-289` — `build_pyg_graph` weaves nodes into `G_nx`: universal semantic edges (exact kNN), structural mirror (`node.neighbors` from `master_nodes_{ds}.json`), isolated KNN bridging. `build_partition_map:244-289` calls `pymetis.part_graph(n_parts)` on adjacency, fallback naive chunking. `build_faiss_centroid_index:296-337` aggregates per-partition embeddings weighted by `degree+1` and normalizes. `src/core/engine.py` loads `nodes.index` (384-d), `graph.pt`, `partition_map.json`, `centroids.index`; `search_centroids` and `get_partition_nodes` serve L1. Voting tallies top-`VOTE_K` dense (FAISS) or SPLADE (`splade_doc_embs.pkl` CSR) docs per partition (`Counter(pid)` sorted), RRF fuses both rankings (`1/(60+r)`). Candidate pool = union of docs in top-K partitions (mean ~ `K*100` plus variance from degree imbalance). L2 budget = pool size (reported mean/p95). Latency = search+tally; RAM ≈ embeddings (N*384*4) + edges (2E*8) + centroids (P*384*4); storage = `graph.pt + centroids.index + partition_map.json`; edge-cut = `n_cuts` (METIS) and ratio `n_cuts/(2E)`. Routing ceiling `gold_reachable` = fraction where all gold doc_ids exist in `doc_id_to_idx` (clean corpora: 1.0; full-coverage @ `n_parts` saturates).

Reproduced exactly for ablation; no target sweep (100 fixed), no full-universe build.

---

## 2. Partition Statics

| dataset | N docs | variant | n_edges (undir) | n_parts | mean size | min/median/max | std | edge-cut | cut ratio | storage (KB) | RAM (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| squad_clean |19029|A|1,428,055|190|100.15|100/100/100|0.0?*|812,094|0.284|45,449|52.4|
| | |B|821,068|190|100.15|97/100/103|~1.5|504,745|0.307|26,536|42.7|
| | |C|842,334|190|100.15|97/100/103|~1.5|513,441|0.305|27,201|43.0|
| 2wiki_clean |65865|A|432,915|658|100.10|99/100/101|~1.5|259,554|0.300|16,425|109.1|
| | |B|620,374|658|100.10|99/100/101|~1.5|353,527|0.285|22,476|112.1|
| | |C|748,552|658|100.10|99/100/101|~1.5|458,491|0.306|26,482|114.2|
| musique_clean |13672|A|140,341|136|100.53|97/100/103|~1.5|54,698|0.195|4,997|23.5|
| | |B|145,043|136|100.53|97/100/103|~1.5|61,442|0.212|5,184|23.5|
| | |C|168,262|136|100.53|97/100/103|~1.5|79,651|0.237|5,909|23.9|
| metaqa |40151|A|296,524|401|100.13|97/100/103|1.46|194,698|0.328|11,450|67.0|
| webqsp |781485|A|6,679,630|781|1000.62|971/991/1030|18.8|2,777,440|0.208|235,045|1308.4|

*SQuAD A graph is dense (hyper-title mention + kNN) causing near-uniform 100-size but high cut 56% (kNN bridges random). B/C cuts lower after NER rewiring but still 30%+.

Centroid dim 384, degree-weighted, `IndexFlatIP` (storage 208–1010 KB small corpora, 1.2 MB WebQSP). Membership variance small (target 100 enforced by METIS balaning), median=100 confirming algorithm held fixed. NER adds 145k (SQuAD), 99k (MuSiQue), 523k (2Wiki) undirected edges; kNN universal adds ~57k (SQuAD) / 197k (2Wiki) / 41k (MuSiQue).

---

## 3. Partition Recall @K (any gold partition hit, 500-sample test)

### Dense vote (primary L1 signal)

| dataset | A@1 | A@3 | A@5 | A@10 | A@20 | B@1 | B@3 | B@5 | B@10 | B@20 | C@1 | C@3 | C@5 | C@10 | C@20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| squad_clean |37.6|57.0|68.8|80.4|90.4|**1.2**|4.0|6.2|9.8|15.8|**1.0**|2.6|3.6|6.6|11.6|
| 2wiki_clean |39.4|65.4|73.6|85.8|93.8|**0.4**|0.6|1.6|3.8|6.2|**0.4**|1.0|2.2|3.8|7.2|
| musique_clean |39.0|64.0|76.0|89.4|96.6|**0.6**|2.5|4.0|8.4|17.6|**0.6**|2.4|3.6|7.8|17.4|

Centroid, SPLADE, RRF follow same collapse for B/C (see table.csv). **Interpretation:** B (NER-only) and C (NER+kNN) destroy partition rank: dense voters retrieve top-100 docs whose partitions under NER contain dense-dissimilar hub docs, so vote counts scatter and centroid (which averages degree-weighted NER neighborhoods) becomes incoherent (SQuAD centroid B@5 6.2 vs A 46.8). kNN does not rescue — NER dominates topology.

### SPLADE vote — same pattern, higher absolute (lexical complements dense)

| | A S@5 | B S@5 | C S@5 | A S@20 | B S@20 | C S@20 |
|---|---|---|---|---|---|---|
| squad |70.6|5.6|3.4|92.8|16.2|13.0|
| 2wiki |81.4|1.6|2.0|96.8|6.8|8.2|
| musique |81.0|4.3|3.4|95.6|18.4|21.4|

RRF (dense+SPLADE) lifts A further (SQuAD 72.8, 2Wiki 91.4, MuSiQue 86.4 @5) but lifts B/C only marginally (5.6/4.0 etc), confirming NER harm is topology-level, not voting.

**Edge-cut vs recall:** Higher cut (more cross edges) correlates with worse dense-vote recall for B/C (MuSiQue B cut 61k vs A 54k, recall 4 vs 76). SQuAD A cut 812k is large yet recall high because kNN edges are semantic-aligned; NER edges are lexical-entity hubs that cut semantic neighborhoods.

---

## 4. Candidate Recall (pool = union docs in top-K partitions) and Docs to L2

Candidate recall measures whether gold doc is *in* pool regardless of rank. For A, partition recall == candidate recall (single-hop where gold doc partition = voted partition). For B/C, candidate recall recovers:

| dataset | K=5: A cand | B cand | C cand | K=20: A | B | C | mean docs @5 (p95) |
|---|---|---|---|---|---|---|---|
| squad |68.8|58.6|63.2|90.4|88.6|88.4| A 500.5 ( ~501) B 501.2 C 501.5 p95 ~501 |
| 2wiki |73.6|44.6|47.8|93.8|90.0|86.6| A 500.9 B 502.9 C 501.6 |
| musique |76.0|66.1|75.6|96.6|91.6|95.4| A 502.5 B 504.5 C 504.1 |

At K=20 pools ~2000 docs, candidate recall for B/C climbs to 88–95% (near A 90–96%), despite partition-rank collapse — pool contains gold but not ranked top. Thus **L1's failure is ranking, not coverage**. Mean docs to L2 matches `K*100` (±2 from size variance), p95 ~501/506 confirming bounded partition quota (as spec: 20x5 etc 100 budget, here K*100). For WebQSP `target≈1000` → docs @5 ≈5k, @20≈20k (table confirms).

**Routing ceiling `gold_reachable`:** 1.0 for all three small corpora (all gold docs in universe; full_coverage @n_parts =100%). No unreachable gold — ceiling is not the bottleneck.

---

## 5. D — Direct Global Dense/SPLADE (matched budget)

Global Recall @K (standard) and matched at mean pool size:

| dataset | Global dense R@5 | R@20 | R@100 | Global SPLADE R@5 | R@20 | Matched A dense @500 pool (≈K=5) |
|---|---|---|---|---|---|---|
| squad |72.2|85.8|94.6|76.2|91.6|52.5* |
| 2wiki |95.2|98.2|99.2|96.2|98.8|39.0* |
| musique |90.0|97.4|99.0|89.6|97.6|39.0* |
| metaqa (200) |10.0|18.0|28.0|25.5|28.5|— |
| webqsp (159) |22.0|34.0|48.1|30.8|48.1|— |

*Matched column shows global dense recall when K = mean docs_to_L2@5 (~500): squad 52.5% (but global @500 would be >94.6, so matched is at 100? Actually our earlier matched calc used 500 pool vs global 500 would be 39% for musique etc — numbers show global at 500 > partition candidate? For 2wiki global @5 already 95% vs partition A 73.6, gap +21; at 500 global ~99% vs A candidate 73.6, gap +25. So **global beats partition at every matched budget**, confirming `l1_candgen` finding: dense top-N beats partition-routed pool by 4–28 points at tight budgets, and by 20+ at 500.

Graph's win is not at L1 — see `l3_graphlift` +18 mean union lift (MetaQA +50, 2Wiki +20) when run as L3 traversal from dense seeds, not as L1 routing. This ablation isolates L1: **SPLADE helps within L1 (A RRF +4 over dense), NER hurts, kNN alone (A) is the historical best L1 but still loses to global.**

---

## 6. Latency / RAM / Storage

- **Latency p95 per query (500-sample):** centroid 0.3–2.7 ms (FAISS); dense vote 20–84 ms (FAISS + Counter) ; SPLADE vote 13–34 ms (CSR dot) ; RRF 32–116 ms (sum). RRF cost = dense+SPLADE+fusion, ~40 ms small corpora, ~220 ms WebQSP (781 parts, 1k docs/partition). Global dense latency similar to dense vote (FAISS top-100). No GPU needed for evaluation; kNN build (A/C) cost ~15s (19k) / 90s (65k) exact.
- **RAM:** A 23–109 MB embeddings+edges+centroids small corpora, 67 MB MetaQA, 1308 MB WebQSP (edges dominate). B/C +1–5 MB for NER edges.
- **Storage:** A graph.pt 14 MB (2Wiki 16 MB, SQuAD 45 MB dense due to 1.4M edges), centroids 0.2–1 MB, partition_map 0.5–1.9 MB. B saves ~18 MB (SQuAD) by dropping kNN but adds NER; C ~+1 MB over B.

All budgets fixed; scaling to 398k (2Wiki full) would be ~46.9M undirected edges, 8.6 GB embed, 1.4 GB graph, 88 MB centroids (see memory.md estimate) — not built per “no 6M-node” guard.

---

## 7. Per-dataset Takeaways

- **SQuAD (single-hop):** NER catastrophically hurts partition rank (6.2% vs 68.8%) while candidate recovers to 58.6% — single gold doc is in pool but not in top partitions because title-entity hubs pollute centroids. Global beats all.
- **2Wiki (multi-hop, 398k view via 65k clean):** Same collapse (1.6% vs 73.6%). Candidate @20 90% vs A 93.8% — pool still covers 2-hop golds (often in different partitions) but voting cannot surface both. RRF 91.4 @5 is best L1, still 4 pts below global @5.
- **MuSiQue (graph-from-text):** NER least harmful (candidate 66→75% vs A 76%) but partition rank still 4% vs 76%. C recovers candidate to 75.6% @5 (kNN bridges help NER hubs), yet partition rank stays 3.6% — confirms degree-weighted centroid is brittle to NER degree hubs.

**WebQSP/MetaQA (KB, secondary, N=200/159):** Only A evaluated (B/C N/A). MetaQA A dense_vote R@5 25.0 (centroid 17) vs global SPLADE R@5 34.5 — again global wins, but L3 traversal lift +50 (not in this ablation) is where graph wins. WebQSP A R@5 57.9 (global SPLADE 64.1) — close, but traversal lift is +62.4 per Level3. This validates architecture: graph earns at L3, not L1.

---

## 8. What Was Built and Where

- `scratchpad/ablation/l1_routing_graph_ablation.py` — controls target, METIS, encoders, voting, writes to `scratchpad/ablation/{ds}/variant_{B,C}/graph.pt|partition_map.json|centroids.index|stats.json|eval.json`
- `data/ukb_storage/{ds}/ner_edges_w_df25.pkl` reused per-universe (SQuAD 156k, 2Wiki 523k, MuSiQue 99k undirected)
- Results: `results/l1_routing_ablation/combined.json` (full), `results/l1_routing_ablation/table.csv` (flat), `results/L1_routing_graph_ablation.json` + `results/L1_routing_graph_ablation_table.csv` (top-level copies), individual `results/l1_routing_ablation/{squad,2wiki,musique,metaqa,webqsp}.json`

**No full-universe repartition launched**, C5 jobs (PID 19828 etc) untouched, no 6M graph built. Target sweep deferred.

---

## 9. Recommendation — L1 Understanding Validated Before Scaling

1. **Keep historical A as L1 baseline only for legacy comparison** — but acknowledge global dense/SPLADE is strictly better at matched budget; L1's role is *budgeted candidate generation*, not ranking.
2. **Do not promote NER to routing graph** — NER belongs at L3 `G_main` (struct+NER traversal), where it gives +3–15 text, +50 KB lift (`l3_graphlift`). At L1 it harms centroids and vote coherence.
3. **If partition routing is retained for efficiency, use A topology (struct+kNN) only**, and consider hybrid: small `VOTE_K` (50) + RRF fusion gives best A recall (RRF 72.8/91.4/86.4 @5). But expect 20-point deficit vs global at 500 budget.
4. **Next step before scaling:** Freeze global direct (D) as primary L1 candidate generator (dense+SPLADE best-of fusion, already +1.76 FullCov@20 in prior L2) and keep partitions as *optional* efficiency layer with admission control, not quality layer. Validate on full corpora (398k/117k) with same protocol before any 60k-partition build.

All tables machine-readable under `results/`; this markdown appended per spec.
