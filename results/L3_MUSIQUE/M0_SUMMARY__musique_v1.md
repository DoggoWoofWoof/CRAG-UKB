# MuSiQue text-L3, step M0 (2026-10-04): reachability + ranked graph signal on the CANONICAL substrate

Development population only (2,000 rows of `results/L1_DEV/loc_population.json`, 417 dev + 1,583 train; historically exposed, not held-out; no held-out row read). Descriptive p-values. Laptop, idle priority.
Records: `M0a_HITS__musique_v1/v2.json` (cached FLAT_RRF front end; FLAT gold positions == `results/L1_DEV/loc_musique__v1.npz` bit for bit), `M0b_REACH__musique_v1.json`, `M0c_RANKED__musique_v1.json`.
Code: `scratchpad/_l3m_hits.py`, `_l3m_hits_v2.py`, `_l3m_reach.py`, `_l3m_ranked.py`.

## M0b — an unranked h-hop reach set is NOT better than FLAT at the same size
Seeds = top-s FLAT hits, graph = fixed STRUCT (+NER) (+KNN), hub rule deg > 300 not expanded. ALL golds in the reach set vs ALL golds in FLAT's top-|reach set| (same per-question budget):

| graph, s | h=1 reach / FLAT@eq (median size) | h=2 | h=3 |
|---|---|---|---|
| G_SNK, 10 | .596 / .663 (362) | .781 / .892 (5.9k) | .932 / .990 (53.5k) |
| G_SNK, 50 | .774 / .792 (1.5k) | .922 / .957 (19.6k) | .986 / .999 (90.4k) |
| G_SNK, 200 | .868 / .882 (4.8k) | .971 / .988 (44.9k) | .998 / 1.000 (108.7k of 117.5k) |

Connectivity is real (98.6 % all-gold reach at 3 hops, G_SNK s=50; title-mention-only G_S only 82.5 %, the legacy "31 %" ceiling does not carry over to canonical + NER + KNN) but it is bought with 77–92 % of the corpus: FLAT at the same budget is ahead (flat-only > reach-only pairs at every h, s, graph). Reachability alone is therefore not the lever.

## M0c — a ranked graph signal (PPR fused with FLAT) beats FLAT, does NOT beat the L1 one-hop localisation
ALL-gold recall at served-list size B_N = 100 / 250 / 500 / 1000 / 2000 / 5000:

| arm | 100 | 250 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|---|
| FLAT | .538 | .631 | .695 | .757 | .814 | .884 |
| IR_L1 (one-hop localisation; L1) | .640 | .742 | .793 | .840 | .880 | .928 |
| PPR G_SNK s10 dn (best PPR @1000) | .606 | .723 | .790 | .845 | .888 | .926 |
| PPR G_SN s10 dn | .615 | .721 | .789 | .848 | .884 | .916 |
| PPR G_SNK s50 dn (best @5000) | .586 | .695 | .769 | .838 | .878 | .930 |

PPR (G_SNK s10 dn) vs FLAT: +7..+10 pts at B_N 100–2000, +4 at 5000, McNemar p < 1e-6 at every B_N. PPR vs IR_L1: IR_L1 significantly better at B_N 100/250 (p ≤ 6e-4), tied from 500 up (G_SNK s10 dn @1000: +35 / −25, p .25; @5000: +42 / −46, p .75). Seeding from more hits (s = 50, 200) is monotonically worse; degree damping (dn) helps a little. By hop @1000 (IR_L1 / PPR s10 dn): 2-hop .912/.919, 3-hop .760/.772, 4-hop .510/.485.

Verdict (development, one fixed alpha = 0.5 / 8 iterations chosen a priori, un-weighted edges): **PPR from FLAT seeds over STRUCT+NER+KNN adds nothing on MuSiQue beyond what IR_L1 already delivers inside L1**; it cannot be promoted as an L3 gain, and a bounded-frontier PPR implementation is not worth building for this population. The 3-/4-hop residual (@1000: 24 % / 49 % of questions miss a gold) is not graph-linked through the inference-safe edge families in a way a diffusion can exploit.
Not tested here (open, would each need a ruling or a new arm): NER edges weighted by 1/df, PPR seeded from the IR_L1 list, embedding-guided beam (BALANCED_H2), typed/relation schedules (inapplicable: one relation).
