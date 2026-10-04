# L1_COVPART — coverage-aligned static partitioning (DEV_A, plus §30's fresh split A; DEV_B untouched)

Opened 2026-09-14 on the ruling: *"open exactly one new lane: COVERAGE-ALIGNED STATIC PARTITIONING … H4_SK current baseline /
H4_SK_PATH2 new candidate where PATH2 encodes bounded query-independent two-step structural co-membership … no relation labels,
no typed walk, no query … metrics: ALL-gold @ P50, UNREACHED fraction, median rank of missed gold block … the goal is
specifically: drive the UNREACHED MetaQA mass down."*  Everything below is DEV_A (sha1(qid)[:8] & 1 == 0, n = 1002 metaqa /
1000 squad); DEV_B was not read (spent twice already; any held-out look needs a ruling); the one exception is §30, which also reports the pre-registered fresh split-A populations of WebQSP / HotpotQA / 2Wiki (split B sealed).  Nothing under `data/` was written;
canonical partitions, caches and L1 results are untouched.  All scratch partitions live under `results/L1_COVPART/parts/`.

## 0. Verdict

| question | answer |
|---|---|
| Is the hypothesis ("first-order H4_SK does not preserve higher-order STRUCT neighbourhoods inside P50") true as a structural statement? | **Yes.** Under the served partitions only 8.4 % (metaqa) of the non-hub 2-hop STRUCT balls that would fit in one block (≤ 100 nodes; 87 % of all balls) lie inside one block; the mean such ball is spread over 16.4 blocks. |
| Does the one principled higher-order construction (H4_SK_PATH2, §3) fix that? | **Structurally, partly:** balls intact 8.4 % → 17.2 %, blocks spanned 16.4 → 12.9, PATH2 km1 −27 %, STRUCT km1 −20 % — bought by losing KNN compactness (KNN stars in one block 5.1 % → 1.6 %, KNN km1 +24 %). |
| Does it drive UNREACHED down? | **Marginally:** 25.6 → 24.8 pts. BASE_ALL 0.6727 → 0.6856 (+49/−36, p = 0.19); hop2 +5.8 (p = 0.002), hop3 −2.4 (p = 0.15), hop1 ±0. **NOT_CONFIRMED on DEV_A**; not carried to DEV_B. |
| Why so little, when 100 % of the unreached gold nodes sit within non-hub distance ≤ 3 of a served hit? | A query-aware oracle that moves *every* unreached gold into a block that receives a vote (UNREACHED 25.6 → 8.8) raises BASE_ALL by **+1.0** only; landing each in the *best-ranked* voting block gives **+5.2** (0.7246); constrained to voters inside the gold's own 2-hop ball, **+4.3** (§5). Reach is necessary, not sufficient: ALL-gold @ P50 needs every gold block inside the top 50 of the fused order, and a block that merely gains a vote sits at rank ~100–140. |
| Is there a hard floor? | ≈ 8.5 pts of UNREACHED survive even oracle packing: the same answer entity is gold for several DEV_A queries whose anchors vote for different blocks (84–328 placement conflicts per run), and a node has one block. No single static partition can serve them all. |
| Text-corpus non-regression | squad_phg: BASE_ALL 0.9829 → 0.9788 (+6/−10, p = 0.45, ns), reach 0.999 = 0.999, D2d 0.9879 → 0.9869 (ns). |
| Follow-ups (same day): 1-hop memberships voting before P50 (§10) and mass-conserving / support-normalised votes (§11) | §10: the served chain *already* votes through static 1-hop STRUCT memberships before P50 (directed out-edges) and has no post-P50 node halo; making the table undirected (frozen hub rule) is a hop-1-only KB gain (+1.7 / +2.2, p = 0.03 / 0.003) that does not transfer to text (≤ 0) and turns multi-hop UNREACHED into reached-weak (median missed rank 143 → 260). §11: normalising each hit's vote over its memberships (A2 uniform, A3 support a(v,b)/Σ) is **significantly worse** than both A1 and the served A0 on every cache (metaqa MtK 0.6567 → 0.6128 / 0.6148; PHG 0.6946 → 0.6407 / 0.6327), reached-weak rises, median missed-reached rank stays ≥ 207. **1-hop structural-vote idea CLOSED** per the ruling's own rule. |
| Route 1 (§12): parameter-free, walk-free, query-conditioned discriminator over already-reached blocks (typed 1-hop memberships gated by the frozen lexical relation match, `S_q = S_base + compat`) | **NOT_CONFIRMED** by the pre-registered rule: metaqa MtK 0.6397 → 0.6577 (+26/−8, p = 0.003) but hop2+hop3 pooled +13/−8 (p = 0.38); PHG 0.6727 → 0.6856 (+19/−6, p = 0.015; pooled +7/−6); text corpora identical by construction. All gain is hop-1; in 80 % of the remaining reached multi-hop failures the gold block is exposed only by hits at rank ≥ 20 (the intermediate entity is not a hit) — a second-hop problem, conceded to dynamic relation reasoning (L3). |
| Edge / triplet geometry voting (§13): valid graph facts as a parameter-free retrieval unit (closed-form `S_mid`, `S_dir` from `q·e_s`, `q·e_o`, `e_s·e_o`; top-100 edges vote `{P(s), P(o)}`; frozen RRF; P50) | **NOT_CONFIRMED** by the pre-registered rule: metaqa MtK 0.6397 → 0.6687 (+36/−7, p = 9e-6) and PHG 0.6727 → 0.7056 (+38/−5, p = 2e-7) pass the material gate, but hop2+hop3 pooled fails on MtK (+13/−7, p = 0.26; PHG +16/−5, p = 0.027); text ns (squad −0.7, squad_phg −0.5, musique −1.2). All gain is hop-1 and is A1's population selected by geometry (22/23, 22/22 overlap; 21/22 via an edge incident to the rank-0 hit); reached multi-hop failures have their gold block at edge rank ≥ 20 or unseen in 95–100 %. Ruling: failed as a universal L1 channel; the geometry is re-used at the right scale inside µL3 (§14). |
| µL3 (§14): a tiny bounded dynamic repair stage — frozen 5 seeds → two hops over the actual admissible STRUCT adjacency (beam 100, §13 closed-form transition geometry + frozen RRF) → visited nodes vote through the served membership table → frozen symmetric RRF with the served order → P50 (`MICRO_L3_H2`, pre-registered) | **Gate passed, NOT PROMOTABLE → CLOSED (§14.3).** hop2 + hop3 pooled: metaqa MtK 0.5059 → 0.5651 (+42/−2, p = 1e-10, +5.9), PHG 0.5547 → 0.6050 (+36/−2, p = 5e-9, +5.0) — `CANDIDATE`, not STRONG (< +10); hop 3 alone ns (+6/−1, +3/−1); overall metaqa +4.9 / +4.3, reach 0.713 → 0.869. Text universality violated: squad −20.7, squad_phg −20.9, musique −15.9 (p ≤ 3e-35). Attribution: the beam instantiates the answer entity for 87 % of hop-2 questions at 22 ms / 373 transitions per query (discovery works); the frozen mass vote (top repair block ≈ 62 votes vs ≈ 2 for the gold block; 201/432 blocks voted) and the equal-authority fusion (gold block unranked in the repair channel in 204/207 squad losses, seeds do not vote) lose it. Ruling (#2 only, §14.3, POSTHOC_MECHANISM_TEST): the pre-existing asymmetric `HOLD` interface (served order authoritative, tail-only exchange) retains 0.83 / 0.88 of the hop-2 gain (+35/−6, p = 5e-6; +31/−3, p = 8e-7) but still loses significantly on squad (0/−8, p = 0.008) and musique (+11/−48, p = 1e-6) — musique's served tail (positions 25–49) carries 19.8 % of its correct answers vs 2.5–2.8 % elsewhere (`served_tail_gold_share_A.json`); the tightest fixed core (CORE6) is text-safe by significance but retains only 0.29 / 0.31 (information only, not selectable). **FAIL by the pre-registered rule → `MICRO_L3_H2` CLOSED as an L1-repair mechanism; #1 / #3 not run.** Latent global beam (§14.4): strictly worse (metaqa MtK +4/−79 vs L1; reach +41/0 vs +157/0 for real edges; 8 s vs 22 ms per query). |
| H2 as a SAFE candidate generator (§15): the pinned µL3 beam's visited blocks, ordered by best beam rank, appended to the frozen six-slot structure swap's candidate pool (`spos' = min(static, H2)`; no fourth channel, no cap, no agreement rule; evaluated against canonical SAFE on all five DEV_A caches; pre-registered) | **NOT_CONFIRMED as run — text-safe but inert (FAIL by the pre-registered rule, on the branch the ruling did not anticipate).** SAFE_H2 vs SAFE: squad +1/−2, squad_phg +1/−1, musique +7/−6 (all p = 1.0; SAFE's musique gain kept, +55/−2 vs L1); metaqa hop 2 +2/−1 (p = 1.0) / +4/0 (p = 0.125) = 0.029 / 0.125 of §14.1's hop-2 gain; ANY-gold +1 pt. The generator is not the bottleneck: it proposes the worst missing hop-2 gold block for 100 / 177 (91 / 162) failures vs 21 (10) for the static expansion, and the SAFE_H2 pool holds complete repairs for 98 / 86 hop-2 failures vs 19 / 11 canonical — the frozen admission realises 4 / 4. Why: the ruled order puts ≈ 16 hop-1 blocks ahead of each hop-2 answer block, and 53 / 96 answer blocks are supported by H2 alone — one channel short under the additive F6 score (mean gap 0.0087 ≈ a whole channel at rank 50). A channel-count-first agreement rule would rank them lower still (reading of the records; not run). Next question = a universal, parameter-free admission rule for single-witness beam evidence — needs a ruling; `MICRO_L3_H2` repair stays closed. |
| Query-local structure patch `H2_PATCH1` (§16): canonical SAFE minus its weakest admitted block, plus one query-local block of the same size holding the pinned µL3 beam's visited nodes that the 49 kept blocks do not expose (frozen first-visit order, capacity cut, remainder filled from the removed block; |served nodes| matched exactly); ALL-gold at the node level; pre-registered, the only order decision taken from a gold-free statistic | **PASS — MECHANISM_CONFIRMED_ON_DEV_A (POSTHOC; not promoted).** vs canonical SAFE: metaqa 0.6407 → 0.7515 (+111/0, p = 8e-34; hop 2 0.4942 → 0.7413, +85/0, p = 5e-26; hop 1 +25/0; hop 3 +1/0), metaqa_phg 0.6747 → 0.7735 (+100/−1; hop 2 0.5291 → 0.7442, +74/0, p = 1e-22), squad 0/0 (exactly unchanged), squad_phg +2/0, musique 0.7118 → 0.7329 (+21/0, p = 1e-6; +76/−3 vs L1); ANY-gold 0.915 → 0.983. Recovery = 2.4 / 2.3 × §14.1's hop-2 gain with none of its text loss. The ruling's diagnostic: of the 98 (86) hop-2 failures with a complete repair in §15's SAFE_H2 pool, canonical SAFE covers 3 (1) and the patch covers 85 (72) — every failure whose missing gold nodes the beam visited, on every cache and hop (0 lost to the capacity cut, 0 to the removed block). The unit of repair was the problem, not H2 and not ranking. Residual: hop-2 failures with an unvisited gold node (89 / 88 of 174 / 162; beam recall) and hop 3 (depth-2 beam visits 0.8 % of the missing hop-3 gold nodes) — not run. Transfer needs an untouched population (DEV_B read twice → WebQSP or another untouched set, by ruling). |
| Ninth ruling, part 1 — the frozen real-edge beam one hop deeper under the identical one-block patch, `H3_PATCH1` (§17): pinned µL3 beam with DEPTH 2 → 3 as the only change (hops 1–2 asserted identical to §14.1 / §16), novel = visited nodes of beams 1 + 2 + 3 in the frozen first-visit order, the §16 patch rule and budget unchanged; pre-registered (order decided from a gold-free statistic; hop-3 gain vs `H2_PATCH1` p < 0.01 on both metaqa caches, text safety, no hop-2 regression) | **FAIL by the pre-registered rule — NOT_CONFIRMED as a hop-3 repair; text-safe; `H2_PATCH1` exactly intact.** hop 3 vs `H2_PATCH1`: metaqa +1/0 (0.5241 → 0.5271, p = 1.0), metaqa_phg 0/0; hop 2 0/0 on both; squad 0/0, squad_phg 0/0, musique +5/−2 (p = 0.45; +26/−2 vs SAFE). The depth-3 beam raises the visited share of the missing hop-3 gold nodes 0.008 → 0.115 (7 → 268 nodes) and 108 / 101 hop-3 gold nodes enter the patch, but a failing hop-3 query is missing 16.3 / 17.3 gold nodes and only 4 / 1 of 159 / 137 have all of them visited (2 / 0 covered, 2 / 1 cut) — an unbounded patch would complete 4 / 1. Attribution (`h3patch_why`): the hop-2 residual is beam *width* (630 / 624 of the 1,169 / 1,139 missing hop-2 gold nodes are scored hop-2 candidates cut by the top-100; 173 / 160 of 174 / 162 failures lie entirely inside the depth-2 candidate sets, 215 candidates per query), the hop-3 residual is reach compounded from that cut (64–65 % of the missing hop-3 gold nodes never adjacent to the beam). Depth was the wrong axis; width is pinned and coupled to the patch budget → needs a ruling. Cost 72–80 ms / query at depth 3 (22 ms at depth 2). |
| Ninth ruling, part 2 — overlap as part of the partition definition, `STRUCT_VERTEXCUT_V1` (§18): STRUCT edges partitioned by the validated PHG on the dual incidence hypergraph (vertex per edge, net per node of degree ≥ 2, CONNECTIVITY = Σ_v (λ(h_v) − 1) = node replication), B_p = endpoint sets; k grid {432, 864, 1,296} + matched k* = 1,004 (mean \|B_p\| = 102) + deterministic capacity repair; query-free metrics (RF, 1-hop, exact 2-hop wedge, exact 3-path, §2 ball) vs the served hard partitions, the served voting unit, the undirected halo and PATH2; pre-registered before any build; no KNN, no queries, no gold, no replay | **NOT_MATERIAL by the pre-registered rule — the strict-capacity cell fails the residual clause (398 / 1,004 blocks above C = 100, max 104; ≤ 1 % allowed) and the secondary ball clause (1.95 × vs ≥ 2 ×); its containment clauses pass by a wide margin.** Non-hub 2-hop wedge containment: served hard 0.0226 (MtK) / 0.0159 (PHG), PATH2 0.0365 → vertex-cut 0.2788 at k* (RF 2.38), 0.2649 at the repaired cell (RF 2.34; 11.7 ×, +24.2 pts), 0.2589 at k = 1,296 (81 nodes / block, 3 blocks above 100, RF 2.44 — satisfies every clause but the ball one); 1-hop 0.18 → 1.00; non-hub 3-paths 0.003 → 0.046; §2 balls 0.094 → 0.183. The served voting unit reaches 0.56 only unbounded (blocks up to 16,030 nodes, RF 2.5–2.6); the halo 1.0 at RF 4.3. Post hoc, labelled: the residual clause was infeasible by construction at k* (Σ\|B_p\| = 101,181 > k*·C = 100,400; the 781 residual = the arithmetic minimum) and the served hard partitions violate it too (272 / 364 blocks above 100, max 104) — a flaw of this pre-registration, not of the representation. Closes the arm at the structural level by its own clause; whether the replay precondition is met is a ruling on the capacity clause; the replay itself needs the served-unit contract (P50 over overlapping blocks, exposure matching, hub replication ~110 blocks, degree-0 homes on text, KNN). squad / musique curves not run. |
| Tenth ruling — the vertex-cut blocks replayed through the frozen router, `STRUCT_VERTEXCUT_V1_REPLAY` (§19): the unchanged capacity-repaired k* = 1,004 cell (RF 2.34, \|B_p\| 100–104) as home + overlap (M(v) = blocks holding an edge of v; H(v) = argmax incident edges, ties by block id); served dense / SPLADE top-100 hits vote through the frozen numerics with non-hubs → M(v), the 104 hubs → H(v) only; frozen RRF; P50; node-level ALL with gold served through M(g); actual unique exposure reported; secondary matched-budget cell (largest fused prefix at or under the query's hard exposure); pre-registered before the run; metaqa DEV_A only; §18's verdict untouched | **MECHANISM_POSITIVE_AT_LOWER_EXPOSURE by the pre-registered rule (post hoc; not a confirmation; not promotable).** P50: 0.6397 → 0.6776 (+72 / −34, p = 2.8 × 10⁻⁴) at 4,166 unique nodes vs the hard P50's 5,023 (17 % fewer; every query below its own hard exposure); matched budget (61 blocks, 4,986 nodes): 0.6876 (+75 / −27, p = 2.1 × 10⁻⁶). The gain is hop 1 (0.917 → 0.994, +26 / −1; 1-hop containment 1.0 by construction) and hop 2 (0.486 → 0.555, +33 / −9, p = 2.7 × 10⁻⁴); hop 3 goes the other way (0.527 → 0.494, +13 / −24, p = 0.10; vs the PHG base +6 / −35) so hop 2+3 pooled is not significant at P50 (p = 0.18; p = 0.011 matched). Diagnostics: the served hard BASE voted through its own-block table gives 0.179 — its 0.64 is the frozen legacy own + out-neighbour vote expansion, so the contrast is generic directed 1-hop vote spread over hard blocks vs partitioner-decided overlap with no expansion, and the overlap wins at lower exposure; home-only voting on the vertex-cut falls to 0.602 (the membership vote carries the gain). vs the served PHG base: level overall (+62 / −57, p = 0.71) at 18 % lower exposure. Residual 323 failures: 199 UNREACHED / 124 out-ranked (worst gold block at fused position 565 / 1,004 on average) — the reach problem is unchanged in kind. Not combined with H2_PATCH1 (ruling); k = 1,296 and the text corpora not replayed. |
| Eleventh ruling — the vertex-cut BASE carried to squad and musique with one universal degree-0 rule, `STRUCT_VERTEXCUT_V1_TRANSFER` (§20): the §18 chain unchanged on each text graph (k-grid, k* fixed point, capacity repair at C = 100 on the STRUCT edges: squad k* = 2,499, musique k* = 12,518), then every degree-0 node placed once in the deterministic home H(u) of its nearest anchored frozen-KNN neighbour (highest cosine weight, ties by node id; no chaining) or, without one, in the currently smallest block in node order; §19's representation, routing, serving, exposure and matched-budget cells through the pinned functions; gate = each dataset's canonical served hard BASE (squad Mt-KaHyPar, musique PHG); labels LOSS / GAIN / NEUTRAL at α = 0.01 and an exposure flag, pre-set before any placement or number; DEV_A only | **squad `NEUTRAL_EXPOSURE_LE_HARD`, musique `LOSS_EXPOSURE_LE_HARD`; universal statement `NOT_SUPPORTED` by the pre-registered rule (post hoc; nothing promoted; no composition run).** squad P50 0.9859 → 0.9768 (+4 / −13, p = 0.049) at 2,208 unique nodes (43 % of the hard 5,089); matched budget 0.9909 (+9 / −4, p = 0.27; 144 blocks). musique P50 0.6596 → 0.6014 (+65 / −123, p = 2.8 × 10⁻⁵) at 3,204 nodes (63 % of 5,117); matched budget level, 0.6687 (+93 / −84, p = 0.55; 87 blocks). Mechanism: the text STRUCT graphs are dense (mean STRUCT degree 107 / 49 vs 5.8 on metaqa), so ≤ 100-node vertex-cut blocks need RF 10–13 (vs 2.3) and 50 of them hold only 43–63 % of the hard budget's unique nodes — the primary cell is a budget cut on text, level once matched; and there is no 1-hop vote expansion to replace: the served hard partitions voted through their own block score 0.994 (squad, above the legacy BASE) and 0.645 (musique, −1.5, ns) against 0.18 vs 0.64 on metaqa; home-only voting on the vertex-cut is level with membership voting on both text sets. The degree-0 rule ran as specified (83 % / 97 % anchored; 9 of 23 / 46 of 397 failures involve an isolated gold) and is not the failure point. The exposure-efficient vertex-cut BASE is a sparse-relational-graph (MetaQA) result; text-safe only at matched exposure. Next per the ruled order: WebQSP watcher unchanged, then frozen H2_PATCH1 → WebQSP. |
| Twelfth ruling, arm B — the frozen beam's hop-2 candidates compressed by a parameter-free per-parent rank interleave, `BALANCED_H2_PATCH1` (§21): §16 / §17 reused verbatim (frozen seeds, §13 geometry, frozen RRF, depth 2, BEAM 100, canonical SAFE, the one-block same-size patch, exact exposure match); the only change is the beam's compression at each hop — dedupe (parent, target) in the fused order, rank within parent, lexsort by (within-parent rank, fused position), keep the first 100 distinct unvisited targets (round-robin A1 B1 C1 … A2 B2 C2 …); no local K, no weight, no dataset constant; pre-registered (v1 before any run; v2 supersedes it after a diagnostic-only fix, with the seen metaqa numbers stated) | **PASS — `MECHANISM_CONFIRMED_ON_DEV_A (POSTHOC; not promoted)`.** vs `H2_PATCH1`: metaqa 0.7515 → 0.7764 (+30 / −5, p = 2.2e-5; hop 2 0.7413 → 0.8140), PHG 0.7735 → 0.7984 (+31 / −6, p = 4.1e-5; hop 2 0.7442 → 0.8169); hop 1 and hop 3 identical (0 / 0); squad 0 / 0, squad_phg 0 / −2 (p = 0.5), musique +2 / −8 (p = 0.11; +15 / 0 vs SAFE). Every flip is a beam flip (58 / 61 gold nodes newly visited; 8 / 9 no longer visited; none cut by the patch); on the §17 population the fully-visited failures rise 85 → 110 / 74 → 99 and are all covered; distinct parents in the kept hop-2 beam 19.1 → 24.0, largest parent share 0.234 → 0.105, wall-clock unchanged. Residual hop-2 misses (64 / 63) are still width (516 / 525 gold nodes scored but not kept). On musique the interleave trades depth inside strong parents for breadth (visited missing gold 18 → 11): a sparse-graph gain, neutral-to-slightly-negative on dense graphs. Not promoted; `H2_PATCH1` untouched for WebQSP. |
| Twelfth ruling, arm A — bounded-overlap static partitioning, every node in at most two blocks, `BOUNDED_VERTEXCUT_R2` (§22): owners = validated PHG hard partition of the primal STRUCT graph (degree ≥ 1) into k = 2 k_f half-size blocks (864 / 404 / 2,350), degree-0 by §20's rule, then one static greedy alternate per node (gain = STRUCT neighbours whose home is the foreign block; best gain first, ties by id; first block with live size < C = 100; refused otherwise): λ ≤ 2, |B_p| ≤ C, RF ≤ 2 by construction, the same rule on every graph; §19 / §20 routing, serving, exposure and matched cell unchanged; matched cell = the pre-registered signal; R = 1 owners-only control; V1 recomputed and asserted on metaqa; pre-registered before any build | **`NOT_SUPPORTED` by the pre-registered rule — text SAFE and EFFICIENT, MetaQA not RETAINED.** RF 1.53 / 1.61 / 1.53 (V1 2.34 / 10.35 / 9.25), 1-hop containment 0.547 / 0.527 / 0.311 (owners-only 0.329 / 0.180 / 0.118; served hard 0.181 on metaqa), metaqa wedge 0.219 (V1 0.265). squad: P50 0.9899 vs 0.9859 (+12 / −8, ns) at 69 % of the hard unique exposure, matched 0.9929 (+13 / −6); musique: P50 0.6647 vs 0.6596 (+116 / −111, ns) at 75 %, matched 0.6978 (+131 / −93, p = 0.013, above α). metaqa: P50 **0.3333** vs 0.6397 (+65 / −372, p = 2e-53) at 81 %, matched 0.3473 (+69 / −362); owners-only 0.2265 / 0.2525; vs V1 −34 pts. Post-hoc reach diagnostic (`br2_why_A_metaqa.json`): the served hard BASE reaches every hub gold (38 % of MetaQA's gold nodes) through the legacy *vote* spread (own + out-neighbour blocks; 2.5–3.4 votes per hit), V1 through the *membership* spread (the mean gold node sits in 43.7 of its blocks; served through its home only, V1's reach falls to 0.480, below R2's 0.535); R = 2 has neither (hub gold reach 0.627, 1.7 votes per hit) → 46.5 % of the queries are unreached before ranking. §19's gain was replication concentrated on the answer population, which a λ ≤ 2 rule cannot reproduce; the bounded substrate under the legacy spread would reach 0.807 (reach only; not evaluated, not pre-registered). V1 stays frozen as ruled; no composition. |
| Thirteenth ruling — the frozen §22 bounded substrate served through the canonical router, `R2_LEGACY_ROUTER` (§23): memberships read verbatim from the pinned R2 files (k = 864 / 404 / 2,350, C = 100, RF 1.53 / 1.61 / 1.53); the ONLY change is the vote table — each top-100 hit votes for its own vote set (hub → home, non-hub → both blocks) and for the vote sets of its directed STRUCT out-neighbours, the served BASE's legacy rule applied to the multi-membership table (asserted to reproduce `legacy_mem` entry for entry on every gate partition); same P50, same RRF, same serving, exposure and matched-budget contracts; R = 1 control (owners under the same router), §22's cell recomputed and asserted, reach diagnostics asserted against the seen numbers; pre-registered before any ranking or coverage number (rule: both text sets P50 and matched ≠ LOSS, and metaqa P50 GAIN or EFFICIENT); roles frozen by the same ruling: `H2_PATCH1` = transfer reference (WebQSP primary), `BALANCED_H2_PATCH1` = current best experimental successor (not transfer-confirmed), `H3_PATCH1` closed, `BOUNDED_VERTEXCUT_R2` rejected, no `BALANCED` + R2-router composition | **`FAIL` by the pre-registered rule.** metaqa P50 **0.5359** vs 0.6397 (+80 / −184, p = 1.3e-10) at 87 % of the hard unique exposure, matched 0.5469 (+85 / −178); the router restores 20.3 of §22's 30.6-pt deficit (0.333 → 0.536, +215 / −12; hub-gold reach 0.627 → 0.980) but hop 3 stays 0.211 vs 0.527 (+6 / −111) with hop-3 reach level (0.605 vs 0.611) — REACHED_WEAK, not reach: the R = 1 control under the identical router is 0.514 at the matched budget (−12.6 pts at equal exposure; hop 3 0.190), so most of the residual sits in the half-size k = 2 k_f owner partition, not in the routing; alternates +6 pts. squad SAFE (0.9778 vs 0.9859, +5 / −13 ns, at 71 %; matched 0.9879); musique NOT safe (0.5853 vs 0.6596, +61 / −135, p = 1e-7; matched 0.6135, +67 / −113) — the legacy spread over-spreads a λ ≤ 2 table on dense graphs (8.8 votes per hit vs 5.8 hard / 1.5 membership-only; evidence blocks 469 of 2,350 vs 227 of 1,175) and is worse than §22's membership-only vote on both text sets (squad +1 / −13, musique +62 / −141). By the ruling's letter: no R3 / R4, no alternate routing; the bounded-substrate line closes with §22 + §23; V1 frozen; roles frozen; WebQSP primary = frozen `H2_PATCH1`. |
| Fourteenth ruling, item 1 — the one composition `QMAX_F0` ranking → SAFE → `BALANCED_H2_PATCH1` = `QMAX_BALANCED_H2_PATCH1` (§24): the L1_GEOM `QMAX_F0` block order (served dense / SPLADE channels + exhaustive dense block max, frozen RRF) substituted for `base_rank` in the frozen SAFE view — the only array through which the F6 swap reads the ranking (challengers and the µL3 beam are seed-based; asserted identical under both rankings) — then the frozen F6 admission (`QMAX_SAFE`) and the §16 one-block patch filled by the §21 balanced beam, exposure matched to `QMAX_SAFE`; reference chain and `QMAX_F0` recomputed and asserted equal to the §16 / §21 / L1_GEOM records; no `QMAX_H2_PATCH1`, no D2d / interleave / re-weighting; pre-registered after an identity-only dry run (rule: no LOSS vs either parent on any of the five caches AND musique GAIN vs `BALANCED` AND both metaqa caches GAIN vs `QMAX_F0`; α 0.01) | **`PASS_STACKS` — MECHANISM_CONFIRMED_ON_DEV_A (composition; not promoted; NOT transfer-confirmed): the universal L1+ candidate of the ruling.** metaqa **0.7754** (`BALANCED` 0.7764, +2 / −3 NEUTRAL; `QMAX_F0` 0.6327, +143 / 0), metaqa_phg **0.7964** (0.7984, +2 / −4 NEUTRAL; +122 / 0), squad **0.9950** (+3 / 0 vs both parents), squad_phg **0.9899** (+1 / 0), musique **0.7600** (`BALANCED` 0.7269, +34 / −1, p = 2e-9; `QMAX_F0` 0.7319, +36 / −8, p = 3e-5) → cells CARRIES / CARRIES / NEUTRAL / NEUTRAL / STACKS; vs the served L1: +138 / −2, +127 / −3, +9 / 0, +7 / 0, +101 / −1. Stages: on MetaQA the patch is the whole gain (hop 2 0.480 → 0.808) and the aggregation is neutral; on MuSiQue all three stages add (aggregation +7.2, SAFE swap over the aggregation +1.6, patch +1.2; 35 of the 37 gold nodes newly served vs `BALANCED` sit in a kept block of `QMAX_SAFE`). ADDITIVE_ON_MUSIQUE true, ADDITIVE_ON_METAQA false, SQUAD_NON_REGRESSION true. DEV_A only (the ruling's own caution: over-read); `H2_PATCH1` stays the WebQSP transfer reference; the composite is the candidate for the 2Wiki / Hotpot text transfer when substrates and a resource ruling exist; its compression stays the §21 rule until item 2 (§25) has a verdict. |
| Fourteenth ruling, item 2 — the one coverage-then-score frontier `COVSCORE_H2_PATCH1` (§25): the §21 chain with one change of compression — phase 1 = the best child of every active hop parent in frontier order (A1 B1 C1 …), phase 2 = every remaining (parent, target) pair in the frozen global fused order (A2 A3 B2 …), the first 100 distinct unvisited targets of phase 1 then phase 2; no k_local, no λ, no weight, no dataset branch; beam, patch, budget, F6 and SAFE verbatim §16 / §21 (reference arms asserted equal to their records); pre-registered after a unit test and a gold-free dry run (hop 2 Jaccard 0.935 vs the pinned beam, 0.622 vs the balanced one) with the rule: PASS_SUCCESSOR iff hop-2 GAIN vs `BALANCED_H2_PATCH1` on both metaqa caches (α 0.01) AND no hop-1 / hop-3 regression AND text safety vs `BALANCED` / `H2_PATCH1` / SAFE (0.05); FAIL_WORSE on any metaqa LOSS / regression / text violation | **`FAIL_WORSE` — the coverage-then-score rule is worse than the §21 round-robin on DEV_A and CLOSED; `BALANCED_H2_PATCH1` keeps the frontier role; item 3 runs on the balanced frontier.** metaqa **0.7535** (`BALANCED` 0.7764: +5 / −28, p = 6.6e-5 LOSS; `H2_PATCH1` 0.7515: +4 / −2 ns; SAFE +113 / 0), metaqa_phg **0.7764** (0.7984: +5 / −27, p = 1.1e-4 LOSS; 0.7735: +4 / −1 ns), squad 0.9919 (0 / 0), squad_phg 0.9909 (+2 / 0), musique 0.7309 (`BALANCED` 0.7269 +6 / −2 ns; `H2_PATCH1` 0.7329 0 / −2 ns; SAFE +19 / 0); hop 1 and hop 3 identical to `BALANCED`; the whole loss is hop 2 (0.7471 vs 0.8140; PHG 0.7529 vs 0.8169). Why: phase 1 gives one child to 23.0 parents (balanced 24.0, pinned 19.1) but the global fill re-concentrates the other 75 slots on the parents the pinned beam favours (largest-parent share 0.226 vs pinned 0.234, balanced 0.105) — missing gold nodes visited at hop 2 on SAFE's hop-2 failures 534 (pinned 536, balanced 650); all 53 gold nodes lost vs `BALANCED` are no longer visited (none cut by the patch). §21's gain lives in the later round-robin rounds (the second, third … children of the right parents), so the sparse end of the trade-off is the better hop-2 compression and the remaining hop-2 residual is WIDTH (§17), not order. Text: within noise of both parents; musique's residual is reach (331–334 of 366 missing gold nodes never scored within depth 2). `H2_PATCH1` stays the WebQSP transfer reference; the §24 composite keeps §21's compression. |
| Fourteenth ruling, item 3 — the one `H3_PATCH1` successor on the balanced frontier `BALANCED_H3_PATCH1` (§26): §17's depth-3 arm (the pinned µL3 beam exec'd at DEPTH = 3 with §17's substitutions, the §16 one-block patch in first-visit order hop 1 / 2 / 3, topped up from the removed block, matched exposure) with §21's round-robin compression at every hop instead of the pinned global order; no wider patch, no H3-specific scoring, no new constant; reference arms `H2_PATCH1` / `H3_PATCH1` / `BALANCED_H2_PATCH1` recomputed and asserted equal to their records; pre-registered after a gold-free dry run (hop 3: 77.2 distinct parents vs 47.3 pinned, largest share 0.024 vs 0.095, Jaccard 0.29, 2.4 rounds; 23.5 hop-3 nodes admitted per query, 45.7 cut) with the stated prediction FAIL_NOT_BETTER and the rule: PASS_H3 iff hop-3 GAIN vs `BALANCED_H2_PATCH1` on both metaqa caches (α 0.01) AND no hop-1 / hop-2 regression AND text safety vs `BALANCED_H2` / `H2_PATCH1` / SAFE (0.05) | **`FAIL_NOT_BETTER` — depth stays CLOSED also under the balanced frontier; `BALANCED_H2_PATCH1` keeps the successor role; the hop-3 residual is reach + width + capacity, not order.** metaqa **0.7794** (`BALANCED_H2` 0.7764: +3 / 0, p = 0.25 NEUTRAL; `H3_PATCH1` 0.7525: +32 / −5 GAIN = §21's hop-2 gain carried to depth 3; SAFE +139 / 0), hop 3 0.5331 vs 0.5241 (+3 / 0); metaqa_phg **0.8024** (0.7984: +4 / 0, p = 0.125; `H3_PATCH1` 0.7735: +35 / −6), hop 3 0.5964 vs 0.5843 (+4 / 0); hops 1–2 identical to `BALANCED_H2` (0 / 0); squad 0.9909 (0 / −1), squad_phg 0.9879 (0 / −1), musique 0.7269 (+3 / −3; vs `H3_PATCH1` +3 / −12 p = 0.035 = §21's musique −6 plus fill displacement, descriptive) — every text flip is a fill node from the removed block displaced by hop-3 novel nodes, all ns. Why: on SAFE's 159 metaqa hop-3 failures (2,597 missing gold nodes, 16.3 per failure) the balanced frontier is a real depth-3 frontier effect (missing gold visited 261 → 310, scored-but-cut 655 → 800, never scored 1,673 → 1,479; gold-bearing hop-2 parents with a gold child kept 274 → 358; failures fully scored 33 → 53; PHG alike) but 57 % of the missing hop-3 gold is still never a candidate of any node in a 100-wide depth-3 beam (reach), 31 % is scored and cut by a hop-3 round-robin that gives ~85 hop-2 parents one or two children each (width), and the one-block patch admits ~24 hop-3 nodes against 16 missing per failure that must all be served (capacity); only 5 of 159 failures have every missing node visited (4 covered). A MetaQA hop-3 repair needs a wider hop-2 frontier or different admissibility *and* a wider patch — rulings, not order rules; neither run. The fourteenth ruling's three frontiers are all answered on DEV_A (§24 PASS_STACKS, §25 FAIL_WORSE, §26 FAIL_NOT_BETTER); item 4 (transfer) and the MuSiQue scoring line remain. |
| Fifteenth ruling — the architecture frozen around the §24 composite (§27): `FREEZE_QMAX_BALANCED_H2_PATCH1.json` (roles: `H2_PATCH1` = WebQSP primary one shot, the composite = the arm for 2Wiki / HotpotQA, BASE / SAFE = every transfer's reference; DEV_A exhausted for selection; ten mechanism classes closed permanently for now); the 2Wiki / HotpotQA resource survey before any infrastructure; the streamed stage 2 (`_l1c_transfer_blockmax.py`) with its pre-registered identity gate; the transfer pre-registration (`PREREGISTRATION_TRANSFER_2WIKI_HOTPOTQA.json`: frozen 2,000-row populations with pinned `row_query_ids_sha256`, split A one shot, PRIMARY vs canonical SAFE and L1, TRANSFER_PASS / NEUTRAL / FAIL, predictions PASS on both, resource gate and labels) | **Frozen; transfer pre-registered; resource-blocked on this host.** PHG at NP = 4 is predicted INFEASIBLE for both text corpora inside the 7,789 MB WSL VM (hotpotqa summed 6.4–6.9 GB → needs ≥ 9,596 MB available under the pinned 1.4× margin; 2wiki 9.5–10.2 GB → exceeds the VM; the pinned semantic gate 8–19 GB of host Python; the pinned fp32 stage 2 32 / 37 GB) — NP, a cheaper partitioner, corpus subsetting and the foreign process are not levers; the WSL cap, another machine or a contract ruling are the user's. The streamed stage 2 is ACCEPTED by the pre-registered gate on all five DEV_A caches: squad / squad_phg bitwise identical (one chunk), metaqa / metaqa_phg / musique differ in 1.4 / 1.4 / 0.7 % of block maxima by ≤ 2.7e-7 (9 ulp; fp32 sgemm summation order, shown unreproducible on this OpenBLAS), 0 rows with a different fused P50 set, served sets identical on every split-A row, **0 coverage flips** for `QMAX_F0` / `QMAX_SAFE` / `PRIMARY` — the §24 records reproduced exactly (0.7754 / 0.7964 / 0.995 / 0.9899 / 0.760; CARRIES / CARRIES / NEUTRAL / NEUTRAL / STACKS). WebQSP: watcher CONTENDED on every sample and the re-arm **lapsed** 2026-09-16T11:38Z → `WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED` a second time (1,867 samples, 0 clean; `PHG_WEBQSP_RUNS.json` 830f27ae…; a third arm needs the user's ruling). Nothing launched, nothing under `data/`. |
| The partition-prototype ladder, re-centred on the IVF question (§28): can an IVF-style coarse quantizer — q scored against a few fixed, query-free representatives per served block (CENT_MEAN, CENT_NORM, FPS_m = medoid + farthest-point prototypes, RAND_m control; m = 1 … 32), the top 50 blocks probed, every node inside served — replace node voting (dense + SPLADE top-100 hits voting for their own and their STRUCT out-neighbours' blocks)? Pre-registered with a QMAX-referenced rule (533adfbd…) and re-centred by an addendum (063e423e…) before squad_phg or musique had finished or the squad record had been opened | **`IVF_DOES_NOT_REPLACE_NODE_VOTING`, on text and universally.** No FPS_m with m ≤ 32 preserves node voting on any cache. At m = 32, with ≈ 32 % of N scored as prototypes: squad 0.9728 vs 0.9859 (+8/−21), squad_phg 0.9587 vs 0.9829 (+9/−33), musique 0.5231 vs 0.6596 (+69/−205). The KB collapses at every m: metaqa 0.10–0.18 vs 0.6397, with hop 2 / 3 at most 0.14 / 0.04 — the out-neighbour votes carry them. Only the ladder's limit (every node a representative = block-max dense) preserves on text (squad +10/−8, squad_phg +12/−4) and **beats node voting on musique** (0.7008 vs 0.6596, +105/−64, p = 0.002); on the KB even the limit fails (0.18 / 0.20). The centroid beats the medoid and small-m FPS; FPS ≈ RAND. Secondary (the pre-registered QMAX question): FPS_PRESERVES_QMAX_AT_m=2 at PRIMARY (non-monotone: FPS_8 fails on musique), m* = 32 at SAFE, none at F0 or ALONE. Nothing adopted; FREEZE f81ebb84 unchanged. |
| Do the partitions help at all over flat dense + SPLADE rankings (§29)? Partition-free exhaustive FLAT_DENSE / FLAT_SPLADE / FLAT_RRF (frozen node RRF to full depth) served at each route's per-query exposure, vs node-voting L1@P (P = 1 … 50) and the frozen composite PRIMARY; five DEV_A caches (three datasets); pre-registered (8ef5134b…) after four gold-free dry runs. WebQSP / HotpotQA / 2Wiki have no canonical partitions (resource / ruling), so they are not covered | **`REGIME_DEPENDENT`.** **Text: flat beats partitions** at every budget. musique L1@50 0.6596 vs flat 0.8815 (+254/−33), PRIMARY 0.7600 vs 0.8815 (+159/−38); squad L1@50 0.9859 vs 0.9990. Flat reaches musique's L1@50 recall with 500 nodes instead of ~5,100; the gap grows with the number of golds (4-gold 0.304 vs 0.696). **KB: partitions help** at every budget. metaqa L1@50 0.6397 vs 0.3922 (+50/−298), PRIMARY 0.7754 vs 0.3922; one block holds the 1-hop answer set for 36.5 % vs 4.3 % for the flat top-100. But the routes carry STRUCT (blocks, out-neighbour votes, µL3) and the flat arms are text-only, so this is partition + graph vs text-only (a flat + graph-expansion arm was not measured). F1 / F2 / F4 / F6 held; F3 / F5 (musique) observed flat. Nothing adopted; FREEZE f81ebb84 unchanged; a dataset-agnostic flat + graph L1 is a ruling. |
| Are partitions necessary, or is the KB gain graph expansion (§30)? The user's narrow reopening of the L1 freeze for one isolated ablation, `FLAT_BALANCED_H2`: the top (M − 100) FLAT_RRF nodes + the first 100 novel nodes of §21's frozen BALANCED_H2 beam (served seeds, depth 2 over STRUCT) + flat fill, exactly M nodes; no partition, block, vote, SAFE or new scorer. Pre-registered (v2 f5f6741e…; v1 b35f954a… in `_history/`, superseded after a disclosed non-dry fall-through, §30.3) before any real-gold number. D1 = FLAT@5000 vs FLAT_BALANCED_H2@5000, one shot on fresh split A (WebQSP train_holdout, HotpotQA, 2Wiki; whole corpora; split B sealed); D2 / D3 on DEV_A, non-selecting: vs the frozen composite PRIMARY and vs FLAT, at PRIMARY's exposure | **`FRESH_VERDICT = FLAT_BALANCED_H2_UNIVERSAL`; `DEV_A_KB_READING = PARTITIONS_BEAT_FLAT_H2_ON_THE_DEV_A_KB`.** Fresh D1: webqsp 0.4962 → 0.7595 (+207/−0), hotpotqa 0.9331 → 0.9942 (+63/−0), 2wiki 0.6763 → 0.9213 (+246/−0); FLAT_BALANCED_H2@500 beats FLAT@5000 on all three. DEV_A: musique PRIMARY 0.7600 vs flat + H2 0.9036 (+159/−16), squad_phg +9/−0, squad ns; metaqa PRIMARY 0.7754 / 0.7964 vs flat + H2 0.7016 (+19/−93, +15/−110). The patch recovers ~80 % of the composite's lead over FLAT (0.3922 → 0.7016, +310/−0); the residual is 3-hop (net 57 of 74) and ≥ 5-gold answer sets (net 55 of 74). Every missed prediction underestimated the patch. Nothing adopted; FREEZE f81ebb84 unchanged; adoption, a split-B confirmation or any KB-residual mechanism is the user's ruling. |

## 1. Metrics (as ruled) and what "reach" means here

* **BASE_ALL** = ALL-gold @ P50 under the frozen numerics (`_l1s_core`: dense top-100 ∪ SPLADE top-100 → legacy block table → per-channel block rank → RRF K0 = 60 → P50).
* **reach / UNREACHED**: a served hit `h` votes for `block(h)` and for `block(v)` of every *directed* STRUCT out-neighbour `v` (the frozen legacy table); a gold node is REACHED iff some hit votes for its block. UNREACHED pts = feasible (≤ 50 gold blocks) failed queries with ≥ 1 unvoted gold block, in points of DEV_A. The fail decomposition also reports *reached-weak* (all gold blocks voted, worst one outside P50, not fixable by any fusion of the two block channels) and *fusion-fixable* (the union of the two channels' top-50 would have covered it).
* **median rank of the missed gold block**: rank of the worst gold block in the fused order among failed queries (432 = last = unvoted).

## 2. Reach diagnosis — metaqa, served Mt-KaHyPar cache, DEV_A (`reach_diag_A_metaqa.json`, `_l1c_reach_diag.py`)

BASE_ALL 0.6397, reach_all 0.7126, feasible 0.982; 361 failures of which 270 feasible-UNREACHED (26.9 pts). 2,018 unreached gold nodes:

| | value |
|---|---|
| hubs among unreached golds (closed STRUCT star > 100 = cannot be kept with its neighbourhood by any balanced partition; 104 hubs, deg 100–4176, 14.7 % of pins) | **0.0 %** |
| non-hub undirected STRUCT distance from the nearest served hit | d1 5.8 %, **d2 53.6 %, d3 40.6 %**, > 3: 0 |
| by hop: hop2 unreached golds | d1 9.7 %, d2 90.3 % |
| by hop: hop3 unreached golds | d2 38.0 %, d3 62.0 % |
| reachable by *directed* out-edges from a hit (the legacy table's vote direction) | **0.2 %** (99.8 % are > 4 out-hops away: KB edges point subject → object, hits are the objects) |
| reached golds for comparison: distance | d0 5.8 %, d1 37.5 %, d2 37.0 %, d3 19.7 %; 29.6 % hubs; block pos median 13 |
| size of the nearest hit's closed 2-hop non-hub ball | ≤ 50: 24.7 %, 51–100: 32.9 %, 101–200: 31.2 %, 201–500: 11.2 % |
| per-query convertible mass if co-location were perfect within radius r | r = 1: 2.2 pts, **r = 2: 18.1**, r = 3: 26.9 (all) |
| of the r = 2 mass, queries whose nearest hits' 2-hop balls all fit one block | **13.9 pts** (capacity ceiling for a ball-preserving partition) |
| worst gold block position, median, reached-but-failed queries | 143 |

So the residual is a **co-location** problem at distance 2–3 on the undirected non-hub STRUCT graph — the out-neighbour vote mechanism is dead on MetaQA (edge direction), and the partition is the only static lever that can put those golds into a voted block.

## 3. The construction — H4_SK_PATH2 (`_l1c_path2_build.py`, `parts/<ds>__H4_SK_PATH2.{npz,json}`)

`H4_SK_PATH2 = frozen H4_SK arrays (bit-identical, STRUCT + KNN stars) + one more H4_SPLIT_PRESERVE star family over the PATH2 adjacency`, built by the frozen builder (`src/l1_canonical/hypergraph.build_hypergraph`, frozen cap = 100, retention order, split rule, weights `max(1, rint(1000/(|e|−1)))`), same k = N // 100, same partitioner contract downstream.

PATH2 adjacency (query-independent, label-free, no parameter beyond the frozen cap): `hub(v) := deg_STRUCT(v) + 1 > cap`; `(u, w) ∈ A2 iff u ≠ w, both non-hub, and u–w is a STRUCT edge or u–v–w is a STRUCT path through a non-hub v`. The star of u is its closed 2-hop neighbourhood on the non-hub STRUCT subgraph — exactly the set the diagnosis says the served hits sit next to.

| | metaqa | squad |
|---|---|---|
| STRUCT keys → PATH2 keys | 124,669 → 841,772 | 744,734 → 207,301 (4,900 hubs; half the nodes have an empty non-hub 2-hop ball) |
| PATH2 family | 49,540 hyperedges, 1.73 M pins, mean size 35.0, weight 6.09 M (frozen families 32.3 M) | 6,393 hyperedges, 0.42 M pins, weight 0.16 M (2.4 % of frozen) |
| total | 136,336 hyperedges / 2.26 M pins (frozen 86,796 / 0.53 M) | 45,539 / 2.01 M |
| file sha256 | 95d65483… (content digest 4c825363…) | c95ef34e… (9f047115…) |

## 4. Result — same partitioner as the served metaqa_phg baseline (Zoltan-PHG substitute, `_l1c_phg.py`, `eval_A_metaqa_phg.json`)

The frozen Mt-KaHyPar contract could not run in the window (host available 0.5–1.5 GB against a foreign 8 GB job; the canonical driver's own rule cap = avail − 1 GB refuses) — it is queued behind a memory gate (§7). The validated PHG substitute (`src/l1_lowmem/phg.py`, identical driver, parameter list, NP = 4, IMBALANCE_TOL 1.03, CONNECTIVITY, DETERMINISTIC; `phg_repair.repair` for empty blocks) was used through a lane wrapper whose pipeline was first proven on the frozen H4_SK: shard bytes == served `H4_SK_STREAM_V1` shards (metaqa and squad) and output == served `LOWMEM__PHG_REPAIR1_con` (metaqa) / `LOWMEM__PHG_con` (squad), km1 equal. PATH2 run: km1 70,202,928, 3 empty blocks repaired at Δkm1 0, max block 103 ≤ 104, 35–80 MB/rank.

metaqa_phg, DEV_A n = 1002, paired McNemar vs FROZEN:

| | FROZEN (H4_SK, PHG_REPAIR1_con) | H4_SK_PATH2 (PHG_con) | paired |
|---|---|---|---|
| **BASE_ALL** | 0.6727 | **0.6856** | +49 / −36, p = 0.19 |
| reach_all | 0.7275 | 0.7415 | +49 / −35, p = 0.16 |
| fail pts: infeasible / **UNREACHED** / reached-weak / fusion-fixable | 1.6 / **25.6** / 4.5 / 1.0 | 1.1 / **24.8** / 4.0 / 1.6 | |
| median worst-gold-block rank, failed (reached-but-failed) | 432 (139) | 432 (108) | |
| hop1 BASE / reach / UNREACHED | 0.9172 / 0.9448 / 5.5 | 0.9202 / 0.9479 / 5.2 | +11 / −10 |
| hop2 | 0.5291 / 0.6279 / 36.0 | **0.5872** / 0.6715 / 32.0 | **+30 / −10, p = 0.002** |
| hop3 | 0.5813 / 0.6175 / 34.6 | 0.5572 / 0.6114 / 36.4 | +8 / −16, p = 0.15 |
| D2d channel on top (the L1_GEOM candidate) | 0.6737 | 0.6826 | +44 / −35, p = 0.37 |
| gold blocks per query / feasible | 5.09 / 0.984 | 4.68 / 0.989 | |
| non-hub 2-hop balls (≤ cap) inside one block / blocks spanned | 8.4 % / 16.4 | **17.2 % / 12.9** | |
| km1 STRUCT / KNN / PATH2 (M) | 18.9 / 26.7 / 30.1 | 15.1 / **33.1** / 22.0 | |
| stars in one block STRUCT / KNN / PATH2 | 34.3 % / 5.1 % / 6.6 % | 38.7 % / **1.6 %** / 13.6 % | |
| directed STRUCT edges cut | 81.7 % | 75.1 % | |

Reading: the partitioner does what the objective asks — 2-hop structural compactness doubles — and the hop2 population (golds at distance 2) gains significantly; the hop3 population (62 % of its unreached golds at distance 3, i.e. outside any 2-hop ball) loses, and the KNN (semantic-neighbour) compactness that the served partition had is traded away. Net on the ruled metric: UNREACHED −0.8 pts, BASE +1.3 ns.

## 5. Ceiling probe — query-aware, gold-aware packing (diagnostic; never a candidate) (`_l1c_ceiling.py`, `ceiling_A_metaqa_phg__*.json`)

Start from the served PHG partition; for each DEV_A query in scope and each of its golds outside the target set, move the gold into a chosen block (balance bound 104 = ceil(1.03·N/k) respected; if the block is full, swap with a non-gold node of that block); greedy in query order, moves never undone; a gold already placed for an earlier query is left where it is (*conflict*). The result is one concrete balanced partition, re-scored with the frozen numerics — a lower bound on the query-aware optimum, and an upper envelope for any query-*independent* partition (which cannot know which node is gold for which query).

| placement (scope) | reach_all | UNREACHED pts | **BASE_ALL** | hop1 / hop2 / hop3 BASE | moves / swaps / conflicts |
|---|---|---|---|---|---|
| served PHG | 0.7275 | 25.6 | 0.6727 | 0.917 / 0.529 / 0.581 | — |
| any voting block, most room (unreached golds) | 0.896 | **8.8** | **0.6826 (+1.0)** | 0.914 / 0.558 / 0.584 | 1219 / 0 / 79 |
| best-ranked voting block *inside the gold's PATH2 ball* (unreached) | 0.899 | 8.5 | **0.7156 (+4.3)** | 0.933 / 0.631 / 0.590 | 637 / 529 / 61 (13 stuck, 3 without a voter in the ball) |
| best-ranked voting block, any (unreached) | 0.901 | 8.3 | **0.7246 (+5.2)** | 0.939 / 0.622 / 0.621 | 1229 / 0 / 84 |
| best-ranked in-ball block (every gold outside P50, i.e. also reached-weak) | 0.894 | 9.0 | 0.7944 | 0.963 / 0.773 / 0.651 | 1190 / 993 / 235 |
| best-ranked any block (every gold outside P50) | 0.898 | 8.6 | 0.8164 | 0.969 / 0.808 / 0.675 | 2225 / 0 / 328 |

(`__protect_voters` variants — swap partners restricted to nodes that vote for no DEV_A query — give the same picture: 0.6806 / 0.7206 / 0.7246 / 0.8044 / 0.8164.)

Three facts fall out. (i) *Reach without rank is worth ~1 point*: the ruling's scenario "UNREACHED 27 → 10" is realised by the first row and BASE moves 0.6727 → 0.6826. (ii) The *whole* partition lever, with oracle knowledge of the golds, fixing the unreached golds tops out near **0.72** on this fusion; it reaches ~0.80–0.82 only when it also re-homes the reached-weak golds — which is not a reach problem but a block-rank problem. (iii) ≈ 8.5 pts of UNREACHED are unfixable by *any* single partition: shared answer entities whose queries' anchors vote for different blocks. For comparison the query-dependent typed-path selector (L1_P90_EXPLOIT) reaches 0.97 on the same DEV split — the reach these queries need is query-conditioned.

## 6. Non-regression — squad_phg (`eval_A_squad_phg.json`)

| | FROZEN (LOWMEM__PHG_con) | H4_SK_PATH2 (PHG_con) | paired |
|---|---|---|---|
| BASE_ALL | 0.9829 | 0.9788 | +6 / −10, p = 0.45 |
| reach_all / UNREACHED | 0.999 / 0.1 | 0.999 / 0.1 | ±0 |
| D2d_ALL | 0.9879 | 0.9869 | +4 / −5, p = 1.0 |
| median worst-gold-block rank, failed | 74 | 65 | |

No significant change (the PATH2 family carries 2.4 % of the weight on squad).

## 7. Mt-KaHyPar same-partitioner comparison — status

`_l1c_partition.py metaqa H4_SK_PATH2` (frozen worker `scratchpad/_l1hu_local_worker.py`, DETERMINISTIC_QUALITY, KM1, eps 0.03, seed 0, unit vertex weights; RSS guard as in `src/l1_canonical/partition.py`; host-memory gate so the foreign ~8 GB job is never squeezed).

* Attempt 1 (gate 2.0 GB, cap 2.0 GB, 4 threads): armed 12:25Z, launched 12:52Z after three clean samples (2.2–2.4 GB available), **killed by the RSS guard at 2,003 MB after 379 s** → `FAILED_MEMORY_CAP` (`parts/metaqa__H4_SK_PATH2.RUN__attempt1_cap2gb_FAILED_MEMORY_CAP.json`). The served H4_SK (0.53 M pins) peaked at 966 MB; PATH2 carries 2.26 M pins, so the frozen recipe needs > 2 GB here.
* Attempt 2: re-armed 13:10Z with gate 4.5 GB / cap 3.5 GB / 8 h budget (`parts/metaqa__H4_SK_PATH2.driver.log`); it runs only if the host frees ≥ 4.5 GB for three consecutive minutes, otherwise records `NOT_RUN_HOST_MEMORY_CONTENDED`. If it completes: `python -u _l1c_eval.py metaqa H4_SK_PATH2` on the served Mt-KaHyPar cache (FROZEN BASE 0.6397, UNREACHED 26.9).
* **Outcome of attempt 2: `NOT_RUN_HOST_MEMORY_CONTENDED`** (`parts/metaqa__H4_SK_PATH2.RUN.json`, armed 2026-09-14T13:08Z, 481 one-minute samples over the 8 h budget, host available 0.5–2.5 GB, never ≥ 4.5 GB; nothing launched, nothing killed). The Mt-KaHyPar same-partitioner cell of this lane therefore stays open; the PHG cell (§4) is the lane's same-partitioner comparison.

The PHG result (§4) is the lane's primary same-partitioner comparison either way — PHG is the served metaqa_phg partitioner, and the oracle envelope (§5) does not depend on the partitioner.

## 8. What this closes and what it does not

* Closed for this construction class: a bounded two-step structural co-membership family, under the frozen cap and the frozen split-preserve rule, redistributes MetaQA reach between hop populations (hop2 up, hop3 down) and moves the UNREACHED mass by < 1 pt. The hypothesis was right about *where* the golds are (distance 2–3, non-hub, undirected) and wrong about the size of the prize: the capacity-limited co-location ceiling (§2: 13.9 pts of reach at radius 2) converts to ≈ 1–5 pts of BASE because the fused block rank, not reach, is binding (§5).
* Not concluded: that a different *objective* could not do better than PHG/Mt-KaHyPar on this hypergraph — but the oracle envelope says even a perfect gold-aware placement of the unreached golds stays ≤ 0.72–0.73 on the frozen fusion, so the ceiling of the partition-only lever on MetaQA is ~+5 pts, of which a query-independent method will realise a fraction (PATH2: +1.3).
* Not re-litigated: SK vs SKN, edge families, block size, the fusion. No factorial was run; one construction, one oracle probe.

## 10. Follow-up (2026-09-14, second ruling of the day): "owner + static 1-hop STRUCT memberships that vote before P50" — what the served chain already does, and the one thing it does not (`_l1c_memdir.py`, `memdir_A_<cache>.json`)

**Code answer.** The canonical chain applies the 1-hop STRUCT boundary at two places, both at the *block* level, and never as a node halo:

1. *Pre-P50 votes* — `src/l1_canonical/replay_cache.py:154-165` builds `mem(v) = {P(v)} ∪ {P(u) : v → u is a directed STRUCT out-edge}` ("legacy load_topology semantics"); `:279-280` ranks blocks per channel with `_ta_prepartition.partition_ranking` (`scratchpad/_ta_prepartition.py:117-136`): the hit at rank r adds 1/(K0 + r) to **every** membership block (sum and max), then RRF, then P50. So "for each hit, vote for all membership blocks M(v)" *is* the served BASE — in the OUT direction. (`_l1s_core.Data.legacy_mem` replays it; the lane's OUT arm reproduces the served BASE exactly.)
2. *SAFE* (`src/l1_canonical/l1_eval.py:4-6`, `scratchpad/_l1kb_core.py:1-12, 63-107`) — `final = base_rank[:44] ∪ X`, |X| = 6 chosen by F6 from out-of-P50 blocks whose evidence is the first M_struct = 64 structural-residual nodes (1-hop neighbours of hits) and M_ret = 32 retrieval challengers. A block-swap selector: always exactly 50 core blocks.

There is **no post-P50 node halo** in the canonical chain; the node halo (`O4_FULL_C`, F6/SP1 halo rankers) lived in the legacy old-substrate lane and was never promoted (`PROMOTED=NONE`), and that lane's P17 "halo nodes also vote" was significant only on 2wiki (`FINAL_RANKING_POLICY = CORE_ONLY`). So "SP1_POST vs SP1_PRE" is not a pair the served chain can offer: PRE is served, POST is not served.

**The one difference between M(v) = {P(v)} ∪ {P(u) : (u, v) ∈ E_STRUCT} and the served table is the direction.** The served table is out-only; §2 says 99.8 % of the unreached gold nodes are unreachable by out-edges and every hop-1 unreached gold is an in-neighbour of a hit. So the narrow test — same partition, same graph, same 1-hop boundary, same P50, votes only (exposure unchanged, ≈ 5,09x nodes in every arm) — is the direction of the table:

* `OUT` served; `UND` {P(v)} ∪ {P(u) : u ~ v}; `UND_NONHUB` = OUT ∪ in-neighbour memberships for non-hub v only (frozen cap rule deg+1 ≤ 100, the H4/PATH2 hub definition; hubs keep the served table); `IN` attribution only.

| cache (partition) | table | BASE_ALL | paired vs OUT | reach | UNREACHED | reached-weak | blocks voted / query | median worst gold rank, reached-failed | \|M\| mean / p95 / max (RF) | D2d on top |
|---|---|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) | OUT | 0.6397 | — | 0.713 | 26.9 | 6.1 | 219 | 143 | 2.5 / 8 / 32 | 0.6337 |
| | UND | 0.6567 | +37/−20, p = 0.033 | 0.862 | 12.2 | 19.0 | 369 | 264 | 4.3 / 11 / **431** | 0.6577 (+37/−13, p = 0.0009) |
| | UND_NONHUB | 0.6567 | +35/−18, p = 0.027 | 0.853 | 13.1 | 18.0 | 350 | 259 | 3.9 / 11 / 83 | 0.6577 |
| | IN | 0.1946 | +35/−481 | 0.539 | 44.4 | 30.7 | 330 | 260 | 2.8 / 9 / 431 | 0.2226 |
| metaqa_phg | OUT | 0.6727 | — | 0.728 | 25.6 | 4.5 | 211 | 139 | 2.6 / 8 / 31 | 0.6737 |
| | UND | 0.6896 | +35/−18, p = 0.027 | 0.879 | 10.7 | 17.8 | 367 | 291 | 4.4 / 11 / 420 | 0.6866 |
| | UND_NONHUB | **0.6946** | +36/−14, p = 0.0026 | 0.854 | 13.1 | 15.1 | 344 | 266 | 4.0 / 11 / 90 | 0.6896 (+32/−16, p = 0.03) |
| squad (MtK) | OUT | 0.9859 | — | 0.999 | 0.1 | 0.5 | 124 | 61 | 5.0 / 15 / 42 | 0.9919 |
| | UND | 0.9768 | **0/−9, p = 0.004** | 1.000 | 0.0 | 0.9 | 177 | 66 | 10.4 / 36 / 200 | 0.9899 |
| | UND_NONHUB | 0.9829 | 0/−3, p = 0.25 | 1.000 | 0.0 | 0.8 | 137 | 70 | 5.9 / 17 / 45 | 0.9929 |
| squad_phg | OUT | 0.9829 | — | 0.999 | 0.1 | 0.7 | 133 | 71 | 6.0 / 19 / 49 | 0.9879 |
| | UND | 0.9748 | **0/−8, p = 0.008** | 0.999 | 0.1 | 1.4 | 182 | 70 | 12.4 / 42 / 201 | 0.9889 |
| | UND_NONHUB | 0.9798 | 0/−3, p = 0.25 | 0.999 | 0.1 | 1.1 | 145 | 71 | 7.0 / 21 / 50 | 0.9879 |
| musique (PHG) | OUT | 0.6596 | — | 0.890 | 11.0 | 15.6 | 227 | 98 | 4.5 / 12 / 93 | 0.7460 |
| | UND | 0.6225 | **+31/−68, p = 0.0003** | 0.946 | 5.4 | 24.8 | 565 | 138 | 11.0 / 33 / 1110 | 0.7369 |
| | UND_NONHUB | 0.6496 | +18/−28, p = 0.18 | 0.904 | 9.6 | 18.1 | 289 | 102 | 5.4 / 18 / 93 | 0.7490 |

metaqa per hop (both partitions alike): **hop1 0.917 → 0.991** (+25/−1, p = 8e-7; reach 0.945 → 1.000 — object-type hits such as persons and genres now vote for the blocks of their subject movies); hop2 0.486 → 0.483 (9/10) although its reach goes 0.590 → 0.846 (UNREACHED 39.8 → 14.8); hop3 0.527 → 0.509 (3/9; 1/7 with the hub rule) although its reach goes 0.611 → 0.744.

**Reading.**
* Overlap *already* participates in the partition ranking; changing its direction is a real but narrow effect: a hop-1 fix on the KB (+1.7 MtK / +2.2 PHG, all of it hop1) and a loss on the text corpora (raw UND significant on squad ×2 and musique; with the frozen hub rule the losses become non-significant but stay ≤ 0 everywhere). Not universal on this evidence; not the multi-hop answer.
* The multi-hop mass repeats the §5 lesson with a real static mechanism instead of an oracle: UNREACHED falls by 15–25 pts on hop2/hop3 and reached-weak rises by the same amount — with 350–370 of 432 blocks voted per query, a gold block reached through a neighbour list sits at median rank ~260, and ALL-gold @ P50 needs rank ≤ 50 for every gold block. Reach is cheap to manufacture; rank is the constraint.
* The hub caveat is real: raw UND gives hubs 172 memberships on average (max 431 = every block; musique max 1110), RF 4.3–11; the frozen cap rule bounds |M| at 83–93 and removes most of the text-corpus damage at no cost on the KB.
* No exposure change anywhere (scope 5,02x–5,11x nodes in every arm), so no matched-node-budget correction is needed for this test; the "core ∪ halo served" representation of the first proposal is a different object (the legacy O4 halo) and was not rebuilt.

## 11. Follow-up (2026-09-14, third ruling of the day): mass-conserving, support-sensitive 1-hop votes — A2 / A3 (`_l1c_massvote.py`, `massvote_A_<cache>.json`; attribution `_l1c_massvote_why.py`, `massvote_why_A_<cache>.json`)

**Ruling.** Close "more overlap / better reach / PATH2"; reach is no longer the main problem, vote strength is. Keep the exact §10 UND_NONHUB
table (same partition, same graph, same 1-hop boundary, same admissible/hub policy, same P50, same frozen `rr(S)+rr(M)` per channel and
RRF K0 = 60) and change only how a retrieved node's fixed vote budget is split among its membership blocks:

* **A0** OUT (served): every membership block gets the full `w_r = 1/(K0 + r)`.
* **A1** UND_NONHUB, equal votes (§10): every membership block gets the full `w_r` (budget = |M(v)| · w_r).
* **A2** UND_NONHUB, mass-conserving uniform: `vote(v,b) = w_r / |M(v)|`.
* **A3** UND_NONHUB, support-normalised: `a(v,b) = 1[P(v)=b] + |{u ∈ N_adm(v) : P(u)=b}|`, `p(b|v) = a(v,b)/Σ_c a(v,c)`, `vote(v,b) = w_r · p(b|v)` — the row of `T = D⁻¹(I + A_adm)H`.

Implementation: `weighted_table` builds `(ptr, blocks, probs)` per node with `a(v,b)` from `np.unique(..., return_counts=True)` (own block
counts 1 + every admissible neighbour in b); `weighted_partition_ranking` replays `_ta_prepartition.partition_ranking` with a per-membership
weight (`v = (w_r · p).astype(float32)`; `S[q, blocks] += v`; `np.maximum.at(M[q], blocks, v)`; `votes = rr(S) + rr(M)`), and asserts on
every cache that the A0 table equals `D.legacy_mem()` and that the weighted ranking with unit weights reproduces the frozen
`TA.partition_ranking` bit-for-bit (dense channel). Decision rule set before the run: "if A2/A3 don't materially improve MetaQA, close the
1-hop structural-vote idea completely". Expectation stated: UNREACHED stays near A1's value, reached-weak drops substantially, median
missed-reached rank 260 → <100 (ideally <50), hop2/hop3 reach ↑ *and* rank ↑. Exposure = 50 core blocks in every arm (scope 5,02x–5,12x nodes).

**Result (DEV_A; every arm on every cache).** Reach and UNREACHED are identical to A1 by construction (the membership *sets* do not change);
everything else moves the wrong way.

| cache (partition) | arm | BASE_ALL | paired vs A0 | paired vs A1 | reach | UNREACHED | reached-weak | median worst gold rank: reached-failed | all-reached | own-block share / max share, mean over nodes with \|M\|>1 | D2d on top (vs A1) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) | A0 | 0.6397 | — | — | 0.713 | 26.9 | 6.1 | 143 | 1 | 1.00 / 1.00 | 0.6337 |
|  | A1 | 0.6567 | **+35/−18, p = 0.027** | — | 0.853 | 13.1 | 18.0 | 259 | 3 | 1.00 / 1.00 | 0.6577 |
|  | A2 | 0.6128 | **+27/−54, p = 0.004** | **+8/−52, p = 5e-09** | 0.853 | 13.1 | 21.0 | 207 | 4 | 0.26 / 0.26 | 0.5978 (**+5/−65, p = 2e-14**) |
|  | A3 | 0.6148 | **+29/−54, p = 0.008** | **+15/−57, p = 7e-07** | 0.853 | 13.1 | 21.4 | 216 | 4 | 0.32 / 0.39 | 0.5858 (**+6/−78, p = 5e-17**) |
| metaqa_phg | A0 | 0.6727 | — | — | 0.728 | 25.6 | 4.5 | 139 | 1 | 1.00 / 1.00 | 0.6737 |
|  | A1 | 0.6946 | **+36/−14, p = 0.003** | — | 0.854 | 13.1 | 15.1 | 266 | 2 | 1.00 / 1.00 | 0.6896 |
|  | A2 | 0.6407 | **+24/−56, p = 5e-04** | **+4/−58, p = 3e-13** | 0.854 | 13.1 | 19.5 | 225 | 4 | 0.26 / 0.26 | 0.6387 (**+6/−57, p = 2e-11**) |
|  | A3 | 0.6327 | **+25/−65, p = 3e-05** | **+5/−67, p = 6e-15** | 0.854 | 13.1 | 21.1 | 214 | 3 | 0.31 / 0.38 | 0.6168 (**+5/−78, p = 6e-18**) |
| squad (MtK) | A0 | 0.9859 | — | — | 0.999 | 0.1 | 0.5 | 61 | 1 | 1.00 / 1.00 | 0.9919 |
|  | A1 | 0.9829 | +0/−3, p = 0.250 | — | 1.000 | 0.0 | 0.8 | 70 | 1 | 1.00 / 1.00 | 0.9929 |
|  | A2 | 0.9819 | +8/−12, p = 0.503 | +9/−10, p = 1.000 | 1.000 | 0.0 | 1.5 | 82 | 1 | 0.21 / 0.21 | 0.9919 (+1/−2, p = 1.000) |
|  | A3 | 0.9627 | **+8/−31, p = 3e-04** | **+9/−29, p = 0.002** | 1.000 | 0.0 | 2.6 | 69 | 1 | 0.14 / 0.45 | 0.9869 (**+0/−6, p = 0.031**) |
| squad_phg | A0 | 0.9829 | — | — | 0.999 | 0.1 | 0.7 | 71 | 1 | 1.00 / 1.00 | 0.9879 |
|  | A1 | 0.9798 | +0/−3, p = 0.250 | — | 0.999 | 0.1 | 1.1 | 71 | 1 | 1.00 / 1.00 | 0.9879 |
|  | A2 | 0.9698 | **+5/−18, p = 0.011** | +8/−18, p = 0.076 | 0.999 | 0.1 | 1.5 | 68 | 1 | 0.19 / 0.19 | 0.9929 (+6/−1, p = 0.125) |
|  | A3 | 0.9415 | **+6/−47, p = 6e-09** | **+8/−46, p = 1e-07** | 0.999 | 0.1 | 4.3 | 78 | 2 | 0.10 / 0.42 | 0.9849 (+4/−7, p = 0.549) |
| musique (PHG) | A0 | 0.6596 | — | — | 0.890 | 11.0 | 15.6 | 98 | 17 | 1.00 / 1.00 | 0.7460 |
|  | A1 | 0.6496 | +18/−28, p = 0.184 | — | 0.904 | 9.6 | 18.1 | 102 | 17 | 1.00 / 1.00 | 0.7490 |
|  | A2 | 0.6376 | +49/−71, p = 0.055 | +53/−65, p = 0.311 | 0.904 | 9.6 | 20.4 | 96 | 21 | 0.27 / 0.27 | 0.7349 (+27/−41, p = 0.114) |
|  | A3 | 0.5653 | **+42/−136, p = 9e-13** | **+43/−127, p = 8e-11** | 0.904 | 9.6 | 28.8 | 104 | 31 | 0.16 / 0.53 | 0.7018 (**+19/−66, p = 3e-07**) |

metaqa per hop, `BASE (paired vs A1) · reach · UNREACHED pts · reached-weak pts · median worst gold rank among reached-failed`:

| cache | arm | hop1 (n = 326) | hop2 (n = 344) | hop3 (n = 332) |
|---|---|---|---|---|
| metaqa (MtK) | A0 | 0.917 · reach 0.945 · UNR 5.5 · weak 2.1 · rank 139 | 0.485 · reach 0.590 · UNR 39.8 · weak 9.0 · rank 146 | 0.527 · reach 0.611 · UNR 34.6 · weak 6.9 · rank 145 |
|  | A1 | 0.991 · reach 1.000 · UNR 0.0 · weak 0.6 · rank 123 | 0.483 · reach 0.837 · UNR 15.7 · weak 33.1 · rank 244 | 0.509 · reach 0.726 · UNR 23.2 · weak 19.3 · rank 289 |
|  | A2 | 0.874 (**+1/−39, p = 7e-11**) · reach 1.000 · UNR 0.0 · weak 11.0 · rank 91 | 0.485 (+4/−3, p = 1.000) · reach 0.837 · UNR 15.7 · weak 31.1 · rank 234 | 0.488 (+3/−10, p = 0.092) · reach 0.726 · UNR 23.2 · weak 20.2 · rank 265 |
|  | A3 | 0.850 (**+1/−47, p = 3e-13**) · reach 1.000 · UNR 0.0 · weak 13.2 · rank 86 | 0.497 (+9/−4, p = 0.267) · reach 0.837 · UNR 15.7 · weak 30.8 · rank 237 | 0.506 (+5/−6, p = 1.000) · reach 0.726 · UNR 23.2 · weak 19.6 · rank 282 |
| metaqa_phg | A0 | 0.917 · reach 0.945 · UNR 5.5 · weak 2.1 · rank 138 | 0.529 · reach 0.628 · UNR 36.0 · weak 7.8 · rank 133 | 0.581 · reach 0.618 · UNR 34.6 · weak 3.3 · rank 139 |
|  | A1 | 0.991 · reach 1.000 · UNR 0.0 · weak 0.3 · rank 69 | 0.541 · reach 0.811 · UNR 18.0 · weak 26.2 · rank 251 | 0.563 · reach 0.756 · UNR 20.8 · weak 18.1 · rank 300 |
|  | A2 | 0.865 (**+1/−42, p = 1e-11**) · reach 1.000 · UNR 0.0 · weak 12.0 · rank 80 | 0.509 (**+1/−12, p = 0.003**) · reach 0.811 · UNR 18.0 · weak 27.6 · rank 245 | 0.557 (+2/−4, p = 0.688) · reach 0.756 · UNR 20.8 · weak 18.4 · rank 310 |
|  | A3 | 0.841 (**+1/−50, p = 5e-14**) · reach 1.000 · UNR 0.0 · weak 14.7 · rank 81 | 0.514 (**+3/−12, p = 0.035**) · reach 0.811 · UNR 18.0 · weak 27.9 · rank 253 | 0.551 (+1/−5, p = 0.219) · reach 0.756 · UNR 20.8 · weak 20.2 · rank 303 |

**Against the stated expectations (metaqa, MtK / PHG):**

| expectation | observed |
|---|---|
| UNREACHED remains near A1's value | yes, by construction: 13.1 / 13.1 pts (sets unchanged) |
| reached-weak drops substantially | **rises**: 18.0 → 21.0 (A2) / 21.4 (A3) on MtK; 15.1 → 19.5 / 21.1 on PHG |
| median missed-reached gold rank 260 → <100, ideally <50 | 259 → 207 / 216 (MtK); 266 → 225 / 214 (PHG). Still ~4× the P50 cut; and the median over *all* reached queries gets worse (3 → 4; 2 → 4 / 3) |
| hop2 / hop3: reach ↑ and rank ↑ | reach unchanged vs A1; hop2 BASE 0.483 → 0.485 / 0.497 and hop3 0.509 → 0.488 / 0.506 on MtK (all ns vs A1); on PHG hop2 −12/+1 (A2, p = 0.003) and −12/+3 (A3, p = 0.035) |
| "materially improve MetaQA" | **no**: BASE_ALL 0.6567 (A1) → **0.6128** (A2, +8/−52, p = 5e-9) / **0.6148** (A3, +15/−57, p = 7e-7) on MtK; 0.6946 → **0.6407** / **0.6327** on PHG (p = 3e-13 / 6e-15). Both are also significantly below the *served* A0 (0.6397 / 0.6727). |

The text corpora and musique agree: squad A3 0.9829 → 0.9627 (+9/−29, p = 0.002 vs A1), squad_phg A3 0.9798 → 0.9415 (+8/−46, p = 1e-7),
musique A3 0.6496 → 0.5653 (+43/−127, p = 8e-11; reached-weak 18.1 → 28.8); A2 is never significantly better than A1 anywhere
(musique +53/−65, squad +9/−10) and is significantly worse than the served A0 on metaqa ×2, squad_phg, and at p = 0.055 on musique. D2d on top
of A2/A3 inherits the loss (metaqa 0.6577 → 0.598 / 0.586).

**Where the losses come from (`_l1c_massvote_why.py`; A1-correct queries that A3 loses, per hop).** The hypothesis "normalisation starves the
hit's own block" is *not* what the lost queries show: among the hop-1 losses the block that falls out of P50 is the own block of the rank-1
hit in 1 / 47 (MtK) and 0 / 50 (PHG) cases, and the own block of *any* top-10 hit in 1 / 47 and 2 / 50.

| cache | hop | n | A1 ok → A3 ok | lost / gained (A1 → A3) | fallen gold block = own block of the rank-1 hit | … of any top-10 hit | fallen block's fused position under A3 (median) |
|---|---|---|---|---|---|---|---|
| metaqa (MtK) | hop1 | 326 | 323 → 277 | 47 / 1 | 1 | 1 | 84 |
|  | hop2 | 344 | 166 → 171 | 4 / 9 | 0 | 0 | 102 |
|  | hop3 | 332 | 169 → 168 | 6 / 5 | 0 | 0 | 66 |
| metaqa_phg | hop1 | 326 | 323 → 274 | 50 / 1 | 0 | 2 | 79 |
|  | hop2 | 344 | 186 → 177 | 12 / 3 | 2 | 4 | 97 |
|  | hop3 | 332 | 187 → 183 | 5 / 1 | 0 | 1 | 96 |

So the lost gold blocks are exactly the blocks that A1 reaches through a **neighbour** membership: for a hop-1 anchor such as a person or
genre node, the gold movies' blocks are in-neighbour memberships (§2, §10), and each holds a small share of the anchor's support (own-block
share of an |M|>1 node: 0.26 mean under A2, 0.32 under A3; the *largest* share any block gets is 0.39 mean under A3; a(v, gold block) is
typically 1–2 out of 1 + deg). Under A1 that neighbour vote arrives at full `w_r` — the top hit counts as a top hit in every block it exposes,
which is what put the gold block inside P50 (hop1 BASE 0.991). Under A2/A3 the top hit's vote to the gold block is `w_0 / |M|` ≈ 1/(60·4)
… 1/(60·10), i.e. weaker than the *full* own-block vote of a rank-100 hit (≈ 1/160); the gold block then loses to blocks that merely contain
deep-rank hits and settles at median fused position 79–84 (hop1). Mass conservation is therefore the wrong invariant here: on this KB the
evidence that a top-ranked hit carries for its neighbours' blocks is *not* a fixed budget to be shared — the full-weight duplication of A1
was the mechanism, and the support signal a(v,b) (what A3 adds) points to the *wrong* blocks often enough that A3 < A2 on four of the five caches (metaqa MtK: A3 ahead of A2 by 2 queries).

**Verdict.** A2 and A3 do not materially improve MetaQA — both are significantly worse than A1 and than the served A0 on both metaqa caches;
A3 is significantly worse than both on every cache, and A2 is never better than A1 anywhere — so, per the ruling's own rule, the 1-hop structural-vote idea is **closed completely**: the served OUT table
stays; UND_NONHUB (A1) remains a recorded hop-1-only KB effect that does not transfer to text (§10); no mass-conserving or support-weighted
variant is carried forward, no further weighting variants were run (no factorial), and DEV_B was not read. The question "how should a
retrieved node's fixed evidence be distributed among the partitions it exposes" has, on this evidence, the answer the served chain already
gives — *undivided* — and the remaining MetaQA multi-hop mass is a rank problem that no static per-node weighting of the same votes moved
(reached-weak 18–21 pts, median missed-reached rank ≥ 207 in every arm).

## 12. Route 1 (2026-09-14, fourth ruling of the day): a query-conditioned, parameter-free discriminator over already-reached blocks — RELSIG (`_l1c_relsig.py`, `relsig_A_<cache>.json`, pre-registered in `PREREGISTRATION_RELSIG_DEV_A.json`; attribution `_l1c_relsig_why.py`, `relsig_why_A_<cache>.json`)

**Ruling recorded first.** `ONE_HOP_STRUCTURAL_VOTE = CLOSED` — served OUT/full vote remains canonical; UND_NONHUB/full vote = MetaQA
diagnostic only; mass-conserving variants rejected; support-normalised vote rejected (§10–§11). Membership is an existential routing
statement, not a probability distribution: a rank-1 hit stays a rank-1-quality reason for every block it exposes. Next question, and the
only one run: *can a query-conditioned discriminator over blocks that the served votes and the static 1-hop memberships already reach —
no walk, no expansion, no changed membership sets, no learned weights — convert A1's reach into top-50 rank?* One version, decision rule
and module sha written before any number existed (`PREREGISTRATION_RELSIG_DEV_A.json`, sha256 9e52d1ca6fd1…).

**The version.** `S_q(b) = S_base(b) + compat(q, static relation signature of the hit → b membership)`:

* static (from edge IDs, like the served table): `TM(v) = {(P(u), ρ)}` over the admissible typed STRUCT edges — every directed out-edge,
  plus in-edges for non-hub v (the frozen cap rule; identical admissibility to A1);
* query side (existing lexical representation only): `R(q)` = the **set** of relation labels of the frozen typed-path rule's schedule
  (`results/L1_P90_EXPLOIT/_uni_<cache>.npz` `orders`: lexical Porter/irregular-lemma match of relation labels in the question, mention
  masked; order ignored; empty for 6.7 % of metaqa DEV_A, mean |R(q)| = 1.47; empty for every query of an untyped graph);
* runtime, per channel, hit v at rank r: base blocks (own + out-neighbour, served) get `w_r`; compat blocks `{b : (b, ρ) ∈ TM(v), ρ ∈ R(q)}`
  get `w_r`; a block in both gets `2 w_r`; then the frozen `rr(S)+rr(M)`, frozen RRF, P50. `R(q) = ∅` ⇒ exactly the served ranking.
  Untyped graph (< 2 relation labels) ⇒ the compat table is never built ⇒ served by construction (squad/musique run as the identity check).
* why membership-level and not a per-block relation histogram: a per-block relation-count signature is a node-type composition
  (movie-rich vs person-rich blocks) and cannot separate "movies X starred in" from "movies X directed" — both are movie blocks; the
  distinction lives on the relation label of the hit → block membership, which is the static signature scored here.

Assertions passed on every cache: A0 == frozen `partition_ranking` and served evidence; A1 == §11's A1.
**Pre-registered rule:** CANDIDATE iff on BOTH metaqa caches BASE_ALL R1 vs A0 has gained > lost with exact McNemar p < 0.01 **and**
hop2+hop3 pooled R1 vs A0 has gained > lost with p < 0.05, and squad/squad_phg/musique are identical to A0; else NOT_CONFIRMED → route 2.

**Result (DEV_A).**

| cache (partition) | arm | BASE_ALL | paired vs A0 | paired vs A1 | hop2+hop3 pooled vs A0 | reach | UNREACHED | reached-weak | median worst gold rank: reached-failed | all-reached | blocks voted / query | D2d on top (vs A0) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) | A0 | 0.6397 | — | — | — | 0.713 | 26.9 | 6.1 | 143 | 1 | 219 | 0.6337 |
|  | A1 | 0.6567 | **+35/−18, p = 0.027** | — | +10/−17, p = 0.248 | 0.853 | 13.1 | 18.0 | 259 | 3 | 350 | 0.6577 (**+36/−12, p = 7e-04**) |
|  | R1 | 0.6577 | **+26/−8, p = 0.003** | +17/−16, p = 1.000 | +13/−8, p = 0.383 | 0.823 | 16.1 | 14.4 | 231 | 1 | 312 | 0.6467 (**+17/−4, p = 0.007**) |
| metaqa_phg | A0 | 0.6727 | — | — | — | 0.728 | 25.6 | 4.5 | 139 | 1 | 211 | 0.6737 |
|  | A1 | 0.6946 | **+36/−14, p = 0.003** | — | +11/−13, p = 0.839 | 0.854 | 13.1 | 15.1 | 266 | 2 | 344 | 0.6896 (**+32/−16, p = 0.029**) |
|  | R1 | 0.6856 | **+19/−6, p = 0.015** | +10/−19, p = 0.136 | +7/−6, p = 1.000 | 0.818 | 16.6 | 11.7 | 225 | 1 | 308 | 0.6776 (+12/−8, p = 0.503) |
| squad (MtK) | A0 | 0.9859 | — | — | — | 0.999 | 0.1 | 0.5 | 61 | 1 | 124 | 0.9919 |
|  | A1 | 0.9829 | +0/−3, p = 0.250 | — | — | 1.000 | 0.0 | 0.8 | 70 | 1 | 137 | 0.9929 (+1/−0, p = 1.000) |
|  | R1 | 0.9859 | +0/−0, p = 1.000 | +3/−0, p = 0.250 | — | 0.999 | 0.1 | 0.5 | 61 | 1 | 124 | 0.9919 (+0/−0, p = 1.000) |
| squad_phg | A0 | 0.9829 | — | — | — | 0.999 | 0.1 | 0.7 | 71 | 1 | 133 | 0.9879 |
|  | A1 | 0.9798 | +0/−3, p = 0.250 | — | — | 0.999 | 0.1 | 1.1 | 71 | 1 | 145 | 0.9879 (+0/−0, p = 1.000) |
|  | R1 | 0.9829 | +0/−0, p = 1.000 | +3/−0, p = 0.250 | — | 0.999 | 0.1 | 0.7 | 71 | 1 | 133 | 0.9879 (+0/−0, p = 1.000) |
| musique (PHG) | A0 | 0.6596 | — | — | — | 0.890 | 11.0 | 15.6 | 98 | 17 | 227 | 0.7460 |
|  | A1 | 0.6496 | +18/−28, p = 0.184 | — | — | 0.904 | 9.6 | 18.1 | 102 | 17 | 289 | 0.7490 (+18/−15, p = 0.728) |
|  | R1 | 0.6596 | +0/−0, p = 1.000 | +28/−18, p = 0.184 | — | 0.890 | 11.0 | 15.6 | 98 | 17 | 227 | 0.7460 (+0/−0, p = 1.000) |

metaqa per hop, `BASE (paired vs A0) · reach · UNREACHED pts · reached-weak pts · median worst gold rank among reached-failed`:

| cache | arm | hop1 (n = 326) | hop2 (n = 344) | hop3 (n = 332) |
|---|---|---|---|---|
| metaqa (MtK) | A0 | 0.917 · reach 0.945 · UNR 5.5 · weak 2.1 · rank 139 | 0.485 · reach 0.590 · UNR 39.8 · weak 9.0 · rank 146 | 0.527 · reach 0.611 · UNR 34.6 · weak 6.9 · rank 145 |
|  | A1 | 0.991 (**+25/−1, p = 8e-07**) · reach 1.000 · UNR 0.0 · weak 0.6 · rank 123 | 0.483 (+9/−10, p = 1.000) · reach 0.837 · UNR 15.7 · weak 33.1 · rank 244 | 0.509 (+1/−7, p = 0.070) · reach 0.726 · UNR 23.2 · weak 19.3 · rank 289 |
|  | R1 | 0.957 (**+13/−0, p = 2e-04**) · reach 0.985 · UNR 1.5 · weak 2.5 · rank 71 | 0.500 (+11/−6, p = 0.332) · reach 0.785 · UNR 20.9 · weak 25.6 · rank 233 | 0.527 (+2/−2, p = 1.000) · reach 0.705 · UNR 25.3 · weak 14.5 · rank 243 |
| metaqa_phg | A0 | 0.917 · reach 0.945 · UNR 5.5 · weak 2.1 · rank 138 | 0.529 · reach 0.628 · UNR 36.0 · weak 7.8 · rank 133 | 0.581 · reach 0.618 · UNR 34.6 · weak 3.3 · rank 139 |
|  | A1 | 0.991 (**+25/−1, p = 8e-07**) · reach 1.000 · UNR 0.0 · weak 0.3 · rank 69 | 0.541 (+10/−6, p = 0.454) · reach 0.811 · UNR 18.0 · weak 26.2 · rank 251 | 0.563 (+1/−7, p = 0.070) · reach 0.756 · UNR 20.8 · weak 18.1 · rank 300 |
|  | R1 | 0.954 (**+12/−0, p = 5e-04**) · reach 0.994 · UNR 0.6 · weak 3.4 · rank 55 | 0.535 (+7/−5, p = 0.774) · reach 0.764 · UNR 22.4 · weak 19.8 · rank 223 | 0.578 (+0/−1, p = 1.000) · reach 0.702 · UNR 26.2 · weak 11.4 · rank 270 |

**Verdict against the pre-registered rule: NOT_CONFIRMED.** metaqa (MtK): BASE_ALL 0.6397 → 0.6577 (**+26/−8, p = 0.003**) passes the first gate, but the
hop2+hop3 pooled test (+13/−8, p = 0.383) fails the second; metaqa_phg: 0.6727 → 0.6856 (**+19/−6, p = 0.015**) fails the first (p ≥ 0.01) and the pooled test (+7/−6, p = 1.000) the second.
The identity checks hold (squad, squad_phg, musique: R1 = A0, 0 gained / 0 lost). Every point R1 gains over the served chain is hop-1
(MtK +13/−0, PHG +12/−0): the relation-compatible in-neighbour memberships of the anchor ("movies starring X"), i.e. A1's hop-1 effect
made text-safe by the typed gate, minus the 13 hop-1 queries where R(q) is empty or mismatched (R1 < A1 on hop1: +2/−13, +1/−13). On
hop2/hop3 the discriminator is better than A1's undirected spray (MtK pooled +15/−3 vs A1, p = 0.008; PHG +9/−6, ns) — reached-weak
18.0 → 14.4 / 15.1 → 11.7 — but not better than the served chain (pooled vs A0 +13/−8 / +7/−6). Blocks voted per query stay at 307–312
(A1 350, A0 211–219): with R(q) ⊇ {starred_actors, directed_by, written_by} for most queries, "compatible" removes little of the spray.

**Why it stops there (`_l1c_relsig_why.py`; DEV_A, per hop; `best vote rank` = the best 0-based hit rank, over both channels, of a hit
whose membership voted for the query's worst-ranked gold block; buckets rank0 / rank1–4 / rank5–19 / rank20–99 / none):**

| cache | hop | n | R1 ok | reached-failed | reached-failed: worst gold block, best ANY vote rank | … best COMPAT vote rank | worst gold block fused position (median) | R1-correct queries: worst gold block, best COMPAT vote rank | queries where every gold block has a rank-0 / top-5 compat vote |
|---|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) | hop1 | 326 | 312 | 9 | 5 / 0 / 1 / 3 / 0 | 5 / 0 / 1 / 0 / 3 | 71 | 252 / 3 / 0 / 0 / 57 | 257 / 260 |
|  | hop2 | 344 | 172 | 96 | 0 / 2 / 17 / 77 / 0 | 0 / 2 / 12 / 68 / 14 | 231 | 58 / 84 / 12 / 8 / 10 | 57 / 143 |
|  | hop3 | 332 | 175 | 59 | 0 / 1 / 10 / 48 / 0 | 0 / 1 / 5 / 45 / 8 | 243 | 114 / 26 / 15 / 12 / 8 | 112 / 139 |
| metaqa_phg | hop1 | 326 | 311 | 13 | 8 / 0 / 2 / 3 / 0 | 8 / 0 / 1 / 0 / 4 | 55 | 248 / 4 / 0 / 0 / 59 | 256 / 260 |
|  | hop2 | 344 | 184 | 79 | 0 / 2 / 17 / 60 / 0 | 0 / 2 / 10 / 51 / 16 | 223 | 58 / 83 / 18 / 13 / 12 | 57 / 141 |
|  | hop3 | 332 | 192 | 41 | 0 / 2 / 7 / 32 / 0 | 0 / 2 / 4 / 29 / 6 | 270 | 103 / 44 / 23 / 14 / 8 | 103 / 147 |

When R1 gets a hop-2/hop-3 query right, the worst gold block was exposed by a top-5 hit's relation-compatible membership in 142 / 172
(hop2) and 140 / 175 (hop3) cases on MtK — the discriminator works exactly where a top hit exposes the gold block. When it still fails on a
reached query, the gold block's best exposing hit sits at rank ≥ 20 in 80 % (hop2 77 / 96) and 81 % (hop3 48 / 59) of the cases, and at
rank 0 in none: the anchor does not expose it at all. Those blocks are reached *coincidentally* by deep hits (median fused position 223–270),
and no re-scoring of the memberships that exist can promote them — the block that would expose them belongs to the intermediate entity
(the actor, the director), which is not a retrieval hit. Only 143 / 344 hop-2 and 139 / 332 hop-3 queries have every gold block inside a
top-5 hit's compatible exposure at all, and R1 is already correct on 172 / 344 and 175 / 332 — more than that set — so the
discriminator has exhausted what a top hit's typed 1-hop memberships can expose. The remaining MetaQA multi-hop mass is
therefore a second-hop problem, not a discrimination problem: the information that is missing is the intermediate node, and producing it
is a walk.

**Closing.** Route 1 is closed on this evidence: the one parameter-free, walk-free, query-conditioned discriminator that carries relation
intent (typed 1-hop memberships gated by the frozen lexical relation match) converts the part of A1's reach that a top hit exposes and
nothing else; it does not meet the pre-registered multi-hop gate on either metaqa cache. Per the ruling, MetaQA's remaining multi-hop gap
(hop2 0.50 / hop3 0.53 at P50 against the typed walk's 0.97) is conceded to dynamic relation reasoning (L3 / the typed-path selector).
Not carried forward without a ruling: R1's hop-1-only, text-safe gain (MtK +1.8, p = 0.003; PHG +1.3, p = 0.015) — it is the same
phenomenon as A1's hop-1 gain with the text cost removed by the typed gate, and would need a genuinely untouched transfer population
(WebQSP, whose Freebase labels the same lexical matcher tokenises) before any claim. No variant of R1 was run; DEV_B untouched.

## 13. Fifth ruling of the day: edge / triplet geometry voting — valid graph facts as a parameter-free retrieval unit (`_l1c_edgegeo.py`, `edgegeo_A_<cache>.json`, pre-registered in `PREREGISTRATION_EDGEGEO_DEV_A.json`; attribution `_l1c_edgegeo_why.py`, `edgegeo_why_A_<cache>.json`)

**Ruling recorded first.** Two versions of "triplet voting" were separated. Version 1 — node hit → its incident triplets → their
blocks — is what §10–§12 approached and is closed. Version 2 — retrieve the *facts themselves* and let each retrieved fact vote for
`{P(s), P(o)}` — is genuinely different and was authorised as an explicitly declared **triplet channel**; the node channel stays
NAME_ONLY (the NAME_ONLY ruling applies to node representations, not to a declared edge channel). Hard constraint: one algorithm, one
mathematical rule, no dataset switches, no training, no relation-specific parameters, no relation text, no new or retrained embeddings, no
query-time walk, no hand-tuned thresholds; consume only `q`, `e_v`, STRUCT edges, `P(v)` and the served Dense/SPLADE ranks; the
structural unit is the pair `(s, o)`, `r` is never read; single-shot independent scoring only (no result-conditioned second stage); tiny
controlled experiment on DEV_A first; "if it doesn't improve MetaQA materially, close it; if it does, build the scalable global
implementation". Research question: *can valid graph facts themselves serve as a parameter-free retrieval unit, using only frozen node
geometry, without relation embeddings, training, or traversal?* Design, arms, decision rule and module sha written before any number
existed (`PREREGISTRATION_EDGEGEO_DEV_A.json`, sha256 746f63ee9dce…).

**The version.** Unit = undirected STRUCT edge `t = {s, o}` from the frozen expansion key set (deduped, self-loops dropped; metaqa
124,669 edges, squad 744,734, musique 2,651,280). With unit node vectors and the unit query vector, every edge score is a closed form in three
dot products — `a_s = q·e_s`, `a_o = q·e_o` (one `q·E` per query, chunked over the fp16 memmap) and `c = e_s·e_o` (query-independent,
precomputed once per edge):

* `S_mid(t) = cos(q, norm(e_s + e_o)) = (a_s + a_o) / √(2 + 2c)` — the edge-centre signal;
* `S_dir(t) = max over the two orientations of cos(q − e_s, e_o − e_s) = (a_o − a_s − c + 1) / (√(2 − 2a_s) √(2 − 2c))` — the
  parameter-free symmetric reading for an unordered pair (identical endpoint vectors ⇒ no direction ⇒ −1);
* `S_src / S_dst` = the endpoint similarities under the better orientation (used only in the information arm T3b).

Per edge channel the top K_LOCK = 100 edges vote `w_r = 1/(K0 + r)` for each endpoint block `{P(s), P(o)}` (set semantics), through the
frozen per-channel rule `rr(S) + rr(M)` (float32 accumulators exactly as `partition_ranking`), fused by the frozen RRF (K0 = 60), P50.
Candidate sets: **C1000** (primary, the ruling's controlled set) = edges with an endpoint in the query's served node top-1000 pool
(dense ∪ SPLADE); **ALL** = every STRUCT edge, brute force (the preview of the global static index). Arms: **T0** canonical BASE
(asserted equal to the served BASE_ALL on every cache); **T1** mid-edge channel alone; **T2** dir-edge channel alone; **T3** = frozen
RRF(node dense, node SPLADE, mid-edge, dir-edge) — the decision arm; **T3b** = RRF(node dense, node SPLADE, triplet) with triplet rank =
frozen RRF(src, dst, dir, mid) over the candidate edges (information). Same code path on all five caches; no typed-graph test, no dataset
branch, no constant beyond K0 / K_LOCK / P50 / the served pool depth. Node norms 1.000 ± 0.0005 on every corpus (the fp16 rows are unit
vectors; the normalisation is a no-op).
**Pre-registered rule:** CANDIDATE iff on BOTH metaqa caches T3_C1000 vs T0 has gained > lost with exact McNemar p < 0.01 **and** an
absolute gain ≥ +2.0 points, **and** hop2+hop3 pooled gained > lost with p < 0.05, **and** no text cache with a significant regression;
else `EDGE_GEOMETRY_VOTE = CLOSED (NOT_CONFIRMED on DEV_A)`.

**Result (DEV_A).**

| cache (partition) | arm | BASE_ALL | paired vs T0 | hop2+hop3 pooled vs T0 | reach (paired vs T0) | UNREACHED | reached-weak | fusion-fixable | median worst gold rank: reached-failed | all-reached | blocks voted / query | scope nodes @P50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) | T0 | 0.6397 | — | — | 0.713 | 26.9 | 6.1 | 1.2 | 143 | 1 | 219 | 5023 |
|  | T1_C1000 | 0.5240 | **+35/−151, p = 2e-18** | **+12/−146, p = 2e-30** | 0.570 (**+31/−174, p = 2e-25**) | 41.2 | 4.6 | 0.0 | 74 | 5 | 85 | 5025 |
|  | T2_C1000 | 0.6228 | +38/−55, p = 0.097 | **+17/−41, p = 0.002** | 0.660 (**+32/−85, p = 1e-06**) | 32.2 | 3.7 | 0.0 | 69 | 2 | 93 | 5018 |
|  | T3_C1000 | 0.6687 | **+36/−7, p = 9e-06** | +13/−7, p = 0.263 | 0.769 (**+57/−0, p = 1e-17**) | 21.3 | 6.9 | 3.2 | 178 | 3 | 267 | 5026 |
|  | T3b_C1000 | 0.6487 | +16/−7, p = 0.093 | +7/−7, p = 1.000 | 0.755 (**+43/−0, p = 2e-13**) | 22.7 | 6.2 | 4.5 | 131 | 1 | 241 | 5026 |
|  | T3_ALL | 0.6687 | **+36/−7, p = 9e-06** | +13/−7, p = 0.263 | 0.769 (**+57/−0, p = 1e-17**) | 21.3 | 6.9 | 3.2 | 178 | 3 | 268 | 5026 |
| metaqa_phg | T0 | 0.6727 | — | — | 0.728 | 25.6 | 4.5 | 1.0 | 139 | 1 | 211 | 5089 |
|  | T1_C1000 | 0.5449 | **+31/−159, p = 6e-22** | **+9/−155, p = 2e-35** | 0.586 (**+25/−167, p = 6e-27**) | 39.7 | 4.2 | 0.0 | 74 | 5 | 86 | 5058 |
|  | T2_C1000 | 0.6577 | +38/−53, p = 0.142 | **+15/−40, p = 0.001** | 0.694 (**+30/−64, p = 6e-04**) | 29.0 | 3.6 | 0.0 | 73 | 3 | 92 | 5058 |
|  | T3_C1000 | 0.7056 | **+38/−5, p = 2e-07** | **+16/−5, p = 0.027** | 0.779 (**+52/−0, p = 4e-16**) | 20.5 | 4.9 | 2.5 | 174 | 2 | 260 | 5067 |
|  | T3b_C1000 | 0.6936 | **+22/−1, p = 6e-06** | **+11/−1, p = 0.006** | 0.764 (**+37/−0, p = 1e-11**) | 22.0 | 4.6 | 2.5 | 134 | 2 | 234 | 5073 |
|  | T3_ALL | 0.7056 | **+38/−5, p = 2e-07** | **+16/−5, p = 0.027** | 0.778 (**+51/−0, p = 9e-16**) | 20.6 | 4.8 | 2.5 | 174 | 2 | 261 | 5068 |
| squad (MtK) | T0 | 0.9859 | — | — | 0.999 | 0.1 | 0.5 | 0.8 | 61 | 1 | 124 | 5089 |
|  | T1_C1000 | 0.9385 | **+3/−50, p = 6e-12** | — | 0.925 (**+0/−73, p = 2e-22**) | 6.1 | 0.0 | 0.0 | None | 0 | 20 | 5046 |
|  | T2_C1000 | 0.9123 | **+2/−75, p = 4e-20** | — | 0.900 (**+0/−98, p = 6e-30**) | 8.6 | 0.2 | 0.0 | 54 | 0 | 24 | 5052 |
|  | T3_C1000 | 0.9788 | +4/−11, p = 0.118 | — | 0.999 (+0/−0, p = 1.000) | 0.1 | 0.4 | 1.6 | 76 | 0 | 128 | 5085 |
|  | T3b_C1000 | 0.9829 | +3/−6, p = 0.508 | — | 0.999 (+0/−0, p = 1.000) | 0.1 | 0.5 | 1.1 | 69 | 1 | 125 | 5087 |
|  | T3_ALL | 0.9788 | +4/−11, p = 0.118 | — | 0.999 (+0/−0, p = 1.000) | 0.1 | 0.4 | 1.6 | 76 | 0 | 128 | 5085 |
| squad_phg | T0 | 0.9829 | — | — | 0.999 | 0.1 | 0.7 | 0.9 | 71 | 1 | 133 | 5089 |
|  | T1_C1000 | 0.9466 | **+3/−39, p = 6e-09** | — | 0.929 (**+0/−69, p = 3e-21**) | 5.3 | 0.0 | 0.0 | None | 1 | 22 | 5050 |
|  | T2_C1000 | 0.9123 | **+4/−74, p = 1e-17** | — | 0.898 (**+0/−100, p = 2e-30**) | 8.7 | 0.1 | 0.0 | 57 | 1 | 26 | 5054 |
|  | T3_C1000 | 0.9778 | +3/−8, p = 0.227 | — | 0.999 (+0/−0, p = 1.000) | 0.1 | 0.5 | 1.6 | 76 | 1 | 137 | 5098 |
|  | T3b_C1000 | 0.9839 | +3/−2, p = 1.000 | — | 0.999 (+0/−0, p = 1.000) | 0.1 | 0.7 | 0.8 | 104 | 1 | 135 | 5095 |
|  | T3_ALL | 0.9778 | +3/−8, p = 0.227 | — | 0.999 (+0/−0, p = 1.000) | 0.1 | 0.5 | 1.6 | 76 | 1 | 137 | 5098 |
| musique (PHG) | T0 | 0.6596 | — | — | 0.890 | 11.0 | 15.6 | 7.4 | 98 | 17 | 227 | 5117 |
|  | T1_C1000 | 0.5663 | **+43/−136, p = 2e-12** | — | 0.575 (**+3/−316, p = 1e-89**) | 42.5 | 0.9 | 0.0 | 53 | 5 | 35 | 4786 |
|  | T2_C1000 | 0.4187 | **+22/−262, p = 3e-53** | — | 0.420 (**+0/−468, p = 3e-141**) | 57.8 | 0.3 | 0.0 | 57 | 6 | 35 | 4768 |
|  | T3_C1000 | 0.6476 | +41/−53, p = 0.256 | — | 0.894 (+4/−0, p = 0.125) | 10.5 | 12.2 | 12.4 | 133 | 12 | 242 | 4822 |
|  | T3b_C1000 | 0.6767 | **+41/−24, p = 0.046** | — | 0.892 (+2/−0, p = 0.500) | 10.8 | 13.3 | 8.2 | 132 | 12 | 232 | 4995 |
|  | T3_ALL | 0.6476 | +41/−53, p = 0.256 | — | 0.894 (+4/−0, p = 0.125) | 10.5 | 12.2 | 12.4 | 133 | 12 | 242 | 4822 |

Candidate sets and what the edge unit retrieves (`no endpoint in node top-100` = share of a channel's top-100 edges whose two endpoints
are both outside the query's node top-100 hits; composition from `edgegeo_why`):

| cache | N | STRUCT edges | C1000 candidate edges / query | top-100 edges with no endpoint in node top-100: mid | dir | trip | incident to the rank-0 hit (mid / dir) | incident to a node top-10 hit (mid / dir) |
|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) | 43234 | 124669 | 18672 | 21.5 % | 24.0 % | 15.7 % | 6.1 % / 6.3 % | 28.6 % / 30.3 % |
| metaqa_phg | 43234 | 124669 | 18672 | 21.5 % | 24.0 % | 15.7 % | 6.1 % / 6.3 % | 28.6 % / 30.3 % |
| squad (MtK) | 20233 | 744734 | 108892 | 0.4 % | 0.4 % | 0.2 % | — | — |
| squad_phg | 20233 | 744734 | 108892 | 0.4 % | 0.4 % | 0.2 % | — | — |
| musique (PHG) | 117534 | 2651280 | 85610 | 4.7 % | 1.2 % | 2.1 % | — | — |

metaqa per hop, `BASE (paired vs T0) · reach · UNREACHED pts · reached-weak pts · median worst gold rank among reached-failed`:

| cache | arm | hop1 (n = 326) | hop2 (n = 344) | hop3 (n = 332) |
|---|---|---|---|---|
| metaqa (MtK) | T0 | 0.917 · reach 0.945 · UNR 5.5 · weak 2.1 · rank 139 | 0.485 · reach 0.590 · UNR 39.8 · weak 9.0 · rank 146 | 0.527 · reach 0.611 · UNR 34.6 · weak 6.9 · rank 145 |
|  | T2_C1000 | 0.939 (+21/−14, p = 0.311) · reach 0.969 · UNR 3.1 · weak 3.1 · rank 66 | 0.474 (+14/−18, p = 0.597) · reach 0.529 · UNR 45.9 · weak 5.5 · rank 70 | 0.467 (**+3/−23, p = 9e-05**) · reach 0.491 · UNR 46.7 · weak 2.4 · rank 75 |
|  | T3_C1000 | 0.988 (**+23/−0, p = 2e-07**) · reach 1.000 · UNR 0.0 · weak 0.3 · rank 91 | 0.497 (+10/−6, p = 0.454) · reach 0.674 · UNR 31.4 · weak 11.6 · rank 162 | 0.533 (+3/−1, p = 0.625) · reach 0.642 · UNR 31.6 · weak 8.4 · rank 279 |
|  | T3b_C1000 | 0.945 (**+9/−0, p = 0.004**) · reach 1.000 · UNR 0.0 · weak 0.3 · rank 79 | 0.488 (+6/−5, p = 1.000) · reach 0.648 · UNR 34.0 · weak 10.5 · rank 151 | 0.524 (+1/−2, p = 1.000) · reach 0.626 · UNR 33.1 · weak 7.5 · rank 192 |
| metaqa_phg | T0 | 0.917 · reach 0.945 · UNR 5.5 · weak 2.1 · rank 138 | 0.529 · reach 0.628 · UNR 36.0 · weak 7.8 · rank 133 | 0.581 · reach 0.618 · UNR 34.6 · weak 3.3 · rank 139 |
|  | T2_C1000 | 0.948 (+23/−13, p = 0.132) · reach 0.966 · UNR 3.4 · weak 1.8 · rank 65 | 0.497 (+12/−23, p = 0.090) · reach 0.561 · UNR 42.7 · weak 6.4 · rank 75 | 0.539 (**+3/−17, p = 0.003**) · reach 0.563 · UNR 40.1 · weak 2.4 · rank 88 |
|  | T3_C1000 | 0.985 (**+22/−0, p = 5e-07**) · reach 1.000 · UNR 0.0 · weak 0.0 · rank 71 | 0.555 (**+12/−3, p = 0.035**) · reach 0.701 · UNR 28.8 · weak 10.5 · rank 181 | 0.587 (+4/−2, p = 0.688) · reach 0.645 · UNR 31.9 · weak 3.9 · rank 184 |
|  | T3b_C1000 | 0.951 (**+11/−0, p = 1e-03**) · reach 1.000 · UNR 0.0 · weak 0.0 · rank 70 | 0.552 (**+9/−1, p = 0.021**) · reach 0.671 · UNR 31.7 · weak 9.6 · rank 170 | 0.587 (+2/−0, p = 0.500) · reach 0.629 · UNR 33.4 · weak 3.9 · rank 173 |

**Verdict against the pre-registered rule: NOT_CONFIRMED** — but not for lack of a MetaQA gain. metaqa (MtK): BASE_ALL 0.6397 → 0.6687
(**+36/−7, p = 9e-06**; +2.9 points) passes the first two gates, and the hop2+hop3 pooled test (+13/−7, p = 0.263) fails the third; metaqa_phg: 0.6727 → 0.7056 (**+38/−5, p = 2e-07**; +3.3)
passes all three (pooled **+16/−5, p = 0.027**). Text corpora: squad (MtK) 0.9859 → 0.9788 (+4/−11, p = 0.118); squad_phg 0.9829 → 0.9778 (+3/−8, p = 0.227); musique (PHG) 0.6596 → 0.6476 (+41/−53, p = 0.256); squad (MtK) reach 0.999 → 0.999 (+0/−0, p = 1.000); squad_phg reach 0.999 → 0.999 (+0/−0, p = 1.000); musique (PHG) reach 0.890 → 0.894 (+4/−0, p = 0.125).
Against the ruling's plain criterion ("improve MetaQA materially") the answer is yes: +2.9 / +3.3 points on the two caches, both
p < 1e-5, reach +5.7 / +5.2 points with zero losses, and every hop-1 gain without a single hop-1 loss (MtK +23/−0, PHG +22/−0). What the
pre-registered rule adds — and what fails on MtK — is the requirement that the gain be multi-hop.

Information arms: an edge channel alone is far below the node chain (T1 mid-only 0.524 / 0.545, T2 dir-only 0.623 / 0.658: the edge
unit is a complement, not a replacement); the four-view triplet RRF as one channel (T3b) is weaker than the two separate edge channels
(0.649 / 0.694 vs 0.669 / 0.706); C1000 and ALL are identical to the fourth decimal on every metaqa arm — the top-100 edges of a channel
lie inside the served top-1000 pool anyway, so on graphs of this size the "scalable global implementation" is the same closed form over
the full edge list (three dot products per edge; no ANN needed for ≤ 10⁶ edges).

**Where the gain comes from (`_l1c_edgegeo_why.py`; T3 = T3_C1000; A1 = the closed UND_NONHUB full-vote table of §10–§11; `first edge
vote rank` = the best 0-based rank, over the mid and dir channels, of an edge that voted for a gold block that entered P50; buckets
rank0 / 1–4 / 5–19 / 20–99 / none):**

| cache | hop | n | ok: T0 / A1 / T3 | T3 gained / lost | T3 gains also A1 gains / T3-only | A1 gained / A1-only | gained: first edge vote rank | gained via an edge incident to the rank-0 hit | gained block was UNREACHED under T0 | T3 reached-failed / UNREACHED | reached-failed: worst gold block, best edge vote rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) | hop1 | 326 | 299 / 323 / 322 | 23 / 0 | 22 / 1 | 25 / 3 | 15 / 7 / 0 / 1 / 0 | 21 | 16 | 4 / 0 | 0 / 1 / 1 / 1 / 1 |
|  | hop2 | 344 | 167 / 166 / 171 | 10 / 6 | 8 / 2 | 9 / 1 | 6 / 2 / 1 / 1 / 0 | 7 | 5 | 61 / 108 | 1 / 0 / 2 / 32 / 26 |
|  | hop3 | 332 | 175 / 169 / 177 | 3 / 1 | 0 / 3 | 1 / 1 | 0 / 0 / 1 / 2 / 0 | 0 | 0 | 36 / 105 | 0 / 0 / 0 / 13 / 23 |
| metaqa_phg | hop1 | 326 | 299 / 323 / 321 | 22 / 0 | 22 / 0 | 25 / 3 | 15 / 7 / 0 / 0 / 0 | 22 | 15 | 5 / 0 | 1 / 0 / 3 / 0 / 1 |
|  | hop2 | 344 | 182 / 186 / 191 | 12 / 3 | 5 / 7 | 10 / 5 | 2 / 1 / 5 / 4 / 0 | 4 | 3 | 50 / 99 | 1 / 0 / 2 / 21 / 26 |
|  | hop3 | 332 | 193 / 187 / 195 | 4 / 2 | 1 / 3 | 1 / 0 | 0 / 0 / 3 / 1 / 0 | 0 | 0 | 19 / 106 | 0 / 0 / 1 / 10 / 8 |

The mechanism is the closed A1 table, selected by geometry. On hop 1, 22 of the 23 (MtK) and 22 of the 22 (PHG) queries T3 gains are
exactly the queries A1's undirected membership table gained in §10 — the in-neighbour blocks of the anchor ("the movies X starred in") —
and 21 / 22 of them enter through an edge incident to the rank-0 node hit; the entering gold block's first edge vote is at edge rank 0
in 15 / 15 and at rank 1–4 in 7 / 7 of the gained queries. T3 does not cover A1's hop-1 population fully (A1-only 3 / 3) and adds one
query of its own on MtK, none on PHG. On the multi-hop residual the edge unit behaves like every
1-hop mechanism before it: where T3 still fails on a reached hop-2 query, the worst gold block's best edge vote sits at rank ≥ 20 or does
not exist in 58 / 61 (MtK) and 47 / 50 (PHG) cases; on hop 3, 36 / 36 and 18 / 19. The answer block of a two-hop question is incident to
the intermediate entity, and the intermediate entity is neither a node hit nor — with 76–79 % of the top-100 edges touching a node
top-100 hit and only 6 % incident to the rank-0 hit — the endpoint of a highly-ranked *global* edge: a globally scored edge list is
dominated by whichever hit is most similar to the query, not by the anchor's neighbourhood. T3 lost 6 / 3 hop-2 and 1 / 2 hop-3 queries
by pushing a gold block from inside P50 to positions 51–126 with two more channels in the RRF; that dilution is what the text corpora
show as well: squad −0.7, squad_phg −0.5, musique −1.2 points, all ns, reach unchanged, and on text the edge unit retrieves nothing new
(≥ 95 % of its top-100 edges touch a node top-100 hit). The information arm T3b (four-view triplet RRF as one channel) is the only
positive text number (musique +41/−24, p = 0.046, +1.7; squad ns) and is weaker than T3 on metaqa; it is recorded, not carried.

**Closing.** `EDGE_GEOMETRY_VOTE` is **NOT_CONFIRMED** as a universal L1 channel: the multi-hop gate fails on MtK, and the MetaQA gain it
does deliver (+2.9 / +3.3, hop-1 only) is the same population the closed A1 table reached, bought with a small, non-significant text
cost. Per the ruling that followed this result, the signal is not discarded: the same closed-form edge geometry, applied to a
*handful of anchors' actual incident edges* instead of to every edge in the graph (hundreds to a few thousand transitions per query
instead of the 19k–109k pool edges or 0.12–2.65 million graph edges), becomes the transition ranker of the bounded dynamic repair stage µL3 (§14) — the right sort of signal at
the right scale and stage. No variant of T3 was run; DEV_B untouched.

## 14. Sixth ruling of the day: µL3 — a tiny, bounded, dynamic repair stage between L1 and full L3 (`_l1c_microl3.py`, `microl3_A_<cache>.json`, pre-registered in `PREREGISTRATION_MICROL3_DEV_A.json`; attribution `_l1c_microl3_why.py`, `microl3_why_A_<cache>.json`)

**Ruling recorded first.** *"We should stop treating the boundary as 'either pure L1 or expensive full L3.' There is room for a tiny
bounded dynamic stage whose only job is to repair the specific failure L1 has exposed … µL3 / micro-L3, not L1 … operationally it
can behave like an L1 booster."* Cascade: canonical L1 (Dense + SPLADE → static block voting → initial P50) → µL3 (tiny bounded
graph expansion → re-rank / repair P50) → only if still uncertain, full L3. *"µL3 should attack exactly the thing RELSIG could not
do: s → m → a. L1 retrieves s. µL3 is allowed to actually instantiate m."* Real graph as the primary mechanism, edge geometry (§13)
re-used as the cheap ranking signal over a handful of anchors' actual incident edges; frozen non-hub / degree-cap policy; no
training, no modified embeddings, no relation lists, no templates, same algorithm on every graph; *"don't ask µL3 to answer the
question. Its job … find blocks L1 should have selected."* One arm first, `MICRO_L3_H2`, compared against canonical L1 on DEV_A;
*"The hop2/hop3 result is the gate. We already know almost anything can improve MetaQA hop1. If hop2+hop3 move strongly, then we've
finally found the missing mechanism. If they don't, stop."* Latent global beam kept as an ablation inside the experiment, not as
the main mechanism.

**Design, exactly as pre-registered** (`PREREGISTRATION_MICROL3_DEV_A.json`, written before any run; module sha256
`19774617f5841fb02471498ad2af29ba9098f32e11aec06831724d569c99e6c5`; every constant is a frozen contract constant — nothing was
chosen for this experiment):

- **B0** = the frozen `SEED_K = 5` replay-cache seeds (the frozen RRF top-5 entity hits). Seeds are marked visited and do *not* vote
  in the repair channel (they already vote through the served channels).
- **Graph** = the actual directed STRUCT adjacency of the canonical corpus; **admissibility** = the frozen policy (cap = round(N / k)
  = 100 on every cache; hub := deg_undirected + 1 > cap; from v: all directed out-neighbours, in-neighbours only when v is a non-hub;
  visited targets excluded). **Depth exactly 2**; **beam = K_LOCK = 100** distinct targets per hop (≤ 205 visited nodes per query).
- **Transition ranking** (frozen node embeddings only, unit rows): `S_dst = cos(q, e_u)`, `S_dir = cos(q − e_v, e_u − e_v)`,
  `S_mid = cos(q, norm(e_v + e_u))` (the §13 closed forms), fused by the frozen RRF (K0 = 60); the beam is the first 100 distinct
  unvisited targets in fused order.
- **Repair channel** = the visited nodes in order (hop-1 beam, then hop-2 beam) voting through the served static membership table
  (own block + directed out-neighbour blocks) exactly like L1 hits: w_r = 1/(K0 + r), frozen rr(S) + rr(M); unvoted blocks unranked.
- **Fusion** = `_l1x90_core.fuse(C, base_rank, repair, "RRF")`: the frozen symmetric two-way RRF between the served fused block order
  and the repair order (unranked → weight 0), top-50. Output exactly P50.
- **Arms**: `L1` (served, asserted identical to the frozen BASE on every cache); `MICRO_L3_H2` (primary); `MICRO_L3_H1` = the same run
  truncated after hop 1 (a breakdown, not a selectable arm); `LATENT_H2/H1` = identical scoring / beam / repair / fusion but every
  non-visited node is a candidate target (the latent global beam), metaqa caches only, diagnostic.
- **Pre-registered rule.** Gate: hop2 + hop3 pooled ALL-gold@P50, `MICRO_L3_H2` vs `L1`, on **both** metaqa caches: McNemar
  gained > lost with p < 0.01 **and** pooled gain ≥ +5.0 points on each. Gate + ≥ +10.0 on each → `CANDIDATE_STRONG` ("the missing
  mechanism"); gate with [+5, +10) → `CANDIDATE` ("moves, not strongly" — report and ask for a ruling before any variant); else
  `NOT_CONFIRMED` → stop. Text constraint: a significant regression (lost > gained, p < 0.05) on any of squad / squad_phg / musique
  is a universality violation and blocks promotion regardless of the metaqa gate. Hop-1 gains carry no weight. Forbidden after
  seeing numbers: any change to beam, depth, seeds, scoring, fusion or admissibility; reading DEV_B.

### 14.1 Result — five caches, DEV_A (`microl3_A_<cache>.json`)

| cache | L1 | `MICRO_L3_H1` (after hop 1) | `MICRO_L3_H2` (primary) | H2 vs L1 gained / lost / p | reach L1 → H2 | UNREACHED pts L1 → H2 | reached-failed pts L1 → H2 |
|---|---|---|---|---|---|---|---|
| metaqa (MtK) | 0.6397 | 0.6986 (+62/−3, p = 2e-15) | **0.6886** (+4.9) | 51 / 2 / 3e-13 | 0.713 → 0.869 (+157/0) | 26.9 → 11.3 | 7.3 → 18.1 |
| metaqa_phg | 0.6727 | 0.7236 (+55/−4, p = 2e-12) | **0.7156** (+4.3) | 46 / 3 / 7e-11 | 0.728 → 0.875 (+148/0) | 25.6 → 10.9 | 5.5 → 16.0 |
| squad (MtK) | 0.9859 | 0.8780 (+1/−108) | **0.7792** (−20.7) | 2 / 207 / 5e-59 | 0.999 = 0.999 | 0.1 = 0.1 | 1.3 → 22.0 |
| squad_phg | 0.9829 | 0.8750 (0/−107) | **0.7742** (−20.9) | 0 / 207 / 1e-62 | 0.999 = 0.999 | 0.1 = 0.1 | 1.6 → 22.5 |
| musique (PHG) | 0.6596 | 0.5030 (+14/−170) | **0.5010** (−15.9) | 15 / 173 / 3e-35 | 0.890 → 0.919 (+29/0) | 11.0 → 8.1 | 23.0 → 41.8 |

MetaQA per hop (`MICRO_L3_H2` vs `L1`; the pre-registered gate is the pooled row):

| hop | n | L1 MtK → H2 | gained / lost / p | L1 PHG → H2 | gained / lost / p | initial UNREACHED → reached after hop 1 → after hop 2 → ALL@P50 after hop 2 (MtK / PHG) |
|---|---|---|---|---|---|---|
| hop1 | 326 | 0.9172 → 0.9448 | 9 / 0 / 0.004 | 0.9172 → 0.9448 | 10 / 1 / 0.012 | 18 → 18 → 18 → 4 / 18 → 18 → 18 → 6 |
| hop2 | 344 | 0.4855 → 0.5872 | 36 / 1 / 6e-10 | 0.5291 → 0.6221 | 33 / 1 / 4e-9 | 137 → 79 → 100 → 17 / 124 → 69 → 89 → 15 |
| hop3 | 332 | 0.5271 → 0.5422 | 6 / 1 / 0.125 | 0.5813 → 0.5873 | 3 / 1 / 0.625 | 115 → 3 → 39 → 0 / 115 → 0 → 41 → 0 |
| **hop2 + hop3 pooled** | 676 | **0.5059 → 0.5651 (+5.9)** | 42 / 2 / 1e-10 | **0.5547 → 0.6050 (+5.0)** | 36 / 2 / 5e-9 | 252 → 82 → 139 → 17 / 239 → 69 → 130 → 15 |

Cost (per DEV_A query, real per-query gathers from the fp16 memmap, single thread, contended host):

| cache | transitions scored hop 1 / hop 2 / total (mean; p95) | nodes visited | beam wall-clock ms mean / p50 / p95 | repair + fusion ms | queries whose seeds are all hubs |
|---|---|---|---|---|---|
| metaqa, metaqa_phg | 37.8 / 334.9 / 372.7 (p95 760) | 128.0 | 21.9 / 17.8 / 52.1 | 5.6 | 0 / 1002 |
| squad, squad_phg | 238.2 / 3436.7 / 3675.0 (p95 7254) | 181.9 | 87.8 / 78.9 / 206.8 | 5.4 | 58 / 992 |
| musique | 185.2 / 3198.0 / 3383.3 (p95 6493) | 180.3 | 91.2 / 77.6 / 235.2 | 7.6 | 16 / 996 |

**Verdict by the pre-registered rule.** The gate passes on both metaqa caches — pooled hop2 + hop3 +42/−2 (p = 1e-10, +5.9 points)
on MtK and +36/−2 (p = 5e-9, +5.0 points) on PHG — but neither reaches the STRONG threshold (+10), so the metaqa outcome is
`CANDIDATE` ("moves, not strongly"). The text constraint is violated on every text cache (squad −20.7, squad_phg −20.9, musique
−15.9 points, p ≤ 3e-35), which blocks promotion regardless of the metaqa gate. `MICRO_L3_H2` as pre-registered is therefore
**NOT PROMOTABLE**; no variant was run; DEV_B untouched. The hop-2 movement is real and significant, hop 3 does not move
(+6/−1, +3/−1), and the attribution below shows *which half* of the stage produced each number.

### 14.2 Attribution (`_l1c_microl3_why.py`, `microl3_why_A_{metaqa,metaqa_phg,squad,musique}.json`; same beam re-run, no new arm)

**The dynamic half works — at the node level.** On metaqa (both caches: the beam does not depend on the partition), the answer
entity itself is visited for 321/326 hop-1 questions (309 by the hop-1 beam), **298/344 hop-2 questions (87 %; 277 of them by the
hop-2 beam — s → m → a instantiated on the actual graph)**, and 214/332 hop-3 questions (197 by the hop-1 beam: hop-3 answer sets
average 14.7 entities and are partly adjacent to a seed; all answers visited for only 38/332). This costs 373 transitions and 22 ms
per query. Reach follows: initial UNREACHED hop-2 queries 137 → 100 reached after hop 2 (PHG 124 → 89); hop-3 115 → 39 / 41.

**The static half — the block vote and the fusion — is what fails, on every cache.**

- *MetaQA, reached-but-failed (hop 2: 101 MtK / 91 PHG queries; hop 3: 62 / 51).* The gold block *is* voted, and early: its first
  repair vote comes from a hop-1 beam node at ranks 0–49 in 71/101 (hop 2, MtK), i.e. the intermediate m carries P(a) in its served
  membership. But the repair order is a **mass** ranking (rr(S) + rr(M)): the top repair block collects ≈ 62 votes (89 on hop 3)
  because the beam clusters in a few blocks (the repair channel saturates: 49.6 blocks voted after hop 1 → 201 of 432 after hop 2;
  hop-2 beam nodes carry 5.1 blocks each vs 2.5 for an L1 hit), while the gold block collects ≈ 1.9. The gold block therefore sits at
  repair position ≥ 50 in 62/101 and, being unranked in the served order in 85/101, lands at fused position ≥ 60 in 93/101 —
  it is found and voted, then out-massed. The same mechanism explains H1 > H2 on BASE_ALL (0.6986 vs 0.6886; 0.7236 vs 0.7156):
  adding the hop-2 votes moves 10 (MtK) / 9 (PHG) already-correct hop-2 gold blocks from fused positions 24–49 to 50–80, and the
  hop-2 discoveries themselves enter the repair list at ranks 100–199 (weights ≤ 1/160). Across the 202 hop-2 queries correct under H2, the
  worst gold block received its first repair vote from a hop-1 beam node at rank 0–9 in 190 — the gold block enters through the
  served membership of the *first* intermediate found, not through the visit of the answer node itself.
- *Text (squad 207/207 losses, squad_phg 207, musique 173).* The lost gold block sat at served position 0–9 in 168/207 (squad) and
  10–49 in 150/173 (musique) and is **unranked in the repair channel in 204/207 and 144/173**: the gold node is itself a seed
  (890/992 squad, 857/996 musique) and seeds do not vote in the repair, while the served hits' blocks are not re-voted either. Under
  the symmetric RRF a repair-only block at position r carries 1/(60 + r) — the same weight as a served block at position r — so the
  86–243 blocks of the beam's neighbourhood displace 11–14 served P50 blocks per query, the gold one among them. This is an artefact
  of the *instantiation* of "repair" (a separate channel with equal authority and zero weight for what it does not vote), not of
  the beam: reach is unchanged on squad (0.999) and rises on musique (+29/0).

**What the experiment establishes.** (i) The bounded two-hop real-edge beam *does* instantiate the intermediate and the answer
entity for the large majority of hop-2 questions at ≈ 22 ms per query — the discovery mechanism the ruling asked for exists and is
cheap. (ii) The translation of that discovery into P50 through the frozen L1 vote (existential membership summed as mass) and the
frozen symmetric fusion loses most of it on MetaQA (157 newly reached → 21 newly correct) and destroys the text caches. (iii)
Hop 3 needs two intermediates; a depth-2 beam cannot supply them (hop-3 gains 6/3, ns). Under the pre-registered rule this is
`CANDIDATE` on MetaQA with a universality violation — a ruling is required before anything further; the three things a ruling
could change are separate pre-registrations, not knobs, and all three are post-hoc on a population (DEV_A) that has now been read
by this lane: **(a)** existential repair evidence — a block's repair rank = the best beam rank of any visited node that votes for
it (M-only / first-vote order; "membership is an existential routing statement", §11 ruling) instead of rr(S) + rr(M); **(b)**
asymmetric fusion — the served order keeps authority and the repair may only fill or replace the weakest slots (the ruling's own
words, "replace weakest blocks in P50"; the frozen `HOLD` / `CORE` rules of `_l1x90_core.fuse` are the existing forms); **(c)** the
served hits keep voting (repair list = served top-K_LOCK hits followed by the beam, a continuation rather than a second channel).
None was run. Any confirmation would need a population this lane has not read (WebQSP, or DEV_B by ruling).

**Latent global beam (diagnostic).** Recorded in §14.4 (metaqa MtK complete; metaqa_phg running).

### 14.3 Ruling #2 — asymmetric repair fusion (`_l1c_microl3_asym.py`, `microl3_asym_A_<cache>.json`, pre-registered in `PREREGISTRATION_MICROL3_ASYM_DEV_A.json`, STATUS = POSTHOC_MECHANISM_TEST)

**Ruling recorded first.** *"I would not stop yet, because the experiment separated the problem very cleanly: the dynamic mechanism
works; the fusion/repair interface is wrong; and the cost is only ~22 ms/query on MetaQA … My ruling: run #2 only — asymmetric
repair fusion. Do not run all three. Freeze the beam, transition scoring, seeds, depth, hub policy, embeddings, everything else
exactly as in `MICRO_L3_H2`. Change only the interface between the served ranking and repair ranking: the served order remains
authoritative; µL3 can repair only the weak tail of P50 … Use the already-existing frozen `HOLD/CORE` rule in `fuse`. Do not
invent a new hold size from these results … pre-register it as exploratory, not confirmation … continuation gate: MetaQA hop2 must
retain a substantial fraction of H2's gain, while none of the three text caches may show significant loss versus canonical L1.
Don't invent a numerical MetaQA target … just report retention. If this fails, close µL3 H2 as an L1-repair mechanism. Don't then
proceed to #1 and #3."* Also ruled: hop 3 has not falsified the mechanism (a depth-2 stage instantiates only through m₂ of
s → m₁ → m₂ → a; hop2 ≈ +10 / hop3 ≈ 0 is what a correctly functioning depth-2 process predicts); do not run H3 before the H2
evidence can be consumed; the latent diagnostic may finish but must not modify this arm.

**What was run.** The pinned µL3 beam (`_l1c_microl3.py`, sha `19774617…`, asserted) is re-executed verbatim — seeds, adjacency,
hub policy, depth 2, beam 100, transition scoring, repair channel — and the symmetric arm rebuilt from it is asserted equal to the
§14.1 record on every cache (identical beam output). The only change is the fusion: **`H2_HOLD`** (primary) = the pre-existing,
parameter-free `HOLD` rule of `_l1x90_core.fuse` (L1_P90_EXPLOIT, 2026-09-14): incumbents = the served P50; challengers = repair-
ranked blocks that are not incumbents, in repair order; the worst incumbent (largest served position) is replaced by the next
challenger iff the challenger's repair position is smaller than that incumbent's served position; stop at the first challenger that
fails; exactly 50. Because the j-th challenger has repair position ≥ j and must beat served position 49 − j, HOLD can exchange at
most 25 slots and never touches served positions 0–24; there is no hold size to choose. `CORE(B)` with B ∈ {6, 12, 25} — the values
the x90 lane explored and never chose — are recorded as information only (protected served top-(50 − B), symmetric competition for
the B remaining slots); the pre-registration forbids selecting a B. Gate: no significant loss vs L1 (lost > gained, p < 0.05) on
any of squad / squad_phg / musique, and a significant hop-2 gain (p < 0.01) on both metaqa caches; retention reported without a
threshold.

| cache | L1 | H2_RRF (§14.1) | **H2_HOLD** (primary) | HOLD vs L1 gained/lost/p | HOLD vs H2_RRF | served P50 blocks replaced / query (HOLD; RRF) | significant loss vs L1? |
|---|---|---|---|---|---|---|---|
| metaqa (MtK) | 0.6397 | 0.6886 (+4.9) | **0.6816** (+4.2) | 53/11/1e-7 | 15/22/0.324 | 15.6; 16.6 | no |
| metaqa_phg | 0.6727 | 0.7156 (+4.3) | **0.7086** (+3.6) | 43/7/2e-7 | 14/21/0.311 | 14.1; 15.1 | no |
| squad (MtK) | 0.9859 | 0.7792 (−20.7) | **0.9778** (−0.8) | 0/8/0.008 | 199/2/1e-56 | 10.6; 11.3 | **yes** |
| squad_phg | 0.9829 | 0.7742 (−20.9) | **0.9778** (−0.5) | 0/5/0.062 | 204/2/4e-58 | 10.2; 10.4 | no |
| musique (PHG) | 0.6596 | 0.5010 (−15.9) | **0.6225** (−3.7) | 11/48/1e-6 | 136/15/1e-25 | 12.6; 11.6 | **yes** |

MetaQA per hop, `H2_HOLD` vs `L1` (retention = (HOLD − L1) / (H2_RRF − L1) on ALL-gold@P50):

| hop | L1 MtK → HOLD | gained/lost/p | retention MtK | L1 PHG → HOLD | gained/lost/p | retention PHG |
|---|---|---|---|---|---|---|
| hop1 | 0.9172 → 0.9479 | 11/1/0.006 | 1.11 | 0.9172 → 0.9417 | 9/1/0.021 | 0.89 |
| hop2 | 0.4855 → 0.5698 | 35/6/5e-6 | 0.83 | 0.5291 → 0.6105 | 31/3/8e-7 | 0.88 |
| hop3 | 0.5271 → 0.5361 | 7/4/0.549 | 0.60 | 0.5813 → 0.5813 | 3/3/1.000 | 0.00 |
| **hop2 + hop3 pooled** | **0.5059 → 0.5533** | 42/10/9e-6 | 0.80 | **0.5547 → 0.5962** | 34/6/8e-6 | 0.82 |

Information arms (fixed protected core, `CORE(B)`, B from the L1_P90_EXPLOIT exploration; not selectable):

| cache | CORE6 | vs L1 | CORE12 | vs L1 | CORE25 | vs L1 | hop-2 retention CORE6 / 12 / 25 |
|---|---|---|---|---|---|---|---|
| metaqa (MtK) | 0.6517 | 12/0/5e-4 | 0.6627 | 23/0/2e-7 | 0.6796 | 41/1/2e-11 | 0.29 / 0.51 / 0.77 |
| metaqa_phg | 0.6866 | 14/0/1e-4 | 0.6936 | 21/0/1e-6 | 0.7056 | 35/2/1e-8 | 0.31 / 0.50 / 0.75 |
| squad (MtK) | 0.9819 | 0/4/0.125 | 0.9768 | 0/9/0.004 **sig. loss** | 0.9758 | 2/12/0.013 **sig. loss** | – |
| squad_phg | 0.9808 | 0/2/0.500 | 0.9788 | 0/4/0.125 | 0.9738 | 0/9/0.004 **sig. loss** | – |
| musique (PHG) | 0.6486 | 10/21/0.071 | 0.6406 | 13/32/0.007 **sig. loss** | 0.6014 | 15/73/3e-10 **sig. loss** | – |

**Verdict by the pre-registered rule: FAIL → `MICRO_L3_H2 = CLOSED as an L1-repair mechanism`.** The MetaQA half of the gate holds —
`H2_HOLD` retains 0.83 / 0.88 of the hop-2 gain (+35/−6, p = 5e-6; +31/−3, p = 8e-7) and 0.80 / 0.82 of the pooled hop2+hop3
gain — but the text half fails: squad 0/−8 (p = 0.008) and musique +11/−48 (p = 1e-6) are significant losses against canonical L1
(squad_phg 0/−5, p = 0.06). Per the ruling, candidates #1 (existential repair evidence) and #3 (served hits keep voting) are **not
run**, and no CORE size may be chosen: the information rows show the same trade-off at every fixed core — the tightest protected
core (B = 6) still loses on text (squad 0/−4, musique +10/−21, both ns) while retaining only 0.29 / 0.31 of the hop-2 gain, and
every wider core loses significantly on musique.

**Why the asymmetric interface cannot make the beam evidence safe (`served_tail_gold_share_A.json`, a static property of the served
ranking, no beam involved).** HOLD's exchange region is served positions 25–49, and it is entered whenever a non-incumbent appears within the first 49
repair positions — which on every cache happens for almost every query (10–16 served blocks replaced per query, only
32–33 squad queries with no replacement). Whether that costs anything depends on how often canonical L1's correct answers *rely* on
those tail slots: the worst gold block sits at served position 25–49 in 2.5–2.8 % of L1-correct queries on metaqa, metaqa_phg,
squad and squad_phg (18 / 18 / 24 / 27 queries), but in **19.8 % on musique (130 of 657; 54 % at positions 10–49)** — musique's
served ranking is weak, so its P50 tail is not junk, it is where a fifth of its correct answers live. HOLD evicted 8 of squad's 24
tail-dependent answers and 48 of musique's 130; CORE6 evicted 4 and 21; the symmetric RRF evicted 207 and 173 because it could
reach the head as well. On the KB caches the same exchange is nearly free (2.7–2.8 %) and the challengers are gold-bearing (the beam's
hop-1 intermediates carry P(a)), which is exactly the KB / text asymmetry the "same algorithm everywhere" constraint forbids
exploiting. The beam's repair order on a text graph is not gold-bearing at any rank the interface is allowed to use: on musique
the hop-2 beam does reach 29 previously unreached gold blocks (+29/0 reach) and gains 10–15 queries under every rule, but its
challengers displace more golds than they bring at every core size (HOLD +11/−48, CORE6 +10/−21, CORE12 +13/−32, CORE25 +15/−73).

**Closing.** The two halves are now separated and measured. (i) A training-free, query-conditioned, depth-2 beam over the actual
STRUCT graph instantiates the answer entity for 87 % of MetaQA hop-2 questions (298/344 visited; 277 of them first reached by the hop-2 step) at 373 transitions
and ≈ 22 ms per query, and hop 3 behaves exactly as a depth-2 process should (reach +12 points; ALL@P50 +6/−1, ns). (ii) Neither
the symmetric fusion (§14.1) nor the pre-existing asymmetric interfaces (§14.3) consume that evidence without a significant text
loss, because the served tail carries a fifth of musique's correct answers and the beam's block votes on text graphs are not
gold-bearing at the ranks the interface exposes. `MICRO_L3_H2` is closed as an L1-repair mechanism by the pre-registered rule and
the ruling that set it; the discovery result stands as the finding to carry forward (retrieval and selection are separable; the
selection side is where the remaining MetaQA multi-hop mass is lost). #1 and #3 unrun; no CORE size chosen; DEV_B untouched;
nothing under `data/` written.

### 14.4 Latent global beam — diagnostic ablation (`microl3_A_<cache>__latent.json`; metaqa caches only; not used to modify any arm)

Identical scoring, beam width, depth, repair channel and symmetric fusion as `MICRO_L3_H2`, but the candidate targets of a transition
are every non-visited node (the latent global beam) instead of the actual admissible neighbours: 216k transitions at hop 1 and
4.31 M at hop 2 per query, 7.97 s per query (vs 373 transitions and 22 ms). On the metaqa MtK cache `LATENT_H2` = 0.5649 vs L1
0.6397 (+4/−79, p = 4e-19; hop1 +2/−63, hop2 +1/−5, hop3 +1/−11; pooled hop2+hop3 +2/−16, p = 0.001); reach 0.713 → 0.754
(+41/0) against 0.869 (+157/0) for the real-edge beam; initial-UNREACHED 270 → 41 reached → 0 correct (real edges: 157 → 21).
The pre-registered secondary expectation holds: the actual topology carries the information the latent offsets cannot — the same
geometry over all nodes finds a quarter of the gold blocks the real edges find, at ≈ 360× the cost, and destroys hop 1.
The metaqa_phg run of the same diagnostic (`microl3_A_metaqa_phg__latent.json`, completed 2026-09-15 14:24, 9,483 s for 1,002
queries; 9.40 s per query, p95 12.0 s, on the contended host) says the same: `LATENT_H2` = 0.5988 vs L1 0.6727 (+10/−84,
p = 1e-15; hop1 +1/−63, hop2 +8/−12 (p = 0.50), hop3 +1/−9; pooled hop2+hop3 +9/−21, p = 0.043); reach 0.728 → 0.757 (+29/0)
against 0.875 (+148/0) for the real-edge beam; initial-UNREACHED 257 → 29 reached → 0 correct (real edges: 148 → 21). Both
metaqa caches agree: the latent global beam finds a fifth to a quarter of what the real edges find, at ≈ 360–430× the cost, and
destroys hop 1. Diagnostic only; no arm was modified.

## 15. Seventh ruling (2026-09-15): H2 as a SAFE candidate generator — `H2_AS_SAFE_CANDIDATE_GENERATOR` (`_l1c_h2safe.py`, `h2safe_A_<cache>.json`, `h2safe_A_SUMMARY.json`, pre-registered in `PREREGISTRATION_H2_SAFE_DEV_A.json`, STATUS = POSTHOC_MECHANISM_TEST; attribution `_l1c_h2safe_why.py`, `h2safe_why_A_<cache>.json`; six-slot ceiling `_l1c_h2safe_ceiling.py`, `h2safe_ceiling_A_<cache>.json`)

**Ruling recorded first.** *"I would not conclude that L1 is stuck at ~0.67. The latest failure is telling us something narrower:
dynamic structure is good at generating candidates, but we gave those candidates too much authority. And we already have a
mechanism whose whole purpose is to solve exactly that problem: SAFE's structure swap … Keep the canonical L1 completely unchanged
through its BASE ranking. Then: Dense + SPLADE → canonical block ranking → top 44 locked → SAFE candidate generation + H2 discovered
nodes/blocks → existing SAFE selection machinery → choose exactly six → P50. The crucial rule: H2 does not obtain its own RRF
channel. Its output is merely another candidate source for the existing structure swap … I'd preserve the H2 first-visit / beam
order, but use it only to order H2 candidates before feeding them into SAFE. So: H2 candidate block rank = best beam rank of any
node in that block … all three [STATIC STRUCT, H2 STRUCT, RETRIEVAL] must pass through the same six-slot SAFE bottleneck … Same
pipeline everywhere … And if SAFE still admits bad H2 challengers? Then that, rather than H2, is where I'd improve L1 … potentially
solvable parameter-free using agreement … But I would not jump there yet. First test whether simply adding H2 candidates to the
existing SAFE machinery already works … Not `MICRO_L3_H2` repair. That remains closed. Instead open a distinct hypothesis:
`H2_AS_SAFE_CANDIDATE_GENERATOR` … evaluate it against canonical SAFE, not BASE … on all five DEV_A caches. If MetaQA hop2
increases substantially without significant text losses, then we have found a genuinely better L1-ish retrieval stage. If it still
damages text, then the next problem isn't retrieval anymore — we need a better universal SAFE admission criterion, probably based on
cross-signal agreement … structure should propose; it should not be allowed to overthrow the semantic ranking by itself."*

**What was run (pre-registered before any run, `PREREGISTRATION_H2_SAFE_DEV_A.json`).** Canonical SAFE is the frozen
B6_S4_F6_Ms64_Mr32 selector executed through its own code (`_l1ps_router.build_cache` + `_l1kb_core.contexts` / `f6_select`,
unchanged): protected core = served positions 0–43; boundary = 44–49; candidates = boundary ∪ structural challengers (S4 order of
the frozen `expand_dir` residual beam, M_struct = 64 nodes) ∪ retrieval challengers (the RRF-200 continuation, M_ret = 32 nodes);
F6 score = 1/(60 + cpos) + 1/(60 + spos) + 1/(60 + rpos), an absent channel contributing 0; ties by served rank then block id;
exactly six admitted; exactly 50 served. Its recomputed indicator equals the frozen replay records' `_ind_SAFE` on every DEV_A row
of all five caches (asserted; `results/L1_CANONICAL/L1_REPLAY_{metaqa,squad}.json`, `results/L1_LOWMEM/L1_REPLAY_{…}.json`). The
H2 beam is the pinned µL3 module (sha `19774617…`, asserted), re-executed verbatim, and the symmetric §14.1 arm rebuilt from it
matches the §14.1 record on every cache (identical beam output). The only change: the beam's visited nodes in first-visit order
(hop-1 beam in fused order, then hop-2 beam; seeds excluded as in the frozen vote) are mapped to their own blocks `hard[v]` and
de-duplicated → an H2 block list whose position is the best beam rank of any node in the block (`h2pos`); the blocks not in the
served P50 are appended to the challenger list, and the structural position used by the unchanged F6 score becomes
`spos'[b] = min(spos[b], h2pos[b])` — a block's structural rank is its best rank across the two structural generators (incumbents
receive the same credit; F6's symmetric-evidence rule). No fourth channel, no weight, no cap, no hold size, no agreement rule, no
dataset branch; the alternative merge rules (append after the static list; RRF inside the structural channel; a fourth channel)
were listed as not run and remain not run. The H2 list is the largest candidate source everywhere (metaqa 82.7 blocks per query vs
49.3 static and 24.2 retrieval; squad 30.8 / 22.2 / 13.0; musique 52.0 / 38.1 / 18.5). Gate (pre-registered): no significant loss
vs canonical SAFE (lost > gained, p < 0.05) on any of squad / squad_phg / musique AND hop-2 ALL-gold@P50 gained > lost with
p < 0.01 vs SAFE on both metaqa caches; "substantial" reported as points and as the fraction of §14.1's `MICRO_L3_H2` hop-2 gain
over L1 that SAFE_H2 recovers over SAFE, no numeric target.

| cache | L1 | SAFE (canonical) | **SAFE_H2** | SAFE_H2 vs SAFE gained/lost/p | SAFE_H2 vs L1 | SAFE vs L1 | H2-credited admissions / query (queries with ≥ 1) | H2-only admissions / query | churn SAFE; SAFE_H2 | gold admitted / evicted vs served P50: SAFE; SAFE_H2 | significant loss vs SAFE? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) | 0.6397 | 0.6407 (+0.1) | **0.6427** (+0.2) | 3/1/0.625 | 5/2/0.453 | 3/2/1.000 | 1.85 (852 / 1002) | 0.44 | 4.59; 4.57 | 63 / 66; 79 / 68 | no |
| metaqa_phg | 0.6727 | 0.6747 (+0.2) | **0.6796** (+0.5) | 5/0/0.062 | 8/1/0.039 | 4/2/0.688 | 1.93 (854 / 1002) | 0.41 | 4.43; 4.53 | 64 / 65; 77 / 59 | no |
| squad (MtK) | 0.9859 | 0.9919 (+0.6) | **0.9909** (−0.1) | 1/2/1.000 | 5/0/0.062 | 6/0/0.031 | 1.34 (649 / 992) | 1.06 | 3.74; 4.12 | 6 / 0; 5 / 0 | no |
| squad_phg | 0.9829 | 0.9889 (+0.6) | **0.9889** (±0.0) | 1/1/1.000 | 6/0/0.031 | 6/0/0.031 | 1.39 (656 / 992) | 0.97 | 3.88; 4.22 | 6 / 0; 6 / 0 | no |
| musique (PHG) | 0.6596 | 0.7118 (+5.2) | **0.7129** (+0.1) | 7/6/1.000 | 55/2/2e-14 | 56/4/9e-13 | 1.38 (741 / 996) | 0.55 | 4.72; 4.55 | 93 / 11; 88 / 7 | no |

MetaQA per hop (ALL-gold@P50; recovery = (SAFE_H2 − SAFE) / (H2_RRF − L1) with H2_RRF from §14.1):

| hop | MtK: L1 → SAFE → SAFE_H2 | SAFE_H2 vs SAFE | SAFE_H2 vs L1 | recovery MtK | PHG: L1 → SAFE → SAFE_H2 | SAFE_H2 vs SAFE | SAFE_H2 vs L1 | recovery PHG |
|---|---|---|---|---|---|---|---|---|
| hop1 | 0.9172 → 0.9172 → 0.9202 | 1/0/1.000 | 1/0/1.000 | 0.111 (0.3 of 2.8 pts) | 0.9172 → 0.9172 → 0.9202 | 1/0/1.000 | 1/0/1.000 | 0.111 (0.3 of 2.8 pts) |
| hop2 | 0.4855 → 0.4942 → 0.4971 | 2/1/1.000 | 4/0/0.125 | 0.029 (0.3 of 10.2 pts) | 0.5291 → 0.5291 → 0.5407 | 4/0/0.125 | 4/0/0.125 | 0.125 (1.2 of 9.3 pts) |
| hop3 | 0.5271 → 0.5211 → 0.5211 | 0/0/1.000 | 0/2/0.500 | 0.000 (0.0 of 1.5 pts) | 0.5813 → 0.5873 → 0.5873 | 0/0/1.000 | 3/1/0.625 | 0.000 (0.0 of 0.6 pts) |
| **hop2 + hop3 pooled** | **0.5059 → 0.5074 → 0.5089** | 2/1/1.000 | 4/2/0.688 | 0.025 (0.1 of 5.9 pts) | **0.5547 → 0.5577 → 0.5636** | 4/0/0.125 | 7/1/0.070 | 0.118 (0.6 of 5.0 pts) |
| all (ANY-gold: MtK 0.9132 → 0.9152 → 0.9251; PHG 0.9261 → 0.9251 → 0.9331) | 0.6397 → 0.6407 → 0.6427 | 3/1/0.625 | 5/2/0.453 | 0.041 (0.2 of 4.9 pts) | 0.6727 → 0.6747 → 0.6796 | 5/0/0.062 | 8/1/0.039 | 0.116 (0.5 of 4.3 pts) |

**Verdict by the pre-registered rule: FAIL — on the branch the ruling did not anticipate.** The text half of the gate holds
everywhere: SAFE_H2 vs SAFE is +1/−2 (p = 1.0) on squad, +1/−1 on squad_phg and +7/−6 (p = 1.0) on musique, with SAFE's own
musique gain intact (SAFE_H2 vs L1 +55/−2, p = 2e-14). The MetaQA half does not: hop 2 moves by +2/−1 (p = 1.0) on the MtK cache
and +4/0 (p = 0.125) on PHG — 0.3 and 1.2 points, 0.029 and 0.125 of the hop-2 gain the symmetric fusion had shown (pooled
hop2+hop3 0.025 / 0.118; overall 0.041 / 0.116). ANY-gold@P50 rises by ≈ 1 point on both metaqa caches (0.9152 → 0.9251;
0.9251 → 0.9331): the H2 source does put *a* gold block into the six slots for 8–10 more queries, not *all* of them. H2-credited
admissions occur in 65–85 % of queries on every cache (1.3–1.9 per query), so the source is used; every newly covered query on
every cache enters through an H2-credited gold block (3 / 5 / 1 / 1 / 7 queries) and every newly lost one is displaced by H2-credited
blocks (1 / 0 / 2 / 1 / 6 queries; static or retrieval challengers displace none). The mechanism behaves exactly as designed and is
universally safe — and it is inert on the one thing it was built for. In the ruling's own branch structure the outcome is neither
"a genuinely better L1-ish retrieval stage" nor "it still damages text": it is *safe but inert*, and the attribution below says why
without running any arm.

**Candidate-generator recall (recorded with the arm; a static property of the three candidate lists).**

| cache · group | queries with a gold block outside the served P50 | worst missing gold block proposed by: static structural / retrieval / H2 / none | H2 first-visit position of it: 0–5 / 6–24 / 25–49 / 50+ | all missing gold blocks proposed by H2 | static-structural rank of it: 0–5 / 6–24 / 25–49 / 50+ |
|---|---|---|---|---|---|
| metaqa (MtK) · hop1 | 27 | 7 / 0 / 26 / 1 | 8 / 15 / 1 / 2 | 25 | 0 / 1 / 6 / 0 |
| metaqa (MtK) · hop2 | 177 | 21 / 2 / 100 / 73 | 3 / 39 / 31 / 27 | 96 | 1 / 8 / 9 / 3 |
| metaqa (MtK) · hop3 | 157 | 5 / 1 / 11 / 142 | 0 / 1 / 7 / 3 | 11 | 0 / 0 / 4 / 1 |
| metaqa_phg · hop1 | 27 | 4 / 1 / 26 / 1 | 8 / 16 / 1 / 1 | 26 | 0 / 1 / 3 / 0 |
| metaqa_phg · hop2 | 162 | 10 / 2 / 91 / 71 | 2 / 31 / 31 / 27 | 89 | 3 / 1 / 6 / 0 |
| metaqa_phg · hop3 | 139 | 4 / 2 / 9 / 128 | 0 / 1 / 6 / 2 | 9 | 1 / 1 / 1 / 1 |
| squad (MtK) · all | 14 | 0 / 7 / 2 / 5 | 0 / 1 / 1 / 0 | 2 | 0 / 0 / 0 / 0 |
| squad_phg · all | 17 | 3 / 8 / 5 / 5 | 0 / 1 / 2 / 2 | 5 | 0 / 2 / 1 / 0 |
| musique (PHG) · all | 339 | 57 / 75 / 71 / 184 | 3 / 29 / 31 / 8 | 66 | 6 / 33 / 18 / 0 |

**Why the frozen admission cannot use the proposals — three measurements (`h2safe_why_A_{metaqa,metaqa_phg}.json`,
`h2safe_ceiling_A_<cache>.json`; the same six-slot competition re-scored, nothing selected, no parameter).**

*(i) The generator's recall is not the bottleneck.* On metaqa hop 2 the H2 list proposes the worst missing gold block for 100 of
the 177 DEV_A failures (91 / 162 on PHG), where the frozen static expansion proposes it for 21 (10) and the retrieval continuation
for 2 (2); it proposes *all* missing gold blocks for 96 (89). On hop 3 it proposes 11 (9) of 157 (139) — the depth-2 signature of
§14.3, not a failure. On the text caches the picture is the one the ruling predicted: squad's 14 failures are mostly
retrieval-proposable (7) and H2 proposes 2; musique's 339 are split (static 57, retrieval 75, H2 71, none 184).

*(ii) The six-slot competition is where the proposals die.*

| metaqa hop-2 failures whose missing gold blocks the H2 source proposes in full | MtK | PHG |
|---|---|---|
| queries (of the hop-2 failures) | 96 (of 177) | 89 (of 162) |
| missing gold blocks per failing query (mean) | 6.08 | 6.46 |
| slot rank of the worst missing gold block in the unchanged six-slot competition: 1–6 / 7–12 / 13–24 / 25+ | 3 / 3 / 12 / 78 | 4 / 4 / 11 / 70 |
| its channels: canonical top-200 / static structural / retrieval / **H2 only** | 27 / 23 / 0 / **53** | 28 / 13 / 2 / **53** |
| first reached by the beam at hop 1 / hop 2 | 17 / 79 | 14 / 75 |
| within-hop beam rank of its best node: 0–5 / 6–24 / 25–49 / 50+ | 19 / 39 / 17 / 21 | 17 / 35 / 14 / 23 |
| H2 first-visit position (the rank SAFE sees): 0–5 / 6–24 / 25–49 / 50+ | 4 / 44 / 24 / 24 | 4 / 32 / 33 / 20 |
| hop-1 blocks preceding it in the H2 order (mean) | 15.7 | 16.8 |
| what fills the six slots instead (slot counts): incumbent kept / static structural / retrieval / H2-credited hop-1 block / H2-credited hop-2 block; gold among them | 153 / 199 / 64 / 98 / 62; 19 | 143 / 162 / 77 / 93 / 59; 19 |
| F6-score gap, sixth slot − worst missing gold (mean) | 0.0087 | 0.0090 |

Two things are visible. First, **ordering**: the ruled H2 order (best beam rank of any node in the block; hop-1 beam before
hop-2 beam) necessarily places every hop-1 block ahead of a block first reached at hop 2, and 79 / 96 (75 / 89) of the proposed
hop-2 answer blocks are first reached at hop 2 — so 15.7 (16.8) hop-1 blocks precede the answer block on average and only 4 / 96
(4 / 89) of them are seen by SAFE at positions 0–5, although 19 (17) are within the first six of their own hop. Second, and
decisive, **scoring**: 53 / 96 (53 / 89) of the proposed answer blocks are supported by the H2 list alone — outside the served
top-200 block ranking, the static expansion and the retrieval continuation — and under the additive F6 score a single channel at
position r is worth 1/(60 + r) (0.0167 at r = 0, 0.0092 at r = 49) while the blocks that take the slots carry two or three
channels. The mean gap between the sixth admitted block and the worst missing gold block (0.0087 / 0.0090) is the value of an
*entire additional channel at rank ≈ 50* (1/110 = 0.0091): the gold is one channel short, not a few positions short — which is why
even on hop 1, where 11 of the 25 complete-repair failures have the decisive missing block at H2 positions 0–5, only 1 reaches
slots 1–6. The six slots go instead to incumbents kept (153), static structural challengers (199), retrieval challengers (64) and
H2-credited blocks (160, of which 98 are hop-1 blocks — the beam's intermediates, which are exactly the blocks that also carry
served and static-structural support); 19 of the 576 slots hold a gold block.

*(iii) The bottleneck has a hard ceiling under ALL-gold@P50.* MetaQA questions have several answers, so a failing hop-2 query
misses 6.1 (6.5) gold blocks on average, and a query missing more than six blocks cannot be repaired by *any* six-slot rule:

| cache · group | failing queries | missing gold blocks: 1 / 2–3 / 4–6 / 7–12 / 13+ | repairable by *any* six-slot rule (missing ≤ 6) | … and all missing in the canonical SAFE pool | … and all missing in the SAFE_H2 pool | … and all missing proposed by H2 alone | realised newly covered vs L1: SAFE; SAFE_H2 |
|---|---|---|---|---|---|---|---|
| metaqa (MtK) · hop1 | 27 | 10 / 9 / 4 / 3 / 1 | 23 | 3 | 22 | 21 | 0; 1 |
| metaqa (MtK) · hop2 | 177 | 58 / 48 / 25 / 26 / 20 | 131 | 19 | 98 | 92 | 3; 4 |
| metaqa (MtK) · hop3 | 157 | 32 / 22 / 20 / 26 / 57 | 74 | 6 | 15 | 11 | 0; 0 |
| metaqa_phg · hop1 | 27 | 12 / 7 / 4 / 2 / 2 | 23 | 2 | 22 | 22 | 0; 1 |
| metaqa_phg · hop2 | 162 | 48 / 44 / 25 / 26 / 19 | 117 | 11 | 86 | 85 | 1; 4 |
| metaqa_phg · hop3 | 139 | 18 / 20 / 23 / 24 / 54 | 61 | 6 | 11 | 9 | 3; 3 |
| squad (MtK) · all | 14 | 14 / 0 / 0 / 0 / 0 | 14 | 7 | 9 | 2 | 6; 5 |
| musique (PHG) · all | 339 | 264 / 74 / 1 / 0 / 0 | 339 | 115 | 155 | 66 | 56; 55 |

Of metaqa's 177 hop-2 failures, 131 are repairable at all, and the SAFE_H2 candidate pool contains a complete repair for 98 of
them (92 through H2 alone) against 19 for the canonical pool — a five-fold larger reachable set; the frozen admission realises 4
(canonical SAFE: 3). On PHG: 117 repairable, 86 in the pool (85 through H2) vs 11 canonical, 4 realised (1). On hop 1 the pool
also grows from 3 to 22, with 1 realised. On musique the pool grows 115 → 155 and the realised gain does not change (56 → 55; at
most 7 of the 40 additional complete repairs are realised, SAFE_H2 vs SAFE +7/−6), and the 6 newly lost queries are displaced by
H2-credited blocks (11 of them). After this section the MetaQA multi-hop mass that §14 could reach is therefore a *known set of
candidate blocks inside SAFE's pool* — complete repairs for 98 / 86 hop-2 failures — that the frozen six-slot admission never
selects.

**Reading for the next decision (from the recorded numbers; nothing run).** The ruling's fallback — a universal admission
criterion ranked first by the number of independent supporting channels, A(b) = 1[H2] + 1[STRUCT] + 1[Dense] + 1[SPLADE] — would be
*safer* than F6 (a strictly stronger preference for multi-channel blocks), but on the recorded hop-2 failures it would rank the
answer blocks *lower*, not higher: in the decomposition available here (served top-200 block ranking / static expansion / retrieval
continuation / H2), 53 of the 96 (53 of 89) complete-repair queries have their answer block supported by H2 alone, i.e. support
count 1, behind every two-channel incumbent and static challenger. Agreement is the right instrument against the text losses §14.3
showed; it is not the instrument that admits a hop-2 answer block whose only witness is the beam that walked s → m → a. What such
a block has, and no static candidate has, is *provenance*: it was reached through a scored real-edge path from a served seed, and
its within-hop beam rank is 0–5 for 19 / 96 and ≤ 24 for 58 / 96. Whether that is admissible evidence for a slot, and under which
universal, parameter-free rule, is the question this section isolates; it is not answered here and no criterion has been tried.
`H2_AS_SAFE_CANDIDATE_GENERATOR` = **NOT_CONFIRMED as run (text-safe, inert)**; `MICRO_L3_H2` repair remains closed; no merge rule
chosen; DEV_B untouched; nothing under `data/` written.

## 16. Eighth ruling (2026-09-15): a query-local structure patch in place of the weakest static block — `H2_PATCH1` (`_l1c_h2patch.py`, `h2patch_A_<cache>.json`, `h2patch_A_SUMMARY.json`, pre-registered in `PREREGISTRATION_H2_PATCH1_DEV_A.json`, STATUS = POSTHOC_MECHANISM_TEST; gold-free input statistic `_l1c_h2patch_stat.py`, `h2patch_stat_A_<cache>.json`)

**Ruling recorded first.** *"We should stop trying to make H2-discovered answer nodes win six whole partition slots. That is the
wrong unit of repair … The next experiment I would run is a query-local structure patch … canonical SAFE 50 static blocks → H2
beam discovers actual nodes → collect H2 nodes NOT already exposed by SAFE → pack them into one query-local virtual block → replace
only the weakest SAFE block → 49 static blocks + 1 H2 patch. Call it `H2_PATCH1` … Keep the experiment brutally simple. I would
not tune anything on DEV_A. Freeze §14.1's beam exactly. Start from canonical SAFE … 1. Compute the union of nodes already exposed
by SAFE's 50 selected blocks. 2. Take H2 visited nodes not already in that union. 3. Order them using the already frozen H2 beam
order. 4. Take at most one block's capacity — ideally the same number of nodes as the block being replaced. 5. Remove the weakest
selected SAFE block. 6. Insert the query-local patch … If the removed block contains n_b nodes and H2 has only k < n_b novel nodes,
don't quietly increase/decrease the budget. Fill the remaining n_b − k positions from the removed block itself.
Then |served nodes_PATCH| = |served nodes_SAFE| exactly … Even cleaner would be simply PATCH = H2 nodes not already served and among
those preserve frozen beam score/order. If most hop1 nodes disappear because they're already served, we may not need a depth rule
at all. I would inspect that before preregistration, without using gold … Compare SAFE vs SAFE+PATCH1 across all five DEV_A
caches, exploratory only … And one very important diagnostic: Of the 98 MetaQA failures for which H2 currently has a complete
repair somewhere in its candidate pool, how many become covered when those H2 nodes are packed together rather than represented by
their owner blocks? If that number jumps dramatically, we've proven the problem wasn't H2 or ranking. It was static block
granularity … Don't run: agreement scoring, fourth H2 channel, support-count admission, another F6 merge, new RRF constants …
Close block-level H2→SAFE admission experiments for now. Open exactly one new arm: `H2_PATCH1`, a matched-budget query-local
virtual structure block replacing the weakest canonical SAFE block."*

**Gold-free input statistic, read before pre-registration (`_l1c_h2patch_stat.py`, `h2patch_stat_A_<cache>.json`; gold not
read).** After canonical SAFE selects its 50 blocks and its weakest admission (the sixth block of the frozen F6 competition order)
is removed, the pinned beam's visited nodes that are *not* exposed by the 49 kept blocks number 8.3 hop-1 + 70.8 hop-2 = 79.2 per
query on metaqa (MtK; 26 % of hop-1 visits, 74 % of hop-2 visits), 8.3 + 69.7 = 78.0 on metaqa_phg, 5.9 + 38.0 = 43.9 on musique,
1.8 + 19.7 = 21.6 on squad (84 queries with none) and 2.3 + 19.9 = 22.3 on squad_phg (71 with none); they come from 61 distinct
owner blocks per query on metaqa (26 on musique, 9–10 on squad) — six slots could never carry them as blocks. The removed block
holds 100–102 nodes (p10–p90 94–104). The novel set exceeds that capacity — the only case in which the patch *order* matters — in
79 / 1002 metaqa queries (7.9 %), 47 on metaqa_phg (4.7 %), 4 on musique, 1 on squad, 0 on squad_phg, and there the two candidate
orders differ by less than one node per query (first-visit order cuts 0.78 / 0.63 / ≤ 0.06 hop-2 novel nodes per query; deepest-
first cuts 0.7 / 0.3 / ≤ 0.03 hop-1 novel nodes); the hop-1 share of the patch is 10.6 % vs 9.7 % (MtK), 10.7 % vs 10.3 % (PHG),
13.4 % vs 13.3 % (musique), 8.5 % vs 8.5 % (squad) under the two orders. Decision taken from this and recorded in the
pre-registration: the frozen first-visit order (hop-1 beam, then hop-2 beam, seeds excluded), no depth rule — most hop-1
discoveries are already served (74–98 % of hop-1 visits), and the orders differ only inside the 0–8 % truncated queries and there
by a fraction of a node. Deepest-hop-first was **not run** ("exactly one new arm").

**What was run (pre-registered before any run, `PREREGISTRATION_H2_PATCH1_DEV_A.json`).** Per query: (1) canonical SAFE's final 50
= served positions 0–43 ∪ the six blocks admitted by the frozen F6 competition, through SAFE's own code (`_l1ps_router.build_cache`
+ `_l1kb_core.contexts` / `f6_select`, unchanged); (2) weakest = the sixth block of the F6 competition order (SAFE's own weakest
admission, ties already resolved by F6's rule), removed; (3) kept = the other 49 blocks, served whole; (4) novel = pinned-beam
visited nodes in the frozen fused first-visit order whose own block `hard[v]` is not among the 49 kept blocks (visited nodes of the
removed block therefore count as novel and keep their beam position); (5) capacity n_b = |removed block|, patch = the first
min(k, n_b) novel nodes; (6) if k < n_b, the remaining n_b − k positions are the removed block's own nodes not already in the
patch, in the frozen fused node order (`ret_rrf` position ascending for ranked nodes, then node id) — the budget is never
increased or decreased; (7) served = nodes of the 49 kept blocks ∪ patch, |served_PATCH| = |served_SAFE| exactly (asserted per
query). Nothing else: no second arm, no cap or constant chosen from DEV_A numbers, no change to the beam (5 frozen seeds, actual
STRUCT adjacency under the frozen hub policy, depth exactly 2, beam 100 per hop, frozen transition geometry + RRF; pinned module
sha `19774617…` asserted), no dataset branch. Invariants asserted on every cache: the recomputed canonical SAFE indicator equals
the frozen replay records' `_ind_SAFE` on every DEV_A row (`results/L1_CANONICAL/L1_REPLAY_{metaqa,squad}.json`,
`results/L1_LOWMEM/L1_REPLAY_{…}.json`); the beam rebuilt inside this run reproduces §14.1's `MICRO_L3_H2` exactly (0.6886, +51/−2;
0.7156, +46/−3; 0.7792, +2/−207; 0.7742, 0/−207; 0.5010, +15/−173); §15's SAFE_H2 pool counts are reproduced from
`h2safe_ceiling_A_*.json`. Metric = ALL-gold under matched unique-node exposure: every gold *node* of the query inside the served
node set. For L1 and SAFE this is identical to the frozen block-level ALL-gold@P50 because gold blocks are exactly the blocks of
the gold nodes (asserted against `ind_base` and `_ind_SAFE`); for the patch arm it is the only honest metric, since the patch is a
node list. ANY-gold and per-hop numbers reported; exact two-sided McNemar on the paired indicator. Decision rule (pre-registered):
text safety on each of squad, squad_phg, musique (NOT lost > gained with p < 0.05 vs SAFE) AND hop-2 ALL-gold gained > lost with
p < 0.01 vs SAFE on both metaqa caches; magnitudes reported, not thresholded; PASS = MECHANISM_CONFIRMED_ON_DEV_A (POSTHOC — DEV_A
has been read by §§2–15; promotion needs an untouched population).

**Result — five caches, DEV_A (`h2patch_A_<cache>.json`, `h2patch_A_SUMMARY.json`; ALL-gold under matched node exposure; points
in parentheses; gained/lost/p exact McNemar):**

| cache | L1 | SAFE (canonical) | **H2_PATCH1** | PATCH vs SAFE gained/lost/p | PATCH vs L1 | SAFE vs L1 | ANY-gold SAFE → PATCH (vs SAFE) | novel beam nodes / query: hop-1 + hop-2 | patch composition / query: hop-1 novel / hop-2 novel / fill from the removed block (n_b) | queries: novel > n_b (truncated) / no novel node (patch = removed block) | gold nodes gained vs SAFE (queries) | gold nodes lost with the removed block (queries) | significant loss vs SAFE? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) | 0.6397 | 0.6407 (+0.1) | **0.7515** (+11.1) | 111/0/8e-34 | 114/2/2e-31 | 3/2/1.000 | 0.9152 → 0.9830 (68/0/7e-21) | 8.3 + 70.8 = 79.2 | 8.3 / 70.1 / 21.7 (100.1) | 79 / 0 | 634 (185) | 10 (7) | no |
| metaqa_phg | 0.6727 | 0.6747 (+0.2) | **0.7735** (+9.9) | 100/1/8e-29 | 102/1/2e-29 | 4/2/0.688 | 0.9251 → 0.9830 (58/0/7e-18) | 8.3 + 69.7 = 78.0 | 8.3 / 69.1 / 23.8 (101.2) | 47 / 0 | 613 (174) | 6 (6) | no |
| squad (MtK) | 0.9859 | 0.9919 (+0.6) | **0.9919** (+0.0) | 0/0/1.000 | 6/0/0.031 | 6/0/0.031 | 0.9919 → 0.9919 (0/0/1.000) | 1.8 + 19.7 = 21.6 | 1.8 / 19.7 / 79.0 (100.5) | 1 / 84 | 0 (0) | 0 (0) | no |
| squad_phg | 0.9829 | 0.9889 (+0.6) | **0.9909** (+0.2) | 2/0/0.500 | 8/0/0.008 | 6/0/0.031 | 0.9889 → 0.9909 (2/0/0.500) | 2.3 + 19.9 = 22.3 | 2.3 / 19.9 / 79.0 (101.3) | 0 / 71 | 2 (2) | 0 (0) | no |
| musique (PHG) | 0.6596 | 0.7118 (+5.2) | **0.7329** (+2.1) | 21/0/1e-06 | 76/3/3e-19 | 56/4/9e-13 | 0.9910 → 0.9910 (0/0/1.000) | 5.9 + 38.0 = 43.9 | 5.9 / 37.9 / 57.7 (101.6) | 4 / 1 | 24 (24) | 0 (0) | no |

MetaQA per hop (recovery = (PATCH − SAFE) / (H2_RRF − L1) with H2_RRF = §14.1's symmetric `MICRO_L3_H2`, the arm that bought its
hop-2 gain with squad −20.7 / musique −15.9; a value above 1 means the patch gains more over SAFE than §14.1's full fusion repair (≈ 16
served blocks replaced per query) gained over L1):

| hop | MtK: L1 → SAFE → PATCH | PATCH vs SAFE | PATCH vs L1 | recovery MtK | gold nodes gained / lost with the removed block, MtK | PHG: L1 → SAFE → PATCH | PATCH vs SAFE | PATCH vs L1 | recovery PHG | gold nodes gained / lost, PHG |
|---|---|---|---|---|---|---|---|---|---|---|
| hop1 | 0.9172 → 0.9172 → 0.9939 | 25/0/6e-08 | 25/0/6e-08 | 2.778 (7.7 of 2.8 pts) | 94 / 0 | 0.9172 → 0.9172 → 0.9969 | 26/0/3e-08 | 26/0/3e-08 | 2.889 (8.0 of 2.8 pts) | 100 / 0 |
| hop2 | **0.4855 → 0.4942 → 0.7413** | **85/0/5e-26** | 88/0/6e-27 | 2.429 (24.7 of 10.2 pts) | 533 / 1 | **0.5291 → 0.5291 → 0.7442** | **74/0/1e-22** | 74/0/1e-22 | 2.313 (21.5 of 9.3 pts) | 509 / 2 |
| hop3 | 0.5271 → 0.5211 → 0.5241 | 1/0/1.000 | 1/2/1.000 | 0.200 (0.3 of 1.5 pts) | 7 / 9 | 0.5813 → 0.5873 → 0.5843 | 0/1/1.000 | 2/1/1.000 | −0.500 (−0.3 of 0.6 pts) | 4 / 4 |
| hop2 + hop3 pooled | 0.5059 → 0.5074 → 0.6346 | 86/0/3e-26 | 89/2/3e-24 | 2.150 (12.7 of 5.9 pts) | 540 / 10 | 0.5547 → 0.5577 → 0.6657 | 74/1/4e-21 | 76/1/1e-21 | 2.147 (10.8 of 5.0 pts) | 513 / 6 |
| all (ANY-gold: MtK 0.9132 → 0.9152 → 0.9830; PHG 0.9261 → 0.9251 → 0.9830) | 0.6397 → 0.6407 → 0.7515 | 111/0/8e-34 | 114/2/2e-31 | 2.265 (11.1 of 4.9 pts) | 634 / 10 | 0.6727 → 0.6747 → 0.7735 | 100/1/8e-29 | 102/1/2e-29 | 2.302 (9.9 of 4.3 pts) | 613 / 6 |

**Verdict by the pre-registered rule: PASS — `H2_PATCH1` = MECHANISM_CONFIRMED_ON_DEV_A (POSTHOC).** Both halves hold. Text:
squad 0/0 (p = 1.0; 0 gold nodes gained or lost — in 84 queries the beam finds no novel node and the patch *is* the removed block,
and in the rest the patch carries no gold node either way), squad_phg +2/0 (p = 0.5), musique +21/0 (p = 1e-6; SAFE's own musique
gain intact and extended, +76/−3 vs L1). MetaQA hop 2: **+85/0 (p = 5e-26)** on the Mt-KaHyPar cache and **+74/0 (p = 1e-22)** on
PHG — 0.4942 → 0.7413 and 0.5291 → 0.7442, i.e. 24.7 and 21.5 points, 2.4 and 2.3 times the hop-2 gain that §14.1's symmetric
fusion had shown over L1 (10.2 / 9.3 points) and 2.15 / 2.15 times its pooled hop2+hop3 gain — with zero hop-2 query lost. Hop 1
gains +25/0 and +26/0 (0.9172 → 0.9939 / 0.9969): the hop-1 answer nodes whose block is outside the served P50 are direct
neighbours of the seeds and enter as hop-1 novel nodes (94 / 100 of the 290 / 272 gold nodes entering through the patch in the
newly covered queries are hop-1 novel nodes). Hop 3 is unchanged (+1/0; 0/1). Overall 0.6407 → **0.7515** (+111/0, p = 8e-34;
+114/−2 vs L1) and 0.6747 → **0.7735** (+100/−1, p = 8e-29; +102/−1 vs L1); ANY-gold 0.9152 → 0.9830 and 0.9251 → 0.9830 (+68/0;
+58/0). For scale within this lane: §14.1's symmetric fusion bought hop 2 +10.2 / +9.3 points at squad −20.7 / musique −15.9;
§14.3's asymmetric hold kept 0.83 / 0.88 of that and still lost significantly on squad and musique; §15 moved hop 2 by +0.3 /
+1.2 points. This arm moves hop 2 by +24.7 / +21.5 points, ALL-gold by +11.1 / +9.9, with 0 / 1 queries lost anywhere and
squad exactly unchanged.

**The diagnostic the ruling asked for (block-level pool repairs, recomputed and asserted against §15's `h2safe_ceiling_A_*.json`):**

| cache · group | L1 failures (a gold block outside the served P50) | repairable by *any* six-slot rule (missing ≤ 6) | complete repair in the SAFE_H2 pool (§15) | … covered by canonical SAFE | … covered by **H2_PATCH1** | … whose missing gold nodes are all visited by the beam | … visited and covered by H2_PATCH1 |
|---|---|---|---|---|---|---|---|
| metaqa (MtK) · hop1 | 27 | 23 | 22 | 0 | 21 | 21 | 21 |
| metaqa (MtK) · hop2 | 177 | 131 | 98 | 3 | **85** | 85 | 85 |
| metaqa (MtK) · hop3 | 157 | 74 | 15 | 0 | 1 | 1 | 1 |
| metaqa_phg · hop1 | 27 | 23 | 22 | 0 | 22 | 22 | 22 |
| metaqa_phg · hop2 | 162 | 117 | 86 | 1 | **72** | 72 | 72 |
| metaqa_phg · hop3 | 139 | 61 | 11 | 3 | 2 | 3 | 2 |
| squad (MtK) · all | 14 | 14 | 9 | 6 | 6 | 6 | 6 |
| squad_phg · all | 17 | 17 | 12 | 6 | 8 | 8 | 8 |
| musique (PHG) · all | 339 | 339 | 155 | 56 | 76 | 76 | 76 |

Of the **98** metaqa hop-2 failures whose missing gold *blocks* all sit in §15's SAFE_H2 candidate pool — the set the frozen
six-slot admission converted into 3 covered queries — packing the beam's visited nodes into one query-local block covers **85**
(PHG: 86 → 1 under SAFE, **72** under the patch). The remaining 13 (14) are block-level repairs only: the owner block contains the
gold nodes but the beam visited just some of them, so a block admission would have carried them and a node patch cannot (a
failure whose missing gold nodes are all visited necessarily has all its missing blocks in the pool, so the 85 / 72 are exactly the
node-level-available subset of the 98 / 86, next table). Hop 1: 22 → 0 under SAFE,
21 (22) under the patch. Hop 3: 15 (11) in the pool, 1 (2) covered — the depth-2 beam does not reach depth-3 answers (below). The
number the ruling asked about therefore moves from 3 to 85 of 98 with the same beam, the same seeds, the same discoveries, the
same SAFE and the same node budget; the only thing that changed is the unit of repair.

Node-level availability (the honest ceiling of *this* patch: a SAFE failure is repairable by the patch only if every missing gold
node was visited by the beam):

| cache · group | SAFE failures (a gold node not served) | all missing gold nodes visited by the beam (node-level repair available) | … covered by **H2_PATCH1** | not covered: beyond the patch capacity / dropped with the removed block / other | missing gold nodes per failure (mean) | visited fraction of the missing gold nodes (mean) | failures with no missing gold node visited |
|---|---|---|---|---|---|---|---|
| metaqa (MtK) · hop1 | 27 | 25 | 25 | 0 / 0 / 0 | 3.59 | 0.926 | 2 |
| metaqa (MtK) · hop2 | 174 | 85 | **85** | 0 / 0 / 0 | 6.72 | 0.669 | 20 |
| metaqa (MtK) · hop3 | 159 | 1 | 1 | 0 / 0 / 0 | 16.33 | 0.008 | 153 |
| metaqa_phg · hop1 | 27 | 26 | 26 | 0 / 0 / 0 | 3.74 | 0.963 | 1 |
| metaqa_phg · hop2 | 162 | 74 | **74** | 0 / 0 / 0 | 7.03 | 0.650 | 18 |
| metaqa_phg · hop3 | 137 | 0 | 0 | 0 / 0 / 0 | 17.31 | 0.003 | 133 |
| squad (MtK) · all | 8 | 0 | 0 | 0 / 0 / 0 | 1.00 | 0.000 | 8 |
| squad_phg · all | 11 | 2 | 2 | 0 / 0 / 0 | 1.00 | 0.182 | 9 |
| musique (PHG) · all | 287 | 21 | 21 | 0 / 0 / 0 | 1.28 | 0.078 | 263 |

Every SAFE failure whose missing gold nodes are all visited by the beam is covered by the patch, on every cache and every hop:
85 / 85 (hop 2, MtK), 74 / 74 (hop 2, PHG), 25 / 25 and 26 / 26 (hop 1), 1 / 1 and 0 / 0 (hop 3), 21 / 21 (musique), 2 / 2
(squad_phg), 0 / 0 (squad). Zero repairs were lost to the capacity cut (79 / 47 truncated queries on the metaqa caches, none of
them dropping a gold node), zero to the removed block (the fill rule keeps the removed block's nodes whenever there is room), zero
to anything else — the patch converts 100 % of what the beam makes available. What it does not cover is therefore exactly what the
beam did not visit: 89 of the 174 hop-2 SAFE failures on MtK (88 of 162 on PHG) have at least one missing gold node the beam never
reached (20 / 18 have none reached; the visited fraction of the missing gold nodes averages 0.669 / 0.650, with 6.7 / 7.0 missing
gold nodes per failing query), and hop 3 is untouched by construction — a depth-2 walk from 5 seeds visits on average 0.8 % / 0.3 % of a
failing query's missing hop-3 gold nodes and none at all for 153 / 133 of the 159 / 137 hop-3 failures. The residual is beam recall (width,
seeds, hub policy, depth), not the patch and not selection. On the text caches the beam visits essentially nothing that is missing
(visited fraction 0.000 squad, 0.182 squad_phg, 0.078 musique), which is why the patch is inert there rather than harmful.

Attribution of the flips vs SAFE:

| cache | newly covered vs SAFE | gold nodes entering through the patch (of which hop-1 novel) | newly covered queries needing more than one patch node | newly lost vs SAFE | gold nodes dropped with the removed block in them | of the newly lost, also gaining a gold node |
|---|---|---|---|---|---|---|
| metaqa (MtK) | 111 | 290 (94) | 58 | 0 | 0 | 0 |
| metaqa_phg | 100 | 272 (100) | 54 | 1 | 1 | 0 |
| squad (MtK) | 0 | 0 (0) | 0 | 0 | 0 | 0 |
| squad_phg | 2 | 2 (0) | 0 | 0 | 0 | 0 |
| musique (PHG) | 21 | 21 (6) | 0 | 0 | 0 | 0 |

The 111 (100) newly covered metaqa queries receive 290 (272) gold nodes through the patch; 58 (54) of them needed more than one
patch node — multi-answer hop-2 questions whose gold nodes are spread over several owner blocks (a failing hop-2 query misses
6.7 gold nodes on average), which is the case a six-slot block rule structurally cannot repair and the reason §15's generator
could be right and inert at the same time. The removed block costs 10 gold nodes in 7 queries on MtK, but none of those queries
was covered by SAFE in the first place (0 newly lost); on PHG one hop-3 query loses its only missing gold node with the removed
block and is the single newly lost query (6 gold nodes dropped in 6 queries). The patch itself is 8.3 hop-1 novel + 70.1 hop-2
novel + 21.7 fill nodes on MtK (8.3 / 69.1 / 23.8 PHG; 5.9 / 37.9 / 57.7 musique; 1.8 / 19.7 / 79.0 squad): on the KB caches it is
78 % beam discoveries, on squad it is 79 % the removed block itself.

**What this proves and what it does not.** Proven on DEV_A (POSTHOC): the failure of §§14.3 and 15 was the *unit* of repair. Under
identical beam output, identical seeds, identical SAFE selection for 49 of 50 slots and identical node exposure, moving the H2
evidence from "six whole owner blocks competing for six slots" to "one query-local block of the visited nodes" turns an inert
mechanism (hop 2 +2/−1, +4/0; §15) into +85/0, +74/0, and it does so with 0 / 0 on squad and +21/0 on musique where the
symmetric fusion had lost 207 and 173 queries (§14.1) and the asymmetric hold still lost significantly (§14.3, squad 0/−8,
musique 11/−48). No ranking beyond the frozen first-visit order was involved, no constant was chosen from DEV_A, and the arm
degenerates to canonical SAFE exactly wherever the beam finds nothing novel. Not proven: (i) transfer — DEV_A has been read by
every section of this lane; the standing ruling is that DEV_B has been read twice and is not to be read again, so the confirmation
population is WebQSP or another genuinely untouched population, and that run needs a ruling; (ii) hop 3 — the residual there
(visited fraction ≤ 0.008) is depth, and a depth-3 or wider beam is a different arm, not run; (iii) the hop-2 residual (89 / 88
queries with an unvisited missing gold node) is beam recall — seeds, width, hub policy — also not run; (iv) downstream: the served
unit becomes 49 static blocks + one query-local node list of the same size, so the L1→L2 contract (block-scoped L2, and the
standing caution `L1_GAIN_SURVIVES_FROZEN_L2 = NO` from the pre-partition audit) is unverified for the patch; (v) cost: the beam's
22 ms per query plus set operations, not timed here. Not run, per the ruling: agreement scoring, a fourth H2 channel,
support-count admission, another F6 merge, new RRF constants, the deepest-first order, any other weakest / capacity / fill rule;
block-level H2→SAFE admission experiments are closed. `H2_PATCH1` = **PASS — MECHANISM_CONFIRMED_ON_DEV_A (POSTHOC)**; not
promoted; `MICRO_L3_H2` repair and `H2_AS_SAFE_CANDIDATE_GENERATOR` stay closed; DEV_B untouched; nothing under `data/` written.

## 17. Ninth ruling (2026-09-15), part 1: the frozen real-edge beam one hop deeper under the identical one-block patch — `H3_PATCH1` (`_l1c_h3patch.py`, `h3patch_A_<cache>.json`, `h3patch_A_SUMMARY.json`, pre-registered in `PREREGISTRATION_H3_PATCH1_DEV_A.json`, STATUS = POSTHOC_MECHANISM_TEST; gold-free input statistic `_l1c_h3patch_stat.py`, `h3patch_stat_A_<cache>.json`; attribution `_l1c_h3patch_why.py`, `h3patch_why_A_{metaqa,metaqa_phg}.json`)

**Ruling recorded first.** *"`H2_PATCH1` tells us something extremely strong: once the beam actually visits a missing gold node,
the node-level patch converts it essentially perfectly. So for deeper hops the bottleneck is now beam recall, not partition
selection. We don't need a heavyweight L3. Take the exact frozen real-edge beam and implement each step as batched sparse
operations … An H3 arm could simply do: L1 seeds → H1 beam → H2 beam → H3 beam → novel visited nodes → same one-block PATCH. No
new learned model, no new embeddings, no dataset logic … For immediate recall improvement: extend the already-successful real-edge
beam to a frozen vectorized H3 while keeping `PATCH1` at exactly one block-equivalent. That directly targets the hop-3
residual."* The ruling separates this from the partition question (§18); the two were run as two arms, nothing shared beyond the
frozen inputs.

**What was run.** The pinned beam module (`_l1c_microl3.py`, sha256 `19774617…`, unchanged on disk) is exec'd with `DEPTH 2 → 3`
and the capture of the third beam as the only string substitutions (asserted to occur exactly once each and recorded in every
output); beam width 100, the hub rule, the directed admissibility, the closed-form transition geometry, the RRF fusion of the three
ranks and the first-unvisited selection are the §14 code. Hops 1 and 2 of the depth-3 walk are therefore the §14.1 walk, and the
script asserts it: `MICRO_L3_H2` rebuilt from beams 1 + 2 equals its §14.1 record (`BASE_ALL`, gained / lost) on every cache, and
`H2_PATCH1` rebuilt from beams 1 + 2 under the §16 rule equals `h2patch_A_<cache>.json` (ALL-gold, gained / lost vs SAFE, per hop)
on every cache — `equals_section16_record = true` five times. The arm is the §16 rule with one more hop of visited nodes in the
novel set: kept = canonical SAFE minus its weakest admitted block; novel = visited nodes of beams 1, 2, 3 (frozen order within each
hop, seeds excluded) whose own block is not among the 49 kept; patch = the first *n_b* of them, remaining positions filled from the
removed block by (ret_rrf position, node id); |served nodes| = canonical SAFE per query (asserted). ALL-gold at the node level;
McNemar exact; DEV_A only; the batched multi-query implementation the ruling describes is deferred — it must reproduce this
per-query record exactly when promoted, and nothing in the result depends on it.

**Gold-free input statistic, read before pre-registration (`_l1c_h3patch_stat.py`; gold not read).** The only decision it
informed was the patch *order* when the novel set exceeds the removed block, and it was recorded in the pre-registration before
any gold was read:

| cache | visited / query: hop-1 / hop-2 / hop-3 | novel (not exposed by the 49 kept blocks): hop-1 / hop-2 / hop-3 | novel total: depth 3 vs depth 2 (capacity ≈ 100) | queries truncated: depth 3 vs depth 2 | first-visit order: hop-3 nodes admitted / cut per query; hop-2 nodes cut | deepest-first order: hop-2 nodes cut per query (queries affected) | beam cost depth 3: transitions scored hop-1 / hop-2 / hop-3; ms mean / p95 |
|---|---|---|---|---|---|---|---|
| metaqa (MtK) | 32.5 / 95.6 / 100.0 | 8.3 / 70.8 / 67.9 | 147.1 vs 79.2 | 967 vs 79 | 20.9 / 47.0; 0.78 | 39.5 (965 of 1002) | 38 / 335 / 1020; 72 / 221 |
| metaqa_phg | 32.5 / 95.6 / 100.0 | 8.3 / 69.7 / 62.6 | 140.6 vs 78.0 | 964 vs 47 | 22.9 / 39.6; 0.63 | 31.9 (955 of 1002) | 38 / 335 / 1020; 68 / 230 |
| squad (MtK) | 85.2 / 96.7 / 96.8 | 1.8 / 19.7 / 25.9 | 47.5 vs 21.6 | 67 vs 1 | 24.9 / 1.0; 0.01 | 0.8 (56 of 992) | 238 / 3437 / 5441; 189 / 415 |
| squad_phg | 85.2 / 96.7 / 96.8 | 2.3 / 19.9 / 27.0 | 49.3 vs 22.3 | 65 vs 0 | 26.1 / 0.9; 0.00 | 0.7 (47 of 992) | 238 / 3437 / 5441; 188 / 366 |
| musique (PHG) | 80.3 / 100.0 / 100.0 | 5.9 / 38.0 / 46.6 | 90.5 vs 43.9 | 391 vs 4 | 37.3 / 9.3; 0.06 | 6.8 (321 of 996) | 185 / 3198 / 4586; 189 / 367 |

At depth 3 the novel set exceeds the capacity in 967 / 1002 metaqa queries (79 at depth 2) and 964 on metaqa_phg, so at depth 3 the
order is decisive on the KB caches. Under the frozen first-visit order (hop-1 beam, hop-2 beam, hop-3 beam) the depth-2 patch is
preserved almost exactly — 0.78 / 0.63 hop-2 novel nodes per query are cut on the metaqa caches, ≤ 0.06 on text — and the hop-3
nodes take exactly the positions §16 filled from the removed block (21.7 / 23.8 / 79.0 / 79.0 / 57.7 fill positions per query; 20.9
/ 22.9 / 24.9 / 26.1 / 37.3 hop-3 nodes admitted). The deepest-first order would cut 39.5 / 31.9 hop-2 novel nodes per query in
96 % of the metaqa queries, i.e. dismantle the confirmed §16 mechanism to make room for hop-3 material; it was not run. Consequence
stated in advance: on metaqa the one-block budget admits ~21 of ~68 hop-3 novel visits per query (47 cut; 88 / 49 queries admit
none), so the hop-3 effect was expected to be budget-bound rather than beam-bound. The pre-registered rule (`decision_rule`):
text safety on squad / squad_phg / musique vs `H2_PATCH1` *and* vs SAFE (not lost > gained with p < 0.05); hop-3 gain vs
`H2_PATCH1` on both metaqa caches (gained > lost, p < 0.01); no hop-2 regression vs `H2_PATCH1` on both metaqa caches; PASS =
all three.

**Result (`h3patch_A_<cache>.json`, `h3patch_A_SUMMARY.json`) — FAIL on the hop-3 rule; text-safe; `H2_PATCH1` unchanged.**

| cache | L1 | SAFE | H2_PATCH1 (§16, reproduced) | **H3_PATCH1** | H3 vs H2_PATCH1 gained/lost/p | H3 vs SAFE | H3 vs L1 | ANY-gold H2 → H3 (vs H2) | patch composition / query: hop-1 / hop-2 / hop-3 novel / fill (H2_PATCH1 fill) | gold nodes gained vs SAFE by hop 1 / 2 / 3 (H2_PATCH1 total) | gold nodes lost with the removed block H2 → H3 | significant loss vs H2 or vs SAFE? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) | 0.6397 | 0.6407 | 0.7515 | **0.7525** (+0.1) | 1/0/1.000 | 112/0/4e-34 | 115/2/8e-32 | 0.9830 → 0.9840 (1/0/1.000) | 8.3 / 70.1 / 20.9 / 0.8 (21.7) | 95 / 539 / 108 (634) | 10 → 12 | no |
| metaqa_phg | 0.6727 | 0.6747 | 0.7735 | **0.7735** (+0.0) | 0/0/1.000 | 100/1/8e-29 | 102/1/2e-29 | 0.9830 → 0.9830 (0/0/1.000) | 8.3 / 69.1 / 22.9 / 0.8 (23.8) | 101 / 512 / 101 (613) | 6 → 12 | no |
| squad (MtK) | 0.9859 | 0.9919 | 0.9919 | **0.9919** (+0.0) | 0/0/1.000 | 0/0/1.000 | 6/0/0.031 | 0.9919 → 0.9919 (0/0/1.000) | 1.8 / 19.7 / 24.9 / 54.1 (79.0) | 0 / 0 / 0 (0) | 0 → 0 | no |
| squad_phg | 0.9829 | 0.9889 | 0.9909 | **0.9909** (+0.0) | 0/0/1.000 | 2/0/0.500 | 8/0/0.008 | 0.9909 → 0.9909 (0/0/1.000) | 2.3 / 19.9 / 26.1 / 52.9 (79.0) | 0 / 2 / 0 (2) | 0 → 0 | no |
| musique (PHG) | 0.6596 | 0.7118 | 0.7329 | **0.7359** (+0.3) | 5/2/0.453 | 26/2/3e-06 | 79/3/4e-20 | 0.9910 → 0.9910 (0/0/1.000) | 5.9 / 37.9 / 37.3 / 20.4 (57.7) | 6 / 18 / 6 (24) | 0 → 2 | no |

Per hop on the two metaqa caches:

| hop | MtK: L1 → SAFE → H2_PATCH1 → **H3_PATCH1** | H3 vs H2 | H3 vs SAFE | gold nodes gained vs SAFE by hop 1 / 2 / 3 (H3) | PHG: L1 → SAFE → H2_PATCH1 → **H3_PATCH1** | H3 vs H2 | H3 vs SAFE | gold nodes gained by hop (H3) |
|---|---|---|---|---|---|---|---|---|
| hop1 | 0.9172 → 0.9172 → 0.9939 → 0.9939 | 0/0/1.000 | 25/0/6e-08 | 94 / 0 / 0 | 0.9172 → 0.9172 → 0.9969 → 0.9969 | 0/0/1.000 | 26/0/3e-08 | 100 / 0 / 0 |
| hop2 | 0.4855 → 0.4942 → 0.7413 → 0.7413 | 0/0/1.000 | 85/0/5e-26 | 0 / 533 / 0 | 0.5291 → 0.5291 → 0.7442 → 0.7442 | 0/0/1.000 | 74/0/1e-22 | 1 / 508 / 0 |
| hop3 | **0.5271 → 0.5211 → 0.5241 → 0.5271** | **1/0/1.000** | 2/0/0.500 | 1 / 6 / 108 | **0.5813 → 0.5873 → 0.5843 → 0.5843** | **0/0/1.000** | 0/1/1.000 | 0 / 4 / 101 |
| hop2 + hop3 pooled | 0.5059 → 0.5074 → 0.6346 → 0.6361 | 1/0/1.000 | 87/0/1e-26 | — | 0.5547 → 0.5577 → 0.6657 → 0.6657 | 0/0/1.000 | 74/1/4e-21 | — |
| all | 0.6397 → 0.6407 → 0.7515 → 0.7525 | 1/0/1.000 | 112/0/4e-34 | 95 / 539 / 108 | 0.6727 → 0.6747 → 0.7735 → 0.7735 | 0/0/1.000 | 100/1/8e-29 | 101 / 512 / 101 |

`H3_PATCH1` is `H2_PATCH1` plus one query on metaqa (hop 3: 0.5241 → 0.5271, +1/0, p = 1.0) and identical to it on metaqa_phg
(hop 3 0/0) — the hop-3 gain the ruling targeted did not materialise; hop 2 is untouched on both (0/0), as the gold-free
statistic predicted for this order. The text caches are exactly unchanged on squad and squad_phg (0/0 vs `H2_PATCH1`) and
+5/−2 on musique (p = 0.45; +26/−2 vs SAFE, p = 3e-6), i.e. text-safe by the pre-registered criterion. The hop-3 patch positions
are not empty: 108 (101) hop-3 gold nodes enter the served set through them on MtK (PHG) — as many as the 95 (101) hop-1 gold
nodes §16 recovers — but they complete one (zero) hop-3 query.

**Why: at hop 3 the failure is the answer set, not (only) the budget (`diagnostic_node_level_availability`).** The availability
diagnostic re-done at depth 3, on the SAFE failures of every group (a "missing" gold node = one whose block canonical SAFE does
not serve):

| cache · group | SAFE failures | missing gold nodes (total; per failure) | missing gold nodes visited by the beam: depth 2 → depth 3 (at hop 1 / 2 / 3) | visited fraction per failure: depth 2 → 3 | failures with no missing gold node visited: depth 2 → 3 | failures with ALL missing gold nodes visited: depth 2 → 3 | … covered by the patch: depth 2 → 3 | … not covered at depth 3: beyond the patch capacity / dropped with the removed block / other |
|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) · hop1 | 27 | 97; 3.6 | 94 → 94 (94 / 0 / 0) | 0.926 → 0.926 | 2 → 2 | 25 → 25 | 25 → 25 | 0 / 0 / 0 |
| metaqa (MtK) · hop2 | 174 | 1169; 6.7 | 536 → 536 (0 / 536 / 0) | 0.669 → 0.669 | 20 → 20 | 85 → 85 | 85 → 85 | 0 / 0 / 0 |
| metaqa (MtK) · hop3 | 159 | 2597; 16.3 | 7 → **268** (1 / 6 / 261) | 0.008 → 0.115 | 153 → 75 | 1 → 4 | 1 → 2 | 2 / 0 / 0 |
| metaqa_phg · hop1 | 27 | 101; 3.7 | 100 → 100 (100 / 0 / 0) | 0.963 → 0.963 | 1 → 1 | 26 → 26 | 26 → 26 | 0 / 0 / 0 |
| metaqa_phg · hop2 | 162 | 1139; 7.0 | 512 → 512 (1 / 511 / 0) | 0.65 → 0.65 | 18 → 18 | 74 → 74 | 74 → 74 | 0 / 0 / 0 |
| metaqa_phg · hop3 | 137 | 2371; 17.3 | 4 → **211** (0 / 4 / 207) | 0.003 → 0.108 | 133 → 58 | 0 → 1 | 0 → 0 | 1 / 0 / 0 |
| squad (MtK) · all | 8 | 8; 1.0 | 0 → 0 (0 / 0 / 0) | 0.0 → 0.0 | 8 → 8 | 0 → 0 | 0 → 0 | 0 / 0 / 0 |
| squad_phg · all | 11 | 11; 1.0 | 2 → 2 (0 / 2 / 0) | 0.182 → 0.182 | 9 → 9 | 2 → 2 | 2 → 2 | 0 / 0 / 0 |
| musique (PHG) · all | 287 | 366; 1.3 | 24 → 30 (6 / 18 / 6) | 0.078 → 0.096 | 263 → 257 | 21 → 26 | 21 → 26 | 0 / 0 / 0 |

The depth-3 beam does reach hop 3: the missing hop-3 gold nodes it visits go from 7 to 268 on MtK (visited fraction 0.008 → 0.115)
and from 4 to 211 on PHG (0.003 → 0.108), and the hop-3 failures with *no* missing gold node visited fall from 153 to 75 of 159
(133 → 58 of 137). But a failing hop-3 query is missing 16.3 (17.3) gold nodes — a list question whose answers are a whole
frontier — and ALL-gold needs every one of them: only 4 (1) of the 159 (137) hop-3 failures have all their missing gold nodes
visited at depth 3, 2 (0) of those are covered by the patch and 2 (1) are cut by the capacity. So the budget the ruling asked to
keep is not what bounds hop 3 on DEV_A: even an unbounded patch of the depth-3 visits would complete 4 (1) hop-3 queries; the
remaining 155 (136) have at least one gold node the width-100 beam never visits at depth 3. Hops 1 and 2 are exactly as in §16
(25 / 25 and 85 / 85 covered of the failures with everything visited; 0 lost to the cut, 0 to the removed block) — the third hop
neither helps nor disturbs them.

| cache | newly covered vs H2_PATCH1 | gold nodes entering via hop-3 positions / via hop-1-2 positions | newly lost vs H2_PATCH1 | of the lost gold nodes: were fill under H2_PATCH1 / were cut hop-2 positions / other |
|---|---|---|---|---|
| metaqa (MtK) | 1 | 1 / 0 | 0 | 0 / 0 / 0 |
| metaqa_phg | 0 | 0 / 0 | 0 | 0 / 0 / 0 |
| squad (MtK) | 0 | 0 / 0 | 0 | 0 / 0 / 0 |
| squad_phg | 0 | 0 / 0 | 0 | 0 / 0 / 0 |
| musique (PHG) | 5 | 5 / 0 | 2 | 2 / 0 / 0 |

**Width or reach? (`_l1c_h3patch_why.py`, `h3patch_why_A_{metaqa,metaqa_phg}.json`; the same depth-3 beam re-run with the
scored candidate sets of every hop recorded; no arm, no selection change; post-hoc attribution on DEV_A).** For every missing gold
node of a SAFE failure: visited by the beam at hop *h*; or scored as a hop-*h* candidate — adjacent to a hop-(*h*−1) beam node under
the frozen admissibility and scored by the frozen geometry — but left out by the width-100 cut; or never adjacent to any beam
node at all:

| cache · group | SAFE failures | missing gold nodes | visited (first at hop 1 / 2 / 3) | scored as a candidate but cut by the width (first at hop 1 / 2 / 3) | never scored (not adjacent to any beam node) | seed / hub | failures with every missing gold node scored within depth 3 (within depth 2) | unique candidates scored per query, all queries of the group (hop 1 / 2 / 3) |
|---|---|---|---|---|---|---|---|---|
| metaqa (MtK) · hop1 | 27 | 97 | 94 / 0 / 0 | 0 / 0 / 0 | 3 | 0 / 0 | 25 (25) | 32 / 260 / 741 |
| metaqa (MtK) · hop2 | 174 | 1,169 | 0 / 536 / 0 | 0 / **630** / 0 | 2 | 1 / 0 | **173 (173)** | 27 / 215 / 772 |
| metaqa (MtK) · hop3 | 159 | 2,597 | 1 / 6 / 261 | 0 / 1 / 655 | **1,673** | 0 / 0 | 33 (1) | 38 / 319 / 777 |
| metaqa_phg · hop1 | 27 | 101 | 100 / 0 / 0 | 0 / 0 / 0 | 1 | 0 / 0 | 26 (26) | 32 / 260 / 741 |
| metaqa_phg · hop2 | 162 | 1,139 | 1 / 511 / 0 | 0 / **624** / 0 | 3 | 0 / 0 | **160 (160)** | 27 / 215 / 772 |
| metaqa_phg · hop3 | 137 | 2,371 | 0 / 4 / 207 | 0 / 2 / 621 | **1,537** | 0 / 3 | 21 (0) | 38 / 319 / 777 |

The two residuals have different causes. On the hop-2 failures the missing gold nodes are almost all *scored*: of 1,169 (1,139)
missing hop-2 gold nodes, 536 (511) are visited at hop 2, 630 (624) are hop-2 candidates that the width-100 cut leaves out, and
2 (3) are never adjacent to a beam node; 173 of 174 (160 of 162) hop-2 failures have every missing gold node inside the depth-2
candidate sets (215 unique hop-2 candidates per query on average, of which the beam keeps 100). The §16 hop-2 residual (89 / 88
queries with an unvisited gold node) is beam *width* at hop 2 — not reach, not seeds, not the hub rule. On the hop-3 failures it is
the opposite: of 2,597 (2,371) missing hop-3 gold nodes, 261 (207) are visited at hop 3, 655 (621) are scored hop-3 candidates cut
by the width, and 1,673 (1,537) — 64 % / 65 % — are never adjacent to any beam node, because the hop-2 beam that has to carry the
frontier is itself cut to 100 of ~215 candidates and the hop-3 frontier grows only from what survives; 33 (21) of the 159 (137)
hop-3 failures have all their missing gold nodes scored within depth 3 (1 / 0 within depth 2). Hop 3 is reach compounded from
the hop-2 width cut, and the two are one parameter. Width alone is not an arm, though: the patch is one block-equivalent and the
hop-2 novel set already takes 70 of its ~100 positions, so a beam that visited all ~215 hop-2 candidates would present ~150 novel
nodes to a 100-node patch in the same fused order that ranked the gold nodes below the top 100 — width, budget and order would
have to be ruled on together, and the width-100 cut is part of the pinned beam. Not run.

**Cost (depth 3, per query).** Transitions scored 38 / 335 / 1,020 at hops 1 / 2 / 3 on metaqa (238 / 3,437 / 5,441 squad;
185 / 3,198 / 4,586 musique); wall-clock 72–80 ms mean, 185–196 ms p95 on the metaqa caches (§14.1 depth 2: 22 ms), 233–317 ms
squad, 369 ms musique — the per-query Python loop, not the batched sparse form the ruling describes.

**What this closes and what it does not.** Under the ruling's own constraint — the frozen beam, one more hop, exactly one
block-equivalent — `H3_PATCH1` is **FAIL by the pre-registered rule: NOT_CONFIRMED as a hop-3 repair** (hop 3 +1/0 and 0/0), while
being text-safe and leaving `H2_PATCH1` exactly intact; §16 stands as recorded. The diagnosis is specific: the depth-3 beam raises
hop-3 node availability 14× (0.008 → 0.115 of the missing gold nodes) and those nodes do enter the patch (108 / 101 gold nodes),
but hop-3 questions on MetaQA are multi-answer with ~16–17 unserved gold nodes per failing query, and a width-100 real-edge beam
from five seeds visits all of them for 4 / 1 queries — the residual is beam *recall* in the sense of answer-set coverage, not the
one-block budget and not selection. The attribution names the parameter: at hop 2 the unvisited gold nodes are scored
candidates cut by the width (630 / 624 of 1,169 / 1,139; 173 / 160 of the 174 / 162 failures fully inside the depth-2 candidate
sets), and at hop 3 two thirds of the missing gold nodes are never adjacent to the beam because the hop-2 width cut has already
dropped the part of the frontier they hang from. Depth was the wrong axis; width is the axis — and it is a pinned constant of
the beam whose change also changes what the one-block patch has to hold. Not run, by the pre-registration: the deepest-first or
any other order, a second block, a larger capacity, another fill or weakest rule, depth 4, a wider beam, new seeds. Hop 2's
remaining 89 / 88 and hop 3 through this channel need a ruling on width together with the patch budget / order, and any such arm
meets the same transfer constraint as §16 (DEV_A read; DEV_B read twice; WebQSP or another untouched population). Nothing under
`data/` written; DEV_B untouched; no CONTRACT_FILE edited.

## 18. Ninth ruling (2026-09-15), part 2: overlap as part of the partition definition — `STRUCT_VERTEXCUT_V1`, a query-free structural diagnostic (`_l1c_vcut_build.py`, `_l1c_vcut_lib.py`, `_l1c_vcut_kstar.py`, `_l1c_vcut_metrics.py`; `parts/metaqa__STRUCT_VCUT_V1_k<k>.{npz,json}`, `…__PHG_con.{npy,RUN.json}`, `…_k1004__PHG_con__CAP100.npy`, `vcut_kstar_metaqa.json`, `vcut_struct_metaqa.json`; pre-registered in `PREREGISTRATION_STRUCT_VERTEXCUT_V1.json` before any build)

**Ruling recorded first.** *"Change the partition representation itself from 'hard nodes + halo afterward' to 'overlap is part of
the partition definition' … vertex-cut / overlapping partitioning: edges are assigned to partitions and a vertex is replicated into
every partition containing one of its edges … partition STRUCT edges, not nodes … z: E → {1,…,k}; B_p = {v : ∃ e ∋ v, z(e) = p} …
There is no special halo operation afterward. The overlap is the partition … dual incidence hypergraph — one hypergraph vertex
for each STRUCT edge x_e; for every original graph node v, a hyperedge h_v = {x_e : e incident to v}. Then partition the x_e's …
Σ_v (λ(h_v) − 1) is basically minimizing node replication … We construct a query-independent overlapping representation that
preserves local graph topology and short connected paths under a bounded replication budget. No queries. No gold answers. No
workload statistics. No labels required … one engineering issue: block capacity … partition edge-vertices with PHG; materialize
endpoint sets; measure actual unique nodes/block; perform deterministic capacity repair if necessary; count replication factor.
Do not silently compare a 170-node overlapping block against a 100-node hard block. Keep either: same unique-node exposure at
P50, or a strict block capacity … The metrics I would use before even running retrieval: Replication factor RF = Σ_p |B_p| / |V|;
1-hop edge containment; 2-hop wedge containment … this is the really interesting one; 3-hop path containment similarly as a
diagnostic. Then compare those against current PHG hard+halo. If the new scheme doesn't materially improve two-hop containment
for a reasonable replication factor, we don't even need the expensive retrieval replay … one new partition arm, not another
factorial: `STRUCT_VERTEXCUT_V1` … No relation labels. No queries. No gold. No training. No embedding changes. … I would not
immediately contaminate this first diagnostic with KNN."*

**What was built (pre-registered, `PREREGISTRATION_STRUCT_VERTEXCUT_V1.json`, 09:53 UTC, before any build).** Graph: the
undirected, de-duplicated STRUCT edges of canonical metaqa exactly as `CanonicalDataset("metaqa").keysets()` returns them
(N = 43,234, |E| = 124,669, 104 hubs, 14,660 degree-1 nodes, no degree-0 node; no KNN, no NERX, no relation labels, no direction;
key sha256 `92bfb08a…`). Dual incidence hypergraph: one vertex per STRUCT edge (unit weight); one net per node of degree ≥ 2
(h_v = its incident edges; a 0- or 1-pin net can never be cut, so degree-0/1 nodes form no net and the map is stored); 28,574
nets, 234,678 pins. Objective: Zoltan-PHG CONNECTIVITY (km1) with unit net weights = Σ_v (λ(h_v) − 1) = the total node
replication of the endpoint sets — the ruling's objective verbatim — run by the lane's validated PHG substitute exactly as
`_l1c_phg.py` runs every scratch hypergraph (NP = 4, IMBALANCE_TOL 1.03 on edge counts, driver sha checked against
`PHG_BUILD.json`, generic empty-part repair). Blocks: B_p = endpoint set of the edges of part p, materialised after partitioning.
Grid: k ∈ {k_f, 2 k_f, 3 k_f} = {432, 864, 1,296} for the replication / containment curve, plus the matched k* — the fixed point
of k_{i+1} = round(RF(k_i) · N / C) from the 2 k_f run so that the mean unique nodes per block equals the hard capacity
C = round(N / k_f) = 100 — which converged in one step: k* = 1,004 (mean |B_p| 102.35, within the pre-registered 5 %). Strict
capacity cell: the pre-registered deterministic repair on the k* partition (evict from the largest over-full block the node with
the fewest p-edges, ties larger degree then smaller id; move each of its p-edges to the lowest-class target — class 0 both
endpoints present, 1 the other endpoint present with room, 2 the node present with room, 3 neither with ≤ C − 2 — ties smallest
size then id; nodes that cannot be fully evicted are skipped; blocks with no evictable node stay over-full and are reported).
Baselines under the same metrics on the same graph: the served hard partitions (`hard_MtK` = the canonical Mt-KaHyPar H4_SK
partition from `replay_cache.npz`, `hard_PHG` = the served PHG+repair partition), the served *voting* unit (`legacy_*`: own block
+ the blocks of the directed STRUCT out-neighbours, `_l1s_core.Data.legacy_mem` — the ruling's "current PHG hard+halo",
unmatched), the DistDGL-style undirected halo (`halo_*` = B_p ∪ N(B_p), unmatched) and the §3 higher-order hard construction
`PATH2_PHG`. Metrics, all query-free: RF; block sizes; λ statistics for hubs and non-hubs; 1-hop edge containment; exact 2-hop
wedge containment per middle node by the pair-union rule (over all wedges, over wedges with a non-hub middle — the pre-registered
primary number, because hub-centred wedges are ≥ 90 % of all wedges and no bounded block can keep them together — the per-node
mean and the share of non-hub middles fully contained); exact 3-path containment per middle edge (every middle edge; |E| ≤
300,000); the §2 ball metric (closed 2-hop non-hub ball ≤ C inside one block; best-block coverage). Pre-registered rule
(`decision_rule_structural_only`): *at the strict-capacity cell (k\*, capacity-repaired; residual over-full blocks ≤ 1 % of
blocks): non-hub-middle 2-hop wedge containment ≥ 2 × the served hard partition's value AND an absolute gain ≥ +20 pts AND
RF ≤ 3.0; secondary: §2 ball intact fraction ≥ 2 × the served value* → MATERIAL; otherwise NOT_MATERIAL, which "closes
STRUCT_VERTEXCUT_V1 at the structural level"; either way the full curve is reported; metaqa is the primary dataset (squad and
musique are secondary curves, not run). Forbidden and not done: reading any query, seed, gold or replay cache beyond the served
`hard` vectors; KNN / NERX / labels / direction; any partitioner parameter outside the validated contract; any k outside the grid
+ fixed-point rule; a second capacity rule; tuning on the numbers; a retrieval replay; anything under `data/`.

**The k grid and k\* (`vcut_kstar_metaqa.json`; PHG runs under `phg_data/metaqa/`).**

| k | role | PHG km1 = Σ_v (λ(h_v) − 1) | wall s | peak RSS MB / rank | empty-block repair moves | raw RF | raw mean \|B_p\| | raw max \|B_p\| | raw blocks > C |
|---|---|---|---|---|---|---|---|---|---|
| 432 | curve | 49,147 | 12.8 | 29–30–31–32 | 2 | 2.137 | 213.84 | 261 | 428 |
| 864 | curve | 57,160 | 5.3 | 27–29–31 | 6 | 2.322 | 116.20 | 142 | 812 |
| 1296 | curve | 62,092 | 6.9 | 28–29–31 | 5 | 2.436 | 81.27 | 102 | 3 |
| 1004 | kstar_1 | 59,530 | 15.9 | 27–29–31 | 4 | 2.377 | 102.35 | 120 | 707 |

Balance is on edges per part (the PHG contract), so the unique-node block size is an outcome: at k_f = 432 a vertex-cut block
holds 214 unique nodes (2.1 × the hard block — the "170-node block" the ruling warned about), at 2 k_f 116, at 3 k_f 81, and at the
matched k* = 1,004 102.35 (max 120; 707 blocks above 100). RF rises slowly with k (2.14 → 2.44) because the replication is
dominated by the hubs: after partitioning, a hub (104 nodes) sits in 120 blocks on average and a non-hub node of degree ≥ 2 in
2.66, with 37 % of the latter in exactly one block; the 14,660 degree-1 nodes are in exactly one block by construction. The
capacity repair on k* (`capacity_repair` in `vcut_struct_metaqa.json`; 12.9 s, deterministic) makes 4,345 evictions and 5,514 edge
moves (3,212 of class 0 — the move removes a replica outright — 686 / 1,098 / 518 of classes 1 / 2 / 3; 4.4 % of the edges
reassigned), lowers RF 2.377 → 2.340 and the overflow from 5,184 memberships in 707 blocks to 781 in 398 blocks (max |B_p| 120 →
104), and then stalls: 77,332 candidate evictions are skipped because no target block has room. The stall is a pigeonhole, not a
weakness of the rule: after the repair Σ_p |B_p| = 101,181 memberships against k* · C = 100,400 places, and the residual overflow
is exactly the difference, 781 — every block is at or above C, so the strict cap at k* is feasible only if RF can be pushed below
k* · C / N = 2.322, which class-0 moves alone do not reach. The pre-registered strict-capacity cell therefore cannot satisfy its
own residual clause at a k* whose target was mean |B_p| = C: matching the *mean* to C and then demanding *max* ≤ C at the same k
are inconsistent unless the repair also lowers the replication, and the record shows how far it gets.

| capacity repair (k* = 1004, C = 100) | value |
|---|---|
| rule | deterministic capacity repair (pre-registered) |
| C | 100 |
| evictions | 4345 |
| edge_moves | 5514 |
| moves_by_class | [3212, 686, 1098, 518] |
| candidate_nodes_skipped | 77332 |
| over_full_blocks_before | 707 |
| overflow_nodes_before | 5184 |
| over_full_blocks_after_residual | 398 |
| overflow_nodes_after_residual | 781 |
| stuck_blocks | 398 |
| memberships_before | 102764 |
| memberships_after | 101181 |
| RF_before | 2.3769 |
| RF_after | 2.3403 |
| edges_reassigned_fraction | 0.044 |
| seconds | 12.9 |

**Result — every cell, the same metrics (`vcut_struct_metaqa.json`, 3,317 s).**

| cell | k | RF = Σ\|B_p\| / N | \|B_p\|: mean / p90 / max (blocks > C) | 1-hop edge containment | 2-hop wedge containment: non-hub middles / all / per-node mean / middles fully contained | 3-path containment: non-hub middles / all | §2 ball metric: intact / best-block coverage | λ mean: hubs / non-hub deg ≥ 2 (share with λ = 1) |
|---|---|---|---|---|---|---|---|---|
| hard_MtK | 432 | 1.000 | 100.1 / 104 / 104 (272) | 0.1810 | 0.0226 / 0.0011 / 0.074 / 4.0 % | 0.0029 / 0.0001 | 0.0938 / 0.381 | 1.000 / 1.000 (100.0 %) |
| legacy_MtK | 432 | 2.539 | 254.1 / 309 / 16,030 (431) | 1.0000 | 0.5615 / 0.9784 / 0.525 / 42.7 % | 0.0777 / 0.2404 | 0.1381 / 0.611 | 1.404 / 3.336 (42.6 %) |
| halo_MtK | 432 | 4.327 | 433.1 / 488 / 16,084 (431) | 1.0000 | 1.0000 / 1.0000 / 1.000 / 100.0 % | 0.2791 / 0.2739 | 0.5001 / 0.853 | 172.365 / 5.411 (4.0 %) |
| hard_PHG | 432 | 1.000 | 100.1 / 103 / 104 (364) | 0.1682 | 0.0159 / 0.0008 / 0.061 / 3.4 % | 0.0010 / 0.0000 | 0.0844 / 0.360 | 1.000 / 1.000 (100.0 %) |
| legacy_PHG | 432 | 2.596 | 259.8 / 268 / 15,620 (424) | 1.0000 | 0.5578 / 0.9783 / 0.522 / 42.7 % | 0.0842 / 0.2376 | 0.1248 / 0.596 | 1.433 / 3.422 (42.6 %) |
| halo_PHG | 432 | 4.431 | 443.4 / 498 / 15,668 (428) | 1.0000 | 1.0000 / 1.0000 / 1.000 / 100.0 % | 0.2074 / 0.2682 | 0.4916 / 0.848 | 174.452 / 5.552 (3.4 %) |
| PATH2_PHG | 432 | 1.000 | 100.1 / 103 / 103 (356) | 0.2279 | 0.0365 / 0.0017 / 0.145 / 8.7 % | 0.0043 / 0.0001 | 0.1724 / 0.460 | 1.000 / 1.000 (100.0 %) |
| vcut_k432 | 432 | 2.137 | 213.8 / 235 / 261 (428) | 1.0000 | 0.3463 / 0.0355 / 0.686 / 42.4 % | 0.0756 / 0.0120 | 0.2144 / 0.549 | 91.077 / 2.397 (42.1 %) |
| vcut_k864 | 864 | 2.322 | 116.2 / 128 / 142 (812) | 1.0000 | 0.2885 / 0.0242 / 0.634 / 38.6 % | 0.0517 / 0.0061 | 0.1881 / 0.516 | 113.702 / 2.596 (38.5 %) |
| vcut_k1296 | 1296 | 2.436 | 81.3 / 90 / 102 (3) | 1.0000 | 0.2589 / 0.0200 / 0.599 / 35.9 % | 0.0417 / 0.0043 | 0.1739 / 0.499 | 126.202 / 2.724 (35.8 %) |
| vcut_k1004 | 1004 | 2.377 | 102.3 / 112 / 120 (707) | 1.0000 | 0.2788 / 0.0228 / 0.621 / 37.4 % | 0.0476 / 0.0055 | 0.1849 / 0.511 | 120.000 / 2.656 (37.1 %) |
| **vcut_k1004_CAP100** | 1004 | 2.340 | 100.8 / 103 / 104 (398) | 1.0000 | 0.2649 / 0.0212 / 0.599 / 36.1 % | 0.0455 / 0.0050 | 0.1830 / 0.509 | 107.981 / 2.645 (36.0 %) |

**Reading.** *The served hard partitions keep almost nothing of the STRUCT neighbourhood inside a block:* 82 % (MtK) / 83 % (PHG)
of the STRUCT edges are cut, so 1-hop containment is 0.18 / 0.17, and a non-hub node's neighbour pairs share its block in
2.3 % / 1.6 % of the cases (per-node mean 0.074 / 0.061; 4.0 % / 3.4 % of the non-hub middles fully contained); 3-paths 0.3 % /
0.1 %. The §3 construction `PATH2_PHG` moved these to 0.23 / 0.037 / 0.004 — a hard partition of 100-node blocks cannot hold a
2-hop neighbourhood when the mean STRUCT degree is 5.8 and the middles' neighbours have neighbourhoods of their own; that was §2's
diagnosis and §4's marginal result. *The vertex-cut keeps it by construction:* every edge is in exactly one block with both
endpoints (1-hop containment 1.000 at every k), and at k* — 1,004 blocks of 102 unique nodes, RF 2.38 — 27.9 % of the non-hub
wedges are contained (per-node mean 0.62; 37 % of the non-hub middles fully contained), 4.8 % of the non-hub 3-paths and 18.5 % of
the §2 balls, at 12.3 × / 16 × / 2.0 × the served MtK values. At the strict-capacity cell (k* repaired; 1,004 blocks, mean 100.8,
max 104 — the same effective maximum as the served hard partitions, whose Mt-KaHyPar / PHG imbalance tolerance leaves 272 / 364
of their 432 blocks above 100 with max 104) the numbers are 0.2649 / 0.599 / 36.1 % / 0.0455 / 0.1830 at RF 2.34, and at the
smallest-block grid cell k = 1,296 (81 unique nodes per block — *less* exposure per block than the hard block — 3 blocks above
100, max 102, RF 2.44) 0.2589 / 0.599 / 35.9 % / 0.0417 / 0.1739. The containment does not depend on the block size in the way the
hard partitions' does: from 214-node blocks to 81-node blocks the non-hub wedge containment falls 0.346 → 0.259, while the hard
partition needs the unmatched voting unit (`legacy_*`: own block + out-neighbours' blocks, 254–260 memberships per block with a
16,030-node maximum, RF 2.54–2.60) to reach 0.56 and the full undirected halo (RF 4.3–4.4, hubs in 172 blocks each) to reach
1.0. Per unit of replication the vertex-cut is the better representation of the local topology: at RF 2.34–2.44 and ≤ 104
nodes per block it holds 26–28 % of the non-hub wedges and 4–5 % of the non-hub 3-paths; the served voting unit holds 56 % of the
wedges and 8 % of the 3-paths at RF 2.5–2.6 but with blocks of unbounded size (p90 268–309, max 15,620–16,030), i.e. it is not a
capacity-bounded exposure unit at all, and the served hard block — the unit P50 actually exposes — holds 2 %. The secondary §2
ball metric moves less (0.094 → 0.183–0.185, 1.95–1.97 ×; 0.174 at k = 1,296): a closed 2-hop ball of up to 100 nodes must sit
entirely in one ~100-node block, which the edge balance rarely produces; `PATH2_PHG` reached 0.172 at RF 1 because the ball was
its construction target — the ball metric is the one a hard partition *can* improve, the wedge metric is the one it cannot.

**Verdict by the pre-registered rule.**

| pre-registered structural verdict (metaqa) | value |
|---|---|
| served_reference | hard_MtK |
| strict_capacity_cell | vcut_k1004_CAP100 |
| wedge_nonhub_served | 0.0226 |
| wedge_nonhub_vcut_CAP | 0.2649 |
| ratio | 11.72 |
| gain_pts | 24.2 |
| RF_vcut_CAP | 2.3403 |
| residual_overfull_blocks | 398 |
| residual_ok (<= 1 % of blocks) | False |
| ball2_served | 0.0938 |
| ball2_vcut_CAP | 0.183 |
| ball2_ratio | 1.95 |
| rule | at the strict-capacity cell (k*, capacity-repaired; residual over-full blocks <= 1 % of blocks): non-hub-middle 2-hop wedge containment >= 2 x the served hard partition's value AND an absolute gain >= +20 pts AND RF <= 3.0; secondary: section-2 ball intact fraction >= 2 x the served value (metaqa: 8.4 % -> >= 16.8 %) |
| MATERIAL | False |
| secondary_ball2_2x | False |
| verdict | NOT_MATERIAL |

`STRUCT_VERTEXCUT_V1` = **NOT_MATERIAL by the pre-registered rule** — the strict-capacity cell fails the residual clause (398 of
1,004 blocks above C = 100, i.e. 39.6 % against the ≤ 1 % allowed) and the secondary ball clause (1.95 × against ≥ 2 ×); its three
containment clauses pass by a wide margin (0.0226 → 0.2649: 11.7 ×, +24.2 pts, RF 2.34 ≤ 3.0). Recorded as such, no clause
reinterpreted. Two facts about the residual clause are stated post hoc and labelled as such: (i) it was infeasible by
construction at k* (the pigeonhole above — 781 overflow memberships is the arithmetic minimum at k* · C = 100,400 places), which is a
flaw in how this pre-registration combined "mean = C" with "max ≤ C at the same k", not a property of the vertex-cut; (ii) the
served hard partitions do not satisfy it either (272 / 364 blocks above 100, max 104), and at their own effective maximum (104)
the repaired cell is exactly matched (max 104, mean 100.8 vs 100.1). The pre-registered grid cell k = 1,296, which has lower
exposure per block than the hard block, satisfies every clause of the rule including the residual one (3 blocks above 100 =
0.2 %) except the secondary ball clause (1.85 ×) — it is reported as the curve cell it is, not promoted to "the" cell. By the
pre-registration's own consequence clause NOT_MATERIAL closes the arm at the structural level; whether the containment result
meets the ruling's precondition for the replay ("materially improve two-hop containment for a reasonable replication factor") is
a ruling on the capacity clause, not a number this lane can re-decide after seeing it.

**What this shows and what it does not.** Shown, query-free: a hard 100-node partition of the MetaQA STRUCT graph exposes
2 % of a non-hub node's 2-hop wedges inside the served block and the higher-order hard construction 4 %; an edge-partitioned
overlapping representation with the same or smaller unique-node exposure per block exposes 26–28 % at a replication factor of
2.3–2.4 (2.1 excluding the 104 hubs, which carry 11 % of all memberships), with every edge intact; the served voting unit reaches
its 56 % only by being unbounded. Not shown: any retrieval number. A replay needs the served-unit contract the pre-registration
names — how P50 selects among overlapping blocks (a node now votes for λ(v) blocks; the frozen membership table and the RRF are
defined on a hard `hard[v]`), how exposure is matched (50 overlapping blocks of 100 expose fewer than 5,000 unique nodes, and hubs
replicated into ~110 blocks would dominate any block-level score), how degree-0 nodes are homed on squad (6,349) and musique
(9,858), and whether the KNN family joins the construction (excluded here by ruling). The replay caches are frozen against the
partition hash, so a served version is a new cache, not a scratch overlay. Nothing under `data/` written; no served partition,
cache or record touched; the squad and musique curves (secondary by the pre-registration) not run — the metaqa cell run took
55 min and the text graphs are several times larger; they can be run with the same four scripts if ruled useful.

## 19. Tenth ruling (2026-09-15): the vertex-cut blocks replayed through the frozen router — `STRUCT_VERTEXCUT_V1_REPLAY` (`_l1c_vcut_replay.py`, `vcut_replay_A_metaqa.{json,log}`, pre-registered in `PREREGISTRATION_STRUCT_VERTEXCUT_V1_REPLAY.json` before the run, STATUS = POSTHOC_STRUCTURAL_MECHANISM_EXPERIMENT_DEV_A)

**Ruling recorded first.** *"Keep the original §18 verdict exactly as written: `NOT_MATERIAL` under that preregistration. Do not
retroactively pass it. But the mechanism result is far too strong to close: 11.7× better 2-hop containment and ~2× the §2 ball
containment at RF ≈ 2.34 is absolutely material enough to justify a new, corrected replay preregistration. The capacity failure was
a specification mistake, not evidence against the representation. Your current hard partitions themselves operate under the
PHG/MtK balance envelope up to about 104 nodes, so the next preregistration should use the actual served effective cap, not an
impossible exact-100 cap. I would use the capacity-repaired k* = 1004 cell as the primary substrate … Do not use k = 1296 as primary
… Keep it as a structural sensitivity point only. The important replay contract: the representation should now explicitly be home +
overlap … M(v) = {all vertex-cut parts containing v} … H(v) = argmax_p #{incident STRUCT edges of v assigned to p}. Tie-break by
block ID. Then: home(v) = one canonical block; halo(v) = M(v) − {home(v)} … the partitioner decides from graph structure which
boundaries should overlap, rather than us adding a generic 1-hop halo after hard partitioning. No queries. No labels. No training.
For routing, reuse the already frozen hub philosophy. Non-hubs can vote to all vertex-cut memberships; pathological hubs should not
spray into ~110 partitions. A hub can vote only to its deterministic home … I would preregister both: P50 overlapping blocks and
report the actual unique-node exposure. Also report a secondary matched-unique-node-budget result … Do not silently compensate by
selecting extra blocks. If vertex-cut improves recall even while exposing fewer than the hard P50's unique nodes, that's an
especially strong result … 1. Vertex-cut replay on MetaQA DEV_A as a new post-hoc structural-mechanism experiment. This is allowed
because we're explicitly studying a new representation, not claiming confirmation."* §18's verdict stands unchanged.

**What was run (pre-registered 11:41:58 UTC, run 11:43 UTC; 50 s).** Substrate: the unchanged §18 file
`parts/metaqa__STRUCT_VCUT_V1_k1004__PHG_con__CAP100.npy` (sha `9a331cf8…`; k = 1,004 blocks, RF 2.3403, |B_p| 100–104, mean
100.78, non-hub 2-hop wedge containment 0.2649). Representation: M(v) = the blocks holding an edge of v (λ(v) = |M(v)|; hubs mean 108,
max 744; non-hubs mean 2.09; 57.6 % of nodes lie in exactly one block; mean halo 1.34); H(v) = the block holding most of v's edges,
ties (12.7 % of nodes) broken by the smallest block id; no degree-0 node on metaqa (asserted; the text corpora need a degree-0
ruling and were not run). Routing: the served dense and SPLADE top-100 hits of the replay cache's own rows vote through the
frozen `_ta_prepartition.partition_ranking` with the table `mem_vc` = M(v) for non-hubs, {H(v)} for the 104 hubs (90,055 entries
against 101,181 memberships: the hubs' 11,230 memberships collapse to 104 home votes; hubs are 0.08 % of dense and 1.0 % of SPLADE
top-100 hits), fused by the frozen RRF (`rrf_ranks`, dense tie-break), P50 = the first 50 fused blocks. Nothing added to the
memberships (no halo, no out-neighbour expansion), no constant changed (K0 60, K_LOCK 100, P_MAIN 50). Serving: gold node g is served
iff some block of M(g) is selected (the block physically contains g; the hub rule limits votes, not serving); ALL = every gold node
served; the node-level rule was asserted equal to the served block-level `base_all` on both hard baselines. Exposure = unique nodes
of the selected blocks (hard blocks are disjoint, so the served `scope_nodes` is already unique). Cells: `VCUT_P50` (primary),
`VCUT_MATCHED_LE` (secondary: per query the largest fused prefix whose unique exposure does not exceed that query's hard MtK P50
exposure — never above the hard budget), `VCUT_MATCHED_GE` (informational), two labelled diagnostics (`VCUT_HOME_ONLY_VOTE`: every
node votes only to its home; `HARD_MTK_OWN_BLOCK_VOTE`: the served hard partition voted through its own-block table instead of the
frozen legacy own + directed-out-neighbour table) and a reach decomposition. Baselines: the served hard MtK BASE (canonical H4_SK,
432 blocks, legacy vote table) and the served hard PHG BASE (`metaqa_phg`, same rows asserted; contrast only). DEV_A only
(1,002 queries; hop 1 / 2 / 3 = 326 / 344 / 332); exact two-sided McNemar; α = 0.01 pre-set; DEV_B and TEST never read.

**Result (`vcut_replay_A_metaqa.json`).**

| cell (DEV_A, n = 1,002) | blocks | unique exposure, mean (nominal Σ\|B_p\|) | ALL all | hop 1 (326) | hop 2 (344) | hop 3 (332) | hop 2+3 (676) | ANY all | vs served hard MtK BASE, all (McNemar) |
|---|---|---|---|---|---|---|---|---|---|
| HARD_MTK_BASE — served (canonical H4_SK Mt-KaHyPar, 432 blocks; legacy own + directed-out-neighbour vote table) | 50 | 5,023 | 0.6397 | 0.9172 | 0.4855 | 0.5271 | 0.5059 | — | — |
| HARD_PHG_BASE — served `metaqa_phg` (contrast only) | 50 | 5,089 | 0.6727 | 0.9172 | 0.5291 | 0.5813 | 0.5547 | — | — |
| **VCUT_P50 (primary)** — 50 fused vertex-cut blocks; non-hubs vote to M(v), hubs to H(v) | 50 | 4,166 (5,037; every query below its hard exposure; ratio 0.829) | **0.6776** | 0.9939 | 0.5552 | 0.4940 | 0.5251 | 0.9471 | +72 / −34, p = 2.8e-04 |
| **VCUT_MATCHED_LE (secondary)** — largest fused prefix at or under the query's hard exposure | 61.0 (58–75) | 4,986 | **0.6876** | 0.9939 | 0.5669 | 0.5120 | 0.5399 | 0.9531 | +75 / −27, p = 2.1e-06 |
| VCUT_MATCHED_GE (informational) — smallest fused prefix at or above it | 62.0 (59–76) | 5,059 (overshoot mean 36) | 0.6906 | 0.9939 | 0.5727 | 0.5151 | 0.5444 | 0.9531 | +78 / −27, p < 10⁻⁶ |
| VCUT_HOME_ONLY_VOTE (labelled diagnostic) — every node votes only to H(v); serving unchanged | 50 | 4,235 | 0.6018 | 0.8006 | 0.5262 | 0.4849 | 0.5059 | 0.9072 | +50 / −88, p = 0.002 |
| HARD_MTK_OWN_BLOCK_VOTE (labelled diagnostic) — the served hard partition voted through its own-block table | 50 | 5,023 | 0.1786 | 0.3865 | 0.1163 | 0.0392 | 0.0784 | 0.4541 | +2 / −464, p < 10⁻⁶ |

| hop (n) | hard MtK | VCUT_P50 | P50 vs MtK | VCUT_MATCHED_LE | MATCHED_LE vs MtK | hard PHG | P50 vs PHG |
|---|---|---|---|---|---|---|---|
| hop 1 (326) | 0.9172 | 0.9939 | +26 / −1, p < 10⁻⁶ | 0.9939 | +26 / −1, p < 10⁻⁶ | 0.9172 | +26 / −1, p < 10⁻⁶ |
| hop 2 (344) | 0.4855 | 0.5552 | +33 / −9, p = 2.7e-04 | 0.5669 | +36 / −8, p = 2.5e-05 | 0.5291 | +30 / −21, p = 0.262 |
| hop 3 (332) | 0.5271 | 0.4940 | +13 / −24, p = 0.099 | 0.5120 | +13 / −18, p = 0.473 | 0.5813 | +6 / −35, p = 4.9e-06 |
| hop 2+3 (676) | 0.5059 | 0.5251 | +46 / −33, p = 0.177 | 0.5399 | +49 / −26, p = 0.011 | 0.5547 | +36 / −56, p = 0.047 |
| all (1,002) | 0.6397 | 0.6776 | +72 / −34, p = 2.8e-04 | 0.6876 | +75 / −27, p = 2.1e-06 | 0.6727 | +62 / −57, p = 0.714 |

| representation / reach (metaqa, CAP cell) | value |
|---|---|
| k, RF, \|B_p\| | 1004, 2.3403, 100–104 (mean 100.78) |
| λ(v): hubs mean / max; non-hubs mean (deg ≥ 2); share of nodes with λ = 1; mean halo | 107.98 / 744; 2.09 (2.64); 57.6 %; 1.34 |
| homes with a tie broken by block id | 5,504 (12.7 %) |
| vote-table entries: mem_vc / all memberships / home-only | 90,055 / 101,181 / 43,234 |
| hub share of the served top-100 hits: dense / SPLADE | 0.08 % / 1.01 % |
| gold nodes that are hubs (mean per query); queries with ≥ 1 hub gold | 38.2 %; 405 |
| VCUT_P50 failures: UNREACHED / REACHED_WEAK (reached rate) | 323: 199 / 124 (0.8014) |
| worst gold block position among failures: mean / p50 / p90 / min; failures with worst ≤ 100 / ≤ 200 | 565 / 559 / 978 / 50; 38 / 70 |
| failures with a hub gold node; gold nodes per query (mean / p50 / max) | 36; 7.2 / 2 / 119 |
| failure overlap with the served hard BASE: both / vertex-cut only / hard only | 289 / 34 / 72 |
| pre-registered verdict | **MECHANISM_POSITIVE_AT_LOWER_EXPOSURE** (α = 0.01; primary +72 / −34, p = 2.8e-04 at 4166 ≤ 5023 nodes; secondary +75 / −27, p = 2.1e-06) |

**Verdict by the pre-registered rule: `MECHANISM_POSITIVE_AT_LOWER_EXPOSURE`.** The primary cell serves more complete gold
sets than the served hard P50 (0.6397 → 0.6776; +72 / −34, p = 2.8 × 10⁻⁴) while exposing 17 % fewer unique nodes (4,166 vs
5,023; every one of the 1,002 DEV_A queries is below its own hard exposure — 50 overlapping blocks of ≈ 101 nodes hold 4,166
distinct nodes), and the secondary matched-budget cell — the largest fused prefix that stays at or under each query's hard
exposure (61 blocks, 4,986 nodes) — passes as well (0.6876; +75 / −27, p = 2.1 × 10⁻⁶). Status exactly as pre-registered: a
post-hoc structural-mechanism result on DEV_A (read many times in this lane), not a confirmation, not promotable, the served L1
unchanged.

**Where the gain is, and where it is not.** Hop 1: 0.917 → 0.994 (+26 / −1). The vertex-cut has 1-hop edge containment 1.0 by
construction — every STRUCT edge lies inside a block — so a 1-hop answer set is served by the seed entity's own blocks; the hard
partition cuts 82 % of the edges and needs the legacy vote expansion to reach across them. Hop 2: 0.486 → 0.555 (+33 / −9,
p = 2.7 × 10⁻⁴), the wedge-containment gain (0.023 → 0.265) showing up at retrieval. Hop 3 goes the other way: 0.527 → 0.494
(+13 / −24, p = 0.099 at P50; +13 / −18, p = 0.47 at the matched budget) and against the hard PHG base the hop-3 loss is clear
(+6 / −35, p < 10⁻⁴). Non-hub 3-path containment is 0.046 at this RF, so 3-hop chains are not inside blocks; the vertex-cut cell
routes with no vote expansion at all, whereas the served hard BASE spreads every hit's vote to its out-neighbours' blocks — on hop 3
that generic spread is worth more than the structural overlap. The pooled hop 2+3 is therefore not significant at P50 (+46 / −33,
p = 0.18) and sits just above α at the matched budget (+49 / −26, p = 0.011): the whole verdict is a hop-1 + hop-2 result.

**The served hard BASE is itself an expanded router.** Voting the same served MtK partition through its own-block table gives
0.179 (hop 3: 0.039; −464 queries against the served BASE). The 0.64 of the served BASE therefore comes almost entirely from the
frozen legacy membership (own block + the blocks of the directed STRUCT out-neighbours: mean 254–260 nodes per hit, unbounded, RF
2.5–2.6 in §18's terms). So the contrast this section measures is *generic directed 1-hop vote expansion over hard ≈ 100-node blocks*
versus *partitioner-decided overlap at RF 2.34 with ≤ 104-node blocks and no expansion* — and the overlap wins at 17 % lower
exposure, which is the ruling's claim ("the partitioner decides … which boundaries should overlap"). The home-only diagnostic closes
the other side: a vertex-cut used as a hard partition (every node votes only to H(v)) falls to 0.602 (hop 1 0.801; +25 / −101 against
`VCUT_P50`, worse than the served hard BASE, +50 / −88) — the membership vote, not the block boundaries alone, carries the gain.

**PHG contrast (no verdict).** Against the served PHG+repair hard base (0.6727 at 5,089 nodes) the primary cell is level overall
(0.6776; +62 / −57, p = 0.71) at 18 % lower exposure, better on hop 1 (+26 / −1), level on hop 2 (+30 / −21, p = 0.26) and worse on
hop 3 (+6 / −35). The matched-budget cell (0.6876) is above it; no test was pre-registered for that pair.

**Residual (P50 failures, 323 of 1,002).** 199 are UNREACHED — at least one gold node has no block with a single vote from either
channel's top-100 — and 124 are reached but out-ranked; the worst gold block sits at fused position 565 of 1,004 on average (p50 559,
min 50; only 38 failures have it within the first 100, 70 within 200). Gold nodes are hubs 38 % of the time (405 queries carry at
least one hub answer; hubs are served through their full M(g), so only 36 failures involve a hub answer). Failure overlap with the
hard BASE: 289 fail under both, 34 only under the vertex-cut, 72 only under the hard partition. The reach problem (§2) is
unchanged in kind: the static overlap co-locates short structure, it does not put a vote on a block three hops away.

**Against the query-conditioned patch (§16).** `H2_PATCH1` moves metaqa 0.641 → 0.752 (hop 2 +85 / 0) by spending one block on
the beam's visited nodes; the static overlap moves 0.640 → 0.678 / 0.688 (hop 2 +33 / −9, +36 / −8) with no query-time
structure at all and at a lower exposure. They are not combined here (ruling: "I would not combine them yet"); the ruled
eventual order is static overlap → Dense + SPLADE → P50 → a small H2 patch only for residual nodes.

**Not done, by the ruling.** No k = 1,296 replay (granularity confound), no squad / musique (degree-0 homes need a ruling), no
SAFE / H2 / beam combination, no DEV_B, no constant or rule change after the numbers, no served-cache or `data/` write; the §18
verdict is untouched. Cost: 50 s for the whole replay (rankings 30 s; the membership table and homes 0.9 s).

## 20. Eleventh ruling (2026-09-15): the vertex-cut BASE carried to the text corpora with one universal degree-0 rule — `STRUCT_VERTEXCUT_V1_TRANSFER` (`_l1c_vcut_cap.py`, `_l1c_vcut_transfer.py`, `vcut_transfer_A_{squad,musique}.{json,log}`, `vcut_transfer_SUMMARY.json`; pre-registered in `PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json` before any placement, repair or replay number existed; STATUS = POSTHOC_STRUCTURAL_MECHANISM_TRANSFER_DEV_A)

**Ruling recorded first.** *"`STRUCT_VERTEXCUT_V1_REPLAY` should remain `MECHANISM_POSITIVE_AT_LOWER_EXPOSURE`, not promoted yet. But I
would absolutely keep it alive … 1. Do not combine VCUT with H2_PATCH1 yet. First establish whether VCUT is universal rather than a
MetaQA-specific structural win. 2. Do not touch H2 width/depth yet. Frozen H2_PATCH1 still needs WebQSP transfer. 3. Run VCUT transfer
on SQuAD and MuSiQue next, with one universal handling rule for structurally isolated nodes frozen before seeing results. 4. Leave the
WebQSP watcher alone. Do not kill the recurring `m3b_run.py` … Use one universal rule: Nodes with STRUCT incidence participate normally
in vertex-cut. A node with zero STRUCT degree receives exactly one non-overlapping home determined by its nearest frozen KNN neighbor
that already has a VCUT membership; use that neighbor's deterministic home. If no such anchored neighbor exists, place it in the
currently smallest VCUT block, canonical-node-order tie break … M(v) = VCUT memberships(v) if d_STRUCT(v) > 0; {H_KNN-anchor(v)} if
d_STRUCT(v) = 0. Hubs still vote only through deterministic H(v). That should be preregistered once and applied unchanged to SQuAD and
MuSiQue … The clean gate should be against each dataset's canonical hard BASE: no significant ALL-gold loss on SQuAD or MuSiQue;
unique-node exposure reported and preferably ≤ hard P50 exposure; then report gain/loss separately at matched exposure. If it passes
that, we have evidence for a universal statement: Overlap-native partitioning can replace generic 1-hop block expansion with a more
exposure-efficient static routing representation … No composition yet."* §18 and §19 verdicts untouched; H2_PATCH1 frozen; the WebQSP
watcher untouched (still CONTENDED at every sample; nothing signalled; NP unchanged).

**What was pre-registered (12:23:51 UTC) and run.** Construction = the §18 chain unchanged per dataset (`_l1c_vcut_kstar.py <ds>`:
k ∈ {k_f, 2k_f, 3k_f}, then the k* fixed point, at most three k* runs; the validated PHG driver at NP = 4; then the §18 deterministic
capacity repair at C = 100 applied once to the k* cell over the STRUCT edges by the new repair-only wrapper `_l1c_vcut_cap.py`). The
degree-0 rule exactly as ruled, with its reading fixed before the run: anchored = d_STRUCT > 0 (no chaining through other degree-0
nodes, so every home is a function of the frozen graph alone); "nearest" = the highest float32 cosine weight among the anchored
neighbours in the frozen `knn.npz` (one row per unordered pair), ties → smallest node id; the home is the anchor's H(u); fallback in
ascending node id into the currently smallest block (live sizes, all anchored placements applied first, ties → smallest block id);
degree-0 nodes are non-hubs with a single membership, so they vote to and are served through that block; no re-repair afterwards.
Representation, routing, serving, exposure and the matched-budget cells are §19's, through the pinned `_l1c_vcut_replay` functions
(non-hubs → M(v), hubs → H(v); frozen K0 / K_LOCK / P_MAIN / RRF; node-level ALL through M(g); unique exposure; largest fused prefix at
or under the gate's exposure). Gate baselines: squad = the served `squad` cache (canonical H4_SK Mt-KaHyPar, 202 blocks; `squad_phg`
as contrast only); musique = the served `musique` cache (canonical LOWMEM PHG_C1, 1,175 blocks; its only served partition). Labels
per dataset, pre-set: LOSS = lost > gained with p < 0.01 at P50 vs the gate; GAIN symmetric; NEUTRAL otherwise; exposure flag
LE_HARD / ABOVE_HARD on the DEV_A means; the matched cell labelled the same way and reported separately; the universal statement
SUPPORTED iff §19 positive AND neither text dataset LOSS AND both LE_HARD, PARTIALLY_SUPPORTED iff no LOSS but some exposure above
hard, NOT_SUPPORTED iff any LOSS. DEV_A only (squad 992 queries, one gold paragraph each, ALL = ANY; musique 996 queries, 2–4 gold
paragraphs, no hop labels in the served cache); exact two-sided McNemar; α = 0.01; DEV_B and TEST never read.

**Construction on text (structural, no verdict).** The text STRUCT graphs are dense where metaqa's is sparse: mean STRUCT degree
among STRUCT nodes 107 on squad (13,884 nodes, 744,734 edges, 6,349 isolated), 49 on musique (107,676 nodes, 2,651,280 edges,
9,858 isolated, 9,360 of degree 1), 5.8 on metaqa. The k* fixed point therefore lands at k = 2,499 on squad (three runs, mean |B_p|
106.9, not within the ±5 tolerance; RF 13.20) and k = 12,518 on musique (converged; mean 99.3; RF 10.58) — 12× / 11× the hard k_f
against 2.3× on metaqa — and the capacity repair, which on metaqa moved 4,345 nodes, here evicts 57,995 (squad; 86,717 of 87,321
edge moves are class 0, i.e. both endpoints already share another block) and 218,930 (musique; 195,273 class 0), ending at RF 10.35 /
9.25 with every block ≤ 100 and no residual (42 s / 315 s). The degree-0 rule then places 5,294 of 6,349 (squad) and 9,599 of 9,858
(musique) isolated nodes through an anchored KNN neighbour (mean anchor weight 0.60 / 0.56, no ties); the 1,055 / 259 fallbacks are
nodes whose KNN neighbours are all isolated too (960 / 248 of them would have been anchored under a chaining reading — reported, not
run). After placement: mean |B_p| 86.3 / 87.7, max 112 / 132, none above 1.5 C; RF 10.66 / 9.34. Every STRUCT node of squad sits in
6.9 blocks (non-hubs) — 1.3 % of them in exactly one — and hubs are 35 % of squad's STRUCT nodes (their λ 30, max 383); on musique
non-hubs sit in 4.7 blocks, 26 % in one, hubs 9 % (λ 64, max 2,444). The §18 structural metrics (wedge / path / ball containment)
were not computed for the text cells — they gate nothing here and are expensive on graphs this dense.

**Result (`vcut_transfer_A_squad.json`, `vcut_transfer_A_musique.json`, `vcut_transfer_SUMMARY.json`).**

| squad cell (DEV_A, n = 992) | blocks | unique exposure, mean (nominal Σ\|B_p\|) | ALL all | ANY all | vs HARD_MTK_BASE (McNemar) |
|---|---|---|---|---|---|
| HARD_MTK_BASE — served `squad` (canonical H4_SK Mt-KaHyPar, 202 blocks; legacy own + directed-out-neighbour vote table) — GATE | 50 | 5,089 | 0.9859 | — | — |
| HARD_PHG_BASE — served `squad_phg` (contrast only) | 50 | 5,089 | 0.9829 | — | — |
| **VCUT_P50 (primary)** — 50 fused vertex-cut blocks; non-hubs vote to M(v), hubs to H(v); degree-0 nodes in their one home | 50 | 2,208 (4,332; 992 of 992 queries below their hard exposure; ratio 0.434) | **0.9768** | 0.9768 | +4 / −13, p = 0.049 |
| **VCUT_MATCHED_LE (secondary)** — largest fused prefix at or under the query's hard exposure | 143.8 (100–222) | 5,071 | **0.9909** | 0.9909 | +9 / −4, p = 0.267 |
| VCUT_MATCHED_GE (informational) — smallest fused prefix at or above it | 144.7 (101–223) | 5,108 (overshoot mean 19) | 0.9909 | 0.9909 | +9 / −4, p = 0.267 |
| VCUT_HOME_ONLY_VOTE (labelled diagnostic) — every node votes only to H(v); serving unchanged | 50 | 2,386 | 0.9849 | 0.9849 | +8 / −9, p = 1.000 |
| HARD_OWN_BLOCK_VOTE (labelled diagnostic) — the served hard partition voted through its own-block table | 50 | 5,036 | 0.9940 | 0.9940 | +11 / −3, p = 0.057 |

| musique cell (DEV_A, n = 996) | blocks | unique exposure, mean (nominal Σ\|B_p\|) | ALL all | ANY all | vs HARD_PHG_BASE (McNemar) |
|---|---|---|---|---|---|
| HARD_PHG_BASE — served `musique` (canonical LOWMEM PHG_C1, 1,175 blocks; legacy vote table) — GATE | 50 | 5,117 | 0.6596 | — | — |
| **VCUT_P50 (primary)** — 50 fused vertex-cut blocks; non-hubs vote to M(v), hubs to H(v); degree-0 nodes in their one home | 50 | 3,204 (4,573; 996 of 996 queries below their hard exposure; ratio 0.626) | **0.6014** | 0.9709 | +65 / −123, p = 2.8e-05 |
| **VCUT_MATCHED_LE (secondary)** — largest fused prefix at or under the query's hard exposure | 86.9 (67–182) | 5,088 | **0.6687** | 0.9789 | +93 / −84, p = 0.548 |
| VCUT_MATCHED_GE (informational) — smallest fused prefix at or above it | 87.9 (68–183) | 5,145 (overshoot mean 28) | 0.6707 | 0.9789 | +94 / −83, p = 0.452 |
| VCUT_HOME_ONLY_VOTE (labelled diagnostic) — every node votes only to H(v); serving unchanged | 50 | 3,230 | 0.6145 | 0.9839 | +90 / −135, p = 0.003 |
| HARD_OWN_BLOCK_VOTE (labelled diagnostic) — the served hard partition voted through its own-block table | 50 | 4,712 | 0.6446 | 0.9930 | +86 / −101, p = 0.306 |

| representation / degree-0 / reach | squad | musique |
|---|---|---|
| k*, RF (STRUCT only → after placement), \|B_p\| after placement mean / p50 / max (blocks > C, > 1.5 C) | 2,499, 10.35 → 10.66, 86.3 / 100 / 112 (284, 0) | 12,518, 9.25 → 9.34, 87.7 / 100 / 132 (2492, 0) |
| hubs (share of STRUCT nodes); λ hubs mean / max; λ non-hub STRUCT nodes mean; STRUCT nodes with λ = 1; mean halo (STRUCT nodes) | 4,900 (35.3 %); 30.0 / 383; 6.94; 1.3 %; 14.08 | 9,727 (9.0 %); 64.1 / 2444; 4.74; 26.4 %; 9.10 |
| degree-0 nodes: anchored / fallback (fallback with a placed degree-0 KNN neighbour); anchor weight mean / min; anchored ties by node id | 6,349: 5,294 / 1,055 (960); 0.596 / 0.412; 0 | 9,858: 9,599 / 259 (248); 0.562 / 0.359; 0 |
| vote-table entries: mem_vc / all memberships / home-only | 73,605 / 215,782 / 20,233 | 483,704 / 1,097,546 / 117,534 |
| served top-100 hits that are hubs: dense / SPLADE; that are degree-0: dense / SPLADE | 25.8 % / 26.3 %; 29.1 % / 29.6 % | 14.0 % / 14.6 %; 4.7 % / 5.1 % |
| gold nodes that are hubs (mean fraction; queries with ≥ 1); that are degree-0 (mean fraction; queries with ≥ 1) | 21.7 % (215); 26.8 % (266) | 15.1 % (333); 3.3 % (78) |
| VCUT_P50 failures: UNREACHED / REACHED_WEAK (reached rate); with a hub gold; with a degree-0 gold | 23: 2 / 21 (0.9980); 4; 9 | 397: 155 / 242 (0.8444); 139; 46 |
| worst gold block position among failures: mean / p50 / p90 / min; failures with worst ≤ 100 / ≤ 200 | 192 / 90 / 316 / 50; 13 / 16 | 2311 / 408 / 7977 / 50; 81 / 133 |
| gold nodes per query: mean / p50 / max | 1.0 / 1 / 1 | 2.6 / 2 / 4 |
| failure overlap with the gate: both / vertex-cut only / hard only | 10 / 13 / 4 | 274 / 123 / 65 |
| pre-registered per-dataset verdict (primary label · exposure flag); matched label | **NEUTRAL_EXPOSURE_LE_HARD**; matched **NEUTRAL** (α = 0.01; primary +4 / −13, p = 0.049 at 2,208 ≤ 5,089 nodes; matched +9 / −4, p = 0.267) | **LOSS_EXPOSURE_LE_HARD**; matched **NEUTRAL** (α = 0.01; primary +65 / −123, p = 2.8e-05 at 3,204 ≤ 5,117 nodes; matched +93 / −84, p = 0.548) |
| wall clock (s) | 72 | 76 |

**Verdicts by the pre-registered rule.** squad: `NEUTRAL_EXPOSURE_LE_HARD` (P50 0.9859 → 0.9768, +4 / −13, p = 0.049 — not a
significant loss at α = 0.01 — at 2,208 unique nodes, 43 % of the hard P50's 5,089; matched budget `NEUTRAL`, 0.9909 at 143.8 blocks,
+9 / −4, p = 0.27). musique: **`LOSS_EXPOSURE_LE_HARD`** (P50 0.6596 → 0.6014, +65 / −123, p = 2.8 × 10⁻⁵, at 3,204 unique nodes,
63 % of the hard 5,117; matched budget `NEUTRAL`, 0.6687 at 86.9 blocks, +93 / −84, p = 0.55). Universal statement:
**`NOT_SUPPORTED`** — the ruling's pass condition ("no significant ALL-gold loss") fails on musique at P50, so the exposure-efficient
replacement claim does not transfer. Status exactly as pre-registered: post-hoc DEV_A evidence on all three datasets; nothing promoted;
the served L1 and §19's verdict unchanged; no composition with SAFE or H2_PATCH1 run (ruling: "No composition yet").

**Why it does not transfer (mechanism, from the pre-registered diagnostics).**

1. *P50 is not the same unit on text.* Fifty vertex-cut blocks are 50 × ~100 nominal nodes everywhere, but their unique content is set
by the replication factor: at RF 2.3 (metaqa) the 50 blocks hold 4,166 distinct nodes (83 % of the hard budget), at RF 10–11 they hold
2,208 (squad, 43 %) and 3,204 (musique, 63 %). The primary cell on text is therefore a budget cut of 37–57 % in unique nodes, and the
musique loss is what that cut costs: at the matched unique budget (86.9 blocks) the same representation is level (+93 / −84), and on
squad level as well (+9 / −4 at 143.8 blocks). The "fewer unique nodes and better" result of §19 needs the sparse-graph regime where
RF stays near 2.

2. *There is no 1-hop vote expansion to replace on text.* §19's contrast was the served hard BASE's legacy own + directed-out-neighbour
vote table against partitioner-decided overlap, and on metaqa that expansion is worth 46 points (own-block vote 0.179 vs 0.640). On
text it is worth nothing: the served squad partition voted through its own block scores 0.9940 — *above* the legacy BASE (0.9859;
+11 / −3, p = 0.057) — and the served musique partition 0.6446 vs 0.6596 (+86 / −101, p = 0.31). The mechanism §19 measured — overlap
substituting for generic expansion — has no counterpart to substitute on paragraph graphs whose STRUCT edges are article / entity
co-membership rather than relational hops; the dense + SPLADE hits already land in the right hard block.

3. *The membership vote adds nothing on text.* On metaqa, home-only voting fell from 0.678 to 0.602 (the overlap vote carried the gain);
on squad home-only is 0.9849 vs the primary 0.9768 (+13 / −5, p = 0.10) and on musique 0.6145 vs 0.6014 (+79 / −66, p = 0.32) — voting
a hit into its 5–7 blocks does not rank the gold's blocks higher than voting it into one. With λ ≈ 5–7 per non-hub hit the vote mass is
spread over blocks that share an edge with the hit, which on a co-membership graph is most of the article or entity neighbourhood.

4. *Degree-0 handling is not the failure point.* Isolated nodes are 29 % of squad's served top-100 hits and 27 % of its gold paragraphs
(266 of 992 queries), 5 % of musique's hits and 3 % of its gold (78 queries). 9 of squad's 23 primary failures and 46 of musique's 397
involve a degree-0 gold; the rest fail with fully STRUCT-anchored gold. The rule ran as specified (83 % / 97 % anchored, no re-repair,
block sizes reported); a chaining reading would have moved 960 / 248 fallbacks and cannot change the verdict's direction.

5. *Where the musique loss sits (post hoc, descriptive only; `vcut_transfer_A_musique__posthoc_by_gold_count.json`, pooled numbers
asserted equal to the record).* By the number of gold paragraphs — musique's 2 / 3 / 4-hop question type, since the served cache
carries no hop labels — the P50 loss is broad: 2-gold (n 543) 0.786 → 0.746 (+30 / −52, p = 0.020), 3-gold (n 295) 0.617 → 0.522
(+25 / −53, p = 0.002), 4-gold (n 158) 0.304 → 0.253 (+10 / −18, p = 0.19); at the matched budget every type is level or slightly up
(2-gold +34 / −41; 3-gold 0.654, +43 / −32, p = 0.25; 4-gold 0.335, +16 / −11). Reach: 155 of the 397 primary failures are UNREACHED
and 242 out-ranked (worst gold block at fused position 408 of 12,518 on the median failure, mean 2,311) — with 12,518 blocks the
fused ranking has far more places for a gold block to sit than the 1,004 of metaqa.

**What this settles.** The ruling's question — universal, or a MetaQA-specific structural win — is answered: the exposure-efficient form
of the vertex-cut BASE (fewer unique nodes, no loss) is specific to the sparse relational graph; on dense paragraph graphs it is a
budget cut that loses at P50 and is level at the matched budget, and the mechanism that made it win (replacing the legacy 1-hop vote
spread) has nothing to replace there. It is text-safe only at matched exposure, which is not the claim. By the pre-registered
what-follows clause the composition with SAFE and H2_PATCH1 is not opened; the ruled order continues with the WebQSP watcher
(unchanged; it will record its own NOT_RUN if the budget expires contended) and the frozen H2_PATCH1 → WebQSP transfer after it.

**Not done, by the ruling.** No SAFE / H2 / beam composition; no second degree-0 rule or chaining variant; no re-repair or re-balancing
after placement; no other k replayed; no k* re-targeting for the degree-0 nodes; no DEV_B; no constant changed; no served cache or
`data/` write; §18 / §19 verdicts untouched; the foreign process untouched; NP unchanged. Cost: squad k-chain 506 s + repair 42 s +
replay 72 s; musique k-chain 2,568 s (six PHG runs of 2.65 M objects, 147–183 s each, ≤ 331 MB / rank) + repair 315 s + replay 76 s.

**Twelfth ruling (2026-09-15, on this result).** V1 is frozen as `STRUCT_VERTEXCUT_V1: MetaQA mechanism positive / universal
replacement NOT_SUPPORTED`; it is not composed with SAFE or `H2_PATCH1`. The ruling reads the failure as specific — overlap was
optimised without a budget on overlap itself — and opens two new exploratory arms, both pre-registered before any run:
`BOUNDED_VERTEXCUT_R2` (§22: every node in at most two blocks, the same rule on every graph) and `BALANCED_H2_PATCH1` (§21: the
frozen beam's hop-2 candidate set compressed by a parameter-free per-parent rank interleave instead of the global top-100 cut).
Frozen `H2_PATCH1` stays untouched for the WebQSP transfer.

## 21. Twelfth ruling (2026-09-15), arm B: the frozen beam's hop-2 candidates compressed by a parameter-free per-parent rank interleave — `BALANCED_H2_PATCH1` (`_l1c_h2balanced.py`, `h2balanced_A_<cache>.{json,log}`, `h2balanced_A_SUMMARY.json`; pre-registered in `PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json` (v2; v1 in `_history/`), STATUS = POSTHOC_MECHANISM_TEST)

**Why this arm (the ruling's reading of §17).** The hop-2 failures of `H2_PATCH1` are not a reach problem: 173 / 174 (MtK) and
160 / 162 (PHG) of the SAFE failures at hop 2 have every missing gold node already scored as a hop-2 candidate of the pinned beam;
630 / 624 of the 1,169 / 1,139 missing gold nodes are candidates cut by the global top-100 width. The ruling therefore asks for a better
*compression* of the ~260 unique hop-2 candidates into the 100-node beam — not more traversal, not more candidates, not more exposure —
and forbids a local K (another constant): the parameter-free form is round-robin by within-parent rank (A1 B1 C1 … A2 B2 C2 … until the
global budget is full). The only constant remains the existing beam / patch capacity.

**The one change.** Everything of §16 / §17 is reused verbatim: the frozen five seeds, the §13 closed-form transition geometry and the
frozen RRF over (parent rank, candidate rank), depth 2, BEAM = 100, the canonical SAFE, the one-block patch of the same size as the
removed weakest block, the exact `|served nodes|` match, node-level ALL-gold. The pinned beam keeps, at each hop, the first 100 distinct
unvisited targets of the fused (parent, target) order. `rr_compress` replaces that global cut at both hops by: dedupe (parent, target)
keeping the first occurrence in the fused order; rank each parent's surviving candidates by that order; lexsort by (within-parent rank,
fused position); keep the first BEAM distinct unvisited targets. On metaqa hop 1 has ≤ 5 parents and ~35 (parent, target) pairs, so its kept set
is identical on 1,001 / 1,002 queries and the change acts at hop 2; on the text graphs hop 1 already exceeds the beam (155 / 131 unique
candidates) and its kept set differs on 613 / 544 queries (Jaccard 0.83 / 0.86). `H2_PATCH1` is rebuilt inside the same run and asserted equal to
`h2patch_A_<cache>.json` (ALL, ANY, per hop, novel counts) on all five caches.

**Pre-registration and its supersession (stated, not hidden).** v1 (17:50:31Z) was written before any run: arms, metric, the rule
(hop-2 ALL gain vs `H2_PATCH1` with gained > lost and p < 0.01 on both metaqa caches; text safety on squad / squad_phg / musique = no
significant loss vs `H2_PATCH1` and vs SAFE; no hop-1 / hop-3 regression), the compression function verbatim and its unit test. The first
metaqa run computed every arm number and then died at a *diagnostic* cross-check (the §17 population recomputed 172 vs the record's 173:
a missing gold node that is a frozen seed was classed "never scored", where §17 skips seed gold nodes). The fix touches only that
diagnostic loop; because the module sha was pinned, v1 went to `_history/` and v2 (17:54:28Z) carries the new sha, the reason and every
number that had been seen (`supersedes.numbers_seen_before_the_fix`), with the rule, arms, thresholds and compression function asserted
byte-identical. The metaqa (MtK) numbers below are therefore a reproduction of already-seen numbers; the other four caches were unseen.

**Result — five caches, DEV_A (`h2balanced_A_<cache>.json`, matched exposure asserted per query).**

| cache | L1 | SAFE | `H2_PATCH1` | `BALANCED_H2_PATCH1` | vs `H2_PATCH1` | vs SAFE |
|---|---|---|---|---|---|---|
| metaqa (MtK) | 0.6397 | 0.6407 | 0.7515 | **0.7764** | +30 / −5, p = 2.2e-5 | +136 / 0, p = 2e-41 |
| metaqa_phg | 0.6727 | 0.6747 | 0.7735 | **0.7984** | +31 / −6, p = 4.1e-5 | +125 / −1, p = 3e-36 |
| squad | 0.9859 | 0.9919 | 0.9919 | 0.9919 | 0 / 0 | 0 / 0 |
| squad_phg | 0.9829 | 0.9889 | 0.9909 | 0.9889 | 0 / −2, p = 0.5 | 0 / 0 |
| musique | 0.6596 | 0.7118 | 0.7329 | 0.7269 | +2 / −8, p = 0.11 | +15 / 0, p = 6e-5 |

Per hop (metaqa MtK / PHG, n 326 / 344 / 332): hop 1 0.9939 / 0.9969 = `H2_PATCH1` exactly (0 / 0); hop 2 0.7413 → **0.8140**
(+30 / −5) and 0.7442 → **0.8169** (+31 / −6); hop 3 0.5241 / 0.5843 = `H2_PATCH1` exactly (0 / 0). vs L1 at hop 2: +113 / 0 and
+99 / 0. Gold nodes gained vs SAFE 748 / 710 (`H2_PATCH1`: 634 / 613), lost with the removed block 8 / 7.

**Verdict by the pre-registered rule — PASS: `MECHANISM_CONFIRMED_ON_DEV_A (POSTHOC; not promoted)`.** hop-2 gain on both metaqa
caches (p < 0.01), no hop-1 / hop-3 regression, text safe (no significant loss vs `H2_PATCH1` or SAFE on squad / squad_phg / musique).
Not promoted, not composed, DEV_B unread.

**What the beam did differently (`beam_diversity_DEV_A`; same scored transitions and candidates per query — 37.8 / 334.9 on metaqa —
only the kept set changes).** metaqa hop 2: the kept set differs from the pinned one on 864 / 1,002 queries (Jaccard 0.595), distinct
parents in the kept beam 19.1 → 24.0, the largest single parent's share 0.234 → 0.105, 9.6 round-robin rounds fill the 100 slots;
wall-clock 17.6 → 13.5 ms (MtK) and 13.5 → 13.1 ms (PHG) — the interleave is a sort, not a traversal. Text hop 2: squad 17.9 → 36.9
parents (share 0.258 → 0.065, 6.9 rounds), musique 23.2 → 53.4 parents (share 0.254 → 0.039, 3.9 rounds; the kept set is different on
995 / 996 queries, Jaccard 0.243).

**Attribution (`attribution_flips_vs_H2_PATCH1`, `diagnostic_node_level_availability`).** Every flip is a beam flip, none a patch
flip: on metaqa the 30 newly covered queries have 58 gold nodes newly *visited* by the balanced beam (0 were visited by both beams and
cut by the patch), the 5 newly lost have 8 gold nodes no longer visited (0 beyond the patch capacity); PHG 31 / 61 and 6 / 9. On the §17
population (hop-2 SAFE failures whose missing gold nodes are all scored within depth 2 — 173 of 174, reproduced against
`h3patch_why_A_metaqa.json` with the seed case counted as in §17): all missing nodes visited by the pinned beam 85 → balanced 110
(PHG 74 → 99), and exactly those are covered by the respective patch (85 → 110, 74 → 99). Of the 1,169 missing hop-2 gold nodes, visited
536 → 650, scored-but-not-kept 630 → 516, never scored 2. The 64 remaining hop-2 failures (MtK; PHG 63) are still width: 516 / 525 missing gold nodes are scored candidates that are not
kept — the balanced beam keeps each parent's top ~4 (9.6 rounds) and the gold sits deeper in a productive parent's list.

**Text: why the interleave does nothing or slightly less.** On squad the patch rarely holds a gold at all (0 gold nodes gained vs SAFE
under either compression; 54–84 queries have no novel node). On musique the balanced beam visits *fewer* of the missing gold nodes
(hop-2 visited 18 → 11 of 366; failures with all missing nodes visited 21 → 14), and the 8 newly lost queries are all "no longer
visited": with ~77 productive parents and 1,252 unique hop-2 candidates per query, 3.9 rounds keep only each parent's top 3–4, so a
gold that was the 5th-best child of a strong parent (kept by the global cut) is dropped. The −6 net (+2 / −8, p = 0.11) is inside the
pre-registered safety clause but is the mechanism's cost on a dense graph: breadth across parents is bought with depth inside them.
The interleave is therefore a sparse-graph gain (branch diversity where one parent dominated the global cut) and neutral-to-slightly-
negative where the frontier is wide already. Nothing here changes served L1; `H2_PATCH1` stays the frozen reference for WebQSP.

**Not done, by the ruling.** No local K, no learned weight, no dataset constant; no change to depth, candidate generation or
exposure; no composition with §22; no DEV_B; no served cache or `data/` write. Cost: 53 + 46 + 256 + 244 + 298 s.

## 22. Twelfth ruling (2026-09-15), arm A: bounded-overlap static partitioning, every node in at most two blocks — `BOUNDED_VERTEXCUT_R2` (`_l1c_br2_lib.py`, `_l1c_br2_build.py`, `_l1c_br2_assign.py`, `_l1c_br2_replay.py`, `_l1c_br2_summary.py`; `parts/<ds>__BR2_OWNERS_k<k>.{npz,json}`, `…__PHG_con.{npy,RUN.json,log}`, `parts/<ds>__BR2_k<k>__R2.{npz,json,log}`, `br2_replay_A_{metaqa,squad,musique}.{json,log}`, `br2_SUMMARY.json`; pre-registered in `PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json` before any build; STATUS = POSTHOC_STRUCTURAL_MECHANISM_EXPERIMENT_DEV_A; post-hoc reach diagnostic `_l1c_br2_why.py`, `br2_why_A_metaqa.json`)

**The ruling's question.** §19 / §20 showed the unbounded vertex-cut wins on MetaQA (0.640 → 0.678 at P50, 0.688 matched) and becomes
a budget cut on the dense text graphs (RF 10.35 / 9.25; P50 = 43–63 % of the hard unique exposure; musique LOSS). The ruling's reading:
overlap was optimised without a budget on overlap. Bounded-overlap partitioning — 1 ≤ |M(v)| ≤ R with one small universal R, R = 2 as
the most defensible first value (owner + one alternate boundary region), min Σ_v (λ_v − 1) s.t. λ_v ≤ 2 plus balance / capacity — is
the natural successor. Question A: can we retain MetaQA's vertex-cut locality gain while preventing the RF explosion on dense text
graphs? One rule everywhere: no `if dataset`, no density threshold, no query information.

**The construction (universal; `_l1c_br2_lib.py`, docstring reproduced verbatim in the pre-registration).** Node-labelling relaxation
of the §18 edge partition: (1) *owners* = the validated Zoltan-PHG hard partition (NP = 4, the §18 parameters) of the primal STRUCT graph
restricted to nodes of degree ≥ 1 (objects = nodes, nets = edges with two pins, unit weights) into k = 2 k_f blocks (metaqa 864, squad
404, musique 2,350) — half-size owner blocks so that a block can hold its owners and their alternates under the existing cap C = 100;
(2) degree-0 nodes placed by §20's universal rule (`T.place_degree0`, unchanged; metaqa has none, squad 6,349, musique 9,858);
(3) *alternates* = one static greedy pass — gain(v, p) = number of v's STRUCT neighbours whose home is the foreign block p; nodes in
best-gain-descending order (ties by node id), candidate blocks in gain-descending order (ties by block id), gain ≥ 1 only, the first
block with live size < C takes v, otherwise v is refused. λ ≤ 2, |B_p| ≤ C and RF ≤ 2 hold by construction; the objective realised is
the maximum number of contained STRUCT edges under λ ≤ 2 and the cap (a node with degree > 2 (C − 1) = 198 necessarily keeps cut edges).
Representation, routing, serving and exposure are §19 / §20 unchanged: votes through the membership table with hubs → home only,
frozen RRF, P50, node-level ALL with gold served through M(g), actual unique exposure, the matched-budget cell. Pre-registered decision
rule (α = 0.01, McNemar): metaqa RETAINED iff `R2_MATCHED_LE` vs `HARD_MTK_BASE` = GAIN; text SAFE iff `R2_MATCHED_LE` vs the gate ≠
LOSS; EFFICIENT iff `R2_P50` ≠ LOSS and unique exposure ≤ hard; SUPPORTED = RETAINED ∧ both SAFE ∧ both EFFICIENT; PARTIALLY_SUPPORTED
= RETAINED ∧ both SAFE; else NOT_SUPPORTED. The matched cell is the signal cell (stated difference from §20, with the reason). Controls:
OWNERS_ONLY (the same owner partition, R = 1) and, on metaqa, §19's V1 recomputed and asserted equal to its frozen record.

**The substrates (`parts/<ds>__BR2_k<k>__R2.json`).**

| | metaqa k 864 | squad k 404 | musique k 2,350 |
|---|---|---|---|
| owner PHG run (rc 0; km1; max block / bound; wall) | 83,701; 52 / 52; 1.65 s partition inside a 726 s job (the WSL VM stalled at the mpirun launcher — RCU stalls in `dmesg`; the ranks then ran normally) | 610,801; 36 / 36; 402 / 404 blocks used → 2 empty blocks repaired by the frozen `PR.repair` (2 moves, km1 unchanged); 47 s | 2,337,486; 48 / 48; 78.5 s, ≤ 230 MB / rank |
| degree-0 placement (§20 rule) | none | 6,349: 5,294 anchored / 1,055 fallback (owner blocks 34.4 → 50.1 mean, max 93) | 9,858: 9,599 / 259 (45.8 → 50.0, max 67) |
| alternates assigned / refused for capacity | 22,860 / 5,998 (of 28,858 with a foreign neighbour) | 12,338 / 1,543 | 62,279 / 40,812 |
| RF; share λ = 2 (STRUCT nodes) | **1.529**; 0.529 | **1.610**; 0.889 | **1.530**; 0.578 |
| block size mean / max (at C) | 76.5 / 100 (228 blocks at C) | 80.6 / 100 (129) | 76.5 / 100 (837) |
| 1-hop STRUCT containment R2 / owners-only / served hard | **0.547** / 0.329 / 0.181 | **0.527** / 0.180 / — | **0.311** / 0.118 / — |
| non-hub 2-hop wedge containment (metaqa full metrics) | 0.219 (V1 0.265; hard 0.023; owners-only 0.078) | cheap metrics only | cheap metrics only |
| §2 ball containment (metaqa) | 0.173 (V1 0.183; hard 0.094; owners-only 0.123) | — | — |

The bound did what it was asked: RF 1.53–1.61 on every graph (V1: 2.34 / 10.35 / 9.25), no block above C, and a locality that is
1.7–2.9× the owners-only partition's and, on metaqa, 83 % of V1's wedge containment at 65 % of its replication.

**Result — DEV_A (`br2_replay_A_<ds>.json`; gate = each dataset's canonical served hard BASE; `br2_SUMMARY.json`).**

| dataset | gate | `R2_P50` (unique exposure) | vs gate | `R2_MATCHED_LE` (signal) | vs gate | owners-only P50 / matched | verdict |
|---|---|---|---|---|---|---|---|
| metaqa | `HARD_MTK_BASE` 0.6397 (5,023 nodes) | **0.3333** (4,068 = 81 %) | +65 / −372, p = 2e-53 | **0.3473** (62.0 blocks, 4,984 nodes) | +69 / −362, p = 5e-49 | 0.2265 / 0.2525 | `LOSS_EXPOSURE_LE_HARD|MATCHED_LOSS` — not RETAINED |
| squad | `HARD_MTK_BASE` 0.9859 (5,089) | 0.9899 (3,488 = 69 %) | +12 / −8, p = 0.50 | 0.9929 (74.8 blocks) | +13 / −6, p = 0.17 | 0.9869 / 0.9919 | `NEUTRAL_EXPOSURE_LE_HARD|MATCHED_NEUTRAL` — SAFE, EFFICIENT |
| musique | `HARD_PHG_BASE` 0.6596 (5,117) | 0.6647 (3,814 = 75 %) | +116 / −111, p = 0.79 | 0.6978 (68.1 blocks) | +131 / −93, p = 0.013 | 0.5823 / 0.6516 | `NEUTRAL_EXPOSURE_LE_HARD|MATCHED_NEUTRAL` — SAFE, EFFICIENT |

metaqa per hop at P50 (hard → R2): hop 1 0.9172 → 0.6319 (+15 / −108), hop 2 0.4855 → 0.2820 (+48 / −118), hop 3 0.5271 → 0.0934
(+2 / −146); vs V1 at P50 +39 / −384 (p = 2e-72), matched +43 / −384. The alternates are not inert — R2 beats the owners-only control
on every dataset at equal exposure (metaqa +114 / −19, p = 1e-17; musique +83 / −37, p = 3e-5; squad +3 / −2) — the owner partition
with R = 1 is simply far below the served hard BASE on metaqa (0.2265 at P50, half-size blocks voting only for themselves), and one
alternate per node closes a tenth of that gap.

**Verdict by the pre-registered rule — `NOT_SUPPORTED`** (metaqa not RETAINED; squad and musique SAFE and EFFICIENT). Nothing
promoted; §18–§20 unchanged; V1 stays frozen as ruled.

**Why MetaQA collapses and text does not (post hoc, reach only; `br2_why_A_metaqa.json`, v1 with the gold side measured through the
vote table superseded to `_history/`).** Two different spreads carry the served hard BASE's 0.64 and V1's 0.68, and R = 2 has neither:

| metaqa DEV_A, table | votes per top-100 hit (dense / SPLADE) | λ of gold nodes (mean) | gold node reached: hub / non-hub | all gold reached (all / hop2 / hop3) | coverage seen |
|---|---|---|---|---|---|
| hard, own block only (k 432) | 1.0 / 1.0 | 1.0 | 0.071 / 0.416 | 0.263 / 0.192 / 0.093 | 0.179 |
| owners-only k 864 (R = 1) | 1.0 / 1.0 | 1.0 | 0.279 / 0.372 | 0.305 / 0.262 / 0.111 | 0.227 |
| **R2 k 864** | 1.72 / 1.67 | 1.82 | 0.627 / 0.680 | **0.535** / 0.517 / 0.238 | 0.333 |
| served hard BASE (legacy table: own + directed out-neighbour blocks) | 2.54 / 3.37 | 1.0 | **1.000** / 0.646 | 0.713 / 0.590 / 0.611 | 0.640 |
| V1 k 1,004 (votes hub → home; served through all incident-edge blocks) | 3.20 / 2.28 | **43.7** | **1.000** / 0.794 | 0.801 / 0.756 / 0.654 | 0.678 |
| V1 served through the vote table only (not §19's rule) | 3.20 / 2.28 | 3.8 | 0.353 / 0.794 | 0.480 / 0.462 / 0.184 | — |
| R2 + the legacy out-neighbour spread (REACH ONLY, not a cell) | 4.38 / 5.55 | 1.82 | 0.980 / 0.867 | 0.807 / 0.823 / 0.605 | not evaluated |

Reach = every gold node of the query lies in at least one block that receives a vote from the top-100 hits (coverage at any budget
cannot exceed it). 38 % of MetaQA's gold nodes (1,543 of 7,239 on DEV_A) are hubs (degree > 100). The served hard BASE reaches every
hub gold through the *vote* spread — a hit votes for the blocks of its out-neighbours, so a hub's block collects the votes of every hit
that links to it. V1 reaches them through the *membership* spread — the mean gold node sits in 43.7 of V1's blocks (RF 2.34 over all
nodes), so it is served whenever any block holding one of its edges is selected; served through its home block only, V1's hub reach
drops to 0.353 and its all-gold reach to 0.480, i.e. below R2. §19's "locality gain at lower exposure" was replication concentrated
exactly on MetaQA's answer population — the RF explosion the R = 2 budget forbids, measured on the nodes that matter. R = 2 gives a hub
gold two blocks (reach 0.627) and a hit 1.7 votes, so 46.5 % of the queries are unreached before ranking starts; the hop-3 reach 0.238
against the hard 0.611 is the −43 pts at hop 3. On the text graphs no gold is far from its hits (reach 0.997 squad, 0.80 musique vs
V1-transfer's 0.80), the own-block vote already equals the BASE (§20), so a bounded substrate is judged on exposure alone — and there
it is level at 25–31 % fewer unique nodes, with musique +3.8 at the matched budget (p = 0.013, above α).

**What this settles.** Bounded overlap does prevent the text budget cut: R2 is the first overlap substrate that is SAFE and EFFICIENT
on both text graphs under one universal rule. It does not carry the MetaQA gain, and the reason is now exact: that gain was not
locality of the blocks but replication of high-degree gold nodes, which no λ ≤ 2 rule can reproduce, and MetaQA's served BASE gets the
same effect from its routing table (the legacy 1-hop vote spread), not from its partition. A bounded substrate served through the
frozen legacy spread (each hit voting for the R2 memberships of its out-neighbours) has reach 0.807 on metaqa — above the hard BASE's
0.713 and V1's 0.801 — but that is a *routing* change over the substrate, was not pre-registered, and its coverage is not computed here.
The reach table is the only post-hoc addition; no rule, R, k, C or serving variant was changed or added.

**Not done, by the ruling and the pre-registration.** No R = 3, no second alternate, no re-balancing after the alternates, no other k,
no density switch, no composition with SAFE / `H2_PATCH1` / `BALANCED_H2_PATCH1`, no coverage cell for any new serving rule, no DEV_B,
no served cache or `data/` write, the foreign process untouched, NP = 4 unchanged. Cost: builds < 2 s each; owner PHG 726 s (VM stall)
/ 47 s / 101 s; assignment 238 s (metaqa full structure metrics) / 2 s / 10 s; replay 80 / 54 / 69 s; diagnostic 10 s.

## 23. Thirteenth ruling (2026-09-16 local; records stamped 2026-09-15T19:11–19:16Z): the frozen bounded substrate served through the canonical router — `R2_LEGACY_ROUTER` (`_l1c_r2router.py`, `_l1c_r2router_summary.py`, `_l1c_r2router_prereg.py`, `_l1c_r2router_chain.sh`; `r2router_A_{metaqa,squad,musique}.{json,log}`, `r2router_SUMMARY.{json,log}`; pre-registered in `PREREGISTRATION_R2_LEGACY_ROUTER.json` before any ranking or coverage number of the arm existed; STATUS = POSTHOC_PARTITION_ROUTING_MECHANISM_TEST_DEV_A)

**The ruling.** "`BOUNDED_VERTEXCUT_R2` failed for a very specific reason: you removed the mechanism that gives hub answers enough routing
reach. The post-hoc reach diagnostic is strong enough to justify exactly one new preregistered arm: R2 geometry / memberships stay
frozen + the canonical frozen own + directed-out-neighbour vote spread + same P50 / same RRF. … The hypothesis is not 'R2 itself is
better.' It is: can bounded overlap reduce replication / exposure while the existing universal router supplies the hub reach that R2
cannot encode with λ ≤ 2? … partitioning and routing are separate mechanisms. … The gate should require no significant text
regression and a significant MetaQA gain or exposure-efficiency improvement. Run no R3 / R4 or alternate routing if it fails. The
post-hoc fact that R2 + legacy spread has reach 0.807, above both hard 0.713 and V1 0.801, is enough to justify that single test. It
does not justify claiming it will improve ALL-gold ranking." The same ruling froze the roles: **`H2_PATCH1` = transfer reference**
(the WebQSP primary transfer runs it exactly as §16 — not `BALANCED`); **`BALANCED_H2_PATCH1` = current best experimental successor**
(LEADING SUCCESSOR / MECHANISM_CONFIRMED_ON_DEV_A_POSTHOC / NOT YET TRANSFER-CONFIRMED; not run on WebQSP in the same pass; later on
another untouched, ideally KB-like, population); **`H3_PATCH1` = closed** (depth is not the current bottleneck); **`BOUNDED_VERTEXCUT_R2`
= partition-only hypothesis rejected**; **R2 + canonical router = the one justified new partition / routing hypothesis**; and
"don't combine `BALANCED` + R2-router yet — keep the mechanisms separate until each earns transfer evidence."

**The arm (`_l1c_r2router.py`; nothing built, nothing tuned).** The §22 substrate is read verbatim from the pinned
`parts/<ds>__BR2_k<k>__R2.npz` (metaqa k = 864, squad 404, musique 2,350; C = 100; RF 1.529 / 1.610 / 1.530; shas asserted). The only
change is the vote table: T(h) = V(h) ∪ ⋃_{t ∈ N_out(h)} V(t), where V(v) = {H(v)} for a hub and M(v) otherwise (§22's `mem_vc`) and
N_out is the canonical adapter's directed STRUCT out-neighbourhood — i.e. `_l1c_br2_why.spread_table`, the function whose reach
numbers justified the arm, imported unchanged and sha-pinned. Applied to a hard partition the same rule reproduces
`_l1s_core.Data.legacy_mem` — the served BASE's map — entry for entry (asserted on every gate partition: 109,790 / 101,222 / 522,798
entries). Routing (`partition_ranking` over the served top-100 ids, `rrf_ranks`, P50), serving (M(g) meets the selection), unique
exposure and matched prefixes are the pinned §19 functions. Cells: **`R2ROUTER_P50` (primary)**, **`R2ROUTER_MATCHED_LE` (secondary;
largest fused prefix whose unique exposure ≤ the gate's P50 exposure of the same query)**, `R2ROUTER_MATCHED_GE`, §22's
`R2_P50` recomputed and asserted equal to its record (the router's value on the same substrate), the **R = 1 control** (the same
owner partition without alternates under the same router: `OWNERS_LEGACY_P50 / _MATCHED_LE`; the alternates' value under the router),
the gate / contrast BASE; reach diagnostics for the four tables (metaqa asserted equal to `br2_why_A_metaqa.json`), the
UNREACHED / REACHED_WEAK decomposition of the primary cell's failures, failure overlap with the gate, and the ladder from the frozen
records. Pre-registered rule (α = 0.01, exact McNemar on DEV_A): TEXT_SAFE(ds) = P50 ≠ LOSS **and** matched ≠ LOSS vs the gate;
METAQA_GAIN = P50 = GAIN vs `HARD_MTK_BASE`; METAQA_EFFICIENT = P50 ≠ LOSS and mean unique exposure < the hard mean (stated in the
record as expected by construction on the exposure side, so decided by the coverage side); OUTCOME = PASS_GAIN (both text SAFE and
GAIN), PASS_EFFICIENCY (both text SAFE, EFFICIENT without GAIN), else FAIL — and FAIL means the ruling's letter: no R3 / R4, no
alternate routing. Before the record a dry run on metaqa built the tables and checked the two identities (no ranking, no coverage, no
text-set number); the record states every seen number (reach only) and that coverage of this table had never been computed.

**Result — `r2router_SUMMARY.json`: `FAIL`** (squad SAFE; musique NOT safe; metaqa LOSS at P50 and at the matched budget).

| DEV_A, ALL-gold | gate (P50 exposure) | `R2ROUTER_P50` (unique; ratio) | vs gate | `R2ROUTER_MATCHED_LE` (blocks) | vs gate | §22 `R2_P50` (no router) → router's value | `OWNERS_LEGACY` P50 / matched (R = 1 control) | alternates' value under the router P50 / matched | verdict |
|---|---|---|---|---|---|---|---|---|---|
| metaqa | `HARD_MTK_BASE` 0.6397 (5,023) | **0.5359** (4,353; 0.867) | +80 / −184, p = 1.3e-10 **LOSS** | 0.5469 (57.8) | +85 / −178, p = 1.0e-8 **LOSS** | 0.3333 → +215 / −12, p = 2.9e-49 GAIN | 0.4770 (2,557 nodes) / 0.5140 (98.0 blocks) | +75 / −16 GAIN / +75 / −42, p = 0.003 GAIN | `LOSS_EXPOSURE_BELOW_HARD\|MATCHED_LOSS` |
| squad | `HARD_MTK_BASE` 0.9859 (5,089) | 0.9778 (3,628; 0.713) | +5 / −13, p = 0.096 NEUTRAL | 0.9879 (72.1) | +7 / −5, p = 0.77 NEUTRAL | 0.9899 → +1 / −13, p = 0.0018 LOSS | 0.9718 / 0.9899 | +8 / −2 NEUTRAL / +1 / −3 NEUTRAL | `NEUTRAL_EXPOSURE_BELOW_HARD\|MATCHED_NEUTRAL` → SAFE |
| musique | `HARD_PHG_BASE` 0.6596 (5,117) | **0.5853** (4,366; 0.853) | +61 / −135, p = 1.3e-7 **LOSS** | 0.6135 (58.6) | +67 / −113, p = 7.5e-4 **LOSS** | 0.6647 → +62 / −141, p = 3.0e-8 LOSS | 0.4880 / 0.6104 | +107 / −10 GAIN / +67 / −64 NEUTRAL | `LOSS_EXPOSURE_BELOW_HARD\|MATCHED_LOSS` → NOT safe |

Contrasts (no verdict): metaqa vs `HARD_PHG_BASE` 0.6727: +74 / −211 at P50; squad vs `HARD_PHG_BASE` 0.9829: +7 / −12 (ns).
`R2ROUTER_MATCHED_GE`: 0.5479 / 0.9889 / 0.6145. Failure overlap with the gate at P50: metaqa both 281, router-only 184, hard-only 80;
squad 9 / 13 / 5; musique 278 / 135 / 61.

**MetaQA per hop (P50 vs `HARD_MTK_BASE`; matched in brackets).**

| hop | n | hard | `R2ROUTER` | vs hard | control (R = 1) | §22 (no router) | reach router / hard |
|---|---|---|---|---|---|---|---|
| 1 | 326 | 0.9172 | 0.9479 [0.9479] | +15 / −5, p = 0.041 (ns at α) | 0.8988 | 0.6319 | 0.997 / 0.945 |
| 2 | 344 | 0.4855 | 0.4593 [0.4797] | +59 / −68, p = 0.48 [+63 / −65] | 0.3547 | 0.2820 | 0.823 / 0.590 |
| 3 | 332 | 0.5271 | **0.2108** [0.2229] | +6 / −111, p = 4.0e-26 [+7 / −108] | 0.1898 | 0.0934 | 0.605 / 0.611 |

**Reach under each table (evidence through the vote table; gold served through the memberships; DEV_A).**

| dataset | table | votes / hit (dense / SPLADE) | blocks with evidence per query (of k) | reach all-gold | hub gold / non-hub gold reached | primary-cell failures: UNREACHED / REACHED_WEAK |
|---|---|---|---|---|---|---|
| metaqa | `HARD_LEGACY` (gate) | 2.54 / 3.37 | 219 of 432 | 0.713 | 1.000 / 0.646 | (hard failures 361) |
| metaqa | §22 membership-only vote | 1.72 / 1.67 | 237 of 864 | 0.535 | 0.627 / 0.680 | — |
| metaqa | **`R2_LEGACY_ROUTER`** | 4.38 / 5.55 | 423 of 864 | **0.807** | 0.980 / 0.867 | 465: **193 / 272** |
| metaqa | `OWNERS_LEGACY` (R = 1) | 2.78 / 3.84 | 310 of 864 | 0.634 | 0.920 / 0.600 | — |
| squad | `HARD_LEGACY` (gate) | 4.98 / 5.04 | 124 of 202 | 0.999 | 1.000 / 0.999 | — |
| squad | §22 membership-only vote | 1.38 / 1.36 | 107 of 404 | 0.997 | 0.995 / 0.997 | — |
| squad | **`R2_LEGACY_ROUTER`** | 5.94 / 5.99 | 202 of 404 | 0.999 | 1.000 / 0.999 | 22: 1 / 21 |
| squad | `OWNERS_LEGACY` (R = 1) | 5.42 / 5.51 | 192 of 404 | 0.997 | 1.000 / 0.996 | — |
| musique | `HARD_LEGACY` (gate) | 5.77 / 5.74 | 227 of 1,175 | 0.890 | 0.988 / 0.942 | — |
| musique | §22 membership-only vote | 1.51 / 1.50 | 169 of 2,350 | 0.801 | 0.922 / 0.907 | — |
| musique | **`R2_LEGACY_ROUTER`** | 8.78 / 8.88 | 469 of 2,350 | 0.882 | 0.991 / 0.940 | 413: 118 / 295 |
| musique | `OWNERS_LEGACY` (R = 1) | 7.63 / 7.75 | 400 of 2,350 | 0.816 | 0.981 / 0.903 | — |

**What the pre-registered cells say (no new arm, no new number).** Three decompositions on metaqa, all from cells in the record:
(1) *the router's value on the same substrate* — 0.333 → 0.536 (+215 / −12): the canonical spread restores 20.3 of the 30.6-pt §22
deficit, hop 1 0.632 → 0.948, hop 2 0.282 → 0.459, hop 3 0.093 → 0.211; hub-gold reach 0.627 → 0.980, exactly the mechanism the
ruling named. (2) *the alternates' value under the router* — 0.477 → 0.536 at P50 (+75 / −16), 0.514 → 0.547 at the matched budget
(+75 / −42, p = 0.003): real but small. (3) *the owner partition under the router* — the R = 1 control at its matched budget (98.0
half-size blocks, 4,998 unique nodes) is 0.514 against the served `HARD_MTK_BASE` 0.640 at 5,023 nodes with the *identical* router:
−12.6 pts at equal exposure, hop 3 0.190 vs 0.527. Most of the residual deficit therefore sits in the substrate's owner partition —
the k = 2 k_f half-size PHG partition of the primal STRUCT graph that the R = 2 budget requires — not in the routing. The hop-3
collapse is not a reach problem: the router's hop-3 reach (0.605) equals the hard BASE's (0.611) while coverage is 0.211 vs 0.527, so
the hop-3 failures are REACHED_WEAK — the answer sets (DEV_A queries hold 7.2 gold nodes on average, up to 119) are spread over roughly
twice as many half-size blocks, and 50 of 864 blocks (5.8 % of the blocks; 87 % of the hard exposure) cannot hold them; the matched prefix
(57.8 blocks) does not close it (0.223). The router's extra reach lands at hop 1 (0.997 vs 0.945) and hop 2 (0.823 vs 0.590), where
coverage is level or better (hop 1 +15 / −5, ns). On the text graphs the same router *over-spreads* a multi-membership table: 5.9
(squad) and 8.8 (musique) votes per hit against 5.0 / 5.8 for the served λ = 1 table and 1.4 / 1.5 for §22's membership-only vote;
the blocks with evidence per query double (202 of 404; 469 of 2,350) while reach does not move (0.999; 0.882 vs 0.890) — the ranking
is diluted, and the router's value is negative on both (squad +1 / −13, p = 0.002; musique +62 / −141, p = 3e-8). §22's
membership-only vote — SAFE and EFFICIENT — remains the better router for the bounded substrate on text, while the same spread is what
makes the served hard BASE work on MetaQA (own-block vote 0.18 → 0.64). Partitioning and routing are separate mechanisms, as the
ruling said; their pairing is not free: the spread that a λ = 1 table needs is too much for a λ ≤ 2 table on a dense graph and not
enough to make half-size owner blocks hold MetaQA's hop-3 answer sets.

**What this settles.** `R2_LEGACY_ROUTER` = **`FAIL`** by the pre-registered rule: no text regression on squad (SAFE at 71 % of the
hard exposure), a text regression on musique at P50 and at the matched budget, and on MetaQA a LOSS at both budgets (the 20-pt router
recovery leaves −10.4 pts at P50, all of it hop 3). Neither clause of the ruling's gate fires. By the ruling's letter — "Run no R3 / R4
or alternate routing if it fails" — the bounded-substrate line closes in this lane with §22 (partition only) and §23 (partition +
canonical router) as its record: no R = 3 / 4, no second alternate, no undirected / 2-hop / weighted / hub-rule-changed spread, no other
router on the bounded substrate, no re-partition of the owners. V1 stays frozen as "MetaQA mechanism positive / universal replacement
NOT_SUPPORTED"; the roles frozen above stand; nothing is composed; the WebQSP primary transfer remains frozen `H2_PATCH1` exactly as
§16. The one durable finding beyond the verdict: MetaQA's served BASE owes its multi-hop coverage jointly to the 100-node H4_SK
Mt-KaHyPar blocks *and* the legacy spread — a bounded substrate can borrow the spread but not the blocks.

**Not done, by the ruling and the pre-registration.** No rebuild or change of the substrate, R, C, k, K0 / K_LOCK / P_MAIN, RRF, hub
rule, serving, exposure or matched-budget rule; no other router; no composition with SAFE / `H2_PATCH1` / `BALANCED_H2_PATCH1`; no
DEV_B, no TEST, no WebQSP; no served cache or `data/` write; the foreign process and the WebQSP watcher untouched; the two new modules
unedited after the record (shas asserted at run time; `_l1c_br2_why.py`, `_l1c_br2_lib.py`, `_l1c_vcut_replay.py` pinned as imports).
Cost: dry run 8 s; replays 102 / 95 / 117 s; summary < 1 s.

## 24. Fourteenth ruling (2026-09-16 local; records stamped 2026-09-15T20:02–20:21Z), item 1: the one composition — `QMAX_F0` ranking → SAFE → `BALANCED_H2_PATCH1` = `QMAX_BALANCED_H2_PATCH1` (`_l1c_qmaxbal.py`, `_l1c_qmaxbal_summary.py`, `_l1c_qmaxbal_prereg.py`, `_l1c_qmaxbal_chain.sh`; `qmaxbal_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}`, `qmaxbal_SUMMARY.{json,log}`; pre-registered in `PREREGISTRATION_QMAX_BALANCED_H2_PATCH1.json` (sha 4230b261…) after an identity-only dry run and before any coverage number of a composed arm existed; STATUS = PREREGISTERED_COMPOSITION_TEST_DEV_A)

**The ruling.** "There are three real frontiers left. I would stop inventing new partitioners and focus on these." The failure map it
starts from: MetaQA `BALANCED` ≈ 79.8 % ALL-gold (bottleneck = frontier compression, then hop 3); MuSiQue best ≈ 73 % (ranking /
aggregation, not structural reach); SQuAD ≈ 99 % (ceiling; non-regression guard); WebQSP transfer pending; 2Wiki / Hotpot unused
(fresh text transfer populations). Item 1: "Combine the two mechanisms that solve different problems … the next universal candidate …
`QMAX_F0 ranking → SAFE → BALANCED_H2_PATCH1`. Same algorithm everywhere. No training. No relation dictionaries. No embedding
modification. … Do **one composition**, not a factorial of D2d/QMAX/interleave/etc. I'd use the cheaper `QMAX_F0` formulation because it
retained most of D2d's MuSiQue gain and does not need the five directional matrix scores. The scientific question is simply: Does
stronger semantic node→block aggregation and structurally balanced dynamic patching stack? If yes, that's probably your real universal
L1+ candidate." Items 2–4 (coverage-then-score frontier; an H3 successor only after the H2 compression is fixed; transfer = WebQSP for
the frozen structural mechanism, 2Wiki / Hotpot for the composite, SQuAD non-regression) are separate pre-registrations. The ruling
also states the caution this section inherits: "We have over-read DEV_A. So don't use MetaQA/MuSiQue indefinitely to select every next
rule."

**The composition (`_l1c_qmaxbal.py`; one substitution, nothing else).** The SAFE machinery of §16 / §21 (`_l1ps_router.build_cache`
+ `_l1kb_core.contexts` / `f6_select`) depends on the block ranking through exactly one array, `z["base_rank"]` — the challengers
(`s_node` = the frozen seed-based directional expansion; `ret_rrf`) are seed-based and the µL3 beam never reads the ranking (both facts
asserted on every cache: `spos` / `rpos` identical under the two rankings; the pinned beam re-run with candidate recording equals the
exec'd beam). The composed chain therefore substitutes the first TOPP = 200 blocks of the L1_GEOM `QMAX_F0` ranking (`_l1g_core.F0`
over `[Cd, Cs, Cmax]`: the served dense / SPLADE block channels through the legacy own + out-neighbour table, plus the exhaustive dense
block max `_l1g_candidate.dense_blockmax_rank`, frozen RRF K0 = 60, dense tie-break; `F0([Cd, Cs])` asserted equal to the served
`base_rank` on every row) for `base_rank` in the cache view, and runs the identical frozen path: protected = its first 44 blocks,
boundary = its blocks 45–50, F6 admission (B = 6, M_struct = 64, M_ret = 32, S4) → `QMAX_SAFE`; weakest = the sixth admitted; §16
patch = `QMAX_SAFE` minus the weakest block + one query-local block of the same size holding the §21 balanced beam's visited nodes
that the 49 kept blocks do not expose (first-visit order), topped up from the removed block (`ret_rrf` position, then id); unique
exposure matched to `QMAX_SAFE` per query (asserted). Reference chain L1 → SAFE → `H2_PATCH1` / `BALANCED_H2_PATCH1` recomputed and
asserted equal to the §16 / §21 records (ALL, ANY, gained / lost of every recorded comparison, per hop, novel counts, patch sizes); the
balanced beam asserted equal to every deterministic figure of `h2balanced_A_<cache>.json`; `QMAX_F0` asserted equal to its L1_GEOM
DEV_A record (metaqa 0.6327 +3/−10; musique 0.7319 +77/−5). Arms: L1, SAFE, `H2_PATCH1`, `BALANCED_H2_PATCH1` (parent 2),
`QMAX_F0` (parent 1), `QMAX_SAFE` (attribution stage), `QMAX_BALANCED_H2_PATCH1` (PRIMARY). Not run, by the ruling's "one
composition": `QMAX_H2_PATCH1` (would make a 2 × 2 factorial), D2d / directional compositions, interleaves, re-weightings, any
change of the compression (item 2 is separate). Dry run (metaqa, gold-free) before the record: QMAX P50 shares 36.45 of 50 blocks
with the BASE P50; `QMAX_SAFE` shares 38.68 of 50 with SAFE and is identical to it for 0 of 1,002 queries; same weakest block for 57;
admitted-six overlap 2.19; nominal exposure 5,023.0 / 5,023.1 / 5,028.3 / 5,027.3 (L1 / SAFE / `QMAX_F0` / `QMAX_SAFE`). Pre-registered
rule (α = 0.01, exact McNemar, labels GAIN / LOSS / NEUTRAL): per cache, CONFLICT if PRIMARY is LOSS vs either parent, STACKS if GAIN
vs both, CARRIES if GAIN vs one and NEUTRAL vs the other; OUTCOME = PASS_STACKS iff (a) no CONFLICT on any of the five caches AND (b)
PRIMARY vs `BALANCED_H2_PATCH1` = GAIN on musique AND (c) PRIMARY vs `QMAX_F0` = GAIN on both metaqa caches; FAIL_CONFLICT / FAIL_NOT_STACKED
otherwise; predictions stated in the record: musique 0.74–0.77 and GAIN vs `BALANCED`; metaqa within ±1 pt of `BALANCED`; squad neutral.

**Result — five caches, DEV_A (`qmaxbal_SUMMARY.json`: OUTCOME `PASS_STACKS`).**

| cache (n) | L1 | SAFE | `H2_PATCH1` | `BALANCED` | `QMAX_F0` | `QMAX_SAFE` | **PRIMARY** | PRIMARY vs `BALANCED` | PRIMARY vs `QMAX_F0` | PRIMARY vs `H2_PATCH1` | cell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa (1,002) | 0.6397 | 0.6407 | 0.7515 | 0.7764 | 0.6327 | 0.6377 | **0.7754** | +2 / −3, p = 1.0 NEUTRAL | +143 / 0, p = 2e-43 GAIN | +31 / −7, p = 1e-4 GAIN | CARRIES |
| metaqa_phg (1,002) | 0.6727 | 0.6747 | 0.7735 | 0.7984 | 0.6747 | 0.6687 | **0.7964** | +2 / −4, p = 0.69 NEUTRAL | +122 / 0, p = 4e-37 GAIN | +33 / −10, p = 6e-4 GAIN | CARRIES |
| squad (992) | 0.9859 | 0.9919 | 0.9919 | 0.9919 | 0.9919 | 0.9950 | **0.9950** | +3 / 0, p = 0.25 NEUTRAL | +3 / 0, p = 0.25 NEUTRAL | +3 / 0 NEUTRAL | NEUTRAL |
| squad_phg (992) | 0.9829 | 0.9889 | 0.9909 | 0.9889 | 0.9889 | 0.9899 | **0.9899** | +1 / 0 NEUTRAL | +1 / 0 NEUTRAL | +1 / −2 NEUTRAL | NEUTRAL |
| musique (996) | 0.6596 | 0.7118 | 0.7329 | 0.7269 | 0.7319 | 0.7480 | **0.7600** | +34 / −1, p = 2e-9 GAIN | +36 / −8, p = 3e-5 GAIN | +36 / −9, p = 7e-5 GAIN | STACKS |

Per hop on metaqa (PRIMARY / `BALANCED` / `QMAX_F0`): hop 1 0.9939 / 0.9939 / 0.9110 (vs `BALANCED` 0/0; vs `QMAX_F0` +27/0), hop 2
0.8081 / 0.8140 / 0.4797 (0/−2; +113/0), hop 3 0.5271 / 0.5241 / 0.5181 (+2/−1; +3/0); metaqa_phg hop 1 0.9969 / 0.9969 / 0.9172
(0/0; +26/0), hop 2 0.8198 / 0.8169 / 0.5494 (+1/0; +93/0), hop 3 0.5753 / 0.5843 / 0.5663 (+1/−4; +3/0). Rule components: (a) no
CONFLICT on any cache — true; (b) musique GAIN over `BALANCED` — true; (c) metaqa and metaqa_phg GAIN over `QMAX_F0` — true. Secondary
flags: ADDITIVE_ON_MUSIQUE = true (PRIMARY vs `QMAX_F0` GAIN on musique), ADDITIVE_ON_METAQA = false (NEUTRAL vs `BALANCED` on both
metaqa caches), SQUAD_NON_REGRESSION = true. Unique exposure: matched inside each chain by construction; across chains the nominal
exposure differs by < 0.25 % (metaqa 5,023 vs 5,027; musique 5,114 vs 5,104; squad 5,086 vs 5,074 — the QMAX chain is slightly
*smaller* on text).

**Stage increments (which stage carries which regime).** BASE chain: metaqa L1 → SAFE +3/−2 (0.6397 → 0.6407) → `BALANCED` +136/0
(0.7764); musique L1 → SAFE +56/−4 (0.6596 → 0.7118) → `BALANCED` +15/0 (0.7269). QMAX chain: metaqa `QMAX_F0` 0.6327 → `QMAX_SAFE`
+5/0 (0.6377) → PRIMARY +138/0 (0.7754); metaqa_phg 0.6747 → 0.6687 (+2/−8, ns) → 0.7964 (+128/0); musique `QMAX_F0` 0.7319 →
`QMAX_SAFE` +25/−9, p = 0.009 (0.7480) → PRIMARY +16/−4, p = 0.012 (0.7600; chain +36/−8). So on MetaQA the whole gain is the patch
(hop 2: 0.4797 → 0.8081, +113/0; 338 of the 359 gold nodes newly served vs `QMAX_F0` sit in the patch nodes, 21 in a kept block) and
the ranking is neutral — the composite lands 0.1 / 0.2 pt under `BALANCED` (2 queries newly covered by a kept block of `QMAX_SAFE`, 3 /
4 lost because the QMAX protected set no longer keeps a block that held 5 / 4 gold nodes the beam never visits); on MuSiQue all three
stages add: the aggregation (+7.2 over BASE), the SAFE swap over the aggregation ranking (+1.6; 39/−3 vs canonical SAFE) and the patch
(+1.2). The flips vs `BALANCED` on musique say where the +3.3 come from: 34 queries newly covered (37 gold nodes, 35 of them served by a
kept block of `QMAX_SAFE` — the ranking — and 2 by the patch nodes) against 1 lost; vs `QMAX_F0` 36 newly covered (38 gold nodes: 21 by
a kept block = the SAFE swap, 17 by the patch) against 8 lost (all eight: a block the QMAX P50 keeps that `QMAX_SAFE` drops, the gold not
visited by the beam). `QMAX_SAFE` vs canonical SAFE is NEUTRAL on metaqa (+5/−8; +7/−13) and GAIN on musique (+39/−3) and squad
0.995 (+3/0, the highest squad value among the arms of this lane). The composite's patch statistics equal `BALANCED`'s (metaqa: 78.7 novel
nodes per query, 731 gold nodes gained vs `QMAX_SAFE` for 211 queries, 14 lost with the removed block; musique 54.2 novel, 18 gained,
4 lost).

**What this settles.** By the pre-registered rule the composition **`PASS_STACKS`**: the two mechanisms do not interfere on any cache
(no CONFLICT), the aggregation's MuSiQue ranking gain shows through the structural chain (0.7269 → 0.7600, +34/−1) and the structural
hop-2 gain shows through the aggregation ranking (0.6327 → 0.7754 on metaqa, +143/0; 0.6747 → 0.7964 on PHG, +122/0). They stack in
the strict sense only on MuSiQue (STACKS); on MetaQA the composite *carries* `BALANCED`'s gain at `BALANCED`'s level (the aggregation
does not add there, as predicted), and on SQuAD both are at the ceiling. Recorded status, exactly as the record's `what_follows`:
**`QMAX_BALANCED_H2_PATCH1` = the universal L1+ candidate of the fourteenth ruling — MECHANISM_CONFIRMED_ON_DEV_A (composition; not
promoted; NOT transfer-confirmed)**. One algorithm on every cache, no training, no relation dictionary, no embedding modification,
no dataset branch: ALL-gold at P50-equivalent exposure metaqa 0.640 → 0.775, metaqa_phg 0.673 → 0.796, squad 0.986 → 0.995, squad_phg
0.983 → 0.990, musique 0.660 → 0.760 vs the served L1 (metaqa +138/−2, PHG +127/−3, squad +9/0, squad_phg +7/0, musique +101/−1). The
caveat is the ruling's own: DEV_A selected `BALANCED` (§21) and now labels the composite; `QMAX_F0`'s musique gain was DEV_B-confirmed as
a component (L1_GEOM), the composition itself has no held-out confirmation. Roles unchanged: `H2_PATCH1` stays the frozen WebQSP
transfer reference exactly as §16; the composite is the candidate for the multi-hop text transfer populations (2Wiki / HotpotQA) when
their substrates exist and a resource ruling allows it; its compression stays the §21 round-robin until item 2 earns its own verdict.

**Not done, by the ruling and the pre-registration.** No `QMAX_H2_PATCH1`, no D2d / directional composition, no interleave, no
re-weighting, no change of compression, patch, beam, F6, TOPP, K0 / K_LOCK / P_MAIN, B, M_struct, M_ret, RRF or hub rule; no DEV_B, no
TEST, no WebQSP / 2Wiki / Hotpot; no served cache or `data/` write; the foreign process and the WebQSP watcher untouched; the two new
modules unedited after the record (shas asserted at run time; `_l1g_candidate.py`, `_l1g_core.py`, `_l1c_microl3.py`, `_l1c_h2patch.py`,
`_l1c_h2balanced.py` and the SAFE / adapter modules pinned as imports). Cost: dry run 55 s; runs 54 / 46 / 362 / 454 / 217 s (the text
caches are dominated by the two beam re-runs; the exhaustive block max takes 24 s on musique with the fp32 node matrix loaded
transiently); summary < 1 s.

## 25. Fourteenth ruling, item 2 (records stamped 2026-09-15T20:18–20:35Z): the one coverage-then-score frontier — `COVSCORE_H2_PATCH1` (`_l1c_covscore.py`, `_l1c_covscore_summary.py`, `_l1c_covscore_prereg.py`, `_l1c_covscore_chain.sh`; `covscore_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}`, `covscore_SUMMARY.{json,log}`; pre-registered in `PREREGISTRATION_COVSCORE_H2_PATCH1.json` (sha fdac27a8…) after a unit test and a gold-free dry run and before any coverage number of the new arm existed; STATUS = PREREGISTERED_SUCCESSOR_TEST_DEV_A)

**The ruling.** Item 2 of the fourteenth ruling: "BALANCED is good, but its sparse/dense tradeoff can probably be improved once … I
would try exactly one parameter-free rule: **Coverage-then-score frontier**: First give every active parent one child (A1 B1 C1 D1 …),
then fill the remaining global budget using the original frozen global candidate ranking (A2 A3 B2 A4 …). No k_local. No λ. No learned
weighting. No dataset branch. B = {best child of each parent} ∪ TopK(remaining candidates) until capacity is reached. … I would not run
five diversity methods. One coverage-then-score successor is enough." Item 3 is conditional on this verdict: "Only after fixing H2
compression should you reopen H3 … If coverage-then-score or BALANCED keeps more correct H2 parents, then run **one H3_PATCH1
successor** using that frontier."

**The one change (`_l1c_covscore.py`, `cs_compress_core`; everything else verbatim §21 / §16).** Beam, admissibility, transition
geometry, RRF fused order, BEAM = 100, DEPTH = 2, frozen top-5 seeds, the §16 one-block patch (SAFE minus its weakest admitted block +
one query-local block of the beam's unexposed visited nodes in first-visit order, topped up from the removed block; unique exposure
matched to SAFE per query, asserted) and the SAFE machinery are the pinned code of §16 / §21; the pinned beam re-run with candidate
recording is asserted equal to the exec'd µL3 beam and the balanced beam asserted equal to every deterministic figure of
`h2balanced_A_<cache>.json` (only the wall clock is exempt). The compression at each hop: (1) (parent, target) pairs in the frozen fused
transition order, a pair reached twice keeping its first occurrence; (2) within-parent rank r = position among the parent's pairs in
that order (§21's definition); (3) phase 1 = the pairs with r = 0 in frontier order (A1 B1 C1 D1 …: the best child of every active
parent), phase 2 = all remaining pairs in the frozen global fused order (A2 A3 B2 A4 …); (4) the first 100 distinct unvisited targets
of phase 1 then phase 2, a target already taken being skipped, visited marked exactly as the pinned beam; a parent whose best child is
already taken contributes nothing in phase 1. §21 continues round-robin after round 1 (A2 B2 C2 … A3 B3 …); this rule fills the
remainder in the global order; both share round 1 exactly, and where a hop has at most 100 distinct unvisited candidates all three beams
(pinned, balanced, covscore) are identical. No constant beyond the frozen BEAM / DEPTH / K0 and the frozen patch budget. Five synthetic
examples are asserted at every start (one of them separates the rule from the round-robin: frontier A, B, C with A holding the best
global candidates gives `[a1, b1, c1, a2, a3]`, the round-robin `[a1, b1, c1, a2, b2]`). Dry run (metaqa, gold-free, before the
pre-registration): hop 1 is never truncated (32.5 candidates for 5 seeds; beam identical to the pinned beam for 1,002 / 1,002 queries
and to the balanced beam for 1,001); at hop 2 the covscore beam is far closer to the pinned beam than to the balanced one — Jaccard
0.935 vs pinned, 0.622 vs balanced; identical kept set for 295 / 138 of 1,002 queries; 866 queries truncated; of the 95.6 kept nodes
20.5 come from phase 1 (coverage) and 75.1 from phase 2 (score); distinct parents represented 23.0 (pinned 19.1, balanced 24.0);
largest single-parent share 0.226 (pinned 0.234, balanced 0.105). The pre-registration states the mechanism this implies and an
honestly uncertain direction: "if the [§21] gain came from round 1 (coverage), COVSCORE keeps it and may add the global fill's
precision (PASS_SUCCESSOR); if it came from the later rounds (the second and third children of the right parents), COVSCORE lands
between H2_PATCH1 and BALANCED … The dry run's identity (covscore closer to the pinned beam) makes the second reading more likely on
metaqa; text caches: within noise of BALANCED". Pre-registered rule (exact McNemar; gain α = 0.01, regression / safety 0.05):
PASS_SUCCESSOR iff hop-2 ALL of `COVSCORE_H2_PATCH1` vs `BALANCED_H2_PATCH1` is a GAIN on BOTH metaqa caches AND no hop-1 / hop-3
regression vs `BALANCED` on either AND no significant ALL loss vs `BALANCED`, `H2_PATCH1` or SAFE on squad / squad_phg / musique;
FAIL_WORSE if not PASS and (a LOSS vs `BALANCED` at hop 2 or on ALL on a metaqa cache, or a hop-1 / hop-3 regression, or a text-safety
violation); FAIL_NOT_BETTER otherwise; `frontier_for_item_3` = COVSCORE if PASS else `BALANCED_H2_PATCH1`.

**Result — five caches, DEV_A (`covscore_SUMMARY.json`: OUTCOME `FAIL_WORSE`).**

| cache (n) | L1 | SAFE | `H2_PATCH1` | `BALANCED` | **`COVSCORE`** | COVSCORE vs `BALANCED` | vs `H2_PATCH1` | vs SAFE | hop 2 (COVSCORE / `BALANCED` / `H2_PATCH1`; vs `BALANCED`) |
|---|---|---|---|---|---|---|---|---|---|
| metaqa (1,002) | 0.6397 | 0.6407 | 0.7515 | 0.7764 | **0.7535** | +5 / −28, p = 6.6e-5 **LOSS** | +4 / −2, p = 0.69 NEUTRAL | +113 / 0 GAIN | 0.7471 / 0.8140 / 0.7413; +5 / −28 |
| metaqa_phg (1,002) | 0.6727 | 0.6747 | 0.7735 | 0.7984 | **0.7764** | +5 / −27, p = 1.1e-4 **LOSS** | +4 / −1, p = 0.38 NEUTRAL | +103 / −1 GAIN | 0.7529 / 0.8169 / 0.7442; +5 / −27 |
| squad (992) | 0.9859 | 0.9919 | 0.9919 | 0.9919 | **0.9919** | 0 / 0 NEUTRAL | 0 / 0 | 0 / 0 | — |
| squad_phg (992) | 0.9829 | 0.9889 | 0.9909 | 0.9889 | **0.9909** | +2 / 0, p = 0.5 NEUTRAL | 0 / 0 | +2 / 0 | — |
| musique (996) | 0.6596 | 0.7118 | 0.7329 | 0.7269 | **0.7309** | +6 / −2, p = 0.29 NEUTRAL | 0 / −2, p = 0.5 | +19 / 0 GAIN | — |

Hop 1 and hop 3 on both metaqa caches are identical across `COVSCORE`, `BALANCED` and `H2_PATCH1` (0.9939 / 0.5241 and 0.9969 /
0.5843; 0 / 0). Rule components: hop-2 GAIN vs `BALANCED` on both metaqa caches — false (a LOSS on both); no hop-1 / hop-3 regression —
true; text safety vs `BALANCED` / `H2_PATCH1` / SAFE — true; metaqa LOSS vs `BALANCED` (hop 2 and ALL, p < 0.01) — true → **FAIL_WORSE**;
`frontier_for_item_3` = `BALANCED_H2_PATCH1`. ANY-gold: metaqa 0.987 (`BALANCED` 0.990, `H2_PATCH1` 0.983), metaqa_phg 0.988 (0.991,
0.983). Patch statistics of the new arm equal `H2_PATCH1`'s rather than `BALANCED`'s: metaqa 78.8 novel nodes per query (`H2_PATCH1`
79.2, `BALANCED` 76.5), 631 gold nodes gained vs SAFE for 195 queries (634 / 185; 748 / 204), 77 queries truncated (79; 56).

**Why the round-robin wins — the gold-free identity and the gold-bearing diagnostic agree.** At hop 2 on metaqa the covscore beam
represents almost as many distinct hop-1 parents as the balanced beam (23.0 vs 24.0; pinned 19.1) but its largest single-parent share
is the pinned beam's (0.226 vs 0.234; balanced 0.105): phase 1 spends 20.5 of the 95.6 slots on one child per parent and the 75 slots
of phase 2 go, in the frozen global order, to the same dominant parents the pinned beam favours — hence Jaccard 0.935 vs the pinned
beam. The gold diagnostic on SAFE's 174 hop-2 failures (1,168 missing gold nodes; PHG 162 / 1,139) says exactly that: missing gold
nodes visited at hop 2 by the pinned / balanced / covscore beam 536 / 650 / **534** (PHG 511 / 610 / 509); missing gold scored at hop
2 but cut by the compression 630 / 516 / 632 (624 / 525 / 626); gold-bearing hop-1 parents (a missing gold node among their hop-2
candidates) 450 on both, of which with a gold child kept 357 / 410 / 377 (420: 330 / 384 / 350); failures whose missing gold nodes
are all visited — every one of them covered by the arm — 85 / 110 / 87 (74 / 99 / 77). So the coverage phase does what it says (20
more gold-bearing parents get a child than under the pinned beam), but the gold of a MetaQA hop-2 question sits in the *second, third,
…* children of the right parents (the hop-2 answers are siblings under a few hop-1 entities), and the global-order fill hands those
slots back to the parents with the most candidates; the round-robin is the rule that keeps giving every parent its next child. The
flips are entirely the frontier, not the patch: vs `BALANCED` 5 queries newly covered (8 gold nodes, all newly visited by the covscore
beam) against 28 newly lost (53 gold nodes, all 53 no longer visited; 0 visited-but-beyond-capacity); PHG 5 / 27 (8 / 52); vs
`H2_PATCH1` +4 / −2 and +4 / −1. On the text caches hop 1 is where the rules differ (truncated for 655 squad / 557 musique queries):
covscore keeps the pinned hop-1 beam (Jaccard 1.0 / 0.999 vs pinned; 0.831 / 0.858 vs balanced — one child per seed, then the global
top; the round-robin's ~20 per seed), and at hop 2 on musique it is between the two (Jaccard 0.624 vs pinned, 0.395 vs balanced;
49.5 distinct parents vs 23.2 / 53.4; largest share 0.195 vs 0.254 / 0.039; phase 1 fills 44.7 of the 100 slots). Coverage follows:
musique 0.7309, between `BALANCED` 0.7269 and `H2_PATCH1` 0.7329 (+6 / −2 and 0 / −2, both ns), where 331–334 of the 366 missing gold nodes
of SAFE's 287 failures are never scored within depth 2 by any of the three beams (reach, as §24 and the ruling say — not
compression); squad's 8 failures are all unreachable within depth 2 and the three arms coincide at 0.9919; squad_phg +2 / 0 vs
`BALANCED` (= `H2_PATCH1`'s two queries).

**What this settles.** By the pre-registered rule **`FAIL_WORSE`**, and by the record's `what_follows`, verbatim: "the
coverage-then-score rule is recorded as worse than the round-robin on DEV_A and CLOSED (no k_local, no lambda, no other diversity
rule — 'one coverage-then-score successor is enough'); BALANCED_H2_PATCH1 keeps the frontier role; item 3 runs on the balanced
frontier". The pre-registration's second reading is what the data show: §21's hop-2 gain (+30 / −5, +31 / −6 vs `H2_PATCH1`) comes
from the later rounds of the round-robin, not from round 1, so the sparse end of the sparse / dense trade-off (every parent its next
child; largest-parent share 0.105) is the better compression on MetaQA hop 2 and any global-score fill after coverage re-concentrates
the budget on the parents the pinned beam already over-served. The compression trade-off is therefore not the open lever the ruling
hoped: `BALANCED`'s remaining hop-2 losses (516 / 525 scored-but-cut gold nodes on the two caches) are the WIDTH residual of §17 under
a better order, not a mis-ordering. Roles after this section: `BALANCED_H2_PATCH1` keeps the frontier role (best experimental
successor, MECHANISM_CONFIRMED_ON_DEV_A_POSTHOC as §21; NOT transfer-confirmed); `H2_PATCH1` stays the frozen WebQSP transfer reference
exactly as §16; the §24 composite keeps §21's compression; item 3 = one pre-registered `H3_PATCH1` successor on the balanced frontier
("No wider patch initially. No H3-specific scoring. Same real-edge maths.") is the next and separate record.

**Not done, by the ruling and the pre-registration.** No k_local, no λ, no weight, no second diversity rule, no dataset branch, no
change of beam, patch, F6, TOPP, K0 / K_LOCK / P_MAIN, B, M_struct, M_ret, RRF or hub rule; no DEV_B, no TEST, no WebQSP / 2Wiki /
Hotpot; no served cache or `data/` write; the foreign process and the WebQSP watcher untouched; the two new modules unedited after the
pre-registration (shas asserted at run time; `_l1c_microl3.py`, `_l1c_h2patch.py`, `_l1c_h2balanced.py`, the `_l1g` / `_l1s` / `_l1x90`
cores, `_l1ps_router.py`, `_l1kb_core.py` and the adapter / hypergraph modules pinned as imports). Cost: unit test + dry run 176 s; runs
53 / 48 / 206 / 203 / 284 s (the text caches are dominated by the three beam runs); summary < 1 s; the covscore beam's wall clock is the
pinned beam's (10–68 ms per query across the caches).

## 26. Fourteenth ruling, item 3 (records stamped 2026-09-15T21:00–21:50Z): the one `H3_PATCH1` successor on the balanced frontier — `BALANCED_H3_PATCH1` (`_l1c_h3bal.py`, `_l1c_h3bal_summary.py`, `_l1c_h3bal_prereg.py`, `_l1c_h3bal_chain.sh`; `h3bal_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}`, `h3bal_SUMMARY.{json,log}`; pre-registered in `PREREGISTRATION_BALANCED_H3_PATCH1.json` (sha 471e6918…) after a gold-free dry run and before any coverage number of the new arm existed; STATUS = PREREGISTERED_SUCCESSOR_TEST_DEV_A)

**The ruling.** Item 3 of the fourteenth ruling: "Only after fixing H2 compression should you reopen H3 … If coverage-then-score or
BALANCED keeps more correct H2 parents, then run **one H3_PATCH1 successor** using that frontier. No wider patch initially. No
H3-specific scoring. Same real-edge maths." The condition is met by `BALANCED_H2_PATCH1` (§25's diagnostic on SAFE's metaqa hop-2
failures: gold-bearing hop-1 parents with a gold child kept 410 vs 357 pinned of 450, PHG 384 vs 330 of 420; §21 hop 2 +30 / −5 and
+31 / −6 vs `H2_PATCH1`), and the coverage-then-score frontier failed (§25), so the frontier is the round-robin. The tenth ruling had
frozen H3 after §17 ("WIDTH = bottleneck, not widened"); this section is the one reopening the fourteenth ruling allows, on the one
frontier it names.

**The one change (`_l1c_h3bal.py`; everything else verbatim §17 / §21 / §16).** The pinned µL3 head is exec'd at DEPTH = 3 with
exactly §17's five substitutions (asserted equal to `PREREGISTRATION_H3_PATCH1_DEV_A.json` and to `h3patch_A_<cache>.json`), so the
pinned depth-3 beam, its transition counts (metaqa 37.8 / 334.9 / 1,020.2 per query, asserted equal to §17's `cost_depth3`) and
`H3_PATCH1` are recomputed; the balanced beam of §21 (`rr_compress`, copied verbatim: (parent, target) pairs in the frozen fused order,
within-parent rank r, sequence A1 B1 C1 … A2 B2 C2 …, the first 100 distinct unvisited targets) is run for three hops instead of two —
the same rule at hop 1, hop 2 and hop 3, no hop branch — so its hops 1–2 *are* the §21 beam (asserted: every deterministic figure of
`h2balanced_A_<cache>.json` `beam_diversity_DEV_A` reproduced, and `BALANCED_H2_PATCH1` rebuilt from them equal to its record: ALL,
ANY, vs SAFE / `H2_PATCH1` / L1, per hop, novel counts, gold gained / lost, patch size and composition; likewise `H2_PATCH1` vs
`h2patch_A_<cache>.json` and `H3_PATCH1` vs `h3patch_A_<cache>.json`). The patch is §16 / §17's (`build_patches` of `_l1c_h3patch.py`,
copied): SAFE minus its weakest admitted block + one query-local block of the same size holding the beam's visited nodes not exposed by
the 49 kept blocks in first-visit order — hop 1, hop 2, then hop 3, the order §17 pre-registered, which preserves the depth-2 patch and
admits hop-3 nodes only into the capacity the depth-2 novel nodes leave — topped up from the removed block (ret_rrf position, then node
id); unique exposure matched to SAFE per query, asserted. No wider patch, no H3-specific scoring, no hop-3 ordering rule, no new
constant, no dataset branch. Dry run (metaqa, gold-free, before the pre-registration): at hop 3 the balanced beam represents 77.2
distinct hop-2 parents (pinned 47.3) with a largest single-parent share of 0.024 (pinned 0.095) and a Jaccard of 0.29 vs the pinned
hop-3 beam (hop 1 / hop 2: 1.0 / 0.595 — the §21 figures); the round-robin at hop 3 uses 2.4 rounds on average (902 of 1,002 queries
stop in round 2: ~85 parents with candidates, 746 distinct candidates, 100 slots — about one child per parent, where hop 2 ran 9.6
rounds over ~25 parents); it scores fewer transitions than the pinned beam at hop 3 (965 vs 1,020) and is faster (52.8 vs 83.3 ms
per query). Patch-capacity arithmetic (gold-free): 145.7 novel nodes per query at depth 3 (hop 1 / 2 / 3 = 8.3 / 68.2 / 69.2) against
a 100.1-node patch, so 23.5 hop-3 nodes per query are admitted and 45.7 cut; 66 queries admit no hop-3 node; 969 queries truncated
(depth 2: 56). The pre-registration states the prediction from this arithmetic: "FAIL_NOT_BETTER is the most likely outcome (hop-3 ALL
vs BALANCED_H2_PATCH1 within +0 / +3 on each metaqa cache …); FAIL_WORSE is unlikely (hops 1–2 unchanged up to fill displacement …);
PASS_H3 would require the balanced hop-2 frontier to expose the hop-3 gold siblings that the pinned frontier never scored AND the
one-block capacity to hold them … The informative output of this test is the hop-3 reach diagnostic under the balanced frontier
(never scored / scored-but-cut / visited), which decides whether the hop-3 residual is FRONTIER (fixable by order) or WIDTH / CAPACITY".
Pre-registered rule (exact McNemar; gain α = 0.01, regression / safety 0.05): PASS_H3 iff hop-3 ALL of `BALANCED_H3_PATCH1` vs
`BALANCED_H2_PATCH1` is a GAIN on BOTH metaqa caches AND no hop-1 / hop-2 regression vs `BALANCED_H2_PATCH1` on either AND no
significant ALL loss vs `BALANCED_H2_PATCH1`, `H2_PATCH1` or SAFE on squad / squad_phg / musique; FAIL_WORSE if not PASS and (a LOSS vs
`BALANCED_H2_PATCH1` at hop 3 or on ALL on a metaqa cache, or a hop-1 / hop-2 regression, or a text-safety violation); FAIL_NOT_BETTER
otherwise. `BALANCED_H3_PATCH1` vs `H3_PATCH1` (the frontier effect at depth 3) is descriptive only.

**Result — five caches, DEV_A (`h3bal_SUMMARY.json`: OUTCOME `FAIL_NOT_BETTER`).**

| cache (n) | L1 | SAFE | `H2_PATCH1` | `H3_PATCH1` | `BALANCED_H2` | **`BALANCED_H3`** | BAL_H3 vs `BALANCED_H2` | vs `H3_PATCH1` | vs `H2_PATCH1` | vs SAFE | hop 3 (BAL_H3 / `BALANCED_H2` / `H3_PATCH1`; vs `BALANCED_H2`) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa (1,002) | 0.6397 | 0.6407 | 0.7515 | 0.7525 | 0.7764 | **0.7794** | +3 / 0, p = 0.25 NEUTRAL | +32 / −5, p = 7.4e-6 GAIN | +33 / −5 GAIN | +139 / 0 GAIN | 0.5331 / 0.5241 / 0.5271; +3 / 0 |
| metaqa_phg (1,002) | 0.6727 | 0.6747 | 0.7735 | 0.7735 | 0.7984 | **0.8024** | +4 / 0, p = 0.125 NEUTRAL | +35 / −6, p = 4.9e-6 GAIN | +35 / −6 GAIN | +129 / −1 GAIN | 0.5964 / 0.5843 / 0.5843; +4 / 0 |
| squad (992) | 0.9859 | 0.9919 | 0.9919 | 0.9919 | 0.9919 | **0.9909** | 0 / −1, p = 1 NEUTRAL | 0 / −1 | 0 / −1 | 0 / −1 | — |
| squad_phg (992) | 0.9829 | 0.9889 | 0.9909 | 0.9909 | 0.9889 | **0.9879** | 0 / −1, p = 1 NEUTRAL | 0 / −3, p = 0.25 | 0 / −3 | 0 / −1 | — |
| musique (996) | 0.6596 | 0.7118 | 0.7329 | 0.7359 | 0.7269 | **0.7269** | +3 / −3, p = 1 NEUTRAL | +3 / −12, p = 0.035 | +4 / −10, p = 0.18 | +18 / −3 GAIN | — |

Hop 1 and hop 2 on both metaqa caches are identical to `BALANCED_H2_PATCH1` (0.9939 / 0.8140 and 0.9969 / 0.8169; 0 / 0 — no fill
displacement on metaqa, exactly as §17 saw for the pinned beam), so the whole difference vs `H3_PATCH1` (+32 / −5, +35 / −6) is §21's
hop-2 gain carried to depth 3 (hop 2 +30 / −5, +31 / −6), and the depth itself adds +3 / 0 and +4 / 0 at hop 3. Rule components: hop-3
GAIN vs `BALANCED_H2_PATCH1` on both metaqa caches — false (NEUTRAL on both); no hop-1 / hop-2 regression — true; text safety vs
`BALANCED_H2_PATCH1` / `H2_PATCH1` / SAFE — true; metaqa LOSS — false → **FAIL_NOT_BETTER**, as pre-registered. ANY-gold: metaqa 0.994
(`BALANCED_H2` +4 / 0), metaqa_phg 0.995 (+4 / 0), squad 0.9909 / squad_phg 0.9879 (= ALL, as always on SQuAD), musique 0.991 (unchanged). Patch of the new arm on metaqa: 145.7 novel
nodes per query (hop 1 / 2 / 3 = 8.3 / 68.2 / 69.2; `H3_PATCH1` 147.1 = 8.3 / 70.8 / 67.9), composition 8.3 / 67.6 / 23.5 novel + 0.7
fill (`H3_PATCH1` 8.3 / 70.1 / 20.9 + 0.8), 969 queries truncated (967); gold nodes gained vs SAFE 862 = 95 / 653 / 114 by hop
(`H3_PATCH1` 742 = 95 / 539 / 108; `BALANCED_H2` 748 = 95 / 653 / 0), 12 lost with the removed block (12). PHG: 814 = 101 / 609 / 104
(`H3_PATCH1` 714 = 101 / 512 / 101), 12 lost. The flips vs `BALANCED_H2_PATCH1` are all hop 3 and all frontier: metaqa 3 queries newly
covered (3 gold nodes, each entering through a hop-3 position, each newly visited by the balanced beam), 0 lost; PHG 4 / 0 (5 gold
nodes); on the text caches the only flips are fill displacement — squad 0 / −1 and squad_phg 0 / −1 (the one lost gold node was a fill
node from the removed block under `BALANCED_H2`, pushed out by the hop-3 novel nodes, which now take 32.2 of the 71–72 slots per query
that were fill under `BALANCED_H2`),
musique +3 / −3 (3 gained through hop-3 positions, 3 lost fill nodes) — the §17 text mechanism ("gold nodes previously restored by the
fill dropped") again, and within noise. musique's −12 vs `H3_PATCH1` (p = 0.035; descriptive, not a rule clause) is §21's known
musique −6 vs `H2_PATCH1` (0.7269 vs 0.7329, ns) plus the fill displacement: 11 of the 12 dropped gold nodes were beam-visited
positions under the pinned depth-3 beam that the balanced beam does not visit (musique's gold-bearing hop-2 parents with a gold child
kept: 19 of 53 pinned vs 10 of 38 balanced), 1 was fill; the balanced hop-3 beam on musique represents 77.2 parents (pinned 29.9) at
2.0 rounds and scores fewer transitions (4,252 vs 4,586 per query).

**Why the depth does not convert — the hop-3 reach diagnostic under the balanced frontier (`diagnostic_depth3_reach_vs_gold`).** On
SAFE's 159 metaqa hop-3 failures (2,597 missing gold nodes, 16.3 per failure; PHG 137 / 2,371), the balanced frontier *is* a frontier
effect at depth 3 — pinned → balanced: missing gold nodes visited within depth 3 261 → 310 (PHG 207 → 256); scored as a hop-3 candidate
but cut 655 → 800 (621 → 747); never scored within depth 3 1,673 → 1,479 (1,537 → 1,362); gold-bearing hop-2 parents (a missing gold
node among their hop-3 candidates) 736 → 813, of which with a gold child kept 274 → 358 (650 → 727; 235 → 289); failures with every
missing node scored within depth 3 33 → 53 (21 → 46); failures with no missing node visited at all 75 → 53 (58 → 35) — and it converts
3 / 4 queries because the residual is three things the frontier does not touch: (i) **reach** — 1,479 of 2,597 (57 %; PHG 57 %)
missing hop-3 gold nodes are never a candidate of any node in the balanced depth-3 beam (their hop-2 parent is not among the 100
kept, or the edge is not admissible); (ii) **width at hop 3** — 800 (31 %) are scored and cut, and the round-robin over ~85 hop-2
parents with 100 slots gives each parent one or two children (2.4 rounds; the largest-parent share 0.024) while a MetaQA hop-3 answer
set is again a sibling set under a few hop-2 parents (only 5 of 159 failures have every missing node visited: 4 covered, 1 beyond the
patch capacity; pinned 4: 2 / 2); (iii) **capacity** — the one-block patch admits 23.5 hop-3 nodes per query after the depth-2 novel
nodes, against 16.3 missing nodes per failure that must *all* be served. The three converted metaqa queries are the ones whose few
missing nodes the balanced hop-2 frontier put within one child of a kept parent. The hop-2 diagnostic (174 failures; 1,168 missing) is
unchanged from §25 for the pinned / balanced beams (visited 536 / 650, scored-but-cut 630 / 516, gold-bearing hop-1 parents with a
gold child kept 357 / 410 of 450; all-visited failures 85 / 110, every one covered), and hop-3 nodes never reach a hop-2 failure's
missing gold (0 / 0 first visited at hop 3) — the depth-3 arm cannot help hop 2 by construction (the depth-2 patch is preserved). Text:
squad's 8 failures and squad_phg's 9–10 are never scored within depth 3 by either beam; musique 316–317 of 366 missing gold nodes are
never scored within depth 3 (reach — the §24 / §25 statement), 41 of 287 failures have every missing node scored, all-visited failures
26 pinned / 18 balanced, every one covered by its arm.

**What this settles.** By the pre-registered rule **`FAIL_NOT_BETTER`**, and by the record's `what_follows`, verbatim: "depth stays
CLOSED also under the balanced frontier (the tenth ruling's 'H3 frozen' stands); BALANCED_H2_PATCH1 keeps the successor role; the
diagnostic's cause (frontier / width / capacity) is reported; a wider patch needs a separate ruling and is not run". The cause: the
hop-3 residual of §17 is not a frontier effect — the better hop-2 frontier moves 194 hop-3 gold nodes from never-scored to scored and
49 more into the beam, and 20 more failures become fully scored, but 57 % of the missing hop-3 gold is still unreachable from any
100-wide frontier, 31 % is cut by the 100-wide hop-3 beam that must now serve ~85 parents, and the one-block patch holds ~24 hop-3
nodes against 16 missing per failure. A hop-3 repair on MetaQA needs reach (a frontier wider than 100 at hop 2, or a different
admissibility) *and* a patch wider than one block — both are rulings, neither is an order rule, and neither is run here. Roles after
this section: `BALANCED_H2_PATCH1` keeps the successor role (MECHANISM_CONFIRMED_ON_DEV_A_POSTHOC as §21; NOT transfer-confirmed);
`BALANCED_H3_PATCH1` is recorded as NOT_BETTER on DEV_A (+3 / +4 at hop 3, text-safe) and is not a candidate for anything;
`H2_PATCH1` stays the frozen WebQSP transfer reference exactly as §16; the §24 composite (`QMAX_F0` → SAFE → `BALANCED_H2_PATCH1`)
keeps depth 2. The fourteenth ruling's three frontiers are now all answered on DEV_A — item 1 PASS_STACKS (§24), item 2 FAIL_WORSE
(§25), item 3 FAIL_NOT_BETTER (this section) — and what remains of it is item 4 (transfer: WebQSP for the frozen `H2_PATCH1`,
2Wiki / HotpotQA for the composite, SQuAD as the ceiling guard), blocked on the host and the resource ruling, and the MuSiQue line the
ruling names separately ("better scoring of evidence already present … iterative query-conditioned scoring, not another partition"),
which has no pre-registration and needs the user's go.

**Not done, by the ruling and the pre-registration.** No wider patch, no H3-specific scoring, no hop-3 ordering rule, no depth 4, no
second compression rule, no k_local / λ / weight, no dataset branch, no composition with `QMAX_F0`, no change of beam width, patch,
F6, TOPP, K0 / K_LOCK / P_MAIN, B, M_struct, M_ret, RRF or hub rule; no DEV_B, no TEST, no WebQSP / 2Wiki / Hotpot; no served cache or
`data/` write; the foreign process and the WebQSP watcher untouched; the two new modules unedited after the pre-registration (shas
asserted at run time; `_l1c_microl3.py`, `_l1c_h2patch.py`, `_l1c_h2balanced.py`, `_l1c_h3patch.py`, the `_l1g` / `_l1s` / `_l1x90`
cores, `_l1ps_router.py`, `_l1kb_core.py` and the adapter / hypergraph modules pinned as imports; 30 records read-only, sha-pinned).
Cost: dry run 194 s; runs 184 / 174 / 650 / 891 / 1,042 s (three depth-3 beams per cache: the pinned beam exec'd, the pinned beam
with candidate recording, the balanced beam; musique 338 / 344 ms per query per beam); summary < 1 s.

## 27. Fifteenth ruling (2026-09-16 local; records stamped 2026-09-15T22:26–23:29Z): the architecture frozen around the §24 composite — `FREEZE_QMAX_BALANCED_H2_PATCH1.json`, the 2Wiki / HotpotQA resource survey, the streamed stage 2 and its identity test, and the transfer pre-registration (`_l1c_transfer_blockmax.py`, `_l1c_transfer_composite.py`, `_l1c_transfer_repro_chain.sh`; `TRANSFER_RESOURCE_SURVEY_2WIKI_HOTPOTQA.json`, `PREREGISTRATION_TRANSFER_2WIKI_HOTPOTQA.json` (sha 3af9ad9b…), `transfer_repro_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}`; STATUS = FROZEN_LEADING_UNIVERSAL_CANDIDATE / PREREGISTERED_TRANSFER_SPLIT_A_ONE_SHOT / IDENTITY_TEST_STREAMED_STAGE2_DEV_A)

**The ruling (verbatim essentials; the full text is embedded in the freeze record).** "At this point I would freeze the architecture
around the §24 composite and stop opening new structural arms on DEV_A." The candidate chain is `QMAX_F0` / strong node → block
aggregation → canonical SAFE → `BALANCED_H2_PATCH1` → the final P50-equivalent context; "§26 tells us not to chase depth anymore."
First priority: independent transfer — "DEV_A is exhausted for mechanism selection."; WebQSP primary stays `H2_PATCH1` (§16) exactly
as ruled, one shot, the composite afterwards "as successor evidence, but not retroactively called the original confirmation"; "I would
authorize 2Wiki and HotpotQA next … Freeze the exact composite before building their replay infrastructure." (2Wiki: does QMAX +
BALANCED patch generalize to multi-hop text? HotpotQA: same under a second text topology; SQuAD: easy / near-ceiling safety.) Second
priority, MuSiQue ranking not on DEV_A: only if QMAX generalizes on 2Wiki / Hotpot, iterative query-conditioned rescoring of the
already retrieved candidate set ("No new nodes, no graph walk, no learned weights."), "future work until transfer says the QMAX effect
generalizes." Third priority, efficiency / gating (`QMAX + SAFE → cheap confidence → stop | BALANCED H2 PATCH`; Dense / SPLADE
agreement): "calibrate this after the final retrieval candidate is frozen." Closed permanently for now: PATH3 / PATH4 partitions, more
vertex-cut variants, R3 / R4 overlap, different halo voting, mass-normalized voting, relation-signature variants, latent offsets,
global triplet geometry, pure H3 depth, coverage-then-score frontier compression. Operative sentence: "**Freeze
`QMAX_BALANCED_H2_PATCH1` as the leading universal candidate. Stop selecting mechanisms on DEV_A. Finish the frozen WebQSP transfer,
authorize 2Wiki + Hotpot replay/transfer for the composite, and use those results to decide whether any further scoring refinement is
justified.**" If it transfers across WebQSP + 2Wiki + Hotpot: "consider the retrieval architecture essentially settled and move the
research effort to L2 and selective full-L3".

**Step 1 — the freeze (`FREEZE_QMAX_BALANCED_H2_PATCH1.json`, sha f81ebb84…, 2026-09-15T22:26:53Z, write-once).** Stages 1–7 verbatim
from the §24 pre-registration with every constant (K0 60 / K_LOCK 100 / P_MAIN 50 / TOPP 200 / SEED_K 5 / B 6 / BEAM 100 / DEPTH 2 /
cq 200), the §24 evidence copied from `qmaxbal_SUMMARY.json` (PASS_STACKS: musique STACKS 0.727 → 0.760, +34 / −1 vs BALANCED and
+36 / −8 vs `QMAX_F0`; metaqa / PHG CARRIES; squad NEUTRAL at 0.995), the roles (`H2_PATCH1` = WebQSP primary, one shot;
`BALANCED_H2_PATCH1` and the composite = successor evidence afterwards; the composite = the arm carried to 2Wiki / HotpotQA; served
BASE / SAFE unchanged as every transfer's reference), the populations (DEV_A exhausted for selection — readable only to assert a
reproduction or for a descriptive diagnostic; DEV_B sealed; TEST never; WebQSP = the KB transfer population under the armed watcher;
2wiki / hotpotqa = the text populations, one shot each, pre-registered before any partition, cache or replay number exists; the corpus
never subset by query), the three priorities and the closed list, the fifteenth ruling verbatim, and the pins (contract 35fdbcdef0…,
the 17 CONTRACT_FILES, the pinned modules, the records). Nothing was run for it.

**Step 2 — the resource survey before any infrastructure (`TRANSFER_RESOURCE_SURVEY_2WIKI_HOTPOTQA.json`, sha 6f01efa7…; nothing
built, nothing under `data/` written).** The transfer chain is H4_SK hypergraph → PHG (NP = 4, the validated substitute, inside the
WSL VM, with the pinned semantic gate and repair) → canonical replay cache → the composite. Counted from the canonical `keys.npz` with
the frozen H4_SPLIT_PRESERVE arithmetic (reproduced exactly against `H4_SK.json` for metaqa M 86,796 / P 531,006, squad 39,146 /
1,589,832, musique 250,732 / 6,014,938): webqsp M 5,224,363 / P 29,617,113 (k 25,928), hotpotqa M 9,955,983 / P 61,440,325 (k 52,333),
2wiki M 11,539,902 / P 91,353,972 (k 59,898). PHG memory from the two measured NP = 4 runs (squad R1 max rank 53.1 MB / summed 211.3 MB
at 1.59 M pins; musique R1 177.9 / 670.9 MB at 6.01 M pins; summed / max 3.77 — the SUMMED footprint binds the one WSL VM, NP is not
a lever): webqsp summed 3,122–3,304 MB (FEASIBLE once the host is clean; the WebQSP authorization's own derivation said ≤ 3.5 GB),
**hotpotqa 6,427–6,854 MB → INFEASIBLE_UNDER_PINNED_HYGIENE_CRITERION** (needs ≥ 9,596 MB WSL-available under the pinned 1.4× margin
against a 7,789 MB VM; a launch would run the VM at 82–88 %), **2wiki 9,534–10,192 MB → INFEASIBLE_EXCEEDS_WSL_VM** (needs ≥ 14,268 MB;
more than the whole VM). Host-Python stages: the pinned semantic gate 8.3–12.5 GB (hotpotqa) / 12.3–18.5 GB (2wiki) on a 15.69 GB host;
the replay builder takes the BIG path (fp16 shards memory-mapped, ~1 GB of arrays); **the pinned stage 2 of the composite
(`_l1g_candidate.dense_blockmax_rank` through the fp32 whole-corpus matrix) needs 32.2 / 36.8 GB and is impossible here** — a streamed
re-implementation reads the 16.1 / 18.4 GB fp16 shards once, holds a 2000 × k fp32 block-max matrix (0.42 / 0.48 GB) and costs
3.2e13 / 3.7e13 FLOP. Disk: 21.9 GB free against ~1.2–2.4 GB of artefacts per corpus. The populations are frozen by the builder's own
rule (seed 0, shuffle the eval rows, sorted first 2,000, drop rows without gold; A / B by sha1 parity): hotpotqa validation 7,405 → 2,000
(A 1,031 / B 969; `row_query_ids_sha256` e68f5061…; every query 2 gold; hop −1; types comparison / bridge, levels easy / medium /
hard), 2wiki dev 12,576 → 2,000 (A 1,004 / B 996; 1781c90d…; 1,556 queries with 2 gold, 444 with 4; types comparison / compositional /
inference / bridge_comparison). What is NOT a lever: NP, a cheaper partitioner, corpus subsetting, killing the foreign process. What is
the user's: raising the WSL memory cap (`.wslconfig` is a system setting), another machine for the partition stage (the replay cache
and the composite are then feasible here on a clean host with the streamed stage 2), or a separate explicit ruling on the partition
contract. Labels if never run: `HOTPOTQA_TRANSFER_NOT_RUN_RESOURCE_INFEASIBLE_ON_THIS_HOST` /
`2WIKI_TRANSFER_NOT_RUN_RESOURCE_INFEASIBLE_ON_THIS_HOST`.

**Step 3 — the streamed stage 2 (`_l1c_transfer_blockmax.py`) and what "identical" can mean on this BLAS.** `blockmax_scores_streamed`
computes the same quantity as the pinned function — max_{v ∈ B} q · e_v over ALL nodes, fp16 rows → fp32, unit query, stable
descending argsort with ties by block id — from the fp16 shards walked in position order in chunks of 40,000 rows (one canonical
shard), never holding the fp32 corpus; the pinned function is neither changed nor copied. Measured before anything was pre-registered
(metaqa, 200 rows, scipy-openblas 0.3.30 DYNAMIC_ARCH Haswell, 12 threads): a chunked `Q @ E[a:b].T` does NOT reproduce the
whole-matrix `Q @ E.T` bit-for-bit — OpenBLAS's column tiling at a call's tail (groups of 8 / 16 columns) and its thread partition of
the column range change the fp32 summation order for ~1 % of entries multithreaded (|δ| ≤ 2.4e-7 at |q · e| ~ 1, a few ulps of a
1536-term dot product), ~0.1 % single-threaded; the difference depends on the chunk start, the chunk length and the thread count, and
query batches m ≥ 199 equal m = 200. So bitwise identity of a streamed stage 2 is not attainable, and the pinned output is itself a
function of N and the thread count. The survey's "must reproduce bit-for-bit" clause is therefore replaced (the survey is not
edited) by a pre-registered acceptance gate on what matters: g1 max |Δ block max| ≤ 1e-5 (a logic bug — a wrong block assignment, a
missed shard — gives ≥ 1e-3; summation noise is ~1e-7), g2 on split A the rows whose `PRIMARY` (and `QMAX_F0`) coverage flips between
the streamed and the pinned chain ≤ max(2, 0.5 % of n_A), g3 the pinned chain reproduces the records (asserted); accepted only if g1–g3
hold on all five DEV_A caches, otherwise the transfer is not run under the record.

**Step 4 — the transfer runner (`_l1c_transfer_composite.py`, `_l1c_qmaxbal.py` copied verbatim: exec of the pinned µL3 head,
`block_channel` / `F0` / `build_cache` / `contexts` / `f6_select` / `rr_compress` / `build_patches` / `served_by` / `evaluate` /
`flip_table` / the §24 cell; no dataset branch, no new constant; transfer caches registered at run time by name without editing a
pinned module) and the pre-registration (`PREREGISTRATION_TRANSFER_2WIKI_HOTPOTQA.json`, 2026-09-15T23:12Z, written after a gold-free
metaqa dry run — F0(Cd, Cs) == the served `base_rank`; the pinned-stage-2 QMAX chain == the L1_GEOM record; the balanced beam == §21;
canonical SAFE == the L1_REPLAY record; the composed-chain identities of the streamed and the pinned chain equal to each other and to
§24 (36.45 / 0 / 38.68 / 57 / 2.19 / [4.59, 4.92]) — and before any identity-test record, any transfer cache and any number on either
corpus existed, asserted at write time).** Population: the frozen 2,000-row caches above, split A one shot (hotpotqa 1,031 rows, 2wiki
1,004), split B sealed, TEST never opened, per-type / per-level tables descriptive only (no hop labels). Arms at the same 50-block
budget: `PRIMARY` = the composite (decision arm), canonical SAFE (reference 1), L1 (reference 2), attribution stages `QMAX_F0` /
`QMAX_SAFE` and the BASE-chain `H2_PATCH1` / `BALANCED_H2_PATCH1` (successor evidence, descriptive), the §24 cell as a secondary
flag. Statistics: exact McNemar on split A, α 0.01, significant loss at 0.05; four decision comparisons in total (PRIMARY vs SAFE and
vs L1 per corpus). Rule fixed now: TRANSFER_PASS iff PRIMARY vs SAFE = GAIN and vs L1 = GAIN and no significant loss vs SAFE;
TRANSFER_FAIL iff LOSS vs SAFE or a significant loss; TRANSFER_NEUTRAL otherwise; across corpora TRANSFERS (both PASS) / PARTIAL (one
PASS, no FAIL) / DOES_NOT_TRANSFER (any FAIL or no PASS). Predictions stated now (basis musique: PRIMARY 0.760 vs SAFE 0.712 +49 / −1,
vs L1 0.660 +101 / −1; STACKS): PASS on hotpotqa (absolute ALL higher; the QMAX ranking gain shows, the patch smaller) and a weaker
PASS on 2wiki (22 % four-gold queries; compositional / inference types carry the gain); STACKS on hotpotqa, STACKS or CARRIES on
2wiki; a LOSS vs SAFE would mean the QMAX protected set drops paragraphs the served ranking kept. Resource gate: the PHG hygiene
criterion of the WebQSP authorization verbatim (NP = 4 never changed silently; the foreign process never signalled; `.wslconfig` not
mine), the survey's predicted requirements, the labels above plus `<DS>_TRANSFER_NOT_RUN_HOST_RESOURCE_CONTENDED`. Forbidden: any
change to the composite (including cq / BLOCK of the streamed stage), any selection on the identity-test numbers, any edit of the two
new modules (shas pinned and asserted at run time with every import, cache and record), split B / TEST / DEV_B, tuning or re-runs
after a number is seen.

**Step 5 — the identity test on the five DEV_A caches (`transfer_repro_<cache>.json`; the composite's DEV_A coverage was already
recorded in §24, so this is a reproduction under the only feasible memory path, not a selection).** Every assertion held on every cache: the BASE chain (L1 / SAFE / `H2_PATCH1` / `BALANCED_H2_PATCH1`) equals `h2patch_A` / `h2balanced_A` / `qmaxbal_A`, the QMAX chain under the PINNED stage 2 equals `qmaxbal_A_<cache>.json` (`QMAX_F0`, `QMAX_SAFE`, `PRIMARY`: ALL, ANY, every paired comparison, per hop, novel counts, patch size, gold accounting, the §24 cell), the balanced beam equals §21, canonical SAFE equals the L1_REPLAY record. Streamed vs pinned stage 2 (default OpenBLAS threads): squad and squad_phg **bitwise identical** (N ≤ 40,000 = one chunk = one sgemm call; 0 of 404,000 block maxima differ); metaqa 11,993 of 863,136 block maxima differ (max |Δ| 2.68e-7 = 9 ulp; 0 > 1e-6), 2 of 1,998 rows with a different stage-2 order, 0 with a different first-200 fused `QMAX_F0` order, 0 with a different fused P50 set; metaqa_phg 12,034 / 863,136 (2.68e-7), 4 rows / 1 row / 0; musique 17,260 of 2,350,000 (2.38e-7), 3 / 0 / 0. On split A the served sets of `QMAX_F0`, `QMAX_SAFE` and `PRIMARY` are identical on every row (1,002 / 1,002 / 992 / 992 / 996) and the coverage vectors flip **0 rows** for every arm on every cache: `PRIMARY` 0.7754 / 0.7964 / 0.995 / 0.9899 / 0.760, cells CARRIES / CARRIES / NEUTRAL / NEUTRAL / STACKS — the §24 record, exactly. Gate: g1 (≤ 1e-5) ✓, g2 (≤ 5 flips) ✓ with 0, g3 ✓ on all five → **STREAMED_STAGE2_ACCEPTED** (`transfer_repro_SUMMARY.json`, sha c106e978…; prediction met: 0 flips, ~1e-7, not bitwise on the multi-chunk corpora). The streamed stage 2 is therefore the transfer's stage 2, and the stage-2 numerics are now understood: the only non-reproducible part of the frozen composite is the fp32 summation order inside sgemm, which never reached a served block on 3,988 cache rows.

**WebQSP (unchanged; context).** The watcher (pid 23252) has taken 1,312+ preflight samples since the re-arm, 0 clean (host
available 0.6–1.9 GB, 14–44 k pages in / s, swap 4.6–5.3 GB; the foreign `m3b_run.py --stage screen`, pid 22760, at 5.8–7.7 GB RSS);
it lapses at 2026-09-16T11:38:18Z with `WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED` if the host stays contended. Nothing was launched.

**What this closes and what it leaves.** DEV_A mechanism selection is over: the lane's last DEV_A numbers are the §24 records and the
identity test's reproduction of them. The composite is frozen with its roles; 2Wiki and HotpotQA are authorized and fully
pre-registered, and both are resource-blocked on this host at the partition stage (and 2wiki also at the semantic gate), independently
of the foreign job; the streamed stage 2 is the one implementation the transfer needs that the pinned code could not provide, and its
acceptance is decided by the pre-registered gate on DEV_A. Next, in the ruling's order and none of it mine to start: the WebQSP
one-shot when PHG can run; a resource decision for the text corpora (WSL cap / another machine / a contract ruling); then the transfer
runs unchanged; the MuSiQue rescoring line and the gating calibration wait for the transfer verdict and their own rulings.

**Not done, by the ruling and the pre-registrations.** No new arm, constant, compression, patch, beam, admission, ranking or partition
rule; no DEV_B, no TEST, no split B of any corpus; no PHG launch, no H4_SK build, no replay cache of hotpotqa / 2wiki; nothing under
`data/` written; no pinned module or CONTRACT_FILE edited; the foreign process and the WebQSP watcher untouched; NP unchanged;
`.wslconfig` unchanged. Cost: freeze / survey writers < 1 min each (the survey's degree counts ~5 min over 3 key files); dry run 171 s;
identity test 172 / 63 / 176 / 179 / 300 s (streamed stage 2 alone 32 / 12 / 6 / 7 / 27 s vs the pinned 34 / 11 / 6 / 6 / 39 s; process RSS at the end 0.95 / 0.95 / 0.59 / 0.53 / 1.04 GB); summary < 1 s.

**Addendum (2026-09-16T11:39Z): the WebQSP re-arm lapsed.** The tenth ruling's 24 h re-arm (`PHG_WEBQSP_REARM.json`, armed 2026-09-15T11:38:18Z) reached its deadline with the host contended on every one of its samples: the frozen program's own `not_run` and `report` subcommands wrote `results/L1_LOWMEM/PHG_WEBQSP_RUNS.json` (sha 830f27ae…, utc 2026-09-16T11:39:08Z) and `PHG_WEBQSP_REPORT.json` (sha 36394c38…, STATUS `STOP_FOR_REVIEW`), DECISION `WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED`, superseding the 2026-09-14 outcome into `_history/` (ca3e8b87… / b62b0e4e…). Preflight history since the lane's first arm: 1,867 samples (2026-09-13T20:41:09Z → 2026-09-16T11:39:06Z), 0 clean, never a streak of 1; the last samples: host available 0.36–0.95 of 15.69 GB, pages_in/s 12,000–63,000, swap used 11.1–11.7 GB (the pagefile grew to 22.72 GB), WSL available 7.19 of 7.79 GB (the VM itself was never the constraint), disk free 2.0 GB (below the criterion's 8 GB — a new failing check; 21.4 GB were free twelve hours earlier). The original foreign pid 8628 is long gone; the same job re-launched as `m3b_run.py --stage screen` (pid 22760, 2026-09-15 20:11 local) and then `--stage seeds` (pid 24960, 2026-09-16 04:59 local, RSS ≈ 6.1 GB) and was never signalled. Nothing was launched: no mpirun, no PHG dump, no partition, no replay cache; the canonical webqsp partition and replay slots stay empty; the record's `integrity` block lists no frozen record, data pin or musique file changed. This is a host-resource outcome, not evidence about PHG or about `H2_PATCH1`: the WebQSP one shot (§16 exactly, one prereg) is still owed and still first in the ruling's order. A third arm is not mine to start — the tenth ruling authorised one re-arm — so the WebQSP lane now waits for the user: a new re-arm ruling once the foreign job's daily restarts stop (or the host gets ≥ 6 GB clean for three consecutive samples), another machine, or a change to the hygiene criterion, which is a contract-level decision.

## 28. The partition-prototype ladder and the IVF question (2026-09-25; records stamped 2026-09-25T16:00–20:12Z) — `PROTOTYPE_LADDER` (`_l1c_proto.py`, `_l1c_proto_summary.py`, `_l1c_proto_chain.sh`, `_l1c_proto_ivf_summary.py`; `proto_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}`, `proto_SUMMARY.{json,log}`, `proto_IVF_SUMMARY.json`; pre-registered in `PREREGISTRATION_PROTOTYPE_LADDER.json` (sha 533adfbd…) after two identity-only dry runs; re-centred by `PREREGISTRATION_PROTOTYPE_LADDER_IVF_ADDENDUM.json` (sha 063e423e…) before any squad_phg or musique record existed and before the squad record was opened; STATUS = PREREGISTERED_DIAGNOSTIC_DEV_A, no selection, no adoption)

**28.1 The request and the re-centring.** The request (2026-09-25T15:17Z; sha ae0921a8… of the verbatim text): *"How few fixed representatives per graph partition are sufficient to preserve the ALL-gold recall of our current node-level routing?"* The first pre-registration read "our current routing" as the frozen composite's stage 2 (QMAX(B, q) = max over the block's nodes of q·e_v): QMAX was the reference at every level and the headline sat at PRIMARY, where the dense and SPLADE node votes are still inside the fusion. The user then clarified (verbatim): *"the point of this wasnt to make qmax better it was to see if we can instead of node voting do an ivfesque approach for selecting the parititions"*. The addendum re-centres the **reading** and edits nothing: the original rule, code and summary stand and are reported in 28.4. The IVF endpoint is the ladder's ALONE level: blocks are ranked by the representative channel alone, the top 50 are probed (nprobe = 50 = the P50 budget) and every node of a probed block is served. The reference is node voting L1: dense and SPLADE top-100 hits vote for their own block and for the blocks of their directed STRUCT out-neighbours, rr(sum) + rr(max) per channel, frozen RRF. The dense node vote alone is the dense-only fairness reference. Disclosure at writing (16:18:32Z): the metaqa and metaqa_phg log lines had been read, and their numbers are quoted in the addendum. The squad record existed (written 90 s earlier) but had not been opened, and squad_phg and musique had not finished. The rule, per cache: PRESERVES_NODE_VOTE means net loss vs L1 ≤ TOL = max(2, round(0.005·n_A)) = 5, and not (lost > gained with p < 0.05). The headline m*_text is the smallest FPS m that preserves node voting on all three text caches.

**28.2 Design.** Every representative is built once per block from the block's own frozen fp16 vectors and never sees the query:
- CENT_MEAN = q·mean.
- CENT_NORM = q·norm(Σ norm e_v).
- FPS_m = the max over the medoid and m − 1 farthest-point prototypes (Euclidean, ties → lowest id).
- RAND_m = the max over the first m nodes of a seeded, nested per-block permutation.
- m ∈ {1, 2, 4, 8, 16, 32}.

Node-subset scores are QMAX restricted to the subset, computed from the same fp32 products as the accepted streamed stage 2. The following are asserted on every cache: the all-nodes subset equals stage 2 bit for bit; subset ≤ QMAX everywhere; subset = QMAX on blocks with |B| ≤ m; the subsets are nested.

There are four levels:
- ALONE: the IVF endpoint.
- F0: frozen RRF of [Cd, Cs, C_variant] → P50.
- SAFE: the frozen F6 swap on that ranking.
- PRIMARY: the §21 balanced patch on that SAFE, i.e. the composite end to end.

The QMAX and BASE chains reproduce `transfer_repro_<cache>.json` and the replay records (asserted).

**28.3 The IVF answer (the addendum's reading).** ALONE level, split A, ALL-gold; each cell is ALL (+gained / −lost vs node voting L1); ✗ = does not PRESERVE:

| cache | node vote (L1) | dense vote | SPLADE vote | CENT_MEAN | CENT_NORM | FPS_1 | FPS_2 | FPS_4 | FPS_8 | FPS_16 | FPS_32 | RAND_32 | QMAX_ALONE (ladder limit) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metaqa | 0.6397 | 0.6337 | 0.6128 | 0.1437 (+8/−505) ✗ | 0.1277 (+7/−520) ✗ | 0.1088 (+7/−539) ✗ | 0.1178 (+7/−530) ✗ | 0.1008 (+6/−546) ✗ | 0.1038 (+8/−545) ✗ | 0.1118 (+6/−535) ✗ | 0.1307 (+9/−519) ✗ | 0.1477 (+9/−502) ✗ | 0.1816 (+6/−465) ✗ |
| metaqa_phg | 0.6727 | 0.6577 | 0.6537 | 0.1188 (+13/−568) ✗ | 0.1317 (+12/−554) ✗ | 0.1118 (+5/−567) ✗ | 0.1198 (+5/−559) ✗ | 0.1018 (+5/−577) ✗ | 0.1048 (+5/−574) ✗ | 0.1138 (+6/−566) ✗ | 0.1367 (+9/−546) ✗ | 0.1427 (+8/−539) ✗ | 0.1956 (+7/−485) ✗ |
| squad | 0.9859 | 0.9829 | 0.9829 | 0.9093 (+3/−79) ✗ | 0.9083 (+3/−80) ✗ | 0.8155 (+2/−171) ✗ | 0.8256 (+8/−167) ✗ | 0.8548 (+10/−140) ✗ | 0.8982 (+9/−96) ✗ | 0.9435 (+10/−52) ✗ | 0.9728 (+8/−21) ✗ | 0.9425 (+9/−52) ✗ | 0.9879 (+10/−8) |
| squad_phg | 0.9829 | 0.9778 | 0.9788 | 0.9093 (+7/−80) ✗ | 0.9214 (+5/−66) ✗ | 0.8014 (+5/−185) ✗ | 0.8226 (+7/−166) ✗ | 0.8286 (+8/−161) ✗ | 0.8831 (+10/−109) ✗ | 0.9355 (+10/−57) ✗ | 0.9587 (+9/−33) ✗ | 0.9446 (+10/−48) ✗ | 0.9909 (+12/−4) |
| musique | 0.6596 | 0.6546 | 0.6004 | 0.3283 (+41/−371) ✗ | 0.4056 (+44/−297) ✗ | 0.2610 (+17/−414) ✗ | 0.2610 (+21/−418) ✗ | 0.2761 (+28/−410) ✗ | 0.3163 (+33/−375) ✗ | 0.3996 (+56/−315) ✗ | 0.5231 (+69/−205) ✗ | 0.4880 (+72/−243) ✗ | **0.7008 (+105/−64)** |

- **Text: `IVF_DOES_NOT_REPLACE_NODE_VOTING_ON_TEXT`.** No FPS_m with m ≤ 32 preserves node voting on any text cache (binding: all three). At m = 32 the quantizer already scores ≈ 32 % of N as prototypes (6,464 of 20,233 on squad; 37,294 of 117,534 on musique) and still loses 1.3 / 2.4 / 13.7 points: squad +8/−21 (p = 0.024), squad_phg +9/−33, musique +69/−205. Status on squad, squad_phg and musique: ONLY_THE_LADDER_LIMIT_PRESERVES.
- **KB: `NOT_EVEN_THE_LADDER_LIMIT_PRESERVES_NODE_VOTE`.** metaqa scores 0.10–0.18 against 0.6397 at every rung, and metaqa_phg 0.10–0.20 against 0.6727 (−460 to −580 queries).
  - By hop on metaqa (hop 1 / 2 / 3): node voting 0.917 / 0.486 / 0.527; the ladder limit 0.374 / 0.137 / 0.039; FPS_32 0.261 / 0.102 / 0.033.
  - The KB's gold blocks are not text-similar to the query (the node text is NAME_ONLY). They are reached through the out-neighbour votes, which no per-block representative can express.
  - Universal verdict: `IVF_DOES_NOT_REPLACE_NODE_VOTING_UNIVERSALLY`.
- **The ladder's limit on text.** Block-max dense alone (every node a representative) preserves node voting on squad (0.9879, +10/−8) and squad_phg (0.9909, +12/−4, p = 0.08). **On musique it beats node voting:** 0.7008 vs 0.6596 (+105/−64, p = 0.002), and +107/−61 (p = 0.0005) against the dense node vote. On the multi-hop text corpus, dense evidence at node level, aggregated per block by max, routes better than node voting — but only with every node scored. That is exhaustive dense, the opposite of an IVF saving, and it is already the composite's stage 2 (QMAX_F0).
- **Representatives.**
  - The centroid beats the medoid and every FPS_m up to m = 8 on text (musique CENT_NORM 0.4056 vs FPS_8 0.3163).
  - Farthest-point selection is barely better than random. FPS_32 vs RAND_32: squad 0.9728 vs 0.9425; squad_phg 0.9587 vs 0.9446; musique 0.5231 vs 0.4880; on metaqa RAND is ahead (0.1477 vs 0.1307).
- **Predictions (addendum).** T1 held: CENT_NORM does not preserve on squad or squad_phg. T2 held: nor on musique. **T3 failed**: it predicted that some FPS_m ≤ 32 would preserve on squad and squad_phg, and none does. T4 held: none on musique, so m*_text = None.

**28.4 The original QMAX-referenced question (the pre-registered rule; secondary).** PRESERVES means net loss vs the QMAX chain at the same level ≤ 5, and not (lost > gained, p < 0.05). m*(L) is the smallest FPS m that preserves at level L on all five caches.

| level | m* (FPS) | smallest preserving FPS m (metaqa / metaqa_phg / squad / squad_phg / musique) | binding | m* (RAND) | label |
|---|---|---|---|---|---|
| ALONE | None | none / none / none / none / none | all five | None | NO_FPS_LADDER_POINT_PRESERVES_QMAX |
| F0 | None | 1 / 1 / 4 / 2 / none | squad, musique | None | NO_FPS_LADDER_POINT_PRESERVES_QMAX |
| SAFE | 32 | 1 / 1 / 2 / 1 / 32 | musique | 32 | FPS_PRESERVES_QMAX_AT_m=32 |
| **PRIMARY (headline)** | **2** | 1 / 1 / 2 / 1 / 1 | squad | 16 | **FPS_PRESERVES_QMAX_AT_m=2**, non-monotone: musique FPS_8 fails (0.7460 vs 0.7600, +14/−28); flagged, not repaired |

Inside the fusion, the SAFE swap and the patch absorb most of stage 2's marginal value, so cheap representatives (even two) stay within tolerance of QMAX at PRIMARY. At F0 no rung preserves on musique: FPS_32 is 0.7038 vs 0.7319 (+13/−41) and keeps 61 % of QMAX's gain over no stage 2.

Predictions:
- Held: P1, P2b, P2c, P3b, P3c, P4, P5 and P6.
- **P2a failed**: on musique at PRIMARY, CENT_MEAN (+19/−13) and FPS_1 (+15/−18) do preserve.
- **P3a failed**: m*(PRIMARY) is 2, not 16 or 32.

**28.5 Diagnostics (descriptive).**
- **Outliers.** The gold nodes that QMAX serves and a representative drops are peripheral in their block, and they are usually the node that makes the block win under QMAX.
  - musique, ALONE, CENT_NORM: dropped (n = 400) vs retained gold nodes have median centroid-distance percentile 0.725 vs 0.376 (Mann-Whitney p = 1.6e-65), peripheral-quartile share 0.458 vs 0.127, and block-max-node share 0.777 vs 0.537.
  - squad shows the same (0.950 vs 0.447, p = 1.5e-27).
  - At FPS_32 the percentile gap closes (musique 0.431 vs 0.461, p = 0.75). The residual loss is then a prototype max that falls below the gold's own score (median block-score gap 0.155).
- **Cohesion (query-free).** On text the blocks are more cohesive than a size-matched random partition: resultant length squad 0.543 vs 0.409 and musique 0.546 vs 0.395; mean pairwise cosine 0.289 vs 0.159 and 0.295 vs 0.147. On the NAME_ONLY KB they barely are (metaqa 0.633 vs 0.610 and 0.395 vs 0.366): the KB blocks are structural, not semantic. 32 prototypes shrink the covering radius only to 0.77–0.79 of r_1 (random 0.82–0.89).

**28.6 What this closes and what it does not.**
- **Closed on DEV_A:** a fixed, query-free, dense-side coarse quantizer over the served partitions (≤ 32 representatives per block, nprobe = 50) as a replacement for node voting. On text it loses at every m, and on the KB it cannot reach the graph-carried golds at all.
- **Not tested (each needs a ruling):**
  - SPLADE-side block prototypes.
  - Membership-expanded prototypes: a block represented over its nodes plus the nodes whose STRUCT out-edges point into it — the IVF analogue of the out-neighbour vote.
- **The QMAX reading** is a calibrated efficiency observation (two representatives ≈ QMAX at PRIMARY, within tolerance, non-monotone), not a proposal. FREEZE f81ebb84 is unchanged.
- **Cost:**
  - Run times 299 / 239 / 410 / 513 / 719 s for metaqa, metaqa_phg, squad, squad_phg, musique; the balanced beam accounts for 38–273 s of each.
  - Process RSS at the end 0.17–0.73 GB; summaries < 1 s.
- **Hygiene:** nothing under `data/`; no pinned module or CONTRACT_FILE edited; the foreign processes untouched.

## 29. Do the partitions help at all over flat dense + SPLADE rankings? (2026-09-26 local; records stamped 2026-09-25T21:03–21:34Z) — `FLAT_VS_PARTITION` (`_l1c_flat.py`, `_l1c_flat_summary.py`, `_l1c_flat_chain.sh`; `flat_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}`, `flat_SUMMARY.{json,log}` (sha 3fedf574…); pre-registered in `PREREGISTRATION_FLAT_VS_PARTITION.json` (sha 8ef5134b…, 21:03:30Z) after four gold-free dry runs and before any flat coverage number of the real gold nodes existed; STATUS = PREREGISTERED_DIAGNOSTIC_DEV_A, no selection, no adoption)

**29.1 The question.** The user asked (verbatim): *"when we come back to our question do the partitions even help than purely having dense and splade rankings? have we measured that with our hypergraph approach?"* The answer to the second half was **no**:
- Every canonical-era comparison was partition against partition. The G2 exposure audit's CANON arm, for example, is itself a vote-ranked partition list.
- The one flat test (2026-08-10, `l1_candgen`) does not answer it. It used legacy partitions, pooled corpora (musique_clean N = 13,672 vs canonical 117,534), verbalised-triple MetaQA, a dense-only flat list and mean recall. It is disclosed as a prior in the pre-registration.

The follow-up asked for the results on the proper datasets "if we have the partitions ready". Canonical partitions and replay caches exist for MetaQA, SQuAD and MuSiQue only, and the pre-registration records the reason for each missing dataset:
- **WebQSP:** the PHG arm ran twice and ended `WEBQSP_NOT_RUN_HOST_RESOURCE_CONTENDED`. A third arm needs a ruling.
- **HotpotQA:** PHG is infeasible under the pinned hygiene criterion on the WSL VM, and the composite's stage 2 needs 32.2 GB of fp32 vectors.
- **2Wiki:** PHG exceeds the VM, and stage 2 needs 36.8 GB.

No partition was built for them: no cheaper partitioner, no corpus subsetting, no WSL cap change.

**29.2 Design.**

*Flat arms.* All are partition-free and exhaustive over all N nodes, computed from the frozen stored vectors, with no tuned constant:
- **FLAT_DENSE**: q·e_v from the same fp32 products as the accepted streamed stage 2. Its block maxima equal stage 2 bit for bit on every cache row (asserted).
- **FLAT_SPLADE**: the canonical sparse dot product. Only positive-score nodes are ranked.
- **FLAT_RRF**: the frozen node-level RRF (`_ta_prepartition.node_rrf`) extended to full depth. A node absent from SPLADE takes that list's depth, and ties go to dense order. It reproduces the served `ret_rrf` on every row whose top-200 lists match the served ones (asserted).

*Routes.* These are the served routes, reproduced and asserted against `transfer_repro_<cache>.json` and the replay record:
- **L1@P**: the top P blocks of the node-vote ranking F0(Cd, Cs), P ∈ {1, 2, 5, 10, 20, 50}; L1 = L1@50.
- **DENSE_VOTE / SPLADE_VOTE**: one vote channel alone.
- **PRIMARY**: the frozen composite QMAX_BALANCED_H2_PATCH1.

*Matched exposure.* For route R and query q, B_q(R) is the number of distinct nodes R serves. The paired flat arm serves its top B_q(R) nodes.

*Test and headline.* Exact McNemar on DEV_A split A, ALL-gold. "Gained" means the flat arm covers every gold node and the route does not. The labels at α = 0.01 are PARTITIONS_HELP, FLAT_BEATS_PARTITIONS and NO_SIGNIFICANT_DIFFERENCE. The headline per cache is L1@50 vs FLAT_RRF and PRIMARY vs FLAT_RRF.

*Gates.* Every record passed:
- stage-2 identity, bitwise;
- the reference ALL/ANY equal transfer_repro, and L1 and SAFE match the replay per query;
- the exhaustive dense and SPLADE top-100 sets agree with the served lists (mean ≥ 0.99999 on all five caches);
- `ret_rrf` identity on 964–976 rows per cache.

*Dry runs.* Four dry runs used stand-in gold nodes, and the pre-registration discloses each one. squad and musique ran module v1, metaqa ran v2, and metaqa ran the pinned v3 (27aafbfc…).
- v2 = v1 plus a descriptive `where_they_differ` fix: a never-ranked sentinel had printed as a depth quantile.
- v3 = v2 plus the code sha hashed at start. v1/v2 hashed the file at record assembly, so the musique dry record stamped v2 although it ran v1; the pre-registration corrects it with the evidence.
- Both version diffs were verified mechanically.
- A first squad attempt (NameError, before any statistic) is also disclosed.

**29.3 Headline (ALL-gold, split A; route vs flat at the route's per-query exposure; +gained/−lost = flat vs route).**

| cache | N | nodes served (mean) | L1@50 vs FLAT_RRF | PRIMARY vs FLAT_RRF |
|---|---|---|---|---|
| metaqa | 43,234 | 5,023 | 0.6397 vs 0.3922 (+50/−298, p 4e-44) **PARTITIONS_HELP** | 0.7754 vs 0.3922 (+18/−402, p 1e-95) **PARTITIONS_HELP** |
| metaqa_phg | 43,234 | 5,089 | 0.6727 vs 0.3932 (+38/−318, p 3e-56) **PARTITIONS_HELP** | 0.7964 vs 0.3922 (+14/−419, p 7e-105) **PARTITIONS_HELP** |
| squad | 20,233 | 5,089 | 0.9859 vs 0.9990 (+14/−1, p 1e-03) **FLAT_BEATS_PARTITIONS** | 0.9950 vs 0.9990 (+5/−1, p 0.22) NO_SIGNIFICANT_DIFFERENCE |
| squad_phg | 20,233 | 5,089 | 0.9829 vs 0.9990 (+16/−0, p 3e-05) **FLAT_BEATS_PARTITIONS** | 0.9899 vs 0.9990 (+9/−0, p 0.004) **FLAT_BEATS_PARTITIONS** |
| musique | 117,534 | 5,117 | 0.6596 vs **0.8815** (+254/−33, p 2e-43) **FLAT_BEATS_PARTITIONS** | 0.7600 vs **0.8815** (+159/−38, p 8e-19) **FLAT_BEATS_PARTITIONS** |

Cross-cache verdict on both headline pairs: **`REGIME_DEPENDENT`**.
- KB (metaqa, metaqa_phg): PARTITIONS_HELP.
- Text (squad, squad_phg, musique): FLAT_BEATS_PARTITIONS; the only exception is squad PRIMARY, which is not significant.

The flat side is partition-free: metaqa = metaqa_phg and squad = squad_phg give identical flat gold positions (asserted).

**29.4 Budget curve: L1@P vs FLAT_RRF at L1@P's exposure** (route / flat ALL; PART = partitions help, FLAT = flat beats; α 0.01).

| cache | P=1 (~100 nodes) | P=2 (~200) | P=5 (~510) | P=10 (~1,020) | P=20 (~2,040) | P=50 (~5,100) |
|---|---|---|---|---|---|---|
| metaqa | 0.339 / 0.023 PART (+17/−334) | 0.420 / 0.038 PART (+15/−398) | 0.507 / 0.107 PART (+19/−420) | 0.569 / 0.181 PART (+22/−411) | 0.607 / 0.279 PART (+39/−367) | 0.640 / 0.392 PART (+50/−298) |
| metaqa_phg | 0.299 / 0.023 PART (+18/−295) | 0.391 / 0.038 PART (+19/−373) | 0.531 / 0.107 PART (+20/−445) | 0.597 / 0.182 PART (+24/−440) | 0.646 / 0.280 PART (+25/−391) | 0.673 / 0.393 PART (+38/−318) |
| squad | 0.444 / 0.990 FLAT (+542/−0) | 0.586 / 0.994 FLAT (+406/−1) | 0.766 / 0.996 FLAT (+229/−1) | 0.870 / 0.998 FLAT (+127/−0) | 0.950 / 0.998 FLAT (+49/−1) | 0.986 / 0.999 FLAT (+14/−1) |
| squad_phg | 0.373 / 0.990 FLAT (+612/−0) | 0.524 / 0.994 FLAT (+466/−0) | 0.708 / 0.996 FLAT (+287/−1) | 0.839 / 0.998 FLAT (+158/−0) | 0.930 / 0.998 FLAT (+67/−0) | 0.983 / 0.999 FLAT (+16/−0) |
| musique | 0.012 / 0.490 FLAT (+477/−1) | 0.044 / 0.557 FLAT (+516/−5) | 0.140 / 0.663 FLAT (+539/−18) | 0.301 / 0.753 FLAT (+477/−27) | 0.477 / 0.802 FLAT (+365/−41) | 0.660 / 0.881 FLAT (+254/−33) |

All 30 cells are significant and one-directional per regime. The equal-recall exposure makes the size of the effect concrete (descriptive; fixed-budget flat values from the records):
- **musique:** FLAT_RRF reaches L1@50's 0.6596 with **500 nodes** (0.6596), not ~5,100, and passes PRIMARY's 0.7600 between 1,000 (0.7500) and 2,000 nodes (0.8012).
- **squad:** FLAT_RRF with **100 nodes** (0.9899) beats L1@50's 0.9859.
- **metaqa:** 20,000 flat nodes (46 % of the corpus: 0.6028) still fall short of L1@50 (0.6397).

**29.5 Each channel against its own flat list** (route vs flat at the route's exposure).

| cache | DENSE_VOTE vs FLAT_DENSE | SPLADE_VOTE vs FLAT_SPLADE |
|---|---|---|
| metaqa | 0.6337 vs 0.1936 (+39/−480) **PARTITIONS_HELP** | 0.6128 vs 0.3313 (+33/−315) **PARTITIONS_HELP** |
| metaqa_phg | 0.6577 vs 0.1936 (+31/−496) **PARTITIONS_HELP** | 0.6537 vs 0.3323 (+20/−342) **PARTITIONS_HELP** |
| squad | 0.9829 vs 0.9990 (+17/−1) **FLAT_BEATS_PARTITIONS** | 0.9829 vs 0.9980 (+16/−1) **FLAT_BEATS_PARTITIONS** |
| squad_phg | 0.9778 vs 0.9990 (+21/−0) **FLAT_BEATS_PARTITIONS** | 0.9788 vs 0.9980 (+20/−1) **FLAT_BEATS_PARTITIONS** |
| musique | 0.6546 vs 0.8715 (+244/−28) **FLAT_BEATS_PARTITIONS** | 0.6004 vs 0.7841 (+256/−73) **FLAT_BEATS_PARTITIONS** |

On MetaQA, FLAT_SPLADE beats FLAT_DENSE (0.33 vs 0.19 at ~5,000 nodes; NAME_ONLY text is lexical). On musique, fusion adds 1 point over dense alone at ~5,100 nodes (0.8815 vs 0.8715).

**29.6 Where the regimes come from.**
- **MetaQA by hop** (Mt-KaHyPar cache; route / flat; PHG is the same in every cell's direction):

  | pair | hop 1 (n 326) | hop 2 (n 344) | hop 3 (n 332) |
  |---|---|---|---|
  | L1@1 vs FLAT_RRF | 0.365 / 0.043 (+10/−115) | 0.294 / 0.017 (+4/−99) | 0.361 / 0.009 (+3/−120) |
  | L1@50 vs FLAT_RRF | 0.917 / 0.485 (+3/−144) | 0.485 / 0.401 (+29/−58, p 0.002) | 0.527 / 0.292 (+18/−96) |
  | PRIMARY vs FLAT_RRF | 0.994 / 0.485 (+0/−166) | 0.808 / 0.401 (+1/−141) | 0.527 / 0.292 (+17/−95) |
  | DENSE_VOTE vs FLAT_DENSE | 0.905 / 0.334 | 0.485 / 0.195 | 0.521 / 0.054 |

  A single block (~100 nodes) holds the whole 1-hop answer set for 36.5 % of hop-1 questions; the flat top-100 holds it for 4.3 %. The answer entities are NAME_ONLY neighbours of the topic entity. The text-only flat lists find the topic entity but not its neighbours. The routes reach them through STRUCT: the blocks are cut from the STRUCT+KNN hypergraph, the votes flow along directed STRUCT out-edges, and at PRIMARY the µL3 beam traverses real edges (hop 2 0.808 vs 0.485 at L1@50).
- **MuSiQue by number of gold paragraphs** (L1@50 vs FLAT_RRF): 2 golds 0.786 vs 0.934 (+99/−19); 3 golds 0.617 vs 0.885 (+89/−10); 4 golds **0.304 vs 0.696** (+66/−4). The route loses more the more golds a question has: whole blocks spend the budget on each gold's block-mates, and a gold whose block ranks low is lost. The flat list spends the budget node by node.
- **Exposure needed to cover every gold node, budget-free** (median [p25, p75] nodes):
  - L1 block ranking vs FLAT_RRF: musique 2,299 [924, 8,458] vs **109 [15, 1,001]** (flat needs fewer on 894 / 996 queries); squad 203 [103, 514] vs **1 [1, 2]** (989 / 992); metaqa **509 [104, 28,038]** vs 12,602 [1,664, 30,294] (flat needs more on 678 / 1,002).
  - The KB distribution is bimodal: a gold block is either near the top or very deep.
- **With §28:** on musique the best block selection measured (QMAX_ALONE, exhaustive block-max dense, 0.7008) stays 17 points under FLAT_DENSE at the same ~5,100 nodes (0.8715). On text, most of the loss is the block granularity itself, not the block ranking.

**29.7 Predictions (stated in the pre-registration).**
- **F1 held**: MetaQA, both caches and both headlines, PARTITIONS_HELP.
- **F2 held**: SQuAD never PARTITIONS_HELP.
- **F4 held**: text at every P ≤ 5, FLAT_BEATS_PARTITIONS on all nine cells.
- **F6 held**: MetaQA at every P ≥ 10, PARTITIONS_HELP.
- F3 and F5, the two musique headlines, were left unpredicted. Both came out FLAT_BEATS_PARTITIONS.

The legacy prior's direction (flat ahead on multi-hop text at small pools) holds. On the canonical full corpus with ALL-gold the gap is larger at comparable pools: musique −52 points at ~510 nodes vs the legacy −19.7 at ~600, and −22 at ~5,100 vs the legacy −1.65 at ~5,000.

**29.8 What this says, and what it does not.**
- **Says (DEV_A, three datasets):**
  - On the text corpora, the partition routes put fewer complete gold sets in front of L2 than a flat dense + SPLADE list of the same size. This holds at every budget, for node voting, for each vote channel alone and for the frozen composite. On musique it costs 22 points at L1@50 and 12 points at PRIMARY.
  - On the NAME_ONLY KB, the partition routes win by 25–28 points at L1@50 and 38–40 at PRIMARY, and hugely at tight budgets.
- **Does not separate "partitions" from "graph".** On the KB the routes carry STRUCT (blocks cut from it, out-neighbour votes, the µL3 beam), while the flat arms are text-only. A partition-free, graph-aware arm was not measured: flat seeds plus STRUCT expansion at matched exposure. The KB win is therefore partition-plus-graph routing over text-only ranking, not evidence that blocks are needed for the graph gain (cf. `results/L1_P90_EXPLOIT/`: the query-conditioned typed-path selector took metaqa ALL 0.64 → 0.97 at the same budget).
- **Not measured:**
  - Compute: the flat arms score every node. The composite's stage 2 already computes the exhaustive dense products; exhaustive SPLADE scoring is new. No latency claim either way.
  - The blocks' other roles: L2 locality features, L3 seeds, scoping speed.
  - Downstream QA.
  - DEV_B, TEST, 2Wiki, HotpotQA and WebQSP.
- **Decides nothing.** FREEZE f81ebb84 is unchanged. A dataset-agnostic L1 that keeps both regimes' gains would be a new mechanism — for example, the flat list and the partition/graph route sharing one exposure budget. With DEV_A exhausted for mechanism selection and DEV_B spent, both opening it and choosing where to validate it are rulings.
- **Cost:** 158 / 193 / 403 / 339 / 668 s for metaqa, metaqa_phg, squad, squad_phg, musique; the flat pass is 30–213 s of that. RSS ≤ 1.07 GB. Host at 0.37–1.51 GiB available and 10.0–13.0 GiB swap at the starts.
- **Hygiene:** nothing under `data/`; no pinned module or CONTRACT_FILE edited; the foreign processes untouched.

## 30. Are partitions necessary, or is the KB gain graph expansion? (2026-09-26 local; records stamped 2026-09-26T04:08–08:34Z) — the one isolated ablation `FLAT_BALANCED_H2` (`_l1c_flath2.py`, `_l1c_flath2_summary.py`, `_l1c_flath2_chain.sh`, `_l1c_flath2_agree_probe.py`; `flath2_G_{webqsp,hotpotqa,2wiki}.{json,npz,log}`, `flath2_E_{hotpotqa,2wiki}.{json,log}`, `flath2_E_webqsp.json` (its lines are in `flath2_G_webqsp.log`, §30.3), `flath2_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}`, `flath2_SUMMARY.{json,log}`; pre-registered in `PREREGISTRATION_FLAT_BALANCED_H2.json` v2 (sha f5f6741e…; v1 b35f954a…, 04:05:14Z, in `_history/` with byte-identical copies of the v1 module, summary and chain) before any FLAT_BALANCED_H2 number of real gold existed; STATUS = PREREGISTERED_ISOLATED_ABLATION: fresh split A one-shot, DEV_A non-selecting, no adoption)

**30.1 The ruling.** After §29 the user reopened the L1 freeze narrowly, for one question only (verbatim): *"Are partitions actually necessary, or is the KB gain just graph expansion/co-location?"* The ruling:
- defines the arm: the top (M − C) flat semantic nodes plus up to C novel BALANCED-H2 nodes, with any shortfall filled from the next flat-ranked nodes;
- excludes every partition device: *"No PHG. No H4_SK. No block ownership. No node→block voting. No SAFE. No QMAX block aggregation. And critically, no new scoring mechanism."*;
- frames two outcomes on MetaQA. If flat + H2 ≈ 0.78 against partition + H2 ≈ 0.80, most of the partition gain was semantic seeds plus real graph traversal, and *"we should seriously consider deleting partitioning from the universal retrieval architecture"*. If flat + H2 ≈ 0.60 against 0.80, *"partitions are doing something genuinely important: co-locating related answer sets"*;
- names the fresh populations (HotpotQA split A, 2Wiki split A, WebQSP train_holdout) and asks two questions of them: whether flat + graph keeps the flat advantage on text, and whether graph expansion rescues NAME_ONLY flat retrieval on a KB without partitions;
- keeps DEV_A and DEV_B closed for selection and asks for the hypothesis to be frozen before any run.

**30.2 Design** (constants in the pre-registration; none tuned after a number).
- **FLAT_RRF**: §29's frozen node RRF extended to full depth over all N nodes, fv = 1/(60 + p_dense) + 1/(60 + p_splade).
  - Products are exhaustive fp32.
  - A node with no positive SPLADE score takes that list's depth.
  - Ties go to dense order.
- **FLAT@M** = FLAT_RRF[:M].
- **FLAT_BALANCED_H2@M** = FLAT_RRF[:M − C] + the first C beam nodes not in that base + fill from FLAT_RRF[M − C:], giving exactly M nodes (asserted).
  - C = 100, one block-equivalent: cap = round(N / (N // 100)) = 100 on every corpus.
  - The beam is §21's frozen BALANCED_H2, unchanged and exec'd from the pinned function sources:
    - served-list seeds node_rrf(dense top-200, SPLADE top-200)[:5];
    - depth 2 over the directed STRUCT family, with in-edges at non-hubs only;
    - §13 transition geometry fused by RRF;
    - per-parent round-robin to BEAM 100 per hop.
  - The patch takes hop-1 nodes, then hop-2 nodes, in first-visit order.
- **Exact top-M on the fresh corpora** (2.6–6.0 M nodes): a two-sweep bounded algorithm (K_TOP 12,000 > K0 + 2M − 2).
  - It is not a new scorer.
  - Before any fresh run it was validated on every DEV_A split-A row against a full materialisation of the same products. The DEV_A records re-assert this: `F2 == FULL` on all five caches.
- **Wiring gate**: the exhaustive dense and SPLADE top-100 sets must overlap the served H100 retrieval caches by at least 0.90 on average.
  - A gold-free probe on 40 rows per fresh dataset measured 1.00 before freezing.
  - The runs measured webqsp 0.99991 / 1.0, hotpotqa 1.0 / 1.0 and 2wiki 1.0 / 1.0.
  - The seeds recomputed from the exhaustive lists equal the served seeds wherever both top-200 orders are identical (asserted on 730 / 938 / 927 rows).
- **Populations**: the frozen rule (seed-0 shuffle, first 2,000, sorted, rows with gold), split A = sha1(qid) even; split B sealed.
  - webqsp train_holdout: 786 of 1,503 (46 without gold dropped).
  - hotpotqa validation: 1,031 of 2,000.
  - 2wiki dev: 1,004 of 2,000.
  - Whole served corpora, no subsetting.
- **Tests**: exact two-sided McNemar on ALL-gold, α 0.01.
  - D1 (fresh, primary): FLAT@5000 vs FLAT_BALANCED_H2@5000.
  - D2 (DEV_A): PRIMARY (the frozen composite QMAX_BALANCED_H2_PATCH1, its §29 per-row indicator) vs FLAT_BALANCED_H2 at PRIMARY's per-query exposure B_q.
  - D3 (DEV_A): FLAT@B_q vs FLAT_BALANCED_H2@B_q.
  - Fixed M ∈ {500, 1000, 2000, 5000}, the strata, the patch statistics and ANY are descriptive.
- **Verdict rules**:
  - FRESH_VERDICT = NOT_UNIVERSAL if any D1 is LOSS; FLAT_BALANCED_H2_UNIVERSAL if none is LOSS and webqsp gains; NO_KB_GAIN otherwise.
  - DEV_A_KB_READING (non-selecting) = PARTITIONS_BEAT_FLAT_H2_ON_THE_DEV_A_KB if metaqa and metaqa_phg D2 are both LOSS; PARTITIONS_NOT_SHOWN_NECESSARY if both are NEUTRAL or GAIN; MIXED otherwise.
- **DEV_A identity gates** (every record passed):
  - the partition-free STRUCT arrays, queries, seeds and beam equal the pinned µL3 head and §21's bal1 / bal2 on every split-A row;
  - the flat pass reproduces §29's deepest gold position on every row;
  - FLAT@B_q and PRIMARY equal §29's PRIMARY pair.

**30.3 Incident and the v2 supersession.**
- **What happened.** In v1, `finish()` wrote the record and returned without exiting. The `--dry` path exits inside `finish()`, so the 11-step dry chain could not show it.
  - The G_webqsp process therefore fell through into the stage-E code.
  - It passed the E write-once guard, re-read its own just-written G record and arrays under the sha asserts a separate E process makes, computed stage E with the pinned v1 code, and wrote `flath2_E_webqsp.json`.
  - The chain's E_webqsp step then stopped at its write-once guard, and no other step ran.
- **What it affects.** The E record is computed exactly as a separate v1 E process would compute it.
  - Its `host_at_start`, `seconds`, `seconds_by_stage` and `process_rss_mb_at_end` describe the G process.
  - Its lines are in `flath2_G_webqsp.log`.
- **The v2 fix.**
  - The module adds one line (`sys.exit(0)` after the non-dry write). The dry path is unchanged, so the v1 dry runs stand.
  - The summary accepts the webqsp records under the v1 shas pinned in v2, and adds `--check-prereg`.
  - The chain runs the nine remaining steps.
- **Disclosure.** v2 was written after the webqsp D1 had been seen, and records it verbatim. v2 changes no design, constant, prediction, rule or population.
- **Before the relaunch.** All nine steps and the summary passed `--check-prereg`, including the v1 prereg, the v1 code copies and the four v1-produced records by sha.

**30.4 Headline: the fresh one-shot transfers.** D1 is pre-registered: ALL-gold on split A, with both arms at M = 5,000 nodes. +gained/−lost counts FLAT_BALANCED_H2 against FLAT.

| dataset (regime) | split | N (whole corpus) | n | FLAT@5000 | FLAT_BALANCED_H2@5000 | Δ pts | +/− | p | D1 |
|---|---|---|---|---|---|---|---|---|---|
| webqsp (NAME_ONLY KB) | train_holdout | 2,592,894 | 786 | 0.4962 | **0.7595** | +26.34 | +207/−0 | 9.7e-63 | **GAIN** |
| hotpotqa (text) | validation | 5,233,329 | 1,031 | 0.9331 | **0.9942** | +6.11 | +63/−0 | 2.2e-19 | **GAIN** |
| 2wiki (text, multi-hop) | dev | 5,989,847 | 1,004 | 0.6763 | **0.9213** | +24.50 | +246/−0 | 1.8e-74 | **GAIN** |

**`FRESH_VERDICT = FLAT_BALANCED_H2_UNIVERSAL`** (`flath2_SUMMARY.json`, sha b50155be…).
- No fresh D1 is LOSS, and webqsp gains.
- Across the three untouched populations, 516 queries gain and none loses.
- **The ruling's text question** (*"Does flat + graph preserve the enormous flat advantage on text?"*): yes, and the patch extends it.
- **The ruling's KB question** (*"Does graph expansion rescue NAME_ONLY flat retrieval on a KB without graph partitions?"*): yes, 0.4962 → 0.7595. No partition arm exists on webqsp to compare against (§30.7).

*Budget curve.* Descriptive, except the M = 5,000 column, which is D1. Each cell gives FLAT / FLAT_BALANCED_H2 ALL-gold and +gained/−lost. Every cell is significant in the patch's favour at α 0.01, and no cell loses more than 5 queries.

| dataset | M = 500 | 1,000 | 2,000 | 5,000 |
|---|---|---|---|---|
| webqsp | 0.3117 / 0.6641 (+281/−4) | 0.3690 / 0.6947 (+261/−5) | 0.4275 / 0.7239 (+234/−1) | 0.4962 / 0.7595 (+207/−0) |
| hotpotqa | 0.8274 / 0.9835 (+163/−2) | 0.8613 / 0.9884 (+131/−0) | 0.8904 / 0.9903 (+103/−0) | 0.9331 / 0.9942 (+63/−0) |
| 2wiki | 0.5448 / 0.9044 (+362/−1) | 0.5827 / 0.9104 (+329/−0) | 0.6285 / 0.9143 (+287/−0) | 0.6763 / 0.9213 (+246/−0) |

On all three fresh corpora, FLAT_BALANCED_H2 with **500** nodes covers more complete gold sets than FLAT with 5,000 (not a paired test).

*What the patch does* (M = 5,000).
- **On text it completes chains; it does not find new questions.**
  - ANY-gold is 0.9990 / 0.9990 on hotpotqa and 1.0000 / 1.0000 on 2wiki (FLAT / FLAT_BALANCED_H2). The flat list almost always holds one gold paragraph, and the real-edge hops bring the rest.
  - 2wiki, 2 golds (790 queries): 0.8241 → 0.9684 (+114/−0).
  - 2wiki, 4 golds (214 queries): **0.1308 → 0.7477** (+132/−0).
- **On the KB it also finds first golds.** webqsp ANY rises 0.7252 → 0.9326. By gold count:

  | golds | n | FLAT | FLAT_BALANCED_H2 | +/− |
  |---|---|---|---|---|
  | 1 | 401 | 0.6708 | 0.9252 | +102/−0 |
  | 2 | 100 | 0.5200 | 0.9000 | +38/−0 |
  | 3 | 70 | 0.3571 | 0.6429 | +20/−0 |
  | 4 | 38 | 0.4211 | 0.6842 | +10/−0 |
  | 5–10 | 89 | 0.2697 | 0.5843 | +28/−0 |
  | 11+ | 88 | 0.0455 | 0.1477 | +9/−0 |

  No bucket loses a query.
- **Gold accounting** (gold nodes rescued by the patch / displaced from the FLAT tail): webqsp 1,088 / 8, hotpotqa 63 / 0, 2wiki 328 / 0.
- **Patch size.** Of the 100 slots, the mean number filled by novel beam nodes is 72.66 / 81.99 / 92.35 (webqsp / hotpotqa / 2wiki). Of those, 17.59 / 16.22 / 26.56 are from hop 1. The rest of the 100 is flat fill.

**30.5 DEV_A, non-selecting.** D2 and D3 are taken at PRIMARY's per-query exposure B_q (mean 5,027–5,104 nodes). In D2, "gained" means FLAT_BALANCED_H2 covers the query and PRIMARY does not.

| cache | N | n | D2: PRIMARY vs FLAT_BALANCED_H2 | D3: FLAT vs FLAT_BALANCED_H2 |
|---|---|---|---|---|
| metaqa | 43,234 | 1,002 | 0.7754 vs 0.7016 (−7.39; +19/−93, p 6.7e-13) **PARTITIONS_BEAT_FLAT_H2** | 0.3922 vs 0.7016 (+30.94; +310/−0, p 9.6e-94) **GRAPH_PATCH_HELPS** |
| metaqa_phg | 43,234 | 1,002 | 0.7964 vs 0.7016 (−9.48; +15/−110, p 4.9e-19) **PARTITIONS_BEAT_FLAT_H2** | 0.3922 vs 0.7016 (+30.94; +310/−0, p 9.6e-94) **GRAPH_PATCH_HELPS** |
| squad | 20,233 | 992 | 0.9950 vs 0.9990 (+0.40; +5/−1, p 0.22) NO_SIGNIFICANT_DIFFERENCE | 0.9990 vs 0.9990 (+0/−0) NO_SIGNIFICANT_DIFFERENCE |
| squad_phg | 20,233 | 992 | 0.9899 vs 0.9990 (+0.91; +9/−0, p 0.0039) **FLAT_H2_BEATS_PARTITIONS** | 0.9990 vs 0.9990 (+0/−0) NO_SIGNIFICANT_DIFFERENCE |
| musique | 117,534 | 996 | 0.7600 vs **0.9036** (+14.36; +159/−16, p 8.5e-31) **FLAT_H2_BEATS_PARTITIONS** | 0.8815 vs 0.9036 (+2.21; +24/−2, p 1.05e-5) **GRAPH_PATCH_HELPS** |

**`DEV_A_KB_READING = PARTITIONS_BEAT_FLAT_H2_ON_THE_DEV_A_KB`.** metaqa D2 is LOSS under both partitionings, which by the pre-registered rule is the ruling's second scenario. The magnitude sits between the ruling's two illustrations:
- flat + H2 = 0.70, against partition + H2 = 0.78 / 0.80;
- the illustrations were 0.60 (scenario 2) and 0.78 (scenario 1).

PRIMARY, the partition composite, leads FLAT by 38.3 points (Mt-KaHyPar) and 40.4 points (PHG). The partition-free patch recovers 30.9 of them, 81 % and 77 % respectively. The remaining 7.4 / 9.5 points are significant.

For scale, FLAT_BALANCED_H2's 0.7016 also sits above §29's plain node-voting route at P = 50 (about 5,100 nodes, no patch): 0.640 / 0.673. That comparison is descriptive, not a paired test.

On the two MetaQA caches, FLAT@B_q and FLAT_BALANCED_H2@B_q give row-for-row identical ALL indicators (checked against the records' per-row arrays). Two things explain this:
- the flat side and the beam are partition-free;
- the caches' B_q differ by about 58 nodes on average (5,027 vs 5,086), which moves no ALL-gold indicator.

The patch sizes differ slightly.

Every DEV_A record passed the identity gates of §30.2, including `F2 == FULL` (bounded top-M equals the full materialisation) on every split-A row.

*Where the MetaQA residual lives.* D2 by stratum, for both partitionings. The strata are descriptive; ns means p ≥ 0.01.

| stratum | n | FLAT (D3 ref) | FLAT_BALANCED_H2 | PRIMARY (Mt-KaHyPar) | +/− vs Mt-KaHyPar | PRIMARY (PHG) | +/− vs PHG |
|---|---|---|---|---|---|---|---|
| hop 1 | 326 | 0.4847 | 0.9877 | 0.9939 | +0/−2 (ns) | 0.9969 | +0/−3 (ns) |
| hop 2 | 344 | 0.4012 | 0.7645 | 0.8081 | +1/−16 | 0.8198 | +0/−19 |
| hop 3 | 332 | 0.2922 | **0.3554** | **0.5271** | +18/−75 | **0.5753** | +15/−88 |
| 1 gold | 422 | 0.6517 | 0.9905 | 0.9905 | +3/−3 (ns) | 0.9976 | +1/−4 (ns) |
| 2 golds | 146 | 0.4110 | 0.9041 | 0.9384 | +5/−10 (ns) | 0.9315 | +5/−9 (ns) |
| 3 golds | 82 | 0.3049 | 0.7561 | 0.8780 | +3/−13 (ns, p 0.021) | 0.8902 | +4/−15 (ns, p 0.019) |
| 4 golds | 46 | 0.1522 | 0.6304 | 0.7174 | +4/−8 (ns) | 0.7174 | +4/−8 (ns) |
| 5–10 golds | 137 | 0.1241 | **0.3431** | **0.5620** | +4/−34 | **0.6277** | +1/−40 |
| 11+ golds | 169 | 0.0533 | **0.0888** | **0.2367** | +0/−25 | **0.2899** | +0/−34 |

- **Where flat + H2 matches the partitions.** On 1-hop questions (−2, ns) and single-answer questions (+3/−3) it matches the composite. The graph patch alone lifts those strata from 0.48 / 0.65 to 0.99. ANY-gold at B_q rises from 0.7814 (FLAT) to 0.9830: flat + H2 serves at least one gold node for 98 % of the questions.
- **Where the partitions win.** Of the net 74-query deficit on the Mt-KaHyPar cache (93 − 19):
  - by hop, 57 are 3-hop, 15 are 2-hop and 2 are 1-hop;
  - by gold count, 55 have five or more gold nodes.
- **A reading, not a test.** Those are the strata where a depth-2 patch with 100 slots runs out of reach or capacity: the hop-2 beam is truncated on 866 of 1,002 rows. A block, by contrast, co-locates the answer set. This is the ruling's scenario-2 mechanism, confined to those strata.
- **At a tight budget.** FLAT_BALANCED_H2@500 = 0.5988, above §29's node-voting L1@5 at about 510 nodes (0.507 / 0.531). This is descriptive, not a pre-registered comparison.

*DEV_A fixed-M curves* (descriptive; FLAT / FLAT_BALANCED_H2 ALL-gold, +gained/−lost).

| cache | M = 500 | 1,000 | 2,000 | 5,000 |
|---|---|---|---|---|
| metaqa (both caches) | 0.1048 / 0.5988 (+497/−2) | 0.1776 / 0.6168 (+441/−1) | 0.2764 / 0.6487 (+375/−2) | 0.3922 / 0.7016 (+310/−0) |
| musique | 0.6596 / 0.7259 (+82/−16) | 0.7500 / 0.7972 (+57/−10) | 0.8012 / 0.8444 (+45/−2) | 0.8805 / 0.9036 (+24/−1) |
| squad (both caches) | 0.9960 / 0.9950 (+0/−1) | 0.9980 / 0.9980 | 0.9980 / 0.9980 | 0.9990 / 0.9990 |

*MuSiQue by gold count.* In D2 the composite loses to flat + H2 at every gold count, and the gap widens with the number of golds.

| golds | n | PRIMARY | FLAT_BALANCED_H2 | D2 +/− | D3 +/− vs FLAT |
|---|---|---|---|---|---|
| 2 | 543 | 0.8527 | 0.9632 | +64/−4 | +16/−0 (FLAT 0.9337) |
| 3 | 295 | 0.7695 | 0.8949 | +44/−7 | +5/−2 ns (FLAT 0.8847) |
| 4 | 158 | 0.4241 | 0.7152 | +51/−5 | +3/−0 ns (FLAT 0.6962) |

At B_q the MuSiQue patch serves 87.63 novel nodes, 44.84 of them from hop 1. The gold accounting is 24 rescued and 2 displaced.

*SQuAD.* The patch is inert.
- No gold is rescued or displaced at B_q.
- 36 rows have an empty patch: 32 whose seeds have no STRUCT edges (both hops empty), and 4 whose beam nodes are all already in the flat base.
- The one cost is a single query at M = 500 (0.9960 → 0.9950, descriptive).
- squad_phg's D2 GAIN is §29's flat advantage over its PHG composite, which the patch leaves intact.

**30.6 Predictions (v1, written before any run) vs outcomes.**

| prediction | outcome |
|---|---|
| webqsp D1 GAIN, +3 to +15 | GAIN held; size **+26.3**, above the range |
| hotpotqa D1 NEUTRAL | **FAILED**: GAIN +6.1 (+63/−0) |
| 2wiki D1 NEUTRAL or a small GAIN | GAIN held; size **+24.5**, not small |
| metaqa D2 LOSS, FLAT_H2 ≈ 0.45–0.60 vs PRIMARY 0.78 / 0.80, "one 100-node patch is exhausted by hop-1 nodes" | LOSS held; FLAT_H2 = **0.70**, above the range. The stated mechanism did not hold: hop-1 nodes are 24.4 of the 82.3 patch nodes served, and the residual is 3-hop / large answer sets |
| metaqa_phg D2 LOSS | held |
| metaqa / metaqa_phg D3 GAIN +8 to +20, mostly on hop-1 queries | GAIN held; size **+30.9**; hop 1 +164, hop 2 +125, hop 3 +21 (not mostly hop 1) |
| squad D2 / D3 NEUTRAL / NEUTRAL | held |
| squad_phg D2 / D3 NEUTRAL / NEUTRAL | D2 **FAILED** (FLAT_H2_BEATS_PARTITIONS, +9/−0, p 0.0039); D3 held |
| musique D2 GAIN, FLAT_H2 ≈ 0.88 vs PRIMARY 0.76 | held (0.9036) |
| musique D3 NEUTRAL | **FAILED**: GAIN +2.2 (+24/−2) |

Every miss underestimated the patch. No prediction erred in the other direction.

**30.7 What this says, and what it does not.**
- **Says:**
  - On three untouched populations, adding one block-equivalent of the frozen real-edge beam to flat dense + SPLADE, at a fixed total node budget, raises ALL-gold coverage significantly with zero losses. That holds on text (+6.1, +24.5) and on the NAME_ONLY KB (+26.3).
  - On DEV_A text, at the composite's own exposure, flat + H2 beats or ties the frozen partition composite: musique +14.4, squad_phg +0.9, squad ns.
  - On the DEV_A KB, the composite still wins by 7–9 points. Flat + H2 recovers about 80 % of the composite's lead over flat without any partition. The rest is 3-hop and large-answer-set co-location.
  - Against the §29 regime split: the text half now holds with the graph included, and the KB half narrows from 38–40 points to 7–9.
- **Does not say:**
  - *No partition arm on the fresh corpora.* No canonical partition exists there, and the ruling does not require one. Whether QMAX_BALANCED_H2_PATCH1 beats FLAT_BALANCED_H2 on webqsp is unmeasured; on MetaQA it does.
  - *WebQSP's corpus is benchmark-conditioned.* It is the RoG universe (WEBQSP_ROG_RESOLVED / BENCHMARK_CONDITIONED), so the KB gain is measured on that corpus.
  - *Compute and latency.* FLAT_RRF scores every node.
    - The bounded two-sweep exact top-M took 1,856 / 2,625 / 2,941 s for 786 / 1,031 / 1,004 queries. That is about 2.4–2.9 s per query, single-process, on this contended CPU host.
    - The beam took 1.14 s per query on webqsp (p95 3.2 s; 3,195 hop-2 transitions scored per query) and 0.28–0.30 s on hotpotqa and 2wiki.
    - No latency claim either way.
  - *Downstream effects.* This is ALL-gold coverage at L1 only. No L2, L3 or QA effect was measured.
  - *Held-out data.* Split B of every fresh population is sealed. DEV_B and TEST were not read.
  - *Population reuse.* This ablation has now read the hotpotqa and 2wiki split-A gold. 3af9ad9b had reserved those rows for the composite's (blocked) transfer; the pre-registration flags the reuse.
  - *The 2wiki corpus size.* 3af9ad9b states N = 11,539,902, while the served corpus has 5,989,847 nodes. The discrepancy is recorded, not resolved.
- **Decides nothing.** FREEZE f81ebb84 is unchanged, and adoption is the user's ruling. The measured trade-off:
  - Replacing the composite with flat semantic retrieval → bounded graph expansion → node-level context (the ruling's *"attractive possible endpoint"*) wins on every text corpus and on the fresh KB.
  - It costs 7–9 points of MetaQA ALL-gold, concentrated in 3-hop and large-answer-set questions.
  - Any follow-up needs a ruling and a validation population. Examples: a confirmation on the sealed split B, or a partition-free treatment of the KB residual, which would be a new mechanism.

**30.8 Cost and hygiene.**
- **Wall-clock:**
  - webqsp G 2,892.6 s (beam 898.1, bounded flat 1,855.9), with E inside the same process (§30.3).
  - hotpotqa G 2,988.2 s (beam 294.2, bounded flat 2,625.1), E 13.5 s.
  - 2wiki G 3,368.4 s (beam 302.0, bounded flat 2,941.1), E 6.6 s.
  - DEV_A: metaqa 162.7, metaqa_phg 152.5, squad 292.2, squad_phg 310.1, musique 570.5 s.
- **Memory:** process RSS ≤ 1.11 GB. At the starts the host had 0.49–2.19 GiB available and 9.1–14.2 GiB of swap in use.
- **Foreign processes, all untouched.** At the webqsp start, two foreign Python processes were running: pid 3724 (1.5 GB) and pid 20560 (4.4 GB). Every later start shows pid 12096 (7.2–8.4 GB RSS).
- **Pins.** After the chain, all 49 pins of the v2 pre-registration re-verify: code, imports, records, the five replay caches, the fresh derived files and the probe. All 26 `flath2_*` files check against their records and the summary.
- **Nothing under `data/` was written:** `find -newer` returns nothing and git shows no change.
- **Nothing pinned was edited:** no pinned module and no CONTRACT_FILE.

## 31. L1 as a development loop: static partition localization as a NODE score (steps 1–2 of the user's iteration order), the pause, and the edge-family localization diagnostic (2026-09-26/27 local) (`_l1d_lib.py`, `_l1d_loc.py`, `_l1d_soft.py`, `_l1d_ceiling.py`, `_l1d_edgediag.py`, `_l1d_degprior.py`, summaries `_l1d_loc_summary.py`, `_l1d_soft_summary.py`; built but without a v1 record: `_l1d_arms.py`, `_l1d_hier.py`, `_l1d_act.py`, `_l1d_reach.py`, `_l1d_var_summary.py`; records under `results/L1_DEV/`; STATUS = DEVELOPMENT: descriptive numbers, no pre-registration, no verdicts, no adoption)

**31.1 The ruling (2026-09-26).** The user answered two questions and, with them, changed how L1 work is done. The answers are in the session transcript; the operative sentences, verbatim:
- Workflow: *"stop treating every experiment like a final confirmatory trial. For this project, I'd switch to a normal ML/retrieval workflow: development → model/mechanism selection → final held-out evaluation."* Development numbers answer *"What works? Why? What should we build next? Not: Can I publish this p-value as untouched confirmation?"*; *"Once L1 is actually settled, freeze **the architecture once**, then evaluate it on genuinely held-out populations: official TEST where untouched, sealed split B where genuinely appropriate, WebQSP holdout, HotpotQA/2Wiki held-out sets, or newly reserved populations"*.
- Scoring: *"a **budget curve**, not one arbitrary P50 number: R_ALL(M) for something like M∈{100,250,500,1000,2000,5000}. For every mechanism report: **ALL-gold recall** — primary; ANY-gold recall; mean fractional gold recall; per-hop ALL-gold; answer-cardinality strata; nodes exposed; query-time latency; memory/index size. Then compare methods at **equal node exposure**."*
- Static localization: *"three clear baselines: FLAT, STATIC LOCALIZATION, FLAT + STATIC LOCALIZATION with **no H2/PPR/traversal anywhere in the L1 comparison**. And don't make structural localization buy whole blocks unless we're explicitly testing block serving. We should test whether partitions can produce a **better node score**. For example: A(B)=Σ_{v∈top semantic hits∩B} w(v), then each node gets structural evidence: L(u)=A(P(u))."* Iteration order: *"1. hard node-level localization; 2. soft memberships; 3. hierarchical regions; 4. alternative static incidence/activation; 5. only then decide whether static partition localization has hit a fundamental limit. We no longer need a bureaucratic "reopening ruling" for every one."*
- Separation: *"L1: static localization; L2: learned reranking; L3: dynamic traversal/reasoning. So during L1 development: **No H2. No PPR. No learned MLP.**"*
- Dataset marking: *"**development** — previously seen, freely reusable; **validation** — used for architecture/model selection; **test/transfer** — untouched until final architecture. If a dataset has historical exposure, call it development. Done."*; *"let's score the damn system properly now, iterate on L1 until it is actually good, and save the freezing ceremony for the final architecture."*
- MetaQA: *"**Descriptive only now, and move on to the actual localization experiment.**"* — the dev population after the obvious exclusions (not the old 15k `l1_optimize` sample); hops 1/2/3, cardinality, budget; FLAT vs LOC vs FLAT+LOC; Type A/B/C; *"whether static localization actually recovers semantically invisible KB answers"*.

This supersedes the confirmatory parts of the HARD_LOC_A rulings (pre-registration, decisive McNemar verdict, eligibility genealogy, PRIMARY and H2 reference arms) and the §14–§30 reopening machinery. The confirmatory harnesses `_l1c_hardloc.py` and `_l1c_hardloc_primary.py` were never run for a result; they are parked, not deleted. FREEZE f81ebb84 stays a historical record; `FLAT_BALANCED_H2` (§30) is not L1 (H2 belongs to L3).

**31.2 Design.**
- **Population** (`loc_population.json`, sha c7f70806…; label *development*). Rows exclude only the obvious in-tree sets: the DEV_A/DEV_B replay-cache rows of every cell and the LEGACY_CONTINUITY lane rows. Eligible rows need ≥ 1 gold node. Seed 20260926.
  - metaqa: 1,998 dev rows, 666 per hop (eligible 36,512).
  - squad: 2,000 dev rows (eligible 9,873).
  - musique: all 417 remaining dev rows plus 1,583 train rows (a `per_source` stratum keeps them apart).
- **FLAT** = the pinned `rrf_full`: exhaustive fp32 dense (Qwen) + SPLADE over all N nodes, fv = 1/(60 + p_dense) + 1/(60 + p_splade), full rank `frank`. Every batch asserts ≥ 0.90 top-100 overlap with the stored retrieval-cache lists.
- **Localization node score** L(u) ≥ 0 per variant. The LOC order = every node with L > 0 by (−L, frank), so a block's nodes come in FLAT order and tied blocks interleave by FLAT rank.
- **Arms**, each serving exactly M nodes:
  - FLAT@M;
  - LOC@M = LOC order, then FLAT fill;
  - FLAT+LOC@M: f(u) = 1/(60 + frank) + [u ∈ LOC]/(60 + LOC rank), full `lexsort((frank, −f))`;
  - step 1 only: SLOT_C@5000 = FLAT[:5000 − C] + C novel LOC nodes, C ∈ {25, 50, 100, 200, 500}.
- **Scoring** at M ∈ {100, 250, 500, 1000, 2000, 5000}:
  - ALL (primary), ANY, FRAC;
  - per hop (metaqa 1/2/3; musique 2/3/4) and per answer cardinality (1, 2, 3, 4, 5–10, 11+);
  - gained/lost queries vs FLAT@M (McNemar p descriptive);
  - the FLAT budget M′ that matches the arm's ALL;
  - semantically invisible gold = FLAT rank ≥ 5000;
  - LOC-order length (nodes exposed), latency, index bytes, peak RSS.
- **Audit** of every FLAT@M-missed gold (step 1):
  - Type A: L = 0 (block silent);
  - Type B: its block's first LOC position ≥ M;
  - Type C: the block starts before M but the gold sits deeper.
- **Cells** (frozen partitions, never rebuilt):
  - metaqa: H4_SK (Mt-KaHyPar) and LOWMEM__PHG_REPAIR1_con, 432 blocks each, N = 43,234;
  - squad: H4_SK and LOWMEM__PHG_con, 202 blocks, N = 20,233;
  - musique: LOWMEM__PHG_C1_con, 1,175 blocks, N = 117,534.
- **Identities asserted across harnesses**, exact on every gold position:
  - FLAT and every HARD variant of step 2 equal the step-1 record, on all three datasets;
  - the edge diagnostic asserts the step-1 FLAT gold ranks (MetaQA 14,489 and MuSiQue 4,830 gold nodes);
  - steps 3 and 4 assert FLAT and HARD / F_OWN_SUM against the step-1 record (checked on SQuAD smoke runs of 50 rows), and step 4 asserts F_MEM_SUM = step-2 MEMACT when the soft record exists;
  - step 4's SERVED_MEM reproduces the replay caches' stored `base_rank` (mode CHECK: 50/50 rows in all 5 cells).

**31.3 Steps 1–2: what a partition-derived node score does (records `loc_{metaqa,musique,squad}__v1.{json,npz}`, `loc_SUMMARY__v1.json`, `soft_{metaqa,musique,squad}__v1.{json,npz}`, `soft_SUMMARY__v1.json`, `ceiling_blocks__v1.json`).** ALL-gold recall at M = 100 / 250 / 500 / 1000 / 2000 / 5000. Step 2's HARD rows equal step 1's HARD record exactly.
- **MetaQA** (1,998 rows). FLAT is 0.025 / 0.055 / 0.103 / 0.179 / 0.254 / 0.365.
  - Step 1, HARD (L(u) = A(P(u))). LOC 0.032 / 0.061 / 0.088 / 0.125 / 0.164 / 0.192. FLAT+LOC 0.034 / 0.066 / 0.104 / 0.177 / 0.271 / 0.395: a small gain at the ends of the curve, a loss in the middle.
  - Step 2, MEMACT (a hit votes for its own block and the blocks of its STRUCT out-neighbours: the served membership table, as a node score). LOC 0.316 / 0.353 / 0.385 / 0.433 / 0.498 / 0.597 on H4_SK and 0.291 / 0.382 / 0.424 / 0.466 / 0.530 / 0.625 on PHG_REPAIR1. MEMACT LOC@100 matches FLAT@3,277 and LOC@5000 matches FLAT@19,627.
  - HALO LOC 0.140 / 0.199 / 0.248 / 0.356 / 0.462 / 0.615.
  - The soft neighbourhood memberships (NBR_HA, NBR) are below FLAT at most budgets. ENS FLAT+LOC is 0.050 … 0.420.
- **MuSiQue** (2,000 rows). FLAT is 0.538 / 0.631 / 0.695 / 0.757 / 0.814 / 0.884 (corrected 2026-09-29 from the exact counts 1,075 / 1,262 / 1,389 / 1,513 / 1,628 / 1,767; first printed as 0.537, 0.756 and 0.883, see §37.3).
  - Every LOC arm is below FLAT at every budget, for example HARD 0.036 … 0.735 and MEMACT 0.004 … 0.525.
  - Every FLAT+LOC arm is below FLAT as well, except HALO: +0.007 at M = 250 and a tie at M = 500.
- **SQuAD** (2,000 rows). FLAT is 0.994 / 0.997 / 0.999 / 1.000 / 1.000 / 1.000, i.e. at the ceiling. Every LOC arm is below FLAT. FLAT+LOC ties only from M = 2000.
- **Reading.** The same static rule is a large gain on the KB and a loss on both text corpora. The gain comes from the STRUCT out-edges in the membership table, not from blocks. This is what the user's pause (31.4) addressed.

**31.4 The pause and the edge diagnostic (2026-09-26).** Steps 3 and 4 (hierarchical regions, alternative incidence) were ready to run; the user paused them. Verbatim:
- *"The step-2 result is already telling us something important enough that I would pause before letting steps 3 and 4 become another search spiral."*
- *"So the problem is no longer "our activation rule is weak." It is more fundamental: the meaning of graph locality is different across the datasets"*
- *"Before building another localization mechanism, measure: Does static graph adjacency actually predict gold co-relevance?"*
- The diagnostic they asked for: *"1. For MetaQA and MuSiQue, measure missing-gold enrichment under each edge source/type. 2. Measure how many blocks each top semantic hit activates under each source. 3. Measure precision: activated blocks containing missing gold / activated blocks. 4. Compare STRUCT-only, KNN-only, and STRUCT+KNN."*
- The hypothesis it tests: *"If KNN-only behaves much better on MuSiQue while STRUCT-only drives MetaQA, then we have a very promising universal static mechanism: edge-type-aware static localization"*, provided that *"the weighting/selection criterion is itself static and universal—for example based on edge coherence, specificity, degree, or some corpus-level reliability statistic."*

Implemented as `_l1d_edgediag.py`, a gold-aware diagnostic that is never a candidate, over the same population and the same FLAT; the v1 FLAT gold ranks are asserted exactly.
- **Hits.** S = FLAT[:200]. Missing gold = golds of FLAT rank ≥ M, for M ∈ {100, 1000, 5000}.
- **Families.** STRUCT_out, STRUCT_in, STRUCT, KNN, NER, TITLE (MuSiQue same-title pairs), SK, SKN, OUT+KNN, and the 18 MetaQA relation families by direction. Also BLOCK (same block as the hit) and OWN (a hit's own block).
- **Per hit and per edge.**
  - Enrichment = observed gold (or missing gold) among a hit's neighbours ÷ the random expectation. For missing gold, the random node is drawn from outside FLAT@M.
  - Broken down by hit rank bucket, gold versus non-gold hit, hit spread |N_F(s)|, neighbour degree, and edge coherence (the dense cosine of the two endpoints, quintiles within each family).
- **Node set.** U = the 1-hop neighbours of the top-10 / top-200 hits outside FLAT@M. Its recall of missing gold is compared with FLAT extended by |U| nodes, i.e. equal exposure.
- **Blocks.** mem_F(s) = the hit's own block + the blocks of N_F(s). Reported:
  - blocks per hit, added blocks per hit, blocks per query;
  - precision = activated blocks holding a missing gold ÷ activated blocks, over all and over added blocks, with enrichment over the random block share;
  - recall, and the new node mass compared with FLAT extended by the same mass;
  - precision by hit spread and by vote count.
- **Gold pairs** (independent of FLAT): the share of a query's ordered gold pairs that are adjacent in the family, against the family density; and the same-block rate.
- **Label-free family statistics**: degree, coherence against random pairs, block-internal share, overlap with KNN and with STRUCT.
- **Degree-matched control** (`_l1d_degprior.py`, records `degprior_<ds>__v1.json`). The neighbour-degree enrichment is re-divided by the gold prior of the same degree bin, so that gold-rich hubs are not credited to adjacency.

**31.5 Edge diagnostic: results (records `edgediag_metaqa__v1.json` sha 9068bf31…, `edgediag_musique__v1.json` sha 6f0790e3…, `degprior_metaqa__v1.json` sha 3bcb130b…, `degprior_musique__v1.json` sha 186c4bbd…).** Both edge records assert the step-1 FLAT gold ranks on every gold node: MetaQA 14,489 and MuSiQue 4,830. Missing gold nodes at M = 100 / 1000 / 5000:
- MetaQA: 14,131 / 11,828 / 7,908.
- MuSiQue: 1,172 / 570 / 259. The MuSiQue @5000 columns rest on 259 nodes in 233 queries.

*(1) Missing-gold enrichment per edge family, per hit.* All top-200 hits unless marked. Values are observed ÷ random; for missing gold the random node is drawn from outside FLAT@M.

| family | MetaQA nbrs/hit | m@1000 | m@5000 | m@5000, top-10 hits | MuSiQue nbrs/hit | m@1000 | m@5000 | m@5000, top-10 hits |
|---|---|---|---|---|---|---|---|---|
| STRUCT_out | 3.8 | 57.1 | 31.4 | 80.3 | 31.4 | 27.8 | 15.1 | 89.5 |
| STRUCT_in | 4.5 | 7.5 | 9.0 | 116 | 36.9 | 12.9 | 9.3 | 20.1 |
| STRUCT | 8.2 | 26.9 | 16.9 | 94.6 | 65.7 | 20.1 | 12.4 | 56.7 |
| KNN | 4.6 | 2.2 | 1.4 | 3.9 | 5.0 | 44.2 | 32.0 | 346 |
| NER | 0.4 | 0.7 | 0.6 | 7.7 | 16.7 | 35.4 | 26.7 | 319 |
| TITLE | – | – | – | – | 6.6 | 17.0 | 6.5 | 37.1 |
| SK = STRUCT ∪ KNN | 12.8 | 21.9 | 14.6 | 81.6 | 69.9 | 20.4 | 12.6 | 58.5 |
| SKN = SK ∪ NER | 13.0 | 21.5 | 14.4 | 80.5 | 84.2 | 21.2 | 14.0 | 85.7 |
| BLOCK (same block as the hit) | 99.5 | 1.6 | 1.5 | 4.5 | 100 | 9.2 | 5.5 | 20.3 |

- **Which hits activate.** Enrichment for missing@5000 gold, split into non-gold hits and gold hits.
  - MetaQA (577 gold hits among 399,600): STRUCT_out 31.5 vs 3.4; STRUCT 17.0 vs 0.4; KNN 1.2 vs 62.1.
  - MuSiQue (3,872 gold hits): STRUCT_out 11.1 vs 669; STRUCT 9.9 vs 392; KNN 15.3 vs 2,394; NER 13.4 vs 2,241; TITLE 6.5 vs 0.
- **Gold co-relevance, independent of FLAT.** The share of a query's ordered gold pairs that are adjacent in the family, with enrichment over the family's density.
  - MetaQA: STRUCT 0.001 (4×); KNN 0.009 (90×); NER < 0.001 (30×); same block 0.099 (43×).
  - MuSiQue: STRUCT 0.177 (461×); KNN 0.047 (1,229×); NER 0.152 (1,308×); TITLE 0.017 (475×); SKN 0.284 (550×); same block 0.077 (85×).

*(2) Blocks activated.* mem_F(s) = the hit's own block + the blocks of N_F(s).

| family | MetaQA (432 blocks): blocks/hit | blocks/query | MuSiQue (1,175 blocks): blocks/hit | blocks/query |
|---|---|---|---|---|
| OWN | 1.0 | 121 | 1.0 | 90 |
| STRUCT_out | 2.9 | 225 | 5.6 | 249 |
| STRUCT_in | 4.5 | 335 | 11.5 | 552 |
| STRUCT | 6.4 | 373 | 15.3 | 614 |
| KNN | 3.4 | 264 | 3.0 | 206 |
| NER | 1.2 | 137 | 11.6 | 674 |
| TITLE | – | – | 2.3 | 155 |
| SK | 8.7 | 398 | 16.8 | 649 |
| SKN | 8.9 | 400 | 25.6 | 872 |

*(3) Block precision.* Precision = activated blocks holding a missing@5000 gold ÷ activated blocks. It is reported as enrichment over the random block share, for all activated blocks / added blocks only.
- MetaQA: every family is at 1.0–1.2. STRUCT is 1.1 / 1.1. NER's added blocks reach 2.0, on 0.2 added blocks per hit.
- MuSiQue:
  - OWN 4.1;
  - STRUCT_out 2.6 / 1.6; STRUCT 1.6 / 1.0;
  - KNN 2.7 / 1.6; NER 1.5 / 1.0; TITLE 3.3 / 2.1;
  - SK 1.5 / 1.1; SKN 1.3 / 0.9.
- **At equal new mass**, recall of missing@5000 gold against FLAT extended by the same number of nodes:
  - On MuSiQue, every family's block union loses. STRUCT_out 0.602 vs 0.792 at 22,659 nodes per query; KNN 0.552 vs 0.722; NER 0.907 vs 0.969; SKN 0.977 vs 0.992.
  - On MetaQA the unions tie, because they cover most of the corpus: STRUCT 0.955 vs 0.947 at 32,912 of 43,234 nodes.
- **Within a family**, block precision rises with votes and falls with hit spread. These bins are trends only (see the caveats).
  - MetaQA, 11+ votes: STRUCT_out 7.1, STRUCT 5.4.
  - MuSiQue, lowest → highest vote bin: STRUCT_out 1.8 → 5.1; NER 0.5 → 6.8.

*(4) STRUCT vs KNN vs STRUCT+KNN at node level.* U = the 1-hop neighbours of the hits, outside FLAT@M.
- Each cell gives the ALL of the set FLAT@M ∪ U, at per-query exposure M + |U_q|, vs FLAT at that same exposure.
- Brackets give new nodes per query.
- The numbers are exact counts from the record, not a ranked arm.

| arm | MetaQA @100 | @1000 | @5000 | MuSiQue @100 | @1000 | @5000 |
|---|---|---|---|---|---|---|
| FLAT@M | 0.025 | 0.179 | 0.365 | 0.538 | 0.757 | 0.884 |
| top-10 STRUCT | 0.390 vs 0.038 [64] | 0.453 vs 0.188 [60] | 0.568 vs 0.367 [50] | 0.668 vs 0.673 [481] | 0.820 vs 0.783 [436] | 0.906 vs 0.886 [361] |
| top-10 KNN | 0.031 vs 0.030 [26] | 0.180 vs 0.181 [16] | 0.365 vs 0.365 [9] | 0.578 vs 0.560 [25] | 0.774 vs 0.757 [11] | 0.886 vs 0.883 [4] |
| top-10 SK | 0.393 vs 0.041 [90] | 0.453 vs 0.189 [76] | 0.568 vs 0.367 [59] | 0.682 vs 0.681 [502] | 0.824 vs 0.784 [446] | 0.908 vs 0.886 [365] |
| top-10 NER | 0.030 vs 0.026 [2] | 0.179 vs 0.179 [2] | 0.365 vs 0.365 [1] | 0.659 vs 0.616 [128] | 0.829 vs 0.764 [103] | 0.909 vs 0.884 [73] |
| top-10 SKN | 0.393 vs 0.042 [91] | 0.453 vs 0.189 [77] | 0.568 vs 0.367 [59] | 0.744 vs 0.703 [611] | 0.861 vs 0.792 [539] | 0.923 vs 0.887 [432] |
| top-200 STRUCT | 0.529 vs 0.218 [1,337] | 0.567 vs 0.268 [1,277] | 0.651 vs 0.388 [1,089] | 0.781 vs 0.863 [5,301] | 0.863 vs 0.878 [5,070] | 0.927 vs 0.919 [4,501] |
| top-200 SKN | 0.543 vs 0.252 [1,939] | 0.572 vs 0.286 [1,684] | 0.651 vs 0.394 [1,327] | 0.880 vs 0.898 [7,391] | 0.913 vs 0.907 [6,936] | 0.949 vs 0.930 [5,948] |

- MuSiQue, top-10 hits by direction:
  - STRUCT_out: 0.644 vs 0.639 [194] @100; 0.809 vs 0.771 [177] @1000.
  - STRUCT_in recovers fewer missing@1000 gold nodes than FLAT extended by the same 272 nodes: 0.053 vs 0.068.
- **Per hop, as the share of missing-gold queries completed** (U vs FLAT at the same exposure):
  - MetaQA, top-10 SKN at @100 / @1000 / @5000:
    - hop1 0.997 / 0.996 / 0.995 vs 0.029 / 0.016 / 0.003;
    - hop2 0.063 / 0.032 / 0.018 vs 0.015 / 0.011 / 0.002;
    - hop3 0.105 / 0.069 / 0.053 vs 0.009 / 0.010 / 0.007.
  - MetaQA, top-200 SKN: hop2 0.327 / 0.268 / 0.205 vs 0.202 / 0.130 / 0.050; hop3 0.290 / 0.241 / 0.233 vs 0.207 / 0.128 / 0.046.
  - MuSiQue, top-10 SKN @1000: hop2 0.507 vs 0.100; hop3 0.440 vs 0.201; hop4 0.286 vs 0.143.
- Against step 2 at about the same exposure:
  - MetaQA: MEMACT LOC was 0.353 @250 vs 0.393 here at about 191 nodes, and 0.433 @1000 vs 0.453 here at about 1,077. MEMACT at 5000 was 0.597 (PHG 0.625), vs 0.568 at about 5,059 (top-10) and 0.651 at about 6,327 (top-200).
  - MuSiQue: every step-2 arm was below FLAT. Top-10 SKN on FLAT@1000 is 0.861 at about 1,539 nodes, between FLAT@2000 (0.814) and FLAT@5000 (0.883).

*(5) Label-free family statistics: can one choose the family?*
- **Coherence** = the mean dense cosine of an edge's endpoints. Random pairs score 0.366 on MetaQA and 0.147 on MuSiQue.
  - MetaQA: STRUCT 0.340, below random; KNN 0.735; NER 0.707. The ranking is inverted: only STRUCT localizes.
  - MuSiQue: KNN 0.575, TITLE 0.416, NER 0.368, STRUCT 0.266. TITLE ranks second but localizes nothing at node level (@5000 0.019 vs FLAT 0.019).
- **Degree.**
  - KNN is k-regular (mean 4.5 on both datasets), so it would rank as the most specific family everywhere.
  - STRUCT's tail differs: p99 28 and max 4,176 on MetaQA; p99 592 and max 14,294 on MuSiQue.
- **Block-internal share.** KNN edges are 37% (MetaQA) and 55% (MuSiQue) block-internal: already co-located with their hits. STRUCT is 18% and 3%.

*(6) Within-family edge statistics.*
- **Hit rank.** Missing@5000 enrichment for hits at ranks 1–10 / 11–50 / 51–200:
  - MetaQA STRUCT 94.6 / 16.4 / 12.9.
  - MuSiQue: NER 319 / 24.6 / 10.5; KNN 346 / 52.5 / 16.6; STRUCT_out 89.5 / 15.9 / 9.9.
- **Hit spread |N_F(s)|**, i.e. the IDF of the activating hit. Missing@5000 enrichment:
  - MetaQA STRUCT_out at spread 1 / 2–5 / 6–25 / 26–100: 174 / 36.7 / 30.2 / 25.0.
  - MetaQA STRUCT_in: 18.3 at spread 1 → 2.2 at spread 301+.
  - MuSiQue STRUCT_out at spread 1 / 2–5 / 6–25 / 26–100 / 101–300: 693 / 154 / 21.7 / 10.4 / 8.0.
  - MuSiQue NER at spread 1 / 2–5 / 6–25 / 26–100: 229 / 62.7 / 38.8 / 15.6.
- **Neighbour degree, degree-matched** = P(gold | adjacent, degree bin) ÷ P(gold | degree bin). Bins are degree 1 / 2–5 / 6–25 / 26–100 / 101–300 / 301+.
  - MetaQA STRUCT, gold: 12.4 / 11.3 / 6.8 / 2.1 / 1.8 / 2.2.
    - The degree-blind values are 3.9 / 6.4 / 7.4 / 15.1 / 112 / 313.
    - The blind rise with degree is a node-class prior, not adjacency: nodes of degree > 100 are 0.24% of MetaQA's nodes but 21% of its golds (genres, years and languages as answers).
  - MuSiQue STRUCT_out, gold: 290 / 361 / 326 / 217 / 85.6 / 15.6.
    - Degree 301+ holds 1.9% of nodes and 6.0% of golds, but 69% of STRUCT_out entries.
  - MuSiQue NER, gold: 225 / 147 / 97.7 / 78.9 / 43.6.
- **Coherence within a family.**
  - MuSiQue STRUCT missing@5000 by coherence quintile: 2.1 → 38.4 (gold 2.9 → 195). NER also rises.
  - MetaQA STRUCT is flat: 17.0–17.9.

*Caveats.*
- **The node-set rows are set unions, not a ranked arm.** Their exposure is M + |U_q| per query, not a fixed M. A real arm has to order U, and at M = 100–500 it can afford only part of it: MuSiQue top-10 SKN is 540–610 nodes per query.
- **Block numbers at union level ignore weights.**
- **The vote-bin and spread-bin block precisions are diluted.** Their numerators are taken over all queries (and per hit for spread), while the random share is taken over queries with missing gold. The approximate correction is ×1.57 for MetaQA and ×8.58 for MuSiQue. Read them as within-family trends only.
- **The degree-matched missing@M prior is approximate**: it treats FLAT@M as degree-neutral. The gold prior is exact.
- **Scope.** These are gold-aware diagnostics on the development population. None of them is a candidate or a verdict.

**31.6 Reading.**
- **The user's hypothesis holds for MetaQA and only partly for MuSiQue.**
  - STRUCT drives MetaQA. It is the only family that localizes there; KNN and NER sit at the random rate for missing gold.
  - MuSiQue is not KNN-only. KNN is the sharpest family (346× from the top-10 hits, 2,394× from gold hits). But its neighbours are mostly FLAT-reachable already: at top-200 they cover 0.070 of missing@5000 nodes.
  - NER carries MuSiQue's coverage: from the top-10 hits it recovers 0.304 of missing@1000 nodes with 103 new nodes per query, vs FLAT 0.042. STRUCT_out comes next. STRUCT_in and TITLE do not beat FLAT.
- **Graph locality means different things because the hit that localizes is different.**
  - KB: anchor → answer. The anchor is the question's entity, usually not gold. The edges are semantically incoherent (STRUCT cosine is below random), and the golds are not adjacent to each other.
  - Text: found gold → missing gold, along coherent edges (evidence chains), and the golds are adjacent to each other.
  - Both are a static 1-hop relation from the top FLAT hits.
- **At node granularity both beat FLAT at equal exposure.** Top-10 SKN gains, at @100 / @1000 / @5000:
  - MetaQA +0.35 / +0.26 / +0.20 ALL;
  - MuSiQue +0.04 / +0.07 / +0.04 ALL.
- **At block granularity the signal disappears.** Blocks activated through any family hold missing gold at about the random rate on MetaQA, and lose to FLAT at equal mass on MuSiQue.
  - So step 2's text loss is a serving-granularity loss (whole blocks), not the absence of a static signal.
  - MEMACT's MetaQA gain is reproduced at node level at low and middle budgets with similar exposure: 0.393 vs 0.353 at about 200–250 nodes, and 0.453 vs 0.433 at about 1,000. So the gain comes from the STRUCT out-edges in its membership table, not from the blocks. At 5000, MEMACT (0.597) is above top-10 SKN (0.568) and below top-200 SKN (0.651, at about 6,300 nodes).
- **No corpus-level family statistic picks the right family on both datasets, and none is needed.**
  - The union SKN at top-10 hits equals the best single family on MetaQA (0.453 @1000, the same as STRUCT). It beats every single family on MuSiQue.
  - A family that does not localize on a dataset costs few nodes: on MetaQA, KNN and NER add about 17 nodes per query to STRUCT's 60.
  - The per-edge statistics behave the same way on both datasets: hit rank, hit spread, and neighbour degree once degree-matched. They are label-free and static.
  - Coherence cannot select families. At most it is a soft weight within a family.
- **1-hop cannot reach MetaQA's hop2/hop3 golds.**
  - Top-10 SKN completes only 2–6% of hop2 and 5–11% of hop3 missing-gold queries.
  - Top-200 SKN reaches 20–33%. Its STRUCT hop3 signal, after degree matching, is mostly the hub prior.
  - Those golds sit 2–3 STRUCT hops from the anchor: reasoning edges, i.e. L3 in the user's split.
- **Steps 3 and 4, as built, test the dominated granularity.**
  - Step 3 (hierarchical regions) serves block unions.
  - Step 4 (`_l1d_act.py`) varies activation over OWN/MEM block incidence.
  - Neither has a record, and both stay paused.

**31.7 Proposal, pending the user (nothing started).** A node-level static localization score through ego-network incidence: step 4 re-scoped from blocks to neighbourhoods.
- **Score:** L(u) = Σ_{s ∈ FLAT[:K]} w(s) · Σ_F 1[u ∈ N_F(s)] · g(|N_F(s)|) · h(deg_F(u)), with F ∈ {STRUCT_out, STRUCT_in, KNN, NER}.
- **It is one sparse matrix-vector product over fixed adjacency.** No beam, no iteration, no path choice, and no parameter fitted to labels.
- **The same edges are used for every dataset** (the union). The weights are label-free and universal:
  - w = FLAT's own score of the hit (or its rank);
  - g = 1/log2(1 + spread), an IDF of the hit;
  - h = 1/log2(1 + deg), neighbour specificity;
  - optionally a soft coherence factor.
- **Grid.** Votes only; + w; + g; + g·h; + coherence. h appears both with and without, because it removes MetaQA's hub prior, which vote accumulation otherwise credits.
- **Evaluation.** Served through the step-1/2 LOC and FLAT+LOC arms on the budget curve for MetaQA, MuSiQue and SQuAD (SQuAD as a non-regression check at the ceiling). Reported: ALL / ANY / FRAC, per hop, per cardinality, exposure, latency.
- **Two decisions are the user's:**
  - (a) Is a 1-hop ego-network score from FLAT hits L1 static activation, or does it cross the "no traversal" line? It is H1 without a beam; the diagnostic measured it as u ∈ N_F(s); MEMACT's out-edge table is its block form; H2 (§30) was a depth-2 beam.
  - (b) Should it replace steps 3/4 as built?
- **Answered 2026-09-27: yes to both** (§32.1). NODELOC-1H (§32) is this score with the user's ladder: h enters only at rung L2, coherence is deferred, and partitions leave the node score and become routing units.

**31.8 Cost and hygiene.**
- **Wall-clock and memory:**
  - edge diagnostic: MetaQA 774 s, peak RSS 710 MB; MuSiQue 1,332 s, 1,313 MB;
  - degree control: 12 s and 135 MB; 33 s and 494 MB;
  - step 2: MetaQA 864 s / 697 MB; MuSiQue 1,506 s / 1,086 MB; SQuAD 650 s / 498 MB.
- **Host:** 0.52–2.03 GiB available at the starts, 10.8–13.7 GiB of swap in use.
  - Foreign pid 12096 (4.2–8.6 GB RSS) ran throughout, untouched.
  - One heavy process at a time.
- **The step-3 MetaQA run was stopped at the pause**, before it wrote a record: `l1d_hier_metaqa.log` ends at its start line. `l1d_chain_34.sh` was not relaunched.
- **Nothing under `data/` was written. No pinned module or CONTRACT_FILE was edited.**
- **New code only:** `_l1d_edgediag.py` (874bfc49…) and `_l1d_degprior.py` (b5931c4d…), sha-pinned by their records.

## 32. NODELOC-1H: static one-hop NODE localization (L1a) and partition routing with two budgets (L1b) (2026-09-27 local) (`_l1d_node1h.py`, summary `_l1d_node1h_summary.py`, on the shared runner `_l1d_arms.py`; records `results/L1_DEV/node1h_{metaqa,musique,squad}__v1.{json,npz}`, `node1h_SUMMARY__v1.json`; STATUS = DEVELOPMENT: descriptive numbers, no pre-registration, no verdicts, no adoption)

**32.1 The rulings (2026-09-27 local; transcript 2026-09-26T20:00Z and 20:03Z).** Two messages. The first answered 31.7 ("Yes to both."). The second arrived three minutes later, while the harness was being built. The operative sentences, verbatim:
- **The L1 boundary.**
  - *"I would define L1 operationally as: one application of a fixed, query-independent sparse incidence/adjacency operator to query-time semantic scores. Let x(q) be the FLAT score vector, nonzero only for the selected semantic hits, and let A_f be a fixed adjacency matrix for edge family f. Then: L_f(q)=A_f^⊤x(q)"*.
  - With fixed weights: *"L(u|q)= Σ_{s∈H_q} w_q(s) Σ_f A_f(s,u) g(deg_f^+(s)) h(deg_f(u))."*
  - *"adjacency is built offline; exactly one propagation; no newly found node becomes a new source; no beam; no recursive state; no stopping condition; no path search; no PPR iteration."*
  - *"L1 may statically project semantic evidence through a fixed one-step localization operator, but may not recursively reason from newly activated nodes."* So x → A^⊤x is L1, while *"x→A^⊤x→A^⊤(A^⊤x) starts becoming traversal/message passing, and a pruned/stateful version such as H2 clearly belongs in L3."*
- **Steps 3/4.**
  - *"Yes. I would stop the hierarchical-block route for now."* *"the edges already contain the localization signal; quantizing that signal back into blocks destroys much of it."*
  - *"The new Step 4 should become something like NODELOC-1H"*: Query → Dense + SPLADE → FLAT node scores → top semantic hits → {STRUCT_out, STRUCT_in, KNN, NER} → single fixed one-hop projection → node localization score L(u) → LOC ranking and FLAT + LOC ranking.
  - *"No partitions need to be involved in scoring. Keep partitions as an ablation / historical structural compression baseline, not as a requirement of the new localizer."*
- **The ladder.**
  - L0(u) = Σ_{s,f} w(s)·1[(s,u) ∈ E_f]. L1 divides by φ(deg_f^+(s)). L2 also divides by ψ(deg(u)).
  - *"Use simple parameter-free forms first, e.g. log/inverse-rank style transforms rather than tuning arbitrary exponents."*
  - *"I would leave coherence as an optional later modifier, because its behavior is not universal"*.
  - *"don't choose edge families dataset-by-dataset"*: *"one universal edge set: E = E_STRUCT-out ∪ E_STRUCT-in ∪ E_KNN ∪ E_NER."*
- **The test.**
  - *"At each budget: M∈{100,500,1000,2000,5000}, compare: FLAT@M, LOC@M, and most importantly: FLAT+LOC@M."*
  - *"I would combine FLAT and LOC by rank/RRF initially, not a tuned scalar"*.
  - *"Does static one-hop graph localization improve the node ranking at the exact same exposure? That is the L1 question."*
- **Architecture.**
  - L1 = *"Dense/SPLADE + one-hop static node localization"*;
  - L2 = *"MLP query/state-conditioned reranking"*;
  - L3 = *"H2/PPR/bounded walks/dynamic reasoning."*
- **Partitions as shards (the second message).**
  - *"keep partitions at the center of the system — but as physical/routing units, not as the unit we score relevance on. The mistake so far was coupling these two ideas: partition selected ⇒ serve every node in partition."*
  - The flow: *"node-level localization → identify relevant partitions/shards → bring only those shards → rank nodes inside them"*.
  - Stage 3 is *"R(P_j|q) = Agg_{u∈P_j} L(u|q)"*, either the max or a top-k sum. *"Inside each selected partition: TopK_{u∈P_j} S(u|q) returns the best local candidates."*
  - *"partition retrieval cost ≠ context exposure cost"*. *"We need two budgets, not one"*:
    - B_P = |partitions contacted| (*"Or bytes transferred / RPCs / machines"*);
    - B_N = |nodes returned|.
    - *"Previously P50 roughly implied: B_P=50 and: B_N≈5000. We tied them together."*
  - The two layers are *"L1a: Node Localization"* and *"L1b: Partition Routing"*. *"Then selected partitions return their strongest nodes. Still no traversal. Still no learning. Still one L1."*
  - PHG's objective becomes *"place structurally related computation/data together while minimizing cross-machine communication"*. *"PHG answers: Where should the graph physically live? Node-level L1 answers: Where should this query look?"*
  - *"graph-aware distributed coarse router"*, whose coarse score comes from fine-grained node evidence rather than a centroid.
  - *"Partitions localize computation and data; nodes localize relevance."* *"I think that should be the architectural principle going forward."*

Both rulings are adopted as stated, and they answer 31.7's two decisions. Steps 3 and 4 as built (`_l1d_hier.py`, `_l1d_act.py`) are retired without a record, and `l1d_chain_34.sh` stays unlaunched.

**32.2 Design** (`_l1d_node1h.py`, sha 8265d6bc…, on the shared runner `_l1d_arms.py`, sha f5f52b79…).
- **Population, FLAT and arms are as in §31.2.**
  - Population: `loc_population.json` (c7f70806…; *development*): metaqa 1,998 rows (666 per hop), musique 2,000, squad 2,000.
  - FLAT is the pinned `rrf_full`.
  - LOC@M serves the LOC order (every node with L > 0, by (−L, FLAT rank)), then FLAT fill.
  - FLAT+LOC@M is RRF: f(u) = 1/(60 + FLAT rank) + [u ∈ LOC]/(60 + LOC rank).
  - Every arm serves exactly M nodes, M ∈ {100, 250, 500, 1000, 2000, 5000}: the user's five budgets plus 250 from the 2026-09-26 curve.
- **L1a, the node score.**
  - Hits: H_q = FLAT[:200], the activation depth of steps 1–2.
  - One edge set on every dataset, read-only from the canonical graph. It uses the edge diagnostic's key construction: self loops dropped, pairs deduplicated.
    - STRUCT_out: s → u for a structural edge s → u.
    - STRUCT_in: s → u for a structural edge u → s.
    - KNN and NER: the undirected semantic-kNN and shares-entity families, in both orientations.
  - L(u|q) = Σ_{s∈H_q} w(s) Σ_f A_f(s,u) g_f(s) h_f(u). A pair in several families counts once per family. This is one sparse product per query.
- **The ladder.** The log base only rescales L, so no ranking depends on it.
  - L0: g = h = 1.
  - L1: g_f(s) = 1/log2(1 + deg_f^+(s)), where deg_f^+(s) = the edges of E_f leaving s.
  - L2: L1 and h_f(u) = 1/log2(1 + deg_f(u)), where deg_f(u) = the edges of E_f incident to u.
- **Two hit weights.**
  - FV: w(s) = fv(s), the FLAT score. This is the literal x(q).
  - IR: w(s) = 1/(FLAT rank of s, 1-based). The reason: the edge diagnostic found that hits of rank 1–10 carry 6–30× the missing-gold enrichment of ranks 11–200, while fv falls only about 1.4× over that range.
- **Variants.**
  - FV_L0, FV_L1, FV_L2, IR_L0, IR_L1 and IR_L2 are node scores that do not depend on the cell.
  - Per cell, `<cell>__HARD` is the step-1 hard LOC: the identity anchor and the historical partition baseline.
- **L1b, routing.** A cell's frozen partition is used only as a shard map.
  - For each served order O (FLAT, or a rung's FLAT+LOC order) and each rule, the partitions are ranked and the first B_P are contacted.
  - The first B_N nodes of O inside the contacted partitions are returned. This equals each contacted shard sending its best nodes by the served score, followed by a global merge that keeps B_N. Fewer nodes come back when the contacted shards hold fewer than B_N.
  - The rules:
    - S: R(P) = max_{u∈P} of the served score, i.e. partitions in order of first appearance in O;
    - LMAX: R(P) = max_{u∈P} L(u), the user's Agg = max; ties and L = 0 fall back to first appearance in O;
    - LSUM: R(P) = Σ_{u∈P} L(u), the top-k sum with k = |P|.
  - The grid is B_P ∈ {1, 2, 5, 10, 20, 50, 100} × B_N ∈ the M curve.
  - Reported: the contacted node mass, the returned exposure, the paired loss against the unrouted arm at the same B_N, and P*(M) = the number of partitions that the unrouted top-M touches.
- **Cells** are as in §31.2:
  - metaqa: H4_SK and LOWMEM__PHG_REPAIR1_con, 432 blocks each;
  - squad: H4_SK and LOWMEM__PHG_con, 202 blocks;
  - musique: LOWMEM__PHG_C1_con, 1,175 blocks.
- **Identities asserted**, exact on every gold node:
  - v1 FLAT and every `<cell>__HARD` equal the step-1 record;
  - the fused ranks used for routing equal the runner's FLAT+LOC positions, so contacting every partition is the unrouted arm;
  - under S, every gold ranked before the (B_P + 1)-th partition's first node keeps its rank (checked per row);
  - the step-2 record's rows and FLAT positions match before the MEMACT pairing.
- **Diagnostics:**
  - paired comparisons along the ladder (L0 → L1 → L2) and between the hit weights (FV → IR);
  - paired comparisons against step-1 HARD and step-2 MEMACT;
  - per gold node, the families that reach it from the top-10 and the top-200 hits;
  - the support (nodes with L > 0) and the new nodes outside FLAT@M, cross-checked against the edge diagnostic's SKN node set;
  - index bytes, entries per row, latency.
- **No learned weight and no tuned constant.** K0 = 60 and the 200 hits are the frozen contract values.
- Before the launch, smoke runs of 200 rows per dataset, outside the repository, checked the identities. Their numbers are not used.

**32.3 L1a: budget curves at equal exposure (records `node1h_metaqa__v1.json` sha f0ef9bcc…, `node1h_musique__v1.json` sha 4e42017b…, `node1h_squad__v1.json` sha 1f94df8d…, each with its `.npz`; summary `node1h_SUMMARY__v1.json` sha 3061228f…; stamped 2026-09-26T20:34–21:24Z).** ALL-gold recall at M = 100 / 250 / 500 / 1000 / 2000 / 5000. On all three datasets, v1 FLAT and every `<cell>__HARD` equal the step-1 record on every gold node (MetaQA 14,489, MuSiQue 4,830, SQuAD 2,000), and both routing identities hold.
- **MetaQA** (1,998 rows). FLAT is 0.025 / 0.055 / 0.103 / 0.179 / 0.254 / 0.365.
  - IR_L0: LOC 0.408 / 0.435 / 0.463 / 0.501 / 0.541 / 0.636; FLAT+LOC 0.388 / 0.429 / 0.458 / 0.505 / 0.556 / 0.636. Against FLAT, FLAT+LOC gains / loses +730/−5 at M = 100, +669/−18 at 1000 and +561/−19 at 5000.
  - FV_L0: LOC 0.360 / 0.454 / 0.483 / 0.503 / 0.541 / 0.636; FLAT+LOC 0.202 / 0.418 / 0.475 / 0.520 / 0.557 / 0.636.
  - Every rung, in both arms, is above FLAT at every budget. FLAT needs 6,002 / 7,665 / 9,119 / 12,248 / 15,982 / 22,208 nodes to match IR_L0 FLAT+LOC.
  - At M = 5000 every rung and both arms give 0.636, with identical strata. The support (mean 2,015 nodes) fits in the budget for almost every row, so the rungs serve the same set.
  - Below M = 1000, LOC alone is at or above FLAT+LOC (IR_L0 0.408 vs 0.388 at 100; FV_L0 0.360 vs 0.202): where FLAT is weak, its ranks dilute the fusion. From M = 1000, FLAT+LOC is ahead (except FV_L2 at 1000: 0.446 vs 0.452).
  - Per hop, IR_L0 FLAT+LOC at M = 100 / 1000 / 5000:
    - hop1 0.990 / 0.999 / 1.000 (FLAT 0.059 / 0.261 / 0.437);
    - hop2 0.074 / 0.273 / 0.457 (FLAT 0.017 / 0.147 / 0.341);
    - hop3 0.101 / 0.243 / 0.452 (FLAT 0.000 / 0.129 / 0.317).
  - By answer-set size at M = 1000 (1 / 2 / 3 / 4 / 5–10 / 11+ golds; 819 / 286 / 180 / 87 / 287 / 339 rows): 0.801 / 0.573 / 0.444 / 0.402 / 0.185 / 0.062, vs FLAT 0.347 / 0.157 / 0.083 / 0.081 / 0.017 / 0.006.
  - ANY (at 100 / 1000 / 5000) is 0.608 / 0.810 / 0.917 vs FLAT 0.118 / 0.496 / 0.765. FRAC is 0.445 / 0.623 / 0.771 vs 0.042 / 0.281 / 0.553.
  - **Against steps 1–2.** HARD LOC is 0.032 … 0.192 (H4_SK) and 0.028 … 0.201 (PHG). Step 2's MEMACT LOC was 0.316 / 0.433 / 0.597 at 100 / 1000 / 5000 on H4_SK and 0.291 / 0.466 / 0.625 on PHG.
    - IR_L0 LOC vs MEMACT, paired: +515/−332, +381/−245 and +227/−148 on H4_SK; +529/−296, +396/−325 and +199/−177 on PHG.
    - MEMACT still wins 148–177 queries at 5000. That is consistent with its blocks holding golds that one hop does not reach (the routed hop3 gain in 32.5).
- **MuSiQue** (2,000 rows). FLAT is 0.538 / 0.631 / 0.695 / 0.757 / 0.814 / 0.884 (corrected 2026-09-29 from the exact counts 1,075 / 1,262 / 1,389 / 1,513 / 1,628 / 1,767; first printed as 0.537, 0.756 and 0.883, see §37.3).
  - FLAT+LOC by rung:
    - IR_L2: 0.648 / 0.747 / 0.798 / 0.845 / 0.880 / 0.928, i.e. +277/−55, +270/−38, +248/−40, +207/−31, +166/−33 and +114/−25 vs FLAT;
    - IR_L1: 0.639 / 0.742 / 0.793 / 0.840 / 0.880 / 0.927;
    - IR_L0: 0.633 / 0.729 / 0.788 / 0.835 / 0.876 / 0.926;
    - FV_L0: 0.553 / 0.664 / 0.740 / 0.816 / 0.870 / 0.926.
  - Every FLAT+LOC rung is above FLAT at every budget. FLAT needs 311 / 908 / 1,701 / 2,878 / 4,642 / 9,977 nodes to match IR_L2 FLAT+LOC.
  - LOC alone is below FLAT at every budget, e.g. IR_L2 0.291 / 0.410 / 0.499 / 0.581 / 0.660 / 0.770. On text, the hits' neighbourhoods add to the hits; they do not replace them.
  - Per hop, IR_L2 FLAT+LOC vs FLAT at M = 100 / 1000 / 5000:
    - hop2 (1,364 rows): 0.776 / 0.914 / 0.966 vs 0.672 / 0.847 / 0.935;
    - hop3 (442): 0.462 / 0.776 / 0.896 vs 0.324 / 0.640 / 0.837;
    - hop4 (194): 0.175 / 0.510 / 0.737 vs 0.083 / 0.387 / 0.629.
  - Per source: dev (417 rows) 0.585 / 0.815 / 0.880 vs FLAT 0.420 / 0.652 / 0.818; train (1,583) 0.665 / 0.852 / 0.941 vs 0.569 / 0.784 / 0.901.
  - ANY is 0.99–1.00 for FLAT and every FLAT+LOC arm. FRAC (IR_L2 FLAT+LOC) is 0.834 / 0.934 / 0.972 vs FLAT 0.779 / 0.895 / 0.953.
  - **Against steps 1–2.** HARD FLAT+LOC is 0.502 / 0.593 / 0.665 / 0.739 / 0.803 / 0.870, below FLAT everywhere, as in §31.3. MEMACT FLAT+LOC was 0.499 / 0.730 / 0.874 at 100 / 1000 / 5000. IR_L2 FLAT+LOC vs MEMACT is +317/−19, +241/−11 and +120/−12.
- **SQuAD** (2,000 rows, one gold each, so ALL = ANY = FRAC). FLAT is 0.994 / 0.997 / 0.999 / 1.000 / 1.000 / 1.000: the ceiling.
  - FLAT+LOC is 0.985–0.989 at M = 100, 0.995–0.996 at 250, 0.997–0.998 at 500, and 1.000 from 1000 on every rung. The losses vs FLAT: 10–19 queries at 100, 4 at 500, 1 at 1000 and none from 2000. The mechanism is the fusion itself: RRF interleaves LOC-ranked nodes with FLAT's, so a gold with a middling FLAT rank and a weak LOC rank can fall below the budget.
  - LOC alone is far below FLAT at small M and rises with the rung: at M = 100, FV_L0 0.462, IR_L0 0.624, IR_L1 0.744, IR_L2 0.825, FV_L2 0.852. At 5000 it is 0.981–0.983. The reach is 0.983 (32.6), so 1.7% of golds lie outside the support.
  - HARD LOC is 0.469 … 0.998 (H4_SK) and 0.462 … 0.999 (PHG). MEMACT LOC was 0.317 / 0.752 / 0.953 (H4_SK).

**32.4 The ladder and the hit weight** (paired on ALL gold; gained / lost, McNemar p descriptive).
- **g (L0 → L1), under IR, changes the KB by at most 12 of 1,998 queries and gains on text.**
  - MetaQA: LOC +6/−6 at M = 100, +14/−4 at 1000. FLAT+LOC +4/−16 and +7/−1.
  - MuSiQue: LOC +93/−11 and +99/−6. FLAT+LOC +40/−27 and +12/−2.
  - SQuAD: LOC +244/−4 and +49/−0. FLAT+LOC +2/−0.
  - Under FV, g costs MetaQA's LOC at M = 100 (+43/−150) and still gains in FLAT+LOC (+93/−63).
- **h (L1 → L2) splits by regime.**
  - MetaQA loses: IR LOC +0/−79 at 100 and +2/−85 at 1000; FLAT+LOC +3/−88 and +3/−68. FV_L1 → FV_L2 FLAT+LOC: +49/−220 and +0/−140.
  - MuSiQue gains: IR LOC +118/−9 and +105/−13; FLAT+LOC +43/−25 and +15/−5.
  - SQuAD gains in LOC (+165/−4 and +34/−2); FLAT+LOC is tied.
  - This matches §31.5's degree control. h divides by target degree, and MetaQA's golds are degree-rich: nodes of degree > 100 are 0.24% of its nodes and 21% of its golds.
- **Hit weight (FV → IR).** IR is ahead at small M in FLAT+LOC:
  - MetaQA: +421/−50 (L0), +377/−48 (L1), +432/−17 (L2) at M = 100. At M = 1000, FV is ahead on L0 and L1 (+3/−33, +9/−25). FV's LOC is also ahead on MetaQA hop2 / hop3 at M = 100 (0.146 vs 0.108; 0.138 vs 0.119), IR on hop1 (0.997 vs 0.797).
  - MuSiQue: +181/−21, +155/−19, +142/−28 at 100; +47/−9, +41/−14, +39/−16 at 1000.
  - SQuAD: FLAT+LOC is tied (+2/−4 … +1/−6). In LOC, IR leads at L0 (+409/−85) and FV at L2 (+48/−103).
- **One variant is never far from the best.** Gap of each FLAT+LOC variant to the best FLAT+LOC variant at the same M, maximized over M ∈ {100, 250, 500, 1000, 2000, 5000} and the three datasets:
  - IR_L1: 0.015 (MetaQA at 500, behind FV_L0; MuSiQue ≤ 0.009 behind IR_L2; SQuAD ≤ 0.003);
  - IR_L0: 0.018;
  - every other variant: at least 0.049 somewhere (IR_L2 on MetaQA at 100; the FV rungs 0.17–0.26 on MetaQA at 100).
  - This is noted, not selected. The per-regime best rungs are IR_L0 / FV_L0 on the KB and IR_L2 on MuSiQue.

**32.5 L1b: partition routing with two budgets.** A cell's frozen partition is a shard map only (32.2). Routed = the first B_N nodes of the served order O inside the first B_P partitions. When B_N reaches the contacted mass, routing serves the contacted partitions whole; below it, a partition returns only its best nodes by the served score. The user's examples: *"The new system can have B_P=10 but B_N=500. Or B_P=50 and still B_N=1000."*
- **MetaQA** (432 partitions of about 100 nodes; contacted mass at B_P = 1 / 2 / 5 / 10 / 20 / 50 / 100 is 100 / 200 / 500 / 1,002 / 2,007 / 5,021 / 10,032 on H4_SK and 101 … 10,147 on PHG).
  - **Routing by L returns more gold than the unrouted arm at the same B_N, from 10 partitions.** H4_SK / PHG vs unrouted:
    - IR_L0 · LMAX: (10, 100) 0.431 / 0.423 vs 0.388; (10, 500) 0.532 / 0.530 vs 0.458; (10, 1000) 0.561 / 0.562 vs 0.505; (20, 2000) 0.603 / 0.621 vs 0.556; (50, 5000) 0.635 / 0.670 vs 0.636.
    - IR_L0 · LSUM: (10, 500) 0.518 / 0.536; (20, 1000) 0.583 / 0.604; (50, 1000) 0.564 / 0.584; (50, 5000) 0.641 / 0.678.
    - At the user's points: (10, 500) gives 0.518–0.536 vs unrouted 0.458 and FLAT@500 0.103; (50, 1000) gives 0.564–0.587 (LMAX, LSUM) vs 0.505.
    - IR_L1 is within 0.02 of IR_L0 at these points, e.g. LMAX (10, 500) 0.526 / 0.515 and LSUM (50, 5000) 0.644 / 0.679.
    - The largest routed-minus-unrouted margins on the grid are +0.103 (H4_SK, FV_L0 · LMAX at (20, 100)) and +0.112 (PHG, IR_L2 · LSUM at (20, 500)).
  - For comparison, the unrouted IR_L0 FLAT+LOC top-M touches P* = 57 / 218 / 325 / 403 / 431 partitions (H4_SK; M = 100 / 500 / 1000 / 2000 / 5000). LMAX and LSUM match (within 0.01) or beat the unrouted arm at B_P = 10 for every B_N ≤ 2000.
  - **Where the extra gold is: multi-hop answers co-located in the contacted partitions.**
    - IR_L0 · LSUM hop3 is 0.395 / 0.461 at (20, 1000) vs the unrouted 0.243, and 0.482 / 0.572 at (50, 5000) vs 0.452.
    - hop2 is 0.384 / 0.390 at (20, 1000) vs 0.273.
    - At these two points hop1 stays at 0.96–0.995.
    - The L-selected partitions are served nearly whole (B_N ≈ contacted mass), and they carry 2–3-hop golds that one hop does not reach (32.6). This is the static co-location the partitioner built. Once B_N ≥ 1000, PHG is ahead of H4_SK by up to 0.04.
  - **An L rule beats first appearance (S) at all 252 grid points of both MetaQA cells.** S needs B_P = 50 to reach the unrouted arm: IR_L0 · S (10, 500) is 0.165 / 0.151; (50, 1000) is 0.556 / 0.571.
  - **FLAT · S.** At (1, 100), the partition of the top FLAT hit served whole gives 0.116 / 0.106 vs FLAT@100 0.025. With more partitions or nodes it never exceeds 0.146, and (50, 5000) is 0.173 / 0.185 vs unrouted 0.365. Beyond the top hit's own partition, the partitions of FLAT's top nodes do not hold the answers.
- **MuSiQue** (1,175 partitions; mass 102 / 203 / 508 / 1,016 / 2,030 / 5,070 / 10,131).
  - **No routed point is more than 0.003 above the unrouted arm.**
    - IR_L2 · S: (10, 500) 0.438 vs 0.798; (20, 1000) 0.636 vs 0.845; (50, 1000) 0.781; (100, 1000) 0.822; (50, 5000) 0.791; (100, 5000) 0.858 vs 0.928.
    - S is the best rule from B_P = 10. At B_P ≤ 5, an L rule leads by up to 0.023 on the IR_L1 / IR_L2 orders. At (50, 1000), LMAX gives 0.689 and LSUM 0.709.
  - **Routing comes within 0.01 of unrouted only as B_P approaches P*(B_N).** Unrouted IR_L2 FLAT+LOC touches P* = 42 / 163 / 277 / 446 / 764 partitions at M = 100 … 5000; FLAT touches 51 / 183 / 296 / 455 / 723. IR_L2 · S is within 0.01 at B_P = 50 for B_N = 100 and at B_P = 100 for B_N = 500. For B_N ≥ 1000 it is not within 0.01 at any B_P ≤ 100.
  - **Against unrouted FLAT, the node score still gains at the user's second point.** IR_L2 · S at (50, 1000) is 0.781 and at (100, 1000) 0.822, vs FLAT@1000 0.756. At (10, 500) it is 0.438 vs FLAT@500 0.695.
  - FLAT · S: (10, 500) 0.495 vs 0.695; (50, 1000) 0.701 vs 0.756; (100, 5000) 0.803 vs 0.883.
- **SQuAD** (202 partitions; mass 101 / 202 / 505 / 1,010 / 2,019 / 5,040 / 10,066).
  - FLAT · S at B_P = 1 already serves 0.820: for 82% of queries the single gold sits in the partition of FLAT's top hit.
  - No routed point is more than 0.001 above unrouted. S is within 0.01 of unrouted from B_P = 20–50: FLAT · S 0.988 at (20, 100) vs 0.994; IR_L1 · S 0.982 vs 0.986; both 0.997 at (50, 1000).
  - LMAX and LSUM are lower at small B_P (IR_L1 · LSUM (5, 100) 0.792 vs S 0.883). An L rule leads S only at B_P ≤ 5 on the IR_L2 order, by at most 0.014.
  - P* = 35 / 109 / 152 / 185 / 201 (FLAT) and 26 / 89 / 133 / 173 / 199 (IR_L1 FLAT+LOC).

**32.6 Reach and support** (per gold node, the families that connect it to the top-200 FLAT hits; the exclusive share, reached by no other family, in brackets).
- **MetaQA.**
  - Support: mean 2,015 nodes with L > 0 per query (median 1,956; 1,051–7,971); from the top-10 hits alone, 108.
  - The families reach 0.423 of gold nodes: STRUCT_out 0.258 (0.234), STRUCT_in 0.127 (0.116), KNN 0.071 (0.037), NER 0.008 (0.000). From the top-10 hits: 0.164.
  - By hop: hop1 1.000; hop2 0.402 (mostly STRUCT_in, 0.275); hop3 0.344 (mostly STRUCT_out, 0.301).
  - Of the 7,908 gold nodes FLAT misses at 5000, 0.306 are reached (STRUCT_in 0.171, STRUCT_out 0.133).
  - So hop2 and hop3 are capped by one-hop reach together with FLAT: their ALL curves flatten at 0.457 and 0.452, vs FLAT@5000 0.341 and 0.317.
- **MuSiQue.**
  - Support: mean 7,473 (median 6,058; 1,282–43,705); from the top-10, 643.
  - Reach 0.876: KNN 0.708 (0.098), NER 0.588 (0.053), STRUCT_in 0.480 (0.034), STRUCT_out 0.363 (0.023). From the top-10: 0.592.
  - FLAT-missed@1000 (570 gold nodes): 0.663, i.e. NER 0.437 (0.163), STRUCT_out 0.291 (0.083), STRUCT_in 0.242 (0.072), KNN 0.154 (0.026).
  - FLAT-missed@5000 (259): 0.587, i.e. NER 0.344 (0.166), STRUCT_out 0.228 (0.070), STRUCT_in 0.201 (0.104), KNN 0.070 (0.027).
  - As in §31.5, KNN reaches the golds FLAT already serves; NER carries the missing ones.
- **SQuAD.**
  - Support: mean 5,513 (median 5,500; 2,143–9,200); from the top-10, 527.
  - Reach 0.983: KNN 0.953 (0.133), NER 0.667, STRUCT_in 0.456, STRUCT_out 0.344. From the top-10: 0.837.
  - FLAT misses 13 golds at M = 100 and none from 1000.
- **Cross-check.** On MetaQA and MuSiQue, the new nodes per query outside FLAT@M and the missing golds found reproduce the edge diagnostic's SKN cells exactly (§31.5; top-10 and top-200 hits at M = 100 / 1000 / 5000; all_match). SQuAD has no edge-diagnostic record.

**32.7 Reading** (development numbers: descriptive; selection is the user's step).
- **The L1 question.** *"Does static one-hop graph localization improve the node ranking at the exact same exposure?"*
  - On the KB and on multi-hop text: yes, at every budget and on every rung of FLAT+LOC.
    - MetaQA IR_L0: 0.388 vs FLAT 0.025 at M = 100; 0.636 vs 0.365 at 5000.
    - FLAT needs 4.4–60× the exposure to match MetaQA IR_L0 FLAT+LOC.
    - MuSiQue IR_L2: +0.111 at 100 to +0.045 at 5000, a 2.0–3.6× FLAT-exposure equivalent.
  - On SQuAD, FLAT is at the ceiling. FLAT+LOC gives up 10–19 of 2,000 queries at M = 100, 4 at 500, 1 at 1000 and none from 2000.
  - One edge set, one sparse product, no learned weight.
  - Blocks are no longer part of the score, and step 2's text loss (§31.3) is gone: every FLAT+LOC rung beats both FLAT and MEMACT on MuSiQue.
- **Where it gains.**
  - KB: the anchor → answer edge. hop1 is solved at M = 100 (0.990).
  - Text: every hop stratum gains, and the relative gain grows with the hop count (hop4 0.083 → 0.175 at 100).
- **What one hop leaves.** MetaQA's hop2 / hop3 golds are 0.40 / 0.34 reachable from the hits, and their curves flatten at 0.46 / 0.45. By the ruling, A^⊤(A^⊤x) is L3.
- **Fusion.** RRF of FLAT and LOC is the only arm that is at or above FLAT on all three datasets from M = 1000. LOC alone leads on the KB below 1000 and is far below FLAT on text.
- **The ladder is regime-dependent only in h.** Under IR, g barely moves the KB and helps text. h helps text and costs the KB's degree-rich golds. IR_L1 (inverse rank · hit spread, no target-degree term) is within 0.015 of the best FLAT+LOC variant everywhere; IR_L0 is within 0.018.
- **Two budgets, and the principle, in the data.**
  - On text, the partition layer adds no relevance. Node scores carry the relevance, and routing is lossless only as B_P approaches the partitions the unrouted top-B_N touches (P* ≈ 0.28–0.30 × B_N at B_N = 1000 on MuSiQue). *"Partitions localize computation and data; nodes localize relevance"* holds literally there.
  - On the KB, the partitions also localize relevance. L-routed B_P = 10 returns more gold than unrouted at the same B_N.
    - The extra gold is 2–3-hop answers co-located with the L mass, served when B_N is close to the contacted mass.
    - That is static co-location, not traversal: one fixed incidence (node → partition) applied to the node scores. It reaches past one hop.
  - The rules split by regime. From B_P = 10, first appearance (S) is the best rule on text; on the KB it is below an L rule at every grid point. No single rule is best on both.
- **For the user** (no mechanism is selected, and nothing is frozen):
  - (a) Which node score goes forward to selection: IR_L1, the only variant never more than 0.015 from the best, or a per-regime rung.
  - (b) The routing rule and B_P policy. A fixed B_P, or B_P derived from B_N through P*, is a systems choice the numbers above price.
- **Answered 2026-09-27** (§33.1): (a) IR_L1 on every dataset, without the target-degree term; (b) neither: B_P is query-adaptive, ⌈N_eff⌉ of the partition-score distribution, under ONE aggregation rule chosen on development evidence (§33).

**32.8 Cost and hygiene.**
- **Wall-clock and memory:** MetaQA 922 s, peak RSS 775 MB; MuSiQue 2,058 s, 1,105 MB; SQuAD 496 s, 458 MB. Run one at a time, 02:04–03:03 local.
- **Per-query latency** (MetaQA / MuSiQue / SQuAD):
  - FLAT: products 14.7 / 58.7 / 10.2 ms (amortized) + RRF 55 / 141 / 23 ms;
  - node score + LOC order: 3.1–3.2 / 9.2–9.9 / 7.0–7.5 ms per rung;
  - fused order: 18–19 / 54–56 / 8.6–9.1 ms per rung;
  - routing: 1–2 ms per configuration.
  - Graph index: 6.0 / 41.8 / 10.5 MB.
- **Host at the starts:** 0.89–1.36 GiB available, 11.5–11.9 GiB of swap in use. Foreign pid 12096 (7.9–8.4 GB RSS) ran throughout, untouched.
- **Nothing under `data/` was written. No pinned module or CONTRACT_FILE was edited.** The records pin `_l1d_node1h.py` (8265d6bc…), `_l1d_arms.py` (f5f52b79…), `_l1d_lib.py` (dafe39b2…) and, through `structures.imports`, `_l1d_edgediag.py` (874bfc49…).
- **The summary module** `_l1d_node1h_summary.py` (1154c0d1…) is new. Its per-hop routing configurations were chosen before its first v1 run, and the smoke outputs (outside the repository) were not used.

## 33. ADAPTBP: query-adaptive partition fan-out B_P(q) from partition-score concentration (2026-09-27 local) (`_l1d_adaptbp.py`, summary `_l1d_adaptbp_summary.py`, post-hoc view `_l1d_adaptbp_view.py`, on the shared runner `_l1d_arms.py`; records `results/L1_DEV/adaptbp_{metaqa,musique,squad}__v1.{json,npz}`, `adaptbp_SUMMARY__v1.json`, `adaptbp_VIEW__v1.json`; STATUS = DEVELOPMENT: descriptive numbers, no pre-registration, no verdicts, no adoption)

**33.1 The decisions (2026-09-27 local).** The user answered 32.7's two questions in one message, sent as pasted text and taken as the user's decisions. The operative sentences, verbatim:
- **Decision 1, the node score.**
  - *"Carry IR_L1 universally"*; *"Do not choose a different node scorer by dataset type."*
  - The score is L(u|q) = Σ_{s∈H_q} (1/rank(s))·(1/g(spread(s)))·1[(s,u)∈E] *"with the frozen universal edge union."*
  - *"I would not include the target-degree term in the canonical candidate."*
  - *"IR_L1 = universal L1a candidate"*; *"No dataset branch."*
- **Decision 2, the fan-out.** *"Do NOT use a globally fixed B_P"*. *"But I also would not choose: B_P=cB_N with a global constant c."* Instead: *"make B_P query-adaptive from partition-score concentration while keeping the algorithm dataset-independent."*
  - The quantities:
    - a_j = Σ_{u∈P_j} L(u|q), *"using only the candidate/localization scores, not gold"*;
    - p_j = a_j/Σ_k a_k;
    - N_eff = 1/Σ_j p_j²;
    - B_P(q) = ⌈N_eff(q)⌉.
  - The cap: *"possibly bounded by an external systems maximum: B_P(q)=min(B_P^max,⌈N_eff⌉)"*. *"B_P^max becomes a deployment resource constraint, not a retrieval hyperparameter."*
  - The alternative is H(P|q) = −Σ_j p_j log p_j with B_P(q) = ⌈e^H⌉. *"I slightly prefer the participation ratio because it's simpler to interpret and compute."*
- **The aggregation.** *"There is one subtle issue we should measure next: The partition aggregation itself"*, with three candidates: sum, max and top-k.
  - *"I would not dataset-switch these either. The system should have one aggregation rule."*
  - *"My starting candidate would be top-k sum with a tiny fixed k"*. The reasons given: max is too sensitive to one accidental node; sum rewards large or noisy partitions; top-k measures repeated evidence without letting block size dominate.
  - *"use those development results to choose the most universal one before final evaluation."*
- **Surfaces.** *"B_P and B_N are genuinely independent… We should report surfaces: R(B_P,B_N) not just: R(M)."* The example points are (5, 500), (20, 500), (20, 2000), (100, 1000) and (unrestricted, 1000).
- **The experiment.**
  - *"Take the already-computed IR_L1 node scores and evaluate: B_P(q)=⌈1/Σ_j p_j²⌉ against the full fixed-B_P routing grid you already measured."*
  - The test: whether *"that one parameter-free rule lands near the good part of the routing surface on MetaQA, MuSiQue and SQuAD simultaneously"*.
  - The principle: *"Partitions localize computation; nodes localize relevance; query concentration determines distributed fan-out."*

This answers 32.7's (a) and (b). Nothing is frozen. The aggregation is the user's choice, on the development evidence below.

**33.2 Design** (`_l1d_adaptbp.py`, sha 984e3aef…; a new module on the shared runner that imports `_l1d_node1h.py` read-only).
- **Node score, served order and population are those of §32.**
  - IR_L1: hits FLAT[:200]; E = STRUCT_out ∪ STRUCT_in ∪ KNN ∪ NER; g_f(s) = 1/log2(1 + deg_f^+(s)); no h.
  - Served through the IR_L1 FLAT+LOC order O (RRF, K0 = 60).
  - The node1h records hold per-gold positions but no per-node scores. So the harness recomputes the score per query and asserts that it is the §32 computation (the identities below).
- **Aggregations.** For each partition P_j of a cell:
  - SUM: a_j = Σ L, §32's LSUM;
  - MAX: a_j = max L, §32's LMAX;
  - TOPk: a_j = the sum of the k largest L in P_j. k = 3 was fixed before the first run as the user's "tiny fixed k". k = 2, 5 and 10 are a declared sensitivity band, not a search.
- **The rule.** Each aggregation ranks the partitions: descending a_j, with ties and a_j = 0 falling back to first appearance in O, as in §32. Its own p_j then give the count:
  - NEFF: B_P(q) = ⌈N_eff⌉;
  - EXPH: B_P(q) = ⌈exp H⌉.
  - The ceilings use a 1e−9 tolerance. Both counts are at most the number of partitions with a_j > 0.
  - A query without evidence (Σ a_j = 0) would contact every partition. No development query has zero evidence.
- **Arms**, named ranking·aggregation·count (the records write `OWN|TOP3|NEFF`); 2 rankings × 6 aggregations × 2 counts = 24:
  - OWN·agg·NEFF and OWN·agg·EXPH are the candidates: the aggregation both ranks and counts;
  - S·agg·NEFF and S·agg·EXPH are a reference: partitions ranked by first appearance in O (§32's best text rule), with the aggregation's count.
  - "L ranking" below means an OWN arm's ranking: the partitions in descending a_j.
- **The fixed grid, extended to every integer.** 7 rankings (S plus the 6 aggregations) × every B_P from 1 to |partitions| × every B_N of the M curve.
  - Per query, the harness stores the interval [LO, HI(B_N)] of fan-outs at which all of the query's golds are returned. LO is the fan-out that contacts every gold partition.
  - This is one interval because a contacted gold's routed rank can only grow with B_P. So contacting more partitions can also lose a gold, once the added shards push it past B_N.
  - R(B_P, B_N) is the share of queries whose interval contains B_P.
- **References at each B_N:**
  - unrouted;
  - the same ranking's fixed curve at the arm's mean fan-out (interpolated between integers; paired exactly at the rounded mean);
  - the S curve at that fan-out;
  - the same ranking's best fixed B_P;
  - the best fixed point of any ranking, a gold-chosen hindsight envelope;
  - the smallest fixed B_P of the same ranking that reaches the arm's ALL.
- **Caps** min(B_P^max, B_P(q)) for B_P^max ∈ {5, 10, 20, 50, 100}. This is a deployment table only.
- **Gold diagnostic** (never a rule):
  - LO per ranking and its rank correlation with N_eff;
  - the share of queries whose B_P(q) contacts every gold partition;
  - the share over-contacted: servable at a smaller fan-out but lost at B_P(q).
- **Identities asserted on every gold node:**
  - FLAT equals the step-1 record;
  - IR_L1's LOC and FLAT+LOC positions equal `node1h_<ds>__v1.npz`;
  - the routed rank of every gold and the contacted masses under S / MAX / SUM, at B_P ∈ {1, 2, 5, 10, 20, 50, 100}, equal §32's `route_rr` / `route_mass` for IR_L1|S / LMAX / LSUM in every cell;
  - contacting every partition equals the unrouted arm, per row and per ranking;
  - each adaptive arm's per-gold ALL equals its interval test.
- Smoke runs of 200 rows per dataset, outside the repository, checked the identities before the launch. Their numbers are not used.

**33.3 The fan-out B_P(q)** (records `adaptbp_metaqa__v1.json` sha 1be6a78e…, `adaptbp_musique__v1.json` sha 8477f48b…, `adaptbp_squad__v1.json` sha f07cdeb1…, each with its `.npz`; summary `adaptbp_SUMMARY__v1.json` sha 8426849f…; stamped 2026-09-26T22:25–22:46Z). Every identity of 33.2 holds on every gold node (MetaQA 14,489, MuSiQue 4,830, SQuAD 2,000) and in every cell. No development query is without evidence. Mean / median / p90 / max of B_P(q); an S arm uses the count of the OWN arm with the same aggregation. All numbers in 33.3–33.8 come from the exact per-query counts in the npz (33.9): shares to 3 decimals and means to 1, halves rounded away from zero; 4 decimals where the text needs a finer value.

| B_P(q): mean / median / p90 / max | MetaQA H4_SK | MetaQA PHG | MuSiQue | SQuAD H4_SK | SQuAD PHG |
|---|---|---|---|---|---|
| partitions; with a_j > 0 (mean) | 432; 400 | 432; 399 | 1,175; 872 | 202; 198 | 202; 199 |
| OWN·SUM·NEFF | 36.1 / 34 / 57 / 159 | 41.2 / 39 / 63 / 161 | 47.9 / 40 / 85 / 403 | 26.7 / 24 / 46 / 104 | 29.2 / 26 / 49 / 108 |
| OWN·MAX·NEFF | 54.9 / 54 / 74 / 297 | 56.6 / 56 / 76 / 285 | 96.6 / 82 / 164 / 664 | 42.7 / 38 / 68 / 146 | 45.9 / 41 / 73 / 152 |
| OWN·TOP3·NEFF | 49.5 / 48 / 70 / 248 | 53.3 / 53 / 74 / 243 | 84.0 / 68 / 150 / 680 | 38.5 / 34 / 65 / 154 | 41.7 / 37 / 70 / 158 |
| OWN·SUM·EXPH | 108.2 / 109 / 145 / 269 | 116.8 / 117 / 151 / 265 | 137.5 / 122 / 235 / 722 | 55.6 / 54 / 85 / 152 | 59.8 / 58 / 90 / 156 |
| OWN·MAX·EXPH | 128.3 / 128 / 153 / 329 | 129.2 / 130 / 155 / 318 | 226.1 / 209 / 356 / 942 | 75.2 / 74 / 105 / 180 | 78.8 / 77 / 110 / 183 |
| OWN·TOP3·EXPH | 128.0 / 129 / 156 / 301 | 131.8 / 133 / 161 / 293 | 210.2 / 190 / 346 / 934 | 71.9 / 71 / 104 / 185 | 75.6 / 75 / 110 / 187 |

- **N_eff is 5–23% of the partitions with evidence.** The one-hop support touches almost every partition (MetaQA 399–400 of 432, MuSiQue 872 of 1,175, SQuAD 198–199 of 202). The count therefore comes from how the mass is concentrated, not from the support.
- **The aggregation orders the counts the same way in every cell:** SUM < TOP10 < TOP5 < TOP3 < TOP2 < MAX. SUM gives the most concentrated distribution, MAX the flattest. Mean NEFF counts of the band, cells in T1's order:
  - TOP2 52.2 / 54.9 / 90.0 / 40.4 / 43.6;
  - TOP5 46.3 / 51.3 / 75.5 / 36.1 / 39.4;
  - TOP10 42.4 / 47.9 / 63.9 / 32.9 / 36.2.
- **exp H is 1.7–3.0× N_eff** (never less).
- **The count is largest on MuSiQue, and within a dataset it grows slowly with the hop count.** OWN·TOP3·NEFF means: MetaQA hop1 / 2 / 3 46.1 / 48.9 / 53.5 (PHG 49.6 / 53.0 / 57.4); MuSiQue hop2 / 3 / 4 79.0 / 93.0 / 98.6.

**33.4 Against the fixed grid, per cell.** ALL gold at each B_N, with each arm's mean fan-out in brackets. The references (33.2) are:
- the TOP3 ranking at a fixed B_P equal to OWN·TOP3·NEFF's mean fan-out (interpolated between integers);
- that ranking's best fixed B_P at each B_N;
- the envelope, the best fixed point of any of the 7 rankings.

The last two are chosen with the gold (hindsight). Paired tests are on ALL gold at B_N = 1000: gained / lost for the arm named second, McNemar p descriptive.

**MetaQA** (1,998 rows; 432 partitions per cell; H4_SK; PHG).

| ALL gold (H4_SK; PHG) at B_N = | 100 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|
| unrouted (every partition) | 0.382 | 0.460 | 0.508 | 0.556 | 0.636 |
| OWN·SUM·NEFF [36.1; 41.2] | 0.400; 0.404 | 0.529; 0.534 | 0.572; 0.590 | 0.600; 0.625 | 0.627; 0.666 |
| OWN·MAX·NEFF [54.9; 56.6] | 0.400; 0.402 | 0.521; 0.522 | 0.572; 0.581 | 0.602; 0.620 | 0.630; 0.658 |
| OWN·TOP3·NEFF [49.5; 53.3] | 0.401; 0.400 | 0.525; 0.527 | 0.572; 0.585 | 0.598; 0.621 | 0.630; 0.662 |
| TOP3 ranking at a fixed B_P = OWN·TOP3·NEFF's mean (interpolated) | 0.399; 0.397 | 0.521; 0.523 | 0.569; 0.583 | 0.600; 0.624 | 0.640; 0.662 |
| TOP3 ranking, best fixed B_P (hindsight) | 0.419 (16); 0.424 (15) | 0.548 (20); 0.561 (19) | 0.582 (26); 0.600 (27) | 0.605 (41); 0.628 (19) | 0.640 (48); 0.674 (49) |
| envelope: best fixed point of any ranking (hindsight) | 0.432 (MAX 13); 0.434 (MAX 15) | 0.551 (MAX 20); 0.562 (TOP5 18) | 0.582 (TOP3 26); 0.603 (SUM 22) | 0.607 (SUM 20); 0.638 (SUM 20) | 0.644 (TOP10 49); 0.681 (SUM 49) |
| OWN·TOP3·EXPH [128.0; 131.8] | 0.383; 0.385 | 0.473; 0.478 | 0.531; 0.539 | 0.584; 0.598 | 0.637; 0.655 |
| S·TOP3·NEFF [49.5; 53.3] | 0.375; 0.373 | 0.492; 0.497 | 0.538; 0.547 | 0.565; 0.582 | 0.597; 0.622 |
| S·TOP3·EXPH [128.0; 131.8] | 0.382; 0.382 | 0.473; 0.477 | 0.528; 0.537 | 0.579; 0.590 | 0.628; 0.650 |

- **Every OWN·*·NEFF arm is above unrouted at every B_N ≤ 2000, in both cells.** The margin is +0.06–0.08 at B_N 500–1000.
  - Paired at 1000: TOP3 +167/−40 and +185/−32; MAX +165/−38 and +178/−33; SUM +180/−53 and +203/−39.
  - At 5000 the margin is between −0.01 and +0.03.
- **T2's three arms sit close to the best fixed B_P of their own ranking:** within 0.039 at every B_N, and within 0.018 from B_N = 1000 (TOP3 0.572 vs 0.582 at B_P 26; 0.585 vs 0.600 at 27).
  - The TOP3 ranking peaks at B_P 15–27 for B_N ≤ 1000. N_eff's mean is about twice that.
  - That point is on a flat stretch of the surface (T7: TOP3 0.576 at B_P 20, 0.568 at 50).
- **Adaptivity adds nothing over the same mean held fixed.** The interpolated difference is −0.010 to +0.005 over B_N and the three aggregations. Paired against the same ranking at the rounded mean fan-out, at 1000:
  - TOP3 +19/−15 (p 0.61) and +16/−14 (0.86);
  - MAX +21/−12 (0.16) and +15/−14 (1.00);
  - SUM +17/−28 (0.14) and +15/−20 (0.50).
- **The EXPH count (mean 128–132) keeps less of the gain.**
  - It is +0.023 / +0.031 over unrouted at 1000 (+69/−24, +75/−13).
  - That is 0.041 / 0.046 below TOP3·NEFF (NEFF → EXPH +22/−104, +22/−113).
  - At 5000 the two counts are tied (+73/−58, +62/−76).
- **With the S ranking, the same counts do worse than both the L rankings and S held fixed at the same mean.**
  - S·TOP3·NEFF is 0.538 / 0.547 at 1000 (OWN → S +5/−73, +7/−82).
  - Against S fixed at the rounded mean: +20/−57, +16/−66.
- **Per hop at 1000** (OWN·TOP3·NEFF, H4_SK / PHG; unrouted in brackets):
  - hop1 0.994 / 0.995 (1.000);
  - hop2 0.351 / 0.356 (0.281);
  - hop3 0.369 / 0.402 (0.243).
  - The gain is §32.5's co-located 2–3-hop answers.
  - Under EXPH, hop3 is 0.282 / 0.300. The wider fan-out pushes co-located golds past B_N: 0.116 / 0.136 of queries are over-contacted under EXPH, against 0.060 / 0.077 under NEFF (33.7).

**MuSiQue** (2,000 rows; 1,175 partitions).

| ALL gold at B_N = | 100 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|
| unrouted (every partition) | 0.640 | 0.793 | 0.840 | 0.880 | 0.928 |
| OWN·SUM·NEFF [47.9] | 0.538 | 0.607 | 0.622 | 0.634 | 0.640 |
| OWN·MAX·NEFF [96.6] | 0.605 | 0.699 | 0.722 | 0.737 | 0.748 |
| OWN·TOP3·NEFF [84.0] | 0.598 | 0.689 | 0.716 | 0.731 | 0.745 |
| TOP3 ranking at a fixed B_P = OWN·TOP3·NEFF's mean (interpolated) | 0.611 | 0.718 | 0.745 | 0.763 | 0.777 |
| S ranking at that fixed B_P | 0.640 | 0.779 | 0.810 | 0.827 | 0.839 |
| TOP3 ranking, best fixed B_P (hindsight) | 0.640 (1056) | 0.793 (1056) | 0.840 (764) | 0.880 (1056) | 0.928 (1056) |
| envelope: best fixed point of any ranking (hindsight) | 0.643 (S 45) | 0.794 (S 176) | 0.842 (S 273) | 0.882 (S 362) | 0.928 (S 640) |
| OWN·MAX·EXPH [226.1] | 0.633 | 0.770 | 0.808 | 0.835 | 0.861 |
| OWN·TOP3·EXPH [210.2] | 0.631 | 0.768 | 0.806 | 0.835 | 0.864 |
| S·TOP3·NEFF [84.0] | 0.639 | 0.767 | 0.797 | 0.813 | 0.824 |
| S·MAX·NEFF [96.6] | 0.639 | 0.777 | 0.806 | 0.827 | 0.840 |
| S·TOP3·EXPH [210.2] | 0.640 | 0.793 | 0.837 | 0.871 | 0.902 |
| S·MAX·EXPH [226.1] | 0.640 | 0.794 | 0.839 | 0.875 | 0.907 |

- **Every OWN·*·NEFF arm is below unrouted at every B_N.**
  - TOP3 is below by 0.042 at 100, 0.104 at 500, 0.124 at 1000 and 0.183 at 5000 (+27/−274 at 1000).
  - MAX is below by 0.118 at 1000 (+26/−262), SUM by 0.218 (+23/−458).
- **Here adaptivity is worse than the same mean held fixed.**
  - At 1000, TOP3·NEFF is 0.716, against 0.745 for the TOP3 ranking fixed at its mean fan-out (84).
  - Paired at the rounded mean: TOP3 +40/−97 (p 1e−6), MAX +33/−80 (1e−5), SUM +59/−126 (9e−7).
  - Over B_N and the three aggregations, the interpolated difference is −0.006 to −0.034.
  - Under EXPH the two tie (TOP3 +32/−35, p 0.81).
- **The count is short of what either ranking needs.**
  - At B_N = 1000, the TOP3 ranking comes within 0.01 of unrouted from B_P = 326 and peaks at 764. S does so from 167 and peaks at 273, which is the envelope (T7).
  - N_eff's mean is 84 (TOP3) and 97 (MAX).
  - exp H's mean, 210–226, is closer: TOP3·EXPH is −0.034 at 1000 (+15/−82).
- **With the S ranking, most of the loss is recovered at the same counts.**
  - S·TOP3·NEFF is 0.797 at 1000 (OWN → S +192/−30).
  - S·MAX·EXPH is 0.839, −0.0005 vs unrouted (+5/−6, p 1.00). S·TOP3·EXPH is 0.837 (+8/−14).
  - At 5000, S·MAX·EXPH is −0.021 (+24/−65, p 2e−5).
- **Per hop at 1000.** OWN·TOP3·NEFF, with unrouted and then the TOP3 ranking fixed at the overall rounded mean (84) in brackets:
  - hop2 0.819 (0.912; 0.848);
  - hop3 0.586 (0.760; 0.593);
  - hop4 0.289 (0.510; 0.361).
  - The other arms: S·TOP3·NEFF 0.883 / 0.699 / 0.412; OWN·TOP3·EXPH 0.887 / 0.713 / 0.448.
  - The count grows with the hop count (79 / 93 / 99). The loss grows faster (−0.093 / −0.174 / −0.222).

**SQuAD** (2,000 rows, one gold each; 202 partitions per cell; H4_SK; PHG). Unrouted is 0.9995 at B_N = 1000 (one query), shown as 1.000, and 1.000 from 2000. B_N = 2000 is left out because every arm has the same ALL at 1000, 2000 and 5000.

| ALL gold (H4_SK; PHG) at B_N = | 100 | 500 | 1000 | 5000 |
|---|---|---|---|---|
| unrouted (every partition) | 0.986 | 0.997 | 1.000 | 1.000 |
| OWN·SUM·NEFF [26.7; 29.2] | 0.967; 0.963 | 0.972; 0.970 | 0.972; 0.971 | 0.972; 0.971 |
| OWN·MAX·NEFF [42.7; 45.9] | 0.980; 0.980 | 0.986; 0.988 | 0.987; 0.989 | 0.987; 0.989 |
| OWN·TOP3·NEFF [38.5; 41.7] | 0.979; 0.979 | 0.986; 0.988 | 0.987; 0.989 | 0.987; 0.989 |
| TOP3 ranking at a fixed B_P = OWN·TOP3·NEFF's mean (interpolated) | 0.975; 0.976 | 0.980; 0.983 | 0.980; 0.983 | 0.980; 0.983 |
| TOP3 ranking, best fixed B_P (hindsight) | 0.986 (101); 0.986 (178) | 0.997 (143); 0.997 (178) | 1.000 (143); 1.000 (178) | 1.000 (181); 1.000 (178) |
| envelope: best fixed point of any ranking (hindsight) | 0.986 (S 38); 0.986 (S 42) | 0.998 (S 72); 0.998 (S 84) | 1.000 (S 127); 1.000 (S 100) | 1.000 (S 138); 1.000 (S 100) |
| OWN·TOP3·EXPH [71.9; 75.6] | 0.986; 0.984 | 0.996; 0.993 | 0.997; 0.996 | 0.997; 0.996 |
| S·TOP3·NEFF [38.5; 41.7] | 0.986; 0.986 | 0.996; 0.996 | 0.996; 0.997 | 0.996; 0.997 |
| S·TOP3·EXPH [71.9; 75.6] | 0.986; 0.986 | 0.997; 0.998 | 0.999; 1.000 | 0.999; 1.000 |

- **At B_N = 1000, the OWN·*·NEFF arms lose 0.011–0.029.**
  - TOP3 and MAX lose 0.011–0.013 (TOP3 +0/−26 and +1/−23).
  - SUM loses 0.028 / 0.029 (+0/−55, +1/−59).
- **Here adaptivity is better than the same mean held fixed.**
  - Paired at the rounded mean, at 1000: TOP3 +15/−2 (p 0.002) and +13/−2 (0.007); MAX +14/−3 (0.01) and +20/−3 (5e−4).
  - Interpolated, the difference is +0.003 to +0.009 over B_N and the three aggregations.
- **The count lands; the loss comes from the ranking.**
  - B_P(q) contacts every gold partition for 0.9865 / 0.9885 of queries (TOP3·NEFF). From B_N = 1000, no query is over-contacted (33.7).
  - A fixed TOP3 fan-out needs 56–59 partitions to come within 0.01 of unrouted; S needs 21–24 (T7).
  - With the S ranking, the same NEFF count is −0.004 / −0.003 at 1000 (S·TOP3·NEFF; +0/−7, +0/−5).
  - OWN·TOP3·EXPH is −0.003 / −0.004, and S·TOP3·EXPH is −0.001 / +0.0005.

**33.5 Universality** (for each arm: the worst of the five cells of arm − unrouted at each B_N, and every cell at B_N = 1000). Source: `adaptbp_VIEW__v1.json` (sha 6e1430c2…), a post-hoc descriptive view written after the summary had been read (33.9).

| arm | worst cell of (arm − unrouted), B_N = 100 / 500 / 1000 / 2000 / 5000 | at B_N = 1000: MetaQA H4_SK; MetaQA PHG; MuSiQue; SQuAD H4_SK; SQuAD PHG |
|---|---|---|
| OWN·SUM·NEFF | −0.102 / −0.187 / −0.218 / −0.246 / −0.288 | +0.064; +0.082; −0.218; −0.028; −0.029 |
| OWN·MAX·NEFF | −0.035 / −0.094 / −0.118 / −0.143 / −0.180 | +0.064; +0.073; −0.118; −0.013; −0.011 |
| OWN·TOP2·NEFF | −0.035 / −0.095 / −0.115 / −0.139 / −0.174 | +0.063; +0.079; −0.115; −0.013; −0.011 |
| OWN·TOP3·NEFF | −0.042 / −0.104 / −0.124 / −0.149 / −0.183 | +0.064; +0.077; −0.124; −0.013; −0.011 |
| OWN·TOP5·NEFF | −0.052 / −0.116 / −0.139 / −0.165 / −0.200 | +0.064; +0.073; −0.139; −0.014; −0.012 |
| OWN·TOP10·NEFF | −0.072 / −0.142 / −0.166 / −0.192 / −0.231 | +0.063; +0.077; −0.166; −0.017; −0.016 |
| OWN·SUM·EXPH | −0.021 / −0.052 / −0.062 / −0.081 / −0.104 | +0.027; +0.034; −0.062; −0.012; −0.008 |
| OWN·MAX·EXPH | −0.007 / −0.024 / −0.032 / −0.046 / −0.067 | +0.028; +0.034; −0.032; −0.003; −0.003 |
| OWN·TOP3·EXPH | −0.009 / −0.025 / −0.034 / −0.045 / −0.064 | +0.023; +0.031; −0.034; −0.003; −0.004 |
| S·SUM·NEFF | −0.049 / −0.062 / −0.095 / −0.126 / −0.169 | −0.021; +0.005; −0.095; −0.006; −0.005 |
| S·MAX·NEFF | −0.001 / −0.017 / −0.034 / −0.054 / −0.088 | +0.043; +0.054; −0.034; −0.004; −0.003 |
| S·TOP3·NEFF | −0.009 / −0.026 / −0.043 / −0.068 / −0.104 | +0.030; +0.039; −0.043; −0.004; −0.003 |
| S·SUM·EXPH | −0.001 / −0.002 / −0.012 / −0.026 / −0.051 | +0.027; +0.037; −0.012; −0.002; −0.001 |
| S·MAX·EXPH | +0.000 / +0.000 / −0.001 / −0.005 / −0.021 | +0.020; +0.029; −0.001; −0.001; +0.001 |
| S·TOP3·EXPH | +0.000 / +0.000 / −0.003 / −0.010 / −0.026 | +0.020; +0.029; −0.003; −0.001; +0.001 |

- **From B_N = 1000, no arm is at or above unrouted in every cell.**
  - The closest are S·MAX·EXPH (−0.001 at 1000, −0.021 at 5000) and S·TOP3·EXPH (−0.003, −0.026).
  - At 1000 they keep +0.020 / +0.029 on the KB, against +0.064 / +0.077 for OWN·TOP3·NEFF.
- **Every OWN·*·NEFF arm holds the KB gain (+0.063 to +0.082 at 1000), and its worst cell is MuSiQue** (−0.115 to −0.218).
- **Against the envelope** (the best fixed point of any ranking, per cell and B_N; gold-chosen hindsight). The worst-cell distance at B_N = 1000 is:
  - −0.041 for S·MAX·NEFF (MetaQA PHG);
  - −0.051 for S·TOP2·NEFF, −0.056 for S·TOP3·NEFF and −0.058 for S·SUM·EXPH;
  - −0.061 for OWN·MAX·EXPH and −0.064 for OWN·TOP3·EXPH;
  - −0.066 for S·MAX·EXPH and S·TOP3·EXPH (MetaQA PHG);
  - −0.120 for OWN·MAX·NEFF, −0.126 for OWN·TOP3·NEFF and −0.220 for OWN·SUM·NEFF (MuSiQue).
  - The closest arm changes with B_N:
    - at 100, OWN·MAX·NEFF and OWN·TOP2·NEFF (−0.038);
    - at 500 and 1000, S·MAX·NEFF (−0.057 at 500);
    - at 2000, S·SUM·EXPH (−0.042);
    - at 5000, S·MAX·EXPH (−0.029).
- **The aggregation**, paired at B_N = 1000 (cells in T5's order):
  - MAX → TOP3 (NEFF): +14/−14; +16/−8; +34/−45 (p 0.26); +3/−3; +1/−2. No cell has p < 0.1 at B_N = 100, 1000 or 5000.
  - SUM → TOP3 (NEFF): +21/−21; +11/−22 (0.08); +205/−17 (4e−42); +30/−1 (3e−8); +37/−1 (3e−10).
  - SUM → MAX (NEFF): +29/−29; +17/−36 (0.01); +229/−30; +31/−2; +39/−2.
  - MAX → TOP3 under EXPH: +2/−13 (p 0.01) on MetaQA H4_SK; p ≥ 0.18 in the other cells.
  - The TOPk band, TOP3 → TOP2 / TOP5 / TOP10:
    - MuSiQue +29/−11 (p 0.01) / +8/−39 (6e−6) / +16/−100 (5e−16);
    - SQuAD, TOP10 only: +2/−10 and +1/−11 (p 0.04, 0.01);
    - the KB: p ≥ 0.14 for every k.
- **The count**, NEFF → EXPH at 1000:
  - TOP3: +22/−104; +22/−113; +197/−17; +21/−0; +14/−0.
  - MAX: +24/−95; +24/−101; +189/−17; +20/−0; +16/−0.
  - For TOP3, EXPH gives up 0.041 / 0.046 on MetaQA and gains 0.090 on MuSiQue and 0.0105 / 0.007 on SQuAD.
- **The ranking**, OWN → S at 1000:
  - TOP3·NEFF: +5/−73; +7/−82; +192/−30; +19/−0; +18/−1.
  - TOP3·EXPH: +6/−12; +7/−11; +73/−12; +4/−1; +9/−0.
  - §32's split persists under both counts. S is ahead on text. The L ranking is ahead on the KB under NEFF; under EXPH the KB difference is within noise (p ≥ 0.24).

**33.6 Surfaces R(B_P, B_N) and the user's points.** This is the fixed grid itself, with no adaptive rule: every ranking at every integer B_P and every B_N. The per-query intervals are in the npz, so any point of R(B_P, B_N) can be read without recomputation. The user's points, for S and the three primary aggregations:

| ALL gold at (B_P, B_N) = | (5, 500) | (20, 500) | (20, 2000) | (100, 1000) | (unrestricted, 1000) |
|---|---|---|---|---|---|
| MetaQA H4_SK · S | 0.109 | 0.358 | 0.404 | 0.536 | 0.508 |
| MetaQA H4_SK · SUM | 0.465 | 0.541 | 0.607 | 0.536 | 0.508 |
| MetaQA H4_SK · MAX | 0.413 | 0.551 | 0.605 | 0.548 | 0.508 |
| MetaQA H4_SK · TOP3 | 0.441 | 0.548 | 0.604 | 0.542 | 0.508 |
| MetaQA PHG · S | 0.105 | 0.328 | 0.369 | 0.551 | 0.508 |
| MetaQA PHG · SUM | 0.478 | 0.557 | 0.638 | 0.548 | 0.508 |
| MetaQA PHG · MAX | 0.378 | 0.561 | 0.622 | 0.556 | 0.508 |
| MetaQA PHG · TOP3 | 0.431 | 0.558 | 0.628 | 0.554 | 0.508 |
| MuSiQue · S | 0.269 | 0.638 | 0.646 | 0.819 | 0.840 |
| MuSiQue · SUM | 0.183 | 0.472 | 0.482 | 0.752 | 0.840 |
| MuSiQue · MAX | 0.249 | 0.510 | 0.515 | 0.745 | 0.840 |
| MuSiQue · TOP3 | 0.268 | 0.542 | 0.549 | 0.759 | 0.840 |
| SQuAD H4_SK · S | 0.884 | 0.987 | 0.987 | 0.999 | 1.000 |
| SQuAD H4_SK · SUM | 0.793 | 0.951 | 0.951 | 0.998 | 1.000 |
| SQuAD H4_SK · MAX | 0.867 | 0.958 | 0.958 | 0.997 | 1.000 |
| SQuAD H4_SK · TOP3 | 0.879 | 0.960 | 0.960 | 0.997 | 1.000 |
| SQuAD PHG · S | 0.857 | 0.986 | 0.986 | 1.000 | 1.000 |
| SQuAD PHG · SUM | 0.742 | 0.947 | 0.947 | 0.996 | 1.000 |
| SQuAD PHG · MAX | 0.837 | 0.953 | 0.953 | 0.996 | 1.000 |
| SQuAD PHG · TOP3 | 0.849 | 0.955 | 0.955 | 0.997 | 1.000 |

The B_N = 1000 slice at B_P = 10 / 20 / 50 / 100, for all 7 rankings, with each ranking's best fixed B_P and the smallest B_P within 0.01 of unrouted (S and TOP3). SQuAD's 1.000 at this B_N is 0.9995, as for unrouted.

| ranking | MetaQA H4_SK | MetaQA PHG | MuSiQue | SQuAD H4_SK | SQuAD PHG |
|---|---|---|---|---|---|
| S | 0.158 / 0.383 / 0.556 / 0.536 | 0.147 / 0.347 / 0.571 / 0.551 | 0.448 / 0.645 / 0.779 / 0.819 | 0.958 / 0.987 / 0.997 / 0.999 | 0.953 / 0.986 / 0.997 / 1.000 |
| SUM | 0.543 / 0.577 / 0.565 / 0.536 | 0.566 / 0.597 / 0.582 / 0.548 | 0.335 / 0.479 / 0.665 / 0.752 | 0.898 / 0.951 / 0.983 / 0.998 | 0.868 / 0.947 / 0.981 / 0.996 |
| MAX | 0.554 / 0.578 / 0.573 / 0.548 | 0.547 / 0.586 / 0.588 / 0.556 | 0.387 / 0.514 / 0.666 / 0.745 | 0.933 / 0.958 / 0.984 / 0.997 | 0.917 / 0.953 / 0.983 / 0.996 |
| TOP2 | 0.553 / 0.577 / 0.572 / 0.543 | 0.552 / 0.588 / 0.586 / 0.555 | 0.411 / 0.543 / 0.682 / 0.752 | 0.936 / 0.960 / 0.986 / 0.998 | 0.919 / 0.955 / 0.986 / 0.997 |
| TOP3 | 0.552 / 0.576 / 0.568 / 0.542 | 0.555 / 0.590 / 0.587 / 0.554 | 0.417 / 0.547 / 0.689 / 0.759 | 0.937 / 0.960 / 0.987 / 0.997 | 0.923 / 0.955 / 0.987 / 0.997 |
| TOP5 | 0.549 / 0.575 / 0.568 / 0.539 | 0.557 / 0.594 / 0.586 / 0.551 | 0.410 / 0.543 / 0.688 / 0.760 | 0.935 / 0.961 / 0.987 / 0.998 | 0.918 / 0.954 / 0.987 / 0.996 |
| TOP10 | 0.547 / 0.575 / 0.566 / 0.536 | 0.559 / 0.593 / 0.583 / 0.548 | 0.379 / 0.525 / 0.687 / 0.760 | 0.925 / 0.962 / 0.984 / 0.998 | 0.906 / 0.953 / 0.986 / 0.996 |
| best fixed B_P: S; TOP3 (ALL @ B_P) | 0.559 @ 44; 0.582 @ 26 | 0.573 @ 52; 0.600 @ 27 | 0.842 @ 273; 0.840 @ 764 | 1.000 @ 127; 1.000 @ 143 | 1.000 @ 100; 1.000 @ 178 |
| smallest B_P within 0.01 of unrouted: S; TOP3 | 28; 8 | 30; 7 | 167; 326 | 21; 59 | 24; 56 |

- **On the KB, recall is not monotone in B_P.**
  - At B_N = 1000 the L rankings peak at B_P 22–29 (SUM 24 / 22, MAX 27 / 29, TOP3 26 / 27). They then fall towards unrouted (0.508, all 432 partitions).
  - Past the peak, each added partition adds nodes that push co-located golds beyond B_N.
  - S rises until B_P 44 / 52 (0.559 / 0.573), below every L ranking's peak.
- **On text, recall rises with B_P towards the region where routing is lossless.**
  - MuSiQue S is 0.448 / 0.645 / 0.779 / 0.819 at B_P 10 / 20 / 50 / 100 and peaks at 0.842 at 273. At each of these four B_P, the L rankings are 0.03–0.17 below S.
  - SQuAD S is 0.958 / 0.987 / 0.997 at 10 / 20 / 50 (PHG 0.953 / 0.986 / 0.997). The L rankings are 0.868–0.937 at 10 and 0.981–0.987 at 50.
- **At B_P = 20 on text, the partitions contacted bound recall, not B_N.** B_P = 20 contacts about 2,000 nodes. Raising B_N from 500 to 2000 adds 0.0055–0.0095 on MuSiQue and nothing on SQuAD, against 0.042–0.082 on MetaQA (T6).
- **At (100, 1000)**, the L rankings give:
  - 0.536–0.556 on MetaQA, above unrestricted (0.508) and below their peak near B_P 25 (0.580–0.603);
  - 0.745–0.759 on MuSiQue, against S 0.819 and unrestricted 0.840.
- **At (5, 500)**, SUM gives the most on MetaQA (0.465 / 0.478), and S the most on text (MuSiQue 0.269; SQuAD 0.884 / 0.857).

**33.7 Caps and the gold diagnostic.** A cap B_P(q) = min(B_P^max, ⌈N_eff⌉) is a deployment table only (33.1). OWN·TOP3·NEFF at B_N = 1000, with the mean capped fan-out in brackets:

| OWN·TOP3·NEFF, ALL at B_N = 1000 [mean fan-out] | uncapped | B_P^max = 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|---|
| MetaQA H4_SK | 0.572 [49.5] | 0.443 [5.0] | 0.552 [10.0] | 0.576 [20.0] | 0.576 [43.4] | 0.572 [49.4] |
| MetaQA PHG | 0.585 [53.3] | 0.433 [5.0] | 0.555 [10.0] | 0.590 [20.0] | 0.592 [45.0] | 0.585 [53.2] |
| MuSiQue | 0.716 [84.0] | 0.268 [5.0] | 0.417 [10.0] | 0.545 [19.9] | 0.667 [45.2] | 0.704 [67.8] |
| SQuAD H4_SK | 0.987 [38.5] | 0.879 [5.0] | 0.937 [10.0] | 0.960 [19.2] | 0.985 [34.0] | 0.987 [38.2] |
| SQuAD PHG | 0.989 [41.7] | 0.849 [5.0] | 0.923 [10.0] | 0.955 [19.5] | 0.985 [35.9] | 0.989 [41.2] |

- A cap of 100 changes only MuSiQue (−0.0125).
- A cap of 50:
  - raises MetaQA by 0.004 / 0.0075, because it removes over-contact;
  - lowers SQuAD by 0.0015 / 0.0035;
  - lowers MuSiQue by 0.0495.
- A cap of 20 costs 0.0265 / 0.034 on SQuAD and 0.1715 on MuSiQue; on MetaQA it is +0.004 / +0.005.
- A cap of 10 costs every cell: 0.020 / 0.030 (MetaQA), 0.299 (MuSiQue), 0.0495 / 0.066 (SQuAD).

**Gold diagnostic** (never a rule; cells in T5's order):
- LO, the fan-out that contacts every gold partition, under the TOP3 ranking:
  - median 8 / 7.5 / 16 / 1 / 1;
  - mean 113 / 102 / 81 / 4.2 / 4.6;
  - p90 398 / 394 / 241 / 7 / 8.
  - Under S the median is 25 / 27 / 12 / 1 / 1.
- The share of queries whose B_P(q) contacts every gold partition:
  - TOP3·NEFF 0.6391 / 0.6782 / 0.7485 / 0.9865 / 0.9885;
  - TOP3·EXPH 0.6862 / 0.7187 / 0.8865 / 0.9970 / 0.9955.
- Over-contacted at B_N = 1000 (servable at a smaller fan-out, lost at B_P(q)):
  - TOP3·NEFF 0.0596 / 0.0771 / 0.0140 / 0 / 0;
  - TOP3·EXPH 0.1156 / 0.1356 / 0.0265 / 0 / 0.
- Servable at some fixed B_P, a per-query oracle, at B_N = 1000:
  - TOP3 ranking 0.6617 / 0.6842 / 0.8755 / 0.9995 / 1.0000, against its best single fixed B_P 0.5821 / 0.5996 / 0.8395;
  - S 0.6351 / 0.6532 / 0.8800, against 0.5591 / 0.5731 / 0.8415.
  - The per-query headroom is 0.036–0.085 on MetaQA and MuSiQue.
- The concentration barely predicts LO.
  - Spearman ρ(N_eff, LO) is 0.09–0.28 over the cells and the three primary aggregations (TOP3 0.175 / 0.164 / 0.243 / 0.093 / 0.115).
  - ρ(exp H, LO) is −0.04 to 0.29.

**33.8 Reading** (development numbers: descriptive; selection is the user's step).
- **The question** (33.1): does ⌈N_eff⌉ *"land near the good part of the routing surface on MetaQA, MuSiQue and SQuAD simultaneously"*?
  - **MetaQA: near.**
    - From B_N = 1000, the SUM, MAX and TOP3 NEFF arms are within 0.018 of their ranking's best fixed B_P, and within 0.039 at B_N 100 and 500.
    - They are +0.06–0.08 over unrouted at B_N 500–1000.
  - **SQuAD: the count lands; the L ranking costs 0.011–0.013.**
    - B_P(q) contacts every gold partition for 0.99 of queries.
    - With the S ranking, the same count is within 0.004 of unrouted.
  - **MuSiQue: not near.**
    - N_eff's mean (84–97) is below what either ranking needs at B_N = 1000: S needs 167 to come within 0.01 of unrouted, TOP3 needs 326.
    - From B_N = 500, OWN·TOP3·NEFF and OWN·MAX·NEFF lose 0.094–0.183.
  - **No arm is near the good part in all five cells at once.**
    - The measured arms span a frontier. At one end, OWN·TOP3·NEFF and OWN·MAX·NEFF keep the KB gain (+0.06–0.08) and lose 0.12 on MuSiQue. At the other, S·MAX·EXPH and S·TOP3·EXPH are within 0.003 of unrouted on text up to B_N = 1000 and keep +0.02–0.03 on the KB.
    - In its worst cell, the arm closest to the gold-chosen envelope is S·MAX·NEFF at B_N = 1000 (−0.041).
- **Adaptivity against the same mean held fixed.**
  - It ties on the KB (−0.010 to +0.005).
  - It is ahead on SQuAD (+0.003 to +0.009; p ≤ 0.01 for TOP3 and MAX at 1000).
  - It is behind on MuSiQue under NEFF (−0.006 to −0.034).
  - Per query there is headroom (the oracle is 0.036–0.085 above the best fixed B_P on MetaQA and MuSiQue). The concentration does not find it: ρ(N_eff, LO) ≤ 0.28.
  - N_eff measures how the one-hop evidence mass spreads over partitions. How far the golds spread is only weakly related to it.
- **One multiplier does not fit both regimes.** exp H is 1.7–3.0× N_eff.
  - The larger count serves MuSiQue (−0.034 against −0.124 at 1000).
  - It costs MetaQA 0.041–0.046, as over-contact rises from 0.06–0.08 to 0.12–0.14 of queries.
  - The smaller count does the reverse.
- **The aggregation** (the user's step: *"choose the most universal one"*).
  - SUM is the least universal. Its worst cell is −0.218 at B_N = 1000 under NEFF. TOP3 over SUM is +205/−17 on MuSiQue and +30/−1, +37/−1 on SQuAD, and tied on the KB.
  - TOP3 and MAX tie as rules in every cell under NEFF: no paired difference has p < 0.1 at B_N 100 / 1000 / 5000. Under EXPH, MAX is ahead on MetaQA H4_SK (+2/−13).
  - In the TOPk band, smaller k is better on text (TOP10 −0.042 against TOP3 on MuSiQue at 1000), and k does not matter on the KB. TOP2 sits between TOP3 and MAX.
  - On MuSiQue the arms differ mainly through their counts. As fixed-B_P rankings, SUM, MAX and TOP3 are within 0.025 of each other at B_P 50–100 (T7), while their mean counts are 48 / 97 / 84.
- **The ranking.** §32's split persists under both counts. S is ahead on text (MuSiQue, OWN → S +192/−30 for TOP3·NEFF), the L ranking on the KB (+5/−73, +7/−82).
- **For the user** (nothing is selected, and nothing is frozen):
  - (a) The aggregation. TOP3 (the stated starting candidate) and MAX are tied as rules here. SUM is below both on text and not above them on the KB.
  - (b) The count.
    - ⌈N_eff⌉ (the stated preference) keeps the KB gain and costs MuSiQue 0.12 at B_N = 1000 with the L ranking (TOP3); with S it costs 0.043 and keeps +0.030 / +0.039.
    - ⌈exp H⌉ costs MuSiQue 0.034 with the L ranking and 0.003 with S, and keeps +0.02–0.03 of the KB gain.
    - A cap B_P^max = 100 changes only MuSiQue (−0.0125).
  - (c) The ranking inside the routing: the aggregation's own order (L), or first appearance in O (S). This is §32.7's open split, now under adaptive counts.
  - Each option is one dataset-independent rule. None of the 24 measured here comes within 0.04 of the gold-chosen envelope in its worst cell at B_N = 1000.

**33.9 Cost and hygiene.**
- **Wall-clock and memory:** MetaQA 245 s, peak RSS 768 MB; MuSiQue 808 s, 1,197 MB; SQuAD 150 s, 497 MB. The runs went one at a time, 03:55–04:16 local.
- **Per-query latency** (MetaQA / MuSiQue / SQuAD):
  - FLAT: products 10.1 / 77.0 / 9.4 ms (amortized) + RRF 44 / 156 / 19 ms;
  - node score + LOC order: 2.5 / 8.0 / 4.9 ms;
  - fused order: 15.5 / 61.1 / 7.2 ms;
  - selection, all 24 arms of a cell (6 aggregations, 7 rankings, 12 counts): 3.7–3.9 / 12.0 / 5.1–5.2 ms;
  - evaluation only (every fan-out of the 7 rankings): 7.3–7.8 / 8.0 / 4.5 ms.
- **Host at the starts:** 0.85–1.24 GiB available, 10.5–12.1 GiB of swap in use. Foreign pid 12096 (7.6–8.5 GB RSS) ran throughout, untouched.
- **Population:** the development population c7f70806… (§31). Nothing held out was read.
- **Nothing under `data/` was written. No pinned module or CONTRACT_FILE was edited.**
  - The records pin `_l1d_adaptbp.py` (984e3aef…), `_l1d_arms.py` (f5f52b79…) and `_l1d_lib.py` (dafe39b2…). Through `structures.imports` they also pin `_l1d_node1h.py` (8265d6bc…) and `_l1d_edgediag.py` (874bfc49…).
  - They assert identity with the node1h records (`node1h_<ds>__v1.{json,npz}`, shas recorded).
- **The summary module** `_l1d_adaptbp_summary.py` (14e12a2b…) is new.
  - It was written at 03:58 local, before any v1 record had been read and before the MetaQA and MuSiQue runs had finished. It was not modified afterwards.
  - It was checked on the smoke outputs (outside the repository), which are not used.
- **The view module** `_l1d_adaptbp_view.py` (1deb7fd5…) is new and post hoc.
  - Its contents were chosen after the summary had been read: every arm against unrouted, the worst cell, the surfaces of all 7 rankings, and paired comparisons between arms.
  - It writes `adaptbp_VIEW__v1.json` (6e1430c2…). It asserts the three npz shas and the harness sha, and asserts each arm's ALL equal to its record.
- **Rounding.** The JSON records round shares to 4 decimals and fan-out means to 2. Printing a stored MetaQA share (a count over 1,998 rows) with 3 decimals can round to the wrong side, and at 2,000 rows (MuSiQue, SQuAD) a 4th-decimal 5 is an exact half.
  - The tables and numbers of 33.3–33.8 are therefore recomputed from the npz (per-query LO, HI(B_N) and B_P(q)) with the harness's own `surface` and `served_at`, and every value is checked against its record (4,165 values).
  - Per-hop shares are rebuilt from the records' per-hop n.
  - Shares are shown to 3 decimals and means to 1, with halves rounded away from zero.
- **The envelope distances in 33.5 are in no record as a list.**
  - They are the summary's envelope (`best_fixed_any_ranking`) minus the view's arm values, from the same exact counts.
  - For the 12 arms of the summary's `universality` table, they equal its `delta_vs_best_fixed_any_ranking` to its 4 decimals.

## 34. ROUTE: non-parametric partition routing (L1b) in three rounds (2026-09-27 local) (`_l1d_route.py`, `_l1d_route2.py`, `_l1d_route3.py`, summary `_l1d_route_summary.py`, post-hoc view `_l1d_route_view.py`, on the shared runner `_l1d_arms.py`; records `results/L1_DEV/route{,2,3}_{metaqa,musique,squad}__v1.{json,npz}`, `route_SUMMARY__v1.{json,npz}`, `route_VIEW__v1.json`; STATUS = DEVELOPMENT: descriptive numbers, no pre-registration, no verdicts, no adoption)

**34.1 The request (2026-09-27 local).** After §33 the user sent two messages. Both are taken as the user's.
- **A proposal, pasted without comment** (05:54 local). Its operative sentences, verbatim. The pasted text shows each formula twice, rendered and as LaTeX source; formulas are shown once here, and the numbered test list is joined into one line.
  - *"Yes — there are still several good non-learning / parameter-free directions, and given your original L1 contract, I would exhaust those before introducing XGBoost."*
  - The core idea: *"select partitions greedily by marginal uncovered node evidence"*, P_t = argmax_P Σ_{u∈C_P∖U_{t−1}} L(u), with C_P the partition's nodes with IR_L1 evidence and U_{t−1} the evidence already covered.
  - *"An even stronger version operates on semantic-source coverage, not merely nodes"*: gain(P) = Σ_{s∈S(P)∖S_covered} w(s), with S(P) the FLAT seeds that contribute evidence to P.
  - The stop: *"we let the evidence itself tell us when additional partitions stop adding genuinely new information."* The candidates are G_{t+1} < (1/t)Σ_{i≤t} G_i, a covered fraction ≥ τ (*"technically τ is a hyperparameter, but there are no learned parameters"*), or a knee of the cumulative evidence C(k) = Σ_{i≤k} G_i / Σ_i G_i: *"Pick the point of maximum curvature."*
  - MMR: score(P) = relevance(P) − λ·max_{P'∈S} similarity(P, P'). *"We can even avoid a tuned λ by normalizing both terms and setting them equally, or derive λ from evidence concentration."*
  - One-hop partition diffusion: r = r_0 + α A_P^⊤ r_0, with r_0(P) = Σ_{u∈P} L(u) and the cut-edge adjacency normalized by deg(P'). *"I would still classify it as L1 rather than L3, provided we don't iterate it."*
  - Facility location: F(S) = Σ_u L(u)·max_{P∈S} s(u, P).
  - The test list: *"1. current MAX/TOP3 baseline; 2. marginal evidence coverage; 3. seed-diversity coverage; 4. facility-location / submodular routing; 5. optionally one-hop partition-graph diffusion before 2–4."*, evaluated *"on exactly the same"* R(B_P, B_N) *"surface"*.
  - *"The key test is whether a method can reproduce both behaviors automatically"*: MetaQA → few shards, MuSiQue → many diverse shards.
  - *"My strongest candidate right now would actually be greedy marginal evidence coverage with seed diversity"*; *"So no — I would not conclude that we need XGBoost yet."*
- **The request** (05:58 local), verbatim: *"exhaustively try out all these different ways of non parametric L1 where we are only dealing with partitions the ranking or the flat features etc we have discovered using those we can influnce and bring those partitions then we will have a partition centric system too, why i want to have partitions is if we can say scale this to thousands of nodes and distribute the graph into multiple systems using partitions and identify only the right ones to bring that is what L1 should be, avoid a model or learned method as much as possible and test out everything we can based on the suggestions and based on the failures we have encountered or will encouter and try to mitigate them"*
- **The bounds kept** (the standing rulings of §31–§33):
  - IR_L1 is the L1a node score, unchanged, and the partitions are routing shards (L1b). L1 is one static A^⊤x, so a second hop (A^⊤(A^⊤x), H2, PPR) belongs to L3.
  - One rule for every dataset: no dataset branch, no target-degree term, no learned or tuned constant, no gold in any rule. B_P^max is a deployment constraint only.
  - Development → the user's selection → ONE final held-out evaluation. Nothing held out was read.
- **Three rounds, each designed after reading the previous round's records:**
  - round 1 (`_l1d_route.py`): the proposal's five items, the flat (semantic) features as shard rankings, and fusions of the two;
  - round 2 (`_l1d_route2.py`): six failures measured in round 1, each with a mitigation;
  - round 3 (`_l1d_route3.py`): the failure left after round 2 (one rule has to serve two regimes), a count that looks for a natural break, and three failures a distributed deployment will meet (router state, coupling to B_N, load), measured.

**34.2 Design.**
- **Fixed** (each asserted per query; the identities are below):
  - the node score IR_L1 of §32–§33: L(u) = Σ_{s∈H_q} w(s) Σ_f A_f(s, u) g_f(s), with H_q = FLAT[:200], w(s) = 1/rank(s), g_f(s) = 1/log2(1 + deg_f^+(s)) and E = STRUCT_out ∪ STRUCT_in ∪ KNN ∪ NER;
  - the served list: the first B_N nodes of the IR_L1 FLAT+LOC order O (RRF, K0 = 60) inside the contacted shards;
  - the cells, their frozen single-owner partitions (the shards) and the population, as in §33: MetaQA H4_SK and PHG (432 shards, 1,998 rows), MuSiQue (1,175 shards, 2,000 rows), SQuAD H4_SK and PHG (202 shards, 2,000 rows); population c7f70806….
- **Varied: only the router.** A router is a ranking of the shards (the order in which they are contacted) and a count B_P(q) (how many are contacted). Each is a fixed function of the query's own FLAT / IR_L1 evidence and the static shard map.
  - The only constants are inherited from §32–§33: K0 = 60; the depth 200 (H_q = FLAT[:200]; O[:200], LOC[:200] and O'[:200] for the spread counts and λ2 / λ3); k = 3 (TOP3).
  - The shared library also records DEG_CAP = 300 (the frozen hub cap of the expansion contract), which no §34 module uses. CQ = 200 and BLOCK = 40,000 are batch sizes; N_AGREE = 100 and AGREE_MIN = 0.90 set the served-list check.
  - The greedies use a relative tie band of 1e−9 and a zero-gain threshold of 1e−12 (numerical tolerances).
- **Notation** (as the modules write it):
  - Cm[s, P] = the IR_L1 mass seed s puts into shard P (its one-hop entries in P, each w(s)·g_f(s)); m_s = Σ_P Cm[s, P]; P(s) = the seed's own shard;
  - fv = the FLAT fused score;
  - O' = RRF(H_q, LOC) with K0 = 60, the router's own fused order (34.4, failure 6).
- **The rounds** (records stamped 2026-09-27T01:00–02:58Z, 06:30–08:28 local; one run at a time):

| round | module (sha) | rankings: new + reused | counts: new + reused | native arms | records: json sha (npz sha) |
|---|---|---|---|---|---|
| 1 | `_l1d_route.py` (7b420e54…) | 22 + 0 | 32 + 0 | 36 | metaqa b16df23f… (9abb1d6c…); musique a42cd098… (add25dee…); squad 0db5c8da… (0a7d13a6…) |
| 2 | `_l1d_route2.py` (890566c7…) | 21 + 5 | 45 + 15 | 53 | metaqa 90fe29a6… (60283117…); musique 9c7d5173… (2c3395fc…); squad 4d7e2e74… (9cd66d90…) |
| 3 | `_l1d_route3.py` (283a711b…) | 25 + 6 | 48 + 6 | 75 | metaqa 426f9780… (df3f0b9a…); musique d22b0481… (a730fdb0…); squad 7a458676… (8ffad2c0…) |
| merged | `_l1d_route_summary.py` (6fd17ca9…) | 68 | 125 + PSTAR_BN | 163 distinct (164 slots) | `route_SUMMARY__v1.json` 83e02d15… (npz 3385e06d…) |

- A reused ranking or count is recomputed from the earlier round's definition and asserted equal to that round's record per query. So the three rounds merge into one factorial.
- Reused rankings: round 2 S, FSUM, SUM, TOP3, SDIV; round 3 S, TOP3, SDIV, ES, NSUM, SDIVS.
- Reused counts: round 2 the 15 counts NEFF_x and KNEE_x for x ∈ {FSUM, FSUMFV, SUM, MAX, TOP3, SDIV}, EXPH_TOP3, NSEEDP and NOACT; round 3 KNEE_SDIV, KNEE_TOP3, BPI_SDIV, BPI_TOP3, NOACT_E and NSEEDP.
- **The factorial.** Every ranking × every count × B_N ∈ {100, 250, 500, 1000, 2000, 5000} gives 68 × 126 = 8,568 arms. 8,500 of them are rules; PSTAR_BN is a reference (below).
  - Each record stores, per query and ranking, the interval [LO, HI(B_N)] of fan-outs at which all the query's golds are served (§33.2). Every ranking·count pair is therefore evaluated post hoc from the stored counts, with nothing re-run.
  - An arm is written ranking·count; the records write `RANKING|COUNT`. A union is its own ranking and count (UM_KS_L4·UM_KS_L4).
  - The **native** arms are the pairs a round's design named: 164 slots, 163 distinct (TOP3·EXPH_TOP3 is native in rounds 1 and 2). Every other pair is a cross arm.
  - No rule count depends on B_N.
- **References at each B_N:**
  - unrouted (every shard contacted);
  - each ranking's best fixed B_P (hindsight);
  - the envelope: the best fixed point of any of the 68 rankings (gold-chosen hindsight; not §33's 7-ranking envelope);
  - PSTAR_BN: the distinct shards of O[:B_N], the fan-out at which S reproduces the unrouted list (asserted). It is coupled to B_N and not necessarily the smallest lossless fan-out. It needs O, so it is a reference, not a rule.
- **What a router must read** (the summary's `ranking_information` and the modules' docstrings):
  - a per-node shard sketch (the shards of each node's one-hop neighbours, with multiplicities per family for SUM and NSUM) for the FSUM, SUM, SDIV and NSUM scores and the counts on them;
  - the seeds' one-hop node rows for MAX, TOP3, the greedies, ES and O', the λ shares and everything built on them;
  - a global node order (every node's O or FLAT rank) for S, F, D, SP, FT3, OT3 and the fusions with S or F.
  - FLAT ranks themselves are global in every tier.
  - Ties: round 1 falls back to O (a global order); the new rankings of rounds 2–3 fall back to ES (node rows); SDK falls back to FSUM, then the static shard id (sketch only).
- **Identities asserted** on every gold node (MetaQA 14,489, MuSiQue 4,830, SQuAD 2,000) and in every cell:
  - every round:
    - FLAT, LOC, FLAT+LOC and LOC-order positions equal `loc_<ds>__v1.npz`;
    - IR_L1's LOC and FLAT+LOC positions equal node1h v1;
    - the runner's served-list check passes (dense and SPLADE top-100 overlap 1.0);
    - contacting every shard equals unrouted, per row and ranking;
    - on the first 3 rows of every cell, every ranking is a permutation.
  - round 1:
    - LO / HI of S, SUM, MAX and TOP3, and the counts NEFF_SUM, NEFF_MAX, NEFF_TOP3 and EXPH_TOP3, equal adaptbp v1 (§33);
    - S at PSTAR_BN equals unrouted at every B_N;
    - the per-seed masses sum to SUM, and the greedy gains never increase (every row);
    - on the first 3 rows, the literal greedy node cover equals SUM, and each incremental greedy equals a from-scratch greedy.
  - round 2:
    - the reused rankings, the 15 reused counts and PSTAR equal round 1;
    - the per-seed masses sum to SUM, the NSUM / NSUMS totals are checked, every PC_LS row sums to 1, and the noisy-OR gains never increase (every row);
    - on the first 3 rows, the incremental noisy-OR greedies equal from-scratch ones, and every union prefix equals the union.
  - round 3:
    - the reused rankings, the 6 reused counts and PSTAR equal round 2;
    - PSTAR_BN equals the set of shards whose first O rank is < B_N (every row);
    - on the first 3 rows, the mixtures at λ = 0 / 1 equal their endpoints (ES / SDE or T3E; C / BPI_SDIV), and every union prefix equals the union.
- **Smoke runs** of 200 rows per dataset, outside the repository, checked the identities before each launch. Their numbers are not used, except that the round-3 smoke led to λ4 (34.5, FLAGGED).

**34.3 Round 1: the proposal's items** (`_l1d_route.py`; records `route_<ds>__v1`).
- **Definitions.** A ranking orders the shards by descending score. Ties and zero scores fall back to the first appearance in O (§32–§33).
  - *Item 1, the baseline:* SUM, MAX and TOP3 of L per shard (§33). Alongside them, SDIV = Σ w(s) over the seeds that have localised evidence in the shard (distinct supporting seeds).
  - *Item 2, marginal node-evidence coverage.* On disjoint shards, a shard's marginal gain is its whole SUM, so the greedy order is SUM (the literal greedy is asserted equal). Node-level facility location with s(u, P) = 1[u ∈ P] is modular for the same reason (= SUM) and was not run separately.
  - *Items 3–4, seed diversity and facility location:* the greedy maximum of F(S) = Σ_{s∈H_q} w(s)·max_{P∈S} φ(s, P):
    - SC_L: φ = 1[P holds localised evidence of s] (the proposal's semantic-source coverage);
    - SC_LS: as SC_L, or P holds s itself;
    - FL_L: φ = Cm[s, P]/m_s, the share of s's localised mass in P (facility location over seeds);
    - FL_LS: φ(s, P(s)) = 1, otherwise as FL_L.
    - The greedy takes the largest marginal gain while it is positive (ties within 1e−9 relative → O), then the rest in O order.
  - *Item 5, one-hop partition diffusion:* PD = r0 + A_P^⊤ D^−1 r0.
    - r0 = SUM; A_P[P', P] = the static entries of E from P' to P ≠ P'; D = their row sums; α = 1; applied once.
    - FLAGGED: a second hop at shard granularity (Message A places the node-level A^⊤(A^⊤x) in L3). It was run because the user asked for every suggestion.
  - *The flat features as shard rankings:*
    - S = first appearance in O; F = first appearance in FLAT;
    - D and SP = the dense and SPLADE orders alone;
    - FSUM = Σ w(s) over the shard's seeds; FSUMFV = Σ fv(s); FT3 = the sum of the 3 largest fv in the shard;
    - OT3 = the sum of the 3 largest fused O scores.
  - *Fusions:*
    - PF_S_T3 / PF_F_T3: shard-level RRF of S (or F) with TOP3 (K0 = 60, 0-based ranks);
    - IL_S_T3: round-robin interleave of S and TOP3, S first;
    - U_NEFF / U_KNEE: the union S[:c_F] ∪ TOP3[:c_L], keyed by min(rank_S/c_F, rank_TOP3/c_L), with (c_F, c_L) = (NEFF_FSUMFV, NEFF_TOP3) or (KNEE_FSUMFV, KNEE_TOP3). The union's size is its count.
  - *Counts* (each within [1, number of shards]; as in §33, a query without evidence for a score contacts every shard):
    - NEFF_x = ⌈1/Σp²⌉ over x's positive scores (§33).
    - KNEE_x = the number of positive scores ≥ their mean. This is the Kneedle knee of the query's cumulative evidence curve with both axes normalized: for a non-increasing sequence, the chord maximum of C(k) − k/K is the last k whose gain is ≥ the mean. It is the proposal's *"point of maximum curvature"*.
    - EXPH_TOP3 (§33).
    - NSEEDP / NOACT / NLOCP = the distinct shards of H_q / O[:200] / LOC[:200] (the evidence spread at the frozen depth 200).
    - For each greedy: NEFFG and KNEEG over its marginal gains, and COVER = the number of positive gains.
    - The proposal's mean-gain stop G_{t+1} < (1/t)ΣG_i was not run: on a non-increasing gain sequence it stops at the first strict decrease.

**Round 1 at B_N = 1000**: ALL − unrouted per cell; unrouted ALL is 0.508 / 0.508 / 0.840 / 1.000 / 1.000, SQuAD's being 0.9995. All numbers in 34.3–34.9 come from exact query counts (34.12).

| round-1 native arm | MetaQA H4_SK | MetaQA PHG | MuSiQue | SQuAD H4_SK | SQuAD PHG | mean B_P (five cells) |
|---|---|---|---|---|---|---|
| SUM·NEFF_SUM | +0.064 | +0.082 | −0.218 | −0.028 | −0.029 | 36.1 / 41.2 / 47.9 / 26.7 / 29.2 |
| SUM·KNEE_SUM | +0.040 | +0.055 | −0.063 | −0.020 | −0.020 | 76.0 / 76.8 / 130.9 / 38.5 / 39.7 |
| MAX·NEFF_MAX | +0.064 | +0.073 | −0.118 | −0.013 | −0.011 | 54.9 / 56.6 / 96.6 / 42.7 / 45.9 |
| MAX·KNEE_MAX | +0.049 | +0.057 | −0.057 | −0.014 | −0.012 | 77.6 / 77.3 / 162.0 / 46.8 / 48.5 |
| TOP3·NEFF_TOP3 | +0.064 | +0.077 | −0.124 | −0.013 | −0.011 | 49.5 / 53.3 / 84.0 / 38.5 / 41.7 |
| TOP3·KNEE_TOP3 | +0.042 | +0.056 | −0.051 | −0.013 | −0.011 | 78.1 / 78.1 / 157.6 / 44.6 / 45.9 |
| TOP3·EXPH_TOP3 | +0.023 | +0.031 | −0.034 | −0.003 | −0.004 | 128.0 / 131.8 / 210.2 / 71.9 / 75.6 |
| TOP3·NLOCP | +0.036 | +0.043 | −0.137 | −0.020 | −0.016 | 100.0 / 104.2 / 60.7 / 35.6 / 38.8 |
| SDIV·NEFF_SDIV | +0.051 | +0.061 | −0.074 | −0.009 | −0.007 | 74.2 / 78.2 / 168.9 / 70.4 / 75.3 |
| SDIV·KNEE_SDIV | +0.038 | +0.047 | −0.058 | −0.017 | −0.015 | 92.2 / 91.5 / 198.3 / 58.6 / 60.4 |
| SC_L·SC_L_NEFFG | −0.143 | −0.147 | −0.799 | −0.458 | −0.516 | 3.6 / 3.7 / 2.9 / 2.8 / 2.7 |
| SC_L·SC_L_KNEEG | −0.141 | −0.153 | −0.797 | −0.467 | −0.529 | 4.0 / 4.1 / 2.7 / 2.8 / 2.7 |
| SC_L·SC_L_COVER | −0.116 | −0.097 | −0.709 | −0.277 | −0.328 | 29.5 / 28.7 / 17.4 / 19.5 / 18.7 |
| SC_LS·SC_LS_NEFFG | −0.143 | −0.145 | −0.799 | −0.459 | −0.517 | 3.6 / 3.7 / 2.9 / 2.8 / 2.7 |
| SC_LS·SC_LS_KNEEG | −0.141 | −0.150 | −0.797 | −0.467 | −0.530 | 4.0 / 4.1 / 2.7 / 2.8 / 2.7 |
| SC_LS·SC_LS_COVER | −0.116 | −0.099 | −0.706 | −0.276 | −0.327 | 29.4 / 28.5 / 17.3 / 19.5 / 18.7 |
| FL_L·FL_L_NEFFG | −0.084 | −0.091 | −0.703 | −0.319 | −0.346 | 7.0 / 6.6 / 5.8 / 5.2 / 5.2 |
| FL_L·FL_L_KNEEG | −0.083 | −0.081 | −0.664 | −0.278 | −0.299 | 14.0 / 13.6 / 9.5 / 8.0 / 7.8 |
| FL_L·FL_L_COVER | −0.102 | −0.085 | −0.401 | −0.074 | −0.091 | 106.5 / 102.5 / 69.0 / 56.4 / 55.7 |
| FL_LS·FL_LS_NEFFG | −0.081 | −0.086 | −0.372 | −0.034 | −0.035 | 12.3 / 13.7 / 10.1 / 6.4 / 6.8 |
| FL_LS·FL_LS_KNEEG | −0.073 | −0.076 | −0.300 | −0.023 | −0.025 | 20.1 / 21.9 / 14.8 / 9.1 / 9.8 |
| FL_LS·FL_LS_COVER | −0.094 | −0.084 | −0.066 | −0.001 | +0.001 | 122.4 / 128.6 / 91.4 / 60.4 / 63.5 |
| PD·NEFF_PD | +0.057 | +0.062 | −0.151 | −0.017 | −0.015 | 59.0 / 66.9 / 82.2 / 49.2 / 51.9 |
| PD·KNEE_PD | +0.041 | +0.052 | −0.046 | −0.018 | −0.019 | 81.6 / 82.6 / 190.1 / 48.3 / 49.0 |
| FSUM·NEFF_FSUM | −0.371 | −0.365 | −0.353 | −0.029 | −0.028 | 13.6 / 15.4 / 11.0 / 7.0 / 7.5 |
| FSUM·KNEE_FSUM | −0.363 | −0.354 | −0.291 | −0.021 | −0.021 | 21.3 / 22.9 / 15.3 / 9.4 / 10.1 |
| FSUMFV·NEFF_FSUMFV | −0.337 | −0.320 | −0.223 | −0.023 | −0.029 | 61.6 / 72.7 / 35.3 / 17.7 / 19.8 |
| FSUMFV·KNEE_FSUMFV | −0.348 | −0.345 | −0.277 | −0.038 | −0.042 | 35.6 / 38.7 / 23.6 / 14.5 / 15.9 |
| S·NSEEDP | +0.021 | +0.031 | −0.025 | −0.002 | −0.001 | 121.5 / 127.7 / 90.2 / 60.1 / 63.2 |
| F·NSEEDP | −0.295 | −0.280 | −0.067 | −0.001 | +0.001 | 121.5 / 127.7 / 90.2 / 60.1 / 63.2 |
| S·NOACT | +0.025 | +0.035 | −0.035 | −0.003 | −0.001 | 103.7 / 109.8 / 73.1 / 45.6 / 48.5 |
| S·NLOCP | +0.028 | +0.042 | −0.049 | −0.005 | −0.005 | 100.0 / 104.2 / 60.7 / 35.6 / 38.8 |
| S·NEFF_FSUMFV | +0.027 | +0.046 | −0.141 | −0.015 | −0.015 | 61.6 / 72.7 / 35.3 / 17.7 / 19.8 |
| S·KNEE_FSUMFV | +0.018 | +0.037 | −0.185 | −0.019 | −0.020 | 35.6 / 38.7 / 23.6 / 14.5 / 15.9 |
| U_NEFF·U_NEFF | +0.044 | +0.052 | −0.077 | −0.008 | −0.005 | 73.4 / 83.7 / 91.9 / 41.4 / 45.0 |
| U_KNEE·U_KNEE | +0.041 | +0.054 | −0.043 | −0.009 | −0.007 | 80.6 / 82.1 / 158.2 / 45.8 / 47.2 |

- **The coverage greedies saturate.**
  - A seed counts as covered by any shard that holds one of its one-hop entries, so a few shards cover most seeds at once.
  - The gain-based counts are 2.7–4.1 shards for SC_L / SC_LS and 5.2–14.0 for FL_L. Every positive gain is used up after 17.3–29.5 shards (SC COVER).
  - Every native SC and FL_L arm is below unrouted in every cell (MuSiQue −0.401 to −0.799).
  - FL_LS, which also credits each seed's own shard, is the least affected: FL_LS·FL_LS_COVER is within 0.001 on SQuAD, −0.066 on MuSiQue and −0.094 / −0.084 on MetaQA.
- **PD behaves like SUM and TOP3** at the same count types (NEFF: +0.057 / +0.062 / −0.151; KNEE: +0.041 / +0.052 / −0.046). As a fixed-B_P ranking it sets part of the MetaQA envelope (34.6).
- **The flat orders do not find the KB answer shards.** F, D, SP and FT3 never exceed unrouted on MetaQA, at any B_N. At B_N 1000 they come within 0.01 of it only at 412–432 of the 432 shards. FSUM and FSUMFV, as fixed-B_P rankings, reach +0.015 / +0.021 at 164 / 171 shards; with their own counts they lose 0.320–0.371 on MetaQA and 0.223–0.353 on MuSiQue.
- **S with a spread count comes closest to lossless on text.**
  - S·NSEEDP is −0.025 and S·NOACT −0.035 on MuSiQue, both within 0.003 on SQuAD, at 46–128 shards.
  - They keep +0.021 / +0.031 and +0.025 / +0.035 on the KB.
  - With F's order, the same NSEEDP count loses 0.295 / 0.280 on MetaQA.
- **KNEE against NEFF** (the localised scores SUM, MAX, TOP3 and SDIV).
  - The KNEE counts are larger, most on MuSiQue (TOP3 157.6 against 84.0); only SDIV's is smaller on SQuAD (58.6 / 60.4 against 70.4 / 75.3). On FSUMFV the KNEE count is the smaller one in every cell.
  - They cut the MuSiQue loss: TOP3 −0.051 against −0.124; SUM −0.063 against −0.218; MAX −0.057 against −0.118; SDIV −0.058 against −0.074.
  - They give up 0.013–0.028 of the KB gain.

**34.4 Round 2: round 1's failures and their mitigations** (`_l1d_route2.py`; records `route2_<ds>__v1`). The failures are quoted from the module's docstring. Every new ranking falls back to ES (not O) for ties and zero scores.
1. **Coverage saturation:** *"a seed counts as covered by ANY partition holding one of its 1-hop entries, so hub partitions cover most seeds at once and the greedy stops after 3-30 partitions"*.
   - NSUM(P) = Σ_s w(s)·Cm[s, P]/m_s, additive facility location. Every seed spreads exactly w(s) over its shards, so the score is modular and never saturates.
   - PC_L / PC_LS: the greedy maximum of the noisy-OR coverage Σ_s w(s)[1 − Π_{P∈S}(1 − p(s, P))].
     - PC_L uses p = Cm/m_s; PC_LS uses p = (Cm + m_s·1[P = P(s)])/(2m_s), and p = 1[P = P(s)] for a seed without one-hop mass.
     - A seed's uncovered share decays multiplicatively instead of vanishing at the first shard that touches it.
2. **Hub shards collect mass from any seed.** IN(P), the entries of E that end in a shard, has mean 1,058 and max 36,962 on MetaQA H4_SK.
   - SUMH = SUM/log2(1 + IN(P)).
   - FLAGGED: this is the shard-level analogue of the target-degree term that Message C kept out of the node score. It was measured, not proposed.
3. **Each source misses the other's shards:** *"localised rankings miss the FLAT seed's own partition on text (SQuAD SUM|NEFF_SUM -0.028 at B_N 1000), FLAT rankings miss the answer partitions on the KB (MetaQA F / D / SP / FT3: no gain)"*.
   - NSUMS = NSUM + FSUM; SDIVS = SDIV + FSUM; SHF = FSUM/ΣFSUM + SUM/ΣSUM (equal shares of the two sources).
   - Two-source unions UF_* = FSUM[:c_F] ∪ X[:c_X], ordered by min(rank_F/c_F, rank_X/c_X). Each side is ranked by its own mass and counted by its own concentration. The six are UF_SUM_NEFF, UF_SUM_KNEE, UF_T3_NEFF, UF_T3_KNEE, UF_SD_NEFF (SDIV) and UF_NS_NEFF (NSUM).
4. **Crowding:** *"FLAT junk crowds the KB served list, localised noise the text list."*
   - SL: the shards with localised evidence (SUM > 0) first, in ES order.
   - AND_T3 / AND_SD: ordered by max(rank_ES, rank_X) for X = TOP3 / SDIV, so both rankings must admit the shard.
   - PF_E_T3 and PF_E_SD (shard RRF of ES and X), and IL_E_T3 and IL_E_SD (interleave, ES first): round 1's PF_S_T3 / IL_S_T3 rebuilt on the router's own information.
5. **Count calibration:** *"NEFF is the Hill number of order 2 only."*
   - EXPH_x (order 1) and BPI_x = ⌈1/max p⌉ (order ∞) for every scored ranking.
   - NEFF and KNEE of the new scores.
   - NEV = the shards holding any evidence; NOACT_E = the distinct shards of O'[:200].
6. **Deployability:** *"S needs every node's O rank (a global FLAT order)."* A router that has contacted no shard knows only H_q with its FLAT ranks, the seeds' one-hop rows and the static maps.
   - ES = first appearance in O' = RRF(H_q, LOC), K0 = 60.
   - Ties go to seeds by FLAT rank, then LOC rank. Shards without evidence follow in static id order.
- **Not run** (each needs a hyperparameter): MMR (λ), xQuAD (λ) and τ-coverage (τ).

**Round 2 at B_N = 1000** (ALL − unrouted; as round 1's table):

| round-2 native arm | MetaQA H4_SK | MetaQA PHG | MuSiQue | SQuAD H4_SK | SQuAD PHG | mean B_P (five cells) |
|---|---|---|---|---|---|---|
| NSUM·NEFF_NSUM | +0.060 | +0.086 | −0.230 | −0.030 | −0.036 | 31.9 / 36.3 / 34.5 / 20.4 / 22.1 |
| NSUM·KNEE_NSUM | +0.042 | +0.057 | −0.055 | −0.016 | −0.018 | 73.5 / 74.4 / 115.9 / 34.1 / 34.9 |
| NSUM·EXPH_NSUM | +0.033 | +0.039 | −0.073 | −0.010 | −0.010 | 96.0 / 103.9 / 102.7 / 44.5 / 47.5 |
| NSUM·BPI_NSUM | +0.009 | +0.032 | −0.479 | −0.103 | −0.122 | 8.7 / 9.6 / 10.1 / 7.4 / 7.9 |
| PC_L·PC_L_NEFFG | +0.019 | +0.052 | −0.351 | −0.066 | −0.073 | 19.3 / 21.9 / 20.6 / 13.2 / 14.2 |
| PC_L·PC_L_KNEEG | +0.045 | +0.059 | −0.077 | −0.022 | −0.026 | 63.6 / 64.8 / 93.4 / 28.3 / 29.0 |
| PC_L·PC_L_COVER | −0.001 | 0 | 0 | 0 | 0 | 400.1 / 399.1 / 871.9 / 197.5 / 198.8 |
| PC_LS·PC_LS_NEFFG | −0.062 | −0.033 | −0.298 | −0.028 | −0.028 | 16.0 / 18.5 / 14.1 / 8.3 / 9.0 |
| PC_LS·PC_LS_KNEEG | −0.006 | +0.019 | −0.042 | −0.008 | −0.010 | 58.5 / 60.6 / 78.6 / 23.3 / 24.2 |
| PC_LS·PC_LS_COVER | −0.001 | 0 | 0 | 0 | 0 | 400.2 / 399.1 / 872.1 / 197.5 / 198.8 |
| SUMH·NEFF_SUMH | +0.061 | +0.072 | −0.187 | −0.027 | −0.026 | 42.5 / 49.2 / 55.3 / 27.5 / 30.5 |
| SUMH·KNEE_SUMH | +0.038 | +0.052 | −0.057 | −0.019 | −0.018 | 78.7 / 80.1 / 137.6 / 38.9 / 40.4 |
| SUMH·EXPH_SUMH | +0.024 | +0.032 | −0.056 | −0.009 | −0.008 | 116.9 / 127.0 / 152.4 / 57.0 / 61.9 |
| SUMH·BPI_SUMH | +0.013 | +0.045 | −0.448 | −0.107 | −0.112 | 10.9 / 12.3 / 13.8 / 9.3 / 10.1 |
| NSUMS·NEFF_NSUMS | +0.012 | +0.044 | −0.214 | −0.021 | −0.020 | 23.4 / 27.5 / 22.0 / 12.1 / 13.3 |
| NSUMS·KNEE_NSUMS | +0.038 | +0.056 | −0.024 | −0.007 | −0.007 | 67.5 / 69.3 / 98.4 / 28.3 / 29.2 |
| NSUMS·EXPH_NSUMS | +0.034 | +0.047 | −0.059 | −0.007 | −0.005 | 70.3 / 78.5 / 66.0 / 29.3 / 31.8 |
| NSUMS·BPI_NSUMS | −0.085 | −0.071 | −0.413 | −0.053 | −0.050 | 7.2 / 8.1 / 7.1 / 5.0 / 5.4 |
| SDIVS·NEFF_SDIVS | +0.055 | +0.062 | −0.064 | −0.006 | −0.004 | 67.0 / 72.2 / 156.9 / 61.5 / 66.9 |
| SDIVS·KNEE_SDIVS | +0.038 | +0.047 | −0.037 | −0.011 | −0.010 | 89.6 / 89.1 / 194.4 / 56.2 / 58.2 |
| SDIVS·EXPH_SDIVS | +0.016 | +0.025 | −0.014 | −0.001 | −0.002 | 156.9 / 160.0 / 300.9 / 97.7 / 102.8 |
| SDIVS·BPI_SDIVS | +0.065 | +0.088 | −0.240 | −0.033 | −0.029 | 15.3 / 16.9 / 37.2 / 20.2 / 22.7 |
| SHF·NEFF_SHF | +0.016 | +0.048 | −0.207 | −0.020 | −0.016 | 24.6 / 28.9 / 24.6 / 13.8 / 15.2 |
| SHF·KNEE_SHF | +0.042 | +0.059 | −0.024 | −0.006 | −0.009 | 68.0 / 70.0 / 105.5 / 30.4 / 31.4 |
| SHF·EXPH_SHF | +0.038 | +0.047 | −0.052 | −0.005 | −0.003 | 75.6 / 84.1 / 78.9 / 34.3 / 37.3 |
| SHF·BPI_SHF | −0.077 | −0.057 | −0.411 | −0.053 | −0.050 | 7.4 / 8.3 / 7.5 / 5.4 / 5.8 |
| UF_SUM_NEFF·UF_SUM_NEFF | +0.062 | +0.082 | −0.156 | −0.016 | −0.012 | 37.8 / 43.7 / 50.0 / 27.5 / 30.0 |
| UF_SUM_KNEE·UF_SUM_KNEE | +0.040 | +0.054 | −0.048 | −0.012 | −0.009 | 76.8 / 78.5 / 131.6 / 39.3 / 40.7 |
| UF_T3_NEFF·UF_T3_NEFF | +0.063 | +0.076 | −0.106 | −0.012 | −0.008 | 50.5 / 54.9 / 84.8 / 38.9 / 42.2 |
| UF_T3_KNEE·UF_T3_KNEE | +0.040 | +0.054 | −0.043 | −0.009 | −0.007 | 79.1 / 80.1 / 158.0 / 45.4 / 46.8 |
| UF_SD_NEFF·UF_SD_NEFF | +0.051 | +0.059 | −0.063 | −0.006 | −0.003 | 75.7 / 80.0 / 169.4 / 70.6 / 75.5 |
| UF_NS_NEFF·UF_NS_NEFF | +0.060 | +0.083 | −0.166 | −0.018 | −0.018 | 33.7 / 39.1 / 36.7 / 21.0 / 22.8 |
| SL·NEFF_SUM | −0.037 | −0.011 | −0.095 | −0.006 | −0.003 | 36.1 / 41.2 / 47.9 / 26.7 / 29.2 |
| SL·KNEE_SUM | +0.038 | +0.052 | −0.015 | −0.003 | −0.003 | 76.0 / 76.8 / 130.9 / 38.5 / 39.7 |
| AND_T3·NEFF_TOP3 | +0.054 | +0.065 | −0.084 | −0.008 | −0.009 | 49.5 / 53.3 / 84.0 / 38.5 / 41.7 |
| AND_T3·KNEE_TOP3 | +0.041 | +0.054 | −0.040 | −0.008 | −0.008 | 78.1 / 78.1 / 157.6 / 44.6 / 45.9 |
| AND_SD·NEFF_SDIV | +0.048 | +0.060 | −0.060 | −0.004 | −0.003 | 74.2 / 78.2 / 168.9 / 70.4 / 75.3 |
| AND_SD·NEFF_SUM | +0.042 | +0.064 | −0.187 | −0.021 | −0.021 | 36.1 / 41.2 / 47.9 / 26.7 / 29.2 |
| PF_E_T3·KNEE_TOP3 | +0.041 | +0.055 | −0.018 | −0.005 | −0.003 | 78.1 / 78.1 / 157.6 / 44.6 / 45.9 |
| IL_E_T3·NEFF_SDIV | +0.047 | +0.063 | −0.014 | −0.002 | −0.001 | 74.2 / 78.2 / 168.9 / 70.4 / 75.3 |
| PF_E_SD·KNEE_SDIV | +0.036 | +0.048 | −0.012 | −0.003 | −0.002 | 92.2 / 91.5 / 198.3 / 58.6 / 60.4 |
| IL_E_SD·NEFF_SDIV | +0.047 | +0.063 | −0.016 | −0.002 | −0.001 | 74.2 / 78.2 / 168.9 / 70.4 / 75.3 |
| SUM·EXPH_SUM | +0.027 | +0.034 | −0.062 | −0.012 | −0.008 | 108.2 / 116.8 / 137.5 / 55.6 / 59.8 |
| SUM·BPI_SUM | +0.008 | +0.041 | −0.486 | −0.114 | −0.124 | 9.4 / 10.3 / 12.5 / 9.2 / 9.8 |
| TOP3·EXPH_TOP3 (also round 1) | +0.023 | +0.031 | −0.034 | −0.003 | −0.004 | 128.0 / 131.8 / 210.2 / 71.9 / 75.6 |
| TOP3·BPI_TOP3 | +0.043 | +0.054 | −0.330 | −0.049 | −0.054 | 12.9 / 14.0 / 19.0 / 12.2 / 13.3 |
| SDIV·EXPH_SDIV | +0.017 | +0.023 | −0.024 | −0.001 | −0.002 | 166.6 / 167.9 / 313.3 / 104.2 / 108.7 |
| SDIV·BPI_SDIV | +0.071 | +0.093 | −0.263 | −0.038 | −0.037 | 16.5 / 18.0 / 42.1 / 27.2 / 29.9 |
| ES·NOACT_E | +0.022 | +0.034 | −0.037 | −0.003 | −0.001 | 105.2 / 111.3 / 74.3 / 47.2 / 50.0 |
| ES·NEV | −0.001 | 0 | 0 | 0 | 0 | 400.2 / 399.1 / 872.1 / 197.5 / 198.8 |
| ES·NSEEDP | +0.022 | +0.031 | −0.023 | −0.002 | 0 | 121.5 / 127.7 / 90.2 / 60.1 / 63.2 |
| ES·KNEE_TOP3 | +0.036 | +0.051 | −0.014 | −0.003 | −0.002 | 78.1 / 78.1 / 157.6 / 44.6 / 45.9 |
| ES·KNEE_SDIV | +0.032 | +0.046 | −0.008 | −0.002 | −0.001 | 92.2 / 91.5 / 198.3 / 58.6 / 60.4 |

- **Noisy-OR does not saturate, but then it does not stop.**
  - Its COVER counts are within 0.2 shards of NEV (399.1–400.2 / 871.9–872.1 / 197.5–198.8), which is 0.742–0.984 of the shards. So PC_L·PC_L_COVER and PC_LS·PC_LS_COVER are within 0.001 of unrouted.
  - PC_L's KNEEG count keeps +0.045 / +0.059 on the KB and loses 0.077 on MuSiQue.
- **NSUM and SUMH behave like SUM.** Under NEFF they are +0.060 / +0.086 / −0.230 and +0.061 / +0.072 / −0.187, against SUM's +0.064 / +0.082 / −0.218. Additive facility location removes the saturation, but its NEFF count is smaller than SUM's (34.5 against 47.9 shards on MuSiQue).
- **Adding the seed's own shard helps text at the KNEE counts.**
  - NSUMS·KNEE_NSUMS and SHF·KNEE_SHF lose 0.024 on MuSiQue (NSUM·KNEE_NSUM loses 0.055) and 0.006–0.009 on SQuAD. They keep +0.038 to +0.059 on the KB.
  - Under NEFF the text gain is small (MuSiQue −0.207 / −0.214 against −0.230). The KB gain also falls with the smaller count (NSUMS·NEFF_NSUMS +0.012 / +0.044 at 23.4 / 27.5 shards).
- **The ES-first interleaves lose least on MuSiQue** among the 31 round-1–2 native arms that keep both KB cells above +0.04.
  - IL_E_T3·NEFF_SDIV is +0.047 / +0.063 / −0.014 / −0.002 / −0.001, and IL_E_SD·NEFF_SDIV loses 0.016 on MuSiQue. Both contact 70–169 shards (0.144–0.373 of the shards).
  - Next come PF_E_T3·KNEE_TOP3 (−0.018) and SHF·KNEE_SHF (−0.024).
- **The Rényi order sets the trade-off.**
  - BPI (order ∞) is the smallest count and gives the largest KB gain: SDIV·BPI_SDIV is +0.071 / +0.093 at 16.5 / 18.0 shards, and −0.263 on MuSiQue at 42.1.
  - EXPH (order 1) is the largest: SDIV·EXPH_SDIV is −0.024 on MuSiQue at 313.3 shards, with +0.017 / +0.023 on the KB.
- **The router's own order costs little.**
  - ES·KNEE_SDIV is +0.032 / +0.046 / −0.008 / −0.002 / −0.001.
  - S·KNEE_SDIV, a cross arm on the global order, is +0.033 / +0.046 on the KB, +0.001 on MuSiQue and within 0.002 on SQuAD.
  - ES·NSEEDP (−0.023 on MuSiQue) matches S·NSEEDP (−0.025).

**34.5 Round 3: the regime conflict, the count shape and deployment** (`_l1d_route3.py`; records `route3_<ds>__v1`).
- **(7) The regime conflict.** The module's statement, verbatim: *"on the text cells the gold partitions are FLAT-seed partitions spread over many shards (every loss is reach, B_P(q) < LO; MuSiQue ES|KNEE_SDIV -0.008 at 198 partitions), on the KB cells the answer partitions are a few shards reached through the seeds' 1-hop rows (the gain is crowding relief at a small B_P; MetaQA SDIV|BPI_SDIV +0.071 / +0.093 at 16.5 / 18.0 partitions, -0.263 on MuSiQue). Every count of rounds 1-2 is a concentration of ONE evidence distribution; their MuSiQue / MetaQA ratios are 1.5-2.5x while the two regimes need ~6-10x."*
  - **A per-query regime share λ(q) ∈ [0, 1]**, read off the router's own evidence (0 for a query without one-hop evidence):
    - λ1 = 1 − Σ_{u∈H_q} L(u)/Σ_u L(u), the off-seed share of the IR_L1 mass;
    - λ2 = the share of O'[:200] that are not seeds;
    - λ3 = the share of LOC[:200] outside H_q;
    - λ4 = 1 − BPI_SDIV/NSEEDP, clipped to [0, 1]: how many times fewer effective shards the one-hop evidence occupies than the seeds do.
    - FLAGGED: λ4 was added after the 200-row smoke showed λ1 and λ3 at about 0.74–0.80 on MetaQA and MuSiQue alike (λ2 about 0.33 on both). It was designed with rounds 1–2's router means in view: about 0.86 on MetaQA, about 0.5 on MuSiQue and SQuAD.
  - **Mixtures of a text endpoint A = (ES, C) and a KB endpoint B = (SDE, BPI_SDIV).** FLAGGED: both endpoints were chosen on rounds 1–2's development results; only the per-query share is new.
    - Rankings: MXS_Lk orders by (1 − λ)/(K0 + rank_ES) + λ/(K0 + rank_SDE), ties → ES. MXT_Lk is the same with T3E.
    - Counts: MA_C_Lk = ⌈(1 − λ)C + λ·BPI_SDIV⌉ (arithmetic) and MG_C_Lk = ⌈C^(1−λ)·BPI_SDIV^λ⌉ (geometric), with C ∈ {KNEE_SDIV (KS), KNEE_TOP3 (KT), NOACT_E (NA)}.
    - Unions: UM_C_Lk = ES[:⌈(1 − λ)C⌉] ∪ SDE[:⌈λ·BPI_SDIV⌉], in the order min(rank_ES/c_E, rank_SDE/c_L), ties → ES. The union's size is its count.
    - FSE, SUME, T3E and SDE are FSUM, SUM, TOP3 and SDIV with ties → ES (node rows). SDK is SDIV with ties → FSUM → static shard id (sketch only).
- **(8) The count shape:** *"every count so far measures mass concentration (NEFF / EXPH / BPI / KNEE); none places a natural break."*
  - GAP_x = the position of the largest drop in x's sorted positive scores; LGAP_x = the position of the largest ratio drop.
  - x ∈ {FSUM, SUM, TOP3, SDIV, NSUM, SDIVS}, each with its own ranking.
- **Deployment failures, measured (no rule):**
  - (9) router state: what a router in front of a sharded graph must hold and read;
  - (10) coupling to B_N: ESTAR_BN = the distinct shards of O'[:B_N], the router-side estimate of PSTAR_BN, run as the diagnostic arm ES·ESTAR_BN (not in the factorial);
  - (11) load: the per-shard contact counts of each arm over the population.
- **Oracle** (gold-dependent; used by no rule):
  - per query, the better of A = ES·C and B = SDE·BPI_SDIV, which bounds any per-query switch between the two from above;
  - the AUC of each λ for the B-only against the A-only queries.
- **Not run:** MMR, xQuAD and τ-coverage (a hyperparameter each), and a sampled (CSI) router (§28's prototype ladder already measured fixed per-shard representatives).

**The mixtures at B_N = 1000.** ALL − unrouted as MetaQA H4_SK / MetaQA PHG / MuSiQue / SQuAD H4_SK / SQuAD PHG. MXS and MXT share each count, so they share its mean B_P.

| count (with its λ's ranking) | MXS | MXT | mean B_P |
|---|---|---|---|
| MA_KS_L1 | +0.069 / +0.088 / −0.135 / −0.027 / −0.024 | +0.068 / +0.085 / −0.094 / −0.016 / −0.014 | 34.6 / 35.6 / 66.5 / 34.8 / 37.2 |
| MA_KT_L1 | +0.071 / +0.092 / −0.148 / −0.028 / −0.026 | +0.070 / +0.085 / −0.104 / −0.017 / −0.015 | 31.4 / 32.5 / 59.7 / 31.5 / 33.8 |
| MA_NA_L1 | +0.066 / +0.084 / −0.182 / −0.027 / −0.027 | +0.066 / +0.084 / −0.136 / −0.018 / −0.015 | 38.1 / 40.7 / 48.1 / 32.7 / 35.4 |
| MG_KS_L1 | +0.073 / +0.094 / −0.168 / −0.027 / −0.026 | +0.064 / +0.084 / −0.120 / −0.017 / −0.015 | 25.2 / 26.8 / 54.0 / 32.6 / 35.2 |
| MG_KT_L1 | +0.073 / +0.094 / −0.173 / −0.029 / −0.027 | +0.063 / +0.085 / −0.128 / −0.018 / −0.016 | 24.2 / 25.8 / 51.8 / 30.4 / 32.8 |
| MG_NA_L1 | +0.076 / +0.096 / −0.190 / −0.028 / −0.027 | +0.066 / +0.084 / −0.145 / −0.018 / −0.016 | 26.1 / 28.3 / 46.2 / 31.2 / 33.9 |
| MA_KS_L2 | +0.046 / +0.065 / −0.016 / −0.004 / −0.002 | +0.045 / +0.065 / −0.016 / −0.004 / −0.002 | 67.1 / 67.1 / 142.3 / 49.2 / 51.4 |
| MA_KT_L2 | +0.055 / +0.067 / −0.023 / −0.006 / −0.005 | +0.055 / +0.068 / −0.021 / −0.005 / −0.004 | 57.7 / 58.2 / 116.1 / 39.6 / 41.4 |
| MA_NA_L2 | +0.041 / +0.052 / −0.047 / −0.005 / −0.003 | +0.041 / +0.053 / −0.045 / −0.004 / −0.002 | 75.7 / 80.2 / 62.9 / 41.3 / 44.1 |
| MG_KS_L2 | +0.058 / +0.074 / −0.027 / −0.004 / −0.002 | +0.058 / +0.072 / −0.022 / −0.004 / −0.002 | 51.5 / 52.7 / 112.1 / 45.9 / 48.3 |
| MG_KT_L2 | +0.061 / +0.076 / −0.029 / −0.006 / −0.005 | +0.059 / +0.073 / −0.025 / −0.005 / −0.005 | 46.1 / 47.4 / 96.7 / 38.1 / 40.0 |
| MG_NA_L2 | +0.056 / +0.073 / −0.052 / −0.005 / −0.003 | +0.056 / +0.071 / −0.053 / −0.005 / −0.002 | 56.1 / 59.9 / 58.8 / 39.3 / 42.1 |
| MA_KS_L3 | +0.070 / +0.090 / −0.115 / −0.023 / −0.022 | +0.069 / +0.086 / −0.080 / −0.016 / −0.013 | 34.6 / 35.6 / 72.9 / 36.0 / 38.4 |
| MA_KT_L3 | +0.072 / +0.091 / −0.135 / −0.025 / −0.024 | +0.071 / +0.085 / −0.090 / −0.017 / −0.015 | 31.3 / 32.5 / 64.5 / 32.2 / 34.5 |
| MA_NA_L3 | +0.067 / +0.088 / −0.175 / −0.025 / −0.024 | +0.068 / +0.083 / −0.130 / −0.016 / −0.013 | 37.8 / 40.4 / 48.3 / 32.8 / 35.5 |
| MG_KS_L3 | +0.074 / +0.095 / −0.147 / −0.024 / −0.024 | +0.065 / +0.084 / −0.107 / −0.017 / −0.014 | 25.1 / 26.7 / 57.6 / 33.4 / 36.1 |
| MG_KT_L3 | +0.076 / +0.095 / −0.157 / −0.025 / −0.024 | +0.067 / +0.085 / −0.116 / −0.017 / −0.015 | 24.1 / 25.7 / 54.7 / 31.0 / 33.4 |
| MG_NA_L3 | +0.076 / +0.096 / −0.180 / −0.026 / −0.024 | +0.068 / +0.085 / −0.139 / −0.017 / −0.015 | 25.9 / 28.0 / 46.2 / 31.3 / 34.0 |
| MA_KS_L4 | +0.075 / +0.090 / −0.053 / −0.009 / −0.005 | +0.069 / +0.084 / −0.039 / −0.006 / −0.005 | 27.5 / 29.1 / 116.7 / 41.0 / 43.7 |
| MA_KT_L4 | +0.076 / +0.095 / −0.062 / −0.011 / −0.007 | +0.072 / +0.086 / −0.047 / −0.007 / −0.006 | 25.3 / 27.0 / 97.0 / 34.5 / 36.8 |
| MA_NA_L4 | +0.074 / +0.094 / −0.100 / −0.009 / −0.006 | +0.071 / +0.087 / −0.082 / −0.007 / −0.006 | 28.9 / 31.5 / 51.8 / 34.2 / 37.1 |
| MG_KS_L4 | +0.075 / +0.090 / −0.076 / −0.011 / −0.006 | +0.064 / +0.078 / −0.059 / −0.007 / −0.006 | 21.7 / 23.6 / 93.3 / 38.2 / 41.1 |
| MG_KT_L4 | +0.074 / +0.094 / −0.082 / −0.012 / −0.008 | +0.065 / +0.079 / −0.064 / −0.007 / −0.006 | 21.1 / 22.8 / 81.9 / 33.2 / 35.6 |
| MG_NA_L4 | +0.077 / +0.094 / −0.108 / −0.011 / −0.007 | +0.067 / +0.082 / −0.092 / −0.007 / −0.006 | 22.0 / 24.1 / 48.8 / 32.6 / 35.5 |

**The unions, the endpoints and the gap counts at B_N = 1000** (the other 27 native arms of round 3):

| round-3 native arm | MetaQA H4_SK | MetaQA PHG | MuSiQue | SQuAD H4_SK | SQuAD PHG | mean B_P (five cells) |
|---|---|---|---|---|---|---|
| UM_KS_L1·UM_KS_L1 | +0.048 | +0.067 | −0.077 | −0.028 | −0.023 | 27.6 / 28.3 / 51.7 / 26.7 / 28.5 |
| UM_KS_L2·UM_KS_L2 | +0.041 | +0.061 | −0.021 | −0.005 | −0.003 | 61.5 / 61.1 / 127.8 / 41.6 / 43.1 |
| UM_KS_L3·UM_KS_L3 | +0.050 | +0.067 | −0.064 | −0.022 | −0.019 | 27.4 / 28.1 / 55.8 / 26.9 / 28.7 |
| UM_KS_L4·UM_KS_L4 | +0.066 | +0.081 | −0.040 | −0.005 | −0.005 | 21.5 / 22.5 / 100.6 / 31.8 / 33.9 |
| UM_KT_L1·UM_KT_L1 | +0.048 | +0.068 | −0.092 | −0.033 | −0.026 | 25.2 / 26.0 / 46.7 / 24.5 / 26.4 |
| UM_KT_L2·UM_KT_L2 | +0.048 | +0.064 | −0.027 | −0.006 | −0.006 | 52.2 / 52.2 / 101.8 / 32.5 / 33.6 |
| UM_KT_L3·UM_KT_L3 | +0.049 | +0.071 | −0.080 | −0.027 | −0.022 | 25.0 / 25.8 / 49.4 / 24.5 / 26.3 |
| UM_KT_L4·UM_KT_L4 | +0.067 | +0.082 | −0.050 | −0.008 | −0.008 | 19.9 / 21.0 / 82.0 / 26.4 / 28.1 |
| UM_NA_L1·UM_NA_L1 | +0.053 | +0.073 | −0.172 | −0.032 | −0.027 | 30.5 / 32.4 / 40.1 / 25.6 / 27.7 |
| UM_NA_L2·UM_NA_L2 | +0.041 | +0.055 | −0.063 | −0.006 | −0.004 | 70.2 / 74.2 / 51.2 / 34.2 / 36.3 |
| UM_NA_L3·UM_NA_L3 | +0.053 | +0.068 | −0.153 | −0.027 | −0.024 | 29.9 / 31.8 / 38.9 / 24.9 / 26.9 |
| UM_NA_L4·UM_NA_L4 | +0.065 | +0.081 | −0.092 | −0.009 | −0.007 | 22.5 / 24.3 / 40.3 / 25.6 / 27.8 |
| SDE·BPI_SDIV and SDK·BPI_SDIV (equal) | +0.071 | +0.093 | −0.263 | −0.038 | −0.037 | 16.5 / 18.0 / 42.1 / 27.2 / 29.9 |
| T3E·BPI_TOP3 | +0.043 | +0.054 | −0.330 | −0.049 | −0.054 | 12.9 / 14.0 / 19.0 / 12.2 / 13.3 |
| FSE·GAP_FSUM | −0.391 | −0.397 | −0.712 | −0.173 | −0.164 | 1.2 / 1.2 / 1.4 / 1.3 / 1.4 |
| FSE·LGAP_FSUM | −0.389 | −0.394 | −0.620 | −0.111 | −0.109 | 1.6 / 1.5 / 2.4 / 2.2 / 2.3 |
| SUME·GAP_SUM | −0.183 | −0.192 | −0.767 | −0.454 | −0.542 | 1.6 / 1.5 / 1.8 / 2.0 / 1.8 |
| SUME·LGAP_SUM | −0.132 | −0.145 | −0.669 | −0.196 | −0.195 | 9.4 / 12.7 / 68.1 / 77.8 / 95.5 |
| T3E·GAP_TOP3 | −0.226 | −0.234 | −0.729 | −0.233 | −0.249 | 1.8 / 1.9 / 2.0 / 2.4 / 2.7 |
| T3E·LGAP_TOP3 | −0.154 | −0.148 | −0.576 | −0.078 | −0.073 | 28.9 / 40.4 / 102.5 / 72.0 / 88.8 |
| SDE·GAP_SDIV | −0.170 | −0.187 | −0.727 | −0.256 | −0.270 | 1.7 / 1.8 / 5.0 / 8.3 / 9.9 |
| SDE·LGAP_SDIV | −0.109 | −0.111 | −0.608 | −0.034 | −0.023 | 5.3 / 7.4 / 45.6 / 122.4 / 130.2 |
| NSUM·GAP_NSUM | −0.193 | −0.200 | −0.755 | −0.384 | −0.460 | 1.5 / 1.5 / 1.7 / 1.8 / 1.7 |
| NSUM·LGAP_NSUM | −0.120 | −0.118 | −0.493 | −0.134 | −0.131 | 94.1 / 118.1 / 277.0 / 92.2 / 102.6 |
| SDIVS·GAP_SDIVS | −0.167 | −0.186 | −0.712 | −0.192 | −0.208 | 1.7 / 1.7 / 3.1 / 3.2 / 3.8 |
| SDIVS·LGAP_SDIVS | −0.118 | −0.135 | −0.594 | −0.055 | −0.043 | 4.2 / 9.4 / 41.9 / 93.2 / 104.5 |

- **Only λ4 differs between the regimes.** The means are λ1 0.76 / 0.82 / 0.75, λ2 0.34 / 0.35 / 0.30, λ3 0.76 / 0.79 / 0.72 and λ4 0.86 / 0.54 / 0.55, on MetaQA / MuSiQue / SQuAD (34.9).
- **The λ4 mixtures and unions keep most of the KB gain with a smaller text loss** than the round-1–2 natives with a similar KB gain.
  - MXS_L4·MA_KS_L4 is +0.075 / +0.090 / −0.053 / −0.009 / −0.005, and UM_KS_L4·UM_KS_L4 is +0.066 / +0.081 / −0.040 / −0.005 / −0.005.
  - Of the round-1–2 natives, 9 keep ≥ +0.06 in both KB cells, and each loses 0.106 or more on MuSiQue: UF_T3_NEFF·UF_T3_NEFF −0.106, MAX·NEFF_MAX −0.118, TOP3·NEFF_TOP3 −0.124, UF_SUM_NEFF·UF_SUM_NEFF −0.156, SUMH·NEFF_SUMH −0.187, SUM·NEFF_SUM −0.218, NSUM·NEFF_NSUM −0.230, SDIVS·BPI_SDIVS −0.240, SDIV·BPI_SDIV −0.263.
- **λ2 gives larger counts** (λ2 is about 0.3 in every cell) and the smallest MuSiQue losses among the mixtures: MXS_L2·MA_KS_L2 is −0.016 at 142.3 shards, with +0.046 / +0.065 on the KB.
- **λ1 and λ3 (0.72–0.82 in every cell) sit near the KB endpoint.** Their 30 arms lose 0.064–0.190 on MuSiQue.
- **MG ≤ MA for every query** (a geometric mean against an arithmetic one), so the MG counts are smaller, with more KB gain and more MuSiQue loss.
- **MXT against MXS.** With λ1, λ3 and λ4, MXT (T3E on the KB side) loses 0.014–0.048 less on MuSiQue and 0–0.012 less on SQuAD, and keeps up to 0.015 less on the KB (MA_NA_L1 / MA_NA_L3 keep 0.001 more on H4_SK); with λ2 the two orders are within 0.006 in every cell.
- **GAP is degenerate.**
  - The largest absolute drop comes after 1.2–2.7 shards on average; GAP_SDIV and GAP_SDIVS reach 3.1–9.9 on text. LGAP ranges from 1.5 to 277.0.
  - Every native GAP and LGAP arm is below unrouted in every cell, by 0.023–0.767.
- **SDK·BPI_SDIV and SDE·BPI_SDIV have the same ALL count at every B_N in every cell**: the sketch-only tie-break costs nothing. SDK, SDE and SDIV tie for the MetaQA H4_SK envelope at B_N 100–500 and for PHG's at 250–2000 (34.6).

**34.6 The envelope, fan-out needs and counts: few shards or many.** ALL gold, with the ranking and fixed B_P of each best point in brackets:

| ALL gold at B_N = | 100 | 250 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|---|
| MetaQA (both cells): unrouted | 0.382 | 0.428 | 0.460 | 0.508 | 0.556 | 0.636 |
| MetaQA H4_SK: envelope | 0.435 (SDK 16; SDE, SDIV tie) | 0.508 (SDK 15) | 0.557 (SDK 16) | 0.586 (PD 31) | 0.617 (PD 48) | 0.661 (PD 49) |
| MetaQA H4_SK: S, best fixed B_P | 0.388 (47) | 0.450 (40) | 0.512 (44) | 0.559 (44) | 0.590 (51) | 0.637 (417) |
| MetaQA PHG: envelope | 0.434 (MAX 15) | 0.509 (SDK 15) | 0.574 (SDK 18) | 0.609 (SDK 26) | 0.649 (SDK 19) | 0.690 (PD 49) |
| MetaQA PHG: S, best fixed B_P | 0.390 (52) | 0.457 (45) | 0.519 (44) | 0.573 (52) | 0.611 (54) | 0.656 (129) |
| MuSiQue: unrouted | 0.640 | 0.742 | 0.793 | 0.840 | 0.880 | 0.928 |
| MuSiQue: envelope | 0.645 (D 440) | 0.743 (UM_KT_L3 130) | 0.795 (PF_F_T3 178) | 0.843 (OT3 281; PF_F_T3, SHF tie) | 0.883 (FL_L 370) | 0.928 (SC_LS 640; all 68 tie) |
| MuSiQue: S, best fixed B_P | 0.643 (45) | 0.742 (127) | 0.794 (176) | 0.842 (273) | 0.882 (362) | 0.928 (640) |
| SQuAD (both cells): unrouted | 0.986 | 0.996 | 0.997 | 1.000 | 1.000 | 1.000 |
| SQuAD H4_SK: envelope | 0.989 (F 46) | 0.996 (FL_LS 43) | 0.998 (FL_LS 43) | 1.000 (D 96) | 1.000 (FT3 128) | 1.000 (FT3 128) |
| SQuAD PHG: envelope | 0.989 (FSUM 39) | 0.997 (F 66) | 0.999 (FSUM 46) | 1.000 (D 88) | 1.000 (D 88) | 1.000 (D 88) |
| envelope − unrouted, in queries (MetaQA H4_SK; PHG; MuSiQue; SQuAD H4_SK; SQuAD PHG) | 107; 105; 11; 6; 7 | 160; 161; 3; 1; 2 | 193; 228; 3; 2; 3 | 156; 201; 6; 0; 1 | 123; 186; 5; 0; 0 | 50; 108; 0; 0; 0 |

- SQuAD's unrouted is 0.9955 at B_N 250 (1,991 of 2,000) and its H4_SK envelope 0.996 (1,992). Its 1.000 at 1000 is 0.9995, as in §33.
- Ties at the envelope (rankings whose best fixed point equals it):
  - MetaQA: SDE and SDIV tie with every SDK entry, at the same B_P. H4_SK at 1000–5000 and PHG at 5000 are PD alone; PHG at 100 is MAX alone.
  - MuSiQue: OT3 281, PF_F_T3 306 and SHF 467 at 1000; all 68 rankings at 5000; one ranking at the other B_N.
  - SQuAD H4_SK: F 46 and FT3 51 at 100; 65 rankings at 250; F 68, FSUM 45, FSUMFV 45, FT3 51, FL_LS 43 and FSE 45 at 500; all 68 from 1000.
  - SQuAD PHG: F 41, FSUM 39, FSUMFV 45 and FSE 39 at 100; F 66 and FSUM, FSUMFV, FSE 67 at 250; FSUM 46, FSUMFV 47 and FSE 46 at 500; 21 rankings at 1000; all 68 from 2000.
- **On text, the envelope is at most 0.006 above unrouted** (MuSiQue 0–11 queries, SQuAD 0–7). **On the KB it is +0.053 to +0.114 above unrouted for B_N ≤ 2000**, and +0.025 / +0.054 at 5000.
- **PD (FLAGGED) sets the MetaQA H4_SK envelope from B_N 1000** and PHG's at 5000:
  - H4_SK: by 1 query at 1000 (without PD, SDK 0.586), 6 queries at 2000 (SDIVS 0.614) and 33 queries at 5000 (UF_SUM_KNEE 0.645);
  - PHG: by 18 queries at 5000 (SUME 0.681).

**The fan-out a ranking needs.** The smallest fixed B_P within 0.01 of unrouted, at B_N = 1000:

| ranking | MetaQA H4_SK | MetaQA PHG | MuSiQue | SQuAD H4_SK | SQuAD PHG | MuSiQue / MetaQA H4_SK |
|---|---|---|---|---|---|---|
| S | 28 | 30 | 167 | 21 | 24 | 5.96 |
| ES | 29 | 31 | 217 | 21 | 24 | 7.48 |
| OT3 | 48 | 29 | 129 | 28 | 34 | 2.69 |
| IL_S_T3 | 12 | 11 | 183 | 33 | 30 | 15.25 |
| UM_KS_L4 | 8 | 8 | 179 | 26 | 26 | 22.38 |
| MXS_L4 | 7 | 7 | 219 | 40 | 37 | 31.29 |
| TOP3 | 8 | 7 | 326 | 59 | 56 | 40.75 |
| SDIV | 6 | 6 | 431 | 84 | 81 | 71.83 |
| the envelope point's B_P (for comparison) | 31 | 26 | 281 | 96 | 88 | 9.06 |

- F, D, SP and FT3 on MetaQA need 412–432 shards and never exceed unrouted's 0.508; FSUM and FSUMFV need 139 / 146.

**PSTAR_BN**, the fan-out at which S reproduces the unrouted list (mean):

| PSTAR_BN at B_N = | 100 | 250 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|---|
| MetaQA H4_SK | 56.0 | 124.7 | 210.9 | 316.5 | 400.6 | 431.1 |
| MetaQA PHG | 59.6 | 131.9 | 218.9 | 321.8 | 400.6 | 428.6 |
| MuSiQue | 40.1 | 88.1 | 154.7 | 263.9 | 429.7 | 744.1 |
| SQuAD H4_SK | 25.6 | 54.4 | 89.0 | 132.6 | 172.9 | 198.8 |
| SQuAD PHG | 27.3 | 57.7 | 94.0 | 138.4 | 177.2 | 199.5 |

- At B_N 1000, PSTAR_BN is 0.733 / 0.745 of MetaQA's shards, 0.225 of MuSiQue's and 0.656 / 0.685 of SQuAD's.
- The golds' distinct shards (mean) are 5.1 / 5.1 / 2.3 / 1.0 / 1.0. MetaQA's median is 1, its p90 13 / 12 and its max 120 / 117 (H4_SK / PHG); MuSiQue's median is 2, its p90 3 and its max 4.
- S's LO (the fan-out that contacts every gold shard; mean) is 120.97 / 112.32 / 59.90 / 3.09 / 3.23 (stored two-decimal means).

**The counts: MuSiQue's fan-out against MetaQA's** (mean B_P; a selection of the 125 counts, all of which are in the summary npz as `BPSUM__<cell>`):

| count | MetaQA H4_SK | MetaQA PHG | MuSiQue | SQuAD H4_SK | SQuAD PHG | MuSiQue / MetaQA H4_SK |
|---|---|---|---|---|---|---|
| NEFF_FSUMFV (round 1) | 61.6 | 72.7 | 35.3 | 17.7 | 19.8 | 0.57 |
| SC_L_COVER | 29.5 | 28.7 | 17.4 | 19.5 | 18.7 | 0.59 |
| FL_L_KNEEG | 14.0 | 13.6 | 9.5 | 8.0 | 7.8 | 0.68 |
| NOACT | 103.7 | 109.8 | 73.1 | 45.6 | 48.5 | 0.71 |
| NSEEDP | 121.5 | 127.7 | 90.2 | 60.1 | 63.2 | 0.74 |
| FL_LS_COVER | 122.4 | 128.6 | 91.4 | 60.4 | 63.5 | 0.75 |
| SC_L_NEFFG | 3.6 | 3.7 | 2.9 | 2.8 | 2.7 | 0.80 |
| NEFF_FSUM | 13.6 | 15.4 | 11.0 | 7.0 | 7.5 | 0.81 |
| NEFF_SUM | 36.1 | 41.2 | 47.9 | 26.7 | 29.2 | 1.33 |
| EXPH_TOP3 | 128.0 | 131.8 | 210.2 | 71.9 | 75.6 | 1.64 |
| NEFF_TOP3 | 49.5 | 53.3 | 84.0 | 38.5 | 41.7 | 1.70 |
| KNEE_SUM | 76.0 | 76.8 | 130.9 | 38.5 | 39.7 | 1.72 |
| NEFF_MAX | 54.9 | 56.6 | 96.6 | 42.7 | 45.9 | 1.76 |
| KNEE_TOP3 | 78.1 | 78.1 | 157.6 | 44.6 | 45.9 | 2.02 |
| KNEE_SDIV | 92.2 | 91.5 | 198.3 | 58.6 | 60.4 | 2.15 |
| NEFF_SDIV | 74.2 | 78.2 | 168.9 | 70.4 | 75.3 | 2.28 |
| KNEE_PD | 81.6 | 82.6 | 190.1 | 48.3 | 49.0 | 2.33 |
| NOACT_E (round 2) | 105.2 | 111.3 | 74.3 | 47.2 | 50.0 | 0.71 |
| NEFF_NSUM | 31.9 | 36.3 | 34.5 | 20.4 | 22.1 | 1.08 |
| NEFF_SUMH | 42.5 | 49.2 | 55.3 | 27.5 | 30.5 | 1.30 |
| BPI_TOP3 | 12.9 | 14.0 | 19.0 | 12.2 | 13.3 | 1.47 |
| PC_L_KNEEG | 63.6 | 64.8 | 93.4 | 28.3 | 29.0 | 1.47 |
| EXPH_SDIVS | 156.9 | 160.0 | 300.9 | 97.7 | 102.8 | 1.92 |
| NEV | 400.2 | 399.1 | 872.1 | 197.5 | 198.8 | 2.18 |
| NEFF_SDIVS | 67.0 | 72.2 | 156.9 | 61.5 | 66.9 | 2.34 |
| BPI_SDIV | 16.5 | 18.0 | 42.1 | 27.2 | 29.9 | 2.55 |
| UM_NA_L4 (round 3) | 22.5 | 24.3 | 40.3 | 25.6 | 27.8 | 1.79 |
| MA_KS_L2 | 67.1 | 67.1 | 142.3 | 49.2 | 51.4 | 2.12 |
| MG_KS_L2 | 51.5 | 52.7 | 112.1 | 45.9 | 48.3 | 2.18 |
| MA_KT_L4 | 25.3 | 27.0 | 97.0 | 34.5 | 36.8 | 3.83 |
| UM_KT_L4 | 19.9 | 21.0 | 82.0 | 26.4 | 28.1 | 4.13 |
| MA_KS_L4 | 27.5 | 29.1 | 116.7 | 41.0 | 43.7 | 4.24 |
| MG_KS_L4 | 21.7 | 23.6 | 93.3 | 38.2 | 41.1 | 4.29 |
| UM_KS_L4 | 21.5 | 22.5 | 100.6 | 31.8 | 33.9 | 4.68 |
| LGAP_SUM | 9.4 | 12.7 | 68.1 | 77.8 | 95.5 | 7.21 |
| LGAP_SDIV | 5.3 | 7.4 | 45.6 | 122.4 | 130.2 | 8.61 |
| LGAP_SDIVS | 4.2 | 9.4 | 41.9 | 93.2 | 104.5 | 10.05 |
| PSTAR_BN at B_N 1000 (reference) | 316.5 | 321.8 | 263.9 | 132.6 | 138.4 | 0.83 |

- **What "few" and "many" are here.**
  - MetaQA's golds lie in few shards (5.1 on average, median 1) but late in S's order (LO 120.97 / 112.32).
  - MuSiQue's lie in 2.3 shards (at most 4), spread along S's order (LO 59.90 on average, median 12, p90 168). Every MuSiQue loss is reach: the gold shards are not contacted (B_P(q) < LO; round 3's docstring).
  - SQuAD's lie in one shard (LO 3.09 / 3.23).
- **The spread the counts give.**
  - Rounds 1–2's 77 counts give MuSiQue 0.57–2.55× MetaQA H4_SK's fan-out. The spread counts (NSEEDP 0.74, NOACT 0.71) and the round-1 greedies' COVER counts (0.59–0.75) are below 1×; the noisy-OR COVER counts are within 0.2 shards of NEV (2.18×).
  - λ4's KS / KT mixtures and unions give 3.83–4.68×.
  - LGAP_SUM / SDIV / SDIVS give 7.21–10.05×, but at the wrong scale (MetaQA 4.2–9.4 shards; SQuAD 77.8–130.2).
  - The fixed-B_P needs at B_N 1000 are 5.96× for S and 9.06× at the envelope, and more for the localised rankings.
- **The need depends on B_N; no rule count does.**
  - S's best fixed B_P on MuSiQue grows from 45 at B_N 100 to 640 at 5000. MetaQA's stays at 40–54 up to 2000, then 417 / 129 at 5000.
  - PSTAR_BN grows with B_N in every cell (MetaQA 56.0 → 431.1, MuSiQue 40.1 → 744.1).

**34.7 Across B_N: a panel of arms.** Source: `route_VIEW__v1.json` (sha d549cb76…), a post-hoc view. FLAGGED: the panel's arms were named after the summary existed. Each entry gives the MetaQA H4_SK / PHG difference from unrouted, then the worst text cell, then [the worst cell's distance to the envelope]. The first column gives the arm's mean B_P over the five cells and its largest fan-out fraction.

| arm [mean B_P; largest fraction] | B_N 100 | 250 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|---|
| UM_KT_L4·MG_KT_L1 [24.2 / 25.8 / 51.8 / 30.4 / 32.8; 0.163] | +0.037 / +0.036; −0.003 [−0.017] | +0.066 / +0.066; −0.026 [−0.028] | +0.085 / +0.103; −0.045 [−0.046] | +0.069 / +0.091; −0.070 [−0.073] | +0.048 / +0.076; −0.100 [−0.103] | −0.022 / +0.019; −0.145 [−0.145] |
| UM_KT_L4·MA_KT_L4 [25.3 / 27.0 / 97.0 / 34.5 / 36.8; 0.182] | +0.035 / +0.036; −0.001 [−0.019] | +0.066 / +0.066; −0.008 [−0.015] | +0.086 / +0.099; −0.022 [−0.024] | +0.070 / +0.090; −0.037 [−0.040] | +0.046 / +0.075; −0.060 [−0.062] | −0.020 / +0.020; −0.097 [−0.097] |
| UM_KS_L4·MA_KS_L4 [27.5 / 29.1 / 116.7 / 41.0 / 43.7; 0.216] | +0.030 / +0.031; 0 [−0.024] | +0.059 / +0.063; −0.004 [−0.021] | +0.083 / +0.096; −0.018 [−0.019] | +0.065 / +0.085; −0.031 [−0.034] | +0.042 / +0.071; −0.048 [−0.051] | −0.021 / +0.020; −0.082 [−0.082] |
| UM_KS_L4·UM_KS_L4 [21.5 / 22.5 / 100.6 / 31.8 / 33.9; 0.168] | +0.033 / +0.029; 0 [−0.024] | +0.063 / +0.063; −0.008 [−0.018] | +0.083 / +0.092; −0.027 [−0.028] | +0.066 / +0.081; −0.040 [−0.043] | +0.039 / +0.062; −0.063 [−0.066] | −0.031 / +0.001; −0.101 [−0.101] |
| MXS_L4·MA_KS_L4 [as UM_KS_L4·MA_KS_L4; 0.216] | +0.040 / +0.040; −0.013 [−0.018] | +0.069 / +0.067; −0.025 [−0.027] | +0.088 / +0.099; −0.042 [−0.044] | +0.075 / +0.090; −0.053 [−0.056] | +0.054 / +0.076; −0.071 [−0.074] | −0.010 / +0.025; −0.103 [−0.103] |
| IL_S_T3·MG_KS_L2 [51.5 / 52.7 / 112.1 / 45.9 / 48.3; 0.239] | +0.009 / +0.011; 0 [−0.045] | +0.028 / +0.030; −0.002 [−0.052] | +0.059 / +0.067; −0.011 [−0.048] | +0.060 / +0.078; −0.021 [−0.024] | +0.043 / +0.064; −0.038 [−0.041] | −0.002 / +0.027; −0.070 [−0.070] |
| UM_KT_L4·NEFF_SDIVS [67.0 / 72.2 / 156.9 / 61.5 / 66.9; 0.331] | +0.008 / +0.009; 0 [−0.046] | +0.020 / +0.017; −0.001 [−0.064] | +0.050 / +0.052; −0.007 [−0.062] | +0.053 / +0.064; −0.016 [−0.037] | +0.040 / +0.061; −0.028 [−0.033] | −0.004 / +0.024; −0.050 [−0.050] |
| SHF·KNEE_SDIV [92.2 / 91.5 / 198.3 / 58.6 / 60.4; 0.299] | +0.001 / +0.003; −0.001 [−0.053] | +0.005 / +0.007; −0.001 [−0.075] | +0.025 / +0.034; −0.001 [−0.081] | +0.029 / +0.045; −0.006 [−0.056] | +0.033 / +0.053; −0.016 [−0.040] | −0.003 / +0.021; −0.031 [−0.034] |
| S·KNEE_TOP3 [78.1 / 78.1 / 157.6 / 44.6 / 45.9; 0.227] | +0.001 / 0; 0 [−0.053] | +0.010 / +0.014; −0.001 [−0.070] | +0.034 / +0.044; −0.002 [−0.071] | +0.037 / +0.053; −0.005 [−0.048] | +0.033 / +0.052; −0.015 [−0.042] | −0.013 / +0.014; −0.034 [−0.040] |
| S·KNEE_SDIV [92.2 / 91.5 / 198.3 / 58.6 / 60.4; 0.299] | 0 / −0.001; 0 [−0.054] | +0.007 / +0.009; −0.001 [−0.074] | +0.031 / +0.037; −0.001 [−0.078] | +0.033 / +0.046; −0.002 [−0.055] | +0.029 / +0.048; −0.008 [−0.045] | −0.011 / +0.015; −0.026 [−0.040] |
| ES·KNEE_SDIV [the same B_P; 0.299] | 0 / −0.001; 0 [−0.054] | +0.006 / +0.007; 0 [−0.074] | +0.029 / +0.036; −0.003 [−0.078] | +0.032 / +0.046; −0.008 [−0.055] | +0.029 / +0.048; −0.018 [−0.046] | −0.012 / +0.014; −0.036 [−0.040] |
| S·EXPH_SDIVS [156.9 / 160.0 / 300.9 / 97.7 / 102.8; 0.509] | 0 / 0; 0 [−0.054] | 0 / 0; 0 [−0.081] | +0.007 / +0.011; 0 [−0.104] | +0.015 / +0.025; 0 [−0.076] | +0.018 / +0.030; −0.004 [−0.064] | −0.009 / +0.010; −0.012 [−0.045] |
| SDIV·BPI_SDIV [16.5 / 18.0 / 42.1 / 27.2 / 29.9; 0.148] | +0.049 / +0.046; −0.139 [−0.145] | +0.081 / +0.080; −0.195 [−0.197] | +0.094 / +0.107; −0.227 [−0.229] | +0.071 / +0.093; −0.263 [−0.266] | +0.047 / +0.076; −0.297 [−0.300] | −0.033 / +0.004; −0.343 [−0.343] |
| TOP3·NEFF_TOP3 [49.5 / 53.3 / 84.0 / 38.5 / 41.7; 0.206] | +0.019 / +0.019; −0.042 [−0.047] | +0.031 / +0.031; −0.077 [−0.078] | +0.065 / +0.067; −0.104 [−0.106] | +0.064 / +0.077; −0.124 [−0.127] | +0.043 / +0.066; −0.149 [−0.152] | −0.007 / +0.026; −0.183 [−0.183] |
| SUM·NEFF_SUM [36.1 / 41.2 / 47.9 / 26.7 / 29.2; 0.145] | +0.019 / +0.022; −0.102 [−0.108] | +0.038 / +0.042; −0.156 [−0.157] | +0.069 / +0.074; −0.187 [−0.188] | +0.064 / +0.082; −0.218 [−0.221] | +0.045 / +0.070; −0.246 [−0.249] | −0.010 / +0.030; −0.288 [−0.288] |
| ES·ESTAR_BN [coupled to B_N; 34.9] | 0 / +0.001; −0.001 [−0.054] | +0.001 / +0.001; −0.001 [−0.080] | +0.002 / +0.002; −0.004 [−0.112] | +0.001 / +0.003; −0.006 [−0.098] | −0.002 / 0; −0.007 [−0.093] | 0 / 0; −0.002 [−0.054] |

- At B_N 1000, every cell of five of these arms:
  - UM_KS_L4·MA_KS_L4: +0.065 / +0.085 / −0.031 / −0.004 / −0.003 (fan-out fractions 0.064 / 0.067 / 0.099 / 0.203 / 0.216);
  - IL_S_T3·MG_KS_L2: +0.060 / +0.078 / −0.021 / −0.005 / −0.003;
  - S·KNEE_TOP3: +0.037 / +0.053 / −0.005 / −0.003 / −0.003 (fractions 0.181 / 0.181 / 0.134 / 0.221 / 0.227);
  - ES·KNEE_SDIV: +0.032 / +0.046 / −0.008 / −0.002 / −0.001 (fractions 0.214 / 0.212 / 0.169 / 0.290 / 0.299);
  - S·EXPH_SDIVS: +0.015 / +0.025 / +0.001 / 0 / 0.
  - At B_N 500, UM_KS_L4·MA_KS_L4 is +0.083 / +0.096 / −0.018 / −0.002 / −0.001.
- **At B_N 5000** every rule of the panel is below unrouted on MetaQA H4_SK (−0.002 to −0.033) and +0.001 to +0.030 on PHG.
- **For a fixed count, the text loss grows with B_N**, because the need grows (34.6).
  - UM_KS_L4·MA_KS_L4's worst text cell goes 0 / −0.004 / −0.018 / −0.031 / −0.048 / −0.082 from B_N 100 to 5000; S·KNEE_SDIV's goes 0 / −0.001 / −0.001 / −0.002 / −0.008 / −0.026.
  - ES·ESTAR_BN, whose count grows with B_N, stays within 0.007 at every B_N.
- **Where the KB gain peaks.**
  - The λ4-ranked arms with a mixture or union count peak at B_N 500: +0.083 to +0.088 on H4_SK, +0.092 to +0.103 on PHG (UM_KT_L4·NEFF_SDIVS, with a round-2 count, peaks at 1000: +0.053 / +0.064).
  - The KNEE arms peak at B_N 1000–2000: +0.032 to +0.037 on H4_SK, +0.048 to +0.053 on PHG.

**Paired against unrouted at B_N = 1000** (gained / lost queries; exact McNemar p, descriptive, where cited):

| arm | MetaQA H4_SK | MetaQA PHG | MuSiQue | SQuAD H4_SK | SQuAD PHG |
|---|---|---|---|---|---|
| TOP3·NEFF_TOP3 | +167/−40 | +185/−32 | +27/−274 | +0/−26 | +1/−23 |
| MAX·NEFF_MAX | +165/−38 | +178/−33 | +26/−262 | +0/−26 | +1/−22 |
| SUM·NEFF_SUM | +180/−53 | +203/−39 | +23/−458 | +0/−55 | +1/−59 |
| SDIV·KNEE_SDIV | +104/−28 | +118/−25 | +15/−131 | +0/−34 | +1/−30 |
| SC_L·SC_L_COVER | +168/−399 | +207/−401 | +7/−1424 | +0/−553 | +0/−655 |
| PD·NEFF_PD | +148/−34 | +151/−28 | +21/−323 | +0/−34 | +1/−31 |
| SDE·BPI_SDIV | +215/−73 | +257/−71 | +26/−552 | +0/−76 | +1/−74 |
| MXS_L4·MA_KS_L4 | +195/−45 | +221/−42 | +26/−131 | +0/−17 | +1/−10 |
| UM_KS_L4·UM_KS_L4 | +208/−76 | +242/−80 | +30/−110 | +0/−10 | +0/−9 |
| UM_KS_L4·MA_KS_L4 | +190/−60 | +221/−52 | +24/−85 (p 3.5e−9) | +0/−7 | +1/−6 |
| IL_S_T3·MG_KS_L2 | +156/−36 | +188/−33 | +25/−67 (p 1.4e−5) | +0/−9 | +1/−7 |
| S·KNEE_TOP3 | +110/−36 | +132/−26 | +18/−27 (p 0.23) | +0/−6 | +0/−5 |
| ES·KNEE_SDIV | +97/−34 | +118/−27 | +15/−31 (p 0.026) | +0/−3 | +1/−2 |
| ES·ESTAR_BN | +7/−6 | +6/−0 | +10/−21 (p 0.071) | +0/−1 | +0/−0 |
| S·PSTAR_BN | 0 | 0 | 0 | 0 | 0 |

- PC_L·PC_L_KNEEG: +131/−41, +148/−30, +24/−177. NSUM·NEFF_NSUM: +187/−67, +220/−49, +27/−486.
- At B_N 500:
  - UM_KS_L4·MA_KS_L4 is +201/−36, +226/−34 and +13/−48 (p 7.7e−6);
  - on MuSiQue, S·KNEE_TOP3 is +6/−8 (p 0.79), ES·KNEE_SDIV +4/−10 (p 0.18) and IL_S_T3·MG_KS_L2 +12/−33 (p 0.0025).

**34.8 Universality, lossless routing and the frontier** (rules only; the 68-ranking envelope).

| B_N | best worst-cell distance to the envelope: all 8,500 rules | rounds 1–2's factorial | round 1's factorial | §33's 12 arms |
|---|---|---|---|---|
| 100 | −0.017 UM_KT_L4·MG_KT_L1 | −0.034 IL_E_SD·UF_NS_NEFF | −0.039 TOP3·NEFF_MAX (U_KNEE·NEFF_MAX ties) | −0.041 MAX·NEFF_MAX |
| 250 | −0.015 UM_KT_L4·MA_KT_L4 | −0.046 IL_E_T3·UF_SUM_NEFF | −0.052 IL_S_T3·NEFF_TOP3 | −0.067 S·NEFF_MAX |
| 500 | −0.019 UM_KS_L4·MA_KS_L4 | −0.051 PF_S_T3·NEFF_TOP3 (IL_S_T3·NEFF_TOP3 and PF_E_T3·NEFF_TOP3 tie) | −0.051 (the same arm; IL_S_T3·NEFF_TOP3 ties) | −0.069 S·NEFF_MAX |
| 1000 | −0.024 IL_S_T3·MG_KS_L2 | −0.032 IL_S_T3·PC_L_KNEEG | −0.038 IL_S_T3·NEFF_SDIV | −0.047 S·NEFF_MAX |
| 2000 | −0.033 UM_KT_L4·NEFF_SDIVS | −0.035 IL_E_SD·NEFF_SDIVS (UF_SUM_NEFF·KNEE_MAX and UF_SUM_NEFF·KNEE_PD tie) | −0.036 PF_S_T3·KNEE_SUM (IL_S_T3·KNEE_SUM ties) | −0.049 MAX·EXPH_MAX |
| 5000 | −0.034 SHF·KNEE_SDIV | −0.034 (the same arm) | −0.035 PF_S_T3·KNEE_PD (IL_S_T3·KNEE_SDIV ties) | −0.038 S·EXPH_MAX |

- **Round 3 moves the best worst-cell distance** from −0.034 / −0.046 / −0.051 / −0.032 (rounds 1–2) to −0.017 / −0.015 / −0.019 / −0.024 at B_N 100 / 250 / 500 / 1000. At 2000 and 5000 it moves by at most 0.002.
- **§33's arms against the larger envelope.** §33's 12 arms (S and the own ranking for SUM / MAX / TOP3, each with NEFF and EXPH) are −0.038 to −0.069 from it. §33 reported −0.041 at B_N 1000 against its own 7-ranking envelope.
- IL_E_T3·MG_KS_L2, the same rule on the router's ES order, is next at 1000.

**Lossless and near-lossless rules** (rules within δ of unrouted in every cell):

| B_N | 100 | 250 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|---|
| δ = 0 (at or above unrouted everywhere) | 674 | 277 | 53 | 50 | 9 | 0 |
| δ = 0.005 | 1,685 | 923 | 605 | 329 | 214 | 192 |
| δ = 0.01 | 2,231 | 1,310 | 929 | 450 | 284 | 195 |
| δ = 0.02 | 2,857 | 1,968 | 1,423 | 901 | 440 | 267 |
| δ = 0.05 | 4,002 | 3,217 | 2,664 | 2,019 | 1,397 | 743 |
| at δ = 0, the smallest largest fan-out fraction over the cells (arm) | 0.168 (UM_KS_L4·UM_KS_L4) | 0.242 (S·KNEE_PD) | 0.374 (S·EXPH_TOP3) | 0.509 (S·EXPH_SDIVS) | 0.984 (PF_S_T3·PC_L_COVER) | none |

- At δ = 0: S·KNEE_PD at B_N 250 is +0.010 / +0.013 / 0 / 0 / 0, and S·EXPH_TOP3 at 500 is +0.013 / +0.018 / 0 / 0 / +0.001.
- The smallest footprints within δ = 0.005 / 0.01 / 0.02 / 0.05:
  - at B_N 1000: S·KNEE_TOP3 (0.227), OT3·KNEE_SUM (0.197; +0.020 / +0.035 / −0.009 / −0.003 / −0.004), OT3·KNEE_NSUMS (0.160) and IL_S_T3·UM_KT_L4 (0.139);
  - at B_N 100: IL_S_T3·UM_KT_L3 (0.130, for both δ = 0.005 and 0.01), UM_KS_L4·BPI_SDIVS (0.112) and S·KNEE_FSUMFV (0.090).

**The text constraint.** Rules that keep every text cell within 0.01 of unrouted:

| B_N | 100 | 250 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|---|
| rules (of 8,500) | 2,790 | 1,492 | 1,023 | 504 | 314 | 208 |
| … and both KB cells ≥ +0.03 | 33 | 37 | 363 | 60 | 1 | 0 |
| … and both KB cells ≥ +0.05 | 0 | 12 | 3 | 0 | 0 | 0 |
| native arms (of 163) | 48 | 29 | 17 | 4 | 3 | 3 |

- At B_N 1000 the 4 native arms are ES·KNEE_SDIV and three arms that contact nearly every shard with evidence: ES·NEV, PC_L·PC_L_COVER and PC_LS·PC_LS_COVER.

**The frontier.** For each tolerance t on every text cell, the largest KB gain (the smaller of the two MetaQA differences) and the arm that reaches it:

| B_N | t = 0 | 0.005 | 0.01 | 0.02 | 0.03 | 0.05 | 0.1 |
|---|---|---|---|---|---|---|---|
| 100 | +0.033 UM_KS_L4·MA_KT_L4 | +0.036 UM_KT_L4·MG_KT_L1 | +0.036 (same) | +0.043 MXS_L4·MA_KT_L4 | +0.044 MXS_L4·MG_KT_L4 | +0.050 MAX·MG_KS_L4 | +0.050 (same; 4 tie) |
| 250 | +0.013 S·NEFF_SDIV | +0.063 UM_KT_L4·MA_KS_L4 | +0.066 UM_KT_L4·MA_KT_L4 | +0.066 (same) | +0.067 MXS_L4·MA_KS_L4 | +0.073 MXS_L4·UM_KT_L4 | +0.080 MAX·MG_KS_L4 |
| 500 | +0.014 S·EXPH_MAX | +0.046 IL_S_T3·MA_KS_L2 | +0.058 UM_KS_L4·MG_KS_L2 | +0.084 UM_KT_L4·MA_KS_L4 | +0.086 UM_KT_L4·MA_KT_L4 | +0.092 MXS_L4·MA_KT_L4 | +0.094 MXS_L4·MG_KT_L4 |
| 1000 | +0.015 S·EXPH_SDIVS | +0.039 PF_S_T3·KNEE_PD | +0.045 S·NEFF_SDIV | +0.057 IL_S_T3·MA_KT_L2 | +0.065 UM_KT_L4·MG_KT_L2 | +0.072 MXT_L4·MA_KT_L4 | +0.078 MXS_L4·MG_KS_L3 |
| 2000 | 0 (9 arms tie, e.g. PF_S_T3·PC_L_COVER) | +0.024 SC_L·EXPH_MAX (2 tie) | +0.030 S·KNEE_SDIVS | +0.037 PF_S_T3·KNEE_PD | +0.042 UF_NS_NEFF·MA_KS_L2 | +0.046 UM_KS_L4·UM_KT_L2 (3 tie) | +0.054 MXS_L4·MA_KS_L4 |
| 5000 | −0.002 OT3·PC_L_COVER (3 tie) | −0.001 U_NEFF·PC_L_COVER (3 tie) | −0.001 (same) | −0.001 (same) | −0.001 (same; 4 tie) | +0.006 PD·EXPH_SDIVS | +0.015 PD·KNEE_MAX |

- The B_N 1000 row in full:
  - t = 0: S·EXPH_SDIVS, +0.015 / +0.025 / +0.001 / 0 / 0 at 156.9 / 160.0 / 300.9 / 97.7 / 102.8 shards;
  - t = 0.005: PF_S_T3·KNEE_PD, +0.039 / +0.054 / −0.005 / −0.004 / −0.003;
  - t = 0.01: S·NEFF_SDIV, +0.045 / +0.056 / −0.010 / −0.001 / +0.001;
  - t = 0.02: IL_S_T3·MA_KT_L2, +0.057 / +0.069 / −0.020 / −0.005 / −0.005;
  - t = 0.03: UM_KT_L4·MG_KT_L2, +0.065 / +0.081 / −0.027 / −0.004 / −0.004;
  - t = 0.05: MXT_L4·MA_KT_L4, +0.072 / +0.086 / −0.047 / −0.007 / −0.006.

**The reverse frontier.** For a KB gain of at least g in both MetaQA cells, the best worst text cell and its arm:

| g | B_N 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| +0.03 | 0 UM_KS_L4·MA_KT_L4 (2 tie) | −0.002 UM_KT_L4·MG_KS_L2 | −0.001 S·KNEE_SDIV (3 tie) | −0.002 S·KNEE_SDIV | −0.009 S·KNEE_SDIVS |
| +0.04 | −0.018 MXS_L4·MA_KT_L4 | −0.004 UM_KS_L4·MA_KS_L4 | −0.002 UM_NA_L2·UF_SD_NEFF | −0.008 IL_S_T3·KNEE_PD | −0.028 UM_KT_L4·NEFF_SDIVS |
| +0.05 | none | −0.004 (same) | −0.009 IL_S_T3·MA_KT_L2 | −0.014 IL_S_T3·NEFF_SDIVS | −0.071 MXS_L4·MA_KS_L4 |
| +0.06 | none | −0.005 UM_KT_L4·MA_KS_L4 | −0.015 IL_S_T3·MG_KT_L2 (2 tie) | −0.021 IL_S_T3·MG_KS_L2 | −0.195 PD·MA_KS_L3 |
| +0.07 | none | −0.035 MXS_L4·UM_KS_L4 | −0.016 IL_S_T3·MA_KS_L4 | −0.047 MXT_L4·MA_KT_L4 | none |
| +0.08 | none | −0.148 MAX·UM_NA_L4 | −0.017 UM_KT_L4·MA_KS_L4 | −0.202 SDIVS·UM_NA_L4 | none |

- At B_N 5000 no rule reaches +0.03 in both KB cells.
- **Pareto at B_N 1000** (ALL against mean B_P, per cell, over the 8,500 rules):
  - MetaQA H4_SK: from 4.0 to 21.7 shards the front is SDE (SDIV, ties → ES), mostly with BPI counts: 0.520 at 7.4 (BPI_SHF), 0.543–0.582 at 8.7–15.9, and 0.585–0.586 at 19.9–21.7 (the UM_KT_L4, UM_KS_L4 and MG_KS_L4 counts). SDIVS·UM_NA_L4 gives 0.588 (1,175 rows) at 22.5 shards, above the envelope's 1,171: a query-adaptive count beats every fixed B_P there.
  - MetaQA PHG: SDIVS·UM_NA_L4 gives 0.609 at 24.3 shards, equal to the envelope (1,216 rows).
  - MuSiQue: the rules' best is 0.842 (1,683 rows; envelope 1,685, unrouted 1,679), from PF_F_T3·EXPH_SDIV at 313.3 shards. Along the front: S·BPI_SDIVS 0.734 at 37.2; OT3 with KNEE counts 0.823–0.838 at 98.4–190.1; SC_LS·KNEE_SDIV 0.841 at 198.3 (S·KNEE_SDIV the same).
  - SQuAD H4_SK: D·EXPH_TOP3 reaches unrouted's 1,999 at 71.9 shards (F·EXPH_TOP3 the same). SQuAD PHG: 1,999 at 47.5 (FSUMFV·EXPH_NSUM) and 2,000 at 59.8 (FSE·EXPH_SUM).
  - The GAP counts hold the front's smallest fan-outs (1.2–2.0 shards).
- **Split-half** (even against odd rows, each against its own half's envelope, worst-cell distance at B_N 1000):
  - the Spearman correlation over the 8,568 arms is 0.9959, but only 4 arms are in both halves' top 10: IL_S_T3·MG_KS_L2, PF_S_T3·MG_KS_L2, UM_KS_L4·MG_KS_L2 and UM_KT_L4·MG_KS_L2;
  - all 20 top-10 entries use a λ2 or λ4 count, and MG_KS_L2 appears in 11 of them.

**34.9 Deployment diagnostics** (round 3; measured, not rules).

**The regime shares** (mean [p10, p90]):

| λ | MetaQA H4_SK | MetaQA PHG | MuSiQue | SQuAD H4_SK | SQuAD PHG |
|---|---|---|---|---|---|
| λ1, the off-seed share of the IR_L1 mass | 0.76 [0.66, 0.85] | 0.76 [0.66, 0.85] | 0.82 [0.66, 0.94] | 0.75 [0.56, 0.90] | 0.75 [0.56, 0.90] |
| λ2, the non-seed share of O'[:200] | 0.34 [0.29, 0.38] | 0.34 [0.29, 0.38] | 0.35 [0.25, 0.43] | 0.30 [0.21, 0.39] | 0.30 [0.21, 0.39] |
| λ3, the share of LOC[:200] outside H_q | 0.76 [0.68, 0.84] | 0.76 [0.68, 0.84] | 0.79 [0.62, 0.92] | 0.72 [0.57, 0.87] | 0.72 [0.57, 0.87] |
| λ4 = 1 − BPI_SDIV/NSEEDP | 0.86 [0.79, 0.92] | 0.86 [0.78, 0.92] | 0.54 [0.19, 0.76] | 0.55 [0.32, 0.79] | 0.54 [0.30, 0.78] |

- λ1–λ3 do not depend on the partition.
- On MetaQA, λ4 is unrelated to λ1 (Spearman 0.003), while λ1 and λ3 are related (0.755).

**The per-query oracle between the two endpoints** (B_N = 1000; A = ES·C, B = SDE·BPI_SDIV; difference from unrouted in brackets):

| cell, C | A | B | oracle (the better per query) | A-only / B-only queries | AUC of λ1 / λ2 / λ3 / λ4 (B-only against A-only) |
|---|---|---|---|---|---|
| MetaQA H4_SK, KNEE_SDIV | 0.540 (+0.032) | 0.579 (+0.071) | 0.605 (+0.097) | 51 / 130 | 0.510 / 0.460 / 0.481 / 0.508 |
| MetaQA PHG, KNEE_SDIV | 0.554 (+0.046) | 0.601 (+0.093) | 0.630 (+0.122) | 58 / 153 | 0.638 / 0.545 / 0.532 / 0.489 |
| MuSiQue, KNEE_SDIV | 0.832 (−0.008) | 0.577 (−0.263) | 0.843 (+0.004) | 533 / 23 | 0.748 / 0.781 / 0.782 / 0.540 |
| SQuAD H4_SK, KNEE_SDIV | 0.998 (−0.002) | 0.962 | 0.998 (= A) | 73 / 0 | — |
| SQuAD PHG, KNEE_SDIV | 0.999 (−0.001) | 0.963 | 0.999 (= A) | 72 / 0 | — |

- With C = KNEE_TOP3 or NOACT_E, the MetaQA H4_SK oracle is 0.604 or 0.605 (A = 0.544 or 0.530; 49 / 119 and 51 / 149 queries), and the MuSiQue oracle is 0.837 (−0.003) or 0.818 (−0.022).
- On MetaQA, where the switch is worth +0.065 / +0.077 over A, the λ AUCs (B-only against A-only queries) are 0.460–0.510 on H4_SK and 0.489–0.638 on PHG. λ4's is 0.489–0.540 in the three cells where it is defined; SQuAD has no B-only query at B_N 1000. On MuSiQue, λ1–λ3 reach 0.748–0.782, where the switch is worth +0.012 over A (23 B-only queries).

**Coupling to B_N: ES·ESTAR_BN.**

| B_N | 100 | 250 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|---|
| − unrouted: MetaQA H4_SK / PHG; worst text cell | 0 / +0.001; −0.001 | +0.001 / +0.001; −0.001 | +0.002 / +0.002; −0.004 | +0.001 / +0.003; −0.006 | −0.002 / 0; −0.007 | 0 / 0; −0.002 |
| mean ESTAR_BN (five cells) | 56.6 / 60.1 / 40.6 / 26.3 / 28.0 | 127.3 / 134.0 / 90.1 / 56.5 / 59.8 | 209.2 / 217.0 / 153.6 / 86.3 / 91.4 | 312.9 / 318.3 / 254.8 / 121.1 / 127.8 | 415.7 / 414.8 / 433.1 / 163.7 / 169.6 | 432.0 / 432.0 / 909.1 / 198.7 / 199.7 |
| its fan-out fraction | 0.131 / 0.139 / 0.035 / 0.130 / 0.139 | 0.295 / 0.310 / 0.077 / 0.280 / 0.296 | 0.484 / 0.502 / 0.131 / 0.427 / 0.452 | 0.724 / 0.737 / 0.217 / 0.600 / 0.633 | 0.962 / 0.960 / 0.369 / 0.810 / 0.840 | 1.000 / 1.000 / 0.774 / 0.984 / 0.988 |
| share of PSTAR_BN's shards inside ESTAR_BN (float mean) | 0.981 / 0.980 / 0.975 / 0.977 / 0.976 | 0.971 / 0.969 / 0.967 / 0.969 / 0.967 | 0.896 / 0.900 / 0.884 / 0.898 / 0.901 | 0.874 / 0.884 / 0.771 / 0.825 / 0.839 | 0.978 / 0.978 / 0.784 / 0.892 / 0.908 | 1.000 / 1.000 / 0.913 / 0.989 / 0.992 |

- ESTAR_BN is the router-side estimate of PSTAR_BN (34.6's table). It stays within 0.007 of unrouted at every B_N.
- Its fan-out grows with B_N. It saves most on MuSiQue (0.217 of the shards at 1000) and least on the KB (0.724 / 0.737 at 1000; 1.000 at 5000).

**Load** (per arm, the per-shard contact counts over the population; B_N-free for the rules, PSTAR / ESTAR at B_N 1000). Each entry is the largest contact rate (the share of queries that contact the most-contacted shard) / peak ÷ mean / Gini; shards never contacted in brackets when there are any:

| arm | MetaQA H4_SK | MetaQA PHG | MuSiQue | SQuAD H4_SK | SQuAD PHG |
|---|---|---|---|---|---|
| S·KNEE_TOP3 | 0.976 / 5.4 / 0.344 | 0.972 / 5.4 / 0.357 | 0.895 / 6.7 / 0.228 | 0.459 / 2.1 / 0.188 | 0.583 / 2.6 / 0.183 |
| S·KNEE_SDIV | 0.980 / 4.6 / 0.335 | 0.979 / 4.6 / 0.350 | 0.940 / 5.6 / 0.216 | 0.601 / 2.1 / 0.170 | 0.701 / 2.3 / 0.163 |
| ES·KNEE_SDIV | 0.979 / 4.6 / 0.338 | 0.978 / 4.6 / 0.354 | 0.959 / 5.7 / 0.224 [1] | 0.578 / 2.0 / 0.165 | 0.677 / 2.3 / 0.159 |
| SDE·BPI_SDIV | 0.982 / 25.7 / 0.310 | 0.981 / 23.5 / 0.361 | 0.894 / 24.9 / 0.411 [8] | 0.445 / 3.3 / 0.314 | 0.641 / 4.3 / 0.320 |
| TOP3·BPI_TOP3 | 0.913 / 30.6 / 0.330 | 0.913 / 28.2 / 0.356 [1] | 0.691 / 42.8 / 0.472 [9] | 0.313 / 5.2 / 0.369 | 0.385 / 5.9 / 0.358 |
| MXS_L4·MA_KS_L4 | 0.986 / 15.5 / 0.324 | 0.986 / 14.6 / 0.361 | 0.956 / 9.6 / 0.295 [2] | 0.576 / 2.8 / 0.252 | 0.729 / 3.4 / 0.249 |
| UM_KS_L4·UM_KS_L4 | 0.978 / 19.7 / 0.357 | 0.975 / 18.7 / 0.390 | 0.856 / 10.0 / 0.258 [2] | 0.392 / 2.5 / 0.225 | 0.506 / 3.0 / 0.223 |
| UM_NA_L4·UM_NA_L4 | 0.978 / 18.7 / 0.359 | 0.975 / 17.3 / 0.388 | 0.788 / 23.0 / 0.340 [3] | 0.346 / 2.7 / 0.255 | 0.467 / 3.4 / 0.250 |
| S·PSTAR_BN | 0.999 / 1.4 / 0.122 | 0.999 / 1.3 / 0.130 | 0.990 / 4.4 / 0.202 | 0.983 / 1.5 / 0.110 | 0.993 / 1.4 / 0.103 |
| ES·ESTAR_BN | 1.000 / 1.4 / 0.132 | 1.000 / 1.4 / 0.141 | 0.999 / 4.6 / 0.221 [1] | 0.993 / 1.7 / 0.138 | 0.999 / 1.6 / 0.128 |

- Under every rule shown, one MetaQA shard is contacted by 0.913–0.986 of the queries.
- Of the arms shown, the BPI arms concentrate load most (peak ÷ mean 23.5–30.6 on MetaQA; 24.9 / 42.8 on MuSiQue), then the λ4 arms (14.6–19.7 on MetaQA; 9.6–23.0 on MuSiQue), then the KNEE arms (4.6–5.4; 5.6–6.7). The lossless references at B_N 1000 are the most even in every cell (1.3–4.6). Some degenerate GAP counts (not shown) concentrate load further.
- The record's load block covers 85 arms. UM_KS_L4·MA_KS_L4 and IL_S_T3·MG_KS_L2 are cross arms and are not in it.

**Router state** (static, the four families together):

| | MetaQA H4_SK | MetaQA PHG | MuSiQue | SQuAD H4_SK | SQuAD PHG |
|---|---|---|---|---|---|
| node-row entries | 457,076 | 457,076 | 7,619,440 | 2,146,578 | 2,146,578 |
| shard-sketch entries, distinct (node, shard) (share of the node-row entries) | 296,444 (0.649) | 317,093 (0.694) | 2,300,430 (0.302) | 334,630 (0.156) | 379,879 (0.177) |
| entries read per query for the 200 seeds, mean: node rows / distinct-(node, shard) sketch | 2,650.2 / 1,778.5 | 2,650.2 / 1,901.7 | 17,992.8 / 5,108.9 | 22,267.9 / 3,486.5 | 22,267.9 / 3,934.7 |
| co-partitioned share of the entries: STRUCT / KNN / NER | 0.181 / 0.371 / 0.457 | 0.168 / 0.270 / 0.348 | 0.043 / 0.554 / 0.125 | 0.120 / 0.675 / 0.203 | 0.089 / 0.636 / 0.177 |

- A sketch-only router reads 3.52–6.39× fewer entries per query than one that reads the seeds' node rows on text (MuSiQue 17,992.8 against 5,108.9; SQuAD 22,267.9 against 3,486.5 / 3,934.7) and 1.39–1.49× fewer on MetaQA.
- STRUCT and NER entries mostly cross shards (co-partitioned shares 0.043–0.181 and 0.125–0.457). KNN entries mostly stay inside a shard on text (0.554–0.675) but not on MetaQA (0.371 / 0.270). So the node-level evidence of a query lives on many shards.
- The router's own fused list O' has 2,076.7 / 7,519.9 / 5,540.9 nodes on average (MetaQA / MuSiQue / SQuAD).

**34.10 Not run, and flags.**
- **Not run:**
  - MMR, xQuAD and τ-coverage, which need a hyperparameter each. The proposal's equal-weight MMR (*"normalizing both terms and setting them equally"*) is one fixed choice of λ and was not run either.
  - The mean-gain stop, which is degenerate on non-increasing gains (34.3).
  - A sampled (CSI) router: §28 measured fixed per-shard representatives.
- **FLAGGED:**
  - PD is a second hop at shard granularity. It sets the MetaQA H4_SK envelope from B_N 1000 and PHG's at 5000 (34.6).
  - SUMH is the shard-level analogue of the target-degree term.
  - Round 3's endpoints (ES; SDE / T3E; BPI_SDIV) and its C were chosen on rounds 1–2's development results. λ4 was added after the round-3 smoke.
  - Each round was designed after reading the previous round's records, on the same population. The summary module was last modified after rounds 1–2 had been read (34.12). The view and its panel are post hoc.
  - With 8,568 arms evaluated on one population, the best arms carry a winner's curse. The split-half check (34.8) is the only guard: its halves agree on the ranking of all arms (Spearman 0.9959) but share only 4 of their top 10.
  - PSTAR_BN and ESTAR_BN are coupled to B_N; they are references, not rules. GAP is degenerate.
  - FLAT ranks are global in every router tier. Even a sketch-only rule needs the query's FLAT top-200 over the whole corpus.
  - The envelope is over 68 rankings, not §33's 7, so the distances are not comparable with §33's. §33's 12 arms are re-measured against it in 34.8. TOP2 / TOP5 / TOP10 are not in this factorial.
  - The lossless counts exclude PSTAR_BN.
  - **Scale.** The request's *"thousands"* of shards was not tested. The cells have 202–1,175 shards of about 100 nodes. More shards would need a new partitioning, and that needs a ruling (the Mt-KaHyPar contract).

**34.11 Reading** (development numbers, descriptive; the selection is the user's step).
- **The proposal's items, as measured.**
  - Item 1 (the MAX / TOP3 baseline) is §33's. It keeps the KB gain (+0.064 / +0.073–0.077 at B_N 1000) and loses 0.118–0.124 on MuSiQue.
  - Item 2 (marginal node coverage) is SUM on disjoint shards, the least universal aggregation of §33 (−0.218 on MuSiQue).
  - Items 3–4 as greedies (seed coverage, facility location) saturate: their gain-based counts are 2.7–21.9 shards on average, and every native SC and FL_L arm loses in every cell. FL_LS·FL_LS_COVER is within 0.001 on SQuAD, but −0.066 on MuSiQue and −0.094 / −0.084 on MetaQA.
    - As plain scores (SDIV; NSUM) they behave like the localised aggregations.
    - As noisy-OR coverage (PC_*) they do not stop before nearly every shard with evidence is contacted.
  - Item 5 (PD, FLAGGED as a second hop) tracks SUM / TOP3 as a rule and sets part of the MetaQA envelope as a fixed ranking.
  - The knee (*"point of maximum curvature"*) on the localised scores loses less on MuSiQue than NEFF at B_N 1000 (TOP3 −0.051 against −0.124), at a lower KB gain (0.013–0.028 less).
- **The two regimes are asymmetric.**
  - On text the envelope is at most 0.006 above unrouted, so a router can at best preserve recall. What it saves is fan-out: at B_N 1000, S comes within 0.01 of unrouted at 167 of MuSiQue's 1,175 shards and 21–24 of SQuAD's 202.
  - On the KB, routing adds recall. The envelope is +0.053 to +0.114 above unrouted for B_N ≤ 1000, at 15–31 shards. Contacting fewer shards keeps FLAT's non-answer nodes out of the served list: round 3's *"crowding relief"*.
- **Few against many: partly reproduced.**
  - The λ4 mixtures and unions on KNEE_SDIV / KNEE_TOP3 send MuSiQue queries to 3.83–4.68× as many shards as MetaQA H4_SK queries (UM_KS_L4 100.6 against 21.5; on NOACT_E 1.79–2.22×). Rounds 1–2's counts give at most 2.55×.
  - The need at B_N 1000 is 5.96× (S) to 9.06× (the envelope), and it grows with B_N on text while no rule count does.
  - So each rule fits a band of B_N:
    - the λ4 unions and mixtures come closest to the envelope at B_N 100–500;
    - the KNEE counts on the S / ES order stay near lossless on text up to B_N 1000–2000;
    - at B_N 5000 no rule within 0.03 of unrouted on text gains on both KB cells (the envelope there is +0.025 / +0.054).
- **λ4 separates the corpora, not the queries.**
  - Its cell means differ (0.86 against 0.54–0.55). Within a cell it does not rank the queries that need the KB endpoint above those that need the text endpoint (AUC 0.489–0.540).
  - The per-query oracle between the two endpoints is +0.097 / +0.122 over unrouted on MetaQA and +0.004 on MuSiQue at B_N 1000. On MetaQA the λ AUCs are 0.460–0.638.
  - λ4 reads no dataset id: each query's own evidence sets its count. Its effect is nonetheless mostly between corpora.
- **For the user** (nothing is selected, and nothing is frozen):
  - (a) **The λ4 union / mixture family** (FLAGGED: dev-selected endpoints; λ4 added after the smoke). For example, rank by the UM_KS_L4 key and contact MA_KS_L4 shards.
    - At B_N 500 this is +0.083 / +0.096 / −0.018 / −0.002 / −0.001, at 6–22% of the shards.
    - Its worst cell is −0.019 to −0.024 from the envelope at B_N 100–500. At 1000 it loses 0.031 on MuSiQue.
    - It needs the seeds' node rows.
  - (b) **Knee counts on the fused order.**
    - S·KNEE_SDIV keeps every text cell within 0.002 of unrouted up to B_N 1000 (0.008 at 2000). It gains +0.029 to +0.033 / +0.037 to +0.048 on the KB at B_N 500–2000.
    - S·KNEE_TOP3 keeps text within 0.005 up to 1000, with +0.034 to +0.037 / +0.044 to +0.053 at 500–1000.
    - Both contact 13–30% of the shards and need a global order. ES·KNEE_SDIV is the router-only version (within 0.008 up to B_N 1000).
  - (c) **IL_S_T3·MG_KS_L2.** It is the closest to the envelope at B_N 1000 (−0.024; +0.060 / +0.078 / −0.021 / −0.005 / −0.003 at 10–24% of the shards). MG_KS_L2 is the count common to both split halves' top 10.
  - (d) **The lossless references.**
    - S·EXPH_SDIVS is at or above unrouted in every cell up to B_N 1000, at 26–51% of the shards, and +0.015 / +0.025 on the KB at 1000.
    - ES·ESTAR_BN is within 0.007 at every B_N. It is coupled to B_N, and its KB gain is at most +0.003.
  - The frontier (34.8) gives the trade-off at each B_N. At B_N 1000 the largest KB gain is +0.015 at text tolerance 0, +0.039 at 0.005, +0.045 at 0.01, +0.057 at 0.02, +0.065 at 0.03 and +0.072 at 0.05.
  - Each option is one dataset-independent rule, and the choice precedes the ONE final held-out evaluation.

**34.12 Cost and hygiene.**
- **Wall-clock and memory** (one run at a time):

| run | start (local) | seconds | peak RSS (MB) |
|---|---|---|---|
| route MetaQA | 06:30:56 | 365.7 | 766.6 |
| route SQuAD | 06:37:29 | 202.8 | 498.7 |
| route MuSiQue | 06:41:01 | 717.1 | 1,109.6 |
| route2 MetaQA | 07:06:38 | 477.4 | 771.6 |
| route2 SQuAD | 07:14:51 | 279.1 | 498.7 |
| route2 MuSiQue | 07:19:42 | 847.5 | 1,144.6 |
| route3 MetaQA | 08:12:02 | 243.7 | 772.9 |
| route3 SQuAD | 08:16:19 | 163.8 | 519.2 |
| route3 MuSiQue | 08:19:16 | 555.5 | 1,239.7 |

- **Per-query latency** of the harness (ms; MetaQA / MuSiQue / SQuAD; PHG in brackets where it differs):
  - round 1:
    - node score + LOC order 1.4 / 5.0 / 2.8; fused order 8.0 / 28.5 / 4.0;
    - the coverage greedies 30.8 (28.8) / 66.9 / 16.8 (15.7);
    - the semantic rankings 12.6 / 44.8 / 5.5; the localised 12.5 / 47.0 / 7.4; PD 0.3 / 1.0 / 0.1;
    - FLAT on MetaQA / MuSiQue / SQuAD: products 5.5 / 32.0 / 6.2 ms plus RRF 23.1 / 74.5 / 10.5 ms.
  - round 2: the noisy-OR greedies 69.8 (64.9) / 184.1 / 35.7 (34.9); orders, counts and unions 2.9 / 4.1 / 2.3.
  - round 3: the λ shares 1.0 / 1.1 / 1.0; the mixtures 2.4 / 4.65 / 1.8; ESTAR and load 2.4 / 2.7 / 2.1; fused order 10.8 / 36.7 / 5.5.
  - The static families and router state were built in 0.5 / 10.5 / 2.7 s (round 1), 0.5 / 11.7 / 3.3 s (round 2) and 1.8 / 22.9 / 5.2 s (round 3).
- **Host at the starts:** 0.63–1.51 GiB available of 15.69 GiB, and 9.58–11.78 GiB of swap in use. Foreign pid 12096 (5,551–9,007 MB RSS) ran throughout, untouched.
- **Population:** the development population c7f70806… (§31). Nothing held out was read.
- **Nothing under `data/` was written. No pinned module or CONTRACT_FILE was edited.**
  - Each record pins its harness, `_l1d_arms.py` (f5f52b79…) and `_l1d_lib.py` (dafe39b2…).
  - Through `structures.imports`, they also pin `_l1d_node1h.py` (8265d6bc…), `_l1d_edgediag.py` (874bfc49…) and `_l1d_adaptbp.py` (984e3aef…). Round 2 adds `_l1d_route.py`; round 3 adds `_l1d_route.py` and `_l1d_route2.py`.
  - The round-1 records assert identity with the node1h and adaptbp records (`structures.reproduces`). Rounds 2 and 3 pin their predecessors' records by sha (`structures.extends`).
- **The summary module** `_l1d_route_summary.py` (6fd17ca9…) is new.
  - It was last modified at 08:07:25 local, after rounds 1–2's records had been read and before the round-3 runs (08:12). It was not modified afterwards.
  - It merges the nine records, asserts their json and npz shas and the reused identities, and wrote `route_SUMMARY__v1.{json,npz}` at 08:31 local.
- **The view module** `_l1d_route_view.py` (0fdbd3f9…) is new and post hoc (08:42:58 local).
  - It writes `route_VIEW__v1.json` (d549cb76…, 08:43:51) with the panel, the paired tests and the few-against-many counts.
  - It pins the summary's sha.
- **Rounding.**
  - Every table of 34.3–34.9 was recomputed by read-only scripts outside the repository from the exact query counts in the summary npz: `ALLN__<cell>` [ranking, count, B_N] and `BPSUM__<cell>` [count, B_N]. The summary's exact-count blocks supply the rest.
  - Shares are shown to 3 decimals, means to 1 and ratios to 2, with halves rounded away from zero. An exact zero is written 0.
  - The load figures come from the integer contact profile. The ESTAR-contains-PSTAR share and the λ statistics are float summaries.
  - The λ table is shown to 2 decimals (the stored two-decimal means and float quantiles). Latencies are shown to 1 decimal from the records' 3-decimal means, except 4.65: the stored 4.650 does not decide its 1-decimal rounding. S's LO means and the gold-shard p90s are the summary's stored values.
  - The frontier, the δ-lossless counts, the text-constraint counts, the Pareto points and the ties are in no record as lists. They come from the same exact counts.

## 35. SCALE: does IR_L1 + ES·KNEE_SDIV scale with the number of partitions? (2026-09-29 local) (`_l1d_scale_parts.py` (shard maps), `_l1d_scale.py`, summary `_l1d_scale_summary.py`, on the shared runner `_l1d_arms.py`; records `results/L1_DEV/scale_{metaqa,musique,squad}__v1.{json,npz}`, `scale_SUMMARY__v1.json`; shard maps and build records `results/L1_COVPART/parts/<ds>__H4_SK_k<K>*`; STATUS = DEVELOPMENT: descriptive numbers, no pre-registration, no verdicts, no adoption)

**35.1 The direction (2026-09-29 local).** After §34 the user pasted a second advisor text without comment, then wrote *"Try again"*. The text is taken as the user's direction. Its operative sentences, verbatim:
- *"This is a good stopping point for the non-parametric search. The 8,500-rule sweep has told us something pretty definitive: there is no magic parameter-free rule that simultaneously gives MetaQA's aggressive shard compression and MuSiQue's near-lossless distributed coverage. The remaining trade-off is structural, not because we haven't tried enough formulas."*
- The candidate carried forward is §34.11 (b)'s router-only arm: *"For the architecture you actually want, I would carry forward: ES·KNEE_SDIV rather than `S·KNEE_SDIV`."* *"`S·KNEE_SDIV` is slightly stronger, but it assumes access to the global fused node ordering. `ES·KNEE_SDIV` can operate from router-side information alone … That distinction becomes crucial once partitions physically live on different machines."*
- *"So the canonical non-learning L1 would become: L1a: IR_L1 one-hop node localization followed by L1b: ES partition ranking + KNEE_SDIV fan-out. No XGBoost. No learned parameters. No dataset identity."*
- The other finalists are set aside: λ4, because *"its endpoints came from earlier dev results and λ4 itself was introduced after looking at a smoke run"* and it *"still loses about 0.031 on MuSiQue at B_N=1000"*; IL_S_T3·MG_KS_L2, because *"only 4 of the top-10 rules overlap between halves"*; S·EXPH_SDIVS, because it *"contacts 26–51% of all shards"* and *"preserves retrieval by largely giving up the routing objective."*
- *"But I would NOT go to the final held-out evaluation yet"*. The open question: *"Does this architecture actually scale as the number of partitions grows?"* §32–§34 cover only 202–1,175 shards. *"That should be the next experiment. Not another 8,000 routing rules."*
- The experiment: *"Take the same development datasets and construct multiple partition granularities, for example: K∈{100,250,500,1000,2000,5000} where feasible. Keep the algorithm identical: IR_L1 + ES·KNEE_SDIV."* Measure R_ALL(B_N, K), the absolute fan-out B_P(q, K) and the relative fan-out ρ(q, K) = B_P(q, K)/K, *"perhaps the most important systems metric"*.
  - Sublinear growth of B_P in K is the good case and linear the weak one: *"We need to know which one is true."*
  - K\* = argmin_K [network fan-out + local scan cost + cross-shard communication] subject to R_ALL ≥ R_min.
  - *"At each K, compare at least: balanced random partition versus PHG/H4_SK structural partition."* The question it answers: *"For the same number and size of shards, does graph-aware partitioning reduce the number of machines required to recover the evidence?"*
- *"So I would stop here with heuristic development. No XGBoost yet."* *"That's the result I'd want before spending the final held-out evaluation."*
- **One correction to the text.** Its "about 0.008" is ES·KNEE_SDIV's distance to *unrouted* on MuSiQue at B_N 1000 (§34.11 (b): 1,663 against 1,679 of 2,000). The distance to S·KNEE_SDIV is larger: 18 queries at B_N 1000 (1,663 against 1,681; 0.009), 20 at 2000 (1,725 against 1,745; 0.010) and 21 at 5000 (1,783 against 1,804; 0.011). On the other four cells ES is within 3 queries of S (at most 0.002), and on SQuAD PHG it is 1 query ahead from B_N 250 on. (§34's exact counts, `route_SUMMARY__v1.npz` `ALLN__<cell>`; the scale records reproduce them per query on the served cells.)
- **Status.** ES·KNEE_SDIV is the carried-forward development candidate. It is not frozen and nothing held out was read; the one final held-out evaluation is still the user's step.

**35.2 Design.**
- **Fixed** (each asserted per query on the served cells; 35.3):
  - the L1a node score IR_L1, the served order O (IR_L1 FLAT+LOC, RRF K0 = 60) and the serving rule (the first B_N nodes of O inside the contacted shards);
  - the router's own order O' = RRF(H_q, LOC) (K0 = 60; ties to the seeds by FLAT rank, then LOC rank);
  - the L1b rule ES·KNEE_SDIV (§34): shards in the order of their first appearance in O' (shards without evidence last, in id order), and B_P(q) = the number of positive SDIV at or above their mean, SDIV(P) = the sum of 1/rank(s) over the seeds s ∈ H_q = FLAT[:200] with IR_L1 one-hop evidence in P (a query without evidence contacts every shard);
  - the reference arm S·KNEE_SDIV (first appearance in O, which needs the global order; the same count);
  - the population c7f70806… (MetaQA 1,998 rows, MuSiQue and SQuAD 2,000).
- **Varied: only the shard map.** A cell is (partitioner, K) with K ∈ {100, 250, 500, 1000, 2000, 5000} plus the native K = N // 100 (MetaQA 432, MuSiQue 1,175, SQuAD 202).
  - **The hypergraph at each K** is the frozen H4_SPLIT_PRESERVE rule. Its hyperedge cap is round(N/K), so it is rebuilt per K (`_l1d_scale_parts.py HG`); at the native K the rebuild's content digest equals the frozen hypergraph's (3 of 3, asserted).
  - **PHG:** the validated Zoltan-PHG substitute, `_l1c_phg.py` unchanged (NP 4, IMBALANCE_TOL 1.03, CONNECTIVITY, then PHG_NONEMPTY_REPAIR_V1). Its gate refuses a block above ceil(1.03 N/K).
  - **MTK:** the frozen Mt-KaHyPar contract (DETERMINISTIC_QUALITY, KM1, ε 0.03, seed 0, 8 threads) through the guarded WSL worker; MetaQA and SQuAD only, since MuSiQue's native run is FAILED_MEMORY_CAP.
    - The guard's cap is the contract's rule, min(6.0, available − 1.0) GB. A run killed at a cap the *host* limited (below 6.0 GB) is run again, unchanged, once the same rule gives a larger cap (`--retry-memory-cap`). Mt-KaHyPar in this preset is deterministic (both native-K rebuilds reproduce the frozen H4_SK exactly), so a retry can change only whether the run finishes, not which partition it returns. The earlier attempt's FAILED record and worker log move to `results/L1_COVPART/_history/` as `…__MTK__attempt<n>.*`, and the new record lists them with their sha256. A run killed at 6.0 GB itself is never retried.
  - **RAND0–2:** balanced random, perm = numpy default_rng(s).permutation(N) and P(perm[i]) = i mod K, so every shard holds ⌊N/K⌋ or ⌈N/K⌉ nodes.
  - At the native K the PHG and MTK cells are the served shard maps; each rebuild must equal them (asserted).
  - A cell whose partitioner fails is **absent**. It is never repaired beyond its own contract, relaxed, or replaced by another partitioner.
- **Measured per cell** (`_l1d_scale.py`):
  - R_ALL(B_N) of ES·KNEE_SDIV and S·KNEE_SDIV at B_N ∈ {100, 250, 500, 1000, 2000, 5000}, against unrouted (every shard contacted);
  - B_P(q), ρ(q) = B_P(q)/K and the local scan CMASS(q) (the nodes inside the contacted shards);
  - the fixed fan-out surfaces ALL(b, B_N) of ES and S;
  - the lossless references: NGP (the distinct gold shards, gold-dependent: the fewest shards that hold the evidence); LO (the ES position of the last gold shard); PSTAR_M (the distinct shards of O[:M]; S at PSTAR_M reproduces unrouted, asserted); ESTAR_M (the shards whose first O' position is < M; §34's router-side estimate);
  - communication proxies: NSEEDP (the distinct shards of H_q, the shards that own the seeds' rows); XC (the seeds' one-hop entries whose target lies outside the seed's shard); NEV (the shards holding evidence); the router's reads for the 200 seeds (node rows RDN against the union shard sketch RDS); the static cross-shard share of all entries of E;
  - load: the per-shard contacts of ES·KNEE_SDIV, S·KNEE_SDIV, ES|ESTAR@1000 and S|PSTAR@1000 (§34's load_stats).
- **The summary** (`_l1d_scale_summary.py`) recomputes every number from the stored per-query arrays. It gives:
  - the K-curves per cell, with RAND as the mean over the three seeds;
  - log-log slopes over the grid K (the native K is off-grid). When a structural cell is absent, RAND is refitted over the same K;
  - structural against random at the same K (paired, descriptive McNemar);
  - the K\* cost components, with no invented weights, and the cells that meet R_ALL ≥ unrouted − tol (tol 0.01 and 0) with their non-dominated set on (mean B_P, mean CMASS, mean NSEEDP).

**35.3 Builds, identities, absent cells and retries.**
- **The shard maps** (`results/L1_COVPART/parts/`). "cap" is the hyperedge cap round(N/K); "sizes" are the smallest..largest block; "repair" counts PHG_NONEMPTY_REPAIR_V1's moves ("none": no empty block). Seconds are each partitioner's own wall-clock. MTK MB is the worker's peak RSS; "guard" is the RSS guard's cap in GB.

MetaQA (N 43,234):

| K | cap | hyperedges | pins | PHG sizes | repair | PHG s | MTK sizes | MTK km1 | MTK MB | MTK s | guard |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 432 | 86,507 | 530,717 | 1..446 | 1 | 15.0 | 5..445 | 36,566,391 | 445.6 | 24.2 | 3.69 |
| 250 | 173 | 86,617 | 530,827 | 1..178 | 2 | 11.1 | 11..178 | 39,579,639 | 772.9 | 40.5 | 4.30 |
| 432 (native) | 100 | 86,796 | 531,006 | 1..104 | 2 | 17.3 | 14..104 | 41,244,100 | 966.8 | 41.6 | 6.00 |
| 500 | 86 | 86,860 | 531,070 | 1..89 | 1 | 18.1 | 14..89 | 41,786,919 | 1,131.5 | 45.6 | 3.92 |
| 1000 | 43 | 87,403 | 531,613 | 1..45 | 5 | 13.1 | 22..45 | 44,054,647 | 2,446.3 | 83.5 | 3.56 |
| 2000 | 22 | 89,126 | 533,336 | 1..23 | 20 | 15.7 | 2..22 | 48,110,380 | 3,582.4 | 163.9 | 6.00 (retry) |
| 5000 | 9 | 103,521 | 547,731 | PARTITION_INVALID | | | 1..9 | 64,609,358 | 5,369.3 | 409.8 | 5.91 (retry) |

MuSiQue (N 117,534; PHG only):

| K | cap | hyperedges | pins | PHG sizes | repair | PHG s |
|---|---|---|---|---|---|---|
| 100 | 1,175 | 223,086 | 5,987,292 | 429..1,211 | none | 53.4 |
| 250 | 470 | 225,604 | 5,989,810 | 1..484 | 1 | 72.8 |
| 500 | 235 | 230,653 | 5,994,859 | 1..243 | 3 | 71.6 |
| 1000 | 118 | 245,105 | 6,009,311 | 1..122 | 8 | 55.8 |
| 1175 (native) | 100 | 250,732 | 6,014,938 | 1..104 | 9 | 93.4 |
| 2000 | 59 | 283,672 | 6,047,878 | 1..61 | 18 | 73.4 |
| 5000 | 24 | 410,385 | 6,174,591 | 1..25 | 53 | 120.7 |

SQuAD (N 20,233):

| K | cap | hyperedges | pins | PHG sizes | repair | PHG s | MTK sizes | MTK km1 | MTK MB | MTK s | guard |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | 202 | 32,972 | 1,583,658 | 137..208 | none | 13.2 | 101..208 | 3,809,434 | 466.5 | 21.1 | 4.18 |
| 202 (native) | 100 | 39,146 | 1,589,832 | 60..103 | none | 28.2 | 49..104 | 6,525,934 | 553.7 | 27.1 | 2.16 |
| 250 | 81 | 42,284 | 1,592,970 | 25..84 | none | 14.0 | 55..83 | 7,641,905 | 564.7 | 28.2 | 4.20 |
| 500 | 40 | 61,244 | 1,611,930 | 1..42 | 2 | 19.2 | 1..42 | 14,555,111 | 679.8 | 36.8 | 4.17 |
| 1000 | 20 | 102,018 | 1,652,704 | 1..21 | 6 | 23.3 | 1..21 | 28,392,716 | 835.6 | 44.8 | 4.07 |
| 2000 | 10 | 188,697 | 1,739,383 | 1..11 | 8 | 27.7 | 1..11 | 62,363,760 | 1,421.1 | 60.2 | 3.82 |
| 5000 | 4 | 526,539 | 2,077,225 | 1..5 | 36 | 73.8 | 1..5 | 288,175,344 | 5,025.0 | 87.0 | 5.39 (retry) |

- **The hypergraph changes with K.** The frozen rule caps a hyperedge at round(N/K) pins and splits a larger anchor set into several hyperedges. Smaller shards therefore mean more hyperedges:
  - from K 100 to K 5000, SQuAD goes from 32,972 to 526,539 hyperedges (pins 1,583,658 → 2,077,225), MetaQA from 86,507 to 103,521, and MuSiQue from 223,086 to 410,385;
  - before the cap the pins are the same at every K (MetaQA 530,676, MuSiQue 5,986,489, SQuAD 1,581,445). The growth is the rule's anchor-duplication pins (K 100 → K 5000: MetaQA 41 → 17,055, MuSiQue 803 → 188,102, SQuAD 2,213 → 495,780), and no pin of either family (STRUCT, KNN) is dropped (retention 1.0000).
  - So a cell varies the hypergraph along with the block count, under one unchanged rule.
- **Mt-KaHyPar's memory grows with K** at nearly fixed pins. MetaQA needs 445.6 MB at K 100, 966.8 at K 432, 2,446.3 at K 1000 and 5,369.3 at K 5000 (24.2 s → 409.8 s). SQuAD needs 466.5 MB → 5,025.0 MB (21.1 s → 87.0 s).
- **Identities** (asserted by the builder and by the harness):
  - At the native K, the rebuilt hypergraph's content digest equals the frozen H4_SK's (3 of 3). The PHG rebuild equals the served PHG map (3 of 3), and the Mt-KaHyPar rebuild equals the frozen H4_SK map (MetaQA, SQuAD).
  - Every shard map's sha256 equals its build record.
  - FLAT equals loc v1. The IR_L1 LOC and FLAT+LOC positions equal node1h v1 on every gold node (MetaQA 14,489, MuSiQue 4,830, SQuAD 2,000).
  - On the served cells (MetaQA MTK_k432 and PHG_k432, MuSiQue PHG_k1175, SQuAD MTK_k202 and PHG_k202), per query:
    - the ES LO / HI / PRG and the KNEE_SDIV, NSEEDP and NEV counts equal route v2;
    - the S LO / HI / PRG and PSTAR equal route v1;
    - ESTAR and the per-shard load of the four load arms equal route v3.
  - On every cell, contacting every shard reproduces unrouted (per row, per ranking), and S at PSTAR_M reproduces unrouted at every M.
  - So §34's native-K results are reproduced query by query, including the corrected ES − S gap on MuSiQue (35.1).
- **Balance.**
  - Every PHG map passes `_l1c_phg.py`'s gate (largest block ≤ ceil(1.03 N/K)), and every Mt-KaHyPar map is within its ε.
  - Neither kind is equal-sized:
    - PHG_NONEMPTY_REPAIR_V1 fills empty blocks with single nodes, so most PHG maps have a smallest block of 1. The repair moves 1–53 nodes; none are needed at MuSiQue K 100 or SQuAD K ≤ 250.
    - Mt-KaHyPar itself leaves blocks of 1 node at SQuAD K ≥ 500 and MetaQA K 5000.
  - The RAND maps hold ⌊N/K⌋ or ⌈N/K⌉ nodes. So "the same size" means the same mean and nearly the same maximum, not the same minimum.
- **Absent: PHG on MetaQA at K 5000** (`metaqa__H4_SK_k5000__PHG_con.FAILED.json`, e8a76d9a…).
  - The raw Zoltan partition has 52 empty blocks and a largest block of 10 nodes. That is above the contract bound ceil(1.03 × 43,234 / 5,000) = 9 and above Zoltan's own tolerance (1.03 × the mean = 8.906).
  - `_l1c_phg.py` refuses it as PARTITION_INVALID before its repair step.
  - The cell is absent: not relaxed, not repaired beyond the contract and not replaced.
  - The v0 builder stopped on this refusal with an assertion. v1 records it from the driver log without rerunning (35.12).
- **Not attempted: Mt-KaHyPar on MuSiQue.**
  - Its native run is FAILED_MEMORY_CAP at the contract cap of 6.0 GB (6,014,938 pins; `data/l1_canonical/musique/parts/H4_SK.FAILED.json`), and the host had less memory free than that cap.
  - The scale hypergraphs have 5,987,292–6,174,591 pins.
- **Retried: three MTK cells** (the retry rule of 35.2).
  - In the first pass, MetaQA K 2000 and K 5000 and SQuAD K 5000 were killed at caps the host limited: 2.18, 2.54 and 3.54 GB (1.0 GB below the 3.18 / 3.54 / 4.54 GB then available).
  - Rerun unchanged, one at a time, at the caps the same rule then gave (6.00 / 5.91 / 5.39 GB), all three finished. The guard's peaks were 3,663,652 / 5,497,724 / 5,144,856 kB, in 167.8 / 414.0 / 88.9 s.
  - The first attempts' FAILED records (be4935d5… / 56f99426… / 32fdd5f7…) and their empty worker logs are in `results/L1_COVPART/_history/` as `<ds>__H4_SK_k<K>__MTK__attempt1.*`. The new RUN records list them under `previous_attempts`, with the rule text.

**35.4 Fan-out: B_P, ρ and the slopes.** The table gives the mean B_P of ES·KNEE_SDIV, with [ρ] and (median / p90). RAND is the mean over the three seeds, with the per-seed means in brackets. The structural ÷ random ratio is 3 × the structural sum over the three seeds' sums of the per-query counts.

MetaQA (N 43,234; native K 432):

| K | N/K | PHG | MTK | RAND0–2 | PHG ÷ RAND | MTK ÷ RAND |
|---|---|---|---|---|---|---|
| 100 | 432.3 | 28.4 [0.284] (28.0 / 34) | 28.7 [0.287] (29.0 / 35) | 35.3 [0.353] (35.4, 35.2, 35.3) | 0.81 | 0.81 |
| 250 | 172.9 | 60.7 [0.243] (60.0 / 74) | 62.0 [0.248] (61.0 / 76) | 74.1 [0.297] (74.1, 74.3, 74.0) | 0.82 | 0.84 |
| 432 (native) | 100.1 | 91.5 [0.212] (91.0 / 112) | 92.2 [0.214] (91.0 / 113) | 114.8 [0.266] (115.1, 114.6, 114.6) | 0.80 | 0.80 |
| 500 | 86.5 | 101.1 [0.202] (100.0 / 125) | 101.4 [0.203] (100.0 / 125) | 127.8 [0.256] (127.7, 127.8, 127.7) | 0.79 | 0.79 |
| 1000 | 43.2 | 146.3 [0.146] (144.0 / 186) | 147.6 [0.148] (146.0 / 188) | 198.1 [0.198] (198.9, 197.3, 198.1) | 0.74 | 0.74 |
| 2000 | 21.6 | 187.5 [0.094] (182.0 / 249) | 188.4 [0.094] (182.0 / 251) | 263.2 [0.132] (264.2, 262.1, 263.3) | 0.71 | 0.72 |
| 5000 | 8.6 | absent | 227.6 [0.046] (216.0 / 306) | 316.2 [0.063] (316.5, 315.3, 316.9) | | 0.72 |

MuSiQue (N 117,534; native K 1175):

| K | N/K | PHG | RAND0–2 | PHG ÷ RAND |
|---|---|---|---|---|
| 100 | 1,175.3 | 31.8 [0.318] (31.0 / 40) | 45.7 [0.457] (45.7, 45.7, 45.8) | 0.70 |
| 250 | 470.1 | 67.6 [0.270] (65.0 / 90) | 98.3 [0.393] (98.2, 98.7, 98.1) | 0.69 |
| 500 | 235.1 | 114.0 [0.228] (108.0 / 162) | 173.5 [0.347] (173.3, 173.7, 173.4) | 0.66 |
| 1000 | 117.5 | 181.2 [0.181] (170.0 / 275) | 302.2 [0.302] (302.4, 302.4, 301.8) | 0.60 |
| 1175 (native) | 100.0 | 198.3 [0.169] (185.0 / 306) | 341.8 [0.291] (342.1, 342.1, 341.3) | 0.58 |
| 2000 | 58.8 | 258.9 [0.129] (240.0 / 401) | 501.9 [0.251] (502.2, 502.6, 501.0) | 0.52 |
| 5000 | 23.5 | 366.8 [0.073] (328.0 / 563) | 824.0 [0.165] (824.1, 825.2, 822.8) | 0.45 |

SQuAD (N 20,233; native K 202):

| K | N/K | PHG | MTK | RAND0–2 | PHG ÷ RAND | MTK ÷ RAND |
|---|---|---|---|---|---|---|
| 100 | 202.3 | 35.0 [0.350] (34.0 / 44) | 33.0 [0.330] (32.0 / 43) | 47.9 [0.479] (47.9, 47.7, 48.0) | 0.73 | 0.69 |
| 202 (native) | 100.2 | 60.4 [0.299] (58.0 / 78) | 58.6 [0.290] (57.0 / 76) | 85.3 [0.422] (85.3, 85.0, 85.6) | 0.71 | 0.69 |
| 250 | 80.9 | 71.6 [0.286] (69.0 / 94) | 68.1 [0.273] (66.0 / 91) | 101.8 [0.407] (101.9, 101.7, 101.8) | 0.70 | 0.67 |
| 500 | 40.5 | 114.8 [0.230] (112.0 / 162) | 111.1 [0.222] (109.0 / 152) | 178.6 [0.357] (178.0, 178.8, 179.0) | 0.64 | 0.62 |
| 1000 | 20.2 | 177.2 [0.177] (175.0 / 240) | 171.8 [0.172] (169.5 / 232) | 302.1 [0.302] (301.4, 302.6, 302.4) | 0.59 | 0.57 |
| 2000 | 10.1 | 256.3 [0.128] (251.0 / 354) | 247.9 [0.124] (241.5 / 342) | 488.1 [0.244] (487.3, 488.7, 488.3) | 0.53 | 0.51 |
| 5000 | 4.0 | 421.2 [0.084] (408.0 / 595) | 400.6 [0.080] (389.0 / 563) | 765.3 [0.153] (764.5, 766.9, 764.6) | 0.55 | 0.52 |

The slopes table gives log-log least-squares slopes over the grid K (the native K is off-grid), with (×) = y(5000) ÷ y(500) against K's own 10×. MetaQA PHG is fitted over K ≤ 2000, and RAND is refitted over the same K. "need" is the smallest fixed B_P of the ES surface within 0.01 of unrouted at B_N 1000, listed per grid K (RAND: the mean over the seeds).

| dataset | family | K used | B_P | NGP | LO (ES) | PSTAR@1000 | ESTAR@1000 | NSEEDP | CMASS | need@1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA | PHG | 100–2000 | 0.64 | 0.10 | 0.90 | 0.59 | 0.58 | 0.25 | −0.36 | 0.25 (28 / 28 / 32 / 42 / 59) |
| MetaQA | RAND on PHG's K | 100–2000 | 0.68 | 0.05 | 0.87 | 0.70 | 0.70 | 0.26 | −0.32 | 0.68 (92 / 195 / 320.67 / 518.33 / 692.67) |
| MetaQA | MTK | all six | 0.53 (×2.24) | 0.11 (×1.25) | 0.99 (×10.57) | 0.48 (×1.90) | 0.47 (×1.85) | 0.19 (×1.25) | −0.47 (×0.23) | 0.60 (25 / 26 / 29 / 43 / 94 / 261) |
| MetaQA | RAND | all six | 0.57 (×2.48) | 0.03 (×1.03) | 0.88 (×7.81) | 0.57 (×2.11) | 0.57 (×2.11) | 0.20 (×1.19) | −0.43 (×0.25) | 0.57 (92 / 195 / 320.67 / 518.33 / 692.67 / 795.67) |
| MuSiQue | PHG | all six | 0.63 (×3.22) | 0.03 (×1.05) | 0.80 (×6.86) | 0.45 (×2.34) | 0.40 (×2.02) | 0.32 (×1.85) | −0.37 (×0.32) | 0.71 (33 / 47 / 79 / 166 / 269 / 437) |
| MuSiQue | RAND | all six | 0.75 (×4.75) | 0.00 (×1.00) | 0.69 (×5.16) | 0.56 (×2.10) | 0.56 (×2.10) | 0.20 (×1.19) | −0.25 (×0.48) | 0.75 (86 / 192.33 / 349 / 605 / 920.33 / 1608.67) |
| SQuAD | PHG | all six | 0.63 (×3.67) | 0 (×1.00) | 0.41 (×3.03) | 0.50 (×2.57) | 0.45 (×2.49) | 0.32 (×1.78) | −0.36 (×0.37) | 0.38 (18 / 27 / 35 / 46 / 63 / 77) |
| SQuAD | MTK | all six | 0.63 (×3.61) | 0 (×1.00) | 0.43 (×3.11) | 0.50 (×2.56) | 0.46 (×2.54) | 0.33 (×1.80) | −0.34 (×0.40) | 0.39 (16 / 24 / 37 / 46 / 55 / 72) |
| SQuAD | RAND | all six | 0.72 (×4.29) | 0 (×1.00) | 0.17 (×1.52) | 0.57 (×2.13) | 0.57 (×2.13) | 0.20 (×1.19) | −0.28 (×0.43) | 0.21 (47 / 73.67 / 92.67 / 101 / 104.33 / 111.67) |

(MuSiQue RAND's NGP slope, 0.00, is the summary's stored value: a nonzero slope that rounds to zero.)

- **B_P grows sublinearly in K on every map.**
  - Slopes: 0.53–0.64 on the structural maps and 0.57–0.75 on the random ones (0.68 on MetaQA PHG's K range).
  - From K 500 to K 5000 (10×), the structural B_P grows 2.24× (MetaQA MTK) to 3.67× (SQuAD PHG); the random one grows 2.48× to 4.75×.
- **ρ falls as K grows.**
  - Structural: from 0.284–0.350 at K 100 to 0.046–0.084 at K 5000 (MetaQA PHG: 0.094 at K 2000).
  - Random: from 0.353–0.479 to 0.063–0.165.
- **The local scan falls with it.**
  - The mean scanned share CMASS/N goes from 0.287–0.351 at K 100 to 0.046–0.090 at K 5000 on the structural maps (CMASS slopes −0.34 to −0.47).
  - On the random maps (the mean over the seeds) it goes from 0.353–0.479 to 0.063–0.165.
- **Structural maps contact fewer shards than random ones at the same K.**
  - Structural ÷ random: 0.71–0.84 on MetaQA, 0.45–0.70 on MuSiQue and 0.51–0.73 on SQuAD.
  - The ratio is lower at large K: MetaQA 0.81 at K 100 against 0.71–0.72 at K 2000–5000, MuSiQue 0.70 against 0.45, SQuAD 0.69–0.73 against 0.51–0.53 at K 2000 (0.52–0.55 at K 5000).
- **The tail.** The largest B_P of any query is 48–89 at K 100 (0.480–0.890 of the shards) and 1,300 / 3,092 / 1,422 / 1,354 at K 5000 (MetaQA MTK, MuSiQue PHG, SQuAD PHG / MTK; 0.260 / 0.618 / 0.284 / 0.271).
- **The evidence stays in a few shards, but it sits later in the router's order.**
  - The gold shards (NGP) barely grow with K (slopes 0–0.11).
  - The ES position of the last gold shard (LO) grows almost linearly on the multi-hop corpora: slopes 0.80–0.99 structural and 0.69–0.88 random. On SQuAD it grows with slope 0.41 / 0.43 (0.17 random).
  - Taken together, the shards holding the evidence stay few (NGP), while the rest of the corpus is cut into more shards, and more of them land ahead of the last gold one.

**35.5 Retrieval R_ALL(B_N, K).** The table gives ES·KNEE_SDIV − unrouted at B_N 100 / 500 / 1000 / 2000 / 5000, with (ES − S at 1000). RAND is the mean over the seeds, with each seed's Δ at 1000 in brackets. Unrouted serves ALL gold for:
- MetaQA 763 / 855 / 919 / 1,015 / 1,110 / 1,271 of 1,998 rows;
- MuSiQue 1,279 / 1,483 / 1,586 / 1,679 / 1,760 / 1,855 of 2,000;
- SQuAD 1,971 / 1,991 / 1,994 / 1,999 / 2,000 / 2,000 of 2,000

(at B_N 100 / 250 / 500 / 1000 / 2000 / 5000).

| MetaQA K | PHG (ES − S @1000) | MTK (ES − S @1000) | RAND mean (seeds @1000) |
|---|---|---|---|
| 100 | −0.029 / −0.010 / −0.015 / −0.014 / −0.046 (−0.004) | −0.019 / +0.002 / +0.002 / +0.004 / −0.030 (−0.005) | −0.101 / −0.155 / −0.197 / −0.238 / −0.308 (−0.206, −0.194, −0.190) |
| 250 | −0.001 / +0.034 / +0.039 / +0.044 / +0.016 (+0.001) | 0 / +0.032 / +0.039 / +0.037 / −0.002 (+0.001) | −0.006 / −0.058 / −0.100 / −0.141 / −0.214 (−0.099, −0.095, −0.106) |
| 432 (native) | −0.001 / +0.036 / +0.046 / +0.048 / +0.014 (−0.001) | 0 / +0.029 / +0.032 / +0.029 / −0.012 (−0.002) | −0.003 / −0.049 / −0.092 / −0.133 / −0.202 (−0.095, −0.086, −0.096) |
| 500 | 0 / +0.038 / +0.047 / +0.056 / +0.017 (−0.001) | 0 / +0.032 / +0.036 / +0.029 / −0.014 (0) | −0.002 / −0.043 / −0.086 / −0.129 / −0.201 (−0.085, −0.083, −0.091) |
| 1000 | 0 / +0.033 / +0.044 / +0.041 / +0.005 (−0.002) | 0 / +0.023 / +0.012 / −0.006 / −0.058 (−0.001) | −0.001 / −0.030 / −0.075 / −0.119 / −0.193 (−0.070, −0.078, −0.077) |
| 2000 | 0 / +0.031 / +0.041 / +0.038 / −0.013 (−0.004) | 0 / +0.013 / −0.007 / −0.029 / −0.095 (−0.002) | −0.001 / −0.025 / −0.070 / −0.115 / −0.192 (−0.070, −0.070, −0.072) |
| 5000 | absent | 0 / +0.003 / −0.014 / −0.047 / −0.125 (−0.002) | −0.001 / −0.024 / −0.069 / −0.115 / −0.196 (−0.071, −0.068, −0.068) |

| MuSiQue K | PHG (ES − S @1000) | RAND mean (seeds @1000) |
|---|---|---|
| 100 | +0.001 / −0.004 / −0.011 / −0.018 / −0.033 (+0.001) | −0.024 / −0.116 / −0.148 / −0.176 / −0.210 (−0.152, −0.136, −0.156) |
| 250 | 0 / +0.002 / −0.002 / −0.007 / −0.025 (−0.001) | −0.004 / −0.062 / −0.093 / −0.123 / −0.160 (−0.096, −0.100, −0.083) |
| 500 | 0 / −0.002 / −0.006 / −0.014 / −0.031 (−0.006) | −0.002 / −0.030 / −0.061 / −0.092 / −0.128 (−0.065, −0.061, −0.059) |
| 1000 | 0 / −0.003 / −0.006 / −0.014 / −0.028 (−0.004) | 0 / −0.017 / −0.043 / −0.074 / −0.115 (−0.048, −0.046, −0.036) |
| 1175 (native) | 0 / −0.003 / −0.008 / −0.018 / −0.036 (−0.009) | 0 / −0.020 / −0.045 / −0.073 / −0.112 (−0.043, −0.045, −0.048) |
| 2000 | 0 / −0.004 / −0.010 / −0.024 / −0.043 (−0.012) | 0 / −0.012 / −0.037 / −0.067 / −0.105 (−0.038, −0.040, −0.034) |
| 5000 | 0 / −0.003 / −0.015 / −0.036 / −0.067 (−0.013) | 0 / −0.008 / −0.027 / −0.058 / −0.101 (−0.023, −0.029, −0.030) |

| SQuAD K | PHG (ES − S @1000) | MTK (ES − S @1000) | RAND mean (seeds @1000) |
|---|---|---|---|
| 100 | 0 / −0.001 / −0.002 / −0.003 / −0.003 (0) | 0 / −0.001 / −0.002 / −0.002 / −0.002 (+0.001) | −0.001 / −0.007 / −0.009 / −0.010 / −0.010 (−0.012, −0.008, −0.008) |
| 202 (native) | 0 / 0 / −0.001 / −0.001 / −0.001 (+0.001) | 0 / +0.001 / −0.002 / −0.002 / −0.002 (0) | −0.000 / −0.004 / −0.006 / −0.007 / −0.007 (−0.005, −0.009, −0.005) |
| 250 | 0 / +0.001 / −0.001 / −0.002 / −0.002 (−0.001) | 0 / −0.001 / −0.002 / −0.003 / −0.003 (−0.001) | −0.001 / −0.004 / −0.006 / −0.007 / −0.007 (−0.005, −0.006, −0.008) |
| 500 | 0 / +0.001 / −0.001 / −0.001 / −0.001 (0) | 0 / +0.001 / −0.001 / −0.001 / −0.001 (+0.001) | 0 / −0.001 / −0.003 / −0.004 / −0.004 (−0.003, −0.004, −0.004) |
| 1000 | 0 / 0 / −0.002 / −0.003 / −0.003 (0) | 0 / +0.001 / −0.002 / −0.002 / −0.002 (0) | 0 / +0.000 / −0.002 / −0.002 / −0.002 (−0.002, −0.003, −0.002) |
| 2000 | 0 / 0 / −0.002 / −0.003 / −0.003 (−0.001) | 0 / +0.001 / −0.002 / −0.002 / −0.002 (+0.001) | 0 / +0.001 / −0.001 / −0.002 / −0.002 (−0.001, −0.002, −0.002) |
| 5000 | 0 / 0 / −0.002 / −0.003 / −0.003 (−0.002) | 0 / 0 / −0.002 / −0.003 / −0.003 (−0.002) | 0 / +0.000 / −0.002 / −0.002 / −0.002 (−0.002, −0.001, −0.002) |

(The RAND means are the summary's stored strings. "−0.000" and "+0.000" are nonzero means over the seeds that round to zero.)

**Where ES·KNEE_SDIV loses** (shares of the rows):
- "reach" is a count below LO: the rule stops before the last gold shard. It does not depend on B_N, and it includes rows that no count serves at B_N 1000.
- "crowding" is LO ≤ HI(1000) < count: every gold shard is reached, but the rule contacts shards past HI(1000), and their nodes push gold out of the first 1,000.

| MetaQA K | PHG reach / crowding | MTK reach / crowding | RAND0 reach / crowding |
|---|---|---|---|
| 100 | 0.348 / 0.054 | 0.331 / 0.071 | 0.667 / 0.011 |
| 250 | 0.299 / 0.082 | 0.317 / 0.084 | 0.568 / 0.009 |
| 432 | 0.304 / 0.080 | 0.342 / 0.070 | 0.565 / 0.008 |
| 500 | 0.304 / 0.085 | 0.343 / 0.071 | 0.558 / 0.009 |
| 1000 | 0.335 / 0.066 | 0.412 / 0.048 | 0.551 / 0.005 |
| 2000 | 0.369 / 0.049 | 0.459 / 0.036 | 0.558 / 0.002 |
| 5000 | absent | 0.489 / 0.016 | 0.561 / 0.002 |

| MuSiQue K | PHG reach / crowding | RAND0 reach / crowding |
|---|---|---|
| 100 | 0.070 / 0.028 | 0.273 / 0.012 |
| 250 | 0.067 / 0.037 | 0.229 / 0.007 |
| 500 | 0.075 / 0.034 | 0.200 / 0.009 |
| 1000 | 0.080 / 0.035 | 0.188 / 0.008 |
| 1175 | 0.092 / 0.033 | 0.178 / 0.010 |
| 2000 | 0.103 / 0.034 | 0.174 / 0.009 |
| 5000 | 0.134 / 0.019 | 0.168 / 0.004 |

On SQuAD, reach is 0.001–0.003 on the structural maps and 0.001–0.013 on RAND0; crowding is 0 on every map.

**Per hop at B_N 1000** (counts that serve ALL gold; RAND = seeds 0, 1, 2).

MetaQA (666 rows per hop). Unrouted serves 666 / 187 / 162:

| K | hop 1: PHG / MTK / RAND | hop 2: PHG / MTK / RAND | hop 3: PHG / MTK / RAND |
|---|---|---|---|
| 100 | 588 / 600 / 456, 471, 483 | 192 / 203 / 82, 80, 83 | 206 / 215 / 66, 76, 69 |
| 250 | 656 / 660 / 645, 643, 643 | 215 / 212 / 92, 97, 82 | 221 / 220 / 80, 85, 78 |
| 432 | 662 / 664 / 656, 659, 657 | 218 / 211 / 87, 97, 85 | 226 / 203 / 83, 88, 82 |
| 500 | 663 / 664 / 659, 656, 659 | 218 / 213 / 95, 95, 82 | 228 / 210 / 91, 99, 93 |
| 1000 | 664 / 664 / 662, 662, 662 | 215 / 200 / 118, 109, 107 | 224 / 174 / 96, 89, 93 |
| 2000 | 665 / 666 / 664, 664, 664 | 213 / 179 / 113, 119, 115 | 219 / 156 / 98, 93, 93 |
| 5000 | absent / 666 / 665, 665, 665 | absent / 177 / 112, 120, 120 | absent / 144 / 96, 95, 94 |

MuSiQue (1,364 / 442 / 194 rows). Unrouted serves 1,244 / 336 / 99:

| K | hop 2: PHG / RAND | hop 3: PHG / RAND | hop 4: PHG / RAND |
|---|---|---|---|
| 100 | 1,231 / 1,083, 1,107, 1,087 | 333 / 240, 248, 234 | 93 / 52, 53, 47 |
| 250 | 1,239 / 1,153, 1,166, 1,169 | 335 / 275, 260, 276 | 101 / 60, 54, 69 |
| 500 | 1,238 / 1,187, 1,197, 1,200 | 335 / 285, 286, 283 | 95 / 77, 75, 79 |
| 1000 | 1,235 / 1,206, 1,202, 1,214 | 334 / 294, 300, 309 | 99 / 84, 85, 85 |
| 1175 | 1,235 / 1,210, 1,211, 1,209 | 335 / 304, 300, 297 | 93 / 80, 78, 77 |
| 2000 | 1,235 / 1,213, 1,216, 1,207 | 330 / 306, 302, 316 | 94 / 85, 82, 89 |
| 5000 | 1,228 / 1,226, 1,217, 1,219 | 328 / 319, 317, 314 | 93 / 88, 88, 86 |

- **MetaQA: the routing gain holds on PHG and decays on MTK.**
  - At B_N 500–2000, PHG stays above unrouted from K 250 to K 2000: +0.031 to +0.056, and +0.039 to +0.047 at B_N 1000.
  - MTK's gain at B_N 1000 falls from K 250 on: +0.039 (K 250), +0.036 (K 500), +0.012 (K 1000), −0.007 (K 2000), −0.014 (K 5000).
  - At K 100 both lose at B_N 1000 or give nothing (PHG −0.015, MTK +0.002).
  - The gain is in hops 2–3. PHG serves 213–218 (hop 2) and 219–228 (hop 3) against unrouted's 187 / 162 for K 250–2000. MTK's hop 3 falls from 220 (K 250) to 144 (K 5000).
  - At B_N 5000 most cells fall below unrouted; PHG K 250–1000 stays at +0.005 to +0.017.
- **MuSiQue: recall erodes as K grows.**
  - At B_N ≤ 500 (100 / 250 / 500), PHG is within 0.004 of unrouted at every K.
  - At B_N 1000 it is −0.002 at K 250, −0.006 at K 500 and 1000, −0.008 at K 1175, −0.010 at K 2000 and −0.015 at K 5000 (−0.011 at K 100).
  - At B_N 5000 it is −0.025 at K 250 and −0.067 at K 5000 (−0.028 to −0.043 between them, −0.033 at K 100).
  - Per hop, PHG is at most 16 queries below unrouted (hop 2 at K 5000: 1,228 against 1,244); hop 3 at most 8 and hop 4 at most 6.
- **SQuAD: within 0.003 of unrouted** on every structural map at every K and B_N. The RAND mean is within 0.010 (seed 0 at K 100 and B_N 1000: −0.012), and within 0.002 from K 1000 on.
- **The losses shift from crowding to reach as K grows.**
  - MetaQA MTK: reach 0.317 → 0.489 and crowding 0.084 → 0.016 from K 250 to K 5000.
  - MuSiQue PHG: reach 0.067 → 0.134 and crowding 0.037 → 0.019.
  - The random maps lose almost only to reach (MetaQA 0.551–0.667, MuSiQue 0.168–0.273; crowding ≤ 0.012).
- **The router-only ranking costs little except on MuSiQue at large K.**
  - ES − S is within 0.005 on MetaQA and within 0.002 on SQuAD, at every K and B_N.
  - On MuSiQue it is within 0.003 at K ≤ 250. At B_N 1000 it is −0.006 at K 500, −0.004 at K 1000, −0.009 at the native K (the 18 queries of 35.1), −0.012 at K 2000 and −0.013 at K 5000; at B_N 5000, −0.011 at the native K, −0.017 at K 2000 and −0.022 at K 5000.
- **Random maps lose on both multi-hop sets at every K**, and by less at larger K. At B_N 1000:
  - MetaQA: −0.197 (K 100) to −0.069 (K 5000);
  - MuSiQue: −0.148 to −0.027.
  - The random maps' hop-1 recall on MetaQA at K 100 is 456–483 of 666.

**35.6 How many shards does the evidence need?** At B_N 1000: NGP, the shards holding gold; LO, the ES position of the last gold shard; PSTAR [÷K], the distinct shards of O[:1000] (S at PSTAR reproduces unrouted per query); ESTAR, its router-side estimate, with (ES|ESTAR − unrouted); need, the smallest fixed B_P of the ES surface within 0.01 of unrouted. RAND is the mean over the seeds, with each seed's need.

MetaQA:

| K | PHG NGP / LO / PSTAR [÷K] / ESTAR (Δ) / need | MTK NGP / LO / PSTAR [÷K] / ESTAR (Δ) / need | RAND NGP / LO / PSTAR / need (seeds 0, 1, 2) |
|---|---|---|---|
| 100 | 4.4 / 33.2 / 98.2 [0.982] / 98.1 (0) / 28 | 4.3 / 32.2 / 97.5 [0.975] / 97.0 (0) / 25 | 6.2 / 57.0 / 100.0 / 92, 94, 90 |
| 250 | 4.8 / 70.0 / 221.4 [0.886] / 221.6 (+0.001) / 28 | 4.9 / 73.2 / 221.0 [0.884] / 220.1 (+0.001) / 26 | 6.8 / 125.0 / 245.6 / 193, 199, 193 |
| 432 (native) | 5.1 / 112.8 / 321.8 [0.745] / 318.3 (+0.003) / 31 | 5.1 / 123.4 / 316.5 [0.733] / 312.9 (+0.001) / 29 | 7.0 / 201.4 / 391.2 / 308, 314, 298 |
| 500 | 5.2 / 129.8 / 352.5 [0.705] / 346.7 (+0.005) / 32 | 5.2 / 138.5 / 344.6 [0.689] / 341.2 (+0.005) / 29 | 7.0 / 228.6 / 433.6 / 332, 328, 302 |
| 1000 | 5.5 / 254.8 / 476.8 [0.477] / 467.6 (+0.009) / 42 | 5.7 / 285.9 / 454.8 [0.455] / 443.1 (+0.007) / 43 | 7.1 / 421.0 / 635.3 / 492, 530, 533 |
| 2000 | 5.8 / 490.8 / 572.6 [0.286] / 557.1 (+0.014) / 59 | 6.2 / 600.3 / 553.7 [0.277] / 535.9 (+0.005) / 94 | 7.2 / 775.4 / 793.8 / 728, 651, 699 |
| 5000 | absent | 6.5 / 1,463.5 / 656.1 [0.131] / 631.2 (+0.007) / 261 | 7.2 / 1,785.8 / 915.7 / 810, 770, 807 |

MuSiQue:

| K | PHG NGP / LO / PSTAR [÷K] / ESTAR (Δ) / need | RAND NGP / LO / PSTAR / need (seeds 0, 1, 2) |
|---|---|---|
| 100 | 2.0 / 10.1 / 73.6 [0.736] / 76.7 (−0.002) / 33 | 2.4 / 32.9 / 100.0 / 88, 83, 87 |
| 250 | 2.1 / 18.4 / 130.1 [0.520] / 134.7 (0) / 47 | 2.4 / 60.2 / 245.5 / 184, 211, 182 |
| 500 | 2.2 / 33.3 / 183.1 [0.366] / 185.4 (−0.001) / 79 | 2.4 / 95.4 / 432.8 / 362, 323, 362 |
| 1000 | 2.2 / 55.2 / 247.0 [0.247] / 240.8 (−0.004) / 166 | 2.4 / 155.8 / 633.8 / 586, 655, 574 |
| 1175 (native) | 2.3 / 67.5 / 263.9 [0.225] / 254.8 (−0.006) / 217 | 2.4 / 175.5 / 675.3 / 624, 640, 712 |
| 2000 | 2.3 / 100.9 / 319.2 [0.160] / 298.1 (−0.010) / 269 | 2.4 / 254.9 / 789.8 / 853, 1,080, 828 |
| 5000 | 2.3 / 228.7 / 428.6 [0.086] / 375.1 (−0.012) / 437 | 2.4 / 491.9 / 910.0 / 1,430, 1,503, 1,893 |

SQuAD:

| K | PHG NGP / LO / PSTAR [÷K] / ESTAR (Δ) / need | MTK NGP / LO / PSTAR [÷K] / ESTAR (Δ) / need | RAND NGP / LO / PSTAR / need (seeds 0, 1, 2) |
|---|---|---|---|
| 100 | 1.0 / 2.8 / 84.8 [0.848] / 81.7 (0) / 18 | 1.0 / 2.4 / 82.2 [0.822] / 78.3 (0) / 16 | 1.0 / 8.3 / 100.0 / 52, 43, 46 |
| 202 (native) | 1.0 / 3.2 / 138.4 [0.685] / 127.8 (0) / 24 | 1.0 / 3.1 / 132.6 [0.656] / 121.1 (−0.001) / 21 | 1.0 / 9.6 / 200.8 / 70, 76, 73 |
| 250 | 1.0 / 3.5 / 158.5 [0.634] / 144.5 (−0.001) / 27 | 1.0 / 3.1 / 150.6 [0.603] / 134.7 (0) / 24 | 1.0 / 9.9 / 245.8 / 70, 71, 80 |
| 500 | 1.0 / 4.2 / 233.8 [0.468] / 199.1 (0) / 35 | 1.0 / 4.0 / 224.9 [0.450] / 187.7 (0) / 37 | 1.0 / 10.8 / 435.5 / 93, 95, 90 |
| 1000 | 1.0 / 6.4 / 334.7 [0.335] / 268.7 (−0.001) / 46 | 1.0 / 5.8 / 324.7 [0.325] / 258.5 (−0.001) / 46 | 1.0 / 12.0 / 640.7 / 100, 104, 99 |
| 2000 | 1.0 / 9.0 / 444.7 [0.222] / 352.6 (−0.003) / 63 | 1.0 / 7.7 / 422.3 [0.211] / 334.7 (−0.002) / 55 | 1.0 / 13.1 / 801.4 / 103, 106, 104 |
| 5000 | 1.0 / 12.9 / 600.7 [0.120] / 495.9 (−0.003) / 77 | 1.0 / 12.5 / 576.3 [0.115] / 477.2 (−0.003) / 72 | 1.0 / 16.5 / 926.4 / 112, 112, 111 |

The rule against the fixed fan-outs at B_N 1000 (structural maps): the rule's mean B_P; the need (within 0.01); the smallest fixed B_P at or above unrouted; the best fixed B_P of the ES surface, with its ALL. Unrouted is 0.508 / 0.840 / 1.000. The best fixed B_P is chosen on this population, so it is a reference, not a rule.

MetaQA:

| K | PHG rule mean B_P / need / at or above / best fixed B_P (ALL) | MTK rule mean B_P / need / at or above / best fixed B_P (ALL) |
|---|---|---|
| 100 | 28.4 / 28 / 30 / 41 (0.526) | 28.7 / 25 / 26 / 35 (0.530) |
| 250 | 60.7 / 28 / 29 / 43 (0.555) | 62.0 / 26 / 27 / 46 (0.555) |
| 432 (native) | 91.5 / 31 / 33 / 53 (0.570) | 92.2 / 29 / 31 / 44 (0.558) |
| 500 | 101.1 / 32 / 33 / 55 (0.575) | 101.4 / 29 / 30 / 55 (0.559) |
| 1000 | 146.3 / 42 / 44 / 95 (0.566) | 147.6 / 43 / 50 / 97 (0.527) |
| 2000 | 187.5 / 59 / 66 / 154 (0.553) | 188.4 / 94 / 282 / 770 (0.515) |
| 5000 | absent | 227.6 / 261 / 405 / 748 (0.518) |

MuSiQue:

| K | PHG rule mean B_P / need / at or above / best fixed B_P (ALL) |
|---|---|
| 100 | 31.8 / 33 / 51 / 56 (0.841) |
| 250 | 67.6 / 47 / 82 / 124 (0.841) |
| 500 | 114.0 / 79 / 182 / 213 (0.841) |
| 1000 | 181.2 / 166 / 653 / 798 (0.840) |
| 1175 (native) | 198.3 / 217 / 368 / 453 (0.841) |
| 2000 | 258.9 / 269 / 548 / 717 (0.841) |
| 5000 | 366.8 / 437 / 845 / 1,585 (0.843) |

SQuAD:

| K | PHG rule mean B_P / need / at or above / best fixed B_P (ALL) | MTK rule mean B_P / need / at or above / best fixed B_P (ALL) |
|---|---|---|
| 100 | 35.0 / 18 / 72 / 72 (1.000) | 33.0 / 16 / 60 / 60 (1.000) |
| 202 (native) | 60.4 / 24 / 80 / 97 (1.000) | 58.6 / 21 / 155 / 155 (1.000) |
| 250 | 71.6 / 27 / 110 / 110 (1.000) | 68.1 / 24 / 106 / 117 (1.000) |
| 500 | 114.8 / 35 / 159 / 159 (1.000) | 111.1 / 37 / 157 / 157 (1.000) |
| 1000 | 177.2 / 46 / 839 / 839 (1.000) | 171.8 / 46 / 722 / 722 (1.000) |
| 2000 | 256.3 / 63 / 1,833 / 1,833 (1.000) | 247.9 / 55 / 1,148 / 1,148 (1.000) |
| 5000 | 421.2 / 77 / 1,801 / 1,801 (1.000) | 400.6 / 72 / 1,771 / 1,771 (1.000) |

- **Exact reproduction is expensive; near-lossless is not.**
  - PSTAR covers 0.975–0.982 of MetaQA's shards at K 100 and 0.131 at K 5000 (MTK); MuSiQue 0.736 → 0.086; SQuAD 0.822–0.848 → 0.115–0.120. Its slopes are 0.45–0.59 on the structural maps and 0.56–0.70 on the random ones.
  - ESTAR tracks PSTAR. ES|ESTAR is within 0.014 of unrouted on the structural maps (MetaQA 0 to +0.014, MuSiQue −0.012 to 0, SQuAD −0.003 to 0) and within 0.026 on the random ones (MetaQA −0.007 to +0.001, MuSiQue −0.026 to 0, SQuAD −0.003 to +0.001).
  - The need is a small fraction of PSTAR. On the structural maps it goes from 28 to 59 on MetaQA PHG (K 100 → 2000; slope 0.25), 25 to 261 on MetaQA MTK (slope 0.60; 94 at K 2000), 33 to 437 on MuSiQue (slope 0.71) and 18 to 77 / 16 to 72 on SQuAD (slopes 0.38 / 0.39).
- **Where the rule's mean B_P falls below the need, the losses of 35.5 appear.**
  - MuSiQue: the rule is below the need at K 100 (31.8 against 33) and from the native K on (198.3 against 217, 258.9 against 269, 366.8 against 437). Its B_P slope is 0.63; the need's is 0.71.
  - MetaQA MTK: below only at K 5000 (227.6 against 261); slopes 0.53 against 0.60.
  - MetaQA PHG and SQuAD: the rule is above the need at every K (MetaQA PHG 28.4–187.5 against 28–59; SQuAD 35.0–421.2 / 33.0–400.6 against 18–77 / 16–72). Being above is not enough: MetaQA PHG at K 100 is 28.4 against 28 and still loses 0.015 (35.5).
  - The need is one fixed count for every query and the rule's count is per query, so this comparison of a mean with a fixed count is descriptive.
- **The best fixed fan-out is above unrouted by 0.007–0.067 on the KB and by at most 0.004 on text** (differences from the exact counts).
  - On MetaQA the best fixed B_P is 41–154 (PHG) and 35–770 (MTK). Its ALL is 0.526–0.575 / 0.515–0.559, against the rule's 0.493–0.555 / 0.494–0.547 and unrouted's 0.508. It is below the rule's mean B_P from K 250 to K 1000 on both maps and at PHG K 2000, and above it at K 100 and at MTK K 2000–5000.
  - On MuSiQue the best fixed ALL is 0.840–0.843 against unrouted's 0.840, 1 to 7 queries more (+0.004 at K 5000: 1,686 against 1,679). On SQuAD it is 1.000, the same count as unrouted or 1 query more. As in §34.11, on text routing can do little more than preserve recall.

**35.7 Structural against random at the same K.** This is the direction's machine question: *"For the same number and size of shards, does graph-aware partitioning reduce the number of machines required to recover the evidence?"* Each structural cell is paired per query with each random seed at the same K (ES·KNEE_SDIV; gained = the structural cell serves ALL gold and the random one does not; McNemar, descriptive). The need ratio is the random seeds' mean need over the structural need, both at B_N 1000 within 0.01 (35.6). NGP and LO are structural / the RAND mean.

MetaQA:

| cell | Δ ALL @1000 per seed [gained / lost] | largest p @1000 | need: RAND ÷ structural | NGP: structural / RAND | LO: structural / RAND |
|---|---|---|---|---|---|
| PHG K 100 | +0.191 [405 / 23], +0.180 [385 / 26], +0.176 [384 / 33] | 5.9e−77 | 3.29 | 4.4 / 6.2 | 33.2 / 57.0 |
| PHG K 250 | +0.138 [286 / 11], +0.134 [278 / 11], +0.145 [295 / 6] | 5.1e−68 | 6.96 | 4.8 / 6.8 | 70.0 / 125.0 |
| PHG K 432 (native) | +0.140 [291 / 11], +0.131 [269 / 7], +0.141 [290 / 8] | 3.8e−70 | 9.89 | 5.1 / 7.0 | 112.8 / 201.4 |
| PHG K 500 | +0.132 [271 / 7], +0.130 [266 / 7], +0.138 [277 / 2] | 2.8e−69 | 10.02 | 5.2 / 7.0 | 129.8 / 228.6 |
| PHG K 1000 | +0.114 [239 / 12], +0.122 [252 / 9], +0.121 [249 / 8] | 5.8e−56 | 12.34 | 5.5 / 7.1 | 254.8 / 421.0 |
| PHG K 2000 | +0.111 [229 / 7], +0.111 [232 / 11], +0.113 [235 / 10] | 5.2e−55 | 11.74 | 5.8 / 7.2 | 490.8 / 775.4 |
| MTK K 100 | +0.207 [429 / 15], +0.196 [410 / 19], +0.192 [408 / 25] | 2.5e−90 | 3.68 | 4.3 / 6.2 | 32.2 / 57.0 |
| MTK K 250 | +0.138 [288 / 13], +0.134 [282 / 15], +0.145 [295 / 6] | 5.5e−65 | 7.50 | 4.9 / 6.8 | 73.2 / 125.0 |
| MTK K 432 (native) | +0.126 [263 / 11], +0.117 [242 / 8], +0.127 [268 / 14] | 3.9e−61 | 10.57 | 5.1 / 7.0 | 123.4 / 201.4 |
| MTK K 500 | +0.121 [254 / 12], +0.119 [248 / 11], +0.127 [259 / 6] | 1.6e−59 | 11.06 | 5.2 / 7.0 | 138.5 / 228.6 |
| MTK K 1000 | +0.081 [182 / 20], +0.089 [189 / 11], +0.088 [184 / 8] | 6.9e−34 | 12.05 | 5.7 / 7.1 | 285.9 / 421.0 |
| MTK K 2000 | +0.063 [135 / 9], +0.063 [137 / 12], +0.065 [138 / 9] | 4.9e−28 | 7.37 | 6.2 / 7.2 | 600.3 / 775.4 |
| MTK K 5000 | +0.057 [120 / 6], +0.054 [114 / 7], +0.054 [122 / 14] | 1.1e−22 | 3.05 | 6.5 / 7.2 | 1,463.5 / 1,785.8 |

MuSiQue:

| cell | Δ ALL @1000 per seed [gained / lost] | largest p @1000 | need: RAND ÷ structural | NGP: structural / RAND | LO: structural / RAND |
|---|---|---|---|---|---|
| PHG K 100 | +0.141 [315 / 33], +0.125 [289 / 40], +0.145 [330 / 41] | 1.1e−47 | 2.61 | 2.0 / 2.4 | 10.1 / 32.9 |
| PHG K 250 | +0.094 [225 / 38], +0.098 [225 / 30], +0.081 [201 / 40] | 5.3e−27 | 4.09 | 2.1 / 2.4 | 18.4 / 60.2 |
| PHG K 500 | +0.060 [156 / 37], +0.055 [146 / 36], +0.053 [138 / 32] | 6.9e−17 | 4.42 | 2.2 / 2.4 | 33.3 / 95.4 |
| PHG K 1000 | +0.042 [117 / 33], +0.041 [109 / 28], +0.030 [97 / 37] | 2.2e−7 | 3.64 | 2.2 / 2.4 | 55.2 / 155.8 |
| PHG K 1175 (native) | +0.035 [99 / 30], +0.037 [106 / 32], +0.040 [108 / 28] | 8.4e−10 | 3.04 | 2.3 / 2.4 | 67.5 / 175.5 |
| PHG K 2000 | +0.028 [80 / 25], +0.030 [84 / 25], +0.024 [78 / 31] | 7.7e−6 | 3.42 | 2.3 / 2.4 | 100.9 / 254.9 |
| PHG K 5000 | +0.008 [43 / 27], +0.014 [47 / 20], +0.015 [48 / 18] | 0.072 | 3.68 | 2.3 / 2.4 | 228.7 / 491.9 |

SQuAD:

| cell | Δ ALL @1000 per seed [gained / lost] | largest p @1000 | need: RAND ÷ structural | NGP: structural / RAND | LO: structural / RAND |
|---|---|---|---|---|---|
| PHG K 100 | +0.010 [22 / 2], +0.006 [16 / 5], +0.006 [13 / 2] | 0.027 | 2.61 | 1.0 / 1.0 | 2.8 / 8.3 |
| PHG K 202 (native) | +0.005 [10 / 1], +0.008 [16 / 0], +0.004 [9 / 1] | 0.021 | 3.04 | 1.0 / 1.0 | 3.2 / 9.6 |
| PHG K 250 | +0.004 [10 / 2], +0.005 [10 / 0], +0.007 [13 / 0] | 0.039 | 2.73 | 1.0 / 1.0 | 3.5 / 9.9 |
| PHG K 500 | +0.002 [5 / 1], +0.004 [7 / 0], +0.003 [7 / 1] | 0.22 | 2.65 | 1.0 / 1.0 | 4.2 / 10.8 |
| PHG K 1000 | −0.001 [1 / 2], +0.001 [1 / 0], −0.001 [2 / 3] | 1 | 2.20 | 1.0 / 1.0 | 6.4 / 12.0 |
| PHG K 2000 | −0.002 [0 / 3], 0 [1 / 1], −0.001 [1 / 2] | 1 | 1.66 | 1.0 / 1.0 | 9.0 / 13.1 |
| PHG K 5000 | 0 [1 / 1], −0.001 [1 / 3], 0 [1 / 1] | 1 | 1.45 | 1.0 / 1.0 | 12.9 / 16.5 |
| MTK K 100 | +0.011 [23 / 2], +0.006 [16 / 4], +0.006 [14 / 2] | 0.012 | 2.94 | 1.0 / 1.0 | 2.4 / 8.3 |
| MTK K 202 (native) | +0.004 [11 / 4], +0.007 [14 / 0], +0.003 [9 / 3] | 0.15 | 3.48 | 1.0 / 1.0 | 3.1 / 9.6 |
| MTK K 250 | +0.003 [8 / 2], +0.004 [9 / 1], +0.006 [12 / 1] | 0.11 | 3.07 | 1.0 / 1.0 | 3.1 / 9.9 |
| MTK K 500 | +0.002 [5 / 1], +0.004 [7 / 0], +0.003 [7 / 1] | 0.22 | 2.50 | 1.0 / 1.0 | 4.0 / 10.8 |
| MTK K 1000 | 0 [2 / 2], +0.001 [2 / 0], 0 [2 / 2] | 1 | 2.20 | 1.0 / 1.0 | 5.8 / 12.0 |
| MTK K 2000 | −0.001 [0 / 2], +0.001 [1 / 0], 0 [2 / 2] | 1 | 1.90 | 1.0 / 1.0 | 7.7 / 13.1 |
| MTK K 5000 | 0 [1 / 1], −0.001 [1 / 3], 0 [1 / 1] | 1 | 1.55 | 1.0 / 1.0 | 12.5 / 16.5 |

- **The structural maps recover more evidence under the same rule.**
  - At B_N 1000, MetaQA gains +0.054 to +0.207 on every cell and seed (p ≤ 1.1e−22).
  - MuSiQue gains +0.024 to +0.145 up to K 2000 (p ≤ 7.7e−6) and +0.008 to +0.015 at K 5000 (p 0.072).
  - SQuAD gains +0.003 to +0.011 up to K 250 and +0.002 to +0.004 at K 500, and is within 0.002 from K 1000 on (p 1).
  - At B_N 500 / 2000 the ranges are +0.025 to +0.169 / +0.068 to +0.249 (MetaQA), +0.003 to +0.121 / +0.021 to +0.166 (MuSiQue) and −0.001 to +0.010 / −0.002 to +0.011 (SQuAD).
  - The gain shrinks as K grows: MetaQA MTK goes from +0.192 to +0.207 at K 100 to +0.054 to +0.057 at K 5000, and MuSiQue from +0.125 to +0.145 to +0.008 to +0.015.
- **Machines: at the same K a random map needs 1.45–12.34× the structural map's fan-out.**
  - MetaQA: 3.05–12.34. On PHG the ratio rises from 3.29 (K 100) to 12.34 (K 1000) and 11.74 (K 2000). On MTK it rises from 3.68 to 12.05 (K 1000), then falls to 7.37 and 3.05 as MTK's own need grows (94 and 261 at K 2000 and 5000).
  - MuSiQue: 2.61–4.42 (4.42 at K 500, 3.68 at K 5000).
  - SQuAD: 1.45–3.48. It is highest near the native K (3.04 / 3.48) and falls to 1.45 / 1.55 at K 5000.
  - The rule's own count points the same way with a smaller ratio (structural ÷ random B_P 0.45–0.84, 35.4). On the random maps it stops short of their need and loses recall instead (35.5).
- **Why:** the structural maps keep the evidence in fewer shards and bring the last gold shard earlier in the router's order.
  - NGP is 4.3–6.5 against 6.2–7.2 on MetaQA, and 2.0–2.3 against 2.4 on MuSiQue. On SQuAD it is 1.0 on every map, since each query has one gold node.
  - LO is 32.2–1,463.5 against 57.0–1,785.8 on MetaQA, 10.1–228.7 against 32.9–491.9 on MuSiQue and 2.4–12.9 against 8.3–16.5 on SQuAD.
- **"The same size" holds for the mean only.** The structural blocks are unequal: PHG's smallest block is 1 node on most maps, and so is MTK's on SQuAD K ≥ 500 and MetaQA K 5000. The random blocks differ by at most one node (35.3). So the per-query scan and the load differ at equal K (35.8, 35.9).

**35.8 Cost: fan-out against local scan (K\*).** The cost components of ES·KNEE_SDIV per cell are the mean B_P (network fan-out) and the mean CMASS (the nodes inside the contacted shards: the local scan). Communication is in 35.9. No weights are invented. One shard contact is priced as c scanned nodes, so the cost is B_P · c + CMASS, and c is the deployment's to supply. RAND1–2's CMASS is listed next to RAND0's.

MetaQA:

| K | PHG B_P / CMASS | MTK B_P / CMASS | RAND0 B_P / CMASS | RAND1–2 CMASS |
|---|---|---|---|---|
| 100 | 28.4 / 12,408.9 | 28.7 / 12,708.9 | 35.4 / 15,310.6 | 15,198.9, 15,266.6 |
| 250 | 60.7 / 10,636.1 | 62.0 / 10,760.5 | 74.1 / 12,818.7 | 12,844.1, 12,804.3 |
| 432 (native) | 91.5 / 9,282.1 | 92.2 / 9,259.0 | 115.1 / 11,514.7 | 11,469.6, 11,472.7 |
| 500 | 101.1 / 8,849.0 | 101.4 / 8,793.4 | 127.7 / 11,049.9 | 11,050.1, 11,042.0 |
| 1000 | 146.3 / 6,402.5 | 147.6 / 6,407.0 | 198.9 / 8,602.9 | 8,529.4, 8,561.9 |
| 2000 | 187.5 / 4,116.7 | 188.4 / 4,085.1 | 264.2 / 5,711.7 | 5,668.6, 5,689.8 |
| 5000 | absent | 227.6 / 1,999.1 | 316.5 / 2,742.1 | 2,735.3, 2,749.7 |

MuSiQue:

| K | PHG B_P / CMASS | RAND0 B_P / CMASS | RAND1–2 CMASS |
|---|---|---|---|
| 100 | 31.8 / 37,551.5 | 45.7 / 53,721.1 | 53,723.6, 53,779.1 |
| 250 | 67.6 / 32,041.4 | 98.2 / 46,173.4 | 46,379.6, 46,113.6 |
| 500 | 114.0 / 27,101.6 | 173.3 / 40,737.2 | 40,833.8, 40,769.8 |
| 1000 | 181.2 / 21,576.3 | 302.4 / 35,536.5 | 35,542.5, 35,470.7 |
| 1175 (native) | 198.3 / 20,126.1 | 342.1 / 34,222.9 | 34,215.8, 34,141.5 |
| 2000 | 258.9 / 15,455.0 | 502.2 / 29,514.0 | 29,535.2, 29,443.8 |
| 5000 | 366.8 / 8,776.5 | 824.1 / 19,376.9 | 19,406.2, 19,346.3 |

SQuAD:

| K | PHG B_P / CMASS | MTK B_P / CMASS | RAND0 B_P / CMASS | RAND1–2 CMASS |
|---|---|---|---|---|
| 100 | 35.0 / 7,104.7 | 33.0 / 6,707.6 | 47.9 / 9,694.1 | 9,644.3, 9,706.5 |
| 202 (native) | 60.4 / 6,087.4 | 58.6 / 5,909.2 | 85.3 / 8,546.3 | 8,511.6, 8,575.6 |
| 250 | 71.6 / 5,837.5 | 68.1 / 5,527.2 | 101.9 / 8,245.6 | 8,228.6, 8,236.7 |
| 500 | 114.8 / 4,704.7 | 111.1 / 4,545.1 | 178.0 / 7,203.9 | 7,233.9, 7,246.0 |
| 1000 | 177.2 / 3,633.4 | 171.8 / 3,529.7 | 301.4 / 6,101.3 | 6,124.0, 6,121.8 |
| 2000 | 256.3 / 2,633.6 | 247.9 / 2,637.2 | 487.3 / 4,935.7 | 4,950.8, 4,944.1 |
| 5000 | 421.2 / 1,741.3 | 400.6 / 1,822.2 | 764.5 / 3,104.1 | 3,111.1, 3,102.1 |

- **The scan falls as K grows, while the fan-out rises** (35.4).
  - Structural CMASS: MetaQA 12,408.9 → 4,116.7 (PHG, K 2000) and 12,708.9 → 1,999.1 (MTK); MuSiQue 37,551.5 → 8,776.5; SQuAD 7,104.7 → 1,741.3 (PHG) and 6,707.6 → 1,822.2 (MTK).
  - The random maps scan more at every K: RAND0 goes 15,310.6 → 2,742.1, 53,721.1 → 19,376.9 and 9,694.1 → 3,104.1.
- **The feasible cells** (R_ALL ≥ unrouted − tol):
  - At B_N 1000 with tol 0.01:
    - MetaQA 11: MTK K 100, and PHG and MTK at K 250, 432, 500, 1000 and 2000.
    - MuSiQue 5: PHG K 250 to K 2000. K 2000 sits exactly on the bound (−0.010: 1,659 against 1,679).
    - SQuAD 34 of 35: all but RAND0 K 100.
  - At B_N 1000 with tol 0: MetaQA 10 (MTK K 2000 drops out); MuSiQue 0; SQuAD 0.
  - At B_N 500 with tol 0.01: MetaQA 13; MuSiQue 11, including RAND2 K 2000 and RAND0–2 K 5000; SQuAD 35. With tol 0: 12 / 1 (PHG K 250) / 21.
  - At B_N 2000 with tol 0.01: 10 / 1 (PHG K 250) / 34. With tol 0: 9 / 0 / 0.
  - No random cell is feasible on MetaQA at any of these B_N, nor on MuSiQue at B_N 1000–2000.
- **The band.** At B_N 1000 with tol 0.01, every structural cell with 250 ≤ K ≤ 2000 is feasible.
  - Outside the band, four cells lose more than 0.01: MetaQA PHG K 100 (−0.015), MuSiQue K 100 (−0.011), MetaQA MTK K 5000 (−0.014) and MuSiQue K 5000 (−0.015).
  - MetaQA MTK K 100 (+0.002) and SQuAD at every K are feasible.
- **The break-even hull** at B_N 1000, tol 0.01: per partitioner, the hull cell that minimises B_P · c + CMASS for each range of c.
  - MetaQA PHG: K 250 for c ≥ 51.4 and K 2000 below it (K 432, 500 and 1000 are off the hull).
  - MetaQA MTK: K 100 for c ≥ 58.5, K 250 for 52.8–58.5 and K 2000 below 52.8. At tol 0: K 250 for 50.9–58.5 and K 1000 below 50.9.
  - MuSiQue PHG: K 250 for c ≥ 106.3, K 500 for 82.8–106.3, K 1175 for 77.1–82.8 and K 2000 below 77.1 (K 1000 is off the hull).
  - SQuAD PHG: K 100 for c ≥ 40.0, then K 202 / 500 / 1000 / 2000 / 5000 with breaks at 25.4 / 17.2 / 12.6 / 5.4 (K 250 is off the hull). MTK: K 100 / 250 / 500 / 1000 / 2000 / 5000 with breaks at 33.6 / 22.8 / 16.7 / 11.7 / 5.3 (K 202 is off the hull).
  - SQuAD RAND0: K 202 for c ≥ 18.2 down to K 5000 for c ≤ 6.5 (K 2000 is off the hull). RAND1–2 add K 100 for c ≥ 30.4 / 30.0.
  - So K\* is set by c. On the multi-hop sets, K\* is the band's small end (K 250; MetaQA MTK's K 100 above 58.5) when a contact costs more than 51.4 (MetaQA PHG), 52.8 (MetaQA MTK) or 106.3 (MuSiQue) scanned nodes. It is K 2000 when a contact costs less than 51.4 / 52.8 / 77.1. In between, MuSiQue passes through K 500 and K 1175.
- **Communication favours small K.** NSEEDP rises monotonically with K on every structural map (35.9), so any positive weight on it moves K\* down.

**35.9 Communication and load.** The communication table gives, per cell:
- NSEEDP;
- NEV;
- XC;
- cross-E, the static cross-shard share of all entries of E (4 decimals);
- RDS ÷ RDN, the router's union-sketch reads over its node-row reads. RDN does not depend on K: the records' means are 2,650.2 / 17,992.77 / 22,267.92 entries per query.

The load table gives, per cell, the ES·KNEE_SDIV contacts over the population: the largest contact rate (the share of queries that contact the most-contacted shard) / peak ÷ mean / Gini. Never-contacted shards are in brackets. The tables show RAND0; the ranges in the bullets cover all three seeds.

MetaQA:

| K | PHG NSEEDP / NEV / XC / cross-E / RDS÷RDN | MTK NSEEDP / NEV / XC / cross-E / RDS÷RDN | RAND0 NSEEDP / NEV / XC / cross-E / RDS÷RDN |
|---|---|---|---|
| 100 | 72.0 / 99.0 / 0.779 / 0.7411 / 0.616 | 72.4 / 98.4 / 0.741 / 0.7044 / 0.573 | 86.1 / 100.0 / 0.992 / 0.9899 / 0.871 |
| 250 | 106.3 / 245.1 / 0.796 / 0.7669 / 0.674 | 106.8 / 245.9 / 0.749 / 0.7202 / 0.638 | 136.3 / 249.9 / 0.997 / 0.9961 / 0.928 |
| 432 (native) | 127.7 / 399.1 / 0.812 / 0.7836 / 0.718 | 121.5 / 400.2 / 0.755 / 0.7303 / 0.671 | 159.7 / 427.2 / 0.997 / 0.9978 / 0.947 |
| 500 | 129.2 / 453.5 / 0.814 / 0.7862 / 0.725 | 124.9 / 452.5 / 0.763 / 0.7340 / 0.681 | 164.8 / 490.9 / 0.999 / 0.9981 / 0.953 |
| 1000 | 146.2 / 726.4 / 0.827 / 0.8020 / 0.764 | 137.5 / 714.0 / 0.768 / 0.7448 / 0.713 | 181.2 / 870.9 / 0.999 / 0.9991 / 0.967 |
| 2000 | 154.6 / 1,002.6 / 0.831 / 0.8120 / 0.791 | 148.7 / 989.7 / 0.781 / 0.7588 / 0.748 | 190.6 / 1,294.6 / 1.000 / 0.9996 / 0.975 |
| 5000 | absent | 156.6 / 1,277.0 / 0.806 / 0.7902 / 0.792 | 196.6 / 1,718.7 / 1.000 / 0.9998 / 0.981 |

| K | PHG largest rate / peak÷mean / Gini [never] | MTK largest rate / peak÷mean / Gini [never] | RAND0 largest rate / peak÷mean / Gini [never] |
|---|---|---|---|
| 100 | 0.937 / 3.30 / 0.278 | 0.950 / 3.32 / 0.279 | 0.822 / 2.32 / 0.174 |
| 250 | 0.980 / 4.04 / 0.322 | 0.982 / 3.96 / 0.298 | 0.882 / 2.98 / 0.216 |
| 432 (native) | 0.978 / 4.62 / 0.354 | 0.979 / 4.59 / 0.338 | 0.935 / 3.51 / 0.249 |
| 500 | 0.981 / 4.85 / 0.355 | 0.984 / 4.85 / 0.347 | 0.953 / 3.73 / 0.259 |
| 1000 | 0.988 / 6.75 / 0.410 [1] | 0.989 / 6.71 / 0.392 | 0.976 / 4.91 / 0.315 |
| 2000 | 0.991 / 10.57 / 0.459 | 0.990 / 10.51 / 0.443 | 0.980 / 7.42 / 0.370 |
| 5000 | absent | 0.991 / 21.77 / 0.522 | 0.979 / 15.47 / 0.460 |

MuSiQue:

| K | PHG NSEEDP / NEV / XC / cross-E / RDS÷RDN | RAND0 NSEEDP / NEV / XC / cross-E / RDS÷RDN |
|---|---|---|
| 100 | 36.2 / 99.6 / 0.796 / 0.7748 / 0.150 | 86.6 / 100.0 / 0.990 / 0.9900 / 0.427 |
| 250 | 53.8 / 241.4 / 0.865 / 0.8443 / 0.202 | 137.8 / 250.0 / 0.996 / 0.9960 / 0.577 |
| 500 | 68.9 / 449.5 / 0.898 / 0.8814 / 0.243 | 165.0 / 499.8 / 0.998 / 0.9980 / 0.676 |
| 1000 | 85.7 / 777.6 / 0.913 / 0.8974 / 0.276 | 181.5 / 992.7 / 0.999 / 0.9990 / 0.757 |
| 1175 (native) | 90.2 / 872.1 / 0.920 / 0.9040 / 0.284 | 184.1 / 1,159.6 / 0.999 / 0.9992 / 0.771 |
| 2000 | 103.1 / 1,218.3 / 0.926 / 0.9122 / 0.297 | 190.6 / 1,882.0 / 1.000 / 0.9995 / 0.817 |
| 5000 | 127.7 / 1,865.7 / 0.942 / 0.9304 / 0.312 | 196.2 / 3,627.5 / 1.000 / 0.9998 / 0.872 |

| K | PHG largest rate / peak÷mean / Gini [never] | RAND0 largest rate / peak÷mean / Gini [never] |
|---|---|---|
| 100 | 0.737 / 2.32 / 0.145 | 0.593 / 1.30 / 0.038 |
| 250 | 0.847 / 3.13 / 0.171 | 0.665 / 1.69 / 0.056 |
| 500 | 0.894 / 3.92 / 0.188 | 0.637 / 1.84 / 0.073 |
| 1000 | 0.962 / 5.31 / 0.215 [1] | 0.675 / 2.23 / 0.094 |
| 1175 (native) | 0.959 / 5.68 / 0.224 [1] | 0.718 / 2.46 / 0.106 |
| 2000 | 0.979 / 7.56 / 0.256 | 0.793 / 3.16 / 0.141 |
| 5000 | 0.920 / 12.53 / 0.318 [2] | 0.866 / 5.25 / 0.212 |

SQuAD:

| K | PHG NSEEDP / NEV / XC / cross-E / RDS÷RDN | MTK NSEEDP / NEV / XC / cross-E / RDS÷RDN | RAND0 NSEEDP / NEV / XC / cross-E / RDS÷RDN |
|---|---|---|---|
| 100 | 47.3 / 99.9 / 0.863 / 0.8574 / 0.149 | 44.2 / 99.7 / 0.836 / 0.8277 / 0.131 | 86.8 / 100.0 / 0.990 / 0.9901 / 0.421 |
| 202 (native) | 63.2 / 198.8 / 0.882 / 0.8763 / 0.177 | 60.1 / 197.5 / 0.855 / 0.8458 / 0.157 | 127.2 / 202.0 / 0.995 / 0.9950 / 0.554 |
| 250 | 69.0 / 243.6 / 0.885 / 0.8780 / 0.186 | 65.5 / 241.8 / 0.852 / 0.8423 / 0.162 | 137.8 / 250.0 / 0.996 / 0.9961 / 0.591 |
| 500 | 91.3 / 451.3 / 0.886 / 0.8778 / 0.204 | 87.4 / 447.2 / 0.859 / 0.8481 / 0.179 | 165.5 / 499.9 / 0.998 / 0.9980 / 0.687 |
| 1000 | 117.5 / 779.9 / 0.911 / 0.9019 / 0.228 | 114.0 / 767.7 / 0.902 / 0.8926 / 0.213 | 182.0 / 995.5 / 0.999 / 0.9990 / 0.757 |
| 2000 | 139.3 / 1,236.1 / 0.947 / 0.9417 / 0.271 | 133.5 / 1,193.9 / 0.943 / 0.9369 / 0.257 | 191.0 / 1,896.4 / 1.000 / 0.9995 / 0.802 |
| 5000 | 162.8 / 2,180.1 / 0.978 / 0.9758 / 0.380 | 157.1 / 2,068.0 / 0.975 / 0.9726 / 0.369 | 197.0 / 3,569.1 / 1.000 / 0.9998 / 0.836 |

| K | PHG largest rate / peak÷mean / Gini [never] | MTK largest rate / peak÷mean / Gini [never] | RAND0 largest rate / peak÷mean / Gini [never] |
|---|---|---|---|
| 100 | 0.691 / 1.97 / 0.141 | 0.503 / 1.52 / 0.144 | 0.578 / 1.21 / 0.045 |
| 202 (native) | 0.677 / 2.26 / 0.159 | 0.578 / 1.99 / 0.165 | 0.589 / 1.39 / 0.067 |
| 250 | 0.616 / 2.15 / 0.166 | 0.626 / 2.30 / 0.164 | 0.584 / 1.43 / 0.072 |
| 500 | 0.706 / 3.07 / 0.194 | 0.705 / 3.17 / 0.189 | 0.682 / 1.91 / 0.098 |
| 1000 | 0.829 / 4.68 / 0.220 | 0.815 / 4.74 / 0.221 | 0.756 / 2.51 / 0.131 |
| 2000 | 0.871 / 6.80 / 0.278 | 0.876 / 7.06 / 0.297 [1] | 0.876 / 3.59 / 0.183 |
| 5000 | 0.927 / 11.00 / 0.353 | 0.909 / 11.34 / 0.402 [2] | 0.933 / 6.10 / 0.275 |

- **Cross-shard traffic.**
  - On the structural maps, 0.741–0.978 of the seeds' one-hop entries leave the seed's shard (XC), and 0.7044–0.9758 of all of E's entries cross shards. Both grow with K, apart from a dip at SQuAD MTK K 250.
  - On the random maps they are 0.989–1.000 and 0.9899–0.9998: nearly every entry crosses.
- **Router reads.** Reading the union shard sketch instead of the seeds' node rows costs RDS ÷ RDN:
  - on the structural maps, 0.573–0.792 on MetaQA, 0.150–0.312 on MuSiQue and 0.131–0.380 on SQuAD, growing with K;
  - on the random maps, 0.421–0.981.
- **Hot shards.** The rule's load concentrates as K grows.
  - The largest contact rate is 0.503–0.991 on the structural maps (0.815–0.991 from K 1000 on). On MetaQA, from K 250 on, some shard is contacted by 0.978–0.991 of the queries.
  - Peak ÷ mean rises from 3.30–3.32 / 2.32 / 1.52–1.97 at K 100 to 21.77 (MTK) / 12.53 / 11.00–11.34 at K 5000 (MetaQA / MuSiQue / SQuAD).
  - Gini is 0.141–0.522 on the structural maps and 0.038–0.460 on the random ones (peak ÷ mean 1.21–15.53).
  - At most 2 shards are never contacted (MuSiQue K 5000, SQuAD MTK K 5000), and none on the random maps.
  - On the served cells, the ES|ESTAR@1000 reference spreads the load more evenly than the rule: peak ÷ mean 1.36 / 4.60 / 1.58 against 4.62 / 5.68 / 2.26 (MetaQA PHG, MuSiQue, SQuAD PHG).
  - A deployment would replicate or cache these shards. No replication model was run (35.10).

**35.10 Not run, and flags.**
- **Not run:**
  - Mt-KaHyPar on MuSiQue. Its native run is FAILED_MEMORY_CAP at the contract cap; a smaller K might fit, but that is untested.
  - PHG on MetaQA at K 5000 (PARTITION_INVALID; absent, 35.3).
  - K above 5000, and the three large corpora (WebQSP, HotpotQA, 2Wiki).
  - A network or latency model. B_P, CMASS and NSEEDP are counts, and c is left to the reader (35.8).
  - Shard replication or caching for the hot shards (35.9).
  - No new rule, no XGBoost and no learned parameter. Nothing held out was read.
- **FLAGGED:**
  - The retry rule for host-limited memory kills (35.2) was written after the first pass had failed at host-limited caps. It reruns only unchanged runs, and Mt-KaHyPar is deterministic, so a retry can change only whether a run finishes.
  - The hypergraph changes with K under the frozen cap rule (35.3), so each cell varies the hyperedges along with the block count.
  - "The same size" is the same mean size only (35.3, 35.7).
  - ES − S grows with K on MuSiQue, to −0.013 at B_N 1000 and −0.022 at B_N 5000 (K 5000): the router-only order costs more on the finest maps.
  - The summary module was last modified after the two 200-row smokes had been read and before the full records existed. The smokes overlapped the MuSiQue PHG builds (35.12).
  - The population is §31–§34's development population, on which ES·KNEE_SDIV was selected. §35 is development evidence on the same queries.
  - "reach" losses include queries that no count serves at B_N 1000 (35.5).
  - K\* uses means. The tails (a query's B_P reaches 0.618 of MuSiQue's shards at K 5000; 35.4) and the hot shards (35.9) are not in the hull.
  - The native K is off the slope grid, and MetaQA PHG's slopes cover K ≤ 2000 only. The slopes are least-squares fits on at most six points.
  - "−0.000" and "+0.000" are the summary's stored strings for nonzero means that round to zero.

**35.11 Reading** (development numbers, descriptive; the decision is the user's step).
- **Sublinear, not linear.**
  - On every map, B_P grows sublinearly in K. The slopes are 0.53–0.64 on the structural maps and 0.57–0.75 on the random ones.
  - ρ falls from 0.284–0.350 at K 100 to 0.046–0.084 at K 5000 on the structural maps.
  - So the answer to *"We need to know which one is true"* is: sublinear, on all three corpora and for both structural partitioners.
- **But the evidence's position grows nearly linearly.**
  - LO has slope 0.80–0.99 on the multi-hop corpora. The rule's count grows more slowly than LO, and on MuSiQue and MetaQA MTK more slowly than the need.
  - The result is erosion at large K. At B_N 1000, MuSiQue loses 0.010 at K 2000 and 0.015 at K 5000, and MetaQA MTK's KB gain goes from +0.039 (K 250) to −0.014 (K 5000).
  - SQuAD (one gold node per query) and MetaQA PHG (up to K 2000) hold.
- **Graph-aware partitioning reduces the machines.**
  - At the same K, a balanced random map needs 1.45–12.34× as many shards as the structural map for near-lossless recall. Under the rule, the structural map contacts 0.45–0.84× as many shards as the random one.
  - The structural maps also recover more evidence: +0.054 to +0.207 on MetaQA and +0.008 to +0.145 on MuSiQue at B_N 1000.
  - On SQuAD the retrieval difference vanishes from K 1000 on, but the need ratio stays 1.45–2.20.
- **K\* depends on c.**
  - Within 250 ≤ K ≤ 2000, every structural cell is within 0.01 of unrouted at B_N 1000.
  - The hull picks the band's small end when a shard contact costs more than about 51–106 scanned nodes, and K 2000 below about 51–77.
  - Communication (NSEEDP) and the hot shards favour smaller K.
- **For the user** (nothing is selected, and nothing is frozen):
  - (a) Spend the ONE held-out evaluation at a fixed K inside the 250–2000 band: either the native K (N // 100, as in §32–§34) or a K taken from the hull for an assumed c.
  - (b) Report the K-curve as it stands, with the large-K erosion (MuSiQue, MetaQA MTK) and the load concentration as limitations of the rule.
  - (c) Rule first on the open build items: MuSiQue Mt-KaHyPar at small K, the absent MetaQA PHG K 5000, and a replication or network model for c and the hot shards.
  - The held-out evaluation is the user's step. ES·KNEE_SDIV stays a development candidate.

**35.12 Cost and hygiene.**
- **Builds** (`_l1d_scale_parts.py`, one partitioner at a time):
  - Hypergraphs: 1.4–2.8 s per K on MetaQA, 27.0 s (K 100) to 140.6 s (K 5000) on MuSiQue and 9.0–24.6 s on SQuAD (09:15:53–09:26:55 local).
  - PHG: 199.7 s for SQuAD's 7 cells, and 90.6 s for MetaQA's 6 cells plus the refused K 5000 (about 09:28–09:32). MuSiQue took 541.2 s (about 09:34–09:43).
  - MTK: the first pass took 326.4 s (MetaQA) and 277.5 s (SQuAD), from 09:44 to 09:53:32. The three retries took 175.4 s (MetaQA K 2000), 95.9 s (SQuAD K 5000) and 421.7 s (MetaQA K 5000; its log ends at 10:22:54).
  - Builder versions (all three in `_history/`; each build record names its builder's sha):
    - v0 2090b01e… built the 21 hypergraphs and 13 PHG maps, and stopped at the MetaQA K 5000 refusal;
    - v1 acaa619e… recorded that refusal from the existing driver log, built MuSiQue's 7 PHG maps and the 11 first-pass MTK maps, and wrote the 3 FAILED first attempts;
    - v2 906a41c2… ran the 3 retries.
- **Wall-clock and memory** (the harness runs, one at a time):

| run | start (local) | seconds (loop) | peak RSS (MB) | available at start (GiB) |
|---|---|---|---|---|
| scale SQuAD | 10:23:18 | 275.0 (224.4) | 530.6 | 5.77 |
| scale MetaQA | 10:28:08 | 226.9 (215.9) | 820.8 | 7.36 |
| scale MuSiQue | 10:32:07 | 630.7 (399.0) | 1,392.3 | 7.26 |

- **Per-query latency** (ms; MetaQA / MuSiQue / SQuAD):
  - node score + LOC order 0.6 / 1.8 / 1.3; fused order 4.4 / 15.6 / 2.4;
  - FLAT: products 2.7 / 12.4 / 2.5 plus RRF 13.6 / 43.9 / 6.0;
  - per cell (scores, both rankings, counts and evaluation): 0.6–9.0 / 1.4–10.3 / 0.8–9.0.
- **Host at the starts:** 5.77–7.36 GiB available of 15.69 GiB, with 1.44 GiB of swap in use. No other Python process was above 500 MB.
- **The smokes.** The two 200-row smokes (SQuAD at 09:40:10 and MetaQA at 09:41:32 local, written outside the repository) ran while the MuSiQue PHG builds were running. The full runs were sequential.
- **Population:** the development population c7f70806… (§31). Nothing held out was read.
- **Nothing under `data/` was written. No pinned module or CONTRACT_FILE was edited.**
  - `_l1c_phg.py` (7f2a416c…) ran unchanged. So did the Mt-KaHyPar worker `_l1hu_local_worker.py` (a2dff223…), the driver `src/l1_canonical/partition.py` (45be47b5…), the hypergraph builder `src/l1_canonical/hypergraph.py` (9c868b09…) and the frozen rule module `_l1hu_build.py` (f440c2e4…).
  - The MTK runs went through the builder's guard script `results/L1_COVPART/parts/_scale_mtk_guard.sh` (96d0c91c…). Each MTK RUN record pins it and asserts that its text equals `partition.py`'s GUARD_SH.
  - Each harness record pins `_l1d_scale.py` (ba0a1837…), `_l1d_arms.py` (f5f52b79…) and `_l1d_lib.py` (dafe39b2…).
  - Through `structures.imports`, the harness records also pin `_l1d_node1h.py` (8265d6bc…), `_l1d_edgediag.py` (874bfc49…), `_l1d_adaptbp.py` (984e3aef…), `_l1d_route.py` (7b420e54…), `_l1d_route2.py` (890566c7…), `_l1d_route3.py` (283a711b…) and `_l1d_scale_parts.py` (906a41c2…).
  - `structures.reproduces` pins the route, route2 and route3 records (json and npz shas).
- **The summary module** `_l1d_scale_summary.py` (08696c8b…) is new.
  - It was last modified at 09:52:15 local, after the smokes and before the full records existed.
  - Previews written outside the repository at 10:28:17 (SQuAD only) and 10:32:14 (two datasets) did not lead to a change.
  - It wrote `scale_SUMMARY__v1.json` (4cb75202…) at 10:42:48 in 1.5 s. It pins the summary and harness code and each dataset record's json and npz shas.
- **Rounding.**
  - The summary recomputes every number from the stored per-query arrays (exact integer counts): shares to 3 decimals, means to 1, ratios and slopes to 2, with halves rounded away from zero. An exact zero is written 0.
  - The fan-out ratios (35.4) come from exact per-query count sums, and the need ratios (35.7) from the integer needs. Read-only scripts outside the repository computed them. Two fan-out ratios differ from the ratio of the rounded means: MetaQA PHG K 100 (0.81) and MTK K 1000 (0.74).
  - The p-values are the summary's (descriptive McNemar), shown to 2 significant digits.
  - The cross-E shares are the summary's 4-decimal values. RDN is the records' stored mean. Latencies are shown to 1 decimal from the records' 3-decimal means.

## 36. KSCALE: K-aware fan-out for ES·KNEE_SDIV, B_P = ⌈KNEE_SDIV(K 100 map) · (K/100)^α⌉ (2026-09-29 local) (`_l1d_kscale.py` on `_l1d_scale.py` (subclassed, not edited) and the shared runner `_l1d_arms.py`, summary `_l1d_kscale_summary.py`; records `results/L1_DEV/kscale_{metaqa,musique,squad}__v1.{json,npz}`, `kscale_SUMMARY__v1.json`; STATUS = DEVELOPMENT: descriptive numbers, no pre-registration, no verdicts, no adoption)

**36.1 The direction (2026-09-29 local).** After §35 the user pasted a third advisor text without comment. As with the second (§35.1), it is taken as the user's direction. Its operative sentences, verbatim (math in plain text; the paste's doubled rendering, e.g. "KK" for K, is shown once):
- The diagnosis: *"What is failing at large K is not partitioning itself. It is the count policy."* *"That means I would not freeze one K yet and I would not spend the final held-out evaluation yet."*
- The question: *"how should fan-out scale with partition granularity K?"* *"We have been using a query-adaptive count, but it is almost K-agnostic. That is now clearly wrong."*
- The form: *"B_P(q,K) = B_query(q)·g(K)"*, *"where g(K) increases slowly with the number of shards"*. For example *"B_P(q,K) = B_ES·KNEE(q,K0) (K/K0)^α with a fixed, non-learned α. I would not blindly tune α, but testing something like α∈{0.65,0.75,0.85,1.0} is now a very targeted systems experiment, unlike the previous thousands-of-router-formulas sweep."*
- The goal: *"to find the smallest scaling law that preserves: R_ALL(K) while still keeping: B_P/K ↓."* Its example: *"if we can get: B_P∝K^0.75 while gold spread behaves around K^0.85, we may recover most of the lost recall while still having: B_P/K∝K^−0.25."*
- The constraint: *"Keep everything else fixed: IR_L1; ES shard ordering; same partitions; same node budget; no learning; no new structural mechanism. Only modify how the number of shards scales with K."*
- The test: *"If a simple K-aware scaling law removes the −1 to −1.5 point large-K degradation while preserving the rapidly declining fraction of contacted shards, then you've got the result you actually wanted"*.
- **Corrections to the text** (§35's records, `scale_SUMMARY__v1.json`; the exponents from the exact per-query sums):
  - *"almost K-agnostic"*: KNEE_SDIV is computed on each K map, and its mean grows with K. The least-squares slopes over the grid are 0.53–0.64 on the structural maps and 0.57–0.75 on the random ones (§35.4; the text's own "K^0.53–0.64"). Measured from K 100, its effective exponent log(B_P(K)/B_P(100)) / log(K/100) on the structural maps is 0.78–0.84 at K 250 and falls to 0.53–0.64 at K 5000 (MetaQA PHG: 0.63 at K 2000). On MuSiQue it is 0.82 at K 250, 0.70 at K 2000 (258.9 against 31.8) and 0.63 at K 5000 (366.8). The rule is concave in log K, not flat.
  - *"On MuSiQue the last required gold shard grows with slope around 0.80–0.99"*: the slope of LO (the ES position of the last gold shard) is 0.80 on MuSiQue. The upper end of the range is MetaQA's (0.90 on PHG over K ≤ 2000, 0.99 on Mt-KaHyPar); on SQuAD it is 0.41 and 0.43 (§35.6).
  - LO's growth is not the fan-out a count needs. LO is one query's last gold shard, and on MuSiQue at K 5000 its mean (228.7) is below the rule's mean B_P (366.8); the losses there are in the tail. The need, the smallest fixed B_P within 0.01 of unrouted at B_N 1000, grows with slope 0.71 on MuSiQue and 0.60 on MetaQA Mt-KaHyPar, against the rule's 0.63 and 0.53. It grows more slowly than the rule on MetaQA PHG (0.25 over K ≤ 2000) and on SQuAD (0.38 and 0.39) (§35.6).
- **Status.** ES·KNEE_SDIV is the carried-forward development candidate, and the α arms are development variants of its count. Nothing is frozen and nothing held out was read. The one final held-out evaluation is still the user's step.

**36.2 Design** (stated in the harness's docstring before any result of it was seen).
- **The count.** B_P^α(q, K) = min(K, ⌈KNEE_SDIV(q; P_100) · (K/100)^α⌉) with α ∈ {0.65, 0.75, 0.85, 1.0}. KNEE_SDIV(q; P_100) is §35's count on the same partitioner's K = 100 map (for RAND<s>, the same seed's map). The text's K0 is this reference granularity, K_REF = 100, not the RRF constant K0 = 60. A query without localised evidence contacts every shard, as in §35; none occurs.
- **Exact arithmetic.** For each K the harness builds a table ⌈b0 · (K/100)^α⌉, b0 = 1 … 100, in Decimal arithmetic (precision 60, then ROUND_CEILING). For α = 1.0 it is exact, since K/100 is a finite decimal (2.02, 2.5, 4.32, 5, 10, 11.75, 20, 50). The closest non-integral product lies 9.79e−4 from an integer (K 5000, α 0.75; the harness asserts a distance above 1e−30), and float64 arithmetic gives the same tables at every K. Because b0 ≤ 100, the cap K never binds strictly.
- **Why K_REF = 100.** It is the coarsest grid K and the same constant on every dataset. Every cell has K ≥ 100, so each query's count is non-decreasing in α (asserted). At K = 100 every α arm is ES·KNEE_SDIV exactly (asserted), so §35's K 100 losses remain by construction. The criteria below therefore start at K 250.
- **Fixed** (unchanged and reproduced, 36.3):
  - IR_L1, the served order O and the serving rule;
  - the router's order O' and the ES ranking on the K map;
  - §35's shard maps: 97 cells (MetaQA 34, MuSiQue 28, SQuAD 35), with MetaQA PHG K 5000 absent;
  - the node budgets B_N ∈ {100, 250, 500, 1000, 2000, 5000};
  - the population c7f70806….
- **Reference arm.** §35's ES·KNEE_SDIV, with KNEE_SDIV on the K map.
- **Measured** per cell and arm:
  - exact counts of queries with ALL gold served at each B_N, against unrouted and against ES·KNEE_SDIV (paired, descriptive McNemar);
  - the queries lost to reach (B_P < LO) and to crowding (LO ≤ HI(B_N) < B_P), per hop;
  - B_P, ρ = B_P/K, CMASS (the nodes inside the contacted shards) and CMASS/N;
  - the per-shard load;
  - structural against balanced random at the same K, paired per query under the same arm.
- **Criteria** (fixed in the summary's docstring before the full records were read):
  - An arm holds tol at B_N on a family if its ALL count is ≥ unrouted − tol × rows at every K ≥ 250, for tol ∈ {0.01, 0.005, 0}. The grid K and the native K both count when ≥ 250 (SQuAD's native K 202 does not); a structural family skips its absent cell; RAND is judged on the mean over the seeds and on every seed.
  - "B_P/K ↓" means the mean ρ decreases strictly along the grid K. With K_REF fixed, the α rule gives ρ ≈ (b0/100) · (K/100)^(α−1). So α = 1.0 keeps ρ at its K 100 value by construction and cannot meet this criterion, and α = 0.75 is the text's K^−0.25.
- **Deployment.** The α rule needs KNEE_SDIV on the K = 100 map, so the router holds that map's node → shard table (N entries) besides the K map's. This is router-side state; no global order is involved. Both maps of a structural/random pair scale by the same factor, so their fan-out ratio stays near its K = 100 value.

**36.3 Runs, identities and reproduction.** The three harness runs were sequential (SQuAD, MetaQA, MuSiQue), one heavy process at a time. Each run subclasses §35's harness, reruns it in full, and then adds the α arms.

| dataset | rows | N | cells | seconds (loop / total) | peak RSS MB | added ms per row, all cells (mean / p95) | queries without evidence | record json / npz sha | reproduces scale v1 json sha |
|---|---|---|---|---|---|---|---|---|---|
| MetaQA | 1,998 | 43,234 | 34 | 254.0 / 270.5 | 826.1 | 10.201 / 13.659 | 0 | `91cefe4cc0b8` / `695d59783a1a` | `6c50b404624e` |
| MuSiQue | 2,000 | 117,534 | 28 | 496.7 / 747.2 | 1,379.0 | 25.721 / 57.559 | 0 | `1ad319daadeb` / `62a9d3a8d372` | `4ff9cfcebaa9` |
| SQuAD | 2,000 | 20,233 | 35 | 302.5 / 360.3 | 535.7 | 22.331 / 30.18 | 0 | `2233a85be246` / `70f07c2740d9` | `382aa1dc6468` |

- **Added time.** "Added ms per row" is the per-row time of the K-aware addition: the router's order O', the ES order on every map and the α counts. It covers all of the row's cells together.
- **Reproduction of §35.** Each run asserted §35's own identities (§35.3). It also asserted that every per-query array equals the scale v1 record's npz:
  - LO, HI, CNT, CMASS, PSTAR, ESTAR, NGP, NSEEDP, NEV, XC, RDS and PRG;
  - the query-side ENT, RDN, OQLEN and HOPS;
  - the runner's IR_L1 positions;
  - MSUM and LOAD over the full population.

  The record pins the scale v1 record it reproduces by json sha (last column), npz sha and harness sha.
- **The α counts.** Per row and cell, the recomputed ES order gives §35's LO and CMASS. At K = 100 every α arm's counts, masses and load equal ES·KNEE_SDIV's. Every count lies in [1, K] and is non-decreasing in α. Each query's evidence status is the same in every cell, and no query lacks localised evidence.
- **The summary** (`kscale_SUMMARY__v1.json`) recomputes every count from the per-query arrays. It asserts each one equal to the harness's own diagnostics: ALL gold served, unrouted, lost to reach, lost to crowding, and the B_P and CMASS sums. In every cell and at every B_N, contacting every shard reproduces unrouted for every query.
- **Unrouted** serves ALL gold, at B_N 100 / 250 / 500 / 1000 / 2000 / 5000, for:

- MetaQA 763 / 855 / 919 / 1,015 / 1,110 / 1,271 of 1,998 rows;
- MuSiQue 1,279 / 1,483 / 1,586 / 1,679 / 1,760 / 1,855 of 2,000 rows;
- SQuAD 1,971 / 1,991 / 1,994 / 1,999 / 2,000 / 2,000 of 2,000 rows.

**36.4 The direction's test in one table.** The table gives each arm's Δ at B_N 1000 (the ALL-gold count minus unrouted, as a share of the rows), with the mean ρ = B_P/K in brackets. It covers K = 100 and the three largest K of each structural family. At K = 100 every arm is ES·KNEE_SDIV by construction.

| dataset | family | K | ES·KNEE_SDIV | α 0.65 | α 0.75 | α 0.85 | α 1.0 |
|---|---|---|---|---|---|---|---|
| MetaQA | PHG | 100 | −0.015 [0.284] | −0.015 [0.284] | −0.015 [0.284] | −0.015 [0.284] | −0.015 [0.284] |
| MetaQA | PHG | 500 | +0.047 [0.202] | +0.063 [0.163] | +0.055 [0.191] | +0.044 [0.224] | +0.033 [0.284] |
| MetaQA | PHG | 1000 | +0.044 [0.146] | +0.055 [0.128] | +0.042 [0.160] | +0.031 [0.202] | +0.022 [0.284] |
| MetaQA | PHG | 2000 | +0.041 [0.094] | +0.044 [0.100] | +0.037 [0.135] | +0.026 [0.182] | +0.013 [0.284] |
| MetaQA | MTK | 100 | +0.002 [0.287] | +0.002 [0.287] | +0.002 [0.287] | +0.002 [0.287] | +0.002 [0.287] |
| MetaQA | MTK | 1000 | +0.012 [0.148] | +0.014 [0.129] | +0.010 [0.162] | +0.013 [0.203] | +0.011 [0.287] |
| MetaQA | MTK | 2000 | −0.007 [0.094] | −0.008 [0.101] | −0.005 [0.136] | +0.005 [0.183] | +0.005 [0.287] |
| MetaQA | MTK | 5000 | −0.014 [0.046] | −0.006 [0.073] | +0.005 [0.108] | +0.007 [0.160] | +0.008 [0.287] |
| MuSiQue | PHG | 100 | −0.011 [0.318] | −0.011 [0.318] | −0.011 [0.318] | −0.011 [0.318] | −0.011 [0.318] |
| MuSiQue | PHG | 1175 (native) | −0.008 [0.169] | −0.012 [0.135] | −0.010 [0.172] | −0.008 [0.220] | −0.002 [0.318] |
| MuSiQue | PHG | 2000 | −0.010 [0.129] | −0.014 [0.112] | −0.009 [0.151] | −0.005 [0.203] | +0.002 [0.318] |
| MuSiQue | PHG | 5000 | −0.015 [0.073] | −0.012 [0.081] | −0.005 [0.120] | −0.003 [0.177] | +0.003 [0.318] |
| SQuAD | PHG | 100 | −0.002 [0.350] | −0.002 [0.350] | −0.002 [0.350] | −0.002 [0.350] | −0.002 [0.350] |
| SQuAD | PHG | 1000 | −0.002 [0.177] | −0.003 [0.157] | −0.002 [0.197] | −0.001 [0.248] | −0.001 [0.350] |
| SQuAD | PHG | 2000 | −0.002 [0.128] | −0.003 [0.123] | −0.002 [0.166] | −0.002 [0.223] | −0.001 [0.350] |
| SQuAD | PHG | 5000 | −0.002 [0.084] | −0.002 [0.089] | −0.002 [0.132] | −0.002 [0.195] | 0 [0.350] |
| SQuAD | MTK | 100 | −0.002 [0.330] | −0.002 [0.330] | −0.002 [0.330] | −0.002 [0.330] | −0.002 [0.330] |
| SQuAD | MTK | 1000 | −0.002 [0.172] | −0.002 [0.148] | −0.002 [0.186] | −0.001 [0.234] | −0.001 [0.330] |
| SQuAD | MTK | 2000 | −0.002 [0.124] | −0.003 [0.116] | −0.001 [0.156] | −0.001 [0.211] | −0.001 [0.330] |
| SQuAD | MTK | 5000 | −0.002 [0.080] | −0.002 [0.084] | −0.002 [0.124] | −0.002 [0.184] | 0 [0.330] |

The row counts below are exact differences from unrouted at B_N 1000.
- **MuSiQue PHG** is where §35 measured the large-K erosion: ES·KNEE_SDIV loses 20 rows at K 2000 and 30 at K 5000, of 2,000.
  - α 0.75 loses 18 and 9 rows, with ρ 0.151 and 0.120 (0.318 at K 100).
  - α 0.85 loses 9 and 5 rows, with ρ 0.203 and 0.177.
  - α 1.0 gains 4 and 5 rows, with ρ at its K 100 value.
  - α 0.65 loses 27 and 23 rows. At K 2000 it contacts fewer shards than ES·KNEE_SDIV (mean 223.5 against 258.9).
  - The native K 1175 is the worst K ≥ 250 for α 0.75 (−19 rows), α 0.85 (−15) and α 1.0 (−3). ES·KNEE_SDIV loses 16 rows there.
- **MetaQA MTK** is §35's KB erosion: ES·KNEE_SDIV loses 14 rows at K 2000 and 28 at K 5000, of 1,998.
  - α 0.75: −9 and +9 rows. α 0.85: +9 and +13. α 1.0: +10 and +16. α 0.65: −16 and −11.
  - ρ at K 5000 is 0.108 (α 0.75), 0.160 (α 0.85) and 0.287 (α 1.0), against 0.046 for ES·KNEE_SDIV.
- **MetaQA PHG** is above unrouted under every arm at every K ≥ 250, by 25 to 125 rows. From K 432 up, the gain shrinks as α grows: at K 2000 it is +88 rows under α 0.65, +73 under α 0.75, +51 under α 0.85 and +25 under α 1.0 (ES·KNEE_SDIV +82).
- **SQuAD PHG and MTK** are at most 6 rows below unrouted under every arm at every K, K 100 included.

**36.5 Fan-out: B_P, ρ and the slopes.** The tables give the mean B_P per query, with the mean ρ = B_P/K in brackets. RAND is the mean over the three seeds. At K = 100 every column equals ES·KNEE_SDIV by construction.

MetaQA (N 43,234; native K 432; PHG K 5000 absent, as in §35):

| family | K | ES·KNEE_SDIV | α 0.65 | α 0.75 | α 0.85 | α 1.0 |
|---|---|---|---|---|---|---|
| PHG | 100 | 28.4 [0.284] | 28.4 [0.284] | 28.4 [0.284] | 28.4 [0.284] | 28.4 [0.284] |
| PHG | 250 | 60.7 [0.243] | 52.1 [0.208] | 56.9 [0.227] | 62.5 [0.250] | 71.3 [0.285] |
| PHG | 432 (native) | 91.5 [0.212] | 74.1 [0.172] | 85.3 [0.197] | 99.1 [0.229] | 123.2 [0.285] |
| PHG | 500 | 101.1 [0.202] | 81.5 [0.163] | 95.5 [0.191] | 112.2 [0.224] | 142.1 [0.284] |
| PHG | 1000 | 146.3 [0.146] | 127.5 [0.128] | 160.3 [0.160] | 201.8 [0.202] | 284.3 [0.284] |
| PHG | 2000 | 187.5 [0.094] | 200.0 [0.100] | 269.4 [0.135] | 363.3 [0.182] | 568.6 [0.284] |
| PHG | 5000 | absent | absent | absent | absent | absent |
| MTK | 100 | 28.7 [0.287] | 28.7 [0.287] | 28.7 [0.287] | 28.7 [0.287] | 28.7 [0.287] |
| MTK | 250 | 62.0 [0.248] | 52.5 [0.210] | 57.3 [0.229] | 63.0 [0.252] | 71.9 [0.288] |
| MTK | 432 (native) | 92.2 [0.214] | 74.7 [0.173] | 86.0 [0.199] | 99.9 [0.231] | 124.3 [0.288] |
| MTK | 500 | 101.4 [0.203] | 82.2 [0.164] | 96.3 [0.193] | 113.1 [0.226] | 143.4 [0.287] |
| MTK | 1000 | 147.6 [0.148] | 128.6 [0.129] | 161.7 [0.162] | 203.5 [0.203] | 286.7 [0.287] |
| MTK | 2000 | 188.4 [0.094] | 201.7 [0.101] | 271.6 [0.136] | 366.4 [0.183] | 573.4 [0.287] |
| MTK | 5000 | 227.6 [0.046] | 365.1 [0.073] | 539.6 [0.108] | 797.6 [0.160] | 1,433.5 [0.287] |
| RAND0–2 | 100 | 35.3 [0.353] | 35.3 [0.353] | 35.3 [0.353] | 35.3 [0.353] | 35.3 [0.353] |
| RAND0–2 | 250 | 74.1 [0.297] | 64.5 [0.258] | 70.6 [0.282] | 77.4 [0.310] | 88.5 [0.354] |
| RAND0–2 | 432 (native) | 114.8 [0.266] | 91.9 [0.213] | 105.9 [0.245] | 122.8 [0.284] | 152.9 [0.354] |
| RAND0–2 | 500 | 127.8 [0.256] | 101.0 [0.202] | 118.5 [0.237] | 139.2 [0.278] | 176.5 [0.353] |
| RAND0–2 | 1000 | 198.1 [0.198] | 158.1 [0.158] | 199.0 [0.199] | 250.3 [0.250] | 352.9 [0.353] |
| RAND0–2 | 2000 | 263.2 [0.132] | 248.1 [0.124] | 334.3 [0.167] | 450.9 [0.225] | 705.9 [0.353] |
| RAND0–2 | 5000 | 316.2 [0.063] | 449.3 [0.090] | 664.1 [0.133] | 981.8 [0.196] | 1,764.7 [0.353] |

MuSiQue (N 117,534; native K 1175):

| family | K | ES·KNEE_SDIV | α 0.65 | α 0.75 | α 0.85 | α 1.0 |
|---|---|---|---|---|---|---|
| PHG | 100 | 31.8 [0.318] | 31.8 [0.318] | 31.8 [0.318] | 31.8 [0.318] | 31.8 [0.318] |
| PHG | 250 | 67.6 [0.270] | 58.1 [0.233] | 63.6 [0.254] | 69.8 [0.279] | 79.7 [0.319] |
| PHG | 500 | 114.0 [0.228] | 91.0 [0.182] | 106.7 [0.213] | 125.3 [0.251] | 158.9 [0.318] |
| PHG | 1000 | 181.2 [0.181] | 142.5 [0.142] | 179.2 [0.179] | 225.5 [0.225] | 317.8 [0.318] |
| PHG | 1175 (native) | 198.3 [0.169] | 158.1 [0.135] | 202.2 [0.172] | 258.5 [0.220] | 373.8 [0.318] |
| PHG | 2000 | 258.9 [0.129] | 223.5 [0.112] | 301.1 [0.151] | 406.0 [0.203] | 635.6 [0.318] |
| PHG | 5000 | 366.8 [0.073] | 404.6 [0.081] | 598.1 [0.120] | 884.1 [0.177] | 1,589.0 [0.318] |
| RAND0–2 | 100 | 45.7 [0.457] | 45.7 [0.457] | 45.7 [0.457] | 45.7 [0.457] | 45.7 [0.457] |
| RAND0–2 | 250 | 98.3 [0.393] | 83.5 [0.334] | 91.4 [0.366] | 100.1 [0.401] | 114.6 [0.458] |
| RAND0–2 | 500 | 173.5 [0.347] | 130.7 [0.261] | 153.4 [0.307] | 180.1 [0.360] | 228.6 [0.457] |
| RAND0–2 | 1000 | 302.2 [0.302] | 204.8 [0.205] | 257.6 [0.258] | 324.2 [0.324] | 457.2 [0.457] |
| RAND0–2 | 1175 (native) | 341.8 [0.291] | 227.4 [0.194] | 290.7 [0.247] | 371.8 [0.316] | 537.6 [0.458] |
| RAND0–2 | 2000 | 501.9 [0.251] | 321.1 [0.161] | 432.9 [0.216] | 583.9 [0.292] | 914.5 [0.457] |
| RAND0–2 | 5000 | 824.0 [0.165] | 581.9 [0.116] | 860.2 [0.172] | 1,271.9 [0.254] | 2,286.2 [0.457] |

SQuAD (N 20,233; native K 202):

| family | K | ES·KNEE_SDIV | α 0.65 | α 0.75 | α 0.85 | α 1.0 |
|---|---|---|---|---|---|---|
| PHG | 100 | 35.0 [0.350] | 35.0 [0.350] | 35.0 [0.350] | 35.0 [0.350] | 35.0 [0.350] |
| PHG | 202 (native) | 60.4 [0.299] | 55.7 [0.276] | 59.7 [0.296] | 64.0 [0.317] | 71.0 [0.351] |
| PHG | 250 | 71.6 [0.286] | 63.9 [0.256] | 69.9 [0.280] | 76.7 [0.307] | 87.7 [0.351] |
| PHG | 500 | 114.8 [0.230] | 100.1 [0.200] | 117.4 [0.235] | 137.8 [0.276] | 174.8 [0.350] |
| PHG | 1000 | 177.2 [0.177] | 156.7 [0.157] | 197.1 [0.197] | 248.0 [0.248] | 349.7 [0.350] |
| PHG | 2000 | 256.3 [0.128] | 245.8 [0.123] | 331.2 [0.166] | 446.7 [0.223] | 699.3 [0.350] |
| PHG | 5000 | 421.2 [0.084] | 445.1 [0.089] | 658.0 [0.132] | 972.7 [0.195] | 1,748.3 [0.350] |
| MTK | 100 | 33.0 [0.330] | 33.0 [0.330] | 33.0 [0.330] | 33.0 [0.330] | 33.0 [0.330] |
| MTK | 202 (native) | 58.6 [0.290] | 52.7 [0.261] | 56.4 [0.279] | 60.5 [0.299] | 67.1 [0.332] |
| MTK | 250 | 68.1 [0.273] | 60.4 [0.242] | 66.0 [0.264] | 72.5 [0.290] | 82.8 [0.331] |
| MTK | 500 | 111.1 [0.222] | 94.5 [0.189] | 110.9 [0.222] | 130.2 [0.260] | 165.1 [0.330] |
| MTK | 1000 | 171.8 [0.172] | 148.0 [0.148] | 186.2 [0.186] | 234.3 [0.234] | 330.2 [0.330] |
| MTK | 2000 | 247.9 [0.124] | 232.1 [0.116] | 312.8 [0.156] | 421.8 [0.211] | 660.4 [0.330] |
| MTK | 5000 | 400.6 [0.080] | 420.4 [0.084] | 621.4 [0.124] | 918.5 [0.184] | 1,650.9 [0.330] |
| RAND0–2 | 100 | 47.9 [0.479] | 47.9 [0.479] | 47.9 [0.479] | 47.9 [0.479] | 47.9 [0.479] |
| RAND0–2 | 202 (native) | 85.3 [0.422] | 76.1 [0.377] | 81.6 [0.404] | 87.5 [0.433] | 97.1 [0.480] |
| RAND0–2 | 250 | 101.8 [0.407] | 87.3 [0.349] | 95.7 [0.383] | 104.8 [0.419] | 119.9 [0.479] |
| RAND0–2 | 500 | 178.6 [0.357] | 136.7 [0.273] | 160.5 [0.321] | 188.4 [0.377] | 239.3 [0.479] |
| RAND0–2 | 1000 | 302.1 [0.302] | 214.3 [0.214] | 269.6 [0.270] | 339.3 [0.339] | 478.5 [0.479] |
| RAND0–2 | 2000 | 488.1 [0.244] | 336.0 [0.168] | 453.0 [0.227] | 611.1 [0.306] | 957.0 [0.479] |
| RAND0–2 | 5000 | 765.3 [0.153] | 609.0 [0.122] | 900.2 [0.180] | 1,331.0 [0.266] | 2,392.5 [0.479] |

**Slopes.** The next table gives the least-squares slope of log(mean B_P) on log K over the grid K present; the native K is off-grid. In parentheses: whether the mean ρ decreases strictly along those K. "RAND0–2 on the K of PHG" restricts RAND to the grid K of a family with an absent cell.

| dataset | family | grid K used | ES·KNEE_SDIV | α 0.65 | α 0.75 | α 0.85 | α 1.0 |
|---|---|---|---|---|---|---|---|
| MetaQA | PHG | 100, 250, 500, 1000, 2000 | 0.64 (yes) | 0.65 (yes) | 0.75 (yes) | 0.85 (yes) | 1.00 (no) |
| MetaQA | MTK | 100, 250, 500, 1000, 2000, 5000 | 0.53 (yes) | 0.65 (yes) | 0.75 (yes) | 0.85 (yes) | 1.00 (no) |
| MetaQA | RAND0–2 | 100, 250, 500, 1000, 2000, 5000 | 0.57 (yes) | 0.65 (yes) | 0.75 (yes) | 0.85 (yes) | 1.00 (no) |
| MetaQA | RAND0–2 on the K of PHG | 100, 250, 500, 1000, 2000 | 0.68 (yes) | 0.65 (yes) | 0.75 (yes) | 0.85 (yes) | 1.00 (no) |
| MuSiQue | PHG | 100, 250, 500, 1000, 2000, 5000 | 0.63 (yes) | 0.65 (yes) | 0.75 (yes) | 0.85 (yes) | 1.00 (no) |
| MuSiQue | RAND0–2 | 100, 250, 500, 1000, 2000, 5000 | 0.75 (yes) | 0.65 (yes) | 0.75 (yes) | 0.85 (yes) | 1.00 (no) |
| SQuAD | PHG | 100, 250, 500, 1000, 2000, 5000 | 0.63 (yes) | 0.65 (yes) | 0.75 (yes) | 0.85 (yes) | 1.00 (no) |
| SQuAD | MTK | 100, 250, 500, 1000, 2000, 5000 | 0.63 (yes) | 0.65 (yes) | 0.75 (yes) | 0.85 (yes) | 1.00 (no) |
| SQuAD | RAND0–2 | 100, 250, 500, 1000, 2000, 5000 | 0.72 (yes) | 0.65 (yes) | 0.75 (yes) | 0.85 (yes) | 1.00 (no) |

**Where an α arm contacts fewer shards than ES·KNEE_SDIV** at K > 100 (the exact B_P sums of the records' diagnostics; for RAND, summed over the three seeds):

- MetaQA PHG: α 0.65 at K 250, 432, 500, 1000; α 0.75 at K 250, 432, 500; α 0.85 at no K; α 1.0 at no K.
- MetaQA MTK: α 0.65 at K 250, 432, 500, 1000; α 0.75 at K 250, 432, 500; α 0.85 at no K; α 1.0 at no K.
- MetaQA RAND0–2: α 0.65 at K 250, 432, 500, 1000, 2000; α 0.75 at K 250, 432, 500; α 0.85 at no K; α 1.0 at no K.
- MuSiQue PHG: α 0.65 at K 250, 500, 1000, 1175, 2000; α 0.75 at K 250, 500, 1000; α 0.85 at no K; α 1.0 at no K.
- MuSiQue RAND0–2: α 0.65 at K 250, 500, 1000, 1175, 2000, 5000; α 0.75 at K 250, 500, 1000, 1175, 2000; α 0.85 at no K; α 1.0 at no K.
- SQuAD PHG: α 0.65 at K 202, 250, 500, 1000, 2000; α 0.75 at K 202, 250; α 0.85 at no K; α 1.0 at no K.
- SQuAD MTK: α 0.65 at K 202, 250, 500, 1000, 2000; α 0.75 at K 202, 250, 500; α 0.85 at no K; α 1.0 at no K.
- SQuAD RAND0–2: α 0.65 at K 202, 250, 500, 1000, 2000, 5000; α 0.75 at K 202, 250, 500, 1000, 2000; α 0.85 at no K; α 1.0 at no K.

- **Slopes.** Each α arm's slope is its α (0.65, 0.75, 0.85, 1.00) on every family, since its count is the K = 100 count times (K/100)^α, rounded up. ES·KNEE_SDIV's slopes are those of §35.4: 0.53–0.64 on the structural maps and 0.57–0.75 on the random ones.
- **ρ.** The mean ρ decreases strictly along the grid K under ES·KNEE_SDIV and under α 0.65, 0.75 and 0.85, on every family. Under α 1.0 it stays at 0.284–0.285 on MetaQA PHG, 0.287–0.288 on MetaQA MTK, 0.318–0.319 on MuSiQue PHG, 0.350–0.351 on SQuAD PHG and 0.330–0.332 on SQuAD MTK.
- **At K 5000** the mean ρ is 0.073, 0.081, 0.120, 0.177 and 0.318 on MuSiQue PHG (ES·KNEE_SDIV, then α 0.65 to 1.0), and 0.046, 0.073, 0.108, 0.160 and 0.287 on MetaQA MTK. α 1.0 contacts 1,589.0 shards per query on MuSiQue PHG and 1,433.5 on MetaQA MTK, against 366.8 and 227.6 under ES·KNEE_SDIV.
- **Crossing.** On the structural maps, ES·KNEE_SDIV's effective exponent from K 100 is 0.78–0.84 at K 250 and 0.53–0.64 at the largest K (36.1).
  - So α 0.65 and α 0.75 contact fewer shards than ES·KNEE_SDIV at the smaller K and more at the larger K (the list above).
  - α 0.85 and α 1.0 never contact fewer.
  - On the random maps of MuSiQue and SQuAD, α 0.65 contacts fewer at every K > 100.

**36.6 Retrieval R_ALL(B_N 1000, K).** The tables give the ALL-gold count minus unrouted at B_N 1000, as a share of the rows (3 decimals). RAND is the mean over the seeds.

Every arm equals ES·KNEE_SDIV at K = 100. Its K = 100 values at every B_N (§35.5) therefore remain under every α:

| dataset | family | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| MetaQA | PHG | −0.029 | −0.030 | −0.010 | −0.015 | −0.014 | −0.046 |
| MetaQA | MTK | −0.019 | −0.019 | +0.002 | +0.002 | +0.004 | −0.030 |
| MetaQA | RAND0–2 | −0.101 | −0.134 | −0.155 | −0.197 | −0.238 | −0.308 |
| MuSiQue | PHG | +0.001 | −0.002 | −0.004 | −0.011 | −0.018 | −0.033 |
| MuSiQue | RAND0–2 | −0.024 | −0.085 | −0.116 | −0.148 | −0.176 | −0.210 |
| SQuAD | PHG | 0 | −0.001 | −0.001 | −0.002 | −0.003 | −0.003 |
| SQuAD | MTK | 0 | 0 | −0.001 | −0.002 | −0.002 | −0.002 |
| SQuAD | RAND0–2 | −0.001 | −0.008 | −0.007 | −0.009 | −0.010 | −0.010 |

MetaQA:

| family | K | ES·KNEE_SDIV | α 0.65 | α 0.75 | α 0.85 | α 1.0 |
|---|---|---|---|---|---|---|
| PHG | 100 | −0.015 | −0.015 | −0.015 | −0.015 | −0.015 |
| PHG | 250 | +0.039 | +0.045 | +0.045 | +0.043 | +0.035 |
| PHG | 432 (native) | +0.046 | +0.057 | +0.049 | +0.042 | +0.033 |
| PHG | 500 | +0.047 | +0.063 | +0.055 | +0.044 | +0.033 |
| PHG | 1000 | +0.044 | +0.055 | +0.042 | +0.031 | +0.022 |
| PHG | 2000 | +0.041 | +0.044 | +0.037 | +0.026 | +0.013 |
| PHG | 5000 | absent | absent | absent | absent | absent |
| MTK | 100 | +0.002 | +0.002 | +0.002 | +0.002 | +0.002 |
| MTK | 250 | +0.039 | +0.042 | +0.042 | +0.041 | +0.034 |
| MTK | 432 (native) | +0.032 | +0.039 | +0.034 | +0.027 | +0.020 |
| MTK | 500 | +0.036 | +0.044 | +0.038 | +0.030 | +0.023 |
| MTK | 1000 | +0.012 | +0.014 | +0.010 | +0.013 | +0.011 |
| MTK | 2000 | −0.007 | −0.008 | −0.005 | +0.005 | +0.005 |
| MTK | 5000 | −0.014 | −0.006 | +0.005 | +0.007 | +0.008 |
| RAND0–2 | 100 | −0.197 | −0.197 | −0.197 | −0.197 | −0.197 |
| RAND0–2 | 250 | −0.100 | −0.111 | −0.101 | −0.093 | −0.081 |
| RAND0–2 | 432 (native) | −0.092 | −0.105 | −0.097 | −0.088 | −0.071 |
| RAND0–2 | 500 | −0.086 | −0.098 | −0.088 | −0.079 | −0.057 |
| RAND0–2 | 1000 | −0.075 | −0.090 | −0.073 | −0.059 | −0.036 |
| RAND0–2 | 2000 | −0.070 | −0.070 | −0.056 | −0.036 | −0.011 |
| RAND0–2 | 5000 | −0.069 | −0.045 | −0.023 | −0.002 | +0.005 |

MuSiQue:

| family | K | ES·KNEE_SDIV | α 0.65 | α 0.75 | α 0.85 | α 1.0 |
|---|---|---|---|---|---|---|
| PHG | 100 | −0.011 | −0.011 | −0.011 | −0.011 | −0.011 |
| PHG | 250 | −0.002 | −0.004 | −0.003 | −0.003 | −0.001 |
| PHG | 500 | −0.006 | −0.010 | −0.004 | −0.003 | −0.001 |
| PHG | 1000 | −0.006 | −0.012 | −0.008 | −0.004 | −0.001 |
| PHG | 1175 (native) | −0.008 | −0.012 | −0.010 | −0.008 | −0.002 |
| PHG | 2000 | −0.010 | −0.014 | −0.009 | −0.005 | +0.002 |
| PHG | 5000 | −0.015 | −0.012 | −0.005 | −0.003 | +0.003 |
| RAND0–2 | 100 | −0.148 | −0.148 | −0.148 | −0.148 | −0.148 |
| RAND0–2 | 250 | −0.093 | −0.122 | −0.104 | −0.086 | −0.065 |
| RAND0–2 | 500 | −0.061 | −0.093 | −0.073 | −0.055 | −0.032 |
| RAND0–2 | 1000 | −0.043 | −0.073 | −0.053 | −0.038 | −0.023 |
| RAND0–2 | 1175 (native) | −0.045 | −0.071 | −0.051 | −0.039 | −0.021 |
| RAND0–2 | 2000 | −0.037 | −0.058 | −0.045 | −0.027 | −0.013 |
| RAND0–2 | 5000 | −0.027 | −0.046 | −0.027 | −0.015 | −0.005 |

SQuAD:

| family | K | ES·KNEE_SDIV | α 0.65 | α 0.75 | α 0.85 | α 1.0 |
|---|---|---|---|---|---|---|
| PHG | 100 | −0.002 | −0.002 | −0.002 | −0.002 | −0.002 |
| PHG | 202 (native) | −0.001 | −0.001 | −0.001 | −0.001 | 0 |
| PHG | 250 | −0.001 | −0.003 | −0.001 | −0.001 | −0.001 |
| PHG | 500 | −0.001 | −0.001 | −0.001 | −0.001 | 0 |
| PHG | 1000 | −0.002 | −0.003 | −0.002 | −0.001 | −0.001 |
| PHG | 2000 | −0.002 | −0.003 | −0.002 | −0.002 | −0.001 |
| PHG | 5000 | −0.002 | −0.002 | −0.002 | −0.002 | 0 |
| MTK | 100 | −0.002 | −0.002 | −0.002 | −0.002 | −0.002 |
| MTK | 202 (native) | −0.002 | −0.003 | −0.003 | −0.002 | −0.002 |
| MTK | 250 | −0.002 | −0.003 | −0.002 | −0.002 | −0.001 |
| MTK | 500 | −0.001 | −0.003 | −0.001 | −0.001 | 0 |
| MTK | 1000 | −0.002 | −0.002 | −0.002 | −0.001 | −0.001 |
| MTK | 2000 | −0.002 | −0.003 | −0.001 | −0.001 | −0.001 |
| MTK | 5000 | −0.002 | −0.002 | −0.002 | −0.002 | 0 |
| RAND0–2 | 100 | −0.009 | −0.009 | −0.009 | −0.009 | −0.009 |
| RAND0–2 | 202 (native) | −0.006 | −0.009 | −0.008 | −0.007 | −0.005 |
| RAND0–2 | 250 | −0.006 | −0.007 | −0.006 | −0.005 | −0.004 |
| RAND0–2 | 500 | −0.003 | −0.005 | −0.004 | −0.002 | −0.001 |
| RAND0–2 | 1000 | −0.002 | −0.003 | −0.002 | −0.002 | −0.002 |
| RAND0–2 | 2000 | −0.001 | −0.002 | −0.002 | −0.002 | −0.001 |
| RAND0–2 | 5000 | −0.002 | −0.003 | −0.002 | −0.001 | −0.001 |

**Per hop** at B_N 1000. The tables give ALL-gold counts for the structural cells at the three largest K; each entry lists the hops in order.

MetaQA (hop1 666 rows, unrouted 666; hop2 666 rows, unrouted 187; hop3 666 rows, unrouted 162):

| family | K | ES·KNEE_SDIV (hop1 / hop2 / hop3) | α 0.65 (hop1 / hop2 / hop3) | α 0.75 (hop1 / hop2 / hop3) | α 0.85 (hop1 / hop2 / hop3) | α 1.0 (hop1 / hop2 / hop3) |
|---|---|---|---|---|---|---|
| PHG | 500 | 663 / 218 / 228 | 663 / 228 / 249 | 663 / 227 / 234 | 663 / 215 / 224 | 665 / 209 / 206 |
| PHG | 1000 | 664 / 215 / 224 | 663 / 217 / 244 | 665 / 210 / 224 | 666 / 206 / 205 | 666 / 205 / 188 |
| PHG | 2000 | 665 / 213 / 219 | 666 / 217 / 220 | 666 / 214 / 208 | 666 / 210 / 190 | 666 / 196 / 178 |
| MTK | 1000 | 664 / 200 / 174 | 664 / 202 / 176 | 665 / 199 / 171 | 666 / 203 / 172 | 666 / 198 / 172 |
| MTK | 2000 | 666 / 179 / 156 | 666 / 178 / 155 | 666 / 182 / 158 | 666 / 197 / 161 | 666 / 194 / 165 |
| MTK | 5000 | 666 / 177 / 144 | 666 / 188 / 150 | 666 / 200 / 158 | 666 / 195 / 167 | 666 / 192 / 173 |

MuSiQue (hop2 1,364 rows, unrouted 1,244; hop3 442 rows, unrouted 336; hop4 194 rows, unrouted 99):

| family | K | ES·KNEE_SDIV (hop2 / hop3 / hop4) | α 0.65 (hop2 / hop3 / hop4) | α 0.75 (hop2 / hop3 / hop4) | α 0.85 (hop2 / hop3 / hop4) | α 1.0 (hop2 / hop3 / hop4) |
|---|---|---|---|---|---|---|
| PHG | 1175 (native) | 1,235 / 335 / 93 | 1,232 / 333 / 91 | 1,233 / 335 / 92 | 1,233 / 334 / 97 | 1,240 / 338 / 98 |
| PHG | 2000 | 1,235 / 330 / 94 | 1,231 / 328 / 93 | 1,235 / 330 / 96 | 1,239 / 334 / 97 | 1,243 / 340 / 100 |
| PHG | 5000 | 1,228 / 328 / 93 | 1,229 / 331 / 96 | 1,237 / 335 / 98 | 1,238 / 336 / 100 | 1,244 / 339 / 101 |

- **MuSiQue PHG**, in exact rows against unrouted at B_N 1000 (K 250, 500, 1000, 1175, 2000, 5000):
  - ES·KNEE_SDIV: −4, −11, −11, −16, −20, −30.
  - α 0.65: −8, −19, −24, −23, −27, −23. It contacts fewer shards than ES·KNEE_SDIV up to K 2000 and loses more rows there.
  - α 0.75: −6, −8, −15, −19, −18, −9. It is within 4 rows of ES·KNEE_SDIV up to K 2000 and 21 rows above it at K 5000.
  - α 0.85: −6, −6, −8, −15, −9, −5.
  - α 1.0: −2, −2, −2, −3, +4, +5.
- **Per hop on MuSiQue at K 5000**, ES·KNEE_SDIV's 30 lost rows are 16 two-hop, 8 three-hop and 6 four-hop rows. α 1.0 is at or above unrouted on every hop (1,244 / 339 / 101 against 1,244 / 336 / 99).
- **MetaQA MTK.** ES·KNEE_SDIV goes from +23 rows at K 1000 to −14 at K 2000 and −28 at K 5000.
  - α 0.85 and α 1.0 are above unrouted at every K ≥ 250, by at least 9 and 10 rows.
  - α 0.75 is below it only at K 2000 (−9), and α 0.65 at K 2000 and 5000 (−16, −11).
  - Hop 1 is at unrouted under every arm at K 2000 and 5000. ES·KNEE_SDIV's K 5000 loss is 10 two-hop and 18 three-hop rows.
- **MetaQA PHG.** The gain over unrouted is on hops 2 and 3. At K 1000, hop 3 has 244 rows under α 0.65 and 188 under α 1.0 (unrouted 162); hop 2 has 217 and 205 (unrouted 187).
- **RAND** (the mean over the seeds). The ALL count does not fall as α grows, at any K > 100 on any dataset. At K 5000, α 1.0 is at +0.005 on MetaQA, −0.005 on MuSiQue and −0.001 on SQuAD.

**36.7 Which α holds R_ALL over K ≥ 250.** The tables are per family and B_N.
- **Worst.** Each arm's worst Δ over K ≥ 250 (the grid and the native K), with the K where it occurs. Among ties, that is the smallest such K.
- **Holds.** The α arms that hold tol ∈ {0.01, 0.005, 0}: an ALL count ≥ unrouted − tol × rows at every such K. "all" means all four α arms, "base" that ES·KNEE_SDIV also holds, and "none" that no α arm holds.
- **RAND.** The criterion is on the mean over the seeds. "(every seed: …)" is shown where requiring it of each seed changes the list.

MetaQA:

| family | B_N | ES·KNEE_SDIV worst (K) | α 0.65 worst (K) | α 0.75 worst (K) | α 0.85 worst (K) | α 1.0 worst (K) | holds tol 0.01 | holds tol 0.005 | holds tol 0 |
|---|---|---|---|---|---|---|---|---|---|
| PHG | 100 | −0.001 (250) | −0.001 (432) | 0 (432) | −0.001 (250) | 0 (250) | all; base | all; base | 0.75, 1.0 |
| PHG | 250 | 0 (2000) | 0 (2000) | 0 (2000) | 0 (1000) | 0 (1000) | all; base | all; base | all; base |
| PHG | 500 | +0.031 (2000) | +0.028 (2000) | +0.011 (2000) | +0.004 (2000) | +0.002 (2000) | all; base | all; base | all; base |
| PHG | 1000 | +0.039 (250) | +0.044 (2000) | +0.037 (2000) | +0.026 (2000) | +0.013 (2000) | all; base | all; base | all; base |
| PHG | 2000 | +0.038 (2000) | +0.041 (2000) | +0.046 (1000) | +0.043 (1000) | +0.028 (2000) | all; base | all; base | all; base |
| PHG | 5000 | −0.013 (2000) | −0.001 (2000) | +0.007 (1000) | +0.010 (1000) | +0.012 (1000) | all | all | 0.75, 0.85, 1.0 |
| MTK | 100 | 0 (250) | 0 (432) | 0 (432) | 0 (432) | 0 (250) | all; base | all; base | all; base |
| MTK | 250 | −0.001 (5000) | 0 (2000) | 0 (2000) | 0 (1000) | 0 (1000) | all; base | all; base | all |
| MTK | 500 | +0.003 (5000) | −0.001 (5000) | +0.002 (5000) | +0.002 (5000) | +0.001 (2000) | all; base | all; base | 0.75, 0.85, 1.0; base |
| MTK | 1000 | −0.014 (5000) | −0.008 (2000) | −0.005 (2000) | +0.005 (2000) | +0.005 (2000) | all | 0.75, 0.85, 1.0 | 0.85, 1.0 |
| MTK | 2000 | −0.047 (5000) | −0.031 (2000) | −0.024 (2000) | −0.008 (2000) | −0.002 (2000) | 0.85, 1.0 | 1.0 | none |
| MTK | 5000 | −0.125 (5000) | −0.094 (5000) | −0.079 (2000) | −0.058 (2000) | −0.040 (2000) | none | none | none |
| RAND0–2 | 100 | −0.006 (250) | −0.011 (250) | −0.005 (250) | −0.003 (250) | −0.002 (250) | 0.75, 0.85, 1.0; base | 0.85, 1.0 (every seed: 1.0) | none |
| RAND0–2 | 250 | −0.035 (250) | −0.043 (250) | −0.035 (250) | −0.029 (250) | −0.021 (250) | none | none | none |
| RAND0–2 | 500 | −0.058 (250) | −0.068 (250) | −0.058 (250) | −0.052 (250) | −0.041 (250) | none | none | none |
| RAND0–2 | 1000 | −0.100 (250) | −0.111 (250) | −0.101 (250) | −0.093 (250) | −0.081 (250) | none | none | none |
| RAND0–2 | 2000 | −0.141 (250) | −0.153 (250) | −0.142 (250) | −0.133 (250) | −0.120 (250) | none | none | none |
| RAND0–2 | 5000 | −0.214 (250) | −0.227 (250) | −0.215 (250) | −0.206 (250) | −0.190 (250) | none | none | none |

MuSiQue:

| family | B_N | ES·KNEE_SDIV worst (K) | α 0.65 worst (K) | α 0.75 worst (K) | α 0.85 worst (K) | α 1.0 worst (K) | holds tol 0.01 | holds tol 0.005 | holds tol 0 |
|---|---|---|---|---|---|---|---|---|---|
| PHG | 100 | 0 (250) | 0 (250) | 0 (250) | 0 (250) | 0 (250) | all; base | all; base | all; base |
| PHG | 250 | −0.001 (1000) | −0.001 (500) | 0 (500) | 0 (250) | 0 (250) | all; base | all; base | 0.75, 0.85, 1.0 |
| PHG | 500 | −0.004 (2000) | −0.004 (2000) | −0.003 (1000) | −0.002 (1175) | −0.002 (1175) | all; base | all; base | none |
| PHG | 1000 | −0.015 (5000) | −0.014 (2000) | −0.010 (1175) | −0.008 (1175) | −0.002 (1175) | 0.75, 0.85, 1.0 | 1.0 | none |
| PHG | 2000 | −0.036 (5000) | −0.028 (2000) | −0.022 (2000) | −0.017 (1175) | −0.009 (1175) | 1.0 | none | none |
| PHG | 5000 | −0.067 (5000) | −0.059 (5000) | −0.044 (1175) | −0.031 (1175) | −0.019 (500) | none | none | none |
| RAND0–2 | 100 | −0.004 (250) | −0.004 (250) | −0.002 (250) | −0.001 (250) | 0 (500) | all; base | all; base (every seed: all) | 1.0 |
| RAND0–2 | 250 | −0.036 (250) | −0.054 (250) | −0.041 (250) | −0.027 (250) | −0.015 (250) | none | none | none |
| RAND0–2 | 500 | −0.062 (250) | −0.086 (250) | −0.071 (250) | −0.055 (250) | −0.036 (250) | none | none | none |
| RAND0–2 | 1000 | −0.093 (250) | −0.122 (250) | −0.104 (250) | −0.086 (250) | −0.065 (250) | none | none | none |
| RAND0–2 | 2000 | −0.123 (250) | −0.154 (250) | −0.136 (250) | −0.118 (250) | −0.092 (250) | none | none | none |
| RAND0–2 | 5000 | −0.160 (250) | −0.196 (250) | −0.176 (250) | −0.154 (250) | −0.126 (250) | none | none | none |

SQuAD:

| family | B_N | ES·KNEE_SDIV worst (K) | α 0.65 worst (K) | α 0.75 worst (K) | α 0.85 worst (K) | α 1.0 worst (K) | holds tol 0.01 | holds tol 0.005 | holds tol 0 |
|---|---|---|---|---|---|---|---|---|---|
| PHG | 100 | 0 (250) | 0 (250) | 0 (250) | 0 (250) | 0 (250) | all; base | all; base | all; base |
| PHG | 250 | 0 (250) | 0 (250) | 0 (250) | 0 (250) | 0 (250) | all; base | all; base | all; base |
| PHG | 500 | 0 (1000) | −0.001 (250) | 0 (500) | −0.001 (2000) | 0 (500) | all; base | all; base | 0.75, 1.0; base |
| PHG | 1000 | −0.002 (1000) | −0.003 (2000) | −0.002 (1000) | −0.002 (2000) | −0.001 (1000) | all; base | all; base | none |
| PHG | 2000 | −0.003 (1000) | −0.004 (2000) | −0.003 (1000) | −0.003 (2000) | −0.002 (1000) | all; base | all; base | none |
| PHG | 5000 | −0.003 (1000) | −0.004 (2000) | −0.003 (1000) | −0.003 (2000) | −0.002 (1000) | all; base | all; base | none |
| MTK | 100 | 0 (250) | 0 (250) | 0 (250) | 0 (250) | 0 (250) | all; base | all; base | all; base |
| MTK | 250 | 0 (250) | −0.001 (250) | 0 (250) | 0 (250) | 0 (250) | all; base | all; base | 0.75, 0.85, 1.0; base |
| MTK | 500 | −0.001 (250) | −0.002 (250) | −0.001 (250) | −0.001 (250) | −0.001 (250) | all; base | all; base | none |
| MTK | 1000 | −0.002 (250) | −0.003 (250) | −0.002 (5000) | −0.002 (250) | −0.001 (250) | all; base | all; base | none |
| MTK | 2000 | −0.003 (250) | −0.004 (250) | −0.003 (5000) | −0.002 (250) | −0.002 (250) | all; base | all; base | none |
| MTK | 5000 | −0.003 (250) | −0.004 (250) | −0.003 (5000) | −0.002 (250) | −0.002 (250) | all; base | all; base | none |
| RAND0–2 | 100 | −0.001 (250) | 0 (500) | 0 (250) | 0 (250) | 0 (250) | all; base | all; base | all (every seed: 0.65, 1.0) |
| RAND0–2 | 250 | −0.003 (250) | −0.004 (250) | −0.003 (250) | −0.002 (250) | −0.001 (250) | all; base | all; base | none |
| RAND0–2 | 500 | −0.004 (250) | −0.005 (250) | −0.004 (250) | −0.002 (250) | −0.002 (250) | all; base | all; base (every seed: all) | none |
| RAND0–2 | 1000 | −0.006 (250) | −0.007 (250) | −0.006 (250) | −0.005 (250) | −0.004 (250) | all; base | 0.85, 1.0 | none |
| RAND0–2 | 2000 | −0.007 (250) | −0.008 (250) | −0.007 (250) | −0.005 (250) | −0.004 (250) | all; base | 1.0 | none |
| RAND0–2 | 5000 | −0.007 (250) | −0.008 (250) | −0.007 (250) | −0.005 (250) | −0.004 (250) | all; base | 1.0 | none |

The smallest α that holds each tolerance over K ≥ 250 at B_N 1000, and whether ES·KNEE_SDIV holds it. "(ρ not ↓)" marks an α whose mean ρ does not decrease strictly along the grid K:

| dataset | family | ES·KNEE_SDIV holds (tol 0.01 / 0.005 / 0) | smallest α holding tol 0.01 | smallest α holding tol 0.005 | smallest α holding tol 0 |
|---|---|---|---|---|---|
| MetaQA | PHG | yes / yes / yes | 0.65 | 0.65 | 0.65 |
| MetaQA | MTK | no / no / no | 0.65 | 0.75 | 0.85 |
| MetaQA | RAND0–2 | no / no / no | none | none | none |
| MuSiQue | PHG | no / no / no | 0.75 | 1.0 (ρ not ↓) | none |
| MuSiQue | RAND0–2 | no / no / no | none | none | none |
| SQuAD | PHG | yes / yes / no | 0.65 | 0.65 | none |
| SQuAD | MTK | yes / yes / no | 0.65 | 0.65 | none |
| SQuAD | RAND0–2 | yes / no / no | 0.65 | 0.85 | none |

- **B_N 1000, the five structural families together** (MetaQA PHG and MTK, MuSiQue PHG, SQuAD PHG and MTK):
  - tol 0.01: the smallest α is 0.75. MuSiQue PHG binds: α 0.75's worst there is −19 rows at K 1175, one row inside the bound of −20. α 0.65 misses there (−27 at K 2000), and so does ES·KNEE_SDIV (−30 at K 5000).
  - tol 0.005: only α 1.0 holds on MuSiQue PHG (worst −3 rows at K 1175; α 0.85 has −15 against a bound of −10). α 1.0 keeps ρ at its K = 100 value.
  - tol 0: no α holds on MuSiQue PHG or on SQuAD, where every arm's worst over K ≥ 250 is 2 to 6 rows below unrouted.
- **Smaller B_N.** At B_N 100–500, every arm, ES·KNEE_SDIV included, holds tol 0.005 on every structural family.
- **Larger B_N.**
  - At B_N 2000, only α 1.0 holds tol 0.01 on MuSiQue PHG (−17 rows at K 1175), and α 0.85 and α 1.0 hold it on MetaQA MTK.
  - At B_N 5000, no α holds tol 0.01 on MuSiQue PHG (α 1.0: −37 rows at K 500) or on MetaQA MTK (α 1.0: −79 at K 2000). The K = 100 anchor loses 66 and 59 rows there.
- **RAND.** No arm holds any tolerance on MetaQA or MuSiQue at B_N ≥ 250. On SQuAD every arm holds tol 0.01 at every B_N.

**36.8 Against the need.** The need is §35's smallest fixed B_P on the ES surface whose ALL count is ≥ unrouted − 0.01 × rows at B_N 1000. In parentheses: the smallest fixed B_P at or above unrouted. RAND's need is the mean over the seeds' needs.
- The arm columns give each arm's mean B_P ÷ the need.
- A ratio above 1 does not make an arm lossless, since its B_P varies per query. A ratio below 1 does not make it lossy.

MetaQA:

| family | K | need within 0.01 (at or above) | ES·KNEE_SDIV ÷ need | α 0.65 ÷ need | α 0.75 ÷ need | α 0.85 ÷ need | α 1.0 ÷ need |
|---|---|---|---|---|---|---|---|
| PHG | 100 | 28 (30) | 1.02 | 1.02 | 1.02 | 1.02 | 1.02 |
| PHG | 250 | 28 (29) | 2.17 | 1.86 | 2.03 | 2.23 | 2.55 |
| PHG | 432 (native) | 31 (33) | 2.95 | 2.39 | 2.75 | 3.20 | 3.98 |
| PHG | 500 | 32 (33) | 3.16 | 2.55 | 2.98 | 3.50 | 4.44 |
| PHG | 1000 | 42 (44) | 3.48 | 3.04 | 3.82 | 4.80 | 6.77 |
| PHG | 2000 | 59 (66) | 3.18 | 3.39 | 4.57 | 6.16 | 9.64 |
| MTK | 100 | 25 (26) | 1.15 | 1.15 | 1.15 | 1.15 | 1.15 |
| MTK | 250 | 26 (27) | 2.38 | 2.02 | 2.21 | 2.42 | 2.77 |
| MTK | 432 (native) | 29 (31) | 3.18 | 2.58 | 2.97 | 3.45 | 4.29 |
| MTK | 500 | 29 (30) | 3.50 | 2.83 | 3.32 | 3.90 | 4.94 |
| MTK | 1000 | 43 (50) | 3.43 | 2.99 | 3.76 | 4.73 | 6.67 |
| MTK | 2000 | 94 (282) | 2.00 | 2.15 | 2.89 | 3.90 | 6.10 |
| MTK | 5000 | 261 (405) | 0.87 | 1.40 | 2.07 | 3.06 | 5.49 |
| RAND0–2 | 100 | 92.0 (100.0) | 0.38 | 0.38 | 0.38 | 0.38 | 0.38 |
| RAND0–2 | 250 | 195.0 (241.3) | 0.38 | 0.33 | 0.36 | 0.40 | 0.45 |
| RAND0–2 | 432 (native) | 306.7 (397.3) | 0.37 | 0.30 | 0.35 | 0.40 | 0.50 |
| RAND0–2 | 500 | 320.7 (432.0) | 0.40 | 0.31 | 0.37 | 0.43 | 0.55 |
| RAND0–2 | 1000 | 518.3 (761.3) | 0.38 | 0.31 | 0.38 | 0.48 | 0.68 |
| RAND0–2 | 2000 | 692.7 (1,008.3) | 0.38 | 0.36 | 0.48 | 0.65 | 1.02 |
| RAND0–2 | 5000 | 795.7 (956.0) | 0.40 | 0.56 | 0.83 | 1.23 | 2.22 |

MuSiQue:

| family | K | need within 0.01 (at or above) | ES·KNEE_SDIV ÷ need | α 0.65 ÷ need | α 0.75 ÷ need | α 0.85 ÷ need | α 1.0 ÷ need |
|---|---|---|---|---|---|---|---|
| PHG | 100 | 33 (51) | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 |
| PHG | 250 | 47 (82) | 1.44 | 1.24 | 1.35 | 1.48 | 1.70 |
| PHG | 500 | 79 (182) | 1.44 | 1.15 | 1.35 | 1.59 | 2.01 |
| PHG | 1000 | 166 (653) | 1.09 | 0.86 | 1.08 | 1.36 | 1.91 |
| PHG | 1175 (native) | 217 (368) | 0.91 | 0.73 | 0.93 | 1.19 | 1.72 |
| PHG | 2000 | 269 (548) | 0.96 | 0.83 | 1.12 | 1.51 | 2.36 |
| PHG | 5000 | 437 (845) | 0.84 | 0.93 | 1.37 | 2.02 | 3.64 |
| RAND0–2 | 100 | 86.0 (99.7) | 0.53 | 0.53 | 0.53 | 0.53 | 0.53 |
| RAND0–2 | 250 | 192.3 (246.0) | 0.51 | 0.43 | 0.48 | 0.52 | 0.60 |
| RAND0–2 | 500 | 349.0 (463.0) | 0.50 | 0.37 | 0.44 | 0.52 | 0.66 |
| RAND0–2 | 1000 | 605.0 (923.7) | 0.50 | 0.34 | 0.43 | 0.54 | 0.76 |
| RAND0–2 | 1175 (native) | 658.7 (1,132.7) | 0.52 | 0.35 | 0.44 | 0.56 | 0.82 |
| RAND0–2 | 2000 | 920.3 (1,595.7) | 0.55 | 0.35 | 0.47 | 0.63 | 0.99 |
| RAND0–2 | 5000 | 1,608.7 (3,305.3) | 0.51 | 0.36 | 0.53 | 0.79 | 1.42 |

SQuAD:

| family | K | need within 0.01 (at or above) | ES·KNEE_SDIV ÷ need | α 0.65 ÷ need | α 0.75 ÷ need | α 0.85 ÷ need | α 1.0 ÷ need |
|---|---|---|---|---|---|---|---|
| PHG | 100 | 18 (72) | 1.94 | 1.94 | 1.94 | 1.94 | 1.94 |
| PHG | 202 (native) | 24 (80) | 2.52 | 2.32 | 2.49 | 2.67 | 2.96 |
| PHG | 250 | 27 (110) | 2.65 | 2.37 | 2.59 | 2.84 | 3.25 |
| PHG | 500 | 35 (159) | 3.28 | 2.86 | 3.35 | 3.94 | 5.00 |
| PHG | 1000 | 46 (839) | 3.85 | 3.41 | 4.29 | 5.39 | 7.60 |
| PHG | 2000 | 63 (1,833) | 4.07 | 3.90 | 5.26 | 7.09 | 11.10 |
| PHG | 5000 | 77 (1,801) | 5.47 | 5.78 | 8.54 | 12.63 | 22.70 |
| MTK | 100 | 16 (60) | 2.06 | 2.06 | 2.06 | 2.06 | 2.06 |
| MTK | 202 (native) | 21 (155) | 2.79 | 2.51 | 2.69 | 2.88 | 3.19 |
| MTK | 250 | 24 (106) | 2.84 | 2.52 | 2.75 | 3.02 | 3.45 |
| MTK | 500 | 37 (157) | 3.00 | 2.55 | 3.00 | 3.52 | 4.46 |
| MTK | 1000 | 46 (722) | 3.73 | 3.22 | 4.05 | 5.09 | 7.18 |
| MTK | 2000 | 55 (1,148) | 4.51 | 4.22 | 5.69 | 7.67 | 12.01 |
| MTK | 5000 | 72 (1,771) | 5.56 | 5.84 | 8.63 | 12.76 | 22.93 |
| RAND0–2 | 100 | 47.0 (82.7) | 1.02 | 1.02 | 1.02 | 1.02 | 1.02 |
| RAND0–2 | 202 (native) | 73.0 (156.0) | 1.17 | 1.04 | 1.12 | 1.20 | 1.33 |
| RAND0–2 | 250 | 73.7 (200.0) | 1.38 | 1.19 | 1.30 | 1.42 | 1.63 |
| RAND0–2 | 500 | 92.7 (380.7) | 1.93 | 1.48 | 1.73 | 2.03 | 2.58 |
| RAND0–2 | 1000 | 101.0 (702.0) | 2.99 | 2.12 | 2.67 | 3.36 | 4.74 |
| RAND0–2 | 2000 | 104.3 (1,156.0) | 4.68 | 3.22 | 4.34 | 5.86 | 9.17 |
| RAND0–2 | 5000 | 111.7 (2,837.0) | 6.85 | 5.45 | 8.06 | 11.92 | 21.43 |

- **MuSiQue PHG.** The need grows from 33 shards at K 100 to 437 at K 5000.
  - ES·KNEE_SDIV's mean B_P is 1.44 × the need at K 250 and 500, and 0.84 × at K 5000, where it loses 30 rows.
  - Over K ≥ 250 the α arms' ratios are 0.73–1.24 (α 0.65), 0.93–1.37 (α 0.75), 1.19–2.02 (α 0.85) and 1.70–3.64 (α 1.0).
- **MetaQA MTK.** The need is 25–43 shards up to K 1000, 94 at K 2000 and 261 at K 5000.
  - ES·KNEE_SDIV's ratio falls from 3.43 at K 1000 to 2.00 and 0.87.
  - Over K ≥ 250, α 0.75's ratio is at least 2.07, α 0.85's at least 2.42 and α 1.0's at least 2.77.
- **Ratios and losses.** At K 2000 on MetaQA MTK, α 0.75 is 2.89 × the need and loses 9 rows. At K 1175 on MuSiQue it is 0.93 × the need and loses 19 rows, inside the tolerance.
- **MetaQA PHG and SQuAD.** The need grows slowly: 28 to 59 on MetaQA PHG (K ≤ 2000), 18 to 77 on SQuAD PHG and 16 to 72 on SQuAD MTK. Every α arm's ratio grows with K. At K 5000 on SQuAD, α 1.0 contacts 22.70 (PHG) and 22.93 (MTK) times the need.
- **RAND.** On MetaQA and MuSiQue every ratio is below 1 at K ≤ 1000 (0.30–0.76). Above 1 are only α 1.0 at K 5000 (2.22 on MetaQA, 1.42 on MuSiQue), α 1.0 at K 2000 on MetaQA (1.02) and α 0.85 at K 5000 on MetaQA (1.23).

**36.9 Structural against random under the α arms.** An α arm multiplies both maps' K = 100 counts by the same table. So the structural ÷ random ratio of the total B_P (3 × the structural sum over the three seeds' sums) stays near its K = 100 value:

| dataset | family | K | ES·KNEE_SDIV | α 0.65 | α 0.75 | α 0.85 | α 1.0 |
|---|---|---|---|---|---|---|---|
| MetaQA | PHG | 100 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | PHG | 250 | 0.82 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | PHG | 432 (native) | 0.80 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | PHG | 500 | 0.79 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | PHG | 1000 | 0.74 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | PHG | 2000 | 0.71 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | MTK | 100 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | MTK | 250 | 0.84 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | MTK | 432 (native) | 0.80 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | MTK | 500 | 0.79 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | MTK | 1000 | 0.74 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | MTK | 2000 | 0.72 | 0.81 | 0.81 | 0.81 | 0.81 |
| MetaQA | MTK | 5000 | 0.72 | 0.81 | 0.81 | 0.81 | 0.81 |
| MuSiQue | PHG | 100 | 0.70 | 0.70 | 0.70 | 0.70 | 0.70 |
| MuSiQue | PHG | 250 | 0.69 | 0.70 | 0.70 | 0.70 | 0.70 |
| MuSiQue | PHG | 500 | 0.66 | 0.70 | 0.70 | 0.70 | 0.70 |
| MuSiQue | PHG | 1000 | 0.60 | 0.70 | 0.70 | 0.70 | 0.70 |
| MuSiQue | PHG | 1175 (native) | 0.58 | 0.70 | 0.70 | 0.70 | 0.70 |
| MuSiQue | PHG | 2000 | 0.52 | 0.70 | 0.70 | 0.70 | 0.70 |
| MuSiQue | PHG | 5000 | 0.45 | 0.70 | 0.70 | 0.70 | 0.70 |
| SQuAD | PHG | 100 | 0.73 | 0.73 | 0.73 | 0.73 | 0.73 |
| SQuAD | PHG | 202 (native) | 0.71 | 0.73 | 0.73 | 0.73 | 0.73 |
| SQuAD | PHG | 250 | 0.70 | 0.73 | 0.73 | 0.73 | 0.73 |
| SQuAD | PHG | 500 | 0.64 | 0.73 | 0.73 | 0.73 | 0.73 |
| SQuAD | PHG | 1000 | 0.59 | 0.73 | 0.73 | 0.73 | 0.73 |
| SQuAD | PHG | 2000 | 0.53 | 0.73 | 0.73 | 0.73 | 0.73 |
| SQuAD | PHG | 5000 | 0.55 | 0.73 | 0.73 | 0.73 | 0.73 |
| SQuAD | MTK | 100 | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 |
| SQuAD | MTK | 202 (native) | 0.69 | 0.69 | 0.69 | 0.69 | 0.69 |
| SQuAD | MTK | 250 | 0.67 | 0.69 | 0.69 | 0.69 | 0.69 |
| SQuAD | MTK | 500 | 0.62 | 0.69 | 0.69 | 0.69 | 0.69 |
| SQuAD | MTK | 1000 | 0.57 | 0.69 | 0.69 | 0.69 | 0.69 |
| SQuAD | MTK | 2000 | 0.51 | 0.69 | 0.69 | 0.69 | 0.69 |
| SQuAD | MTK | 5000 | 0.52 | 0.69 | 0.69 | 0.69 | 0.69 |

The next table gives structural − random at B_N 1000, as the mean over the seeds (exact), with each seed's value in brackets. "−0.000" and "+0.000" are nonzero means that round to zero:

| dataset | family | K | ES·KNEE_SDIV | α 0.65 | α 0.75 | α 0.85 | α 1.0 |
|---|---|---|---|---|---|---|---|
| MetaQA | PHG | 100 | +0.182 [+0.191, +0.180, +0.176] | +0.182 [+0.191, +0.180, +0.176] | +0.182 [+0.191, +0.180, +0.176] | +0.182 [+0.191, +0.180, +0.176] | +0.182 [+0.191, +0.180, +0.176] |
| MetaQA | PHG | 250 | +0.139 [+0.138, +0.134, +0.145] | +0.156 [+0.155, +0.152, +0.161] | +0.146 [+0.144, +0.141, +0.152] | +0.136 [+0.135, +0.131, +0.143] | +0.115 [+0.115, +0.111, +0.121] |
| MetaQA | PHG | 432 (native) | +0.137 [+0.140, +0.131, +0.141] | +0.162 [+0.164, +0.159, +0.163] | +0.145 [+0.147, +0.139, +0.150] | +0.129 [+0.131, +0.126, +0.131] | +0.103 [+0.109, +0.100, +0.102] |
| MetaQA | PHG | 500 | +0.133 [+0.132, +0.130, +0.138] | +0.160 [+0.159, +0.155, +0.167] | +0.143 [+0.142, +0.139, +0.147] | +0.122 [+0.122, +0.119, +0.126] | +0.090 [+0.093, +0.090, +0.088] |
| MetaQA | PHG | 1000 | +0.119 [+0.114, +0.122, +0.121] | +0.145 [+0.142, +0.146, +0.147] | +0.115 [+0.111, +0.118, +0.117] | +0.090 [+0.085, +0.094, +0.092] | +0.058 [+0.055, +0.060, +0.058] |
| MetaQA | PHG | 2000 | +0.111 [+0.111, +0.111, +0.113] | +0.114 [+0.111, +0.116, +0.117] | +0.093 [+0.092, +0.092, +0.094] | +0.061 [+0.061, +0.062, +0.061] | +0.023 [+0.024, +0.022, +0.023] |
| MetaQA | MTK | 100 | +0.198 [+0.207, +0.196, +0.192] | +0.198 [+0.207, +0.196, +0.192] | +0.198 [+0.207, +0.196, +0.192] | +0.198 [+0.207, +0.196, +0.192] | +0.198 [+0.207, +0.196, +0.192] |
| MetaQA | MTK | 250 | +0.139 [+0.138, +0.134, +0.145] | +0.152 [+0.151, +0.148, +0.157] | +0.142 [+0.141, +0.138, +0.148] | +0.134 [+0.133, +0.129, +0.141] | +0.114 [+0.114, +0.110, +0.120] |
| MetaQA | MTK | 432 (native) | +0.123 [+0.126, +0.117, +0.127] | +0.144 [+0.146, +0.141, +0.145] | +0.130 [+0.132, +0.124, +0.135] | +0.115 [+0.117, +0.112, +0.116] | +0.091 [+0.096, +0.087, +0.090] |
| MetaQA | MTK | 500 | +0.122 [+0.121, +0.119, +0.127] | +0.141 [+0.140, +0.136, +0.148] | +0.126 [+0.125, +0.122, +0.130] | +0.108 [+0.108, +0.105, +0.112] | +0.080 [+0.083, +0.080, +0.078] |
| MetaQA | MTK | 1000 | +0.086 [+0.081, +0.089, +0.088] | +0.104 [+0.101, +0.105, +0.106] | +0.083 [+0.079, +0.086, +0.085] | +0.072 [+0.067, +0.076, +0.074] | +0.046 [+0.044, +0.049, +0.046] |
| MetaQA | MTK | 2000 | +0.063 [+0.063, +0.063, +0.065] | +0.062 [+0.059, +0.064, +0.065] | +0.052 [+0.051, +0.051, +0.053] | +0.040 [+0.040, +0.041, +0.040] | +0.016 [+0.017, +0.015, +0.016] |
| MetaQA | MTK | 5000 | +0.055 [+0.057, +0.054, +0.054] | +0.040 [+0.039, +0.039, +0.041] | +0.027 [+0.026, +0.028, +0.029] | +0.009 [+0.010, +0.007, +0.009] | +0.003 [+0.003, +0.002, +0.005] |
| MuSiQue | PHG | 100 | +0.137 [+0.141, +0.125, +0.145] | +0.137 [+0.141, +0.125, +0.145] | +0.137 [+0.141, +0.125, +0.145] | +0.137 [+0.141, +0.125, +0.145] | +0.137 [+0.141, +0.125, +0.145] |
| MuSiQue | PHG | 250 | +0.091 [+0.094, +0.098, +0.081] | +0.118 [+0.120, +0.123, +0.112] | +0.101 [+0.100, +0.113, +0.091] | +0.083 [+0.086, +0.091, +0.074] | +0.064 [+0.063, +0.070, +0.059] |
| MuSiQue | PHG | 500 | +0.056 [+0.060, +0.055, +0.053] | +0.084 [+0.088, +0.089, +0.074] | +0.069 [+0.072, +0.071, +0.063] | +0.052 [+0.053, +0.056, +0.046] | +0.031 [+0.033, +0.028, +0.033] |
| MuSiQue | PHG | 1000 | +0.038 [+0.042, +0.041, +0.030] | +0.061 [+0.063, +0.067, +0.054] | +0.045 [+0.047, +0.049, +0.041] | +0.034 [+0.036, +0.039, +0.027] | +0.022 [+0.024, +0.023, +0.020] |
| MuSiQue | PHG | 1175 (native) | +0.037 [+0.035, +0.037, +0.040] | +0.059 [+0.059, +0.061, +0.058] | +0.042 [+0.042, +0.042, +0.042] | +0.031 [+0.028, +0.031, +0.035] | +0.020 [+0.017, +0.020, +0.024] |
| MuSiQue | PHG | 2000 | +0.027 [+0.028, +0.030, +0.024] | +0.045 [+0.046, +0.051, +0.038] | +0.036 [+0.039, +0.040, +0.030] | +0.023 [+0.025, +0.025, +0.019] | +0.015 [+0.013, +0.021, +0.012] |
| MuSiQue | PHG | 5000 | +0.012 [+0.008, +0.014, +0.015] | +0.034 [+0.032, +0.036, +0.035] | +0.023 [+0.023, +0.023, +0.022] | +0.013 [+0.010, +0.014, +0.014] | +0.008 [+0.006, +0.010, +0.008] |
| SQuAD | PHG | 100 | +0.007 [+0.010, +0.006, +0.006] | +0.007 [+0.010, +0.006, +0.006] | +0.007 [+0.010, +0.006, +0.006] | +0.007 [+0.010, +0.006, +0.006] | +0.007 [+0.010, +0.006, +0.006] |
| SQuAD | PHG | 202 (native) | +0.006 [+0.005, +0.008, +0.004] | +0.008 [+0.008, +0.011, +0.007] | +0.007 [+0.006, +0.010, +0.006] | +0.006 [+0.005, +0.008, +0.005] | +0.005 [+0.004, +0.007, +0.004] |
| SQuAD | PHG | 250 | +0.005 [+0.004, +0.005, +0.007] | +0.005 [+0.005, +0.005, +0.005] | +0.005 [+0.005, +0.006, +0.006] | +0.004 [+0.004, +0.004, +0.004] | +0.003 [+0.004, +0.003, +0.004] |
| SQuAD | PHG | 500 | +0.003 [+0.002, +0.004, +0.003] | +0.004 [+0.005, +0.004, +0.004] | +0.003 [+0.003, +0.003, +0.003] | +0.002 [+0.001, +0.002, +0.002] | +0.001 [+0.001, +0.002, +0.001] |
| SQuAD | PHG | 1000 | −0.000 [−0.001, +0.001, −0.001] | +0.000 [0, +0.001, +0.001] | +0.000 [0, +0.001, +0.001] | +0.001 [+0.001, +0.002, 0] | +0.001 [0, +0.002, 0] |
| SQuAD | PHG | 2000 | −0.001 [−0.002, 0, −0.001] | −0.001 [−0.002, −0.001, −0.002] | −0.000 [−0.001, +0.001, −0.001] | −0.000 [−0.001, +0.001, −0.001] | −0.001 [−0.002, +0.001, −0.001] |
| SQuAD | PHG | 5000 | −0.000 [0, −0.001, 0] | +0.001 [+0.001, +0.001, +0.001] | 0 [0, −0.001, +0.001] | −0.001 [0, −0.001, −0.001] | +0.001 [+0.001, +0.001, 0] |
| SQuAD | MTK | 100 | +0.008 [+0.011, +0.006, +0.006] | +0.008 [+0.011, +0.006, +0.006] | +0.008 [+0.011, +0.006, +0.006] | +0.008 [+0.011, +0.006, +0.006] | +0.008 [+0.011, +0.006, +0.006] |
| SQuAD | MTK | 202 (native) | +0.005 [+0.004, +0.007, +0.003] | +0.007 [+0.006, +0.009, +0.005] | +0.005 [+0.004, +0.008, +0.004] | +0.005 [+0.004, +0.007, +0.004] | +0.003 [+0.003, +0.005, +0.003] |
| SQuAD | MTK | 250 | +0.004 [+0.003, +0.004, +0.006] | +0.004 [+0.004, +0.005, +0.005] | +0.005 [+0.004, +0.005, +0.005] | +0.003 [+0.003, +0.003, +0.004] | +0.003 [+0.003, +0.003, +0.003] |
| SQuAD | MTK | 500 | +0.003 [+0.002, +0.004, +0.003] | +0.002 [+0.003, +0.002, +0.002] | +0.003 [+0.003, +0.003, +0.003] | +0.002 [+0.001, +0.002, +0.002] | +0.001 [+0.001, +0.002, +0.001] |
| SQuAD | MTK | 1000 | +0.000 [0, +0.001, 0] | +0.001 [+0.001, +0.002, +0.002] | +0.001 [+0.001, +0.001, +0.001] | +0.001 [+0.001, +0.002, 0] | +0.001 [+0.001, +0.002, +0.001] |
| SQuAD | MTK | 2000 | −0.000 [−0.001, +0.001, 0] | −0.001 [−0.001, 0, −0.001] | +0.001 [+0.001, +0.002, +0.001] | +0.001 [0, +0.002, +0.001] | 0 [−0.001, +0.001, 0] |
| SQuAD | MTK | 5000 | −0.000 [0, −0.001, 0] | +0.001 [+0.001, +0.001, +0.001] | 0 [0, −0.001, +0.001] | −0.001 [0, −0.001, −0.001] | +0.001 [+0.001, +0.001, 0] |

- **The fan-out ratio.** Under every α arm, the structural ÷ random B_P ratio stays at its K = 100 value (2 decimals) at every K: 0.81 on MetaQA, 0.70 on MuSiQue, 0.73 on SQuAD PHG and 0.69 on SQuAD MTK. Under ES·KNEE_SDIV it is lower at the large K (§35.7): 0.45 on MuSiQue at K 5000, 0.71–0.72 on MetaQA at K 2000 and 5000, and 0.51–0.55 on SQuAD at K 2000 and 5000.
- **Retrieval.** Structural − random is positive for every seed, at every K and under every arm, on MetaQA (both families) and on MuSiQue.
  - It is smaller at the largest K than at K 100 under every arm.
  - At the largest K it is smaller the larger α is. On MuSiQue at K 5000 it is +0.034, +0.023, +0.013 and +0.008 (α 0.65 to 1.0; ES·KNEE_SDIV +0.012). On MetaQA MTK it is +0.040, +0.027, +0.009 and +0.003 (ES·KNEE_SDIV +0.055).
- **SQuAD.** From K 1000, the structural and random ALL counts differ by at most 8 rows, summed over the three seeds (of 6,000), under every arm on both families.

**36.10 Local scan and load** (K ≥ 1000). Each entry is CMASS/N · the load's peak ÷ mean. CMASS/N is the mean share of the corpus inside the contacted shards. Peak ÷ mean is the most-contacted shard's contacts over the mean per shard; for RAND it is the range over the seeds.

| dataset | family | K | ES·KNEE_SDIV | α 0.65 | α 0.75 | α 0.85 | α 1.0 |
|---|---|---|---|---|---|---|---|
| MetaQA | PHG | 1000 | 0.148 · 6.75 | 0.129 · 7.77 | 0.162 · 6.21 | 0.204 · 4.94 | 0.288 · 3.51 |
| MetaQA | PHG | 2000 | 0.095 · 10.57 | 0.102 · 9.96 | 0.137 · 7.41 | 0.184 · 5.50 | 0.288 · 3.52 |
| MetaQA | MTK | 1000 | 0.148 · 6.71 | 0.129 · 7.70 | 0.162 · 6.16 | 0.204 · 4.90 | 0.287 · 3.49 |
| MetaQA | MTK | 2000 | 0.094 · 10.51 | 0.101 · 9.88 | 0.136 · 7.35 | 0.184 · 5.46 | 0.287 · 3.49 |
| MetaQA | MTK | 5000 | 0.046 · 21.77 | 0.074 · 13.68 | 0.110 · 9.26 | 0.162 · 6.27 | 0.290 · 3.49 |
| MetaQA | RAND0–2 | 1000 | 0.198 · 4.90–4.93 | 0.158 · 6.11–6.13 | 0.199 · 4.92–4.94 | 0.250 · 3.93–3.97 | 0.353 · 2.81–2.84 |
| MetaQA | RAND0–2 | 2000 | 0.132 · 7.42–7.47 | 0.124 · 7.91–7.96 | 0.167 · 5.92–5.95 | 0.226 · 4.41–4.44 | 0.353 · 2.82–2.84 |
| MetaQA | RAND0–2 | 5000 | 0.063 · 15.45–15.53 | 0.090 · 11.05–11.13 | 0.133 · 7.49–7.54 | 0.197 · 5.07–5.10 | 0.355 · 2.82–2.84 |
| MuSiQue | PHG | 1000 | 0.184 · 5.31 | 0.144 · 6.74 | 0.182 · 5.50 | 0.228 · 4.41 | 0.322 · 3.14 |
| MuSiQue | PHG | 1175 (native) | 0.171 · 5.68 | 0.136 · 7.14 | 0.175 · 5.73 | 0.223 · 4.52 | 0.323 · 3.14 |
| MuSiQue | PHG | 2000 | 0.131 · 7.56 | 0.114 · 8.92 | 0.153 · 6.64 | 0.206 · 4.93 | 0.323 · 3.15 |
| MuSiQue | PHG | 5000 | 0.075 · 12.53 | 0.082 · 11.97 | 0.122 · 8.28 | 0.180 · 5.63 | 0.323 · 3.15 |
| MuSiQue | RAND0–2 | 1000 | 0.302 · 2.23–2.26 | 0.205 · 2.74–2.79 | 0.258 · 2.49–2.50 | 0.324 · 2.26–2.29 | 0.457 · 1.93–1.97 |
| MuSiQue | RAND0–2 | 1175 (native) | 0.291 · 2.45–2.46 | 0.194 · 3.04–3.06 | 0.247 · 2.75–2.78 | 0.316 · 2.48–2.54 | 0.458 · 2.00–2.04 |
| MuSiQue | RAND0–2 | 2000 | 0.251 · 3.16–3.21 | 0.161 · 4.00–4.05 | 0.216 · 3.58–3.64 | 0.292 · 3.04–3.06 | 0.457 · 2.11–2.13 |
| MuSiQue | RAND0–2 | 5000 | 0.165 · 5.24–5.36 | 0.116 · 6.89–7.06 | 0.172 · 5.22–5.30 | 0.254 · 3.74–3.78 | 0.457 · 2.17–2.18 |
| SQuAD | PHG | 1000 | 0.180 · 4.68 | 0.159 · 5.13 | 0.200 · 4.79 | 0.251 · 4.00 | 0.354 · 2.86 |
| SQuAD | PHG | 2000 | 0.130 · 6.80 | 0.125 · 7.61 | 0.168 · 5.95 | 0.227 · 4.47 | 0.355 · 2.86 |
| SQuAD | PHG | 5000 | 0.086 · 11.00 | 0.091 · 11.10 | 0.134 · 7.59 | 0.199 · 5.14 | 0.356 · 2.86 |
| SQuAD | MTK | 1000 | 0.174 · 4.74 | 0.150 · 5.13 | 0.189 · 4.93 | 0.238 · 4.18 | 0.335 · 3.02 |
| SQuAD | MTK | 2000 | 0.130 · 7.06 | 0.122 · 7.89 | 0.165 · 6.27 | 0.222 · 4.72 | 0.346 · 3.03 |
| SQuAD | MTK | 5000 | 0.090 · 11.34 | 0.094 · 11.59 | 0.139 · 8.00 | 0.206 · 5.44 | 0.365 · 3.03 |
| SQuAD | RAND0–2 | 1000 | 0.302 · 2.42–2.51 | 0.214 · 2.59–2.69 | 0.270 · 2.57–2.70 | 0.339 · 2.45–2.54 | 0.479 · 2.01–2.02 |
| SQuAD | RAND0–2 | 2000 | 0.244 · 3.51–3.59 | 0.168 · 4.24–4.40 | 0.227 · 3.88–3.95 | 0.306 · 3.13–3.17 | 0.479 · 2.08–2.09 |
| SQuAD | RAND0–2 | 5000 | 0.153 · 6.03–6.10 | 0.122 · 7.58–7.63 | 0.181 · 5.44–5.48 | 0.267 · 3.74–3.76 | 0.480 · 2.08–2.10 |

- **α 1.0** keeps the peak ÷ mean flat over K ≥ 1000: 3.14–3.15 on MuSiQue PHG, 3.51–3.52 on MetaQA PHG, 3.49 on MetaQA MTK, 2.86 on SQuAD PHG and 3.02–3.03 on SQuAD MTK. Its CMASS/N is 0.322–0.323 on MuSiQue PHG and 0.287–0.290 on MetaQA MTK (0.335–0.365 on SQuAD MTK).
- **ES·KNEE_SDIV** scans less and concentrates the load as K grows. From K 1000 to K 5000, CMASS/N falls from 0.184 to 0.075 on MuSiQue PHG and from 0.148 to 0.046 on MetaQA MTK. Peak ÷ mean grows from 5.31 to 12.53 and from 6.71 to 21.77.
- **At K 5000, α 0.75 and α 0.85** lie between them. Their CMASS/N is 0.122 and 0.180 on MuSiQue PHG and 0.110 and 0.162 on MetaQA MTK. Their peak ÷ mean is 8.28 and 5.63 on MuSiQue PHG and 9.26 and 6.27 on MetaQA MTK.
- **CMASS/N tracks ρ** (36.5). On MuSiQue PHG at K 5000 it is 0.075, 0.082, 0.122, 0.180 and 0.323, against ρ 0.073, 0.081, 0.120, 0.177 and 0.318.
- **RAND.** At every K and arm in the table, the random maps' CMASS/N is above each structural map's, and every seed's peak ÷ mean is below it.

**36.11 Not run, and flags.**
- **Not run.** MetaQA PHG K 5000 is absent, as in §35 (the PHG builder refused it). No partition, hypergraph or cache was built; the maps are §35's. No α other than the direction's four ran, and no K_REF other than 100.
- **K = 100 is fixed by construction.** Every α arm is ES·KNEE_SDIV at K = 100, so §35's K = 100 losses remain under every α (36.6). The criteria start at K 250 for that reason.
- **α 1.0 keeps ρ at its K = 100 value.** It cannot meet "B_P/K ↓" by construction (36.5).
- **An α arm is not above ES·KNEE_SDIV at every K.** Where α is below the base's effective exponent from K 100, the arm contacts fewer shards (36.5's list). Its difference from the base therefore mixes fewer shards at the smaller K with more at the larger K.
- **Selection on development numbers.** The four α values are the direction's. Choosing among them on these numbers is a selection on the development population c7f70806…, as §35's was. The one final held-out evaluation is still the user's step.
- **The need is a fixed-fan-out quantity** (36.8); an arm's B_P varies per query.
- **p-values.** The summary stores descriptive McNemar values: each arm against ES·KNEE_SDIV, and structural against each random seed. They are not tabulated here.
- **Router state.** The α rule needs the K = 100 map's node → shard table at the router, besides the K map's (N entries each).

**36.12 Reading** (development numbers, descriptive; the decision is the user's step).
- **The text's test** (*"removes the −1 to −1.5 point large-K degradation while preserving the rapidly declining fraction of contacted shards"*).
  - At B_N 1000 and K 5000, MuSiQue PHG goes from −0.015 (ES·KNEE_SDIV) to −0.005, −0.003 and +0.003 under α 0.75, 0.85 and 1.0. MetaQA MTK goes from −0.014 to +0.005, +0.007 and +0.008.
  - ρ still falls with K under α 0.75 and 0.85: at K 5000 it is 0.120 and 0.177 on MuSiQue and 0.108 and 0.160 on MetaQA MTK, from 0.318 and 0.287 at K 100. Under α 1.0 it does not fall.
- **The loss follows the count, but not over the whole band.**
  - The maps, the order, the budgets and the population are unchanged, and the K 5000 loss shrinks as the count grows.
  - On MuSiQue PHG at K 1000, 1175 and 2000, α 0.75 loses 15, 19 and 18 rows, against ES·KNEE_SDIV's 11, 16 and 20. α 0.85 loses 8, 15 and 9.
  - Only α 1.0 is at most 3 rows below unrouted at every K ≥ 250 on MuSiQue PHG, and it keeps ρ at its K = 100 value.
- **The tolerance picks the α.**
  - At B_N 1000, over the five structural families, the smallest α is 0.75 for tol 0.01 (the text's B_P/K ∝ K^−0.25), 1.0 for tol 0.005, and none for tol 0.
  - At B_N 5000 no α holds tol 0.01 on MuSiQue PHG or MetaQA MTK. The K = 100 count, which every arm scales, already loses 0.033 and 0.030 there.
- **Fewer shards help on MetaQA PHG.** There the gain over unrouted shrinks as α grows (at K 2000: +88 rows under α 0.65, +25 under α 1.0), mostly on hop 3. The α that keeps MuSiQue's reach is not the one that gains most on MetaQA PHG.
- **Structure and load.**
  - Under every α the structural maps beat random on every seed on MetaQA and MuSiQue, while contacting 0.70–0.81 times as many shards.
  - α 1.0 keeps the peak ÷ mean flat (2.86–3.52). At K 5000, ES·KNEE_SDIV's reaches 12.53 on MuSiQue and 21.77 on MetaQA MTK, and α 0.75's 8.28 and 9.26.
- **For the user** (nothing is selected, and nothing is frozen):
  - (a) Keep ES·KNEE_SDIV, with K fixed inside §35's 250–2000 band, where it is within 0.01 of unrouted at B_N 1000 on every structural family (§35.11 (a)).
  - (b) Replace its count with the α rule, for any K:
    - α 0.75: tol 0.01 at B_N 1000, with ρ falling;
    - α 0.85: a wider margin (MuSiQue at worst −15 rows), and tol 0.01 at B_N 2000 on MetaQA MTK;
    - α 1.0: tol 0.005 at B_N 1000, with ρ constant.
  - (c) Rule first on what no α reaches: tol 0 on MuSiQue and SQuAD, tol 0.005 at B_N 2000 on MuSiQue, and tol 0.01 at B_N 5000 on MuSiQue and MetaQA MTK. The K = 100 count also loses there. A different K_REF or reference count would be a new rule; none ran.
  - Then the ONE held-out evaluation at the chosen K and count. It is the user's step. ES·KNEE_SDIV and the α arms stay development candidates.

**36.13 Cost and hygiene.**
- **Wall-clock and memory** (the harness runs, one at a time):

| run | start (local) | seconds (loop) | peak RSS (MB) | available at start (GiB) | swap used (GiB) | other Python processes > 500 MB |
|---|---|---|---|---|---|---|
| kscale SQuAD | 12:59:54 | 360.3 (302.5) | 535.7 | 5.56 | 1.45 | 0 |
| kscale MetaQA | 13:05:59 | 270.5 (254.0) | 826.1 | 5.54 | 1.45 | 0 |
| kscale MuSiQue | 13:10:32 | 747.2 (496.7) | 1,379.0 | 5.52 | 1.45 | 0 |

- **The smoke.** A 200-row SQuAD smoke ran at 12:58:00 local (81.9 s, peak 433.4 MB), written outside the repository, before the full runs.
- **The summary** took 3.0 s, a light process run after the three harness runs, and wrote `kscale_SUMMARY__v1.json` (sha `db15a9e0ffc9`).
- **Population:** the development population c7f70806… (§31). Nothing held out was read.
- **Nothing under `data/` was written. No pinned module or CONTRACT_FILE was edited.**
  - `_l1d_scale.py` (ba0a1837…) was imported and subclassed unchanged. `_l1d_arms.py` (f5f52b79…) and `_l1d_lib.py` (dafe39b2…) ran unchanged. The summary imported `_l1d_scale_summary.py` (08696c8b…) unchanged as its helpers.
  - The modules pinned through §35's `structures.imports` hash to their recorded values: `_l1d_node1h.py` (8265d6bc…), `_l1d_edgediag.py` (874bfc49…), `_l1d_adaptbp.py` (984e3aef…), `_l1d_route.py` (7b420e54…), `_l1d_route2.py` (890566c7…), `_l1d_route3.py` (283a711b…) and `_l1d_scale_parts.py` (906a41c2…).
  - New modules: `_l1d_kscale.py` (b47cfaf4…) and `_l1d_kscale_summary.py` (125ce9dc…).

## 37. The ruling: IR_L1 · ES · ⌈KNEE_SDIV(K 100 map) · (K/100)^0.75⌉ carried forward, L1 routing development stopped, the held-out evaluation postponed; and the host shard-map lane (2026-09-29 local) (`_l1d_carry.py`; record `results/L1_DEV/L1_ROUTING_CARRY_FORWARD__v1.json` (sha aaae66fa…); host lane `_l1h_host.py`, `_l1h_place.py`, `_l1h_apt.sh`, `_l1h_apt_time.sh`, `_l1h_diag.sh`, `rx.toml`, records `results/L1_HOST/`; STATUS = CARRIED_FORWARD: development evidence only, not held-out confirmed, the architecture is not final)

**37.1 The ruling (2026-09-29 local).** After §36 the user forwarded a fourth advisor text, then answered the question about the held-out population. The advisor's operative sentences, verbatim (math in plain text):
- "Given §36, I'd choose (b) with α=0.75 as the current L1 routing rule."
- "Carry forward: IR_L1, ES shard ranking, KNEE_SDIV coarse fan-out, B_P(q,K)=⌈B_100(q)(K/100)^0.75⌉ and treat B_N=5000 parity as a known limitation of the coarse-anchor count, not something to tune away now."
- "Then stop L1 development."
- "I'd like to see the corresponding α=.75 load number prominently in the final report too, because this is a distributed-systems property, not merely retrieval quality."
- "We ultimately care about: Recall fan-out local scan load skew not recall alone."

The advisor's next step was the held-out evaluation. The user's answer overrides it:
- "Yes — do not spend the held-out evaluation yet." "I would choose 'Record the ruling only' for now." That option was to write up the ruling and hash-freeze the spec, but read no held-out rows yet.
- "Do not read the held-out rows yet."
- "I would not use the reserved MetaQA/SQuAD/MuSiQue rows now to decide how to fix MetaQA." "We already have plenty of legacy-exposed development data. Use that aggressively."
- "I think improving the KB side should be the highest priority now, rather than spending more time squeezing another 0.2% out of MuSiQue routing."
- The B_N at which the held-out 1 % criterion is stated: no preference (open).

**37.2 The carried-forward rule** (`L1_ROUTING_CARRY_FORWARD__v1.json`, written once by `_l1d_carry.py`).
- **L1** is one static A^T x per query: no recursion and no learned parameter. Partitions are routing shards.
- **L1a (node localisation).** IR_L1 = RRF(FLAT, LOC), K0 60 (§32–§34). The served order is the first B_N nodes of IR_L1 inside the contacted shards.
- **L1b (shard routing).**
  - ES ranks the shards of the deployed K map.
  - B100(q) is KNEE_SDIV on the same partitioner's K = 100 map (§35's arithmetic).
  - The fan-out is B_P(q, K) = min(K, ⌈B100(q) · (K/100)^0.75⌉).
  - A query without localised evidence contacts every shard.
- **The min never binds at K ≥ 100.** B100(q) ≤ 100 and (K/100)^0.75 ≤ K/100, so B100(q) · (K/100)^0.75 ≤ K.
- **Router state.** Two node → shard tables of the same partitioner: the deployed K map and the K = 100 map (one small integer per node each).
- **B_N** is not fixed by the ruling.
- **What the record pins.**
  - The three kscale records (json and npz shas), the §36 summary (db15a9e0…) and the population c7f70806….
  - Every map the evidence used: file and sha for the structural maps, and rule and vector sha for the random ones.
  - The 14 module shas it asserts, from `_l1d_kscale.py` (b47cfaf4…) to `_l1d_loc.py` (d4570b9e…).
  - The α 0.75 evidence, copied from the summary's strings and exact counts (37.3–37.7).
  - The limitations (37.8), the held-out plan and the next steps (37.9).

**37.3 The L1 picture at B_N 1000** (exact counts; the carried-forward rule at the native K). Each cell gives the ALL-gold rows, with the share and, for the routed columns, the difference from unrouted IR_L1.

| dataset | rows | FLAT | unrouted IR_L1 | PHG at the native K | MTK at the native K |
|---|---|---|---|---|---|
| MetaQA | 1,998 | 358 (0.179) | 1,015 (0.508) | K 432: 1,112 (0.557, +97) | K 432: 1,082 (0.542, +67) |
| MuSiQue | 2,000 | 1,513 (0.757) | 1,679 (0.840) | K 1175: 1,660 (0.830, −19) | — |
| SQuAD | 2,000 | 2,000 (1.000) | 1,999 (1.000) | K 202: 1,998 (0.999, −1) | K 202: 1,994 (0.997, −5) |

Per hop (FLAT → unrouted → PHG native → MTK native):
- MetaQA hop 1 (666 rows): 174 (0.261) → 666 (1.000) → 664 (0.997) → 664 (0.997).
- MetaQA hop 2 (666): 98 (0.147) → 187 (0.281) → 220 (0.330) → 213 (0.320).
- MetaQA hop 3 (666): 86 (0.129) → 162 (0.243) → 228 (0.342) → 205 (0.308).
- MuSiQue hop 2 (1,364): 1,155 (0.847) → 1,244 (0.912) → 1,233 (0.904).
- MuSiQue hop 3 (442): 283 (0.640) → 336 (0.760) → 335 (0.758).
- MuSiQue hop 4 (194): 75 (0.387) → 99 (0.510) → 92 (0.474).

The one-hop L1 solves MetaQA hop 1. Hop 2 and hop 3 stay at 0.281–0.342 ALL-gold. That residual is the user's KB priority (37.9).

**Correction.** §31.3 and §32 printed MuSiQue FLAT as 0.537 / 0.631 / 0.695 / 0.756 / 0.814 / 0.883, and §31's FLAT@M table row as 0.537 / 0.756 / 0.883.
- The exact counts are 1,075 / 1,262 / 1,389 / 1,513 / 1,628 / 1,767 of 2,000. Under ROUND_HALF_UP that is 0.538 / 0.631 / 0.695 / 0.757 / 0.814 / 0.884.
- Three stored 4-decimal shares (0.5375, 0.7565, 0.8835) had been rounded half to even. The three places are corrected in place, the two text lines with a pointer here.
- The user's answer quoted "0.756 → 0.840"; the exact figures are 1,513 → 1,679 rows (0.757 → 0.840).

**37.4 The α 0.75 systems table** (the advisor's four measures, on every structural map).
- **Δ** is the ALL-gold count minus unrouted IR_L1 at the same B_N, in rows.
- **B_P and ρ = B_P/K** are means per query.
- **CMASS/N** is the mean share of the corpus inside the contacted shards (the local scan).
- **Busiest share** is the share of queries that contact the most-contacted shard.
- **Peak ÷ mean** is that shard's contacts over the mean contacts per shard.

| dataset | family | K | Δ B_N 500 | Δ B_N 1000 | Δ B_N 2000 | Δ B_N 5000 | B_P mean | ρ | CMASS/N | busiest share | peak ÷ mean | shards never contacted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MetaQA | PHG | 100 | −19 | −29 | −27 | −92 | 28.4 | 0.284 | 0.287 | 0.937 | 3.30 | 0 |
| MetaQA | PHG | 250 | +74 | +90 | +96 | +39 | 56.9 | 0.227 | 0.231 | 0.976 | 4.29 | 0 |
| MetaQA | PHG | 432 (native) | +80 | +97 | +98 | +23 | 85.3 | 0.197 | 0.200 | 0.980 | 4.97 | 0 |
| MetaQA | PHG | 500 | +83 | +109 | +124 | +38 | 95.5 | 0.191 | 0.193 | 0.985 | 5.16 | 0 |
| MetaQA | PHG | 1000 | +57 | +84 | +91 | +14 | 160.3 | 0.160 | 0.162 | 0.996 | 6.21 | 1 |
| MetaQA | PHG | 2000 | +21 | +73 | +93 | +25 | 269.4 | 0.135 | 0.137 | 0.997 | 7.41 | 0 |
| MetaQA | MTK | 100 | +4 | +3 | +7 | −59 | 28.7 | 0.287 | 0.294 | 0.950 | 3.32 | 0 |
| MetaQA | MTK | 250 | +79 | +83 | +80 | +2 | 57.3 | 0.229 | 0.230 | 0.979 | 4.27 | 0 |
| MetaQA | MTK | 432 (native) | +62 | +67 | +59 | −22 | 86.0 | 0.199 | 0.200 | 0.982 | 4.93 | 0 |
| MetaQA | MTK | 500 | +67 | +75 | +65 | −27 | 96.3 | 0.193 | 0.193 | 0.985 | 5.12 | 0 |
| MetaQA | MTK | 1000 | +37 | +20 | −10 | −117 | 161.7 | 0.162 | 0.162 | 0.995 | 6.16 | 0 |
| MetaQA | MTK | 2000 | +5 | −9 | −48 | −158 | 271.6 | 0.136 | 0.136 | 0.998 | 7.35 | 0 |
| MetaQA | MTK | 5000 | +3 | +9 | −10 | −116 | 539.6 | 0.108 | 0.110 | 0.999 | 9.26 | 0 |
| MuSiQue | PHG | 100 | −7 | −22 | −35 | −66 | 31.8 | 0.318 | 0.319 | 0.737 | 2.32 | 0 |
| MuSiQue | PHG | 250 | +1 | −6 | −12 | −51 | 63.6 | 0.254 | 0.257 | 0.847 | 3.33 | 0 |
| MuSiQue | PHG | 500 | −4 | −8 | −25 | −62 | 106.7 | 0.213 | 0.216 | 0.916 | 4.29 | 0 |
| MuSiQue | PHG | 1000 | −5 | −15 | −30 | −66 | 179.2 | 0.179 | 0.182 | 0.986 | 5.50 | 1 |
| MuSiQue | PHG | 1175 (native) | −4 | −19 | −43 | −87 | 202.2 | 0.172 | 0.175 | 0.987 | 5.73 | 1 |
| MuSiQue | PHG | 2000 | −5 | −18 | −44 | −80 | 301.1 | 0.151 | 0.153 | 1.000 | 6.64 | 1 |
| MuSiQue | PHG | 5000 | −4 | −9 | −35 | −71 | 598.1 | 0.120 | 0.122 | 0.991 | 8.28 | 1 |
| SQuAD | PHG | 100 | −2 | −4 | −5 | −5 | 35.0 | 0.350 | 0.351 | 0.691 | 1.97 | 0 |
| SQuAD | PHG | 202 (native) | 0 | −1 | −2 | −2 | 59.7 | 0.296 | 0.297 | 0.691 | 2.34 | 0 |
| SQuAD | PHG | 250 | +1 | −2 | −3 | −3 | 69.9 | 0.280 | 0.282 | 0.619 | 2.21 | 0 |
| SQuAD | PHG | 500 | 0 | −2 | −3 | −3 | 117.4 | 0.235 | 0.238 | 0.765 | 3.26 | 0 |
| SQuAD | PHG | 1000 | 0 | −4 | −5 | −5 | 197.1 | 0.197 | 0.200 | 0.944 | 4.79 | 0 |
| SQuAD | PHG | 2000 | 0 | −4 | −5 | −5 | 331.2 | 0.166 | 0.168 | 0.986 | 5.95 | 0 |
| SQuAD | PHG | 5000 | 0 | −4 | −5 | −5 | 658.0 | 0.132 | 0.134 | 0.999 | 7.59 | 0 |
| SQuAD | MTK | 100 | −1 | −3 | −4 | −4 | 33.0 | 0.330 | 0.332 | 0.503 | 1.52 | 0 |
| SQuAD | MTK | 202 (native) | −1 | −5 | −6 | −6 | 56.4 | 0.279 | 0.282 | 0.566 | 2.02 | 0 |
| SQuAD | MTK | 250 | −1 | −3 | −4 | −4 | 66.0 | 0.264 | 0.265 | 0.632 | 2.39 | 0 |
| SQuAD | MTK | 500 | 0 | −1 | −2 | −2 | 110.9 | 0.222 | 0.224 | 0.761 | 3.43 | 0 |
| SQuAD | MTK | 1000 | 0 | −3 | −4 | −4 | 186.2 | 0.186 | 0.189 | 0.918 | 4.93 | 0 |
| SQuAD | MTK | 2000 | 0 | −2 | −3 | −3 | 312.8 | 0.156 | 0.165 | 0.981 | 6.27 | 1 |
| SQuAD | MTK | 5000 | −1 | −4 | −5 | −5 | 621.4 | 0.124 | 0.139 | 0.995 | 8.00 | 1 |

- **The load identity: peak ÷ mean = busiest share ÷ ρ.** The peak load is the busiest share × the rows, and the mean load per shard is ρ × the rows. Every row above satisfies it to the stored rounding.
- **One shard is contacted by almost every query.**
  - Under α 0.75 the busiest share rises with K: from 0.503–0.950 at K = 100 to 0.991–0.999 at the largest K of every family.
  - At large K, therefore, peak ÷ mean ≈ 1/ρ: 7.41 (MetaQA PHG, K 2000), 9.26 (MetaQA MTK, K 5000), 8.28 (MuSiQue, K 5000), and 7.59 and 8.00 (SQuAD PHG and MTK, K 5000).
  - While that shard stays hot, a count whose ρ falls with K raises the skew. That is a placement and serving question (for example, replicating that shard), not an L1 count. Nothing about it ran.
- **Fan-out.** ρ falls with K on every family, from K 100 to the largest K:
  - 0.284 → 0.135 (MetaQA PHG, K 2000) and 0.287 → 0.108 (MetaQA MTK, K 5000);
  - 0.318 → 0.120 (MuSiQue, K 5000);
  - 0.350 → 0.132 and 0.330 → 0.124 (SQuAD PHG and MTK, K 5000).
  - CMASS/N tracks ρ; it is at most 0.015 above it (SQuAD MTK K 5000: 0.139 against 0.124).
- **Recall.**
  - MetaQA PHG is above unrouted at every K ≥ 250 and every B_N in the table (at least +14). Its K = 100 anchor is below (−19 to −92).
  - MetaQA MTK is above unrouted at B_N 500 at every K, and at B_N 1000 at every K but 2000 (−9). At B_N 2000 it is below from K 1000 (at worst −48, at K 2000). At B_N 5000 it is below at every K but 250 (at worst −158, at K 2000).
  - MuSiQue is below unrouted at every K and every B_N from 1000. Over K ≥ 250 it is at worst −19 at B_N 1000 (K 1175), −44 at B_N 2000 (K 2000) and −87 at B_N 5000 (K 1175). At B_N 500 it is within 7 rows.
  - SQuAD is within 6 rows of unrouted everywhere (at worst −6, MTK K 202).

**37.5 α 0.75 between ES·KNEE_SDIV and α 1.0** (the largest K of each family; Δ at B_N 1000 in rows).

| dataset | family | K | count | B_P mean | ρ | busiest share | peak ÷ mean | Δ B_N 1000 |
|---|---|---|---|---|---|---|---|---|
| MetaQA | PHG | 2000 | ES·KNEE_SDIV | 187.5 | 0.094 | 0.991 | 10.57 | +82 |
| MetaQA | PHG | 2000 | α 0.75 | 269.4 | 0.135 | 0.997 | 7.41 | +73 |
| MetaQA | PHG | 2000 | α 1.0 | 568.6 | 0.284 | 1.000 | 3.52 | +25 |
| MetaQA | MTK | 5000 | ES·KNEE_SDIV | 227.6 | 0.046 | 0.991 | 21.77 | −28 |
| MetaQA | MTK | 5000 | α 0.75 | 539.6 | 0.108 | 0.999 | 9.26 | +9 |
| MetaQA | MTK | 5000 | α 1.0 | 1,433.5 | 0.287 | 1.000 | 3.49 | +16 |
| MuSiQue | PHG | 5000 | ES·KNEE_SDIV | 366.8 | 0.073 | 0.920 | 12.53 | −30 |
| MuSiQue | PHG | 5000 | α 0.75 | 598.1 | 0.120 | 0.991 | 8.28 | −9 |
| MuSiQue | PHG | 5000 | α 1.0 | 1,589.0 | 0.318 | 1.000 | 3.15 | +5 |
| SQuAD | PHG | 5000 | ES·KNEE_SDIV | 421.2 | 0.084 | 0.927 | 11.00 | −4 |
| SQuAD | PHG | 5000 | α 0.75 | 658.0 | 0.132 | 0.999 | 7.59 | −4 |
| SQuAD | PHG | 5000 | α 1.0 | 1,748.3 | 0.350 | 1.000 | 2.86 | 0 |
| SQuAD | MTK | 5000 | ES·KNEE_SDIV | 400.6 | 0.080 | 0.909 | 11.34 | −4 |
| SQuAD | MTK | 5000 | α 0.75 | 621.4 | 0.124 | 0.995 | 8.00 | −4 |
| SQuAD | MTK | 5000 | α 1.0 | 1,650.9 | 0.330 | 1.000 | 3.03 | 0 |

- **ES·KNEE_SDIV** contacts the fewest shards and concentrates the load most: peak ÷ mean 10.57–21.77.
- **α 1.0** keeps ρ at its K = 100 value (0.284–0.350) and flattens the skew (2.86–3.52). It contacts 2.11 (K 2000) to 2.66 (K 5000) times α 0.75's mean B_P, i.e. (K/100)^0.25.
- **α 0.75** lies between them: peak ÷ mean 7.41–9.26 at ρ 0.108–0.135. At K 5000 its skew is 0.66 of ES·KNEE_SDIV's on MuSiQue (8.28 against 12.53) and 0.43 of it on MetaQA MTK (9.26 against 21.77).

**37.6 Where α 0.75 holds.** The table gives the worst Δ over K ≥ 250 in rows, per B_N, with the K where it occurs. Tol 0.01 allows −19 on MetaQA (1,998 rows) and −20 elsewhere; tol 0.005 allows −9 and −10.

| family | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|
| MetaQA PHG | 0 | 0 | +21 | +73 | +91 | +14 |
| MetaQA MTK | 0 | 0 | +3 | −9 (K 2000) | −48 (K 2000) | −158 (K 2000) |
| MuSiQue PHG | 0 | 0 | −5 (K 1000) | −19 (K 1175) | −44 (K 2000) | −87 (K 1175) |
| SQuAD PHG | 0 | 0 | 0 | −4 (K 1000) | −5 (K 1000) | −5 (K 1000) |
| SQuAD MTK | 0 | 0 | −1 (K 250) | −4 (K 5000) | −5 (K 5000) | −5 (K 5000) |

The K = 100 anchor (the count every α scales) at B_N 100 / 250 / 500 / 1000 / 2000 / 5000:
- MetaQA PHG −58 / −59 / −19 / −29 / −27 / −92;
- MetaQA MTK −38 / −38 / +4 / +3 / +7 / −59;
- MuSiQue +1 / −3 / −7 / −22 / −35 / −66;
- SQuAD PHG 0 / −1 / −2 / −4 / −5 / −5;
- SQuAD MTK 0 / 0 / −1 / −3 / −4 / −4.

- **Holds.** α 0.75 holds tol 0.005 on every structural family up to B_N 500, and tol 0.01 up to B_N 1000. MuSiQue's −19 at K 1175 is the binding cell.
- **Fails.** At B_N 2000 it fails tol 0.01 on MuSiQue (−44) and MetaQA MTK (−48). At B_N 5000 it fails on the same two (−87 and −158).
- **The anchor.** At B_N 5000 the K = 100 anchor is already −66 (MuSiQue) and −59 (MetaQA MTK). The ruling treats B_N 5000 parity as a known limitation of the coarse-anchor count.
- **The other α at B_N 2000,** for the record: α 0.85 is at worst −34 (MuSiQue) and −15 (MetaQA MTK); α 1.0 −17 and −4.

**37.7 Structural against balanced random under α 0.75.** The values are structural − random ALL share, per seed, at B_N 500–5000 (the summary's per-seed values).
- **MetaQA** (PHG and MTK): positive for every seed, K and B_N, from +0.002 (MTK K 5000, B_N 500) to +0.287 (MTK K 100, B_N 5000).
- **MuSiQue:** positive for every seed, K and B_N, from +0.003 (K 5000, B_N 500) to +0.184 (K 100, B_N 5000).
- **SQuAD:** +0.001 to +0.011 at K ≤ 500. At K ≥ 1000 it is −0.001 to +0.002 (net −1 to +3 rows per seed), i.e. placement-neutral.
- **Fewer shards.** The structural ÷ random mean B_P is 0.81 (MetaQA), 0.70 (MuSiQue), 0.73 (SQuAD PHG) and 0.69 (SQuAD MTK) at every K.
- The advisor's "structural partitions continue beating random placement at every scale" therefore holds on MetaQA and MuSiQue. On SQuAD it holds only up to K 500.

**37.8 Corrections and limits** (what the evidence behind the ruling does not show).
- **Grid minimality.** "α = 0.75 is the smallest scaling exponent" holds over the tested grid {0.65, 0.75, 0.85, 1.0}. At B_N 1000, α 0.65 fails tol 0.01 on MuSiQue (−27 at K 2000). No α between 0.65 and 0.75 ran.
- **B_N 2000.** The 1 % criterion holds at B_N ≤ 1000 but not at 2000 (37.6). B_N 2000 is inside the advisor's "B_N=500–2000" operating band.
- **Load.** Under α 0.75 the skew still grows with K: peak ÷ mean is 1.52–3.32 at K = 100 and 7.41–9.26 at the largest K (37.4). α 0.75 lowers ES·KNEE_SDIV's skew but does not flatten it (37.5).
- **B_N 5000.** The anchor-count cause is attributed, not isolated: the K = 100 count is already −66 and −59 there under every α. No experiment separated it, and the ruling does not ask for one.
- **The anchor is computed on another map.** B100(q) comes from the K = 100 map, and the served K maps are not nested in it:
  - The share of nodes in their shard's majority K = 100 shard is 0.145 (MetaQA PHG K 250) to 0.612 (MetaQA MTK K 5000).
  - The number of K-map shards that lie wholly inside one K = 100 shard ranges from 0 of 250 (SQuAD PHG K 250) to 1,537 of 5,000 (SQuAD MTK K 5000).
- **min(K, ·)** never binds at K ≥ 100 (37.2). It matters only below K 100, where nothing ran.
- **SQuAD** is placement-neutral from K 1000 (37.7).
- **Development only.**
  - The population c7f70806… is legacy-exposed.
  - The evidence covers three mapped corpora, at the grid K plus the native K.
  - MetaQA PHG K 5000 is absent (PARTITION_INVALID).

**37.9 What comes next** (the user's plan; nothing held out is read).
- **Now: MetaQA development** on legacy-exposed rows, aimed at the residual:
  - hop-2 / hop-3 ALL-gold;
  - 5–10 and 11+ answer queries;
  - golds outside the one-hop L1 but inside routed shards;
  - golds that need another STRUCT hop.
  - The purpose is to decide what recovers them: partition-local completion, L2 or L3. There is no recursive traversal in L1.
- **Now: WebQSP shard maps** as development / transfer infrastructure (37.10). Split B stays sealed. Which WebQSP rows form a development population is still open.
- **Then:** finalize L1 + L2 + L3.
- **Then held-out A** (in-domain):
  - MetaQA 666 × 3 hops = 1,998 never-used dev rows;
  - SQuAD 2,000 never-used dev rows;
  - MuSiQue 2,000 never-used train rows (every dev row is already used).
  - No row has been selected or read.
- **Later held-out B** (transfer, sealed): WebQSP 717, HotpotQA 969, 2Wiki 996.
- **Open:** the B_N of the held-out 1 % criterion.

**37.10 The host shard-map lane** (records `results/L1_HOST/`; the shared lab host `gpu`, driven only through `rx`, project `crag`).
- **Why.** The big three's PHG maps were out of reach on the 16 GB laptop. §27's resource survey rated HotpotQA INFEASIBLE_UNDER_PINNED_HYGIENE_CRITERION and 2Wiki INFEASIBLE_EXCEEDS_WSL_VM, and named "another machine for the partition stage" as the user's option. The user offered the host: "use this for the partitioning". The partitioner and its contract are unchanged: Zoltan-PHG, NP 4, the same PARAMS and PHG_NONEMPTY_REPAIR_V1. No Mt-KaHyPar replacement.
- **Declaration.** `HOST_LANE_DECLARATION.json` (addb4426…) names:
  - the stage, L1_HOST_SHARD_MAPS;
  - the inputs pushed: the six stamped key sets, by sha, with `--force` named for the HotpotQA and 2Wiki files;
  - the outputs, the placement test and the scheduling.
- **Toolchain.** The user authorized the apt install. Its log is `apt/apt_20260929T095207Z.log` (09:52:07–09:53:28Z).
  - 96 packages were newly installed: gcc 13.3, OpenMPI 4.1.6, Zoltan 13.2.0-5build2 and their dependencies.
  - 4 packages were upgraded as dependencies: libc6, libc-bin and locales (2.39-0ubuntu8.8 → 8.9) and libevent-core-2.1-7t64 (2.1.12-stable-9ubuntu2.1 → 2.2).
  - `time` was already present (1.9-0.2build1).
- **Build** (`PHG_BUILD_HOST.json`). The laptop's driver source and wrapper, compiled with the same command. The binary (0d996c71…) differs from the laptop's, since the toolchain differs.
- **Placement test** (`PLACEMENT_TEST__v1.json`, 16316ee3…): **BIT_IDENTICAL**, 21 of 21 cells. On MetaQA, SQuAD and MuSiQue, at the native K and every grid K, the host reproduced bit for bit:
  - every hypergraph digest (21/21);
  - every PHG shard file (20/20 runs);
  - every raw KM1 and every final vector (20/20), including the three served vectors;
  - the one refusal: MetaQA K 5000, PARTITION_INVALID on both machines.
- **Authorization** (`HOST_LANE_AUTHORIZATION__2026-09-29.json`, 743ecb32…).
  - It covers WebQSP, HotpotQA and 2Wiki, in that order, at the grid K and the native K (25928, 52333, 59898).
  - It pins `_l1h_host.py` (ae4a5f45…); a functional change voids it.
  - The maps are labelled PHG.
- **WebQSP** (N 2,592,894; at K 100, 5,181,621 hyperedges and 29,574,371 pins).
  - K 100, 250, 500 and 1000 are done: mpirun took 407–418 s, and the whole PHG step 446–478 s, at 4 CPUs.
  - Peak RSS was 0.74–1.14 GB per rank.
  - The largest block is at the contract bound ⌈1.03 N/K⌉ (26,706 against 26,707 at K 100).
  - 0, 0, 2 and 3 empty blocks were repaired.
  - K 2000 and 5000 are done: 24 and 45 empty blocks repaired, the whole PHG step 607.8 and 688.1 s. KM1 runs from 994,832,221 (K 100) to 1,906,956,334 (K 5000).
  - K 25928 is done: 326 empty blocks were repaired, the whole PHG step took 4,138.1 s (mpirun 295.0 s), KM1 is 2,288,620,639 and the largest block, 104, is at the bound. All seven WebQSP maps are on the laptop, with verified npy shas.
- **HotpotQA.** Its key set (dea82ccf…) and 2Wiki's (ef9e1270…) reached the host with matching shas (1.3 GB in 10 min 7 s).
  - K 100, 250, 500, 1000 and 2000 are done. mpirun took 769–992 s, and the whole PHG step 896–1,111 s.
  - Peak RSS was 1.53–2.50 GB per rank. The largest block is at or one under the contract bound. 0, 0, 1, 1 and 0 empty blocks were repaired.
  - K 5000 is done: 31 empty blocks were repaired, the whole PHG step took 1,958.5 s and KM1 is 3,912,242,584.
  - K 52333 is done: 657 empty blocks were repaired (KM1 unchanged), the whole PHG step took 12,446.8 s (mpirun 1,158.4 s), peak RSS was 1.46–1.58 GB per rank, KM1 is 4,985,688,456, and the largest block, 104, is at the contract bound (status at 2026-09-29 21:12 local; earlier: move 519 at 10,549 s at 18:56).
- **2Wiki.**
  - K 100, 250, 500, 1000 and 2000 are done. mpirun took 1,028–1,114 s, and the whole PHG step 1,204–1,541 s. Peak RSS was 2.22–3.56 GB per rank. The largest block is at or one under the contract bound. 0, 0, 0, 0 and 5 empty blocks were repaired.
  - K 5000 is **PARTITION_INVALID**, as MetaQA K 5000 was. The raw partition's largest block is 1,235 against the contract bound 1,234 (45 empty blocks). No map is written and the contract is not relaxed, so the cell is absent.
  - K 59898 is done: 712 empty blocks were repaired (KM1 unchanged), the whole PHG step took 22,067.8 s (mpirun 1,078.5 s), peak RSS was 2.06–2.15 GB per rank, KM1 is 5,739,097,882, and the largest block, 104, is at the contract bound (status at 2026-09-29 22:34 local; earlier: move 646 at 18,622.5 s at 21:32, move 330 at 9,284 s at 18:56). The json, RUN.json and npy were fetched (npy sha OK; the local npy is 47,918,904 bytes). With this map, all 2Wiki cells except K 5000 (PARTITION_INVALID) are done.
- (Status updated 2026-09-29 22:34 local. §40.13 covers the L3 lane on the same host, §41.2 its encoder lane and the host-yield wrapper that the crag jobs now run under.)
- The npy assignments of HotpotQA stay on the host until a stage needs them; their RUN records carry the shas. (Correction, 2026-09-30: this line said "HotpotQA and 2Wiki". The six valid 2Wiki maps, K 100 … 2000 and K 59898, are on the laptop, each 47,918,904 bytes, sha-equal to their RUN records. The K59898 line above said the same by mistake.)
- **GPU.** No L1 stage is GPU-bound. Partitioning is CPU/MPI, and the dense and SPLADE scores are already cached.

**37.11 Cost and hygiene.**
- `_l1d_carry.py` read only development records and wrote one record, in about a second. It read no held-out, split-B or TEST row.
- Nothing under `data/` was written, on either machine. No pinned module or CONTRACT_FILE was edited.
- The host ran only the declared stage, as `rx` jobs of the project `crag` at 4 CPUs each. The capacity reserve and other users' jobs were left alone.
- **New code:** `_l1d_carry.py`, `_l1h_host.py`, `_l1h_place.py`, `_l1h_apt.sh`, `_l1h_apt_time.sh`, `_l1h_diag.sh`, and `rx.toml` (the push list).

## 38. KBRES: where the MetaQA residual lives — L2, partition-local completion or L3? (2026-09-29 local) (`_l1d_kbres.py` on the shared runner `_l1d_arms.py`, L3 split `_l1d_kbres_summary.py`; records `results/L1_DEV/kbres_metaqa__v1.{json,npz}` (json sha 282791b4…, npz 6c34e20c…) and `kbres_metaqa__v1__L3SPLIT.json` (68a6d596…); STATUS = DEVELOPMENT DIAGNOSTIC: descriptive, no rule selected, L1 unchanged)

**38.1 The question.** After the ruling (§37.1) the user made the KB residual the priority. Verbatim:
- "First, improve MetaQA on development data, but focus specifically on the residual: hop-2 / hop-3 ALL-gold; 5–10 and 11+ answer queries; golds that are outside one-hop L1 but inside routed partitions; golds genuinely requiring another STRUCT hop."
- "This tells us whether the missing recall should be recovered by better partition-local completion, L2, or L3."
- "That doesn't mean we should destroy the L1/L3 boundary by adding recursive traversal into L1."

KBRES is a diagnostic: it changes nothing and selects nothing.
- **L1** is the carried-forward rule (§37.2): IR_L1, ES and α 0.75, at MetaQA's native K 432 on both served maps (PHG and MTK). A third cell, UNROUTED, contacts every shard.
- **Rows:** the L1_DEV population (c7f70806…): 1,998 rows (666 per hop) with 14,489 gold nodes, all legacy-exposed. No held-out, split-B or TEST row was read.
- **Identities asserted:**
  - the IR_L1 positions equal the kscale v1 npz on every gold;
  - per cell, row and B_N, the ALL-served flags equal `served_at` of the §37 evidence, so the ALL counts are §37.3's (1,015 / 1,112 / 1,082 at B_N 1000);
  - every gold falls in exactly one class;
  - the distance identities hold.

**38.2 Classes and levers.** Terms:
- **V_q** is the L1a evidence: the 200 FLAT hits plus every node with L > 0 (a one-hop STRUCT, KNN or NER neighbour of a hit).
- **dS** is the undirected STRUCT distance from the 200 hits. **dT**, the distance from the annotated topic entity, is explanatory only.
- An **intermediate** is a STRUCT neighbour of both a hit and the gold.
- "Contacted" means the cell's ES fan-out contacts that shard.

| class | condition (per cell and B_N) | lever |
|---|---|---|
| SERVED_EV | served, in V_q | — |
| SERVED_FILL | served, not in V_q (the FLAT fill inside the contacted shards) | — |
| CROWD_EV | its shard contacted, in V_q, beyond the B_N cut | **L2** (ordering / pool depth) |
| CROWD_2L | its shard contacted, not in V_q, dS 2, an intermediate in a contacted shard | **LOCAL** (partition-local completion) |
| CROWD_2X | its shard contacted, not in V_q, dS 2, every intermediate in an uncontacted shard | L3: **X2** |
| CROWD_3 | its shard contacted, not in V_q, dS ≥ 3 | L3: **HOP3** |
| REACH_EV | its shard not contacted, in V_q | **ROUTE** (the frozen fan-out) |
| REACH_2 | its shard not contacted, not in V_q, dS 2 | L3: **PUSH2** if an intermediate is in a contacted shard, else **FAR2** |
| REACH_3 | its shard not contacted, not in V_q, dS ≥ 3 | L3: **HOP3** |

- **Oracle bounds.** A row that is not ALL-served needs the levers of all its missed golds. "ALL if levers X" counts the rows whose every missed gold has a lever in X. That count is an **oracle upper bound** (perfect recovery, zero exposure), not a mechanism.
- **The L3 split** comes from the L3SPLIT record. It re-derives the v1 classes from the v1 npz and asserts:
  - PUSH2 ∪ FAR2 = REACH_2 and X2 = CROWD_2X;
  - CROWD_2L and CROWD_2X from the contacted and intermediate flags;
  - the v1 ALL counts and lever-set counts.

**38.3 ALL-gold rows at B_N 1000.**

| stratum | rows | unrouted IR_L1 | PHG K 432 | MTK K 432 |
|---|---|---|---|---|
| all | 1,998 | 1,015 (0.508) | 1,112 (0.557) | 1,082 (0.542) |
| hop 1 | 666 | 666 (1.000) | 664 (0.997) | 664 (0.997) |
| hop 2 | 666 | 187 (0.281) | 220 (0.330) | 213 (0.320) |
| hop 3 | 666 | 162 (0.243) | 228 (0.342) | 205 (0.308) |
| 1 answer | 819 | 660 (0.806) | 681 (0.832) | 674 (0.823) |
| 2 answers | 286 | 166 (0.580) | 182 (0.636) | 179 (0.626) |
| 3 answers | 180 | 80 (0.444) | 95 (0.528) | 92 (0.511) |
| 4 answers | 87 | 35 (0.402) | 42 (0.483) | 37 (0.425) |
| 5–10 answers | 287 | 53 (0.185) | 80 (0.279) | 75 (0.261) |
| 11+ answers | 339 | 21 (0.062) | 32 (0.094) | 25 (0.074) |
| hop 2, 5–10 | 90 | 8 (0.089) | 9 (0.100) | 7 (0.078) |
| hop 2, 11+ | 100 | 4 (0.040) | 1 (0.010) | 1 (0.010) |
| hop 3, 5–10 | 161 | 9 (0.056) | 35 (0.217) | 32 (0.199) |
| hop 3, 11+ | 226 | 4 (0.018) | 19 (0.084) | 12 (0.053) |

**38.4 Oracle upper bounds at B_N 1000.** Each lever column gives the ALL-gold rows if the named levers recovered every missed gold. "…" repeats the column before it. The last column is every lever except HOP3; adding HOP3 gives every row.

| map | stratum | rows | ALL | +L2 | +LOCAL | +L2 +LOCAL | +L2 +LOCAL +PUSH2 | +L2 +LOCAL +ROUTE | … +PUSH2 | … +X2 | … +FAR2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PHG | all | 1,998 | 1,112 | 1,117 | 1,315 | 1,334 | 1,558 | 1,381 | 1,687 | 1,723 | 1,753 |
| PHG | hop 2 | 666 | 220 | 220 | 334 | 334 | 548 | 371 | 666 | 666 | 666 |
| PHG | hop 3 | 666 | 228 | 233 | 317 | 336 | 346 | 344 | 355 | 391 | 421 |
| PHG | 5–10 answers | 287 | 80 | 83 | 132 | 141 | 194 | 152 | 217 | 225 | 232 |
| PHG | 11+ answers | 339 | 32 | 33 | 57 | 66 | 90 | 81 | 171 | 194 | 209 |
| PHG | hop 3, 11+ | 226 | 19 | 20 | 44 | 53 | 56 | 54 | 58 | 81 | 96 |
| MTK | all | 1,998 | 1,082 | 1,089 | 1,258 | 1,284 | 1,519 | 1,343 | 1,677 | 1,699 | 1,751 |
| MTK | hop 2 | 666 | 213 | 214 | 315 | 317 | 531 | 357 | 666 | 666 | 666 |
| MTK | hop 3 | 666 | 205 | 211 | 279 | 303 | 324 | 320 | 345 | 367 | 419 |
| MTK | 5–10 answers | 287 | 75 | 78 | 120 | 129 | 179 | 143 | 213 | 221 | 232 |
| MTK | 11+ answers | 339 | 25 | 25 | 45 | 57 | 79 | 76 | 170 | 184 | 209 |
| MTK | hop 3, 11+ | 226 | 12 | 12 | 31 | 43 | 51 | 47 | 57 | 71 | 96 |

UNROUTED has no ROUTE, PUSH2, X2 or FAR2. Its bounds are:
- **+L2:** 1,105 in all; 234 on hop 2; 205 on hop 3.
- **+LOCAL:** 1,437; 523; 248.
- **+L2 +LOCAL:** 1,749; 666; 417 (5–10 answers 231, 11+ answers 208).

**38.5 Gold classes at B_N 1000.** The PUSH2 / FAR2 split of REACH_2 is in brackets.

| cell | stratum | golds | SERVED_EV | SERVED_FILL | CROWD_EV | CROWD_2L | CROWD_2X | CROWD_3 | REACH_EV | REACH_2 [PUSH2 / FAR2] | REACH_3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UNROUTED | all | 14,489 | 4,566 | 242 | 1,665 | 5,520 | 0 | 2,496 | 0 | 0 | 0 |
| UNROUTED | hop 2 | 4,613 | 878 | 74 | 1,007 | 2,654 | 0 | 0 | 0 | 0 | 0 |
| UNROUTED | hop 3 | 8,542 | 2,354 | 168 | 658 | 2,866 | 0 | 2,496 | 0 | 0 | 0 |
| PHG | all | 14,489 | 4,903 | 693 | 114 | 1,426 | 423 | 496 | 1,214 | 3,246 [2,194 / 1,052] | 1,974 |
| PHG | hop 1 | 1,334 | 1,328 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 |
| PHG | hop 2 | 4,613 | 821 | 148 | 75 | 775 | 0 | 0 | 989 | 1,805 [1,805 / 0] | 0 |
| PHG | hop 3 | 8,542 | 2,754 | 545 | 39 | 651 | 423 | 496 | 219 | 1,441 [389 / 1,052] | 1,974 |
| PHG | 11+ answers | 10,273 | 2,308 | 528 | 101 | 940 | 384 | 428 | 1,100 | 2,738 [1,762 / 976] | 1,746 |
| MTK | all | 14,489 | 4,826 | 589 | 147 | 1,445 | 318 | 517 | 1,258 | 3,429 [2,314 / 1,115] | 1,960 |
| MTK | hop 2 | 4,613 | 811 | 142 | 94 | 809 | 0 | 0 | 980 | 1,777 [1,777 / 0] | 0 |
| MTK | hop 3 | 8,542 | 2,688 | 447 | 53 | 636 | 318 | 517 | 271 | 1,652 [537 / 1,115] | 1,960 |

The user's two gold populations (B_N 1000):
- **Outside the one-hop L1 but inside the routed shards** (not in V_q, shard contacted). PHG has 3,038 golds (923 hop 2, 2,115 hop 3):
  - 693 are already served by the FLAT fill, i.e. co-location completes them;
  - 1,426 are CROWD_2L, 423 CROWD_2X and 496 CROWD_3;
  - by distance, 2,476 are at dS 2 and 562 at dS 3.
  - MTK has 2,869, of which 589 are fill-served.
- **Needing another STRUCT hop** (not in V_q, dS 2): 5,722 golds (hop 2 2,728, hop 3 2,994).
  - PHG: the gold's shard is contacted for 2,476 (627 of them fill-served); an intermediate is in a contacted shard for 4,104; an intermediate is in the gold's own shard for 790.
  - MTK: 2,293 / 530 / 4,159 / 931.

**38.6 How deep a second hop would have to go** (the probe: measured only, a traversal step).
- **The probe** is L2(u): the next application of IR_L1's operator restricted to the two STRUCT families, A^T(A^T x) with IR_L1's weights. By the ruling that is L3, not L1. C2 is the set of nodes outside V_q with L2 > 0, ordered by (−L2, FLAT rank).
- **C2 per query:** mean 17,074.59 nodes (min 4,568, max 23,395), about 40 % of the 43,234-node KB. The dS level sizes average 200 / 1,330.2 / 17,419.7 / 22,852.3 / 1,431.8 beyond. Inside the contacted shards, C2 averages 3,432.97 (PHG) and 3,359.07 (MTK), against a contacted mass of 8,652.76 and 8,633.53 nodes.
- **Hop-2 golds at dS 2 outside V_q** (2,728), by global rank in C2: < 100 for 2,130 (0.781), < 500 for 2,483, < 1,000 for 2,588 (0.949), < 5,000 for 2,720.
- **Hop-3 golds at dS 2 outside V_q** (2,994): < 100 for 56 (0.019), < 1,000 for 131, < 5,000 for 397 (0.133).
- **Crowded dS-2 golds** (CROWD_2L ∪ CROWD_2X), by rank among the C2 nodes inside the contacted shards:
  - PHG: hop 2 729 of 775 (0.941) < 100 and 774 < 1,000; hop 3 69 of 1,074 < 100 and 231 < 1,000.
  - MTK: hop 2 760 of 809 < 100; hop 3 70 of 954 < 100 and 227 < 1,000.
- **Why hop 3's dS-2 golds rank low.** They are reached through lower-ranked hits:
  - From the 200 hits, 3,181 of the 8,542 hop-3 golds are at dS 2. From the top-10 hits, only 900 are at distance 2, and 6,766 are at 3.
  - The topic entity is FLAT rank 0 in 1,851 rows, in the top 10 in 1,977, and a hit in all 1,998.
  - Every hop-2 gold is at dT 2; hop-3 golds are at dT 3 in 8,161 of 8,542.
- **Crowded evidence depth** (the L2 lever's golds, by restricted rank):
  - PHG's 114 lie at 1,004–1,890 (median 1,183); MTK's 147 at 1,000–1,673.
  - Unrouted, 1,665 golds are crowded, at 1,001–4,782 (median 1,891).

**38.7 B_N and query types.** ALL-gold rows per B_N, with the L2 +LOCAL bound in brackets:

| stratum | cell | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|
| hop 2 | UNROUTED | 136 (666) | 187 (666) | 235 (666) | 304 (666) |
| hop 2 | PHG | 174 (334) | 220 (334) | 251 (334) | 298 (334) |
| hop 2 | MTK | 164 (317) | 213 (317) | 244 (317) | 294 (317) |
| hop 3 | UNROUTED | 117 (417) | 162 (417) | 209 (420) | 301 (434) |
| hop 3 | PHG | 161 (322) | 228 (336) | 293 (361) | 332 (376) |
| hop 3 | MTK | 153 (292) | 205 (303) | 261 (311) | 291 (320) |
| 11+ answers | UNROUTED | 14 (208) | 21 (208) | 31 (209) | 72 (209) |
| 11+ answers | PHG | 18 (58) | 32 (66) | 58 (81) | 78 (88) |
| 11+ answers | MTK | 15 (50) | 25 (57) | 51 (64) | 62 (67) |

- **B_N does not close hop 2.** Even at B_N 5000, unrouted hop 2 is at 304 of 666, and routed hop 2 stays under its LOCAL bound (298 of 334 PHG, 294 of 317 MTK).
- **The worst query types** are the chains through a high-degree middle, especially movie → actor → movie (PHG, B_N 1000, ALL rows):
  - movie_to_actor_to_movie (hop 2): 1 of 59; L2 +LOCAL bound 2.
  - movie_to_actor_to_movie_to_writer: 2 of 67; bound 3.
  - movie_to_actor_to_movie_to_director: 1 of 56; bound 1.
  - movie_to_director_to_movie: 1 of 52 (unrouted 8); bound 8.
- **By contrast, the same chains ending in a low-cardinality attribute are served far better:**
  - movie_to_director_to_movie_to_year: 32 of 48;
  - movie_to_actor_to_movie_to_language: 24 of 40;
  - movie_to_writer_to_movie_to_year: 35 of 45.

**38.8 The answer** (development evidence; oracle bounds, not mechanisms).
- **Not L2.**
  - Under routing, L2 alone lifts ALL from 1,112 to 1,117 rows (PHG) and from 1,082 to 1,089 (MTK).
  - The routed crowded evidence is only 114 / 147 golds, and all of them lie within restricted rank 1,890.
  - Unrouted, L2 would matter more (1,015 → 1,105), because 1,665 golds crowd there. Routing already removes that crowding.
- **Hop 2 is one more STRUCT hop from local sources.**
  - Every hop-2 gold is at dT 2.
  - PHG misses 3,644 hop-2 golds. 2,580 of them are outside V_q at dS 2 (775 LOCAL, 1,805 PUSH2). 989 are in V_q in an uncontacted shard (ROUTE). 75 are crowded (L2).
  - On both maps, every missed hop-2 gold at dS 2 has an intermediate in a contacted shard: X2 = FAR2 = 0 on hop 2. The second hop's sources are therefore always inside the contacted shards.
  - **Partition-local completion (LOCAL)** lifts hop 2 from 220 to 334 of 666 rows (0.330 → 0.502). That is 114 of the 446 missing rows.
  - Letting that hop emit targets outside the contacted shards (PUSH2) lifts it to 548 (0.823). Adding ROUTE lifts it to 666.
  - MTK: 213 → 315 → 531 → 666.
  - The probe ranks these golds high: 2,130 of 2,728 are in the global top 100 of about 17,075 second-hop candidates.
  - The second hop is A^T(A^T x). By the ruling (§32, §37.2) that is L3's first step, not L1. Run partition-locally, it can recover at most the LOCAL part. The rest needs the step to emit targets in uncontacted shards (targeted fetches) or a wider fan-out.
- **Hop 3 and the large answer sets are L3 proper.**
  - HOP3 (dS ≥ 3) holds 2,470 of PHG's 5,243 missed hop-3 golds (MTK: 2,477 of 5,407).
  - Every lever except HOP3 together lifts hop 3 only from 228 to 421 of 666 (PHG; MTK 205 → 419).
  - Hop 3's dS-2 shortcuts run through lower-ranked hits, and the probe ranks them low (56 of 2,994 in the global top 100). A probe-ranked second hop at a small budget would not reach even that bound.
  - 11+ answers: 226 of the 339 rows are hop 3. ALL is 32 (PHG); every lever except HOP3 reaches 209.
- **In short:**
  - L2 is not the lever for the KB residual under routing.
  - Partition-local completion is the in-shard part of hop 2's missing hop (+114 rows at most).
  - The rest of hop 2 needs the same hop to reach targets outside the contacted shards.
  - Hop 3 and the 11+ answer sets need multi-step traversal from the topic.
  - Both are L3, so the KB path needs L3.

**38.9 Limits.**
- **Oracle bounds assume perfect recovery at zero exposure.** The second-hop candidate set is large: about 17,075 per query globally and about 3,433 inside PHG's contacted shards. Any real mechanism must be measured at equal exposure (§31's budget curves).
- **One scorer.** The probe is one fixed arithmetic (STRUCT only, IR_L1's weights). Another L3 scorer would rank differently.
- **Distances are from the hits.** dS is measured from the 200 hits, not the topic. dT uses the annotation, which the system cannot see, and enters no class.
- **Scope.** The two native-K maps (PHG and MTK at K 432), at B_N 500–5000. No other K and no other corpus was diagnosed.
- **LOCAL** requires both the gold's shard and an intermediate's shard to be contacted. It does not model how a shard stores adjacency that crosses shards.
- **Development only.** The rows are legacy-exposed and every number is descriptive; no p-value was computed.

**38.10 What comes next** (§37.9's order stands).
- **MetaQA development moves to L3, not L2.** On top of the carried-forward L1, on the same development rows:
  - a bounded second STRUCT hop, in its partition-local form and in a form that fetches its targets;
  - then a third hop for hop 3;
  - all measured at equal exposure, with fan-out and fetches counted.
- L1 is unchanged. The WebQSP shard maps continue (§37.10). Held-out A and B stay unread.

**38.11 Cost and hygiene.**
- **Run.** `_l1d_kbres.py` took 242.1 s on the laptop (peak RSS 778.4 MB; per row mean 77.7 ms, p95 187.0 ms). The L3 split read only the v1 json and npz.
- **Smoke runs need a multiple of 200 rows** (the runner's CQ). A 60-row smoke failed the FLAT identity check, because the matrix-product batch shape changes the float sums; 200 rows passed.
- **Hygiene.** Nothing under `data/` was written, and no pinned module or CONTRACT_FILE was edited. No held-out, split-B or TEST row was read.
- **New code:** `_l1d_kbres.py` and `_l1d_kbres_summary.py`. Each is sha-pinned by its record.

## 39. HOP: a bounded second and third STRUCT hop behind the carried-forward L1, at equal exposure (2026-09-29 local) (`_l3d_hop.py`, on `_l1d_kbres.py`'s constructor and the shared runner `_l1d_arms.py`; record `results/L3_DEV/l3hop_metaqa__v1.{json,npz}` (json sha df08ca0b…, npz 1f851d0b…); STATUS = DEVELOPMENT: descriptive numbers, p-values descriptive, no rule selected, L1 unchanged)

**39.1 The step.** §38.10 moved MetaQA development to L3: "a bounded second STRUCT hop, in its partition-local form and in a form that fetches its targets; then a third hop for hop 3; all measured at equal exposure, with fan-out and fetches counted." HOP is that measurement.
- **After L1, not inside it.** The traversal reads only the carried-forward L1's outputs (§37.2): the served order, the contacted shards, V_q and the STRUCT part of the IR_L1 score. L1 is unchanged.
- **Rows:** the L1_DEV population (c7f70806…), 1,998 rows (666 per hop) with 14,489 gold nodes, all legacy-exposed. No reserved held-out, split-B or TEST row was read.
- **No learned weight and no tuned constant.** The RRF constant is the frozen K0 60. The hop label is not visible to the system; it only stratifies the report.

**39.2 The operator and the arms.** The operator is the next application of IR_L1's arithmetic, restricted to the two STRUCT families (the §38 probe's operator).
- **LS** is IR_L1's score restricted to STRUCT_out ∪ STRUCT_in: the first hop's STRUCT mass, with support inside V_q.
- **prop:** y(u) = Σ_f Σ_{m ∈ src} A_f(m, u) g_f(m) w(m), over both STRUCT families (undirected), with g = 1/log2(1 + row length).
- **Hop 2:** y2 = prop(src2, LS). src2 is the LS support inside the contacted shards (all of it, unrouted).
- **Hop 3:** y3 = prop(src3, y2). src3 is the y2 support inside the contacted shards, or all of it; in the second case the adjacency rows of nodes in uncontacted shards are read, and counted.
- **Candidates.** C2 holds the nodes with y2 > 0 outside V_q, ordered by (−y2, FLAT rank). C3 holds the nodes with y3 > 0 outside V_q ∪ C2, ordered by (−y3, FLAT rank).
- **Merge:** RRF with K0. f2(u) = f(u) + [u ∈ C2]/(K0 + C2 rank(u)) and f3(u) = f2(u) + [u ∈ C3]/(K0 + C3 rank(u)); ties → FLAT rank.

The arms per routed cell (PHG_k432, MTK_k432) are below. Every arm serves exactly n(q, B_N) = min(B_N, contacted mass(q)) nodes, as L1 does (equal exposure). A served node outside the contacted shards is **fetched**; it is counted, with its distinct extra shards.

| arm | second hop | third hop |
|---|---|---|
| L1 | — | — |
| H2_LOCAL | sources and targets inside the contacted shards (partition-local completion) | — |
| H2_PUSH | sources inside the contacted shards, targets anywhere | — |
| H3_LOCAL | as H2_LOCAL | sources and targets inside the contacted shards |
| H3_PUSH | as H2_PUSH | sources: the y2 support inside the contacted shards; targets anywhere |
| H3_PUSH_ROWS | as H2_PUSH | sources: the whole y2 support (rows outside the contacted shards are read); targets anywhere |

UNROUTED contacts every shard and serves B_N nodes. Its arms are L1 (the IR_L1 order), H2 and H3.

**39.3 Identities asserted.**
- FLAT == loc v1, and the dense / SPLADE top-100 agreement gate holds.
- Per row, the recomputed ES shard order == kscale v1.
- On every gold node, these equal kbres v1: the UNROUTED L1 position, the routed contacted flag and restricted rank, and the UNROUTED C2 rank (the §38 probe).
- Per row, the contacted mass and |C2 UNROUTED| == kbres v1.
- The L1 ALL counts at every B_N == the kbres v1 tables (1,015 / 1,112 / 1,082 at B_N 1000).
- LOCAL arms never serve outside the contacted shards. Every arm serves exactly n(q, B_N) nodes.

**39.4 ALL-gold rows** (of 1,998; share in brackets).

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 763 (0.382) | 855 (0.428) | 919 (0.460) | 1,015 (0.508) | 1,110 (0.556) | 1,271 (0.636) |
| UNROUTED | H2 | 970 (0.485) | 1,102 (0.552) | 1,169 (0.585) | 1,259 (0.630) | 1,395 (0.698) | 1,560 (0.781) |
| UNROUTED | H3 | 916 (0.458) | 1,070 (0.536) | 1,145 (0.573) | 1,235 (0.618) | 1,364 (0.683) | 1,548 (0.775) |
| PHG | L1 | 763 (0.382) | 876 (0.438) | 999 (0.500) | 1,112 (0.557) | 1,208 (0.605) | 1,294 (0.648) |
| PHG | H2_LOCAL | 830 (0.415) | 938 (0.469) | 1,004 (0.503) | 1,122 (0.562) | 1,231 (0.616) | 1,360 (0.681) |
| PHG | H2_PUSH | 974 (0.487) | 1,104 (0.553) | 1,175 (0.588) | 1,287 (0.644) | 1,390 (0.696) | 1,485 (0.743) |
| PHG | H3_LOCAL | 792 (0.396) | 911 (0.456) | 984 (0.492) | 1,055 (0.528) | 1,183 (0.592) | 1,348 (0.675) |
| PHG | H3_PUSH | 930 (0.465) | 1,077 (0.539) | 1,161 (0.581) | 1,237 (0.619) | 1,358 (0.680) | 1,486 (0.744) |
| PHG | H3_PUSH_ROWS | 923 (0.462) | 1,069 (0.535) | 1,149 (0.575) | 1,228 (0.615) | 1,352 (0.677) | 1,512 (0.757) |
| MTK | L1 | 763 (0.382) | 871 (0.436) | 981 (0.491) | 1,082 (0.542) | 1,169 (0.585) | 1,249 (0.625) |
| MTK | H2_LOCAL | 821 (0.411) | 928 (0.464) | 992 (0.496) | 1,096 (0.549) | 1,193 (0.597) | 1,293 (0.647) |
| MTK | H2_PUSH | 974 (0.487) | 1,103 (0.552) | 1,175 (0.588) | 1,274 (0.638) | 1,360 (0.681) | 1,452 (0.727) |
| MTK | H3_LOCAL | 790 (0.395) | 904 (0.452) | 971 (0.486) | 1,038 (0.520) | 1,145 (0.573) | 1,287 (0.644) |
| MTK | H3_PUSH | 929 (0.465) | 1,084 (0.543) | 1,164 (0.583) | 1,237 (0.619) | 1,342 (0.672) | 1,461 (0.731) |
| MTK | H3_PUSH_ROWS | 921 (0.461) | 1,074 (0.538) | 1,154 (0.578) | 1,231 (0.616) | 1,342 (0.672) | 1,486 (0.744) |

**39.5 By hop** (666 rows each). Hop 2:

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 42 (0.063) | 104 (0.156) | 136 (0.204) | 187 (0.281) | 235 (0.353) | 304 (0.456) |
| UNROUTED | H2 | 277 (0.416) | 354 (0.532) | 406 (0.610) | 457 (0.686) | 535 (0.803) | 614 (0.922) |
| UNROUTED | H3 | 253 (0.380) | 330 (0.495) | 382 (0.574) | 433 (0.650) | 506 (0.760) | 596 (0.895) |
| PHG | L1 | 42 (0.063) | 115 (0.173) | 174 (0.261) | 220 (0.330) | 251 (0.377) | 298 (0.447) |
| PHG | H2_LOCAL | 136 (0.204) | 191 (0.287) | 231 (0.347) | 292 (0.438) | 325 (0.488) | 334 (0.502) |
| PHG | H2_PUSH | 280 (0.420) | 358 (0.538) | 406 (0.610) | 472 (0.709) | 526 (0.790) | 545 (0.818) |
| PHG | H3_PUSH_ROWS | 265 (0.398) | 329 (0.494) | 389 (0.584) | 426 (0.640) | 490 (0.736) | 539 (0.809) |
| MTK | L1 | 42 (0.063) | 113 (0.170) | 164 (0.246) | 213 (0.320) | 244 (0.366) | 294 (0.441) |
| MTK | H2_LOCAL | 127 (0.191) | 180 (0.270) | 219 (0.329) | 273 (0.410) | 308 (0.462) | 317 (0.476) |
| MTK | H2_PUSH | 281 (0.422) | 357 (0.536) | 405 (0.608) | 461 (0.692) | 506 (0.760) | 528 (0.793) |
| MTK | H3_PUSH_ROWS | 260 (0.390) | 330 (0.495) | 388 (0.583) | 422 (0.634) | 478 (0.718) | 520 (0.781) |

Hop 3:

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 65 (0.098) | 87 (0.131) | 117 (0.176) | 162 (0.243) | 209 (0.314) | 301 (0.452) |
| UNROUTED | H2 | 55 (0.083) | 86 (0.129) | 99 (0.149) | 136 (0.204) | 194 (0.291) | 280 (0.420) |
| UNROUTED | H3 | 51 (0.077) | 79 (0.119) | 99 (0.149) | 137 (0.206) | 192 (0.288) | 286 (0.429) |
| PHG | L1 | 65 (0.098) | 99 (0.149) | 161 (0.242) | 228 (0.342) | 293 (0.440) | 332 (0.498) |
| PHG | H2_LOCAL | 55 (0.083) | 85 (0.128) | 111 (0.167) | 166 (0.249) | 242 (0.363) | 362 (0.544) |
| PHG | H2_PUSH | 55 (0.083) | 84 (0.126) | 107 (0.161) | 151 (0.227) | 200 (0.300) | 276 (0.414) |
| PHG | H3_PUSH_ROWS | 50 (0.075) | 79 (0.119) | 98 (0.147) | 138 (0.207) | 198 (0.297) | 309 (0.464) |
| MTK | L1 | 65 (0.098) | 95 (0.143) | 153 (0.230) | 205 (0.308) | 261 (0.392) | 291 (0.437) |
| MTK | H2_LOCAL | 55 (0.083) | 86 (0.129) | 110 (0.165) | 159 (0.239) | 221 (0.332) | 312 (0.468) |
| MTK | H2_PUSH | 55 (0.083) | 84 (0.126) | 107 (0.161) | 149 (0.224) | 190 (0.285) | 260 (0.390) |
| MTK | H3_PUSH_ROWS | 53 (0.080) | 83 (0.125) | 103 (0.155) | 145 (0.218) | 200 (0.300) | 302 (0.453) |

**39.6 Large answer sets.** 11+ answers (339 rows):

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 8 (0.024) | 13 (0.038) | 14 (0.041) | 21 (0.062) | 31 (0.091) | 72 (0.212) |
| UNROUTED | H2 | 8 (0.024) | 16 (0.047) | 21 (0.062) | 25 (0.074) | 42 (0.124) | 93 (0.274) |
| UNROUTED | H3 | 4 (0.012) | 15 (0.044) | 18 (0.053) | 22 (0.065) | 33 (0.097) | 75 (0.221) |
| PHG | L1 | 8 (0.024) | 12 (0.035) | 18 (0.053) | 32 (0.094) | 58 (0.171) | 78 (0.230) |
| PHG | H2_LOCAL | 6 (0.018) | 12 (0.035) | 12 (0.035) | 15 (0.044) | 32 (0.094) | 83 (0.245) |
| PHG | H2_PUSH | 8 (0.024) | 16 (0.047) | 18 (0.053) | 22 (0.065) | 35 (0.103) | 63 (0.186) |
| PHG | H3_PUSH_ROWS | 4 (0.012) | 15 (0.044) | 18 (0.053) | 18 (0.053) | 27 (0.080) | 52 (0.153) |
| MTK | L1 | 8 (0.024) | 12 (0.035) | 15 (0.044) | 25 (0.074) | 51 (0.150) | 62 (0.183) |
| MTK | H2_LOCAL | 6 (0.018) | 12 (0.035) | 13 (0.038) | 17 (0.050) | 26 (0.077) | 65 (0.192) |
| MTK | H2_PUSH | 8 (0.024) | 16 (0.047) | 18 (0.053) | 19 (0.056) | 27 (0.080) | 56 (0.165) |
| MTK | H3_PUSH_ROWS | 4 (0.012) | 15 (0.044) | 18 (0.053) | 19 (0.056) | 25 (0.074) | 42 (0.124) |

Hop 3 with 11+ answers (226 rows):

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 0 (0.000) | 0 (0.000) | 0 (0.000) | 4 (0.018) | 10 (0.044) | 43 (0.190) |
| UNROUTED | H2 | 0 (0.000) | 0 (0.000) | 0 (0.000) | 0 (0.000) | 5 (0.022) | 26 (0.115) |
| UNROUTED | H3 | 0 (0.000) | 0 (0.000) | 0 (0.000) | 0 (0.000) | 1 (0.004) | 19 (0.084) |
| PHG | L1 | 0 (0.000) | 0 (0.000) | 5 (0.022) | 19 (0.084) | 45 (0.199) | 65 (0.288) |
| PHG | H2_LOCAL | 0 (0.000) | 0 (0.000) | 0 (0.000) | 3 (0.013) | 19 (0.084) | 70 (0.310) |
| PHG | H2_PUSH | 0 (0.000) | 0 (0.000) | 0 (0.000) | 1 (0.004) | 8 (0.035) | 30 (0.133) |
| PHG | H3_PUSH_ROWS | 0 (0.000) | 0 (0.000) | 0 (0.000) | 0 (0.000) | 5 (0.022) | 23 (0.102) |
| MTK | L1 | 0 (0.000) | 0 (0.000) | 3 (0.013) | 12 (0.053) | 37 (0.164) | 48 (0.212) |
| MTK | H2_LOCAL | 0 (0.000) | 0 (0.000) | 0 (0.000) | 4 (0.018) | 12 (0.053) | 51 (0.226) |
| MTK | H2_PUSH | 0 (0.000) | 0 (0.000) | 0 (0.000) | 0 (0.000) | 6 (0.027) | 29 (0.128) |
| MTK | H3_PUSH_ROWS | 0 (0.000) | 0 (0.000) | 0 (0.000) | 0 (0.000) | 4 (0.018) | 20 (0.088) |

Hop 2 with 5–10 answers (90 rows):

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 0 (0.000) | 0 (0.000) | 4 (0.044) | 8 (0.089) | 15 (0.167) | 25 (0.278) |
| UNROUTED | H2 | 20 (0.222) | 31 (0.344) | 36 (0.400) | 47 (0.522) | 67 (0.744) | 86 (0.956) |
| UNROUTED | H3 | 10 (0.111) | 29 (0.322) | 32 (0.356) | 43 (0.478) | 59 (0.656) | 84 (0.933) |
| PHG | L1 | 0 (0.000) | 0 (0.000) | 3 (0.033) | 9 (0.100) | 15 (0.167) | 17 (0.189) |
| PHG | H2_LOCAL | 0 (0.000) | 0 (0.000) | 2 (0.022) | 10 (0.111) | 18 (0.200) | 18 (0.200) |
| PHG | H2_PUSH | 20 (0.222) | 31 (0.344) | 35 (0.389) | 47 (0.522) | 66 (0.733) | 68 (0.756) |
| PHG | H3_PUSH_ROWS | 15 (0.167) | 29 (0.322) | 32 (0.356) | 38 (0.422) | 53 (0.589) | 67 (0.744) |
| MTK | L1 | 0 (0.000) | 0 (0.000) | 2 (0.022) | 7 (0.078) | 13 (0.144) | 15 (0.167) |
| MTK | H2_LOCAL | 1 (0.011) | 2 (0.022) | 3 (0.033) | 8 (0.089) | 17 (0.189) | 17 (0.189) |
| MTK | H2_PUSH | 21 (0.233) | 31 (0.344) | 34 (0.378) | 44 (0.489) | 58 (0.644) | 62 (0.689) |
| MTK | H3_PUSH_ROWS | 13 (0.144) | 30 (0.333) | 32 (0.356) | 36 (0.400) | 48 (0.533) | 60 (0.667) |

**39.7 Paired against the cell's L1** (rows gained / lost, McNemar p, descriptive). At B_N 1000:

| cell | arm | all rows | hop 1 | hop 2 | hop 3 |
|---|---|---|---|---|---|
| UNROUTED | H2 | +310 / −66 (p 6.4e-39) | 0 / 0 | +303 / −33 | +7 / −33 |
| UNROUTED | H3 | +313 / −93 (p 6.2e-29) | 0 / −1 | +292 / −46 | +21 / −46 |
| PHG | H2_LOCAL | +115 / −105 (p 0.54) | 0 / 0 | +105 / −33 | +10 / −72 |
| PHG | H2_PUSH | +293 / −118 (p 2.8e-18) | 0 / 0 | +288 / −36 | +5 / −82 |
| PHG | H3_LOCAL | +114 / −171 (p 8.8e-04) | 0 / 0 | +99 / −66 | +15 / −105 |
| PHG | H3_PUSH | +300 / −175 (p 1.1e-08) | 0 / 0 | +276 / −67 | +24 / −108 |
| PHG | H3_PUSH_ROWS | +292 / −176 (p 9.2e-08) | 0 / 0 | +275 / −69 | +17 / −107 |
| MTK | H2_LOCAL | +100 / −86 (p 0.34) | 0 / 0 | +92 / −32 | +8 / −54 |
| MTK | H2_PUSH | +289 / −97 (p 2.9e-23) | 0 / 0 | +285 / −37 | +4 / −60 |
| MTK | H3_LOCAL | +102 / −146 (p 6.2e-03) | 0 / 0 | +89 / −64 | +13 / −82 |
| MTK | H3_PUSH | +306 / −151 (p 3.5e-13) | 0 / 0 | +278 / −68 | +28 / −83 |
| MTK | H3_PUSH_ROWS | +302 / −153 (p 2.5e-12) | 0 / 0 | +277 / −68 | +25 / −85 |

At B_N 5000:

| cell | arm | all rows | hop 1 | hop 2 | hop 3 |
|---|---|---|---|---|---|
| UNROUTED | H2 | +335 / −46 (p 2.7e-55) | 0 / 0 | +313 / −3 | +22 / −43 |
| UNROUTED | H3 | +349 / −72 (p 1.1e-44) | 0 / 0 | +301 / −9 | +48 / −63 |
| PHG | H2_LOCAL | +77 / −11 (p 2.4e-13) | 0 / 0 | +36 / 0 | +41 / −11 |
| PHG | H2_PUSH | +257 / −66 (p 9.2e-28) | 0 / 0 | +247 / 0 | +10 / −66 |
| PHG | H3_LOCAL | +68 / −14 (p 1.1e-09) | 0 / 0 | +36 / 0 | +32 / −14 |
| PHG | H3_PUSH | +273 / −81 (p 1.9e-25) | 0 / 0 | +242 / 0 | +31 / −81 |
| PHG | H3_PUSH_ROWS | +301 / −83 (p 3.9e-30) | 0 / 0 | +242 / −1 | +59 / −82 |
| MTK | H2_LOCAL | +51 / −7 (p 2.4e-09) | 0 / 0 | +23 / 0 | +28 / −7 |
| MTK | H2_PUSH | +251 / −48 (p 2.5e-34) | 0 / 0 | +234 / 0 | +17 / −48 |
| MTK | H3_LOCAL | +48 / −10 (p 4.5e-07) | 0 / 0 | +23 / 0 | +25 / −10 |
| MTK | H3_PUSH | +284 / −72 (p 6.7e-31) | 0 / 0 | +228 / 0 | +56 / −72 |
| MTK | H3_PUSH_ROWS | +310 / −73 (p 7.2e-36) | 0 / 0 | +227 / −1 | +83 / −72 |

**39.8 Fetches and traversal size.** The fetches per query (served nodes outside the contacted shards):

| cell | arm | B_N | fetched nodes per query: mean (share of served) | extra shards per query: mean / p95 / max |
|---|---|---|---|---|
| PHG | H2_PUSH | 100 | 17.2 (0.172) | 15.6 / 23.0 / 27 |
| PHG | H2_PUSH | 500 | 174.5 (0.349) | 126.8 / 156.0 / 179 |
| PHG | H2_PUSH | 1000 | 450.2 (0.450) | 234.6 / 267.0 / 292 |
| PHG | H2_PUSH | 5000 | 3,117.3 (0.624) | 341.2 / 364.0 / 391 |
| PHG | H3_PUSH | 100 | 20.2 (0.202) | 17.7 / 24.0 / 31 |
| PHG | H3_PUSH | 500 | 185.7 (0.371) | 122.7 / 146.0 / 173 |
| PHG | H3_PUSH | 1000 | 462.0 (0.462) | 224.7 / 253.0 / 297 |
| PHG | H3_PUSH | 5000 | 2,913.5 (0.583) | 343.2 / 365.0 / 389 |
| PHG | H3_PUSH_ROWS | 100 | 16.5 (0.165) | 14.4 / 21.0 / 29 |
| PHG | H3_PUSH_ROWS | 500 | 179.4 (0.359) | 111.1 / 134.0 / 157 |
| PHG | H3_PUSH_ROWS | 1000 | 467.3 (0.467) | 214.8 / 243.0 / 272 |
| PHG | H3_PUSH_ROWS | 5000 | 3,227.9 (0.646) | 343.4 / 366.0 / 390 |
| MTK | H2_PUSH | 100 | 17.1 (0.171) | 15.5 / 23.0 / 29 |
| MTK | H2_PUSH | 500 | 165.5 (0.331) | 122.4 / 152.0 / 175 |
| MTK | H2_PUSH | 1000 | 433.4 (0.433) | 231.0 / 264.0 / 293 |
| MTK | H2_PUSH | 5000 | 3,077.8 (0.616) | 344.4 / 369.0 / 400 |
| MTK | H3_PUSH | 100 | 22.7 (0.227) | 19.7 / 27.0 / 33 |
| MTK | H3_PUSH | 500 | 189.2 (0.378) | 127.4 / 149.0 / 180 |
| MTK | H3_PUSH | 1000 | 465.8 (0.466) | 229.7 / 258.0 / 290 |
| MTK | H3_PUSH | 5000 | 2,926.2 (0.586) | 345.1 / 369.0 / 394 |
| MTK | H3_PUSH_ROWS | 100 | 22.8 (0.228) | 19.5 / 27.0 / 35 |
| MTK | H3_PUSH_ROWS | 500 | 195.4 (0.391) | 123.5 / 146.0 / 177 |
| MTK | H3_PUSH_ROWS | 1000 | 490.7 (0.491) | 225.7 / 252.0 / 284 |
| MTK | H3_PUSH_ROWS | 5000 | 3,285.9 (0.658) | 345.0 / 369.0 / 393 |

The traversal size per query, over all 1,998 rows:
- **src2:** the second-hop sources. **src3** (UNROUTED), **src3c** and **src3r:** the third-hop sources, all of them / inside the contacted shards (H3_PUSH) / anywhere (H3_PUSH_ROWS).
- **ent2, ent3, ent3c, ent3r:** the adjacency entries those sources scan.
- **nC2, nC3** (UNROUTED): the candidate lists. **nC2L / nC2P:** the second-hop candidates inside the contacted shards / anywhere.
- **rows3_out / shards3_out:** the rows H3_PUSH_ROWS reads outside the contacted shards, and their distinct shards.

| cell | quantity | mean | median | p90 | max |
|---|---|---|---|---|---|
| UNROUTED | src2 | 1,344.6 | 1,280.5 | 1,772.3 | 7,584 |
| UNROUTED | ent2 | 42,851.6 | 43,613.0 | 50,385.3 | 96,276 |
| UNROUTED | nC2 | 17,074.6 | 17,094.0 | 18,205.5 | 23,395 |
| UNROUTED | src3 | 18,577.1 | 18,536.5 | 20,180.3 | 30,613 |
| UNROUTED | ent3 | 186,011.5 | 187,076.5 | 196,396.8 | 227,745 |
| UNROUTED | nC3 | 22,663.1 | 22,748.5 | 24,004.0 | 25,486 |
| PHG | src2 | 459.1 | 446.5 | 588.3 | 2,118 |
| PHG | ent2 | 32,991.5 | 33,873.5 | 39,530.2 | 50,963 |
| PHG | nC2L | 2,880.2 | 2,882.0 | 3,604.3 | 5,494 |
| PHG | nC2P | 14,314.8 | 14,649.0 | 15,246.3 | 15,699 |
| PHG | src3c | 3,370.0 | 3,363.0 | 4,213.2 | 7,743 |
| PHG | ent3c | 61,657.9 | 62,983.0 | 72,144.2 | 92,663 |
| PHG | src3r | 15,637.7 | 16,013.5 | 16,698.3 | 21,897 |
| PHG | ent3r | 160,931.9 | 163,292.0 | 171,311.0 | 202,521 |
| PHG | rows3_out | 12,267.7 | 12,536.0 | 13,467.0 | 15,492 |
| PHG | shards3_out | 343.7 | 346.0 | 361.0 | 400 |
| MTK | src2 | 469.1 | 459.0 | 608.0 | 2,196 |
| MTK | ent2 | 32,480.8 | 33,453.5 | 38,255.6 | 51,169 |
| MTK | nC2L | 2,896.6 | 2,917.0 | 3,641.0 | 5,349 |
| MTK | nC2P | 14,370.9 | 14,729.0 | 15,230.3 | 15,841 |
| MTK | src3c | 3,418.6 | 3,425.5 | 4,280.3 | 7,692 |
| MTK | ent3c | 60,957.6 | 61,939.0 | 71,251.0 | 87,927 |
| MTK | src3r | 15,702.1 | 16,033.0 | 16,775.9 | 21,977 |
| MTK | ent3r | 161,714.2 | 163,960.0 | 172,259.5 | 202,698 |
| MTK | rows3_out | 12,283.5 | 12,551.5 | 13,411.0 | 15,004 |
| MTK | shards3_out | 345.0 | 345.0 | 363.0 | 402 |

**39.9 Reading** (development numbers, descriptive).
- **The untyped second hop recovers hop 2.**
  - PHG H2_PUSH serves ALL gold on 472 of the 666 hop-2 rows at B_N 1000 (0.709), against L1's 220 (0.330). MTK: 461 against 213.
  - H2_LOCAL reaches 292 (PHG). At B_N 5000 it reaches 334, exactly §38.4's LOCAL bound; MTK reaches 317, its bound too.
- **At equal exposure it costs hop 3.** H2_PUSH's candidates displace hop-3 golds that L1 served.
  - PHG hop 3 falls from 228 to 151 (+5 / −82 at B_N 1000).
  - Over all rows H2_PUSH is +293 / −118 (p 2.8e-18); H2_LOCAL is +115 / −105 (p 0.54).
- **An untyped third hop does not help hop 3.**
  - At B_N 1000, H3_PUSH_ROWS serves 138 hop-3 rows (PHG), under L1's 228. All rows fall from H2_PUSH's 1,287 to 1,228.
  - The frontier explodes. Unrouted, C2 has a mean 17,074.6 of the 43,234 nodes and C3 22,663.1. H3_PUSH_ROWS reads a mean 12,283.5 rows (MTK) outside the contacted shards, from 345.0 of the 432 shards.
- **The 11+ answer rows stay out of reach.** No hop arm serves more than 22 of the 339 rows at B_N 1000, under PHG L1's 32.
- **The fetch cost is high.**
  - At B_N 1000, H2_PUSH fetches a mean 450.2 served nodes per query (0.450 of the served list), from 234.6 extra shards.
  - At B_N 5000 it fetches 3,117.3 nodes from 341.2 shards, most of the 432-shard map.
- **So:** traversal recovers hop 2's missing hop, but an untyped frontier is the wrong operator for hop 3 and for large answer sets. The next arm types the walk (§40).

**39.10 Cost and hygiene.**
- **Run.** `_l3d_hop.py` took 295.2 s on the laptop (peak RSS 784.6 MB). The smoke runs used multiples of 200 rows (§38.11).
- **Hygiene.** Nothing under `data/` was written, and no pinned module or CONTRACT_FILE was edited. No held-out, split-B or TEST row was read.
- **New code:** `_l3d_hop.py` (70777dd9…), sha-pinned by its record. The record also pins the carry-forward record (aaae66fa…) and the kscale v1 record and npz, and asserts identity with the kbres v1 npz.

## 40. TYPED: the relation-typed scheduled walk as L3 completion, its read budget, and the host L3 lane (2026-09-29 local) (`_l3d_typed.py`, which imports the L1_P90 lane's functions sha-pinned to `results/L1_P90_EXPLOIT/code_snapshot/`, and `_l3d_budget.py`; records `results/L3_DEV/l3typed_metaqa__v1.{json,npz}` (json a6e80216…, npz 0ee65e76…) and `l3budget_metaqa__v1.{json,npz}` (json c96eecd2…, npz db20afb6…); host lane `_l3h_run.py`, `_l3h_place.py`, `_l3h_bundle.py`, `_l3h_blasprobe.py`, records `results/L3_HOST/`; STATUS = DEVELOPMENT: descriptive numbers, p-values descriptive, no rule or budget selected, L1 unchanged)

**40.1 Why a typed walk.** HOP (§39) recovered hop 2 but hurt hop 3: an untyped frontier is too wide.
- The L1_P90 lane (2026-09-14, DEV_B confirmed, gate G2) had a relation-typed walk that took MetaQA hop 3 to 0.944 at P50 blocks.
- That walk is traversal, so under the 2026-09-26 boundary it belongs to L3.
- TYPED measures it there: at node level, behind the carried-forward routing, on §39's rows and with §39's exposure rule.

**40.2 The typed channel** (the lane's functions, imported unchanged).
- **T** is the served structural family as a relation-typed undirected multigraph (deduplicated, loop-free). The graph counts as typed iff it carries at least 2 relation labels.
- **Seeds** are the nodes whose full name occurs verbatim in the question (longest spans, case-exact preferred, cap 5). With no such node, the lane's fallback applies: the row's dense top-1 and SPLADE top-1.
- **Schedule:** `schedule3(T, question, seeds, key="wh")` → (relation order, anchored). It matches relation-name tokens to question tokens outside the seed spans, and orders the matched relations by their position relative to the seed.
- **Gate G2** (the lane's confirmed gate) is: typed graph and anchored. Outside the gate every typed arm serves exactly the L1 list (asserted).
- **Walk:** the lane's hard scheduled frontier walk, H 3.
  - Step k may use the relations order[:k].
  - The mass pushed along the schedule's last relation is the answer channel ans; the rest is oth.
  - Nodes already reached are zeroed, and each layer carries unit mass.
  - The untyped walk (empty schedule) gives unt.
- **Score:** M(u) = seed(u) + ans(u) + EPS (oth(u) + unt(u)), with EPS 1e-6 (the lane's constant).
- **CT**, the typed candidates: the nodes with seed + ans > 0, ordered by (−M, FLAT rank).
- **Read beyond L1:** the question text of the population rows (a TEST row is refused), the node names and the structural relation labels. The topic-entity annotation enters only a diagnostic (is the topic among the seeds?). The hop label only stratifies the report.

**40.3 Masks and arms.** The walk is re-implemented with shard masks. Unmasked, it equals the pinned relwalk bit for bit at float32, asserted on every row.
- **ROWS:** unmasked. Any adjacency row may be read; rows of nodes in uncontacted shards are counted, with their distinct shards. Targets may be anywhere.
- **PUSH:** a step expands only the mass on contacted nodes. Arrivals anywhere are kept as candidates but not expanded further.
- **LOCAL:** seeds, sources and targets all stay inside the contacted shards.

The arms per routed cell (PHG_k432, MTK_k432), each at equal exposure n(q, B_N):
- **L1** and **H2_PUSH:** §39's references, asserted equal to the HOP v1 record.
- **T1_<mask>:** CT first, then the L1 order of the remaining nodes.
- **TR_<mask>:** RRF with K0, f(u) + [u ∈ CT]/(K0 + CT rank(u)); ties → FLAT rank.
- **TR_H2_PUSH:** the RRF of the L1 score, §39's C2 (PUSH) and CT (PUSH).
- **UNROUTED:** L1, H2, T1, TR and TR_H2.

**40.4 Identities asserted.**
- FLAT == loc v1, and the agreement gate holds.
- Per row, the unmasked typed and untyped walks == the pinned relwalk.
- On every gold node, these equal kbres v1: the UNROUTED L1 position, the routed contacted flag and L1 rank, and the contacted mass. The L1, H2 and H2_PUSH positions == the HOP v1 record.
- The L1 ALL counts == kbres v1, and the H2 / H2_PUSH ALL counts == HOP v1.
- LOCAL arms serve only contacted nodes, and exactly the contacted mass. Outside the gate every typed arm == L1.

**40.5 Gate, seeds and schedules.**

| stratum | rows | lexical seeds | anchored = gated | dense+SPLADE fallback | topic in seeds (gated) | schedule length 0 / 1 / 2 (gated) | seeds 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|
| all | 1,998 | 1,997 | 1,994 | 1 | 1,994 | 149 / 798 / 1,047 | 1,793 / 201 / 4 |
| hop1 | 666 | 665 | 665 | 1 | 665 | 146 / 519 / 0 | 570 / 94 / 2 |
| hop2 | 666 | 666 | 664 | 0 | 664 | 3 / 245 / 416 | 582 / 83 / 1 |
| hop3 | 666 | 666 | 665 | 0 | 665 | 0 / 34 / 631 | 641 / 24 / 1 |

- 1,994 of the 1,998 rows are gated. 1,997 have lexical seeds; 1 uses the dense + SPLADE fallback. The topic entity is among the seeds on every gated row.
- No schedule has more than 2 relations. The walk still takes 3 steps, because its later steps reuse the whole order: the chain movie → director → movie → genre needs only {directed_by, has_genre}.

**40.6 ALL-gold rows** (of 1,998).

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 763 (0.382) | 855 (0.428) | 919 (0.460) | 1,015 (0.508) | 1,110 (0.556) | 1,271 (0.636) |
| UNROUTED | H2 | 970 (0.485) | 1,102 (0.552) | 1,169 (0.585) | 1,259 (0.630) | 1,395 (0.698) | 1,560 (0.781) |
| UNROUTED | T1 | 1,916 (0.959) | 1,960 (0.981) | 1,962 (0.982) | 1,966 (0.984) | 1,968 (0.985) | 1,972 (0.987) |
| UNROUTED | TR | 1,778 (0.890) | 1,930 (0.966) | 1,960 (0.981) | 1,966 (0.984) | 1,968 (0.985) | 1,972 (0.987) |
| UNROUTED | TR_H2 | 1,748 (0.875) | 1,913 (0.957) | 1,960 (0.981) | 1,966 (0.984) | 1,970 (0.986) | 1,980 (0.991) |
| PHG | L1 | 763 (0.382) | 876 (0.438) | 999 (0.500) | 1,112 (0.557) | 1,208 (0.605) | 1,294 (0.648) |
| PHG | H2_PUSH | 974 (0.487) | 1,104 (0.553) | 1,175 (0.588) | 1,287 (0.644) | 1,390 (0.696) | 1,485 (0.743) |
| PHG | T1_LOCAL | 1,101 (0.551) | 1,138 (0.570) | 1,195 (0.598) | 1,256 (0.629) | 1,306 (0.654) | 1,334 (0.668) |
| PHG | TR_LOCAL | 1,101 (0.551) | 1,138 (0.570) | 1,195 (0.598) | 1,256 (0.629) | 1,306 (0.654) | 1,334 (0.668) |
| PHG | T1_PUSH | 1,433 (0.717) | 1,481 (0.741) | 1,541 (0.771) | 1,602 (0.802) | 1,655 (0.828) | 1,683 (0.842) |
| PHG | TR_PUSH | 1,385 (0.693) | 1,466 (0.734) | 1,541 (0.771) | 1,602 (0.802) | 1,655 (0.828) | 1,683 (0.842) |
| PHG | T1_ROWS | 1,916 (0.959) | 1,959 (0.980) | 1,963 (0.982) | 1,967 (0.984) | 1,970 (0.986) | 1,973 (0.987) |
| PHG | TR_ROWS | 1,778 (0.890) | 1,933 (0.967) | 1,963 (0.982) | 1,967 (0.984) | 1,970 (0.986) | 1,973 (0.987) |
| PHG | TR_H2_PUSH | 1,377 (0.689) | 1,446 (0.724) | 1,490 (0.746) | 1,532 (0.767) | 1,575 (0.788) | 1,640 (0.821) |
| MTK | L1 | 763 (0.382) | 871 (0.436) | 981 (0.491) | 1,082 (0.542) | 1,169 (0.585) | 1,249 (0.625) |
| MTK | H2_PUSH | 974 (0.487) | 1,103 (0.552) | 1,175 (0.588) | 1,274 (0.638) | 1,360 (0.681) | 1,452 (0.727) |
| MTK | T1_LOCAL | 1,073 (0.537) | 1,112 (0.557) | 1,166 (0.584) | 1,214 (0.608) | 1,257 (0.629) | 1,279 (0.640) |
| MTK | TR_LOCAL | 1,072 (0.537) | 1,112 (0.557) | 1,166 (0.584) | 1,214 (0.608) | 1,257 (0.629) | 1,279 (0.640) |
| MTK | T1_PUSH | 1,435 (0.718) | 1,485 (0.743) | 1,540 (0.771) | 1,589 (0.795) | 1,634 (0.818) | 1,659 (0.830) |
| MTK | TR_PUSH | 1,388 (0.695) | 1,466 (0.734) | 1,540 (0.771) | 1,589 (0.795) | 1,634 (0.818) | 1,659 (0.830) |
| MTK | T1_ROWS | 1,916 (0.959) | 1,960 (0.981) | 1,963 (0.982) | 1,967 (0.984) | 1,970 (0.986) | 1,973 (0.987) |
| MTK | TR_ROWS | 1,778 (0.890) | 1,932 (0.967) | 1,962 (0.982) | 1,967 (0.984) | 1,970 (0.986) | 1,973 (0.987) |
| MTK | TR_H2_PUSH | 1,379 (0.690) | 1,452 (0.727) | 1,493 (0.747) | 1,535 (0.768) | 1,575 (0.788) | 1,631 (0.816) |

**40.7 By hop and answer count.** Hop 2 (666 rows):

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 42 (0.063) | 104 (0.156) | 136 (0.204) | 187 (0.281) | 235 (0.353) | 304 (0.456) |
| UNROUTED | H2 | 277 (0.416) | 354 (0.532) | 406 (0.610) | 457 (0.686) | 535 (0.803) | 614 (0.922) |
| UNROUTED | T1 | 644 (0.967) | 659 (0.989) | 659 (0.989) | 659 (0.989) | 660 (0.991) | 662 (0.994) |
| PHG | L1 | 42 (0.063) | 115 (0.173) | 174 (0.261) | 220 (0.330) | 251 (0.377) | 298 (0.447) |
| PHG | H2_PUSH | 280 (0.420) | 358 (0.538) | 406 (0.610) | 472 (0.709) | 526 (0.790) | 545 (0.818) |
| PHG | T1_LOCAL | 324 (0.486) | 330 (0.495) | 330 (0.495) | 333 (0.500) | 333 (0.500) | 333 (0.500) |
| PHG | T1_PUSH | 641 (0.962) | 657 (0.986) | 657 (0.986) | 660 (0.991) | 660 (0.991) | 660 (0.991) |
| PHG | T1_ROWS | 644 (0.967) | 659 (0.989) | 659 (0.989) | 662 (0.994) | 662 (0.994) | 662 (0.994) |
| MTK | L1 | 42 (0.063) | 113 (0.170) | 164 (0.246) | 213 (0.320) | 244 (0.366) | 294 (0.441) |
| MTK | H2_PUSH | 281 (0.422) | 357 (0.536) | 405 (0.608) | 461 (0.692) | 506 (0.760) | 528 (0.793) |
| MTK | T1_LOCAL | 307 (0.461) | 313 (0.470) | 313 (0.470) | 316 (0.474) | 316 (0.474) | 316 (0.474) |
| MTK | T1_PUSH | 641 (0.962) | 657 (0.986) | 657 (0.986) | 660 (0.991) | 660 (0.991) | 660 (0.991) |
| MTK | T1_ROWS | 644 (0.967) | 659 (0.989) | 659 (0.989) | 662 (0.994) | 662 (0.994) | 662 (0.994) |

Hop 3 (666 rows):

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 65 (0.098) | 87 (0.131) | 117 (0.176) | 162 (0.243) | 209 (0.314) | 301 (0.452) |
| UNROUTED | H2 | 55 (0.083) | 86 (0.129) | 99 (0.149) | 136 (0.204) | 194 (0.291) | 280 (0.420) |
| UNROUTED | T1 | 611 (0.917) | 636 (0.955) | 637 (0.956) | 641 (0.962) | 642 (0.964) | 644 (0.967) |
| PHG | L1 | 65 (0.098) | 99 (0.149) | 161 (0.242) | 228 (0.342) | 293 (0.440) | 332 (0.498) |
| PHG | H2_PUSH | 55 (0.083) | 84 (0.126) | 107 (0.161) | 151 (0.227) | 200 (0.300) | 276 (0.414) |
| PHG | T1_LOCAL | 117 (0.176) | 145 (0.218) | 201 (0.302) | 259 (0.389) | 309 (0.464) | 337 (0.506) |
| PHG | T1_PUSH | 131 (0.197) | 160 (0.240) | 219 (0.329) | 277 (0.416) | 330 (0.495) | 358 (0.538) |
| PHG | T1_ROWS | 611 (0.917) | 636 (0.955) | 639 (0.959) | 640 (0.961) | 643 (0.965) | 646 (0.970) |
| MTK | L1 | 65 (0.098) | 95 (0.143) | 153 (0.230) | 205 (0.308) | 261 (0.392) | 291 (0.437) |
| MTK | H2_PUSH | 55 (0.083) | 84 (0.126) | 107 (0.161) | 149 (0.224) | 190 (0.285) | 260 (0.390) |
| MTK | T1_LOCAL | 106 (0.159) | 136 (0.204) | 189 (0.284) | 234 (0.351) | 277 (0.416) | 299 (0.449) |
| MTK | T1_PUSH | 133 (0.200) | 164 (0.246) | 218 (0.327) | 264 (0.396) | 309 (0.464) | 334 (0.502) |
| MTK | T1_ROWS | 611 (0.917) | 637 (0.956) | 639 (0.959) | 640 (0.961) | 643 (0.965) | 646 (0.970) |

5–10 answers (287 rows):

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 36 (0.125) | 36 (0.125) | 44 (0.153) | 53 (0.185) | 76 (0.265) | 112 (0.390) |
| UNROUTED | H2 | 51 (0.178) | 68 (0.237) | 73 (0.254) | 90 (0.314) | 122 (0.425) | 176 (0.613) |
| UNROUTED | T1 | 270 (0.941) | 272 (0.948) | 272 (0.948) | 272 (0.948) | 272 (0.948) | 272 (0.948) |
| PHG | L1 | 36 (0.125) | 36 (0.125) | 50 (0.174) | 80 (0.279) | 102 (0.355) | 115 (0.401) |
| PHG | H2_PUSH | 51 (0.178) | 67 (0.233) | 72 (0.251) | 91 (0.317) | 126 (0.439) | 155 (0.540) |
| PHG | T1_LOCAL | 60 (0.209) | 62 (0.216) | 76 (0.265) | 98 (0.341) | 108 (0.376) | 119 (0.415) |
| PHG | T1_PUSH | 136 (0.474) | 138 (0.481) | 153 (0.533) | 175 (0.610) | 186 (0.648) | 197 (0.686) |
| PHG | T1_ROWS | 270 (0.941) | 272 (0.948) | 272 (0.948) | 272 (0.948) | 273 (0.951) | 276 (0.962) |
| MTK | L1 | 36 (0.125) | 36 (0.125) | 48 (0.167) | 75 (0.261) | 96 (0.334) | 110 (0.383) |
| MTK | H2_PUSH | 52 (0.181) | 67 (0.233) | 73 (0.254) | 88 (0.307) | 117 (0.408) | 146 (0.509) |
| MTK | T1_LOCAL | 59 (0.206) | 63 (0.220) | 76 (0.265) | 99 (0.345) | 108 (0.376) | 117 (0.408) |
| MTK | T1_PUSH | 134 (0.467) | 138 (0.481) | 152 (0.530) | 175 (0.610) | 185 (0.645) | 195 (0.679) |
| MTK | T1_ROWS | 270 (0.941) | 272 (0.948) | 272 (0.948) | 272 (0.948) | 273 (0.951) | 276 (0.962) |

11+ answers (339 rows):

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 8 (0.024) | 13 (0.038) | 14 (0.041) | 21 (0.062) | 31 (0.091) | 72 (0.212) |
| UNROUTED | H2 | 8 (0.024) | 16 (0.047) | 21 (0.062) | 25 (0.074) | 42 (0.124) | 93 (0.274) |
| UNROUTED | T1 | 306 (0.903) | 330 (0.973) | 330 (0.973) | 331 (0.976) | 331 (0.976) | 332 (0.979) |
| PHG | L1 | 8 (0.024) | 12 (0.035) | 18 (0.053) | 32 (0.094) | 58 (0.171) | 78 (0.230) |
| PHG | H2_PUSH | 8 (0.024) | 16 (0.047) | 18 (0.053) | 22 (0.065) | 35 (0.103) | 63 (0.186) |
| PHG | T1_LOCAL | 11 (0.032) | 14 (0.041) | 20 (0.059) | 38 (0.112) | 63 (0.186) | 79 (0.233) |
| PHG | T1_PUSH | 99 (0.292) | 112 (0.330) | 118 (0.348) | 136 (0.401) | 163 (0.481) | 179 (0.528) |
| PHG | T1_ROWS | 306 (0.903) | 329 (0.971) | 330 (0.973) | 330 (0.973) | 330 (0.973) | 330 (0.973) |
| MTK | L1 | 8 (0.024) | 12 (0.035) | 15 (0.044) | 25 (0.074) | 51 (0.150) | 62 (0.183) |
| MTK | H2_PUSH | 8 (0.024) | 16 (0.047) | 18 (0.053) | 19 (0.056) | 27 (0.080) | 56 (0.165) |
| MTK | T1_LOCAL | 11 (0.032) | 14 (0.041) | 17 (0.050) | 28 (0.083) | 53 (0.156) | 63 (0.186) |
| MTK | T1_PUSH | 99 (0.292) | 112 (0.330) | 115 (0.339) | 127 (0.375) | 153 (0.451) | 164 (0.484) |
| MTK | T1_ROWS | 306 (0.903) | 329 (0.971) | 329 (0.971) | 329 (0.971) | 329 (0.971) | 329 (0.971) |

**40.8 Paired** (rows gained / lost, McNemar p, descriptive). Against the cell's L1 at B_N 1000:

| cell | arm | all rows | hop 1 | hop 2 | hop 3 |
|---|---|---|---|---|---|
| UNROUTED | T1 | +951 / 0 (p 1.1e-286) | 0 / 0 | +472 / 0 | +479 / 0 |
| UNROUTED | TR | +951 / 0 (p 1.1e-286) | 0 / 0 | +472 / 0 | +479 / 0 |
| UNROUTED | TR_H2 | +953 / −2 (p 3.0e-282) | 0 / 0 | +472 / 0 | +481 / −2 |
| PHG | T1_LOCAL | +144 / 0 (p 9.0e-44) | 0 / 0 | +113 / 0 | +31 / 0 |
| PHG | T1_PUSH | +490 / 0 (p 6.3e-148) | +1 / 0 | +440 / 0 | +49 / 0 |
| PHG | T1_ROWS | +855 / 0 (p 8.3e-258) | +1 / 0 | +442 / 0 | +412 / 0 |
| PHG | TR_ROWS | +855 / 0 (p 8.3e-258) | +1 / 0 | +442 / 0 | +412 / 0 |
| PHG | TR_H2_PUSH | +484 / −64 (p 8.2e-81) | +1 / 0 | +440 / −3 | +43 / −61 |
| MTK | T1_LOCAL | +132 / 0 (p 3.7e-40) | 0 / 0 | +103 / 0 | +29 / 0 |
| MTK | T1_PUSH | +507 / 0 (p 4.8e-153) | +1 / 0 | +447 / 0 | +59 / 0 |
| MTK | T1_ROWS | +885 / 0 (p 7.8e-267) | +1 / 0 | +449 / 0 | +435 / 0 |
| MTK | TR_ROWS | +885 / 0 (p 7.8e-267) | +1 / 0 | +449 / 0 | +435 / 0 |
| MTK | TR_H2_PUSH | +502 / −49 (p 1.1e-95) | +1 / 0 | +447 / −3 | +54 / −46 |

At B_N 100:

| cell | arm | all rows | hop 1 | hop 2 | hop 3 |
|---|---|---|---|---|---|
| UNROUTED | T1 | +1,154 / −1 (p 0) | +5 / 0 | +603 / −1 | +546 / 0 |
| UNROUTED | TR | +1,016 / −1 (p 1.4e-303) | +5 / 0 | +549 / −1 | +462 / 0 |
| UNROUTED | TR_H2 | +994 / −9 (p 6.4e-281) | +5 / −6 | +557 / −2 | +432 / −1 |
| PHG | T1_LOCAL | +338 / 0 (p 3.6e-102) | +4 / 0 | +282 / 0 | +52 / 0 |
| PHG | T1_PUSH | +670 / 0 (p 4.1e-202) | +5 / 0 | +599 / 0 | +66 / 0 |
| PHG | T1_ROWS | +1,154 / −1 (p 0) | +5 / 0 | +603 / −1 | +546 / 0 |
| PHG | TR_ROWS | +1,016 / −1 (p 1.4e-303) | +5 / 0 | +549 / −1 | +462 / 0 |
| PHG | TR_H2_PUSH | +626 / −12 (p 1.5e-167) | +5 / −6 | +559 / −2 | +62 / −4 |
| MTK | T1_LOCAL | +310 / 0 (p 9.6e-94) | +4 / 0 | +265 / 0 | +41 / 0 |
| MTK | T1_PUSH | +672 / 0 (p 1.0e-202) | +5 / 0 | +599 / 0 | +68 / 0 |
| MTK | T1_ROWS | +1,154 / −1 (p 0) | +5 / 0 | +603 / −1 | +546 / 0 |
| MTK | TR_ROWS | +1,016 / −1 (p 1.4e-303) | +5 / 0 | +549 / −1 | +462 / 0 |
| MTK | TR_H2_PUSH | +629 / −13 (p 5.0e-167) | +5 / −6 | +558 / −2 | +66 / −5 |

Against the cell's H2_PUSH (§39) at B_N 1000:

| cell | arm | all rows | hop 1 | hop 2 | hop 3 |
|---|---|---|---|---|---|
| PHG | T1_LOCAL | +159 / −190 (p 0.11) | 0 / 0 | +46 / −185 | +113 / −5 |
| PHG | T1_PUSH | +319 / −4 (p 5.3e-89) | +1 / 0 | +188 / 0 | +130 / −4 |
| PHG | T1_ROWS | +680 / 0 (p 4.0e-205) | +1 / 0 | +190 / 0 | +489 / 0 |
| PHG | TR_H2_PUSH | +245 / 0 (p 3.5e-74) | +1 / 0 | +185 / 0 | +59 / 0 |
| MTK | T1_LOCAL | +137 / −197 (p 1.2e-03) | 0 / 0 | +49 / −194 | +88 / −3 |
| MTK | T1_PUSH | +318 / −3 (p 2.6e-90) | +1 / 0 | +199 / 0 | +118 / −3 |
| MTK | T1_ROWS | +693 / 0 (p 4.9e-209) | +1 / 0 | +201 / 0 | +491 / 0 |
| MTK | TR_H2_PUSH | +261 / 0 (p 5.4e-79) | +1 / 0 | +196 / 0 | +64 / 0 |

At B_N 5000:

| cell | arm | all rows | hop 1 | hop 2 | hop 3 |
|---|---|---|---|---|---|
| PHG | T1_LOCAL | +70 / −221 (p 2.1e-19) | 0 / 0 | 0 / −212 | +70 / −9 |
| PHG | T1_PUSH | +207 / −9 (p 4.7e-50) | +1 / 0 | +117 / −2 | +89 / −7 |
| PHG | T1_ROWS | +491 / −3 (p 7.9e-142) | +1 / 0 | +119 / −2 | +371 / −1 |
| PHG | TR_H2_PUSH | +155 / 0 (p 4.4e-47) | +1 / 0 | +117 / 0 | +37 / 0 |
| MTK | T1_LOCAL | +54 / −227 (p 2.1e-26) | 0 / 0 | 0 / −212 | +54 / −15 |
| MTK | T1_PUSH | +218 / −11 (p 4.4e-51) | +1 / 0 | +134 / −2 | +83 / −9 |
| MTK | T1_ROWS | +524 / −3 (p 1.1e-151) | +1 / 0 | +136 / −2 | +387 / −1 |
| MTK | TR_H2_PUSH | +179 / 0 (p 2.6e-54) | +1 / 0 | +134 / 0 | +44 / 0 |

**40.9 Where the golds rank, what is fetched and what it costs.** The gold nodes in the typed candidate list CT (14,489 golds: hop 1 1,334, hop 2 4,613, hop 3 8,542):

| list | gold nodes in list | rank < 10 | rank < 100 | hop 1 | hop 2 | hop 3 |
|---|---|---|---|---|---|---|
| UNROUTED CT | 13,705 (0.946) | 5,855 | 13,365 | 907 (0.680) | 4,556 (0.988) | 8,242 (0.965) |
| PHG CT (PUSH) | 7,769 (0.536) | 4,324 | 7,497 | 907 (0.680) | 4,553 (0.987) | 2,309 (0.270) |
| PHG CT (LOCAL) | 4,438 (0.306) | 3,559 | 4,438 | 905 (0.678) | 1,785 (0.387) | 1,748 (0.205) |
| MTK CT (PUSH) | 7,947 (0.548) | 4,403 | 7,646 | 907 (0.680) | 4,553 (0.987) | 2,487 (0.291) |
| MTK CT (LOCAL) | 4,527 (0.312) | 3,541 | 4,527 | 906 (0.679) | 1,818 (0.394) | 1,803 (0.211) |

The fetches per query (served nodes outside the contacted shards):

| cell | arm | B_N | fetched nodes per query: mean (share of served) | extra shards per query: mean / p95 / max |
|---|---|---|---|---|
| PHG | H2_PUSH | 100 | 17.2 (0.172) | 15.6 / 23.0 / 27 |
| PHG | H2_PUSH | 500 | 174.5 (0.349) | 126.8 / 156.0 / 179 |
| PHG | H2_PUSH | 1000 | 450.2 (0.450) | 234.6 / 267.0 / 292 |
| PHG | H2_PUSH | 5000 | 3,117.3 (0.624) | 341.2 / 364.0 / 391 |
| PHG | T1_PUSH | 100 | 18.1 (0.181) | 15.8 / 66.1 / 78 |
| PHG | T1_PUSH | 500 | 63.9 (0.128) | 38.7 / 217.0 / 254 |
| PHG | T1_PUSH | 1000 | 94.6 (0.095) | 44.7 / 279.1 / 325 |
| PHG | T1_PUSH | 5000 | 215.5 (0.043) | 47.3 / 309.1 / 370 |
| PHG | T1_ROWS | 100 | 21.2 (0.212) | 18.0 / 65.0 / 78 |
| PHG | T1_ROWS | 500 | 69.6 (0.139) | 42.5 / 216.0 / 246 |
| PHG | T1_ROWS | 1000 | 101.0 (0.101) | 48.8 / 279.1 / 325 |
| PHG | T1_ROWS | 5000 | 222.0 (0.044) | 51.4 / 309.1 / 370 |
| PHG | TR_H2_PUSH | 100 | 21.0 (0.210) | 19.1 / 33.0 / 48 |
| PHG | TR_H2_PUSH | 500 | 183.9 (0.368) | 132.0 / 165.0 / 194 |
| PHG | TR_H2_PUSH | 1000 | 463.7 (0.464) | 237.9 / 271.0 / 298 |
| PHG | TR_H2_PUSH | 5000 | 3,130.2 (0.626) | 341.2 / 364.0 / 391 |
| MTK | H2_PUSH | 100 | 17.1 (0.171) | 15.5 / 23.0 / 29 |
| MTK | H2_PUSH | 500 | 165.5 (0.331) | 122.4 / 152.0 / 175 |
| MTK | H2_PUSH | 1000 | 433.4 (0.433) | 231.0 / 264.0 / 293 |
| MTK | H2_PUSH | 5000 | 3,077.8 (0.616) | 344.4 / 369.0 / 400 |
| MTK | T1_PUSH | 100 | 17.9 (0.179) | 15.4 / 66.0 / 78 |
| MTK | T1_PUSH | 500 | 63.2 (0.126) | 38.4 / 220.1 / 248 |
| MTK | T1_PUSH | 1000 | 93.5 (0.094) | 44.6 / 277.0 / 328 |
| MTK | T1_PUSH | 5000 | 213.7 (0.043) | 47.6 / 321.0 / 381 |
| MTK | T1_ROWS | 100 | 21.5 (0.215) | 18.1 / 65.0 / 78 |
| MTK | T1_ROWS | 500 | 69.9 (0.140) | 43.0 / 220.0 / 248 |
| MTK | T1_ROWS | 1000 | 101.2 (0.101) | 49.5 / 277.0 / 328 |
| MTK | T1_ROWS | 5000 | 222.7 (0.045) | 52.6 / 321.1 / 381 |
| MTK | TR_H2_PUSH | 100 | 20.7 (0.207) | 18.8 / 32.0 / 49 |
| MTK | TR_H2_PUSH | 500 | 175.1 (0.350) | 127.6 / 162.0 / 190 |
| MTK | TR_H2_PUSH | 1000 | 447.0 (0.447) | 234.7 / 270.0 / 294 |
| MTK | TR_H2_PUSH | 5000 | 3,090.3 (0.618) | 344.4 / 369.0 / 400 |

The per-query sizes. The typed quantities are over the 1,994 gated rows; nC2 and nC2P are over all rows.
- **seeds_contacted:** the seeds inside the contacted shards.
- **nCT** (UNROUTED), **nCTL / nCTP:** the typed list's length, unrouted / under LOCAL / under PUSH.
- **nCTP_out / nCTR_out:** the PUSH / ROWS candidates outside the contacted shards.
- **rows_typed / rows_untyped** (UNROUTED) and **rows_LOCAL / rows_PUSH:** the adjacency rows the walks read.
- **rows_out_typed:** the typed rows the ROWS walk reads outside the contacted shards. **rows_out** counts every row it reads there, typed and untyped; **shards_out** counts their distinct shards.
- **nC2 / nC2P:** §39's second-hop list, unrouted / PUSH.
- **contacted_mass** and **B_P:** L1's contacted nodes and shards.

| cell | quantity | mean | median | p90 | max |
|---|---|---|---|---|---|
| UNROUTED | nCT | 310.4 | 12.0 | 531.4 | 9,836 |
| UNROUTED | rows_typed | 266.5 | 11.0 | 345.1 | 8,425 |
| UNROUTED | rows_untyped | 1,886.9 | 712.5 | 4,885.0 | 9,002 |
| UNROUTED | nC2 | 17,074.6 | 17,094.0 | 18,205.5 | 23,395 |
| PHG | seeds_contacted | 1.1 | 1.0 | 1.7 | 3 |
| PHG | nCTL | 65.4 | 4.0 | 120.0 | 1,981 |
| PHG | nCTP | 299.4 | 6.0 | 504.0 | 9,836 |
| PHG | nCTP_out | 234.0 | 2.0 | 398.4 | 8,047 |
| PHG | nCTR_out | 240.5 | 5.0 | 413.0 | 8,047 |
| PHG | rows_LOCAL | 367.0 | 152.5 | 984.7 | 2,139 |
| PHG | rows_PUSH | 367.0 | 152.5 | 984.7 | 2,139 |
| PHG | rows_out_typed | 210.3 | 4.0 | 277.1 | 7,095 |
| PHG | rows_out | 1,519.4 | 539.5 | 3,971.8 | 7,579 |
| PHG | shards_out | 193.5 | 255.5 | 348.0 | 392 |
| PHG | nC2P | 14,314.8 | 14,649.0 | 15,246.3 | 15,699 |
| PHG | contacted_mass | 8,652.8 | 8,561.0 | 10,429.9 | 14,823 |
| PHG | B_P | 85.3 | 84.0 | 102.0 | 147 |
| MTK | seeds_contacted | 1.1 | 1.0 | 1.7 | 3 |
| MTK | nCTL | 64.5 | 4.0 | 111.7 | 2,046 |
| MTK | nCTP | 296.7 | 6.0 | 503.0 | 9,836 |
| MTK | nCTP_out | 232.2 | 2.0 | 401.1 | 8,321 |
| MTK | nCTR_out | 241.3 | 5.0 | 412.4 | 8,321 |
| MTK | rows_LOCAL | 381.8 | 160.0 | 1,029.0 | 2,089 |
| MTK | rows_PUSH | 381.8 | 160.0 | 1,029.0 | 2,089 |
| MTK | rows_out_typed | 210.0 | 4.0 | 268.7 | 7,033 |
| MTK | rows_out | 1,504.6 | 534.5 | 3,936.8 | 7,449 |
| MTK | shards_out | 194.4 | 255.5 | 354.0 | 400 |
| MTK | nC2P | 14,370.9 | 14,729.0 | 15,230.3 | 15,841 |
| MTK | contacted_mass | 8,633.5 | 8,701.5 | 10,520.9 | 14,452 |
| MTK | B_P | 86.0 | 87.0 | 105.0 | 144 |

The latency per row (laptop, one process):

| stage | mean ms | median ms | p95 ms |
|---|---|---|---|
| base (IR_L1 score, served order, router order, V_q, LS) | 6.2 | 5.8 | 8.3 |
| UNROUTED H2 | 11.1 | 10.6 | 14.1 |
| seeds + schedule | 0.4 | 0.3 | 0.6 |
| UNROUTED walks (typed + untyped) | 18.7 | 18.4 | 25.1 |
| UNROUTED orders | 12.5 | 12.0 | 15.9 |
| identity check (pinned relwalk x2) | 26.9 | 27.2 | 35.9 |
| MTK_k432 ES contacted set + L1 order + H2_PUSH | 7.3 | 7.1 | 9.6 |
| MTK_k432 LOCAL + PUSH walks (typed + untyped each) | 40.2 | 39.8 | 51.7 |
| MTK_k432 orders | 8.0 | 7.6 | 10.9 |
| PHG_k432 ES contacted set + L1 order + H2_PUSH | 7.3 | 7.0 | 9.6 |
| PHG_k432 LOCAL + PUSH walks (typed + untyped each) | 40.7 | 40.0 | 53.2 |
| PHG_k432 orders | 7.9 | 7.5 | 10.7 |

**40.10 What the best arm misses.** A row that is not ALL-served falls in the first matching class:
- **NO_GATE:** outside the gate.
- **NOT_REACHED, schedule shorter than hop:** a missed gold is not a typed candidate at all, and the schedule has fewer relations than the question has hops.
- **CT_TOO_LONG:** every missed gold is a typed candidate, but beyond the exposure.
- No row was TOPIC_MISS (the topic not seeded) or NOT_REACHED with a full-length schedule.

| cell, arm | B_N | missed rows | NO_GATE | NOT_REACHED, schedule shorter than hop | CT_TOO_LONG | NOT_REACHED* golds reached / golds |
|---|---|---|---|---|---|---|
| UNROUTED T1 | 100 | 82 | 3 (hop2 2, hop3 1) | 70 (hop1 5, hop2 12, hop3 53) | 9 (hop2 8, hop3 1) | 755 / 1,056 |
| UNROUTED T1 | 1000 | 32 | 3 (hop2 2, hop3 1) | 29 (hop2 5, hop3 24) | 0 | 38 / 218 |
| UNROUTED T1 | 5000 | 26 | 3 (hop2 2, hop3 1) | 23 (hop2 2, hop3 21) | 0 | 0 / 169 |
| PHG_k432 T1_ROWS | 100 | 82 | 3 (hop2 2, hop3 1) | 70 (hop1 5, hop2 12, hop3 53) | 9 (hop2 8, hop3 1) | 755 / 1,056 |
| PHG_k432 T1_ROWS | 1000 | 31 | 3 (hop2 2, hop3 1) | 28 (hop1 1, hop2 2, hop3 25) | 0 | 42 / 230 |
| PHG_k432 T1_ROWS | 5000 | 25 | 3 (hop2 2, hop3 1) | 22 (hop1 1, hop2 2, hop3 19) | 0 | 42 / 203 |
| MTK_k432 T1_ROWS | 100 | 82 | 3 (hop2 2, hop3 1) | 70 (hop1 5, hop2 12, hop3 53) | 9 (hop2 8, hop3 1) | 755 / 1,056 |
| MTK_k432 T1_ROWS | 1000 | 31 | 3 (hop2 2, hop3 1) | 28 (hop1 1, hop2 2, hop3 25) | 0 | 110 / 298 |
| MTK_k432 T1_ROWS | 5000 | 25 | 3 (hop2 2, hop3 1) | 22 (hop1 1, hop2 2, hop3 19) | 0 | 110 / 271 |

The NOT_REACHED rows at B_N 1000 (PHG T1_ROWS), by query type and lexical schedule:

| hop | question type | lexical schedule | rows |
|---|---|---|---|
| hop3 | movie_to_director_to_movie_to_genre | directed_by | 11 |
| hop3 | movie_to_actor_to_movie_to_genre | starred_actors | 7 |
| hop3 | movie_to_writer_to_movie_to_genre | written_by | 4 |
| hop3 | movie_to_writer_to_movie_to_director | written_by,directed_by | 2 |
| hop1 | tag_to_movie |  | 1 |
| hop2 | actor_to_movie_to_actor |  | 1 |
| hop2 | actor_to_movie_to_genre | starred_actors | 1 |
| hop3 | movie_to_actor_to_movie_to_writer | starred_actors,written_by | 1 |

- **Why genre is missing.** 23 of the 28 NOT_REACHED rows end in a genre, and their schedule lacks has_genre.
  - All 23 phrase the genre as "types" ("what types are the films …"), and no token of has_genre matches that word.
  - Among the served genre-final rows the word is "genres" on 186, "genre" on 53, "types" on 22, "type" on 7, "kind" on 6 (plus 1 with both "genre" and "kind") and "sort" on 4.
  - The 22 served "types" rows are served by the L1 order, not the typed channel: none of their 34 golds is a typed candidate. Their answer sets are small (1–3 genres).
- **The other 5 NOT_REACHED rows.**
  - Three hop-3 rows carry the same schedule as served rows of their type: two movie → writer → movie → director and one movie → actor → movie → writer. Their golds are not typed candidates for a reason not diagnosed here.
  - Two rows have an empty schedule: a hop-1 tag question and a hop-2 co-actor question.
- **The 3 rows outside the gate** each name a film whose title is a number ("1911", "2012", "300").

**40.11 E1: the read budget** (`_l3d_budget.py`). T1_PUSH and T1_ROWS differ only in what the walk may read outside the contacted shards: PUSH reads nothing there, ROWS reads every row. E1 puts a budget between them.
- **The walk** is TYPED v1's PUSH walk with a readable set R_k. R_0 is the contacted nodes. Before each step k the readable set grows greedily from the walk's current frontier:
  - **S<s>** (shard budget): the frontier mass outside R_{k−1} is summed per shard. Shards open in order of that mass (ties → shard id) until s extra shards are open in total. Every node of an open shard stays readable (a shard fetch).
  - **R<r>** (row budget): frontier nodes outside R_{k−1} open one by one, in order of mass (ties → FLAT rank), until r extra node rows are open in total (a row fetch).
  - **SINF:** every frontier shard opens.
- **The untyped EPS channel** walks as in T1_PUSH: the budget is spent on the typed channel only.
- **Serving.** CT_B is formed as in TYPED v1. The served list is T1 (CT_B, then the L1 order), at n(q, B_N).
- **Grids:** S 1, 2, 4, …, 128 and R 1, 2, 4, …, 1024. No value is selected here.
- **Identities.** S0 == T1_PUSH's walk and SINF == the unmasked walk, each on all 3,988 gated (row, cell) pairs. All 98 TYPED v1 arrays are recomputed and equal the TYPED v1 npz (0ee65e76…).

ALL-gold rows, PHG:

| arm | B_N 100 | B_N 1000 | B_N 5000 | hop 2 @ 1000 (of 666) | hop 3 @ 1000 (of 666) |
|---|---|---|---|---|---|
| T1_PUSH (S0 == R0) | 1,433 (0.717) | 1,602 (0.802) | 1,683 (0.842) | 660 (0.991) | 277 (0.416) |
| T1_SINF | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_S1 | 1,505 (0.753) | 1,655 (0.828) | 1,733 (0.867) | 661 (0.992) | 329 (0.494) |
| T1_S2 | 1,552 (0.777) | 1,685 (0.843) | 1,757 (0.879) | 662 (0.994) | 358 (0.538) |
| T1_S4 | 1,618 (0.810) | 1,737 (0.869) | 1,795 (0.898) | 662 (0.994) | 410 (0.616) |
| T1_S8 | 1,706 (0.854) | 1,805 (0.903) | 1,847 (0.924) | 662 (0.994) | 478 (0.718) |
| T1_S16 | 1,798 (0.900) | 1,872 (0.937) | 1,902 (0.952) | 662 (0.994) | 545 (0.818) |
| T1_S32 | 1,876 (0.939) | 1,930 (0.966) | 1,946 (0.974) | 662 (0.994) | 603 (0.905) |
| T1_S64 | 1,913 (0.957) | 1,963 (0.982) | 1,970 (0.986) | 662 (0.994) | 636 (0.955) |
| T1_S128 | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_R1 | 1,497 (0.749) | 1,651 (0.826) | 1,725 (0.863) | 661 (0.992) | 325 (0.488) |
| T1_R2 | 1,540 (0.771) | 1,674 (0.838) | 1,747 (0.874) | 662 (0.994) | 347 (0.521) |
| T1_R4 | 1,608 (0.805) | 1,727 (0.864) | 1,791 (0.896) | 662 (0.994) | 400 (0.601) |
| T1_R8 | 1,692 (0.847) | 1,785 (0.893) | 1,838 (0.920) | 662 (0.994) | 458 (0.688) |
| T1_R16 | 1,781 (0.891) | 1,859 (0.930) | 1,892 (0.947) | 662 (0.994) | 532 (0.799) |
| T1_R32 | 1,858 (0.930) | 1,915 (0.958) | 1,934 (0.968) | 662 (0.994) | 588 (0.883) |
| T1_R64 | 1,911 (0.956) | 1,960 (0.981) | 1,969 (0.985) | 662 (0.994) | 633 (0.950) |
| T1_R128 | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_R256 | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_R512 | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_R1024 | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_ROWS (TYPED v1; also reads untyped rows) | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |

The cost on PHG, per gated row. "Opened" means extra shards for S arms and node rows for R arms:

| arm | opened per gated row: mean / median / p90 / max | readable node rows opened: mean | typed rows read outside: mean / median / max | served fetch @ 1000: nodes / extra shards / walk ∪ fetch shards (means) | vs T1_PUSH @ 1000 | vs T1_ROWS @ 1000 |
|---|---|---|---|---|---|---|
| T1_PUSH (S0 == R0) | 0 | 0 | 0 | 94.6 / 44.7 / — | — | — |
| T1_SINF | 39.7 / 4.0 / 182.4 / 367 | 3,996.8 | 210.3 / 4.0 / 7,095 | 101.0 / 48.8 / 68.9 | +365 / 0 (p 2.7e-110) | 0 / 0 (p 1.00) |
| T1_S1 | 0.8 / 1.0 / 1.0 / 1 | 80.3 | 2.2 / 1.0 / 35 | 96.1 / 45.8 / 46.1 | +53 / 0 (p 2.2e-16) | 0 / −312 (p 2.4e-94) |
| T1_S2 | 1.4 / 2.0 / 2.0 / 2 | 146.0 | 4.1 / 2.0 / 70 | 96.7 / 46.3 / 46.9 | +83 / 0 (p 2.1e-25) | 0 / −282 (p 2.6e-85) |
| T1_S4 | 2.5 / 4.0 / 4.0 / 4 | 254.4 | 7.4 / 4.0 / 129 | 97.5 / 46.8 / 47.9 | +135 / 0 (p 4.6e-41) | 0 / −230 (p 1.2e-69) |
| T1_S8 | 4.2 / 4.0 / 8.0 / 8 | 428.7 | 13.3 / 4.0 / 250 | 98.6 / 47.4 / 49.4 | +203 / 0 (p 1.6e-61) | 0 / −162 (p 3.4e-49) |
| T1_S16 | 6.9 / 4.0 / 16.0 / 16 | 693.5 | 23.1 / 4.0 / 491 | 99.7 / 48.1 / 51.3 | +270 / 0 (p 1.1e-81) | 0 / −95 (p 5.0e-29) |
| T1_S32 | 10.6 / 4.0 / 32.0 / 32 | 1,065.3 | 39.4 / 4.0 / 901 | 100.6 / 48.6 / 53.5 | +328 / 0 (p 3.7e-99) | 0 / −37 (p 1.5e-11) |
| T1_S64 | 15.6 / 4.0 / 64.0 / 64 | 1,575.6 | 67.1 / 4.0 / 1,694 | 101.0 / 48.8 / 55.8 | +361 / 0 (p 4.3e-109) | 0 / −4 (p 0.12) |
| T1_S128 | 23.2 / 4.0 / 128.0 / 128 | 2,341.2 | 113.8 / 4.0 / 3,122 | 101.0 / 48.8 / 59.2 | +365 / 0 (p 2.7e-110) | 0 / 0 (p 1.00) |
| T1_R1 | 0.8 / 1.0 / 1.0 / 1 | 0.8 | 0.8 / 1.0 / 1 | 95.5 / 45.5 / 45.8 | +49 / 0 (p 3.6e-15) | 0 / −316 (p 1.5e-95) |
| T1_R2 | 1.4 / 2.0 / 2.0 / 2 | 1.4 | 1.4 / 2.0 / 2 | 96.3 / 46.0 / 46.7 | +72 / 0 (p 4.2e-22) | 0 / −293 (p 1.3e-88) |
| T1_R4 | 2.5 / 4.0 / 4.0 / 4 | 2.5 | 2.5 / 4.0 / 4 | 97.1 / 46.5 / 47.7 | +125 / 0 (p 4.7e-38) | 0 / −240 (p 1.1e-72) |
| T1_R8 | 4.3 / 4.0 / 8.0 / 8 | 4.3 | 4.3 / 4.0 / 8 | 98.2 / 47.2 / 49.1 | +183 / 0 (p 1.6e-55) | 0 / −182 (p 3.3e-55) |
| T1_R16 | 7.0 / 4.0 / 16.0 / 16 | 7.0 | 7.0 / 4.0 / 16 | 99.4 / 47.9 / 51.0 | +257 / 0 (p 8.6e-78) | 0 / −108 (p 6.2e-33) |
| T1_R32 | 11.0 / 4.0 / 32.0 / 32 | 11.0 | 11.0 / 4.0 / 32 | 100.4 / 48.5 / 53.1 | +313 / 0 (p 1.2e-94) | 0 / −52 (p 4.4e-16) |
| T1_R64 | 16.4 / 4.0 / 64.0 / 64 | 16.4 | 16.4 / 4.0 / 64 | 100.9 / 48.8 / 55.3 | +358 / 0 (p 3.4e-108) | 0 / −7 (p 0.02) |
| T1_R128 | 24.4 / 4.0 / 128.0 / 128 | 24.4 | 24.4 / 4.0 / 128 | 101.0 / 48.8 / 57.6 | +365 / 0 (p 2.7e-110) | 0 / 0 (p 1.00) |
| T1_R256 | 38.4 / 4.0 / 256.0 / 256 | 38.4 | 38.4 / 4.0 / 256 | 101.0 / 48.8 / 61.0 | +365 / 0 (p 2.7e-110) | 0 / 0 (p 1.00) |
| T1_R512 | 60.4 / 4.0 / 277.1 / 512 | 60.4 | 60.4 / 4.0 / 512 | 101.0 / 48.8 / 64.8 | +365 / 0 (p 2.7e-110) | 0 / 0 (p 1.00) |
| T1_R1024 | 90.3 / 4.0 / 277.1 / 1,024 | 90.3 | 90.3 / 4.0 / 1,024 | 101.0 / 48.8 / 67.7 | +365 / 0 (p 2.7e-110) | 0 / 0 (p 1.00) |
| T1_ROWS (TYPED v1; also reads untyped rows) | 0 | 0 | 0 | 101.0 / 48.8 / — | — | — |

ALL-gold rows, MTK:

| arm | B_N 100 | B_N 1000 | B_N 5000 | hop 2 @ 1000 (of 666) | hop 3 @ 1000 (of 666) |
|---|---|---|---|---|---|
| T1_PUSH (S0 == R0) | 1,435 (0.718) | 1,589 (0.795) | 1,659 (0.830) | 660 (0.991) | 264 (0.396) |
| T1_SINF | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_S1 | 1,500 (0.751) | 1,635 (0.818) | 1,698 (0.850) | 661 (0.992) | 309 (0.464) |
| T1_S2 | 1,558 (0.780) | 1,680 (0.841) | 1,737 (0.869) | 662 (0.994) | 353 (0.530) |
| T1_S4 | 1,639 (0.820) | 1,743 (0.872) | 1,789 (0.895) | 662 (0.994) | 416 (0.625) |
| T1_S8 | 1,707 (0.854) | 1,798 (0.900) | 1,835 (0.918) | 662 (0.994) | 471 (0.707) |
| T1_S16 | 1,797 (0.899) | 1,864 (0.933) | 1,897 (0.949) | 662 (0.994) | 537 (0.806) |
| T1_S32 | 1,874 (0.938) | 1,931 (0.966) | 1,944 (0.973) | 662 (0.994) | 604 (0.907) |
| T1_S64 | 1,913 (0.957) | 1,964 (0.983) | 1,970 (0.986) | 662 (0.994) | 637 (0.956) |
| T1_S128 | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_R1 | 1,494 (0.748) | 1,628 (0.815) | 1,695 (0.848) | 661 (0.992) | 302 (0.453) |
| T1_R2 | 1,544 (0.773) | 1,670 (0.836) | 1,731 (0.866) | 662 (0.994) | 343 (0.515) |
| T1_R4 | 1,626 (0.814) | 1,728 (0.865) | 1,777 (0.889) | 662 (0.994) | 401 (0.602) |
| T1_R8 | 1,698 (0.850) | 1,785 (0.893) | 1,830 (0.916) | 662 (0.994) | 458 (0.688) |
| T1_R16 | 1,779 (0.890) | 1,852 (0.927) | 1,885 (0.943) | 662 (0.994) | 525 (0.788) |
| T1_R32 | 1,861 (0.931) | 1,921 (0.961) | 1,936 (0.969) | 662 (0.994) | 594 (0.892) |
| T1_R64 | 1,911 (0.956) | 1,960 (0.981) | 1,969 (0.985) | 662 (0.994) | 633 (0.950) |
| T1_R128 | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_R256 | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_R512 | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_R1024 | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |
| T1_ROWS (TYPED v1; also reads untyped rows) | 1,916 (0.959) | 1,967 (0.984) | 1,973 (0.987) | 662 (0.994) | 640 (0.961) |

The cost on MTK:

| arm | opened per gated row: mean / median / p90 / max | readable node rows opened: mean | typed rows read outside: mean / median / max | served fetch @ 1000: nodes / extra shards / walk ∪ fetch shards (means) | vs T1_PUSH @ 1000 | vs T1_ROWS @ 1000 |
|---|---|---|---|---|---|---|
| T1_PUSH (S0 == R0) | 0 | 0 | 0 | 93.5 / 44.6 / — | — | — |
| T1_SINF | 40.0 / 4.0 / 182.2 / 373 | 4,000.4 | 210.0 / 4.0 / 7,033 | 101.2 / 49.5 / 69.9 | +378 / 0 (p 3.2e-114) | 0 / 0 (p 1.00) |
| T1_S1 | 0.8 / 1.0 / 1.0 / 1 | 79.3 | 2.4 / 1.0 / 41 | 95.5 / 45.8 / 46.2 | +46 / 0 (p 2.8e-14) | 0 / −332 (p 2.3e-100) |
| T1_S2 | 1.5 / 2.0 / 2.0 / 2 | 145.6 | 4.3 / 2.0 / 65 | 96.3 / 46.4 / 47.0 | +91 / 0 (p 8.1e-28) | 0 / −287 (p 8.0e-87) |
| T1_S4 | 2.5 / 4.0 / 4.0 / 4 | 254.3 | 7.6 / 4.0 / 122 | 97.2 / 47.1 / 48.2 | +154 / 0 (p 8.8e-47) | 0 / −224 (p 7.4e-68) |
| T1_S8 | 4.2 / 4.0 / 8.0 / 8 | 424.9 | 13.3 / 4.0 / 235 | 98.4 / 47.8 / 49.8 | +209 / 0 (p 2.4e-63) | 0 / −169 (p 2.7e-51) |
| T1_S16 | 6.9 / 4.0 / 16.0 / 16 | 687.2 | 23.2 / 4.0 / 457 | 99.6 / 48.6 / 51.8 | +275 / 0 (p 3.3e-83) | 0 / −103 (p 2.0e-31) |
| T1_S32 | 10.5 / 4.0 / 32.0 / 32 | 1,053.1 | 39.3 / 4.0 / 863 | 100.7 / 49.2 / 54.1 | +342 / 0 (p 2.2e-103) | 0 / −36 (p 2.9e-11) |
| T1_S64 | 15.5 / 4.0 / 64.0 / 64 | 1,554.1 | 66.9 / 4.0 / 1,615 | 101.1 / 49.5 / 56.4 | +375 / 0 (p 2.6e-113) | 0 / −3 (p 0.25) |
| T1_S128 | 23.1 / 4.0 / 128.0 / 128 | 2,313.3 | 113.1 / 4.0 / 3,035 | 101.2 / 49.5 / 59.8 | +378 / 0 (p 3.2e-114) | 0 / 0 (p 1.00) |
| T1_R1 | 0.8 / 1.0 / 1.0 / 1 | 0.8 | 0.8 / 1.0 / 1 | 95.2 / 45.7 / 46.1 | +39 / 0 (p 3.6e-12) | 0 / −339 (p 1.8e-102) |
| T1_R2 | 1.5 / 2.0 / 2.0 / 2 | 1.5 | 1.5 / 2.0 / 2 | 96.0 / 46.3 / 46.9 | +81 / 0 (p 8.3e-25) | 0 / −297 (p 7.9e-90) |
| T1_R4 | 2.6 / 4.0 / 4.0 / 4 | 2.6 | 2.6 / 4.0 / 4 | 97.0 / 46.9 / 48.0 | +139 / 0 (p 2.9e-42) | 0 / −239 (p 2.3e-72) |
| T1_R8 | 4.3 / 4.0 / 8.0 / 8 | 4.3 | 4.3 / 4.0 / 8 | 98.0 / 47.6 / 49.5 | +196 / 0 (p 2.0e-59) | 0 / −182 (p 3.3e-55) |
| T1_R16 | 7.0 / 4.0 / 16.0 / 16 | 7.0 | 7.0 / 4.0 / 16 | 99.3 / 48.4 / 51.4 | +263 / 0 (p 1.3e-79) | 0 / −115 (p 4.8e-35) |
| T1_R32 | 10.9 / 4.0 / 32.0 / 32 | 10.9 | 10.9 / 4.0 / 32 | 100.5 / 49.1 / 53.6 | +332 / 0 (p 2.3e-100) | 0 / −46 (p 2.8e-14) |
| T1_R64 | 16.3 / 4.0 / 64.0 / 64 | 16.3 | 16.3 / 4.0 / 64 | 101.1 / 49.4 / 55.8 | +371 / 0 (p 4.2e-112) | 0 / −7 (p 0.02) |
| T1_R128 | 24.3 / 4.0 / 128.0 / 128 | 24.3 | 24.3 / 4.0 / 128 | 101.2 / 49.5 / 58.1 | +378 / 0 (p 3.2e-114) | 0 / 0 (p 1.00) |
| T1_R256 | 38.3 / 4.0 / 256.0 / 256 | 38.3 | 38.3 / 4.0 / 256 | 101.2 / 49.5 / 61.5 | +378 / 0 (p 3.2e-114) | 0 / 0 (p 1.00) |
| T1_R512 | 60.2 / 4.0 / 268.7 / 512 | 60.2 | 60.2 / 4.0 / 512 | 101.2 / 49.5 / 65.3 | +378 / 0 (p 3.2e-114) | 0 / 0 (p 1.00) |
| T1_R1024 | 90.3 / 4.0 / 268.7 / 1,024 | 90.3 | 90.3 / 4.0 / 1,024 | 101.2 / 49.5 / 68.5 | +378 / 0 (p 3.2e-114) | 0 / 0 (p 1.00) |
| T1_ROWS (TYPED v1; also reads untyped rows) | 0 | 0 | 0 | 101.2 / 49.5 / — | — | — |

**40.12 Reading** (development numbers, descriptive).
- **The typed walk is the L3 completion MetaQA needs.**
  - Unrouted, T1 serves ALL gold on 1,966 of the 1,998 rows at B_N 1000 (0.984), and on 1,916 at B_N 100 (0.959). IR_L1 serves 1,015 and 763.
  - Hop 3: 641 of 666 at B_N 1000. 11+ answers: 331 of 339.
- **The typed list is short and precise.** Unrouted, CT has a median 12 nodes (mean 310.4). It holds 13,705 of the 14,489 golds (0.946), 13,365 of them in its first 100.
- **Routing binds through the walk's reads, not through the served list.** At B_N 1000 on PHG:
  - T1_LOCAL serves 1,256 rows (0.629), T1_PUSH 1,602 (0.802) and T1_ROWS 1,967 (0.984).
  - Hop 2 is nearly complete already under PUSH (660 of 666). The PUSH loss is hop 3, 277 rows against ROWS's 640: the walk's later steps must expand nodes outside the contacted shards.
  - T1_ROWS against L1: +855 / 0 (p 8.3e-258). Against H2_PUSH: +680 / 0 (p 4.0e-205).
- **Fetches are small.** At B_N 1000, T1_ROWS fetches a mean 101.0 served nodes per query (0.101 of the served list) from 48.8 extra shards (p95 279.1). H2_PUSH fetched 450.2 nodes from 234.6 shards. The typed arm serves more golds with under a quarter of the fetches.
- **RRF and the untyped mix do not help.**
  - TR equals T1 from B_N 1000 on, but is lower at small B_N (1,778 against 1,916 at B_N 100, unrouted).
  - Mixing in §39's untyped C2 (TR_H2_PUSH) reaches only 1,532 on PHG at B_N 1000, and loses 64 rows against L1.
- **The read budget (E1).**
  - **A row budget of 128 equals unbounded reads.** R128 serves exactly T1_ROWS's ALL counts at every B_N on both maps (paired 0 / 0). It reads a mean 24.4 typed rows outside the contacted shards per gated row (median 4, max 128). T1_ROWS reads a mean 210.3 (max 7,095).
  - R64 is within 7 rows of T1_ROWS (p 0.02), at a mean 16.4 rows.
  - **Rows are far cheaper than shards.** S64 reaches 1,963 (−4 against ROWS, p 0.12) but opens 15.6 whole shards per gated row: 1,575.6 node rows of extra local scan. R64 reaches 1,960 with 16.4 rows. At each budget step S<b> scans about two orders of magnitude more nodes than R<b> for at most a few more rows.
  - **The untyped rows T1_ROWS reads outside buy nothing at these exposures.** SINF opens only the typed walk's shards and equals T1_ROWS (0 / 0 on both maps).
  - **The fan-out beyond L1.** The served fetch at B_N 1000 is 101.0 nodes from 48.8 extra shards for every R arm from 64 on. With R128, the walk's rows and the served fetches together touch a mean 57.6 extra shards beyond L1's B_P (mean 85.3 shards on PHG).
  - So the typed completion runs behind the routing with a small, bounded read budget: a mean of about 25 row fetches per query, plus the served fetches.
- **The residual is the schedule.** T1_ROWS misses 31 rows at B_N 1000 (PHG). 28 are NOT_REACHED with a short schedule and 3 are outside the gate; none is crowded out by the exposure. 23 of the 28 are genre questions worded "types", which the lexical schedule cannot match to has_genre.

**40.13 The host L3 lane** (stage L3_HOST; records `results/L3_HOST/`).
- **Why.** The user asked for full use of the host. The laptop runs one heavy process at a time, and the L3 harnesses read only the L1_DEV population's rows. So the lane moves code and a population bundle, not the caches.
- **Declaration** (`HOST_STAGE_DECLARATION__L3_HOST__v1.json`, 37ff06f7…).
  - It names the stage, the push set, the priority policy, the placement test and the never list.
  - The two top-1000 retrieval caches (2 × 2,445,078,580 bytes) stay on the laptop. `bundle_metaqa__L1DEV.npz` (`_l3h_bundle.py`; json 35aa1b06…) carries exactly the population's cache rows.
  - The runner `_l3h_run.py` serves the caches from the bundle, refusing any other row and any open of the full caches. It sets BELOW_NORMAL priority before any work and writes a sidecar per run under `runs/`.
  - A laptop check of the runner against the direct harness (a 200-row typed smoke) was BIT_IDENTICAL.
- **Attempt 1** (job 260929-175057, 1.0 s, rc 1). The run opened the query-embedding shard 9, which the push set (traced from a 200-row smoke that reads only shard 8) lacked. Addendum 1 (12642119…) pushed the two frozen query shards 9 (dense and SPLADE, 136,471,621 bytes), read-only.
- **Attempt 2** (job 260929-175715, 266.7 s, rc 1). The harness computed every row, then refused at its own identity check: "FLAT positions differ from the v1 record".
  - **Cause:** `rx` sets OPENBLAS_NUM_THREADS (and the other thread variables) to the job's `--cpus`. OpenBLAS partitions the float32 dense product by thread count, so the float sums differ and near-tied FLAT ranks flip.
  - The laptop probe (`BLAS_PROBE__laptop.json`, 6774d25f…) shows the product equal to the laptop's default only at 12 threads. At 1, 2, 4 and 8 threads 190,921–193,317 of its 8,000,000 entries differ, at 16 threads 3,866,291 and at 32 threads 5,776,133.
- **Addendum 2** (c12815c6…). Every L3_HOST job sets OPENBLAS_NUM_THREADS=12 (`--var`, a thread count, not a credential) and reserves `--cpus 2 --mem 3`. A host probe gates the rerun.
  - The host probe `BLAS_PROBE__host_t12.json` (2b1559cb…) reproduced the laptop's inputs, its default product and all seven per-thread-count shas.
  - The host (i9-14900) and the laptop (i5-1335U) run the same OpenBLAS 0.3.30 with the Haswell kernel.
- **Attempt 3** (job 260929-181403, 272.8 s, rc 0) was **BIT_IDENTICAL** to the laptop run (`PLACEMENT_TEST__L3_HOST__v1.json`, 62135f84…): all 101 npz members (6,947,018 array bytes), and the json apart from its volatile keys.
- **Authorization** (`HOST_STAGE_AUTHORIZATION__L3_HOST__2026-09-29.json`, cb94b481…).
  - It covers L3 development runs of the L1_DEV MetaQA population on the host with `_l3d_typed.py` and `_l3d_hop.py`.
  - Terms: through `rx` and `_l3h_run.py`, OPENBLAS_NUM_THREADS=12, `--cpus 2 --mem 3`, no GPU.
  - A new harness joins only after its first host run is BIT_IDENTICAL to its laptop run and a dated addendum names it.
- **The budget harness joined.**
  - Addendum 3 (78e0644b…) declared its placement test and pushed `_l3d_budget.py` and the TYPED v1 records it compares with.
  - The host run (job 260929-182107, 571.6 s, rc 0) was BIT_IDENTICAL (`PLACEMENT_TEST__L3_BUDGET__v1.json`, 5086a636…): 421 npz members, identical npz bytes and 0 json differences.
  - Addendum 4 (854a483a…) names it.
- **Speed.** At BELOW_NORMAL, beside other users' work, the host ran TYPED in 272.8 s (laptop 417.8 s) and BUDGET in 571.6 s (laptop 1,593.2 s).
- **GPU.** No L3 stage is GPU-bound: the walks are sparse CPU work over cached scores. The lane requests no GPU.

**40.14 Limits.**
- **The schedule is lexical.** It needs relation names whose tokens appear in the question, so a paraphrase defeats it ("types" for genre). That is the whole measured residual of T1_ROWS apart from the gate.
- **One KB and one population.** MetaQA's KB has 9 relation labels with short names; Freebase's schema is far larger. Nothing here ran on WebQSP, whose development population is the user's decision.
- **The ROWS and budget arms read adjacency outside the contacted shards.** The reads are counted, but a deployed system would need rows addressable by node, beside the shard store.
- **The lane's constants** (EPS, H 3, gate G2) were confirmed on DEV_B for the P90 lane, not re-derived here.
- **Development only.** The rows are legacy-exposed and every p-value is descriptive.

**40.15 What comes next** (§37.9's order stands; nothing held out is read).
- **E2, a dataset-agnostic relation schedule** (has run: §41; this bullet is the proposal as written before the run). The idea: score relation names against the question with the canonical encoders (the dense gte-Qwen2 encoder and SPLADE, both already cached locally) instead of, or beside, the lexical match. It targets the 28 schedule misses, and it is the step that would carry the typed walk to WebQSP's schema.
- **WebQSP** needs a development population, which is the user's decision (§37.9). Its maps are done (§37.10).
- **Then:** finalize L1 + L2 + L3 on the development rows, then held-out A.

**40.16 Cost and hygiene.**
- **Laptop.** TYPED took 417.8 s (peak RSS 825.7 MB) and BUDGET 1,593.2 s (827.7 MB), one heavy process at a time.
- **Host.** Five `rx` jobs of the project `crag` ran at BELOW_NORMAL: two failed placement attempts (1 CPU each), the probe and two placement runs (2 CPUs each). Other users' jobs and the capacity reserve were left alone. No credential was placed on the host. Only code, the bundle, the two query shards and the TYPED v1 records were pushed, each as a declared transfer.
- **Hygiene.** Nothing under `data/` was written on either machine, and no pinned module or CONTRACT_FILE was edited. No held-out, split-B or TEST row was read.
- **New code:** `_l3d_typed.py` (8eb4cbde…), `_l3d_budget.py` (cd95edc9…), `_l3h_run.py` (757df4b7…), `_l3h_place.py` (3ca471a1…), `_l3h_bundle.py` (ec720a40…) and `_l3h_blasprobe.py` (3ed95dd6…), plus the `rx.toml` push list. The harnesses are sha-pinned by their records, and the host code by the declaration and the authorization.

## 41. E2: an encoder-scored relation schedule for the typed walk, and the host encoder lane (2026-09-29 local) (`_l3d_e2.py`, which imports the pinned `_l3d_typed.py` walker and the L1_P90 lane's functions; records `results/L3_DEV/l3e2_metaqa__v1.{json,npz}` (json e3814532…, npz 51200669…) and `relenc_{metaqa,webqsp}__v1.{json,npz}`; host encoder lane `_enc_hfdl.py` and `_enc_host.py`, host-yield wrapper `_host_yield.py`, records `results/L3_HOST/` and `results/HOST_YIELD/`; STATUS = DEVELOPMENT: descriptive numbers, p-values descriptive, no rule selected, L1 unchanged)

**41.1 Why.** TYPED (§40) serves ALL gold on 1,967 of the 1,998 rows at B_N 1000 (PHG T1_ROWS). Its measured residual is the lexical schedule.
- 28 rows are NOT_REACHED with a schedule shorter than the question's hops. 23 of them are genre questions worded "types" (§40.10).
- A lexical schedule also needs relation names that share tokens with the questions. WebQSP's frozen graph has 7,058 relation labels in Freebase's naming, which promises no such overlap.
- E2 scores the relation names against the question with the frozen canonical encoders, instead of or beside the lexical match (§40.15).
- No encoder training, no LLM, no learned weight and no tuned constant: K0 60, EPS 1e-6 and H 3 are the frozen and lane constants.

**41.2 The host encoder lane** (stage L3_HOST, addenda 5–8; records `results/L3_HOST/`).
- **Why the host.** The laptop had 1,075 MB of 16,069 MB free (19:25 local), too little to load the 1.5B dense encoder beside mpr's running jobs. The weights (about 8 GB) are too large for the relay. The user directed: "just download the model there".
- **Download** (addendum 5, 5de9f9bd…; `_enc_hfdl.py`, b490b3cf…).
  - It was anonymous, from the public repositories, with no token of any kind.
  - It fetched gte-Qwen2-1.5B-instruct at a9af15a6… (the laptop cache's revision) and SPLADE cocondenser-ensembledistil at 49cf4c7b… and 83f6dbef….
  - `HF_DOWNLOAD__v1.json` (bd183aa0…): 24 files, 7,993,483,339 bytes, 193.3 s. Every file was verified against the laptop cache (20 by git blob sha1, 4 by sha256).
- **Setup and probe** (addendum 6, dbf7ea22…; `_enc_host.py`).
  - Dense: SentenceTransformer in fp16 with sdpa attention and `use_cache` off (the remote Qwen code calls a cache method that transformers 4.57.3 lacks).
  - SPLADE: max-pooled log(1 + ReLU) logits, max length 256.
  - A CPU probe loads both models and encodes two fixed strings before the GPU is asked for.
  - **Probe attempt 1** (job 260929-195557, CPU) crashed before any encoding. In offline mode, the tokenizer loader calls the Hub's model_info for a repo id. Addendum 7 (9cbbe6ef…) records the change: both models now load from their local snapshot directories (module d9373dfc… → 22367a4f…).
  - **Probe attempt 2** (`ENC_PROBE__v1.json`, 7d0f1d56…) was OK in 8.9 s. It also found that the two SPLADE weight files differ in key count: pytorch_model.bin has 205 keys and model.safetensors 203. The 203 shared tensors are identical. The 2 bin-only keys, `cls.predictions.decoder.{weight,bias}`, are tied tensors that the safetensors conversion stores once.
- **Addendum 8** (ebc80655…, before any REPRO run).
  - Every job of the stage runs under the host-yield wrapper (below).
  - The REPRO gate's weight-file item now reads: identical shared keys, and every bin-only key an exact copy of its tie target.
  - RELENC writes nothing until both datasets are encoded. The module became d26727ae….
- **Reproduction test** (`ENC_REPRO__v1.json`, 7bb60d8f…; job 260929-201959; RTX 4500 Ada, torch 2.8.0+cu128, transformers 4.57.3; 251.3 s). The host re-encodes all 43,234 MetaQA node texts in the canonical batches, plus the 1,998 population queries, and compares them with the frozen stores:

| store | rows | cosine mean | cosine min | nearest frozen row |
|---|---|---|---|---|
| dense docs | 43,234 | 0.9999967 | 0.9994217 | same text: 43,234 |
| SPLADE docs (at the pointer row) | 43,234 | 1.0 | 0.9999997 | same stored row: 43,234 |
| dense queries, `question` | 1,998 | 0.9999974 | 0.9999636 | — (190 rows exactly equal) |
| dense queries, `question_plain` | 1,998 | 0.9833879 | 0.8749582 | — |
| SPLADE queries, `question` (reported, not gated) | 1,998 | 0.9998854 | 0.9956521 | — |

  - **SPLADE pointers.** The SPLADE doc store reuses a row for 3,087 positions (3,083 of them case or accent variants of an earlier name), so it holds 40,147 distinct rows. The archived pointer index shows this, so frozen position i is compared with the re-encoding at its pointer row.
  - **Gate** (declared in addendum 6, before any encoder ran):
    - docs: cosine mean ≥ 0.999, min ≥ 0.99, and a nearest-row hit share ≥ 0.999;
    - queries: the better dense field has mean ≥ 0.999 and min ≥ 0.99;
    - the weight-file item.
  - **PASS.** The frozen queries were encoded from the `question` field.
- **Authorization** (`HOST_STAGE_AUTHORIZATION__L3_HOST_ENC__2026-09-29.json`, 9313ed0f…, 20:27:21 local).
  - It lets RELENC run: the relation-name strings of the two frozen graph manifests only. No query row or query text is read, and nothing is evaluated.
  - Any other encoding needs its own addendum.
- **RELENC** (job 260929-202738, 20:28:13–20:37:19 local; 1 GPU, BELOW_NORMAL). Each string is encoded alone, as a document. A relation label is verbalised as the lane does it, with `_` and `.` turned into spaces.
  - `relenc_metaqa__v1` (json 6290f3bd…, npz 17231a29…): the 9 relation labels plus the 72 ordered pairs of distinct labels ("<r1>, <r2>"), 81 strings, 7.1 s.
  - `relenc_webqsp__v1` (json d16a1767…, npz aaec8d03…): the 7,058 relation labels of the frozen WebQSP graph manifest (4130ad1f…), 515.6 s. Nothing reads it yet.
- **The host-yield wrapper** (`HOST_YIELD_POLICY__v1.json`, b468a0d5…, declared 20:19:22; `_host_yield.py`, b96c2bb8…). The user set the host order mpr > crag > jigsaw: "you have the priority over jigsaw", and "if your launcher detects any job from mpr in the queue you should prempt it".
  - **Who enforces what.** rx admits jobs by its own priority map ({mpr: 10}, unchanged). Jigsaw's launcher yields to mpr and crag. The wrapper makes crag's own jobs yield.
  - **Rule.** A crag job yields when a queued mpr job, refused by rx now, would be admitted once jigsaw's jobs and the fewest crag jobs (youngest first) gave back what they hold, and this job is one of those. Room that jigsaw's jobs alone would free counts until the mpr job has waited 120 s.
  - **Mechanics.** A yield stops the job's process tree (exit 75).
    - A waiter sized 1e-11 CPU and GB (below rx's fit tolerance) relaunches the same command once rx would admit it. Every hop checks the code shas and stops on a change (exit 3).
    - A guard job applies the same rule to the crag jobs started without the wrapper and cancels one that must yield. It relaunches nothing.
  - **Tests.** The rule: 34 checks, 0 failed (`_host_yield_test_rule.py`, ee6e1776…). End to end on a fake rx home and agent: 15 checks, 0 failed (`_host_yield_test_e2e.py`, 97657f75…). In that test a GPU job yielded to a queued mpr GPU job in 6.6 s (confirm window 2 s).
  - **So far.** REPRO, RELENC and the E2 placement run each ran under the wrapper (logs `results/HOST_YIELD/{crag-encrepro,crag-relenc,l3h-e2-h1}.jsonl`). Each logged start, child and done with rc 0, and none yielded.
  - The guard (job 260929-201957) watches the one unwrapped crag job, the 2Wiki K 59898 map (§37.10). It has cancelled nothing.

**41.3 Scores and schedules.** The seeds, the gate (G2: typed graph and anchored) and the walker are TYPED v1's, imported unchanged. What is new is the relation order the walker receives.
- **Scores.** Per row, each of the 81 strings is scored from the frozen vectors only; no model runs in E2.
  - **D:** the dense cosine between the row's frozen query vector (the one FLAT reads) and the string's vector, float64.
  - **S:** the SPLADE dot product, float64.
  - **DS:** the RRF of the D rank and the S rank, K0 60; ties → the D rank.
  - Every ranking breaks remaining ties by the string index.
- **Schedules** (c ∈ {D, S, DS}):
  - **LEX:** the lexical `schedule3` (== TYPED v1, asserted).
  - **XT_c:** LEX, plus the channel's top relation appended as the answer relation when LEX does not hold it.
  - **XA_c:** LEX plus the channel's best relation outside LEX, always appended. It is a diagnostic of what appending costs a complete schedule.
  - **P_c:** the channel's best of the 81 strings (one relation or an ordered pair), without the lexical match. This is the transfer case.
- **The walk** (TYPED v1). H 3; step k walks the relations order[:k], and the mass along order[−1] is the answer channel. So a one-relation schedule walks that relation on all three steps, and an empty schedule leaves the seed as the only candidate.
- **Candidates and arms.**
  - CT(schedule) is the unmasked (ROWS) typed candidate list.
  - UA_c is the RRF of CT(LEX) and CT(XA_c), K0 60; ties → FLAT rank.
  - Every arm serves its list first, then the cell's L1 order, at n(q, B_N) = min(B_N, contacted mass). Outside the gate every arm serves L1 (asserted).
  - The cells are UNROUTED, PHG_k432 and MTK_k432.
- **Read beyond TYPED v1:** the relation-name encodings and the population rows' frozen query vectors. The topic entity and the question type only stratify the report.

**41.4 Identities asserted.**
- FLAT == loc v1 on all 14,489 gold nodes. The dense and SPLADE top-100 agreement gate holds (overlap mean 1.0 for both).
- The typed graph's relation vocabulary (9 labels, 267,142 typed edge entries) == the RELENC strings' labels. No row has an all-zero SPLADE score.
- On all 1,998 rows, the seeds, the lexical schedule and the gate == TYPED v1.
- On every gold node, these equal TYPED v1: the L1 positions (unrouted and routed), the contacted flags, the contacted mass and B_P.
- LEX == TYPED v1's T1 (unrouted) and T1_ROWS (routed), and the gold ranks in CT(LEX) == TYPED v1's.

**41.5 ALL-gold rows** (of 1,998).

| cell | arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|---|
| UNROUTED | L1 | 763 (0.382) | 855 (0.428) | 919 (0.460) | 1,015 (0.508) | 1,110 (0.556) | 1,271 (0.636) |
| UNROUTED | LEX | 1,916 (0.959) | 1,960 (0.981) | 1,962 (0.982) | 1,966 (0.984) | 1,968 (0.985) | 1,972 (0.987) |
| UNROUTED | XT_D | 1,873 (0.937) | 1,917 (0.959) | 1,928 (0.965) | 1,939 (0.970) | 1,947 (0.974) | 1,964 (0.983) |
| UNROUTED | XT_S | 1,900 (0.951) | 1,943 (0.972) | 1,948 (0.975) | 1,956 (0.979) | 1,960 (0.981) | 1,972 (0.987) |
| UNROUTED | XT_DS | 1,877 (0.939) | 1,920 (0.961) | 1,928 (0.965) | 1,941 (0.971) | 1,949 (0.975) | 1,967 (0.984) |
| UNROUTED | XA_D | 628 (0.314) | 754 (0.377) | 858 (0.429) | 998 (0.499) | 1,107 (0.554) | 1,303 (0.652) |
| UNROUTED | XA_S | 599 (0.300) | 758 (0.379) | 862 (0.431) | 1,017 (0.509) | 1,115 (0.558) | 1,304 (0.653) |
| UNROUTED | XA_DS | 602 (0.301) | 738 (0.369) | 845 (0.423) | 999 (0.500) | 1,104 (0.553) | 1,305 (0.653) |
| UNROUTED | P_D | 999 (0.500) | 1,121 (0.561) | 1,186 (0.594) | 1,313 (0.657) | 1,397 (0.699) | 1,564 (0.783) |
| UNROUTED | P_S | 893 (0.447) | 1,019 (0.510) | 1,087 (0.544) | 1,213 (0.607) | 1,317 (0.659) | 1,522 (0.762) |
| UNROUTED | P_DS | 1,076 (0.539) | 1,192 (0.597) | 1,254 (0.628) | 1,380 (0.691) | 1,460 (0.731) | 1,619 (0.810) |
| UNROUTED | UA_D | 1,877 (0.939) | 1,946 (0.974) | 1,958 (0.980) | 1,968 (0.985) | 1,974 (0.988) | 1,986 (0.994) |
| UNROUTED | UA_S | 1,887 (0.944) | 1,967 (0.984) | 1,975 (0.988) | 1,981 (0.991) | 1,983 (0.992) | 1,992 (0.997) |
| UNROUTED | UA_DS | 1,876 (0.939) | 1,945 (0.973) | 1,955 (0.978) | 1,967 (0.984) | 1,973 (0.987) | 1,988 (0.995) |
| PHG | L1 | 763 (0.382) | 876 (0.438) | 999 (0.500) | 1,112 (0.557) | 1,208 (0.605) | 1,294 (0.648) |
| PHG | LEX | 1,916 (0.959) | 1,959 (0.980) | 1,963 (0.982) | 1,967 (0.984) | 1,970 (0.986) | 1,973 (0.987) |
| PHG | XT_D | 1,873 (0.937) | 1,918 (0.960) | 1,928 (0.965) | 1,941 (0.971) | 1,948 (0.975) | 1,963 (0.982) |
| PHG | XT_S | 1,900 (0.951) | 1,942 (0.972) | 1,949 (0.975) | 1,957 (0.979) | 1,962 (0.982) | 1,972 (0.987) |
| PHG | XT_DS | 1,877 (0.939) | 1,920 (0.961) | 1,929 (0.965) | 1,944 (0.973) | 1,951 (0.976) | 1,968 (0.985) |
| PHG | XA_D | 628 (0.314) | 773 (0.387) | 934 (0.467) | 1,094 (0.548) | 1,199 (0.600) | 1,314 (0.658) |
| PHG | XA_S | 599 (0.300) | 776 (0.388) | 936 (0.468) | 1,109 (0.555) | 1,205 (0.603) | 1,318 (0.660) |
| PHG | XA_DS | 602 (0.301) | 757 (0.379) | 920 (0.460) | 1,093 (0.547) | 1,194 (0.598) | 1,316 (0.659) |
| PHG | P_D | 999 (0.500) | 1,132 (0.567) | 1,225 (0.613) | 1,384 (0.693) | 1,456 (0.729) | 1,555 (0.778) |
| PHG | P_S | 893 (0.447) | 1,035 (0.518) | 1,137 (0.569) | 1,291 (0.646) | 1,407 (0.704) | 1,546 (0.774) |
| PHG | P_DS | 1,076 (0.539) | 1,203 (0.602) | 1,291 (0.646) | 1,448 (0.725) | 1,521 (0.761) | 1,611 (0.806) |
| PHG | UA_D | 1,877 (0.939) | 1,946 (0.974) | 1,958 (0.980) | 1,967 (0.984) | 1,972 (0.987) | 1,982 (0.992) |
| PHG | UA_S | 1,887 (0.944) | 1,966 (0.984) | 1,974 (0.988) | 1,979 (0.990) | 1,981 (0.991) | 1,988 (0.995) |
| PHG | UA_DS | 1,876 (0.939) | 1,944 (0.973) | 1,955 (0.978) | 1,966 (0.984) | 1,971 (0.986) | 1,984 (0.993) |
| MTK | L1 | 763 (0.382) | 871 (0.436) | 981 (0.491) | 1,082 (0.542) | 1,169 (0.585) | 1,249 (0.625) |
| MTK | LEX | 1,916 (0.959) | 1,960 (0.981) | 1,963 (0.982) | 1,967 (0.984) | 1,970 (0.986) | 1,973 (0.987) |
| MTK | XT_D | 1,873 (0.937) | 1,918 (0.960) | 1,928 (0.965) | 1,939 (0.970) | 1,947 (0.974) | 1,960 (0.981) |
| MTK | XT_S | 1,900 (0.951) | 1,943 (0.972) | 1,949 (0.975) | 1,957 (0.979) | 1,962 (0.982) | 1,972 (0.987) |
| MTK | XT_DS | 1,877 (0.939) | 1,920 (0.961) | 1,929 (0.965) | 1,942 (0.972) | 1,949 (0.975) | 1,965 (0.983) |
| MTK | XA_D | 628 (0.314) | 769 (0.385) | 918 (0.459) | 1,062 (0.532) | 1,162 (0.582) | 1,273 (0.637) |
| MTK | XA_S | 599 (0.300) | 772 (0.386) | 920 (0.460) | 1,078 (0.540) | 1,169 (0.585) | 1,276 (0.639) |
| MTK | XA_DS | 602 (0.301) | 753 (0.377) | 905 (0.453) | 1,061 (0.531) | 1,158 (0.580) | 1,274 (0.638) |
| MTK | P_D | 999 (0.500) | 1,132 (0.567) | 1,218 (0.610) | 1,352 (0.677) | 1,421 (0.711) | 1,521 (0.761) |
| MTK | P_S | 893 (0.447) | 1,033 (0.517) | 1,125 (0.563) | 1,263 (0.632) | 1,360 (0.681) | 1,487 (0.744) |
| MTK | P_DS | 1,076 (0.539) | 1,203 (0.602) | 1,285 (0.643) | 1,418 (0.710) | 1,485 (0.743) | 1,572 (0.787) |
| MTK | UA_D | 1,877 (0.939) | 1,946 (0.974) | 1,958 (0.980) | 1,967 (0.984) | 1,972 (0.987) | 1,982 (0.992) |
| MTK | UA_S | 1,887 (0.944) | 1,967 (0.984) | 1,974 (0.988) | 1,979 (0.990) | 1,981 (0.991) | 1,988 (0.995) |
| MTK | UA_DS | 1,876 (0.939) | 1,944 (0.973) | 1,955 (0.978) | 1,966 (0.984) | 1,971 (0.986) | 1,984 (0.993) |

**41.6 By hop and answer count.** At B_N 1000:

| cell | arm | hop 1 (666) | hop 2 (666) | hop 3 (666) | 5–10 answers (287) | 11+ answers (339) |
|---|---|---|---|---|---|---|
| UNROUTED | L1 | 666 (1.000) | 187 (0.281) | 162 (0.243) | 53 (0.185) | 21 (0.062) |
| UNROUTED | LEX | 666 (1.000) | 659 (0.989) | 641 (0.962) | 272 (0.948) | 331 (0.976) |
| UNROUTED | XT_D | 649 (0.974) | 652 (0.979) | 638 (0.958) | 267 (0.930) | 327 (0.965) |
| UNROUTED | XT_S | 656 (0.985) | 659 (0.989) | 641 (0.962) | 270 (0.941) | 330 (0.973) |
| UNROUTED | XT_DS | 647 (0.971) | 656 (0.985) | 638 (0.958) | 268 (0.934) | 328 (0.968) |
| UNROUTED | XA_D | 608 (0.913) | 207 (0.311) | 183 (0.275) | 60 (0.209) | 23 (0.068) |
| UNROUTED | XA_S | 628 (0.943) | 202 (0.303) | 187 (0.281) | 64 (0.223) | 24 (0.071) |
| UNROUTED | XA_DS | 613 (0.920) | 203 (0.305) | 183 (0.275) | 64 (0.223) | 23 (0.068) |
| UNROUTED | P_D | 614 (0.922) | 353 (0.530) | 346 (0.520) | 129 (0.449) | 145 (0.428) |
| UNROUTED | P_S | 618 (0.928) | 340 (0.511) | 255 (0.383) | 105 (0.366) | 124 (0.366) |
| UNROUTED | P_DS | 623 (0.935) | 407 (0.611) | 350 (0.526) | 138 (0.481) | 172 (0.507) |
| UNROUTED | UA_D | 649 (0.974) | 663 (0.995) | 656 (0.985) | 278 (0.969) | 334 (0.985) |
| UNROUTED | UA_S | 656 (0.985) | 662 (0.994) | 663 (0.995) | 283 (0.986) | 335 (0.988) |
| UNROUTED | UA_DS | 647 (0.971) | 663 (0.995) | 657 (0.986) | 280 (0.976) | 334 (0.985) |
| PHG | L1 | 664 (0.997) | 220 (0.330) | 228 (0.342) | 80 (0.279) | 32 (0.094) |
| PHG | LEX | 665 (0.998) | 662 (0.994) | 640 (0.961) | 272 (0.948) | 330 (0.973) |
| PHG | XT_D | 648 (0.973) | 656 (0.985) | 637 (0.956) | 266 (0.927) | 326 (0.962) |
| PHG | XT_S | 655 (0.983) | 662 (0.994) | 640 (0.961) | 270 (0.941) | 329 (0.971) |
| PHG | XT_DS | 646 (0.970) | 660 (0.991) | 638 (0.958) | 268 (0.934) | 327 (0.965) |
| PHG | XA_D | 606 (0.910) | 238 (0.357) | 250 (0.375) | 89 (0.310) | 35 (0.103) |
| PHG | XA_S | 626 (0.940) | 231 (0.347) | 252 (0.378) | 91 (0.317) | 36 (0.106) |
| PHG | XA_DS | 611 (0.917) | 232 (0.348) | 250 (0.375) | 91 (0.317) | 35 (0.103) |
| PHG | P_D | 612 (0.919) | 386 (0.580) | 386 (0.580) | 145 (0.505) | 152 (0.448) |
| PHG | P_S | 616 (0.925) | 377 (0.566) | 298 (0.447) | 122 (0.425) | 132 (0.389) |
| PHG | P_DS | 621 (0.932) | 440 (0.661) | 387 (0.581) | 155 (0.540) | 177 (0.522) |
| PHG | UA_D | 648 (0.973) | 664 (0.997) | 655 (0.983) | 278 (0.969) | 333 (0.982) |
| PHG | UA_S | 655 (0.983) | 663 (0.995) | 661 (0.992) | 283 (0.986) | 334 (0.985) |
| PHG | UA_DS | 646 (0.970) | 664 (0.997) | 656 (0.985) | 280 (0.976) | 333 (0.982) |

At B_N 100 (PHG):

| cell | arm | hop 1 (666) | hop 2 (666) | hop 3 (666) | 5–10 answers (287) | 11+ answers (339) |
|---|---|---|---|---|---|---|
| PHG | L1 | 656 (0.985) | 42 (0.063) | 65 (0.098) | 36 (0.125) | 8 (0.024) |
| PHG | LEX | 661 (0.992) | 644 (0.967) | 611 (0.917) | 270 (0.941) | 306 (0.903) |
| PHG | XT_D | 628 (0.943) | 638 (0.958) | 607 (0.911) | 260 (0.906) | 303 (0.894) |
| PHG | XT_S | 643 (0.965) | 646 (0.970) | 611 (0.917) | 265 (0.923) | 305 (0.900) |
| PHG | XT_DS | 627 (0.941) | 642 (0.964) | 608 (0.913) | 262 (0.913) | 304 (0.897) |
| PHG | XA_D | 473 (0.710) | 71 (0.107) | 84 (0.126) | 32 (0.111) | 8 (0.024) |
| PHG | XA_S | 446 (0.670) | 62 (0.093) | 91 (0.137) | 35 (0.122) | 8 (0.024) |
| PHG | XA_DS | 453 (0.680) | 64 (0.096) | 85 (0.128) | 33 (0.115) | 8 (0.024) |
| PHG | P_D | 487 (0.731) | 248 (0.372) | 264 (0.396) | 112 (0.390) | 131 (0.386) |
| PHG | P_S | 530 (0.796) | 209 (0.314) | 154 (0.231) | 83 (0.289) | 103 (0.304) |
| PHG | P_DS | 503 (0.755) | 305 (0.458) | 268 (0.402) | 118 (0.411) | 154 (0.454) |
| PHG | UA_D | 628 (0.943) | 642 (0.964) | 607 (0.911) | 270 (0.941) | 281 (0.829) |
| PHG | UA_S | 643 (0.965) | 631 (0.947) | 613 (0.920) | 276 (0.962) | 269 (0.794) |
| PHG | UA_DS | 627 (0.941) | 639 (0.959) | 610 (0.916) | 272 (0.948) | 280 (0.826) |

**41.7 Paired** (rows gained / lost, McNemar p, descriptive), on PHG. At B_N 1000:

| arm | vs LEX: all rows | hop 1 | hop 2 | hop 3 | vs L1 | gold nodes vs LEX |
|---|---|---|---|---|---|---|
| LEX | — | — | — | — | +855 / 0 (p 8.3e-258) | — |
| XT_D | +1 / −27 (p 2.2e-7) | 0 / −17 | +1 / −7 | 0 / −3 | +846 / −17 (p 6.5e-225) | +15 / −150 |
| XT_S | +1 / −11 (p 6.3e-3) | 0 / −10 | +1 / −1 | 0 / 0 | +855 / −10 (p 5.0e-238) | +15 / −60 |
| XT_DS | +1 / −24 (p 1.5e-6) | 0 / −19 | +1 / −3 | 0 / −2 | +851 / −19 (p 1.2e-223) | +15 / −139 |
| XA_D | +17 / −890 (p 8.7e-238) | 0 / −59 | +2 / −426 | +15 / −405 | +40 / −58 (p 0.09) | +49 / −8,117 |
| XA_S | +22 / −880 (p 4.3e-228) | 0 / −39 | +1 / −432 | +21 / −409 | +36 / −39 (p 0.82) | +63 / −8,380 |
| XA_DS | +18 / −892 (p 5.7e-237) | 0 / −54 | +2 / −432 | +16 / −406 | +34 / −53 (p 0.05) | +50 / −8,254 |
| P_D | +1 / −584 (p 9.3e-174) | 0 / −53 | +1 / −277 | 0 / −254 | +366 / −94 (p 6.0e-39) | +5 / −5,511 |
| P_S | +1 / −677 (p 1.1e-201) | 0 / −49 | +1 / −286 | 0 / −342 | +294 / −115 (p 3.6e-19) | +15 / −5,742 |
| P_DS | +6 / −525 (p 8.7e-147) | 0 / −44 | +2 / −224 | +4 / −257 | +424 / −88 (p 9.4e-54) | +29 / −4,729 |
| UA_D | +17 / −17 (p 1.00) | 0 / −17 | +2 / 0 | +15 / 0 | +872 / −17 (p 1.6e-232) | +48 / −61 |
| UA_S | +22 / −10 (p 0.05) | 0 / −10 | +1 / 0 | +21 / 0 | +877 / −10 (p 1.5e-244) | +63 / −41 |
| UA_DS | +18 / −19 (p 1.00) | 0 / −19 | +2 / 0 | +16 / 0 | +873 / −19 (p 4.8e-230) | +49 / −58 |

At B_N 100:

| arm | vs LEX: all rows | hop 1 | hop 2 | hop 3 | vs L1 | gold nodes vs LEX |
|---|---|---|---|---|---|---|
| LEX | — | — | — | — | +1,154 / −1 (p 0) | — |
| XT_D | +3 / −46 (p 7.0e-11) | 0 / −33 | +3 / −9 | 0 / −4 | +1,144 / −34 (p 0) | +22 / −220 |
| XT_S | +3 / −19 (p 8.6e-4) | 0 / −18 | +3 / −1 | 0 / 0 | +1,156 / −19 (p 0) | +22 / −99 |
| XT_DS | +3 / −42 (p 8.7e-10) | 0 / −34 | +3 / −5 | 0 / −3 | +1,149 / −35 (p 0) | +22 / −199 |
| XA_D | +26 / −1,314 (p 0) | 0 / −188 | +9 / −582 | +17 / −544 | +50 / −185 (p 2.1e-19) | +144 / −11,213 |
| XA_S | +34 / −1,351 (p 0) | 0 / −215 | +7 / −589 | +27 / −547 | +49 / −213 (p 1.5e-25) | +174 / −11,593 |
| XA_DS | +27 / −1,341 (p 0) | 0 / −208 | +9 / −589 | +18 / −544 | +44 / −205 (p 5.0e-26) | +149 / −11,426 |
| P_D | +2 / −919 (p 4.8e-272) | 0 / −174 | +2 / −398 | 0 / −347 | +432 / −196 (p 2.3e-21) | +17 / −7,046 |
| P_S | +4 / −1,027 (p 4.1e-300) | 0 / −131 | +4 / −439 | 0 / −457 | +293 / −163 (p 1.2e-9) | +32 / −7,929 |
| P_DS | +10 / −850 (p 1.5e-236) | 0 / −158 | +5 / −344 | +5 / −348 | +494 / −181 (p 1.9e-34) | +74 / −6,226 |
| UA_D | +26 / −65 (p 5.3e-5) | 0 / −33 | +9 / −11 | +17 / −21 | +1,148 / −34 (p 0) | +130 / −904 |
| UA_S | +33 / −62 (p 3.8e-3) | 0 / −18 | +6 / −19 | +27 / −25 | +1,144 / −20 (p 0) | +166 / −1,189 |
| UA_DS | +27 / −67 (p 4.5e-5) | 0 / −34 | +9 / −14 | +18 / −19 | +1,148 / −35 (p 0) | +135 / −953 |

At B_N 5000:

| arm | vs LEX: all rows | hop 1 | hop 2 | hop 3 | vs L1 | gold nodes vs LEX |
|---|---|---|---|---|---|---|
| LEX | — | — | — | — | +679 / 0 (p 8.0e-205) | — |
| XT_D | +1 / −11 (p 6.3e-3) | 0 / −3 | +1 / −5 | 0 / −3 | +672 / −3 (p 6.5e-196) | +8 / −94 |
| XT_S | +1 / −2 (p 1.00) | 0 / −1 | +1 / −1 | 0 / 0 | +679 / −1 (p 2.7e-202) | +8 / −27 |
| XT_DS | +1 / −6 (p 0.13) | 0 / −2 | +1 / −2 | 0 / −2 | +676 / −2 (p 3.7e-199) | +8 / −81 |
| XA_D | +12 / −671 (p 9.9e-181) | 0 / −10 | +2 / −353 | +10 / −308 | +29 / −9 (p 1.7e-3) | +25 / −6,262 |
| XA_S | +16 / −671 (p 3.1e-175) | 0 / −2 | +1 / −357 | +15 / −312 | +25 / −1 (p 8.0e-7) | +33 / −6,462 |
| XA_DS | +13 / −670 (p 5.1e-179) | 0 / −3 | +2 / −357 | +11 / −310 | +24 / −2 (p 1.0e-5) | +26 / −6,365 |
| P_D | +1 / −419 (p 3.1e-124) | 0 / −5 | +1 / −243 | 0 / −171 | +271 / −10 (p 3.8e-67) | +3 / −4,233 |
| P_S | +1 / −428 (p 6.2e-127) | 0 / −2 | +1 / −230 | 0 / −196 | +267 / −15 (p 8.1e-61) | +8 / −3,677 |
| P_DS | +6 / −368 (p 1.9e-100) | 0 / −3 | +2 / −194 | +4 / −171 | +325 / −8 (p 4.0e-85) | +17 / −3,549 |
| UA_D | +12 / −3 (p 0.04) | 0 / −3 | +2 / 0 | +10 / 0 | +691 / −3 (p 1.4e-201) | +25 / −17 |
| UA_S | +16 / −1 (p 2.7e-4) | 0 / −1 | +1 / 0 | +15 / 0 | +695 / −1 (p 4.2e-207) | +33 / −8 |
| UA_DS | +13 / −2 (p 7.4e-3) | 0 / −2 | +2 / 0 | +11 / 0 | +692 / −2 (p 5.9e-204) | +26 / −10 |

- **MTK and UNROUTED.** For XT_S, UA_D, UA_S and UA_DS, MTK's paired counts against LEX equal PHG's at B_N 100, 1000 and 5000. UNROUTED equals them at B_N 100.
- UNROUTED at B_N 1000: UA_S +25 / −10 (p 0.02), UA_D +19 / −17 (p 0.87), UA_DS +20 / −19 (p 1.00), XT_S +1 / −11 (p 6.3e-3).
- UNROUTED at B_N 5000: UA_S +21 / −1 (p 1.1e-5), UA_D +17 / −3 (p 2.6e-3), UA_DS +18 / −2 (p 4.0e-4), XT_S +1 / −1 (p 1.00).

**41.8 What the arms miss** (B_N 1000, PHG). The classes are §40.10's, plus one: NOT_REACHED with a schedule at least as long as the question's hops. No row is TOPIC_MISS.

| arm | missed rows | NO_GATE | NOT_REACHED, schedule shorter than hop | NOT_REACHED, schedule at least hop long | CT_TOO_LONG |
|---|---|---|---|---|---|
| LEX | 31 | 3 (hop 2 2, hop 3 1) | 28 (hop 1 1, hop 2 2, hop 3 25) | 0 | 0 |
| XT_D | 57 | 3 (hop 2 2, hop 3 1) | 26 (hop 2 1, hop 3 25) | 28 (hop 1 18, hop 2 7, hop 3 3) | 0 |
| XT_S | 41 | 3 (hop 2 2, hop 3 1) | 26 (hop 2 1, hop 3 25) | 12 (hop 1 11, hop 2 1) | 0 |
| XT_DS | 54 | 3 (hop 2 2, hop 3 1) | 26 (hop 2 1, hop 3 25) | 25 (hop 1 20, hop 2 3, hop 3 2) | 0 |
| XA_D | 904 | 3 (hop 2 2, hop 3 1) | 7 (hop 3 7) | 894 (hop 1 60, hop 2 426, hop 3 408) | 0 |
| XA_S | 889 | 3 (hop 2 2, hop 3 1) | 1 (hop 3 1) | 885 (hop 1 40, hop 2 433, hop 3 412) | 0 |
| XA_DS | 905 | 3 (hop 2 2, hop 3 1) | 6 (hop 3 6) | 896 (hop 1 55, hop 2 432, hop 3 409) | 0 |
| P_D | 614 | 3 (hop 2 2, hop 3 1) | 280 (hop 2 1, hop 3 279) | 330 (hop 1 53, hop 2 277) | 1 (hop 1 1) |
| P_S | 707 | 3 (hop 2 2, hop 3 1) | 367 (hop 3 367) | 337 (hop 1 50, hop 2 287) | 0 |
| P_DS | 550 | 3 (hop 2 2, hop 3 1) | 279 (hop 2 1, hop 3 278) | 268 (hop 1 45, hop 2 223) | 0 |
| UA_D | 31 | 3 (hop 2 2, hop 3 1) | 7 (hop 3 7) | 21 (hop 1 18, hop 3 3) | 0 |
| UA_S | 19 | 3 (hop 2 2, hop 3 1) | 1 (hop 3 1) | 15 (hop 1 11, hop 2 1, hop 3 3) | 0 |
| UA_DS | 32 | 3 (hop 2 2, hop 3 1) | 6 (hop 3 6) | 23 (hop 1 20, hop 3 3) | 0 |

For a UA arm, "schedule" means XA_c's, the longer of its two.

**The 28 NOT_REACHED rows of LEX** (§40.10). Of them, each arm serves: XT_D, XT_S and XT_DS 1 each; XA_D 17, XA_S 22, XA_DS 18; P_D 1, P_S 1, P_DS 6; UA_D 17, UA_S 22, UA_DS 18.
- **XT equals LEX on all 23 "types" rows**, for every channel. The encoders' top relation is the bridge relation that LEX already holds (directed_by 11, starred_actors 8, written_by 4).
- **XA appends has_genre** on 21 of the 23 for S (directed_by on the other 2), on 16 for D (directed_by 6, written_by 1) and on 17 for DS (directed_by 6). UA_c serves exactly the "types" rows on which XA_c appended has_genre, plus row 791.
- **P names the right relations in the wrong order.** P_S's schedule on all 23 starts with has_genre (has_genre,directed_by 11, has_genre,starred_actors 8, has_genre,written_by 4). The walk's answer relation is the last one, so P_S walks to the bridge relation's targets and serves none of the 23.
- **Row 791**, a hop-2 co-actor question with an empty LEX: every channel appends starred_actors. XT, XA and UA serve it on all three channels, and so do P_S and P_DS.
- **UA_S misses 6 of the 28.**
  - Row 611: a hop-1 tag question with an empty LEX, where S picks directed_by.
  - Rows 1298 (hop 2) and 1588 (hop 3): "types" rows where S appends directed_by. UA_D and UA_DS serve 1298.
  - Rows 1477, 1707 and 1901: the three hop-3 rows of §40.10 that carry the same schedule as served rows of their type (written_by,directed_by twice, starred_actors,written_by once). Every channel appends the same off-path relation (starred_actors on the first two, directed_by on the third), and no arm serves them.

**41.9 Schedule diagnostics** (the 1,994 gated rows).
- **XT differs from LEX** on 167 (D), 152 (S) and 161 (DS) rows. Hop 1 holds 150 / 148 / 149 of them; 146 gated hop-1 rows have an empty LEX.
  - On hop 1, XT_S appends in_language 45, has_genre 33, directed_by 31, starred_actors 25, written_by 9, release_year 4 and has_imdb_votes 1.
  - XT_D appends has_genre 45, starred_actors 40, has_tags 17, written_by 17, directed_by 16, has_imdb_votes 6, in_language 6, release_year 2 and has_imdb_rating 1.
  - On hops 2 and 3, XT_S differs on 4 and 0 rows, XT_D on 13 and 4, and XT_DS on 9 and 3.
- **XA always appends.** Its appended relation is mostly a frequent bridge relation. For S it is directed_by on 352 / 407 / 301 rows of hops 1 / 2 / 3, and starred_actors on 148 / 161 / 141.
- **P usually differs from LEX.** It equals LEX exactly on 594 (D), 559 (S) and 761 (DS) rows, and has LEX's relation set on 1,136 / 1,373 / 1,359.
  - P is a pair on 1,817 / 1,629 / 1,654 rows.
  - Its most frequent schedule is written_by,directed_by for D (320) and DS (316), and directed_by,written_by for S (257). has_genre,directed_by is chosen on 150 / 191 / 203 rows.

**41.10 Where the arms lose.**
- **Flood (an empty LEX).** There are 146 gated hop-1 rows with an empty LEX: movie_to_tags 65, tag_to_movie 27, actor_to_movie 21, movie_to_genre 17, movie_to_writer 14, movie_to_imdbrating 1 and movie_to_imdbvotes 1. Their CT(LEX) is the seed alone, so LEX serves the L1 order: 145 of the 146 at B_N 1000.
  - XA_S picks in_language 43, has_genre 33, directed_by 31, starred_actors 25, written_by 9, release_year 4 and has_imdb_votes 1. It never picks has_tags, the tag questions' relation; XA_D picks it on 17 rows and XA_DS on 3.
  - A one-relation schedule walks its relation on all three steps. On a tag question, has_genre floods: movie → genre → every film of that genre → their genres.
  - All 10 of UA_S's losses at B_N 1000 are such movie_to_tags rows with has_genre appended. UA_S's list there holds 1,201–7,470 nodes, which moves the last gold from position 31–102 (LEX) to 1,242–7,527.
  - XT_S loses the same 10 rows, plus one hop-2 row (written_by → written_by,has_genre). UA_D loses 16 movie_to_tags rows and 1 tag_to_movie row, and UA_DS 18 and 1.
  - ALL on these 146 rows at B_N 1000: LEX 145, UA_S 135, XT_S 135, UA_D 128 and UA_DS 126.
- **Dilution (a small B_N).** At B_N 100, UA_S loses 62 rows against LEX:
  - hop 1, 18 rows, all with an empty LEX: movie_to_tags 15 (has_genre 10, in_language 5), tag_to_movie 2 and movie_to_imdbvotes 1;
  - hop 2, 19 rows: movie → actor → movie 15 and actor → movie → actor 3 (directed_by appended), and movie → director → movie 1;
  - hop 3, 25 rows: complete LEX schedules with an off-path relation appended (e.g. movie → actor → movie → writer 8, with directed_by).
  - The RRF interleaves CT(XA)'s wrong answer set with CT(LEX), so at 100 nodes the last golds drop out. From B_N 1000 on, UA_S loses no hop-2 or hop-3 row.

**41.11 Cost.** The typed candidate lists (unrouted) and where the golds rank in them (14,489 gold nodes):

| list | gold nodes in list | rank < 10 | rank < 100 | hop 1 (of 1,334) | hop 2 (of 4,613) | hop 3 (of 8,542) |
|---|---|---|---|---|---|---|
| CT(LEX) | 13,705 (0.946) | 5,855 | 13,365 | 907 | 4,556 | 8,242 |
| CT(XT_D) | 13,823 (0.954) | 5,978 | 13,482 | 1,110 | 4,542 | 8,171 |
| CT(XT_S) | 13,829 (0.954) | 5,946 | 13,488 | 1,034 | 4,553 | 8,242 |
| CT(XT_DS) | 13,811 (0.953) | 5,960 | 13,470 | 1,088 | 4,549 | 8,174 |
| CT(XA_D) | 1,379 (0.095) | 682 | 1,370 | 215 | 301 | 863 |
| CT(XA_S) | 1,010 (0.070) | 541 | 1,006 | 134 | 214 | 662 |
| CT(XA_DS) | 1,170 (0.081) | 601 | 1,166 | 192 | 251 | 727 |
| CT(P_D) | 5,817 (0.401) | 2,407 | 5,772 | 282 | 1,122 | 4,413 |
| CT(P_S) | 5,114 (0.353) | 1,804 | 5,024 | 434 | 1,456 | 3,224 |
| CT(P_DS) | 6,921 (0.478) | 2,792 | 6,830 | 455 | 2,009 | 4,457 |
| UA_D | 14,063 (0.971) | 5,035 | 12,965 | 1,119 | 4,591 | 8,353 |
| UA_S | 14,026 (0.968) | 4,699 | 12,605 | 1,038 | 4,584 | 8,404 |
| UA_DS | 14,046 (0.969) | 5,002 | 12,887 | 1,096 | 4,591 | 8,359 |

The union holds more golds than CT(LEX) (14,026 against 13,705 for S) but ranks fewer in its first 100 (12,605 against 13,365): the dilution of 41.10. The list lengths over the gated rows:

| list | mean | median | p90 | max |
|---|---|---|---|---|
| CT(LEX) | 310.4 | 12 | 531.4 | 9,836 |
| CT(XT_D) | 374.5 | 15 | 724.8 | 9,836 |
| CT(XT_S) | 356.7 | 14 | 665.6 | 9,836 |
| CT(XT_DS) | 376.1 | 15 | 749.6 | 9,836 |
| CT(XA_D) | 180.9 | 5 | 156.0 | 9,169 |
| CT(XA_S) | 126.4 | 7 | 109.0 | 7,470 |
| CT(XA_DS) | 161.3 | 5 | 154.4 | 7,470 |
| CT(P_D) | 287.5 | 8 | 598.7 | 9,836 |
| CT(P_S) | 365.2 | 11 | 913.7 | 8,809 |
| CT(P_DS) | 304.6 | 10 | 641.0 | 9,836 |
| UA_D | 488.7 | 28 | 1,073.8 | 15,438 |
| UA_S | 434.7 | 29.5 | 977.8 | 11,028 |
| UA_DS | 469.4 | 30 | 1,085.0 | 11,028 |

- **Walks.** A gated row needs a mean 4.4 walks (median 4, range 3–8): its distinct schedules plus the untyped walk.
- **Rows read.** The record's rows-read fields count the typed and untyped reads together. That is the untyped walk's reads for every arm, since a typed walk follows a subset of the untyped edges from the same seeds. So they equal TYPED v1's untyped counts: 1,886.9 per gated row unrouted, and 1,519.4 outside the contacted shards on PHG (1,504.6 on MTK). The typed walks' own reads were not recorded, and E1's budgets (§40.11) were not rerun for the E2 schedules.

The fetches at B_N 1000 (served nodes outside the contacted shards):

| cell | arm | fetched nodes per query: mean (share of served) | queries with any | extra shards per query: mean / p95 / max |
|---|---|---|---|---|
| PHG | LEX | 101.0 (0.101) | 1,310 | 48.8 / 279.1 / 325 |
| PHG | XT_D | 118.6 (0.119) | 1,410 | 56.2 / 285.0 / 325 |
| PHG | XT_S | 112.9 (0.113) | 1,376 | 53.6 / 284.0 / 325 |
| PHG | XT_DS | 118.7 (0.119) | 1,399 | 56.1 / 285.0 / 325 |
| PHG | XA_D | 58.7 (0.059) | 1,129 | 27.0 / 227.3 / 320 |
| PHG | XA_S | 50.8 (0.051) | 1,170 | 26.0 / 222.0 / 317 |
| PHG | XA_DS | 59.3 (0.059) | 1,161 | 28.0 / 234.3 / 320 |
| PHG | P_D | 103.5 (0.104) | 1,086 | 48.3 / 279.0 / 325 |
| PHG | P_S | 132.1 (0.132) | 1,212 | 60.5 / 281.0 / 317 |
| PHG | P_DS | 112.9 (0.113) | 1,168 | 52.7 / 282.0 / 325 |
| PHG | UA_D | 137.8 (0.138) | 1,645 | 63.3 / 284.0 / 325 |
| PHG | UA_S | 133.5 (0.134) | 1,649 | 61.9 / 285.0 / 325 |
| PHG | UA_DS | 138.9 (0.139) | 1,647 | 63.7 / 285.0 / 325 |
| MTK | LEX | 101.2 (0.101) | 1,366 | 49.5 / 277.0 / 328 |
| MTK | XT_D | 118.8 (0.119) | 1,462 | 56.9 / 283.1 / 328 |
| MTK | XT_S | 113.2 (0.113) | 1,433 | 54.3 / 283.0 / 328 |
| MTK | XT_DS | 118.8 (0.119) | 1,451 | 56.7 / 284.0 / 328 |
| MTK | XA_D | 59.7 (0.060) | 1,168 | 27.8 / 236.4 / 313 |
| MTK | XA_S | 51.6 (0.052) | 1,199 | 26.8 / 225.0 / 322 |
| MTK | XA_DS | 60.3 (0.060) | 1,191 | 28.9 / 236.4 / 324 |
| MTK | P_D | 105.3 (0.105) | 1,131 | 49.6 / 281.0 / 328 |
| MTK | P_S | 135.7 (0.136) | 1,217 | 62.2 / 285.0 / 322 |
| MTK | P_DS | 114.8 (0.115) | 1,208 | 53.9 / 284.0 / 328 |
| MTK | UA_D | 138.8 (0.139) | 1,671 | 64.4 / 283.0 / 329 |
| MTK | UA_S | 134.4 (0.134) | 1,671 | 63.0 / 285.1 / 329 |
| MTK | UA_DS | 139.9 (0.140) | 1,678 | 64.9 / 285.0 / 329 |

The latency per row (laptop, one process):

| stage | mean ms | median ms | p95 ms |
|---|---|---|---|
| base (IR_L1 score, served order, router order) | 10.0 | 6.1 | 44.1 |
| seeds + lexical schedule + encoder schedules | 0.7 | 0.4 | 2.6 |
| walks (distinct schedules + untyped) | 86.4 | 61.9 | 251.8 |
| candidate lists + UNROUTED orders | 8.4 | 4.5 | 32.6 |
| MTK_k432 ES contacted set + L1 order + arm orders | 6.5 | 3.4 | 24.8 |
| PHG_k432 ES contacted set + L1 order + arm orders | 6.1 | 3.3 | 21.8 |
| FLAT products (amortized) | 4.2 | 4.4 | 5.7 |
| FLAT RRF | 24.6 | 15.9 | 88.4 |

**41.12 E2 on the host.**
- **Addendum 9** (30a4fd5b…, 20:46:11 local) declared the placement test and pushed `_l3d_e2.py`. The RELENC MetaQA record the harness reads was made on the host.
- **The host run** (job 260929-204652, under the wrapper) took 234.6 s at BELOW_NORMAL (235.8 s by the runner's clock); the laptop run took 307.2 s.
- It was **BIT_IDENTICAL** to the laptop run (`PLACEMENT_TEST__L3_E2__v1.json`, 7363fc61…): all 190 npz members (18,596,114 array bytes), identical npz bytes and 0 json differences.
- **Addendum 10** (835aa711…, 20:52:16 local) names it. E2 runs of the L1_DEV MetaQA population may run on the host, under the authorization's terms and the wrapper.

**41.13 Reading** (development numbers, descriptive).
- **Only the union reaches the hop-3 residual.** On PHG at B_N 1000, UA_S serves 1,979 rows against LEX's 1,967 (+22 / −10, p 0.05).
  - The gains are hop 3 (+21 / 0) and hop 2 (+1 / 0): 21 of the 23 "types" rows, plus row 791.
  - At B_N 5000 UA_S is +16 / −1 (p 2.7e-4).
  - UA_D and UA_DS gain about as much at hop 3, but lose more hop-1 rows (17 and 19 at B_N 1000, p 1.00).
- **The encoders' top relation does not repair the schedule.** XT never gains more than 3 rows against LEX, and loses 11–46 rows at B_N 100–1000.
  - On the "types" rows the top relation is the bridge relation that LEX already holds.
  - On the empty-LEX tag questions it is a wrong relation, and the walk floods.
- **Appending alone is harmful; it helps only inside the union.** XA alone serves 1,093–1,109 rows on PHG at B_N 1000, below L1's 1,112: on a complete schedule the appended relation replaces the right answer relation. In the union, CT(LEX) keeps those rows and CT(XA) adds the missing answer relation.
- **The pure encoder schedule does not transfer.** P_DS, the best of the three, serves 1,448 rows on PHG at B_N 1000, 519 below LEX.
  - P equals LEX exactly on only 559–761 of the 1,994 gated rows. It favours a frequent pair (written_by,directed_by).
  - On the "types" rows it names the right relations in the wrong order.
- **Both loss modes are gold-free to avoid.**
  - From B_N 500 on, every loss of UA_S is an empty-LEX row (the flood), and an empty LEX is visible without golds.
  - The losses at B_N 100 come from the RRF interleaving (the dilution). A fill order (CT(LEX) first, then CT(XA)'s new candidates) would keep LEX's list ahead.
  - Both are E2b (41.15).
- **Channels.** S is the best channel for the union at every B_N on every cell; D and DS lose more hop-1 tag rows.
- **Routing does not bind here.** Under ROWS reads, the routed and unrouted counts of the LEX, XT and UA arms are within 4 rows at every B_N.

**41.14 Limits.**
- **One KB.** MetaQA has 9 short, distinct relation labels, and the pair strings exist only for it. WebQSP's 7,058 labels are encoded (RELENC), but nothing ran on WebQSP. Its development population is the user's decision.
- **Whole-question scoring.** A relation is scored against the whole question's vector, not against the part of the question that names its hop. So the channels' top relation is often the bridge relation, and a score says nothing about where in the schedule the relation belongs.
- **Always appending is wrong on complete schedules** (XA); it is useful only inside the union.
- **No gold-free signal of an incomplete schedule was tested** beyond LEX's emptiness (41.15).
- **The typed walks' own reads were not recorded** (41.11).
- **Development only.** The rows are legacy-exposed and every p-value is descriptive. The strings use the lane's verbalisation; no prompt or instruction was tuned.

**41.15 What comes next** (§37.9's order stands; nothing held out is read).
- **E2b, a guarded union** (has run: §42; this bullet is the proposal as written before the run, and §42.1 confirms that the UAG table below was reproduced exactly; a new module `_l3d_e2b.py`).
  - **UAG_c:** UA_c where LEX is non-empty, else LEX. It switches row by row on a gold-free signal, so its counts follow from the v1 arrays (a post-hoc view below, not a run). It gives up row 791 and keeps the dilution at B_N 100.
  - **UFG_c:** CT(LEX), then the candidates of CT(XA_c) not in CT(LEX), then the L1 order, where LEX is non-empty. It targets the dilution. It needs the candidate lists, so it must run.

UAG post-hoc, ALL-gold rows (gained / lost against LEX), PHG:

| B_N | UAG_S | UAG_D | UAG_DS |
|---|---|---|---|
| 100 | 1,902 (+30 / −44) | 1,907 (+23 / −32) | 1,907 (+24 / −33) |
| 250 | 1,982 (+26 / −3) | 1,975 (+19 / −3) | 1,976 (+20 / −3) |
| 500 | 1,987 (+24 / 0) | 1,981 (+18 / 0) | 1,982 (+19 / 0) |
| 1000 | 1,988 (+21 / 0) | 1,983 (+16 / 0) | 1,984 (+17 / 0) |
| 2000 | 1,988 (+18 / 0) | 1,984 (+14 / 0) | 1,985 (+15 / 0) |
| 5000 | 1,988 (+15 / 0) | 1,984 (+11 / 0) | 1,985 (+12 / 0) |

On UNROUTED, UAG_S at B_N 1000 is 1,990 (+24 / 0). From B_N 500 on, UAG loses no row against LEX on PHG or UNROUTED, for any channel (MTK not computed).

- **WebQSP** needs a development population, which is the user's decision (§37.9). Its relation names are encoded, and its maps are done (§37.10).
- **Then:** text L3, then finalize L1 + L2 + L3 on the development rows, then held-out A.

**41.16 Cost and hygiene.**
- **Laptop.** E2 took 307.2 s (loop 295.8 s, peak RSS 904.6 MB) after a 200-row smoke, one heavy process at a time.
- **Host.** The project `crag` ran these jobs:
  - the download (2 CPUs);
  - two CPU probes (the first crashed);
  - REPRO and RELENC (1 GPU and 4 CPUs each);
  - the guard (1e-11 CPU and GB);
  - the E2 placement run (2 CPUs).
- The encoder jobs and the E2 run ran at BELOW_NORMAL. Other projects' jobs, rx's priority map and the capacity reserve were left alone. No credential was placed on the host: the download was anonymous.
- **Transfers.** Code, records, the two archived pointer indexes (216,668 bytes each) and the WebQSP graph manifest were pushed, each as a declared transfer. The model weights were downloaded on the host, not pushed.
- **Hygiene.** Nothing under `data/` was written on either machine. No pinned module or CONTRACT_FILE was edited. No held-out, split-B or TEST row was read: REPRO encoded only the population's 1,998 query rows, parsed from their own lines of `queries/dev.jsonl`.
- **New code:** `_l3d_e2.py` (642d4a71…), `_enc_hfdl.py` (b490b3cf…), `_enc_host.py` (d26727ae…), `_host_yield.py` (b96c2bb8…) and its tests `_host_yield_test_rule.py` (ee6e1776…) and `_host_yield_test_e2e.py` (97657f75…), plus the `rx.toml` push list.


## 42. E2b: a guarded, fill-ordered union of the lexical and the encoder-appended typed walks (2026-09-29 local) (`_l3d_e2b.py`, a generated copy of `_l3d_e2.py` (e1441f3a…); records `results/L3_DEV/l3e2b_metaqa__v1.{json,npz,log}` (json a81a1aae…, npz caf183c6…); STATUS = DEVELOPMENT: descriptive numbers, p-values descriptive, no rule selected, L1 unchanged)

**42.1 What was run.** E2b is the comparison pre-registered in §41.15, on the same 1,998 L1_DEV rows, the same three cells (UNROUTED, PHG_k432, MTK_k432 with the ROWS mask), the same exposure n(q, B_N) = min(B_N, contacted mass) and the same six B_N (100 … 5000).
- **Two switches, four unions** (per channel c in D, S, DS):

| union | how the two candidate lists merge | guard |
|---|---|---|
| UA_c | RRF of CT(LEX) and CT(XA_c), K0 60, ties → FLAT rank (E2 v1's arm) | none |
| UAG_c | as UA_c | UA_c where the lexical schedule matched at least one relation, else CT(LEX) |
| UF_c | fill: CT(LEX) in its own order, then the candidates of CT(XA_c) that are not in CT(LEX), in CT(XA_c)'s own order | none |
| UFG_c | as UF_c | UF_c where the lexical schedule matched at least one relation, else CT(LEX) |

- **Both switches are parameter-free and gold-free.** The guard is the lexical schedule's length, known before any walk. The fill order has no fusion and no constant. Every union serves its list, then the cell's L1 order.
- **Nothing else changed.** The walker is the pinned `_l3d_typed.py`. The schedules are E2 v1's (LEX and XA_c; XT and P are not re-run). No model ran. No row outside the L1_DEV population was read.
- **Identities asserted before anything was written** (all passed): TYPED v1's (§40); and against the E2 v1 record (json and npz sha-pinned), on all 1,998 rows the schedules and on all 14,489 gold nodes the served positions of LEX, XA_c and UA_c in every cell, their candidate gold ranks, list lengths and fetch counts. UAG_c's positions equal, on every gold node, E2 v1's UA_c where the lexical schedule is non-empty and LEX where it is empty. That is the post-hoc view printed in §41.15: its UAG table is reproduced exactly (all three channels, all six B_N).

**42.2 Results** (ALL-gold rows of 1,998; gained / lost against LEX; PHG_k432; the LEX row is the reference).

| arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|
| LEX | 1,916 | 1,959 | 1,963 | 1,967 | 1,970 | 1,973 |
| UA_S | 1,887 (+33 / −62) | 1,966 (+27 / −20) | 1,974 (+25 / −14) | 1,979 (+22 / −10) | 1,981 (+19 / −8) | 1,988 (+16 / −1) |
| UAG_S | 1,902 (+30 / −44) | 1,982 (+26 / −3) | 1,987 (+24 / 0) | 1,988 (+21 / 0) | 1,988 (+18 / 0) | 1,988 (+15 / 0) |
| UF_S | 1,912 (+27 / −31) | 1,968 (+27 / −18) | 1,974 (+25 / −14) | 1,979 (+22 / −10) | 1,981 (+19 / −8) | 1,988 (+16 / −1) |
| UFG_S | 1,927 (+24 / −13) | 1,984 (+26 / −1) | 1,987 (+24 / 0) | 1,988 (+21 / 0) | 1,988 (+18 / 0) | 1,988 (+15 / 0) |
| UFG_D | 1,924 (+19 / −11) | 1,977 (+19 / −1) | 1,981 (+18 / 0) | 1,983 (+16 / 0) | 1,984 (+14 / 0) | 1,984 (+11 / 0) |
| UFG_DS | 1,925 (+20 / −11) | 1,978 (+20 / −1) | 1,982 (+19 / 0) | 1,984 (+17 / 0) | 1,985 (+15 / 0) | 1,985 (+12 / 0) |

- **The guard removes the flood.**
  - From B_N 500 on, UAG and UFG lose no row against LEX, for any channel, on PHG. UFG_S also loses none on UNROUTED and MTK.
  - At B_N 1000, UAG_S and UFG_S serve ALL gold on 1,988 rows (+21 / 0 against LEX, hop 3 +21 / 0, p 9.5e-7). UA_S had 1,979 (+22 / −10, p 0.05).
  - The 10 rows UA_S lost at B_N 1000 are the empty-lexical-schedule rows that E2 v1 traced to the flood (§41.10). Row 791's gain is given up (UAG_S hop 2 0 / 0).
- **The fill order removes most of the dilution and none of the flood.**
  - UF_S at B_N 1000 is UA_S's count exactly (1,979, +22 / −10): the flooded rows are the same, because the fill order does not look at the schedule.
  - At B_N 100, UF_S loses 31 rows against LEX where UA_S loses 62. At B_N 250 and 500 it loses 18 and 14 where UA_S loses 20 and 14.
- **Together (UFG_S), the smallest loss at every B_N.** B_N 100: 1,927 (+24 / −13, p 0.10) against LEX's 1,916. B_N 250: 1,984 (+26 / −1, p 4.2e-7). UAG_S at B_N 100 is 1,902 (+30 / −44, p 0.13), so at the smallest exposure the fill order is what puts the arm above LEX in count.
- **S is the best channel in every union**, as in E2: at B_N 1000, UFG_S 1,988 against UFG_DS 1,984 and UFG_D 1,983. After the guard no channel loses a row against LEX from B_N 500 on.
- **UNROUTED and MTK.**
  - UNROUTED, UFG_S: 1,927 (+24 / −13) @100, 1,985 (+27 / −2) @250, 1,988 (+26 / 0) @500, 1,990 (+24 / 0) @1000, 1,990 (+22 / 0) @2000, 1,992 (+20 / 0) @5000. LEX is 1,916, 1,960, 1,962, 1,966, 1,968, 1,972.
  - MTK, UFG_S: the PHG counts, except 1,985 (+26 / −1) @250.
  - Routing does not bind under ROWS reads (§41.13): for every union arm and channel the routed (PHG) and unrouted counts are within 4 rows at every B_N.

**42.3 The 13 rows UFG_S still loses at B_N 100** (PHG; a diagnostic on the record's arrays, not a new run).
- All 13 are hop 3, and 10 are rows with 11 or more answers. UFG_S gains 24 there (hop 2 +3, hop 3 +21).
- The 14 gold nodes that LEX serves and UFG_S does not were all **in LEX's L1 filler, none in LEX's own candidate list**. LEX's list holds 10–75 nodes in those rows (median 37) and the L1 order fills the rest of the 100. UFG_S's list holds 27–188 nodes (median 77), so the appended walk's extra candidates displace the filler.
- UAG_S's 44 losses at B_N 100 are the other kind: 636 of the 650 lost gold nodes were in LEX's own list (the RRF's dilution), and 14 in the filler. So the fill order fixes the dilution and leaves the filler displacement, and a longer list at equal exposure pushes the filler down by the length it adds.
- No gold-free rule for the remaining 13 was tested.

**42.4 What each arm misses** (B_N 1000, PHG; the classes are §41.8's).

| arm | missed rows | NO_GATE | NOT_REACHED, schedule shorter than hop | NOT_REACHED, schedule at least hop long | CT_TOO_LONG |
|---|---|---|---|---|---|
| LEX | 31 | 3 (hop 2 2, hop 3 1) | 28 (hop 1 1, hop 2 2, hop 3 25) | 0 | 0 |
| UA_S | 19 | 3 (hop 2 2, hop 3 1) | 1 (hop 3 1) | 15 (hop 1 11, hop 2 1, hop 3 3) | 0 |
| UAG_S = UFG_S | 10 | 3 (hop 2 2, hop 3 1) | 2 (hop 2 1, hop 3 1) | 5 (hop 1 1, hop 2 1, hop 3 3) | 0 |

- The 10 rows missed by UAG_S and UFG_S are 3 NO_GATE (outside the gate), 2 with a schedule shorter than the hops, and 5 whose schedule is long enough but whose walk does not reach the gold. No row is TOPIC_MISS, and none is CT_TOO_LONG.
- The other arms' classes are in the record (`misses`).

**42.5 Candidate lists.** The fill order keeps the lexical ranks; the RRF moves them (UNROUTED lists, gold nodes of 14,489).

| list | gold nodes in list | rank < 10 | rank < 100 |
|---|---|---|---|
| CT(LEX) | 13,705 (0.946) | 5,855 | 13,365 |
| UA_S | 14,026 (0.968) | 4,699 | 12,605 |
| UAG_S | 13,872 (0.957) | 4,599 | 12,452 |
| UF_S | 14,026 (0.968) | 5,979 | 13,622 |
| UFG_S | 13,872 (0.957) | 5,879 | 13,469 |

- UF_S and UFG_S have more golds inside rank 10 than CT(LEX) does: rows whose lexical list is empty or short now put the appended walk's golds at rank below 10.
- The guard gives up 154 gold nodes in list (14,026 → 13,872; hop 1 131, hop 2 23): the golds that the appended walk found on rows whose lexical schedule is empty.

**42.6 Cost.**
- **Same walks as E2.** 3.5 walks per gated row (median 3, max 5): LEX, XA_c per channel and the untyped walk. The four unions add no walk, only a merge, so the typed-walk cost of §41.11 stands.
- **List lengths** (gated rows): CT(LEX) mean 310.4 (median 12, p90 531.4, max 9,836); UA_S 434.7 (29.5, 977.8, 11,028); UFG_S and UAG_S 388.3 (25.5, 802.5, 11,028).
- **Fetched nodes per query at B_N 1000** (served outside the contacted shards, PHG): LEX 101.0 (share of served 0.101); UA_S and UF_S 133.5 and 134.0 (0.134); UAG_S and UFG_S 121.5 and 122.0 (0.122). Queries with any fetch: 1,310 (LEX), 1,649 (UA_S), 1,580 (UFG_S). Extra shards per query: mean 48.8, 61.9, 57.3; p95 279.1, 285.0, 285.1; max 325 for the three.
- **Laptop.** 202.1 s (loop 195.8 s, peak RSS 880.4 MB), after a 200-row smoke. Per row: walks 48.5 ms mean (median 39.7), candidate lists 7.4 ms, the base 7.6 ms.
- **Host placement run** (job `l3h-e2b-h1`, 2 CPUs, 3 GB, under the yield wrapper): BIT_IDENTICAL to the laptop run: all 193 npz members (19,403,642 array bytes) are equal and the two npz files are byte-identical (caf183c6…); the json differs nowhere outside its volatile keys (`PLACEMENT_TEST__L3_E2B__v1.json`, 692b5d9f…). It took 225.4 s (loop 208.7 s, peak RSS 869.4 MB), from 21:41:40 to 21:45:28 local, and the wrapper's log holds three events (start, child, done): no yield.

**42.7 Limits.**
- **In-sample.** The guard and the fill order were chosen after E2 v1's losses on these same 1,998 rows: the flood and the dilution are diagnoses of this population. That UAG and UFG lose no row from B_N 500 on is therefore what the rules were built to do here. It is not a confirmation. Nothing held out was read.
- **The guard is lexical.** It is defined by the lexical schedule's emptiness. Whether WebQSP's 7,058 Freebase labels give a lexical schedule that is empty on the same kind of rows is not known: nothing ran on WebQSP.
- **One KB, nine relations.** The gains are 21 rows at B_N 1000 of 1,998 (hop 3), and the encoder never places a relation in the schedule (§41.14).
- **Equal exposure.** At n = min(B_N, contacted mass) a longer candidate list displaces filler (42.3). The counts are at equal exposure, not equal list length.
- **p-values are descriptive.** No rule is selected here. Carrying a KB-path schedule rule forward is a ruling, which this section does not make.

**42.8 What comes next** (§41.15's order stands; nothing held out is read).
- **WebQSP** still needs a development population, which is the user's decision (§37.9). Its relation names are encoded (RELENC) and its maps are done (§37.10).
- **Then:** text L3, then finalize L1 + L2 + L3 on the development rows, then held-out A.

**42.9 Hygiene.** `_l3d_e2b.py` was generated by asserted single-occurrence replacements of `_l3d_e2.py` (the list is in its docstring), so `_l3d_e2.py` (642d4a71…) and every pinned module are untouched. Nothing under `data/` was written; no CONTRACT_FILE was edited; no held-out, split-B or TEST row was read. The host placement was declared in L3_HOST addendum 11 (`results/L3_HOST/HOST_STAGE_DECLARATION__L3_HOST__v1__ADDENDUM_11.json`, fd2eac25…) and the harness joined the host lane in addendum 12 (157d49aa…, after the BIT_IDENTICAL run). The push list in `rx.toml` gained the two addenda, `_l3d_e2b.py` and the E2 v1 json and npz that the harness pins. The 200-row smoke ran outside the repository.


## 43. DATA_FREEZE_VFINAL: the six canonical substrates are closed (2026-09-30 local) (`scratchpad/_data_freeze.py`; records `results/DATA_FREEZE/`; STATUS = FINAL for the six datasets' canonical substrate; nothing under `data/` written)

**43.1 What was done** (the user's ruling of 2026-09-30, in this order): the seven HotpotQA PHG maps were fetched from the host; one manifest was written; a full local SHA-256 pass ran; the declaration followed after all three passed.
- **Fetch.** `rx fetch --glob "results/L1_HOST/parts/hotpotqa__*.npy"` brought K 100, 250, 500, 1000, 2000, 5000 and 52333 (41,866,760 bytes each, 279.5 MB); each is sha-equal to its `RUN.json` `output.sha256` (7 of 7). They were the only host-only canonical artifacts. (The six valid 2Wiki maps were already local; §37.10's earlier "host-only" line for them was wrong and is corrected there.)
- **Full pass.** `python scratchpad/_data_freeze.py MANIFEST` hashed 1,179 files (91.234 GB) in 98.1 s, 4 threads, local reads only. All 991 files of the six trees equal a pin written earlier (the freeze record, `DATASET.json` or an embedding manifest); 0 are first-value only. `CANONICAL_FREEZE.json` recomputes to its own RECORD_SHA256 58958f33… and RECORD_SHA256_LF ed369b48…. `python scratchpad/_data_freeze.py VERIFY --full` then re-read the manifest: 1,221 entries, 0 missing, 0 resized, 0 sha mismatches (52 s).
- **Not run:** `verify_canonical.py`, because it overwrites `data/final_canonical/VERIFICATION.json`. The verifier writes only under `results/DATA_FREEZE/`.

**43.2 The partition cells.** 42 cells: 40 valid (metaqa 6, squad 7, musique 7, hotpotqa 7, 2wiki 6, webqsp 7), each map sha-equal to its `RUN.json` pin, and **2 `PARTITION_INVALID`** with `artifact = null`: 2wiki K 5000 (max block 1,235 > bound 1,234; 45 empty blocks) and metaqa K 5000 (max block 10 > bound 9; 52 empty blocks). By the PHG contract they are absent, never retried or tuned, and part of the freeze state. The per-cell hypergraph `.npz` files are host-side partitioner inputs pinned by sha in each `RUN.json`; they are reconstructible and not part of the retained substrate. `data/l1_canonical/` holds the older Mt-KaHyPar / low-memory maps and hypergraphs only for metaqa, squad and musique.

**43.3 Tree byte totals.** The walked totals exceed the frozen `tree_bytes` by 773–804 bytes per tree (metaqa +773, squad +788, musique +793, hotpotqa +801, 2wiki +804, webqsp +803). The frozen builder measures `tree_bytes` (`freeze_canonical.py` line 342) before it writes `DATASET.json` (line 352), which is consistent with a few hundred bytes of shift; that is a reading, not a proof. Each file is individually pinned, so the total carries no weight; the verifier treats a difference above 4,096 bytes as a problem.

**43.4 The declaration** (`results/DATA_FREEZE/DATA_FREEZE_VFINAL__2026-09-30__v1.json`): corpus cleaning, canonical encodings, graph construction, graph families and canonical partition maps of the six datasets are permanently closed. L2 / L3 work may add derived experiment artifacts (L2 features, L3 relation encodings, caches, results) and may not modify the frozen substrate. A change needs a superseding record on a new user ruling. Text-L3 encodings and the frozen-L2 substrate that exists only for the 2wiki+musique dev pair do not block it.

**43.5 Retained and not deleted.** `data/final_canonical/freebase_v3/canonical/` (37.8 GB) is **kept**: the user's later ruling of 2026-09-30 reversed the earlier deletion authorization, because it is the builder input of the FREEBASE_SCALE lane. The 24 reflog-only blobs are untouched; they are to be listed (sizes, commits, files) as a separate repo-cleanup step. The Freebase served tree (`data/final_canonical/freebase/`, 302M nodes) is pinned by `CANONICAL_FREEZE.json` and was not hashed in this pass (its gate is `verify_freebase.py --full`).

**43.6 What comes next** (the user's ruling): WebQSP KB-L3 transfer first, with a development population of all scoreable non-held-out WebQSP dev rows (only sealed split B, TEST and malformed or unscoreable rows excluded; no random 2,000-row sample), then MuSiQue text-L3; FREEBASE_SCALE is a separate scale lane (systems validation first: graph size, partition feasibility and balance, build memory and time, localization latency, shards contacted, load skew, local scan volume, cross-partition edge rate) that must not delay the WebQSP L3 experiment.

**43.7 Records.** `results/DATA_FREEZE/DATA_FREEZE_2026-09-30__v1.json` (file sha a49efff6…, RECORD_SHA256 5e11ca92…) · `DATA_FREEZE_VERIFICATION_2026-09-30__v1.json` · `DATA_FREEZE_VFINAL__2026-09-30__v1.json` (file sha d83cc1e0…); code `scratchpad/_data_freeze.py`.


## 44. WEBQSP_L3_TRANSFER: the frozen L1 and the typed / encoder-scheduled L3 arms on the 786 WebQSP split-A rows (2026-09-30 local) (`_l3w_transfer.py` (a668cd13…), population `results/L3_DEV/l3w_population_webqsp__v1.json` (7e67c9b1…); records `results/L3_DEV/l3w_webqsp__v1.{json,npz,log}` (json 658bf843…, npz b121f94e…, log 89aaee83…); STATUS = DEVELOPMENT (transfer only): descriptive numbers, p-values descriptive, no rule selected, no arm added or tuned, L1 unchanged)

**44.1 What was run.** The user's ruling of 2026-09-30 (§43.6): WebQSP KB-L3 transfer first, on all scoreable non-held-out dev rows, with no random sample. The MetaQA arms are carried over unchanged; nothing was tuned for WebQSP.
- **Population** (786 rows, 5,606 gold nodes; answers per row: 1 on 401 rows, 2 on 100, 3 on 70, 4 on 38, 5–10 on 89, 11 or more on 88). It is split A of §30: the 1,549 `train_holdout` rows, minus the 46 without a gold node, carved by the parity of sha1(query id), which leaves 786 (A) and 717 (B). Split B and TEST were not read. The 1,549 train rows outside `train_holdout` are **not** in the population (they have no L1 cache and were never a development population); if the user reads the ruling as including them, that is a new population record, and this one is not edited.
- **L1 as carried forward in §37.** IR_L1 on the FLAT dense + SPLADE RRF (K0 60, ACT 200), one-hop localization over the STRUCT / KNN / NER families, ES shard ranking on the native PHG map K 25,928 (cell `PHG_k25928`), B_P = min(K, ⌈KNEE_SDIV(K 100 map) · (K/100)^0.75⌉); the `UNROUTED` cell serves the same order with no shard limit. Served list = the first n = min(B_N, contacted mass) nodes of the order inside the contacted shards; B_N in 100, 250, 500, 1000, 2000, 5000. A row without localised evidence contacts every shard (none here).
- **L3 arms, unchanged:** H2 (§39), LEX (the lexical schedule's typed walk, §40), XT_c / XA_c / P_c and the four unions UA_c / UAG_c / UF_c / UFG_c (§41, §42) for c in D, S, DS. The relation channel scores WebQSP's 7,058 relation label strings (RELENC, `relenc_webqsp__v1.npz` aaec8d03…) against the question with the dense encoder (D), SPLADE (S) or their RRF (DS). The walker is the pinned `_l3d_typed.py` (H = 3, gate G2 = typed graph and anchored, seeds = lexical seeds capped at 5, else dense top-1 + SPLADE top-1).
- **Identities asserted before anything was written (all passed).** The FLAT top-5,000 and the dense / SPLADE top-200 lists of all 786 rows equal `flath2_G_webqsp.npz` (strict; the FLAT products are batch-shape sensitive, so the run uses the record's 200-row batches); the freshly computed dense / SPLADE top-100 lists agree with the cache's top-200 lists at 0.99991 / 1.0 (mean overlap; the floor is 0.9); 21 gated rows' typed and untyped walks equal the pinned `_l1x90_relwalk.py`; 3 rows' prefix orders equal the pinned lexsort. Before the run the harness reproduced on MetaQA (200 rows) 63 (cell, arm) served-position arrays of the E2b / E2 / TYPED v1 records on their 514 gold nodes, with the walk counts, seeds and gate diagnostics equal.
- No model ran. No held-out, split-B or TEST row was read. The 200-row smoke and the identity check ran outside the repository.

**44.2 Results** (ALL-gold rows of 786; PHG_k25928; gained / lost against L1; p-values descriptive, exact sign test).

| arm | B_N 100 | B_N 250 | B_N 500 | B_N 1000 | B_N 2000 | B_N 5000 |
|---|---|---|---|---|---|---|
| L1 | 370 (0.471) | 507 (0.645) | 605 (0.770) | 676 (0.860) | 710 (0.903) | 727 (0.925) |
| H2 | 329 (+3 / −44) | 453 (+5 / −59) | 554 (+9 / −60) | 639 (+10 / −47) | 707 (+17 / −20) | 737 (+17 / −7) |
| LEX | 370 (+1 / −1) | 506 (0 / −1) | 605 (0 / 0) | 676 (0 / 0) | 710 (0 / 0) | 727 (0 / 0) |
| XT_S | 369 (+1 / −2) | 507 (+1 / −1) | 606 (+1 / 0) | 676 (0 / 0) | 710 (0 / 0) | 727 (0 / 0) |
| UFG_S | 370 (+1 / −1) | 506 (0 / −1) | 605 (0 / 0) | 676 (0 / 0) | 710 (0 / 0) | 727 (0 / 0) |
| P_D | 415 (+48 / −3) | 532 (+27 / −2) | 620 (+16 / −1) | 677 (+2 / −1) | 710 (0 / 0) | 727 (0 / 0) |
| P_S | 414 (+47 / −3) | 535 (+30 / −2) | 621 (+16 / 0) | 679 (+3 / 0) | 713 (+3 / 0) | 729 (+2 / 0) |
| P_DS | 413 (+46 / −3) | 533 (+28 / −2) | 618 (+14 / −1) | 678 (+3 / −1) | 711 (+2 / −1) | 727 (+1 / −1) |

- **The lexical-schedule arms are L1.** Every LEX, XT_c, XA_c, UA_c, UAG_c, UF_c and UFG_c count is within 1 row of L1 at every B_N in both cells (XT_S 369 / 507 / 606 / 676 / 710 / 727; the other 18 arms are within the same band). Their paired counts are at most +2 / −2, with p = 1.0 in every one of the 228 comparisons (19 arms, 2 cells, 6 B_N). §44.3 says why.
- **P_c is the only L3 arm that moves L1, and only at the small exposures.** P_S against L1: +47 / −3 at B_N 100 (p 3.7e-11), +30 / −2 at 250 (p 2.5e-7), +16 / 0 at 500 (p 3.1e-5), then +3 / 0, +3 / 0, +2 / 0 (p 0.25, 0.25, 0.50). P_D and P_DS are within 3 rows of P_S at every B_N; S is the best channel at every B_N from 250, D at 100 (415 against 414). P_S loses no row against L1 from B_N 500 on.
- **H2 loses up to B_N 1000 and gains only at 5,000.** Against L1, +3 / −44 (p 2.5e-10) at B_N 100 through +10 / −47 (p 7.5e-7) at 1000; at 2,000 707 (+17 / −20, p 0.74); at 5,000 737 (+17 / −7, p 0.064; ANY 780 against 769).
- **Gold nodes served** (of 5,606). L1: 1,376 (0.245) @100, 2,576 (0.460) @250, 4,495 (0.802) @1000, 5,153 (0.919) @5000. P_S: 1,605 (0.286), 2,685 (0.479), 4,504 (0.803), 5,155 (0.920). H2: 1,124 (0.200) @100, 5,029 (0.897) @5000. P_S against LEX: +249 / −18 gold nodes at B_N 100, +124 / −12 at 250, +45 / −7 at 500, +9 / 0 at 1000.
- **UNROUTED** (counts): L1 370, 507, 602, 668, 705, 730; P_S 414, 535, 620, 671, 708, 732; H2 329, 453, 554, 638, 702, 738. Routing does not bind: the routed L1 is at most 8 rows from the unrouted one (0, 0, +3, +8, +5, −3), and every arm's routed and unrouted counts are within 8 rows at every B_N.
- **What is left at B_N 1000.** L1 and LEX miss 110 rows (0.140); P_S 107 (0.136). At 5,000: L1 59 (0.075), P_S 57 (0.073).

**44.3 Why LEX, XT, XA and the unions equal L1: on WebQSP the lexical schedule is longer than the walk.** The lexical rule (`SE.schedule3`, `key = "wh"`, from the L1_P90 lane) schedules every relation whose label shares a token with a question token outside the seed mention, ordered by attachment, side and distance. WebQSP's 7,058 labels are verbalised Freebase strings, so the list is long.
- **Length** (all 786 rows): 0 relations on 71 rows, 1 on 12, 2 on 1, 3 on 10, 4–7 on 44, **8 or more on 648 (0.824)**. Mean 83.8, median 58, p90 217.5, max 706. More than H = 3 relations on **692 rows (0.880)**, and on **595 of the 687 gated rows (0.866)**. The MetaQA schedules had at most the 9 relations of that KB (§41).
- **Consequence.** With H = 3 the walk stops before the schedule's last relation, which is the answer channel. The candidate list is then the seeds alone: LEX's typed candidate list has mean 1.49 nodes (median 1, max 5) on gated rows and holds **10 of 5,606 gold nodes** (0.002). XT_c and XA_c append one relation to a schedule that is already too long (17–31 gold nodes in list); UAG_c and UFG_c fall back to LEX where the schedule is empty (the guard, gate and lexical schedule of at least 1 relation, holds on 618 rows), which changes nothing when LEX is L1.
- **What the miss classes say** (§41.8's classes; B_N 1000, LEX): 110 missed rows = 19 NO_GATE, 10 NOT_REACHED with a schedule shorter than H, 81 NOT_REACHED with a schedule at least H long, 0 CT_TOO_LONG. The last class is the collapse.
- **Reading, not a proof.** The schedule's length is a property of this rule on this label vocabulary. The comparison that §42 pre-registered (guard and fill on the lexical union) cannot be tested on WebQSP: there is no schedule to guard. No WebQSP-specific rule (a schedule capped at H, or H set to the schedule's length) was run; that would be new development.

**44.4 Where P_c's gain sits.** P_c is the encoder channel's single top relation as a one-relation schedule (one typed hop from the seeds); it is gold-free.
- **Rows** (P_S against L1, B_N 100; rows in each stratum in brackets). Gated +47 / −3 (687 rows), not gated 0 / 0 (99). Lexical seeds +44 / −2 (644), fallback seeds +3 / −1 (142). Topic entity among the seeds **+45 / −1** (415 rows, 0.528), topic not among the seeds +2 / −2 (371). By answers: 1 answer +17 / −1 (p 1.4e-4; 401 rows), 2 answers +6 / −1, 3 +5 / 0, 4 +4 / 0, 5–10 +9 / −1 (p 0.021), 11 or more +6 / 0 (p 0.031). At B_N 250 the topic-in-seeds stratum is +28 / −1 and the other +2 / −1. (The topic entity is the dataset's annotation, used as a stratum only, never as a rule input.)
- **What it does.** P_S's candidate list holds 389 of the 5,606 gold nodes (0.069; 191 at rank below 10, 387 below 100); P_D 356 (0.064), P_DS 353 (0.063). The list is ordered by (−M, FLAT rank), so the gain is placement: golds that the one-hop walk reaches are served inside the first B_N nodes, ahead of L1's fill. The list is heavy-tailed: mean 143.5 nodes on gated rows, median 2, p90 13, max 55,597 (a hub relation).
- **It fades with the exposure**, because L1 already serves the gold: at B_N 1000 L1 serves ALL gold on 676 rows, and P_S adds 3.
- **§41 for comparison.** XT and P lost on MetaQA there. This is a different KB, a different vocabulary and a different arm role (P re-ranks a served list here; there the lexical schedule was already shorter than the hops); both readings are descriptive.

**44.5 What each arm misses** (PHG_k25928; §41.8's classes; missed rows of 786).

| arm | B_N | missed rows | NO_GATE | NOT_REACHED, schedule shorter than H | NOT_REACHED, schedule at least H long | CT_TOO_LONG |
|---|---|---|---|---|---|---|
| LEX | 100 | 416 | 50 | 51 | 315 | 0 |
| LEX | 1000 | 110 | 19 | 10 | 81 | 0 |
| LEX | 5000 | 59 | 9 | 7 | 43 | 0 |
| P_S | 100 | 372 | 50 | 321 | 0 | 1 |
| P_S | 1000 | 107 | 19 | 88 | 0 | 0 |
| P_S | 5000 | 57 | 9 | 48 | 0 | 0 |

- P_S's schedule has one relation, so every gated miss of P_S is "shorter than H" by construction: the class says the walk did not reach the gold, not why.
- **Reach** (PHG_k25928, 786 rows). B_P: mean 1,407.0, median 1,293, p10 711, p90 2,262, min 130, max 4,394 of 25,928 shards. Shards needed to hold every gold in ES order: median 42, mean 1,161.4, p90 492, max 25,545. **55 rows (0.070) have B_P below that number** (lost to reach); 731 do not. Contacted mass: mean 143,290 nodes, p10 72,764, p90 231,806. No row contacts every shard.

**44.6 Cost** (laptop; the walks are the cost).
- **Walks per gated row:** mean 6.0 (median 6, p10 4, p90 8, max 8), against 3.5 on MetaQA (§42.6): the arms' distinct schedules (LEX, three XT, three XA, three P) plus the untyped walk, de-duplicated by schedule.
- **Adjacency rows read per gated row over all its walks:** mean 140,697 (median 161,030, p10 2,090, p90 261,135, max 340,377), of which **130,746 (0.929)** lie outside the contacted shards. No read budget was applied here (§40's budget arms were not part of the transfer).
- **Per row, ms** (mean / median / p95): base IR_L1 117.9 / 84.6 / 273.7; B100 + ES + contacted set 5.3 / 2.2 / 17.1; seeds + lexical + encoder schedules 180.9 / 135.3 / 441.1; H2 313.3 / 223.4 / 704.0; walks 4,759.7 / 4,058.6 / 10,150.2; candidate lists + arm orders 435.0 / 306.7 / 939.7. The FLAT products cost 695.5 ms per row amortised, and the FLAT RRF 2,087.6.
- **Fetched nodes per query** (served outside the contacted shards; P_S): 2.84 (share of served 0.028) @100, 10.35 (0.010) @1000, 27.79 (0.006) @5000; queries with any fetch 235, 237, 238 of 786. H2 at B_N 1000: 257.4 nodes per query (0.257), on all 786 queries; its candidate list holds a mean 381,042 nodes (409,045 UNROUTED).
- **Run:** 7,225 s (loop 6,987.5 s, peak RSS 7,740.6 MB) on the laptop, after the 200-row smoke.

**44.7 Limits.**
- **Transfer only, on the development population.** No rule was selected and no arm was added or tuned; the arms are fixed by §39–§42. That is why the lexical arms are not made to work here. Nothing held out was read.
- **A different comparison from MetaQA's.** With the lexical schedule collapsed, WebQSP tests L1 against a one-hop, encoder-chosen typed re-ranker (P_c) and against the untyped H2. It does not test §42's guard and fill.
- **WebQSP is a name-only KB** (§37 / the encoding ruling), so its L1 numbers are on the frozen K 25,928 map and are not comparable with the legacy KB numbers.
- **Equal exposure.** Counts are at n = min(B_N, contacted mass), not at equal candidate-list length; the P lists are long and heavy-tailed (§44.4).
- **The topic strata use the annotation** and are diagnostics of where the gain sits; no rule reads them.
- **p-values are descriptive.** Carrying any WebQSP arm forward is a ruling, which this section does not make.

**44.8 What comes next** (nothing held out is read).
- **The decision this result puts to the user.** The frozen lexical schedule does not transfer to WebQSP (§44.3), and a single encoder-chosen relation hop helps at B_N 100–500 (+47 / −3, +30 / −2, +16 / 0 against L1) and adds at most 3 rows from 1000. Two ways on: (a) stop WebQSP here and move to MuSiQue text-L3 as planned; (b) a WebQSP development round for a schedule that fits the walk (for example the H best-ranked relations, or H set to the schedule's length), which is new development and would be pre-registered first. Neither is started.
- **MuSiQue text-L3** is next in the ruled order. **FREEBASE_SCALE** stays a separate lane and did not delay this run; its tree transfer to the host ran alongside and passed full verification (§45).

**44.9 Hygiene.** `_l3w_transfer.py` is a new module; the 18 pinned modules it imports (`_l1d_{lib,arms,edgediag,route,kbres,node1h,adaptbp,kscale,scale}.py`, `_l3d_{hop,typed,e2,e2b}.py`, `_l1x90_{core,relwalk,seeds,typed,universal}.py`) are untouched and pinned by sha in the record (`code`), with `relenc_webqsp__v1.{json,npz}`, the lane code snapshot, both PHG maps and their run records, and the population record. The harness asserts its own and the imported modules' shas again before writing, so a change during the run would have aborted it. Nothing under `data/` was written; no CONTRACT_FILE was edited; no held-out, split-B or TEST row was read. The disk-backed product matrices of the FLAT step lived under `%TEMP%` and are not records.


## 45. FBX_HOST: the Freebase served tree on the GPU host, verified byte for byte (2026-09-30 local) (`scratchpad/_fbx_{stage,hf_upload,hf_urls,hf_pull,verify_host,declare}.py`; declaration `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_HOST__v1.json` (5a2e772f…); record `results/FREEBASE_SCALE/FBX_HOST_VERIFY__v1.json` (d8ff06fd…); STATUS = TRANSFER VERIFIED, no FREEBASE_SCALE compute run yet; nothing under `data/final_canonical` written on the laptop)

**45.1 Why and how.** The laptop-to-host path is a DERP relay (`rx doctor --speed` 868 KB/s up, about 25 h for 79.2 GB). The user chose a private Hugging Face dataset repo as the intermediary and the full served tree as the scope. The laptop uploaded the tree (hard links under `data/_cache/fbx_stage`, so no extra disk and the frozen tree gained no cache folder): 120 of 120 files, 79.223 GB, **10,134 s** (7.8 MB/s average). The laptop then resolved per-file signed CDN links (about 1 h lifetime; the host holds **no token**), and the host pulled them with the standard library: 121 manifest entries (the 120 tree files and the hub's own 2,504-byte `.gitattributes`, deleted on the host afterwards because the verifier fails on any unpinned file), 79.223 GB in **979 s at 80.9 MB/s**, about 93 times the relay's speed. The pull script had first been tried on the laptop: two files came back sha-identical to the frozen copies.

**45.2 Verification.** On the host, 120 files and 79,223,455,868 bytes (equal to the source's), no `.part` or `.tmp` leftovers, under a directory junction `ws/data/final_canonical/freebase` → `C:/Users/Student2/rx/projects/crag/data/freebase`. `_fbx_verify_host.py` ran `verify_freebase(freeze, full=True)` under the yield wrapper (2 CPUs, 10 GB, job `260930-043330-fbx-verify-98ed`): **PASS, 156 checks, 0 failed, 672 s.** That is 119 pin checks (every pinned file hashed against `CANONICAL_FREEZE.json` / `DATASET.json`), no unpinned file in the tree, 301,977,131 nodes in the three index arrays, the kind and name census, the relation table, both CSRs (every block), the metadata and type tables, the bridge shapes and the loader smoke. One check did not run its full form: the bridge re-derivation is skipped when `PYTHONHASHSEED` is not 0 or the host has no WebQSP tree (recorded as a pass with that note). The frozen tree's own `VERIFICATION.json` was not touched; the run wrote a new record.

**45.3 State and what is the user's.** The host now holds the Freebase tree read-only for the FREEBASE_SCALE systems-validation lane (§43.6), which is not started. The laptop's hard-link stage was removed. The private repo `Swastik9895/crag-freebase-tree` still exists: whether to delete it is the user's call, and the token that was pasted in chat (never used) should be revoked. The declaration's priority rules held: the jobs ran under `_host_yield.py` and no other project's job or process was touched. `rx.toml`'s push list gained the declaration, `_fbx_hf_pull.py`, `_fbx_verify_host.py`, the link manifest and the three verifier modules.


## 46. FBX_SCALE: the frozen partition and localisation recipes measured against the 302M-node Freebase graph (2026-09-30 local) (`scratchpad/_fbx_{stats,feasibility,encode_feasibility,scale_declare}.py`; declaration `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1.json` (99624a10…); records `FBX_GRAPH_STATS__v1.json` (09dbfc11…), `FBX_PARTITION_FEASIBILITY__v1.json` (17ee5b06…), `FBX_LOCALISATION_FEASIBILITY__v1.json` (5d063bf6…); STATUS = DEVELOPMENT (systems): statistics measured, nothing partitioned, embedded or routed; the lane is STOPPED FOR A RULING (§46.6); nothing under `data/final_canonical` written)

**46.1 What was asked and what was run.** The user ruled (2026-09-30) that FREEBASE_SCALE is finished before MuSiQue text-L3: run the intended large-K partitioning, record balance, cut, build time, memory and block sizes, run the frozen non-learning L1 router (IR_L1 localisation → ES routing → KNEE_SDIV) against that partition map if the pinned workload can be replayed, report B_P and B_P/K, measure latency and bytes touched, and tune nothing. A declaration (write-once) fixes the lane's refusals. Stage 1 is a read-only streaming pass over the served CSR on the host (128 chunks of about 32 million edge slots, 8 workers, 113.6 s, under the yield wrapper); its chunk code equals brute-force Python sets on nodes [0, 3,000) (undirected degree sum 23,675, identical top hubs). Stage 2 and 2b are calculations from existing records; no host job could exhaust the shared VM.

**46.2 The graph (stage 1, exact).** 301,977,131 nodes and 2,062,430,072 directed edges as served; 55,785 self loops; 2,062,374,287 non-self directed edges collapse to **1,975,803,843 unique undirected pairs** (the STRUCT key set of the six datasets; multiplicity 1.04). 301,926,705 nodes have an undirected neighbour (50,426 are isolated); 180,693,966 nodes (0.598) have no out-edge and 6,985,654 (0.023) no in-edge. Mean undirected degree 13.1; the typical node is tiny and a few are enormous: undirected p50 1, p90 27, p99 48, p99.9 126 (out p50 0 / p99 47, in p50 1 / p99 13), yet **195 nodes above 1,000,000 neighbours hold 976,671,952 of the 3,951,607,686 degree endpoints (0.247)**, 1,480 above 100,000 hold 0.354, and 421,106 above 100 hold 0.424. The largest are schema and literal nodes, not entities (names read from the served node shards): "Topic" (47,552,470 out-edges), "notable_for" (30,870,098 out), "/type/object/type" (27,519,877 in), two "Musical Recording" nodes (32,664,694 in; 11,194,964 out), "Non-Agent", "Abstract", "Release track", and language-tagged literals: "Skladba"@sk (11,425,660 in) and "Музыкальная дорожка"@ru, "Pista musical"@ca and "Musiikkikappale"@fi (10,888,179 in each). Two relations carry most edges: `common.notable_for.display_name` **1,151,814,885 (0.558)** and the type-mirror `type.type.instance` 254,882,143 (0.124, the figure the freeze recorded), together 1,406,697,028 (0.682); 780,200 of the 784,928 relations have an edge.

**46.3 The frozen H4 input (stage 1).** H4_SPLIT_PRESERVE makes one hyperedge per anchor (the closed neighbourhood), so STRUCT-only pins = anchors + 2F = **4,253,534,391** (17.0 GB as int32 indices; 46.6 times the largest cell measured so far, 2Wiki's 91,353,972). The split rule chops any hyperedge above cap = round(N/K): at K = 1,000 that is 622 anchors holding 1,245,910,871 pins (0.293), at K = 5,000 2,007 anchors and 1,437,393,901 pins (0.338), at K = 20,000 4,866 and 1,513,718,798 (0.356), at the frozen native K = 3,019,771 (cap 100) 427,773 anchors and 1,677,473,778 (0.394). The order of an oversize anchor's members is fixed by |N(u) ∩ N(v)| for each pin, so a third of the pins need a neighbourhood intersection that is not measured here. No KNN family exists (no vectors for 302M nodes, §46.5), so the graph would be H4_S, not the frozen H4_SK.

**46.4 Feasibility of the frozen PHG contract (stage 2).** From the 40 measured host cells (0.53M to 91.4M pins, NP = 4): the 20 cells above 25M pins use **94.2 to 140.9 bytes per pin** (summed over the 4 ranks; the largest rank holds 0.287 of the sum). At 4,253,534,391 pins that is **401 to 599 GB in total, 115 to 172 GB per rank**; a power-law fit gives 337 GB (all cells, exponent 0.91) and 532 GB (the 20 big cells, exponent 1.00). The shared WSL VM has 94.3 GB, of which 81.6 GB were available at 10:05 local (Windows RAM 127.7 GB, 46.9 GB in use by other users): the prediction is **4.9 to 7.3 times what is available**. Time, if the memory existed: 8.4 h (linear in pins from 2Wiki K2000) to 30.3 to 46.0 h (fits, exponents 1.28 and 1.40). Recorded but unmeasured: the intersection ordering above; the empty-block repair (7 s per move at N = 2.6M); and the frozen builder's whole-STRUCT int64 adjacency (3,951,607,686 entries, 31.6 GB) before the hypergraph is written. **Verdict FBX_PHG_NP4_PREDICTED_INFEASIBLE_ON_HOST.** It was not attempted (a launch above the VM would harm other users' jobs) and NP was not changed; this is a prediction from 40 cells, not an out-of-memory observation.

**46.5 Feasibility of the frozen localisation (stage 2b).** IR_L1 needs the frozen dense (fp16, 1536-d) and SPLADE vectors of every node. On the host GPU the canonical encoders ran at 482 dense and 1,817 SPLADE documents per second (REPRO, §40; shared GPU, BELOW_NORMAL). For 301,977,131 nodes: dense **927.7 GB** (2.40 times the host's 386.8 GB free) and 174.0 h (7.3 days); SPLADE 12.32 billion non-zeros, about 73.9 GB, 46.2 h. **Verdict FBX_IRL1_FULL_REPLAY_INFEASIBLE_ON_HOST**: the frozen router cannot localise on the full graph. What can be replayed is the WebQSP corpus's own frozen localisation (its retrieval cache over 2,592,894 nodes) mapped to Freebase positions through `bridge/webqsp_positions.npy` (1,618,950 of the 2,592,894 WebQSP nodes have a position, 0.624; 973,944 have none); that measures where a query's localised and gold nodes land, not how well the router localises on 302M nodes.

**46.6 What this leaves, and the decision (the user's).** No thousands-of-shards partition of Freebase exists, so B_P, B_P/K, balance, cut, skew, latency and bytes touched are not measured; the stage-1 statistics and the two verdicts are the lane's results so far. The standing rule is that the partitioner is not swapped silently. Three ways forward: **(A, recommended)** rule a lower-memory partitioner for this lane only: a deterministic label-propagation partitioner on the STRUCT CSR (no hypergraph; O(N) memory), the same balance bound (largest block at most ceil(1.03 N/K)), gated by a calibration on WebQSP where the seven PHG maps exist (the partition-objective proxy and the frozen router on the fixed 786-row population, the criterion ruled on 2026-09-14) and only then run on Freebase at K in {1,000, 2,000, 5,000, 10,000, 20,000} against a hash-placement control, with the workload described in §46.5 and the label-free measures (blocks touched by a closed neighbourhood, cut rate, load skew); **(B)** close the lane here as a measured negative for the frozen recipes (about 0.4 to 0.6 TB RAM for the frozen PHG, about 0.93 TB disk and 7 days of GPU for the frozen vectors) and continue with MuSiQue text-L3; **(C)** run the frozen recipe on a machine with at least 0.7 TB of RAM, which this session cannot provide. Nothing is tuned on Freebase under any of them.


## 47. FBX_SCALE stage 3a: which cheap partitioner preserves the frozen router's recall? The WebQSP calibration of the user's ladder (2026-09-30 local) (`scratchpad/_fbx_{part,lp,cn,calib,calib_cn,lp_maps,cn_maps,scale_build,scale_build_lp,scale_addendum1,scale_addendum2,scale_addendum3}.py`, C twins `_fbx_{ldg,lp,cn}.c`; addenda 1–3 `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_{1,2,3}.json` (1eaf5f66…, 7b938de8…, 23b186b9…); records `CALIB_MAPS_WEBQSP__{v1,lp_v1,cn_v1}.json`, `CALIB_EVAL_WEBQSP__{v1,cn_v1}.{json,npz}` (v1 json 5e037a1c…, cn_v1 json dccb5b1b…); STATUS = DEVELOPMENT (calibration of a systems substrate): every cheap rung and one designed variant FAILED the gate fixed before scoring; no Freebase partition built; the fork is the user's)

**47.1 Question and the fixed gate.** The user ruled (2026-09-30, addendum 1) that the partitioner for the Freebase lane is the cheapest one that preserves recall, chosen by a WebQSP calibration against the seven PHG maps (H4_SK, NP 4) with the frozen router on the 786 split-A rows, ladder hash → streaming LDG/FENNEL → balanced label propagation → two-level PHG, early stop at the first recall-preserving rung, "roughly 1 point" = preserving, "3–5 points lower" = keep searching. The gate fixed before any candidate was scored: cells K ∈ {100, 250, 500}, graph S (STRUCT only, the deployable arm), every B_N ∈ {100, 250, 500, 1000, 2000, 5000}, ALL-gold count at the PHG cell's own B_P per row (matched B_P); PASS = at most 7 rows below PHG in every cell, FAIL = 24 or more rows below in any cell (or B_P = K on over half the rows), else BORDERLINE (to the user). SK (STRUCT ∪ KNN) is diagnostic. Everything but the node → block table is the frozen IR_L1 (FLAT dense + SPLADE RRF → one-hop localisation → ES routing → KNEE_SDIV with the K = 100 map of the SAME partitioner and the α 0.75 table).

**47.2 Machinery and its checks.** The WebQSP inputs (6.04 GB, 121 files) moved to the host through the private HF repo (1,010 s up, 106 s down; 152 input files sha-identical laptop and host). A BUNDLE run computed the frozen FLAT + L1a arithmetic once per row (786 rows, 35.6 min; FLAT identity 786 / 786 on the full run, 200 / 200 on a 200-row run; an 8-row smoke shows 0 / 8 because the batch product shape decides the near-tie order, not a bug) and cached the full L1 order and each gold's position. The v1 EVAL re-derives L1 ALL on PHG_k25928 = 370 / 507 / 605 / 676 / 710 / 727 and UNROUTED 370 / 507 / 602 / 668 / 705 / 730 and equals the §44 record on B_P, contacted mass, LO and both position arrays for all 786 rows before it writes anything (PASS). The C twins (`_fbx_ldg.c`, `_fbx_lp.c`, `_fbx_cn.c`; the host env has no numba) are each bit-identical to a plain-Python reference on random graphs with one and with two adjacency streams (TESTC, TESTLP, TESTCN). A 200-row EVAL smoke (`CALIB_EVAL_WEBQSP__smoke200.*`, a472b55f…) was read before addendum 2 and is disclosed there.

**47.3 The verdicts (max rows below PHG at matched B_P over the gate cells; 786 rows).** **R1 balanced hash FAIL (411; 341–411 at every gate cell); R2 LDG on S FAIL (181; 181 / 164 / 135 at K 100 / 250 / 500, diagnostic LDG_SK 155); R3 balanced LP on S FAIL (156; 156 / 150 / 115, LP_SK 138); R3b closed-neighbourhood LP on S FAIL (128; 128 / 127 / 99, CN_SK 96 / 94 / 45).** The loss is not a point failure: it is near zero at B_N 100 (CN_S K500: +1) and grows along B_N to its maximum at B_N 5000 in every gate cell, so the missing golds are the ones that sit in blocks the cheap maps do not co-locate. ALL at B_N 5000 (of 786), PHG / R3 / R3b on S at matched B_P: K 100 586 / 430 / 458; K 250 634 / 484 / 507; K 500 660 / 545 / 561; K 1000 672 / 598 / 598. Paired at CN_S K 100, B_N 5000: 169 rows lost, 41 gained (net −128). No map has B_P = K on any row. Every gate-cell loss is far beyond the "3–5 points" the user named (128 rows of 786 = 16.3 points at K 100).

**47.4 The finding: recall tracks closed-neighbourhood connectivity, not edge cut.** The frozen H4 hypergraph makes one hyperedge per anchor (its closed neighbourhood); its partitioner minimises the number of blocks that hyperedge spans. Mean distinct blocks of a closed neighbourhood on S: K 100 PHG 1.943 < CN 2.150 < LP 2.295 < LDG 2.493 < hash 4.105; K 500 2.245 < 2.432 < 2.572 < 2.792 < 4.610; K 1000 2.421 < 2.578 < 2.711 < 2.923 < 4.769. That order is the recall order at every K. The edge cut does not order it: at K ≥ 500 LP's cut is LOWER than PHG's (K 1000 0.596 vs 0.646) and its ALL@5000 is 598 vs 672. The same order holds for the diagnostic SK arm. Edge-cut objectives (LDG, LP) therefore optimise the wrong quantity for this router, which agrees with the user's instruction not to optimise edge cut alone.

**47.5 R3b: the closed-neighbourhood-aware refinement, and why it was still not enough.** Designed after reading §47.3–47.4 (addendum 3 discloses this, and that it is the fourth partitioner scored on the same rows). It refines the R3 map by moving a vertex only when that strictly lowers KM1 = Σ_u (λ(e(u)) − 1), with exact per-anchor histograms (T = 4 candidate blocks by neighbour count, hubs above 5,000 entries skipped, ≤ 10 sweeps, cap 1.03 / floor 0.97). It does what it was built to do (KM1 −11 % at K 100, 3,357,011 → 2,980,955 on S; distinct blocks 2.295 → 2.150) and buys back 28 of LP's 156 rows, but it closes only about 41–46 % of the structural gap to PHG (K 100: 0.145 of the 0.352 distinct blocks between LP 2.295 and PHG 1.943; K 1000: 0.133 of 0.290) and 18 % of the worst gate loss (156 → 128). Own-B_P recall is closer (CN_S K 500 ALL@5000 626 vs PHG 660) but only because it contacts 32 % more of the corpus (B_P/K 0.194 vs 0.146); that is not the gate.

**47.6 Cost of the cheap rungs (WebQSP, N = 2,592,894, one thread).** Hash 1.1–1.3 s; LDG 0.8–0.9 s; LP 1.0–1.3 s on top; R3b 4.1–7.2 s on top (5–7 sweeps; C peak RSS about 0.2 GB including the memory-mapped adjacency, pool 88–100 MB). At Freebase scale R3b's exact-histogram pool is estimated at about 35 GB plus 5 GB of tables (to be measured, not assumed), LDG/LP at O(N) memory. All are orders of magnitude below PHG's 401–599 GB (§46.4). What they do not have is the recall.

**47.7 What this leaves, and the fork (the user's).** By addendum 1 (fail → next rung or the user) and addendum 3 (no further variant without a ruling) the calibration is closed on its own terms and no Freebase partition has been built. The remaining options are: **(A)** run the systems measurement on Freebase with the best cheap partitioner (R3b or LP) and hash as the structural control, stated plainly as a recall-degraded partitioner whose WebQSP calibration verdict is FAIL (−99 to −128 rows of 786 at the gate cells); it answers the systems questions (build time, RAM, balance, cut, B_P/K, latency, bytes touched) and none about recall; **(B)** rung 4, but note what §47.4 implies: a coarse level of cheap quality (its closed neighbourhoods span about 2.1–2.3 coarse blocks) bounds the final connectivity, so a two-level scheme needs a genuinely hypergraph-aware coarse level, which is a multilevel PHG on a coarsened or predicate-filtered hypergraph, not a cheap LP followed by per-part PHG; it is days of host compute per K and needs its own ruling on the input graph (two relations carry 68.2 % of the edges, §46.2); **(C)** close the lane as a measured negative for cheap partitioners (recall) and for the frozen PHG (memory), and continue with MuSiQue text-L3. The user's own pending calls are unchanged: the private HF repo `Swastik9895/crag-freebase-tree` (now also holding `webqsp/`, 121 files), the pasted HF token (never used), and the roughly 10.4 GB of legacy weights.


## 48. FBX_SCALE stage 4A: a hypergraph-aware multilevel partitioner (rung 4, ML_S) on WebQSP against the frozen PHG (2026-09-30 local) (`scratchpad/_ml_{agg.c,coarsen.py,run.py,eval.py,addendum4.py}`, `src/l1_lowmem/phg_driver/phg_driver_vw.c`; addendum 4 `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_4.json` (6dd58473…); records `ml/{ML_DRIVER_BUILD,ML_DRIVER_REGRESSION}.json`, `ml/webqsp__ML_S__clusters.RUN.json`, `ml/webqsp__ML_S_k{100,250,500}.{npy,RUN.json}`, `ml/webqsp__ML_S_k1000.FAILED.json`, `CALIB_EVAL_WEBQSP__{phgs_v1,ml_v1}.{json,npz}` (phgs_v1 json 20f50346…, ml_v1 json 87678ae9…, npz 69a009ee…); STATUS = DEVELOPMENT (calibration of a systems substrate): the gate as fixed FAILED (48 rows), the same-graph reading is BORDERLINE (12 rows), so by the pre-declared table it goes to the user; no Freebase partition built)

**48.1 What was asked and what was built.** The user ruled (2026-09-30) for rung 4 as the FINAL recall-preservation attempt, tightly scoped: a hypergraph-aware multilevel prototype on WebQSP only, scored against canonical PHG under the exact existing gate, the canonical structural input untouched (the two relations carrying 68.2 % of the edges kept), "same graph semantics, different scalable partitioning algorithm", the coarse level itself hypergraph-aware (not a cheap LP followed by per-part PHG), no rung 5. Pass → a progressive Freebase build from K = 1000; fail → the lane is closed as "unresolved". ML_S is: the frozen H4_SPLIT_PRESERVE hypergraph on the STRUCT arm (H4_S) → a K-independent vertex clustering (open-twin contraction, then heavy-connectivity agglomeration with rating Σ w_e/(|e|−1) over nets of 2..200 pins, cluster weight ≤ 32, three passes; 57.4 s, 1.43 GB) → the exact quotient hypergraph (single-pin nets dropped, identical nets merged with summed weights, vertex weight = cluster size; the weighted KM1 of the quotient equals the fine weighted KM1 of the projected partition, asserted in every cell) → the frozen Zoltan PHG driver with vertex weights (NP 4, the same 25 parameters; the vertex-weight build reproduces the frozen driver's partition exactly on a unit-weight manifest, fmt 1 and fmt 3, `ML_DRIVER_REGRESSION.json` PASS) → projection → the R3b closed-neighbourhood KM1 refinement (unchanged). Parameters were fixed before the maps were scored and were not touched afterwards. It is the fifth partitioner scored on the same 786 split-A rows.

**48.2 Two things read before scoring (addendum 4 discloses both).** (1) The gate is not attainable by graph S even for the frozen partitioner: PHG on H4_S (the control, `CALIB_EVAL_WEBQSP__phgs_v1`) FAILS the gate against PHG on H4_SK at 59 rows (K 100 17 / 29 / 33 / 46 / 55 / 59 by B_N 100…5000; K 250 0 / 12 / 17 / 28 / 33 / 31; K 500 1 / 7 / 13 / 21 / 26 / 26). The difference between the arms is the KNN half of the input, not the partitioner. ML_S was therefore scored against two references: THE GATE (PHG on SK, exactly addendum 1) and SAME GRAPH (PHG_S at PHG_S's own B_P). (2) Hypergraph-aware coarsening reaches 22.5× on vertices (2,592,894 → 115,105) but only 2.48–2.59× on pins (14.36 M → 5.54 M / 5.63 M / 5.70 M / 5.79 M at K 100 / 250 / 500 / 1000): about 729 k nets of ~7.6 pins remain on 115 k clusters (anchor-net multiplicity), and the last pass stopped because it removed 0.9 % (< 5 %) of the vertices.

**48.3 The verdicts (rows of 786 below the reference at matched B_P; K 100 / 250 / 500, B_N 100 / 250 / 500 / 1000 / 2000 / 5000).** **THE GATE — ML_S vs PHG_SK: FAILED, max 48** (K 100 14 / 29 / 25 / 40 / 47 / 48; K 250 −1 / 10 / 10 / 24 / 32 / 29; K 500 1 / 9 / 16 / 30 / 37 / 33); ALL at B_N 5000 538 / 605 / 627 vs PHG 586 / 634 / 660. **SAME GRAPH — ML_S vs PHG_S: BORDERLINE, max 12** (K 100 0 / 10 / −2 / 2 / 0 / −4; K 250 1 / 4 / 1 / 9 / 12 / 9; K 500 1 / 7 / 7 / 8 / 12 / 8); ALL at B_N 5000 (matched to PHG_S's own B_P) 560 / 605 / 634 vs PHG_S 556 / 614 / 642. Paired counts against PHG_S at B_N 5000 (lost / gained): K 100 55 / 59, K 250 45 / 36, K 500 28 / 20; the descriptive two-sided sign p-values are ≥ 0.039 in all 18 cells (one at 0.039, K 500 B_N 250). Diagnostics: the projection alone (ML_S_proj) FAILS the gate at 56, so the refinement buys 8 rows (56 → 48); the earlier rungs' worst losses against the same PHG were hash 411, LDG 181, LP 156, CN 128 (§47.3). B_P/K (own) is 0.230 / 0.184 / 0.154 for ML_S against 0.232 / 0.185 / 0.156 for PHG_S and 0.218 / 0.174 / 0.146 for PHG; no row has B_P = K in any method. So the multilevel scheme lands on the frozen partitioner's recall on the same graph (within 12 rows, 1.5 points, mixed sign, at a 2.5× pin reduction) and 48 rows below the gate reference because the gate reference has the KNN arm. Its weighted KM1 is above PHG_S's (K 100 276,951,362 vs 263,496,586, +5.1 %; K 250 312,365,571 vs 299,528,103, +4.3 %; K 500 367,008,684 vs 330,759,543, +11.0 %).

**48.4 K = 1000 (diagnostic, outside the gate).** ML_S is PARTITION_INVALID at K = 1000: Zoltan returned imbalance 1.036 on the cluster weights (tolerance 1.03), six blocks came out empty (repaired by the frozen rule) and the projected maximum block is 2,687 against the contract bound 2,671; no map exists and none of it enters a verdict (`ml/webqsp__ML_S_k1000.FAILED.json`, 2f6f81d5…). Cluster granularity (clusters of up to 32 against ~2,593 vertices per block) is the likely cause and is untested; no parameter was changed in response.

**48.5 What Freebase would need (estimates from these measurements; nothing run on Freebase).** The coarse PHG costs 131–133 B per coarse pin (sum of rank peaks 739–762 MB) against 140–149 B per pin for the fine PHG_S (1.92–2.03 GB). Freebase S has 4,253,534,391 pins: a 2.5× reduction (the WebQSP figure, unverified at Freebase scale) leaves about 1.70 B coarse pins, about 223 GB against the 81.6 GB available; fitting the 0.9 × 81.6 GB budget needs a pin reduction of about 7.6×, three times what the coarsener reaches here. The coarsener itself (numpy) peaked at about 100 B per fine pin (1.43 GB on 14.36 M pins), so about 425 GB at Freebase scale as implemented: it would need an out-of-core rewrite before its reduction could even be measured. By addendum 4's feasibility rule the 4B build is therefore predicted infeasible on this host on memory alone, and it was not attempted.

**48.6 The fork (the user's; addendum 4's pre-declared table sends BORDERLINE here, no closure and no Freebase build until the ruling).** The ruling's literal fail clause (gate FAILED at 48 ≥ 24) closes the lane; the same-graph reading (BORDERLINE at 12) does not, and the table declared beforehand that a BORDERLINE goes to the user. Options: **(A) close the lane** as "Freebase recall-preserving low-memory partitioning = unresolved", with the measured conclusion above (cheap O(N) graph partitioners preserve the cut, not the closed-neighbourhood locality — 128–411 rows lost; a hypergraph-aware multilevel scheme recovers the frozen partitioner's recall on the same graph at 2.5× fewer pins but does not shrink the memory enough for Freebase, and its K = 1000 balance fails), then MuSiQue text-L3; **(B)** accept the same-graph reading as the pass and proceed to 4B, whose first step is a structure-only Freebase coarsening that first needs an out-of-core coarsener (a new build, then the K = 1000 partition itself); on the estimates above it is predicted infeasible, so this is a measurement of that prediction rather than a route to a partition; **(C)** the systems-only measurement with the cheapest partitioners and hash as the structural control, stated as recall-degraded. The pending calls of the user are unchanged: the private HF repo `Swastik9895/crag-freebase-tree`, the pasted HF token (never used), the roughly 10.4 GB of legacy weights.

## 49. FBX_SCALE stage 4C: the SK-to-SK partitioner laboratory on WebQSP — recall against pin compression, M1–M4 versus the frozen PHG_SK (2026-09-30 local) (`scratchpad/_ml2_{agg.c,coarsen.py,run.py,cn.c,cn.py,cell.py,eval.py,oracle.py,addendum5.py,launch.py,curve.py}`; addendum 5 `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_5.json` (ddff074c…); records `ml2/ORACLE_PIN_REDUCTION__v1.json` (46d515e0…), `ml2/webqsp__<cand>_k<K>.{npy,_proj.npy,RUN.json}` (100 cells, 36 candidates, none failed), `ml2/webqsp__M0R_k{100,250,500}.RUN.json`, `CALIB_EVAL_WEBQSP__ml2_v1.{json,npz}` (json 82ee5650…), `ML2_COMPRESSION_CURVE__v1.json` (13b4bf82…) and `ml2/ML2_COMPRESSION_CURVE__v1.png`; STATUS = DEVELOPMENT (calibration of a systems substrate): the declared grid is scored, the finish line is NOT reached; no Freebase partition built)

**49.1 What was asked.** The user's two-track ruling (2026-09-30): Track B uses WebQSP as a partitioner laboratory, **SK to SK only** (same nodes, same STRUCT and KNN, balance 1.03, same K, frozen routing and evaluation). M0 = the frozen PHG_SK (the recall target). M1 heavy-connectivity multilevel, M2 closed-neighbourhood-aware coarsening, M3 repeated modest contraction, M4 a deterministic hybrid. Three gates in order: (1) ALL-gold recall against PHG_SK at matched (B_P, B_N) (primary); (2) pin reduction r_pins, where Freebase needs about 7.6x; (3) systems cost among recall-equivalent methods. Several compression levels, recall plotted against pin compression with the predicted Freebase RAM. Finish line: a WebQSP point with r_pins at least the Freebase requirement, recall inside the PHG envelope, balance at most 1.03.

**49.2 What was run.** Addendum 5 (written before any scoring) fixes the grid: the final map of every (method, Wmax in {32, 128, 512, 2048}) pair plus the intermediate levels of the W512 runs with pin reduction of at least 1.9x, 36 candidates, K in {100, 250, 500} (W2048 at K100 only: the attempt rule Wmax <= 0.10 N/K). Every candidate is: cluster the frozen H4_SK hypergraph, quotient it (the weighted KM1 of the coarse partition equals that of its projection, asserted in every cell), partition the vertex-weighted coarse hypergraph with the frozen PHG driver (NP 4), project, refine the boundary with one shared refinement. **M0R** puts the frozen PHG_SK map through the same refinement with no clustering, to separate the refinement from the coarsening. Scoring is `_ml2_eval.py`, one pass (786 rows, 3,400 s), with every PHG cell asserted bit-identical to the earlier `ml_v1` record (PASS, six cells). All 100 cells are balance-valid (maximum block at most ceil(1.03 N/K), every block used, no PARTITION_INVALID).

**49.3 Result.** Verdicts by addendum 1's table (at most 7 of 786 rows below PHG at matched B_P over every B_N = PASS, at least 24 = FAIL): 3 RECALL_PRESERVING (M0R, M1_W512_L2, M2_W512_L2), 3 BORDERLINE (M2_W512_L3, M4_W512_L4, M3_W512_L4), 31 FAILED.

| candidate | pin reduction (K100) | worst rows below PHG (786) | verdict | Freebase RAM at 133.4 B/coarse pin |
|---|---|---|---|---|
| M0R (refinement only) | 1.00x | 1 | RECALL_PRESERVING | 567 GB |
| M1_W512_L2 | 1.971x | 6 | RECALL_PRESERVING | 288 GB |
| M2_W512_L2 | 1.918x | 7 | RECALL_PRESERVING | 296 GB |
| M2_W512_L3 | 2.422x | 9 | BORDERLINE | 234 GB |
| M4_W512_L4 | 1.943x | 15 | BORDERLINE | 292 GB |
| M3_W512_L4 | 1.915x | 23 | BORDERLINE | 296 GB |
| M2_W512_L6 | 3.271x | 29 | FAILED | 173 GB |
| M1_W2048_L7 | 3.670x | 30 | FAILED | 155 GB |
| M2_W2048_L7 (most compression, K100) | 3.885x | 35 (4.45 points of ALL-gold) | FAILED | 146 GB |
| M3_W512_L10 | 2.770x | 109 (13.9 points) | FAILED | 205 GB |

The budget is 73.4 GB (0.9 of 81.6 GB); the compression it requires is 7.73x. The RAM column counts the coarse hypergraph the vertex-weighted PHG must hold, so it is a lower bound (the coarsener's own memory is not in it).

(a) **Recall against compression is monotone and the loss starts at about 2x.** Up to about 1.97x the refinement-equipped candidates stay inside the PHG gate; at 2.4x M2 is already at the BORDERLINE edge (9 rows); from 3.2x the loss is 26 to 35 rows (3.3 to 4.5 points) for M1 and M2, and 10 to 15 points for M3 and M4 at comparable compression. (b) **M2 (closed-neighbourhood-aware) dominates at equal compression**, M1 is next, M4 and M3 are clearly worse, which matches the §47 finding that recall tracks closed-neighbourhood connectivity, not cut. (c) **The refinement is not the cause**: M0R is never more than 1 row below PHG (worst cell: 1 row below at K100 / K250 on the B_N grid; up to 6 rows above at K100), so the loss is the coarsening. (d) **The compression ceiling is structural.** The seven PHG_SK maps read as clusterings give 10.51x at 100 clusters, 6.98x at 250, 4.53x at 1,000, 3.84x at 2,000, 3.29x at 5,000 and 2.72x at 25,928 clusters (blocks of about 100 nodes): the pin reduction is set by the number of clusters V, not by the clustering rule. The best candidate (M2_W2048_L7: V = 1,467 clusters, 3.885x) is on that curve (between 4.53x at 1,000 and 3.84x at 2,000 clusters). 7.73x needs V of about 200 clusters, that is two clusters per K=100 block, where the clustering is already the partition and the coarse partitioner has nothing left to decide; and keeping balance at 1.03 needs clusters of at most about a tenth of a block (the attempt rule), which puts a floor of V = 1,000 at K100, 2,500 at K250 and 5,000 at K500 and so a ceiling of about 4.5x, 3.8x and 3.3x on the reduction. So a better scoring rule inside cluster-and-quotient cannot reach 7.73x. (e) **The finish line is not reached**: the largest recall-preserving point is 1.97x (Freebase about 288 GB), the largest point at all is 3.89x (about 146 GB, recall FAILED at 35 rows), both far from 7.73x and from 73.4 GB.

**49.4 What this decides and what it does not.** It decides that no single-level cluster-and-quotient partitioner on the frozen hypergraph gives Freebase a partition inside this host, by the gate the user fixed. It does not test a different class (for example a hierarchical two-level PHG, or an out-of-core partitioner that keeps the hypergraph on disk); that is the user's call, not a further heuristic search. The out-of-core engine (stream pins, merge proposals, external sort, resolve, rewrite, deduplicate, checkpoint) was not built: the user's own condition for running a candidate at Freebase scale is a WebQSP compression level that preserves PHG recall and is large enough, and none is. No candidate was run on Freebase.

**49.5 Track A state.** STRUCT: exact connected components of the served graph (`FBX_STRUCT_COMPONENTS__v1.json`, 3313d854…): 1,682,366 components, the largest holds 298,652,676 nodes (98.9 %), the second 436, 50,426 singletons; with the existing statistics (`FBX_GRAPH_STATS__v1`), host verification and the localisation feasibility this closes the STRUCT family's statistics. NER: the frozen recipe (spaCy en_core_web_sm) is rebuilt as a sharded extraction (`scratchpad/_fbx_ner_build.py`, unit test PASS) but the host has no spaCy and the laptop has no RAM free; it needs the user's OK to install spaCy and the model into a crag-owned directory on the host. KNN: Freebase has no node embeddings (dense encoding of 302M nodes is 7.3 GPU-days and 927.7 GB), so a KNN family as defined cannot be built without a ruling on the node representation.

## 50. FBX_SCALE Track B: a two-level (nested) PHG on WebQSP against the frozen flat PHG_SK (2026-10-01 local; host) (`scratchpad/_h2l.py` (e578fc4b…), `scratchpad/_h2l_eval.py` (e9d90750…), `scratchpad/_h2l_addendum7.py`; addendum 7 `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_7.json` (a3867580…, written before any H2L arm was partitioned or scored); records `h2l/parts/webqsp__H4_H2LTOP_k{5,25}__PHG_con.{npy,RUN.json}`, `h2l/parts/webqsp__H4_H2L{5,25}_k{100,250,500}__PHG_con.{npy,RUN.json}`, `CALIB_EVAL_WEBQSP__h2l_v1.{json,npz}` (json e08b7c4e…, npz 1af2d87f…); STATUS = DEVELOPMENT (calibration of a systems substrate): both arms BORDERLINE under the pre-declared table, so the decision goes to the user; no Freebase partition built)

**50.1 Question and construction.** The installed Zoltan PHG holds its hypergraph in RAM (measured ~129 B/pin), so "disk-backed Zoltan" does not exist. The one structure that bounds memory by construction is a nested partition: a top S-way PHG on H4_SK at cap N/S (frozen driver, NP 4, IMBALANCE_TOL 1.01, the only declared deviation), then one independent K/S-way PHG per top block on the block's net-split K-way hypergraph (nets keep their pins inside the block, nets left with fewer than 2 pins are dropped, weights unchanged; Zoltan tolerance floor_4dp(1.03 (N/K)(K/S)/n_b), so its own bound equals the contract bound 1.03 N/K). Peak memory is then the largest block's pins. Addendum 7 asked, before anything was scored, whether this keeps ALL-gold recall under addendum 1's gate (K in {100, 250, 500}, matched B_P, every B_N; at most 7 rows below PHG_SK = RECALL_PRESERVING, at least 24 = FAILED, else BORDERLINE). Arms S = 5 and S = 25 (both divide every gate K). The top level stays flat and in memory here (WebQSP fits); making it out-of-core is the top-level question and is not started (it was called addendum 8 when this section was written; addendum 8 became the noise floor of the gate, §51, and the top-level surrogate is addendum 9).

**50.2 Controls and validity.** `_h2l.py TEST` passes (restriction identity and K-way KM1 = top KM1 + sum of sub-problem KM1, exact). CONTROL: the K-way hypergraph of this path has the frozen H4_SK content digest at K = 100, 250 and 500 (EQUAL). Both top maps pass the frozen validity rule (S = 5: largest block 523,764 of bound 534,137; S = 25: 104,752 of 106,828). All six K-way maps pass ceil(1.03 N/K) (largest block = one below the bound: 26,706 / 10,682 / 5,341), no empty block, no repair, nothing refused. The six PHG_SK cells of the reference are bit-identical to `CALIB_EVAL_WEBQSP__ml2_v1` (786412f4…).

**50.3 The gate (addendum 1, unchanged).** Rows below PHG_SK's ALL-gold count at matched B_P, per B_N in {100, 250, 500, 1000, 2000, 5000} (negative = more rows served than PHG):

| arm | K = 100 | K = 250 | K = 500 |
|---|---|---|---|
| H2L5 | 3, −1, −11, −14, −11, −9 | 0, −5, −2, 1, **8**, **8** | 0, −2, 1, −1, 2, 1 |
| H2L25 | 1, −4, −6, −8, −7, −7 | 0, −1, 4, 4, **8**, 7 | 0, 0, 4, 7, 6, 7 |

Verdicts: **H2L5 BORDERLINE (max loss 8), H2L25 BORDERLINE (max loss 8)**; no cell reaches the FAILED line of 24, no arm has B_P = K on any row (B_P/K own: 0.221/0.177/0.149 for H2L5, 0.219/0.175/0.147 for H2L25, PHG 0.218/0.174/0.146). The max loss exceeds the 7-row pass line by one row in each arm: 8 at K = 250, B_N 2000 and 5000 (H2L5) and at K = 250, B_N 2000 (H2L25); H2L25 also sits on the line (7) at K = 250, B_N 5000 and K = 500, B_N 1000 and 5000. Descriptive, not gating: the paired churn is two-sided — gained/lost rows at matched B_P (B_N 5000) are 55/46, 24/32, 18/19 for H2L5 and 40/33, 25/32, 19/26 for H2L25 at K 100/250/500; the descriptive sign-test p-values are at least 0.15 in every cell. The nested map is a different partition with the same recall to within single-digit rows; it is not a bit-for-bit copy and none was expected.

**50.4 KM1 (diagnostic) and what the nesting costs in memory.** K-way KM1 of the combined map against the flat PHG_SK (994.83M / 1127.79M / 1253.82M at K 100/250/500): H2L5 1017.64M (+2.3 %) / 1132.84M (+0.4 %) / 1220.94M (−2.6 %); H2L25 1001.30M (+0.7 %) / 1124.67M (−0.3 %) / 1206.64M (−3.8 %). The sub-problem hypergraphs carry 95.7 % (S = 5) and 90.5 % (S = 25) of the flat pins in total (net splitting drops nets with fewer than 2 pins inside a block). Largest sub-problem: S = 5 6.82M pins (of 29.57M, 4.3x smaller), 266–274 MB per rank at NP 4; S = 25 1.84M pins (16x smaller), 77–89 MB per rank; Zoltan time per sub-problem a few seconds to about 115 s (one 205 s outlier at S = 25, K = 100 on a host shared with other jobs; cause not investigated). Whole-job time, top map excluded: 192–341 s (S = 5), 92–386 s (S = 25). The top level is what is left at full size: 345 s (S = 5) and 248 s (S = 25) at 29.57M pins, peak 1.28 GB per rank.

**50.5 Reading and the fork (the user's; the decision table of addendum 7 sends BORDERLINE here).** What is established: a nested partition with 5 or 25 independent blocks reproduces the flat partition's closed-neighbourhood locality to within 8 rows of 786 at every scored cell (KM1 within +2.3 % of the flat cut at K = 100, better at K = 500), so the K-way stage of an out-of-core partitioner can be made per-block at about 1/S of the memory with no loss that resembles the 48-, 128- or 411-row failures of §47/§48. What is not established: that it meets the gate as written (8 > 7 in both arms). No other S is run after seeing this (addendum 7: no re-run with other S). Options: **(A) accept H2L as the K-way stage on the one-row margin, treating 8 vs 7 as gate noise** (no run-to-run noise floor of the flat PHG was measured, so the one-row margin cannot be judged against one); **(B) hold at BORDERLINE and proceed to addendum 8 only with the top level**, since the top level is where the remaining cost lives; **(C) close the nested path** and stay with a flat in-memory PHG only (needs about 775 GB at Freebase scale, about 390 GB after a 2x M2 coarsening; host 81.6 GB usable): recommended only if you want the gate to bind literally. Recommendation: (B). Memory at Freebase scale for the K-way stage becomes the largest block's pins times the per-pin cost: the WebQSP sub-problems measured 161 B/pin (S = 5) and 193 B/pin (S = 25) summed over the four ranks (the flat job measured 129), and the largest block holds 1.15x (S = 5) and 1.56x (S = 25) of P/S pins. With about 6e9 pins that is about 180 GB at S = 5 and about 60 GB at S = 25 (81.6 GB usable), so S >= 25 is the regime that fits the host; WebQSP's block imbalance need not carry to Freebase, so S >= 40 is the safer read. The top level still needs a coarsened hypergraph (addendum 9; M2 alone cannot provide one at Freebase scale: its pin reduction saturates at about 3.9x on WebQSP).

## 51. FBX_SCALE Track C: is a product-quantised KNN good enough to freeze into the Freebase encoding? OPQ+PQ m = 64 / 128 / 192 on WebQSP against the frozen PHG_SK (2026-10-01 local; host) (`scratchpad/{_pq_knn,_pq_eval,_fbx_enc_probe,_fbx_encode}.py`; addenda 6/6A `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_{6,6A}.json`; records `pq/WEBQSP_{DISTINCT_ROWS,EXACT_KNN,PQ64_KNN,PQ128_KNN,PQ192_KNN}__*.json`, `pq/parts/*`, `CALIB_EVAL_WEBQSP__pq_v1.{json,npz}` (json bb0246d2…, npz 3fdb91d8…), `enc/FBX_ENC_PROBE__v1.json` (c263da35…), `enc/FBX_QWEN_ENCODING_CONTRACT__v1.json` (aef5fb58…); STATUS = DEVELOPMENT (calibration of a systems substrate): the end-to-end gate of addendum 6 accepts m = 64; the Freebase encode is started under that contract)

**51.1 Question and construction.** The user's ruling: Freebase KNN must use the same canonical Qwen encoder with compressed ANN storage, every batch quantised at once and the float batch discarded, and the compression is validated on WebQSP BEFORE any Freebase name is encoded. Arms: OPQ rotation (faiss OPQMatrix(1536, m), 20 training iterations) + ProductQuantizer(1536, m, 8 bit), m in {64, 128, 192} bytes per row (48x / 24x / 16x smaller than fp16), trained on 131,072 distinct rows (RandomState 0), every distinct row encoded. Search is exhaustive decoded-vs-decoded (the deployed regime; IVF/nprobe is a separate, later measurement), then the canonical merge, then the frozen H4_SK hypergraph rule and the frozen Zoltan PHG driver (NP 4) at K in {100, 250, 500}, scored by addendum 1's rule unchanged. The keys-to-hypergraph path reproduces the frozen H4_SK content digest (CONTROL), the canonical exact search recomputes the frozen KNN family, and every PHG cell of the reference reproduces the earlier EVAL bit for bit (all six `equal`).

**51.2 The gate (addendum 1, unchanged).** Rows below PHG_SK's ALL-gold count at matched B_P, per B_N in {100, 250, 500, 1000, 2000, 5000} (negative = more rows served than PHG):

| arm | K = 100 | K = 250 | K = 500 | max loss | verdict |
|---|---|---|---|---|---|
| PQ64 (64 B/row) | 0, 3, −4, −9, −10, −8 | 1, 2, 3, −1, 2, −2 | 1, 0, 2, 3, 5, 6 | 6 | RECALL_PRESERVING |
| PQ128 (128 B/row) | 0, −9, −15, −20, −19, −23 | 1, 0, 2, 3, 7, 5 | 0, 0, 5, 7, 9, 9 | 9 | BORDERLINE |
| PQ192 (192 B/row) | 3, 2, −3, −2, −1, 1 | 1, 1, 0, −2, 1, −1 | 0, −2, −2, −5, −6, −5 | 3 | RECALL_PRESERVING |

All nine cells are scored, none has B_P = K on a row (B_P/K 0.215/0.172/0.145 for PQ64, PHG 0.218/0.174/0.146). The verdict is not monotone in m (PQ128 loses 9 rows at K = 500, PQ64 loses 6): the decision table reads the pre-declared verdicts, not a trend; the same-algorithm replicate floor of addendum 8 (PHG re-run under a shuffled input order or NP 3) is being scored separately and is reported beside these numbers, not used to re-classify an arm.

**51.3 Diagnostics (not gating; addendum 6 forbids neighbour recall from accepting or rejecting an arm).** Reconstruction cosine on 100,000 rows, mean / 5th percentile: PQ64 0.906 / 0.849, PQ128 0.942 / 0.909, PQ192 0.961 / 0.940. Row-wise neighbour recall@3 against the exact search: 0.657 / 0.789 / 0.849; nearest-neighbour recall 0.848 / 0.950 / 0.979. The partition gate passes at m = 64 although only 66 % of the three neighbours survive, which is the measured sense in which the graph's closed-neighbourhood structure, not the individual edge, is what the L1 router consumes.

**51.4 Decision (addendum 6's pre-declared table, applied, not chosen).** The smallest accepted m is 64 bytes per row: PQ64 is RECALL_PRESERVING at K = 100, 250 and 500. PQ128 being BORDERLINE does not bear on the choice (the table reads the SMALLEST passing arm). The Freebase encoding contract is therefore frozen with m = 64, max_seq_length 512 (0 of 16,000 sampled names exceed 433 tokens), batch 64, and written BEFORE the first Freebase vector (`enc/FBX_QWEN_ENCODING_CONTRACT__v1.json`, sha aef5fb58…; it pins this eval record by hash). Measured encode throughput (RTX 4500 Ada, fp16, length-sorted batches): 478 names/s at batch 64, batch-size numerics cos >= 0.9994 between batch 32/64/128/256; 245,350,737 distinct names extrapolate to ~143 GPU hours uncontended. Storage: 245.35M x 64 B = 15.7 GB of PQ codes, against 754 GB for the fp16 matrix.

**51.5 What this does and does not establish.** Established: a 48x-compressed KNN built only from decoded vectors keeps the frozen router's ALL-gold recall within the gate on WebQSP at three partition granularities. Not established: that the same m holds at 245M rows (the WebQSP calibration set has 1.79M distinct rows; the ANN index at Freebase scale is a separate IVF measurement against this exhaustive truth), that the gate transfers to a KB with longer names, or anything about answer quality. The Freebase KNN, the IVF index and the canonical H4_SK on Freebase are not built; NER stays out of the partition hypergraph (user ruling) and is for IR_L1 localisation only.

## 52. FBX_SCALE Track B: the run-to-run floor of the gate — same-algorithm replicates of the frozen flat PHG scored against PHG_SK (2026-10-01 local; host) (`scratchpad/_h2lm_noise.py`, `scratchpad/_h2lm_eval.py`, `scratchpad/_h2lm_stats.py`; addendum 8 `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_8.json` (47838cec…, written before either replicate was partitioned or scored); records `h2l/parts/webqsp__H4_SK{R1,N3}_k{100,250,500}__PHG_con.{npy,RUN.json}`, `CALIB_EVAL_WEBQSP__noise_v1.{json,npz}` (json 1f3e45eb…, npz 782b2823…), `h2l/H2LM_NETSIZE_SURVEY__webqsp__v1.json`; STATUS = DEVELOPMENT (calibration of a systems substrate): the replicates are not candidate partitioners and nothing is adopted from them)

**52.1 Question and construction.** Addendum 7's nested arms scored 8 rows below PHG_SK against a pass line of 7 (§50). Before reading that as a recall cost, how many of the 786 rows does a SAME-ALGORITHM replicate of the frozen flat PHG lose to PHG_SK under addendum 1's rule? Two replicates, each on the frozen H4_SK hypergraph with the frozen driver, repair and validity gate and nothing else changed except one perturbation: SKR1 = PHG_RANDOMIZE_INPUT 1 (the input order is shuffled; the frozen run uses 0), SKN3 = three MPI ranks instead of four. K in {100, 250, 500}, same grid and rule.

**52.2 The gate (addendum 1, unchanged).** Rows below PHG_SK's ALL-gold count at matched B_P, per B_N in {100, 250, 500, 1000, 2000, 5000} (negative = more rows served than PHG_SK):

| replicate | K = 100 | K = 250 | K = 500 | max loss | verdict |
|---|---|---|---|---|---|
| SKR1 (shuffled input) | 0, 5, 2, 0, 0, 1 | 1, 1, 2, 4, **8**, **8** | 0, 0, 4, 2, 2, 5 | 8 | BORDERLINE |
| SKN3 (NP 3) | 1, 2, −5, −3, −2, 0 | −1, 3, 2, 0, 6, 4 | 0, −1, 0, 3, 0, 4 | 6 | RECALL_PRESERVING |

All six cells are scored for both replicates (raw Zoltan output left empty blocks in SKR1 at K = 500 (2) and in SKN3 at K = 250 (2) and K = 500 (3); the frozen empty-block repair fixed them, and the SKN3 repair lines show one single-vertex move per empty block with KM1 unchanged); every PHG cell of the reference reproduces the earlier EVAL bit for bit. F, addendum 8's statistic (the larger of the two replicates' maximum cell losses), is 8.

**52.3 Reading (addendum 8's pre-declared rule, applied).** F = 8 >= 8: the pass line of 7 lies INSIDE the flat recipe's own replicate-to-replicate variation on this population. The gate as written cannot certify a same-algorithm replicate of the reference, and the 8-row results of the nested arms of §50 (H2L5, H2L25: max loss 8 at K = 250) are not distinguishable from re-running PHG_SK with a shuffled input order. This is reported with the tables; the gate is not relaxed here, and nothing about addendum 7's recorded BORDERLINE verdicts changes. Two replicates estimate a floor, not a distribution: the result is "a same-algorithm replicate scored 8", never a confidence interval. The same reading applies descriptively beside §51: PQ64's 6, PQ128's 9 and PQ192's 3 are all of the replicate floor's order. Whether to move the line, to accept the nested arms as RECALL_PRESERVING within replicate noise, or to keep the line and close the nested path is the user's ruling on addendum 7's options A / B / C; the data support that the 8 is not a recall deficiency of the nested design as distinct from the frozen recipe's own noise.

## 53. FBX_SCALE Track C step 2: does an inverted-file search over the PQ64 codes keep the KNN family inside the gate? IVF over OPQ+PQ 64 B on WebQSP against the frozen PHG_SK (2026-10-01 local; host) (`scratchpad/{_ivf_knn,_ivf_phg,_ivf_eval,_ivf_declare}.py`; addendum 10 `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_10.json` (decf737e…, written before any IVF recall or IVF-keyed partition existed); records `ivf/WEBQSP_IVF_SAMPLE__v1.json` (8b21c872…), `ivf/WEBQSP_IVF_IVF_T{90,95,98}_KNN__v1.json` (99a68617…, b4162a7f…, bb4035f8…), `ivf/parts/webqsp__H4_IVF_T{90,95,98}_k{100,250,500}__PHG_con.{npy,RUN.json}`, `CALIB_EVAL_WEBQSP__ivf_v1.{json,npz}` (json 69104ac7…, npz 77ec3f83…); STATUS = DEVELOPMENT (calibration of a systems substrate): all three IVF arms are RECALL_PRESERVING under addendum 1's rule; no Freebase index, KNN or transfer is claimed)

**53.1 Question and construction.** §51 accepted m = 64 on EXHAUSTIVE decoded-vs-decoded search. The Freebase run cannot search exhaustively (245,350,737 x 245,350,737), so the deployed search is an inverted file over the stored codes. Which (lists, lists probed) keeps WebQSP's KNN family inside the end-to-end gate, and what does that point cost? Database = the 1,791,533 distinct WebQSP rows as their PQ64 codes with the accepted codebook (centroid table rounded to fp16, so a decoded vector is bitwise the fp16 storage vector of the exhaustive search of §51). Coarse quantiser = faiss spherical k-means (20 iterations, seed 2026) on 262,144 decoded unit-norm rows, nlist in {1024, 4096}. Scoring = faiss IndexIVFPQ, by_residual False, inner product (ADC = <decoded query, decoded row> exactly), then the best R = 512 candidates are re-ranked by cosine from the stored decoded norms. R was fixed BEFORE any grid cell was read by a full-probe wiring check on 300 queries: ADC ranks by ||x|| cos and the decoded norms spread 0.689–1.018, so a shallow re-rank loses true neighbours even with every list probed (R = 32: 0.9789, R = 128: 0.9978, R = 512: 1.0000 tie-tolerant recall@3). Phase A (100,000 queries) measured recall against the exhaustive PQ64 truth over the declared grid; phase B ran every row as a query at the arms the pre-declared rule selected, canonically merged the result into a KNN key set (STRUCT unchanged), and pushed H4_{STRUCT + KNN_IVF} through the frozen H4_SK rule and Zoltan PHG driver (NP 4, frozen repair and validity gate) at K in {100, 250, 500}. NER is not in the hypergraph (user ruling).

**53.2 Phase A: neighbour recall against the exhaustive PQ64 truth (tie-tolerant recall@3, 100,000 queries, 3 threads).**

| nlist | nprobe | scan fraction | recall@3 | thread-ms / query |
|---|---|---|---|---|
| 1024 | 8 / 16 / 32 / 64 / 128 / 256 | 0.96 / 1.89 / 3.74 / 7.38 / 14.38 / 27.74 % | 0.9159 / 0.9480 / 0.9701 / 0.9881 / 0.9972 / 0.9987 | 1.98 / 3.33 / 4.81 / 9.11 / 15.77 / 29.13 |
| 4096 | 32 / 64 / 128 / 256 / 512 / 1024 | 0.89 / 1.75 / 3.46 / 6.84 / 13.52 / 26.58 % | 0.9456 / 0.9672 / 0.9838 / 0.9949 / 0.9979 / 0.9991 | 1.93 / 3.04 / 4.98 / 8.53 / 16.09 / 30.99 |

The pre-declared arm rule picks nlist* = 4096 (its smallest nprobe reaching 0.95, 64, scans 1.75 %; nlist 1024 needs nprobe 32 and scans 3.74 %). Targets 0.90 / 0.95 / 0.98 therefore give IVF_T90 = nprobe 32, IVF_T95 = nprobe 64, IVF_T98 = nprobe 128 (all at 4096 lists, list length about 437 rows).

**53.3 Phase B: the KNN families (every one of the 1,791,533 rows as a query).**

| arm | nprobe | scan fraction | recall@3 vs PQ64 exhaustive | KNN pairs | kept of PQ64 exhaustive pairs | kept of exact pairs | precision vs exact |
|---|---|---|---|---|---|---|---|
| IVF_T90 | 32 | 0.92 % | 0.9452 | 6,524,177 | 96.9 % | 63.0 % | 61.0 % |
| IVF_T95 | 64 | 1.79 % | 0.9669 | 6,500,561 | 98.4 % | 64.0 % | 62.1 % |
| IVF_T98 | 128 | 3.48 % | 0.9838 | 6,477,229 | 99.4 % | 64.5 % | 62.9 % |
| (PQ64 exhaustive, §51) | – | 100 % | 1 | 6,432,568 | 100 % | 64.8 % | – |

(exact family: 6,313,874 pairs.) `self_in_top4` is 0.999999 for every arm.

**53.4 The gate (addendum 1, unchanged).** Rows below PHG_SK's ALL-gold count at matched B_P, per B_N in {100, 250, 500, 1000, 2000, 5000} (negative = more rows served than PHG_SK):

| arm | K = 100 | K = 250 | K = 500 | max loss | verdict |
|---|---|---|---|---|---|
| IVF_T90 | 2, −5, −10, −13, −13, −14 | 2, 2, 5, 0, 4, 2 | 0, 1, 1, −1, 0, 0 | 5 | RECALL_PRESERVING |
| IVF_T95 | 4, 4, −3, −12, −9, −8 | 2, 1, 3, −4, −1, −1 | 0, −2, −2, 2, 3, 2 | 4 | RECALL_PRESERVING |
| IVF_T98 | 5, 2, −8, −15, −13, −12 | 1, −3, −3, −4, 2, 2 | 0, 0, 0, 0, 0, 2 | 5 | RECALL_PRESERVING |

All nine cells of every arm are scored; no row has B_P = K (B_P / K 0.215 / 0.172 / 0.145, as PHG_SK's 0.218 / 0.174 / 0.146). Control: the six PHG_SK cells reproduce the earlier EVAL bit for bit (npz 3fdb91d8…). The partition cells needed the frozen empty-block repair at K = 250 and 500 (one empty block, one single-vertex move, KM1 unchanged, validity PASS), as the PQ arms did.

**53.5 Reading.** Under addendum 10's decision table an arm that clears the gate makes its scan fraction / probe count a candidate operating point, and the cheapest is the candidate: IVF_T90 (4096 lists, 32 probed, 0.92 % of the rows scanned), which keeps 96.9 % of the exhaustive PQ64 pairs. What the numbers do and do not separate: (i) the three max losses (5, 4, 5) and PQ64 exhaustive's 6 all sit below the pass line of 7 and below addendum 8's replicate floor F = 8; the data order none of them (loss is not monotone in recall: T95 loses 4, T98 loses 5), so the gate says only that none of these scan fractions crosses the line. (ii) IVF_T90 is the SMALLEST probe count on the declared grid (nprobe 32 at 4096 lists), so the pass does not locate the smallest passing probe count: lower counts were off the declared grid and are not run here. (iii) At K = 100 and B_N >= 1000 all three arms serve 8–15 MORE rows than PHG_SK (negative entries); descriptive only, of the order of the replicate floor in the other direction.

**53.6 First-order cost at Freebase (a plan input, not a claim).** A linear fit of the measured thread-seconds per query on the rows scanned over the 12 phase-A cells (3 threads) is 1.14 ms + 59.4 ns x rows. Applied to N = 245,350,737 query rows: (a) at a CONSTANT SCAN FRACTION (0.92 / 1.79 / 3.48 %) each query scans 2.26M / 4.39M / 8.55M rows, about 9,200 / 17,800 / 34,600 thread-hours; on the host's 30 cpu reservation that is 12.8 days at the cheapest arm with every cpu, so a fixed scan fraction does not scale. (b) At a CONSTANT LIST LENGTH (437 rows) and constant probe count, nlist grows with N to about 561,000: 32 / 64 / 128 probed lists scan 14.0k / 28.0k / 56.0k rows per query, about 134 / 191 / 304 thread-hours of scanning, but the coarse assignment of every query to 561,000 centroids is 245M x 561k x 1536 x 2 = 4.2e17 FLOP, which is hours on the GPU (5.9 h at an ASSUMED 20 TFLOP/s sustained, to be measured) and not feasible on the cpu, so the coarse quantiser must run on the GPU or be two-level. Hypothesis (b) is the natural transfer of the WebQSP point (same list length and probe count); it is a hypothesis.

**53.7 What this does and does not establish.** Established: on WebQSP the end-to-end gate of addendum 1 holds for an IVF search over the accepted PQ64 codes scanning 0.92 % (and 1.79 %, 3.48 %) of the rows, with a re-rank depth of 512. Not established: that a probe count or list length transfers to 245M rows (the WebQSP name set has 1.79M distinct rows; the transfer is measured later, on growing prefixes of the real Freebase codes against exhaustive GPU truth, in its own addendum), that a 561k-list coarse quantiser holds the recall, that the gate transfers to a KB with longer names, or anything about answer quality. Nothing under the Freebase tree was written; the Freebase KNN edge list, ANN index and canonical H4_SK are not built. The next step in the user's sequence is the transfer measurement, which needs real Freebase codes: the resumable CHUNKS encode (245.35M names, about 143 GPU hours uncontended) is running.

## 54. FBX_SCALE Track C step 3: does the IVF operating point transfer to real Freebase PQ64 codes as the database grows? T1 prefixes of the encode chunks, and the re-rank-depth diagnostic (2026-10-01 local; host) (`scratchpad/{_ivf_transfer,_ivf_transfer_declare,_ivf_rdiag,_ivf_rdiag_declare}.py`; addenda 11 and 12 `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_{11,12}.json` (00a5558a…, 01aee835…), each written before the recall it governs existed; records `ivf/FBX_IVF_TRANSFER_T1__v1.json` (1df3ca46…), `ivf/FBX_IVF_TRANSFER_T1R__v1.json` (1eb11e99…); STATUS = DEVELOPMENT (a systems measurement of neighbour recall; no KNN family, no partition, no gate is run, nothing is claimed about 245 M rows)

**54.1 Question and construction.** §53 fixed on WebQSP (1,791,533 names) an inverted file over the accepted PQ64 codes (4,096 lists of about 437 rows, 32 / 64 / 128 lists probed, re-rank of the best R = 512 by cosine) that keeps the end-to-end gate. The Freebase run searches 245,350,737 names. T1 runs that SAME search (addendum 10's scoring, unchanged) on prefixes of the first c in {1, 2, 4, 8, 16, 32} encode chunks of 131,072 distinct names each (131,072 … 4,194,304 rows, at most 1.7 % of the names), every chunk verified against the frozen contract and codebook, against the EXACT decoded-vs-decoded top 4 for 4,000 queries (blocked fp32 inner product over every database row). Two layouts: L437 (list length held at WebQSP's 437.4, nlist = n / 437.4 = 300 … 9,589) and F4096 (nlist = 4,096, from c = 4), nprobe 32 … 512 (never above nlist / 2), coarse k-means on min(n, 64 x nlist) rows. Primary metric: tie-tolerant recall@3 (addendum 10); four interleaved query folds give the SE. The prefix is the first c x 131,072 names in node first-occurrence order, NOT a random sample (addendum 11); a strided sample (T2) waits for the encode.

**54.2 T1: tie-tolerant recall@3 (± fold SE) at the declared cells (selected rows; the record holds all 47).**

| layout | nlist | rows | nprobe 32 | 64 | 128 | 256 | 512 | self in top 4 |
|---|---|---|---|---|---|---|---|---|
| (WebQSP, §53) | 4,096 | 1.79 M | 0.9456 | 0.9672 | 0.9838 | 0.9949 | 0.9979 | – |
| L437 | 1,199 | 0.52 M | 0.9708 | 0.9821 | 0.9860 | 0.9880 | 0.9888 | 0.9948 |
| L437 | 2,397 | 1.05 M | 0.9574 | 0.9687 | 0.9761 | 0.9793 | 0.9802 | 0.9878 |
| L437 | 4,795 | 2.10 M | 0.9442 | 0.9572 | 0.9646 | 0.9682 | 0.9696 | 0.9820 |
| L437 | 9,589 | 4.19 M | 0.9337 ± 0.0043 | 0.9493 | 0.9582 | 0.9627 | 0.9663 | 0.9800 |
| F4096 | 4,096 | 0.52 M | 0.9563 | 0.9723 | 0.9812 | 0.9859 | 0.9877 | 0.9948 |
| F4096 | 4,096 | 4.19 M | 0.9437 ± 0.0017 | 0.9543 | 0.9619 | 0.9656 | 0.9680 | 0.9800 |

Fold SEs are 0.001–0.004. At c = 32 (L437) the scan is 0.38 / 0.74 / 1.44 / 2.84 / 5.60 % of the rows (15.8k … 235k rows per query), 2,031 … 377 queries per second on 6 threads; F4096 scans a constant 0.87 … 13.05 % (36k … 547k rows per query) with recall within 0.010 of L437 at the same nprobe.

**54.3 Pre-declared readings (addendum 11).** R1 (L437, c = 32 against the WebQSP baseline at the same nprobe; holds if D >= -0.010): D = -0.0119 / -0.0179 / -0.0256 at nprobe 32 / 64 / 128, so DEGRADES at all three. R2 (L437 slope of recall per doubling of n over c = 4 … 32; falling if < -0.005): -0.0125 / -0.0110 / -0.0095 / -0.0087 / -0.0078 at nprobe 32 / 64 / 128 / 256 / 512, so FALLING (F4096: -0.0042 / -0.0062 / -0.0068 / -0.0072 / -0.0070). R3: the nprobe needed for 0.95 grows 32 (c = 4, 8) -> 64 (c = 16) -> 128 (c = 32) on L437; 0.98 is not reached at c >= 16 (and at c = 8 only at 512). The WebQSP-sized point (F4096, c = 32, nprobe 32) is within 0.002 of WebQSP's (0.9437 vs 0.9456) but recall at larger nprobe saturates near 0.968 (WebQSP: 0.9979). `self_in_top4` falls 0.99925 / 0.9975 / 0.99475 / 0.98825 / 0.9825 / 0.98125 over c = 1 … 32. By the declared decision table (R1 degrades, R2 falling) the next lever is the user's call; T1 alone does not say why.

**54.4 The re-rank-depth diagnostic (addendum 12).** The saturation and the falling `self_in_top4` suggested the loss is not in the probing: ADC ranks by ||x|| cos and cosine is applied only to the best R = 512, so a larger database may push true neighbours out of the best 512 even when their list is probed. Addendum 12 (written before any R > 512 recall on Freebase codes existed) re-used the T1 indexes and varied ONLY R in {512, 2048, 8192}. Tie-tolerant recall@3:

| rows | layout | nlist | nprobe | R = 512 | R = 2048 | R = 8192 | gain 8192 - 512 | q/s R 512 / 8192 |
|---|---|---|---|---|---|---|---|---|
| 1.05 M | L437 | 2,397 | 32 | 0.9574 | 0.9732 | 0.9765 | +0.0191 | 4,675 / 989 |
| 1.05 M | L437 | 2,397 | 128 | 0.9761 | 0.9918 | 0.9958 | +0.0197 | 1,600 / 642 |
| 4.19 M | L437 | 9,589 | 32 | 0.9337 | 0.9517 | 0.9623 | +0.0287 | 4,198 / 1,150 |
| 4.19 M | L437 | 9,589 | 128 | 0.9582 | 0.9748 | 0.9867 | +0.0285 | 2,541 / 904 |
| 4.19 M | L437 | 9,589 | 512 | 0.9663 | 0.9832 | 0.9952 | +0.0288 | 794 / 423 |
| 4.19 M | L437 | 9,589 | 9,589 (every list) | 0.9683 | 0.9852 | 0.9972 | +0.0288 | 31 / 36 |
| 4.19 M | F4096 | 4,096 | 32 | 0.9437 | 0.9607 | 0.9726 | +0.0288 | 2,879 / 762 |
| 4.19 M | F4096 | 4,096 | 512 | 0.9680 | 0.9849 | 0.9968 | +0.0288 | 269 / 184 |

Readings: Q1 R-LIMITED at every cell (gain >= 0.010; 0.019 at c = 8, 0.029 at c = 32, the same 0.029 from nprobe 32 to every list). Q2 PASS: every list probed with R = 8192 gives 0.9972 (>= 0.995). Q3 HOLDS with R = 8192: D = +0.0167 (nprobe 32) and +0.0029 (nprobe 128) against the WebQSP R = 512 baseline (both >= -0.010; the baseline is the frozen R = 512 point, not WebQSP re-measured at R = 8192). Q4 is not resolved by the grid (the smallest R within 0.003 of R = 8192 is 8192 at both sizes); descriptively at nprobe 128 R = 2048 is 0.004 below R = 8192 at c = 8 and 0.012 below at c = 32, so the depth needed grows with n. Q5 cost: below a full probe R = 8192 runs at 0.2 - 0.5 x the queries per second of R = 512 at the same nprobe (at every list probed the scan dominates: 31 vs 36 q/s).

**54.5 What the numbers do and do not say.** (i) The recall loss on growing Freebase prefixes is a loss of RE-RANK DEPTH, not of probing: with R = 8192 it falls from -0.026 to +0.003 against WebQSP at nprobe 128, and the full-probe ceiling is 0.997. (ii) The mechanism I proposed (a small-norm true neighbour pushed out by larger-norm rows) is NOT supported by the one descriptive check: at c = 32 L437 nprobe 128 R 512 the 575 missed true neighbours (of 12,000) have a HIGHER median decoded norm (0.978) than the returned ones (0.923) and than all rows (0.922). Why the best 512 by ADC exclude them is not identified here; dense or near-duplicate regions of the name space are a candidate and untested. (iii) No end-to-end claim: §53's gate passed at neighbour recall 0.945 (nprobe 32 on WebQSP), and the gate's noise floor (F = 8 rows) cannot separate recalls of this size; whether 0.934 (c = 32, R 512, nprobe 32) or 0.962 (R 8192) matters end to end is not measured. (iv) The prefix is node-order, not a random sample, so the size trend may be partly a property of the order; T2 (strided chunks) tests it after the encode. (v) The Freebase-scale cost of a larger R, of 561 k lists, and of the coarse quantiser on the GPU are NOT extrapolated from these cells.

**54.6 Decision for the user (addendum 12's decision table: Q1 R-limited and Q3 holds).** The frozen WebQSP operating point has R = 512; the Freebase index should not silently inherit it. Options: (a) keep R = 512 and accept the measured neighbour-recall loss (0.934 at 4 M rows and falling about one point per doubling), relying on the end-to-end gate's insensitivity at that level; (b) R grows with n (R = 8192 restores WebQSP-level recall at 4 M rows for 2.8 - 3.7 x fewer queries per second at nprobe 128 / 32; the depth needed at 245 M is not known); (c) a different scoring that does not rank by ||x|| cos (not tried here). Nothing is chosen.

## 55. FBX_SCALE Track B: how far can the TOP hypergraph of the nested partition be thinned before the gate breaks? Net-size caps and the M2 quotient on WebQSP, K in {100, 250, 500} (2026-10-01 local; host) (`scratchpad/{_h2lt,_h2lt_run,_h2lt_subk,_h2lt_eval}.py`; addendum 9 `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_9.json` (572a62c5…, written before any arm was scored); `h2l/parts/webqsp__H4_H2LT_<spec>_top_k25__PHG_con.RUN.json` (7 tops) and `…_<spec>_25_k<K>__PHG_con.*` (21 two-level maps); `CALIB_EVAL_WEBQSP__top_v1.{json,npz}` (775d9528…, cd70e751…; the six PHG_SK control cells reproduce the earlier EVAL bit for bit); STATUS = DEVELOPMENT (calibration of a systems substrate); no Freebase partition is built and none is claimed)

**55.1 Question and construction.** §50 showed that a nested partition (a 25-way top, then K-way inside each top block) reproduces the flat PHG_SK on WebQSP to within the pass line (BORDERLINE 8), and §52 that 8 rows is the same-algorithm replicate floor F. Its top stage still reads every pin of H4_SK at k = 25. At Freebase scale the top must be about 20 x thinner (about 6e9 pins against about 0.3e9 that fit the 81.6 GB host at about 190 B/pin). Addendum 9 asks how much the pins of the TOP hypergraph can be reduced before the gate leaves, with seven surrogates: N<c> drops every net of more than c pins (weight max(1, rint(1000 / deg))); Q<W>L<l> quotients the top hypergraph by the frozen M2 clustering (vertex weights = cluster sizes); Q<W>L<l>N<c> does the net cap first, then the quotient. The surrogate is partitioned at k = 25 by the frozen vertex-weighted driver (NP 4, tolerance 1.01), projected to the vertices, validated on the FULL top hypergraph, and handed to the unchanged net-split K-way stage. No refinement; no gold label is read. Scoring is addendum 1's gate, unchanged (matched B_P, six B_N, K in {100, 250, 500}; at most 7 of 786 rows below PHG_SK RECALL_PRESERVING, at least 24 FAILED, else BORDERLINE); F = 8 is reported beside each maximum loss and relaxes nothing.

**55.2 The frontier.**

| surrogate | top vertices | top pins | pin reduction | K = 100 loss at B_N 100 … 5000 | K = 250 | K = 500 | max loss | verdict |
|---|---|---|---|---|---|---|---|---|
| (flat top, §50) | 2,592,894 | 29,574,350 | 1.00 x | – | – | – | – | – |
| N16 | 2,592,894 | 20,559,883 | 1.44 x | 0, -5, -16, -16, -16, -16 | 2, 4, 4, -1, 3, 2 | 0, -1, -2, -6, -8, -7 | 4 | RECALL_PRESERVING |
| N8 | 2,592,894 | 17,491,223 | 1.69 x | 6, 3, -4, -6, -2, 1 | 1, -2, 3, -2, 0, 2 | 0, 0, 3, 4, 8, 10 | 10 | BORDERLINE |
| N4 | 2,592,894 | 11,473,697 | 2.58 x | 7, 15, 14, 21, 22, 21 | 0, 16, 31, 38, 43, 36 | 0, 3, 11, 18, 22, 24 | 43 | FAILED |
| Q512L2 | 455,513 | 15,352,968 | 1.93 x | 1, 0, -2, -9, -7, -6 | 1, 2, 3, 0, 4, 4 | 0, -1, 0, 3, 3, 4 | 4 | RECALL_PRESERVING |
| Q512L6 | 5,913 | 8,978,534 | 3.29 x | 9, 9, 11, 15, 18, 19 | 3, 5, 11, 15, 19, 20 | 0, 0, 10, 22, 25, 28 | 28 | FAILED |
| Q512L6N8 | 5,913 | 4,670,841 | 6.33 x | 7, 12, 17, 26, 29, 27 | 4, 4, 17, 25, 28, 25 | 0, 0, 8, 16, 19, 22 | 29 | FAILED |
| Q512L6N4 | 5,913 | 2,089,781 | 14.15 x | 8, 16, 16, 20, 26, 29 | 1, 9, 20, 28, 31, 30 | 0, 2, 5, 7, 13, 15 | 31 | FAILED |

(rows below PHG_SK's ALL-gold count at matched B_P; negative = more rows served than PHG_SK; all 18 cells of every arm scored, none PARTITION_INVALID. Top-surrogate wall time on the host 213 … 1,208 s.)

**55.3 Reading (addendum 9's decision table).** The largest RECALL_PRESERVING reduction is Q512L2 at 1.93 x (max loss 4), with N16 at 1.44 x (max loss 4); the requirement is about 20 x, so the gap to 20 x is about 10 x. N8 (1.69 x, 10) is BORDERLINE, 2 above the floor F = 8 and 3 above the pass line: its decision is the user's. Every arm that thins the top by 2.6 x or more FAILS (28 … 43 rows), against a pass line of 7 and a floor of 8. What the table separates and what it does not: (i) loss is NOT a function of pin reduction: Q512L6 (3.29 x, 28), Q512L6N8 (6.33 x, 29) and Q512L6N4 (14.15 x, 31) lose about the same while the pins fall 4.3 x further, so once the quotient is at 5,913 clusters (level 6) the net cap costs at most 3 more rows; the loss attaches to the cluster count of the quotient, as in §49 (recall kept only up to about 2 x pin compression, set by the number of clusters). (ii) The net cap alone is cheap up to N16 / N8 (0.69 / 0.59 of the pins kept, 0.97 / 0.92 of the weighted signal kept) and expensive at N4 (0.39 / 0.72, 10,487 vertices left without a net: 43 rows). (iii) A quotient between level 2 (455,513 clusters, preserves, 1.93 x) and level 6 (5,913 clusters, fails) with a net cap was NOT scored, so the frontier between 1.9 x and 3.3 x is not located and nothing here supports or excludes a quotient+cap arm that reaches several x while preserving; a 20 x preserving top is not found on this ladder by a factor of at least 10. (iv) These are 786 rows of one benchmark with a same-algorithm replicate floor of 8: the 2-row difference between N8 and F, and the order of N16, Q512L2 (4, 4) are not distinguished.

**55.4 What this does and does not establish.** Established: on WebQSP the nested design's top can lose at most about 1.4 - 1.9 x of its pins (net cap 16, or the level-2 M2 quotient) and stay inside the gate; every declared surrogate that thins more fails, and a quotient's loss follows its cluster count, not its pin count. Not established: any Freebase partition, memory or run time of a thinned top at 6e9 pins, that a different surrogate (a streaming top with the 1.01 balance bound, net sampling, a sparser quotient; the user's ruling lists them) reaches 20 x, or that the WebQSP frontier transfers to Freebase. The decision of whether to continue the nested path (the user's A accept on the one-row margin / B top-level work only / C close the nested path) is open: B has now been run and returns a frontier of about 1.9 x against a requirement of 20 x.

## 56. FBX_SCALE Track A: the Freebase NER shares-entity family E_NER^FB, built once on the host and frozen (2026-10-01 local; host) (`scratchpad/{_fbx_ner_build,_fbx_ner_chain,_ner_unit_specs,_ner_supersede}.py`, env `crag-ner`; records `results/FREEBASE_SCALE/ner/FBX_NER_FAMILY__v1.json` (2034f8db…), `FBX_NER_SUPERSEDE__653b0410__v1.json`, `FBX_NAME_TAIL__v1.json`, 64 `group_b*.json` and 64 `agg_r*.json`; arrays `data/freebase_scale/ner/ner_raw/r<NN>.{key,w}.npy` (2,248,080,088 B); STATUS = SUBSTRATE (a frozen derived artifact; no partition, no gate, no recall is run on it)

**56.1 What was built.** The user's ruling (late 2026-09-30) approved one NER run on the 302 M Freebase node names in the project environment `crag-ner`, hashed and frozen as E_NER^FB, and kept NER OUT of the partition hypergraph (H4_SK = STRUCT + KNN) unless explicitly ablated; its role is IR_L1 localization. The recipe is the frozen `src/pipeline/ner_edges.build_ner_edges` applied to the node names (spaCy 3.8.15, `en_core_web_sm` 3.8.0, parser and lemmatizer disabled, label whitelist EVENT / FAC / GPE / LOC / NORP / ORG / PERSON / PRODUCT / WORK_OF_ART, key length > 2, key = lower().strip(), keep keys with 2 <= df <= 25, edge weight = sum of 1/df over the shared keys). Pipeline: EXTRACT (604 parquet row-group units, 12 processes) -> GROUP (64 hash buckets) -> AGG (64 u-ranges) -> FINALIZE (write-once family record, which asserts one extract software fingerprint over all units). Edges are stored as a strictly increasing int64 key u * N + v (u < v) with a float32 weight, in 64 u-ranges.

**56.2 The family (from the record).**

| quantity | value |
|---|---|
| nodes N (rows) | 301,977,131 |
| rows with at least one entity key | 144,454,901 (47.8 %) |
| entity mentions | 224,843,814 |
| distinct keys | 39,415,882: df 1 = 24,836,947 (63.0 %); df 2-25 = 14,087,109 (35.7 %, kept); df > 25 = 491,826 (1.2 %, dropped) |
| pair rows before aggregation | 190,542,364 |
| distinct undirected edges | 187,338,642 |
| nodes with an edge / isolated | 49,734,606 (16.5 %) / 252,242,525 (83.5 %) |
| degree over nodes with an edge | mean 7.53; median 4, p90 18, p99 43, p99.9 83, max 16,575 |
| weight | sum 22,063,135; min 0.04 (= 1/25); max 528.2; mean 0.118 |
| integrity | 0 self-loops, 0 duplicate pairs, keys strictly increasing across and within ranges |

Wall time on the host for the final post-extract pass: GROUP 134 s, AGG 71 s, FINALIZE 129 s (334 s in all); the redo of 48 extract units (below) took 43 min on 12 processes.

**56.3 One consistency correction (supersession, not repair).** The first FINALIZE REFUSED: the assertion that every unit was extracted by the same software found two extract code hashes. 48 units (s000g0 ... s005g7, extracted 17:55-18:20 UTC) carried 653b0410…, the other 556 carried 8e096647…; the script had been patched at 18:34 UTC (the MAXCHARS rule, which cuts a name longer than spaCy's `max_length` of 1,000,000 characters; the name-length survey `FBX_NAME_TAIL__v1.json` shows names up to 1,984,811 characters exist). The assertion was NOT weakened. `_ner_supersede.py` moved the 48 units and everything derived from them (370 paths: pair files, `ner_raw`, group buckets, aggregated ranges) to `data/freebase_scale/ner/_history/extract_653b0410/`, wrote `FBX_NER_SUPERSEDE__653b0410__v1.json`, and the 48 units were re-extracted with the current code; GROUP, AGG and FINALIZE were then re-run. All 604 units carry 8e096647…. **Equivalence check (declared in the supersede record before it ran):** all 64 aggregated ranges (`pairs_in`, `distinct_pairs`, and the sha256 and byte count of both array files, 128 hashes) and all 64 group-bucket records are IDENTICAL to the superseded run's, so the patched rule changed nothing in the product for those 48 units.

**56.4 What this does and does not establish.** Established: a single-software, hash-pinned NER edge family over all 302 M Freebase names, with the frozen recipe's df window, whose build is reproducible from the units (the superseded and rebuilt aggregates agree to the byte). Not established: that the family helps localization or recall on Freebase (no gate or query has touched it; §31 found NER and KNN edges localize the text sets on the smaller corpora, but nothing is tested here); that spaCy's keys are identical across machines (the build script's `XCHECK` laptop-vs-host key hash is not part of this record); anything about the 83.5 % of nodes with no NER edge, which carry no signal in this family. NER stays out of the partition hypergraph.

## 57. FBX_SCALE Track C step 3c: does the §54 transfer picture hold on a stratified sample of the name space, and how does the re-rank depth R needed grow with the database size? T2 (2026-10-02 local; host) (`scratchpad/{_ivf_t2,_fbx_enc_order}.py`; addendum 13 `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_13.json` (3bb657c2…, written before any T2 recall existed; it also records the user's ruling on the encode order); record `ivf/FBX_IVF_TRANSFER_T2__v1.json` (0b70c3c0…); STATUS = DEVELOPMENT (a systems measurement of neighbour recall; no KNN family, no partition, no gate is run, nothing is claimed about 245 M rows)

**57.1 Question and construction.** §54's T1/T1R used prefixes of the encode chunks, i.e. the first names in node first-occurrence order, which is not a spread sample. On the user's ruling of 2026-10-01 (item 3, "keep the final encoded artifact identical") the encode job was re-ordered so that a 32-chunk stratified sample (the chunk at the midpoint of each of 32 equal strata, nested by bit-reversed stratum index so that the first 2^j chunks are an even spread) was encoded first. T2 runs addendum 10's search unchanged on the PQ64 codes of SAMPLE_ORDER[:c], c in {1, 2, 4, 8, 16, 32} (131,072 … 4,194,304 rows), with R in {512, 2048, 8192} as a design axis, layout L437 (nlist = n / 437.4, 300 … 9,589; nprobe 32 … 512, never above nlist / 2) and F4096 (nlist 4,096, c = 32), against the EXACT decoded-vs-decoded top 4 of 4,000 queries; every chunk record verifies against the frozen contract (aef5fb58…) and codebook (ac0214fb…). Six threads, 3,182 s.

**57.2 Tie-tolerant recall@3, L437 (selected cells; the record holds all 90; ± fold SE 0.001 – 0.004).**

| c | rows | nlist | nprobe | R = 512 | R = 2048 | R = 8192 | q/s R 512 / 8192 | scan |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.52 M | 1,199 | 128 | 0.9832 | 0.9970 | 0.9979 | 1,617 / 663 | 11.2 % |
| 8 | 1.05 M | 2,397 | 32 | 0.9584 | 0.9751 | 0.9800 | 2,311 / 615 | 1.46 % |
| 8 | 1.05 M | 2,397 | 128 | 0.9748 | 0.9910 | 0.9963 | 883 / 395 | 5.6 % |
| 16 | 2.10 M | 4,795 | 128 | 0.9567 | 0.9804 | 0.9925 | 790 / 398 | 2.9 % |
| 32 | 4.19 M | 9,589 | 32 | 0.9265 | 0.9528 | 0.9667 | 2,326 / 754 | 0.37 % |
| 32 | 4.19 M | 9,589 | 128 | 0.9473 | 0.9720 | 0.9876 | 975 / 458 | 1.4 % |
| 32 | 4.19 M | 9,589 | 512 | 0.9533 | 0.9782 | 0.9938 | 436 / 281 | 5.6 % |
| 32 (F4096) | 4.19 M | 4,096 | 128 | 0.9500 | 0.9748 | 0.9905 | 484 / 334 | 3.3 % |

Duplicate-code fraction (rows whose 64-byte code equals another row's) rises 0.52 % → 1.19 % over c = 1 → 32.

**57.3 Pre-declared readings (addendum 13).** P1 (strided minus T1 prefix at R = 512, c in {8, 16, 32} x nprobe {32, 128, 512}): D between +0.0010 and -0.0131 over the nine declared cells, two of them below -0.010 (c = 32, nprobe 128 and 512: -0.0109 and -0.0131; c = 32 at nprobe 32 / 64 / 256: -0.0072 / -0.0100 / -0.0117), and none above +0.010, so STRIDED_LOWER: the node-order prefix of §54 was OPTIMISTIC; by the declared table T2 supersedes T1/T1R for the transfer question (they stay as records). P2 (least-squares slope per doubling of n over c = 4 … 32): R = 512 at nprobe 128 -0.0126 → FALLING; R = 8192 -0.0035 (nprobe 128), -0.0056 (nprobe 32), -0.0019 (nprobe 512): still falling but 3 – 6 times slower. P3: R_LIMITED at every cell (G = R8192 - R512 = +0.022 at c = 8, +0.040 at c = 32, the same for nprobe 32 / 128 / 512); the smallest R within 0.003 of the R = 8192 recall at nprobe 128 is 8192 at both sizes (descriptively R = 2048 is 0.005 / 0.016 below at c = 8 / 32). P4 (R = 8192, c = 32 against the frozen WebQSP R = 512 baseline): D = +0.0212 (nprobe 32), +0.0121 (64), +0.0038 (128), -0.0032 (256), -0.0041 (512) → HOLDS at the declared nprobe 32 and 128. P5 (ADC rank of the true non-self top-3 at c = 32, L437, 12,000 pairs): at nprobe 128, 94.6 % rank < 512, 2.5 % rank 512 … 2,047, 1.6 % rank 2,048 … 8,191, 0.45 % in a probed list beyond rank 8,192, 0.8 % in no probed list; at nprobe 32 the same 92.6 / 2.6 / 1.5 / 0.11 / 3.2 %. So at nprobe 128 the R = 512 miss (5.4 %) is 4.1 points a DEPTH problem (rank 512 … 8,191), 0.45 beyond R = 8,192 and 0.8 a PROBING problem.

**57.4 What the numbers do and do not say.** (i) Everything §54 found holds in direction on a spread sample and is somewhat WORSE in size: at R = 512 the recall at 4.2 M rows is 0.927 / 0.947 / 0.953 (nprobe 32 / 128 / 512) instead of 0.934 / 0.958 / 0.966, and falls 1.1 – 1.5 points per doubling over c = 4 … 32 instead of 0.8 – 1.25 (§54.3). (ii) Re-rank depth is the lever again, and the depth needed grows with n: at nprobe 128 the R = 8192 recall is 0.9979 / 0.9963 / 0.9925 / 0.9876 at c = 4 / 8 / 16 / 32, i.e. it too erodes (about 0.4 points per doubling at c = 8 → 32), and it costs 1.5 – 3.8 x fewer queries per second than R = 512 at the same nprobe (the table cells). (iii) The WebQSP-level recall is restored at 4.2 M rows only with R = 8192 (and only up to nprobe 128), and the trend gives no reason to expect R = 8192 to suffice at 245 M rows; that value is T3's to measure, not to extrapolate. (iv) No end-to-end claim: the gate's noise floor (F = 8 rows) cannot separate recalls of this size, and no KNN family or partition is built here.

**57.5 Decision for the user (addendum 13's table: P1 STRIDED_LOWER, so T2 supersedes T1/T1R for the transfer question; P3 R_LIMITED; P4 HOLDS).** The R choice still waits for T3 (the full 245.35 M codes, GPU coarse quantiser, after the full encode and VERIFY), as ruled. T2 sharpens the options of §54.6: (a) keep R = 512 and accept 0.947 at 4 M rows (nprobe 128) falling about 1.3 points per doubling; (b) R grown with n, at 1.5 – 3.8 x lower throughput per query; (c) a different scoring that does not rank by ||x|| cos (not tried). Nothing is chosen.

## 58. FBX_SCALE Track B: does a V-cycle top level (refinement at every level of the ladder) keep the gate at the 14.2 x pin reduction the Freebase top needs? Four arms on WebQSP, K in {100, 250, 500} (2026-10-02 local; host) (`scratchpad/{_vc,_vc_sub,_vc_subk,_vc_eval}.py`, `_vc_ref.c`, `_vc_ref2.c`; addendum 14 `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_14.json` (67c99b68…, written before any V-cycle top was sub-partitioned or scored); `h2l/parts/webqsp__H4_H2LT_VCQ512*_25_k<K>__PHG_con.*` (12 two-level maps, 4 tops); `CALIB_EVAL_WEBQSP__vc_v1.{json,npz}` (e05d8a2f…, 0e27e2fb…); STATUS = DEVELOPMENT (calibration of a systems substrate); no Freebase partition is built and none is claimed)

**58.1 Question and construction.** §55 found that no unrefined top thinned by 2.6 x or more survives the gate (28 … 43 rows), while Freebase needs about 15 x (4.25e9 STRUCT pins -> 0.28e9, or 6e9 SK pins -> 0.40e9, against about 0.39e9 pins that fit the 73.4 GB budget at about 190 B/pin). The surrogate tops were refined once, at level 0, where strict-gain label propagation has nothing left to move. Addendum 14 keeps the stored §55 surrogate tops of Q512L6, Q512L6N8 and Q512L6N4 (5,913 clusters; 3.29 x, 6.33 x, 14.15 x fewer pins) and adds a V-cycle: the 25-way labels are projected from level 6 to level 0 and refined at EVERY level on the exact quotient of the full top hypergraph (not of the net-capped surrogate). The refiner is a gain-cache Fiduccia-Mattheyses pass on the k-way KM1 objective (FM2: 6 passes, stop after 500 non-improving moves, per-block cap 104,752) or, for the control, strict-gain label propagation (LP2: reproduces the v1 LP bit for bit). The nested K-way stage, the population (786 rows), matched B_P, six B_N and the verdict rule are §55's and addendum 1's, unchanged: at most 7 of 786 rows below PHG_SK RECALL_PRESERVING, at least 24 FAILED, else BORDERLINE. The label-free KM1 ratio (V-cycle top vs flat top, written down in the addendum before any gold row was read) was a proxy, not a criterion. No gold label is read before the EVAL.

**58.2 The gate.**

| arm | top pins | pin reduction | KM1 vs flat top (D1) | K = 100 loss at B_N 100 … 5000 | K = 250 | K = 500 | max loss | verdict |
|---|---|---|---|---|---|---|---|---|
| VCQ512L6FM2 | 8,978,534 | 3.29 x | 0.979 | -1, -8, -15, -20, -21, -20 | 1, -4, -3, -7, -3, -9 | 0, 0, 0, 4, 3, 4 | 4 | RECALL_PRESERVING |
| VCQ512L6N8FM2 | 4,670,841 | 6.33 x | 1.005 | 1, -12, -18, -17, -16, -17 | 1, -2, -7, -7, -3, -6 | 0, -3, 2, 2, 5, 7 | 7 | RECALL_PRESERVING (at the line) |
| VCQ512L6N4FM2 | 2,089,781 | 14.15 x | 0.981 | 5, -2, -3, -5, -4, -4 | 2, 1, 5, 7, 14, 13 | 0, -2, 3, 5, 8, 10 | 14 | BORDERLINE |
| VCQ512L6N4LP2 (control) | 2,089,781 | 14.15 x | 1.060 | 2, 1, -2, 0, 4, 5 | 1, 4, 7, 8, 15, 10 | 0, 0, 4, 3, 10, 12 | 15 | BORDERLINE |

(rows below PHG_SK's ALL-gold count at matched B_P; negative = more rows served than PHG_SK; all 18 cells of every arm scored, none PARTITION_INVALID after the frozen PHG_NONEMPTY_REPAIR_V1 recorded in each RUN.json: one empty block at K = 500 for each FM2 arm (blocks 494, 317, 364) and for N4LP2 at K = 250 (249) and K = 500 (497, 499; 2 moves). The six PHG_SK control cells reproduce `CALIB_EVAL_WEBQSP__top_v1` bit for bit.)

For reference, the same three surrogate tops UNREFINED (§55.2) lost 28 / 29 / 31 rows; the refinement at every level takes them to 4 / 7 / 14. The worst cell of each arm is two-sided churn, not a one-way slide: N4FM2 at K = 250, B_N 2000: 18 rows gained and 32 lost against PHG_SK (net 14, 614 of 786 rows ALL-gold vs 628); N8FM2 at K = 500, B_N 5000: 12 gained, 19 lost (653 vs 660).

**58.3 Reading (addendum 14's decision table).** (i) The 14.2 x FM2 arm is BORDERLINE (14 rows): addendum 14 sends this to the user, and I do not relax the gate. Beside the same-algorithm replicate floor F = 8 (§52) it is 6 rows over F, 7 over the pass line and 10 under the FAILED line. (ii) The largest RECALL_PRESERVING declared arm is Q512L6N8FM2 at 6.33 x with a max loss of exactly 7, i.e. ON the pass line and 1 under F (the loss is in the deepest cells, K = 500 at B_N 5000; at K = 100 and 250 it serves more rows than PHG_SK); Q512L6FM2 at 3.29 x preserves with 4. On Freebase 6.33 x gives 0.67e9 pins for STRUCT-only and 0.95e9 for SK, i.e. 1.7 x / 2.4 x over the 0.39e9 that fit; 14.15 x gives 0.30e9 (fits) and 0.42e9 (SK: 8 % over, about 15.4 x would be needed). (iii) The control: label propagation at every level, which ends 8 % higher on KM1 than FM (1.060 vs 0.981 of the flat top), loses 15 rows against FM2's 14; the gate does not separate the two refiners at 14.2 x. This is descriptive and selects nothing: it says only that the streaming implementation's refiner is not forced to be FM by the gate, while FM is what the KM1 proxy prefers. (iv) The KM1 proxy ordered the arms correctly but under-predicted the loss at 14.2 x: addendum 14 expected all three FM2 arms RECALL_PRESERVING from KM1 <= 1.005; two are, N4FM2 (KM1 0.981, the lowest of the 14 x arms) is not. So a flat-top-or-better KM1 on the full top hypergraph is not sufficient at 14.2 x; the extra loss attaches to the thinned surrogate (5,913 clusters plus a net cap of 4), as in §55.3 (i), not to the refinement. (v) The bundle behind this EVAL (`work/FBX_CALIB/bundle_v1`) had been deleted by the 2026-10-02 host cleanup (`results/L1_X/HOST_CLEANUP__2026-10-02.json`, FBX_CALIB 10.66 GiB) and was rebuilt with the unchanged `_fbx_calib.py` (a46001f5…) before this EVAL: FLAT identity 786 / 786 rows, `meta.json` differs from the one pinned in `top_v1` (1b1b482a… vs 2b33beff…) and the six PHG control cells equal the earlier EVAL's, which is the check that the rebuild is the same bundle for every gate cell.

**58.4 What this does and does not establish.** Established: on WebQSP, refinement at every level of the ladder moves the nested design's thinned top from FAILED to RECALL_PRESERVING at 3.3 x and 6.3 x and to BORDERLINE (14) at 14.2 x, against 28 – 31 rows unrefined; at 14.2 x the lost rows are two-sided churn. Not established: that 14 rows is acceptable (user), that a reduction between 6.3 x and 14.2 x preserves (not scored: a net cap of 6 or 5 on the same quotient is the untested interior), any streaming / out-of-core implementation (the refiner here holds the level in RAM, WebQSP being 29.6 M pins), any Freebase partition, or that WebQSP's pins-per-net and hub structure transfer. One population (786 rows), one S (25), K <= 500.

**58.5 Decision for the user (addendum 14: BORDERLINE at 14.2 x).** A: accept N4FM2 on a 14-row margin (6 over F = 8) and move to the streaming implementation and a Freebase STRUCT-only systems measurement. B (recommended): one more pre-registered arm between the two, Q512L6N6FM2 (net cap 6; its pin reduction lies between 6.33 x and 14.15 x and is measured, not assumed), before deciding; it needs one new Zoltan top and three K-way sub-partitions (a few host hours), and either brackets the frontier at an interior point or closes it. C: stay at 6.3 x (preserving, on the line) and accept the larger top: 0.67e9 STRUCT pins / 0.95e9 SK pins would need the coarsest surrogate to be partitioned with more memory than the 73.4 GB budget, i.e. a coarser first stage, not this one. Nothing is launched on this section until you rule.

## 59. FBX_SCALE Track B: the interior of the V-cycle frontier — where between 6.3 x and 14.2 x does the nested design stop preserving? Three pre-registered arms on WebQSP, K in {100, 250, 500} (2026-10-03; host) (`scratchpad/{_vc16_run,_vc16_eval}.py` on `_vc,_vc_sub,_vc_subk,_h2lt,_h2lt_eval`; addendum 16 `results/FREEBASE_SCALE/HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_16.json` (9faedf1c…bea209c, written before any of its tops was partitioned, sub-partitioned or scored); `results/FREEBASE_SCALE/vc/webqsp__VC_Q512L{5N4,4N4,6N5}_fm2_top_k25.{D1.json,npy}`; `h2l/parts/webqsp__H4_H2LT_VCQ512L{5N4,4N4,6N5}FM2_25_k<K>__PHG_con.*` (9 two-level maps); `CALIB_EVAL_WEBQSP__vc16_v1.{json,npz}` (c2f25e91…, b2aed3e0…); STATUS = DEVELOPMENT (calibration of a systems substrate); no Freebase partition is built and none is claimed)

**59.1 Question and construction.** §58 left one arm at each end: 6.33 x preserving (7 rows, on the line) and 14.15 x BORDERLINE (14 rows), with the STRUCT-only Freebase top needing >= 10.9 x (4.25e9 pins -> <= 0.39e9). Addendum 16 changes only the surrogate (everything else is §58's V-cycle: FM2 refinement at every level of the ladder on the exact quotient of the full top hypergraph, the nested K-way stage, the 786-row population, the addendum-1 verdict rule). Three new surrogate tops: Q512L5N4 (11,639 clusters, net cap 4: the §58 cap on a quotient with 2.0 x more clusters), Q512L4N4 (37,560 clusters, cap 4: 6.4 x more clusters) and Q512L6N5 (5,913 clusters, cap 5: the cap axis at the coarsest quotient). Each top was partitioned with the frozen vertex-weighted Zoltan driver, V-cycle refined (D1: KM1 only, no gold), sub-partitioned at K = 100 / 250 / 500 (nine two-level maps), and scored once. The six PHG_SK control cells reproduce `CALIB_EVAL_WEBQSP__vc_v1` bit for bit (the eval's own regression check: all six cells `equal`). The empty-block repair of the K-way stage fired for L5N4 at K = 500 (block 319), L4N4 at K = 250 and 500 (blocks 99 and 199), and for no L6N5 map; one single-vertex move each.

**59.2 The gate.**

| arm | clusters | cap | top pins | pin reduction | KM1 vs flat top (D1) | K = 100 loss at B_N 100 … 5000 | K = 250 | K = 500 | max loss | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| VCQ512L6N5FM2 | 5,913 | 5 | 3,070,354 | 9.63 x | 1.0009 | 2, -7, -11, -12, -8, -6 | 0, -4, -6, -7, -5, -5 | 0, -1, 1, 2, 1, 0 | 2 | RECALL_PRESERVING |
| VCQ512L4N4FM2 | 37,560 | 4 | 2,431,731 | 12.16 x | 1.0066 | 5, 6, 1, 1, 2, 4 | 0, 3, 4, 2, 7, 2 | 0, 0, 3, 1, 0, 1 | **7** | **RECALL_PRESERVING (on the line)** |
| VCQ512L5N4FM2 | 11,639 | 4 | 2,176,319 | 13.59 x | 1.0068 | 6, 7, -1, 2, 4, 3 | 1, 0, 3, -1, 5, 1 | 0, 0, 4, 10, 12, 13 | 13 | BORDERLINE |

(rows below PHG_SK's ALL-gold count at matched B_P; negative = more rows served than PHG_SK; all 18 cells of every arm scored, none PARTITION_INVALID; the gate compares at PHG_SK's matched B_P; on the arms' own routing B_P / K is 0.217 / 0.228 / 0.223 (L6N5 / L4N4 / L5N4) at K = 100 against PHG_SK's 0.218, 0.174 / 0.182 / 0.178 at K = 250 against 0.174, and 0.146 / 0.153 / 0.150 at K = 500 against 0.146, far from B_P = K.) Nested K-way KM1 relative to the flat PHG_SK at the same K (full K-way hypergraph, weighted): L6N5 0.993 / 0.975 / 0.938, L4N4 0.999 / 0.983 / 0.943, L5N4 0.997 / 0.976 / 0.938 at K = 100 / 250 / 500 (so all three are at or below flat on the objective, as in §58). The worst cell of each arm is two-sided churn, not a one-way slide: L4N4 at K = 250, B_N 2000: 24 rows gained and 31 lost against PHG_SK (621 of 786 ALL-gold vs 628); L5N4 at K = 500, B_N 5000: 15 gained, 28 lost (647 vs 660); L6N5 at K = 100, B_N 100: 5 gained, 7 lost (366 vs 368).

**59.3 Frontier of every FM2 arm of addenda 14 and 16, sorted by pin reduction** (F = 8 is the same-algorithm replicate floor of §52, descriptive only; the 10.9 x STRUCT-only and 15.4 x SK requirements are marked):

| arm | pin reduction | clusters / cap | max loss | verdict | Freebase coarsest pins, STRUCT-only (budget 0.39e9) | SK (6e9 pins) |
|---|---|---|---|---|---|---|
| VCQ512L6FM2 | 3.29 x | 5,913 / none | 4 | RECALL_PRESERVING | 1.29e9 | 1.82e9 |
| VCQ512L6N8FM2 | 6.33 x | 5,913 / 8 | 7 | RECALL_PRESERVING (on the line) | 0.67e9 | 0.95e9 |
| VCQ512L6N5FM2 | 9.63 x | 5,913 / 5 | 2 | RECALL_PRESERVING | 0.44e9 | 0.62e9 |
| **VCQ512L4N4FM2** | **12.16 x** | 37,560 / 4 | **7** | **RECALL_PRESERVING (on the line)** | **0.349e9 (fits)** | 0.493e9 (27 % over) |
| VCQ512L5N4FM2 | 13.59 x | 11,639 / 4 | 13 | BORDERLINE | 0.313e9 (fits) | 0.442e9 (13 % over) |
| VCQ512L6N4FM2 | 14.15 x | 5,913 / 4 | 14 | BORDERLINE | 0.300e9 (fits) | 0.424e9 (9 % over) |

**59.4 Reading (addendum 16's decision table).** (i) Row 1 of the table fires: **VCQ512L4N4FM2 (12.16 x) is RECALL_PRESERVING, so it is the reported Track B top for a Freebase STRUCT-only systems measurement** — 349.5 M coarsest surrogate pins against the ~390 M that fit the 73.4 GB budget, 10 % under. It is the largest preserving arm at or above 10.9 x (the two arms above it are BORDERLINE at 13 and 14). The SK requirement (15.4 x) stays open: nothing measured reaches it, and the best borderline arm (14.15 x) is still 9 % over for SK. (ii) **It is on the line, not comfortably inside it.** 7 rows is exactly the pass threshold and 1 under F = 8; all of it is one cell (K = 250, B_N 2000, 24 gained / 31 lost), every other of its 18 cells is <= 6. The verdict is also not ordered by reduction: L6N5 at 9.63 x loses 2 rows where L6N8 at 6.33 x loses 7. At 786 rows, loss differences of a handful of rows are at the replicate floor, so the right reading is "indistinguishable from the 6.3 x arm at 12.2 x", not "the loss is 7 and stays 7". F was measured on the flat PHG_SK, not on the nested V-cycle top; a second Zoltan seed of the L4N4 top would put a number on it and has NOT been run. (iii) The cluster-count hypothesis of addendum 16 is supported at net cap 4: 5,913 -> 11,639 -> 37,560 clusters give 14 -> 13 -> 7 rows, while the pin reduction moves the other way (14.15 -> 13.59 -> 12.16 x); the finer quotient buys back about half of the loss for a 14 % smaller reduction. Cap 5 on the coarsest quotient (9.63 x) preserves with margin but is below the STRUCT-only requirement. (iv) KM1 still does not track recall: L5N4 and L4N4 have the same KM1 on the full top hypergraph (1.0068 / 1.0066 of the flat top) and lose 13 and 7 rows, and the 14.15 x arm of §58 (0.981 of flat) loses 14 where L6N5 (1.0009) loses 2; the nested K-way maps are at or below flat on KM1 at every K.

**59.5 What this does and does not establish.** Established: on WebQSP at S = 25 the V-cycle nested design clears the gate at 12.16 x, a reduction that fits the STRUCT-only Freebase top, and a finer quotient at the same net cap buys back the loss at a given reduction (14 -> 13 -> 7 rows over a 6.4 x increase in clusters). Not established: that 7 rows is robust (one top per arm; the replicate floor of the nested top is unmeasured and F = 8 on the flat map is larger than the margin), any streaming / out-of-core implementation (the refiner of addendum 14 holds each ladder level in RAM; WebQSP is 29.6 M pins, Freebase STRUCT-only 4.25e9), any Freebase partition, that WebQSP's pins-per-net, hub structure and cluster-size profile transfer to a 302 M-node KB, anything for the SK requirement (15.4 x), a net cap of 3 (366,406 vertices without a net), K above 500, or any other S. One population (786 rows), one S (25).

**59.6 Next (pre-registered by addendum 16, row 1).** The streaming implementation of the L4N4 V-cycle top — out-of-core hierarchy construction and FM2 refinement at levels 0 .. 4, regressed bit for bit against the WebQSP records above (D1 map sha 4c043f0c…, KM1 796,782,889 on the full top hypergraph) — and the Freebase STRUCT-only systems measurement. It needs its own addendum (state declared before any code runs): a disk budget, a memory ceiling per level and a bit-identity regression on WebQSP. Optional and cheap relative to that: one replicate of the L4N4 top with a different Zoltan seed (one top + three K-way sub-partitions, ~3 host hours) to measure the nested top's own run-to-run floor before the streaming work is committed to this arm. Neither is launched by this section.

## 9. Records (DEV_A, §30's fresh split-A records and §31–§49's development records; nothing under `data/`)

`reach_diag_A_metaqa.json` · `parts/{metaqa,squad}__H4_SK_PATH2.{npz,json}` · `parts/<ds>__H4_SK__PHG_con.{npy,RUN.json}` (pipeline checks) · `parts/<ds>__H4_SK_PATH2__PHG_con.{npy,RUN.json}` · `eval_A_metaqa.json` (Mt-KaHyPar cache, FROZEN profile) · `eval_A_metaqa_phg.json` · `eval_A_squad_phg.json` · `ceiling_A_metaqa_phg__FROZEN_{capacity,rank,rank_ball2}_{unreached,missed}[__protect_voters].json` · `memdir_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.json` · `massvote_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.json` · `massvote_why_A_{metaqa,metaqa_phg}.json` · `PREREGISTRATION_RELSIG_DEV_A.json` · `relsig_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.json` · `relsig_why_A_{metaqa,metaqa_phg}.json` · `PREREGISTRATION_EDGEGEO_DEV_A.json` · `edgegeo_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.json` · `edgegeo_why_A_{metaqa,metaqa_phg}.json` · `PREREGISTRATION_MICROL3_DEV_A.json` · `microl3_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.json` · `microl3_A_metaqa__latent.json` (+ `microl3_A_metaqa_phg__latent.json` when the running diagnostic completes) · `microl3_why_A_{metaqa,metaqa_phg,squad,musique}.json` · `PREREGISTRATION_MICROL3_ASYM_DEV_A.json` · `microl3_asym_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.json` · `served_tail_gold_share_A.json` · `PREREGISTRATION_H2_SAFE_DEV_A.json` · `h2safe_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.json` · `h2safe_A_SUMMARY.json` · `h2safe_why_A_{metaqa,metaqa_phg}.json` · `h2safe_ceiling_A_{metaqa,metaqa_phg,squad,musique}.json` · `PREREGISTRATION_H2_PATCH1_DEV_A.json` · `h2patch_stat_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.json` · `h2patch_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.json` · `h2patch_A_SUMMARY.json` · `PREREGISTRATION_H3_PATCH1_DEV_A.json` · `h3patch_stat_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.json` · `h3patch_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.json` · `h3patch_A_SUMMARY.json` · `h3patch_why_A_{metaqa,metaqa_phg}.json` · `PREREGISTRATION_STRUCT_VERTEXCUT_V1.json` · `parts/metaqa__STRUCT_VCUT_V1_k{432,864,1296,1004}.{npz,json}` · `parts/metaqa__STRUCT_VCUT_V1_k{432,864,1296,1004}__PHG_con.{npy,RUN.json}` · `parts/metaqa__STRUCT_VCUT_V1_k1004__PHG_con__CAP100.npy` · `vcut_kstar_metaqa.json` · `vcut_struct_metaqa.json` · `PREREGISTRATION_STRUCT_VERTEXCUT_V1_REPLAY.json` · `vcut_replay_A_metaqa.{json,log}` · `PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json` · `vcut_kstar_{squad,musique}.{json,log}` · `parts/squad__STRUCT_VCUT_V1_k{202,404,606,1438,2144,2499}.{npz,json}` (+ `__PHG_con.{npy,RUN.json}`) · `parts/musique__STRUCT_VCUT_V1_k{1175,2350,3525,8252,11640,12518}.{npz,json}` (+ `__PHG_con.{npy,RUN.json}`) · `parts/squad__STRUCT_VCUT_V1_k2499__PHG_con__CAP100.{npy,json,log}` · `parts/musique__STRUCT_VCUT_V1_k12518__PHG_con__CAP100.{npy,json,log}` · `vcut_transfer_A_{squad,musique}.{json,log}` · `vcut_transfer_A_musique__posthoc_by_gold_count.{json,log}` · `vcut_transfer_SUMMARY.json` · `PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A.json` (v2; v1 in `_history/PREREGISTRATION_BALANCED_H2_PATCH1_DEV_A__v1_2026-09-15T17-50-31Z.json`) · `h2balanced_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}` · `h2balanced_A_SUMMARY.{json,log}` · `h2balanced_A_metaqa__run1_failed_at_the_section17_crosscheck.log` · `PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json` · `parts/{metaqa__BR2_OWNERS_k864,squad__BR2_OWNERS_k404,musique__BR2_OWNERS_k2350}.{npz,json,build.log}` (+ `__PHG_con.{npy,RUN.json,log}`) · `parts/{metaqa__BR2_k864,squad__BR2_k404,musique__BR2_k2350}__R2.{npz,json,log}` · `br2_replay_A_{metaqa,squad,musique}.{json,log}` · `br2_SUMMARY.{json,log}` · `br2_why_A_metaqa.{json,log}` (v1 in `_history/br2_why_A_metaqa__v1_gold_side_via_vote_table.{json,log}`) · `PREREGISTRATION_R2_LEGACY_ROUTER.json` · `r2router_A_{metaqa,squad,musique}.{json,log}` · `r2router_SUMMARY.{json,log}` · `PREREGISTRATION_QMAX_BALANCED_H2_PATCH1.json` · `qmaxbal_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}` · `qmaxbal_SUMMARY.{json,log}` · `PREREGISTRATION_COVSCORE_H2_PATCH1.json` · `covscore_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}` · `covscore_SUMMARY.{json,log}` · `PREREGISTRATION_BALANCED_H3_PATCH1.json` · `h3bal_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}` · `h3bal_SUMMARY.{json,log}` · `FREEZE_QMAX_BALANCED_H2_PATCH1.json` · `TRANSFER_RESOURCE_SURVEY_2WIKI_HOTPOTQA.json` · `PREREGISTRATION_TRANSFER_2WIKI_HOTPOTQA.json` · `transfer_repro_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}` · `transfer_repro_SUMMARY.{json,log}` · `PREREGISTRATION_PROTOTYPE_LADDER.json` · `PREREGISTRATION_PROTOTYPE_LADDER_IVF_ADDENDUM.json` · `proto_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}` · `proto_SUMMARY.{json,log}` · `proto_IVF_SUMMARY.json` · `PREREGISTRATION_FLAT_VS_PARTITION.json` · `flat_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}` · `flat_SUMMARY.{json,log}` · `PREREGISTRATION_FLAT_BALANCED_H2.json` (v2; v1 and byte-identical copies of the v1 module / summary / chain in `_history/`) · `flath2_G_{webqsp,hotpotqa,2wiki}.{json,npz,log}` · `flath2_E_{webqsp,hotpotqa,2wiki}.json` · `flath2_E_{hotpotqa,2wiki}.log` · `flath2_A_{metaqa,metaqa_phg,squad,squad_phg,musique}.{json,log}` · `flath2_SUMMARY.{json,log}` (§30: fresh split A + DEV_A) · `phg_data/` (shards + Zoltan run dirs) · code: `scratchpad/_l1c_{reach_diag,path2_build,partition,phg,eval,ceiling,memdir,massvote,massvote_why,relsig,relsig_why,edgegeo,edgegeo_why,microl3,microl3_why,microl3_asym,served_tail,h2safe,h2safe_why,h2safe_ceiling,h2patch_stat,h2patch,h3patch_stat,h3patch,h3patch_why,vcut_build,vcut_lib,vcut_kstar,vcut_metrics,vcut_replay,vcut_transfer_prereg,vcut_cap,vcut_transfer,vcut_transfer_summary,vcut_transfer_posthoc,h2balanced,h2balanced_prereg,h2balanced_prereg_v2,br2_prereg,br2_lib,br2_build,br2_assign,br2_replay,br2_summary,br2_why,r2router_prereg,r2router,r2router_summary,qmaxbal_prereg,qmaxbal,qmaxbal_summary,covscore_prereg,covscore,covscore_summary,h3bal_prereg,h3bal,h3bal_summary,transfer_blockmax,transfer_composite,transfer_repro_summary,proto,proto_summary,proto_ivf_summary,flat,flat_summary,flath2,flath2_summary,flath2_agree_probe}.py` + `scratchpad/_l1c_h2balanced_chain.sh` + `scratchpad/_l1c_r2router_chain.sh` + `scratchpad/_l1c_qmaxbal_chain.sh` + `scratchpad/_l1c_covscore_chain.sh` + `scratchpad/_l1c_h3bal_chain.sh` + `scratchpad/_l1c_transfer_repro_chain.sh` + `scratchpad/_l1c_proto_chain.sh` + `scratchpad/_l1c_flat_chain.sh` + `scratchpad/_l1c_flath2_chain.sh` (new modules; no CONTRACT_FILE edited; `_l1c_h2balanced.py`, the five `_l1c_br2_*` modules, `_l1c_r2router{,_summary}.py`, `_l1c_qmaxbal{,_summary}.py`, `_l1c_covscore{,_summary}.py`, `_l1c_h3bal{,_summary}.py` and `_l1c_transfer_{blockmax,composite}.py` are sha-pinned by their pre-registrations; `_l1c_br2_why.py` is pinned as an import of §23; `_l1g_candidate.py` / `_l1g_core.py` as imports of §24; `_l1c_h3patch.py` as an import of §26; `_l1c_proto{,_summary}.py` and `_l1c_proto_chain.sh` by `PREREGISTRATION_PROTOTYPE_LADDER.json`, `_l1c_proto_ivf_summary.py` by its IVF addendum; `_l1c_flat{,_summary}.py` and `_l1c_flat_chain.sh` by `PREREGISTRATION_FLAT_VS_PARTITION.json`; `_l1c_flath2{,_summary,_agree_probe}.py` and `_l1c_flath2_chain.sh` by `PREREGISTRATION_FLAT_BALANCED_H2.json` v2). §31 (development records, `results/L1_DEV/`): `loc_population.json` · `loc_{metaqa,musique,squad}__v1.{json,npz}` · `loc_SUMMARY__v1.json` · `soft_{metaqa,musique,squad}__v1.{json,npz}` · `soft_SUMMARY__v1.json` · `ceiling_blocks__v1.json` · `edgediag_{metaqa,musique}__v1.json` · `degprior_{metaqa,musique}__v1.json`; code `scratchpad/_l1d_{lib,loc,loc_summary,soft,soft_summary,ceiling,edgediag,degprior}.py` (new modules, sha-pinned by their records) and, built but without a v1 record, `scratchpad/_l1d_{arms,hier,act,reach,var_summary}.py`. §32 (development records, `results/L1_DEV/`): `node1h_{metaqa,musique,squad}__v1.{json,npz}` · `node1h_SUMMARY__v1.json`; code `scratchpad/_l1d_{node1h,node1h_summary}.py` (new modules). The node1h records pin `_l1d_node1h.py`, the shared runner `_l1d_arms.py` (so it now has a v1 record), `_l1d_lib.py` and `_l1d_edgediag.py`. `_l1d_hier.py` and `_l1d_act.py` are retired without a record (§32.1). §33 (development records, `results/L1_DEV/`): `adaptbp_{metaqa,musique,squad}__v1.{json,npz}` · `adaptbp_SUMMARY__v1.json` · `adaptbp_VIEW__v1.json` (post-hoc view); code `scratchpad/_l1d_{adaptbp,adaptbp_summary,adaptbp_view}.py` (new modules). The adaptbp records pin `_l1d_adaptbp.py`, `_l1d_arms.py`, `_l1d_lib.py` and, through `structures.imports`, `_l1d_node1h.py` and `_l1d_edgediag.py`; they assert identity with the node1h records. The view record pins `_l1d_adaptbp_view.py`, `_l1d_adaptbp.py` and `_l1d_lib.py`, and the shas of the records it reads. §34 (development records, `results/L1_DEV/`): `route_{metaqa,musique,squad}__v1.{json,npz}` · `route2_{metaqa,musique,squad}__v1.{json,npz}` · `route3_{metaqa,musique,squad}__v1.{json,npz}` · `route_SUMMARY__v1.{json,npz}` · `route_VIEW__v1.json` (post-hoc view); code `scratchpad/_l1d_{route,route2,route3,route_summary,route_view}.py` (new modules). Each round's records pin its harness, `_l1d_arms.py`, `_l1d_lib.py` and, through `structures.imports`, `_l1d_node1h.py`, `_l1d_edgediag.py` and `_l1d_adaptbp.py`; the route2 records add `_l1d_route.py`, the route3 records `_l1d_route.py` and `_l1d_route2.py`. The round-1 records assert identity with the node1h and adaptbp records (`structures.reproduces`); the route2 and route3 records pin their predecessors' records by sha (`structures.extends`). The summary record pins `_l1d_route_summary.py`, the three harnesses and the nine records (json and npz shas); the view record pins `_l1d_route_view.py` and the summary's sha. §35 (development records, `results/L1_DEV/`): `scale_{metaqa,musique,squad}__v1.{json,npz}` · `scale_SUMMARY__v1.json`; shard maps and build records in `results/L1_COVPART/parts/`: `<ds>__H4_SK_k<K>.{json,npz}` (the hypergraph at each K), `<ds>__H4_SK_k<K>__PHG_con.{npy,RUN.json,SCALE.json,driver.log}` (MetaQA K 5000: `driver.log` and `FAILED.json`) and `<ds>__H4_SK_k<K>__MTK.{npy,RUN.json,stats.json,worker.log}` (MetaQA and SQuAD), 180 files, plus the builder's guard script `_scale_mtk_guard.sh`; `results/L1_COVPART/_history/` holds the builder's three versions `_l1d_scale_parts__v{0,1,2}_<sha8>.py` (v2 = the current module) and the three retried runs' `…__MTK__attempt1.{FAILED.json,worker.log}`; code `scratchpad/_l1d_{scale_parts,scale,scale_summary}.py` (new modules). Each scale record pins `_l1d_scale.py`, `_l1d_arms.py` and `_l1d_lib.py` and, through `structures.imports`, `_l1d_node1h.py`, `_l1d_edgediag.py`, `_l1d_adaptbp.py`, `_l1d_route.py`, `_l1d_route2.py`, `_l1d_route3.py` and `_l1d_scale_parts.py`; it pins the three route rounds' records by sha and reproduces their counts on the served cells (`structures.reproduces`). Each build record pins the builder version that wrote it, `hypergraph.py` and `_l1hu_build.py`; the PHG and MTK records add `partition.py`, `_l1c_phg.py` and `_l1hu_local_worker.py`, the PHG run records the driver binary, its wrapper and its source, and the MTK run records the guard script (its text equals `partition.py`'s `GUARD_SH`). The summary record pins `_l1d_scale_summary.py`, the harness and the three records (json and npz shas). §36 (development records, `results/L1_DEV/`): `kscale_{metaqa,musique,squad}__v1.{json,npz}` · `kscale_SUMMARY__v1.json`; code `scratchpad/_l1d_{kscale,kscale_summary}.py` (new modules; `_l1d_kscale.py` subclasses `_l1d_scale.py`'s ScaleSpec, imported unchanged). Each kscale record pins `_l1d_kscale.py`, `_l1d_arms.py` and `_l1d_lib.py`; through the subclassed harness it carries the scale records' `structures.imports` and `structures.reproduces`, and `structures.kscale.reproduces_scale_v1` pins the scale v1 record it reproduces (json and npz sha, and its harness sha, which equals `_l1d_scale.py`'s). The summary record pins `_l1d_kscale_summary.py`, the harness, `_l1d_scale_summary.py` (its helpers), `_l1d_scale.py` and the three records (json and npz shas). §37 (development record, `results/L1_DEV/`): `L1_ROUTING_CARRY_FORWARD__v1.json` (sha aaae66fa…); code `scratchpad/_l1d_carry.py` (new module, sha-pinned by its record, which asserts the 14 module shas and pins the kscale records, the summary, the population and every map it copies evidence from). Host shard-map lane (`results/L1_HOST/`): `HOST_LANE_DECLARATION.json` (addb4426…) · `apt/apt_20260929T095207Z.log` · `PHG_BUILD_HOST.json` · `PLACEMENT_TEST__v1.json` (16316ee3…) · `HOST_LANE_AUTHORIZATION__2026-09-29.json` (743ecb32…) · `parts/<ds>__H4_SK_k<K>.json` (the hypergraph records; the npz stay on the host) · `parts/<ds>__H4_SK_k<K>__PHG_con.{npy,RUN.json}` (MetaQA K 5000: `FAILED.json`); code `scratchpad/_l1h_{host,place}.py`, `scratchpad/_l1h_{apt,apt_time,diag}.sh` and `rx.toml` (new; the host module is pinned by the authorization, the placement module by its record). §38 (development records, `results/L1_DEV/`): `kbres_metaqa__v1.{json,npz}` (json 282791b4…, npz 6c34e20c…) · `kbres_metaqa__v1__L3SPLIT.json` (68a6d596…); code `scratchpad/_l1d_{kbres,kbres_summary}.py` (new modules). The kbres record pins `_l1d_kbres.py`, `_l1d_arms.py` and `_l1d_lib.py`. It also pins the carry-forward record (aaae66fa…) and the kscale v1 record and npz, and it asserts identity with that npz. The L3SPLIT record pins `_l1d_kbres_summary.py` and the kbres record (json and npz shas), and reproduces its ALL and lever-set counts. §39 (development records, `results/L3_DEV/`): `l3hop_metaqa__v1.{json,npz,log}` (json df08ca0b…, npz 1f851d0b…); code `scratchpad/_l3d_hop.py` (new module, sha-pinned by its record, which also pins the carry-forward record and the kscale v1 record and npz, and asserts identity with the kbres v1 npz). §40 (development records, `results/L3_DEV/`): `l3typed_metaqa__v1.{json,npz,log}` (json a6e80216…, npz 0ee65e76…) · `l3budget_metaqa__v1.{json,npz,log}` (json c96eecd2…, npz db20afb6…) · the host placement runs `l3typed_metaqa__h1.{json,npz}` and `l3budget_metaqa__h1.{json,npz}` (npz byte-identical to v1); code `scratchpad/_l3d_{typed,budget}.py` (new modules, sha-pinned by their records; the typed record pins the L1_P90 lane functions it imports, the budget record pins `_l3d_typed.py` and the TYPED v1 record). Host L3 lane (`results/L3_HOST/`): `HOST_STAGE_DECLARATION__L3_HOST__v1.json` (37ff06f7…) and its addenda 1–4 (12642119…, c12815c6…, 78e0644b…, 854a483a…) · `bundle_metaqa__L1DEV.{json,npz}` · `BLAS_PROBE__{laptop,host_t12}.json` · `PLACEMENT_TEST__L3_HOST__v1.json` (62135f84…) · `PLACEMENT_TEST__L3_BUDGET__v1.json` (5086a636…) · `HOST_STAGE_AUTHORIZATION__L3_HOST__2026-09-29.json` (cb94b481…) · `runs/*.run.json` (the runner's sidecars); code `scratchpad/_l3h_{run,place,bundle,blasprobe}.py` (new; pinned by the declaration, its addenda and the authorization) and `rx.toml` (the push list). §41 (development records, `results/L3_DEV/`): `l3e2_metaqa__v1.{json,npz,log}` (json e3814532…, npz 51200669…) · the host placement run `l3e2_metaqa__h1.{json,npz}` (npz byte-identical to v1) · `relenc_metaqa__v1.{json,npz}` (6290f3bd…, 17231a29…) · `relenc_webqsp__v1.{json,npz}` (d16a1767…, aaec8d03…); host encoder lane (`results/L3_HOST/`): `HF_DOWNLOAD__v1.json` (bd183aa0…) · `ENC_PROBE__v1.json` (7d0f1d56…) · `ENC_REPRO__v1.json` (7bb60d8f…) · addenda 5–10 (5de9f9bd…, dbf7ea22…, 9cbbe6ef…, ebc80655…, 30a4fd5b…, 835aa711…) · `HOST_STAGE_AUTHORIZATION__L3_HOST_ENC__2026-09-29.json` (9313ed0f…) · `PLACEMENT_TEST__L3_E2__v1.json` (7363fc61…) · `runs/_l3d_e2__RUN__metaqa__h1.run.json` (7bd08c86…); `results/HOST_YIELD/`: `HOST_YIELD_POLICY__v1.json` (b468a0d5…) and the job logs `{crag-encrepro,crag-relenc,l3h-e2-h1}.jsonl` (eb882be0…, 566fe6ce…, a98d41e4…); code `scratchpad/_l3d_e2.py` (new module, sha-pinned by its record, which also pins the RELENC MetaQA record and the TYPED v1 record), `_enc_hfdl.py`, `_enc_host.py`, `_host_yield.py` and its tests `_host_yield_test_{rule,e2e}.py`. §42 (development records, `results/L3_DEV/`): `l3e2b_metaqa__v1.{json,npz,log}` (json a81a1aae…, npz caf183c6…, log f14eeaea…) · the host placement run `l3e2b_metaqa__h1.{json,npz}` (npz byte-identical to v1; json 14946b65…) · `results/L3_HOST/`: addenda 11–12 (fd2eac25…, 157d49aa…), `PLACEMENT_TEST__L3_E2B__v1.json` (692b5d9f…), `runs/_l3d_e2b__RUN__metaqa__h1.run.json` (5076f8fc…) · `results/HOST_YIELD/l3h-e2b-h1.jsonl` (e65a44be…); code `scratchpad/_l3d_e2b.py` (e1441f3a…, generated from `_l3d_e2.py`; sha-pinned by its record, which also pins the E2 v1 json and npz). §43 (closure records, `results/DATA_FREEZE/`): `DATA_FREEZE_2026-09-30__v1.json` (a49efff6…) · `DATA_FREEZE_VERIFICATION_2026-09-30__v1.json` · `DATA_FREEZE_VFINAL__2026-09-30__v1.json` (d83cc1e0…); code `scratchpad/_data_freeze.py`. §44 (development records, `results/L3_DEV/`): `l3w_population_webqsp__v1.json` (7e67c9b1…) · `l3w_webqsp__v1.{json,npz,log}` (json 658bf843…, npz b121f94e…, log 89aaee83…); code `scratchpad/_l3w_pop.py` and `scratchpad/_l3w_transfer.py` (a668cd13…; new modules; the run record pins the harness and the 18 imported modules by sha, the population record, `relenc_webqsp__v1.{json,npz}`, the lane code snapshot and the two PHG maps with their run records). §45 (transfer records, `results/FREEBASE_SCALE/`): `HOST_STAGE_DECLARATION__FBX_HOST__v1.json` (5a2e772f…) · `FBX_HOST_VERIFY__v1.json` (d8ff06fd…); code `scratchpad/_fbx_{stage,hf_upload,hf_urls,hf_pull,verify_host,declare}.py` (new; the declaration pins all of them except `_fbx_declare.py`, plus `_host_yield.py` and the three verifier modules, by sha) and `rx.toml` (the push list). §46 (systems records, `results/FREEBASE_SCALE/`): `HOST_STAGE_DECLARATION__FBX_SCALE__v1.json` (99624a10…) · `FBX_GRAPH_STATS__v1.json` (09dbfc11…) · `FBX_PARTITION_FEASIBILITY__v1.json` (17ee5b06…; two superseded versions in `_history/`, f47377ff and 86366b05) · `FBX_LOCALISATION_FEASIBILITY__v1.json` (5d063bf6…); code `scratchpad/_fbx_{stats,feasibility,encode_feasibility,scale_declare}.py` (new; the declaration pins `_fbx_stats.py` and `_host_yield.py` by sha; the stats record pins its own code, the feasibility records pin their inputs by sha) and `rx.toml` (the push list: the declaration and `_fbx_stats.py`). §47 (calibration records, `results/FREEBASE_SCALE/`): addenda 1–3 `HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_{1,2,3}.json` (1eaf5f66…, 7b938de8…, 23b186b9…) · `CALIB_MAPS_WEBQSP__{v1,lp_v1,cn_v1}.json` (6f6fa45d…, 6c296202…, 7f4f5242…) · `CALIB_EVAL_WEBQSP__smoke200.{json,npz}` (a472b55f…; a smoke read before addendum 2, disclosed there) · `CALIB_EVAL_WEBQSP__v1.{json,npz}` (json 5e037a1c…; the full-population verdict for hash / LDG / LP; regression on §44 PASS) · `CALIB_EVAL_WEBQSP__cn_v1.{json,npz}` (json dccb5b1b…, npz 3f9fc854…; R3b, with every PHG and LP_S cell asserted bit-identical to v1) · `FBX_BUILD__webqsp__{LDG_S,HASH,LP_S}__regr.json` (builder regressions, reproduce the map hashes) · `_wq_sha_host.json` (transfer integrity, 152 files) · host bundles `work/FBX_CALIB/bundle_{smoke,s200,v1}`; code `scratchpad/_fbx_{part,lp,cn,calib,calib_cn,lp_maps,cn_maps,scale_build,scale_build_lp,scale_route,scale_addendum1,scale_addendum2,scale_addendum3}.py` and `scratchpad/_fbx_{ldg,lp,cn}.c` and `scratchpad/_wq_{missing,stage,sha}.py` (new modules; each EVAL record pins its own code by sha; `_fbx_scale_route.py` is built for stage 3b and not yet run) and `rx.toml` (the push list). §48 (rung-4 records, `results/FREEBASE_SCALE/`): addendum 4 `HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_4.json` (6dd58473…) · `ml/ML_DRIVER_BUILD.json` (a8cec35a…) · `ml/ML_DRIVER_REGRESSION.json` (9375fca3…, PASS) · `ml/webqsp__ML_S__clusters.{npy,RUN.json}` (RUN.json f22033f5…) · `ml/webqsp__ML_S_k{100,250,500}.{npy,RUN.json}` and `_proj.npy` · `ml/webqsp__ML_S_k1000.FAILED.json` (2f6f81d5…, PARTITION_INVALID) · `parts_S/webqsp__H4_S_k{100,250,500,1000}__PHG_con.{npy,RUN.json}` (the PHG_S control maps) · `CALIB_EVAL_WEBQSP__phgs_v1.{json,npz}` (json 20f50346…; PHG_S control, FAILED the gate at 59) · `CALIB_EVAL_WEBQSP__ml_v1.{json,npz}` (json 87678ae9…, npz 69a009ee…; ML_S: gate FAILED 48, same graph BORDERLINE 12; every PHG and PHG_S cell asserted bit-identical to phgs_v1); code `scratchpad/_ml_{agg.c,coarsen.py,run.py,eval.py,addendum4.py}` and `src/l1_lowmem/phg_driver/phg_driver_vw.c` (new; each record pins its own code by sha) and `rx.toml`. §49 (laboratory records, `results/FREEBASE_SCALE/`): addendum 5 `HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_5.json` (ddff074c…) · `ml2/ORACLE_PIN_REDUCTION__v1.json` (46d515e0…) · `ml2/webqsp__M{1,2,3,4}_W<Wmax>_L<level>_k<K>.{npy,_proj.npy,RUN.json}` (100 cells of the 36 declared candidates; maps stay on the host) · `ml2/webqsp__M0R_k{100,250,500}.RUN.json` · `CALIB_EVAL_WEBQSP__ml2_v1.{json,npz}` (json 82ee5650…; every PHG cell asserted bit-identical to ml_v1; npz stays on the host) · `ML2_COMPRESSION_CURVE__v1.json` (13b4bf82…) and `ml2/ML2_COMPRESSION_CURVE__v1.png` · `FBX_STRUCT_COMPONENTS__v1.json` (3313d854…) · `FBX_NER_PROBE__v1.json` (95cf620c…); code `scratchpad/_ml2_{agg.c,coarsen.py,run.py,cn.c,cn.py,cell.py,eval.py,oracle.py,addendum5.py,launch.py,curve.py}`, `scratchpad/_fbx_cc.{c,py}` and `scratchpad/_fbx_ner_build.py` (new; each record pins its own code by sha; the NER builder is tested but not run) and `rx.toml` (the push list). §50 (Track B records, `results/FREEBASE_SCALE/`): addendum 7 `HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_7.json` (a3867580…) · `h2l/parts/webqsp__H4_H2LTOP_k{5,25}__PHG_con.{npy,RUN.json}` · `h2l/parts/webqsp__H4_H2L{5,25}_k{100,250,500}__PHG_con.{npy,RUN.json}` · `CALIB_EVAL_WEBQSP__h2l_v1.{json,npz}` (json e08b7c4e…, npz 1af2d87f…; the six PHG cells asserted bit-identical to ml2_v1); code `scratchpad/_h2l.py`, `scratchpad/_h2l_eval.py`, `scratchpad/_h2l_addendum7.py` (new; each record pins its own code by sha). §51 (Track C records, `results/FREEBASE_SCALE/`): addenda 6/6A `HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_{6,6A}.json` · `pq/WEBQSP_{DISTINCT_ROWS,EXACT_KNN,PQ64_KNN,PQ128_KNN,PQ192_KNN}__*.json` · `pq/parts/*` · `CALIB_EVAL_WEBQSP__pq_v1.{json,npz}` (json bb0246d2…, npz 3fdb91d8…) · `enc/FBX_ENC_PROBE__v1.json` (c263da35…) · `enc/FBX_QWEN_ENCODING_CONTRACT__v1.json` (aef5fb58…); code `scratchpad/{_pq_knn,_pq_eval,_fbx_enc_probe,_fbx_encode}.py`. §52 (run-to-run floor): addendum 8 `…ADDENDUM_8.json` (47838cec…) · `h2l/parts/webqsp__H4_SK{R1,N3}_k{100,250,500}__PHG_con.{npy,RUN.json}` · `CALIB_EVAL_WEBQSP__noise_v1.{json,npz}` (json 1f3e45eb…, npz 782b2823…) · `h2l/H2LM_NETSIZE_SURVEY__webqsp__v1.json`; code `scratchpad/{_h2lm_noise,_h2lm_eval,_h2lm_stats}.py`. §53 (IVF over PQ64): addendum 10 (decf737e…) · `ivf/WEBQSP_IVF_SAMPLE__v1.json` (8b21c872…) · `ivf/WEBQSP_IVF_IVF_T{90,95,98}_KNN__v1.json` (99a68617…, b4162a7f…, bb4035f8…) · `ivf/parts/webqsp__H4_IVF_T{90,95,98}_k{100,250,500}__PHG_con.{npy,RUN.json}` · `CALIB_EVAL_WEBQSP__ivf_v1.{json,npz}` (json 69104ac7…, npz 77ec3f83…); code `scratchpad/{_ivf_knn,_ivf_phg,_ivf_eval,_ivf_declare}.py`. §54 (IVF transfer T1 / T1R): addenda 11 / 12 (00a5558a…, 01aee835…) · `ivf/FBX_IVF_TRANSFER_T1__v1.json` (1df3ca46…) · `ivf/FBX_IVF_TRANSFER_T1R__v1.json` (1eb11e99…); code `scratchpad/{_ivf_transfer,_ivf_transfer_declare,_ivf_rdiag,_ivf_rdiag_declare}.py`. §55 (TOP hypergraph thinning): addendum 9 (572a62c5…) · `h2l/parts/webqsp__H4_H2LT_<spec>_top_k25__PHG_con.*` · `CALIB_EVAL_WEBQSP__top_v1.{json,npz}` (775d9528…, cd70e751…); code `scratchpad/{_h2lt,_h2lt_run,_h2lt_subk,_h2lt_eval}.py`. §56 (NER family): `ner/FBX_NER_FAMILY__v1.json` (2034f8db…) · `ner/FBX_NER_SUPERSEDE__653b0410__v1.json` · `ner/FBX_NAME_TAIL__v1.json` · 64 `ner/group_b*.json` and 64 `ner/agg_r*.json`; code `scratchpad/{_fbx_ner_build,_fbx_ner_chain,_ner_unit_specs,_ner_supersede}.py`. §57 (IVF transfer T2): addendum 13 (3bb657c2…) · `ivf/FBX_IVF_TRANSFER_T2__v1.json` (0b70c3c0…); code `scratchpad/{_ivf_t2,_fbx_enc_order}.py`.
