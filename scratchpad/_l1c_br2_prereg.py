"""Write-once pre-registration for BOUNDED_VERTEXCUT_R2 (twelfth ruling, 2026-09-15): the bounded-overlap static substrate (every node
in at most R = 2 blocks, block cap = the lane's existing C = 100, k = R * k_f, one construction on every graph, no density switch, no
query information) replayed as a routing BASE against each dataset's canonical hard BASE on DEV_A.  Written before any owner
hypergraph, any PHG owner run, any degree-0 placement, any alternate assignment and any replay number exists.

    python -u scratchpad/_l1c_br2_prereg.py   -> results/L1_COVPART/PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1s_core as S  # noqa: E402
import _l1g_core  # noqa: E402,F401  (registers the squad_phg cache)
import _l1c_vcut_lib as V  # noqa: E402
import _l1c_br2_lib as L  # noqa: E402
from src.l1_lowmem import phg as P  # noqa: E402

X = S.X
OUT = os.path.join(X.REPO, "results", "L1_COVPART")
PDIR = os.path.join(OUT, "parts")
FP = os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json")
assert not os.path.exists(FP), "write-once"
DS = ("metaqa", "squad", "musique")
assert not any("BR2" in f for f in os.listdir(PDIR)), "a BR2 substrate file exists already"
assert not any(f.startswith("br2_") for f in os.listdir(OUT)), "a BR2 result exists already"
assert not os.path.exists(os.path.join(X.REPO, "results", "L1_COVPART", "phg_data", "metaqa", "stream_BR2_OWNERS_k864"))
sha_file, pin = L.sha_file, L.pin

graphs = {}
for ds in DS:
    Gr = V.Graph(ds)
    graphs[ds] = {"N": int(Gr.N), "struct_edges": int(Gr.E), "k_f": int(Gr.k_f), "C": int(Gr.C), "k_R2": L.k_of(Gr), "owners_tag": L.owners_tag(Gr), "hubs": int(Gr.hub.sum()),
                  "degree_0": int((Gr.deg == 0).sum()), "struct_nodes": int((Gr.deg >= 1).sum()), "mean_struct_degree": round(float(Gr.deg.mean()), 2),
                  "nodes_with_degree_gt_2(C-1) (cannot have every edge contained under lambda <= 2 and cap C)": int((Gr.deg > 2 * (Gr.C - 1)).sum()),
                  "owners_per_block_mean": round(float((Gr.deg >= 1).sum()) / L.k_of(Gr), 2), "struct_keys_sha256": Gr.keys_sha}
    del Gr
assert all(g["C"] == 100 for g in graphs.values())

lib_doc = open(os.path.join(HERE, "_l1c_br2_lib.py"), encoding="utf-8").read().split('"""')[1]

RULING = (
    "This result is actually very useful because it tells us why the same vertex-cut construction cannot be the universal replacement as currently "
    "formulated. For MetaQA, overlap is informative. For SQuAD/MuSiQue, overlap is mostly redundant ... RF_MetaQA ~ 2.34 versus RF_SQuAD ~ 10.35, "
    "RF_MuSiQue ~ 9.25 ... Once you restore the same unique-node exposure, VCUT becomes statistically level with the hard substrate. So there is no free "
    "structural signal there. That means I would freeze: STRUCT_VERTEXCUT_V1: MetaQA mechanism positive / Universal replacement NOT_SUPPORTED and "
    "definitely not compose this V1 with SAFE/H2_PATCH1. But I would not conclude that overlap-aware partitioning is dead. The actual failure is more "
    "specific: we optimized overlap without placing a budget on overlap itself. That is the next partitioning problem. The natural successor is "
    "bounded-overlap partitioning ... 1 <= |M(v)| <= R with a small universal R. The most defensible first value is R = 2 ... R=1 -> hard partition; R=2 "
    "-> owner + one alternate boundary region ... min sum_v (lambda_v - 1) subject to lambda_v <= 2 plus balance/capacity. Or equivalently, allow each "
    "original node's incident edges to be distributed over at most two edge partitions ... The universal algorithm is identical everywhere. No "
    "if dataset == 'metaqa'. No graph-density threshold. No query information ... one representation rule across every graph ... replication needs to be "
    "a first-class constraint, not an after-the-fact diagnostic ... A. BOUNDED_VERTEXCUT_R2: query-independent structural substrate: each node belongs "
    "to <= 2 overlapping regions; block cap = existing effective cap; same rule all datasets. Question: Can we retain MetaQA's VCUT locality gain while "
    "preventing RF explosion on dense text graphs? This is the static-L1 workstream ... I would not execute either on MetaQA immediately without a clean "
    "preregistration ... we need to represent the graph under a controlled overlap budget and compress the dynamic frontier without destroying branch diversity.")

rec = {
    "RECORD": "PREREGISTRATION",
    "name": "BOUNDED_VERTEXCUT_R2",
    "STATUS": "POSTHOC_STRUCTURAL_MECHANISM_TEST_DEV_A",
    "written_before_any_run": True,
    "written_before": ["any owner hypergraph", "any PHG owner run", "any degree-0 placement on the owner partition", "any alternate assignment", "any structural number of the R2 substrate", "any replay number"],
    "what_was_run_before_this_record": "the library's alternate / membership / containment functions were unit-tested on two synthetic 6-8 node graphs (no data read); V.Graph was read here for the frozen graph facts (N, |E|, k_f, C, hubs, degree-0) that already stand in sections 18-20",
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ruling_2026-09-15_twelfth_verbatim_essentials": RULING,
    "question": "Can a bounded-overlap static substrate (every node in <= 2 blocks, cap C = 100, the same construction on every graph) retain MetaQA's vertex-cut locality gain (section 19: 0.640 -> 0.678 at P50, 0.688 at the matched budget) while preventing the replication explosion that made the unbounded vertex-cut a budget cut on the dense text graphs (section 20: RF 10.35 / 9.25, P50 = 43-63 % of the hard unique exposure, musique LOSS at P50)?",
    "relation_to_sections_18_19_20": {
        "section_18_19": "STRUCT_VERTEXCUT_V1 = NOT_MATERIAL (structural letter) / MECHANISM_POSITIVE_AT_LOWER_EXPOSURE (metaqa replay); frozen by the twelfth ruling as 'MetaQA mechanism positive / universal replacement NOT_SUPPORTED'; never composed with SAFE / H2_PATCH1; untouched here (its metaqa cells are recomputed only to be asserted equal and compared informationally)",
        "section_20": "STRUCT_VERTEXCUT_V1_TRANSFER = NOT_SUPPORTED; its universal degree-0 rule and its routing / serving / exposure / matched-budget contracts (imported functions) are reused VERBATIM; its gate design (each dataset's canonical hard BASE, node-level check, matched exposure reported separately) is reused with ONE stated difference: the matched-exposure cell is the SIGNAL cell here (see decision rule) because section 20 established that P50 is not a fixed unique budget once RF > 1",
        "this_record": "a new construction (owners + at most one alternate under a cap) with replication bounded BY CONSTRUCTION; a new arm, not a modification of V1",
    },
    "what_this_is": "a post-hoc structural-mechanism test on DEV_A of metaqa, squad and musique: does the R = 2 bounded-overlap BASE (static overlap decided by a deterministic construction, no 1-hop vote expansion, no query information) reach or beat each dataset's served canonical hard BASE at the same unique-node budget, and at what P50 exposure. DEV_A has been read many times in this lane; DEV_B and TEST are never read; nothing is promoted; the served L1 is unchanged.",
    "constants": {"R": L.R_MAX, "C": 100, "k_rule": "k = R * k_f (k_f = N // 100 frozen by the lane): k blocks of cap C hold R * N memberships; k is derived, not chosen per dataset",
                  "k_per_dataset": {ds: graphs[ds]["k_R2"] for ds in DS}, "K0": S.K0, "K_LOCK": S.K_LOCK, "P_MAIN": S.P_MAIN, "alpha": 0.01,
                  "PHG": {"NP": P.NP, "parameters": P.PARAMS, "imbalance": "1.03 (validity bound ceil(1.03 N_obj / k) on the owner blocks)", "objective": "CONNECTIVITY (= edge cut on 2-pin nets)", "repair": "the pre-registered non-empty repair of src/l1_lowmem/phg_repair.py, unchanged"},
                  "hub": "deg + 1 > C (frozen definition; hubs vote through H(v) only)"},
    "construction_universal_verbatim": lib_doc,
    "construction_steps": {
        "1_owner_hypergraph": "scratchpad/_l1c_br2_build.py <ds>: objects = STRUCT nodes of degree >= 1 (re-indexed ascending), nets = the STRUCT edges (2 pins each, unit weight), k = R * k_f -> parts/<ds>__BR2_OWNERS_k<k>.npz + .json (file sha, content digest)",
        "2_owner_partition": "python -u scratchpad/_l1c_phg.py <ds> BR2_OWNERS_k<k> (the pinned driver: shards, Zoltan-PHG NP = 4, validity bound, non-empty repair) -> parts/<ds>__BR2_OWNERS_k<k>__PHG_con.npy + .RUN.json; ONE run per dataset, whatever its cut; no k retarget, no second seed",
        "3_homes": "H(v) = owner block for every STRUCT node; degree-0 nodes by the section-20 rule (scratchpad/_l1c_vcut_transfer.place_degree0 imported unchanged: one home = H(nearest anchored frozen-KNN neighbour), anchored = d_STRUCT > 0, highest float32 cosine, ties -> smallest node id; else the currently smallest block in ascending canonical node id; live sizes) applied ONCE on the owner sizes",
        "4_alternates": "scratchpad/_l1c_br2_lib.alternates: one static greedy pass -- gain(v, p) = number of v's STRUCT neighbours whose HOME is the foreign block p; candidate list of v = its foreign blocks by gain descending, block id ascending; nodes processed by best gain descending, node id ascending, only nodes with gain >= 1; v takes the first candidate block whose live size (homes + degree-0 placements + alternates so far) < C; otherwise refused (lambda = 1). Degree-0 nodes have no STRUCT neighbour and never get an alternate (no fake structural overlap for isolated nodes, as in section 20)",
        "5_membership": "M(v) = {H(v)} or {H(v), alt(v)}; lambda(v) <= 2 and |B_p| <= C by construction (asserted); RF = mean lambda <= 2 (asserted); parts/<ds>__BR2_k<k>__R2.npz (home, alt) + .json record",
        "6_replay": "scratchpad/_l1c_br2_replay.py <ds> with the section-19 / 20 functions: mem_vc (non-hub -> M(v), hub -> {H(v)}), PR_d / PR_s over the served dense / SPLADE top-100 ids of the replay cache, RRF fusion, served(g) iff M(g) meets the selection, unique exposure = |union B_p|, matched prefixes vs the gate's P50 exposure",
        "one_code_path": "the dataset name only selects the frozen graph, the served caches, the KNN file and the output names; no branch on dataset, density, degree or hub share anywhere in the construction",
    },
    "objective_and_infeasibility_statement": {
        "ruling_objective": "min sum_v (lambda_v - 1) s.t. lambda_v <= 2 + balance / capacity, or equivalently each node's incident edges over at most two edge partitions",
        "why_the_edge_form_is_infeasible_under_a_node_cap": "with |B_p| <= C and lambda_v <= 2, a node of STRUCT degree d can share a block with at most 2 (C - 1) neighbours; every node with d > 2 (C - 1) = 198 therefore keeps cut edges whatever the partition (metaqa %d, squad %d, musique %d such nodes), and on the dense text graphs (mean degree %.1f / %.1f) the all-edges-covered requirement is far out of reach for ordinary nodes too. The cut edges are the residual of the construction, not a violation" % (
            graphs["metaqa"]["nodes_with_degree_gt_2(C-1) (cannot have every edge contained under lambda <= 2 and cap C)"], graphs["squad"]["nodes_with_degree_gt_2(C-1) (cannot have every edge contained under lambda <= 2 and cap C)"],
            graphs["musique"]["nodes_with_degree_gt_2(C-1) (cannot have every edge contained under lambda <= 2 and cap C)"], graphs["squad"]["mean_struct_degree"], graphs["musique"]["mean_struct_degree"]),
        "objective_realised": "maximise the number of STRUCT edges contained by a shared block subject to lambda <= 2 and |B_p| <= C (owners by the partitioner's edge cut; the second membership spent greedily where it contains the most home-edges), replication bounded by construction rather than minimised as a soft term: R = 2 IS the overlap budget of the ruling",
        "not_optimised_here": "short-path (2-hop) containment per unit of replication -- reported (metaqa wedge / path-3 / ball metrics of section 18), not optimised; no second alternate, no swap pass, no iteration",
    },
    "representation_routing_serving_exposure": {
        "identical_to_sections_19_20": "M(v), H(v), hub rule, mem_vc, PR_d / PR_s = _ta_prepartition.partition_ranking over the served top-K_LOCK ids, _l1s_core.rrf_ranks, P50 = the first P_MAIN fused blocks, served(g) iff M(g) meets the selection, node-level ALL / ANY, unique exposure, matched prefixes -- via the imported, pinned _l1c_vcut_replay functions",
        "no_added_halo": "no 1-hop / out-neighbour expansion of any kind; the served hard BASE keeps its frozen legacy vote table exactly as served",
        "node_level_check": "the node-level ALL through each served hard partition's own-block table must equal the served block-level base_all (asserted)",
    },
    "population": {ds: "served %s cache rows; DEV_A = sha1(query_id)[:8] & 1 == 0 exactly as _l1x90_core; hop labels as present in the cache (metaqa per hop; musique per hop where present; squad none)" % ds for ds in DS} | {"DEV_B": "never read", "TEST": "never read"},
    "baselines_gate": {
        "metaqa": {"HARD_MTK_BASE (GATE)": "served `metaqa` cache = canonical H4_SK Mt-KaHyPar partition (k_f = 432), served P50 through the frozen legacy vote table", "HARD_PHG_BASE (contrast only)": "served `metaqa_phg` cache; reported, no verdict"},
        "squad": {"HARD_MTK_BASE (GATE)": "served `squad` cache (k_f = 202)", "HARD_PHG_BASE (contrast only)": "served `squad_phg` cache"},
        "musique": {"HARD_PHG_BASE (GATE)": "served `musique` cache = LOWMEM PHG_C1 partition (k_f = 1,175), the dataset's only served hard BASE"},
        "matched_budget_target": "per query, the GATE baseline's P50 exposure (sum of its selected hard block sizes = unique nodes)",
    },
    "cells_per_dataset": {
        "R2_P50 (protocol cell)": "50 fused R2 blocks per query; ALL / ANY (per hop where labelled); unique and nominal exposure; McNemar vs the GATE (and vs the contrast, no verdict); exposure flag EXPOSURE_LE_HARD / EXPOSURE_ABOVE_HARD by the mean unique exposure over DEV_A vs the gate's mean P50 exposure",
        "R2_MATCHED_LE (SIGNAL cell)": "per query, the LARGEST fused prefix whose unique exposure <= the GATE's P50 exposure of the same query (never exceeds the hard budget); McNemar vs the GATE. Signal cell because the question is about the representation at an equal unique-node budget; P50 compares different budgets once RF > 1 (section 20)",
        "R2_MATCHED_GE (informational)": "the smallest prefix with unique exposure >= the hard exposure; overshoot reported",
        "R2_HOME_ONLY_VOTE (diagnostic)": "every node votes only to H(v); serving through M(v); P50",
        "OWNERS_ONLY_P50 / OWNERS_ONLY_MATCHED_LE (R = 1 control)": "the same owner partition (k = 2 k_f, degree-0 placed) WITHOUT alternates: home vote, home serving, owner blocks for exposure; at P50 (about half the hard budget) and at the matched budget. R2_MATCHED_LE vs OWNERS_ONLY_MATCHED_LE = the alternates' value at equal exposure and equal owners (informational label GAIN / NEUTRAL / LOSS)",
        "HARD_OWN_BLOCK_VOTE (diagnostic)": "the GATE hard partition voted through its own-block table; P50",
        "V1 cells (metaqa only, informational)": "the section-19 VCUT_P50 and VCUT_MATCHED_LE cells recomputed from the pinned CAP file with the pinned functions, asserted equal to vcut_replay_A_metaqa.json, then R2 vs V1 at P50 and at the matched budget (McNemar; no decision)",
        "reach_decomposition (diagnostic)": "as sections 19 / 20 for R2_P50 (UNREACHED / REACHED_WEAK, worst gold position, hub / degree-0 gold); failure overlap with the gate for R2_P50 and R2_MATCHED_LE",
    },
    "structure_metrics": {"full_on": ["metaqa"], "path3_stride": 1, "cheap_on": ["squad", "musique"],
                          "note": "the section-18 wedge / path-3 / ball metrics (V.structure_metrics) are computed for the R2 membership and for the owners-only membership on metaqa (the only graph with section-18 reference cells: hard_MtK wedge_all 0.0011, vcut_k1004_CAP100 0.0212 / RF 2.34); on the text graphs only RF, block sizes and 1-hop containment (the wedge enumeration is not affordable on the contended host and gates nothing)"},
    "statistics": {"test": "exact two-sided McNemar on paired DEV_A 0/1 ALL vectors (_l1x90_core.mcnemar): gained / lost / p", "alpha": 0.01,
                   "labels": {"GAIN": "gained > lost and p < 0.01", "LOSS": "lost > gained and p < 0.01", "NEUTRAL": "otherwise"},
                   "multiplicity_note": "one signal cell (R2_MATCHED_LE vs gate) per dataset decides; R2_P50 vs gate gives the efficiency flag; every other comparison is descriptive"},
    "decision_rule_pre_registered": {
        "metaqa": {"RETAINED": "R2_MATCHED_LE vs HARD_MTK_BASE = GAIN (the section-19 locality gain survives the R = 2 bound at the equal budget)",
                   "EFFICIENT": "R2_P50 vs HARD_MTK_BASE != LOSS AND EXPOSURE_LE_HARD"},
        "text (squad, musique)": {"SAFE": "R2_MATCHED_LE vs the GATE != LOSS (no significant ALL-gold loss at the equal budget)",
                                  "EFFICIENT": "R2_P50 vs the GATE != LOSS AND EXPOSURE_LE_HARD (the 50-block protocol cell is not a budget cut that costs gold)"},
        "universal_outcome": {"SUPPORTED": "metaqa RETAINED AND squad SAFE AND musique SAFE AND squad EFFICIENT AND musique EFFICIENT",
                              "PARTIALLY_SUPPORTED": "metaqa RETAINED AND both text SAFE, but at least one text dataset not EFFICIENT",
                              "NOT_SUPPORTED": "otherwise (metaqa not RETAINED, or a text LOSS at the matched budget)"},
        "difference_from_section_20_stated": "section 20 labelled the P50 cell primary and the matched cell secondary; here the matched cell is the signal because the bounded construction is a statement about the representation at a fixed replication budget, and section 20 showed the P50 comparison mixes representation with budget; the P50 cell keeps its exposure flag as the efficiency clause",
        "what_follows": "SUPPORTED / PARTIALLY_SUPPORTED = post-hoc DEV_A evidence that a bounded-overlap substrate retains the sparse-graph locality gain without the text budget cut; composition with SAFE / H2_PATCH1 / BALANCED_H2_PATCH1 needs a further ruling and is NOT run here. NOT_SUPPORTED = the R = 2 construction as specified does not carry the gain (the owners-only control and the containment / wedge numbers say whether the alternates were inert or the owners too small); no rule change, no R = 3, no second alternate without a ruling",
        "no_promotion": "nothing here changes the served L1 or any frozen record; sections 18-20 are unchanged",
    },
    "forbidden": [
        "reading DEV_B or TEST",
        "any change to R, C, the k rule, K0 / K_LOCK / P_MAIN, the RRF, the hub definition, the degree-0 rule, the greedy's gain / order / tie-breaks, the capacity rule",
        "a second alternate, a swap or refinement pass, a re-run of the owner partition with another k / seed / imbalance, a re-repair after placement, a density or degree switch, a dataset branch",
        "selecting more than 50 blocks except in the labelled matched-budget cells",
        "any SAFE / H2_PATCH1 / BALANCED_H2_PATCH1 / H3 / V1 composition",
        "any halo or out-neighbour expansion added to the memberships",
        "editing any CONTRACT_FILE or pinned module; editing _l1c_vcut_replay.py, _l1c_vcut_lib.py, _l1c_vcut_transfer.py, _l1c_phg.py; editing the five new BR2 modules after this record (their shas are pinned below and asserted at run time)",
        "changing NP or the PHG parameters; signalling the foreign process; touching the WebQSP watcher",
        "writing under data/",
        "tuning after seeing numbers; changing a label, cell or threshold after a number is seen; re-running a step after seeing its number",
    ],
    "code": {
        "new_modules": {n: pin(os.path.join(HERE, n)) for n in ("_l1c_br2_lib.py", "_l1c_br2_build.py", "_l1c_br2_assign.py", "_l1c_br2_replay.py", "_l1c_br2_summary.py")},
        "usage": ["python -u scratchpad/_l1c_br2_build.py <ds>", "python -u scratchpad/_l1c_phg.py <ds> BR2_OWNERS_k<k>", "python -u scratchpad/_l1c_br2_assign.py <ds>", "python -u scratchpad/_l1c_br2_replay.py <ds>", "python -u scratchpad/_l1c_br2_summary.py"],
        "imports_unchanged": {**{n: pin(os.path.join(HERE, n)) for n in ("_l1s_core.py", "_l1x90_core.py", "_ta_prepartition.py", "_l1g_core.py", "_l1c_vcut_lib.py", "_l1c_vcut_replay.py", "_l1c_vcut_transfer.py", "_l1c_phg.py")},
                              **{"src/l1_lowmem/%s.py" % n: pin(os.path.join(X.REPO, "src", "l1_lowmem", n + ".py")) for n in ("phg", "phg_repair", "common", "freight")},
                              **{"src/l1_canonical/%s.py" % n: pin(os.path.join(X.REPO, "src", "l1_canonical", n + ".py")) for n in ("adapter", "hypergraph")}},
        "phg_driver_build_record": pin(os.path.join(X.REPO, "results", "L1_LOWMEM", "PHG_BUILD.json")) if os.path.exists(os.path.join(X.REPO, "results", "L1_LOWMEM", "PHG_BUILD.json")) else None,
    },
    "graphs_frozen": graphs,
    "records_read_only": {n: pin(os.path.join(OUT, n)) for n in ("vcut_replay_A_metaqa.json", "vcut_struct_metaqa.json", "vcut_transfer_A_squad.json", "vcut_transfer_A_musique.json", "vcut_transfer_SUMMARY.json",
                                                              "PREREGISTRATION_STRUCT_VERTEXCUT_V1.json", "PREREGISTRATION_STRUCT_VERTEXCUT_V1_REPLAY.json", "PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json")},
    "v1_substrate_metaqa": pin(os.path.join(PDIR, "metaqa__STRUCT_VCUT_V1_k1004__PHG_con__CAP100.npy")),
    "caches_read_only": {n: pin(X.CACHES[n]) for n in ("metaqa", "metaqa_phg", "squad", "squad_phg", "musique")},
    "knn_read_only": {ds: pin(os.path.join(X.REPO, "data", "final_canonical", ds, "graph", "knn.npz")) for ds in DS},
}
S.wj(FP, rec)
print("written", os.path.relpath(FP, X.REPO), sha_file(FP), os.path.getsize(FP), rec["utc"])
