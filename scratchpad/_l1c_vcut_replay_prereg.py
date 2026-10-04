"""Write results/L1_COVPART/PREREGISTRATION_STRUCT_VERTEXCUT_V1_REPLAY.json (tenth ruling, 2026-09-15) BEFORE any replay run.
Write-once: refuses if the record exists.  Pins the substrate (the capacity-repaired k* = 1004 cell), the served caches and the
frozen numerics modules by sha; fixes the home + overlap contract, the vote rule, the exposure definition, the cells, the
baselines, the statistics and the decision rule.  Nothing else is written."""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, "results", "L1_COVPART")
PDIR = os.path.join(OUT, "parts")
REC = os.path.join(OUT, "PREREGISTRATION_STRUCT_VERTEXCUT_V1_REPLAY.json")
if os.path.exists(REC):
    raise SystemExit("exists: %s" % REC)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pin(p):
    return {"path": os.path.relpath(p, REPO).replace("\\", "/"), "sha256": sha(p), "bytes": os.path.getsize(p)}


S = json.load(open(os.path.join(OUT, "vcut_struct_metaqa.json"), encoding="utf-8"))
cell = S["cells"]["vcut_k1004_CAP100"]
cap = {k: v for k, v in cell["capacity_repair"].items() if k != "file"}
K = 1004

RULING = ("For vertex-cut, keep the original section-18 verdict exactly as written: NOT_MATERIAL under that preregistration. Do not retroactively pass it. "
          "But the mechanism result is far too strong to close: 11.7x better 2-hop containment and ~2x the section-2 ball containment at RF ~2.34 is absolutely "
          "material enough to justify a new, corrected replay preregistration. The capacity failure was a specification mistake, not evidence against the "
          "representation. Your current hard partitions themselves operate under the PHG/MtK balance envelope up to about 104 nodes, so the next preregistration "
          "should use the actual served effective cap, not an impossible exact-100 cap. I would use the capacity-repaired k*=1004 cell as the primary substrate: "
          "STRUCT_VERTEXCUT_V1_REPLAY k = 1004, RF ~2.34, mean block size ~100.8, max block size = 104, 2-hop wedge containment = 0.2649. Do not use k=1296 as "
          "primary. Its ~81-node blocks change the granularity and confound 'overlap works' with 'we used substantially smaller blocks.' Keep it as a structural "
          "sensitivity point only. The important replay contract: The representation should now explicitly be home + overlap, so the existing pipeline still has a "
          "canonical owner: For node v, M(v) = {all vertex-cut parts containing v} and define a deterministic home: H(v) = argmax_p #{incident STRUCT edges of v "
          "assigned to p}. Tie-break by block ID. Then: home(v) = one canonical block; halo(v) = M(v) - {home(v)}. ... the partitioner decides from graph structure "
          "which boundaries should overlap, rather than us adding a generic 1-hop halo after hard partitioning. No queries. No labels. No training. For routing, "
          "reuse the already frozen hub philosophy. Non-hubs can vote to all vertex-cut memberships; pathological hubs should not spray into ~110 partitions. A hub "
          "can vote only to its deterministic home. That is the closest universal analogue of the A1 admissibility rule, and it is query-free. I would preregister "
          "both: P50 overlapping blocks and report the actual unique-node exposure. Also report a secondary matched-unique-node-budget result, because overlap means "
          "50 x block size is no longer equal to 5,000 unique nodes. Do not silently compensate by selecting extra blocks. If vertex-cut improves recall even while "
          "exposing fewer than the hard P50's unique nodes, that's an especially strong result. ... 1. Vertex-cut replay on MetaQA DEV_A as a new post-hoc "
          "structural-mechanism experiment. This is allowed because we're explicitly studying a new representation, not claiming confirmation.")

rec = {
    "RECORD": "PREREGISTRATION", "name": "STRUCT_VERTEXCUT_V1_REPLAY", "STATUS": "POSTHOC_STRUCTURAL_MECHANISM_EXPERIMENT_DEV_A",
    "written_before_any_run": True, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "ruling_2026-09-15_tenth_verbatim_essentials": RULING,
    "relation_to_section_18": "PREREGISTRATION_STRUCT_VERTEXCUT_V1.json and its verdict NOT_MATERIAL stand exactly as written (not retroactively passed). This is a NEW "
                              "pre-registration of a retrieval replay on the same, unchanged substrate file; the capacity clause is not re-litigated: the served effective "
                              "cap of the hard partitions (max block 104 under the PHG/MtK balance envelope) is the reference, and the capacity-repaired cell "
                              "(max |B_p| = 104) meets it.",
    "what_this_is": "A post-hoc mechanism experiment on MetaQA DEV_A: does the frozen dense + SPLADE partition router, re-run over the vertex-cut overlapping blocks "
                    "(home + overlap representation, hub-limited voting), serve more complete gold sets than the served hard P50 at no larger unique-node exposure? "
                    "DEV_A has been read many times in this lane -> no confirmation claim, no promotion, no DEV_B, no TEST. Structural mechanism only.",
    "substrate_frozen": {
        "dataset": "metaqa", "k": K, "cell": "vcut_k1004_CAP100 (k* = 1004, deterministic capacity repair C = 100 as pre-registered in section 18; unchanged file)",
        "edge_assignment_file": pin(os.path.join(PDIR, "metaqa__STRUCT_VCUT_V1_k1004__PHG_con__CAP100.npy")),
        "phg_output_unrepaired": pin(os.path.join(PDIR, "metaqa__STRUCT_VCUT_V1_k1004__PHG_con.npy")),
        "phg_run_record": pin(os.path.join(PDIR, "metaqa__STRUCT_VCUT_V1_k1004__PHG_con.RUN.json")),
        "dual_hypergraph": {fn: pin(os.path.join(PDIR, fn)) for fn in ("metaqa__STRUCT_VCUT_V1_k1004.npz", "metaqa__STRUCT_VCUT_V1_k1004.json")},
        "structural_record": pin(os.path.join(OUT, "vcut_struct_metaqa.json")), "section18_preregistration": pin(os.path.join(OUT, "PREREGISTRATION_STRUCT_VERTEXCUT_V1.json")),
        "graph": S["graph"],
        "cell_stats_from_section_18": {"RF": cell["RF"], "memberships_total": cell["memberships_total"], "block_size": cell["block_size"], "lambda_max": cell["lambda_max"],
                                       "lambda_mean_hub": cell["lambda_mean_hub"], "lambda_mean_nonhub": cell["lambda_mean_nonhub"],
                                       "wedge_containment_nonhub_middle": cell["wedge_2hop"]["containment_nonhub_middle"], "edge_containment_1hop": cell["edge_containment_1hop"],
                                       "capacity_repair": cap},
        "k1296": "NOT replayed (structural sensitivity point only, per the ruling: ~81-node blocks confound overlap with granularity)",
        "edge_index_convention": "z[e] for the e-th undirected STRUCT edge in ascending key order (keys u*N+v, u<v) exactly as _l1c_vcut_lib.Graph enumerates them (keys sha "
                                 "pinned in the structural record); B_p = {u, v : z[e] = p}",
    },
    "representation_contract_home_plus_overlap": {
        "M(v)": "the set of vertex-cut blocks containing v = {z[e] : e incident to v} (endpoint-set membership; lambda(v) = |M(v)|)",
        "H(v)": "deterministic home = argmax_p #{incident STRUCT edges of v assigned to p}; ties broken by the smallest block id",
        "halo(v)": "M(v) - {H(v)}",
        "degree_0_nodes": "M(v) = empty -> no home, never votes, never served; metaqa has 0 such nodes (asserted at run time); text corpora (squad 6,349 / musique 9,858 "
                          "degree-0 nodes) need a separate ruling and are NOT run here",
        "hub": "the frozen definition of the pinned beam / _l1c_vcut_lib.Graph: hub(v) <=> deg(v) + 1 > C with C = round(N / k_frozen) = 100 (104 hubs on metaqa); nothing tuned",
        "no_added_halo": "no 1-hop / out-neighbour expansion of any kind is added to the vertex-cut memberships: the partitioner alone decided the overlap",
    },
    "routing_contract_frozen_numerics": {
        "vote_table": "mem_vc = (ptr, flat) over the N nodes: non-hub v -> all of M(v); hub v -> {H(v)} only (a hub never sprays into its ~108 blocks)",
        "channels": "PR_d = _ta_prepartition.partition_ranking([dense top-K_LOCK ids], mem_vc, k); PR_s = the same with the SPLADE top-K_LOCK ids -- the served hits of the "
                    "replay cache's own rows (Data.d_ids / s_ids, first 200 asserted equal to the cache); vote 1/(K0 + r) per member block, S sum + M max, rr(S) + rr(M)",
        "fusion": "_l1s_core.rrf_ranks([PR_d, PR_s]) = _ta_prepartition.rrf_partitions (dense tie-break) -> full fused ranking over the k = 1004 blocks",
        "P50": "the first P_MAIN = 50 blocks of the fused ranking (the served rule 'T' applied to the fused ranking)",
        "constants_unchanged": {"K0": 60, "K_LOCK": 100, "P_MAIN": 50, "RRF": "frozen"},
        "served_hard_baseline_votes_through_the_frozen_legacy_table": "own block + blocks of the DIRECTED STRUCT out-neighbours (_l1s_core.Data.legacy_mem) -- exactly the "
                                                                       "served BASE; it is NOT modified.  The vertex-cut cell votes through mem_vc only.  A labelled diagnostic "
                                                                       "(HARD_MTK_OWN_BLOCK_VOTE) shows how much of the hard BASE comes from that vote expansion.",
    },
    "serving_and_exposure": {
        "served(g)": "gold node g is served by a selection Sel iff M(g) intersect Sel is non-empty (the block physically contains g; the hub vote limit does NOT limit serving)",
        "ALL": "every gold node of the query is served (the served node-level ALL; for a hard partition it equals the served block-level ALL -- asserted on both hard baselines)",
        "ANY": "at least one gold node served",
        "unique_exposure": "|union of B_p over the selected blocks| unique nodes per query (reported as the mean over DEV_A, plus the nominal sum |B_p|)",
        "hard_exposure": "sum of the selected hard block sizes = unique nodes (disjoint blocks) = the served scope_nodes",
    },
    "population": {"cache": "data/l1_canonical/metaqa/replay_cache.npz rows (served population); DEV_A = sha1(query_id)[:8] & 1 == 0 exactly as _l1x90_core", "DEV_B": "never read",
                   "TEST": "never read", "per_hop": "hop 1 / 2 / 3, hop 2+3 pooled, all"},
    "cells": {
        "VCUT_P50 (primary)": "50 fused vertex-cut blocks per query; ALL / ANY / per hop; unique exposure reported; McNemar vs HARD_MTK_BASE",
        "VCUT_MATCHED_LE (secondary)": "per query, the LARGEST fused prefix whose unique exposure <= the HARD_MTK_BASE P50 exposure of the same query (never exceeds the hard "
                                       "budget; blocks used and exposure reported); McNemar vs HARD_MTK_BASE",
        "VCUT_MATCHED_GE (informational only)": "the SMALLEST fused prefix whose unique exposure >= the hard exposure (overshoot reported); not a decision cell",
        "VCUT_HOME_ONLY_VOTE (labelled diagnostic)": "all nodes vote only to H(v); serving unchanged (M(g)); P50; not a decision cell",
        "HARD_MTK_OWN_BLOCK_VOTE (labelled diagnostic)": "the served hard MtK partition voted through the own-block table (mem_hard) instead of the legacy table; P50; not a decision cell",
        "reach_decomposition (diagnostic)": "gold node g is EVIDENCED iff some block of M(g) received >= 1 vote from either channel's top-K_LOCK hits through mem_vc; a query is "
                                            "REACHED iff every gold node is evidenced; failures split REACHED_WEAK vs UNREACHED; worst gold position = max over gold nodes of the "
                                            "best fused position among M(g); hub gold nodes counted",
    },
    "baselines": {
        "HARD_MTK_BASE": "the served replay BASE of the canonical H4_SK Mt-KaHyPar partition (432 blocks; legacy vote table; P50) = Data('metaqa').base_all",
        "HARD_PHG_BASE": "the served replay BASE of the metaqa PHG+repair partition (metaqa_phg cache; same rows asserted) -- contrast only, not part of the rule",
    },
    "statistics": {"test": "exact two-sided McNemar on paired DEV_A 0/1 ALL vectors (_l1x90_core.mcnemar): gained / lost / p", "alpha": 0.01, "no_multiplicity_note":
                   "two decision cells (primary, secondary) each at alpha = 0.01; every other number is descriptive"},
    "decision_rule_pre_registered": {
        "MECHANISM_POSITIVE_AT_LOWER_EXPOSURE": "VCUT_P50 vs HARD_MTK_BASE on DEV_A: gained > lost with p < 0.01 AND mean unique exposure of VCUT_P50 <= mean exposure of HARD_MTK_BASE",
        "MECHANISM_POSITIVE_AT_MATCHED_BUDGET": "otherwise, VCUT_MATCHED_LE vs HARD_MTK_BASE: gained > lost with p < 0.01",
        "MECHANISM_POSITIVE_ONLY_WITH_MORE_EXPOSURE": "VCUT_P50 passes the significance clause but exposes more unique nodes than the hard P50 AND VCUT_MATCHED_LE does not pass",
        "NEGATIVE": "none of the above",
        "what_a_positive_means": "a structural-mechanism result on DEV_A (post hoc), reported as such; it is NOT a confirmation, NOT promotable, and does not change the served L1",
        "PHG_contrast": "reported (VCUT_P50 vs HARD_PHG_BASE), no verdict",
    },
    "forbidden": ["reading DEV_B or TEST", "any change to K0 / K_LOCK / P_MAIN / the RRF / the hub definition / the tie-break", "selecting more than 50 blocks except in the "
                  "labelled matched-budget cells defined above", "any SAFE / H2_PATCH1 / beam combination", "k = 1296 or any other k as a replay cell", "any halo or "
                  "out-neighbour expansion added to the vertex-cut memberships", "re-partitioning, re-repairing or editing the substrate file", "editing any CONTRACT_FILE or "
                  "pinned module", "writing under data/", "tuning after seeing numbers; a second variant of any rule"],
    "code": {"replay": "scratchpad/_l1c_vcut_replay.py (new module; pinned by sha in the result record) -> results/L1_COVPART/vcut_replay_A_metaqa.json + .log",
             "imports_unchanged": {"_l1s_core.py": pin(os.path.join(HERE, "_l1s_core.py")), "_l1x90_core.py": pin(os.path.join(HERE, "_l1x90_core.py")),
                                   "_ta_prepartition.py": pin(os.path.join(HERE, "_ta_prepartition.py")), "_l1g_core.py": pin(os.path.join(HERE, "_l1g_core.py")),
                                   "_l1c_vcut_lib.py": pin(os.path.join(HERE, "_l1c_vcut_lib.py"))}},
    "caches_read_only": {"metaqa": pin(os.path.join(REPO, "data", "l1_canonical", "metaqa", "replay_cache.npz")),
                         "metaqa_phg": pin(os.path.join(REPO, "data", "l1_lowmem", "metaqa", "replay_cache__LOWMEM__PHG_REPAIR1_con.npz"))},
}
with open(REC, "w", encoding="utf-8", newline="\n") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print("written", os.path.relpath(REC, REPO), sha(REC)[:16], "utc", rec["utc"])
