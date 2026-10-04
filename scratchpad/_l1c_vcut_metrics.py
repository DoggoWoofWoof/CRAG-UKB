"""STRUCT_VERTEXCUT_V1 step 3: the pre-registered query-free structural comparison (PREREGISTRATION_STRUCT_VERTEXCUT_V1.json).

Cells (all on the same STRUCT graph, the same metrics):
    hard_MtK / hard_PHG         the served hard partitions (replay-cache 'hard' vectors; k_f blocks of ~C nodes)      -- the served exposure unit
    legacy_MtK / legacy_PHG     the served VOTING unit: own block + blocks of the directed out-neighbours (unmatched, larger)
    halo_MtK / halo_PHG         B_p + N(B_p), the DistDGL-style undirected halo (unmatched, larger)
    PATH2_PHG                   the section 3/4 higher-order hard construction where it exists
    vcut_k<k>                   STRUCT_VERTEXCUT_V1 endpoint sets for the k grid and k* (raw PHG partitions)
    vcut_k<k*>_CAP<C>           the k* partition after the deterministic capacity repair (strict |B_p| <= C where achievable)
    python -u _l1c_vcut_metrics.py <ds>   -> results/L1_COVPART/vcut_struct_<ds>.json (+ parts/<ds>__STRUCT_VCUT_V1_k<k*>__PHG_con__CAP<C>.npy)
"""
import json
import os
import sys
import time

import numpy as np

import _l1g_core as G  # noqa: F401  (registers the squad_phg cache path)
import _l1c_vcut_lib as V

X = V.X
ds = sys.argv[1]
t0 = time.time()
Gr = V.Graph(ds)
C = Gr.C
stride = 1 if Gr.E <= V.PATH3_EXACT_MAX else int(np.ceil(Gr.E / float(V.PATH3_SAMPLE)))
SERVED = {"metaqa": [("hard_MtK", "metaqa"), ("hard_PHG", "metaqa_phg")], "squad": [("hard_MtK", "squad"), ("hard_PHG", "squad_phg")], "musique": [("hard_PHG", "musique")]}[ds]
ks = json.load(open(os.path.join(V.OUT, "vcut_kstar_%s.json" % ds), encoding="utf-8"))
assert ks["C"] == C
k_star = ks["k_star"]
cells = {}
pre = json.load(open(os.path.join(V.OUT, "PREREGISTRATION_STRUCT_VERTEXCUT_V1.json"), encoding="utf-8"))


def run(label, Y, **kw):
    cells[label] = V.structure_metrics(Gr, Y, stride, label, **kw)


hards = {}
for label, cache in SERVED:
    z = np.load(X.CACHES[cache], allow_pickle=True)
    hard = z["hard"].astype(np.int64)
    assert len(hard) == Gr.N and hard.max() + 1 == Gr.k_f
    hards[label] = hard
    run(label, V.Y_hard(Gr, hard))
    run(label.replace("hard", "legacy"), V.Y_legacy_membership(Gr, hard))
    run(label.replace("hard", "halo"), V.Y_undirected_halo(Gr, hard))
p2 = os.path.join(V.PDIR, "%s__H4_SK_PATH2__PHG_con.npy" % ds)
if os.path.exists(p2):
    hard = np.load(p2).astype(np.int64)
    assert len(hard) == Gr.N
    run("PATH2_PHG", V.Y_hard(Gr, hard))
runs = {}
for s in ks["sequence"]:
    k = int(s["k"])
    z, meta, rr = V.load_vcut(ds, k)
    runs["vcut_k%d" % k] = {"k": k, "role": s["role"], "hypergraph": meta["file"], "hypergraph_sha256": meta["file_sha256"], "phg_km1": rr["raw"]["km1"],
                            "phg_wall_seconds": rr["run"]["wall_seconds_outer"], "phg_peak_rss_mb_per_rank": [round(x / 1024.0) for x in rr["run"]["memory"]["peak_rss_kb_per_rank_time_v"]],
                            "empty_block_repair_moves": (rr["repair"] or {}).get("moves", 0)}
    run("vcut_k%d" % k, V.Y_vcut(Gr, z, k))
# strict-capacity cell: the deterministic repair on k*
z, meta, rr = V.load_vcut(ds, k_star)
zc, rep = V.capacity_repair(Gr, z, k_star, C)
capf = os.path.join(V.PDIR, "%s__%s_k%d__PHG_con__CAP%d.npy" % (ds, V.TAG, k_star, C))
np.save(capf, zc.astype(np.int64))
rep["file"] = os.path.relpath(capf, X.REPO).replace("\\", "/")
V.log("  capacity repair k* %d: %s" % (k_star, json.dumps(rep)))
run("vcut_k%d_CAP%d" % (k_star, C), V.Y_vcut(Gr, zc, k_star))
cells["vcut_k%d_CAP%d" % (k_star, C)]["capacity_repair"] = rep

# ---------------------------------------------------------------- the pre-registered structural verdict (metaqa primary; every dataset reported)
served_hard = cells["hard_MtK"] if "hard_MtK" in cells else cells["hard_PHG"]
cap_cell = cells["vcut_k%d_CAP%d" % (k_star, C)]
w_h, w_v = served_hard["wedge_2hop"]["containment_nonhub_middle"], cap_cell["wedge_2hop"]["containment_nonhub_middle"]
b_h, b_v = served_hard["ball2_section2"]["intact_fraction"], cap_cell["ball2_section2"]["intact_fraction"]
residual_ok = cap_cell["block_size"]["blocks_gt_C"] <= 0.01 * k_star
verdict = {"served_reference": served_hard["label"], "strict_capacity_cell": cap_cell["label"],
           "wedge_nonhub_served": w_h, "wedge_nonhub_vcut_CAP": w_v, "ratio": round(w_v / max(w_h, 1e-9), 2), "gain_pts": round(100 * (w_v - w_h), 1),
           "RF_vcut_CAP": cap_cell["RF"], "residual_overfull_blocks": cap_cell["block_size"]["blocks_gt_C"], "residual_ok (<= 1 % of blocks)": bool(residual_ok),
           "ball2_served": b_h, "ball2_vcut_CAP": b_v, "ball2_ratio": round(b_v / max(b_h, 1e-9), 2),
           "rule": pre["decision_rule_structural_only"]["MATERIAL"]}
verdict["MATERIAL"] = bool(residual_ok and w_v >= 2 * w_h and (w_v - w_h) >= 0.20 and cap_cell["RF"] <= 3.0)
verdict["secondary_ball2_2x"] = bool(b_v >= 2 * b_h)
verdict["verdict"] = "MATERIAL" if verdict["MATERIAL"] else "NOT_MATERIAL"
res = {"RECORD": "STRUCT_VERTEXCUT_V1_STRUCTURAL_DIAGNOSTIC", "dataset": ds, "primary": ds == "metaqa", "prereg": "PREREGISTRATION_STRUCT_VERTEXCUT_V1.json",
       "graph": {"N": Gr.N, "struct_edges": Gr.E, "k_frozen": Gr.k_f, "C": C, "hubs": int(Gr.hub.sum()), "nodes_degree0": int((Gr.deg == 0).sum()),
                 "nodes_degree1": int((Gr.deg == 1).sum()), "struct_keys_sha256": Gr.keys_sha, "path3_sample_stride": stride},
       "k_star": ks, "vcut_runs": runs, "cells": cells, "verdict": verdict, "seconds": round(time.time() - t0, 1)}
V.S.wj(os.path.join(V.OUT, "vcut_struct_%s.json" % ds), res)
V.log("VERDICT %s: %s" % (ds, json.dumps(verdict)))
V.log("done %.0fs" % (time.time() - t0))
