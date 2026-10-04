"""BOUNDED_VERTEXCUT_R2 step 3 (twelfth ruling): from the PHG owner partition of the primal STRUCT graph make the bounded-overlap
membership -- (1) home of every STRUCT node = its owner block; (2) the frozen universal degree-0 rule of section 20
(_l1c_vcut_transfer.place_degree0, imported unchanged: one home = H(nearest anchored frozen-KNN neighbour) else the currently smallest
block in canonical node order); (3) at most ONE alternate block per node by the single static greedy pass of _l1c_br2_lib.alternates
(live sizes, cap C).  Writes the membership (home, alt) and a structural record; nothing under data/ is touched.

    python -u scratchpad/_l1c_br2_assign.py <ds>   -> parts/<ds>__BR2_k<k>__R2.npz + .json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1c_br2_lib as L  # noqa: E402
import _l1c_vcut_lib as V  # noqa: E402
import _l1c_vcut_transfer as T  # noqa: E402  (section-20 degree-0 rule; its main() is never called)
from src.l1_canonical.adapter import CanonicalDataset  # noqa: E402

X = L.X
OUT = L.OUT
PDIR = L.PDIR
log = L.log
T0 = time.time()


def main(ds):
    pre = json.load(open(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json"), encoding="utf-8"))
    for mod, pn in pre["code"]["new_modules"].items():
        if mod in ("_l1c_br2_lib.py", "_l1c_br2_assign.py"):
            assert L.sha_file(os.path.join(HERE, mod)) == pn["sha256"], "%s changed since the pre-registration" % mod
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert L.sha_file(os.path.join(X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    knn_p = os.path.join(X.REPO, "data", "final_canonical", ds, "graph", "knn.npz")
    assert L.sha_file(knn_p) == pre["knn_read_only"][ds]["sha256"], "knn changed"

    Gr = V.Graph(ds)
    assert Gr.keys_sha == pre["graphs_frozen"][ds]["struct_keys_sha256"] and Gr.C == pre["constants"]["C"]
    k, C, N = L.k_of(Gr), int(Gr.C), int(Gr.N)
    out_p = L.r2_path(ds, Gr)
    assert not os.path.exists(out_p), "write-once: %s exists" % out_p

    # ---- owners (the PHG partition of the primal STRUCT graph, non-empty repaired, validity PASS)
    tag = L.owners_tag(Gr)
    hg_p = os.path.join(PDIR, "%s__%s.npz" % (ds, tag))
    hg_meta = json.load(open(hg_p[:-4] + ".json", encoding="utf-8"))
    assert L.sha_file(hg_p) == hg_meta["file_sha256"] and hg_meta["k"] == k and hg_meta["struct_keys_sha256"] == Gr.keys_sha
    own_p = os.path.join(PDIR, "%s__%s__PHG_con.npy" % (ds, tag))
    run = json.load(open(os.path.join(PDIR, "%s__%s__PHG_con.RUN.json" % (ds, tag)), encoding="utf-8"))
    assert run["output"]["sha256"] == L.sha_file(own_p) and run["hypergraph"]["sha256"] == hg_meta["file_sha256"] and run["hypergraph"]["k"] == k
    val = (run.get("repaired") or run["raw"])["validity"]
    assert val["gate"] == "PASS", val
    node_of_obj = np.load(hg_p)["node_of_obj"].astype(np.int64)
    owners = np.load(own_p).astype(np.int64)
    assert len(owners) == len(node_of_obj) == int((Gr.deg >= 1).sum()) and owners.min() >= 0 and owners.max() < k
    home0 = np.full(N, -1, np.int64)
    home0[node_of_obj] = owners
    deg0 = Gr.deg == 0
    assert (home0[~deg0] >= 0).all() and (home0[deg0] == -1).all()
    sizes0 = np.bincount(home0[~deg0], minlength=k).astype(np.int64)
    assert int((sizes0 > 0).sum()) == k, "an empty owner block survived the non-empty repair"
    bound = int(np.ceil(1.03 * len(node_of_obj) / float(k)))
    assert sizes0.max() <= bound
    hub = Gr.hub
    km1 = (run.get("repaired") or run["raw"])["km1"]
    own_rec = {"tag": tag, "k": k, "objects": int(len(node_of_obj)), "nets_struct_edges": int(Gr.E), "cut_edges_km1": int(km1), "cut_fraction": round(km1 / float(Gr.E), 4),
               "contained_fraction_owners_only": round(1.0 - km1 / float(Gr.E), 4), "validity": val, "repair": run["repair"],
               "block_size_owners": {"mean": round(float(sizes0.mean()), 2), "min": int(sizes0.min()), "max": int(sizes0.max()), "bound_ceil_1.03_N_over_k": bound},
               "phg_run": {kk: run["run"][kk] for kk in ("name", "np", "rc", "wall_seconds_outer") if kk in run["run"]},
               "peak_rss_mb_per_rank": [round(x / 1024.0) for x in run["run"]["memory"]["peak_rss_kb_per_rank_time_v"]] if "memory" in run["run"] else None}
    log("%s owners: k %d, %d objects, cut %d / %d STRUCT edges (%.4f), owner blocks mean %.1f max %d (bound %d)" % (
        ds, k, len(node_of_obj), km1, Gr.E, own_rec["cut_fraction"], sizes0.mean(), sizes0.max(), bound))

    # ---- the universal degree-0 rule (section 20, unchanged; once)
    cd = CanonicalDataset(ds)
    src, dst, w, ent = cd.family("knn")
    home_d0, anchored, fallback, sizes1, d0 = T.place_degree0(Gr, home0, sizes0, k, (src, dst, w))
    home = np.where(deg0, home_d0, home0).astype(np.int64)
    assert (home >= 0).all() and np.array_equal(np.bincount(home, minlength=k), sizes1)
    log("degree-0: %s" % json.dumps(d0))

    # ---- alternates (one static greedy pass; live sizes; cap C)
    alt, sizes2, alt_st, gain_max, home_cnt = L.alternates(Gr, home, C, k)
    assert (alt[deg0] == -1).all() and ((alt < 0) | (alt != home)).all() and sizes2.max() <= C
    mem_all, Y, blocks_csc, sizes = L.membership(home, alt, N, k)
    lam = np.diff(mem_all[0])
    assert lam.max() <= L.R_MAX and lam.min() == 1 and np.array_equal(sizes, sizes2) and int(sizes.sum()) == int(lam.sum())
    ptr, flat = mem_all
    assert all(home[v] in set(flat[ptr[v]:ptr[v + 1]].tolist()) for v in range(0, N, 97))
    RF = float(lam.sum()) / N
    assert RF <= L.R_MAX
    log("alternates: %s" % json.dumps(alt_st))

    # ---- containment + structure (R2 and the owners-only R = 1 control; the section-18 wedge / path3 / ball metrics on metaqa only)
    con = L.containment(Gr, home, alt, Y)
    none = np.full(N, -1, np.int64)
    mem_own, Y_own, blocks_own, sizes_own = L.membership(home, none, N, k)
    con_own = L.containment(Gr, home, none, Y_own)
    full = bool(pre["structure_metrics"]["full_on"].count(ds))
    stride = int(pre["structure_metrics"]["path3_stride"])
    sm = V.structure_metrics(Gr, Y, stride, "BR2_k%d" % k, wedges=full, path3=full, ball=full)
    sm_own = V.structure_metrics(Gr, Y_own, stride, "OWNERS_ONLY_k%d" % k, wedges=full, path3=full, ball=full)
    log("containment R2: %s" % json.dumps(con))
    log("containment owners-only: %s" % json.dumps(con_own))

    rep = {"R": L.R_MAX, "k": k, "C": C, "N": N, "RF": round(RF, 4), "memberships": int(lam.sum()), "replication_beyond_one": int((lam - 1).sum()),
           "lambda": {"max": int(lam.max()), "share_lambda_2_all_nodes": round(float((lam == 2).mean()), 4), "share_lambda_2_struct_nodes": round(float((lam[~deg0] == 2).mean()), 4),
                      "share_lambda_2_nonhub_struct": round(float((lam[(~hub) & (~deg0)] == 2).mean()), 4), "share_lambda_2_hubs": round(float((lam[hub] == 2).mean()), 4) if hub.any() else None,
                      "share_lambda_2_nonhub_deg_ge2": round(float((lam[(~hub) & (Gr.deg >= 2)] == 2).mean()), 4)},
           "block_size": {"mean": round(float(sizes.mean()), 2), "min": int(sizes.min()), "p50": float(np.percentile(sizes, 50)), "p90": float(np.percentile(sizes, 90)),
                          "max": int(sizes.max()), "at_C": int((sizes == C).sum()), "gt_C": int((sizes > C).sum()), "owners_only_mean": round(float(sizes_own.mean()), 2),
                          "owners_only_max": int(sizes_own.max()), "after_degree_0_mean": round(float(sizes1.mean()), 2), "after_degree_0_max": int(sizes1.max())},
           "hubs": int(hub.sum()), "hub_share_of_struct_nodes": round(float(hub.sum() / max(1, (~deg0).sum())), 4), "degree_0": int(deg0.sum()),
           "nodes_with_a_foreign_neighbour": int((gain_max >= 1).sum()), "home_neighbour_count": {"mean_struct_nodes": round(float(home_cnt[~deg0].mean()), 2)},
           "contained_edges_R2_over_owners_only": round(con["contained"] / float(max(1, con_own["contained"])), 3),
           "containment_1hop_R2": con["containment_1hop"], "containment_1hop_owners_only": con_own["containment_1hop"]}
    log("representation: %s" % json.dumps(rep))

    np.savez_compressed(out_p, home=home, alt=alt, k=np.array([k]), N=np.array([N]), C=np.array([C]), R=np.array([L.R_MAX]))
    rec = {"RECORD": "BOUNDED_VERTEXCUT_R2_ASSIGNMENT", "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "preregistration": L.pin(os.path.join(OUT, "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json")), "code": L.pin(os.path.abspath(__file__)), "lib": L.pin(os.path.join(HERE, "_l1c_br2_lib.py")),
           "graph": {"N": N, "struct_edges": int(Gr.E), "k_f": int(Gr.k_f), "C": C, "hubs": int(hub.sum()), "degree_0": int(deg0.sum()), "struct_keys_sha256": Gr.keys_sha},
           "owner_hypergraph": L.pin(hg_p), "owner_partition": L.pin(own_p), "owner_run_record": L.pin(os.path.join(PDIR, "%s__%s__PHG_con.RUN.json" % (ds, tag))), "knn": L.pin(knn_p),
           "owners": own_rec, "degree_0_placement": d0, "alternates": alt_st, "representation": rep,
           "containment": {"R2": con, "owners_only_R1": con_own}, "structure": {"R2": sm, "owners_only_R1": sm_own, "full_metrics": full, "path3_stride": stride},
           "output": {**L.pin(out_p), "arrays": "home (N), alt (N; -1 = none), k, N, C, R"}, "seconds": round(time.time() - T0, 1)}
    L.S.wj(out_p[:-4] + ".json", rec)
    log("%s R2: k %d RF %.4f |B| mean %.1f max %d (at C %d) lambda-2 share %.3f (struct %.3f) 1-hop containment %.4f (owners-only %.4f; %.2fx)%s (%.0fs) -> %s" % (
        ds, k, RF, sizes.mean(), sizes.max(), rep["block_size"]["at_C"], rep["lambda"]["share_lambda_2_all_nodes"], rep["lambda"]["share_lambda_2_struct_nodes"],
        con["containment_1hop"], con_own["containment_1hop"], rep["contained_edges_R2_over_owners_only"],
        (" wedge nh %.4f all %.4f ball2 %.4f" % (sm["wedge_2hop"]["containment_nonhub_middle"], sm["wedge_2hop"]["containment_all"], sm["ball2_section2"]["intact_fraction"])) if full else "",
        time.time() - T0, os.path.relpath(out_p, X.REPO)))
    return rec


if __name__ == "__main__":
    main(sys.argv[1])
