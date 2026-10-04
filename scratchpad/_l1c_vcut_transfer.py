"""STRUCT_VERTEXCUT_V1_TRANSFER (eleventh ruling, 2026-09-15; PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json written first): the
section-19 vertex-cut BASE replay carried unchanged to squad and musique with the ONE universal degree-0 rule of the ruling, gated
against each dataset's canonical hard BASE on DEV_A.  Post-hoc structural-mechanism transfer: DEV_A only, no DEV_B, no TEST, no
confirmation claim, nothing promoted.

Everything frozen is imported unchanged: the section-19 functions (_l1c_vcut_replay: memberships, vote_table, home_table, fused_rank,
evidence, served, unique_exposure, matched_prefixes, hop_masks, compare, rates, stat), the vertex-cut library, the served Data loader
and the frozen numerics.  The only new code is the degree-0 placement (place_degree0) and the per-dataset gate labels of the
pre-registration.  The substrate files are read, never written; nothing under data/ is touched.

usage: python -u scratchpad/_l1c_vcut_transfer.py <squad|musique>   -> results/L1_COVPART/vcut_transfer_A_<ds>.json (+ .log via the shell)
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _l1s_core as S  # noqa: E402
import _l1g_core as G  # noqa: E402
import _l1c_vcut_lib as V  # noqa: E402
import _l1c_vcut_replay as R  # noqa: E402  (section-19 functions; its main() is never called)
from src.l1_canonical.adapter import CanonicalDataset  # noqa: E402

X = S.X
K0, K_LOCK, P_MAIN = S.K0, S.K_LOCK, S.P_MAIN
OUT = V.OUT
PDIR = V.PDIR
log = S.log
sha_file, pin = R.sha_file, R.pin
ALPHA = 0.01
GATE = {"squad": ("HARD_MTK_BASE", "squad", "squad_phg"), "musique": ("HARD_PHG_BASE", "musique", None)}   # (gate label, gate cache, contrast cache)
T0 = time.time()


# ------------------------------------------------------------------------------------------------ the universal degree-0 rule
def place_degree0(Gr, home0, sizes0, k, knn):
    """pre-registered rule: a degree-0 node gets exactly one home = H(nearest anchored KNN neighbour) (max float32 weight, ties -> smallest
    node id; anchored = d_STRUCT > 0; no chaining); otherwise, in ascending node id, the currently smallest block (live sizes, ties ->
    smallest block id).  Returns home_d0 (-1 on STRUCT nodes), anchored mask, fallback mask, sizes after placement, descriptive counts."""
    t = time.time()
    N = Gr.N
    deg0 = Gr.deg == 0
    src, dst, w = (np.asarray(knn[0], np.int64), np.asarray(knn[1], np.int64), np.asarray(knn[2], np.float32))
    assert (src != dst).all() and src.min() >= 0 and dst.min() >= 0 and max(src.max(), dst.max()) < N
    m1 = deg0[src] & ~deg0[dst]
    m2 = deg0[dst] & ~deg0[src]
    v = np.concatenate([src[m1], dst[m2]])
    u = np.concatenate([dst[m1], src[m2]])
    ww = np.concatenate([w[m1], w[m2]])
    o = np.lexsort((u, -ww.astype(np.float64), v))          # by v, then weight descending, then neighbour id ascending
    v, u, ww = v[o], u[o], ww[o]
    first = np.ones(len(v), bool)
    first[1:] = v[1:] != v[:-1]
    anchor = np.full(N, -1, np.int64)
    anchor_w = np.full(N, np.nan, np.float32)
    anchor[v[first]] = u[first]
    anchor_w[v[first]] = ww[first]
    anchored = deg0 & (anchor >= 0)
    assert (home0[anchor[anchored]] >= 0).all()
    home_d0 = np.full(N, -1, np.int64)
    home_d0[anchored] = home0[anchor[anchored]]
    sizes = np.asarray(sizes0, np.int64).copy()
    np.add.at(sizes, home_d0[anchored], 1)
    fb = np.where(deg0 & ~anchored)[0]                       # ascending canonical node id
    for x in fb:
        p = int(np.argmin(sizes))                            # first minimum = smallest block id
        home_d0[x] = p
        sizes[p] += 1
    fallback = deg0 & ~anchored
    assert (home_d0[deg0] >= 0).all() and (home_d0[~deg0] == -1).all()
    # descriptive: fallback nodes that a chaining reading would have anchored (a KNN neighbour that is a degree-0 node placed in pass 1)
    m3 = deg0[src] & deg0[dst]
    a, b = src[m3], dst[m3]
    chain = np.zeros(N, bool)
    chain[a[anchored[b]]] = True
    chain[b[anchored[a]]] = True
    n_chain = int((chain & fallback).sum())
    # ties at the anchored maximum (equal float32 weight, broken by node id)
    tie_cnt = 0
    if len(v):
        grp_first = np.where(first)[0]
        grp_end = np.append(grp_first[1:], len(v))
        for s0, e0 in zip(grp_first, grp_end):
            if e0 - s0 > 1 and ww[s0 + 1] == ww[s0]:
                tie_cnt += 1
    deg0_with_knn = np.zeros(N, bool)
    deg0_with_knn[src[deg0[src]]] = True
    deg0_with_knn[dst[deg0[dst]]] = True
    info = {"degree_0": int(deg0.sum()), "anchored": int(anchored.sum()), "fallback": int(fallback.sum()), "degree_0_without_any_knn_row": int((deg0 & ~deg0_with_knn).sum()),
            "fallback_with_a_placed_degree_0_neighbour (chaining reading would anchor)": n_chain, "anchored_ties_broken_by_node_id": tie_cnt,
            "anchor_weight": {"mean": round(float(np.nanmean(anchor_w[anchored])), 4), "min": round(float(np.nanmin(anchor_w[anchored])), 4)} if anchored.any() else None,
            "knn_rows": int(len(src)), "seconds": round(time.time() - t, 1)}
    return home_d0, anchored, fallback, sizes, info


def extend_membership(mem_all0, home0, home_d0, deg0, N, k):
    ptr0, flat0 = mem_all0
    lists = [flat0[ptr0[x]:ptr0[x + 1]] if not deg0[x] else np.array([home_d0[x]], np.int64) for x in range(N)]
    mem_ext = R.table_from_lists(N, lists)
    home_ext = np.where(deg0, home_d0, home0).astype(np.int64)
    ptr, flat = mem_ext
    Y = sp.csr_matrix((np.ones(len(flat), np.int8), flat, ptr), shape=(N, k))
    Yc = Y.tocsc()
    blocks_csc = (np.asarray(Yc.indptr, np.int64), np.asarray(Yc.indices, np.int64))
    sizes = np.diff(blocks_csc[0]).astype(np.int64)
    return mem_ext, home_ext, blocks_csc, sizes


def label(cmp):
    if cmp["lost"] > cmp["gained"] and cmp["p"] < ALPHA:
        return "LOSS"
    if cmp["gained"] > cmp["lost"] and cmp["p"] < ALPHA:
        return "GAIN"
    return "NEUTRAL"


# ------------------------------------------------------------------------------------------------ main
def main(ds):
    gate_label, gate_cache, contrast_cache = GATE[ds]
    pre = json.load(open(os.path.join(OUT, "PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json"), encoding="utf-8"))
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert sha_file(os.path.join(X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod
    for nm in [gate_cache] + ([contrast_cache] if contrast_cache else []):
        assert sha_file(X.CACHES[nm]) == pre["caches_read_only"][nm]["sha256"], "cache %s changed" % nm
    knn_p = os.path.join(X.REPO, "data", "final_canonical", ds, "graph", "knn.npz")
    assert sha_file(knn_p) == pre["knn_read_only"][ds]["sha256"], "knn changed"
    ks = json.load(open(os.path.join(OUT, "vcut_kstar_%s.json" % ds), encoding="utf-8"))
    k = int(ks["k_star"])
    cap_p = os.path.join(PDIR, "%s__%s_k%d__PHG_con__CAP100.npy" % (ds, V.TAG, k))
    cap_rec = json.load(open(cap_p[:-4] + ".json", encoding="utf-8"))
    assert sha_file(cap_p) == cap_rec["output_assignment"]["sha256"] and cap_rec["k_star"] == k and cap_rec["C"] == 100

    D = G.Data(gate_cache, dense_fp32=False)
    D2 = G.Data(contrast_cache, dense_fp32=False) if contrast_cache else None
    if D2 is not None:
        assert (D.rows == D2.rows).all() and D.C.qids == D2.C.qids, "the two served caches must hold the same rows"
    mA = D.C.split == "A"
    masks = R.hop_masks(D, mA)
    log("%s: gate %s (%s) contrast %s | nq %d DEV_A %d hops %s" % (ds, gate_label, gate_cache, contrast_cache, D.nq, int(mA.sum()), {kk: int(vv.sum()) for kk, vv in masks.items()}))

    Gr = V.Graph(ds)
    assert Gr.N == D.N and Gr.keys_sha == cap_rec["graph"]["struct_keys_sha256"] and Gr.C == 100
    z = np.load(cap_p).astype(np.int64)
    assert len(z) == Gr.E and z.min() >= 0 and z.max() < k
    t = time.time()
    mem_all0, lam0, home0, ties, sizes0, blocks_csc0 = R.memberships(Gr, z, k)
    deg0 = Gr.deg == 0
    assert (home0[~deg0] >= 0).all() and (home0[deg0] == -1).all() and int(lam0[deg0].sum()) == 0 and int((sizes0 > 0).sum()) == k
    ptr0, flat0 = mem_all0
    assert all(home0[x] in set(flat0[ptr0[x]:ptr0[x + 1]].tolist()) for x in np.where(~deg0)[0][::97])

    # ---- the universal degree-0 rule (once; no re-repair afterwards)
    cd = CanonicalDataset(ds)
    src, dst, w, ent = cd.family("knn")
    home_d0, anchored, fallback, sizes_after, d0 = place_degree0(Gr, home0, sizes0, k, (src, dst, w))
    mem_all, home, blocks_csc, sizes = extend_membership(mem_all0, home0, home_d0, deg0, Gr.N, k)
    assert np.array_equal(sizes, sizes_after) and int(sizes.sum()) == int(lam0.sum()) + int(deg0.sum())
    lam = np.diff(mem_all[0])
    assert (lam[deg0] == 1).all() and np.array_equal(lam[~deg0], lam0[~deg0])
    hub = Gr.hub
    assert not hub[deg0].any()
    mem_vc = R.vote_table(mem_all, home, hub)
    mem_home = R.home_table(home)
    ptr, flat = mem_all
    rep = {"k": k, "C": int(Gr.C), "RF_struct_only": round(float(lam0.sum()) / Gr.N, 4), "RF_after_placement": round(float(lam.sum()) / Gr.N, 4),
           "block_size_struct_only": {"mean": round(float(sizes0.mean()), 2), "max": int(sizes0.max()), "min": int(sizes0.min()), "blocks_gt_C": int((sizes0 > Gr.C).sum())},
           "block_size_after_placement": {"mean": round(float(sizes.mean()), 2), "max": int(sizes.max()), "min": int(sizes.min()), "p50": float(np.percentile(sizes, 50)),
                                          "p90": float(np.percentile(sizes, 90)), "blocks_gt_C": int((sizes > Gr.C).sum()), "blocks_gt_1.5C": int((sizes > 1.5 * Gr.C).sum())},
           "lambda": {"hub_mean": round(float(lam[hub].mean()), 2), "hub_max": int(lam[hub].max()), "nonhub_struct_mean": round(float(lam[(~hub) & (~deg0)].mean()), 4),
                      "nonhub_struct_deg_ge2_mean": round(float(lam[(~hub) & (Gr.deg >= 2)].mean()), 4), "share_lambda_1_all_nodes": round(float((lam == 1).mean()), 4),
                      "share_lambda_1_struct_nodes": round(float((lam[~deg0] == 1).mean()), 4)},
           "home_ties_broken_by_id_struct_nodes": int(ties), "home_ties_fraction_struct_nodes": round(ties / float((~deg0).sum()), 4),
           "halo_mean_size_struct_nodes": round(float((lam[~deg0] - 1).mean()), 4), "hubs": int(hub.sum()), "hub_share_of_struct_nodes": round(float(hub.sum() / (~deg0).sum()), 4),
           "vote_table_entries": {"mem_vc": int(len(mem_vc[1])), "mem_all": int(len(flat)), "mem_home": int(len(mem_home[1]))},
           "degree_0_placement": d0, "seconds": round(time.time() - t, 1)}
    log("representation: %s" % json.dumps(rep))

    # ---- hub / degree-0 share of the served hits and of the gold (descriptive)
    di, si = D.d_ids[:, :K_LOCK], D.s_ids[:, :K_LOCK]
    assert di.min() >= 0 and si.min() >= 0
    hub_gold = np.array([float(hub[g].mean()) if len(g) else 0.0 for g in D.C.gold_nodes])
    q_with_hub_gold = np.array([bool(hub[g].any()) for g in D.C.gold_nodes])
    d0_gold = np.array([float(deg0[g].mean()) if len(g) else 0.0 for g in D.C.gold_nodes])
    q_with_d0_gold = np.array([bool(deg0[g].any()) for g in D.C.gold_nodes])
    shares = {"served_hits_hub_share": {"dense_top%d" % K_LOCK: round(float(hub[di].mean()), 4), "splade_top%d" % K_LOCK: round(float(hub[si].mean()), 4)},
              "served_hits_degree_0_share": {"dense_top%d" % K_LOCK: round(float(deg0[di].mean()), 4), "splade_top%d" % K_LOCK: round(float(deg0[si].mean()), 4)},
              "gold_hub_share": {"gold_nodes_hub_fraction_A_mean": round(float(hub_gold[mA].mean()), 4), "queries_A_with_hub_gold": int((q_with_hub_gold & mA).sum())},
              "gold_degree_0_share": {"gold_nodes_degree_0_fraction_A_mean": round(float(d0_gold[mA].mean()), 4), "queries_A_with_degree_0_gold": int((q_with_d0_gold & mA).sum())}}

    # ---- baselines (served) + the node-level check
    base = D.base_all.astype(np.int8)
    a_chk, _ = R.served(D, D.base_sel, D.mem_hard)
    assert (a_chk == base).all(), "node-level ALL != served block-level ALL (gate)"
    exp = np.array([int(D.sizes[np.asarray(s, np.int64)].sum()) for s in D.base_sel])
    if D2 is not None:
        base2 = D2.base_all.astype(np.int8)
        a_chk2, _ = R.served(D2, D2.base_sel, D2.mem_hard)
        assert (a_chk2 == base2).all(), "node-level ALL != served block-level ALL (contrast)"
        exp2 = np.array([int(D2.sizes[np.asarray(s, np.int64)].sum()) for s in D2.base_sel])
        log("baselines: %s ALL(A) %.4f exposure %.1f | contrast %s ALL(A) %.4f exposure %.1f" % (gate_label, base[mA].mean(), exp[mA].mean(), contrast_cache, base2[mA].mean(), exp2[mA].mean()))
    else:
        log("baselines: %s ALL(A) %.4f exposure %.1f" % (gate_label, base[mA].mean(), exp[mA].mean()))

    # ---- vertex-cut routing (frozen numerics)
    t = time.time()
    R_vc = R.fused_rank(D, mem_vc, k)
    R_home = R.fused_rank(D, mem_home, k)
    R_own = R.fused_rank(D, D.mem_hard, D.npart)
    log("rankings computed (%.0fs)" % (time.time() - t))
    sel50 = [R_vc[i, :P_MAIN] for i in range(D.nq)]
    all50, any50 = R.served(D, sel50, mem_all)
    u50, nom50 = R.unique_exposure(sel50, blocks_csc, Gr.N)
    L_le, U_le, L_ge, U_ge = R.matched_prefixes(R_vc, blocks_csc, Gr.N, exp)
    sel_le = [R_vc[i, :L_le[i]] for i in range(D.nq)]
    sel_ge = [R_vc[i, :L_ge[i]] for i in range(D.nq)]
    all_le, any_le = R.served(D, sel_le, mem_all)
    all_ge, any_ge = R.served(D, sel_ge, mem_all)
    assert (U_le <= exp).all() and (L_ge >= L_le).all()
    sel_home = [R_home[i, :P_MAIN] for i in range(D.nq)]
    all_home, any_home = R.served(D, sel_home, mem_all)
    u_home, _ = R.unique_exposure(sel_home, blocks_csc, Gr.N)
    sel_own = [R_own[i, :P_MAIN] for i in range(D.nq)]
    all_own, any_own = R.served(D, sel_own, D.mem_hard)
    exp_own = np.array([int(D.sizes[np.asarray(s, np.int64)].sum()) for s in sel_own])

    # ---- reach decomposition for the primary cell
    ev = R.evidence(D, mem_vc, k)
    pos = G.positions(R_vc)
    reached = np.zeros(D.nq, bool)
    worst = np.full(D.nq, -1, np.int64)
    n_gold = np.zeros(D.nq, np.int64)
    for i, g in enumerate(D.C.gold_nodes):
        if len(g) == 0:
            continue
        n_gold[i] = len(g)
        okv, best = [], []
        for x in g:
            mb = flat[ptr[x]:ptr[x + 1]]
            okv.append(bool(ev[i, mb].any()))
            best.append(int(pos[i, mb].min()))
        reached[i] = all(okv)
        worst[i] = max(best)
    fail = (all50 == 0) & mA
    dec = {"failures_A": int(fail.sum()), "UNREACHED": int((fail & ~reached).sum()), "REACHED_WEAK": int((fail & reached).sum()),
           "reached_rate_A": round(float(reached[mA].mean()), 4),
           "worst_gold_position_among_failures": R.stat(worst[fail]) if fail.any() else None,
           "failures_with_worst_le_100": int((fail & (worst < 100)).sum()), "failures_with_worst_le_200": int((fail & (worst < 200)).sum()),
           "failures_with_hub_gold": int((fail & q_with_hub_gold).sum()), "failures_with_degree_0_gold": int((fail & q_with_d0_gold).sum()),
           "gold_nodes_per_query_A": R.stat(n_gold[mA])}
    hard_fail = (base == 0) & mA
    both = {"fail_both": int((fail & hard_fail).sum()), "fail_vcut_only": int((fail & ~hard_fail).sum()), "fail_hard_only": int((~fail & hard_fail).sum())}

    # ---- statistics (DEV_A only)
    vs = "vs_" + gate_label
    cells = {
        "VCUT_P50": {"ALL": R.rates(all50, masks), "ANY": R.rates(any50, masks), vs: R.compare(base, all50, masks),
                     "exposure_unique_A": R.stat(u50, mA), "exposure_nominal_A": R.stat(nom50, mA), "blocks": P_MAIN,
                     "queries_A_with_unique_le_hard": int(((u50 <= exp) & mA).sum()), "unique_over_hard_ratio_A": round(float(u50[mA].mean() / exp[mA].mean()), 4)},
        "VCUT_MATCHED_LE": {"ALL": R.rates(all_le, masks), "ANY": R.rates(any_le, masks), vs: R.compare(base, all_le, masks),
                            "blocks_used_A": R.stat(L_le, mA), "exposure_unique_A": R.stat(U_le, mA), "rule": "largest fused prefix with unique exposure <= the %s P50 exposure of the same query" % gate_label},
        "VCUT_MATCHED_GE (informational)": {"ALL": R.rates(all_ge, masks), "ANY": R.rates(any_ge, masks), vs: R.compare(base, all_ge, masks),
                                            "blocks_used_A": R.stat(L_ge, mA), "exposure_unique_A": R.stat(U_ge, mA), "overshoot_nodes_A": R.stat(U_ge - exp, mA)},
        "VCUT_HOME_ONLY_VOTE (diagnostic)": {"ALL": R.rates(all_home, masks), "ANY": R.rates(any_home, masks), vs: R.compare(base, all_home, masks),
                                             "vs_VCUT_P50": R.compare(all50, all_home, masks), "exposure_unique_A": R.stat(u_home, mA)},
        "HARD_OWN_BLOCK_VOTE (diagnostic)": {"ALL": R.rates(all_own, masks), "ANY": R.rates(any_own, masks), vs: R.compare(base, all_own, masks),
                                             "exposure_A": R.stat(exp_own, mA), "partition": gate_cache},
        gate_label: {"ALL": R.rates(base, masks), "exposure_A": R.stat(exp, mA), "npart": int(D.npart), "cache": gate_cache, "role": "GATE"},
    }
    if D2 is not None:
        cells["VCUT_P50"]["vs_HARD_PHG_BASE (contrast)"] = R.compare(base2, all50, masks)
        cells["HARD_PHG_BASE (contrast)"] = {"ALL": R.rates(base2, masks), "exposure_A": R.stat(exp2, mA), "npart": int(D2.npart), "cache": contrast_cache, "role": "contrast only"}
    p50 = cells["VCUT_P50"][vs]["all"]
    le = cells["VCUT_MATCHED_LE"][vs]["all"]
    mean_u = cells["VCUT_P50"]["exposure_unique_A"]["mean"]
    mean_h = cells[gate_label]["exposure_A"]["mean"]
    primary = label(p50)
    flag = "EXPOSURE_LE_HARD" if mean_u <= mean_h else "EXPOSURE_ABOVE_HARD"
    matched = label(le)
    ver = {"VERDICT": "%s_%s" % (primary, flag), "primary_label": primary, "exposure_flag": flag, "matched_label": matched, "passes_ruling_no_significant_loss": primary != "LOSS",
           "alpha": ALPHA, "gate": gate_label, "primary_VCUT_P50_vs_gate": p50, "primary_mean_unique_exposure": mean_u, "hard_mean_exposure": mean_h,
           "secondary_VCUT_MATCHED_LE_vs_gate": le,
           "status": "post-hoc structural-mechanism transfer result on DEV_A; not a confirmation; not promotable; served L1 unchanged"}
    res = {"RECORD": "STRUCT_VERTEXCUT_V1_TRANSFER", "dataset": ds, "cache": gate_cache, "contrast_cache": contrast_cache, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "DEV": "A",
           "preregistration": pin(os.path.join(OUT, "PREREGISTRATION_STRUCT_VERTEXCUT_V1_TRANSFER.json")), "substrate": pin(cap_p), "cap_record": pin(cap_p[:-4] + ".json"),
           "kstar_record": pin(os.path.join(OUT, "vcut_kstar_%s.json" % ds)), "knn": pin(knn_p), "code": pin(os.path.abspath(__file__)),
           "n_DEV_A": int(mA.sum()), "hop_counts_A": {kk: int(vv.sum()) for kk, vv in masks.items()},
           "representation": rep, "shares": shares, "cells": cells, "reach_decomposition_VCUT_P50": dec, "failure_overlap_with_gate_A": both, "verdict": ver,
           "constants": {"K0": K0, "K_LOCK": K_LOCK, "P_MAIN": P_MAIN}, "seconds": round(time.time() - T0, 1)}
    fp = os.path.join(OUT, "vcut_transfer_A_%s.json" % ds)
    S.wj(fp, res)
    log("%s VCUT_P50 ALL(A) %.4f (gate %.4f; +%d/-%d p=%.3g) unique exposure %.1f vs hard %.1f | MATCHED_LE ALL %.4f (+%d/-%d p=%.3g; %.1f blocks, %.1f nodes) | "
        "MATCHED_GE ALL %.4f | HOME_ONLY %.4f | OWN_BLOCK %.4f -> %s (matched %s) (%.0fs) -> %s" % (
            ds, cells["VCUT_P50"]["ALL"]["all"], cells[gate_label]["ALL"]["all"], p50["gained"], p50["lost"], p50["p"], mean_u, mean_h,
            cells["VCUT_MATCHED_LE"]["ALL"]["all"], le["gained"], le["lost"], le["p"], cells["VCUT_MATCHED_LE"]["blocks_used_A"]["mean"],
            cells["VCUT_MATCHED_LE"]["exposure_unique_A"]["mean"], cells["VCUT_MATCHED_GE (informational)"]["ALL"]["all"], cells["VCUT_HOME_ONLY_VOTE (diagnostic)"]["ALL"]["all"],
            cells["HARD_OWN_BLOCK_VOTE (diagnostic)"]["ALL"]["all"], ver["VERDICT"], matched, time.time() - T0, os.path.relpath(fp, X.REPO)))
    for nm in masks:
        if nm == "all":
            continue
        c = cells["VCUT_P50"][vs][nm]
        d = cells["VCUT_MATCHED_LE"][vs][nm]
        log("  %s n %d: P50 %.4f vs hard %.4f (+%d/-%d p=%.3g) | MATCHED_LE %.4f (+%d/-%d p=%.3g)" % (nm, c["n"], c["cell_ALL"], c["base_ALL"], c["gained"], c["lost"], c["p"],
                                                                                                      d["cell_ALL"], d["gained"], d["lost"], d["p"]))
    log("  reach: %s" % json.dumps(dec))
    log("  degree-0: %s" % json.dumps(d0))
    return res


if __name__ == "__main__":
    main(sys.argv[1])
