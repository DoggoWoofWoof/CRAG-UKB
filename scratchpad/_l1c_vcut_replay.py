"""STRUCT_VERTEXCUT_V1_REPLAY (tenth ruling, 2026-09-15; PREREGISTRATION_STRUCT_VERTEXCUT_V1_REPLAY.json written first): the frozen
dense + SPLADE partition router re-run over the capacity-repaired k* = 1004 vertex-cut blocks of the metaqa STRUCT graph, with the
home + overlap representation (M(v), H(v)) and hub-limited voting, evaluated node-level on DEV_A against the served hard BASE at the
actual unique-node exposure.  Post-hoc structural-mechanism experiment: DEV_A only, no DEV_B, no TEST, no confirmation claim.

Everything frozen is imported unchanged (_l1s_core / _l1x90_core / _ta_prepartition / _l1g_core / _l1c_vcut_lib); the substrate file
is read, never written; nothing under data/ is touched.  Output: results/L1_COVPART/vcut_replay_A_metaqa.json (+ .log via the shell).

usage: python -u scratchpad/_l1c_vcut_replay.py metaqa
"""
import hashlib
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

X = S.X
K0, K_LOCK, P_MAIN = S.K0, S.K_LOCK, S.P_MAIN
OUT = V.OUT
PDIR = V.PDIR
log = S.log
K_VC = 1004
ALPHA = 0.01
T0 = time.time()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pin(p):
    return {"path": os.path.relpath(p, X.REPO).replace("\\", "/"), "sha256": sha_file(p), "bytes": os.path.getsize(p)}


# ------------------------------------------------------------------------------------------------ representation
def memberships(Gr, z, k):
    """M(v) as a (ptr, flat) table (sorted block ids), lambda(v), H(v) (argmax incident-edge count, ties -> smallest id), block sizes |B_p|."""
    Y = V.Y_vcut(Gr, z, k).tocsr()
    ptr, flat = np.asarray(Y.indptr, np.int64), np.asarray(Y.indices, np.int64)
    lam = np.diff(ptr)
    Cnt = sp.coo_matrix((np.ones(2 * Gr.E, np.int64), (np.concatenate([Gr.eu, Gr.ev]), np.concatenate([z, z]))), shape=(Gr.N, k)).tocsr()
    Cnt.sum_duplicates()
    Cnt.sort_indices()
    home = np.full(Gr.N, -1, np.int64)
    tie = 0
    for v in range(Gr.N):
        a, b = Cnt.indptr[v], Cnt.indptr[v + 1]
        if b > a:
            d = Cnt.data[a:b]
            j = int(np.argmax(d))                      # first maximum = smallest block id (indices sorted)
            home[v] = Cnt.indices[a + j]
            tie += int((d == d[j]).sum() > 1)
    sizes = np.asarray(Y.sum(axis=0)).ravel().astype(np.int64)
    Yc = Y.tocsc()
    return (ptr, flat), lam, home, tie, sizes, (np.asarray(Yc.indptr, np.int64), np.asarray(Yc.indices, np.int64))


def table_from_lists(N, lists):
    """(ptr, flat) from per-node arrays of block ids."""
    cnt = np.array([len(x) for x in lists], np.int64)
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(cnt)
    flat = np.concatenate(lists).astype(np.int64) if cnt.sum() else np.zeros(0, np.int64)
    return ptr, flat


def vote_table(mem_all, home, hub):
    ptr, flat = mem_all
    lists = [flat[ptr[v]:ptr[v + 1]] if not hub[v] else np.array([home[v]], np.int64) for v in range(len(home))]
    return table_from_lists(len(home), lists)


def home_table(home):
    return table_from_lists(len(home), [np.array([h], np.int64) for h in home])


# ------------------------------------------------------------------------------------------------ routing (frozen numerics)
def fused_rank(D, mem, npart):
    PR_d = S.canon_channel_rank([D.d_ids[i, :K_LOCK] for i in range(D.nq)], mem, npart)
    PR_s = S.canon_channel_rank([D.s_ids[i, :K_LOCK] for i in range(D.nq)], mem, npart)
    return S.rrf_ranks([PR_d, PR_s])


def evidence(D, mem, npart):
    w = np.ones((D.nq, K_LOCK), np.float64)
    ev = S.hits_to_blocks(D.d_ids[:, :K_LOCK], w, mem, npart, "sum") > 0
    ev |= S.hits_to_blocks(D.s_ids[:, :K_LOCK], w, mem, npart, "sum") > 0
    return ev


# ------------------------------------------------------------------------------------------------ serving / exposure
def served(D, sel_rows, memb):
    """node-level ALL / ANY through a serving membership (ptr, flat): gold g served iff M(g) meets the selection."""
    ptr, flat = memb
    allv = np.zeros(D.nq, np.int8)
    anyv = np.zeros(D.nq, np.int8)
    for i, s in enumerate(sel_rows):
        s = set(int(x) for x in s)
        g = D.C.gold_nodes[i]
        if len(g) == 0:
            continue
        ok = [any(int(b) in s for b in flat[ptr[x]:ptr[x + 1]]) for x in g]
        allv[i] = int(all(ok))
        anyv[i] = int(any(ok))
    return allv, anyv


def block_nodes(blocks_csc, p):
    ptr, flat = blocks_csc
    return flat[ptr[p]:ptr[p + 1]]


def unique_exposure(sel_rows, blocks_csc, N):
    out = np.zeros(len(sel_rows), np.int64)
    nom = np.zeros(len(sel_rows), np.int64)
    for i, s in enumerate(sel_rows):
        parts = [block_nodes(blocks_csc, int(p)) for p in s]
        nom[i] = sum(len(x) for x in parts)
        out[i] = len(np.unique(np.concatenate(parts))) if parts else 0
    return out, nom


def matched_prefixes(rank, blocks_csc, N, target):
    """per query: L_le = largest fused prefix with unique exposure <= target (never exceeds); L_ge = smallest prefix with unique >= target."""
    nq = rank.shape[0]
    L_le = np.zeros(nq, np.int64)
    U_le = np.zeros(nq, np.int64)
    L_ge = np.zeros(nq, np.int64)
    U_ge = np.zeros(nq, np.int64)
    mask = np.zeros(N, bool)
    for i in range(nq):
        mask[:] = False
        u = 0
        j = 0
        tgt = int(target[i])
        while True:
            nodes = block_nodes(blocks_csc, int(rank[i, j]))
            new = int((~mask[nodes]).sum())
            if u + new > tgt:
                L_le[i], U_le[i] = j, u
                mask[nodes] = True
                L_ge[i], U_ge[i] = j + 1, u + new
                break
            mask[nodes] = True
            u += new
            j += 1
            if u == tgt or j >= rank.shape[1]:
                L_le[i], U_le[i] = j, u
                L_ge[i], U_ge[i] = j, u
                break
    return L_le, U_le, L_ge, U_ge


# ------------------------------------------------------------------------------------------------ statistics
def hop_masks(D, mA):
    hops = np.asarray(D.C.hops)
    out = {"all": mA.copy()}
    for h in sorted(set(int(x) for x in hops[hops >= 0])):
        out["hop%d" % h] = mA & (hops == h)
    if "hop2" in out and "hop3" in out:
        out["hop2+3"] = out["hop2"] | out["hop3"]
    return out


def compare(base, cell, masks):
    res = {}
    for nm, m in masks.items():
        g, l, p = X.mcnemar(base[m], cell[m])
        res[nm] = {"n": int(m.sum()), "base_ALL": round(float(base[m].mean()), 4), "cell_ALL": round(float(cell[m].mean()), 4),
                   "delta_pts": round(100 * float(cell[m].mean() - base[m].mean()), 2), "gained": g, "lost": l, "p": p}
    return res


def rates(v, masks):
    return {nm: round(float(v[m].mean()), 4) for nm, m in masks.items()}


def stat(x, m=None):
    x = np.asarray(x)
    if m is not None:
        x = x[m]
    return {"mean": round(float(x.mean()), 2), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)), "min": int(x.min()), "max": int(x.max())}


# ------------------------------------------------------------------------------------------------ main
def main(name):
    ds = X.DS_OF[name]
    assert name == "metaqa", "pre-registered for metaqa only (text corpora need the degree-0 ruling)"
    pre = json.load(open(os.path.join(OUT, "PREREGISTRATION_STRUCT_VERTEXCUT_V1_REPLAY.json"), encoding="utf-8"))
    cap_p = os.path.join(X.REPO, pre["substrate_frozen"]["edge_assignment_file"]["path"])
    assert sha_file(cap_p) == pre["substrate_frozen"]["edge_assignment_file"]["sha256"], "substrate changed"
    for mod, pn in pre["code"]["imports_unchanged"].items():
        assert sha_file(os.path.join(X.REPO, pn["path"])) == pn["sha256"], "%s changed since the pre-registration" % mod

    D = G.Data(name, dense_fp32=False)
    D2 = G.Data(name + "_phg", dense_fp32=False)
    assert (D.rows == D2.rows).all() and D.C.qids == D2.C.qids, "the two served caches must hold the same rows"
    mA = D.C.split == "A"
    masks = hop_masks(D, mA)
    log("%s: nq %d DEV_A %d hops %s" % (name, D.nq, int(mA.sum()), {k: int(v.sum()) for k, v in masks.items()}))

    Gr = V.Graph(ds)
    assert Gr.N == D.N and int((Gr.deg == 0).sum()) == 0, "degree-0 nodes present -> ruling needed"
    assert Gr.keys_sha == pre["substrate_frozen"]["graph"]["struct_keys_sha256"]
    z = np.load(cap_p).astype(np.int64)
    assert len(z) == Gr.E and z.min() >= 0 and z.max() < K_VC
    t = time.time()
    mem_all, lam, home, ties, sizes, blocks_csc = memberships(Gr, z, K_VC)
    assert (home >= 0).all() and int((sizes > 0).sum()) == K_VC
    hub = Gr.hub
    mem_vc = vote_table(mem_all, home, hub)
    mem_home = home_table(home)
    # home is a member; halo = M(v) - home
    ptr, flat = mem_all
    assert all(home[v] in set(flat[ptr[v]:ptr[v + 1]].tolist()) for v in range(0, Gr.N, 97))
    rep = {"k": K_VC, "RF": round(float(lam.sum()) / Gr.N, 4), "block_size": {"mean": round(float(sizes.mean()), 2), "max": int(sizes.max()), "min": int(sizes.min())},
           "lambda": {"hub_mean": round(float(lam[hub].mean()), 2), "hub_max": int(lam[hub].max()), "nonhub_mean": round(float(lam[~hub].mean()), 4),
                      "nonhub_deg_ge2_mean": round(float(lam[(~hub) & (Gr.deg >= 2)].mean()), 4), "share_lambda_1": round(float((lam == 1).mean()), 4)},
           "home_ties_broken_by_id": int(ties), "home_ties_fraction": round(ties / float(Gr.N), 4),
           "halo_mean_size": round(float((lam - 1).mean()), 4), "hubs": int(hub.sum()), "C": int(Gr.C),
           "vote_table_entries": {"mem_vc": int(len(mem_vc[1])), "mem_all": int(len(flat)), "mem_home": int(len(mem_home[1]))},
           "seconds": round(time.time() - t, 1)}
    log("representation: %s" % json.dumps(rep))

    # ---- hub share of the served hits (descriptive)
    di, si = D.d_ids[:, :K_LOCK], D.s_ids[:, :K_LOCK]
    assert di.min() >= 0 and si.min() >= 0
    hd = float(hub[di].mean())
    hs = float(hub[si].mean())
    hub_gold = np.array([float(hub[g].mean()) if len(g) else 0.0 for g in D.C.gold_nodes])
    q_with_hub_gold = np.array([bool(hub[g].any()) for g in D.C.gold_nodes])

    # ---- baselines (served) + assertion that the node-level rule reproduces the served block-level ALL on hard partitions
    base_mtk = D.base_all.astype(np.int8)
    a_chk, _ = served(D, D.base_sel, D.mem_hard)
    assert (a_chk == base_mtk).all(), "node-level ALL != served block-level ALL (MtK)"
    base_phg = D2.base_all.astype(np.int8)
    a_chk2, _ = served(D2, D2.base_sel, D2.mem_hard)
    assert (a_chk2 == base_phg).all(), "node-level ALL != served block-level ALL (PHG)"
    exp_mtk = np.array([int(D.sizes[np.asarray(s, np.int64)].sum()) for s in D.base_sel])
    exp_phg = np.array([int(D2.sizes[np.asarray(s, np.int64)].sum()) for s in D2.base_sel])
    log("baselines: HARD_MTK_BASE ALL(A) %.4f exposure %.1f | HARD_PHG_BASE ALL(A) %.4f exposure %.1f" % (
        base_mtk[mA].mean(), exp_mtk[mA].mean(), base_phg[mA].mean(), exp_phg[mA].mean()))

    # ---- vertex-cut routing
    t = time.time()
    R_vc = fused_rank(D, mem_vc, K_VC)
    R_home = fused_rank(D, mem_home, K_VC)
    R_own = fused_rank(D, D.mem_hard, D.npart)
    log("rankings computed (%.0fs)" % (time.time() - t))
    sel50 = [R_vc[i, :P_MAIN] for i in range(D.nq)]
    all50, any50 = served(D, sel50, mem_all)
    u50, nom50 = unique_exposure(sel50, blocks_csc, Gr.N)
    # matched budgets against the served hard MtK exposure of the same query
    L_le, U_le, L_ge, U_ge = matched_prefixes(R_vc, blocks_csc, Gr.N, exp_mtk)
    sel_le = [R_vc[i, :L_le[i]] for i in range(D.nq)]
    sel_ge = [R_vc[i, :L_ge[i]] for i in range(D.nq)]
    all_le, any_le = served(D, sel_le, mem_all)
    all_ge, any_ge = served(D, sel_ge, mem_all)
    assert (U_le <= exp_mtk).all() and (L_ge >= L_le).all()
    # labelled diagnostics
    sel_home = [R_home[i, :P_MAIN] for i in range(D.nq)]
    all_home, any_home = served(D, sel_home, mem_all)
    u_home, _ = unique_exposure(sel_home, blocks_csc, Gr.N)
    sel_own = [R_own[i, :P_MAIN] for i in range(D.nq)]
    all_own, any_own = served(D, sel_own, D.mem_hard)
    exp_own = np.array([int(D.sizes[np.asarray(s, np.int64)].sum()) for s in sel_own])

    # ---- reach decomposition for the primary cell
    ev = evidence(D, mem_vc, K_VC)
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
           "worst_gold_position_among_failures": stat(worst[fail]) if fail.any() else None,
           "failures_with_worst_le_100": int((fail & (worst < 100)).sum()), "failures_with_worst_le_200": int((fail & (worst < 200)).sum()),
           "failures_with_hub_gold": int((fail & q_with_hub_gold).sum()), "gold_nodes_per_query_A": stat(n_gold[mA])}
    hard_fail = (base_mtk == 0) & mA
    both = {"fail_both": int((fail & hard_fail).sum()), "fail_vcut_only": int((fail & ~hard_fail).sum()), "fail_hard_only": int((~fail & hard_fail).sum())}

    # ---- statistics (DEV_A only)
    cells = {
        "VCUT_P50": {"ALL": rates(all50, masks), "ANY": rates(any50, masks), "vs_HARD_MTK_BASE": compare(base_mtk, all50, masks), "vs_HARD_PHG_BASE": compare(base_phg, all50, masks),
                     "exposure_unique_A": stat(u50, mA), "exposure_nominal_A": stat(nom50, mA), "blocks": P_MAIN,
                     "queries_A_with_unique_le_hard": int(((u50 <= exp_mtk) & mA).sum()), "unique_over_hard_ratio_A": round(float(u50[mA].mean() / exp_mtk[mA].mean()), 4)},
        "VCUT_MATCHED_LE": {"ALL": rates(all_le, masks), "ANY": rates(any_le, masks), "vs_HARD_MTK_BASE": compare(base_mtk, all_le, masks),
                            "blocks_used_A": stat(L_le, mA), "exposure_unique_A": stat(U_le, mA), "rule": "largest fused prefix with unique exposure <= the hard MtK P50 exposure of the same query"},
        "VCUT_MATCHED_GE (informational)": {"ALL": rates(all_ge, masks), "ANY": rates(any_ge, masks), "vs_HARD_MTK_BASE": compare(base_mtk, all_ge, masks),
                                            "blocks_used_A": stat(L_ge, mA), "exposure_unique_A": stat(U_ge, mA), "overshoot_nodes_A": stat(U_ge - exp_mtk, mA)},
        "VCUT_HOME_ONLY_VOTE (diagnostic)": {"ALL": rates(all_home, masks), "ANY": rates(any_home, masks), "vs_HARD_MTK_BASE": compare(base_mtk, all_home, masks),
                                             "vs_VCUT_P50": compare(all50, all_home, masks), "exposure_unique_A": stat(u_home, mA)},
        "HARD_MTK_OWN_BLOCK_VOTE (diagnostic)": {"ALL": rates(all_own, masks), "ANY": rates(any_own, masks), "vs_HARD_MTK_BASE": compare(base_mtk, all_own, masks),
                                                 "exposure_A": stat(exp_own, mA)},
        "HARD_MTK_BASE": {"ALL": rates(base_mtk, masks), "exposure_A": stat(exp_mtk, mA), "npart": int(D.npart)},
        "HARD_PHG_BASE": {"ALL": rates(base_phg, masks), "exposure_A": stat(exp_phg, mA), "npart": int(D2.npart)},
    }
    p50 = cells["VCUT_P50"]["vs_HARD_MTK_BASE"]["all"]
    le = cells["VCUT_MATCHED_LE"]["vs_HARD_MTK_BASE"]["all"]
    sig50 = p50["gained"] > p50["lost"] and p50["p"] < ALPHA
    lower = cells["VCUT_P50"]["exposure_unique_A"]["mean"] <= cells["HARD_MTK_BASE"]["exposure_A"]["mean"]
    sigle = le["gained"] > le["lost"] and le["p"] < ALPHA
    if sig50 and lower:
        verdict = "MECHANISM_POSITIVE_AT_LOWER_EXPOSURE"
    elif sigle:
        verdict = "MECHANISM_POSITIVE_AT_MATCHED_BUDGET"
    elif sig50:
        verdict = "MECHANISM_POSITIVE_ONLY_WITH_MORE_EXPOSURE"
    else:
        verdict = "NEGATIVE"
    ver = {"VERDICT": verdict, "alpha": ALPHA, "primary_VCUT_P50_vs_HARD_MTK_BASE": p50, "primary_exposure_le_hard": bool(lower),
           "primary_mean_unique_exposure": cells["VCUT_P50"]["exposure_unique_A"]["mean"], "hard_mean_exposure": cells["HARD_MTK_BASE"]["exposure_A"]["mean"],
           "secondary_VCUT_MATCHED_LE_vs_HARD_MTK_BASE": le,
           "status": "post-hoc structural-mechanism result on DEV_A; not a confirmation; not promotable; served L1 unchanged"}
    res = {"RECORD": "STRUCT_VERTEXCUT_V1_REPLAY", "cache": name, "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "DEV": "A",
           "preregistration": pin(os.path.join(OUT, "PREREGISTRATION_STRUCT_VERTEXCUT_V1_REPLAY.json")), "substrate": pin(cap_p), "code": pin(os.path.abspath(__file__)),
           "n_DEV_A": int(mA.sum()), "hop_counts_A": {k: int(v.sum()) for k, v in masks.items()},
           "representation": rep, "served_hits_hub_share": {"dense_top%d" % K_LOCK: round(hd, 4), "splade_top%d" % K_LOCK: round(hs, 4)},
           "gold_hub_share": {"gold_nodes_hub_fraction_A_mean": round(float(hub_gold[mA].mean()), 4), "queries_A_with_hub_gold": int((q_with_hub_gold & mA).sum())},
           "cells": cells, "reach_decomposition_VCUT_P50": dec, "failure_overlap_with_HARD_MTK_BASE_A": both, "verdict": ver,
           "constants": {"K0": K0, "K_LOCK": K_LOCK, "P_MAIN": P_MAIN}, "seconds": round(time.time() - T0, 1)}
    fp = os.path.join(OUT, "vcut_replay_A_%s.json" % name)
    S.wj(fp, res)
    log("VCUT_P50 ALL(A) %.4f (hard %.4f; +%d/-%d p=%.3g) unique exposure %.1f vs hard %.1f | MATCHED_LE ALL %.4f (+%d/-%d p=%.3g; %.1f blocks, %.1f nodes) | "
        "MATCHED_GE ALL %.4f | HOME_ONLY %.4f | OWN_BLOCK %.4f | PHG base %.4f -> %s (%.0fs) -> %s" % (
            cells["VCUT_P50"]["ALL"]["all"], cells["HARD_MTK_BASE"]["ALL"]["all"], p50["gained"], p50["lost"], p50["p"], cells["VCUT_P50"]["exposure_unique_A"]["mean"],
            cells["HARD_MTK_BASE"]["exposure_A"]["mean"], cells["VCUT_MATCHED_LE"]["ALL"]["all"], le["gained"], le["lost"], le["p"], cells["VCUT_MATCHED_LE"]["blocks_used_A"]["mean"],
            cells["VCUT_MATCHED_LE"]["exposure_unique_A"]["mean"], cells["VCUT_MATCHED_GE (informational)"]["ALL"]["all"], cells["VCUT_HOME_ONLY_VOTE (diagnostic)"]["ALL"]["all"],
            cells["HARD_MTK_OWN_BLOCK_VOTE (diagnostic)"]["ALL"]["all"], cells["HARD_PHG_BASE"]["ALL"]["all"], verdict, time.time() - T0, os.path.relpath(fp, X.REPO)))
    for nm in ("hop1", "hop2", "hop3", "hop2+3"):
        if nm in masks:
            c = cells["VCUT_P50"]["vs_HARD_MTK_BASE"][nm]
            d = cells["VCUT_MATCHED_LE"]["vs_HARD_MTK_BASE"][nm]
            log("  %s n %d: P50 %.4f vs hard %.4f (+%d/-%d p=%.3g) | MATCHED_LE %.4f (+%d/-%d p=%.3g)" % (nm, c["n"], c["cell_ALL"], c["base_ALL"], c["gained"], c["lost"], c["p"],
                                                                                                          d["cell_ALL"], d["gained"], d["lost"], d["p"]))
    log("  reach: %s" % json.dumps(dec))
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "metaqa")
