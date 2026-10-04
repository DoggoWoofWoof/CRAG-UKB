"""L1_COVPART step 3 (DEV_A): the frozen selector on the scratch partition vs on the frozen partition -- same queries, same served
hits, same numerics; only the node -> block map changes.

    python -u _l1c_eval.py <cache> <tag>[,<tag>...]        tags name results/L1_COVPART/parts/<ds>__<tag>.npy

Reported per partition: ALL-gold P50 (BASE), per hop, paired McNemar vs the frozen partition's BASE; the failure decomposition
(infeasible / UNREACHED / reached-but-weak / fusion-fixable); reach_all and its paired test (the lane's direct hypothesis test:
does the partition convert UNREACHED queries?); median rank of the worst gold block among failed queries; structural
statistics of the partition (block sizes, directed STRUCT cut, family KM1s, fraction of non-hub 2-hop balls and stars kept
inside one block).  Secondary, informational: BASE + the confirmed directional channel (D2d) on the same partition.
Record: results/L1_COVPART/eval_A_<cache>.json.  DEV_B is not read.
"""
import io
import json
import os
import sys
import time

import numpy as np

import _l1g_candidate as CAND
import _l1g_core as G
from src.l1_canonical import hypergraph as HG  # noqa: E402  (imported, never edited)

OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
PDIR = os.path.join(OUT, "parts")
CAP = 100
name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
tags = sys.argv[2].split(",") if len(sys.argv) > 2 else ["H4_SK_PATH2"]
ds = G.X.DATASET_OF[name] if hasattr(G.X, "DATASET_OF") else name.replace("_phg", "")
t0 = time.time()
D = G.Data(name, dense_fp32=True)
C = D.C
nq, N = D.nq, D.N
m = D.A
hops = np.asarray(C.hops)
frozen_hard = D.hard.copy()
frozen_npart = D.npart
xu, au = D.cd.struct_csr(directed=False); au = np.asarray(au, np.int64); xu = np.asarray(xu, np.int64)
xo, ao = D.cd.struct_csr(directed=True); ao = np.asarray(ao, np.int64); xo = np.asarray(xo, np.int64)
D.cd._csr.clear()
deg_u = np.diff(xu)
hub = (deg_u + 1) > CAP
nonhub = ~hub
src_o = np.repeat(np.arange(N, dtype=np.int64), np.diff(xo))
# hypergraph families for KM1 accounting (frozen H4_SK + the PATH2 family from the scratch build, if present)
fam = {}
zf = np.load(os.path.join(D.cd.derived_dir, "hypergraph", "H4_SK.npz"))
mf = json.load(io.open(os.path.join(D.cd.derived_dir, "hypergraph", "H4_SK.json"), encoding="utf-8"))
nS = mf["per_family"]["STRUCT"]["hyperedges_out"]
ef, xf, wf = zf["eptr"].astype(np.int64), zf["eidx"].astype(np.int64), zf["ew"].astype(np.int64)
fam["STRUCT"] = (ef[:nS + 1], xf[:ef[nS]], wf[:nS])
fam["KNN"] = (ef[nS:] - ef[nS], xf[ef[nS]:], wf[nS:])
p2 = os.path.join(PDIR, "%s__H4_SK_PATH2.npz" % ds)
if os.path.exists(p2):
    z2 = np.load(p2)
    e2, x2, w2 = z2["eptr"].astype(np.int64), z2["eidx"].astype(np.int64), z2["ew"].astype(np.int64)
    nf = len(ef) - 1
    fam["PATH2"] = (e2[nf:] - e2[nf], x2[e2[nf]:], w2[nf:])


def km1(hard, eptr, eidx, ew):
    """weighted connectivity-minus-one and the fraction of hyperedges kept inside one block."""
    lam = np.zeros(len(eptr) - 1, np.int64)
    blk = hard[eidx]
    key = np.repeat(np.arange(len(eptr) - 1), np.diff(eptr)) * np.int64(hard.max() + 1) + blk
    u = np.unique(key)
    lam = np.bincount(u // np.int64(hard.max() + 1), minlength=len(eptr) - 1)
    return int(((lam - 1) * ew).sum()), round(float((lam == 1).mean()), 4), round(float(lam.mean()), 3)


def ball2_intact(hard):
    """fraction of non-hub nodes whose closed 2-hop non-hub neighbourhood (<= cap) lies in one block; mean blocks spanned."""
    keep, span = [], []
    for u in np.nonzero(nonhub)[0]:
        n1 = au[xu[u]:xu[u + 1]]
        n1e = n1[nonhub[n1]]
        n1 = n1[nonhub[n1]]
        if len(n1e):
            cnt = xu[n1e + 1] - xu[n1e]
            idx = np.repeat(xu[n1e], cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
            n2 = au[idx]
            ball = np.unique(np.concatenate([[u], n1, n2[nonhub[n2]]]))
        else:
            ball = np.unique(np.concatenate([[u], n1]))
        if len(ball) > CAP:
            continue
        b = np.unique(hard[ball])
        keep.append(len(b) == 1)
        span.append(len(b))
    return round(float(np.mean(keep)), 4), round(float(np.mean(span)), 3), int(len(keep))


def evaluate(tag, hard):
    D.override_partition(hard, tag)
    npart = D.npart
    ev = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)
    Cd, Cs = G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)
    base = G.F0([Cd, Cs], npart)
    pos = G.positions(base)
    gm = G.gold_mask(D)
    gs = gm.sum(axis=1)
    feas = gs <= G.P_MAIN
    base_all = (gm & (pos < G.P_MAIN)).sum(axis=1) == gs
    assert (base_all == D.base_all).all()
    reach = (gm & ~ev).sum(axis=1) == 0
    in50 = (G.positions(Cd) < G.P_MAIN) | (G.positions(Cs) < G.P_MAIN)
    union50 = ((gm & ~in50).sum(axis=1) == 0) & feas
    fail = ~base_all
    worst = np.where(gm, np.where(ev, pos, npart), -1).max(axis=1)
    out = {"blocks": int(npart), "size_min": int(D.sizes.min()), "size_max": int(D.sizes.max()),
           "gold_blocks_per_q": round(float(gs[m].mean()), 3), "feasible": round(float(feas[m].mean()), 4),
           "BASE_ALL": round(float(base_all[m].mean()), 4), "reach_all": round(float(reach[m].mean()), 4),
           "fail_pts": round(float(fail[m].mean()) * 100, 1),
           "fail_infeasible_pts": round(float((fail & ~feas)[m].mean()) * 100, 1),
           "fail_UNREACHED_pts": round(float((fail & feas & ~reach)[m].mean()) * 100, 1),
           "fail_reached_weak_pts": round(float((fail & feas & reach & ~union50)[m].mean()) * 100, 1),
           "fail_fusion_fixable_pts": round(float((fail & feas & reach & union50)[m].mean()) * 100, 1),
           "worst_gold_block_rank_median_failed": int(np.median(worst[m & fail])) if (m & fail).any() else None,
           "worst_gold_block_rank_median_reached_failed": int(np.median(worst[m & fail & reach])) if (m & fail & reach).any() else None,
           "scope_nodes_P50": round(float(np.mean([D.sizes[base[i, :G.P_MAIN]].sum() for i in np.nonzero(m)[0]])), 1),
           "per_hop": {}}
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        out["per_hop"]["hop%d" % h] = {"BASE_ALL": round(float(base_all[s].mean()), 4), "reach_all": round(float(reach[s].mean()), 4),
                                       "UNREACHED_pts": round(float((fail & feas & ~reach)[s].mean()) * 100, 1)}
    # structure
    hard64 = hard.astype(np.int64)
    out["struct_directed_cut_fraction"] = round(float((hard64[src_o] != hard64[ao]).mean()), 4)
    out["family_km1"] = {}
    for f, (e, x, w) in fam.items():
        k1, intact, lam = km1(hard64, e, x, w)
        out["family_km1"][f] = {"km1_weighted": k1, "hyperedges_in_one_block": intact, "mean_blocks_spanned": lam}
    bi, bs, nb = ball2_intact(hard64)
    out["ball2_nonhub_le_cap_in_one_block"] = bi
    out["ball2_nonhub_le_cap_mean_blocks_spanned"] = bs
    out["ball2_nonhub_le_cap_count"] = nb
    # secondary: the confirmed directional channel on this partition
    rdir = CAND.directional_block_rank(D)
    d2d = G.F0([Cd, Cs, rdir], npart)
    o, v = D.eval_rank(d2d, "D2d[%s]" % tag, quiet=True)
    out["D2d_ALL"] = o["A"]["ALL"]
    out["D2d_per_hop"] = {k: v_ for k, v_ in o["A"].items() if k.startswith("hop")}
    return out, base_all, reach, v.astype(bool)


res = {"cache": name, "partition_frozen": G.PARTITION_OF[name], "n_DEV_A": int(m.sum()), "families_accounted": list(fam), "arms": {}}
fro, ba0, re0, d0 = evaluate("FROZEN", frozen_hard)
res["arms"]["FROZEN"] = fro
G.log("FROZEN  %s" % json.dumps({k: fro[k] for k in ("BASE_ALL", "reach_all", "fail_UNREACHED_pts", "fail_reached_weak_pts", "fail_fusion_fixable_pts", "worst_gold_block_rank_median_failed", "D2d_ALL")}))
for tag in tags:
    p = os.path.join(PDIR, "%s__%s.npy" % (ds, tag))
    if not os.path.exists(p):
        G.log("%s: no partition file (%s)" % (tag, p))
        res["arms"][tag] = {"status": "NO_PARTITION"}
        continue
    hard = np.load(p).astype(np.int64)
    e, ba, re_, dd = evaluate(tag, hard)
    for lab, a0, a1 in (("BASE_ALL_vs_FROZEN", ba0, ba), ("reach_all_vs_FROZEN", re0, re_), ("D2d_ALL_vs_FROZEN_D2d", d0, dd)):
        g, l, pv = G.X.mcnemar(a0[m], a1[m])
        e[lab] = {"gained": g, "lost": l, "p": pv}
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        g, l, pv = G.X.mcnemar(ba0[s], ba[s])
        e["per_hop"]["hop%d" % h]["BASE_vs_FROZEN"] = {"gained": g, "lost": l, "p": pv}
    run = os.path.join(PDIR, "%s__%s.RUN.json" % (ds, tag))
    if os.path.exists(run):
        r = json.load(io.open(run, encoding="utf-8"))
        e["partition_run"] = {"worker_stats": r.get("worker_stats"), "guard": r.get("guard"), "hypergraph": r.get("hypergraph")}
    res["arms"][tag] = e
    G.log("%-12s %s" % (tag, json.dumps({k: e[k] for k in ("BASE_ALL", "BASE_ALL_vs_FROZEN", "reach_all", "reach_all_vs_FROZEN", "fail_UNREACHED_pts", "fail_reached_weak_pts", "fail_fusion_fixable_pts", "worst_gold_block_rank_median_failed", "D2d_ALL", "D2d_ALL_vs_FROZEN_D2d")})))
    G.log("             per hop %s" % json.dumps(e["per_hop"]))
    G.log("             structure %s | ball2 intact %s (span %s of %d) | frozen: %s | ball2 intact %s (span %s)" % (
        json.dumps(e["family_km1"]), e["ball2_nonhub_le_cap_in_one_block"], e["ball2_nonhub_le_cap_mean_blocks_spanned"], e["ball2_nonhub_le_cap_count"],
        json.dumps(fro["family_km1"]), fro["ball2_nonhub_le_cap_in_one_block"], fro["ball2_nonhub_le_cap_mean_blocks_spanned"]))
D.override_partition(frozen_hard, "FROZEN")
G.S.wj(os.path.join(OUT, "eval_A_%s.json" % name), res)
G.log("done %.0fs" % (time.time() - t0))
