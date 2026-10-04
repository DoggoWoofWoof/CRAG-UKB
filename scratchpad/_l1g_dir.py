"""Directional evidence (DEV_A): use the query offset as an ORIENTATION test over real nodes, not as a point prediction.
For the frozen SEED_K=5 seeds s_i of a query and every actual node v:
    D_i(v) = cos(q - e_{s_i}, e_v - e_{s_i})            ("from this seed, does v lie in the direction the query indicates?")
Scored exactly over ALL nodes (no candidate pool needed at these corpus sizes), then:
    D1a  node score mean_i D_i(v)          -> top-100 nodes -> legacy table -> frozen block channel
    D1b  node RRF over the 5 per-seed rankings (top-200 lists, missing = rank 200, frozen K0) -> top-100 -> block channel
    D1c  block score max_{v in B} mean_i D_i(v) over the block's actual nodes -> block ranking (no centroid, no table)
    D1d  per-seed block score max_{v in B} D_i(v) -> 5 block rankings -> frozen RRF -> block ranking
    D2   Dense block rank + SPLADE block rank + D1x -> frozen RRF -> P50
Diagnostics: where do the GOLD nodes rank under the directional score vs under Dense(q)?  (orientation information test)
No relation labels, no fitted threshold, no learned offset, no traversal."""
import os
import sys
import time

import numpy as np

import _l1g_core as G
import _ta_prepartition as TA

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
t0 = time.time()
D = G.Data(name, dense_fp32=True)
C = D.C
E, Q = D.E, D.Q
N, dim = E.shape
npart, nq = D.npart, D.nq
m = D.A
K0, KL = G.K0, G.K_LOCK
TOP200 = TA.TOP200
seeds = C.seeds.astype(np.int64)                    # (nq, 5)
ns = seeds.shape[1]
res = {"cache": name, "partition": G.PARTITION_OF[name], "n_DEV_A": int(m.sum()), "BASE_A": D.r_base["ALL_split"]["A"], "arms": {}, "diag": {}}
G.log("%s [%s] DEV_A n=%d N=%d blocks %d BASE %.4f" % (name, G.PARTITION_OF[name], int(m.sum()), N, npart, res["BASE_A"]["ALL"]))

# block membership for max-over-actual-nodes: nodes sorted by block, reduceat boundaries
hard = D.hard.astype(np.int64)
perm = np.argsort(hard, kind="stable")
starts = np.searchsorted(hard[perm], np.arange(npart))
assert (np.bincount(hard, minlength=npart) > 0).all()

Es = E[seeds]                                        # (nq, 5, dim) unit
R = Q[:, None, :] - Es                               # query directions from each seed
Rn = R / (np.linalg.norm(R, axis=2, keepdims=True) + 1e-9)
rs_dot = np.einsum("qsd,qsd->qs", Rn, Es)            # Rn_i . e_{s_i}

ids_a = np.zeros((nq, KL), np.int64)                 # D1a top-100 by mean cosine
ids_b = np.zeros((nq, KL), np.int64)                 # D1b top-100 by per-seed node RRF
blk_c = np.zeros((nq, npart), np.int64)              # D1c block ranking
blk_d = np.zeros((nq, npart), np.int64)              # D1d block ranking
gold_best_rank_dir = np.full(nq, -1, np.int64)
gold_in100_dir = np.zeros(nq, bool)
gold_in100_dense = np.zeros(nq, bool)
gold_in100_rrf = np.zeros(nq, bool)
gold_in100_seed0 = np.zeros(nq, bool)                # first seed only (the RRF top-1 node as the single anchor)
gold_in100_max = np.zeros(nq, bool)                  # best anchor per node (max over seeds)
gold_in100_ext = np.zeros(nq, bool)                  # G2-style single point 2q - mu, for reference
mean_cos_all = np.zeros(nq, np.float32)
gold_nodes = [set(g.tolist()) for g in C.gold_nodes]
cq = max(8, min(200, int(2e7 // (ns * N))))
for a in range(0, nq, cq):
    b = min(nq, a + cq)
    n = b - a
    V = np.concatenate([Rn[a:b].reshape(n * ns, dim), Es[a:b].reshape(n * ns, dim)], axis=0)
    P = V @ E.T                                                                   # (2 n ns, N)
    A = P[:n * ns].reshape(n, ns, N)                                              # Rn_i . e_v
    B = P[n * ns:].reshape(n, ns, N)                                              # e_{s_i} . e_v
    num = A - rs_dot[a:b][:, :, None]                                             # Rn_i . (e_v - e_{s_i})
    den = np.sqrt(np.maximum(2.0 - 2.0 * B, 1e-12))                               # |e_v - e_{s_i}| (unit vectors)
    cos = num / den                                                               # (n, ns, N)
    for j in range(ns):                                                           # the seed itself is not a candidate
        cos[np.arange(n), j, seeds[a:b, j]] = -2.0
    mc = cos.mean(axis=1)                                                         # (n, N)
    mean_cos_all[a:b] = mc.mean(axis=1)
    mx = cos.max(axis=1)
    c0 = cos[:, 0]
    for i in range(n):
        gn = np.array(sorted(gold_nodes[a + i]), np.int64)
        if len(gn):
            gold_in100_seed0[a + i] = bool((c0[i, gn] >= np.partition(c0[i], N - KL)[N - KL]).any())
            gold_in100_max[a + i] = bool((mx[i, gn] >= np.partition(mx[i], N - KL)[N - KL]).any())
    top = np.argpartition(-mc, KL - 1, axis=1)[:, :KL]
    o = np.argsort(-np.take_along_axis(mc, top, axis=1), axis=1, kind="stable")
    ids_a[a:b] = np.take_along_axis(top, o, axis=1)
    # per-seed top-200 lists -> node RRF (frozen K0, missing = TOP200)
    for i in range(n):
        lists = []
        for j in range(ns):
            t = np.argpartition(-cos[i, j], TOP200 - 1)[:TOP200]
            lists.append(t[np.argsort(-cos[i, j][t], kind="stable")])
        pos = {}
        for j, lst in enumerate(lists):
            for k, v in enumerate(lst.tolist()):
                pos.setdefault(v, [TOP200] * ns)[j] = k
        uni = list(pos.keys())
        sc = np.array([sum(1.0 / (K0 + r) for r in pos[v]) for v in uni])
        order = np.argsort(-sc, kind="stable")[:KL]
        ids_b[a + i] = np.array(uni, np.int64)[order]
        # gold diagnostics
        gr = np.argsort(-mc[i], kind="stable")
        rk = np.empty(N, np.int64)
        rk[gr] = np.arange(N)
        gn = np.array(sorted(gold_nodes[a + i]), np.int64)
        if len(gn):
            gold_best_rank_dir[a + i] = int(rk[gn].min())
            gold_in100_dir[a + i] = bool((rk[gn] < KL).any())
            gold_in100_dense[a + i] = bool(len(set(D.d_ids[a + i, :KL].tolist()) & gold_nodes[a + i]) > 0)
            gold_in100_rrf[a + i] = bool(len(set(ids_b[a + i].tolist()) & gold_nodes[a + i]) > 0)
    # block max over ACTUAL nodes
    bm = np.maximum.reduceat(mc[:, perm], starts, axis=1)                         # (n, npart)
    blk_c[a:b] = np.argsort(-bm, axis=1, kind="stable")
    bms = np.maximum.reduceat(cos.reshape(n * ns, N)[:, perm], starts, axis=1).reshape(n, ns, npart)
    rks = [np.argsort(-bms[:, j], axis=1, kind="stable").astype(np.int64) for j in range(ns)]
    blk_d[a:b] = G.S.rrf_ranks(rks)
    if a == 0:
        G.log("  chunk %d queries in %.0fs (chunks of %d)" % (n, time.time() - t0, cq))
del E, D.E
G.log("  scored %d x %d x %d directional cosines in %.0fs" % (nq, ns, N, time.time() - t0))
res["diag"]["orientation"] = {
    "mean_cos_over_all_nodes": round(float(mean_cos_all[m].mean()), 4),
    "gold_best_rank_under_mean_cos_median": float(np.median(gold_best_rank_dir[m & (gold_best_rank_dir >= 0)])),
    "q_with_gold_node_in_top100_dense_q": round(float(gold_in100_dense[m].mean()), 4),
    "q_with_gold_node_in_top100_directional_mean": round(float(gold_in100_dir[m].mean()), 4),
    "q_with_gold_node_in_top100_directional_rrf": round(float(gold_in100_rrf[m].mean()), 4),
    "q_with_gold_node_in_top100_directional_seed0_only": round(float(gold_in100_seed0[m].mean()), 4),
    "q_with_gold_node_in_top100_directional_max_over_seeds": round(float(gold_in100_max[m].mean()), 4),
    "q_with_gold_node_in_top100_either": round(float((gold_in100_dense | gold_in100_dir)[m].mean()), 4)}
hops = C.hops
if (hops >= 0).any():
    res["diag"]["orientation"]["by_hop"] = {}
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        mm = m & (hops == h)
        res["diag"]["orientation"]["by_hop"]["hop%d" % h] = {
            "n": int(mm.sum()), "dense_q": round(float(gold_in100_dense[mm].mean()), 3), "dir_mean": round(float(gold_in100_dir[mm].mean()), 3),
            "dir_rrf": round(float(gold_in100_rrf[mm].mean()), 3), "dir_seed0": round(float(gold_in100_seed0[mm].mean()), 3),
            "dir_max": round(float(gold_in100_max[mm].mean()), 3),
            "gold_best_rank_median": float(np.median(gold_best_rank_dir[mm & (gold_best_rank_dir >= 0)]))}
G.log("  orientation: %s" % res["diag"]["orientation"])
vecs = {}


def run(tag, rank):
    out, allv = D.eval_rank(rank, tag)
    res["arms"][tag] = out["A"]
    vecs[tag] = allv[m].astype(np.int8)


rd, rs = G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)
evd, evs = G.evidence(D, D.d_ids), G.evidence(D, D.s_ids)
gm = G.gold_mask(D)
ev_ref = evd | evs
run("D0 BASE (frozen RRF)", G.F0([rd, rs], npart))
ra, rb = G.block_channel(D, ids_a), G.block_channel(D, ids_b)
for tag, ids in (("D1a mean-cos nodes", ids_a), ("D1b per-seed node RRF", ids_b)):
    ne = G.new_evidence(D, ids, D.d_ids, gm, ev_ref, m)
    res["diag"].setdefault("new_evidence", {})[tag] = ne
    G.log("  %-26s new hits %.2f | q with new gold-node hit %.3f | q gaining an unreached gold block %.3f | reach -> %.3f" % (
        tag, ne["frac_new_hits"], ne["q_with_new_gold_node_hit"], ne["q_gaining_unreached_gold_block"], ne["reach_after"]))
run("D1a mean-cos nodes -> table | alone", ra)
run("D1b per-seed node RRF -> table | alone", rb)
run("D1c block max of mean-cos over actual nodes | alone", blk_c)
run("D1d per-seed block max -> RRF over seeds | alone", blk_d)
run("D2a BASE + D1a (frozen RRF, 3 ch)", G.F0([rd, rs, ra], npart))
run("D2b BASE + D1b (frozen RRF, 3 ch)", G.F0([rd, rs, rb], npart))
run("D2c BASE + D1c (frozen RRF, 3 ch)", G.F0([rd, rs, blk_c], npart))
run("D2d BASE + D1d (frozen RRF, 3 ch)", G.F0([rd, rs, blk_d], npart))
run("D2ac BASE + D1a + D1c (frozen RRF, 4 ch)", G.F0([rd, rs, ra, blk_c], npart))
G.S.wj(os.path.join(G.OUT, "dir_A_%s.json" % name), res)
np.savez_compressed(os.path.join(G.OUT, "dirvecs_A_%s.npz" % name), rows=D.rows[m], **vecs)
G.log("done %.0fs" % (time.time() - t0))
