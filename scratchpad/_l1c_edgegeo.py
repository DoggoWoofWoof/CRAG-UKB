"""L1_COVPART, fifth ruling of 2026-09-14: EDGE / TRIPLET GEOMETRY VOTING -- valid graph facts as a parameter-free retrieval unit.

Universal rule (one equation everywhere; no relation semantics, no training, no new embeddings, no walk, no dataset switch):
    the structural unit is the undirected STRUCT edge t = {s, o} (the frozen expansion's deduped, loop-free key set);
    with the frozen unit node vectors e_v and the frozen unit query vector q, an edge receives closed-form scores from
        a_s = q.e_s, a_o = q.e_o, c = e_s.e_o            (three dot products; c is query-independent and precomputed once)
      S_mid(t) = cos(q, norm(e_s + e_o))                 = (a_s + a_o) / sqrt(2 + 2c)
      S_dir(t) = max over the two orientations of cos(q - e_s, e_o - e_s)
               = max( (a_o - a_s - c + 1) / (sqrt(2 - 2a_s) sqrt(2 - 2c)),  (a_s - a_o - c + 1) / (sqrt(2 - 2a_o) sqrt(2 - 2c)) )
      S_src / S_dst = similarity of the source / target endpoint under the better orientation   (information arm only)
    single-shot, independent scoring of every candidate edge; the top K_LOCK = 100 edges of a channel vote 1/(K0 + r) for {P(s), P(o)}
    through the frozen block-vote rule (sum + max, rr(S) + rr(M)); channels are fused by the frozen RRF (K0 = 60); P50.
Candidate sets: C1000 = edges with an endpoint in the query's existing node top-1000 pool (dense u SPLADE) [controlled, primary];
                ALL   = every STRUCT edge (brute-force preview of the global static index) [secondary].
Arms: T0 canonical BASE; T1 mid-edge channel alone; T2 dir-edge channel alone; T3 = RRF(node dense, node SPLADE, mid-edge, dir-edge);
      T3b = RRF(node dense, node SPLADE, triplet channel) with triplet rank = RRF(src, dst, dir, mid) [information].  DEV_A only.

    python -u _l1c_edgegeo.py <cache>        -> results/L1_COVPART/edgegeo_A_<cache>.json
"""
import json
import os
import sys
import time

import numpy as np

import _l1g_core as G

OUT = os.path.join(G.X.REPO, "results", "L1_COVPART")
name = sys.argv[1]
t0 = time.time()
D = G.Data(name, dense_fp32=False)
C = D.C
nq, N, npart = D.nq, D.N, D.npart
m = D.A
rowsA = np.nonzero(m)[0]
nA = len(rowsA)
hops = np.asarray(C.hops)
hard = D.hard.astype(np.int64)
K0, K_LOCK = G.S.K0, G.K_LOCK
P_MAIN = G.P_MAIN
EB = 20000                                                   # node-embedding block (rows of the fp16 memmap read at once)
QB = 128                                                     # DEV_A queries scored per pass over the embeddings

# ---------------------------------------------------------------- edges (undirected, deduped, loop-free: the frozen STRUCT key set)
_, STRUCT, _, _ = D.cd.keysets()
s_e = (STRUCT // np.int64(N)).astype(np.int64)
o_e = (STRUCT % np.int64(N)).astype(np.int64)
nE = len(s_e)
bs_e, bo_e = hard[s_e], hard[o_e]
dim = int(D.Q.shape[1])
G.log("%s: N %d blocks %d DEV_A %d; STRUCT undirected edges %d; dim %d" % (name, N, npart, nA, nE, dim))

# ---------------------------------------------------------------- node norms (the formulas assume unit vectors) and c[t] = e_s . e_o
norms = np.zeros(N, np.float32)
for a in range(0, N, EB):
    b = min(N, a + EB)
    norms[a:b] = np.linalg.norm(np.asarray(D._E16[a:b], np.float32), axis=1)
norms = np.maximum(norms, 1e-9)
c_e = np.zeros(nE, np.float32)
CE = 8192
for a in range(0, nE, CE):
    b = min(nE, a + CE)
    Es = np.asarray(D._E16[s_e[a:b]], np.float32) / norms[s_e[a:b]][:, None]
    Eo = np.asarray(D._E16[o_e[a:b]], np.float32) / norms[o_e[a:b]][:, None]
    c_e[a:b] = (Es * Eo).sum(axis=1)
G.log("node norms mean %.4f (min %.4f max %.4f); edge endpoint cos mean %.3f p50 %.3f; %.0fs" % (norms.mean(), norms.min(), norms.max(), c_e.mean(), np.median(c_e), time.time() - t0))

# ---------------------------------------------------------------- per-query pool (existing node top-1000, dense u SPLADE) and top-100 hit sets
d1000 = np.asarray(D.d_ids[rowsA, :1000], np.int64)
s1000 = np.asarray(D.s_ids[rowsA, :1000], np.int64)
top100 = [set(int(x) for x in D.d_ids[i, :K_LOCK] if x >= 0) | set(int(x) for x in D.s_ids[i, :K_LOCK] if x >= 0) for i in rowsA]


def scores(a_row):
    """all four closed-form edge scores for one query from its node similarities a_row = q . e_v (arrays over all edges)."""
    a_s, a_o, c = a_row[s_e], a_row[o_e], c_e
    den_c = np.sqrt(np.maximum(2.0 - 2.0 * c, 1e-12))
    mid = (a_s + a_o) / np.sqrt(np.maximum(2.0 + 2.0 * c, 1e-12))
    d_so = (a_o - a_s - c + 1.0) / (np.sqrt(np.maximum(2.0 - 2.0 * a_s, 1e-12)) * den_c)
    d_os = (a_s - a_o - c + 1.0) / (np.sqrt(np.maximum(2.0 - 2.0 * a_o, 1e-12)) * den_c)
    degenerate = (2.0 - 2.0 * c) < 1e-6                                   # identical endpoint vectors: the edge has no direction
    d_so[degenerate] = -1.0
    d_os[degenerate] = -1.0
    so = d_so >= d_os
    dr = np.where(so, d_so, d_os)
    src = np.where(so, a_s, a_o)
    dst = np.where(so, a_o, a_s)
    return {"mid": mid.astype(np.float32), "dir": dr.astype(np.float32), "src": src.astype(np.float32), "dst": dst.astype(np.float32)}


def topk(score, k=K_LOCK):
    """indices of the k largest scores, descending, ties by index (stable)."""
    k = min(k, len(score))
    idx = np.argpartition(-score, k - 1)[:k]
    return idx[np.argsort(-score[idx], kind="stable")]


def rr_rank(vals):
    """rank (0 = best) of each entry of vals (descending), ties by index (stable)."""
    order = np.argsort(-vals, kind="stable")
    rank = np.empty(len(vals), np.int64)
    rank[order] = np.arange(len(vals))
    return rank


def edge_block_channel(top_edges):
    """top_edges: list over DEV_A queries of ranked edge indices -> (block rank (nq, npart), S (nq, npart)).
    frozen block-vote rule: the edge at rank r adds 1/(K0 + r) to each endpoint block (sum S and max M); votes = rr(S) + rr(M)."""
    S = np.zeros((nq, npart), np.float32)
    M = np.zeros((nq, npart), np.float32)
    for j, qi in enumerate(rowsA):
        for r, e in enumerate(top_edges[j]):
            w = 1.0 / (K0 + r)
            ps = np.unique(np.array([bs_e[e], bo_e[e]]))
            S[qi, ps] += w
            np.maximum.at(M[qi], ps, w)

    def rr(score):
        order = np.argsort(-score, axis=1)
        rank = np.empty((nq, npart), np.int32)
        rank[np.arange(nq)[:, None], order] = np.arange(npart)[None, :]
        return 1.0 / (K0 + rank)
    votes = rr(S) + rr(M)
    return np.argsort(-votes, axis=1).astype(np.int64), S


# ---------------------------------------------------------------- score every DEV_A query (chunked passes over the fp16 embeddings)
tops = {cs: {k: [] for k in ("mid", "dir", "trip")} for cs in ("C1000", "ALL")}
stats = {cs: {"cand_edges_mean": 0.0, "top100_edges_with_no_endpoint_in_node_top100": {"mid": 0.0, "dir": 0.0, "trip": 0.0}} for cs in tops}
QA = D.Q[rowsA].astype(np.float32)
pm = np.zeros(N, bool)
for q0 in range(0, nA, QB):
    q1 = min(nA, q0 + QB)
    A = np.zeros((q1 - q0, N), np.float32)
    for a in range(0, N, EB):
        b = min(N, a + EB)
        A[:, a:b] = QA[q0:q1] @ (np.asarray(D._E16[a:b], np.float32) / norms[a:b][:, None]).T
    for j in range(q0, q1):
        sc = scores(A[j - q0])
        pm[:] = False
        pm[d1000[j][d1000[j] >= 0]] = True
        pm[s1000[j][s1000[j] >= 0]] = True
        mask = pm[s_e] | pm[o_e]
        cand_c = np.nonzero(mask)[0]
        stats["C1000"]["cand_edges_mean"] += float(len(cand_c))
        stats["ALL"]["cand_edges_mean"] += float(nE)
        for cs, cand in (("C1000", cand_c), ("ALL", None)):
            if cand is None:
                t_mid, t_dir = topk(sc["mid"]), topk(sc["dir"])
                fused = sum(1.0 / (K0 + rr_rank(sc[k])) for k in ("src", "dst", "dir", "mid"))
                t_trip = topk(fused)
            elif len(cand) == 0:
                t_mid = t_dir = t_trip = np.zeros(0, np.int64)
            else:
                t_mid, t_dir = cand[topk(sc["mid"][cand])], cand[topk(sc["dir"][cand])]
                fused = sum(1.0 / (K0 + rr_rank(sc[k][cand])) for k in ("src", "dst", "dir", "mid"))
                t_trip = cand[topk(fused)]
            tops[cs]["mid"].append(t_mid)
            tops[cs]["dir"].append(t_dir)
            tops[cs]["trip"].append(t_trip)
            h100 = top100[j]
            for k, t in (("mid", t_mid), ("dir", t_dir), ("trip", t_trip)):
                if len(t):
                    stats[cs]["top100_edges_with_no_endpoint_in_node_top100"][k] += float(np.mean([(int(s_e[e]) not in h100) and (int(o_e[e]) not in h100) for e in t]))
    del A
    G.log("  scored %d / %d queries (%.0fs)" % (q1, nA, time.time() - t0))
for cs in stats:
    stats[cs]["cand_edges_mean"] = round(stats[cs]["cand_edges_mean"] / nA, 1)
    for k in stats[cs]["top100_edges_with_no_endpoint_in_node_top100"]:
        stats[cs]["top100_edges_with_no_endpoint_in_node_top100"][k] = round(stats[cs]["top100_edges_with_no_endpoint_in_node_top100"][k] / nA, 4)
G.log("candidate stats %s" % json.dumps(stats))
CH_E = {cs: {k: edge_block_channel(tops[cs][k]) for k in ("mid", "dir", "trip")} for cs in tops}
G.log("edge block channels built (%.0fs)" % (time.time() - t0))

# ---------------------------------------------------------------- node channels (served) and evaluation
Cd, Cs = G.block_channel(D, D.d_ids), G.block_channel(D, D.s_ids)
ev_node = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)
gm = G.gold_mask(D)
gs = gm.sum(axis=1)
feas = gs <= P_MAIN


def evaluate(tag, chans, evs):
    """chans: list of (nq, npart) block ranks fused by the frozen RRF (a single channel -> its own order); evs: evidence masks."""
    base = G.F0(chans, npart) if len(chans) > 1 else chans[0]
    pos = G.positions(base)
    ev = np.zeros((nq, npart), bool)
    for e in evs:
        ev |= e
    in50 = pos < P_MAIN
    base_all = (gm & in50).sum(axis=1) == gs
    reach = (gm & ~ev).sum(axis=1) == 0
    union = np.zeros((nq, npart), bool)
    for ch in chans:
        union |= G.positions(ch) < P_MAIN
    union50 = ((gm & ~union).sum(axis=1) == 0) & feas
    fail = ~base_all
    worst = np.where(gm, np.where(ev, pos, npart), -1).max(axis=1)
    out = {"arm": tag,
           "BASE_ALL": round(float(base_all[m].mean()), 4), "ANY": round(float((gm & in50).any(axis=1)[m].mean()), 4),
           "reach_all": round(float(reach[m].mean()), 4), "feasible": round(float(feas[m].mean()), 4),
           "fail_pts": round(float(fail[m].mean()) * 100, 1),
           "fail_infeasible_pts": round(float((fail & ~feas)[m].mean()) * 100, 1),
           "fail_UNREACHED_pts": round(float((fail & feas & ~reach)[m].mean()) * 100, 1),
           "fail_reached_weak_pts": round(float((fail & feas & reach & ~union50)[m].mean()) * 100, 1),
           "fail_fusion_fixable_pts": round(float((fail & feas & reach & union50)[m].mean()) * 100, 1),
           "worst_gold_block_rank_median_reached_failed": int(np.median(worst[m & fail & reach])) if (m & fail & reach).any() else None,
           "worst_gold_block_rank_median_all_reached": int(np.median(worst[m & reach])) if (m & reach).any() else None,
           "blocks_voted_per_query_mean": round(float(ev[m].sum(axis=1).mean()), 1),
           "scope_nodes_P50": round(float(np.mean([D.sizes[base[i, :P_MAIN]].sum() for i in rowsA])), 1),
           "per_hop": {}}
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        out["per_hop"]["hop%d" % h] = {"n": int(s.sum()), "BASE_ALL": round(float(base_all[s].mean()), 4), "reach_all": round(float(reach[s].mean()), 4),
                                       "UNREACHED_pts": round(float((fail & feas & ~reach)[s].mean()) * 100, 1),
                                       "reached_weak_pts": round(float((fail & feas & reach & ~union50)[s].mean()) * 100, 1),
                                       "worst_gold_block_rank_median_reached_failed": int(np.median(worst[s & fail & reach])) if (s & fail & reach).any() else None}
    return out, base_all, reach


ARMS = [("T0", [Cd, Cs], [ev_node])]
for cs in ("C1000", "ALL"):
    (Cm, Sm), (Cr, Sr), (Ct, St) = CH_E[cs]["mid"], CH_E[cs]["dir"], CH_E[cs]["trip"]
    ARMS += [("T1_%s" % cs, [Cm], [Sm > 0]), ("T2_%s" % cs, [Cr], [Sr > 0]),
             ("T3_%s" % cs, [Cd, Cs, Cm, Cr], [ev_node, Sm > 0, Sr > 0]), ("T3b_%s" % cs, [Cd, Cs, Ct], [ev_node, St > 0])]
res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": nA, "N": N, "edges": int(nE), "dim": dim,
       "node_norms": {"mean": round(float(norms.mean()), 4), "min": round(float(norms.min()), 4), "max": round(float(norms.max()), 4)},
       "edge_endpoint_cos": {"mean": round(float(c_e.mean()), 4), "p50": round(float(np.median(c_e)), 4)},
       "candidate_stats": stats,
       "definitions": {"unit": "undirected STRUCT edge {s,o} (frozen key set); scores from a_s = q.e_s, a_o = q.e_o, c = e_s.e_o (unit vectors)",
                       "S_mid": "(a_s + a_o) / sqrt(2 + 2c)", "S_dir": "max over orientations of (a_o - a_s - c + 1) / (sqrt(2 - 2a_s) sqrt(2 - 2c))",
                       "votes": "top K_LOCK=100 edges per channel, edge at rank r adds 1/(K0+r) to {P(s), P(o)}; rr(S)+rr(M); frozen RRF K0=60; P50",
                       "C1000": "edges with an endpoint in the query's node top-1000 pool (dense u SPLADE)", "ALL": "every STRUCT edge",
                       "T3b": "triplet rank = RRF(src, dst, dir, mid) over the candidate edges -> one channel (information)"},
       "arms": {}}
kept = {}
for tag, chans, evs in ARMS:
    o, b, r = evaluate(tag, chans, evs)
    kept[tag] = (b, r)
    if tag == "T0":
        assert abs(o["BASE_ALL"] - round(float(D.base_all[m].mean()), 4)) < 1e-9, (o["BASE_ALL"], float(D.base_all[m].mean()))
    else:
        for key, cur, base_ in (("BASE_ALL", b, kept["T0"][0]), ("reach_all", r, kept["T0"][1])):
            g_, l_, p_ = G.X.mcnemar(base_[m], cur[m])
            o[key + "_vs_T0"] = {"gained": g_, "lost": l_, "p": p_}
        for h in o["per_hop"]:
            s = m & (hops == int(h[3:]))
            g_, l_, p_ = G.X.mcnemar(kept["T0"][0][s], b[s])
            o["per_hop"][h]["BASE_vs_T0"] = {"gained": g_, "lost": l_, "p": p_}
        if o["per_hop"]:
            s = m & (hops >= 2)
            g_, l_, p_ = G.X.mcnemar(kept["T0"][0][s], b[s])
            o["hop2_hop3_pooled_BASE_vs_T0"] = {"n": int(s.sum()), "gained": g_, "lost": l_, "p": p_}
    res["arms"][tag] = o
    G.log("%-10s %s" % (tag, json.dumps({k: o[k] for k in ("BASE_ALL", "ANY", "reach_all", "fail_UNREACHED_pts", "fail_reached_weak_pts", "fail_fusion_fixable_pts",
                                                           "worst_gold_block_rank_median_reached_failed", "blocks_voted_per_query_mean", "scope_nodes_P50")})))
    if o["per_hop"]:
        G.log("           per hop %s" % json.dumps({h: (x["BASE_ALL"], x["reach_all"], x["UNREACHED_pts"], x["reached_weak_pts"]) for h, x in o["per_hop"].items()}))
    if tag != "T0":
        G.log("           vs T0: BASE %s reach %s | %s | hop2+3 pooled %s" % (o["BASE_ALL_vs_T0"], o["reach_all_vs_T0"],
                                                                           {h: x["BASE_vs_T0"] for h, x in o["per_hop"].items()}, o.get("hop2_hop3_pooled_BASE_vs_T0")))
G.S.wj(os.path.join(OUT, "edgegeo_A_%s.json" % name), res)
G.log("done %.0fs" % (time.time() - t0))
