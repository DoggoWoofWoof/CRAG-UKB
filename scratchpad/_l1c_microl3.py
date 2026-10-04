"""MICRO_L3_H2 -- a tiny, bounded, dynamic graph expansion that repairs L1's P50 (ruling of 2026-09-14, sixth of the day).

Cascade:  canonical L1 (Dense + SPLADE -> static block voting -> initial P50)  ->  uL3 (this module)  ->  repaired P50.
uL3 (same algorithm on every graph; no training, no modified embeddings, no relation lists, no dataset rule):
    B0      = the frozen SEED_K = 5 L1 seeds of the replay cache (RRF top-5 entity hits)
    hop h   = 1, 2:  candidate transitions v -> u for every v in B(h-1) over the ACTUAL STRUCT adjacency under the frozen
              admissibility policy (all directed out-edges; in-edges only when v is a non-hub, hub := deg_undirected + 1 > cap = N/k),
              u not visited yet; every candidate transition is scored with the frozen node geometry only
                  S_dst = cos(q, e_u)      S_dir = cos(q - e_v, e_u - e_v)      S_mid = cos(q, norm(e_v + e_u))
              ranks fused by the frozen RRF (K0 = 60); B(h) = the first BEAM = K_LOCK = 100 distinct u in fused order
    repair  = the visited nodes in order (B(1) then B(2)) vote through the served static membership table exactly like L1 hits
              (w_r = 1/(K0 + r), frozen rr(S) + rr(M)); unvoted blocks stay unranked; the repaired P50 = the frozen symmetric RRF
              of the served fused block order (base_rank) and the repair order (_l1x90_core.fuse rule "RRF"); output exactly P50.
Arms:  L1 (served, asserted == frozen BASE);  MICRO_L3_H2 (real-edge beam, primary);  MICRO_L3_H1 = the same truncated after hop 1
       (breakdown, not a separate mechanism);  LATENT_H2 / LATENT_H1 (diagnostic: identical scoring, beam and repair, but the candidate
       set of a transition is EVERY node instead of the actual neighbours -- the latent global beam; metaqa caches only).
    python -u _l1c_microl3.py <cache> [--latent]      -> results/L1_COVPART/microl3_A_<cache>.json
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import _l1g_core as G

X = G.X
OUT = os.path.join(X.REPO, "results", "L1_COVPART")
name = sys.argv[1]
LATENT = "--latent" in sys.argv
t0 = time.time()
D = G.Data(name, dense_fp32=False)
C = D.C
nq, N, npart = D.nq, D.N, D.npart
m = D.A
rowsA = np.nonzero(m)[0]
nA = len(rowsA)
hops = np.asarray(C.hops)
hard = D.hard.astype(np.int64)
K0, K_LOCK, P_MAIN = G.K0, G.K_LOCK, G.P_MAIN
BEAM = K_LOCK                                                    # beam width per hop = the frozen router width
DEPTH = 2
cap = int(round(N / npart))
mem_ptr, mem_flat = np.asarray(D.mem[0], np.int64), np.asarray(D.mem[1], np.int64)

# ---------------------------------------------------------------- actual STRUCT adjacency + the frozen admissibility policy
xo, ao = D.cd.struct_csr(directed=True)
xo, ao = np.asarray(xo, np.int64), np.asarray(ao, np.int64)
xu, _ = D.cd.struct_csr(directed=False)
deg_u = np.diff(np.asarray(xu, np.int64))
D.cd._csr.clear()
hub = (deg_u + 1) > cap
A_in = sp.csr_matrix((np.ones(len(ao), np.int8), ao, xo), shape=(N, N)).T.tocsr()
xi, ai = np.asarray(A_in.indptr, np.int64), np.asarray(A_in.indices, np.int64)
del A_in
seeds = np.asarray(C.seeds, np.int64)
G.log("%s: N %d blocks %d cap %d hubs %d DEV_A %d; directed STRUCT edges %d; seeds %s; %s" % (name, N, npart, cap, int(hub.sum()), nA, len(ao), seeds.shape, "LATENT" if LATENT else "REAL"))


def admissible(v):
    """actual neighbours of v allowed by the frozen policy: out-neighbours always, in-neighbours only for a non-hub v."""
    out = ao[xo[v]:xo[v + 1]]
    if hub[v]:
        return out
    return np.concatenate([out, ai[xi[v]:xi[v + 1]]])


# ---------------------------------------------------------------- frozen geometry of a transition v -> u (unit vectors)
def transition_scores(a_v, a_u, c):
    den_c = np.sqrt(np.maximum(2.0 - 2.0 * c, 1e-12))
    dr = (a_u - a_v - c + 1.0) / (np.sqrt(np.maximum(2.0 - 2.0 * a_v, 1e-12)) * den_c)
    dr = np.where((2.0 - 2.0 * c) < 1e-6, -1.0, dr)
    mid = (a_v + a_u) / np.sqrt(np.maximum(2.0 + 2.0 * c, 1e-12))
    return a_u.astype(np.float32), dr.astype(np.float32), mid.astype(np.float32)


def rr_rank(vals):
    order = np.argsort(-vals, kind="stable")
    rank = np.empty(len(vals), np.int64)
    rank[order] = np.arange(len(vals))
    return rank


def fused_order(dst, dr, mid):
    f = sum(1.0 / (K0 + rr_rank(x)) for x in (dst, dr, mid))
    return np.argsort(-f, kind="stable")


def rows(idx):
    """unit fp32 rows of the frozen node embeddings for node ids idx (gathered from the fp16 memmap)."""
    E = np.asarray(D._E16[idx], np.float32)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


if LATENT:
    EF = np.zeros((N, D.Q.shape[1]), np.float32)
    for a in range(0, N, 20000):
        EF[a:min(N, a + 20000)] = rows(np.arange(a, min(N, a + 20000)))
    G.log("latent arm: full unit node matrix %s (%.0f MB)" % (EF.shape, EF.nbytes / 2 ** 20))


def expand_real(q, V, visited):
    """candidate transitions from the beam V over actual admissible edges -> (v_idx, u_idx, scores...)"""
    vs, us = [], []
    for v in V:
        u = admissible(int(v))
        if len(u):
            keep = ~visited[u]
            u = u[keep]
            vs.append(np.full(len(u), int(v), np.int64))
            us.append(u)
    if not us:
        return None
    v_idx, u_idx = np.concatenate(vs), np.concatenate(us)
    uniq, inv = np.unique(np.concatenate([v_idx, u_idx]), return_inverse=True)
    E = rows(uniq)
    a = E @ q
    iv, iu = inv[:len(v_idx)], inv[len(v_idx):]
    c = np.einsum("ij,ij->i", E[iv], E[iu])
    return v_idx, u_idx, transition_scores(a[iv], a[iu], c)


def expand_latent(q, V, visited, a_all):
    """the latent global beam: the same transition geometry, but every non-visited node is a candidate target of every v in V."""
    V = np.asarray(V, np.int64)
    Cm = EF[V] @ EF.T                                            # (|V|, N) cosines e_v . e_u
    a_v = np.repeat(a_all[V], N)
    a_u = np.tile(a_all, len(V))
    v_idx = np.repeat(V, N)
    u_idx = np.tile(np.arange(N, dtype=np.int64), len(V))
    keep = ~visited[u_idx] & (u_idx != v_idx)
    dst, dr, mid = transition_scores(a_v[keep], a_u[keep], Cm.ravel()[keep])
    return v_idx[keep], u_idx[keep], (dst, dr, mid)


def beam_search(j, qi):
    """returns per-hop beams (lists of node ids in fused order), edges scored per hop, wall-clock seconds."""
    q = D.Q[qi].astype(np.float32)
    visited = np.zeros(N, bool)
    B = [int(s) for s in seeds[qi] if s >= 0]
    visited[B] = True
    beams, n_edges = [], []
    a_all = EF @ q if LATENT else None
    t = time.perf_counter()
    for h in range(DEPTH):
        cand = expand_latent(q, B, visited, a_all) if LATENT else expand_real(q, B, visited)
        if cand is None:
            beams.append([])
            n_edges.append(0)
            B = []
            continue
        v_idx, u_idx, (dst, dr, mid) = cand
        n_edges.append(int(len(u_idx)))
        order = fused_order(dst, dr, mid)
        nb = []
        for k in order:
            u = int(u_idx[k])
            if not visited[u]:
                visited[u] = True
                nb.append(u)
                if len(nb) >= BEAM:
                    break
        beams.append(nb)
        B = nb
    return beams, n_edges, time.perf_counter() - t


def repair_channel(lists):
    """visited-node lists (per cache row; empty outside DEV_A) -> (block order with unvoted blocks = -1, S)  [frozen vote rule]."""
    S = np.zeros((nq, npart), np.float32)
    M = np.zeros((nq, npart), np.float32)
    for qi in range(nq):
        for r, nd in enumerate(lists[qi]):
            w = 1.0 / (K0 + r)
            ps = mem_flat[mem_ptr[nd]:mem_ptr[nd + 1]]
            S[qi, ps] += w
            np.maximum.at(M[qi], ps, w)

    def rr(score):
        order = np.argsort(-score, axis=1)
        rank = np.empty((nq, npart), np.int32)
        rank[np.arange(nq)[:, None], order] = np.arange(npart)[None, :]
        return 1.0 / (K0 + rank)
    votes = rr(S) + rr(M)
    order = np.argsort(-votes, axis=1).astype(np.int64)
    voted = np.take_along_axis(S > 0, order, axis=1)
    return np.where(voted, order, -1), S


# ---------------------------------------------------------------- run the beam on DEV_A
beams1, beams2 = [[] for _ in range(nq)], [[] for _ in range(nq)]
edges_h = np.zeros((nA, DEPTH), np.int64)
secs = np.zeros(nA, np.float64)
seed_all_hub = np.zeros(nA, bool)
for j, qi in enumerate(rowsA):
    beams, n_edges, dt = beam_search(j, qi)
    beams1[qi], beams2[qi] = beams[0], beams[1]
    edges_h[j] = n_edges
    secs[j] = dt
    seed_all_hub[j] = bool(hub[seeds[qi][seeds[qi] >= 0]].all())
    if j % 200 == 0:
        G.log("  %d / %d queries (%.0fs) edges %s beam %d/%d %.1f ms" % (j, nA, time.time() - t0, n_edges, len(beams[0]), len(beams[1]), 1000 * dt))
nodes_visited = np.array([len(beams1[qi]) + len(beams2[qi]) for qi in rowsA])
G.log("beam done: edges scored / query hop1 %.1f hop2 %.1f (total mean %.1f, p95 %.0f); nodes visited / query mean %.1f; wall-clock / query mean %.1f ms p50 %.1f p95 %.1f; seeds all hubs %d / %d" % (
    edges_h[:, 0].mean(), edges_h[:, 1].mean(), edges_h.sum(1).mean(), np.percentile(edges_h.sum(1), 95), nodes_visited.mean(),
    1000 * secs.mean(), 1000 * np.median(secs), 1000 * np.percentile(secs, 95), int(seed_all_hub.sum()), nA))

# ---------------------------------------------------------------- repair + evaluation
t1 = time.time()
rep1, S1 = repair_channel(beams1)
rep2, S2 = repair_channel([beams1[qi] + beams2[qi] for qi in range(nq)])
base_rank = np.asarray(C.base_rank, np.int64)
sel0 = X.fuse(C, base_rank, base_rank, "T")
sel1 = X.fuse(C, base_rank, rep1, "RRF")
sel2 = X.fuse(C, base_rank, rep2, "RRF")
t_repair = (time.time() - t1) / max(nA, 1)
gm = G.gold_mask(D)
gs = gm.sum(axis=1)
feas = gs <= P_MAIN
ev_node = G.evidence(D, D.d_ids) | G.evidence(D, D.s_ids)
tag = ("LATENT" if LATENT else "MICRO_L3")


def evaluate(arm, sel, ev):
    allv, anyv = C.cover(sel)
    allv, anyv = allv.astype(bool), anyv.astype(bool)
    reach = (gm & ~ev).sum(axis=1) == 0
    fail = ~allv
    pos = np.full((nq, npart), npart, np.int64)
    for i, s in enumerate(sel):
        pos[i, np.asarray(s, np.int64)] = np.arange(len(s))
    out = {"arm": arm, "BASE_ALL": round(float(allv[m].mean()), 4), "ANY": round(float(anyv[m].mean()), 4),
           "reach_all": round(float(reach[m].mean()), 4), "feasible": round(float(feas[m].mean()), 4),
           "fail_pts": round(float(fail[m].mean()) * 100, 1),
           "fail_UNREACHED_pts": round(float((fail & feas & ~reach)[m].mean()) * 100, 1),
           "fail_reached_pts": round(float((fail & feas & reach)[m].mean()) * 100, 1),
           "blocks_voted_per_query_mean": round(float(ev[m].sum(axis=1).mean()), 1),
           "scope_nodes_P50": round(float(np.mean([D.sizes[np.asarray(sel[i], np.int64)].sum() for i in rowsA])), 1), "per_hop": {}}
    for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
        s = m & (hops == h)
        out["per_hop"]["hop%d" % h] = {"n": int(s.sum()), "BASE_ALL": round(float(allv[s].mean()), 4), "reach_all": round(float(reach[s].mean()), 4),
                                       "UNREACHED_pts": round(float((fail & feas & ~reach)[s].mean()) * 100, 1),
                                       "reached_failed_pts": round(float((fail & feas & reach)[s].mean()) * 100, 1)}
    return out, allv, reach


res = {"cache": name, "partition": G.PARTITION_OF.get(name), "n_DEV_A": nA, "N": N, "cap": cap, "hubs": int(hub.sum()), "arm_family": tag,
       "constants": {"SEED_K": int(seeds.shape[1]), "BEAM": BEAM, "DEPTH": DEPTH, "K0": K0, "P50": P_MAIN, "hub_rule": "deg_undirected + 1 > cap = N / npart (frozen)"},
       "cost": {"edges_scored_per_query_mean": round(float(edges_h.sum(1).mean()), 1), "edges_scored_per_query_p95": int(np.percentile(edges_h.sum(1), 95)),
                "edges_scored_hop1_mean": round(float(edges_h[:, 0].mean()), 1), "edges_scored_hop2_mean": round(float(edges_h[:, 1].mean()), 1),
                "nodes_visited_per_query_mean": round(float(nodes_visited.mean()), 1),
                "beam_wall_clock_ms_mean": round(1000 * float(secs.mean()), 2), "beam_wall_clock_ms_p50": round(1000 * float(np.median(secs)), 2),
                "beam_wall_clock_ms_p95": round(1000 * float(np.percentile(secs, 95)), 2),
                "repair_and_fusion_ms_per_query": round(1000 * t_repair, 2), "queries_whose_seeds_are_all_hubs": int(seed_all_hub.sum())},
       "arms": {}}
kept = {}
for arm, sel, ev in (("L1", sel0, ev_node), (tag + "_H1", sel1, ev_node | (S1 > 0)), (tag + "_H2", sel2, ev_node | (S2 > 0))):
    o, allv, reach = evaluate(arm, sel, ev)
    kept[arm] = (allv, reach)
    if arm == "L1":
        assert (allv[m] == np.asarray(D.base_all, bool)[m]).all()
        unreached0 = ~allv & feas & ~reach
    else:
        a0, r0 = kept["L1"]
        for key, cur, base_ in (("BASE_ALL", allv, a0), ("reach_all", reach, r0)):
            g_, l_, p_ = X.mcnemar(base_[m], cur[m])
            o[key + "_vs_L1"] = {"gained": g_, "lost": l_, "p": p_}
        for h in o["per_hop"]:
            s = m & (hops == int(h[3:]))
            g_, l_, p_ = X.mcnemar(a0[s], allv[s])
            o["per_hop"][h]["BASE_vs_L1"] = {"gained": g_, "lost": l_, "p": p_}
            o["per_hop"][h]["initial_UNREACHED"] = int((unreached0 & s).sum())
            o["per_hop"][h]["initial_UNREACHED_now_reached"] = int((unreached0 & s & reach).sum())
            o["per_hop"][h]["initial_UNREACHED_now_ALL_gold_P50"] = int((unreached0 & s & allv).sum())
        if o["per_hop"]:
            s = m & (hops >= 2)
            g_, l_, p_ = X.mcnemar(a0[s], allv[s])
            o["hop2_hop3_pooled_BASE_vs_L1"] = {"n": int(s.sum()), "gained": g_, "lost": l_, "p": p_,
                                                "pooled_BASE_ALL_L1": round(float(a0[s].mean()), 4), "pooled_BASE_ALL": round(float(allv[s].mean()), 4)}
        o["initial_UNREACHED"] = int(unreached0[m].sum())
        o["initial_UNREACHED_now_reached"] = int((unreached0 & reach)[m].sum())
        o["initial_UNREACHED_now_ALL_gold_P50"] = int((unreached0 & allv)[m].sum())
    res["arms"][arm] = o
    G.log("%-13s %s" % (arm, json.dumps({k: o[k] for k in ("BASE_ALL", "ANY", "reach_all", "fail_UNREACHED_pts", "fail_reached_pts", "blocks_voted_per_query_mean", "scope_nodes_P50")})))
    if o["per_hop"]:
        G.log("              per hop %s" % json.dumps({h: (x["BASE_ALL"], x["reach_all"], x["UNREACHED_pts"], x["reached_failed_pts"]) for h, x in o["per_hop"].items()}))
    if arm != "L1":
        G.log("              vs L1: BASE %s reach %s | %s | pooled %s | UNREACHED %d -> reached %d -> ALL@P50 %d" % (
            o["BASE_ALL_vs_L1"], o["reach_all_vs_L1"], {h: x["BASE_vs_L1"] for h, x in o["per_hop"].items()}, o.get("hop2_hop3_pooled_BASE_vs_L1"),
            o["initial_UNREACHED"], o["initial_UNREACHED_now_reached"], o["initial_UNREACHED_now_ALL_gold_P50"]))
G.log("cost %s" % json.dumps(res["cost"]))
G.S.wj(os.path.join(OUT, "microl3_A_%s%s.json" % (name, "__latent" if LATENT else "")), res)
G.log("done %.0fs" % (time.time() - t0))
