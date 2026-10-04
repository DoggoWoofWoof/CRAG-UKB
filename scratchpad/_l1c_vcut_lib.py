"""STRUCT_VERTEXCUT_V1 helpers (ninth ruling, 2026-09-15; PREREGISTRATION_STRUCT_VERTEXCUT_V1.json): the STRUCT graph, block
memberships as sparse node x block 0/1 matrices, the query-free containment metrics, and the deterministic capacity repair.
Everything here is label-free: no query, no seed, no gold, no embedding is read.  Served 'hard' vectors are read from the replay
caches ONLY as baseline partitions (their other arrays are never touched).
"""
import json
import os
import time

import numpy as np
import scipy.sparse as sp

import _l1s_core as S
from src.l1_canonical import hypergraph as HG  # noqa: E402  (imported, never edited)
from src.l1_canonical.adapter import CanonicalDataset, sha_file  # noqa: E402

X = S.X
OUT = os.path.join(X.REPO, "results", "L1_COVPART")
PDIR = os.path.join(OUT, "parts")
TAG = "STRUCT_VCUT_V1"
PATH3_SAMPLE = 200000
PATH3_EXACT_MAX = 300000


def log(*a):
    S.log(*a)


class Graph(object):
    """undirected STRUCT graph from the frozen adapter keys (+ the directed CSR for the served voting membership)."""

    def __init__(self, ds):
        t = time.time()
        self.ds = ds
        cd = CanonicalDataset(ds)
        N, ST, KN, NX = cd.keysets()
        ST = np.asarray(ST, np.int64)
        self.N, self.E = int(N), int(len(ST))
        self.eu, self.ev = (ST // N).astype(np.int64), (ST % N).astype(np.int64)
        self.k_f = HG.frozen_k(N)
        self.C = int(round(N / self.k_f))
        self.deg = np.bincount(np.concatenate([self.eu, self.ev]), minlength=N).astype(np.int64)
        self.hub = (self.deg + 1) > self.C
        A = sp.coo_matrix((np.ones(2 * self.E, np.int8), (np.concatenate([self.eu, self.ev]), np.concatenate([self.ev, self.eu]))), shape=(N, N)).tocsr()
        A.sum_duplicates()
        self.xu, self.au = np.asarray(A.indptr, np.int64), np.asarray(A.indices, np.int64)
        assert np.array_equal(np.diff(self.xu), self.deg)
        xo, ao = cd.struct_csr(directed=True)
        self.xo, self.ao = np.asarray(xo, np.int64), np.asarray(ao, np.int64)
        cd._csr.clear()
        inc = sp.coo_matrix((np.ones(2 * self.E, np.int8), (np.concatenate([self.eu, self.ev]), np.concatenate([np.arange(self.E), np.arange(self.E)]))), shape=(N, self.E)).tocsr()
        self.inc_ptr, self.inc_e = np.asarray(inc.indptr, np.int64), np.asarray(inc.indices, np.int64)   # node -> incident edge ids
        self.keys_sha = __import__("hashlib").sha256(np.ascontiguousarray(ST).tobytes()).hexdigest()
        log("%s graph: N %d |E| %d k_f %d C %d hubs %d deg0 %d deg1 %d (%.1fs)" % (ds, N, self.E, self.k_f, self.C, int(self.hub.sum()), int((self.deg == 0).sum()), int((self.deg == 1).sum()), time.time() - t))

    def nbrs(self, v):
        return self.au[self.xu[v]:self.xu[v + 1]]

    def inc(self, v):
        return self.inc_e[self.inc_ptr[v]:self.inc_ptr[v + 1]]


# ------------------------------------------------------------------------------------------------ memberships (node x block)
def Y_from_pairs(N, k, rows, cols):
    Y = sp.coo_matrix((np.ones(len(rows), np.int8), (np.asarray(rows, np.int64), np.asarray(cols, np.int64))), shape=(N, k)).tocsr()
    Y.sum_duplicates()
    Y.data[:] = 1
    Y.sort_indices()
    return Y


def Y_hard(G, hard):
    k = int(hard.max()) + 1
    return Y_from_pairs(G.N, k, np.arange(G.N), hard)


def Y_legacy_membership(G, hard):
    """the served voting unit: own block + blocks of the DIRECTED STRUCT out-neighbours (_l1s_core.Data.legacy_mem)."""
    k = int(hard.max()) + 1
    r = np.concatenate([np.arange(G.N, dtype=np.int64), np.repeat(np.arange(G.N, dtype=np.int64), np.diff(G.xo))])
    p = np.concatenate([hard, hard[G.ao]])
    return Y_from_pairs(G.N, k, r, p)


def Y_undirected_halo(G, hard):
    """B_p + N(B_p): own block + blocks of all undirected STRUCT neighbours."""
    k = int(hard.max()) + 1
    r = np.concatenate([np.arange(G.N, dtype=np.int64), np.repeat(np.arange(G.N, dtype=np.int64), G.deg)])
    p = np.concatenate([hard, hard[G.au]])
    return Y_from_pairs(G.N, k, r, p)


def Y_vcut(G, z, k):
    """endpoint sets of an edge partition z: E -> {0..k-1}."""
    return Y_from_pairs(G.N, k, np.concatenate([G.eu, G.ev]), np.concatenate([z, z]))


# ------------------------------------------------------------------------------------------------ metrics
def _pair_union_count(Y, rows, cols):
    """(ordered pairs (i, j) of 'rows', i == j included, that share >= 1 column of 'cols' in Y;  rows with >= 1 such column)."""
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0
    W = Y[rows][:, cols]
    if W.nnz == 0:
        return 0, 0
    Gm = W @ W.T
    Gm.eliminate_zeros()
    return int(Gm.nnz), int((W.getnnz(axis=1) > 0).sum())


def wedge_containment(G, Y, log_every=0):
    """exact 2-hop wedge containment: for every middle m (deg >= 2), pairs {u, v} of distinct neighbours with a block holding u, m, v."""
    t = time.time()
    tot_all = con_all = tot_nh = con_nh = 0
    frac_nh = []
    Yc = Y.tocsr()
    for m in range(G.N):
        d = int(G.deg[m])
        if d < 2:
            continue
        pairs = d * (d - 1) // 2
        Pm = Yc.indices[Yc.indptr[m]:Yc.indptr[m + 1]]
        if len(Pm) == 0:
            c = 0
        else:
            nn, diag = _pair_union_count(Yc, G.nbrs(m), Pm)
            c = (nn - diag) // 2
        tot_all += pairs
        con_all += c
        if not G.hub[m]:
            tot_nh += pairs
            con_nh += c
            frac_nh.append(c / float(pairs))
        if log_every and m % log_every == 0:
            log("    wedges %d / %d (%.0fs)" % (m, G.N, time.time() - t))
    return {"wedges_all": int(tot_all), "contained_all": int(con_all), "containment_all": round(con_all / float(max(tot_all, 1)), 4),
            "wedges_nonhub_middle": int(tot_nh), "contained_nonhub_middle": int(con_nh), "containment_nonhub_middle": round(con_nh / float(max(tot_nh, 1)), 4),
            "containment_nonhub_middle_per_node_mean": round(float(np.mean(frac_nh)), 4) if frac_nh else None,
            "nonhub_middles_fully_contained_fraction": round(float(np.mean([f >= 1.0 for f in frac_nh])), 4) if frac_nh else None,
            "seconds": round(time.time() - t, 1)}


def path3_containment(G, Y, sample_stride):
    """exact 3-path containment per middle edge (m1, m2): simple paths u-m1-m2-v with a block holding all four."""
    t = time.time()
    Yc = Y.tocsr()
    idx = np.arange(0, G.E, sample_stride, dtype=np.int64)
    tot_all = con_all = tot_nh = con_nh = 0
    n_mid = 0
    for e in idx:
        m1, m2 = int(G.eu[e]), int(G.ev[e])
        n1 = G.nbrs(m1)
        n2 = G.nbrs(m2)
        n1 = n1[n1 != m2]
        n2 = n2[n2 != m1]
        if len(n1) == 0 or len(n2) == 0:
            continue
        common = np.intersect1d(n1, n2, assume_unique=True)
        total = int(len(n1)) * int(len(n2)) - int(len(common))
        if total <= 0:
            continue
        n_mid += 1
        P1 = Yc.indices[Yc.indptr[m1]:Yc.indptr[m1 + 1]]
        P2 = Yc.indices[Yc.indptr[m2]:Yc.indptr[m2 + 1]]
        P12 = np.intersect1d(P1, P2, assume_unique=True)
        if len(P12) == 0:
            c = 0
        else:
            W1 = Yc[n1][:, P12]
            W2 = Yc[n2][:, P12]
            Gm = W1 @ W2.T
            Gm.eliminate_zeros()
            c = int(Gm.nnz)
            if len(common):
                c -= int((Yc[common][:, P12].getnnz(axis=1) > 0).sum())     # u == v (triangle) entries are not paths
        tot_all += total
        con_all += c
        if not (G.hub[m1] or G.hub[m2]):
            tot_nh += total
            con_nh += c
    return {"middle_edges_evaluated": int(n_mid), "sample_stride": int(sample_stride), "exact": bool(sample_stride == 1),
            "paths_all": int(tot_all), "contained_all": int(con_all), "containment_all": round(con_all / float(max(tot_all, 1)), 4),
            "paths_nonhub_middles": int(tot_nh), "contained_nonhub_middles": int(con_nh), "containment_nonhub_middles": round(con_nh / float(max(tot_nh, 1)), 4),
            "seconds": round(time.time() - t, 1)}


def ball2_containment(G, Y):
    """section 2 metric: closed 2-hop non-hub STRUCT ball of every non-hub node (<= C nodes) inside ONE block; best single-block coverage."""
    t = time.time()
    Yc = Y.tocsr()
    nonhub = ~G.hub
    intact, cover, n = 0, [], 0
    for u in np.nonzero(nonhub)[0]:
        n1 = G.nbrs(u)
        n1 = n1[nonhub[n1]]
        if len(n1):
            cnt = G.xu[n1 + 1] - G.xu[n1]
            idx = np.repeat(G.xu[n1], cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
            n2 = G.au[idx]
            ball = np.unique(np.concatenate([[u], n1, n2[nonhub[n2]]]))
        else:
            ball = np.unique(np.concatenate([[u], n1]))
        if len(ball) > G.C:
            continue
        n += 1
        cs = np.asarray(Yc[ball].sum(axis=0)).ravel()
        best = int(cs.max()) if cs.size else 0
        intact += int(best == len(ball))
        cover.append(best / float(len(ball)))
    return {"balls_le_C": int(n), "intact_fraction": round(intact / float(max(n, 1)), 4), "best_block_coverage_mean": round(float(np.mean(cover)), 4) if cover else None,
            "seconds": round(time.time() - t, 1)}


def structure_metrics(G, Y, path3_stride, label, wedges=True, path3=True, ball=True):
    t = time.time()
    Y = Y.tocsr()
    k = Y.shape[1]
    sizes = np.asarray(Y.sum(axis=0)).ravel().astype(np.int64)
    lam = np.asarray(Y.sum(axis=1)).ravel().astype(np.int64)
    used = sizes > 0
    e_con = (Y[G.eu].multiply(Y[G.ev]).getnnz(axis=1) > 0)
    out = {"label": label, "k": int(k), "blocks_used": int(used.sum()),
           "RF": round(float(Y.nnz) / G.N, 4), "memberships_total": int(Y.nnz), "replication_beyond_one": int(np.maximum(lam - 1, 0).sum()),
           "nodes_in_no_block": int((lam == 0).sum()), "lambda_max": int(lam.max()),
           "lambda_mean_hub": round(float(lam[G.hub].mean()), 3) if G.hub.any() else None, "lambda_mean_nonhub": round(float(lam[~G.hub].mean()), 4),
           "lambda_mean_nonhub_deg_ge2": round(float(lam[(~G.hub) & (G.deg >= 2)].mean()), 4),
           "nonhub_deg_ge2_with_lambda_1": round(float((lam[(~G.hub) & (G.deg >= 2)] == 1).mean()), 4),
           "block_size": {"min": int(sizes[used].min()), "mean": round(float(sizes[used].mean()), 2), "p50": int(np.percentile(sizes[used], 50)),
                          "p90": int(np.percentile(sizes[used], 90)), "max": int(sizes[used].max()),
                          "blocks_gt_C": int((sizes > G.C).sum()), "blocks_gt_1.5C": int((sizes > 1.5 * G.C).sum()), "C": G.C},
           "edge_containment_1hop": round(float(e_con.mean()), 4), "edges_cut": int((~e_con).sum())}
    if wedges:
        out["wedge_2hop"] = wedge_containment(G, Y)
    if path3:
        out["path_3hop"] = path3_containment(G, Y, path3_stride)
    if ball:
        out["ball2_section2"] = ball2_containment(G, Y)
    out["seconds"] = round(time.time() - t, 1)
    log("  %s: k %d RF %.3f |B| mean %.1f max %d (>C %d) 1hop %.4f%s%s%s (%.0fs)" % (
        label, k, out["RF"], out["block_size"]["mean"], out["block_size"]["max"], out["block_size"]["blocks_gt_C"], out["edge_containment_1hop"],
        " wedge nh %.4f all %.4f" % (out["wedge_2hop"]["containment_nonhub_middle"], out["wedge_2hop"]["containment_all"]) if wedges else "",
        " path3 nh %.4f all %.4f" % (out["path_3hop"]["containment_nonhub_middles"], out["path_3hop"]["containment_all"]) if path3 else "",
        " ball2 %.4f" % out["ball2_section2"]["intact_fraction"] if ball else "", out["seconds"]))
    return out


# ------------------------------------------------------------------------------------------------ vertex-cut partitions on disk
def load_vcut(ds, k):
    npz = os.path.join(PDIR, "%s__%s_k%d.npz" % (ds, TAG, k))
    meta = json.load(open(npz[:-4] + ".json", encoding="utf-8"))
    assert sha_file(npz) == meta["file_sha256"]
    z = np.load(os.path.join(PDIR, "%s__%s_k%d__PHG_con.npy" % (ds, TAG, k))).astype(np.int64)
    run = json.load(open(os.path.join(PDIR, "%s__%s_k%d__PHG_con.RUN.json" % (ds, TAG, k)), encoding="utf-8"))
    assert run["output"]["STATUS"] == "OK" and len(z) == meta["N"] and z.min() >= 0 and z.max() < k
    return z, meta, run


def vcut_size_stats(G, z, k):
    Y = Y_vcut(G, z, k)
    sizes = np.asarray(Y.sum(axis=0)).ravel()
    return {"RF": round(float(Y.nnz) / G.N, 4), "mean_block_size": round(float(Y.nnz) / k, 2), "max_block_size": int(sizes.max()),
            "blocks_gt_C": int((sizes > G.C).sum()), "blocks_used": int((sizes > 0).sum())}


# ------------------------------------------------------------------------------------------------ deterministic capacity repair
def capacity_repair(G, z_in, k, C, log_every=500):
    """pre-registered rule: evict nodes from over-full blocks by moving their edges; see PREREGISTRATION_STRUCT_VERTEXCUT_V1.json."""
    t = time.time()
    z = z_in.astype(np.int64).copy()
    N, E = G.N, G.E
    part_nodes = [dict() for _ in range(k)]            # part -> {node: number of its edges in the part}
    node_parts = [dict() for _ in range(N)]            # node -> {part: number of its edges there}
    for e in range(E):
        for v in (int(G.eu[e]), int(G.ev[e])):
            p = int(z[e])
            part_nodes[p][v] = part_nodes[p].get(v, 0) + 1
            node_parts[v][p] = node_parts[v].get(p, 0) + 1
    sizes = np.array([len(d) for d in part_nodes], np.int64)
    rep0 = int(sum(len(d) for d in node_parts))
    overflow0 = int(np.maximum(sizes - C, 0).sum())
    over0 = int((sizes > C).sum())
    stuck = np.zeros(k, bool)
    evictions = moves = 0
    cls_count = [0, 0, 0, 0]
    skipped_nodes = 0
    ids = np.arange(k, dtype=np.int64)

    def add(v, p):
        if p in node_parts[v]:
            node_parts[v][p] += 1
            part_nodes[p][v] += 1
            return 0
        node_parts[v][p] = 1
        part_nodes[p][v] = 1
        sizes[p] += 1
        return 1

    def remove(v, p):
        node_parts[v][p] -= 1
        part_nodes[p][v] -= 1
        if node_parts[v][p] == 0:
            del node_parts[v][p]
            del part_nodes[p][v]
            sizes[p] -= 1
            return 1
        return 0

    while True:
        cand_blocks = np.where((sizes > C) & (~stuck))[0]
        if len(cand_blocks) == 0:
            break
        p = int(cand_blocks[np.lexsort((cand_blocks, -sizes[cand_blocks]))[0]])
        nodes = sorted(part_nodes[p].items(), key=lambda kv: (kv[1], -int(G.deg[kv[0]]), kv[0]))
        done = False
        for v, cnt in nodes:
            E_vp = sorted(int(e) for e in G.inc(v) if z[e] == p)
            assert len(E_vp) == cnt
            tent = {}                       # part -> tentative size increase
            tentV = set()
            tentW = {}
            plan = []
            ok = True
            for e in E_vp:
                w = int(G.ev[e]) if int(G.eu[e]) == v else int(G.eu[e])
                Pv = set(node_parts[v]) | tentV
                Pw = set(node_parts[w]) | tentW.get(w, set())
                Pv.discard(p)
                Pw.discard(p)
                eff = lambda q: int(sizes[q]) + tent.get(q, 0)
                best = None
                c0 = [q for q in Pv & Pw]
                if c0:
                    best = (0, min(c0, key=lambda q: (eff(q), q)))
                if best is None:
                    c1 = [q for q in Pw - Pv if eff(q) + 1 <= C]
                    if c1:
                        best = (1, min(c1, key=lambda q: (eff(q), q)))
                if best is None:
                    c2 = [q for q in Pv - Pw if eff(q) + 1 <= C]
                    if c2:
                        best = (2, min(c2, key=lambda q: (eff(q), q)))
                if best is None:
                    effa = sizes.copy()
                    for q, dq in tent.items():
                        effa[q] += dq
                    mask = effa + 2 <= C
                    mask[p] = False
                    for q in Pv | Pw:
                        mask[q] = False
                    if mask.any():
                        cand = ids[mask]
                        best = (3, int(cand[np.lexsort((cand, effa[cand]))[0]]))
                if best is None:
                    ok = False
                    break
                cl, tq = best
                plan.append((e, tq, cl, w))
                if cl in (1, 3):
                    tentV.add(tq)
                    tent[tq] = tent.get(tq, 0) + 1
                if cl in (2, 3):
                    tentW.setdefault(w, set()).add(tq)
                    tent[tq] = tent.get(tq, 0) + 1
            if not ok:
                skipped_nodes += 1
                continue
            for e, tq, cl, w in plan:
                z[e] = tq
                remove(v, p)
                remove(w, p)
                add(v, tq)
                add(w, tq)
                cls_count[cl] += 1
                moves += 1
            assert v not in part_nodes[p]
            evictions += 1
            done = True
            break
        if not done:
            stuck[p] = True
        if log_every and evictions % log_every == 0 and done:
            log("    repair: %d evictions, %d moves, over-full blocks %d, overflow %d (%.0fs)" % (evictions, moves, int((sizes > C).sum()), int(np.maximum(sizes - C, 0).sum()), time.time() - t))
    rep1 = int(sum(len(d) for d in node_parts))
    chk = np.asarray(Y_vcut(G, z, k).sum(axis=0)).ravel()
    assert np.array_equal(chk, sizes)
    rec = {"rule": "deterministic capacity repair (pre-registered)", "C": C, "evictions": evictions, "edge_moves": moves, "moves_by_class": cls_count,
           "candidate_nodes_skipped": skipped_nodes, "over_full_blocks_before": over0,
           "overflow_nodes_before": overflow0, "over_full_blocks_after_residual": int((sizes > C).sum()), "overflow_nodes_after_residual": int(np.maximum(sizes - C, 0).sum()),
           "stuck_blocks": int(stuck.sum()), "memberships_before": rep0, "memberships_after": rep1, "RF_before": round(rep0 / float(N), 4), "RF_after": round(rep1 / float(N), 4),
           "edges_reassigned_fraction": round(float((z != z_in).mean()), 4), "seconds": round(time.time() - t, 1)}
    return z, rec
