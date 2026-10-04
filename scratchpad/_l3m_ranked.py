"""MuSiQue text-L3, step M0c: does a RANKED graph signal beat plain FLAT / the L1 one-hop localisation (IR_L1) at the same served-list size B_N?  (development population; descriptive; no held-out row is read.)

  python -u scratchpad/_l3m_ranked.py RUN [--rows=K] [--graphs=G_SN,G_SNK] [--seeds=10,50,200]

Arms (fixed a priori, no tuning, no learned weight, no LLM, no encoder training):
  FLAT                     the L1 front end FLAT_RRF (dense + SPLADE, K0 = 60); cached ranks (_l3m_hits_v2.py)
  IR_L1                    the stored FLAT+LOC (one-hop localisation, E = STRUCT u KNN u NER) positions of results/L1_DEV/route3_musique__v1.npz (reproduces the L1_DEV scoreboard: .639/.742/.793/.840/.880/.927)
  PPR_<graph>_s<s>_<norm>  personalised PageRank over a FIXED query-independent graph, restart alpha = 0.5, 8 power iterations; restart mass on the top-s FLAT hits weighted 1/(60 + rank);
                           the hub rule of the frozen expansion contract (a node of degree > 300 does not propagate: its mass stays); served list = RRF (K0 = 60) of the FLAT rank and the PPR rank over the union
                           U = FLAT top-5000 u PPR top-5000, ordered by the fused value (ties: node id).  norm raw: PPR score;  dn: PPR score / sqrt(1 + degree) (the usual hub damping)
Graphs: G_SN = STRUCT_out u STRUCT_in u NER;  G_SNK = G_SN u KNN   (G_S is the pure title-mention graph; NER / KNN are the inference-safe shares-entity / semantic-kNN edges, un-weighted here).
Metric: ALL gold nodes inside the served top-B_N (the L1_DEV convention), B_N in {100, 250, 500, 1000, 2000, 5000}; by question hop count (2 / 3 / 4) and on the headroom set (FLAT@1000 misses a gold); paired counts vs FLAT and vs IR_L1.
Cost note: dense power iteration reads the whole graph per query -- it measures the ACCURACY of a ranked graph signal; a bounded-frontier implementation (best-first) is only worth building if this ranking earns it.
Output (write-once): results/L3_MUSIQUE/M0c_RANKED__musique_v1.json"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import _l1d_lib as D
import _l1d_node1h as H
import _l3m_reach as RC

HITS_NPZ = os.path.join(D.REPO, "data", "_cache", "l3m_hits_musique_v2.npz")
HITS_REC = os.path.join(D.REPO, "results", "L3_MUSIQUE", "M0a_HITS__musique_v2.json")
REF_ROUTE = os.path.join(D.OUT, "route3_musique__v1.npz")
OUT_JSON = os.path.join(D.REPO, "results", "L3_MUSIQUE", "M0c_RANKED__musique_v1.json")
ALPHA, ITERS, K0, CAP = 0.5, 8, 60, D.DEG_CAP
MS = (100, 250, 500, 1000, 2000, 5000)
BIG = 10 ** 6
CQ = 100


def transition(xadj, adj, N):
    deg = np.diff(xadj)
    hub = deg > CAP
    rows = np.repeat(np.arange(N), deg)
    data = np.where(hub[rows], 0.0, 1.0 / np.maximum(deg, 1)[rows]).astype(np.float32)
    P = sp.csr_matrix((data, adj, xadj), shape=(N, N)) + sp.diags((hub | (deg == 0)).astype(np.float32))
    return P.T.tocsr().astype(np.float32), deg


def ppr(T, X0):
    X = X0.copy()
    for _ in range(ITERS):
        X = (1.0 - ALPHA) * (T @ X) + ALPHA * X0
    return X


def served_positions(p, top_j, g, N, rf_dense):
    """positions (BIG = not served) of the gold nodes g in the fused FLAT + PPR order over U"""
    nz = int((p > 0).sum())
    ordp = np.argsort(-p, kind="stable")
    rp = np.empty(N, np.int64)
    rp[ordp] = np.arange(N)
    rp = np.minimum(rp, nz)
    U = np.union1d(top_j, ordp[:min(5000, nz)])
    v = 1.0 / (K0 + rf_dense[U]) + 1.0 / (K0 + rp[U])
    ordU = np.lexsort((U, -v))
    inv = np.empty(len(U), np.int64)
    inv[ordU] = np.arange(len(U))
    gi = np.searchsorted(U, g)
    ok = (gi < len(U)) & (U[np.minimum(gi, len(U) - 1)] == g)
    pos = np.full(len(g), BIG, np.int64)
    pos[ok] = inv[gi[ok]]
    return pos


def all_at(pos, gptr, M):
    return np.array([bool((pos[gptr[j]:gptr[j + 1]] < M).all()) for j in range(len(gptr) - 1)])


def mcnemar(a, b):
    from scipy.stats import binomtest
    x, y = int((a & ~b).sum()), int((~a & b).sum())
    return [x, y, round(float(binomtest(x, x + y, 0.5).pvalue), 6) if x + y else 1.0]


def main():
    assert sys.argv[1:2] == ["RUN"], __doc__
    rows_limit, graphs, seeds = None, ["G_SN", "G_SNK"], [10, 50, 200]
    for a in sys.argv[2:]:
        if a.startswith("--rows="):
            rows_limit = int(a.split("=", 1)[1])
        elif a.startswith("--graphs="):
            graphs = a.split("=", 1)[1].split(",")
        elif a.startswith("--seeds="):
            seeds = [int(x) for x in a.split("=", 1)[1].split(",")]
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:
        pass
    out_json = OUT_JSON + (".smoke" if rows_limit else "")
    assert not os.path.exists(out_json), "write-once: output exists"
    t_all = time.time()
    hrec = json.load(open(HITS_REC))
    assert D.sha_file(HITS_NPZ) == hrec["npz"]["sha256"] and hrec["topk"] == 5000, "hits cache changed"
    z = np.load(HITS_NPZ)
    cd = D.AD.CanonicalDataset("musique")
    N = int(cd.n_nodes)
    pop = D.Population(cd, rows_limit)
    nq, gptr, golds, hops = pop.nq, pop.gptr, pop.golds, pop.hops
    assert (z["rows"][:nq] == pop.rows).all() and int(z["gptr"][nq]) == pop.ng_tot
    top = z["top1000"][:nq].astype(np.int64)                      # width 5000 in v2
    assert top.shape[1] == 5000
    pos_flat = z["pos_flat"][:pop.ng_tot]
    zr = np.load(REF_ROUTE)
    assert (zr["rows"][:nq] == pop.rows).all() and (zr["pos_FLAT"][:pop.ng_tot] == pos_flat).all(), "FLAT positions differ from the route3 record"
    pos_ir = zr["pos_FLATLOC__IR_L1"][:pop.ng_tot]
    D.log("M0c musique: N %d, %d rows, %d gold nodes; graphs %s seeds %s" % (N, nq, pop.ng_tot, graphs, seeds))
    F, frec = H.build_families(cd, N)
    famsets = {"G_SN": ("STRUCT_out", "STRUCT_in", "NER"), "G_SNK": ("STRUCT_out", "STRUCT_in", "NER", "KNN"), "G_S": ("STRUCT_out", "STRUCT_in")}
    POS = {"FLAT": pos_flat, "IR_L1": pos_ir}
    rf_dense = np.full(N, 5000, np.int64)
    ar = np.arange(5000)
    t_ppr = {}
    for gname in graphs:
        xadj, adj = RC.union_csr(F, famsets[gname], N)
        T, deg = transition(xadj, adj, N)
        dn = (1.0 / np.sqrt(1.0 + deg)).astype(np.float32)
        D.log("%s: nnz %d, hubs %d" % (gname, len(adj), int((deg > CAP).sum())))
        for s in seeds:
            for nm in ("raw", "dn"):
                POS["PPR_%s_s%d_%s" % (gname, s, nm)] = np.zeros(pop.ng_tot, np.int64)
            t0 = time.time()
            for j0 in range(0, nq, CQ):
                j1 = min(nq, j0 + CQ)
                X0 = np.zeros((N, j1 - j0), np.float32)
                w = (1.0 / (K0 + np.arange(s))).astype(np.float32)
                w /= w.sum()
                for i, j in enumerate(range(j0, j1)):
                    X0[top[j, :s], i] = w
                X = ppr(T, X0)
                del X0
                for i, j in enumerate(range(j0, j1)):
                    rf_dense[top[j]] = ar
                    sl = slice(gptr[j], gptr[j + 1])
                    p = X[:, i]
                    POS["PPR_%s_s%d_raw" % (gname, s)][sl] = served_positions(p, top[j], golds[j], N, rf_dense)
                    POS["PPR_%s_s%d_dn" % (gname, s)][sl] = served_positions(p * dn, top[j], golds[j], N, rf_dense)
                    rf_dense[top[j]] = 5000
                D.log("  %s s%d rows %d / %d (%.0fs, RSS %.0f MB)" % (gname, s, j1, nq, time.time() - t0, D._rss_mb()))
                del X
            t_ppr["%s_s%d" % (gname, s)] = round(time.time() - t0, 1)
    flat1000 = all_at(pos_flat, gptr, 1000)
    strata = {"all": np.ones(nq, bool), "2hop": hops == 2, "3hop": hops == 3, "4hop": hops == 4, "headroom (FLAT@1000 misses a gold)": ~flat1000}
    arms = {}
    base = {}
    for name, pos in POS.items():
        A = {M: all_at(pos, gptr, M) for M in MS}
        base[name] = A
        arms[name] = {"ALL by M=%s" % list(MS): [round(float(A[M].mean()), 4) for M in MS],
                      "gold_fraction by M": [round(float(np.mean([(pos[gptr[j]:gptr[j + 1]] < M).mean() for j in range(nq)])), 4) for M in MS],
                      "strata ALL": {sn: [round(float(A[M][m].mean()), 4) for M in MS] for sn, m in strata.items() if m.sum()}}
    for name in POS:
        if name in ("FLAT", "IR_L1"):
            continue
        arms[name]["paired vs FLAT (gained, lost, p) by M"] = [mcnemar(base[name][M], base["FLAT"][M]) for M in MS]
        arms[name]["paired vs IR_L1 (gained, lost, p) by M"] = [mcnemar(base[name][M], base["IR_L1"][M]) for M in MS]
    rec = {"stage": "MuSiQue text-L3 M0c: ranked graph signal (PPR) vs FLAT / IR_L1 at equal served-list size (development population)", "N": N, "n_rows": nq, "n_gold_nodes": int(pop.ng_tot),
           "constants": {"ALPHA": ALPHA, "ITERS": ITERS, "K0": K0, "hub_cap": CAP, "budgets": list(MS), "union": "FLAT top-5000 u PPR top-5000"}, "population": pop.record,
           "front_end": {"hits_record": D.rel(HITS_REC), "hits_npz_sha256": hrec["npz"]["sha256"], "IR_L1_reference": D.rel(REF_ROUTE), "IR_L1_reference_sha256": D.sha_file(REF_ROUTE)},
           "arms": arms, "ppr_seconds": t_ppr, "code": {"path": "scratchpad/_l3m_ranked.py", "sha256": D.sha_file(os.path.abspath(__file__))}, "pinned": D.PINNED, "platform": D.platform_record(),
           "seconds": round(time.time() - t_all, 1), "process_peak_rss_mb": D.peak_rss_mb(),
           "read_as": "accuracy of a ranked graph signal on the development population; one fixed alpha / iteration count chosen a priori; not a held-out result; PPR cost is the dense power iteration, not a bounded-frontier L3"}
    json.dump(rec, open(out_json, "w"), indent=1)
    for name in POS:
        D.log("%-26s ALL %s" % (name, arms[name]["ALL by M=%s" % list(MS)]))
    D.log("done %s (%.0f s)" % (D.rel(out_json), time.time() - t_all))


if __name__ == "__main__":
    main()
