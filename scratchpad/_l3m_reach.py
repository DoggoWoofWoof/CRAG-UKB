"""MuSiQue text-L3, step M0b: RE-MEASURE graph reachability on the CANONICAL musique substrate (117,534 nodes) -- the earlier 31% / 55.9% numbers were measured on the legacy 13,672-node musique_clean.

  python -u scratchpad/_l3m_reach.py RUN [--rows=K]

Question (measurement only, no ranking, no traversal policy): from the L1 front end FLAT_RRF (cached by _l3m_hits.py) take the top-s hits as seeds (s = 10 / 50 / 200), expand h = 1..3 hops over a FIXED query-independent graph
(hub rule of the frozen expansion contract: a node whose degree exceeds DEG_CAP = 300 is not expanded; targets are unrestricted), and record, per question,
  * the reach set R_h = seeds U N_<=h(seeds)   and its size |R_h|  (the frontier / candidate-set cost)
  * whether ALL gold nodes lie in R_h, how many golds do, and the first hop at which each gold is reached
  * the paired control "FLAT at the SAME per-question budget": ALL gold within FLAT top-|R_h|  (so a reach set is compared with plain dense+SPLADE going just as deep)
Graphs: G_S = STRUCT_out u STRUCT_in (title mentions, both directions);  G_SN = G_S u NER;  G_SNK = G_SN u KNN  (the universal L1 edge set E, read-only, same key construction as _l1d_node1h.build_families).
Population: the 2,000-row development population (results/L1_DEV/loc_population.json; historically exposed, freely reusable, NOT held-out).  Golds are used only to SCORE reachability (the L1_DEV evaluation convention); nothing here is
a ranking or a selection, so there is nothing to overfit.  Descriptive statistics; no p-value is a verdict.
Output (write-once): results/L3_MUSIQUE/M0b_REACH__musique_v1.json (+ .npz of the per-question arrays)."""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_node1h as H

HITS_NPZ = os.path.join(D.REPO, "data", "_cache", "l3m_hits_musique_v1.npz")
HITS_REC = os.path.join(D.REPO, "results", "L3_MUSIQUE", "M0a_HITS__musique_v1.json")
OUT_JSON = os.path.join(D.REPO, "results", "L3_MUSIQUE", "M0b_REACH__musique_v1.json")
OUT_NPZ = os.path.join(D.REPO, "results", "L3_MUSIQUE", "M0b_REACH__musique_v1.npz")
SEEDS = (10, 50, 200)
HOPS = 3
GRAPHS = {"G_S": ("STRUCT_out", "STRUCT_in"), "G_SN": ("STRUCT_out", "STRUCT_in", "NER"), "G_SNK": ("STRUCT_out", "STRUCT_in", "NER", "KNN")}
CAP = D.DEG_CAP


def union_csr(F, fams, N):
    import scipy.sparse as sp
    M = None
    for f in fams:
        a = sp.csr_matrix((np.ones(len(F[f]["adj"]), np.int8), np.asarray(F[f]["adj"], np.int64), np.asarray(F[f]["xadj"], np.int64)), shape=(N, N))
        M = a if M is None else M + a
    M = M.tocsr()
    M.sum_duplicates()
    return M.indptr.astype(np.int64), M.indices.astype(np.int32)


def expand(xadj, adj, deg, vis, tmp, F):
    """one hop from the frontier F (unique int array): neighbours of the non-hub members of F that are not yet visited."""
    Fs = F[deg[F] <= CAP]
    lens = (xadj[Fs + 1] - xadj[Fs]).astype(np.int64)
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, np.int64), 0
    off = np.cumsum(lens) - lens
    idx = np.arange(tot, dtype=np.int64) - np.repeat(off, lens) + np.repeat(xadj[Fs], lens)
    nb = adj[idx]
    tmp[nb] = True
    tmp &= ~vis
    new = np.flatnonzero(tmp)
    tmp[new] = False
    return new, tot


def main():
    assert sys.argv[1:2] == ["RUN"], __doc__
    rows_limit = None
    for a in sys.argv[2:]:
        if a.startswith("--rows="):
            rows_limit = int(a.split("=", 1)[1])
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:
        pass
    smoke = rows_limit is not None
    out_json, out_npz = (OUT_JSON + ".smoke", OUT_NPZ + ".smoke.npz") if smoke else (OUT_JSON, OUT_NPZ)
    assert not os.path.exists(out_json) and not os.path.exists(out_npz), "write-once: output exists"
    t_all = time.time()
    hrec = json.load(open(HITS_REC))
    assert D.sha_file(HITS_NPZ) == hrec["npz"]["sha256"], "hits cache changed"
    z = np.load(HITS_NPZ)
    cd = D.AD.CanonicalDataset("musique")
    N = int(cd.n_nodes)
    pop = D.Population(cd, rows_limit)
    nq = pop.nq
    assert (z["rows"][:nq] == pop.rows).all() and int(z["gptr"][nq]) == pop.ng_tot
    top = z["top1000"][:nq].astype(np.int64)
    pos_flat = z["pos_flat"][:pop.ng_tot]
    gptr, golds, hops = pop.gptr, pop.golds, pop.hops
    D.log("M0b musique: N %d, %d rows, %d gold nodes, hits sha %s" % (N, nq, pop.ng_tot, hrec["npz"]["sha256"][:12]))
    F, frec = H.build_families(cd, N)
    # ---- FLAT reference: ALL gold within the top-B (B = 100, 1000) of the hit list
    flat_ref = {}
    for B in (100, 1000):
        okq = np.array([bool((pos_flat[gptr[j]:gptr[j + 1]] < B).all()) for j in range(nq)])
        flat_ref["FLAT@%d" % B] = {"ALL": round(float(okq.mean()), 4), "n_all": int(okq.sum())}
    flat1000_all = np.array([bool((pos_flat[gptr[j]:gptr[j + 1]] < 1000).all()) for j in range(nq)])
    res, arrays = {}, {}
    for gname, fams in GRAPHS.items():
        t_g = time.time()
        xadj, adj = union_csr(F, fams, N)
        deg = np.diff(xadj)
        gstats = {"families": list(fams), "directed_entries": int(len(adj)), "mean_row_length": round(float(deg.mean()), 2), "nodes_over_hub_cap": int((deg > CAP).sum()), "isolated_nodes": int((deg == 0).sum()),
                  "gold_nodes_that_are_hubs": round(float((deg[np.concatenate(golds)] > CAP).mean()), 4)}
        D.log("%s: %s" % (gname, json.dumps(gstats)))
        res[gname] = {"graph": gstats}
        vis = np.zeros(N, bool)
        tmp = np.zeros(N, bool)
        for s in SEEDS:
            size = np.zeros((nq, HOPS + 1), np.int64)                 # |R_h| for h = 0..3
            allg = np.zeros((nq, HOPS + 1), bool)
            ngr = np.zeros((nq, HOPS + 1), np.int64)                  # golds inside R_h
            flat_equal = np.zeros((nq, HOPS + 1), bool)               # ALL golds within FLAT top-|R_h|
            first = np.full(pop.ng_tot, 9, np.int8)                   # first hop at which each gold is reached (0 = seed, 9 = not within HOPS)
            edges_read = np.zeros((nq, HOPS), np.int64)
            for j in range(nq):
                g = golds[j]
                sl = slice(gptr[j], gptr[j + 1])
                seeds = np.unique(top[j, :s])
                vis[seeds] = True
                Fr = seeds
                touched = [seeds]
                for h in range(HOPS + 1):
                    if h > 0:
                        Fr, tot = expand(xadj, adj, deg, vis, tmp, Fr)
                        vis[Fr] = True
                        touched.append(Fr)
                        edges_read[j, h - 1] = tot
                    size[j, h] = int(vis.sum()) if h == 0 else size[j, h - 1] + len(Fr)
                    hit = vis[g]
                    ngr[j, h] = int(hit.sum())
                    allg[j, h] = bool(hit.all())
                    f_ = first[sl]
                    f_[(f_ == 9) & hit] = h
                    first[sl] = f_
                    flat_equal[j, h] = bool((pos_flat[sl] < size[j, h]).all())
                for t in touched:
                    vis[t] = False
            hop_strat = {}
            for hname, m in (("2hop", hops == 2), ("3hop", hops == 3), ("4hop", hops == 4), ("headroom (FLAT@1000 misses >=1 gold)", ~flat1000_all)):
                if m.sum() == 0:
                    continue
                hop_strat[hname] = {"n": int(m.sum()), "reach_ALL by h=0..3": [round(float(allg[m, h].mean()), 4) for h in range(HOPS + 1)], "FLAT_ALL at equal per-query budget": [round(float(flat_equal[m, h].mean()), 4) for h in range(HOPS + 1)]}
            fr = first[first != 0]
            res[gname]["seeds_%d" % s] = {
                "reach_ALL by h=0..3": [round(float(allg[:, h].mean()), 4) for h in range(HOPS + 1)],
                "FLAT_ALL at the same per-query budget": [round(float(flat_equal[:, h].mean()), 4) for h in range(HOPS + 1)],
                "paired h=1..3 (reach_ALL and not FLAT_ALL / FLAT_ALL and not reach_ALL)": [[int((allg[:, h] & ~flat_equal[:, h]).sum()), int((~allg[:, h] & flat_equal[:, h]).sum())] for h in range(1, HOPS + 1)],
                "gold_fraction_reached by h=0..3": [round(float(ngr[:, h].sum() / pop.ng_tot), 4) for h in range(HOPS + 1)],
                "reach_size |R_h| mean": [round(float(size[:, h].mean()), 1) for h in range(HOPS + 1)],
                "reach_size |R_h| median": [int(np.median(size[:, h])) for h in range(HOPS + 1)],
                "reach_size |R_h| p95": [int(np.percentile(size[:, h], 95)) for h in range(HOPS + 1)],
                "edges_read_per_query mean (hop 1..3)": [round(float(edges_read[:, h].mean()), 0) for h in range(HOPS)],
                "first_hop_of_golds_not_in_seeds (1,2,3,unreached)": [int((fr == v).sum()) for v in (1, 2, 3, 9)],
                "strata": hop_strat}
            arrays["%s_s%d_size" % (gname, s)] = size.astype(np.int32)
            arrays["%s_s%d_all" % (gname, s)] = allg
            arrays["%s_s%d_flat_equal" % (gname, s)] = flat_equal
            arrays["%s_s%d_first" % (gname, s)] = first
            D.log("  %s seeds %d: reach_ALL %s | FLAT_ALL@equal %s | |R| median %s" % (gname, s, res[gname]["seeds_%d" % s]["reach_ALL by h=0..3"], res[gname]["seeds_%d" % s]["FLAT_ALL at the same per-query budget"],
                                                                                        res[gname]["seeds_%d" % s]["reach_size |R_h| median"]))
        res[gname]["seconds"] = round(time.time() - t_g, 1)
    rec = {"stage": "MuSiQue text-L3 M0b: reachability re-measure on the canonical substrate (development population)", "N": N, "n_rows": nq, "n_gold_nodes": int(pop.ng_tot), "hub_cap": CAP, "seeds": list(SEEDS), "hops": HOPS,
           "population": pop.record, "front_end": {"hits_record": D.rel(HITS_REC), "hits_npz_sha256": hrec["npz"]["sha256"]}, "FLAT_reference": flat_ref, "graphs": res, "families_record": frec,
           "population_hops_histogram": {str(int(h)): int((hops == h).sum()) for h in np.unique(hops)},
           "code": {"path": "scratchpad/_l3m_reach.py", "sha256": D.sha_file(os.path.abspath(__file__))}, "pinned": D.PINNED, "platform": D.platform_record(), "seconds": round(time.time() - t_all, 1),
           "process_peak_rss_mb": D.peak_rss_mb(),
           "read_as": "measurement of what an UNRANKED h-hop reach set contains and costs; ranking inside the reach set (PPR / best-first / typed) is the next step and is not measured here"}
    np.savez_compressed(out_npz, rows=pop.rows, **arrays)
    rec["npz"] = {"path": D.rel(out_npz), "sha256": D.sha_file(out_npz)}
    json.dump(rec, open(out_json, "w"), indent=1)
    D.log("done %s (%.0f s)" % (D.rel(out_json), time.time() - t_all))


if __name__ == "__main__":
    main()
