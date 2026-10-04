"""MuSiQue text-L3, step M0a: cache the L1 FRONT END of the 2,000-row development population once, so every later L3 measurement is a cheap array computation.

  python -u scratchpad/_l3m_hits.py RUN [--rows=K]

Per row (the pinned L1 functions of _l1d_lib, exactly the arithmetic of the L1_DEV harnesses): the FLAT_RRF order (dense + SPLADE, K0 = 60) of the whole canonical musique corpus (N = 117,534), top-1000 kept, the dense and SPLADE
top-200 (the activation lists), npos, and the FLAT rank of every gold node.  REGRESSION: the FLAT gold positions must equal results/L1_DEV/loc_musique__v1.npz pos_FLAT bit for bit (the stored L1_DEV record on the same rows).
No gold is read by any ranking: golds are used only to record their FLAT ranks (the L1_DEV evaluation convention), the cached hit lists depend on the question text alone.
Population: results/L1_DEV/loc_population.json (sha c7f70806..., 417 dev + 1,583 train rows; development population, historically exposed, freely reusable; NOT held-out).  No held-out row is read.
Output (write-once): data/_cache/l3m_hits_musique_v1.npz + results/L3_MUSIQUE/M0a_HITS__musique_v1.json.  Runs at idle process priority (the laptop has ~0.8 GB free)."""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D

TOPK = 1000
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = D.REPO
OUT_NPZ = os.path.join(REPO, "data", "_cache", "l3m_hits_musique_v1.npz")
OUT_JSON = os.path.join(REPO, "results", "L3_MUSIQUE", "M0a_HITS__musique_v1.json")
REF = os.path.join(D.OUT, "loc_musique__v1.npz")


def main():
    assert sys.argv[1:2] == ["RUN"], __doc__
    rows_limit = None
    for a in sys.argv[2:]:
        if a.startswith("--rows="):
            rows_limit = int(a.split("=", 1)[1])
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception as e:                                                  # priority is a courtesy to the desktop, not a result
        print("could not lower priority:", e)
    smoke = rows_limit is not None
    out_npz, out_json = (OUT_NPZ + ".smoke.npz", OUT_JSON + ".smoke") if smoke else (OUT_NPZ, OUT_JSON)
    assert not os.path.exists(out_npz) and not os.path.exists(out_json), "write-once: output exists"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    t_all = time.time()
    cd = D.AD.CanonicalDataset("musique")
    N = int(cd.n_nodes)
    pop = D.Population(cd, rows_limit)
    nq = pop.nq
    D.log("M0a musique: N %d, %d rows, %d gold nodes" % (N, nq, pop.ng_tot))
    top = np.zeros((nq, TOPK), np.int32)
    dtop = np.zeros((nq, D.ACT), np.int32)
    stop = np.full((nq, D.ACT), -1, np.int32)
    npos = np.zeros(nq, np.int64)
    pos_flat = np.zeros(pop.ng_tot, np.int64)
    Qu = D.unit_queries(cd, pop.rows)
    t_ = time.time()
    for j0, j1, SD, SS, sec, _ in D.batches(cd, pop.rows, Qu, N):
        for i in range(j1 - j0):
            j = j0 + i
            of, od, os_, npj, fv, frank = D.flat_row(SD[i], SS[i])
            top[j] = of[:TOPK]
            dtop[j] = od[:D.ACT]
            k_ = min(D.ACT, len(os_))
            stop[j, :k_] = os_[:k_]
            npos[j] = npj
            pos_flat[pop.gptr[j]:pop.gptr[j + 1]] = frank[pop.golds[j]]
        D.log("  rows %d / %d (%.0fs, RSS %.0f MB)" % (j1, nq, time.time() - t_, D._rss_mb()))
    z = np.load(REF)
    ng = int(z["gptr"][nq])
    assert (z["rows"][:nq] == pop.rows).all() and ng == pop.ng_tot
    assert (z["pos_FLAT"][:ng] == pos_flat).all(), "FLAT gold positions differ from results/L1_DEV/loc_musique__v1.npz"
    D.log("regression: FLAT positions == loc_musique__v1.npz on all %d gold nodes" % ng)
    np.savez_compressed(out_npz, rows=pop.rows, gptr=pop.gptr, gold_nodes=np.concatenate(pop.golds).astype(np.int64), pos_flat=pos_flat, top1000=top, dense_top200=dtop, splade_top200=stop, npos=npos, hops=pop.hops)
    rec = {"stage": "MuSiQue text-L3 M0a: cached L1 front end (FLAT_RRF) of the development population", "N": N, "n_rows": nq, "n_gold_nodes": int(pop.ng_tot), "topk": TOPK, "act": D.ACT, "K0": D.K0,
           "population": pop.record, "regression": "pos_FLAT == %s (sha %s) bit for bit on all %d gold nodes" % (D.rel(REF), D.sha_file(REF)[:16], ng),
           "npz": {"path": D.rel(out_npz), "sha256": D.sha_file(out_npz), "bytes": os.path.getsize(out_npz)}, "code": {"path": "scratchpad/_l3m_hits.py", "sha256": D.sha_file(os.path.abspath(__file__))},
           "pinned": D.PINNED, "pinned_repo": D.PINNED_REPO, "constants": D.CONSTANTS, "platform": D.platform_record(), "seconds": round(time.time() - t_all, 1), "process_peak_rss_mb": D.peak_rss_mb()}
    json.dump(rec, open(out_json, "w"), indent=1)
    D.log("done %s (%.0f s)" % (D.rel(out_json), time.time() - t_all))


if __name__ == "__main__":
    main()
