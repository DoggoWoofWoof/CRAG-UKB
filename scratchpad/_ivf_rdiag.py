"""FBX_SCALE Track C step 3b (addendum 12) -- is the recall loss of the IVF search on growing Freebase prefixes limited by the number of lists PROBED or by the re-rank depth R?

T1 (addendum 11) found tie-tolerant recall@3 falling by about one point per doubling of n at a fixed list length and saturating near 0.97 at high probe counts, with `self_in_top4` falling from 0.999 to 0.98.
ADC ranks candidates by <q, x_hat> = ||x_hat|| cos, so in a larger database more rows with a large decoded norm can crowd a true neighbour of a small decoded norm out of the best R = 512 by ADC even when its list is probed.
This script re-uses the T1 indexes (coarse quantiser + assignment caches, the same 4,000 queries, the same exact truth) and varies ONLY R (and, for the wiring check, the probe count up to every list).

  python -u scratchpad/_ivf_rdiag.py TEST
  python -u scratchpad/_ivf_rdiag.py RDIAG [--threads=N]          # resumable per cell; one write-once record at the end

Cells (declared in addendum 12 before any R > 512 recall was read):
  c = 32, L437  (nlist 9589):  nprobe {32, 128, 512, 9589 = every list}  x  R {512, 2048, 8192}
  c = 32, F4096 (nlist 4096):  nprobe {32, 512}                           x  R {512, 2048, 8192}
  c =  8, L437  (nlist 2397):  nprobe {32, 128}                           x  R {512, 2048, 8192}
Everything else is addendum 11's (decode, exact truth, tie-tolerant recall, scan fraction).  `norm_diag` (descriptive): at c = 32 L437, nprobe 128, R 512, the median decoded norm of the true non-self
top-3 neighbours that the search missed (by id) vs those it returned, beside the median over all rows.
"""
import os
import sys

import _ivf_transfer as T                                # imports _ivf_knn (threads parsed from argv) and _fbx_encode
import _ivf_knn as K

import io
import json
import time

import numpy as np

E = T.E
RS = [512, 2048, 8192]
CELLS = [("c32", 32, "L437", [32, 128, 512, "FULL"]), ("c32", 32, "F4096", [32, 512]), ("c08", 8, "L437", [32, 128])]
OUT = os.path.join(T.RECS, "FBX_IVF_TRANSFER_T1R__v1.json")
WORKR = os.path.join(T.WORK, "t1r")


def norm_diag(qrows, tn, nb, nrm):
    """median decoded norm of missed vs returned true non-self top-3 neighbours (ids), and of all rows."""
    miss, hit = [], []
    for i, q in enumerate(qrows):
        tr = [int(x) for x in tn[i] if x != q and x >= 0][:3]
        got = set(int(x) for x in nb[i] if x >= 0)
        for x in tr:
            (hit if x in got else miss).append(float(nrm[x]))
    return {"true_neighbours": len(miss) + len(hit), "missed": len(miss), "median_norm_missed": round(float(np.median(miss)), 4) if miss else None,
            "median_norm_returned": round(float(np.median(hit)), 4) if hit else None, "median_norm_all_rows": round(float(np.median(nrm)), 4)}


def cmd_rdiag():
    import faiss
    faiss.omp_set_num_threads(K._TH)
    assert not os.path.exists(OUT), "write-once"
    t0 = time.time()
    contract, csha = E.load_contract()
    P = E.train_paths()
    cbsha = json.load(io.open(P["rec"], encoding="utf-8"))["codebook_sha256"]
    assert E.sha_file(P["codebook"]) == cbsha
    cb16 = T.load_codebook()
    cells_all, diag = [], None
    for tag, c, layout, probes in CELLS:
        n = c * T.CHUNK_ROWS
        wd = os.path.join(T.WORK, "t1", "c%02d" % c)
        rd = os.path.join(WORKR, tag)
        os.makedirs(rd, exist_ok=True)
        nl = T.nlist_of(layout, n)
        pl = [nl if p == "FULL" else p for p in probes]
        todo = [(p, R) for p in pl for R in RS if not os.path.exists(os.path.join(rd, "cell_%s_p%d_R%d.json" % (layout, p, R)))]
        if not todo:
            for p in pl:
                for R in RS:
                    cells_all.append(json.load(io.open(os.path.join(rd, "cell_%s_p%d_R%d.json" % (layout, p, R)), encoding="utf-8")))
            continue
        for need in ("truth.npz", "coarse_%s.npy" % layout, "assign_%s.npy" % layout):
            assert os.path.exists(os.path.join(wd, need)), "T1 cache missing: %s (the R diagnostic re-uses the T1 index; a re-train is not the same index)" % os.path.join(wd, need)
        codes, _ = T.load_chunks(c, csha, cbsha)
        assert len(codes) == n
        nrm = K.all_norms(codes, cb16)
        qrows = np.sort(np.random.RandomState(T.Q_SEED).choice(n, size=T.NQ, replace=False))
        z = np.load(os.path.join(wd, "truth.npz"))
        tn, ts = z["nbr"], z["score"]
        C = np.load(os.path.join(wd, "coarse_%s.npy" % layout))
        asg = np.load(os.path.join(wd, "assign_%s.npy" % layout))
        assert len(C) == nl and len(asg) == n
        idx, counts = K.build_index(codes, cb16, C, asg)
        for p in pl:
            for R in RS:
                cp = os.path.join(rd, "cell_%s_p%d_R%d.json" % (layout, p, R))
                if os.path.exists(cp):
                    cells_all.append(json.load(io.open(cp, encoding="utf-8")))
                    continue
                t = time.time()
                nb, sc = K.search_rows(idx, codes, cb16, nrm, qrows, p, R=R)
                secs = time.time() - t
                sf = K.scan_fraction(idx, codes, cb16, nrm, qrows, counts, p)
                rec = K.recalls(qrows, tn, ts, nb, sc)
                folds = [K.recalls(qrows[f::4], tn[f::4], ts[f::4], nb[f::4], sc[f::4])["recall3_tie_tolerant"] for f in range(4)]
                cell = {"layout": layout, "chunks": c, "database_rows": int(n), "nlist": int(nl), "nprobe": int(p), "R": int(R), "scan_fraction": round(sf, 6), "queries": int(T.NQ), "seconds": round(secs, 1),
                        "queries_per_second": round(T.NQ / secs, 1), "threads": K._TH, "tie_tolerant_folds": folds, "tie_tolerant_fold_se": round(float(np.std(folds, ddof=1) / 2.0), 6)}
                cell.update(rec)
                if tag == "c32" and layout == "L437" and p == 128 and R == 512:
                    cell["norm_diag"] = norm_diag(qrows, tn, nb, nrm)
                T.wjson(cp, cell)
                cells_all.append(cell)
                T.log("c %2d %-5s nlist %5d p %5d R %5d: scan %.4f  tie-tol %.4f  recall3 %.4f  nearest %.4f  self %.4f  %.1f s (%.0f q/s)" % (c, layout, nl, p, R, sf, rec["recall3_tie_tolerant"], rec["recall3_ids"], rec["nearest_recall"], rec["self_in_top4"], secs, T.NQ / secs))
        del idx, codes, nrm
    cells_all.sort(key=lambda d: (d["chunks"], d["layout"], d["nprobe"], d["R"]))
    rec = {"RECORD": "FBX_IVF_TRANSFER_T1R", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "design": {"cells": [[a, b, c_, [str(x) for x in d]] for a, b, c_, d in CELLS], "R": RS, "queries": T.NQ, "query_seed": T.Q_SEED},
           "t1_record_sha256": E.sha_file(T.OUT), "contract_sha256": csha, "codebook_sha256": cbsha, "cells": cells_all, "seconds": round(time.time() - t0, 1), "threads": K._TH,
           "code_sha256": {"scratchpad/_ivf_rdiag.py": E.sha_file(os.path.abspath(__file__)), "scratchpad/_ivf_transfer.py": E.sha_file(os.path.abspath(T.__file__)), "scratchpad/_ivf_knn.py": E.sha_file(os.path.abspath(K.__file__))}}
    T.wjson(OUT, rec)
    T.log("RDIAG done %.0fs; record %s" % (time.time() - t0, OUT))


def _test():
    import faiss
    faiss.omp_set_num_threads(2)
    rs = np.random.RandomState(0)
    n = 3000
    cb16 = rs.randn(T.M, 256, T.SUB).astype(np.float16).astype(np.float32)
    base = rs.randint(0, 256, size=(40, T.M)).astype(np.uint8)
    codes = base[rs.randint(0, 40, size=n)].copy()
    flip = rs.rand(n, T.M) < 0.08
    codes[flip] = rs.randint(0, 256, size=int(flip.sum())).astype(np.uint8)
    nrm = K.all_norms(codes, cb16)
    qrows = np.sort(rs.choice(n, size=200, replace=False))
    tn, ts = T.exact_truth(codes, cb16, nrm, qrows, blk=700)
    X, nr = K.decode(codes, cb16, 0, n)
    X /= nr[:, None]
    nl = 24
    km = faiss.Kmeans(T.DIM, nl, niter=5, verbose=False, spherical=True, seed=1)
    km.train(np.ascontiguousarray(X[:1500].astype(np.float32)))
    C = np.ascontiguousarray(km.centroids.astype(np.float32))
    q = faiss.IndexFlatIP(T.DIM)
    q.add(C)
    asg = q.search(np.ascontiguousarray(X.astype(np.float32)), 1)[1][:, 0].astype(np.int32)
    idx, counts = K.build_index(codes, cb16, C, asg)
    r = {}
    for R in (8, 64, 4096):                                           # R above n: every scanned row is re-ranked; R tiny loses neighbours
        nb, sc = K.search_rows(idx, codes, cb16, nrm, qrows, nl, R=R)
        r[R] = K.recalls(qrows, tn, ts, nb, sc)["recall3_tie_tolerant"]
    assert r[4096] > 0.999 and r[8] <= r[4096], r
    nb, sc = K.search_rows(idx, codes, cb16, nrm, qrows, nl, R=8)
    d = norm_diag(qrows, tn, nb, nrm)
    assert d["true_neighbours"] > 0 and d["median_norm_all_rows"] > 0, d
    print("TEST ok: recall by R", r, "norm_diag", d)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a == ["TEST"]:
        _test()
    elif a == ["RDIAG"]:
        cmd_rdiag()
    else:
        raise SystemExit(__doc__)
