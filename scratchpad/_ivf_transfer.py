"""FBX_SCALE Track C step 3 (addendum 11) -- does the IVF operating point of addendum 10 transfer to REAL Freebase PQ64 codes as N grows?  Pilot T1 on prefixes of the chunk files.

Addendum 10 fixed, on WebQSP (N = 1,791,533), an inverted file over the PQ64 codes (4096 lists of ~437 rows, 32..128 lists probed, re-rank R = 512) that passes the end-to-end gate.  Whether that point holds at
245 M names is not measurable on WebQSP.  Here the SAME search is run on growing PREFIXES of the Freebase chunk codes (chunk c holds distinct-name rows [c * 131072, (c + 1) * 131072), in node order) against the
EXACT decoded-vs-decoded truth for a query sample, in two layouts:
  L437  : nlist = round(n / 437.4), the WebQSP list length held constant (nlist grows with n)
  F4096 : nlist = 4096 for every n (the scan fraction held constant at a given nprobe / nlist), only for n >= 4 chunks
over nprobe in {32, 64, 128, 256, 512} (a probe count above nlist / 2 is not run).  Nothing here gates a partition; it measures neighbour recall, scan fraction and cost.

  python -u scratchpad/_ivf_transfer.py TEST                          # synthetic: blocked exact truth == brute force; the index at nprobe = nlist reproduces it; nlist rule
  python -u scratchpad/_ivf_transfer.py T1 [--threads=N]              # the pilot (resumable per stage and cell; one write-once record at the end)

Definitions (frozen by addendum 11, written before any Freebase IVF recall exists):
  database     = the PQ64 codes of chunks 0 .. c-1 (c in {1, 2, 4, 8, 16, 32}), n = c * 131072 rows; every chunk record must verify against the frozen contract and codebook
  decode       = the accepted codebook's centroid table rounded to fp16 (as addendum 10)
  query        = 4,000 rows of the prefix (RandomState(1)), decoded-vs-decoded
  truth        = EXACT: every database row decoded, renormalised in fp32, inner product in fp32, top 4 including self (the regime of addendum 6), in row blocks of 32,768
  coarse       = faiss spherical k-means, niter 20, nredo 1, seed 2026, trained on min(n, 64 * nlist) decoded unit-norm rows (RandomState(0)) (WebQSP: 262,144 = 64 * 4096); rows listed under the largest inner product
  scoring      = addendum 10's: IndexIVFPQ by_residual False, ADC = <decoded query, decoded row>, best R = 512 re-ranked by cosine from stored decoded norms, top 4 with self
  recall       = addendum 10's tie-tolerant recall@3 (primary), ids recall@3, nearest recall; scan fraction = mean rows in the probed lists / n; rows scanned per query
"""
import os
import sys

import _ivf_knn as K                                    # parses --threads=N out of argv at import; its helpers are pinned by addendum 10 (not edited)

import hashlib
import io
import json
import time

import numpy as np

import _fbx_encode as E

DIM, M, SUB = K.DIM, K.M, K.SUB
REPO = K.REPO
WORK = os.path.join(REPO, "data", "freebase_scale", "ivf_transfer")
RECS = os.path.join(REPO, "results", "FREEBASE_SCALE", "ivf")
OUT = os.path.join(RECS, "FBX_IVF_TRANSFER_T1__v1.json")
CHUNK_ROWS = E.CHUNK_ROWS
CS = [1, 2, 4, 8, 16, 32]
LIST_LEN = 1791533 / 4096.0                               # WebQSP's mean list length (437.4)
NPROBES = [32, 64, 128, 256, 512]
FIXED_NLIST = 4096
TRAIN_PER_LIST = 64
NQ = 4000
Q_SEED = 1
TRUTH_BLOCK = 32768
LAYOUTS = ("L437", "F4096")
WEBQSP_BASELINE = {"nlist": 4096, "database_rows": 1791533, "tie_tolerant_recall3": {"32": 0.9456, "64": 0.9672, "128": 0.9838, "256": 0.9949, "512": 0.9979}, "record": "results/FREEBASE_SCALE/ivf/WEBQSP_IVF_SAMPLE__v1.json"}


def log(*a):
    K.log(*a)


def nlist_of(layout, n):
    return max(2, int(round(n / LIST_LEN))) if layout == "L437" else FIXED_NLIST


def cells_of(c):
    n = c * CHUNK_ROWS
    out = []
    for layout in LAYOUTS:
        if layout == "F4096" and c < 4:
            continue
        nl = nlist_of(layout, n)
        out.extend((layout, nl, p) for p in NPROBES if p <= nl // 2)
    return out


# ---------------------------------------------------------------------------------------------------------------- data
def load_chunks(c, contract_sha, cb_sha):
    parts, recs = [], []
    for i in range(c):
        if not E.chunk_ok(i, contract_sha, cb_sha):
            raise SystemExit("chunk %d is not present / does not verify against the frozen contract and codebook" % i)
        cp, jp = E.chunk_paths(i)
        a = np.load(cp)
        assert a.dtype == np.uint8 and a.shape[1] == M
        parts.append(a)
        r = json.load(io.open(jp, encoding="utf-8"))
        recs.append({"chunk": i, "codes_sha256": r["codes_sha256"], "row_first": r["row_first"], "row_last_exclusive": r["row_last_exclusive"]})
    return np.ascontiguousarray(np.concatenate(parts)), recs


def load_codebook():
    P = E.train_paths()
    z = np.load(P["codebook"])
    cb = z["pq_centroids"]
    assert cb.shape == (M, 256, SUB)
    return cb.astype(np.float16).astype(np.float32)


def exact_truth(codes, cb16, nrm, qrows, blk=TRUTH_BLOCK):
    """exact top-4 (incl. self) by cosine of the decoded vectors for the query rows, over every row of `codes` (blocks of `blk`)."""
    nq, n = len(qrows), len(codes)
    Qx, _ = K.decode(codes[qrows], cb16, 0, nq)
    Qx /= np.maximum(nrm[qrows], 1e-12)[:, None]
    bs = np.full((nq, 4), -3.0, np.float32)
    bi = np.full((nq, 4), -1, np.int64)
    for a in range(0, n, blk):
        b = min(a + blk, n)
        X, nr = K.decode(codes, cb16, a, b)
        X /= np.maximum(nr, 1e-12)[:, None]
        S = Qx @ X.T
        k = min(4, S.shape[1])
        p = np.argpartition(S, S.shape[1] - k, axis=1)[:, S.shape[1] - k:]
        cs = np.concatenate([bs, np.take_along_axis(S, p, 1)], 1)
        ci = np.concatenate([bi, p.astype(np.int64) + a], 1)
        o = np.argsort(-cs, axis=1, kind="stable")[:, :4]
        bs, bi = np.take_along_axis(cs, o, 1), np.take_along_axis(ci, o, 1)
    return bi.astype(np.int32), bs


def train_coarse_rows(codes, cb16, nlist, path):
    import faiss
    if os.path.exists(path):
        return np.load(path), None
    n = len(codes)
    ntr = min(n, TRAIN_PER_LIST * nlist)
    tr = np.sort(np.random.RandomState(0).choice(n, size=ntr, replace=False))
    Xs = np.empty((ntr, DIM), np.float32)
    for a in range(0, ntr, 65536):
        sel = tr[a:a + 65536]
        x, nr = K.decode(codes[sel], cb16, 0, len(sel))
        Xs[a:a + len(sel)] = x / np.maximum(nr, 1e-12)[:, None]
    t = time.time()
    km = faiss.Kmeans(DIM, nlist, niter=20, nredo=1, verbose=False, spherical=True, seed=K.KM_SEED, max_points_per_centroid=1024)
    km.train(Xs)
    C = np.ascontiguousarray(km.centroids.astype(np.float32))
    sec = time.time() - t
    log("coarse k-means nlist %d on %d rows: %.0fs" % (nlist, ntr, sec))
    K.save_npy(path, C)
    return C, {"train_rows": int(ntr), "kmeans_seconds": round(sec, 1)}


def wjson(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    io.open(tmp, "w", encoding="utf-8", newline="\n").write(json.dumps(obj, indent=1))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------------------------- pilot
def cmd_t1():
    import faiss
    faiss.omp_set_num_threads(K._TH)
    assert not os.path.exists(OUT), "write-once"
    t0 = time.time()
    contract, csha = E.load_contract()
    P = E.train_paths()
    cb_rec = json.load(io.open(P["rec"], encoding="utf-8"))
    cbsha = cb_rec["codebook_sha256"]
    assert E.sha_file(P["codebook"]) == cbsha
    cb16 = load_codebook()
    cells_all, costs = [], {}
    for c in CS:
        n = c * CHUNK_ROWS
        wd = os.path.join(WORK, "t1", "c%02d" % c)
        os.makedirs(wd, exist_ok=True)
        kp = os.path.join(wd, "costs.json")
        cc = json.load(io.open(kp, encoding="utf-8")) if os.path.exists(kp) else {}
        costs["c%d" % c] = cc
        todo = [x for x in cells_of(c) if not os.path.exists(os.path.join(wd, "cell_%s_p%d.json" % (x[0], x[2])))]
        if not todo:
            log("c %d: every cell present" % c)
            for x in cells_of(c):
                cells_all.append(json.load(io.open(os.path.join(wd, "cell_%s_p%d.json" % (x[0], x[2])), encoding="utf-8")))
            continue
        codes, _ = load_chunks(c, csha, cbsha)
        assert len(codes) == n, "chunk %d is short (n = %d)" % (c, len(codes))
        nrm = K.all_norms(codes, cb16)
        qrows = np.sort(np.random.RandomState(Q_SEED).choice(n, size=NQ, replace=False))
        tp = os.path.join(wd, "truth.npz")
        if os.path.exists(tp):
            z = np.load(tp)
            tn, ts = z["nbr"], z["score"]
        else:
            t = time.time()
            tn, ts = exact_truth(codes, cb16, nrm, qrows)
            np.savez(tp + ".tmp.npz", nbr=tn, score=ts)
            os.replace(tp + ".tmp.npz", tp)
            cc["truth_seconds"] = round(time.time() - t, 1)
            wjson(kp, cc)
            log("c %d (n %d): exact truth for %d queries: %.0fs" % (c, n, NQ, time.time() - t))
        for layout in LAYOUTS:
            lc = [x for x in cells_of(c) if x[0] == layout]
            if not lc:
                continue
            nl = lc[0][1]
            pend = [x for x in lc if x in todo]
            if not pend:
                for x in lc:
                    cells_all.append(json.load(io.open(os.path.join(wd, "cell_%s_p%d.json" % (x[0], x[2])), encoding="utf-8")))
                continue
            cpath = os.path.join(wd, "coarse_%s.npy" % layout)
            C, ci = train_coarse_rows(codes, cb16, nl, cpath)
            apath = os.path.join(wd, "assign_%s.npy" % layout)
            t = time.time()
            had = os.path.exists(apath)
            asg = K.assign_all(codes, cb16, C, apath)
            asec = None if had else round(time.time() - t, 1)
            t = time.time()
            idx, counts = K.build_index(codes, cb16, C, asg)
            bsec = round(time.time() - t, 1)
            old = cc.get(layout, {})
            cc[layout] = {"nlist": int(nl), "list_len_mean": round(n / nl, 1), "list_len_max": int(counts.max()), "list_len_median": float(np.median(counts)), "assign_seconds": asec, "build_seconds": bsec}
            if ci:
                cc[layout].update(ci)
            for k_ in ("train_rows", "kmeans_seconds", "assign_seconds"):          # a resumed run keeps the timings of the stage that did the work
                if cc[layout].get(k_) is None and old.get(k_) is not None:
                    cc[layout][k_] = old[k_]
            wjson(kp, cc)
            for x in lc:
                cp = os.path.join(wd, "cell_%s_p%d.json" % (x[0], x[2]))
                if os.path.exists(cp):
                    cells_all.append(json.load(io.open(cp, encoding="utf-8")))
                    continue
                npb = x[2]
                t = time.time()
                nb, sc = K.search_rows(idx, codes, cb16, nrm, qrows, npb)
                secs = time.time() - t
                sf = K.scan_fraction(idx, codes, cb16, nrm, qrows, counts, npb)
                rec = K.recalls(qrows, tn, ts, nb, sc)
                cell = {"layout": layout, "chunks": c, "database_rows": int(n), "nlist": int(nl), "nprobe": int(npb), "scan_fraction": round(sf, 6), "rows_scanned_per_query": round(sf * n, 0),
                        "queries": int(NQ), "seconds": round(secs, 1), "queries_per_second": round(NQ / secs, 1), "threads": K._TH}
                cell.update(rec)
                folds = [K.recalls(qrows[f::4], tn[f::4], ts[f::4], nb[f::4], sc[f::4])["recall3_tie_tolerant"] for f in range(4)]     # four interleaved query folds: spread of the tie-tolerant recall
                cell["tie_tolerant_folds"] = folds
                cell["tie_tolerant_fold_se"] = round(float(np.std(folds, ddof=1) / 2.0), 6)
                wjson(cp, cell)
                cells_all.append(cell)
                log("c %2d %-5s nlist %5d p %3d: scan %.4f  rows/q %8.0f  recall3 %.4f  tie-tol %.4f  nearest %.4f  self %.4f  %.0f q/s" % (c, layout, nl, npb, sf, sf * n, rec["recall3_ids"], rec["recall3_tie_tolerant"], rec["nearest_recall"], rec["self_in_top4"], NQ / secs))
            del idx
        del codes, nrm
    cells_all.sort(key=lambda d: (d["layout"], d["chunks"], d["nprobe"]))
    chunk_recs = []
    for i in range(max(CS)):
        r = json.load(io.open(E.chunk_paths(i)[1], encoding="utf-8"))
        chunk_recs.append({"chunk": i, "codes_sha256": r["codes_sha256"], "row_first": r["row_first"], "row_last_exclusive": r["row_last_exclusive"]})
    rec = {"RECORD": "FBX_IVF_TRANSFER_T1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "design": {"chunks": CS, "layouts": list(LAYOUTS), "nprobes": NPROBES, "list_len": LIST_LEN, "fixed_nlist": FIXED_NLIST,
           "train_rows_per_list": TRAIN_PER_LIST, "queries": NQ, "query_seed": Q_SEED, "rerank_R": K.R_RERANK, "truth_block": TRUTH_BLOCK}, "webqsp_baseline": WEBQSP_BASELINE,
           "contract_sha256": csha, "codebook_sha256": cbsha, "chunk_records": chunk_recs, "cells": cells_all, "costs": costs, "seconds": round(time.time() - t0, 1), "threads": K._TH,
           "code_sha256": {"scratchpad/_ivf_transfer.py": E.sha_file(os.path.abspath(__file__)), "scratchpad/_ivf_knn.py": E.sha_file(os.path.abspath(K.__file__))}}
    wjson(OUT, rec)
    log("T1 done %.0fs; record %s" % (time.time() - t0, OUT))


# ---------------------------------------------------------------------------------------------------------------- test
def _test():
    import faiss
    faiss.omp_set_num_threads(2)
    rs = np.random.RandomState(0)
    n = 3000
    cb16 = rs.randn(M, 256, SUB).astype(np.float16).astype(np.float32)
    base = rs.randint(0, 256, size=(40, M)).astype(np.uint8)
    codes = base[rs.randint(0, 40, size=n)].copy()                       # clustered rows (each row is one of 40 prototypes with a few sub-codes changed)
    flip = rs.rand(n, M) < 0.08
    codes[flip] = rs.randint(0, 256, size=int(flip.sum())).astype(np.uint8)
    nrm = K.all_norms(codes, cb16)
    qrows = np.sort(rs.choice(n, size=200, replace=False))
    tn, ts = exact_truth(codes, cb16, nrm, qrows, blk=700)               # block boundary check
    X, nr = K.decode(codes, cb16, 0, n)
    X /= nr[:, None]
    S = X[qrows] @ X.T
    bf = np.argsort(-S, axis=1, kind="stable")[:, :4]
    assert np.allclose(np.take_along_axis(S, bf, 1), ts, atol=1e-5), "blocked exact truth != brute force"
    assert nlist_of("L437", 4 * CHUNK_ROWS) == int(round(4 * CHUNK_ROWS / LIST_LEN)) and nlist_of("F4096", 10) == 4096
    assert [x[2] for x in cells_of(1)] == [32, 64, 128] and [x[1] for x in cells_of(1)] == [300] * 3, cells_of(1)
    assert all(x[2] <= x[1] // 2 for c in CS for x in cells_of(c))
    nl = 24
    km = faiss.Kmeans(DIM, nl, niter=5, verbose=False, spherical=True, seed=1)
    km.train((X[:1500] / np.linalg.norm(X[:1500], axis=1, keepdims=True)).astype(np.float32))
    C = np.ascontiguousarray(km.centroids.astype(np.float32))
    q = faiss.IndexFlatIP(DIM)
    q.add(C)
    asg = q.search(np.ascontiguousarray(X.astype(np.float32)), 1)[1][:, 0].astype(np.int32)
    idx, counts = K.build_index(codes, cb16, C, asg)
    nb, sc = K.search_rows(idx, codes, cb16, nrm, qrows, nl)                # every list probed, R = 512 > n/6: the exact result up to ties
    rec = K.recalls(qrows, tn, ts, nb, sc)
    assert rec["recall3_tie_tolerant"] > 0.999, rec
    nb4, sc4 = K.search_rows(idx, codes, cb16, nrm, qrows, 3)               # partial probe: recall below 1, folds defined
    folds = [K.recalls(qrows[f::4], tn[f::4], ts[f::4], nb4[f::4], sc4[f::4])["recall3_tie_tolerant"] for f in range(4)]
    assert len(folds) == 4 and 0.0 <= min(folds) and max(folds) <= 1.0, folds
    print("folds", folds)
    print("TEST ok:", rec, "lists", int(counts.min()), int(counts.max()))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a == ["TEST"]:
        _test()
    elif a == ["T1"]:
        cmd_t1()
    else:
        raise SystemExit(__doc__)
