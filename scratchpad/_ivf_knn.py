"""FBX_SCALE Track C step 2 (addendum 10) -- IVF approximate search over the PQ64 codes of WebQSP, scored against the exhaustive PQ64 truth.

At Freebase scale the exhaustive decoded-vs-decoded search of addendum 6 is not an option (245 M names x 245 M names); the deployed search is an inverted file over the stored PQ codes: a coarse
quantiser picks `nprobe` lists, the codes of those lists are scored by asymmetric distance computation (ADC), the best R candidates are re-ranked by cosine.  This script measures how much of the
exhaustive PQ64 neighbourhood that approximation keeps (phase A, SAMPLE) and produces the full KNN family it yields (phase B, FULL), which `_ivf_phg.py` / `_ivf_eval.py` then push through the
frozen partitioner and the addendum-1 gate.

  python -u scratchpad/_ivf_knn.py TEST                                 # synthetic: decode == faiss decode; nprobe = nlist reproduces brute force; recall grows with nprobe
  python -u scratchpad/_ivf_knn.py SMOKE                                # wiring check on the real data at nprobe = nlist (recall ~1 by construction; also caches the nlist 1024 coarse quantiser)
  python -u scratchpad/_ivf_knn.py SAMPLE                               # phase A: nlist x nprobe grid on a 100,000-row query sample vs the exhaustive PQ64 truth (write-once record)
  python -u scratchpad/_ivf_knn.py FULL <nlist> <nprobe> <tag>          # phase B: every row as a query at (nlist, nprobe); merge; keys file; recall vs the exhaustive truth and the frozen exact KNN
Threads: --threads=N anywhere on the command line (or env IVF_THREADS; default 4); it is also the --cpus of the rx job.

Definitions (frozen by addendum 10, written before any IVF result exists):
  database      = the 1,791,533 distinct rows of WebQSP, as their PQ64 codes (data/freebase_scale/pq/pq64/codes.npy); the PQ centroid table is rounded to fp16 first, so a decoded vector is exactly
                  the fp16 storage vector of the exhaustive search (addendum 6).
  query         = the decoded vector of the query row (decoded-vs-decoded, as in the deployed regime).
  coarse        = faiss spherical k-means, niter 20, nredo 1, seed 2026, trained on 262,144 decoded rows (RandomState(0)), unit-norm; a row belongs to the list of its largest inner product.
  scoring       = IndexIVFPQ, by_residual = False, inner product: ADC = <decoded query, decoded row> exactly (orthogonal sub-spaces); the best R = 512 by ADC are re-ranked by cosine
                  ADC / (||query|| ||row||) with the stored decoded row norm; the top 4 (self included, as in the exhaustive search) are returned.
  recall        = addendum 6's rowwise recall@3 (self dropped, first 3 others; ids) and nearest-neighbour recall; plus a tie-tolerant score recall (a returned neighbour counts if its cosine is at least the
                  exhaustive third-best non-self cosine minus 1e-5), because rows with identical codes tie.
  scan fraction = mean over queries of (rows in the probed lists) / N: the quantity that transfers across N.
"""
import os
import sys

_TH = int(os.environ.get("IVF_THREADS", "4"))
for _a in list(sys.argv):
    if _a.startswith("--threads="):
        _TH = int(_a.split("=", 1)[1])
        sys.argv.remove(_a)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(_TH)

import io
import json
import time

import numpy as np

import _pq_knn as Q

log, sha_file, sha_arr, rel, wj, save_npy = Q.log, Q.sha_file, Q.sha_arr, Q.rel, Q.wj, Q.save_npy
DIM, M, SUB = Q.DIM, 64, 24
REPO = Q.REPO
PQD = os.path.join(Q.WORK, "pq64")
IVFD = os.path.join(Q.WORK, "ivf")
RECS = os.path.join(REPO, "results", "FREEBASE_SCALE", "ivf")
R_RERANK = 512                  # fixed from the full-probe wiring check (SMOKE): R 32 -> 0.979, 128 -> 0.998, 512 -> 1.000 recall@3 at nprobe = nlist (decoded norms spread 0.69-1.02, ADC ranks by ||x|| cos)
TRAIN_ROWS = 262144
KM_SEED = 2026
SAMPLE_ROWS = 100000
SAMPLE_SEED = 1
SAMPLE_GRID = {1024: [8, 16, 32, 64, 128, 256], 4096: [32, 64, 128, 256, 512, 1024]}       # nprobe / nlist = 1/128 ... 1/4
QBLOCK = 10000
FULL_BLOCK = 20000
TIE_EPS = 1e-5


# ---------------------------------------------------------------------------------------------------------------- decode
def decode(codes, cb16, a, b):
    """fp32 decoded vectors of rows a:b (the fp16 centroid table: bitwise the exhaustive search's fp16 storage vectors) and their L2 norms."""
    c = codes[a:b]
    out = np.empty((len(c), DIM), np.float32)
    for m in range(M):
        out[:, m * SUB:(m + 1) * SUB] = cb16[m][c[:, m]]
    return out, np.linalg.norm(out, axis=1).astype(np.float32)


def all_norms(codes, cb16, chunk=131072):
    nrm = np.empty(len(codes), np.float32)
    for a in range(0, len(codes), chunk):
        nrm[a:a + chunk] = decode(codes, cb16, a, min(a + chunk, len(codes)))[1]
    return nrm


def load_pq():
    codes = np.load(os.path.join(PQD, "codes.npy"))
    cb = np.load(os.path.join(PQD, "pq_centroids.npy"))
    assert codes.dtype == np.uint8 and codes.shape[1] == M and cb.shape == (M, 256, SUB)
    return codes, cb.astype(np.float16).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------- index
def train_coarse(codes, cb16, nlist, path):
    import faiss
    if os.path.exists(path):
        return np.load(path)
    nu = len(codes)
    tr = np.sort(np.random.RandomState(0).choice(nu, size=min(TRAIN_ROWS, nu), replace=False))
    Xs = np.empty((len(tr), DIM), np.float32)
    for a in range(0, len(tr), 65536):
        sel = tr[a:a + 65536]
        x = np.empty((len(sel), DIM), np.float32)
        for m in range(M):
            x[:, m * SUB:(m + 1) * SUB] = cb16[m][codes[sel, m]]
        Xs[a:a + len(sel)] = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    t = time.time()
    km = faiss.Kmeans(DIM, nlist, niter=20, nredo=1, verbose=False, spherical=True, seed=KM_SEED, max_points_per_centroid=1024)
    km.train(Xs)
    C = np.ascontiguousarray(km.centroids.astype(np.float32))
    log("coarse k-means nlist %d on %d rows: %.0fs" % (nlist, len(Xs), time.time() - t))
    save_npy(path, C)
    return C


def assign_all(codes, cb16, C, path, chunk=32768):
    import faiss
    if os.path.exists(path):
        return np.load(path)
    t = time.time()
    q = faiss.IndexFlatIP(DIM)
    q.add(C)
    a_ = np.empty(len(codes), np.int32)
    for a in range(0, len(codes), chunk):
        x, nr = decode(codes, cb16, a, min(a + chunk, len(codes)))
        x /= np.maximum(nr, 1e-12)[:, None]
        a_[a:a + len(x)] = q.search(x, 1)[1][:, 0]
    log("assigned %d rows to %d lists: %.0fs" % (len(codes), len(C), time.time() - t))
    save_npy(path, a_)
    return a_


def build_index(codes, cb16, C, assign):
    import faiss
    nlist = len(C)
    quant = faiss.IndexFlatIP(DIM)
    quant.add(C)
    idx = faiss.IndexIVFPQ(quant, DIM, nlist, M, 8, faiss.METRIC_INNER_PRODUCT)
    idx.by_residual = False
    faiss.copy_array_to_vector(np.ascontiguousarray(cb16).ravel(), idx.pq.centroids)
    idx.is_trained = True
    order = np.argsort(assign, kind="stable")
    counts = np.bincount(assign, minlength=nlist)
    ends = np.cumsum(counts)
    il = idx.invlists
    s = 0
    for l in range(nlist):
        e = int(ends[l])
        if e > s:
            ids = np.ascontiguousarray(order[s:e].astype(np.int64))
            cl = np.ascontiguousarray(codes[ids])
            il.add_entries(l, len(ids), faiss.swig_ptr(ids), faiss.swig_ptr(cl))
        s = e
    idx.ntotal = len(codes)
    return idx, counts.astype(np.int64)


def search_rows(idx, codes, cb16, nrm, qrows, nprobe, R=None):
    """top 4 (cosine re-rank of the best R_RERANK by ADC) for the query rows qrows; returns (ids int32 [n,4], cosine float32 [n,4])."""
    n = len(qrows)
    Qx = np.empty((n, DIM), np.float32)
    for m in range(M):
        Qx[:, m * SUB:(m + 1) * SUB] = cb16[m][codes[qrows, m]]
    qn = nrm[qrows]
    idx.nprobe = int(nprobe)
    D, I = idx.search(Qx, int(R or R_RERANK))
    valid = I >= 0
    cos = np.where(valid, D / np.maximum(nrm[np.where(valid, I, 0)], 1e-12) / np.maximum(qn, 1e-12)[:, None], -2.0).astype(np.float32)
    o = np.argsort(-cos, axis=1, kind="stable")[:, :4]
    ids4 = np.take_along_axis(I, o, 1)
    sc4 = np.take_along_axis(cos, o, 1)
    ids4 = np.where(sc4 > -1.5, ids4, -1).astype(np.int32)
    return ids4, sc4


def scan_fraction(idx, codes, cb16, nrm, qrows, counts, nprobe):
    """mean rows in the probed lists / N, over the given queries (the coarse quantiser's own ranking of the lists)."""
    n = len(qrows)
    Qx = np.empty((n, DIM), np.float32)
    for m in range(M):
        Qx[:, m * SUB:(m + 1) * SUB] = cb16[m][codes[qrows, m]]
    I = idx.quantizer.search(Qx, int(nprobe))[1]
    return float(counts[I].sum(axis=1).mean() / len(codes))


# ---------------------------------------------------------------------------------------------------------------- recall
def others(ids, nb, sc, keep=3):
    """per row the first `keep` entries of nb that are not the row itself, with their scores (padded -1 / -2)."""
    n = len(nb)
    ns = (nb != ids[:, None]) & (nb >= 0)
    rank = np.cumsum(ns, axis=1) * ns
    oi = np.full((n, keep), -1, np.int64)
    os_ = np.full((n, keep), -2.0, np.float64)
    for c in range(1, keep + 1):
        r, cc = np.nonzero(rank == c)
        oi[r, c - 1] = nb[r, cc]
        os_[r, c - 1] = sc[r, cc]
    return oi, os_


def recalls(ids, ref_nb, ref_sc, new_nb, new_sc):
    ri, rs = others(ids, ref_nb.astype(np.int64), ref_sc, 3)
    ni, ns_ = others(ids, new_nb.astype(np.int64), new_sc, 3)
    full = (ri >= 0).all(axis=1)
    hit = (ri[:, :, None] == ni[:, None, :]).any(axis=2) & (ri >= 0)
    r3 = float(hit[full].sum() / (full.sum() * 3))
    r1 = float((ri[full, 0][:, None] == ni[full]).any(axis=1).mean())
    kth = rs[:, 2] - TIE_EPS
    tt = float(((ns_ >= kth[:, None]) & (ni >= 0))[full].sum() / (full.sum() * 3))
    self_in = float((new_nb == ids[:, None]).any(axis=1).mean())
    return {"recall3_ids": round(r3, 6), "nearest_recall": round(r1, 6), "recall3_tie_tolerant": round(tt, 6), "self_in_top4": round(self_in, 6), "rows_scored": int(full.sum())}


def load_truth(nu):
    d = os.path.join(PQD, "search")
    nbr = np.full((nu, 4), -1, np.int32)
    sc = np.zeros((nu, 4), np.float32)
    nsh = (nu + Q.QSHARD - 1) // Q.QSHARD
    for s in range(nsh):
        z = np.load(os.path.join(d, "q%05d.npz" % s))
        r0 = s * Q.QSHARD
        nbr[r0:r0 + len(z["nbr"])], sc[r0:r0 + len(z["nbr"])] = z["nbr"], z["score"]
    return nbr, sc


# ---------------------------------------------------------------------------------------------------------------- phase A
def cmd_sample():
    import faiss
    faiss.omp_set_num_threads(_TH)
    out = os.path.join(RECS, "WEBQSP_IVF_SAMPLE__v1.json")
    assert not os.path.exists(out), "write-once"
    t0 = time.time()
    codes, cb16 = load_pq()
    nu = len(codes)
    nrm = all_norms(codes, cb16)
    # decode check against faiss's own ProductQuantizer on real rows
    pq = faiss.ProductQuantizer(DIM, M, 8)
    faiss.copy_array_to_vector(np.load(os.path.join(PQD, "pq_centroids.npy")).ravel(), pq.centroids)
    chk = np.sort(np.random.RandomState(5).choice(nu, size=2000, replace=False))
    ref16 = pq.decode(np.ascontiguousarray(codes[chk])).astype(np.float16)
    mine16 = np.stack([decode(codes, cb16, int(i), int(i) + 1)[0][0] for i in chk]).astype(np.float16)
    assert (ref16.view(np.uint16) == mine16.view(np.uint16)).all(), "decoded vectors differ from faiss's decode"
    tn, ts = load_truth(nu)
    qids = np.sort(np.random.RandomState(SAMPLE_SEED).choice(nu, size=SAMPLE_ROWS, replace=False))
    ref_nb, ref_sc = tn[qids], ts[qids]
    cells, files = [], {}
    os.makedirs(os.path.join(IVFD, "sample"), exist_ok=True)
    for nlist, probes in SAMPLE_GRID.items():
        C = train_coarse(codes, cb16, nlist, os.path.join(IVFD, "coarse_n%d.npy" % nlist))
        asg = assign_all(codes, cb16, C, os.path.join(IVFD, "assign_n%d.npy" % nlist))
        files["coarse_n%d" % nlist] = sha_arr(C)
        idx, counts = build_index(codes, cb16, C, asg)
        log("nlist %d: list sizes min %d median %d max %d (mean %.0f)" % (nlist, counts.min(), np.median(counts), counts.max(), counts.mean()))
        for npb in probes:
            cp = os.path.join(IVFD, "sample", "cell_n%d_p%d.json" % (nlist, npb))
            if os.path.exists(cp):
                cells.append(json.load(io.open(cp, encoding="utf-8")))
                continue
            t = time.time()
            nb = np.empty((len(qids), 4), np.int32)
            sc = np.empty((len(qids), 4), np.float32)
            for a in range(0, len(qids), QBLOCK):
                nb[a:a + QBLOCK], sc[a:a + QBLOCK] = search_rows(idx, codes, cb16, nrm, qids[a:a + QBLOCK], npb)
            secs = time.time() - t
            sf = scan_fraction(idx, codes, cb16, nrm, qids[:20000], counts, npb)
            rec = recalls(qids, ref_nb, ref_sc, nb, sc)
            cell = {"nlist": nlist, "nprobe": npb, "nprobe_over_nlist": round(npb / nlist, 6), "scan_fraction": round(sf, 6), "queries": len(qids), "seconds": round(secs, 1),
                    "queries_per_second": round(len(qids) / secs, 1), "threads": _TH}
            cell.update(rec)
            io.open(cp, "w", encoding="utf-8", newline="\n").write(json.dumps(cell, indent=1))
            cells.append(cell)
            log("n%d p%d: scan %.4f  recall3 %.4f  nearest %.4f  tie-tol %.4f  self %.4f  %.0f q/s" % (nlist, npb, sf, rec["recall3_ids"], rec["nearest_recall"], rec["recall3_tie_tolerant"], rec["self_in_top4"], len(qids) / secs))
        del idx
    wj(out, {"RECORD": "PQ_IVF_SAMPLE", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "database_rows": int(nu), "query_rows": int(len(qids)), "query_seed": SAMPLE_SEED, "rerank_R": R_RERANK,
             "coarse_train_rows": TRAIN_ROWS, "kmeans_seed": KM_SEED, "grid": {str(k): v for k, v in SAMPLE_GRID.items()}, "cells": cells, "coarse_sha256": files,
             "truth": "exhaustive PQ64 decoded-vs-decoded search (data/freebase_scale/pq/pq64/search/q*.npz)", "seconds": round(time.time() - t0, 1), "threads": _TH,
             "code_sha256": sha_file(os.path.abspath(__file__))})
    log("SAMPLE done %.0fs" % (time.time() - t0))


def cmd_smoke():
    """wiring check on the real data at nprobe = nlist (every list probed: recall is ~1 only once R is deep enough; this is how R was fixed; the grid's recall is NOT read here); builds and caches the nlist 1024 coarse quantiser SAMPLE reuses."""
    import faiss
    faiss.omp_set_num_threads(_TH)
    codes, cb16 = load_pq()
    nu = len(codes)
    nrm = all_norms(codes, cb16)
    pq = faiss.ProductQuantizer(DIM, M, 8)
    faiss.copy_array_to_vector(np.load(os.path.join(PQD, "pq_centroids.npy")).ravel(), pq.centroids)
    chk = np.sort(np.random.RandomState(5).choice(nu, size=2000, replace=False))
    ref16 = pq.decode(np.ascontiguousarray(codes[chk])).astype(np.float16)
    mine16 = np.stack([decode(codes, cb16, int(i), int(i) + 1)[0][0] for i in chk]).astype(np.float16)
    assert (ref16.view(np.uint16) == mine16.view(np.uint16)).all(), "decoded vectors differ from faiss's decode"
    os.makedirs(IVFD, exist_ok=True)
    C = train_coarse(codes, cb16, 1024, os.path.join(IVFD, "coarse_n1024.npy"))
    asg = assign_all(codes, cb16, C, os.path.join(IVFD, "assign_n1024.npy"))
    idx, counts = build_index(codes, cb16, C, asg)
    assert int(counts.sum()) == nu and idx.ntotal == nu
    tn, ts = load_truth(nu)
    qids = np.sort(np.random.RandomState(99).choice(nu, size=300, replace=False))
    sf = scan_fraction(idx, codes, cb16, nrm, qids, counts, 1024)
    assert sf > 0.999
    for R in (32, 128, 512):                                  # the full-probe ceiling as a function of the re-rank depth (ADC ranks by <q, x> = ||x|| cos; the re-rank divides by ||x||)
        nb, sc = search_rows(idx, codes, cb16, nrm, qids, 1024, R)
        rec = recalls(qids, tn[qids], ts[qids], nb, sc)
        log("SMOKE (nprobe = nlist, 300 queries, R %d): scan %.3f  %s" % (R, sf, json.dumps(rec)))
    log("decoded norm: min %.4f p05 %.4f median %.4f p95 %.4f max %.4f" % tuple(np.percentile(nrm, [0, 5, 50, 95, 100])))


# ---------------------------------------------------------------------------------------------------------------- phase B
def cmd_full(nlist, nprobe, tag):
    import faiss
    faiss.omp_set_num_threads(_TH)
    nlist, nprobe = int(nlist), int(nprobe)
    out = os.path.join(RECS, "WEBQSP_IVF_%s_KNN__v1.json" % tag)
    assert not os.path.exists(out), "write-once"
    t0 = time.time()
    codes, cb16 = load_pq()
    nu = len(codes)
    nrm = all_norms(codes, cb16)
    C = np.load(os.path.join(IVFD, "coarse_n%d.npy" % nlist))                  # the coarse quantiser of phase A, unchanged
    asg = np.load(os.path.join(IVFD, "assign_n%d.npy" % nlist))
    idx, counts = build_index(codes, cb16, C, asg)
    d = os.path.join(IVFD, tag)
    os.makedirs(os.path.join(d, "search"), exist_ok=True)
    nbr = np.empty((nu, 4), np.int32)
    sc = np.empty((nu, 4), np.float32)
    nbl = (nu + FULL_BLOCK - 1) // FULL_BLOCK
    for b in range(nbl):
        r0, r1 = b * FULL_BLOCK, min((b + 1) * FULL_BLOCK, nu)
        cp = os.path.join(d, "search", "q%05d.npz" % b)
        if os.path.exists(cp):
            z = np.load(cp)
            nbr[r0:r1], sc[r0:r1] = z["nbr"], z["score"]
            continue
        t = time.time()
        nbr[r0:r1], sc[r0:r1] = search_rows(idx, codes, cb16, nrm, np.arange(r0, r1), nprobe)
        tmp = cp + ".tmp.npz"
        np.savez(tmp, nbr=nbr[r0:r1], score=sc[r0:r1], row_start=r0)
        os.replace(tmp, cp)
        log("  block %d/%d (%d rows) %.0fs" % (b + 1, nbl, r1 - r0, time.time() - t))
    sf = scan_fraction(idx, codes, cb16, nrm, np.arange(0, nu, 50), counts, nprobe)
    pos2u, first = Q.load_prep()
    N = len(pos2u)
    a, b, w = Q.merge(nbr, sc, pos2u, first)
    tn, ts = load_truth(nu)
    ids_all = np.arange(nu)
    rec = recalls(ids_all, tn, ts, nbr, sc)
    z = np.load(Q.KEYS)
    assert int(z["N"][0]) == N
    S, K_ex = z["STRUCT"], z["KNN"]
    kz = np.load(os.path.join(PQD, "keys_pq64.npz"))
    K_ivf = np.setdiff1d(Q.pair_keys(a, b, N), S, assume_unique=True)
    inter = int(len(np.intersect1d(K_ivf, K_ex, assume_unique=True)))
    inter_pq = int(len(np.intersect1d(K_ivf, kz["KNN"], assume_unique=True)))
    kp = os.path.join(d, "keys_%s.npz" % tag)
    np.savez(kp, N=np.array([N]), STRUCT=S, KNN=K_ivf)
    wj(out, {"RECORD": "PQ_IVF_FULL_KNN", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tag": tag, "nlist": nlist, "nprobe": nprobe, "scan_fraction": round(sf, 6), "rerank_R": R_RERANK,
             "distinct_rows": int(nu), "recall_vs_exhaustive_pq64": rec,
             "knn_family": {"pairs_exact": int(len(K_ex)), "pairs_pq64_exhaustive": int(len(kz["KNN"])), "pairs_ivf": int(len(K_ivf)), "common_with_exact": inter, "recall_of_exact_pairs": round(inter / len(K_ex), 6),
                            "common_with_pq64_exhaustive": inter_pq, "recall_of_pq64_pairs": round(inter_pq / len(kz["KNN"]), 6), "precision_vs_exact": round(inter / len(K_ivf), 6)},
             "files": {"keys": {"file": rel(kp), "sha256": sha_file(kp), "STRUCT": int(len(S)), "KNN": int(len(K_ivf))}, "coarse_sha256": sha_arr(C)},
             "seconds": round(time.time() - t0, 1), "threads": _TH, "code_sha256": sha_file(os.path.abspath(__file__))})
    log("FULL %s: scan %.4f recall3 %.4f tie-tol %.4f; KNN pairs %d (exact recall %.4f, pq64 recall %.4f)" % (tag, sf, rec["recall3_ids"], rec["recall3_tie_tolerant"], len(K_ivf), inter / len(K_ex), inter_pq / len(kz["KNN"])))


# ---------------------------------------------------------------------------------------------------------------- test
def _test():
    import faiss
    faiss.omp_set_num_threads(_TH)
    rng = np.random.RandomState(7)
    nu, nlist = 30000, 64
    cb16 = (rng.randn(M, 256, SUB) * 0.05).astype(np.float16).astype(np.float32)
    base = rng.randint(0, 256, size=(300, M)).astype(np.uint8)                          # clustered codes: each row copies a base row with ~8 positions changed
    codes = base[rng.randint(0, 300, nu)].copy()
    flip = rng.rand(nu, M) < 0.12
    codes[flip] = rng.randint(0, 256, size=int(flip.sum())).astype(np.uint8)
    pq = faiss.ProductQuantizer(DIM, M, 8)
    faiss.copy_array_to_vector(cb16.astype(np.float32).ravel(), pq.centroids)
    ref16 = pq.decode(np.ascontiguousarray(codes[:500])).astype(np.float16)
    mine = decode(codes, cb16, 0, 500)[0]
    assert (ref16.view(np.uint16) == mine.astype(np.float16).view(np.uint16)).all()
    nrm = all_norms(codes, cb16)
    X, _ = decode(codes, cb16, 0, nu)
    Xn = X / nrm[:, None]
    C = Xn[rng.choice(nu, nlist, replace=False)].copy()
    for _ in range(5):
        a = np.argmax(Xn @ C.T, axis=1)
        for l in range(nlist):
            s = Xn[a == l]
            if len(s):
                c = s.mean(0)
                C[l] = c / np.linalg.norm(c)
    asg = np.argmax(Xn @ C.T, axis=1).astype(np.int32)
    idx, counts = build_index(codes, cb16, C, asg)
    assert int(counts.sum()) == nu and idx.ntotal == nu
    qr = np.sort(rng.choice(nu, 400, replace=False))
    S = Xn[qr] @ Xn.T
    ref_nb = np.argsort(-S, axis=1, kind="stable")[:, :4].astype(np.int32)
    ref_sc = np.take_along_axis(S, ref_nb.astype(np.int64), 1).astype(np.float32)
    nb, sc = search_rows(idx, codes, cb16, nrm, qr, nlist)                               # every list probed: only the re-rank depth R separates it from brute force
    assert np.abs(np.sort(sc, axis=1)[:, ::-1] - np.sort(ref_sc, axis=1)[:, ::-1]).max() < 2e-4, "full probe disagrees with brute force"
    full = recalls(qr, ref_nb, ref_sc, nb, sc)
    assert full["recall3_tie_tolerant"] > 0.999 and full["self_in_top4"] > 0.99, full
    prev = -1.0
    for npb in (1, 4, 16, 64):
        nb, sc = search_rows(idx, codes, cb16, nrm, qr, npb)
        r = recalls(qr, ref_nb, ref_sc, nb, sc)["recall3_tie_tolerant"]
        assert r >= prev - 0.02, (npb, r, prev)
        prev = r
        sf = scan_fraction(idx, codes, cb16, nrm, qr, counts, npb)
        print("  nprobe %2d: scan %.3f tie-tolerant recall@3 %.3f" % (npb, sf, r))
    # others() on a tie: self absent, duplicates
    oi, os_ = others(np.array([5, 6]), np.array([[5, 1, 2, 3], [9, 8, 7, 6]]), np.array([[1.0, .9, .8, .7], [.9, .8, .7, .6]], np.float32), 3)
    assert oi.tolist() == [[1, 2, 3], [9, 8, 7]]
    print("TEST PASS: decode == faiss decode; full probe == brute force; recall rises with nprobe; scan fraction measured")


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "TEST":
        _test()
    elif cmd == "SMOKE":
        cmd_smoke()
    elif cmd == "SAMPLE":
        cmd_sample()
    elif cmd == "FULL" and len(a) == 4:
        cmd_full(a[1], a[2], a[3])
    else:
        raise SystemExit(__doc__)
