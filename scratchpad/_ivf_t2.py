"""FBX_SCALE Track C step 3c (addendum 13) -- T2: the T1 / T1R measurements on a STRATIFIED sample of the Freebase PQ64 chunks, with the re-rank depth R as a design axis.

T1 (addendum 11) and T1R (addendum 12) ran on PREFIXES of the encode chunks, which are the first names in node first-occurrence order and not a random sample of the 245,350,737 names.  Addendum 13 changes the encode ORDER (not any
chunk) so that 32 chunks, one at the midpoint of each of 32 equal strata of the 1,872 chunks, exist first (scratchpad/_fbx_enc_order.py).  T2 re-runs the same search (addendum 10's scoring, unchanged, the T1 code paths) on the
nested sets of 1, 2, 4, 8, 16, 32 sample chunks (each prefix of 2^j in the bit-reversed stratum order SAMPLE_ORDER is an even spread over the name space), against the EXACT decoded-vs-decoded top 4 for 4,000 queries, with
R in {512, 2048, 8192} (T1: R = 512 only; T1R: three R at a few cells).  Nothing here gates a partition; it measures neighbour recall, scan fraction and cost.

  python -u scratchpad/_ivf_t2.py TEST                          # synthetic: sets, cells, the ADC-rank decomposition and the readings arithmetic
  python -u scratchpad/_ivf_t2.py T2 [--threads=N]              # the measurement (resumable per stage and cell; one write-once record at the end)

Definitions (frozen by addendum 13, written before any T2 recall existed): the T1 definitions of scratchpad/_ivf_transfer.py with these changes only --
  database   = the PQ64 codes of SAMPLE_ORDER[:c] (c in {1, 2, 4, 8, 16, 32}), concatenated in that order; every chunk record must verify against the frozen contract and codebook
  cells      = L437 (nlist = round(n / 437.386)) for every c: nprobe {32, 64, 128, 256, 512} (never above nlist / 2) x R {512, 2048, 8192};  F4096 (nlist 4096) at c = 32: nprobe {32, 128, 512} x R {512, 2048, 8192}
  ADC rank   = at (c in {8, 32}, L437, nprobe in {32, 128}): for the true non-self top-3 neighbours of every query, the rank of each in the ADC order of the probed lists' best 8,192 (or 'beyond' when absent)
  codes      = the fraction of database rows whose 64-byte code is shared with another row (exact duplicates), per c (descriptive)
"""
import os
import sys

import _ivf_knn as K                                    # parses --threads=N at import; pinned by addendum 10 (not edited)

import io
import json
import time

import numpy as np

import _fbx_encode as E
import _fbx_enc_order as O
import _ivf_transfer as T                               # pinned by addendum 11 (not edited): its load / truth / coarse helpers are reused unchanged

M, DIM = K.M, K.DIM
REPO = K.REPO
WORK = os.path.join(REPO, "data", "freebase_scale", "ivf_transfer", "t2")
RECS = os.path.join(REPO, "results", "FREEBASE_SCALE", "ivf")
OUT = os.path.join(RECS, "FBX_IVF_TRANSFER_T2__v1.json")
T1_REC = os.path.join(RECS, "FBX_IVF_TRANSFER_T1__v1.json")
CHUNK_ROWS = E.CHUNK_ROWS
CS = T.CS
RS = [512, 2048, 8192]
NPROBES = T.NPROBES
F4096_C = 32
F4096_NPROBES = [32, 128, 512]
ADC_CELLS = [(8, 32), (8, 128), (32, 32), (32, 128)]
ADC_RMAX = 8192
NQ = T.NQ
Q_SEED = T.Q_SEED
log = K.log


def chunks_for(c):
    return list(O.SAMPLE_ORDER[:c])


def cells_of(c):
    """[(layout, nlist, nprobe, R)] of one size."""
    n = c * CHUNK_ROWS
    out = []
    for layout in T.LAYOUTS:
        if layout == "F4096" and c != F4096_C:
            continue
        nl = T.nlist_of(layout, n)
        pl = [p for p in (NPROBES if layout == "L437" else F4096_NPROBES) if p <= nl // 2]
        out.extend((layout, nl, p, R) for p in pl for R in RS)
    return out


def cell_path(wd, x):
    return os.path.join(wd, "cell_%s_p%d_R%d.json" % (x[0], x[2], x[3]))


def load_list(chunks, csha, cbsha):
    parts, recs = [], []
    for i in chunks:
        if not E.chunk_ok(i, csha, cbsha):
            raise SystemExit("chunk %d is not present / does not verify against the frozen contract and codebook" % i)
        cp, jp = E.chunk_paths(i)
        a = np.load(cp)
        assert a.dtype == np.uint8 and a.shape[1] == M
        parts.append(a)
        r = json.load(io.open(jp, encoding="utf-8"))
        recs.append({"chunk": int(i), "codes_sha256": r["codes_sha256"], "row_first": r["row_first"], "row_last_exclusive": r["row_last_exclusive"]})
    return np.ascontiguousarray(np.concatenate(parts)), recs


def duplicate_code_fraction(codes):
    """fraction of rows whose code string equals another row's (exact duplicates), by sorting the 64-byte rows as void scalars."""
    v = np.ascontiguousarray(codes).view(np.dtype((np.void, codes.shape[1]))).ravel()
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    return float((cnt[inv] > 1).mean())


def adc_rank_diag(idx, C, asg, codes, cb16, nrm, qrows, tn, nprobe, rmax=ADC_RMAX):
    """where the true non-self top-3 neighbours sit in the ADC order of the probed lists.
    bins: kept at R = 512 (rank < 512) | 512 .. 2047 | 2048 .. 8191 | in a probed list but rank >= rmax | not in a probed list.   Counts are over (query, true neighbour) pairs."""
    n = len(qrows)
    Qx = np.empty((n, DIM), np.float32)
    for m in range(M):
        Qx[:, m * K.SUB:(m + 1) * K.SUB] = cb16[m][codes[qrows, m]]
    idx.nprobe = int(nprobe)
    _, I = idx.search(Qx, int(rmax))
    S = Qx @ C.T                                                                    # the coarse quantiser's inner products
    top = np.argpartition(S, S.shape[1] - nprobe, axis=1)[:, S.shape[1] - nprobe:]
    bins = np.zeros(5, np.int64)
    total = 0
    for q in range(n):
        tr = [t for t in tn[q].tolist() if t != qrows[q] and t >= 0][:3]
        pos = {int(v): j for j, v in enumerate(I[q].tolist()) if v >= 0}
        probed = set(top[q].tolist())
        for t in tr:
            total += 1
            j = pos.get(int(t))
            if j is not None:
                bins[0 if j < 512 else 1 if j < 2048 else 2] += 1
            elif int(asg[t]) in probed:
                bins[3] += 1
            else:
                bins[4] += 1
    assert int(bins.sum()) == total
    names = ["rank<512 (kept at R=512)", "512<=rank<2048", "2048<=rank<8192", "probed list, rank>=%d" % rmax, "not in a probed list"]
    return {"pairs": int(total), "bins": {nm: int(b) for nm, b in zip(names, bins)}, "fractions": {nm: round(float(b) / max(total, 1), 5) for nm, b in zip(names, bins)}}


def readings(cells, t1):
    """addendum 13's pre-declared readings, computed from the cells (and the T1 record), nothing else."""
    def rec(layout, c, p, R):
        for x in cells:
            if (x["layout"], x["chunks"], x["nprobe"], x["R"]) == (layout, c, p, R):
                return x["recall3_tie_tolerant"]
        return None
    out = {}
    # P1 representativeness: strided (T2, R = 512) minus prefix (T1) at the same size and cell, L437
    t1c = {(x["layout"], x["chunks"], x["nprobe"]): x["recall3_tie_tolerant"] for x in t1["cells"]}
    d = {}
    for c in CS:
        for p in NPROBES:
            a, b = rec("L437", c, p, 512), t1c.get(("L437", c, p))
            if a is not None and b is not None:
                d["c%d_p%d" % (c, p)] = round(a - b, 4)
    big = [v for k, v in d.items() if int(k.split("_")[0][1:]) >= 8 and int(k.split("_p")[1]) in (32, 128, 512)]
    out["P1_strided_minus_prefix_R512"] = {"differences": d, "worst_abs_at_c_ge_8_nprobe_32_128_512": round(max([abs(v) for v in big] or [0.0]), 4),
                                           "label": ("AGREE" if big and max(abs(v) for v in big) <= 0.010 else "STRIDED_LOWER" if big and min(big) < -0.010 and max(big) <= 0.010 else "STRIDED_HIGHER" if big and max(big) > 0.010 and min(big) >= -0.010 else "MIXED")}
    # P2 slope per doubling over c = 4 .. 32 (least squares on log2 c), R = 512 and R = 8192
    sl = {}
    for R in (512, 8192):
        for p in NPROBES:
            ys = [(np.log2(c), rec("L437", c, p, R)) for c in (4, 8, 16, 32) if rec("L437", c, p, R) is not None]
            if len(ys) == 4:
                sl["R%d_p%d" % (R, p)] = round(float(np.polyfit([a for a, _ in ys], [b for _, b in ys], 1)[0]), 5)
    out["P2_slope_per_doubling_c4_to_32"] = {"slopes": sl, "label_R512_p128": ("FALLING" if sl.get("R512_p128", 0.0) < -0.005 else "FLAT")}
    # P3 the R curve at c = 8 and 32
    g = {}
    for c in (8, 32):
        for p in (32, 128, 512):
            a, b = rec("L437", c, p, 512), rec("L437", c, p, 8192)
            if a is not None and b is not None:
                g["c%d_p%d" % (c, p)] = round(b - a, 4)
    lab = {k: ("R_LIMITED" if v >= 0.010 else "NOT_R_LIMITED" if v < 0.003 else "PARTLY") for k, v in g.items()}
    need = {}
    for c in (8, 32):
        r8 = rec("L437", c, 128, 8192)
        if r8 is not None:
            need["c%d_p128" % c] = next((R for R in RS if rec("L437", c, 128, R) is not None and r8 - rec("L437", c, 128, R) <= 0.003), "not reached")
    out["P3_R_gain_8192_minus_512"] = {"gains": g, "labels": lab, "smallest_R_within_0.003_of_R8192": need}
    # P4 transfer with the larger R against the frozen WebQSP R = 512 baseline
    base = T.WEBQSP_BASELINE["tie_tolerant_recall3"]
    out["P4_D8192_vs_WebQSP"] = {"c32_p%d" % p: round(rec("L437", 32, p, 8192) - base[str(p)], 4) for p in (32, 64, 128, 256, 512) if rec("L437", 32, p, 8192) is not None}
    out["P4_D8192_vs_WebQSP"]["label"] = "HOLDS" if all(v >= -0.010 for k, v in out["P4_D8192_vs_WebQSP"].items() if k in ("c32_p32", "c32_p128")) else "DEGRADES"
    return out


def cmd_t2():
    import faiss
    faiss.omp_set_num_threads(K._TH)
    assert not os.path.exists(OUT), "write-once"
    assert os.path.exists(T1_REC), "T1 record needed for the paired readings"
    t0 = time.time()
    contract, csha = E.load_contract()
    P = E.train_paths()
    cbsha = json.load(io.open(P["rec"], encoding="utf-8"))["codebook_sha256"]
    assert E.sha_file(P["codebook"]) == cbsha
    cb16 = T.load_codebook()
    cells_all, costs, adc_all, dups = [], {}, {}, {}
    for c in CS:
        n = c * CHUNK_ROWS
        wd = os.path.join(WORK, "c%02d" % c)
        os.makedirs(wd, exist_ok=True)
        kp = os.path.join(wd, "costs.json")
        cc = json.load(io.open(kp, encoding="utf-8")) if os.path.exists(kp) else {}
        costs["c%d" % c] = cc
        adcp = os.path.join(wd, "adc_rank.json")
        todo = [x for x in cells_of(c) if not os.path.exists(cell_path(wd, x))]
        need_adc = [a for a in ADC_CELLS if a[0] == c] and not os.path.exists(adcp)
        if not todo and not need_adc:
            for x in cells_of(c):
                cells_all.append(json.load(io.open(cell_path(wd, x), encoding="utf-8")))
            if os.path.exists(adcp):
                adc_all["c%d" % c] = json.load(io.open(adcp, encoding="utf-8"))
            dups["c%d" % c] = cc.get("duplicate_code_fraction")
            log("c %d: every cell present" % c)
            continue
        codes, _ = load_list(chunks_for(c), csha, cbsha)
        assert len(codes) == n
        if cc.get("duplicate_code_fraction") is None:
            cc["duplicate_code_fraction"] = round(duplicate_code_fraction(codes), 6)
            T.wjson(kp, cc)
        dups["c%d" % c] = cc["duplicate_code_fraction"]
        nrm = K.all_norms(codes, cb16)
        qrows = np.sort(np.random.RandomState(Q_SEED).choice(n, size=NQ, replace=False))
        tp = os.path.join(wd, "truth.npz")
        if os.path.exists(tp):
            z = np.load(tp)
            tn, ts = z["nbr"], z["score"]
        else:
            t = time.time()
            tn, ts = T.exact_truth(codes, cb16, nrm, qrows)
            np.savez(tp + ".tmp.npz", nbr=tn, score=ts)
            os.replace(tp + ".tmp.npz", tp)
            cc["truth_seconds"] = round(time.time() - t, 1)
            T.wjson(kp, cc)
            log("c %d (n %d): exact truth for %d queries: %.0fs" % (c, n, NQ, time.time() - t))
        for layout in T.LAYOUTS:
            lc = [x for x in cells_of(c) if x[0] == layout]
            if not lc:
                continue
            nl = lc[0][1]
            pend = [x for x in lc if x in todo]
            adc_here = [a for a in ADC_CELLS if a[0] == c] if (layout == "L437" and not os.path.exists(adcp)) else []
            if not pend and not adc_here:
                for x in lc:
                    cells_all.append(json.load(io.open(cell_path(wd, x), encoding="utf-8")))
                continue
            cpath = os.path.join(wd, "coarse_%s.npy" % layout)
            C, ci = T.train_coarse_rows(codes, cb16, nl, cpath)
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
            for k_ in ("train_rows", "kmeans_seconds", "assign_seconds"):
                if cc[layout].get(k_) is None and old.get(k_) is not None:
                    cc[layout][k_] = old[k_]
            T.wjson(kp, cc)
            for x in lc:
                cp = cell_path(wd, x)
                if os.path.exists(cp):
                    cells_all.append(json.load(io.open(cp, encoding="utf-8")))
                    continue
                npb, R = x[2], x[3]
                t = time.time()
                nb, sc = K.search_rows(idx, codes, cb16, nrm, qrows, npb, R=R)
                secs = time.time() - t
                sf = K.scan_fraction(idx, codes, cb16, nrm, qrows, counts, npb)
                rec = K.recalls(qrows, tn, ts, nb, sc)
                cell = {"layout": layout, "chunks": c, "database_rows": int(n), "nlist": int(nl), "nprobe": int(npb), "R": int(R), "scan_fraction": round(sf, 6), "rows_scanned_per_query": round(sf * n, 0),
                        "queries": int(NQ), "seconds": round(secs, 1), "queries_per_second": round(NQ / secs, 1), "threads": K._TH}
                cell.update(rec)
                folds = [K.recalls(qrows[f::4], tn[f::4], ts[f::4], nb[f::4], sc[f::4])["recall3_tie_tolerant"] for f in range(4)]
                cell["tie_tolerant_folds"] = folds
                cell["tie_tolerant_fold_se"] = round(float(np.std(folds, ddof=1) / 2.0), 6)
                T.wjson(cp, cell)
                cells_all.append(cell)
                log("c %2d %-5s nlist %5d p %3d R %4d: scan %.4f  tie-tol %.4f  nearest %.4f  self %.4f  %.0f q/s" % (c, layout, nl, npb, R, sf, rec["recall3_tie_tolerant"], rec["nearest_recall"], rec["self_in_top4"], NQ / secs))
            if adc_here:
                res = {}
                for (cc_, p) in adc_here:
                    t = time.time()
                    res["p%d" % p] = adc_rank_diag(idx, C, asg, codes, cb16, nrm, qrows, tn, p)
                    res["p%d" % p]["seconds"] = round(time.time() - t, 1)
                    log("c %d L437 nprobe %d ADC rank of the true neighbours: %s" % (c, p, res["p%d" % p]["fractions"]))
                T.wjson(adcp, res)
                adc_all["c%d" % c] = res
            del idx
        del codes, nrm
    cells_all.sort(key=lambda d: (d["layout"], d["chunks"], d["nprobe"], d["R"]))
    t1 = json.load(io.open(T1_REC, encoding="utf-8"))
    chunk_recs = []
    for i in sorted(chunks_for(max(CS))):
        r = json.load(io.open(E.chunk_paths(i)[1], encoding="utf-8"))
        chunk_recs.append({"chunk": i, "codes_sha256": r["codes_sha256"], "row_first": r["row_first"], "row_last_exclusive": r["row_last_exclusive"]})
    rec = {"RECORD": "FBX_IVF_TRANSFER_T2", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "design": {"sample_order": O.SAMPLE_ORDER, "sample_by_stratum": O.SAMPLE_BY_STRATUM, "chunks": CS, "R": RS, "nprobes_L437": NPROBES, "F4096_chunks": F4096_C, "nprobes_F4096": F4096_NPROBES, "queries": NQ, "query_seed": Q_SEED,
                      "list_len": T.LIST_LEN, "fixed_nlist": T.FIXED_NLIST, "train_rows_per_list": T.TRAIN_PER_LIST, "adc_rank_cells": ADC_CELLS, "adc_rmax": ADC_RMAX},
           "webqsp_baseline": T.WEBQSP_BASELINE, "contract_sha256": csha, "codebook_sha256": cbsha, "chunk_records": chunk_recs, "cells": cells_all, "adc_rank": adc_all, "duplicate_code_fraction": dups,
           "readings": readings(cells_all, t1), "costs": costs, "seconds": round(time.time() - t0, 1), "threads": K._TH,
           "code_sha256": {"scratchpad/_ivf_t2.py": E.sha_file(os.path.abspath(__file__)), "scratchpad/_ivf_transfer.py": E.sha_file(os.path.abspath(T.__file__)), "scratchpad/_ivf_knn.py": E.sha_file(os.path.abspath(K.__file__)),
                           "scratchpad/_fbx_enc_order.py": E.sha_file(os.path.abspath(O.__file__))}}
    T.wjson(OUT, rec)
    log("T2 done %.0fs; record %s" % (time.time() - t0, OUT))


# ---------------------------------------------------------------------------------------------------------------- test
def _test():
    import faiss
    faiss.omp_set_num_threads(2)
    # the sets are nested prefixes of the sample, distinct, and spread over the whole range
    for c in CS:
        s = chunks_for(c)
        assert len(set(s)) == c and set(chunks_for(max(c // 2, 1))) <= set(s)
    assert chunks_for(2) == [29, 965] and chunks_for(4)[2:] == [497, 1433]
    s8 = sorted(chunks_for(8))
    assert s8[0] < 300 and s8[-1] > 1500 and max(np.diff(s8)) < 400
    # the cells: L437 for every size (nprobe <= nlist / 2, three R), F4096 at 32 only
    assert all(x[0] == "L437" for c in CS if c != F4096_C for x in cells_of(c))
    assert {x[0] for x in cells_of(F4096_C)} == {"L437", "F4096"} and len({x[3] for x in cells_of(1)}) == 3
    assert all(x[2] <= x[1] // 2 for c in CS for x in cells_of(c))
    # duplicate fraction, ADC-rank decomposition on a synthetic clustered set
    rs = np.random.RandomState(0)
    n = 3000
    cb16 = rs.randn(M, 256, K.SUB).astype(np.float16).astype(np.float32)
    base = rs.randint(0, 256, size=(40, M)).astype(np.uint8)
    codes = base[rs.randint(0, 40, size=n)].copy()
    flip = rs.rand(n, M) < 0.08
    codes[flip] = rs.randint(0, 256, size=int(flip.sum())).astype(np.uint8)
    d = np.concatenate([codes, codes[:30]])
    assert abs(duplicate_code_fraction(d) - duplicate_code_fraction(codes)) < 0.02 and duplicate_code_fraction(np.unique(codes, axis=0)) == 0.0
    nrm = K.all_norms(codes, cb16)
    qrows = np.sort(rs.choice(n, size=200, replace=False))
    tn, ts = T.exact_truth(codes, cb16, nrm, qrows, blk=700)
    X, nr = K.decode(codes, cb16, 0, n)
    X /= nr[:, None]
    nl = 24
    km = faiss.Kmeans(DIM, nl, niter=5, verbose=False, spherical=True, seed=1)
    km.train((X[:1500]).astype(np.float32))
    C = np.ascontiguousarray(km.centroids.astype(np.float32))
    q = faiss.IndexFlatIP(DIM)
    q.add(C)
    asg = q.search(np.ascontiguousarray(X.astype(np.float32)), 1)[1][:, 0].astype(np.int32)
    idx, counts = K.build_index(codes, cb16, C, asg)
    full = adc_rank_diag(idx, C, asg, codes, cb16, nrm, qrows, tn, nl, rmax=n)       # every list probed, rmax = n: no pair can be unprobed or beyond
    assert full["bins"]["not in a probed list"] == 0 and full["bins"]["probed list, rank>=%d" % n] == 0 and full["pairs"] > 0, full
    part = adc_rank_diag(idx, C, asg, codes, cb16, nrm, qrows, tn, 1, rmax=1000)             # one list probed: the 'not in a probed list' bin is checked against an independent count
    assert sum(part["bins"].values()) == part["pairs"], part
    Qx = np.empty((len(qrows), DIM), np.float32)
    for m in range(M):
        Qx[:, m * K.SUB:(m + 1) * K.SUB] = cb16[m][codes[qrows, m]]
    top1 = (Qx @ C.T).argmax(1)
    exp = sum(1 for q in range(len(qrows)) for t in [t for t in tn[q].tolist() if t != qrows[q] and t >= 0][:3] if int(asg[t]) != int(top1[q]))
    assert part["bins"]["not in a probed list"] == exp, (part, exp)
    # R as a design axis moves the result only through the re-rank depth: recall is non-decreasing in R at a fixed probe count
    r = [K.recalls(qrows, tn, ts, *K.search_rows(idx, codes, cb16, nrm, qrows, 3, R=R))["recall3_tie_tolerant"] for R in (8, 64, 512)]
    assert r[0] <= r[1] + 1e-9 <= r[2] + 2e-9, r
    # the readings arithmetic on a fabricated record
    fab = []
    for c in CS:
        for p in NPROBES:
            for R in RS:
                fab.append({"layout": "L437", "chunks": c, "nprobe": p, "R": R, "recall3_tie_tolerant": 0.9 + 0.01 * (R == 2048) + 0.02 * (R == 8192) - 0.01 * np.log2(c) / 5})
    t1 = {"cells": [{"layout": "L437", "chunks": c, "nprobe": p, "recall3_tie_tolerant": 0.9 - 0.01 * np.log2(c) / 5} for c in CS for p in NPROBES]}
    out = readings(fab, t1)
    assert out["P1_strided_minus_prefix_R512"]["label"] == "AGREE" and abs(out["P3_R_gain_8192_minus_512"]["gains"]["c32_p128"] - 0.02) < 1e-9 and out["P3_R_gain_8192_minus_512"]["labels"]["c32_p128"] == "R_LIMITED"
    print("TEST ok: sets", chunks_for(8), "| cells at 32:", len(cells_of(32)), "| adc", full["fractions"], "| recall vs R", [round(v, 4) for v in r])


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a == ["TEST"]:
        _test()
    elif a == ["T2"]:
        cmd_t2()
    else:
        raise SystemExit(__doc__)
