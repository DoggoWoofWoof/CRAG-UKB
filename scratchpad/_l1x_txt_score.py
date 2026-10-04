"""L1X -- the 90+ scoreboard for HotpotQA and 2Wiki: the shipped L1 (IR_L1, ES router, KNEE_SDIV count with alpha .75) on the development population of TXT_DEV_POPULATION__v1.json.  (development only)
(User 2026-10-02: "try to get 90+ for all datasets".  These two text datasets had no development measurement of the routed L1 at node budgets.)

Per row: FLAT = RRF(dense, SPLADE) full order; hits = top-200 FLAT, weight 1/(rank+1) * g; LOC = one-hop localisation over STRUCT_out u STRUCT_in u KNN u NER; served order O = RRF(FLAT, LOC), K0 = 60;
router: ES order = partitions by first appearance in O' = RRF(H_q, LOC) over V_q = hits u LOC; count B_P(q,K) = min(K, ceil(B_100(q) (K/100)^0.75)), B_100 = KNEE_SDIV on the K = 100 hit curve (the shipped rule).
served = the first B_N nodes of O inside the contacted partitions; ALL = every gold served.  Exact arithmetic of _l1x_actserve.py (identity-checked there against the stored MetaQA records); no pool cap: the served rank of a gold is its rank
among ALL contacted nodes sorted by (-f, FLAT rank).  Also reported: UNROUTED (rank of the gold in O) and FLAT alone, for the same B_N.   Partition maps: results/L1_HOST/parts/<ds>__H4_SK_k<K>__PHG_con.npy (sha verified against their RUN.json).

  python -u scratchpad/_l1x_txt_score.py RUN <hotpotqa|2wiki> <tag> [--rows=N] [--cq=C]      -> results/L1_X/txtscore_<ds>__<tag>.{json,npz}   (write-once; --rows = smoke, writes nothing)
  python -u scratchpad/_l1x_txt_score.py SHOW <ds> <tag>
"""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_edgediag as E
import _l1d_route as RT
import _l1d_kscale as KS
import _l1x_feat as FE

log = D.log
K0 = D.K0
ACT = D.ACT
OUT = os.path.join(D.REPO, "results", "L1_X")
POPF = os.path.join(OUT, "TXT_DEV_POPULATION__v1.json")
PARTS = os.path.join(D.REPO, "results", "L1_HOST", "parts")
MC = D.M_CURVE
NM = len(MC)
CELLS = (100, 250, 500, 1000, 2000)
K_REF = 100
ALPHA = "0.75"


class TPop(object):
    def __init__(self, cd, rows_limit):
        rec = json.load(open(POPF, encoding="utf-8"))
        pp = rec["populations"][cd.name]
        rows = np.asarray(pp["rows"], np.int64)
        assert [cd.query_ids[int(r)] for r in rows] == pp["query_ids"] and D.sha_text(",".join(pp["query_ids"])) == pp["query_ids_sha256"]
        assert pp["dataset_pins"] == {"DATASET_json_RECORD_SHA256": cd.record_sha, "query_index.npz": D.sha_file(cd._query_index_path())}
        assert cd.split_ranges["train"][0] <= rows.min() and rows.max() < cd.split_ranges["train"][1], "only TRAIN rows are development rows here"
        if rows_limit is not None:
            rows = rows[:rows_limit]
        self.rows, self.nq = rows, len(rows)
        D.qindex(cd)
        self.golds = [np.asarray(g, np.int64) for g in cd.gold(rows)]
        self.ngold = np.array([len(g) for g in self.golds], np.int64)
        self.record = {"file": D.rel(POPF), "sha256": D.sha_file(POPF), "n": self.nq}


def load_map(ds, K, N):
    rj = os.path.join(PARTS, "%s__H4_SK_k%d__PHG_con.RUN.json" % (ds, K))
    R = json.load(open(rj, encoding="utf-8"))
    p = os.path.join(D.REPO, R["output"]["file"])
    assert D.sha_file(p) == R["output"]["sha256"], "PHG map %s K %d differs from its RUN.json" % (ds, K)
    assert R["STATUS"] == "OK" and R["balance"]["every_block_used"], (ds, K)
    h = np.load(p).astype(np.int64)
    assert len(h) == N and int(h.max()) + 1 == K, (K, int(h.max()) + 1)
    return h


def mem_gb():
    """(working set, private bytes) of this process in GB -- Windows only; the working set includes reclaimable file-mapped pages, private bytes is the real demand."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]
        pm = PMC()
        pm.cb = ctypes.sizeof(PMC)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
        return round(pm.WorkingSetSize / 1e9, 2), round(pm.PrivateUsage / 1e9, 2), round(pm.PeakWorkingSetSize / 1e9, 2)
    except Exception:
        return None


def served_rank(cm_nodes, f, frank, g):
    """rank (0-based) of each gold among the contacted nodes sorted by (-f, FLAT rank); -1 if a gold is not contacted."""
    S = cm_nodes[np.lexsort((frank[cm_nodes], -f[cm_nodes]))]
    srt = np.argsort(S, kind="stable")
    ix = np.searchsorted(S[srt], g)
    ok = (ix < len(S)) & (S[srt][np.minimum(ix, len(S) - 1)] == g)
    out = np.where(ok, srt[np.minimum(ix, len(S) - 1)], -1)
    return out


def run():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 3 and a[0] == "RUN" and a[1] in ("hotpotqa", "2wiki", "metaqa"), __doc__
    ds, tag = a[1], a[2]
    rows_limit = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--rows=")), None)
    cq = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--cq=")), None)
    if cq:
        D.CQ = cq
    fo = os.path.join(OUT, "txtscore_%s__%s" % (ds, tag))
    assert rows_limit is not None or not os.path.exists(fo + ".json"), "write-once: %s exists" % fo
    t_all = time.time()
    me = os.path.abspath(__file__)
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    global CELLS
    st = None
    if ds == "metaqa":                                                  # SELF-TEST: the same arithmetic must reproduce the stored MetaQA verdicts (shipped routed ALL) on every (row, B_N)
        import _l1x_rrt as R
        pop = D.Population(cd, rows_limit)
        nq = pop.nq
        CELLS = R.CELLS["metaqa"]
        maps = R.load_maps(cd, ds)
        st = np.load(os.path.join(D.OUT, "kscale_metaqa__v1.npz"))
        assert (st["rows"][:nq] == pop.rows).all()
    else:
        pop = TPop(cd, rows_limit)
        nq = pop.nq
        maps = {K: load_map(ds, K, N) for K in sorted(set(CELLS) | {K_REF})}
    F, _ = FE.families(cd, N)
    tabs = {K: KS.mult_tables(K)[0][ALPHA] for K in CELLS}
    log("L1X txtscore %s %s: N %d, %d rows, cells %s, CQ %s" % (ds, tag, N, nq, CELLS, D.CQ))
    log("  memory after setup (working set, private, peak working set; GB): %s" % (mem_gb(),))
    Qu = D.unit_queries(cd, pop.rows)
    ALL = {K: np.zeros((nq, NM), bool) for K in CELLS}
    UNR = np.zeros((nq, NM), bool)
    FLT = np.zeros((nq, NM), bool)
    LOST = {K: np.zeros(nq, bool) for K in CELLS}        # a gold lies outside the contacted partitions
    BPQ = {K: np.zeros(nq, np.int32) for K in CELLS}
    NLOC = np.zeros(nq, np.int64)
    MC_ = np.array(MC)
    H100 = maps[K_REF]
    wseed = 1.0 / np.arange(1.0, ACT + 1.0)
    # resumable (the host is shared and preempts): after every product batch the finished rows are checkpointed; a restart skips them (arithmetic per row is independent of the batching)
    ckf = os.path.join(D.REPO, "data", "_cache", "txtscore_%s__%s.partial.npz" % (ds, tag))
    me_sha = D.sha_file(me)
    done = 0
    if rows_limit is None and os.path.exists(ckf):
        with np.load(ckf) as z:                                             # closed before the next checkpoint replaces the file (Windows refuses to replace an open file)
            assert (z["rows"] == pop.rows).all() and str(z["code_sha"]) == me_sha and int(z["CQ"]) == int(D.CQ), "stale checkpoint %s" % ckf
            done = int(z["done"])
            for K in CELLS:
                ALL[K], LOST[K], BPQ[K] = z["ALL__%d" % K].copy(), z["LOST__%d" % K].copy(), z["BPQ__%d" % K].copy()
            UNR, FLT, NLOC = z["UNR"].copy(), z["FLT"].copy(), z["NLOC"].copy()
        log("RESUME from %s: %d / %d rows already done" % (ckf, done, nq))
    t0 = time.time()
    for j0, j1, SD, SS, sec, _ in (D.batches(cd, pop.rows[done:], Qu[done:], N) if done < nq else ()):
        j0, j1 = j0 + done, j1 + done
        for i in range(j1 - j0):
            j = j0 + i
            of, od, os_, npos, fv, frank = D.flat_row(SD[i], SS[i])
            g = pop.golds[j]
            top = of[:ACT]
            nh = len(top)
            U, H, GW = [], [], []
            for f_ in FE.FAMS:
                Fm = F[f_]
                s_ = Fm["xadj"][top]
                hit, pos = E.entries(s_, Fm["xadj"][top + 1] - s_)
                U.append(Fm["adj"][pos].astype(np.int64))
                H.append(hit)
                GW.append(Fm["g"][top][hit])
            u, hit, gw = np.concatenate(U), np.concatenate(H), np.concatenate(GW)
            x = wseed[hit] * gw
            L = np.bincount(u, weights=x, minlength=N)
            Lord = D.order_from_score(L, frank)
            NLOC[j] = len(Lord)
            f = 1.0 / (K0 + frank.astype(np.float64))
            f[Lord] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
            # ---- unrouted: rank of each gold in O (sorted by (-f, frank)) and FLAT alone (frank)
            FLT[j] = frank[g].max() < MC_
            fg = f[g]
            cand = np.flatnonzero(f >= fg.min())                    # every node ahead of the worst gold, hence the exact O rank of each gold
            UNR[j] = served_rank(cand, f, frank, g).max() < MC_
            # ---- router: O' = RRF(H_q, LOC) over V_q
            vq = np.union1d(top, Lord)
            fq = np.zeros(len(vq))
            tq = np.zeros(len(vq))
            it = np.searchsorted(vq, top)
            fq[it] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
            il = np.searchsorted(vq, Lord)
            fq[il] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
            tq[il] = ACT + np.arange(len(Lord))
            tq[it] = np.arange(nh)
            oq = vq[np.lexsort((tq, -fq))]
            Cm = np.bincount(hit * K_REF + H100[u], weights=x, minlength=nh * K_REF).reshape(nh, K_REF)
            kn_, npz_ = RT.knee(wseed[:nh] @ (Cm > 0))
            b0 = kn_ if npz_ else K_REF
            for K in CELLS:
                hard = maps[K]
                b = K if not npz_ else int(tabs[K][b0])
                BPQ[K][j] = b
                hq = hard[oq]
                up, first = np.unique(hq, return_index=True)
                fe = np.full(K, len(oq), np.int64)
                fe[up] = first
                ro = np.lexsort((np.arange(K), fe))
                cm = np.zeros(K, bool)
                cm[ro[:b]] = True
                if not cm[hard[g]].all():
                    LOST[K][j] = True
                    continue
                cn = cand[cm[hard[cand]]]                           # contacted nodes with f >= the worst gold's f: a prefix of the sorted contacted list, so every gold's position is unchanged
                rk = served_rank(cn, f, frank, g)
                assert (rk >= 0).all()
                ALL[K][j] = rk.max() < MC_
        log("  rows %d / %d (%.0fs, %.2fs per row)  mem(ws,private,peak ws) %s" % (j1, nq, time.time() - t0, (time.time() - t0) / max(1, j1 - done), mem_gb()))
        if rows_limit is None:
            os.makedirs(os.path.dirname(ckf), exist_ok=True)
            ck = {"rows": pop.rows, "done": np.int64(j1), "code_sha": np.array(me_sha), "CQ": np.int64(D.CQ), "UNR": UNR, "FLT": FLT, "NLOC": NLOC}
            for K in CELLS:
                ck["ALL__%d" % K], ck["LOST__%d" % K], ck["BPQ__%d" % K] = ALL[K], LOST[K], BPQ[K]
            np.savez(ckf + ".tmp.npz", **ck)
            os.replace(ckf + ".tmp.npz", ckf)
    ng = pop.ngold
    if st is not None:
        for K in CELLS:
            bp = st["CNTA__PHG_k%d" % K][:nq, 1]
            sv = np.stack([(st["LOES__PHG_k%d" % K][:nq] <= bp) & (bp <= st["HIES__PHG_k%d" % K][:nq, mi]) for mi in range(NM)], axis=1)
            assert (BPQ[K] == bp).all() and (sv == ALL[K]).all(), "SELF-TEST FAILED at K %d: %d cells differ" % (K, int((sv != ALL[K]).sum()))
        log("SELF-TEST PASS: shipped routed ALL == the stored MetaQA verdict on every (row, B_N) for K %s (%d rows)" % (CELLS, nq))
        return
    res = {"mode": "L1X_TXT_SCOREBOARD (development; descriptive; nothing chosen)", "definitions": __doc__, "dataset": ds, "tag": tag, "N": N, "n_rows": nq, "population": pop.record,
           "cells": [str(K) for K in CELLS], "budgets": list(MC), "mean_B_P": {str(K): round(float(BPQ[K].mean()), 2) for K in CELLS}, "mean_LOC_size": round(float(NLOC.mean()), 1),
           "ALL": {"unrouted_O": [round(float(UNR[:, m].mean()), 4) for m in range(NM)], "flat_only": [round(float(FLT[:, m].mean()), 4) for m in range(NM)]},
           "routed_ALL": {str(K): [round(float(ALL[K][:, m].mean()), 4) for m in range(NM)] for K in CELLS},
           "routed_gold_partition_lost_share": {str(K): round(float(LOST[K].mean()), 4) for K in CELLS}, "by_n_gold": {}}
    for k_ in sorted(set(ng.tolist())):
        m_ = ng == k_
        res["by_n_gold"][str(k_)] = {"n": int(m_.sum()), "unrouted_O": [round(float(UNR[m_, m].mean()), 4) for m in range(NM)], "routed_K500": [round(float(ALL[500][m_, m].mean()), 4) for m in range(NM)]}
    res["seconds"] = round(time.time() - t_all, 1)
    res["code"] = {"path": D.rel(me), "sha256": D.sha_file(me)}
    res["pinned"] = D.PINNED
    if rows_limit is not None:
        log("SMOKE (--rows): not writing; unrouted_O %s | routed K500 %s | lost K500 %.3f" % (res["ALL"]["unrouted_O"], res["routed_ALL"]["500"], res["routed_gold_partition_lost_share"]["500"]))
        return
    arrays = {"rows": pop.rows, "ngold": ng, "UNR": np.packbits(UNR, axis=1), "FLT": np.packbits(FLT, axis=1)}
    for K in CELLS:
        arrays["ALL__%d" % K] = np.packbits(ALL[K], axis=1)
        arrays["LOST__%d" % K] = LOST[K]
        arrays["BPQ__%d" % K] = BPQ[K]
    np.savez_compressed(fo + ".npz", **arrays)
    res["npz"] = {"path": os.path.basename(fo + ".npz"), "sha256": D.sha_file(fo + ".npz")}
    D.G.S.wj(fo + ".json", res)
    log("done (%.0fs) -> %s" % (time.time() - t_all, fo))


def show():
    ds, tag = sys.argv[2], sys.argv[3]
    r = json.load(open(os.path.join(OUT, "txtscore_%s__%s.json" % (ds, tag)), encoding="utf-8"))
    print("%s: %d dev rows, B_N %s, mean LOC size %.0f" % (ds, r["n_rows"], r["budgets"], r["mean_LOC_size"]))
    print("  FLAT only             %s" % " ".join("%.3f" % v for v in r["ALL"]["flat_only"]))
    print("  unrouted O (shipped)  %s" % " ".join("%.3f" % v for v in r["ALL"]["unrouted_O"]))
    for K in r["cells"]:
        print("  routed K %-5s B_P %6s %s | gold partition lost %.3f" % (K, r["mean_B_P"][K], " ".join("%.3f" % v for v in r["routed_ALL"][K]), r["routed_gold_partition_lost_share"][K]))
    for k, v in r["by_n_gold"].items():
        print("  n_gold %s (n=%d): unrouted %s | K500 %s" % (k, v["n"], " ".join("%.3f" % t for t in v["unrouted_O"]), " ".join("%.3f" % t for t in v["routed_K500"])))


if __name__ == "__main__":
    if sys.argv[1] == "RUN":
        run()
    elif sys.argv[1] == "SHOW":
        show()
    else:
        raise SystemExit(__doc__)
