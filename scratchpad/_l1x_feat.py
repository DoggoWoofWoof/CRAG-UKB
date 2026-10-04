"""L1X -- SELECT lane, step 1: the candidate FEATURE DUMP of the carried-forward node-level L1 (development only; gold-aware diagnostic, never a
candidate).  (User ruling 2026-10-01: "l1 should only be parametric free routing and selection it cant have traversals" -- so the lane may
only re-ORDER what the one static application of A^T already supplies: no walk, no second hop, no fitted or learned weight in any L1 rule.)

What it records.  For every population row the L1a evidence of the shipped node-level L1 is recomputed with the arithmetic of
_l1d_node1h.py / _l1d_scale.py (hits H_q = FLAT_RRF[:200]; IR_L1 = one static one-hop localisation over STRUCT_out u STRUCT_in u KNN u NER with
hit weight 1/(FLAT rank + 1) and g = 1/log2(1 + deg^+(s)); the served order O = RRF(FLAT, LOC), K0 = 60) and the POOL P_q = O[:POOL] (POOL =
5000 = the largest node budget B_N) is written with per-candidate features and the gold flag:
    0 fpos      position in O (0 .. POOL-1)                      10 Ssum    the IR_L1 score L(u)
    1 frank     FLAT rank (capped 2**24)                         11 nseed   distinct supporting hits (top-200)
    2 drank     dense rank                                       12 nfam    supporting families (0 .. 4)
    3 srank     SPLADE rank (npos if no overlap)                 13 minhit  best supporting hit rank (200 if none)
    4 dsc       dense score                                      14 maxc    largest single contribution w(s) g(s)
    5 ssc       SPLADE score                                     15 incs    incident edges of u summed over the families
    6-9 S_f     per-family IR_L1 score (S_out,S_in,KNN,NER)      16 lrank   rank in the LOC order (-1: unscored)
                                                                 17 top10   the part of L(u) due to the top-10 hits
    18 gold     1 if u is a gold node of the query
and per query: dmax, smax (largest dense / SPLADE score), nsupp (|LOC order|), ngold, and the gold nodes' positions in O (pos_O, capped).
Nothing here selects, tunes or confirms a rule: it is the input of the offline rule search (_l1x_rules.py) and of the information-ceiling
diagnostic.  Development rows only (the L1_DEV population; WebQSP split A); no sealed split-B row, no TEST row is read.

Identity checks (CHECK mode / always on the first rows when a stored record exists): the baseline positions of the gold nodes equal
results/L1_DEV/kscale_<ds>__v1.npz  pos_FLATLOC__IR_L1  (clipped at POOL) on every gold node.

Usage: python -u scratchpad/_l1x_feat.py RUN <dataset> <tag> [--rows=K] [--out=<dir>]   -> <out or results/L1_X>/feat_<ds>__<tag>.{json,npz} (write-once)
"""
import io
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_edgediag as E
import _l1d_node1h as NH

log = D.log
K0, ACT = D.K0, D.ACT
FAMS = NH.FAMS
POOL = 5000
OUT = os.path.join(D.REPO, "results", "L1_X")
FEAT_NAMES = ["fpos", "frank", "drank", "srank", "dsc", "ssc", "S_out", "S_in", "S_knn", "S_ner", "Ssum", "nseed", "nfam", "minhit", "maxc",
              "incs", "lrank", "top10", "gold"]
NF = len(FEAT_NAMES)
CAP = 1 << 24


class WPop(object):
    """the development population of a dataset without a hop field (WebQSP split A): rows, gold nodes (flattened with pointers)."""

    def __init__(self, cd, rows):
        self.rows = np.asarray(rows, np.int64)
        self.nq = len(self.rows)
        self.qids = [cd.query_ids[int(r)] for r in self.rows]
        self.golds = [np.asarray(g, np.int64) for g in cd.gold(self.rows)]
        self.ngold = np.array([len(g) for g in self.golds], np.int64)
        assert (self.ngold >= 1).all()
        self.gptr = np.zeros(self.nq + 1, np.int64)
        self.gptr[1:] = np.cumsum(self.ngold)
        self.ng_tot = int(self.gptr[-1])
        self.hops = None
        self.ST = {}
        self.record = {}


def population(cd, rows_limit):
    if cd.name in ("metaqa", "musique", "squad"):
        return D.Population(cd, rows_limit)
    assert cd.name == "webqsp", cd.name
    rec = json.load(io.open(os.path.join(D.REPO, "results", "L3_DEV", "l3w_population_webqsp__v1.json"), encoding="utf-8"))
    rows = np.asarray(rec["rows"], np.int64)
    assert [cd.query_ids[int(r)] for r in rows] == rec["query_ids"]
    if rows_limit is not None:
        rows = rows[:rows_limit]
    p = WPop(cd, rows)
    p.record = {"file": "results/L3_DEV/l3w_population_webqsp__v1.json", "sha256": D.sha_file(os.path.join(D.REPO, "results", "L3_DEV", "l3w_population_webqsp__v1.json")),
                "n": p.nq}
    return p


def families(cd, N):
    """the four families of E with the row length (out-degree) and incident degree of every node."""
    F, rec = NH.build_families(cd, N)
    for f in FAMS:
        Fm = F[f]
        Fm["row"] = np.diff(Fm["xadj"]).astype(np.int64)
        h = Fm["h"]
        inc = np.zeros(N, np.int64)
        m = h > 0
        x = 2.0 ** (1.0 / h[m]) - 1.0
        assert np.abs(x - np.rint(x)).max() < 1e-6
        inc[m] = np.rint(x).astype(np.int64)
        Fm["inc"] = inc
    return F, rec


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 3 and a[0] == "RUN", __doc__
    ds, tag = a[1], a[2]
    rows_limit = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--rows=")), None)
    outdir = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--out=")), OUT)
    os.makedirs(outdir, exist_ok=True)
    fo = os.path.join(outdir, "feat_%s__%s" % (ds, tag))
    assert not os.path.exists(fo + ".json"), "write-once: %s exists" % fo
    t_all = time.time()
    me = os.path.abspath(__file__)
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    pop = population(cd, rows_limit)
    nq, gptr = pop.nq, pop.gptr
    F, famrec = families(cd, N)
    log("L1X feat %s %s: N %d, %d rows, %d gold nodes" % (ds, tag, N, nq, pop.ng_tot))
    Qu = D.unit_queries(cd, pop.rows)
    stored = None
    fs = os.path.join(D.OUT, "kscale_%s__v1.npz" % ds)
    if os.path.exists(fs):
        stored = np.load(fs)
        assert (stored["rows"][:nq] == pop.rows).all()
    Q, FEAT = [], []
    QINFO = np.zeros((nq, 4), np.float64)                      # dmax, smax, nsupp, ngold
    POS_O = np.zeros(pop.ng_tot, np.int64)
    POS_FLAT = np.zeros(pop.ng_tot, np.int64)
    t0 = time.time()
    for j0, j1, SD, SS, sec, _ in D.batches(cd, pop.rows, Qu, N):
        for i in range(j1 - j0):
            j = j0 + i
            of, od, os_, npos, fv, frank = D.flat_row(SD[i], SS[i])
            g = pop.golds[j]
            sl = slice(gptr[j], gptr[j + 1])
            POS_FLAT[sl] = frank[g]
            top = of[:ACT]
            nh = len(top)
            U, H, GW, FAMI, INCU = [], [], [], [], []
            for fi, f in enumerate(FAMS):
                Fm = F[f]
                st = Fm["xadj"][top]
                hit, pos = E.entries(st, Fm["xadj"][top + 1] - st)
                u = Fm["adj"][pos].astype(np.int64)
                U.append(u)
                H.append(hit)
                GW.append(Fm["g"][top][hit])
                FAMI.append(np.full(len(u), fi, np.int8))
                INCU.append(Fm["inc"][u])
            u, hit, gw, fam = np.concatenate(U), np.concatenate(H), np.concatenate(GW), np.concatenate(FAMI)
            x = (1.0 / np.arange(1.0, nh + 1.0))[hit] * gw
            # ---- baseline IR_L1 (identical arithmetic to _l1d_scale.py) and the served order O restricted to the pool
            L = np.bincount(u, weights=x, minlength=N)
            Lord = D.order_from_score(L, frank)
            nL = len(Lord)
            lrank = np.full(N, -1, np.int64)
            lrank[Lord] = np.arange(nL)
            cand = np.union1d(Lord, of[:POOL])
            fc = 1.0 / (K0 + frank[cand].astype(np.float64))
            lc = lrank[cand]
            fc = fc + np.where(lc >= 0, 1.0 / (K0 + np.maximum(lc, 0).astype(np.float64)), 0.0)
            order = np.lexsort((frank[cand], -fc))
            pool = cand[order[:POOL]]
            # gold positions in O (exact below POOL)
            fg = 1.0 / (K0 + frank[g].astype(np.float64)) + np.where(lrank[g] >= 0, 1.0 / (K0 + np.maximum(lrank[g], 0).astype(np.float64)), 0.0)
            pg = np.empty(len(g), np.int64)
            for k in range(len(g)):
                pg[k] = int((fc > fg[k]).sum() + ((fc == fg[k]) & (frank[cand] < frank[g][k])).sum())
            pg = np.where(np.isin(g, cand), pg, POOL)
            POS_O[sl] = np.minimum(pg, POOL)
            # ---- features of the pool
            m = len(pool)
            ft = np.zeros((m, NF), np.float32)
            ft[:, 0] = np.arange(m)
            ft[:, 1] = np.minimum(frank[pool], CAP)
            pd = np.empty(N, np.int64)
            pd[od] = np.arange(N)
            ps = np.full(N, npos, np.int64)
            ps[os_] = np.arange(npos)
            ft[:, 2] = pd[pool]
            ft[:, 3] = ps[pool]
            ft[:, 4] = SD[i][pool]
            ft[:, 5] = SS[i][pool]
            for fi in range(len(FAMS)):
                mk = fam == fi
                ft[:, 6 + fi] = np.bincount(u[mk], weights=x[mk], minlength=N)[pool]
            ft[:, 10] = L[pool]
            key = u * ACT + hit                                   # distinct (node, hit) pairs
            ku = np.unique(key)
            ns = np.bincount(ku // ACT, minlength=N)
            ft[:, 11] = ns[pool]
            kf = np.unique(u * 4 + fam.astype(np.int64))
            nfm = np.bincount(kf // 4, minlength=N)
            ft[:, 12] = nfm[pool]
            mh = np.full(N, ACT, np.int64)
            np.minimum.at(mh, u, hit)
            ft[:, 13] = mh[pool]
            mx = np.zeros(N)
            np.maximum.at(mx, u, x)
            ft[:, 14] = mx[pool]
            inc = np.zeros(N, np.int64)
            for f in FAMS:
                inc += F[f]["inc"]
            ft[:, 15] = inc[pool]
            ft[:, 16] = lrank[pool]
            t10 = hit < 10
            ft[:, 17] = np.bincount(u[t10], weights=x[t10], minlength=N)[pool]
            ft[:, 18] = np.isin(pool, g)
            Q.append(np.full(m, j, np.int32))
            FEAT.append(ft)
            QINFO[j] = [float(SD[i].max()), float(SS[i].max()), nL, len(g)]
            if stored is not None and j < 4:
                st_ = np.minimum(stored["pos_FLATLOC__IR_L1"][sl], POOL)
                assert (st_ == POS_O[sl]).all(), "baseline positions differ from the kscale record (row %d)" % j
        log("  rows %d / %d (%.0fs)" % (j1, nq, time.time() - t0))
    Qa, Fa = np.concatenate(Q), np.concatenate(FEAT)
    if stored is not None:
        ng = pop.ng_tot
        assert (np.minimum(stored["pos_FLATLOC__IR_L1"][:ng], POOL) == POS_O).all(), "baseline positions differ from the kscale record"
        assert (stored["pos_FLAT"][:ng] == POS_FLAT).all()
        chk = "baseline O positions (clipped at %d) == %s on all %d gold nodes" % (POOL, D.rel(fs), ng)
    else:
        chk = "no stored kscale record (WebQSP): FLAT positions only"
    log("identity: " + chk)
    # the baseline ALL@M from the dump
    ngold = pop.ngold
    res = {"mode": "L1X_SELECT_FEATURE_DUMP (development; gold-aware diagnostic, never a candidate)", "definitions": __doc__, "dataset": ds, "tag": tag,
           "N": N, "n_rows": nq, "n_gold_nodes": pop.ng_tot, "pool": POOL, "features": FEAT_NAMES, "population": pop.record, "identity_check": chk,
           "baseline_ALL": {}, "families": famrec, "seconds": round(time.time() - t_all, 1),
           "code": {"path": D.rel(me), "sha256": D.sha_file(me)}, "pinned": D.PINNED, "constants": D.CONSTANTS}
    for M in D.M_CURVE:
        a_ = D.per_query(POS_O, M, gptr, ngold)[0]
        res["baseline_ALL"][str(M)] = {"ALL": D.q4(a_.mean()), "n_ALL": int(a_.sum())}
    log("baseline IR_L1 ALL " + " ".join("%d:%.4f" % (M, res["baseline_ALL"][str(M)]["ALL"]) for M in D.M_CURVE))
    np.savez_compressed(fo + ".npz", q=Qa, feat=Fa, qinfo=QINFO, rows=pop.rows, gptr=gptr, pos_O=POS_O, pos_flat=POS_FLAT,
                        hops=(pop.hops if pop.hops is not None else np.zeros(nq, np.int64)))
    res["npz"] = {"path": os.path.basename(fo + ".npz"), "sha256": D.sha_file(fo + ".npz"), "candidates": int(len(Qa))}
    D.G.S.wj(fo + ".json", res)
    log("done (%.0fs) -> %s" % (time.time() - t_all, fo))


if __name__ == "__main__":
    main()
