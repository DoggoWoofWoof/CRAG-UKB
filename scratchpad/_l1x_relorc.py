"""L1X -- SELECT lane, step 4: the RELATION-ORACLE ceiling of a one-hop relation-conditioned weight (development only; gold-aware DIAGNOSTIC, never a candidate).
(User ruling 2026-10-01: L1 = parameter-free routing and SELECTION only -- no traversal.  A relation-conditioned weight on the single edge from a hit to a
candidate is still ONE application of A^T; this script asks only how much such a weight could ever add, by giving it the right relations for free.)

For every population row of _l1x_feat.py (same hits H_q = FLAT_RRF[:200], same weights x = 1/(hit+1) g(s), same pool P_q = O[:5000], asserted equal to the
feature dump through the fpos / frank columns) it re-scores the pool nodes with the STRUCT_out edges KEPT PER RELATION (the shipped CSR keeps one edge per
node pair):
    Sdup   sum over (hit, relation) STRUCT_out edges hit -> u of x                          (the shipped S_out up to pairs carried by several relations)
    Sorc   the same sum restricted to the relations R*_q = {relation of an edge hit -> gold node, over the hits and all gold nodes of the row}  (ORACLE)
    Nrel   the number of distinct relations on edges hit -> u (a label-free relation-diversity count)
The oracle knows the gold nodes: it bounds a perfect question-relation matcher on the one-hop evidence; it is not a rule.

Usage: python -u scratchpad/_l1x_relorc.py RUN <dataset> <tag> [--rows=K]  -> results/L1_X/relorc_<ds>__<tag>.{json,npz} (write-once)
"""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_edgediag as E
import _l1x_feat as FE

log = D.log
K0, ACT, POOL = D.K0, D.ACT, FE.POOL
OUT = os.path.join(D.REPO, "results", "L1_X")


def rel_csr(cd, N):
    s, d, r, ent = cd.family("structural")
    s = np.asarray(s, np.int64)
    d = np.asarray(d, np.int64)
    r = np.asarray(r, np.int64)
    o = np.argsort(s, kind="stable")
    xadj = np.zeros(N + 1, np.int64)
    xadj[1:] = np.cumsum(np.bincount(s, minlength=N))
    return xadj, d[o].astype(np.int32), r[o].astype(np.int32), len(ent.get("relation_vocabulary") or [])


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 3 and a[0] == "RUN", __doc__
    ds, tag = a[1], a[2]
    rows_limit = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--rows=")), None)
    fo = os.path.join(OUT, "relorc_%s__%s" % (ds, tag))
    assert not os.path.exists(fo + ".json"), "write-once: %s exists" % fo
    t_all = time.time()
    me = os.path.abspath(__file__)
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    pop = FE.population(cd, rows_limit)
    nq, gptr = pop.nq, pop.gptr
    F, famrec = FE.families(cd, N)
    xadjR, adjR, relR, nrel = rel_csr(cd, N)
    log("L1X relorc %s %s: N %d, %d rows, %d relations, %d structural edges" % (ds, tag, N, nq, nrel, len(adjR)))
    Qu = D.unit_queries(cd, pop.rows)
    stored = None
    fs = os.path.join(D.OUT, "kscale_%s__v1.npz" % ds)
    if os.path.exists(fs):
        stored = np.load(fs)
        assert (stored["rows"][:nq] == pop.rows).all()
    NC = 7                                                     # frank, Ssum, S_out, Sdup, Sorc, Nrel, gold
    COLS = np.zeros((nq, POOL, NC), np.float32)
    ROWINFO = np.zeros((nq, 3), np.int64)                     # |R*|, golds adjacent to a hit by an out-edge, golds
    t0 = time.time()
    for j0, j1, SD, SS, sec, _ in D.batches(cd, pop.rows, Qu, N):
        for i in range(j1 - j0):
            j = j0 + i
            of, od, os_, npos, fv, frank = D.flat_row(SD[i], SS[i])
            g = pop.golds[j]
            top = of[:ACT]
            nh = len(top)
            U, H, GW, FAMI = [], [], [], []
            for fi, f in enumerate(FE.FAMS):
                Fm = F[f]
                st = Fm["xadj"][top]
                hit, pos = E.entries(st, Fm["xadj"][top + 1] - st)
                U.append(Fm["adj"][pos].astype(np.int64))
                H.append(hit)
                GW.append(Fm["g"][top][hit])
                FAMI.append(np.full(len(hit), fi, np.int8))
            u, hit, gw, fam = np.concatenate(U), np.concatenate(H), np.concatenate(GW), np.concatenate(FAMI)
            x = (1.0 / np.arange(1.0, nh + 1.0))[hit] * gw
            L = np.bincount(u, weights=x, minlength=N)
            Lord = D.order_from_score(L, frank)
            lrank = np.full(N, -1, np.int64)
            lrank[Lord] = np.arange(len(Lord))
            cand = np.union1d(Lord, of[:POOL])
            fc = 1.0 / (K0 + frank[cand].astype(np.float64))
            lc = lrank[cand]
            fc = fc + np.where(lc >= 0, 1.0 / (K0 + np.maximum(lc, 0).astype(np.float64)), 0.0)
            order = np.lexsort((frank[cand], -fc))
            pool = cand[order[:POOL]]
            m = len(pool)
            # ---- relation-kept STRUCT_out entries of the hits
            st = xadjR[top]
            hitR, posR = E.entries(st, xadjR[top + 1] - st)
            uR = adjR[posR].astype(np.int64)
            rR = relR[posR].astype(np.int64)
            gR = F["STRUCT_out"]["g"][top][hitR]
            xR = (1.0 / np.arange(1.0, nh + 1.0))[hitR] * gR
            isg = np.isin(uR, g)
            Rs = np.unique(rR[isg])
            sdup = np.bincount(uR, weights=xR, minlength=N)
            sorc = np.bincount(uR[np.isin(rR, Rs)], weights=xR[np.isin(rR, Rs)], minlength=N)
            pairkey = np.unique(uR * (nrel + 1) + rR)
            nrl = np.bincount(pairkey // (nrel + 1), minlength=N)
            so = np.bincount(u[fam == 0], weights=x[fam == 0], minlength=N)
            COLS[j, :m, 0] = np.minimum(frank[pool], FE.CAP)
            COLS[j, :m, 1] = L[pool]
            COLS[j, :m, 2] = so[pool]
            COLS[j, :m, 3] = sdup[pool]
            COLS[j, :m, 4] = sorc[pool]
            COLS[j, :m, 5] = nrl[pool]
            COLS[j, :m, 6] = np.isin(pool, g)
            if stored is not None:
                ix = np.full(N, POOL, np.int64)
                ix[pool] = np.arange(m)
                sl = slice(gptr[j], gptr[j + 1])
                assert (ix[g] == np.minimum(stored["pos_FLATLOC__IR_L1"][sl], POOL)).all(), "baseline positions differ from the kscale record (row %d)" % j
            ROWINFO[j] = [len(Rs), int(np.isin(g, uR).sum()), len(g)]
        log("  rows %d / %d (%.0fs)" % (j1, nq, time.time() - t0))
    res = {"mode": "L1X_RELATION_ORACLE (development; gold-aware diagnostic, never a candidate)", "definitions": __doc__, "dataset": ds, "tag": tag,
           "N": N, "n_rows": nq, "n_relations": nrel, "pool": POOL, "columns": ["frank", "Ssum", "S_out", "Sdup", "Sorc", "Nrel", "gold"], "identity_check": ("gold positions in the pool == %s (clipped at %d) on every row" % (D.rel(fs), POOL)) if stored is not None else "no stored kscale record",
           "rows_with_oracle_relations": int((ROWINFO[:, 0] > 0).sum()), "mean_oracle_relations": float(ROWINFO[:, 0].mean()),
           "seconds": round(time.time() - t_all, 1), "code": {"path": D.rel(me), "sha256": D.sha_file(me)}, "pinned": D.PINNED}
    np.savez_compressed(fo + ".npz", cols=COLS, rowinfo=ROWINFO, rows=pop.rows)
    res["npz"] = {"path": os.path.basename(fo + ".npz"), "sha256": D.sha_file(fo + ".npz")}
    D.G.S.wj(fo + ".json", res)
    log("done (%.0fs) -> %s" % (time.time() - t_all, fo))


if __name__ == "__main__":
    main()
