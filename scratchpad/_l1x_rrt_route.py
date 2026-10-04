"""L1X -- SELECT lane, step 10: ROUTER-ORDER variants on WebQSP, for the served rules of the routed rule table  (development only)
(User ruling 2026-10-01: L1 = parameter-free routing and SELECTION only -- no traversal, no fitted or learned rule.)

Step 9 (_l1x_rrt.py) showed that on WebQSP the PHG partition router costs ALL-gold recall (K = 500, B_N = 1000: routed .834 vs unrouted .920 for the typed rule), because a gold
partition is not among the first B_P(q) partitions of the ES order.  Here ONLY the router's partition ORDER is varied (the count B_P(q) is the shipped one, asserted per row):
  ES    shipped: partitions by first appearance in O' = RRF(H_q, LOC)
  ES_a  RRF(H_q, TYP)         ES_b  RRF(H_q, LOC, TYP)         ES_c  TYP alone          TYP = RRF(Sro, Sri, Tin) (equal weight, K0 = 60) rank over V_q, ties -> the shipped O'
and the served rule is one of RULES (the shipped order, the 3-view and the 4-view typed rule).  Same identities as step 9 for (ES, SHIPPED).  Nothing is chosen here.

  python -u scratchpad/_l1x_rrt_route.py RUN webqsp <tag> [--rows=K]      -> results/L1_X/rrtroute_webqsp__<tag>.{json,npz}  (write-once)
  python -u scratchpad/_l1x_rrt_route.py SHOW <tag> [rule]                 (prints half B vs (ES, SHIPPED) and vs (ES, rule); no record)
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
import _l1x_relsel as RS
import _l1x_rrt as R

log = D.log
K0, ACT, POOL = D.K0, D.ACT, FE.POOL
OUT = R.OUT
MC = D.M_CURVE
NM = len(MC)
RULES = [(), ("Sro", "Sri", "Tin"), ("S_out", "Sro", "Sri", "Tin")]
RNAMES = ["SHIPPED" if not r_ else "+".join(r_) for r_ in RULES]
ROUTERS = ["ES", "ES_a", "ES_b", "ES_c"]


def run():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 3 and a[0] == "RUN" and a[1] == "webqsp", __doc__
    ds, tag = a[1], a[2]
    rows_limit = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--rows=")), None)
    fo = os.path.join(OUT, "rrtroute_%s__%s" % (ds, tag))
    assert rows_limit is not None or not os.path.exists(fo + ".json"), "write-once: %s exists" % fo
    t_all = time.time()
    me = os.path.abspath(__file__)
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    pop = FE.population(cd, rows_limit)
    nq, gptr = pop.nq, pop.gptr
    F, _ = FE.families(cd, N)
    s, d, r, ent = cd.family("structural")
    s, d, r = np.asarray(s, np.int64), np.asarray(d, np.int64), np.asarray(r, np.int64)
    nrel = len(ent.get("relation_vocabulary") or [])
    xO, aO, rO = RS.csr_by(s, d, r, N)
    xI, aI, rI = RS.csr_by(d, s, r, N)
    pIn, relIn = RS.profile(d, r, N, nrel)
    pOut, relOut = RS.profile(s, r, N, nrel)
    del s, d, r
    RM = RS.relation_matrix(ds, nrel)
    cells = R.CELLS[ds]
    maps = R.load_maps(cd, ds)
    tabs = {K: KS.mult_tables(K)[0][R.ALPHA] for K in cells}
    nr = len(RULES)
    log("L1X rrt_route %s %s: N %d, %d rows, cells %s, rules %s, routers %s" % (ds, tag, N, nq, cells, RNAMES, ROUTERS))
    cz = np.load(R.CALIB)
    ngo = np.diff(gptr)
    BP_ST = {K: cz["BP__PHG__k%d__own" % K][:nq] for K in cells}
    SV_ST = {K: cz["SV__PHG__k%d__own" % K][:nq] == ngo[:, None] for K in cells}
    Qu = D.unit_queries(cd, pop.rows)
    ALL = {(K, rv): np.zeros((nr, nq, NM), bool) for K in cells for rv in ROUTERS}
    BPQ = {K: np.zeros(nq, np.int32) for K in cells}
    H100 = maps[R.K_REF]
    wseed = 1.0 / np.arange(1.0, ACT + 1.0)
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
                s_ = Fm["xadj"][top]
                hit, pos = E.entries(s_, Fm["xadj"][top + 1] - s_)
                U.append(Fm["adj"][pos].astype(np.int64))
                H.append(hit)
                GW.append(Fm["g"][top][hit])
                FAMI.append(np.full(len(hit), fi, np.int8))
            u, hit, gw, fam = np.concatenate(U), np.concatenate(H), np.concatenate(GW), np.concatenate(FAMI)
            x = (1.0 / np.arange(1.0, nh + 1.0))[hit] * gw
            L = np.bincount(u, weights=x, minlength=N)
            Lord = D.order_from_score(L, frank)
            f = 1.0 / (K0 + frank.astype(np.float64))
            f[Lord] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
            FO = np.lexsort((frank, -f))
            frk = np.empty(N, np.int64)
            frk[FO] = np.arange(N)
            pool = FO[:POOL]
            pg = frk[g]
            vq = np.union1d(top, Lord)
            fq = np.zeros(len(vq))
            tq = np.zeros(len(vq))
            fq_hit = np.zeros(len(vq))
            fq_hit[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
            fq[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
            il = np.searchsorted(vq, Lord)
            fq[il] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
            tq[il] = ACT + np.arange(len(Lord))
            tq[np.searchsorted(vq, top)] = np.arange(nh)
            perm = np.lexsort((tq, -fq))
            oq = vq[perm]
            # ---- the views
            cs = RM @ Qu[j].astype(np.float32)
            rr = np.empty(nrel, np.int64)
            rr[np.argsort(-cs, kind="stable")] = np.arange(nrel)
            w = 1.0 / (1.0 + rr)
            xh = wseed[:nh] * F["STRUCT_out"]["g"][top]
            xhi = wseed[:nh] * F["STRUCT_in"]["g"][top]
            s_ = xO[top]
            hO, pO = E.entries(s_, xO[top + 1] - s_)
            sro = np.bincount(aO[pO].astype(np.int64), weights=xh[hO] * w[rO[pO].astype(np.int64)], minlength=N)
            s_ = xI[top]
            hI, pI = E.entries(s_, xI[top + 1] - s_)
            sri = np.bincount(aI[pI].astype(np.int64), weights=xhi[hI] * w[rI[pI].astype(np.int64)], minlength=N)
            so = np.bincount(u[fam == 0], weights=x[fam == 0], minlength=N)

            def tmax(ptr, rels, nodes):
                s2 = ptr[nodes]
                ln = ptr[nodes + 1] - s2
                out = np.zeros(len(nodes))
                if ln.sum() > 0:
                    h_, p_ = E.entries(s2, ln)
                    np.maximum.at(out, h_, w[rels[p_]])
                return out

            raw = {"S_out": so[pool], "Sro": sro[pool], "Sri": sri[pool], "Tin": tmax(pIn, relIn, pool)}
            inv = {}
            for n_, v in raw.items():
                o = np.argsort(-v, kind="stable")
                rk = np.empty(POOL, np.int64)
                rk[o] = np.arange(POOL)
                inv[n_] = 1.0 / (K0 + rk)
            gin = pg < POOL
            pgc = np.minimum(pg, POOL - 1)
            NI = np.empty((nr, POOL), np.int64)
            for ri, r_ in enumerate(RULES):
                if not r_:
                    NI[ri] = np.arange(POOL)
                else:
                    S = inv[r_[0]].copy()
                    for v in r_[1:]:
                        S += inv[v]
                    NI[ri][np.argsort(-S, kind="stable")] = np.arange(POOL)
            GN = NI[:, pgc]
            # ---- router-order variants: TYP over V_q (ties -> the shipped O')
            vo = oq

            def vrank(v):
                o_ = np.argsort(-v, kind="stable")
                rk_ = np.empty(len(vo), np.int64)
                rk_[o_] = np.arange(len(vo))
                return rk_
            typs = sum(1.0 / (K0 + vrank(v)) for v in (sro[vo], sri[vo], tmax(pIn, relIn, vo)))
            rt_vq = np.empty(len(vq), np.int64)
            rt_vq[perm] = vrank(typs)
            ft = 1.0 / (K0 + rt_vq)
            OQ = {"ES": oq, "ES_a": vq[np.lexsort((tq, -(fq_hit + ft)))], "ES_b": vq[np.lexsort((tq, -(fq + ft)))], "ES_c": vq[np.lexsort((tq, -ft))]}
            # ---- count (shipped) and routed ALL per cell x router
            Cm = np.bincount(hit * R.K_REF + H100[u], weights=x, minlength=nh * R.K_REF).reshape(nh, R.K_REF)
            kn_, npz_ = RT.knee(wseed[:nh] @ (Cm > 0))
            b0 = kn_ if npz_ else R.K_REF
            for K in cells:
                hard = maps[K]
                b = K if not npz_ else int(tabs[K][b0])
                assert b == BP_ST[K][j], "B_P differs from the stored record (K %d row %d: %d vs %d)" % (K, j, b, BP_ST[K][j])
                BPQ[K][j] = b
                cp_all = None
                for rv in ROUTERS:
                    hq = hard[OQ[rv]]
                    up, first = np.unique(hq, return_index=True)
                    fe = np.full(K, len(OQ[rv]), np.int64)
                    fe[up] = first
                    ro = np.lexsort((np.arange(K), fe))
                    cm = np.zeros(K, bool)
                    cm[ro[:b]] = True
                    if not cm[hard[g]].all():
                        continue
                    cp = cm[hard[pool]]
                    cT = np.cumsum(cm[hard[FO[:max(int(pg.max()) + 1, POOL)]]]) - 1
                    tail = cT[np.minimum(pg, len(cT) - 1)]
                    CPN = np.zeros((nr, POOL), np.int32)
                    CPN[np.arange(nr)[:, None], NI] = cp[None, :]
                    CN = np.cumsum(CPN, axis=1) - 1
                    rp = np.where(gin[None, :], np.take_along_axis(CN, GN, axis=1), tail[None, :])
                    ALL[(K, rv)][:, j] = rp.max(axis=1)[:, None] < np.array(MC)[None, :]
        log("  rows %d / %d (%.0fs)" % (j1, nq, time.time() - t0))
    ident = {}
    for K in cells:
        bad = int((SV_ST[K] != ALL[(K, "ES")][0]).sum())
        ident[str(K)] = {"mismatching_(row,B_N)_cells": bad, "of": int(SV_ST[K].size)}
        assert bad == 0, "shipped (ES, SHIPPED) routed ALL differs from the stored verdict (K %d: %d cells)" % (K, bad)
    log("identities: " + json.dumps(ident))
    A, B = (pop.rows % 2) == 0, (pop.rows % 2) == 1
    res = {"mode": "L1X_ROUTER_ORDER_VARIANTS (development; descriptive; nothing chosen)", "definitions": __doc__, "dataset": ds, "tag": tag, "N": N, "n_rows": nq,
           "cells": [str(K) for K in cells], "rules": RNAMES, "routers": ROUTERS, "budgets": list(MC), "identity_vs_stored": ident,
           "rows_A": int(A.sum()), "rows_B": int(B.sum()), "mean_B_P": {str(K): round(float(BPQ[K].mean()), 2) for K in cells}, "table": {}}
    for K in cells:
        for rv in ROUTERS:
            res["table"]["%d|%s" % (K, rv)] = {nm: {"A": [round(float(ALL[(K, rv)][ri, A, mi].mean()), 4) for mi in range(NM)],
                                                    "B": [round(float(ALL[(K, rv)][ri, B, mi].mean()), 4) for mi in range(NM)]} for ri, nm in enumerate(RNAMES)}
    res["seconds"] = round(time.time() - t_all, 1)
    res["code"] = {"path": D.rel(me), "sha256": D.sha_file(me)}
    res["pinned"] = D.PINNED
    if rows_limit is not None:
        log("SMOKE (--rows): identities passed; not writing")
        return
    arrays = {"rows": pop.rows}
    for K in cells:
        for rv in ROUTERS:
            arrays["ALL__%d__%s" % (K, rv)] = np.packbits(ALL[(K, rv)], axis=2)
        arrays["BPQ__%d" % K] = BPQ[K]
    np.savez_compressed(fo + ".npz", **arrays)
    res["npz"] = {"path": os.path.basename(fo + ".npz"), "sha256": D.sha_file(fo + ".npz")}
    D.G.S.wj(fo + ".json", res)
    log("done (%.0fs) -> %s" % (time.time() - t_all, fo))


def show():
    tag = sys.argv[2]
    rule = sys.argv[3] if len(sys.argv) > 3 else "Sro+Sri+Tin"
    rec = json.load(open(os.path.join(OUT, "rrtroute_webqsp__%s.json" % tag), encoding="utf-8"))
    print("WebQSP routed ALL, half B (n=%d), B_N %s" % (rec["rows_B"], rec["budgets"]))
    base = rec["table"]
    for K in rec["cells"]:
        s = base["%s|ES" % K]["SHIPPED"]["B"]
        print("K %s  (mean B_P %s)" % (K, rec["mean_B_P"][K]))
        for rv in rec["routers"]:
            for nm in ("SHIPPED", rule):
                v = base["%s|%s" % (K, rv)][nm]["B"]
                print("   %-5s %-26s %s | d vs (ES,SHIPPED) %s" % (rv, nm, " ".join("%.3f" % t for t in v), " ".join("%+.3f" % (t - u_) for t, u_ in zip(v, s))))


if __name__ == "__main__":
    if sys.argv[1] == "RUN":
        run()
    elif sys.argv[1] == "SHOW":
        show()
    else:
        raise SystemExit(__doc__)
