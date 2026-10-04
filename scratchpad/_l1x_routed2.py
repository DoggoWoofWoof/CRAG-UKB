"""L1X -- SELECT lane, step 8: parameter-free ROUTER-ORDER variants on top of the routed selection of _l1x_routed.py  (development only)
(User ruling 2026-10-01: L1 = parameter-free routing and SELECTION only -- no traversal, no fitted or learned rule.)

The unrouted tables of _l1x_cross / _l1x_relcross re-order the 5,000-node pool of the carried-forward L1 and read ALL-gold at a node budget B_N.  The shipped
L1 does not serve the whole index: the router (ES partition order + KNEE_SDIV count, alpha .75 at K != 100, unchanged here) contacts the first B_P(q)
partitions and the node budget B_N is spent on the first B_N nodes of the served order INSIDE them.  This harness replaces ONLY the served order:
    O_new = (a rule's re-ordering of the pool O[:5000])  followed by the shipped tail O[5000:]
and varies the router's PARTITION ORDER only (ROUTERS below: the shipped O', or O' with the LOC list replaced by / joined with the typed rank TYP = RRF(Sro, Sri, Tin) over
V_q); the evidence set V_q and the KNEE_SDIV count B_P(q) are exactly as shipped.  A gold node is served iff its partition is contacted and fewer than B_N contacted nodes precede it in O_new.

Rules are equal-weight reciprocal-rank fusions (K0 = 60) of per-candidate views (the views of _l1x_relsel.py; ties -> the shipped position); the rule list is
declared on the command line (default below) and nothing is chosen here.  Identity checks (always on): per row the recomputed B_P(q) of every cell equals the
stored kscale CNT, and the shipped order's routed ALL equals the stored ES (LO, HI, B_P) verdict at every B_N; the unrouted ALL of every rule equals
the unrouted table of the relcross record when it exists.

  python -u scratchpad/_l1x_routed2.py RUN <dataset> <tag> [--rows=K] [--cells=PHG_k100,...] [--rules=a+b,c+d,...] [--count=knee|alpha75]
      -> results/L1_X/routed2_<ds>__<tag>.{json,npz}  (write-once)
"""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_edgediag as E
import _l1d_route as RT
import _l1d_scale as SC
import _l1d_scale_parts as SP
import _l1x_feat as FE
import _l1x_relsel as RS

log = D.log
K0, ACT, POOL = D.K0, D.ACT, FE.POOL
OUT = os.path.join(D.REPO, "results", "L1_X")
MC = D.M_CURVE
DEFAULT_RULES = ["", "Sro+Sri+Tin", "S_out+Sro+Sri+Tin", "frank+Sro+Sri+Tin", "Ssum+S_out"]
ROUTERS = ["ES", "ES_a", "ES_b", "ES_c"]
# ES   shipped: partitions by first appearance in O' = RRF(H_q, LOC)            ES_a  RRF(H_q, TYP)   TYP = RRF(Sro, Sri, Tin) rank over V_q
# ES_b RRF(H_q, LOC, TYP)                                                       ES_c  TYP alone

DEFAULT_CELLS = {"metaqa": ["PHG_k100", "PHG_k250", "PHG_k432", "PHG_k500", "PHG_k1000"], "webqsp": ["PHG_k100", "PHG_k250", "PHG_k500"],
                 "musique": ["PHG_k100", "PHG_k250", "PHG_k500"]}
VIEWS = ["fpos", "frank", "Ssum", "S_out", "S_in", "Sro", "Sri", "Tin", "Tout"]


def load_cells(cd, names):
    """{cell: HardPart} for the requested PHG cells (the native K is the served partition itself)."""
    ds, N = cd.name, int(cd.n_nodes)
    kn = SP.HG.frozen_k(N)
    out = {}
    for nm in names:
        pz, ks = nm.split("_k")
        K = int(ks)
        assert pz == "PHG", "this harness evaluates the PHG cells (the structural partition of the shipped L1); got " + nm
        if K == kn:
            tag = D.TAG_OF[ds + "_phg"]
            P = D.Part(cd, tag)
            out[nm] = SC.HardPart(P.hard, nm, P.path)
        else:
            fp = SP.phg_npy(ds, K)
            sr = json.load(open(fp[:-4] + ".SCALE.json", encoding="utf-8"))
            assert D.sha_file(fp) == sr["output"]["sha256"], "%s changed since its record" % fp
            out[nm] = SC.HardPart(np.load(fp), nm, fp)
        assert out[nm].npart == K and len(out[nm].hard) == N, (nm, out[nm].npart)
    return out


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 3 and a[0] == "RUN", __doc__
    ds, tag = a[1], a[2]
    opt = lambda k: next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--%s=" % k)), None)
    rows_limit = int(opt("rows")) if opt("rows") else None
    rules = opt("rules").split(",") if opt("rules") else DEFAULT_RULES
    rules = [r for r in rules]
    assert rules[0] == "", "the first rule must be the shipped order (empty)"
    cells = opt("cells").split(",") if opt("cells") else DEFAULT_CELLS[ds]
    count_mode = opt("count") or "knee"
    assert count_mode in ("knee", "alpha75"), count_mode
    fo = os.path.join(OUT, "routed2_%s__%s" % (ds, tag))
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
    parts = load_cells(cd, cells)
    log("L1X routed %s %s: N %d, %d rows, %d relations, cells %s, %d rules" % (ds, tag, N, nq, nrel, cells, len(rules)))
    fs = os.path.join(D.OUT, "kscale_%s__v1.npz" % ds)
    stored = np.load(fs)
    assert (stored["rows"][:nq] == pop.rows).all()
    CNT_K = {c: stored["CNT__" + c][:nq] for c in cells}                    # KNEE_SDIV on the cell's own K
    CNT_ST = {c: (CNT_K[c] if count_mode == "knee" else stored["CNTA__" + c][:nq, 1]) for c in cells}   # the served count (alpha .75: min(K, ceil(B_100 (K/100)^.75)))
    LO_ST = {c: stored["LOES__" + c][:nq] for c in cells}
    HI_ST = {c: stored["HIES__" + c][:nq] for c in cells}
    Qu = D.unit_queries(cd, pop.rows)
    nM = len(MC)
    keys = [c if rv == "ES" else "%s@%s" % (c, rv) for rv in ROUTERS for c in cells]
    ALL = {k: np.zeros((len(rules), nq, nM), bool) for k in keys + ["UNR"]}
    BPQ = {c: np.zeros(nq, np.int32) for c in cells}
    ISP = {k: np.zeros(nq, np.float64) for k in keys}      # share of the pool inside the contacted partitions
    t0 = time.time()
    ar5 = np.arange(POOL, dtype=np.float64)
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
            # ---- the served order O (arithmetic of _l1d_scale.row)
            f = 1.0 / (K0 + frank.astype(np.float64))
            f[Lord] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
            FO = np.lexsort((frank, -f))
            frk = np.empty(N, np.int64)
            frk[FO] = np.arange(N)
            pool = FO[:POOL]
            pg = frk[g]                                             # shipped position of every gold
            # ---- the router (unchanged): O', ES partition order, KNEE_SDIV count, per cell
            vq = np.union1d(top, Lord)
            fq = np.zeros(len(vq))
            tq = np.zeros(len(vq))
            fq[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
            il = np.searchsorted(vq, Lord)
            fq[il] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
            tq[il] = ACT + np.arange(len(Lord))
            tq[np.searchsorted(vq, top)] = np.arange(nh)
            oq = vq[np.lexsort((tq, -fq))]
            wseed = 1.0 / np.arange(1.0, nh + 1.0)
            # ---- the views on the pool
            cs = RM @ Qu[j].astype(np.float32)
            rr = np.empty(nrel, np.int64)
            rr[np.argsort(-cs, kind="stable")] = np.arange(nrel)
            w = 1.0 / (1.0 + rr)
            xh = wseed * F["STRUCT_out"]["g"][top]
            xhi = wseed * F["STRUCT_in"]["g"][top]
            st = xO[top]
            hO, pO = E.entries(st, xO[top + 1] - st)
            sro = np.bincount(aO[pO].astype(np.int64), weights=xh[hO] * w[rO[pO].astype(np.int64)], minlength=N)
            st = xI[top]
            hI, pI = E.entries(st, xI[top + 1] - st)
            sri = np.bincount(aI[pI].astype(np.int64), weights=xhi[hI] * w[rI[pI].astype(np.int64)], minlength=N)
            so = np.bincount(u[fam == 0], weights=x[fam == 0], minlength=N)
            si = np.bincount(u[fam == 1], weights=x[fam == 1], minlength=N)

            def tmax(ptr, rels, nodes):
                st_ = ptr[nodes]
                ln = ptr[nodes + 1] - st_
                out = np.zeros(len(nodes))
                if ln.sum() > 0:
                    h_, p_ = E.entries(st_, ln)
                    np.maximum.at(out, h_, w[rels[p_]])
                return out

            raw = {"frank": -np.minimum(frank[pool], FE.CAP).astype(np.float64), "Ssum": L[pool], "S_out": so[pool], "S_in": si[pool],
                   "Sro": sro[pool], "Sri": sri[pool], "Tin": tmax(pIn, relIn, pool), "Tout": tmax(pOut, relOut, pool), "fpos": -ar5}
            # ---- router-order variants: the typed rank TYP over V_q (ties -> the shipped O' order); the count B_P(q) is unchanged
            perm = np.lexsort((tq, -fq))
            vo = vq[perm]

            def vrank(v):
                o_ = np.argsort(-v, kind="stable")
                rk_ = np.empty(len(vo), np.int64)
                rk_[o_] = np.arange(len(vo))
                return rk_
            typs = sum(1.0 / (K0 + vrank(v)) for v in (sro[vo], sri[vo], tmax(pIn, relIn, vo)))
            rt_vq = np.empty(len(vq), np.int64)
            rt_vq[perm] = vrank(typs)
            fq_hit = np.zeros(len(vq))
            fq_hit[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
            fq_loc = fq - fq_hit
            ft = 1.0 / (K0 + rt_vq)
            OQ = {"ES": oq, "ES_a": vq[np.lexsort((tq, -(fq_hit + ft)))], "ES_b": vq[np.lexsort((tq, -(fq + ft)))], "ES_c": vq[np.lexsort((tq, -ft))]}
            inv = {}
            for n_, v in raw.items():
                o = np.argsort(-v, kind="stable")
                rk = np.empty(POOL, np.int64)
                rk[o] = np.arange(POOL)
                inv[n_] = 1.0 / (K0 + rk)
            newidx = []                                             # per rule: new pool index of every pool node (inverse permutation of the order)
            for rname in rules:
                if rname == "":
                    ni = np.arange(POOL)
                else:
                    S = sum(inv[v] for v in rname.split("+"))
                    o = np.argsort(-S, kind="stable")
                    ni = np.empty(POOL, np.int64)
                    ni[o] = np.arange(POOL)
                newidx.append(ni)
            gin = pg < POOL                                         # golds inside the pool
            # ---- unrouted ALL (every partition contacted): the worst new position of a gold
            for ri, ni in enumerate(newidx):
                npos_g = np.where(gin, ni[np.minimum(pg, POOL - 1)], pg)
                mx = int(npos_g.max())
                ALL["UNR"][ri, j] = [mx < M for M in MC]
            # ---- routed ALL per cell
            for c in cells:
                P = parts[c]
                npart, hard = P.npart, P.hard
                hs = hard[top]
                hu = hard[u]
                Cm = np.bincount(hit * npart + hu, weights=x, minlength=nh * npart).reshape(nh, npart)
                sdiv = wseed @ (Cm > 0)
                kn_, npz_ = RT.knee(sdiv)
                b = kn_ if npz_ else npart
                assert b == CNT_K[c][j], "KNEE_SDIV differs from the kscale record (%s row %d: %d vs %d)" % (c, j, b, CNT_K[c][j])
                b = int(CNT_ST[c][j])
                BPQ[c][j] = b
                hp = hard[pool]
                hFo = hard[FO[:max(int(pg.max()) + 1, POOL)]]
                for rv in ROUTERS:
                    k_ = c if rv == "ES" else "%s@%s" % (c, rv)
                    hq = hard[OQ[rv]]
                    up, first = np.unique(hq, return_index=True)
                    fe = np.full(npart, len(OQ[rv]), np.int64)
                    fe[up] = first
                    ro = np.lexsort((np.arange(npart), fe))
                    cm = np.zeros(npart, bool)
                    cm[ro[:b]] = True
                    cp = cm[hp]
                    ISP[k_][j] = cp.mean()
                    cT = np.cumsum(cm[hFo]) - 1                         # contacted nodes strictly before each shipped position
                    gc = cm[hard[g]]                                    # gold partition contacted
                    if not gc.all():
                        continue                                        # some gold is in an uncontacted partition: not served at any B_N
                    for ri, ni in enumerate(newidx):
                        inv_ = np.empty(POOL, np.int64)
                        inv_[ni] = np.arange(POOL)
                        cn = np.cumsum(cp[inv_]) - 1
                        rp = np.where(gin, cn[ni[np.minimum(pg, POOL - 1)]], cT[np.minimum(pg, len(cT) - 1)])
                        m_ = int(rp.max())
                        ALL[k_][ri, j] = [m_ < M for M in MC]
        log("  rows %d / %d (%.0fs)" % (j1, nq, time.time() - t0))
    # ---- identity: the shipped order's routed ALL == the stored ES verdict
    ident = {}
    for c in cells:
        st_all = np.stack([((LO_ST[c] <= CNT_ST[c]) & (CNT_ST[c] <= stored["HIES__" + c][:nq, mi])) for mi in range(nM)], axis=1)
        bad = int((st_all != ALL[c][0]).sum())
        ident[c] = {"mismatching_(row,B_N)_cells": bad, "of": int(st_all.size)}
        assert bad == 0, "shipped routed ALL differs from the stored ES verdict in %s (%d cells)" % (c, bad)
    log("identity vs kscale ES verdict: " + json.dumps(ident))
    # ---- unrouted identity against the kscale shipped positions
    pos_all = stored["pos_FLATLOC__IR_L1"]
    unr0 = np.zeros((nq, nM), bool)
    for j in range(nq):
        p_ = pos_all[gptr[j]:gptr[j + 1]]
        unr0[j] = [int(p_.max()) < M for M in MC]
    assert (unr0 == ALL["UNR"][0]).all(), "shipped unrouted ALL differs from the stored positions"
    A, B = (pop.rows % 2) == 0, (pop.rows % 2) == 1
    hops = getattr(pop, "hops", None)
    names = ["SHIPPED" if r == "" else r for r in rules]
    res = {"mode": "L1X_ROUTED_SELECTION (development; descriptive; nothing chosen)", "definitions": __doc__, "dataset": ds, "tag": tag, "N": N, "n_rows": nq,
           "cells": keys + ["UNR"], "rules": names, "budgets": list(MC), "identity_vs_kscale": ident, "count": count_mode,
           "rows_A": int(A.sum()), "rows_B": int(B.sum()), "table": {}, "mean_B_P": {c: round(float(BPQ[c].mean()), 2) for c in cells},
           "pool_share_in_contacted": {c: round(float(ISP[c].mean()), 4) for c in keys}}
    for c in keys + ["UNR"]:
        res["table"][c] = {}
        for ri, nm in enumerate(names):
            e = {"A": [round(float(ALL[c][ri, A, mi].mean()), 4) for mi in range(nM)], "B": [round(float(ALL[c][ri, B, mi].mean()), 4) for mi in range(nM)]}
            if hops is not None and len(set(np.asarray(hops).tolist())) > 1:
                e["hopB100"] = {int(h): round(float(ALL[c][ri, B & (np.asarray(hops) == h), 0].mean()), 4) for h in sorted(set(np.asarray(hops).tolist()))}
            res["table"][c][nm] = e
    res["seconds"] = round(time.time() - t_all, 1)
    res["code"] = {"path": D.rel(me), "sha256": D.sha_file(me)}
    res["pinned"] = D.PINNED
    if rows_limit is not None:
        log("SMOKE (--rows): not writing; table follows")
        print(json.dumps(res["table"], indent=0)[:3000])
        return
    arrays = {"rows": pop.rows}
    for c in keys + ["UNR"]:
        arrays["ALL__" + c] = ALL[c]
    for c in cells:
        arrays["BPQ__" + c] = BPQ[c]
    np.savez_compressed(fo + ".npz", **arrays)
    res["npz"] = {"path": os.path.basename(fo + ".npz"), "sha256": D.sha_file(fo + ".npz")}
    D.G.S.wj(fo + ".json", res)
    log("done (%.0fs) -> %s" % (time.time() - t_all, fo))


if __name__ == "__main__":
    main()
