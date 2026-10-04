"""L1X -- SELECT lane, step 9: the ROUTED RULE TABLE -- every equal-weight RRF rule of the typed views, scored inside the REAL routed pipeline  (development only)
(User ruling 2026-10-01: L1 = parameter-free routing and SELECTION only -- no traversal, no fitted or learned rule.)

Routing is the shipped one and is NOT varied: the router order O' = RRF(H_q, LOC), ES partition order, B_P(q) = min(K, ceil(B_100(q) (K / 100)^0.75)) with
B_100(q) = KNEE_SDIV on the SAME partitioner's K = 100 map (a query without evidence contacts every partition), served list = the first B_N nodes of the
served order O inside the contacted partitions.  Only the SERVED ORDER is replaced:  O_new = (a rule's re-ordering of the pool O[:5000]) + the shipped tail.
A rule = the equal-weight reciprocal-rank fusion (K0 = 60; ties -> the shipped position) of a subset (size 1..4) of the nine per-candidate views of
_l1x_relsel.py: fpos frank Ssum S_out S_in Sro Sri Tin Tout (255 rules + the shipped order).  Nothing is chosen here; COMBINE prints the maximin table.

Identity checks (always on):  per row B_P(q) == the stored record (MetaQA: results/L1_DEV/kscale_metaqa__v1.npz CNTA alpha .75; WebQSP: the PHG reference of
results/FREEBASE_SCALE/CALIB_EVAL_WEBQSP__v1.npz BP__PHG__k<K>__own), and the shipped order's routed ALL equals the stored verdict at every B_N (MetaQA: LOES / HIES;
WebQSP: SV__PHG__k<K>__own == #gold); the shipped unrouted ALL equals the stored gold positions (MetaQA kscale pos_FLATLOC__IR_L1; WebQSP: the unrouted count of CALIB_EVAL
is not stored, so it is checked against feat_webqsp__v1.npz pos_O).

  python -u scratchpad/_l1x_rrt.py RUN <metaqa|webqsp> <tag> [--rows=K]        -> results/L1_X/rrt_<ds>__<tag>.{json,npz}  (write-once)
  python -u scratchpad/_l1x_rrt.py COMBINE <tag> <ds> <ds> ... [--tol=0.003]   (prints; no record)
"""
import itertools
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

log = D.log
K0, ACT, POOL = D.K0, D.ACT, FE.POOL
OUT = os.path.join(D.REPO, "results", "L1_X")
MC = D.M_CURVE
NM = len(MC)
VIEWS = ["fpos", "frank", "Ssum", "S_out", "S_in", "Sro", "Sri", "Tin", "Tout"]
CELLS = {"metaqa": (100, 250, 432, 500, 1000), "webqsp": (100, 250, 500)}
K_REF = 100
ALPHA = "0.75"
PARTS_W = os.path.join(D.REPO, "results", "L1_HOST", "parts")
CALIB = os.path.join(D.REPO, "results", "FREEBASE_SCALE", "CALIB_EVAL_WEBQSP__v1.npz")


def sha_file(p):
    return D.sha_file(p)


def rule_list():
    rl = [()]
    for k in (1, 2, 3, 4):
        rl += list(itertools.combinations(VIEWS, k))
    return rl


def load_maps(cd, ds):
    N = int(cd.n_nodes)
    maps = {}
    if ds == "metaqa":
        import _l1d_scale_parts as SP                             # MetaQA's scale cells (laptop); the WebQSP maps are read from results/L1_HOST/parts
        kn = SP.HG.frozen_k(N)
        for K in CELLS[ds]:
            if K == kn:
                P = D.Part(cd, D.TAG_OF[ds + "_phg"])
                maps[K] = np.asarray(P.hard, np.int64)
            else:
                fp = SP.phg_npy(ds, K)
                sr = json.load(open(fp[:-4] + ".SCALE.json", encoding="utf-8"))
                assert sha_file(fp) == sr["output"]["sha256"], "%s changed since its record" % fp
                maps[K] = np.load(fp).astype(np.int64)
    else:
        for K in set(CELLS[ds]) | {K_REF}:
            rj = os.path.join(PARTS_W, "webqsp__H4_SK_k%d__PHG_con.RUN.json" % K)
            R = json.load(open(rj, encoding="utf-8"))
            p = os.path.join(D.REPO, R["output"]["file"])
            assert sha_file(p) == R["output"]["sha256"], "PHG map K %d differs from its RUN.json" % K
            assert R["STATUS"] == "OK" and R["balance"]["every_block_used"]
            maps[K] = np.load(p).astype(np.int64)
    for K, h in maps.items():
        assert len(h) == N and int(h.max()) + 1 == K, (K, int(h.max()) + 1)
    return maps


def run():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 3 and a[0] == "RUN", __doc__
    ds, tag = a[1], a[2]
    rows_limit = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--rows=")), None)
    fo = os.path.join(OUT, "rrt_%s__%s" % (ds, tag))
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
    cells = CELLS[ds]
    maps = load_maps(cd, ds)
    tabs = {K: KS.mult_tables(K)[0][ALPHA] for K in cells}
    rules = rule_list()
    names = ["SHIPPED" if not r_ else "+".join(r_) for r_ in rules]
    nr = len(rules)
    log("L1X rrt %s %s: N %d, %d rows, %d relations, cells %s, %d rules" % (ds, tag, N, nq, nrel, cells, nr))
    # ---- the stored records
    if ds == "metaqa":
        st = np.load(os.path.join(D.OUT, "kscale_metaqa__v1.npz"))
        assert (st["rows"][:nq] == pop.rows).all()
        BP_ST = {K: st["CNTA__PHG_k%d" % K][:nq, 1] for K in cells}
        SV_ST = {K: np.stack([(st["LOES__PHG_k%d" % K][:nq] <= BP_ST[K]) & (BP_ST[K] <= st["HIES__PHG_k%d" % K][:nq, mi]) for mi in range(NM)], axis=1) for K in cells}
        POS_ST = st["pos_FLATLOC__IR_L1"]
        UNR_ST = np.stack([np.maximum.reduceat(POS_ST, gptr[:-1])[:nq] < M for M in MC], axis=1) if rows_limit is None else None
        if rows_limit is not None:
            UNR_ST = np.array([[int(POS_ST[gptr[j]:gptr[j + 1]].max()) < M for M in MC] for j in range(nq)])
    else:
        cz = np.load(CALIB)
        ngo = np.diff(gptr)
        BP_ST = {K: cz["BP__PHG__k%d__own" % K][:nq] for K in cells}
        SV_ST = {K: cz["SV__PHG__k%d__own" % K][:nq] == ngo[:, None] for K in cells}
        fz = np.load(os.path.join(OUT, "feat_webqsp__v1.npz"))
        assert (fz["rows"][:nq] == pop.rows).all()
        PO = fz["pos_O"]
        UNR_ST = np.array([[int(PO[fz["gptr"][j]:fz["gptr"][j + 1]].max()) < M for M in MC] for j in range(nq)])
    Qu = D.unit_queries(cd, pop.rows)
    ALL = {K: np.zeros((nr, nq, NM), bool) for K in cells}
    ALL["UNR"] = np.zeros((nr, nq, NM), bool)
    BPQ = {K: np.zeros(nq, np.int32) for K in cells}
    H100 = maps[K_REF]
    wseed = 1.0 / np.arange(1.0, ACT + 1.0)
    ar5 = np.arange(POOL, dtype=np.float64)
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
            fq[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
            il = np.searchsorted(vq, Lord)
            fq[il] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
            tq[il] = ACT + np.arange(len(Lord))
            tq[np.searchsorted(vq, top)] = np.arange(nh)
            oq = vq[np.lexsort((tq, -fq))]
            # ---- the views on the pool
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
            si = np.bincount(u[fam == 1], weights=x[fam == 1], minlength=N)

            def tmax(ptr, rels, nodes):
                s2 = ptr[nodes]
                ln = ptr[nodes + 1] - s2
                out = np.zeros(len(nodes))
                if ln.sum() > 0:
                    h_, p_ = E.entries(s2, ln)
                    np.maximum.at(out, h_, w[rels[p_]])
                return out

            raw = {"fpos": -ar5, "frank": -np.minimum(frank[pool], FE.CAP).astype(np.float64), "Ssum": L[pool], "S_out": so[pool], "S_in": si[pool],
                   "Sro": sro[pool], "Sri": sri[pool], "Tin": tmax(pIn, relIn, pool), "Tout": tmax(pOut, relOut, pool)}
            inv = {}
            for n_, v in raw.items():
                o = np.argsort(-v, kind="stable")
                rk = np.empty(POOL, np.int64)
                rk[o] = np.arange(POOL)
                inv[n_] = 1.0 / (K0 + rk)
            gin = pg < POOL
            pgc = np.minimum(pg, POOL - 1)
            NI = np.empty((nr, POOL), np.int64)                       # per rule: new pool index of every pool node
            for ri, r_ in enumerate(rules):
                if not r_:
                    NI[ri] = np.arange(POOL)
                else:
                    S = inv[r_[0]].copy()
                    for v in r_[1:]:
                        S += inv[v]
                    o = np.argsort(-S, kind="stable")
                    NI[ri][o] = np.arange(POOL)
            GN = NI[:, pgc]                                           # (nr, ngold) new pool index of every gold (valid where gin)
            # ---- unrouted ALL
            mx = np.where(gin[None, :], GN, pg[None, :]).max(axis=1)
            ALL["UNR"][:, j] = mx[:, None] < np.array(MC)[None, :]
            # ---- the routed ALL per cell
            Cm = np.bincount(hit * K_REF + H100[u], weights=x, minlength=nh * K_REF).reshape(nh, K_REF)
            kn_, npz_ = RT.knee(wseed[:nh] @ (Cm > 0))
            b0 = kn_ if npz_ else K_REF
            hp_pool = None
            for K in cells:
                hard = maps[K]
                b = K if not npz_ else int(tabs[K][b0])
                assert b == BP_ST[K][j], "B_P differs from the stored record (K %d row %d: %d vs %d)" % (K, j, b, BP_ST[K][j])
                BPQ[K][j] = b
                hq = hard[oq]
                up, first = np.unique(hq, return_index=True)
                fe = np.full(K, len(oq), np.int64)
                fe[up] = first
                ro = np.lexsort((np.arange(K), fe))
                cm = np.zeros(K, bool)
                cm[ro[:b]] = True
                if not cm[hard[g]].all():
                    continue                                          # a gold partition is not contacted: not served at any B_N
                cp = cm[hard[pool]]
                cT = np.cumsum(cm[hard[FO[:max(int(pg.max()) + 1, POOL)]]]) - 1     # contacted nodes strictly before each shipped position
                tail = cT[np.minimum(pg, len(cT) - 1)]
                # contacted nodes before every pool node in the new order, per rule: cumulative sum of cp over the new order
                # position t of rule r holds the pool node with NI[r, node] == t  ->  cn[r, NI[r, node]] = #contacted among positions <= t
                CPN = np.zeros((nr, POOL), np.int32)
                CPN[np.arange(nr)[:, None], NI] = cp[None, :]
                CN = np.cumsum(CPN, axis=1) - 1
                rp = np.where(gin[None, :], np.take_along_axis(CN, GN, axis=1), tail[None, :])
                m_ = rp.max(axis=1)
                ALL[K][:, j] = m_[:, None] < np.array(MC)[None, :]
        log("  rows %d / %d (%.0fs)" % (j1, nq, time.time() - t0))
    # ---- identities
    ident = {}
    for K in cells:
        bad = int((SV_ST[K] != ALL[K][0]).sum())
        ident[str(K)] = {"mismatching_(row,B_N)_cells": bad, "of": int(SV_ST[K].size)}
        assert bad == 0, "shipped routed ALL differs from the stored verdict (K %d: %d cells)" % (K, bad)
    bad = int((UNR_ST != ALL["UNR"][0]).sum())
    ident["unrouted"] = {"mismatching_(row,B_N)_cells": bad, "of": int(UNR_ST.size)}
    assert bad == 0, "shipped unrouted ALL differs from the stored positions (%d cells)" % bad
    log("identities: " + json.dumps(ident))
    A, B = (pop.rows % 2) == 0, (pop.rows % 2) == 1
    hops = np.asarray(pop.hops) if getattr(pop, "hops", None) is not None else None
    Hs = sorted(set(hops.tolist())) if hops is not None else []
    res = {"mode": "L1X_ROUTED_RULE_TABLE (development; descriptive; nothing chosen)", "definitions": __doc__, "dataset": ds, "tag": tag, "N": N, "n_rows": nq,
           "cells": [str(K) for K in cells] + ["UNR"], "rules": names, "views": VIEWS, "budgets": list(MC), "identity_vs_stored": ident,
           "rows_A": int(A.sum()), "rows_B": int(B.sum()), "mean_B_P": {str(K): round(float(BPQ[K].mean()), 2) for K in cells}, "hops": Hs, "table": {}}
    for c in list(cells) + ["UNR"]:
        T = {}
        for ri, nm in enumerate(names):
            e = {"A": [round(float(ALL[c][ri, A, mi].mean()), 4) for mi in range(NM)], "B": [round(float(ALL[c][ri, B, mi].mean()), 4) for mi in range(NM)]}
            if len(Hs) > 1:
                e["hopA100"] = [round(float(ALL[c][ri, A & (hops == h), 0].mean()), 4) for h in Hs]
                e["hopB100"] = [round(float(ALL[c][ri, B & (hops == h), 0].mean()), 4) for h in Hs]
            T[nm] = e
        res["table"][str(c)] = T
    res["seconds"] = round(time.time() - t_all, 1)
    res["code"] = {"path": D.rel(me), "sha256": sha_file(me)}
    res["pinned"] = D.PINNED
    if rows_limit is not None:
        log("SMOKE (--rows): identities passed; not writing")
        return
    arrays = {"rows": pop.rows}
    for c in list(cells) + ["UNR"]:
        arrays["ALL__%s" % c] = np.packbits(ALL[c], axis=2)
    for K in cells:
        arrays["BPQ__%d" % K] = BPQ[K]
    np.savez_compressed(fo + ".npz", **arrays)
    res["npz"] = {"path": os.path.basename(fo + ".npz"), "sha256": sha_file(fo + ".npz")}
    D.G.S.wj(fo + ".json", res)
    log("done (%.0fs) -> %s" % (time.time() - t_all, fo))


def obj4(v):
    return float(np.mean(v[:4]))


def combine(tag, dss, tol):
    recs = {ds: json.load(open(os.path.join(OUT, "rrt_%s__%s.json" % (ds, tag)), encoding="utf-8")) for ds in dss}
    gate = {ds: [str(K) for K in CELLS[ds]] for ds in dss}

    def objA(ds, nm, half):
        return float(np.mean([obj4(recs[ds]["table"][c][nm][half]) for c in gate[ds]]))
    names = set.intersection(*[set(r["table"][gate[ds][0]].keys()) for ds, r in recs.items()])
    rows = []
    for n in names:
        dA = {ds: objA(ds, n, "A") - objA(ds, "SHIPPED", "A") for ds in dss}
        dB = {ds: objA(ds, n, "B") - objA(ds, "SHIPPED", "B") for ds in dss}
        rows.append((n, float(np.mean(list(dA.values()))), min(dA.values()), dA, dB))
    ok = sorted([r for r in rows if r[2] >= -tol], key=lambda r: -r[1])
    print("datasets %s ; rules %d ; admissible %d (tol %.4f) ; objective = routed ALL, mean over B_N 100..1000 and over the gate cells %s" % (dss, len(rows), len(ok), tol, gate))
    print("shipped objA %s objB %s" % ({ds: round(objA(ds, "SHIPPED", "A"), 4) for ds in dss}, {ds: round(objA(ds, "SHIPPED", "B"), 4) for ds in dss}))
    for n, m, mn, dA, dB in ok[:15]:
        print("  %-30s meanDA %+.4f minDA %+.4f | dA %s | dB %s" % (n, m, mn, {k: round(v, 4) for k, v in dA.items()}, {k: round(v, 4) for k, v in dB.items()}))
    print("--- best per dataset by A:")
    for ds in dss:
        b = max(rows, key=lambda r: r[3][ds])
        print("  %-8s %-30s dA %+.4f dB %+.4f" % (ds, b[0], b[3][ds], b[4][ds]))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a[0] == "RUN":
        run()
    elif a[0] == "COMBINE":
        tol = next((float(x[6:]) for x in sys.argv if x.startswith("--tol=")), 0.003)
        combine(a[1], a[2:], tol)
    else:
        raise SystemExit(__doc__)
