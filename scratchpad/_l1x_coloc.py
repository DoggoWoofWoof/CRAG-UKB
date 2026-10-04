"""L1X -- SELECT lane, step 13: COLOCATION REACH DIAGNOSTIC -- are the higher-hop gold answers reachable through the partition?  (development only; dataset-agnostic; measures, chooses nothing)
(User ruling 2026-10-04: "i dont think the higher hops are unreachable that is the entire point of having partitions or colocation".)

The earlier 'ceiling' (MetaQA ALL <= .64 at B_N 5000) was measured on the node POOL O[:5000] (one-hop support of the top-ACT FLAT hits); it says nothing about what the PARTITION co-locates.  Here, per gold node g of a
question with h hops (h = hops of the dataset's own question annotation) and per partition map K (shipped router count B_P(q), asserted per row):
  block      b(g) = hard[g]
  rank       rank of b(g) in the router's block order: ES (shipped, first appearance in O'), ES_m (BM desc, ties ES), ES_mr (equal-weight RRF of the block ranks ES, BM, BRin; ties ES)    [BM, BRin as _l1x_blockrel.py]
  contacted  rank < B_P(q)
  colocation flags of the gold's block:  HIT (it holds a top-ACT FLAT hit), SUP (it holds a node of the one-hop support LOC), BM (it receives relation-conditioned mass, BM > 0)
  served     the gold's position in the served order of the contacted nodes under the shipped order and under Sro+Sri+BRin+BM (-1 = block not contacted); per-gold recall @ B_N
and per question the number of DISTINCT gold blocks n_gb (a question with n_gb > B_P can never be fully contacted whatever the router: the budget feasibility ceiling).  Everything is split by hop count.

  python -u scratchpad/_l1x_coloc.py RUN <ds> <tag> [--rows=K]      -> results/L1_X/coloc_<ds>__<tag>.{json,npz}  (write-once; typed graphs only, like _l1x_blockrel.py)
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
import _l1x_blockrel as BR

log = D.log
K0, ACT = D.K0, D.ACT
OUT = R.OUT
MC = D.M_CURVE
NM = len(MC)
CELLS, K_REF, ALPHA = R.CELLS, R.K_REF, R.ALPHA
ROUTERS = ["ES", "ES_m", "ES_mr"]
BEST = ("Sro", "Sri", "BRin", "BM")


def run():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 3 and a[0] == "RUN", __doc__
    ds, tag = a[1], a[2]
    rows_limit = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--rows=")), None)
    fo = os.path.join(OUT, "coloc_%s__%s" % (ds, tag))
    assert rows_limit is not None or not os.path.exists(fo + ".json"), "write-once: %s exists" % fo
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:
        pass
    t_all = time.time()
    me = os.path.abspath(__file__)
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    pop = FE.population(cd, rows_limit)
    nq, gptr = pop.nq, pop.gptr
    ng_tot = int(gptr[-1])
    F, _ = FE.families(cd, N)
    s, d, r, ent = cd.family("structural")
    s, d, r = np.asarray(s, np.int64), np.asarray(d, np.int64), np.asarray(r, np.int64)
    nrel = len(ent.get("relation_vocabulary") or [])
    assert nrel > 1, "typed-graph switch: the view needs a structural family with more than one relation"
    cells = CELLS[ds]
    maps = R.load_maps(cd, ds)
    LIFT = {K: BR.block_tables(maps[K], s, d, r, K, nrel) for K in cells}
    xO, aO, rO = RS.csr_by(s, d, r, N)
    xI, aI, rI = RS.csr_by(d, s, r, N)
    del s, d, r
    RM = RS.relation_matrix(ds, nrel)
    tabs = {K: KS.mult_tables(K)[0][ALPHA] for K in cells}
    log("L1X coloc %s %s: N %d, %d rows, %d gold nodes, %d relations, cells %s" % (ds, tag, N, nq, ng_tot, nrel, cells))
    if ds == "metaqa":
        st = np.load(os.path.join(D.OUT, "kscale_metaqa__v1.npz"))
        assert (st["rows"][:nq] == pop.rows).all()
        BP_ST = {K: st["CNTA__PHG_k%d" % K][:nq, 1] for K in cells}
        SV_ST = {K: np.stack([(st["LOES__PHG_k%d" % K][:nq] <= BP_ST[K]) & (BP_ST[K] <= st["HIES__PHG_k%d" % K][:nq, mi]) for mi in range(NM)], axis=1) for K in cells}
    else:
        cz = np.load(R.CALIB)
        ngo = np.diff(gptr)
        BP_ST = {K: cz["BP__PHG__k%d__own" % K][:nq] for K in cells}
        SV_ST = {K: cz["SV__PHG__k%d__own" % K][:nq] == ngo[:, None] for K in cells}
    Qu = D.unit_queries(cd, pop.rows)
    G = {}
    for K in cells:
        G[K] = {"rank": {rn: np.full(ng_tot, -1, np.int32) for rn in ROUTERS}, "flags": np.zeros(ng_tot, np.uint8),
                "pos": {(rn, ru): np.full(ng_tot, -1, np.int32) for rn in ("ES", "ES_mr") for ru in ("SHIPPED", "BEST")},
                "ngb": np.zeros(nq, np.int32), "bp": np.zeros(nq, np.int32)}
    ALLS = {K: np.zeros((nq, NM), bool) for K in cells}               # shipped served order, shipped router: identity
    H100 = maps[K_REF]
    wseed = 1.0 / np.arange(1.0, ACT + 1.0)
    MCa = np.array(MC)
    t0 = time.time()
    for j0, j1, SD, SS, sec, _ in D.batches(cd, pop.rows, Qu, N):
        for i in range(j1 - j0):
            j = j0 + i
            of, od, os_, npos, fv, frank = D.flat_row(SD[i], SS[i])
            g = pop.golds[j]
            sl = slice(int(gptr[j]), int(gptr[j + 1]))
            top = of[:ACT]
            nh = len(top)
            U, H, GW = [], [], []
            for fi, f in enumerate(FE.FAMS):
                Fm = F[f]
                s_ = Fm["xadj"][top]
                hit, pos = E.entries(s_, Fm["xadj"][top + 1] - s_)
                U.append(Fm["adj"][pos].astype(np.int64))
                H.append(hit)
                GW.append(Fm["g"][top][hit])
            u, hit, gw = np.concatenate(U), np.concatenate(H), np.concatenate(GW)
            x = (1.0 / np.arange(1.0, nh + 1.0))[hit] * gw
            L = np.bincount(u, weights=x, minlength=N)
            Lord = D.order_from_score(L, frank)
            f = 1.0 / (K0 + frank.astype(np.float64))
            f[Lord] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
            FO = np.lexsort((frank, -f))
            vq = np.union1d(top, Lord)
            fq = np.zeros(len(vq))
            tq = np.zeros(len(vq))
            fq[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
            il = np.searchsorted(vq, Lord)
            fq[il] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
            tq[il] = ACT + np.arange(len(Lord))
            tq[np.searchsorted(vq, top)] = np.arange(nh)
            oq = vq[np.lexsort((tq, -fq))]
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
            sm = sro + sri
            Cm = np.bincount(hit * K_REF + H100[u], weights=x, minlength=nh * K_REF).reshape(nh, K_REF)
            kn_, npz_ = RT.knee(wseed[:nh] @ (Cm > 0))
            b0 = kn_ if npz_ else K_REF
            for K in cells:
                hard = maps[K]
                b = K if not npz_ else int(tabs[K][b0])
                assert b == BP_ST[K][j], "B_P differs from the stored record (K %d row %d: %d vs %d)" % (K, j, b, BP_ST[K][j])
                gb = hard[g]
                Gk = G[K]
                Gk["bp"][j] = b
                Gk["ngb"][j] = len(np.unique(gb))
                hq = hard[oq]
                up, first = np.unique(hq, return_index=True)
                fe = np.full(K, len(oq), np.int64)
                fe[up] = first
                ro_es = np.lexsort((np.arange(K), fe))
                rk_es = np.empty(K, np.int64)
                rk_es[ro_es] = np.arange(K)
                lin, _lo = LIFT[K]
                bm = np.bincount(hard, weights=sm, minlength=K)
                brin = lin @ w
                o_m = np.lexsort((rk_es, -bm))
                rk_m = np.empty(K, np.int64)
                rk_m[o_m] = np.arange(K)
                o_b = np.lexsort((rk_es, -brin))
                rk_b = np.empty(K, np.int64)
                rk_b[o_b] = np.arange(K)
                sc = 1.0 / (K0 + rk_es) + 1.0 / (K0 + rk_m) + 1.0 / (K0 + rk_b)
                o_mr = np.lexsort((rk_es, -sc))
                rk_mr = np.empty(K, np.int64)
                rk_mr[o_mr] = np.arange(K)
                RK = {"ES": rk_es, "ES_m": rk_m, "ES_mr": rk_mr}
                for rn in ROUTERS:
                    Gk["rank"][rn][sl] = RK[rn][gb]
                has_hit = np.bincount(hard[top], minlength=K) > 0
                has_sup = np.bincount(hard[Lord], minlength=K) > 0
                Gk["flags"][sl] = has_hit[gb].astype(np.uint8) | (has_sup[gb].astype(np.uint8) << 1) | ((bm[gb] > 0).astype(np.uint8) << 2)
                for rn in ("ES", "ES_mr"):
                    cm = np.zeros(K, bool)
                    cm[(ro_es if rn == "ES" else o_mr)[:b]] = True
                    if not cm[gb].all():
                        # partly contacted questions: record positions of the contacted golds only (rule order over the contacted nodes)
                        pass
                    C = FO[cm[hard[FO]]]
                    m = len(C)
                    posC = np.full(N, -1, np.int64)
                    posC[C] = np.arange(m)
                    gi = posC[g]
                    ok = gi >= 0
                    Gk["pos"][(rn, "SHIPPED")][sl] = np.where(ok, gi, -1)
                    if rn == "ES":
                        ALLS[K][j] = (ok.all()) & (gi.max() < MCa) if ok.all() else False
                    if ok.any():
                        hb = hard[C]
                        lin_, _ = LIFT[K]
                        raw = {"Sro": sro[C], "Sri": sri[C], "BRin": brin[hb], "BM": bm[hb]}
                        S = np.zeros(m)
                        for v in BEST:
                            o = np.argsort(-raw[v], kind="stable")
                            rk = np.empty(m, np.int64)
                            rk[o] = np.arange(m)
                            S += 1.0 / (K0 + rk)
                        O = np.argsort(-S, kind="stable")
                        NI = np.empty(m, np.int64)
                        NI[O] = np.arange(m)
                        Gk["pos"][(rn, "BEST")][sl] = np.where(ok, NI[np.maximum(gi, 0)], -1)
        if (j1 // 50) != (j0 // 50):
            log("  rows %d / %d (%.0fs)" % (j1, nq, time.time() - t0))
    ident = {}
    for K in cells:
        bad = int((SV_ST[K] != ALLS[K]).sum())
        ident[str(K)] = {"mismatching_(row,B_N)_cells": bad, "of": int(SV_ST[K].size)}
        assert bad == 0, "shipped routed ALL differs from the stored verdict (K %d: %d cells)" % (K, bad)
    log("identities: " + json.dumps(ident))
    hops_row = np.asarray(pop.hops) if getattr(pop, "hops", None) is not None else np.zeros(nq, int)
    hops_g = np.repeat(hops_row, np.diff(gptr))
    Hs = sorted(set(hops_row.tolist()))
    summ = {}
    for K in cells:
        Gk = G[K]
        bp_g = np.repeat(Gk["bp"], np.diff(gptr))
        S_ = {}
        for h in Hs + ["all"]:
            mg = np.ones(ng_tot, bool) if h == "all" else hops_g == h
            mq = np.ones(nq, bool) if h == "all" else hops_row == h
            if not mg.any():
                continue
            e = {"gold_nodes": int(mg.sum()), "questions": int(mq.sum()),
                 "questions_with_n_gold_blocks_le_B_P (budget feasibility ceiling)": round(float((Gk["ngb"][mq] <= Gk["bp"][mq]).mean()), 4),
                 "median_distinct_gold_blocks": float(np.median(Gk["ngb"][mq])),
                 "mean_B_P": round(float(Gk["bp"][mq].mean()), 2),
                 "gold_block_has_FLAT_hit": round(float((Gk["flags"][mg] & 1).mean()), 4),
                 "gold_block_has_support_node": round(float(((Gk["flags"][mg] >> 1) & 1).mean()), 4),
                 "gold_block_receives_BM_mass": round(float(((Gk["flags"][mg] >> 2) & 1).mean()), 4)}
            for rn in ROUTERS:
                rk = Gk["rank"][rn][mg]
                e["gold_block_contacted_%s" % rn] = round(float((rk < bp_g[mg]).mean()), 4)
                e["gold_block_rank_over_B_P_p50_p90_%s" % rn] = [round(float(np.median(rk / bp_g[mg])), 3), round(float(np.percentile(rk / bp_g[mg], 90)), 3)]
            for key, arr in Gk["pos"].items():
                p = arr[mg]
                e["gold_recall_at_B_N_%s_%s" % key] = [round(float(((p >= 0) & (p < M)).mean()), 4) for M in MC]
            S_[str(h)] = e
        summ[str(K)] = S_
    res = {"mode": "L1X_COLOCATION_REACH_DIAGNOSTIC (development; descriptive; nothing chosen)", "definitions": __doc__, "dataset": ds, "tag": tag, "N": N, "n_rows": nq, "gold_nodes": ng_tot,
           "n_relations": nrel, "cells": [str(K) for K in cells], "routers": ROUTERS, "budgets": list(MC), "hops": Hs, "identity_vs_stored": ident, "summary": summ,
           "seconds": round(time.time() - t_all, 1), "code": {"path": D.rel(me), "sha256": D.sha_file(me)}, "pinned": D.PINNED}
    if rows_limit is not None:
        log("SMOKE (--rows): identities passed; not writing")
        print(json.dumps({K: summ[K]["all"] for K in list(summ)[:1]}, indent=1)[:1800])
        return
    arrays = {"rows": pop.rows, "gptr": gptr, "hops_g": hops_g}
    for K in cells:
        arrays["ngb__%d" % K] = G[K]["ngb"]
        arrays["bp__%d" % K] = G[K]["bp"]
        arrays["flags__%d" % K] = G[K]["flags"]
        for rn in ROUTERS:
            arrays["rank__%d|%s" % (K, rn)] = G[K]["rank"][rn]
        for key, arr in G[K]["pos"].items():
            arrays["pos__%d|%s|%s" % (K, key[0], key[1])] = arr
    np.savez_compressed(fo + ".npz", **arrays)
    res["npz"] = {"path": os.path.basename(fo + ".npz"), "sha256": D.sha_file(fo + ".npz")}
    D.G.S.wj(fo + ".json", res)
    log("done (%.0fs) -> %s" % (time.time() - t_all, fo))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a and a[0] == "RUN":
        run()
    else:
        raise SystemExit(__doc__)
