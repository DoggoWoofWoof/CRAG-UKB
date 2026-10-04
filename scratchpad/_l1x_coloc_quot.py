"""L1X -- SELECT lane, step 14: BLOCK-QUOTIENT COLOCATION DIAGNOSTIC -- is the hop-2/3 gold block ADJACENT to a hit block in the partition?  (development only; dataset-agnostic; measures, chooses nothing)
(User ruling 2026-10-04: "i dont think the higher hops are unreachable that is the entire point of having partitions or colocation".)

Step 13 (_l1x_coloc.py, MetaQA) found: the gold blocks ARE co-located (median 1-2 distinct gold blocks per question; n_gold_blocks <= B_P for 94-100% of questions) but the shipped router and the hit/relation-mass router (ES_m) CONTACT the hop-2 gold block only 37-46% of the time
(hop 1 .99, hop 3 .55-.65) and only 22-77% of hop-2 gold blocks hold a top-ACT FLAT hit.  Here the QUOTIENT graph of the partition (static, from the structural edges: Q[b',b] = edges from block b' to block b, optionally relation-weighted by the query's w(q,r)) is used to ask whether the
gold block is the hit block's NEIGHBOUR, i.e. whether colocation 'one block over' carries the hop-2/3 golds.
Block scores (all non-parametric; Qs = symmetrised quotient, row-normalised; w(q,r) = 1/(1+rank of relation r by dense cosine with the relation encodings), as _l1x_blockrel.py):
  hm        block hit mass: sum of the seed weights 1/(1+i) of the top-ACT FLAT hits that sit in the block (the block-level seed vector)
  bm        block relation mass BM of _l1x_blockrel.py
  QSu       hm  @ Qs_unweighted        (one block-quotient step from the hit blocks, relations ignored)
  QSr       hm  @ Qs_relation-weighted (same, edges weighted by w(q, r))
  QMr       bm  @ Qs_relation-weighted (the block-quotient step applied to the one-hop relation mass: two propagation steps in all)
Routers (block order; ties -> ES position): ES (shipped) | ES_m (BM) | QSu | QSr | QMr | ES_qs (equal RRF of ES, BM, QSr) | ES_qm (equal RRF of ES, BM, QMr).
Per gold: block rank under each router, contacted = rank < B_P(q) (asserted equal to the stored B_P); flags: gold block is a HIT block, or ADJ (Qs_u[hit block, gold block] > 0 for some hit block).
Per question: ALL-contact = every gold block contacted.  Split by hop.

  python -u scratchpad/_l1x_coloc_quot.py RUN <ds> <tag> [--rows=K]      -> results/L1_X/colocq_<ds>__<tag>.{json,npz}  (write-once; typed graphs only)
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
K0, ACT = D.K0, D.ACT
OUT = R.OUT
CELLS, K_REF, ALPHA = R.CELLS, R.K_REF, R.ALPHA
ROUTERS = ["ES", "ES_m", "QSu", "QSr", "QMr", "ES_qs", "ES_qm"]


def quotient_tables(hard, s, d, r, K, nrel):
    """Sparse (pair, relation) edge counts of the block quotient: pair = block(src)*K + block(dst)."""
    key = (hard[s] * K + hard[d]) * nrel + r
    u, c = np.unique(key, return_counts=True)
    pair, rel = u // nrel, u % nrel
    return pair.astype(np.int64), rel.astype(np.int64), c.astype(np.float64)


def sym_norm(M):
    M = M + M.T
    rs = M.sum(1, keepdims=True)
    return M / np.maximum(rs, 1e-12)


def run():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 3 and a[0] == "RUN", __doc__
    ds, tag = a[1], a[2]
    rows_limit = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--rows=")), None)
    fo = os.path.join(OUT, "colocq_%s__%s" % (ds, tag))
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
    QT = {K: quotient_tables(maps[K], s, d, r, K, nrel) for K in cells}
    QU = {}
    for K in cells:
        pair, rel, c = QT[K]
        M = np.bincount(pair, weights=c, minlength=K * K).reshape(K, K)
        QU[K] = sym_norm(M)
        QT[K] = (pair, rel, c)
    xO, aO, rO = RS.csr_by(s, d, r, N)
    xI, aI, rI = RS.csr_by(d, s, r, N)
    del s, d, r
    RM = RS.relation_matrix(ds, nrel)
    tabs = {K: KS.mult_tables(K)[0][ALPHA] for K in cells}
    log("L1X colocq %s %s: N %d, %d rows, %d gold nodes, %d relations, cells %s" % (ds, tag, N, nq, ng_tot, nrel, cells))
    if ds == "metaqa":
        st = np.load(os.path.join(D.OUT, "kscale_metaqa__v1.npz"))
        assert (st["rows"][:nq] == pop.rows).all()
        BP_ST = {K: st["CNTA__PHG_k%d" % K][:nq, 1] for K in cells}
    else:
        cz = np.load(R.CALIB)
        BP_ST = {K: cz["BP__PHG__k%d__own" % K][:nq] for K in cells}
    Qu = D.unit_queries(cd, pop.rows)
    G = {}
    for K in cells:
        G[K] = {"rank": {rn: np.full(ng_tot, -1, np.int32) for rn in ROUTERS}, "flags": np.zeros(ng_tot, np.uint8),
                "allc": {rn: np.zeros(nq, bool) for rn in ROUTERS}, "ngb": np.zeros(nq, np.int32), "bp": np.zeros(nq, np.int32)}
    H100 = maps[K_REF]
    wseed = 1.0 / np.arange(1.0, ACT + 1.0)
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
                bm = np.bincount(hard, weights=sm, minlength=K)
                hm = np.bincount(hard[top], weights=wseed[:nh], minlength=K)
                pair, rel, c = QT[K]
                Qw = np.bincount(pair, weights=c * w[rel], minlength=K * K).reshape(K, K)
                Pw = sym_norm(Qw)
                Pu = QU[K]
                SC = {"ES_m": bm, "QSu": hm @ Pu, "QSr": hm @ Pw, "QMr": bm @ Pw}
                RK = {"ES": rk_es}
                for rn, sc in SC.items():
                    o = np.lexsort((rk_es, -sc))
                    rk = np.empty(K, np.int64)
                    rk[o] = np.arange(K)
                    RK[rn] = rk
                for rn, parts in (("ES_qs", ("ES", "ES_m", "QSr")), ("ES_qm", ("ES", "ES_m", "QMr"))):
                    sc = sum(1.0 / (K0 + RK[p]) for p in parts)
                    o = np.lexsort((rk_es, -sc))
                    rk = np.empty(K, np.int64)
                    rk[o] = np.arange(K)
                    RK[rn] = rk
                for rn in ROUTERS:
                    Gk["rank"][rn][sl] = RK[rn][gb]
                    Gk["allc"][rn][j] = bool((RK[rn][gb] < b).all())
                hit_b = np.bincount(hard[top], minlength=K) > 0
                adj_b = (Pu[hit_b].sum(0) > 0) | hit_b
                Gk["flags"][sl] = hit_b[gb].astype(np.uint8) | (adj_b[gb].astype(np.uint8) << 1)
        if (j1 // 50) != (j0 // 50):
            log("  rows %d / %d (%.0fs)" % (j1, nq, time.time() - t0))
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
            e = {"gold_nodes": int(mg.sum()), "questions": int(mq.sum()), "mean_B_P": round(float(Gk["bp"][mq].mean()), 2),
                 "random_contact_share_B_P_over_K": round(float((Gk["bp"][mq] / K).mean()), 4),
                 "gold_block_is_HIT_block": round(float((Gk["flags"][mg] & 1).mean()), 4),
                 "gold_block_is_HIT_or_quotient_adjacent": round(float(((Gk["flags"][mg] >> 1) & 1).mean()), 4)}
            for rn in ROUTERS:
                rk = Gk["rank"][rn][mg]
                e["gold_block_contacted_%s" % rn] = round(float((rk < bp_g[mg]).mean()), 4)
                e["question_ALL_gold_blocks_contacted_%s" % rn] = round(float(Gk["allc"][rn][mq].mean()), 4)
                e["gold_block_rank_over_B_P_p50_p90_%s" % rn] = [round(float(np.median(rk / bp_g[mg])), 3), round(float(np.percentile(rk / bp_g[mg], 90)), 3)]
            S_[str(h)] = e
        summ[str(K)] = S_
    res = {"mode": "L1X_BLOCK_QUOTIENT_COLOCATION_DIAGNOSTIC (development; descriptive; nothing chosen)", "definitions": __doc__, "dataset": ds, "tag": tag, "N": N, "n_rows": nq, "gold_nodes": ng_tot,
           "n_relations": nrel, "cells": [str(K) for K in cells], "routers": ROUTERS, "hops": Hs, "summary": summ,
           "seconds": round(time.time() - t_all, 1), "code": {"path": D.rel(me), "sha256": D.sha_file(me)}, "pinned": D.PINNED}
    if rows_limit is not None:
        log("SMOKE (--rows): B_P identities passed; not writing")
        print(json.dumps({K: summ[K]["all"] for K in list(summ)[:1]}, indent=1)[:2400])
        return
    arrays = {"rows": pop.rows, "gptr": gptr, "hops_g": hops_g}
    for K in cells:
        arrays["ngb__%d" % K] = G[K]["ngb"]
        arrays["bp__%d" % K] = G[K]["bp"]
        arrays["flags__%d" % K] = G[K]["flags"]
        for rn in ROUTERS:
            arrays["rank__%d|%s" % (K, rn)] = G[K]["rank"][rn]
            arrays["allc__%d|%s" % (K, rn)] = G[K]["allc"][rn]
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
