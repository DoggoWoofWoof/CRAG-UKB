"""L1X -- SELECT lane, step 5: one-hop RELATION-CONDITIONED selection features of the carried-forward L1 (development only).
(User ruling 2026-10-01: L1 = parameter-free routing and SELECTION only -- no traversal, no fitted or learned rule.  Every feature below uses the SINGLE edge
from a hit to a candidate, or the static relation profile of the candidate itself: one application of A^T, nothing is walked, no second hop.)

The only new ingredient is a question-to-relation score: the cosine of the (unit, instruction-prefixed) dense query vector of the row with the dense encoding
of each relation label (results/L3_DEV/relenc_<ds>__v1.npz: the canonical dense encoder, labels verbalised with '_' and '.' -> ' ', encoded as DOCUMENTS; the
encodings of the earlier L3 lane, reused unchanged).  rr(q, r) = the 0-based rank of relation r among ALL relations of the graph by that cosine, and the
relation weight is w(q, r) = 1 / (1 + rr(q, r))  -- the same 1 / (rank + 1) convention as the hit weights; no constant is fitted, no threshold.

Per pool node u of a row (the pool P_q = O[:5000] of the shipped order, unchanged):
    Sro   sum over STRUCT_out edges hit -> u (relation r) of x_hit * w(q, r)              x_hit = 1 / (hit rank + 1) * g(hit), the shipped hit weight
    Sri   the same over STRUCT_in edges (the stored edge u -> hit carries relation r)
    Tin   max over the relations r of the edges INTO u of w(q, r)     (the static in-relation profile of u: what u is the object of)
    Tout  max over the relations r of the edges OUT of u of w(q, r)    (what u is the subject of)
and, as in _l1x_relorc.py, frank / Ssum / S_out / S_in / the gold flag, so the offline search needs no other file.  Diagnostics per row: the number of
oracle relations R*_q (relations of edges hit -> gold or gold -> hit) and the best rr(q, .) among them (how high the matcher puts a right relation).

Usage: python -u scratchpad/_l1x_relsel.py RUN <dataset> <tag> [--rows=K]  -> results/L1_X/relsel_<ds>__<tag>.{json,npz} (write-once)
"""
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
COLN = ["frank", "Ssum", "S_out", "S_in", "Sro", "Sri", "Tin", "Tout", "gold"]
NC = len(COLN)


def csr_by(key, other, rel, N):
    o = np.argsort(key, kind="stable")
    xadj = np.zeros(N + 1, np.int64)
    xadj[1:] = np.cumsum(np.bincount(key, minlength=N))
    return xadj, other[o].astype(np.int32), rel[o].astype(np.int32)


def profile(key, rel, N, nrel):
    """CSR of the DISTINCT relations of every node: (ptr, rels) from the keys (node, relation)."""
    u = np.unique(key * nrel + rel)
    node, r = u // nrel, u % nrel
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(np.bincount(node, minlength=N))
    return ptr, r.astype(np.int32)


def relation_matrix(ds, nrel):
    z = np.load(os.path.join(D.REPO, "results", "L3_DEV", "relenc_%s__v1.npz" % ds), allow_pickle=True)
    kind, r1 = z["kind"], z["r1"]
    sel = np.flatnonzero(kind == 0)
    assert len(sel) == nrel and (r1[sel] == np.arange(nrel)).all(), "the single-relation encodings do not cover the vocabulary in order"
    M = z["dense"][sel].astype(np.float32)
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 3 and a[0] == "RUN", __doc__
    ds, tag = a[1], a[2]
    rows_limit = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--rows=")), None)
    fo = os.path.join(OUT, "relsel_%s__%s" % (ds, tag))
    assert not os.path.exists(fo + ".json"), "write-once: %s exists" % fo
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
    xO, aO, rO = csr_by(s, d, r, N)                              # edges OUT of a node: (node -> aO, relation)
    xI, aI, rI = csr_by(d, s, r, N)                              # edges INTO a node: (aI -> node, relation)
    pIn, relIn = profile(d, r, N, nrel)
    pOut, relOut = profile(s, r, N, nrel)
    del s, d, r
    RM = relation_matrix(ds, nrel)
    log("L1X relsel %s %s: N %d, %d rows, %d relations" % (ds, tag, N, nq, nrel))
    Qu = D.unit_queries(cd, pop.rows)
    stored = None
    fs = os.path.join(D.OUT, "kscale_%s__v1.npz" % ds)
    if os.path.exists(fs):
        stored = np.load(fs)
        assert (stored["rows"][:nq] == pop.rows).all()
    COLS = np.zeros((nq, POOL, NC), np.float32)
    ROWINFO = np.zeros((nq, 4), np.float64)                   # |R*|, best rr among R*, golds adjacent to a hit (either direction), golds
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
            # ---- the question-to-relation weights
            cs = RM @ Qu[j].astype(np.float32)
            rr = np.empty(nrel, np.int64)
            rr[np.argsort(-cs, kind="stable")] = np.arange(nrel)
            w = 1.0 / (1.0 + rr)
            xh = (1.0 / np.arange(1.0, nh + 1.0)) * F["STRUCT_out"]["g"][top]          # shipped hit weight on the STRUCT_out row length
            xhi = (1.0 / np.arange(1.0, nh + 1.0)) * F["STRUCT_in"]["g"][top]
            # STRUCT_out entries of the hits (relation kept)
            st = xO[top]
            hO, pO = E.entries(st, xO[top + 1] - st)
            uO, rrO = aO[pO].astype(np.int64), rO[pO].astype(np.int64)
            sro = np.bincount(uO, weights=xh[hO] * w[rrO], minlength=N)
            st = xI[top]
            hI, pI = E.entries(st, xI[top + 1] - st)
            uI, rrI = aI[pI].astype(np.int64), rI[pI].astype(np.int64)
            sri = np.bincount(uI, weights=xhi[hI] * w[rrI], minlength=N)
            so = np.bincount(u[fam == 0], weights=x[fam == 0], minlength=N)
            si = np.bincount(u[fam == 1], weights=x[fam == 1], minlength=N)

            def tmax(ptr, rels):
                st_ = ptr[pool]
                ln = ptr[pool + 1] - st_
                out = np.zeros(m)
                if ln.sum() > 0:
                    h_, p_ = E.entries(st_, ln)
                    np.maximum.at(out, h_, w[rels[p_]])
                return out

            COLS[j, :m, 0] = np.minimum(frank[pool], FE.CAP)
            COLS[j, :m, 1] = L[pool]
            COLS[j, :m, 2] = so[pool]
            COLS[j, :m, 3] = si[pool]
            COLS[j, :m, 4] = sro[pool]
            COLS[j, :m, 5] = sri[pool]
            COLS[j, :m, 6] = tmax(pIn, relIn)
            COLS[j, :m, 7] = tmax(pOut, relOut)
            COLS[j, :m, 8] = np.isin(pool, g)
            # ---- diagnostics: relations of the hit-gold edges
            gs = np.isin(uO, g)
            gi = np.isin(uI, g)
            Rs = np.unique(np.concatenate([rrO[gs], rrI[gi]]))
            ROWINFO[j] = [len(Rs), (rr[Rs].min() if len(Rs) else -1), int(np.isin(g, np.concatenate([uO, uI])).sum()), len(g)]
            if stored is not None:
                ix = np.full(N, POOL, np.int64)
                ix[pool] = np.arange(m)
                sl = slice(gptr[j], gptr[j + 1])
                assert (ix[g] == np.minimum(stored["pos_FLATLOC__IR_L1"][sl], POOL)).all(), "baseline positions differ from the kscale record (row %d)" % j
        log("  rows %d / %d (%.0fs)" % (j1, nq, time.time() - t0))
    res = {"mode": "L1X_RELATION_CONDITIONED_FEATURES (development; descriptive; nothing chosen)", "definitions": __doc__, "dataset": ds, "tag": tag,
           "N": N, "n_rows": nq, "n_relations": nrel, "pool": POOL, "columns": COLN,
           "identity_check": ("gold positions in the pool == %s (clipped at %d) on every row" % (D.rel(fs), POOL)) if stored is not None else "no stored kscale record",
           "relation_encodings": {"path": "results/L3_DEV/relenc_%s__v1.npz" % ds, "sha256": D.sha_file(os.path.join(D.REPO, "results", "L3_DEV", "relenc_%s__v1.npz" % ds))},
           "oracle_relation_rank": {"rows_with_oracle_relations": int((ROWINFO[:, 0] > 0).sum()),
                                    "share_best_oracle_relation_in_top1": float(((ROWINFO[:, 1] == 0) & (ROWINFO[:, 0] > 0)).sum() / max(1, (ROWINFO[:, 0] > 0).sum())),
                                    "share_in_top3": float(((ROWINFO[:, 1] >= 0) & (ROWINFO[:, 1] < 3)).sum() / max(1, (ROWINFO[:, 0] > 0).sum())),
                                    "share_in_top10": float(((ROWINFO[:, 1] >= 0) & (ROWINFO[:, 1] < 10)).sum() / max(1, (ROWINFO[:, 0] > 0).sum())),
                                    "median_best_rank": float(np.median(ROWINFO[ROWINFO[:, 0] > 0, 1])) if (ROWINFO[:, 0] > 0).any() else None},
           "seconds": round(time.time() - t_all, 1), "code": {"path": D.rel(me), "sha256": D.sha_file(me)}, "pinned": D.PINNED}
    np.savez_compressed(fo + ".npz", cols=COLS, rowinfo=ROWINFO, rows=pop.rows)
    res["npz"] = {"path": os.path.basename(fo + ".npz"), "sha256": D.sha_file(fo + ".npz")}
    D.G.S.wj(fo + ".json", res)
    log("done (%.0fs) -> %s" % (time.time() - t_all, fo))


if __name__ == "__main__":
    main()
