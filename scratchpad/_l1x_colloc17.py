"""L1X -- SELECT lane, step 16: COLLOC LADDER = block evidence -> node CO-LOCATION prior (development only; dataset-agnostic typed-graph switch; ONE propagation step, NO traversal).
User direction 2026-10-04: 94-100 % of MetaQA questions fit every gold in the B_P contacted blocks, but the one-hop router contacts the hop-2 gold block only ~40 %: L1 headroom is the relevance already inside the
contacted blocks.  A NON-recursive promotion channel projects the block evidence back to nodes,  C(u|q) = B(P(u)|q) * R(u|q),  and is fused with the node order S0 (S0 is never replaced):
  B(P)   = BM(P), the relation-conditioned one-hop mass the block receives from the FLAT hits (step 11)               [COLLOC-B == the frozen candidate's BM view: an identity check]
  R(u)   = sum over the DISTINCT relations r incident to u (in or out) of w(q, r)  (w = 1 / (1 + rank of r by dense cosine): the step-5 query relation mass; nothing fitted)
  Rs(u)  = the same with each term divided by log(2 + deg_r(u))  (specificity-normalised)
  D(P)   = independent-source diversity of the block's mass = participation ratio (sum_h a_h)^2 / sum_h a_h^2 over the FLAT seeds h, a_h = the mass seed h sends into P  (1 = one hub seed, 3 = three equal seeds)
  mechanisms (the ladder; 5 interpretable arms, no search):  B = B      R = B*R      Rs = B*Rs      D = B*D      RD = B*D*R
  fusion (each mechanism x each fusion):
    F4    equal-weight RRF (K0 = 60) of Sro, Sri, Tin and the channel                                   (B in F4 == 'Sro+Sri+Tin+BM', the candidate)
    F2    RRF of the typed-rule order T = RRF(Sro, Sri, Tin) and the channel                              (the channel is half the vote)
    PROT  conservative: the top ceil(B_N/2) of T are PROTECTED, the remaining B_N - ceil(B_N/2) slots are filled in the F2 order
  routers: ES (shipped) and ES_m (the candidate router); count B_P(q) is the shipped one (asserted per row); the served candidate set is every node of the contacted blocks.
Identity (always on): the shipped, 'Sro+Sri+Tin', 'Sro+Sri+Tin+BM' and 'Sro+Sri+BRin+BM' arms must equal the step-12 record blockrelroute_<ds>__v1 cell for cell (rows are the same).
Text datasets (one relation) are not run: the typed switch is off, so they are byte-identical to the shipped L1 by construction.
  python -u scratchpad/_l1x_colloc17.py RUN <metaqa|webqsp> <tag> [--rows=K]      -> results/L1_X/colloc17_<ds>__<tag>.{json,npz}  (write-once)

STEP 12 (parent): ROUTER ORDER with block relation support (development only; dataset-agnostic).  Step 11 (_l1x_blockrel.py) showed the block views lift the served order and that the
shipped router now binds (gold blocks all contacted: .65-.70).  Here ONLY the order of the partitions is varied (count B_P(q) is the shipped one, asserted per row):
  ES     shipped (first appearance in O' = RRF(H_q, LOC))
  ES_m   blocks by BM desc (the relation-conditioned evidence mass the block receives), ties -> ES
  ES_mr  equal-weight RRF (K0 = 60) of the block ranks ES, BM and BRin, ties -> ES
and the served rule is one of a FIXED short list (the step-11 table rules; nothing re-selected).

STEP 11 DOCSTRING FOLLOWS (unchanged ingredients):
L1X step 11: BLOCK RELATION SUPPORT as one more equal-weight view of the typed-graph switch
(User ruling 2026-10-01: L1 = parameter-free routing and SELECTION only -- no traversal, no fitted or learned rule.  User 2026-10-04: relation support features for KB
datasets -- if the relations of a partition / block match the relations of the query, that block should benefit -- dataset-agnostic, "just another feature to add if the
dataset has it".)

Switch (build-time, no constant): the view exists iff the structural family carries > 1 relation label (n_relations > 1) and a relation-label encoding exists
(results/L3_DEV/relenc_<ds>__v1.npz).  Text datasets (one relation) are not run and stay byte-identical to the shipped L1.

Ingredients (all static, query-independent, computed once per partition map; the only query-side input is the step-5 relation weight w(q, r) = 1 / (1 + rank of relation r by
dense cosine of the query with the label encoding), i.e. nothing is fitted):
    E_in[b, r]   number of structural edges INTO a node of block b with relation r        E_out[b, r]  the same for edges OUT of a node of b
    lift[b, r]   (E[b, r] / sum_r E[b, r]) / (global share of relation r)                  (enrichment of r in the block over the graph: "this block is about r")
Per query q, per block b (all blocks of the map):
    BRin(b)   = sum_r w(q, r) lift_in[b, r]       BRout(b) = sum_r w(q, r) lift_out[b, r]
    BM(b)     = sum over nodes u of b of ( Sro(u) + Sri(u) )   the relation-conditioned one-hop evidence mass the block receives from the FLAT hits (step-5 views)
The per-NODE view of a block feature is the value of the node's block (ties inside a block -> the shipped order).

Candidate set (the new part): the nodes of the CONTACTED blocks (shipped router ES, shipped count B_P(q), asserted per row), in the shipped order; all of them are rankable (the earlier steps
re-ranked only the pool O[:5000] and kept the shipped tail).  Views over those nodes: ship (shipped position), Sro, Sri, Tin, Tout (step 5), BRin, BRout, BM.  A rule = the equal-weight
reciprocal-rank fusion (K0 = 60; ties -> the shipped position) of a subset of 1..4 views; 162 rules + the shipped order.  Nothing is chosen here.

Identity (always on): per row B_P(q) == the stored record and the shipped order's routed ALL equals the stored verdict at every B_N (as in _l1x_rrt.py).

  python -u scratchpad/_l1x_colloc17.py RUN <metaqa|webqsp> <tag> [--rows=K]      -> results/L1_X/blockrelroute_<ds>__<tag>.{json,npz}  (write-once)
  python -u scratchpad/_l1x_colloc17.py SHOW <ds> <tag> [rule ...]                 (prints; no record)
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
import _l1x_rrt as R

log = D.log
K0, ACT, POOL = D.K0, D.ACT, FE.POOL
OUT = R.OUT
MC = D.M_CURVE
NM = len(MC)
VIEWS = ["ship", "Sro", "Sri", "Tin", "Tout", "BRin", "BRout", "BM", "T", "CR", "CRs", "CD", "CRD", "RN", "RB"]
CELLS, K_REF, ALPHA = R.CELLS, R.K_REF, R.ALPHA


ROUTERS = ["ES", "ES_m", "ES_md", "ES_cov"]
CH = ["BM", "CRD", "RN", "RB"]                                         # channels: COLLOC-B, -RD, node relation compatibility, within-block-normalised relation compatibility
RULES = [(), ("Sro", "Sri", "Tin"), ("Sro", "Sri", "BRin", "BM")] + [("Sro", "Sri", "Tin", c) for c in CH] + [("T", "CRD"), ("T", "RB")]
PROT_C = []


def cov_order(A_, b, base, rkm):
    """greedy capacity-capped coverage of every seed's one-hop block mass: pick the block with the largest covered mass sum_h min(A[h, blk], rem_h) (ties -> the better BM rank), rem_h -= covered;
    the first b picks, then the remaining blocks in the base (BM) order.  One propagation step; no parameters."""
    nz = np.flatnonzero(A_.sum(0) > 0)
    Z = A_[:, nz]
    rem = A_.sum(1).copy()
    taken = np.zeros(len(nz), bool)
    out = []
    rz = rkm[nz]
    for _ in range(min(b, len(nz))):
        g = np.minimum(Z, rem[:, None]).sum(0)
        g[taken] = -1.0
        mx = g.max()
        if mx <= 0:
            break
        cand = np.flatnonzero(g >= mx * (1.0 - 1e-12))
        i = int(cand[np.argmin(rz[cand])])
        taken[i] = True
        out.append(int(nz[i]))
        rem = np.maximum(rem - Z[:, i], 0.0)
    out = np.asarray(out, np.int64)
    rest = base[~np.isin(base, out)]
    return np.concatenate([out, rest])


def rinv(v):
    o = np.argsort(-v, kind="stable")
    rk = np.empty(len(v), np.int64)
    rk[o] = np.arange(len(v))
    return 1.0 / (K0 + rk)


def profc(key, rel, N, nrel):
    """CSR of the DISTINCT relations of every node with the edge count of each (node, relation): (ptr, rels, cnt)"""
    u, c = np.unique(key * nrel + rel, return_counts=True)
    node, r = u // nrel, u % nrel
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(np.bincount(node, minlength=N))
    return ptr, r.astype(np.int32), c.astype(np.int32)


def rule_list():
    return list(RULES)


def rsum(ptr, rels, cnts, nodes, spec):
    """sum over the DISTINCT relations incident to every node of the weight w(q, r) -- w is set per row in the caller's scope through the module global _W"""
    s2 = ptr[nodes]
    ln = ptr[nodes + 1] - s2
    out = np.zeros(len(nodes))
    if ln.sum() > 0:
        h_, p_ = E.entries(s2, ln)
        ww = _W[0][rels[p_]]
        if spec:
            ww = ww / np.log(2.0 + cnts[p_])
        out = np.bincount(h_, weights=ww, minlength=len(nodes))
    return out


_W = [None]


def block_tables(hard, s, d, r, K, nrel):
    """lift_in, lift_out (K, nrel) float32 -- enrichment of every relation in every block (incoming / outgoing edges)"""
    out = []
    for node in (d, s):
        E_ = np.bincount(hard[node] * nrel + r, minlength=K * nrel).reshape(K, nrel).astype(np.float64)
        tot = E_.sum(1, keepdims=True)
        p = E_ / np.maximum(tot, 1.0)
        glob = E_.sum(0) / max(E_.sum(), 1.0)
        out.append((p / np.maximum(glob, 1e-12)[None, :]).astype(np.float32))
    return out[0], out[1]


def run():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 3 and a[0] == "RUN", __doc__
    ds, tag = a[1], a[2]
    rows_limit = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--rows=")), None)
    fo = os.path.join(OUT, "colloc17_%s__%s" % (ds, tag))
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
    F, _ = FE.families(cd, N)
    s, d, r, ent = cd.family("structural")
    s, d, r = np.asarray(s, np.int64), np.asarray(d, np.int64), np.asarray(r, np.int64)
    nrel = len(ent.get("relation_vocabulary") or [])
    assert nrel > 1, "typed-graph switch: the view needs a structural family with more than one relation"
    cells = CELLS[ds]
    maps = R.load_maps(cd, ds)
    LIFT = {K: block_tables(maps[K], s, d, r, K, nrel) for K in cells}
    xO, aO, rO = RS.csr_by(s, d, r, N)
    xI, aI, rI = RS.csr_by(d, s, r, N)
    pIn, relIn, cIn = profc(d, r, N, nrel)
    pOut, relOut, cOut = profc(s, r, N, nrel)
    del s, d, r
    RM = RS.relation_matrix(ds, nrel)
    tabs = {K: KS.mult_tables(K)[0][ALPHA] for K in cells}
    rules = rule_list()
    names = ["SHIPPED" if not r_ else "+".join(r_) for r_ in rules] + ["PROT:T+" + c for c in PROT_C]
    nr, nrules = len(names), len(rules)
    RULEM = np.zeros((nrules, len(VIEWS)), np.float64)
    for ri, r_ in enumerate(rules):
        for v in r_:
            RULEM[ri, VIEWS.index(v)] = 1.0
    log("L1X blockrel %s %s: N %d, %d rows, %d relations, cells %s, %d rules" % (ds, tag, N, nq, nrel, cells, nr))
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
    CK = [(K, rn) for K in cells for rn in ROUTERS]
    ALL = {c: np.zeros((nr, nq, NM), bool) for c in CK}
    BPQ = {K: np.zeros(nq, np.int32) for K in cells}
    SIG = {K: {n_: np.zeros(nq, np.float32) for n_ in ("dpr1", "dprb", "marg", "ent", "agree1", "agreeb", "agreec", "relconc")} for K in cells}
    CSZ = {c: np.zeros(nq, np.int64) for c in CK}                    # contacted nodes
    CONT = {c: np.zeros(nq, bool) for c in CK}                       # all gold blocks contacted
    H100 = maps[K_REF]
    wseed = 1.0 / np.arange(1.0, ACT + 1.0)
    MCa = np.array(MC)
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
            vq = np.union1d(top, Lord)
            fq = np.zeros(len(vq))
            tq = np.zeros(len(vq))
            fq[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
            il = np.searchsorted(vq, Lord)
            fq[il] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
            tq[il] = ACT + np.arange(len(Lord))
            tq[np.searchsorted(vq, top)] = np.arange(nh)
            oq = vq[np.lexsort((tq, -fq))]
            # ---- relation-conditioned one-hop views over ALL nodes (step 5)
            cs = RM @ Qu[j].astype(np.float32)
            rr = np.empty(nrel, np.int64)
            rr[np.argsort(-cs, kind="stable")] = np.arange(nrel)
            w = 1.0 / (1.0 + rr)
            _W[0] = w
            xh = wseed[:nh] * F["STRUCT_out"]["g"][top]
            xhi = wseed[:nh] * F["STRUCT_in"]["g"][top]
            s_ = xO[top]
            hO, pO = E.entries(s_, xO[top + 1] - s_)
            sro = np.bincount(aO[pO].astype(np.int64), weights=xh[hO] * w[rO[pO].astype(np.int64)], minlength=N)
            s_ = xI[top]
            hI, pI = E.entries(s_, xI[top + 1] - s_)
            sri = np.bincount(aI[pI].astype(np.int64), weights=xhi[hI] * w[rI[pI].astype(np.int64)], minlength=N)
            sm = sro + sri
            nO_, vO_ = aO[pO].astype(np.int64), xh[hO] * w[rO[pO].astype(np.int64)]
            nI_, vI_ = aI[pI].astype(np.int64), xhi[hI] * w[rI[pI].astype(np.int64)]

            def tmax(ptr, rels, nodes):
                s2 = ptr[nodes]
                ln = ptr[nodes + 1] - s2
                out = np.zeros(len(nodes))
                if ln.sum() > 0:
                    h_, p_ = E.entries(s2, ln)
                    np.maximum.at(out, h_, w[rels[p_]])
                return out

            _rm = np.bincount(np.concatenate([rO[pO], rI[pI]]).astype(np.int64), weights=np.concatenate([xh[hO] * w[rO[pO].astype(np.int64)], xhi[hI] * w[rI[pI].astype(np.int64)]]), minlength=nrel)
            relconc = float(_rm.max() / max(_rm.sum(), 1e-300))
            # ---- routing (shipped, not varied)
            Cm = np.bincount(hit * K_REF + H100[u], weights=x, minlength=nh * K_REF).reshape(nh, K_REF)
            kn_, npz_ = RT.knee(wseed[:nh] @ (Cm > 0))
            b0 = kn_ if npz_ else K_REF
            for K in cells:
                hard = maps[K]
                b = K if not npz_ else int(tabs[K][b0])
                assert b == BP_ST[K][j], "B_P differs from the stored record (K %d row %d: %d vs %d)" % (K, j, b, BP_ST[K][j])
                BPQ[K][j] = b
                hq = hard[oq]
                up, first = np.unique(hq, return_index=True)
                fe = np.full(K, len(oq), np.int64)
                fe[up] = first
                ro_es = np.lexsort((np.arange(K), fe))
                rk_es = np.empty(K, np.int64)
                rk_es[ro_es] = np.arange(K)
                lin, lout = LIFT[K]
                bm = np.bincount(hard, weights=sm, minlength=K)
                key_ = np.concatenate([hO * K + hard[nO_], hI * K + hard[nI_]])
                A_ = np.bincount(key_, weights=np.concatenate([vO_, vI_]), minlength=nh * K).reshape(nh, K)
                bmA = A_.sum(0)
                assert np.allclose(bmA, bm, rtol=1e-6, atol=1e-9), "block mass from the (seed, block) table differs from BM"
                Dpr = np.divide(bmA ** 2, (A_ ** 2).sum(0), out=np.zeros(K), where=bmA > 0)
                brin = lin @ w
                o_m = np.lexsort((rk_es, -bm))
                rk_m = np.empty(K, np.int64)
                rk_m[o_m] = np.arange(K)
                o_b = np.lexsort((rk_es, -brin))
                rk_b = np.empty(K, np.int64)
                rk_b[o_b] = np.arange(K)
                sc = 1.0 / (K0 + rk_es) + 1.0 / (K0 + rk_m) + 1.0 / (K0 + rk_b)
                o_md = np.lexsort((rk_es, -(bm * Dpr)))
                o_cov = cov_order(A_, b, o_m, rk_m)
                RO = {"ES": ro_es, "ES_m": o_m, "ES_mr": np.lexsort((rk_es, -sc)), "ES_md": o_md, "ES_cov": o_cov}
                tb = o_m[:b]
                bmt = bm[tb]
                pb = bm / max(bm.sum(), 1e-300)
                SIG[K]["dpr1"][j] = Dpr[o_m[0]]
                SIG[K]["dprb"][j] = float((bmt * Dpr[tb]).sum() / max(bmt.sum(), 1e-300))
                SIG[K]["marg"][j] = (1.0 - bm[o_m[1]] / bm[o_m[0]]) if K > 1 and bm[o_m[0]] > 0 else 0.0
                SIG[K]["ent"][j] = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum() / np.log(K)) if K > 1 else 0.0
                SIG[K]["agree1"][j] = float(o_m[0] == o_md[0])
                SIG[K]["agreeb"][j] = float(len(np.intersect1d(o_m[:b], o_md[:b])) / b)
                SIG[K]["agreec"][j] = float(len(np.intersect1d(o_m[:b], o_cov[:b])) / b)
                SIG[K]["relconc"][j] = relconc
                for rn in ROUTERS:
                    cm = np.zeros(K, bool)
                    cm[RO[rn][:b]] = True
                    if not cm[hard[g]].all():
                        continue                                      # a gold partition is not contacted: not served at any B_N
                    CONT[(K, rn)][j] = True
                    C = FO[cm[hard[FO]]]                              # contacted nodes in the shipped order
                    m = len(C)
                    CSZ[(K, rn)][j] = m
                    posC = np.full(N, -1, np.int64)
                    posC[C] = np.arange(m)
                    gi = posC[g]
                    hb = hard[C]
                    raw = {"ship": -np.arange(m, dtype=np.float64), "Sro": sro[C], "Sri": sri[C], "Tin": tmax(pIn, relIn, C), "Tout": tmax(pOut, relOut, C),
                           "BRin": brin[hb], "BRout": (lout @ w)[hb], "BM": bm[hb]}
                    raw["T"] = rinv(raw["Sro"]) + rinv(raw["Sri"]) + rinv(raw["Tin"])
                    Rn = rsum(pIn, relIn, cIn, C, False) + rsum(pOut, relOut, cOut, C, False)
                    Rs = rsum(pIn, relIn, cIn, C, True) + rsum(pOut, relOut, cOut, C, True)
                    raw["CR"], raw["CRs"] = bm[hb] * Rn, bm[hb] * Rs
                    raw["CD"], raw["CRD"] = bm[hb] * Dpr[hb], bm[hb] * Dpr[hb] * Rn
                    bmx = np.zeros(K)
                    np.maximum.at(bmx, hb, Rn)
                    raw["RN"], raw["RB"] = Rn, Rn / np.maximum(bmx[hb], 1e-12)
                    INV = np.empty((len(VIEWS), m))
                    for vi, n_ in enumerate(VIEWS):
                        o = np.argsort(-raw[n_], kind="stable")
                        rk = np.empty(m, np.int64)
                        rk[o] = np.arange(m)
                        INV[vi] = 1.0 / (K0 + rk)
                    S = RULEM @ INV                                   # (nr, m); the shipped rule is all-zero -> ties -> the shipped order
                    O = np.argsort(-S, axis=1, kind="stable")
                    NI = np.empty((nrules, m), np.int32)
                    NI[np.arange(nrules)[:, None], O] = np.arange(m, dtype=np.int32)[None, :]
                    mx = NI[:, gi].max(axis=1)
                    ALL[(K, rn)][:nrules, j] = mx[:, None] < MCa[None, :]
                    rkT = np.empty(m, np.int64)
                    rkT[np.argsort(-raw["T"], kind="stable")] = np.arange(m)
                    iT = INV[VIEWS.index("T")]
                    for pi, cn in enumerate(PROT_C):
                        O2 = np.argsort(-(iT + INV[VIEWS.index(cn)]), kind="stable")
                        P2 = np.empty(m, np.int64)
                        P2[O2] = np.arange(m)
                        for mi, mb in enumerate(MC):
                            h_ = (mb + 1) // 2
                            prot = rkT < h_
                            posnp = np.cumsum(~prot[O2]) - 1
                            ALL[(K, rn)][nrules + pi, j, mi] = bool((prot[gi] | (posnp[P2[gi]] < mb - h_)).all())
        log("  rows %d / %d (%.0fs)" % (j1, nq, time.time() - t0))
    ident = {}
    for K in cells:
        bad = int((SV_ST[K] != ALL[(K, "ES")][0]).sum())
        ident[str(K)] = {"mismatching_(row,B_N)_cells": bad, "of": int(SV_ST[K].size)}
        assert bad == 0, "shipped routed ALL differs from the stored verdict (K %d: %d cells)" % (K, bad)
    log("identities: " + json.dumps(ident))
    prev = np.load(os.path.join(OUT, "blockrelroute_%s__v1.npz" % ds))
    pj = json.load(open(os.path.join(OUT, "blockrelroute_%s__v1.json" % ds), encoding="utf-8"))
    assert (prev["rows"][:nq] == pop.rows).all()
    nbad = 0
    for c in CK:
        if c[1] not in ("ES", "ES_m"):
            continue
        pa = np.unpackbits(prev["ALL__%d|%s" % c], axis=2)[:, :, :NM].astype(bool)
        for nm in ("SHIPPED", "Sro+Sri+Tin", "Sro+Sri+BRin+BM"):
            nbad += int((pa[pj["rules"].index(nm)][:nq] != ALL[(c[0], c[1])][names.index(nm)]).sum())
        nbad += int((pa[pj["rules"].index("Sro+Sri+Tin+BM")][:nq] != ALL[(c[0], c[1])][names.index("Sro+Sri+Tin+BM")]).sum())
    log("identity vs the step-12 record: %d mismatching cells" % nbad)
    assert nbad == 0, "the re-implemented step-12 arms differ from blockrelroute_%s__v1" % ds
    A, B = (pop.rows % 2) == 0, (pop.rows % 2) == 1
    hops = np.asarray(pop.hops) if getattr(pop, "hops", None) is not None else None
    Hs = sorted(set(hops.tolist())) if hops is not None else []
    res = {"mode": "L1X_COLLOC_LADDER (development; descriptive; nothing chosen)", "definitions": __doc__, "dataset": ds, "tag": tag, "N": N, "n_rows": nq,
           "n_relations": nrel, "cells": ["%d|%s" % c for c in CK], "routers": ROUTERS, "rules": names, "views": VIEWS, "budgets": list(MC), "identity_vs_stored": ident,
           "rows_A": int(A.sum()), "rows_B": int(B.sum()), "mean_B_P": {str(K): round(float(BPQ[K].mean()), 2) for K in cells},
           "gold_blocks_all_contacted": {"%d|%s" % c: round(float(CONT[c].mean()), 4) for c in CK},
           "mean_contacted_nodes": {"%d|%s" % c: round(float(CSZ[c].mean()), 1) for c in CK}, "hops": Hs, "table": {}}
    for c in CK:
        T = {}
        for ri, nm in enumerate(names):
            e = {"A": [round(float(ALL[c][ri, A, mi].mean()), 4) for mi in range(NM)], "B": [round(float(ALL[c][ri, B, mi].mean()), 4) for mi in range(NM)]}
            if len(Hs) > 1:
                e["hopA"] = {str(h): [round(float(ALL[c][ri, A & (hops == h), mi].mean()), 4) for mi in range(NM)] for h in Hs}
                e["hopB"] = {str(h): [round(float(ALL[c][ri, B & (hops == h), mi].mean()), 4) for mi in range(NM)] for h in Hs}
            T[nm] = e
        res["table"]["%d|%s" % c] = T
    res["seconds"] = round(time.time() - t_all, 1)
    res["code"] = {"path": D.rel(me), "sha256": D.sha_file(me)}
    res["pinned"] = D.PINNED
    if rows_limit is not None:
        log("SMOKE (--rows): identities passed; not writing")
        return
    arrays = {"rows": pop.rows}
    for c in CK:
        arrays["ALL__%d|%s" % c] = np.packbits(ALL[c], axis=2)
    for K in cells:
        arrays["BPQ__%d" % K] = BPQ[K]
        for n_, v_ in SIG[K].items():
            arrays["SIG__%d__%s" % (K, n_)] = v_
    np.savez_compressed(fo + ".npz", **arrays)
    res["npz"] = {"path": os.path.basename(fo + ".npz"), "sha256": D.sha_file(fo + ".npz")}
    D.G.S.wj(fo + ".json", res)
    log("done (%.0fs) -> %s" % (time.time() - t_all, fo))


def show(ds, tag, want):
    rec = json.load(open(os.path.join(OUT, "colloc17_%s__%s.json" % (ds, tag)), encoding="utf-8"))
    cells = rec["cells"]
    print("dataset %s  rows A/B %d/%d  mean B_P %s  gold blocks all contacted %s  mean contacted nodes %s" % (ds, rec["rows_A"], rec["rows_B"], rec["mean_B_P"], rec["gold_blocks_all_contacted"], rec["mean_contacted_nodes"]))
    obj = lambda v: float(np.mean(v[:4]))
    names = list(rec["table"][cells[0]].keys())
    byA = sorted(names, key=lambda n: -np.mean([obj(rec["table"][c][n]["A"]) for c in cells]))
    show_n = ["SHIPPED", "Sro+Sri+Tin"] + [n for n in want] + byA[:8]
    seen = []
    for n in show_n:
        if n in rec["table"][cells[0]] and n not in seen:
            seen.append(n)
    print("ALL-gold at B_N", rec["budgets"], " (half B; objective = mean over A over the first four B_N and the cells)")
    for n in seen:
        oa = float(np.mean([obj(rec["table"][c][n]["A"]) for c in cells]))
        ob = float(np.mean([obj(rec["table"][c][n]["B"]) for c in cells]))
        print("  %-34s objA %.4f objB %.4f" % (n, oa, ob))
        for c in cells:
            print("      K%-5s B %s" % (c, rec["table"][c][n]["B"][:4]))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a and a[0] == "RUN":
        run()
    elif a and a[0] == "SHOW":
        show(a[1], a[2], a[3:])
    else:
        raise SystemExit(__doc__)
