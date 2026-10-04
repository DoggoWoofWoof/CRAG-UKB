"""L1 DEVELOPMENT -- ROUTE2: the second batch of non-parametric L1b partition routers (user 2026-09-27: "test out everything we
can based on the suggestions and based on the failures we have encountered or will encouter and try to mitigate them").
Development numbers; any p-value is descriptive.

Fixed, exactly as _l1d_route.py (route v1; asserted per query): the L1a node score IR_L1, the served order O (IR_L1 FLAT+LOC,
RRF K0 = 60), the cells and their frozen hard shards, the population and the serving rule (the first B_N nodes of O inside the
B_P contacted partitions).  Only the partition ORDER (a ranking) and the partition COUNT (a stopping rule) vary; each is a fixed
function of the query's H_q = FLAT[:200], its IR_L1 1-hop evidence and the static shard map -- no learned weight, no tuned
constant, no gold.  The reference rankings REF_RANKS, the reference counts REF_COUNTS and PSTAR are recomputed and asserted
identical to route v1 (per query), so the two records merge into one factorial in the summary.

Notation: seeds s in H_q with weight w(s) = 1/rank(s); Cm[s, P] = the IR_L1 mass seed s puts into partition P (its 1-hop
entries u in P, each w(s) g_f(s)); m_s = sum_P Cm[s, P]; P(s) = the partition of seed s; FSUM(P) = sum_{s: P(s) = P} w(s).

Round-1 failures and the mitigations run here:
 (1) coverage saturation: a seed counts as covered by ANY partition holding one of its 1-hop entries, so hub partitions cover
     most seeds at once and the greedy stops after 3-30 partitions (route v1: SC_* / FL_* with their own counts lose 0.07-0.53).
     -> NSUM(P) = sum_s w(s) Cm[s, P] / m_s: additive facility location (every seed spreads exactly w(s); modular; never
        saturates; the exact expected coverage when a seed's evidence lies in ONE partition with probability equal to its share).
     -> PC_L / PC_LS: the greedy maximum of the probabilistic (noisy-OR) coverage sum_s w(s) [1 - prod_{P in S} (1 - p(s, P))],
        p = Cm / m_s (PC_L) or p = (Cm + m_s 1[P = P(s)]) / (2 m_s) (PC_LS; a seed without 1-hop mass has p(s, P(s)) = 1): a
        seed's uncovered share decays multiplicatively instead of vanishing at the first touching partition.
 (2) hub partitions collect mass from any seed.
     -> SUMH(P) = SUM(P) / log2(1 + IN(P)), IN(P) = the entries of E (four families) that end in P (static).  FLAGGED: the
        partition-level analogue of the target-degree term that ruling C kept out of the node score.
 (3) localised rankings miss the FLAT seed's own partition on text (SQuAD SUM|NEFF_SUM -0.028 at B_N 1000), FLAT rankings miss
     the answer partitions on the KB (MetaQA F / D / SP / FT3: no gain).
     -> NSUMS = NSUM + FSUM (a seed puts w(s) on its own partition and w(s) spread over its 1-hop), SDIVS = SDIV + FSUM,
        SHF = FSUM / sum FSUM + SUM / sum SUM (equal-share fusion of the two sources),
     -> two-source unions FSUM[:c_F] u X[:c_X], each source ranked by its own mass and counted by its own concentration:
        UF_SUM_NEFF, UF_SUM_KNEE, UF_T3_NEFF, UF_T3_KNEE (X = SUM / TOP3 with NEFF or KNEE on both sides), UF_SD_NEFF (SDIV),
        UF_NS_NEFF (NSUM).  Order: min(rank_F / c_F, rank_X / c_X) (the union is the prefix; its size is the count).
 (4) FLAT junk crowds the KB served list, localised noise the text list.
     -> SL: the partitions with localised evidence (SUM > 0) first, in ES order, then the rest.
     -> AND_T3 / AND_SD: order by max(rank_ES, rank_X) for X = TOP3 / SDIV (both rankings must admit the partition; ties ->
        the smaller rank -> ES).
     -> PF_E_T3 / PF_E_SD (partition RRF, K0 = 60, of ES and X) and IL_E_T3 / IL_E_SD (round-robin interleave, ES first): the
        route-v1 fusions PF_S_T3 / IL_S_T3 rebuilt on the router's information (route v1's factorial ranked IL_S_T3 / PF_S_T3
        with the SDIV counts first by worst-cell distance to the envelope).
 (5) count calibration: NEFF is the Hill number of order 2 only.
     -> EXPH_x (order 1, exp entropy) and BPI_x (order infinity, 1 / the largest share) for every scored ranking; NEFF / KNEE of
        the new scores; NEV = the partitions holding evidence (H_q or L > 0); NOACT_E = the distinct partitions of O'[:200].
 (6) deployability: S needs every node's O rank (a global FLAT order).  A router that has not contacted any shard knows H_q with
     its FLAT ranks, the seeds' 1-hop rows (hence L and the LOC order) and the static maps -- nothing else.
     -> ES = first appearance in O' = RRF(H_q, LOC) (K0 = 60; ties -> seeds by FLAT rank, then LOC rank); partitions without
        evidence after, in static partition-id order.  Every NEW ranking here falls back to ES (not to O) for zero scores, so
        every new ranking and count uses only the router's information.
Not run (each needs a hyperparameter): MMR (lambda), xQuAD (lambda), tau-coverage (tau).
Counts: NEFF_x = ceil(1 / sum p^2), EXPH_x = ceil(exp(-sum p log p)), BPI_x = ceil(1 / max p) over the positive scores of x
(within [1, #positive]); KNEE_x = #{positive scores >= their mean}; NEFFG / KNEEG / COVER over a greedy's marginal gains (COVER =
the number of positive gains); a union's count is its size.  A query without evidence for a score contacts every partition
(section 33 / route v1).
Identities asserted: IR_L1 LOC / FLAT+LOC positions == node1h v1 (all gold nodes); LO / HI / PRG of REF_RANKS, every REF_COUNT
and PSTAR == route v1 (per query, every cell); every partition contacted == unrouted (per row, per ranking); per-seed masses sum
to SUM; NSUM / NSUMS totals; the probabilistic-coverage gains never increase; on the first rows: incremental greedy == a
from-scratch greedy, every ranking a permutation, every union prefix == the union.

Usage: python scratchpad/_l1d_route2.py RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L1_DEV/route2_<dataset>__<tag>.{json,npz} (write-once)
"""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_arms as R
import _l1d_edgediag as E
import _l1d_node1h as NH
import _l1d_adaptbp as AB
import _l1d_route as RT

log = D.log
K0, ACT = D.K0, D.ACT
FAMS = NH.FAMS
NODE = "IR_L1"
TOPK = RT.TOPK
TOL, TOLG, EPS_G, CHECK_ROWS = RT.TOL, RT.TOLG, RT.EPS_G, RT.CHECK_ROWS
REF_RANKS = ("S", "FSUM", "SUM", "TOP3", "SDIV")
NEW_SCORED = ("NSUM", "NSUMS", "SDIVS", "SHF", "SUMH")
ORDERS = ("ES", "SL", "AND_T3", "AND_SD", "PF_E_T3", "IL_E_T3", "PF_E_SD", "IL_E_SD")
PCS = ("PC_L", "PC_LS")
UNIONS = (("UF_SUM_NEFF", "SUM", "NEFF"), ("UF_SUM_KNEE", "SUM", "KNEE"), ("UF_T3_NEFF", "TOP3", "NEFF"),
          ("UF_T3_KNEE", "TOP3", "KNEE"), ("UF_SD_NEFF", "SDIV", "NEFF"), ("UF_NS_NEFF", "NSUM", "NEFF"))
UN = tuple(u for u, _, _ in UNIONS)
RANKS = REF_RANKS + NEW_SCORED + ORDERS + PCS + UN
HILL = ("FSUM", "FSUMFV", "SUM", "MAX", "TOP3", "SDIV") + NEW_SCORED
REF_COUNTS = (tuple("%s_%s" % (p, x) for p in ("NEFF", "KNEE") for x in ("FSUM", "FSUMFV", "SUM", "MAX", "TOP3", "SDIV"))
              + ("EXPH_TOP3", "NSEEDP", "NOACT"))
COUNTS = (REF_COUNTS + tuple("%s_%s" % (p, x) for p in ("NEFF", "KNEE") for x in NEW_SCORED)
          + tuple("EXPH_" + x for x in HILL if x != "TOP3") + tuple("BPI_" + x for x in HILL)
          + tuple("%s_%s" % (c, k) for c in PCS for k in ("NEFFG", "KNEEG", "COVER")) + ("NEV", "NOACT_E") + UN)
NATIVE = ([(x, "%s_%s" % (p, x)) for x in NEW_SCORED for p in ("NEFF", "KNEE", "EXPH", "BPI")]
          + [(x, "%s_%s" % (p, x)) for x in ("SUM", "TOP3", "SDIV") for p in ("EXPH", "BPI")]
          + [(c, "%s_%s" % (c, k)) for c in PCS for k in ("NEFFG", "KNEEG", "COVER")]
          + [(u, u) for u in UN]
          + [("ES", "NOACT_E"), ("ES", "NEV"), ("ES", "NSEEDP"), ("ES", "KNEE_TOP3"), ("ES", "KNEE_SDIV"), ("SL", "NEFF_SUM"),
             ("SL", "KNEE_SUM"), ("AND_T3", "NEFF_TOP3"), ("AND_T3", "KNEE_TOP3"), ("AND_SD", "NEFF_SDIV"), ("AND_SD", "NEFF_SUM"),
             ("PF_E_T3", "KNEE_TOP3"), ("IL_E_T3", "NEFF_SDIV"), ("PF_E_SD", "KNEE_SDIV"), ("IL_E_SD", "NEFF_SDIV")])
ROUTER = "H_q = FLAT[:200] + the seeds' 1-hop rows"
INFO = {"NSUM": ROUTER, "NSUMS": ROUTER, "SDIVS": ROUTER, "SHF": ROUTER, "SUMH": ROUTER + " + static partition in-degree",
        "ES": ROUTER + " (the router's own fused order O')", "SL": ROUTER, "AND_T3": ROUTER, "AND_SD": ROUTER,
        "PF_E_T3": ROUTER, "IL_E_T3": ROUTER, "PF_E_SD": ROUTER, "IL_E_SD": ROUTER, "PC_L": ROUTER, "PC_LS": ROUTER}
INFO.update({u: ROUTER for u in UN})
BP_SHOW = AB.BP_SHOW
MSA = np.asarray(D.M_CURVE, np.int64)
RT_PATH = os.path.join(D.HERE, "_l1d_route.py")
AP_CHUNK = RT.AP_CHUNK
assert len(set(RANKS)) == len(RANKS) and len(set(COUNTS)) == len(COUNTS) and set(INFO) == set(RANKS) - set(REF_RANKS)
assert set(REF_RANKS) <= set(RT.RANKS) and set(REF_COUNTS) <= set(RT.COUNTS)
assert all(r in RANKS and c in COUNTS for r, c in NATIVE) and FAMS == RT.FAMS


def hill(v):
    """(order-1 Hill number exp H, order-infinity Hill number 1 / max p, #positive) of the positive values of v."""
    v = v[v > 0]
    if not len(v):
        return None, None, 0
    p = v / v.sum()
    return float(np.exp(-np.dot(p, np.log(p)))), 1.0 / float(p.max()), len(v)


def pc_greedy(Pm, w):
    """greedy maximum of sum_s w(s) [1 - prod_{P in S} (1 - Pm[s, P])] over the columns of Pm (column order = the tie
    priority), incremental: gain(P) = sum_s w(s) r(s) Pm[s, P] with the residual r(s) = prod over the chosen (1 - Pm[s, .])."""
    nh, T = Pm.shape
    r = np.ones(nh)
    gains = w @ Pm
    alive = np.ones(T, bool)
    sel, G = [], []
    g0 = None
    while alive.any():
        k, gk = RT.pick(gains, alive)
        if g0 is None:
            g0 = gk
        if not (gk > 0 and gk > EPS_G * g0):
            break
        sel.append(k)
        G.append(gk)
        alive[k] = False
        col = Pm[:, k]
        dl = np.flatnonzero(col > 0)
        if len(dl):
            gains -= (w[dl] * r[dl] * col[dl]) @ Pm[dl]
            r[dl] *= 1.0 - col[dl]
    return np.asarray(sel, np.int64), np.asarray(G, np.float64)


def pc_greedy_ref(Pm, w):
    """the same greedy with the gains recomputed from scratch at every step (a check on the first rows)."""
    nh, T = Pm.shape
    r = np.ones(nh)
    alive = np.ones(T, bool)
    sel, G = [], []
    g0 = None
    while alive.any():
        gains = (w * r) @ Pm
        k, gk = RT.pick(gains, alive)
        if g0 is None:
            g0 = gk
        if not (gk > 0 and gk > EPS_G * g0):
            break
        sel.append(k)
        G.append(gk)
        alive[k] = False
        r = r * (1.0 - Pm[:, k])
    return np.asarray(sel, np.int64), np.asarray(G, np.float64)


def union_order(ra, ca, rb, cb, tie):
    """the ranking whose first n entries are A[:ca] u B[:cb] (0-based ranks ra, rb), ordered by min(ra / ca, rb / cb)."""
    ro = np.lexsort((tie, np.minimum(ra / float(ca), rb / float(cb))))
    inU = (ra < ca) | (rb < cb)
    n = int(inU.sum())
    return ro, n, inU


def partition_indegree(F, hard, npart):
    """IN(P) = the entries of the four families of E whose target lies in P (static)."""
    IN = np.zeros(npart, np.int64)
    for f in FAMS:
        adj = F[f]["adj"]
        for c0 in range(0, len(adj), AP_CHUNK):
            IN += np.bincount(hard[adj[c0:c0 + AP_CHUNK].astype(np.int64)], minlength=npart)
    return IN


class Route2Spec(object):
    def __init__(self, cd, pop, parts):
        self.ds, self.parts, self.cells = cd.name, parts, list(parts)
        N = self.N = int(cd.n_nodes)
        nq, ng = pop.nq, pop.ng_tot
        # ---- the route v1 record this harness extends (same code for the shared arithmetic, same population and partitions)
        fr = os.path.join(D.OUT, "route_%s__v1.json" % self.ds)
        RV = json.load(open(fr, encoding="utf-8"))
        self.route_npz = os.path.join(D.OUT, RV["npz"]["path"])
        assert D.sha_file(self.route_npz) == RV["npz"]["sha256"], "route v1 npz changed"
        self.sha_rt = D.sha_file(RT_PATH)
        assert self.sha_rt == RV["code"]["harness"]["sha256"], "_l1d_route.py differs from the code of its v1 record"
        for k_, p_ in RV["structures"]["imports"].items():
            assert D.sha_file(os.path.join(D.REPO, k_)) == p_, "%s differs from route v1's import" % k_
        assert RV["code"]["arms"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_arms.py"))
        assert RV["code"]["lib"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))
        n1 = RV["structures"]["reproduces"]["node1h"]
        self.node1h_npz = os.path.join(D.OUT, json.load(open(os.path.join(D.REPO, n1["path"]), encoding="utf-8"))["npz"]["path"])
        assert D.sha_file(os.path.join(D.REPO, n1["path"])) == n1["sha256"] and D.sha_file(self.node1h_npz) == n1["npz_sha256"]
        rc = RV["structures"]["cells"]
        assert sorted(rc) == sorted(self.cells)
        for c, P in parts.items():
            assert rc[c]["partition"] == P.tag and rc[c]["npart"] == P.npart
        t0 = time.time()
        self.F, famrec = NH.build_families(cd, N)
        t_fam = time.time() - t0
        assert famrec == RV["structures"]["families"], "families differ from route v1"
        self.IN = {c: partition_indegree(self.F, P.hard, P.npart) for c, P in parts.items()}
        self.LOGIN = {c: np.log2(1.0 + self.IN[c].astype(np.float64)) for c in self.cells}
        self.names = [NODE]
        self.hard_ref = {NODE: None}
        self.v1_hard = {}
        self.lat = {NODE: []}
        self.FUSE_LAT = []
        self.TL = {c: {k: [] for k in ("reference", "new_scores", "coverage", "orders_counts_unions", "evaluation")} for c in self.cells}
        self.FRK = np.zeros(ng, np.int64)
        nr, ncn, nm = len(RANKS), len(COUNTS), len(D.M_CURVE)
        self.LO = {c: np.zeros((nq, nr), np.int32) for c in self.cells}
        self.HI = {c: np.zeros((nq, nr, nm), np.int32) for c in self.cells}
        self.PRG = {c: np.zeros((ng, nr), np.int32) for c in self.cells}
        self.CNT = {c: np.zeros((nq, ncn), np.int32) for c in self.cells}
        self.CRAW = {c: np.full((nq, ncn), np.nan) for c in self.cells}
        self.CMASS = {c: np.zeros((nq, nr, ncn), np.int32) for c in self.cells}
        self.PSTAR = {c: np.zeros((nq, nm), np.int32) for c in self.cells}
        self.MSUM = {c: np.zeros((nr, P.npart)) for c, P in parts.items()}
        self.RET = {c: np.zeros((nr, nm, P.npart)) for c, P in parts.items()}
        self.GSTEPS = {c: {k: [] for k in PCS} for c in self.cells}
        self.NOEV = {c: 0 for c in self.cells}
        self.checks = {c: 0 for c in self.cells}
        self.record = {
            "node_score": "IR_L1 (sections 32-33, recomputed; asserted identical through node1h v1 and route v1)",
            "served_order": "IR_L1 FLAT+LOC (RRF K0 = 60); the first B_N nodes of it inside the contacted partitions",
            "rankings": list(RANKS), "reference_rankings (== route v1)": list(REF_RANKS), "counts": list(COUNTS),
            "reference_counts (== route v1)": list(REF_COUNTS), "native_arms": ["%s|%s" % x for x in NATIVE],
            "unions": {u: {"flat_side": "FSUM", "localised_side": x, "count_rule": k} for u, x, k in UNIONS},
            "ranking_information": INFO, "topk": TOPK,
            "greedy": {"tie_band_relative": TOLG, "zero_gain_relative": EPS_G, "tie_priority": "ES order"},
            "families": famrec, "seconds_families": round(t_fam, 1),
            "cells": {c: {"partition": P.tag, "npart": P.npart, "partition_in_entries IN(P)": D.stats(self.IN[c])}
                      for c, P in parts.items()},
            "extends": {"route_v1": {"path": D.rel(fr), "sha256": D.sha_file(fr), "npz_sha256": RV["npz"]["sha256"]}},
            "imports": dict(RV["structures"]["imports"], **{"scratchpad/_l1d_route.py": self.sha_rt})}

    def row(self, j, of, od, os_, npos, fv, frank, g, sl):
        N = self.N
        top = of[:ACT]
        nh = len(top)
        # ---- L1a: the IR_L1 node score (the arithmetic of _l1d_route for this rung)
        t0 = time.perf_counter()
        U, H, GW = [], [], []
        for f in FAMS:
            Fm = self.F[f]
            st = Fm["xadj"][top]
            hit, pos = E.entries(st, Fm["xadj"][top + 1] - st)
            U.append(Fm["adj"][pos].astype(np.int64))
            H.append(hit)
            GW.append(Fm["g"][top][hit])
        u, hit, gw = np.concatenate(U), np.concatenate(H), np.concatenate(GW)
        x = (1.0 / np.arange(1.0, nh + 1.0))[hit]
        x = x * gw
        L = np.bincount(u, weights=x, minlength=N)
        Lord = D.order_from_score(L, frank)
        self.lat[NODE].append(time.perf_counter() - t0)
        # ---- the served order O = IR_L1 FLAT+LOC
        t0 = time.perf_counter()
        f = 1.0 / (K0 + frank.astype(np.float64))
        f[Lord] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
        FO = np.lexsort((frank, -f))
        frk = np.empty(N, np.int64)
        frk[FO] = np.arange(N)
        self.FUSE_LAT.append(time.perf_counter() - t0)
        pg = frk[g]
        self.FRK[sl] = pg
        lim = int(pg.max()) + 1
        # ---- the router's own fused order O' = RRF(H_q, LOC) over the evidence set V_q = H_q u {L > 0}
        vq = np.union1d(top, Lord)
        fq = np.zeros(len(vq))
        tq = np.zeros(len(vq))
        fq[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
        il = np.searchsorted(vq, Lord)
        fq[il] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
        tq[il] = ACT + np.arange(len(Lord))
        tq[np.searchsorted(vq, top)] = np.arange(nh)
        oq = vq[np.lexsort((tq, -fq))]
        nz = np.flatnonzero(L)
        Lv = L[nz]
        wseed = 1.0 / np.arange(1.0, nh + 1.0)
        chk = j < CHECK_ROWS
        for c in self.cells:
            P = self.parts[c]
            npart, hard, TL = P.npart, P.hard, self.TL[c]
            SC, RO, CN, CR = {}, {}, {}, {}
            # ---- reference scores and rankings (route v1 arithmetic)
            t1 = time.perf_counter()
            fa = np.minimum.reduceat(frk[P.order_nodes], P.ptr[:-1])
            hs = hard[top]
            SC["FSUM"] = np.bincount(hs, weights=wseed, minlength=npart)
            SC["FSUMFV"] = np.bincount(hs, weights=fv[top], minlength=npart)
            ph = hard[nz]
            SC["SUM"] = np.bincount(ph, weights=Lv, minlength=npart)
            mx = np.zeros(npart)
            np.maximum.at(mx, ph, Lv)
            SC["MAX"] = mx
            o = np.lexsort((-Lv, ph))
            phs, lvs = ph[o], Lv[o]
            wr = np.arange(len(phs)) - np.searchsorted(phs, phs, side="left")
            m = wr < TOPK
            SC["TOP3"] = np.bincount(phs[m], weights=lvs[m], minlength=npart)
            Cm = np.bincount(hit * npart + hard[u], weights=x, minlength=nh * npart).reshape(nh, npart)
            assert np.allclose(Cm.sum(0), SC["SUM"], rtol=1e-9, atol=1e-15), "per-seed masses do not sum to SUM"
            SC["SDIV"] = wseed @ (Cm > 0)
            RO["S"] = np.argsort(fa, kind="stable")
            for r_ in ("FSUM", "SUM", "TOP3", "SDIV"):
                RO[r_] = np.lexsort((fa, -SC[r_]))
            TL["reference"].append(time.perf_counter() - t1)
            # ---- ES (the router's first appearance) and the new scores, every one falling back to ES
            t1 = time.perf_counter()
            hq = hard[oq]
            up, first = np.unique(hq, return_index=True)
            fe = np.full(npart, len(oq), np.int64)
            fe[up] = first
            RO["ES"] = np.lexsort((np.arange(npart), fe))
            esr = RT.inv(RO["ES"])
            ms_ = Cm.sum(1)
            okm = ms_ > 0
            Sh = np.zeros((nh, npart))
            Sh[okm] = Cm[okm] / ms_[okm][:, None]
            SC["NSUM"] = wseed @ Sh
            SC["NSUMS"] = SC["NSUM"] + SC["FSUM"]
            assert abs(SC["NSUM"].sum() - wseed[okm].sum()) <= 1e-9 * wseed.sum() and abs(SC["NSUMS"].sum() - wseed.sum() - wseed[okm].sum()) <= 1e-9 * wseed.sum()
            SC["SDIVS"] = SC["SDIV"] + SC["FSUM"]
            ssum = SC["SUM"].sum()
            SC["SHF"] = SC["FSUM"] / SC["FSUM"].sum() + (SC["SUM"] / ssum if ssum > 0 else 0.0)
            SC["SUMH"] = np.zeros(npart)
            okp = self.IN[c] > 0
            SC["SUMH"][okp] = SC["SUM"][okp] / self.LOGIN[c][okp]
            assert (SC["SUM"][~okp] == 0).all()
            for r_ in NEW_SCORED:
                RO[r_] = np.lexsort((esr, -SC[r_]))
            T3E, SDE, FSE = np.lexsort((esr, -SC["TOP3"])), np.lexsort((esr, -SC["SDIV"])), np.lexsort((esr, -SC["FSUM"]))
            TL["new_scores"].append(time.perf_counter() - t1)
            # ---- probabilistic (noisy-OR) coverage, columns in ES order
            t1 = time.perf_counter()
            PmLS = np.zeros((nh, npart))
            PmLS[okm] = 0.5 * Sh[okm]
            PmLS[np.arange(nh), hs] += np.where(okm, 0.5, 1.0)
            assert np.allclose(PmLS.sum(1), 1.0, rtol=0, atol=1e-12)
            GG = {}
            for name, Pm in (("PC_L", Sh), ("PC_LS", PmLS)):
                Ps = Pm[:, RO["ES"]]
                sel, G = pc_greedy(Ps, wseed)
                if len(G) > 1:
                    assert (np.diff(G) <= TOLG * G[0]).all(), "coverage gains increased (%s %s)" % (c, name)
                if chk:
                    s2, G2 = pc_greedy_ref(Ps, wseed)
                    assert len(s2) == len(sel) and (s2 == sel).all() and np.allclose(G2, G, rtol=1e-9, atol=0), \
                        "incremental coverage greedy != from-scratch (%s %s)" % (c, name)
                chosen = RO["ES"][sel]
                RO[name] = np.concatenate([chosen, RO["ES"][np.isin(RO["ES"], chosen, invert=True)]])
                GG[name] = G
                self.GSTEPS[c][name].append(len(G))
            TL["coverage"].append(time.perf_counter() - t1)
            # ---- counts, orders, unions
            t1 = time.perf_counter()
            for x_ in ("FSUM", "FSUMFV", "SUM", "MAX", "TOP3", "SDIV") + NEW_SCORED:
                ne, npz_ = RT.neff(SC[x_])
                CR["NEFF_" + x_] = ne if npz_ else np.nan
                CN["NEFF_" + x_] = AB.bp_of(ne, npz_, npart) if npz_ else npart
                kn, npz_ = RT.knee(SC[x_])
                CR["KNEE_" + x_] = kn if npz_ else np.nan
                CN["KNEE_" + x_] = kn if npz_ else npart
                h1, hinf, npz_ = hill(SC[x_])
                CR["EXPH_" + x_] = h1 if npz_ else np.nan
                CN["EXPH_" + x_] = AB.bp_of(h1, npz_, npart) if npz_ else npart
                CR["BPI_" + x_] = hinf if npz_ else np.nan
                CN["BPI_" + x_] = AB.bp_of(hinf, npz_, npart) if npz_ else npart
            if not (SC["TOP3"] > 0).any():
                self.NOEV[c] += 1
            CN["NSEEDP"] = CR["NSEEDP"] = int(len(np.unique(hs)))
            CN["NOACT"] = CR["NOACT"] = int(len(np.unique(hard[FO[:ACT]])))
            CN["NEV"] = CR["NEV"] = int(len(up))
            CN["NOACT_E"] = CR["NOACT_E"] = int(len(np.unique(hq[:ACT])))
            for name in PCS:
                G = GG[name]
                if len(G):
                    ne, _ = RT.neff(G)
                    CR[name + "_NEFFG"], CN[name + "_NEFFG"] = ne, AB.bp_of(ne, len(G), npart)
                    kn, _ = RT.knee(G)
                    CR[name + "_KNEEG"], CN[name + "_KNEEG"] = kn, kn
                    CR[name + "_COVER"], CN[name + "_COVER"] = len(G), len(G)
                else:
                    for k_ in ("NEFFG", "KNEEG", "COVER"):
                        CN["%s_%s" % (name, k_)] = npart
            RO["SL"] = np.lexsort((esr, SC["SUM"] <= 0))
            for sfx, X in (("T3", T3E), ("SD", SDE)):
                rX = RT.inv(X)
                RO["AND_" + sfx] = np.lexsort((esr, np.minimum(esr, rX), np.maximum(esr, rX)))
                RO["PF_E_" + sfx] = np.lexsort((esr, -(1.0 / (K0 + esr) + 1.0 / (K0 + rX))))
                RO["IL_E_" + sfx] = np.argsort(np.minimum(2 * esr, 2 * rX + 1), kind="stable")
            rF = RT.inv(FSE)
            XO = {"SUM": np.lexsort((esr, -SC["SUM"])), "TOP3": T3E, "SDIV": SDE, "NSUM": RO["NSUM"]}
            for uname, xs, rule in UNIONS:
                cF, cX = CN["%s_FSUM" % rule], CN["%s_%s" % (rule, xs)]
                ro, n_, inU = union_order(rF, cF, RT.inv(XO[xs]), cX, esr)
                if chk:
                    assert inU[ro[:n_]].all(), "union prefix != union (%s %s)" % (c, uname)
                RO[uname] = ro
                CN[uname] = CR[uname] = n_
            TL["orders_counts_unions"].append(time.perf_counter() - t1)
            cnt = np.array([CN[k_] for k_ in COUNTS], np.int64)
            assert (cnt >= 1).all() and (cnt <= npart).all(), (c, dict(zip(COUNTS, cnt.tolist())))
            self.CNT[c][j] = cnt
            self.CRAW[c][j] = [CR.get(k_, np.nan) for k_ in COUNTS]
            for mi, M in enumerate(D.M_CURVE):
                self.PSTAR[c][j, mi] = len(np.unique(hard[FO[:M]]))
            # ---- evaluation (gold-dependent; not part of any rule): every fan-out of every ranking
            t1 = time.perf_counter()
            hO, hg = hard[FO[:lim]], hard[g]
            for ri, rk in enumerate(RANKS):
                ro = RO[rk]
                if chk:
                    assert len(ro) == npart and (np.sort(ro) == np.arange(npart)).all(), "not a permutation (%s %s)" % (c, rk)
                pr = RT.inv(ro)
                prg = pr[hg]
                C = AB.cumcounts(pr[hO], pg, npart)
                assert (C[:, npart - 1] - 1 == pg).all(), "every partition contacted != unrouted (%s %s)" % (c, rk)
                ms = np.cumsum(P.sizes[ro])
                lo = int(prg.max()) + 1
                qmax = (C[:, lo - 1:] - 1).max(axis=0)
                assert (np.diff(qmax) >= 0).all()
                self.LO[c][j, ri] = lo
                self.HI[c][j, ri] = lo - 1 + np.searchsorted(qmax, MSA, side="left")
                self.PRG[c][sl, ri] = prg
                self.MSUM[c][ri] += ms
                self.RET[c][ri] += np.minimum(MSA[:, None], ms[None, :])
                self.CMASS[c][j, ri] = ms[cnt - 1]
            TL["evaluation"].append(time.perf_counter() - t1)
            if chk:
                self.checks[c] += 1
        return {NODE: Lord}

    def finish(self, ctx):
        POS, gptr, ngold, pop, ST = ctx["POS"], ctx["gptr"], ctx["ngold"], ctx["pop"], ctx["ST"]
        nq, ng = pop.nq, pop.ng_tot
        assert D.sha_file(RT_PATH) == self.sha_rt, "imported code changed during the run"
        allv = lambda pos, M: D.per_query(pos, M, gptr, ngold)[0]
        diag, arrays = {}, {}
        # (0) identities against node1h v1 and route v1
        z = np.load(self.node1h_npz)
        assert (z["rows"][:nq] == pop.rows).all() and (z["gptr"][:nq + 1] == gptr).all()
        assert (z["pos_LOC__" + NODE][:ng] == POS[NODE]["LOC"]).all(), "IR_L1 LOC positions differ from node1h v1"
        assert (z["pos_FLATLOC__" + NODE][:ng] == POS[NODE]["FLAT+LOC"]).all(), "IR_L1 FLAT+LOC positions differ from node1h v1"
        assert (self.FRK == POS[NODE]["FLAT+LOC"]).all()
        zr = np.load(self.route_npz)
        assert (zr["rows"][:nq] == pop.rows).all() and (zr["gptr"][:nq + 1] == gptr).all()
        vr, vc = [str(s) for s in zr["ranks"]], [str(s) for s in zr["counts"]]
        for c in self.cells:
            for rk in REF_RANKS:
                a, b = vr.index(rk), RANKS.index(rk)
                assert (zr["LO__" + c][:nq, a] == self.LO[c][:, b]).all(), "LO differs from route v1 (%s %s)" % (c, rk)
                assert (zr["HI__" + c][:nq, a] == self.HI[c][:, b]).all(), "HI differs from route v1 (%s %s)" % (c, rk)
                assert (zr["PRG__" + c][:ng, a] == self.PRG[c][:, b]).all(), "PRG differs from route v1 (%s %s)" % (c, rk)
            for cn in REF_COUNTS:
                assert (zr["CNT__" + c][:nq, vc.index(cn)] == self.CNT[c][:, COUNTS.index(cn)]).all(), "count differs from route v1 (%s %s)" % (c, cn)
            assert (zr["PSTAR__" + c][:nq] == self.PSTAR[c]).all(), "PSTAR differs from route v1 (%s)" % c
        diag["identity"] = ("IR_L1 LOC and FLAT+LOC positions == node1h v1 on all %d gold nodes; LO / HI / PRG of %s, the counts %s and "
                            "PSTAR == route v1 in every cell; every partition contacted == unrouted (per row, per ranking); per-seed "
                            "masses sum to SUM, NSUM / NSUMS totals and PC_LS rows checked, coverage gains never increase (every row); "
                            "on the first %d rows of every cell: incremental coverage greedy == from-scratch for %s, every ranking a "
                            "permutation, every union prefix == the union" % (ng, list(REF_RANKS), list(REF_COUNTS), CHECK_ROWS, list(PCS)))
        diag["checked_rows"] = self.checks
        unr = {M: allv(POS[NODE]["FLAT+LOC"], M) for M in D.M_CURVE}
        hop_m = ST.get("per_hop", {})
        cells_out = {}
        for c in self.cells:
            P = self.parts[c]
            npart = P.npart
            LO, HI, CNT = self.LO[c].astype(np.int64), self.HI[c].astype(np.int64), self.CNT[c].astype(np.int64)
            e_c = {"unrouted_ALL": {str(M): D.q4(unr[M].mean()) for M in D.M_CURVE}, "rankings": {}, "counts": {}, "native_arms": {}}
            SF = {}
            for ri, rk in enumerate(RANKS):
                S_ = AB.surface(LO[:, ri], HI[:, ri], npart)
                SF[rk] = S_
                e = {"contacted_mass_mean": {str(b): round(float(self.MSUM[c][ri, b - 1] / nq), 1) for b in BP_SHOW if b <= npart},
                     "gold_partition_contacted_share": {str(b): D.q4((self.PRG[c][:, ri] < b).mean()) for b in BP_SHOW if b <= npart},
                     "LO": AB.dist(LO[:, ri])}
                for mi, M in enumerate(D.M_CURVE):
                    row_ = S_[mi]
                    assert abs(row_[-1] - unr[M].mean()) < 1e-12, "surface at every partition != unrouted (%s %s %d)" % (c, rk, M)
                    e[str(M)] = {"ALL_at_B_P": {str(b): D.q4(row_[b - 1]) for b in BP_SHOW if b <= npart},
                                 "best_fixed": {"B_P": int(np.argmax(row_)) + 1, "ALL": D.q4(row_.max())},
                                 "smallest_B_P_within_0.01_of_unrouted": int(np.flatnonzero(row_ >= row_[-1] - 0.01)[0]) + 1,
                                 "smallest_B_P_at_or_above_unrouted": int(np.flatnonzero(row_ >= row_[-1] - 1e-12)[0]) + 1}
                e_c["rankings"][rk] = e
            for ci, cn in enumerate(COUNTS):
                fin = np.isfinite(self.CRAW[c][:, ci])
                e_c["counts"][cn] = {"B_P(q)": AB.dist(CNT[:, ci]), "raw": AB.dist(self.CRAW[c][:, ci][fin]) if fin.any() else None}
            for rk, cn in NATIVE:
                ri, ci = RANKS.index(rk), COUNTS.index(cn)
                b = CNT[:, ci]
                e = {"B_P(q)": AB.dist(b), "contacted_mass": AB.dist(self.CMASS[c][:, ri, ci])}
                for mi, M in enumerate(D.M_CURVE):
                    a_ = AB.served_at(LO[:, ri], HI[:, ri, mi], b)
                    row_ = SF[rk][mi]
                    bm = float(b.mean())
                    bl, bh = int(np.floor(bm)), int(np.ceil(bm))
                    e[str(M)] = {"ALL": D.q4(a_.mean()), "delta_vs_unrouted": D.q4(a_.mean() - unr[M].mean()),
                                 "paired_vs_unrouted": D.paired(unr[M], a_),
                                 "same_ranking_fixed_at_mean_fan_out_interpolated": D.q4(row_[bl - 1] + (bm - bl) * (row_[bh - 1] - row_[bl - 1])),
                                 "same_ranking_best_fixed": {"B_P": int(np.argmax(row_)) + 1, "ALL": D.q4(row_.max())},
                                 "lost_reach (B_P(q) < LO)": D.q4((b < LO[:, ri]).mean()),
                                 "lost_crowding (LO <= HI < B_P(q))": D.q4(((LO[:, ri] <= HI[:, ri, mi]) & (HI[:, ri, mi] < b)).mean()),
                                 "unservable_at_any_B_P (LO > HI)": D.q4((LO[:, ri] > HI[:, ri, mi]).mean())}
                    if hop_m:
                        e[str(M)]["per_hop"] = {hk: {"n": int(qm.sum()), "ALL": D.q4(a_[qm].mean()), "unrouted_ALL": D.q4(unr[M][qm].mean())}
                                                for hk, qm in hop_m.items()}
                e_c["native_arms"]["%s|%s" % (rk, cn)] = e
            e_c["coverage_greedy_steps"] = {k: AB.dist(v) for k, v in self.GSTEPS[c].items()}
            e_c["queries_without_localised_evidence"] = self.NOEV[c]
            cells_out[c] = e_c
            for nm_, A_ in (("LO", self.LO), ("HI", self.HI), ("PRG", self.PRG), ("CNT", self.CNT), ("CRAW", self.CRAW),
                            ("CMASS", self.CMASS), ("PSTAR", self.PSTAR), ("MSUM", self.MSUM), ("RET", self.RET)):
                arrays["%s__%s" % (nm_, c)] = A_[c]
        diag["cells"] = cells_out
        diag["latency_ms"] = {"node_score_and_LOC_order": D.ms_stats(self.lat[NODE]), "fused_order": D.ms_stats(self.FUSE_LAT),
                              "per_cell": {c: {k: D.ms_stats(v) for k, v in self.TL[c].items()} for c in self.cells}}
        arrays["ranks"] = np.array(RANKS)
        arrays["counts"] = np.array(COUNTS)
        arrays["native"] = np.array(["%s|%s" % x for x in NATIVE])
        arrays["m_curve"] = MSA
        return diag, arrays


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    assert mode == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]"
    R.run(sys.argv[2], sys.argv[3], "route2", "L1_DEVELOPMENT_ROUTE2", __file__, Route2Spec, __doc__)


if __name__ == "__main__":
    main()
