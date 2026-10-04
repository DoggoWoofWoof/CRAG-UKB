"""L1 DEVELOPMENT -- ROUTE3: the third batch of non-parametric L1b partition routers (user 2026-09-27: "test out everything we
can based on the suggestions and based on the failures we have encountered or will encouter and try to mitigate them").
Development numbers; any p-value is descriptive.

Fixed, exactly as _l1d_route.py / _l1d_route2.py (asserted per query): the L1a node score IR_L1, the served order O (IR_L1
FLAT+LOC, RRF K0 = 60), the cells and their frozen hard shards, the population and the serving rule (the first B_N nodes of O
inside the B_P contacted partitions).  Only the partition ORDER (a ranking) and the partition COUNT (a stopping rule) vary; each
is a fixed function of the router's information -- H_q = FLAT[:200] with its ranks, the seeds' 1-hop rows (hence L, the LOC
order and O' = RRF(H_q, LOC)) and the static shard map -- no learned weight, no tuned constant, no gold.  The reference rankings
REF_RANKS, the reference counts REF_COUNTS and PSTAR are recomputed and asserted identical to route v2 (per query), so the three
records merge into one factorial in the summary.

The failure left by rounds 1-2 and the mitigations run here:
 (7) regime conflict: on the text cells the gold partitions are FLAT-seed partitions spread over many shards (every loss is
     reach, B_P(q) < LO; MuSiQue ES|KNEE_SDIV -0.008 at 198 partitions), on the KB cells the answer partitions are a few shards
     reached through the seeds' 1-hop rows (the gain is crowding relief at a small B_P; MetaQA SDIV|BPI_SDIV +0.071 / +0.093 at
     16.5 / 18.0 partitions, -0.263 on MuSiQue).  Every count of rounds 1-2 is a concentration of ONE evidence distribution;
     their MuSiQue / MetaQA ratios are 1.5-2.5x while the two regimes need ~6-10x.
     -> a per-query regime share lam(q) in [0, 1] read off the router's own evidence (no constant):
        L1: LAM1 = the off-seed share of the IR_L1 mass, 1 - sum_{u in H_q} L(u) / sum_u L(u);
        L2: LAM2 = the share of O'[:200] that are not seeds;
        L3: LAM3 = the share of LOC[:200] outside H_q (the disagreement of the two lists);
        L4: LAM4 = 1 - BPI_SDIV / NSEEDP within [0, 1] (per cell: how many times fewer effective partitions the 1-hop
            evidence occupies than the seeds; router means of rounds 1-2 ~0.86 on MetaQA, ~0.5 on MuSiQue / SQuAD);
        each 0 for a query without 1-hop evidence.  FLAGGED: L4 was added after the 200-row smoke showed L1 / L3 ~0.74-0.80 on
        MetaQA and MuSiQue alike (L2 ~0.33 on both), i.e. the node-level off-seed shares do not separate the two regimes;
     -> the mixtures of the text endpoint A = (ES, C) and the KB endpoint B = (SDE, BPI_SDIV):
        rankings MXS_Lk: order by (1 - lam) / (K0 + rank_ES) + lam / (K0 + rank_SDE) (ties -> ES; lam = 0 -> ES, 1 -> SDE),
                 MXT_Lk: the same with T3E;
        counts   MA_<C>_Lk = ceil((1 - lam) C + lam BPI_SDIV) (arithmetic), MG_<C>_Lk = ceil(C^(1 - lam) BPI_SDIV^lam)
                 (geometric), C in {KNEE_SDIV (KS), KNEE_TOP3 (KT), NOACT_E (NA)};
        unions   UM_<C>_Lk = ES[:ceil((1 - lam) C)] u SDE[:ceil(lam BPI_SDIV)] (order min(rank_ES / c_E, rank_SDE / c_L), ties ->
                 ES; the union is the prefix, its size the count).
     FLAGGED: the endpoints were chosen on the development results of rounds 1-2; only the per-query share is new.
 (8) count shape: every count so far measures mass concentration (NEFF / EXPH / BPI / KNEE); none places a natural break.
     -> GAP_x = the position of the largest drop of the sorted positive scores of x, LGAP_x = the position of the largest ratio
        (relative drop); x in {FSUM, SUM, TOP3, SDIV, NSUM, SDIVS}, each with its own ranking (ties -> ES); a single positive
        score -> 1; all positive scores equal -> all of them.
Anticipated failures of a distributed deployment (measured here, not mitigated by a rule):
 (9) router state: a router in front of a sharded graph cannot hold the node-level graph.  Scores of the FSUM / SUM / SDIV /
     NSUM family need only a partition sketch per node (the partitions of its 1-hop neighbours; with multiplicities per family
     for SUM / NSUM); MAX / TOP3 / ES / O' / the regime shares need the seeds' node-level rows.  Measured statically per cell:
     node-row entries vs sketch entries per family and for the union, and the co-partitioned share; per query: the entries the
     router reads for its 200 seeds.  Ties: route v1's SDIV breaks ties by S (needs O, not router information), route v2's SDE
     by ES (node rows); SDK = SDIV with ties -> FSUM -> static partition id needs the sketch only.
 (10) B_N coupling: the lossless reference PSTAR_BN (the distinct partitions of O[:B_N]) needs O.  ESTAR_BN = the distinct
     partitions of the router's O'[:B_N] is its router-side estimate (the first ESTAR_BN partitions of ES are exactly those);
     ES|ESTAR_BN is a B_N-coupled diagnostic arm (not in the factorial), with the share of PSTAR's partitions ESTAR contains.
 (11) load: a rule that sends every query to the same hub shards overloads them.  Per arm, the per-partition contact counts
     over the population -> the largest contact rate, peak / mean, the share of all contacts on the top 1% of partitions, the
     Gini coefficient and the partitions never contacted (B_N-free counts; PSTAR / ESTAR at B_N 1000).
Oracle (gold-dependent; used by no rule): per query, served under A = ES|C and / or B = SDE|BPI_SDIV; the switch that picks the
better endpoint per query (an upper bound for ANY per-query switch between the two) and the AUC of each lam for B-only vs A-only
queries (descriptive).
Not run: MMR / xQuAD / tau-coverage (a hyperparameter each); a sampled (CSI) router (section 28's prototype ladder is the
evidence on fixed per-partition representatives).
Counts: as route v2 (a query without evidence for a score contacts every partition); MA / MG / UM within [1, npart].
Identities asserted: IR_L1 LOC / FLAT+LOC positions == node1h v1 (all gold nodes); LO / HI / PRG of REF_RANKS, every REF_COUNT
and PSTAR == route v2 (per query, every cell); every partition contacted == unrouted (per row, per ranking); PSTAR_BN == the
partitions whose first O rank is < B_N; on the first rows: every ranking a permutation, every union prefix == the union, the
mixtures at lam = 0 / 1 == ES / SDE (T3E), MA / MG at lam = 0 / 1 == C / BPI_SDIV.

Usage: python scratchpad/_l1d_route3.py RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L1_DEV/route3_<dataset>__<tag>.{json,npz} (write-once)
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
import _l1d_route2 as RT2

log = D.log
K0, ACT = D.K0, D.ACT
FAMS = NH.FAMS
NODE = "IR_L1"
TOPK = RT.TOPK
TOL, CHECK_ROWS = RT.TOL, RT.CHECK_ROWS
LAMS = ("L1", "L2", "L3", "L4")
LAM_DEF = {"L1": "LAM1 = 1 - sum_{u in H_q} L(u) / sum_u L(u) (the off-seed share of the IR_L1 mass)",
           "L2": "LAM2 = the share of O'[:200] that are not seeds",
           "L3": "LAM3 = the share of LOC[:200] outside H_q",
           "L4": "LAM4 = 1 - BPI_SDIV / NSEEDP within [0, 1] (per cell; added after the 200-row smoke, FLAGGED)"}
CSRC = (("KS", "KNEE_SDIV"), ("KT", "KNEE_TOP3"), ("NA", "NOACT_E"))
LOCAL = "BPI_SDIV"
GAPX = ("FSUM", "SUM", "TOP3", "SDIV", "NSUM", "SDIVS")
REF_RANKS = ("S", "TOP3", "SDIV", "ES", "NSUM", "SDIVS")
BASE = ("FSE", "SUME", "T3E", "SDE", "SDK")
MIXR = tuple("%s_%s" % (m, l) for m in ("MXS", "MXT") for l in LAMS)
UMS = tuple("UM_%s_%s" % (k, l) for k, _ in CSRC for l in LAMS)
RANKS = REF_RANKS + BASE + MIXR + UMS
REF_COUNTS = ("KNEE_SDIV", "KNEE_TOP3", "BPI_SDIV", "BPI_TOP3", "NOACT_E", "NSEEDP")
GAPC = tuple("%s_%s" % (p, x) for x in GAPX for p in ("GAP", "LGAP"))
MIXC = tuple("%s_%s_%s" % (f, k, l) for f in ("MA", "MG") for k, _ in CSRC for l in LAMS)
COUNTS = REF_COUNTS + GAPC + MIXC + UMS
GAP_RANK = {"FSUM": "FSE", "SUM": "SUME", "TOP3": "T3E", "SDIV": "SDE", "NSUM": "NSUM", "SDIVS": "SDIVS"}
NATIVE = ([("%s_%s" % (m, l), "%s_%s_%s" % (f, k, l)) for l in LAMS for m in ("MXS", "MXT") for f in ("MA", "MG") for k, _ in CSRC]
          + [(u, u) for u in UMS]
          + [("SDE", "BPI_SDIV"), ("SDK", "BPI_SDIV"), ("T3E", "BPI_TOP3")]
          + [(GAP_RANK[x], "%s_%s" % (p, x)) for x in GAPX for p in ("GAP", "LGAP")])
REF_LOAD = (("S", "KNEE_TOP3"), ("S", "KNEE_SDIV"), ("ES", "KNEE_SDIV"), ("ES", "KNEE_TOP3"), ("ES", "NSEEDP"), ("ES", "NOACT_E"),
            ("SDIV", "BPI_SDIV"), ("TOP3", "BPI_TOP3"))
M_LOAD = 1000
BN_LOAD = (("S", "PSTAR_BN@%d" % M_LOAD), ("ES", "ESTAR_BN@%d" % M_LOAD))
LOAD_ARMS = tuple(NATIVE) + REF_LOAD + BN_LOAD
ORACLE_B = ("SDE", LOCAL)
ORACLE_A = tuple(("ES", cn) for _, cn in CSRC)
ROUTER = "H_q = FLAT[:200] + the seeds' 1-hop rows"
SKETCH = "H_q = FLAT[:200] + the seeds' partition sketch (the partitions of their 1-hop neighbours)"
INFO = {"FSE": ROUTER, "SUME": ROUTER, "T3E": ROUTER, "SDE": ROUTER, "SDK": SKETCH}
INFO.update({r: ROUTER + " + its regime share" for r in MIXR + UMS})
BP_SHOW = AB.BP_SHOW
MSA = np.asarray(D.M_CURVE, np.int64)
MI_LOAD = D.M_CURVE.index(M_LOAD)
RT_PATH = os.path.join(D.HERE, "_l1d_route.py")
RT2_PATH = os.path.join(D.HERE, "_l1d_route2.py")
assert len(set(RANKS)) == len(RANKS) and len(set(COUNTS)) == len(COUNTS) and set(INFO) == set(RANKS) - set(REF_RANKS)
assert set(REF_RANKS) <= set(RT2.RANKS) and set(REF_COUNTS) <= set(RT2.COUNTS)
assert not (set(RANKS) - set(REF_RANKS)) & (set(RT.RANKS) | set(RT2.RANKS))
assert not (set(COUNTS) - set(REF_COUNTS)) & (set(RT.COUNTS) | set(RT2.COUNTS))
assert all(r in RANKS and c in COUNTS for r, c in NATIVE + list(REF_LOAD)) and FAMS == RT2.FAMS
assert all(r in RANKS for r, _ in BN_LOAD) and ORACLE_B in NATIVE


def ceil_tol(x):
    return int(np.ceil(x - TOL))


def mix_rank(esr, rX, lam):
    """order by (1 - lam) / (K0 + rank_ES) + lam / (K0 + rank_X), ties -> ES (0-based ranks, as route v2's PF_E_*)."""
    return np.lexsort((esr, -((1.0 - lam) / (K0 + esr) + lam / (K0 + rX))))


def mix_counts(C, B, lam, npart):
    """(arithmetic raw, geometric raw, arithmetic count, geometric count) between the counts C (lam = 0) and B (lam = 1)."""
    ma = (1.0 - lam) * C + lam * B
    mg = float(np.exp((1.0 - lam) * np.log(C) + lam * np.log(B)))
    return ma, mg, min(max(ceil_tol(ma), 1), npart), min(max(ceil_tol(mg), 1), npart)


def union0(ra, ca, rb, cb, tie):
    """the ranking whose first n entries are A[:ca] u B[:cb] (0-based ranks ra, rb; a side with c = 0 contributes nothing),
    ordered by min(ra / ca, rb / cb), ties -> tie."""
    ka = ra / float(ca) if ca > 0 else np.full(len(ra), np.inf)
    kb = rb / float(cb) if cb > 0 else np.full(len(rb), np.inf)
    ro = np.lexsort((tie, np.minimum(ka, kb)))
    inU = (ra < ca) | (rb < cb)
    return ro, int(inU.sum()), inU


def gaps(v, npart):
    """(GAP, LGAP) of the positive values of v: the position of the largest drop / the largest ratio of the sorted values."""
    v = np.sort(v[v > 0])[::-1]
    n = len(v)
    if n == 0:
        return npart, npart
    if n == 1:
        return 1, 1
    d = v[:-1] - v[1:]
    q = v[:-1] / v[1:]
    dm, qm = float(d.max()), float(q.max())
    g = int(np.flatnonzero(d >= dm - TOL * dm)[0]) + 1 if dm > 0 else n
    lg = int(np.flatnonzero(q >= qm - TOL * qm)[0]) + 1 if qm > 1.0 + TOL else n
    return g, lg


def auc(pos, neg):
    """P(pos > neg) + P(pos == neg) / 2 (Mann-Whitney), None when a side is empty."""
    if not len(pos) or not len(neg):
        return None
    r = AB.rankavg(np.concatenate([pos, neg]))
    u = r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2.0
    return D.q4(u / (len(pos) * len(neg)))


def load_stats(cnt, nq):
    """per-partition contact counts over nq queries -> the load profile of one arm."""
    npart = len(cnt)
    tot = int(cnt.sum())
    if tot == 0:
        return None
    srt = np.sort(cnt)
    k1 = int(np.ceil(0.01 * npart))
    i = np.arange(1, npart + 1, dtype=np.float64)
    gini = float((2.0 * (i * srt).sum()) / (npart * srt.sum()) - (npart + 1.0) / npart)
    return {"contacts_per_query_mean": round(tot / float(nq), 2), "largest_contact_rate": D.q4(srt[-1] / float(nq)),
            "peak_over_mean": round(float(srt[-1]) / (tot / float(npart)), 2), "top_1pct_partitions_share": D.q4(srt[-k1:].sum() / float(tot)),
            "gini": D.q4(gini), "never_contacted": int((cnt == 0).sum())}


def router_state(F, hard, npart, N):
    """static: node-row entries vs partition-sketch entries per family and for the union; per-node lengths."""
    rec, rowlen, famlen, keys = {}, np.zeros(N, np.int64), np.zeros(N, np.int64), []
    for f in FAMS:
        xadj = F[f]["xadj"].astype(np.int64)
        deg = np.diff(xadj)
        tot = int(xadj[-1])
        assert len(deg) == N and len(F[f]["adj"]) == tot
        rowlen += deg
        src = np.repeat(np.arange(N, dtype=np.int64), deg)
        hd = hard[F[f]["adj"].astype(np.int64)]
        co = int((hard[src] == hd).sum())
        key = np.unique(src * npart + hd)
        del src, hd
        famlen += np.bincount(key // npart, minlength=N)
        rec[f] = {"node_row_entries": tot, "sketch_entries (distinct (node, partition))": int(len(key)),
                  "sketch / node-row entries": D.q4(len(key) / float(tot)) if tot else None,
                  "co_partitioned_entries (inside the source's own partition)": co,
                  "co_partitioned_share (entries inside the source's own partition)": D.q4(co / float(tot)) if tot else None}
        keys.append(key)
    uk = np.unique(np.concatenate(keys))
    unilen = np.bincount(uk // npart, minlength=N)
    tot_rows = int(rowlen.sum())
    rec["all_families"] = {"node_row_entries": tot_rows, "sketch_entries_per_family_summed (SUM / NSUM multiplicities)": int(famlen.sum()),
                           "sketch_entries_union (SDIV: distinct (node, partition))": int(len(uk)),
                           "union_sketch / node-row entries": D.q4(len(uk) / float(tot_rows)),
                           "nodes": int(N), "partitions": int(npart),
                           "union_sketch_row_length": D.stats(unilen)}
    return rec, rowlen, famlen, unilen


class Route3Spec(object):
    def __init__(self, cd, pop, parts):
        self.ds, self.parts, self.cells = cd.name, parts, list(parts)
        N = self.N = int(cd.n_nodes)
        nq, ng = pop.nq, pop.ng_tot
        # ---- the route v2 record this harness extends (same arithmetic for the shared parts, same population and partitions)
        f2 = os.path.join(D.OUT, "route2_%s__v1.json" % self.ds)
        RV2 = json.load(open(f2, encoding="utf-8"))
        self.route2_npz = os.path.join(D.OUT, RV2["npz"]["path"])
        assert D.sha_file(self.route2_npz) == RV2["npz"]["sha256"], "route v2 npz changed"
        self.sha_rt2 = D.sha_file(RT2_PATH)
        assert self.sha_rt2 == RV2["code"]["harness"]["sha256"], "_l1d_route2.py differs from the code of its v1 record"
        for k_, p_ in RV2["structures"]["imports"].items():
            assert D.sha_file(os.path.join(D.REPO, k_)) == p_, "%s differs from route v2's import" % k_
        self.sha_rt = D.sha_file(RT_PATH)
        assert RV2["structures"]["imports"]["scratchpad/_l1d_route.py"] == self.sha_rt
        assert RV2["code"]["arms"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_arms.py"))
        assert RV2["code"]["lib"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))
        f1 = os.path.join(D.REPO, RV2["structures"]["extends"]["route_v1"]["path"])
        assert D.sha_file(f1) == RV2["structures"]["extends"]["route_v1"]["sha256"]
        RV1 = json.load(open(f1, encoding="utf-8"))
        n1 = RV1["structures"]["reproduces"]["node1h"]
        self.node1h_npz = os.path.join(D.OUT, json.load(open(os.path.join(D.REPO, n1["path"]), encoding="utf-8"))["npz"]["path"])
        assert D.sha_file(os.path.join(D.REPO, n1["path"])) == n1["sha256"] and D.sha_file(self.node1h_npz) == n1["npz_sha256"]
        rc = RV2["structures"]["cells"]
        assert sorted(rc) == sorted(self.cells)
        for c, P in parts.items():
            assert rc[c]["partition"] == P.tag and rc[c]["npart"] == P.npart
        t0 = time.time()
        self.F, famrec = NH.build_families(cd, N)
        t_fam = time.time() - t0
        assert famrec == RV2["structures"]["families"], "families differ from route v2"
        t0 = time.time()
        self.RS, self.FAMLEN, self.UNILEN = {}, {}, {}
        for c, P in parts.items():
            self.RS[c], self.ROWLEN, self.FAMLEN[c], self.UNILEN[c] = router_state(self.F, P.hard, P.npart, N)
        t_rs = time.time() - t0
        self.names = [NODE]
        self.hard_ref = {NODE: None}
        self.v1_hard = {}
        self.lat = {NODE: []}
        self.FUSE_LAT, self.LAM_LAT = [], []
        self.TL = {c: {k: [] for k in ("reference", "base_gaps", "mixtures_unions", "estar_load", "evaluation")} for c in self.cells}
        self.FRK = np.zeros(ng, np.int64)
        nr, ncn, nm = len(RANKS), len(COUNTS), len(D.M_CURVE)
        self.LAM = np.zeros((nq, 3))
        self.LAM4 = {c: np.zeros(nq) for c in self.cells}
        self.RDN = np.zeros(nq, np.int64)
        self.RDS = {c: np.zeros((nq, 2), np.int64) for c in self.cells}
        self.LO = {c: np.zeros((nq, nr), np.int32) for c in self.cells}
        self.HI = {c: np.zeros((nq, nr, nm), np.int32) for c in self.cells}
        self.PRG = {c: np.zeros((ng, nr), np.int32) for c in self.cells}
        self.CNT = {c: np.zeros((nq, ncn), np.int32) for c in self.cells}
        self.CRAW = {c: np.full((nq, ncn), np.nan) for c in self.cells}
        self.CMASS = {c: np.zeros((nq, nr, ncn), np.int32) for c in self.cells}
        self.PSTAR = {c: np.zeros((nq, nm), np.int32) for c in self.cells}
        self.ESTAR = {c: np.zeros((nq, nm), np.int32) for c in self.cells}
        self.ESREC = {c: np.zeros((nq, nm)) for c in self.cells}
        self.OQLEN = np.zeros(nq, np.int64)
        self.MSUM = {c: np.zeros((nr, P.npart)) for c, P in parts.items()}
        self.RET = {c: np.zeros((nr, nm, P.npart)) for c, P in parts.items()}
        self.LOAD = {c: np.zeros((len(LOAD_ARMS), P.npart), np.int64) for c, P in parts.items()}
        self.NOEV = {c: 0 for c in self.cells}
        self.checks = {c: 0 for c in self.cells}
        self.record = {
            "node_score": "IR_L1 (sections 32-33, recomputed; asserted identical through node1h v1 and route v2)",
            "served_order": "IR_L1 FLAT+LOC (RRF K0 = 60); the first B_N nodes of it inside the contacted partitions",
            "rankings": list(RANKS), "reference_rankings (== route v2)": list(REF_RANKS), "counts": list(COUNTS),
            "reference_counts (== route v2)": list(REF_COUNTS), "native_arms": ["%s|%s" % x for x in NATIVE],
            "regime_shares": LAM_DEF, "mixture_endpoints (FLAGGED: chosen on rounds 1-2 development results)": {
                "A (lam = 0)": {"ranking": "ES", "counts": dict(CSRC)}, "B (lam = 1)": {"ranking": "SDE (MXS) / T3E (MXT)", "count": LOCAL}},
            "unions": {u: {"seed_side": "ES[:ceil((1 - lam) C)]", "localised_side": "SDE[:ceil(lam %s)]" % LOCAL} for u in UMS},
            "gap_counts": {c_: GAP_RANK[c_.split("_", 1)[1]] for c_ in GAPC},
            "b_n_coupled_diagnostic": "ES|ESTAR_BN, ESTAR_BN = the distinct partitions of O'[:B_N]",
            "load_arms": ["%s|%s" % x for x in LOAD_ARMS], "oracle": {"A": ["%s|%s" % x for x in ORACLE_A], "B": "%s|%s" % ORACLE_B},
            "ranking_information": INFO, "topk": TOPK,
            "families": famrec, "seconds_families": round(t_fam, 1),
            "router_state (static)": {c: self.RS[c] for c in self.cells}, "seconds_router_state": round(t_rs, 1),
            "cells": {c: {"partition": P.tag, "npart": P.npart} for c, P in parts.items()},
            "extends": {"route_v2": {"path": D.rel(f2), "sha256": D.sha_file(f2), "npz_sha256": RV2["npz"]["sha256"]},
                        "route_v1": RV2["structures"]["extends"]["route_v1"]},
            "imports": dict(RV2["structures"]["imports"], **{"scratchpad/_l1d_route2.py": self.sha_rt2})}

    def row(self, j, of, od, os_, npos, fv, frank, g, sl):
        N = self.N
        top = of[:ACT]
        nh = len(top)
        # ---- L1a: the IR_L1 node score (the arithmetic of _l1d_route / _l1d_route2 for this rung)
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
        self.OQLEN[j] = len(oq)
        nz = np.flatnonzero(L)
        Lv = L[nz]
        wseed = 1.0 / np.arange(1.0, nh + 1.0)
        # ---- the regime shares (cell-independent)
        t0 = time.perf_counter()
        ltot = float(L.sum())
        lam1 = min(max(1.0 - float(L[top].sum()) / ltot, 0.0), 1.0) if ltot > 0 else 0.0
        lam2 = float(np.isin(oq[:ACT], top, invert=True).mean())
        nl = min(ACT, len(Lord))
        lam3 = 1.0 - float(np.isin(Lord[:nl], top).sum()) / nl if nl else 0.0
        self.LAM[j] = (lam1, lam2, lam3)
        if not len(Lord):
            assert lam2 == 0.0
        self.RDN[j] = int(self.ROWLEN[top].sum())
        self.LAM_LAT.append(time.perf_counter() - t0)
        chk = j < CHECK_ROWS
        for c in self.cells:
            P = self.parts[c]
            npart, hard, TL = P.npart, P.hard, self.TL[c]
            self.RDS[c][j] = (int(self.FAMLEN[c][top].sum()), int(self.UNILEN[c][top].sum()))
            SC, RO, CN, CR = {}, {}, {}, {}
            # ---- reference scores and rankings (route v1 / v2 arithmetic)
            t1 = time.perf_counter()
            fa = np.minimum.reduceat(frk[P.order_nodes], P.ptr[:-1])
            hs = hard[top]
            SC["FSUM"] = np.bincount(hs, weights=wseed, minlength=npart)
            ph = hard[nz]
            SC["SUM"] = np.bincount(ph, weights=Lv, minlength=npart)
            o = np.lexsort((-Lv, ph))
            phs, lvs = ph[o], Lv[o]
            wr = np.arange(len(phs)) - np.searchsorted(phs, phs, side="left")
            m = wr < TOPK
            SC["TOP3"] = np.bincount(phs[m], weights=lvs[m], minlength=npart)
            Cm = np.bincount(hit * npart + hard[u], weights=x, minlength=nh * npart).reshape(nh, npart)
            assert np.allclose(Cm.sum(0), SC["SUM"], rtol=1e-9, atol=1e-15), "per-seed masses do not sum to SUM"
            SC["SDIV"] = wseed @ (Cm > 0)
            RO["S"] = np.argsort(fa, kind="stable")
            for r_ in ("TOP3", "SDIV"):
                RO[r_] = np.lexsort((fa, -SC[r_]))
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
            SC["SDIVS"] = SC["SDIV"] + SC["FSUM"]
            for r_ in ("NSUM", "SDIVS"):
                RO[r_] = np.lexsort((esr, -SC[r_]))
            for x_ in ("SDIV", "TOP3"):
                kn, npz_ = RT.knee(SC[x_])
                CR["KNEE_" + x_] = kn if npz_ else np.nan
                CN["KNEE_" + x_] = kn if npz_ else npart
                _, hinf, npz_ = RT2.hill(SC[x_])
                CR["BPI_" + x_] = hinf if npz_ else np.nan
                CN["BPI_" + x_] = AB.bp_of(hinf, npz_, npart) if npz_ else npart
            if not (SC["TOP3"] > 0).any():
                self.NOEV[c] += 1
            CN["NSEEDP"] = CR["NSEEDP"] = int(len(np.unique(hs)))
            CN["NOACT_E"] = CR["NOACT_E"] = int(len(np.unique(hq[:ACT])))
            lam4 = min(max(1.0 - CN[LOCAL] / float(CN["NSEEDP"]), 0.0), 1.0)
            if not (SC["SDIV"] > 0).any():
                assert lam4 == 0.0
            self.LAM4[c][j] = lam4
            lam = (lam1, lam2, lam3, lam4)
            TL["reference"].append(time.perf_counter() - t1)
            # ---- the ES-tied score rankings, the sketch-only SDK, the gap counts
            t1 = time.perf_counter()
            RO["FSE"] = np.lexsort((esr, -SC["FSUM"]))
            RO["SUME"] = np.lexsort((esr, -SC["SUM"]))
            RO["T3E"] = np.lexsort((esr, -SC["TOP3"]))
            RO["SDE"] = np.lexsort((esr, -SC["SDIV"]))
            RO["SDK"] = np.lexsort((np.arange(npart), -SC["FSUM"], -SC["SDIV"]))
            for x_ in GAPX:
                g_, lg_ = gaps(SC[x_], npart)
                CN["GAP_" + x_] = CR["GAP_" + x_] = g_
                CN["LGAP_" + x_] = CR["LGAP_" + x_] = lg_
            TL["base_gaps"].append(time.perf_counter() - t1)
            # ---- the regime mixtures: rankings, counts, unions
            t1 = time.perf_counter()
            sdr, t3r = RT.inv(RO["SDE"]), RT.inv(RO["T3E"])
            B = CN[LOCAL]
            if chk:
                for rX, X in ((sdr, "SDE"), (t3r, "T3E")):
                    assert (mix_rank(esr, rX, 0.0) == RO["ES"]).all() and (mix_rank(esr, rX, 1.0) == RO[X]).all(), "mixture endpoints (%s %s)" % (c, X)
                for _, cn in CSRC:
                    e0, e1 = mix_counts(CN[cn], B, 0.0, npart), mix_counts(CN[cn], B, 1.0, npart)
                    assert e0[2] == e0[3] == CN[cn] and e1[2] == e1[3] == B, "count mixture endpoints (%s %s)" % (c, cn)
            for li, l in enumerate(LAMS):
                lm = lam[li]
                RO["MXS_" + l] = mix_rank(esr, sdr, lm)
                RO["MXT_" + l] = mix_rank(esr, t3r, lm)
                for k, cn in CSRC:
                    C_ = CN[cn]
                    ma, mg, cma, cmg = mix_counts(C_, B, lm, npart)
                    CR["MA_%s_%s" % (k, l)], CN["MA_%s_%s" % (k, l)] = ma, cma
                    CR["MG_%s_%s" % (k, l)], CN["MG_%s_%s" % (k, l)] = mg, cmg
                    cE, cL = max(ceil_tol((1.0 - lm) * C_), 0), max(ceil_tol(lm * B), 0)
                    assert cE + cL >= 1
                    ro, n_, inU = union0(esr, cE, sdr, cL, esr)
                    if chk:
                        assert inU[ro[:n_]].all() and not inU[ro[n_:]].any(), "union prefix != union (%s %s %s)" % (c, k, l)
                    un = "UM_%s_%s" % (k, l)
                    RO[un] = ro
                    CN[un] = CR[un] = n_
            TL["mixtures_unions"].append(time.perf_counter() - t1)
            cnt = np.array([CN[k_] for k_ in COUNTS], np.int64)
            assert (cnt >= 1).all() and (cnt <= npart).all(), (c, dict(zip(COUNTS, cnt.tolist())))
            self.CNT[c][j] = cnt
            self.CRAW[c][j] = [CR.get(k_, np.nan) for k_ in COUNTS]
            # ---- PSTAR / ESTAR (B_N-coupled) and the load of every load arm
            t1 = time.perf_counter()
            for mi, M in enumerate(D.M_CURVE):
                ps_ = int(len(np.unique(hard[FO[:M]])))
                pin, ein = fa < M, fe < M
                assert ps_ == int(pin.sum())
                self.PSTAR[c][j, mi] = ps_
                self.ESTAR[c][j, mi] = int(ein.sum())
                self.ESREC[c][j, mi] = float((pin & ein).sum()) / ps_
            for ai, (rk, cn) in enumerate(LOAD_ARMS):
                if cn.startswith("PSTAR_BN@"):
                    b = int(self.PSTAR[c][j, MI_LOAD])
                elif cn.startswith("ESTAR_BN@"):
                    b = int(self.ESTAR[c][j, MI_LOAD])
                else:
                    b = CN[cn]
                self.LOAD[c][ai, RO[rk][:b]] += 1
            TL["estar_load"].append(time.perf_counter() - t1)
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
        assert D.sha_file(RT2_PATH) == self.sha_rt2 and D.sha_file(RT_PATH) == self.sha_rt, "imported code changed during the run"
        allv = lambda pos, M: D.per_query(pos, M, gptr, ngold)[0]
        diag, arrays = {}, {}
        # (0) identities against node1h v1 and route v2
        z = np.load(self.node1h_npz)
        assert (z["rows"][:nq] == pop.rows).all() and (z["gptr"][:nq + 1] == gptr).all()
        assert (z["pos_LOC__" + NODE][:ng] == POS[NODE]["LOC"]).all(), "IR_L1 LOC positions differ from node1h v1"
        assert (z["pos_FLATLOC__" + NODE][:ng] == POS[NODE]["FLAT+LOC"]).all(), "IR_L1 FLAT+LOC positions differ from node1h v1"
        assert (self.FRK == POS[NODE]["FLAT+LOC"]).all()
        zr = np.load(self.route2_npz)
        assert (zr["rows"][:nq] == pop.rows).all() and (zr["gptr"][:nq + 1] == gptr).all()
        vr, vc = [str(s) for s in zr["ranks"]], [str(s) for s in zr["counts"]]
        for c in self.cells:
            for rk in REF_RANKS:
                a, b = vr.index(rk), RANKS.index(rk)
                assert (zr["LO__" + c][:nq, a] == self.LO[c][:, b]).all(), "LO differs from route v2 (%s %s)" % (c, rk)
                assert (zr["HI__" + c][:nq, a] == self.HI[c][:, b]).all(), "HI differs from route v2 (%s %s)" % (c, rk)
                assert (zr["PRG__" + c][:ng, a] == self.PRG[c][:, b]).all(), "PRG differs from route v2 (%s %s)" % (c, rk)
            for cn in REF_COUNTS:
                assert (zr["CNT__" + c][:nq, vc.index(cn)] == self.CNT[c][:, COUNTS.index(cn)]).all(), "count differs from route v2 (%s %s)" % (c, cn)
            assert (zr["PSTAR__" + c][:nq] == self.PSTAR[c]).all(), "PSTAR differs from route v2 (%s)" % c
        diag["identity"] = ("IR_L1 LOC and FLAT+LOC positions == node1h v1 on all %d gold nodes; LO / HI / PRG of %s, the counts %s and "
                            "PSTAR == route v2 in every cell; every partition contacted == unrouted (per row, per ranking); PSTAR_BN == "
                            "the partitions whose first O rank is < B_N (every row); on the first %d rows of every cell: every ranking "
                            "a permutation, every union prefix == the union, the mixtures at lam = 0 / 1 == ES / SDE (T3E), MA / MG at "
                            "lam = 0 / 1 == C / %s" % (ng, list(REF_RANKS), list(REF_COUNTS), CHECK_ROWS, LOCAL))
        diag["checked_rows"] = self.checks
        unr = {M: allv(POS[NODE]["FLAT+LOC"], M) for M in D.M_CURVE}
        hop_m = ST.get("per_hop", {})
        diag["regime_shares (L4 per cell)"] = {l: {"definition": LAM_DEF[l], "distribution": AB.dist(self.LAM[:, li]),
                                                   "per_hop": {hk: AB.dist(self.LAM[qm, li]) for hk, qm in hop_m.items()} if hop_m else None}
                                               for li, l in enumerate(LAMS[:3])}
        diag["router_reads_per_query (entries read for the 200 seeds)"] = {
            "node_rows (all four families)": AB.dist(self.RDN),
            "per_cell": {c: {"sketch_per_family_summed": AB.dist(self.RDS[c][:, 0]), "sketch_union": AB.dist(self.RDS[c][:, 1])}
                         for c in self.cells}}
        diag["router_order_length |O'| = |H_q u {L > 0}|"] = AB.dist(self.OQLEN)
        cells_out = {}
        for c in self.cells:
            P = self.parts[c]
            npart = P.npart
            LO, HI, CNT = self.LO[c].astype(np.int64), self.HI[c].astype(np.int64), self.CNT[c].astype(np.int64)
            e_c = {"unrouted_ALL": {str(M): D.q4(unr[M].mean()) for M in D.M_CURVE}, "rankings": {}, "counts": {}, "native_arms": {}}
            LAM = np.column_stack([self.LAM, self.LAM4[c]])
            e_c["regime_share_L4"] = {"distribution": AB.dist(LAM[:, 3]),
                                      "per_hop": {hk: AB.dist(LAM[qm, 3]) for hk, qm in hop_m.items()} if hop_m else None}
            e_c["regime_share_spearman"] = {"%s~%s" % (LAMS[a], LAMS[b]): AB.spearman(LAM[:, a], LAM[:, b])
                                            for a in range(len(LAMS)) for b in range(a + 1, len(LAMS))}
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
            # (10) the B_N-coupled router-side lossless estimate
            ri_es = RANKS.index("ES")
            bn = {}
            for mi, M in enumerate(D.M_CURVE):
                b = self.ESTAR[c][:, mi].astype(np.int64)
                a_ = AB.served_at(LO[:, ri_es], HI[:, ri_es, mi], b)
                bn[str(M)] = {"ALL": D.q4(a_.mean()), "delta_vs_unrouted": D.q4(a_.mean() - unr[M].mean()),
                              "paired_vs_unrouted": D.paired(unr[M], a_), "ESTAR_BN": AB.dist(b), "PSTAR_BN": AB.dist(self.PSTAR[c][:, mi]),
                              "ESTAR_fraction_of_partitions_mean": D.q4(b.mean() / npart),
                              "share_of_PSTAR_partitions_inside_ESTAR": AB.dist(self.ESREC[c][:, mi]),
                              "rows_with_every_PSTAR_partition_inside_ESTAR": D.q4((self.ESREC[c][:, mi] >= 1.0 - 1e-12).mean()),
                              "rows_with_|O'| < B_N": D.q4((self.OQLEN < M).mean()),
                              "lost_reach (ESTAR < LO_ES)": D.q4((b < LO[:, ri_es]).mean()),
                              "lost_crowding (LO <= HI < ESTAR)": D.q4(((LO[:, ri_es] <= HI[:, ri_es, mi]) & (HI[:, ri_es, mi] < b)).mean())}
            e_c["b_n_coupled_arm ES|ESTAR_BN"] = bn
            # oracle regime switch between A = ES|C and B = SDE|BPI_SDIV (gold-dependent; used by no rule)
            orc = {}
            rB, kB = RANKS.index(ORACLE_B[0]), COUNTS.index(ORACLE_B[1])
            for rkA, cnA in ORACLE_A:
                rA, kA = RANKS.index(rkA), COUNTS.index(cnA)
                o_ = {}
                for mi, M in enumerate(D.M_CURVE):
                    A_ = AB.served_at(LO[:, rA], HI[:, rA, mi], CNT[:, kA])
                    B_ = AB.served_at(LO[:, rB], HI[:, rB, mi], CNT[:, kB])
                    ao, bo = A_ & ~B_, B_ & ~A_
                    o_[str(M)] = {"A_ALL": D.q4(A_.mean()), "B_ALL": D.q4(B_.mean()), "oracle_switch_ALL": D.q4((A_ | B_).mean()),
                                  "oracle_minus_unrouted": D.q4((A_ | B_).mean() - unr[M].mean()),
                                  "A_only": int(ao.sum()), "B_only": int(bo.sum()), "both": int((A_ & B_).sum()), "neither": int((~A_ & ~B_).sum()),
                                  "AUC_lam (B-only vs A-only)": {l: auc(LAM[bo, li], LAM[ao, li]) for li, l in enumerate(LAMS)}}
                orc["%s|%s vs %s|%s" % (rkA, cnA, ORACLE_B[0], ORACLE_B[1])] = o_
            e_c["oracle_regime_switch"] = orc
            # (11) load profile of every load arm
            e_c["load (per-partition contacts over the population)"] = {
                "%s|%s" % (rk, cn): load_stats(self.LOAD[c][ai], nq) for ai, (rk, cn) in enumerate(LOAD_ARMS)}
            e_c["queries_without_localised_evidence"] = self.NOEV[c]
            cells_out[c] = e_c
            for nm_, A_ in (("LO", self.LO), ("HI", self.HI), ("PRG", self.PRG), ("CNT", self.CNT), ("CRAW", self.CRAW),
                            ("CMASS", self.CMASS), ("PSTAR", self.PSTAR), ("MSUM", self.MSUM), ("RET", self.RET),
                            ("ESTAR", self.ESTAR), ("ESREC", self.ESREC), ("LOAD", self.LOAD), ("RDS", self.RDS), ("LAM4", self.LAM4)):
                arrays["%s__%s" % (nm_, c)] = A_[c]
        diag["cells"] = cells_out
        diag["latency_ms"] = {"node_score_and_LOC_order": D.ms_stats(self.lat[NODE]), "fused_order": D.ms_stats(self.FUSE_LAT),
                              "regime_shares": D.ms_stats(self.LAM_LAT),
                              "per_cell": {c: {k: D.ms_stats(v) for k, v in self.TL[c].items()} for c in self.cells}}
        arrays["LAM"] = self.LAM
        arrays["RDN"] = self.RDN
        arrays["OQLEN"] = self.OQLEN
        arrays["ranks"] = np.array(RANKS)
        arrays["counts"] = np.array(COUNTS)
        arrays["native"] = np.array(["%s|%s" % x for x in NATIVE])
        arrays["load_arms"] = np.array(["%s|%s" % x for x in LOAD_ARMS])
        arrays["m_curve"] = MSA
        return diag, arrays


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    assert mode == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]"
    R.run(sys.argv[2], sys.argv[3], "route3", "L1_DEVELOPMENT_ROUTE3", __file__, Route3Spec, __doc__)


if __name__ == "__main__":
    main()
