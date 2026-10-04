"""L1 DEVELOPMENT -- KSCALE: K-aware fan-out scaling of ES.KNEE_SDIV.  (user 2026-09-29, a forwarded direction after REPORT
section 35: "What is failing at large K is not partitioning itself. It is the count policy." ... "B_P(q,K) = B_ES.KNEE(q,K0)
(K/K0)^alpha with a fixed, non-learned alpha. I would not blindly tune alpha, but testing something like alpha in
{0.65,0.75,0.85,1.0} is now a very targeted systems experiment" ... "The goal would be to find the smallest scaling law that
preserves: R_ALL(K) while still keeping: B_P/K decreasing." ... "Keep everything else fixed: IR_L1; ES shard ordering; same
partitions; same node budget; no learning; no new structural mechanism. Only modify how the number of shards scales with K.")
Development numbers; any p-value is descriptive.

Design (fixed before any result of this harness was seen):
  count    B_P^alpha(q, K) = min(K, ceil(KNEE_SDIV(q; P_ref) * (K / K_REF)^alpha)), K_REF = 100, alpha in {0.65, 0.75, 0.85, 1.0}.
           KNEE_SDIV(q; P_ref) is section 35's count on the same partitioner's map at K = 100 (for RAND<s> the same seed's map); the
           direction's K0 is this reference granularity (not the RRF constant K0 = 60).  A query without localised evidence contacts
           every partition (B_P = K), as in section 35.  The multiplier is exact: per K a table t[b0] = ceil(b0 * (K / 100)^alpha),
           b0 = 1 .. 100, in Decimal arithmetic (precision 60, then ROUND_CEILING); alpha = 1.0 is exact (K / 100 is a finite
           decimal); the record states the smallest distance of a non-integral product to an integer and whether float64
           arithmetic gives the same table.
  K_REF    the coarsest K of the grid, the same constant on every dataset.  Every cell has K >= K_REF, so the count is
           non-decreasing in alpha at every K (asserted) and "the smallest alpha" is well defined; at K = K_REF every alpha arm is
           section 35's ES.KNEE_SDIV exactly (asserted), so the K = 100 losses of section 35 remain by construction; the cap K never
           binds strictly (b0 <= K_REF, asserted).
  fixed    IR_L1, the served order O, the router order O' and the ES ranking on the K map, the partitions (section 35's cells),
           the node budgets B_N (M_CURVE), the population.  This harness reruns section 35's computation (scratchpad/_l1d_scale.py,
           imported and subclassed, not edited) and asserts every per-query array equal to the scale v1 record.
  deploy   the count needs KNEE_SDIV on the K = 100 map, so the router keeps that map's node -> shard table (N entries) besides the
           K map's (router-side state; no global order).  Both maps of a (structural, random) pair scale by the same factor, so the
           structural / random fan-out ratio stays near its K = 100 value.
Measured per cell and arm (the four alpha arms and section 35's ES.KNEE_SDIV on the K map): exact counts of ALL gold served at
B_N in M_CURVE, against unrouted and against ES.KNEE_SDIV (paired, descriptive McNemar), lost to reach (B_P < LO) / crowding
(LO <= HI < B_P), per hop; B_P(q), rho(q) = B_P / K, CMASS(q) (the nodes inside the contacted partitions) and CMASS / N; the
per-partition load (route v3's load_stats); structural vs balanced random at the same K (paired ALL of the same arm).
Identities asserted: every identity of scale v1 (through ScaleSpec.finish: node1h v1, route v1 / v2 / v3 on the served cells,
every partition contacted == unrouted, S at PSTAR == unrouted); every per-query array of ScaleSpec (LO, HI, PRG, CNT, CMASS, PSTAR,
ESTAR, NGP, NSEEDP, NEV, XC, RDS, ENT, RDN, OQLEN, HOPS) and the runner's positions == scale v1's npz, and on the full population its
MSUM and LOAD; per row and cell, the recomputed ES order gives scale v1's LO and CMASS; at K = 100 every alpha arm's counts, masses
and load == ES.KNEE_SDIV's; every count within [1, K] and non-decreasing in alpha; the evidence status of a query is the same in
every cell.

Usage: python scratchpad/_l1d_kscale.py RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L1_DEV/kscale_<dataset>__<tag>.{json,npz} (write-once)
"""
import decimal
import json
import math
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_arms as R
import _l1d_adaptbp as AB
import _l1d_route as RT
import _l1d_route3 as RT3
import _l1d_scale as SC

log = D.log
K0, ACT = D.K0, D.ACT
NODE = SC.NODE
K_REF = 100
ALPHAS = ("0.65", "0.75", "0.85", "1.0")
DEC_PREC = 60
DIST_MIN = decimal.Decimal("1e-30")          # a non-integral product this close to an integer would make the ceiling fragile
BASE = "ES|KNEE_SDIV"
ANAME = {a: "ES|KSCALE_a%s" % a for a in ALPHAS}
ARM_ORDER = (BASE,) + tuple(ANAME[a] for a in ALPHAS)
RULE = ("B_P(q, K) = min(K, ceil(KNEE_SDIV(q; the same partitioner's K = %d map) * (K / %d)^alpha)); a query without localised "
        "evidence contacts every partition; ranking ES on the K map" % (K_REF, K_REF))
SCALE_PATH = os.path.join(D.HERE, "_l1d_scale.py")
PER_Q = ("LO", "HI", "CNT", "CMASS", "PSTAR", "ESTAR", "NGP", "NSEEDP", "NEV", "XC", "RDS")
PER_G = ("PRG",)
POP_SUMS = ("MSUM", "LOAD")
Q_SIDE = ("ENT", "RDN", "OQLEN", "HOPS")
IES = SC.RANKS.index("ES")
assert IES == 0 and SC.LOAD_ARMS[0] == BASE and SC.COUNT == "KNEE_SDIV" and min(SC.KGRID) == K_REF


def mult_tables(K):
    """({alpha: int64[K_REF + 1]} with t[b0] = min(K, ceil(b0 * (K / K_REF)^alpha)), t[0] = 0; the record of the arithmetic)."""
    ctx = decimal.Context(prec=DEC_PREC)
    q = ctx.divide(decimal.Decimal(K), decimal.Decimal(K_REF))
    tabs, rec = {}, {"K": K, "K_over_K_REF": str(q), "multiplier": {}, "min_distance_to_integer": {}, "float64_same_table": {}}
    for a in ALPHAS:
        r = ctx.power(q, decimal.Decimal(a))
        if decimal.Decimal(a) == 1:
            assert r == q, (K, a, r)
        t = np.zeros(K_REF + 1, np.int64)
        mind, same = None, True
        for b0 in range(1, K_REF + 1):
            v = ctx.multiply(decimal.Decimal(b0), r)
            c = int(v.to_integral_value(rounding=decimal.ROUND_CEILING))
            if v != c:
                d = min(v - (c - 1), c - v)
                mind = d if mind is None or d < mind else mind
            assert c <= K, "the cap binds strictly (K %d alpha %s b0 %d -> %d)" % (K, a, b0, c)
            t[b0] = c
            same &= math.ceil(b0 * (K / float(K_REF)) ** float(a)) == c
        assert mind is None or mind > DIST_MIN, (K, a, mind)
        assert (np.diff(t[1:]) >= 0).all() and t[1] >= 1
        tabs[a] = t
        rec["multiplier"][a] = str(ctx.create_decimal(r).quantize(decimal.Decimal("1e-15"))) if r != q else str(q)
        rec["min_distance_to_integer"][a] = None if mind is None else "%.3e" % mind
        rec["float64_same_table"][a] = bool(same)
    return tabs, rec


class KScaleSpec(SC.ScaleSpec):
    def __init__(self, cd, pop, parts_served):
        ds = cd.name
        # ---- the scale v1 record this harness reproduces (its harness must be the code imported here)
        fp = os.path.join(D.OUT, "scale_%s__v1.json" % ds)
        rec = json.load(open(fp, encoding="utf-8"))
        self.scale_npz = os.path.join(D.OUT, rec["npz"]["path"])
        assert D.sha_file(self.scale_npz) == rec["npz"]["sha256"], "scale v1 npz changed"
        assert D.sha_file(SCALE_PATH) == rec["code"]["harness"]["sha256"], "_l1d_scale.py differs from the code of scale v1"
        assert rec["code"]["arms"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_arms.py"))
        assert rec["code"]["lib"]["sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))
        self.scale_rec = {"path": D.rel(fp), "sha256": D.sha_file(fp), "npz": D.rel(self.scale_npz), "npz_sha256": rec["npz"]["sha256"],
                          "n_rows": rec["n_rows"], "harness_sha256": rec["code"]["harness"]["sha256"]}
        super(KScaleSpec, self).__init__(cd, pop, parts_served)
        z = np.load(self.scale_npz)
        assert [str(s) for s in z["cells"]] == self.cells, "cells differ from scale v1"
        assert sorted(rec["structures"]["absent_cells"]) == sorted(self.absent), "absent cells differ from scale v1"
        nq, na = pop.nq, len(ALPHAS)
        # ---- the reference map of each cell and the exact multiplier tables (one per K)
        self.ref, self.TAB, tabrec = {}, {}, {}
        for c in self.cells:
            cr = self.cellrec[c]
            pz, K = cr["partitioner"], cr["K"]
            assert K >= K_REF and self.parts[c].npart == K
            rc = SC.cell_name("RAND%d" % cr["seed"] if pz == "RAND" else pz, K_REF)
            assert rc in self.parts and self.parts[rc].npart == K_REF, (c, rc)
            self.ref[c] = rc
            if K not in tabrec:
                tabs, tabrec[K] = mult_tables(K)
                tabrec[K]["_tabs"] = np.stack([tabs[a] for a in ALPHAS])
            self.TAB[c] = tabrec[K]["_tabs"]
        self.CNTA = {c: np.zeros((nq, na), np.int32) for c in self.cells}
        self.CMASSA = {c: np.zeros((nq, na), np.int64) for c in self.cells}
        self.LOADA = {c: np.zeros((na, self.parts[c].npart), np.int64) for c in self.cells}
        self.NOEVQ = np.zeros(nq, bool)
        self.KLAT = []
        self.kchecks = 0
        self.record["kscale"] = {
            "rule": RULE, "K_REF": K_REF, "alphas": list(ALPHAS), "arms": {ANAME[a]: "alpha = %s" % a for a in ALPHAS},
            "reference_arm": "%s (section 35: KNEE_SDIV on the K map)" % BASE,
            "reference_cells": self.ref,
            "multiplier_tables": {str(K): {k: v for k, v in r_.items() if k != "_tabs"} for K, r_ in sorted(tabrec.items())},
            "decimal_precision": DEC_PREC,
            "deployment": "the router keeps the K = %d map's node -> shard table (N entries) besides the K map's" % K_REF,
            "reproduces_scale_v1": self.scale_rec}
        log("KSCALE %s: %d cells, K_REF %d, alphas %s, tables for K %s" % (ds, len(self.cells), K_REF, list(ALPHAS), sorted(tabrec)))

    def row(self, j, of, od, os_, npos, fv, frank, g, sl):
        noev0 = dict(self.NOEV)
        res = super(KScaleSpec, self).row(j, of, od, os_, npos, fv, frank, g, sl)
        t0 = time.perf_counter()
        inc = set(self.NOEV[c] - noev0[c] for c in self.cells)
        assert len(inc) == 1, "the evidence status differs between cells (row %d)" % j
        noev = inc.pop() == 1
        self.NOEVQ[j] = noev
        # ---- the router's order O' (ScaleSpec.row's arithmetic; ScaleSpec keeps only its length)
        top = of[:ACT]
        nh = len(top)
        Lord = res[NODE]
        vq = np.union1d(top, Lord)
        fq = np.zeros(len(vq))
        tq = np.zeros(len(vq))
        fq[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
        il = np.searchsorted(vq, Lord)
        fq[il] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
        tq[il] = ACT + np.arange(len(Lord))
        tq[np.searchsorted(vq, top)] = np.arange(nh)
        oq = vq[np.lexsort((tq, -fq))]
        assert len(oq) == self.OQLEN[j]
        for c in self.cells:
            P = self.parts[c]
            npart, hard = P.npart, P.hard
            up, first = np.unique(hard[oq], return_index=True)
            fe = np.full(npart, len(oq), np.int64)
            fe[up] = first
            ro = np.lexsort((np.arange(npart), fe))
            ms = np.cumsum(P.sizes[ro])
            assert ms[int(self.CNT[c][j]) - 1] == self.CMASS[c][j, IES], "recomputed ES mass differs (%s row %d)" % (c, j)
            assert int(RT.inv(ro)[hard[g]].max()) + 1 == self.LO[c][j, IES], "recomputed ES order differs (%s row %d)" % (c, j)
            b0 = int(self.CNT[self.ref[c]][j])
            for ai in range(len(ALPHAS)):
                b = npart if noev else int(self.TAB[c][ai, b0])
                assert 1 <= b <= npart
                self.CNTA[c][j, ai] = b
                self.CMASSA[c][j, ai] = ms[b - 1]
                self.LOADA[c][ai, ro[:b]] += 1
        self.kchecks += 1
        self.KLAT.append(time.perf_counter() - t0)
        return res

    def finish(self, ctx):
        POS, gptr, ngold, pop, ST = ctx["POS"], ctx["gptr"], ctx["ngold"], ctx["pop"], ctx["ST"]
        nq, ng, N = pop.nq, pop.ng_tot, self.N
        na = len(ALPHAS)
        assert D.sha_file(SCALE_PATH) == self.scale_rec["harness_sha256"], "_l1d_scale.py changed during the run"
        sdiag, sarr = super(KScaleSpec, self).finish(ctx)
        # (0) section 35 reproduced: every array of ScaleSpec and the runner's positions == scale v1's npz
        z = np.load(self.scale_npz)
        full = nq == self.scale_rec["n_rows"]
        assert (z["rows"][:nq] == pop.rows).all() and (z["gptr"][:nq + 1] == gptr).all()
        assert (z["pos_FLAT"][:ng] == ctx["POS_FLAT"]).all()
        for key, arm in (("pos_LOC__", "LOC"), ("pos_FLATLOC__", "FLAT+LOC"), ("lpos__", "lpos")):
            assert (z[key + NODE][:ng] == POS[NODE][arm]).all(), "%s differs from scale v1" % key
        assert (z["loc_len__" + NODE][:nq] == ctx["LLEN"][NODE]).all()
        for c in self.cells:
            for nm_ in PER_Q:
                assert (z["%s__%s" % (nm_, c)][:nq] == sarr["%s__%s" % (nm_, c)]).all(), "%s differs from scale v1 (%s)" % (nm_, c)
            for nm_ in PER_G:
                assert (z["%s__%s" % (nm_, c)][:ng] == sarr["%s__%s" % (nm_, c)]).all(), "%s differs from scale v1 (%s)" % (nm_, c)
            assert (z["SIZES__" + c] == sarr["SIZES__" + c]).all()
            if full:
                for nm_ in POP_SUMS:
                    assert np.array_equal(z["%s__%s" % (nm_, c)], sarr["%s__%s" % (nm_, c)]), "%s differs from scale v1 (%s)" % (nm_, c)
        for nm_ in Q_SIDE:
            assert (z[nm_][:nq] == sarr[nm_]).all(), "%s differs from scale v1" % nm_
        # (1) the alpha arms: K = 100 invariance, range, monotone in alpha
        for c in self.cells:
            npart = self.parts[c].npart
            A = self.CNTA[c].astype(np.int64)
            assert (A >= 1).all() and (A <= npart).all()
            assert (np.diff(A, axis=1) >= 0).all(), "a count decreases in alpha (%s)" % c
            if npart == K_REF:
                assert self.ref[c] == c
                for ai in range(na):
                    assert (A[:, ai] == self.CNT[c]).all(), "K = %d count != ES.KNEE_SDIV (%s)" % (K_REF, c)
                    assert (self.CMASSA[c][:, ai] == self.CMASS[c][:, IES]).all() and (self.LOADA[c][ai] == self.LOAD[c][0]).all()
        diag = {"identity": ("section 35 (scale v1, %s sha %s) reproduced: its identities (%s) and every per-query array %s, PRG, the "
                             "query-side arrays %s and the runner's IR_L1 positions == its npz%s; per row and cell the recomputed ES order "
                             "gives its LO and CMASS; at K = %d every alpha arm's counts, masses and load == %s; every count in [1, K] "
                             "and non-decreasing in alpha; the evidence status of each query is the same in every cell"
                             % (self.scale_rec["path"], self.scale_rec["sha256"][:16], sdiag["identity"], list(PER_Q), list(Q_SIDE),
                                ", and MSUM / LOAD (full population)" if full else " (first rows; MSUM / LOAD not compared)",
                                K_REF, BASE)),
                "checked_rows_section35": sdiag["checked_rows"], "kscale_rows": self.kchecks,
                "queries_without_localised_evidence": int(self.NOEVQ.sum()), "unrouted_ALL": sdiag["unrouted_ALL"]}
        # (2) per cell and arm
        allv = lambda pos, M: D.per_query(pos, M, gptr, ngold)[0]
        unr = {M: allv(POS[NODE]["FLAT+LOC"], M) for M in D.M_CURVE}
        hop_m = ST.get("per_hop", {})
        cells_out, SERV = {}, {}
        for c in self.cells:
            P = self.parts[c]
            npart = P.npart
            LO = self.LO[c][:, IES].astype(np.int64)
            HI = self.HI[c][:, IES].astype(np.int64)
            for mi, M in enumerate(D.M_CURVE):
                assert (AB.served_at(LO, HI[:, mi], npart) == unr[M]).all(), "every partition != unrouted (%s %d)" % (c, M)
            base_b = self.CNT[c].astype(np.int64)
            arms = {BASE: (base_b, self.CMASS[c][:, IES], self.LOAD[c][0])}
            for ai, a in enumerate(ALPHAS):
                arms[ANAME[a]] = (self.CNTA[c][:, ai].astype(np.int64), self.CMASSA[c][:, ai], self.LOADA[c][ai])
            base_serv = {M: AB.served_at(LO, HI[:, mi], base_b) for mi, M in enumerate(D.M_CURVE)}
            e_c = {"partitioner": self.cellrec[c]["partitioner"], "K": npart, "native": self.cellrec[c]["native"],
                   "reference_cell": self.ref[c], "arms": {}}
            for an in ARM_ORDER:
                b, cm, ld = arms[an]
                e = {"B_P(q)": AB.dist(b), "B_P_sum": int(b.sum()), "rho(q) = B_P / K": AB.dist(b / float(npart)),
                     "contacted_mass CMASS": AB.dist(cm), "CMASS_sum": int(cm.sum()),
                     "contacted_mass_fraction_of_N": AB.dist(cm / float(N)), "load": RT3.load_stats(ld, nq)}
                if an != BASE:
                    e["B_P_over_%s (per query)" % BASE] = AB.dist(b / base_b.astype(np.float64))
                    e["queries_above_below_%s" % BASE] = [int((b > base_b).sum()), int((b < base_b).sum())]
                bn = {}
                for mi, M in enumerate(D.M_CURVE):
                    a_ = AB.served_at(LO, HI[:, mi], b)
                    SERV[(an, c, M)] = a_
                    ee = {"ALL": D.q4(a_.mean()), "n_ALL": int(a_.sum()), "n_unrouted": int(unr[M].sum()),
                          "delta_vs_unrouted": D.q4(a_.mean() - unr[M].mean()), "paired_vs_unrouted": D.paired(unr[M], a_),
                          "n_lost_reach (B_P < LO)": int((b < LO).sum()),
                          "n_lost_crowding (LO <= HI < B_P)": int(((LO <= HI[:, mi]) & (HI[:, mi] < b)).sum()),
                          "n_unservable (LO > HI)": int((LO > HI[:, mi]).sum())}
                    if an != BASE:
                        ee["n_%s" % BASE] = int(base_serv[M].sum())
                        ee["paired_vs_%s" % BASE] = D.paired(base_serv[M], a_)
                    if hop_m:
                        ee["per_hop"] = {hk: {"n": int(qm.sum()), "n_ALL": int(a_[qm].sum()), "n_unrouted": int(unr[M][qm].sum())}
                                         for hk, qm in hop_m.items()}
                    bn[str(M)] = ee
                e["B_N"] = bn
                e_c["arms"][an] = e
            cells_out[c] = e_c
        diag["cells"] = cells_out
        # (3) structural vs balanced random at the same K, per arm (descriptive McNemar)
        svr = {}
        for c in self.cells:
            pz, K = self.cellrec[c]["partitioner"], self.cellrec[c]["K"]
            if pz not in SC.STRUCTURAL:
                continue
            o = {}
            for an in ARM_ORDER:
                o[an] = {}
                for s in SC.RAND_SEEDS:
                    rc = SC.cell_name("RAND%d" % s, K)
                    o[an]["RAND%d" % s] = {str(M): dict(D.paired(SERV[(an, rc, M)], SERV[(an, c, M)]),
                                                        n_structural=int(SERV[(an, c, M)].sum()), n_random=int(SERV[(an, rc, M)].sum()))
                                           for M in D.M_CURVE}
            svr[c] = o
        diag["structural_vs_random (gained = the structural cell serves ALL gold, the random cell does not)"] = svr
        diag["latency_ms_per_row (kscale addition: O', the ES order and the alpha counts in every cell)"] = D.ms_stats(self.KLAT)
        arrays = {}
        for c in self.cells:
            arrays["CNTA__" + c] = self.CNTA[c]
            arrays["CMASSA__" + c] = self.CMASSA[c]
            arrays["LOADA__" + c] = self.LOADA[c]
            arrays["LOES__" + c] = self.LO[c][:, IES].copy()
            arrays["HIES__" + c] = self.HI[c][:, IES].copy()
            arrays["CNT__" + c] = self.CNT[c]
            arrays["CMASSES__" + c] = self.CMASS[c][:, IES].copy()
            arrays["LOADES__" + c] = self.LOAD[c][0].copy()
            arrays["SIZES__" + c] = self.parts[c].sizes
            arrays["TAB__" + c] = self.TAB[c]
        arrays["HOPS"] = sarr["HOPS"]
        arrays["NOEVQ"] = self.NOEVQ
        arrays["cells"] = np.array(self.cells)
        arrays["ref_cells"] = np.array([self.ref[c] for c in self.cells])
        arrays["alphas"] = np.array(ALPHAS)
        arrays["m_curve"] = SC.MSA
        return diag, arrays


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    assert mode == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]"
    R.run(sys.argv[2], sys.argv[3], "kscale", "L1_DEVELOPMENT_KSCALE", __file__, KScaleSpec, __doc__)


if __name__ == "__main__":
    main()
