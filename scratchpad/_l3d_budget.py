"""L3 DEVELOPMENT -- BUDGET: typed completion (the TYPED v1 channel) under a read budget beyond the routed shards, at equal exposure.
(User 2026-09-29: "we need to finish the full KB path: L1 localization -> L2 reranking -> L3 multi-hop completion"; the advisor's
criteria: "Recall, fan-out, local scan, load skew".  TYPED v1 (results/L3_DEV/l3typed_metaqa__v1.json, a6e80216...) on PHG at B_N 1000:
T1_PUSH -- the typed walk expands only contacted nodes -- serves ALL gold on 1,602 of 1,998 rows; T1_ROWS -- any adjacency row may be
read -- on 1,967, reading a mean 210 (median 4) typed rows outside the contacted shards.  This harness puts a budget between the two and
measures recall against what is read.  STATUS: DEVELOPMENT -- descriptive numbers, every p-value descriptive, no budget selected here.)

L1, the routing and the typed channel are exactly those of _l3d_typed (imported, sha-pinned): every TYPED v1 arm is recomputed by
_l3d_typed.TypedSpec.row and its positions, contacted flags, costs, candidate ranks and gate diagnostics are asserted == the TYPED v1
record (the first rows of it for a smoke).  Development rows only: the L1_DEV population (c7f70806...), legacy-exposed; no reserved
held-out row, no split-B row and no TEST row is read.  No learned weight, no tuned constant (K0 60, EPS 1e-6, H 3 as in TYPED v1).

Budgeted typed walk (per routed cell, gated rows).  The PUSH walk of TYPED v1 expands only the mass on the readable nodes R_k; R_0 = the
contacted nodes, and before each step k = 1..H the readable set grows greedily from the frontier x_k (the typed walk's current layer):
  S<s>  shard budget: the frontier mass on nodes outside R_{k-1} is summed per shard of the cell's map; shards open in order of that
        mass (ties -> shard id) until s extra shards are open in total; every node of an open shard is readable at this and every
        later step (a shard fetch: the whole shard's adjacency rows)
  R<r>  row budget: frontier nodes outside R_{k-1} open one by one in order of mass (ties -> FLAT rank) until r extra node rows are
        open in total; an open row stays readable (a row fetch: one node's adjacency row)
  SINF  every frontier shard opens (no budget): the typed walk == the unmasked TYPED v1 walk (asserted per row, bit for bit)
The untyped EPS tie-break channel walks as in T1_PUSH (sources = contacted nodes): the budget is spent on the typed channel only.
(T1_ROWS of TYPED v1 also reads untyped rows outside the contacted shards; SINF vs T1_ROWS measures what that buys.)
S0 == R0 == T1_PUSH (the S0 walk is asserted == TYPED v1's PUSH walk per row).
CT_B = tcands(seed, ans_B, oth_B, unt_PUSH); the served list is T1: CT_B first, then the cell's L1 order; exposure n(q, B_N) =
min(B_N, contacted mass), as for every arm.
Arms added per routed cell: T1_SINF, T1_S1 .. T1_S128, T1_R1 .. T1_R1024 (the grids below; no value selected here).
Cost per row and arm: extra shards opened (S arms) or node rows opened (R arms); the nodes of the opened shards (S arms: the extra local
scan; R arms: the opened rows); typed rows read outside the contacted shards and their distinct shards; per B_N the served nodes
fetched from outside the contacted shards, their distinct extra shards, and the union of the walk's extra shards and the fetched ones
(the total fan-out beyond L1's B_P shards).

Usage: python scratchpad/_l3d_budget.py RUN metaqa <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L3_DEV/l3budget_<dataset>__<tag>.{json,npz} (write-once)
"""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_arms as R
import _l1d_route as RT
import _l1d_kbres as KB
import _l3d_hop as HOP
import _l3d_typed as TYP
import _l1x90_seeds as SE

log = D.log
K0, ACT = D.K0, D.ACT
FAMS = KB.FAMS
M_CURVE = D.M_CURVE
UNR = KB.UNR
OUT3 = HOP.OUT3
S_GRID = (1, 2, 4, 8, 16, 32, 64, 128)
R_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
BUD_ARMS = ("T1_SINF",) + tuple("T1_S%d" % s for s in S_GRID) + tuple("T1_R%d" % r for r in R_GRID)
TYPED_V1_JSON = os.path.join(OUT3, "l3typed_metaqa__v1.json")
TYPED_V1_JSON_SHA = "a6e80216a802d12ec91ded5d5a7d6c59f24e7339a2efd34f6e447486cc7b90a4"
TYPED_V1_NPZ = os.path.join(OUT3, "l3typed_metaqa__v1.npz")
TYPED_V1_NPZ_SHA = "0ee65e76660223d0378708582e8bd19142e58595ff495340705d5112d29e1355"
TYPED_MOD_SHA = "8eb4cbde3cc54c147f519a7aa0ae9974ae910ac6b32ad90ee673e53064266ac9"
MODULES = TYP.MODULES + ("_l3d_typed.py",)
positions, first_then, tcands = HOP.positions, TYP.first_then, TYP.tcands


class BudgetWalker(TYP.Walker):
    """TYPED v1's Walker plus the greedy budgeted typed walk (same arithmetic per step; only the source mask grows).  The edge
    subsets v[sel], u[sel], v[sl], u[sl] of a schedule prefix are cached: the same arrays, so the same bincount sums."""

    def __init__(self, T, N, H):
        TYP.Walker.__init__(self, T, N, H)
        self.subs = {}

    def sub_of(self, allowed, last):
        key = (tuple(sorted(allowed)), last)
        if key not in self.subs:
            T = self.T
            sel, cnt, lc = self.sel_of(allowed)
            if last not in lc:
                lc[last] = sel & (T.r == last)
            sl = lc[last]
            self.subs[key] = (T.v[sel], T.u[sel], T.v[sl], T.u[sl])
        return self.subs[key]

    def plan(self, x0, order, base, hard, frank, S=None, Rr=None):
        """(ans, oth, rows, opened shards, opened rows): the typed walk whose readable set starts at base and grows before each step."""
        T, N = self.T, self.N
        u, v, deg = T.u, T.v, T.deg
        x = x0.astype(np.float64)
        seen = x > 0
        ans = np.zeros(N)
        oth = np.zeros(N)
        rows = np.zeros(N, bool)
        last = order[-1] if order else None
        readable = base.copy()
        opened_sh = []
        opened_rows = []
        for k in range(1, self.H + 1):
            cand = np.flatnonzero((x > 0) & ~readable)
            if len(cand):
                if S is not None and len(opened_sh) < S:
                    ush, inv = np.unique(hard[cand], return_inverse=True)
                    sm = np.bincount(inv, weights=x[cand])
                    take = ush[np.lexsort((ush, -sm))][:S - len(opened_sh)]
                    opened_sh.extend(int(t) for t in take)
                    readable |= np.isin(hard, take)
                if Rr is not None and len(opened_rows) < Rr:
                    take = cand[np.lexsort((frank[cand], -x[cand]))][:Rr - len(opened_rows)]
                    opened_rows.extend(int(t) for t in take)
                    readable[take] = True
            # == TYPED v1 Walker.walk, one step, with srcm = readable
            allowed = set(order[:k]) if order else set()
            _, cnt, _ = self.sel_of(allowed)
            has = cnt > 0
            xs = np.where(readable, x, 0.0)
            rows |= xs > 0
            if allowed:
                src = xs / np.maximum(cnt, 1.0)
                src[~has] = 0.0
                vs, us, vl, ul = self.sub_of(allowed, last)
                y_typed = np.bincount(vs, weights=src[us], minlength=N).astype(np.float64)
                y_ans = np.bincount(vl, weights=src[ul], minlength=N).astype(np.float64)
                y_oth = y_typed - y_ans
            else:
                y_ans = np.zeros(N)
                y_oth = np.zeros(N)
            if not allowed:
                src_u = xs / np.maximum(deg, 1.0)
                y_oth = y_oth + np.bincount(v, weights=src_u[u], minlength=N).astype(np.float64)
            y_ans[seen] = 0.0
            y_oth[seen] = 0.0
            s = y_ans.sum() + y_oth.sum()
            if s > 0:
                y_ans /= s
                y_oth /= s
            x = y_ans + y_oth
            seen |= x > 0
            ans += y_ans
            oth += y_oth
        return ans, oth, rows, opened_sh, opened_rows


def budget_of(a, npart):
    if a == "T1_SINF":
        return npart, None
    if a.startswith("T1_S"):
        return int(a[4:]), None
    return None, int(a[4:])


class BudgetSpec(TYP.TypedSpec):
    def __init__(self, cd, pop, parts, Q):
        TYP.TypedSpec.__init__(self, cd, pop, parts, Q)
        self.W = BudgetWalker(self.T, self.N, TYP.HW)
        nq, ng = pop.nq, pop.ng_tot
        for c in self.cells:
            self.arms[c] = tuple(self.arms[c]) + BUD_ARMS
            for a in BUD_ARMS:
                self.POS[(c, a)] = np.full(ng, -1, np.int64)
                self.FETCH[(c, a)] = np.zeros((nq, len(M_CURVE)), np.int64)
                self.XSH[(c, a)] = np.zeros((nq, len(M_CURVE)), np.int64)
        self.BC = {(c, a): {k: np.zeros(nq, np.int64) for k in ("opened", "opened_shard_nodes", "rows_read_out", "rows_read_out_shards")}
                   for c in self.cells for a in BUD_ARMS}
        self.SHSIZE = {c: np.bincount(self.K.parts[c].hard, minlength=self.K.parts[c].npart) for c in self.cells}
        self.UXS = {(c, a): np.zeros((nq, len(M_CURVE)), np.int64) for c in self.cells for a in BUD_ARMS}
        for c in self.cells:
            self.LAT["%s budget walks (%d arms)" % (c, len(BUD_ARMS))] = []
        self.n_s0 = 0
        self.n_sinf = 0
        self.record["arms"] = {c: list(a) for c, a in self.arms.items()}
        self.record["budget"] = {"S_grid": list(S_GRID), "R_grid": list(R_GRID), "arms": list(BUD_ARMS),
                                 "untyped channel": "PUSH (sources = contacted nodes), as in T1_PUSH"}

    def row(self, j, of, od, os_, npos, frank, g, sl, d_top1):
        TYP.TypedSpec.row(self, j, of, od, os_, npos, frank, g, sl, d_top1)
        K, N, F, W = self.K, self.N, self.F, self.W
        # ---- the carried-forward L1 served order and router order (== TypedSpec.row; checked below through the S0 identity)
        top = of[:ACT]
        nh = len(top)
        U, H, GW = [], [], []
        for f_ in FAMS:
            Fm = F[f_]
            st = Fm["xadj"][top]
            hit, pos = HOP.E.entries(st, Fm["xadj"][top + 1] - st)
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
        # ---- seeds, schedule, gate (== TypedSpec.row)
        Q = self.Q[j]
        lex = SE.lexical_seeds(Q, self.idx, self.names, self.T)
        seeds = [int(s_) for s_ in lex[:5]] if lex else [int(od[0])] + ([int(os_[0])] if npos > 0 and int(os_[0]) != int(od[0]) else [])
        order, anchored = SE.schedule3(self.T, Q, seeds, key="wh")
        order = list(order)
        gate = bool(anchored) and self.typed_graph
        assert gate == bool(self.TD["gate"][j]) and ",".join(self.T.vocab[r_] for r_ in order) == self.ORDERS[j]
        x0 = np.zeros(N)
        x0[np.asarray(seeds, np.int64)] = 1.0 / float(len(seeds))
        aU = oU = None
        if gate:
            aU, oU, _ = W.walk(x0, order)
        for c in self.cells:
            t0 = time.perf_counter()
            P = K.parts[c]
            npart, hard = P.npart, P.hard
            hq = hard[oq]
            up, first = np.unique(hq, return_index=True)
            fe = np.full(npart, len(oq), np.int64)
            fe[up] = first
            pr = RT.inv(np.lexsort((np.arange(npart), fe)))
            contn = pr[hard] < int(K.B[c][j])
            cm = int(contn.sum())
            assert cm == int(self.CMASS[c][j])
            o1 = FO[contn[FO]]
            nmax = min(max(M_CURVE), cm)
            if gate:
                aP, oP, rdP = W.walk(x0, order, srcm=contn)
                _, uP, _ = W.walk(x0, [], srcm=contn)
                a0, o0, rd0, op0, _ = W.plan(x0, order, contn, hard, frank, S=0)
                assert (a0 == aP).all() and (o0 == oP).all() and (rd0 == rdP).all() and not op0, "S0 != the PUSH walk (%s row %d)" % (c, j)
                assert (positions(first_then(tcands(x0, aP, oP, uP, frank), o1, N), g, N) == self.POS[(c, "T1_PUSH")][sl]).all()
                self.n_s0 += 1
            for a in BUD_ARMS:
                bs = self.BC[(c, a)]
                if gate:
                    Sb, Rb = budget_of(a, npart)
                    aB, oB, rdB, opS, opR = W.plan(x0, order, contn, hard, frank, S=Sb, Rr=Rb)
                    if a == "T1_SINF":
                        assert (aB == aU).all() and (oB == oU).all(), "SINF != the unmasked typed walk (%s row %d)" % (c, j)
                        self.n_sinf += 1
                    ctB = tcands(x0, aB, oB, uP, frank)
                    o = first_then(ctB, o1, N)
                    rout = rdB & ~contn
                    wsh = np.unique(hard[np.flatnonzero(rout)])
                    bs["opened"][j] = len(opS) if Rb is None else len(opR)
                    bs["opened_shard_nodes"][j] = int(self.SHSIZE[c][np.asarray(opS, np.int64)].sum()) if Rb is None else len(opR)
                    bs["rows_read_out"][j] = int(rout.sum())
                    bs["rows_read_out_shards"][j] = len(wsh)
                else:
                    o = o1
                    wsh = np.zeros(0, np.int64)
                pa = positions(o, g, N)
                self.POS[(c, a)][sl] = pa
                if not gate:
                    assert (pa == self.POS[(c, "L1")][sl]).all()
                assert len(o) >= cm
                oo = o[:nmax]
                out = ~contn[oo]
                for mi, M in enumerate(M_CURVE):
                    n_ = min(M, cm)
                    ob = out[:n_]
                    xsh = np.unique(hard[oo[:n_][ob]])
                    self.FETCH[(c, a)][j, mi] = int(ob.sum())
                    self.XSH[(c, a)][j, mi] = len(xsh)
                    self.UXS[(c, a)][j, mi] = len(np.union1d(xsh, wsh))
            self.LAT["%s budget walks (%d arms)" % (c, len(BUD_ARMS))].append(time.perf_counter() - t0)

    def finish(self):
        diag, arrays = TYP.TypedSpec.finish(self)
        pop = self.pop
        gate = self.TD["gate"].astype(bool)
        res = diag["results (per cell, arm, B_N)"]
        curves = {}
        for c in self.cells:
            e = {}
            ends = {"T1_PUSH (S0 == R0)": "T1_PUSH"}
            for lab, a in list(ends.items()) + [(a, a) for a in BUD_ARMS] + [("T1_ROWS (TYPED v1; also reads untyped rows)", "T1_ROWS")]:
                ee = {"ALL": {str(M): res[c][a][str(M)]["ALL"] for M in M_CURVE}}
                for h in ("hop2", "hop3"):
                    if h in res[c][a][str(M_CURVE[0])]["strata (ALL rows, rows)"]:
                        ee["ALL " + h] = {str(M): res[c][a][str(M)]["strata (ALL rows, rows)"][h][0] for M in M_CURVE}
                if a in BUD_ARMS:
                    bc = self.BC[(c, a)]
                    ee["walk cost over gated rows (opened, opened shard nodes, typed rows read outside, their shards)"] = {
                        k: KB.AB.dist(v[gate]) for k, v in bc.items()}
                    ee["served fetch per B_N (mean nodes, mean extra shards, mean union with the walk's shards)"] = {
                        str(M): [round(float(self.FETCH[(c, a)][:, mi].mean()), 2), round(float(self.XSH[(c, a)][:, mi].mean()), 2),
                                 round(float(self.UXS[(c, a)][:, mi].mean()), 2)] for mi, M in enumerate(M_CURVE)}
                    for refa in ("T1_PUSH", "T1_ROWS"):
                        ee["paired_vs_%s (B_N 1000; gained = this arm serves ALL gold, %s does not)" % (refa, refa)] = D.paired(
                            self.served(c, refa, 1000)[0], self.served(c, a, 1000)[0])
                e[lab] = ee
            curves[c] = e
            for a in BUD_ARMS:
                for k, v in self.BC[(c, a)].items():
                    arrays["BC__%s__%s__%s" % (c, a, k)] = v
                arrays["FETCH__%s__%s" % (c, a)] = self.FETCH[(c, a)]
                arrays["XSH__%s__%s" % (c, a)] = self.XSH[(c, a)]
                arrays["UXS__%s__%s" % (c, a)] = self.UXS[(c, a)]
        diag["budget_curves (per cell: ALL by B_N, cost, paired at B_N 1000)"] = curves
        diag["budget_identities"] = {"S0 == PUSH walk (gated rows x cells)": self.n_s0, "SINF == unmasked walk (gated rows x cells)": self.n_sinf}
        return diag, arrays


def typed_v1_identity(S, arrays, full):
    """every TYPED v1 array (positions of every v1 arm, contacted flags / mass, B_P, costs, candidate ranks, gate diagnostics) ==
    the TYPED v1 record (its first rows / gold nodes for a smoke)."""
    assert D.sha_file(TYPED_V1_NPZ) == TYPED_V1_NPZ_SHA
    z = np.load(TYPED_V1_NPZ)
    pop = S.pop
    nq, ng = pop.nq, pop.ng_tot
    n_ok = 0
    for k in z.files:
        if k in ("rows", "gptr", "pos_FLAT"):
            continue
        assert k in arrays, "TYPED v1 array %s missing" % k
        a, b = np.asarray(arrays[k]), z[k]
        if a.shape != b.shape:
            n0 = a.shape[0]
            assert not full and a.shape[1:] == b.shape[1:] and n0 in (nq, ng), (k, a.shape, b.shape)
            b = b[:n0]
        # a string array's width follows its longest entry, so on a smoke prefix only the kind must agree
        assert (a.dtype == b.dtype) if (full or a.dtype.kind != "U") else (b.dtype.kind == "U"), "TYPED v1 array %s: dtype %s != %s" % (k, a.dtype, b.dtype)
        assert (a == b).all(), "TYPED v1 array %s differs" % k
        n_ok += 1
    assert (z["rows"][:nq] == pop.rows).all() and (z["gptr"][:nq + 1] == pop.gptr).all()
    return "all %d TYPED v1 arrays == %s (sha %s)%s" % (n_ok, D.rel(TYPED_V1_NPZ), D.sha_file(TYPED_V1_NPZ)[:16],
                                                           "" if full else " on the first %d rows / %d gold nodes" % (nq, ng))


def run(ds, tag):
    ROWS, SMOKE_OUT = R.argv_opts()
    here = os.path.abspath(__file__)
    shas = {"harness": D.sha_file(here)}
    for m in MODULES + tuple(sorted(TYP.LANE)):
        shas[m] = D.sha_file(os.path.join(D.HERE, m))
    tv1 = json.load(open(TYPED_V1_JSON, encoding="utf-8"))
    assert D.sha_file(TYPED_V1_JSON) == TYPED_V1_JSON_SHA, "the TYPED v1 record changed"
    assert shas["_l3d_typed.py"] == TYPED_MOD_SHA == tv1["code"]["harness"]["sha256"], "_l3d_typed.py differs from the harness that wrote TYPED v1"
    assert tv1["npz"]["sha256"] == D.sha_file(TYPED_V1_NPZ) == TYPED_V1_NPZ_SHA, "the TYPED v1 npz differs from its record"
    assert shas["_l1d_kbres.py"] == HOP.KBRES_MOD_SHA and shas["_l3d_hop.py"] == TYP.HOP_MOD_SHA
    for m, s in TYP.LANE.items():
        assert shas[m] == s and D.sha_file(os.path.join(TYP.SNAP, m)) == s, "the lane module %s differs from its code snapshot" % m
    host0 = D.host_state()
    t_all = time.time()
    outdir = SMOKE_OUT or OUT3
    os.makedirs(outdir, exist_ok=True)
    fp_out = os.path.join(outdir, "l3budget_%s__%s.json" % (ds, tag))
    fz = fp_out.replace(".json", ".npz")
    assert not os.path.exists(fp_out) and not os.path.exists(fz), "write-once: %s exists" % fp_out
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    cells = D.CELLS[ds]
    pop = D.Population(cd, ROWS)
    nq, gptr = pop.nq, pop.gptr
    Q, qrec = TYP.population_questions(cd, pop.rows)
    parts = {c: D.Part(cd, D.TAG_OF[c]) for c in cells}
    t_ = time.time()
    S = BudgetSpec(cd, pop, parts, Q)
    S.record["seconds_build"] = round(time.time() - t_, 1)
    S.record["question_records"] = qrec
    log("RUN l3budget %s %s: N %d, %d rows, %d gold nodes, cells %s, budget arms %d" % (ds, tag, N, nq, pop.ng_tot, S.cellsU, len(BUD_ARMS)))
    POS_FLAT = np.zeros(pop.ng_tot, np.int64)
    agd, ags = np.zeros(nq), np.zeros(nq)
    d200 = np.asarray(cd.dense_topk(D.ACT, pop.rows), np.int64)
    s200 = np.asarray(cd.splade_topk(D.ACT, pop.rows), np.int64)
    Qu = D.unit_queries(cd, pop.rows)
    t_ = time.time()
    for j0, j1, SD, SS, sec, _ in D.batches(cd, pop.rows, Qu, N):
        for i in range(j1 - j0):
            j = j0 + i
            of, od, os_, npos_j, fv, frank = D.flat_row(SD[i], SS[i])
            k_ = min(D.N_AGREE, npos_j)
            agd[j] = len(set(od[:D.N_AGREE].tolist()) & set(d200[j, :D.N_AGREE].tolist())) / float(D.N_AGREE)
            ags[j] = (len(set(os_[:k_].tolist()) & set(s200[j, :k_].tolist())) / float(k_)) if k_ else 1.0
            g = pop.golds[j]
            sl = slice(gptr[j], gptr[j + 1])
            POS_FLAT[sl] = frank[g]
            S.row(j, of, od, os_, npos_j, frank, g, sl, d200[j, 0])
        log("  %s rows %d / %d (%.0fs, RSS %.0f MB)" % (ds, j1, nq, time.time() - t_, D._rss_mb()))
    t_loop = round(time.time() - t_, 1)
    agree = {"dense_top100_overlap_mean": round(float(agd.mean()), 5), "splade_top100_overlap_mean": round(float(ags.mean()), 5)}
    assert agree["dense_top100_overlap_mean"] >= D.AGREE_MIN and agree["splade_top100_overlap_mean"] >= D.AGREE_MIN, agree
    assert S.n_identity == nq
    f1 = os.path.join(D.OUT, "loc_%s__%s.npz" % (ds, R.V1_TAG))
    z1 = np.load(f1)
    ng = int(z1["gptr"][nq])
    assert ng == pop.ng_tot and (z1["pos_FLAT"][:ng] == POS_FLAT).all(), "FLAT positions differ from the v1 record"
    idcheck = TYP.l1_identity(S, POS_FLAT, ROWS is None)
    diag, arrays = S.finish()
    tcheck = typed_v1_identity(S, arrays, ROWS is None)
    log("identity: " + tcheck)
    for c in S.cells:
        for a in ("L1", "T1_PUSH") + BUD_ARMS + ("T1_ROWS",):
            log("%-9s %-8s ALL %s" % (c, a, " ".join("%5d" % diag["results (per cell, arm, B_N)"][c][a][str(M)]["ALL"] for M in M_CURVE)))
    res = {"dataset": ds, "tag": tag, "stage": "L3 development (BUDGET)", "status": "DEVELOPMENT (descriptive; p-values descriptive; no budget selected)",
           "definitions": __doc__, "N": N, "n_rows": nq, "n_gold_nodes": pop.ng_tot, "population": pop.record, "structures": S.record,
           "l1_identity": idcheck, "typed_v1_identity": tcheck, "budget_identities": diag["budget_identities"],
           "served_list_agreement": agree, "diagnostics": diag,
           "code": {k: {"path": "scratchpad/" + (os.path.basename(here) if k == "harness" else k), "sha256": v} for k, v in shas.items()},
           "inputs": {"typed_v1_json": {"path": D.rel(TYPED_V1_JSON), "sha256": TYPED_V1_JSON_SHA},
                      "typed_v1_npz": {"path": D.rel(TYPED_V1_NPZ), "sha256": TYPED_V1_NPZ_SHA}},
           "pinned": D.PINNED, "pinned_repo": D.PINNED_REPO, "constants": D.CONSTANTS, "platform": D.platform_record(),
           "host_at_start": host0, "seconds_loop": t_loop, "_row_query_ids": pop.qids}
    arrays.update({"rows": pop.rows, "gptr": gptr, "pos_FLAT": POS_FLAT})
    res["seconds"] = round(time.time() - t_all, 1)
    res["process_peak_rss_mb"] = D.peak_rss_mb()
    for m in MODULES + tuple(sorted(TYP.LANE)):
        assert D.sha_file(os.path.join(D.HERE, m)) == shas[m], "code changed during the run"
    assert D.sha_file(here) == shas["harness"], "code changed during the run"
    np.savez_compressed(fz, **arrays)
    res["npz"] = {"path": os.path.basename(fz), "sha256": D.sha_file(fz)}
    D.G.S.wj(fp_out, res)
    log("done (%.0fs, peak RSS %s MB) -> %s sha256 %s" % (res["seconds"], res["process_peak_rss_mb"], fp_out, D.sha_file(fp_out)[:12]))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    assert mode == "RUN", "usage: RUN metaqa <tag> [--rows=K --out=<dir outside the repository>]"
    ds, tag = sys.argv[2], sys.argv[3]
    assert ds in ("metaqa",), "BUDGET is defined where TYPED v1 exists (MetaQA)"
    run(ds, tag)


if __name__ == "__main__":
    main()
