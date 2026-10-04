"""L1 DEVELOPMENT -- KBRES: where does the KB residual recall live?  (user 2026-09-29: "First, improve MetaQA on development data,
but focus specifically on the residual: hop-2 / hop-3 ALL-gold; 5-10 and 11+ answer queries; golds that are outside one-hop L1
but inside routed partitions; golds genuinely requiring another STRUCT hop. This tells us whether the missing recall should be
recovered by better partition-local completion, L2, or L3.")  A diagnostic: no rule is selected, nothing is tuned and L1 is not
changed ("That doesn't mean we should destroy the L1/L3 boundary by adding recursive traversal into L1").  Development rows
only: the L1_DEV population (c7f70806...), legacy-exposed; no reserved held-out row, no split-B row and no TEST row is read (the
query records of the population rows are the only query lines parsed).

Fixed: the carried-forward rule (results/L1_DEV/L1_ROUTING_CARRY_FORWARD__v1.json, sha pinned):
  L1a  IR_L1: the served order O = RRF(FLAT, LOC), K0 60 (sections 32-34), recomputed with the arithmetic of _l1d_scale.py
  L1b  ES shard ranking on the K map; B_P(q, K) = min(K, ceil(B100(q) (K/100)^0.75)), read per query from the kscale v1 npz
       (CNTA, the alpha 0.75 column; the npz sha pinned by the carry-forward record); the served set = the first B_N nodes of O
       inside the contacted shards
Cells: the two native-K structural maps of the dataset (the served partitions), and UNROUTED (every shard contacted).

Per gold node (cell-independent):
  seed    FLAT rank < 200 (one of the hits H_q)
  LOC     L(u) > 0: a one-hop E-neighbour of a hit, E = STRUCT_out u STRUCT_in u KNN u NER (per-family bits kept)
  EV      seed or LOC: the L1a evidence set V_q = H_q u {L > 0}
  dS      the undirected STRUCT distance from H_q (0 .. 3; 4 = at least 4 or unreachable); dS10 the same from the top-10 hits
  dT      the undirected STRUCT distance from the annotated topic entity (0 .. 4; 5 = at least 5 or unreachable).  Explanatory
          only: the annotation is not system-visible and enters no class
  probe   L2(u) = sum over f in {STRUCT_out, STRUCT_in} of sum_m A_f(m, u) g_f(m) LS(m), where LS is IR_L1's arithmetic restricted
          to the two STRUCT families (the next application of the same operator restricted to STRUCT, i.e. a traversal step that
          belongs to L3; here only a PROBE of how deep a second hop would have to go).  C2 = the nodes outside V_q with L2 > 0 (the
          second-hop candidates), ordered by (-L2, FLAT rank); rank2 = a gold's 0-based position in C2 (globally, and among the
          C2 nodes inside the contacted shards per cell)
Per cell and B_N, every gold node is exactly one class:
  SERVED_EV    served, in V_q
  SERVED_FILL  served, not in V_q: the FLAT fill inside the contacted shards (co-location did the completion)
  CROWD_EV     its shard contacted, in V_q, restricted rank >= B_N                        lever L2 (ordering / pool depth)
  CROWD_2L     its shard contacted, not in V_q, dS = 2 through a STRUCT neighbour of a hit whose shard is contacted
                                                                                          lever LOCAL (partition-local completion)
  CROWD_2X     its shard contacted, not in V_q, dS = 2, every such intermediate in an uncontacted shard     lever L3
  CROWD_3      its shard contacted, not in V_q, dS >= 3                                   lever L3
  REACH_EV     its shard not contacted, in V_q                                            lever ROUTE (the frozen fan-out)
  REACH_2      its shard not contacted, not in V_q, dS = 2                                lever L3
  REACH_3      its shard not contacted, not in V_q, dS >= 3                               lever L3
UNROUTED has no uncontacted shard: CROWD_2L there reads "dS = 2" (no shard condition) and CROWD_2X / REACH_* are empty.
(A gold outside V_q always has dS >= 2: a STRUCT neighbour of a hit has L > 0; asserted.)
Query level: a row not ALL-served needs the union of the levers of its missed golds; the lever sets are counted, and the
"recoverable with levers X" counts (every missed gold's lever in X) are ORACLE UPPER BOUNDS, not a mechanism.
Strata: per hop, per gold-count bucket (1, 2, 3, 4, 5-10, 11+), hop x bucket, per qtype.  Every p-value would be descriptive;
none is computed.

Identities asserted: FLAT == loc v1 (runner); the recomputed IR_L1 positions (LOC, FLAT+LOC, LOC order, LOC length) == the kscale
v1 npz on every gold node; per native cell and row the recomputed ES position of the last gold shard == LOES, and for every B_N
the per-gold served flags give ALL-served == served_at(LOES, HIES, CNTA alpha 0.75) (so the counts equal REPORT section 37.3);
UNROUTED served == FLAT+LOC position < B_N; every gold exactly one class; seeds have dS 0, LOC golds reached by a STRUCT family
dS 1, non-EV golds dS >= 2 and L2 > 0 when dS = 2; the population query records' hop and gold ids == the query index.

Usage: python scratchpad/_l1d_kbres.py RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L1_DEV/kbres_<dataset>__<tag>.{json,npz} (write-once)
"""
import io
import json
import os
import re
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
SFAMS = ("STRUCT_out", "STRUCT_in")
assert FAMS[:2] == SFAMS
NODE = "IR_L1"
ALPHA = "0.75"
TOP10 = 10
DS_MAX, DT_MAX = 3, 4                  # dS in 0..3, 4 = beyond; dT in 0..4, 5 = beyond
M_CURVE = D.M_CURVE
M_HEAD = 1000
CARRY = os.path.join(D.OUT, "L1_ROUTING_CARRY_FORWARD__v1.json")
CARRY_SHA = "aaae66faea2f8037f00f5be131d6a9bbe08ac80e815341090d503e444eeb417c"
UNR = "UNROUTED"
CLASSES = ("SERVED_EV", "SERVED_FILL", "CROWD_EV", "CROWD_2L", "CROWD_2X", "CROWD_3", "REACH_EV", "REACH_2", "REACH_3")
CI = {c: i for i, c in enumerate(CLASSES)}
LEVER = {"CROWD_EV": "L2", "CROWD_2L": "LOCAL", "CROWD_2X": "L3", "CROWD_3": "L3", "REACH_EV": "ROUTE", "REACH_2": "L3",
         "REACH_3": "L3"}
LEVERS = ("L2", "LOCAL", "ROUTE", "L3")
UPPER = (("L2",), ("LOCAL",), ("L2", "LOCAL"), ("L2", "LOCAL", "ROUTE"), ("L2", "LOCAL", "ROUTE", "L3"))
R2_SHOW = (100, 500, 1000, 5000)
_ID_RE = re.compile(r'"node_id":\s*"((?:[^"\\]|\\.)*)"')


def bfs(F, fams, src, N, maxd):
    """undirected distance over the union of the families' CSRs from the sources (0 .. maxd; maxd + 1 = beyond), and the node
    count of each level."""
    d = np.full(N, maxd + 1, np.int8)
    front = np.unique(np.asarray(src, np.int64))
    d[front] = 0
    sizes = [len(front)]
    for k in range(1, maxd + 1):
        nb = []
        for f in fams:
            xa, ad = F[f]["xadj"], F[f]["adj"]
            st = xa[front]
            _, pos = E.entries(st, xa[front + 1] - st)
            nb.append(ad[pos].astype(np.int64))
        nb = np.unique(np.concatenate(nb)) if nb else np.zeros(0, np.int64)
        nb = nb[d[nb] > k]
        d[nb] = k
        sizes.append(len(nb))
        front = nb
        if not len(front):
            sizes += [0] * (maxd - k)
            break
    return d, sizes


def struct_nbrs(F, v):
    """the undirected STRUCT neighbours of node v."""
    out = []
    for f in SFAMS:
        xa, ad = F[f]["xadj"], F[f]["adj"]
        out.append(ad[xa[v]:xa[v + 1]].astype(np.int64))
    return np.unique(np.concatenate(out))


def population_records(cd, rows):
    """{row: (topic node position or -1, qtype, hop, gold node ids)} parsed from the query files for the population rows only."""
    want = {}
    for r in rows:
        r = int(r)
        sp = next(s for s, (a, b) in cd.split_ranges.items() if a <= r < b)
        want.setdefault(sp, {})[r - cd.split_ranges[sp][0]] = r
    recs = {}
    for sp, lines in want.items():
        fp = os.path.join(cd.dir, "queries", "%s.jsonl" % sp)
        last = max(lines)
        with io.open(fp, encoding="utf-8") as f:
            for i, ln in enumerate(f):
                if i in lines:
                    d = json.loads(ln)
                    recs[lines[i]] = d
                if i >= last:
                    break
    assert len(recs) == len(rows)
    need = set(d.get("topic_entity_node_id") for d in recs.values() if d.get("topic_entity_node_id"))
    pos = {}
    with io.open(os.path.join(cd.dir, "nodes.jsonl"), encoding="utf-8") as f:
        for i, ln in enumerate(f):
            m = _ID_RE.search(ln)
            nid = m.group(1)
            if nid in need:
                pos[nid] = i
    out = {}
    for r, d in recs.items():
        t = d.get("topic_entity_node_id")
        out[r] = (pos[t] if t in pos else -1, d.get("qtype") or "", d.get("hop"), list(d.get("gold_node_ids") or []))
    return out, {"topic_ids_needed": len(need), "topic_ids_resolved": len(pos)}


class KBResSpec(object):
    def __init__(self, cd, pop, parts):
        self.ds = ds = cd.name
        N = self.N = int(cd.n_nodes)
        nq, ng = pop.nq, pop.ng_tot
        self.pop = pop
        # ---- the carried-forward spec and the kscale v1 record it pins
        assert D.sha_file(CARRY) == CARRY_SHA, "the carry-forward record changed"
        cf = json.load(open(CARRY, encoding="utf-8"))
        assert cf["spec"]["alpha"] == ALPHA and cf["spec"]["K_REF"] == 100
        pin = cf["pins"]["datasets"][ds]
        fk = os.path.join(D.REPO, pin["record"]["path"])
        assert D.sha_file(fk) == pin["record"]["sha256"], "kscale record changed"
        self.kz_path = os.path.join(D.REPO, pin["record"]["npz"])
        assert D.sha_file(self.kz_path) == pin["record"]["npz_sha256"], "kscale npz changed"
        krec = json.load(open(fk, encoding="utf-8"))
        assert krec["n_rows"] >= nq
        served = pin["served_cells"]
        assert served == krec["structures"]["served_cells"]
        self.cellmap = {}
        for pc, kc in served.items():
            cr = krec["structures"]["cells"][kc]
            assert cr["native"] and cr["K"] == parts[pc].npart
            assert D.sha_file(parts[pc].path) == cr["sha256"], "served partition %s differs from kscale cell %s" % (pc, kc)
            self.cellmap[pc] = kc
        self.cells = [self.cellmap[pc] for pc in sorted(self.cellmap, key=lambda p: self.cellmap[p])]
        self.parts = {self.cellmap[pc]: parts[pc] for pc in self.cellmap}
        z = np.load(self.kz_path)
        ai = [str(a) for a in z["alphas"]].index(ALPHA)
        self.B = {c: z["CNTA__" + c][:nq, ai].astype(np.int64) for c in self.cells}
        self.LOK = {c: z["LOES__" + c][:nq].astype(np.int64) for c in self.cells}
        self.HIK = {c: z["HIES__" + c][:nq].astype(np.int64) for c in self.cells}
        for c in self.cells:
            assert (z["SIZES__" + c] == self.parts[c].sizes).all()
        assert (z["rows"][:nq] == pop.rows).all()
        # ---- families (the IR_L1 edge set, keys as route v3 / scale v1)
        t0 = time.time()
        self.F, famrec = NH.build_families(cd, N)
        assert famrec == krec["structures"]["families"], "families differ from kscale v1"
        t_fam = time.time() - t0
        # ---- the population's query records (topic entity, qtype), population rows only
        t0 = time.time()
        precs, prrec = population_records(cd, pop.rows)
        self.TOPIC = np.array([precs[int(r)][0] for r in pop.rows], np.int64)
        self.QTYPE = [precs[int(r)][1] for r in pop.rows]
        for j, r in enumerate(pop.rows):
            h = precs[int(r)][2]
            assert int(h) == int(pop.hops[j]), "hop differs from the query index (row %d)" % r
            assert len(precs[int(r)][3]) == int(pop.ngold[j]), "gold count differs from the query index (row %d)" % r
        t_rec = time.time() - t0
        # ---- per-row / per-gold outputs
        self.names = [NODE]
        self.hard_ref = {NODE: None}
        self.v1_hard = {}
        self.lat = {NODE: []}
        self.DLAT = []
        self.G_LPOS = np.full(ng, -1, np.int64)
        self.G_FAM = np.zeros(ng, np.int8)
        self.G_DS = np.zeros(ng, np.int8)
        self.G_DS10 = np.zeros(ng, np.int8)
        self.G_DT = np.zeros(ng, np.int8)
        self.G_R2 = np.full(ng, -1, np.int64)
        self.G_FRK = np.zeros(ng, np.int64)
        self.CC = {c: {"cont": np.zeros(ng, bool), "rr": np.full(ng, -1, np.int64), "r2c": np.full(ng, -1, np.int64),
                       "mid_cont": np.zeros(ng, bool), "mid_same": np.zeros(ng, bool)} for c in self.cells}
        self.Q_EV = np.zeros(nq, np.int64)
        self.Q_LOCLEN = np.zeros(nq, np.int64)
        self.Q_C2 = np.zeros(nq, np.int64)
        self.Q_DSN = np.zeros((nq, DS_MAX + 2), np.int64)
        self.Q_TOPRANK = np.full(nq, -1, np.int64)
        self.Q_C2C = {c: np.zeros(nq, np.int64) for c in self.cells}
        self.Q_CMASS = {c: np.zeros(nq, np.int64) for c in self.cells}
        self.Q_EVC = {c: np.zeros(nq, np.int64) for c in self.cells}
        self.record = {
            "carry_forward": {"path": D.rel(CARRY), "sha256": CARRY_SHA, "alpha": ALPHA},
            "kscale_v1": {"path": D.rel(fk), "sha256": D.sha_file(fk), "npz": D.rel(self.kz_path), "npz_sha256": D.sha_file(self.kz_path),
                          "alpha_column": ai},
            "cells": {c: {"served_partition": pc, "tag": parts[pc].tag, "file": D.rel(parts[pc].path),
                          "sha256": D.sha_file(parts[pc].path), "K": parts[pc].npart} for pc, c in self.cellmap.items()},
            "reference": UNR + " (every shard contacted: the IR_L1 FLAT+LOC order itself)",
            "families": famrec, "seconds_families": round(t_fam, 1),
            "query_records": dict(prrec, seconds=round(t_rec, 1),
                                  note="the population rows' lines only are parsed (topic_entity_node_id, qtype, hop, gold ids)"),
            "classes": list(CLASSES), "levers": LEVER, "M_curve": list(M_CURVE), "headline_B_N": M_HEAD,
            "dS_max": DS_MAX, "dT_max": DT_MAX, "top10": TOP10}
        log("KBRES %s: cells %s (alpha %s from %s), topic ids %s" % (ds, self.cellmap, ALPHA, D.rel(self.kz_path), prrec))

    def row(self, j, of, od, os_, npos, fv, frank, g, sl):
        N, F = self.N, self.F
        top = of[:ACT]
        nh = len(top)
        # ---- L1a: the IR_L1 node score (== _l1d_scale.ScaleSpec.row)
        t0 = time.perf_counter()
        U, H, GW = [], [], []
        for f in FAMS:
            Fm = F[f]
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
        t0 = time.perf_counter()
        # ---- the served order O = IR_L1 FLAT+LOC, and the router's order O' (== ScaleSpec.row)
        f = 1.0 / (K0 + frank.astype(np.float64))
        f[Lord] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
        FO = np.lexsort((frank, -f))
        frk = np.empty(N, np.int64)
        frk[FO] = np.arange(N)
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
        # ---- evidence, family bits, distances
        ev = np.zeros(N, bool)
        ev[vq] = True
        fam = np.zeros(len(g), np.int8)
        for fi in range(len(FAMS)):
            fam |= (np.isin(g, U[fi]).astype(np.int8) << fi)
        lrank = np.full(N, -1, np.int64)
        lrank[Lord] = np.arange(len(Lord))
        lp = lrank[g]
        assert ((fam > 0) == (lp >= 0)).all() and ((L[g] > 0) == (lp >= 0)).all()
        dS, dsn = bfs(F, SFAMS, top, N, DS_MAX)
        dS10, _ = bfs(F, SFAMS, top[:TOP10], N, DS_MAX)
        t_ = int(self.TOPIC[j])
        dT = bfs(F, SFAMS, [t_], N, DT_MAX)[0] if t_ >= 0 else np.full(N, DT_MAX + 1, np.int8)
        gd = dS[g]
        seed = frank[g] < ACT
        assert (gd[seed] == 0).all() and (gd[~seed] >= 1).all()
        sreach = (fam & 3) > 0
        assert (gd[sreach & ~seed] == 1).all(), "a STRUCT-reached LOC gold with dS != 1 (row %d)" % j
        gev = ev[g]
        assert (gd[~gev] >= 2).all(), "a non-evidence gold with dS < 2 (row %d)" % j
        # ---- the second-hop probe (a traversal step, measured only)
        nS = len(U[0]) + len(U[1])
        LS = np.bincount(u[:nS], weights=x[:nS], minlength=N)
        ms = np.flatnonzero(LS)
        U2, W2 = [], []
        for fn in SFAMS:
            Fm = F[fn]
            st = Fm["xadj"][ms]
            h2, p2 = E.entries(st, Fm["xadj"][ms + 1] - st)
            U2.append(Fm["adj"][p2].astype(np.int64))
            W2.append((LS[ms] * Fm["g"][ms])[h2])
        L2 = np.bincount(np.concatenate(U2), weights=np.concatenate(W2), minlength=N)
        co = np.flatnonzero((L2 > 0) & ~ev)
        o2 = co[np.lexsort((frank[co], -L2[co]))]
        r2 = np.full(N, -1, np.int64)
        r2[o2] = np.arange(len(o2))
        gr2 = r2[g]
        assert (gr2[(~gev) & (gd == 2)] >= 0).all(), "a dS = 2 non-evidence gold without second-hop score (row %d)" % j
        # ---- intermediates of the dS = 2 non-evidence golds
        k2 = np.flatnonzero((~gev) & (gd == 2))
        mids = {int(k): (lambda nb: nb[dS[nb] == 1])(struct_nbrs(F, int(g[k]))) for k in k2}
        for k in k2:
            assert len(mids[int(k)]) >= 1
        # ---- per native cell: the ES order, the alpha 0.75 contacted set, restricted ranks
        for c in self.cells:
            P = self.parts[c]
            npart, hard = P.npart, P.hard
            hq = hard[oq]
            up, first = np.unique(hq, return_index=True)
            fe = np.full(npart, len(oq), np.int64)
            fe[up] = first
            ro = np.lexsort((np.arange(npart), fe))
            pr = RT.inv(ro)
            b = int(self.B[c][j])
            assert 1 <= b <= npart
            assert int(pr[hard[g]].max()) + 1 == self.LOK[c][j], "recomputed ES LO differs (%s row %d)" % (c, j)
            contn = pr[hard] < b
            cc = np.cumsum(contn[FO])
            gc = contn[g]
            rr = np.where(gc, cc[pg] - 1, -1)
            CC = self.CC[c]
            CC["cont"][sl] = gc
            CC["rr"][sl] = rr
            if len(o2):
                rc2 = np.cumsum(contn[o2]) - 1
                CC["r2c"][sl] = np.where((gr2 >= 0) & gc, rc2[np.maximum(gr2, 0)], -1)
            hg = hard[g]
            mc = np.zeros(len(g), bool)
            msm = np.zeros(len(g), bool)
            for k in k2:
                mm = mids[int(k)]
                mc[k] = bool(contn[mm].any())
                msm[k] = bool((hard[mm] == hg[k]).any())
            CC["mid_cont"][sl] = mc
            CC["mid_same"][sl] = msm
            self.Q_C2C[c][j] = int(contn[o2].sum())
            self.Q_CMASS[c][j] = int(contn.sum())
            self.Q_EVC[c][j] = int(contn[vq].sum())
            for mi, M in enumerate(M_CURVE):
                sv = gc & (rr < M)
                ref = AB.served_at(np.array([self.LOK[c][j]]), np.array([self.HIK[c][j, mi]]), b)[0]
                assert bool(sv.all()) == bool(ref), "ALL-served differs from kscale v1 (%s row %d B_N %d)" % (c, j, M)
        self.G_LPOS[sl] = lp
        self.G_FAM[sl] = fam
        self.G_DS[sl] = gd
        self.G_DS10[sl] = dS10[g]
        self.G_DT[sl] = dT[g]
        self.G_R2[sl] = gr2
        self.G_FRK[sl] = pg
        self.Q_EV[j] = len(vq)
        self.Q_LOCLEN[j] = len(Lord)
        self.Q_C2[j] = len(o2)
        self.Q_DSN[j] = dsn + [N - sum(dsn)]
        self.Q_TOPRANK[j] = int(frank[t_]) if t_ >= 0 else -1
        self.DLAT.append(time.perf_counter() - t0)
        return {NODE: Lord}

    # ---- classification and tables
    def classes(self, c, M, ev, gd):
        """per gold class code (int8) at cell c (or UNR) and B_N M."""
        if c == UNR:
            sv = self.G_FRK < M
            cont = np.ones(len(sv), bool)
            mid = np.ones(len(sv), bool)
        else:
            CC = self.CC[c]
            cont, rr = CC["cont"], CC["rr"]
            sv = cont & (rr < M)
            mid = CC["mid_cont"]
        cl = np.full(len(sv), -1, np.int8)
        cl[sv & ev] = CI["SERVED_EV"]
        cl[sv & ~ev] = CI["SERVED_FILL"]
        cr = cont & ~sv
        cl[cr & ev] = CI["CROWD_EV"]
        cl[cr & ~ev & (gd == 2) & mid] = CI["CROWD_2L"]
        cl[cr & ~ev & (gd == 2) & ~mid] = CI["CROWD_2X"]
        cl[cr & ~ev & (gd >= 3)] = CI["CROWD_3"]
        rc = ~cont
        cl[rc & ev] = CI["REACH_EV"]
        cl[rc & ~ev & (gd == 2)] = CI["REACH_2"]
        cl[rc & ~ev & (gd >= 3)] = CI["REACH_3"]
        assert (cl >= 0).all(), "a gold without a class"
        return cl, sv

    def finish(self, ctx):
        POS, gptr, ngold, pop, ST = ctx["POS"], ctx["gptr"], ctx["ngold"], ctx["pop"], ctx["ST"]
        nq, ng = pop.nq, pop.ng_tot
        rog = pop.row_of_gold
        # (0) the IR_L1 positions == kscale v1 (== node1h v1)
        z = np.load(self.kz_path)
        for key, arm in (("pos_LOC__", "LOC"), ("pos_FLATLOC__", "FLAT+LOC"), ("lpos__", "lpos")):
            assert (z[key + NODE][:ng] == POS[NODE][arm]).all(), "%s differs from kscale v1" % key
        assert (z["loc_len__" + NODE][:nq] == ctx["LLEN"][NODE]).all()
        assert (self.G_FRK == POS[NODE]["FLAT+LOC"]).all() and (self.G_LPOS == POS[NODE]["lpos"]).all()
        assert (z["pos_FLAT"][:ng] == ctx["POS_FLAT"]).all()
        seedg = ctx["POS_FLAT"] < ACT
        ev = seedg | (self.G_LPOS >= 0)
        gd = self.G_DS.astype(np.int64)
        # strata masks over queries
        hops = pop.hops
        QS = {"all": np.ones(nq, bool)}
        for h in sorted(set(int(x) for x in hops)):
            QS["hop%d" % h] = hops == h
        for nm, lo, hi in D.NG_BUCKETS:
            QS["ng_" + nm] = (ngold >= lo) & (ngold <= hi)
        for h in sorted(set(int(x) for x in hops)):
            for nm, lo, hi in D.NG_BUCKETS:
                m = (hops == h) & (ngold >= lo) & (ngold <= hi)
                if m.any():
                    QS["hop%d|ng_%s" % (h, nm)] = m
        qts = sorted(set(self.QTYPE))
        QT = {qt: np.array([q == qt for q in self.QTYPE]) for qt in qts}
        cellsU = [UNR] + list(self.cells)
        out = {"identity": ("FLAT == loc v1 (runner); IR_L1 LOC / FLAT+LOC / LOC-order positions and LOC lengths == %s on all %d gold "
                            "nodes; per native cell, row and B_N: the recomputed ES LO == LOES and ALL-served from the per-gold flags == "
                            "served_at(LOES, HIES, CNTA alpha %s); every gold exactly one class; distance identities (seed dS 0, STRUCT-"
                            "reached LOC dS 1, non-evidence dS >= 2, dS 2 => L2 > 0); query records' hop and gold count == the index"
                            % (D.rel(self.kz_path), ng, ALPHA))}
        CL, SV, QALL, QLEV = {}, {}, {}, {}
        for c in cellsU:
            for M in M_CURVE:
                cl, sv = self.classes(c, M, ev, gd)
                CL[(c, M)] = cl
                SV[(c, M)] = sv
                cnt = np.add.reduceat(sv.astype(np.int64), gptr[:-1])
                qa = cnt == ngold
                QALL[(c, M)] = qa
                # lever set per query (bitmask over LEVERS)
                lb = np.zeros(ng, np.int64)
                for cn, lv in LEVER.items():
                    lb[cl == CI[cn]] = 1 << LEVERS.index(lv)
                QLEV[(c, M)] = np.bitwise_or.reduceat(lb, gptr[:-1])
                assert ((QLEV[(c, M)] == 0) == qa).all()
            # the ALL counts must equal section 37.3 at the headline B_N (asserted through served_at already; cross-check sums)
        UNRALL = {M: D.per_query(POS[NODE]["FLAT+LOC"], M, gptr, ngold)[0] for M in M_CURVE}
        for M in M_CURVE:
            assert (QALL[(UNR, M)] == UNRALL[M]).all()

        def lever_name(bits):
            return "+".join(LEVERS[i] for i in range(len(LEVERS)) if bits >> i & 1) or "NONE"

        def gold_table(c, M, gmask):
            cl = CL[(c, M)][gmask]
            return {cn: int((cl == CI[cn]).sum()) for cn in CLASSES}

        def query_table(c, M, qmask):
            qa, lv = QALL[(c, M)][qmask], QLEV[(c, M)][qmask]
            miss = ~qa
            combos = {}
            for bits in np.unique(lv[miss]):
                combos[lever_name(int(bits))] = int((lv[miss] == bits).sum())
            up = {}
            for X in UPPER:
                xb = sum(1 << LEVERS.index(l) for l in X)
                up["+".join(X)] = int((qa | ((lv & ~xb) == 0)).sum())
            return {"n": int(qmask.sum()), "ALL": int(qa.sum()), "not_ALL": int(miss.sum()),
                    "lever_sets_of_the_not_ALL_rows": dict(sorted(combos.items(), key=lambda kv: -kv[1])),
                    "ALL_if_levers_recover_every_missed_gold (oracle upper bound)": up}

        tabs = {}
        for c in cellsU:
            tabs[c] = {}
            for M in M_CURVE:
                e = {}
                for sn, qm in QS.items():
                    gm = qm[rog]
                    e[sn] = {"queries": query_table(c, M, qm), "gold_nodes": dict(gold_table(c, M, gm), n=int(gm.sum()))}
                tabs[c][str(M)] = e
        out["tables (per cell, B_N, stratum: query level and gold level)"] = tabs
        # per qtype at the headline B_N
        qtab = {}
        for c in cellsU:
            qtab[c] = {}
            for qt, qm in QT.items():
                gm = qm[rog]
                qtab[c][qt] = {"hop": int(hops[qm][0]), "queries": query_table(c, M_HEAD, qm),
                               "gold_nodes": dict(gold_table(c, M_HEAD, gm), n=int(gm.sum()))}
        out["per_qtype_at_B_N_%d" % M_HEAD] = qtab
        # the user's two gold populations at the headline B_N
        pops = {}
        for c in self.cells:
            CC = self.CC[c]
            cl = CL[(c, M_HEAD)]
            outside = (~ev) & CC["cont"]
            e = {"outside_one_hop_L1_inside_routed_shards (not in V_q, shard contacted)": {
                    "n": int(outside.sum()), "served_by_FLAT_fill": int((outside & SV[(c, M_HEAD)]).sum()),
                    "by_class": {cn: int((outside & (cl == CI[cn])).sum()) for cn in CLASSES},
                    "by_dS": {str(d): int((outside & (gd == d)).sum()) for d in range(DS_MAX + 2)},
                    "by_hop": {"hop%d" % h: int((outside & (hops[rog] == h)).sum()) for h in sorted(set(int(x) for x in hops))}}}
            need2 = (~ev) & (gd == 2)
            e["needs_another_STRUCT_hop (not in V_q, dS = 2)"] = {
                "n": int(need2.sum()), "shard_contacted": int((need2 & CC["cont"]).sum()),
                "served_by_FLAT_fill": int((need2 & SV[(c, M_HEAD)]).sum()),
                "intermediate_in_a_contacted_shard": int((need2 & CC["mid_cont"]).sum()),
                "intermediate_in_the_gold's_own_shard": int((need2 & CC["mid_same"]).sum()),
                "by_hop": {"hop%d" % h: int((need2 & (hops[rog] == h)).sum()) for h in sorted(set(int(x) for x in hops))}}
            pops[c] = e
        out["user_gold_populations_at_B_N_%d" % M_HEAD] = pops
        # distance profiles of the golds (cell-independent) and of the missed golds per cell at the headline B_N
        def hist(v, m, top_):
            return {str(d): int(((v == d) & m).sum()) for d in range(top_ + 2)}
        dprof = {"all_golds": {"dS": hist(gd, np.ones(ng, bool), DS_MAX), "dS10": hist(self.G_DS10, np.ones(ng, bool), DS_MAX),
                               "dT": hist(self.G_DT, np.ones(ng, bool), DT_MAX)}}
        for h in sorted(set(int(x) for x in hops)):
            gm = hops[rog] == h
            dprof["hop%d" % h] = {"n": int(gm.sum()), "seed": int((gm & seedg).sum()), "LOC_not_seed": int((gm & ev & ~seedg).sum()),
                                  "dS": hist(gd, gm, DS_MAX), "dS10": hist(self.G_DS10, gm, DS_MAX), "dT": hist(self.G_DT, gm, DT_MAX)}
            for c in cellsU:
                mm = gm & ~SV[(c, M_HEAD)]
                dprof["hop%d" % h]["missed@%d__%s" % (M_HEAD, c)] = {"n": int(mm.sum()), "dS": hist(gd, mm, DS_MAX),
                                                                    "dT": hist(self.G_DT, mm, DT_MAX)}
        out["distance_profiles"] = dprof
        # the second-hop probe: rank of the dS = 2 non-evidence golds among the second-hop candidates
        need2 = (~ev) & (gd == 2)
        probe = {"candidates_C2_per_query": AB.dist(self.Q_C2), "global": {}}
        for h in ["all"] + ["hop%d" % h for h in sorted(set(int(x) for x in hops))]:
            gm = need2 & (np.ones(ng, bool) if h == "all" else hops[rog] == int(h[3:]))
            r = self.G_R2[gm]
            probe["global"][h] = {"n": int(gm.sum()), **{"rank2 < %d" % k: int((r < k).sum()) for k in R2_SHOW}}
        for c in self.cells:
            CC = self.CC[c]
            probe[c] = {"C2_inside_contacted_shards_per_query": AB.dist(self.Q_C2C[c]),
                        "contacted_mass_per_query": AB.dist(self.Q_CMASS[c]), "evidence_inside_contacted_per_query": AB.dist(self.Q_EVC[c])}
            for h in ["all"] + ["hop%d" % h for h in sorted(set(int(x) for x in hops))]:
                gm = need2 & CC["cont"] & ~SV[(c, M_HEAD)] & (np.ones(ng, bool) if h == "all" else hops[rog] == int(h[3:]))
                r = CC["r2c"][gm]
                probe[c]["crowded_dS2@%d__%s" % (M_HEAD, h)] = {"n": int(gm.sum()), **{"rank2_in_contacted < %d" % k: int(((r >= 0) & (r < k)).sum())
                                                                                        for k in R2_SHOW}}
        out["second_hop_probe (a traversal step; measured, not a rule)"] = probe
        # how deep the crowded evidence golds sit (the pool an L2 would need), per cell and hop at the headline B_N
        depth = {}
        for c in cellsU:
            depth[c] = {}
            cl = CL[(c, M_HEAD)]
            rk = self.G_FRK if c == UNR else self.CC[c]["rr"]
            for h in ["all"] + ["hop%d" % h for h in sorted(set(int(x) for x in hops))]:
                m = (cl == CI["CROWD_EV"]) & (np.ones(ng, bool) if h == "all" else hops[rog] == int(h[3:]))
                r = rk[m]
                depth[c][h] = dict({"n": int(m.sum())}, **{"restricted_rank < %d" % k: int((r < k).sum()) for k in (2000, 5000, 10000)},
                                   dist=AB.dist(r) if m.any() else None)
        out["crowded_evidence_depth_at_B_N_%d (restricted rank among the contacted nodes; UNROUTED: the IR_L1 position)" % M_HEAD] = depth
        out["query_side"] = {"evidence_set_V_q": AB.dist(self.Q_EV), "LOC_length": AB.dist(self.Q_LOCLEN),
                             "dS_level_sizes_mean (0..3, beyond)": [round(float(v), 1) for v in self.Q_DSN.mean(0)],
                             "topic_entity_FLAT_rank": {"n_resolved": int((self.Q_TOPRANK >= 0).sum()),
                                                        "rank < 1": int(((self.Q_TOPRANK >= 0) & (self.Q_TOPRANK < 1)).sum()),
                                                        "rank < 10": int(((self.Q_TOPRANK >= 0) & (self.Q_TOPRANK < 10)).sum()),
                                                        "rank < 200 (a hit)": int(((self.Q_TOPRANK >= 0) & (self.Q_TOPRANK < ACT)).sum())}}
        out["latency_ms_per_row (diagnostic: orders, distances, probe, cells)"] = D.ms_stats(self.DLAT)
        arrays = {"G_LPOS": self.G_LPOS, "G_FAM": self.G_FAM, "G_DS": self.G_DS, "G_DS10": self.G_DS10, "G_DT": self.G_DT,
                  "G_R2": self.G_R2, "G_FRK": self.G_FRK, "Q_EV": self.Q_EV, "Q_LOCLEN": self.Q_LOCLEN, "Q_C2": self.Q_C2,
                  "Q_DSN": self.Q_DSN, "Q_TOPIC": self.TOPIC, "Q_TOPRANK": self.Q_TOPRANK, "HOPS": hops,
                  "QTYPE": np.array(self.QTYPE), "cells": np.array(self.cells), "classes": np.array(CLASSES),
                  "levers": np.array(LEVERS), "m_curve": np.asarray(M_CURVE, np.int64)}
        for c in self.cells:
            for k, v in self.CC[c].items():
                arrays["%s__%s" % (k, c)] = v
            arrays["Q_C2C__" + c] = self.Q_C2C[c]
            arrays["Q_CMASS__" + c] = self.Q_CMASS[c]
            arrays["B__" + c] = self.B[c]
        for c in cellsU:
            for M in M_CURVE:
                arrays["CL__%s__%d" % (c, M)] = CL[(c, M)]
        return out, arrays


DOC = __doc__


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    assert mode == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]"
    ds, tag = sys.argv[2], sys.argv[3]
    assert ds in ("metaqa",), "KBRES is defined for the KB development dataset with a served structural pair"
    R.run(ds, tag, "kbres", mode, __file__, lambda cd, pop, parts: KBResSpec(cd, pop, parts), DOC)


if __name__ == "__main__":
    main()
