"""L3 DEVELOPMENT -- TYPED: the relation-typed scheduled frontier walk of the L1_P90 lane as an L3 completion arm, on top of the
carried-forward L1, at equal exposure.
(User 2026-09-29: "we need to finish the full KB path: L1 localization -> L2 reranking -> L3 multi-hop completion"; "First,
improve MetaQA on development data".  The L3 HOP v1 record (results/L3_DEV/l3hop_metaqa__v1.json, df08ca0b...) found that the
untyped second STRUCT hop recovers hop 2 but that untyped traversal hurts hop 3.  The L1_P90_EXPLOIT lane (2026-09-14, DEV_B
confirmed, gate G2) had a relation-typed walk that took MetaQA hop 3 to 0.944 at P50 blocks; that walk is traversal, so under the
2026-09-26 boundary it belongs to L3 -- this harness measures it there, at node level, behind the carried-forward routing.)
L1 is not changed: everything below runs AFTER the carried-forward L1 and reads its outputs (the served order, the contacted
shards, V_q and the STRUCT part of the IR_L1 score) plus the inputs named under "read beyond L1".  Development rows only: the
L1_DEV population (c7f70806...), legacy-exposed; no reserved held-out row, no split-B row and no TEST row is read.  No encoder
training, no LLM, no learned weight, no tuned constant (K0 60, EPS 1e-6 and H 3 are the frozen / lane constants).  STATUS:
DEVELOPMENT -- descriptive numbers, every p-value descriptive, no rule selected here.

Fixed input: the carried-forward L1 exactly as in _l3d_hop (IR_L1 served order O, ES routing on the native K 432 maps PHG / MTK,
B_P = kscale v1 CNTA alpha 0.75, contacted = the nodes of the first B_P shards, V_q = the 200 hits and every node with L > 0).
The typed channel (the lane's operator; the lane functions are imported, sha-pinned to results/L1_P90_EXPLOIT/code_snapshot):
  T        _l1x90_typed.Typed over the served structural family (relation-typed undirected multigraph, deduped, loop-free); the
           graph is typed iff it carries >= 2 relation labels (_l1x90_universal.is_typed_graph)
  seeds    _l1x90_seeds.lexical_seeds(question, name index, node names, T): nodes whose full name occurs verbatim in the question
           (longest spans, case-exact preferred, cap 5); none -> the fallback of the lane: dense top-1 + SPLADE top-1 (here the
           FLAT row's own dense / SPLADE lists)
  schedule _l1x90_seeds.schedule3(T, question, seeds, key="wh") -> (relation order, anchored)
  gate     G2 (the lane's confirmed gate): typed graph AND anchored.  A row outside the gate keeps the L1 order: every typed arm
           serves exactly the L1 list on it (asserted)
  walk     _l1x90_relwalk.relwalk's hard scheduled frontier walk, H = 3: step k may use the relations order[:k]; the mass pushed
           along the schedule's last relation is the answer channel ans, the rest oth; nodes already reached are zeroed; unit mass
           per layer.  The untyped walk (empty schedule) gives unt.  Node level: the lane's block indicator is the identity.
           The walk is re-implemented here with shard masks; unmasked it equals the pinned relwalk bit for bit at float32
           (seed / ans / oth and the untyped oth), asserted on every row
  score    M(u) = seed(u) + ans(u) + EPS (oth(u) + unt(u)), EPS = 1e-6 (the lane's constant, float64 here)
  CT       the typed candidates: seed + ans > 0, ordered by (-M, FLAT rank)
Masks of the walk in a routed cell:
  ROWS     unmasked: any adjacency row may be read (rows of nodes in uncontacted shards are counted, with their distinct shards);
           targets anywhere (a served candidate outside the contacted shards is FETCHED, counted)
  PUSH     a step expands only the mass on contacted nodes (their rows are read in full); arrivals anywhere are kept as candidates
           (fetched when served) but are not expanded further; seeds outside the contacted shards stay candidates, unexpanded
  LOCAL    seeds, sources and targets inside the contacted shards: seed mass outside is dropped and arrivals outside are dropped
           before the per-layer normalisation
Arms per routed cell (PHG_k432, MTK_k432); every arm serves exactly n(q, B_N) = min(B_N, contacted mass(q)) nodes (equal exposure):
  L1          the carried-forward served list (== section 37.3 / 38 / HOP v1)
  H2_PUSH     the HOP v1 untyped second hop, sources contacted, targets anywhere (reference; == the HOP v1 record)
  T1_<mask>   CT first (by -M, FLAT rank), then the L1 order of the remaining nodes
  TR_<mask>   RRF with the frozen K0: f_T(u) = f(u) + [u in CT] / (K0 + CT rank(u)), ties -> FLAT rank; eligible = contacted u CT
              (PUSH, ROWS) or contacted (LOCAL)
  TR_H2_PUSH  the RRF of the L1 score, the HOP v1 second-hop list C2 (PUSH) and CT (PUSH): f(u) + [C2] / (K0 + C2 rank) +
              [CT] / (K0 + CT rank); eligible = contacted u C2 u CT
UNROUTED (every shard contacted; B_N nodes served): L1, H2 (reference), T1, TR, TR_H2.
Read beyond L1: the question text of the population rows (their lines only, parsed like _l1d_kbres.population_records; the lane's
field precedence question_plain -> question -> text; a TEST row is refused), the node names (nodes.jsonl title ->
text, through Typed.names) and the structural relation labels.  The topic-entity annotation is read only for a diagnostic (topic
among the seeds), never by the system; the hop label only stratifies the report.
Reported per cell, arm and B_N (100 .. 5000): ALL / ANY / FRAC and gold nodes served, per hop, answer-count bucket, hop x bucket and
gate; McNemar gained / lost vs the cell's L1 and vs its H2 reference (descriptive); fetched nodes and extra shards per query;
rows read outside the contacted shards (ROWS); the gate / seed / schedule diagnostics; the gold ranks in CT; latency per stage.

Identities asserted: FLAT == loc v1; the dense / SPLADE top-100 agreement gate; per row the unmasked typed and untyped walks == the
pinned relwalk; on every gold node: UNROUTED L1 position == kbres v1 G_FRK, routed contacted flag and L1 rank == kbres v1 cont__ /
rr__, contacted mass == Q_CMASS__, and the L1 / H2 / H2_PUSH positions == the HOP v1 record's POS arrays; on the full population the
L1 ALL counts == the kbres v1 tables and the H2 / H2_PUSH ALL counts == the HOP v1 tables; LOCAL arms serve only contacted nodes and
exactly the contacted mass; every arm's list covers the contacted mass; outside the gate every typed arm == L1 (TR_H2* == H2*).

Usage: python scratchpad/_l3d_typed.py RUN metaqa <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L3_DEV/l3typed_<dataset>__<tag>.{json,npz} (write-once)
"""
import io
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import _l1d_lib as D
import _l1d_arms as R
import _l1d_route as RT
import _l1d_kbres as KB
import _l3d_hop as HOP
import _l1x90_typed as TY
import _l1x90_seeds as SE
import _l1x90_relwalk as RW
import _l1x90_universal as UNI

log = D.log
K0, ACT = D.K0, D.ACT
FAMS, SFAMS = KB.FAMS, KB.SFAMS
M_CURVE = D.M_CURVE
UNR = KB.UNR
OUT3 = HOP.OUT3
EPS, HW = UNI.EPS, UNI.H
prop, clist, ranked, rrf_add, positions, c2rank = HOP.prop, HOP.clist, HOP.ranked, HOP.rrf_add, HOP.positions, HOP.c2rank
LANE = {"_l1x90_core.py": "6eba63051165588c53900d98ec5644782ad107206f05a32ba6500d16cda76068",
        "_l1x90_typed.py": "3eb529e2e0f74bd05d63f441a5283daf57a4afa78101d3a8c43cb9b9773b913f",
        "_l1x90_seeds.py": "58766a2cebb025a21dbecee8ba06408916f8e04ac8b1f86909b0ae78d3f76f78",
        "_l1x90_relwalk.py": "41c3592f4e28702d9d5e91504f9623016426ced46dc7a7c08ccf50405ae62876",
        "_l1x90_universal.py": "ac00db803d732b6fce10f3b8205a68aeb9a659411cb828d316ea64504185ef00"}
SNAP = os.path.join(D.REPO, "results", "L1_P90_EXPLOIT", "code_snapshot")
HOP_MOD_SHA = "70777dd9770918cf119b8b8bb9c9bb779e4ba0ec0a7d0ad2e773a72f3dcf6e2b"
HOP_JSON = os.path.join(OUT3, "l3hop_metaqa__v1.json")
HOP_JSON_SHA = "df08ca0b787327839556a6adae74a5fed877c18e0bcd3ccc15023c14466b2b63"
HOP_NPZ = os.path.join(OUT3, "l3hop_metaqa__v1.npz")
HOP_NPZ_SHA = "1f851d0b5d99a8792689d0817f5b9689e8b1e63244dddc37e37a1ef3eed5dbaa"
ROUTED_ARMS = ("L1", "H2_PUSH", "T1_LOCAL", "TR_LOCAL", "T1_PUSH", "TR_PUSH", "T1_ROWS", "TR_ROWS", "TR_H2_PUSH")
FETCH_ARMS = ("H2_PUSH", "T1_PUSH", "TR_PUSH", "T1_ROWS", "TR_ROWS", "TR_H2_PUSH")
UNR_ARMS = ("L1", "H2", "T1", "TR", "TR_H2")
TYPED_ARMS = ("T1", "TR", "TR_H2", "T1_LOCAL", "TR_LOCAL", "T1_PUSH", "TR_PUSH", "T1_ROWS", "TR_ROWS", "TR_H2_PUSH")
QT_SHOW = (1000, 5000)
MODULES = HOP.MODULES + ("_l3d_hop.py",)
EMPTY = np.zeros(0, np.int64)


class NodeBlocks(object):
    """the lane's cache interface reduced to what Typed and relwalk read (cd, N, npart, blockmat), at node level: the block
    indicator is the identity, so a block mass is the node mass."""

    def __init__(self, cd):
        self.cd = cd
        self.N = int(cd.n_nodes)
        self.npart = self.N
        self._bm = sp.identity(self.N, dtype=np.float32, format="csr")

    def blockmat(self):
        return self._bm


class Walker(object):
    """relwalk's hard scheduled frontier walk (mix=False) with a source mask (only the mass on these nodes is expanded) and a
    target mask (arrivals elsewhere are dropped before the per-layer normalisation).  The arithmetic is that of
    _l1x90_relwalk.relwalk.walk, so the unmasked walk is bit-identical to it (asserted per row against the pinned function)."""

    def __init__(self, T, N, H):
        self.T, self.N, self.H = T, N, H
        self.cache = {}

    def sel_of(self, allowed):
        key = tuple(sorted(allowed))
        if key not in self.cache:
            T = self.T
            sel = np.isin(T.r, np.fromiter(allowed, np.int64)) if allowed else np.zeros(len(T.r), bool)
            cnt = np.bincount(T.u[sel], minlength=self.N).astype(np.float64)
            self.cache[key] = (sel, cnt, {})
        return self.cache[key]

    def walk(self, x0, order, srcm=None, tgtm=None):
        """(ans, oth, rows): rows = the nodes whose adjacency rows the walk reads (mass on an expanded source)."""
        T, N = self.T, self.N
        u, v, r, deg = T.u, T.v, T.r, T.deg
        x = x0.astype(np.float64)
        seen = x > 0
        ans = np.zeros(N)
        oth = np.zeros(N)
        rows = np.zeros(N, bool)
        last = order[-1] if order else None
        for k in range(1, self.H + 1):
            allowed = set(order[:k]) if order else set()
            sel, cnt, lc = self.sel_of(allowed)
            has = cnt > 0
            xs = x if srcm is None else np.where(srcm, x, 0.0)
            rows |= xs > 0
            if allowed:
                src = xs / np.maximum(cnt, 1.0)
                src[~has] = 0.0
                y_typed = np.bincount(v[sel], weights=src[u[sel]], minlength=N).astype(np.float64)
                if last not in lc:
                    lc[last] = sel & (r == last)
                sl = lc[last]
                y_ans = np.bincount(v[sl], weights=src[u[sl]], minlength=N).astype(np.float64)
                y_oth = y_typed - y_ans
            else:
                y_ans = np.zeros(N)
                y_oth = np.zeros(N)
            if not allowed:
                src_u = xs / np.maximum(deg, 1.0)
                y_oth = y_oth + np.bincount(v, weights=src_u[u], minlength=N).astype(np.float64)
            y_ans[seen] = 0.0
            y_oth[seen] = 0.0
            if tgtm is not None:
                y_ans[~tgtm] = 0.0
                y_oth[~tgtm] = 0.0
            s = y_ans.sum() + y_oth.sum()
            if s > 0:
                y_ans /= s
                y_oth /= s
            x = y_ans + y_oth
            seen |= x > 0
            ans += y_ans
            oth += y_oth
        return ans, oth, rows


def tcands(x0, ans, oth, unt, frank):
    """the typed candidate list: seed + ans > 0, ordered by (-M, FLAT rank), M = seed + ans + EPS (oth + unt)."""
    core = x0 + ans
    M = core + EPS * (oth + unt)
    c = np.flatnonzero(core > 0)
    return c[np.lexsort((frank[c], -M[c]))]


def first_then(ct, base, N):
    """ct first, then the nodes of base not in ct, in base order."""
    if not len(ct):
        return base
    inct = np.zeros(N, bool)
    inct[ct] = True
    return np.concatenate([ct, base[~inct[base]]])


def member(ct, N):
    m = np.zeros(N, bool)
    m[ct] = True
    return m


def population_questions(cd, rows):
    """the question text of the population rows only (the lane's precedence question_plain -> question -> text), parsed from the
    split files line by line like _l1d_kbres.population_records; each line's query_id is checked; a TEST row is refused."""
    want = {}
    for r in rows:
        r = int(r)
        sp_ = next(s for s, (a, b) in cd.split_ranges.items() if a <= r < b)
        assert sp_ != "test", "refused: row %d is in the TEST split" % r
        want.setdefault(sp_, {})[r - cd.split_ranges[sp_][0]] = r
    out = {}
    for sp_, lines in want.items():
        fp = os.path.join(cd.dir, "queries", "%s.jsonl" % sp_)
        last = max(lines)
        with io.open(fp, encoding="utf-8") as f:
            for i, ln in enumerate(f):
                if i in lines:
                    d = json.loads(ln)
                    r = lines[i]
                    assert d["query_id"] == cd.query_ids[r], "query id mismatch at row %d" % r
                    out[r] = d.get("question_plain") or d.get("question") or d.get("text") or ""
                if i >= last:
                    break
    assert len(out) == len(rows) and all(out[int(r)] for r in rows)
    return [out[int(r)] for r in rows], {"rows_per_split": {s: len(v) for s, v in want.items()}, "refused_split": "test",
                                         "fields": "question_plain -> question -> text (the lane's questions_of precedence)"}


class TypedSpec(object):
    def __init__(self, cd, pop, parts, Q):
        self.K = KB.KBResSpec(cd, pop, parts)
        K = self.K
        self.N, self.F, self.pop, self.Q = K.N, K.F, pop, Q
        N = self.N
        assert N > max(M_CURVE)
        nq, ng = pop.nq, pop.ng_tot
        # ---- the lane's typed graph, names and name index (node level)
        t0 = time.time()
        self.C = NodeBlocks(cd)
        self.T = TY.Typed(self.C)
        self.typed_graph = UNI.is_typed_graph(self.T)
        assert self.typed_graph, "TYPED is defined on a typed structural graph"
        self.names = self.T.names()
        assert len(self.names) == N
        self.idx = SE.build_name_index(self.names)
        self.W = Walker(self.T, N, HW)
        t_typed = time.time() - t0
        self.cells = list(K.cells)
        self.cellsU = [UNR] + self.cells
        self.arms = {UNR: UNR_ARMS}
        for c in self.cells:
            self.arms[c] = ROUTED_ARMS
        self.ref_h2 = {UNR: "H2"}
        for c in self.cells:
            self.ref_h2[c] = "H2_PUSH"
        self.POS = {(c, a): np.full(ng, -1, np.int64) for c in self.cellsU for a in self.arms[c]}
        self.CONT = {c: np.zeros(ng, bool) for c in self.cells}
        self.CMASS = {c: np.zeros(nq, np.int64) for c in self.cells}
        self.CMASS[UNR] = np.full(nq, N, np.int64)
        # gold ranks in the typed candidate lists (-1: not a candidate)
        self.GR = {k: np.full(ng, -1, np.int64) for k in (["UNR_CT"] + ["%s__%s" % (c, x) for c in self.cells for x in ("CTL", "CTP")])}
        # per row: gate / seed / schedule diagnostics
        self.TD = {k: np.zeros(nq, np.int64) for k in ("lexical", "anchored", "gate", "k", "n_seeds", "topic_in_seeds", "fallback",
                                                       "fallback_dense_top1_eq_cache")}
        self.SEEDS = np.full((nq, 5), -1, np.int64)
        self.ORDERS = [""] * nq
        # per row: sizes and reads
        self.QC = {UNR: {k: np.zeros(nq, np.int64) for k in ("nCT", "rows_typed", "rows_untyped", "nC2")}}
        for c in self.cells:
            self.QC[c] = {k: np.zeros(nq, np.int64) for k in ("seeds_contacted", "nCTL", "nCTP", "nCTP_out", "nCTR_out", "rows_LOCAL",
                                                              "rows_PUSH", "rows_out_typed", "rows_out", "shards_out", "nC2P")}
        self.FETCH = {(c, a): np.zeros((nq, len(M_CURVE)), np.int64) for c in self.cells for a in FETCH_ARMS}
        self.XSH = {(c, a): np.zeros((nq, len(M_CURVE)), np.int64) for c in self.cells for a in FETCH_ARMS}
        self.LAT = {"base (IR_L1 score, served order, router order, V_q, LS)": [], "UNROUTED H2": [], "seeds + schedule": [],
                    "UNROUTED walks (typed + untyped)": [], "UNROUTED orders": [], "identity check (pinned relwalk x2)": []}
        for c in self.cells:
            for s in ("ES contacted set + L1 order + H2_PUSH", "LOCAL + PUSH walks (typed + untyped each)", "orders"):
                self.LAT["%s %s" % (c, s)] = []
        self.n_identity = 0
        self.record = {"l1_spec (the KBRES constructor's pins)": K.record, "cells": self.cells,
                       "arms": {c: list(a) for c, a in self.arms.items()}, "M_curve": list(M_CURVE), "K0": K0, "ACT": ACT,
                       "struct_families": list(SFAMS), "EPS": EPS, "H": HW, "gate": "G2 (typed graph AND anchored)",
                       "typed_graph": {"relation_vocabulary": list(self.T.vocab), "typed_edge_entries (undirected, both directions)": int(len(self.T.u)),
                                       "is_typed_graph": bool(self.typed_graph), "seconds": round(t_typed, 1)}}

    def row(self, j, of, od, os_, npos, frank, g, sl, d_top1):
        K, N, F, W = self.K, self.N, self.F, self.W
        top = of[:ACT]
        nh = len(top)
        t0 = time.perf_counter()
        # ---- the carried-forward L1a (== _l3d_hop.HopSpec.row == _l1d_kbres.KBResSpec.row)
        U, H, GW = [], [], []
        for f_ in FAMS:
            Fm = F[f_]
            st = Fm["xadj"][top]
            hit, pos = HOP.E.entries(st, Fm["xadj"][top + 1] - st)
            U.append(Fm["adj"][pos].astype(np.int64))
            H.append(hit)
            GW.append(Fm["g"][top][hit])
        u, hit, gw = np.concatenate(U), np.concatenate(H), np.concatenate(GW)
        x = (1.0 / np.arange(1.0, nh + 1.0))[hit]
        x = x * gw
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
        ev = np.zeros(N, bool)
        ev[vq] = True
        nS = len(U[0]) + len(U[1])
        LS = np.bincount(u[:nS], weights=x[:nS], minlength=N)
        ms = np.flatnonzero(LS)
        self.LAT["base (IR_L1 score, served order, router order, V_q, LS)"].append(time.perf_counter() - t0)
        allN = np.ones(N, bool)
        pL1 = positions(FO, g, N)
        self.POS[(UNR, "L1")][sl] = pL1
        # ---- UNROUTED H2 (== HOP v1)
        t0 = time.perf_counter()
        y2, _ = prop(F, ms, LS, N)
        m2 = (y2 > 0) & ~ev
        o2 = clist(m2, y2, frank)
        f2 = rrf_add(f, o2)
        oH2 = ranked(f2, allN, frank)
        pH2 = positions(oH2, g, N)
        self.POS[(UNR, "H2")][sl] = pH2
        self.LAT["UNROUTED H2"].append(time.perf_counter() - t0)
        # ---- seeds, schedule, gate (the lane's functions)
        t0 = time.perf_counter()
        Q = self.Q[j]
        lex = SE.lexical_seeds(Q, self.idx, self.names, self.T)
        if lex:
            seeds = [int(s_) for s_ in lex[:5]]
        else:
            seeds = [int(od[0])] + ([int(os_[0])] if npos > 0 and int(os_[0]) != int(od[0]) else [])
        order, anchored = SE.schedule3(self.T, Q, seeds, key="wh")
        order = list(order)
        gate = bool(anchored) and self.typed_graph
        x0 = np.zeros(N)
        x0[np.asarray(seeds, np.int64)] = 1.0 / float(len(seeds))
        td = self.TD
        td["lexical"][j], td["anchored"][j], td["gate"][j], td["k"][j], td["n_seeds"][j] = bool(lex), bool(anchored), gate, len(order), len(seeds)
        td["topic_in_seeds"][j] = int(K.TOPIC[j] >= 0 and int(K.TOPIC[j]) in seeds)
        td["fallback"][j] = int(not lex)
        td["fallback_dense_top1_eq_cache"][j] = int((not lex) and int(od[0]) == int(d_top1))
        self.SEEDS[j, :len(seeds)] = seeds
        self.ORDERS[j] = ",".join(self.T.vocab[r_] for r_ in order)
        self.LAT["seeds + schedule"].append(time.perf_counter() - t0)
        # ---- UNROUTED walks (== the ROWS walks of every cell)
        t0 = time.perf_counter()
        aU, oU, rdU = W.walk(x0, order)
        _, uU, rduU = W.walk(x0, [])
        self.LAT["UNROUTED walks (typed + untyped)"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        sd = np.full((1, 5), -1, np.int64)
        sd[0, :len(seeds)] = seeds
        ref = RW.relwalk(self.C, self.T, [Q], sd, [order], H=HW, mix=False)
        refu = RW.relwalk(self.C, self.T, [Q], sd, [[]], H=HW, mix=False)
        assert (ref["seed"][0] == x0.astype(np.float32)).all(), "seed channel differs from the pinned relwalk (row %d)" % j
        assert (ref["ans"][0] == aU.astype(np.float32)).all(), "ans channel differs from the pinned relwalk (row %d)" % j
        assert (ref["oth"][0] == oU.astype(np.float32)).all(), "oth channel differs from the pinned relwalk (row %d)" % j
        assert (refu["oth"][0] == uU.astype(np.float32)).all(), "untyped channel differs from the pinned relwalk (row %d)" % j
        assert not refu["ans"][0].any()
        self.n_identity += 1
        self.LAT["identity check (pinned relwalk x2)"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ctU = tcands(x0, aU, oU, uU, frank) if gate else EMPTY
        inU = member(ctU, N)
        oT1 = first_then(ctU, FO, N)
        oTR = ranked(rrf_add(f, ctU), allN, frank)
        oTRH2 = ranked(rrf_add(f2, ctU), allN, frank)
        for a, o in (("T1", oT1), ("TR", oTR), ("TR_H2", oTRH2)):
            assert len(o) == N
            self.POS[(UNR, a)][sl] = positions(o, g, N)
        if not gate:
            assert (self.POS[(UNR, "T1")][sl] == pL1).all() and (self.POS[(UNR, "TR")][sl] == pL1).all()
            assert (self.POS[(UNR, "TR_H2")][sl] == pH2).all()
        self.GR["UNR_CT"][sl] = c2rank(ctU, g, N)
        q = self.QC[UNR]
        q["nCT"][j], q["rows_typed"][j], q["rows_untyped"][j], q["nC2"][j] = len(ctU), int(rdU.sum()), int(rduU.sum()), len(o2)
        self.LAT["UNROUTED orders"].append(time.perf_counter() - t0)
        # ---- routed cells
        for c in self.cells:
            t0 = time.perf_counter()
            P = K.parts[c]
            npart, hard = P.npart, P.hard
            hq = hard[oq]
            up, first = np.unique(hq, return_index=True)
            fe = np.full(npart, len(oq), np.int64)
            fe[up] = first
            ro = np.lexsort((np.arange(npart), fe))
            pr = RT.inv(ro)
            b = int(K.B[c][j])
            assert 1 <= b <= npart
            assert int(pr[hard[g]].max()) + 1 == K.LOK[c][j], "recomputed ES LO differs (%s row %d)" % (c, j)
            contn = pr[hard] < b
            cm = int(contn.sum())
            self.CMASS[c][j] = cm
            self.CONT[c][sl] = contn[g]
            o1 = FO[contn[FO]]
            pl1 = positions(o1, g, N)
            self.POS[(c, "L1")][sl] = pl1
            nmax = min(max(M_CURVE), cm)
            src2 = ms[contn[ms]]
            y2c, _ = prop(F, src2, LS, N)
            m2P = (y2c > 0) & ~ev
            o2P = clist(m2P, y2c, frank)
            f2P = rrf_add(f, o2P)
            elP2 = contn | m2P
            oH2P = ranked(f2P, elP2, frank)
            ph2p = positions(oH2P, g, N)
            self.POS[(c, "H2_PUSH")][sl] = ph2p
            self.LAT["%s ES contacted set + L1 order + H2_PUSH" % c].append(time.perf_counter() - t0)
            # typed walks under the masks
            t0 = time.perf_counter()
            q = self.QC[c]
            if gate:
                x0L = x0 * contn
                aL, oL, rdL = W.walk(x0L, order, srcm=contn, tgtm=contn)
                _, uL, rduL = W.walk(x0L, [], srcm=contn, tgtm=contn)
                ctL = tcands(x0L, aL, oL, uL, frank)
                aP, oP, rdP = W.walk(x0, order, srcm=contn)
                _, uP, rduP = W.walk(x0, [], srcm=contn)
                ctP = tcands(x0, aP, oP, uP, frank)
                assert contn[ctL].all() and not (rdL & ~contn).any() and not (rdP & ~contn).any()
                assert not (rduL & ~contn).any() and not (rduP & ~contn).any()
                q["rows_LOCAL"][j] = int((rdL | rduL).sum())
                q["rows_PUSH"][j] = int((rdP | rduP).sum())
            else:
                ctL = ctP = EMPTY
            ctR = ctU
            self.LAT["%s LOCAL + PUSH walks (typed + untyped each)" % c].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            inP, inR = member(ctP, N), inU
            orders = {"T1_LOCAL": first_then(ctL, o1, N), "TR_LOCAL": ranked(rrf_add(f, ctL), contn, frank),
                      "T1_PUSH": first_then(ctP, o1, N), "TR_PUSH": ranked(rrf_add(f, ctP), contn | inP, frank),
                      "T1_ROWS": first_then(ctR, o1, N), "TR_ROWS": ranked(rrf_add(f, ctR), contn | inR, frank),
                      "TR_H2_PUSH": ranked(rrf_add(f2P, ctP), elP2 | inP, frank)}
            for a, o in orders.items():
                self.POS[(c, a)][sl] = positions(o, g, N)
                assert len(o) >= cm
            for a in ("T1_LOCAL", "TR_LOCAL"):
                assert contn[orders[a]].all() and len(orders[a]) == cm
            if not gate:
                for a in ("T1_LOCAL", "TR_LOCAL", "T1_PUSH", "TR_PUSH", "T1_ROWS", "TR_ROWS"):
                    assert (self.POS[(c, a)][sl] == pl1).all()
                assert (self.POS[(c, "TR_H2_PUSH")][sl] == ph2p).all()
            orders["H2_PUSH"] = oH2P
            for a in FETCH_ARMS:
                o = orders[a][:nmax]
                out = ~contn[o]
                for mi, M in enumerate(M_CURVE):
                    n_ = min(M, cm)
                    ob = out[:n_]
                    self.FETCH[(c, a)][j, mi] = int(ob.sum())
                    self.XSH[(c, a)][j, mi] = int(len(np.unique(hard[o[:n_][ob]])))
            self.GR["%s__CTL" % c][sl] = c2rank(ctL, g, N)
            self.GR["%s__CTP" % c][sl] = c2rank(ctP, g, N)
            rout_t = rdU & ~contn if gate else np.zeros(N, bool)
            rout = (rdU | rduU) & ~contn if gate else np.zeros(N, bool)
            sa = np.asarray(seeds, np.int64)
            vals = {"seeds_contacted": int(contn[sa].sum()), "nCTL": len(ctL), "nCTP": len(ctP), "nCTP_out": int((~contn[ctP]).sum()),
                    "nCTR_out": int((~contn[ctR]).sum()), "rows_out_typed": int(rout_t.sum()), "rows_out": int(rout.sum()),
                    "shards_out": int(len(np.unique(hard[np.flatnonzero(rout)]))), "nC2P": len(o2P)}
            for k_, v_ in vals.items():
                q[k_][j] = v_
            self.LAT["%s orders" % c].append(time.perf_counter() - t0)

    # ---- tables
    def served(self, c, a, M):
        pop = self.pop
        cap = np.minimum(M, self.CMASS[c])[pop.row_of_gold]
        sv = self.POS[(c, a)] < cap
        cnt = np.add.reduceat(sv.astype(np.int64), pop.gptr[:-1])
        return cnt == pop.ngold, cnt > 0, cnt / pop.ngold.astype(np.float64), sv

    def finish(self):
        pop = self.pop
        nq, hops, ngold = pop.nq, pop.hops, pop.ngold
        rog = pop.row_of_gold
        gate = self.TD["gate"].astype(bool)
        QS = {"all": np.ones(nq, bool)}
        HS = sorted(set(int(h) for h in hops))
        for h in HS:
            QS["hop%d" % h] = hops == h
        for nm, lo, hi in D.NG_BUCKETS:
            QS["ng_" + nm] = (ngold >= lo) & (ngold <= hi)
        for h in HS:
            for nm, lo, hi in D.NG_BUCKETS:
                m = (hops == h) & (ngold >= lo) & (ngold <= hi)
                if m.any():
                    QS["hop%d|ng_%s" % (h, nm)] = m
        QS["gate"] = gate
        QS["no_gate"] = ~gate
        for h in HS:
            QS["hop%d|gate" % h] = (hops == h) & gate
            QS["hop%d|no_gate" % h] = (hops == h) & ~gate
        qts = sorted(set(self.K.QTYPE))
        QT = {qt: np.array([x == qt for x in self.K.QTYPE]) for qt in qts}
        res = {}
        for c in self.cellsU:
            res[c] = {}
            ref = {M: self.served(c, "L1", M) for M in M_CURVE}
            rh2 = {M: self.served(c, self.ref_h2[c], M) for M in M_CURVE}
            for a in self.arms[c]:
                res[c][a] = {}
                for mi, M in enumerate(M_CURVE):
                    all_, any_, frac, sv = self.served(c, a, M)
                    nserv = np.minimum(M, self.CMASS[c])
                    e = {"ALL": int(all_.sum()), "ANY": int(any_.sum()), "FRAC_mean": D.q4(frac.mean()), "gold_nodes_served": int(sv.sum()),
                         "served_nodes_mean": round(float(nserv.mean()), 2),
                         "strata (ALL rows, rows)": {k: [int(all_[m].sum()), int(m.sum())] for k, m in QS.items()},
                         "strata (gold nodes served, gold nodes)": {k: [int(sv[m[rog]].sum()), int(m[rog].sum())] for k, m in QS.items()
                                                                    if k in ["all", "gate", "no_gate"] + ["hop%d" % h for h in HS]}}
                    if a != "L1":
                        r_all = ref[M][0]
                        e["paired_vs_L1 (gained = arm serves ALL gold, L1 does not)"] = dict(
                            D.paired(r_all, all_), **{"strata": {k: D.paired(r_all, all_, m) for k, m in QS.items() if k != "all"}})
                        e["gold_nodes gained / lost vs L1"] = [int((sv & ~ref[M][3]).sum()), int((~sv & ref[M][3]).sum())]
                    if a in TYPED_ARMS:
                        h_all = rh2[M][0]
                        e["paired_vs_H2 (gained = arm serves ALL gold, the cell's %s does not)" % self.ref_h2[c]] = dict(
                            D.paired(h_all, all_), **{"strata": {k: D.paired(h_all, all_, m) for k, m in QS.items() if k != "all"}})
                    if (c, a) in self.FETCH:
                        fe, xs = self.FETCH[(c, a)][:, mi], self.XSH[(c, a)][:, mi]
                        e["fetched_nodes_per_query (served outside the contacted shards)"] = {
                            "mean": round(float(fe.mean()), 2), "median": float(np.median(fe)), "p95": float(np.percentile(fe, 95)),
                            "max": int(fe.max()), "queries_with_any": int((fe > 0).sum()), "share_of_served": D.q4(fe.sum() / float(nserv.sum()))}
                        e["extra_shards_per_query (distinct uncontacted shards of the fetched nodes)"] = {
                            "mean": round(float(xs.mean()), 2), "median": float(np.median(xs)), "p95": float(np.percentile(xs, 95)),
                            "max": int(xs.max())}
                    if M in QT_SHOW:
                        e["per_qtype (ALL rows, rows)"] = {qt: [int(all_[m].sum()), int(m.sum())] for qt, m in QT.items()}
                    res[c][a][str(M)] = e
        # gate / seed / schedule diagnostics, overall and per hop
        td = {}
        for k, m in [("all", np.ones(nq, bool))] + [("hop%d" % h, hops == h) for h in HS]:
            e = {"rows": int(m.sum())}
            for kk in ("lexical", "anchored", "gate", "fallback", "fallback_dense_top1_eq_cache"):
                e[kk] = int(self.TD[kk][m].sum())
            e["topic_in_seeds (gated rows)"] = int(self.TD["topic_in_seeds"][m & gate].sum())
            e["topic_in_seeds (all rows)"] = int(self.TD["topic_in_seeds"][m].sum())
            e["schedule_length (rows by k; gated rows)"] = {str(x): int(((self.TD["k"] == x) & m & gate).sum()) for x in range(0, 10)}
            e["n_seeds (rows)"] = {str(x): int(((self.TD["n_seeds"] == x) & m).sum()) for x in range(1, 6)}
            td[k] = e
        top_orders = {}
        for h in HS:
            cnt = {}
            for jj in np.flatnonzero((hops == h) & gate):
                cnt[self.ORDERS[jj]] = cnt.get(self.ORDERS[jj], 0) + 1
            top_orders["hop%d" % h] = sorted(([k_, v_] for k_, v_ in cnt.items()), key=lambda kv: (-kv[1], kv[0]))[:15]
        td["most frequent schedules (gated rows, per hop)"] = top_orders
        for c in self.cells:
            td["seeds all contacted (%s; gated rows)" % c] = int(((self.QC[c]["seeds_contacted"] == self.TD["n_seeds"]) & gate).sum())
            td["seeds none contacted (%s; gated rows)" % c] = int(((self.QC[c]["seeds_contacted"] == 0) & gate).sum())
        cost = {}
        for c in self.cellsU:
            cost[c] = {k: (KB.AB.dist(v[gate]) if gate.any() else None) if k not in ("nC2", "nC2P") else KB.AB.dist(v) for k, v in self.QC[c].items()}
            if c != UNR:
                cost[c]["contacted_mass"] = KB.AB.dist(self.CMASS[c])
                cost[c]["B_P"] = KB.AB.dist(self.K.B[c])
        cand = {}
        for k, r in self.GR.items():
            e = {"golds_in_list": int((r >= 0).sum()), **{"rank < %d" % t: int(((r >= 0) & (r < t)).sum()) for t in (10, 100, 1000, 5000)}}
            e["per hop (golds in list, gold nodes)"] = {"hop%d" % h: [int(((r >= 0) & (hops[rog] == h)).sum()), int((hops[rog] == h).sum())] for h in HS}
            cand[k] = e
        lat = {k: D.ms_stats(v) for k, v in self.LAT.items()}
        arrays = {"HOPS": hops, "QTYPE": np.array(self.K.QTYPE), "m_curve": np.asarray(M_CURVE, np.int64), "cells": np.array(self.cellsU),
                  "SEEDS": self.SEEDS, "ORDERS": np.array(self.ORDERS)}
        for k, v in self.TD.items():
            arrays["TD__" + k] = v
        for (c, a), v in self.POS.items():
            arrays["POS__%s__%s" % (c, a)] = v
        for c in self.cells:
            arrays["CONT__" + c] = self.CONT[c]
            arrays["CMASS__" + c] = self.CMASS[c]
            arrays["B__" + c] = self.K.B[c]
            for a in FETCH_ARMS:
                arrays["FETCH__%s__%s" % (c, a)] = self.FETCH[(c, a)]
                arrays["XSH__%s__%s" % (c, a)] = self.XSH[(c, a)]
        for c in self.cellsU:
            for k, v in self.QC[c].items():
                arrays["QC__%s__%s" % (c, k)] = v
        for k, v in self.GR.items():
            arrays["GR__" + k] = v
        return {"results (per cell, arm, B_N)": res, "typed_channel_diagnostics": td,
                "traversal_cost_per_query (typed quantities over gated rows; nC2 / nC2P over all rows)": cost,
                "typed_candidate_ranks_of_gold_nodes": cand, "latency_ms_per_row": lat}, arrays


def l1_identity(S, POS_FLAT, full):
    """the L1 arms against the kbres v1 record and the L1 / H2 / H2_PUSH arms against the HOP v1 record (write-once, sha pinned)."""
    assert D.sha_file(HOP.KBRES_NPZ) == HOP.KBRES_NPZ_SHA and D.sha_file(HOP.KBRES_JSON) == HOP.KBRES_JSON_SHA
    assert D.sha_file(HOP_NPZ) == HOP_NPZ_SHA and D.sha_file(HOP_JSON) == HOP_JSON_SHA
    z = np.load(HOP.KBRES_NPZ)
    h = np.load(HOP_NPZ)
    pop = S.pop
    nq, ng = pop.nq, pop.ng_tot
    for zz in (z, h):
        assert (zz["gptr"][:nq + 1] == pop.gptr).all() and (zz["pos_FLAT"][:ng] == POS_FLAT).all()
    assert (h["rows"][:nq] == pop.rows).all()
    assert (z["G_FRK"][:ng] == S.POS[(UNR, "L1")]).all(), "UNROUTED L1 differs from kbres v1"
    assert (h["POS__%s__L1" % UNR][:ng] == S.POS[(UNR, "L1")]).all()
    assert (h["POS__%s__H2" % UNR][:ng] == S.POS[(UNR, "H2")]).all(), "UNROUTED H2 differs from HOP v1"
    for c in S.cells:
        cont, rr = z["cont__" + c][:ng], z["rr__" + c][:ng]
        assert (cont == S.CONT[c]).all(), "contacted flags differ from kbres v1 (%s)" % c
        assert (np.where(cont, rr, S.N) == S.POS[(c, "L1")]).all(), "routed L1 positions differ from kbres v1 (%s)" % c
        assert (z["Q_CMASS__" + c][:nq] == S.CMASS[c]).all()
        assert (h["POS__%s__L1" % c][:ng] == S.POS[(c, "L1")]).all() and (h["CMASS__" + c][:nq] == S.CMASS[c]).all()
        assert (h["POS__%s__H2_PUSH" % c][:ng] == S.POS[(c, "H2_PUSH")]).all(), "H2_PUSH differs from HOP v1 (%s)" % c
    out = "on all %d gold nodes / %d rows: UNROUTED L1 == kbres v1 G_FRK, routed contacted flags and L1 ranks == kbres v1 cont__ / " \
          "rr__, contacted mass == Q_CMASS__ (%s, sha %s); the L1 / H2 / H2_PUSH positions and the contacted mass == %s (sha %s)" % (
              ng, nq, D.rel(HOP.KBRES_NPZ), HOP.KBRES_NPZ_SHA[:16], D.rel(HOP_NPZ), HOP_NPZ_SHA[:16])
    if full:
        T = json.load(open(HOP.KBRES_JSON, encoding="utf-8"))["diagnostics"]["tables (per cell, B_N, stratum: query level and gold level)"]
        TH = json.load(open(HOP_JSON, encoding="utf-8"))["diagnostics"]["results (per cell, arm, B_N)"]
        for c in S.cellsU:
            for M in M_CURVE:
                assert int(S.served(c, "L1", M)[0].sum()) == T[c][str(M)]["all"]["queries"]["ALL"], "L1 ALL differs (%s, %d)" % (c, M)
                a = S.ref_h2[c]
                assert int(S.served(c, a, M)[0].sum()) == TH[c][a][str(M)]["ALL"], "%s ALL differs from HOP v1 (%s, %d)" % (a, c, M)
        out += "; the L1 ALL counts of every cell and B_N == the kbres v1 tables and the H2 / H2_PUSH ALL counts == the HOP v1 tables"
    return out


def run(ds, tag):
    ROWS, SMOKE_OUT = R.argv_opts()
    here = os.path.abspath(__file__)
    shas = {"harness": D.sha_file(here)}
    for m in MODULES + tuple(sorted(LANE)):
        shas[m] = D.sha_file(os.path.join(D.HERE, m))
    assert shas["_l1d_kbres.py"] == HOP.KBRES_MOD_SHA, "the KBRES module changed"
    assert shas["_l3d_hop.py"] == HOP_MOD_SHA, "the HOP module changed"
    for m, s in LANE.items():
        assert shas[m] == s and D.sha_file(os.path.join(SNAP, m)) == s, "the lane module %s differs from its code snapshot" % m
    host0 = D.host_state()
    t_all = time.time()
    outdir = SMOKE_OUT or OUT3
    os.makedirs(outdir, exist_ok=True)
    fp_out = os.path.join(outdir, "l3typed_%s__%s.json" % (ds, tag))
    fz = fp_out.replace(".json", ".npz")
    assert not os.path.exists(fp_out) and not os.path.exists(fz), "write-once: %s exists" % fp_out
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    cells = D.CELLS[ds]
    pop = D.Population(cd, ROWS)
    nq, gptr = pop.nq, pop.gptr
    t_ = time.time()
    Q, qrec = population_questions(cd, pop.rows)
    qrec["seconds"] = round(time.time() - t_, 1)
    parts = {c: D.Part(cd, D.TAG_OF[c]) for c in cells}
    t_ = time.time()
    S = TypedSpec(cd, pop, parts, Q)
    S.record["seconds_build"] = round(time.time() - t_, 1)
    S.record["question_records"] = qrec
    log("RUN l3typed %s %s: N %d, %d rows, %d gold nodes, cells %s, relations %s" % (ds, tag, N, nq, pop.ng_tot, S.cellsU, S.T.vocab))
    POS_FLAT = np.zeros(pop.ng_tot, np.int64)
    LAT = {"products_amortized": [], "flat_rrf": []}
    agd, ags = np.zeros(nq), np.zeros(nq)
    d200 = np.asarray(cd.dense_topk(D.ACT, pop.rows), np.int64)
    s200 = np.asarray(cd.splade_topk(D.ACT, pop.rows), np.int64)
    Qu = D.unit_queries(cd, pop.rows)
    t_ = time.time()
    for j0, j1, SD, SS, sec, _ in D.batches(cd, pop.rows, Qu, N):
        LAT["products_amortized"] += [sec / float(j1 - j0)] * (j1 - j0)
        for i in range(j1 - j0):
            j = j0 + i
            t0 = time.perf_counter()
            of, od, os_, npos_j, fv, frank = D.flat_row(SD[i], SS[i])
            LAT["flat_rrf"].append(time.perf_counter() - t0)
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
    assert (z1["rows"][:nq] == pop.rows).all()
    ng = int(z1["gptr"][nq])
    assert ng == pop.ng_tot and (z1["pos_FLAT"][:ng] == POS_FLAT).all(), "FLAT positions differ from the v1 record"
    v1check = "FLAT positions == %s (sha %s) on all %d gold nodes" % (D.rel(f1), D.sha_file(f1)[:16], ng)
    log("v1 check: " + v1check)
    idcheck = l1_identity(S, POS_FLAT, ROWS is None)
    log("identity: " + idcheck)
    walkcheck = "on all %d rows the unmasked typed walk (seed / ans / oth) and the untyped walk == the pinned " \
                "_l1x90_relwalk.relwalk (identity block indicator) at float32" % nq
    diag, arrays = S.finish()
    for c in S.cellsU:
        for a in S.arms[c]:
            log("%-9s %-11s ALL %s" % (c, a, " ".join("%5d" % diag["results (per cell, arm, B_N)"][c][a][str(M)]["ALL"] for M in M_CURVE)))
    res = {"dataset": ds, "tag": tag, "stage": "L3 development (TYPED)", "status": "DEVELOPMENT (descriptive; p-values descriptive; no rule selected)",
           "definitions": __doc__, "N": N, "n_rows": nq, "n_gold_nodes": pop.ng_tot, "population": pop.record, "structures": S.record,
           "v1_check": v1check, "l1_identity": idcheck, "walk_identity": walkcheck, "served_list_agreement": agree, "diagnostics": diag,
           "latency_ms_flat": {"products_amortized": D.ms_stats(LAT["products_amortized"]), "flat_rrf": D.ms_stats(LAT["flat_rrf"])},
           "code": {k: {"path": "scratchpad/" + (os.path.basename(here) if k == "harness" else k), "sha256": v} for k, v in shas.items()},
           "inputs": {"kbres_v1_json": {"path": D.rel(HOP.KBRES_JSON), "sha256": HOP.KBRES_JSON_SHA},
                      "kbres_v1_npz": {"path": D.rel(HOP.KBRES_NPZ), "sha256": HOP.KBRES_NPZ_SHA},
                      "l3hop_v1_json": {"path": D.rel(HOP_JSON), "sha256": HOP_JSON_SHA},
                      "l3hop_v1_npz": {"path": D.rel(HOP_NPZ), "sha256": HOP_NPZ_SHA},
                      "lane_code_snapshot": {"path": D.rel(SNAP), "sha256": LANE}},
           "pinned": D.PINNED, "pinned_repo": D.PINNED_REPO, "constants": D.CONSTANTS, "platform": D.platform_record(),
           "host_at_start": host0, "seconds_loop": t_loop, "_row_query_ids": pop.qids}
    arrays.update({"rows": pop.rows, "gptr": gptr, "pos_FLAT": POS_FLAT})
    res["seconds"] = round(time.time() - t_all, 1)
    res["process_peak_rss_mb"] = D.peak_rss_mb()
    for m in MODULES + tuple(sorted(LANE)):
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
    assert ds in ("metaqa",), "TYPED is defined on the KB development dataset with a typed structural graph and a HOP v1 record"
    run(ds, tag)


if __name__ == "__main__":
    main()
