"""L3 DEVELOPMENT -- E2b: a guarded, fill-ordered union of the lexical and the encoder-appended typed walks (REPORT section 41.15),
behind the carried-forward L1, at equal exposure.
(User 2026-09-29: "improving the KB side should be the highest priority now"; "we need to finish the full KB path: L1
localization -> L2 reranking -> L3 multi-hop completion".  E2 v1 (results/L3_DEV/l3e2_metaqa__v1.json, e3814532...; REPORT section
41) found that only the union UA_<c> of the lexical walk and the encoder-appended walk helps (UA_S serves ALL gold on 1,979 rows
at B_N 1000 on PHG_k432 against LEX's 1,967, +22 / -10), and that its 10 losses at B_N 1000 and its 62 losses at B_N 100 have two
causes: a FLOOD (a gated row whose lexical schedule is empty gets a one-relation schedule that a typed walk repeats on all three
steps, and the candidate list becomes thousands of nodes) and a DILUTION (the RRF interleaves the wrong-relation answers of the
appended walk with the lexical walk's own list, so the last golds fall beyond a small B_N).  E2b tests two parameter-free rules
that avoid them, as pre-registered in REPORT section 41.15: a guard on the lexical schedule's emptiness, and a fill order.)
L1 is not changed: everything below runs AFTER the carried-forward L1 exactly as in _l3d_typed (TYPED v1) and _l3d_e2 (E2 v1).
Development rows only: the L1_DEV population (c7f70806...), legacy-exposed; no reserved held-out row, no split-B row and no TEST row
is read.  No encoder training, no LLM, no learned weight, no tuned constant (K0 60, EPS 1e-6 and H 3 are the frozen / lane
constants).  STATUS: DEVELOPMENT -- descriptive numbers, every p-value descriptive, no rule selected here.

Inputs, the relation scores and the schedules are E2 v1's (this module is _l3d_e2.py with the changes listed at the end; no model
runs; the RELENC vectors and the population rows' frozen query vectors are read exactly as there):
  D / S / DS   the channel scores over the 81 relation strings (dense cosine, SPLADE dot, RRF of the two ranks, K0 60)
  LEX          _l1x90_seeds.schedule3(T, question, seeds, key="wh") (== TYPED v1, asserted); k = its length, the number of relations
  XA_<c>       LEX plus the channel's best relation outside LEX, always appended
Candidates: CT(schedule) = the lane's typed candidates (seed + ans > 0, by (-M, FLAT rank)) from the pinned _l3d_typed.Walker,
  unmasked (the ROWS mask of TYPED v1); with an empty schedule CT is the seed nodes alone.
Arms (per channel c in D, S, DS; the four unions differ in two switches: how the two lists are merged, and whether the guard applies):
  UA_<c>   the RRF of CT(LEX) and CT(XA_<c>), K0 60, ties -> FLAT rank (E2 v1's arm, re-run and asserted identical)
  UAG_<c>  UA_<c> where k > 0, else CT(LEX).  The guard is the number of relations the lexical schedule matched: known before any
           walk, without gold, and a property of the question against the schema, not of the dataset
  UF_<c>   the FILL union: CT(LEX) in its own order, then the candidates of CT(XA_<c>) that are not in CT(LEX), in CT(XA_<c>)'s own
           order.  No fusion and no constant: the lexical walk's list stays ahead of everything the appended walk adds
  UFG_<c>  UF_<c> where k > 0, else CT(LEX)
  LEX, XA_<c>  as in E2 v1 (re-run and asserted identical); L1 is the cell's L1 order
An arm serves its list first, then the cell's L1 order of the remaining nodes, n(q, B_N) = min(B_N, contacted mass(q)) nodes (equal
  exposure).  Outside the gate every arm serves the L1 list (asserted).  Cells: UNROUTED, and PHG_k432 / MTK_k432 with the ROWS mask.
Reported per cell, arm and B_N (100 .. 5000): ALL / ANY / FRAC and gold nodes served, per hop, answer-count bucket and gate; McNemar
  gained / lost vs LEX and vs L1 (descriptive); fetched nodes and extra shards (routed); the schedules; the misses by class; the
  28 NOT_REACHED rows of LEX; candidate ranks and list lengths; latency per stage.

Identities asserted: FLAT == loc v1; the dense / SPLADE top-100 agreement gate; the relation vocabulary == the RELENC strings' labels;
on every row the seeds, the lexical schedule and the gate == TYPED v1; on every gold node the L1 positions, the contacted flags and
mass and B_P == TYPED v1, LEX == TYPED v1 T1 (UNROUTED) and T1_ROWS (routed), and the gold ranks in CT(LEX) == TYPED v1 GR__UNR_CT;
against the E2 v1 record (json and npz shas pinned): the lexical and the XA schedules, and on every gold node the served positions
of LEX, XA_<c> and UA_<c> in every cell, their candidate gold ranks and list lengths; and the two guard arms' positions equal, on
every gold node, the E2 v1 positions of UA_<c> where k > 0 and of LEX where k = 0 (UAG), which is the post-hoc derivation of REPORT
section 41.15.

Changes from _l3d_e2.py: the arms XT_<c> and P_<c> are dropped (E2 v1 holds them); the schedule dict holds LEX and XA_<c> only; the
arms UAG_<c>, UF_<c> and UFG_<c> are added; the identity check against the E2 v1 record is added; the record names l3e2b_*.

Usage: python scratchpad/_l3d_e2b.py RUN metaqa <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L3_DEV/l3e2b_<dataset>__<tag>.{json,npz} (write-once)
"""
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
import _l3d_typed as TT
import _l1x90_typed as TY
import _l1x90_seeds as SE
import _l1x90_universal as UNI

log = D.log
K0, ACT = D.K0, D.ACT
FAMS = KB.FAMS
M_CURVE = D.M_CURVE
UNR = KB.UNR
OUT3 = HOP.OUT3
positions, c2rank = HOP.positions, HOP.c2rank
tcands, first_then = TT.tcands, TT.first_then
EMPTY = TT.EMPTY
TYPED_MOD_SHA = "8eb4cbde3cc54c147f519a7aa0ae9974ae910ac6b32ad90ee673e53064266ac9"
TYPED_JSON = os.path.join(OUT3, "l3typed_metaqa__v1.json")
TYPED_JSON_SHA = "a6e80216a802d12ec91ded5d5a7d6c59f24e7339a2efd34f6e447486cc7b90a4"
TYPED_NPZ = os.path.join(OUT3, "l3typed_metaqa__v1.npz")
TYPED_NPZ_SHA = "0ee65e76660223d0378708582e8bd19142e58595ff495340705d5112d29e1355"
RELENC_JSON = os.path.join(OUT3, "relenc_metaqa__v1.json")
RELENC_JSON_SHA = "6290f3bdec5dd5d3b145cd31da80f6c7093e69c4257b6d752ab0624d56ccac30"
RELENC_NPZ = os.path.join(OUT3, "relenc_metaqa__v1.npz")
RELENC_NPZ_SHA = "17231a29d8b8fd0d379a654257e21a3758b1635a170db38f924245527ae7d78e"
MODULES = TT.MODULES + ("_l3d_typed.py",)
LANE = TT.LANE
SNAP = TT.SNAP
CH = ("D", "S", "DS")
E2V1_JSON = os.path.join(OUT3, "l3e2_metaqa__v1.json")
E2V1_JSON_SHA = "e381453219a183400de485ac8454a83e77b31879ce9d2ba8fb01b75ef8688600"
E2V1_NPZ = os.path.join(OUT3, "l3e2_metaqa__v1.npz")
E2V1_NPZ_SHA = "512006698bab305061cbb757d287c9710f56af3583db7029702ee42f9c790412"
SCHED = ("LEX",) + tuple("XA_%s" % c for c in CH)                       # the walked schedules (XT and P are E2 v1's, not re-run)
UNIONS = tuple("%s_%s" % (k, c) for k in ("UA", "UAG", "UF", "UFG") for c in CH)
ARMS = ("L1",) + SCHED + UNIONS
CAND_ARMS = ARMS[1:]
QT_SHOW = (1000, 5000)
MISS_M = (100, 1000, 5000)
NR_CELL, NR_M = "PHG_k432", 1000          # TYPED v1's residual table (REPORT 40.10): the LEX rows NOT_REACHED there


def rank_rows(s, tie=None):
    """per row, the string indices by (-s, tie, index)."""
    n, m = s.shape
    idx = np.arange(m)
    out = np.empty((n, m), np.int64)
    for i in range(n):
        keys = (idx, -s[i]) if tie is None else (idx, tie[i], -s[i])
        out[i] = np.lexsort(keys)
    return out


def inv_rows(o):
    r = np.empty_like(o)
    n, m = o.shape
    r[np.arange(n)[:, None], o] = np.arange(m)[None, :]
    return r


def channel_orders(sD, sS):
    """{channel: per-row string order}: D and S by their own score, DS by the RRF of the two ranks (K0), ties -> D rank."""
    oD = rank_rows(sD)
    oS = rank_rows(sS)
    rD, rS = inv_rows(oD), inv_rows(oS)
    f = 1.0 / (K0 + rD.astype(np.float64)) + 1.0 / (K0 + rS.astype(np.float64))
    oDS = rank_rows(f, tie=rD)
    return {"D": oD, "S": oS, "DS": oDS}


def fuse(a, b, frank, N):
    """RRF of two candidate lists (K0), ties -> FLAT rank; the union of both."""
    f = np.zeros(N)
    f[a] += 1.0 / (K0 + np.arange(len(a), dtype=np.float64))
    f[b] += 1.0 / (K0 + np.arange(len(b), dtype=np.float64))
    c = np.union1d(a, b)
    return c[np.lexsort((frank[c], -f[c]))]


def fill(a, b):
    """the FILL union of two candidate lists: a in its own order, then the elements of b that are not in a, in b's own order."""
    if not len(b):
        return a
    return np.concatenate([a, b[~np.isin(b, a)]])


class RelScores(object):
    """the relation strings' scores against the population rows' frozen query vectors, and the per-row schedules they imply."""

    def __init__(self, cd, pop, vocab):
        assert D.sha_file(RELENC_JSON) == RELENC_JSON_SHA and D.sha_file(RELENC_NPZ) == RELENC_NPZ_SHA, "the RELENC record changed"
        rj = json.load(open(RELENC_JSON, encoding="utf-8"))
        assert rj["dataset"] == "metaqa" and rj["relation_vocabulary"] == list(vocab), "relation vocabulary differs from RELENC"
        z = np.load(RELENC_NPZ)
        kind, r1, r2 = z["kind"].astype(np.int64), z["r1"].astype(np.int64), z["r2"].astype(np.int64)
        V = len(vocab)
        m = len(kind)
        assert m == len(rj["strings"]) == V + V * (V - 1)
        assert (kind[:V] == 0).all() and (r1[:V] == np.arange(V)).all() and (r2[:V] == -1).all() and (kind[V:] == 1).all()
        assert (r1[V:] != r2[V:]).all() and len(set(zip(r1[V:].tolist(), r2[V:].tolist()))) == V * (V - 1)
        verb = [lbl.replace("_", " ").replace(".", " ") for lbl in vocab]
        assert rj["strings"][:V] == verb and all(rj["strings"][i] == verb[r1[i]] + ", " + verb[r2[i]] for i in range(V, m))
        self.V, self.m, self.kind, self.r1, self.r2, self.strings = V, m, kind, r1, r2, rj["strings"]
        t0 = time.time()
        Rd = z["dense"].astype(np.float64)
        Rs = sp.csr_matrix((z["sp_data"], z["sp_indices"], z["sp_indptr"]), shape=tuple(int(x) for x in z["sp_shape"]))
        Qu = D.unit_queries(cd, pop.rows).astype(np.float64)
        Qs = cd.embeddings("splade", "queries").read(pop.rows).tocsr()
        assert Rd.shape == (m, Qu.shape[1]) and Rs.shape == (m, Qs.shape[1])
        self.sD = np.einsum("qd,rd->qr", Qu, Rd)
        self.sS = np.asarray((Qs.astype(np.float64) @ Rs.astype(np.float64).T).todense())
        assert np.isfinite(self.sD).all() and np.isfinite(self.sS).all() and (self.sS >= 0).all()
        # relation level (the 9 labels) and path level (all 81 strings)
        self.rel = channel_orders(self.sD[:, :V], self.sS[:, :V])
        self.path = channel_orders(self.sD, self.sS)
        self.seconds = round(time.time() - t0, 2)
        self.record = {"relenc_json": {"path": D.rel(RELENC_JSON), "sha256": RELENC_JSON_SHA},
                       "relenc_npz": {"path": D.rel(RELENC_NPZ), "sha256": RELENC_NPZ_SHA},
                       "strings": {"relations": V, "ordered_pairs": m - V}, "dense": rj["dense"], "splade": rj["splade"],
                       "query_vectors": "the population rows of cd.query_embeddings (unit, _l1c_flath2.unit_queries) and of the "
                                        "splade query store (the FLAT inputs)",
                       "splade_zero_score_rows (all 9 relations score 0)": int((self.sS[:, :V].max(axis=1) == 0).sum()),
                       "seconds": self.seconds}

    def schedules(self, j, lex):
        """{arm: relation order} of row j given its lexical order."""
        out = {"LEX": list(lex)}
        for c in CH:
            ro = self.rel[c][j]
            out["XA_" + c] = list(lex) + [int(next(r for r in ro if int(r) not in lex))]
        return out


class E2Spec(object):
    def __init__(self, cd, pop, parts, Q):
        self.K = KB.KBResSpec(cd, pop, parts)
        K = self.K
        self.N, self.F, self.pop, self.Q = K.N, K.F, pop, Q
        N = self.N
        nq, ng = pop.nq, pop.ng_tot
        t0 = time.time()
        self.C = TT.NodeBlocks(cd)
        self.T = TY.Typed(self.C)
        self.typed_graph = UNI.is_typed_graph(self.T)
        assert self.typed_graph, "E2 is defined on a typed structural graph"
        self.names = self.T.names()
        assert len(self.names) == N
        self.idx = SE.build_name_index(self.names)
        self.W = TT.Walker(self.T, N, TT.HW)
        t_typed = time.time() - t0
        self.RS = RelScores(cd, pop, self.T.vocab)
        self.cells = list(K.cells)
        self.cellsU = [UNR] + self.cells
        self.POS = {(c, a): np.full(ng, -1, np.int64) for c in self.cellsU for a in ARMS}
        self.CONT = {c: np.zeros(ng, bool) for c in self.cells}
        self.CMASS = {c: np.zeros(nq, np.int64) for c in self.cells}
        self.CMASS[UNR] = np.full(nq, N, np.int64)
        self.GR = {a: np.full(ng, -1, np.int64) for a in CAND_ARMS}
        self.TD = {k: np.zeros(nq, np.int64) for k in ("lexical", "anchored", "gate", "k", "n_seeds", "topic_in_seeds", "fallback")}
        self.SEEDS = np.full((nq, 5), -1, np.int64)
        self.ORD = {a: [""] * nq for a in SCHED}
        self.NCT = {a: np.zeros(nq, np.int64) for a in CAND_ARMS}
        self.ROWS_T = {a: np.zeros(nq, np.int64) for a in SCHED}
        self.ROWS_OUT = {(c, a): np.zeros(nq, np.int64) for c in self.cells for a in SCHED}
        self.FETCH = {(c, a): np.zeros((nq, len(M_CURVE)), np.int64) for c in self.cells for a in CAND_ARMS}
        self.XSH = {(c, a): np.zeros((nq, len(M_CURVE)), np.int64) for c in self.cells for a in CAND_ARMS}
        self.n_walks = np.zeros(nq, np.int64)
        self.LAT = {"base (IR_L1 score, served order, router order)": [], "seeds + lexical schedule + encoder schedules": [],
                    "walks (distinct schedules + untyped)": [], "candidate lists + UNROUTED orders": []}
        for c in self.cells:
            self.LAT["%s ES contacted set + L1 order + arm orders" % c] = []
        self.record = {"l1_spec (the KBRES constructor's pins)": K.record, "cells": self.cells, "arms": list(ARMS),
                       "schedule_arms": list(SCHED), "channels": list(CH), "M_curve": list(M_CURVE), "K0": K0, "ACT": ACT,
                       "EPS": TT.EPS, "H": TT.HW, "gate": "G2 (typed graph AND anchored)", "relation_scores": self.RS.record,
                       "typed_graph": {"relation_vocabulary": list(self.T.vocab),
                                       "typed_edge_entries (undirected, both directions)": int(len(self.T.u)),
                                       "is_typed_graph": bool(self.typed_graph), "seconds": round(t_typed, 1)}}

    def row(self, j, of, od, os_, npos, frank, g, sl):
        K, N, F, W = self.K, self.N, self.F, self.W
        top = of[:ACT]
        nh = len(top)
        t0 = time.perf_counter()
        # ---- the carried-forward L1a (== _l3d_typed.TypedSpec.row == _l3d_hop.HopSpec.row == _l1d_kbres.KBResSpec.row)
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
        self.LAT["base (IR_L1 score, served order, router order)"].append(time.perf_counter() - t0)
        pL1 = positions(FO, g, N)
        self.POS[(UNR, "L1")][sl] = pL1
        # ---- seeds, lexical schedule, gate (the lane's functions, as TYPED v1), then the encoder schedules
        t0 = time.perf_counter()
        Q = self.Q[j]
        lex = SE.lexical_seeds(Q, self.idx, self.names, self.T)
        if lex:
            seeds = [int(s_) for s_ in lex[:5]]
        else:
            seeds = [int(od[0])] + ([int(os_[0])] if npos > 0 and int(os_[0]) != int(od[0]) else [])
        order, anchored = SE.schedule3(self.T, Q, seeds, key="wh")
        order = [int(r_) for r_ in order]
        gate = bool(anchored) and self.typed_graph
        x0 = np.zeros(N)
        x0[np.asarray(seeds, np.int64)] = 1.0 / float(len(seeds))
        td = self.TD
        td["lexical"][j], td["anchored"][j], td["gate"][j], td["k"][j], td["n_seeds"][j] = bool(lex), bool(anchored), gate, len(order), len(seeds)
        td["topic_in_seeds"][j] = int(K.TOPIC[j] >= 0 and int(K.TOPIC[j]) in seeds)
        td["fallback"][j] = int(not lex)
        self.SEEDS[j, :len(seeds)] = seeds
        sch = self.RS.schedules(j, order)
        for a, o in sch.items():
            self.ORD[a][j] = ",".join(self.T.vocab[r_] for r_ in o)
        self.LAT["seeds + lexical schedule + encoder schedules"].append(time.perf_counter() - t0)
        # ---- the walks: one per distinct schedule, unmasked (ROWS), and the untyped walk
        t0 = time.perf_counter()
        CT, RD = {}, {}
        if gate:
            _, uU, rduU = W.walk(x0, [])
            done = {}
            for a, o in sch.items():
                key = tuple(o)
                if key not in done:
                    aU, oU, rdU = W.walk(x0, o)
                    done[key] = (tcands(x0, aU, oU, uU, frank), rdU | rduU)
                CT[a], RD[a] = done[key]
            self.n_walks[j] = len(done) + 1
        else:
            for a in SCHED:
                CT[a], RD[a] = EMPTY, np.zeros(N, bool)
        self.LAT["walks (distinct schedules + untyped)"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        guard = bool(gate) and len(order) > 0          # the lexical schedule matched at least one relation (gold-free)
        for c in CH:
            CT["UA_" + c] = fuse(CT["LEX"], CT["XA_" + c], frank, N) if gate else EMPTY
            CT["UAG_" + c] = CT["UA_" + c] if guard else CT["LEX"]
            CT["UF_" + c] = fill(CT["LEX"], CT["XA_" + c]) if gate else EMPTY
            CT["UFG_" + c] = CT["UF_" + c] if guard else CT["LEX"]
        for a in CAND_ARMS:
            o = first_then(CT[a], FO, N)
            assert len(o) == N
            self.POS[(UNR, a)][sl] = positions(o, g, N)
            if not gate:
                assert (self.POS[(UNR, a)][sl] == pL1).all()
            self.GR[a][sl] = c2rank(CT[a], g, N)
            self.NCT[a][j] = len(CT[a])
        for a in SCHED:
            self.ROWS_T[a][j] = int(RD[a].sum())
        self.LAT["candidate lists + UNROUTED orders"].append(time.perf_counter() - t0)
        # ---- routed cells (ES on the native maps, B_P = kscale v1 CNTA alpha 0.75; == TYPED v1)
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
            for a in CAND_ARMS:
                o = first_then(CT[a], o1, N)
                assert len(o) >= cm
                self.POS[(c, a)][sl] = positions(o, g, N)
                if not gate:
                    assert (self.POS[(c, a)][sl] == pl1).all()
                ob_ = ~contn[o[:nmax]]
                for mi, M in enumerate(M_CURVE):
                    n_ = min(M, cm)
                    ob = ob_[:n_]
                    self.FETCH[(c, a)][j, mi] = int(ob.sum())
                    self.XSH[(c, a)][j, mi] = int(len(np.unique(hard[o[:n_][ob]])))
            for a in SCHED:
                self.ROWS_OUT[(c, a)][j] = int((RD[a] & ~contn).sum())
            self.LAT["%s ES contacted set + L1 order + arm orders" % c].append(time.perf_counter() - t0)

    # ---- tables
    def served(self, c, a, M):
        pop = self.pop
        cap = np.minimum(M, self.CMASS[c])[pop.row_of_gold]
        sv = self.POS[(c, a)] < cap
        cnt = np.add.reduceat(sv.astype(np.int64), pop.gptr[:-1])
        return cnt == pop.ngold, cnt > 0, cnt / pop.ngold.astype(np.float64), sv

    def miss_classes(self, c, a, M):
        """TYPED v1's classes (REPORT 40.10), first match: NO_GATE; NOT_REACHED (a missed gold is not a candidate) with a schedule
        shorter than the hop / at least as long; CT_TOO_LONG (every missed gold is a candidate, beyond the exposure).  A UA arm's
        schedule length is its XA schedule's (the longer of the two it fuses)."""
        pop = self.pop
        gate = self.TD["gate"].astype(bool)
        all_, _, _, sv = self.served(c, a, M)
        ingr = self.GR[a] >= 0
        miss_nc = np.add.reduceat((~sv & ~ingr).astype(np.int64), pop.gptr[:-1]) > 0
        src = a if a in SCHED else "XA_" + a.split("_", 1)[1]
        klen = np.array([len(s.split(",")) if s else 0 for s in self.ORD[src]], np.int64)
        hops = pop.hops
        rows = ~all_
        out = {"missed_rows": int(rows.sum())}
        cls = {"NO_GATE": rows & ~gate}
        rest = rows & gate
        cls["NOT_REACHED, schedule shorter than hop"] = rest & miss_nc & (klen < hops)
        cls["NOT_REACHED, schedule at least hop long"] = rest & miss_nc & (klen >= hops)
        cls["CT_TOO_LONG"] = rest & ~miss_nc
        HS = sorted(set(int(h) for h in hops))
        for k, m in cls.items():
            out[k] = {"rows": int(m.sum()), **{"hop%d" % h: int((m & (hops == h)).sum()) for h in HS}}
        return out, cls

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
        QS["gate"] = gate
        QS["no_gate"] = ~gate
        for h in HS:
            QS["hop%d|gate" % h] = (hops == h) & gate
        qts = sorted(set(self.K.QTYPE))
        QT = {qt: np.array([x == qt for x in self.K.QTYPE]) for qt in qts}
        res = {}
        for c in self.cellsU:
            res[c] = {}
            ref = {M: self.served(c, "L1", M) for M in M_CURVE}
            rlx = {M: self.served(c, "LEX", M) for M in M_CURVE}
            for a in ARMS:
                res[c][a] = {}
                for mi, M in enumerate(M_CURVE):
                    all_, any_, frac, sv = self.served(c, a, M)
                    nserv = np.minimum(M, self.CMASS[c])
                    e = {"ALL": int(all_.sum()), "ANY": int(any_.sum()), "FRAC_mean": D.q4(frac.mean()), "gold_nodes_served": int(sv.sum()),
                         "served_nodes_mean": round(float(nserv.mean()), 2),
                         "strata (ALL rows, rows)": {k: [int(all_[m].sum()), int(m.sum())] for k, m in QS.items()}}
                    if a != "L1":
                        e["paired_vs_L1 (gained = arm serves ALL gold, L1 does not)"] = dict(
                            D.paired(ref[M][0], all_), **{"strata": {k: D.paired(ref[M][0], all_, m) for k, m in QS.items() if k != "all"}})
                    if a not in ("L1", "LEX"):
                        e["paired_vs_LEX (gained = arm serves ALL gold, LEX does not)"] = dict(
                            D.paired(rlx[M][0], all_), **{"strata": {k: D.paired(rlx[M][0], all_, m) for k, m in QS.items() if k != "all"}})
                        e["gold_nodes gained / lost vs LEX"] = [int((sv & ~rlx[M][3]).sum()), int((~sv & rlx[M][3]).sum())]
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
        # ---- the schedules
        sched = {}
        lexo = self.ORD["LEX"]
        for a in SCHED:
            o = self.ORD[a]
            klen = np.array([len(s.split(",")) if s else 0 for s in o], np.int64)
            diff = np.array([o[i] != lexo[i] for i in range(nq)])
            same_set = np.array([set(o[i].split(",")) - {""} == set(lexo[i].split(",")) - {""} for i in range(nq)])
            e = {}
            for k, m in [("all", np.ones(nq, bool))] + [("hop%d" % h, hops == h) for h in HS]:
                mg = m & gate
                e[k] = {"gated_rows": int(mg.sum()), "differs_from_LEX (gated rows)": int((diff & mg).sum()),
                        "same_relation_set_as_LEX (gated rows)": int((same_set & mg).sum()),
                        "schedule_length (gated rows by k)": {str(x): int(((klen == x) & mg).sum()) for x in range(0, 4)}}
                cnt = {}
                for i in np.flatnonzero(mg):
                    cnt[o[i]] = cnt.get(o[i], 0) + 1
                e[k]["most frequent schedules (gated rows)"] = sorted(([k_, v_] for k_, v_ in cnt.items()), key=lambda kv: (-kv[1], kv[0]))[:12]
            if a.startswith("XT_") or a.startswith("XA_"):
                app = {}
                for i in np.flatnonzero(gate & diff):
                    r_ = o[i].split(",")[-1]
                    key = "hop%d" % hops[i]
                    app.setdefault(key, {})
                    app[key][r_] = app[key].get(r_, 0) + 1
                e["appended relation (gated rows where appended, per hop)"] = {k: sorted(v.items(), key=lambda kv: (-kv[1], kv[0])) for k, v in sorted(app.items())}
            sched[a] = e
        # ---- the misses, and the fate of LEX's NOT_REACHED rows
        misses = {}
        for c in self.cellsU:
            misses[c] = {}
            for a in CAND_ARMS:
                misses[c][a] = {str(M): self.miss_classes(c, a, M)[0] for M in MISS_M}
        _, clsL = self.miss_classes(NR_CELL, "LEX", NR_M)
        nr = clsL["NOT_REACHED, schedule shorter than hop"] | clsL["NOT_REACHED, schedule at least hop long"]
        nr_rows = np.flatnonzero(nr)
        fate = {"cell": NR_CELL, "B_N": NR_M, "rows": int(len(nr_rows)),
                "per arm (ALL-served among them)": {a: int(self.served(NR_CELL, a, NR_M)[0][nr_rows].sum()) for a in CAND_ARMS},
                "per row": [{"j": int(i), "hop": int(hops[i]), "qtype": self.K.QTYPE[i], "question": self.Q[i],
                             "schedules": {a: self.ORD[a][i] for a in SCHED},
                             "all_served": [a for a in CAND_ARMS if self.served(NR_CELL, a, NR_M)[0][i]]} for i in nr_rows]}
        cost = {}
        for a in CAND_ARMS:
            cost[a] = {"nCT (gated rows)": KB.AB.dist(self.NCT[a][gate]) if gate.any() else None}
            if a in SCHED:
                cost[a]["typed + untyped rows read (gated rows)"] = KB.AB.dist(self.ROWS_T[a][gate]) if gate.any() else None
                for c in self.cells:
                    cost[a]["rows read outside the contacted shards (%s; gated rows)" % c] = KB.AB.dist(self.ROWS_OUT[(c, a)][gate]) if gate.any() else None
        cost["walks per gated row (distinct schedules + untyped)"] = KB.AB.dist(self.n_walks[gate]) if gate.any() else None
        cand = {}
        for a, r in self.GR.items():
            e = {"golds_in_list": int((r >= 0).sum()), **{"rank < %d" % t: int(((r >= 0) & (r < t)).sum()) for t in (10, 100, 1000, 5000)}}
            e["per hop (golds in list, gold nodes)"] = {"hop%d" % h: [int(((r >= 0) & (hops[rog] == h)).sum()), int((hops[rog] == h).sum())] for h in HS}
            cand[a] = e
        lat = {k: D.ms_stats(v) for k, v in self.LAT.items()}
        arrays = {"HOPS": hops, "QTYPE": np.array(self.K.QTYPE), "m_curve": np.asarray(M_CURVE, np.int64), "cells": np.array(self.cellsU),
                  "SEEDS": self.SEEDS, "RS_D": self.RS.sD, "RS_S": self.RS.sS, "n_walks": self.n_walks}
        for c_ in CH:
            arrays["REL_ORDER__" + c_] = self.RS.rel[c_]
            arrays["PATH_BEST__" + c_] = self.RS.path[c_][:, 0]
        for a in SCHED:
            arrays["ORD__" + a] = np.array(self.ORD[a])
            arrays["ROWS_T__" + a] = self.ROWS_T[a]
        for k, v in self.TD.items():
            arrays["TD__" + k] = v
        for (c, a), v in self.POS.items():
            arrays["POS__%s__%s" % (c, a)] = v
        for c in self.cells:
            arrays["CONT__" + c] = self.CONT[c]
            arrays["CMASS__" + c] = self.CMASS[c]
            arrays["B__" + c] = self.K.B[c]
            for a in CAND_ARMS:
                arrays["FETCH__%s__%s" % (c, a)] = self.FETCH[(c, a)]
                arrays["XSH__%s__%s" % (c, a)] = self.XSH[(c, a)]
            for a in SCHED:
                arrays["ROWS_OUT__%s__%s" % (c, a)] = self.ROWS_OUT[(c, a)]
        for a in CAND_ARMS:
            arrays["GR__" + a] = self.GR[a]
            arrays["NCT__" + a] = self.NCT[a]
        return {"results (per cell, arm, B_N)": res, "schedules": sched, "misses (per cell, arm, B_N; first matching class)": misses,
                "LEX NOT_REACHED rows (REPORT 40.10) and what each arm serves": fate, "cost": cost,
                "candidate_ranks_of_gold_nodes": cand, "latency_ms_per_row": lat}, arrays


def typed_identity(S, POS_FLAT):
    """the L1, the seeds, the lexical schedule, the gate and LEX against the TYPED v1 record (write-once, sha pinned)."""
    assert D.sha_file(TYPED_NPZ) == TYPED_NPZ_SHA and D.sha_file(TYPED_JSON) == TYPED_JSON_SHA
    z = np.load(TYPED_NPZ)
    pop = S.pop
    nq, ng = pop.nq, pop.ng_tot
    assert (z["rows"][:nq] == pop.rows).all() and (z["gptr"][:nq + 1] == pop.gptr).all() and (z["pos_FLAT"][:ng] == POS_FLAT).all()
    assert (z["SEEDS"][:nq] == S.SEEDS).all(), "seeds differ from TYPED v1"
    assert (z["ORDERS"][:nq] == np.array(S.ORD["LEX"])).all(), "lexical schedules differ from TYPED v1"
    for k in ("lexical", "anchored", "gate", "k", "n_seeds", "topic_in_seeds", "fallback"):
        assert (z["TD__" + k][:nq] == S.TD[k]).all(), "TD %s differs from TYPED v1" % k
    assert (z["POS__%s__L1" % UNR][:ng] == S.POS[(UNR, "L1")]).all(), "UNROUTED L1 differs from TYPED v1"
    assert (z["POS__%s__T1" % UNR][:ng] == S.POS[(UNR, "LEX")]).all(), "UNROUTED LEX differs from TYPED v1 T1"
    assert (z["GR__UNR_CT"][:ng] == S.GR["LEX"]).all(), "CT(LEX) gold ranks differ from TYPED v1"
    for c in S.cells:
        assert (z["CONT__" + c][:ng] == S.CONT[c]).all() and (z["CMASS__" + c][:nq] == S.CMASS[c]).all()
        assert (z["B__" + c][:nq] == S.K.B[c]).all()
        assert (z["POS__%s__L1" % c][:ng] == S.POS[(c, "L1")]).all(), "routed L1 differs from TYPED v1 (%s)" % c
        assert (z["POS__%s__T1_ROWS" % c][:ng] == S.POS[(c, "LEX")]).all(), "routed LEX differs from TYPED v1 T1_ROWS (%s)" % c
    return ("on all %d rows the seeds, the lexical schedule and the gate diagnostics == TYPED v1; on all %d gold nodes the L1 "
            "positions (UNROUTED and routed), the contacted flags, the contacted mass and B_P == TYPED v1, LEX == TYPED v1 T1 "
            "(UNROUTED) and T1_ROWS (routed), and the gold ranks in CT(LEX) == TYPED v1 GR__UNR_CT (%s, sha %s)" % (
                nq, ng, D.rel(TYPED_NPZ), TYPED_NPZ_SHA[:16]))


def e2_identity(S, POS_FLAT):
    """LEX, XA_<c> and UA_<c> (schedules, served positions in every cell, candidate gold ranks, list lengths) against the E2 v1
    record (write-once, sha pinned), and the guard arms against the E2 v1 positions (the post-hoc derivation of REPORT 41.15)."""
    assert D.sha_file(E2V1_NPZ) == E2V1_NPZ_SHA and D.sha_file(E2V1_JSON) == E2V1_JSON_SHA
    z = np.load(E2V1_NPZ)
    pop = S.pop
    nq, ng = pop.nq, pop.ng_tot
    assert (z["rows"][:nq] == pop.rows).all() and (z["gptr"][:nq + 1] == pop.gptr).all() and (z["pos_FLAT"][:ng] == POS_FLAT).all()
    for a in SCHED:
        assert (z["ORD__" + a][:nq] == np.array(S.ORD[a])).all(), "schedule %s differs from E2 v1" % a
    rog = pop.row_of_gold
    k0 = (S.TD["k"] == 0)[rog]                     # gold nodes of rows whose lexical schedule is empty
    for c in S.cellsU:
        for a in ("LEX",) + tuple("XA_%s" % ch for ch in CH) + tuple("UA_%s" % ch for ch in CH):
            assert (z["POS__%s__%s" % (c, a)][:ng] == S.POS[(c, a)]).all(), "%s %s differs from E2 v1" % (c, a)
        for ch in CH:
            ua, lx = z["POS__%s__UA_%s" % (c, ch)][:ng], z["POS__%s__LEX" % c][:ng]
            assert (S.POS[(c, "UAG_" + ch)] == np.where(k0, lx, ua)).all(), "UAG_%s (%s) is not the E2 v1 derivation" % (ch, c)
    for a in ("LEX",) + tuple("XA_%s" % ch for ch in CH) + tuple("UA_%s" % ch for ch in CH):
        assert (z["GR__" + a][:ng] == S.GR[a]).all() and (z["NCT__" + a][:nq] == S.NCT[a]).all(), "%s candidates differ from E2 v1" % a
    for c in S.cells:
        for a in ("LEX",) + tuple("XA_%s" % ch for ch in CH) + tuple("UA_%s" % ch for ch in CH):
            assert (z["FETCH__%s__%s" % (c, a)][:nq] == S.FETCH[(c, a)]).all() and (z["XSH__%s__%s" % (c, a)][:nq] == S.XSH[(c, a)]).all()
    return ("on all %d rows the lexical and the XA schedules, and on all %d gold nodes the served positions of LEX, XA_<c> and UA_<c> in every "
            "cell, their candidate gold ranks, list lengths and fetch counts == E2 v1; UAG_<c> == E2 v1 UA_<c> where the lexical schedule is "
            "non-empty and LEX where it is empty (%s, sha %s)" % (nq, ng, D.rel(E2V1_NPZ), E2V1_NPZ_SHA[:16]))


def run(ds, tag):
    ROWS, SMOKE_OUT = R.argv_opts()
    here = os.path.abspath(__file__)
    shas = {"harness": D.sha_file(here)}
    for m in MODULES + tuple(sorted(LANE)):
        shas[m] = D.sha_file(os.path.join(D.HERE, m))
    assert shas["_l1d_kbres.py"] == HOP.KBRES_MOD_SHA, "the KBRES module changed"
    assert shas["_l3d_hop.py"] == TT.HOP_MOD_SHA, "the HOP module changed"
    assert shas["_l3d_typed.py"] == TYPED_MOD_SHA, "the TYPED module changed"
    for m, s in LANE.items():
        assert shas[m] == s and D.sha_file(os.path.join(SNAP, m)) == s, "the lane module %s differs from its code snapshot" % m
    host0 = D.host_state()
    t_all = time.time()
    outdir = SMOKE_OUT or OUT3
    os.makedirs(outdir, exist_ok=True)
    fp_out = os.path.join(outdir, "l3e2b_%s__%s.json" % (ds, tag))
    fz = fp_out.replace(".json", ".npz")
    assert not os.path.exists(fp_out) and not os.path.exists(fz), "write-once: %s exists" % fp_out
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    cells = D.CELLS[ds]
    pop = D.Population(cd, ROWS)
    nq, gptr = pop.nq, pop.gptr
    t_ = time.time()
    Q, qrec = TT.population_questions(cd, pop.rows)
    qrec["seconds"] = round(time.time() - t_, 1)
    parts = {c: D.Part(cd, D.TAG_OF[c]) for c in cells}
    t_ = time.time()
    S = E2Spec(cd, pop, parts, Q)
    S.record["seconds_build"] = round(time.time() - t_, 1)
    S.record["question_records"] = qrec
    log("RUN l3e2b %s %s: N %d, %d rows, %d gold nodes, cells %s, relations %s, arms %s" % (ds, tag, N, nq, pop.ng_tot, S.cellsU,
                                                                                          S.T.vocab, list(ARMS)))
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
            S.row(j, of, od, os_, npos_j, frank, g, sl)
        log("  %s rows %d / %d (%.0fs, RSS %.0f MB)" % (ds, j1, nq, time.time() - t_, D._rss_mb()))
    t_loop = round(time.time() - t_, 1)
    agree = {"dense_top100_overlap_mean": round(float(agd.mean()), 5), "splade_top100_overlap_mean": round(float(ags.mean()), 5)}
    assert agree["dense_top100_overlap_mean"] >= D.AGREE_MIN and agree["splade_top100_overlap_mean"] >= D.AGREE_MIN, agree
    f1 = os.path.join(D.OUT, "loc_%s__%s.npz" % (ds, R.V1_TAG))
    z1 = np.load(f1)
    assert (z1["rows"][:nq] == pop.rows).all()
    ng = int(z1["gptr"][nq])
    assert ng == pop.ng_tot and (z1["pos_FLAT"][:ng] == POS_FLAT).all(), "FLAT positions differ from the v1 record"
    v1check = "FLAT positions == %s (sha %s) on all %d gold nodes" % (D.rel(f1), D.sha_file(f1)[:16], ng)
    log("v1 check: " + v1check)
    idcheck = typed_identity(S, POS_FLAT)
    log("identity: " + idcheck)
    e2check = e2_identity(S, POS_FLAT)
    log("identity vs E2 v1: " + e2check)
    diag, arrays = S.finish()
    for c in S.cellsU:
        for a in ARMS:
            log("%-9s %-6s ALL %s" % (c, a, " ".join("%5d" % diag["results (per cell, arm, B_N)"][c][a][str(M)]["ALL"] for M in M_CURVE)))
    res = {"dataset": ds, "tag": tag, "stage": "L3 development (E2b: guarded, fill-ordered union)",
           "status": "DEVELOPMENT (descriptive; p-values descriptive; no rule selected)", "definitions": __doc__, "N": N, "n_rows": nq,
           "n_gold_nodes": pop.ng_tot, "population": pop.record, "structures": S.record, "v1_check": v1check, "typed_identity": idcheck, "e2_v1_identity": e2check,
           "served_list_agreement": agree, "diagnostics": diag,
           "latency_ms_flat": {"products_amortized": D.ms_stats(LAT["products_amortized"]), "flat_rrf": D.ms_stats(LAT["flat_rrf"])},
           "code": {k: {"path": "scratchpad/" + (os.path.basename(here) if k == "harness" else k), "sha256": v} for k, v in shas.items()},
           "inputs": {"e2_v1_json": {"path": D.rel(E2V1_JSON), "sha256": E2V1_JSON_SHA},
                      "e2_v1_npz": {"path": D.rel(E2V1_NPZ), "sha256": E2V1_NPZ_SHA},
                      "typed_v1_json": {"path": D.rel(TYPED_JSON), "sha256": TYPED_JSON_SHA},
                      "typed_v1_npz": {"path": D.rel(TYPED_NPZ), "sha256": TYPED_NPZ_SHA},
                      "relenc_json": {"path": D.rel(RELENC_JSON), "sha256": RELENC_JSON_SHA},
                      "relenc_npz": {"path": D.rel(RELENC_NPZ), "sha256": RELENC_NPZ_SHA},
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
    assert ds in ("metaqa",), "E2b is defined on the KB development dataset with a typed structural graph and a TYPED v1 record"
    run(ds, tag)


if __name__ == "__main__":
    main()
