"""L3 DEVELOPMENT -- WebQSP KB-L3 TRANSFER: the frozen L1 + the typed-walk / E2 / E2b arms, run once on WebQSP's development rows.
(User ruling 2026-09-30, REPORT 43.6: "WebQSP KB-L3 transfer first, with a development population of all scoreable non-held-out WebQSP
dev rows ...; then MuSiQue text-L3".)  TRANSFER ONLY: no arm is added, tuned or selected; every rule, constant and function is the one
that ran on MetaQA (REPORT sections 37-42), imported from the pinned modules, and the population is results/L3_DEV/l3w_population_webqsp__v1.json
(the 786 split-A rows, _l3w_pop.py).  Nothing held out is read: no split-B row, no TEST row.  No encoder training, no LLM, no learned
weight, no tuned constant (K0 60, EPS 1e-6, H 3, ACT 200, alpha 0.75 are the frozen constants).

Why a new harness: every MetaQA L3 harness asserts ds == 'metaqa' and pins metaqa records; this file re-implements only the parts that must
change for a second KB, and keeps the arithmetic line for line:
  L1   IR_L1 (FLAT_RRF over dense + SPLADE, one-hop LOC over STRUCT_out u STRUCT_in u KNN u NER, RRF K0 60, ACT 200); ES shard ranking on the
       native PHG map (K 25928); B_P(q, K) = min(K, ceil(B100(q) (K / 100)^0.75)), B100(q) = KNEE_SDIV on the PHG K 100 map of the same
       partitioner (WebQSP has no kscale record: B100 is computed here by the kscale arithmetic, _l1d_kscale.mult_tables and _l1d_route.knee).
       L1 serves the first n(q, B_N) = min(B_N, contacted mass) nodes of O inside the contacted shards.
  H2   the untyped second STRUCT hop (_l3d_hop.prop), UNROUTED and PUSH (sources contacted, targets anywhere).
  typed  the lane's relation-typed scheduled frontier walk (_l3d_typed.Walker, H = 3, gate G2, EPS 1e-6) with the lane's lexical seeds and
       schedule; arms per E2 (LEX, XT_c, XA_c, P_c for c in D, S, DS) and the four E2b unions (UA_c, UAG_c, UF_c, UFG_c), each serving its
       candidate list first and then the cell's L1 order (ROWS mask: the walk reads any row; a served node outside the contacted shards is
       FETCHED, counted).  The relation strings are the 7,058 WebQSP labels only (RELENC v1): P_c is the channel's best label.
Cells: UNROUTED (every shard contacted) and PHG_k25928 (the one native routed cell WebQSP has).  Exposure B_N in M_CURVE (100 .. 5000).
The order of every arm is needed only to depth 5,000 (n <= B_N <= 5,000), so the UNROUTED order and the routed orders are the exact
prefixes of the pinned total order (partition + tie-break by FLAT rank, asserted against the pinned lexsort on the first row of each
batch); positions past the prefix read N ('never served').
Identities asserted: FLAT top-5,000 and the dense / SPLADE top-200 == results/L1_COVPART/flath2_G_webqsp.npz (the run that first used
split A); the dense / SPLADE top-100 agreement gate; the K 100 and K 25928 maps == their RUN.json shas; on a sample of rows the typed
and untyped walks == the pinned relwalk bit for bit at float32.  A regression mode reproduces the MetaQA E2b / TYPED v1 records
(positions of every shared arm on every gold node) with this same code, before the WebQSP run is trusted.

Reported per cell, arm and B_N: ALL / ANY / FRAC and the gold nodes served; strata (gold-count bucket, gate, lexical seed, schedule length);
McNemar gained / lost vs L1 and vs LEX (descriptive); fetched nodes and extra shards; rows read outside the contacted shards; schedule /
gate / seed diagnostics; the misses by class; latency per stage.  Every p-value is descriptive.

Usage: python scratchpad/_l3w_transfer.py CHECK metaqa <n_rows>              -> regression against l3e2b / l3typed (writes nothing)
       python scratchpad/_l3w_transfer.py RUN webqsp <tag> [--rows=K --out=<dir outside the repository>]
                                                                            -> results/L3_DEV/l3w_webqsp__<tag>.{json,npz} (write-once)
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
import _l1d_edgediag as E
import _l1d_route as RT
import _l1d_kscale as KS
import _l1d_kbres as KB
import _l3d_hop as HOP
import _l3d_typed as TT
import _l3d_e2 as E2
import _l3d_e2b as E2B
import _l1x90_typed as TY
import _l1x90_seeds as SE
import _l1x90_relwalk as RW
import _l1x90_universal as UNI

log = D.log
K0, ACT = D.K0, D.ACT
FAMS = KB.FAMS
M_CURVE = D.M_CURVE
MMAX = max(M_CURVE)
UNR = KB.UNR
OUT3 = HOP.OUT3
CH = E2.CH
EPS, HW = UNI.EPS, UNI.H
prop, clist, rrf_add = HOP.prop, HOP.clist, HOP.rrf_add
tcands, first_then = TT.tcands, TT.first_then
EMPTY = TT.EMPTY
fuse, fill = E2.fuse, E2B.fill
LANE, SNAP = TT.LANE, TT.SNAP
QT_NONE = ""
WCELL = "PHG_k25928"
PARTS_DIR = os.path.join(D.REPO, "results", "L1_HOST", "parts")
POP_JSON = os.path.join(OUT3, "l3w_population_webqsp__v1.json")
POP_JSON_SHA_PREFIX = "7e67c9b116b2921c"
FLAT_G = os.path.join(D.REPO, "results", "L1_COVPART", "flath2_G_webqsp.npz")
RELENC_WEB_JSON = os.path.join(OUT3, "relenc_webqsp__v1.json")
RELENC_WEB_JSON_SHA = None          # pinned by content below (its npz sha is inside the json)
RELENC_WEB_NPZ = os.path.join(OUT3, "relenc_webqsp__v1.npz")
RELENC_WEB_NPZ_SHA = "aaec8d03c0c15d2b2e8e8b4b30190660c52c233e8833a6d614184d55f2df8988"
K_REF, ALPHA_I = 100, KS.ALPHAS.index("0.75")
SCHED_W = ("LEX",) + tuple("%s_%s" % (k, c) for k in ("XT", "XA", "P") for c in CH)
SCHED_M = ("LEX",) + tuple("%s_%s" % (k, c) for k in ("XT", "XA") for c in CH)
UNIONS = E2B.UNIONS
ID_SAMPLE_STEP = 40


def arms_of(sched):
    return ("L1", "H2") + sched + ("UA_D", "UA_S", "UA_DS") + tuple(u for u in UNIONS if not u.startswith("UA_"))


# ------------------------------------------------------------------------------------------------- small exact helpers
def prefix_order(f, frank, k, mask=None):
    """the first k nodes (of `mask`, or of all) of the total order by (-f, FLAT rank): exactly np.lexsort((frank, -f))[:k]."""
    if mask is None:
        n = len(f)
        if k >= n:
            return np.lexsort((frank, -f))
        tau = np.partition(f, n - k)[n - k]
        c = np.flatnonzero(f >= tau)
    else:
        cand = np.flatnonzero(mask)
        n = len(cand)
        if k >= n:
            return cand[np.lexsort((frank[cand], -f[cand]))]
        fc = f[cand]
        tau = np.partition(fc, n - k)[n - k]
        c = cand[fc >= tau]
    return c[np.lexsort((frank[c], -f[c]))][:k]


class Pos(object):
    """positions(o, g, N) (HOP.positions) without an N-sized allocation per call: the 0-based position of each gold in o; N = absent."""

    def __init__(self, N):
        self.N = N
        self.rk = np.full(N, N, np.int64)

    def __call__(self, o, g):
        rk = self.rk
        rk[o] = np.arange(len(o), dtype=np.int64)
        out = rk[g].copy()
        rk[o] = self.N
        return out


class LRUWalker(TT.Walker):
    """the pinned Walker with a bounded cache: 7,058 relations give one 36 MB entry per distinct allowed set, and the pinned cache is never
    evicted.  The arithmetic is untouched (sel_of is the pinned function; only what is kept between calls changes)."""

    def __init__(self, T, N, H, cap=6):
        super(LRUWalker, self).__init__(T, N, H)
        self.cap = cap

    def sel_of(self, allowed):
        key = tuple(sorted(allowed))
        if key in self.cache:
            v = self.cache.pop(key)
            self.cache[key] = v
            return v
        v = super(LRUWalker, self).sel_of(allowed)
        while len(self.cache) > self.cap:
            self.cache.pop(next(iter(self.cache)))
        return v


class WPop(object):
    """the development population of a dataset without a hop field: rows, gold nodes (flattened with pointers)."""

    def __init__(self, cd, rows):
        self.rows = np.asarray(rows, np.int64)
        self.nq = len(self.rows)
        self.qids = [cd.query_ids[int(r)] for r in self.rows]
        self.golds = [np.asarray(g, np.int64) for g in cd.gold(self.rows)]
        self.ngold = np.array([len(g) for g in self.golds], np.int64)
        assert (self.ngold >= 1).all()
        self.gptr = np.zeros(self.nq + 1, np.int64)
        self.gptr[1:] = np.cumsum(self.ngold)
        self.ng_tot = int(self.gptr[-1])
        self.row_of_gold = np.repeat(np.arange(self.nq), self.ngold)
        self.hops = None


def topic_positions(cd, rows):
    """{row: topic node positions} parsed from the population rows' lines only (a diagnostic; never read by the system); TEST refused."""
    want = {}
    for r in rows:
        r = int(r)
        sp_ = next(s for s, (a, b) in cd.split_ranges.items() if a <= r < b)
        assert sp_ != "test", "refused: row %d is in the TEST split" % r
        want.setdefault(sp_, {})[r - cd.split_ranges[sp_][0]] = r
    out = {}
    for sp_, lines in want.items():
        last = max(lines)
        with io.open(os.path.join(cd.dir, "queries", "%s.jsonl" % sp_), encoding="utf-8") as f:
            for i, ln in enumerate(f):
                if i in lines:
                    d = json.loads(ln)
                    assert d["query_id"] == cd.query_ids[lines[i]]
                    out[lines[i]] = [int(t) for t in (d.get("topic_positions") or [])]
                if i >= last:
                    break
    assert len(out) == len(rows)
    return out


# ------------------------------------------------------------------------------------------------- the relation strings' scores
class RelScoresL(object):
    """E2's relation scores restricted to the label strings (kind 0): D, S and DS orders of the relation labels per row."""

    def __init__(self, cd, pop, vocab, json_path, npz_path, npz_sha, dataset):
        assert D.sha_file(npz_path) == npz_sha, "the RELENC npz changed"
        rj = json.load(open(json_path, encoding="utf-8"))
        assert rj["dataset"] == dataset and rj["relation_vocabulary"] == list(vocab), "relation vocabulary differs from RELENC"
        z = np.load(npz_path)
        V = len(vocab)
        kind = z["kind"].astype(np.int64)
        assert (kind[:V] == 0).all() and z["r1"][:V].astype(np.int64).tolist() == list(range(V))
        verb = [lbl.replace("_", " ").replace(".", " ") for lbl in vocab]
        assert rj["strings"][:V] in (verb, list(vocab)), "label strings differ from the verbalised vocabulary"
        t0 = time.time()
        Rd = z["dense"][:V].astype(np.float64)
        Rs = sp.csr_matrix((z["sp_data"], z["sp_indices"], z["sp_indptr"]), shape=tuple(int(x) for x in z["sp_shape"]))[:V]
        Qu = D.unit_queries(cd, pop.rows).astype(np.float64)
        Qs = cd.embeddings("splade", "queries").read(pop.rows).tocsr()
        assert Rd.shape == (V, Qu.shape[1]) and Rs.shape == (V, Qs.shape[1])
        sD = np.einsum("qd,rd->qr", Qu, Rd)
        sS = np.asarray((Qs.astype(np.float64) @ Rs.astype(np.float64).T).todense())
        assert np.isfinite(sD).all() and np.isfinite(sS).all() and (sS >= 0).all()
        self.V, self.rel = V, E2.channel_orders(sD, sS)
        self.record = {"relenc_json": {"path": D.rel(json_path), "sha256": D.sha_file(json_path)},
                       "relenc_npz": {"path": D.rel(npz_path), "sha256": npz_sha}, "strings": {"relations": V}, "dense": rj["dense"],
                       "splade": rj["splade"], "splade_zero_score_rows (all labels score 0)": int((sS.max(axis=1) == 0).sum()),
                       "dense_score_ties_at_top (rows)": int(sum(1 for i in range(len(sD)) if (sD[i] == sD[i].max()).sum() > 1)),
                       "seconds": round(time.time() - t0, 2)}

    def schedules(self, j, lex, with_p):
        out = {"LEX": list(lex)}
        for c in CH:
            ro = self.rel[c][j]
            top = int(ro[0])
            out["XT_" + c] = list(lex) + ([top] if top not in lex else [])
            out["XA_" + c] = list(lex) + [int(next(r for r in ro if int(r) not in lex))]
            if with_p:
                out["P_" + c] = [top]
        return out


# ------------------------------------------------------------------------------------------------- the L1 providers
class WebQSPL1(object):
    """the routed cell of WebQSP: native PHG map K 25928, B_P from B100 on the PHG K 100 map (kscale arithmetic, alpha 0.75)."""

    def __init__(self, cd, N):
        import _l1d_node1h as NH
        self.N = N
        t0 = time.time()
        self.F, self.famrec = NH.build_families(cd, N)
        self.t_fam = round(time.time() - t0, 1)
        self.maps = {}
        for K in (K_REF, 25928):
            stem = "webqsp__H4_SK_k%d__PHG_con" % K
            run = json.load(open(os.path.join(PARTS_DIR, stem + ".RUN.json"), encoding="utf-8"))
            p = os.path.join(D.REPO, run["output"]["file"])
            assert D.sha_file(p) == run["output"]["sha256"], "the K %d map differs from its RUN.json" % K
            hard = np.load(p).astype(np.int64)
            assert len(hard) == N and hard.min() >= 0 and int(hard.max()) + 1 == K
            assert run["STATUS"] == "OK" and run["balance"]["every_block_used"] and run["balance"]["within_ceil_1.03_N_over_K"]
            sizes = np.bincount(hard, minlength=K).astype(np.int64)
            assert (sizes >= 1).all()
            self.maps[K] = {"hard": hard, "npart": K, "sizes": sizes, "sha256": run["output"]["sha256"], "file": run["output"]["file"],
                            "run_json": D.rel(os.path.join(PARTS_DIR, stem + ".RUN.json")), "run_sha256": D.sha_file(os.path.join(PARTS_DIR, stem + ".RUN.json"))}
        tabs, self.tabrec = KS.mult_tables(25928)
        self.TAB = tabs[KS.ALPHAS[ALPHA_I]]
        assert KS.ALPHAS[ALPHA_I] == "0.75"
        self.cells = [WCELL]
        self.parts = {WCELL: type("P", (), self.maps[25928])}
        self.p100 = self.maps[K_REF]
        self.wseed = 1.0 / np.arange(1.0, ACT + 1.0)
        self.B, self.NOEV, self.B0 = {WCELL: {}}, {}, {}
        self.record = {"cells": {WCELL: {k: self.maps[25928][k] for k in ("file", "sha256", "run_json", "run_sha256")}},
                       "reference_map (B100)": {k: self.maps[K_REF][k] for k in ("file", "sha256", "run_json", "run_sha256")},
                       "rule": KS.RULE + " (alpha 0.75)", "multiplier_table": {k: v for k, v in self.tabrec.items() if k != "_tabs"},
                       "families": self.famrec, "seconds_families": self.t_fam}

    def b_count(self, c, j, hit, u, x, nh):
        """B_P for row j: KNEE_SDIV on the K 100 map (the _l1d_scale.ScaleSpec.row arithmetic), then the alpha table."""
        P = self.p100
        npart = P["npart"]
        Cm = np.bincount(hit * npart + P["hard"][u], weights=x, minlength=nh * npart).reshape(nh, npart)
        sdiv = self.wseed[:nh] @ (Cm > 0)
        kn_, npz_ = RT.knee(sdiv)
        noev = not npz_
        b0 = kn_ if npz_ else npart
        assert 1 <= b0 <= npart
        b = self.maps[25928]["npart"] if noev else int(self.TAB[b0])
        self.B[c][j], self.NOEV[j], self.B0[j] = b, noev, b0
        return b

    def lo_check(self, c, j, lo):
        pass


class MetaQAL1(object):
    """the regression provider: the pinned KBResSpec (carried-forward record, kscale v1 CNTA) of MetaQA."""

    def __init__(self, cd, pop, parts):
        self.K = KB.KBResSpec(cd, pop, parts)
        K = self.K
        self.N, self.F, self.cells, self.parts = K.N, K.F, list(K.cells), K.parts
        self.B = K.B
        self.record = {"kbres": K.record}
        self.TOPIC, self.QTYPE = K.TOPIC, K.QTYPE

    def b_count(self, c, j, hit, u, x, nh):
        return int(self.K.B[c][j])

    def lo_check(self, c, j, lo):
        assert lo == self.K.LOK[c][j], "recomputed ES LO differs (%s row %d)" % (c, j)


# ------------------------------------------------------------------------------------------------- the harness
class Spec(object):
    def __init__(self, cd, pop, Q, topics, L1, sched, with_p, RS):
        self.cd, self.pop, self.Q, self.topics, self.L1 = cd, pop, Q, topics, L1
        self.N = N = L1.N
        self.F, self.cells = L1.F, list(L1.cells)
        self.cellsU = [UNR] + self.cells
        self.sched, self.with_p, self.RS = sched, with_p, RS
        self.ARMS = arms_of(sched)
        self.CAND = tuple(a for a in self.ARMS if a not in ("L1", "H2"))
        nq, ng = pop.nq, pop.ng_tot
        t0 = time.time()
        self.C = TT.NodeBlocks(cd)
        self.T = TY.Typed(self.C)
        self.typed_graph = UNI.is_typed_graph(self.T)
        assert self.typed_graph
        self.names = self.T.names()
        assert len(self.names) == N
        self.idx = SE.build_name_index(self.names)
        self.W = LRUWalker(self.T, N, HW)
        self.t_typed = round(time.time() - t0, 1)
        self.posf = Pos(N)
        self.POS = {(c, a): np.full(ng, -1, np.int64) for c in self.cellsU for a in self.ARMS}
        self.CONT = {c: np.zeros(ng, bool) for c in self.cells}
        self.CMASS = {c: np.zeros(nq, np.int64) for c in self.cells}
        self.CMASS[UNR] = np.full(nq, N, np.int64)
        self.LO = {c: np.zeros(nq, np.int64) for c in self.cells}
        self.PRG = {c: np.zeros(ng, np.int64) for c in self.cells}
        self.BB = {c: np.zeros(nq, np.int64) for c in self.cells}
        self.GR = {a: np.full(ng, -1, np.int64) for a in self.CAND}
        self.FRANK = np.zeros(ng, np.int64)
        self.TD = {k: np.zeros(nq, np.int64) for k in ("lexical", "anchored", "gate", "k", "n_seeds", "topic_in_seeds", "topic_in_top200",
                                                       "fallback", "guard", "k_gt_H", "noev", "n_walks")}
        self.SEEDS = np.full((nq, 5), -1, np.int64)
        self.ORDH = {a: np.full((nq, 4), -1, np.int64) for a in self.sched}
        self.ORDL = {a: np.zeros(nq, np.int64) for a in self.sched}
        self.NCT = {a: np.zeros(nq, np.int64) for a in self.CAND}
        self.ROWS_T = {a: np.zeros(nq, np.int64) for a in self.sched}
        self.ROWS_OUT = {(c, a): np.zeros(nq, np.int64) for c in self.cells for a in self.sched}
        self.FETCH = {(c, a): np.zeros((nq, len(M_CURVE)), np.int64) for c in self.cells for a in self.CAND + ("H2",)}
        self.XSH = {(c, a): np.zeros((nq, len(M_CURVE)), np.int64) for c in self.cells for a in self.CAND + ("H2",)}
        self.NC2 = {c: np.zeros(nq, np.int64) for c in self.cellsU}
        self.LAT = {"base (IR_L1 score, order, router order)": [], "B100 + ES + contacted set": [], "H2": [],
                    "seeds + lexical schedule + encoder schedules": [], "walks (distinct schedules + untyped)": [],
                    "candidate lists + arm orders": [], "identity check (pinned relwalk x2)": []}
        self.n_identity = 0
        self.prefix_checks = 0
        self.record = {"l1_spec": L1.record, "cells": self.cells, "arms": list(self.ARMS), "schedule_arms": list(sched),
                       "channels": list(CH), "M_curve": list(M_CURVE), "K0": K0, "ACT": ACT, "EPS": EPS, "H": HW,
                       "gate": "G2 (typed graph AND anchored)", "relation_scores": RS.record,
                       "typed_graph": {"n_relations": len(self.T.vocab), "typed_edge_entries (undirected, both directions)": int(len(self.T.u)),
                                       "is_typed_graph": bool(self.typed_graph), "seconds": self.t_typed},
                       "walker": "pinned _l3d_typed.Walker with a bounded (6) cache; identity to the pinned relwalk on a row sample"}

    def row(self, j, of, od, os_, npos, frank, g, sl, d_top1=None):
        L1, N, F, W = self.L1, self.N, self.F, self.W
        POSF = self.posf
        top = of[:ACT]
        nh = len(top)
        t0 = time.perf_counter()
        # ---- L1a (== _l1d_kbres.KBResSpec.row == _l3d_hop.HopSpec.row == _l3d_e2.E2Spec.row)
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
        self.FRANK[sl] = frank[g]
        self.LAT["base (IR_L1 score, order, router order)"].append(time.perf_counter() - t0)
        # ---- seeds, lexical schedule, gate (the lane's functions), the encoder schedules
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
        tp = self.topics[j]
        td["topic_in_seeds"][j] = int(any(t in seeds for t in tp))
        td["topic_in_top200"][j] = int(any(t in set(top.tolist()) for t in tp))
        td["fallback"][j] = int(not lex)
        td["k_gt_H"][j] = int(len(order) > HW)
        self.SEEDS[j, :len(seeds)] = seeds
        sch = self.RS.schedules(j, order, self.with_p)
        for a, o in sch.items():
            self.ORDL[a][j] = len(o)
            self.ORDH[a][j, :min(4, len(o))] = o[:4]
        self.LAT["seeds + lexical schedule + encoder schedules"].append(time.perf_counter() - t0)
        # ---- the walks: one per distinct schedule (unmasked, ROWS) and the untyped walk
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
            td["n_walks"][j] = len(done) + 1
        else:
            for a in self.sched:
                CT[a], RD[a] = EMPTY, np.zeros(N, bool)
        self.LAT["walks (distinct schedules + untyped)"].append(time.perf_counter() - t0)
        if gate and j % ID_SAMPLE_STEP == 0 or (gate and j < 4):
            t0 = time.perf_counter()
            sd = np.full((1, 5), -1, np.int64)
            sd[0, :len(seeds)] = seeds
            aU, oU, _ = W.walk(x0, order)
            _, uU, _ = W.walk(x0, [])
            ref = RW.relwalk(self.C, self.T, [Q], sd, [order], H=HW, mix=False)
            refu = RW.relwalk(self.C, self.T, [Q], sd, [[]], H=HW, mix=False)
            assert (ref["seed"][0] == x0.astype(np.float32)).all(), "seed channel differs from the pinned relwalk (row %d)" % j
            assert (ref["ans"][0] == aU.astype(np.float32)).all(), "ans channel differs from the pinned relwalk (row %d)" % j
            assert (ref["oth"][0] == oU.astype(np.float32)).all(), "oth channel differs from the pinned relwalk (row %d)" % j
            assert (refu["oth"][0] == uU.astype(np.float32)).all(), "untyped channel differs from the pinned relwalk (row %d)" % j
            self.n_identity += 1
            self.LAT["identity check (pinned relwalk x2)"].append(time.perf_counter() - t0)
        # ---- candidate lists (E2's UA and E2b's UAG / UF / UFG)
        t0 = time.perf_counter()
        guard = bool(gate) and len(order) > 0
        td["guard"][j] = int(guard)
        for c in CH:
            CT["UA_" + c] = fuse(CT["LEX"], CT["XA_" + c], frank, N) if gate else EMPTY
            CT["UAG_" + c] = CT["UA_" + c] if guard else CT["LEX"]
            CT["UF_" + c] = fill(CT["LEX"], CT["XA_" + c]) if gate else EMPTY
            CT["UFG_" + c] = CT["UF_" + c] if guard else CT["LEX"]
        maxct = max(len(CT[a]) for a in self.CAND)
        # ---- UNROUTED: L1, H2, every candidate arm (prefix orders, exact)
        FOp = prefix_order(f, frank, min(N, MMAX + maxct + 1))
        if self.prefix_checks < 3 and N <= 3_000_000:
            assert (FOp == np.lexsort((frank, -f))[:len(FOp)]).all(), "prefix order differs from the pinned lexsort"
            self.prefix_checks += 1
        self.POS[(UNR, "L1")][sl] = POSF(FOp, g)
        t1 = time.perf_counter()
        y2, e2 = prop(F, ms, LS, N)
        m2 = (y2 > 0) & ~ev
        o2 = clist(m2, y2, frank)
        f2 = rrf_add(f, o2)
        self.POS[(UNR, "H2")][sl] = POSF(prefix_order(f2, frank, MMAX), g)
        self.NC2[UNR][j] = len(o2)
        self.LAT["H2"].append(time.perf_counter() - t1)
        pL1 = self.POS[(UNR, "L1")][sl]
        for a in self.CAND:
            o = first_then(CT[a], FOp, N)
            self.POS[(UNR, a)][sl] = POSF(o, g)
            if not gate:
                assert (self.POS[(UNR, a)][sl] == pL1).all()
            self.GR[a][sl] = POSF(CT[a], g)
            self.GR[a][sl] = np.where(self.GR[a][sl] >= N, -1, self.GR[a][sl])
            self.NCT[a][j] = len(CT[a])
        for a in self.sched:
            self.ROWS_T[a][j] = int(RD[a].sum())
        self.LAT["candidate lists + arm orders"].append(time.perf_counter() - t0)
        # ---- routed cells
        for c in self.cells:
            t0 = time.perf_counter()
            P = L1.parts[c]
            npart, hard = P.npart, P.hard
            b = int(L1.b_count(c, j, hit, u, x, nh))
            hq = hard[oq]
            up, first = np.unique(hq, return_index=True)
            fe = np.full(npart, len(oq), np.int64)
            fe[up] = first
            ro = np.lexsort((np.arange(npart), fe))
            pr = RT.inv(ro)
            assert 1 <= b <= npart
            prg = pr[hard[g]]
            lo = int(prg.max()) + 1
            L1.lo_check(c, j, lo)
            self.LO[c][j], self.PRG[c][sl], self.BB[c][j] = lo, prg, b
            self.LAT["B100 + ES + contacted set"].append(time.perf_counter() - t0)
            contn = pr[hard] < b
            cm = int(contn.sum())
            self.CMASS[c][j] = cm
            self.CONT[c][sl] = contn[g]
            nmax = min(MMAX, cm)
            o1 = prefix_order(f, frank, min(cm, MMAX + maxct + 1), mask=contn)
            pl1 = POSF(o1, g)
            self.POS[(c, "L1")][sl] = pl1
            t1 = time.perf_counter()
            src2 = ms[contn[ms]]
            y2c, _ = prop(F, src2, LS, N)
            m2P = (y2c > 0) & ~ev
            o2P = clist(m2P, y2c, frank)
            elP2 = contn | m2P
            oH2P = prefix_order(rrf_add(f, o2P), frank, min(int(elP2.sum()), MMAX), mask=elP2)
            self.POS[(c, "H2")][sl] = POSF(oH2P, g)
            self.NC2[c][j] = len(o2P)
            self.LAT["H2"].append(time.perf_counter() - t1)
            for a, o in [("H2", oH2P)] + [(a_, first_then(CT[a_], o1, N)) for a_ in self.CAND]:
                if a != "H2":
                    self.POS[(c, a)][sl] = POSF(o, g)
                    if not gate:
                        assert (self.POS[(c, a)][sl] == pl1).all()
                ob_ = ~contn[o[:nmax]]
                for mi, M in enumerate(M_CURVE):
                    n_ = min(M, cm)
                    ob = ob_[:n_]
                    self.FETCH[(c, a)][j, mi] = int(ob.sum())
                    self.XSH[(c, a)][j, mi] = int(len(np.unique(hard[o[:n_][ob]])))
            for a in self.sched:
                self.ROWS_OUT[(c, a)][j] = int((RD[a] & ~contn).sum())
            self.LAT["candidate lists + arm orders"].append(time.perf_counter() - t0)
        if hasattr(L1, "NOEV"):
            td["noev"][j] = int(L1.NOEV[j])

    # ---- tables
    def served(self, c, a, M):
        pop = self.pop
        cap = np.minimum(M, self.CMASS[c])[pop.row_of_gold]
        sv = self.POS[(c, a)] < cap
        cnt = np.add.reduceat(sv.astype(np.int64), pop.gptr[:-1])
        return cnt == pop.ngold, cnt > 0, cnt / pop.ngold.astype(np.float64), sv

    def strata(self):
        pop, td = self.pop, self.TD
        nq, ngold = pop.nq, pop.ngold
        gate = td["gate"].astype(bool)
        QS = {"all": np.ones(nq, bool)}
        for nm, lo, hi in D.NG_BUCKETS:
            QS["ng_" + nm] = (ngold >= lo) & (ngold <= hi)
        QS["gate"], QS["no_gate"] = gate, ~gate
        QS["lexical_seed"], QS["fallback_seed"] = td["lexical"].astype(bool), ~td["lexical"].astype(bool)
        k = td["k"]
        for nm, m in (("k0", k == 0), ("k1", k == 1), ("k2", k == 2), ("k3", k == 3), ("k4plus", k >= 4)):
            QS["gated_" + nm] = gate & m
        QS["topic_in_seeds"], QS["topic_not_in_seeds"] = td["topic_in_seeds"].astype(bool), ~td["topic_in_seeds"].astype(bool)
        return QS

    def miss_classes(self, c, a, M, QS):
        """the E2 classes: NO_GATE; NOT_REACHED (a missed gold is not a candidate) with a schedule shorter than H / at least H long;
        CT_TOO_LONG (every missed gold is a candidate, beyond the exposure).  (WebQSP has no hop label: the schedule is compared to H.)"""
        pop = self.pop
        gate = self.TD["gate"].astype(bool)
        all_, _, _, sv = self.served(c, a, M)
        ingr = self.GR[a] >= 0
        miss_nc = np.add.reduceat((~sv & ~ingr).astype(np.int64), pop.gptr[:-1]) > 0
        src = a if a in self.sched else "XA_" + a.split("_", 1)[1]
        klen = self.ORDL[src]
        rows = ~all_
        cls = {"NO_GATE": rows & ~gate}
        rest = rows & gate
        cls["NOT_REACHED, schedule shorter than H"] = rest & miss_nc & (klen < HW)
        cls["NOT_REACHED, schedule at least H long"] = rest & miss_nc & (klen >= HW)
        cls["CT_TOO_LONG"] = rest & ~miss_nc
        return {"missed_rows": int(rows.sum()), **{k: int(m.sum()) for k, m in cls.items()}}

    def finish(self):
        pop = self.pop
        nq = pop.nq
        QS = self.strata()
        res = {}
        for c in self.cellsU:
            res[c] = {}
            ref = {M: self.served(c, "L1", M) for M in M_CURVE}
            rlx = {M: self.served(c, "LEX", M) for M in M_CURVE}
            for a in self.ARMS:
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
                    if a not in ("L1", "LEX", "H2"):
                        e["paired_vs_LEX (gained = arm serves ALL gold, LEX does not)"] = dict(
                            D.paired(rlx[M][0], all_), **{"strata": {k: D.paired(rlx[M][0], all_, m) for k, m in QS.items() if k != "all"}})
                        e["gold_nodes gained / lost vs LEX"] = [int((sv & ~rlx[M][3]).sum()), int((~sv & rlx[M][3]).sum())]
                    if (c, a) in self.FETCH:
                        fe, xs = self.FETCH[(c, a)][:, mi], self.XSH[(c, a)][:, mi]
                        e["fetched_nodes_per_query (served outside the contacted shards)"] = {
                            "mean": round(float(fe.mean()), 2), "median": float(np.median(fe)), "p95": float(np.percentile(fe, 95)),
                            "max": int(fe.max()), "queries_with_any": int((fe > 0).sum()), "share_of_served": D.q4(fe.sum() / float(nserv.sum()))}
                        e["extra_shards_per_query"] = {"mean": round(float(xs.mean()), 2), "median": float(np.median(xs)),
                                                       "p95": float(np.percentile(xs, 95)), "max": int(xs.max())}
                    res[c][a][str(M)] = e
        # ---- the routed L1's reach: shards needed to hold every gold vs the fan-out
        reach = {}
        for c in self.cells:
            lo, bb = self.LO[c], self.BB[c]
            reach[c] = {"B_P": KB.AB.dist(bb), "LO (shards, in ES order, holding every gold)": KB.AB.dist(lo),
                        "rows with B_P < LO (lost to reach)": int((bb < lo).sum()), "rows with B_P >= LO": int((bb >= lo).sum()),
                        "contacted_mass": KB.AB.dist(self.CMASS[c]), "rows contacting every shard (no localised evidence)": int((bb >= self.L1.parts[c].npart).sum())}
        gate = self.TD["gate"].astype(bool)
        sched = {}
        for a in self.sched:
            klen = self.ORDL[a]
            cnt = {}
            for i in np.flatnonzero(gate):
                key = ",".join(str(int(t)) for t in self.ORDH[a][i] if t >= 0) + ("..." if klen[i] > 4 else "")
                cnt[key] = cnt.get(key, 0) + 1
            sched[a] = {"schedule_length (all rows by k)": {str(x): int((klen == x).sum()) for x in range(0, 8)},
                        "schedule_length_8_plus": int((klen >= 8).sum()), "schedule_length_dist": KB.AB.dist(klen),
                        "gated rows with k > H (the answer channel is empty within H steps)": int((gate & (klen > HW)).sum()),
                        "most frequent schedule heads (gated rows; first 4 relation ids)": sorted(([k_, v_] for k_, v_ in cnt.items()), key=lambda kv: (-kv[1], kv[0]))[:12]}
        diag_seeds = {k: int(v.sum()) for k, v in self.TD.items() if k not in ("k", "n_seeds", "n_walks")}
        diag_seeds["rows"] = nq
        misses = {c: {a: {str(M): self.miss_classes(c, a, M, QS) for M in (100, 1000, 5000)} for a in self.CAND} for c in self.cellsU}
        cost = {a: {"nCT (gated rows)": KB.AB.dist(self.NCT[a][gate]) if gate.any() else None} for a in self.CAND}
        for a in self.sched:
            cost[a]["typed + untyped rows read (gated rows)"] = KB.AB.dist(self.ROWS_T[a][gate]) if gate.any() else None
            for c in self.cells:
                cost[a]["rows read outside the contacted shards (%s; gated rows)" % c] = KB.AB.dist(self.ROWS_OUT[(c, a)][gate]) if gate.any() else None
        cost["walks per gated row (distinct schedules + untyped)"] = KB.AB.dist(self.TD["n_walks"][gate]) if gate.any() else None
        cost["H2 candidate list (C2) size"] = {c: KB.AB.dist(self.NC2[c]) for c in self.cellsU}
        cand = {}
        rog = pop.row_of_gold
        for a, r in self.GR.items():
            cand[a] = {"golds_in_list": int((r >= 0).sum()), "gold_nodes": int(len(r)),
                       **{"rank < %d" % t: int(((r >= 0) & (r < t)).sum()) for t in (10, 100, 1000, 5000)}}
        lat = {k: D.ms_stats(v) for k, v in self.LAT.items() if v}
        arrays = {"m_curve": np.asarray(M_CURVE, np.int64), "cells": np.array(self.cellsU), "SEEDS": self.SEEDS, "FRANK": self.FRANK,
                  "arms": np.array(self.ARMS)}
        for c_ in CH:
            arrays["REL_ORDER_TOP5__" + c_] = self.RS.rel[c_][:, :5]
        for a in self.sched:
            arrays["ORDH__" + a] = self.ORDH[a]
            arrays["ORDL__" + a] = self.ORDL[a]
            arrays["ROWS_T__" + a] = self.ROWS_T[a]
        for k, v in self.TD.items():
            arrays["TD__" + k] = v
        for (c, a), v in self.POS.items():
            arrays["POS__%s__%s" % (c, a)] = v
        for c in self.cells:
            arrays["CONT__" + c] = self.CONT[c]
            arrays["CMASS__" + c] = self.CMASS[c]
            arrays["B__" + c] = self.BB[c]
            arrays["LO__" + c] = self.LO[c]
            arrays["PRG__" + c] = self.PRG[c]
            for a in self.CAND + ("H2",):
                arrays["FETCH__%s__%s" % (c, a)] = self.FETCH[(c, a)]
                arrays["XSH__%s__%s" % (c, a)] = self.XSH[(c, a)]
            for a in self.sched:
                arrays["ROWS_OUT__%s__%s" % (c, a)] = self.ROWS_OUT[(c, a)]
        for a in self.CAND:
            arrays["GR__" + a] = self.GR[a]
            arrays["NCT__" + a] = self.NCT[a]
        return {"results (per cell, arm, B_N)": res, "routed L1 reach": reach, "schedules": sched, "seed / gate diagnostics (rows)": diag_seeds,
                "misses (per cell, arm, B_N; first matching class)": misses, "cost": cost, "candidate_ranks_of_gold_nodes": cand,
                "latency_ms_per_row": lat}, arrays


# ------------------------------------------------------------------------------------------------- the FLAT loop shared by both modes
def batches_mm(cd, rows_g, Qu, N, mm_dir):
    """_l1d_lib.batches with the two (CQ, N) product matrices in disk-backed memmaps when N is large (2 x 200 x 2.59M x 4 B = 4.1 GB
    of RAM otherwise).  Same batches (CQ rows), same chunks (BLOCK nodes), the same pinned product functions in the same order: every
    stored value is the pinned one; only where it lives changes.  The finiteness / non-negativity checks run per chunk."""
    if mm_dir is None:
        for b in D.batches(cd, rows_g, Qu, N):
            yield b
        return
    QsA = cd.embeddings("splade", "queries").read(rows_g).tocsr()
    Es = cd.embeddings("splade", "docs")
    assert int(Es.shard_size) == D.BLOCK and int(Es.n_rows) == N
    n = len(rows_g)
    os.makedirs(mm_dir, exist_ok=True)
    for j0 in range(0, n, D.CQ):
        j1 = min(n, j0 + D.CQ)
        t0 = time.perf_counter()
        pd_, ps_ = os.path.join(mm_dir, "sd_%d.npy" % j0), os.path.join(mm_dir, "ss_%d.npy" % j0)
        SD = np.lib.format.open_memmap(pd_, mode="w+", dtype=np.float32, shape=(j1 - j0, N))
        SS = np.lib.format.open_memmap(ps_, mode="w+", dtype=np.float32, shape=(j1 - j0, N))
        nnz = 0
        for a, b_, blk in D.chunk_iter(cd.node_embeddings):
            Ec = np.ascontiguousarray(np.asarray(blk, np.float32))
            Msp = Es.shard(a // D.BLOCK)
            assert Msp.shape[0] == b_ - a and int(Msp.shape[1]) == int(QsA.shape[1])
            nnz += int(Msp.nnz)
            pdc = D.dense_products(Qu, Ec, j0, j1)
            psc = D.splade_products(Msp, QsA, j0, j1)
            assert np.isfinite(pdc).all() and np.isfinite(psc).all() and (psc >= 0).all()
            SD[:, a:b_] = pdc
            SS[:, a:b_] = psc
            del Ec, Msp, pdc, psc
        SD.flush()
        SS.flush()
        yield j0, j1, SD, SS, time.perf_counter() - t0, nnz
        del SD, SS
        for p in (pd_, ps_):
            try:
                os.remove(p)
            except OSError:
                pass


def flat_loop(cd, pop, S, N, tag, flat_check=None, mm_dir=None, strict_flat=True):
    gptr = pop.gptr
    nq = pop.nq
    agd, ags = np.zeros(nq), np.zeros(nq)
    d200 = np.asarray(cd.dense_topk(D.ACT, pop.rows), np.int64)
    s200 = np.asarray(cd.splade_topk(D.ACT, pop.rows), np.int64)
    Qu = D.unit_queries(cd, pop.rows)
    POS_FLAT = np.zeros(pop.ng_tot, np.int64)
    LAT = {"products_amortized": [], "flat_rrf": []}
    ftop_eq, ftop_ne = 0, []
    t_ = time.time()
    for j0, j1, SD, SS, sec, _ in batches_mm(cd, pop.rows, Qu, N, mm_dir):
        LAT["products_amortized"] += [sec / float(j1 - j0)] * (j1 - j0)
        for i in range(j1 - j0):
            j = j0 + i
            t0 = time.perf_counter()
            of, od, os_, npos_j, fv, frank = D.flat_row(np.array(SD[i]), np.array(SS[i]))
            LAT["flat_rrf"].append(time.perf_counter() - t0)
            k_ = min(D.N_AGREE, npos_j)
            agd[j] = len(set(od[:D.N_AGREE].tolist()) & set(d200[j, :D.N_AGREE].tolist())) / float(D.N_AGREE)
            ags[j] = (len(set(os_[:k_].tolist()) & set(s200[j, :k_].tolist())) / float(k_)) if k_ else 1.0
            if flat_check is not None:
                ft = flat_check["flat_top"][j]
                same = bool((of[:len(ft)] == ft).all() and (od[:D.ACT] == flat_check["top200_dense"][j]).all()
                            and (os_[:D.ACT] == flat_check["top200_splade"][j]).all())
                if strict_flat:
                    assert same, "FLAT top-%d / top-200 differs from flath2_G_webqsp (row %d)" % (len(ft), j)
                ftop_eq += int(same)
                if not same:
                    ftop_ne.append(j)
            g = pop.golds[j]
            sl = slice(gptr[j], gptr[j + 1])
            POS_FLAT[sl] = frank[g]
            S.row(j, of, od, os_, npos_j, frank, g, sl)
        log("  %s rows %d / %d (%.0fs, RSS %.0f MB)" % (tag, j1, nq, time.time() - t_, D._rss_mb()))
    agree = {"dense_top100_overlap_mean": round(float(agd.mean()), 5), "splade_top100_overlap_mean": round(float(ags.mean()), 5)}
    assert agree["dense_top100_overlap_mean"] >= D.AGREE_MIN and agree["splade_top100_overlap_mean"] >= D.AGREE_MIN, agree
    if ftop_ne:
        log("FLAT differs from flath2_G_webqsp on %d rows (batch shape of this run is not the G run's): %s" % (len(ftop_ne), ftop_ne[:20]))
    return POS_FLAT, agree, LAT, ftop_eq, round(time.time() - t_, 1)


def module_pins():
    shas = {"harness": D.sha_file(os.path.abspath(__file__))}
    mods = TT.MODULES + ("_l3d_typed.py", "_l3d_e2.py", "_l3d_e2b.py", "_l1d_kscale.py", "_l1d_scale.py") + tuple(sorted(LANE))
    for m in mods:
        shas[m] = D.sha_file(os.path.join(D.HERE, m))
    assert shas["_l1d_kbres.py"] == HOP.KBRES_MOD_SHA, "the KBRES module changed"
    assert shas["_l3d_hop.py"] == TT.HOP_MOD_SHA, "the HOP module changed"
    assert shas["_l3d_typed.py"] == E2.TYPED_MOD_SHA, "the TYPED module changed"
    for m, s in LANE.items():
        assert shas[m] == s and D.sha_file(os.path.join(SNAP, m)) == s, "the lane module %s differs from its code snapshot" % m
    return shas, mods


# ------------------------------------------------------------------------------------------------- CHECK metaqa: the regression
def check_metaqa(n_rows):
    """this file's row arithmetic on MetaQA == the pinned E2b / TYPED v1 records, on every gold node of the first n_rows population rows."""
    shas, mods = module_pins()
    cd = D.AD.CanonicalDataset("metaqa")
    N = int(cd.n_nodes)
    pop = D.Population(cd, n_rows)
    Q, _ = TT.population_questions(cd, pop.rows)
    parts = {c: D.Part(cd, D.TAG_OF[c]) for c in D.CELLS["metaqa"]}
    L1 = MetaQAL1(cd, pop, parts)
    topics = [[int(L1.TOPIC[j])] if L1.TOPIC[j] >= 0 else [] for j in range(pop.nq)]
    RS = RelScoresL(cd, pop, TY.Typed(TT.NodeBlocks(cd)).vocab, E2.RELENC_JSON, E2.RELENC_NPZ, E2.RELENC_NPZ_SHA, "metaqa")
    S = Spec(cd, pop, Q, topics, L1, SCHED_M, False, RS)
    log("CHECK metaqa: %d rows, %d gold nodes, cells %s, arms %s" % (pop.nq, pop.ng_tot, S.cellsU, list(S.ARMS)))
    POS_FLAT, agree, LAT, _, t_loop = flat_loop(cd, pop, S, N, "metaqa-check")
    z2 = np.load(os.path.join(OUT3, "l3e2b_metaqa__v1.npz"))
    z1 = np.load(os.path.join(OUT3, "l3typed_metaqa__v1.npz"))
    ze = np.load(os.path.join(OUT3, "l3e2_metaqa__v1.npz"))
    ng = pop.ng_tot
    assert (z2["rows"][:pop.nq] == pop.rows).all() and (z2["pos_FLAT"][:ng] == POS_FLAT).all()
    n_cmp, bad = 0, []
    cl = lambda p: np.minimum(p, MMAX)
    for c in S.cellsU:
        for a in S.ARMS:
            if a == "H2":
                key1 = "POS__%s__%s" % (c, "H2" if c == UNR else "H2_PUSH")
                ref = z1[key1][:ng]
            else:
                key = "POS__%s__%s" % (c, a)
                ref = (z2 if key in z2.files else ze)[key][:ng]        # XT_<c> lives in the E2 v1 record only
            ok = (cl(ref) == cl(S.POS[(c, a)])).all()
            n_cmp += 1
            if not ok:
                bad.append("%s/%s" % (c, a))
    for c in S.cells:
        assert (z2["CMASS__" + c][:pop.nq] == S.CMASS[c]).all() and (z2["B__" + c][:pop.nq] == S.BB[c]).all()
        assert (z2["CONT__" + c][:ng] == S.CONT[c]).all()
    for k in ("lexical", "anchored", "gate", "k", "n_seeds", "fallback"):
        assert (z2["TD__" + k][:pop.nq] == S.TD[k]).all(), "TD %s" % k
    assert (z2["SEEDS"][:pop.nq] == S.SEEDS).all()
    assert (z2["n_walks"][:pop.nq] == S.TD["n_walks"]).all(), "walk counts differ from E2b v1"
    log("CHECK metaqa: %d (cell, arm) position arrays compared, %d differ %s; identity walks %d; prefix checks %d; agreement %s (%.0fs)" % (
        n_cmp, len(bad), bad, S.n_identity, S.prefix_checks, agree, t_loop))
    assert not bad, "regression FAILED on %s" % bad
    log("CHECK metaqa PASS: this file reproduces E2b / TYPED v1 positions (clipped at %d) on all %d gold nodes of %d rows" % (MMAX, ng, pop.nq))


# ------------------------------------------------------------------------------------------------- RUN webqsp
def run_webqsp(tag):
    ROWS, SMOKE_OUT = R.argv_opts()
    shas, mods = module_pins()
    host0 = D.host_state()
    t_all = time.time()
    outdir = SMOKE_OUT or OUT3
    os.makedirs(outdir, exist_ok=True)
    fp_out = os.path.join(outdir, "l3w_webqsp__%s.json" % tag)
    fz = fp_out.replace(".json", ".npz")
    assert not os.path.exists(fp_out) and not os.path.exists(fz), "write-once: %s exists" % fp_out
    assert D.sha_file(POP_JSON).startswith(POP_JSON_SHA_PREFIX), "the population record changed"
    prec = json.load(open(POP_JSON, encoding="utf-8"))
    cd = D.AD.CanonicalDataset("webqsp")
    N = int(cd.n_nodes)
    assert prec["dataset_pins"] == {"DATASET_json_RECORD_SHA256": cd.record_sha, "query_index.npz": D.sha_file(cd._query_index_path())}
    rows = prec["rows"][:ROWS] if ROWS else prec["rows"]
    pop = WPop(cd, rows)
    assert pop.qids == prec["query_ids"][:pop.nq]
    zg = np.load(FLAT_G)
    assert (zg["rows"][:pop.nq] == pop.rows).all()
    flat_check = {k: zg[k][:pop.nq] for k in ("flat_top", "top200_dense", "top200_splade")}
    Q, qrec = TT.population_questions(cd, pop.rows)
    topics_d = topic_positions(cd, pop.rows)
    topics = [topics_d[int(r)] for r in pop.rows]
    t_ = time.time()
    L1 = WebQSPL1(cd, N)
    tvocab = TY.Typed(TT.NodeBlocks(cd)).vocab
    RS = RelScoresL(cd, pop, tvocab, RELENC_WEB_JSON, RELENC_WEB_NPZ, RELENC_WEB_NPZ_SHA, "webqsp")
    S = Spec(cd, pop, Q, topics, L1, SCHED_W, True, RS)
    S.record["seconds_build"] = round(time.time() - t_, 1)
    S.record["question_records"] = qrec
    log("RUN l3w webqsp %s: N %d, %d rows, %d gold nodes, cells %s, %d relations, arms %s" % (tag, N, pop.nq, pop.ng_tot, S.cellsU, len(tvocab), list(S.ARMS)))
    strict = ROWS is None or ROWS >= D.CQ              # the G run's first batch is the first CQ split-A rows: same product shape
    mm_dir = os.path.join(os.environ.get("TEMP", outdir), "l3w", "mm_" + tag)
    POS_FLAT, agree, LAT, ftop_eq, t_loop = flat_loop(cd, pop, S, N, "webqsp", flat_check, mm_dir=mm_dir, strict_flat=strict)
    log("FLAT identity: top-%d and the dense / SPLADE top-200 of %d / %d rows == flath2_G_webqsp.npz (strict %s)" % (
        flat_check["flat_top"].shape[1], ftop_eq, pop.nq, strict))
    diag, arrays = S.finish()
    for c in S.cellsU:
        for a in S.ARMS:
            log("%-11s %-8s ALL %s" % (c, a, " ".join("%5d" % diag["results (per cell, arm, B_N)"][c][a][str(M)]["ALL"] for M in M_CURVE)))
    res = {"dataset": "webqsp", "tag": tag, "stage": "L3 development: WebQSP KB-L3 transfer (typed / E2 / E2b arms on the frozen L1)",
           "status": "DEVELOPMENT (transfer only; descriptive; every p-value descriptive; no rule selected; no arm added or tuned)",
           "definitions": __doc__, "N": N, "n_rows": pop.nq, "n_gold_nodes": pop.ng_tot,
           "population": {"path": D.rel(POP_JSON), "sha256": D.sha_file(POP_JSON), "query_ids_sha256": prec["query_ids_sha256"], "n": pop.nq,
                          "first_rows_only": ROWS},
           "structures": S.record, "flat_identity": {"file": D.rel(FLAT_G), "sha256": D.sha_file(FLAT_G), "rows_checked": ftop_eq,
                                                    "flat_top_len": int(flat_check["flat_top"].shape[1])},
           "walker_identity": {"rows_checked_against_pinned_relwalk (typed + untyped)": S.n_identity, "rule": "gated rows j < 4 and j %% %d == 0" % ID_SAMPLE_STEP},
           "prefix_order_identity": {"rows_checked_against_pinned_lexsort": S.prefix_checks},
           "served_list_agreement": agree, "diagnostics": diag,
           "latency_ms_flat": {"products_amortized": D.ms_stats(LAT["products_amortized"]), "flat_rrf": D.ms_stats(LAT["flat_rrf"])},
           "code": {k: {"path": "scratchpad/" + (os.path.basename(os.path.abspath(__file__)) if k == "harness" else k), "sha256": v} for k, v in shas.items()},
           "inputs": {"relenc_json": {"path": D.rel(RELENC_WEB_JSON), "sha256": D.sha_file(RELENC_WEB_JSON)},
                      "relenc_npz": {"path": D.rel(RELENC_WEB_NPZ), "sha256": RELENC_WEB_NPZ_SHA},
                      "lane_code_snapshot": {"path": D.rel(SNAP), "sha256": LANE}},
           "pinned": D.PINNED, "pinned_repo": D.PINNED_REPO, "constants": D.CONSTANTS, "platform": D.platform_record(),
           "host_at_start": host0, "seconds_loop": t_loop, "_row_query_ids": pop.qids}
    arrays.update({"rows": pop.rows, "gptr": pop.gptr, "pos_FLAT": POS_FLAT})
    res["seconds"] = round(time.time() - t_all, 1)
    res["process_peak_rss_mb"] = D.peak_rss_mb()
    for m in mods:
        assert D.sha_file(os.path.join(D.HERE, m)) == shas[m], "code changed during the run"
    assert D.sha_file(os.path.abspath(__file__)) == shas["harness"], "code changed during the run"
    np.savez_compressed(fz, **arrays)
    res["npz"] = {"path": os.path.basename(fz), "sha256": D.sha_file(fz)}
    D.G.S.wj(fp_out, res)
    log("done (%.0fs, peak RSS %s MB) -> %s sha256 %s" % (res["seconds"], res["process_peak_rss_mb"], fp_out, D.sha_file(fp_out)[:12]))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "CHECK":
        assert sys.argv[2] == "metaqa"
        check_metaqa(int(sys.argv[3]))
    elif mode == "RUN":
        assert sys.argv[2] == "webqsp", "usage: RUN webqsp <tag> [--rows=K --out=<dir outside the repository>]"
        run_webqsp(sys.argv[3])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
