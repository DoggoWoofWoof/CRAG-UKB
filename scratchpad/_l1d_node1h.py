"""L1 DEVELOPMENT -- NODELOC-1H: static one-hop NODE localization (L1a) and partition routing with two budgets (L1b)
(user rulings 2026-09-27).  Development numbers; any p-value is descriptive.

The L1 boundary (user): L1 = "one application of a fixed, query-independent sparse incidence/adjacency operator to query-time
semantic scores": L_f(q) = A_f^T x(q), x(q) = the FLAT score vector restricted to the semantic hits.  The adjacency is built
offline; exactly one propagation; no newly found node becomes a source; no beam, recursion, stopping rule, path search or PPR
(x -> A^T(A^T x) is traversal = L3).  This replaces the hierarchical-block step 3 and the alternative-incidence step 4 as
built (_l1d_hier.py, _l1d_act.py): no partition enters the node score.

L1a, node localization.  Hits H_q = FLAT_RRF[:200] (the pinned rrf_full; the activation depth of steps 1-2).  One universal
edge set on every dataset, E = STRUCT_out u STRUCT_in u KNN u NER, read-only from the canonical graph with the key construction
of _l1d_edgediag.build_families (self loops dropped, pairs deduplicated):
    STRUCT_out  s -> u for a structural edge s -> u        STRUCT_in  s -> u for a structural edge u -> s
    KNN, NER    the undirected semantic-kNN and shares-entity families (both orientations)
    L(u | q) = sum_{s in H_q} w(s) sum_f A_f(s, u) g_f(s) h_f(u)        (a pair in several families counts once per family)
The parameter-free ladder (1 / log(1 + d) in any base only rescales L, so no ranking depends on the base):
    L0   g = h = 1
    L1   g_f(s) = 1 / log2(1 + deg_f^+(s))       deg_f^+(s) = the edges of E_f leaving s (its row of A_f)
    L2   L1 and h_f(u) = 1 / log2(1 + deg_f(u))   deg_f(u) = the edges of E_f incident to u
Two hit weights:  FV  w(s) = fv(s), the FLAT_RRF score (the literal x(q))
                  IR  w(s) = 1 / FLAT rank of s (1-based; inverse rank -- the edge diagnostic found hits of rank 1-10 carry
                      6-30x the missing-gold enrichment of ranks 11-200, while fv falls only ~1.4x over that range)
Variants FV_L0 .. FV_L2 and IR_L0 .. IR_L2 are node scores and do not depend on the cell.  Per cell, <cell>__HARD is the step-1
hard LOC: the v1 identity anchor and the historical partition baseline.  Every variant is served through the step-1 arms
(_l1d_arms.py):
  - LOC@M serves the LOC order (every node with L > 0, by (-L, FLAT rank)), then FLAT fill.
  - FLAT+LOC@M is the RRF of the two ranks, f(u) = 1/(K0 + FLAT rank) + [u in LOC] / (K0 + LOC rank).
  Every arm serves exactly M nodes.

L1b, partition routing (user: "Partitions localize computation and data; nodes localize relevance"; two budgets, B_P =
partitions contacted, B_N = nodes returned).  A cell's frozen partition is used only as a shard map.  For each served order O
(FLAT, or the FLAT+LOC order of a rung) and each routing rule, the partitions are ranked, the first B_P are contacted, and
the first B_N nodes of O inside them are returned (fewer when the contacted partitions hold fewer than B_N nodes):
    S      R(P) = max over u in P of the served score (== partitions in order of first appearance in O)
    LMAX   R(P) = max over u in P of L(u)   (the user's Stage 3 with Agg = max; ties and L = 0 fall back to first appearance in O)
    LSUM   R(P) = sum over u in P of L(u)   (Agg = sum over the whole partition: the top-k sum with k = |P|)
The grid is B_P in (1, 2, 5, 10, 20, 50, 100) x B_N in M_CURVE.  Reported per cell: the contacted node mass, the returned
exposure, and the paired loss against the unrouted arm at the same B_N.  Two identities are asserted:
  - the fused ranks computed here equal the runner's FLAT+LOC positions, so contacting every partition IS the unrouted arm;
  - under S, a gold ranked before the (B_P+1)-th partition's first node keeps its rank.

Diagnostics:
  - paired comparisons: along the ladder, between the hit weights, against step-2 MEMACT and against step-1 HARD;
  - per gold node, the families that reach it from the top-10 and the top-200 hits;
  - the support (nodes with L > 0) outside FLAT@M, cross-checked against the edge diagnostic's SKN node set (same population,
    same hits; the support of E is the undirected STRUCT u KNN u NER 1-hop neighbourhood);
  - index bytes and entries per row.
No learned weight and no tuned constant (K0 = 60 and 200 are the frozen contract values).  No traversal: one sparse product
per query.

Usage: python scratchpad/_l1d_node1h.py RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]
       -> results/L1_DEV/node1h_<dataset>__<tag>.{json,npz} (write-once)
"""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_arms as R
import _l1d_edgediag as E

log = D.log
K0, ACT = D.K0, D.ACT
FAMS = ("STRUCT_out", "STRUCT_in", "KNN", "NER")
WEIGHTS = ("FV", "IR")
RUNGS = ("L0", "L1", "L2")
RUNG_NAMES = ["%s_%s" % (w, r) for w in WEIGHTS for r in RUNGS]
ROUTES = ("S", "LMAX", "LSUM")
BP_GRID = (1, 2, 5, 10, 20, 50, 100)
BPMAX = max(BP_GRID)
TOP_REACH = 10                                   # the top-10 hits of the edge diagnostic's node set
MS_SET = (100, 1000, 5000)
SUPP_COLS = (["support_top200", "support_top10"] + ["new_top200@%d" % M for M in MS_SET] + ["new_top10@%d" % M for M in MS_SET]
             + ["found_top200@%d" % M for M in MS_SET] + ["found_top10@%d" % M for M in MS_SET])
SENT = np.iinfo(np.int32).max                    # restricted rank of a gold whose partition is not contacted
SOFT_TAG = EDIAG_TAG = "v1"
E_PATH = os.path.join(D.HERE, "_l1d_edgediag.py")
assert E.MS == MS_SET and E.NH == ACT


def build_families(cd, N):
    """the four families of E as propagation CSRs (row s = the targets u of s, int32) with per-node g = 1/log2(1 + row length)
    and h = 1/log2(1 + incident edges); keys exactly as _l1d_edgediag.build_families."""
    N64 = np.int64(N)
    s, d, _, _ = cd.family("structural")
    SD = E.dkeys(s, d, N)
    del s, d
    ks, kd, _, _ = cd.family("knn")
    KNN = E.ukeys(ks, kd, N)
    del ks, kd
    ns_, nd_, _, _ = cd.family("ner")
    NER = E.ukeys(ns_, nd_, N)
    del ns_, nd_
    keys = {"STRUCT_out": ("dir", SD), "STRUCT_in": ("dir", np.unique((SD % N64) * N64 + SD // N64)), "KNN": ("und", KNN),
            "NER": ("und", NER)}
    F, rec = {}, {}
    for f in FAMS:
        kind, k = keys[f]
        xadj, adj = E.csr_directed(k, N) if kind == "dir" else E.csr_undirected(k, N)
        row = np.diff(xadj)
        inc = np.bincount(k // N64, minlength=N) + np.bincount(k % N64, minlength=N)
        assert kind == "dir" or (inc == row).all()
        g = np.zeros(N)
        g[row > 0] = 1.0 / np.log2(1.0 + row[row > 0])
        h = np.zeros(N)
        h[inc > 0] = 1.0 / np.log2(1.0 + inc[inc > 0])
        F[f] = {"xadj": xadj, "adj": adj, "g": g, "h": h}
        rec[f] = {"kind": kind, "entries": int(len(adj)), "pairs": int(len(k)), "row_length deg_f^+": D.stats(row),
                  "incident_degree deg_f (nodes with edges)": D.stats(inc[inc > 0]) if (inc > 0).any() else None,
                  "nodes_with_out_rows": int((row > 0).sum()),
                  "index_bytes (xadj + adj + g + h)": int(xadj.nbytes + adj.nbytes + g.nbytes + h.nbytes)}
        del row, inc
    del keys, SD, KNN, NER
    return F, rec


def restricted_ranks(prO, pg, prg):
    """0-based rank of each gold among the nodes of the first B_P partitions, for every B_P of BP_GRID (SENT: its partition is not
    contacted).  prO = the routing position of the partition of each node of the served order O, up to the deepest gold;
    pg = the golds' positions in O; prg = the routing positions of the golds' partitions."""
    prc = np.minimum(prO, BPMAX)
    out = np.full((len(pg), len(BP_GRID)), SENT, np.int64)
    cum = np.zeros(BPMAX + 1, np.int64)
    prev = 0
    for k in np.argsort(pg, kind="stable"):
        p = int(pg[k])
        if p >= prev:
            cum += np.bincount(prc[prev:p + 1], minlength=BPMAX + 1)
            prev = p + 1
        cc = np.cumsum(cum)
        for bi, BP in enumerate(BP_GRID):
            if prg[k] < BP:
                out[k, bi] = cc[BP - 1] - 1
    return out


class Node1HSpec(object):
    def __init__(self, cd, pop, parts):
        self.ds, self.parts, self.cells = cd.name, parts, list(parts)
        N = self.N = int(cd.n_nodes)
        self.sha_edgediag = D.sha_file(E_PATH)
        self.F, famrec = build_families(cd, N)
        self.names = list(RUNG_NAMES) + ["%s__HARD" % c for c in self.cells]
        self.hard_ref = {n_: None for n_ in self.names}
        self.v1_hard = {c: "%s__HARD" % c for c in self.cells}
        self.lat = {n_: [] for n_ in self.names}
        nq, ng = pop.nq, pop.ng_tot
        self.ORD = ["FLAT"] + list(RUNG_NAMES)
        self.CFG = [("FLAT", "S")] + [(n_, rt) for n_ in RUNG_NAMES for rt in ROUTES]
        self.CI = {x: i for i, x in enumerate(self.CFG)}
        self.RR = {c: np.full((ng, len(self.CFG), len(BP_GRID)), SENT, np.int32) for c in self.cells}
        self.MASS = {c: np.zeros((nq, len(self.CFG), len(BP_GRID)), np.int64) for c in self.cells}
        self.PSTAR = {c: np.zeros((nq, len(self.ORD), len(D.M_CURVE)), np.int64) for c in self.cells}
        self.RLAT = {c: [[] for _ in self.CFG] for c in self.cells}
        self.FUSE_LAT = {n_: [] for n_ in RUNG_NAMES}
        self.FRK = {n_: np.zeros(ng, np.int64) for n_ in RUNG_NAMES}
        self.REACH = np.zeros(ng, np.uint8)
        self.SUPP = np.zeros((nq, len(SUPP_COLS)), np.int64)
        self.ENT = np.zeros((nq, len(FAMS)), np.int64)
        self.record = {
            "edge_set": "E = STRUCT_out u STRUCT_in u KNN u NER (one universal set; no per-dataset family choice)",
            "hits": "FLAT_RRF[:%d]" % ACT, "families": famrec,
            "index_bytes_total": int(sum(v["index_bytes (xadj + adj + g + h)"] for v in famrec.values())),
            "hit_weights": {"FV": "fv(s) = 1/(K0 + dense rank) + 1/(K0 + SPLADE rank)", "IR": "1 / (FLAT rank + 1), 0-based rank"},
            "ladder": {"L0": "g = h = 1", "L1": "g = 1/log2(1 + deg_f^+(s))", "L2": "g as L1, h = 1/log2(1 + deg_f(u))"},
            "routing": {"rules": list(ROUTES), "B_P": list(BP_GRID), "B_N": list(D.M_CURVE), "configs": ["%s|%s" % x for x in self.CFG],
                        "cells": {c: {"partition": P.tag, "npart": P.npart, "partition_size": D.stats(P.sizes)} for c, P in parts.items()}},
            "imports": {"scratchpad/_l1d_edgediag.py": self.sha_edgediag}}

    def row(self, j, of, od, os_, npos, fv, frank, g, sl):
        N = self.N
        top = of[:ACT]
        nh = len(top)
        # ---- the one-hop entries of the hits over the four families (shared by every rung)
        t0 = time.perf_counter()
        U, H, GW, HW = [], [], [], []
        for fi, f in enumerate(FAMS):
            Fm = self.F[f]
            st = Fm["xadj"][top]
            hit, pos = E.entries(st, Fm["xadj"][top + 1] - st)
            u = Fm["adj"][pos].astype(np.int64)
            U.append(u)
            H.append(hit)
            GW.append(Fm["g"][top][hit])
            HW.append(Fm["h"][u])
            self.ENT[j, fi] = len(u)
        u, hit, gw, hw = np.concatenate(U), np.concatenate(H), np.concatenate(GW), np.concatenate(HW)
        t_ev = time.perf_counter() - t0
        W = {"FV": fv[top], "IR": 1.0 / np.arange(1.0, nh + 1.0)}
        out, Ls = {}, {}
        for w in WEIGHTS:
            for rg in RUNGS:
                t0 = time.perf_counter()
                x = W[w][hit]
                if rg != "L0":
                    x = x * gw
                if rg == "L2":
                    x = x * hw
                L = np.bincount(u, weights=x, minlength=N)
                n_ = "%s_%s" % (w, rg)
                out[n_] = D.order_from_score(L, frank)
                self.lat[n_].append(t_ev + time.perf_counter() - t0)
                Ls[n_] = L
        # ---- support and per-gold family reach (diagnostics; every rung has the same support)
        sup = np.zeros(N, bool)
        sup[u] = True
        sup10 = np.zeros(N, bool)
        sup10[u[hit < TOP_REACH]] = True
        nsup = int(sup.sum())
        assert all(len(out[n_]) == nsup for n_ in RUNG_NAMES)
        gr = frank[g]
        self.SUPP[j] = ([nsup, int(sup10.sum())] + [int((sup & (frank >= M)).sum()) for M in MS_SET]
                        + [int((sup10 & (frank >= M)).sum()) for M in MS_SET] + [int((sup[g] & (gr >= M)).sum()) for M in MS_SET]
                        + [int((sup10[g] & (gr >= M)).sum()) for M in MS_SET])
        rm = np.zeros(len(g), np.uint8)
        mk = np.zeros(N, bool)
        for fi in range(len(FAMS)):
            mk[U[fi]] = True
            rm[mk[g]] |= np.uint8(1 << fi)
            mk[U[fi]] = False
            mk[U[fi][H[fi] < TOP_REACH]] = True
            rm[mk[g]] |= np.uint8(1 << (4 + fi))
            mk[U[fi]] = False
        self.REACH[sl] = rm
        # ---- the step-1 HARD LOC per cell (v1 identity anchor, historical partition baseline)
        for c in self.cells:
            P = self.parts[c]
            t0 = time.perf_counter()
            ub, A = D.activation(top, fv, P.hard)
            out[c + "__HARD"] = D.loc_order(ub, A, frank, P)
            self.lat[c + "__HARD"].append(time.perf_counter() - t0)
        # ---- L1b: partition routing of FLAT and of every rung's FLAT+LOC order
        for oi, n_ in enumerate(self.ORD):
            if n_ == "FLAT":
                FO, frk = of, frank
            else:
                t0 = time.perf_counter()
                Lord = out[n_]
                f = 1.0 / (K0 + frank.astype(np.float64))
                f[Lord] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
                FO = np.lexsort((frank, -f))
                frk = np.empty(N, np.int64)
                frk[FO] = np.arange(N)
                self.FRK[n_][sl] = frk[g]
                self.FUSE_LAT[n_].append(time.perf_counter() - t0)
                nz = np.flatnonzero(Ls[n_])
            pg = frk[g]
            lim = int(pg.max()) + 1
            for c in self.cells:
                P = self.parts[c]
                fa = np.minimum.reduceat(frk[P.order_nodes], P.ptr[:-1])        # first appearance of every partition in O
                fas = np.sort(fa)
                self.PSTAR[c][j, oi] = [int((fa < M).sum()) for M in D.M_CURVE]
                hO, hg = P.hard[FO[:lim]], P.hard[g]
                for rt in (("S",) if n_ == "FLAT" else ROUTES):
                    t1 = time.perf_counter()
                    ci = self.CI[(n_, rt)]
                    if rt == "S":
                        ro = np.argsort(fa, kind="stable")
                    else:
                        if rt == "LMAX":
                            agg = np.zeros(P.npart)
                            np.maximum.at(agg, P.hard[nz], Ls[n_][nz])
                        else:
                            agg = np.bincount(P.hard[nz], weights=Ls[n_][nz], minlength=P.npart)
                        ro = np.lexsort((fa, -agg))
                    pr = np.empty(P.npart, np.int64)
                    pr[ro] = np.arange(P.npart)
                    rr = restricted_ranks(pr[hO], pg, pr[hg])
                    if rt == "S":
                        for bi, BP in enumerate(BP_GRID):
                            m = pg < (fas[BP] if BP < P.npart else N)
                            assert (rr[m, bi] == pg[m]).all(), "S-routing identity (%s %s B_P %d)" % (c, n_, BP)
                    self.RR[c][sl, ci] = rr
                    ms = np.cumsum(P.sizes[ro])
                    self.MASS[c][j, ci] = [int(ms[min(BP, P.npart) - 1]) for BP in BP_GRID]
                    self.RLAT[c][ci].append(time.perf_counter() - t1)
        return out

    def finish(self, ctx):
        POS, POS_FLAT, gptr, ngold, pop, ST = ctx["POS"], ctx["POS_FLAT"], ctx["gptr"], ctx["ngold"], ctx["pop"], ctx["ST"]
        nq, ng = pop.nq, pop.ng_tot
        assert D.sha_file(E_PATH) == self.sha_edgediag, "_l1d_edgediag.py changed during the run"
        allv = lambda pos, M: D.per_query(pos, M, gptr, ngold)[0]
        diag, arrays = {}, {}
        # (0) the routed orders: the fused ranks computed here == the runner's FLAT+LOC positions (every partition contacted == unrouted)
        for n_ in RUNG_NAMES:
            assert (self.FRK[n_] == POS[n_]["FLAT+LOC"]).all(), "fused order differs from the FLAT+LOC arm (%s)" % n_
        diag["routing_identity"] = ("the FLAT+LOC order used for routing == the runner's FLAT+LOC positions for every rung on all %d gold "
                                    "nodes; S-routing keeps the rank of every gold before the (B_P+1)-th partition (asserted per row)" % ng)
        # (1) ladder and hit weight, paired on ALL gold (gained = the second serves ALL gold, the first does not)
        lad = {}
        for arm in ("LOC", "FLAT+LOC"):
            lad[arm] = {}
            for M in D.M_CURVE:
                e = {}
                for w in WEIGHTS:
                    for a, b in (("L0", "L1"), ("L1", "L2"), ("L0", "L2")):
                        e["%s_%s -> %s_%s" % (w, a, w, b)] = D.paired(allv(POS["%s_%s" % (w, a)][arm], M), allv(POS["%s_%s" % (w, b)][arm], M))
                for rg in RUNGS:
                    e["FV_%s -> IR_%s" % (rg, rg)] = D.paired(allv(POS["FV_" + rg][arm], M), allv(POS["IR_" + rg][arm], M))
                lad[arm][str(M)] = e
        diag["paired_ladder_and_weight"] = lad
        # (2) against step-2 MEMACT (soft record) and step-1 HARD, per cell
        fs = os.path.join(D.OUT, "soft_%s__%s.npz" % (self.ds, SOFT_TAG))
        vs = {}
        zs = None
        if os.path.exists(fs):
            zs = np.load(fs)
            assert (zs["rows"][:nq] == pop.rows).all() and (zs["gptr"][:nq + 1] == gptr).all()
            if "pos_FLAT" in zs.files:
                assert (zs["pos_FLAT"][:ng] == POS_FLAT).all()
            vs["soft_record"] = {"path": D.rel(fs), "sha256": D.sha_file(fs)}
        for c in self.cells:
            vs[c] = {}
            for arm, key in (("LOC", "pos_LOC__"), ("FLAT+LOC", "pos_FLATLOC__")):
                refs = [("HARD", POS[c + "__HARD"][arm])]
                if zs is not None:
                    refs.append(("MEMACT", zs[key + c + "__MEMACT"][:ng]))
                vs[c][arm] = {}
                for M in D.M_CURVE:
                    e = {}
                    for rn, rpos in refs:
                        ra = allv(rpos, M)
                        e[rn + "_ALL"] = D.q4(ra.mean())
                        for n_ in RUNG_NAMES:
                            e["%s -> %s" % (rn, n_)] = D.paired(ra, allv(POS[n_][arm], M))
                    vs[c][arm][str(M)] = e
        diag["paired_vs_step1_HARD_and_step2_MEMACT"] = vs
        # (3) L1b routing: per cell, config, B_P, B_N
        rout = {}
        for c in self.cells:
            rc = {}
            for ci, (n_, rt) in enumerate(self.CFG):
                ref_pos = POS_FLAT if n_ == "FLAT" else POS[n_]["FLAT+LOC"]
                e = {}
                for bi, BP in enumerate(BP_GRID):
                    mass = self.MASS[c][:, ci, bi]
                    eb = {"contacted_node_mass": D.stats(mass)}
                    for M in D.M_CURVE:
                        a_, y_, f_, _ = D.per_query(self.RR[c][:, ci, bi].astype(np.int64), M, gptr, ngold)
                        ra = allv(ref_pos, M)
                        eb[str(M)] = {"ALL": D.q4(a_.mean()), "ANY": D.q4(y_.mean()), "FRAC": D.q4(f_.mean()),
                                      "unrouted_ALL": D.q4(ra.mean()), "paired_vs_unrouted": D.paired(ra, a_),
                                      "returned_nodes_mean": round(float(np.minimum(M, mass).mean()), 1)}
                    e[str(BP)] = eb
                rc["%s|%s" % (n_, rt)] = e
            rc["partitions_touched_by_the_unrouted_top_M"] = {n_: {str(M): D.stats(self.PSTAR[c][:, oi, mi]) for mi, M in enumerate(D.M_CURVE)}
                                                               for oi, n_ in enumerate(self.ORD)}
            rc["routing_ms"] = {"%s|%s" % x: D.ms_stats(self.RLAT[c][ci]) for ci, x in enumerate(self.CFG)}
            rout[c] = rc
        diag["routing"] = rout
        diag["fused_order_ms"] = {n_: D.ms_stats(v) for n_, v in self.FUSE_LAT.items()}
        # (4) which families reach the gold (from the top-200 / top-10 hits)
        subsets = [("all_gold", np.ones(ng, bool))] + [("FLAT_missed@%d" % M, POS_FLAT >= M) for M in MS_SET]
        reach = {}
        for lab, m in subsets:
            e = {"gold_nodes": int(m.sum())}
            for scope, off in (("top200", 0), ("top10", 4)):
                bits = (self.REACH[m].astype(np.int64) >> off) & 15
                if not m.any():
                    e[scope] = None
                    continue
                e[scope] = {"any_family": D.q4((bits > 0).mean())}
                for fi, f in enumerate(FAMS):
                    e[scope][f] = {"reached": D.q4(((bits >> fi) & 1).mean()), "only_this_family": D.q4((bits == (1 << fi)).mean())}
            reach[lab] = e
        hop_m = ST.get("per_hop", {})
        reach["per_hop (top200, any family / per family)"] = {}
        for hk, qm in hop_m.items():
            gm = qm[pop.row_of_gold]
            for lab, m in subsets[:3]:
                mm = gm & m
                if mm.any():
                    bits = self.REACH[mm].astype(np.int64) & 15
                    reach["per_hop (top200, any family / per family)"]["%s|%s" % (hk, lab)] = dict(
                        [("gold_nodes", int(mm.sum())), ("any_family", D.q4((bits > 0).mean()))] +
                        [(f, D.q4(((bits >> fi) & 1).mean())) for fi, f in enumerate(FAMS)])
        diag["gold_family_reach"] = reach
        # (5) support vs the edge diagnostic's SKN node set (same population, same hits, same undirected 1-hop neighbourhood)
        cix = {k: i for i, k in enumerate(SUPP_COLS)}
        sup = {"support_per_query (top200 / top10)": [D.stats(self.SUPP[:, 0]), D.stats(self.SUPP[:, 1])],
               "entries_per_query": {f: D.stats(self.ENT[:, fi]) for fi, f in enumerate(FAMS)}}
        fe = os.path.join(D.OUT, "edgediag_%s__%s.json" % (self.ds, EDIAG_TAG))
        if not os.path.exists(fe):
            sup["edge_diagnostic_crosscheck"] = "no edge-diagnostic record for this dataset"
        else:
            Rd = json.load(open(fe, encoding="utf-8"))
            if Rd["n_rows"] != nq or Rd["population"]["query_ids_sha256"] != pop.record["query_ids_sha256"]:
                sup["edge_diagnostic_crosscheck"] = "skipped: the record covers %d rows, this run %d" % (Rd["n_rows"], nq)
            else:
                chk, ok = {}, True
                for top in ("top200", "top10"):
                    for M in MS_SET:
                        rec = Rd["results"]["node_set"]["SKN"][top]["ALL"][str(M)]
                        new = int(self.SUPP[:, cix["new_%s@%d" % (top, M)]].sum())
                        found = int(self.SUPP[:, cix["found_%s@%d" % (top, M)]].sum())
                        miss = int((POS_FLAT >= M).sum())
                        good = (abs(new - rec["new_nodes_per_query"] * nq) <= 0.5 and miss == rec["missing_gold_nodes"]      # 5-decimal ratios
                                and (rec["recall_missing_gold_nodes"] is None or abs(found - rec["recall_missing_gold_nodes"] * miss) <= 0.5))
                        ok = ok and good
                        chk["%s@%d" % (top, M)] = {"new_nodes_per_query": [round(new / float(nq), 5), rec["new_nodes_per_query"]],
                                                   "missing_gold_nodes": [miss, rec["missing_gold_nodes"]],
                                                   "found": [found, rec["recall_missing_gold_nodes"]], "match": bool(good)}
                sup["edge_diagnostic_crosscheck"] = {"record": {"path": D.rel(fe), "sha256": D.sha_file(fe)}, "all_match": bool(ok), "cells": chk}
                log("edge-diagnostic SKN node-set cross-check: %s" % ("MATCH" if ok else "MISMATCH " + json.dumps(chk)))
        diag["support"] = sup
        arrays["reach"] = self.REACH
        arrays["supp"] = self.SUPP
        arrays["entries"] = self.ENT
        arrays["route_cfg"] = np.array(["%s|%s" % x for x in self.CFG])
        arrays["route_bp"] = np.array(BP_GRID, np.int64)
        for c in self.cells:
            arrays["route_rr__" + c] = self.RR[c]
            arrays["route_mass__" + c] = self.MASS[c]
            arrays["route_pstar__" + c] = self.PSTAR[c]
        return diag, arrays


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    assert mode == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]"
    R.run(sys.argv[2], sys.argv[3], "node1h", "L1_DEVELOPMENT_NODELOC_1H", __file__, Node1HSpec, __doc__)


if __name__ == "__main__":
    main()
