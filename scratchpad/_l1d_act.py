"""L1 DEVELOPMENT step 4 -- alternative static incidence / activation (user ruling 2026-09-26, iteration order: 1 hard
node-level LOC, 2 soft memberships, 3 hierarchical regions, 4 alternative static incidence / activation).  Development
numbers; any p-value is descriptive.

Every variant turns the query's semantic hits into a block score through a static node -> block incidence, and serves it as a
NODE order through the step-1 arms (LOC@M: the LOC order, then FLAT fill; FLAT+LOC@M: f(u) = 1/(K0 + FLAT rank) +
[u in LOC] / (K0 + LOC rank)); every arm serves exactly M nodes.  The LOC order = the nodes of the blocks that receive
evidence, by (block score of P(u) descending, FLAT rank), so tied blocks interleave by FLAT rank (as in step 1).
Factorial per cell (one frozen partition):
  evidence   F    the fused FLAT_RRF[:200], w(v) = fv(v)                                        (step 1)
             C    per channel, the exhaustive dense top-100 and SPLADE top-100, w = 1/(K0 + r) at 0-based channel rank r;
                  block score = sum over the two channels of 1/(K0 + channel rank of the block)  (partition RRF; K = 100
                  per channel as in the served router)
  incidence  OWN  {P(v)}
             MEM  mem(v) = {P(v)} + {P(x): x a DIRECTED STRUCT out-neighbour of v}  (the served table, unnormalised, uncapped)
  aggregate  SUM  S(B) = sum of w over the hits whose incidence contains B
             MAX  M(B) = max of those w                                                          (block-max)
             RR   1/(K0 + rank of B by S) + 1/(K0 + rank of B by M)                             (the served rr(S) + rr(M))
  ranks are competition ranks (tied scores share the best rank), so equal evidence gives equal block scores.
Reference variants (block-contiguous, served tie handling):
  SERVED_MEM  the served router itself: TA.partition_ranking (rr(S) + rr(M) per channel) and TA.rrf_partitions over the
              two channels -- the replay caches' base_rank (mode CHECK asserts it reproduces their stored base_rank) --
              restricted to the blocks with evidence, each block served whole in that order (LOC@5000 ~ the served P50)
  SERVED_OWN  the same with the OWN incidence
Identities asserted: F_OWN_SUM == step-1 HARD (the v1 record); F_MEM_SUM == step-2 MEMACT (the soft record, when present).
No learned weight, no tuned constant (K0 = 60, 200, 100 are the frozen contract values), no traversal at query time.

Modes: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]  ->  results/L1_DEV/act_<dataset>__<tag>.{json,npz}
       CHECK <dataset>   SERVED_MEM on the first 50 rows of each cell's replay cache == its stored base_rank (no gold read)
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import _l1d_lib as D
import _l1d_arms as R

log = D.log
TA = D.TA
K0, ACT = D.K0, D.ACT
KCH = TA.K_LOCK                                   # 100, the served per-channel router depth
EVID, INC, AGG = ("F", "C"), ("OWN", "MEM"), ("SUM", "MAX", "RR")
VARIANTS = ["%s_%s_%s" % (e, i, a) for e in EVID for i in INC for a in AGG] + ["SERVED_OWN", "SERVED_MEM"]
SOFT_TAG = "v1"
assert KCH == 100


def served_mem(cd, P):
    """the served router's membership table as a binary N x npart CSR (== _l1d_soft.served_mem == replay_cache.CanonicalInputs.mem):
    own block + blocks of the directed STRUCT out-neighbours; indices sorted within each row."""
    N = len(P.hard)
    xo, ao = cd.struct_csr(directed=True)
    deg_out = np.diff(xo)
    r = np.concatenate([np.arange(N, dtype=np.int64), np.repeat(np.arange(N, dtype=np.int64), deg_out)])
    p = np.concatenate([P.hard, P.hard[ao.astype(np.int64)]])
    keys = np.unique(r * np.int64(P.npart) + p)
    del r, p
    cd._csr.clear()
    Mm = sp.csr_matrix((np.ones(len(keys), np.float64), (keys // P.npart, keys % P.npart)), shape=(N, P.npart))
    assert Mm.max() == 1.0 and (np.diff(Mm.indptr) >= 1).all() and Mm.has_sorted_indices
    return Mm


def expand(nodes, w, memp, memf):
    """(block, weight) pairs of the hits `nodes` (weights w) through the incidence CSR (memp, memf)."""
    a, b = memp[nodes], memp[nodes + 1]
    lens = (b - a).astype(np.int64)
    if not lens.sum():
        return np.zeros(0, np.int64), np.zeros(0)
    start = np.repeat(a.astype(np.int64) - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    return memf[start + np.arange(lens.sum())].astype(np.int64), np.repeat(w, lens)


def sum_max(blocks, wexp, npart):
    S = np.bincount(blocks, weights=wexp, minlength=npart)
    Mx = np.zeros(npart)
    np.maximum.at(Mx, blocks, wexp)
    return S, Mx


def crank(x):
    """0-based competition rank by descending x: the number of entries strictly greater (ties share the best rank)."""
    return np.searchsorted(np.sort(-x), -x, side="left")


def rrk(x):
    return 1.0 / (K0 + crank(x))


def block_order_to_L(border, voted, hard):
    """node score from a block order (block-contiguous serving): blocks with evidence, first block highest; 0 elsewhere."""
    bo = border[voted[border]]
    val = np.zeros(len(voted), np.float64)
    val[bo] = np.arange(len(bo), 0, -1, dtype=np.float64)
    return val[hard]


class ActSpec(object):
    def __init__(self, cd, pop, parts):
        self.ds, self.parts, self.cells = cd.name, parts, list(parts)
        self.inc = {}
        self.record = {"cells": {}}
        N = int(cd.n_nodes)
        for c, P in parts.items():
            Mm = served_mem(cd, P)
            own = (np.arange(N + 1, dtype=np.int64), P.hard.astype(np.int32))
            mem = (Mm.indptr.astype(np.int64), Mm.indices.astype(np.int32))
            self.inc[c] = {"OWN": own, "MEM": mem, "Mm": Mm}
            self.record["cells"][c] = {"partition": P.tag, "npart": P.npart, "served_mem_nnz": int(Mm.nnz),
                                       "served_mem_blocks_per_node": D.stats(np.diff(Mm.indptr))}
        self.names, self.hard_ref, self.v1_hard = [], {}, {}
        for c in self.cells:
            for v in VARIANTS:
                n_ = "%s__%s" % (c, v)
                self.names.append(n_)
                self.hard_ref[n_] = None if v == "F_OWN_SUM" else "%s__F_OWN_SUM" % c
            self.v1_hard[c] = "%s__F_OWN_SUM" % c
        self.lat = {n_: [] for n_ in self.names}
        self.NVOTED = {c: np.zeros((pop.nq, 4), np.int64) for c in self.cells}     # blocks with evidence: F_OWN, F_MEM, C_OWN, C_MEM

    def _put(self, out, name, L, frank, t0):
        out[name] = D.order_from_score(L, frank)
        self.lat[name].append(time.perf_counter() - t0)

    def row(self, j, of, od, os_, npos, fv, frank, g, sl):
        top = of[:ACT]
        wt = fv[top]
        lists = (np.asarray(od[:KCH], np.int64), np.asarray(os_[:KCH], np.int64))
        wch = [1.0 / (K0 + np.arange(len(l_), dtype=np.float64)) for l_ in lists]
        out = {}
        for c in self.cells:
            P = self.parts[c]
            npart, hard = P.npart, P.hard
            for ii, inc in enumerate(INC):
                memp, memf = self.inc[c][inc]
                pre = "%s__" % c
                # ---- evidence F: the fused top-200, w = fv
                t0 = time.perf_counter()
                if inc == "OWN":
                    ub, A = D.activation(top, fv, hard)
                    out[pre + "F_OWN_SUM"] = D.loc_order(ub, A, frank, P)                     # == step-1 HARD
                    self.lat[pre + "F_OWN_SUM"].append(time.perf_counter() - t0)
                    S = np.zeros(npart)
                    S[ub] = A
                else:
                    S = np.asarray(self.inc[c]["Mm"][top].T @ wt).ravel()                    # == step-2 MEMACT
                    self._put(out, pre + "F_MEM_SUM", S[hard], frank, t0)
                t0 = time.perf_counter()
                bl, we = expand(top, wt, memp, memf)
                _, Mx = sum_max(bl, we, npart)
                voted = S > 0
                assert ((Mx > 0) == voted).all()
                self._put(out, pre + "F_%s_MAX" % inc, Mx[hard], frank, t0)
                t0 = time.perf_counter()
                self._put(out, pre + "F_%s_RR" % inc, np.where(voted, rrk(S) + rrk(Mx), 0.0)[hard], frank, t0)
                self.NVOTED[c][j, ii] = int(voted.sum())
                # ---- evidence C: per-channel top-100, w = 1/(K0 + channel rank); partition RRF over the two channels
                t0 = time.perf_counter()
                SM = [sum_max(*expand(l_, w_, memp, memf), npart) for l_, w_ in zip(lists, wch)]
                votedc = (SM[0][0] > 0) | (SM[1][0] > 0)
                t_ev = time.perf_counter() - t0
                for agg in AGG:
                    t0 = time.perf_counter() - t_ev
                    if agg == "SUM":
                        val = rrk(SM[0][0]) + rrk(SM[1][0])
                    elif agg == "MAX":
                        val = rrk(SM[0][1]) + rrk(SM[1][1])
                    else:
                        val = rrk(rrk(SM[0][0]) + rrk(SM[0][1])) + rrk(rrk(SM[1][0]) + rrk(SM[1][1]))
                    self._put(out, pre + "C_%s_%s" % (inc, agg), np.where(votedc, val, 0.0)[hard], frank, t0)
                # ---- the served router (served functions, served tie handling), block-contiguous
                t0 = time.perf_counter() - t_ev
                PR = [TA.partition_ranking([l_], (memp, memf), npart) for l_ in lists]
                border = TA.rrf_partitions(PR, npart)[0]
                self._put(out, pre + "SERVED_%s" % inc, block_order_to_L(border, votedc, hard), frank, t0)
                self.NVOTED[c][j, 2 + ii] = int(votedc.sum())
        return out

    def finish(self, ctx):
        diag, arrays = {}, {}
        POS, pop = ctx["POS"], ctx["pop"]
        for c in self.cells:
            diag[c] = {"blocks_with_evidence_per_query": {k: D.stats(self.NVOTED[c][:, i]) for i, k in
                                                          enumerate(("F_OWN", "F_MEM", "C_OWN", "C_MEM"))}}
            arrays["nvoted__" + c] = self.NVOTED[c]
        # F_MEM_SUM must reproduce the step-2 MEMACT record (same rows, same arithmetic) when it exists
        fs = os.path.join(D.OUT, "soft_%s__%s.npz" % (self.ds, SOFT_TAG))
        chk = "no soft record"
        if os.path.exists(fs):
            zs = np.load(fs)
            nq = pop.nq
            assert (zs["rows"][:nq] == pop.rows).all()
            ng = int(zs["gptr"][nq])
            for c in self.cells:
                for arm, key in (("LOC", "pos_LOC__"), ("FLAT+LOC", "pos_FLATLOC__"), ("lpos", "lpos__")):
                    assert (zs[key + c + "__MEMACT"][:ng] == POS["%s__F_MEM_SUM" % c][arm]).all(), "F_MEM_SUM != soft MEMACT (%s %s)" % (c, arm)
            chk = "every cell's F_MEM_SUM LOC / FLAT+LOC / LOC-order positions == %s MEMACT (sha %s) on all %d gold nodes" % (
                D.rel(fs), D.sha_file(fs)[:16], ng)
        log("soft check: " + chk)
        diag["soft_MEMACT_check"] = chk
        return diag, arrays


def check(ds):
    """SERVED_MEM (served functions + this table) on the first 50 rows of each cell's replay cache == the stored base_rank."""
    cd = D.AD.CanonicalDataset(ds)
    out = {}
    for c in D.CELLS[ds]:
        P = D.Part(cd, D.TAG_OF[c])
        Mm = served_mem(cd, P)
        mem = (Mm.indptr.astype(np.int64), Mm.indices.astype(np.int32))
        z = np.load(D.G.X.CACHES[c])
        n = 50
        dK, sK = z["ret_dense"][:n, :KCH], z["ret_splade"][:n, :KCH]
        PR_d = TA.partition_ranking(list(dK), mem, P.npart)
        PR_s = TA.partition_ranking(list(sK), mem, P.npart)
        base = TA.rrf_partitions([PR_d, PR_s], P.npart)
        br = z["base_rank"][:n]
        eq = (base[:, :br.shape[1]] == br)
        out[c] = {"rows": n, "cache": D.rel(D.G.X.CACHES[c]), "identical_rows": int(eq.all(axis=1).sum()),
                  "top50_set_identical_rows": int(sum(set(base[i, :50].tolist()) == set(br[i, :50].tolist()) for i in range(n)))}
        log("CHECK %s: %s" % (c, out[c]))
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "CHECK":
        check(sys.argv[2])
        return
    assert mode == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>] | CHECK <dataset>"
    R.run(sys.argv[2], sys.argv[3], "act", "L1_DEVELOPMENT_ALTERNATIVE_ACTIVATION", __file__, ActSpec, __doc__)


if __name__ == "__main__":
    main()
