"""L1 DEVELOPMENT step 2 -- soft memberships (user ruling 2026-09-26, iteration order: 1 hard node-level LOC, 2 soft memberships,
3 hierarchical regions, 4 alternative static incidence / activation).  Development numbers; any p-value is descriptive.

Every variant turns a frozen partition into a static NODE score L(u) >= 0; the LOC order is every node with L > 0 sorted by
(-L, FLAT rank); the arms are exactly those of step 1 (_l1d_loc.py): LOC@M (the LOC order, then FLAT fill) and FLAT+LOC@M
(f(u) = 1/(K0 + FLAT rank) + [u in LOC] / (K0 + LOC rank)), each serving exactly M nodes, compared with FLAT@M.
    w(v)      fv(v) for v in FLAT_RRF[:200], else 0  (the step-1 activation weights)
    N[u]      {u} + the SK neighbours of u  (SK = STRUCT u KNN, the edge families of the H4_SK hypergraph every partition here
              was cut from); deg(u) = |SK neighbours|
    pi_n(u,B) = |N[u] n B| / |N[u]|  (soft membership: the block distribution of u's closed neighbourhood; rows sum to 1)
Variants (per cell = one frozen partition)
    HARD      L(u) = A_h(P(u)),               A_h(B) = sum of w over B                       (step 1; asserted == the v1 record)
    MEMACT    L(u) = A_m(P(u)),               A_m(B) = sum_v w(v) [B in mem(v)], mem(v) = {P(v)} + {P(x) : x a DIRECTED STRUCT
              out-neighbour of v}  (the served router's static membership table, replay_cache.CanonicalInputs.mem, unnormalised
              and uncapped as served; here as a node score over the fused top-200 instead of per-channel top-100 block votes)
    NBR_HA    L(u) = sum_B pi_n(u,B) A_h(B)   soft node membership, hard activation
    NBR       L(u) = sum_B pi_n(u,B) A_n(B),  A_n(B) = sum_v w(v) pi_n(v,B)                 soft on both sides
    HALO      L(u) = max over B in halo(u) of A_h(B), halo(u) = {P(u)} + {P(x) : x SK-adjacent to u, deg(x) <= DEG_CAP}
              (overlapping core + halo membership; a hub neighbour -- deg > 300, the frozen expansion cap -- lends no halo)
Per dataset with two frozen partitions (metaqa: H4_SK + PHG_REPAIR1; squad: H4_SK + PHG)
    ENS       L(u) = A_h1(P1(u)) + A_h2(P2(u))  (the two partitions as one two-owner soft membership)
No learned weight, no tuned constant, no traversal at query time: pi_n and halo are precomputed static incidences.

Modes: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]  ->  results/L1_DEV/soft_<dataset>__<tag>.{json,npz}
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import _l1d_lib as D

log = D.log
HERE = D.HERE
REPO = D.REPO
OUT = D.OUT
K0, ACT, DEG_CAP, M_CURVE, MMAX = D.K0, D.ACT, D.DEG_CAP, D.M_CURVE, D.MMAX
ENS_OF = {"metaqa": ("metaqa", "metaqa_phg"), "squad": ("squad", "squad_phg")}
CELL_VARIANTS = ("HARD", "MEMACT", "NBR_HA", "NBR", "HALO")
V1_TAG = "v1"


def served_mem(cd, P):
    """the served router's membership table as a binary N x npart CSR: own block + blocks of the directed STRUCT out-neighbours
    (the rule of src/l1_canonical/replay_cache.CanonicalInputs, rebuilt here from the same adapter call, read-only)."""
    N = len(P.hard)
    xo, ao = cd.struct_csr(directed=True)
    deg_out = np.diff(xo)
    r = np.concatenate([np.arange(N, dtype=np.int64), np.repeat(np.arange(N, dtype=np.int64), deg_out)])
    p = np.concatenate([P.hard, P.hard[ao.astype(np.int64)]])
    keys = np.unique(r * np.int64(P.npart) + p)
    del r, p
    cd._csr.clear()
    Mm = sp.csr_matrix((np.ones(len(keys), np.float64), (keys // P.npart, keys % P.npart)), shape=(N, P.npart))
    assert Mm.max() == 1.0 and (np.diff(Mm.indptr) >= 1).all()
    return Mm

def build_memberships(P, xadj, adj, deg):
    """pi_n (N x npart CSR, rows sum to 1) and the binary halo (N x npart CSR, sorted, own block included)."""
    N = len(P.hard)
    hard = P.hard
    rows = np.concatenate([np.arange(N, dtype=np.int64), np.repeat(np.arange(N, dtype=np.int64), deg)])
    cols = np.concatenate([hard, hard[adj]])
    dat = np.concatenate([1.0 / (deg + 1.0), np.repeat(1.0 / (deg + 1.0), deg)])
    Pn = sp.csr_matrix((dat, (rows, cols)), shape=(N, P.npart))
    Pn.sum_duplicates()
    assert np.allclose(np.asarray(Pn.sum(axis=1)).ravel(), 1.0)
    keep = np.concatenate([np.ones(N, bool), deg[adj] <= DEG_CAP])
    Hh = sp.csr_matrix((np.ones(int(keep.sum()), np.float32), (rows[keep], cols[keep])), shape=(N, P.npart))
    Hh.sum_duplicates()
    Hh.sort_indices()
    assert (np.diff(Hh.indptr) >= 1).all()
    del rows, cols, dat, keep
    return Pn, Hh


def main():
    MODE = sys.argv[1] if len(sys.argv) > 1 else ""
    assert MODE == "RUN", "usage: RUN <dataset> <tag> [--rows=K --out=<dir outside the repository>]"
    ds, tag = sys.argv[2], sys.argv[3]
    ROWS = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--rows=")), None)
    SMOKE_OUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    assert (ROWS is None) == (SMOKE_OUT is None)
    assert SMOKE_OUT is None or not os.path.abspath(SMOKE_OUT).lower().startswith(os.path.abspath(REPO).lower())
    code_sha = D.sha_file(os.path.abspath(__file__))
    lib_sha = D.sha_file(os.path.join(HERE, "_l1d_lib.py"))
    host0 = D.host_state()
    t_all = time.time()
    fp_out = os.path.join(SMOKE_OUT or OUT, "soft_%s__%s.json" % (ds, tag))
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    cells = D.CELLS[ds]
    pop = D.Population(cd, ROWS)
    nq, gptr, ngold, ST = pop.nq, pop.gptr, pop.ngold, pop.ST
    parts = {c: D.Part(cd, D.TAG_OF[c]) for c in cells}
    t_ = time.time()
    xadj, adj, graph_rec = D.sk_csr(cd)
    deg = np.diff(xadj).astype(np.int64)
    MEM = {c: build_memberships(parts[c], xadj, adj, deg) + (served_mem(cd, parts[c]),) for c in cells}
    graph_rec.update({"N": N, "degree": D.stats(deg), "nodes_over_DEG_CAP": int((deg > DEG_CAP).sum()),
                      "seconds_build": round(time.time() - t_, 1),
                      "memberships": {c: {"pi_n_nnz": int(MEM[c][0].nnz), "halo_nnz": int(MEM[c][1].nnz),
                                          "halo_blocks_per_node": D.stats(np.diff(MEM[c][1].indptr)),
                                          "served_mem_nnz": int(MEM[c][2].nnz), "served_mem_blocks_per_node": D.stats(np.diff(MEM[c][2].indptr))}
                                      for c in cells}})
    del xadj, adj
    log("RUN soft %s %s: N %d, %d rows, %d gold nodes, cells %s; graph %s" % (ds, tag, N, nq, pop.ng_tot, cells, json.dumps(graph_rec)))
    VAR = [(c, v) for c in cells for v in CELL_VARIANTS] + ([("ENS", "ENS")] if ds in ENS_OF else [])
    vname = lambda c, v: v if c == "ENS" else "%s__%s" % (c, v)
    names = [vname(c, v) for c, v in VAR]
    POS_FLAT = np.zeros(pop.ng_tot, np.int64)
    POS = {n: {"LOC": np.zeros(pop.ng_tot, np.int64), "FLAT+LOC": np.zeros(pop.ng_tot, np.int64), "lpos": np.zeros(pop.ng_tot, np.int64)} for n in names}
    LLEN = {n: np.zeros(nq, np.int64) for n in names}
    LAT = {n: [] for n in names}
    LAT["products_amortized"], LAT["flat_rrf"] = [], []
    agd, ags = np.zeros(nq), np.zeros(nq)
    d200 = np.asarray(cd.dense_topk(ACT, pop.rows), np.int64)
    s200 = np.asarray(cd.splade_topk(ACT, pop.rows), np.int64)
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
            top = of[:ACT]
            wt = fv[top]
            Ah = {}
            for c in cells:
                P = parts[c]
                Pn, Hh, Mm = MEM[c]
                ub, A = D.activation(top, fv, P.hard)
                Ah[c] = np.zeros(P.npart, np.float64)
                Ah[c][ub] = A
                for v in CELL_VARIANTS:
                    t0 = time.perf_counter()
                    if v == "HARD":
                        Lord = D.loc_order(ub, A, frank, P)
                    elif v == "MEMACT":
                        Am = np.asarray(Mm[top].T @ wt).ravel()
                        Lord = D.order_from_score(Am[P.hard], frank)
                    elif v == "NBR_HA":
                        Lord = D.order_from_score(Pn @ Ah[c], frank)
                    elif v == "NBR":
                        An = np.asarray(Pn[top].T @ wt).ravel()
                        Lord = D.order_from_score(Pn @ An, frank)
                    else:
                        assert v == "HALO"
                        vals = Ah[c][Hh.indices]
                        Lord = D.order_from_score(np.maximum.reduceat(vals, Hh.indptr[:-1]), frank)
                    LAT[vname(c, v)].append(time.perf_counter() - t0)
                    n_ = vname(c, v)
                    pl, pf, lp = D.arm_positions(Lord, frank, g, N)
                    POS[n_]["LOC"][sl], POS[n_]["FLAT+LOC"][sl], POS[n_]["lpos"][sl] = pl, pf, lp
                    LLEN[n_][j] = len(Lord)
            if ds in ENS_OF:
                c1, c2 = ENS_OF[ds]
                t0 = time.perf_counter()
                Lord = D.order_from_score(Ah[c1][parts[c1].hard] + Ah[c2][parts[c2].hard], frank)
                LAT["ENS"].append(time.perf_counter() - t0)
                pl, pf, lp = D.arm_positions(Lord, frank, g, N)
                POS["ENS"]["LOC"][sl], POS["ENS"]["FLAT+LOC"][sl], POS["ENS"]["lpos"][sl] = pl, pf, lp
                LLEN["ENS"][j] = len(Lord)
        log("  %s rows %d / %d (%.0fs, RSS %.0f MB)" % (ds, j1, nq, time.time() - t_, D._rss_mb()))
    t_loop = round(time.time() - t_, 1)
    agree = {"dense_top100_overlap_mean": round(float(agd.mean()), 5), "splade_top100_overlap_mean": round(float(ags.mean()), 5)}
    assert agree["dense_top100_overlap_mean"] >= D.AGREE_MIN and agree["splade_top100_overlap_mean"] >= D.AGREE_MIN, agree
    # ---- the HARD variant and FLAT must reproduce the step-1 record exactly (same rows, same arithmetic)
    f1 = os.path.join(OUT, "loc_%s__%s.npz" % (ds, V1_TAG))
    v1check = "no v1 record"
    if os.path.exists(f1):
        z1 = np.load(f1)
        assert (z1["rows"][:nq] == pop.rows).all()
        ng = int(z1["gptr"][nq])
        assert ng == pop.ng_tot and (z1["pos_FLAT"][:ng] == POS_FLAT).all(), "FLAT positions differ from the v1 record"
        for c in cells:
            assert (z1["pos_LOC__" + c][:ng] == POS[vname(c, "HARD")]["LOC"]).all(), "HARD LOC differs from v1 (%s)" % c
            assert (z1["pos_FLATLOC__" + c][:ng] == POS[vname(c, "HARD")]["FLAT+LOC"]).all(), "HARD FLAT+LOC differs from v1 (%s)" % c
            assert (z1["lpos__" + c][:ng] == POS[vname(c, "HARD")]["lpos"]).all()
        v1check = "FLAT and every cell's HARD LOC / FLAT+LOC / LOC-order positions == %s (sha %s) on all %d gold nodes" % (
            D.rel(f1), D.sha_file(f1)[:16], ng)
    log("v1 check: " + v1check)
    # ---- scores
    maxrank_flat = np.maximum.reduceat(POS_FLAT, gptr[:-1])
    FL = {M: D.per_query(POS_FLAT, M, gptr, ngold) for M in M_CURVE}
    FLAT_RES = {str(M): dict(D.score_block(*FL[M][:3], FL[M][3]), strata=D.strata_block(ST, *FL[M][:3])) for M in M_CURVE}
    inv = POS_FLAT >= MMAX
    RES = {}
    for c, v in VAR:
        n_ = vname(c, v)
        hard_ref = vname(c, "HARD") if c != "ENS" else None
        arms = {}
        for arm in ("LOC", "FLAT+LOC"):
            arms[arm] = {}
            for M in M_CURVE:
                a_, y_, f_, sv = D.per_query(POS[n_][arm], M, gptr, ngold)
                e = D.score_block(a_, y_, f_, sv, ref=(FL[M][0], FL[M][3]))
                e["FLAT_budget_matching_this_ALL"] = D.flat_budget_matching(maxrank_flat, float(a_.mean()))
                if hard_ref and v != "HARD":
                    ha = D.per_query(POS[hard_ref][arm], M, gptr, ngold)[0]
                    e["paired_vs_HARD_same_arm (gained = this variant serves ALL gold, HARD does not)"] = D.paired(ha, a_)
                e["strata"] = D.strata_block(ST, a_, y_, f_, FL[M][0])
                miss = POS_FLAT >= M
                e["of_FLAT@M_missed_gold_nodes"] = {"n": int(miss.sum()), "arm_serves": int((miss & sv).sum()),
                                                    "silent_L0": int((miss & (POS[n_]["lpos"] < 0)).sum())}
                e["invisible_gold_nodes_served (FLAT rank >= %d)" % MMAX] = int((inv & sv).sum())
                arms[arm][str(M)] = e
        RES[n_] = {"cell": c, "variant": v, "partition": (D.TAG_OF[c] if c != "ENS" else [D.TAG_OF[x] for x in ENS_OF[ds]]),
                   "arms": arms, "loc_order_length": D.stats(LLEN[n_]), "gold_in_loc_order": D.q4((POS[n_]["lpos"] >= 0).mean()),
                   "latency_ms (score + LOC order)": D.ms_stats(LAT[n_])}
        log("%-24s ALL  LOC %s | FLAT+LOC %s" % (n_, " ".join("%.3f" % arms["LOC"][str(M)]["ALL"] for M in M_CURVE),
                                                 " ".join("%.3f" % arms["FLAT+LOC"][str(M)]["ALL"] for M in M_CURVE)))
    log("%-24s ALL  %s" % ("FLAT", " ".join("%.3f" % FLAT_RES[str(M)]["ALL"] for M in M_CURVE)))
    res = {"dataset": ds, "tag": tag, "mode": "L1_DEVELOPMENT_SOFT_MEMBERSHIP", "status": "DEVELOPMENT (not confirmatory; p-values descriptive)",
           "N": N, "n_rows": nq, "n_gold_nodes": pop.ng_tot, "population": pop.record, "cells": cells, "variants": names,
           "graph": graph_rec, "v1_check": v1check, "served_list_agreement": agree, "FLAT": FLAT_RES, "variants_result": RES,
           "invisible_gold_nodes (FLAT rank >= %d)" % MMAX: int(inv.sum()),
           "latency_ms_flat": {"products_amortized": D.ms_stats(LAT["products_amortized"]), "flat_rrf": D.ms_stats(LAT["flat_rrf"])},
           "code": {"path": D.rel(os.path.abspath(__file__)), "sha256": code_sha, "lib": {"path": "scratchpad/_l1d_lib.py", "sha256": lib_sha}},
           "pinned": D.PINNED, "pinned_repo": D.PINNED_REPO, "function_sources_sha256": D.SRC_SHA, "constants": D.CONSTANTS,
           "platform": D.platform_record(), "host_at_start": host0, "seconds_loop": t_loop, "_row_query_ids": pop.qids}
    arrays = {"rows": pop.rows, "gptr": gptr, "pos_FLAT": POS_FLAT}
    for n_ in names:
        arrays["pos_LOC__" + n_] = POS[n_]["LOC"]
        arrays["pos_FLATLOC__" + n_] = POS[n_]["FLAT+LOC"]
        arrays["lpos__" + n_] = POS[n_]["lpos"]
        arrays["loc_len__" + n_] = LLEN[n_]
    res["seconds"] = round(time.time() - t_all, 1)
    res["process_peak_rss_mb"] = D.peak_rss_mb()
    assert D.sha_file(os.path.abspath(__file__)) == code_sha and D.sha_file(os.path.join(HERE, "_l1d_lib.py")) == lib_sha, "code changed during the run"
    fz = fp_out.replace(".json", ".npz")
    assert not os.path.exists(fz)
    np.savez_compressed(fz, **arrays)
    res["npz"] = {"path": os.path.basename(fz), "sha256": D.sha_file(fz)}
    D.G.S.wj(fp_out, res)
    log("done (%.0fs, peak RSS %s MB) -> %s sha256 %s" % (res["seconds"], res["process_peak_rss_mb"], fp_out, D.sha_file(fp_out)[:12]))


if __name__ == "__main__":
    main()
