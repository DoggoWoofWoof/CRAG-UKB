"""Shared variant runner of the L1 DEVELOPMENT harnesses from step 3 on (_l1d_hier.py, _l1d_act.py; user ruling 2026-09-26:
development -> mechanism selection -> one final held-out evaluation; no H2 / PPR / traversal / learned weight in L1).

The per-row FLAT_RRF loop, the two arms of every variant, the step-1 reproduction check, the scoring and the write-once
records are the arithmetic of the inline loop of _l1d_soft.py (step 2), factored here so that a harness only defines its
static structures and its per-row node scores.  A harness passes make_spec(cd, pop, parts) -> an object with
    names        variant names, in report order
    hard_ref     {variant: the variant it is paired against in "paired_vs_HARD_same_arm", or None}
    v1_hard      {cell: the variant whose positions must equal that cell's step-1 HARD record (loc_<ds>__v1.npz)}
    record       JSON-able description of the static structures (built once, query-independent)
    lat          {variant: [seconds per row]}  (the harness times its own node score + LOC order)
    row(j, of, od, os_, npos, fv, frank, g, sl) -> {variant: LOC order (node ids, best first)}
    finish(ctx)  -> (JSON-able diagnostics, {name: array} for the npz)
Every arm serves exactly M nodes: LOC@M = the LOC order, then FLAT fill; FLAT+LOC@M = the fused order
f(u) = 1/(K0 + FLAT rank) + [u in LOC] / (K0 + LOC rank).  Development numbers; any p-value is descriptive."""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D

log = D.log
V1_TAG = "v1"
PH_KEY = "paired_vs_HARD_same_arm (gained = this variant serves ALL gold, HARD does not)"


def argv_opts():
    rows = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--rows=")), None)
    smoke = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), None)
    assert (rows is None) == (smoke is None)
    assert smoke is None or not os.path.abspath(smoke).lower().startswith(os.path.abspath(D.REPO).lower())
    return rows, smoke


def run(ds, tag, stem, mode, harness_file, make_spec, doc):
    ROWS, SMOKE_OUT = argv_opts()
    harness_file = os.path.abspath(harness_file)
    shas = {"harness": D.sha_file(harness_file), "arms": D.sha_file(os.path.abspath(__file__)),
            "lib": D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))}
    host0 = D.host_state()
    t_all = time.time()
    fp_out = os.path.join(SMOKE_OUT or D.OUT, "%s_%s__%s.json" % (stem, ds, tag))
    assert not os.path.exists(fp_out), "write-once: %s exists" % fp_out
    cd = D.AD.CanonicalDataset(ds)
    N = int(cd.n_nodes)
    cells = D.CELLS[ds]
    pop = D.Population(cd, ROWS)
    nq, gptr, ngold, ST = pop.nq, pop.gptr, pop.ngold, pop.ST
    parts = {c: D.Part(cd, D.TAG_OF[c]) for c in cells}
    t_ = time.time()
    S = make_spec(cd, pop, parts)
    S.record["seconds_build"] = round(time.time() - t_, 1)
    names = list(S.names)
    log("RUN %s %s %s: N %d, %d rows, %d gold nodes, cells %s, %d variants; structures %s" % (
        stem, ds, tag, N, nq, pop.ng_tot, cells, len(names), json.dumps(S.record)[:1500]))
    POS_FLAT = np.zeros(pop.ng_tot, np.int64)
    POS = {n: {"LOC": np.zeros(pop.ng_tot, np.int64), "FLAT+LOC": np.zeros(pop.ng_tot, np.int64),
               "lpos": np.zeros(pop.ng_tot, np.int64)} for n in names}
    LLEN = {n: np.zeros(nq, np.int64) for n in names}
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
            orders = S.row(j, of, od, os_, npos_j, fv, frank, g, sl)
            assert set(orders) == set(names)
            for n_ in names:
                Lord = orders[n_]
                pl, pf, lp = D.arm_positions(Lord, frank, g, N)
                POS[n_]["LOC"][sl], POS[n_]["FLAT+LOC"][sl], POS[n_]["lpos"][sl] = pl, pf, lp
                LLEN[n_][j] = len(Lord)
        log("  %s rows %d / %d (%.0fs, RSS %.0f MB)" % (ds, j1, nq, time.time() - t_, D._rss_mb()))
    t_loop = round(time.time() - t_, 1)
    agree = {"dense_top100_overlap_mean": round(float(agd.mean()), 5), "splade_top100_overlap_mean": round(float(ags.mean()), 5)}
    assert agree["dense_top100_overlap_mean"] >= D.AGREE_MIN and agree["splade_top100_overlap_mean"] >= D.AGREE_MIN, agree
    # ---- FLAT and each cell's HARD variant must reproduce the step-1 record exactly (same rows, same arithmetic)
    f1 = os.path.join(D.OUT, "loc_%s__%s.npz" % (ds, V1_TAG))
    v1check = "no v1 record"
    if os.path.exists(f1):
        z1 = np.load(f1)
        assert (z1["rows"][:nq] == pop.rows).all()
        ng = int(z1["gptr"][nq])
        assert ng == pop.ng_tot and (z1["pos_FLAT"][:ng] == POS_FLAT).all(), "FLAT positions differ from the v1 record"
        for c, n_ in S.v1_hard.items():
            assert (z1["pos_LOC__" + c][:ng] == POS[n_]["LOC"]).all(), "HARD LOC differs from v1 (%s)" % c
            assert (z1["pos_FLATLOC__" + c][:ng] == POS[n_]["FLAT+LOC"]).all(), "HARD FLAT+LOC differs from v1 (%s)" % c
            assert (z1["lpos__" + c][:ng] == POS[n_]["lpos"]).all()
        v1check = "FLAT and %s LOC / FLAT+LOC / LOC-order positions == %s (sha %s) on all %d gold nodes" % (
            ", ".join(sorted(S.v1_hard.values())), D.rel(f1), D.sha_file(f1)[:16], ng)
    log("v1 check: " + v1check)
    # ---- scores
    maxrank_flat = np.maximum.reduceat(POS_FLAT, gptr[:-1])
    FL = {M: D.per_query(POS_FLAT, M, gptr, ngold) for M in D.M_CURVE}
    FLAT_RES = {str(M): dict(D.score_block(*FL[M][:3], FL[M][3]), strata=D.strata_block(ST, *FL[M][:3])) for M in D.M_CURVE}
    inv = POS_FLAT >= D.MMAX
    RES = {}
    for n_ in names:
        ref = S.hard_ref.get(n_)
        arms = {}
        for arm in ("LOC", "FLAT+LOC"):
            arms[arm] = {}
            for M in D.M_CURVE:
                a_, y_, f_, sv = D.per_query(POS[n_][arm], M, gptr, ngold)
                e = D.score_block(a_, y_, f_, sv, ref=(FL[M][0], FL[M][3]))
                e["FLAT_budget_matching_this_ALL"] = D.flat_budget_matching(maxrank_flat, float(a_.mean()))
                if ref:
                    e[PH_KEY] = D.paired(D.per_query(POS[ref][arm], M, gptr, ngold)[0], a_)
                e["strata"] = D.strata_block(ST, a_, y_, f_, FL[M][0])
                miss = POS_FLAT >= M
                e["of_FLAT@M_missed_gold_nodes"] = {"n": int(miss.sum()), "arm_serves": int((miss & sv).sum()),
                                                    "silent_L0": int((miss & (POS[n_]["lpos"] < 0)).sum())}
                e["invisible_gold_nodes_served (FLAT rank >= %d)" % D.MMAX] = int((inv & sv).sum())
                arms[arm][str(M)] = e
        RES[n_] = {"arms": arms, "loc_order_length": D.stats(LLEN[n_]), "gold_in_loc_order": D.q4((POS[n_]["lpos"] >= 0).mean()),
                   "latency_ms (score + LOC order)": D.ms_stats(S.lat.get(n_, []))}
        log("%-26s ALL  LOC %s | FLAT+LOC %s" % (n_, " ".join("%.3f" % arms["LOC"][str(M)]["ALL"] for M in D.M_CURVE),
                                                 " ".join("%.3f" % arms["FLAT+LOC"][str(M)]["ALL"] for M in D.M_CURVE)))
    log("%-26s ALL  %s" % ("FLAT", " ".join("%.3f" % FLAT_RES[str(M)]["ALL"] for M in D.M_CURVE)))
    ctx = {"POS_FLAT": POS_FLAT, "POS": POS, "LLEN": LLEN, "gptr": gptr, "ngold": ngold, "pop": pop, "ST": ST, "N": N}
    diag, extra_arrays = S.finish(ctx)
    res = {"dataset": ds, "tag": tag, "mode": mode, "status": "DEVELOPMENT (not confirmatory; p-values descriptive)", "definitions": doc,
           "N": N, "n_rows": nq, "n_gold_nodes": pop.ng_tot, "population": pop.record, "cells": cells, "variants": names,
           "structures": S.record, "v1_check": v1check, "served_list_agreement": agree, "FLAT": FLAT_RES, "variants_result": RES,
           "invisible_gold_nodes (FLAT rank >= %d)" % D.MMAX: int(inv.sum()), "diagnostics": diag,
           "latency_ms_flat": {"products_amortized": D.ms_stats(LAT["products_amortized"]), "flat_rrf": D.ms_stats(LAT["flat_rrf"])},
           "code": {"harness": {"path": D.rel(harness_file), "sha256": shas["harness"]},
                    "arms": {"path": "scratchpad/_l1d_arms.py", "sha256": shas["arms"]},
                    "lib": {"path": "scratchpad/_l1d_lib.py", "sha256": shas["lib"]}},
           "pinned": D.PINNED, "pinned_repo": D.PINNED_REPO, "function_sources_sha256": D.SRC_SHA, "constants": D.CONSTANTS,
           "platform": D.platform_record(), "host_at_start": host0, "seconds_loop": t_loop, "_row_query_ids": pop.qids}
    arrays = {"rows": pop.rows, "gptr": gptr, "pos_FLAT": POS_FLAT}
    for n_ in names:
        arrays["pos_LOC__" + n_] = POS[n_]["LOC"]
        arrays["pos_FLATLOC__" + n_] = POS[n_]["FLAT+LOC"]
        arrays["lpos__" + n_] = POS[n_]["lpos"]
        arrays["loc_len__" + n_] = LLEN[n_]
    for k, v in extra_arrays.items():
        assert k not in arrays
        arrays[k] = v
    res["seconds"] = round(time.time() - t_all, 1)
    res["process_peak_rss_mb"] = D.peak_rss_mb()
    assert D.sha_file(harness_file) == shas["harness"] and D.sha_file(os.path.abspath(__file__)) == shas["arms"] and \
        D.sha_file(os.path.join(D.HERE, "_l1d_lib.py")) == shas["lib"], "code changed during the run"
    fz = fp_out.replace(".json", ".npz")
    assert not os.path.exists(fz)
    np.savez_compressed(fz, **arrays)
    res["npz"] = {"path": os.path.basename(fz), "sha256": D.sha_file(fz)}
    D.G.S.wj(fp_out, res)
    log("done (%.0fs, peak RSS %s MB) -> %s sha256 %s" % (res["seconds"], res["process_peak_rss_mb"], fp_out, D.sha_file(fp_out)[:12]))
