"""Summary of the ROUTE development runs, read-only over <dir>/<stem>_<dataset>__<tag>.{json,npz} for the stems route
(_l1d_route.py) and, when given, route2 (_l1d_route2.py) and route3 (_l1d_route3.py).  The records share the population, the served order and the reference
rankings / counts (asserted equal here, query by query), so their rankings and counts merge into ONE factorial: every ranking of
any record x every count of any record (plus PSTAR_BN = the distinct partitions of O[:B_N], a B_N-coupled lossless reference) is
evaluated on the per-query ALL-served intervals [LO, HI(B_N)] (the section-33 machinery: _l1d_adaptbp.surface / served_at), per
cell (metaqa, metaqa_phg, musique, squad, squad_phg).  It reports:
  - per cell: unrouted, the envelope (the best fixed point of any ranking, gold-chosen hindsight), PSTAR, the oracle fan-out
    (the distinct gold partitions of the query);
  - per ranking: the fixed-B_P surface summary (best fixed, smallest B_P within 0.01 of unrouted) and the gold fan-out LO;
  - per count: the fan-out distribution B_P(q) in every cell (the "few shards on MetaQA, many on MuSiQue" question);
  - universality: every arm (ranking, count) ordered by its worst-cell distance to the envelope and, separately, to unrouted;
    the (largest cell fan-out fraction, worst-cell distance) frontier; lossless routing (the smallest fan-out that stays within
    delta of unrouted in EVERY cell, delta a reporting threshold); the per-cell (mean fan-out fraction, ALL) Pareto frontier;
  - the native arms of every record (ALL, reach, over-contact), asserted equal to the records;
  - split-half replication (even / odd rows) of the arm ordering;
  - with route3: its regime shares, the oracle regime switch (recounted here from the merged arrays), the B_N-coupled arm
    ES|ESTAR_BN (recounted), the load profiles and the router state (copied from the records).
Exact counts: every ALL is a count of queries; the markdown rounds exact fractions with Decimal ROUND_HALF_UP (shares to 3 decimals,
means to 1).
Usage: python scratchpad/_l1d_route_summary.py <tag> [--stems=route,route2,route3] [--dir=<input dir>] [--out=<output dir>] [--md=<path>]
       -> <out>/route_SUMMARY__<tag>.{json,npz} (write-once; out defaults to dir, dir to results/L1_DEV) + markdown.
DEVELOPMENT numbers (user rulings 2026-09-26/27): descriptive, no verdicts."""
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

import numpy as np

import _l1d_lib as D
import _l1d_adaptbp as AB
import _l1d_route as RT

TAG = sys.argv[1]
DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), D.OUT)
OUTD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), DIR)
MD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--md=")), None)
STEMS = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--stems=")), "route").split(",")
MODS = {"route": RT}
if "route2" in STEMS:
    import _l1d_route2 as RT2
    MODS["route2"] = RT2
if "route3" in STEMS:
    import _l1d_route3 as RT3
    MODS["route3"] = RT3
    assert "route2" in STEMS
assert STEMS[0] == "route" and len(set(STEMS)) == len(STEMS) and all(s in MODS for s in STEMS)
CELL_ORDER = [("metaqa", "metaqa"), ("metaqa", "metaqa_phg"), ("musique", "musique"), ("squad", "squad"), ("squad", "squad_phg")]
MS_SHOW = (100, 500, 1000, 2000, 5000)
M_MAIN = 1000
LO_K = (1, 2, 3, 5, 10, 20, 50, 100)
TOP_N = 40
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.05)
PSTAR = "PSTAR_BN"
INFO = {"S": "O (FLAT + the seeds' 1-hop rows)", "F": "FLAT", "D": "dense", "SP": "SPLADE", "FSUM": "FLAT[:200]",
        "FSUMFV": "FLAT[:200]", "FT3": "FLAT (every node)", "SUM": "the seeds' 1-hop rows", "MAX": "the seeds' 1-hop rows",
        "TOP3": "the seeds' 1-hop rows", "SDIV": "the seeds' 1-hop rows", "OT3": "O (every node)",
        "SC_L": "the seeds' 1-hop rows", "SC_LS": "the seeds' 1-hop rows", "FL_L": "the seeds' 1-hop rows", "FL_LS": "the seeds' 1-hop rows",
        "PD": "the seeds' 1-hop rows + the static partition graph", "PF_S_T3": "O + the seeds' 1-hop rows",
        "PF_F_T3": "FLAT + the seeds' 1-hop rows", "IL_S_T3": "O + the seeds' 1-hop rows", "U_NEFF": "O + FLAT[:200] + the seeds' 1-hop rows",
        "U_KNEE": "O + FLAT[:200] + the seeds' 1-hop rows"}
assert set(INFO) == set(RT.RANKS)
RANKS, RCOUNTS, NATIVE = list(RT.RANKS), list(RT.COUNTS), [("route", r_, k_) for r_, k_ in RT.NATIVE]
for s_ in STEMS[1:]:
    RANKS += [r_ for r_ in MODS[s_].RANKS if r_ not in RANKS]
    RCOUNTS += [k_ for k_ in MODS[s_].COUNTS if k_ not in RCOUNTS]
    NATIVE += [(s_, r_, k_) for r_, k_ in MODS[s_].NATIVE]
    INFO.update(MODS[s_].INFO)
assert set(INFO) == set(RANKS)
COUNTS = RCOUNTS + [PSTAR]


def rh(fr, nd=3):
    """exact fraction -> string, Decimal ROUND_HALF_UP (ties away from zero)."""
    d = Decimal(fr.numerator) / Decimal(fr.denominator)
    v = d.quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return str(abs(v) if v == 0 else v)


def sh(num, n, nd=3):
    return rh(Fraction(int(num), int(n)), nd)


def dl(a, b, n, nd=3):
    s = rh(Fraction(int(a) - int(b), int(n)), nd)
    return s if s.startswith("-") else "+" + s


def mean1(x):
    x = np.asarray(x, np.int64)
    return rh(Fraction(int(x.sum()), len(x)), 1)


def fr3(num, den):
    return rh(Fraction(int(num), int(den)), 3)


def dist_i(x):
    x = np.asarray(x, np.float64)
    return {"mean": round(float(x.mean()), 2), "median": float(np.median(x)), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "min": float(x.min()), "max": float(x.max())}


def rqf(x, nd=2):
    """a float (not a count share) -> string, Decimal ROUND_HALF_UP of its shortest repr."""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def auc2(pos, neg):
    """Mann-Whitney AUC of pos over neg as exact integers [2U, 2 n_pos n_neg] (ties count one half), None if a side is empty."""
    if not len(pos) or not len(neg):
        return None
    ra = 2.0 * AB.rankavg(np.concatenate([pos, neg]))
    r2 = np.rint(ra).astype(np.int64)
    assert np.abs(r2 - ra).max() < 1e-6
    return [int(r2[:len(pos)].sum()) - len(pos) * (len(pos) - 1), 2 * len(pos) * len(neg)]


def load_exact(cnt, nq):
    """integer load profile of one arm: per-partition contact counts over nq queries."""
    npart, tot = len(cnt), int(cnt.sum())
    assert tot > 0, "an arm with no contacts"
    srt = np.sort(cnt).astype(np.int64)
    k1 = int(np.ceil(0.01 * npart))
    i = np.arange(1, npart + 1, dtype=np.int64)
    return {"nq": int(nq), "npart": npart, "total_contacts": tot, "largest": int(srt[-1]), "top_1pct_k": k1, "top_1pct_sum": int(srt[-k1:].sum()),
            "never_contacted": int((cnt == 0).sum()), "gini_num": int(2 * (i * srt).sum() - (npart + 1) * tot), "gini_den": int(npart * tot)}


summ = {"stems": STEMS, "tag": TAG, "status": "DEVELOPMENT (descriptive; not confirmatory)", "inputs": {}, "cells": {},
        "rounding": "shares from exact query counts, Decimal ROUND_HALF_UP to 3 decimals (means to 1); JSON shares are q4 of the exact value",
        "ranking_information": INFO, "rankings": RANKS, "counts": COUNTS}
REC, Z = {}, {}
HSHA = {s_: set() for s_ in STEMS}
for ds in ("metaqa", "musique", "squad"):
    REC[ds], Z[ds] = {}, {}
    summ["inputs"][ds] = {}
    for s_ in STEMS:
        fj = os.path.join(DIR, "%s_%s__%s.json" % (s_, ds, TAG))
        r = json.load(open(fj, encoding="utf-8"))
        fz = os.path.join(DIR, r["npz"]["path"])
        assert D.sha_file(fz) == r["npz"]["sha256"], "npz changed: %s" % fz
        REC[ds][s_], Z[ds][s_] = r, np.load(fz)
        HSHA[s_].add(r["code"]["harness"]["sha256"])
        summ["inputs"][ds][s_] = {"json": {"path": D.rel(fj), "sha256": D.sha_file(fj)}, "npz": {"path": D.rel(fz), "sha256": r["npz"]["sha256"]},
                                  "n_rows": r["n_rows"], "n_gold_nodes": r["n_gold_nodes"], "identity": r["diagnostics"]["identity"],
                                  "v1_check": r["v1_check"], "seconds": r["seconds"], "peak_rss_mb": r["process_peak_rss_mb"],
                                  "host_at_start": r["host_at_start"]}
summ["harness_sha256"] = {}
for s_ in STEMS:
    assert len(HSHA[s_]) == 1, "the %s records were written by different harness versions" % s_
    summ["harness_sha256"][s_] = HSHA[s_].pop()
    assert summ["harness_sha256"][s_] == D.sha_file(os.path.abspath(MODS[s_].__file__)), "imported %s differs from its records' harness" % s_
nr, nk, nm = len(RANKS), len(COUNTS), len(D.M_CURVE)
nrc = len(RCOUNTS)
mi0 = D.M_CURVE.index(M_MAIN)
ARR = {}
ARMS = [(r_, k_) for r_ in RANKS for k_ in COUNTS]
U = {M: {"denv": np.zeros((len(ARMS), len(CELL_ORDER)), np.int64), "dunr": np.zeros((len(ARMS), len(CELL_ORDER)), np.int64),
         "alln": np.zeros((len(ARMS), len(CELL_ORDER)), np.int64), "n": np.zeros(len(CELL_ORDER), np.int64),
         "frac": np.zeros((len(ARMS), len(CELL_ORDER))), "bpsum": np.zeros((len(ARMS), len(CELL_ORDER)), np.int64),
         "npart": np.zeros(len(CELL_ORDER), np.int64)} for M in D.M_CURVE}
HALF = {h: {"denv": np.zeros((len(ARMS), len(CELL_ORDER)), np.int64), "n": np.zeros(len(CELL_ORDER), np.int64)} for h in (0, 1)}
MERGED = {}
for ci_, (ds, c) in enumerate(CELL_ORDER):
    r0, z0 = REC[ds]["route"], Z[ds]["route"]
    npart = r0["structures"]["cells"][c]["npart"]
    gptr = z0["gptr"].astype(np.int64)
    nq = len(gptr) - 1
    LOd, HId, PRGd, CNTd = {}, {}, {}, {}
    PST = z0["PSTAR__" + c].astype(np.int64)
    for s_ in STEMS:
        z = Z[ds][s_]
        zr, zc = [str(x) for x in z["ranks"]], [str(x) for x in z["counts"]]
        assert zr == list(MODS[s_].RANKS) and zc == list(MODS[s_].COUNTS) and [int(m) for m in z["m_curve"]] == list(D.M_CURVE)
        assert (z["gptr"] == z0["gptr"]).all() and (z["rows"] == z0["rows"]).all() and (z["PSTAR__" + c] == z0["PSTAR__" + c]).all()
        assert REC[ds][s_]["structures"]["cells"][c]["npart"] == npart
        for i, rk in enumerate(zr):
            lo, hi, prg = z["LO__" + c][:, i].astype(np.int64), z["HI__" + c][:, i].astype(np.int64), z["PRG__" + c][:, i].astype(np.int64)
            if rk in LOd:
                assert (LOd[rk] == lo).all() and (HId[rk] == hi).all() and (PRGd[rk] == prg).all(), "ranking %s differs across records (%s)" % (rk, c)
            else:
                LOd[rk], HId[rk], PRGd[rk] = lo, hi, prg
        for i, cn in enumerate(zc):
            b = z["CNT__" + c][:, i].astype(np.int64)
            if cn in CNTd:
                assert (CNTd[cn] == b).all(), "count %s differs across records (%s)" % (cn, c)
            else:
                CNTd[cn] = b
    LO = np.stack([LOd[rk] for rk in RANKS], 1)
    HI = np.stack([HId[rk] for rk in RANKS], 1)
    PRG = np.stack([PRGd[rk] for rk in RANKS], 1)
    CNT = np.stack([CNTd[cn] for cn in RCOUNTS], 1)
    MERGED[c] = {"LO": LO, "HI": HI, "PSTAR": PST, "CNT": CNT}
    ngold = np.diff(gptr)
    # the oracle fan-out: the distinct gold partitions of each query (any ranking's positions identify them)
    ngp = np.array([len(np.unique(PRG[gptr[j]:gptr[j + 1], 0])) for j in range(nq)], np.int64)
    SFN = {}
    for ri, rk in enumerate(RANKS):
        S_ = AB.surface(LO[:, ri], HI[:, ri], npart)
        SFN[rk] = np.rint(S_ * nq).astype(np.int64)
        assert np.abs(SFN[rk] - S_ * nq).max() < 1e-6
    unrN = {M: int(SFN["S"][mi][-1]) for mi, M in enumerate(D.M_CURVE)}
    for rk in RANKS:
        for mi, M in enumerate(D.M_CURVE):
            assert SFN[rk][mi][-1] == unrN[M]
            assert abs(r0["diagnostics"]["cells"][c]["unrouted_ALL"][str(M)] - unrN[M] / nq) < 5e-5
    env = {}
    for mi, M in enumerate(D.M_CURVE):
        best = max(((int(SFN[rk][mi].max()), -(int(np.argmax(SFN[rk][mi])) + 1), rk) for rk in RANKS))
        env[M] = {"n": best[0], "ranking": best[2], "B_P": -best[1]}
    # full factorial: ALLN[r, k, m] = queries with every gold served
    ALLN = np.zeros((nr, nk, nm), np.int64)
    SRV_MAIN = np.zeros((nr, nk, nq), bool)
    for ri in range(nr):
        for ki in range(nk):
            for mi in range(nm):
                b = CNT[:, ki] if ki < nrc else PST[:, mi]
                sv = AB.served_at(LO[:, ri], HI[:, ri, mi], b)
                ALLN[ri, ki, mi] = int(sv.sum())
                if mi == mi0:
                    SRV_MAIN[ri, ki] = sv
    for s_, rk, cn in NATIVE:
        ri, ki = RANKS.index(rk), COUNTS.index(cn)
        for M in D.M_CURVE:
            assert abs(REC[ds][s_]["diagnostics"]["cells"][c]["native_arms"]["%s|%s" % (rk, cn)][str(M)]["ALL"] - ALLN[ri, ki, D.M_CURVE.index(M)] / nq) < 5e-5
    BPS = np.zeros((nk, nm), np.int64)
    for ki in range(nk):
        for mi in range(nm):
            BPS[ki, mi] = int((CNT[:, ki] if ki < nrc else PST[:, mi]).sum())
    BPM = BPS / float(nq)
    ce = {"dataset": ds, "npart": npart, "n_rows": nq, "partition_graph": r0["structures"]["cells"][c]["partition_graph"],
          "unrouted_ALL": {str(M): {"n": unrN[M], "share": D.q4(unrN[M] / nq)} for M in D.M_CURVE},
          "envelope (best fixed point of any ranking; gold-chosen hindsight)": {str(M): dict(env[M], share=D.q4(env[M]["n"] / nq)) for M in D.M_CURVE},
          "PSTAR (distinct partitions of O[:B_N]; S at PSTAR == unrouted)": {str(M): dist_i(PST[:, mi]) for mi, M in enumerate(D.M_CURVE)},
          "distinct_gold_partitions (oracle fan-out)": dist_i(ngp), "gold_nodes_per_query": dist_i(ngold)}
    ce["rankings"] = {}
    for ri, rk in enumerate(RANKS):
        e = {"information": INFO[rk], "LO": dist_i(LO[:, ri]),
             "LO <= k (every gold partition within the first k)": {str(k): D.q4((LO[:, ri] <= k).mean()) for k in LO_K},
             "LO == oracle (gold partitions ranked first)": D.q4((LO[:, ri] == ngp).mean())}
        for mi, M in enumerate(D.M_CURVE):
            row_ = SFN[rk][mi]
            e[str(M)] = {"best_fixed": {"B_P": int(np.argmax(row_)) + 1, "n": int(row_.max())},
                         "smallest_B_P_within_0.01_of_unrouted": int(np.flatnonzero(row_ * 100 >= unrN[M] * 100 - nq)[0]) + 1,
                         "smallest_B_P_at_or_above_unrouted": int(np.flatnonzero(row_ >= unrN[M])[0]) + 1}
        ce["rankings"][rk] = e
    ce["counts"] = {}
    for ki, cn in enumerate(COUNTS):
        if cn == PSTAR:
            ce["counts"][cn] = {str(M): dist_i(PST[:, mi]) for mi, M in enumerate(D.M_CURVE)}
        else:
            ce["counts"][cn] = {"B_P(q)": dist_i(CNT[:, ki]), "fraction_of_partitions_mean": D.q4(CNT[:, ki].mean() / npart),
                                "spearman(B_P(q), LO) under S / TOP3": [AB.spearman(CNT[:, ki], LO[:, RANKS.index("S")]),
                                                                        AB.spearman(CNT[:, ki], LO[:, RANKS.index("TOP3")])]}
    # calibration of every count against every ranking at B_N 1000: reach and over-contact
    REACH = np.zeros((nr, nk), np.int64)
    OVER = np.zeros((nr, nk), np.int64)
    for ri in range(nr):
        for ki in range(nk):
            b = CNT[:, ki] if ki < nrc else PST[:, mi0]
            REACH[ri, ki] = int((b >= LO[:, ri]).sum())
            OVER[ri, ki] = int(((LO[:, ri] <= HI[:, ri, mi0]) & (b > HI[:, ri, mi0])).sum())
    ce["native_arms"] = {}
    for s_, rk, cn in NATIVE:
        ri, ki = RANKS.index(rk), COUNTS.index(cn)
        z = Z[ds][s_]
        rs, ks = list(MODS[s_].RANKS).index(rk), list(MODS[s_].COUNTS).index(cn)
        ce["native_arms"]["%s|%s" % (rk, cn)] = {
            "record": s_, "B_P(q)": dist_i(CNT[:, ki]), "contacted_mass_mean": round(float(z["CMASS__" + c][:, rs, ks].astype(np.int64).mean()), 1),
            "ALL": {str(M): {"n": int(ALLN[ri, ki, mi]), "share": D.q4(ALLN[ri, ki, mi] / nq)} for mi, M in enumerate(D.M_CURVE)},
            "reach (B_P(q) >= LO)": D.q4(REACH[ri, ki] / nq), "over-contact at 1000 (served at a smaller B_P, lost at B_P(q))": D.q4(OVER[ri, ki] / nq),
            "lost at 1000": REC[ds][s_]["diagnostics"]["cells"][c]["native_arms"]["%s|%s" % (rk, cn)]["1000"]}
    # per-cell Pareto frontier at B_N 1000 (rules only): no other arm reaches at least this ALL with a smaller mean fan-out
    pf, best_n = [], -1
    for ai in np.lexsort((-ALLN[:, :nrc, mi0].reshape(-1), BPS[:nrc, mi0][None, :].repeat(nr, 0).reshape(-1))):
        ri, ki = divmod(int(ai), nrc)
        if ALLN[ri, ki, mi0] > best_n:
            best_n = int(ALLN[ri, ki, mi0])
            pf.append({"arm": "%s|%s" % (RANKS[ri], RCOUNTS[ki]), "B_P_sum": int(BPS[ki, mi0]), "mean_B_P": round(BPS[ki, mi0] / float(nq), 2),
                       "fraction_of_partitions": D.q4(BPS[ki, mi0] / float(nq * npart)), "ALL_n": best_n, "ALL": D.q4(best_n / nq)})
    ce["pareto_frontier_at_1000 (rules; mean fan-out -> ALL)"] = pf
    summ["cells"][c] = ce
    ARR["ALLN__" + c] = ALLN
    ARR["BPSUM__" + c] = BPS
    ARR["REACH__" + c] = REACH
    ARR["OVER__" + c] = OVER
    ARR["NGP__" + c] = ngp
    for M in D.M_CURVE:
        mi = D.M_CURVE.index(M)
        U[M]["n"][ci_] = nq
        U[M]["npart"][ci_] = npart
        for ai, (rk, cn) in enumerate(ARMS):
            ri, ki = RANKS.index(rk), COUNTS.index(cn)
            U[M]["alln"][ai, ci_] = ALLN[ri, ki, mi]
            U[M]["denv"][ai, ci_] = ALLN[ri, ki, mi] - env[M]["n"]
            U[M]["dunr"][ai, ci_] = ALLN[ri, ki, mi] - unrN[M]
            U[M]["frac"][ai, ci_] = BPM[ki, mi] / npart
            U[M]["bpsum"][ai, ci_] = BPS[ki, mi]
    ev, od = np.arange(nq) % 2 == 0, np.arange(nq) % 2 == 1
    for h, m_ in ((0, ev), (1, od)):
        HALF[h]["n"][ci_] = int(m_.sum())
        envh = max(int(AB.surface(LO[m_, ri], HI[m_, ri], npart)[mi0].max() * m_.sum() + 0.5) for ri in range(nr))
        for ai, (rk, cn) in enumerate(ARMS):
            HALF[h]["denv"][ai, ci_] = int(SRV_MAIN[RANKS.index(rk), COUNTS.index(cn)][m_].sum()) - envh

# ---------------- universality: one rule on every cell at once
C = [c for _, c in CELL_ORDER]
univ = {}
RULE = np.array([cn != PSTAR for _, cn in ARMS])


def arm_row(u, ai, wenv, wunr, fmax):
    rk, cn = ARMS[ai]
    return {"arm": "%s|%s" % (rk, cn), "worst_cell_delta_vs_envelope": D.q4(wenv[ai]), "worst_cell_delta_vs_unrouted": D.q4(wunr[ai]),
            "max_cell_fraction_of_partitions": D.q4(fmax[ai]),
            "per_cell": {c: {"delta_vs_envelope": D.q4(u["denv"][ai, j] / u["n"][j]), "delta_vs_unrouted": D.q4(u["dunr"][ai, j] / u["n"][j]),
                             "fraction_of_partitions": D.q4(u["frac"][ai, j])} for j, c in enumerate(C)}}


for M in D.M_CURVE:
    u = U[M]
    wenv = (u["denv"] / u["n"][None, :]).min(1)
    wunr = (u["dunr"] / u["n"][None, :]).min(1)
    fmax = u["frac"].max(1)
    fmean = u["frac"].mean(1)
    rows = [arm_row(u, ai, wenv, wunr, fmax) for ai in [a for a in np.lexsort((fmax, -wenv)) if RULE[a]][:TOP_N]]
    rows_u = [arm_row(u, ai, wenv, wunr, fmax) for ai in [a for a in np.lexsort((fmax, -wunr)) if RULE[a]][:TOP_N]]
    front = []
    best_w = -np.inf
    for ai in np.lexsort((-wenv, fmax)):
        if RULE[ai] and wenv[ai] > best_w + 1e-12:
            best_w = wenv[ai]
            front.append(arm_row(u, ai, wenv, wunr, fmax))
    lossless = {}
    for dlt in DELTAS:
        ok = RULE & (u["dunr"] * 1000 >= -int(round(dlt * 1000)) * u["n"][None, :]).all(1)
        cand = np.flatnonzero(ok)
        lossless[str(dlt)] = {"n_arms": int(len(cand)),
                              "smallest_largest_cell_fraction": arm_row(u, int(cand[np.lexsort((fmean[cand], fmax[cand]))[0]]), wenv, wunr, fmax) if len(cand) else None,
                              "smallest_mean_cell_fraction": arm_row(u, int(cand[np.lexsort((fmax[cand], fmean[cand]))[0]]), wenv, wunr, fmax) if len(cand) else None}
    pst = {}
    for rk in RANKS:
        ai = ARMS.index((rk, PSTAR))
        pst[rk] = {"worst_cell_delta_vs_envelope": D.q4(wenv[ai]), "worst_cell_delta_vs_unrouted": D.q4(wunr[ai]),
                   "per_cell_delta_vs_unrouted": {c: D.q4(u["dunr"][ai, j] / u["n"][j]) for j, c in enumerate(C)},
                   "per_cell_fraction_of_partitions": {c: D.q4(u["frac"][ai, j]) for j, c in enumerate(C)}}
    univ[str(M)] = {"ordered_by_worst_cell_delta_vs_envelope (rules only)": rows,
                    "ordered_by_worst_cell_delta_vs_unrouted (rules only)": rows_u,
                    "frontier (largest cell fan-out fraction -> worst-cell envelope distance; rules only)": front,
                    "lossless routing (every cell within delta of unrouted; rules only)": lossless,
                    "PSTAR_BN reference (B_N-coupled; S at PSTAR == unrouted)": pst}
    ARR["UNIV_DENV__%d" % M] = u["denv"]
    ARR["UNIV_DUNR__%d" % M] = u["dunr"]
    ARR["UNIV_FRAC__%d" % M] = u["frac"]
summ["universality"] = univ
w0 = (HALF[0]["denv"] / HALF[0]["n"][None, :]).min(1)
w1 = (HALF[1]["denv"] / HALF[1]["n"][None, :]).min(1)
t0, t1 = set(np.argsort(-w0, kind="stable")[:10].tolist()), set(np.argsort(-w1, kind="stable")[:10].tolist())
summ["split_half (even / odd rows; worst-cell distance to each half's envelope at B_N 1000)"] = {
    "spearman_over_arms": AB.spearman(w0, w1), "top10_overlap": len(t0 & t1),
    "top10_even": ["%s|%s" % ARMS[a] for a in np.argsort(-w0, kind="stable")[:10]],
    "top10_odd": ["%s|%s" % ARMS[a] for a in np.argsort(-w1, kind="stable")[:10]]}
ARR["arms"] = np.array(["%s|%s" % a for a in ARMS])
ARR["cells"] = np.array(C)
ARR["ranks"] = np.array(RANKS)
ARR["counts"] = np.array(COUNTS)
ARR["m_curve"] = np.asarray(D.M_CURVE, np.int64)
ARR["HALF_DENV__even"] = HALF[0]["denv"]
ARR["HALF_DENV__odd"] = HALF[1]["denv"]

# ---------------- round-3 diagnostics: recount the oracle switch and ES|ESTAR_BN from the merged arrays, copy the rest
R3 = None
if "route3" in STEMS:
    R3 = {"datasets": {}, "cells": {}}
    for ds in ("metaqa", "musique", "squad"):
        d3 = REC[ds]["route3"]["diagnostics"]
        z3 = Z[ds]["route3"]
        R3["datasets"][ds] = {"regime_shares (L1-L3)": d3["regime_shares (L4 per cell)"],
                              "router_reads_node_rows": d3["router_reads_per_query (entries read for the 200 seeds)"]["node_rows (all four families)"],
                              "router_reads_node_rows (exact sum, n)": [int(z3["RDN"].astype(np.int64).sum()), int(len(z3["RDN"]))],
                              "router_order_length": d3["router_order_length |O'| = |H_q u {L > 0}|"],
                              "router_order_length (exact sum, n)": [int(z3["OQLEN"].astype(np.int64).sum()), int(len(z3["OQLEN"]))],
                              "latency_ms": d3["latency_ms"]}
    for ds, c in CELL_ORDER:
        r3, z3 = REC[ds]["route3"], Z[ds]["route3"]
        ce3 = r3["diagnostics"]["cells"][c]
        LO, HI, CNT = MERGED[c]["LO"], MERGED[c]["HI"], MERGED[c]["CNT"]
        nq = len(LO)
        LAMc = np.column_stack([z3["LAM"], z3["LAM4__" + c]])
        assert LAMc.shape == (nq, len(RT3.LAMS))
        ri_es, ri_b = RANKS.index("ES"), RANKS.index(RT3.ORACLE_B[0])
        est = z3["ESTAR__" + c].astype(np.int64)
        assert (z3["PSTAR__" + c] == MERGED[c]["PSTAR"]).all()
        esn = {}
        for mi, M in enumerate(D.M_CURVE):
            sv = AB.served_at(LO[:, ri_es], HI[:, ri_es, mi], est[:, mi])
            esn[str(M)] = {"n": int(sv.sum()), "ESTAR_sum": int(est[:, mi].sum()), "PSTAR_sum": int(MERGED[c]["PSTAR"][:, mi].sum()),
                           "share_of_PSTAR_inside_ESTAR_mean_f64": float(z3["ESREC__" + c][:, mi].mean())}
            assert abs(ce3["b_n_coupled_arm ES|ESTAR_BN"][str(M)]["ALL"] - esn[str(M)]["n"] / nq) < 5e-5
        orc = {}
        kB = RCOUNTS.index(RT3.ORACLE_B[1])
        for rkA, cnA in RT3.ORACLE_A:
            kA = RCOUNTS.index(cnA)
            key = "%s|%s vs %s|%s" % (rkA, cnA, RT3.ORACLE_B[0], RT3.ORACLE_B[1])
            o_ = {}
            for mi, M in enumerate(D.M_CURVE):
                A_ = AB.served_at(LO[:, RANKS.index(rkA)], HI[:, RANKS.index(rkA), mi], CNT[:, kA])
                B_ = AB.served_at(LO[:, ri_b], HI[:, ri_b, mi], CNT[:, kB])
                rec_o = ce3["oracle_regime_switch"][key][str(M)]
                bo, ao = B_ & ~A_, A_ & ~B_
                o_[str(M)] = {"A_n": int(A_.sum()), "B_n": int(B_.sum()), "oracle_n": int((A_ | B_).sum()), "A_only": int(ao.sum()),
                              "B_only": int(bo.sum()), "AUC_lam (B-only vs A-only) [numerator x2, denominator x2]": {
                                  l: auc2(LAMc[bo, li], LAMc[ao, li]) for li, l in enumerate(RT3.LAMS)}}
                assert rec_o["A_only"] == o_[str(M)]["A_only"] and rec_o["B_only"] == o_[str(M)]["B_only"]
                for l in RT3.LAMS:
                    a2, r2 = o_[str(M)]["AUC_lam (B-only vs A-only) [numerator x2, denominator x2]"][l], rec_o["AUC_lam (B-only vs A-only)"][l]
                    assert (a2 is None) == (r2 is None) and (a2 is None or abs(a2[0] / float(a2[1]) - r2) < 5e-5)
            orc[key] = o_
        LD = z3["LOAD__" + c].astype(np.int64)
        la = [str(s) for s in z3["load_arms"]]
        assert la == ["%s|%s" % x for x in RT3.LOAD_ARMS] and LD.shape[1] == summ["cells"][c]["npart"]
        R3["cells"][c] = {"regime_share_L4": ce3["regime_share_L4"], "regime_share_spearman": ce3["regime_share_spearman"],
                          "regime_shares_exact_quantiles": {l: {"mean": float(LAMc[:, li].mean()), "p10": float(np.percentile(LAMc[:, li], 10)),
                                                                "p90": float(np.percentile(LAMc[:, li], 90))} for li, l in enumerate(RT3.LAMS)},
                          "oracle_regime_switch (exact counts)": orc, "ES|ESTAR_BN (exact counts)": esn,
                          "load (exact integer profile)": {a_: load_exact(LD[i], nq) for i, a_ in enumerate(la)},
                          "load (record)": ce3["load (per-partition contacts over the population)"],
                          "router_state (static)": r3["structures"]["router_state (static)"][c],
                          "router_reads_per_query": r3["diagnostics"]["router_reads_per_query (entries read for the 200 seeds)"]["per_cell"][c],
                          "router_reads_sketch (exact sums: per-family summed, union; n)": [
                              int(z3["RDS__" + c][:, 0].astype(np.int64).sum()), int(z3["RDS__" + c][:, 1].astype(np.int64).sum()), nq]}
    summ["route3_diagnostics"] = R3

fj = os.path.join(OUTD, "route_SUMMARY__%s.json" % TAG)
fz = fj.replace(".json", ".npz")
assert not os.path.exists(fj) and not os.path.exists(fz), "write-once"
np.savez_compressed(fz, **ARR)
summ["npz"] = {"path": os.path.basename(fz), "sha256": D.sha_file(fz)}
summ["code"] = {"path": "scratchpad/_l1d_route_summary.py", "sha256": D.sha_file(os.path.abspath(__file__))}
D.G.S.wj(fj, summ)
print("wrote %s sha256 %s" % (fj, D.sha_file(fj)[:12]))

# ---------------- markdown (exact counts)
md = ["# ROUTE summary (%s; records %s)" % (TAG, ", ".join(STEMS)), "",
      "harness sha %s; summary sha %s" % (", ".join("%s %s" % (s_, summ["harness_sha256"][s_][:12]) for s_ in STEMS), D.sha_file(fj)[:12]), ""]
md += ["## Cells", "", "| cell | npart | n | unrouted @100 / 500 / 1000 / 2000 / 5000 | envelope @1000 | PSTAR @1000 mean / median | distinct gold partitions mean |",
       "|---|---|---|---|---|---|---|"]
for ds, c in CELL_ORDER:
    ce = summ["cells"][c]
    nq = ce["n_rows"]
    e1 = ce["envelope (best fixed point of any ranking; gold-chosen hindsight)"][str(M_MAIN)]
    p1 = ce["PSTAR (distinct partitions of O[:B_N]; S at PSTAR == unrouted)"][str(M_MAIN)]
    md.append("| %s | %d | %d | %s | %s (%s %d) | %s / %g | %s |" % (
        c, ce["npart"], nq, " / ".join(sh(ce["unrouted_ALL"][str(M)]["n"], nq) for M in MS_SHOW), sh(e1["n"], nq), e1["ranking"], e1["B_P"],
        mean1(MERGED[c]["PSTAR"][:, mi0]), p1["median"], mean1(ARR["NGP__" + c])))
md.append("")
md += ["## Rankings at B_N = 1000: best fixed (B_P); smallest B_P within 0.01 of unrouted; LO mean; LO <= 5 / 20; LO == oracle", ""]
md += ["| ranking | information | " + " | ".join(C) + " |", "|---|---|" + "---|" * len(C)]
for rk in RANKS:
    cells_ = []
    for ds, c in CELL_ORDER:
        e = summ["cells"][c]["rankings"][rk]
        nq = summ["cells"][c]["n_rows"]
        LOr = MERGED[c]["LO"][:, RANKS.index(rk)]
        cells_.append("%s (%d); %d; %s; %s / %s; %s" % (sh(e[str(M_MAIN)]["best_fixed"]["n"], nq), e[str(M_MAIN)]["best_fixed"]["B_P"],
                                                   e[str(M_MAIN)]["smallest_B_P_within_0.01_of_unrouted"], mean1(LOr),
                                                   sh((LOr <= 5).sum(), nq), sh((LOr <= 20).sum(), nq),
                                                   sh((LOr == ARR["NGP__" + c]).sum(), nq)))
    md.append("| %s | %s | %s |" % (rk, INFO[rk], " | ".join(cells_)))
md.append("")
md += ["## Counts: B_P(q) mean / median / p90 per cell (PSTAR at B_N 1000)", ""]
md += ["| count | " + " | ".join(C) + " |", "|---|" + "---|" * len(C)]
for ki, cn in enumerate(COUNTS):
    cells_ = []
    for ds, c in CELL_ORDER:
        b = MERGED[c]["CNT"][:, ki] if cn != PSTAR else MERGED[c]["PSTAR"][:, mi0]
        cells_.append("%s / %g / %g" % (mean1(b), float(np.median(b)), round(float(np.percentile(b, 90)), 1)))
    md.append("| %s | %s |" % (cn, " | ".join(cells_)))
md.append("")


def md_arm(u, ai):
    return " | ".join("%s [%s]" % (dl(u["dunr"][ai, j], 0, u["n"][j]), fr3(u["bpsum"][ai, j], u["n"][j] * u["npart"][j])) for j in range(len(C)))


for M in (500, 1000, 2000):
    u = U[M]
    for key, title in (("ordered_by_worst_cell_delta_vs_envelope (rules only)", "worst-cell distance to the envelope"),
                       ("ordered_by_worst_cell_delta_vs_unrouted (rules only)", "worst-cell distance to unrouted")):
        md += ["## Universality at B_N = %d: arms by %s (top %d)" % (M, title, TOP_N), "", "per cell: delta vs unrouted [mean fan-out fraction]", ""]
        md += ["| arm | worst vs envelope | worst vs unrouted | " + " | ".join(C) + " |", "|---|---|---|" + "---|" * len(C)]
        for row in univ[str(M)][key]:
            ai = ARMS.index(tuple(row["arm"].split("|")))
            j_env = int(np.argmin(u["denv"][ai] / u["n"]))
            j_unr = int(np.argmin(u["dunr"][ai] / u["n"]))
            md.append("| %s | %s (%s) | %s (%s) | %s |" % (row["arm"], dl(u["denv"][ai, j_env], 0, u["n"][j_env]), C[j_env],
                                                       dl(u["dunr"][ai, j_unr], 0, u["n"][j_unr]), C[j_unr], md_arm(u, ai)))
        md.append("")
    md += ["Frontier (largest-cell fan-out fraction -> worst-cell distance to the envelope):", ""]
    for f_ in univ[str(M)]["frontier (largest cell fan-out fraction -> worst-cell envelope distance; rules only)"]:
        ai = ARMS.index(tuple(f_["arm"].split("|")))
        j_env = int(np.argmin(u["denv"][ai] / u["n"]))
        fmx = max(Fraction(int(u["bpsum"][ai, j]), int(u["n"][j] * u["npart"][j])) for j in range(len(C)))
        md.append("- %s: largest-cell fraction %s; worst vs envelope %s (%s); per cell vs unrouted [fraction] %s" % (
            f_["arm"], rh(fmx, 3), dl(u["denv"][ai, j_env], 0, u["n"][j_env]), C[j_env], md_arm(u, ai)))
    md.append("")
    md += ["Lossless routing (every cell within delta of unrouted): the arm with the smallest largest-cell fan-out fraction, and the "
           "arm with the smallest mean-cell fraction", ""]
    md += ["| delta | arms | smallest largest-cell fraction | smallest mean-cell fraction |", "|---|---|---|---|"]
    for dlt in DELTAS:
        e = univ[str(M)]["lossless routing (every cell within delta of unrouted; rules only)"][str(dlt)]
        cells_ = []
        for k_ in ("smallest_largest_cell_fraction", "smallest_mean_cell_fraction"):
            if e[k_] is None:
                cells_.append("none")
            else:
                ai = ARMS.index(tuple(e[k_]["arm"].split("|")))
                cells_.append("%s: %s" % (e[k_]["arm"], md_arm(u, ai)))
        md.append("| %s | %d | %s | %s |" % (dlt, e["n_arms"], cells_[0], cells_[1]))
    md.append("")
    md += ["PSTAR_BN (the distinct partitions of O[:%d]; B_N-coupled lossless reference) under each ranking: per cell vs unrouted "
           "[fan-out fraction]" % M, ""]
    md += ["| ranking | " + " | ".join(C) + " |", "|---|" + "---|" * len(C)]
    for rk in RANKS:
        md.append("| %s | %s |" % (rk, md_arm(u, ARMS.index((rk, PSTAR)))))
    md.append("")
md += ["## Per-cell Pareto frontier at B_N = 1000 (rules; mean fan-out -> ALL; [fan-out fraction])", ""]
for ds, c in CELL_ORDER:
    nq = summ["cells"][c]["n_rows"]
    npart = summ["cells"][c]["npart"]
    md.append("- %s (unrouted %s): " % (c, sh(summ["cells"][c]["unrouted_ALL"][str(M_MAIN)]["n"], nq)) + "; ".join(
        "%s %s @ %s [%s]" % (p["arm"], sh(p["ALL_n"], nq), rh(Fraction(p["B_P_sum"], nq), 1), fr3(p["B_P_sum"], nq * npart))
        for p in summ["cells"][c]["pareto_frontier_at_1000 (rules; mean fan-out -> ALL)"]))
md.append("")
md += ["## Native arms at B_N = 1000: ALL vs unrouted; mean B_P; reach (B_P(q) >= LO); over-contact", ""]
md += ["| arm | record | " + " | ".join(C) + " |", "|---|---|" + "---|" * len(C)]
for s_, rk, cn in NATIVE:
    ri, ki = RANKS.index(rk), COUNTS.index(cn)
    cells_ = []
    for ds, c in CELL_ORDER:
        nq = summ["cells"][c]["n_rows"]
        a_ = int(ARR["ALLN__" + c][ri, ki, mi0])
        cells_.append("%s (%s); %s; %s; %s" % (sh(a_, nq), dl(a_, summ["cells"][c]["unrouted_ALL"][str(M_MAIN)]["n"], nq),
                                               rh(Fraction(int(ARR["BPSUM__" + c][ki, mi0]), nq), 1),
                                               sh(ARR["REACH__" + c][ri, ki], nq), sh(ARR["OVER__" + c][ri, ki], nq)))
    md.append("| %s|%s | %s | %s |" % (rk, cn, s_, " | ".join(cells_)))
md.append("")
sp = summ["split_half (even / odd rows; worst-cell distance to each half's envelope at B_N 1000)"]
md += ["## Split-half (B_N 1000)", "", "spearman over %d arms %s; top-10 overlap %d" % (len(ARMS), sp["spearman_over_arms"], sp["top10_overlap"]),
       "", "even: " + ", ".join(sp["top10_even"]), "", "odd: " + ", ".join(sp["top10_odd"]), ""]


def fd(d):
    return "%s [%s, %s]" % (rqf(d["mean"]), rqf(d["p10"]), rqf(d["p90"]))


if R3 is not None:
    md += ["## Round-3 diagnostics", "", "Regime shares lam(q), mean [p10, p90] (2 decimals; L1-L3 per query, L4 per cell):", "",
           "| cell | L1 | L2 | L3 | L4 |", "|---|---|---|---|---|"]
    for ds, c in CELL_ORDER:
        q_ = R3["cells"][c]["regime_shares_exact_quantiles"]
        md.append("| %s | %s |" % (c, " | ".join(fd(q_[l]) for l in RT3.LAMS)))
    md += ["", "Oracle regime switch at B_N = %d (gold-dependent; per query the better of A = ES|C and B = %s|%s; an upper bound for any "
           "per-query switch between the two): share served [delta vs unrouted]; A-only / B-only queries; AUC of lam for B-only vs "
           "A-only (L1 / L2 / L3 / L4)" % (M_MAIN, RT3.ORACLE_B[0], RT3.ORACLE_B[1]), "",
           "| cell | C | A | B | oracle | A-only / B-only | AUC L1 / L2 / L3 / L4 |", "|---|---|---|---|---|---|---|"]
    for ds, c in CELL_ORDER:
        nq = summ["cells"][c]["n_rows"]
        un = summ["cells"][c]["unrouted_ALL"][str(M_MAIN)]["n"]
        for key, o_ in R3["cells"][c]["oracle_regime_switch (exact counts)"].items():
            e = o_[str(M_MAIN)]
            au = e["AUC_lam (B-only vs A-only) [numerator x2, denominator x2]"]
            md.append("| %s | %s | %s [%s] | %s [%s] | %s [%s] | %d / %d | %s |" % (
                c, key.split(" vs ")[0], sh(e["A_n"], nq), dl(e["A_n"], un, nq), sh(e["B_n"], nq), dl(e["B_n"], un, nq),
                sh(e["oracle_n"], nq), dl(e["oracle_n"], un, nq), e["A_only"], e["B_only"],
                " / ".join("-" if au[l] is None else sh(au[l][0], au[l][1]) for l in RT3.LAMS)))
    md += ["", "B_N-coupled router-side lossless estimate ES|ESTAR_BN (ESTAR_BN = the distinct partitions of O'[:B_N]): share served "
           "[delta vs unrouted]; mean ESTAR / mean PSTAR [ESTAR fraction of partitions]; mean share of PSTAR's partitions inside ESTAR", ""]
    md += ["| cell | " + " | ".join(str(M) for M in D.M_CURVE) + " |", "|---|" + "---|" * len(D.M_CURVE)]
    for ds, c in CELL_ORDER:
        nq, npart = summ["cells"][c]["n_rows"], summ["cells"][c]["npart"]
        cells_ = []
        for M in D.M_CURVE:
            e = R3["cells"][c]["ES|ESTAR_BN (exact counts)"][str(M)]
            un = summ["cells"][c]["unrouted_ALL"][str(M)]["n"]
            cells_.append("%s [%s]; %s / %s [%s]; %s" % (sh(e["n"], nq), dl(e["n"], un, nq), rh(Fraction(e["ESTAR_sum"], nq), 1),
                                                      rh(Fraction(e["PSTAR_sum"], nq), 1), fr3(e["ESTAR_sum"], nq * npart),
                                                      rqf(e["share_of_PSTAR_inside_ESTAR_mean_f64"], 3)))
        md.append("| %s | %s |" % (c, " | ".join(cells_)))
    LOAD_SHOW = (list(RT3.REF_LOAD) + list(RT3.BN_LOAD) + [("SDE", "BPI_SDIV"), ("SDK", "BPI_SDIV"), ("T3E", "BPI_TOP3")]
                 + [("MXS_L4", "MA_%s_L4" % k) for k, _ in RT3.CSRC] + [("UM_%s_L4" % k, "UM_%s_L4" % k) for k, _ in RT3.CSRC])
    md += ["", "Load (per-partition contacts over the population): largest contact rate / peak over mean / share of all contacts on "
           "the top 1% of partitions / Gini (exact integer profiles; unrouted = 1 / 1 / the top-1% partition share / 0)", "",
           "| arm | " + " | ".join(C) + " |", "|---|" + "---|" * len(C)]
    for rk, cn in LOAD_SHOW:
        cells_ = []
        for ds, c in CELL_ORDER:
            e = R3["cells"][c]["load (exact integer profile)"]["%s|%s" % (rk, cn)]
            cells_.append("%s / %s / %s / %s" % (sh(e["largest"], e["nq"]), rh(Fraction(e["largest"] * e["npart"], e["total_contacts"]), 1),
                                                sh(e["top_1pct_sum"], e["total_contacts"]), rh(Fraction(e["gini_num"], e["gini_den"]), 3)))
        md.append("| %s|%s | %s |" % (rk, cn, " | ".join(cells_)))
    md += ["", "Router state (static) and router reads per query (the 200 seeds): node-row entries vs partition sketch entries", "",
           "| cell | node-row entries | union sketch (ratio) | per-family sketch sum | co-partitioned share STRUCT_out / STRUCT_in / KNN / NER | reads per query: node rows / sketch union (mean) |",
           "|---|---|---|---|---|---|"]
    for ds, c in CELL_ORDER:
        rs = R3["cells"][c]["router_state (static)"]
        a = rs["all_families"]
        cp = " / ".join("-" if not rs[f]["node_row_entries"] else fr3(rs[f]["co_partitioned_entries (inside the source's own partition)"], rs[f]["node_row_entries"])
                        for f in ("STRUCT_out", "STRUCT_in", "KNN", "NER"))
        rn = R3["datasets"][ds]["router_reads_node_rows (exact sum, n)"]
        rsk = R3["cells"][c]["router_reads_sketch (exact sums: per-family summed, union; n)"]
        md.append("| %s | %d | %d (%s) | %d | %s | %s / %s |" % (
            c, a["node_row_entries"], a["sketch_entries_union (SDIV: distinct (node, partition))"],
            fr3(a["sketch_entries_union (SDIV: distinct (node, partition))"], a["node_row_entries"]),
            a["sketch_entries_per_family_summed (SUM / NSUM multiplicities)"], cp, rh(Fraction(rn[0], rn[1]), 1), rh(Fraction(rsk[1], rsk[2]), 1)))
    md += ["", "Order length |O'| (mean, exact): " + "; ".join(
        "%s %s" % (ds, rh(Fraction(*R3["datasets"][ds]["router_order_length (exact sum, n)"]), 1)) for ds in ("metaqa", "musique", "squad")), ""]
if MD:
    assert not os.path.exists(MD)
    open(MD, "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("wrote", MD)
