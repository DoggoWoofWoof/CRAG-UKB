"""L1 DEVELOPMENT -- SCALE summary of the _l1d_scale.py records (scale_<ds>__<tag>): per dataset and cell the K-curves of IR_L1 +
ES.KNEE_SDIV, the log-log slopes over the K grid, structural vs balanced random at the same K, and the K* cost components.  Every
number is recomputed from the stored per-query arrays (exact integer counts; Decimal ROUND_HALF_UP: shares 3 decimals, means 1,
ratios and slopes 2; an exact zero is written 0).  Written before the full records were read.  Descriptive only.

  curves    per cell: B_P(q) (mean / median / p90), rho = B_P / K (mean), CMASS / N (mean), R_ALL(B_N) of ES.KNEE_SDIV and
            S.KNEE_SDIV with the delta against unrouted, lost reach / crowding, NGP, LO_ES, PSTAR_M / ESTAR_M, ES|ESTAR_M, NSEEDP,
            NEV, XC share, router reads, the smallest fixed B_P of the ES surface within 0.01 of / at or above unrouted, load
  RAND      the mean over the seeds (sums over seeds / (seeds x rows)) with the per-seed min and max
  slopes    least squares of log(y) on log(K) over the grid K present (six unless a structural cell is absent; then RAND is also
            fitted over the same K; the native K is off-grid and not used), y = mean B_P, NGP, LO_ES, PSTAR_M, ESTAR_M, the smallest
            fixed B_P, NSEEDP, CMASS; plus y(5000) / y(500) against K's own 10x
  structural vs random   at each K and B_N: paired ALL of ES.KNEE_SDIV against each seed (McNemar, descriptive)
  K*        no weights are invented: per cell the three cost components (fan-out B_P, local scan CMASS, communication NSEEDP /
            XC) with R_ALL(B_N) - unrouted; the cells meeting R_ALL >= unrouted - tol (tol 0.01 and 0) and their non-dominated set
            on (mean B_P, mean CMASS, mean NSEEDP); per partitioner, the lower convex hull of the feasible cells on (mean B_P,
            mean CMASS) with the range of c (one shard contact priced as c scanned nodes) over which each hull cell minimises
            B_P * c + CMASS -- the reader supplies c

Usage: python scratchpad/_l1d_scale_summary.py SUMMARY <tag> [--dir=<records dir> --out=<file outside the repository>]
       -> results/L1_DEV/scale_SUMMARY__<tag>.json (write-once)
"""
import io
import json
import os
import sys
import time
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP

import numpy as np

import _l1d_lib as D
import _l1d_adaptbp as AB
import _l1d_route3 as RT3
import _l1d_scale as SC

log = D.log
DSETS = ("metaqa", "musique", "squad")
MC = D.M_CURVE
M_MAIN = 1000
TOLS = ("0.01", "0")
GRID = tuple(SC.KGRID)


def dz(x):
    if isinstance(x, Decimal):
        return x
    if isinstance(x, (int, np.integer)):
        return Decimal(int(x))
    return Decimal(repr(float(x)))


def qz(x, q):
    v = dz(x)
    if v == 0:
        return "0"
    return str(v.quantize(Decimal(q), rounding=ROUND_HALF_UP))


def frac(n, d):
    return Decimal(int(n)) / Decimal(int(d))


def share(n, d):
    return qz(frac(n, d), "0.001")


def sdelta(n, d):
    v = frac(n, d)
    if v == 0:
        return "0"
    s = str(v.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return s if s.startswith("-") else "+" + s


def mean1(tot, n):
    return qz(frac(tot, n), "0.1")


def r2(x):
    return qz(x, "0.01")


def load(dirp, ds, tag):
    fj = os.path.join(dirp, "scale_%s__%s.json" % (ds, tag))
    R = json.load(io.open(fj, encoding="utf-8"))
    fz = os.path.join(dirp, R["npz"]["path"])
    assert D.sha_file(fz) == R["npz"]["sha256"], "%s changed since its record" % fz
    return fj, R, np.load(fz)


def cell_raw(R, z, c, unr, N):
    """exact per-cell sums / counts from the per-query arrays."""
    rec = R["structures"]["cells"][c]
    K = int(rec["K"])
    LO, HI, CNT = z["LO__" + c].astype(np.int64), z["HI__" + c].astype(np.int64), z["CNT__" + c].astype(np.int64)
    nq = len(CNT)
    o = {"partitioner": rec["partitioner"], "seed": rec.get("seed"), "K": K, "native": rec["native"], "nq": nq, "N": N,
         "sizes": z["SIZES__" + c].astype(np.int64)}
    o["BP_sum"], o["BP"] = int(CNT.sum()), CNT
    o["CMASS_sum"] = [int(z["CMASS__" + c][:, ri].sum()) for ri in range(2)]
    o["ALL"] = {}
    o["lost_reach"] = {}
    o["lost_crowding"] = {}
    for ri, rk in enumerate(SC.RANKS):
        o["lost_reach"][rk] = int((CNT < LO[:, ri]).sum())
        for mi, M in enumerate(MC):
            a_ = AB.served_at(LO[:, ri], HI[:, ri, mi], CNT)
            o["ALL"][(rk, M)] = a_
            o["lost_crowding"][(rk, M)] = int(((LO[:, ri] <= HI[:, ri, mi]) & (HI[:, ri, mi] < CNT)).sum())
    ES = z["ESTAR__" + c].astype(np.int64)
    PS = z["PSTAR__" + c].astype(np.int64)
    o["ESTAR_sum"] = {M: int(ES[:, mi].sum()) for mi, M in enumerate(MC)}
    o["PSTAR_sum"] = {M: int(PS[:, mi].sum()) for mi, M in enumerate(MC)}
    o["ESARM"] = {M: AB.served_at(LO[:, 0], HI[:, 0, mi], ES[:, mi]) for mi, M in enumerate(MC)}
    for nm_ in ("NGP", "NSEEDP", "NEV", "XC", "RDS"):
        o[nm_ + "_sum"] = int(z["%s__%s" % (nm_, c)].astype(np.int64).sum())
    o["LO_sum"] = {rk: int(LO[:, ri].sum()) for ri, rk in enumerate(SC.RANKS)}
    # exact count surfaces of both rankings at every B_N
    o["surf"] = {}
    for ri, rk in enumerate(SC.RANKS):
        S_ = np.rint(AB.surface(LO[:, ri], HI[:, ri], K) * nq).astype(np.int64)
        o["surf"][rk] = S_
    o["LOAD"] = z["LOAD__" + c].astype(np.int64)
    return o


def smallest_b(counts_row, n_unr, nq, tol):
    """the smallest fixed B_P whose ALL count is >= unrouted - tol x rows (exact: an integer count >= the ceiling of the bound)."""
    thr = int((Decimal(int(n_unr)) - Decimal(tol) * Decimal(int(nq))).to_integral_value(rounding=ROUND_CEILING))
    return int(np.flatnonzero(np.asarray(counts_row) >= thr)[0]) + 1


def cell_view(o, n_unr, ENT_sum, RDN_sum):
    nq, K, N = o["nq"], o["K"], o["N"]
    v = {"partitioner": o["partitioner"], "seed": o["seed"], "K": K, "native": o["native"],
         "shard_size": {"min": int(o["sizes"].min()), "max": int(o["sizes"].max()), "mean": mean1(N, K)},
         "B_P_mean": mean1(o["BP_sum"], nq), "B_P_median": float(np.median(o["BP"])),
         "B_P_p90 (nearest rank)": int(np.sort(o["BP"])[-(-9 * nq // 10) - 1]), "B_P_max": int(o["BP"].max()),
         "rho_mean": share(o["BP_sum"], nq * K), "CMASS_mean_ES": mean1(o["CMASS_sum"][0], nq),
         "CMASS_over_N_mean_ES": share(o["CMASS_sum"][0], nq * N),
         "NGP_mean": mean1(o["NGP_sum"], nq), "LO_ES_mean": mean1(o["LO_sum"]["ES"], nq), "LO_S_mean": mean1(o["LO_sum"]["S"], nq),
         "NSEEDP_mean": mean1(o["NSEEDP_sum"], nq), "NEV_mean": mean1(o["NEV_sum"], nq),
         "XC_share_pooled": share(o["XC_sum"], ENT_sum), "RDS_over_RDN_pooled": share(o["RDS_sum"], RDN_sum),
         "lost_reach_ES": share(o["lost_reach"]["ES"], nq), "lost_reach_S": share(o["lost_reach"]["S"], nq), "B_N": {}}
    for M in MC:
        a_e, a_s, a_x = o["ALL"][("ES", M)], o["ALL"][("S", M)], o["ESARM"][M]
        v["B_N"][str(M)] = {"ES|KNEE_SDIV": {"n": int(a_e.sum()), "ALL": share(a_e.sum(), nq), "delta": sdelta(int(a_e.sum()) - n_unr[M], nq),
                                             "lost_crowding": share(o["lost_crowding"][("ES", M)], nq)},
                            "S|KNEE_SDIV": {"n": int(a_s.sum()), "ALL": share(a_s.sum(), nq), "delta": sdelta(int(a_s.sum()) - n_unr[M], nq)},
                            "ES_minus_S": sdelta(int(a_e.sum()) - int(a_s.sum()), nq),
                            "ES|ESTAR_M": {"n": int(a_x.sum()), "delta": sdelta(int(a_x.sum()) - n_unr[M], nq), "ESTAR_mean": mean1(o["ESTAR_sum"][M], nq),
                                           "ESTAR_over_K_mean": share(o["ESTAR_sum"][M], nq * K)},
                            "PSTAR_mean": mean1(o["PSTAR_sum"][M], nq), "PSTAR_over_K_mean": share(o["PSTAR_sum"][M], nq * K),
                            "ES_surface": {"best_fixed": {"B_P": int(np.argmax(o["surf"]["ES"][MC.index(M)])) + 1,
                                                          "ALL": share(o["surf"]["ES"][MC.index(M)].max(), nq)},
                                           "smallest_B_P_within_0.01": smallest_b(o["surf"]["ES"][MC.index(M)], n_unr[M], nq, "0.01"),
                                           "smallest_B_P_at_or_above": smallest_b(o["surf"]["ES"][MC.index(M)], n_unr[M], nq, "0")}}
    ls = {}
    for ai, a in enumerate(SC.LOAD_ARMS):
        cnt = o["LOAD"][ai]
        tot = int(cnt.sum())
        srt = np.sort(cnt)
        i = np.arange(1, K + 1, dtype=np.float64)
        gini = float((2.0 * (i * srt).sum()) / (K * srt.sum()) - (K + 1.0) / K)
        ls[a] = {"largest_contact_rate": share(srt[-1], nq), "peak_over_mean": r2(frac(int(srt[-1]) * K, tot)),
                 "gini": qz(gini, "0.001"), "never_contacted": int((cnt == 0).sum())}
    v["load"] = ls
    return v


def slope(K, y):
    K, y = np.asarray(K, np.float64), np.asarray(y, np.float64)
    if (y <= 0).any():
        return None
    return float(np.polyfit(np.log(K), np.log(y), 1)[0])


def nondominated(rows, keys):
    out = []
    for i, a in enumerate(rows):
        dom = False
        for j, b in enumerate(rows):
            if i != j and all(b[k] <= a[k] for k in keys) and any(b[k] < a[k] for k in keys):
                dom = True
                break
        if not dom:
            out.append(a["cell"])
    return out


def break_even(rows):
    """the lower convex hull of (mean B_P, mean CMASS) over the given cells and, for each hull cell, the range of c (one shard
    contact priced as c scanned nodes) over which it minimises B_P * c + CMASS; weight-free (the reader supplies c)."""
    pts = sorted(rows, key=lambda r: (r["B_P"], r["CMASS"]))
    best = []
    for r in pts:  # keep only cells not dominated on (B_P, CMASS)
        if not best or r["CMASS"] < best[-1]["CMASS"]:
            best.append(r)
    hull = []
    for r in best:  # lower-left convex hull, B_P increasing and CMASS decreasing
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (b["B_P"] - a["B_P"]) * (r["CMASS"] - a["CMASS"]) - (b["CMASS"] - a["CMASS"]) * (r["B_P"] - a["B_P"]) <= 0:
                hull.pop()
            else:
                break
        hull.append(r)
    out = []
    for i, r in enumerate(hull):
        hi = None if i == 0 else (hull[i - 1]["CMASS"] - r["CMASS"]) / (r["B_P"] - hull[i - 1]["B_P"])
        lo = None if i == len(hull) - 1 else (r["CMASS"] - hull[i + 1]["CMASS"]) / (hull[i + 1]["B_P"] - r["B_P"])
        out.append({"cell": r["cell"], "B_P_mean": round(r["B_P"], 1), "CMASS_mean": round(r["CMASS"], 1),
                    "optimal_for_c_scanned_nodes_per_contact": [round(lo, 1) if lo is not None else 0, round(hi, 1) if hi is not None else "inf"]})
    return {"hull_from_the_fewest_shards_contacted": out, "off_hull_cells": [r["cell"] for r in pts if r not in hull]}


def summarize(dirp, ds, tag):
    fj, R, z = load(dirp, ds, tag)
    N = int(R["N"])
    gptr = z["gptr"].astype(np.int64)
    ngold = np.diff(gptr)
    nq = len(z["rows"])
    unr = {M: D.per_query(z["pos_FLATLOC__IR_L1"], M, gptr, ngold)[0] for M in MC}
    n_unr = {M: int(unr[M].sum()) for M in MC}
    ENT_sum, RDN_sum = int(z["ENT"].sum()), int(z["RDN"].sum())
    cells = [str(s) for s in z["cells"]]
    raw = {c: cell_raw(R, z, c, unr, N) for c in cells}
    views = {c: cell_view(raw[c], n_unr, ENT_sum, RDN_sum) for c in cells}
    kn = int(R["structures"]["native_K"])
    Ks = sorted(set(GRID) | {kn})
    fams = [p for p in SC.STRUCTURAL if any(raw[c]["partitioner"] == p for c in cells)]
    seeds = list(SC.RAND_SEEDS)
    # ---- RAND means over seeds (sums over seeds / (seeds x rows)) with the per-seed range
    rand = {}
    for K in Ks:
        rc = [SC.cell_name("RAND%d" % s, K) for s in seeds]
        rr = [raw[c] for c in rc]
        e = {"B_P_mean": mean1(sum(r["BP_sum"] for r in rr), len(rr) * nq),
             "B_P_mean_range": [views[c]["B_P_mean"] for c in rc],
             "rho_mean": share(sum(r["BP_sum"] for r in rr), len(rr) * nq * K),
             "CMASS_over_N_mean_ES": share(sum(r["CMASS_sum"][0] for r in rr), len(rr) * nq * N),
             "NGP_mean": mean1(sum(r["NGP_sum"] for r in rr), len(rr) * nq), "LO_ES_mean": mean1(sum(r["LO_sum"]["ES"] for r in rr), len(rr) * nq),
             "NSEEDP_mean": mean1(sum(r["NSEEDP_sum"] for r in rr), len(rr) * nq),
             "XC_share_pooled": share(sum(r["XC_sum"] for r in rr), len(rr) * ENT_sum), "B_N": {}}
        for M in MC:
            ns = [int(r["ALL"][("ES", M)].sum()) for r in rr]
            e["B_N"][str(M)] = {"ES|KNEE_SDIV_ALL_mean": share(sum(ns), len(rr) * nq), "delta_mean": sdelta(sum(ns) - len(rr) * n_unr[M], len(rr) * nq),
                                "delta_range": [sdelta(n_ - n_unr[M], nq) for n_ in ns],
                                "PSTAR_mean": mean1(sum(r["PSTAR_sum"][M] for r in rr), len(rr) * nq),
                                "ESTAR_mean": mean1(sum(r["ESTAR_sum"][M] for r in rr), len(rr) * nq),
                                "smallest_B_P_within_0.01_range": [views[c]["B_N"][str(M)]["ES_surface"]["smallest_B_P_within_0.01"] for c in rc]}
        rand[str(K)] = e
    # ---- log-log slopes over the grid K present (six unless a structural cell is absent)
    def fam_series(fam, key, Ksub=GRID):
        ks, ys = [], []
        for K in Ksub:
            if fam == "RAND":
                rr = [raw[SC.cell_name("RAND%d" % s, K)] for s in seeds]
            else:
                c = SC.cell_name(fam, K)
                if c not in raw:
                    continue
                rr = [raw[c]]
            ks.append(K)
            ys.append(key(rr))
        return ks, ys

    def fit(ks, ys):
        sl = slope(ks, ys) if len(ks) >= 2 else None
        i5, i500 = (ks.index(5000) if 5000 in ks else None), (ks.index(500) if 500 in ks else None)
        return {"K_used": ks, "slope_loglog": r2(sl) if sl is not None else None,
                "y(5000) / y(500)": r2(ys[i5] / ys[i500]) if (i5 is not None and i500 is not None and ys[i500] > 0) else None,
                "y_per_K": {str(K): round(float(y), 2) for K, y in zip(ks, ys)}}
    MET = {"B_P_mean": lambda rr: sum(r["BP_sum"] for r in rr) / float(len(rr) * nq),
           "NGP_mean": lambda rr: sum(r["NGP_sum"] for r in rr) / float(len(rr) * nq),
           "LO_ES_mean": lambda rr: sum(r["LO_sum"]["ES"] for r in rr) / float(len(rr) * nq),
           "NSEEDP_mean": lambda rr: sum(r["NSEEDP_sum"] for r in rr) / float(len(rr) * nq),
           "CMASS_mean_ES": lambda rr: sum(r["CMASS_sum"][0] for r in rr) / float(len(rr) * nq)}
    for M in MC:
        MET["PSTAR_mean@%d" % M] = (lambda M_: lambda rr: sum(r["PSTAR_sum"][M_] for r in rr) / float(len(rr) * nq))(M)
        MET["ESTAR_mean@%d" % M] = (lambda M_: lambda rr: sum(r["ESTAR_sum"][M_] for r in rr) / float(len(rr) * nq))(M)
    slopes = {}
    for fam in list(fams) + ["RAND"]:
        slopes[fam] = {mk: fit(*fam_series(fam, fn)) for mk, fn in MET.items()}
    # a structural family with an absent grid cell: RAND over the same K, so the two slopes are comparable
    ksub = {}
    for fam in fams:
        kf = fam_series(fam, MET["B_P_mean"])[0]
        if kf != list(GRID):
            ksub[fam] = kf
            slopes["RAND_on_the_K_of_%s" % fam] = {mk: fit(*fam_series("RAND", fn, kf)) for mk, fn in MET.items()}
    # the fixed-fan-out need of the ES surface (smallest b within 0.01 of unrouted), per family
    need = {}
    for fam in list(fams) + ["RAND"] + ["RAND_on_the_K_of_%s" % f for f in ksub]:
        d_ = {}
        for M in MC:
            ks, ys = [], []
            for K in (ksub[fam[len("RAND_on_the_K_of_"):]] if fam.startswith("RAND_on_") else GRID):
                if fam.startswith("RAND"):
                    y = float(np.mean([views[SC.cell_name("RAND%d" % s, K)]["B_N"][str(M)]["ES_surface"]["smallest_B_P_within_0.01"] for s in seeds]))
                else:
                    c = SC.cell_name(fam, K)
                    if c not in views:
                        continue
                    y = float(views[c]["B_N"][str(M)]["ES_surface"]["smallest_B_P_within_0.01"])
                ks.append(K)
                ys.append(y)
            d_[str(M)] = fit(ks, ys)
        need[fam] = d_
    # ---- structural vs random at the same K
    svr = {}
    for fam in fams:
        for K in Ks:
            c = SC.cell_name(fam, K)
            if c not in raw:
                continue
            e = {}
            for M in MC:
                a_ = raw[c]["ALL"][("ES", M)]
                per = {}
                for s in seeds:
                    b_ = raw[SC.cell_name("RAND%d" % s, K)]["ALL"][("ES", M)]
                    p = D.paired(b_, a_)
                    per["RAND%d" % s] = {"structural_minus_random": sdelta(int(a_.sum()) - int(b_.sum()), nq), "gained": p["gained"], "lost": p["lost"], "p": p["p"]}
                e[str(M)] = per
            e["B_P_mean"] = {"structural": views[c]["B_P_mean"], "random_mean": rand[str(K)]["B_P_mean"]}
            e["NGP_mean"] = {"structural": views[c]["NGP_mean"], "random_mean": rand[str(K)]["NGP_mean"]}
            e["LO_ES_mean"] = {"structural": views[c]["LO_ES_mean"], "random_mean": rand[str(K)]["LO_ES_mean"]}
            svr[c] = e
    # ---- K* components (no weights) and the non-dominated feasible cells
    kstar = {}
    for M in MC:
        rows = []
        for c in cells:
            o = raw[c]
            rows.append({"cell": c, "delta": frac(int(o["ALL"][("ES", M)].sum()) - n_unr[M], nq), "B_P": o["BP_sum"] / float(nq),
                         "CMASS": o["CMASS_sum"][0] / float(nq), "NSEEDP": o["NSEEDP_sum"] / float(nq)})
        e = {}
        for tol in TOLS:
            feas = [r for r in rows if r["delta"] >= -Decimal(tol)]
            e["R_ALL >= unrouted - %s" % tol] = {"feasible_cells": [r["cell"] for r in feas],
                                                 "non_dominated (B_P, CMASS, NSEEDP)": nondominated(feas, ("B_P", "CMASS", "NSEEDP"))}
            # per partitioner family: the break-even contact cost at which each feasible K minimises B_P * c + CMASS
            be = {}
            for fam in list(fams) + ["RAND%d" % s for s in seeds]:
                fr_ = [r for r in feas if r["cell"].startswith(fam + "_")]
                be[fam] = break_even(fr_) if fr_ else None
            e["R_ALL >= unrouted - %s" % tol]["break_even_contact_cost (c scanned nodes per shard contact; B_P * c + CMASS)"] = be
        kstar[str(M)] = e
    comp = {c: {"B_P_mean": views[c]["B_P_mean"], "CMASS_mean_ES": views[c]["CMASS_mean_ES"], "NSEEDP_mean": views[c]["NSEEDP_mean"],
                "XC_share_pooled": views[c]["XC_share_pooled"],
                "cross_partition_share_of_E_entries (static)": R["structures"]["cells"][c]["cross_partition_share_of_E_entries"],
                "delta_ES|KNEE_SDIV": {str(M): views[c]["B_N"][str(M)]["ES|KNEE_SDIV"]["delta"] for M in MC}} for c in cells}
    # ---- per hop at M_MAIN (MetaQA / MuSiQue)
    hops = z["HOPS"].astype(np.int64)
    ph = {}
    if (hops >= 0).any() and len(set(hops.tolist())) > 1:
        for h in sorted(set(hops.tolist())):
            m = hops == h
            ph["hop%d" % h] = {"n": int(m.sum()), "unrouted": int(unr[M_MAIN][m].sum()),
                               "cells": {c: int(raw[c]["ALL"][("ES", M_MAIN)][m].sum()) for c in cells}}
    return {"dataset": ds, "record": {"path": D.rel(fj) if fj.lower().startswith(D.REPO.lower()) else fj, "sha256": D.sha_file(fj),
                                      "npz_sha256": R["npz"]["sha256"]},
            "N": N, "n_rows": nq, "native_K": kn, "unrouted_n": {str(M): n_unr[M] for M in MC},
            "unrouted_ALL": {str(M): share(n_unr[M], nq) for M in MC}, "absent_cells": R["structures"]["absent_cells"],
            "cells": views, "RAND_mean_over_seeds": rand, "slopes_over_grid_K": slopes, "fixed_fan_out_need_ES_surface": need,
            "structural_vs_random": svr, "K_star_components": comp, "K_star_feasible_non_dominated": kstar, "per_hop_at_%d" % M_MAIN: ph}


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 2 and a[0] == "SUMMARY", "usage: SUMMARY <tag> [--dir=<records dir> --out=<file outside the repository>]"
    tag = a[1]
    dirp = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--dir=")), D.OUT)
    outp = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--out=")), None)
    if outp is None:
        outp = os.path.join(D.OUT, "scale_SUMMARY__%s.json" % tag)
    else:
        assert not os.path.abspath(outp).lower().startswith(os.path.abspath(D.REPO).lower())
    assert not os.path.exists(outp), "write-once: %s exists" % outp
    t0 = time.time()
    res = {"status": "DEVELOPMENT (descriptive; p-values descriptive)", "definitions": __doc__, "tag": tag,
           "rounding": "Decimal ROUND_HALF_UP from exact counts: shares 3 decimals, means 1, ratios / slopes 2; exact zero written 0",
           "code": {"summary": {"path": "scratchpad/_l1d_scale_summary.py", "sha256": D.sha_file(os.path.abspath(__file__))},
                    "harness": {"path": "scratchpad/_l1d_scale.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_scale.py"))}},
           "datasets": {}}
    for ds in DSETS:
        fj = os.path.join(dirp, "scale_%s__%s.json" % (ds, tag))
        if not os.path.exists(fj):
            log("%s: no record at %s" % (ds, fj))
            continue
        res["datasets"][ds] = summarize(dirp, ds, tag)
        log("%s summarized" % ds)
    assert res["datasets"], "no records"
    res["seconds"] = round(time.time() - t0, 1)
    D.G.S.wj(outp, res)
    log("-> %s sha256 %s" % (outp, D.sha_file(outp)[:16]))


if __name__ == "__main__":
    main()
