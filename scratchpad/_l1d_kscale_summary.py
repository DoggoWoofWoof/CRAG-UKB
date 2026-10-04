"""L1 DEVELOPMENT -- KSCALE summary of the _l1d_kscale.py records (kscale_<ds>__<tag>): per dataset, partitioner family and arm
(section 35's ES.KNEE_SDIV on the K map and the four K-aware arms B_P = ceil(KNEE_SDIV(q; K = 100 map) * (K / 100)^alpha)) the
K-curves of R_ALL(B_N) - unrouted, B_P, rho = B_P / K, CMASS / N and the load; the log-log slopes of the mean B_P over the K grid;
per family and B_N the worst delta over K and the arms holding delta >= -tol at every K; the fixed-fan-out need of the ES surface
against each arm's mean B_P; structural vs balanced random under each arm; the feasible cells and the B_P * c + CMASS hull per arm.
Every number is recomputed from the stored per-query arrays (exact integer counts; Decimal ROUND_HALF_UP through
_l1d_scale_summary's helpers: shares 3 decimals, means 1, ratios and slopes 2; an exact zero is written 0) and the per-cell counts
are asserted equal to the harness's own diagnostics.  Written before the full records were read.  Descriptive only.

  RAND     the mean over the seeds (sums over seeds / (seeds x rows)) with the per-seed values
  holds    an arm holds tol at B_N on a family if its ALL count >= unrouted - tol x rows at every K considered: K >= 250 (every
           arm equals ES.KNEE_SDIV at K = 100 by construction) and every K; the grid K plus the native K; a structural family
           skips its absent cells; RAND: on the mean over the seeds and on every seed
  slopes   least squares of log(mean B_P) on log(K) over the grid K present (the native K is off-grid); RAND also over the K of a
           family with an absent grid cell; ES.KNEE_SDIV's effective exponent log(B_P(K) / B_P(100)) / log(K / 100) per K
  need     the smallest fixed B_P of the ES surface whose ALL count >= unrouted - tol x rows (tol 0.01 and 0), per K
  rho      whether the mean rho = B_P / K decreases strictly along the grid K

Usage: python scratchpad/_l1d_kscale_summary.py SUMMARY <tag> [--dir=<records dir> --out=<file outside the repository>]
       -> results/L1_DEV/kscale_SUMMARY__<tag>.json (write-once)
"""
import io
import json
import math
import os
import sys
import time
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

import _l1d_lib as D
import _l1d_adaptbp as AB
import _l1d_scale as SC
import _l1d_scale_summary as SS
import _l1d_kscale as KS

log = D.log
DSETS = SS.DSETS
MC = D.M_CURVE
M_MAIN = 1000
TOLS = ("0.01", "0.005", "0")
KSTAR_TOLS = ("0.01", "0")
GRID = SS.GRID
ARMS = KS.ARM_ORDER
BASE = KS.BASE
K_FROM = 250
SEEDS = list(SC.RAND_SEEDS)


def load(dirp, ds, tag):
    fj = os.path.join(dirp, "kscale_%s__%s.json" % (ds, tag))
    R = json.load(io.open(fj, encoding="utf-8"))
    fz = os.path.join(dirp, R["npz"]["path"])
    assert D.sha_file(fz) == R["npz"]["sha256"], "%s changed since its record" % fz
    return fj, R, np.load(fz)


def sdec(v):
    """a signed exact Decimal difference, 3 decimals ROUND_HALF_UP (_l1d_scale_summary.sdelta's format); exact zero -> 0."""
    if v == 0:
        return "0"
    s = str(v.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return s if s.startswith("-") else "+" + s


def per_arm(z, c, base_key, alpha_key, axis):
    out = {BASE: z[base_key + c].astype(np.int64)}
    A = z[alpha_key + c].astype(np.int64)
    for ai, a in enumerate(KS.ALPHAS):
        out[KS.ANAME[a]] = A[:, ai] if axis == 1 else A[ai]
    return out


def cell_raw(R, z, c, unr, n_unr, N):
    """exact per-cell, per-arm sums / counts from the per-query arrays (and the harness's own counts, asserted equal)."""
    rec = R["structures"]["cells"][c]
    K = int(rec["K"])
    LO, HI = z["LOES__" + c].astype(np.int64), z["HIES__" + c].astype(np.int64)
    nq = len(LO)
    for mi, M in enumerate(MC):
        assert (AB.served_at(LO, HI[:, mi], K) == unr[M]).all(), "every partition != unrouted (%s %d)" % (c, M)
    B = per_arm(z, c, "CNT__", "CNTA__", 1)
    CM = per_arm(z, c, "CMASSES__", "CMASSA__", 1)
    LD = per_arm(z, c, "LOADES__", "LOADA__", 0)
    dg = R["diagnostics"]["cells"][c]
    o = {"partitioner": rec["partitioner"], "seed": rec.get("seed"), "K": K, "native": rec["native"], "nq": nq, "N": N, "arms": {}}
    base_serv = {M: AB.served_at(LO, HI[:, mi], B[BASE]) for mi, M in enumerate(MC)}
    for an in ARMS:
        b = B[an]
        e = {"BP_sum": int(b.sum()), "BP": b, "CMASS_sum": int(CM[an].sum()), "lost_reach": int((b < LO).sum()), "LOAD": LD[an],
             "ALL": {}, "lost_crowding": {}, "vs_base": {}}
        assert e["BP_sum"] == dg["arms"][an]["B_P_sum"] and e["CMASS_sum"] == dg["arms"][an]["CMASS_sum"]
        for mi, M in enumerate(MC):
            a_ = AB.served_at(LO, HI[:, mi], b)
            e["ALL"][M] = a_
            e["lost_crowding"][M] = int(((LO <= HI[:, mi]) & (HI[:, mi] < b)).sum())
            e["vs_base"][M] = D.paired(base_serv[M], a_)
            h = dg["arms"][an]["B_N"][str(M)]
            assert h["n_ALL"] == int(a_.sum()) and h["n_unrouted"] == n_unr[M] and h["n_lost_reach (B_P < LO)"] == e["lost_reach"]
            assert h["n_lost_crowding (LO <= HI < B_P)"] == e["lost_crowding"][M]
        o["arms"][an] = e
    S_ = np.rint(AB.surface(LO, HI, K) * nq).astype(np.int64)
    for mi, M in enumerate(MC):
        assert S_[mi, K - 1] == n_unr[M]
    o["surf"] = S_
    return o


def load_view(cnt, nq, K):
    tot = int(cnt.sum())
    top = int(cnt.max())
    return {"largest_contact_rate": SS.share(top, nq), "peak_over_mean": SS.r2(SS.frac(top * K, tot)), "never_contacted": int((cnt == 0).sum())}


def arm_view(o, an, n_unr):
    nq, K, N = o["nq"], o["K"], o["N"]
    e = o["arms"][an]
    n_base = {M: int(o["arms"][BASE]["ALL"][M].sum()) for M in MC}
    v = {"B_P_mean": SS.mean1(e["BP_sum"], nq), "B_P_median": float(np.median(e["BP"])),
         "B_P_p90 (nearest rank)": int(np.sort(e["BP"])[-(-9 * nq // 10) - 1]), "B_P_max": int(e["BP"].max()),
         "rho_mean": SS.share(e["BP_sum"], nq * K), "CMASS_over_N_mean": SS.share(e["CMASS_sum"], nq * N), "lost_reach": e["lost_reach"],
         "load": load_view(e["LOAD"], nq, K), "B_N": {}}
    for M in MC:
        n = int(e["ALL"][M].sum())
        v["B_N"][str(M)] = {"n": n, "ALL": SS.share(n, nq), "delta": SS.sdelta(n - n_unr[M], nq), "lost_crowding": e["lost_crowding"][M]}
        if an != BASE:
            p = e["vs_base"][M]
            v["B_N"][str(M)]["minus_base"] = SS.sdelta(n - n_base[M], nq)
            v["B_N"][str(M)]["vs_base"] = {"gained": p["gained"], "lost": p["lost"], "p": p["p"]}
    return v


def summarize(dirp, ds, tag):
    fj, R, z = load(dirp, ds, tag)
    assert [str(s) for s in z["alphas"]] == list(KS.ALPHAS)
    N = int(R["N"])
    gptr = z["gptr"].astype(np.int64)
    ngold = np.diff(gptr)
    nq = len(z["rows"])
    unr = {M: D.per_query(z["pos_FLATLOC__IR_L1"], M, gptr, ngold)[0] for M in MC}
    n_unr = {M: int(unr[M].sum()) for M in MC}
    cells = [str(s) for s in z["cells"]]
    raw = {c: cell_raw(R, z, c, unr, n_unr, N) for c in cells}
    views = {c: {"partitioner": raw[c]["partitioner"], "seed": raw[c]["seed"], "K": raw[c]["K"], "native": raw[c]["native"],
                 "reference_cell": R["structures"]["kscale"]["reference_cells"][c],
                 "arms": {an: arm_view(raw[c], an, n_unr) for an in ARMS}} for c in cells}
    kn = int(R["structures"]["native_K"])
    Ks = sorted(set(GRID) | {kn})
    fams = [p for p in SC.STRUCTURAL if any(raw[c]["partitioner"] == p for c in cells)]

    def fam_cells(fam, K):
        if fam == "RAND":
            return [SC.cell_name("RAND%d" % s, K) for s in SEEDS]
        c = SC.cell_name(fam, K)
        return [c] if c in raw else []

    def agg(fam, K, an):
        cs = fam_cells(fam, K)
        if not cs:
            return None
        k = len(cs)
        rr = [raw[c]["arms"][an] for c in cs]
        bps = sum(r["BP_sum"] for r in rr)
        e = {"cells": cs, "B_P_mean": SS.mean1(bps, k * nq), "_bp": bps / float(k * nq), "rho_mean": SS.share(bps, k * nq * K),
             "_rho": SS.frac(bps, k * nq * K), "CMASS_over_N_mean": SS.share(sum(r["CMASS_sum"] for r in rr), k * nq * N),
             "lost_reach_mean": SS.mean1(sum(r["lost_reach"] for r in rr), k),
             "peak_over_mean": [load_view(r["LOAD"], nq, K)["peak_over_mean"] for r in rr], "B_N": {}, "_d": {}, "_dseed": {}}
        for M in MC:
            ns = [int(r["ALL"][M].sum()) for r in rr]
            nb = [int(raw[c]["arms"][BASE]["ALL"][M].sum()) for c in cs]
            e["_d"][M] = SS.frac(sum(ns) - k * n_unr[M], k * nq)
            e["_dseed"][M] = [SS.frac(n_ - n_unr[M], nq) for n_ in ns]
            b = {"delta": SS.sdelta(sum(ns) - k * n_unr[M], k * nq), "lost_crowding_mean": SS.mean1(sum(r["lost_crowding"][M] for r in rr), k)}
            if k > 1:
                b["delta_per_seed"] = [SS.sdelta(n_ - n_unr[M], nq) for n_ in ns]
            if an != BASE:
                b["minus_base"] = SS.sdelta(sum(ns) - sum(nb), k * nq)
            e["B_N"][str(M)] = b
        return e

    AGG = {fam: {an: {K: agg(fam, K, an) for K in Ks} for an in ARMS} for fam in list(fams) + ["RAND"]}
    # ---- K-curves
    curves = {}
    for fam, d_ in AGG.items():
        curves[fam] = {}
        for an, byK in d_.items():
            curves[fam][an] = {str(K): {k_: v_ for k_, v_ in e.items() if not k_.startswith("_")} for K, e in byK.items() if e is not None}
    # ---- holds / worst delta over K (K >= K_FROM and every K)
    holds = {}
    for fam, d_ in AGG.items():
        holds[fam] = {}
        for M in MC:
            hm = {}
            for rng, kmin in (("K>=%d" % K_FROM, K_FROM), ("every K", 0)):
                hr = {}
                for an in ARMS:
                    es = [(K, e) for K, e in sorted(d_[an].items()) if e is not None and K >= kmin]
                    worst_K, worst = min(((K, e["_d"][M]) for K, e in es), key=lambda t: (t[1], t[0]))
                    x = {"K_considered": [K for K, _ in es], "worst_delta": sdec(worst), "worst_at_K": worst_K,
                         "holds": {tol: all(e["_d"][M] >= -Decimal(tol) for _, e in es) for tol in TOLS}}
                    if fam == "RAND":
                        x["holds_every_seed"] = {tol: all(all(v >= -Decimal(tol) for v in e["_dseed"][M]) for _, e in es) for tol in TOLS}
                    hr[an] = x
                smallest = {}
                for tol in TOLS:
                    ok = [a for a in KS.ALPHAS if hr[KS.ANAME[a]]["holds"][tol]]
                    smallest[tol] = {"alphas_holding": ok, "smallest_alpha": ok[0] if ok else None, "base_holds": hr[BASE]["holds"][tol]}
                hm[rng] = {"arms": hr, "by_tol": smallest}
            holds[fam][str(M)] = hm
    # ---- slopes of the mean B_P over the grid K present; rho monotone; the base's effective exponent
    ksub = {}
    for fam in fams:
        kf = [K for K in GRID if AGG[fam][BASE][K] is not None]
        if kf != list(GRID):
            ksub[fam] = kf
    slopes = {}
    for fam in list(fams) + ["RAND"] + ["RAND_on_the_K_of_%s" % f for f in ksub]:
        src = "RAND" if fam.startswith("RAND") else fam
        kl = ksub[fam[len("RAND_on_the_K_of_"):]] if fam.startswith("RAND_on_") else [K for K in GRID if AGG[src][BASE][K] is not None]
        e = {}
        for an in ARMS:
            ys = [AGG[src][an][K]["_bp"] for K in kl]
            rh = [AGG[src][an][K]["_rho"] for K in kl]
            sl = SS.slope(kl, ys)
            i5, i500 = (kl.index(5000) if 5000 in kl else None), (kl.index(500) if 500 in kl else None)
            e[an] = {"K_used": kl, "slope_loglog_B_P_mean": SS.r2(sl) if sl is not None else None,
                     "B_P(5000) / B_P(500)": SS.r2(ys[i5] / ys[i500]) if (i5 is not None and i500 is not None) else None,
                     "rho_strictly_decreasing_over_K_used": all(rh[i + 1] < rh[i] for i in range(len(rh) - 1))}
        slopes[fam] = e
    eff = {}
    for fam in list(fams) + ["RAND"]:
        b100 = AGG[fam][BASE][100]["_bp"]
        eff[fam] = {str(K): SS.r2(math.log(AGG[fam][BASE][K]["_bp"] / b100) / math.log(K / 100.0))
                    for K in Ks if K > 100 and AGG[fam][BASE][K] is not None}
    # ---- the fixed-fan-out need of the ES surface against each arm's mean B_P
    need = {}
    for fam in list(fams) + ["RAND"]:
        need[fam] = {}
        for K in Ks:
            cs = fam_cells(fam, K)
            if not cs:
                continue
            e = {}
            for M in MC:
                mi = MC.index(M)
                nd = {tol: [SS.smallest_b(raw[c]["surf"][mi], n_unr[M], nq, tol) for c in cs] for tol in ("0.01", "0")}
                x = {"need_within_0.01": nd["0.01"][0] if len(cs) == 1 else SS.mean1(sum(nd["0.01"]), len(cs)),
                     "need_at_or_above": nd["0"][0] if len(cs) == 1 else SS.mean1(sum(nd["0"]), len(cs))}
                if len(cs) > 1:
                    x["need_within_0.01_per_seed"], x["need_at_or_above_per_seed"] = nd["0.01"], nd["0"]
                x["arm_B_P_mean_over_need_within_0.01"] = {an: SS.r2(AGG[fam][an][K]["_bp"] / (sum(nd["0.01"]) / float(len(cs)))) for an in ARMS}
                e[str(M)] = x
            need[fam][str(K)] = e
    # ---- structural vs random under each arm
    svr = {}
    for fam in fams:
        for K in Ks:
            c = SC.cell_name(fam, K)
            if c not in raw:
                continue
            e = {}
            for an in ARMS:
                a_s = raw[c]["arms"][an]
                rs = [raw[SC.cell_name("RAND%d" % s, K)]["arms"][an] for s in SEEDS]
                x = {"B_P_mean_structural_over_random": SS.r2(SS.frac(a_s["BP_sum"] * len(rs), sum(r["BP_sum"] for r in rs))), "B_N": {}}
                for M in MC:
                    per = {}
                    for s, r in zip(SEEDS, rs):
                        p = D.paired(r["ALL"][M], a_s["ALL"][M])
                        per["RAND%d" % s] = {"structural_minus_random": SS.sdelta(int(a_s["ALL"][M].sum()) - int(r["ALL"][M].sum()), nq),
                                             "gained": p["gained"], "lost": p["lost"], "p": p["p"]}
                    x["B_N"][str(M)] = per
                e[an] = x
            svr[c] = e
    # ---- per arm: the cells within tol of unrouted and the B_P * c + CMASS hull per family (c = scanned nodes per shard contact)
    kstar = {}
    for an in ARMS:
        kstar[an] = {}
        for M in MC:
            rows = [{"cell": c, "delta": SS.frac(int(raw[c]["arms"][an]["ALL"][M].sum()) - n_unr[M], nq),
                     "B_P": raw[c]["arms"][an]["BP_sum"] / float(nq), "CMASS": raw[c]["arms"][an]["CMASS_sum"] / float(nq)} for c in cells]
            e = {}
            for tol in KSTAR_TOLS:
                feas = [r for r in rows if r["delta"] >= -Decimal(tol)]
                be = {}
                for fam in list(fams) + ["RAND%d" % s for s in SEEDS]:
                    fr_ = [r for r in feas if r["cell"].startswith(fam + "_")]
                    be[fam] = SS.break_even(fr_) if fr_ else None
                e["R_ALL >= unrouted - %s" % tol] = {"feasible_cells": [r["cell"] for r in feas], "break_even_contact_cost": be}
            kstar[an][str(M)] = e
    # ---- per hop at M_MAIN
    hops = z["HOPS"].astype(np.int64)
    ph = {}
    if (hops >= 0).any() and len(set(hops.tolist())) > 1:
        for h in sorted(set(hops.tolist())):
            m = hops == h
            ph["hop%d" % h] = {"n": int(m.sum()), "unrouted": int(unr[M_MAIN][m].sum()),
                               "cells": {c: {an: int(raw[c]["arms"][an]["ALL"][M_MAIN][m].sum()) for an in ARMS} for c in cells}}
    return {"dataset": ds, "record": {"path": D.rel(fj) if fj.lower().startswith(D.REPO.lower()) else fj, "sha256": D.sha_file(fj),
                                      "npz_sha256": R["npz"]["sha256"]},
            "N": N, "n_rows": nq, "native_K": kn, "K_REF": KS.K_REF, "alphas": list(KS.ALPHAS), "arms": list(ARMS),
            "multiplier_tables": R["structures"]["kscale"]["multiplier_tables"],
            "unrouted_n": {str(M): n_unr[M] for M in MC}, "unrouted_ALL": {str(M): SS.share(n_unr[M], nq) for M in MC},
            "absent_cells": R["structures"]["absent_cells"], "cells": views, "curves": curves, "holds_over_K": holds,
            "slopes_over_grid_K": slopes, "base_effective_exponent_from_K100": eff, "fixed_fan_out_need_ES_surface": need,
            "structural_vs_random": svr, "K_star_feasible_and_hull": kstar, "per_hop_at_%d" % M_MAIN: ph}


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    assert len(a) == 2 and a[0] == "SUMMARY", "usage: SUMMARY <tag> [--dir=<records dir> --out=<file outside the repository>]"
    tag = a[1]
    dirp = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--dir=")), D.OUT)
    outp = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--out=")), None)
    if outp is None:
        outp = os.path.join(D.OUT, "kscale_SUMMARY__%s.json" % tag)
    else:
        assert not os.path.abspath(outp).lower().startswith(os.path.abspath(D.REPO).lower())
    assert not os.path.exists(outp), "write-once: %s exists" % outp
    t0 = time.time()
    res = {"status": "DEVELOPMENT (descriptive; p-values descriptive)", "definitions": __doc__, "tag": tag,
           "rounding": "Decimal ROUND_HALF_UP from exact counts: shares 3 decimals, means 1, ratios / slopes 2; exact zero written 0",
           "code": {"summary": {"path": "scratchpad/_l1d_kscale_summary.py", "sha256": D.sha_file(os.path.abspath(__file__))},
                    "harness": {"path": "scratchpad/_l1d_kscale.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_kscale.py"))},
                    "helpers": {"path": "scratchpad/_l1d_scale_summary.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_scale_summary.py"))},
                    "scale_harness": {"path": "scratchpad/_l1d_scale.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_scale.py"))}},
           "datasets": {}}
    for ds in DSETS:
        fj = os.path.join(dirp, "kscale_%s__%s.json" % (ds, tag))
        if not os.path.exists(fj):
            log("%s: no record at %s" % (ds, fj))
            continue
        R_ = json.load(io.open(fj, encoding="utf-8"))
        assert R_["code"]["harness"]["sha256"] == res["code"]["harness"]["sha256"], "%s: harness differs from its record" % ds
        res["datasets"][ds] = summarize(dirp, ds, tag)
        log("%s summarized" % ds)
    assert res["datasets"], "no records"
    res["seconds"] = round(time.time() - t0, 1)
    D.G.S.wj(outp, res)
    log("-> %s sha256 %s" % (outp, D.sha_file(outp)[:16]))


if __name__ == "__main__":
    main()
