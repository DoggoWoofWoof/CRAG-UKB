"""Summary of the ADAPTBP development runs (_l1d_adaptbp.py), read-only over <dir>/adaptbp_<dataset>__<tag>.{json,npz}.
It reports, per cell (metaqa, metaqa_phg, musique, squad, squad_phg):
  - the adaptive fan-out B_P(q) = ceil(N_eff) / ceil(exp H) of every aggregation: its distribution (and per hop);
  - ALL gold at every B_N for each adaptive arm, next to unrouted, the same ranking's fixed-B_P curve at the arm's mean fan-out,
    the S ranking's fixed curve at that fan-out, the best fixed B_P of the same ranking and the best fixed point of ANY ranking
    (the gold-chosen envelope of the surfaces), and the smallest fixed B_P of the same ranking that reaches the arm's ALL;
  - the worst cell of each arm (the universality view: one rule on every cell at once);
  - the surfaces R(B_P, B_N) of the rankings S / SUM / MAX / TOP3 and the user's example points;
  - the deployment caps min(B_P^max, B_P(q)); the gold fan-out diagnostic; latency.
Usage: python scratchpad/_l1d_adaptbp_summary.py <tag> [--dir=<dir>] [--md=<markdown path>]
       -> <dir>/adaptbp_SUMMARY__<tag>.json (write-once) + markdown.
DEVELOPMENT numbers (user rulings 2026-09-26/27): descriptive, no verdicts."""
import json
import os
import sys

import numpy as np

import _l1d_lib as D
import _l1d_adaptbp as AB

TAG = sys.argv[1]
OUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), D.OUT)
MD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--md=")), None)
MS_SHOW = (100, 500, 1000, 2000, 5000)                                   # the user's budget list
PRIMARY_ARMS = ["OWN|%s|NEFF" % a for a in AB.PRIMARY_AGGS] + ["OWN|%s|EXPH" % a for a in AB.PRIMARY_AGGS]
REF_ARMS = ["S|%s|NEFF" % a for a in AB.PRIMARY_AGGS]
SENS_ARMS = ["OWN|TOP%d|NEFF" % k for k in AB.TOPKS if k != AB.TOPK_PRIMARY]
SHOW_ARMS = PRIMARY_ARMS + REF_ARMS + SENS_ARMS
SURF_RANKS = ("S", "SUM", "MAX", "TOP%d" % AB.TOPK_PRIMARY)
BP_TAB = (1, 2, 5, 10, 20, 50, 100, 200)
CELL_ORDER = [("metaqa", "metaqa"), ("metaqa", "metaqa_phg"), ("musique", "musique"), ("squad", "squad"), ("squad", "squad_phg")]


def f3(x):
    return "-" if x is None else "%.3f" % x


def pv(p):
    return "-" if p is None else ("%.1e" % p if p < 1e-3 else "%.3f" % p)


summ = {"stem": "adaptbp", "tag": TAG, "status": "DEVELOPMENT (descriptive; not confirmatory)", "inputs": {}, "cells": {}}
REC, Z = {}, {}
shas = set()
for ds in ("metaqa", "musique", "squad"):
    fj = os.path.join(OUT, "adaptbp_%s__%s.json" % (ds, TAG))
    r = json.load(open(fj, encoding="utf-8"))
    fz = os.path.join(OUT, r["npz"]["path"])
    assert D.sha_file(fz) == r["npz"]["sha256"], "npz changed: %s" % fz
    REC[ds], Z[ds] = r, np.load(fz)
    shas.add(r["code"]["harness"]["sha256"])
    summ["inputs"][ds] = {"json": {"path": D.rel(fj), "sha256": D.sha_file(fj)}, "npz": {"path": D.rel(fz), "sha256": r["npz"]["sha256"]},
                          "n_rows": r["n_rows"], "n_gold_nodes": r["n_gold_nodes"], "identity": r["diagnostics"]["identity"],
                          "v1_check": r["v1_check"], "seconds": r["seconds"], "peak_rss_mb": r["process_peak_rss_mb"],
                          "host_at_start": r["host_at_start"]}
assert len(shas) == 1, "the three records were written by different harness versions"
summ["harness_sha256"] = shas.pop()
assert summ["harness_sha256"] == D.sha_file(os.path.join(D.HERE, "_l1d_adaptbp.py")), "imported _l1d_adaptbp.py differs from the records' harness"
md = ["# ADAPTBP summary (%s)" % TAG, "", "harness sha %s" % summ["harness_sha256"], ""]

univ = {a: {} for a in SHOW_ARMS}
for ds, c in CELL_ORDER:
    r, z = REC[ds], Z[ds]
    dg = r["diagnostics"]
    npart = r["structures"]["cells"][c]["npart"]
    ranks = [str(s) for s in z["ranks"]]
    arms = [str(s) for s in z["arms"]]
    assert ranks == list(AB.RANKS) and arms == AB.ARM_NAMES and [int(m) for m in z["m_curve"]] == list(D.M_CURVE)
    LO, HI, BPQ = z["LO__" + c].astype(np.int64), z["HI__" + c].astype(np.int64), z["BPQ__" + c].astype(np.int64)
    nq = LO.shape[0]
    SF = {rk: AB.surface(LO[:, ri], HI[:, ri], npart) for ri, rk in enumerate(ranks)}
    unr = {M: float(SF["S"][D.M_CURVE.index(M)][-1]) for M in D.M_CURVE}
    for rk in ranks:
        for mi, M in enumerate(D.M_CURVE):
            assert abs(SF[rk][mi][-1] - unr[M]) < 1e-12
    ce = {"dataset": ds, "npart": npart, "n_rows": nq, "unrouted_ALL": {str(M): D.q4(unr[M]) for M in D.M_CURVE}}
    # envelope: the best fixed point over every ranking and every B_P (gold-chosen, per B_N)
    env = {}
    for mi, M in enumerate(D.M_CURVE):
        best = max(((float(SF[rk][mi].max()), -(int(np.argmax(SF[rk][mi])) + 1), rk) for rk in ranks))
        env[M] = {"ALL": D.q4(best[0]), "ranking": best[2], "B_P": -best[1]}
    ce["best_fixed_any_ranking"] = {str(M): env[M] for M in D.M_CURVE}
    # surfaces and the user's points
    ce["surfaces"] = {rk: {str(M): {str(b): D.q4(SF[rk][D.M_CURVE.index(M)][min(b, npart) - 1]) for b in BP_TAB} for M in MS_SHOW} for rk in SURF_RANKS}
    ce["user_points"] = {rk: dg["surfaces"][c][rk]["user_points"] for rk in SURF_RANKS}
    ce["concentration"] = dg["concentration"][c]
    ce["gold_fan_out_diagnostic"] = {rk: dg["gold_fan_out_diagnostic"][c][rk] for rk in SURF_RANKS}
    ce["arms"] = {}
    for an in SHOW_ARMS:
        ai = arms.index(an)
        rr, agg, cc = an.split("|")
        rk = agg if rr == "OWN" else "S"
        ri = ranks.index(rk)
        b = BPQ[:, ai]
        bm = float(b.mean())
        bl, bh = int(np.floor(bm)), int(np.ceil(bm))
        ea = dg["adaptive"][c][an]
        e = {"ranking": rk, "B_P(q)": ea["B_P(q)"], "capped_mean_B_P": ea["capped_mean_B_P"]}
        for mi, M in enumerate(D.M_CURVE):
            srv = AB.served_at(LO[:, ri], HI[:, ri, mi], b)
            allm = float(srv.mean())
            assert abs(allm - ea[str(M)]["ALL"]) < 5e-5
            itp = lambda S_: float(S_[mi][bl - 1] + (bm - bl) * (S_[mi][bh - 1] - S_[mi][bl - 1]))
            same = SF[rk][mi]
            reach = np.flatnonzero(same >= allm - 1e-12)
            em = {"ALL": D.q4(allm), "returned_nodes_mean": ea[str(M)]["returned_nodes_mean"], "unrouted_ALL": D.q4(unr[M]),
                  "paired_vs_unrouted": ea[str(M)]["paired_vs_unrouted"],
                  "same_ranking_fixed_at_mean_fan_out": D.q4(itp(SF[rk])), "S_fixed_at_mean_fan_out": D.q4(itp(SF["S"])),
                  "same_ranking_best_fixed": {"B_P": int(np.argmax(same)) + 1, "ALL": D.q4(same.max())},
                  "best_fixed_any_ranking": env[M],
                  "smallest_same_ranking_fixed_B_P_reaching_this_ALL": (int(reach[0]) + 1) if len(reach) else None,
                  "capped": ea[str(M)]["capped"], "paired (fixed at rounded mean -> adaptive)": ea[str(M)]["fixed_at_mean_fan_out"]["paired (fixed -> adaptive)"]}
            if "per_hop" in ea[str(M)]:
                em["per_hop"] = ea[str(M)]["per_hop"]
            e[str(M)] = em
            u = univ[an].setdefault(str(M), {"delta_vs_best_fixed_any_ranking": [], "delta_vs_same_ranking_at_mean_fan_out": [],
                                              "delta_vs_S_at_mean_fan_out": [], "delta_vs_unrouted": [], "mean_B_P": [], "cells": []})
            u["delta_vs_best_fixed_any_ranking"].append(D.q4(allm - env[M]["ALL"]))
            u["delta_vs_same_ranking_at_mean_fan_out"].append(D.q4(allm - itp(SF[rk])))
            u["delta_vs_S_at_mean_fan_out"].append(D.q4(allm - itp(SF["S"])))
            u["delta_vs_unrouted"].append(D.q4(allm - unr[M]))
            u["mean_B_P"].append(round(bm, 2))
            u["cells"].append(c)
        if "B_P(q)_per_hop" in ea:
            e["B_P(q)_per_hop"] = ea["B_P(q)_per_hop"]
        ce["arms"][an] = e
    ce["latency_ms"] = dg["latency_ms"]
    summ["cells"][c] = ce
for an, v in univ.items():
    for M, u in v.items():
        u["worst_cell"] = {k: min(u[k]) for k in ("delta_vs_best_fixed_any_ranking", "delta_vs_same_ranking_at_mean_fan_out",
                                                  "delta_vs_S_at_mean_fan_out", "delta_vs_unrouted")}
summ["universality"] = univ

# ---------------- markdown
C = [c for _, c in CELL_ORDER]
md += ["## Adaptive fan-out B_P(q) per cell (mean / median / p90 / max; partitions with a_j > 0 = mean)", ""]
md += ["| arm | " + " | ".join(C) + " |", "|---|" + "---|" * len(C)]
for an in PRIMARY_ARMS + SENS_ARMS:
    cells_ = []
    for c in C:
        d_ = summ["cells"][c]["arms"][an]["B_P(q)"]
        cells_.append("%.1f / %g / %g / %g" % (d_["mean"], d_["median"], round(d_["p90"], 1), d_["max"]))
    md.append("| %s | %s |" % (an, " | ".join(cells_)))
md.append("| partitions (npart) / with a_j > 0 (mean) | %s |" % " | ".join(
    "%d / %.0f" % (summ["cells"][c]["npart"], summ["cells"][c]["concentration"]["SUM"]["partitions_with_a_j>0"]["mean"]) for c in C))
md.append("")
for c in C:
    ce = summ["cells"][c]
    md += ["## %s (%d partitions)" % (c, ce["npart"]), ""]
    md += ["ALL gold at B_N; adaptive [mean B_P]; same = same ranking fixed at the mean fan-out; S@ = S fixed at that fan-out; "
           "bestR = the same ranking's best fixed (B_P); env = best fixed point of any ranking (ranking B_P); reach = smallest fixed B_P "
           "of the same ranking reaching the adaptive ALL", ""]
    md += ["| arm | B_N | adaptive [B_P] | returned | unrouted | same | S@ | bestR | env | reach | vs unrouted +/- (p) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for an in PRIMARY_ARMS + REF_ARMS:
        e = ce["arms"][an]
        for M in MS_SHOW:
            em = e[str(M)]
            pu = em["paired_vs_unrouted"]
            md.append("| %s | %d | %s [%.1f] | %.0f | %s | %s | %s | %s (%d) | %s (%s %d) | %s | +%d/-%d (%s) |" % (
                an, M, f3(em["ALL"]), e["B_P(q)"]["mean"], em["returned_nodes_mean"], f3(em["unrouted_ALL"]),
                f3(em["same_ranking_fixed_at_mean_fan_out"]), f3(em["S_fixed_at_mean_fan_out"]), f3(em["same_ranking_best_fixed"]["ALL"]),
                em["same_ranking_best_fixed"]["B_P"], f3(em["best_fixed_any_ranking"]["ALL"]), em["best_fixed_any_ranking"]["ranking"],
                em["best_fixed_any_ranking"]["B_P"], em["smallest_same_ranking_fixed_B_P_reaching_this_ALL"],
                pu["gained"], pu["lost"], pv(pu["p"])))
    md.append("")
    md += ["Surfaces R(B_P, B_N) (ALL; B_P = %s; last = unrouted)" % (list(BP_TAB),), ""]
    md += ["| ranking | B_N | " + " | ".join(str(b) for b in BP_TAB) + " | unrouted |", "|---|---|" + "---|" * (len(BP_TAB) + 1)]
    for rk in SURF_RANKS:
        for M in MS_SHOW:
            row_ = ce["surfaces"][rk][str(M)]
            md.append("| %s | %d | %s | %s |" % (rk, M, " | ".join(f3(row_[str(b)]) for b in BP_TAB), f3(ce["unrouted_ALL"][str(M)])))
    md.append("")
    md.append("User points: " + "; ".join("%s %s" % (rk, json.dumps(ce["user_points"][rk])) for rk in SURF_RANKS))
    md.append("")
    md += ["Caps min(B_P^max, B_P(q)) (ALL at B_N 1000; mean fan-out in brackets)", ""]
    md += ["| arm | uncapped | " + " | ".join("cap %d" % cp for cp in AB.CAPS) + " |", "|---|---|" + "---|" * len(AB.CAPS)]
    for an in PRIMARY_ARMS[:3]:
        e = ce["arms"][an]
        md.append("| %s | %s [%.1f] | %s |" % (an, f3(e["1000"]["ALL"]), e["B_P(q)"]["mean"], " | ".join(
            "%s [%.1f]" % (f3(e["1000"]["capped"][str(cp)]), e["capped_mean_B_P"][str(cp)]) for cp in AB.CAPS)))
    md.append("")
    md += ["Gold fan-out diagnostic (LO = fan-out contacting every gold partition; never a rule)", ""]
    for rk in SURF_RANKS:
        g_ = ce["gold_fan_out_diagnostic"][rk]
        lo = g_["LO (fan-out contacting every gold partition)"]
        s_ = "- %s: LO mean %.1f / median %g / p90 %.1f / max %g; servable@100/1000 %s / %s" % (
            rk, lo["mean"], lo["median"], lo["p90"], lo["max"], f3(g_["servable_at_some_B_P"]["100"]), f3(g_["servable_at_some_B_P"]["1000"]))
        if "spearman(N_eff, LO)" in g_:
            s_ += "; spearman(N_eff, LO) %s, (exp H, LO) %s; NEFF B_P >= LO %s; over-contacted@100/1000 %s / %s" % (
                g_["spearman(N_eff, LO)"], g_["spearman(exp H, LO)"], f3(g_["NEFF: B_P(q) >= LO (all gold partitions contacted)"]),
                f3(g_["NEFF: over-contacted (served at some smaller B_P, lost at B_P(q))"]["100"]),
                f3(g_["NEFF: over-contacted (served at some smaller B_P, lost at B_P(q))"]["1000"]))
        md.append(s_)
    md.append("")
    hop_arm = summ["cells"][c]["arms"]["OWN|TOP%d|NEFF" % AB.TOPK_PRIMARY]
    if "per_hop" in hop_arm["1000"]:
        md += ["Per hop (B_N 100 / 1000 / 5000): adaptive ALL [mean B_P] | fixed same ranking at rounded mean | unrouted", ""]
        for an in ("OWN|SUM|NEFF", "OWN|MAX|NEFF", "OWN|TOP%d|NEFF" % AB.TOPK_PRIMARY, "S|TOP%d|NEFF" % AB.TOPK_PRIMARY):
            e = ce["arms"][an]
            parts_ = []
            for hk in sorted(e["1000"]["per_hop"]):
                parts_.append("%s: %s" % (hk, " / ".join("%s [%.1f] | %s | %s" % (
                    f3(e[str(M)]["per_hop"][hk]["ALL"]), e[str(M)]["per_hop"][hk]["mean_B_P"], f3(e[str(M)]["per_hop"][hk]["fixed_at_rounded_mean_fan_out_ALL"]),
                    f3(e[str(M)]["per_hop"][hk]["unrouted_ALL"])) for M in (100, 1000, 5000))))
            md.append("- %s: %s" % (an, "; ".join(parts_)))
        md.append("")
    lat = ce["latency_ms"]
    md.append("Latency (ms mean): node score %s, fused order %s, selection %s, evaluation-only %s" % (
        lat["node_score_and_LOC_order"]["mean_ms"], lat["fused_order"]["mean_ms"],
        list(lat["selection (6 aggregations + 7 rankings + 12 concentrations, per cell)"].values())[0]["mean_ms"] if len(C) else None,
        {k: v["mean_ms"] for k, v in lat["evaluation_only (every fan-out of 7 rankings, per cell)"].items()}))
    md.append("")
md += ["## Universality: worst cell over %s (adaptive minus reference; mean B_P per cell)" % C, ""]
md += ["| arm | B_N | vs env (any ranking, best B_P) | vs same ranking @ mean fan-out | vs S @ mean fan-out | vs unrouted | mean B_P per cell |",
       "|---|---|---|---|---|---|---|"]
for an in SHOW_ARMS:
    for M in MS_SHOW:
        u = univ[an][str(M)]
        w = u["worst_cell"]
        md.append("| %s | %d | %+.3f | %+.3f | %+.3f | %+.3f | %s |" % (an, M, w["delta_vs_best_fixed_any_ranking"], w["delta_vs_same_ranking_at_mean_fan_out"],
                                                                 w["delta_vs_S_at_mean_fan_out"], w["delta_vs_unrouted"], " / ".join("%.1f" % x for x in u["mean_B_P"])))
md.append("")
fo = os.path.join(OUT, "adaptbp_SUMMARY__%s.json" % TAG)
assert not os.path.exists(fo), "write-once: %s exists" % fo
D.G.S.wj(fo, summ)
print("summary -> %s sha256 %s" % (fo, D.sha_file(fo)))
if MD:
    assert not os.path.abspath(MD).lower().startswith(os.path.abspath(D.REPO).lower())
    open(MD, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("markdown -> %s (%d lines)" % (MD, len(md)))
