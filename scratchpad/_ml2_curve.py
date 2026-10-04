"""FBX_SCALE stage 4C -- the recall x pin-compression x predicted-Freebase-RAM curve, read off the ml2_v1 EVAL record (no new scoring).

  python scratchpad/_ml2_curve.py      # write-once results/FREEBASE_SCALE/ML2_COMPRESSION_CURVE__v1.json + ml2/ML2_COMPRESSION_CURVE__v1.png
Reads: CALIB_EVAL_WEBQSP__ml2_v1.json, ml2/webqsp__<cand>_k<K>.RUN.json (pin reduction), ml2/ORACLE_PIN_REDUCTION__v1.json, addendum 5 (gates, r_star, bytes/pin).
"""
import hashlib
import io
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FS = os.path.join(REPO, "results", "FREEBASE_SCALE")
OUT = os.path.join(FS, "ML2_COMPRESSION_CURVE__v1.json")
PNG = os.path.join(FS, "ml2", "ML2_COMPRESSION_CURVE__v1.png")
KS = (100, 250, 500)
P_FB = 4253534391                # Freebase STRUCT pins (FBX_GRAPH_STATS)
BYTES_PER_COARSE_PIN = 133.4     # the largest of the three measured 4A cells (addendum 5)
BUDGET = 0.9 * 81.6e9            # bytes
r3 = lambda x: float(round(x + 1e-12, 3))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    ev_p = os.path.join(FS, "CALIB_EVAL_WEBQSP__ml2_v1.json")
    E = json.load(io.open(ev_p, encoding="utf-8"))
    R, V, MC = E["results"], E["verdicts"], E["M_curve"]
    n_rows = E["n_rows"]
    add = json.load(io.open(os.path.join(FS, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_5.json"), encoding="utf-8"))
    cells = []
    for name, v in V.items():
        meth = "M0R" if name == "M0R" else name.split("_")[0]
        for K in KS:
            key = "%s|K%d|matched" % (name, K)
            if key not in R:
                continue
            ref = R["PHG|K%d|matched" % K]["ALL"]
            dall = [a - b for a, b in zip(R[key]["ALL"], ref)]
            pr = 1.0
            if name != "M0R":
                rr = json.load(io.open(os.path.join(FS, "ml2", "webqsp__%s_k%d.RUN.json" % (name, K)), encoding="utf-8"))
                pr = rr["coarse_hypergraph"]["pin_reduction"]
            cells.append({"candidate": name, "method": meth, "K": K, "pin_reduction": r3(pr), "delta_ALL_rows_by_B_N": dall, "worst_delta_ALL_rows": int(min(dall)),
                          "worst_delta_ALL_points": r3(100.0 * min(dall) / n_rows), "mean_delta_ALL_points": r3(100.0 * sum(dall) / (len(dall) * n_rows)),
                          "predicted_freebase_ram_gb": r3(P_FB / pr * BYTES_PER_COARSE_PIN / 1e9), "verdict_candidate": v["verdict"]})
    cells.sort(key=lambda c: (c["K"], -c["pin_reduction"]))
    # Pareto frontier at each K: the largest pin reduction whose worst loss is within each tolerance (rows of 786)
    front = {}
    for K in KS:
        cs = [c for c in cells if c["K"] == K]
        front["K%d" % K] = {"tol_rows_%d" % t: max(((c["pin_reduction"], c["candidate"], c["worst_delta_ALL_rows"]) for c in cs if -c["worst_delta_ALL_rows"] <= t), default=None)
                            for t in (2, 7, 10, 20, 35)}
    oracle = json.load(io.open(os.path.join(FS, "ml2", "ORACLE_PIN_REDUCTION__v1.json"), encoding="utf-8"))["rows"]
    best = max((c for c in cells if c["K"] == 100), key=lambda c: c["pin_reduction"])
    best_ok = max((c for c in cells if c["K"] == 100 and c["verdict_candidate"] in ("RECALL_PRESERVING",)), key=lambda c: c["pin_reduction"])
    r_star = P_FB * BYTES_PER_COARSE_PIN / BUDGET
    rec = {"RECORD": "FBX_SCALE_ML2_COMPRESSION_CURVE", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "scope": "WebQSP SK-to-SK (H4_SK, balance 1.03, frozen routing/eval); read off CALIB_EVAL_WEBQSP__ml2_v1.json; delta = candidate ALL-gold rows minus PHG_SK ALL-gold rows at PHG's own B_P (matched), per B_N in %s, of %d rows" % (MC, n_rows),
           "freebase_extrapolation": {"struct_pins": P_FB, "bytes_per_coarse_pin_assumed": BYTES_PER_COARSE_PIN, "budget_bytes": BUDGET, "r_star_required_pin_reduction": r3(r_star),
                                      "rule": "RAM = struct_pins / pin_reduction * bytes_per_coarse_pin (the coarse hypergraph the vertex-weighted PHG must hold; the coarsener's own memory is not counted, so this is a LOWER bound)"},
           "finish_line_reached": bool(best_ok["pin_reduction"] >= r_star), "best_recall_preserving_K100": {k: best_ok[k] for k in ("candidate", "pin_reduction", "worst_delta_ALL_rows", "predicted_freebase_ram_gb")},
           "best_pin_reduction_K100": {k: best[k] for k in ("candidate", "pin_reduction", "worst_delta_ALL_rows", "worst_delta_ALL_points", "predicted_freebase_ram_gb", "verdict_candidate")},
           "structural_ceiling_PHG_SK_maps_as_clusterings": [{"clusters": o["clusters"], "pin_reduction_unsplit": o["pin_reduction_unsplit"]} for o in oracle],
           "frontier_largest_pin_reduction_within_tolerance_rows": front, "eval_record": "results/FREEBASE_SCALE/CALIB_EVAL_WEBQSP__ml2_v1.json", "eval_record_sha256": sha(ev_p),
           "addendum_5_sha256": sha(os.path.join(FS, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_5.json")), "cells": cells,
           "driver_sha256": sha(os.path.abspath(__file__)), "figure": "results/FREEBASE_SCALE/ml2/ML2_COMPRESSION_CURVE__v1.png"}
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    # ---- figure: worst delta_ALL (points) vs pin reduction, K = 100 / 250 / 500, methods coloured; oracle ceiling + r_star marked ----
    col = {"M0R": "#555555", "M1": "#c0392b", "M2": "#1f6fb5", "M3": "#2e8b57", "M4": "#b7791f"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, K in zip(axs, KS):
        for m in ("M0R", "M1", "M2", "M3", "M4"):
            cs = sorted((c for c in cells if c["K"] == K and c["method"] == m), key=lambda c: c["pin_reduction"])
            ax.plot([c["pin_reduction"] for c in cs], [c["worst_delta_ALL_points"] for c in cs], "o-", color=col[m], label=m, ms=5, lw=1.2)
        ax.axhspan(-7 * 100.0 / n_rows, 0.4, color="#dddddd", alpha=0.6, lw=0)
        ax.axvline(r_star, color="k", ls="--", lw=1)
        ax.axvline(next(o["pin_reduction_unsplit"] for o in oracle if o["clusters"] == K) if any(o["clusters"] == K for o in oracle) else 1, color="#999999", ls=":", lw=1)
        ax.set_xlim(0.9, 8.5)
        ax.set_xlabel("pin reduction (x)   [Freebase RAM = %.0f GB / x]" % (P_FB * BYTES_PER_COARSE_PIN / 1e9))
        ax.set_title("K = %d   (shaded: gate PASS band, 7 rows)" % K)
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("worst delta ALL-gold vs PHG_SK (points)")
    axs[0].legend(loc="lower left", fontsize=8)
    axs[2].text(r_star - 0.1, -6, "required %.2fx" % r_star, rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(PNG, dpi=130)
    print(json.dumps({k: rec[k] for k in ("finish_line_reached", "best_recall_preserving_K100", "best_pin_reduction_K100", "frontier_largest_pin_reduction_within_tolerance_rows")}, indent=1))


if __name__ == "__main__":
    main()
