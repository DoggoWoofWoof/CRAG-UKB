"""Post-hoc descriptive VIEW over the ADAPTBP development records (_l1d_adaptbp.py), read-only over
<dir>/adaptbp_<dataset>__<tag>.{json,npz}; its contents were chosen AFTER adaptbp_SUMMARY__<tag>.json had been read.
It adds, per cell (metaqa, metaqa_phg, musique, squad, squad_phg), what the summary does not tabulate:
  - every one of the 24 declared arms (the S|agg|EXPH references included) against unrouted at every B_N, and its worst cell;
  - the fixed-B_P surfaces of all 7 rankings (TOP2 / TOP5 / TOP10 included) at BP_TAB, each ranking's best fixed B_P and
    the smallest B_P within 0.01 of unrouted;
  - paired comparisons BETWEEN arms: aggregation, the TOPk sensitivity band, count (NEFF vs EXPH), ranking (OWN vs S).
Nothing is recomputed from the corpus: every number is the v1 per-query ALL-served interval [LO, HI(B_N)] tested at a
fixed B_P or at the stored per-query B_P(q); each arm's ALL is asserted equal to its v1 record.
Usage: python scratchpad/_l1d_adaptbp_view.py <tag> [--dir=<records dir>] [--out=<output dir>]
       -> <out>/adaptbp_VIEW__<tag>.json (write-once).
DEVELOPMENT numbers (user rulings 2026-09-26/27): descriptive, no verdicts."""
import json
import os
import sys

import numpy as np

import _l1d_lib as D
import _l1d_adaptbp as AB

TAG = sys.argv[1]
DIR = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), D.OUT)
OUTD = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), DIR)
BP_TAB = (1, 2, 5, 10, 20, 50, 100, 200)
CELL_ORDER = [("metaqa", "metaqa"), ("metaqa", "metaqa_phg"), ("musique", "musique"), ("squad", "squad"), ("squad", "squad_phg")]
T = "TOP%d" % AB.TOPK_PRIMARY
PAIRS = ([("aggregation", "OWN|%s|%s" % (a, cc), "OWN|%s|%s" % (b, cc)) for cc in AB.CONCS for a, b in (("MAX", T), ("SUM", T), ("SUM", "MAX"))]
         + [("sensitivity", "OWN|%s|NEFF" % T, "OWN|TOP%d|NEFF" % k) for k in AB.TOPKS if k != AB.TOPK_PRIMARY]
         + [("count", "OWN|%s|NEFF" % a, "OWN|%s|EXPH" % a) for a in AB.PRIMARY_AGGS]
         + [("ranking", "OWN|%s|%s" % (a, cc), "S|%s|%s" % (a, cc)) for a in AB.PRIMARY_AGGS for cc in AB.CONCS])
for _, a1, a2 in PAIRS:
    assert a1 in AB.ARM_NAMES and a2 in AB.ARM_NAMES, (a1, a2)

HARNESS = D.sha_file(os.path.join(D.HERE, "_l1d_adaptbp.py"))
view = {"stem": "adaptbp", "tag": TAG,
        "kind": "VIEW (post hoc, descriptive; chosen after reading adaptbp_SUMMARY__%s.json; no corpus recomputation)" % TAG,
        "status": "DEVELOPMENT (descriptive; not confirmatory; p-values descriptive)",
        "definitions": {"delta_vs_unrouted": "ALL(arm) - ALL(every partition contacted) at the same B_N",
                        "worst_cell": "min over the five cells of delta_vs_unrouted (ties by cell name)",
                        "pairs": "gained = the second arm serves ALL gold and the first does not; lost = the reverse; McNemar p",
                        "surfaces": "ALL at a fixed B_P for every query, from the per-query intervals (as the summary)"},
        "inputs": {}, "pairs": [list(p) for p in PAIRS], "cells": {}, "worst_cell": {}}
fs = os.path.join(DIR, "adaptbp_SUMMARY__%s.json" % TAG)
view["inputs"]["summary (read before this view was written)"] = {"path": D.rel(fs), "sha256": D.sha_file(fs)}
REC, Z = {}, {}
for ds in ("metaqa", "musique", "squad"):
    fj = os.path.join(DIR, "adaptbp_%s__%s.json" % (ds, TAG))
    r = json.load(open(fj, encoding="utf-8"))
    fz = os.path.join(DIR, r["npz"]["path"])
    assert D.sha_file(fz) == r["npz"]["sha256"], "npz changed: %s" % fz
    assert r["code"]["harness"]["sha256"] == HARNESS, "record %s was written by another harness version" % fj
    REC[ds], Z[ds] = r, np.load(fz)
    view["inputs"][ds] = {"json": {"path": D.rel(fj), "sha256": D.sha_file(fj)}, "npz": {"path": D.rel(fz), "sha256": r["npz"]["sha256"]}}
me = os.path.abspath(__file__)
view["code"] = {"view": {"path": D.rel(me), "sha256": D.sha_file(me)},
                "harness (imported for surface / served_at)": {"path": "scratchpad/_l1d_adaptbp.py", "sha256": HARNESS},
                "lib": {"path": "scratchpad/_l1d_lib.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))}}

delta = {an: {str(M): {} for M in D.M_CURVE} for an in AB.ARM_NAMES}
for ds, c in CELL_ORDER:
    r, z = REC[ds], Z[ds]
    npart = r["structures"]["cells"][c]["npart"]
    ranks = [str(s) for s in z["ranks"]]
    arms = [str(s) for s in z["arms"]]
    assert ranks == list(AB.RANKS) and arms == AB.ARM_NAMES and [int(m) for m in z["m_curve"]] == list(D.M_CURVE)
    LO, HI, BPQ = z["LO__" + c].astype(np.int64), z["HI__" + c].astype(np.int64), z["BPQ__" + c].astype(np.int64)
    SF = {rk: AB.surface(LO[:, ri], HI[:, ri], npart) for ri, rk in enumerate(ranks)}
    unr = {M: float(SF["S"][mi][-1]) for mi, M in enumerate(D.M_CURVE)}
    ce = {"dataset": ds, "npart": npart, "n_rows": int(LO.shape[0]), "unrouted_ALL": {str(M): D.q4(unr[M]) for M in D.M_CURVE},
          "surfaces": {}, "arms": {}, "pairs": []}
    for rk in ranks:
        e = {}
        for mi, M in enumerate(D.M_CURVE):
            s = SF[rk][mi]
            assert abs(s[-1] - unr[M]) < 1e-12
            b = int(np.argmax(s)) + 1
            w = int(np.flatnonzero(s >= unr[M] - 0.01)[0]) + 1
            e[str(M)] = {"ALL_at_B_P": {str(bp): D.q4(s[bp - 1]) for bp in BP_TAB if bp <= npart},
                         "best_fixed": {"B_P": b, "ALL": D.q4(s[b - 1])}, "smallest_B_P_within_0.01_of_unrouted": w}
        ce["surfaces"][rk] = e
    served = {}
    for ai, an in enumerate(arms):
        rr, ag, cc = an.split("|")
        ri = ranks.index(ag if rr == "OWN" else "S")
        e = {"ranking": ranks[ri], "mean_B_P": round(float(BPQ[:, ai].mean()), 2)}
        for mi, M in enumerate(D.M_CURVE):
            sv = AB.served_at(LO[:, ri], HI[:, ri, mi], BPQ[:, ai])
            served[(an, M)] = sv
            assert abs(D.q4(sv.mean()) - r["diagnostics"]["adaptive"][c][an][str(M)]["ALL"]) < 1e-9, (c, an, M)
            e[str(M)] = {"ALL": D.q4(sv.mean()), "delta_vs_unrouted": D.q4(sv.mean() - unr[M])}
            delta[an][str(M)][c] = float(sv.mean() - unr[M])
        ce["arms"][an] = e
    for fam, a1, a2 in PAIRS:
        pe = {"family": fam, "first": a1, "second": a2}
        for M in D.M_CURVE:
            p = D.paired(served[(a1, M)], served[(a2, M)])
            pe[str(M)] = {"ALL_first": D.q4(served[(a1, M)].mean()), "ALL_second": D.q4(served[(a2, M)].mean()),
                          "gained": int(p["gained"]), "lost": int(p["lost"]), "p": None if p["p"] is None else float(p["p"])}
        ce["pairs"].append(pe)
    view["cells"][c] = ce
for an in AB.ARM_NAMES:
    view["worst_cell"][an] = {}
    for M in D.M_CURVE:
        d = delta[an][str(M)]
        wc = min(d, key=lambda k: (d[k], k))
        view["worst_cell"][an][str(M)] = {"delta_vs_unrouted": D.q4(d[wc]), "cell": wc, "per_cell": {k: D.q4(v) for k, v in d.items()}}

os.makedirs(OUTD, exist_ok=True)
fo = os.path.join(OUTD, "adaptbp_VIEW__%s.json" % TAG)
assert not os.path.exists(fo), "write-once: %s exists" % fo
D.G.S.wj(fo, view)
print("view -> %s sha256 %s" % (fo, D.sha_file(fo)))
