"""Summary of the NODELOC-1H development runs (_l1d_node1h.py), read-only over <dir>/node1h_<dataset>__<tag>.{json,npz}.
It reports:
  - the budget curves: ALL gold for FLAT@M, LOC@M and FLAT+LOC@M with gained / lost vs FLAT; ANY / FRAC; the FLAT exposure
    that matches each arm; per hop and per cardinality;
  - the ladder L0 -> L1 -> L2 and the hit weights FV vs IR (paired);
  - the comparison with step-2 MEMACT and step-1 HARD;
  - the two-budget partition routing grid (B_P partitions contacted x B_N nodes returned), plus per hop at selected points;
  - which families reach the gold; latency, index bytes and memory.
Usage: python scratchpad/_l1d_node1h_summary.py <tag> [--dir=<dir>]  ->  <dir>/node1h_SUMMARY__<tag>.json (write-once) + markdown.
DEVELOPMENT numbers (user rulings 2026-09-26/27): descriptive, no verdicts."""
import json
import os
import sys

import numpy as np

import _l1d_lib as D

TAG = sys.argv[1]
OUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), D.OUT)
PAIRED = D.PAIRED_KEY
RUNGS = ["FV_L0", "FV_L1", "FV_L2", "IR_L0", "IR_L1", "IR_L2"]
MS_SHOW = (100, 500, 1000, 2000, 5000)                     # the user's budget list
ROUTE_POINTS = ((5, 100), (10, 500), (20, 1000), (50, 1000), (50, 5000), (100, 5000))
HOP_ROUTE_CFGS = ("FLAT|S", "FV_L0|LMAX", "IR_L0|S", "IR_L0|LMAX", "IR_L0|LSUM", "IR_L2|LSUM")


def f3(x):
    return "-" if x is None else "%.3f" % x


def pv(p):
    return "-" if p is None else ("%.1e" % p if p < 1e-3 else "%.3f" % p)


summ = {"stem": "node1h", "tag": TAG, "status": "DEVELOPMENT (descriptive; not confirmatory)", "inputs": {}, "datasets": {}}
md = []
for ds in ("metaqa", "musique", "squad"):
    fj = os.path.join(OUT, "node1h_%s__%s.json" % (ds, TAG))
    if not os.path.exists(fj):
        md.append("\n(%s: no record)" % ds)
        continue
    r = json.load(open(fj, encoding="utf-8"))
    fz = os.path.join(OUT, r["npz"]["path"])
    assert D.sha_file(fz) == r["npz"]["sha256"]
    z = np.load(fz)
    summ["inputs"][ds] = {"json": {"path": D.rel(fj) if fj.lower().startswith(D.REPO.lower()) else fj, "sha256": D.sha_file(fj)},
                          "npz": {"path": r["npz"]["path"], "sha256": r["npz"]["sha256"]}}
    V, FL, dg = r["variants_result"], r["FLAT"], r["diagnostics"]
    cells = r["cells"]
    S = {"N": r["N"], "n_rows": r["n_rows"], "n_gold_nodes": r["n_gold_nodes"], "curves": {}, "strata": {}, "routing": {}}
    md.append("\n## %s (N = %d, %d queries, %d gold nodes)\n" % (ds, r["N"], r["n_rows"], r["n_gold_nodes"]))
    md.append("v1 check: %s  \nrouting identity: %s\n" % (r["v1_check"], dg["routing_identity"]))
    # ---- budget curves
    md.append("**ALL gold at equal exposure M (gained/lost vs FLAT@M)**\n")
    md.append("| arm | " + " | ".join("M=%d" % M for M in MS_SHOW) + " |")
    md.append("|---|" + "---|" * len(MS_SHOW))
    md.append("| FLAT | " + " | ".join("%.3f" % FL[str(M)]["ALL"] for M in MS_SHOW) + " |")
    S["curves"]["FLAT"] = {str(M): {k: FL[str(M)][k] for k in ("ALL", "ANY", "FRAC")} for M in D.M_CURVE}
    for n_ in RUNGS + ["%s__HARD" % c for c in cells]:
        for arm in ("LOC", "FLAT+LOC"):
            a = V[n_]["arms"][arm]
            md.append("| %s %s | " % (n_, arm) + " | ".join("%.3f (+%d/−%d)" % (a[str(M)]["ALL"], a[str(M)][PAIRED]["gained"], a[str(M)][PAIRED]["lost"])
                                                          for M in MS_SHOW) + " |")
            S["curves"]["%s|%s" % (n_, arm)] = {str(M): {"ALL": a[str(M)]["ALL"], "ANY": a[str(M)]["ANY"], "FRAC": a[str(M)]["FRAC"],
                                                         "vs_FLAT": a[str(M)][PAIRED],
                                                         "FLAT_exposure_matching_this_ALL": a[str(M)]["FLAT_budget_matching_this_ALL"]}
                                                for M in D.M_CURVE}
    md.append("\n**ANY / FRAC, and the FLAT exposure that matches the arm's ALL (M = 100 / 1000 / 5000)**\n")
    md.append("| arm | ANY | FRAC | FLAT exposure matching ALL |")
    md.append("|---|---|---|---|")
    md.append("| FLAT | %s | %s | = M |" % (" / ".join(f3(FL[str(M)]["ANY"]) for M in (100, 1000, 5000)),
                                            " / ".join(f3(FL[str(M)]["FRAC"]) for M in (100, 1000, 5000))))
    for n_ in RUNGS:
        for arm in ("LOC", "FLAT+LOC"):
            a = V[n_]["arms"][arm]
            md.append("| %s %s | %s | %s | %s |" % (n_, arm, " / ".join(f3(a[str(M)]["ANY"]) for M in (100, 1000, 5000)),
                                                    " / ".join(f3(a[str(M)]["FRAC"]) for M in (100, 1000, 5000)),
                                                    " / ".join(str(a[str(M)]["FLAT_budget_matching_this_ALL"]) for M in (100, 1000, 5000))))
    # ---- strata
    st0 = FL["100"]["strata"]
    cols = [(sn, k) for sn in st0 for k in st0[sn]]
    for M in (100, 1000, 5000):
        md.append("\n**ALL gold by stratum at M = %d (FLAT, then each rung's LOC / FLAT+LOC)**\n" % M)
        md.append("| arm | " + " | ".join("%s %s (n=%d)" % (sn.replace("per_", ""), k, st0[sn][k]["n"]) for sn, k in cols) + " |")
        md.append("|---|" + "---|" * len(cols))
        md.append("| FLAT | " + " | ".join("%.3f" % FL[str(M)]["strata"][sn][k]["ALL"] for sn, k in cols) + " |")
        S["strata"].setdefault(str(M), {})["FLAT"] = {"%s:%s" % (sn, k): FL[str(M)]["strata"][sn][k]["ALL"] for sn, k in cols}
        for n_ in RUNGS:
            for arm in ("LOC", "FLAT+LOC"):
                s_ = V[n_]["arms"][arm][str(M)]["strata"]
                md.append("| %s %s | " % (n_, arm) + " | ".join("%.3f" % s_[sn][k]["ALL"] for sn, k in cols) + " |")
                S["strata"][str(M)]["%s|%s" % (n_, arm)] = {"%s:%s" % (sn, k): s_[sn][k]["ALL"] for sn, k in cols}
    # ---- ladder / weights
    lad = dg["paired_ladder_and_weight"]
    md.append("\n**Ladder and hit weight, paired on ALL gold (gained / lost / McNemar p)**\n")
    keys = list(lad["FLAT+LOC"]["100"].keys())
    md.append("| comparison | arm | " + " | ".join("M=%d" % M for M in (100, 1000, 5000)) + " |")
    md.append("|---|---|" + "---|" * 3)
    for k in keys:
        for arm in ("LOC", "FLAT+LOC"):
            md.append("| %s | %s | " % (k, arm) + " | ".join("+%d/−%d (p %s)" % (lad[arm][str(M)][k]["gained"], lad[arm][str(M)][k]["lost"], pv(lad[arm][str(M)][k]["p"]))
                                                            for M in (100, 1000, 5000)) + " |")
    S["ladder"] = lad
    # ---- vs MEMACT / HARD
    vs = dg["paired_vs_step1_HARD_and_step2_MEMACT"]
    for c in cells:
        md.append("\n**%s: vs step-1 HARD and step-2 MEMACT (reference ALL; gained/lost of each rung)**\n" % c)
        md.append("| arm | M | HARD ALL | MEMACT ALL | " + " | ".join("%s vs MEMACT" % n_ for n_ in RUNGS) + " |")
        md.append("|---|---|---|---|" + "---|" * len(RUNGS))
        for arm in ("LOC", "FLAT+LOC"):
            for M in (100, 1000, 5000):
                e = vs[c][arm][str(M)]
                md.append("| %s | %d | %s | %s | " % (arm, M, f3(e.get("HARD_ALL")), f3(e.get("MEMACT_ALL"))) +
                          " | ".join(("+%d/−%d" % (e["MEMACT -> " + n_]["gained"], e["MEMACT -> " + n_]["lost"])) if ("MEMACT -> " + n_) in e else "-"
                                     for n_ in RUNGS) + " |")
    S["vs_HARD_MEMACT"] = vs
    # ---- routing
    pop = None
    for c in cells:
        rc = dg["routing"][c]
        npart = r["structures"]["routing"]["cells"][c]["npart"]
        md.append("\n**%s (%d partitions): two-budget routing, ALL gold at (B_P contacted, B_N returned); in brackets the unrouted arm at "
                  "B_N and the mean nodes returned**\n" % (c, npart))
        md.append("| served order and routing | " + " | ".join("B_P=%d, B_N=%d" % bp for bp in ROUTE_POINTS) + " |")
        md.append("|---|" + "---|" * len(ROUTE_POINTS))
        for cfg in [k for k in rc if "|" in k]:
            md.append("| %s | " % cfg.replace("|", " · ") + " | ".join("%.3f [%.3f; %.0f]" % (rc[cfg][str(BP)][str(M)]["ALL"], rc[cfg][str(BP)][str(M)]["unrouted_ALL"],
                                                                        rc[cfg][str(BP)][str(M)]["returned_nodes_mean"]) for BP, M in ROUTE_POINTS) + " |")
        md.append("\npartitions touched by the unrouted top-M (mean): " + "; ".join(
            "%s %s" % (n_, " / ".join("%.0f" % rc["partitions_touched_by_the_unrouted_top_M"][n_][str(M)]["mean"] for M in MS_SHOW))
            for n_ in ["FLAT"] + RUNGS) + "  (M = %s)" % ", ".join(str(M) for M in MS_SHOW))
        S["routing"][c] = {"grid": {cfg: rc[cfg] for cfg in rc if "|" in cfg},
                           "partitions_touched": rc["partitions_touched_by_the_unrouted_top_M"], "routing_ms": rc["routing_ms"]}
        # per hop at the selected points, from the npz
        if pop is None:
            cd = D.AD.CanonicalDataset(ds)
            pop = D.Population(cd, r["n_rows"] if r["population"]["first_rows_only"] else None)
            assert (pop.rows == z["rows"]).all() and (pop.gptr == z["gptr"]).all()
        cfgl = list(z["route_cfg"])
        bpl = list(z["route_bp"])
        RR = z["route_rr__" + c]
        hop_m = pop.ST.get("per_hop", {})
        if hop_m:
            md.append("\nper hop (ALL gold): " + " | ".join("B_P=%d,B_N=%d" % bp for bp in ROUTE_POINTS))
            ph = {}
            for cfg in HOP_ROUTE_CFGS:
                ci = cfgl.index(cfg)
                for hk, qm in hop_m.items():
                    vals = []
                    for BP, M in ROUTE_POINTS:
                        a_ = D.per_query(RR[:, ci, bpl.index(BP)].astype(np.int64), M, pop.gptr, pop.ngold)[0]
                        vals.append(float(a_[qm].mean()))
                    ph["%s|%s" % (cfg, hk)] = [round(v, 4) for v in vals]
                    md.append("- %s %s (n=%d): %s" % (cfg.replace("|", " · "), hk, int(qm.sum()), " / ".join("%.3f" % v for v in vals)))
            S["routing"][c]["per_hop_at_points"] = {"points": [list(p) for p in ROUTE_POINTS], "values": ph}
    # ---- reach, support, costs
    rh = dg["gold_family_reach"]
    md.append("\n**Which families reach the gold (share of gold nodes reached from the top-200 hits; exclusive share in brackets)**\n")
    md.append("| gold set | n | any | " + " | ".join(("STRUCT_out", "STRUCT_in", "KNN", "NER")) + " | any from top-10 |")
    md.append("|---|---|---|" + "---|" * 5)
    for lab in ("all_gold", "FLAT_missed@100", "FLAT_missed@1000", "FLAT_missed@5000"):
        e = rh[lab]
        if e["top200"] is None:
            md.append("| %s | 0 | - | - | - | - | - | - |" % lab)
            continue
        md.append("| %s | %d | %s | %s | %s |" % (lab, e["gold_nodes"], f3(e["top200"]["any_family"]), " | ".join(
            "%s (%s)" % (f3(e["top200"][f]["reached"]), f3(e["top200"][f]["only_this_family"])) for f in ("STRUCT_out", "STRUCT_in", "KNN", "NER")),
            f3(e["top10"]["any_family"])))
    S["reach"] = rh
    sup = dg["support"]
    S["support"] = sup
    cc = sup["edge_diagnostic_crosscheck"]
    md.append("\nsupport (nodes with L > 0) per query, top-200 / top-10 hits: %s / %s; edge-diagnostic cross-check: %s" % (
        sup["support_per_query (top200 / top10)"][0], sup["support_per_query (top200 / top10)"][1],
        cc if isinstance(cc, str) else ("all_match=%s" % cc["all_match"])))
    lat = {n_: V[n_]["latency_ms (score + LOC order)"] for n_ in V}
    S["costs"] = {"latency_ms_node_score_and_LOC_order": lat, "latency_ms_flat": r["latency_ms_flat"], "fused_order_ms": dg["fused_order_ms"],
                  "index_bytes_total": r["structures"]["index_bytes_total"], "process_peak_rss_mb": r["process_peak_rss_mb"],
                  "seconds": r["seconds"], "loc_order_length": {n_: V[n_]["loc_order_length"] for n_ in V}}
    md.append("\ncosts: FLAT products %.1f ms + rrf %.1f ms per query; node score + LOC order (mean ms) %s; fused order (mean ms) %s; "
              "graph index %.1f MB; peak RSS %s MB; %.0f s" % (
                  r["latency_ms_flat"]["products_amortized"]["mean_ms"], r["latency_ms_flat"]["flat_rrf"]["mean_ms"],
                  ", ".join("%s %.2f" % (n_, lat[n_]["mean_ms"]) for n_ in V), ", ".join("%s %.1f" % (k, v["mean_ms"]) for k, v in dg["fused_order_ms"].items()),
                  r["structures"]["index_bytes_total"] / 2 ** 20, r["process_peak_rss_mb"], r["seconds"]))
    summ["datasets"][ds] = S
fo = os.path.join(OUT, "node1h_SUMMARY__%s.json" % TAG)
assert not os.path.exists(fo), "write-once: %s exists" % fo
with open(fo, "w", encoding="utf-8") as f:
    json.dump(summ, f, indent=1, ensure_ascii=True)
print("\n".join(md))
print("\n-> %s sha256 %s" % (fo, D.sha_file(fo)))
