"""Summary of the L1 development soft-membership runs (scratchpad/_l1d_soft.py), read-only over
results/L1_DEV/soft_<dataset>__<tag>.json: variant x budget ALL-gold tables for the LOC and FLAT+LOC arms, per-hop and
per-cardinality splits, silent (L = 0) shares of the FLAT@M misses, invisible-gold recovery, latency.

Usage: python scratchpad/_l1d_soft_summary.py <tag> [--dir=<dir>]  ->  <dir>/soft_SUMMARY__<tag>.json (write-once) + markdown.
DEVELOPMENT numbers (user ruling 2026-09-26): descriptive, no verdicts."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), os.path.join(REPO, "results", "L1_DEV"))
TAG = sys.argv[1]
PAIRED = "paired_vs_FLAT (gained = arm serves ALL gold, FLAT does not; McNemar p descriptive)"
PH = "paired_vs_HARD_same_arm (gained = this variant serves ALL gold, HARD does not)"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


summ = {"tag": TAG, "status": "DEVELOPMENT (descriptive; not confirmatory)", "inputs": {}, "datasets": {}}
md = []
for ds in ("metaqa", "musique", "squad"):
    fj = os.path.join(OUT, "soft_%s__%s.json" % (ds, TAG))
    if not os.path.exists(fj):
        md.append("\n(%s: no record)" % ds)
        continue
    r = json.load(open(fj, encoding="utf-8"))
    summ["inputs"][ds] = {"path": os.path.relpath(fj, REPO).replace("\\", "/"), "sha256": sha_file(fj)}
    Ms = [int(m) for m in r["FLAT"]]
    MMAX = max(Ms)
    V = r["variants_result"]
    md.append("\n### %s (N = %d, %d queries, %d gold nodes; invisible = FLAT rank ≥ %d: %d)\n" % (
        ds, r["N"], r["n_rows"], r["n_gold_nodes"], MMAX, r["invisible_gold_nodes (FLAT rank >= %d)" % MMAX]))
    md.append("v1 check: " + r["v1_check"] + "\n")
    D = {"N": r["N"], "n_rows": r["n_rows"], "FLAT": {k: r["FLAT"][k]["ALL"] for k in r["FLAT"]}, "variants": {}}
    for arm in ("LOC", "FLAT+LOC"):
        md.append("\n**%s arm — ALL-gold (gained/lost vs FLAT@M)**\n" % arm)
        md.append("| variant | " + " | ".join("M=%d" % M for M in Ms) + " | silent share of FLAT@%d misses | invisible served @%d |" % (MMAX, MMAX))
        md.append("|---|" + "---|" * (len(Ms) + 2))
        md.append("| FLAT | " + " | ".join("%.4f" % r["FLAT"][str(M)]["ALL"] for M in Ms) + " | — | 0 |")
        for n_, e in V.items():
            cells = []
            for M in Ms:
                a = e["arms"][arm][str(M)]
                cells.append("%.4f (+%d/−%d)" % (a["ALL"], a[PAIRED]["gained"], a[PAIRED]["lost"]))
            last = e["arms"][arm][str(MMAX)]
            mi = last["of_FLAT@M_missed_gold_nodes"]
            md.append("| %s | %s | %.3f | %d |" % (n_, " | ".join(cells), (mi["silent_L0"] / float(mi["n"])) if mi["n"] else 0.0,
                                                   last["invisible_gold_nodes_served (FLAT rank >= %d)" % MMAX]))
            D["variants"].setdefault(n_, {})[arm] = {str(M): {"ALL": e["arms"][arm][str(M)]["ALL"], "ANY": e["arms"][arm][str(M)]["ANY"],
                                                              "FRAC": e["arms"][arm][str(M)]["FRAC"], "vs_FLAT": e["arms"][arm][str(M)][PAIRED],
                                                              "vs_HARD": e["arms"][arm][str(M)].get(PH),
                                                              "FLAT_budget_matching": e["arms"][arm][str(M)]["FLAT_budget_matching_this_ALL"],
                                                              "missed": e["arms"][arm][str(M)]["of_FLAT@M_missed_gold_nodes"]} for M in Ms}
        # strata at M = 1000 and MMAX
        st0 = r["FLAT"][str(Ms[0])]["strata"]
        for M in (1000, MMAX):
            md.append("\n%s arm, ALL-gold by stratum at M = %d (FLAT first)\n" % (arm, M))
            cols = [(sn, k) for sn in st0 for k in st0[sn]]
            md.append("| variant | " + " | ".join("%s %s (n=%d)" % (sn.replace("per_", ""), k, st0[sn][k]["n"]) for sn, k in cols) + " |")
            md.append("|---|" + "---|" * len(cols))
            md.append("| FLAT | " + " | ".join("%.3f" % r["FLAT"][str(M)]["strata"][sn][k]["ALL"] for sn, k in cols) + " |")
            for n_, e in V.items():
                s_ = e["arms"][arm][str(M)]["strata"]
                md.append("| %s | " % n_ + " | ".join("%.3f" % s_[sn][k]["ALL"] for sn, k in cols) + " |")
    md.append("\nLOC-order length (median) and score latency (mean ms): " + "; ".join(
        "%s %s / %s" % (n_, e["loc_order_length"]["median"], e["latency_ms (score + LOC order)"]["mean_ms"]) for n_, e in V.items()))
    md.append("\nGraph: %s\n" % json.dumps({k: r["graph"][k] for k in ("STRUCT_edges", "KNN_edges", "degree", "nodes_over_DEG_CAP")}))
    D["loc_order_length_median"] = {n_: e["loc_order_length"]["median"] for n_, e in V.items()}
    D["latency_ms"] = {n_: e["latency_ms (score + LOC order)"] for n_, e in V.items()}
    D["graph"] = r["graph"]
    summ["datasets"][ds] = D
fo = os.path.join(OUT, "soft_SUMMARY__%s.json" % TAG)
assert not os.path.exists(fo), "write-once: %s exists" % fo
with open(fo, "w", encoding="utf-8") as f:
    json.dump(summ, f, indent=1, ensure_ascii=True)
print("\n".join(md))
print("\n-> %s sha256 %s" % (fo, sha_file(fo)))
