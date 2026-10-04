"""Summary of the L1 development variant runs written through _l1d_arms.run (step 3 hier_*, step 4 act_*), read-only over
results/L1_DEV/<stem>_<dataset>__<tag>.json: variant x budget ALL-gold tables for the LOC and FLAT+LOC arms (gained / lost vs
FLAT@M), per-hop and per-cardinality ALL-gold at M = 1000 and MMAX, invisible-gold recovery, LOC-order length, latency and
the harness diagnostics.

Usage: python scratchpad/_l1d_var_summary.py <stem> <tag> [--dir=<dir>]  ->  <dir>/<stem>_SUMMARY__<tag>.json (write-once) + markdown.
DEVELOPMENT numbers (user ruling 2026-09-26): descriptive, no verdicts."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUT = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--dir=")), os.path.join(REPO, "results", "L1_DEV"))
STEM, TAG = sys.argv[1], sys.argv[2]
PAIRED = "paired_vs_FLAT (gained = arm serves ALL gold, FLAT does not; McNemar p descriptive)"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


summ = {"stem": STEM, "tag": TAG, "status": "DEVELOPMENT (descriptive; not confirmatory)", "inputs": {}, "datasets": {}}
md = []
for ds in ("metaqa", "musique", "squad"):
    fj = os.path.join(OUT, "%s_%s__%s.json" % (STEM, ds, TAG))
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
        md.append("| variant | " + " | ".join("M=%d" % M for M in Ms) + " | invisible served @%d |" % MMAX)
        md.append("|---|" + "---|" * (len(Ms) + 1))
        md.append("| FLAT | " + " | ".join("%.4f" % r["FLAT"][str(M)]["ALL"] for M in Ms) + " | 0 |")
        for n_, e in V.items():
            cells = []
            for M in Ms:
                a = e["arms"][arm][str(M)]
                cells.append("%.4f (+%d/−%d)" % (a["ALL"], a[PAIRED]["gained"], a[PAIRED]["lost"]))
            last = e["arms"][arm][str(MMAX)]
            md.append("| %s | %s | %d |" % (n_, " | ".join(cells), last["invisible_gold_nodes_served (FLAT rank >= %d)" % MMAX]))
            D["variants"].setdefault(n_, {})[arm] = {str(M): {"ALL": e["arms"][arm][str(M)]["ALL"], "ANY": e["arms"][arm][str(M)]["ANY"],
                                                              "FRAC": e["arms"][arm][str(M)]["FRAC"], "vs_FLAT": e["arms"][arm][str(M)][PAIRED],
                                                              "FLAT_budget_matching": e["arms"][arm][str(M)]["FLAT_budget_matching_this_ALL"],
                                                              "invisible_served": e["arms"][arm][str(M)]["invisible_gold_nodes_served (FLAT rank >= %d)" % MMAX]}
                                                     for M in Ms}
        st0 = r["FLAT"][str(Ms[0])]["strata"]
        for M in (1000, MMAX):
            md.append("\n%s arm, ALL-gold by stratum at M = %d\n" % (arm, M))
            cols = [(sn, k) for sn in st0 for k in st0[sn]]
            md.append("| variant | " + " | ".join("%s %s (n=%d)" % (sn.replace("per_", ""), k, st0[sn][k]["n"]) for sn, k in cols) + " |")
            md.append("|---|" + "---|" * len(cols))
            md.append("| FLAT | " + " | ".join("%.3f" % r["FLAT"][str(M)]["strata"][sn][k]["ALL"] for sn, k in cols) + " |")
            for n_, e in V.items():
                s_ = e["arms"][arm][str(M)]["strata"]
                md.append("| %s | " % n_ + " | ".join("%.3f" % s_[sn][k]["ALL"] for sn, k in cols) + " |")
                D["variants"][n_].setdefault("strata", {}).setdefault(arm, {})[str(M)] = {
                    "%s:%s" % (sn, k): s_[sn][k]["ALL"] for sn, k in cols}
    md.append("\nLOC-order length (median) / score latency (mean ms): " + "; ".join(
        "%s %s / %s" % (n_, e["loc_order_length"]["median"], (e["latency_ms (score + LOC order)"] or {}).get("mean_ms")) for n_, e in V.items()))
    md.append("\nStructures: " + json.dumps(r["structures"])[:3000])
    md.append("\nDiagnostics: " + json.dumps(r["diagnostics"])[:6000] + "\n")
    D["loc_order_length_median"] = {n_: e["loc_order_length"]["median"] for n_, e in V.items()}
    D["latency_ms"] = {n_: e["latency_ms (score + LOC order)"] for n_, e in V.items()}
    D["structures"] = r["structures"]
    D["diagnostics"] = r["diagnostics"]
    summ["datasets"][ds] = D
fo = os.path.join(OUT, "%s_SUMMARY__%s.json" % (STEM, TAG))
assert not os.path.exists(fo), "write-once: %s exists" % fo
with open(fo, "w", encoding="utf-8") as f:
    json.dump(summ, f, indent=1, ensure_ascii=True)
print("\n".join(md))
print("\n-> %s sha256 %s" % (fo, sha_file(fo)))
