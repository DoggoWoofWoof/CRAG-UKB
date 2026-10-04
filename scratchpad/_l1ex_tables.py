"""Emit TABLES.md for the equal-exposure audit straight from diag/exposure.json."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _l1ex_core as EX

D = json.load(open(f"{EX.EXD}/diag/exposure.json"))
N = EX.NAME
OD = [d for d in EX.DSETS if d in D]
ORD = ["CANON@16", "CANON@25", "CANON@32", "SAFE", "CANON@50", "CANON@64", "U_PC5@16",
       "SAFE_POOL", "CANON@75", "U_PC5@25", "U_PC5@32", "CANON@100", "U_PC5@50", "U_PC5@64",
       "U_PC5@75", "CANON@128", "U_PC5@100", "U_PC5@128", "CANON@256", "U_PC5@256"]
L = []
w = L.append


def tbl(hdr, rows):
    w("| " + " | ".join(hdr) + " |")
    w("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        assert len(r) == len(hdr), (len(r), len(hdr), r)
        w("| " + " | ".join(str(x) for x in r) + " |")
    w("")


def cfgs(ds):
    have = D[ds]["PART_BUDGET"]
    return [c for c in ORD if c in have] + ["CANON@full", "U_PC5@full"]


def get(ds, c):
    if c.endswith("@full"):
        return D[ds]["FULL"][c[:-5]]
    return D[ds]["PART_BUDGET"][c]


def nullof(ds, c):
    n = D[ds].get("NULL", {}).get(c if not c.endswith("@full") else None)
    return n


w("# L1 EQUAL-EXPOSURE AUDIT -- TABLES\n")
w("Every exposure figure is the real node union over the hard partition membership "
  "(`np.bincount(z[\"hard\"])`), never partitions times a nominal size. "
  "Coverage is one metric throughout: a query is covered iff **all** of its gold partitions are "
  "inside the exposed set.\n")

w("## T1 -- corpus constants (STEP 1)\n")
tbl(["corpus", "corpus nodes", "partitions", "mean partition size", "queries"],
    [[N[d], D[d]["corpus_nodes"], D[d]["corpus_partitions"], D[d]["mean_partition_size"],
      D[d]["nq"]] for d in OD])
w("Partitions hold ~100 nodes on every corpus by construction, so a fixed partition budget is a "
  "fixed ABSOLUTE node budget and the compression ratio is decided by corpus size alone.\n")

w("## T2 -- exposure distribution of the full configurations (STEP 1)\n")
rows = []
for d in OD:
    for m in ["CANON", "SAFE", "SAFE_POOL", "U_PC5"]:
        e = D[d]["FULL"][m]
        rows.append([N[d], m, e["partitions"]["mean"], e["partitions"]["median"],
                     e["partitions"]["p90"], e["partitions"]["p95"], e["nodes"]["mean"],
                     e["nodes"]["median"], e["nodes"]["p90"], e["nodes"]["p95"],
                     f"{e['EXPOSED_FRACTION']:.2%}"])
tbl(["corpus", "method", "parts mean", "parts median", "parts p90", "parts p95", "nodes mean",
     "nodes median", "nodes p90", "nodes p95", "exposed fraction"], rows)

w("## T3 -- MetaQA coverage vs exposure, per hop (STEPS 3, 6, 7)\n")
rows = []
for c in cfgs("metaqa"):
    e = get("metaqa", c)
    nu = nullof("metaqa", c)
    rows.append([c, e["partitions"]["mean"], e["nodes"]["mean"], f"{e['EXPOSED_FRACTION']:.2%}",
                 f"{e['NODE_COMPRESSION_RATIO']:.2f}x", f"{e['hop1']:.4f}", f"{e['hop2']:.4f}",
                 f"{e['hop3']:.4f}", f"{e['ALL']:.4f}",
                 f"{nu['hop3']:.4f}" if nu else "--"])
tbl(["config", "partitions", "nodes exposed", "EXPOSED_FRACTION", "NODE_COMPRESSION_RATIO",
     "hop1", "hop2", "hop3", "ALL", "hop3 random-exposure null"], rows)

w("## T4 -- all six corpora, ALL coverage vs exposure (STEPS 3, 6)\n")
for d in OD:
    w(f"**{N[d]}** ({D[d]['corpus_nodes']} nodes / {D[d]['corpus_partitions']} partitions)\n")
    rows = []
    for c in cfgs(d):
        e = get(d, c)
        nu = nullof(d, c)
        rows.append([c, e["partitions"]["mean"], e["nodes"]["mean"],
                     f"{e['EXPOSED_FRACTION']:.2%}", f"{e['NODE_COMPRESSION_RATIO']:.2f}x",
                     f"{e['ALL']:.4f}", f"{nu['ALL']:.4f}" if nu else "--",
                     "yes" if any(p["config"] == c and p["pareto"] for p in D[d]["PARETO"])
                     else "no"])
    tbl(["config", "partitions", "nodes exposed", "EXPOSED_FRACTION", "NODE_COMPRESSION_RATIO",
         "ALL", "random-exposure null", "Pareto"], rows)

w("## T5 -- equal-node-budget comparison (STEP 4)\n")
w("Each method truncated by its own ranking/proposal order to the largest prefix fitting the "
  "budget. No gold-based truncation.\n")
for d, k in [("metaqa", "hop3"), ("webqsp", "ALL")]:
    w(f"**{N[d]} {k}** (primary)\n")
    rows = []
    for b in ["2500", "5000", "10000", "20000"]:
        r = [b]
        for m in ["CANON", "SAFE", "SAFE_POOL", "U_PC5"]:
            e = D[d]["NODE_BUDGET"][b][m]
            r.append(f"{e[k]:.4f} @ {e['nodes']['mean']:.0f} ({e['EXPOSED_FRACTION']:.1%})")
        rows.append(r)
    tbl(["node budget", "CANON", "SAFE", "SAFE_POOL", "U_PC5"], rows)
for d in OD:
    if d in ("metaqa", "webqsp"):
        continue
    rows = []
    for b in ["2500", "5000", "10000", "20000"]:
        r = [b]
        for m in ["CANON", "SAFE", "SAFE_POOL", "U_PC5"]:
            e = D[d]["NODE_BUDGET"][b][m]
            r.append(f"{e['ALL']:.4f} @ {e['nodes']['mean']:.0f} ({e['EXPOSED_FRACTION']:.1%})")
        rows.append(r)
    w(f"**{N[d]} ALL**\n")
    tbl(["node budget", "CANON", "SAFE", "SAFE_POOL", "U_PC5"], rows)

w("## T6 -- does the router beat plain canonical retrieval at IDENTICAL exposure? (STEP 3)\n")
tbl(["corpus", "SAFE ALL", "SAFE nodes", "CANON@50 ALL", "CANON@50 nodes",
     "what the routing apparatus buys"],
    [[N[d], f"{D[d]['PART_BUDGET']['SAFE']['ALL']:.4f}",
      f"{D[d]['PART_BUDGET']['SAFE']['nodes']['mean']:.0f}",
      f"{D[d]['PART_BUDGET']['CANON@50']['ALL']:.4f}",
      f"{D[d]['PART_BUDGET']['CANON@50']['nodes']['mean']:.0f}",
      f"{D[d]['PART_BUDGET']['SAFE']['ALL'] - D[d]['PART_BUDGET']['CANON@50']['ALL']:+.4f}"]
     for d in OD])

w("## T7 -- U_PC5@256 against canonical at comparable exposure (STEP 5)\n")
tbl(["corpus", "U_PC5@256 ALL", "nodes", "exposed", "CANON@256 ALL", "nodes", "exposed",
     "coverage diff", "extra nodes"],
    [[N[d], f"{D[d]['PART_BUDGET']['U_PC5@256']['ALL']:.4f}",
      f"{D[d]['PART_BUDGET']['U_PC5@256']['nodes']['mean']:.0f}",
      f"{D[d]['PART_BUDGET']['U_PC5@256']['EXPOSED_FRACTION']:.2%}",
      f"{D[d]['PART_BUDGET']['CANON@256']['ALL']:.4f}",
      f"{D[d]['PART_BUDGET']['CANON@256']['nodes']['mean']:.0f}",
      f"{D[d]['PART_BUDGET']['CANON@256']['EXPOSED_FRACTION']:.2%}",
      f"{D[d]['PART_BUDGET']['U_PC5@256']['ALL'] - D[d]['PART_BUDGET']['CANON@256']['ALL']:+.4f}",
      f"{D[d]['PART_BUDGET']['U_PC5@256']['nodes']['mean'] / D[d]['PART_BUDGET']['CANON@256']['nodes']['mean'] - 1:+.0%}"]
     for d in OD])

w("## T8 -- Pareto frontier over (unique nodes exposed, ALL coverage) (STEP 5)\n")
rows = []
for d in OD:
    fr = [p for p in D[d]["PARETO"] if p["pareto"]]
    rows.append([N[d], len(fr), sum(1 for p in fr if p["config"].startswith("CANON")),
                 sum(1 for p in fr if p["config"].startswith("U_PC5")),
                 sum(1 for p in fr if p["config"].startswith("SAFE")),
                 "yes" if any(p["config"] == "U_PC5@256" for p in fr) else "no",
                 f"{D[d]['PART_BUDGET']['U_PC5@256']['EXPOSED_FRACTION']:.2%}"])
tbl(["corpus", "frontier points", "CANON", "U_PC5", "SAFE", "U_PC5@256 on frontier",
     "at what exposure"], rows)
for d in OD:
    fr = ", ".join(f"`{p['config']}`" for p in D[d]["PARETO"] if p["pareto"])
    w(f"- **{N[d]}**: {fr}")
w("")

w("## T9 -- nodes required to reach a coverage target, by ANY existing method (STEPS 7, 8)\n")
rows = []
for d, ks in [("metaqa", ["hop1", "hop2", "hop3", "ALL"]), ("webqsp", ["ALL"])]:
    for k in ks:
        r = [N[d], k]
        for t in ["0.30", "0.40", "0.50", "0.60"]:
            v = D[d]["TARGETS"][k][t]
            r.append("not reached" if v is None else
                     f"{v['nodes_mean']:.0f} ({v['EXPOSED_FRACTION']:.1%}) via {v['method']}")
        rows.append(r)
tbl(["corpus", "slice", "coverage 0.30", "coverage 0.40", "coverage 0.50", "coverage 0.60"], rows)

w("## T10 -- the global M budget does not have universal exposure semantics (STEP 8)\n")
tbl(["corpus", "corpus nodes", "U_PC5@256 nodes exposed", "EXPOSED_FRACTION",
     "NODE_COMPRESSION_RATIO", "ALL"],
    [[N[d], D[d]["corpus_nodes"], f"{D[d]['FULL']['U_PC5']['nodes']['mean']:.0f}",
      f"{D[d]['FULL']['U_PC5']['EXPOSED_FRACTION']:.2%}",
      f"{D[d]['FULL']['U_PC5']['NODE_COMPRESSION_RATIO']:.2f}x",
      f"{D[d]['FULL']['U_PC5']['ALL']:.4f}"]
     for d in sorted(OD, key=lambda x: -D[x]["FULL"]["U_PC5"]["EXPOSED_FRACTION"])])

open(f"{EX.EXD}/TABLES.md", "w", encoding="utf-8").write("\n".join(L))
print(f"wrote {EX.EXD}/TABLES.md  ({len(L)} lines)")
