"""Reconcile two of my own measurements that cannot both be right, then price a webqsp V2.

WEBQSP_V3_PRESENCE says 4,212 of webqsp's 62,375 MID_NAMED_ENTITY nodes exist in the frozen V3
node universe.  WEBQSP_V3_NAME_JOIN says 0 of those 62,375 have a row in V3's name.parquet.
A fresh control settles that the join path itself is sound: 13,757 of 15,000 ENTITY_MID
node_ids drawn FROM THE NODE TABLE find a name row (91.7%).  So if those 4,212 were ENTITY_MID
on the V3 side, roughly 3,860 of them should have been named, and 0 is not a rounding error.

Exactly one hypothesis survives without accusing either measurement of a bug: webqsp's
node_kind and V3's kind are DIFFERENT CLASSIFIERS over the same MID.  webqsp labels a node
MID_NAMED_ENTITY from webqsp-side evidence; V3 labels the same MID CVT_MEDIATOR from the raw
dump.  And CVT_MEDIATOR -> name.parquet is 0 of 15,000 even for V3's own CVTs, because CVT
display text does not live in name.parquet at all -- which is also why the frozen record's
"CVT_MEDIATOR 0.581%" cannot be reproduced from that table and was never a claim about it.

This script tests that hypothesis directly by cross-tabulating webqsp's kind against V3's kind
for every matched MID, so the answer is a contingency table rather than an argument.

WHY lit_* SHARDS ARE SKIPPED, AND WHY THAT IS NOT AN ASSUMPTION

sorted(listdir) puts all 128 lit_* shards first, then 128 obj_*, then 6 subj_*.  The completed
presence scan reported 0 matches at its 40-, 80- and 120-shard checkpoints and its first
nonzero count at 160 -- i.e. it read every lit_* shard and matched nothing there.  The skip
rests on that finished measurement, not on the kind column's name.  It also records the trap
worth keeping: shard-count progress said 92% while the population holding 98.5% of all
ENTITY_MID nodes (the 6 subj_* shards, 131M rows) was entirely unread.  A zero from that scan,
reported early, would have been a false finding.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/webqsp_v3_reconcile.py
"""

import collections
import json
import os
import sys

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODES = os.path.join(ROOT, "data", "final_canonical", "webqsp", "nodes.jsonl")
V3 = os.path.join(ROOT, "data", "final_canonical", "freebase_v3", "canonical")
OUT = os.path.join(ROOT, "data", "final_canonical", "webqsp", "WEBQSP_V3_RECONCILE.json")

# webqsp kinds that carry a MID and are NOT already readable text
STRUCTURAL = ("CVT_MEDIATOR", "MID_NAMED_ENTITY", "UNRESOLVED_OTHER")


def webqsp_mids():
    """MID -> webqsp-side node_kind, for the structurally-labelled population."""
    want, kinds = {}, collections.Counter()
    with open(NODES, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            k = d["node_kind"]
            kinds[k] += 1
            if k in STRUCTURAL:
                # The field is source_mid.  node_id is a SYNTHETIC id ("webqsp:n0"), so
                # falling back to it silently collected 0 MIDs and the scan ran to no
                # purpose.  source_mid is also the literal string "None" on non-MID rows,
                # which is truthy -- the [:2] guard is what rejects it, so keep both.
                m = d.get("source_mid") or ""
                if m[:2] in ("m.", "g."):
                    want[m] = k
    return want, kinds


def scan(mids):
    """Cross-tabulate webqsp kind x V3 kind over obj_* and subj_* shards."""
    nd = os.path.join(V3, "nodes")
    shards = [s for s in sorted(os.listdir(nd)) if not s.startswith("lit_")]
    xtab = collections.Counter()
    matched = {}
    rows = 0
    for i, sh in enumerate(shards):
        f = pq.ParquetFile(os.path.join(nd, sh))
        for b in f.iter_batches(batch_size=1 << 18, columns=["node_id", "kind"]):
            rows += b.num_rows
            ids = b.column("node_id").to_pylist()
            ks = b.column("kind").to_pylist()
            for j, nid in enumerate(ids):
                wk = mids.get(nid)
                if wk is not None:
                    xtab[(wk, ks[j])] += 1
                    matched[nid] = ks[j]
        if (i + 1) % 20 == 0 or i + 1 == len(shards):
            print("  %d/%d non-literal shards, %d rows, %d matched"
                  % (i + 1, len(shards), rows, len(matched)), flush=True)
    return xtab, matched, rows, len(shards)


def names_for(matched):
    """Which matched MIDs have a name row, split by the V3 kind they matched as."""
    nm = os.path.join(V3, "metadata", "name.parquet")
    named = {}
    rows = 0
    f = pq.ParquetFile(nm)
    for b in f.iter_batches(batch_size=1 << 19, columns=["subject", "lexical"]):
        rows += b.num_rows
        subs = b.column("subject").to_pylist()
        lex = b.column("lexical").to_pylist()
        for j, s in enumerate(subs):
            if s in matched and s not in named:
                named[s] = lex[j]
    return named, rows


def main():
    print("collecting webqsp structural MIDs with their webqsp-side kind ...", flush=True)
    mids, wkinds = webqsp_mids()
    print("  %d MIDs" % len(mids), flush=True)

    print("scanning obj_* + subj_* shards (lit_* skipped -- the completed presence scan read "
          "all 128 and matched nothing) ...", flush=True)
    xtab, matched, rows, nsh = scan(mids)

    print("checking name.parquet for the %d matched MIDs ..." % len(matched), flush=True)
    named, nrows = names_for(matched)

    wk_tot = collections.Counter(mids.values())
    v3_kinds = sorted({v for _, v in xtab})
    print("\nCONTINGENCY  webqsp node_kind (rows) x V3 kind (cols), matched MIDs only")
    hdr = "%-20s" % "webqsp kind" + "".join(" %16s" % k[:16] for k in v3_kinds) + " %10s" % "matched"
    print(hdr)
    for wk in STRUCTURAL:
        cells = [xtab.get((wk, v), 0) for v in v3_kinds]
        print("%-20s" % wk + "".join(" %16d" % c for c in cells) + " %10d" % sum(cells))
    print("%-20s" % "TOTAL" + "".join(
        " %16d" % sum(xtab.get((w, v), 0) for w in STRUCTURAL) for v in v3_kinds)
        + " %10d" % len(matched))

    print("\nNAMING of the matched MIDs, by the V3 kind they matched as")
    print("%-20s %12s %12s %9s" % ("V3 kind", "matched", "has name", "pct"))
    by_v3 = collections.Counter(matched.values())
    named_by_v3 = collections.Counter(matched[m] for m in named)
    for v in v3_kinds:
        t = by_v3[v]
        h = named_by_v3[v]
        print("%-20s %12d %12d %8.3f%%" % (v, t, h, 100.0 * h / max(1, t)))
    print("%-20s %12d %12d %8.3f%%" % ("TOTAL", len(matched), len(named),
                                       100.0 * len(named) / max(1, len(matched))))

    print("\nWHAT A webqsp V2 WOULD ACTUALLY GAIN (names it does not already have)")
    print("%-20s %14s %14s %14s %9s" % ("webqsp kind", "population", "in V3", "V3-named",
                                        "pct of pop"))
    gain = {}
    for wk in STRUCTURAL:
        pop = wk_tot[wk]
        inv3 = sum(xtab.get((wk, v), 0) for v in v3_kinds)
        gn = sum(1 for m, k in matched.items() if mids[m] == wk and m in named)
        gain[wk] = {"population": pop, "in_v3": inv3, "v3_named": gn,
                    "pct_of_population": round(100.0 * gn / max(1, pop), 4)}
        print("%-20s %14d %14d %14d %8.4f%%" % (wk, pop, inv3, gn, 100.0 * gn / max(1, pop)))
    tp = sum(wk_tot[w] for w in STRUCTURAL)
    ti = sum(xtab.values())
    tn = len(named)
    print("%-20s %14d %14d %14d %8.4f%%" % ("TOTAL", tp, ti, tn, 100.0 * tn / max(1, tp)))

    ex = []
    for m, lx in list(named.items())[:12]:
        ex.append({"mid": m, "webqsp_kind": mids[m], "v3_kind": matched[m], "v3_name": lx})

    rec = {"RECORD": "WEBQSP_V3_RECONCILE",
           "_what": "reconciles WEBQSP_V3_PRESENCE with WEBQSP_V3_NAME_JOIN by cross-tabulating "
                    "webqsp's node_kind against V3's kind for every matched MID. MEASUREMENT "
                    "ONLY -- no dataset was modified.",
           "method": {"shards_scanned": nsh, "node_rows_scanned": rows,
                      "lit_shards_skipped": 128,
                      "skip_justification": "the completed WEBQSP_V3_PRESENCE scan read all 128 "
                                            "lit_* shards and matched 0 MIDs there",
                      "name_rows_scanned": nrows},
           "webqsp_population_by_kind": dict(wk_tot),
           "webqsp_all_node_kinds": dict(wkinds),
           "contingency_webqsp_kind_x_v3_kind":
               {"%s|%s" % k: v for k, v in sorted(xtab.items())},
           "matched_by_v3_kind": dict(by_v3),
           "named_by_v3_kind": dict(named_by_v3),
           "v2_gain_by_webqsp_kind": gain,
           "v2_gain_total": {"population": tp, "in_v3": ti, "v3_named": tn,
                             "pct_of_population": round(100.0 * tn / max(1, tp), 4)},
           "control_node_keys_to_name_table": {
               "ENTITY_MID": {"probed": 15000, "named": 13757, "pct": 91.713},
               "CVT_MEDIATOR": {"probed": 15000, "named": 0, "pct": 0.0},
               "note": "drawn from subj_* NODE rows, i.e. the key space webqsp's MIDs live in. "
                       "The 91.7% exceeds the frozen record's global 72.24% because subjects "
                       "are far likelier to be named than objects; it is a sampling difference, "
                       "not a contradiction of the record."},
           "examples": ex}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("\nwrote %s" % os.path.relpath(OUT, ROOT).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
