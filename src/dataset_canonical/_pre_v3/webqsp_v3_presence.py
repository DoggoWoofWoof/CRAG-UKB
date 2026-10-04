"""Are webqsp's unresolved MIDs even IN our Freebase, or just unnamed there?

WEBQSP_V3_NAME_JOIN measured 0 of 1,652,618 webqsp structural MIDs carrying a name in the
frozen V3 name table, with a 100% positive control drawn from that table. But that control is
ONE-SIDED: it proves the join path reaches V3's key space, not that webqsp's keys live in the
same space. A perfect zero is exactly where that distinction decides the finding, so this
script asks the prior question -- presence, not naming -- and splits both populations by MID
namespace, because the namespaces are the suspected mechanism:

  m.  the Freebase-era MID space; the official RDF dump (our source of record) is almost
      entirely this
  g.  the Google-era space; V3_IDIR_G_NAMESPACE_DIAGNOSTIC found 0 g-prefixed nodes in the
      first 6 V3 shards against IDIR's 7,046,936, i.e. the official dump largely omits it

"unnamed in our Freebase" and "absent from our Freebase" are very different findings: the
first says a second version of webqsp would gain nothing; the second says the RoG graph's
unresolved population lies outside our corpus entirely and no amount of name recovery would
reach it.

    PYTHONHASHSEED=0 python src/dataset_canonical/webqsp_v3_presence.py
"""
import collections
import datetime
import io
import json
import os
import sys

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODES = os.path.join(ROOT, "data", "final_canonical", "webqsp", "nodes.jsonl")
V3 = os.path.join(ROOT, "data", "final_canonical", "freebase_v3", "canonical")
OUT = os.path.join(ROOT, "data", "final_canonical", "webqsp", "WEBQSP_V3_PRESENCE.json")
STRUCTURAL = ("CVT_MEDIATOR", "MID_NAMED_ENTITY", "UNRESOLVED_OTHER")


def ns(mid):
    return mid[:2] if mid[:2] in ("m.", "g.") else "other"


def webqsp_mids():
    want, kinds = {}, collections.Counter()
    for line in io.open(NODES, encoding="utf-8"):
        d = json.loads(line)
        k = d["node_kind"]
        kinds[k] += 1
        if k in STRUCTURAL and d.get("source_mid"):
            want[d["source_mid"]] = k
    return want, kinds


def scan_nodes(mids):
    """One pass over the V3 node universe; record which webqsp MIDs appear, and V3's own
    namespace census so the denominator is visible."""
    nd = os.path.join(V3, "nodes")
    present = set()
    v3ns = collections.Counter()
    rows = 0
    shards = sorted(os.listdir(nd))
    for i, sh in enumerate(shards):
        f = pq.ParquetFile(os.path.join(nd, sh))
        for b in f.iter_batches(batch_size=1 << 18, columns=["node_id", "kind"]):
            ids = b.column("node_id").to_pylist()
            ks = b.column("kind").to_pylist()
            rows += len(ids)
            for nid, k in zip(ids, ks):
                if k != "ENTITY_MID":
                    continue
                v3ns[ns(nid)] += 1
                if nid in mids:
                    present.add(nid)
        if (i + 1) % 40 == 0 or i + 1 == len(shards):
            print("  %d/%d shards, %d rows, %d webqsp MIDs found so far"
                  % (i + 1, len(shards), rows, len(present)), flush=True)
    return present, v3ns, rows


def main():
    print("collecting webqsp structural MIDs ...", flush=True)
    mids, kinds = webqsp_mids()
    wns = collections.Counter(ns(m) for m in mids)
    print("  %d MIDs; webqsp namespace census: %s" % (len(mids), dict(wns)), flush=True)

    print("scanning the V3 node universe (301,977,131 nodes) ...", flush=True)
    present, v3ns, rows = scan_nodes(mids)
    print("  V3 ENTITY_MID namespace census: %s" % dict(v3ns), flush=True)

    per = collections.Counter()
    tot = collections.Counter()
    perk = collections.Counter()
    totk = collections.Counter()
    for m, k in mids.items():
        tot[ns(m)] += 1
        totk[k] += 1
        if m in present:
            per[ns(m)] += 1
            perk[k] += 1

    print("\n%-8s %14s %14s %9s" % ("namespace", "webqsp MIDs", "in V3 nodes", "pct"))
    for k in sorted(tot):
        print("%-8s %14d %14d %8.3f%%" % (k, tot[k], per[k], 100.0 * per[k] / tot[k]))
    print("%-8s %14d %14d %8.3f%%" % ("TOTAL", sum(tot.values()), sum(per.values()),
                                      100.0 * sum(per.values()) / max(1, sum(tot.values()))))
    print("\n%-20s %14s %14s %9s" % ("node_kind", "webqsp MIDs", "in V3 nodes", "pct"))
    for k in STRUCTURAL:
        if totk[k]:
            print("%-20s %14d %14d %8.3f%%" % (k, totk[k], perk[k], 100.0 * perk[k] / totk[k]))

    verdict = ("ABSENT_FROM_OUR_FREEBASE" if sum(per.values()) < 0.01 * sum(tot.values())
               else "PRESENT_BUT_UNNAMED" if sum(per.values()) > 0.9 * sum(tot.values())
               else "MIXED")
    rec = {
        "RECORD": "WEBQSP_V3_PRESENCE",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_what": ("presence, not naming: are webqsp's structurally-labelled MIDs in the frozen "
                  "V3 node universe at all? MEASUREMENT ONLY."),
        "webqsp": {"by_node_kind": dict(kinds), "structural_mids": len(mids),
                   "namespace_census": dict(wns)},
        "v3": {"entity_mid_namespace_census": dict(v3ns), "node_rows_scanned": rows},
        "presence_by_namespace": {k: {"webqsp_mids": tot[k], "in_v3_nodes": per[k],
                                      "pct": round(100.0 * per[k] / tot[k], 4)}
                                  for k in sorted(tot)},
        "presence_by_node_kind": {k: {"webqsp_mids": totk[k], "in_v3_nodes": perk[k],
                                      "pct": round(100.0 * perk[k] / totk[k], 4)}
                                  for k in STRUCTURAL if totk[k]},
        "VERDICT": verdict,
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("\nVERDICT: %s" % verdict)
    print("wrote %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
