"""Item 3: name the gold field, and prove it carries final_canonical ids that RESOLVE.

WHAT IS ALREADY KNOWN FROM THE SCHEMA

  The field is gold_node_ids, and it is uniform across all six datasets -- same name, same
  shape (a list of strings), same "<dataset>:<local>" namespace:

    metaqa    metaqa:e32512                    hotpotqa  hotpotqa:c27290714
    webqsp    webqsp:n410653                   2wiki     2wiki:c46961177
    musique   musique:13979297d8bbfff5a3afe0bc  squad     squad:addf6eb35947b70a6c9aa457

WHAT THE SCHEMA DOES NOT ESTABLISH

  That those strings are the SAME id space as nodes.jsonl, and that every one of them
  resolves to a canonical position.  A headroom row needs per-query gold as canonical
  POSITIONS -- position i is line i of nodes.jsonl and pointer row i -- so an id that looks
  right but resolves to nothing silently drops a gold and inflates every ceiling computed
  from it.

  Two ways it could look right and be wrong:
    - the local part is NOT the position.  webqsp:n410653 carries gold_positions [410653]
      alongside it, so for webqsp the number IS the position; nothing says that generalises,
      and metaqa:e32512 / musique:<hex> plainly do not encode a line number the same way.
      So resolution is done by an actual id -> position map read from nodes.jsonl, never by
      parsing the id.
    - the ids resolve but belong to the pre-rebuild substrate.  2wiki was rebuilt from
      398,354 to 5,989,847 nodes and hotpotqa carries TEXTUALIZATION_REV 2, so a stale gold
      file would still resolve as strings while pointing at different documents.  The guard
      is that the map is built from the CURRENT nodes.jsonl and its length must equal the
      pointer-index length, which is checked.

WHAT IS REPORTED

  Per dataset and split: queries, gold-bearing queries, total gold refs, refs that resolve,
  refs that do not, and the max resolved position against the node count.  A split whose
  gold arrays are empty by construction (a hidden test set) is reported as GOLD_ABSENT rather
  than as a failure -- that is a property of the benchmark, not a defect here.

  node_id is extracted by string scan rather than json.loads: 14M lines of full JSON parse is
  ~20 minutes and buys nothing, since the only field needed is one flat string.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/gold_field_check.py [dataset ...]
"""

import glob
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
OUT = os.path.join(FC, "GOLD_FIELD_CHECK.json")
GOLD = "gold_node_ids"
DATASETS = ["metaqa", "webqsp", "hotpotqa", "2wiki", "musique", "squad"]
KEY = '"node_id":'


def id_positions(ds):
    """id -> canonical position, read from the CURRENT nodes.jsonl by string scan."""
    p = os.path.join(FC, ds, "nodes.jsonl")
    m = {}
    with io.open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            j = line.find(KEY)
            if j < 0:
                raise RuntimeError("%s line %d has no node_id" % (ds, i))
            a = line.index('"', j + len(KEY)) + 1
            b = line.index('"', a)
            m[line[a:b]] = i
    return m


def pointer_len(ds):
    z = np.load(os.path.join(FC, ds, "pointer_index", "dense.npz"))
    return int(z["row"].shape[0])


def main():
    want = sys.argv[1:]
    res, fails = {}, []
    for ds in [d for d in DATASETS if not want or d in want]:
        print("=" * 96, flush=True)
        print("%s  building id -> position map from nodes.jsonl ..." % ds, flush=True)
        m = id_positions(ds)
        pl = pointer_len(ds)
        agree = (len(m) == pl)
        print("   nodes=%d  pointer_rows=%d  agree=%s" % (len(m), pl, agree), flush=True)
        if not agree:
            fails.append("%s: id map %d != pointer rows %d" % (ds, len(m), pl))

        # does the local part of the id equal the position? measured, not assumed.
        sample = list(m.items())[:20000]
        numeric_match = 0
        for k, v in sample:
            tail = k.split(":", 1)[-1]
            digits = "".join(c for c in tail if c.isdigit())
            if digits and digits.isdigit() and int(digits) == v:
                numeric_match += 1
        pct_numeric = 100.0 * numeric_match / max(1, len(sample))

        dsres = {"nodes": len(m), "pointer_rows": pl, "map_matches_pointer": agree,
                 "gold_field": GOLD,
                 "id_local_part_equals_position_pct": round(pct_numeric, 2),
                 "splits": {}}
        for qp in sorted(glob.glob(os.path.join(FC, ds, "queries", "*.jsonl"))):
            sp = os.path.basename(qp)[:-6]
            nq = ngold = nref = nres = 0
            miss = []
            mx = -1
            with io.open(qp, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    nq += 1
                    g = d.get(GOLD)
                    if g is None:
                        continue
                    if g:
                        ngold += 1
                    for x in g:
                        nref += 1
                        pos = m.get(x)
                        if pos is None:
                            if len(miss) < 8:
                                miss.append(x)
                        else:
                            nres += 1
                            mx = max(mx, pos)
            if nref == 0:
                v = "GOLD_ABSENT"
            elif nres == nref:
                v = "ALL_RESOLVE"
            else:
                v = "UNRESOLVED_GOLD"
                fails.append("%s.%s: %d of %d gold refs do not resolve"
                             % (ds, sp, nref - nres, nref))
            print("   %-11s q=%-7d gold_q=%-7d refs=%-8d resolved=%-8d max_pos=%-9d %s"
                  % (sp, nq, ngold, nref, nres, mx, v), flush=True)
            if miss:
                print("        unresolved examples: %s" % miss, flush=True)
            dsres["splits"][sp] = {"queries": nq, "queries_with_gold": ngold,
                                   "gold_refs": nref, "gold_refs_resolved": nres,
                                   "gold_refs_unresolved": nref - nres,
                                   "max_resolved_position": mx,
                                   "unresolved_examples": miss, "verdict": v}
        res[ds] = dsres
        del m
        with io.open(OUT, "w", encoding="utf-8") as f:
            json.dump({"RECORD": "GOLD_FIELD_CHECK",
                       "_what": "names the gold field and proves it resolves into the CURRENT "
                                "canonical id space. MEASUREMENT ONLY.",
                       "gold_field": GOLD,
                       "resolution_method": "id -> position map read from the current "
                                            "nodes.jsonl; the id is never parsed for a position",
                       "results": res, "failures": fails,
                       "complete": False}, f, indent=1)

    print("")
    print("=" * 96)
    print("GOLD FIELD: %s, uniform across all six" % GOLD)
    print("ALL GOLD REFS RESOLVE IN EVERY LABELLED SPLIT: %s" % ("YES" if not fails else "NO"))
    for x in fails:
        print("   %s" % x)
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"RECORD": "GOLD_FIELD_CHECK", "gold_field": GOLD,
                   "_what": "names the gold field and proves it resolves into the CURRENT "
                            "canonical id space. MEASUREMENT ONLY.",
                   "resolution_method": "id -> position map read from the current nodes.jsonl; "
                                        "the id is never parsed for a position",
                   "results": res, "failures": fails, "complete": True}, f, indent=1)
    print("wrote %s" % os.path.relpath(OUT, ROOT).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
