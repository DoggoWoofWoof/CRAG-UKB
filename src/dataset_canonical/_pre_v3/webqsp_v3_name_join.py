"""What would a V3-named webqsp actually gain? -- measured, not estimated.

The shipped webqsp corpus renders 1,652,618 of its 2,592,894 nodes (63.74%) from a
STRUCTURAL_SCHEMA_LABEL, not from a name: "unnamed location entity", "topic record. topic
notable for: X.". That rendering was frozen on 2026-09-06, one day BEFORE
CRAG_FREEBASE_CANONICAL (V3) froze, so the corpus has never been joined against our own
Freebase name layer. A second version of webqsp would be exactly that join -- so the
question "is it worth building" is a coverage number, and this script measures it.

Nothing is written into any dataset. This is measurement only.

The join is reported per node_kind, because the kinds are not the same question:
  CVT_MEDIATOR       Freebase gives mediator records no name (V3 CVT display coverage is
                     0.581%), so a near-zero hit here is the CORRECT answer, not a failure.
  MID_NAMED_ENTITY   these are real entities rendered as "unnamed <type> entity". A hit here
                     is a genuine gain.
  UNRESOLVED_OTHER   same, minus a usable type.

ZERO_JOIN_INVARIANT: a zero or near-zero result from a large exact source is not acceptable
without a positive control proving the join path REACHES the population. Here the control
samples MID subjects out of the name table itself and looks them up by the identical path; it
must return ~100%. Without that, "0.6% of CVTs have names" and "my join is broken" are
indistinguishable.

    PYTHONHASHSEED=0 python src/dataset_canonical/webqsp_v3_name_join.py
"""
import collections
import datetime
import io
import json
import os
import random
import sys

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODES = os.path.join(ROOT, "data", "final_canonical", "webqsp", "nodes.jsonl")
NAME = os.path.join(ROOT, "data", "final_canonical", "freebase_v3", "canonical",
                    "metadata", "name.parquet")
OUT = os.path.join(ROOT, "data", "final_canonical", "webqsp",
                   "WEBQSP_V3_NAME_JOIN.json")
STRUCTURAL = ("CVT_MEDIATOR", "MID_NAMED_ENTITY", "UNRESOLVED_OTHER")
CONTROL_N = 20000


def collect_webqsp():
    """source_mid per node_kind, for the structurally-labelled population."""
    want = {}
    kinds = collections.Counter()
    no_mid = collections.Counter()
    for line in io.open(NODES, encoding="utf-8"):
        d = json.loads(line)
        k = d["node_kind"]
        kinds[k] += 1
        if k not in STRUCTURAL:
            continue
        sm = d.get("source_mid")
        if not sm:
            no_mid[k] += 1
            continue
        want[sm] = k
    return want, kinds, no_mid


def join(mids, control):
    """One streaming pass over the 68.4M-row name table, both populations at once."""
    hit = {}
    ctl_hit = set()
    f = pq.ParquetFile(NAME)
    rows = 0
    mid_subjects = []
    for b in f.iter_batches(batch_size=1 << 18, columns=["subject", "lexical"]):
        subj = b.column("subject").to_pylist()
        lex = b.column("lexical").to_pylist()
        rows += len(subj)
        for s, l in zip(subj, lex):
            if s in mids and s not in hit:
                hit[s] = l
            if s in control:
                ctl_hit.add(s)
            # collect a reservoir of real MID subjects to BUILD the control from
            if control is CONTROL_SENTINEL and len(mid_subjects) < CONTROL_N and s[:2] in ("m.", "g."):
                mid_subjects.append(s)
    return hit, ctl_hit, rows, mid_subjects


CONTROL_SENTINEL = frozenset()


def sample_control():
    """Draw MID subjects straight out of the name table -- the population the join must reach."""
    f = pq.ParquetFile(NAME)
    pool = []
    for b in f.iter_batches(batch_size=1 << 18, columns=["subject"]):
        for s in b.column("subject").to_pylist():
            if s[:2] in ("m.", "g."):
                pool.append(s)
        if len(pool) >= CONTROL_N * 20:
            break
    random.Random(0).shuffle(pool)
    return set(pool[:CONTROL_N])


def main():
    print("collecting webqsp source_mid ...")
    mids, kinds, no_mid = collect_webqsp()
    print("  nodes by kind: %s" % dict(kinds))
    print("  structural population with a source_mid: %d" % len(mids))
    print("  structural rows WITHOUT a source_mid:    %s" % dict(no_mid))

    print("sampling the positive control out of the name table ...")
    control = sample_control()
    print("  control size: %d real MID subjects" % len(control))

    print("joining against %s ..." % os.path.relpath(NAME, ROOT))
    hit, ctl_hit, rows, _ = join(mids, control)
    print("  scanned %d name rows" % rows)

    ctl_pct = 100.0 * len(ctl_hit) / max(1, len(control))
    print("\nPOSITIVE CONTROL: %d/%d = %.2f%% -- %s"
          % (len(ctl_hit), len(control), ctl_pct,
             "join path reaches the population" if ctl_pct > 99.0
             else "BROKEN JOIN, every number below is uninterpretable"))

    per = collections.Counter()
    tot = collections.Counter()
    for m, k in mids.items():
        tot[k] += 1
        if m in hit:
            per[k] += 1
    print("\n%-20s %12s %12s %9s" % ("node_kind", "with a MID", "V3 has name", "pct"))
    for k in STRUCTURAL:
        if tot[k]:
            print("%-20s %12d %12d %8.2f%%" % (k, tot[k], per[k], 100.0 * per[k] / tot[k]))
    print("%-20s %12d %12d %8.2f%%" % ("TOTAL", sum(tot.values()), sum(per.values()),
                                       100.0 * sum(per.values()) / max(1, sum(tot.values()))))

    ex = {}
    for m, k in mids.items():
        if m in hit and len(ex.setdefault(k, [])) < 4:
            ex[k].append({"mid": m, "v3_name": hit[m]})
    for k, v in ex.items():
        print("\n  would become, for %s:" % k)
        for e in v:
            print("    %-16s -> %s" % (e["mid"], e["v3_name"]))

    rec = {
        "RECORD": "WEBQSP_V3_NAME_JOIN",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_what": ("measures what a V3-named second version of webqsp would gain. MEASUREMENT "
                  "ONLY -- no dataset was modified."),
        "corpus": {"nodes": sum(kinds.values()), "by_node_kind": dict(kinds),
                   "structural_population_with_mid": len(mids),
                   "structural_rows_without_mid": dict(no_mid)},
        "name_table": {"path": os.path.relpath(NAME, ROOT), "rows_scanned": rows},
        "positive_control": {"n": len(control), "hit": len(ctl_hit), "pct": round(ctl_pct, 4),
                             "drawn_from": "MID subjects sampled out of the name table itself",
                             "interpretable": ctl_pct > 99.0},
        "coverage_by_node_kind": {k: {"with_mid": tot[k], "v3_has_name": per[k],
                                      "pct": round(100.0 * per[k] / tot[k], 4) if tot[k] else None}
                                  for k in STRUCTURAL},
        "examples": ex,
    }
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
