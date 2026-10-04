"""One join, two draws: settle whether the webqsp name-join zero is real.

THE CONTRADICTION THIS EXISTS TO KILL

  WEBQSP_V3_RECONCILE  1,618,950 webqsp MIDs matched in the V3 node table, of which 128,208
                       are V3 ENTITY_MID -- and 0 of the 1,618,950 have a name row. All
                       68,362,456 name rows were scanned, so it is not a truncated read.
  control (earlier)    15,000 ENTITY_MID node_ids drawn from the same subj_* shards -> 13,757
                       named (91.713%).

Both cannot be right. A 0% and a 91.7% over the same key space is a key-space or code-path
difference, not a population difference -- and ZERO_JOIN_INVARIANT says the zero is the
suspect until a positive control that REACHED THE SAME POPULATION says otherwise. The earlier
control was a separate script, so "same key space" was an argument, not a measurement.

So: draw both sets from the SAME shard walk, and join both through the SAME pass over
name.parquet. Then a disagreement can only be the populations, and an agreement can only be
the join.

  A  the first N ENTITY_MID subj_* node_ids, webqsp-agnostic       (reproduces the control)
  B  the first N ENTITY_MID subj_* node_ids that ARE webqsp MIDs   (the disputed population)

B is a subset of A's universe by construction, so if A joins and B does not, the difference
is a real property of webqsp's MIDs and not a bug. If neither joins, the earlier control was
wrong. If both join, the reconcile's name step was wrong.

Only subj_* is walked: the reconcile showed the matches are there (104 of 1,618,950 by shard
120 of 134, the rest in the final block), and the control drew from subj_* too.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/webqsp_name_join_control.py
"""

import collections
import json
import os
import sys

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NODES = os.path.join(ROOT, "data", "final_canonical", "webqsp", "nodes.jsonl")
V3 = os.path.join(ROOT, "data", "final_canonical", "freebase_v3", "canonical")
OUT = os.path.join(ROOT, "data", "final_canonical", "webqsp", "WEBQSP_NAME_JOIN_CONTROL.json")

N = 15000
STRUCTURAL = ("CVT_MEDIATOR", "MID_NAMED_ENTITY", "UNRESOLVED_OTHER")


def webqsp_mids():
    want = {}
    with open(NODES, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["node_kind"] in STRUCTURAL:
                m = d.get("source_mid") or ""
                if m[:2] in ("m.", "g."):
                    want[m] = d["node_kind"]
    return want


def draws(wmids):
    """A: ENTITY_MID subj_* keys, webqsp-agnostic.  B: the same, restricted to webqsp MIDs."""
    nd = os.path.join(V3, "nodes")
    shards = [s for s in sorted(os.listdir(nd)) if s.startswith("subj_")]
    A, B = {}, {}
    rows = 0
    for sh in shards:
        f = pq.ParquetFile(os.path.join(nd, sh))
        for b in f.iter_batches(batch_size=1 << 18, columns=["node_id", "kind"]):
            rows += b.num_rows
            ids = b.column("node_id").to_pylist()
            ks = b.column("kind").to_pylist()
            for j, nid in enumerate(ids):
                if ks[j] != "ENTITY_MID":
                    continue
                if len(A) < N:
                    A[nid] = 1
                if nid in wmids and len(B) < N:
                    B[nid] = wmids[nid]
            if len(A) >= N and len(B) >= N:
                break
        print("  %-28s subj rows=%d  |A|=%d  |B|=%d" % (sh, rows, len(A), len(B)), flush=True)
        if len(A) >= N and len(B) >= N:
            break
    return A, B, rows


def join(keys):
    """ONE pass over name.parquet answering for every key set at once."""
    nm = os.path.join(V3, "metadata", "name.parquet")
    hit = {k: set() for k in keys}
    rows = 0
    f = pq.ParquetFile(nm)
    for b in f.iter_batches(batch_size=1 << 19, columns=["subject", "lexical"]):
        rows += b.num_rows
        subs = b.column("subject").to_pylist()
        for s in subs:
            for k, ks in keys.items():
                if s in ks:
                    hit[k].add(s)
        if rows % (1 << 24) < (1 << 19):
            print("  name rows %d  " % rows
                  + "  ".join("%s=%d" % (k, len(v)) for k, v in hit.items()), flush=True)
    return hit, rows


def main():
    print("collecting webqsp MIDs ...", flush=True)
    wmids = webqsp_mids()
    print("  %d" % len(wmids), flush=True)

    print("drawing A (webqsp-agnostic) and B (webqsp) from the SAME subj_* walk ...", flush=True)
    A, B, nrows = draws(wmids)

    print("joining both draws in ONE pass over name.parquet ...", flush=True)
    hit, namerows = join({"A_agnostic": set(A), "B_webqsp": set(B)})

    print("")
    print("%-14s %8s %8s %9s" % ("draw", "keys", "named", "pct"))
    res = {}
    for k, ks in (("A_agnostic", A), ("B_webqsp", B)):
        h = len(hit[k])
        res[k] = {"keys": len(ks), "named": h,
                  "pct": round(100.0 * h / max(1, len(ks)), 3)}
        print("%-14s %8d %8d %8.3f%%" % (k, len(ks), h, 100.0 * h / max(1, len(ks))))

    bk = collections.Counter(B[m] for m in hit["B_webqsp"])
    print("")
    print("B hits by webqsp-side kind: %s" % (dict(bk) or "none"))

    a, b = res["A_agnostic"]["pct"], res["B_webqsp"]["pct"]
    if a > 50 and b > 50:
        v = ("JOIN_SOUND_RECONCILE_NAME_STEP_WRONG -- both draws join, so the reconcile's "
             "0 of 1,618,950 is a bug in that script's name step, not a property of webqsp")
    elif a > 50 and b < 1:
        v = ("REAL_PROPERTY_OF_WEBQSP_MIDS -- an agnostic draw from the same shards joins and "
             "the webqsp draw does not, through identical code. The zero survives the invariant")
    elif a < 1 and b < 1:
        v = ("EARLIER_CONTROL_WRONG -- neither draw joins through this code path, so the "
             "91.713% control cannot stand and nothing about a V2 gain is established")
    else:
        v = "INCONCLUSIVE at these rates -- read the table, do not summarise it"
    print("")
    print("VERDICT: %s" % v)

    rec = {"RECORD": "WEBQSP_NAME_JOIN_CONTROL",
           "_what": "settles the WEBQSP_V3_RECONCILE name-join zero by putting a webqsp-agnostic "
                    "draw and the webqsp draw through ONE pass over name.parquet. MEASUREMENT "
                    "ONLY -- no dataset was modified.",
           "invariant": "ZERO_JOIN_INVARIANT: a clean zero from a large exact source is not "
                        "accepted until a positive control that reached the same population "
                        "returns nonzero through the same code path",
           "method": {"n_per_draw": N, "subj_rows_walked": nrows, "name_rows_scanned": namerows,
                      "draw_A": "first N ENTITY_MID subj_* node_ids, webqsp-agnostic",
                      "draw_B": "first N ENTITY_MID subj_* node_ids that are webqsp MIDs"},
           "results": res,
           "B_hits_by_webqsp_kind": dict(bk),
           "VERDICT": v}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("wrote %s" % os.path.relpath(OUT, ROOT).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
