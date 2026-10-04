"""Did RoG actually MERGE the colliding MIDs, or simply never extract them?

    python scratchpad/final_canonical_build/webqsp_v1/collision_degree_test.py

ROG_ATTRIBUTABLE_COLLAPSE could not separate those two.  Its NSM-presence conditioning turned out
vacuous: entities_names.json ships with the NSM data, so every MID it names is in the NSM
vocabulary by construction, and all three "bounds" degenerated to the same 97,538.

Degree separates them, on the released RoG graph itself.  If RoG merged g distinct Freebase
entities onto one surface, that node inherits the union of g entities' edges, so its degree should
rise with g.  If RoG simply never extracted the other members, degree should be flat in g.

Control group: RoG surfaces that are SINGLETON names in the same file -- same provenance, same kind
of entity, same naming table, no collision.  So the comparison is not named-vs-unnamed.

Diagnostic only.  Directive section 14: do not modify topology based on these diagnostics.
"""
import json, os, time
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

NODES = "data/final_canonical/webqsp/v1/nodes.parquet"
EDGES = "data/final_canonical/webqsp/v1/edges.parquet"
NAMES = "data/final_canonical/webqsp/_acquisition/nsm/entities_names.json"
OUT = "data/final_canonical/webqsp/COLLISION_DEGREE_TEST.json"


def stats(a):
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": round(float(a.mean()), 3),
            "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": int(a.max())}


def main():
    t0 = time.time()
    eps = pq.read_table(NODES, columns=["source_rog_endpoint"]).column(
        "source_rog_endpoint").to_pylist()
    n = len(eps)
    e = pq.read_table(EDGES, columns=["src_uid", "dst_uid"])
    src = e.column("src_uid").to_numpy()
    dst = e.column("dst_uid").to_numpy()
    deg = np.bincount(src, minlength=n).astype(np.int64) + np.bincount(dst, minlength=n)
    del e, src, dst

    names = json.load(open(NAMES, encoding="utf-8"))
    gsize = defaultdict(int)
    for nm in names.values():
        gsize[nm] += 1

    grp_of = np.zeros(n, dtype=np.int32)      # 0 = not a name in the file
    for uid, s in enumerate(eps):
        g = gsize.get(s)
        if g:
            grp_of[uid] = g

    control = deg[grp_of == 1]
    coll = deg[grp_of >= 2]

    strata = {}
    for lo, hi, lab in ((2, 2, "g=2"), (3, 4, "g=3-4"), (5, 9, "g=5-9"),
                        (10, 29, "g=10-29"), (30, 10 ** 9, "g>=30")):
        sel = deg[(grp_of >= lo) & (grp_of <= hi)]
        strata[lab] = stats(sel)

    ratio = (round(float(coll.mean() / control.mean()), 3)
             if control.size and control.mean() else None)
    mono = [strata[k]["mean"] for k in ("g=2", "g=3-4", "g=5-9", "g=10-29", "g>=30")
            if strata[k]]
    monotone = all(mono[i] < mono[i + 1] for i in range(len(mono) - 1))

    doc = {
        "schema": "COLLISION_DEGREE_TEST/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "measured_on": "released RoG v1 tables (2,592,894 nodes / 8,309,195 edges)",
        "question": "Did RoG merge the colliding MIDs, or never extract them? Degree should rise "
                    "with collision-group size under merging, and be flat under non-extraction.",
        "groups": {
            "CONTROL_singleton_named_surfaces": stats(control),
            "COLLISION_surfaces_g_ge_2": stats(coll),
        },
        "MEAN_DEGREE_RATIO_collision_over_control": ratio,
        "by_collision_group_size": strata,
        "MEAN_DEGREE_MONOTONE_IN_GROUP_SIZE": bool(monotone),
        "verdict_rule_stated_before_reading": "Monotone rise in mean degree with group size = "
                                              "merging observed on the released graph. Flat = the "
                                              "extra MIDs were simply never extracted.",
        "caveat": "Popularity is a confound: entities with common names (Pilot, Home) may also be "
                  "better connected in Freebase independently of merging. Degree lift is therefore "
                  "consistent with merging and is NOT a clean causal estimate of it.",
        "directive": "Section 14 -- diagnostic only. No topology is modified.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
