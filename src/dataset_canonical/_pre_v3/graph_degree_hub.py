"""Item 5: degree and hub structure for all 18 canonical edge families, from the .npz files.

WHY THE COMMITTED TABLE HAD TO BE REDONE RATHER THAN PATCHED

  It was measured from data/canonical/*.tsv.  Two things are wrong with that source now:

  1  The 2wiki tsv files it read were data/canonical/2wiki/graph_*.tsv -- the SUPERSEDED
     398,354-node 2wiki, deleted in the canonical dedup.  2wiki's live corpus is 5,989,847
     nodes in data/canonical/2wiki_universe.  So the old row was not a stale measurement of
     the right graph, it was a measurement of a different graph.
  2  It recorded no kNN family for 2wiki.  There is one: graph2/knn.npz, 13,643,063 edges.
     No graph_knn.tsv ever existed for either 2wiki tree, because that family was built
     after the LOCKED_5_OF_5 freeze and installed straight into graph2/.  A tsv-driven scan
     could not have seen it.

  The .npz families are the authoritative form and both trees must be unioned, so this reads
  graph/ and graph2/ exactly as pointer_resolver_v2.families() does.

THE EDGE-COUNTING CONVENTION, WHICH IS NOT UNIFORM AND WILL HALVE DEGREES IF IGNORED

  Per the resolver's own contract: the ner and knn families are UNDIRECTED and store ONE row
  per pair; structural stores direction as found in the source.  So:

    ner / knn    degree(v) = rows where v appears as src OR dst.  Counting src only would
                 report half the neighbourhood and would do it asymmetrically, since which
                 endpoint got stored is an artifact of the builder.
    structural   in-degree, out-degree and total are reported separately, because for a KG
                 the asymmetry is the signal -- a CVT mediator with out-degree 8 and
                 in-degree 1 is a different object from a hub with the reverse.

  Both are reported for every family so the convention is visible in the table rather than
  buried in this docstring.

WHAT IS REPORTED, AND WHY THESE STATISTICS

  Directions A / C / D turn on hub structure, so the table carries the shape of the tail, not
  just its mean: p50 / p90 / p99 / p99.9 / max degree, the share of edge ENDPOINTS held by
  the top 1% of nodes by degree, and the isolated-node count.  A mean degree cannot tell a
  uniform graph from one where 1% of nodes touch 60% of the mass, and that difference is the
  whole question for a traversal or PPR method.

  Self-loops and repeated rows are counted, not silently deduplicated: they change degree
  and a builder that emitted them is a fact about the family.  In a TYPED family two rows
  with the same (src, dst) and different relations are PARALLEL TYPED EDGES, not duplicates
  -- the first revision of this record called them `duplicate_pairs`, a misnomer that
  Direction B cannot afford -- so typed families report `parallel_typed_edges` (distinct
  (src, dst, rel) beyond the first per pair) and `duplicate_triples` (rows identical in all
  three, the real duplicates) separately; untyped families keep `duplicate_pairs`.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/graph_degree_hub.py [dataset ...]
"""

import glob
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FC = os.path.join(ROOT, "data", "final_canonical")
OUT = os.path.join(FC, "GRAPH_DEGREE_HUB.json")
DATASETS = ["metaqa", "webqsp", "hotpotqa", "2wiki", "musique", "squad"]
UNDIRECTED = ("ner", "knn")


def graph2_supersedes(ds):
    """Families whose graph2 manifest explicitly declares it supersedes the frozen copy.

    Precedence (pointer_resolver_v2): frozen wins for a family both trees declare unless the
    graph2 manifest carries `supersedes.<family>`. Today that is musique/knn only.
    """
    mp = os.path.join(FC, ds, "graph2", "GRAPH_MANIFEST.json")
    if not os.path.isfile(mp):
        return {}
    man = json.load(io.open(mp, encoding="utf-8"))
    return {f: d for f, d in (man.get("supersedes") or {}).items()
            if (man.get("families") or {}).get(f, {}).get("present")}


def n_nodes(ds):
    z = np.load(os.path.join(FC, ds, "pointer_index", "dense.npz"))
    return int(z["row"].shape[0])


def stats(deg):
    nz = deg[deg > 0]
    q = np.percentile(nz, [50, 90, 99, 99.9]) if nz.size else [0, 0, 0, 0]
    order = np.argsort(-deg)
    top1 = max(1, deg.shape[0] // 100)
    return {"mean_over_all_nodes": round(float(deg.mean()), 4),
            "mean_over_nonisolated": round(float(nz.mean()) if nz.size else 0.0, 4),
            "p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), "p99_9": float(q[3]),
            "max": int(deg.max()) if deg.size else 0,
            "isolated_nodes": int((deg == 0).sum()),
            "isolated_pct": round(100.0 * float((deg == 0).sum()) / max(1, deg.shape[0]), 4),
            "top1pct_endpoint_share_pct":
                round(100.0 * float(deg[order[:top1]].sum()) / max(1, float(deg.sum())), 4)}


def main():
    args = sys.argv[1:]
    why = None
    if "--why" in args:
        i = args.index("--why")
        why = args[i + 1]
        args = args[:i] + args[i + 2:]
    want = args
    if os.path.isfile(OUT) and not why:
        sys.exit("%s exists: pass --why \"<what changed and why>\" so the revision_history "
                 "entry states the reason for this revision instead of inheriting the last one"
                 % OUT)
    res = {}
    tot_edges = 0
    print("%-9s %-7s %-11s %10s %9s %8s %7s %7s %8s %9s %8s %9s"
          % ("dataset", "family", "tree", "edges", "nodes", "mean", "p50", "p90", "p99",
             "max", "iso%", "top1%share"))
    for ds in [d for d in DATASETS if not want or d in want]:
        nn = n_nodes(ds)
        res[ds] = {"nodes": nn, "families": {}}
        sup = graph2_supersedes(ds)
        for tree in ("graph", "graph2"):
            for p in sorted(glob.glob(os.path.join(FC, ds, tree, "*.npz"))):
                fam = os.path.basename(p)[:-4]
                # precedence: a frozen family that graph2 supersedes is still measured, under
                # <family>@frozen_superseded, but it is not the served family and is not
                # counted in the total; a graph2 duplicate WITHOUT a declaration is measured
                # under <family>@graph2_shadowed and the frozen one stays the answer
                key = fam
                if tree == "graph" and fam in sup:
                    key = fam + "@frozen_superseded"
                elif tree == "graph2" and fam in res[ds]["families"]:
                    key = fam + "@graph2_shadowed"
                served = (key == fam)
                z = np.load(p)
                src, dst = z["src"], z["dst"]
                m = int(src.shape[0])
                if served:
                    tot_edges += m
                undirected = fam in UNDIRECTED

                if undirected:
                    deg = (np.bincount(src, minlength=nn).astype(np.int64)
                           + np.bincount(dst, minlength=nn).astype(np.int64))
                    d_main = stats(deg)
                    extra = {"convention": "UNDIRECTED, one row per pair; degree counts both "
                                           "endpoints"}
                else:
                    outd = np.bincount(src, minlength=nn).astype(np.int64)
                    ind = np.bincount(dst, minlength=nn).astype(np.int64)
                    deg = outd + ind
                    d_main = stats(deg)
                    extra = {"convention": "DIRECTED as found in the source",
                             "out_degree": stats(outd), "in_degree": stats(ind),
                             "distinct_relations": int(np.unique(z["rel"]).shape[0])
                             if "rel" in z.files else None}

                self_loops = int((src == dst).sum())
                pair = (src.astype(np.int64) << np.int64(32)) | dst.astype(np.int64)
                n_pairs = int(np.unique(pair).shape[0])
                rows_beyond_first_per_pair = m - n_pairs
                if "rel" in z.files:
                    # typed: split the repeat count into parallel typed edges vs true duplicates
                    tri = np.stack([src.astype(np.int64), dst.astype(np.int64),
                                    z["rel"].astype(np.int64)], axis=1)
                    n_triples = int(np.unique(tri, axis=0).shape[0])
                    rep_count = {"parallel_typed_edges": n_triples - n_pairs,
                                 "duplicate_triples": m - n_triples,
                                 "rows_beyond_first_per_pair": rows_beyond_first_per_pair,
                                 "_identity": "rows_beyond_first_per_pair = parallel_typed_edges "
                                              "+ duplicate_triples; the first revision reported "
                                              "this sum as `duplicate_pairs`"}
                else:
                    rep_count = {"duplicate_pairs": rows_beyond_first_per_pair}
                w = {}
                if "weight" in z.files:
                    ww = z["weight"]
                    w = {"weight_min": round(float(ww.min()), 6),
                         "weight_max": round(float(ww.max()), 6),
                         "weight_mean": round(float(ww.mean()), 6)}

                rec = {"tree": tree, "edges": m, "self_loops": self_loops, "served": served}
                if not served:
                    rec["not_served_because"] = (
                        "superseded by graph2/%s.npz (declared in graph2/GRAPH_MANIFEST.json)"
                        % fam if key.endswith("@frozen_superseded")
                        else "frozen copy wins; graph2 duplicate carries no supersedes declaration")
                rec.update(rep_count)
                rec["degree"] = d_main
                rec.update(extra)
                rec.update(w)
                res[ds]["families"][key] = rec
                print("%-9s %-7s %-11s %10d %9d %8.2f %7.0f %7.0f %8.0f %9d %8.3f %9.2f"
                      % (ds, key[:7] if served else fam + "*", tree, m, nn,
                         d_main["mean_over_all_nodes"],
                         d_main["p50"], d_main["p90"], d_main["p99"], d_main["max"],
                         d_main["isolated_pct"], d_main["top1pct_endpoint_share_pct"]),
                      flush=True)
                del z, src, dst, deg, pair

    print("")
    print("total edges over all 18 served families: %d  (* = measured, not served)" % tot_edges)
    history = []
    if os.path.isfile(OUT):
        import hashlib, datetime
        prior_bytes = open(OUT, "rb").read()
        try:
            history = list(json.loads(prior_bytes).get("revision_history") or [])
        except Exception:
            history = []
        history.append({"sha256": hashlib.sha256(prior_bytes).hexdigest(),
                        "superseded_utc": datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "why": why})
    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump({"RECORD": "GRAPH_DEGREE_HUB",
                   "revision_history": history,
                   "_what": "degree and hub structure for all 18 canonical edge families, read "
                            "from the authoritative .npz in graph/ and graph2/. Supersedes the "
                            "committed table that was measured from data/canonical/*.tsv. "
                            "MEASUREMENT ONLY -- no graph was modified.",
                   "why_redone": {
                       "wrong_2wiki": "the old table read data/canonical/2wiki/graph_*.tsv, the "
                                      "superseded 398,354-node 2wiki, since deleted; the live "
                                      "corpus is 5,989,847 nodes in 2wiki_universe",
                       "missing_2wiki_knn": "the old table recorded no kNN for 2wiki; graph2/"
                                            "knn.npz has 13,643,063 edges and no graph_knn.tsv "
                                            "ever existed for either 2wiki tree"},
                   "conventions": {
                       "ner_knn": "UNDIRECTED, one row per pair; degree counts both endpoints",
                       "structural": "DIRECTED as found in the source; in/out reported separately"},
                   "total_edges_all_18_families": tot_edges,
                   "results": res}, f, indent=1)
    print("wrote %s" % os.path.relpath(OUT, ROOT).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
