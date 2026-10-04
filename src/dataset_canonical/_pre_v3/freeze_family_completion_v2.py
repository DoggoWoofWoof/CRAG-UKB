# -*- coding: utf-8 -*-
"""LOCKED_6_OF_6_FAMILY_COMPLETION_V2: V1 plus the musique kNN supersession.

WHAT V2 ADDS OVER V1
--------------------
V1 (2026-09-09) recorded, and did not fix, that the frozen musique kNN is STALE:
frozen_knn_self_consistency.n_gt_1e-2 = 4402 of 266,488 edges do not reproduce from the
dataset's own resolved vectors. It also recorded that the established pipeline, run over the
resolved vectors, reproduces the frozen edge set up to tie churn (TIE_CHURN_ONLY).

V2 carries the five other datasets UNCHANGED (their blocks are copied verbatim from V1 and V1
itself is pinned in BUILDS_ON), and for musique:
  - family_source.knn becomes "graph2": the rebuilt family in musique/graph2/knn.npz, whose
    manifest carries an explicit `supersedes.knn` block naming the frozen file and its sha256
  - `added.knn` pins the new .npz by digest, `graph2_manifest_sha256` pins the manifest
  - coverage.knn is re-measured over the new family; the frozen family's coverage is kept
    under `superseded_frozen_families.knn`
  - new_knn_self_consistency is the every-edge audit of the replacement
    (data/_family_v1/MUSIQUE_GRAPH2_KNN_WEIGHT_AUDIT.json); frozen_knn_self_consistency and
    knn_pipeline_parity_vs_frozen are carried unchanged, because they are the evidence

WHAT IS NOT CHANGED
-------------------
No byte under any <ds>/graph/ moves. musique/graph/knn.npz stays exactly as LOCKED_5_OF_5
pinned it and stays verifiable there; it is superseded by declaration, not deleted or
rewritten. V1 is not edited -- a correction is a new record with its own hash.

PRECEDENCE RULE THIS RECORD INTRODUCES
--------------------------------------
frozen wins whenever graph/ and graph2/ both declare a family, UNLESS the graph2 manifest
declares `supersedes.<family>`. The rule lives in pointer_resolver_v2 (supersedes(),
family_source(), CanonicalGraph()) and is honoured by substrate_map, ukb_manifest,
verify_manifest, handoff_readiness and graph_degree_hub.

Run:
  PYTHONHASHSEED=0 python src/dataset_canonical/freeze_family_completion_v2.py
"""
import copy
import datetime
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
sys.path.insert(0, os.path.abspath("data/final_canonical"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_manifest import record_hash                    # noqa: E402
import pointer_resolver_v2 as V                            # noqa: E402

R = "data/final_canonical"
V1 = R + "/LOCKED_6_OF_6_FAMILY_COMPLETION_V1.json"
OUT = R + "/LOCKED_6_OF_6_FAMILY_COMPLETION_V2.json"
AUDIT = "data/_family_v1/MUSIQUE_GRAPH2_KNN_WEIGHT_AUDIT.json"
DS = "musique"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def rj(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def coverage(ds, fam, n, source):
    g = V.CanonicalGraph(ds, fam, R, source=source)
    deg = np.bincount(np.concatenate([g.src, g.dst]).astype(np.int64), minlength=n)
    touched = int((deg > 0).sum())
    return {"source": g.source, "n_edges": int(g.src.size), "nodes_touched": touched,
            "node_coverage": round(touched / float(n), 6),
            "mean_degree_over_touched": round(float(deg.sum()) / max(1, touched), 3),
            "max_degree": int(deg.max())}


def main():
    t0 = time.time()
    if os.path.exists(OUT):
        sys.exit("%s already exists; a frozen record is never rewritten" % OUT)
    v1 = rj(V1)
    v1_crlf, v1_lf = record_hash(v1), record_hash(v1, crlf=False)
    if v1_crlf != v1.get("RECORD_SHA256"):
        sys.exit("V1 does not self-hash -- stop")

    g2m_path = "%s/%s/graph2/GRAPH_MANIFEST.json" % (R, DS)
    g2m = rj(g2m_path)
    dec = (g2m.get("supersedes") or {}).get("knn")
    if not dec or not g2m["families"].get("knn", {}).get("present"):
        sys.exit("musique/graph2 does not declare supersedes.knn -- run musique_knn_graph2.py first")
    fam_entry = g2m["families"]["knn"]
    npz = "%s/%s/%s" % (R, DS, fam_entry["file"])
    npz_sha = sha_file(npz)
    if npz_sha != fam_entry["npz_sha256"]:
        sys.exit("graph2/knn.npz on disk does not match its manifest digest -- stop")
    frozen_npz = "%s/%s/graph/knn.npz" % (R, DS)
    frozen_sha = sha_file(frozen_npz)
    if frozen_sha != dec["frozen_sha256"]:
        sys.exit("frozen knn.npz digest differs from the supersedes declaration -- stop")
    audit = rj(AUDIT)[DS]
    if audit["npz_sha256"] != npz_sha or audit["n_gt_1e-2"] != 0:
        sys.exit("the replacement audit does not cover this exact .npz or is not clean -- stop")
    if V.family_source(DS, "knn", R) != "graph2":
        sys.exit("pointer_resolver_v2 does not resolve musique/knn to graph2 -- precedence rule missing")

    n = int(g2m["n_nodes"])
    datasets = copy.deepcopy(v1["datasets"])
    m1 = datasets[DS]
    m2 = copy.deepcopy(m1)
    m2["family_source"] = dict(m1["family_source"], knn="graph2")
    m2["graph2_manifest_sha256"] = sha_file(g2m_path)
    m2["added"] = {
        "knn": {"file": "%s/%s" % (DS, fam_entry["file"]), "n_edges": fam_entry["n_edges"],
                "self_loops": fam_entry["self_loops"], "npz_sha256": npz_sha,
                "npz_bytes": os.path.getsize(npz),
                "unresolved_endpoint_edges": fam_entry["unresolved_endpoint_edges"],
                "exists_on_disk": True, "sha_matches_manifest": True,
                "supersedes_frozen": True},
    }
    cov_new = coverage(DS, "knn", n, "graph2")
    cov_frozen = coverage(DS, "knn", n, "frozen")
    m2["coverage"] = dict(m1["coverage"], knn=cov_new)
    m2["superseded_frozen_families"] = {
        "knn": {
            "frozen_file": "%s/graph/knn.npz" % DS,
            "frozen_sha256": frozen_sha,
            "still_pinned_by": "LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json",
            "frozen_bytes_untouched": True,
            "frozen_coverage": cov_frozen,
            "why_superseded": dec["reason"],
            "declared_in": "%s/graph2/GRAPH_MANIFEST.json#supersedes.knn" % DS,
            "parity_vs_frozen": dec.get("parity_vs_frozen"),
        }
    }
    m2["new_knn_self_consistency"] = {k: audit[k] for k in
                                      ("n_edges", "n_endpoint_rows_gathered", "sampled",
                                       "max_abs_err", "mean_abs_err", "p99_abs_err",
                                       "n_gt_1e-3", "n_gt_1e-2", "n_zero_weight", "VERDICT")}
    m2["new_knn_self_consistency"]["audit_file"] = AUDIT
    # V1's evidence travels unchanged
    m2["frozen_knn_self_consistency"] = m1["frozen_knn_self_consistency"]
    m2["knn_pipeline_parity_vs_frozen"] = m1["knn_pipeline_parity_vs_frozen"]
    datasets[DS] = m2

    changed = {DS: sorted(k for k in set(m1) | set(m2) if m1.get(k) != m2.get(k))}
    unchanged = sorted(d for d in datasets if d != DS and datasets[d] == v1["datasets"][d])

    rec = {
        "RECORD": "LOCKED_6_OF_6_FAMILY_COMPLETION_V2",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "STATUS": "FROZEN",
        "SCOPE": ("V1 plus one supersession: the stale frozen musique kNN is replaced BY "
                  "DECLARATION with a rebuilt family in musique/graph2/. Every other dataset "
                  "block is V1's, verbatim. No frozen byte moved."),
        "BUILDS_ON": {
            "record": "LOCKED_6_OF_6_FAMILY_COMPLETION_V1",
            "path": V1,
            "recorded_sha256": v1.get("RECORD_SHA256"),
            "recomputed_sha256": v1_crlf,
            "hash_matches": v1.get("RECORD_SHA256") == v1_crlf,
            "line_ending_independent_sha256": v1_lf,
            "note": "pinned by hash, not re-frozen. V1 is not edited: it recorded the stale "
                    "family truthfully and that record stands as the evidence for this one.",
        },
        "HASHING_CONVENTION": v1.get("HASHING_CONVENTION"),
        "CORE_INVARIANT": v1.get("CORE_INVARIANT"),
        "PRECEDENCE_RULE": {
            "rule": "frozen wins whenever graph/ and graph2/ both declare a family, UNLESS the "
                    "graph2 manifest carries an explicit supersedes.<family> block (frozen "
                    "file, frozen sha256, reason, frozen_bytes_untouched: true).",
            "where": "data/final_canonical/pointer_resolver_v2.py: supersedes(), "
                     "family_source(), CanonicalGraph(source=auto|frozen|graph2)",
            "honoured_by": ["src/dataset_canonical/substrate_map.py",
                            "src/dataset_canonical/ukb_manifest.py",
                            "src/dataset_canonical/verify_manifest.py",
                            "src/dataset_canonical/handoff_readiness.py",
                            "src/dataset_canonical/graph_degree_hub.py"],
            "supersessions_declared": {DS: ["knn"]},
            "readers_that_ignore_graph2": "pointer_resolver.CanonicalGraph reads graph/ only "
                                          "and still returns the frozen (stale) musique kNN; "
                                          "consumers are expected to read through "
                                          "pointer_resolver_v2",
        },
        "WHY_MUSIQUE_KNN_WAS_SUPERSEDED": {
            "finding": m1["frozen_knn_self_consistency"],
            "cause": ("the frozen search ran over the 1,473 musique Phase-C doc rows whose "
                      "vectors were damaged; DENSE_REPAIR now redirects those rows to repaired "
                      "vectors, so the frozen weights no longer describe the substrate the "
                      "resolver serves"),
            "controls": "squad and metaqa, same encoder and same pipeline, reproduce their "
                        "frozen kNN at the fp16 floor (max_abs_err ~1e-6) and EXACT edge sets",
            "rebuild": "the pipeline validated by KNN_PARITY_CHECK, run over the pointer-"
                       "resolved vectors on local CPU; replacement audited every edge",
            "what_was_not_done": "no heal of the Phase-C shards, no edit of graph/, no edit of "
                                 "LOCKED_5_OF_5 or V1. A freeze describes; it does not repair.",
        },
        "WHAT_CHANGED_AND_WHAT_DID_NOT": {
            "changed": "musique.family_source.knn frozen -> graph2; musique gains added.knn, "
                       "graph2_manifest_sha256, superseded_frozen_families, "
                       "new_knn_self_consistency; musique.coverage.knn re-measured",
            "musique_fields_changed": changed[DS],
            "not_changed": "the other five dataset blocks (byte-equal to V1): %s" % ", ".join(unchanged),
            "frozen_trees": "every <ds>/graph/ is byte-identical to its LOCKED_5_OF_5 / "
                            "LOCKED_6_OF_6 pins, musique included",
        },
        "REMOTE_EXECUTION_INTEGRITY": v1.get("REMOTE_EXECUTION_INTEGRITY"),
        "HOW_TO_READ_THE_ADDED_NER_FAMILIES": v1.get("HOW_TO_READ_THE_ADDED_NER_FAMILIES"),
        "datasets": datasets,
        "TOTALS": {"n_added_edges_v1": v1["TOTALS"]["n_added_edges"],
                   "n_added_edges_v2": v1["TOTALS"]["n_added_edges"] + int(fam_entry["n_edges"]),
                   "n_superseded_frozen_edges": int(dec["frozen_n_edges"])},
        "elapsed_s": round(time.time() - t0, 1),
    }
    rec["RECORD_SHA256"] = record_hash(rec)
    rec["RECORD_SHA256_LF"] = record_hash(rec, crlf=False)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    print("V1 %s  self-hash ok" % v1_crlf[:16])
    print("musique knn: graph2 %s edges (sha %s) supersedes frozen %s edges (sha %s)"
          % ("{:,}".format(fam_entry["n_edges"]), npz_sha[:16],
             "{:,}".format(dec["frozen_n_edges"]), frozen_sha[:16]))
    print("coverage new %.6f (mean deg %.3f)  frozen %.6f (mean deg %.3f)"
          % (cov_new["node_coverage"], cov_new["mean_degree_over_touched"],
             cov_frozen["node_coverage"], cov_frozen["mean_degree_over_touched"]))
    print("unchanged datasets: %s" % ", ".join(unchanged))
    print("wrote %s  RECORD_SHA256 %s" % (OUT, rec["RECORD_SHA256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
