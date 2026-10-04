# -*- coding: utf-8 -*-
"""Rebuild the musique kNN family into graph2/ and declare it as SUPERSEDING the frozen one.

WHY
---
data/_family_v1/FROZEN_KNN_WEIGHT_AUDIT.json showed that the frozen musique kNN
(graph/knn.npz, hash-pinned in LOCKED_5_OF_5) does not reproduce from its own vectors: 4,402 of
266,488 edge weights are off by more than 1e-2 (max 0.502, 4,413 zero weights), while squad and
metaqa -- same encoder, same pipeline -- reproduce at the fp16 floor (1e-6). The stale edges
sit on the 1,473 musique doc rows whose Phase-C vectors were damaged and are now served through
the DENSE_REPAIR redirect: the frozen kNN was searched over the damaged vectors, the resolver
now returns the repaired ones. KNN_PARITY_CHECK already ran the established pipeline over the
resolved vectors and found only tie churn beyond those rows (TIE_CHURN_ONLY).

WHAT THIS DOES AND DOES NOT DO
------------------------------
  - graph/knn.npz and graph/GRAPH_MANIFEST.json are NOT touched (pinned; a freeze describes,
    it does not repair). The rebuilt family lands in graph2/knn.npz next to them.
  - graph2/GRAPH_MANIFEST.json gains an explicit `supersedes.knn` block. Readers that
    honour it (pointer_resolver_v2, substrate_map, ukb_manifest, verify_manifest) prefer the
    graph2 family for that dataset; readers that do not still see the frozen one, so nothing
    silently changes under code that never asked.
  - the pipeline is line-for-line the one KNN_PARITY_CHECK validated against squad/metaqa
    (scratchpad/knn_parity_check.py): resolve through the pointer index, similarity over
    DISTINCT source rows, fp32 normalize, exhaustive top-4, drop self, keep 3, undirected,
    lift every canonical position to its source row's neighbours.
  - the result is audited the way the frozen one was (weight == cosine of its endpoints, every
    edge, not a sample) into data/_family_v1/MUSIQUE_GRAPH2_KNN_WEIGHT_AUDIT.json -- a new
    file; the earlier audit files are left as they are.

Run (about 11 minutes of CPU):
  PYTHONHASHSEED=0 python src/dataset_canonical/musique_knn_graph2.py
"""
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
from pointer_resolver import CanonicalEmbeddings          # noqa: E402
from verify_manifest import declared_index                 # noqa: E402

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0 (part of dataset identity)")

R = "data/final_canonical"
DS = "musique"
SEARCH_K = 4
KEEP = 3
TIE = 1e-3
BLOCK = 4096
AUDIT_OUT = "data/_family_v1/MUSIQUE_GRAPH2_KNN_WEIGHT_AUDIT.json"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def rj(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def build(ds):
    z = np.load("%s/%s/pointer_index/dense.npz" % (R, ds))
    src, row = z["src"].astype(np.int64), z["row"].astype(np.int64)
    key = (src << 40) | row
    uniq, pos2u = np.unique(key, return_inverse=True)
    nu = uniq.size
    n = src.size
    first = np.full(nu, -1, dtype=np.int64)
    o = np.arange(n - 1, -1, -1, dtype=np.int64)
    first[pos2u[o]] = o
    E = CanonicalEmbeddings(ds, "dense")
    X = E.gather(first).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    nbr = np.full((nu, SEARCH_K), -1, dtype=np.int64)
    sc = np.full((nu, SEARCH_K), -1e30, dtype=np.float32)
    for b in range(0, nu, BLOCK):
        e = min(b + BLOCK, nu)
        t = X[b:e] @ X.T
        idx = np.argpartition(-t, SEARCH_K - 1, axis=1)[:, :SEARCH_K]
        val = np.take_along_axis(t, idx, 1)
        o2 = np.argsort(-val, axis=1, kind="stable")
        nbr[b:e] = np.take_along_axis(idx, o2, 1)
        sc[b:e] = np.take_along_axis(val, o2, 1)
        if (b // BLOCK) % 5 == 0:
            print("  block %d/%d" % (b // BLOCK + 1, (nu + BLOCK - 1) // BLOCK), flush=True)
    return n, nu, pos2u, first, nbr, sc


def lift(n, pos2u, first, nbr, sc):
    pos = np.arange(n, dtype=np.int64)
    I, J, W = [], [], []
    for c in range(SEARCH_K):
        v = nbr[pos2u, c]
        w = sc[pos2u, c]
        keep = (v >= 0) & (v != pos2u)
        j = first[v[keep]]
        i = pos[keep]
        ok = i != j
        I.append(i[ok])
        J.append(j[ok])
        W.append(w[keep][ok])
    i, j, w = np.concatenate(I), np.concatenate(J), np.concatenate(W)
    a, b = np.minimum(i, j), np.maximum(i, j)
    key = (a << 32) | b
    o = np.lexsort((-w, key))
    key, a, b, w = key[o], a[o], b[o], w[o]
    k = np.ones(key.size, dtype=bool)
    k[1:] = key[1:] != key[:-1]
    return key[k], a[k], b[k], w[k]


def weight_audit(ds, npz_path, out_path):
    """Every edge weight must equal the cosine of its endpoints under the resolver, today."""
    z = np.load(npz_path)
    src, dst = z["src"].astype(np.int64), z["dst"].astype(np.int64)
    w = z["weight"].astype(np.float64)
    need = np.unique(np.concatenate([src, dst]))
    X = CanonicalEmbeddings(ds, "dense").gather(need).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ix = {int(v): i for i, v in enumerate(need)}
    si = np.fromiter((ix[int(v)] for v in src), np.int64, src.size)
    di = np.fromiter((ix[int(v)] for v in dst), np.int64, dst.size)
    d = np.abs(np.einsum("ij,ij->i", X[si], X[di]).astype(np.float64) - w)
    r = {"family_file": npz_path.replace("\\", "/"), "npz_sha256": sha_file(npz_path),
         "n_edges": int(src.size), "n_endpoint_rows_gathered": int(need.size),
         "sampled": False,
         "max_abs_err": float(d.max()), "mean_abs_err": float(d.mean()),
         "p99_abs_err": float(np.percentile(d, 99)),
         "n_gt_1e-3": int((d > 1e-3).sum()), "n_gt_1e-2": int((d > 1e-2).sum()),
         "frac_gt_1e-2": float((d > 1e-2).mean()),
         "weight_min": float(w.min()), "weight_max": float(w.max()),
         "n_zero_weight": int((w == 0).sum())}
    r["VERDICT"] = ("CONSISTENT_WITH_SUBSTRATE" if r["n_gt_1e-2"] == 0
                    else "STALE: %d edges cannot be reproduced from the resolved vectors"
                         % r["n_gt_1e-2"])
    r["method"] = ("weight compared with the fp32 cosine of the two pointer-resolved canonical "
                   "vectors, every edge; same test as FROZEN_KNN_WEIGHT_AUDIT")
    r["created_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with io.open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({ds: r}, indent=1))
    return r


def main():
    t0 = time.time()
    frozen_man = rj("%s/%s/graph/GRAPH_MANIFEST.json" % (R, DS))
    n_nodes = int(frozen_man["n_nodes"])
    frozen_knn_path = "%s/%s/graph/knn.npz" % (R, DS)
    frozen_sha = sha_file(frozen_knn_path)
    decl, _five, _six, _fam = declared_index()
    pinned = decl.get(frozen_knn_path)
    if pinned is None or pinned[1] != frozen_sha:
        sys.exit("frozen musique graph/knn.npz does not match its LOCKED_5_OF_5 pin -- stop")
    print("frozen graph/knn.npz sha256 %s matches LOCKED_5_OF_5 pin" % frozen_sha[:16])

    print("building kNN over resolved vectors ...", flush=True)
    n, nu, pos2u, first, nbr, sc = build(DS)
    if n != n_nodes:
        sys.exit("pointer index has %d positions, frozen manifest says %d nodes" % (n, n_nodes))
    key, a, b, w = lift(n, pos2u, first, nbr, sc)
    src = a.astype(np.int32)
    dst = b.astype(np.int32)
    wt = w.astype(np.float32)
    if int(src.min()) < 0 or int(max(src.max(), dst.max())) >= n_nodes:
        sys.exit("endpoint outside [0,%d)" % n_nodes)
    t_build = time.time() - t0

    # parity vs frozen, recomputed here so the manifest carries its own evidence
    zf = np.load(frozen_knn_path)
    fa = np.minimum(zf["src"], zf["dst"]).astype(np.int64)
    fb = np.maximum(zf["src"], zf["dst"]).astype(np.int64)
    oldkey = np.unique((fa << 32) | fb)
    inter = np.intersect1d(key, oldkey, assume_unique=True)
    only_new = np.setdiff1d(key, oldkey, assume_unique=True)
    only_old = np.setdiff1d(oldkey, key, assume_unique=True)
    tie_rows = int((np.abs(sc[:, 2] - sc[:, 3]) <= TIE).sum())
    parity = {"frozen_edges": int(oldkey.size), "new_edges": int(key.size),
              "shared": int(inter.size), "only_new": int(only_new.size),
              "only_frozen": int(only_old.size),
              "rows_with_k3_k4_tie_within_%g" % TIE: tie_rows,
              "VERDICT": ("EXACT_MATCH" if not only_new.size and not only_old.size
                          else "TIE_CHURN_ONLY" if max(only_new.size, only_old.size) <= tie_rows
                          else "DIFFERS")}

    d = "%s/%s/graph2" % (R, DS)
    os.makedirs(d, exist_ok=True)
    p = "%s/knn.npz" % d
    np.savez_compressed(p, src=src, dst=dst, weight=wt)
    npz_sha = sha_file(p)
    print("wrote %s  edges %s  %.1f MB  (%.0fs)" % (p, "{:,}".format(int(src.size)),
                                                    os.path.getsize(p) / 1e6, t_build), flush=True)

    print("auditing every edge weight against the resolved vectors ...", flush=True)
    audit = weight_audit(DS, p, AUDIT_OUT)
    print("  %s  max %.3e  p99 %.3e  >1e-2 %d" % (audit["VERDICT"], audit["max_abs_err"],
                                                  audit["p99_abs_err"], audit["n_gt_1e-2"]))
    if audit["n_gt_1e-2"] != 0:
        sys.exit("the rebuilt family does not reproduce from its own vectors -- not installing "
                 "a supersession over a stale artifact")

    deg = np.bincount(np.concatenate([src, dst]).astype(np.int64), minlength=n_nodes)
    touched = int((deg > 0).sum())
    frozen_audit = rj("data/_family_v1/FROZEN_KNN_WEIGHT_AUDIT.json").get(DS) or {}

    mp = "%s/GRAPH_MANIFEST.json" % d
    out = rj(mp) if os.path.exists(mp) else {
        "RECORD": "CANONICAL_V1_GRAPH_V2",
        "dataset": DS,
        "n_nodes": n_nodes,
        "endpoint_space": frozen_man["endpoint_space"],
        "frozen_manifest": "graph/GRAPH_MANIFEST.json",
        "frozen_families": sorted(k for k, v in frozen_man["families"].items()
                                  if v.get("present")),
        "families": {},
    }
    out["supersedes_note"] = (
        "graph/ is untouched and byte-identical to its LOCKED_5_OF_5 pins. This file adds no "
        "family the frozen manifest declares absent; it REPLACES one it declares present. The "
        "`supersedes` block below is the only thing that makes a graph2 family take precedence "
        "over a frozen one -- a graph2 manifest without it never shadows graph/.")
    out["supersedes"] = {
        "knn": {
            "frozen_file": "graph/knn.npz",
            "frozen_sha256": frozen_sha,
            "frozen_n_edges": int(frozen_man["families"]["knn"]["n_edges"]),
            "frozen_bytes_untouched": True,
            "pinned_by": "LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json",
            "reason": ("FROZEN_KNN_WEIGHT_AUDIT: %d of %d frozen edge weights do not reproduce "
                       "from the pointer-resolved vectors (max_abs_err %.3f, %d zero weights); "
                       "the frozen search ran over the 1,473 damaged musique Phase-C doc rows "
                       "that DENSE_REPAIR now redirects around. squad and metaqa, same pipeline, "
                       "reproduce at the fp16 floor."
                       % (frozen_audit.get("n_gt_1e-2", -1), frozen_audit.get("n_edges", -1),
                          frozen_audit.get("max_abs_err", float("nan")),
                          frozen_audit.get("n_zero_weight", -1))),
            "frozen_self_consistency": frozen_audit,
            "replacement_self_consistency": {k: audit[k] for k in
                                             ("max_abs_err", "p99_abs_err", "n_gt_1e-2",
                                              "n_zero_weight", "VERDICT")},
            "replacement_audit_file": AUDIT_OUT,
            "parity_vs_frozen": parity,
            "what_a_freeze_does": "describes, it does not repair: the stale family is routed "
                                  "around by declaration, not rewritten",
        }
    }
    out["families"]["knn"] = {
        "present": True, "file": "graph2/knn.npz", "n_edges": int(src.size),
        "self_loops": int((src == dst).sum()), "directed": False,
        "attributes": "weight(float32)",
        "npz_sha256": npz_sha, "npz_bytes": os.path.getsize(p),
        "unresolved_endpoint_edges": 0,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "coverage": {"nodes_touched": touched,
                     "node_coverage": round(touched / float(n_nodes), 6),
                     "mean_degree_over_touched": round(float(deg.sum()) / max(1, touched), 3),
                     "max_degree": int(deg.max())},
        "source_manifest": {
            "dataset": DS,
            "edge_family": "knn",
            "edge_subtype": "semantic_knn_k%d" % KEEP,
            "method": "EXHAUSTIVE_FP32_COMPUTE over RESOLVED canonical vectors (local CPU)",
            "flags": {"FP16_STORAGE": "YES", "FP16_REDUCED_COMPUTE": "NO",
                      "FP32_COMPUTE": "YES", "TF32": "OFF", "ANN": "NO", "EXHAUSTIVE": "YES"},
            "exact": True, "ann": False,
            "search_k": SEARCH_K, "n_neighbors": KEEP, "tie_tolerance": TIE, "block_rows": BLOCK,
            "n_canonical_positions": int(n), "n_distinct_source_rows": int(nu),
            "n_duplicate_positions": int(n - nu),
            "n_edges": int(src.size), "directed": False,
            "endpoint_space": "canonical positions (nodes.jsonl line numbers)",
            "similarity_space": ("distinct source rows; each position inherits the neighbours "
                                 "of its source row, mapped to the lowest canonical position "
                                 "of each neighbour"),
            "vector_source": "pointer-resolved (PHASE_C + REV2_PATCH + DENSE_REPAIR)",
            "pipeline": "src/dataset_canonical/musique_knn_graph2.py == scratchpad/"
                        "knn_parity_check.py (validated EXACT on squad/metaqa frozen kNN)",
            "PYTHONHASHSEED": "0",
            "build_seconds": round(t_build, 1),
        },
    }
    with io.open(mp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=1))
    print("wrote %s  (supersedes.knn declared)  parity %s only_new %d only_frozen %d ties %d"
          % (mp, parity["VERDICT"], parity["only_new"], parity["only_frozen"], tie_rows))
    print("coverage %.6f  mean_deg %.3f  max_deg %d   total %.0fs"
          % (touched / float(n_nodes), float(deg.sum()) / max(1, touched), int(deg.max()),
             time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
