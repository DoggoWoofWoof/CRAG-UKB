# -*- coding: utf-8 -*-
"""Install the newly built edge families NEXT TO the frozen ones, never over them.

data/final_canonical/<ds>/graph/GRAPH_MANIFEST.json, graph/knn.npz and graph/ner.npz are all
hash-pinned in LOCKED_5_OF_5_BENCHMARK_SUBSTRATES.json. Writing the new families into that
directory would break six recorded digests and make a correct package look tampered with, so
the new families land in <ds>/graph2/ with their own manifest, and the frozen tree is left
byte-identical. graph2/GRAPH_MANIFEST.json carries the FULL family table -- the frozen
families by reference, the new ones by content -- so one file answers "what does this dataset
have" without the caller having to know which freeze a family belongs to.

Endpoint resolution is exact and checked, not assumed: every edge endpoint must land in
[0, n_nodes) and every source id must map to exactly one canonical position, or the family is
refused.

    PYTHONHASHSEED=0 python scratchpad/install_family_v2.py metaqa:ner webqsp:ner 2wiki:knn
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")

R = "data/final_canonical"
STAGE = "data/_family_v1"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def node_count(ds):
    return sum(1 for _ in io.open("%s/%s/nodes.jsonl" % (R, ds), encoding="utf-8"))


def position_of(ds):
    """id -> canonical position, for the id space the new family emits.

    metaqa: the NER source shards carry metaqa_ent_<kb_entity_index>; the node table carries
    kb_entity_index directly, so the map is read off the nodes rather than assumed to be the
    identity.  webqsp: the source shards were generated from nodes.jsonl itself, so the id IS
    node_id.
    """
    out = {}
    with io.open("%s/%s/nodes.jsonl" % (R, ds), encoding="utf-8") as f:
        for i, ln in enumerate(f):
            o = json.loads(ln)
            out[o["node_id"]] = i
            if ds == "metaqa":
                out["metaqa_ent_%d" % int(o["kb_entity_index"])] = i
    return out


def load_ner_tsv(ds, n):
    p = "%s/%s/graph_ner.tsv" % (STAGE, ds)
    pos = position_of(ds)
    src, dst, w = [], [], []
    unresolved = set()
    with io.open(p, encoding="utf-8") as f:
        for ln in f:
            a, b, v = ln.rstrip(chr(10)).split(chr(9))
            ia, ib = pos.get(a), pos.get(b)
            if ia is None or ib is None:
                unresolved.add(a if ia is None else b)
                continue
            src.append(ia)
            dst.append(ib)
            w.append(float(v))
    return (np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32),
            np.asarray(w, dtype=np.float32), sorted(unresolved)[:8],
            json.load(io.open("%s/%s/ner_manifest.json" % (STAGE, ds), encoding="utf-8")))


def load_knn_npz(ds, n):
    d = "%s/_knn_out/%s" % (STAGE, ds)
    z = np.load(d + "/knn_canonical_edges.npz")
    man = json.load(io.open(d + "/knn_manifest.json", encoding="utf-8"))
    return (z["src"].astype(np.int32), z["dst"].astype(np.int32),
            z["weight"].astype(np.float32), [], man)


def install(ds, fam):
    t0 = time.time()
    n = node_count(ds)
    src, dst, w, unresolved, man = (load_ner_tsv if fam == "ner" else load_knn_npz)(ds, n)
    if unresolved:
        sys.exit("[%s/%s] unresolved endpoint ids, e.g. %s" % (ds, fam, unresolved))
    if src.size and (int(src.min()) < 0 or int(max(src.max(), dst.max())) >= n):
        sys.exit("[%s/%s] endpoint outside [0,%d)" % (ds, fam, n))
    self_loops = int((src == dst).sum())
    d = "%s/%s/graph2" % (R, ds)
    os.makedirs(d, exist_ok=True)
    p = "%s/%s.npz" % (d, fam)
    np.savez_compressed(p, src=src, dst=dst, weight=w)
    frozen = json.load(io.open("%s/%s/graph/GRAPH_MANIFEST.json" % (R, ds), encoding="utf-8"))
    if int(frozen["n_nodes"]) != n:
        sys.exit("[%s] node count disagrees with the frozen manifest" % ds)
    mp = "%s/GRAPH_MANIFEST.json" % d
    out = (json.load(io.open(mp, encoding="utf-8")) if os.path.exists(mp) else {
        "RECORD": "CANONICAL_V1_GRAPH_V2",
        "dataset": ds,
        "n_nodes": n,
        "endpoint_space": frozen["endpoint_space"],
        "supersedes_nothing": ("graph/ is untouched and still authoritative for the families "
                               "it declares; this file adds families it declares absent"),
        "frozen_manifest": "graph/GRAPH_MANIFEST.json",
        "frozen_families": sorted(k for k, v in frozen["families"].items()
                                  if v.get("present")),
        "families": {},
    })
    out["families"][fam] = {
        "present": True, "file": "graph2/%s.npz" % fam, "n_edges": int(src.size),
        "self_loops": self_loops, "directed": False,
        "attributes": "weight(float32)",
        "npz_sha256": sha_file(p), "npz_bytes": os.path.getsize(p),
        "unresolved_endpoint_edges": 0,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_manifest": man,
    }
    json.dump(out, io.open(mp, "w", encoding="utf-8"), indent=1)
    print("%-9s %-4s edges %11s  self_loops %d  %.1f MB  %.0fs"
          % (ds, fam, "{:,}".format(int(src.size)), self_loops,
             os.path.getsize(p) / 1e6, time.time() - t0))


for spec in sys.argv[1:]:
    ds, fam = spec.split(":")
    install(ds, fam)
