# -*- coding: utf-8 -*-
"""Build the upload bundle for the GPU kNN: the distinct-source-vector map plus base-store
fingerprints.

The canonical vectors are NOT the Phase-C shards on disk -- 2wiki and hotpotqa resolve through
a REV2 patch and a dense-repair patch, so a kNN computed straight off the base tree (which is
what the Aug-28 Modal checkpoints did) is a kNN over ~151k wrong 2wiki vectors. So the bundle
carries the pointer arrays and the patch rows, and Modal reconstructs exactly what
pointer_resolver.CanonicalEmbeddings would hand back.

Similarity runs over DISTINCT source rows, not canonical positions: 30.9% of webqsp's nodes
share an encoder row with another node (identical text), and in position space those ties would
eat all three neighbour slots at cosine 1.0. Each canonical position then inherits its source
row's neighbours, mapped to each neighbour's lowest canonical position -- so every node gets
edges, including the deduplicated ones.
"""
import hashlib
import io
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

R = "data/final_canonical"
OUT = "data/_family_v1/_knn_bundle"
DS = sys.argv[1:] or ["2wiki", "hotpotqa", "webqsp"]
os.makedirs(OUT, exist_ok=True)
pi = json.load(io.open(R + "/POINTER_INDEX.json", encoding="utf-8"))


def sha_file(p, cap=None):
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
            n += len(c)
            if cap and n >= cap:
                break
    return h.hexdigest(), n


report = {}
for ds in DS:
    spec = pi["datasets"][ds]["dense"]
    z = np.load("%s/%s/pointer_index/dense.npz" % (R, ds))
    src, row = z["src"].astype(np.int64), z["row"].astype(np.int64)
    n = src.size
    key = (src << 40) | row                      # row < 2^40 everywhere; exact, no collisions
    uniq, pos2u = np.unique(key, return_inverse=True)
    nu = uniq.size
    u_src = (uniq >> 40).astype(np.int16)
    u_row = (uniq & ((1 << 40) - 1)).astype(np.int64)
    # lowest canonical position holding each distinct source row
    first = np.full(nu, -1, dtype=np.int64)
    order = np.arange(n - 1, -1, -1, dtype=np.int64)
    first[pos2u[order]] = order                  # reverse scan leaves the smallest position
    assert first.min() >= 0
    d = "%s/%s" % (OUT, ds)
    os.makedirs(d, exist_ok=True)
    np.savez(d + "/map.npz", u_src=u_src, u_row=u_row, pos2u=pos2u.astype(np.int32),
             first_pos=first.astype(np.int32))
    stores = []
    for i, s in enumerate(spec["stores"]):
        e = {"i": i, "path": s["path"], "flat": bool(s.get("flat")),
             "shard_size": s.get("shard_size"), "store": s.get("store", "PHASE_C"),
             "n_rows_used": int((u_src == i).sum())}
        if s.get("flat"):
            e["sha256"], e["bytes"] = sha_file(s["path"])
            e["upload"] = True
        else:
            e["upload"] = not os.path.isdir("data/canonical/%s" % s["path"].split("/")[2]) or None
        stores.append(e)
    meta = {"dataset": ds, "n_positions": int(n), "n_distinct_source_rows": int(nu),
            "n_duplicate_positions": int(n - nu), "dim": 1536, "stores": stores,
            "map_sha256": sha_file(d + "/map.npz")[0]}
    json.dump(meta, io.open(d + "/meta.json", "w", encoding="utf-8"), indent=1)
    report[ds] = meta
    print("%-9s positions %9s  distinct %9s  dup %8s (%.1f%%)  stores %d"
          % (ds, "{:,}".format(n), "{:,}".format(nu), "{:,}".format(n - nu),
             100.0 * (n - nu) / n, len(stores)))
    for e in stores:
        print("            store %d %-12s rows_used %9s  %s"
              % (e["i"], e["store"], "{:,}".format(e["n_rows_used"]), e["path"]))
json.dump(report, io.open(OUT + "/BUNDLE.json", "w", encoding="utf-8"), indent=1)
print("wrote " + OUT)
