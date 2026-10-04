# -*- coding: utf-8 -*-
"""Finish the kNN upload bundle: base-store locator, patch rows, and the local half of the
resolution cross-check.

The cross-check is the point of this file. The Modal side reimplements what
pointer_resolver.CanonicalEmbeddings does -- read the pointer arrays, pull each row from the
store it names -- and a silent mistake there would produce a perfectly well-formed kNN over
the wrong vectors, which no shape or count would catch. So the same 4096 distinct source rows
are resolved HERE through the real resolver, and the sha256 of those bytes has to come back
identical from the GPU container. The row list travels with the bundle rather than being drawn
from a seed on both sides, so the check cannot pass or fail on a numpy version difference.
"""
import hashlib
import io
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("data/final_canonical"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pointer_resolver import CanonicalEmbeddings          # noqa: E402

OUT = "data/_family_v1/_knn_bundle"
NSAMP = 4096

STORE0 = {
    "2wiki": {"dir": "/root/vol/data/canonical/2wiki_universe/encodings/dense/docs",
              "pattern": "shard_%05d.npy", "shard_size": 40000,
              "local": "data/canonical/2wiki_universe/encodings/dense/docs"},
    "hotpotqa": {"dir": "/root/vol/data/canonical/hotpotqa/encodings/dense/docs",
                 "pattern": "shard_%05d.npy", "shard_size": 40000,
                 "local": "data/canonical/hotpotqa/encodings/dense/docs"},
    # the webqsp encoder parts were renamed into place as the canonical shards, so the parts
    # still on crag-webqsp-enc ARE that store -- no 5.2 GB upload, just a different filename
    "webqsp": {"dir": "/root/enc", "pattern": "dense__p%04d/dense.npy", "shard_size": 12000,
               "local": "data/canonical/webqsp_rog_v1/encodings/dense/docs"},
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


for ds in sys.argv[1:] or ["2wiki", "hotpotqa", "webqsp"]:
    B = "%s/%s" % (OUT, ds)
    meta = json.load(io.open(B + "/meta.json", encoding="utf-8"))
    m = np.load(B + "/map.npz")
    u_src, u_row, first = m["u_src"].astype(np.int64), m["u_row"].astype(np.int64), m["first_pos"]
    nu = u_src.size
    s0 = dict(STORE0[ds])
    loc = s0.pop("local")
    ss = s0["shard_size"]
    need = int(u_row[u_src == 0].max()) // ss
    pick_sh = sorted(set([0, need // 2, need]))
    s0["verify_shards"] = pick_sh
    s0["local_shard_sha256"] = {str(k): sha_file("%s/shard_%05d.npy" % (loc, k)) for k in pick_sh}
    s0["local_shard_bytes"] = {str(k): os.path.getsize("%s/shard_%05d.npy" % (loc, k))
                               for k in pick_sh}
    s0["max_base_shard_needed"] = need
    json.dump(s0, io.open(B + "/store0.json", "w", encoding="utf-8"), indent=1)

    for st in meta["stores"]:
        if st["i"] == 0 or not st.get("flat"):
            continue
        dst = "%s/store_%d.npy" % (B, st["i"])
        if not os.path.exists(dst):
            shutil.copyfile(st["path"], dst)
        assert sha_file(dst) == st["sha256"], "patch copy mismatch: %s" % dst

    rng = np.random.default_rng(20260909)
    pick = np.sort(rng.choice(nu, size=min(NSAMP, nu), replace=False)).astype(np.int64)
    np.save(B + "/fp_rows.npy", pick)
    E = CanonicalEmbeddings(ds, "dense")
    X = E.gather(first[pick].astype(np.int64)).astype(np.float16)
    fp = hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()
    zero = int((np.abs(X).sum(axis=1) == 0).sum())
    json.dump({"dataset": ds, "n_distinct": int(nu), "n_sample": int(pick.size),
               "sample_sha256": fp, "zero_rows_in_sample": zero,
               "via": "data/final_canonical/pointer_resolver.CanonicalEmbeddings"},
              io.open(B + "/LOCAL_FINGERPRINT.json", "w", encoding="utf-8"), indent=1)
    print("%-9s base shards needed %4d  verify %s  sample %d  zero %d  local sha %s"
          % (ds, need, pick_sh, pick.size, zero, fp[:16]))
print("bundle staged: " + OUT)
