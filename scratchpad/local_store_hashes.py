# -*- coding: utf-8 -*-
"""Local sha256 of every dense shard, to diff against the same census taken on a Modal volume."""
import concurrent.futures as cf
import hashlib
import json
import os
import sys

TREE = {"2wiki": "2wiki_universe", "hotpotqa": "hotpotqa"}


def one(p):
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            n += len(b)
            h.update(b)
    return os.path.basename(p), [n, h.hexdigest()]


for ds in sys.argv[1:]:
    d = "data/canonical/%s/encodings/dense/docs" % TREE[ds]
    fs = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
    with cf.ThreadPoolExecutor(6) as ex:
        out = dict(ex.map(one, [os.path.join(d, f) for f in fs]))
    p = "data/_family_v1/_remote_census/LOCAL.%s.hash.json" % ds
    json.dump({"dir": d, "exists": True, "shards": out}, open(p, "w"), indent=1, sort_keys=True)
    print("%s %d shards -> %s" % (ds, len(out), p), flush=True)
