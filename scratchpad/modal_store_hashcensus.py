# -*- coding: utf-8 -*-
"""Hash every shard of a dense store on a Modal volume and report name/size/sha256.

`modal volume ls` under-reports: extra_ip9HxU's listing showed 54 hotpotqa shards and no
shard_00008, yet `volume put` of shard_00008 came back "already exists". A listing that can
miss a file can also hide a truncated one, and a truncated shard is a silently wrong kNN --
it would sail past the 3-shard spot check the compute stage does. So enumerate and hash the
whole store from inside a container and diff that against local, rather than trusting the CLI.
"""
import hashlib
import json
import os

import modal

app = modal.App("crag-store-census")
image = modal.Image.debian_slim(python_version="3.11")
VOL = modal.Volume.from_name("crag-data-volume")

TREE = {"2wiki": "2wiki_universe", "hotpotqa": "hotpotqa"}


@app.function(image=image, volumes={"/root/vol": VOL}, cpu=4.0, memory=8192, timeout=7200)
def census(ds: str):
    d = "/root/vol/data/canonical/%s/encodings/dense/docs" % TREE[ds]
    if not os.path.isdir(d):
        return {"dir": d, "exists": False, "shards": {}}
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".npy"):
            continue
        p = os.path.join(d, fn)
        h = hashlib.sha256()
        n = 0
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b:
                    break
                n += len(b)
                h.update(b)
        out[fn] = [n, h.hexdigest()]
    return {"dir": d, "exists": True, "shards": out}


@app.local_entrypoint()
def main(ds: str = "hotpotqa", out: str = ""):
    r = census.remote(ds)
    txt = json.dumps(r, indent=1, sort_keys=True)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(txt)
        print("wrote %s  shards=%d" % (out, len(r["shards"])))
    else:
        print(txt)
