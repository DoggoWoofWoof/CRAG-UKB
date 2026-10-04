"""SHA-256 + size of every WebQSP input the FBX_SCALE 3a bundle reads (run on the laptop and on the host; the two JSONs are compared).

Usage: python scratchpad/_wq_sha.py <out.json>     (cwd = repo root; reads data/final_canonical/webqsp/{embeddings,graph,retrieval_cache,v1}/**,
       queries/query_ids.json, queries/train.jsonl and data/l1_canonical/webqsp/{query_index.npz,keys.npz}; writes only <out.json>)
"""
import hashlib
import json
import os
import sys
import time

R = "data/final_canonical/webqsp"
EXTRA = ["data/l1_canonical/webqsp/query_index.npz", "data/l1_canonical/webqsp/keys.npz"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    files = []
    for sub in ("embeddings", "graph", "retrieval_cache", "v1"):
        for dp, dn, fn in os.walk(os.path.join(R, sub)):
            for f in fn:
                if f.endswith(".rxpart") or f.endswith(".rxpart.json") or f.endswith(".part") or f.endswith(".part.done"):
                    continue
                files.append(os.path.join(dp, f).replace("\\", "/"))
    files += [R + "/queries/query_ids.json", R + "/queries/train.jsonl"] + EXTRA
    out, t0 = {}, time.time()
    for p in sorted(files):
        out[p] = {"size": os.path.getsize(p), "sha256": sha(p)}
    with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=0, sort_keys=True)
    print("hashed %d files, %.2f GB in %.0f s" % (len(out), sum(v["size"] for v in out.values()) / 1e9, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
