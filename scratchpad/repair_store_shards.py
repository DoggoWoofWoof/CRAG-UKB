# -*- coding: utf-8 -*-
"""Force-replace remote shards whose content does not match the local canonical bytes.

The mismatches are not a stale textualization -- a differing shard reproduces the local one to
cosine 0.9999979, max abs 4.2e-3, i.e. the same documents encoded on a different accelerator,
fp16 rounding apart. That is still the wrong artifact: the frozen substrate IS the local bytes,
the squad parity run showed k3/k4 gaps down to 6e-8, and a neighbour that flips on rounding
noise would make the kNN unreproducible from the frozen corpus.
"""
import io
import json
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TREE = {"2wiki": "2wiki_universe", "hotpotqa": "hotpotqa"}

for spec in sys.argv[1:]:
    profile, ds, rp = spec.split("=", 2)
    L = json.load(io.open("data/_family_v1/_remote_census/LOCAL.%s.hash.json" % ds,
                          encoding="utf-8"))["shards"]
    R = json.load(io.open(rp, encoding="utf-8"))["shards"]
    bad = sorted(k for k in L if k not in R or L[k] != R[k])
    d = "data/canonical/%s/encodings/dense/docs" % TREE[ds]
    tot = sum(os.path.getsize("%s/%s" % (d, k)) for k in bad)
    print("[%s/%s] replacing %d shards, %.2f GB" % (profile, ds, len(bad), tot / 1e9),
          flush=True)
    env = dict(os.environ, MODAL_PROFILE=profile, PYTHONIOENCODING="utf-8",
               MSYS_NO_PATHCONV="1")
    t0 = time.time()
    for n, k in enumerate(bad, 1):
        rel = "%s/%s" % (d, k)
        r = subprocess.run(["modal", "volume", "put", "--force", "crag-data-volume", rel, rel],
                           env=env, capture_output=True, encoding="utf-8", errors="replace")
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            print("  FAIL %s: %s" % (k, re.sub(r"\s+", " ", out).strip()[:250]), flush=True)
            sys.exit(1)
        print("  %d/%d %s %.0fs" % (n, len(bad), k, time.time() - t0), flush=True)
    print("[%s/%s] REPLACED %d in %.0fs" % (profile, ds, len(bad), time.time() - t0),
          flush=True)
