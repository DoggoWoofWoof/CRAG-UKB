# -*- coding: utf-8 -*-
"""Diff a remote store census against the local one, shard by shard.

Present-but-wrong is the failure this exists to catch: `modal volume put` reports "already
exists" without comparing content, and `modal volume ls` was observed to omit files that are
in fact there, so neither tells you whether the remote copy is the corpus you encoded.
"""
import io
import json
import sys

for ds, rp in (a.split("=") for a in sys.argv[1:]):
    L = json.load(io.open("data/_family_v1/_remote_census/LOCAL.%s.hash.json" % ds,
                          encoding="utf-8"))["shards"]
    R = json.load(io.open(rp, encoding="utf-8"))["shards"]
    missing = sorted(set(L) - set(R))
    extra = sorted(set(R) - set(L))
    bad_size = sorted(k for k in set(L) & set(R) if L[k][0] != R[k][0])
    bad_hash = sorted(k for k in set(L) & set(R) if L[k][0] == R[k][0] and L[k][1] != R[k][1])
    ok = len(set(L) & set(R)) - len(bad_size) - len(bad_hash)
    verdict = "MATCH" if not (missing or bad_size or bad_hash) else "MISMATCH"
    print("[%s] %s local=%d remote=%d identical=%d missing=%d size_wrong=%d hash_wrong=%d "
          "extra=%d" % (ds, verdict, len(L), len(R), ok, len(missing), len(bad_size),
                        len(bad_hash), len(extra)))
    for lbl, v in (("missing", missing), ("size_wrong", bad_size), ("hash_wrong", bad_hash),
                   ("extra", extra)):
        if v:
            print("    %s: %s%s" % (lbl, v[:12], " ..." if len(v) > 12 else ""))
