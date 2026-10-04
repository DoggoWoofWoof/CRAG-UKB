"""Stage a whole splade doc store plus its bundle into a fresh Modal workspace.

The first workspace was disabled mid-run, and volumes do not cross workspaces, so a fresh one
starts empty -- there is no block-level dedup to ride on and these are real bytes.  Only splade
is staged: both dense caches were already produced and downloaded.
"""
import io
import json
import os
import sys
import time

import modal

plan = json.load(io.open("scratchpad/_upload_plan.json", encoding="utf-8"))
data = modal.Volume.from_name("crag-data-volume", create_if_missing=True)
bund = modal.Volume.from_name("crag-cache-bundle", create_if_missing=True)
modal.Volume.from_name("crag-retrieval-out", create_if_missing=True)

for key in sys.argv[1:]:
    p = plan[key]
    st = json.load(io.open("scratchpad/_cache_bundle/%s/store0.json" % key, encoding="utf-8"))
    n = int(st["max_base_shard_needed"]) + 1
    files = [("%s/shard_%05d.%s" % (p["local_dir"], k, p["ext"]),
              "/%s/shard_%05d.%s" % (p["remote_dir"], k, p["ext"])) for k in range(n)]
    tb = sum(os.path.getsize(a) for a, _ in files)
    t0 = time.time()
    print("%s: %d shards, %.2f GB -> crag-data-volume" % (key, n, tb / 1e9), flush=True)
    with data.batch_upload(force=True) as b:
        for a, r in files:
            b.put_file(a, r)
    print("%s: store done %.0fs (%.1f MB/s)"
          % (key, time.time() - t0, tb / 1e6 / max(time.time() - t0, 1)), flush=True)

    t0 = time.time()
    bd = "scratchpad/_cache_bundle/%s" % key
    bb = sum(os.path.getsize(os.path.join(bd, f)) for f in os.listdir(bd))
    print("%s: bundle %.2f GB -> crag-cache-bundle" % (key, bb / 1e9), flush=True)
    with bund.batch_upload(force=True) as b:
        for f in sorted(os.listdir(bd)):
            b.put_file(os.path.join(bd, f), "/%s/%s" % (key, f))
    print("%s: bundle done %.0fs" % (key, time.time() - t0), flush=True)
print("ALL STAGED")
