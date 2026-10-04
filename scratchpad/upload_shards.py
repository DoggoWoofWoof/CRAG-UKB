"""Repair the remote doc stores: upload exactly the shards the remote census found missing or
byte-wrong, with force=True.

WHY force AND WHY A NAMED LIST

  `modal volume put` skips a destination that already exists BY NAME, so a shard that is the
  right size and the wrong bytes -- 9 of hotpotqa's dense shards and 17 of 2wiki's -- would be
  skipped by a plain re-put and stay wrong.  force=True overwrites.  And the list comes from
  the census's sha256 comparison rather than from `volume ls`, which under-lists: the same
  stores read 46/131 and 41/150 by `volume ls` and 37/131 and 24/150 by hashing.

  batch_upload holds one connection for the whole store instead of paying CLI startup 100
  times.

Run:
  python scratchpad/upload_shards.py hotpotqa_splade 2wiki_splade
"""
import io
import json
import os
import sys
import time

import modal

plan = json.load(io.open("scratchpad/_upload_plan.json", encoding="utf-8"))
vol = modal.Volume.from_name("crag-data-volume")

for key in sys.argv[1:]:
    p = plan[key]
    t0 = time.time()
    files = [("%s/shard_%05d.%s" % (p["local_dir"], k, p["ext"]),
              "/%s/shard_%05d.%s" % (p["remote_dir"], k, p["ext"])) for k in p["shards"]]
    for lp, _ in files:
        if not os.path.isfile(lp):
            raise SystemExit("missing LOCAL shard %s -- refusing" % lp)
    print("%s: %d shards, %.2f GB" % (key, len(files), p["bytes"] / 1e9), flush=True)
    with vol.batch_upload(force=True) as b:
        for lp, rp in files:
            b.put_file(lp, rp)
    dt = time.time() - t0
    print("%s: DONE %.0fs  %.1f MB/s" % (key, dt, p["bytes"] / 1e6 / max(dt, 1)), flush=True)
