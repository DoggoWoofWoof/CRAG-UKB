# -*- coding: utf-8 -*-
"""Top up a Modal account's copy of a dense store with the shards it is missing.

The corpus was encoded across the account pool and every account kept only the shards it
produced, so no single volume holds a whole dataset -- extra_wNzonK has 41 of the 150 2wiki
shards, spanishorgay 57, darkphoenix 6. The three accounts that DO hold complete copies are
all over their spend limit, so the work has to run on an account that has credit and be given
the rest of the corpus first.

Uploads one shard at a time on purpose: it is resumable (a shard already there is skipped, and
an interrupted run just restarts at the next one), and the per-file cost is invisible against
a 123 MB transfer.
"""
import io
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TREE = {"2wiki": "2wiki_universe", "hotpotqa": "hotpotqa"}
NSHARD = {"2wiki": 150, "hotpotqa": 131}
CENSUS = "data/_family_v1/_remote_census"


def remote_have(profile, tree):
    p = "%s/%s.%s.txt" % (CENSUS, profile, tree)
    if not os.path.exists(p):
        return None
    return {l.strip() for l in io.open(p, encoding="utf-8") if l.strip()}


def push(profile, ds):
    tree = TREE[ds]
    have = remote_have(profile, tree)
    if have is None:
        sys.exit("no census for %s/%s -- list the volume first" % (profile, tree))
    d = "data/canonical/%s/encodings/dense/docs" % tree
    miss = [i for i in range(NSHARD[ds]) if "shard_%05d.npy" % i not in have]
    tot = sum(os.path.getsize("%s/shard_%05d.npy" % (d, i)) for i in miss)
    print("[%s/%s] %d of %d shards missing, %.2f GB to push"
          % (profile, ds, len(miss), NSHARD[ds], tot / 1e9), flush=True)
    env = dict(os.environ, MODAL_PROFILE=profile, PYTHONIOENCODING="utf-8",
               MSYS_NO_PATHCONV="1")
    t0 = time.time()
    sent = 0
    for n, i in enumerate(miss, 1):
        rel = "%s/shard_%05d.npy" % (d, i)
        # decode as utf-8 explicitly: the modal CLI draws box characters, and the Windows
        # default cp1252 decoder raises on them mid-transfer.
        r = subprocess.run(["modal", "volume", "put", "crag-data-volume", rel, rel],
                           env=env, capture_output=True, encoding="utf-8", errors="replace")
        # the CLI wraps its error text inside a drawn box, so "already exists" can arrive
        # split across two lines with padding. Collapse whitespace before matching.
        out = (r.stdout or "") + (r.stderr or "")
        flat = re.sub(r"[\s│─┌┐└┘]+", " ", out)
        if r.returncode != 0 and "already exists" not in flat:
            print("  FAIL shard %d: %s" % (i, out.strip()[:300]), flush=True)
            return 1
        sent += os.path.getsize(rel)
        if n % 10 == 0 or n == len(miss):
            el = time.time() - t0
            print("  %d/%d  %.2f GB  %.1f MB/s  eta %.0f min"
                  % (n, len(miss), sent / 1e9, sent / 1e6 / max(1.0, el),
                     (tot - sent) / max(1.0, sent / el) / 60.0), flush=True)
    print("[%s/%s] DONE %.2f GB in %.0f min" % (profile, ds, sent / 1e9,
                                                (time.time() - t0) / 60.0), flush=True)
    return 0


rc = 0
for spec in sys.argv[1:]:
    profile, ds = spec.split(":")
    rc |= push(profile, ds)
sys.exit(rc)
