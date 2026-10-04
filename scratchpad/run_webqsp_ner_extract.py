# -*- coding: utf-8 -*-
"""Fan the webqsp NER extract across cores. Each worker owns a disjoint shard range and the
builder is per-shard resumable, so a killed worker costs at most one shard."""
import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NS = 65
W = 6
env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
edges = [round(i * NS / float(W)) for i in range(W + 1)]
procs = []
for i in range(W):
    a, b = edges[i], edges[i + 1] - 1
    if a > b:
        continue
    cmd = [sys.executable, "-u", "scratchpad/build_ner_sharded.py", "--dataset", "webqsp",
           "--stage", "extract", "--canon-root", "data/_family_v1", "--shards", "%d-%d" % (a, b)]
    log = open("scratchpad/_ner_w%d.log" % i, "w", encoding="utf-8")
    procs.append((i, a, b, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
    print("worker %d -> shards %d-%d" % (i, a, b), flush=True)

t0 = time.time()
bad = 0
for i, a, b, p, log in procs:
    rc = p.wait()
    log.close()
    print("worker %d (shards %d-%d) rc=%d  %.0fs" % (i, a, b, rc, time.time() - t0), flush=True)
    bad += (rc != 0)
done = len([f for f in os.listdir("data/_family_v1/webqsp/_ner_work/postings")
            if f.endswith(".done")]) if os.path.isdir("data/_family_v1/webqsp/_ner_work/postings") else 0
print("EXTRACT %s  shards done %d/%d  %.0fs" % ("OK" if bad == 0 else "FAIL", done, NS, time.time() - t0))
sys.exit(1 if bad or done != NS else 0)
