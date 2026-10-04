# -*- coding: utf-8 -*-
"""Progress of a detached kNN compute, read from its checkpoints rather than its log stream.

`modal app logs` on a freshly-built image goes quiet for as long as the image build takes, and
the run is detached anyway. The checkpoint directory is the honest progress signal: one
q<NNNNN>.done per finished query shard, committed to the volume as it lands.
"""
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
NQ = {"hotpotqa": 80, "2wiki": 92, "webqsp": 28}

for spec in sys.argv[1:]:
    profile, ds = spec.split(":")
    env = dict(os.environ, MODAL_PROFILE=profile, PYTHONIOENCODING="utf-8",
               MSYS_NO_PATHCONV="1", COLUMNS="200")
    r = subprocess.run(["modal", "volume", "ls", "crag-knn-bundle", "/%s/_ckpt" % ds],
                       env=env, capture_output=True, encoding="utf-8", errors="replace")
    done = sorted(set(re.findall(r"q(\d{5})\.done", r.stdout or "")))
    n = NQ.get(ds, 0)
    print("[%s/%s] %d/%d query shards done%s"
          % (profile, ds, len(done), n,
             ("  last=q%s" % done[-1]) if done else "  (compute has not written one yet)"),
          flush=True)
