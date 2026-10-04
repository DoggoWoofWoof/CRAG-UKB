"""verify_freebase.py --full on the HOST, result to a NEW record (FREEBASE_SCALE lane transfer, user 2026-09-30).

Standalone verify_freebase.py overwrites data/final_canonical/freebase/VERIFICATION.json (a frozen record read by check_handoff.py; records
are superseded, never edited).  This wrapper runs the same verify_freebase(freeze, True, rep) against the transferred tree and writes
results/FREEBASE_SCALE/FBX_HOST_VERIFY__v1.json instead.  Every pinned byte is hashed against CANONICAL_FREEZE.json / DATASET.json, so a PASS is
also the proof that the transfer was byte-exact.  The tree is only read.

Usage (host, workspace root, tree reachable at data/final_canonical/freebase): python -u scratchpad/_fbx_verify_host.py
"""
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT + "/src/dataset_canonical")
sys.path.insert(0, ROOT + "/src/dataset_canonical/freebase")
OUTP = "results/FREEBASE_SCALE/FBX_HOST_VERIFY__v1.json"

import verify_freebase as V  # noqa: E402  (chdir to ROOT again, imports freeze_canonical)
from freeze_canonical import rj  # noqa: E402
from verify_canonical import Report, OUT  # noqa: E402


def main():
    if os.path.exists(OUTP):
        sys.exit("refusing: %s exists (write-once)" % OUTP)
    t0 = time.time()
    rep = Report()
    freeze = rj(OUT) if os.path.exists(OUT) else None
    V.verify_freebase(freeze, True, rep)
    res = {"RECORD": "VERIFICATION", "dataset": "freebase", "where": "host (rx), tree transferred via a private Hugging Face dataset repo",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": "full", "checks": len(rep.checks), "failed": rep.fail,
           "PASS": rep.fail == 0, "seconds": round(time.time() - t0, 1), "results": rep.checks}
    os.makedirs(os.path.dirname(OUTP), exist_ok=True)
    with io.open(OUTP, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(res, indent=1))
    print("%s  %d checks, %d failed, %.0fs  -> %s" % ("PASS" if rep.fail == 0 else "FAIL", len(rep.checks), rep.fail, time.time() - t0, OUTP), flush=True)
    sys.exit(0 if rep.fail == 0 else 1)


if __name__ == "__main__":
    main()
