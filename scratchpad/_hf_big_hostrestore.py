"""Host job: restore objects of the HF third-copy repo with NO token and prove they are bit-identical to what is on the host.

  python -u scratchpad/_hf_big_hostrestore.py --manifest data/_cache/hfbig_urls.json --work data/_cache/hfbig_work --dest data/_cache/hfbig_restored
        [--compare data/final_canonical] [--wheel data/_cache/whl/zstandard-0.25.0-cp313-cp313-win_amd64.whl] [--record results/HOST_HOUSEKEEPING/HFBIG_HOST_RESTORE__test.json] [--cleanup]

1. scratchpad/_fbx_hf_pull.py pulls the stored objects named in the manifest (signed links, written by `_hf_big.py URLS` on the laptop) into --work;
2. scratchpad/_hf_big_unpack.py decompresses + verifies sha256 against the index (sha256 of the original bytes) into --dest/final_canonical/...;
3. every restored file that also exists under --compare (the live host copy) is hashed on both sides: this is the evidence that deleting the live copy loses nothing;
4. a record is written; --cleanup removes --work and --dest afterwards (a test restore must not leave the bytes behind).
Exit 0 only if the pull, every unpack verification and every live-copy comparison succeeded."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--compare", default="")
    ap.add_argument("--wheel", default="data/_cache/whl/zstandard-0.25.0-cp313-cp313-win_amd64.whl")
    ap.add_argument("--record", default="results/HOST_HOUSEKEEPING/HFBIG_HOST_RESTORE__test.json")
    ap.add_argument("--threads", default="8")
    ap.add_argument("--cleanup", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    rec = {"manifest": a.manifest, "started": time.strftime("%F %T")}
    rc1 = subprocess.call([sys.executable, "-u", "scratchpad/_fbx_hf_pull.py", "--manifest", a.manifest, "--dest", a.work, "--threads", a.threads])
    rec["pull_rc"] = rc1
    rc2 = subprocess.call([sys.executable, "-u", "scratchpad/_hf_big_unpack.py", "--src", a.work, "--dest", a.dest, "--wheel", a.wheel]) if rc1 == 0 else 99
    rec["unpack_rc"] = rc2
    cmp_ok = cmp_bad = cmp_none = 0
    base = os.path.join(a.dest, "final_canonical")
    if rc2 == 0 and a.compare and os.path.isdir(base):
        for d, _, fs in os.walk(base):
            for f in fs:
                p = os.path.join(d, f)
                rel = os.path.relpath(p, base).replace("\\", "/")
                live = os.path.join(a.compare, rel)
                if os.path.exists(live):
                    if os.path.getsize(live) == os.path.getsize(p) and sha(live) == sha(p):
                        cmp_ok += 1
                    else:
                        cmp_bad += 1
                        print("LIVE COPY DIFFERS:", rel, flush=True)
                else:
                    cmp_none += 1
    rec.update({"compare_identical": cmp_ok, "compare_different": cmp_bad, "compare_no_live_copy": cmp_none, "seconds": round(time.time() - t0, 1)})
    os.makedirs(os.path.dirname(a.record), exist_ok=True)
    json.dump(rec, open(a.record, "w"), indent=1)
    print(json.dumps(rec), flush=True)
    if a.cleanup:
        shutil.rmtree(a.work, ignore_errors=True)
        shutil.rmtree(a.dest, ignore_errors=True)
    sys.exit(0 if rc1 == 0 and rc2 == 0 and cmp_bad == 0 else 1)


if __name__ == "__main__":
    main()
