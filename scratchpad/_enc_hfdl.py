"""Stage L3_HOST addendum 5: download the two canonical encoders onto the host, at the revisions the laptop cache holds.

Public repositories on huggingface.co, fetched anonymously (token=False; no token, no credential of any kind is read or sent).
Every file is verified against the laptop cache's blob name, which the addendum lists: the sha256 for the LFS weights, the git
blob sha1 for the small files.  The cache directory is outside the workspace and outside every fetch glob.

Usage (host, through rx): python -u scratchpad/_enc_hfdl.py -> results/L3_HOST/HF_DOWNLOAD__v1.json (exit 1 on any mismatch)
"""
import hashlib
import json
import os
import sys
import time

for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
    os.environ.pop(k, None)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

ADD = "results/L3_HOST/HOST_STAGE_DECLARATION__L3_HOST__v1__ADDENDUM_5.json"
OUT = "results/L3_HOST/HF_DOWNLOAD__v1.json"


def fhash(p, kind):
    h = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
    if kind == "git_sha1":
        h.update(b"blob %d\0" % os.path.getsize(p))
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if os.path.exists(OUT):
        sys.exit("refusing: %s exists (write-once)" % OUT)
    add = json.load(open(ADD, encoding="utf-8"))
    dl = add["download"]
    dest = dl["cache_dir"]
    os.makedirs(dest, exist_ok=True)
    rec = {"addendum": ADD, "addendum_sha256": fhash(ADD, "sha256"), "script_sha256": fhash(__file__, "sha256"),
           "cache_dir": dest, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": []}
    ok = True
    t_all = time.time()
    for e in dl["files"]:
        t0 = time.time()
        p = hf_hub_download(e["repo"], e["file"], revision=e["revision"], cache_dir=dest, token=False)
        dt = time.time() - t0
        n = os.path.getsize(p)
        h = fhash(p, e["hash_kind"])
        good = n == e["bytes"] and h == e["hash"]
        ok = ok and good
        rec["files"].append({"repo": e["repo"], "revision": e["revision"], "file": e["file"], "path": p.replace("\\", "/"),
                             "bytes": n, "hash_kind": e["hash_kind"], "hash": h, "expected": e["hash"],
                             "verified": good, "seconds": round(dt, 1)})
        print("%s %s %s  %d bytes  %.1f s  %s" % ("OK " if good else "BAD", e["repo"], e["file"], n, dt,
                                                  "%.1f MB/s" % (n / 1e6 / dt) if dt > 0 else ""), flush=True)
    rec["seconds"] = round(time.time() - t_all, 1)
    rec["bytes_total"] = sum(f["bytes"] for f in rec["files"])
    rec["status"] = "VERIFIED" if ok else "MISMATCH"
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print("status", rec["status"], "files", len(rec["files"]), "bytes", rec["bytes_total"], "seconds", rec["seconds"], flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
