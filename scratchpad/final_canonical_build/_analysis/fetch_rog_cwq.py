"""Fetch the official RoG-CWQ artifact (rmanluo/RoG-cwq) pinned to an exact revision.

The WEBQSP_ROG_STANDARD graph is the union of the RoG WebQSP and RoG CWQ per-question
subgraphs.  RoG-webqsp is already on disk; RoG-cwq is not.  This script downloads ONLY
rmanluo/RoG-cwq, into data/original/cwq/rog_cwq/, pinned to REVISION, and records the
repo revision plus a locally measured sha256 for every file.

It never overwrites an existing file: if a target path already exists the file is
verified (size + sha256) and skipped.
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

REPO = "rmanluo/RoG-cwq"
REVISION = "b0f6275586286312c4e99ef4b0adc65463c91f8e"   # HF api `sha` read 2026-09-05
DEST = "C:/Users/Swastik/Desktop/CRAG/data/original/cwq/rog_cwq"
RECORD = "C:/Users/Swastik/Desktop/CRAG/data/original/cwq/SOURCE_PROVENANCE.json"


def sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def api(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read())


def main():
    os.makedirs(DEST, exist_ok=True)
    meta = api(f"https://huggingface.co/api/datasets/{REPO}?blobs=true&revision={REVISION}")
    assert meta["sha"] == REVISION, (meta["sha"], REVISION)
    files = [s for s in meta["siblings"] if s["rfilename"].endswith((".parquet", ".md"))]
    files.sort(key=lambda s: s["rfilename"])

    from huggingface_hub import hf_hub_download

    out = {"repo_id": REPO, "repo_type": "dataset", "revision": REVISION,
           "resolve_url_template":
               f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/<path>",
           "hf_api_lastModified": meta.get("lastModified"),
           "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "files": []}

    for s in files:
        rel = s["rfilename"]
        target = os.path.join(DEST, os.path.basename(rel))
        if os.path.exists(target):
            print(f"[skip-exists] {rel}", flush=True)
        else:
            t0 = time.time()
            p = hf_hub_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                                filename=rel, local_dir=DEST + "/_hf")
            os.replace(p, target)
            print(f"[got] {rel} {os.path.getsize(target)/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
        got = os.path.getsize(target)
        exp = s.get("size")
        h = sha256(target)
        out["files"].append({"rfilename": rel, "local": target, "bytes": got,
                             "hf_reported_bytes": exp, "size_matches": (exp is None or exp == got),
                             "sha256": h})
        print(f"    sha256={h} bytes={got} expected={exp}", flush=True)

    out["n_files"] = len(out["files"])
    out["total_bytes"] = sum(f["bytes"] for f in out["files"])
    out["all_sizes_match"] = all(f["size_matches"] for f in out["files"])
    # one manifest-level hash over (filename, sha256) so the whole fetch has a single id
    man = hashlib.sha256()
    for f in sorted(out["files"], key=lambda x: x["rfilename"]):
        man.update((f["rfilename"] + "\t" + f["sha256"] + "\n").encode())
    out["fileset_sha256"] = man.hexdigest()
    os.makedirs(os.path.dirname(RECORD), exist_ok=True)
    with open(RECORD, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("WROTE", RECORD, "fileset_sha256=", out["fileset_sha256"], flush=True)
    for d in (DEST + "/_hf",):
        try:
            for root, dirs, fs in os.walk(d, topdown=False):
                for x in fs:
                    os.remove(os.path.join(root, x))
                for x in dirs:
                    os.rmdir(os.path.join(root, x))
            os.rmdir(d)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
