"""Resume + FULL VERIFY of the official RoG-CWQ artifact (rmanluo/RoG-cwq).

Supersedes fetch_rog_cwq.py, which was killed mid-run (14 of 24 files on disk).
Differences:
  * verifies EVERY file that already exists against the HF-reported LFS sha256
    (the previous script trusted name-existence and only recorded a hash),
  * re-fetches anything whose size or sha256 does not verify (the shard that was
    in flight when the process died can be truncated),
  * knows the true file set is 24 parquet shards (3 test / 18 train / 3 validation),
    not the 21 previously assumed.
Writes only under data/original/cwq/ (authorised) and the scratchpad log.
"""
import hashlib
import json
import os
import shutil
import sys
import time

from huggingface_hub import HfApi, hf_hub_download

REPO = "rmanluo/RoG-cwq"
REVISION = "b0f6275586286312c4e99ef4b0adc65463c91f8e"
DEST = "C:/Users/Swastik/Desktop/CRAG/data/original/cwq/rog_cwq"
TMP = DEST + "/_hf"
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


def main():
    os.makedirs(DEST, exist_ok=True)
    api = HfApi()
    info = api.repo_info(REPO, repo_type="dataset", revision=REVISION, files_metadata=True)
    assert info.sha == REVISION, (info.sha, REVISION)

    sibs = [s for s in info.siblings if s.rfilename.endswith((".parquet", ".md"))]
    sibs.sort(key=lambda s: s.rfilename)
    print(f"[repo] {REPO}@{REVISION}  files={len(sibs)}", flush=True)

    out = {
        "repo_id": REPO, "repo_type": "dataset", "revision": REVISION,
        "resolve_url_template": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/<path>",
        "hf_api_lastModified": str(info.last_modified),
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verification": "every file size+sha256 checked against HF-reported LFS sha256",
        "files": [],
    }
    refetched, verified_ok = [], []

    for s in sibs:
        rel = s.rfilename
        exp_size = s.size
        exp_sha = s.lfs.sha256 if getattr(s, "lfs", None) else None
        target = os.path.join(DEST, os.path.basename(rel))

        action = None
        for attempt in (1, 2):
            if os.path.exists(target):
                got = os.path.getsize(target)
                if exp_size is not None and got != exp_size:
                    print(f"[BAD-SIZE] {rel} {got} != {exp_size} -> refetch", flush=True)
                    os.remove(target)
                else:
                    h = sha256(target)
                    if exp_sha is None or h == exp_sha:
                        action = action or "verified-existing"
                        break
                    print(f"[BAD-SHA] {rel} {h} != {exp_sha} -> refetch", flush=True)
                    os.remove(target)
            t0 = time.time()
            p = hf_hub_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                                filename=rel, local_dir=TMP)
            os.replace(p, target)
            action = "refetched"
            print(f"[got] {rel} {os.path.getsize(target)/1e6:.1f} MB in {time.time()-t0:.0f}s", flush=True)
        else:
            print(f"[FAIL] {rel} did not verify after 2 attempts", flush=True)
            action = "FAILED"

        got = os.path.getsize(target) if os.path.exists(target) else 0
        h = sha256(target) if os.path.exists(target) else None
        ok = (exp_sha is None or h == exp_sha) and (exp_size is None or got == exp_size)
        (verified_ok if ok else refetched).append(rel)
        out["files"].append({
            "rfilename": rel, "local": target, "bytes": got,
            "hf_reported_bytes": exp_size, "hf_lfs_sha256": exp_sha,
            "local_sha256": h, "verifies": ok, "action": action,
        })
        print(f"    {action:18s} sha256={h} bytes={got} verifies={ok}", flush=True)

    out["n_files"] = len(out["files"])
    out["total_bytes"] = sum(f["bytes"] for f in out["files"])
    out["all_verify"] = all(f["verifies"] for f in out["files"])
    out["n_parquet"] = sum(1 for f in out["files"] if f["rfilename"].endswith(".parquet"))
    man = hashlib.sha256()
    for f in sorted(out["files"], key=lambda x: x["rfilename"]):
        man.update((f["rfilename"] + "\t" + (f["local_sha256"] or "") + "\n").encode())
    out["fileset_sha256"] = man.hexdigest()

    os.makedirs(os.path.dirname(RECORD), exist_ok=True)
    with open(RECORD, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"WROTE {RECORD}", flush=True)
    print(f"ALL_VERIFY={out['all_verify']} n_parquet={out['n_parquet']} "
          f"total_bytes={out['total_bytes']} fileset_sha256={out['fileset_sha256']}", flush=True)
    return 0 if out["all_verify"] else 1


if __name__ == "__main__":
    sys.exit(main())
