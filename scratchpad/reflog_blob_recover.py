"""Recover the four reflog-only blobs (user ruling: recover before any git gc) into archive/git_reflog_recovered/.
Per blob: size check vs the recorded size -> stream `git cat-file blob` through Python (no shell redirect: no CRLF / pipe mangling) into
<sha8>__<basename> while hashing sha256 -> `git hash-object --no-filters` of the recovered file must equal the blob sha.
Record: results/DATA_CLEANUP/REFLOG_BLOB_RECOVERY__2026-09-30.json (blob sha, recovered-file sha256, size, original path).  No gc, no expire."""
import hashlib
import json
import os
import subprocess
import time

REPO = "C:/Users/Swastik/Desktop/CRAG"
OUT = REPO + "/archive/git_reflog_recovered"
REC = REPO + "/results/DATA_CLEANUP/REFLOG_BLOB_RECOVERY__2026-09-30.json"
BLOBS = [  # (sha prefix, recorded size, original path)
    ("0545c7b5", 878488729, "metaqa_movie/MetaQA/checkpoint/optimizer.pt"),
    ("4c18c36c", 439279007, "metaqa_movie/MetaQA/checkpoint/pytorch_model.bin"),
    ("2488534e", 510138664, "real/graph/full_graph.pt"),
    ("43826bf8", 353772715, "real/graph/embeddings/embeddings.pt"),
]
CH = 8 << 20


def git(*a):
    return subprocess.run(["git", "-C", REPO] + list(a), capture_output=True, text=True, check=True).stdout.strip()


os.makedirs(OUT, exist_ok=True)
with open(OUT + "/.gitignore", "w", newline="\n") as f:
    f.write("*\n!.gitignore\n!RECOVERY_NOTE.txt\n")
rows = []
for pre, size, orig in BLOBS:
    full = git("rev-parse", "--verify", pre + "^{blob}")
    assert int(git("cat-file", "-s", full)) == size, "size differs from the record: " + pre
    dst = "%s/%s__%s" % (OUT, pre, os.path.basename(orig))
    assert not os.path.exists(dst), "exists: " + dst
    t0 = time.time()
    h = hashlib.sha256()
    n = 0
    p = subprocess.Popen(["git", "-C", REPO, "cat-file", "blob", full], stdout=subprocess.PIPE)
    with open(dst + ".partial", "wb") as o:
        for b in iter(lambda: p.stdout.read(CH), b""):
            h.update(b)
            o.write(b)
            n += len(b)
    assert p.wait() == 0 and n == size, "stream failed: %s (%d of %d bytes)" % (pre, n, size)
    os.replace(dst + ".partial", dst)
    assert git("hash-object", "--no-filters", dst) == full, "recovered file hashes differently from the blob: " + pre
    rows.append({"blob_sha": full, "recovered_file": dst[len(REPO) + 1:], "recovered_sha256": h.hexdigest(), "bytes": n, "original_path": orig,
                 "git_hash_object_matches_blob": True, "seconds": round(time.time() - t0, 1)})
    print("OK %s  %s  %d bytes  sha256 %s  (%.0fs)" % (pre, orig, n, h.hexdigest()[:16], time.time() - t0), flush=True)
with open(REC, "w", encoding="utf-8", newline="\n") as f:
    json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": "reflog-only blobs recovered before any git gc; nothing expired or pruned", "blobs": rows}, f, indent=1, sort_keys=True)
print("record:", REC)
