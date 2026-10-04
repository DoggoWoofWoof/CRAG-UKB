"""Host housekeeping -- READ-ONLY manifest of the crag files on the lab host (rx workspace + the rx project-level data/models dirs), for a laptop-side comparison before anything is deleted.
Writes work/HOST_HOUSEKEEPING/manifest.jsonl: one row per file {root, rel, size, mtime, nlink, sha (files < 64 MB only)}.  Deletes nothing.   python -u scratchpad/_host_manifest.py"""
import hashlib
import json
import os
import time

WS = os.getcwd()
PROJ = os.path.dirname(WS)                                  # ...\rx\projects\crag
ROOTS = [("ws", WS), ("proj_data", os.path.join(PROJ, "data")), ("proj_models", os.path.join(PROJ, "models"))]
SKIP_TOP = {"jobs", ".rx", "__pycache__", ".git"}
OUT = os.path.join(WS, "work", "HOST_HOUSEKEEPING")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
n = 0
tot = 0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


with open(os.path.join(OUT, "manifest.jsonl"), "w", newline="\n") as fo:
    for name, root in ROOTS:
        if not os.path.isdir(root):
            print("missing root", root, flush=True)
            continue
        for dp, dn, fn in os.walk(root):
            if dp == root:
                dn[:] = [d for d in dn if d not in SKIP_TOP]
            if os.path.abspath(dp).startswith(OUT):          # never list our own output
                dn[:] = []
                continue
            for f in fn:
                p = os.path.join(dp, f)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                row = {"root": name, "rel": os.path.relpath(p, root).replace("\\", "/"), "size": st.st_size, "mtime": int(st.st_mtime), "nlink": st.st_nlink}
                if st.st_size < 64 << 20 and not f.endswith(".part"):
                    try:
                        row["sha"] = sha(p)
                    except OSError:
                        pass
                fo.write(json.dumps(row) + "\n")
                n += 1
                tot += st.st_size
                if n % 5000 == 0:
                    print("%d files, %.1f GB, %.0fs" % (n, tot / 1e9, time.time() - t0), flush=True)
print("DONE %d files, %.1f GB, %.0fs" % (n, tot / 1e9, time.time() - t0), flush=True)
