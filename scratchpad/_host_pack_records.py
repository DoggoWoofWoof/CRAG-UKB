"""Host housekeeping -- pack the SMALL files that may exist only on the host (records, run logs, run-dir command/json files, small checkpoints) into one zip so they can be fetched to the laptop.
READ-ONLY on everything except its own output work/HOST_HOUSEKEEPING/<out>.zip (+ .manifest.json).
   python -u scratchpad/_host_pack_records.py [since=<host epoch seconds>] [out=<name>]
since: only files modified at/after that time are packed (incremental sync; _host_sync.py passes the previous run's host clock minus an hour); out: zip stem (default host_records).
The manifest json carries {"host_now": <epoch>} so the next incremental run asks the host clock, not the laptop clock."""
import hashlib
import json
import os
import sys
import time
import zipfile

WS = os.getcwd()
OUT = os.path.join(WS, "work", "HOST_HOUSEKEEPING")
os.makedirs(OUT, exist_ok=True)
ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SINCE = float(ARGS.get("since", 0))
STEM = ARGS.get("out", "host_records")
NOW = time.time()
# root -> (max bytes per file, extensions)
TXT = (".json", ".md", ".txt", ".csv", ".log", ".jsonl", ".py", ".sh", ".toml")
ROOTS = {
    "results": (8 << 20, TXT + (".npz",)),
    "work/L1_HOST": (3 << 20, TXT), "work/FBX_SCALE": (3 << 20, TXT), "work/FBX_CALIB": (3 << 20, TXT),
    "data/_cache": (64 << 20, (".json", ".npz", ".jsonl", ".txt")),            # incl. the host L1 text scorer checkpoints (txtscore_*.partial.npz)
    "data/freebase_scale/enc_test": (3 << 20, TXT),
    "scratchpad": (3 << 20, TXT),
}
SKIP_PARTS = ("HOST_HOUSEKEEPING", "restore_dl", "hfx_urls_")                  # our own transfer scratch + signed-url files (never copy those around)
n = 0
tot = 0
man = []
zp = os.path.join(OUT, STEM + ".zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for r, (maxb, ext) in ROOTS.items():
        base = os.path.join(WS, r)
        if not os.path.isdir(base):
            continue
        for dp, dn, fn in os.walk(base):
            for f in fn:
                p = os.path.join(dp, f)
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                if st.st_size > maxb or not f.lower().endswith(ext) or any(s in p for s in SKIP_PARTS) or st.st_mtime < SINCE:
                    continue
                rel = os.path.relpath(p, WS).replace("\\", "/")
                try:
                    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
                except OSError:
                    continue
                z.write(p, rel)
                man.append({"rel": rel, "size": st.st_size, "sha": h, "mtime": st.st_mtime})
                n += 1
                tot += st.st_size
json.dump({"host_now": NOW, "since": SINCE, "files": man}, open(os.path.join(OUT, STEM + ".manifest.json"), "w"))
import shutil
print("packed %d files, %.1f MB raw -> %.1f MB zip (since %s); host drive free %.1f GB" % (n, tot / 1e6, os.path.getsize(zp) / 1e6, SINCE, shutil.disk_usage(WS).free / 1e9), flush=True)
