"""Pull the Freebase tree onto the host from time-limited links (FREEBASE_SCALE lane transfer, user 2026-09-30).  Standard library only, no token.

Reads the manifest written by _fbx_hf_urls.py on the laptop (pushed to the host), fetches every file with parallel ranged GETs into
<dest>/<path>.part (resumable: finished chunks are recorded in <path>.part.done), and renames to <path> when the size is exact.  A file that
already exists with the right size is skipped.  HTTP 400/401/403 = the link expired: the job stops with exit 3 and the laptop regenerates
the manifest (only the missing files are fetched again).  Integrity is NOT judged here: verify_freebase.py --full hashes every pinned byte.

Usage (host): python -u scratchpad/_fbx_hf_pull.py --manifest data/_cache/fbx_urls.json --dest C:/Users/Student2/rx/projects/crag/data/freebase
              [--threads 16] [--chunk-mb 64] [--only SUBSTR]
"""
import argparse
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

EXPIRED = (400, 401, 403)


class Expired(Exception):
    pass


def fetch_range(url, a, b):
    last = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b), "User-Agent": "crag-fbx-pull/1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != b - a + 1:
                raise IOError("short read %d of %d" % (len(data), b - a + 1))
            return data
        except urllib.error.HTTPError as e:
            if e.code in EXPIRED:
                raise Expired(str(e.code))
            last = e
        except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(min(30, 2 ** attempt))
    raise IOError("range %d-%d failed after retries: %r" % (a, b, last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    man = json.load(open(a.manifest, encoding="utf-8"))
    ch = a.chunk_mb << 20
    files = [f for f in man["files"] if not a.only or any(o in f["path"] for o in a.only)]
    age = time.time() - man["generated_epoch"]
    print("manifest %s: %d files, %.3f GB, links %.0f s old (lifetime %s s)" % (man["repo"], len(files), sum(f["size"] for f in files) / 1e9, age, man.get("url_lifetime_s")), flush=True)
    tasks, todo_bytes = [], 0
    done_lock = threading.Lock()
    state = {}
    for f in sorted(files, key=lambda x: -x["size"]):
        dst = os.path.join(a.dest, f["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) == f["size"]:
            continue
        if "inline_b64" in f:
            data = base64.b64decode(f["inline_b64"])
            assert len(data) == f["size"]
            with open(dst + ".tmp", "wb") as o:
                o.write(data)
            os.replace(dst + ".tmp", dst)
            continue
        part = dst + ".part"
        donef = part + ".done"
        done = set()
        if os.path.exists(donef) and os.path.exists(part):
            done = {int(x) for x in open(donef).read().split()}
        else:
            with open(part, "wb") as o:
                o.truncate(f["size"])
            open(donef, "w").close()
        n = (f["size"] + ch - 1) // ch
        state[f["path"]] = {"left": n - len(done), "dst": dst, "part": part, "donef": donef, "size": f["size"]}
        for i in range(n):
            if i not in done:
                lo = i * ch
                tasks.append((f["path"], f["url"], part, donef, i, lo, min(f["size"], lo + ch) - 1))
                todo_bytes += min(f["size"], lo + ch) - lo
    print("to fetch: %d chunks, %.3f GB" % (len(tasks), todo_bytes / 1e9), flush=True)
    got = [0]
    t0 = time.time()
    stop = threading.Event()

    def work(t):
        path, url, part, donef, i, lo, hi = t
        if stop.is_set():
            return None
        data = fetch_range(url, lo, hi)
        with open(part, "r+b") as o:
            o.seek(lo)
            o.write(data)
        with done_lock:
            with open(donef, "a") as d:
                d.write("%d\n" % i)
            got[0] += len(data)
            s = state[path]
            s["left"] -= 1
            if s["left"] == 0:
                if os.path.getsize(part) != s["size"]:
                    raise IOError("size mismatch after fetch: " + path)
                os.replace(part, s["dst"])
                os.remove(donef)
                print("  done %s (%.2f GB)" % (path, s["size"] / 1e9), flush=True)
        return len(data)

    rc = 0
    last = time.time()
    with ThreadPoolExecutor(a.threads) as ex:
        futs = [ex.submit(work, t) for t in tasks]
        try:
            for fu in as_completed(futs):
                fu.result()
                if time.time() - last > 30:
                    last = time.time()
                    print("  %.2f / %.2f GB  %.1f MB/s" % (got[0] / 1e9, todo_bytes / 1e9, got[0] / 1e6 / (time.time() - t0)), flush=True)
        except Expired:
            stop.set()
            rc = 3
            print("LINKS EXPIRED (HTTP 4xx): regenerate the manifest on the laptop and re-run; finished chunks are kept", flush=True)
        except BaseException:
            stop.set()
            raise
    left = [p for p, s in state.items() if s["left"] > 0]
    print("%s: fetched %.3f GB in %.0f s (%.1f MB/s); files still incomplete: %d" % ("STOPPED" if rc else "OK", got[0] / 1e9, time.time() - t0, got[0] / 1e6 / max(1, time.time() - t0), len(left)), flush=True)
    sys.exit(rc if rc else (0 if not left else 1))


if __name__ == "__main__":
    main()
