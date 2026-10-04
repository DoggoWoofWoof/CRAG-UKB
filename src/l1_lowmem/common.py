"""Shared paths / helpers for the low-memory partitioning experiment."""
import hashlib
import io
import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for _p in (REPO, os.path.join(REPO, "scratchpad")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = os.path.join(REPO, "data", "l1_lowmem")
OUT = os.path.join(REPO, "results", "L1_LOWMEM")
DISTRO = "Ubuntu-22.04"
LOCAL_SETS = ["squad", "metaqa", "musique"]          # the validation order the ruling fixes
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def sha_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def rj(p):
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


def wj(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1))


def pin(p):
    return {"path": os.path.relpath(p, REPO).replace("\\", "/"), "bytes": os.path.getsize(p), "sha256": sha_file(p)} if os.path.exists(p) else None


def ds_dir(ds):
    d = os.path.join(DATA, ds)
    os.makedirs(d, exist_ok=True)
    return d


def wsl_path(p):
    p = os.path.abspath(p).replace("\\", "/")
    return "/mnt/" + p[0].lower() + p[2:]


def wsl(cmd, timeout=None, check=False):
    """run a bash -lc command inside the WSL distro; returns CompletedProcess (utf-8 text)."""
    r = subprocess.run(["wsl", "-d", DISTRO, "-e", "bash", "-lc", cmd], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError("wsl command failed (%d): %s\n%s\n%s" % (r.returncode, cmd, r.stdout[-2000:], r.stderr[-2000:]))
    return r


def source_lines(path, patterns):
    """{pattern: [(lineno, line)]} for exact-substring patterns -- the record quotes the defining code, never a paraphrase."""
    out = {}
    with io.open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for pat in patterns:
        out[pat] = [(i + 1, ln.rstrip()) for i, ln in enumerate(lines) if pat in ln]
    return out
