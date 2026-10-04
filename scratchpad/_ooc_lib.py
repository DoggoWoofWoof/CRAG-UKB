"""FBX_SCALE addendum 17 -- the Python side of the out-of-core engine: build / run scratchpad/_ooc.cpp in the host's WSL distro, raw-file helpers.

The engine (scratchpad/_ooc.cpp) works on raw little-endian files; this module only compiles it (once per source sha, into data/ooc/bin on the workspace drive, as the earlier _vc binaries),
runs one sub-command, and returns the JSON line the sub-command prints.  Paths starting with '/' are WSL paths (the distro's own ext4, used for the big Freebase files); anything else is a
workspace path and goes through /mnt/<drive>.
"""
import hashlib
import json
import os
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DISTRO = os.environ.get("FBX_WSL_DISTRO", "Ubuntu-24.04")
SRC = os.path.join(HERE, "_ooc2.cpp")                   # _ooc2.cpp includes _ooc.cpp and adds transpose / fm2; one binary serves every sub-command
SRC_PARTS = ["_ooc.cpp", "_ooc2.cpp"]
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def wp(p):
    """a path as the WSL distro sees it"""
    p = str(p)
    if p.startswith("/"):
        return p
    p = os.path.abspath(p).replace("\\", "/")
    if os.name == "nt" and len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def _wsl(cmd, capture=True):
    full = (["wsl.exe", "-d", DISTRO, "--exec"] + cmd) if os.name == "nt" else cmd
    r = subprocess.run(full, stdout=subprocess.PIPE, stderr=subprocess.PIPE if capture else None, text=True)
    if r.returncode != 0:
        raise RuntimeError("%s -> exit %d: %s %s" % (" ".join(cmd)[:300], r.returncode, r.stdout[-2000:], (r.stderr or "")[-4000:]))
    return r.stdout


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(src=SRC):
    h = hashlib.sha256()
    for part in SRC_PARTS:
        h.update(sha_file(os.path.join(HERE, part)).encode())
    sha = h.hexdigest()[:12]
    bindir = os.path.join(REPO, "data", "ooc", "bin")
    os.makedirs(bindir, exist_ok=True)
    name = os.path.splitext(os.path.basename(src))[0]
    binp = wp(os.path.join(bindir, "%s_%s" % (name, sha)))
    try:
        _wsl(["test", "-x", binp])
    except RuntimeError:
        log("compiling", os.path.basename(src), sha)
        _wsl(["g++", "-O2", "-std=c++17", "-fopenmp", "-Wall", "-o", binp, wp(src)])
    return binp, sha


def run(sub, args, stream=True, binp=None):
    """run one sub-command; args = {name: value | [values]}; returns the parsed JSON line.  stream: the engine's progress lines go to this process's stderr."""
    binp = binp or build()[0]
    cmd = [binp, sub]
    for k, v in args.items():
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            cmd += ["--" + k, str(x)]
    t = time.time()
    cap = os.environ.get("FBX_ENGINE_ANON_GB")                # the job's measured reservation: RLIMIT_DATA caps the engine's ANONYMOUS memory (file-backed mmaps are not counted), so a mis-sized job fails by itself
    if cap:
        cmd = ["bash", "-c", 'ulimit -d %d; exec "$@"' % int(float(cap) * 1024 * 1024), "_"] + cmd
    full = (["wsl.exe", "-d", DISTRO, "--exec"] + cmd) if os.name == "nt" else cmd
    r = subprocess.run(full, stdout=subprocess.PIPE, stderr=None if stream else subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError("_ooc %s -> exit %d: %s %s" % (sub, r.returncode, r.stdout[-1500:], (r.stderr or "")[-3000:]))
    line = [x for x in r.stdout.strip().splitlines() if x.startswith("{")]
    info = json.loads(line[-1]) if line else {}
    info["wall_seconds"] = round(time.time() - t, 1)
    return info


def wr(arr, path, dt):
    np.ascontiguousarray(arr, dt).tofile(path)
    return path


def rd(path, dt):
    return np.fromfile(path, dtype=dt)
