"""FREIGHT arm of the H4_SK low-memory partitioning validation (KaHIP/FREIGHT, official repository, WSL).

Experiment 1 (engineering equivalence, exact):   A == B == B0 == C_kill == C_abort   position by position
    A       stock freight_con on the official monolithic net-list (hmetis_to_freight_stream output)
    A_forklib  stock driver (upstream app/freight.cpp) linked against the patched library, monolithic net-list (hooks NULL)
    B       fork freight_con_ckpt on the H4_SK_STREAM_V1 shards, one persistent process, checkpoints disabled (--ckpt_every_shards=0)
    B0      fork on the shards, checkpoint after every shard, never interrupted
    C_kill  fork, SIGKILL from a watcher once LATEST reaches the middle shard, then --ckpt_resume
    C_abort fork, --ckpt_abort_after_node (mid-shard _exit(137)), then --ckpt_resume
    + 14 refusal tests (corrupt / truncated / .tmp only / LATEST hash / foreign binary, structure, manifest, commit / wrong objective, k,
      imbalance, seed, passes, shard count) and a 2-pass smoke test (squad)
Experiment 2 input: the B partition is imported as data/l1_canonical/<ds>/parts/LOWMEM__FREIGHT_con.{npy,json}
(canonical manifest shape, EXPERIMENTAL) for l1_downstream.py.  Nothing canonical is modified.

    python -u src/l1_lowmem/freight.py build                 -> results/L1_LOWMEM/FREIGHT_BUILD.json
    python -u src/l1_lowmem/freight.py convert <ds> ...      -> data/l1_lowmem/<ds>/H4_SK.official.netl + bytes gate
    python -u src/l1_lowmem/freight.py run <ds> ...          -> results/L1_LOWMEM/FREIGHT_RUNS_<ds>.json (+ partition import)
Objective: connectivity (weighted KM1) -- the canonical objective family; --imbalance=3 (eps 0.03), --seed=0, one pass.
"""
import io
import json
import math
import os
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from src.l1_lowmem.common import REPO, OUT, DISTRO, log, sha_file, rj, wj, pin, ds_dir, wsl_path, wsl  # noqa: E402
from src.l1_lowmem import hgr as HG  # noqa: E402
from src.l1_lowmem import stream as ST  # noqa: E402

FREIGHT = "~/FREIGHT"                  # pristine official checkout (stock binaries, run A)
FORK = "~/FREIGHT_ckpt"                # copy + apply_fork.py (freight_con_ckpt)
CKPT_HOME = "~/l1_lowmem_ckpt"         # checkpoints live on the WSL ext4 fs (POSIX fsync/rename semantics)
IMBALANCE = 3                          # percent == eps 0.03 of the canonical contract
SEED = 0
TAG = "LOWMEM__FREIGHT_con"
FORK_DIR = os.path.join(REPO, "src", "l1_lowmem", "freight_fork")


def sh(cmd, timeout=None, check=True, quiet=False):
    r = wsl(cmd, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError("wsl rc=%d: %s\nSTDOUT %s\nSTDERR %s" % (r.returncode, cmd, r.stdout[-3000:], r.stderr[-3000:]))
    if not quiet and r.stdout.strip():
        for ln in r.stdout.strip().split("\n")[-6:]:
            log("  |", ln)
    return r


def home():
    return sh("echo $HOME", quiet=True).stdout.strip()


# ----------------------------------------------------------------------------- build
def build():
    t = time.time()
    rec = {"RECORD": "FREIGHT_BUILD", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "distro": DISTRO}
    tc = sh("g++ --version | head -1; cmake --version | head -1; make --version | head -1; uname -srm; lscpu | grep 'Model name' | sed 's/ \\+/ /g'",
            quiet=True).stdout.strip().split("\n")
    rec["toolchain"] = tc
    for ln in tc:
        log("  toolchain:", ln)
    git = sh("cd %s && git rev-parse HEAD && git status --porcelain | wc -l && git remote get-url origin" % FREIGHT, quiet=True).stdout.split("\n")
    rec["upstream"] = {"commit": git[0].strip(), "dirty_files": int(git[1].strip()), "origin": git[2].strip()}
    if rec["upstream"]["dirty_files"] != 0:
        raise RuntimeError("~/FREIGHT is not pristine (%d modified files)" % rec["upstream"]["dirty_files"])
    log("upstream commit", rec["upstream"]["commit"], "clean")
    # pristine tree -> stock binaries
    sh("cd %s && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release > cmake.log 2>&1 && "
       "make -j8 freight_con freight_cut hmetis_to_freight_stream hmetis_to_freight > make.log 2>&1" % FREIGHT, timeout=3600)
    log("stock binaries built")
    # fork tree: fresh copy of the pristine checkout (no build dir), apply_fork.py, build
    sh("rm -rf %s && mkdir -p %s && cd %s && git archive HEAD | tar -x -C %s" % (FORK, FORK, FREIGHT, FORK), timeout=600)
    r = sh("cd %s && python3 %s/apply_fork.py %s" % (FORK, wsl_path(FORK_DIR), FORK), timeout=600)
    rec["fork_apply_stdout"] = r.stdout.strip().split("\n")
    sh("cd %s && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release > cmake.log 2>&1 && "
       "make -j8 freight_con_ckpt freight_cut_ckpt freight_con > make.log 2>&1" % FORK, timeout=3600)
    log("fork binaries built")
    h = home()
    bins = {}
    for name, path in (("freight_con", "%s/build/freight_con" % FREIGHT), ("freight_cut", "%s/build/freight_cut" % FREIGHT),
                       ("hmetis_to_freight_stream", "%s/build/hmetis_to_freight_stream" % FREIGHT),
                       ("hmetis_to_freight", "%s/build/hmetis_to_freight" % FREIGHT),
                       ("freight_con_ckpt", "%s/build/freight_con_ckpt" % FORK), ("freight_cut_ckpt", "%s/build/freight_cut_ckpt" % FORK),
                       ("fork_freight_con_unpatched_driver", "%s/build/freight_con" % FORK)):
        o = sh("sha256sum %s && stat -c %%s %s" % (path, path), quiet=True).stdout.split()
        bins[name] = {"path": path.replace("~", h), "sha256": o[0], "bytes": int(o[-1])}
    rec["binaries"] = bins
    rec["fork"] = {"applied": json.loads(sh("cat %s/FORK_APPLIED.json" % FORK, quiet=True).stdout),
                   "sources": {fn: pin(os.path.join(FORK_DIR, fn)) for fn in ("freight_ckpt.cpp", "ckpt.h", "apply_fork.py")},
                   "diff_stat": sh("cd %s && diff -rq --exclude=build --exclude=FORK_APPLIED.json %s/code_for_hypergraphs %s/code_for_hypergraphs; "
                                   "diff -q %s/CMakeLists.txt %s/CMakeLists.txt; true" % (FORK, FREIGHT, FORK, FREIGHT, FORK), quiet=True).stdout.strip().split("\n")}
    rec["seconds"] = round(time.time() - t, 1)
    wj(os.path.join(OUT, "FREIGHT_BUILD.json"), rec)
    log("wrote results/L1_LOWMEM/FREIGHT_BUILD.json (%.0fs)" % rec["seconds"])
    return rec


def build_record():
    b = rj(os.path.join(OUT, "FREIGHT_BUILD.json"))
    if b is None:
        raise RuntimeError("run `freight.py build` first")
    return b


# ----------------------------------------------------------------------------- /usr/bin/time -v parsing
def parse_time_v(path):
    out = {}
    if not os.path.exists(path):
        return out
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        m = re.match(r"Maximum resident set size \(kbytes\): (\d+)", ln)
        if m:
            out["peak_rss_kb"] = int(m.group(1))
        m = re.match(r"Elapsed \(wall clock\) time .*: (.+)$", ln)
        if m:
            parts = [float(x) for x in m.group(1).split(":")]
            out["wall_seconds"] = round(sum(v * 60 ** (len(parts) - 1 - i) for i, v in enumerate(parts)), 3)
        m = re.match(r"Exit status: (\d+)", ln)
        if m:
            out["exit_status"] = int(m.group(1))
        m = re.match(r"User time \(seconds\): ([\d.]+)", ln)
        if m:
            out["user_seconds"] = float(m.group(1))
        if "Command terminated by signal" in ln:
            out["signal"] = int(ln.split()[-1])
    return out


def parse_freight_stdout(path):
    out = {"checkpoints": []}
    if not os.path.exists(path):
        return out
    pats = (("connectivity", r"^connectivity\s+([\d.eE+-]+)$", float), ("cut", r"^cut\s+([\d.eE+-]+)$", float), ("balance", r"^balance\s+([\d.eE+-]+)$", float),
            ("pin_count", r"^pin count:\s+(\d+)$", int), ("processing_seconds", r"^Total processing time: ([\d.eE+-]+)$", float),
            ("io_seconds", r"^io time: ([\d.eE+-]+)$", float), ("partition_sha256", r"^partition_sha256 ([0-9a-f]{64})$", str),
            ("processed_pins", r"^processed_pins (\d+)$", int), ("nodes", r"^Hypergraph has (\d+) nodes and \d+ nets$", int),
            ("nets", r"^Hypergraph has \d+ nodes and (\d+) nets$", int))
    for ln in io.open(path, encoding="utf-8", errors="replace"):
        s = ln.strip()
        for key, pat, conv in pats:
            m = re.match(pat, s)
            if m:
                out[key] = conv(m.group(1))
        m = re.match(r"^checkpoint (checkpoint_\d+_\d+\.bin) ([0-9a-f]{64}) \(next_node (\d+), next_shard (\d+)\)$", s)
        if m:
            out["checkpoints"].append({"name": m.group(1), "sha256": m.group(2), "next_node": int(m.group(3)), "next_shard": int(m.group(4))})
        if s.startswith("resuming from"):
            out["resumed_from"] = s
        if s.startswith("ABORT (crash test)"):
            out["abort_line"] = s
    return out


# ----------------------------------------------------------------------------- convert (official converter) + bytes gate
def convert(ds):
    b = build_record()
    rec = HG.load(ds)["meta"]
    hrec = rj(os.path.join(ds_dir(ds), "H4_SK.hgr.json"))
    hgr = os.path.join(ds_dir(ds), "H4_SK.hgr")
    if hrec is None or hrec["hgr"]["sha256"] != sha_file(hgr):
        raise RuntimeError("%s: H4_SK.hgr missing or changed -- run hgr.py first" % ds)
    out = os.path.join(ds_dir(ds), "H4_SK.official.netl")
    tpath = out + ".time.txt"
    conv = b["binaries"]["hmetis_to_freight_stream"]["path"]
    t = time.time()
    r = sh("/usr/bin/time -v -o %s %s %s %s.tmp && mv %s.tmp %s" % (wsl_path(tpath), conv, wsl_path(hgr), wsl_path(out), wsl_path(out), wsl_path(out)),
           timeout=7200, quiet=True)
    tv = parse_time_v(tpath)
    R = {"RECORD": "H4_SK_OFFICIAL_NETLIST", "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "converter": {"binary": conv, "sha256": b["binaries"]["hmetis_to_freight_stream"]["sha256"], "upstream_commit": b["upstream"]["commit"],
                       "note": "official two-pass streaming converter; allocates net_ids(total_pins) -> memory O(N+M+pins), not O(N+M) as its banner says"},
         "input_hgr": hrec["hgr"], "ORIGINAL_STRUCTURE_SHA256": hrec["ORIGINAL_STRUCTURE_SHA256"],
         "output": pin(out), "time": tv, "stderr": r.stderr.strip().split("\n")[-3:], "seconds": round(time.time() - t, 1)}
    wj(out + ".json", R)
    log("%-8s official net-list %s  %.1f MB  peak RSS %.0f MB  %.1fs" % (ds, R["output"]["sha256"][:16], R["output"]["bytes"] / 1e6,
                                                                       tv.get("peak_rss_kb", 0) / 1024, tv.get("wall_seconds", 0)))
    g = ST.gate(ds, out)
    R["reconstruction_gate"] = {"gate_digest": g["gate_digest"], "gate_bytes": g["gate_bytes"]}
    wj(out + ".json", R)
    if g["gate_bytes"] != "PASS":
        raise RuntimeError("%s: shard bytes != official net-list bytes (%s) -- STOP" % (ds, g["gate_bytes"]))
    sg = semantic_gate(ds, out)
    R["semantic_gate"] = {k: sg[k] for k in ("gate_semantic", "official_vs_canonical_npz", "official_vs_shards", "STRUCTURE_SHA256_official", "STRUCTURE_SHA256_shards")}
    wj(out + ".json", R)
    if sg["gate_semantic"] != "PASS":
        raise RuntimeError("%s: SEMANTIC MISMATCH between the official net-list, the shards and the canonical H4_SK -- HARD STOP" % ds)
    return R


def parse_netl(path_or_lines, header=None):
    """independent net-list parser (does not share code with stream.py / hgr.py digests).
    header 'N M fmt' (fmt 1 = net weights, unit node weights); one line per node in canonical position order:
    'net w net w ...' with 1-based net ids.  Returns per-pin arrays in FILE ORDER plus a CSR (net -> sorted pins)."""
    if isinstance(path_or_lines, str):
        fh = io.open(path_or_lines, encoding="ascii", newline="\n")
        hdr = fh.readline().split()
        lines = fh
    else:
        hdr = header.split()
        lines = path_or_lines
    N, M = int(hdr[0]), int(hdr[1])
    fmt = int(hdr[2]) if len(hdr) > 2 else 0
    if fmt != 1:
        raise RuntimeError("net-list fmt %d: expected 1 (net weights, unit node weights)" % fmt)
    node_of, nets, ws, n_lines = [], [], [], 0
    for ln in lines:
        if ln.startswith("%"):
            continue
        a = np.array(ln.split(), dtype=np.int64)
        if len(a) % 2:
            raise RuntimeError("odd token count on node line %d" % n_lines)
        node_of.append(np.full(len(a) // 2, n_lines, dtype=np.int64))
        nets.append(a[0::2] - 1)     # -> 0-based net ids
        ws.append(a[1::2])
        n_lines += 1
    if isinstance(path_or_lines, str):
        fh.close()
    if n_lines != N:
        raise RuntimeError("net-list has %d node lines, header says %d" % (n_lines, N))
    node_of, nets, ws = np.concatenate(node_of), np.concatenate(nets), np.concatenate(ws)
    P = int(len(nets))
    if P and (nets.min() < 0 or nets.max() >= M):
        raise RuntimeError("net id out of range")
    # weights: every occurrence of a net must carry the same weight
    order = np.argsort(nets * np.int64(N) + node_of, kind="stable")        # (net, pin) ascending
    sn, sp, sw = nets[order], node_of[order], ws[order]
    sizes = np.bincount(sn, minlength=M)
    if (sizes == 0).any():
        raise RuntimeError("%d nets have no pins" % int((sizes == 0).sum()))
    eptr = np.zeros(M + 1, dtype=np.int64); eptr[1:] = np.cumsum(sizes)
    same = np.diff(sn) == 0
    if (sw[1:][same] != sw[:-1][same]).any():
        raise RuntimeError("inconsistent weights for one net across its pins")
    if (np.diff(sn * np.int64(N) + sp) == 0).any():
        raise RuntimeError("duplicate pin inside one net")
    ew = sw[eptr[:-1]]
    return {"N": N, "M": M, "P": P, "fmt": fmt, "node_weights": "unit (fmt 1: none written)", "file_order": (node_of, nets, ws),
            "eptr": eptr, "eidx": sp, "ew": ew}


def semantic_gate(ds, official_netl):
    """spec step 4: independently parse the official net-list AND the shards and require exact semantic equality
    (N, M, pins, node weights, hyperedge weights, per-node and global memberships, structure digest) with each other
    and with the canonical H4_SK npz.  Any mismatch = HARD STOP."""
    t = time.time()
    H = HG.load(ds)
    man = rj(ST.manifest_path(ds))
    O = parse_netl(official_netl)
    S = parse_netl(ST.iter_shard_lines(ds, man), header=man["header"])
    # canonical npz -> CSR with pins sorted inside each hyperedge
    eptr, eidx, ew = H["eptr"].astype(np.int64), H["eidx"].astype(np.int64), H["ew"].astype(np.int64)
    Mc = len(eptr) - 1
    hid = np.repeat(np.arange(Mc, dtype=np.int64), np.diff(eptr))
    ce = eidx[np.argsort(hid * np.int64(H["N"]) + eidx, kind="stable")]
    dO, cO = HG.structure_digest(O["eptr"], O["eidx"], O["ew"], O["N"])
    dS, cS = HG.structure_digest(S["eptr"], S["eidx"], S["ew"], S["N"])
    orig = rj(os.path.join(ds_dir(ds), "H4_SK.hgr.json"))["ORIGINAL_STRUCTURE_SHA256"]
    ovc = {"N": O["N"] == H["N"], "M": O["M"] == Mc, "P": O["P"] == int(len(eidx)), "node_weights": "unit == unit (canonical vertex weights are unit)",
           "hyperedge_weights": bool(np.array_equal(O["ew"], ew)), "global_net_pins(CSR)": bool(np.array_equal(O["eptr"], eptr) and np.array_equal(O["eidx"], ce)),
           "structure_digest": dO == orig}
    ovs = {"N": O["N"] == S["N"], "M": O["M"] == S["M"], "P": O["P"] == S["P"], "node_weights": O["node_weights"] == S["node_weights"],
           "hyperedge_weights": bool(np.array_equal(O["ew"], S["ew"])),
           "per_node_memberships_in_file_order": bool(all(np.array_equal(a, b) for a, b in zip(O["file_order"], S["file_order"]))),
           "global_net_pins(CSR)": bool(np.array_equal(O["eptr"], S["eptr"]) and np.array_equal(O["eidx"], S["eidx"])),
           "structure_digest": dO == dS}
    ok = all(v is True for k, v in ovc.items() if k != "node_weights") and all(v is True for v in ovs.values())
    res = {"RECORD": "H4_SK_SEMANTIC_GATE", "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "official_netl": pin(official_netl), "STREAM_MANIFEST_SHA256": man["STREAM_MANIFEST_SHA256"], "ORIGINAL_STRUCTURE_SHA256": orig,
           "STRUCTURE_SHA256_official": dO, "STRUCTURE_SHA256_shards": dS, "counts_official": cO, "counts_shards": cS,
           "official_vs_canonical_npz": ovc, "official_vs_shards": ovs, "parser": "freight.parse_netl (independent of stream.py / hgr.py digest code)",
           "gate_semantic": "PASS" if ok else "FAIL", "seconds": round(time.time() - t, 1)}
    wj(os.path.join(ds_dir(ds), "H4_SK.semantic_gate.json"), res)
    log("%-8s semantic gate %s  (official==npz %s, official==shards %s, digest %s)  %.1fs" % (
        ds, res["gate_semantic"], all(v is True for k, v in ovc.items() if k != "node_weights"), all(ovs.values()), dO[:16], res["seconds"]))
    return res


# ----------------------------------------------------------------------------- runs
def load_inputs(ds):
    H = HG.load(ds)
    man = rj(ST.manifest_path(ds))
    if man is None or not man.get("COMPLETE"):
        raise RuntimeError("%s: no complete H4_SK_STREAM_V1 manifest" % ds)
    gate = rj(os.path.join(ST.stream_dir(ds), "RECONSTRUCTION_GATE.json"))
    if gate is None or gate["gate_digest"] != "PASS" or gate["gate_bytes"] != "PASS":
        raise gate_error(ds, gate)
    for c in man["shards"]:
        if sha_file(os.path.join(ST.stream_dir(ds), c["file"])) != c["sha256"]:
            raise RuntimeError("%s: shard %d changed since its record" % (ds, c["index"]))
    off = rj(os.path.join(ds_dir(ds), "H4_SK.official.netl.json"))
    netl = os.path.join(ds_dir(ds), "H4_SK.official.netl")
    if off is None or off["output"]["sha256"] != sha_file(netl):
        raise RuntimeError("%s: official net-list missing or changed -- run `freight.py convert %s`" % (ds, ds))
    sg = rj(os.path.join(ds_dir(ds), "H4_SK.semantic_gate.json"))
    if sg is None or sg["gate_semantic"] != "PASS" or sg["official_netl"]["sha256"] != off["output"]["sha256"]:
        raise RuntimeError("%s: semantic gate missing / FAIL / stale -- HARD STOP" % ds)
    return H, man, gate, off, netl


def gate_error(ds, gate):
    return RuntimeError("%s: reconstruction gate not PASS/PASS: %s" % (ds, None if gate is None else (gate["gate_digest"], gate["gate_bytes"])))


def write_stream_manifest(ds, man):
    """the fork's manifest: line 1 'N M 1', then '<nodes> <wsl path>' per shard in canonical order."""
    p = os.path.join(ds_dir(ds), "freight", "stream_manifest.txt")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    lines = [man["header"]] + ["%d %s" % (c["nodes"], wsl_path(os.path.join(ST.stream_dir(ds), c["file"]))) for c in man["shards"]]
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return p


def run_dir(ds, name):
    d = os.path.join(ds_dir(ds), "freight", name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def base_args(k, passes, part_out):
    return "--k=%d --imbalance=%d --seed=%d --num_streams_passes=%d --output_filename=%s" % (k, IMBALANCE, SEED, passes, wsl_path(part_out))


def meta_args(meta):
    return " ".join("--ckpt_meta=%s=%s" % (k, v) for k, v in meta.items())


def exec_run(name, cmdline, d, timeout=7200, expect_rc=0):
    """run one binary invocation under /usr/bin/time -v; stdout/stderr/time captured in d."""
    tpath, so, se = os.path.join(d, "time.txt"), os.path.join(d, "stdout.txt"), os.path.join(d, "stderr.txt")
    cmd = "/usr/bin/time -v -o %s %s > %s 2> %s" % (wsl_path(tpath), cmdline, wsl_path(so), wsl_path(se))
    with io.open(os.path.join(d, "command.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(cmd + "\n")
    t = time.time()
    r = wsl(cmd, timeout=timeout)
    tv = parse_time_v(tpath)
    fo = parse_freight_stdout(so)
    res = {"name": name, "command": cmd, "rc": r.returncode, "time": tv, "stdout": fo, "wall_seconds_outer": round(time.time() - t, 2),
           "stderr_tail": io.open(se, encoding="utf-8", errors="replace").read().strip().split("\n")[-5:] if os.path.exists(se) else []}
    log("  %-9s rc=%d  wall %.2fs  peak RSS %.1f MB  con %s  cut %s  bal %s" % (name, r.returncode, tv.get("wall_seconds", -1), tv.get("peak_rss_kb", 0) / 1024,
                                                                              fo.get("connectivity"), fo.get("cut"), fo.get("balance")))
    if expect_rc is not None and r.returncode != expect_rc:
        raise RuntimeError("%s: rc %d (expected %d): %s" % (name, r.returncode, expect_rc, res["stderr_tail"]))
    return res


def read_partition(path, N, k):
    a = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if len(a) != N:
        raise RuntimeError("%s: %d lines, N=%d" % (path, len(a), N))
    if a.min() < 0 or a.max() >= k:
        raise RuntimeError("%s: block ids outside [0,k)" % path)
    return a


def km1_metrics(H, hard):
    """authoritative objective recomputation (FREIGHT's own connectivity evaluator silently skips beyond a 256 MiB bitset)."""
    eptr, eidx, ew, N, k, M = H["eptr"], H["eidx"], H["ew"], H["N"], H["k"], H["M"]
    hid = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    key = np.unique(hid * np.int64(k) + hard[eidx])
    lam = np.bincount(key // np.int64(k), minlength=M).astype(np.int64)
    cut = lam > 1
    sizes = np.bincount(hard, minlength=k).astype(np.int64)
    ceil_nk = int(math.ceil(N / float(k)))
    fams = H["meta"].get("per_family") or {}
    return {"km1_weighted": int((ew * (lam - 1)).sum()), "cut_weighted": int(ew[cut].sum()), "cut_nets": int(cut.sum()), "nets": int(M),
            "cut_net_fraction": round(float(cut.mean()), 6), "lambda_mean": round(float(lam.mean()), 4), "lambda_max": int(lam.max()),
            "blocks": {"k": int(k), "used": int((sizes > 0).sum()), "empty": int((sizes == 0).sum()), "min": int(sizes.min()),
                       "p50": float(np.median(sizes)), "p95": float(np.percentile(sizes, 95)), "max": int(sizes.max()),
                       "mean": round(float(sizes.mean()), 3), "ceil_N_over_k": ceil_nk, "max_over_mean": round(float(sizes.max() / sizes.mean()), 4),
                       "eps_actual(max/ceil(N/k)-1)": round(float(sizes.max()) / ceil_nk - 1, 4),
                       "freight_upper_bound": int(math.ceil((100 + IMBALANCE) / 100.0 * N / float(k))),
                       "within_eps_0.03": bool(sizes.max() <= math.ceil(1.03 * N / float(k)))},
            "per_family_note": fams}


def family_cuts(ds, hard):
    from src.l1_canonical import l1_eval as LE
    from src.l1_canonical.adapter import CanonicalDataset
    return LE.cut_stats(CanonicalDataset(ds), hard)


def ckpt_listing(ckdir):
    o = sh("cd %s 2>/dev/null && ls -la --time-style=+%%s | tail -n +2; echo ---; cat LATEST 2>/dev/null; true" % ckdir, quiet=True).stdout
    files, latest = [], None
    part = o.split("---")
    for ln in part[0].strip().split("\n"):
        f = ln.split()
        if len(f) >= 7 and f[-1] not in (".", ".."):
            files.append({"name": f[-1], "bytes": int(f[4])})
    if len(part) > 1 and part[1].strip():
        latest = part[1].strip()
    return {"files": files, "LATEST": latest}


def run(ds):
    b = build_record()
    H, man, gate, off, netl = load_inputs(ds)
    N, k, M, P = H["N"], H["k"], H["M"], H["P"]
    stock, fork = b["binaries"]["freight_con"], b["binaries"]["freight_con_ckpt"]
    smf = write_stream_manifest(ds, man)
    smf_sha = sha_file(smf)
    h = home()
    ckhome = "%s/%s" % (CKPT_HOME.replace("~", h), ds)      # absolute: the binary receives the path verbatim (no shell tilde expansion inside --ckpt_dir=)
    sh("rm -rf %s && mkdir -p %s" % (ckhome, ckhome), quiet=True)
    meta = {"dataset": ds, "binary_sha256": fork["sha256"], "stream_manifest_sha256": smf_sha, "structure_sha256": man["ORIGINAL_STRUCTURE_SHA256"],
            "upstream_commit": b["upstream"]["commit"]}
    shard_first = [c["first_position"] for c in man["shards"]]
    shard_last = [c["last_position"] for c in man["shards"]]
    S = len(man["shards"])
    T = S // 2                                  # crash target: LATEST = checkpoint_0_<T> (shards 0..T-1 done)
    abort_node = shard_first[T] + man["shard_nodes"] // 2
    log("=== FREIGHT %s: N %d M %d P %d k %d shards %d (crash target shard %d, abort node %d) ===" % (ds, N, M, P, k, S, T, abort_node))
    R = {"RECORD": "FREIGHT_RUNS", "dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "inputs": {"N": N, "M": M, "P": P, "k": k, "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"], "reconstruction_gate": gate,
                    "official_netl": off["output"], "stream_manifest": {"path": os.path.relpath(smf, REPO).replace("\\", "/"), "sha256": smf_sha,
                                                                        "H4_SK_STREAM_V1_manifest_sha256": man["STREAM_MANIFEST_SHA256"], "shards": S,
                                                                        "shard_nodes": man["shard_nodes"]},
                    "input_bytes": {"official_netl": off["output"]["bytes"], "shards_total": int(sum(c["bytes"] for c in man["shards"]))}},
         "contract": {"objective": "connectivity (weighted KM1)", "imbalance_percent": IMBALANCE, "seed": SEED, "num_streams_passes": 1, "k": k,
                      "binaries": {"stock": stock, "fork": fork}, "upstream_commit": b["upstream"]["commit"]},
         "runs": {}, "crash": {"target_shard": T, "abort_after_node": abort_node}}
    parts = {}
    # ---- A: stock, monolithic
    d = run_dir(ds, "A"); pA = os.path.join(d, "partition.txt")
    R["runs"]["A"] = exec_run("A", "%s %s %s" % (stock["path"], wsl_path(netl), base_args(k, 1, pA)), d)
    parts["A"] = read_partition(pA, N, k)
    # ---- A_forklib: the unpatched stock driver compiled against the patched library (hooks NULL) -> the library patches are inert
    d = run_dir(ds, "A_forklib"); pAf = os.path.join(d, "partition.txt")
    R["runs"]["A_forklib"] = exec_run("A_forklib", "%s %s %s" % (b["binaries"]["fork_freight_con_unpatched_driver"]["path"], wsl_path(netl), base_args(k, 1, pAf)), d)
    parts["A_forklib"] = read_partition(pAf, N, k)
    # ---- B: fork, sharded stream, one persistent process, NO checkpoints (--ckpt_every_shards=0)
    d = run_dir(ds, "B"); pB = os.path.join(d, "partition.txt"); ck = "%s/B" % ckhome
    sh("mkdir -p %s" % ck, quiet=True)
    R["runs"]["B"] = exec_run("B", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=0 %s" % (
        fork["path"], wsl_path(smf), base_args(k, 1, pB), ck, meta_args(meta)), d)
    R["runs"]["B"]["arm"] = "sharded stream (H4_SK_STREAM_V1 shards), same persistent process/state, no checkpoints written"
    R["runs"]["B"]["ckpt_dir"] = ckpt_listing(ck)
    parts["B"] = read_partition(pB, N, k)
    # ---- B0: fork, sharded stream, checkpoint written at EVERY shard boundary, never interrupted
    d = run_dir(ds, "B0"); pB0 = os.path.join(d, "partition.txt"); ck0 = "%s/B0" % ckhome
    sh("mkdir -p %s" % ck0, quiet=True)
    R["runs"]["B0"] = exec_run("B0", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 %s" % (
        fork["path"], wsl_path(smf), base_args(k, 1, pB0), ck0, meta_args(meta)), d)
    R["runs"]["B0"]["arm"] = "sharded stream, checkpoint after every shard (CHECKPOINT_EVERY_SHARDS=1), uninterrupted"
    R["runs"]["B0"]["ckpt_dir"] = ckpt_listing(ck0)
    parts["B0"] = read_partition(pB0, N, k)
    # ---- C_kill: external SIGKILL once LATEST reaches shard T, then resume
    d = run_dir(ds, "C_kill"); pCk = os.path.join(d, "partition.txt"); ckk = "%s/C_kill" % ckhome
    watcher = ("rm -rf {ck} && mkdir -p {ck}; ( exec {bin} {smf} {args} --ckpt_manifest --ckpt_dir={ck} --ckpt_every_shards=1 {meta} "
               "> {so} 2> {se} ) & WP=$!; killed=0; seen=; "
               "while kill -0 $WP 2>/dev/null; do if [ -f {ck}/LATEST ]; then name=$(cut -d' ' -f1 {ck}/LATEST); idx=${{name#checkpoint_*_}}; "
               "idx=${{idx%.bin}}; idx=$((10#$idx)); seen=$idx; if [ $idx -ge {T} ]; then kill -9 $WP 2>/dev/null && killed=1; break; fi; fi; "
               "sleep 0.005; done; wait $WP; echo wrapper_rc=$? killed=$killed last_seen=$seen").format(
        ck=ckk, bin=fork["path"], smf=wsl_path(smf), args=base_args(k, 1, pCk), meta=meta_args(meta), so=wsl_path(os.path.join(d, "stdout_killed.txt")),
        se=wsl_path(os.path.join(d, "stderr_killed.txt")), T=T)
    r = wsl(watcher, timeout=7200)
    kl = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else ""
    m = re.match(r"wrapper_rc=(\d+) killed=(\d) last_seen=(\d*)", kl)
    R["runs"]["C_kill"] = {"name": "C_kill", "watcher_result": kl, "killed": bool(m and m.group(2) == "1"),
                           "killed_stdout": parse_freight_stdout(os.path.join(d, "stdout_killed.txt")), "ckpt_dir_after_kill": ckpt_listing(ckk)}
    if R["runs"]["C_kill"]["killed"]:
        R["runs"]["C_kill"]["resume"] = exec_run("C_kill", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 --ckpt_resume %s" % (
            fork["path"], wsl_path(smf), base_args(k, 1, pCk), ckk, meta_args(meta)), d)
        R["runs"]["C_kill"]["ckpt_dir_after_resume"] = ckpt_listing(ckk)
        parts["C_kill"] = read_partition(pCk, N, k)
    else:
        R["runs"]["C_kill"]["status"] = "KILL_MISSED (process finished before LATEST reached shard %d; the deterministic C_abort covers mid-shard death)" % T
        log("  C_kill    KILL MISSED (%s)" % kl)
    # ---- C_abort: deterministic mid-shard _exit(137), then resume
    d = run_dir(ds, "C_abort"); pCa = os.path.join(d, "partition.txt"); cka = "%s/C_abort" % ckhome
    sh("rm -rf %s && mkdir -p %s" % (cka, cka), quiet=True)
    R["runs"]["C_abort"] = {"name": "C_abort"}
    da = os.path.join(d, "aborted"); os.makedirs(da)
    R["runs"]["C_abort"]["aborted"] = exec_run("C_abort~", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 --ckpt_abort_after_node=%d %s" % (
        fork["path"], wsl_path(smf), base_args(k, 1, pCa), cka, abort_node, meta_args(meta)), da, expect_rc=137)
    R["runs"]["C_abort"]["ckpt_dir_after_abort"] = ckpt_listing(cka)
    R["runs"]["C_abort"]["partition_written_by_aborted_run"] = os.path.exists(pCa)
    R["runs"]["C_abort"]["resume"] = exec_run("C_abort", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 --ckpt_resume %s" % (
        fork["path"], wsl_path(smf), base_args(k, 1, pCa), cka, meta_args(meta)), d)
    R["runs"]["C_abort"]["ckpt_dir_after_resume"] = ckpt_listing(cka)
    parts["C_abort"] = read_partition(pCa, N, k)
    # ---- refusal tests on copies of the C_abort checkpoint directory (state as left by the crash)
    R["refusals"] = refusal_tests(ds, fork, b["binaries"]["freight_cut_ckpt"], smf, k, cka, meta)
    # ---- exact equality (position by position) + exact metric equality
    E = {"reference": "A", "compared": {}, "ALL_IDENTICAL": True}
    mA = km1_metrics(H, parts["A"])
    for nm, arr in parts.items():
        same = bool(np.array_equal(arr, parts["A"]))
        mm = mA if nm == "A" else km1_metrics(H, arr)
        E["compared"][nm] = {"identical_to_A": same, "first_diff_position": None if same else int(np.nonzero(arr != parts["A"])[0][0]),
                             "n_diff": 0 if same else int((arr != parts["A"]).sum()),
                             "km1_weighted": mm["km1_weighted"], "cut_weighted": mm["cut_weighted"], "block_max": mm["blocks"]["max"],
                             "block_weights_identical": bool(np.array_equal(np.bincount(arr, minlength=k), np.bincount(parts["A"], minlength=k))),
                             "partition_sha256_stdout": R["runs"][nm].get("stdout", R["runs"][nm].get("resume", {}).get("stdout", {})).get("partition_sha256")}
        E["ALL_IDENTICAL"] &= same
    fc = family_cuts(ds, parts["A"])
    for nm in parts:
        E["compared"][nm]["family_cuts"] = fc if np.array_equal(parts[nm], parts["A"]) else family_cuts(ds, parts[nm])
        mm = mA if nm == "A" else km1_metrics(H, parts[nm])
        c = E["compared"][nm]
        c["eps_actual"] = mm["blocks"]["eps_actual(max/ceil(N/k)-1)"]
        c["exact_metrics_equal_to_A"] = {"objective_km1": mm["km1_weighted"] == mA["km1_weighted"], "cut": mm["cut_weighted"] == mA["cut_weighted"],
                                         "block_weights": c["block_weights_identical"], "imbalance": c["eps_actual"] == mA["blocks"]["eps_actual(max/ceil(N/k)-1)"],
                                         "STRUCT_cut": c["family_cuts"]["STRUCT"]["edge_cut_fraction"] == fc["STRUCT"]["edge_cut_fraction"],
                                         "KNN_cut": c["family_cuts"]["KNN"]["edge_cut_fraction"] == fc["KNN"]["edge_cut_fraction"]}
        c["ALL_EXACT"] = bool(c["identical_to_A"] and all(c["exact_metrics_equal_to_A"].values()))
    E["A_eq_A_forklib"] = E["compared"]["A_forklib"]["identical_to_A"]     # fork library inert under the stock driver
    E["forklib_eq_B_B0_Ckill_Cabort"] = all(E["compared"][nm]["identical_to_A"] for nm in ("B", "B0", "C_abort") if nm in E["compared"]) and \
        (E["compared"]["C_kill"]["identical_to_A"] if "C_kill" in E["compared"] else True)
    if not E["A_eq_A_forklib"]:
        log("  STOP: pristine upstream and the fork library differ -> the instrumentation changed algorithmic behaviour")
    E["freight_reported_connectivity_A"] = R["runs"]["A"]["stdout"].get("connectivity")
    E["freight_connectivity_evaluator_skipped"] = bool(M * ((k + 63) // 64) * 8 > 256 * 1024 * 1024)
    # FREIGHT prints its metrics with std::cout default precision (6 significant digits, e.g. 'connectivity 1.8062e+07'); the exact
    # objective is therefore the Python integer recomputation, and the printed value can only be checked for 6-digit consistency.
    def sig6_consistent(exact, printed):
        if printed is None or printed <= 0:
            return False
        ulp = 10.0 ** (math.floor(math.log10(printed)) - 5)
        return bool(abs(float(exact) - float(printed)) <= 0.5 * ulp + 1e-9)
    E["freight_printed_precision"] = "6 significant digits (std::cout default)"
    E["python_km1_consistent_with_freight_printed"] = (None if E["freight_connectivity_evaluator_skipped"] else
                                                       sig6_consistent(mA["km1_weighted"], E["freight_reported_connectivity_A"]))
    E["python_cut_consistent_with_freight_printed"] = sig6_consistent(mA["cut_weighted"], R["runs"]["A"]["stdout"].get("cut"))
    E["python_km1_equals_freight_connectivity"] = E["python_km1_consistent_with_freight_printed"]     # kept for the report reader
    R["EQUIVALENCE"] = E
    R["metrics_B"] = km1_metrics(H, parts["B"]); R["metrics_B"]["family_cuts"] = E["compared"]["B"]["family_cuts"]
    R["VERDICT_EXPERIMENT_1"] = ("EXACT_RESTART_EQUIVALENCE" if E["ALL_IDENTICAL"] and "C_abort" in parts and R["refusals"]["ALL_REFUSED"]
                                 else "CHECKPOINT_STATE_INCOMPLETE" if not E["ALL_IDENTICAL"] else "REFUSAL_TESTS_FAILED")
    # ---- baseline (canonical Mt-KaHyPar H4_SK) metrics on the identical hypergraph, when it exists
    R["baseline_mtkahypar"] = baseline_metrics(ds, H)
    # ---- 2-pass smoke test (multi-pass checkpoint handling), smallest dataset only
    if ds == "squad":
        R["two_pass_smoke"] = two_pass_smoke(ds, stock, fork, netl, smf, k, N, ckhome, meta, shard_first, man["shard_nodes"], S)
    # ---- import B as an EXPERIMENTAL partition in the canonical manifest shape
    if E["ALL_IDENTICAL"]:
        R["import"] = import_partition(ds, H, parts["B"], R, b, off, man)
    fp = os.path.join(OUT, "FREIGHT_RUNS_%s.json" % ds)
    wj(fp, R)
    log("%-8s EXPERIMENT 1: %s  (A==B %s, A==B0 %s, A==C_kill %s, A==C_abort %s; refusals %s)  km1 B %d  peak RSS A %.1f MB / B %.1f MB -> %s" % (
        ds, R["VERDICT_EXPERIMENT_1"], E["compared"]["B"]["identical_to_A"], E["compared"]["B0"]["identical_to_A"],
        E["compared"].get("C_kill", {}).get("identical_to_A", "n/a"), E["compared"].get("C_abort", {}).get("identical_to_A"), R["refusals"]["ALL_REFUSED"],
        R["metrics_B"]["km1_weighted"], R["runs"]["A"]["time"].get("peak_rss_kb", 0) / 1024, R["runs"]["B"]["time"].get("peak_rss_kb", 0) / 1024,
        os.path.relpath(fp, REPO)))
    return R


def refusal_tests(ds, fork, cut_fork, smf, k, cka, meta):
    """every test must exit non-zero, print REFUSING/unusable, and write no partition."""
    out = {"tests": {}, "ALL_REFUSED": True}
    tests = {
        "corrupt_checkpoint_byte": "cp -r {src} {dst} && name=$(cut -d' ' -f1 {dst}/LATEST) && printf '\\x5a\\xa5' | dd of={dst}/$name bs=1 seek=4000 conv=notrunc 2>/dev/null",
        "tmp_only_no_latest": "cp -r {src} {dst} && name=$(cut -d' ' -f1 {dst}/LATEST) && mv {dst}/$name {dst}/$name.tmp && rm -f {dst}/LATEST",
        "latest_hash_mismatch": "cp -r {src} {dst} && name=$(cut -d' ' -f1 {dst}/LATEST) && echo \"$name 0000000000000000000000000000000000000000000000000000000000000000\" > {dst}/LATEST",
        "truncated_checkpoint": "cp -r {src} {dst} && name=$(cut -d' ' -f1 {dst}/LATEST) && truncate -s -100 {dst}/$name",
        "foreign_meta_binary": "cp -r {src} {dst}",
        "wrong_k": "cp -r {src} {dst}",
        "wrong_structure_hash": "cp -r {src} {dst}",          # meta.structure_sha256 (H4_SK identity) differs
        "wrong_stream_manifest_hash": "cp -r {src} {dst}",    # meta.stream_manifest_sha256 differs
        "wrong_upstream_commit": "cp -r {src} {dst}",         # meta.upstream_commit (algorithm version) differs
        "wrong_objective_binary": "cp -r {src} {dst}",        # freight_cut_ckpt resumes a connectivity checkpoint (META objective)
        "wrong_imbalance": "cp -r {src} {dst}",
        "wrong_seed": "cp -r {src} {dst}",
        "wrong_num_passes": "cp -r {src} {dst}",
        "shard_manifest_shard_count": "cp -r {src} {dst}",    # a manifest with one shard fewer (META shard_count)
    }
    smf_short = smf + ".short"
    with io.open(smf, encoding="utf-8") as f, io.open(smf_short, "w", encoding="utf-8", newline="\n") as g:
        g.write("".join(f.readlines()[:-1]))
    for nm, prep in tests.items():
        d = run_dir(ds, "REFUSE_%s" % nm)
        dst = "%s_%s" % (cka, nm)
        sh("rm -rf %s; %s" % (dst, prep.format(src=cka, dst=dst)), quiet=True)
        pth = os.path.join(d, "partition.txt")
        m2 = dict(meta)
        if nm == "foreign_meta_binary":
            m2["binary_sha256"] = "deadbeef"
        elif nm == "wrong_structure_hash":
            m2["structure_sha256"] = "deadbeef"
        elif nm == "wrong_stream_manifest_hash":
            m2["stream_manifest_sha256"] = "deadbeef"
        elif nm == "wrong_upstream_commit":
            m2["upstream_commit"] = "deadbeef"
        kk = k + 1 if nm == "wrong_k" else k
        binp = cut_fork["path"] if nm == "wrong_objective_binary" else fork["path"]
        args = base_args(kk, 2 if nm == "wrong_num_passes" else 1, pth)
        if nm == "wrong_imbalance":
            args = args.replace("--imbalance=%d" % IMBALANCE, "--imbalance=%d" % (IMBALANCE + 1))
        if nm == "wrong_seed":
            args = args.replace("--seed=%d" % SEED, "--seed=%d" % (SEED + 1))
        manifest = smf_short if nm == "shard_manifest_shard_count" else smf
        res = exec_run("R:" + nm, "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 --ckpt_resume %s" % (
            binp, wsl_path(manifest), args, dst, meta_args(m2)), d, expect_rc=None)
        refused = res["rc"] != 0 and not os.path.exists(pth)
        out["tests"][nm] = {"rc": res["rc"], "partition_written": os.path.exists(pth), "stderr_tail": res["stderr_tail"], "REFUSED": refused}
        out["ALL_REFUSED"] &= refused
        log("  refusal %-24s rc=%d refused=%s  %s" % (nm, res["rc"], refused, (res["stderr_tail"] or [""])[-1][:90]))
    return out


def two_pass_smoke(ds, stock, fork, netl, smf, k, N, ckhome, meta, shard_first, shard_nodes, S):
    """--num_streams_passes=2: A2 (stock) == B2 (sharded, checkpoints) == C2 (abort in pass 1, resume)."""
    T = S // 2
    abort_node = shard_first[T] + shard_nodes // 2
    out = {"passes": 2, "abort_after_node_in_pass_1": abort_node}
    parts = {}
    d = run_dir(ds, "P2_A"); p = os.path.join(d, "partition.txt")
    out["A2"] = exec_run("A2", "%s %s %s" % (stock["path"], wsl_path(netl), base_args(k, 2, p)), d); parts["A2"] = read_partition(p, N, k)
    d = run_dir(ds, "P2_B"); p = os.path.join(d, "partition.txt"); ck = "%s/P2_B" % ckhome
    sh("rm -rf %s && mkdir -p %s" % (ck, ck), quiet=True)
    out["B2"] = exec_run("B2", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 %s" % (fork["path"], wsl_path(smf), base_args(k, 2, p), ck, meta_args(meta)), d)
    out["B2"]["ckpt_dir"] = ckpt_listing(ck); parts["B2"] = read_partition(p, N, k)
    d = run_dir(ds, "P2_C"); p = os.path.join(d, "partition.txt"); ck = "%s/P2_C" % ckhome
    sh("rm -rf %s && mkdir -p %s" % (ck, ck), quiet=True)
    da = os.path.join(d, "aborted"); os.makedirs(da)
    # the abort node index counts nodes of the CURRENT pass; abort in pass 1 = after pass 0 completed (checkpoints named checkpoint_1_*)
    out["C2_aborted"] = exec_run("C2~", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 --ckpt_abort_after_node=%d --ckpt_abort_in_pass=1 %s" % (
        fork["path"], wsl_path(smf), base_args(k, 2, p), ck, abort_node, meta_args(meta)), da, expect_rc=137)
    out["C2_ckpt_after_abort"] = ckpt_listing(ck)
    out["C2"] = exec_run("C2", "%s %s %s --ckpt_manifest --ckpt_dir=%s --ckpt_every_shards=1 --ckpt_resume %s" % (fork["path"], wsl_path(smf), base_args(k, 2, p), ck, meta_args(meta)), d)
    parts["C2"] = read_partition(p, N, k)
    out["A2_eq_B2"] = bool(np.array_equal(parts["A2"], parts["B2"])); out["A2_eq_C2"] = bool(np.array_equal(parts["A2"], parts["C2"]))
    out["resumed_in_pass"] = out["C2"]["stdout"].get("resumed_from")
    out["VERDICT"] = "EXACT" if out["A2_eq_B2"] and out["A2_eq_C2"] else "CHECKPOINT_STATE_INCOMPLETE_MULTIPASS"
    log("  2-pass smoke: A2==B2 %s  A2==C2 %s  (%s)" % (out["A2_eq_B2"], out["A2_eq_C2"], out["resumed_in_pass"]))
    return out


def baseline_metrics(ds, H):
    d = H["d"]
    pm = rj(os.path.join(d.derived_dir, "parts", "H4_SK.json"))
    if pm is None or pm.get("STATUS") != "OK":
        fl = rj(os.path.join(d.derived_dir, "parts", "H4_SK.FAILED.json"))
        return {"status": "FAILED" if fl else "ABSENT", "record": fl}
    hard = d.partition("H4_SK")
    m = km1_metrics(H, hard)
    m["family_cuts"] = family_cuts(ds, hard)
    m["manifest"] = {"worker_stats": pm.get("worker_stats"), "guard": pm.get("guard"), "post_checks": pm.get("post_checks"), "contract": pm.get("contract"),
                     "output_sha256": pm["output"]["sha256"]}
    m["python_km1_equals_mtkahypar_objective"] = bool(abs(float(m["km1_weighted"]) - float(pm["worker_stats"]["objective_km1"])) < 0.5)
    m["status"] = "OK"
    log("  baseline mtkahypar km1 %d (worker %s, match %s)  peak RSS %.1f MB  wall %.1fs" % (
        m["km1_weighted"], pm["worker_stats"]["objective_km1"], m["python_km1_equals_mtkahypar_objective"],
        pm["worker_stats"].get("peak_rss_mb", -1), pm["worker_stats"].get("wall_seconds", -1)))
    return m


def import_partition(ds, H, hard, R, b, off, man, tag=TAG):
    d = H["d"]
    pdir = os.path.join(d.derived_dir, "parts")
    os.makedirs(pdir, exist_ok=True)
    npy = os.path.join(pdir, "%s.npy" % tag)
    np.save(npy, hard.astype(np.int64))
    mB = R["metrics_B"]
    rec = {"dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "EXPERIMENTAL": True,
           "not_canonical": "L1_LOWMEM validation arm (FREIGHT); the canonical partition is parts/H4_SK.*; nothing canonical was modified",
           "contract": {"partitioner": "FREIGHT (KaHIP, official repository)", "upstream_commit": b["upstream"]["commit"],
                        "binary": R["contract"]["binaries"]["fork"], "algorithm": "one-pass streaming FREIGHT (fennel_approx_sqrt score, self-sorting block vector)",
                        # the canonical replay reader pins contract.worker_sha256 (partition.py pins its worker script); here the
                        # worker that produced the vector is the fork binary, so its sha256 is the pin
                        "worker": R["contract"]["binaries"]["fork"]["path"], "worker_sha256": R["contract"]["binaries"]["fork"]["sha256"],
                        "worker_note": "fork binary (freight_con_ckpt) driven by src/l1_lowmem/freight.py; A == A_forklib == B position-for-position",
                        "objective": "connectivity = weighted KM1 (freight_con, MODE_CONNECTIVITY)", "epsilon": 0.03, "imbalance_percent": IMBALANCE,
                        "seed": SEED, "num_streams_passes": R["contract"].get("num_streams_passes", 1), "restream_vcycle": R["contract"].get("restream_vcycle", False),
                        "vertex_weights": "unit", "hyperedge_weights": "H4 build (max(1, rint(1000/(|e|-1))))",
                        "k": H["k"], "k_rule": "max(1, N // 100)", "hypergraph_rule": "H4_SPLIT_PRESERVE", "families": H["meta"]["families"],
                        "node_stream_order": "canonical node-position order (H4_SK_STREAM_V1 shards)", "random_shuffle": False},
           "inputs": {"DATASET_json_RECORD_SHA256": d.record_sha, "hypergraph_file": os.path.relpath(H["npz"], REPO).replace("\\", "/"),
                      "hypergraph_sha256": H["npz_sha256"], "hypergraph_content_digest": H["meta"].get("content_digest"),
                      "ORIGINAL_STRUCTURE_SHA256": man["ORIGINAL_STRUCTURE_SHA256"], "official_netl_sha256": off["output"]["sha256"],
                      "stream_manifest_sha256": R["inputs"]["stream_manifest"]["sha256"], "hyperedges": H["M"], "pins": H["P"], "N": H["N"]},
           "STATUS": "OK",
           "worker_stats": {"N": H["N"], "k": H["k"], "hyperedges": H["M"], "pins": H["P"], "objective_km1": mB["km1_weighted"], "objective_cut": mB["cut_weighted"],
                            "peak_rss_mb": round(R["runs"]["B"]["time"].get("peak_rss_kb", 0) / 1024.0, 1), "wall_seconds": R["runs"]["B"]["time"].get("wall_seconds"),
                            "peak_rss_mb_stock_monolithic": round(R["runs"]["A"]["time"].get("peak_rss_kb", 0) / 1024.0, 1),
                            "wall_seconds_stock_monolithic": R["runs"]["A"]["time"].get("wall_seconds"), "ran_on": "local_wsl"},
           "post_checks": {"length_ok": True, "min_block": int(hard.min()), "max_block": int(hard.max()), "blocks_used": mB["blocks"]["used"], "k": H["k"],
                           "every_block_used": mB["blocks"]["empty"] == 0, "size_min": mB["blocks"]["min"], "size_max": mB["blocks"]["max"],
                           "size_mean": mB["blocks"]["mean"], "ceil_N_over_k": mB["blocks"]["ceil_N_over_k"], "balance_max_over_mean": mB["blocks"]["max_over_mean"],
                           "balance_within_eps": mB["blocks"]["within_eps_0.03"]},
           "EQUIVALENCE": {"experiment_1": R["VERDICT_EXPERIMENT_1"], "runs_identical_to_A": {k2: v["identical_to_A"] for k2, v in R["EQUIVALENCE"]["compared"].items()}},
           "output": {"file": os.path.relpath(npy, REPO).replace("\\", "/"), "bytes": os.path.getsize(npy), "sha256": sha_file(npy), "dtype": "int64", "n": int(len(hard))}}
    from src.l1_canonical.adapter import contract_hash
    rec["L1_CONTRACT_SHA256"] = contract_hash()["L1_CONTRACT_SHA256"]
    wj(os.path.join(pdir, "%s.json" % tag), rec)
    log("  imported parts/%s.npy (%s)" % (tag, rec["output"]["sha256"][:16]))
    return rec


def reimport(ds):
    """Regenerate parts/LOWMEM__FREIGHT_con.json from the recorded B partition without re-running FREIGHT (manifest shape fix:
    the canonical replay reader pins contract.worker_sha256).  The .npy must come out byte-identical to the recorded import."""
    b = build_record()
    H, man, gate, off, netl = load_inputs(ds)
    R = rj(os.path.join(OUT, "FREIGHT_RUNS_%s.json" % ds))
    if R is None or not R.get("import") or not R["EQUIVALENCE"]["ALL_IDENTICAL"]:
        raise RuntimeError("%s: no importable FREIGHT_RUNS record" % ds)
    hard = read_partition(os.path.join(ds_dir(ds), "freight", "B", "partition.txt"), H["N"], H["k"])
    npy = os.path.join(H["d"].derived_dir, "parts", "%s.npy" % TAG)
    if sha_file(npy) != R["import"]["output"]["sha256"] or not np.array_equal(np.load(npy), hard):
        raise RuntimeError("%s: the imported partition no longer matches the recorded B run -- refusing to reimport" % ds)
    before = R["import"]["output"]["sha256"]
    rec = import_partition(ds, H, hard, R, b, off, man)
    if rec["output"]["sha256"] != before:
        raise RuntimeError("%s: reimport changed the partition bytes (%s -> %s)" % (ds, before[:16], rec["output"]["sha256"][:16]))
    rec["reimport"] = {"utc": rec["utc"], "reason": "manifest shape: contract.worker / worker_sha256 added for the canonical replay reader; vector unchanged",
                       "previous_import_utc": R["import"]["utc"]}
    wj(os.path.join(H["d"].derived_dir, "parts", "%s.json" % TAG), rec)
    R["import"] = rec
    wj(os.path.join(OUT, "FREIGHT_RUNS_%s.json" % ds), R)
    log("%-8s reimported parts/%s.json (vector %s unchanged)" % (ds, TAG, before[:16]))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "build":
        build()
    elif a[0] == "convert":
        for ds in a[1:]:
            convert(ds)
    elif a[0] == "run":
        for ds in a[1:]:
            run(ds)
    elif a[0] == "reimport":
        for ds in a[1:]:
            reimport(ds)
    else:
        print(__doc__)
