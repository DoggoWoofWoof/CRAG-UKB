"""L1 HOST LANE -- shard maps built on the shared lab host (DESKTOP-SLQMEQH), declared in
results/L1_HOST/HOST_LANE_DECLARATION.json (authorized by the user in chat, 2026-09-29).

The recipes are the laptop's, with only the machine changed:
  HG   the frozen H4_SK rule at K: src/l1_canonical/hypergraph.build_hypergraph(N, {STRUCT, KNN}, K) from the dataset's own
       stamped key sets (cap = round(N / K), H4_SPLIT_PRESERVE), exactly as scratchpad/_l1d_scale_parts.py HG does.
       -> results/L1_HOST/parts/<ds>__H4_SK_k<K>.npz + .json
  PHG  the validated Zoltan-PHG substitute exactly as scratchpad/_l1c_phg.py runs it: the same shard writer (node-centric
       fmt-1 net lists, 5,000 nodes per shard), the same driver source and compile command, the same parameter list
       (src/l1_lowmem/phg.PARAMS), NP = 4 ranks, then PHG_NONEMPTY_REPAIR_V1 (src/l1_lowmem/phg_repair.repair) where a block
       is empty; a raw partition whose largest block exceeds ceil(1.03 N / k) is PARTITION_INVALID and no map is written.
       -> results/L1_HOST/parts/<ds>__H4_SK_k<K>__PHG_con.npy + .RUN.json (or .FAILED.json)
Only the placement differs: Python runs in the host's Windows interpreter (mpr's pinned env mpr-cpu, used read-only) and
mpicc / mpirun run in the host's WSL distro Ubuntu-24.04 (the laptop: Ubuntu-22.04). The lane modules are imported
unchanged; this module redirects their WSL distro and their run directory at runtime (common.DISTRO, common.DATA).
Every record names the machine, the interpreter and the driver build. Nothing under data/ is written.

Usage (on the host, through rx):
  python -u scratchpad/_l1h_host.py BUILD                 compile the driver in the host WSL -> results/L1_HOST/PHG_BUILD_HOST.json
  python -u scratchpad/_l1h_host.py HG   <ds> <K> [...]    hypergraphs (write-once)
  python -u scratchpad/_l1h_host.py PHG  <ds> <K> [...]    partitions (write-once; needs HG first)
  python -u scratchpad/_l1h_host.py CELL <ds> <K> [...]    HG then PHG, per K
"""
import hashlib
import inspect
import io
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from src.l1_canonical import adapter as AD  # noqa: E402
from src.l1_canonical import hypergraph as HG  # noqa: E402
from src.l1_lowmem import common as CM  # noqa: E402
from src.l1_lowmem import freight as FR  # noqa: E402
from src.l1_lowmem import phg as P  # noqa: E402
from src.l1_lowmem import phg_repair as PR  # noqa: E402

LANE = os.path.join(REPO, "results", "L1_HOST")
PDIR = os.path.join(LANE, "parts")
HOST_DISTRO = "Ubuntu-24.04"
BUILD_DIR_WSL = "~/phg_build_crag"
BUILD_REC = os.path.join(LANE, "PHG_BUILD_HOST.json")
LAPTOP_BUILD_REC = os.path.join(REPO, "results", "L1_LOWMEM", "PHG_BUILD.json")
SHARD_NODES = 5000                                   # == scratchpad/_l1c_phg.py SHARD_NODES
COMPILE = "mpicc -O2 -Wall -Wextra -I/usr/include/trilinos phg_driver.c -o phg_driver -ltrilinos_zoltan -lm"
PHG_TIMEOUT_S = 12 * 3600                            # harness timeout only (not a partitioner parameter)
T0 = time.time()

# redirect the lane modules (runtime only; their files are unchanged)
CM.DISTRO = HOST_DISTRO
FR.DISTRO = HOST_DISTRO
P.DISTRO = HOST_DISTRO
CM.DATA = os.path.join(REPO, "work", "L1_HOST", "phg_data")   # PHG shards, run dirs and dumps: host-only, outside the rx output glob
assert P.ds_dir is CM.ds_dir and P.run_dir.__globals__["ds_dir"] is CM.ds_dir and FR.wsl is CM.wsl and P.wsl is CM.wsl
assert COMPILE in inspect.getsource(P.build), "the compile command differs from src/l1_lowmem/phg.build"
assert P.NP == 4


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, REPO).replace("\\", "/")


def wj(p, obj):
    assert not os.path.exists(p), "write-once: %s exists" % p
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1))
    os.replace(tmp, p)


def tag_of(K):
    return "H4_SK_k%d" % int(K)


def hg_npz(ds, K):
    return os.path.join(PDIR, "%s__%s.npz" % (ds, tag_of(K)))


def phg_npy(ds, K):
    return os.path.join(PDIR, "%s__%s__PHG_con.npy" % (ds, tag_of(K)))


def placement():
    import numpy
    return {"machine": platform.node(), "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count_os": os.cpu_count(), "python": sys.version.split()[0], "executable": sys.executable, "prefix": sys.prefix,
            "numpy": numpy.__version__, "wsl_distro": HOST_DISTRO, "rx_job": os.environ.get("RX_JOB_ID"),
            "rx_cpus": os.environ.get("RX_CPUS"), "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}


def code_pins():
    fs = ["scratchpad/_l1h_host.py", "src/l1_canonical/hypergraph.py", "src/l1_canonical/adapter.py", "scratchpad/_l1hu_build.py",
          "src/l1_lowmem/common.py", "src/l1_lowmem/phg.py", "src/l1_lowmem/phg_repair.py", "src/l1_lowmem/freight.py"]
    return {f: sha_file(os.path.join(REPO, f)) for f in fs}


def stamped_keys(cd):
    """the dataset's stored key sets, read-only: the stamp must match (as scratchpad/_l1d_scale_parts.stamped_keys)."""
    z = np.load(cd._keys_path())
    assert json.loads(str(z["meta_json"])).get("DATASET_json_RECORD_SHA256") == cd.record_sha, "stale key sets: %s" % cd.name
    N, ST, KN, _ = cd.keysets()
    assert N == cd.n_nodes
    return N, {"STRUCT": ST, "KNN": KN}


def balance(hard, N, K):
    """== scratchpad/_l1d_scale_parts.balance"""
    s = np.bincount(hard, minlength=K)
    tgt = int(np.ceil(N / float(K)))
    return {"length_ok": bool(len(hard) == N), "ids_in_range": bool(hard.min() >= 0 and hard.max() < K), "blocks_used": int((s > 0).sum()),
            "every_block_used": bool((s > 0).all()), "size_min": int(s.min()), "size_max": int(s.max()), "size_mean": round(float(s.mean()), 3),
            "ceil_N_over_K": tgt, "within_ceil_1.03_N_over_K": bool(s.max() <= int(np.ceil(1.03 * N / float(K))))}


# ----------------------------------------------------------------------------- BUILD
def cmd_build():
    src = os.path.join(P.SRC_DIR, "phg_driver.c")
    wrap = os.path.join(P.SRC_DIR, "rank_wrap.sh")
    lap = json.load(io.open(LAPTOP_BUILD_REC, encoding="utf-8"))
    assert sha_file(src) == lap["source"]["sha256"] and sha_file(wrap) == lap["wrapper_source"]["sha256"], "driver sources differ from the laptop build"
    if os.path.exists(BUILD_REC):                       # write-once: never rebuilt over an existing record
        b = host_build()
        log("driver build unchanged (binary %s)" % b["binary"]["sha256"][:16])
        return b
    h = FR.home()
    bdir = BUILD_DIR_WSL.replace("~", h)
    FR.sh("mkdir -p %s && cp %s %s/phg_driver.c && cp %s %s/rank_wrap.sh && sed -i 's/\\r$//' %s/phg_driver.c %s/rank_wrap.sh && chmod +x %s/rank_wrap.sh" % (
        bdir, CM.wsl_path(src), bdir, CM.wsl_path(wrap), bdir, bdir, bdir, bdir), quiet=True)
    r = FR.sh("cd %s && %s 2>&1; echo rc=$?" % (bdir, COMPILE), quiet=True)
    if not r.stdout.strip().endswith("rc=0"):
        raise RuntimeError("driver build failed:\n%s" % r.stdout[-3000:])
    shas = FR.sh("cd %s && sha256sum phg_driver rank_wrap.sh phg_driver.c" % bdir, quiet=True).stdout.split()
    info = FR.sh("gcc --version | head -1; mpicc --version | head -1; mpirun --version | head -1; ldd %s/phg_driver | grep -E 'zoltan|mpi'; "
                 "dpkg-query -W libtrilinos-zoltan-13.2 libtrilinos-zoltan-dev libopenmpi-dev openmpi-bin gcc time; ls -l /usr/bin/time; "
                 "cat /etc/os-release | head -n 4; uname -srm; nproc --all; free -g | head -n 2" % bdir, quiet=True).stdout.strip().split("\n")
    rec = {"RECORD": "PHG_BUILD_HOST", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "distro": HOST_DISTRO, "build_dir": bdir,
           "compile": COMPILE, "compiler_output": r.stdout.strip().split("\n")[:-1],
           "source": {"path": rel(src), "sha256": sha_file(src)}, "wrapper_source": {"path": rel(wrap), "sha256": sha_file(wrap)},
           "binary": {"path": "%s/phg_driver" % bdir, "sha256": shas[0]}, "wrapper": {"path": "%s/rank_wrap.sh" % bdir, "sha256": shas[2]},
           "source_in_build_dir_sha256": shas[4], "toolchain_and_libs": info,
           "laptop_build": {"record": rel(LAPTOP_BUILD_REC), "record_sha256": sha_file(LAPTOP_BUILD_REC), "binary_sha256": lap["binary"]["sha256"],
                            "wrapper_sha256": lap["wrapper"]["sha256"], "toolchain_and_libs": lap["toolchain_and_libs"]},
           "binary_equals_laptop": shas[0] == lap["binary"]["sha256"], "wrapper_equals_laptop": shas[2] == lap["wrapper"]["sha256"],
           "placement": placement(), "code": code_pins()}
    wj(BUILD_REC, rec)
    log("driver built: binary %s (laptop %s, equal %s); wrapper equal %s" % (shas[0][:16], lap["binary"]["sha256"][:16], rec["binary_equals_laptop"],
                                                                            rec["wrapper_equals_laptop"]))
    for ln in info:
        log("  |", ln)
    return rec


def host_build():
    assert os.path.exists(BUILD_REC), "run BUILD first"
    b = json.load(io.open(BUILD_REC, encoding="utf-8"))
    chk = FR.sh("sha256sum %s %s" % (b["binary"]["path"], b["wrapper"]["path"]), quiet=True).stdout.split()
    assert chk and chk[0] == b["binary"]["sha256"] and chk[2] == b["wrapper"]["sha256"], "host PHG driver binary/wrapper changed since PHG_BUILD_HOST.json"
    t = FR.sh("test -x /usr/bin/time && echo TIME_OK || echo TIME_MISSING", quiet=True).stdout.strip()
    assert t == "TIME_OK", "the rank wrapper execs /usr/bin/time -v, which is missing in %s" % HOST_DISTRO
    return b


# ----------------------------------------------------------------------------- HG
def cmd_hg(ds, Ks):
    cd = AD.CanonicalDataset(ds)
    N, keys = None, None
    for K in Ks:
        K = int(K)
        fp = hg_npz(ds, K)
        if os.path.exists(fp[:-4] + ".json"):
            m = json.load(io.open(fp[:-4] + ".json", encoding="utf-8"))
            assert sha_file(fp) == m["file_sha256"], "%s changed since its record" % fp
            log("%s K %d: HG exists (%s)" % (ds, K, m["file_sha256"][:16]))
            continue
        if N is None:
            t = time.time()
            N, keys = stamped_keys(cd)
            log("%s: keys N %d STRUCT %d KNN %d (%.1fs)" % (ds, N, len(keys["STRUCT"]), len(keys["KNN"]), time.time() - t))
        kn = HG.frozen_k(N)
        assert 1 <= K <= N
        t = time.time()
        arrays, meta = HG.build_hypergraph(N, keys, K, tag=ds)
        dig = HG.arrays_digest(arrays)
        os.makedirs(PDIR, exist_ok=True)
        tmp = fp[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, fp)
        meta.update({"dataset": ds, "file": rel(fp), "bytes": os.path.getsize(fp), "file_sha256": sha_file(fp), "content_digest": dig,
                     "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t, 1),
                     "inputs": {"DATASET_json_RECORD_SHA256": cd.record_sha, "keys_npz_sha256": sha_file(cd._keys_path()),
                                "STRUCT": int(len(keys["STRUCT"])), "KNN": int(len(keys["KNN"]))},
                     "scale": {"K": K, "native_K": kn, "block_size_round_N_over_K": int(round(N / K)),
                               "rule": "the frozen H4_SK rule with k = K (cap = round(N / K)); nothing else changed (as _l1d_scale_parts.py HG)"},
                     "placement": placement(), "code": code_pins()})
        wj(fp[:-4] + ".json", meta)
        log("%s K %d: cap %s, %d hyperedges, %d pins (dup %d), %.1f MB, %.1fs, digest %s" % (
            ds, K, meta["cap"], meta["hyperedges"], meta["pins"], meta["anchor_duplication_pins"], meta["bytes"] / 1e6, meta["seconds"], dig[:16]))
        del arrays


# ----------------------------------------------------------------------------- PHG
def load_hg(ds, K):
    npz = hg_npz(ds, K)
    meta = json.load(io.open(npz[:-4] + ".json", encoding="utf-8"))
    assert sha_file(npz) == meta["file_sha256"]
    z = np.load(npz)
    eptr, eidx, ew = z["eptr"].astype(np.int64), z["eidx"].astype(np.int64), z["ew"].astype(np.int64)
    N, k = int(z["N"][0]), int(z["k"][0])
    return {"npz": npz, "npz_sha256": meta["file_sha256"], "meta": meta, "eptr": eptr, "eidx": eidx, "ew": ew, "N": N, "k": k,
            "M": int(len(eptr) - 1), "P": int(len(eidx))}


def write_shards(H, sdir):
    """== scratchpad/_l1c_phg.write_shards: node-centric net lists (fmt 1): per node '<net> <w> <net> <w> ...', nets 1-based ascending."""
    os.makedirs(sdir, exist_ok=True)
    N, M, eptr, eidx, ew = H["N"], H["M"], H["eptr"], H["eidx"], H["ew"]
    hid = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    order = np.lexsort((hid, eidx))                      # by node, then net id ascending
    node_of, net_of = eidx[order], hid[order]
    deg = np.bincount(node_of, minlength=N)
    off = np.zeros(N + 1, np.int64)
    off[1:] = np.cumsum(deg)
    shards = []
    for i, a in enumerate(range(0, N, SHARD_NODES)):
        b = min(a + SHARD_NODES, N)
        fp = os.path.join(sdir, "shard_%05d.netl" % i)
        with io.open(fp, "w", encoding="ascii", newline="\n") as f:
            buf = []
            for j in range(a, b):
                seg = net_of[off[j]:off[j + 1]]
                if len(seg):
                    pairs = np.empty(2 * len(seg), np.int64)
                    pairs[0::2] = seg + 1
                    pairs[1::2] = ew[seg]
                    buf.append(" ".join(map(str, pairs.tolist())) + "\n")
                else:
                    buf.append("\n")
            f.write("".join(buf))
        shards.append({"index": i, "file": fp, "nodes": b - a, "pins": int(off[b] - off[a]), "sha256": sha_file(fp)})
    smf = os.path.join(sdir, "stream_manifest.txt")
    with io.open(smf, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(["%d %d 1" % (N, M)] + ["%d %s" % (c["nodes"], CM.wsl_path(c["file"])) for c in shards]) + "\n")
    return shards, smf


def cmd_phg(ds, Ks):
    b = host_build()
    for K in Ks:
        K = int(K)
        out = phg_npy(ds, K)
        runp, failp = out[:-4] + ".RUN.json", out[:-4] + ".FAILED.json"
        if os.path.exists(runp) or os.path.exists(failp):
            log("%s K %d: PHG %s" % (ds, K, "exists" if os.path.exists(runp) else "FAILED earlier"))
            continue
        assert os.path.exists(hg_npz(ds, K)[:-4] + ".json"), "build the hypergraph first: HG %s %d" % (ds, K)
        t = time.time()
        tag = tag_of(K)
        H = load_hg(ds, K)
        N, k, M, Pn = H["N"], H["k"], H["M"], H["P"]
        assert k == K
        sdir = os.path.join(CM.DATA, ds, "stream_%s" % tag)
        shards, smf = write_shards(H, sdir)
        log("%s %s: N %d M %d P %d k %d -> %d shards (%.1fs)" % (ds, tag, N, M, Pn, k, len(shards), time.time() - t))
        rec = {"dataset": ds, "tag": tag, "K": K, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "hypergraph": {"file": rel(H["npz"]), "sha256": H["npz_sha256"], "content_digest": H["meta"]["content_digest"], "N": N, "M": M, "P": Pn, "k": k},
               "shards": [{kk: (rel(v) if kk == "file" else v) for kk, v in c.items()} for c in shards],
               "driver": {"build_record": rel(BUILD_REC), "build_record_sha256": sha_file(BUILD_REC), "binary": b["binary"], "wrapper": b["wrapper"],
                          "source_sha256": b["source"]["sha256"], "parameters": P.PARAMS, "NP": P.NP},
               "placement": placement(), "code": code_pins()}
        r1, d1 = P.mpirun(ds, "R1_%s" % tag, "partition", k, smf, b, timeout=PHG_TIMEOUT_S)
        gq = r1["global_from_queries"]
        assert gq["objects"] == N and gq["pins"] == Pn, ("Zoltan saw a different hypergraph", gq)
        hard = P.assemble_partition(d1, N, k)
        v = P.validity(hard, N, k)
        m = FR.km1_metrics(H, hard)
        zc = float(r1["zoltan_eval"]["cutl_global"])
        rec.update({"run": {kk: r1[kk] for kk in ("name", "mode", "np", "rc", "wall_seconds_outer", "memory", "timing", "zoltan_eval", "zoltan_removed_or_warning_lines")},
                    "raw": {"validity": v, "km1": m["km1_weighted"], "cut_weighted": m["cut_weighted"], "blocks": m["blocks"],
                            "zoltan_cutl_vs_python_km1_rel": round(abs(zc - m["km1_weighted"]) / max(m["km1_weighted"], 1), 8)}})
        log("  R1: rc %d, %.1fs, peak RSS/rank %s MB, km1 %d, blocks used %d/%d empty %d, max %d (bound %d) -> %s" % (
            r1["rc"], r1["wall_seconds_outer"], [round(x / 1024.0) for x in r1["memory"]["peak_rss_kb_per_rank_time_v"]], m["km1_weighted"],
            m["blocks"]["used"], k, m["blocks"]["empty"], m["blocks"]["max"], v["contract_bound_ceil_1.03_N_over_k"], v["gate"]))
        if not (v["length_ok"] and v["ids_in_range"] and v["max_block"] <= v["contract_bound_ceil_1.03_N_over_k"]):
            rec.update({"STATUS": "PARTITION_INVALID", "gate": v, "wall_seconds": round(time.time() - t, 1),
                        "rule": "as _l1c_phg.py: a raw partition whose largest block exceeds ceil(1.03 N / k) is refused; no map is written (never relaxed or replaced)"})
            wj(failp, rec)
            log("%s K %d: PHG PARTITION_INVALID -> absent" % (ds, K))
            continue
        if m["blocks"]["empty"] > 0:
            hard2, empties, moves = PR.repair(dict(H, meta=H["meta"]), hard)
            rec["repair"] = {"rule": PR.REPAIR, "empty_blocks": empties, "moves": len(moves), "km1_before": moves[0]["km1_before"] if moves else m["km1_weighted"],
                             "km1_after": moves[-1]["km1_after"] if moves else m["km1_weighted"]}
            hard = hard2
            m2 = FR.km1_metrics(H, hard)
            rec["repaired"] = {"validity": P.validity(hard, N, k), "km1": m2["km1_weighted"], "blocks": m2["blocks"]}
            log("  repair: %d empty blocks -> %d moves, km1 %d -> %d, %s" % (len(empties), len(moves), rec["repair"]["km1_before"], rec["repair"]["km1_after"],
                                                                           rec["repaired"]["validity"]["gate"]))
            assert rec["repaired"]["validity"]["gate"] == "PASS"
        else:
            rec["repair"] = None
        hard = hard.astype(np.int64)
        bal = balance(hard, N, K)
        assert bal["length_ok"] and bal["ids_in_range"] and bal["every_block_used"], bal
        tmp = out[:-4] + ".tmp.npy"
        np.save(tmp, hard)
        os.replace(tmp, out)
        rec.update({"STATUS": "OK", "balance": bal, "wall_seconds": round(time.time() - t, 1),
                    "output": {"file": rel(out), "sha256": sha_file(out), "n": int(len(hard))}})
        wj(runp, rec)
        log("%s K %d: PHG done %.0fs, repair %s, sizes %d..%d -> %s" % (ds, K, rec["wall_seconds"], (rec["repair"] or {}).get("moves"),
                                                                     bal["size_min"], bal["size_max"], rel(out)))


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    log("placement:", json.dumps(placement()))
    if a[0] == "BUILD":
        cmd_build()
        return
    ds, Ks = a[1], [int(x) for x in a[2:]]
    assert ds in AD.DATASETS and Ks
    if a[0] in ("HG", "CELL"):
        cmd_hg(ds, Ks)
    if a[0] in ("PHG", "CELL"):
        cmd_phg(ds, Ks)
    log("done %.0fs" % (time.time() - T0))


if __name__ == "__main__":
    main()
