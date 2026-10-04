"""L1 DEVELOPMENT -- SCALE partitions: the shard maps of the partition-scale experiment (user 2026-09-29, after REPORT section
34: "Does this architecture actually scale as the number of partitions grows? ... construct multiple partition granularities,
for example: K in {100,250,500,1000,2000,5000} where feasible. Keep the algorithm identical: IR_L1 + ES.KNEE_SDIV ... At each
K, compare at least: balanced random partition versus PHG/H4_SK structural partition.").

Nothing under data/ is written.  Every file goes to results/L1_COVPART/parts/ (the lane's scratch partition directory, where
scratchpad/_l1c_phg.py reads and writes).  The structural partitions at K are the served recipes with only K changed:

  HG   the H4_SK hypergraph at K by the FROZEN rule: src/l1_canonical/hypergraph.build_hypergraph(N, {STRUCT, KNN}, K)
       (H4_SPLIT_PRESERVE; the rule itself ties the hyperedge cap to the block size, cap = round(N / K)), from the dataset's
       own stamped key sets.  At the native K (N // 100) the content digest must equal the frozen H4_SK.json (asserted).
       -> <ds>__H4_SK_k<K>.npz + .json
  PHG  scratchpad/_l1c_phg.py <ds> H4_SK_k<K>, run UNCHANGED (the validated Zoltan-PHG substitute: the same compiled driver,
       the same parameter list, NP = 4 ranks, IMBALANCE_TOL 1.03, CONNECTIVITY, then PHG_NONEMPTY_REPAIR_V1)
       -> <ds>__H4_SK_k<K>__PHG_con.npy + .RUN.json (its own record) + .driver.log; this module adds
          <ds>__H4_SK_k<K>__PHG_con.SCALE.json (balance checks; at the native K: equality with the served PHG partition).
  MTK  the frozen Mt-KaHyPar contract: the UNCHANGED worker scratchpad/_l1hu_local_worker.py (DETERMINISTIC_QUALITY, KM1,
       eps 0.03, seed 0, unit vertex weights, H4 hyperedge weights) under WSL, behind the RSS guard of
       src/l1_canonical/partition.py (its GUARD_SH text, written under results/ instead of data/; cap = min(6.0, available
       - 1.0) GB, refused below 0.5 GB).  A run that crosses the cap writes <ds>__H4_SK_k<K>__MTK.FAILED.json and no
       partition: that cell is then absent, never replaced by another partitioner.  At the native K the vector must equal
       data/l1_canonical/<ds>/parts/H4_SK.npy (asserted).  MuSiQue is not attempted: its native Mt-KaHyPar run is
       FAILED_MEMORY_CAP (data/l1_canonical/musique/parts/H4_SK.FAILED.json, 6,014,938 pins, killed at 6.0 GB) and the host has
       less memory free than that cap.
       -> <ds>__H4_SK_k<K>__MTK.npy + .RUN.json (+ .stats.json, .worker.log)
       --retry-memory-cap (RETRY_RULE below): a cell killed at a HOST-LIMITED cap (below the contract's 6.0 GB) is run again,
       unchanged, when the host now allows a larger cap under the same cap rule.  The earlier FAILED record, worker log and
       stats move to results/L1_COVPART/_history/ as <ds>__H4_SK_k<K>__MTK__attempt<n>.*, and the new record lists them (with
       their sha256) under previous_attempts.  A cell killed at the contract cap itself is never retried.
The balanced random partitions (RAND, seeds 0-2) are a deterministic function of (N, K, seed) and are built in memory by the
harness (_l1d_scale.py), which also pins the sha256 of every file listed here.

Usage:  python -u scratchpad/_l1d_scale_parts.py HG  <ds> <K> [<K> ...]
        python -u scratchpad/_l1d_scale_parts.py PHG <ds> <K> [<K> ...]
        python -u scratchpad/_l1d_scale_parts.py MTK <ds> <K> [<K> ...] [--threads=8] [--retry-memory-cap]
        python -u scratchpad/_l1d_scale_parts.py STATUS
"""
import ast
import hashlib
import io
import json
import os
import subprocess
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
from src.l1_canonical import partition as PT  # noqa: E402

PDIR = os.path.join(REPO, "results", "L1_COVPART", "parts")
HISTORY = os.path.join(REPO, "results", "L1_COVPART", "_history")
RETRY_RULE = ("--retry-memory-cap: a cell whose FAILED record is FAILED_MEMORY_CAP at a host-limited cap (rss_cap_gb below the "
              "contract's HARD_CAP_GB 6.0) is run again, unchanged, only when the same cap rule min(6.0, available - 1.0) now gives "
              "a larger cap. The earlier FAILED record, worker log and stats move to results/L1_COVPART/_history/ as "
              "<ds>__H4_SK_k<K>__MTK__attempt<n>.* (sha256 recorded here). Mt-KaHyPar DETERMINISTIC_QUALITY with seed 0 and fixed "
              "threads is deterministic (the native-K rebuilds reproduce the frozen H4_SK exactly), so a retry can change only "
              "whether the run finishes, not which partition it returns. A cell killed at the contract cap itself is never retried.")
KGRID = (100, 250, 500, 1000, 2000, 5000)
DATASETS = ("metaqa", "squad", "musique")
SERVED_PHG = {"metaqa": "LOWMEM__PHG_REPAIR1_con", "squad": "LOWMEM__PHG_con", "musique": "LOWMEM__PHG_C1_con"}
MTK_SETS = ("metaqa", "squad")
GUARD = os.path.join(PDIR, "_scale_mtk_guard.sh")
PHG_SCRIPT = os.path.join(HERE, "_l1c_phg.py")
WORKER = os.path.join(REPO, "scratchpad", "_l1hu_local_worker.py")
T0 = time.time()


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
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=1))


def tag_of(K):
    return "H4_SK_k%d" % int(K)


def hg_npz(ds, K):
    return os.path.join(PDIR, "%s__%s.npz" % (ds, tag_of(K)))


def phg_npy(ds, K):
    return os.path.join(PDIR, "%s__%s__PHG_con.npy" % (ds, tag_of(K)))


def mtk_npy(ds, K):
    return os.path.join(PDIR, "%s__%s__MTK.npy" % (ds, tag_of(K)))


def frozen_hg(cd):
    npz = os.path.join(cd.derived_dir, "hypergraph", "H4_SK.npz")
    meta = json.load(io.open(npz[:-4] + ".json", encoding="utf-8"))
    assert meta["inputs"]["DATASET_json_RECORD_SHA256"] == cd.record_sha and sha_file(npz) == meta["file_sha256"]
    return npz, meta


def stamped_keys(cd):
    """the dataset's stored key sets, read-only: the stamp must match (the adapter would otherwise rebuild under data/)."""
    z = np.load(cd._keys_path())
    assert json.loads(str(z["meta_json"])).get("DATASET_json_RECORD_SHA256") == cd.record_sha, "stale key sets: %s" % cd.name
    N, ST, KN, _ = cd.keysets()
    assert N == cd.n_nodes
    return N, {"STRUCT": ST, "KNN": KN}


def code_pins():
    return {"harness": {"path": rel(os.path.abspath(__file__)), "sha256": sha_file(os.path.abspath(__file__))},
            "hypergraph_builder": {"path": "src/l1_canonical/hypergraph.py", "sha256": sha_file(os.path.join(REPO, "src", "l1_canonical", "hypergraph.py"))},
            "frozen_rule_module": {"path": "scratchpad/_l1hu_build.py", "sha256": sha_file(os.path.join(REPO, "scratchpad", "_l1hu_build.py"))},
            "partition_driver_reused": {"path": "src/l1_canonical/partition.py", "sha256": sha_file(os.path.join(REPO, "src", "l1_canonical", "partition.py"))},
            "phg_script": {"path": rel(PHG_SCRIPT), "sha256": sha_file(PHG_SCRIPT)},
            "mtk_worker": {"path": rel(WORKER), "sha256": sha_file(WORKER)}}


def cmd_hg(ds, Ks):
    cd = AD.CanonicalDataset(ds)
    fnpz, fmeta = frozen_hg(cd)
    N, keys = stamped_keys(cd)
    kn = HG.frozen_k(N)
    for K in Ks:
        K = int(K)
        fp = hg_npz(ds, K)
        if os.path.exists(fp):
            m = json.load(io.open(fp[:-4] + ".json", encoding="utf-8"))
            assert sha_file(fp) == m["file_sha256"], "%s changed since its record" % fp
            log("%s K %d: exists (%s)" % (ds, K, m["file_sha256"][:16]))
            continue
        assert 1 <= K <= N
        t = time.time()
        arrays, meta = HG.build_hypergraph(N, keys, K, tag=ds)
        dig = HG.arrays_digest(arrays)
        eq = None
        if K == kn:
            eq = dig == fmeta["content_digest"]
            assert eq, "%s: the rebuilt native-K hypergraph differs from the frozen H4_SK.npz" % ds
        np.savez_compressed(fp, **arrays)
        meta.update({"dataset": ds, "file": rel(fp), "bytes": os.path.getsize(fp), "file_sha256": sha_file(fp), "content_digest": dig,
                     "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t, 1),
                     "inputs": {"DATASET_json_RECORD_SHA256": cd.record_sha, "keys_npz_sha256": sha_file(cd._keys_path()),
                                "STRUCT": int(len(keys["STRUCT"])), "KNN": int(len(keys["KNN"]))},
                     "scale": {"K": K, "native_K": kn, "block_size_round_N_over_K": int(round(N / K)),
                               "frozen_H4_SK": {"file": rel(fnpz), "sha256": fmeta["file_sha256"], "content_digest": fmeta["content_digest"]},
                               "content_equals_frozen_H4_SK": eq,
                               "rule": "the frozen H4_SK rule with k = K (cap = round(N / K)); nothing else changed"},
                     "code": code_pins()})
        wj(fp[:-4] + ".json", meta)
        log("%s K %d: cap %s, %d hyperedges, %d pins (dup %d), %.1f MB, %.1fs%s" % (
            ds, K, meta["cap"], meta["hyperedges"], meta["pins"], meta["anchor_duplication_pins"], meta["bytes"] / 1e6, meta["seconds"],
            "" if eq is None else ", == frozen H4_SK: %s" % eq))


def balance(hard, N, K):
    s = np.bincount(hard, minlength=K)
    tgt = int(np.ceil(N / float(K)))
    return {"length_ok": bool(len(hard) == N), "ids_in_range": bool(hard.min() >= 0 and hard.max() < K), "blocks_used": int((s > 0).sum()),
            "every_block_used": bool((s > 0).all()), "size_min": int(s.min()), "size_max": int(s.max()), "size_mean": round(float(s.mean()), 3),
            "ceil_N_over_K": tgt, "within_ceil_1.03_N_over_K": bool(s.max() <= int(np.ceil(1.03 * N / float(K))))}


def cmd_phg(ds, Ks):
    cd = AD.CanonicalDataset(ds)
    N = cd.n_nodes
    kn = HG.frozen_k(N)
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    for K in Ks:
        K = int(K)
        src, out = hg_npz(ds, K), phg_npy(ds, K)
        side, failp = out[:-4] + ".SCALE.json", out[:-4] + ".FAILED.json"
        if os.path.exists(side) or os.path.exists(failp):
            log("%s K %d: PHG %s" % (ds, K, "exists" if os.path.exists(side) else "FAILED earlier"))
            continue
        assert os.path.exists(src), "build the hypergraph first: HG %s %d" % (ds, K)
        assert not os.path.exists(out) and not os.path.exists(out[:-4] + ".RUN.json"), "write-once: %s" % out
        t = time.time()
        dlog = out[:-4] + ".driver.log"
        if os.path.exists(dlog):                   # an earlier run stopped without an output: record it, never re-run over its log
            rc, reused = None, True
        else:
            with io.open(dlog, "w", encoding="utf-8", newline="\n") as lf:
                rc = subprocess.run([sys.executable, "-u", PHG_SCRIPT, ds, tag_of(K)], cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
            reused = False
        if rc != 0 or not os.path.exists(out):
            txt = io.open(dlog, encoding="utf-8", errors="replace").read()
            assert "PARTITION_INVALID" in txt, "%s K %d: _l1c_phg.py failed (rc %s) for another reason; see %s" % (ds, K, rc, rel(dlog))
            gate = None
            for ln in txt.splitlines():
                if ln.startswith("AssertionError: {"):
                    gate = ast.literal_eval(ln[len("AssertionError: "):])
            fr = {"dataset": ds, "K": K, "native_K": kn, "STATUS": "PARTITION_INVALID",
                  "partitioner": "Zoltan-PHG (validated substitute; scratchpad/_l1c_phg.py unchanged)",
                  "gate": gate, "rule": "_l1c_phg.py refuses a raw partition whose largest block exceeds ceil(1.03 N / k) (its contract bound)",
                  "hypergraph": {"file": rel(src), "sha256": sha_file(src)}, "driver_log": {"file": rel(dlog), "sha256": sha_file(dlog),
                  "reused_from_an_earlier_run": reused}, "log_tail": txt[-3000:],
                  "note": "no partition is written for this cell; it is absent from the scale factorial (never relaxed or replaced)",
                  "code": code_pins()}
            wj(failp, fr)
            log("%s K %d: PHG PARTITION_INVALID (%s) -> absent" % (ds, K, gate))
            continue
        run = json.load(io.open(out[:-4] + ".RUN.json", encoding="utf-8"))
        assert run["output"]["sha256"] == sha_file(out)
        hard = np.load(out).astype(np.int64)
        b = balance(hard, N, K)
        assert b["length_ok"] and b["ids_in_range"] and b["every_block_used"], b
        eq = None
        if K == kn:
            served = cd.partition(SERVED_PHG[ds])
            eq = bool(np.array_equal(hard, served))
            assert eq, "%s: the native-K PHG rebuild differs from the served %s" % (ds, SERVED_PHG[ds])
        rec = {"dataset": ds, "K": K, "native_K": kn, "partitioner": "Zoltan-PHG (validated substitute; scratchpad/_l1c_phg.py unchanged)",
               "hypergraph": {"file": rel(src), "sha256": sha_file(src)}, "output": {"file": rel(out), "sha256": sha_file(out)},
               "phg_run_record": {"file": rel(out[:-4] + ".RUN.json"), "sha256": sha_file(out[:-4] + ".RUN.json")},
               "repair": run.get("repair"), "balance": b, "equals_served_PHG": eq, "served_PHG": SERVED_PHG[ds] if eq is not None else None,
               "wall_seconds": round(time.time() - t, 1), "run_memory": run["run"].get("memory"), "code": code_pins()}
        wj(side, rec)
        log("%s K %d: PHG done %.0fs, repair %s, sizes %d..%d%s" % (ds, K, rec["wall_seconds"], (run.get("repair") or {}).get("moves"),
                                                               b["size_min"], b["size_max"], "" if eq is None else ", == served: %s" % eq))


def run_guarded(src, out, stats, threads, cap_gb, wlog):
    txt = PT.GUARD_SH
    if not (os.path.exists(GUARD) and io.open(GUARD, encoding="utf-8").read() == txt):
        with io.open(GUARD, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
    args = ["wsl", "-d", PT.DISTRO, "-e", "bash", PT.wsl_path(GUARD), PT.wsl_path(src), PT.wsl_path(out), PT.wsl_path(stats),
            str(threads), str(int(cap_gb * 1e6)), PT.wsl_path(wlog), PT.wsl_path(REPO)]
    t = time.time()
    r = subprocess.run(args, capture_output=True, text=True)
    g = {}
    for ln in r.stdout.replace("\0", "").splitlines():
        if ln.startswith("GUARD "):
            for kv in ln.split()[1:]:
                k, v = kv.split("=")
                g[k] = int(v)
    g["wall_seconds"] = round(time.time() - t, 1)
    g["returncode"] = r.returncode
    g["guard_script"] = {"file": rel(GUARD), "sha256": sha_file(GUARD), "text_equals_partition_py_GUARD_SH": True}
    return g


def cmd_mtk(ds, Ks, threads, retry=False):
    assert ds in MTK_SETS, "%s: Mt-KaHyPar is not attempted (see the module docstring)" % ds
    cd = AD.CanonicalDataset(ds)
    N = cd.n_nodes
    kn = HG.frozen_k(N)
    for K in Ks:
        K = int(K)
        src, out = hg_npz(ds, K), mtk_npy(ds, K)
        runp, failp = out[:-4] + ".RUN.json", out[:-4] + ".FAILED.json"
        stats, wlog = out[:-4] + ".stats.json", out[:-4] + ".worker.log"
        if os.path.exists(runp) or (os.path.exists(failp) and not retry):
            log("%s K %d: MTK already %s" % (ds, K, "done" if os.path.exists(runp) else "FAILED"))
            continue
        assert os.path.exists(src), "build the hypergraph first: HG %s %d" % (ds, K)
        meta = json.load(io.open(src[:-4] + ".json", encoding="utf-8"))
        assert sha_file(src) == meta["file_sha256"]
        assert not os.path.exists(out)
        avail = PT.host_available_gb()
        cap_gb = min(PT.HARD_CAP_GB, (avail - PT.MARGIN_GB) if avail is not None else PT.HARD_CAP_GB)
        assert cap_gb >= 0.5, "%s K %d: only %.1f GB free; refusing to start the partitioner" % (ds, K, avail)
        prev = []
        if os.path.exists(failp):                  # --retry-memory-cap: RETRY_RULE
            fr = json.load(io.open(failp, encoding="utf-8"))
            pcap = fr["guard"]["rss_cap_gb"]
            if fr["STATUS"] != "FAILED_MEMORY_CAP" or pcap >= PT.HARD_CAP_GB or cap_gb <= pcap:
                log("%s K %d: MTK %s earlier at cap %.2f GB; not retried (the cap rule now gives %.2f GB)" % (ds, K, fr["STATUS"], pcap, cap_gb))
                continue
            n = len(fr.get("previous_attempts", [])) + 1
            moved = {}
            for p in (failp, wlog, stats):
                if os.path.exists(p):
                    dst = os.path.join(HISTORY, os.path.basename(p).replace("__MTK.", "__MTK__attempt%d." % n))
                    assert not os.path.exists(dst), "write-once: %s" % dst
                    moved[rel(dst)] = sha_file(p)
                    os.replace(p, dst)
            prev = fr.get("previous_attempts", []) + [{
                "attempt": n, "STATUS": fr["STATUS"], "utc": fr["utc"], "rss_cap_gb": pcap,
                "host_available_gb_at_start": fr["guard"].get("host_available_gb_at_start"), "peak_rss_kb": fr["guard"].get("peak_rss_kb"),
                "wall_seconds": fr["guard"].get("wall_seconds"), "files_moved_to_history": moved}]
            log("%s K %d: retry %d (FAILED_MEMORY_CAP at the host-limited cap %.2f GB; the cap rule now gives %.2f GB)" % (ds, K, n + 1, pcap, cap_gb))
        env = PT.wsl_env()
        log("%s K %d: Mt-KaHyPar start (%d hyperedges, %d pins, cap %.2f GB, %d threads)" % (ds, K, meta["hyperedges"], meta["pins"], cap_gb, threads))
        g = run_guarded(src, out, stats, threads, cap_gb, wlog)
        rec = {"dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "K": K, "native_K": kn,
               "contract": {"partitioner": "mtkahypar", "preset": "DETERMINISTIC_QUALITY", "objective": "KM1", "epsilon": 0.03, "seed": 0,
                            "vertex_weights": "unit", "hyperedge_weights": "H4 build (max(1, rint(1000/(|e|-1))))", "k": K,
                            "k_rule": "scale grid K (the frozen contract's k = N // 100 at the native K)", "hypergraph_rule": "H4_SPLIT_PRESERVE",
                            "families": ["STRUCT", "KNN"], "worker": rel(WORKER), "worker_sha256": sha_file(WORKER), "threads": threads},
               "wsl": env, "guard": dict(g, rss_cap_gb=cap_gb, host_available_gb_at_start=avail, margin_gb=PT.MARGIN_GB),
               "inputs": {"hypergraph_file": rel(src), "hypergraph_sha256": meta["file_sha256"], "hypergraph_content_digest": meta["content_digest"],
                          "hyperedges": meta["hyperedges"], "pins": meta["pins"], "N": N}, "code": code_pins()}
        if prev:
            rec.update({"previous_attempts": prev, "retry_rule": RETRY_RULE})
        if g.get("killed") or g.get("rc") != 0 or not os.path.exists(out):
            rec["STATUS"] = "FAILED_MEMORY_CAP" if g.get("killed") else "FAILED"
            rec["worker_log_tail"] = io.open(wlog, encoding="utf-8", errors="replace").read()[-3000:] if os.path.exists(wlog) else ""
            rec["note"] = "no partition is written for this cell; it is absent from the scale factorial (never replaced by another partitioner)"
            wj(failp, rec)
            log("%s K %d: %s (peak %s kB, %.0fs)" % (ds, K, rec["STATUS"], g.get("peak_rss_kb"), g["wall_seconds"]))
            continue
        hard = np.load(out).astype(np.int64)
        pc = PT.post_checks(cd, hard, K)
        eq = None
        if K == kn:
            eq = bool(np.array_equal(hard, cd.partition("H4_SK")))
            assert eq, "%s: the native-K Mt-KaHyPar rebuild differs from the frozen H4_SK.npy" % ds
        rec.update({"STATUS": "OK", "worker_stats": json.load(io.open(stats, encoding="utf-8")), "post_checks": pc,
                    "equals_frozen_H4_SK": eq,
                    "output": {"file": rel(out), "bytes": os.path.getsize(out), "sha256": sha_file(out), "dtype": "int64", "n": int(len(hard))},
                    "L1_CONTRACT_SHA256": AD.contract_hash()["L1_CONTRACT_SHA256"]})
        wj(runp, rec)
        log("%s K %d: Mt-KaHyPar OK %.0fs, peak %s kB, km1 %s, sizes %s..%s, balance_within_eps %s%s" % (
            ds, K, g["wall_seconds"], g.get("peak_rss_kb"), rec["worker_stats"].get("objective_km1"), pc["size_min"], pc["size_max"],
            pc["balance_within_eps"], "" if eq is None else ", == frozen H4_SK: %s" % eq))


def cmd_status():
    for ds in DATASETS:
        cd = AD.CanonicalDataset(ds)
        kn = HG.frozen_k(cd.n_nodes)
        for K in sorted(set(KGRID) | {kn}):
            st = []
            st.append("HG" if os.path.exists(hg_npz(ds, K)) else "--")
            pp = phg_npy(ds, K)
            st.append("PHG" if os.path.exists(pp[:-4] + ".SCALE.json") else ("PHG_INVALID" if os.path.exists(pp[:-4] + ".FAILED.json") else "---"))
            if ds in MTK_SETS:
                mp = mtk_npy(ds, K)
                st.append("MTK" if os.path.exists(mp[:-4] + ".RUN.json") else ("MTK_FAILED" if os.path.exists(mp[:-4] + ".FAILED.json") else "---"))
            log("%-8s K %5d%s: %s" % (ds, K, " (native)" if K == kn else "", " ".join(st)))


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    threads = next((int(x.split("=", 1)[1]) for x in sys.argv if x.startswith("--threads=")), PT.DEFAULT_THREADS)
    retry = "--retry-memory-cap" in sys.argv
    if not a:
        print(__doc__)
        return
    if a[0] == "STATUS":
        return cmd_status()
    mode, ds, Ks = a[0], a[1], [int(x) for x in a[2:]]
    assert ds in DATASETS and Ks
    {"HG": lambda: cmd_hg(ds, Ks), "PHG": lambda: cmd_phg(ds, Ks), "MTK": lambda: cmd_mtk(ds, Ks, threads, retry)}[mode]()


if __name__ == "__main__":
    main()
