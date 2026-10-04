"""The external >=250 GB lane for WebQSP / HotpotQA / 2Wiki partitions (plan step 5).

    python src/l1_canonical/external_kit.py pack <ds> [ds ...]
        -> data/l1_canonical/_external/l1_partition_kit_<ds>.tar.gz  (self-contained: the frozen
           builder + worker, the adapter, the dataset records and the STRUCT/KNN key sets; no
           embeddings, no nodes, no queries -- the partition needs none of them)
    python src/l1_canonical/external_kit.py import <ds> <dir_with_H4_SK.npy_and_H4_SK.stats.json_and_EXTERNAL_RUN.json>
        -> data/l1_canonical/<ds>/parts/H4_SK.{npy,stats.json,json}  after the same post-checks
           partition.py applies locally (length, every block used, balance envelope, hypergraph
           digest match), pinned to the DATASET.json record

On the external machine (Linux, python3 + numpy + mtkahypar 1.6.2 pip):
    tar xzf l1_partition_kit_<ds>.tar.gz && cd l1_partition_kit_<ds> && bash run_external.sh <ds> [threads]
The contract is unchanged: H4_SPLIT_PRESERVE over STRUCT+KNN closed neighbourhoods, k = N // 100,
Mt-KaHyPar DETERMINISTIC_QUALITY / KM1 / eps 0.03 / seed 0 (thread-count independent).
"""
import io
import json
import os
import shutil
import sys
import tarfile
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from src.l1_canonical.adapter import CanonicalDataset, contract_hash, sha_file  # noqa: E402
from src.l1_canonical.partition import post_checks, WORKER  # noqa: E402

KIT_DIR = os.path.join(REPO, "data", "l1_canonical", "_external")
CODE = ["src/__init__.py", "src/l1_canonical/__init__.py", "src/l1_canonical/adapter.py", "src/l1_canonical/hypergraph.py",
        "scratchpad/_l1hu_build.py", "scratchpad/_l1kn_sub.py", "scratchpad/_l1ep_part.py", "scratchpad/_l1ep_pu.py",
        "scratchpad/_l1ep_sub.py", "scratchpad/_l1hu_local_worker.py",
        "data/final_canonical/canonical.py", "data/final_canonical/CANONICAL_FREEZE.json"]

RUN_SH = r'''#!/bin/bash
# external lane: build the H4_SK hypergraph from the shipped key sets, partition it with the frozen worker
set -e
DS="$1"; THREADS="${2:-16}"
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 PYTHONUTF8=1
# memory pre-flight: KIT.json carries the expectation extrapolated from the two legacy Modal measurements
# (hotpotqa_clean 19.3M pins -> 263 GB, webqsp 10.9M pins -> 141 GB, both 16 threads); refuse below the
# optimistic lower bound unless FORCE=1 -- an OOM kill after hours costs more than this check
python3 - <<EOF
import json, os, sys
kit = json.load(open("KIT.json"))
me = kit["memory_expectation"]
tot = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]
gb = int(tot) / 1e6
print(json.dumps({"MemTotal_GB": round(gb, 1), "expected_lower_bound_GB": me["linear_in_pins_GB"], "expected_if_pins_x_k_GB": me["pins_times_k_GB"]}, indent=1))
if gb < me["linear_in_pins_GB"] and os.environ.get("FORCE") != "1":
    print("REFUSING: MemTotal below the optimistic expectation; rerun with FORCE=1 to try anyway"); sys.exit(3)
EOF
python3 - <<EOF
import json, numpy as np, os, sys, time, platform, resource
sys.path.insert(0, "."); sys.path.insert(0, "scratchpad")
from src.l1_canonical.hypergraph import build_canonical
t = time.time()
m = build_canonical("$DS")
print(json.dumps({"hypergraph_seconds": round(time.time() - t, 1), "hyperedges": m["hyperedges"], "pins": m["pins"], "k": m["k"]}, indent=1))
EOF
mkdir -p "data/l1_canonical/$DS/parts"
/usr/bin/time -v python3 scratchpad/_l1hu_local_worker.py "data/l1_canonical/$DS/hypergraph/H4_SK.npz" \
    "data/l1_canonical/$DS/parts/H4_SK.npy" "data/l1_canonical/$DS/parts/H4_SK.stats.json" "$THREADS" 2> "data/l1_canonical/$DS/parts/H4_SK.time.txt"
python3 - <<EOF
import json, platform, subprocess, time, hashlib
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
try:
    import mtkahypar; v = getattr(mtkahypar, "__version__", "?")
except Exception as e:
    v = repr(e)
pipv = subprocess.run(["pip", "show", "mtkahypar"], capture_output=True, text=True).stdout
rec = {"dataset": "$DS", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ran_on": "external", "host": platform.node(),
       "platform": platform.platform(), "python": platform.python_version(), "mtkahypar___version__": v,
       "mtkahypar_pip": [l for l in pipv.splitlines() if l.lower().startswith("version")], "threads": int("$THREADS"),
       "hypergraph_sha256": sha("data/l1_canonical/$DS/hypergraph/H4_SK.npz"),
       "hypergraph_manifest": json.load(open("data/l1_canonical/$DS/hypergraph/H4_SK.json")),
       "partition_sha256": sha("data/l1_canonical/$DS/parts/H4_SK.npy"),
       "worker_stats": json.load(open("data/l1_canonical/$DS/parts/H4_SK.stats.json")),
       "time_v": open("data/l1_canonical/$DS/parts/H4_SK.time.txt").read()[-3000:]}
json.dump(rec, open("data/l1_canonical/$DS/parts/EXTERNAL_RUN.json", "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("host", "mtkahypar___version__", "threads", "partition_sha256")}, indent=1))
print("RETURN these three files:", "data/l1_canonical/$DS/parts/H4_SK.npy", "H4_SK.stats.json", "EXTERNAL_RUN.json")
EOF
'''


# the two legacy Mt-KaHyPar DETERMINISTIC_QUALITY runs big enough to measure (Modal, 16 threads, scratchpad/_l1hu/parts/*.stats.json)
LEGACY_MEM = {"hotpotqa_clean__H4_SPLIT_PRESERVE__SKN": {"pins": 19345663, "hyperedges": 1473739, "k": 5074, "peak_rss_gb": 262.8},
              "webqsp__H4_SPLIT_PRESERVE__SKN": {"pins": 10900573, "hyperedges": 1846019, "k": 7814, "peak_rss_gb": 140.6}}


def memory_expectation(N, nST, nKN):
    """Peak-RSS expectation for the frozen worker on this corpus, extrapolated from LEGACY_MEM.
    pins ~= N_f + 2|F| per family (verified on the metaqa/squad/musique builds); two models bracket
    the legacy points: linear in pins (13 GB/Mpins, fits both within 5%) and linear in pins*k
    (the only single-parameter model that also fits the small local runs within 5x)."""
    pins = 2 * N + 2 * (nST + nKN)
    k = max(1, N // 100)
    lin = min(v["peak_rss_gb"] / (v["pins"] / 1e6) for v in LEGACY_MEM.values()) * pins / 1e6
    pk = min(v["peak_rss_gb"] / (v["pins"] / 1e6 * v["k"]) for v in LEGACY_MEM.values()) * pins / 1e6 * k
    return {"pins_estimate": int(pins), "k": k, "linear_in_pins_GB": round(lin), "pins_times_k_GB": round(pk),
            "legacy_points": LEGACY_MEM, "threads_measured": 16,
            "note": "both models put every canonical big corpus above the 250 GB lane; memory may also scale with threads -- start at 16"}


def pack(ds):
    d = CanonicalDataset(ds)
    # memory-lean: the key-set counts come from the npz meta member (np.load is lazy per member), the
    # big arrays are never materialised here -- the host is RAM-saturated by foreign processes
    kz = np.load(d._keys_path())
    kmeta = json.loads(str(kz["meta_json"]))
    if kmeta.get("DATASET_json_RECORD_SHA256") != d.record_sha:
        raise RuntimeError("%s: keys.npz was built from another DATASET.json record" % ds)
    N, nST, nKN = int(kmeta["N"]), int(kmeta["STRUCT"]), int(kmeta["KNN"])
    kz.close()
    os.makedirs(KIT_DIR, exist_ok=True)
    stage = os.path.join(KIT_DIR, "l1_partition_kit_%s" % ds)
    if os.path.exists(stage):
        shutil.rmtree(stage)
    for rel in CODE:
        dst = os.path.join(stage, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, rel), dst)
    for rel in ("%s/DATASET.json" % ds, "%s/graph/GRAPH_MANIFEST.json" % ds):
        dst = os.path.join(stage, "data", "final_canonical", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, "data", "final_canonical", rel), dst)
    # the key sets the hypergraph needs (STRUCT, KNN) plus the residual families so keysets() loads unchanged;
    # shipped byte-identical (streamed copy, the tarball's gzip does the compression) so the external
    # hypergraph manifest's inputs.keys_npz_sha256 must equal the local source digest at import time
    # (added straight from its source path when the tarball is written -- no staged copy, disk is tight)
    with io.open(os.path.join(stage, "run_external.sh"), "w", encoding="utf-8", newline="\n") as f:
        f.write(RUN_SH)
    ch = contract_hash()
    info = {"dataset": ds, "N": N, "k": max(1, N // 100), "STRUCT": nST, "KNN": nKN,
            "expected_pins_upper_bound": int(2 * (nST + nKN) + 2 * N),
            "DATASET_json_RECORD_SHA256": d.record_sha, "keys_npz_sha256": sha_file(d._keys_path()),
            "keys_npz_shipped": "byte-identical copy of data/l1_canonical/%s/keys.npz (STORED npz; import verifies the external hypergraph manifest against this digest)" % ds,
            "L1_CONTRACT_SHA256": ch["L1_CONTRACT_SHA256"], "contract_files": ch["files"],
            "memory_expectation": memory_expectation(N, nST, nKN),
            "packed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with io.open(os.path.join(stage, "KIT.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(info, indent=1))
    with io.open(os.path.join(stage, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# L1 partition kit: %s\n\n%s\n\nRun: `bash run_external.sh %s [threads=16]` (python3, numpy, mtkahypar 1.6.2 via pip; "
                "run_external.sh refuses when /proc/meminfo MemTotal is below memory_expectation.linear_in_pins_GB unless FORCE=1). "
                "Return `data/l1_canonical/%s/parts/{H4_SK.npy,H4_SK.stats.json,EXTERNAL_RUN.json}` and import them with "
                "`python src/l1_canonical/external_kit.py import %s <dir>`.\n" % (ds, json.dumps(info, indent=1), ds, ds, ds))
    tgz = stage + ".tar.gz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(stage, arcname=os.path.basename(stage))
        tf.add(d._keys_path(), arcname=os.path.basename(stage) + "/data/l1_canonical/%s/keys.npz" % ds)
    shutil.rmtree(stage)
    print("packed", tgz, "%.1f MB" % (os.path.getsize(tgz) / 1e6), json.dumps({k: info[k] for k in ("N", "k", "STRUCT", "KNN")}))
    return tgz


def import_result(ds, src_dir):
    d = CanonicalDataset(ds)
    hg_json = os.path.join(d.derived_dir, "hypergraph", "H4_SK.json")
    ext = json.load(io.open(os.path.join(src_dir, "EXTERNAL_RUN.json"), encoding="utf-8"))
    st = json.load(io.open(os.path.join(src_dir, "H4_SK.stats.json"), encoding="utf-8"))
    npy = os.path.join(src_dir, "H4_SK.npy")
    if sha_file(npy) != ext["partition_sha256"]:
        raise RuntimeError("H4_SK.npy digest != EXTERNAL_RUN.json")
    hgm = ext["hypergraph_manifest"]
    if hgm["inputs"]["DATASET_json_RECORD_SHA256"] != d.record_sha:
        raise RuntimeError("external hypergraph built from another DATASET.json record")
    if os.path.exists(hg_json):
        local = json.load(io.open(hg_json, encoding="utf-8"))
        if local["content_digest"] != hgm["content_digest"]:
            raise RuntimeError("external hypergraph content digest != local build")
    # the external hypergraph must have been built from exactly the local key sets (kits ship keys.npz byte-identical)
    kz = np.load(d._keys_path())
    kmeta = json.loads(str(kz["meta_json"]))
    kz.close()
    if hgm["inputs"].get("keys_npz_sha256") != sha_file(d._keys_path()):
        raise RuntimeError("external hypergraph inputs.keys_npz_sha256 != local data/l1_canonical/%s/keys.npz" % ds)
    if (int(hgm["inputs"]["STRUCT"]), int(hgm["inputs"]["KNN"]), int(hgm["N"])) != (int(kmeta["STRUCT"]), int(kmeta["KNN"]), d.n_nodes):
        raise RuntimeError("external hypergraph STRUCT/KNN/N counts != local key sets")
    if int(hgm["k"]) != max(1, d.n_nodes // 100):
        raise RuntimeError("external hypergraph k != frozen rule N // 100")
    hard = np.load(npy)
    k = int(hgm["k"])
    pc = post_checks(d, hard, k)
    pdir = os.path.join(d.derived_dir, "parts")
    os.makedirs(pdir, exist_ok=True)
    out_npy = os.path.join(pdir, "H4_SK.npy")
    shutil.copyfile(npy, out_npy)
    shutil.copyfile(os.path.join(src_dir, "H4_SK.stats.json"), os.path.join(pdir, "H4_SK.stats.json"))
    shutil.copyfile(os.path.join(src_dir, "EXTERNAL_RUN.json"), os.path.join(pdir, "EXTERNAL_RUN.json"))
    rec = {"dataset": ds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "contract": {"partitioner": "mtkahypar", "preset": "DETERMINISTIC_QUALITY", "objective": "KM1", "epsilon": 0.03, "seed": 0,
                        "vertex_weights": "unit", "hyperedge_weights": "H4 build (max(1, rint(1000/(|e|-1))))",
                        "k": k, "k_rule": hgm["k_rule"], "hypergraph_rule": hgm["rule"], "families": hgm["families"],
                        "worker": WORKER, "worker_sha256": sha_file(os.path.join(REPO, WORKER)), "threads": ext.get("threads"),
                        "ran_on": "external", "external": {kk: ext.get(kk) for kk in ("host", "platform", "python", "mtkahypar___version__", "mtkahypar_pip", "utc")}},
           "inputs": {"DATASET_json_RECORD_SHA256": d.record_sha, "hypergraph_file": hgm.get("file"), "hypergraph_sha256": ext["hypergraph_sha256"],
                      "hypergraph_content_digest": hgm["content_digest"], "hyperedges": hgm["hyperedges"], "pins": hgm["pins"], "N": hgm["N"]},
           "STATUS": "OK" if pc["every_block_used"] and pc["length_ok"] else "BUILT_WITH_WARNINGS",
           "worker_stats": st, "post_checks": pc,
           "output": {"file": os.path.relpath(out_npy, REPO).replace("\\", "/"), "bytes": os.path.getsize(out_npy), "sha256": sha_file(out_npy),
                      "dtype": str(hard.dtype), "n": int(len(hard))},
           "L1_CONTRACT_SHA256": contract_hash()["L1_CONTRACT_SHA256"]}
    with io.open(os.path.join(pdir, "H4_SK.json"), "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    print(ds, rec["STATUS"], json.dumps(pc))
    return rec


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "pack":
        for ds in a[1:]:
            pack(ds)
    elif a and a[0] == "import" and len(a) == 3:
        import_result(a[1], a[2])
    else:
        print(__doc__)
