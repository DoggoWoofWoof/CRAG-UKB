"""L1_COVPART step 2b: partition a scratch hypergraph with the VALIDATED Zoltan-PHG substitute (src/l1_lowmem/phg.py: the same
compiled driver, the same explicit parameter list, NP = 4 ranks, IMBALANCE_TOL 1.03, CONNECTIVITY objective) followed by the
pre-registered non-empty repair (src/l1_lowmem/phg_repair.repair) exactly as the served metaqa PHG partition was made
(parts/LOWMEM__PHG_REPAIR1_con).  Everything is imported from the lane modules; nothing is re-implemented except the
shard writer, which is checked byte-for-byte against the served H4_SK_STREAM_V1 shards before any scratch run.

    python -u _l1c_phg.py <ds> <tag>            e.g.  metaqa H4_SK_PATH2   (parts/<ds>__<tag>.npz -> parts/<ds>__<tag>__PHG_con.npy)
    python -u _l1c_phg.py <ds> H4_SK             re-partitions the FROZEN H4_SK (shard bytes must equal the served shards; the
                                                 result is compared with the served PHG R1 vector -- the pipeline check)
All files go under results/L1_COVPART/ (ds_dir is redirected there); nothing under data/ is read for writing.
"""
import io
import json
import os
import sys
import time

import numpy as np

import _l1s_core as S
from src.l1_lowmem import common as CM  # noqa: E402
from src.l1_lowmem import freight as FR  # noqa: E402
from src.l1_lowmem import phg as P  # noqa: E402
from src.l1_lowmem import phg_repair as PR  # noqa: E402
from src.l1_canonical.adapter import CanonicalDataset, sha_file  # noqa: E402

LANE = os.path.join(S.X.REPO, "results", "L1_COVPART")
PDIR = os.path.join(LANE, "parts")
CM.DATA = os.path.join(LANE, "phg_data")          # every ds_dir()/run_dir() of the lane modules now resolves under the lane
assert P.ds_dir is CM.ds_dir and P.run_dir.__globals__["ds_dir"] is CM.ds_dir
SHARD_NODES = 5000
ds, tag = sys.argv[1], sys.argv[2]
t0 = time.time()


def load_scratch(ds, tag):
    if tag == "H4_SK":
        cd = CanonicalDataset(ds)
        npz = os.path.join(cd.derived_dir, "hypergraph", "H4_SK.npz")
        meta = json.load(io.open(npz[:-4] + ".json", encoding="utf-8"))
    else:
        npz = os.path.join(PDIR, "%s__%s.npz" % (ds, tag))
        meta = json.load(io.open(npz[:-4] + ".json", encoding="utf-8"))
        assert sha_file(npz) == meta["file_sha256"]
    z = np.load(npz)
    eptr, eidx, ew = z["eptr"].astype(np.int64), z["eidx"].astype(np.int64), z["ew"].astype(np.int64)
    N, k = int(z["N"][0]), int(z["k"][0])
    return {"npz": npz, "npz_sha256": sha_file(npz), "meta": meta, "eptr": eptr, "eidx": eidx, "ew": ew, "N": N, "k": k, "M": int(len(eptr) - 1), "P": int(len(eidx))}


def write_shards(H, sdir):
    """node-centric net lists (fmt 1): per node '<net> <w> <net> <w> ...', nets 1-based ascending -- the served shard format."""
    os.makedirs(sdir, exist_ok=True)
    N, M, eptr, eidx, ew = H["N"], H["M"], H["eptr"], H["eidx"], H["ew"]
    hid = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    order = np.lexsort((hid, eidx))                      # by node, then net id ascending
    node_of, net_of = eidx[order], hid[order]
    deg = np.bincount(node_of, minlength=N)
    off = np.zeros(N + 1, np.int64); off[1:] = np.cumsum(deg)
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


H = load_scratch(ds, tag)
N, k, M, Pn = H["N"], H["k"], H["M"], H["P"]
sdir = os.path.join(CM.DATA, ds, "stream_%s" % tag)
shards, smf = write_shards(H, sdir)
S.log("%s %s: N %d M %d P %d k %d -> %d shards under %s" % (ds, tag, N, M, Pn, k, len(shards), os.path.relpath(sdir, S.X.REPO)))
rec = {"dataset": ds, "tag": tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "hypergraph": {"file": os.path.relpath(H["npz"], S.X.REPO).replace("\\", "/"), "sha256": H["npz_sha256"], "N": N, "M": M, "P": Pn, "k": k},
       "shards": [{kk: (os.path.relpath(v, S.X.REPO).replace("\\", "/") if kk == "file" else v) for kk, v in c.items()} for c in shards]}
if tag == "H4_SK":                                   # pipeline check: my shards must be the served shards, byte for byte
    served = json.load(io.open(os.path.join(S.X.REPO, "data", "l1_lowmem", ds, "stream", "H4_SK_STREAM_V1.json"), encoding="utf-8"))
    same = [c["sha256"] == s["sha256"] for c, s in zip(shards, served["shards"])]
    rec["shard_bytes_equal_served"] = bool(all(same) and len(shards) == len(served["shards"]))
    S.log("  shard bytes == served H4_SK_STREAM_V1 shards: %s" % rec["shard_bytes_equal_served"])
    assert rec["shard_bytes_equal_served"], "shard writer does not reproduce the served shards"
b = CM.rj(os.path.join(CM.OUT, "PHG_BUILD.json"))
chk = FR.sh("sha256sum %s %s" % (b["binary"]["path"], b["wrapper"]["path"]), quiet=True).stdout.split()
assert chk and chk[0] == b["binary"]["sha256"] and chk[2] == b["wrapper"]["sha256"], "PHG driver binary/wrapper changed since PHG_BUILD.json"
rec["driver"] = {"binary": b["binary"], "wrapper": b["wrapper"], "source_sha256": b["source"]["sha256"], "parameters": P.PARAMS, "NP": P.NP}
r1, d1 = P.mpirun(ds, "R1_%s" % tag, "partition", k, smf, b)
gq = r1["global_from_queries"]
assert gq["objects"] == N and gq["pins"] == Pn, ("Zoltan saw a different hypergraph", gq)
hard = P.assemble_partition(d1, N, k)
v = P.validity(hard, N, k)
m = FR.km1_metrics(H, hard)
zc = float(r1["zoltan_eval"]["cutl_global"])
rec.update({"run": {kk: r1[kk] for kk in ("name", "mode", "np", "rc", "wall_seconds_outer", "memory", "timing", "zoltan_eval", "zoltan_removed_or_warning_lines")},
            "raw": {"validity": v, "km1": m["km1_weighted"], "cut_weighted": m["cut_weighted"], "blocks": m["blocks"],
                    "zoltan_cutl_vs_python_km1_rel": round(abs(zc - m["km1_weighted"]) / max(m["km1_weighted"], 1), 8)}})
S.log("  R1: rc %d, %.1fs, peak RSS/rank %s MB, km1 %d (Zoltan cutl %.0f), blocks used %d/%d empty %d, max %d (bound %d) -> %s" % (
    r1["rc"], r1["wall_seconds_outer"], [round(x / 1024.0) for x in r1["memory"]["peak_rss_kb_per_rank_time_v"]], m["km1_weighted"], zc,
    m["blocks"]["used"], k, m["blocks"]["empty"], m["blocks"]["max"], v["contract_bound_ceil_1.03_N_over_k"], v["gate"]))
assert v["length_ok"] and v["ids_in_range"] and v["max_block"] <= v["contract_bound_ceil_1.03_N_over_k"], v
if m["blocks"]["empty"] > 0:
    hard2, empties, moves = PR.repair(dict(H, meta=H["meta"]), hard)
    rec["repair"] = {"rule": PR.REPAIR, "empty_blocks": empties, "moves": len(moves), "km1_before": moves[0]["km1_before"] if moves else m["km1_weighted"],
                     "km1_after": moves[-1]["km1_after"] if moves else m["km1_weighted"]}
    hard = hard2
    m2 = FR.km1_metrics(H, hard)
    rec["repaired"] = {"validity": P.validity(hard, N, k), "km1": m2["km1_weighted"], "blocks": m2["blocks"]}
    S.log("  repair: %d empty blocks -> %d moves, km1 %d -> %d, %s" % (len(empties), len(moves), rec["repair"]["km1_before"], rec["repair"]["km1_after"], rec["repaired"]["validity"]["gate"]))
    assert rec["repaired"]["validity"]["gate"] == "PASS"
else:
    rec["repair"] = None
if tag == "H4_SK":                                   # the served PHG vector: REPAIR1_con where a repair was needed (metaqa), else PHG_con (squad)
    sp_ = [n for n in ("LOWMEM__PHG_REPAIR1_con", "LOWMEM__PHG_con") if os.path.exists(os.path.join(S.X.REPO, "data", "l1_canonical", ds, "parts", n + ".npy"))][0]
    served_r1 = np.load(os.path.join(S.X.REPO, "data", "l1_canonical", ds, "parts", sp_ + ".npy")).astype(np.int64)
    rec["served_partition"] = sp_
    rec["partition_equals_served_PHG"] = bool(np.array_equal(hard, served_r1))
    rec["km1_served_PHG"] = FR.km1_metrics(H, served_r1)["km1_weighted"]
    S.log("  partition == served %s: %s (km1 served %d)" % (sp_, rec["partition_equals_served_PHG"], rec["km1_served_PHG"]))
out = os.path.join(PDIR, "%s__%s__PHG_con.npy" % (ds, tag))
np.save(out, hard.astype(np.int64))
rec["output"] = {"file": os.path.relpath(out, S.X.REPO).replace("\\", "/"), "sha256": sha_file(out), "n": int(len(hard)), "STATUS": "OK"}
S.wj(os.path.join(PDIR, "%s__%s__PHG_con.RUN.json" % (ds, tag)), rec)
S.log("done %.0fs -> %s" % (time.time() - t0, out))
