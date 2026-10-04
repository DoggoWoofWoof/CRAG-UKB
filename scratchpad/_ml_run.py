"""FBX_SCALE stage 4A (user's rung-4 ruling; addendum 4) -- the hypergraph-aware multilevel partitioner ML_S on WebQSP graph S.

    fine H4_S hypergraph (frozen builder, cap round(N/K))
      -> clustering by _ml_coarsen (open twins, then agglomerative heavy-connectivity passes to <= fine pins / R)          [coarsening]
      -> quotient hypergraph with vertex weights = cluster sizes
      -> the frozen Zoltan-PHG driver with a vertex-weight input (phg_driver_vw.c: same calls and parameters, NP = 4)      [coarse partition]
      -> projection to the fine vertices
      -> the R3b closed-neighbourhood-KM1 label propagation on the fine graph (T4 / DMAX 5000 / <= 10 sweeps, unchanged)  [uncoarsening refinement]

  python -u scratchpad/_ml_run.py BUILD                 # compile phg_driver_vw on the host, write-once record
  python -u scratchpad/_ml_run.py REGR                  # vw driver == frozen driver on unit weights (fmt 1 and fmt 3), same manifest
  python -u scratchpad/_ml_run.py COARSEN               # the K-independent cluster map (write-once record)
  python -u scratchpad/_ml_run.py CELL <K> [<K> ...]    # quotient, PHG, projection, refinement per K (write-once records)
"""
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

import _l1h_host as H
import _ml_coarsen as C
import _fbx_cn as CN
import _fbx_part as FP

CM, FR, P, PR, HG = H.CM, H.FR, H.P, H.PR, H.HG
log, wj, sha_file, rel = H.log, H.wj, H.sha_file, H.rel
REPO = H.REPO
DS = "webqsp"
PARTS_S = os.path.join(REPO, "results", "FREEBASE_SCALE", "parts_S")
OUT = os.path.join(REPO, "results", "FREEBASE_SCALE", "ml")
KEYS = os.path.join(REPO, "data", "l1_canonical", DS, "keys.npz")
VW_SRC = os.path.join(P.SRC_DIR, "phg_driver_vw.c")
VW_REC = os.path.join(OUT, "ML_DRIVER_BUILD.json")
COARSE_PARAMS = {"Wmax": 32, "Lmax": 200, "R": 10.0, "MAXLEVELS": 8, "MIN_GAIN": 0.05, "twins": "open"}
CN_PARAMS = {"sweeps": 10, "T": 4, "dmax": 5000}
H.PDIR = PARTS_S
H.tag_of = lambda K: "H4_S_k%d" % int(K)


def graph_S():
    z = np.load(KEYS)
    N = int(z["N"][0])
    k = z["STRUCT"]
    a, b = k // N, k % N
    m = a != b
    k = np.unique(np.minimum(a[m], b[m]) * N + np.maximum(a[m], b[m]))
    return N, k


def peak_rss_mb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)
    except Exception:
        try:
            import resource
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, 1)
        except Exception:
            return None


def code_pins():
    fs = ["scratchpad/_ml_run.py", "scratchpad/_ml_coarsen.py", "scratchpad/_ml_agg.c", "scratchpad/_fbx_cn.py", "scratchpad/_fbx_cn.c", "scratchpad/_fbx_part.py",
          "scratchpad/_fbx_lp.py", "scratchpad/_l1h_host.py", "src/l1_lowmem/phg_driver/phg_driver_vw.c", "src/l1_lowmem/phg.py", "src/l1_canonical/hypergraph.py"]
    return {f: sha_file(os.path.join(REPO, f)) for f in fs}


# ----------------------------------------------------------------------------------------------------------------- BUILD
def cmd_build():
    os.makedirs(OUT, exist_ok=True)
    fro = H.host_build()
    if os.path.exists(VW_REC):
        b = vw_build()
        log("vw driver build unchanged (binary %s)" % b["binary"]["sha256"][:16])
        return b
    h = FR.home()
    bdir = H.BUILD_DIR_WSL.replace("~", h)
    FR.sh("cp %s %s/phg_driver_vw.c && sed -i 's/\\r$//' %s/phg_driver_vw.c" % (CM.wsl_path(VW_SRC), bdir, bdir), quiet=True)
    comp = H.COMPILE.replace("phg_driver.c", "phg_driver_vw.c").replace("-o phg_driver ", "-o phg_driver_vw ")
    r = FR.sh("cd %s && %s 2>&1; echo rc=$?" % (bdir, comp), quiet=True)
    if not r.stdout.strip().endswith("rc=0"):
        raise RuntimeError("vw driver build failed:\n%s" % r.stdout[-3000:])
    shas = FR.sh("cd %s && sha256sum phg_driver_vw phg_driver_vw.c phg_driver" % bdir, quiet=True).stdout.split()
    rec = {"RECORD": "ML_DRIVER_BUILD", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build_dir": bdir, "compile": comp,
           "compiler_output": r.stdout.strip().split("\n")[:-1], "source": {"path": rel(VW_SRC), "sha256": sha_file(VW_SRC)},
           "binary": {"path": "%s/phg_driver_vw" % bdir, "sha256": shas[0]}, "source_in_build_dir_sha256": shas[2],
           "frozen_driver": {"binary_sha256": fro["binary"]["sha256"], "still_equal": shas[4] == fro["binary"]["sha256"], "source_sha256": fro["source"]["sha256"]},
           "diff_vs_frozen_source": "manifest header 'N M 3' (leading vertex-weight token per shard line) and unit weight replaced by that weight in q_obj_list; nothing else",
           "placement": H.placement()}
    wj(VW_REC, rec)
    log("vw driver built: binary %s (frozen binary unchanged: %s)" % (shas[0][:16], rec["frozen_driver"]["still_equal"]))
    return vw_build()


def vw_build():
    rec = json.load(io.open(VW_REC, encoding="utf-8"))
    fro = H.host_build()
    chk = FR.sh("sha256sum %s" % rec["binary"]["path"], quiet=True).stdout.split()
    assert chk and chk[0] == rec["binary"]["sha256"], "vw driver binary changed since its record"
    assert sha_file(VW_SRC) == rec["source"]["sha256"], "phg_driver_vw.c changed since its record"
    return {"binary": rec["binary"], "wrapper": fro["wrapper"], "source": rec["source"]}


# ----------------------------------------------------------------------------------------------------------------- shards with vertex weights
def write_shards_vw(V, eptr, eidx, ew, vw, sdir, with_vw=True):
    """node-centric net lists (fmt 1) or, with_vw, fmt 3: each line '<vertex weight> <net> <w> <net> <w> ...' (nets 1-based ascending)."""
    os.makedirs(sdir, exist_ok=True)
    M = len(eptr) - 1
    hid = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    order = np.lexsort((hid, eidx))
    node_of, net_of = eidx[order], hid[order]
    deg = np.bincount(node_of, minlength=V)
    off = np.zeros(V + 1, np.int64)
    off[1:] = np.cumsum(deg)
    shards = []
    for i, a in enumerate(range(0, V, H.SHARD_NODES)):
        b = min(a + H.SHARD_NODES, V)
        fp = os.path.join(sdir, "shard_%05d.netl" % i)
        with io.open(fp, "w", encoding="ascii", newline="\n") as f:
            buf = []
            for j in range(a, b):
                seg = net_of[off[j]:off[j + 1]]
                pairs = np.empty(2 * len(seg), np.int64)
                pairs[0::2] = seg + 1
                pairs[1::2] = ew[seg]
                head = ("%d " % vw[j]) if with_vw else ""
                buf.append(head + " ".join(map(str, pairs.tolist())) + "\n" if len(seg) else (("%d" % vw[j]) if with_vw else "") + "\n")
            f.write("".join(buf))
        shards.append({"index": i, "file": fp, "nodes": b - a, "pins": int(off[b] - off[a]), "sha256": sha_file(fp)})
    smf = os.path.join(sdir, "stream_manifest.txt")
    with io.open(smf, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(["%d %d %d" % (V, M, 3 if with_vw else 1)] + ["%d %s" % (c["nodes"], CM.wsl_path(c["file"])) for c in shards]) + "\n")
    return shards, smf


# ----------------------------------------------------------------------------------------------------------------- COARSEN
def cluster_path():
    return os.path.join(OUT, "%s__ML_S__clusters.npy" % DS)


def cmd_coarsen():
    os.makedirs(OUT, exist_ok=True)
    out = cluster_path()
    runp = out[:-4] + ".RUN.json"
    assert not os.path.exists(runp), "write-once: %s exists" % runp
    t = time.time()
    N, k = graph_S()
    xa, aa = C.sorted_csr(k // N, k % N, N)
    log("graph S: N %d, %d pairs" % (N, len(k)))
    tmp = tempfile.mkdtemp(prefix="mlco_")
    cp = COARSE_PARAMS
    total, levels, (V, ep, ei, ew, vw) = C.coarsen(N, xa, aa, cp["Lmax"], cp["Wmax"], cp["R"], cp["MAXLEVELS"], tmp, min_gain=cp["MIN_GAIN"])
    cl = total.astype(np.int32)
    assert cl.max() + 1 == V and vw.sum() == N
    np.save(out[:-4] + ".tmp.npy", cl)
    os.replace(out[:-4] + ".tmp.npy", out)
    sizes = np.bincount(cl)
    rec = {"RECORD": "ML_S_CLUSTERS", "dataset": DS, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "params": cp, "N": N, "pairs": int(len(k)),
           "clusters": int(V), "cluster_size": {"max": int(sizes.max()), "mean": round(float(sizes.mean()), 3), "p50": float(np.median(sizes)),
                                                "singletons": int((sizes == 1).sum())},
           "levels": levels, "final_unsplit_hypergraph": {"V": int(V), "M": int(len(ew)), "P": int(len(ei)), "fine_P": levels[0]["P"],
                                                          "pin_reduction": round(levels[0]["P"] / max(len(ei), 1), 3)},
           "output": {"file": rel(out), "sha256": sha_file(out)}, "seconds": round(time.time() - t, 1), "peak_rss_mb": peak_rss_mb(),
           "placement": H.placement(), "code": code_pins()}
    wj(runp, rec)
    log("clusters %d (N/V %.2f), unsplit pins %d -> %d (%.2fx), %.0fs" % (V, N / V, levels[0]["P"], len(ei), rec["final_unsplit_hypergraph"]["pin_reduction"], rec["seconds"]))


# ----------------------------------------------------------------------------------------------------------------- CELL
def blocks_per_net(eptr, eidx, part):
    M = len(eptr) - 1
    net = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    k = int(part.max()) + 1
    key = np.unique(net * k + part[eidx])
    lam = np.bincount(key // k, minlength=M)
    return float(lam.mean())


def cmd_cell(Ks):
    os.makedirs(OUT, exist_ok=True)
    b = vw_build()
    cl = np.load(cluster_path()).astype(np.int64)
    crec = json.load(io.open(cluster_path()[:-4] + ".RUN.json", encoding="utf-8"))
    assert sha_file(cluster_path()) == crec["output"]["sha256"]
    N, k = graph_S()
    xa, aa = C.sorted_csr(k // N, k % N, N)
    stem = os.path.join(tempfile.mkdtemp(prefix="mlcn_"), "s")
    streams = FP.write_csr_raw(xa, aa, stem)
    for K in Ks:
        out = os.path.join(OUT, "%s__ML_S_k%d.npy" % (DS, K))
        runp = out[:-4] + ".RUN.json"
        failp = out[:-4] + ".FAILED.json"
        if os.path.exists(runp) or os.path.exists(failp):
            log("K %d: exists" % K)
            continue
        t0 = time.time()
        Hf = H.load_hg(DS, K)
        assert Hf["N"] == N and Hf["k"] == K
        Vc, ep, ei, ew, vw, st = C.quotient(N, Hf["eptr"], Hf["eidx"], Hf["ew"], np.ones(N, np.int64), cl)
        log("K %d: fine M %d P %d -> coarse V %d M %d P %d (%.2fx pins); nets dropped %d, merged %d" % (K, Hf["M"], Hf["P"], Vc, st["M"], st["P"], Hf["P"] / max(st["P"], 1),
                                                                                                   st["nets_dropped_singleton"], st["nets_merged_identical"]))
        tag = "ML_S_k%d" % K
        sdir = os.path.join(CM.DATA, DS, "stream_%s" % tag)
        shards, smf = write_shards_vw(Vc, ep, ei, ew, vw, sdir, with_vw=True)
        log("K %d: %d shards written (%.1fs)" % (K, len(shards), time.time() - t0))
        r1, d1 = P.mpirun(DS, "R1_%s" % tag, "partition", K, smf, b, timeout=H.PHG_TIMEOUT_S)
        gq = r1["global_from_queries"]
        assert gq["objects"] == Vc and gq["pins"] == st["P"], ("Zoltan saw a different hypergraph", gq)
        pc = P.assemble_partition(d1, Vc, K).astype(np.int64)
        coarse_km1 = C.km1(ep, ei, ew, pc)
        zc = float(r1["zoltan_eval"]["cutl_global"])
        proj = pc[cl]
        m_proj = FR.km1_metrics(Hf, proj)
        assert m_proj["km1_weighted"] == coarse_km1, ("quotient identity broken", coarse_km1, m_proj["km1_weighted"])
        rec = {"RECORD": "ML_S_CELL", "dataset": DS, "tag": tag, "K": K, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "params": COARSE_PARAMS, "cn_params": CN_PARAMS,
               "clusters": {"file": crec["output"]["file"], "sha256": crec["output"]["sha256"], "run_json_sha256": sha_file(cluster_path()[:-4] + ".RUN.json")},
               "fine_hypergraph": {"file": rel(Hf["npz"]), "sha256": Hf["npz_sha256"], "N": N, "M": Hf["M"], "P": Hf["P"]},
               "coarse_hypergraph": dict(st, pin_reduction=round(Hf["P"] / max(st["P"], 1), 3)),
               "driver": {"binary": b["binary"], "source_sha256": b["source"]["sha256"], "parameters": P.PARAMS, "NP": P.NP},
               "phg_coarse": {kk: r1[kk] for kk in ("name", "mode", "np", "rc", "wall_seconds_outer", "memory", "timing", "zoltan_eval", "zoltan_removed_or_warning_lines")},
               "coarse_km1": coarse_km1, "zoltan_cutl_vs_python_km1_rel": round(abs(zc - coarse_km1) / max(coarse_km1, 1), 8)}
        sizes = np.bincount(proj, minlength=K)
        if (sizes == 0).any():
            hard2, empties, moves = PR.repair(dict(Hf), proj)
            rec["repair"] = {"empty_blocks": empties, "moves": len(moves)}
            proj = hard2.astype(np.int64)
            m_proj = FR.km1_metrics(Hf, proj)
        else:
            rec["repair"] = None
        v0 = P.validity(proj, N, K)
        rec["projection"] = {"validity": v0, "km1_weighted": m_proj["km1_weighted"], "lambda_mean": m_proj["lambda_mean"], "blocks": m_proj["blocks"],
                             "blocks_per_net_unweighted": round(blocks_per_net(Hf["eptr"], Hf["eidx"], proj), 4)}
        if not (v0["length_ok"] and v0["ids_in_range"] and v0["max_block"] <= v0["contract_bound_ceil_1.03_N_over_k"]):
            rec.update({"STATUS": "PARTITION_INVALID", "wall_seconds": round(time.time() - t0, 1)})
            wj(failp, rec)
            log("K %d: PARTITION_INVALID after projection (max %s bound %s)" % (K, v0["max_block"], v0["contract_bound_ceil_1.03_N_over_k"]))
            continue
        np.save(out[:-4] + "_proj.npy", proj.astype(np.int32))
        init_raw = os.path.join(os.path.dirname(stem), "init_%d.raw" % K)
        proj.astype(np.int32).tofile(init_raw)
        t1 = time.time()
        lab, info = CN.cn_c(N, K, CN_PARAMS["sweeps"], CN_PARAMS["T"], CN_PARAMS["dmax"], init_raw, os.path.join(os.path.dirname(stem), "out_%d.raw" % K), streams)
        hard = lab.astype(np.int64)
        m_fin = FR.km1_metrics(Hf, hard)
        v1 = P.validity(hard, N, K)
        bal = H.balance(hard, N, K)
        rec["refinement"] = {"cn_info": info, "seconds": round(time.time() - t1, 1), "validity": v1, "km1_weighted": m_fin["km1_weighted"], "lambda_mean": m_fin["lambda_mean"],
                             "blocks": m_fin["blocks"], "blocks_per_net_unweighted": round(blocks_per_net(Hf["eptr"], Hf["eidx"], hard), 4)}
        if not (v1["length_ok"] and v1["ids_in_range"] and v1["max_block"] <= v1["contract_bound_ceil_1.03_N_over_k"]) or not bal["every_block_used"]:
            rec.update({"STATUS": "PARTITION_INVALID", "wall_seconds": round(time.time() - t0, 1)})
            wj(failp, rec)
            log("K %d: PARTITION_INVALID after refinement" % K)
            continue
        np.save(out[:-4] + ".tmp.npy", hard.astype(np.int32))
        os.replace(out[:-4] + ".tmp.npy", out)
        rec.update({"STATUS": "OK", "balance": bal, "wall_seconds": round(time.time() - t0, 1), "peak_rss_mb_python": peak_rss_mb(),
                    "output": {"file": rel(out), "sha256": sha_file(out), "n": int(len(hard))}, "projection_map": {"file": rel(out[:-4] + "_proj.npy"),
                                                                                                                 "sha256": sha_file(out[:-4] + "_proj.npy")},
                    "placement": H.placement(), "code": code_pins()})
        wj(runp, rec)
        log("K %d: ML_S done %.0fs; KM1 projected %d -> refined %d; blocks/net %.3f -> %.3f; sizes %d..%d" % (
            K, rec["wall_seconds"], m_proj["km1_weighted"], m_fin["km1_weighted"], rec["projection"]["blocks_per_net_unweighted"],
            rec["refinement"]["blocks_per_net_unweighted"], bal["size_min"], bal["size_max"]))


# ----------------------------------------------------------------------------------------------------------------- REGR
def cmd_regr(K=100):
    """the vw driver on a unit-weight manifest must reproduce the frozen driver's partition exactly (fmt 1 through both binaries; fmt 3 with all weights 1 through the vw one)."""
    os.makedirs(OUT, exist_ok=True)
    rp = os.path.join(OUT, "ML_DRIVER_REGRESSION.json")
    assert not os.path.exists(rp), "write-once: %s exists" % rp
    bf, bv = H.host_build(), vw_build()
    cl = np.load(cluster_path()).astype(np.int64)
    N, _ = graph_S()
    Hf = H.load_hg(DS, K)
    Vc, ep, ei, ew, vw, st = C.quotient(N, Hf["eptr"], Hf["eidx"], Hf["ew"], np.ones(N, np.int64), cl)
    ones = np.ones(Vc, np.int64)
    res = {}
    for name, with_vw, b in (("frozen_fmt1", False, bf), ("vw_fmt1", False, bv), ("vw_fmt3_unit", True, bv)):
        sdir = os.path.join(CM.DATA, DS, "stream_MLREGR_%s" % name)
        shards, smf = write_shards_vw(Vc, ep, ei, ew, ones, sdir, with_vw=with_vw)
        r, d = P.mpirun(DS, "MLREGR_%s" % name, "partition", K, smf, b, timeout=H.PHG_TIMEOUT_S)
        res[name] = {"part": P.assemble_partition(d, Vc, K), "cutl": r["zoltan_eval"]["cutl_global"], "wall": r["wall_seconds_outer"]}
    same1 = bool(np.array_equal(res["frozen_fmt1"]["part"], res["vw_fmt1"]["part"]))
    same3 = bool(np.array_equal(res["frozen_fmt1"]["part"], res["vw_fmt3_unit"]["part"]))
    rec = {"RECORD": "ML_DRIVER_REGRESSION", "K": K, "coarse": {"V": Vc, "M": st["M"], "P": st["P"]}, "vw_fmt1_equals_frozen": same1, "vw_fmt3_unit_equals_frozen": same3,
           "cutl": {k: v["cutl"] for k, v in res.items()}, "wall_seconds": {k: v["wall"] for k, v in res.items()}, "STATUS": "PASS" if same1 and same3 else "FAIL",
           "binaries": {"frozen": bf["binary"]["sha256"], "vw": bv["binary"]["sha256"]}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": code_pins()}
    wj(rp, rec)
    log("REGR %s: vw fmt1 == frozen %s, vw fmt3(unit) == frozen %s" % (rec["STATUS"], same1, same3))
    assert rec["STATUS"] == "PASS"


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if not a:
        raise SystemExit(__doc__)
    log("placement:", json.dumps(H.placement()))
    if a[0] == "BUILD":
        cmd_build()
    elif a[0] == "REGR":
        cmd_regr()
    elif a[0] == "COARSEN":
        cmd_coarsen()
    elif a[0] == "CELL":
        cmd_cell([int(x) for x in a[1:]])
    else:
        raise SystemExit(__doc__)
