"""FBX_SCALE stage 4C -- the SK-arm multilevel lab on WebQSP, step 2: one candidate clustering -> one partition per K.

  python -u scratchpad/_ml2_cell.py CELL <M1|M2|M3|M4> <Wmax> <level> <K>[,<K>...]

A candidate is a K-independent clustering of the frozen H4_SK nets (results/FREEBASE_SCALE/ml2/clusters__<method>_W<Wmax>_L<level>.npy, written by _ml2_run.py COARSEN).  A cell is:
  frozen H4_SK hypergraph at K (rebuilt from keys.npz, its content digest checked against the frozen record)  ->  exact quotient under the clustering  ->  the frozen Zoltan PHG
  driver (vertex-weighted build, NP = 4, the manifest header 'N M 3'; regression record PASS)  ->  projection  ->  the empty-block repair of the frozen contract (phg_repair)
  ->  the shared refinement _ml2_cn (balance repair, then closed-neighbourhood KM1 label propagation on the two families as two layers)  ->  validity gate.
The refinement and the gate are the same for every candidate and are declared before scoring (addendum 5).  Nothing here reads a gold label.
Every record is write-once under results/FREEBASE_SCALE/ml2/.
"""
import io
import json
import os
import sys
import tempfile
import time

import numpy as np

import _l1h_host as H
import _ml_run as MR                      # its module-level H.PDIR / H.tag_of re-pointing is undone below
import _ml_coarsen as C
import _ml2_cn as CN2
import _ml2_run as R2
import _fbx_part as FP

H.PDIR = os.path.join(H.REPO, "results", "L1_HOST", "parts")
H.tag_of = lambda K: "H4_SK_k%d" % int(K)
CM, FR, P, PR, HG = H.CM, H.FR, H.P, H.PR, H.HG
log, wj, sha_file, rel = H.log, H.wj, H.sha_file, H.rel
REPO = H.REPO
DS = "webqsp"
OUT = R2.OUT
KEYS = R2.KEYS
REFINE = {"sweeps": 10, "T": 4, "dmax": 5000, "layers": ["STRUCT", "KNN"], "anchor_weights": "max(1, rint(1000 / deg_family(u))) = the frozen H4 net weight of the anchor's closed neighbourhood",
          "repair": "cheapest-first out of every block above ceil(1.03 N / K) before the sweeps"}


def code_pins():
    fs = ["scratchpad/_ml2_cell.py", "scratchpad/_ml2_cn.py", "scratchpad/_ml2_cn.c", "scratchpad/_ml2_run.py", "scratchpad/_ml2_coarsen.py", "scratchpad/_ml_run.py",
          "scratchpad/_ml_coarsen.py", "scratchpad/_fbx_part.py", "scratchpad/_l1h_host.py", "src/l1_lowmem/phg_driver/phg_driver_vw.c", "src/l1_lowmem/phg.py",
          "src/l1_canonical/hypergraph.py"]
    return {f: sha_file(os.path.join(REPO, f)) for f in fs}


def hg_sk(K, keys, N):
    """the frozen H4_SK hypergraph at K, rebuilt and checked against its frozen record (content digest); returns the dict of _l1h_host.load_hg."""
    fm = os.path.join(H.PDIR, "webqsp__H4_SK_k%d.json" % K)
    meta = json.load(io.open(fm, encoding="utf-8"))
    t = time.time()
    arrays, m2 = HG.build_hypergraph(N, keys, K, famset="SK", tag=DS)
    dig = HG.arrays_digest(arrays)
    assert dig == meta["content_digest"], "rebuilt H4_SK k=%d differs from its frozen record (%s vs %s)" % (K, dig[:16], meta["content_digest"][:16])
    log("K %d: H4_SK rebuilt and equal to the frozen record (digest %s, %.0fs)" % (K, dig[:16], time.time() - t))
    eptr, eidx, ew = arrays["eptr"].astype(np.int64), arrays["eidx"].astype(np.int64), arrays["ew"].astype(np.int64)
    return {"npz": meta["file"], "npz_sha256": meta["file_sha256"], "meta": meta, "eptr": eptr, "eidx": eidx, "ew": ew, "N": int(N), "k": int(K), "M": int(len(eptr) - 1),
            "P": int(len(eidx)), "content_digest": dig}


def cmd_cell(method, W, level, Ks):
    os.makedirs(OUT, exist_ok=True)
    tagc = "%s_W%d_L%d" % (method, W, level)
    cp = R2.map_path(method, W, level)
    crec = json.load(io.open(R2.rec_path(method, W), encoding="utf-8"))
    cmap = next(m for m in crec["maps"] if m["level"] == level)
    assert sha_file(cp) == cmap["sha256"], "cluster map differs from its COARSEN record"
    cl = np.load(cp).astype(np.int64)
    b = MR.vw_build()
    z = np.load(KEYS)
    N = int(z["N"][0])
    keys = {"STRUCT": z["STRUCT"], "KNN": z["KNN"]}
    _, fams, cnt = R2.graph_SK()
    tmpd = tempfile.mkdtemp(prefix="ml2cn_")
    streams, wspecs = [], []
    for i, (xa, aa) in enumerate(fams):
        streams += FP.write_csr_raw(xa, aa, os.path.join(tmpd, "L%d" % i))
        wspecs.append(CN2.write_weights(CN2.anchor_weights(xa), os.path.join(tmpd, "W%d.i32" % i)))
    for K in Ks:
        out = os.path.join(OUT, "webqsp__%s_k%d.npy" % (tagc, K))
        runp, failp = out[:-4] + ".RUN.json", out[:-4] + ".FAILED.json"
        if os.path.exists(runp) or os.path.exists(failp):
            log("%s K %d: exists" % (tagc, K))
            continue
        t0 = time.time()
        Hf = hg_sk(K, keys, N)
        Vc, ep, ei, ew, vw, st = C.quotient(N, Hf["eptr"], Hf["eidx"], Hf["ew"], np.ones(N, np.int64), cl)
        log("%s K %d: fine M %d P %d -> coarse V %d M %d P %d (%.3fx pins); nets dropped %d, merged %d" % (tagc, K, Hf["M"], Hf["P"], Vc, st["M"], st["P"], Hf["P"] / max(st["P"], 1),
                                                                                                          st["nets_dropped_singleton"], st["nets_merged_identical"]))
        tag = "ML2_%s_k%d" % (tagc, K)
        sdir = os.path.join(CM.DATA, DS, "stream_%s" % tag)
        shards, smf = MR.write_shards_vw(Vc, ep, ei, ew, vw, sdir, with_vw=True)
        r1, d1 = P.mpirun(DS, "R1_%s" % tag, "partition", K, smf, b, timeout=H.PHG_TIMEOUT_S)
        gq = r1["global_from_queries"]
        assert gq["objects"] == Vc and gq["pins"] == st["P"], ("Zoltan saw a different hypergraph", gq)
        pc = P.assemble_partition(d1, Vc, K).astype(np.int64)
        coarse_km1 = C.km1(ep, ei, ew, pc)
        proj = pc[cl]
        m_proj = FR.km1_metrics(Hf, proj)
        assert m_proj["km1_weighted"] == coarse_km1, ("quotient identity broken", coarse_km1, m_proj["km1_weighted"])
        rec = {"RECORD": "ML2_CELL", "dataset": DS, "candidate": tagc, "method": method, "rating": R2.C2.METHODS[method][0], "schedule": R2.C2.METHODS[method][1], "Wmax": W, "level": level,
               "K": K, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "refine_params": REFINE,
               "clusters": {"file": cmap["file"], "sha256": cmap["sha256"], "n_clusters": cmap["clusters"], "cluster_size": cmap["cluster_size"], "coarsen_record_sha256": sha_file(R2.rec_path(method, W))},
               "fine_hypergraph": {"file": Hf["npz"], "npz_sha256": Hf["npz_sha256"], "content_digest": Hf["content_digest"], "N": N, "M": Hf["M"], "P": Hf["P"]},
               "coarse_hypergraph": dict(st, pin_reduction=round(Hf["P"] / max(st["P"], 1), 4), vertex_weight_max=int(vw.max())),
               "driver": {"binary": b["binary"], "source_sha256": b["source"]["sha256"], "parameters": P.PARAMS, "NP": P.NP},
               "phg_coarse": {kk: r1[kk] for kk in ("name", "mode", "np", "rc", "wall_seconds_outer", "memory", "timing", "zoltan_eval", "zoltan_removed_or_warning_lines")},
               "coarse_km1": coarse_km1}
        sizes = np.bincount(proj, minlength=K)
        if (sizes == 0).any():
            hard2, empties, moves = PR.repair(dict(Hf), proj)
            rec["empty_block_repair"] = {"empty_blocks": empties, "moves": len(moves)}
            proj = hard2.astype(np.int64)
            m_proj = FR.km1_metrics(Hf, proj)
        else:
            rec["empty_block_repair"] = None
        v0 = P.validity(proj, N, K)
        rec["projection"] = {"validity": v0, "km1_weighted": m_proj["km1_weighted"], "lambda_mean": m_proj["lambda_mean"], "blocks": m_proj["blocks"],
                             "blocks_per_net_unweighted": round(MR.blocks_per_net(Hf["eptr"], Hf["eidx"], proj), 4)}
        np.save(out[:-4] + "_proj.npy", proj.astype(np.int32))
        init_raw = os.path.join(tmpd, "init_%d.raw" % K)
        proj.astype(np.int32).tofile(init_raw)
        t1 = time.time()
        lab, info = CN2.cn2_c(N, K, REFINE["sweeps"], REFINE["T"], REFINE["dmax"], init_raw, os.path.join(tmpd, "out_%d.raw" % K), streams, wspecs)
        hard = lab.astype(np.int64)
        m_fin = FR.km1_metrics(Hf, hard)
        v1 = P.validity(hard, N, K)
        bal = H.balance(hard, N, K)
        rec["refinement"] = {"cn_info": info, "seconds": round(time.time() - t1, 1), "validity": v1, "km1_weighted": m_fin["km1_weighted"], "lambda_mean": m_fin["lambda_mean"],
                             "blocks": m_fin["blocks"], "blocks_per_net_unweighted": round(MR.blocks_per_net(Hf["eptr"], Hf["eidx"], hard), 4)}
        if v1["gate"] != "PASS" or not bal["every_block_used"]:
            rec.update({"STATUS": "PARTITION_INVALID", "balance": bal, "wall_seconds": round(time.time() - t0, 1)})
            wj(failp, rec)
            log("%s K %d: PARTITION_INVALID (max %s bound %s empty %s)" % (tagc, K, v1["max_block"], v1["contract_bound_ceil_1.03_N_over_k"], v1["empty_blocks"]))
            continue
        np.save(out[:-4] + ".tmp.npy", hard.astype(np.int32))
        os.replace(out[:-4] + ".tmp.npy", out)
        rec.update({"STATUS": "OK", "balance": bal, "wall_seconds": round(time.time() - t0, 1), "peak_rss_mb_python": MR.peak_rss_mb(),
                    "output": {"file": rel(out), "sha256": sha_file(out), "n": int(len(hard))},
                    "projection_map": {"file": rel(out[:-4] + "_proj.npy"), "sha256": sha_file(out[:-4] + "_proj.npy")}, "placement": H.placement(), "code": code_pins()})
        wj(runp, rec)
        log("%s K %d: done %.0fs; pin reduction %.3fx; KM1 projected %d -> refined %d; blocks/net %.3f -> %.3f; repair moves %d; sizes %d..%d" % (
            tagc, K, rec["wall_seconds"], rec["coarse_hypergraph"]["pin_reduction"], m_proj["km1_weighted"], m_fin["km1_weighted"], rec["projection"]["blocks_per_net_unweighted"],
            rec["refinement"]["blocks_per_net_unweighted"], info["repair"]["moves"], bal["size_min"], bal["size_max"]))


def cmd_m0r(Ks):
    """control: the frozen PHG_SK map at K put through the SAME shared refinement (no clustering, no coarse PHG) -- separates the effect of the refinement from the effect of the coarsening."""
    os.makedirs(OUT, exist_ok=True)
    z = np.load(KEYS)
    N = int(z["N"][0])
    keys = {"STRUCT": z["STRUCT"], "KNN": z["KNN"]}
    _, fams, cnt = R2.graph_SK()
    tmpd = tempfile.mkdtemp(prefix="ml2m0r_")
    streams, wspecs = [], []
    for i, (xa, aa) in enumerate(fams):
        streams += FP.write_csr_raw(xa, aa, os.path.join(tmpd, "L%d" % i))
        wspecs.append(CN2.write_weights(CN2.anchor_weights(xa), os.path.join(tmpd, "W%d.i32" % i)))
    for K in Ks:
        out = os.path.join(OUT, "webqsp__M0R_k%d.npy" % K)
        runp = out[:-4] + ".RUN.json"
        if os.path.exists(runp):
            log("M0R K %d: exists" % K)
            continue
        t0 = time.time()
        Hf = hg_sk(K, keys, N)
        Rp = json.load(io.open(os.path.join(H.PDIR, "webqsp__H4_SK_k%d__PHG_con.RUN.json" % K), encoding="utf-8"))
        mp = os.path.join(REPO, Rp["output"]["file"])
        assert sha_file(mp) == Rp["output"]["sha256"] and Rp["STATUS"] == "OK"
        init = np.load(mp).astype(np.int64)
        m0 = FR.km1_metrics(Hf, init)
        init_raw = os.path.join(tmpd, "init_%d.raw" % K)
        init.astype(np.int32).tofile(init_raw)
        lab, info = CN2.cn2_c(N, K, REFINE["sweeps"], REFINE["T"], REFINE["dmax"], init_raw, os.path.join(tmpd, "out_%d.raw" % K), streams, wspecs)
        hard = lab.astype(np.int64)
        m1 = FR.km1_metrics(Hf, hard)
        v1 = P.validity(hard, N, K)
        bal = H.balance(hard, N, K)
        rec = {"RECORD": "ML2_M0R", "dataset": DS, "candidate": "M0R", "K": K, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "refine_params": REFINE,
               "input_map": {"file": Rp["output"]["file"], "sha256": Rp["output"]["sha256"]}, "fine_hypergraph": {"file": Hf["npz"], "content_digest": Hf["content_digest"], "M": Hf["M"], "P": Hf["P"]},
               "km1_weighted_before": m0["km1_weighted"], "km1_weighted_after": m1["km1_weighted"], "lambda_mean_before": m0["lambda_mean"], "lambda_mean_after": m1["lambda_mean"],
               "blocks_per_net_unweighted_before": round(MR.blocks_per_net(Hf["eptr"], Hf["eidx"], init), 4), "blocks_per_net_unweighted_after": round(MR.blocks_per_net(Hf["eptr"], Hf["eidx"], hard), 4),
               "refinement": {"cn_info": info, "validity": v1}, "balance": bal, "wall_seconds": round(time.time() - t0, 1)}
        if v1["gate"] != "PASS" or not bal["every_block_used"]:
            rec["STATUS"] = "PARTITION_INVALID"
            wj(out[:-4] + ".FAILED.json", rec)
            log("M0R K %d: PARTITION_INVALID" % K)
            continue
        np.save(out[:-4] + ".tmp.npy", hard.astype(np.int32))
        os.replace(out[:-4] + ".tmp.npy", out)
        rec.update({"STATUS": "OK", "output": {"file": rel(out), "sha256": sha_file(out), "n": int(len(hard))}, "placement": H.placement(), "code": code_pins()})
        wj(runp, rec)
        log("M0R K %d: weighted KM1 %d -> %d (%.2f %%); blocks/net %.4f -> %.4f; sizes %d..%d; %d sweeps" % (
            K, m0["km1_weighted"], m1["km1_weighted"], 100.0 * (m1["km1_weighted"] - m0["km1_weighted"]) / m0["km1_weighted"], rec["blocks_per_net_unweighted_before"],
            rec["blocks_per_net_unweighted_after"], bal["size_min"], bal["size_max"], info["sweeps_run"]))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 5 and a[0] == "CELL":
        cmd_cell(a[1], int(a[2]), int(a[3]), [int(x) for x in a[4].split(",")])
    elif len(a) == 2 and a[0] == "M0R":
        cmd_m0r([int(x) for x in a[1].split(",")])
    else:
        raise SystemExit(__doc__)
