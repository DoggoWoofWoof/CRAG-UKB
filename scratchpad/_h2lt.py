"""FBX_SCALE Track B -- a SURROGATE for the TOP level of the nested partition (WebQSP, in memory; draft for addendum 9, not yet pinned).

The nested partition (scratchpad/_h2l.py) bounds the K-way stage by the largest top block, but its TOP stage still reads every pin of H4_SK at k = S.  The top only has to cut the vertex set
into S large blocks, so it can be computed on a smaller hypergraph than the one the K-way stage sees.  A top surrogate is a spec name:
  N<c>              drop every net of more than c pins from the top hypergraph (hub nets carry many pins and little weight: weight = max(1, rint(1000 / deg)))
  Q<W>L<l>          quotient the top hypergraph by the frozen M2 clustering (results/FREEBASE_SCALE/ml2/clusters__M2_W<W>_L<l>.npy), vertex weights = cluster sizes
  Q<W>L<l>N<c>      first the net cap, then the quotient
The surrogate hypergraph is partitioned at k = S by the frozen vertex-weighted driver (NP 4, IMBALANCE_TOL 1.01, as the flat top of addendum 7), projected to the vertices, judged by the frozen validity
rule (largest block <= ceil(1.03 N / S), every block used after the frozen empty-block repair) on the FULL top hypergraph, and handed to _h2l.cmd_sub unchanged (net-split K-way sub-problems, tolerance
floor_4dp(1.03 (N/K)(K/S)/n_b)).  No refinement.  Nothing here reads a gold label.

  python -u scratchpad/_h2lt.py TOP <spec> <S>[,<S>...]          # write-once map + RUN.json:  results/FREEBASE_SCALE/h2l/parts/webqsp__H4_H2LT_<spec>_top_k<S>__PHG_con.*
  python -u scratchpad/_h2lt.py TEST                             # net cap / quotient identities on a random hypergraph (no Zoltan)
  python -u scratchpad/_h2lt.py SUB <spec> <S> <K> [<K> ...]     # the two-level K-way map from that top:  ...webqsp__H4_H2LT_<spec>_<S>_k<K>__PHG_con.*   (scored as arm 'H2LT_<spec>_<S>')
"""
import io
import json
import os
import re
import shutil
import sys
import time

import numpy as np

import _ml_run as MR                      # re-points H.PDIR / H.tag_of at import; _h2l (next) and this module's overrides below re-point them again
import _ml_coarsen as C
import _h2l as L
import _l1h_host as H

P, FR, PR, CM = H.P, H.FR, H.PR, H.CM
DS = "webqsp"
OUT = L.OUT
ML2 = os.path.join(H.REPO, "results", "FREEBASE_SCALE", "ml2")
SPEC = re.compile(r"(?:Q(\d+)L(\d+))?(?:N(\d+))?")
CUR = {"spec": None}


def parse(name):
    m = SPEC.fullmatch(name)
    assert name and m, "bad spec %r" % name
    W, lv, c = m.groups()
    return {"name": name, "W": int(W) if W else None, "level": int(lv) if lv else None, "cap": int(c) if c else None}


def tag_of(K):
    if L.MODE["top"]:
        return "H4_H2LT_%s_top_k%d" % (CUR["spec"]["name"], int(K))
    return "H4_SK_k%d" % int(K)


def arm_files(S, K):
    base = os.path.join(OUT, "%s__H4_H2LT_%s_%d_k%d__PHG_con" % (DS, CUR["spec"]["name"], S, K))
    return base + ".npy", base + ".RUN.json", base + ".FAILED.json"


H.tag_of = tag_of
L.arm_files = arm_files


def load_top_hg(S):
    """the addendum-7 top hypergraph H4_SK at k = S (frozen rule, cap = round(N / S)); built by _h2l.py TOP if absent, its file hash checked against its record."""
    saved = H.tag_of
    H.tag_of = L.tag_of
    L.MODE["top"] = True
    try:
        H.cmd_hg(DS, [S])
    finally:
        L.MODE["top"] = False
        H.tag_of = saved
    npz = os.path.join(OUT, "%s__H4_H2LTOP_k%d.npz" % (DS, S))
    meta = json.load(io.open(npz[:-4] + ".json", encoding="utf-8"))
    assert H.sha_file(npz) == meta["file_sha256"], "top hypergraph changed since its record"
    z = np.load(npz)
    eptr, eidx, ew = z["eptr"].astype(np.int64), z["eidx"].astype(np.int64), z["ew"].astype(np.int64)
    return {"npz": npz, "npz_sha256": meta["file_sha256"], "meta": meta, "eptr": eptr, "eidx": eidx, "ew": ew, "N": int(z["N"][0]), "k": int(z["k"][0]), "M": int(len(eptr) - 1), "P": int(len(eidx))}


def cap_nets(eptr, eidx, ew, N, c):
    """drop nets of more than c pins; returns the kept hypergraph and its statistics."""
    sz = np.diff(eptr)
    keep = sz <= c
    hid = np.repeat(np.arange(len(sz), dtype=np.int64), sz)
    m = keep[hid]
    ep = np.zeros(int(keep.sum()) + 1, np.int64)
    np.cumsum(sz[keep], out=ep[1:])
    ei, ewk = eidx[m], ew[keep]
    sig_all = float((ew * (sz - 1)).sum())
    sig_kept = float((ewk * (sz[keep] - 1)).sum())
    st = {"cap": int(c), "nets_kept": int(keep.sum()), "nets_dropped": int((~keep).sum()), "pins_kept": int(len(ei)), "pin_share_kept": round(len(ei) / max(len(eidx), 1), 4),
          "signal_share_kept": round(sig_kept / max(sig_all, 1.0), 4), "vertices_without_a_net": int((np.bincount(ei, minlength=N) == 0).sum())}
    return ep, ei, ewk, st


def cluster_map(W, level):
    rec_p = os.path.join(ML2, "COARSEN__M2_W%d.json" % W)
    cp = os.path.join(ML2, "clusters__M2_W%d_L%d.npy" % (W, level))
    cm = next(m for m in json.load(io.open(rec_p, encoding="utf-8"))["maps"] if m["level"] == level)
    assert H.sha_file(cp) == cm["sha256"], "cluster map differs from its COARSEN record"
    return np.load(cp).astype(np.int64), cm, rec_p


def cmd_top(name, S):
    spec = parse(name)
    CUR["spec"] = spec
    L.MODE["top"] = True
    tp = H.phg_npy(DS, S)
    L.MODE["top"] = False
    runp, failp = tp[:-4] + ".RUN.json", tp[:-4] + ".FAILED.json"
    if os.path.exists(runp) or os.path.exists(failp):
        H.log("top %s S %d: exists" % (name, S))
        return
    t0 = time.time()
    Hf = load_top_hg(S)
    N = Hf["N"]
    eptr, eidx, ew = Hf["eptr"], Hf["eidx"], Hf["ew"]
    rec = {"RECORD": "H2LT_TOP", "dataset": DS, "spec": spec, "S": S, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "top_hypergraph": {"file": H.rel(Hf["npz"]), "sha256": Hf["npz_sha256"], "content_digest": Hf["meta"]["content_digest"], "N": N, "M": Hf["M"], "P": Hf["P"]}}
    if spec["cap"]:
        eptr, eidx, ew, st = cap_nets(eptr, eidx, ew, N, spec["cap"])
        rec["net_cap"] = st
        H.log("top %s S %d: net cap %d -> %d nets, %d pins (%.3f of pins, %.3f of signal), %d vertices without a net" % (
            name, S, spec["cap"], st["nets_kept"], st["pins_kept"], st["pin_share_kept"], st["signal_share_kept"], st["vertices_without_a_net"]))
    cl = None
    if spec["W"]:
        cl, cm, rec_p = cluster_map(spec["W"], spec["level"])
        Vc, ep, ei, ewq, vw, qs = C.quotient(N, eptr, eidx, ew, np.ones(N, np.int64), cl)
        rec["quotient"] = dict(qs, clusters={"file": cm["file"], "sha256": cm["sha256"], "n_clusters": cm["clusters"], "coarsen_record_sha256": H.sha_file(rec_p)}, vertex_weight_max=int(vw.max()))
    else:
        Vc, ep, ei, ewq, vw = N, eptr, eidx, ew, np.ones(N, np.int64)
    rec["surrogate"] = {"V": int(Vc), "M": int(len(ewq)), "P": int(len(ei)), "pin_reduction_vs_top": round(Hf["P"] / max(len(ei), 1), 4)}
    H.log("top %s S %d: surrogate V %d M %d P %d (%.3fx fewer pins than the top hypergraph)" % (name, S, Vc, len(ewq), len(ei), rec["surrogate"]["pin_reduction_vs_top"]))
    b = MR.vw_build()
    sdir = os.path.join(CM.DATA, DS, "stream_H2LT_%s_top_k%d" % (name, S))
    base = list(P.PARAMS)
    P.PARAMS = [(k, L.TOP_TOL if k == "IMBALANCE_TOL" else v) for k, v in base]
    try:
        shards, smf = MR.write_shards_vw(Vc, ep, ei, ewq, vw, sdir, with_vw=True)
        r1, d1 = P.mpirun(DS, "R1_H2LT_%s_%d" % (name, S), "partition", S, smf, b, timeout=H.PHG_TIMEOUT_S)
    finally:
        P.PARAMS = base
    gq = r1["global_from_queries"]
    assert gq["objects"] == Vc and gq["pins"] == len(ei), ("Zoltan saw a different hypergraph", gq)
    pc = P.assemble_partition(d1, Vc, S).astype(np.int64)
    hard = pc[cl] if cl is not None else pc
    v = P.validity(hard, N, S)
    m = FR.km1_metrics(Hf, hard)
    flat = os.path.join(OUT, "%s__H4_H2LTOP_k%d__PHG_con.RUN.json" % (DS, S))
    flat_km1 = json.load(io.open(flat, encoding="utf-8"))["raw"]["km1"] if os.path.exists(flat) else None
    rec.update({"driver": {"binary": b["binary"], "source_sha256": b["source"]["sha256"], "parameters_top": [(k, L.TOP_TOL if k == "IMBALANCE_TOL" else vv) for k, vv in base], "NP": P.NP},
                "phg": {kk: r1[kk] for kk in ("name", "mode", "np", "rc", "wall_seconds_outer", "memory", "timing", "zoltan_eval", "zoltan_removed_or_warning_lines")},
                "raw": {"validity": v, "km1_on_full_top_hypergraph": m["km1_weighted"], "cut_weighted": m["cut_weighted"], "blocks": m["blocks"], "flat_top_km1_addendum7": flat_km1,
                        "km1_ratio_vs_flat_top": (round(m["km1_weighted"] / flat_km1, 4) if flat_km1 else None)},
                "placement": H.placement(), "code": dict(H.code_pins(), **{"scratchpad/_h2lt.py": H.sha_file(os.path.abspath(__file__))})})
    H.log("top %s S %d: Zoltan %.0fs, KM1 on the full top hypergraph %d (flat top %s), blocks used %d/%d, max %d (bound %d) -> %s" % (
        name, S, r1["wall_seconds_outer"], m["km1_weighted"], flat_km1, m["blocks"]["used"], S, m["blocks"]["max"], v["contract_bound_ceil_1.03_N_over_k"], v["gate"]))
    shutil.rmtree(sdir, ignore_errors=True)
    shutil.rmtree(d1, ignore_errors=True)
    if not (v["length_ok"] and v["ids_in_range"] and v["max_block"] <= v["contract_bound_ceil_1.03_N_over_k"]):
        rec.update({"STATUS": "PARTITION_INVALID", "reason": "largest top block %d exceeds ceil(1.03 N / S) = %d" % (v["max_block"], v["contract_bound_ceil_1.03_N_over_k"]), "wall_seconds": round(time.time() - t0, 1)})
        H.wj(failp, rec)
        H.log("top %s S %d: PARTITION_INVALID" % (name, S))
        return
    if m["blocks"]["empty"] > 0:
        hard2, empties, moves = PR.repair(dict(Hf, meta=Hf["meta"]), hard)
        rec["repair"] = {"rule": PR.REPAIR, "empty_blocks": empties, "moves": len(moves)}
        hard = hard2.astype(np.int64)
        assert P.validity(hard, N, S)["gate"] == "PASS"
    else:
        rec["repair"] = None
    bal = H.balance(hard, N, S)
    assert bal["length_ok"] and bal["ids_in_range"] and bal["every_block_used"], bal
    tmp = tp[:-4] + ".tmp.npy"
    np.save(tmp, hard.astype(np.int64))
    os.replace(tmp, tp)
    rec.update({"STATUS": "OK", "balance": bal, "wall_seconds": round(time.time() - t0, 1), "output": {"file": H.rel(tp), "sha256": H.sha_file(tp), "n": int(len(hard))}})
    H.wj(runp, rec)
    H.log("top %s S %d: done %.0fs, sizes %d..%d -> %s" % (name, S, rec["wall_seconds"], bal["size_min"], bal["size_max"], H.rel(tp)))


def _test():
    rng = np.random.RandomState(11)
    N, M = 400, 600
    sz = rng.randint(2, 40, M)
    eptr = np.zeros(M + 1, np.int64)
    np.cumsum(sz, out=eptr[1:])
    eidx = np.concatenate([rng.choice(N, s, replace=False) for s in sz]).astype(np.int64)
    ew = rng.randint(1, 50, M).astype(np.int64)
    for c in (4, 8, 16, 39):
        ep, ei, ewk, st = cap_nets(eptr, eidx, ew, N, c)
        keep = np.flatnonzero(sz <= c)
        assert st["nets_kept"] == len(keep) and len(ewk) == len(keep) and int(ep[-1]) == len(ei) == int(sz[keep].sum())
        for j, e in enumerate(keep):
            assert np.array_equal(ei[ep[j]:ep[j + 1]], eidx[eptr[e]:eptr[e + 1]]) and ewk[j] == ew[e]
        assert bool((np.diff(ep) <= c).all())
    cl = rng.randint(0, 30, N)
    Vc, ep, ei, ewq, vw, qs = C.quotient(N, *cap_nets(eptr, eidx, ew, N, 16)[:3], np.ones(N, np.int64), cl)
    assert int(vw.sum()) == N and Vc == int(cl.max()) + 1
    assert parse("Q512L6N8") == {"name": "Q512L6N8", "W": 512, "level": 6, "cap": 8} and parse("N16")["W"] is None
    H.log("TEST PASS: net cap keeps exactly the nets of <= c pins with their pins and weights; the capped hypergraph quotients; spec names parse")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a == ["TEST"]:
        _test()
    elif len(a) == 3 and a[0] == "TOP":
        H.log("placement:", json.dumps(H.placement()))
        for S in a[2].split(","):
            cmd_top(a[1], int(S))
    elif len(a) >= 4 and a[0] == "SUB":
        CUR["spec"] = parse(a[1])
        S = int(a[2])
        L.MODE["top"] = True
        tfail = H.phg_npy(DS, S)[:-4] + ".FAILED.json"
        L.MODE["top"] = False
        for K in a[3:]:
            if os.path.exists(tfail):                 # the top surrogate was refused by the frozen validity rule: the arm has no map at any K (recorded, never relaxed)
                _, runp, failp = arm_files(S, K)
                if not (os.path.exists(runp) or os.path.exists(failp)):
                    H.wj(failp, {"RECORD": "H2LT_ARM", "STATUS": "PARTITION_INVALID", "spec": CUR["spec"], "S": S, "K": K, "reason": "the top surrogate partition is PARTITION_INVALID",
                                 "top_record": H.rel(tfail), "top_record_sha256": H.sha_file(tfail), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                    H.log("arm %s S %d K %d: top invalid -> arm PARTITION_INVALID" % (a[1], S, K))
                continue
            L.cmd_sub(S, int(K))
    else:
        raise SystemExit(__doc__)
