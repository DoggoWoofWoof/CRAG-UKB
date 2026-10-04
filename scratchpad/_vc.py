"""FBX_SCALE Track B (out-of-core partitioner, development D1) -- a V-cycle over the frozen M2 ladder: partition the COARSEST level (a stored addendum-9 surrogate top), then project and REFINE at every finer level.

Addendum 9 partitioned a surrogate (the quotient at ladder level 6, optionally with the nets of more than c pins dropped) and projected the result with no refinement (the pin reduction was 3.3 / 6.3 / 14.2x and the
nested partition lost recall at all three).  Stage 4C refined ONCE, at the finest level, with label propagation (strict gains), and could not repair it.  What was never tried is the standard multilevel step: refine at
EVERY level on the way up, with a hill-climbing refiner (FM) that can move a whole cluster before it is split.  This module does exactly that, on the WebQSP top hypergraph (S = 25), in memory.

  coarsest labels  = the stored addendum-9 top map (Zoltan-PHG, NP 4, k = 25), taken at one representative per level-6 cluster (asserted constant on every cluster)
  every level l    = the exact quotient of the FULL top hypergraph (identical nets merged, weights summed) under the frozen cumulative cluster map clusters__M2_W512_L<l>.npy; l = 0 is the top hypergraph itself
  refinement       = scratchpad/_vc_ref.c  (mode 0 LP: strict-gain sweeps; mode 1 FM: hill-climbing with rollback), balance cap floor(1.01 N / S) -- Zoltan's own tolerance for the top
  D1 metric        = weighted KM1 of the final partition on the FULL top hypergraph relative to the flat Zoltan top of addendum 7 (791,538,830).  No gold label is read.

  python -u scratchpad/_vc.py D1 <spec> <refiner>         # spec in {Q512L6, Q512L6N8, Q512L6N4}; refiner in REFINERS: none = project only, last_lp = LP at level 0 only (the 4C control),
                                                          # lp / fm / lp2 / fm2 = at the coarsest level and every finer one on the FULL quotient (v1 sources: lp, fm; v2: lp2 = lp bit for bit, fm2 = FM with a gain cache)
  python -u scratchpad/_vc.py TOP <spec> <refiner>        # the same V-cycle, written as a TOP record of the nested pipeline (results/FREEBASE_SCALE/h2l/parts/webqsp__H4_H2LT_VC<spec><REFINER>_top_k25__PHG_con.*), for _vc_sub.py
  python -u scratchpad/_vc.py TEST                        # level identities (projection, KM1 of a level = KM1 of the fine hypergraph under the projected partition) on a random hypergraph
"""
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import time

import numpy as np

import _fbx_part as FP
import _h2lt as T

C, H, P, FR, MR = T.C, T.H, T.P, T.FR, T.MR
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(H.REPO, "results", "FREEBASE_SCALE", "vc")
S = 25
FLAT_KM1 = 791538830                      # addendum 7 flat Zoltan top, KM1 on the full top hypergraph (webqsp__H4_H2LTOP_k25__PHG_con.RUN.json raw.km1)
LADDER_W = 512
SPEC = re.compile(r"Q512L(\d+)(?:N(\d+))?")
LP_PASSES, FM_PASSES, FM_MAXFAIL = 10, 6, 500


def log(*a):
    H.log(*a)


REFINERS = {            # name -> (C source, mode, levels): levels "all" = the coarsest level and every finer one, "last" = level 0 only
    "none": (None, None, None), "last_lp": ("_vc_ref.c", 0, "last"), "lp": ("_vc_ref.c", 0, "all"), "fm": ("_vc_ref.c", 1, "all"),
    "lp2": ("_vc_ref2.c", 0, "all"), "fm2": ("_vc_ref2.c", 1, "all"), "last_fm2": ("_vc_ref2.c", 1, "last"),
}


def c_binary_ref(source="_vc_ref.c"):
    src = os.path.join(HERE, source)
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    bindir = os.path.join(H.REPO, "data", "vc_tmp", "bin")        # on the workspace drive, not WSL /tmp (another job or a distro restart emptied /tmp twice)
    os.makedirs(bindir, exist_ok=True)
    binp = FP.wsl_path(os.path.join(bindir, "%s_%s" % (source[:-2], sha)))
    try:
        FP._run(["test", "-x", binp])
    except RuntimeError:
        FP._run(["gcc", "-O2", "-Wall", "-o", binp, FP.wsl_path(src)])
    return binp, sha


def refine(mode, V, ep, ei, ew, vw, lab, k, cap, passes, maxfail, tmp, source="_vc_ref.c"):
    """one call of _vc_ref on an explicit hypergraph; returns (labels, info)."""
    binp, sha = c_binary_ref(source)
    vptr, vidx = C.transpose(V, ep, ei)
    fs = {}
    for nm, arr, dt in (("vw", vw, np.int64), ("eptr", ep, np.int64), ("eidx", ei, np.int32), ("ew", ew, np.int64), ("vptr", vptr, np.int64), ("vidx", vidx, np.int32), ("lab", lab, np.int32)):
        fs[nm] = os.path.join(tmp, "vc_%s.raw" % nm)
        np.ascontiguousarray(arr, dt).tofile(fs[nm])
    out = os.path.join(tmp, "vc_out.raw")
    cmd = [binp, str(mode), str(V), str(len(ep) - 1), str(k), str(cap), str(passes), str(maxfail)] + [FP.wsl_path(fs[x]) for x in ("vw", "eptr", "eidx", "ew", "vptr", "vidx", "lab")] + [FP.wsl_path(out)]
    for attempt in range(3):
        try:
            txt = FP._run(cmd)
            break
        except RuntimeError as e:
            if "No such file or directory" not in str(e)[:900] or attempt == 2:
                raise
            log("the WSL binary %s is gone (another job reset /tmp or restarted the distro): recompiling, attempt %d" % (binp, attempt + 2))
            binp, sha = c_binary_ref(source)
            cmd[0] = binp
    info = json.loads(txt.strip().splitlines()[-1])
    info["c_source_sha12"] = sha
    res = np.fromfile(out, np.int32)
    assert len(res) == V
    return res, info


def rep_proj(map_fine, map_coarse):
    """for every cluster j of the finer clustering (dense ids), the id of its cluster in the coarser one (both are cumulative maps of the same fine vertices)."""
    _, first = np.unique(map_fine, return_index=True)
    return np.asarray(map_coarse, np.int64)[first]


def ladder(level_top):
    """cumulative maps of ladder levels 0 .. level_top (level 0 = identity); each is dense."""
    maps = {}
    for l in range(1, level_top + 1):
        cl, cm, rec_p = T.cluster_map(LADDER_W, l)
        maps[l] = cl.astype(np.int64)
    return maps


def level_hgs(N, eptr, eidx, ew, maps, level_top):
    """the exact quotient of the FULL top hypergraph at every level 1..level_top (level 0 is the hypergraph itself)."""
    hgs = {0: (N, eptr, eidx.astype(np.int32), ew, np.ones(N, np.int64))}
    qdir = os.path.join(H.REPO, "data", "vc_tmp", "qcache")      # the quotient of the full top hypergraph under a cluster map is deterministic: computed once, reused by every refiner arm
    os.makedirs(qdir, exist_ok=True)
    for l in range(1, level_top + 1):
        key = hashlib.sha256(np.ascontiguousarray(maps[l], np.int64).tobytes() + eptr.tobytes()[:4096] + ew.tobytes()[:4096] + str((N, len(eidx))).encode()).hexdigest()[:16]
        qf = os.path.join(qdir, "q_%s.npz" % key)
        if os.path.exists(qf):
            z = np.load(qf)
            Vc, ep, ei, ewq, vw = int(z["Vc"]), z["ep"], z["ei"], z["ewq"], z["vw"]
        else:
            Vc, ep, ei, ewq, vw, st = C.quotient(N, eptr, eidx, ew, np.ones(N, np.int64), maps[l])
            tmpq = qf + ".%d.tmp.npz" % os.getpid()
            np.savez(tmpq, Vc=np.int64(Vc), ep=ep, ei=ei, ewq=ewq, vw=vw)
            os.replace(tmpq, qf)
        hgs[l] = (Vc, ep, ei, ewq, vw)
        log("level %d: V %d M %d P %d (%.3fx fewer pins than level 0)" % (l, Vc, len(ewq), len(ei), len(eidx) / max(len(ei), 1)))
    return hgs


def km1_level(hg, lab, k):
    V, ep, ei, ew, vw = hg
    return C.km1(ep, ei, ew, lab)


def vcycle(hgs, maps, level_top, lab_top, refiner, k, cap, tmp):
    """lab_top: labels of the level_top vertices.  Returns (fine labels, per-level trajectory)."""
    traj = []
    lab = np.asarray(lab_top, np.int32)
    for l in range(level_top, -1, -1):
        hg = hgs[l]
        if l < level_top:
            proj = rep_proj(maps[l], maps[l + 1]) if l >= 1 else np.asarray(maps[1], np.int64)
            lab = lab[proj]
        before = km1_level(hg, lab, k)
        info = None
        src, mode, levels = REFINERS[refiner]
        if src is not None and (levels == "all" or l == 0):
            lab, info = refine(mode, hg[0], hg[1], hg[2], hg[3], hg[4], lab, k, cap, FM_PASSES if mode else LP_PASSES, FM_MAXFAIL, tmp, src)
            assert info["km1_before"] == before, (info, before)
        after = km1_level(hg, lab, k)
        traj.append({"level": l, "V": hg[0], "P": int(len(hg[2])), "km1_in": before, "km1_out": after, "refine": info})
        log("level %d: KM1 %d -> %d (%+.3f %%)%s" % (l, before, after, 100.0 * (after - before) / max(before, 1), "" if info is None else "  [%s: %d moves, %d passes, %.1fs]" % (refiner, info["moves"], info["passes_run"], info["seconds"])))
    return lab, traj


def stored_top(spec):
    return os.path.join(T.OUT, "webqsp__H4_H2LT_%s_top_k%d__PHG_con" % (spec, S))


def cmd_d1(spec, refiner, final=False):
    m = SPEC.fullmatch(spec)
    assert m and refiner in REFINERS, (spec, refiner)
    level_top = int(m.group(1))
    os.makedirs(OUTD, exist_ok=True)
    name = "VC%s%s" % (spec, refiner.upper())          # the arm name of the nested pipeline (no underscores: the evaluator splits on them)
    base = os.path.join(T.OUT, "webqsp__H4_H2LT_%s_top_k%d__PHG_con" % (name, S)) if final else os.path.join(OUTD, "webqsp__VC_%s_%s_top_k%d" % (spec, refiner, S))
    done = base + (".RUN.json" if final else ".D1.json")
    if os.path.exists(done) or os.path.exists(base + ".FAILED.json"):
        log("%s %s %s: exists" % ("TOP" if final else "D1", spec, refiner))
        return
    t0 = time.time()
    Hf = T.load_top_hg(S)
    N, eptr, eidx, ew = Hf["N"], Hf["eptr"], Hf["eidx"], Hf["ew"]
    rp = stored_top(spec) + ".RUN.json"
    rr = json.load(io.open(rp, encoding="utf-8"))
    assert rr["STATUS"] == "OK" and rr["repair"] is None
    tp = os.path.join(H.REPO, rr["output"]["file"])
    assert H.sha_file(tp) == rr["output"]["sha256"], "stored top changed"
    top = np.load(tp).astype(np.int64)
    maps = ladder(level_top)
    cl = maps[level_top]
    Vc = int(cl.max()) + 1
    lab_top = np.full(Vc, -1, np.int64)
    lab_top[cl] = top
    assert bool((lab_top[cl] == top).all()), "the stored top is not constant on the level-%d clusters" % level_top
    hgs = level_hgs(N, eptr, eidx, ew, maps, level_top)
    cap = int(1.01 * N / S)
    flat = FLAT_KM1
    os.makedirs(os.path.join(H.REPO, "data", "vc_tmp"), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.join(H.REPO, "data", "vc_tmp")) as tmp:
        hard, traj = vcycle(hgs, maps, level_top, lab_top.astype(np.int32), refiner, S, cap, tmp)
    hard = hard.astype(np.int64)
    v = P.validity(hard, N, S)
    mm = FR.km1_metrics(Hf, hard)
    ratio = mm["km1_weighted"] / flat
    log("D1 %s %s: KM1 on the full top hypergraph %d = %.4f x the flat top; blocks used %d/%d, max %d (bound %d) -> %s; %.0fs" % (
        spec, refiner, mm["km1_weighted"], ratio, mm["blocks"]["used"], S, mm["blocks"]["max"], v["contract_bound_ceil_1.03_N_over_k"], v["gate"], time.time() - t0))
    if final:                                           # the TOP map must be the D1 map: same stored top, same ladder, same deterministic binaries -> identical KM1 (addendum 14 control)
        d1p = os.path.join(OUTD, "webqsp__VC_%s_%s_top_k%d.D1.json" % (spec, refiner, S))
        assert os.path.exists(d1p), "no D1 record %s: run D1 first" % d1p
        d1 = json.load(io.open(d1p, encoding="utf-8"))
        assert d1["final"]["km1_on_full_top_hypergraph"] == mm["km1_weighted"], ("TOP KM1 differs from its D1 record", d1["final"]["km1_on_full_top_hypergraph"], mm["km1_weighted"])
        log("TOP %s %s: KM1 equals its D1 record (%d)" % (spec, refiner, mm["km1_weighted"]))
    rec = {"RECORD": "VC_D1", "dataset": "webqsp", "spec": spec, "refiner": refiner, "S": S, "level_top": level_top, "cap": cap, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "stored_top": {"record": H.rel(rp), "map_sha256": rr["output"]["sha256"], "km1_ratio_unrefined": rr["raw"].get("km1_ratio_vs_flat_top")},
           "params": {"LP_PASSES": LP_PASSES, "FM_PASSES": FM_PASSES, "FM_MAXFAIL": FM_MAXFAIL}, "trajectory": traj,
           "final": {"km1_on_full_top_hypergraph": mm["km1_weighted"], "km1_ratio_vs_flat_top": round(ratio, 4), "blocks": mm["blocks"], "validity": v},
           "seconds": round(time.time() - t0, 1), "code": {"scratchpad/_vc.py": H.sha_file(os.path.abspath(__file__)), "scratchpad/_vc_ref.c": H.sha_file(os.path.join(HERE, "_vc_ref.c")), "scratchpad/_vc_ref2.c": H.sha_file(os.path.join(HERE, "_vc_ref2.c"))}}
    if final:
        rec["RECORD"] = "VC_TOP"
        rec["arm"] = name
        if not (v["length_ok"] and v["ids_in_range"] and v["max_block"] <= v["contract_bound_ceil_1.03_N_over_k"] and mm["blocks"]["empty"] == 0):
            rec.update({"STATUS": "PARTITION_INVALID", "reason": "validity %s, empty blocks %d" % (v["gate"], mm["blocks"]["empty"])})
            H.wj(base + ".FAILED.json", rec)
            log("TOP %s %s: PARTITION_INVALID (never relaxed)" % (spec, refiner))
            return
        tmpf = base + ".tmp.npy"
        np.save(tmpf, hard)
        os.replace(tmpf, base + ".npy")
        rec.update({"STATUS": "OK", "output": {"file": H.rel(base + ".npy"), "sha256": H.sha_file(base + ".npy"), "n": int(len(hard))}, "balance": H.balance(hard, N, S)})
        H.wj(base + ".RUN.json", rec)
        return
    np.save(base + ".npy", hard)
    rec["map_sha256"] = H.sha_file(base + ".npy")
    H.wj(base + ".D1.json", rec)


# ----------------------------------------------------------------------------------------------------------------- tests
def _test():
    rng = np.random.RandomState(3)
    N, M = 600, 900
    sz = rng.randint(2, 9, M)
    eptr = np.zeros(M + 1, np.int64)
    np.cumsum(sz, out=eptr[1:])
    eidx = np.concatenate([rng.choice(N, s, replace=False) for s in sz]).astype(np.int64)
    ew = rng.randint(1, 20, M).astype(np.int64)
    # a two-level ladder: level 1 = pairs, level 2 = groups of 4 level-1 clusters
    m1 = (np.arange(N) // 2).astype(np.int64)
    _, m1 = np.unique(m1, return_inverse=True)
    m2 = (m1 // 4).astype(np.int64)
    _, m2 = np.unique(m2, return_inverse=True)
    maps = {1: m1.astype(np.int64), 2: m2.astype(np.int64)}
    hgs = level_hgs(N, eptr, eidx, ew, maps, 2)
    k = 5
    lab2 = rng.randint(0, k, hgs[2][0]).astype(np.int32)
    # projection chain equals the direct map
    fine_direct = lab2[maps[2]]
    lab = lab2
    for l in (1, 0):
        proj = rep_proj(maps[l], maps[l + 1]) if l >= 1 else maps[1]
        lab = lab[proj]
    assert np.array_equal(lab, fine_direct), "level-by-level projection differs from the direct cumulative map"
    # KM1 of every level equals KM1 of the fine hypergraph under the projected partition
    ref = C.km1(eptr, eidx, ew, fine_direct)
    l1 = lab2[rep_proj(maps[1], maps[2])]
    assert km1_level(hgs[2], lab2, k) == ref and km1_level(hgs[1], l1, k) == ref and km1_level(hgs[0], fine_direct.astype(np.int32), k) == ref
    log("TEST PASS: level-by-level projection == the direct cumulative map; the KM1 of every level's quotient == the KM1 of the fine hypergraph under the projected partition (%d)" % ref)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    if a == ["TEST"]:
        _test()
    elif len(a) == 3 and a[0] in ("D1", "TOP"):
        log("placement:", json.dumps(H.placement()))
        cmd_d1(a[1], a[2], final=(a[0] == "TOP"))
    else:
        raise SystemExit(__doc__)
