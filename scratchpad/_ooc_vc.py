"""FBX_SCALE addendum 17 -- stages E4 (the cap-4 level-4 surrogate -> the frozen vertex-weighted Zoltan-PHG top) and E6 (the V-cycle: project, FM2-refine at every ladder level 4..0) on the out-of-core engine.

  python -u scratchpad/_ooc_vc.py SURROGATE <csr_dir> <l1_dir> <work_dir> [threads]    # Freebase: quotient (nets > 4 pins dropped, split-top nets) at level 4 -> shards -> Zoltan top; writes <work>/zlab4.i32
  python -u scratchpad/_ooc_vc.py VCYCLE    <csr_dir> <l1_dir> <work_dir> [threads] [levels=HI-LO] [anon_gb=NN]    # Freebase: V-cycle from <work>/zlab4.i32 -> <work>/lab0.i32 (the 25-way top map) + the record E6_VCYCLE__fbx.json
  python -u scratchpad/_ooc_vc.py TEST_WEBQSP_VC                                       # WebQSP end to end on the engine == the stored Zoltan top, the stored V-cycle top map (sha 4c043f0c...) and its D1 trajectory (host)

The level hypergraphs are the exact quotients of the FULL SPLIT top hypergraph (H4_SPLIT_PRESERVE at S = 25) under the cumulative ladder maps; level 0 is the implicit source; the refiner is the FM2 engine
(_ooc2.cpp fm2, bit-identical to _vc_ref2.c mode 1: 6 passes, stop after 500 non-improving moves, per-block cap int(1.01 N / S)).  No gold label is read anywhere in this file.
"""
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _ooc_lib as L  # noqa: E402
from _ooc_fb import exists, put_json, rm, sh, step  # noqa: E402

REPO = L.REPO
LOG = L.log
S = 25
SUR_CAP = 4
LEVEL_TOP = 4
FM_PASSES, FM_MAXFAIL = 6, 500
TOP_TOL = "1.01"
SURROGATE_PIN_STOP = 450_000_000           # addendum 17 stop rule: a surrogate of more than 0.45e9 pins is not attempted
STATS = os.path.join(REPO, "results", "FREEBASE_SCALE", "FBX_GRAPH_STATS__v1.json")


def cap_of(N):
    return int(1.01 * N / S)


def du(path):
    try:
        return int(sh(["du", "-sb", path]).split()[0])
    except RuntimeError:
        return -1


# ---------------------------------------------------------------------------------------------------------------- E4: surrogate -> shards -> Zoltan
def surrogate(fam, N, l1, work, V4, threads, tag, with_zoltan=True):
    """the cap-4 level-4 quotient of the split top hypergraph -> node-centric shards -> the frozen vertex-weighted Zoltan-PHG driver (NP 4, IMBALANCE_TOL 1.01, k = S) -> <work>/zlab4.i32"""
    sh(["mkdir", "-p", work])
    pre = work + "/sur"
    t0 = time.time()
    q = step(work, "sur_quot", lambda: L.run("quot", {"src": "csr", "N": N, "fam": fam, "split_s": S, "cap": SUR_CAP, "map": l1 + "/cl4.i32", "vout": V4, "out": pre, "threads": threads}))
    LOG("surrogate: V %d, M %d, P %d (nets <= %d pins of the split top, quotient at level 4)" % (q["V"], q["M"], q["P"], SUR_CAP))
    if q["P"] > SURROGATE_PIN_STOP:
        raise SystemExit("STOP RULE: the surrogate has %d pins > %d (addendum 17); not partitioned" % (q["P"], SURROGATE_PIN_STOP))
    tr = step(work, "sur_transpose", lambda: L.run("transpose", {"eptr": pre + ".eptr.i64", "eidx": pre + ".eidx.i32", "vin": V4, "out": pre}))
    sdir = os.path.join(REPO, "data", "fbx_zoltan", "stream_%s" % tag)
    os.makedirs(sdir, exist_ok=True)
    sd = step(work, "sur_shards", lambda: L.run("shards", {"eptr": pre + ".eptr.i64", "ew": pre + ".ew.i64", "vptr": pre + ".vptr.i64", "vidx": pre + ".vidx.i32", "vw": pre + ".vw.i32", "vin": V4,
                                                           "out_dir": L.wp(sdir), "shard_nodes": 5000, "threads": threads}))
    LOG("surrogate shards: %d shards, %.2f GB (%.0fs)" % (sd["shards"], sd["bytes"] / 1e9, sd["seconds"]))
    rec = {"surrogate": q, "transpose": tr, "shards": sd, "disk_bytes_work": du(work), "seconds_build": round(time.time() - t0, 1)}
    if not with_zoltan:
        return rec, sdir
    z = step(work, "sur_zoltan", lambda: zoltan(sdir, V4, q, work, tag))
    rec["zoltan"] = z
    return rec, sdir


def zoltan(sdir, V4, q, work, tag):
    """the frozen vw driver exactly as _h2lt.cmd_top runs it; the part files are assembled into <work>/zlab4.i32; the shards are deleted afterwards"""
    import _ml_run as MR
    import _l1h_host as H
    P = H.P
    b = MR.vw_build()
    base = list(P.PARAMS)
    P.PARAMS = [(k, TOP_TOL if k == "IMBALANCE_TOL" else v) for k, v in base]
    try:
        r1, d1 = P.mpirun("fbx_zoltan", "R1_VC_%s_%d" % (tag, S), "partition", S, os.path.join(sdir, "stream_manifest.txt"), b, timeout=H.PHG_TIMEOUT_S)
    finally:
        P.PARAMS = base
    gq = r1["global_from_queries"]
    assert gq["objects"] == V4 and gq["pins"] == q["P"], ("Zoltan saw a different hypergraph", gq, V4, q["P"])
    lab = assemble(d1, V4)
    L.wr(lab, os.path.join(d1, "lab4.i32.local"), np.int32)
    sh(["cp", L.wp(os.path.join(d1, "lab4.i32.local")), work + "/zlab4.i32"])
    import shutil
    shutil.rmtree(sdir, ignore_errors=True)
    out = {"driver": b["binary"], "source_sha256": b["source"]["sha256"], "NP": P.NP, "parameters_top": [(k, TOP_TOL if k == "IMBALANCE_TOL" else v) for k, v in base],
           "phg": {kk: r1[kk] for kk in ("name", "mode", "np", "rc", "wall_seconds_outer", "memory", "timing", "zoltan_eval", "zoltan_removed_or_warning_lines")}}
    shutil.rmtree(d1, ignore_errors=True)
    return out


def assemble(d, V):
    """part_rank<i>.txt lines 'gid part' of the NP ranks -> the labels of the V objects (every object exactly once)"""
    import _l1h_host as H
    gid, part = [], []
    for i in range(H.P.NP):
        p = os.path.join(d, "part_rank%d.txt" % i)
        a = np.fromstring(io.open(p, encoding="ascii").read(), dtype=np.int64, sep=" ") if os.path.getsize(p) else np.zeros(0, np.int64)
        a = a.reshape(-1, 2)
        gid.append(a[:, 0])
        part.append(a[:, 1])
    gid, part = np.concatenate(gid), np.concatenate(part)
    if len(gid) != V or not np.array_equal(np.sort(gid), np.arange(V)):
        raise RuntimeError("part files do not cover every position exactly once (%d rows, expected %d)" % (len(gid), V))
    lab = np.empty(V, np.int32)
    lab[gid] = part
    return lab


# ---------------------------------------------------------------------------------------------------------------- E6: the V-cycle
def vcycle(fam, N, l1, work, Vs, threads, lab_top=None, passes=FM_PASSES, maxfail=FM_MAXFAIL, check=0, hi=LEVEL_TOP, lo=0):
    """project and refine at every level 4..0.  l1 holds cl<l>.i32 (cumulative maps, level 0 -> l) and inc<l>.i32 (level l-1 -> l, l = 2..4); the labels of level 4 are <lab_top> (default <work>/zlab4.i32).
    Returns the record; the final labels are <work>/lab0.i32."""
    sh(["mkdir", "-p", work])
    cap = cap_of(N)
    traj = []
    t_all = time.time()
    lab_in = lab_top or work + "/zlab4.i32"
    for l in range(hi, lo - 1, -1):                                          # a job may cover a level range (its own measured reservation); the step markers make the ranges compose
        name = "vc_l%d" % l
        if l < LEVEL_TOP:
            a = l1 + "/inc%d.i32" % (l + 1) if l >= 1 else l1 + "/cl1.i32"
            lab_prev = work + "/lab%d.i32" % (l + 1)
            lab_in = work + "/labin%d.i32" % l
            step(work, "proj%d" % l, lambda: L.run("compose", {"a": a, "b": lab_prev, "out": lab_in, "threads": threads}))
        lab_out = work + "/lab%d.i32" % l

        def refine():
            rec = {"level": l}
            if l >= 1:
                pre = "%s/s%d" % (work, l)
                q = L.run("quot", {"src": "csr", "N": N, "fam": fam, "split_s": S, "map": "%s/cl%d.i32" % (l1, l), "vout": Vs[l], "out": pre, "threads": threads})
                tr = L.run("transpose", {"eptr": pre + ".eptr.i64", "eidx": pre + ".eidx.i32", "vin": Vs[l], "out": pre})
                rec.update({"V": q["V"], "M": q["M"], "P": q["P"], "quot_seconds": q["seconds"], "transpose_seconds": tr["seconds"], "disk_bytes_work": du(work)})
                LOG("level %d: V %d, M %d, P %d" % (l, q["V"], q["M"], q["P"]))
                f = L.run("fm2", {"src": "expl", "eptr": pre + ".eptr.i64", "eidx": pre + ".eidx.i32", "ew": pre + ".ew.i64", "vptr": pre + ".vptr.i64", "vidx": pre + ".vidx.i32", "vw": pre + ".vw.i32",
                                  "vin": Vs[l], "k": S, "cap": cap, "passes": passes, "maxfail": maxfail, "lab_in": lab_in, "lab_out": lab_out, "check": check, "threads": threads})
                rm(*[pre + x for x in (".eptr.i64", ".eidx.i32", ".ew.i64", ".vw.i32", ".vptr.i64", ".vidx.i32")])
            else:
                f = L.run("fm2", {"src": "csr", "N": N, "fam": fam, "split_s": S, "k": S, "cap": cap, "passes": passes, "maxfail": maxfail, "lab_in": lab_in, "lab_out": lab_out, "check": check,
                                  "threads": threads})
                rec.update({"V": N, "M": f["M"]})
            rec.update({"km1_in": f["km1_before"], "km1_out": f["km1_after"], "moves": f["moves"], "passes_run": f["passes_run"], "stale_repushes": f["stale_repushes"], "heap_peak_entries": f["heap_peak_entries"],
                        "pushes": f["pushes"], "cache_bytes_per_entry": f["cache_bytes_per_entry"], "max_load_after": f["max_load_after"], "min_load_after": f["min_load_after"], "fm_seconds": f["seconds"],
                        "fm_peak_rss_gb": f["peak_rss_gb"]})
            return rec

        r = step(work, name, refine)
        if l < LEVEL_TOP:
            rm(lab_in)
        traj.append(r)
        LOG("level %d: KM1 %d -> %d (%+.3f %%), %d moves, %d passes, %.0fs" % (l, r["km1_in"], r["km1_out"], 100.0 * (r["km1_out"] - r["km1_in"]) / max(r["km1_in"], 1), r["moves"], r["passes_run"], r["fm_seconds"]))
    return {"cap": cap, "trajectory": traj, "seconds": round(time.time() - t_all, 1), "final": work + "/lab0.i32"}


def validity(path, N):
    """block sizes of the final map under the frozen validity rule (every block used, largest block <= ceil(1.03 N / S))"""
    r = L.run("blocks", {"lab": L.wp(path), "k": S}, stream=False)
    sizes = np.array(r["sizes"], np.int64)
    bound = int(np.ceil(1.03 * N / float(S)))
    ok = r["n"] == N and r["out_of_range"] == 0 and bool((sizes > 0).all()) and int(sizes.max()) <= bound
    return {"length_ok": r["n"] == N, "ids_in_range": r["out_of_range"] == 0, "empty_blocks": int((sizes == 0).sum()), "max_block": int(sizes.max()), "min_block": int(sizes.min()),
            "contract_bound_ceil_1.03_N_over_k": bound, "block_sizes": sizes.tolist(), "gate": "PASS" if ok else "PARTITION_INVALID"}


# ---------------------------------------------------------------------------------------------------------------- WebQSP end to end
def test_webqsp_vc():
    """the whole stage chain on WebQSP (the host): the engine's surrogate shards, Zoltan top and V-cycle must reproduce the stored records byte for byte"""
    import _ml_run as MR
    import _ml2_run as R
    import _h2lt as T
    import _l1h_host as H
    import _ml_coarsen as C
    from _ooc_e1 import cmp_arrays, tmpdir, write_fams
    d = tmpdir("wqvc")
    N, fams, cnt = R.graph_SK()
    fa = write_fams(d, fams, "wv")
    l1 = os.path.join(d, "l1")
    os.makedirs(l1)
    maps = {l: np.load(os.path.join(REPO, "results", "FREEBASE_SCALE", "ml2", "clusters__M2_W512_L%d.npy" % l)).astype(np.int64) for l in (1, 2, 3, 4)}
    Vs = {l: int(maps[l].max()) + 1 for l in maps}
    for l in maps:
        L.wr(maps[l], os.path.join(l1, "cl%d.i32" % l), np.int32)
    for l in (2, 3, 4):                                                    # inc_l: level l-1 cluster -> level l cluster (the SPAN pass of the ladder; == compose-consistent with the cumulative maps)
        _, first = np.unique(maps[l - 1], return_index=True)
        L.wr(maps[l][first], os.path.join(l1, "inc%d.i32" % l), np.int32)
    assert N == len(maps[1])
    # E4: the surrogate hypergraph and its shards == _h2lt.cmd_top's, then the same Zoltan top
    work = os.path.join(d, "work")
    os.makedirs(work)
    spec = T.parse("Q512L4N4")
    rec, sdir = surrogate([x for x in fa], N, L.wp(l1), L.wp(work), Vs[4], 4, "wq%d" % os.getpid(), with_zoltan=False)
    sur = rec["surrogate"]
    Hf = T.load_top_hg(S)
    ep3, ei3, ew3, st = T.cap_nets(Hf["eptr"], Hf["eidx"], Hf["ew"], N, SUR_CAP)
    Vc, ep, ei, ewq, vw, qs = C.quotient(N, ep3, ei3, ew3, np.ones(N, np.int64), maps[4])
    pre = os.path.join(work, "sur")
    cmp_arrays("surrogate eptr", L.rd(pre + ".eptr.i64", np.int64), ep)
    cmp_arrays("surrogate eidx", L.rd(pre + ".eidx.i32", np.int32).astype(np.int64), ei.astype(np.int64))
    cmp_arrays("surrogate ew", L.rd(pre + ".ew.i64", np.int64), ewq)
    sd_py = os.path.join(d, "py_shards")
    shards, smf = MR.write_shards_vw(Vc, ep, ei.astype(np.int64), ewq, vw, sd_py, with_vw=True)
    for c in shards:
        assert open(os.path.join(sdir, os.path.basename(c["file"])), "rb").read() == open(c["file"], "rb").read(), c["file"]
    LOG("E4 PASS (surrogate): V %d, M %d, P %d; the engine's %d shards are byte-identical to _ml_run.write_shards_vw" % (sur["V"], sur["M"], sur["P"], len(shards)))
    # the Zoltan top end to end
    z = zoltan(sdir, Vs[4], sur, L.wp(work), "wq%d" % os.getpid())
    stored = json.load(io.open(os.path.join(T.OUT, "webqsp__H4_H2LT_Q512L4N4_top_k25__PHG_con.RUN.json"), encoding="utf-8"))
    top = np.load(os.path.join(REPO, stored["output"]["file"])).astype(np.int64)
    got = np.fromfile(os.path.join(work, "zlab4.i32"), np.int32).astype(np.int64)
    cmp_arrays("Zoltan top (projected through the level-4 map)", got[maps[4]], top)
    LOG("E4 PASS (Zoltan): the engine's surrogate under the frozen vw driver (NP 4, IMBALANCE_TOL 1.01) reproduces the stored top map (sha %s...)" % stored["output"]["sha256"][:12])
    # E6: the V-cycle
    vcr = vcycle([x for x in fa], N, L.wp(l1), L.wp(work), Vs, 4, check=1)
    d1 = json.load(io.open(os.path.join(REPO, "results", "FREEBASE_SCALE", "vc", "webqsp__VC_Q512L4N4_fm2_top_k25.D1.json"), encoding="utf-8"))
    for t, e in zip(vcr["trajectory"], d1["trajectory"]):
        assert t["level"] == e["level"] and t["km1_in"] == e["km1_in"] and t["km1_out"] == e["km1_out"] and t["moves"] == e["refine"]["moves"] and t["passes_run"] == e["refine"]["passes_run"], (t, e)
        if t["level"] >= 1:
            assert t["V"] == e["V"] and t["P"] == e["P"], (t, e)
    final = np.fromfile(os.path.join(work, "lab0.i32"), np.int32).astype(np.int64)
    out = os.path.join(d, "final.npy")
    np.save(out, final)
    want = os.path.join(REPO, "results", "FREEBASE_SCALE", "vc", "webqsp__VC_Q512L4N4_fm2_top_k25.npy")
    assert H.sha_file(out) == H.sha_file(want), "the V-cycle top map differs from the stored one"
    LOG("E6 PASS: the engine's V-cycle reproduces the stored top map byte for byte (sha %s), the D1 trajectory (km1 in / out, moves, passes at levels 4..0) and V / P of levels 1..4" % H.sha_file(want)[:12])
    v = validity(os.path.join(work, "lab0.i32"), N)
    LOG("validity %s" % json.dumps(v))
    LOG("TEST_WEBQSP_VC PASS")


def test_vc_synth():
    """the V-cycle driver (twins -> ladder -> split-top level quotients -> FM2 at levels 4..0) on a synthetic graph with hubs == the reference chain: _ml_coarsen.quotient of the numpy split top under the
    engine's own cumulative maps + _vc.vcycle with the refiner fm2 (_vc_ref2.c)"""
    import _ooc_fb as FBD
    import _ml_coarsen as C
    import _vc as VC
    from _ooc_e1 import cmp_arrays, ref_split_nets, tmpdir
    rng = np.random.RandomState(7)
    N, E = 24000, 120000
    src, dst = rng.randint(0, N, E), rng.randint(0, N, E)
    hub = rng.randint(0, E, E // 5)
    dst[hub] = rng.choice([0, 1, 2, 3], len(hub))
    o = np.argsort(src, kind="stable")
    oi = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=oi[1:])
    o2 = np.argsort(dst, kind="stable")
    ii = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(dst, minlength=N), out=ii[1:])
    d = tmpdir("vcsyn")
    g = os.path.join(d, "graph")
    os.makedirs(g)
    for nm, a in (("out_indptr", oi), ("out_dst", dst[o].astype(np.int32)), ("in_indptr", ii), ("in_src", src[o2].astype(np.int32))):
        np.save(os.path.join(g, nm + ".npy"), a)
    csr = os.path.join(d, "csr")
    os.makedirs(csr)
    L.run("csr", {"graph": L.wp(g), "out": L.wp(csr), "threads": 2, "chunk": 40000}, stream=False)
    fam = [L.wp(csr) + "/xa.i64," + L.wp(csr) + "/aa.i32"]
    l1 = os.path.join(d, "l1")
    os.makedirs(l1)
    tw = L.run("twins", {"N": N, "fam": fam, "wmax": 512, "out": L.wp(l1) + "/cl1.i32", "threads": 2}, stream=False)
    lad = FBD.ladder(L.wp(csr), L.wp(l1), 2, N=N, V1=tw["clusters"])
    Vs = {l: lad["levels"][l]["V"] for l in (1, 2, 3, 4)}
    LOG("synthetic ladder: V per level %s" % [Vs[l] for l in (1, 2, 3, 4)])
    maps = {l: np.fromfile(os.path.join(l1, "cl%d.i32" % l), np.int32).astype(np.int64) for l in (1, 2, 3, 4)}
    xa, aa = np.fromfile(os.path.join(csr, "xa.i64"), np.int64), np.fromfile(os.path.join(csr, "aa.i32"), np.int32)
    eptr, eidx, ew, capsplit = ref_split_nets(N, [(xa, aa.astype(np.int64))], S)
    LOG("synthetic top (S %d): split cap %d, M %d, P %d, max net %d" % (S, capsplit, len(ew), len(eidx), int(np.diff(eptr).max())))
    assert int(np.diff(eptr).max()) <= capsplit and (np.diff(xa) + 1 > capsplit).sum() > 0, "no anchor above the split cap: not a test of the split rule"
    ones = np.ones(N, np.int64)
    hgs = {0: (N, eptr, eidx.astype(np.int32), ew, ones)}
    for l in (1, 2, 3, 4):
        Vc, ep, ei, ewq, vw, st = C.quotient(N, eptr, eidx.astype(np.int32), ew, ones, maps[l])
        assert Vc == Vs[l]
        hgs[l] = (Vc, ep, ei, ewq, vw)
    cap = cap_of(N)
    cw = np.bincount(maps[4], minlength=Vs[4])
    lab_top = np.zeros(Vs[4], np.int32)
    load = np.zeros(S, np.int64)
    for v in np.argsort(-cw, kind="stable"):                              # longest-processing-time start: the heaviest clusters first, each into the lightest block
        b = int(np.argmin(load))
        lab_top[v] = b
        load[b] += cw[v]
    LOG("start: max block %d (cap %d)" % (load.max(), cap))
    tmp = tmpdir("vcsynref")
    ref_lab, ref_traj = VC.vcycle(hgs, maps, LEVEL_TOP, lab_top, "fm2", S, cap, tmp)
    work = os.path.join(d, "work")
    os.makedirs(work)
    L.wr(lab_top, os.path.join(work, "zlab4.i32"), np.int32)
    rec = vcycle(fam, N, L.wp(l1), L.wp(work), Vs, 3, check=1)
    for t, e in zip(rec["trajectory"], ref_traj):
        assert t["level"] == e["level"] and t["km1_in"] == e["km1_in"] and t["km1_out"] == e["km1_out"] and t["moves"] == e["refine"]["moves"] and t["passes_run"] == e["refine"]["passes_run"], (t, e)
        if t["level"] >= 1:
            assert t["V"] == e["V"] and t["P"] == e["P"], (t, e)
        LOG("  level %d: V %d, km1 %d -> %d, %d moves == the reference chain" % (t["level"], t["V"], t["km1_in"], t["km1_out"], t["moves"]))
    cmp_arrays("V-cycle top map", np.fromfile(os.path.join(work, "lab0.i32"), np.int32).astype(np.int64), ref_lab.astype(np.int64))
    # a second call resumes: every step is skipped and the same record comes back
    rec2 = vcycle(fam, N, L.wp(l1), L.wp(work), Vs, 3, check=1)
    assert json.dumps(rec["trajectory"], sort_keys=True) == json.dumps(rec2["trajectory"], sort_keys=True)
    v = validity(L.wp(os.path.join(work, "lab0.i32")), N)
    assert v["length_ok"] and v["ids_in_range"]
    LOG("TEST_VC PASS: the engine V-cycle (levels 4..0, split-top quotients, implicit level 0) == _vc.vcycle(fm2) on the numpy split top; resumed run identical; blocks %d..%d (bound %d)" % (v["min_block"], v["max_block"], v["contract_bound_ceil_1.03_N_over_k"]))


def freebase_n():
    return int(json.load(io.open(STATS, encoding="utf-8"))["stats"]["nodes"])


def fb_vs():
    rec = json.load(io.open(os.path.join(REPO, "results", "FREEBASE_SCALE", "E3_LADDER__fbx.json"), encoding="utf-8"))
    return {int(l): int(v["V"]) for l, v in rec["levels"].items()}


def main():
    a = sys.argv[1:]
    if a[:1] == ["TEST_WEBQSP_VC"]:
        test_webqsp_vc()
    elif a[:1] == ["TEST_VC"]:
        test_vc_synth()
    elif a[:1] == ["SURROGATE"] and len(a) >= 4:
        csr, l1, work = a[1:4]
        thr = int(a[4]) if len(a) > 4 else 8
        N = freebase_n()
        rec, _ = surrogate([csr + "/xa.i64," + csr + "/aa.i32"], N, l1, work, fb_vs()[4], thr, "fbx")
        rec["N"] = N
        put_json(os.path.join(REPO, "results", "FREEBASE_SCALE", "E4_SURROGATE__fbx.json"), rec)
        print(json.dumps(rec))
    elif a[:1] == ["SURROGATE_BUILD"] and len(a) >= 4:                        # the quotient + transpose + shards only (own reservation); SURROGATE afterwards resumes from the step markers and runs Zoltan
        csr, l1, work = a[1:4]
        thr = int(a[4]) if len(a) > 4 else 8
        N = freebase_n()
        rec, _ = surrogate([csr + "/xa.i64," + csr + "/aa.i32"], N, l1, work, fb_vs()[4], thr, "fbx", with_zoltan=False)
        rec["N"] = N
        put_json(os.path.join(REPO, "results", "FREEBASE_SCALE", "E4_SURROGATE_BUILD__fbx.json"), rec)
        print(json.dumps(rec))
    elif a[:1] == ["VCYCLE"] and len(a) >= 4:
        csr, l1, work = a[1:4]
        thr = int(a[4]) if len(a) > 4 and "=" not in a[4] else 8
        opt = dict(x.split("=", 1) for x in a[4:] if "=" in x)               # levels=4-2 (hi-lo), anon_gb=NN (the engine's anonymous-memory cap = the job's reservation)
        if "anon_gb" in opt:
            os.environ["FBX_ENGINE_ANON_GB"] = opt["anon_gb"]
        hi, lo = (int(t) for t in opt.get("levels", "4-0").split("-"))
        N = freebase_n()
        rec = vcycle([csr + "/xa.i64," + csr + "/aa.i32"], N, l1, work, fb_vs(), thr, hi=hi, lo=lo)
        rec["N"] = N
        if lo == 0:                                                           # the final record: every level's marker, whichever job wrote it
            rec["trajectory"] = [json.loads(sh(["cat", "%s/vc_l%d.done.json" % (work, l)])) for l in range(LEVEL_TOP, -1, -1)]
            rec["validity"] = validity(work + "/lab0.i32", N)
            put_json(os.path.join(REPO, "results", "FREEBASE_SCALE", "E6_VCYCLE__fbx.json"), rec)
        else:
            put_json(os.path.join(REPO, "results", "FREEBASE_SCALE", "E6_VCYCLE_L%d-%d__fbx.json" % (hi, lo)), rec)
        print(json.dumps(rec))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
