"""FBX_SCALE addendum 17 / E5 -- regression of the out-of-core FM2 engine (scratchpad/_ooc2.cpp `fm2`, `transpose`) against the frozen sequential reference scratchpad/_vc_ref2.c (mode 1).

  python -u scratchpad/_ooc_fmtest.py TEST_FM        # random hypergraphs (explicit nets) + the implicit closed-neighbourhood / split-preserve source against the reference on the explicit netdump

The reference reads int64 vertex weights and prints one JSON line (km1 before / after, moves, passes, loads, stale re-pushes); the engine must reproduce the labels array and each of those numbers exactly.
No gold label is read anywhere in this file.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _ooc_lib as L  # noqa: E402
from _ooc_e1 import cmp_arrays, ref_split_nets, tmpdir, write_fams  # noqa: E402

log = L.log
REF2 = os.path.join(HERE, "_vc_ref2.c")
KEYS_FM = ("passes_run", "moves", "km1_before", "km1_after", "max_load_before", "max_load_after", "min_load_after", "stale_repushes")


def ref2_binary():
    """scratchpad/_vc_ref2.c compiled in the engine's distro (once per source sha)"""
    sha = L.sha_file(REF2)[:12]
    bindir = os.path.join(L.REPO, "data", "ooc", "bin")
    os.makedirs(bindir, exist_ok=True)
    binp = L.wp(os.path.join(bindir, "vc_ref2_%s" % sha))
    try:
        L._wsl(["test", "-x", binp])
    except RuntimeError:
        L._wsl(["gcc", "-O2", "-Wall", "-o", binp, L.wp(REF2)])
    return binp


def run_ref2(d, label, V, ep, ei, ew, vw, lab, K, cap, passes=6, maxfail=500):
    import _ml_coarsen as C
    vptr, vidx = C.transpose(V, ep, ei)
    fs = {}
    for nm, arr, dt in (("vw", vw, np.int64), ("eptr", ep, np.int64), ("eidx", ei, np.int32), ("ew", ew, np.int64), ("vptr", vptr, np.int64), ("vidx", vidx, np.int32), ("lab", lab, np.int32)):
        fs[nm] = L.wr(arr, os.path.join(d, "r2_%s_%s" % (label, nm)), dt)
    out = os.path.join(d, "r2_%s_out" % label)
    cmd = [ref2_binary(), "1", str(V), str(len(ep) - 1), str(K), str(cap), str(passes), str(maxfail)] + [L.wp(fs[x]) for x in ("vw", "eptr", "eidx", "ew", "vptr", "vidx", "lab")] + [L.wp(out)]
    info = json.loads(L._wsl(cmd).strip().splitlines()[-1])
    return np.fromfile(out, np.int32), info


def cmp_fm(label, ref_lab, ref_info, lab, info):
    cmp_arrays("fm2 labels (%s)" % label, lab.astype(np.int64), ref_lab.astype(np.int64))
    for k in KEYS_FM:
        assert ref_info[k] == info[k], (label, k, ref_info[k], info[k])


def balanced_start(rng, V, K, vw):
    lab = np.zeros(V, np.int32)
    load = np.zeros(K, np.int64)
    for v in rng.permutation(V):
        b = int(np.argmin(load))
        lab[v] = b
        load[b] += vw[v]
    return lab


def test_fm():
    """the FM2 engine (gain cache in uint16 / uint32 + an int64 side table, uint8 / int32 phi, mmap'd pins) == _vc_ref2.c mode 1; transpose == _ml_coarsen.transpose"""
    import _ml_coarsen as C
    rng = np.random.RandomState(21)
    d = tmpdir("fm")
    # (V, M, K, max net size, max net weight, number of hub nets, cache)
    cases = [(400, 900, 5, 12, 50, 0, "auto"), (1500, 4000, 12, 10, 50, 3, "auto"), (900, 3000, 25, 9, 4000, 0, "u16"), (900, 3000, 25, 9, 4000, 0, "u32"), (700, 2500, 8, 40, 60, 5, "auto")]
    for ci, (V, M, K, maxsz, wmax, nhub, cache) in enumerate(cases):
        eptr, eidx, _ = C._rand_hg(rng, V, M, maxsz)
        nets = [eidx[eptr[e]:eptr[e + 1]] for e in range(M)]
        hubsz = [256, 257, 300, 520, 700][:nhub]                           # around / above the uint8 phi limit (255) and the push limit (256)
        for hs in hubsz:
            nets.append(np.sort(rng.choice(V, size=min(hs, V), replace=False)).astype(np.int32))
        sz = np.array([len(x) for x in nets], np.int64)
        ep = np.zeros(len(nets) + 1, np.int64)
        np.cumsum(sz, out=ep[1:])
        ei = np.concatenate(nets).astype(np.int32)
        ew = rng.randint(1, wmax, len(nets)).astype(np.int64)
        vw = rng.randint(1, 6, V).astype(np.int64)
        cap = int(1.05 * vw.sum() / K)
        lab = balanced_start(rng, V, K, vw)
        ref_lab, ref_info = run_ref2(d, "c%d" % ci, V, ep, ei, ew, vw, lab, K, cap)
        assert ref_info["moves"] > 0, "the case does not move anything: not a test"
        pre = os.path.join(d, "fm_c%d" % ci)
        L.wr(ep, pre + ".eptr.i64", np.int64)
        L.wr(ei, pre + ".eidx.i32", np.int32)
        L.wr(ew, pre + ".ew.i64", np.int64)
        L.wr(vw, pre + ".vw.i32", np.int32)
        L.wr(lab, pre + ".lab_in.i32", np.int32)
        L.run("transpose", {"eptr": L.wp(pre + ".eptr.i64"), "eidx": L.wp(pre + ".eidx.i32"), "vin": V, "out": L.wp(pre)}, stream=False)
        vptr, vidx = C.transpose(V, ep, ei)
        cmp_arrays("transpose vptr", L.rd(pre + ".vptr.i64", np.int64), vptr)
        cmp_arrays("transpose vidx", L.rd(pre + ".vidx.i32", np.int32), vidx)
        info = L.run("fm2", {"src": "expl", "eptr": L.wp(pre + ".eptr.i64"), "eidx": L.wp(pre + ".eidx.i32"), "ew": L.wp(pre + ".ew.i64"), "vptr": L.wp(pre + ".vptr.i64"), "vidx": L.wp(pre + ".vidx.i32"),
                                "vw": L.wp(pre + ".vw.i32"), "vin": V, "k": K, "cap": cap, "lab_in": L.wp(pre + ".lab_in.i32"), "lab_out": L.wp(pre + ".lab_out.i32"), "check": 1, "cache": cache,
                                "threads": 3 if ci % 2 else 8}, stream=False)
        cmp_fm("case %d (%s)" % (ci, cache), ref_lab, ref_info, L.rd(pre + ".lab_out.i32", np.int32), info)
        log("  fm2 case %d: V %d, M %d, K %d, cache %d B/entry, %d moves in %d passes, km1 %d -> %d == _vc_ref2 (heap peak %d, pushes %d)" % (
            ci, V, len(ep) - 1, K, info["cache_bytes_per_entry"], info["moves"], info["passes_run"], info["km1_before"], info["km1_after"], info["heap_peak_entries"], info["pushes"]))
    log("TEST fm2 (explicit) PASS: 5 random hypergraphs (uneven vertex weights, nets of 256..700 pins, heavy-weight vertices in the int64 side table, both cache widths) == _vc_ref2.c mode 1; transpose == _ml_coarsen.transpose")


def test_fm_implicit():
    """level 0: the implicit closed-neighbourhood / split-preserve source == the explicit netdump (the numpy builder) fed to _vc_ref2.c (unit vertex weights)"""
    import _ml2_coarsen as C2
    rng = np.random.RandomState(31)
    d = tmpdir("fmi")
    for ci, (N, pairs, nfam, S, K) in enumerate(((600, 1800, 2, 12, 5), (900, 2600, 2, 40, 9), (800, 2400, 1, 200, 25), (500, 1500, 2, 30, 6), (700, 2000, 2, 0, 7))):
        fams = C2._rand_fams(rng, N, pairs, nfam)
        xa0, aa0 = fams[0]
        extra = rng.choice(N, N // 3, replace=False)                      # three heavy hubs in family 0 (a symmetric edge set)
        cur = set()
        for x in range(N):
            for y in aa0[xa0[x]:xa0[x + 1]]:
                if x < y:
                    cur.add((x, int(y)))
        for h in (7, 99 % N, 311 % N):
            for y in extra:
                if h != y:
                    cur.add((min(h, int(y)), max(h, int(y))))
        uu = np.array([k[0] for k in cur], np.int64)
        vv = np.array([k[1] for k in cur], np.int64)
        src_ = np.concatenate([uu, vv])
        dst_ = np.concatenate([vv, uu])
        o = np.lexsort((dst_, src_))
        xa = np.zeros(N + 1, np.int64)
        np.cumsum(np.bincount(src_, minlength=N), out=xa[1:])
        fams = [(xa, dst_[o].astype(np.int32))] + fams[1:]
        fa = write_fams(d, fams, "fi%d_" % ci)
        if S > 0:
            eptr, eidx, ew, cap_split = ref_split_nets(N, [(x, a.astype(np.int64)) for x, a in fams], S)
        else:                                                              # no split: one closed-neighbourhood net per anchor per family, weight round(1000 / degree)
            eps, eis, ews = [], [], []
            for x, a in fams:
                dg = np.diff(x)
                an = np.flatnonzero(dg >= 1)
                nets = [np.concatenate(([u], a[x[u]:x[u + 1]])) for u in an]
                eps.append(np.array([len(n) for n in nets], np.int64))
                eis.append(np.concatenate(nets))
                ews.append(np.maximum(1, np.rint(1000 / (eps[-1] - 1)).astype(np.int64)))
            sz = np.concatenate(eps)
            eptr = np.zeros(len(sz) + 1, np.int64)
            np.cumsum(sz, out=eptr[1:])
            eidx, ew, cap_split = np.concatenate(eis).astype(np.int32), np.concatenate(ews), 0
        cap = int(1.05 * N / K)                                            # head-room so that the start is not a perfectly full packing (no feasible move)
        lab = (rng.permutation(N) % K).astype(np.int32)
        ref_lab, ref_info = run_ref2(d, "i%d" % ci, N, eptr, eidx, ew, np.ones(N, np.int64), lab, K, cap)
        assert ref_info["moves"] > 0
        labf = L.wr(lab, os.path.join(d, "fi%d.lab_in.i32" % ci), np.int32)
        outf = os.path.join(d, "fi%d.lab_out.i32" % ci)
        a = {"src": "csr", "N": N, "fam": fa, "k": K, "cap": cap, "lab_in": L.wp(labf), "lab_out": L.wp(outf), "check": 1, "threads": 4}
        if S > 0:
            a["split_s"] = S
        info = L.run("fm2", a, stream=False)
        cmp_fm("implicit %d" % ci, ref_lab, ref_info, L.rd(outf, np.int32), info)
        log("  fm2 implicit case %d: N %d, %d families, split S %d (cap %d), K %d: %d moves, km1 %d -> %d == _vc_ref2 on the explicit nets" % (
            ci, N, len(fams), S, cap_split, K, info["moves"], info["km1_before"], info["km1_after"]))
    log("TEST fm2 (implicit level 0) PASS: closed-neighbourhood nets with chunked hubs == the explicit hypergraph under _vc_ref2.c, labels / km1 / moves / loads / re-pushes identical")


def test_shards():
    """E4: the engine's Zoltan shards (fmt 3, vertex-weighted) are byte-identical to _ml_run.write_shards_vw; the manifest differs only in the path prefix"""
    import _ml_coarsen as C
    import _ml_run as MR
    import _l1h_host as H
    rng = np.random.RandomState(41)
    d = tmpdir("shards")
    for ci, (V, M, maxsz) in enumerate(((12345, 9000, 4), (5000, 3000, 3), (5001, 20000, 4))):          # shard boundary cases: V a multiple of 5000, one vertex over; isolated vertices
        eptr, eidx, ew = C._rand_hg(rng, V, M, maxsz)
        vw = rng.randint(1, 900, V).astype(np.int64)
        pre = os.path.join(d, "sh%d" % ci)
        L.wr(eptr, pre + ".eptr.i64", np.int64)
        L.wr(eidx, pre + ".eidx.i32", np.int32)
        L.wr(ew, pre + ".ew.i64", np.int64)
        L.wr(vw, pre + ".vw.i32", np.int32)
        L.run("transpose", {"eptr": L.wp(pre + ".eptr.i64"), "eidx": L.wp(pre + ".eidx.i32"), "vin": V, "out": L.wp(pre)}, stream=False)
        sd_e = os.path.join(d, "eng%d" % ci)
        os.makedirs(sd_e)
        r = L.run("shards", {"eptr": L.wp(pre + ".eptr.i64"), "ew": L.wp(pre + ".ew.i64"), "vptr": L.wp(pre + ".vptr.i64"), "vidx": L.wp(pre + ".vidx.i32"), "vw": L.wp(pre + ".vw.i32"), "vin": V,
                             "out_dir": L.wp(sd_e), "path_prefix": "/P", "shard_nodes": H.SHARD_NODES, "threads": 4}, stream=False)
        sd_p = os.path.join(d, "py%d" % ci)
        shards, smf = MR.write_shards_vw(V, eptr.astype(np.int64), eidx.astype(np.int64), ew, vw, sd_p, with_vw=True)
        assert r["shards"] == len(shards) and r["P"] == len(eidx)
        for c in shards:
            a = open(os.path.join(sd_e, os.path.basename(c["file"])), "rb").read()
            b = open(c["file"], "rb").read()
            assert a == b, "shard %s differs" % os.path.basename(c["file"])
        me = open(os.path.join(sd_e, "stream_manifest.txt"), encoding="utf-8").read().split("\n")
        mp = open(smf, encoding="utf-8").read().split("\n")
        assert me[0] == mp[0] == "%d %d 3" % (V, M) and len(me) == len(mp)
        for x, y in zip(me[1:], mp[1:]):
            if x:
                assert x.split()[0] == y.split()[0] and x.split()[1] == "/P/" + os.path.basename(y.split()[1]), (x, y)
        log("  shards case %d: V %d, M %d, P %d, %d shards, %d bytes == write_shards_vw" % (ci, V, M, len(eidx), len(shards), r["bytes"]))
    log("TEST shards PASS: byte-identical to _ml_run.write_shards_vw (isolated vertices, boundary shard sizes); manifest equal up to the path prefix")


def main():
    a = sys.argv[1:]
    if a[:1] == ["TEST_FM"]:
        test_fm()
        test_fm_implicit()
    elif a[:1] == ["TEST_SHARDS"]:
        test_shards()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
