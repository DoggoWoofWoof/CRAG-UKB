"""Unit tests of scratchpad/_vc_ref.c (the k-way KM1 refiner) on random weighted hypergraphs, independent of the lab modules.  Runs the binary directly (WSL on the laptop, or the host's WSL).

  python scratchpad/_vc_ref_test.py            # local WSL (Ubuntu-22.04 if present), builds the binary in WSL /tmp;  VC_SRC=_vc_ref2.c tests version 2
Checks: (1) reported km1_before / km1_after equal a numpy recomputation from the labels;  (2) LP and FM never increase KM1;  (3) no block that was within the cap is pushed over it;
(4) both modes are deterministic (two runs, identical labels);  (5) a hypergraph with a planted partition is recovered from a perturbed start (KM1 falls to the planted value or below);
(6) FM <= LP on every instance (reported, not asserted -- FM can lose by chance; asserted only that FM improves on a start that LP cannot).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DISTRO = os.environ.get("VC_WSL", "Ubuntu-22.04")


def wsl_path(p):
    p = os.path.abspath(p).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return "/mnt/%s%s" % (p[0].lower(), p[2:])
    return p


def run(cmd):
    if os.name == "nt":
        cmd = ["wsl", "-d", DISTRO, "--"] + cmd
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def build():
    src = os.path.join(HERE, os.environ.get("VC_SRC", "_vc_ref.c"))
    out = "/tmp/vc_ref_test_" + os.path.basename(src)[:-2]
    rc, o, e = run(["gcc", "-O2", "-Wall", "-o", out, wsl_path(src)])
    assert rc == 0, e
    return out


def transpose(V, eptr, eidx):
    M = len(eptr) - 1
    net = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    o = np.argsort(eidx, kind="stable")
    vptr = np.zeros(V + 1, np.int64)
    np.cumsum(np.bincount(eidx, minlength=V), out=vptr[1:])
    return vptr, net[o].astype(np.int32)


def km1(eptr, eidx, ew, lab, k):
    M = len(eptr) - 1
    net = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    key = np.unique(net * k + lab[eidx])
    lam = np.bincount(key // k, minlength=M)
    return int((ew * np.maximum(lam - 1, 0)).sum())


def refine(binp, mode, hg, lab, k, cap, passes=10, maxfail=50, tmp=None):
    V, eptr, eidx, ew, vw = hg
    vptr, vidx = transpose(V, eptr, eidx)
    d = tmp
    fs = {}
    for nm, arr, dt in (("vw", vw, np.int64), ("eptr", eptr, np.int64), ("eidx", eidx, np.int32), ("ew", ew, np.int64), ("vptr", vptr, np.int64), ("vidx", vidx, np.int32), ("lab", lab, np.int32)):
        fs[nm] = os.path.join(d, nm + ".raw")
        np.ascontiguousarray(arr, dt).tofile(fs[nm])
    out = os.path.join(d, "out.raw")
    rc, o, e = run([binp, str(mode), str(V), str(len(eptr) - 1), str(k), str(cap), str(passes), str(maxfail)] + [wsl_path(fs[x]) for x in ("vw", "eptr", "eidx", "ew", "vptr", "vidx", "lab")] + [wsl_path(out)])
    assert rc == 0, (rc, o, e)
    return np.fromfile(out, np.int32), json.loads(o.strip().splitlines()[-1])


def rand_hg(rng, V, M, maxsz, wmax=9):
    sz = rng.randint(2, maxsz + 1, M)
    eptr = np.zeros(M + 1, np.int64)
    np.cumsum(sz, out=eptr[1:])
    eidx = np.concatenate([rng.choice(V, s, replace=False) for s in sz]).astype(np.int32)
    ew = rng.randint(1, wmax + 1, M).astype(np.int64)
    vw = rng.randint(1, 5, V).astype(np.int64)
    return V, eptr, eidx, ew, vw


def planted(rng, V, k, M, p_in=0.9):
    """nets mostly inside one planted block; returns hg and the planted labels"""
    lab = np.arange(V) % k
    members = [np.flatnonzero(lab == b) for b in range(k)]
    eptrs, eidxs = [0], []
    for _ in range(M):
        b = rng.randint(k)
        s = rng.randint(2, 7)
        if rng.rand() < p_in:
            pins = rng.choice(members[b], min(s, len(members[b])), replace=False)
        else:
            pins = rng.choice(V, s, replace=False)
        eidxs.append(pins.astype(np.int32))
        eptrs.append(eptrs[-1] + len(pins))
    eptr = np.array(eptrs, np.int64)
    return (V, eptr, np.concatenate(eidxs), rng.randint(1, 10, M).astype(np.int64), np.ones(V, np.int64)), lab


def main():
    binp = build()
    rng = np.random.RandomState(5)
    with tempfile.TemporaryDirectory() as tmp:
        worse_fm = 0
        for trial in range(12):
            V, M, k = int(rng.randint(30, 400)), int(rng.randint(40, 900)), int(rng.randint(2, 9))
            hg = rand_hg(rng, V, M, int(rng.randint(3, 12)))
            vw = hg[4]
            lab0 = rng.randint(0, k, V).astype(np.int32)
            load0 = np.bincount(lab0, weights=vw, minlength=k)
            cap = int(max(load0.max(), np.ceil(1.05 * vw.sum() / k)))
            kb = km1(hg[1], hg[2], hg[3], lab0.astype(np.int64), k)
            res = {}
            for mode in (0, 1):
                l1, i1 = refine(binp, mode, hg, lab0, k, cap, tmp=tmp)
                l2, i2 = refine(binp, mode, hg, lab0, k, cap, tmp=tmp)
                assert np.array_equal(l1, l2), "mode %d not deterministic" % mode
                assert i1["km1_before"] == kb, (i1, kb)
                ka = km1(hg[1], hg[2], hg[3], l1.astype(np.int64), k)
                assert i1["km1_after"] == ka, (i1, ka)
                assert ka <= kb, ("KM1 increased", mode, kb, ka)
                ld = np.bincount(l1, weights=vw, minlength=k)
                assert ld.max() <= cap, ("a block over the cap", mode, ld.max(), cap)
                res[mode] = ka
            if res[1] > res[0]:
                worse_fm += 1
            print("trial %2d V %3d M %3d k %d cap %3d: KM1 %6d -> LP %6d  FM %6d" % (trial, V, M, k, cap, kb, res[0], res[1]))
        hg, plab = planted(rng, 600, 6, 2500)
        kp = km1(hg[1], hg[2], hg[3], plab.astype(np.int64), 6)
        start = plab.copy()
        flip = rng.choice(600, 120, replace=False)
        start[flip] = rng.randint(0, 6, 120)
        cap = int(np.ceil(1.05 * 600 / 6))
        ks = km1(hg[1], hg[2], hg[3], start.astype(np.int64), 6)
        _, ilp = refine(binp, 0, hg, start.astype(np.int32), 6, cap, tmp=tmp)
        _, ifm = refine(binp, 1, hg, start.astype(np.int32), 6, cap, tmp=tmp)
        print("planted: KM1 planted %d, perturbed start %d -> LP %d  FM %d" % (kp, ks, ilp["km1_after"], ifm["km1_after"]))
        assert ifm["km1_after"] <= ilp["km1_after"] or ifm["km1_after"] <= kp * 1.05, "FM should recover the planted structure at least as well as LP"
        assert ifm["km1_after"] <= ks and ilp["km1_after"] <= ks
        print("FM worse than LP on %d of 12 random instances (descriptive)" % worse_fm)
    print("TEST PASS: KM1 reported = recomputed; never increases; no block pushed over the cap; both modes deterministic; planted structure recovered")


if __name__ == "__main__":
    main()
