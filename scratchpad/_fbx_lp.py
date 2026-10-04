"""FBX_SCALE stage 3 -- rung 3 of the partitioner ladder (addendum 2): balanced label propagation, the plain-Python reference and the C wrapper.

R3 refines the R2 LDG block table (same cap ceil(1.03 N / K), plus a mirror-image floor(0.97 N / K) so no block is drained) with asynchronous
sweeps in position order; a vertex moves to the neighbouring block with the strictly largest neighbour count among blocks with room (ties ->
smaller load, then smaller id).  The rule is spelled out in _fbx_lp.c; lp_reference below is its executable statement, and TESTLP checks the
C program against it bit for bit on random graphs (one and two adjacency streams) before the program is used on anything that is reported.

  python scratchpad/_fbx_lp.py TESTLP
"""
import hashlib
import json
import os
import sys
import tempfile

import numpy as np

import _fbx_part as P

HERE = os.path.dirname(os.path.abspath(__file__))
LO_NUM = 97
MAX_SWEEPS = 64


def bounds(N, K):
    cap = P.cap_of(N, K)
    lo = (97 * int(N)) // (100 * int(K))
    return cap, lo


def lp_reference(xa, aa, xb, ab, N, K, init, sweeps):
    """the rule of _fbx_lp.c in plain Python; returns (labels, moved per sweep)."""
    cap, lo = bounds(N, K)
    label = np.array(init, np.int64)
    load = np.bincount(label, minlength=K).astype(np.int64)
    hist = []
    for _ in range(min(int(sweeps), MAX_SWEEPS)):
        moved = 0
        for v in range(N):
            c = int(label[v])
            cnt = {}
            for xs, as_ in ((xa, aa), (xb, ab)):
                if xs is None:
                    continue
                for e in range(int(xs[v]), int(xs[v + 1])):
                    u = int(as_[e])
                    if u == v:
                        continue
                    p = int(label[u])
                    cnt[p] = cnt.get(p, 0) + 1
            bs = cnt.get(c, 0)
            best = -1
            if load[c] > lo:
                for p in sorted(cnt):
                    if p != c and load[p] < cap:
                        s = cnt[p]
                        if s > bs:
                            bs, best = s, p
                        elif best >= 0 and s == bs and (load[p] < load[best] or (load[p] == load[best] and p < best)):
                            best = p
            if best >= 0:
                label[v] = best
                load[c] -= 1
                load[best] += 1
                moved += 1
        hist.append(moved)
        if moved * 1000 < N:
            break
    return label.astype(np.int32), hist


def c_binary_lp():
    """compile _fbx_lp.c (once per source sha) into the distro's /tmp; returns (binary path inside the distro, source sha12)."""
    src = os.path.join(HERE, "_fbx_lp.c")
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    binp = "/tmp/fbx_lp_%s" % sha
    try:
        P._run(["test", "-x", binp])
    except RuntimeError:
        P._run(["gcc", "-O2", "-ffp-contract=off", "-Wall", "-o", binp, P.wsl_path(src)])
    return binp, sha


def lp_c(N, K, sweeps, init_raw, out_raw, streams):
    """R3 in C: init_raw = raw int32 block table (the LDG output); returns (int32 block table, info)."""
    binp, sha = c_binary_lp()
    out = P._run([binp, str(int(N)), str(int(K)), str(int(sweeps)), P.wsl_path(init_raw), P.wsl_path(out_raw)] + list(streams))
    info = json.loads(out.strip().splitlines()[-1])
    lab = np.fromfile(out_raw, np.int32)
    assert len(lab) == N and (lab >= 0).all() and (lab < K).all()
    info["c_source_sha12"] = sha
    return lab, info


def _testlp():
    rng = np.random.RandomState(23)
    tmp = tempfile.mkdtemp(prefix="fbxlp_")
    for (N, K, m, sw) in ((400, 7, 1500, 6), (1000, 40, 4000, 8), (257, 3, 600, 4), (2000, 200, 3000, 5), (3000, 64, 20000, 10)):
        u = rng.randint(0, N, m)
        v = rng.randint(0, N, m)
        xa, aa = P.csr_from_pairs(u, v, N)
        z = np.zeros(N + 1, np.int64)
        e = np.zeros(0, np.int32)
        init = P.ldg_reference(xa, aa, z, e, N, K)
        ref, hist = lp_reference(xa, aa, None, None, N, K, init, sw)
        sA = P.write_csr_raw(xa, aa, os.path.join(tmp, "a%d" % N))
        init_raw = os.path.join(tmp, "i%d.raw" % N)
        init.astype(np.int32).tofile(init_raw)
        Lc, info = lp_c(N, K, sw, init_raw, os.path.join(tmp, "o%d.raw" % N), sA)
        assert (Lc == ref).all(), "C LP != reference (N %d K %d)" % (N, K)
        assert info["moved_per_sweep"] == hist, (info["moved_per_sweep"], hist)
        cap, lo = bounds(N, K)
        sizes = np.bincount(Lc, minlength=K)
        assert sizes.max() <= max(cap, int(np.bincount(init, minlength=K).max())) and sizes.min() >= min(lo, int(np.bincount(init, minlength=K).min())), "balance contract broken"
        half = (xa[1:] - xa[:-1]) // 2
        xb = np.concatenate([[0], np.cumsum(half)]).astype(np.int64)
        xa2 = np.concatenate([[0], np.cumsum((xa[1:] - xa[:-1]) - half)]).astype(np.int64)
        aA = np.concatenate([aa[xa[x] + half[x]:xa[x + 1]] for x in range(N)]).astype(np.int32)
        aB = np.concatenate([aa[xa[x]:xa[x] + half[x]] for x in range(N)]).astype(np.int32)
        s2 = P.write_csr_raw(xa2, aA, os.path.join(tmp, "s%d" % N)) + P.write_csr_raw(xb, aB, os.path.join(tmp, "t%d" % N))
        L2, _ = lp_c(N, K, sw, init_raw, os.path.join(tmp, "o2_%d.raw" % N), s2)
        assert (L2 == ref).all(), "two-stream C LP != reference"
        cut0 = int((init[np.repeat(np.arange(N), np.diff(xa))] != init[aa]).sum())
        cut1 = int((Lc[np.repeat(np.arange(N), np.diff(xa))] != Lc[aa]).sum())
        print("ok N %d K %d: C LP == reference (1 and 2 streams), sweeps %s, sizes [%d, %d] (cap %d lo %d), cut entries %d -> %d" % (N, K, hist, sizes.min(), sizes.max(), cap, lo, cut0, cut1))
    print("TESTLP PASS (C source sha12 %s)" % c_binary_lp()[1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "TESTLP":
        _testlp()
    else:
        raise SystemExit(__doc__)
