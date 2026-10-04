"""FBX_SCALE stage 4C -- the shared boundary refinement of the SK-arm lab: wrapper and tests for _ml2_cn.c (layers = edge families, repair before the sweeps).

  python scratchpad/_ml2_cn.py TEST

TEST (on the machine that will run it):
  1. one layer, balanced start  -> _ml2_cn output == _fbx_cn output bit for bit (the sweeps are unchanged; the repair is skipped)
  2. two layers                 -> the reported KM1 before / after equals the set-based sum over layers of (|blocks of {u} U N_layer(u)| - 1); the summed deltas explain the change
  3. unbalanced start           -> after repair and sweeps every block is <= cap, the load table sums to N, repair moves are reported and the delta bookkeeping is exact
  4. determinism                -> two runs give identical bytes
  5. anchor weights            -> the reported KM1 is the weighted set-based KM1; unit weights reproduce the unweighted run; the repair under weights keeps the bookkeeping exact
"""
import hashlib
import json
import os
import sys
import tempfile

import numpy as np

import _fbx_cn as CN
import _fbx_part as P

HERE = os.path.dirname(os.path.abspath(__file__))


def c_binary():
    src = os.path.join(HERE, "_ml2_cn.c")
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    binp = "/tmp/ml2_cn_%s" % sha
    try:
        P._run(["test", "-x", binp])
    except RuntimeError:
        P._run(["gcc", "-O2", "-Wall", "-o", binp, P.wsl_path(src)])
    return binp, sha


def anchor_weights(xadj):
    """the frozen H4 net weight of the anchor's closed neighbourhood: max(1, rint(1000 / deg)) for deg >= 1 (0 for an isolated vertex, whose net is empty); int32."""
    deg = np.diff(xadj).astype(np.int64)
    w = np.zeros(len(deg), np.int64)
    m = deg >= 1
    w[m] = np.maximum(1, np.rint(1000 / deg[m]).astype(np.int64))
    return w.astype(np.int32)


def write_weights(w, path):
    np.ascontiguousarray(w, np.int32).tofile(path)
    return "w=" + P.wsl_path(path)


def cn2_c(N, K, sweeps, T, dmax, init_raw, out_raw, layers, weights=()):
    """layers = flat list of the argument specs of one or two (xadj, adj) pairs; weights = () or one 'w=<file>' spec per layer; returns (int32 block table, info)."""
    binp, sha = c_binary()
    out = P._run([binp, str(int(N)), str(int(K)), str(int(sweeps)), str(int(T)), str(int(dmax)), P.wsl_path(init_raw), P.wsl_path(out_raw)] + list(layers) + list(weights))
    info = json.loads(out.strip().splitlines()[-1])
    lab = np.fromfile(out_raw, np.int32)
    assert len(lab) == N and (lab >= 0).all() and (lab < K).all()
    info["c_source_sha12"] = sha
    return lab, info


def set_km1(label, xa, aa, N, w=None):
    return sum((1 if w is None else int(w[u])) * (len({int(label[u])} | {int(label[x]) for x in aa[int(xa[u]):int(xa[u + 1])] if int(x) != u}) - 1) for u in range(N))


def _rand_layer(rng, N, m):
    u = rng.randint(0, N, m)
    v = rng.randint(0, N, m)
    keep = u != v
    k = np.unique(np.minimum(u[keep], v[keep]) * N + np.maximum(u[keep], v[keep]))
    return P.csr_from_pairs(k // N, k % N, N)


def _test():
    rng = np.random.RandomState(47)
    tmp = tempfile.mkdtemp(prefix="ml2cn_")
    n_repairs = 0
    for ci, (N, K, m1, m2, sw, T, dmax) in enumerate(((300, 6, 700, 900, 6, 3, 1000), (600, 12, 1500, 2500, 8, 4, 1000), (1200, 40, 3000, 5000, 8, 3, 1000), (500, 30, 1600, 900, 6, 4, 9))):
        xa, aa = _rand_layer(rng, N, m1)
        xb, ab = _rand_layer(rng, N, m2)
        z = np.zeros(N + 1, np.int64)
        e0 = np.zeros(0, np.int32)
        sA = P.write_csr_raw(xa, aa, os.path.join(tmp, "a%d" % ci))
        sB = P.write_csr_raw(xb, ab, os.path.join(tmp, "b%d" % ci))
        cap = -(-103 * N // (100 * K))
        # 1. balanced start, one layer == _fbx_cn
        init = P.ldg_reference(xa, aa, z, e0, N, K)
        assert np.bincount(init, minlength=K).max() <= cap
        ir = os.path.join(tmp, "i%d.raw" % ci)
        init.astype(np.int32).tofile(ir)
        l0, i0 = CN.cn_c(N, K, sw, T, dmax, ir, os.path.join(tmp, "o0_%d.raw" % ci), sA)
        l1, i1 = cn2_c(N, K, sw, T, dmax, ir, os.path.join(tmp, "o1_%d.raw" % ci), sA)
        assert (l0 == l1).all(), "one-layer ml2_cn != fbx_cn (case %d)" % ci
        assert i1["repair"]["moves"] == 0 and i1["km1_before"] == i0["km1_before"] and i1["km1_after"] == i0["km1_after"] and i1["moved_per_sweep"] == i0["moved_per_sweep"]
        # 2. two layers, set-based KM1 per layer
        l2, i2 = cn2_c(N, K, sw, T, dmax, ir, os.path.join(tmp, "o2_%d.raw" % ci), sA + sB)
        k0 = set_km1(init, xa, aa, N) + set_km1(init, xb, ab, N)
        k1 = set_km1(l2, xa, aa, N) + set_km1(l2, xb, ab, N)
        assert i2["km1_before"] == k0 and i2["km1_after"] == k1 and i2["km1_bookkeeping_consistent"] == 1, (i2["km1_before"], k0, i2["km1_after"], k1)
        assert k1 <= k0
        l2b, _ = cn2_c(N, K, sw, T, dmax, ir, os.path.join(tmp, "o2b_%d.raw" % ci), sA + sB)
        assert (l2 == l2b).all(), "not deterministic"
        # 3. unbalanced start: a quarter of the vertices piled into block 0, the rest spread by LDG order; repair must fix it
        skew = init.copy()
        pile = rng.choice(N, size=max(2, int(0.25 * N)), replace=False)
        skew[pile] = 0
        assert np.bincount(skew, minlength=K).max() > cap
        sr = os.path.join(tmp, "s%d.raw" % ci)
        skew.astype(np.int32).tofile(sr)
        l3, i3 = cn2_c(N, K, sw, T, dmax, sr, os.path.join(tmp, "o3_%d.raw" % ci), sA + sB)
        sizes = np.bincount(l3, minlength=K)
        assert sizes.max() <= cap and int(sizes.sum()) == N, (sizes.max(), cap)
        assert i3["repair"]["overfull_blocks_before"] >= 1 and i3["repair"]["moves"] > 0 and i3["repair"]["overfull_blocks_after"] == 0 and i3["repair"]["stuck"] == 0
        assert i3["km1_bookkeeping_consistent"] == 1
        assert i3["km1_before"] == set_km1(skew, xa, aa, N) + set_km1(skew, xb, ab, N) and i3["km1_after"] == set_km1(l3, xa, aa, N) + set_km1(l3, xb, ab, N)
        # 5. weighted anchors: the reported KM1 is the weighted set-based one, unit weights reproduce the unweighted run, a weighted run never raises the weighted KM1 on a balanced start
        wA, wB = anchor_weights(xa), anchor_weights(xb)
        wsp = [write_weights(wA, os.path.join(tmp, "wa%d.i32" % ci)), write_weights(wB, os.path.join(tmp, "wb%d.i32" % ci))]
        l5, i5 = cn2_c(N, K, sw, T, dmax, ir, os.path.join(tmp, "o5_%d.raw" % ci), sA + sB, wsp)
        assert i5["weighted"] == 1 and i5["km1_bookkeeping_consistent"] == 1
        assert i5["km1_before"] == set_km1(init, xa, aa, N, wA) + set_km1(init, xb, ab, N, wB) and i5["km1_after"] == set_km1(l5, xa, aa, N, wA) + set_km1(l5, xb, ab, N, wB)
        assert i5["km1_after"] <= i5["km1_before"]
        ones = [write_weights(np.ones(N, np.int32), os.path.join(tmp, "u%d.i32" % ci))] * 2
        l6, _ = cn2_c(N, K, sw, T, dmax, ir, os.path.join(tmp, "o6_%d.raw" % ci), sA + sB, ones)
        assert (l6 == l2).all(), "unit weights != no weights"
        l7, i7 = cn2_c(N, K, sw, T, dmax, sr, os.path.join(tmp, "o7_%d.raw" % ci), sA + sB, wsp)
        assert np.bincount(l7, minlength=K).max() <= cap and i7["km1_bookkeeping_consistent"] == 1 and i7["repair"]["overfull_blocks_after"] == 0
        n_repairs += i3["repair"]["moves"]
        print("ok case %d N %d K %d: 1-layer == fbx_cn; 2-layer KM1 %d -> %d (set-based); skew: max load %d -> %d (cap %d), %d repair moves (dKM1 %d), %d passes" % (
            ci, N, K, k0, k1, i3["repair"]["max_load_before"], sizes.max(), cap, i3["repair"]["moves"], i3["repair"]["km1_delta"], i3["repair"]["passes"]))
    print("TEST PASS (C source sha12 %s; %d repair moves checked)" % (c_binary()[1], n_repairs))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "TEST":
        _test()
    else:
        raise SystemExit(__doc__)
