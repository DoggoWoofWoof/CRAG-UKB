"""FBX_SCALE stage 3 -- rung 3b (addendum 3): closed-neighbourhood-connectivity label propagation; the plain-Python reference and the C wrapper.

The rule is spelled out in _fbx_cn.c; cn_reference below is its executable statement, and TESTCN checks the C program against it bit for bit on random graphs (one and two
adjacency streams, with and without self loops and multi-edges) and checks on multi-edge-free graphs that the reported KM1 after equals the set-based definition
sum_u (|labels of {u} U N(u)| - 1) and that the summed chosen deltas explain the change.

  python scratchpad/_fbx_cn.py TESTCN
"""
import hashlib
import json
import os
import sys
import tempfile

import numpy as np

import _fbx_part as P
import _fbx_lp as LP

HERE = os.path.dirname(os.path.abspath(__file__))


def cn_reference(xa, aa, xb, ab, N, K, init, sweeps, T, dmax):
    """the rule of _fbx_cn.c in plain Python; returns (labels, moved per sweep, km1_before, km1_after, deltas per sweep)."""
    cap, lo = LP.bounds(N, K)
    label = [int(x) for x in init]
    load = np.bincount(np.asarray(label), minlength=K).astype(np.int64).tolist()

    def entries(v):
        out = []
        for xs, as_ in ((xa, aa), (xb, ab)):
            if xs is not None:
                out.extend(int(w) for w in as_[int(xs[v]):int(xs[v + 1])])
        return out

    ent = [entries(v) for v in range(N)]
    adj = [[w for w in ent[v] if w != v] for v in range(N)]
    hist = []
    for u in range(N):
        h = {label[u]: 1}
        for w in adj[u]:
            h[label[w]] = h.get(label[w], 0) + 1
        hist.append(h)
    km1_before = sum(len(h) - 1 for h in hist)
    moved_hist, delta_hist = [], []

    def dec(u, b):
        hist[u][b] -= 1
        if hist[u][b] == 0:
            del hist[u][b]

    for _ in range(min(int(sweeps), LP.MAX_SWEEPS)):
        moved, dsum = 0, 0
        for v in range(N):
            dv = len(ent[v])
            if dv == 0 or dv > dmax:
                continue
            c = label[v]
            if load[c] <= lo:
                continue
            nb = {}
            for w in adj[v]:
                nb[label[w]] = nb.get(label[w], 0) + 1
            cands = sorted(((-s, p) for p, s in nb.items() if p != c and load[p] < cap))[:T]
            best, bd, bn = -1, 0, 0
            if cands:
                base = -(hist[v].get(c, 0) == 1) - sum(1 for u in adj[v] if hist[u].get(c, 0) == 1)
                for ns, b in cands:
                    d = base + (hist[v].get(b, 0) == 0) + sum(1 for u in adj[v] if hist[u].get(b, 0) == 0)
                    s = -ns
                    if d < 0 and (best < 0 or d < bd or (d == bd and (s > bn or (s == bn and (load[b] < load[best] or (load[b] == load[best] and b < best)))))):
                        best, bd, bn = b, d, s
            if best >= 0:
                label[v] = best
                load[c] -= 1
                load[best] += 1
                moved += 1
                dsum += bd
                dec(v, c)
                hist[v][best] = hist[v].get(best, 0) + 1
                for u in adj[v]:
                    dec(u, c)
                    hist[u][best] = hist[u].get(best, 0) + 1
        moved_hist.append(moved)
        delta_hist.append(dsum)
        if moved * 1000 < N:
            break
    km1_after = sum(len(h) - 1 for h in hist)
    return np.array(label, np.int32), moved_hist, km1_before, km1_after, delta_hist


def c_binary_cn():
    """compile _fbx_cn.c (once per source sha) into the distro's /tmp; returns (binary path inside the distro, source sha12)."""
    src = os.path.join(HERE, "_fbx_cn.c")
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    binp = "/tmp/fbx_cn_%s" % sha
    try:
        P._run(["test", "-x", binp])
    except RuntimeError:
        P._run(["gcc", "-O2", "-ffp-contract=off", "-Wall", "-o", binp, P.wsl_path(src)])
    return binp, sha


def cn_c(N, K, sweeps, T, dmax, init_raw, out_raw, streams):
    """R3b in C: init_raw = raw int32 block table; returns (int32 block table, info)."""
    binp, sha = c_binary_cn()
    out = P._run([binp, str(int(N)), str(int(K)), str(int(sweeps)), str(int(T)), str(int(dmax)), P.wsl_path(init_raw), P.wsl_path(out_raw)] + list(streams))
    info = json.loads(out.strip().splitlines()[-1])
    lab = np.fromfile(out_raw, np.int32)
    assert len(lab) == N and (lab >= 0).all() and (lab < K).all()
    info["c_source_sha12"] = sha
    return lab, info


def _set_km1(label, xa, aa, N):
    return sum(len({int(label[u])} | {int(label[w]) for w in aa[int(xa[u]):int(xa[u + 1])] if int(w) != u}) - 1 for u in range(N))


def _testcn():
    rng = np.random.RandomState(31)
    tmp = tempfile.mkdtemp(prefix="fbxcn_")
    cases = ((300, 6, 700, 6, 3, 1000, True), (600, 12, 1500, 8, 4, 1000, True), (257, 3, 500, 4, 2, 1000, True), (500, 30, 1600, 6, 4, 9, True), (800, 16, 5000, 8, 4, 1000, False),
             (1200, 40, 3000, 8, 3, 1000, True))
    for ci, (N, K, m, sw, T, dmax, unique) in enumerate(cases):
        u = rng.randint(0, N, m)
        v = rng.randint(0, N, m)
        if unique:
            keep = u != v
            k = np.unique(np.minimum(u[keep], v[keep]) * N + np.maximum(u[keep], v[keep]))
            u, v = k // N, k % N
        xa, aa = P.csr_from_pairs(u, v, N)
        z = np.zeros(N + 1, np.int64)
        e = np.zeros(0, np.int32)
        init = P.ldg_reference(xa, aa, z, e, N, K)
        init, _ = LP.lp_reference(xa, aa, None, None, N, K, init, 4)
        ref, hist, k0, k1, dh = cn_reference(xa, aa, None, None, N, K, init, sw, T, dmax)
        sA = P.write_csr_raw(xa, aa, os.path.join(tmp, "a%d" % ci))
        init_raw = os.path.join(tmp, "i%d.raw" % ci)
        init.astype(np.int32).tofile(init_raw)
        Lc, info = cn_c(N, K, sw, T, dmax, init_raw, os.path.join(tmp, "o%d.raw" % ci), sA)
        assert (Lc == ref).all(), "C CN != reference (case %d)" % ci
        assert info["moved_per_sweep"] == hist and info["km1_delta_per_sweep"] == dh and info["km1_before"] == k0 and info["km1_after"] == k1, (info, hist, dh, k0, k1)
        cap, lo = LP.bounds(N, K)
        sizes = np.bincount(Lc, minlength=K)
        i0 = np.bincount(init, minlength=K)
        assert sizes.max() <= max(cap, int(i0.max())) and sizes.min() >= min(lo, int(i0.min())), "balance contract broken"
        if unique:
            assert info["km1_bookkeeping_consistent"] == 1 and _set_km1(init, xa, aa, N) == k0 and _set_km1(Lc, xa, aa, N) == k1, "KM1 != set-based definition"
        half = (xa[1:] - xa[:-1]) // 2
        xb = np.concatenate([[0], np.cumsum(half)]).astype(np.int64)
        xa2 = np.concatenate([[0], np.cumsum((xa[1:] - xa[:-1]) - half)]).astype(np.int64)
        aA = np.concatenate([aa[xa[x] + half[x]:xa[x + 1]] for x in range(N)]).astype(np.int32)
        aB = np.concatenate([aa[xa[x]:xa[x] + half[x]] for x in range(N)]).astype(np.int32)
        s2 = P.write_csr_raw(xa2, aA, os.path.join(tmp, "s%d" % ci)) + P.write_csr_raw(xb, aB, os.path.join(tmp, "t%d" % ci))
        L2, _ = cn_c(N, K, sw, T, dmax, init_raw, os.path.join(tmp, "o2_%d.raw" % ci), s2)
        assert (L2 == ref).all(), "two-stream C CN != reference (case %d)" % ci
        print("ok case %d N %d K %d T %d dmax %d%s: C CN == reference (1 and 2 streams), sweeps moved %s, KM1 %d -> %d (deltas %s), sizes [%d, %d]" % (
            ci, N, K, T, dmax, "" if unique else " (multi-edges)", hist, k0, k1, dh, sizes.min(), sizes.max()))
    print("TESTCN PASS (C source sha12 %s)" % c_binary_cn()[1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "TESTCN":
        _testcn()
    else:
        raise SystemExit(__doc__)
