"""FBX_SCALE stage 3 -- the low-memory partitioners of the calibration ladder (addendum 1, results/FREEBASE_SCALE/): R1 balanced hash and R2 LDG.

Both are deterministic and parameter-free.  Both read only a CSR (or, for Freebase, the served OUT and IN CSRs as two adjacency streams) and
hold O(N) memory: the block table (int32), the loads (int64, K entries) and a heap of K keys.

  hash_balanced(N, K, seed)      nodes ordered by splitmix64(position xor seed) (ties impossible: the map is a bijection on 64 bits);
                                 block(order[i]) = i mod K, so block sizes differ by at most 1.
  ldg(xa, aa, xb, ab, N, K)      Linear Deterministic Greedy, one pass in position order, hard capacity C = ceil(1.03 N / K):
                                     v -> argmax_{i: load_i < C} |N(v) n P_i| * (1 - load_i / C)
                                 (evaluated as the exact integer |N(v) n P_i| * (C - load_i)) over the blocks that hold a placed neighbour (ties -> smaller load, then smaller id); a vertex with no placed
                                 neighbour, or whose touched blocks are all full, goes to the least-loaded block (ties -> smaller id).
                                 Neighbours = the entries of adjacency stream A (rows xa[v]:xa[v+1] of aa) and stream B (xb, ab); a undirected
                                 CSR is stream A with an empty stream B.  Multi-edges count with their multiplicity.
  map_stats(...)                 block sizes, cut, closed-neighbourhood connectivity (the objective of the reference hypergraph, per anchor).

Usage (smoke): python scratchpad/_fbx_part.py TEST
"""
import hashlib
import heapq
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                  # the GPU host's env has no numba: the C twin (_fbx_ldg.c) is what runs there
    HAVE_NUMBA = False

    def njit(*a, **k):
        return lambda f: f

BAL = 1.03


def cap_of(N, K):
    """ceil(1.03 N / K) in exact integer arithmetic: 1.03 = 103 / 100."""
    return -((-103 * int(N)) // (100 * int(K)))


def splitmix64(x):
    x = x.astype(np.uint64, copy=True)
    x += np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def hash_balanced(N, K, seed):
    """balanced hash placement (R1): rank of splitmix64(position xor splitmix64(seed)) mod K; sizes floor(N/K) or ceil(N/K)."""
    with np.errstate(over="ignore"):
        s = splitmix64(np.array([seed], np.uint64))[0]
        key = splitmix64(np.arange(N, dtype=np.uint64) ^ s)
        order = np.argsort(key, kind="stable")
    blk = np.empty(N, np.int32)
    blk[order] = (np.arange(N, dtype=np.int64) % K).astype(np.int32)
    return blk


@njit(cache=True)
def _ldg(xa, aa, xb, ab, N, K, cap, label):
    load = np.zeros(K, np.int64)
    cnt = np.zeros(K, np.int32)
    touched = np.empty(K, np.int64)
    heap = [np.int64(p) for p in range(K)]            # key = load * K + p; a block has at most one valid key (its latest)
    heapq.heapify(heap)
    Kk = np.int64(K)
    fallback = 0
    for v in range(N):
        nt = 0
        for e in range(xa[v], xa[v + 1]):
            w = aa[e]
            p = label[w]
            if p >= 0:
                if cnt[p] == 0:
                    touched[nt] = p
                    nt += 1
                cnt[p] += 1
        for e in range(xb[v], xb[v + 1]):
            w = ab[e]
            p = label[w]
            if p >= 0:
                if cnt[p] == 0:
                    touched[nt] = p
                    nt += 1
                cnt[p] += 1
        best = -1
        bs = np.int64(-1)
        for t in range(nt):
            p = touched[t]
            if load[p] < cap:
                s = np.int64(cnt[p]) * (cap - load[p])          # == cnt * (1 - load / cap) * cap, exact in int64
                if best < 0 or s > bs or (s == bs and (load[p] < load[best] or (load[p] == load[best] and p < best))):
                    bs = s
                    best = p
            cnt[p] = 0
        if best < 0:
            fallback += 1
            while True:
                key = heapq.heappop(heap)
                p = key % Kk
                if key // Kk == load[p]:
                    best = p
                    break
        label[v] = best
        load[best] += 1
        if load[best] < cap:
            heapq.heappush(heap, load[best] * Kk + best)
    return load, fallback


def ldg(xa, aa, xb, ab, N, K):
    """R2 LDG; returns (int32 block table, info)."""
    cap = cap_of(N, K)
    assert cap * K >= N
    label = np.full(N, -1, np.int32)
    t0 = time.time()
    load, fb = _ldg(xa, aa, xb, ab, np.int64(N), np.int64(K), np.int64(cap), label)
    assert (label >= 0).all() and int(load.sum()) == N and int(load.max()) <= cap
    return label, {"cap": cap, "seconds": round(time.time() - t0, 3), "least_loaded_fallbacks": int(fb)}


def ldg_reference(xa, aa, xb, ab, N, K):
    """the same rule in plain Python (tests only)."""
    cap = cap_of(N, K)
    label = np.full(N, -1, np.int64)
    load = np.zeros(K, np.int64)
    for v in range(N):
        cnt = {}
        for w in list(aa[xa[v]:xa[v + 1]]) + list(ab[xb[v]:xb[v + 1]]):
            p = int(label[w])
            if p >= 0:
                cnt[p] = cnt.get(p, 0) + 1
        best, bs = -1, -1
        for p, c in cnt.items():
            if load[p] < cap:
                s = c * (cap - int(load[p]))
                if best < 0 or s > bs or (s == bs and (load[p] < load[best] or (load[p] == load[best] and p < best))):
                    best, bs = p, s
        if best < 0:
            best = min((int(load[p]), p) for p in range(K) if load[p] < cap)[1]
        label[v] = best
        load[best] += 1
    return label.astype(np.int32)


@njit(cache=True)
def _stats(xa, aa, xb, ab, N, K, label):
    """(cut entries, sum over v of (distinct blocks in {v} u N(v)) - 1, max distinct, entries)."""
    mark = np.full(K, -1, np.int64)
    cut = 0
    km1 = 0
    mx = 0
    ent = 0
    for v in range(N):
        pv = label[v]
        mark[pv] = v
        d = 1
        for e in range(xa[v], xa[v + 1]):
            p = label[aa[e]]
            ent += 1
            if p != pv:
                cut += 1
            if mark[p] != v:
                mark[p] = v
                d += 1
        for e in range(xb[v], xb[v + 1]):
            p = label[ab[e]]
            ent += 1
            if p != pv:
                cut += 1
            if mark[p] != v:
                mark[p] = v
                d += 1
        km1 += d - 1
        if d > mx:
            mx = d
    return cut, km1, mx, ent


def map_stats(label, K, xa, aa, xb, ab):
    N = len(label)
    sz = np.bincount(label, minlength=K).astype(np.int64)
    cut, km1, mx, ent = _stats(xa, aa, xb, ab, np.int64(N), np.int64(K), label)
    return {"N": int(N), "K": int(K), "blocks_used": int((sz > 0).sum()), "size_min": int(sz.min()), "size_max": int(sz.max()), "size_mean": round(float(sz.mean()), 3),
            "load_skew_max_over_mean": round(float(sz.max() / sz.mean()), 4), "cap_ceil_1.03_N_over_K": cap_of(N, K), "within_cap": bool(sz.max() <= cap_of(N, K)),
            "adjacency_entries": int(ent), "cut_entries": int(cut), "cut_fraction": round(cut / float(max(ent, 1)), 6),
            "closed_nbhd_connectivity_minus_1_sum (KM1 of the anchor hyperedges)": int(km1), "closed_nbhd_distinct_blocks_mean": round(1.0 + km1 / float(N), 4),
            "closed_nbhd_distinct_blocks_max": int(mx)}


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def csr_from_pairs(u, v, N):
    """undirected CSR (int64 xadj, int32 adj) of the pair list (u, v), both directions, self loops dropped."""
    m = u != v
    u, v = u[m], v[m]
    src = np.concatenate([u, v]).astype(np.int64)
    dst = np.concatenate([v, u]).astype(np.int32)
    o = np.argsort(src, kind="stable")
    src, dst = src[o], dst[o]
    xadj = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=xadj[1:])
    return xadj, dst


# ---------------------------------------------------------------------------------------------------------- the C twin (_fbx_ldg.c)
HERE = os.path.dirname(os.path.abspath(__file__))
DISTRO = os.environ.get("FBX_WSL_DISTRO", "Ubuntu-24.04")


def wsl_path(p):
    p = os.path.abspath(p).replace("\\", "/")
    if os.name == "nt" and len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def _run(cmd):
    """cmd (a list) inside the WSL distro from Windows, or directly on Linux."""
    full = (["wsl.exe", "-d", DISTRO, "--exec"] + cmd) if os.name == "nt" else cmd
    r = subprocess.run(full, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("%s -> exit %d: %s %s" % (" ".join(cmd), r.returncode, r.stdout[-2000:], r.stderr[-4000:]))
    return r.stdout


def c_binary():
    """compile _fbx_ldg.c (once per source sha) into the distro's /tmp; returns the binary's path inside the distro."""
    src = os.path.join(HERE, "_fbx_ldg.c")
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    binp = "/tmp/fbx_ldg_%s" % sha
    try:
        _run(["test", "-x", binp])
    except RuntimeError:
        _run(["gcc", "-O2", "-ffp-contract=off", "-Wall", "-o", binp, wsl_path(src)])
    return binp, sha


def npy_spec(path):
    """'path@offset' for a 1-D .npy: the C program maps the file and skips the header."""
    with open(path, "rb") as f:
        ver = np.lib.format.read_magic(f)
        (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        off = f.tell()
    return "%s@%d" % (wsl_path(path), off)


def write_csr_raw(xadj, adj, stem):
    """raw little-endian int64 xadj / int32 adj files for the C program; returns their argument specs."""
    xf, af = stem + ".xadj.i64", stem + ".adj.i32"
    np.ascontiguousarray(xadj, np.int64).tofile(xf)
    np.ascontiguousarray(adj, np.int32).tofile(af)
    return [wsl_path(xf), wsl_path(af)]


def ldg_c(N, K, streams, out_raw):
    """R2 LDG in C; streams = the argument specs of one or two (xadj, adj) pairs (a flat list); returns (int32 block table, info)."""
    binp, sha = c_binary()
    out = _run([binp, "ldg", str(int(N)), str(int(K)), wsl_path(out_raw)] + list(streams))
    info = json.loads(out.strip().splitlines()[-1])
    lab = np.fromfile(out_raw, np.int32)
    assert len(lab) == N and (lab >= 0).all() and (lab < K).all()
    info["c_source_sha12"] = sha
    return lab, info


def stats_c(N, K, labels_npy, streams):
    """map_stats' counts from the C program (labels_npy: an .npy of int32)."""
    binp, sha = c_binary()
    out = _run([binp, "stats", str(int(N)), str(int(K)), npy_spec(labels_npy)] + list(streams))
    r = json.loads(out.strip().splitlines()[-1])
    cap = cap_of(N, K)
    ent, cut, km1 = r["adjacency_entries"], r["cut_entries"], r["km1_sum"]
    sz_min, sz_max = r["size_min"], r["size_max"]
    return {"N": int(N), "K": int(K), "blocks_used": r["blocks_used"], "size_min": sz_min, "size_max": sz_max, "size_mean": round(N / float(K), 3),
            "load_skew_max_over_mean": round(sz_max / (N / float(K)), 4), "cap_ceil_1.03_N_over_K": cap, "within_cap": bool(sz_max <= cap),
            "adjacency_entries": ent, "cut_entries": cut, "cut_fraction": round(cut / float(max(ent, 1)), 6),
            "closed_nbhd_connectivity_minus_1_sum (KM1 of the anchor hyperedges)": km1, "closed_nbhd_distinct_blocks_mean": round(1.0 + km1 / float(N), 4),
            "closed_nbhd_distinct_blocks_max": r["closed_nbhd_distinct_blocks_max"], "stats_seconds": r["seconds"], "stats_peak_rss_kb": r["peak_rss_kb"]}


def _testc():
    rng = np.random.RandomState(11)
    tmp = tempfile.mkdtemp(prefix="fbxldg_")
    for (N, K, m) in ((400, 7, 1500), (1000, 40, 4000), (257, 3, 600), (2000, 200, 3000), (5000, 64, 30000)):
        u = rng.randint(0, N, m)
        v = rng.randint(0, N, m)
        xa, aa = csr_from_pairs(u, v, N)
        z = np.zeros(N + 1, np.int64)
        e = np.zeros(0, np.int32)
        ref = ldg_reference(xa, aa, z, e, N, K)
        # one stream
        sA = write_csr_raw(xa, aa, os.path.join(tmp, "a%d" % N))
        Lc, info = ldg_c(N, K, sA, os.path.join(tmp, "l%d.raw" % N))
        assert (Lc == ref).all(), "C LDG != reference (N %d K %d)" % (N, K)
        if HAVE_NUMBA:
            Ln, _ = ldg(xa, aa, z, e, N, K)
            assert (Ln == ref).all(), "numba LDG != reference"
        # two streams: the same entries split between a stream A and a stream B (the neighbour COUNTS are what the rule reads)
        half = (xa[1:] - xa[:-1]) // 2
        xb = np.concatenate([[0], np.cumsum(half)]).astype(np.int64)
        xa2 = np.concatenate([[0], np.cumsum((xa[1:] - xa[:-1]) - half)]).astype(np.int64)
        aA = np.concatenate([aa[xa[x] + half[x]:xa[x + 1]] for x in range(N)]).astype(np.int32)
        aB = np.concatenate([aa[xa[x]:xa[x] + half[x]] for x in range(N)]).astype(np.int32)
        s2 = write_csr_raw(xa2, aA, os.path.join(tmp, "s%d" % N)) + write_csr_raw(xb, aB, os.path.join(tmp, "t%d" % N))
        L2, _ = ldg_c(N, K, s2, os.path.join(tmp, "l2_%d.raw" % N))
        assert (L2 == ref).all(), "two-stream C LDG != reference"
        # statistics
        lab_npy = os.path.join(tmp, "lab%d.npy" % N)
        np.save(lab_npy, Lc)
        st = stats_c(N, K, lab_npy, sA)
        st_py = map_stats(Lc, K, xa, aa, z, e) if HAVE_NUMBA else None
        km1 = sum(len({int(Lc[x])} | {int(Lc[w]) for w in aa[xa[x]:xa[x + 1]]}) - 1 for x in range(N))
        cut = sum(int(Lc[x] != Lc[w]) for x in range(N) for w in aa[xa[x]:xa[x + 1]])
        assert st["cut_entries"] == cut and st["closed_nbhd_connectivity_minus_1_sum (KM1 of the anchor hyperedges)"] == km1, "C stats != set-based"
        assert st["adjacency_entries"] == len(aa) and st["within_cap"] and st["size_max"] == int(np.bincount(Lc, minlength=K).max())
        if st_py is not None:
            for k_ in ("cut_entries", "adjacency_entries", "closed_nbhd_connectivity_minus_1_sum (KM1 of the anchor hyperedges)", "size_min", "size_max"):
                assert st[k_] == st_py[k_], k_
        print("ok C N %d K %d: LDG == reference (1 and 2 streams%s), stats == set-based, cut %.3f, fallbacks %d" % (N, K, " == numba" if HAVE_NUMBA else "", st["cut_fraction"], info["least_loaded_fallbacks"]))
    print("TESTC PASS (C source sha12 %s)" % c_binary()[1])


def _test():
    rng = np.random.RandomState(7)
    for (N, K, m) in ((400, 7, 1500), (1000, 40, 4000), (257, 3, 600), (2000, 200, 3000)):
        u = rng.randint(0, N, m)
        v = rng.randint(0, N, m)
        xa, aa = csr_from_pairs(u, v, N)
        # split the same adjacency into two streams (as Freebase's OUT / IN CSRs) and check both readings agree
        z = np.zeros(N + 1, np.int64)
        e = np.zeros(0, np.int32)
        L1, _ = ldg(xa, aa, z, e, N, K)
        Lr = ldg_reference(xa, aa, z, e, N, K)
        assert (L1 == Lr).all(), "numba LDG != reference (N %d K %d)" % (N, K)
        st = map_stats(L1, K, xa, aa, z, e)
        assert st["within_cap"] and st["adjacency_entries"] == len(aa)
        h = hash_balanced(N, K, 0)
        sz = np.bincount(h, minlength=K)
        assert sz.max() - sz.min() <= 1 and (hash_balanced(N, K, 0) == h).all() and not (hash_balanced(N, K, 1) == h).all()
        # closed-neighbourhood distinct-block count against a set-based check
        km1 = sum(len({int(L1[x])} | {int(L1[w]) for w in aa[xa[x]:xa[x + 1]]}) - 1 for x in range(N))
        assert km1 == st["closed_nbhd_connectivity_minus_1_sum (KM1 of the anchor hyperedges)"]
        cut = sum(int(L1[x] != L1[w]) for x in range(N) for w in aa[xa[x]:xa[x + 1]])
        assert cut == st["cut_entries"]
        # two-stream reading == one-stream reading: halve every row between the streams
        half = (xa[1:] - xa[:-1]) // 2
        xb = np.concatenate([[0], np.cumsum(half)]).astype(np.int64)
        xa2 = np.concatenate([[0], np.cumsum((xa[1:] - xa[:-1]) - half)]).astype(np.int64)
        aA = np.concatenate([aa[xa[x] + half[x]:xa[x + 1]] for x in range(N)]) if N else e
        aB = np.concatenate([aa[xa[x]:xa[x] + half[x]] for x in range(N)]) if N else e
        L2, _ = ldg(xa2, aA.astype(np.int32), xb, aB.astype(np.int32), N, K)
        # order of neighbours differs between the readings but the counts do not: the result must be identical
        assert (L2 == L1).all(), "two-stream LDG differs from one-stream LDG"
        print("ok N %d K %d: LDG == reference, cut %.3f, closed-nbhd blocks mean %.2f, hash sizes %d..%d" % (N, K, st["cut_fraction"], st["closed_nbhd_distinct_blocks_mean"], sz.min(), sz.max()))
    print("TEST PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "TEST":
        _test()
    elif len(sys.argv) > 1 and sys.argv[1] == "TESTC":
        _testc()
    else:
        raise SystemExit(__doc__)
