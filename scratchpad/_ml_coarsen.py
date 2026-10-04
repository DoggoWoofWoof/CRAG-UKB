"""FBX_SCALE stage 4A -- the hypergraph-aware coarsening of the rung-4 multilevel prototype (addendum 4).

Input: the canonical structural graph S (undirected pairs) and the FROZEN H4 hypergraph it defines (one net per anchor u = its closed neighbourhood N[u], weight
max(1, rint(1000 / (|e| - 1))); the frozen builder chunks nets larger than round(N / K), which only concerns a handful of hub anchors).

A coarsening is a vertex clustering; the coarse hypergraph is the QUOTIENT of the fine one (pins mapped to their clusters, duplicate pins inside a net removed, nets left with
one pin dropped, identical nets merged with their weights summed, vertex weight = cluster size).  The quotient is exact: for every partition of the clusters, the weighted
connectivity (KM1) of the quotient equals the weighted KM1 of the fine hypergraph under the projected partition (TESTML checks this).

Clustering operators (both hypergraph-aware: they read net membership, net weight and net size):
  T  open-twin contraction: vertices with identical open neighbourhoods (they lie in exactly the same nets except their own anchor net) are contracted, in chunks of
     at most Wmax.  Exactly checked (neighbour lists compared, not only hashed).
  A  agglomerative heavy-connectivity clustering (_ml_agg.c, the rating of a multilevel hypergraph partitioner: sum over shared nets of w_e / (|e| - 1)) with a cluster
     weight cap Wmax and nets larger than Lmax ignored for the rating.  Repeated on the quotient until the pin count is <= fine pins / R, or a pass removes < 5 % of the
     vertices, or MAXLEVELS passes were made.

  python scratchpad/_ml_coarsen.py TESTML        # C == plain-Python reference bit for bit; quotient KM1 identity; twin exactness
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _fbx_part as P  # noqa: E402

WSCALE = 1000
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def log(*a):
    print("[%7.1fs]" % (time.time() - _T0), *a, flush=True)


_T0 = time.time()


def mix(x, c):
    with np.errstate(over="ignore"):
        x = x.astype(np.uint64) + np.uint64(c)
        x ^= x >> np.uint64(30)
        x *= _M1
        x ^= x >> np.uint64(27)
        x *= _M2
        x ^= x >> np.uint64(31)
    return x


def seg_sum(vals, ptr):
    """sum of vals over the segments [ptr[i], ptr[i+1]) (uint64, wrapping); empty segments give 0."""
    n = len(ptr) - 1
    out = np.zeros(n, np.uint64)
    nz = np.flatnonzero(ptr[1:] > ptr[:-1])
    if len(nz):
        with np.errstate(over="ignore"):
            out[nz] = np.add.reduceat(vals, ptr[:-1][nz])
    return out


def sorted_csr(u, v, N):
    """symmetric CSR of the unique undirected pairs (u < v), rows sorted ascending."""
    src = np.concatenate([u, v]).astype(np.int64)
    dst = np.concatenate([v, u]).astype(np.int64)
    o = np.lexsort((dst, src))
    src, dst = src[o], dst[o]
    xa = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=xa[1:])
    return xa, dst.astype(np.int32)


def closed_nbhd_nets(xa, aa, N):
    """the unsplit H4 nets of the graph: net u = {u} U N(u), weight max(1, rint(1000 / deg)); nets with fewer than two pins are dropped.  Returns eptr, eidx, ew, anchor."""
    deg = np.diff(xa)
    anchors = np.flatnonzero(deg >= 1)
    sz = deg[anchors] + 1
    eptr = np.zeros(len(anchors) + 1, np.int64)
    np.cumsum(sz, out=eptr[1:])
    eidx = np.empty(eptr[-1], np.int32)
    eidx[eptr[:-1]] = anchors
    pos = np.arange(len(aa), dtype=np.int64)
    row = np.repeat(np.arange(N, dtype=np.int64), deg)
    rank_in_row = pos - xa[row]
    net_of_row = np.full(N, -1, np.int64)
    net_of_row[anchors] = np.arange(len(anchors))
    eidx[eptr[net_of_row[row]] + 1 + rank_in_row] = aa
    ew = np.maximum(1, np.rint(WSCALE / (sz - 1).astype(np.float64)).astype(np.int64))
    return eptr, eidx, ew, anchors


def transpose(V, eptr, eidx):
    """vertex -> nets (ascending net id): (vptr int64, vidx int32)."""
    M = len(eptr) - 1
    net = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    o = np.lexsort((net, eidx))
    vptr = np.zeros(V + 1, np.int64)
    np.cumsum(np.bincount(eidx, minlength=V), out=vptr[1:])
    return vptr, net[o].astype(np.int32)


# ----------------------------------------------------------------------------------------------------------------- the quotient
def quotient(V, eptr, eidx, ew, vw, cl, verify=True):
    """quotient of the hypergraph (V, eptr, eidx, ew, vw) under the clustering cl (dense 0..V'-1).  Returns (V', eptr', eidx', ew', vw', stats)."""
    cl = np.asarray(cl, np.int64)
    Vn = int(cl.max()) + 1
    vwn = np.bincount(cl, weights=vw, minlength=Vn).astype(np.int64)
    M = len(eptr) - 1
    net = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    pin = cl[eidx]
    o = np.lexsort((pin, net))
    net, pin = net[o], pin[o]
    keep = np.ones(len(net), bool)
    keep[1:] = (net[1:] != net[:-1]) | (pin[1:] != pin[:-1])
    net, pin = net[keep], pin[keep]
    sz = np.bincount(net, minlength=M)
    ok = sz >= 2
    m2 = ok[net]
    net, pin = net[m2], pin[m2]
    kept = np.flatnonzero(ok)
    renum = np.full(M, -1, np.int64)
    renum[kept] = np.arange(len(kept))
    net = renum[net]
    ew1 = np.asarray(ew, np.int64)[kept]
    sz1 = sz[kept]
    ptr = np.zeros(len(kept) + 1, np.int64)
    np.cumsum(sz1, out=ptr[1:])
    # identical nets: (size, two independent hashes) group; the grouping is then verified exactly
    h1 = seg_sum(mix(pin, 0x51ED270B), ptr)
    h2 = seg_sum(mix(pin, 0x2545F4914F6CDD1D), ptr)
    g = np.lexsort((np.arange(len(kept)), h2, h1, sz1))
    first = np.ones(len(g), bool)
    first[1:] = (sz1[g][1:] != sz1[g][:-1]) | (h1[g][1:] != h1[g][:-1]) | (h2[g][1:] != h2[g][:-1])
    gid = np.cumsum(first) - 1
    rep = g[np.flatnonzero(first)][gid]                  # representative (first member) of every net in sorted order
    member = g[~first]
    if verify and len(member):
        r = rep[~first]
        s = sz1[member]
        off = np.arange(int(s.sum()), dtype=np.int64) - np.repeat(np.cumsum(s) - s, s)
        a = pin[np.repeat(ptr[member], s) + off]
        b = pin[np.repeat(ptr[r], s) + off]
        assert np.array_equal(a, b), "identical-net grouping collided (hash) -- not merged"
    wsum = np.bincount(gid, weights=ew1[g], minlength=int(gid[-1]) + 1 if len(gid) else 0).astype(np.int64)
    reps = g[first]
    ord_reps = np.sort(reps)                             # keep the surviving nets in their original order
    new_of_rep = np.full(len(kept), -1, np.int64)
    new_of_rep[ord_reps] = np.arange(len(ord_reps))
    gid_of_rep = np.empty(len(kept), np.int64)
    gid_of_rep[g[first]] = gid[first]
    ew_out = wsum[gid_of_rep[ord_reps]]
    sz_out = sz1[ord_reps]
    ptr_out = np.zeros(len(ord_reps) + 1, np.int64)
    np.cumsum(sz_out, out=ptr_out[1:])
    sel = np.repeat(ptr[ord_reps], sz_out) + (np.arange(int(sz_out.sum()), dtype=np.int64) - np.repeat(ptr_out[:-1], sz_out))
    eidx_out = pin[sel].astype(np.int32)
    st = {"V": Vn, "M_in": M, "P_in": int(len(eidx)), "nets_dropped_singleton": int((~ok).sum()), "nets_merged_identical": int(len(kept) - len(ord_reps)),
          "M": int(len(ord_reps)), "P": int(len(eidx_out)), "weight_conserved_vertex": int(vwn.sum()) == int(np.asarray(vw).sum())}
    return Vn, ptr_out, eidx_out, ew_out, vwn, st


def km1(eptr, eidx, ew, part):
    """weighted KM1 = sum_e w_e (lambda_e - 1) of the partition part (one block id per vertex)."""
    M = len(eptr) - 1
    net = np.repeat(np.arange(M, dtype=np.int64), np.diff(eptr))
    b = np.asarray(part, np.int64)[eidx]
    k = int(b.max()) + 1
    key = np.unique(net * k + b)
    lam = np.bincount(key // k, minlength=M)
    return int((np.asarray(ew, np.int64) * np.maximum(lam - 1, 0)).sum())


# ----------------------------------------------------------------------------------------------------------------- T: open twins
def open_twins(xa, aa, N, Wmax):
    """cluster ids (dense) contracting vertices with identical open neighbourhoods, in chunks of <= Wmax; returns (cl, info)."""
    deg = np.diff(xa)
    h1 = seg_sum(mix(aa, 0x9E3779B97F4A7C15), xa)
    h2 = seg_sum(mix(aa, 0x123456789ABCDEF1), xa)
    idx = np.flatnonzero(deg >= 1)
    o = idx[np.lexsort((idx, h2[idx], h1[idx], deg[idx]))]
    first = np.ones(len(o), bool)
    first[1:] = (deg[o][1:] != deg[o][:-1]) | (h1[o][1:] != h1[o][:-1]) | (h2[o][1:] != h2[o][:-1])
    cls = np.cumsum(first) - 1
    start = np.flatnonzero(first)
    rank = np.arange(len(o)) - start[cls]
    rep = o[start][cls]
    m = ~first
    if m.any():                                           # exact check: the neighbour lists of a class are identical
        v, r = o[m], rep[m]
        s = deg[v]
        off = np.arange(int(s.sum()), dtype=np.int64) - np.repeat(np.cumsum(s) - s, s)
        assert np.array_equal(aa[np.repeat(xa[v], s) + off], aa[np.repeat(xa[r], s) + off]), "open-twin hash collision"
    chunk = rank // Wmax
    key = np.full(N, -1, np.int64)
    key[o] = cls.astype(np.int64) * (int(chunk.max()) + 1 if len(chunk) else 1) + chunk
    iso = np.flatnonzero(deg == 0)
    key[iso] = int(key.max()) + 1 + np.arange(len(iso))
    _, cl = np.unique(key, return_inverse=True)
    sizes = np.bincount(cl)
    info = {"twin_classes": int(cls[-1] + 1) if len(cls) else 0, "vertices_in_nontrivial_classes": int(np.bincount(cls)[np.bincount(cls) > 1].sum()) if len(cls) else 0,
            "clusters": int(cl.max() + 1), "max_cluster": int(sizes.max())}
    return cl.astype(np.int64), info


# ----------------------------------------------------------------------------------------------------------------- A: agglomeration pass
def agg_reference(V, eptr, eidx, ew, vw, vptr, vidx, Lmax, Wmax):
    """plain-Python statement of _ml_agg.c."""
    lab = list(range(V))
    state = [0] * V
    cw = [int(x) for x in vw]
    order = sorted(range(V), key=lambda v: (int(vw[v]), v))
    for u in order:
        if state[u] != 0:
            continue
        score, touched = {}, []
        for e in vidx[vptr[u]:vptr[u + 1]]:
            s = int(eptr[e + 1] - eptr[e])
            if s < 2 or s > Lmax:
                continue
            om = float(ew[e]) / float(s - 1)
            for v in eidx[eptr[e]:eptr[e + 1]]:
                if v == u:
                    continue
                c = lab[int(v)]
                if c not in score:
                    score[c] = 0.0
                    touched.append(c)
                score[c] += om
        best, bs = -1, 0.0
        for c in touched:
            if cw[c] + int(vw[u]) <= Wmax and (score[c] > bs or (score[c] == bs and best >= 0 and c < best)):
                best, bs = c, score[c]
        if best >= 0:
            lab[u] = best
            cw[best] += int(vw[u])
            state[u] = 2
            state[best] = 1
    return np.array(lab, np.int32)


def c_binary_agg():
    src = os.path.join(HERE, "_ml_agg.c")
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    binp = "/tmp/ml_agg_%s" % sha
    try:
        P._run(["test", "-x", binp])
    except RuntimeError:
        P._run(["gcc", "-O2", "-ffp-contract=off", "-Wall", "-o", binp, P.wsl_path(src)])
    return binp, sha


def agg_c(V, eptr, eidx, ew, vw, vptr, vidx, Lmax, Wmax, tmp):
    binp, sha = c_binary_agg()
    fs = {}
    for name, arr, dt in (("vw", vw, np.int32), ("eptr", eptr, np.int64), ("eidx", eidx, np.int32), ("ew", ew, np.int64), ("vptr", vptr, np.int64), ("vidx", vidx, np.int32)):
        fs[name] = os.path.join(tmp, "agg_%s.raw" % name)
        np.ascontiguousarray(arr, dt).tofile(fs[name])
    out = os.path.join(tmp, "agg_out.raw")
    txt = P._run([binp, str(V), str(len(eptr) - 1), str(Lmax), str(Wmax)] + [P.wsl_path(fs[k]) for k in ("vw", "eptr", "eidx", "ew", "vptr", "vidx")] + [P.wsl_path(out)])
    info = json.loads(txt.strip().splitlines()[-1])
    info["c_source_sha12"] = sha
    lab = np.fromfile(out, np.int32)
    assert len(lab) == V
    return lab, info


def dense(lab):
    _, cl = np.unique(lab, return_inverse=True)
    return cl.astype(np.int64)


# ----------------------------------------------------------------------------------------------------------------- the level loop
def coarsen(N, xa, aa, Lmax, Wmax, R, max_levels, tmp, min_gain=0.05, use_twins=True):
    """cluster map of the fine vertices (int64, dense), the per-level statistics and the last coarse hypergraph."""
    eptr, eidx, ew, anchors = closed_nbhd_nets(xa, aa, N)
    vw = np.ones(N, np.int64)
    P0 = int(len(eidx))
    total = np.arange(N, dtype=np.int64)
    levels = [{"level": 0, "kind": "fine", "V": N, "M": int(len(ew)), "P": P0}]
    V = N
    if use_twins:
        t = time.time()
        cl, info = open_twins(xa, aa, N, Wmax)
        V2, eptr, eidx, ew, vw, st = quotient(V, eptr, eidx, ew, vw, cl)
        total = cl[total]
        levels.append(dict(level=1, kind="T_open_twins", seconds=round(time.time() - t, 1), **info, **st))
        log("T: V %d -> %d, M %d -> %d, P %d -> %d (%.2fx)" % (V, V2, st["M_in"], st["M"], st["P_in"], st["P"], st["P_in"] / max(st["P"], 1)))
        V = V2
    while len(levels) <= max_levels and len(eidx) > P0 / float(R):
        t = time.time()
        vptr, vidx = transpose(V, eptr, eidx)
        lab, info = agg_c(V, eptr, eidx, ew, vw, vptr, vidx, Lmax, Wmax, tmp)
        cl = dense(lab)
        Vn = int(cl.max()) + 1
        if V - Vn < min_gain * V:
            levels.append({"level": len(levels), "kind": "A_agglomerate", "stopped": "gain %.3f < %.2f" % ((V - Vn) / V, min_gain), **info})
            log("A: stop, the pass removed %.1f %% of the vertices" % (100.0 * (V - Vn) / V))
            break
        V2, eptr, eidx, ew, vw, st = quotient(V, eptr, eidx, ew, vw, cl)
        total = cl[total]
        levels.append(dict(level=len(levels), kind="A_agglomerate", seconds=round(time.time() - t, 1), agg=info, **st))
        log("A%d: V %d -> %d, M %d -> %d, P %d -> %d, max weight %d (%.2fx of fine pins)" % (len(levels) - 1, V, V2, st["M_in"], st["M"], st["P_in"], st["P"], int(vw.max()),
                                                                                               P0 / max(st["P"], 1)))
        V = V2
    return total, levels, (V, eptr, eidx, ew, vw)


# ----------------------------------------------------------------------------------------------------------------- tests
def _rand_hg(rng, V, M, maxsz):
    nets = []
    for _ in range(M):
        s = int(rng.randint(2, maxsz + 1))
        nets.append(np.sort(rng.choice(V, size=min(s, V), replace=False)))
    eptr = np.zeros(M + 1, np.int64)
    np.cumsum([len(x) for x in nets], out=eptr[1:])
    return eptr, np.concatenate(nets).astype(np.int32), rng.randint(1, 50, M).astype(np.int64)


def _testml():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="mltest_")
    rng = np.random.RandomState(5)
    # 1. C == reference on random hypergraphs (uneven vertex weights)
    for ci, (V, M, ms, Lm, Wm) in enumerate(((60, 90, 6, 4, 9), (200, 400, 9, 6, 20), (500, 600, 30, 12, 6), (300, 700, 5, 5, 40))):
        eptr, eidx, ew = _rand_hg(rng, V, M, ms)
        vw = rng.randint(1, 4, V).astype(np.int64)
        vptr, vidx = transpose(V, eptr, eidx)
        ref = agg_reference(V, eptr, eidx, ew, vw, vptr, vidx, Lm, Wm)
        lab, info = agg_c(V, eptr, eidx, ew, vw, vptr, vidx, Lm, Wm, tmp)
        assert np.array_equal(lab, ref), "C pass != reference (case %d)" % ci
        cw = np.bincount(lab, weights=vw, minlength=V)
        assert cw.max() <= max(Wm, vw.max()), "cluster weight cap broken"
        assert info["clusters"] == len(np.unique(lab))
        assert (lab[lab] == lab).all(), "a label is not a leader"
        log("agg case %d: V %d -> %d clusters, C == reference" % (ci, V, info["clusters"]))
    # 2. the quotient identity: weighted KM1 of the quotient == weighted KM1 of the fine hypergraph under the projected partition
    for ci in range(6):
        V, M = 400 + 50 * ci, 700
        eptr, eidx, ew = _rand_hg(rng, V, M, 12)
        dup = rng.randint(0, M, 60)                        # make some nets collapse to identical ones
        vw = np.ones(V, np.int64)
        cl = dense(rng.randint(0, V // 3, V))
        Vn, p2, i2, w2, vw2, st = quotient(V, eptr, eidx, ew, vw, cl)
        for _ in range(4):
            k = int(rng.randint(2, 20))
            pc = rng.randint(0, k, Vn)
            assert km1(p2, i2, w2, pc) == km1(eptr, eidx, ew, pc[cl]), "quotient KM1 identity broken"
        assert vw2.sum() == V and (np.diff(p2) >= 2).all()
        assert len({tuple(i2[p2[e]:p2[e + 1]]) for e in range(len(p2) - 1)}) == len(p2) - 1, "identical nets survived"
    # a quotient with forced identical nets
    eptr = np.array([0, 3, 6, 9, 11], np.int64)
    eidx = np.array([0, 1, 2, 0, 1, 3, 4, 5, 2, 6, 7], np.int32)
    ew = np.array([5, 7, 3, 4], np.int64)
    Vn, p2, i2, w2, vw2, st = quotient(8, eptr, eidx, ew, np.ones(8, np.int64), dense(np.array([0, 0, 1, 1, 2, 2, 3, 3])))
    assert st["nets_merged_identical"] == 1 and st["nets_dropped_singleton"] == 1 and sorted(w2.tolist()) == [3, 12] and st["M"] == 2, (st, w2)
    log("quotient identity holds; forced merge case: %s" % json.dumps(st))
    # 3. open twins: classes are exactly the vertices with identical neighbour sets, chunked by Wmax
    for ci in range(4):
        N = 300
        u = rng.randint(0, 40, 600)
        v = rng.randint(40, N, 600)                        # bipartite -> lots of twins among the right side
        k = np.unique(np.minimum(u, v) * N + np.maximum(u, v))
        xa, aa = sorted_csr(k // N, k % N, N)
        Wm = [3, 5, 1000, 2][ci]
        cl, info = open_twins(xa, aa, N, Wm)
        nb = [tuple(aa[xa[x]:xa[x + 1]]) for x in range(N)]
        classes = {}
        for x in range(N):
            classes.setdefault(nb[x], []).append(x)
        want = 0
        for key, mem in classes.items():
            if len(key) == 0:
                want += len(mem)
            else:
                want += -(-len(mem) // Wm)
        assert info["clusters"] == want, (info, want)
        for c in range(info["clusters"]):
            mem = np.flatnonzero(cl == c)
            assert len({nb[int(x)] for x in mem}) == 1 and len(mem) <= Wm, "twin cluster not homogeneous"
    log("open-twin classes exact")
    # 4. end to end on a small random graph: the coarse partition projected keeps its KM1 identity against the frozen-rule closed-neighbourhood nets
    N = 700
    u = rng.randint(0, N, 2500)
    v = rng.randint(0, N, 2500)
    m = u != v
    k = np.unique(np.minimum(u[m], v[m]) * N + np.maximum(u[m], v[m]))
    xa, aa = sorted_csr(k // N, k % N, N)
    total, levels, (Vc, ep, ei, ewc, vwc) = coarsen(N, xa, aa, 50, 8, 4.0, 6, tmp)
    eptr0, eidx0, ew0, _ = closed_nbhd_nets(xa, aa, N)
    assert vwc.sum() == N and total.max() + 1 == Vc
    pc = rng.randint(0, 9, Vc)
    assert km1(ep, ei, ewc, pc) == km1(eptr0, eidx0, ew0, pc[total]), "end-to-end quotient identity broken"
    log("end to end: %d levels, V %d -> %d" % (len(levels), N, Vc))
    log("TESTML PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "TESTML":
        _testml()
    else:
        raise SystemExit(__doc__)
