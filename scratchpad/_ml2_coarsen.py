"""FBX_SCALE stage 4C -- the SK-arm multilevel coarsening lab: hypergraph-aware clusterings of the frozen H4_SK hypergraph, two rating rules x two cap schedules.

The fine hypergraph is the frozen H4_SPLIT_PRESERVE one on STRUCT u KNN: one net per anchor per FAMILY (the anchor and its neighbours in that family), weight max(1, rint(1000 / (|e| - 1))).
The clustering is K-independent and is built on the unsplit nets of the two adjacencies (the K-dependent splitting touches 13-165 anchors; the quotient is later applied to each K's own fine
npz, so the exact identity  weighted KM1 (coarse partition) == weighted KM1 (fine hypergraph, projected partition)  holds for every K).

  rating rule (the C pass, scratchpad/_ml2_agg.c)
    HEAVY  r(u, C) = sum over nets e of u, 2 <= |e| <= Lmax, of ew[e] / (|e| - 1) * (pins of e other than u in C)      (the rule of scratchpad/_ml_agg.c)
    SPAN   r(u, C) = sum over nets e of u, 2 <= |e| <= Lmax, touching C, of ew[e]                                      (the weighted net-span reduction of putting u into C)
  cap schedule
    FLAT   every pass uses the final cluster-weight cap Wmax
    GEOM   caps 2, 4, 8, ... , Wmax (one pass each, modest contraction per level), then Wmax while a pass still removes >= MIN_GAIN of the vertices

  the four candidates       M1 = HEAVY/FLAT   M2 = SPAN/FLAT   M3 = HEAVY/GEOM   M4 = SPAN/GEOM       (each after exact open-twin contraction in BOTH families)

  python scratchpad/_ml2_coarsen.py TEST
"""
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

import _ml_coarsen as C
import _fbx_part as P

HERE = C.HERE
log = C.log
METHODS = {"M1": ("heavy", "flat"), "M2": ("span", "flat"), "M3": ("heavy", "geom"), "M4": ("span", "geom")}
MODE = {"heavy": 0, "span": 1}


# ----------------------------------------------------------------------------------------------------------------- the SK nets and the twins
def sk_nets(N, fams):
    """nets of every family, concatenated in family order: (eptr, eidx, ew).  fams = [(xa, aa), ...]"""
    eps, eis, ews = [], [], []
    for xa, aa in fams:
        ep, ei, ew, _ = C.closed_nbhd_nets(xa, aa, N)
        eps.append(ep)
        eis.append(ei)
        ews.append(ew)
    sz = np.concatenate([np.diff(e) for e in eps])
    eptr = np.zeros(len(sz) + 1, np.int64)
    np.cumsum(sz, out=eptr[1:])
    return eptr, np.concatenate(eis), np.concatenate(ews)


def open_twins_multi(fams, N, Wmax):
    """dense cluster ids contracting vertices whose open neighbourhoods are identical in EVERY family, in chunks of <= Wmax; returns (cl, info)."""
    degs = [np.diff(xa) for xa, _ in fams]
    hs = []
    for f, (xa, aa) in enumerate(fams):
        hs.append(C.seg_sum(C.mix(aa, 0x9E3779B97F4A7C15 + f), xa))
        hs.append(C.seg_sum(C.mix(aa, 0x123456789ABCDEF1 + 3 * f), xa))
    anyd = np.zeros(N, bool)
    for d in degs:
        anyd |= d >= 1
    idx = np.flatnonzero(anyd)
    cols = degs + hs
    o = idx[np.lexsort(tuple([idx] + [c[idx] for c in cols[::-1]]))]
    first = np.ones(len(o), bool)
    diff = np.zeros(len(o) - 1, bool)
    for c in cols:
        diff |= c[o][1:] != c[o][:-1]
    first[1:] = diff
    cls = np.cumsum(first) - 1
    start = np.flatnonzero(first)
    rank = np.arange(len(o)) - start[cls]
    rep = o[start][cls]
    m = ~first
    if m.any():                                           # exact check, family by family: the neighbour lists of a class are identical
        v, r = o[m], rep[m]
        for (xa, aa), d in zip(fams, degs):
            s = d[v]
            off = np.arange(int(s.sum()), dtype=np.int64) - np.repeat(np.cumsum(s) - s, s)
            assert np.array_equal(aa[np.repeat(xa[v], s) + off], aa[np.repeat(xa[r], s) + off]), "open-twin hash collision"
    chunk = rank // Wmax
    key = np.full(N, -1, np.int64)
    key[o] = cls.astype(np.int64) * (int(chunk.max()) + 1 if len(chunk) else 1) + chunk
    iso = np.flatnonzero(~anyd)
    key[iso] = int(key.max()) + 1 + np.arange(len(iso))
    _, cl = np.unique(key, return_inverse=True)
    sizes = np.bincount(cl)
    nb = np.bincount(cls)
    info = {"twin_classes": int(cls[-1] + 1) if len(cls) else 0, "vertices_in_nontrivial_classes": int(nb[nb > 1].sum()) if len(cls) else 0,
            "clusters": int(cl.max() + 1), "max_cluster": int(sizes.max())}
    return cl.astype(np.int64), info


# ----------------------------------------------------------------------------------------------------------------- the C pass and its reference
def agg_reference2(mode, V, eptr, eidx, ew, vw, vptr, vidx, Lmax, Wmax):
    """plain-Python statement of _ml2_agg.c."""
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
            om = float(ew[e]) / float(s - 1) if mode == 0 else float(ew[e])
            seen = set()
            for v in eidx[eptr[e]:eptr[e + 1]]:
                if v == u:
                    continue
                c = lab[int(v)]
                if mode == 1:
                    if c in seen:
                        continue
                    seen.add(c)
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


def c_binary_agg2():
    src = os.path.join(HERE, "_ml2_agg.c")
    sha = hashlib.sha256(open(src, "rb").read()).hexdigest()[:12]
    binp = "/tmp/ml2_agg_%s" % sha
    try:
        P._run(["test", "-x", binp])
    except RuntimeError:
        P._run(["gcc", "-O2", "-ffp-contract=off", "-Wall", "-o", binp, P.wsl_path(src)])
    return binp, sha


def agg_c2(mode, V, eptr, eidx, ew, vw, vptr, vidx, Lmax, Wmax, tmp):
    binp, sha = c_binary_agg2()
    fs = {}
    for name, arr, dt in (("vw", vw, np.int32), ("eptr", eptr, np.int64), ("eidx", eidx, np.int32), ("ew", ew, np.int64), ("vptr", vptr, np.int64), ("vidx", vidx, np.int32)):
        fs[name] = os.path.join(tmp, "agg2_%s.raw" % name)
        np.ascontiguousarray(arr, dt).tofile(fs[name])
    out = os.path.join(tmp, "agg2_out.raw")
    txt = P._run([binp, str(mode), str(V), str(len(eptr) - 1), str(Lmax), str(Wmax)] + [P.wsl_path(fs[k]) for k in ("vw", "eptr", "eidx", "ew", "vptr", "vidx")] + [P.wsl_path(out)])
    info = json.loads(txt.strip().splitlines()[-1])
    info["c_source_sha12"] = sha
    lab = np.fromfile(out, np.int32)
    assert len(lab) == V
    return lab, info


# ----------------------------------------------------------------------------------------------------------------- the level loop
def cap_schedule(kind, Wmax, max_levels):
    if kind == "flat":
        return [Wmax] * max_levels
    caps, c = [], 2
    while c < Wmax:
        caps.append(c)
        c *= 2
    return caps + [Wmax] * max(0, max_levels - len(caps))


def coarsen2(N, fams, method, Lmax, Wmax, R, max_levels, tmp, min_gain=0.05, use_twins=True):
    """cluster map (int64, dense) of the fine vertices, the per-level statistics, the last coarse hypergraph and the cumulative cluster map after every level
    ({level: int32 map}), for one of M1..M4.  Stops when the unsplit pins are <= fine pins / R, when a pass at the FINAL cap removes < min_gain of the vertices, or after max_levels passes."""
    rating, sched = METHODS[method]
    mode = MODE[rating]
    eptr, eidx, ew = sk_nets(N, fams)
    vw = np.ones(N, np.int64)
    P0 = int(len(eidx))
    total = np.arange(N, dtype=np.int64)
    levels = [{"level": 0, "kind": "fine", "V": N, "M": int(len(ew)), "P": P0}]
    maps = {}
    V = N
    if use_twins:
        t = time.time()
        cl, info = open_twins_multi(fams, N, Wmax)
        V2, eptr, eidx, ew, vw, st = C.quotient(V, eptr, eidx, ew, vw, cl)
        total = cl[total]
        levels.append(dict(level=1, kind="T_open_twins", seconds=round(time.time() - t, 1), **info, **st))
        maps[1] = total.astype(np.int32)
        log("T: V %d -> %d, M %d -> %d, P %d -> %d (%.2fx)" % (V, V2, st["M_in"], st["M"], st["P_in"], st["P"], st["P_in"] / max(st["P"], 1)))
        V = V2
    for cap in cap_schedule(sched, Wmax, max_levels):
        if len(levels) > max_levels or len(eidx) <= P0 / float(R):
            break
        t = time.time()
        vptr, vidx = C.transpose(V, eptr, eidx)
        lab, info = agg_c2(mode, V, eptr, eidx, ew, vw, vptr, vidx, Lmax, cap, tmp)
        cl = C.dense(lab)
        Vn = int(cl.max()) + 1
        if V - Vn < min_gain * V:
            levels.append({"level": len(levels), "kind": "A_%s_cap%d" % (rating, cap), "skipped": "gain %.3f < %.2f" % ((V - Vn) / V, min_gain), **info})
            log("A(%s, cap %d): the pass removed %.1f %% of the vertices" % (rating, cap, 100.0 * (V - Vn) / V))
            if cap >= Wmax:
                break
            continue
        V2, eptr, eidx, ew, vw, st = C.quotient(V, eptr, eidx, ew, vw, cl)
        total = cl[total]
        levels.append(dict(level=len(levels), kind="A_%s_cap%d" % (rating, cap), seconds=round(time.time() - t, 1), agg=info, **st))
        maps[len(levels) - 1] = total.astype(np.int32)
        log("A%d(%s, cap %d): V %d -> %d, M %d -> %d, P %d -> %d, max weight %d (%.2fx of fine pins)" % (len(levels) - 1, rating, cap, V, V2, st["M_in"], st["M"], st["P_in"], st["P"],
                                                                                                   int(vw.max()), P0 / max(st["P"], 1)))
        V = V2
    return total, levels, (V, eptr, eidx, ew, vw), maps


# ----------------------------------------------------------------------------------------------------------------- tests
def _rand_fams(rng, N, pairs_each, nfam=2):
    """random symmetric CSRs (rows sorted).  Planted: vertices 0-3 have the same 3 hub neighbours in BOTH families (open twins in both);
    vertices 10-13 have the same hubs in family 0 but distinct single neighbours N-4-i in family 1 (twins in family 0 only)."""
    hubs = [N - 1, N - 2, N - 3]
    fams = []
    for f in range(nfam):
        a = rng.randint(0, N, pairs_each)
        b = rng.randint(0, N, pairs_each)
        m = (a != b) & (a > 13) & (b > 13)
        u, v = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
        eu, ev = [], []
        for g in range(0, 4):
            for h in hubs:
                eu.append(g)
                ev.append(h)
        for i, g in enumerate(range(10, 14)):
            if f == 0:
                for h in hubs:
                    eu.append(g)
                    ev.append(h)
            else:
                eu.append(g)
                ev.append(N - 4 - i)
        k = np.unique(np.concatenate([u.astype(np.int64) * N + v, np.array(eu, np.int64) * N + np.array(ev, np.int64)]))
        fams.append(C.sorted_csr(k // N, k % N, N))
    return fams


def _test():
    tmp = tempfile.mkdtemp(prefix="ml2test_")
    rng = np.random.RandomState(7)
    # 1. C mode 0 == _ml_agg.c bit for bit, C mode 1 == the Python reference, on random hypergraphs with uneven vertex weights
    for trial in range(4):
        V, M = 60 + 10 * trial, 90 + 20 * trial
        eptr, eidx, ew = C._rand_hg(rng, V, M, 8)
        vw = rng.randint(1, 4, V).astype(np.int64)
        vptr, vidx = C.transpose(V, eptr, eidx)
        for Lmax, Wmax in ((6, 7), (10, 12), (10, 40)):
            for mode in (0, 1):
                lc, _ = agg_c2(mode, V, eptr, eidx, ew, vw, vptr, vidx, Lmax, Wmax, tmp)
                lr = agg_reference2(mode, V, eptr, eidx, ew, vw, vptr, vidx, Lmax, Wmax)
                assert np.array_equal(lc, lr), "C != reference, mode %d trial %d" % (mode, trial)
            l0, _ = agg_c2(0, V, eptr, eidx, ew, vw, vptr, vidx, Lmax, Wmax, tmp)
            lo, _ = C.agg_c(V, eptr, eidx, ew, vw, vptr, vidx, Lmax, Wmax, tmp)
            assert np.array_equal(l0, lo), "mode 0 differs from _ml_agg.c"
    log("TEST 1 PASS: C == reference (both modes), mode 0 == _ml_agg.c, 4 hypergraphs x 3 (Lmax, Wmax)")
    # 2. the SPAN rating credits a net once per candidate cluster: three pins of one net in one cluster count ew, not 3 ew
    eptr = np.array([0, 5], np.int64)
    eidx = np.array([0, 1, 2, 3, 4], np.int32)
    ew = np.array([10], np.int64)
    vw = np.ones(5, np.int64)
    vptr, vidx = C.transpose(5, eptr, eidx)
    l1 = agg_reference2(1, 5, eptr, eidx, ew, vw, vptr, vidx, 5, 5)
    assert len(set(l1.tolist())) == 1
    log("TEST 2 PASS: single-net span pass merges the net")
    # 3. multi-family twins: identical in both families -> merged; identical in one family only -> not merged; exact-checked, chunked by Wmax
    N = 400
    fams = _rand_fams(rng, N, 500)
    cl, info = open_twins_multi(fams, N, 32)
    assert len(set(cl[0:4].tolist())) == 1, "planted 2-family twins not merged"
    assert len(set(cl[10:14].tolist())) == 4, "family-0-only twins must NOT merge (family 1 lists differ)"
    cl2, _ = open_twins_multi(fams, N, 2)
    assert np.bincount(cl2).max() <= 2
    log("TEST 3 PASS: multi-family twins %s" % info)
    # 4. sk_nets == the nets built family by family, and the level loop's clustering satisfies the quotient KM1 identity (exact) for every method
    eptr, eidx, ew = sk_nets(N, fams)
    M0 = sum(int((np.diff(xa) >= 1).sum()) for xa, _ in fams)
    P00 = sum(int(np.diff(xa).sum() + (np.diff(xa) >= 1).sum()) for xa, _ in fams)
    assert len(ew) == M0 and len(eidx) == P00
    for method in METHODS:
        total, lv, (V, ep, ei, ew2, vw2), maps = coarsen2(N, fams, method, 50, 8, 100.0, 8, tmp)
        assert vw2.sum() == N and total.max() + 1 == V and np.array_equal(maps[max(maps)], total.astype(np.int32)) and len(maps) == sum(1 for x in lv if x["level"] > 0 and "skipped" not in x and "stopped" not in x)
        part = rng.randint(0, 5, V)
        fine_part = part[total]
        eptr0, eidx0, ew0 = sk_nets(N, fams)
        assert C.km1(ep, ei, ew2, part) == C.km1(eptr0, eidx0, ew0, fine_part), "quotient KM1 identity broken (%s)" % method
    log("TEST 4 PASS: sk_nets counts, quotient KM1 identity for M1..M4")
    log("TEST PASS")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "TEST":
        _test()
    else:
        raise SystemExit(__doc__)
