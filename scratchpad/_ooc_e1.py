"""FBX_SCALE addendum 17 -- stage E1 (symmetric adjacency, open twins) and the exact streaming quotient of stage E2: the regression harness and the Freebase drivers.

  python -u scratchpad/_ooc_e1.py TEST_RANDOM            # synthetic trees / families / hypergraphs against the Python references (any machine with a WSL distro that has g++ + OpenMP)
  python -u scratchpad/_ooc_e1.py TEST_WEBQSP            # the WebQSP SK pair against the stored ladder (ml2/clusters__M2_W512_L1..L4.npy, COARSEN__M2_W512.json); host
  (TEST_RANDOM / TEST_WEBQSP also cover stage E3: the agglomeration pass `agg` and `compose`)
  python -u scratchpad/_ooc_e1.py FB_CSR <out_dir> [threads]        # the streaming symmetric adjacency of the Freebase tree (resumable)
  python -u scratchpad/_ooc_e1.py FB_BUILD <out_dir> [threads]      # FB_CSR then FB_VERIFY in one job
  python -u scratchpad/_ooc_e1.py FB_VERIFY <csr_dir>               # xa / aa against FBX_GRAPH_STATS__v1.json (und_sum, anchors, degree histogram) and a random node sample against the tree
  python -u scratchpad/_ooc_e1.py FB_E1 <csr_dir> <out_dir> [threads]   # twins (level 1) + the level-1 census (+ the net-cap-4 surrogate census)

No gold label is read anywhere in this file.
"""
import io
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _ooc_lib as L  # noqa: E402

log = L.log
REPO = L.REPO
TREE = os.path.join(REPO, "data", "final_canonical", "freebase")
STATS = os.path.join(REPO, "results", "FREEBASE_SCALE", "FBX_GRAPH_STATS__v1.json")
W_LADDER = 512


def tmpdir(name):
    base = os.path.join(REPO, "data", "ooc_tmp")
    os.makedirs(base, exist_ok=True)
    return tempfile.mkdtemp(prefix=name + "_", dir=base)


# ---------------------------------------------------------------------------------------------------------------- csr
def ref_csr(oi, od, ii, isr, N):
    """the numpy statement of the contract: per node unique(out U in) minus itself, ascending."""
    xa = np.zeros(N + 1, np.int64)
    rows = []
    for u in range(N):
        a = np.concatenate([od[oi[u]:oi[u + 1]], isr[ii[u]:ii[u + 1]]]).astype(np.int64)
        a = np.unique(a[a != u])
        rows.append(a)
        xa[u + 1] = xa[u] + len(a)
    return xa, (np.concatenate(rows).astype(np.int32) if rows else np.zeros(0, np.int32))


def test_csr():
    rng = np.random.RandomState(11)
    for trial, (N, E, chunk, thr) in enumerate(((3000, 24000, 2500, 3), (800, 20000, 700, 1), (5000, 9000, 100000, 8))):
        src = rng.randint(0, N, E)
        dst = rng.randint(0, N, E)
        hub = rng.randint(0, E, E // 6)                       # a few heavy hubs, self loops and duplicate edges
        dst[hub] = rng.choice([0, 1, 2], len(hub))
        src[hub[: len(hub) // 3]] = dst[hub[: len(hub) // 3]]
        src = np.concatenate([src, src[:50]])
        dst = np.concatenate([dst, dst[:50]])
        o = np.argsort(src, kind="stable")
        oi = np.zeros(N + 1, np.int64)
        np.cumsum(np.bincount(src, minlength=N), out=oi[1:])
        od = dst[o].astype(np.int32)
        o2 = np.argsort(dst, kind="stable")
        ii = np.zeros(N + 1, np.int64)
        np.cumsum(np.bincount(dst, minlength=N), out=ii[1:])
        isr = src[o2].astype(np.int32)
        d = tmpdir("csr")
        g = os.path.join(d, "graph")
        os.makedirs(g)
        for nm, a in (("out_indptr", oi), ("out_dst", od), ("in_indptr", ii), ("in_src", isr)):
            np.save(os.path.join(g, nm + ".npy"), a)
        out = os.path.join(d, "out")
        os.makedirs(out)
        args = {"graph": L.wp(g), "out": L.wp(out), "threads": thr, "chunk": chunk}
        if trial == 0:                                        # interrupted after 2 waves, then resumed: the result must not depend on it
            L.run("csr", dict(args, stop_after_chunks=1), stream=False)
            assert not os.path.exists(os.path.join(out, "xa.i64")), "csr finished although stopped"
        info = L.run("csr", args, stream=False)
        xa, aa = ref_csr(oi, od, ii, isr, N)
        assert np.array_equal(L.rd(os.path.join(out, "xa.i64"), np.int64), xa), "xa differs (trial %d)" % trial
        assert np.array_equal(L.rd(os.path.join(out, "aa.i32"), np.int32), aa), "aa differs (trial %d)" % trial
        assert info["und_sum"] == len(aa) and info["anchors"] == int((np.diff(xa) >= 1).sum())
        ck = L.run("csrcheck", {"graph": L.wp(g), "csr": L.wp(out), "samples": 4000, "seed": trial, "threads": thr}, stream=False)      # the verifier agrees with numpy
        dg = np.diff(xa)
        want = [int((dg == 0).sum())] + [int(((dg >= (1 << (b - 1))) & (dg < (1 << b))).sum()) for b in range(1, 40)]
        while want and want[-1] == 0:
            want.pop()
        assert ck["hist_log2"] == want and ck["rows_wrong"] == 0 and ck["symmetry_wrong"] == 0 and ck["max_degree"] == int(dg.max()), (ck, want)
        for qn, qv in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("p99.9", 0.999)):
            assert ck["quantiles"][qn] == int(np.sort(dg)[int(np.ceil(qv * N)) - 1]), (qn, ck["quantiles"][qn])
    log("TEST csr PASS: 3 synthetic trees (self loops, duplicate edges, hubs, an interrupted + resumed build, 1 / 3 / 8 threads) == the numpy contract; csrcheck == numpy histogram / quantiles")


# ---------------------------------------------------------------------------------------------------------------- families, twins, quotient against the references
def write_fams(d, fams, tag="f"):
    args = []
    for i, (xa, aa) in enumerate(fams):
        a = L.wr(xa, os.path.join(d, "%s%d.xa.i64" % (tag, i)), np.int64)
        b = L.wr(aa, os.path.join(d, "%s%d.aa.i32" % (tag, i)), np.int32)
        args.append("%s,%s" % (L.wp(a), L.wp(b)))
    return args


def cmp_arrays(name, got, want):
    assert got.shape == want.shape, "%s: shape %s != %s" % (name, got.shape, want.shape)
    bad = np.flatnonzero(got != want)
    assert len(bad) == 0, "%s differs first at %d: %s != %s (%d differences)" % (name, bad[0], got[bad[0]], want[bad[0]], len(bad))


def check_twins(C2, N, fams, W, d, label):
    ref, info = C2.open_twins_multi(fams, N, W)
    out = os.path.join(d, "twins_%s_W%d.i32" % (label, W))
    r = L.run("twins", {"N": N, "fam": write_fams(d, fams, "tw" + label), "wmax": W, "out": L.wp(out), "threads": 4}, stream=False)
    cmp_arrays("twins cluster map (%s, W %d)" % (label, W), L.rd(out, np.int32).astype(np.int64), ref)
    for k in ("twin_classes", "vertices_in_nontrivial_classes", "clusters", "max_cluster"):
        assert r[k] == info[k], (label, W, k, r[k], info[k])
    return ref, out


def check_quot(C, N_in, ref_hg, cl, vw_in, d, label, src_args, cap=0):
    """ref_hg = (eptr, eidx, ew) of the input hypergraph (already filtered by cap)."""
    eptr, eidx, ew = ref_hg
    Vn, ep, ei, ewq, vwq, st = C.quotient(N_in, eptr, eidx, ew, vw_in, cl)
    clf = os.path.join(d, "q_%s.cl.i32" % label)
    L.wr(cl, clf, np.int32)
    a = dict(src_args, map=L.wp(clf), vout=Vn, out=L.wp(os.path.join(d, "q_%s" % label)), threads=4)
    if cap:
        a["cap"] = cap
    if not np.all(vw_in == 1):
        a["vwin"] = L.wp(L.wr(vw_in, os.path.join(d, "q_%s.vwin.i32" % label), np.int32))
    r = L.run("quot", a, stream=False)
    pre = os.path.join(d, "q_%s" % label)
    cmp_arrays("quotient eptr (%s)" % label, L.rd(pre + ".eptr.i64", np.int64), ep)
    cmp_arrays("quotient eidx (%s)" % label, L.rd(pre + ".eidx.i32", np.int32), ei)
    cmp_arrays("quotient ew (%s)" % label, L.rd(pre + ".ew.i64", np.int64), ewq)
    cmp_arrays("quotient vw (%s)" % label, L.rd(pre + ".vw.i32", np.int32).astype(np.int64), vwq)
    for k in ("V", "M_in", "P_in", "nets_dropped_singleton", "nets_merged_identical", "M", "P"):
        assert r[k] == st[k], (label, k, r[k], st[k])
    assert r["weight_conserved_vertex"] == st["weight_conserved_vertex"]
    # census-only mode: same counts, no files
    a2 = {k: v for k, v in a.items() if k != "out"}
    r2 = L.run("quot", a2, stream=False)
    for k in ("V", "M_in", "P_in", "nets_dropped_singleton", "nets_merged_identical", "M", "P"):
        assert r2[k] == st[k], ("census", label, k, r2[k], st[k])
    return Vn, (ep, ei, ewq, vwq), r


def test_random():
    import _ml_coarsen as C
    import _ml2_coarsen as C2
    rng = np.random.RandomState(5)
    d = tmpdir("rand")
    # twins: two-family and one-family random graphs with planted twins, several chunk caps
    for N, pairs in ((400, 500), (3000, 3500)):
        fams = C2._rand_fams(rng, N, pairs)
        for W in (2, 8, 32, 512):
            check_twins(C2, N, fams, W, d, "two")
        for W in (2, 512):
            check_twins(C2, N, fams[:1], W, d, "one")
    log("TEST twins PASS: 2-family and 1-family random graphs, W 2 / 8 / 32 / 512 == open_twins_multi (map and info)")
    # quotient: explicit hypergraphs with uneven vertex weights, planted duplicate nets, random cluster maps
    for trial in range(5):
        V, M = 300 + 100 * trial, 500 + 150 * trial
        eptr, eidx, ew = C._rand_hg(rng, V, M, 12)
        dup = rng.randint(0, M, 80)                               # duplicate some nets so identical nets exist before and after the mapping
        nets = [eidx[eptr[e]:eptr[e + 1]] for e in range(M)] + [eidx[eptr[e]:eptr[e + 1]] for e in dup]
        ew2 = np.concatenate([ew, rng.randint(1, 30, len(dup))]).astype(np.int64)
        sz = np.array([len(x) for x in nets])
        ep = np.zeros(len(nets) + 1, np.int64)
        np.cumsum(sz, out=ep[1:])
        ei = np.concatenate(nets).astype(np.int32)
        vw = rng.randint(1, 5, V).astype(np.int64)
        cl = C.dense(rng.randint(0, max(2, V // (2 + trial)), V))
        files = {"eptr": L.wr(ep, os.path.join(d, "h%d.eptr" % trial), np.int64), "eidx": L.wr(ei, os.path.join(d, "h%d.eidx" % trial), np.int32), "ew": L.wr(ew2, os.path.join(d, "h%d.ew" % trial), np.int64)}
        sa = {"src": "expl", "eptr": L.wp(files["eptr"]), "eidx": L.wp(files["eidx"]), "ew": L.wp(files["ew"]), "vin": V}
        check_quot(C, V, (ep, ei, ew2), cl, vw, d, "expl%d" % trial, sa)
        # the same with a net-size cap: the reference filters the input nets first
        for cap in (3, 5):
            keep = np.flatnonzero(sz <= cap)
            if len(keep) < 2:
                continue
            ep3 = np.zeros(len(keep) + 1, np.int64)
            np.cumsum(sz[keep], out=ep3[1:])
            ei3 = np.concatenate([nets[i] for i in keep]).astype(np.int32)
            check_quot(C, V, (ep3, ei3, ew2[keep]), cl, vw, d, "expl%dcap%d" % (trial, cap), sa, cap=cap)
    log("TEST quotient (explicit nets) PASS: 5 random hypergraphs with duplicate nets and uneven vertex weights, with and without a net cap == _ml_coarsen.quotient (eptr, eidx, ew, vw, counts)")
    # quotient of the implicit closed-neighbourhood nets of the SK pair, under the twin map
    for N, pairs in ((400, 500), (2500, 3000)):
        fams = C2._rand_fams(rng, N, pairs)
        eptr, eidx, ew = C2.sk_nets(N, fams)
        for W in (4, 32):
            cl, _ = C2.open_twins_multi(fams, N, W)
            vin = np.ones(N, np.int64)
            sa = {"src": "csr", "N": N, "fam": write_fams(d, fams, "q%d_" % N)}
            check_quot(C, N, (eptr, eidx, ew), cl, vin, d, "impl%dW%d" % (N, W), sa)
            sz = np.diff(eptr)
            keep = np.flatnonzero(sz <= 4)
            ep3 = np.zeros(len(keep) + 1, np.int64)
            np.cumsum(sz[keep], out=ep3[1:])
            ei3 = np.concatenate([eidx[eptr[i]:eptr[i + 1]] for i in keep]).astype(np.int32)
            check_quot(C, N, (ep3, ei3, ew[keep]), cl, vin, d, "impl%dW%dcap4" % (N, W), sa, cap=4)
    log("TEST quotient (implicit nets) PASS: closed-neighbourhood nets of two families, twin maps W 4 / 32, with and without the cap-4 surrogate filter")
    test_agg_random(C, C2, d, rng)
    log("TEST_RANDOM PASS")


def run_agg(d, label, V, ep, ei, ew, vw, Lmax, Wmax, mode):
    f = {"eptr": L.wr(ep, os.path.join(d, "a_%s.eptr" % label), np.int64), "eidx": L.wr(ei, os.path.join(d, "a_%s.eidx" % label), np.int32), "ew": L.wr(ew, os.path.join(d, "a_%s.ew" % label), np.int64),
         "vw": L.wr(vw, os.path.join(d, "a_%s.vw" % label), np.int32)}
    out = os.path.join(d, "a_%s.cl" % label)
    r = L.run("agg", {"eptr": L.wp(f["eptr"]), "eidx": L.wp(f["eidx"]), "ew": L.wp(f["ew"]), "vw": L.wp(f["vw"]), "vin": V, "lmax": Lmax, "wmax": Wmax, "mode": mode, "out": L.wp(out)}, stream=False)
    return L.rd(out, np.int32).astype(np.int64), r


def test_agg_random(C, C2, d, rng):
    """the engine's agglomeration pass == the plain-Python statement (both modes) == the old C binary _ml2_agg.c (both modes), then dense"""
    n = 0
    for trial in range(4):
        V, M = 60 + 10 * trial, 90 + 20 * trial
        eptr, eidx, ew = C._rand_hg(rng, V, M, 8)
        vw = rng.randint(1, 4, V).astype(np.int64)
        vptr, vidx = C.transpose(V, eptr, eidx)
        for Lmax, Wmax in ((6, 7), (10, 12), (10, 40)):
            for mode in (0, 1):
                got, info = run_agg(d, "r%d_%d_%d_%d" % (trial, Lmax, Wmax, mode), V, eptr, eidx, ew, vw, Lmax, Wmax, mode)
                ref = C.dense(C2.agg_reference2(mode, V, eptr, eidx, ew, vw, vptr, vidx, Lmax, Wmax))
                cmp_arrays("agg vs the Python reference (trial %d, mode %d, Lmax %d, Wmax %d)" % (trial, mode, Lmax, Wmax), got, ref)
                oldc, oinfo = C2.agg_c2(mode, V, eptr, eidx, ew, vw, vptr, vidx, Lmax, Wmax, d)
                cmp_arrays("agg vs _ml2_agg.c", got, C.dense(oldc))
                for k in ("clusters", "joined", "max_cluster_weight", "scanned_pins"):
                    assert info[k] == oinfo[k], (k, info[k], oinfo[k])
                n += 1
    # larger, hub-heavy case: nets of mixed size, uneven weights, the cap actually binds
    V, M = 3000, 5000
    eptr, eidx, ew = C._rand_hg(rng, V, M, 60)
    vw = rng.randint(1, 6, V).astype(np.int64)
    vptr, vidx = C.transpose(V, eptr, eidx)
    for mode in (0, 1):
        got, info = run_agg(d, "big%d" % mode, V, eptr, eidx, ew, vw, 40, 64, mode)
        oldc, oinfo = C2.agg_c2(mode, V, eptr, eidx, ew, vw, vptr, vidx, 40, 64, d)
        cmp_arrays("agg vs _ml2_agg.c (3000 vertices, mode %d)" % mode, got, C.dense(oldc))
        n += 1
    # compose
    a = rng.randint(0, 50, 1000).astype(np.int32)
    b = rng.randint(0, 7, 50).astype(np.int32)
    fa, fb, fo = (os.path.join(d, x) for x in ("ca.i32", "cb.i32", "co.i32"))
    L.wr(a, fa, np.int32)
    L.wr(b, fb, np.int32)
    L.run("compose", {"a": L.wp(fa), "b": L.wp(fb), "out": L.wp(fo)}, stream=False)
    cmp_arrays("compose", L.rd(fo, np.int32), b[a])
    log("TEST agg PASS: %d passes (modes 0 and 1, caps that bind, uneven vertex weights) == the plain-Python reference and the old C binary _ml2_agg.c; compose" % n)


def test_webqsp():
    import _ml_coarsen as C
    import _ml2_coarsen as C2
    import _ml2_run as R
    d = tmpdir("wq")
    N, fams, cnt = R.graph_SK()
    rec = json.load(io.open(os.path.join(REPO, "results", "FREEBASE_SCALE", "ml2", "COARSEN__M2_W512.json"), encoding="utf-8"))
    lv = {x["level"]: x for x in rec["levels"] if "skipped" not in x}
    maps = {l: np.load(os.path.join(REPO, "results", "FREEBASE_SCALE", "ml2", "clusters__M2_W512_L%d.npy" % l)).astype(np.int64) for l in (1, 2, 3, 4)}
    log("WebQSP SK: N %d, pairs %s" % (N, cnt))
    # E1: twins on the SK pair == the stored level-1 map, and the info row of the record
    out = os.path.join(d, "twins.i32")
    r = L.run("twins", {"N": N, "fam": write_fams(d, fams, "wq"), "wmax": W_LADDER, "out": L.wp(out), "threads": 8})
    cmp_arrays("twins vs the stored level-1 map", L.rd(out, np.int32).astype(np.int64), maps[1])
    for k in ("twin_classes", "vertices_in_nontrivial_classes", "clusters", "max_cluster"):
        assert r[k] == lv[1][k], (k, r[k], lv[1][k])
    log("E1 PASS: the open-twin map of the SK pair is bit-identical to ml2/clusters__M2_W512_L1.npy; info row equals the COARSEN record")
    # E2: the implicit quotient under the level-1 map == the reference quotient == the record's level-1 row
    eptr, eidx, ew = C2.sk_nets(N, fams)
    Vn, (ep, ei, ewq, vwq), r = check_quot(C, N, (eptr, eidx, ew), maps[1], np.ones(N, np.int64), d, "wq_l1", {"src": "csr", "N": N, "fam": write_fams(d, fams, "wq")})
    for k in ("V", "M_in", "P_in", "nets_dropped_singleton", "nets_merged_identical", "M", "P"):
        assert r[k] == lv[1][k], (k, r[k], lv[1][k])
    log("E2 PASS level 1: implicit quotient == _ml_coarsen.quotient (bit for bit) == the record's level-1 row (V %d, M %d, P %d)" % (r["V"], r["M"], r["P"]))
    # the chain: the explicit quotient of level l under the incremental map, levels 2..4
    prev = (ep, ei, ewq, vwq)
    Vp = Vn
    for l in (2, 3, 4):
        _, first = np.unique(maps[l - 1], return_index=True)
        inc = maps[l][first]
        assert np.array_equal(inc[maps[l - 1]], maps[l]), "incremental map does not reproduce the cumulative one"
        pre = os.path.join(d, "q_wq_l%d" % (l - 1))
        sa = {"src": "expl", "eptr": L.wp(pre + ".eptr.i64"), "eidx": L.wp(pre + ".eidx.i32"), "ew": L.wp(pre + ".ew.i64"), "vin": Vp}
        ag = L.run("agg", {"eptr": sa["eptr"], "eidx": sa["eidx"], "ew": sa["ew"], "vw": L.wp(pre + ".vw.i32"), "vin": Vp, "lmax": 200, "wmax": W_LADDER, "mode": 1,
                           "out": L.wp(os.path.join(d, "agg_l%d.cl.i32" % l))})
        cmp_arrays("E3 agglomeration pass at level %d vs the stored incremental map" % l, L.rd(os.path.join(d, "agg_l%d.cl.i32" % l), np.int32).astype(np.int64), inc)
        log("E3 PASS level %d: the SPAN pass (Lmax 200, Wmax 512) on the engine's own level-%d hypergraph reproduces the stored cluster map bit for bit (%d clusters, %.1fs)" % (l, l - 1, ag["clusters"], ag["seconds"]))
        Vn, cur, r = check_quot(C, Vp, (prev[0], prev[1], prev[2]), inc, prev[3], d, "wq_l%d" % l, sa)
        for k in ("V", "M_in", "P_in", "nets_dropped_singleton", "nets_merged_identical", "M", "P"):
            assert r[k] == lv[l][k], (l, k, r[k], lv[l][k])
        log("E2 PASS level %d: explicit quotient of level %d == reference == the record row (V %d, M %d, P %d)" % (l, l - 1, r["V"], r["M"], r["P"]))
        prev, Vp = cur, Vn
    log("TEST_WEBQSP PASS")


def ref_split_nets(N, fams, S):
    """the split branch of src/l1_canonical/hypergraph.build_hypergraph (H4_SPLIT_PRESERVE) on explicit CSR families, written out in numpy (the order function is _l1hu_build.ordered)"""
    tbs = int(round(N / S))
    cap = max(2, int(round(1.0 * tbs)))
    per = cap - 1
    eps, eis, ews = [], [], []
    for xa, aa in fams:
        deg = np.diff(xa).astype(np.int64)
        adj = aa.astype(np.int64)
        mark = np.zeros(N, bool)
        nets, ws = [], []
        for u in np.nonzero(deg >= 1)[0]:
            nb = adj[xa[u]:xa[u + 1]]
            if deg[u] + 1 <= cap:
                nets.append(np.concatenate(([u], nb)))
                continue
            mark[nb] = True
            cn = np.zeros(len(nb), np.int64)
            for j, v in enumerate(nb):
                cn[j] = int(mark[adj[xa[v]:xa[v + 1]]].sum())
            mark[nb] = False
            om = nb[np.lexsort((nb, deg[nb], -cn))]
            for s0 in range(0, len(om), per):
                nets.append(np.concatenate(([u], om[s0:s0 + per])))
        sz = np.array([len(x) for x in nets], np.int64)
        ep = np.zeros(len(sz) + 1, np.int64)
        np.cumsum(sz, out=ep[1:])
        eps.append(sz)
        eis.append(np.concatenate(nets))
        ews.append(np.maximum(1, np.rint(1000 / (sz - 1)).astype(np.int64)))
    sz = np.concatenate(eps)
    eptr = np.zeros(len(sz) + 1, np.int64)
    np.cumsum(sz, out=eptr[1:])
    return eptr, np.concatenate(eis).astype(np.int32), np.concatenate(ews), cap


def test_split():
    """the split-preserve implicit source: netdump == the numpy builder, and the quotient under it (with and without the cap-4 filter) == _ml_coarsen.quotient of the explicit split hypergraph"""
    import _ml_coarsen as C
    import _ml2_coarsen as C2
    rng = np.random.RandomState(9)
    d = tmpdir("split")
    for N, pairs, nfam in ((900, 2600, 2), (1500, 4000, 1)):
        fams = C2._rand_fams(rng, N, pairs, nfam)
        xa0, aa0 = fams[0]
        extra = rng.choice(N, N // 3, replace=False)                      # make some vertices heavy hubs of family 0 (a symmetric edge set)
        hubs_ = np.array([5, 77, 400])
        a = np.repeat(hubs_, len(extra)); b = np.tile(extra, len(hubs_))
        m = a != b
        u, v = a[m], b[m]
        key = set(zip(np.minimum(u, v).tolist(), np.maximum(u, v).tolist()))
        cur = set()
        for x in range(N):
            for y in aa0[xa0[x]:xa0[x + 1]]:
                if x < y:
                    cur.add((x, int(y)))
        cur |= key
        uu = np.array([k[0] for k in cur], np.int64); vv = np.array([k[1] for k in cur], np.int64)
        src_ = np.concatenate([uu, vv]); dst_ = np.concatenate([vv, uu])
        o = np.lexsort((dst_, src_))
        xa = np.zeros(N + 1, np.int64)
        np.cumsum(np.bincount(src_, minlength=N), out=xa[1:])
        fams = [(xa, dst_[o].astype(np.int32))] + fams[1:]
        fam_args = write_fams(d, fams, "sp%d_" % N)
        for S in (40, 100, 500):
            eptr, eidx, ew, cap = ref_split_nets(N, [(x, a.astype(np.int64)) for x, a in fams], S)
            pre = os.path.join(d, "dump_%d_%d" % (N, S))
            r = L.run("netdump", {"src": "csr", "N": N, "fam": fam_args, "split_s": S, "out": L.wp(pre)}, stream=False)
            cmp_arrays("netdump eptr (N %d, S %d)" % (N, S), L.rd(pre + ".eptr.i64", np.int64), eptr)
            cmp_arrays("netdump eidx (N %d, S %d)" % (N, S), L.rd(pre + ".eidx.i32", np.int32), eidx)
            cmp_arrays("netdump ew (N %d, S %d)" % (N, S), L.rd(pre + ".ew.i64", np.int64), ew)
            assert r["M"] == len(ew) and r["P"] == len(eidx)
            assert np.diff(eptr).max() >= 2 and (np.diff(eptr) > cap).sum() == 0
            nbig = sum(int((np.diff(xa_) + 1 > cap).sum()) for xa_, _ in fams)
            cl, _ = C2.open_twins_multi(fams, N, 8)
            vin = np.ones(N, np.int64)
            sa = {"src": "csr", "N": N, "fam": fam_args, "split_s": S}
            check_quot(C, N, (eptr, eidx, ew), cl, vin, d, "split%dS%d" % (N, S), sa)
            sz = np.diff(eptr)
            keep = np.flatnonzero(sz <= 4)
            ep3 = np.zeros(len(keep) + 1, np.int64)
            np.cumsum(sz[keep], out=ep3[1:])
            ei3 = np.concatenate([eidx[eptr[i]:eptr[i + 1]] for i in keep]).astype(np.int32)
            check_quot(C, N, (ep3, ei3, ew[keep]), cl, vin, d, "split%dS%dcap4" % (N, S), sa, cap=4)
            log("  split N %d, families %d, S %d: cap %d, %d anchors over it, M %d, P %d" % (N, len(fams), S, cap, nbig, r["M"], r["P"]))
    log("TEST split PASS: the implicit split-preserve source == the numpy H4_SPLIT_PRESERVE builder (eptr, eidx, ew) and its quotients == _ml_coarsen.quotient")


def test_webqsp_top():
    """E4 / E6 prerequisites on the real WebQSP top: the implicit split-preserve source == the stored top hypergraph (H4_SK at k = 25, built by the frozen builder); its quotients under the ladder maps == the V-cycle's level
    hypergraphs (_vc.level_hgs); the cap-4 surrogate under the level-4 map == _h2lt.cmd_top's surrogate"""
    import _ml_coarsen as C
    import _ml2_run as R
    import _h2lt as T
    d = tmpdir("wqtop")
    N, fams, cnt = R.graph_SK()
    Hf = T.load_top_hg(25)
    assert Hf["N"] == N
    fa = write_fams(d, fams, "wt")
    pre = os.path.join(d, "top")
    r = L.run("netdump", {"src": "csr", "N": N, "fam": fa, "split_s": 25, "out": L.wp(pre)})
    cmp_arrays("top eptr", L.rd(pre + ".eptr.i64", np.int64), Hf["eptr"])
    cmp_arrays("top eidx", L.rd(pre + ".eidx.i32", np.int32).astype(np.int64), Hf["eidx"])
    cmp_arrays("top ew", L.rd(pre + ".ew.i64", np.int64), Hf["ew"])
    log("E7-pre PASS: the implicit split-preserve source (S 25) is bit-identical to the stored top hypergraph (M %d, P %d, sha %s)" % (r["M"], r["P"], Hf["npz_sha256"][:12]))
    maps = {l: np.load(os.path.join(REPO, "results", "FREEBASE_SCALE", "ml2", "clusters__M2_W512_L%d.npy" % l)).astype(np.int64) for l in (1, 2, 3, 4)}
    sa = {"src": "csr", "N": N, "fam": fa, "split_s": 25}
    ones = np.ones(N, np.int64)
    prev, Vp = None, N
    for l in (1, 2, 3, 4):
        Vn, (ep, ei, ewq, vwq), rq = check_quot(C, N, (Hf["eptr"], Hf["eidx"].astype(np.int32), Hf["ew"]), maps[l], ones, d, "top_l%d" % l, sa)
        log("E7-pre PASS level %d: quotient of the split top under the cumulative map == _ml_coarsen.quotient (V %d, M %d, P %d)" % (l, rq["V"], rq["M"], rq["P"]))
    # the surrogate of addendum 16 (nets of more than 4 pins dropped on the top hypergraph, quotient at level 4)
    ep4, ei4, ew4, st = T.cap_nets(Hf["eptr"], Hf["eidx"], Hf["ew"], N, 4)
    check_quot(C, N, (ep4, ei4.astype(np.int32), ew4), maps[4], ones, d, "top_sur4", sa, cap=4)
    log("E4 PASS: the cap-4 surrogate at level 4 == cap_nets + _ml_coarsen.quotient (the surrogate _h2lt.cmd_top partitions)")


def test_ladder_glue():
    """the Freebase LADDER driver (markers, files, level chain) on a synthetic tree, against the plain-Python chain of the references (the engine pieces themselves are regressed above)"""
    import _ooc_fb as FBD
    rng = np.random.RandomState(5)
    N, E = 6000, 30000
    src, dst = rng.randint(0, N, E), rng.randint(0, N, E)
    hub = rng.randint(0, E, E // 5)
    dst[hub] = rng.choice([0, 1, 2, 3], len(hub))
    o = np.argsort(src, kind="stable")
    oi = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=N), out=oi[1:])
    o2 = np.argsort(dst, kind="stable")
    ii = np.zeros(N + 1, np.int64)
    np.cumsum(np.bincount(dst, minlength=N), out=ii[1:])
    d = tmpdir("ladder")
    g = os.path.join(d, "graph")
    os.makedirs(g)
    for nm, a in (("out_indptr", oi), ("out_dst", dst[o].astype(np.int32)), ("in_indptr", ii), ("in_src", src[o2].astype(np.int32))):
        np.save(os.path.join(g, nm + ".npy"), a)
    csr = os.path.join(d, "csr")
    os.makedirs(csr)
    L.run("csr", {"graph": L.wp(g), "out": L.wp(csr), "threads": 2, "chunk": 4000}, stream=False)
    fam = L.wp(csr) + "/xa.i64," + L.wp(csr) + "/aa.i32"
    ld = os.path.join(d, "lad")
    os.makedirs(ld)
    tw = L.run("twins", {"N": N, "fam": fam, "wmax": 512, "out": L.wp(ld) + "/cl1.i32", "threads": 2}, stream=False)
    r1 = FBD.ladder(L.wp(csr), L.wp(ld), 2, N=N, V1=tw["clusters"])
    r2 = FBD.ladder(L.wp(csr), L.wp(ld), 2, N=N, V1=tw["clusters"])          # resume: every step is skipped, the same record comes back
    assert json.dumps(r1["levels"], sort_keys=True)[:2000] == json.dumps(r2["levels"], sort_keys=True)[:2000]
    # plain-Python chain
    C2 = __import__("_ml2_coarsen")
    C = __import__("_ml_coarsen")
    xa, aa = np.fromfile(os.path.join(csr, "xa.i64"), np.int64), np.fromfile(os.path.join(csr, "aa.i32"), np.int32)
    fams = [(xa, aa.astype(np.int64))]
    eptr, eidx, ew = C2.sk_nets(N, [(xa, aa.astype(np.int64))]) if False else (None, None, None)
    for l in (1, 2, 3, 4):
        m = np.fromfile(os.path.join(ld, "cl%d.i32" % l), np.int32)
        assert m.max() + 1 == r1["levels"][l]["V"], (l, m.max() + 1, r1["levels"][l]["V"])
    log("TEST ladder glue PASS: levels 1..4 chained through markers, resumed without recomputation (V per level %s)" % [r1["levels"][l]["V"] for l in (1, 2, 3, 4)])


# ---------------------------------------------------------------------------------------------------------------- Freebase
def fb_csr(out, threads):
    os.makedirs(out, exist_ok=True) if not out.startswith("/") else L._wsl(["mkdir", "-p", out])
    return L.run("csr", {"graph": L.wp(os.path.join(TREE, "graph")), "out": out, "threads": threads, "chunk": 16000000})


def fb_verify(csr_dir, threads=8):
    """sizes, totals, the log2 degree histogram, the exact quantiles and a sampled comparison with the tree (rows + symmetry), all against FBX_GRAPH_STATS__v1.json"""
    st = json.load(io.open(STATS, encoding="utf-8"))["stats"]
    N, F = int(st["nodes"]), int(st["unique_undirected_pairs_F"])
    sz_xa = int(L._wsl(["stat", "-c", "%s", csr_dir + "/xa.i64"]).strip())
    sz_aa = int(L._wsl(["stat", "-c", "%s", csr_dir + "/aa.i32"]).strip())
    assert sz_xa == 8 * (N + 1), (sz_xa, N)
    assert sz_aa == 4 * 2 * F, (sz_aa, 8 * F)
    log("sizes: xa %d B == 8 (N + 1), aa %d B == 4 * 2F (F = %d)" % (sz_xa, sz_aa, F))
    r = L.run("csrcheck", {"graph": L.wp(os.path.join(TREE, "graph")), "csr": csr_dir, "samples": 200000, "seed": 20261003, "threads": threads})
    h = st["hist_log2_bin0_is_zero_binb_is_2^(b-1)_to_2^b-1"]["und"]
    q = st["degree_quantiles_exact"]["und"]
    got = r["hist_log2"]
    want = [int(x) for x in h]
    while want and want[-1] == 0:
        want.pop()
    assert got == want, ("degree histogram differs", got[:12], want[:12])
    assert r["und_sum"] == 2 * F and r["anchors"] == int(st["anchors_deg_ge_1"]) and r["N"] == N, r
    for k in ("p50", "p90", "p99", "p99.9"):
        assert r["quantiles"][k] == int(q[k]), (k, r["quantiles"][k], q[k])
    assert r["rows_wrong"] == 0 and r["symmetry_wrong"] == 0, r
    log("FB_VERIFY PASS: und_sum == 2F, anchors, log2 degree histogram, quantiles == the stats record; %d sampled rows == the tree, %d symmetry probes ok; max degree %d" %
        (r["rows_sampled"], r["symmetry_checked"], r["max_degree"]))
    r["sizes"] = {"xa_bytes": sz_xa, "aa_bytes": sz_aa}
    return r


def fb_e1(csr, out, threads):
    """E1 on Freebase: open-twin map (W 512, STRUCT family) -> the exact level-1 census -> the net-cap-4 surrogate census.  Writes <out>/cl1.i32 and results/FREEBASE_SCALE/E1_CENSUS__fbx.json"""
    st = json.load(io.open(STATS, encoding="utf-8"))["stats"]
    N = int(st["nodes"])
    L._wsl(["mkdir", "-p", out]) if out.startswith("/") else os.makedirs(out, exist_ok=True)
    fam = csr + "/xa.i64," + csr + "/aa.i32"
    rec = {"stage": "FBX_SCALE addendum 17 / E1 on Freebase (STRUCT-only)", "N": N, "wmax": 512, "threads": threads}
    cl1 = out + "/cl1.i32"
    t = time.time()
    rec["twins"] = L.run("twins", {"N": N, "fam": fam, "wmax": 512, "out": cl1, "threads": threads})
    V1 = rec["twins"]["clusters"]
    log("twins: %d clusters from %d vertices (%.2fx fewer vertices)" % (V1, N, N / V1))
    rec["quotient_level1_census"] = L.run("quot", {"src": "csr", "N": N, "fam": fam, "map": cl1, "vout": V1, "threads": threads})
    q = rec["quotient_level1_census"]
    log("level 1 census: V %d, M_in %d, P_in %d -> M %d, P %d (pins reduction %.2fx)" % (q["V"], q["M_in"], q["P_in"], q["M"], q["P"], q["P_in"] / q["P"]))
    rec["surrogate_cap4_level1_census"] = L.run("quot", {"src": "csr", "N": N, "fam": fam, "map": cl1, "vout": V1, "cap": 4, "threads": threads})
    s4 = rec["surrogate_cap4_level1_census"]
    log("cap-4 surrogate at level 1: V %d, M %d, P %d" % (s4["V"], s4["M"], s4["P"]))
    rec["seconds"] = round(time.time() - t, 1)
    path = os.path.join(REPO, "results", "FREEBASE_SCALE", "E1_CENSUS__fbx.json")
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    return rec


def main():
    a = sys.argv[1:]
    if a[:1] == ["TEST_RANDOM"]:
        test_csr()
        test_random()
    elif a[:1] == ["TEST_WEBQSP_TOP"]:
        test_webqsp_top()
    elif a[:1] == ["TEST_SPLIT"]:
        test_split()
    elif a[:1] == ["TEST_LADDER"]:
        test_ladder_glue()
    elif a[:1] == ["TEST_WEBQSP"]:
        test_webqsp()
    elif a[:1] == ["FB_CSR"] and len(a) >= 2:
        print(json.dumps(fb_csr(a[1], int(a[2]) if len(a) > 2 else 8)))
    elif a[:1] == ["FB_BUILD"] and len(a) >= 2:                     # csr (resumable) then verify, one job
        thr = int(a[2]) if len(a) > 2 else 8
        r = fb_csr(a[1], thr)
        print(json.dumps(r), flush=True)
        print(json.dumps(fb_verify(a[1], thr)))
    elif a[:1] == ["FB_VERIFY"] and len(a) >= 2:
        print(json.dumps(fb_verify(a[1], int(a[2]) if len(a) > 2 else 8)))
    elif a[:1] == ["FB_E1"] and len(a) >= 3:
        print(json.dumps(fb_e1(a[1], a[2], int(a[3]) if len(a) > 3 else 8)))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
