"""FBX_SCALE stage 3a -- the WebQSP calibration of the low-memory partitioner ladder (addendum 1, results/FREEBASE_SCALE/).

Question (user ruling 2026-09-30): which cheapest partitioner preserves the frozen L1's ALL-gold recall of the WebQSP PHG reference maps?
Everything but the node -> block table is the frozen IR_L1 of REPORT 44.1: FLAT dense + SPLADE RRF (K0 60), one-hop LOC over
STRUCT_out u STRUCT_in u KNN u NER (ACT 200), ES shard ranking by first occurrence in the router order, served list = the first
n = min(B_N, contacted mass) nodes of the L1 order inside the contacted shards, B_P from KNEE_SDIV on the SAME partitioner's K 100 map and
the alpha 0.75 KSCALE table.  Population: the 786 split-A rows of REPORT 44 (no held-out row).

Modes (each write-once):
  MAPS    build the candidate maps (R1 balanced hash x seeds 0..2, R2 LDG on graph S and on S u KNN) at the K grid; map statistics and cost
          -> work/FBX_CALIB/maps/*.npy + results/FREEBASE_SCALE/CALIB_MAPS_WEBQSP__<tag>.json
  BUNDLE  run the frozen FLAT + L1a arithmetic once per row and cache what any map needs: the router order oq, the localisation entries
          (u, hit, x), the FULL L1 total order (int32, N per row) and each gold's position in it
          -> work/FBX_CALIB/bundle_<tag>/ (needs the WebQSP tree: run on the host)
  EVAL    score every map (and the PHG reference maps) against the bundle: MATCHED B_P (the reference's B_P per row) and OWN B_P (the map's own
          KNEE_SDIV + alpha table); ALL / ANY / gold counts at every B_N; paired counts against the PHG cell; the gate verdict fixed in the
          addendum -> results/FREEBASE_SCALE/CALIB_EVAL_WEBQSP__<tag>.{json,npz}

  python scratchpad/_fbx_calib.py MAPS <tag>
  python scratchpad/_fbx_calib.py BUNDLE <tag> [--rows=K]
  python scratchpad/_fbx_calib.py EVAL <tag> <bundle_tag> [--rows=K]
"""
import io
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D
import _l1d_edgediag as E
import _l1d_route as RT
import _l1d_kscale as KS
import _fbx_part as P

log = D.log
K0, ACT, M_CURVE = D.K0, D.ACT, D.M_CURVE
MMAX = max(M_CURVE)
KGRID = (100, 250, 500, 1000, 2000, 5000, 25928)
GATE_K = (100, 250, 500)
SEEDS = (0, 1, 2)
K_REF = 100
ALPHA = "0.75"
WORK = os.path.join(D.REPO, "work", "FBX_CALIB")
OUTR = os.path.join(D.REPO, "results", "FREEBASE_SCALE")
PARTS = os.path.join(D.REPO, "results", "L1_HOST", "parts")
KEYS = os.path.join(D.REPO, "data", "l1_canonical", "webqsp", "keys.npz")
POP_JSON = os.path.join(D.REPO, "results", "L3_DEV", "l3w_population_webqsp__v1.json")
POP_SHA_PREFIX = "7e67c9b116b2921c"
FLAT_G = os.path.join(D.REPO, "results", "L1_COVPART", "flath2_G_webqsp.npz")
L3W_NPZ = os.path.join(D.REPO, "results", "L3_DEV", "l3w_webqsp__v1.npz")
ADDENDUM = os.path.join(OUTR, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_1.json")
ADDENDUM2 = os.path.join(OUTR, "HOST_STAGE_DECLARATION__FBX_SCALE__v1__ADDENDUM_2.json")
FAMS = ("STRUCT_out", "STRUCT_in", "KNN", "NER")
METHODS = ("PHG",) + tuple("HASH_s%d" % s for s in SEEDS) + ("LDG_S", "LDG_SK", "LP_S", "LP_SK")   # LP_*: rung 3 of addendum 2, maps from _fbx_lp_maps.py
LP_MAPS_TAG = "lp_v1"
REG_L1 = [370, 507, 605, 676, 710, 727]             # REPORT 44.2, L1 on PHG_k25928
REG_UNR = [370, 507, 602, 668, 705, 730]            # REPORT 44.2, UNROUTED L1
LOSS_PASS, LOSS_FAIL = 7, 24                        # rows of 786 (addendum 1: 'roughly 1 point' / '3 points')


def wj(p, o):
    D.G.S.wj(p, o)


def jl(p):
    return json.load(io.open(p, encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------------ MAPS
def map_path(name, K):
    return os.path.join(WORK, "maps", "webqsp__%s__k%d.npy" % (name, K))


def build_maps(tag):
    """the candidate maps: hash in numpy, LDG and every statistic in the C twin (the host env has no numba); every C result was checked against the
    pure-Python reference by _fbx_part.py TESTC before it is used here."""
    rec_path = os.path.join(OUTR, "CALIB_MAPS_WEBQSP__%s.json" % tag)
    assert not os.path.exists(rec_path), "write-once: %s exists" % rec_path
    os.makedirs(os.path.join(WORK, "maps"), exist_ok=True)
    os.makedirs(os.path.join(WORK, "csr"), exist_ok=True)
    z = np.load(KEYS)
    N = int(z["N"][0])
    S, Kn = z["STRUCT"], z["KNN"]
    assert len(np.intersect1d(S, Kn, assume_unique=True)) == 0

    def und(k):
        """undirected, deduplicated, loop-free pair keys (the H4_SK hypergraph's edge families, as sk_csr builds them)."""
        a, b = k // N, k % N
        m = a != b
        return np.unique(np.minimum(a[m], b[m]) * N + np.maximum(a[m], b[m]))
    graphs = {"S": und(S), "SK": und(np.concatenate([S, Kn]))}
    streams, csr_sec = {}, {}
    for g, k in graphs.items():
        t0 = time.time()
        xa, aa = P.csr_from_pairs(k // N, k % N, N)
        assert len(aa) == 2 * len(k)
        streams[g] = P.write_csr_raw(xa, aa, os.path.join(WORK, "csr", "webqsp_%s" % g))
        csr_sec[g] = round(time.time() - t0, 2)
        log("graph %s: %d undirected pairs, CSR %d entries (%.1fs)" % (g, len(k), len(aa), csr_sec[g]))
        del xa, aa
    _, c_sha = P.c_binary()
    rec = {"stage": "FBX_SCALE 3a: candidate maps on WebQSP", "N": N, "K_grid": list(KGRID), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "graphs": {g: {"undirected_pairs": int(len(graphs[g])), "csr_seconds": csr_sec[g]} for g in graphs}, "c_source_sha12": c_sha,
           "keys_npz": {"path": D.rel(KEYS), "sha256": D.sha_file(KEYS)}, "maps": {}, "reference_stats_on_graph_S_and_SK": {}}

    def stats_both(npy, K):
        return {g: P.stats_c(N, K, npy, streams[g]) for g in ("S", "SK")}
    for K in KGRID:
        ph = np.load(os.path.join(PARTS, "webqsp__H4_SK_k%d__PHG_con.npy" % K)).astype(np.int32)
        pnpy = os.path.join(WORK, "maps", "ref_PHG_k%d.npy" % K)
        np.save(pnpy, ph)
        rec["reference_stats_on_graph_S_and_SK"]["PHG__k%d" % K] = stats_both(pnpy, K)
        for s in SEEDS:
            t0 = time.time()
            h = P.hash_balanced(N, K, s)
            sec = round(time.time() - t0, 3)
            np.save(map_path("HASH_s%d" % s, K), h)
            rec["maps"]["HASH_s%d__k%d" % (s, K)] = {"method": "R1 balanced hash", "seed": s, "K": K, "seconds": sec, "sha256": D.sha_file(map_path("HASH_s%d" % s, K)),
                                                     "python_peak_rss_mb_high_water": D.peak_rss_mb(), "stats": stats_both(map_path("HASH_s%d" % s, K), K)}
        for g in ("S", "SK"):
            nm = "LDG_%s" % g
            lab, info = P.ldg_c(N, K, streams[g], os.path.join(WORK, "maps", "%s_k%d.raw" % (nm, K)))
            np.save(map_path(nm, K), lab)
            rec["maps"]["%s__k%d" % (nm, K)] = {"method": "R2 LDG on graph %s (C)" % g, "K": K, "seconds": info["seconds"], "cap": info["cap"], "least_loaded_fallbacks": info["least_loaded_fallbacks"],
                                                "c_peak_rss_kb": info["peak_rss_kb"], "c_anon_bytes": info["anon_bytes"], "sha256": D.sha_file(map_path(nm, K)),
                                                "stats": stats_both(map_path(nm, K), K)}
        log("K %d: maps built (LDG_S %.2fs, LDG_SK %.2fs)" % (K, rec["maps"]["LDG_S__k%d" % K]["seconds"], rec["maps"]["LDG_SK__k%d" % K]["seconds"]))
    rec["code"] = {"scratchpad/_fbx_calib.py": D.sha_file(os.path.abspath(__file__)), "scratchpad/_fbx_part.py": D.sha_file(os.path.join(D.HERE, "_fbx_part.py")),
                   "scratchpad/_fbx_ldg.c": D.sha_file(os.path.join(D.HERE, "_fbx_ldg.c"))}
    rec["addendum"] = {D.rel(ADDENDUM): D.sha_file(ADDENDUM)}
    wj(rec_path, rec)
    log("MAPS -> %s %s" % (rec_path, D.sha_file(rec_path)[:12]))


# ------------------------------------------------------------------------------------------------------------ BUNDLE
class WPop(object):
    """WebQSP population (a copy of _l3w_transfer.WPop, which cannot be imported without the L3 modules)."""

    def __init__(self, cd, rows):
        self.rows = np.asarray(rows, np.int64)
        self.nq = len(self.rows)
        self.qids = [cd.query_ids[int(r)] for r in self.rows]
        self.golds = [np.asarray(g, np.int64) for g in cd.gold(self.rows)]
        self.ngold = np.array([len(g) for g in self.golds], np.int64)
        assert (self.ngold >= 1).all()
        self.gptr = np.zeros(self.nq + 1, np.int64)
        self.gptr[1:] = np.cumsum(self.ngold)
        self.ng_tot = int(self.gptr[-1])


def build_families(cd, N):
    """the four propagation families exactly as _l1d_node1h.build_families (imported: it is the pinned function)."""
    import _l1d_node1h as NH
    F, rec = NH.build_families(cd, N)
    return F, rec


class Bundle(object):
    """per-row L1a arithmetic (== _l3w_transfer.Spec.row up to the routed cells, which are what a map changes)."""

    def __init__(self, cd, pop, N, tag, rows_arg):
        self.N, self.pop = N, pop
        self.F, self.famrec = build_families(cd, N)
        self.dir = os.path.join(WORK, "bundle_%s" % tag)
        os.makedirs(self.dir, exist_ok=False)
        self.ford = np.lib.format.open_memmap(os.path.join(self.dir, "ford.npy"), mode="w+", dtype=np.int32, shape=(pop.nq, N))
        self.gpos = np.zeros(pop.ng_tot, np.int64)
        self.oq, self.eu, self.eh, self.ex, self.nh = [], [], [], [], np.zeros(pop.nq, np.int64)
        self.frank_g = np.zeros(pop.ng_tot, np.int64)
        self.rk = np.full(N, N, np.int64)

    def row(self, j, of, od, os_, npos, frank, g, sl):
        N, F = self.N, self.F
        top = of[:ACT]
        nh = len(top)
        U, H, GW = [], [], []
        for f_ in FAMS:
            Fm = F[f_]
            st = Fm["xadj"][top]
            hit, pos = E.entries(st, Fm["xadj"][top + 1] - st)
            U.append(Fm["adj"][pos].astype(np.int64))
            H.append(hit)
            GW.append(Fm["g"][top][hit])
        u, hit, gw = np.concatenate(U), np.concatenate(H), np.concatenate(GW)
        x = (1.0 / np.arange(1.0, nh + 1.0))[hit] * gw
        L = np.bincount(u, weights=x, minlength=N)
        Lord = D.order_from_score(L, frank)
        f = 1.0 / (K0 + frank.astype(np.float64))
        f[Lord] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
        vq = np.union1d(top, Lord)
        fq = np.zeros(len(vq))
        tq = np.zeros(len(vq))
        fq[np.searchsorted(vq, top)] += 1.0 / (K0 + np.arange(nh, dtype=np.float64))
        il = np.searchsorted(vq, Lord)
        fq[il] += 1.0 / (K0 + np.arange(len(Lord), dtype=np.float64))
        tq[il] = ACT + np.arange(len(Lord))
        tq[np.searchsorted(vq, top)] = np.arange(nh)
        oq = vq[np.lexsort((tq, -fq))]
        order = np.lexsort((frank, -f))                      # the full L1 total order (== _l3w_transfer.prefix_order(f, frank, N))
        self.ford[j] = order.astype(np.int32)
        rk = self.rk
        rk[order] = np.arange(N, dtype=np.int64)
        self.gpos[self.pop.gptr[j]:self.pop.gptr[j + 1]] = rk[g]
        rk[order] = N
        self.frank_g[self.pop.gptr[j]:self.pop.gptr[j + 1]] = frank[g]
        self.oq.append(oq.astype(np.int32))
        self.eu.append(u.astype(np.int32))
        self.eh.append(hit.astype(np.int16))
        self.ex.append(x.astype(np.float32))
        self.nh[j] = nh

    def finish(self, meta):
        self.ford.flush()
        cat = lambda L: np.concatenate(L) if L else np.zeros(0)
        ptr = lambda L: np.concatenate([[0], np.cumsum([len(a) for a in L])]).astype(np.int64)
        np.savez(os.path.join(self.dir, "ev.npz"), oq=cat(self.oq), oq_ptr=ptr(self.oq), eu=cat(self.eu), eh=cat(self.eh), ex=cat(self.ex), e_ptr=ptr(self.eu),
                 nh=self.nh, gpos=self.gpos, frank_g=self.frank_g, rows=self.pop.rows, gptr=self.pop.gptr, gold=np.concatenate(self.pop.golds).astype(np.int64))
        meta["ford_sha256"] = D.sha_file(os.path.join(self.dir, "ford.npy"))
        meta["ev_sha256"] = D.sha_file(os.path.join(self.dir, "ev.npz"))
        wj(os.path.join(self.dir, "meta.json"), meta)


def flat_loop(cd, pop, S, N, tag, flat_check):
    """_l3w_transfer.flat_loop with the products in RAM (the host has it) and a non-strict FLAT identity (counted, reported)."""
    gptr = pop.gptr
    nq = pop.nq
    agd, ags = np.zeros(nq), np.zeros(nq)
    d200 = np.asarray(cd.dense_topk(D.ACT, pop.rows), np.int64)
    s200 = np.asarray(cd.splade_topk(D.ACT, pop.rows), np.int64)
    Qu = D.unit_queries(cd, pop.rows)
    eq, ne = 0, []
    t_ = time.time()
    for b in D.batches(cd, pop.rows, Qu, N):
        j0, j1, SD, SS = b[0], b[1], b[2], b[3]
        for i in range(j1 - j0):
            j = j0 + i
            of, od, os_, npos_j, fv, frank = D.flat_row(np.array(SD[i]), np.array(SS[i]))
            k_ = min(D.N_AGREE, npos_j)
            agd[j] = len(set(od[:D.N_AGREE].tolist()) & set(d200[j, :D.N_AGREE].tolist())) / float(D.N_AGREE)
            ags[j] = (len(set(os_[:k_].tolist()) & set(s200[j, :k_].tolist())) / float(k_)) if k_ else 1.0
            if flat_check is not None:
                ft = flat_check["flat_top"][j]
                same = bool((of[:len(ft)] == ft).all() and (od[:D.ACT] == flat_check["top200_dense"][j]).all() and (os_[:D.ACT] == flat_check["top200_splade"][j]).all())
                eq += int(same)
                if not same:
                    ne.append(j)
            g = pop.golds[j]
            S.row(j, of, od, os_, npos_j, frank, g, slice(gptr[j], gptr[j + 1]))
        log("  %s rows %d / %d (%.0fs, RSS %.0f MB)" % (tag, j1, nq, time.time() - t_, D._rss_mb()))
    agree = {"dense_top100_overlap_mean": round(float(agd.mean()), 5), "splade_top100_overlap_mean": round(float(ags.mean()), 5)}
    assert agree["dense_top100_overlap_mean"] >= D.AGREE_MIN and agree["splade_top100_overlap_mean"] >= D.AGREE_MIN, agree
    return agree, eq, ne, round(time.time() - t_, 1)


def rows_opt():
    return next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--rows=")), None)


def run_bundle(tag):
    ROWS = rows_opt()
    assert D.sha_file(POP_JSON).startswith(POP_SHA_PREFIX), "the population record changed"
    prec = jl(POP_JSON)
    cd = D.AD.CanonicalDataset("webqsp")
    N = int(cd.n_nodes)
    assert prec["dataset_pins"] == {"DATASET_json_RECORD_SHA256": cd.record_sha, "query_index.npz": D.sha_file(cd._query_index_path())}
    rows = prec["rows"][:ROWS] if ROWS else prec["rows"]
    pop = WPop(cd, rows)
    assert pop.qids == prec["query_ids"][:pop.nq]
    zg = np.load(FLAT_G)
    assert (zg["rows"][:pop.nq] == pop.rows).all()
    flat_check = {k: zg[k][:pop.nq] for k in ("flat_top", "top200_dense", "top200_splade")}
    t0 = time.time()
    B = Bundle(cd, pop, N, tag, ROWS)
    log("BUNDLE %s: N %d, %d rows, %d gold nodes, families built (%.0fs)" % (tag, N, pop.nq, pop.ng_tot, time.time() - t0))
    agree, eq, ne, sec = flat_loop(cd, pop, B, N, "bundle", flat_check)
    log("FLAT identity vs flath2_G_webqsp: %d / %d rows equal; differing rows %s" % (eq, pop.nq, ne[:30]))
    meta = {"stage": "FBX_SCALE 3a bundle (frozen FLAT + L1a arithmetic, cached)", "tag": tag, "N": N, "n_rows": pop.nq, "n_gold_nodes": pop.ng_tot, "first_rows_only": ROWS,
            "population": {"path": D.rel(POP_JSON), "sha256": D.sha_file(POP_JSON)}, "flat_identity_rows_equal": eq, "flat_identity_rows_differing": ne, "served_list_agreement": agree,
            "families": B.famrec, "seconds_loop": sec, "peak_rss_mb": D.peak_rss_mb(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "constants": D.CONSTANTS, "platform": D.platform_record(), "code": {"scratchpad/_fbx_calib.py": D.sha_file(os.path.abspath(__file__))},
            "note": "the FLAT identity is counted, not asserted: a host BLAS may order float32 near-ties differently; EVAL's regression gate on REPORT 44.2 decides"}
    B.finish(meta)
    log("BUNDLE done -> %s" % B.dir)


# ------------------------------------------------------------------------------------------------------------ EVAL
def load_maps(tag_maps):
    """{(method, K): int32 block table}, checked against their records."""
    mrec = jl(os.path.join(OUTR, "CALIB_MAPS_WEBQSP__%s.json" % tag_maps))
    lrec = jl(os.path.join(OUTR, "CALIB_MAPS_WEBQSP__%s.json" % LP_MAPS_TAG))
    maps, meta = {}, {}
    for K in KGRID:
        rj = os.path.join(PARTS, "webqsp__H4_SK_k%d__PHG_con.RUN.json" % K)
        R = jl(rj)
        p = os.path.join(D.REPO, R["output"]["file"])
        assert D.sha_file(p) == R["output"]["sha256"], "PHG map K %d differs from its RUN.json" % K
        assert R["STATUS"] == "OK" and R["balance"]["every_block_used"]
        maps[("PHG", K)] = np.load(p).astype(np.int32)
        meta[("PHG", K)] = {"sha256": R["output"]["sha256"], "run_json_sha256": D.sha_file(rj), "file": R["output"]["file"]}
        for m in METHODS[1:]:
            p = map_path(m, K)
            src = lrec if m.startswith("LP_") else mrec
            assert D.sha_file(p) == src["maps"]["%s__k%d" % (m, K)]["sha256"], "map %s K %d differs from its record" % (m, K)
            maps[(m, K)] = np.load(p)
            meta[(m, K)] = {"sha256": src["maps"]["%s__k%d" % (m, K)]["sha256"]}
    mrec = dict(mrec)
    mrec["maps"] = dict(mrec["maps"], **lrec["maps"])
    return maps, meta, mrec


def run_eval(tag, btag, tag_maps):
    ROWS = rows_opt()
    out_j = os.path.join(OUTR, "CALIB_EVAL_WEBQSP__%s.json" % tag)
    out_z = out_j.replace(".json", ".npz")
    assert not os.path.exists(out_j), "write-once: %s exists" % out_j
    bdir = os.path.join(WORK, "bundle_%s" % btag)
    bmeta = jl(os.path.join(bdir, "meta.json"))
    assert bmeta["ev_sha256"] == D.sha_file(os.path.join(bdir, "ev.npz")), "the bundle's ev.npz differs from its meta"
    with np.load(os.path.join(bdir, "ev.npz")) as zev:
        ev = {k: zev[k] for k in zev.files}                     # loaded once: NpzFile re-reads a member on every access
    ford = np.load(os.path.join(bdir, "ford.npy"), mmap_mode="r")
    N = bmeta["N"]
    nq = bmeta["n_rows"] if not ROWS else min(ROWS, bmeta["n_rows"])
    gptr, gpos, rows = ev["gptr"], ev["gpos"], ev["rows"]
    assert D.sha_file(POP_JSON).startswith(POP_SHA_PREFIX) and [int(r) for r in jl(POP_JSON)["rows"][:nq]] == [int(r) for r in rows[:nq]]
    golds = [ev["gold"][gptr[j]:gptr[j + 1]] for j in range(nq)]
    maps, mmeta, mrec = load_maps(tag_maps)
    tabs = {K: KS.mult_tables(K)[0][ALPHA] for K in KGRID}
    wseed = 1.0 / np.arange(1.0, ACT + 1.0)
    NB = len(M_CURVE)
    MB = np.array(M_CURVE)
    ngo = np.diff(gptr)[:nq]
    keys = [(m, K, mode) for m in METHODS for K in KGRID for mode in ("own", "matched")]
    SV = {k: np.zeros((nq, NB), np.int16) for k in keys}
    BP = {k: np.zeros(nq, np.int64) for k in keys}
    CM = {k: np.zeros(nq, np.int64) for k in keys}
    LO = {k: np.zeros(nq, np.int64) for k in keys}
    NOEV = {m: np.zeros(nq, bool) for m in METHODS}
    B0 = {m: np.zeros(nq, np.int64) for m in METHODS}
    SVU = np.zeros((nq, NB), np.int16)
    MPOS = {}                                   # masked gold positions of PHG_k25928 own, for the position regression
    LAT = {"es_order + b + contacted set": [], "served positions": []}
    H100 = {m: maps[(m, K_REF)].astype(np.int64) for m in METHODS}
    t_ = time.time()
    for j in range(nq):
        a, b_ = gptr[j], gptr[j + 1]
        g = golds[j]
        gp = gpos[a:b_]
        SVU[j] = [(gp < min(M, N)).sum() for M in M_CURVE]
        fo = np.asarray(ford[j])
        oq = ev["oq"][ev["oq_ptr"][j]:ev["oq_ptr"][j + 1]].astype(np.int64)
        s0, s1 = ev["e_ptr"][j], ev["e_ptr"][j + 1]
        u = ev["eu"][s0:s1].astype(np.int64)
        hit = ev["eh"][s0:s1].astype(np.int64)
        x = ev["ex"][s0:s1].astype(np.float64)
        nh = int(ev["nh"][j])
        for m in METHODS:                        # B100(q) on the SAME partitioner's K 100 map
            h100 = H100[m]
            Cm = np.bincount(hit * K_REF + h100[u], weights=x, minlength=nh * K_REF).reshape(nh, K_REF)
            sdiv = wseed[:nh] @ (Cm > 0)
            kn_, npz_ = RT.knee(sdiv)
            NOEV[m][j] = not npz_
            B0[m][j] = kn_ if npz_ else K_REF
        for K in KGRID:
            b_ref = None
            for m in METHODS:
                t0 = time.perf_counter()
                hard = maps[(m, K)]
                hq = hard[oq]
                up, first = np.unique(hq, return_index=True)
                fe = np.full(K, len(oq), np.int64)
                fe[up] = first
                ro = np.lexsort((np.arange(K), fe))
                pr = RT.inv(ro)
                prh = pr[hard]
                lo = int(pr[hard[g]].max()) + 1
                b_own = K if NOEV[m][j] else int(tabs[K][B0[m][j]])
                if m == "PHG":
                    b_ref = b_own
                LAT["es_order + b + contacted set"].append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                for mode, b in (("own", b_own), ("matched", b_ref)):
                    if mode == "matched" and (m == "PHG" or b == b_own):     # the same B_P: the same cell (identical arithmetic, computed once)
                        k1 = (m, K, "own")
                        for arr in (SV, BP, CM, LO):
                            arr[(m, K, mode)][j] = arr[k1][j]
                        continue
                    contn = prh < b
                    cm = int(contn.sum())
                    cs = np.cumsum(contn[fo])
                    cg = contn[g]
                    mp = cs[gp] - 1
                    n_ = np.minimum(MB, cm)
                    sv = np.array([(cg & (mp < n_[i])).sum() for i in range(NB)])
                    k1 = (m, K, mode)
                    SV[k1][j], BP[k1][j], CM[k1][j], LO[k1][j] = sv, b, cm, lo
                    if m == "PHG" and K == 25928 and mode == "own":
                        MPOS[j] = np.where(cg, mp, N)
                LAT["served positions"].append(time.perf_counter() - t0)
        if j % 25 == 0 or j == nq - 1:
            log("  EVAL rows %d / %d (%.0fs, RSS %.0f MB)" % (j + 1, nq, time.time() - t_, D._rss_mb()))
    sec_loop = round(time.time() - t_, 1)
    # ---- the regression gate (REPORT 44.2 / the l3w record) -----------------------------------------------------------------------
    allc = lambda sv: [int((sv[:, i] == ngo).sum()) for i in range(NB)]
    reg = {"L1_PHG_k25928_ALL": allc(SV[("PHG", 25928, "own")]), "expected_L1": REG_L1, "UNROUTED_ALL": allc(SVU), "expected_UNROUTED": REG_UNR, "rows": nq}
    z = np.load(L3W_NPZ)
    if nq == len(rows) and nq == 786:
        assert reg["L1_PHG_k25928_ALL"] == REG_L1, "regression FAILED: PHG_k25928 L1 %s != REPORT 44.2 %s" % (reg["L1_PHG_k25928_ALL"], REG_L1)
        assert reg["UNROUTED_ALL"] == REG_UNR, "regression FAILED: UNROUTED %s != %s" % (reg["UNROUTED_ALL"], REG_UNR)
        assert (BP[("PHG", 25928, "own")] == z["B__PHG_k25928"]).all(), "regression FAILED: B_P differs from the l3w record"
        assert (CM[("PHG", 25928, "own")] == z["CMASS__PHG_k25928"]).all(), "regression FAILED: contacted mass differs"
        assert (LO[("PHG", 25928, "own")] == z["LO__PHG_k25928"]).all(), "regression FAILED: LO differs"
        posreg = np.concatenate([np.minimum(MPOS[j], MMAX) for j in range(nq)])
        assert (posreg == np.minimum(z["POS__PHG_k25928__L1"], MMAX)).all(), "regression FAILED: served positions differ from POS__PHG_k25928__L1"
        assert (np.minimum(gpos, MMAX) == np.minimum(z["POS__UNROUTED__L1"], MMAX)).all(), "regression FAILED: UNROUTED positions differ"
        reg["status"] = "PASS: counts, B_P, contacted mass, LO and both position arrays equal the l3w_webqsp__v1 record on all 786 rows"
    else:                                                                       # a prefix run: the same per-row arrays over the rows it has, counted not asserted
        ng_ = int(gptr[nq])
        posreg = np.concatenate([np.minimum(MPOS[j], MMAX) for j in range(nq)])
        pre = {"B_P_rows_equal": int((BP[("PHG", 25928, "own")] == z["B__PHG_k25928"][:nq]).sum()), "contacted_mass_rows_equal": int((CM[("PHG", 25928, "own")] == z["CMASS__PHG_k25928"][:nq]).sum()),
               "LO_rows_equal": int((LO[("PHG", 25928, "own")] == z["LO__PHG_k25928"][:nq]).sum()),
               "served_positions_equal": int((posreg == np.minimum(z["POS__PHG_k25928__L1"][:ng_], MMAX)).sum()), "unrouted_positions_equal": int((np.minimum(gpos[:ng_], MMAX) == np.minimum(z["POS__UNROUTED__L1"][:ng_], MMAX)).sum()),
               "gold_positions_compared": ng_}
        reg["prefix_check_vs_l3w_record"] = pre
        reg["status"] = "SMOKE (%d rows): regression on the full population not asserted; prefix check %s" % (nq, pre)
    log("regression: %s" % reg["status"])
    # ---- tables ----------------------------------------------------------------------------------------------------------------
    from scipy.stats import binomtest
    res = {}
    for K in KGRID:
        ref_all = SV[("PHG", K, "own")] == ngo[:, None]
        for m in METHODS:
            for mode in ("own", "matched"):
                k1 = (m, K, mode)
                sv = SV[k1]
                allm = sv == ngo[:, None]
                e = {"ALL": allc(sv), "ANY": [int((sv[:, i] > 0).sum()) for i in range(NB)], "gold_served": [int(sv[:, i].sum()) for i in range(NB)],
                     "B_P": {"mean": round(float(BP[k1].mean()), 1), "median": float(np.median(BP[k1])), "min": int(BP[k1].min()), "max": int(BP[k1].max())},
                     "B_P_over_K_mean": round(float((BP[k1] / float(K)).mean()), 4), "rows_with_B_P_eq_K": int((BP[k1] == K).sum()),
                     "contacted_mass": {"mean": round(float(CM[k1].mean()), 0), "median": float(np.median(CM[k1]))}, "contacted_mass_over_N_mean": round(float((CM[k1] / float(N)).mean()), 4),
                     "lost_to_reach (B_P < LO)": int((BP[k1] < LO[k1]).sum())}
                if m != "PHG" or mode != "own":
                    gain = [int((allm[:, i] & ~ref_all[:, i]).sum()) for i in range(NB)]
                    lost = [int((~allm[:, i] & ref_all[:, i]).sum()) for i in range(NB)]
                    e["vs_PHG_gained"], e["vs_PHG_lost"] = gain, lost
                    e["vs_PHG_p_sign_descriptive"] = [(1.0 if g_ + l_ == 0 else float("%.3g" % binomtest(g_, g_ + l_, 0.5).pvalue)) for g_, l_ in zip(gain, lost)]
                res["%s|K%d|%s" % (m, K, mode)] = e
    # ---- the gate, exactly as fixed in addendum 1 ---------------------------------------------------------------------------------
    def method_verdict(names):
        loss_max, fail = 0, False
        cells = {}
        for K in GATE_K:
            for nm in names:
                ph = res["PHG|K%d|own" % K]["ALL"]
                mm = res["%s|K%d|matched" % (nm, K)]["ALL"]
                ls = [p - q for p, q in zip(ph, mm)]
                cells["%s K%d" % (nm, K)] = ls
                loss_max = max(loss_max, max(ls))
                if max(ls) >= LOSS_FAIL:
                    fail = True
                own = res["%s|K%d|own" % (nm, K)]
                if own["rows_with_B_P_eq_K"] > nq / 2.0:
                    fail = True
        v = "FAILED" if fail else ("RECALL_PRESERVING" if loss_max <= LOSS_PASS else "BORDERLINE")
        return {"verdict": v, "max_loss_rows_vs_PHG_matched_over_gate_cells": int(loss_max), "loss_rows_by_cell_and_B_N": cells}
    verdicts = {"R1 HASH (worst of seeds 0,1,2)": method_verdict(["HASH_s%d" % s for s in SEEDS]), "R2 LDG on S (deployable; the gate)": method_verdict(["LDG_S"]),
                "R2 LDG on S u KNN (diagnostic)": method_verdict(["LDG_SK"]),
                "R3 LP on S (deployable; the gate)": method_verdict(["LP_S"]), "R3 LP on S u KNN (diagnostic)": method_verdict(["LP_SK"])}
    for k_, v_ in verdicts.items():
        log("%-40s %s (max loss %d rows)" % (k_, v_["verdict"], v_["max_loss_rows_vs_PHG_matched_over_gate_cells"]))
    for K in KGRID:
        for m in METHODS:
            log("K %5d %-8s matched ALL %s | own ALL %s  B_P/K %.3f" % (K, m, res["%s|K%d|matched" % (m, K)]["ALL"], res["%s|K%d|own" % (m, K)]["ALL"], res["%s|K%d|own" % (m, K)]["B_P_over_K_mean"]))
    rec = {"stage": "FBX_SCALE 3a: WebQSP calibration of the low-memory partitioner ladder", "tag": tag, "status": "DEVELOPMENT (calibration of a systems substrate)",
           "definitions": __doc__, "N": N, "n_rows": nq, "M_curve": list(M_CURVE), "K_grid": list(KGRID), "gate_K": list(GATE_K), "loss_pass_rows": LOSS_PASS, "loss_fail_rows": LOSS_FAIL,
           "regression": reg, "unrouted_ALL": allc(SVU), "results": res, "verdicts": verdicts,
           "map_costs_and_structure": {k: {kk: vv for kk, vv in v.items() if kk in ("method", "seconds", "K", "cap", "least_loaded_fallbacks", "sha256", "peak_rss_mb_process_high_water", "stats")} for k, v in mrec["maps"].items()},
           "reference_structure": mrec["reference_stats_on_graph_S_and_SK"],
           "latency_ms_per_row_per_map": {k: D.ms_stats(v) for k, v in LAT.items()}, "seconds_loop": sec_loop,
           "inputs": {"bundle_meta": {"path": D.rel(os.path.join(bdir, "meta.json")), "sha256": D.sha_file(os.path.join(bdir, "meta.json"))}, "maps_record": {"path": D.rel(os.path.join(OUTR, "CALIB_MAPS_WEBQSP__%s.json" % tag_maps)),
                     "sha256": D.sha_file(os.path.join(OUTR, "CALIB_MAPS_WEBQSP__%s.json" % tag_maps))},
                     "lp_maps_record": {"path": D.rel(os.path.join(OUTR, "CALIB_MAPS_WEBQSP__%s.json" % LP_MAPS_TAG)), "sha256": D.sha_file(os.path.join(OUTR, "CALIB_MAPS_WEBQSP__%s.json" % LP_MAPS_TAG))}, "phg_reference": {"%s" % K: mmeta[("PHG", K)] for K in KGRID},
                     "population": {"path": D.rel(POP_JSON), "sha256": D.sha_file(POP_JSON)}, "regression_target": {"path": D.rel(L3W_NPZ), "sha256": D.sha_file(L3W_NPZ)},
                     "addendum": {D.rel(ADDENDUM): D.sha_file(ADDENDUM), D.rel(ADDENDUM2): D.sha_file(ADDENDUM2)}},
           "code": {"scratchpad/_fbx_calib.py": D.sha_file(os.path.abspath(__file__)), "scratchpad/_fbx_part.py": D.sha_file(os.path.join(D.HERE, "_fbx_part.py"))},
           "pinned": D.PINNED, "pinned_repo": D.PINNED_REPO, "platform": D.platform_record(), "process_peak_rss_mb": D.peak_rss_mb(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    arrays = {"rows": rows[:nq], "gptr": gptr[:nq + 1], "ngold": ngo}
    for (m, K, mode) in keys:
        arrays["SV__%s__k%d__%s" % (m, K, mode)] = SV[(m, K, mode)]
        arrays["BP__%s__k%d__%s" % (m, K, mode)] = BP[(m, K, mode)]
    np.savez_compressed(out_z, **arrays)
    rec["npz"] = {"path": os.path.basename(out_z), "sha256": D.sha_file(out_z)}
    wj(out_j, rec)
    log("EVAL -> %s %s" % (out_j, D.sha_file(out_j)[:12]))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "MAPS":
        build_maps(sys.argv[2])
    elif mode == "BUNDLE":
        run_bundle(sys.argv[2])
    elif mode == "EVAL":
        run_eval(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else sys.argv[2])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
