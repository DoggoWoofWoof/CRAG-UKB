"""FULL_VISITED PARTITION CALIBRATION AUDIT -- substrate.

Builds, per query, ONE partition-level table over the FULL_VISITED node universe (every node the
frozen bounded search scored, admitted or beam-pruned, in the nested order
`admitted (frozen `added` order) ++ pruned (static sdir desc, node id asc)`) -- exactly the universe
the contract audit measured, so `S4_FULL_VISITED` is reproducible from this table by construction.

Nothing here re-searches, adds an edge, or scores a node.  Every column is an existing quantity the
beam already accumulated over ALL arrivals before its prune.  Gold never enters a column.

The static partition-side quantities (size, degree, boundary degree, adjacent-partition count) are
pure corpus geometry.  The measured exposure prior is label-free and query-independent, and is
additionally reported under a 2-fold split so it cannot be memorising the evaluation queries.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1ps_router as RT
import _l1sr_eval as EV
import _l1bm_core as BM
import _l1bm_run as RUN
import _l1ss_core as SS

PCD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_PARTITION_CALIBRATION"
K0 = EV.K0
os.makedirs(f"{PCD}/diag", exist_ok=True)
os.makedirs(f"{PCD}/data", exist_ok=True)

# the per-(query, partition) columns of the cached table
COLS = ["part", "first_jj", "nnodes", "rrf_sum", "min_hop", "max_sdir", "path_support",
        "nseeds", "max_psc", "sum_rr", "max_psum", "max_pmin", "n_adm",
        "in_b50", "cpos", "r_rrf", "r_dense", "r_splade"]
BIGR = 10 ** 6


# --------------------------------------------------------------- static corpus-side geometry
def partition_geometry(ds, hard, npart, log=print):
    """size / degree / boundary degree / adjacent-partition count, from the graph alone."""
    fp = f"{PCD}/data/geom_{ds}.npz"
    if os.path.exists(fp):
        return dict(np.load(fp))
    adjp, adji, deg = RUN.topology(ds)
    adjp = np.asarray(adjp); adji = np.asarray(adji)
    n = len(hard)
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(adjp).astype(np.int64))
    sp = hard[src].astype(np.int64)
    dp = hard[adji].astype(np.int64)
    ok = (sp >= 0) & (dp >= 0)
    sp, dp = sp[ok], dp[ok]
    G = {"psize": np.bincount(hard[hard >= 0].astype(np.int64), minlength=npart).astype(np.float64),
         "pdeg": np.bincount(sp, minlength=npart).astype(np.float64),
         "pbdeg": np.bincount(sp[sp != dp], minlength=npart).astype(np.float64)}
    pair = np.unique(sp.astype(np.int64) * npart + dp.astype(np.int64))
    G["padj"] = np.bincount((pair // npart).astype(np.int64), minlength=npart).astype(np.float64)
    G["pnodedeg"] = G["pdeg"] / np.maximum(G["psize"], 1.0)
    np.savez_compressed(fp, **G)
    log(f"   geom {ds}: {npart} partitions, size {G['psize'].mean():.1f}, "
        f"deg {G['pdeg'].mean():.1f}, bdeg {G['pbdeg'].mean():.1f}, adj {G['padj'].mean():.1f}")
    return G


# --------------------------------------------------------------- the FULL_VISITED partition table
def _agg_query(nodes, jj_hop, jj_sdir, jj_cnt, jj_smask, jj_psc, jj_rr, jj_psum, jj_pmin,
               hard, n_adm_nodes, npart):
    """vectorised S4-shaped aggregation over ONE query's visited node list."""
    pr = hard[nodes].astype(np.int64)
    ok = pr >= 0
    pr = pr[ok]
    idx = np.nonzero(ok)[0]
    if len(pr) == 0:
        return None
    up, inv = np.unique(pr, return_inverse=True)
    m = len(up)
    first = np.full(m, BIGR, np.int64); np.minimum.at(first, inv, idx)
    nn = np.bincount(inv, minlength=m).astype(np.int64)
    rrf = np.zeros(m); np.add.at(rrf, inv, 1.0 / (K0 + idx))
    mh = np.full(m, 1 << 20, np.int64); np.minimum.at(mh, inv, jj_hop[ok].astype(np.int64))
    ms = np.full(m, -1e18); np.maximum.at(ms, inv, jj_sdir[ok])
    ps = np.zeros(m, np.int64); np.add.at(ps, inv, jj_cnt[ok].astype(np.int64))
    sm = np.zeros(m, np.int64); np.bitwise_or.at(sm, inv, jj_smask[ok].astype(np.int64))
    mp = np.full(m, -1e18); np.maximum.at(mp, inv, jj_psc[ok])
    rr = np.zeros(m); np.add.at(rr, inv, jj_rr[ok])
    su = np.full(m, -1e18); np.maximum.at(su, inv, jj_psum[ok])
    mn = np.full(m, -1e18); np.maximum.at(mn, inv, jj_pmin[ok])
    adm = np.zeros(m, np.int64)
    a_ok = idx < n_adm_nodes
    if a_ok.any():
        np.add.at(adm, inv[a_ok], 1)
    ns = np.array([bin(int(x)).count("1") for x in sm], np.int64)
    return dict(part=up, first_jj=first, nnodes=nn, rrf_sum=rrf, min_hop=mh, max_sdir=ms,
                path_support=ps, nseeds=ns, max_psc=mp, sum_rr=rr, max_psum=su, max_pmin=mn,
                n_adm=adm)


def build(ds, log=print, nq_max=None):
    """replay the frozen search once and cache the per-query FULL_VISITED partition table."""
    fp = f"{PCD}/data/table_{ds}.npz" if nq_max is None else None
    S = EV.substrate(ds)
    if fp and os.path.exists(fp):
        return S, dict(np.load(fp, allow_pickle=False))
    z = S["z"]
    nq = S["nq"] if nq_max is None else min(S["nq"], nq_max)
    hard = S["hard"].astype(np.int64)
    npart = int(hard.max()) + 1
    adjp, adji, deg = RUN.topology(ds)
    Xn, Qm, _ = RUN.load_emb_frozen(ds, z)
    ndocs = (Xn.A.shape[0] if hasattr(Xn, "A") else Xn.shape[0])
    W = SS.workspace(ndocs)
    fz = {k: np.asarray(z[k]) for k in ("s_node", "s_hop", "s_sdir", "s_cnt")}
    rdn, rsp, rrf_ = np.asarray(z["ret_dense"]), np.asarray(z["ret_splade"]), np.asarray(z["ret_rrf"])
    cols = {c: [] for c in COLS}
    qptr = np.zeros(nq + 1, np.int64)
    work = {"edges": 0, "cands": 0, "visited": 0, "admitted": 0}
    par = 0
    t0 = time.time()
    for qi in range(nq):
        sd = [int(s) for s in z["seeds"][qi] if s >= 0]
        rq = TA.residual(Qm[qi].astype(np.float64), sd, Xn)
        added, vmeta, st, FE = SS.expand_feat(sd, rq, adjp, adji, deg, Xn, W,
                                              want_future=False, want_visited=True)
        sn, sh, sdd, sc = BM.arrays_of(added, vmeta)
        par += int((sn == fz["s_node"][qi]).all() and (sh == fz["s_hop"][qi]).all()
                   and (sdd == fz["s_sdir"][qi]).all() and (sc == fz["s_cnt"][qi]).all())
        na = len(FE["node"])
        nodes = np.concatenate([FE["node"], FE["VIS_PRUNED"]])
        A = _agg_query(nodes,
                       np.concatenate([FE["HOP"], FE["VIS_PRUNED_HOP"].astype(np.float64)]),
                       np.concatenate([FE["STATIC_SDIR"], FE["VIS_PRUNED_SDIR"]]),
                       np.concatenate([FE["PATH_SUPPORT"], FE["VIS_PRUNED_CNT"].astype(np.float64)]),
                       np.concatenate([FE["ADM_SMASK"], FE["VIS_PRUNED_SMASK"]]),
                       np.concatenate([FE["PARENT_SCORE"], FE["VIS_PRUNED_PSC"]]),
                       np.concatenate([FE["RR_PARENT_SUPPORT"], FE["VIS_PRUNED_RR"]]),
                       np.concatenate([FE["PATH_SUM"], FE["VIS_PRUNED_PSUM"]]),
                       np.concatenate([FE["PATH_MIN"], FE["VIS_PRUNED_PMIN"]]),
                       hard, na, npart)
        if A is None:
            # a query whose bounded search visits no partition-bearing node (e.g. no usable seed).
            # emit zero rows: the challenger universe is genuinely empty, so every downstream
            # ordering is the empty order and the frozen contract keeps its canonical top-50.
            A = {k: np.zeros(0, np.int64 if k in ("part", "first_jj", "nnodes", "min_hop",
                                                  "path_support", "nseeds", "n_adm")
                             else np.float64) for k in
                 ("part", "first_jj", "nnodes", "rrf_sum", "min_hop", "max_sdir", "path_support",
                  "nseeds", "max_psc", "sum_rr", "max_psum", "max_pmin", "n_adm")}
        c = S["ctxs"][qi]
        b50 = c["base50"]; cpos = c["cpos"]
        p = A["part"]
        rk = {}
        for nm, arr in (("r_rrf", rrf_), ("r_dense", rdn), ("r_splade", rsp)):
            d = {}
            for jjx, v in enumerate(arr[qi]):
                v = int(v)
                if v < 0:
                    break
                q = int(hard[v])
                if q >= 0 and q not in d:
                    d[q] = jjx
            rk[nm] = np.array([d.get(int(x), BIGR) for x in p], np.int64)
        for k, v in A.items():
            cols[k].append(v)
        cols["in_b50"].append(np.array([int(int(x) in b50) for x in p], np.int64))
        cols["cpos"].append(np.array([cpos.get(int(x), BIGR) for x in p], np.int64))
        for nm in ("r_rrf", "r_dense", "r_splade"):
            cols[nm].append(rk[nm])
        qptr[qi + 1] = qptr[qi] + len(p)
        work["edges"] += st["edges"]; work["cands"] += st["cand_total"]
        work["visited"] += FE["VIS_TOTAL"]; work["admitted"] += na
        if (qi + 1) % 500 == 0:
            log(f"   {ds} {qi+1}/{nq}  parity {par}/{qi+1}  {time.time()-t0:.0f}s")
    T = {c: np.concatenate(cols[c]) for c in COLS}
    T["qptr"] = qptr
    T["nq"] = np.array([nq])
    T["npart"] = np.array([npart])
    T["parity"] = np.array([par])
    T["work"] = np.array([work["edges"] / nq, work["cands"] / nq, work["visited"] / nq,
                          work["admitted"] / nq, 1000 * (time.time() - t0) / nq])
    if fp:
        np.savez_compressed(fp, **T)
    log(f"[{ds}] frozen replay parity {par}/{nq}   rows {len(T['part']):,}   "
        f"{time.time()-t0:.0f}s")
    return S, T


# --------------------------------------------------------------- S4 over an arbitrary support col
def s4_order(part, first_jj, support, min_hop, max_sdir):
    """RT.order_struct(..., 'S4') verbatim, vectorised, with `support` in place of the node count.

    S4 is an equal-weight RRF over four sub-rankings, ties always broken by partition id:
        first arrival index  asc
        support              desc
        minimum hop          asc
        best node score      desc
    Substituting a normalised support changes ONE of the four channels and nothing else."""
    m = len(part)
    rk = np.zeros(m)
    for key, sign in ((first_jj.astype(np.float64), +1.0), (support.astype(np.float64), -1.0),
                      (min_hop.astype(np.float64), +1.0), (max_sdir.astype(np.float64), -1.0)):
        o = np.lexsort((part, sign * key))
        r = np.empty(m, np.int64); r[o] = np.arange(m)
        rk += 1.0 / (K0 + r)
    return part[np.lexsort((part, -rk))]


def rows(T, qi):
    a, b = int(T["qptr"][qi]), int(T["qptr"][qi + 1])
    return {c: T[c][a:b] for c in COLS}
