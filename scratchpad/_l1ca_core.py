"""EXACT-P50 STRUCTURAL CONTRACT AUDIT -- the three nested structural evidence universes.

Design decision that makes the audit clean: the three universes are strict PREFIXES of one order,

    S4_M64            first 64 admitted nodes, in the frozen `added` order   (== frozen, gated)
    S4_FULL_ADMITTED  every admitted node,     in the frozen `added` order
    S4_FULL_VISITED   the above, then every VISITED-but-beam-pruned node, by static score desc

so S4_M64 subset S4_FULL_ADMITTED subset S4_FULL_VISITED and the only thing that varies between
them is how much evidence reaches the aggregation.  Node ORDERING is held fixed, so the comparison
isolates truncation and cannot be confounded by a re-scoring.  The aggregation itself is the frozen
S4 verbatim (`RT.order_struct(..., "S4")` over the same six statistics).

Nothing here re-searches.  The pruned nodes were already scored by the bounded beam; the beam simply
discards them.  `want_visited` on the frozen replay keeps them.

    S4_M64            -> "read truncation" is present
    S4_FULL_ADMITTED  -> read truncation removed, beam truncation still present
    S4_FULL_VISITED   -> both removed, within the SAME bounded search
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _ta_prepartition as TA
import _l1ps_router as RT
import _l1sr_eval as EV
import _l1sr_diag as DG
import _l1bm_core as BM
import _l1bm_run as RUN
import _l1ss_core as SS

CAD = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_CONTRACT_AUDIT"
K0 = EV.K0
UNIVERSES = ["S4_M64", "S4_FULL_ADMITTED", "S4_FULL_VISITED"]
BS = [6, 8, 12, 20, 50]
os.makedirs(f"{CAD}/diag", exist_ok=True)
os.makedirs(f"{CAD}/data", exist_ok=True)


def sf_prefix(node, hop, sdir, cnt, hard, depth):
    """the frozen S4 aggregation over the first `depth` nodes of the nested order.

    Byte-for-byte the body of EV.sf_from_cache, with the node universe passed in instead of read
    off the frozen cache; at depth=64 over the admitted prefix it reproduces the frozen ranking."""
    agg = {}
    for jj in range(min(len(node), depth)):
        v = int(node[jj])
        if v < 0:
            break
        p = int(hard[v])
        if p < 0:
            continue
        a = agg.get(p)
        if a is None:
            agg[p] = [jj, 1, 1.0 / (K0 + jj), int(hop[jj]), float(sdir[jj]), int(cnt[jj])]
        else:
            a[1] += 1
            a[2] += 1.0 / (K0 + jj)
            a[3] = min(a[3], int(hop[jj]))
            a[4] = max(a[4], float(sdir[jj]))
            a[5] += int(cnt[jj])
    return RT.order_struct(agg, "S4"), len(agg)


def universes(ds, log=print, nq_max=None):
    """per query: the nested node order, the split points, and the internal work counters."""
    S = EV.substrate(ds)
    z = S["z"]
    nq = S["nq"] if nq_max is None else min(S["nq"], nq_max)
    adjp, adji, deg = RUN.topology(ds)
    Xn, Qm, _ = RUN.load_emb_frozen(ds, z)
    ndocs = (Xn.A.shape[0] if hasattr(Xn, "A") else Xn.shape[0])
    W = SS.workspace(ndocs)
    fz = {k: np.asarray(z[k]) for k in ("s_node", "s_hop", "s_sdir", "s_cnt")}
    OUT = {"node": [], "hop": [], "sdir": [], "cnt": [], "n_adm": np.zeros(nq, np.int64),
           "n_vis": np.zeros(nq, np.int64)}
    work = {"edges": 0, "cands": 0, "visited": 0, "admitted": 0, "scope": 0}
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
        a_node = FE["node"]
        a_hop = FE["HOP"].astype(np.int64)
        a_sdir = FE["STATIC_SDIR"]
        a_cnt = FE["PATH_SUPPORT"].astype(np.int64)
        OUT["node"].append(np.concatenate([a_node, FE["VIS_PRUNED"]]))
        OUT["hop"].append(np.concatenate([a_hop, FE["VIS_PRUNED_HOP"]]))
        OUT["sdir"].append(np.concatenate([a_sdir, FE["VIS_PRUNED_SDIR"]]))
        OUT["cnt"].append(np.concatenate([a_cnt, FE["VIS_PRUNED_CNT"]]))
        OUT["n_adm"][qi] = len(a_node)
        OUT["n_vis"][qi] = len(a_node) + len(FE["VIS_PRUNED"])
        work["edges"] += st["edges"]
        work["cands"] += st["cand_total"]
        work["visited"] += FE["VIS_TOTAL"]
        work["admitted"] += len(a_node)
        work["scope"] += st["scope"]
        if (qi + 1) % 500 == 0:
            log(f"   {ds} {qi+1}/{nq}  parity {par}/{qi+1}  {time.time()-t0:.0f}s")
    work = {k: round(v / nq, 1) for k, v in work.items()}
    work["PARITY"] = f"{par}/{nq}"
    work["seconds_total"] = round(time.time() - t0, 1)
    work["ms_per_query"] = round(1000 * (time.time() - t0) / nq, 2)
    return S, OUT, work, par == nq
