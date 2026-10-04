"""STEP 7 -- clean cost isolation for the exact-P50 contract audit.

Separates the three cost components, on an otherwise idle CPU:

    search      the bounded structural beam            IDENTICAL across all three universes
    keep        retaining the visited-but-pruned nodes  (arrays only, no re-scoring)
    aggregate   the frozen S4 aggregation at depth 64 vs depth = |visited|

and measures the transient memory of the visited-node arrays.  Downstream exposure is EXACTLY 50
partitions in every case, so none of this is downstream cost.
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
import _l1ca_core as CA

NQ = 400


def run(ds, log=print):
    S = EV.substrate(ds)
    z = S["z"]
    nq = min(S["nq"], NQ)
    adjp, adji, deg = RUN.topology(ds)
    Xn, Qm, _ = RUN.load_emb_frozen(ds, z)
    ndocs = (Xn.A.shape[0] if hasattr(Xn, "A") else Xn.shape[0])
    W = SS.workspace(ndocs)
    hard = S["hard"]
    seeds = [[int(s) for s in z["seeds"][qi] if s >= 0] for qi in range(nq)]

    # warm the caches so the first query does not pay for lazy allocation
    for qi in range(3):
        rq = TA.residual(Qm[qi].astype(np.float64), seeds[qi], Xn)
        SS.expand_feat(seeds[qi], rq, adjp, adji, deg, Xn, W, want_future=False, want_visited=False)

    out = {"ds": ds, "queries": nq}

    t = time.perf_counter()
    for qi in range(nq):
        rq = TA.residual(Qm[qi].astype(np.float64), seeds[qi], Xn)
        SS.expand_feat(seeds[qi], rq, adjp, adji, deg, Xn, W, want_future=False, want_visited=False)
    out["ms_search_frozen"] = round(1000 * (time.perf_counter() - t) / nq, 3)

    t = time.perf_counter()
    keep = []
    for qi in range(nq):
        rq = TA.residual(Qm[qi].astype(np.float64), seeds[qi], Xn)
        added, vmeta, st, FE = SS.expand_feat(seeds[qi], rq, adjp, adji, deg, Xn, W,
                                              want_future=False, want_visited=True)
        keep.append((FE, st, added, vmeta))
    out["ms_search_plus_keep_visited"] = round(1000 * (time.perf_counter() - t) / nq, 3)
    out["ms_keep_overhead"] = round(out["ms_search_plus_keep_visited"] - out["ms_search_frozen"], 3)

    NODE, HOP, SDIR, CNT, NADM, NVIS = [], [], [], [], [], []
    nbytes = 0
    for FE, st, added, vmeta in keep:
        a = np.concatenate([FE["node"], FE["VIS_PRUNED"]])
        NODE.append(a)
        HOP.append(np.concatenate([FE["HOP"].astype(np.int64), FE["VIS_PRUNED_HOP"]]))
        SDIR.append(np.concatenate([FE["STATIC_SDIR"], FE["VIS_PRUNED_SDIR"]]))
        CNT.append(np.concatenate([FE["PATH_SUPPORT"].astype(np.int64), FE["VIS_PRUNED_CNT"]]))
        NADM.append(len(FE["node"]))
        NVIS.append(len(a))
        nbytes += (a.nbytes + HOP[-1].nbytes + SDIR[-1].nbytes + CNT[-1].nbytes)
    out["bytes_per_query_visited_arrays"] = int(round(nbytes / nq))
    out["nodes_admitted_per_query"] = round(float(np.mean(NADM)), 1)
    out["nodes_visited_per_query"] = round(float(np.mean(NVIS)), 1)

    for nm, dep in (("M64", "m64"), ("FULL_ADMITTED", "adm"), ("FULL_VISITED", "vis")):
        t = time.perf_counter()
        npart = []
        for qi in range(nq):
            d = (min(EV.M_STRUCT, NADM[qi]) if dep == "m64" else
                 (NADM[qi] if dep == "adm" else NVIS[qi]))
            _, k = CA.sf_prefix(NODE[qi], HOP[qi], SDIR[qi], CNT[qi], hard, d)
            npart.append(k)
        out[f"ms_aggregate_{nm}"] = round(1000 * (time.perf_counter() - t) / nq, 3)
        out[f"partitions_{nm}"] = round(float(np.mean(npart)), 1)
        out[f"accumulations_{nm}"] = round(float(np.mean(
            [min(EV.M_STRUCT, NADM[qi]) if dep == "m64" else
             (NADM[qi] if dep == "adm" else NVIS[qi]) for qi in range(nq)])), 1)

    out["ms_total_frozen_M64"] = round(out["ms_search_frozen"] + out["ms_aggregate_M64"], 3)
    out["ms_total_FULL_VISITED"] = round(out["ms_search_plus_keep_visited"]
                                         + out["ms_aggregate_FULL_VISITED"], 3)
    out["FINAL_PARTITIONS_DOWNSTREAM"] = EV.P
    log(json.dumps(out))
    return out


if __name__ == "__main__":
    ALL = {}
    for d in (sys.argv[1:] or ["metaqa"]):
        ALL[d] = run(d)
    fp = f"{CA.CAD}/diag/cost_step7.json"
    old = json.load(open(fp)) if os.path.exists(fp) else {}
    old.update(ALL)
    json.dump(old, open(fp, "w"), indent=1)
    print("wrote", fp)
