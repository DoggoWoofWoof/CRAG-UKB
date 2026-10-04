"""STEPS 8/9/11 -- R0..R6 on all six corpora, MetaQA per hop, and the latency contract.

Every rule is measured against BOTH references, as required: BASE (no swap at all) and SAFE_F6
(= R0, verified bit-for-bit identical to the frozen selector).  Gold is used only to score.

  python scratchpad/_l1cal_main.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1cal_core as CC
import _l1pp_core as PP

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
B = 6


def oracle(ctxs, goldp, Bv):
    ind = np.zeros(len(ctxs), np.int8)
    for qi, c in enumerate(ctxs):
        miss = goldp[qi] - c["prot_set"]
        ind[qi] = int(len(goldp[qi]) <= CC.P and len(miss) <= Bv
                      and miss <= (set(c["bnd"]) | set(c["chal"])))
    return ind


def run_ds(ds, OUT):
    S = CC.substrate(ds, B)
    ctxs, KS, goldp, nq = S["ctxs"], S["KS"], S["goldp"], S["nq"]
    hops = S["hops"]
    ib = S["ind_base"]
    res, inds = {}, {}
    for name in CC.RULES:
        t = time.perf_counter()
        r = CC.apply_rule(ctxs, KS, name, B, goldp, S["base50s"])
        r["sec_per_query"] = round((time.perf_counter() - t) / nq, 8)
        inds[name] = r["ind"]
        res[name] = r
    f6 = inds["R0_RAW_RRF"]
    orc = oracle(ctxs, goldp, B)
    out = {}
    for name in CC.RULES:
        r, ind = res[name], inds[name]
        e = {"ALL": round(r["ALL"], 4),
             "dBASE": round(float(ind.mean() - ib.mean()), 4),
             "dSAFE_F6": round(float(ind.mean() - f6.mean()), 4),
             "vs_BASE": CC.mc(ind, ib), "vs_SAFE_F6": CC.mc(ind, f6),
             "newly_covered_vs_F6": int(((ind == 1) & (f6 == 0)).sum()),
             "newly_uncovered_vs_F6": int(((ind == 0) & (f6 == 1)).sum()),
             "churn_mean": r["churn_mean"], "swaps": r["swaps"],
             "gold_admitted": r["gold_admitted"], "gold_evicted": r["gold_evicted"],
             "GOOD": r["GOOD"], "BAD": r["BAD"], "NEUTRAL": r["NEUTRAL"],
             "GOOD_over_BAD": round(r["GOOD"] / max(1, r["BAD"]), 3),
             "sec_per_query": r["sec_per_query"]}
        if ds == "metaqa":
            e["per_hop"] = {str(h): {
                "n": int((hops == h).sum()),
                "ALL": round(float(ind[hops == h].mean()), 4),
                "dBASE": round(float(ind[hops == h].mean() - ib[hops == h].mean()), 4),
                "dSAFE_F6": round(float(ind[hops == h].mean() - f6[hops == h].mean()), 4),
                "vs_SAFE_F6": CC.mc(ind[hops == h], f6[hops == h])}
                for h in (1, 2, 3) if (hops == h).sum()}
        out[name] = e
        log("%-15s %-26s ALL %.4f  dBASE %+.4f  dF6 %+.4f (p %.4f)  swaps %5d  G/B %5.2f  churn %.2f"
            % (ds[:14], name, e["ALL"], e["dBASE"], e["dSAFE_F6"],
               e["vs_SAFE_F6"]["mcnemar_p"], e["swaps"], e["GOOD_over_BAD"], e["churn_mean"]))
    out["BASE"] = {"ALL": round(float(ib.mean()), 4), "dBASE": 0.0,
                   "dSAFE_F6": round(float(ib.mean() - f6.mean()), 4),
                   "churn_mean": 0.0, "swaps": 0}
    out["ORACLE_B6"] = {"ALL": round(float(orc.mean()), 4),
                        "dBASE": round(float(orc.mean() - ib.mean()), 4),
                        "dSAFE_F6": round(float(orc.mean() - f6.mean()), 4)}
    if ds == "metaqa":
        out["BASE"]["per_hop"] = {str(h): {"ALL": round(float(ib[hops == h].mean()), 4)}
                                  for h in (1, 2, 3) if (hops == h).sum()}
        out["ORACLE_B6"]["per_hop"] = {str(h): {"ALL": round(float(orc[hops == h].mean()), 4)}
                                       for h in (1, 2, 3) if (hops == h).sum()}
    # ---- STEP 11 latency contract
    Lm = np.array([KS[qi]["L"] for qi in range(nq)])
    out["_LATENCY"] = {
        "ONLINE_GRAPH_EDGES_TOUCHED": 0,
        "new_encoder_passes": 0,
        "additional_cached_bytes_per_query": 0,
        "ranks_read_per_query": int(Lm.sum(1).mean()),
        "float_ops_per_query_extra_vs_R0": int(3 * Lm.sum(1).mean()),
        "mean_candidates_scored_per_query": round(
            float(np.mean([len(CC.cands_of(c)) for c in ctxs])), 2)}
    OUT[ds] = out
    ind_np = {k: np.asarray(v, np.int8) for k, v in inds.items()}
    np.savez_compressed(f"{CC.CALD}/diag/ind_{ds}.npz", base=ib, oracle=orc, **ind_np)


def main():
    fp = f"{CC.CALD}/diag/main_R0R6.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in (sys.argv[1:] or CC.DSETS):
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote main_R0R6.json")


if __name__ == "__main__":
    main()
