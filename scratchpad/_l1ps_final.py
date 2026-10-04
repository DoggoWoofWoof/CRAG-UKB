"""STEP 10 -- full-DEV confirmation of ONE universal fixed-P50 configuration.

Single code path: the selection itself comes from _l1ps_router.evaluate (the same function the
search rounds used), so the confirmed numbers cannot drift from the scoreboard.  This adds only
the metrics the scoreboard does not carry: BASE ANY, per-hop BASE, %-of-direct-gain recovered,
%-of-oracle captured, runtime and peak RAM.

  python scratchpad/_l1ps_final.py <B> <agg> <fusion_tag> <M_struct> <M_ret>
      e.g.  python scratchpad/_l1ps_final.py 4 S4 F6 32 32
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT

ROOT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH"
DSETS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
B = int(sys.argv[1]); AGG = sys.argv[2]; TAG = sys.argv[3]
MS = int(sys.argv[4]); MR = int(sys.argv[5])
CFG = dict(B=B, M_struct=MS, M_ret=MR, agg=AGG, fusion=TAG[:2],
           T=(int(TAG[3:]) if len(TAG) > 2 and TAG[2] == "T" else 1), tag=TAG)
NAME = f"B{B}_{AGG}_{TAG}_Ms{MS}_Mr{MR}"


def peak_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 1e9, 2)
    except Exception:
        return None


def main():
    OUT = {"config": dict(CFG, K0=RT.K0, P_MAIN=RT.P, name=NAME,
                          learned_parameters=0, uses_gold_at_inference=False,
                          uses_dataset_identity=False,
                          output="exactly 50 canonical C partitions, no stray nodes"),
           "DATASETS": {}}
    for ds in DSETS:
        t0 = time.time()
        z = np.load(f"{ROOT}/runs/cache_{ds}.npz", allow_pickle=True)
        meta = json.loads(str(z["meta_json"]))
        C = RT.build_cache(ds, z, meta, [MS], [MR], [AGG])
        assert abs(round(C["BASE_ALL"], 4) - meta["BASE_ALL_P50"]) < 1e-9, f"{ds}: BASE parity"
        t1 = time.time()
        r = RT.evaluate(ds, z, meta, CFG, C)
        route_s = time.time() - t1
        ind = r.pop("_ind")
        nq = meta["n_dev_queries"]
        base50, goldp = C["base50"], C["goldp"]
        any_base = float(np.mean([bool(goldp[qi] & base50[qi]) for qi in range(nq)]))

        cj = json.load(open(f"results/GENERALIZATION/_g2_comb_{ds}.json"))
        assert cj["n_dev_queries"] == nq and abs(cj["BASE_ALL_P50"] - meta["BASE_ALL_P50"]) < 1e-9
        d_direct = cj["RESULTS"]["STRUCT32"]["dALL_vs_BASE"]
        d_comb_direct = cj["RESULTS"]["COMBINED_32_32"]["dALL_vs_BASE"]
        oc = json.load(open(f"{ROOT}/oracle/fixed_p50_ceiling.json"))["RESULTS"][ds]
        d_or = oc["B"][str(B)]["COMBINED"]["dALL"] if str(B) in oc["B"] else None
        d = r["dALL"]
        r.update({
            "n": nq, "sample_rule": meta["sample_rule"],
            "BASE_ALL": round(meta["BASE_ALL_P50"], 4), "BASE_ANY": round(any_base, 4),
            "BASE_hard_union_nodes_mean": meta["BASE_SCOPE_NODES"],
            "scope_growth_pct": round(100.0 * (r["final_scope_nodes_mean"]
                                               / meta["BASE_SCOPE_NODES"] - 1.0), 3),
            "direct_STRUCT_NODE32_dALL": d_direct,
            "direct_COMBINED_NODE_32_32_dALL": d_comb_direct,
            "RECOVERED_STRUCTURAL_GAIN_FRACTION": (round(d / d_direct, 3)
                                                   if d_direct > 1e-9 else None),
            "recovered_of_direct_COMBINED": (round(d / d_comb_direct, 3)
                                             if d_comb_direct > 1e-9 else None),
            "oracle_dALL_at_this_B": d_or,
            "fraction_of_oracle_captured": (round(d / d_or, 3) if d_or and d_or > 1e-9 else None),
            "cache_build_sec": round(t1 - t0, 1),
            "route_sec_total": round(route_s, 2),
            "route_ms_per_query": round(1e3 * route_s / nq, 3),
            "peak_rss_gb": peak_gb(),
            "struct_traversal_sec": meta["COST"]["STRUCT_sec"],
            "struct_edges_traversed": meta["COST"]["STRUCT_edges"],
            "struct_x_BASE": meta["COST"]["STRUCT_x_BASE"]})
        OUT["DATASETS"][ds] = r
        print(f"[{ds}] ALL {r['BASE_ALL']:.4f} -> {r['ALL']:.4f} ({d:+.4f}, net {r['net']:+}, "
              f"p={r['mcnemar_p']}) ANY {any_base:.4f} -> {r['ANY']:.4f} | churn "
              f"{r['churn_per_query']:.2f} | gold a{r['gold_parts_admitted']}/"
              f"e{r['gold_parts_evicted']} | src {r['decisive_channel_for_newly_covered']}",
              flush=True)

    D = OUT["DATASETS"]
    OUT["UNIVERSAL"] = {
        "worst_dALL": round(min(v["dALL"] for v in D.values()), 4),
        "macro_dALL": round(float(np.mean([v["dALL"] for v in D.values()])), 4),
        "n_sig_regressions": sum(1 for v in D.values() if v["sig"] and v["dALL"] < 0),
        "n_positive": sum(1 for v in D.values() if v["dALL"] > 0),
        "n_sig_improvements": sum(1 for v in D.values() if v["sig"] and v["dALL"] > 0),
        "mean_churn": round(float(np.mean([v["churn_per_query"] for v in D.values()])), 3),
        "total_net_queries": int(sum(v["net"] for v in D.values())),
        "macro_recovered_structural_gain_fraction": round(float(np.mean(
            [v["RECOVERED_STRUCTURAL_GAIN_FRACTION"] for v in D.values()
             if v["RECOVERED_STRUCTURAL_GAIN_FRACTION"] is not None])), 3)}
    fp = f"{ROOT}/runs/final_{NAME}.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    print(f"\nUNIVERSAL {NAME}: {json.dumps(OUT['UNIVERSAL'])}\nwrote {fp}")


if __name__ == "__main__":
    main()
