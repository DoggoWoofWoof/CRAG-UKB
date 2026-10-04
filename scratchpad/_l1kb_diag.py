"""STEP 1 + STEP 14 -- HOP-STRATIFIED DIAGNOSIS BEFORE ANY NEW ROUTING.

Answers, per corpus and (for MetaQA) per hop:

  BASE / current B6 router / direct STRUCT32 / STRUCT64 / COMBINED / fixed-P50 oracle B={1,2,4,6}
  uncovered-query counts, |need| distribution, canonical rank of the missing gold partitions,
  STRUCT reach, RET reach, COMBINED-reaches-ALL-simultaneously, minimum replacement count,

and decomposes the residual of the CURRENT router into the four actionable failure modes:

  CAPACITY   |need| > B                       -- no fixed-P50 router at this B can fix it
  REACH      some needed partition is proposed by no channel
  EVICTION   oracle-recoverable, but the router threw a gold partition out of the boundary
  CO_SELECT  oracle-recoverable, router admitted a nonempty STRICT subset of what was needed
  RANKING    oracle-recoverable, router admitted none of the needed partitions

WebQSP has no hop labels; it is bucketed post-hoc (analysis only, never an inference feature) by
|missing gold partitions| and by shortest graph distance from the retrieval seeds to the missing
gold partition.

  python scratchpad/_l1kb_diag.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT
import _l1ps_oracle as OR
import _l1kb_core as KB
import _l1kb_router as JR

B_GRID = [1, 2, 4, 6]
OUT = f"{KB.KBD}/diag/hop_diagnosis.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def bucket_label(n):
    return "0" if n == 0 else ("1" if n == 1 else ("2" if n == 2 else "3+"))


def main():
    SB4 = json.load(open(f"{KB.ROOT}/scoreboard_round4.json"))["SCOREBOARD"]
    RES = {}
    for ds in KB.DSETS:
        z, meta = KB.load(ds)
        nq = meta["n_dev_queries"]; hard = z["hard"]; base_rank = z["base_rank"]
        hops = z["hops"]; s_node = z["s_node"]; s_hop = z["s_hop"]
        goldp = KB.goldparts(z, meta)
        C, _GRP = KB.substrate(ds, z, meta, JR.build_groups)
        base50 = C["base50"]; ind_base = C["ind_base"]
        BASE_ALL = float(ind_base.mean())
        assert abs(round(BASE_ALL, 4) - meta["BASE_ALL_P50"]) < 1e-9

        # ---- current safe universal router, replayed two independent ways
        ref = RT.evaluate(ds, z, meta, dict(KB.BASE_CFG), C)
        ref.pop("_ind")
        sb = SB4[ds]["configs"]["B6_S4_F6_Ms64_Mr32"]
        assert ref["ALL"] == sb["ALL"] and ref["net"] == sb["net"], f"{ds}: scoreboard drift"
        ctx6 = KB.contexts(z, meta, C, 6)
        r6 = KB.run_selector(ctx6, KB.sel_f6, goldp, ind_base)
        assert abs(r6["ALL"] - ref["ALL"]) < 5e-5, f"{ds}: F6 replay {r6['ALL']} vs {ref['ALL']}"
        b0 = KB.run_selector(ctx6, KB.sel_base, goldp, ind_base)
        assert abs(b0["ALL"] - BASE_ALL) < 1e-12 and b0["churn"].sum() == 0

        # ---- challenger pools: the frozen oracle pool (M32/M32) and the FULL reach pool
        st32, rt32, stF, rtF = [], [], [], []
        for qi in range(nq):
            s, r = OR.challengers(z, qi, hard, base50[qi], 32, 32)
            st32.append(s); rt32.append(r)
            sf = set()
            for v in s_node[qi]:
                if v < 0:
                    break
                p = int(hard[v])
                if p >= 0 and p not in base50[qi]:
                    sf.add(p)
            rf = set()
            for v in z["ret_rrf"][qi]:
                if v < 0:
                    break
                p = int(hard[v])
                if p >= 0 and p not in base50[qi]:
                    rf.add(p)
            stF.append(sf); rtF.append(rf)

        # ---- fixed-P50 oracle, per B, with per-hop / per-bucket resolution
        nmiss = np.array([len(goldp[qi] - base50[qi]) for qi in range(nq)], np.int32)
        # shortest graph distance from the retrieval seeds to a missing gold partition
        gdist = np.full(nq, -1, np.int8)
        for qi in range(nq):
            miss = goldp[qi] - base50[qi]
            if not miss:
                continue
            best = 99
            for jj, v in enumerate(s_node[qi]):
                if v < 0:
                    break
                if int(hard[v]) in miss:
                    best = min(best, int(s_hop[qi][jj]))
            gdist[qi] = best if best < 99 else 0     # 0 == not reached by structure at all
        ora = {}
        for B in B_GRID:
            prot = [set(int(x) for x in base_rank[qi][:KB.P - B]) for qi in range(nq)]
            bnd = [set(int(x) for x in base_rank[qi][KB.P - B:KB.P]) for qi in range(nq)]
            need = [goldp[qi] - prot[qi] for qi in range(nq)]
            ok = np.zeros(nq, np.int8)
            for qi in range(nq):
                pool = bnd[qi] | st32[qi] | rt32[qi]
                ok[qi] = int(len(need[qi]) <= B and need[qi] <= pool)
            assert ok.mean() + 1e-12 >= BASE_ALL
            ora[str(B)] = {"ALL": round(float(ok.mean()), 4),
                           "dALL": round(float(ok.mean()) - BASE_ALL, 4),
                           "ind": ok}

        # ---- STEP 14 failure decomposition of the CURRENT router (B=6)
        B = 6
        prot6 = [set(int(x) for x in base_rank[qi][:KB.P - B]) for qi in range(nq)]
        need6 = [goldp[qi] - prot6[qi] for qi in range(nq)]
        cls = np.zeros(nq, np.int8)          # 0 neutral 1 recovered 2 damaged 3 partial-uncov
        why = []
        for qi in range(nq):
            fs = r6["finals"][qi]
            cov, cb = int(r6["ind"][qi]), int(ind_base[qi])
            if cov and not cb:
                cls[qi] = 1
            elif cb and not cov:
                cls[qi] = 2
            elif not cov:
                cls[qi] = 3 if len(goldp[qi] & fs) > len(goldp[qi] & base50[qi]) else 0
            if cov:
                why.append("COVERED"); continue
            nd = need6[qi]
            pool = set(ctx6[qi]["bnd"]) | set(ctx6[qi]["chal"])
            poolF = set(ctx6[qi]["bnd"]) | stF[qi] | rtF[qi]
            if len(nd) > B:
                why.append("CAPACITY")
            elif not nd <= poolF:
                why.append("REACH")
            elif not nd <= pool:
                why.append("REACH_AT_ROUTER_M")
            elif goldp[qi] & (base50[qi] - fs):
                why.append("EVICTION")
            else:
                got = nd & fs
                why.append("CO_SELECT" if (0 < len(got) < len(nd)) else
                           "RANKING" if len(got) == 0 else "OTHER")
        why = np.array(why)

        def slice_stats(mask):
            m = np.asarray(mask)
            n = int(m.sum())
            if n == 0:
                return None
            unc = m & (ind_base == 0)
            miss_sizes = nmiss[unc]
            rk = []
            for qi in np.where(unc)[0]:
                for p in (goldp[qi] - base50[qi]):
                    rk.append(ctx6[qi]["cpos"].get(p, 10 ** 6))
            rk = np.array(rk) if rk else np.array([10 ** 6])
            reach_s = np.mean([(goldp[qi] - base50[qi]) <= stF[qi] for qi in np.where(unc)[0]]) \
                if unc.any() else None
            reach_r = np.mean([(goldp[qi] - base50[qi]) <= rtF[qi] for qi in np.where(unc)[0]]) \
                if unc.any() else None
            reach_c = np.mean([(goldp[qi] - base50[qi]) <= (stF[qi] | rtF[qi])
                               for qi in np.where(unc)[0]]) if unc.any() else None
            d = {"n": n,
                 "BASE": round(float(ind_base[m].mean()), 4),
                 "B6": round(float(r6["ind"][m].mean()), 4),
                 "ORACLE": {b: round(float(ora[b]["ind"][m].mean()), 4) for b in ora},
                 "n_uncovered_BASE": int(unc.sum()),
                 "mean_missing_gold_parts": round(float(miss_sizes.mean()), 3)
                 if len(miss_sizes) else 0.0,
                 "missing_count_hist": {k: int(v) for k, v in
                                        zip(*np.unique([bucket_label(x) for x in miss_sizes],
                                                       return_counts=True))} if len(miss_sizes) else {},
                 "missing_part_canonical_rank": {
                     "median": int(np.median(rk)), "frac_within_top200": round(float((rk < 200).mean()), 4),
                     "frac_beyond_top200": round(float((rk >= 200).mean()), 4)},
                 "STRUCT_reaches_all_missing": round(float(reach_s), 4) if reach_s is not None else None,
                 "RET_reaches_all_missing": round(float(reach_r), 4) if reach_r is not None else None,
                 "COMBINED_reaches_all_missing": round(float(reach_c), 4) if reach_c is not None else None,
                 "min_replacements_needed_mean": round(float(np.mean([len(need6[qi])
                                                                     for qi in np.where(unc)[0]])), 3)
                 if unc.any() else None,
                 "router_outcome": {["NEUTRAL", "RECOVERED", "DAMAGED", "PARTIAL"][int(k)]: int(v)
                                    for k, v in zip(*np.unique(cls[m], return_counts=True))},
                 "residual_failure_modes": {k: int(v) for k, v in
                                            zip(*np.unique(why[m & (r6["ind"] == 0)],
                                                           return_counts=True))}}
            return d

        rec = {"n": nq, "BASE_ALL": round(BASE_ALL, 4), "BASE_ANY": meta["BASE_ANY_P50"],
               "B6_ALL": round(r6["ALL"], 4), "B6_ANY": round(r6["ANY"], 4),
               "B6_gold_admitted": r6["gold_admitted"], "B6_gold_evicted": r6["gold_evicted"],
               "B6_churn_mean": round(float(r6["churn"].mean()), 3),
               "ORACLE": {b: {"ALL": ora[b]["ALL"], "dALL": ora[b]["dALL"]} for b in ora},
               "ALL_QUERIES": slice_stats(np.ones(nq, bool))}
        cj = json.load(open(f"results/GENERALIZATION/_g2_comb_{ds}.json"))["RESULTS"]
        rec["direct_node"] = {k: {"ALL": cj[k]["ALL"],
                                  "per_hop": cj[k].get("per_hop")}
                              for k in ("STRUCT32", "STRUCT64", "RET32", "COMBINED_32_32")}
        if (hops >= 0).any():
            rec["PER_HOP"] = {str(h): slice_stats(hops == h)
                              for h in sorted(set(int(x) for x in hops if x >= 0))}
            rec["PER_HOP_paired_B6_vs_BASE"] = KB.paired_hop(r6["ind"], ind_base, hops)
        rec["PER_MISSING_COUNT"] = {bucket_label(k): slice_stats(
            (nmiss == k) if k < 3 else (nmiss >= 3)) for k in (0, 1, 2, 3)}
        rec["PER_SEED_GRAPH_DISTANCE"] = {
            ("unreached" if d == 0 else str(d)): slice_stats(gdist == d)
            for d in sorted(set(int(x) for x in gdist if x >= 0))}
        RES[ds] = rec
        log(f"{ds:15s} BASE {BASE_ALL:.4f} B6 {r6['ALL']:.4f} "
            f"oracle B1 {ora['1']['ALL']:.4f} B6 {ora['6']['ALL']:.4f} | "
            f"modes " + " ".join(f"{k}:{v}" for k, v in
                                 sorted(rec["ALL_QUERIES"]["residual_failure_modes"].items())))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"B_GRID": B_GRID, "ROUTER": KB.BASE_CFG, "RESULTS": RES}, open(OUT, "w"), indent=1)
    log(f"wrote {OUT}")

    print("\n=== MetaQA per hop (mandatory STEP-12 table) ===")
    m = RES["metaqa"]
    rows = [("BASE", {h: m["PER_HOP"][h]["BASE"] for h in m["PER_HOP"]}),
            ("current B6", {h: m["PER_HOP"][h]["B6"] for h in m["PER_HOP"]})]
    for k in ("STRUCT32", "STRUCT64", "COMBINED_32_32"):
        ph = m["direct_node"][k]["per_hop"]
        rows.append((f"direct {k}", {h: ph[h]["ALL"] for h in ph}))
    for b in B_GRID:
        rows.append((f"oracle B{b}", {h: m["PER_HOP"][h]["ORACLE"][str(b)] for h in m["PER_HOP"]}))
    print(f"{'':22s} " + " ".join(f"{'hop'+h:>9s}" for h in sorted(m["PER_HOP"])))
    for nm, d in rows:
        print(f"{nm:22s} " + " ".join(f"{d[h]:9.4f}" for h in sorted(d)))


if __name__ == "__main__":
    main()
