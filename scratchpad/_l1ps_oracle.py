"""STEP 2 -- FIXED-P50 ORACLE CEILING.  Evaluation only; gold is used ONLY to compute the ceiling.

Architecture under test (exactly 50 partitions out, no stray nodes):

    base_rank[:50-B]                protected core, can never change
    base_rank[50-B:50]              boundary, the only B slots that may be replaced
    challengers                     out-of-top-50 partitions proposed by structure / retrieval

A query is ORACLE-ALL-covered at budget B iff

    need = gold_partitions \\ protected          (what the core does not already contain)
    |need| <= B                                  capacity
    need subset of (boundary u challengers)      reachability

because an oracle may fill the B free slots with exactly `need` (and keeping the original boundary
is always available, so ORACLE >= BASE by construction -- asserted).

Challenger families:
    STRUCT      partitions of the top M_struct structural residual nodes, minus base top-50
    RET         partitions of the next M_ret retrieval-continuation nodes, minus base top-50
    COMBINED    union of the two
    UNBOUNDED   every partition (isolates pure capacity: is B the binding constraint, or reach?)

  python scratchpad/_l1ps_oracle.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np

ROOT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH"
DSETS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
B_GRID = [1, 2, 4, 6, 8, 12]   # 6 added: the selected universal budget
M_STRUCT = 32
M_RET = 32
P = 50


def load(ds):
    p = f"{ROOT}/runs/cache_{ds}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return z, json.loads(str(z["meta_json"]))


def challengers(z, qi, hard, base50, M_struct, M_ret):
    """partitions proposed by each family, always excluding anything already in the base top-50."""
    st, rt = [], []
    sn = z["s_node"][qi]
    # node-budget (not partition-budget) semantics: the first M_struct out-of-P50 NODES, so the
    # challenger pool is derived from exactly the STRUCT32 node set already validated.
    got = 0
    for v in sn:
        if v < 0:
            break
        p = int(hard[v])
        if p < 0 or p in base50:
            continue
        got += 1
        if p not in st:
            st.append(p)
        if got >= M_struct:
            break
    got = 0
    for v in z["ret_rrf"][qi]:
        if v < 0:
            break
        p = int(hard[v])
        if p < 0 or p in base50:
            continue
        got += 1
        if p not in rt:
            rt.append(p)
        if got >= M_ret:
            break
    return set(st), set(rt)


def main():
    OUT = {}
    for ds in DSETS:
        L = load(ds)
        if L is None:
            print(f"[skip] {ds}: cache missing"); continue
        z, meta = L
        hard = z["hard"]; base_rank = z["base_rank"]; gp = z["gold_part"]; gptr = z["gold_ptr"]
        gc = z["gold_cnt"]; part_sizes = z["part_sizes"]; hops = z["hops"]
        nq = meta["n_dev_queries"]
        BASE_ALL = meta["BASE_ALL_P50"]; BASE_ANY = meta["BASE_ANY_P50"]

        base50 = [set(int(x) for x in base_rank[qi][:P]) for qi in range(nq)]
        goldp = [set(int(x) for x in gp[gptr[qi]:gptr[qi + 1]]) for qi in range(nq)]
        ind_base = np.array([int(goldp[qi] <= base50[qi]) for qi in range(nq)], np.int8)
        assert abs(round(float(ind_base.mean()), 4) - BASE_ALL) < 1e-9, \
            f"{ds}: BASE ALL replay mismatch ({ind_base.mean():.6f} vs {BASE_ALL})"
        BASE_ALL = float(ind_base.mean())          # exact, not the 4dp manifest value

        # ---- direct-node diagnostics (STEP 11 denominators). Taken from the already-measured
        # _g2_comb_{ds}.json, whose BASE is parity-checked against this cache, rather than replayed
        # here: node-level coverage needs gold NODE rows, and the cache stores gold PARTITIONS.
        cp = f"results/GENERALIZATION/_g2_comb_{ds}.json"
        CJ = json.load(open(cp))
        assert abs(CJ["BASE_ALL_P50"] - round(BASE_ALL, 4)) < 1e-9 and CJ["n_dev_queries"] == nq, \
            f"{ds}: direct-node diagnostic came from a different BASE/sample"
        diag = {f"STRUCT_NODE{M}": CJ["RESULTS"][f"STRUCT{M}"]["ALL"] for M in (32, 64)}
        diag["RET32"] = CJ["RESULTS"]["RET32"]["ALL"]
        diag["COMBINED_NODE_32_32"] = CJ["RESULTS"]["COMBINED_32_32"]["ALL"]

        st_all, rt_all = [], []
        for qi in range(nq):
            s, r = challengers(z, qi, hard, base50[qi], M_STRUCT, M_RET)
            st_all.append(s); rt_all.append(r)

        rec = {"n": nq, "BASE_ALL": BASE_ALL, "BASE_ANY": BASE_ANY,
               "direct_node_diagnostic": diag,
               "challenger_partitions_mean": {
                   "STRUCT": round(float(np.mean([len(s) for s in st_all])), 2),
                   "RET": round(float(np.mean([len(s) for s in rt_all])), 2),
                   "COMBINED": round(float(np.mean([len(st_all[i] | rt_all[i]) for i in range(nq)])), 2),
                   "overlap": round(float(np.mean([len(st_all[i] & rt_all[i]) for i in range(nq)])), 2)},
               "B": {}}

        for B in B_GRID:
            prot = [set(int(x) for x in base_rank[qi][:P - B]) for qi in range(nq)]
            bnd = [set(int(x) for x in base_rank[qi][P - B:P]) for qi in range(nq)]
            need = [goldp[qi] - prot[qi] for qi in range(nq)]
            cell = {"mean_need": round(float(np.mean([len(x) for x in need])), 3),
                    "frac_need_le_B": round(float(np.mean([len(x) <= B for x in need])), 4)}
            for fam in ("STRUCT", "RET", "COMBINED", "UNBOUNDED"):
                ok = np.zeros(nq, np.int8); scope = np.zeros(nq, np.int64)
                for qi in range(nq):
                    nd = need[qi]
                    if len(nd) > B:
                        cand_ok = False
                    elif fam == "UNBOUNDED":
                        cand_ok = True
                    else:
                        pool = bnd[qi] | (st_all[qi] if fam in ("STRUCT", "COMBINED") else set()) \
                               | (rt_all[qi] if fam in ("RET", "COMBINED") else set())
                        cand_ok = nd <= pool
                    ok[qi] = int(cand_ok)
                    if cand_ok:
                        fill = list(nd)[:B]
                        rest = [x for x in sorted(bnd[qi]) if x not in nd][:B - len(fill)]
                        scope[qi] = int(part_sizes[sorted(prot[qi] | set(fill) | set(rest))].sum())
                assert ok.mean() + 1e-12 >= BASE_ALL, f"{ds} B={B} {fam}: oracle below BASE"
                d = round(float(ok.mean()) - BASE_ALL, 4)
                den = diag["STRUCT_NODE32"] - BASE_ALL
                cell[fam] = {"ALL": round(float(ok.mean()), 4), "dALL": d,
                             "newly_covered": int(((ok == 1) & (ind_base == 0)).sum()),
                             "frac_of_STRUCT_NODE32_gain": (round(d / den, 3) if den > 1e-9 else None),
                             "final_scope_nodes_mean": round(float(scope[ok == 1].mean()), 1)
                             if ok.any() else None}
            if (hops >= 0).any():
                cell["per_hop_COMBINED"] = {}
                for h in sorted(set(int(x) for x in hops if x >= 0)):
                    m = hops == h
                    okh = np.zeros(int(m.sum()), np.int8); k = 0
                    for qi in np.where(m)[0]:
                        nd = need[qi]
                        pool = bnd[qi] | st_all[qi] | rt_all[qi]
                        okh[k] = int(len(nd) <= B and nd <= pool); k += 1
                    cell["per_hop_COMBINED"][str(h)] = round(float(okh.mean()), 4)
            rec["B"][str(B)] = cell
        OUT[ds] = rec
        print(f"[{ds}] BASE {BASE_ALL:.4f} | direct STRUCT_NODE32 {diag['STRUCT_NODE32']:.4f} | "
              + " ".join(f"B{B}:{rec['B'][str(B)]['COMBINED']['ALL']:.4f}" for B in B_GRID), flush=True)

    os.makedirs(f"{ROOT}/oracle", exist_ok=True)
    json.dump({"B_GRID": B_GRID, "M_STRUCT": M_STRUCT, "M_RET": M_RET, "P": P, "RESULTS": OUT},
              open(f"{ROOT}/oracle/fixed_p50_ceiling.json", "w"), indent=1)
    print("\nwrote " + f"{ROOT}/oracle/fixed_p50_ceiling.json")


if __name__ == "__main__":
    main()
