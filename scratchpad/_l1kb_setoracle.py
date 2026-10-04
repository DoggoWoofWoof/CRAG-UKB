"""STEP 2 (analysis) + STEP 3 -- PATH PROVENANCE ANALYSIS AND THE SET-COMPLETION ORACLE.

A structural node carries a provenance chain

    seed -> a -> b -> node        (nodes)
    P_seed -> P_a -> P_b -> P_node (canonical partitions, deduplicated, order preserved)

A PATH GROUP is the set of partitions on one chain that are not already in the protected core;
promoting a group means paying a slot for EVERY partition in it.  That atomicity is the whole
question: if the needed partitions of a multi-hop query lie on one coherent chain, joint
selection is representable; if group atomicity blows the budget, it is not.

Three ceilings, all at exactly 50 partitions out:

  A INDEPENDENT_ORACLE  need <= bnd u STRUCT u RET,        |need| <= B     (the frozen ceiling)
  B PATH_SET_ORACLE     admitted set must be a union of WHOLE path groups (+ retained boundary)
  C COMBINED_SET_ORACLE path groups u individual retrieval-continuation partitions

Exact minimum-cost cover by branch and bound (branch on the groups containing the first
uncovered needed partition); gold is used only to define `need`, i.e. only to compute a ceiling.

  python scratchpad/_l1kb_setoracle.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ps_router as RT
import _l1ps_oracle as OR
import _l1kb_core as KB

B_GRID = [1, 2, 4, 6]
M_STRUCT_GROUPS = 64          # the router's structural node budget
OUT = f"{KB.KBD}/oracle/set_completion.json"
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def chains(z, PZ, qi, hard, M):
    """ordered partition chains for the first M structural nodes of one query.
    Returns list of (tuple_of_partitions, terminal_partition, depth, node_rank j)."""
    sn = z["s_node"][qi]; pr = PZ["p_root"][qi]; pa = PZ["p_a"][qi]; pb = PZ["p_b"][qi]
    pl = PZ["p_len"][qi]
    out = []
    for j in range(min(len(sn), M)):
        v = int(sn[j])
        if v < 0:
            break
        seq = [int(pr[j]), int(pa[j]), int(pb[j]), v]
        ps = []
        for x in seq:
            if x < 0:
                continue
            p = int(hard[x])
            if p < 0:
                continue
            if not ps or ps[-1] != p:
                ps.append(p)
        if not ps:
            continue
        out.append((tuple(ps), int(hard[v]), int(pl[j]), j))
    return out


def min_cover(need, groups, B):
    """exact minimum |X| such that need <= X, X a union of whole groups (restricted to the
    payable partitions) -- branch and bound on the first uncovered element.  Returns None if no
    cover of cost <= B exists."""
    need = frozenset(need)
    idx = {}
    for gi, g in enumerate(groups):
        for p in g:
            idx.setdefault(p, []).append(gi)
    if not need <= set(idx):
        return None
    best = [B + 1]

    def rec(cov, cost):
        if cost >= best[0]:
            return
        rem = need - cov
        if not rem:
            best[0] = cost
            return
        p = min(rem)
        for gi in idx[p]:
            g = groups[gi]
            nc = cov | g
            rec(nc, len(nc))
    rec(frozenset(), 0)
    return best[0] if best[0] <= B else None


def main():
    RES = {}
    for ds in KB.DSETS:
        z, meta = KB.load(ds)
        PZ = KB.load_paths(ds)
        assert PZ is not None, f"{ds}: path cache missing"
        nq = meta["n_dev_queries"]; hard = z["hard"]; base_rank = z["base_rank"]
        hops = z["hops"]
        goldp = KB.goldparts(z, meta)
        base50 = [set(int(x) for x in base_rank[qi][:KB.P]) for qi in range(nq)]
        ind_base = np.array([int(goldp[qi] <= base50[qi]) for qi in range(nq)], np.int8)
        BASE_ALL = float(ind_base.mean())
        assert abs(round(BASE_ALL, 4) - meta["BASE_ALL_P50"]) < 1e-9

        CH = [chains(z, PZ, qi, hard, M_STRUCT_GROUPS) for qi in range(nq)]
        st32, rt32, st64 = [], [], []
        for qi in range(nq):
            s, r = OR.challengers(z, qi, hard, base50[qi], 32, 32)
            st32.append(s); rt32.append(r)
            s64, _ = OR.challengers(z, qi, hard, base50[qi], M_STRUCT_GROUPS, 32)
            st64.append(s64)
        # every partition appearing anywhere on a cached chain, treated as INDIVIDUALLY selectable
        chainu = [set(p for ps, *_ in CH[qi] for p in ps) - base50[qi] for qi in range(nq)]

        # ---- path provenance shape (STEP 2 analysis)
        depth = [c[2] for q in CH for c in q]
        nparts = [len(c[0]) for q in CH for c in q]
        prov = {"mean_chains_per_query": round(float(np.mean([len(q) for q in CH])), 2),
                "chain_depth_hist": {str(k): int(v) for k, v in
                                     zip(*np.unique(depth, return_counts=True))},
                "distinct_partitions_per_chain_hist": {str(k): int(v) for k, v in
                                                       zip(*np.unique(nparts, return_counts=True))},
                "mean_distinct_partitions_per_chain": round(float(np.mean(nparts)), 3),
                "mean_distinct_chain_partition_sets_per_query": round(float(np.mean(
                    [len({c[0] for c in q}) for q in CH])), 2)}

        cell = {}
        for B in B_GRID:
            prot = [set(int(x) for x in base_rank[qi][:KB.P - B]) for qi in range(nq)]
            bnd = [set(int(x) for x in base_rank[qi][KB.P - B:KB.P]) for qi in range(nq)]
            need = [goldp[qi] - prot[qi] for qi in range(nq)]
            okA = np.zeros(nq, np.int8); okB = np.zeros(nq, np.int8); okC = np.zeros(nq, np.int8)
            okA64 = np.zeros(nq, np.int8); okD = np.zeros(nq, np.int8)
            need_on_one_chain = 0; n_unc = 0
            for qi in range(nq):
                nd = need[qi]
                if len(nd) <= B and nd <= (bnd[qi] | st32[qi] | rt32[qi]):
                    okA[qi] = 1
                # control 1: same INDEPENDENT rule, wider structural node budget (M=64)
                if len(nd) <= B and nd <= (bnd[qi] | st64[qi] | rt32[qi]):
                    okA64[qi] = 1
                # control 2: chain partitions as INDIVIDUALLY selectable items -- isolates the
                # reach contributed by path intermediates from the cost of group atomicity
                if len(nd) <= B and nd <= (bnd[qi] | chainu[qi] | rt32[qi]):
                    okD[qi] = 1
                # payable view: a partition already in prot costs nothing; bnd/outside cost 1
                gs = []
                seen = set()
                for ps, term, dep, j in CH[qi]:
                    g = frozenset(p for p in ps if p not in prot[qi])
                    if g and g not in seen:
                        seen.add(g); gs.append(g)
                singles_b = [frozenset([p]) for p in bnd[qi]]
                gsB = gs + singles_b
                cB = min_cover(nd, gsB, B) if nd else 0
                okB[qi] = int(nd == set() or cB is not None)
                gsC = gsB + [frozenset([p]) for p in rt32[qi]]
                cC = min_cover(nd, gsC, B) if nd else 0
                okC[qi] = int(nd == set() or cC is not None)
                if not ind_base[qi]:
                    n_unc += 1
                    miss = goldp[qi] - base50[qi]
                    if any(miss <= set(ps) for ps, *_ in CH[qi]):
                        need_on_one_chain += 1
            for nm, ok in (("A_INDEPENDENT", okA), ("A64_INDEPENDENT", okA64),
                           ("D_CHAIN_UNION_INDEPENDENT", okD),
                           ("B_PATH_SET", okB), ("C_COMBINED_SET", okC)):
                assert ok.mean() + 1e-12 >= BASE_ALL, f"{ds} B={B} {nm} below BASE"
            e = {"A_INDEPENDENT": round(float(okA.mean()), 4),
                 "A64_INDEPENDENT": round(float(okA64.mean()), 4),
                 "D_CHAIN_UNION_INDEPENDENT": round(float(okD.mean()), 4),
                 "B_PATH_SET": round(float(okB.mean()), 4),
                 "C_COMBINED_SET": round(float(okC.mean()), 4),
                 "frac_uncovered_whose_missing_set_lies_on_ONE_chain":
                     round(need_on_one_chain / max(n_unc, 1), 4)}
            if (hops >= 0).any():
                e["per_hop"] = {str(h): {"A": round(float(okA[hops == h].mean()), 4),
                                         "A64": round(float(okA64[hops == h].mean()), 4),
                                         "D": round(float(okD[hops == h].mean()), 4),
                                         "B": round(float(okB[hops == h].mean()), 4),
                                         "C": round(float(okC[hops == h].mean()), 4)}
                                for h in sorted(set(int(x) for x in hops if x >= 0))}
            cell[str(B)] = e
        RES[ds] = {"n": nq, "BASE_ALL": round(BASE_ALL, 4), "PATH_PROVENANCE": prov, "B": cell}
        log(f"{ds:15s} " + "  ".join(
            f"B{b}: A {cell[str(b)]['A_INDEPENDENT']:.4f} P {cell[str(b)]['B_PATH_SET']:.4f} "
            f"C {cell[str(b)]['C_COMBINED_SET']:.4f}" for b in B_GRID))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"B_GRID": B_GRID, "M_STRUCT_GROUPS": M_STRUCT_GROUPS, "RESULTS": RES},
              open(OUT, "w"), indent=1)
    log(f"wrote {OUT}")
    if "metaqa" in RES and "per_hop" in RES["metaqa"]["B"]["6"]:
        print("\n=== MetaQA set-completion oracle per hop ===")
        print(f"{'B':>3s} {'':6s}" + "".join(f"{'hop'+h:>10s}" for h in ("1", "2", "3")))
        for b in B_GRID:
            ph = RES["metaqa"]["B"][str(b)]["per_hop"]
            for tag in ("A", "A64", "D", "B", "C"):
                print(f"{b:3d} {tag:6s}" + "".join(f"{ph[h][tag]:10.4f}" for h in ("1", "2", "3")))


if __name__ == "__main__":
    main()
