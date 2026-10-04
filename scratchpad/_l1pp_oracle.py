"""STEP 7 addendum -- separate SELECTOR failure from POOL failure.

The STEP-7 taxonomy in round C calls a query RANKING when every missing gold partition lies
somewhere in the broad universe (canonical top-200 + structural aggregate + retrieval aggregate)
and at most B of them sit outside BASE50.  That over-counts what the router could actually do: a
partition at canonical rank 150 with no structural and no retrieval evidence can never win the RRF
competition, because its score 1/(K0+150) is below every challenger's.

So this recomputes the headroom against the universe the selector really sees -- the protected core
plus the boundary plus the challengers -- and asks the exact combinatorial question:

    ORACLE_B : does there exist X, |X| = B, X subset of (boundary + challengers),
               with gold subset of (protected core + X)?
             <=> |gold \\ prot| <= B  AND  gold \\ prot subset of (boundary + challengers)

ORACLE_B minus ACTUAL is the coverage a perfect parameter-free selector could still win at that B
without any contract change.  Anything above ORACLE_B needs a bigger candidate pool (M_struct,
M_ret, MAX_HOPS) or a bigger P, which IS a contract change.

  python scratchpad/_l1pp_oracle.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1kb_core as KB
import _l1kb_router as JR

OUT = {}
for ds in PP.DSETS:
    z, meta = PP.load(ds); nq = meta["n_dev_queries"]
    goldp = PP.goldparts(z, meta); hops = z["hops"]
    C, GRP = KB.substrate(ds, z, meta, JR.build_groups)
    row = {}
    for Bv in (6, 8, 12):
        ctxs = KB.contexts(z, meta, C, Bv)
        act = np.zeros(nq, np.int8); orc = np.zeros(nq, np.int8)
        need = np.zeros(nq, np.int32); inpool = np.zeros(nq, np.int8)
        for qi, c in enumerate(ctxs):
            X, _ = KB.f6_select(c["bnd"], c["chal"], c["spos"], c["rpos"], c["cpos"], Bv)
            act[qi] = int(goldp[qi] <= (c["prot_set"] | set(X)))
            uni = set(c["bnd"]) | set(c["chal"])
            miss = goldp[qi] - c["prot_set"]
            need[qi] = len(miss)
            inpool[qi] = int(miss <= uni)
            orc[qi] = int(inpool[qi] and len(miss) <= Bv)
        row[f"B{Bv}"] = {
            "ACTUAL_ALL": round(float(act.mean()), 4),
            "ORACLE_ALL": round(float(orc.mean()), 4),
            "SELECTOR_HEADROOM": round(float(orc.mean() - act.mean()), 4),
            "uncovered_by_oracle": int((1 - orc).sum()),
            "of_those_POOL_failure": int(((1 - orc) & (1 - inpool)).astype(bool).sum()),
            "of_those_need_more_than_B_swaps": int(((1 - orc) & inpool & (need > Bv)).sum()),
            "mean_partitions_needing_replacement": round(float(need.mean()), 3)}
        if ds == "metaqa":
            row[f"B{Bv}"]["per_hop"] = {
                str(h): {"n": int((hops == h).sum()),
                         "ACTUAL": round(float(act[hops == h].mean()), 4),
                         "ORACLE": round(float(orc[hops == h].mean()), 4),
                         "SELECTOR_HEADROOM": round(float(orc[hops == h].mean()
                                                          - act[hops == h].mean()), 4),
                         "POOL_failure": int(((1 - orc) & (1 - inpool))[hops == h].sum())}
                for h in (1, 2, 3) if (hops == h).sum()}
    OUT[ds] = row
    print("%-15s " % ds[:14] + "  ".join(
        "B%-2d act %.4f orc %.4f head %+.4f (pool-fail %d, >B %d)"
        % (b, row[f"B{b}"]["ACTUAL_ALL"], row[f"B{b}"]["ORACLE_ALL"],
           row[f"B{b}"]["SELECTOR_HEADROOM"], row[f"B{b}"]["of_those_POOL_failure"],
           row[f"B{b}"]["of_those_need_more_than_B_swaps"]) for b in (6, 8, 12)), flush=True)
json.dump(OUT, open(f"{PP.PPD}/diag/oracle.json", "w"), indent=1)
print("wrote oracle.json")
