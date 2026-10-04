"""STEP 11 -- re-run B = {6, 8, 12} ONCE, now that candidate generation has moved the pool.

B was measured as non-binding in the ceiling audit against the OLD pool.  The directive is explicit
that this must not be treated as settled after the pool improves, so it is re-measured here against
the expanded pool -- both the pool oracle (where more slots can pay) and the frozen selector.

The protected core shrinks as B grows (prot = base_rank[:P-B]), so every B is a different frozen
configuration and each gets its own substrate.

  python scratchpad/_l1cg_s11.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1cal_core as CC
import _l1cg_core as CG
import _l1cg_run as RUN

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
BS = [6, 8, 12]
BEST = ["B_SPLADE_PARTITION", "I_CANON_CONT", "C_NODE_DENSE_CONT",
        "D_NODE_SPLADE_CONT", "G_GRAPH_NBR_RAW"]                    # U_PC5, zero online edges
M = 256


def run_ds(ds, OUT):
    S0 = CG.substrate(ds); O = CG.proposals(ds, S0)
    SL = RUN.slices(ds, S0)
    ex = [RUN.rr(O, BEST, qi, M) for qi in range(S0["nq"])]
    row = {}
    for Bv in BS:
        S = CC.substrate(ds, Bv)
        S["T"] = CG.targets(S)
        base, _ = CG.run_frozen(S, None, Bv)
        ind, churn = CG.run_frozen(S, ex, Bv)
        o0 = CG.pool_oracle(S, S["T"], None, Bv)
        o1 = CG.pool_oracle(S, S["T"], ex, Bv)
        row[str(Bv)] = {"mean_churn": round(float(churn.mean()), 3)}
        for k, m in SL.items():
            row[str(Bv)][k] = {
                "SAFE": round(float(base[m].mean()), 4),
                "EXPANDED_ACTUAL": round(float(ind[m].mean()), 4),
                "CURRENT_POOL_ORACLE": round(float(o0[m].mean()), 4),
                "EXPANDED_POOL_ORACLE": round(float(o1[m].mean()), 4)}
        k0 = "hop3" if ds == "metaqa" else "ALL"
        v = row[str(Bv)][k0]
        log(f"   {ds} B={Bv:<3d} SAFE {v['SAFE']:.4f}  actual {v['EXPANDED_ACTUAL']:.4f}  "
            f"pool-oracle {v['CURRENT_POOL_ORACLE']:.4f} -> {v['EXPANDED_POOL_ORACLE']:.4f}")
    OUT[ds] = {"proposer": BEST, "M": M, "B": row}


def main():
    os.makedirs(f"{CG.CGD}/diag", exist_ok=True)
    fp = f"{CG.CGD}/diag/step11.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in (sys.argv[1:] or CG.DSETS):
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote step11.json")


if __name__ == "__main__":
    main()
