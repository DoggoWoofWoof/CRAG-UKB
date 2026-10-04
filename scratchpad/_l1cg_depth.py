"""The one candidate-generation knob that is CONVERTIBLE under the frozen scoring rule.

Every family in STEP 1 proposes partitions that carry no structural or retrieval rank, so the frozen
R0/F6 arithmetic scores them canonical-only and they are provably dominated (see _l1cg_block.py).
M_struct / M_ret are different: they are the depths at which the GENERATOR aggregates, so raising
them both enlarges the candidate pool and extends the spos / rpos rank lists that already exist.

    scoring rule    1/(K0+cpos) + 1/(K0+spos) + 1/(K0+rpos), K0 = 60, top-B      -- UNCHANGED
    generator depth M_struct 64 -> {128, 256},  M_ret 32 -> {64, 128}            -- CHANGED

Ranks of partitions already in the lists are unchanged (the lists are prefix-extended), so this is
not a re-calibration; it is strictly "how many candidates does the generator produce".  Whether that
counts as inside the frozen contract is a judgement call and is reported as such, separately.

  python scratchpad/_l1cg_depth.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1ps_router as RT
import _l1kb_core as KB
import _l1cg_core as CG
import _l1cg_run as RUN

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
GRID = [(64, 32), (128, 32), (256, 32), (64, 64), (64, 128), (256, 128)]
B = CG.B


def ctx_for(ds, z, meta, Ms, Mr):
    C = RT.build_cache(ds, z, meta, [Ms], [Mr], ["S4"])
    return KB.contexts(z, meta, C, B, dict(KB.BASE_CFG, M_struct=Ms, M_ret=Mr, agg="S4", B=B))


def run_ds(ds, OUT):
    z, meta = PP.load(ds)
    # the cache is a lazy NpzFile; the aggregators index it once per query, so decompressing it
    # once here instead of once per access is a ~100x speedup and changes no value.
    z = {k: z[k] for k in z.files}
    nq = meta["n_dev_queries"]
    goldp = PP.goldparts(z, meta)
    S0 = CG.substrate(ds)
    SL = RUN.slices(ds, S0)
    row = {}
    base = None
    for Ms, Mr in GRID:
        t = time.time()
        ctxs = ctx_for(ds, z, meta, Ms, Mr)
        S = dict(S0); S["ctxs"] = ctxs; S["T"] = CG.targets(S)
        ind, churn = CG.run_frozen(S, None, B)
        orc = CG.pool_oracle(S, S["T"], None, B)
        if base is None:
            base = ind
        e = {"mean_pool": round(float(np.mean([len(set(c["chal"])) for c in ctxs])), 1),
             "mean_churn": round(float(churn.mean()), 3), "secs": round(time.time() - t, 1)}
        for k, m in SL.items():
            st = PP.mcnemar(ind[m], base[m])
            e[k] = {"ACTUAL": round(float(ind[m].mean()), 4),
                    "delta_vs_Ms64_Mr32": round(float(ind[m].mean() - base[m].mean()), 4),
                    "POOL_ORACLE": round(float(orc[m].mean()), 4),
                    "gained": st["gained"], "lost": st["lost"], "p": st["mcnemar_p"],
                    "sig": st["sig"]}
        row[f"Ms{Ms}_Mr{Mr}"] = e
        k0 = "hop3" if ds == "metaqa" else "ALL"
        v = e[k0]
        log(f"   {ds} Ms={Ms:<4d} Mr={Mr:<4d} pool {e['mean_pool']:6.1f}  actual {v['ACTUAL']:.4f} "
            f"({v['delta_vs_Ms64_Mr32']:+.4f}) oracle {v['POOL_ORACLE']:.4f} "
            f"+{v['gained']}/-{v['lost']} p={v['p']:.3g}")
    OUT[ds] = row


def main():
    os.makedirs(f"{CG.CGD}/diag", exist_ok=True)
    fp = f"{CG.CGD}/diag/depth.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in (sys.argv[1:] or ["metaqa"]):
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote depth.json")


if __name__ == "__main__":
    main()
