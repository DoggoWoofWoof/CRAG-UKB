"""STEP 7 budget curve on all six corpora (frozen ranking, cache path)."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP, _l1ps_router as RT, _l1sr_eval as EV, _l1sr_seq as SQ
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
fp = f"{SQ.SRD}/diag/step7_curve_all.json"
OUT = json.load(open(fp)) if os.path.exists(fp) else {}
for ds in (sys.argv[1:] or PP.DSETS):
    S = EV.substrate(ds); hops = S["hops"]
    SF = [RT.order_struct(c["sagg"], "S4") for c in S["ctxs"]]
    ind, fill = EV.evaluate_budget(S, SF, "A_F6", EV.BUDGETS)
    OUT[ds] = {f"k{k}": {**EV.by_hop(ind[k], hops), **fill[k]} for k in EV.BUDGETS}
    OUT[ds]["_deltas"] = {f"{a}->{b}": round(float(ind[b].mean() - ind[a].mean()), 4)
                          for a, b in ((50, 64), (64, 80), (80, 100), (100, 128), (128, 256))}
    log(f"  {ds}: " + "  ".join(f"k{k} {OUT[ds][f'k{k}']['ALL']:.4f}"
                                f"({OUT[ds][f'k{k}']['mean_partitions']:.0f}p)"
                                for k in EV.BUDGETS))
    del S
    json.dump(OUT, open(fp, "w"), indent=1)
log(f"wrote {fp}")
