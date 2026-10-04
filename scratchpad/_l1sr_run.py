"""STEP 4-7 + 9 runner."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP, _l1ps_router as RT, _l1sr_eval as EV, _l1sr_seq as SQ

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def run(ds, aggs=EV.AGGS, variants=None, rules=("A_F6",), budgets=(50,)):
    variants = variants or [v for v, _, _ in EV.VARIANTS]
    S = EV.substrate(ds); hops = S["hops"]; hard = S["hard"]
    Bm = EV.beams(ds, S, log)
    SAFE = [RT.order_struct(c["sagg"], "S4") for c in S["ctxs"]]
    ind0, _ = EV.evaluate(S, SAFE, "A_F6", (50,))
    R = {"ds": ds, "SAFE": EV.by_hop(ind0[50], hops), "beam_parity": Bm["_parity_static"],
         "nq": S["nq"], "edges": Bm["_edges"], "cells": {}}
    for nm in variants:
        for ag in aggs:
            SFs = [EV.sf_of(Bm[nm][qi], EV.KEYOF[nm], hard, ag) for qi in range(S["nq"])]
            for rule in rules:
                ind, ch = EV.evaluate(S, SFs, rule, budgets)
                k = f"{nm}x{ag}x{rule}"
                R["cells"][k] = {"P50": EV.by_hop(ind[50], hops),
                                 "churn": round(float(ch.mean()), 3),
                                 **{f"budget{b}": EV.by_hop(ind[b], hops)
                                    for b in budgets if b != 50},
                                 **{f"vs_SAFE_{m}": v for m, v in
                                    PP.mcnemar(ind[50], ind0[50]).items()}}
                v = R["cells"][k]
                log(f"  {k:34s} ALL {v['P50']['ALL']:.4f}  " +
                    "  ".join(f"h{h} {v['P50'].get(f'hop{h}', float('nan')):.4f}"
                              for h in (1, 2, 3) if f"hop{h}" in v["P50"]) +
                    f"   net {v['vs_SAFE_net']:+d} p={v['vs_SAFE_mcnemar_p']:.3g}"
                    f"{' SIG' if v['vs_SAFE_sig'] else ''}")
    os.makedirs(f"{SQ.SRD}/diag", exist_ok=True)
    fp = f"{SQ.SRD}/diag/step456_{ds}.json"
    prev = json.load(open(fp)) if os.path.exists(fp) else {}
    if prev.get("ds") == ds:
        prev["cells"].update(R["cells"]); R = prev
    json.dump(R, open(fp, "w"), indent=1)
    log(f"wrote {fp}")
    return R


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0] if a else "metaqa")
