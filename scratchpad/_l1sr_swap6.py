"""STEP 9 universality check -- the three swap rules on all six corpora, frozen ranking, exact P50.

Structure is only one of the three channels F6 fuses, and the fixed-P50 router was adopted BECAUSE
symmetric competition was universal where single channels were not.  A swap rule that wins on one KB
therefore means nothing until it is measured on all six.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP, _l1ps_router as RT, _l1sr_eval as EV, _l1sr_seq as SQ

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
RULES = ("A_F6", "B_STRUCT_DIRECT", "C_PAIRWISE_EVIDENCE")


def main(*dsets):
    dsets = list(dsets) or PP.DSETS
    fp = f"{SQ.SRD}/diag/step9_all.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in dsets:
        S = EV.substrate(ds); hops = S["hops"]
        SF = [RT.order_struct(c["sagg"], "S4") for c in S["ctxs"]]
        base = None; OUT[ds] = {}
        for rule in RULES:
            ind, ch = EV.evaluate(S, SF, rule, (EV.P,))
            if base is None:
                base = ind[EV.P]
            m = PP.mcnemar(ind[EV.P], base)
            OUT[ds][rule] = {"P50": EV.by_hop(ind[EV.P], hops), "churn": round(float(ch.mean()), 3),
                             **{f"vs_F6_{k}": v for k, v in m.items()}}
            v = OUT[ds][rule]
            log(f"  {ds:16s} {rule:22s} ALL {v['P50']['ALL']:.4f}  " + "  ".join(
                f"h{h} {v['P50'][f'hop{h}']:.4f}" for h in (1, 2, 3) if f"hop{h}" in v["P50"])
                + f"   net {m['net']:+d} p={m['mcnemar_p']:.3g}{' SIG' if m['sig'] else ''}")
        del S
        json.dump(OUT, open(fp, "w"), indent=1)
    log(f"wrote {fp}")


if __name__ == "__main__":
    main(*sys.argv[1:])
