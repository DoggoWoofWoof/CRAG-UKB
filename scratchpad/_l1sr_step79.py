"""STEP 7 (candidate-budget curve) and STEP 9 (swap rules), on the cached beams.

STEP 7 asks whether the structural gain available at a large budget can be made available at a small
one -- "move the curve left".  The contract shape is held fixed and ONLY the budget moves, so k = 50
is exactly the frozen P50 output and every other point is the same score order read deeper.

STEP 9 is run last, and only on the ranking that survived STEPS 4-6:
    A_F6                current frozen selector (control)
    B_STRUCT_DIRECT     the strong structural challenger replaces the boundary outright
    C_PAIRWISE_EVIDENCE one deterministic pairwise rule -- a candidate outranks another iff it
                        carries strictly more evidence CHANNELS.  No weights, no thresholds.

  python scratchpad/_l1sr_step79.py [ds] [variant] [agg]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1ps_router as RT
import _l1sr_eval as EV
import _l1sr_seq as SQ

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def main(ds="metaqa", variant="N0", agg="P0_S4"):
    S = EV.substrate(ds); hops = S["hops"]; hard = S["hard"]
    Bm = EV.beams(ds, S, log)
    SAFE = [RT.order_struct(c["sagg"], "S4") for c in S["ctxs"]]
    ind0, _ = EV.evaluate(S, SAFE, "A_F6", (EV.P,))
    OUT = {"ds": ds, "variant": variant, "agg": agg,
           "SAFE_P50": EV.by_hop(ind0[EV.P], hops), "STEP7": {}, "STEP9": {}}

    # ---- STEP 7: budget curve, on the frozen structural ranking and on the survivor
    for nm, SFs in (("SAFE_FROZEN", SAFE),
                    (f"{variant}x{agg}", [EV.sf_of(Bm[variant][qi], EV.KEYOF[variant], hard, agg)
                                          for qi in range(S["nq"])])):
        ind, short = EV.evaluate_budget(S, SFs, "A_F6", EV.BUDGETS)
        OUT["STEP7"][nm] = {f"k{k}": EV.by_hop(ind[k], hops) for k in EV.BUDGETS}
        OUT["STEP7"][nm]["_fill"] = {f"k{k}": short[k] for k in EV.BUDGETS}
        d = OUT["STEP7"][nm]
        OUT["STEP7"][nm]["_deltas"] = {
            f"{a}->{b}": {m: round(d[f"k{b}"][m] - d[f"k{a}"][m], 4) for m in d["k50"]}
            for a, b in ((50, 64), (64, 80), (80, 100), (100, 128), (128, 256))}
        log(f"  STEP7 {nm}")
        for k in EV.BUDGETS:
            v = d[f"k{k}"]
            log(f"    k={k:3d}  ALL {v['ALL']:.4f}  " + "  ".join(
                f"h{h} {v[f'hop{h}']:.4f}" for h in (1, 2, 3) if f"hop{h}" in v)
                + (f"   [only {short[k]['mean_partitions']:.1f} partitions available"
                   f" on {short[k]['underfilled']} queries]" if short[k]["underfilled"] else ""))

    # ---- STEP 9: swap rules, only now that the ranking question is settled
    SFs = [EV.sf_of(Bm[variant][qi], EV.KEYOF[variant], hard, agg) for qi in range(S["nq"])]
    for rule in ("A_F6", "B_STRUCT_DIRECT", "C_PAIRWISE_EVIDENCE"):  # STEP 9
        ind, ch = EV.evaluate(S, SFs, rule, (EV.P,))
        m = PP.mcnemar(ind[EV.P], ind0[EV.P])
        OUT["STEP9"][rule] = {"P50": EV.by_hop(ind[EV.P], hops),
                              "churn": round(float(ch.mean()), 3),
                              **{f"vs_SAFE_{k}": v for k, v in m.items()}}
        v = OUT["STEP9"][rule]
        log(f"  STEP9 {rule:22s} ALL {v['P50']['ALL']:.4f}  " + "  ".join(
            f"h{h} {v['P50'][f'hop{h}']:.4f}" for h in (1, 2, 3) if f"hop{h}" in v["P50"])
            + f"   net {m['net']:+d} p={m['mcnemar_p']:.3g}{' SIG' if m['sig'] else ''}")

    fp = f"{SQ.SRD}/diag/step79_{ds}.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    log(f"wrote {fp}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
