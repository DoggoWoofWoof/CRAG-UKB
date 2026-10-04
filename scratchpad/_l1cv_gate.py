"""STEP 4 -- the conversion gate.  MetaQA only; nothing else runs until it passes.

Hard gate: NEW_PROPOSALS_SELECTED > 0 for S1 or S2.  If a truly-new proposal still cannot be
chosen, the new score has not actually opened the selector and there is nothing to measure.

  python scratchpad/_l1cv_gate.py [ds]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1cv_core as CV

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def main(ds="metaqa"):
    S = CV.substrate(ds)
    U = CV.u_lists(ds, S, log)
    U2 = CV.u_lists(ds, S, log, cache=False)
    det = len(U) == len(U2) and all(a == b for a, b in zip(U, U2))
    log(f"U_RANK deterministic: {det};  mean |U| = {np.mean([len(u) for u in U]):.1f}")
    assert det, "u_rank is not deterministic"

    inc = float(np.mean([sum(1 for p in U[qi] if p in S["ctxs"][qi]["base50"])
                         for qi in range(S["nq"])]))
    log(f"mean U entries that are base50 incumbents = {inc:.1f}  (symmetry gate: must be > 0)")
    assert inc > 0, "u_rank was built after excluding base50 -- asymmetric"

    out = {"ds": ds, "deterministic": bool(det),
           "mean_U_len": float(np.mean([len(u) for u in U])),
           "mean_U_entries_inside_base50": round(inc, 2), "RULES": {}}
    ref = None
    for r in CV.RULES:
        m = CV.run(r, S, U)
        if ref is None:
            ref = m
        st = PP.mcnemar(m["ind"], ref["ind"])
        e = {"ALL": round(float(m["ind"].mean()), 4),
             "hop3": round(float(m["ind"][S["hops"] == 3].mean()), 4),
             "NEW_PROPOSALS_OFFERED": int(m["new_offered"].sum()),
             "NEW_PROPOSALS_SELECTED": int(m["new_selected"].sum()),
             "queries_with_a_new_proposal_selected": int((m["new_selected"] > 0).sum()),
             "queries_whose_P50_changes": int(sum(1 for a, b in zip(m["final"], ref["final"])
                                                  if a != b)),
             "new_gold_admitted": int(m["new_gold_admitted"].sum()),
             "gold_incumbents_evicted": int(m["gold_incumbents_evicted"].sum()),
             "mean_churn": round(float(m["churn"].mean()), 3),
             "gained": st["gained"], "lost": st["lost"], "p": st["mcnemar_p"]}
        out["RULES"][r] = e
        log(f"   {r:14s} ALL {e['ALL']:.4f} hop3 {e['hop3']:.4f}  NEW_SEL "
            f"{e['NEW_PROPOSALS_SELECTED']:6d} on {e['queries_with_a_new_proposal_selected']:5d} q  "
            f"P50-changed {e['queries_whose_P50_changes']:5d}  newgold {e['new_gold_admitted']:5d}  "
            f"evicted {e['gold_incumbents_evicted']:5d}  churn {e['mean_churn']:.2f}  "
            f"+{e['gained']}/-{e['lost']}")
    gate = any(out["RULES"][r]["NEW_PROPOSALS_SELECTED"] > 0
               for r in ["S1_CSU_RAW", "S2_CSU_LIFT"])
    out["CONVERSION_GATE_PASSED"] = bool(gate)
    os.makedirs(f"{CV.CVD}/diag", exist_ok=True)
    json.dump(out, open(f"{CV.CVD}/diag/gate.json", "w"), indent=1)
    log(f"CONVERSION_GATE_PASSED = {gate}")
    return gate


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
