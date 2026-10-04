"""STEPS 5 / 6 / 7 -- the conversion measurement.

STEP 5  MetaQA primary: hop1 hop2 hop3 ALL for S0..S3 plus the conversion bookkeeping.
STEP 6  CONVERSION_EFFICIENCY = (ACTUAL_NEW - SAFE) / (U_PC5_POOL_ORACLE - SAFE).
STEP 7  the same table on all six corpora for the rules that cleared the STEP 4 gate.

Bookends reported next to every ACTUAL so a gain can be read against what was actually available:
    SAFE                     the frozen R0/B6_S4_F6 scoreboard
    CURRENT_POOL_ORACLE      perfect selector, frozen pool
    U_PC5_POOL_ORACLE        perfect selector, pool + U_PC5 proposals   <-- the target
    UNLIMITED_POOL_ORACLE    perfect selector, every partition, still B=6
    FULL_UNIVERSE_P50_ORACLE any 50 partitions at all

  python scratchpad/_l1cv_main.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1cg_core as CG
import _l1cg_run as RUN
import _l1cv_core as CV

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def bookends(S, extra, Bv=CV.B):
    T = S["T"]
    return {"CURRENT_POOL_ORACLE": CG.pool_oracle(S, T, None, Bv),
            "U_PC5_POOL_ORACLE": CG.pool_oracle(S, T, extra, Bv),
            "UNLIMITED_POOL_ORACLE": np.array(
                [int(t["feasible"] and len(t["miss"]) <= Bv) for t in T], np.int8),
            "FULL_UNIVERSE_P50_ORACLE": np.array([int(t["feasible"]) for t in T], np.int8)}


def run_ds(ds, OUT, Bv=CV.B):
    S = CV.substrate(ds, Bv)
    U = CV.u_lists(ds, S, log)
    SL = RUN.slices(ds, S)
    base50 = [S["ctxs"][qi]["base50"] for qi in range(S["nq"])]
    extra = [CV.proposals(S["ctxs"][qi], U, qi) for qi in range(S["nq"])]
    BK = bookends(S, extra, Bv)
    R = {r: CV.run(r, S, U, Bv) for r in CV.RULES}
    b = R["S0_SAFE"]
    row = {"B": Bv, "nq": int(S["nq"]),
           "mean_U_len": round(float(np.mean([len(u) for u in U])), 1),
           "mean_U_inside_base50": round(float(np.mean(
               [sum(1 for p in U[qi] if p in base50[qi]) for qi in range(S["nq"])])), 1),
           "mean_U_PC5_proposals": round(float(np.mean([len(e) for e in extra])), 1),
           "mean_new_proposals_offered": round(float(b["new_offered"].mean()), 1),
           "BOOKENDS": {}, "RULES": {}}
    for k, m in SL.items():
        row["BOOKENDS"][k] = {"SAFE": round(float(b["ind"][m].mean()), 4),
                              **{n: round(float(v[m].mean()), 4) for n, v in BK.items()}}
    for r in CV.RULES:
        m = R[r]
        e = {"mean_churn": round(float(m["churn"].mean()), 3),
             "NEW_PROPOSALS_SELECTED": int(m["new_selected"].sum()),
             "new_selected_per_query": round(float(m["new_selected"].mean()), 3),
             "queries_with_a_new_proposal_selected": int((m["new_selected"] > 0).sum()),
             "gold_bearing_U_proposals_admitted": int(m["new_gold_admitted"].sum()),
             "gold_bearing_incumbents_evicted": int(m["gold_incumbents_evicted"].sum()),
             "queries_whose_P50_changes": int(sum(1 for x, y in zip(m["final"], b["final"])
                                                  if x != y))}
        for k, msk in SL.items():
            st = PP.mcnemar(m["ind"][msk], b["ind"][msk])
            act = float(m["ind"][msk].mean()); safe = float(b["ind"][msk].mean())
            hi = float(BK["U_PC5_POOL_ORACLE"][msk].mean())
            e[k] = {"ACTUAL": round(act, 4), "delta_vs_SAFE": round(act - safe, 4),
                    "newly_covered": st["gained"], "newly_uncovered": st["lost"],
                    "net": st["net"], "mcnemar_p": st["mcnemar_p"], "sig": st["sig"],
                    "CONVERSION_EFFICIENCY": (round((act - safe) / (hi - safe), 4)
                                              if hi - safe > 1e-12 else None)}
        row["RULES"][r] = e
        k0 = "hop3" if ds == "metaqa" else "ALL"
        v, a = e[k0], e["ALL"]
        ce = v["CONVERSION_EFFICIENCY"]
        cs = "  n/a " if ce is None else f"{ce:+.1%}"
        log(f"   {ds:12s} {r:14s} ALL {a['ACTUAL']:.4f} ({a['delta_vs_SAFE']:+.4f}) "
            f"{k0} {v['ACTUAL']:.4f} ({v['delta_vs_SAFE']:+.4f}) conv {cs:>7s} "
            f"+{a['newly_covered']}/-{a['newly_uncovered']} p={a['mcnemar_p']:.3g} "
            f"newsel {e['NEW_PROPOSALS_SELECTED']:5d} newgold {e['gold_bearing_U_proposals_admitted']:4d} "
            f"churn {e['mean_churn']:.2f}")
    OUT[ds] = row


def main():
    os.makedirs(f"{CV.CVD}/diag", exist_ok=True)
    fp = f"{CV.CVD}/diag/main.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in (sys.argv[1:] or ["metaqa"]):
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote main.json")


if __name__ == "__main__":
    main()
