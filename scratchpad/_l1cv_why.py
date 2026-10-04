"""STEP 5 mechanism -- why the admitted gold does not convert.

The gate is open and gold-bearing proposals ARE admitted, yet coverage barely moves.  A query is
covered only when the WHOLE missing set is swapped in, so this decomposes the residual:

    feasible          miss subset of candidates and |miss| <= B  (the U_PC5 pool oracle)
    covered           miss subset of X
    partial           mean fraction of miss actually selected
    where the misses  rank of each unselected needed partition inside the rule score order,
                      against the B-th selected candidate

  python scratchpad/_l1cv_why.py [ds]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1cg_run as RUN
import _l1cv_core as CV

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
K0 = PP.K0


def score_of(rule, p, cp, sp, up):
    if rule == "S2_CSU_LIFT":
        f = lambda pos, L: (1.0 / (K0 + pos[p]) - 1.0 / (K0 + L)) if p in pos else 0.0
        return f(cp, len(cp)) + f(sp, len(sp)) + f(up, len(up))
    g = lambda pos: (1.0 / (K0 + pos[p])) if p in pos else 0.0
    if rule == "S1_CSU_RAW":
        return g(cp) + g(sp) + g(up)
    if rule == "S3_CU_ABLATION":
        return g(cp) + g(up)
    return g(cp) + g(sp)


def main(ds="metaqa"):
    S = CV.substrate(ds)
    U = CV.u_lists(ds, S, log)
    SL = RUN.slices(ds, S)
    goldp, ctxs = S["goldp"], S["ctxs"]
    out = {"ds": ds, "RULES": {}}
    for rule in CV.RULES:
        nq = S["nq"]
        feas = np.zeros(nq, np.int8); cov = np.zeros(nq, np.int8)
        frac = np.full(nq, np.nan); nmiss = np.zeros(nq, np.int32)
        admitted_gold = np.zeros(nq, np.int32); rescued = np.zeros(nq, np.int8)
        rk, marg, src = [], [], {"in_base50": 0, "in_chal": 0, "truly_new": 0}
        for qi, c in enumerate(ctxs):
            X, cands = CV.select(rule, c, U, qi)
            miss = goldp[qi] - c["prot_set"]
            nmiss[qi] = len(miss)
            cs = set(cands)
            feas[qi] = int(len(goldp[qi]) <= CV.P and len(miss) <= CV.B and miss <= cs)
            cov[qi] = int(miss <= set(X))
            if miss:
                frac[qi] = len(miss & set(X)) / len(miss)
            new = {p for p in CV.proposals(c, U, qi) if p not in (c["base50"] | set(c["chal"]))}
            admitted_gold[qi] = len(set(X) & new & goldp[qi])
            if not feas[qi] or cov[qi]:
                continue
            cp, sp, up = c["cpos"], c["spos"], CV.upos_of(U, qi)
            sc = sorted(((-score_of(rule, p, cp, sp, up), cp.get(p, 10 ** 6), p) for p in cands))
            pos = {p: i for i, (_, _, p) in enumerate(sc)}
            cut = -sc[CV.B - 1][0]
            for p in miss - set(X):
                rk.append(pos[p])
                marg.append(cut - score_of(rule, p, cp, sp, up))
                src["in_base50" if p in c["base50"] else
                    ("in_chal" if p in set(c["chal"]) else "truly_new")] += 1
        e = {}
        for k, m in SL.items():
            f, cv = feas[m].astype(bool), cov[m].astype(bool)
            gap = f & ~cv
            e[k] = {"feasible": round(float(f.mean()), 4), "covered": round(float(cv.mean()), 4),
                    "feasible_but_uncovered": int(gap.sum()),
                    "mean_frac_of_miss_selected": round(float(np.nanmean(frac[m])), 4),
                    "mean_frac_of_miss_selected_on_gap": (
                        round(float(np.nanmean(frac[m][gap])), 4) if gap.sum() else None),
                    "mean_n_miss_on_gap": (round(float(nmiss[m][gap].mean()), 2)
                                           if gap.sum() else None),
                    "queries_admitting_a_gold_proposal": int((admitted_gold[m] > 0).sum())}
        e["UNSELECTED_NEEDED_PARTITIONS"] = {
            "n": len(rk), "source": src,
            "rank_in_rule_order_p50": (int(np.percentile(rk, 50)) if rk else None),
            "rank_in_rule_order_p90": (int(np.percentile(rk, 90)) if rk else None),
            "within_top_12": int(sum(1 for r in rk if r < 12)),
            "within_top_50": int(sum(1 for r in rk if r < 50)),
            "median_score_margin_below_cut": (round(float(np.median(marg)), 8) if marg else None)}
        out["RULES"][rule] = e
        k0 = "hop3" if ds == "metaqa" else "ALL"
        v, u = e[k0], e["UNSELECTED_NEEDED_PARTITIONS"]
        log(f"   {rule:14s} {k0}: feas {v['feasible']:.4f} cov {v['covered']:.4f} "
            f"gap {v['feasible_but_uncovered']:5d} q  frac(miss sel) all {v['mean_frac_of_miss_selected']:.3f} "
            f"gap {v['mean_frac_of_miss_selected_on_gap']}  |miss|@gap {v['mean_n_miss_on_gap']}  "
            f"| unselected-needed n={u['n']} p50rank {u['rank_in_rule_order_p50']} "
            f"top12 {u['within_top_12']} src {u['source']}")
    os.makedirs(f"{CV.CVD}/diag", exist_ok=True)
    fp = f"{CV.CVD}/diag/why.json"
    A = json.load(open(fp)) if os.path.exists(fp) else {}
    A[ds] = out; json.dump(A, open(fp, "w"), indent=1)
    log("wrote why.json")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
