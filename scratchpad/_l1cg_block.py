"""Why does an expanded pool convert to exactly zero?  Direct verification of the blocker.

CLAIM.  Under the frozen F6/R0 selector, a candidate q outside base50 and outside SF u RF can never
be selected, for any B and on any corpus.

  s(q)  = 1/(K0 + cpos[q])  if q is in cpos else 0        (no spos/rpos: q is not in SF u RF)
  q not in base50           =>  cpos[q] >= P  =>  s(q) <= 1/(K0 + P)
  p in bnd = base_rank[P-B:P] =>  cpos[p] <= P-1  =>  s(p) >= 1/(K0 + P - 1) > 1/(K0 + P)

There are exactly B incumbents and exactly B slots, so every such q is strictly dominated.  This
script checks the claim empirically rather than trusting the algebra: it counts truly-new candidates
offered, how many are ever selected, and the realised score margin.

  python scratchpad/_l1cg_block.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1kb_core as KB
import _l1cg_core as CG
import _l1cg_run as RUN

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def run_ds(ds, OUT, fams=("A_DENSE_PARTITION", "B_SPLADE_PARTITION", "F_PPR_REACH",
                          "I_CANON_CONT", "G_GRAPH_NBR_RAW"), M=256):
    S = CG.substrate(ds); O = CG.proposals(ds, S)
    nq = S["nq"]; ctxs = S["goldp"], S["ctxs"]
    offered = sel_new = qs_with_new = 0
    margins, best_new, worst_inc = [], [], []
    ident = 0
    for qi, c in enumerate(S["ctxs"]):
        ex = RUN.rr(O, list(fams), qi, M)
        new = [p for p in ex if p not in c["base50"] and p not in set(c["chal"])]
        offered += len(new); qs_with_new += int(bool(new))
        X0, _ = KB.f6_select(c["bnd"], list(c["chal"]), c["spos"], c["rpos"], c["cpos"], CG.B)
        X1, sc = KB.f6_select(c["bnd"], list(c["chal"]) + ex, c["spos"], c["rpos"], c["cpos"], CG.B)
        ident += int(set(X0) == set(X1))
        sel_new += len(set(X1) & set(new))
        sd = {p: -s for s, _, p in sc}
        if new:
            bn = max(sd[p] for p in new); wi = min(sd[p] for p in X1)
            best_new.append(bn); worst_inc.append(wi); margins.append(wi - bn)
    r = {"families": list(fams), "M": M, "nq": nq,
         "queries_offered_a_truly_new_candidate": qs_with_new,
         "truly_new_candidates_offered_total": offered,
         "truly_new_candidates_ever_SELECTED": sel_new,
         "selections_bit_identical_to_SAFE": ident,
         "max_score_of_a_new_candidate_median": round(float(np.median(best_new)), 8) if best_new else None,
         "min_score_of_a_selected_incumbent_median": round(float(np.median(worst_inc)), 8) if worst_inc else None,
         "margin_min_over_queries": round(float(np.min(margins)), 8) if margins else None,
         "theoretical_bound_new_max": round(1.0 / (CG.K0 + CG.P), 8),
         "theoretical_bound_incumbent_min": round(1.0 / (CG.K0 + CG.P - 1), 8)}
    log(f"   {ds}: new offered {offered} over {qs_with_new}/{nq} queries -> SELECTED {sel_new}; "
        f"identical {ident}/{nq}; min margin {r['margin_min_over_queries']}")
    OUT[ds] = r


def main():
    os.makedirs(f"{CG.CGD}/diag", exist_ok=True)
    fp = f"{CG.CGD}/diag/blocker.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in (sys.argv[1:] or CG.DSETS):
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote blocker.json")


if __name__ == "__main__":
    main()
