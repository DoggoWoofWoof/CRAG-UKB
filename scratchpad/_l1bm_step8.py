"""STEP 3 identity check, STEP 8 exact-P50 end-to-end, STEP 9 work-vs-scope accounting.

Nothing downstream of the beam moves: S4 (P0), F6, B = 6 and P = 50 are the frozen ones, read through
the same `EV.sf_from_cache` / `EV.evaluate` path the parity gate validated.  A policy is credited
only with the nodes it put in front of them.

STEP 3 is settled by an identity check rather than by rhetoric: a lexicographic rule can only differ
from its own primary key where that primary key TIES, so the honest report is the measured tie mass
and whether the kept node sets actually diverge.

  python scratchpad/_l1bm_step8.py ds [pol,pol,...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1ps_router as RT
import _l1sr_eval as EV
import _l1bm_core as BM
import _l1bm_run as RUN

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def load(ds, pol):
    fp = f"{BM.BMD}/cache/{ds}__{pol}.npz"
    if not os.path.exists(fp):
        return None
    z = np.load(fp, allow_pickle=True)
    R = {k: z[k] for k in z.files if k not in ("stats",)}
    R["stats"] = json.loads(str(z["stats"]))
    return R


def final_scope(S, SFs, rule="A_F6"):
    """unique corpus nodes exposed by the EXACT 50 partitions (the real downstream scope)."""
    ps = np.asarray(S["z"]["part_sizes"])
    tot = np.zeros(S["nq"], np.int64)
    for qi, c in enumerate(S["ctxs"]):
        sc, _ = EV.order_for(c, SFs[qi], rule)
        fs = c["prot_set"] | set(p for _, _, p in sc[:EV.B])
        tot[qi] = int(ps[sorted(fs)].sum())
    return tot


def main(ds="metaqa", policies=None):
    policies = policies or BM.POLICIES
    S = EV.substrate(ds); nq = S["nq"]; hard = S["hard"]; hops = S["hops"]
    SAFE = [RT.order_struct(c["sagg"], "S4") for c in S["ctxs"]]
    i_safe, _ = EV.evaluate(S, SAFE, "A_F6", (EV.P,))
    sc_safe = final_scope(S, SAFE)
    OUT = {"ds": ds, "nq": nq, "SAFE": EV.by_hop(i_safe[EV.P], hops),
           "SAFE_scope_nodes": round(float(sc_safe.mean()), 1), "STEP8": {}, "STEP3": {}}
    log(f"SAFE {OUT['SAFE']}  scope {OUT['SAFE_scope_nodes']:.0f} nodes")

    nodes = {}
    for pol in policies:
        R = load(ds, pol)
        if R is None:
            log(f"  [missing] {ds} {pol}"); continue
        nodes[pol] = R["s_node"]
        SF = [RUN.sf_arrays(R, qi, hard) for qi in range(nq)]
        ind, churn = EV.evaluate(S, SF, "A_F6", (EV.P,))
        m = PP.mcnemar(ind[EV.P], i_safe[EV.P])
        scope = final_scope(S, SF)
        st = R["stats"]
        blocks = {}
        hs = sorted(set(int(x) for x in hops)) if hops is not None else []
        if hs and min(hs) >= 0:                       # per-hop significance, not just pooled
            for h in hs:
                mk = np.asarray(hops) == h
                blocks[f"hop{h}"] = PP.mcnemar(ind[EV.P][mk], i_safe[EV.P][mk])
        OUT["STEP8"][pol] = {
            "P50": EV.by_hop(ind[EV.P], hops), "churn": round(float(churn.mean()), 3),
            **{f"vs_SAFE_{k}": v for k, v in m.items()}, "vs_SAFE_by_hop": blocks,
            "INTERNAL": {"graph_edges_per_q": st["edges_per_q"],
                         "lookahead_edges_per_q": st["look_edges_per_q"],
                         "total_edges_per_q": round(st["edges_per_q"] + st["look_edges_per_q"], 1),
                         "distinct_nodes_evaluated_per_q": st["cand_total_per_q"],
                         "beam_scope_nodes_per_q": st["scope_per_q"],
                         "latency_ms_per_q": st["latency_ms"]},
            "OUTPUT": {"partitions": EV.P, "scope_nodes_per_q": round(float(scope.mean()), 1)}}
        v = OUT["STEP8"][pol]
        log(f"  {pol:26s} ALL {v['P50']['ALL']:.4f}  " + "  ".join(
            f"h{h} {v['P50'][f'hop{h}']:.4f}" for h in (1, 2, 3) if f"hop{h}" in v["P50"])
            + f"   net {m['net']:+d} p={m['mcnemar_p']:.3g}{' SIG' if m['sig'] else ''}"
            + f"   {v['INTERNAL']['total_edges_per_q']:.0f} e/q  "
              f"{v['INTERNAL']['latency_ms_per_q']:.1f} ms/q  "
              f"scope {v['OUTPUT']['scope_nodes_per_q']:.0f}")

    # ---- STEP 3: a lexicographic rule can only move where its primary key ties
    for lex, prim in (("L3_FUTURE_THEN_CURRENT", "L1_MAX_FUTURE"),
                      ("L4_CURRENT_THEN_FUTURE", "M0_BASELINE")):
        if lex in nodes and prim in nodes:
            same = int((nodes[lex] == nodes[prim]).all(1).sum())
            OUT["STEP3"][f"{lex}_vs_{prim}"] = {
                "identical_node_lists": same, "of": nq,
                "identical_frac": round(same / nq, 4)}
            log(f"  STEP3 {lex} == {prim} on {same}/{nq} queries")

    fp = f"{BM.BMD}/diag/step8_{ds}.json"
    json.dump(OUT, open(fp, "w"), indent=1)
    log(f"wrote {fp}")
    return OUT


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "metaqa",
         sys.argv[2].split(",") if len(sys.argv) > 2 else None)
