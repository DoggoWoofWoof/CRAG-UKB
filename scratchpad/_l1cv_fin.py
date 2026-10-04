"""STEP 7 pooled significance + STEP 10 latency accounting.

Pooled McNemar over all six corpora (the per-corpus tests are the promotion gate; the pooled test
only asks whether a rule that is individually non-significant everywhere is a real effect at all).

Latency is measured as the marginal online cost of the conversion design over SAFE:
    U lookup + merge   the five cached source lists, round-robin, dedup   (per query)
    selector           scoring |candidates| and taking the top B          (per query)
ONLINE_GRAPH_EDGES_TOUCHED must be 0: G_GRAPH_NBR_RAW reads precomputed depth-1 neighbour rows.

  python scratchpad/_l1cv_fin.py
"""
import os, sys, json, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1kb_core as KB
import _l1cg_core as CG
import _l1cv_core as CV

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def main():
    OUT = {"POOLED": {}, "LATENCY": {}, "ONLINE_GRAPH_EDGES_TOUCHED": 0}
    IND = {}
    for ds in CV.DSETS:
        S = CV.substrate(ds)
        U = CV.u_lists(ds, S)
        IND[ds] = {r: CV.run(r, S, U)["ind"] for r in CV.RULES}

        # ---- STEP 10 latency, measured on this corpus
        ctxs, nq = S["ctxs"], S["nq"]
        n = min(400, nq)
        t = time.time()
        for qi in range(n):
            CV.upos_of(U, qi)
        t_u = (time.time() - t) / n * 1e3
        for rule in ("S0_SAFE", "S1_CSU_RAW"):
            t = time.time()
            for qi in range(n):
                CV.select(rule, ctxs[qi], U, qi)
            OUT["LATENCY"].setdefault(ds, {})[f"selector_{rule}_ms"] = round(
                (time.time() - t) / n * 1e3, 3)
        L = OUT["LATENCY"][ds]
        L["u_rank_lookup_ms"] = round(t_u, 3)
        L["mean_candidates"] = round(float(np.mean(
            [len(CV.candidates(ctxs[qi], U, qi)[0]) for qi in range(n)])), 1)
        L["u_cache_bytes"] = os.path.getsize(f"{CV.CVD}/u/u_{ds}.pkl")
        L["partition_graph_cache_bytes"] = os.path.getsize(f"{CG.CGD}/pg/pg_{ds}.npz")
        log(f"   {ds:16s} u_lookup {L['u_rank_lookup_ms']:.3f} ms  "
            f"sel SAFE {L['selector_S0_SAFE_ms']:.3f} -> S1 {L['selector_S1_CSU_RAW_ms']:.3f} ms  "
            f"cands {L['mean_candidates']:.0f}  u-cache {L['u_cache_bytes']/1e6:.1f} MB  "
            f"pg-cache {L['partition_graph_cache_bytes']/1e6:.1f} MB")

    for r in CV.RULES[1:]:
        a = np.concatenate([IND[d]["S0_SAFE"] for d in CV.DSETS])
        b = np.concatenate([IND[d][r] for d in CV.DSETS])
        st = PP.mcnemar(b, a)
        deltas = [float(IND[d][r].mean() - IND[d]["S0_SAFE"].mean()) for d in CV.DSETS]
        OUT["POOLED"][r] = {
            "pooled_delta": round(float(b.mean() - a.mean()), 5),
            "macro_delta": round(float(np.mean(deltas)), 5),
            "worst_corpus_delta": round(min(deltas), 5),
            "corpora_improved": int(sum(1 for d in deltas if d > 0)),
            "gained": st["gained"], "lost": st["lost"], "net": st["net"],
            "mcnemar_p": st["mcnemar_p"], "sig": st["sig"]}
        v = OUT["POOLED"][r]
        log(f"   POOLED {r:14s} pooled {v['pooled_delta']:+.5f} macro {v['macro_delta']:+.5f} "
            f"worst {v['worst_corpus_delta']:+.5f} up-on {v['corpora_improved']}/6  "
            f"+{v['gained']}/-{v['lost']} p={v['mcnemar_p']:.3g} sig={v['sig']}")

    json.dump(OUT, open(f"{CV.CVD}/diag/fin.json", "w"), indent=1)
    log("wrote fin.json")


if __name__ == "__main__":
    main()
