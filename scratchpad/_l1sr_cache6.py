"""STEP 5/6 on the cache path: P0/P1/P2 aggregations over the FROZEN structural nodes, all corpora.

No beam rebuild and no node embeddings, so this runs on the large corpora too.  P0 is the frozen S4
(parity-checked), so the SAFE row is exact and every other row differs from it only in how the same
structural nodes are aggregated into a partition ranking.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP, _l1sr_eval as EV, _l1sr_seq as SQ

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
AGGS = ["P0_S4", "P1_MAX_NODE", "P2_TOP2_SUPPORT"]


def main(*dsets):
    dsets = list(dsets) or PP.DSETS
    fp = f"{SQ.SRD}/diag/step56_cache_all.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in dsets:
        S = EV.substrate(ds); hops = S["hops"]; hard = S["hard"]; z = S["z"]
        base = None; OUT[ds] = {}
        for ag in AGGS:
            SFs = [EV.sf_from_cache(z, qi, hard, ag) for qi in range(S["nq"])]
            ind, ch = EV.evaluate(S, SFs, "A_F6", (EV.P,))
            if base is None:
                base = ind[EV.P]
            m = PP.mcnemar(ind[EV.P], base)
            OUT[ds][ag] = {"P50": EV.by_hop(ind[EV.P], hops), "churn": round(float(ch.mean()), 3),
                           **{f"vs_P0_{k}": v for k, v in m.items()}}
            v = OUT[ds][ag]
            log(f"  {ds:16s} {ag:16s} ALL {v['P50']['ALL']:.4f}  " + "  ".join(
                f"h{h} {v['P50'][f'hop{h}']:.4f}" for h in (1, 2, 3) if f"hop{h}" in v["P50"])
                + f"   net {m['net']:+d} p={m['mcnemar_p']:.3g}{' SIG' if m['sig'] else ''}")
        del S, z
        json.dump(OUT, open(fp, "w"), indent=1)
    log(f"wrote {fp}")


if __name__ == "__main__":
    main(*sys.argv[1:])
