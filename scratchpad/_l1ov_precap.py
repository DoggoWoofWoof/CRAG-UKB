"""PHASE 3 -- the hyperedge size distribution BEFORE any cap is applied.

The program is explicit: if enormous hyperedges exist, report them BEFORE capping, and then use
ONE identical universal cap.  The export manifest recorded only post-cap sizes plus a dropped
count, so this fills that gap.  A closed neighbourhood {v} u N(v) has size deg(v)+1, so the
pre-cap distribution is exactly the degree distribution of the family graph, plus one.

Nothing here changes any partition -- it is a measurement of the input.

  python scratchpad/_l1ov_precap.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1kn_sub as KS
import _l1ep_part as PP

OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_UNIVERSAL_PARTITION_SEARCH"
CAP = PP.HYPER_CAP
FAMS = (("H_STRUCT_LOCAL", 1), ("H_KNN_LOCAL", 2))


def stats(N, keys, cap):
    u = (keys // np.int64(N)).astype(np.int64)
    v = (keys % np.int64(N)).astype(np.int64)
    deg = np.bincount(np.concatenate([u, v]), minlength=N).astype(np.int64)
    sz = deg[deg >= 1] + 1                       # closed neighbourhood size, pre-cap
    over = sz > cap
    return {"hyperedges_precap": int(len(sz)), "pins_precap": int(sz.sum()),
            "size_mean": round(float(sz.mean()), 2),
            "size_p50": float(np.percentile(sz, 50)), "size_p90": float(np.percentile(sz, 90)),
            "size_p99": float(np.percentile(sz, 99)),
            "size_p999": float(np.percentile(sz, 99.9)), "size_max": int(sz.max()),
            "n_over_cap": int(over.sum()),
            "frac_over_cap": round(float(over.mean()), 5),
            "pins_over_cap": int(sz[over].sum()),
            "frac_pins_over_cap": round(float(sz[over].sum() / sz.sum()), 5),
            "largest_5": sorted(sz.tolist(), reverse=True)[:5]}


def main():
    fp = f"{OUT}/hypergraph/PRECAP_HYPEREDGE_SIZES.json"
    out = {"CAP": int(CAP),
           "CAP_NOTE": ("one identical cap on every corpus and every family; chosen before any "
                        "retrieval measurement and never tuned per corpus"),
           "DEFINITION": "hyperedge = closed neighbourhood {v} u N(v), so size = deg(v)+1",
           "PER_CORPUS": {}}
    for ds in KS.DS:
        N, S, K, X = KS.keysets(ds, lambda *a: None)
        out["PER_CORPUS"][ds] = {"N": int(N),
                                 "H_STRUCT_LOCAL": stats(N, S, CAP),
                                 "H_KNN_LOCAL": stats(N, K, CAP)}
        for fam in ("H_STRUCT_LOCAL", "H_KNN_LOCAL"):
            s = out["PER_CORPUS"][ds][fam]
            print(f"{ds:16s} {fam:16s} n={s['hyperedges_precap']:>8,}  mean {s['size_mean']:6.2f}  "
                  f"p99 {s['size_p99']:6.1f}  p99.9 {s['size_p999']:7.1f}  max {s['size_max']:>7,}  "
                  f"over cap {s['n_over_cap']:>6,} ({s['frac_over_cap']:.3%}), "
                  f"{s['frac_pins_over_cap']:.2%} of pins")
    os.makedirs(f"{OUT}/hypergraph", exist_ok=True)
    json.dump(out, open(fp, "w"), indent=1)
    print("wrote", fp)
    return out


if __name__ == "__main__":
    main()
