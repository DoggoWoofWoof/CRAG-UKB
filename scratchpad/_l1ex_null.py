"""Random-exposure null: coverage a method would get if its ranking carried no information.

Not a router and not a proposal family -- an analysis control, so that "real routing" can be
measured against "scanning" rather than asserted.  For a query needing g gold partitions, exposing
k of N partitions uniformly at random covers it with probability prod_{i<g} (k-i)/(N-i).

Uses the same actual partition membership, so the null is stated at the same node exposure.

  python scratchpad/_l1ex_null.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ex_core as EX

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def phit(g, k, N):
    if g == 0:
        return 1.0
    if k < g:
        return 0.0
    p = 1.0
    for i in range(g):
        p *= (k - i) / (N - i)
    return p


def main():
    fp = f"{EX.EXD}/diag/exposure.json"
    D = json.load(open(fp))
    for ds in EX.DSETS:
        S = EX.CV.substrate(ds)
        N = D[ds]["corpus_partitions"]
        gsz = np.array([len(g) for g in S["goldp"]], np.int32)
        SL = {"ALL": np.ones(len(gsz), bool)}
        if ds == "metaqa":
            for h in (1, 2, 3):
                SL[f"hop{h}"] = S["hops"] == h
        D[ds]["NULL"] = {}
        for cfg, e in D[ds]["PART_BUDGET"].items():
            k = int(round(e["partitions"]["mean"]))
            pv = np.array([phit(int(g), k, N) for g in gsz])
            D[ds]["NULL"][cfg] = {"k_partitions": k,
                                  **{s: round(float(pv[m].mean()), 4) for s, m in SL.items()}}
        log(f"   {ds:15s} null at 50 parts: ALL {D[ds]['NULL']['SAFE']['ALL']:.4f}"
            + (f"  hop3 {D[ds]['NULL']['SAFE']['hop3']:.4f}" if ds == "metaqa" else "")
            + f"   (mean gold partitions {gsz.mean():.2f} of {N})")
    json.dump(D, open(fp, "w"), indent=1)
    log("updated exposure.json with NULL")


if __name__ == "__main__":
    main()
