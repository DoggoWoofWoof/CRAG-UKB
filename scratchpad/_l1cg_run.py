"""STEPS 5 / 7 / 8 -- expanded candidate pools under the FROZEN selector.

STEP 8 is the discipline this file enforces: the selector is never touched.  Every configuration
reports BOTH (1) the oracle of the expanded pool and (2) the actual result of the frozen R0/F6
selector over exactly that pool, so a pool gain and a selector conversion are never confused.

Union proposers are round-robin interleaves of their member families -- parameter-free, identical
on every corpus, no learning, one global budget M_TOTAL.

  python scratchpad/_l1cg_run.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1pp_core as PP
import _l1cg_core as CG

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

MPPR = [16, 32, 64, 128]                 # STEP 5
MTOT = [32, 64, 128, 256]                # STEP 7
UNIONS = {
    "U_RET2":    ["B_SPLADE_PARTITION", "I_CANON_CONT"],
    "U_RET3":    ["B_SPLADE_PARTITION", "I_CANON_CONT", "A_DENSE_PARTITION"],
    "U_RET2_G":  ["B_SPLADE_PARTITION", "I_CANON_CONT", "G_GRAPH_NBR_RAW"],
    "U_RET2_P":  ["B_SPLADE_PARTITION", "I_CANON_CONT", "F_PPR_REACH"],
    "U_RET2_NODE": ["B_SPLADE_PARTITION", "I_CANON_CONT", "D_NODE_SPLADE_CONT", "C_NODE_DENSE_CONT"],
    "U_BROAD":   ["B_SPLADE_PARTITION", "I_CANON_CONT", "A_DENSE_PARTITION", "F_PPR_REACH",
                  "G_GRAPH_NBR_RAW"],
    "U_PC4":     ["B_SPLADE_PARTITION", "I_CANON_CONT", "D_NODE_SPLADE_CONT",
                  "G_GRAPH_NBR_RAW"],
    "U_PC5":     ["B_SPLADE_PARTITION", "I_CANON_CONT", "C_NODE_DENSE_CONT",
                  "D_NODE_SPLADE_CONT", "G_GRAPH_NBR_RAW"],
    "U_ALL":     list(CG.FAMS),
}
# families that consult NO graph structure at query time (pure cached retrieval rankings)
GRAPH_FREE = {"A_DENSE_PARTITION", "B_SPLADE_PARTITION", "C_NODE_DENSE_CONT",
              "D_NODE_SPLADE_CONT", "I_CANON_CONT"}
# families served entirely by corpus-side precomputed tables (lookup + merge, 0 online edges)
PRECOMPUTED = GRAPH_FREE | {"G_GRAPH_NBR_RAW", "G_GRAPH_NBR_NORM", "H_STRUCT_FRONTIER"}


def rr(O, fams, qi, M):
    """round-robin interleave, dedup, cut at M.  Parameter-free and corpus-independent."""
    out, seen, i = [], set(), 0
    ls = [O[f][qi] for f in fams]
    while len(out) < M:
        prog = False
        for l in ls:
            if i < len(l):
                prog = True
                p = l[i]
                if p not in seen:
                    seen.add(p); out.append(p)
                    if len(out) >= M:
                        return out
        if not prog:
            break
        i += 1
    return out


def slices(ds, S):
    out = {"ALL": np.ones(S["nq"], bool)}
    if ds == "metaqa":
        for h in (1, 2, 3):
            out[f"hop{h}"] = S["hops"] == h
    return out


def report(S, SL, base_ind, extra, Bv=CG.B):
    ind, churn = CG.run_frozen(S, extra, Bv)
    orc = CG.pool_oracle(S, S["T"], extra, Bv)
    r = {"mean_proposals": round(float(np.mean([len(e) for e in extra])), 1) if extra else 0.0,
         "mean_churn": round(float(churn.mean()), 3)}
    for k, m in SL.items():
        a, b = base_ind[m], ind[m]
        st = PP.mcnemar(b, a)
        r[k] = {"ACTUAL": round(float(b.mean()), 4),
                "delta_vs_SAFE": round(float(b.mean() - a.mean()), 4),
                "POOL_ORACLE": round(float(orc[m].mean()), 4),
                "gained": st["gained"], "lost": st["lost"], "p": st["mcnemar_p"],
                "sig": st["sig"]}
    return r, ind


def run_ds(ds, OUT):
    S = CG.substrate(ds); O = CG.proposals(ds, S)
    SL = slices(ds, S); nq = S["nq"]
    base_ind, base_churn = CG.run_frozen(S, None)
    row = {"SAFE": {k: round(float(base_ind[m].mean()), 4) for k, m in SL.items()},
           "SAFE_mean_churn": round(float(base_churn.mean()), 3),
           "CURRENT_POOL_ORACLE": {k: round(float(CG.pool_oracle(S, S["T"])[m].mean()), 4)
                                   for k, m in SL.items()},
           "STEP5_PPR_PROPOSAL_ONLY": {}, "SINGLE_FAMILY": {}, "STEP7_UNION": {}}
    k0 = "hop3" if ds == "metaqa" else "ALL"
    log(f"   SAFE({k0}) {row['SAFE'][k0]:.4f}  current-pool-oracle {row['CURRENT_POOL_ORACLE'][k0]:.4f}")

    for M in MPPR:                                            # STEP 5
        ex = [O["F_PPR_REACH"][qi][:M] for qi in range(nq)]
        r, _ = report(S, SL, base_ind, ex)
        row["STEP5_PPR_PROPOSAL_ONLY"][str(M)] = r
        log(f"   PPR-prop M={M:<4d} actual {r[k0]['ACTUAL']:.4f} ({r[k0]['delta_vs_SAFE']:+.4f}) "
            f"oracle {r[k0]['POOL_ORACLE']:.4f} +{r[k0]['gained']}/-{r[k0]['lost']} p={r[k0]['p']:.3g}")

    for f in CG.FAMS:                                         # single-family reference at 128
        ex = [O[f][qi][:128] for qi in range(nq)]
        r, _ = report(S, SL, base_ind, ex)
        row["SINGLE_FAMILY"][f] = {"M": 128, **r}

    for name, fams in UNIONS.items():                         # STEP 7
        row["STEP7_UNION"][name] = {"families": fams,
                                    "graph_free": all(f in GRAPH_FREE for f in fams),
                                    "fully_precomputed": all(f in PRECOMPUTED for f in fams)}
        for M in MTOT:
            ex = [rr(O, fams, qi, M) for qi in range(nq)]
            r, _ = report(S, SL, base_ind, ex)
            row["STEP7_UNION"][name][str(M)] = r
            log(f"   {name:<14s} M={M:<4d} actual {r[k0]['ACTUAL']:.4f} ({r[k0]['delta_vs_SAFE']:+.4f}) "
                f"oracle {r[k0]['POOL_ORACLE']:.4f} +{r[k0]['gained']}/-{r[k0]['lost']} p={r[k0]['p']:.3g}")
    OUT[ds] = row


def main():
    os.makedirs(f"{CG.CGD}/diag", exist_ok=True)
    fp = f"{CG.CGD}/diag/run.json"
    OUT = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in (sys.argv[1:] or CG.DSETS):
        log(ds)
        run_ds(ds, OUT)
        json.dump(OUT, open(fp, "w"), indent=1)
    log("wrote run.json")


if __name__ == "__main__":
    main()
