"""PHASES 1-3 -- export the exact partitioner inputs for the Modal CPU runners.

Three representations, so partitioner quality and graph representation stay separable:

  G0_TOPOLOGY_C          S u N u K, unweighted -- bit-for-bit what production METIS saw
  G1_LOCAL_HYPER_CLIQUE  the P4_CE_LOCAL_ONLY graph: local closures clique-expanded with the
                         canonical 1/(|e|-1) weight, quantised exactly as pymetis received it
  G2_TRUE_HYPERGRAPH     the SAME local closures as genuine hyperedges -- {v} u N(v) handed to
                         Mt-KaHyPar whole, instead of being shattered into pairwise cliques

G2 is the point of the exercise.  The previous program could only approximate {A,B,C,D,E} with
ten pairwise edges; a real hypergraph partitioner takes the group.

LIMITATION, recorded rather than worked around: the H_NER family cannot be exported as true
hyperedges.  Only `ner_edges_w_df25.pkl` survives and it is ALREADY clique-expanded with 1/df
weights -- the entity -> document groups that produced it are not on disk, and re-extracting
them would be a new NER build.  This costs nothing for the decisive comparison, because the
winning recipe P4_CE_LOCAL_ONLY contains no NER family at all: it is exactly H_STRUCT_LOCAL +
H_KNN_LOCAL, both of which reconstruct exactly from the frozen CSRs.

k and the balance tolerance are the production values; nothing here is tuned.

  python scratchpad/_l1ov_export.py [ds ...]
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1kn_sub as KS
import _l1ep_part as PP
import _l1ep_pu as PU

OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_UNIVERSAL_PARTITION_SEARCH"
EXP = "scratchpad/_l1ov/graphs"
log = PP.log
HYPER_CAP = PP.HYPER_CAP


def true_hyperedges(N, keys, cap=HYPER_CAP):
    """closed neighbourhoods {v} u N(v) as genuine hyperedges, same cap as the clique expansion.

    Identical membership to what hyper_clique() expands, so G1 and G2 differ ONLY in whether the
    group is shattered into pairs -- which is precisely the question Phase 3 asks.
    """
    xadj, adj = PP._csr_from_keys(keys, N)
    deg = np.diff(xadj).astype(np.int64)
    keep = np.nonzero((deg >= 1) & (deg < cap))[0]
    sizes = deg[keep] + 1
    eptr = np.zeros(len(keep) + 1, np.int64)
    eptr[1:] = np.cumsum(sizes)
    eidx = np.empty(int(eptr[-1]), np.int64)
    for t, v in enumerate(keep):
        s = eptr[t]
        eidx[s] = v
        eidx[s + 1:eptr[t + 1]] = adj[xadj[v]:xadj[v + 1]]
    return eptr, eidx, {"hyperedges": int(len(keep)), "pins": int(len(eidx)),
                        "size_mean": round(float(sizes.mean()), 2),
                        "size_p90": float(np.percentile(sizes, 90)),
                        "size_p99": float(np.percentile(sizes, 99)),
                        "size_max": int(sizes.max()),
                        "dropped_over_cap": int(((deg >= cap)).sum()), "cap": int(cap)}


def export(ds, log=log):
    os.makedirs(EXP, exist_ok=True)
    hard, npart = PU.load_assignment(ds, "CURRENT")
    N, S, K, X = KS.keysets(ds, log)
    k = int(npart)
    meta = {"ds": ds, "N": int(N), "k": k,
            "production_npart": int(npart), "ufactor": 30,
            "families": {"STRUCT": int(len(S)), "KNN": int(len(K)), "NERX": int(len(X))}}

    # ---- G0: exactly the production METIS input (topology C, unweighted)
    C = np.unique(np.concatenate([S, X, K]))
    ptr, idx, deg = KS._csr(C, N)
    np.savez_compressed(f"{EXP}/{ds}__G0_TOPOLOGY_C.npz", ptr=ptr.astype(np.int64),
                        idx=idx.astype(np.int32), k=np.array([k]))
    meta["G0_TOPOLOGY_C"] = {"undirected_edges": int(len(C)),
                             "directed": int(ptr[-1]), "isolated": int((deg == 0).sum())}
    log(f"  {ds} G0_TOPOLOGY_C: {len(C):,} undirected edges")

    # ---- G1: the winning clique-expanded weighted graph, built by the frozen code path
    uk, q = PP.hyper_clique(ds, N, S, K, X, fams=("H_STRUCT_LOCAL", "H_KNN_LOCAL"),
                            weighted=True, log=log)
    p1, i1, w1 = PP._csr_weighted(uk, q, N)
    np.savez_compressed(f"{EXP}/{ds}__G1_LOCAL_HYPER_CLIQUE.npz", ptr=p1.astype(np.int64),
                        idx=i1.astype(np.int32), w=w1.astype(np.int32), k=np.array([k]))
    meta["G1_LOCAL_HYPER_CLIQUE"] = {"undirected_edges": int(len(uk)),
                                     "weight_min": int(q.min()), "weight_max": int(q.max())}
    log(f"  {ds} G1_LOCAL_HYPER_CLIQUE: {len(uk):,} weighted pairs")
    del uk, q, p1, i1, w1

    # ---- G2: the same local closures, NOT shattered
    hm = {}
    eptrs, eidxs, off = [], [], 0
    for fam, keys in (("H_STRUCT_LOCAL", S), ("H_KNN_LOCAL", K)):
        ep, ei, st = true_hyperedges(N, keys)
        hm[fam] = st
        eptrs.append(ep[1:] + off); eidxs.append(ei); off += int(ep[-1])
        log(f"  {ds} {fam}: {st['hyperedges']:,} hyperedges, {st['pins']:,} pins, "
            f"mean {st['size_mean']} p99 {st['size_p99']} max {st['size_max']}")
    eptr = np.concatenate([[0]] + eptrs).astype(np.int64)
    eidx = np.concatenate(eidxs).astype(np.int64)
    np.savez_compressed(f"{EXP}/{ds}__G2_TRUE_HYPERGRAPH__hyper.npz", eptr=eptr, eidx=eidx,
                        k=np.array([k]))
    # Mt-KaHyPar still needs a vertex count carrier alongside the hypergraph
    np.savez_compressed(f"{EXP}/{ds}__G2_TRUE_HYPERGRAPH.npz", ptr=ptr.astype(np.int64),
                        idx=idx.astype(np.int32), k=np.array([k]))
    hm["TOTAL"] = {"hyperedges": int(len(eptr) - 1), "pins": int(len(eidx))}
    meta["G2_TRUE_HYPERGRAPH"] = hm
    meta["H_NER_AS_TRUE_HYPEREDGES"] = ("UNAVAILABLE -- ner_edges_w_df25.pkl is already "
                                        "clique-expanded and the entity groups are not on disk; "
                                        "the winning P4_CE_LOCAL_ONLY recipe contains no NER "
                                        "family, so the decisive comparison is unaffected")
    log(f"  {ds} G2_TRUE_HYPERGRAPH: {len(eptr)-1:,} hyperedges, {len(eidx):,} pins")
    return meta


if __name__ == "__main__":
    os.makedirs(f"{OUT}/hypergraph", exist_ok=True)
    fp = f"{OUT}/hypergraph/EXPORT_MANIFEST.json"
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    for ds in (sys.argv[1:] or list(KS.DS)):
        rec[ds] = export(ds)
        json.dump(rec, open(fp, "w"), indent=1)
    log("wrote", fp)
