"""FINAL L1 EDGE-SUBSTRATE + PARTITION UTILITY PROGRAM -- A0/A1/A2: edge algebra and substrates.

Canonical edge families (undirected, canonical key min(u,v)*N + max(u,v), self-loops dropped):

    E_STRUCT = S   master_nodes_{ds}.json `neighbors`  -- and ONLY this is traversed by frozen L1
    E_NERX   = N   ner_edges_w_df25.pkl  MINUS S
    E_KNN    = K   gte_qwen/graph.pt     MINUS S          (A = S u K by construction)

S n N = 0 and S n K = 0 hold by construction.  N n K is NOT empty -- a Qwen neighbour can also be
an entity co-mention -- so provenance is a BITMASK (FAMBIT), never a partition of the edges.

Canonical substrates (A1, full-work) and matched-work controls (A2):

  E0_STRUCT              S                 == frozen traversal CSR, bit-exact (hard gate)
  E1_NERX_ONLY           N
  E2_KNN_ONLY            K
  E3_STRUCT_NERX         S u N
  E4_STRUCT_KNN          S u K
  E5_NERX_KNN            N u K             diagnostic (no structural edges at all)
  E6_TOPOLOGY_C          S u N u K         what topology C actually is
  M0_STRUCT              = E0_STRUCT       the matched-work baseline
  M1_STRUCT_NERX_MATCHED S/N   reallocated inside E0's OWN per-node budget
  M2_STRUCT_KNN_MATCHED  S/K   reallocated inside E0's OWN per-node budget
  M3_TOPOLOGY_C_MATCHED  S/(NuK) reallocated inside E0's OWN per-node budget

The matched rule is ONE deterministic schedule, identical for M1/M2/M3, already implemented in the
kNN audit and preserved here: node i keeps EXACTLY deg_S(i) entries, of which
    n_new = min(deg_new(i), deg_S(i) // 2)
come from the new family (in ascending neighbour id, the frozen adjacency order) and the remaining
deg_S(i) - n_new are its first structural neighbours.  `adj_ptr` and `deg` are therefore E0's arrays
UNCHANGED, so DEG_CAP behaves identically and every frontier node costs exactly the edge
inspections it costs in E0 -- matched by construction, not by tuning.

Old T-names from the kNN audit map onto the canonical names and their caches are REUSED verbatim:
  T0_FROZEN->E0_STRUCT/M0_STRUCT  T4_NERX_ONLY->E1_NERX_ONLY  T1_KNN_ONLY->E2_KNN_ONLY
  T2_FULL_UNION->E4_STRUCT_KNN    T5_TOPOLOGY_C->E6_TOPOLOGY_C  T3_MATCHED_HYBRID->M2_..._MATCHED

  python scratchpad/_l1ep_sub.py [ds ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1kn_sub as KS

CACHE = "scratchpad/_l1ep"
OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_EDGE_AND_PARTITION_FINAL"
DS = KS.DS
FAMBIT = KS.FAMBIT
FNAME = KS.FNAME

E_SUBS = ["E0_STRUCT", "E1_NERX_ONLY", "E2_KNN_ONLY", "E3_STRUCT_NERX", "E4_STRUCT_KNN",
          "E5_NERX_KNN", "E6_TOPOLOGY_C"]
M_SUBS = ["M0_STRUCT", "M1_STRUCT_NERX_MATCHED", "M2_STRUCT_KNN_MATCHED", "M3_TOPOLOGY_C_MATCHED"]
SUBS = E_SUBS + M_SUBS
# canonical -> already-computed kNN-audit cache (no recompute for these six)
ALIAS = {"E0_STRUCT": "T0_FROZEN", "M0_STRUCT": "T0_FROZEN", "E1_NERX_ONLY": "T4_NERX_ONLY",
         "E2_KNN_ONLY": "T1_KNN_ONLY", "E4_STRUCT_KNN": "T2_FULL_UNION",
         "E6_TOPOLOGY_C": "T5_TOPOLOGY_C", "M2_STRUCT_KNN_MATCHED": "T3_MATCHED_HYBRID"}
NEWFAM = {"M1_STRUCT_NERX_MATCHED": "NERX", "M2_STRUCT_KNN_MATCHED": "KNN",
          "M3_TOPOLOGY_C_MATCHED": "NERX_KNN"}

T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)
keysets = KS.keysets
_csr = KS._csr
_ukeys = KS._ukeys


def substrate(ds, name, log=log):
    """(adjp, adji, deg, meta) for one canonical substrate.  Cached; aliases reuse the audit cache."""
    assert name in SUBS, name
    if name in ALIAS:
        p, i, d, m = KS.substrate(ds, ALIAS[name], log)
        m = dict(m, sub=name, alias_of=ALIAS[name])
        return p, i, d, m
    os.makedirs(CACHE, exist_ok=True)
    fp = f"{CACHE}/sub_{ds}__{name}.npz"
    if os.path.exists(fp):
        z = np.load(fp, allow_pickle=True)
        return z["ptr"], z["idx"], z["deg"], json.loads(str(z["meta"]))
    N, S, K, X = keysets(ds, log)
    nkept = None
    if name in NEWFAM:
        ps, ais, dsg = KS.substrate(ds, "T0_FROZEN", log)[:3]
        fam = NEWFAM[name]
        NEW = X if fam == "NERX" else (K if fam == "KNN" else np.union1d(X, K))
        pn, ain, _ = _csr(NEW, N)
        idx, nkept = KS._interleave(ps, ais, pn, ain, N)
        ptr, deg = ps, dsg
    else:
        KEY = {"E3_STRUCT_NERX": np.union1d(S, X), "E5_NERX_KNN": np.union1d(X, K)}[name]
        ptr, idx, deg = _csr(KEY, N)
    meta = {"ds": ds, "sub": name, "N": int(N), "directed_edges": int(ptr[-1]),
            "undirected_edges": int(ptr[-1]) // 2, "mean_deg": round(float(deg.mean()), 3),
            "max_deg": int(deg.max()), "isolated_nodes": int((deg == 0).sum()),
            "nodes_over_DEG_CAP": int((deg > 300).sum())}
    if nkept is not None:
        meta["new_family"] = NEWFAM[name]
        meta["new_slots_taken"] = int(nkept)
        meta["frac_slots_new"] = round(nkept / max(int(ptr[-1]), 1), 4)
    np.savez_compressed(fp, ptr=ptr, idx=idx, deg=deg, meta=json.dumps(meta))
    return ptr, idx, deg, meta


def edge_algebra(ds, log=log):
    """A0.  Every pairwise / union cardinality from source artifacts, plus the topology-C identity."""
    import torch, pickle
    N, S, K, X = keysets(ds, log)
    g = torch.load(f"data/ukb_storage/{ds}/gte_qwen/graph.pt", map_location="cpu",
                   weights_only=False)
    A = _ukeys(g.edge_index[0].numpy(), g.edge_index[1].numpy(), N)
    del g
    ner = pickle.load(open(f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl", "rb")).tocoo()
    NER = _ukeys(ner.row, ner.col, N)
    del ner
    C = torch.load(f"scratchpad/ablation_qwen/{ds}/variant_C/graph.pt", map_location="cpu",
                   weights_only=False)
    CK = _ukeys(C.edge_index[0].numpy(), C.edge_index[1].numpy(), N)
    del C
    U3 = np.union1d(np.union1d(S, X), K)
    ix = lambda a, b: int(len(np.intersect1d(a, b, assume_unique=True)))
    r = {"ds": ds, "N": int(N),
         "|E_STRUCT|": int(len(S)), "|E_NERX|": int(len(X)), "|E_KNN|": int(len(K)),
         "|NER_raw|": int(len(NER)), "|A_gte_qwen|": int(len(A)),
         "|S n N|": ix(S, X), "|S n K|": ix(S, K), "|N n K|": ix(X, K),
         "|S n NER_raw|": ix(S, NER), "|S u N|": int(len(np.union1d(S, X))),
         "|S u K|": int(len(np.union1d(S, K))), "|N u K|": int(len(np.union1d(X, K))),
         "|S u N u K|": int(len(U3)), "|topology_C_file|": int(len(CK)),
         "S_u_K_equals_A": bool(len(np.union1d(S, K)) == len(A) and
                                np.array_equal(np.union1d(S, K), A)),
         "S_in_A": int(len(S) - ix(S, A)), "SuNuK_equals_C_file": bool(np.array_equal(U3, CK)),
         "SuNuK_minus_C_file": int(len(np.setdiff1d(U3, CK, assume_unique=True))),
         "C_file_minus_SuNuK": int(len(np.setdiff1d(CK, U3, assume_unique=True))),
         "frac_C_never_traversed": round(1.0 - len(S) / max(len(U3), 1), 4)}
    r["|S n N|_is_zero"] = (r["|S n N|"] == 0)
    r["|S n K|_is_zero"] = (r["|S n K|"] == 0)
    return r


if __name__ == "__main__":
    todo = sys.argv[1:] or DS
    os.makedirs(f"{OUT}/EDGE_SUBSTRATE", exist_ok=True)
    fp = f"{OUT}/EDGE_SUBSTRATE/A0_A1_A2_substrates.json"
    out = json.load(open(fp)) if os.path.exists(fp) else {}
    for d in todo:
        rec = {"A0_edge_algebra": edge_algebra(d), "A1_A2_substrates": {}}
        log(d, "A0", json.dumps({k: v for k, v in rec["A0_edge_algebra"].items() if k != "ds"}))
        for s in SUBS:
            m = substrate(d, s)[3]
            rec["A1_A2_substrates"][s] = m
            log(f"    {s:24s} und={m['undirected_edges']:>11,d} mean_deg={m['mean_deg']:>7.2f} "
                f"max={m['max_deg']:>7,d} iso={m['isolated_nodes']:>8,d} "
                f"cap={m['nodes_over_DEG_CAP']:>6,d}"
                + (f" new_slots={m['frac_slots_new']:.3f}" if "frac_slots_new" in m else ""))
        out[d] = rec
        json.dump(out, open(fp, "w"), indent=1)
    print("wrote", fp)
