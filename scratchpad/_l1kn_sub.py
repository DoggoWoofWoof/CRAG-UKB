"""L1 QWEN-KNN EDGE SUBSTRATE AUDIT -- STEP 1: the traversal substrates.

The provenance audit established `frozen traversal edges  n  Qwen-kNN edges = 0` on all six corpora.
The reason is structural: `_ta_prepartition.load_topology` builds the CSR from
`master_nodes_{ds}.json` `neighbors`, which is the STRUCT family alone.  Topology C is
`A u NER = (STRUCT u KNN) u NER`, so TWO precomputed families never participate in L1 traversal:

    KNN   = A \\ STRUCT            the Qwen semantic family     (the phase's subject)
    NERX  = NER \\ STRUCT          the entity family's non-structural part  (a necessary control:
            without it a T2 gain cannot be attributed to SEMANTICS rather than to MORE EDGES)

Substrates, all in the same doc-row index space, all cached as CSR:

  T0_FROZEN          STRUCT                       exactly `TA.load_topology` -- bit-identical
  T1_KNN_ONLY        KNN
  T2_FULL_UNION      STRUCT u KNN                 = A
  T3_MATCHED_HYBRID  per-node reallocation of T0's OWN budget between STRUCT and KNN
  T4_NERX_ONLY       NERX                         control family
  T5_TOPOLOGY_C      STRUCT u KNN u NERX          what topology C was believed to be

T3 is the matched-work control and is exact by construction, not by tuning: node i keeps a list of
EXACTLY `deg_struct(i)` entries -- `n_k = min(deg_knn(i), deg_struct(i)//2)` kNN neighbours and
`n_s = deg_struct(i) - n_k` structural ones (equal split, remainder to STRUCT; the unique
parameter-free schedule).  `adj_ptr` and `deg` are therefore T0's arrays UNCHANGED, so DEG_CAP
behaves identically and every frontier node costs the same number of edge inspections it costs in
T0.  Which structural neighbours are dropped is decided by node id -- the frozen adjacency order,
which is content-independent.

No new graph is computed anywhere here: every key comes from a file already on disk.

  python scratchpad/_l1kn_sub.py [ds ...]
"""
import os, sys, json, pickle, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np

CACHE = "scratchpad/_l1kn"
KND = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_KNN"
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
SUBS = ["T0_FROZEN", "T1_KNN_ONLY", "T2_FULL_UNION", "T3_MATCHED_HYBRID", "T4_NERX_ONLY",
        "T5_TOPOLOGY_C"]
FAMBIT = {"STRUCT": 1, "KNN": 2, "NERX": 4}
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def _ukeys(u, v, N):
    u = np.asarray(u, np.int64); v = np.asarray(v, np.int64)
    m = u != v
    u, v = u[m], v[m]
    return np.unique(np.minimum(u, v) * np.int64(N) + np.maximum(u, v))


def keysets(ds, log=log):
    """(N, STRUCT, KNN, NERX) as sorted undirected int64 key arrays.  Cached."""
    os.makedirs(CACHE, exist_ok=True)
    fp = f"{CACHE}/keys_{ds}.npz"
    if os.path.exists(fp):
        z = np.load(fp)
        return int(z["N"][0]), z["STRUCT"], z["KNN"], z["NERX"]
    import torch
    import _ta_prepartition as TA
    hard, _, _, (ap, ai), _, _ = TA.load_topology(ds, lambda *a: None)
    N = len(hard)
    src = np.repeat(np.arange(N, dtype=np.int64), np.diff(ap))
    STRUCT = _ukeys(src, ai, N)
    g = torch.load(f"data/ukb_storage/{ds}/gte_qwen/graph.pt", map_location="cpu",
                   weights_only=False)
    A = _ukeys(g.edge_index[0].numpy(), g.edge_index[1].numpy(), N)
    del g
    ner = pickle.load(open(f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl", "rb")).tocoo()
    NER = _ukeys(ner.row, ner.col, N)
    del ner
    KNN = np.setdiff1d(A, STRUCT, assume_unique=True)
    NERX = np.setdiff1d(NER, STRUCT, assume_unique=True)
    np.savez_compressed(fp, N=np.array([N]), STRUCT=STRUCT, KNN=KNN, NERX=NERX)
    log(f"  {ds}: N={N} STRUCT={len(STRUCT):,} KNN={len(KNN):,} NERX={len(NERX):,} "
        f"(A={len(A):,} NER={len(NER):,})")
    return N, STRUCT, KNN, NERX


def raw_ner_keys(ds, log=log):
    """Canonical (NON-residualized) NER key array -- STRUCT-overlapping pairs INCLUDED.

    `keysets()` computes this same raw NER array internally (as `ner`/`NER` before the
    `setdiff1d` step) but only ever caches+returns the residualized NERX = NER \\ STRUCT.
    `_l1hu_build.py`'s "NER" partition-build family had been silently reusing that residualized
    array (via `keysets()`'s 4th return value) -- this function exists so partition-BUILD can use
    genuine canonical NER instead, per the family's own name.  Cached separately so it never
    touches `keysets()`'s frozen 4-tuple cache/contract that other scripts depend on.
    """
    os.makedirs(CACHE, exist_ok=True)
    fp = f"{CACHE}/keys_raw_ner_{ds}.npz"
    if os.path.exists(fp):
        z = np.load(fp)
        return int(z["N"][0]), z["NER_RAW"]
    import _ta_prepartition as TA
    hard, _, _, _, _, _ = TA.load_topology(ds, lambda *a: None)
    N = len(hard)
    ner = pickle.load(open(f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl", "rb")).tocoo()
    NER_RAW = _ukeys(ner.row, ner.col, N)
    np.savez_compressed(fp, N=np.array([N]), NER_RAW=NER_RAW)
    log(f"  {ds}: raw NER (non-residualized) N={N} NER_RAW={len(NER_RAW):,}")
    return N, NER_RAW


def _csr(keys, N):
    """symmetrised CSR (sorted neighbour ids) from undirected keys."""
    u = (keys // np.int64(N)).astype(np.int64)
    v = (keys % np.int64(N)).astype(np.int64)
    r = np.concatenate([u, v]); c = np.concatenate([v, u])
    o = np.lexsort((c, r))
    r, c = r[o], c[o]
    deg = np.bincount(r, minlength=N).astype(np.int32)
    ptr = np.zeros(N + 1, np.int64); ptr[1:] = np.cumsum(deg)
    return ptr, c.astype(np.int32), deg


def _interleave(ps, ais, pk, aik, N):
    """T3: per node keep n_s STRUCT + n_k KNN with n_k = min(deg_knn, deg_struct//2).  Same ptr."""
    out = np.empty(int(ps[-1]), np.int32)
    ds_ = (np.diff(ps)).astype(np.int64)
    dk_ = (np.diff(pk)).astype(np.int64)
    nk = np.minimum(dk_, ds_ // 2)
    ns = ds_ - nk
    for i in range(N):
        L = int(ds_[i])
        if L == 0:
            continue
        a, b = int(ns[i]), int(nk[i])
        o = int(ps[i])
        if b == 0:
            out[o:o + L] = ais[o:o + L]
            continue
        out[o:o + a] = ais[o:o + a]
        out[o + a:o + L] = aik[int(pk[i]):int(pk[i]) + b]
    return out, int(nk.sum())


def substrate(ds, name, log=log):
    """(adjp, adji, deg, meta) for one traversal substrate.  Cached."""
    assert name in SUBS, name
    os.makedirs(CACHE, exist_ok=True)
    fp = f"{CACHE}/sub_{ds}__{name}.npz"
    if os.path.exists(fp):
        z = np.load(fp, allow_pickle=True)
        return z["ptr"], z["idx"], z["deg"], json.loads(str(z["meta"]))
    N, STRUCT, KNN, NERX = keysets(ds, log)
    if name == "T0_FROZEN":
        import _ta_prepartition as TA
        _, _, _, (ap, ai), dg, _ = TA.load_topology(ds, lambda *a: None)
        ptr, idx, deg = np.asarray(ap), np.asarray(ai), np.asarray(dg)
        # the frozen CSR must be reproducible from the key set alone, or the family split is wrong
        p2, i2, d2 = _csr(STRUCT, N)
        assert (deg == d2).all() and (idx == i2).all() and (ptr == p2).all(), \
            "frozen CSR != CSR(STRUCT keys)"
    elif name == "T3_MATCHED_HYBRID":
        ps, ais, dsg = substrate(ds, "T0_FROZEN", log)[:3]
        pk, aik, _ = _csr(KNN, N)
        idx, nkept = _interleave(ps, ais, pk, aik, N)
        ptr, deg = ps, dsg
    else:
        K = {"T1_KNN_ONLY": KNN, "T2_FULL_UNION": np.union1d(STRUCT, KNN),
             "T4_NERX_ONLY": NERX,
             "T5_TOPOLOGY_C": np.union1d(np.union1d(STRUCT, KNN), NERX)}[name]
        ptr, idx, deg = _csr(K, N)
    meta = {"ds": ds, "sub": name, "N": int(N), "directed_edges": int(ptr[-1]),
            "undirected_edges": int(ptr[-1]) // 2,
            "mean_deg": round(float(deg.mean()), 3), "max_deg": int(deg.max()),
            "isolated_nodes": int((deg == 0).sum()),
            "nodes_over_DEG_CAP": int((deg > 300).sum())}
    if name == "T3_MATCHED_HYBRID":
        meta["knn_slots_taken"] = int(nkept)
        meta["frac_slots_knn"] = round(nkept / max(int(ptr[-1]), 1), 4)
    np.savez_compressed(fp, ptr=ptr, idx=idx, deg=deg, meta=json.dumps(meta))
    return ptr, idx, deg, meta


class Lab:
    """family bitmask for an arbitrary (u, v) array -- used by STEP 7 path provenance."""

    def __init__(self, ds, log=log):
        self.N, S, K, X = keysets(ds, log)
        self.k = np.concatenate([S, K, X])
        self.b = np.concatenate([np.full(len(S), FAMBIT["STRUCT"], np.uint8),
                                 np.full(len(K), FAMBIT["KNN"], np.uint8),
                                 np.full(len(X), FAMBIT["NERX"], np.uint8)])
        o = np.argsort(self.k, kind="stable")
        self.k, self.b = self.k[o], self.b[o]
        # STRUCT is disjoint from both others by construction, but KNN n NERX is NOT empty: an edge
        # can be both a Qwen neighbour and an entity co-mention.  Merge duplicates into one bitmask
        # rather than letting the first one win.
        uk, first = np.unique(self.k, return_index=True)
        if len(uk) != len(self.k):
            mb = np.zeros(len(uk), np.uint8)
            np.bitwise_or.at(mb, np.searchsorted(uk, self.k), self.b)
            self.k, self.b = uk, mb
        self.dup = int(len(o) - len(self.k))

    def of(self, u, v):
        u = np.asarray(u, np.int64); v = np.asarray(v, np.int64)
        q = np.minimum(u, v) * np.int64(self.N) + np.maximum(u, v)
        i = np.clip(np.searchsorted(self.k, q), 0, len(self.k) - 1)
        ok = self.k[i] == q
        out = np.zeros(len(q), np.uint8)
        out[ok] = self.b[i[ok]]
        return out, ok


FNAME = {0: "NONE", 1: "STRUCT", 2: "KNN", 3: "STRUCT+KNN", 4: "NERX", 5: "STRUCT+NERX",
         6: "KNN+NERX", 7: "ALL3"}


if __name__ == "__main__":
    todo = sys.argv[1:] or DS
    os.makedirs(f"{KND}/diag", exist_ok=True)
    fp = f"{KND}/diag/step1_substrates.json"
    out = json.load(open(fp)) if os.path.exists(fp) else {}
    for d in todo:
        N, S, K, X = keysets(d)
        rec = {"N": int(N), "STRUCT_undirected": int(len(S)), "KNN_undirected": int(len(K)),
               "NERX_undirected": int(len(X)),
               "KNN_and_NERX_overlap": int(len(np.intersect1d(K, X, assume_unique=True))),
               "frac_topologyC_edges_never_traversed":
                   round(len(np.union1d(K, X)) / max(len(np.union1d(S, np.union1d(K, X))), 1), 4),
               "subs": {}}
        for s in SUBS:
            rec["subs"][s] = substrate(d, s)[3]
        out[d] = rec
        log(d, json.dumps({k: rec[k] for k in rec if k != "subs"}))
        for s in SUBS:
            m = rec["subs"][s]
            log(f"    {s:20s} undirected={m['undirected_edges']:>10,d} mean_deg={m['mean_deg']:>7.2f} "
                f"max={m['max_deg']:>6d} isolated={m['isolated_nodes']:>8,d} "
                f"over_cap={m['nodes_over_DEG_CAP']:>6,d}"
                + (f" knn_slots={m['frac_slots_knn']:.3f}" if "frac_slots_knn" in m else ""))
    json.dump(out, open(fp, "w"), indent=1)
    print("wrote", fp)
