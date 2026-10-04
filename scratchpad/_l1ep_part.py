"""FINAL L1 EDGE-SUBSTRATE + PARTITION UTILITY PROGRAM -- B9/B10/B11: alternative partitionings.

Every assignment produced here obeys the SAME contract as production: k = N // 100 blocks, no
query and no gold label anywhere in the build, deterministic, one universal rule for all corpora.

B9  METIS graph-composition ablation -- what the partitioner is allowed to SEE:
      PM0_STRUCT           METIS on S
      PM1_STRUCT_NERX      METIS on S u N
      PM2_STRUCT_KNN       METIS on S u K
      PM3_TOPOLOGY_C       METIS on S u N u K      (should reproduce production)
      PM_CURRENT_EXACT     the production assignment as shipped

B10 partitioner ablation -- HOW the blocks are cut, holding the graph at topology C:
      P0_RANDOM_BALANCED   random assignment with EXACTLY the production size histogram (n seeds)
      P1_METIS_CURRENT     production
      P2_METIS_STRONG      stronger multilevel search  (KaHIP SUBSTITUTE -- see note)
      P3_FENNEL            streaming FENNEL, gamma = 1.5, alpha = m k^(g-1) / n^g  (Tsourakakis et al.)

B11 P4_HYPERGRAPH_CE     universal hypergraph families, partitioned through CLIQUE EXPANSION
      H_NER              one hyperedge per named entity  (2 <= df <= 25)
      H_STRUCT_LOCAL     closed structural neighbourhood of each node
      H_KNN_LOCAL        closed Qwen-kNN neighbourhood of each node
    with ONE universal size cap for all corpora and all families.

SUBSTITUTION NOTE -- stated, not hidden.  `kahypar` and `KaHIP` have no wheel for this platform
(Windows / CPython 3.13) and neither builds from source here, so `P2_KAHIP_STRONG` and
`P4_KAHYPAR_UNIVERSAL` cannot be run as named.  They are replaced by the closest available
mechanism -- a stronger METIS multilevel search, and hypergraph partitioning via the standard
clique expansion (weight 1/(|e|-1) per pair, which for H_NER is EXACTLY the 1/df weighting the
frozen NER artifact already stores).  The substitution is recorded in every artifact.

  python scratchpad/_l1ep_part.py <ds> [which ...]
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ep_sub as EP

OUT = EP.OUT
PDIR = f"{OUT}/PARTITIONERS"
CACHE = "scratchpad/_l1ep/parts"
TARGET = 100                      # frozen: build_canonical_topo.TARGET
HYPER_CAP = 25                    # frozen: the df cap the NER artifact itself uses (universal)
FENNEL_GAMMA = 1.5                # canonical FENNEL exponent (Tsourakakis et al. 2014)
RANDOM_SEEDS = [0, 1, 2, 3, 4]
METIS_SEEDS = [1, 2, 3, 4]        # B15: same graph, same algorithm, different RNG -> noise floor
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

ALL = ["PM0_STRUCT", "PM1_STRUCT_NERX", "PM2_STRUCT_KNN", "PM3_TOPOLOGY_C",
       "P2_METIS_STRONG", "P3_FENNEL", "P4_HYPERGRAPH_CE"] + \
      [f"P0_RANDOM_BALANCED_s{s}" for s in RANDOM_SEEDS]


def _csr_from_keys(keys, N):
    u = (keys // np.int64(N)).astype(np.int64); v = (keys % np.int64(N)).astype(np.int64)
    r = np.concatenate([u, v]); c = np.concatenate([v, u])
    o = np.lexsort((c, r)); r, c = r[o], c[o]
    deg = np.bincount(r, minlength=N).astype(np.int64)
    xadj = np.zeros(N + 1, np.int64); xadj[1:] = np.cumsum(deg)
    return xadj, c


def _csr_weighted(keys, w, N):
    """symmetric CSR with integer edge weights (pymetis requires ints > 0)."""
    u = (keys // np.int64(N)).astype(np.int64); v = (keys % np.int64(N)).astype(np.int64)
    r = np.concatenate([u, v]); c = np.concatenate([v, u]); ww = np.concatenate([w, w])
    o = np.lexsort((c, r)); r, c, ww = r[o], c[o], ww[o]
    deg = np.bincount(r, minlength=N).astype(np.int64)
    xadj = np.zeros(N + 1, np.int64); xadj[1:] = np.cumsum(deg)
    return xadj, c, ww


def metis(N, xadj, adjncy, k, eweights=None, strong=False, seed=None):
    """pymetis on numpy CSR (never Python lists -- 17M adjacency entries would not fit).

    Defaults are METIS's own, which is what production used: k-way (pymetis picks recursive only
    for k <= 8), objective = edge cut, ufactor = 30 (1.03 imbalance), unweighted vertices."""
    import pymetis
    kw = {}
    if eweights is not None:
        kw["eweights"] = np.asarray(eweights, np.int32)
    if strong or seed is not None:
        o = pymetis.Options()
        if strong:
            o.niter = 100      # refinement iterations per level   (METIS default 10)
            o.ncuts = 8        # independent partitionings, best kept (METIS default 1)
        if seed is not None:
            o.seed = int(seed)
        kw["options"] = o
    _, mem = pymetis.part_graph(
        k, adjacency=pymetis.CSRAdjacency(np.asarray(xadj, np.int64),
                                          np.asarray(adjncy, np.int32)), **kw)
    return np.asarray(mem, np.int64)


def fennel(N, xadj, adjncy, k):
    """streaming FENNEL: assign v to the block maximising  |N(v) n P| - alpha*g*|P|^(g-1).

    Node order is the frozen doc-row order (content-independent).  A hard capacity of
    ceil(1.03*N/k) reproduces the METIS balance tolerance, so the result is directly comparable."""
    a = FENNEL_GAMMA
    m = len(adjncy) // 2
    alpha = m * (k ** (a - 1)) / (N ** a)
    cap = int(np.ceil(1.03 * N / k))
    part = np.full(N, -1, np.int64)
    size = np.zeros(k, np.int64)
    cnt = np.zeros(k, np.float64)
    for v in range(N):
        s, e = int(xadj[v]), int(xadj[v + 1])
        nb = adjncy[s:e]
        touched = np.empty(0, np.int64)
        if e > s:
            pn = part[nb]
            pn = pn[pn >= 0]
            if len(pn):
                touched, c = np.unique(pn, return_counts=True)
                cnt[touched] = c
        room = size < cap
        if not room.any():
            room = np.ones(k, bool)
        score = cnt - alpha * a * np.power(size, a - 1.0)
        score[~room] = -np.inf
        b = int(np.argmax(score))
        part[v] = b; size[b] += 1
        if len(touched):
            cnt[touched] = 0.0
        if (v + 1) % 200000 == 0:
            log(f"   fennel {v+1}/{N}")
    return part


def random_balanced(N, sizes, seed):
    """random assignment with EXACTLY the given size histogram -- the B6 control."""
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    rng.shuffle(lab)
    return lab


def _reduce(kk, ww):
    """merge (key, weight) fragments into one sorted, deduplicated pair -- bounded peak memory."""
    k = np.concatenate(kk); w = np.concatenate(ww)
    o = np.argsort(k, kind="stable"); k, w = k[o], w[o]
    if len(k):
        first = np.empty(len(k), bool); first[0] = True
        np.not_equal(k[1:], k[:-1], out=first[1:])
        idx = np.cumsum(first) - 1
        uk = k[first]
        uw = np.zeros(len(uk), np.float32)
        np.add.at(uw, idx, w)
        return uk, uw
    return k, w


def hyper_clique(ds, N, S, K, X, cap=HYPER_CAP, chunk=12_000_000, fams=("H_NER", "H_STRUCT_LOCAL",
                 "H_KNN_LOCAL"), weighted=True, log=log):
    """B11: clique expansion of the three universal hyperedge families -> ONE weighted graph.

    H_NER is already stored clique-expanded with weight 1/df by the frozen NER artifact (every
    entity with 2 <= df <= 25 contributes 1/df to each pair of its documents), so it is consumed
    verbatim -- the identical object, with zero re-extraction.
    H_STRUCT_LOCAL / H_KNN_LOCAL are the closed neighbourhoods N[v], clique-expanded with the
    canonical 1/(|e|-1) weight and dropped when |e| > cap.  ONE universal cap, every corpus.
    Expansion is done degree-block by degree-block (all hyperedges of one size at once) and reduced
    whenever the pending buffer passes `chunk`, so peak memory never tracks the total pair count."""
    import pickle
    kk, ww, nH, pend = [], [], {}, 0
    if "H_NER" in fams:
        A = pickle.load(open(f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl", "rb")).tocoo()
        m = A.row < A.col
        kk.append(A.row[m].astype(np.int64) * np.int64(N) + A.col[m].astype(np.int64))
        ww.append(A.data[m].astype(np.float32))
        nH["H_NER"] = int(m.sum())
        del A, m
        pend = len(kk[0])
    for fam, keys in (("H_STRUCT_LOCAL", S), ("H_KNN_LOCAL", K)):
        if fam not in fams:
            continue
        xadj, adj = _csr_from_keys(keys, N)
        deg = np.diff(xadj).astype(np.int64)
        nH[fam] = 0
        for d in range(1, cap):
            vs = np.nonzero(deg == d)[0]
            if not len(vs):
                continue
            nH[fam] += len(vs)
            M = np.empty((len(vs), d + 1), np.int64)
            M[:, 0] = vs
            off = xadj[vs]
            for t in range(d):
                M[:, t + 1] = adj[off + t]
            ii, jj = np.triu_indices(d + 1, 1)
            a = M[:, ii].ravel(); b = M[:, jj].ravel()
            lo = np.minimum(a, b); hi = np.maximum(a, b)
            sel = lo != hi
            kk.append(lo[sel] * np.int64(N) + hi[sel])
            ww.append(np.full(int(sel.sum()), np.float32(1.0 / d)))
            pend += int(sel.sum())
            del M, a, b, lo, hi, sel
            if pend > chunk:
                uk, uw = _reduce(kk, ww)
                kk, ww, pend = [uk], [uw], len(uk)
        log(f"   hyper {fam}: {nH[fam]:,} hyperedges, buffer {pend:,}")
    uk, uw = _reduce(kk, ww)
    if not weighted:
        uw = np.ones(len(uk), np.float32)
    q = np.maximum(np.rint(uw / max(float(uw.max()), 1e-12) * 1000.0), 1).astype(np.int32)
    log(f"   hyper families {json.dumps(nH)} -> {len(uk):,} weighted pairs")
    return uk, q


def build(ds, which, log=log):
    os.makedirs(CACHE, exist_ok=True)
    fp = f"{CACHE}/{ds}__{which}.npy"
    if os.path.exists(fp):
        return np.load(fp)
    N, S, K, X = EP.keysets(ds, log)
    k = max(1, N // TARGET)
    t = time.time()
    if which.startswith("P0_RANDOM_BALANCED_s"):
        import _ta_prepartition as TA
        hard, _, npart, _, _, _ = TA.load_topology(ds, lambda *a: None)
        sizes = np.bincount(np.asarray(hard, np.int64), minlength=npart)
        mem = random_balanced(N, sizes, int(which.split("_s")[1]))
    elif which == "PM4_TOPOLOGY_C_NERW":
        import pickle
        A = pickle.load(open(f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl", "rb")).tocoo()
        m = A.row < A.col
        nk = A.row[m].astype(np.int64) * np.int64(N) + A.col[m].astype(np.int64)
        nw = A.data[m].astype(np.float32)
        del A, m
        C3 = np.union1d(np.union1d(S, X), K)
        w = np.ones(len(C3), np.float32)                      # struct / kNN keep unit weight
        pos = np.searchsorted(C3, nk)
        ok = (pos < len(C3)) & (C3[np.minimum(pos, len(C3) - 1)] == nk)
        np.maximum.at(w, pos[ok], nw[ok])                     # NER pairs carry their stored 1/df
        q = np.maximum(np.rint(w / max(float(w.max()), 1e-12) * 1000.0), 1).astype(np.int32)
        log(f"   {ds} topology C with stored NER weights: {int(ok.sum()):,} of {len(C3):,} pairs "
            f"weighted, w in [{float(w.min()):.4f}, {float(w.max()):.4f}]")
        xadj, adj, ww = _csr_weighted(C3, q, N)
        mem = metis(N, xadj, adj, k, eweights=ww)
    elif which.startswith("P4_"):
        FAM = {"P4_HYPERGRAPH_CE": (("H_NER", "H_STRUCT_LOCAL", "H_KNN_LOCAL"), True),
               "P4_CE_UNWEIGHTED": (("H_NER", "H_STRUCT_LOCAL", "H_KNN_LOCAL"), False),
               "P4_CE_NER_ONLY": (("H_NER",), True),
               "P4_CE_LOCAL_ONLY": (("H_STRUCT_LOCAL", "H_KNN_LOCAL"), True)}[which]
        uk, q = hyper_clique(ds, N, S, K, X, fams=FAM[0], weighted=FAM[1])
        xadj, adj, w = _csr_weighted(uk, q, N)
        log(f"   {ds} hypergraph CE: {len(uk):,} weighted edges")
        mem = metis(N, xadj, adj, k, eweights=w)
    else:
        C3 = np.union1d(np.union1d(S, X), K)
        KEY = {"PM0_STRUCT": S, "PM1_STRUCT_NERX": np.union1d(S, X),
               "PM2_STRUCT_KNN": np.union1d(S, K), "PM3_TOPOLOGY_C": C3,
               "P2_METIS_STRONG": C3, "P3_FENNEL": C3}.get(which)
        if KEY is None and which.startswith("PM3_TOPOLOGY_C_seed"):
            KEY = C3
        xadj, adj = _csr_from_keys(KEY, N)
        if which == "P3_FENNEL":
            mem = fennel(N, xadj, adj, k)
        else:
            sd = int(which.split("_seed")[1]) if "_seed" in which else None
            mem = metis(N, xadj, adj, k, strong=(which == "P2_METIS_STRONG"), seed=sd)
    sz = np.bincount(mem, minlength=k)
    log(f"  {ds} {which}: k={k} blocks_used={int((sz>0).sum())} min={sz.min()} max={sz.max()} "
        f"CV={sz.std()/max(sz.mean(),1e-9):.4f} ({time.time()-t:.1f}s)")
    np.save(fp, mem)
    return mem


if __name__ == "__main__":
    ds = sys.argv[1]
    todo = sys.argv[2:] or ALL
    os.makedirs(PDIR, exist_ok=True)
    fp = f"{PDIR}/build_{ds}.json"
    rec = json.load(open(fp)) if os.path.exists(fp) else {}
    rec["_SUBSTITUTIONS"] = {
        "KAHYPAR_AVAILABLE": "NO (no wheel for win32/CPython3.13; source build unavailable)",
        "KAHIP_AVAILABLE": "NO (same)",
        "P2_KAHIP_STRONG": "SUBSTITUTED BY P2_METIS_STRONG (niter=100, ncuts=8)",
        "P4_KAHYPAR_UNIVERSAL": "SUBSTITUTED BY P4_HYPERGRAPH_CE (clique expansion + weighted METIS)",
        "HYPER_CAP": HYPER_CAP, "FENNEL_GAMMA": FENNEL_GAMMA, "TARGET_PER_PARTITION": TARGET}
    for w in todo:
        mem = build(ds, w)
        sz = np.bincount(mem)
        rec[w] = {"k": int(mem.max()) + 1, "blocks_used": int((sz > 0).sum()),
                  "min": int(sz.min()), "max": int(sz.max()),
                  "CV": round(float(sz.std() / max(sz.mean(), 1e-9)), 4),
                  "imbalance": round(float(sz.max() / max(sz.mean(), 1e-9)), 4),
                  "path": f"{CACHE}/{ds}__{w}.npy"}
        json.dump(rec, open(fp, "w"), indent=1)
    print("wrote", fp)
