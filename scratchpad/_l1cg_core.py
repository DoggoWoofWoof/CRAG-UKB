"""L1 CANDIDATE GENERATION PHASE -- shared substrate.

ARCHITECTURAL SPLIT (the rule this phase exists to respect).  PROPOSAL/DISCOVERY is separated from
RANKING/SELECTION.  A family here only ever contributes CANDIDATE MEMBERSHIP.  No family may
contribute a score to the frozen F6/R0 selector, and the selector is not modified.

    final = base_rank[:P-B]  u  X,   X subset of (bnd u chal u NEW),  |X| = B = 6

so a proposal family can only ever change WHICH partitions compete for the same 6 slots.

PROPOSAL FAMILIES (all inference-safe, no gold anywhere in generation):
    A DENSE_PARTITION       PR_d  -- dense partition ranking            (cached replay)
    B SPLADE_PARTITION      PR_s  -- splade partition ranking           (cached replay)
    C NODE_DENSE_CONT       hard[ret_dense]  node continuation          (cache)
    D NODE_SPLADE_CONT      hard[ret_splade] node continuation          (cache)
    E S4_STRUCT             frozen S4 aggregation, M extended to 256    (cache)
    F PPR_REACH             partition-PPR mass, PROPOSAL ONLY           (cache)
    G1/G2/G3 GRAPH_NBR_d1/2/3   precomputed partition-graph neighbour tables of the retrieval
                                seed partitions -- pure lookup + merge, 0 online graph edges
    H STRUCT_FRONTIER       G1 u G2 u G3, depth-ordered
    I CANON_CONT            the FUSED canonical ranking beyond the top-50, base_rank[P:] -- the
                            pool omission STEP 4 exposed (77% of MetaQA hop3 missing partitions are
                            inside canonical top-200 yet outside the current pool)

BUDGET SEMANTICS.  A proposal already inside base50 is a no-op, so every family is filtered to
partitions OUTSIDE base50 BEFORE the top-M cut.  M therefore counts real proposals.
"""
import os, sys, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import scipy.sparse as sp
import _ta_prepartition as TA
import _l1pp_core as PP
import _l1kb_core as KB
import _l1cal_core as CC

K0, P = PP.K0, PP.P
ROOT = PP.ROOT
CGD = f"{ROOT}/L1_CANDIDATE_GEN"
DSETS = PP.DSETS
B = 6
MS = [8, 16, 32, 64, 128, 256]
KNBR = 32                      # neighbours kept per partition per depth in the precomputed tables

FAMS = ["A_DENSE_PARTITION", "B_SPLADE_PARTITION", "C_NODE_DENSE_CONT", "D_NODE_SPLADE_CONT",
        "E_S4_STRUCT", "F_PPR_REACH", "G_GRAPH_NBR_RAW", "G_GRAPH_NBR_NORM",
        "H_STRUCT_FRONTIER", "I_CANON_CONT"]


# ------------------------------------------------------------------ static partition graph
def part_graph(ds, log=lambda *a: None):
    """corpus-side ONLY.  Weighted partition adjacency + top-KNBR neighbour tables at depth 1/2/3.

    Built once per corpus and cached.  At query time nothing here is recomputed and no graph edge
    is traversed -- the depth tables are plain arrays indexed by seed partition."""
    os.makedirs(f"{CGD}/pg", exist_ok=True)
    fp = f"{CGD}/pg/pg_{ds}.npz"
    if os.path.exists(fp):
        z = np.load(fp)
        return {k: z[k] for k in z.files}
    hard, mem, npart, adj, deg, _ = TA.load_topology(ds, log=lambda *a: None)
    ap, ai = adj
    n = len(hard)
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(ap))
    A = sp.csr_matrix((np.ones(len(ai), np.float32), (rows, ai.astype(np.int64))), shape=(n, n))
    Mm = sp.csr_matrix((np.ones(n, np.float32), (np.arange(n, dtype=np.int64), hard.astype(np.int64))),
                       shape=(n, npart))
    G = (Mm.T @ A @ Mm).tocsr()
    G.setdiag(0); G.eliminate_zeros()
    vol = np.asarray(G.sum(1)).ravel() + 1e-9
    log(f"[pg] {ds} npart={npart} partition-edges={G.nnz}")

    def topk(Gx):
        """top-KNBR neighbours per partition, by weight desc then id."""
        out = np.full((npart, KNBR), -1, np.int32)
        ind, ptr, dat = Gx.indices, Gx.indptr, Gx.data
        for p in range(npart):
            a, b = ptr[p], ptr[p + 1]
            if b <= a:
                continue
            nb, wt = ind[a:b], dat[a:b]
            o = np.lexsort((nb, -wt))[:KNBR]
            out[p, :len(o)] = nb[o]
        return out

    Gn = G.multiply(1.0 / np.sqrt(vol)[:, None]).multiply(1.0 / np.sqrt(vol)[None, :]).tocsr()
    N1, N1n = topk(G), topk(Gn)
    # depth 2 / 3 by composing the SPARSIFIED table -- itself a corpus-side precomputation
    S = sp.csr_matrix((np.ones(int((N1 >= 0).sum()), np.float32),
                       (np.repeat(np.arange(npart), (N1 >= 0).sum(1)), N1[N1 >= 0])),
                      shape=(npart, npart))
    S2 = (S @ S).tocsr(); S2.setdiag(0); S2.eliminate_zeros()
    S3 = (S2 @ S).tocsr(); S3.setdiag(0); S3.eliminate_zeros()
    N2, N3 = topk(S2), topk(S3)
    o = {"N1": N1, "N1n": N1n, "N2": N2, "N3": N3, "npart": np.int64(npart), "vol": vol}
    np.savez_compressed(fp, **o)
    return o


# ------------------------------------------------------------------ proposal families
def _dedup(seq, drop):
    out, seen = [], set()
    for p in seq:
        p = int(p)
        if p < 0 or p in seen or p in drop:
            continue
        seen.add(p); out.append(p)
    return out


def _merge_tables(tab, seedp, drop):
    """merge precomputed neighbour rows of the seed partitions.  Order = support count desc
    (how many seeds reach it), then the best position in any seed's row, then id.  Pure arithmetic
    over cached rows: ONLINE_GRAPH_EDGES_TOUCHED = 0."""
    sup, bestpos = {}, {}
    for sp_ in seedp:
        r = tab[sp_]
        for j, q in enumerate(r):
            q = int(q)
            if q < 0 or q in drop:
                continue
            sup[q] = sup.get(q, 0) + 1
            if q not in bestpos or j < bestpos[q]:
                bestpos[q] = j
    return [q for q in sorted(sup, key=lambda q: (-sup[q], bestpos[q], q))]


def proposals(ds, S, log=lambda *a: None):
    """ordered proposal list per family per query, already excluding base50."""
    os.makedirs(f"{CGD}/prop", exist_ok=True)
    fp = f"{CGD}/prop/prop_{ds}.pkl"
    if os.path.exists(fp):
        with open(fp, "rb") as fh:
            return _augment(ds, S, pickle.load(fh), fp, log)
    z, meta, nq, ctxs = S["z"], S["meta"], S["nq"], S["ctxs"]
    hard = z["hard"]
    ch = PP.channels(ds, z, meta)
    PG = part_graph(ds, log)
    pl = None
    f = f"{PP.PPD}/ppr/mass_{ds}.npz"
    if os.path.exists(f):
        m = np.load(f)["mass_global"]
        pl = (m, np.argsort(-m, axis=1, kind="stable"))
    seeds = z["seeds"]
    O = {k: [] for k in FAMS}
    for qi in range(nq):
        drop = ctxs[qi]["base50"]
        O["A_DENSE_PARTITION"].append(_dedup(ch["PR_d"][qi], drop))
        O["B_SPLADE_PARTITION"].append(_dedup(ch["PR_s"][qi], drop))
        O["C_NODE_DENSE_CONT"].append(_dedup(hard[z["ret_dense"][qi]], drop))
        O["D_NODE_SPLADE_CONT"].append(_dedup(hard[z["ret_splade"][qi]], drop))
        agg = __import__("_l1ps_router").struct_aggregate_full(z, qi, hard, 256)
        O["E_S4_STRUCT"].append(_dedup(__import__("_l1ps_router").order_struct(agg, "S4"), drop))
        if pl is not None:
            m, order = pl
            O["F_PPR_REACH"].append(_dedup([p for p in order[qi] if m[qi, p] > 0], drop))
        else:
            O["F_PPR_REACH"].append([])
        sp_ = sorted({int(hard[int(v)]) for v in seeds[qi] if v >= 0})
        O["G_GRAPH_NBR_RAW"].append(_merge_tables(PG["N1"], sp_, drop))
        O["G_GRAPH_NBR_NORM"].append(_merge_tables(PG["N1n"], sp_, drop))
        d1 = _merge_tables(PG["N1"], sp_, drop)
        d2 = _merge_tables(PG["N2"], sp_, drop | set(d1))
        d3 = _merge_tables(PG["N3"], sp_, drop | set(d1) | set(d2))
        O["H_STRUCT_FRONTIER"].append(d1 + d2 + d3)
        O.setdefault("_D1", []).append(d1)
        O.setdefault("_D2", []).append(d2)
        O.setdefault("_D3", []).append(d3)
    with open(fp, "wb") as fh:
        pickle.dump(O, fh, protocol=5)
    return _augment(ds, S, O, fp, log)


def _augment(ds, S, O, fp, log=lambda *a: None):
    """families added after the first cache was written, computed from cached artifacts only."""
    if "I_CANON_CONT" in O:
        return O
    ch = PP.channels(ds, S["z"], S["meta"])
    br = ch["base_rank_replay"]
    O["I_CANON_CONT"] = [_dedup(br[qi], S["ctxs"][qi]["base50"]) for qi in range(S["nq"])]
    log(f"[aug] {ds} I_CANON_CONT added")
    with open(fp, "wb") as fh:
        pickle.dump(O, fh, protocol=5)
    return O


# ------------------------------------------------------------------ pool / oracle helpers
def pool_of(c):
    """the CURRENT frozen candidate pool: boundary incumbents plus F6 challengers."""
    return set(c["bnd"]) | set(c["chal"])


def targets(S):
    """per query: the partitions that must be swapped in, and whether the current pool has them.

    miss      = gold outside the protected core   (what any selector must admit)
    outside   = the part of miss the current pool cannot even see
    feasible  = |gold| <= P, i.e. not a P50 capacity failure
    """
    goldp, ctxs = S["goldp"], S["ctxs"]
    out = []
    for qi, c in enumerate(ctxs):
        g = goldp[qi]
        miss = g - c["prot_set"]
        pool = pool_of(c)
        out.append({"miss": miss, "outside": miss - pool, "pool": pool,
                    "feasible": len(g) <= P})
    return out


def pool_oracle(S, T, extra=None, Bv=B):
    """ALL coverage of a PERFECT selector restricted to (current pool u extra), at budget Bv."""
    nq = len(T)
    ind = np.zeros(nq, np.int8)
    for qi in range(nq):
        t = T[qi]
        if not t["feasible"] or len(t["miss"]) > Bv:
            continue
        uni = t["pool"] if extra is None else (t["pool"] | set(extra[qi]))
        ind[qi] = int(t["miss"] <= uni)
    return ind


def run_frozen(S, extra=None, Bv=B):
    """the FROZEN F6/R0 selector over (bnd u chal u extra).  The selector is NOT modified: the
    extra candidates simply enter the same rank arithmetic, scoring 0.0 on any channel that has no
    rank for them -- exactly the frozen absent-channel semantics."""
    ctxs, goldp = S["ctxs"], S["goldp"]
    nq = len(ctxs)
    ind = np.zeros(nq, np.int8); churn = np.zeros(nq, np.int32)
    for qi, c in enumerate(ctxs):
        chal = list(c["chal"]) + (list(extra[qi]) if extra is not None else [])
        X, _ = KB.f6_select(c["bnd"], chal, c["spos"], c["rpos"], c["cpos"], Bv)
        fs = c["prot_set"] | set(X)
        assert len(fs) == P, f"{len(fs)} != {P}"
        ind[qi] = int(goldp[qi] <= fs)
        churn[qi] = len(fs - c["base50"])
    return ind, churn


def substrate(ds):
    S = CC.substrate(ds, B)
    S["T"] = targets(S)
    return S


def mc(a, b):
    return PP.mcnemar(a, b)
