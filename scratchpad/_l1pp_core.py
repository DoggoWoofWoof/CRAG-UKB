"""L1 SIMPLIFY / PPR PHASE -- shared substrate.

ARCHITECTURAL PRINCIPLE.  L1 answers only "where should we search?".  It stays parameter-free,
partition-level, low-latency, maximally corpus-side precomputable, and emits EXACTLY P=50
canonical partitions.

CHANNEL REDEFINITION (STEP 1).  The previous phase established that `base_rank` and `ret_rrf` are
the SAME evidence at two granularities: `base_rank = rrf_partitions([PR_dense, PR_splade])` and
`ret_rrf = node_rrf(dense200, splade200)`, both from one Dense+SPLADE retrieval.  So the
conceptual channel set is NOT {base_rank, ret_rrf, structure}.  The orthogonal channels are

    DENSE_PARTITION_RANK      from the dense top-K_LOCK nodes
    SPLADE_PARTITION_RANK     from the splade top-K_LOCK nodes
    STRUCTURAL_PARTITION_RANK from the frozen directional expansion / partition graph

and each evidence family appears exactly ONCE in the fusion.

PPR CONTRACT (STEP 10) -- audited, and the audit is a negative finding.  `src/experiments/
l3_methods.py::_ppr` and `l3_solvers.py` implement `p = (1-a)*s + a*(p @ P)` with `iters=20` over
`P = D^-1 A` (symmetrised, row-normalised), uniform seed mass `1/N_seed`.  Those parts are a real
inherited contract.  The damping `a` is NOT: both call sites SELECT it per run from {0.3,0.5,0.7,
0.9} / {0.5,0.7,0.9} by gold recall and store it as `ppr_best_alpha`, and LEVEL3_README marks the
PPR alpha as exploratory.  A gold-selected constant cannot be inherited here.  We therefore fix

    PPR_ALPHA = 0.85   -- the only NON-gold-selected alpha in the codebase, the default of
                          l3_solvers._qppr_ball, and the standard PageRank damping
    PPR_ITERS = 20     -- inherited unchanged

as ONE global contract for every corpus and every variant.  Alpha is never selected on gold; a
small robustness sweep is reported as sensitivity only.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import scipy.sparse as sp
import _ta_prepartition as TA

ROOT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH"
PPD = f"{ROOT}/L1_PPR_SIMPLIFY"
DSETS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"]
K0, K_LOCK, P = TA.K0, TA.K_LOCK, TA.P_MAIN
PPR_ALPHA = 0.85
PPR_ITERS = 20


def load(ds):
    z = np.load(f"{ROOT}/runs/cache_{ds}.npz", allow_pickle=True)
    return z, json.loads(str(z["meta_json"]))


def goldparts(z, meta):
    gp, gptr = z["gold_part"], z["gold_ptr"]
    return [set(int(x) for x in gp[gptr[qi]:gptr[qi + 1]]) for qi in range(meta["n_dev_queries"])]


# ------------------------------------------------------------------ STEP 1: orthogonal channels
def channels(ds, z, meta, topo=None):
    """DENSE / SPLADE partition rankings, reconstructed exactly as _l1ps_cache built them, plus a
    parity gate proving rrf(PR_d, PR_s) reproduces the cached canonical base_rank bit-for-bit.

    Nothing here is new evidence: it only SPLITS the already-frozen canonical ranking back into
    the two modalities it was built from, so each can be given exactly one vote."""
    hard, mem, npart, adj, deg, id2row = topo or TA.load_topology(ds, log=lambda *a: None)
    dK = z["ret_dense"][:, :K_LOCK].astype(np.int64)
    sK = z["ret_splade"][:, :K_LOCK].astype(np.int64)
    PR_d = TA.partition_ranking(list(dK), mem, npart)
    PR_s = TA.partition_ranking(list(sK), mem, npart)
    br = TA.rrf_partitions([PR_d, PR_s], npart)
    cached = z["base_rank"]
    ok = bool(np.array_equal(br[:, :cached.shape[1]], cached))
    return {"PR_d": PR_d, "PR_s": PR_s, "base_rank_replay": br, "npart": npart,
            "hard": hard, "mem": mem, "adj": adj, "deg": deg,
            "BASE_RANK_PARITY": "EXACT" if ok else "MISMATCH"}


def rank_pos(rk, npart):
    """rank matrix (nq, npart) of argsort-order -> position lookup."""
    nq = rk.shape[0]
    pos = np.empty((nq, npart), np.int32)
    pos[np.arange(nq)[:, None], rk] = np.arange(npart)[None, :]
    return pos


def rrf_top(rankings, npart, k=P, tie=None):
    """canonical parameter-free RRF over full partition rankings -> top-k partitions.
    `tie` (default rankings[0]) supplies the deterministic tie-break order, exactly as
    TA.rrf_partitions does, so ties never depend on numpy sort internals."""
    order = TA.rrf_partitions(rankings if tie is None else [tie] + list(rankings), npart)
    return order[:, :k] if k else order


def rrf_from_scores(chan_ranks, npart, tie):
    """RRF where each channel is given as a per-query dict {partition: rank}; a channel with no
    evidence for a partition contributes 0 (canonical per-query masking)."""
    nq = tie.shape[0]
    fv = np.zeros((nq, npart), np.float64)
    for ch in chan_ranks:
        for qi in range(nq):
            for p, r in ch[qi].items():
                fv[qi, p] += 1.0 / (K0 + r)
    fv_in = np.take_along_axis(fv, tie, axis=1)
    idx = np.argsort(-fv_in, axis=1, kind="stable")
    return np.take_along_axis(tie, idx, axis=1).astype(np.int32)


# ------------------------------------------------------------------ STEP 2: partition graph
def partition_graph(ds, topo=None, log=print):
    """Canonical partition transition graph under frozen MASTER_TOPOLOGY=C.

    For every node edge u -> v accumulate mass on part(u) -> part(v).  The node adjacency from
    TA.load_topology is the symmetrised undirected CSR, so both directions are present and the
    accumulated partition graph carries the true edge mass in each direction.  Self-loops
    (part(u) == part(v)) are counted and stored SEPARATELY: they are real internal mass but they
    move no probability between partitions, so the primary transition matrix excludes them and
    the internal mass is kept as a per-partition statistic for the P3 local-PPR variant.

    No repartitioning, no encoder work, cached permanently."""
    fp = f"{PPD}/graph/pgraph_{ds}.npz"
    if os.path.exists(fp):
        z = np.load(fp)
        return {k: z[k] for k in z.files}
    hard, mem, npart, (adj_ptr, adj_idx), deg, id2row = topo or TA.load_topology(ds, log=log)
    n = len(hard)
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(adj_ptr))
    dst = adj_idx.astype(np.int64)
    hp = hard.astype(np.int64)
    pu, pv = hp[src], hp[dst]
    keep = (pu >= 0) & (pv >= 0)
    pu, pv = pu[keep], pv[keep]
    self_mask = pu == pv
    internal = np.bincount(pu[self_mask], minlength=npart).astype(np.int64)
    pu2, pv2 = pu[~self_mask], pv[~self_mask]
    key = pu2 * npart + pv2
    uk, cnt = np.unique(key, return_counts=True)
    eu, ev = (uk // npart).astype(np.int32), (uk % npart).astype(np.int32)
    out = {"npart": np.array(npart), "n_nodes": np.array(n),
           "edge_src": eu, "edge_dst": ev, "edge_w": cnt.astype(np.int64),
           "internal_mass": internal,
           "part_size": np.bincount(hp[hp >= 0], minlength=npart).astype(np.int64)}
    os.makedirs(f"{PPD}/graph", exist_ok=True)
    np.savez_compressed(fp, **out)
    return out


def transition(G, npart=None):
    """row-normalised sparse transition matrix P = D^-1 A over the partition graph."""
    npart = int(npart or G["npart"])
    A = sp.csr_matrix((G["edge_w"].astype(np.float64), (G["edge_src"], G["edge_dst"])),
                      shape=(npart, npart))
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1.0
    return sp.diags(1.0 / d) @ A


def ppr(S, P, alpha=PPR_ALPHA, iters=PPR_ITERS):
    """the inherited iteration, unchanged: p = (1-a) s + a (p P), 20 sweeps.
    S is (nq, npart) row-stochastic personalization; returns (nq, npart) mass."""
    S = np.asarray(S, np.float64)
    p = S.copy()
    for _ in range(iters):
        p = (1.0 - alpha) * S + alpha * (p @ P)
    return p


def seed_personalization(z, hard, npart, nq, k=K_LOCK):
    """UNIVERSAL query seed logic, identical to the frozen router: Dense + SPLADE retrieval fused
    by fixed node RRF, mapped to canonical partitions, mass aggregated per partition.
    No dataset identity, no gold, no fitted weight."""
    S = np.zeros((nq, npart), np.float64)
    rr = z["ret_rrf"]
    for qi in range(nq):
        for r, nd in enumerate(rr[qi][:k]):
            nd = int(nd)
            if nd < 0:
                break
            p = int(hard[nd])
            if p >= 0:
                S[qi, p] += 1.0 / (K0 + r)
    s = S.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return S / s


def mcnemar(cur, base):
    from math import comb
    win = int(((cur == 1) & (base == 0)).sum()); los = int(((cur == 0) & (base == 1)).sum())
    n2 = win + los
    p = 1.0 if n2 == 0 else min(1.0, 2.0 * sum(comb(n2, i) for i in range(min(win, los) + 1))
                                / (2.0 ** n2))
    return {"gained": win, "lost": los, "net": win - los, "mcnemar_p": round(p, 5),
            "sig": bool(p < 0.05)}


def score_top50(top50, goldp, part_sizes=None):
    """ALL / ANY coverage of an exactly-P50 partition selection."""
    nq = len(goldp)
    ind = np.zeros(nq, np.int8); ind_any = np.zeros(nq, np.int8)
    scope = np.zeros(nq, np.int64)
    for qi in range(nq):
        fs = set(int(x) for x in top50[qi])
        assert len(fs) == P, f"query {qi}: {len(fs)} partitions, not {P}"
        ind[qi] = int(goldp[qi] <= fs)
        ind_any[qi] = int(bool(goldp[qi] & fs))
        if part_sizes is not None:
            scope[qi] = int(part_sizes[sorted(fs)].sum())
    return {"ind": ind, "ind_any": ind_any, "scope": scope,
            "ALL": float(ind.mean()), "ANY": float(ind_any.mean()),
            "scope_mean": float(scope.mean()) if part_sizes is not None else None}
