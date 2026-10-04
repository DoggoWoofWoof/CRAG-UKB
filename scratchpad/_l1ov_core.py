"""UNIVERSAL BALANCED PARTITION + OVERLAP SEARCH -- overlap primitives.

A hard partitioning forces phi(v) = P_i, so a node that participates in several coherent regions
must be assigned to exactly one of them and the rest of that locality is cut.  A balanced core
plus a 1-hop halo lifts that restriction WITHOUT touching the cores:

    ranking unit = balanced CORE          (disjoint, balanced, unchanged)
    fetch unit   = CORE + 1-hop HALO      (halo nodes may belong to many blocks)

    B_j = C_j u H_j       H_j = { v : exists u in C_j, (u,v) in E, phi(v) != j }

Everything here is static: corpus topology only, no query, no gold, no learning.

The central object is the HALO PAIR SET: the set of (block, node) pairs (j, v) with v in H_j,
stored as sorted int64 keys j*N + v.  Two facts make that the right representation:

  * membership is existential over edges, so the FULL_C halo is EXACTLY the union of the
    per-family halo pair sets -- no separate pass over the union graph is needed, and the
    family intersections that Phase 7 asks for are plain set operations on the same keys;
  * both directions the evaluation needs are cheap transposes of it -- block -> halo nodes
    (for exposure) and node -> blocks (for coverage).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1kn_sub as KS

CACHE = "scratchpad/_l1ov"
OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_UNIVERSAL_PARTITION_SEARCH"
DS = KS.DS
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

# Phase 7 canonical halo families.  O0_CORE is the no-halo control and has no pair set.
FAMS = ["O1_STRUCT", "O2_NERX", "O3_KNN", "O4_FULL_C"]
FAMBIT = {"O1_STRUCT": 1, "O2_NERX": 2, "O3_KNN": 4}


def _pairs_for_keys(keys, hard, N):
    """(j, v) halo pairs induced by one undirected key array, as sorted int64 j*N + v.

    An undirected key u*N+v contributes in both directions: v joins block phi(u)'s halo and u
    joins block phi(v)'s halo, in each case only when the two endpoints sit in different blocks.
    """
    u = (keys // np.int64(N)).astype(np.int64)
    v = (keys % np.int64(N)).astype(np.int64)
    hu, hv = hard[u], hard[v]
    cross = hu != hv
    u, v, hu, hv = u[cross], v[cross], hu[cross], hv[cross]
    pk = np.concatenate([hu * np.int64(N) + v, hv * np.int64(N) + u])
    return np.unique(pk)


def halo_pairs(ds, hard, tag, fam, N=None, log=log):
    """sorted (block, node) halo pair keys for one family.  Cached per (ds, core tag, family)."""
    os.makedirs(CACHE, exist_ok=True)
    fp = f"{CACHE}/halo_{ds}__{tag}__{fam}.npy"
    if os.path.exists(fp):
        return np.load(fp, mmap_mode="r")
    n, S, K, X = KS.keysets(ds, lambda *a: None)
    N = N or n
    hard = np.asarray(hard, np.int64)
    if fam == "O4_FULL_C":
        p = np.unique(np.concatenate([halo_pairs(ds, hard, tag, f, N, log)
                                      for f in ("O1_STRUCT", "O2_NERX", "O3_KNN")]))
    else:
        p = _pairs_for_keys({"O1_STRUCT": S, "O2_NERX": X, "O3_KNN": K}[fam], hard, N)
    np.save(fp, p)
    log(f"    {ds} {tag} {fam}: {len(p):,} halo pairs")
    return p


def to_block_csr(pairs, npart, N):
    """block -> sorted halo node ids (pairs are already sorted by j*N+v, so groups are contiguous)."""
    j = (np.asarray(pairs) // np.int64(N)).astype(np.int64)
    v = (np.asarray(pairs) % np.int64(N)).astype(np.int32)
    cnt = np.bincount(j, minlength=npart).astype(np.int64)
    ptr = np.zeros(npart + 1, np.int64)
    ptr[1:] = np.cumsum(cnt)
    return ptr, v


def to_node_csr(pairs, npart, N):
    """node -> sorted block ids that carry it in their halo (the transpose)."""
    p = np.asarray(pairs)
    j = (p // np.int64(N)).astype(np.int32)
    v = (p % np.int64(N)).astype(np.int64)
    o = np.argsort(v, kind="stable")
    v, j = v[o], j[o]
    cnt = np.bincount(v, minlength=N).astype(np.int64)
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(cnt)
    return ptr, j


def _pct(a, qs=(50, 90, 99)):
    a = np.asarray(a)
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def explosion(ds, hard, npart, tag, fams=FAMS, log=log):
    """PHASE 8 -- the structural cost of overlap, reported BEFORE any retrieval evaluation."""
    N = len(hard)
    core_sizes = np.bincount(np.asarray(hard, np.int64), minlength=npart).astype(np.int64)
    out = {"ds": ds, "core_tag": tag, "N": int(N), "npart": int(npart),
           "core_memberships": int(N),
           "O0_CORE": {"TOTAL_MEMBERSHIPS": int(N), "REPLICATION_FACTOR": 1.0,
                       "HALO_CORE_RATIO": 0.0,
                       "NODE_MULTIPLICITY": {"mean": 1.0, "median": 1.0, "p90": 1.0,
                                             "p99": 1.0, "max": 1},
                       "BLOCK_SIZE": {"min": int(core_sizes.min()), "mean": float(core_sizes.mean()),
                                      "median": float(np.median(core_sizes)),
                                      "max": int(core_sizes.max())},
                       "frac_nodes_in_gt": {"1": 0.0, "2": 0.0, "5": 0.0, "10": 0.0}}}
    for fam in fams:
        p = halo_pairs(ds, hard, tag, fam, N, log)
        nh = len(p)
        bptr, _ = to_block_csr(p, npart, N)
        halo_sz = np.diff(bptr)
        blk = core_sizes + halo_sz
        mult = 1 + np.bincount((np.asarray(p) % np.int64(N)).astype(np.int64),
                               minlength=N).astype(np.int64)
        out[fam] = {
            "halo_memberships": int(nh),
            "TOTAL_MEMBERSHIPS": int(N + nh),
            "REPLICATION_FACTOR": round((N + nh) / N, 4),
            "HALO_CORE_RATIO": round(nh / N, 4),
            "NODE_MULTIPLICITY": dict({"mean": round(float(mult.mean()), 4),
                                       "median": float(np.median(mult)),
                                       "max": int(mult.max())}, **_pct(mult, (90, 99))),
            "BLOCK_SIZE": dict({"min": int(blk.min()), "mean": round(float(blk.mean()), 2),
                                "median": float(np.median(blk)), "max": int(blk.max())},
                               **_pct(blk, (90, 99))),
            "HALO_SIZE": dict({"min": int(halo_sz.min()), "mean": round(float(halo_sz.mean()), 2),
                               "median": float(np.median(halo_sz)), "max": int(halo_sz.max())},
                              **_pct(halo_sz, (90, 99))),
            "frac_nodes_in_gt": {str(k): round(float((mult > k).mean()), 4)
                                 for k in (1, 2, 5, 10)},
        }
        del p, mult
    # PHASE 7 -- provenance kept as a bitmask over the three primitive families, not collapsed.
    if all(f in out for f in ("O1_STRUCT", "O2_NERX", "O3_KNN")):
        S = halo_pairs(ds, hard, tag, "O1_STRUCT", N, log)
        X = halo_pairs(ds, hard, tag, "O2_NERX", N, log)
        K = halo_pairs(ds, hard, tag, "O3_KNN", N, log)
        sx = np.intersect1d(S, X, assume_unique=True)
        xk = np.intersect1d(X, K, assume_unique=True)
        sk = np.intersect1d(S, K, assume_unique=True)
        sxk = np.intersect1d(sx, K, assume_unique=True)
        tot = out["O4_FULL_C"]["halo_memberships"]
        multi = len(sx) + len(xk) + len(sk) - 2 * len(sxk)
        out["PROVENANCE"] = {
            "STRUCT_only": int(len(S) - len(sx) - len(sk) + len(sxk)),
            "NERX_only": int(len(X) - len(sx) - len(xk) + len(sxk)),
            "KNN_only": int(len(K) - len(sk) - len(xk) + len(sxk)),
            "STRUCT_and_NERX": int(len(sx)), "NERX_and_KNN": int(len(xk)),
            "STRUCT_and_KNN": int(len(sk)), "all_three": int(len(sxk)),
            "multi_family_pairs": int(multi),
            "frac_multi_family": round(multi / max(tot, 1), 4)}
    return out


# ------------------------------------------------------------------ PHASE 13: bounded halos
def boundary_mass(ds, hard, tag, fam, N, npart, log=log):
    """s(v,P) = sum over u in P adjacent to v of w(u,v)/deg_E(u), on the family's own graph.

    Parameter-free and static: it rewards a node reached by MANY core members and discounts
    edges emitted by hubs.  w is 1 on the deduplicated union graph, exactly as the partitioner
    sees it, so nothing here introduces a tunable.
    """
    fp = f"{CACHE}/bmass_{ds}__{tag}__{fam}.npz"
    if os.path.exists(fp):
        z = np.load(fp)
        return z["pairs"], z["mass"]
    n, S, K, X = KS.keysets(ds, lambda *a: None)
    keys = {"O1_STRUCT": S, "O2_NERX": X, "O3_KNN": K}.get(fam)
    if keys is None:
        keys = np.unique(np.concatenate([S, X, K]))
    u = (keys // np.int64(N)).astype(np.int64)
    v = (keys % np.int64(N)).astype(np.int64)
    deg = np.bincount(np.concatenate([u, v]), minlength=N).astype(np.float64)
    deg[deg == 0] = 1.0
    hu, hv = hard[u], hard[v]
    cross = hu != hv
    u, v, hu, hv = u[cross], v[cross], hu[cross], hv[cross]
    pk = np.concatenate([hu * np.int64(N) + v, hv * np.int64(N) + u])
    w = np.concatenate([1.0 / deg[u], 1.0 / deg[v]])
    o = np.argsort(pk, kind="stable")
    pk, w = pk[o], w[o]
    uniq, start = np.unique(pk, return_index=True)
    mass = np.add.reduceat(w, start)
    np.savez_compressed(fp, pairs=uniq, mass=mass)
    log(f"    {ds} {tag} {fam}: boundary mass for {len(uniq):,} pairs")
    return uniq, mass


def bounded_pairs(pairs, mass, hard, npart, N, beta):
    """|H_j| <= beta * |C_j|, keeping the highest boundary mass.  One beta for every corpus."""
    j = (np.asarray(pairs) // np.int64(N)).astype(np.int64)
    core = np.bincount(np.asarray(hard, np.int64), minlength=npart).astype(np.int64)
    cap = np.floor(beta * core).astype(np.int64)
    # rank within block by descending mass; pairs are sorted by j so groups are contiguous
    order = np.lexsort((-np.asarray(mass), j))
    js = j[order]
    start = np.searchsorted(js, np.arange(npart), side="left")
    rank = np.arange(len(js), dtype=np.int64) - start[js]
    keep = rank < cap[js]
    return np.sort(np.asarray(pairs)[order][keep])
