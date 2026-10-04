"""BOUNDED_VERTEXCUT_R2 (twelfth ruling, 2026-09-15; pre-registered in PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json before any build):
the bounded-overlap static substrate -- every node in at most R = 2 blocks, block cap = the lane's existing C = round(N / k_f) = 100,
the same construction on every graph, no density switch, no query information.

Construction (universal; the constants are R = 2 and the frozen C / k_f of the lane):
    k        = R * k_f                      blocks sized for the replication budget: k blocks of cap C hold R * N memberships
    owners   = a hard partition of the STRUCT nodes (degree >= 1) into k blocks by the validated PHG driver on the primal STRUCT graph
               (objects = nodes, nets = STRUCT edges of unit weight, CONNECTIVITY objective = the edge cut; IMBALANCE 1.03; non-empty repair)
    degree-0 = the frozen universal degree-0 rule of section 20 (_l1c_vcut_transfer.place_degree0, imported unchanged): one home =
               H(nearest anchored frozen-KNN neighbour) else the currently smallest block in canonical node order
    alternate= at most ONE extra block per node, chosen by a single deterministic greedy pass: for node v, the foreign blocks are ranked
               by the number of v's STRUCT neighbours whose HOME is that block (ties -> smaller block id); nodes are processed in
               descending best-gain order (ties -> node id); v takes the first block of its ranked list whose current size < C (live
               sizes, homes + alternates); a node with no foreign neighbour keeps lambda = 1; a node whose every candidate block is full
               keeps lambda = 1 (counted as refused).  Replication is therefore bounded by construction (lambda <= 2) and by the cap.
Objective realised: maximise the STRUCT edges contained by a shared block under lambda <= 2 and |B_p| <= C, spending replication only
where it contains >= 1 edge (the feasible counterpart of "min sum(lambda - 1) s.t. lambda <= 2 + capacity" -- every-edge-covered is
infeasible under a node cap once a node has more than 2 (C - 1) neighbours, so the cut edges are the residual, not a violation).
Representation for routing / serving = section 19 / 20 unchanged: M(v) = {home} or {home, alternate}; H(v) = home; hubs (deg + 1 > C)
vote through H(v) only; serving = M(g) meets the selection; unique exposure = |union B_p|.
"""
import hashlib
import json
import os
import time

import numpy as np
import scipy.sparse as sp

import _l1s_core as S
import _l1c_vcut_lib as V

X = S.X
OUT = V.OUT
PDIR = V.PDIR
R_MAX = 2
TAG = "BR2"
log = S.log


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pin(p):
    return {"path": os.path.relpath(p, X.REPO).replace("\\", "/"), "sha256": sha_file(p), "bytes": os.path.getsize(p)}


def k_of(Gr):
    return R_MAX * int(Gr.k_f)


def owners_tag(Gr):
    return "%s_OWNERS_k%d" % (TAG, k_of(Gr))


def r2_path(ds, Gr):
    return os.path.join(PDIR, "%s__%s_k%d__R2.npz" % (ds, TAG, k_of(Gr)))


# ------------------------------------------------------------------------------------------------ step 1: the owner hypergraph (primal STRUCT graph)
def build_owner_hypergraph(ds, Gr):
    """objects = STRUCT nodes of degree >= 1 (re-indexed in ascending node id), nets = STRUCT edges (2 pins, unit weight), k = R * k_f."""
    t = time.time()
    k = k_of(Gr)
    nodes = np.nonzero(Gr.deg >= 1)[0].astype(np.int64)                 # node_of_obj
    obj = np.full(Gr.N, -1, np.int64)
    obj[nodes] = np.arange(len(nodes), dtype=np.int64)
    eu, ev = obj[Gr.eu], obj[Gr.ev]
    assert (eu >= 0).all() and (ev >= 0).all()
    E = Gr.E
    eptr = (2 * np.arange(E + 1, dtype=np.int64))
    eidx = np.empty(2 * E, np.int32)
    eidx[0::2] = eu
    eidx[1::2] = ev
    ew = np.ones(E, np.int32)
    arrays = dict(eptr=eptr, eidx=eidx, N=np.array([len(nodes)]), k=np.array([int(k)]), ew=ew)
    fp = os.path.join(PDIR, "%s__%s.npz" % (ds, owners_tag(Gr)))
    assert not os.path.exists(fp), "owner hypergraph already built: %s" % fp
    np.savez_compressed(fp, node_of_obj=nodes, n_nodes=np.array([Gr.N]), **arrays)
    from src.l1_canonical import hypergraph as HG  # noqa: E402  (imported, never edited)
    meta = {"tag": owners_tag(Gr), "family": TAG, "dataset": ds, "N": int(len(nodes)), "k": int(k), "n_nodes": int(Gr.N), "struct_edges": int(E), "k_frozen": int(Gr.k_f),
            "R": R_MAX, "capacity_C": int(Gr.C), "objects": "STRUCT nodes of degree >= 1 (node_of_obj in the npz)", "nets": "STRUCT edges, 2 pins, unit weight",
            "objective": "CONNECTIVITY = number of cut STRUCT edges (each net has 2 pins)", "objects_excluded_degree_0": int((Gr.deg == 0).sum()),
            "objects_per_block_mean": round(len(nodes) / float(k), 2), "validity_bound_objects_per_block": int(np.ceil(1.03 * len(nodes) / float(k))),
            "hypergraph": {"hyperedges": int(E), "pins": int(2 * E), "size_min": 2, "size_max": 2, "size_mean": 2.0, "weight_sum": int(E)},
            "file": os.path.relpath(fp, X.REPO).replace("\\", "/"), "bytes": os.path.getsize(fp), "file_sha256": sha_file(fp),
            "content_digest": HG.arrays_digest(arrays), "struct_keys_sha256": Gr.keys_sha,
            "prereg": "PREREGISTRATION_BOUNDED_VERTEXCUT_R2.json", "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t, 1)}
    S.wj(fp[:-4] + ".json", meta)
    log("%s owners k=%d: %d objects (%d degree-0 nodes excluded), %d nets (STRUCT edges), %.1f objects/block (bound %d), %.1f MB (%.0fs)" % (
        ds, k, len(nodes), meta["objects_excluded_degree_0"], E, meta["objects_per_block_mean"], meta["validity_bound_objects_per_block"], meta["bytes"] / 1e6, time.time() - t))
    return fp, meta


# ------------------------------------------------------------------------------------------------ step 3: the alternate assignment
def alternates(Gr, home, C, k):
    """one deterministic static greedy pass (see the module docstring).  home: every node's home (>= 0).  Returns alt (-1 = none), sizes, stats."""
    t = time.time()
    N = Gr.N
    A = sp.csr_matrix((np.ones(len(Gr.au), np.int32), Gr.au, Gr.xu), shape=(N, N))            # undirected STRUCT adjacency (deduplicated)
    Hm = sp.csr_matrix((np.ones(N, np.int32), (np.arange(N), home)), shape=(N, k))
    AH = (A @ Hm).tocsr()                                                                      # AH[v, p] = neighbours of v with home p
    Cnt = AH.tocoo()
    keep = Cnt.col != home[Cnt.row]
    rows, cols, data = Cnt.row[keep].astype(np.int64), Cnt.col[keep].astype(np.int64), Cnt.data[keep].astype(np.int64)
    o = np.lexsort((cols, -data, rows))                                                       # by node, gain descending, block id ascending
    rows, cols, data = rows[o], cols[o], data[o]
    ptr = np.searchsorted(rows, np.arange(N + 1))
    has = ptr[1:] > ptr[:-1]
    gain_max = np.zeros(N, np.int64)
    gain_max[has] = data[ptr[:-1][has]]
    home_cnt = np.asarray(AH[np.arange(N), home]).ravel().astype(np.int64)                     # neighbours sharing the home block
    order = np.lexsort((np.arange(N), -gain_max))
    order = order[gain_max[order] >= 1]
    sizes = np.bincount(home, minlength=k).astype(np.int64)
    sizes_before = sizes.copy()
    alt = np.full(N, -1, np.int64)
    chosen_gain = np.zeros(N, np.int64)
    choice_rank = np.full(N, -1, np.int64)
    refused = 0
    for v in order:
        a, b = ptr[v], ptr[v + 1]
        for j in range(a, b):
            p = cols[j]
            if sizes[p] < C:
                alt[v] = p
                sizes[p] += 1
                chosen_gain[v] = data[j]
                choice_rank[v] = j - a
                break
        else:
            refused += 1
    got = alt >= 0
    hub = Gr.hub
    st = {"nodes": int(N), "nodes_with_a_foreign_neighbour (gain >= 1)": int((gain_max >= 1).sum()), "alternates_assigned": int(got.sum()),
          "refused_for_capacity": int(refused), "RF": round(1.0 + float(got.sum()) / N, 4),
          "alternates_hubs": int((got & hub).sum()), "hubs": int(hub.sum()), "alternates_nonhub": int((got & ~hub).sum()),
          "alternates_first_choice": int((choice_rank == 0).sum()), "alternates_later_choice": int((choice_rank > 0).sum()),
          "chosen_gain": {"mean": round(float(chosen_gain[got].mean()), 2), "p50": float(np.percentile(chosen_gain[got], 50)), "min": int(chosen_gain[got].min()), "max": int(chosen_gain[got].max())} if got.any() else None,
          "alternates_with_gain_ge_home_count (boundary nodes)": int((got & (chosen_gain >= home_cnt)).sum()),
          "alternates_with_gain_1": int((got & (chosen_gain == 1)).sum()),
          "gain_max_distribution": {str(q): int(np.percentile(gain_max[gain_max >= 1], q)) for q in (50, 90, 99, 100)} if (gain_max >= 1).any() else None,
          "block_size_before_alternates": {"mean": round(float(sizes_before.mean()), 2), "min": int(sizes_before.min()), "max": int(sizes_before.max()), "gt_C": int((sizes_before > C).sum())},
          "block_size_after": {"mean": round(float(sizes.mean()), 2), "min": int(sizes.min()), "max": int(sizes.max()), "at_C": int((sizes == C).sum()), "gt_C": int((sizes > C).sum())},
          "seconds": round(time.time() - t, 1)}
    return alt, sizes, st, gain_max, home_cnt


def membership(home, alt, N, k):
    """(ptr, flat) M(v) sorted block ids; Y (N x k) csr; blocks_csc; sizes."""
    lists = [np.array(sorted({int(home[v]), int(alt[v])}), np.int64) if alt[v] >= 0 else np.array([int(home[v])], np.int64) for v in range(N)]
    cnt = np.array([len(x) for x in lists], np.int64)
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(cnt)
    flat = np.concatenate(lists).astype(np.int64)
    Y = sp.csr_matrix((np.ones(len(flat), np.int8), flat, ptr), shape=(N, k))
    Yc = Y.tocsc()
    blocks_csc = (np.asarray(Yc.indptr, np.int64), np.asarray(Yc.indices, np.int64))
    sizes = np.diff(blocks_csc[0]).astype(np.int64)
    return (ptr, flat), Y, blocks_csc, sizes


def containment(Gr, home, alt, Y):
    """STRUCT edges contained by a shared block: overall, by case (home-home / via an alternate), non-hub endpoints only."""
    e_con = (Y[Gr.eu].multiply(Y[Gr.ev]).getnnz(axis=1) > 0)
    hh = home[Gr.eu] == home[Gr.ev]
    nh = (~Gr.hub[Gr.eu]) & (~Gr.hub[Gr.ev])
    return {"edges": int(Gr.E), "contained": int(e_con.sum()), "containment_1hop": round(float(e_con.mean()), 4),
            "contained_home_home": int(hh.sum()), "contained_via_alternate": int((e_con & ~hh).sum()), "cut": int((~e_con).sum()),
            "edges_nonhub_both": int(nh.sum()), "containment_1hop_nonhub_both": round(float(e_con[nh].mean()), 4) if nh.any() else None,
            "edges_touching_a_hub": int((~nh).sum()), "containment_1hop_touching_a_hub": round(float(e_con[~nh].mean()), 4) if (~nh).any() else None}


def load_r2(ds, Gr):
    fp = r2_path(ds, Gr)
    z = np.load(fp)
    return fp, z["home"].astype(np.int64), z["alt"].astype(np.int64), int(z["k"][0])
