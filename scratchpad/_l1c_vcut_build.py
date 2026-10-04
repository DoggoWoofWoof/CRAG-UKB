"""STRUCT_VERTEXCUT_V1 step 1 (ninth ruling, 2026-09-15; pre-registered in PREREGISTRATION_STRUCT_VERTEXCUT_V1.json): the dual
incidence hypergraph of the undirected STRUCT graph.

    hypergraph vertex  x_e   = one per undirected, de-duplicated STRUCT edge (ascending key order; unit weight)
    hyperedge (net)    h_v   = {x_e : e incident to v} for every node v of STRUCT degree >= 2 (unit weight; degree-0/1 nodes form no net --
                               a 0- or 1-pin net can never be cut; the net -> node map is stored)

Partitioning the x_e's with the CONNECTIVITY objective minimises sum_v (lambda(h_v) - 1) = the node replication of the endpoint sets
B_p = {v : some edge of v is in part p}.  No KNN, no NERX, no relation labels, no direction, no queries, no gold, no training.
    python -u _l1c_vcut_build.py <ds> <k>   -> results/L1_COVPART/parts/<ds>__STRUCT_VCUT_V1_k<k>.{npz,json}
The npz carries the hypergraph in the lane's scratch format (eptr, eidx, ew, N = |E| objects, k) plus edge_u, edge_v, net_node, n_nodes.
Nothing under data/ is written; the STRUCT keys are read through the frozen adapter.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

import _l1s_core as S
from src.l1_canonical import hypergraph as HG  # noqa: E402  (imported, never edited)
from src.l1_canonical.adapter import CanonicalDataset, sha_file  # noqa: E402

OUT = os.path.join(S.X.REPO, "results", "L1_COVPART")
PDIR = os.path.join(OUT, "parts")
os.makedirs(PDIR, exist_ok=True)
TAG = "STRUCT_VCUT_V1"


def build(name, k):
    t = time.time()
    cd = CanonicalDataset(name)
    N, ST, KN, NX = cd.keysets()
    ST = np.asarray(ST, np.int64)
    assert np.all(np.diff(ST) > 0), "STRUCT keys must be sorted and unique"
    k_f = HG.frozen_k(N)
    C = int(round(N / k_f))
    E = int(len(ST))
    eu = (ST // N).astype(np.int64)
    ev = (ST % N).astype(np.int64)
    assert np.all(eu < ev)
    deg = np.bincount(np.concatenate([eu, ev]), minlength=N).astype(np.int64)
    net_node = np.nonzero(deg >= 2)[0].astype(np.int64)                   # net id -> node id
    M = int(len(net_node))
    node_net = np.full(N, -1, np.int64)
    node_net[net_node] = np.arange(M, dtype=np.int64)
    # pins: for every edge e and each endpoint with a net, pin (net, e); nets sorted by net id, pins within a net by edge id
    ends = np.concatenate([eu, ev])
    eid = np.concatenate([np.arange(E, dtype=np.int64), np.arange(E, dtype=np.int64)])
    keep = node_net[ends] >= 0
    nets = node_net[ends[keep]]
    pins = eid[keep]
    order = np.lexsort((pins, nets))
    nets, pins = nets[order], pins[order]
    cnt = np.bincount(nets, minlength=M).astype(np.int64)
    assert np.all(cnt == deg[net_node])
    eptr = np.zeros(M + 1, np.int64)
    eptr[1:] = np.cumsum(cnt)
    eidx = pins.astype(np.int32)
    ew = np.ones(M, np.int32)
    arrays = dict(eptr=eptr, eidx=eidx, N=np.array([E]), k=np.array([int(k)]), ew=ew)
    fp = os.path.join(PDIR, "%s__%s_k%d.npz" % (name, TAG, k))
    np.savez_compressed(fp, edge_u=eu, edge_v=ev, net_node=net_node, n_nodes=np.array([N]), **arrays)
    hub = (deg + 1) > C
    meta = {"tag": "%s_k%d" % (TAG, k), "family": TAG, "dataset": name, "N": E, "k": int(k), "n_nodes": N, "struct_edges": E, "k_frozen": k_f,
            "capacity_C": C, "definition": "dual incidence hypergraph of the undirected STRUCT graph: objects = STRUCT edges (unit weight), nets = nodes of "
                                            "degree >= 2 (unit weight) with their incident edges as pins; CONNECTIVITY objective = node replication of the endpoint sets",
            "nets": M, "pins": int(len(eidx)), "pins_dropped_degree1_nodes": int((deg == 1).sum()), "nodes_degree0": int((deg == 0).sum()),
            "nodes_degree1": int((deg == 1).sum()), "nodes_degree_ge2": M, "hubs": int(hub.sum()), "edges_touching_a_hub": int((hub[eu] | hub[ev]).sum()),
            "degree_quantiles": {str(q): int(np.percentile(deg, q)) for q in (10, 25, 50, 75, 90, 95, 99, 100)},
            "edges_per_part_mean": round(E / float(k), 2), "validity_bound_edges_per_part": int(np.ceil(1.03 * E / float(k))),
            "hypergraph": {"hyperedges": M, "pins": int(len(eidx)), "size_min": int(cnt.min()), "size_max": int(cnt.max()), "size_mean": round(float(cnt.mean()), 3), "weight_sum": M},
            "file": os.path.relpath(fp, S.X.REPO).replace("\\", "/"), "bytes": os.path.getsize(fp), "file_sha256": sha_file(fp),
            "content_digest": HG.arrays_digest(arrays), "struct_keys_sha256": hashlib.sha256(np.ascontiguousarray(ST).tobytes()).hexdigest(),
            "prereg": "PREREGISTRATION_STRUCT_VERTEXCUT_V1.json", "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t, 1)}
    S.wj(fp[:-4] + ".json", meta)
    S.log("%s k=%d: N(nodes) %d, |E| %d objects, %d nets (deg >= 2), %d pins (%d degree-1 pins dropped, %d isolated nodes), net size max %d, "
          "%.1f edges/part (bound %d), hubs %d, %.1f MB (%.0fs)" % (name, k, N, E, M, len(eidx), meta["pins_dropped_degree1_nodes"], meta["nodes_degree0"],
                                                                     cnt.max(), E / float(k), meta["validity_bound_edges_per_part"], hub.sum(), meta["bytes"] / 1e6, time.time() - t))
    return meta


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]))
