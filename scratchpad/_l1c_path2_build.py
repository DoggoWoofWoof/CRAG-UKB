"""L1_COVPART step 1: the ONE higher-order static construction -- H4_SK_PATH2.

    H4_SK_PATH2  =  the frozen H4_SK hypergraph (STRUCT + KNN stars, bit-identical arrays)
                 +  the same H4_SPLIT_PRESERVE star family (frozen builder, frozen cap = block size, frozen retention
                    order, frozen split rule, frozen weights 1000/(|e|-1)) applied to the PATH2 adjacency

    PATH2 adjacency (query-independent, label-free, no parameters beyond the frozen cap):
        hub(v)      := closed STRUCT star of v larger than the block capacity (deg(v) + 1 > cap); a hub cannot be kept
                       with its neighbourhood by any balanced partition, so it is neither an anchor, a middle nor a pin here
        (u, w) in A2 iff u != w, both non-hub, and u-w is a STRUCT edge or u-v-w is a STRUCT path with v non-hub
    i.e. the family's star of u is u's closed 2-hop neighbourhood on the non-hub STRUCT subgraph, exactly what the reach
    diagnostic says the served hits sit next to (results/L1_COVPART/reach_diag_A_metaqa.json: 100 % of unreached gold
    nodes within non-hub distance 3 of a served hit, 59 % within 2; 0 % hubs).

Same k (N // 100), same partitioner contract downstream.  Output: results/L1_COVPART/parts/<ds>__H4_SK_PATH2.npz (+ .json).
Nothing under data/ is written; the frozen H4_SK arrays are read and concatenated unchanged.
"""
import io
import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

import _l1s_core as S
from src.l1_canonical import hypergraph as HG  # noqa: E402  (imported, never edited)
from src.l1_canonical.adapter import CanonicalDataset, sha_file  # noqa: E402

OUT = os.path.join(S.X.REPO, "results", "L1_COVPART")
PDIR = os.path.join(OUT, "parts")
os.makedirs(PDIR, exist_ok=True)
TAG = "H4_SK_PATH2"


def path2_keys(N, ST, cap):
    """undirected keys u*N+v (u<v) of the PATH2 adjacency from the STRUCT keys."""
    u = (ST // N).astype(np.int64)
    v = (ST % N).astype(np.int64)
    A = sp.coo_matrix((np.ones(2 * len(u), np.int8), (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(N, N)).tocsr()
    A.data[:] = 1
    deg = np.diff(A.indptr)
    hub = (deg + 1) > cap
    keep = sp.diags((~hub).astype(np.int8))
    An = keep @ A @ keep                        # STRUCT restricted to non-hub nodes
    An.eliminate_zeros()
    A2 = (An + An @ An).tocsr()
    A2.data[:] = 1
    A2.setdiag(0)
    A2.eliminate_zeros()
    A2 = sp.triu(A2, k=1).tocoo()
    keys = np.unique(A2.row.astype(np.int64) * N + A2.col.astype(np.int64))
    d2 = np.diff(A2.tocsr().indptr) + np.diff(A2.tocsc().indptr)      # 2-hop degree per node (symmetric count)
    return keys, hub, deg, d2


def build(name):
    t = time.time()
    cd = CanonicalDataset(name)
    N, ST, KN, NX = cd.keysets()
    k = HG.frozen_k(N)
    tbs = int(round(N / k))
    frozen_npz = os.path.join(cd.derived_dir, "hypergraph", "H4_SK.npz")
    frozen_meta = json.load(io.open(frozen_npz[:-4] + ".json", encoding="utf-8"))
    z = np.load(frozen_npz)
    assert int(z["N"][0]) == N and int(z["k"][0]) == k
    cap = frozen_meta["cap"]
    assert cap == max(2, int(round(1.0 * tbs))), (cap, tbs)
    keys2, hub, deg, d2 = path2_keys(N, ST, cap)
    S.log("%s: N=%d k=%d cap=%d STRUCT keys %d -> PATH2 keys %d; hubs %d; 2-hop degree quantiles %s" % (
        name, N, k, cap, len(ST), len(keys2), int(hub.sum()), {q: int(np.percentile(d2[~hub], q)) for q in (50, 75, 90, 99)}))
    # the frozen star family over the PATH2 adjacency: build_hypergraph with famset "S" reads keys["STRUCT"] only
    arr2, meta2 = HG.build_hypergraph(N, {"STRUCT": keys2, "KNN": KN}, k, famset="S", tag=name + " PATH2", log=S.log)
    assert meta2["cap"] == cap
    e0, i0, w0 = z["eptr"].astype(np.int64), z["eidx"].astype(np.int32), z["ew"].astype(np.int32)
    e2, i2, w2 = arr2["eptr"].astype(np.int64), arr2["eidx"].astype(np.int32), arr2["ew"].astype(np.int32)
    eptr = np.concatenate([e0, e2[1:] + e0[-1]])
    eidx = np.concatenate([i0, i2])
    ew = np.concatenate([w0, w2])
    arrays = dict(eptr=eptr, eidx=eidx, N=np.array([N]), k=np.array([k]), ew=ew)
    fp = os.path.join(PDIR, "%s__%s.npz" % (name, TAG))
    np.savez_compressed(fp, **arrays)
    sz = np.diff(eptr)
    meta = {"tag": TAG, "dataset": name, "N": N, "k": k, "cap": cap, "hubs": int(hub.sum()),
            "definition": "H4_SK (frozen arrays, unchanged) + H4_SPLIT_PRESERVE star family over the PATH2 adjacency (non-hub closed 2-hop STRUCT neighbourhoods)",
            "path2_keys": int(len(keys2)), "struct_keys": int(len(ST)), "knn_keys": int(len(KN)),
            "twohop_degree_quantiles_nonhub": {str(q): int(np.percentile(d2[~hub], q)) for q in (10, 25, 50, 75, 90, 95, 99)},
            "frozen_H4_SK": {"file": frozen_meta["file"], "sha256": frozen_meta["file_sha256"], "content_digest": frozen_meta["content_digest"],
                             "hyperedges": int(len(e0) - 1), "pins": int(len(i0))},
            "path2_family": {kk: meta2[kk] for kk in ("hyperedges", "pins", "pins_precap_total", "anchor_duplication_pins", "pin_retention_total", "size_min", "size_max", "size_mean", "weight_min", "weight_max", "weight_sum")},
            "hypergraph": {"hyperedges": int(len(sz)), "pins": int(len(eidx)), "size_max": int(sz.max()), "size_mean": round(float(sz.mean()), 2),
                           "weight_sum_frozen": int(w0.sum()), "weight_sum_path2": int(w2.sum())},
            "file": os.path.relpath(fp, S.X.REPO).replace("\\", "/"), "bytes": os.path.getsize(fp), "file_sha256": sha_file(fp),
            "content_digest": HG.arrays_digest(arrays), "frozen_rule_sha256": sha_file(os.path.join(S.X.REPO, "scratchpad", "_l1hu_build.py")),
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t, 1)}
    S.wj(fp[:-4] + ".json", meta)
    S.log("%s: %s -> %d hyperedges, %d pins (frozen %d + PATH2 %d), size max %d, weight sums %d + %d, %.1f MB (%.0fs)" % (
        name, TAG, len(sz), len(eidx), len(i0), len(i2), sz.max(), w0.sum(), w2.sum(), meta["bytes"] / 1e6, time.time() - t))
    return meta


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["metaqa"]):
        build(n)
