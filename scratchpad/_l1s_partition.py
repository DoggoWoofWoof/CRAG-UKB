"""Partition lever (DEV_A): does a different, still dataset-agnostic partition make the static router cover more?

Variants are single global rule changes applied to the frozen H4_SK hypergraph (Mt-KaHyPar DETERMINISTIC_QUALITY,
KM1, eps 0.03, seed 0, k = N//100 -- the frozen worker, unchanged):
  FROZEN   the served H4_SK partition (must reproduce BASE)
  W1       hyperedge weight = 1 for every hyperedge  (frozen: max(1, rint(1000/(|e|-1))) = equal weight per neighbourhood)
  WSQRT    hyperedge weight = max(1, rint(1000/sqrt(|e|-1)))  (between per-pin and per-neighbourhood)
  S        STRUCT-only hypergraph (no KNN family), frozen weights
Outputs under results/L1_STATIC/parts/ (nothing under data/ is touched).  Evaluation: served dense/SPLADE top-100
-> legacy table (own + directed out-neighbour blocks) -> frozen numerics -> P50 -> ALL over the ADAPTER gold nodes.
"""
import io
import json
import os
import sys
import time

import numpy as np

import _l1s_core as S
from src.l1_canonical import hypergraph as HG  # noqa: E402  (imported, never edited)
from src.l1_canonical import partition as PT  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "metaqa"
variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ["W1", "WSQRT", "S"]
D = S.Data(name, dense_fp32=False)
C = D.C
cd = D.cd
N, nq = D.N, D.nq
PDIR = os.path.join(S.OUT, "parts")
os.makedirs(PDIR, exist_ok=True)
res = {"cache": name, "variants": {}}
frozen_hg = os.path.join(cd.derived_dir, "hypergraph", "H4_SK.npz")
z = np.load(frozen_hg)
eptr, eidx, k = z["eptr"], z["eidx"], int(z["k"][0])
sz = np.diff(eptr).astype(np.int64)
S.log("%s frozen hypergraph: %d hyperedges, %d pins, k=%d, weights [%d, %d]" % (name, len(sz), len(eidx), k, z["ew"].min(), z["ew"].max()))
xo, ao = cd.struct_csr(directed=True)
ao = np.asarray(ao, np.int64)
deg_out = np.diff(xo)
cd._csr.clear()


def legacy_mem(hard):
    r = np.concatenate([np.arange(N, dtype=np.int64), np.repeat(np.arange(N, dtype=np.int64), deg_out)])
    p = np.concatenate([hard, hard[ao]])
    keys = np.unique(r * np.int64(k) + p)
    rr = keys // k
    mlen = np.bincount(rr, minlength=N).astype(np.int64)
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(mlen)
    return (ptr, (keys % k).astype(np.int64))


def evaluate_partition(tag, hard):
    hard = hard.astype(np.int64)
    assert hard.min() >= 0 and hard.max() < k
    sizes = np.bincount(hard, minlength=k)
    mem = legacy_mem(hard)
    gb = [set(int(x) for x in hard[g]) for g in C.gold_nodes]
    ngb = np.array([len(g) for g in gb])
    PR_d = S.canon_channel_rank([D.d_ids[i, :S.K_LOCK] for i in range(nq)], mem, k)
    PR_s = S.canon_channel_rank([D.s_ids[i, :S.K_LOCK] for i in range(nq)], mem, k)
    base = S.rrf_ranks([PR_d, PR_s])

    def cover(rank):
        allv = np.zeros(nq, bool)
        for i in range(nq):
            s = set(rank[i, :S.P_MAIN].tolist())
            allv[i] = all(p in s for p in gb[i])
        return allv

    def summ(allv):
        m = D.A
        out = {"ALL": round(float(allv[m].mean()), 4)}
        hops = C.hops
        for h in sorted(set(int(x) for x in hops[m] if x >= 0)):
            out["hop%d" % h] = round(float(allv[m & (hops == h)].mean()), 4)
        return out

    e = {"blocks": int(k), "size_min": int(sizes.min()), "size_max": int(sizes.max()),
         "gold_blocks_per_query_mean": round(float(ngb.mean()), 3), "feasible_P50": round(float((ngb <= S.P_MAIN).mean()), 4),
         "mean_table_blocks_per_node": round(float(np.diff(mem[0]).mean()), 3)}
    allv = cover(base)
    e["BASE"] = summ(allv)
    e["BASE"]["scope_nodes"] = round(float(np.mean([sizes[base[i, :S.P_MAIN]].sum() for i in range(nq)])), 1)
    g, l, p = S.X.mcnemar(D.base_all[D.A], allv[D.A])
    e["BASE"]["vs_frozen_BASE"] = {"gained": g, "lost": l, "p": p}
    # undirected 1-hop table fused with BASE (the universal candidate), on this partition
    xadj, adj = cd.struct_csr(directed=False)
    adj = np.asarray(adj, np.int64)
    cd._csr.clear()
    src1 = np.repeat(np.arange(N, dtype=np.int64), np.diff(xadj))
    keys = np.unique(np.concatenate([np.arange(N, dtype=np.int64), src1]) * np.int64(k) + np.concatenate([hard, hard[adj]]))
    rr = keys // k
    mlen = np.bincount(rr, minlength=N).astype(np.int64)
    ptr = np.zeros(N + 1, np.int64)
    ptr[1:] = np.cumsum(mlen)
    mem1 = (ptr, (keys % k).astype(np.int64))
    r1d = S.canon_channel_rank([D.d_ids[i, :S.K_LOCK] for i in range(nq)], mem1, k)
    r1s = S.canon_channel_rank([D.s_ids[i, :S.K_LOCK] for i in range(nq)], mem1, k)
    allv1 = cover(S.rrf_ranks([PR_d, PR_s, r1d, r1s]))
    e["mem1_undirected_plus_BASE"] = summ(allv1)
    g, l, p = S.X.mcnemar(D.base_all[D.A], allv1[D.A])
    e["mem1_undirected_plus_BASE"]["vs_frozen_BASE"] = {"gained": g, "lost": l, "p": p}
    S.log("  %-8s blocks %d sizes [%d,%d] gold-blocks/q %.2f feasible %.3f table %.2f | BASE DEV_A %s (vs frozen +%d/-%d p=%.1e) scope %s | mem1+BASE %s" % (
        tag, k, sizes.min(), sizes.max(), ngb.mean(), (ngb <= 50).mean(), np.diff(mem[0]).mean(), e["BASE"], e["BASE"]["vs_frozen_BASE"]["gained"],
        e["BASE"]["vs_frozen_BASE"]["lost"], e["BASE"]["vs_frozen_BASE"]["p"], e["BASE"]["scope_nodes"], e["mem1_undirected_plus_BASE"]))
    return e


res["variants"]["FROZEN"] = evaluate_partition("FROZEN", D.hard)
assert abs(res["variants"]["FROZEN"]["BASE"]["ALL"] - D.r_base["ALL_split"]["A"]["ALL"]) < 1e-9

for var in variants:
    src = os.path.join(PDIR, "%s__%s.npz" % (name, var))
    out_npy = os.path.join(PDIR, "%s__%s.npy" % (name, var))
    stats = os.path.join(PDIR, "%s__%s.stats.json" % (name, var))
    logp = os.path.join(PDIR, "%s__%s.worker.log" % (name, var))
    if not os.path.exists(out_npy):
        if var == "S":
            Nn, ST, KN, NX = cd.keysets()
            arrays, meta = HG.build_hypergraph(N, {"STRUCT": ST, "KNN": KN}, k, famset="S", tag=name)
        else:
            arrays = {key: z[key] for key in z.files}
            if var == "W1":
                arrays["ew"] = np.ones(len(sz), np.int32)
            elif var == "WSQRT":
                arrays["ew"] = np.maximum(1, np.rint(1000.0 / np.sqrt(np.maximum(sz - 1, 1)))).astype(np.int32)
            else:
                raise ValueError(var)
        np.savez_compressed(src, **arrays)
        avail = PT.host_available_gb()
        S.log("  %s: hypergraph %d hyperedges %d pins, weights [%d,%d]; host available %.2f GB -> worker (8 threads, cap 2.5 GB)" % (
            var, len(arrays["eptr"]) - 1, len(arrays["eidx"]), arrays["ew"].min(), arrays["ew"].max(), avail))
        g = PT.run_worker(src, out_npy, stats, 8, 2.5, logp)
        S.log("  %s: worker %s" % (var, g))
        if g.get("killed") or g.get("rc", 1) != 0 or not os.path.exists(out_npy):
            res["variants"][var] = {"status": "WORKER_FAILED", "guard": g}
            continue
        res["variants"][var] = {"guard": g}
    hard_v = np.load(out_npy).astype(np.int64)
    st = json.load(io.open(stats, encoding="utf-8")) if os.path.exists(stats) else {}
    e = evaluate_partition(var, hard_v)
    e["worker_stats"] = {kk: st.get(kk) for kk in ("objective_km1", "objective_cut", "imbalance", "wall_seconds", "peak_rss_mb")}
    res["variants"].setdefault(var, {}).update(e)
    S.wj(os.path.join(S.OUT, "partition_A_%s.json" % name), res)
S.wj(os.path.join(S.OUT, "partition_A_%s.json" % name), res)
