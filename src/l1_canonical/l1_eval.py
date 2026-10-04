"""L1 BASE and L1 SAFE on a canonical replay cache -- the frozen replay of scratchpad/_l1ep_c.replay
/ _l1hu_hard.score, reading data/l1_canonical/<ds>/replay_cache.npz instead of the legacy cache.

  BASE   Dense+SPLADE partition ranking at P=50, no selector          (base_rank[:50])
  SAFE   the frozen B6_S4_F6_Ms64_Mr32 selector, mechanically reaggregated, nothing retuned
  CORR   SAFE - BASE, exact two-sided McNemar on the paired per-query indicator

The canonical cache already carries the H4_SK partition (hard / base_rank / gold_part /
part_sizes were computed under it by replay_cache.py), so no _l1ep_c.rebuild is needed here; the
selector code path is _l1ps_router.build_cache + _l1kb_core.contexts/run_selector unchanged.

    python src/l1_canonical/l1_eval.py <ds> [ds ...]   -> results/L1_CANONICAL/L1_REPLAY_<ds>.json
"""
import io
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for p in (REPO, os.path.join(REPO, "scratchpad")):
    if p not in sys.path:
        sys.path.insert(0, p)
import _l1ps_router as RT  # noqa: E402
import _l1kb_core as KB  # noqa: E402
from src.l1_canonical.adapter import CanonicalDataset, contract_hash, sha_file  # noqa: E402

CFG = dict(KB.BASE_CFG)                    # B=6, M_struct=64, M_ret=32, S4, F6 -- unchanged
MAX_OVER_MEAN = 1.05                       # production balance envelope (_l1hu_hard A7)
OUT = os.path.join(REPO, "results", "L1_CANONICAL")
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def cut_stats(d, hard):
    """fraction of undirected family edges cut by the partition (STRUCT, KNN, NERX; canonical keys)."""
    N, ST, KN, NX = d.keysets()
    out = {}
    for nm, K in (("STRUCT", ST), ("KNN", KN), ("NERX", NX)):
        if K.size == 0:
            out[nm] = None
            continue
        u = (K // np.int64(N)).astype(np.int64)
        v = (K % np.int64(N)).astype(np.int64)
        cut = int((hard[u] != hard[v]).sum())
        out[nm] = {"edges": int(K.size), "cut": cut, "edge_cut_fraction": round(cut / K.size, 4)}
    return out


def evaluate(ds, tag="H4_SK"):
    d = CanonicalDataset(ds)
    cp = d.cache_path()
    man = json.load(io.open(cp[:-4] + ".json", encoding="utf-8"))
    if man["sha256"] != sha_file(cp):
        raise RuntimeError("%s: replay cache digest changed since its manifest" % ds)
    z0 = np.load(cp, allow_pickle=True)
    meta = json.loads(str(z0["meta_json"]))
    z = {k: z0[k] for k in z0.files if k != "meta_json"}
    fa = meta["FROZEN_AGAINST"]
    if fa["dataset_hash"] != d.record_sha:
        raise RuntimeError("%s: cache frozen against another DATASET.json record" % ds)
    if fa["partition_hash"] != sha_file(d.partition_path(tag)):
        raise RuntimeError("%s: cache frozen against another %s partition" % (ds, tag))
    hard = np.asarray(z["hard"], np.int64)
    npart = int(meta["npart"])
    nq = meta["n_dev_queries"]
    t = time.time()
    C = RT.build_cache(ds, z, meta, [CFG["M_struct"]], [CFG["M_ret"]], [CFG["agg"]])
    goldp = [set(int(x) for x in z["gold_part"][z["gold_ptr"][qi]:z["gold_ptr"][qi + 1]]) for qi in range(nq)]
    ind_base = np.array([int(goldp[qi] <= C["base50"][qi]) for qi in range(nq)], np.int8)
    ctxs = KB.contexts(z, meta, C, CFG["B"], CFG)
    r6 = KB.run_selector(ctxs, KB.sel_f6, goldp, ind_base)
    rb = KB.run_selector(ctxs, KB.sel_base, goldp, ind_base)
    sizes = np.asarray(z["part_sizes"])
    scope = np.array([int(sizes[sorted(C["base50"][qi])].sum()) for qi in range(nq)])
    scope6 = np.array([int(sizes[sorted(fs)].sum()) for fs in r6["finals"]])
    mom = round(float(sizes.max() / max(sizes.mean(), 1e-9)), 4)
    R = {"RECORD": "L1_CANONICAL_REPLAY", "dataset": ds, "partition_tag": tag, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "population": {"eval_split": d.eval_split, "sample_rule": meta["sample_rule"], "nq": nq, "row_query_ids_sha256": meta["row_query_ids_sha256"]},
         "npart": npart, "n_docs": int(len(hard)),
         "BASE_ALL_P50": round(float(ind_base.mean()), 4),
         "BASE_ANY_P50": round(float(np.mean([int(bool(goldp[qi] & C["base50"][qi])) for qi in range(nq)])), 4),
         "BASE_SCOPE_NODES": round(float(scope.mean()), 1),
         "SAFE_ALL_P50": round(r6["ALL"], 4), "SAFE_ANY_P50": round(r6["ANY"], 4), "SAFE_SCOPE_NODES": round(float(scope6.mean()), 1),
         "BND_ALL_P50": round(rb["ALL"], 4),
         "CORR": {"dALL": round(r6["ALL"] - float(ind_base.mean()), 4), "gained": r6["gained"], "lost": r6["lost"], "net": r6["net"],
                  "mcnemar_p": r6["mcnemar_p"], "sig": r6["sig"], "gold_admitted": r6["gold_admitted"], "gold_evicted": r6["gold_evicted"],
                  "churn_per_query": round(float(r6["churn"].mean()), 3), "churn_max": int(r6["churn"].max()),
                  "queries_with_zero_churn": int((r6["churn"] == 0).sum())},
         "SELECTOR": {"cfg": {k: v for k, v in CFG.items()}, "code": "_l1ps_router.build_cache + _l1kb_core.contexts/run_selector/sel_f6 (unchanged)"},
         "BALANCE": {"npart": npart, "blocks_used": int((sizes > 0).sum()), "size_min": int(sizes.min()), "size_max": int(sizes.max()),
                     "size_median": float(np.median(sizes[sizes > 0])), "size_mean": round(float(sizes.mean()), 2),
                     "size_p90": float(np.percentile(sizes[sizes > 0], 90)), "size_p99": float(np.percentile(sizes[sizes > 0], 99)),
                     "size_cv": round(float(sizes.std() / max(sizes.mean(), 1e-9)), 4), "max_over_mean": mom,
                     "ELIGIBLE": bool(mom <= MAX_OVER_MEAN and bool((sizes > 0).all()))},
         "CUT": cut_stats(d, hard),
         "FROZEN_AGAINST": fa, "cache": {"file": man["cache"], "sha256": man["sha256"], "bytes": man["bytes"]},
         "L1_CONTRACT_SHA256": contract_hash()["L1_CONTRACT_SHA256"],
         "seconds": round(time.time() - t, 1)}
    hops = np.asarray(z["hops"])
    if (hops >= 0).any():
        R["by_hop"] = {}
        for h in sorted(set(int(x) for x in hops if x >= 0)):
            m = hops == h
            R["by_hop"]["hop%d" % h] = {"n": int(m.sum()), "BASE_ALL": round(float(ind_base[m].mean()), 4),
                                       "SAFE_ALL": round(float(r6["ind"][m].mean()), 4)}
    R["_ind_SAFE"] = r6["ind"].tolist()
    R["_ind_BASE"] = ind_base.tolist()
    os.makedirs(OUT, exist_ok=True)
    fp = os.path.join(OUT, "L1_REPLAY_%s.json" % ds)
    with io.open(fp, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(R, indent=1))
    log("%-9s nq %4d  BASE %.4f  SAFE %.4f  corr %+.4f (p=%s, +%d/-%d)  ANY %.4f/%.4f  scope %.0f  max/mean %.3f  cut STRUCT %.4f KNN %.4f -> %s"
        % (ds, nq, R["BASE_ALL_P50"], R["SAFE_ALL_P50"], R["CORR"]["dALL"], R["CORR"]["mcnemar_p"], R["CORR"]["gained"], R["CORR"]["lost"],
           R["BASE_ANY_P50"], R["SAFE_ANY_P50"], R["BASE_SCOPE_NODES"], mom, R["CUT"]["STRUCT"]["edge_cut_fraction"], R["CUT"]["KNN"]["edge_cut_fraction"],
           os.path.relpath(fp, REPO)))
    return R


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        evaluate(ds)
