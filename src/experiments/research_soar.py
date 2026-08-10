"""
SOAR-style overlap vs naive centroid-kNN overlap (research improvement).
=========================================================================
Literature (ScaNN/SOAR, NeurIPS 2023): when spilling a vector into extra
partitions, WHICH partition matters more than HOW MANY -- assign the secondary
partition to cover the RESIDUAL the primary misses (orthogonality-amplified),
not merely the 2nd-nearest centroid. Our current knn-overlap is the naive
"m-nearest" baseline SOAR improves upon.

Compares on the FINAL 100-docs/partition substrate, training-free (raw dense ->
centroid, isolating the assignment policy from the router), at matched m
(== matched pool), via pool-matched FullCov:
  - knn{m}:  doc joins its m nearest centroids (naive multiple assignment)
  - soar{m}: doc joins nearest c1, then greedily the centroids nearest to the
             residual r = x - proj_span(assigned)(x)  (SOAR spilling)
If soar{m} > knn{m} at equal pool, the paper's lesson transfers.
Writes results/research/soar_{dataset}.json.
"""
import os
import json
import logging
import argparse

import numpy as np
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership

log = logging.getLogger("experiments.research_soar")


def _knn_assign(sims, m):
    return np.argsort(-sims, axis=1)[:, :m]


def _soar_assign(nv, Cn, sims, m):
    """c1 = nearest centroid; each further pick = centroid nearest to the residual
    of x after removing the span of already-assigned centroids."""
    n = nv.shape[0]
    out = np.argmax(sims, axis=1).reshape(-1, 1)
    if m == 1:
        return out
    full = np.zeros((n, m), dtype=np.int64)
    full[:, 0] = out[:, 0]
    for i in range(n):
        assigned = [int(full[i, 0])]
        for slot in range(1, m):
            r = nv[i].copy()
            for a in assigned:                       # Gram-Schmidt deflation (small m)
                ca = Cn[a]
                r = r - (r @ ca) * ca
            nrm = np.linalg.norm(r)
            if nrm < 1e-6:
                assigned.append(assigned[-1]); continue
            scores = Cn @ (r / nrm)
            for a in assigned:
                scores[a] = -1e9
            assigned.append(int(np.argmax(scores)))
        full[i] = assigned
    return full


def _eval_mem(mem, ranked_pids, gold_idx, budgets):
    """Pool-matched FullCov: route partitions in ranked order, accumulate overlap pool
    (sum of member counts), record whether all gold docs covered when pool >= budget."""
    from collections import defaultdict
    rev = defaultdict(list)
    for i, ps in enumerate(mem):
        for p in ps:
            rev[p].append(i)
    psize = {p: len(v) for p, v in rev.items()}
    fc = {b: [] for b in budgets}
    for qi, gi in enumerate(gold_idx):
        gm = [mem[g] for g in gi]
        if not gm:
            continue
        topset, cum, done = set(), 0, {}
        for p in ranked_pids[qi]:
            topset.add(p); cum += psize.get(p, 0)
            for b in budgets:
                if b not in done and cum >= b:
                    done[b] = all(ms & topset for ms in gm)
            if len(done) == len(budgets):
                break
        for b in budgets:
            fc[b].append(1.0 if done.get(b, all(ms & topset for ms in gm)) else 0.0)
    return {f"fcov@{int(b)}": round(float(np.mean(v)) * 100, 2) for b, v in fc.items()}


def run(dataset, ms=(1, 2, 3)):
    engine = CoreEngine(source=dataset)
    nv = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(nv)
    npart = max(int(p) for p in engine.partition_map.values()) + 1
    id2idx = engine.node_id_to_idx
    part_of = np.zeros(len(engine.nodes), dtype=np.int64)
    for nid, p in engine.partition_map.items():
        if nid in id2idx:
            part_of[id2idx[nid]] = int(p)
    Cn = _reconstruct(engine.centroid_index).astype("float32"); faiss.normalize_L2(Cn)
    cpids = [int(p) for p in engine.centroid_pids] or list(range(npart))
    row2pid = np.array(cpids)
    sims_doc = nv @ Cn.T

    sp = _splits(engine, _hard_membership(engine)); test = sp["test"]
    q = DenseEncoder().encode([qn.content for qn, _, _ in test]).astype("float32"); faiss.normalize_L2(q)
    gold_idx = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in test]
    ranked_pids = row2pid[np.argsort(-(q @ Cn.T), axis=1)]          # partition ids in routed order
    n = len(engine.nodes); budgets = [round(n * f) for f in (0.05, 0.10, 0.20)]

    def mem_from_assign(assign_rows):
        pids = row2pid[assign_rows]
        return [set([int(part_of[i])]) | set(int(x) for x in pids[i]) for i in range(n)]

    out = {"dataset": dataset, "npart": npart, "n_docs": n, "pool_budgets": budgets, "results": {}}
    out["results"]["hard"] = _eval_mem([{int(part_of[i])} for i in range(n)], ranked_pids, gold_idx, budgets)
    for m in ms:
        rk = _eval_mem(mem_from_assign(_knn_assign(sims_doc, m)), ranked_pids, gold_idx, budgets)
        rs = _eval_mem(mem_from_assign(_soar_assign(nv, Cn, sims_doc, m)), ranked_pids, gold_idx, budgets)
        out["results"][f"knn{m}"] = rk
        out["results"][f"soar{m}"] = rs
        log.info(f"  [{dataset} m={m}] knn {rk}  |  soar {rs}")

    os.makedirs("results/research", exist_ok=True)
    with open(f"results/research/soar_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved results/research/soar_{dataset}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="SOAR-style overlap vs naive kNN overlap.")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "metaqa"])
    p.add_argument("--ms", nargs="+", type=int, default=[1, 2, 3])
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== SOAR vs kNN overlap: {ds.upper()} =====")
        run(ds, ms=tuple(a.ms))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
