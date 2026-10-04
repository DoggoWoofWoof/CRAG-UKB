"""L3 pool RERANK — close the pool-vs-ranked gap.

The graph traversal reaches 87-100% of golds (union_recall), but the final ranked R@5 stays low (WebQSP
29.7) because the pool is ranked by COSINE (query similarity) and the recovered golds are 2nd-hop — reached by
traversal precisely because they are NOT query-similar. So cosine buries exactly what the graph recovered.

This reranks the SAME traversal pool by a NON-cosine signal: the RELATION-OFFSET head (q + learned relation
direction -> answer region), max-pooled over K directions. If offset-rerank > cosine and approaches the pool
ceiling, the fix for WebQSP (and every dataset) is pool reranking, not more traversal.

Per query, pool = dense-top-B  UNION  2-hop graph neighbourhood of the dense seeds (the graphlift pool).
Scorers ranked over that pool: cosine | offset(maxK) | dense+offset (z-fused). Ceiling = pool recall.
Writes results/L2/l3_rerank_{subdir}.json.
"""
import os
import json
import logging

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _z(x):
    x = np.asarray(x, dtype="float64"); s = x.std()
    return (x - x.mean()) / (s + 1e-8) if s > 0 else x * 0.0


def _run(datasets, subdir="gte_qwen", n_seed=20, budget=100, hops=2, te_cap=2000, K=16):
    from src.experiments.l3_graphlift import _prep, _reach
    from src.experiments.l1l3_recall import _graph
    from src.experiments.relation_route import _learn_offsets
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorers = ["cosine", "offset", "dense+offset"]
    out = {}
    for d in datasets:
        eng, n, id2idx, qn, gold_idx, Xt = _prep(d, subdir, te_cap, device)
        _, A, _ = _graph(eng, n, id2idx, sources=("struct", "syn")); A = A.tocsr()
        R = _learn_offsets(Xt.cpu().numpy(), eng, id2idx, K)                  # (K,dim) relation directions
        Rt = torch.tensor(R, device=device) if len(R) else None
        B = min(budget, n)
        rec = {s: {k: [] for k in (5, 20, 50)} for s in scorers}; ceil = []
        with torch.no_grad():
            for qi in range(len(qn)):
                g = set(gold_idx[qi])
                if not g:
                    continue
                q = torch.tensor(qn[qi], device=device)
                sim = q @ Xt.T
                order = torch.topk(sim, min(max(B, n_seed), n)).indices.cpu().numpy()
                seeds = [int(x) for x in order[:n_seed]]
                pool = list(set(int(x) for x in order[:B]) | _reach(A, seeds, hops))
                Xp = Xt[torch.tensor(pool, device=device)]                    # (|pool|,dim)
                ceil.append(len(g & set(pool)) / len(g))
                cos = (Xp @ q).cpu().numpy()
                if Rt is not None:
                    V = F.normalize(q[None, :] + Rt, dim=1)                    # (K,dim): q shifted by each relation
                    off = (Xp @ V.T).max(1).values.cpu().numpy()              # best-matching relation per pool node
                else:
                    off = cos
                scores = {"cosine": cos, "offset": off, "dense+offset": _z(cos) + _z(off)}
                for s in scorers:
                    ranked = [pool[i] for i in np.argsort(-scores[s])]
                    for k in (5, 20, 50):
                        rec[s][k].append(len(g & set(ranked[:k])) / len(g))
        out[d] = {"corpus_N": int(n), "n_seed": n_seed, "budget": B, "hops": hops, "K": int(len(R)),
                  "pool_ceiling": round(100 * float(np.mean(ceil)), 2),
                  **{s: {f"R@{k}": round(100 * float(np.mean(rec[s][k])), 2) for k in (5, 20, 50)} for s in scorers}}
        r = out[d]
        log.info("[l3-rerank/%s] ceiling=%.1f | R@5 cosine=%.1f offset=%.1f dense+offset=%.1f",
                 d, r["pool_ceiling"], r["cosine"]["R@5"], r["offset"]["R@5"], r["dense+offset"]["R@5"])
        del Xt
        import gc; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    os.makedirs("results/L2", exist_ok=True)
    path = f"results/L2/l3_rerank_{subdir}.json"
    json.dump(out, open(path, "w"), indent=2)
    log.info("-> %s", path)
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   default=["webqsp", "metaqa", "2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean"])
    p.add_argument("--subdir", default="gte_qwen")
    p.add_argument("--n-seed", type=int, default=20)
    p.add_argument("--budget", type=int, default=100)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--te-cap", type=int, default=2000)
    p.add_argument("--K", type=int, default=16)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    _run(a.datasets, a.subdir, a.n_seed, a.budget, a.hops, a.te_cap, a.K)


if __name__ == "__main__":
    main()
