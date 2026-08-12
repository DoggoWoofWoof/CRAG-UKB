"""
End-to-end C-RAG pipeline — compose L1 → L2 → L3 and measure final Recall@k.
============================================================================
MODULAR BY DESIGN: each level is a swappable function with one stable contract —
    (query state) -> a per-query ranked list of document indices.
So a level can be re-implemented (a new L2 reranker, a different L3 walk) without touching the
others or the composition; the flow does not break. Levels here:

  L1  candidate generation / scope  (level1_pool)      : query -> working pool (indices)
  L2  best-of fusion rerank         (level2_order)      : (query, pool) -> reranked order + seeds
  L3  traversal from seeds          (level3_order)      : (seeds, graph) -> traversal order

Final order = best-of(L2 order, L3 order) (min-rank union — the same fusion the system uses
everywhere). We report L2-alone vs composed L2+L3 so the traversal's end-to-end lift is explicit.
Writes results/L2/e2e_pipeline_{subdir}.json.
"""
import os
import json
import logging
import argparse

import numpy as np
import torch
import faiss
import scipy.sparse as sp

from src.experiments.l2_seed import (_load, _train_universal, _scoped_order, _splade_scoped_order,
                                      _topP, _recall, MAXK, KS)
from src.experiments.l3_graphlift import _neighbors
from src.experiments.l1l3_recall import _graph

log = logging.getLogger(__name__)


def _merge_minrank(orders):
    """Best-of: rank each doc by its BEST (lowest) position across the given orders (union-preserving)."""
    best = {}
    for order in orders:
        for r, doc in enumerate(order):
            doc = int(doc)
            if doc < 0:
                continue
            if doc not in best or r < best[doc]:
                best[doc] = r
    return [d for d, _ in sorted(best.items(), key=lambda kv: kv[1])]


def _compose(l2_qi, l3_order, n_head):
    """Head-preserving fusion: L2's confident top-n_head is trusted verbatim (so tight-budget recall never
    regresses); traversal competes only in the TAIL, min-rank-fused with L2's own tail. This is where
    graph traversal earns its place — recovering dense-missed golds at generous budget without letting
    unguided graph hubs displace L2's head."""
    head = [int(x) for x in l2_qi[:n_head] if int(x) >= 0]
    seen = set(head)
    l2_tail = [int(x) for x in l2_qi[n_head:] if int(x) >= 0 and int(x) not in seen]
    l3_tail = [int(x) for x in l3_order if int(x) >= 0 and int(x) not in seen]
    return head + _merge_minrank([l2_tail, l3_tail])


# ------- modular level components (stable contract: state in -> ranked doc indices out) -------
def level1_pool(I_qi, mem_idx, npart, scope_topk):
    """L1: routed working pool for a query (empty set == full corpus, no scoping)."""
    return set()  # scope handled by L2's topP; kept as a seam so an L1 candgen can be swapped in


def level2_order(d, data_d, per_d, heads, scope_topk, device):
    """L2: best-of fusion rerank (dense + rel_hard + mlpT + SPLADE). Returns per-query orders + golds."""
    X_t = per_d["Xt"]; hard = data_d["hard"]; hard_t = torch.tensor(hard, device=device)
    mem_idx = data_d["mem_idx"]; npart = data_d["npart"]; qte, ste, gte = data_d["test"]
    _, I = per_d["faiss"].search(qte, MAXK)
    topP = ([set()] * len(qte)) if not scope_topk else _topP(I, mem_idx, npart, scope_topk)
    with torch.no_grad():
        qt = torch.tensor(qte, device=device); sv = X_t[torch.tensor(ste, device=device)]
        pos = {"dense": qt, "rel_hard": heads["hard"](qt, sv), "mlpT": heads["mix_hard"](qt, sv)}
    orders = {m: _scoped_order(pos[m].cpu(), X_t, hard_t, topP, m == "mlpT", device)[0] for m in pos}
    sigs = [orders["dense"], orders["rel_hard"], orders["mlpT"]]
    sp = data_d.get("splade")
    if sp is not None:
        sigs.append(_splade_scoped_order(sp, data_d["test_texts"], hard, topP, dataset=d))
    l2 = [_merge_minrank([o[qi] for o in sigs]) for qi in range(len(qte))]
    return l2, gte


def _transition(A):
    """Row-normalised adjacency -> transition matrix P (P[i,j] = prob i->j)."""
    A = A.tocsr().astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ A


def level3_order(l2_order_qi, P, qvec, Xn, n_seed, alpha=0.5, iters=4):
    """L3: query-GUIDED Personalized PageRank seeded by L2's top-n_seed. PPR mass flows to docs graph-central
    to the seeds; we multiply by query relevance so a doc must be BOTH reachable and on-topic to surface —
    this is what stops unguided graph hubs from displacing L2's golds. Returns the guided traversal order."""
    n = P.shape[0]
    seeds = [int(x) for x in l2_order_qi[:n_seed] if 0 <= int(x) < n]
    if not seeds:
        return [int(x) for x in l2_order_qi]
    p = np.zeros(n, dtype=np.float32); p[seeds] = 1.0 / len(seeds)
    r = p.copy()
    for _ in range(iters):                          # r = α·p + (1-α)·Pᵀr  (mass flows into graph-central docs)
        r = alpha * p + (1 - alpha) * (P.T @ r)
    rel = np.clip(Xn @ qvec, 0, None)               # query relevance gate: reachable AND on-topic
    score = r * rel
    return np.argsort(-score).tolist()


HPR_EVAL = ["musique_hpr_clean", "2wiki_hpr_clean", "hotpot_hpr_clean"]
# training mix for the ONE universal head: the two cache-ready multi-hop corpora (no encoder load).
# hotpot is held OUT entirely -> hotpot_hpr is a fully zero-shot transfer point.
HEAD_MIX = ["musique_clean", "2wiki_clean"]


def run(datasets=None, head_datasets=None, subdir="gte_qwen", scope_topk=0, n_seed=10, epochs=15, device=None):
    import gc
    datasets = datasets or HPR_EVAL
    head_datasets = head_datasets or HEAD_MIX
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- STEP 1: train ONE universal head on the broad mix (reused via head-cache on later runs) ----
    log.info("=== training ONE universal head on %s (applied to %s WITHOUT refitting) ===",
             head_datasets, datasets)
    per_tr = {}
    for d in head_datasets:
        dd = _load(d, subdir, 8000, 3000, 1)                       # te_cap=1: head-training needs no test set
        idx = faiss.IndexFlatIP(dd["X"].shape[1]); idx.add(dd["X"])
        per_tr[d] = {"train": dd["train"], "Xt": torch.tensor(dd["X"], device=device), "index": idx}
        log.info("  [head-train] loaded %s: X%s", d, dd["X"].shape)
    heads = {k: _train_universal(k, per_tr, device, epochs, K=8) for k in ("hard", "mix_hard")}
    del per_tr; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- STEP 2: apply the champion head to each _hpr eval corpus (no refitting), compose L1->L2->L3 ----
    from src.core.engine import CoreEngine
    out = {}
    for d in datasets:
        data_d = _load(d, subdir, 8000, 3000, 2000)
        idx = faiss.IndexFlatIP(data_d["X"].shape[1]); idx.add(data_d["X"])
        per_d = {"train": data_d["train"], "Xt": torch.tensor(data_d["X"], device=device),
                 "index": idx, "faiss": idx}
        l2, gte = level2_order(d, data_d, per_d, heads, scope_topk, device)
        eng = CoreEngine(source=d, index_subdir=subdir)
        X = data_d["X"]; n = X.shape[0]; id2idx = eng.node_id_to_idx
        _, A, _ = _graph(eng, n, id2idx, sources=("struct", "syn"))
        P = _transition(A)                                          # one-time per dataset
        Xn = X.copy(); faiss.normalize_L2(Xn)
        qn = data_d["test"][0].copy(); faiss.normalize_L2(qn)       # normalised queries for the relevance gate
        composed = []
        for qi in range(len(gte)):
            l3 = level3_order(l2[qi], P, qn[qi], Xn, n_seed)
            composed.append(_compose(l2[qi], l3, n_head=5))         # L2 head trusted; L3 augments the tail
        zs = " [zero-shot: domain held out of head training]" if d.split("_")[0] not in \
             {h.split("_")[0] for h in head_datasets} else ""
        out[d] = {"L2_only": _recall(l2, gte), "L2_plus_L3": _recall(composed, gte), "zero_shot": bool(zs)}
        r2, r3 = out[d]["L2_only"], out[d]["L2_plus_L3"]
        log.info("[e2e/%s]%s L2 R@2=%.1f R@5=%.1f  ->  L2+L3 R@2=%.1f R@5=%.1f (Δ@5 %+.1f)",
                 d, zs, r2.get(2, 0), r2.get(5, 0), r3.get(2, 0), r3.get(5, 0), r3.get(5, 0) - r2.get(5, 0))
        del data_d, per_d, X, Xn, A, P; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    out["_meta"] = {"head_trained_on": head_datasets, "evaluated_on": datasets,
                    "note": "ONE universal offset head trained on the broad mix, applied to each eval corpus "
                            "without refitting; hotpot held out of training (zero-shot)."}
    os.makedirs("results/L2", exist_ok=True)
    json.dump(out, open(f"results/L2/e2e_pipeline_{subdir}.json", "w"), indent=2)
    log.info("-> results/L2/e2e_pipeline_%s.json", subdir)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="End-to-end L1->L2->L3 pipeline; final Recall@k (L2 vs L2+L3). "
                                            "ONE universal head trained on --head-datasets, applied to --datasets.")
    p.add_argument("--datasets", nargs="+", default=None, help="eval corpora (default: the 3 _hpr sets)")
    p.add_argument("--head-datasets", nargs="+", default=None,
                   help="corpora the ONE universal head trains on (default: musique/2wiki/squad _clean)")
    p.add_argument("--subdir", default="gte_qwen")
    p.add_argument("--scope-topk", type=int, default=0)
    p.add_argument("--n-seed", type=int, default=10)
    p.add_argument("--epochs", type=int, default=15)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, head_datasets=a.head_datasets, subdir=a.subdir,
        scope_topk=a.scope_topk, n_seed=a.n_seed, epochs=a.epochs)


if __name__ == "__main__":
    main()
