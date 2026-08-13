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


def _merge_rrf(orders, k0=60):
    """Reciprocal Rank Fusion: score(d) = Σ_signals 1/(k0 + rank). Robust to a weak signal — a lone
    top-ranked non-gold contributes only 1/k0, while a gold ranked well by ≥2 signals accumulates more.
    Unlike min-rank, one bad OOD signal cannot single-handedly displace a strong signal's golds."""
    score = {}
    for order in orders:
        for r, doc in enumerate(order):
            doc = int(doc)
            if doc < 0:
                continue
            score[doc] = score.get(doc, 0.0) + 1.0 / (k0 + r)
    return [d for d, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def _merge_anchored(orders, n_head=5):
    """Dense-anchored fusion: signals[0] (dense) OWNS the top-n_head (never displaced), the rest is RRF over
    all signals. Guarantees best-of ≥ dense at every k ≤ n_head, then lets the learned signals add recall."""
    dense = [int(x) for x in orders[0] if int(x) >= 0]
    head = dense[:n_head]
    seen = set(head)
    tail = [d for d in _merge_rrf(orders) if d not in seen]
    return head + tail


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


def level2_order(d, data_d, per_d, heads, scope_topk, device, adapter=None):
    """L2 signals: dense + rel_hard + mlpT + SPLADE (+ adapted-dense if an adapter is given).
    Returns the list of per-signal orders + golds; run() does the fusion. Dense stays sigs[0] (the anchor)."""
    X_t = per_d["Xt"]; hard = data_d["hard"]; hard_t = torch.tensor(hard, device=device)
    mem_idx = data_d["mem_idx"]; npart = data_d["npart"]; qte, ste, gte = data_d["test"]
    _, I = per_d["faiss"].search(qte, MAXK)
    topP = ([set()] * len(qte)) if not scope_topk else _topP(I, mem_idx, npart, scope_topk)
    with torch.no_grad():
        qt = torch.tensor(qte, device=device); sv = X_t[torch.tensor(ste, device=device)]
        pos = {"dense": qt, "rel_hard": heads["hard"](qt, sv), "mlpT": heads["mix_hard"](qt, sv)}
    orders = {m: _scoped_order(pos[m].cpu(), X_t, hard_t, topP, m == "mlpT", device)[0] for m in pos}
    sigs = [orders["dense"], orders["rel_hard"], orders["mlpT"]]     # dense MUST stay sigs[0] (anchor)
    sp = data_d.get("splade")
    if sp is not None:
        sigs.append(_splade_scoped_order(sp, data_d["test_texts"], hard, topP, dataset=d))
    if adapter is not None:                                          # adapted-dense = orthogonal "help-the-weak" axis
        with torch.no_grad():
            aX = adapter(X_t); aq = adapter(qt)
        sigs.append(_scoped_order(aq.cpu(), aX, hard_t, topP, False, device)[0])
    return sigs, gte                                                 # raw per-signal orders; run() fuses


def _transition(A):
    """Row-normalised adjacency -> transition matrix P (P[i,j] = prob i->j)."""
    A = A.tocsr().astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ A


def _ppr(P, seeds, n, alpha=0.5, iters=8):
    p = np.zeros(n, np.float32)
    for s in seeds:
        if 0 <= s < n:
            p[s] = 1.0 / max(len(seeds), 1)
    r = p.copy()
    for _ in range(iters):
        r = alpha * p + (1 - alpha) * (P.T @ r)
    return r


def _ner_compose(l2_order_qi, P, n, n_seed=10, w=0.2, pool=100):
    """L3 over the struct+weighted-NER graph: PPR mass from L2's top-n_seed seeds, blended (weight w) with
    the L2 rank-score over L2's top-`pool` candidates. Promotes NER-reachable bridge docs (shared-entity
    2nd hops) into the head without the hub blow-ups the old kNN graph caused. Best-of both: L2 head + graph."""
    seeds = [int(x) for x in l2_order_qi[:n_seed] if int(x) >= 0]
    if not seeds:
        return [int(x) for x in l2_order_qi]
    mass = _ppr(P, seeds, n)
    cands = [int(x) for x in l2_order_qi[:pool] if int(x) >= 0]
    m = np.array([mass[c] for c in cands], dtype=np.float32); m = m / (m.max() + 1e-9)
    rs = np.array([1.0 - r / len(cands) for r in range(len(cands))], dtype=np.float32)
    return [cands[j] for j in np.argsort(-(rs + w * m))]


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


def run(datasets=None, head_datasets=None, subdir="gte_qwen", scope_topk=0, n_seed=10, epochs=15,
        device=None, use_adapter=False, adapter_epochs=40):
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
    adapter = None
    if use_adapter:                                                 # universal dense adapter (raises the floor)
        from src.experiments.dense_adapter import train_or_load
        adapter = train_or_load(head_datasets, subdir, adapter_epochs, device)
        gc.collect()
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
        sigs, gte = level2_order(d, data_d, per_d, heads, scope_topk, device, adapter=adapter)
        nq = len(gte)
        perq = lambda fuse: [fuse([s[qi] for s in sigs]) for qi in range(nq)]
        l2_min = perq(_merge_minrank)                               # current system fusion (fragile)
        l2_rrf = perq(_merge_rrf)                                   # robust: RRF
        l2_anc = perq(_merge_anchored)                              # robust + dense-anchored (≥ dense guaranteed)
        eng = CoreEngine(source=d, index_subdir=subdir)
        X = data_d["X"]; n = X.shape[0]; id2idx = eng.node_id_to_idx
        _, A_str, _ = _graph(eng, n, id2idx, sources=("struct",))
        from src.pipeline.ner_edges import build_ner_edges
        A_ner = build_ner_edges(d, [eng.nodes[i].content for i in range(len(eng.nodes))], n)
        P = _transition((A_str + A_ner).tocsr())                    # struct + weighted-NER edges (kNN dropped)
        composed = [_ner_compose(l2_min[qi], P, n, n_seed=n_seed) for qi in range(nq)]   # NER-blend L3 on min-rank base
        zs = " [zero-shot]" if d.split("_")[0] not in {h.split("_")[0] for h in head_datasets} else ""
        out[d] = {"L2_minrank": _recall(l2_min, gte), "L2_rrf": _recall(l2_rrf, gte),
                  "L2_anchored": _recall(l2_anc, gte), "L2_plus_nerL3": _recall(composed, gte),
                  "zero_shot": bool(zs)}
        mn, cp = out[d]["L2_minrank"], out[d]["L2_plus_nerL3"]
        log.info("[e2e/%s]%s R@5  L2=%.1f  ->  L2+NER-L3=%.1f  (Δ %+.1f)",
                 d, zs, mn.get(5, 0), cp.get(5, 0), cp.get(5, 0) - mn.get(5, 0))
        del data_d, per_d, X, A_str, A_ner, P; gc.collect()
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
    p.add_argument("--use-adapter", action="store_true", help="add the universal dense adapter as an L2 signal")
    p.add_argument("--adapter-epochs", type=int, default=40)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, head_datasets=a.head_datasets, subdir=a.subdir, scope_topk=a.scope_topk,
        n_seed=a.n_seed, epochs=a.epochs, use_adapter=a.use_adapter, adapter_epochs=a.adapter_epochs)


if __name__ == "__main__":
    main()
