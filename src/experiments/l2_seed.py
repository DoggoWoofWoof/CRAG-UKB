"""
Level 2 — seed-finding WITHIN the routed top-K partitions.
==========================================================
Faithful pipeline: L1 routes to the top-K partitions; L2 sees ONLY the docs paged in from those
partitions (by hard membership) and must find the gold SEED DOCS there. The offset head trains on the
full corpus, but retrieval + recall are scoped to the top-K pool.

L1 finding: the offset is REDUNDANT at partition routing (overlap-voting does the hop). Its home is L2 —
docRecall is the real hole (metaqa dense ~27-41 @20). This eval trains ONE universal head (base / mlpT)
on pooled (query→gold-doc) triples (reusing the L1 trainer) and reports, per dataset, **gold-DOC
recall@{5,20,50} within the top-K pool** for: dense · offset(base) · mlpT · best-of(dense,·), plus the
POOL CEILING (fraction of golds whose partition is in the top-K = max recall L2 can reach).

Encoder gte_qwen (cached queries). scope_topk=50 (L1 FullCov@50≈98). Reuses l1_universal_head.
"""
import os
import json
import pickle
import argparse
import logging

import numpy as np
import torch
import faiss

from src.experiments.l1_universal_head import _load, _train_universal, DATASETS
from src.experiments.l1_rerank100 import _feats, _rr, MAXK

log = logging.getLogger("experiments.l2_seed")
KS = [2, 5, 20, 50]


def _topP(dense_order, mem_idx, npart, scope_topk):
    """Per-query set of top-`scope_topk` partitions from dense + overlap-voting (matches L1)."""
    S, M = _feats(dense_order, mem_idx, npart, topn=200)
    votes = _rr(S) + _rr(M)
    return [set(int(p) for p in np.argsort(-votes[qi])[:scope_topk]) for qi in range(len(votes))]


def _scoped_order(pos, X_t, hard_t, topP, is_mix, device, bs=128, k=MAXK):
    """Retrieve within each query's top-K partition pool: docs whose HARD partition ∈ topP[qi]."""
    n = X_t.shape[0]; kk = min(k, n)
    if is_mix:                                                 # (b,K,n) einsum is the memory hog on big corpora
        Kd = pos.shape[1]                                      # cap peak floats (b*K*n) so full-corpus hotpot (507k) fits 24GB
        bs = max(4, min(bs, 2_500_000_000 // (4 * max(1, Kd * n))))
    orders = np.full((pos.shape[0], kk), -1, dtype=np.int64)
    scores = np.full((pos.shape[0], kk), -1e9, dtype=np.float32)   # top-k sims -> confidence for gating
    with torch.no_grad():
        for s in range(0, pos.shape[0], bs):
            p = pos[s:s + bs].to(device)
            sim = (torch.einsum("bkd,nd->bkn", p, X_t).max(1).values if is_mix else p @ X_t.T)  # (b,n)
            for j in range(p.shape[0]):
                tp = topP[s + j]
                if tp:
                    allow = torch.isin(hard_t, torch.tensor(list(tp), device=device))
                    sim[j] = torch.where(allow, sim[j], torch.full_like(sim[j], -1e9))
            vals, idx = torch.topk(sim, kk, dim=1)
            orders[s:s + bs] = idx.cpu().numpy(); scores[s:s + bs] = vals.cpu().numpy()
    return orders, scores


def _scoped_order_gather(pos, X_cpu, hard, topP, device, k=MAXK):
    """DEPLOYMENT-path scoped scoring: the corpus X stays on CPU/mmap and only each query's routed-pool
    rows X[pool] are fetched to GPU and scored — O(pool) GPU working set instead of _scoped_order's O(N)
    score-all-then-mask. Correctness-equivalent to _scoped_order (verified by l2_mem_bench: topk_mismatch=0),
    it's the implementation that REALIZES the N/P memory saving the partition routing makes possible.
    pos: (nq,d) OffsetHead/dense positions; X_cpu: (n,d) CPU float tensor; hard: (n,) doc->partition."""
    nq = pos.shape[0]; n = X_cpu.shape[0]; kk = min(k, n)
    orders = np.full((nq, kk), -1, dtype=np.int64)
    scores = np.full((nq, kk), -1e9, dtype=np.float32)
    harr = np.asarray(hard)
    with torch.no_grad():
        for qi in range(nq):
            tp = topP[qi]
            rows = np.where(np.isin(harr, np.fromiter(tp, int)))[0] if tp else np.arange(n)
            if rows.size == 0:
                continue
            Xg = X_cpu[torch.from_numpy(rows)].to(device)         # ONLY the pool (P×d) crosses to GPU
            sim = (pos[qi:qi + 1].to(device) @ Xg.T)[0]           # (P,)
            m = min(kk, rows.size)
            v, loc = torch.topk(sim, m)
            orders[qi, :m] = rows[loc.cpu().numpy()]              # map pool-local -> global doc id
            scores[qi, :m] = v.cpu().numpy()
            del Xg, sim
    return orders, scores


def _scoped_order_perdir(pos, X_t, hard_t, topP, device, bs=32, k=MAXK):
    """Mixture head, but keep EACH of the K answer-directions as its own doc order (nq, K, kk) instead
    of max-pooling — so a multi-gold query can be covered by fusing directions (each hop finds a gold)."""
    nq, K, _ = pos.shape
    n = X_t.shape[0]; kk = min(k, n)
    orders = np.full((nq, K, kk), -1, dtype=np.int64)
    with torch.no_grad():
        for s in range(0, nq, bs):
            p = pos[s:s + bs].to(device)                       # (b,K,d)
            sim = torch.einsum("bkd,nd->bkn", p, X_t)          # (b,K,n)
            for j in range(p.shape[0]):
                tp = topP[s + j]
                if tp:
                    allow = torch.isin(hard_t, torch.tensor(list(tp), device=device))
                    sim[j] = torch.where(allow.unsqueeze(0), sim[j], torch.full_like(sim[j], -1e9))
            orders[s:s + bs] = torch.topk(sim, kk, dim=2).indices.cpu().numpy()
    return orders


def _recall(order, gold_lists, ks=KS):
    out = {k: 0.0 for k in ks}; hit = {k: 0.0 for k in ks}; nq = 0
    for qi, g in enumerate(gold_lists):
        if not g:
            continue
        nq += 1; gs = set(g); row = order[qi]
        for k in ks:
            found = gs & set(int(x) for x in row[:k] if x >= 0)
            out[k] += len(found) / len(g)                      # fraction of ALL golds (strict)
            hit[k] += 1.0 if found else 0.0                    # >=1 gold = a usable seed (seed-finding metric)
    r = {k: round(100 * out[k] / max(nq, 1), 2) for k in ks}
    r.update({f"hit{k}": round(100 * hit[k] / max(nq, 1), 2) for k in ks})
    return r


def _pool_stats(hard, npart, topP):
    """Routed-pool size per query = # docs whose partition ∈ the top-K pool. Quantifies the memory claim:
    L2 need only hold/score P pool docs, not the full corpus N. reduction = N/mean(P). At scope=0 (empty
    topP) pool=N -> reduction 1 (honest: no scoping, no saving)."""
    harr = np.asarray(hard); n = len(harr)
    psize = np.bincount(harr, minlength=npart)                 # docs per partition (precomputed offline)
    sizes = np.array([int(psize[np.fromiter(tp, int)].sum()) if tp else n for tp in topP], dtype=np.int64)
    mp = float(sizes.mean())
    return {"corpus_N": int(n), "mean_pool": round(mp, 1), "max_pool": int(sizes.max()),
            "pool_reduction": round(n / max(1.0, mp), 2)}


def _gather_mem_demo(qt, holder, key, hard, topP, device, sample=96):
    """Demonstrate the gather saving with a REAL measured GPU peak: FULL (q@X^T, corpus resident) vs
    GATHER (corpus on CPU, only the routed pool rows fetched to GPU per query). Frees the resident matrix
    before the gather leg so its peak reflects a non-resident (disk/mmap) store. Best-effort; {} if not cuda."""
    import gc
    import torch
    if "cuda" not in str(device):
        return {}
    try:
        X_t = holder[key]; n = X_t.shape[0]; qs = qt[:sample]
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        s = qs @ X_t.T; _ = torch.topk(s, min(MAXK, n), dim=1)     # full-corpus score, matrix resident
        peak_full = torch.cuda.max_memory_allocated() / 1e6
        del s, _
        Xc = X_t.detach().to("cpu"); holder[key] = None; del X_t   # evict corpus from GPU (simulate gather store)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        harr = np.asarray(hard)
        for qi in range(min(sample, len(topP))):
            tp = topP[qi]
            rows = np.where(np.isin(harr, np.fromiter(tp, int)))[0] if tp else np.arange(n)
            Xg = Xc[torch.from_numpy(rows)].to(device)            # fetch ONLY the routed pool (P×d) to GPU
            r = qs[qi:qi + 1] @ Xg.T; _ = torch.topk(r, min(MAXK, Xg.shape[0]), dim=1)
            del Xg, r, _
        peak_gather = torch.cuda.max_memory_allocated() / 1e6
        del Xc; gc.collect(); torch.cuda.empty_cache()
        return {"peak_full_MB": round(peak_full, 1), "peak_gather_MB": round(peak_gather, 1),
                "mem_reduction": round(peak_full / max(1.0, peak_gather), 2)}
    except Exception as e:
        log.warning(f"[gather-demo/{key}] skipped ({e})")
        return {}


def _bm25_scoped_order(bm, texts, hard, topP, k=MAXK):
    """Lexical (BM25) doc order within each query's scoped partitions — an ORTHOGONAL axis to the
    dense/head orders. hard = doc->partition (corpus-aligned); docs outside topP[qi] are masked out."""
    n = len(hard); kk = min(k, n); harr = np.asarray(hard)
    orders = np.full((len(texts), kk), -1, dtype=np.int64)
    for qi, t in enumerate(texts):
        sc = bm.get_scores(str(t).lower().split()).astype(np.float32)
        tp = topP[qi]
        if tp:
            sc = np.where(np.isin(harr, list(tp)), sc, -1e9)
        orders[qi] = np.argsort(sc)[::-1][:kk]
    return orders


def _splade_query_vecs(spl, texts, dataset=None, bs=64):
    """Encode query texts with SPLADE -> sparse CSR (nq, vocab). DISK-CACHED per dataset (sparse) so the
    many L2 re-runs never re-encode. Model itself is cached module-level in splade_scorer."""
    import scipy.sparse
    cache = os.path.join("data", "ukb_storage", dataset, "splade_q_test.pkl") if dataset else None
    if cache and os.path.exists(cache):
        m = pickle.load(open(cache, "rb"))
        if m.shape[0] >= len(texts):
            log.info(f"[SPLADE] reused cached query vecs {m.shape} for {dataset}")
            return m[:len(texts)]
    spl._ensure_model()
    out = []
    for s in range(0, len(texts), bs):
        inp = spl._tokenizer([str(t) for t in texts[s:s + bs]], return_tensors="pt",
                             padding=True, truncation=True, max_length=64).to(spl._device)
        with torch.no_grad():
            lg = spl._model(**inp).logits
            v = torch.max(torch.log(1 + torch.relu(lg)) * inp.attention_mask.unsqueeze(-1), dim=1).values.cpu().numpy()
        out.append(scipy.sparse.csr_matrix(v))
    m = scipy.sparse.vstack(out).tocsr()
    if cache:
        pickle.dump(m, open(cache, "wb"))
        try:
            from src.experiments.backends import commit_persistent_storage
            commit_persistent_storage()                        # persist so later runs reuse (Modal volume)
        except Exception:
            pass
    return m


def _splade_scoped_order(spl_aligned, texts, hard, topP, dataset=None, k=MAXK):
    """SPLADE (learned-lexical) doc order within each query's scoped partitions — orthogonal to dense."""
    spl, aligned = spl_aligned
    qv = _splade_query_vecs(spl, texts, dataset)               # sparse CSR (nq, vocab)
    n = len(hard); kk = min(k, n); harr = np.asarray(hard)
    orders = np.full((len(texts), kk), -1, dtype=np.int64)
    for qi in range(len(texts)):
        sc = np.asarray(aligned.dot(qv[qi].T).todense()).ravel().astype(np.float32)   # (n,)
        tp = topP[qi]
        if tp:
            sc = np.where(np.isin(harr, list(tp)), sc, -1e9)
        orders[qi] = np.argsort(sc)[::-1][:kk]
    return orders


def _bestof(orders):
    nq = orders[0].shape[0]; out = []
    for qi in range(nq):
        best = {}
        for od in orders:
            for r, nd in enumerate(od[qi]):
                nd = int(nd)
                if nd >= 0 and (nd not in best or r < best[nd]):
                    best[nd] = r
        out.append([nd for nd, _ in sorted(best.items(), key=lambda kv: kv[1])])
    m = max((len(r) for r in out), default=1)
    return np.array([r + [-1] * (m - len(r)) for r in out], dtype=np.int64)


def run(datasets=None, subdir="gte_qwen", limit=8000, tr_cap=3000, te_cap=2000, epochs=20, K=8,
        scope_topk=50, device=None, with_bm25=False):
    datasets = datasets or DATASETS
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, per_ds = {}, {}
    for d in datasets:
        data[d] = _load(d, subdir, limit, tr_cap, te_cap)
        idx = faiss.IndexFlatIP(data[d]["X"].shape[1]); idx.add(data[d]["X"])   # "faiss" = one-time dense doc order; mining is GPU-topk over Xt
        per_ds[d] = {"train": data[d]["train"], "Xt": torch.tensor(data[d]["X"], device=device), "index": idx,
                     "faiss": idx}
        log.info(f"  loaded {d}: X{data[d]['X'].shape} npart={data[d]['npart']}")

    heads = {}
    for kind in ("hard", "mix_hard"):                     # L1 doc-recall MVPs: hard-negatives (rel_hard) + mlpT+hardneg
        log.info(f"=== training universal '{kind}' head (seed-finding, hard-neg mining) ===")
        heads[kind] = _train_universal(kind, per_ds, device, epochs, K=K)

    MIX = {"mlpT": True}                                   # which head names use MixtureHead (soft-OR) ordering
    out = {}
    for d in datasets:
        X = data[d]["X"]; hard = data[d]["hard"]; mem_idx = data[d]["mem_idx"]; npart = data[d]["npart"]
        qte, ste, gte = data[d]["test"]
        X_t = per_ds[d]["Xt"]; hard_t = torch.tensor(hard, device=device)
        _, I = per_ds[d]["faiss"].search(qte, MAXK)                      # dense doc order (for voting)
        if not scope_topk:                                     # full-corpus (no partition scope): empty topP -> order fns skip
            topP = [set()] * len(qte)                          # the mask entirely (avoids a wasteful full-set isin on 507k docs)
            ceil = 1.0                                          # every gold reachable when nothing is scoped out
        else:
            topP = _topP(I, mem_idx, npart, scope_topk)
            ceil = float(np.mean([np.mean([1.0 if hard[g] in topP[qi] else 0.0 for g in gg]) for qi, gg in enumerate(gte) if gg]))
        with torch.no_grad():
            qt = torch.tensor(qte, device=device); sv = X_t[torch.tensor(ste, device=device)]
            pos = {"dense": qt, "rel_hard": heads["hard"](qt, sv), "mlpT": heads["mix_hard"](qt, sv)}
        so = {m: _scoped_order(pos[m].cpu(), X_t, hard_t, topP, m in MIX, device) for m in pos}
        orders = {m: so[m][0] for m in pos}
        orders["dense+rel_hard"] = _bestof([orders["dense"], orders["rel_hard"]])
        orders["dense+mlpT"] = _bestof([orders["dense"], orders["mlpT"]])
        three = _bestof([orders["dense"], orders["rel_hard"], orders["mlpT"]])
        orders["dense+rel_hard+mlpT"] = three
        dsc = so["dense"][1]                                    # gated 3-way: dense where it's already confident, else 3-way
        margin = dsc[:, 0] - dsc[:, min(20, dsc.shape[1]) - 1]  # dense top1-vs-top20 sim gap = confidence
        thr = float(np.quantile(margin, 0.5))                   # per-query select (dense/3-way rows differ in width -> ragged)
        orders["gated_3way"] = [orders["dense"][qi] if margin[qi] >= thr else three[qi] for qi in range(len(margin))]
        bm = data[d].get("bm25") if with_bm25 else None        # BM25 get_scores loops the corpus/query (slow) + it's a wash -> opt-in
        if bm is not None:
            bmo = _bm25_scoped_order(bm, data[d]["test_texts"], hard, topP)
            orders["bm25"] = bmo
            orders["dense+mlpT+bm25"] = _bestof([orders["dense"], orders["mlpT"], bmo])
            orders["3way+bm25"] = _bestof([orders["dense"], orders["rel_hard"], orders["mlpT"], bmo])
        sp = data[d].get("splade")                             # HYBRID: learned-lexical (SPLADE) axis
        if sp is not None:
            spo = _splade_scoped_order(sp, data[d]["test_texts"], hard, topP, dataset=d)
            orders["splade"] = spo
            orders["dense+mlpT+splade"] = _bestof([orders["dense"], orders["mlpT"], spo])
            orders["3way+splade"] = _bestof([orders["dense"], orders["rel_hard"], orders["mlpT"], spo])
        out[d] = {"pool_ceiling@" + str(scope_topk): round(100 * ceil, 2)}
        out[d].update({m: _recall(orders[m], gte) for m in orders})
        out[d].update(_pool_stats(hard, npart, topP))          # memory claim: routed-pool size N/P
        log.info(f"[{d}] ceil={out[d]['pool_ceiling@'+str(scope_topk)]} | " +
                 " | ".join(f"{m}@20={out[d][m][20]}" for m in orders))
        log.info(f"[{d}/mem] N={out[d]['corpus_N']} mean_pool={out[d]['mean_pool']} "
                 f"reduction={out[d]['pool_reduction']}x")
        if scope_topk:                                         # skip at scope=0 (no pool -> would gather full corpus, slow+pointless)
            out[d].update(_gather_mem_demo(qt, per_ds[d], "Xt", hard, topP, device))  # LAST: evicts GPU X_t
        if "mem_reduction" in out[d]:
            log.info(f"[{d}/mem] peak_full={out[d]['peak_full_MB']}MB peak_gather={out[d]['peak_gather_MB']}MB "
                     f"({out[d]['mem_reduction']}x)")
        import gc; gc.collect()

    methods = [m for m in ["dense", "rel_hard", "mlpT", "dense+rel_hard", "dense+mlpT", "dense+rel_hard+mlpT",
                           "gated_3way", "bm25", "dense+mlpT+bm25", "3way+bm25",
                           "splade", "dense+mlpT+splade", "3way+splade"]
               if all(m in out[d] for d in datasets)]
    for kk in KS:
        print(f"\n=== L2 gold-DOC recall@{kk} within top-{scope_topk} partitions ({subdir}) ===")
        print(f"{'dataset':16s} {'ceil':>6s}  " + " ".join(f"{m:>13s}" for m in methods))
        for d in datasets:
            print(f"{d:16s} {out[d]['pool_ceiling@'+str(scope_topk)]:6.1f}  " +
                  " ".join(f"{out[d][m][kk]:13.2f}" for m in methods))
        print(f"{'MEAN':16s} {'':>6s}  " + " ".join(f"{np.mean([out[d][m][kk] for d in datasets]):13.2f}" for m in methods))
    os.makedirs("results/L2", exist_ok=True)
    _tag = f"top{scope_topk}_K{K}"                             # keep scope+K in the name so variants don't clobber
    json.dump(out, open(f"results/L2/L2_seed_{subdir}_{_tag}.json", "w"), indent=2)
    log.info(f"-> results/L2/L2_seed_{subdir}_{_tag}.json")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L2 seed-finding within top-K partitions: dense vs offset/mlpT doc-recall.")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--subdir", default="gte_qwen")
    p.add_argument("--tr-cap", type=int, default=3000)
    p.add_argument("--te-cap", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--scope-topk", type=int, default=50, help="partitions paged in (0 = full corpus)")
    p.add_argument("--with-bm25", action="store_true", help="also compute the (slow, wash) BM25 lexical baseline")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, subdir=a.subdir, tr_cap=a.tr_cap, te_cap=a.te_cap, epochs=a.epochs, K=a.K,
        scope_topk=a.scope_topk, with_bm25=a.with_bm25)


if __name__ == "__main__":
    main()
