"""
STEP 1 — the best SINGLE universal relational head.
===================================================
Find the ONE offset head, trained ONCE across all datasets in the shared frozen space (BGE-large),
that best recovers golds. "It might not be mlpT" — so we COMPARE candidates head-to-head:
  base      OffsetHead, in-batch negatives (plain learned offset)
  hard      OffsetHead + hard-negative mining        (roadmap MVP)
  mix       MixtureHead K directions, in-batch        (mlpT)
  mix_hard  MixtureHead K directions + hard negatives (mlpT + MVP)

All are trained on the SAME pooled (query->gold) triples across all 5 datasets (one model, not
per-dataset). Hard-negatives are mined from a COMBINED index over all datasets' docs. Each head is
then evaluated per-dataset: standalone partition-FullCov@20 AND fused with dense via equal-RRF
(step-2 fusion, static). Winner = the single universal head to carry forward.

Needs ample RAM (combined doc index ~1-2 GB) -> run on Modal-CPU or Lightning, not the laptop.
Query embeddings are cached into the subdir (queries_{split}.npy) so reruns skip encoding.
Writes results/L1_universal_head.json.
"""
import os
import gc
import json
import random as _random
import logging
import argparse

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # defrag: fit gte (1536-d) on A10G

import numpy as np
import torch
import torch.nn.functional as F
import faiss

from src.core.engine import CoreEngine
from src.experiments.encoder_swap import load_docs_and_encoder
from src.experiments.overlap_retrain import _splits, _hard_membership, _onehop_membership, INIT_SEED
from src.experiments.l1_ablate import MixtureHead, MAXK, TAU
from src.experiments.query_relation import OffsetHead
from src.experiments.l1_rerank100 import _feats, _rr, _fullcov

log = logging.getLogger("experiments.l1_universal_head")

DATASETS = ["musique_clean", "2wiki_clean", "squad_clean", "metaqa", "hotpotqa_clean"]
HEADS = ["base", "hard", "mix", "mix_hard"]


def _load(dataset, subdir, limit, tr_cap, te_cap):
    """Load one dataset in the frozen space; encode+cache train/test queries; return everything
    needed both for pooled training (seeds/golds as LOCAL indices) and per-dataset eval."""
    eng = CoreEngine(source=dataset, index_subdir=subdir)   # per-encoder graph + partitions
    X, eq, tag = load_docs_and_encoder(eng, dataset, subdir)
    X = X.astype("float32")
    n = X.shape[0]
    npart = max(int(p) for p in eng.partition_map.values()) + 1
    id2idx = eng.node_id_to_idx
    idx2id = {v: k for k, v in id2idx.items()}
    hard = np.array([int(eng.partition_map.get(idx2id[i], -1)) for i in range(n)])
    mem = _onehop_membership(eng)
    mem_idx = [sorted(mem.get(idx2id[i], {int(hard[i])})) for i in range(n)]
    sp = _splits(eng, _hard_membership(eng))
    caps = {"train": tr_cap or limit, "test": te_cap or limit}

    def prep(split):
        qs = sp[split][:caps[split]]
        cache = os.path.join("data", "ukb_storage", dataset, subdir, f"queries_{split}.npy") if subdir else None
        if cache and os.path.exists(cache) and len(np.load(cache)) >= len(qs):
            q = np.load(cache)[:len(qs)].astype("float32")
        else:
            q = eq([nd.content for nd, _, _ in qs]).astype("float32")
            if cache:
                np.save(cache, q)                          # cache for future runs (no re-encode)
        seed = np.argmax(q @ X.T, axis=1).astype(np.int64)
        gold = [[id2idx[g] for g in gg if g in id2idx] for _, _, gg in qs]
        return q, seed, gold, [nd.content for nd, _, _ in qs]

    qtr, str_, gtr, tr_texts = prep("train")
    qte, ste, gte, te_texts = prep("test")
    splade = None                                              # optional learned-lexical axis for L2 hybrid
    try:
        from src.core.splade_scorer import SpladeScorer
        _spl = SpladeScorer(dataset)
        if _spl.available():
            _spl._ensure_matrix()
            _row = {nid: r for r, nid in _spl._idx_to_id.items()}          # node_id -> matrix row
            _perm = np.array([_row.get(idx2id[i], -1) for i in range(n)])  # X row -> matrix row
            if (_perm >= 0).all():
                splade = (_spl, _spl._matrix[_perm])           # (scorer, CSR reordered to X/hard alignment)
    except Exception as _e:
        log.warning(f"[SPLADE] unavailable for {dataset}: {_e}")
    return {"X": X, "n": n, "npart": npart, "mem_idx": mem_idx, "hard": hard, "tag": tag,
            "train": (qtr, str_, gtr), "test": (qte, ste, gte),
            "test_texts": te_texts, "train_texts": tr_texts,               # query texts for lexical (SPLADE) signals
            "bm25": getattr(eng, "bm25", None),
            "splade": splade}


def _head_order(head, q, seed_vec, X, device, k=MAXK, bs=256):
    """Retrieval order per query. Handles OffsetHead (B,d) and MixtureHead soft-OR (B,K,d)."""
    Xd = torch.tensor(X, device=device)
    bs = max(8, min(bs, 20_000_000 // max(1, X.shape[0])))  # cap (bs,K,N) einsum on hotpot-scale N (OOM guard)
    kk = min(k, X.shape[0])
    orders = np.empty((len(q), kk), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(q), bs):
            qn = torch.tensor(q[s:s + bs], device=device)
            sd = torch.tensor(seed_vec[s:s + bs], device=device)
            pos = head(qn, sd)
            sim = (torch.einsum("bkd,nd->bkn", pos, Xd).max(1).values if pos.dim() == 3 else pos @ Xd.T)
            orders[s:s + bs] = torch.topk(sim, kk, dim=1).indices.cpu().numpy()
    del Xd
    return orders


def _dense_order(q, X, device, k=MAXK, bs=256):
    Xd = torch.tensor(X, device=device)
    kk = min(k, X.shape[0])
    orders = np.empty((len(q), kk), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(q), bs):
            sim = torch.tensor(q[s:s + bs], device=device) @ Xd.T
            orders[s:s + bs] = torch.topk(sim, kk, dim=1).indices.cpu().numpy()
    del Xd
    return orders


def _votes(order, mem_idx, npart):
    S, M = _feats(order, mem_idx, npart)
    return _rr(S) + _rr(M)


def _bestof(orders, mem_idx, npart, k0=60, topn=200):
    """Best-of fusion: per node take its MIN rank across the given retriever orders, then MAX-vote per
    partition (weight 1/(k0+minrank)). Keeps each retriever's STRONGEST signal instead of averaging it,
    so a head that's weak on a saturated dataset can't dilute dense's strong votes."""
    nq = orders[0].shape[0]
    M = np.zeros((nq, npart), np.float32)
    for qi in range(nq):
        best = {}
        for od in orders:
            row = od[qi]
            for r in range(len(row)):
                nd = int(row[r])
                if nd not in best or r < best[nd]:
                    best[nd] = r
        for nd, bn in sorted(best.items(), key=lambda kv: kv[1])[:topn]:
            w = 1.0 / (k0 + bn)
            for p in mem_idx[nd]:
                if w > M[qi, p]:
                    M[qi, p] = w
    return _rr(M)


def _rrf(orders, mem_idx, npart, k0=60, topn=200):
    """Reciprocal-rank fusion: per node SUM 1/(k0+rank) across orders (vs bestof's MIN rank), then
    MAX-vote per partition. Rewards nodes several retrievers agree on — complements min-rank's 'any
    retriever's strong hit'. Same frozen orderings, different combine rule."""
    nq = orders[0].shape[0]
    M = np.zeros((nq, npart), np.float32)
    for qi in range(nq):
        acc = {}
        for od in orders:
            row = od[qi]
            for r in range(len(row)):
                nd = int(row[r]); acc[nd] = acc.get(nd, 0.0) + 1.0 / (k0 + r)
        for nd, sc in sorted(acc.items(), key=lambda kv: -kv[1])[:topn]:
            for p in mem_idx[nd]:
                if sc > M[qi, p]:
                    M[qi, p] = sc
    return _rr(M)


def _gated(v_dense, v_bestof, frac=0.5):
    """Confidence gate: for the most-confident `frac` of queries (large dense top1-vs-top20 vote margin ->
    gold already locked in dense's top) use DENSE ONLY (the head only risks adding noise); for the rest
    (dense unsure) use best-of. Stops the head from hurting saturated datasets."""
    srt = np.sort(v_dense, axis=1)
    margin = srt[:, -1] - srt[:, -min(20, srt.shape[1])]
    thr = np.quantile(margin, frac)
    out = v_bestof.copy()
    out[margin >= thr] = v_dense[margin >= thr]
    return out


def _train_universal(kind, per_ds, device, epochs, K=8, bs=256):
    """ONE shared head trained across all datasets; hard-negs mined from each dataset's OWN (small)
    index -> fast + same-domain negatives (the combined-index mining was ~3-15x slower). Mirrors
    l1_ablate._train_offset's loss, generalized to iterate datasets updating a single head."""
    dsl = list(per_ds.keys())
    dim = per_ds[dsl[0]]["Xt"].shape[1]
    torch.manual_seed(INIT_SEED)
    head = (MixtureHead(dim, K) if kind.startswith("mix") else OffsetHead(dim)).to(device)
    import hashlib                                              # cache the trained head (deterministic) so re-runs skip the ~13min hotpot mining
    fp = hashlib.md5(f"{kind}|K{K}|e{epochs}|d{dim}".encode())
    for d in sorted(per_ds.keys()):                            # fingerprint = config + seeds/golds (data version) + encoder sample
        fp.update(d.encode()); fp.update(np.asarray(per_ds[d]["train"][1]).tobytes())
        fp.update(np.asarray([g for gl in per_ds[d]["train"][2] for g in gl], dtype=np.int64).tobytes())
        fp.update(np.ascontiguousarray(per_ds[d]["Xt"][:64].detach().cpu().numpy()).tobytes())
    _hcache = os.path.join("data", "ukb_storage", "_head_cache", f"head_{fp.hexdigest()[:16]}.pt")
    if os.path.exists(_hcache):
        try:
            head.load_state_dict(torch.load(_hcache, map_location=device)); head.eval()
            log.info(f"[head-cache] reused {kind} K={K} <- {os.path.basename(_hcache)}")
            return head
        except Exception as _e:
            log.warning(f"[head-cache] load failed ({_e}); retraining")
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    trips = {d: [(i, int(s), int(g)) for i, (s, gl) in enumerate(zip(per_ds[d]["train"][1], per_ds[d]["train"][2]))
                 for g in gl] for d in dsl}
    qts = {d: torch.tensor(per_ds[d]["train"][0], device=device) for d in dsl}
    for ep in range(epochs):
        head.train()
        order = dsl[:]; _random.Random(ep).shuffle(order)
        for d in order:
            Xt = per_ds[d]["Xt"]; index = per_ds[d]["index"]; qt = qts[d]
            trip = trips[d]; _random.Random(ep * 131 + len(d)).shuffle(trip)
            for s in range(0, len(trip), bs):
                b = trip[s:s + bs]
                qn = qt[[t[0] for t in b]]; seedv = Xt[[t[1] for t in b]]; goldv = Xt[[t[2] for t in b]]
                own = torch.tensor([t[2] for t in b], device=device)
                if kind.startswith("mix"):
                    pred = head(qn, seedv)                             # (B,K,d)
                    sim = torch.einsum("bkd,jd->bkj", pred, goldv).max(1).values / TAU
                    if kind == "mix_hard":
                        with torch.no_grad():                 # GPU topk over Xt (already on GPU): no faiss, no CPU transfer
                            neg = torch.topk(pred.reshape(-1, dim) @ Xt.T, 8, dim=1).indices.reshape(-1)
                        hard = torch.einsum("bkd,nd->bkn", pred, Xt[neg]).max(1).values / TAU
                        sim = torch.cat([sim, hard.masked_fill(neg.unsqueeze(0) == own.unsqueeze(1), -1e4)], dim=1)
                    loss = F.cross_entropy(sim, torch.arange(len(b), device=device))
                else:
                    pred = head(qn, seedv)                             # (B,d)
                    logits = pred @ goldv.T / TAU
                    if kind == "hard":
                        with torch.no_grad():                 # GPU topk over Xt: no faiss, no CPU transfer (was O(corpus)/batch on CPU)
                            neg = torch.topk(pred @ Xt.T, 16, dim=1).indices.reshape(-1)
                        hard = (pred @ Xt[neg].T / TAU).masked_fill(neg.unsqueeze(0) == own.unsqueeze(1), -1e4)
                        logits = torch.cat([logits, hard], dim=1)
                    loss = F.cross_entropy(logits, torch.arange(len(b), device=device))
                opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    try:                                                       # persist so subsequent runs skip retraining
        os.makedirs(os.path.dirname(_hcache), exist_ok=True)
        torch.save(head.state_dict(), _hcache)
        from src.experiments.backends import commit_persistent_storage
        commit_persistent_storage()
        log.info(f"[head-cache] saved {kind} -> {os.path.basename(_hcache)}")
    except Exception as _e:
        log.warning(f"[head-cache] save failed ({_e})")
    return head


def run(datasets=None, subdir="bge_large", limit=8000, tr_cap=3000, te_cap=2000, epochs=20, K=8,
        heads=None, device=None):
    datasets = datasets or DATASETS
    heads = heads or HEADS
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load all datasets; per-dataset index (small -> fast mining) + Xt for a SHARED head ----
    data, per_ds = {}, {}
    for d in datasets:
        data[d] = _load(d, subdir, limit, tr_cap, te_cap)
        idx = faiss.IndexFlatIP(data[d]["X"].shape[1]); idx.add(data[d]["X"])   # kept for API symmetry; mining is GPU-topk over Xt
        per_ds[d] = {"train": data[d]["train"], "Xt": torch.tensor(data[d]["X"], device=device), "index": idx}
        log.info(f"  loaded {d}: X{data[d]['X'].shape} npart={data[d]['npart']} enc={data[d]['tag']}")
    n_tr = sum(len(per_ds[d]["train"][0]) for d in datasets)
    log.info(f"per-dataset indices built; ~{n_tr} pooled train queries -> ONE shared head, per-dataset mining")

    # ---- train each candidate head ONCE (shared across datasets); eval per dataset ----
    dense_cache = {}
    results = {h: {} for h in heads}
    head_orders = {d: {} for d in datasets}                    # keep each head's doc order for cross-head fusion
    for h in heads:
        log.info(f"=== training universal head '{h}' (epochs={epochs}, K={K}) ===")
        head = _train_universal(h, per_ds, device, epochs, K=K)
        for d in datasets:
            X = data[d]["X"]; mem_idx = data[d]["mem_idx"]; npart = data[d]["npart"]
            qte, ste, gte = data[d]["test"]
            gpl = [[mem_idx[g] for g in gg] for gg in gte]
            ho = _head_order(head, qte, X[ste], X, device)
            head_orders[d][h] = ho                             # cache for cross-head combination sweep
            v_head = _votes(ho, mem_idx, npart)
            if d not in dense_cache:
                do = _dense_order(qte, X, device)
                dense_cache[d] = (do, _votes(do, mem_idx, npart))
            do, v_dense = dense_cache[d]
            v_bestof = _bestof([do, ho], mem_idx, npart)          # min-rank fusion (no dilution)
            v_gated = _gated(v_dense, v_bestof)                    # dense where confident, best-of where not
            results[h][d] = {
                "dense@20": _fullcov(v_dense, gpl, npart)[20],
                "head@20": _fullcov(v_head, gpl, npart)[20],
                "equal@20": _fullcov(v_dense + v_head, gpl, npart)[20],
                "bestof@20": _fullcov(v_bestof, gpl, npart)[20],
                "gated@20": _fullcov(v_gated, gpl, npart)[20],
                "n_test": len(gpl),
            }
            r = results[h][d]
            log.info(f"  [{h}/{d}] dense={r['dense@20']} head={r['head@20']} equal={r['equal@20']} "
                     f"bestof={r['bestof@20']} gated={r['gated@20']}")
        del head; gc.collect()

    # ---- COMBINATION SWEEP: fuse frozen orderings (heads + optional SPLADE lexical axis) ----
    spl_ord = {}                                               # full-corpus SPLADE doc order per dataset (L1 = no scope)
    try:
        from src.experiments.l2_seed import _splade_scoped_order   # deferred: avoid circular import
        for d in datasets:
            sp = data[d].get("splade")
            if sp is not None:
                spl_ord[d] = _splade_scoped_order(sp, data[d]["test_texts"], data[d]["hard"],
                                                  [None] * len(data[d]["test_texts"]), dataset=d)  # topP=None -> full corpus
    except Exception as _e:
        log.warning(f"[splade-L1] unavailable ({_e})")
    COMBOS = {                                                 # name -> (heads to fuse w/ dense, use_splade, fusion fn, gate_frac|None)
        "d+hard+mlpT|bestof":       (["hard", "mix_hard"], False, _bestof, None),
        "d+all|bestof":             (["base", "hard", "mix", "mix_hard"], False, _bestof, None),
        "d+splade|bestof":          ([], True, _bestof, None),
        "d+hard+splade|bestof":     (["hard"], True, _bestof, None),
        "d+hard+mlpT+splade|bestof": (["hard", "mix_hard"], True, _bestof, None),
        # gated: use DENSE for the most-confident `frac` of queries (recovers saturated hotpot),
        # the splade combo for the rest (keeps the 2wiki/musique multi-hop lift). Pure combination layer.
        "d+hard+splade|gated0.5":   (["hard"], True, _bestof, 0.5),
        "d+hard+splade|gated0.3":   (["hard"], True, _bestof, 0.3),
        "d+splade|gated0.5":        ([], True, _bestof, 0.5),
    }
    combo_res = {c: {} for c in COMBOS}
    for d in datasets:
        mem_idx = data[d]["mem_idx"]; npart = data[d]["npart"]
        gpl = [[mem_idx[g] for g in gg] for gg in data[d]["test"][2]]
        do = dense_cache[d][0]; v_dense = dense_cache[d][1]
        for c, (hs, use_spl, fn, gate) in COMBOS.items():
            if use_spl and d not in spl_ord:
                continue
            orders = [do] + [head_orders[d][h] for h in hs if h in head_orders[d]]
            if use_spl:
                orders.append(spl_ord[d])
            fused = fn(orders, mem_idx, npart)
            if gate is not None:
                fused = _gated(v_dense, fused, frac=gate)          # dense where confident, else the fused combo
            combo_res[c][d] = _fullcov(fused, gpl, npart)[20]
        log.info("  [combos/%s] %s", d, {c: combo_res[c].get(d) for c in COMBOS})

    # ---- summarize: best (head, fusion) combo by mean over datasets ----
    FUS = ["equal", "bestof", "gated"]
    summary = {"encoder_subdir": subdir, "K": K, "epochs": epochs, "tr_cap": tr_cap,
               "datasets": datasets, "heads": {}}
    dense_mean = round(float(np.mean([results[heads[0]][d]["dense@20"] for d in datasets])), 2)
    for h in heads:
        entry = {"mean_head@20": round(float(np.mean([results[h][d]["head@20"] for d in datasets])), 2),
                 "per_dataset": results[h]}
        for fz in FUS:
            entry[f"mean_{fz}@20"] = round(float(np.mean([results[h][d][f"{fz}@20"] for d in datasets])), 2)
            entry[f"over95_{fz}"] = f"{sum(1 for d in datasets if results[h][d][f'{fz}@20'] >= 95)}/{len(datasets)}"
        summary["heads"][h] = entry
    combos = [(h, fz, summary["heads"][h][f"mean_{fz}@20"]) for h in heads for fz in FUS]
    for c in COMBOS:                                            # fold combinations into the best search (robust to missing splade)
        avail = [combo_res[c][d] for d in datasets if d in combo_res[c]]
        if not avail:
            continue
        m = round(float(np.mean(avail)), 2)
        summary.setdefault("combinations", {})[c] = {"mean@20": m, "n": len(avail), "per_dataset": combo_res[c]}
        combos.append((c, "combo", m))
    best_h, best_fz, best_m = max(combos, key=lambda x: x[2])
    summary["dense_only_mean@20"] = dense_mean
    summary["best"] = {"head": best_h, "fusion": best_fz, "mean@20": best_m}
    os.makedirs("results", exist_ok=True)
    json.dump(summary, open(os.path.join("results", f"L1_universal_head_{subdir}.json"), "w"), indent=2)
    log.info("==== UNIVERSAL HEAD x FUSION (mean partition FullCov@20) ====")
    log.info(f"  dense-only: {dense_mean}")
    for h in heads:
        s = summary["heads"][h]
        log.info(f"  {h:9s} head={s['mean_head@20']:.2f}  equal={s['mean_equal@20']:.2f}  "
                 f"bestof={s['mean_bestof@20']:.2f}({s['over95_bestof']})  gated={s['mean_gated@20']:.2f}({s['over95_gated']})")
    for c in COMBOS:
        log.info(f"  COMBO {c:22s} mean@20={summary['combinations'][c]['mean@20']}")
    log.info(f"  BEST = head '{best_h}' + {best_fz} = {best_m} -> results/L1_universal_head_{subdir}.json")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="STEP 1: best single universal relational head (compare candidates).")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--subdir", default="bge_large")
    p.add_argument("--limit", type=int, default=8000)
    p.add_argument("--tr-cap", type=int, default=3000)
    p.add_argument("--te-cap", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--heads", nargs="+", default=None)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, subdir=a.subdir, limit=a.limit, tr_cap=a.tr_cap, te_cap=a.te_cap,
        epochs=a.epochs, K=a.K, heads=a.heads)


if __name__ == "__main__":
    main()
