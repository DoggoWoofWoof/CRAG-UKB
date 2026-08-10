"""
L1 UNIFIED PARTITION RANKER (UPR) — one novel dataset-agnostic model, everything fused.
=======================================================================================
L1's job: rank PARTITIONS (candidate generation for L2/L3). This is the single model that
fuses every signal discussed. Per (query, partition p) it builds evidence features:

  centroid_sim   q . centroid_p                          semantic route
  route_mlp      learned multi-signal router (q+seed+nbr) -> centroids
  vote_dense     dense node retrieval  -> partition votes
  vote_relhard   rel_hard OFFSET node retrieval -> votes  (relational / KG signal)
  vote_rel2hop   trained 2-hop OFFSET node retrieval -> votes  (multi-hop KB signal)
  vote_mlpT      mlpT K node-direction union -> votes     (1-to-many / multi-partition)

Node->partition vote: score(p) = sum_r 1/(k0+r) * [p in OVERLAP-membership(node_r)], so a
node votes for its own + neighbour partitions (cross-boundary multi-hop routable without
traversal). The relational offsets are the KG signal and contribute regardless of dataset
(esp. metaqa/2wiki/musique). Features are standardized PER QUERY, so they're comparable
across datasets and ONE model generalizes. A small MLP learns the partition logit; multi-
label softmax-KL vs the gold-partition set.

Models: learned_perds (per dataset) and learned_univ (ONE model pooled over ALL datasets =
the dataset-agnostic ranker). Reported vs the best fixed node-vote fusion and each individual
vote (so we SEE where the relational offset earns its keep). Headline = partition FullCov@P.
Modes:
  --mode solo       compute features, train+eval per-ds combiner, CACHE features (npz)
  --mode universal  load cached features from --datasets, train ONE universal model, eval each
Writes L1_select/unified_ranker_{ds}.json (+ cache upr_feat_{ds}.npz). Relative within-run;
winner re-verified on the frozen substrate via the canonical harness before it is a paper number.
"""
import os
import json
import random
import logging
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import (
    _reconstruct, _hard_membership, _onehop_membership, _centroids, _splits, TAU, HNK, INIT_SEED,
)
from src.experiments.multisignal_route import MultiRouter, _concat, _train_router, KS
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse, MAXK
from src.experiments.l1_dynamic import _train_hop2

log = logging.getLogger("experiments.l1_unified_ranker")
K0 = 60
FEATURES = ["centroid_sim", "route_mlp", "vote_dense", "vote_relhard", "vote_rel2hop", "vote_mlpT"]


class Combiner(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _cov_pre(order, gold_part_lists, npart, budgets=KS):
    """Partition FullCov / gt-recall from precomputed per-gold partition lists (no membership)."""
    fc = {k: [] for k in budgets}; gt = {k: [] for k in budgets}
    for qi, gpl in enumerate(gold_part_lists):
        if not gpl:
            continue
        rank_of = {int(p): r for r, p in enumerate(order[qi])}
        best = [min((rank_of.get(p, npart) for p in gp), default=npart) for gp in gpl]
        for k in budgets:
            cov = sum(1 for b in best if b < k)
            fc[k].append(1.0 if cov == len(best) else 0.0)
            gt[k].append(cov / len(best))
    return ({k: round(float(np.mean(fc[k])) * 100, 2) for k in budgets},
            {k: round(float(np.mean(gt[k])) * 100, 2) for k in budgets})


def _prep(qs, X, encoder, index, id2idx, topk=10):
    q = encoder.encode([n.content for n, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
    _, order = index.search(q, topk)
    seed_idx = order[:, 0]
    return {"q": q, "seed": X[seed_idx].astype("float32"), "nbr": X[order].mean(axis=1).astype("float32"),
            "seed_idx": seed_idx,
            "gold_idx": [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]}


def _vote(node_lists, mem_idx, npart, topn, k0=K0):
    nq = len(node_lists); scores = np.zeros((nq, npart), np.float32)
    for qi in range(nq):
        for r, nd in enumerate(node_lists[qi][:topn]):
            w = 1.0 / (k0 + r)
            for p in mem_idx[int(nd)]:
                scores[qi, p] += w
    return scores


def _standardize(mats):
    F_ = np.stack([mats[k] for k in FEATURES], axis=-1).astype("float32")   # (nq,npart,D)
    mu = F_.mean(axis=1, keepdims=True); sd = F_.std(axis=1, keepdims=True) + 1e-6
    return (F_ - mu) / sd


def _node_orders(heads, sig, Xt, index, device, K):
    g1, g_hard, g_mix, g2 = heads
    qte = torch.tensor(sig["q"], device=device)
    seed_t = Xt[torch.tensor(sig["seed_idx"], device=device)]

    def pos(head):
        with torch.no_grad():
            return head(qte, seed_t).cpu().numpy()
    dense = _order(sig["q"], index)
    relhard = _order(pos(g_hard), index)
    base = _order(pos(g1), index)
    s1 = base[:, 0]
    with torch.no_grad():
        hop2 = _order(g2(qte, Xt[torch.tensor(s1, device=device)]).cpu().numpy(), index)
    rel2hop = _rrf_fuse([_ranks(base), _ranks(hop2)], [1.0, 1.0], k=MAXK)
    with torch.no_grad():
        mixp = g_mix(qte, seed_t).cpu().numpy()                              # (nq,K,d)
    mlpT = _rrf_fuse([_ranks(_order(np.ascontiguousarray(mixp[:, k, :]), index)) for k in range(K)],
                     [1.0] * K, k=MAXK)
    return {"vote_dense": dense, "vote_relhard": relhard, "vote_rel2hop": rel2hop, "vote_mlpT": mlpT}


def _features_for_split(sig, heads, route_model, C, Cg, mem_idx, npart, Xt, index, device, K, topn):
    node_orders = _node_orders(heads, sig, Xt, index, device, K)
    mats = {"centroid_sim": sig["q"] @ C.T}
    with torch.no_grad():
        f = _concat(sig, ["q", "seed", "nbr"])
        mats["route_mlp"] = (route_model(torch.tensor(f, device=device)) @ Cg.T).cpu().numpy()
    for name, order in node_orders.items():
        mats[name] = _vote(order, mem_idx, npart, topn)
    return _standardize(mats), node_orders


def _teacher(gp, npart, device):
    t = torch.zeros(len(gp), npart, device=device)
    for i, ps in enumerate(gp):
        for p in ps:
            t[i, p] = 1.0
    return t / t.sum(dim=1, keepdim=True).clamp_min(1e-9)


def _train_combiner(blocks, D, device, epochs=60, lr=1e-3):
    """blocks: list of {F_tr,gp_tr,F_va,gpl_va,npart}. Pools across datasets (universal if >1)."""
    torch.manual_seed(INIT_SEED)
    model = Combiner(D).to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    ten = [{"Ft": torch.tensor(b["F_tr"], device=device), "teach": _teacher(b["gp_tr"], b["npart"], device),
            "Fv": torch.tensor(b["F_va"], device=device)} for b in blocks]
    best, best_state, noimp = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for bi, b in enumerate(blocks):
            Ft = ten[bi]["Ft"]; teach = ten[bi]["teach"]; nq = Ft.shape[0]
            idx = list(range(nq)); random.Random(ep * 97 + bi).shuffle(idx)
            for s in range(0, nq, 128):
                sel = idx[s:s + 128]
                loss = F.kl_div(F.log_softmax(model(Ft[sel]), dim=1), teach[sel], reduction="batchmean")
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); vs = []
        with torch.no_grad():
            for bi, b in enumerate(blocks):
                lg = model(ten[bi]["Fv"]).cpu().numpy()
                vs.append(_cov_pre(np.argsort(-lg, axis=1), b["gpl_va"], b["npart"])[0][20])
        v = float(np.mean(vs))
        if v > best:
            best, best_state, noimp = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}, 0
        else:
            noimp += 1
        if noimp >= 12:
            break
    model.load_state_dict(best_state); model.eval()
    return model


def _gp_gpl(splits, split, membership):
    """Per query: gp = union of golds' partitions (teacher); gpl = per-gold partition lists (eval)."""
    gp_list, gpl_list = [], []
    for (_, _, golds) in splits[split]:
        gpl = [sorted(membership[g]) for g in golds if g in membership]
        gp = sorted(set(p for lst in gpl for p in lst))
        gp_list.append(gp); gpl_list.append(gpl)
    return gp_list, gpl_list


def _compute_dataset(dataset, epochs, off_epochs, limit, K, topn, device):
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index); X = node_vecs.astype("float32"); faiss.normalize_L2(X)
    n = X.shape[0]; npart = max(int(p) for p in engine.partition_map.values()) + 1
    id2idx = engine.node_id_to_idx; idx2id = {v: k for k, v in id2idx.items()}
    hard = np.array([int(engine.partition_map.get(idx2id[i], -1)) for i in range(n)])
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X); Xt = torch.tensor(X, device=device)

    splits = _splits(engine, _hard_membership(engine))
    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}
    sig = {s: _prep(splits[s], X, encoder, index, id2idx) for s in ("train", "val", "test")}

    tr = sig["train"]
    log.info(f"[{dataset}] training offset heads (base, hard, mix, 2hop)...")
    g1 = _train_offset("base", tr["q"], tr["seed_idx"], tr["gold_idx"], Xt, index, device, off_epochs)
    g_hard = _train_offset("hard", tr["q"], tr["seed_idx"], tr["gold_idx"], Xt, index, device, off_epochs)
    g_mix = _train_offset("mix", tr["q"], tr["seed_idx"], tr["gold_idx"], Xt, index, device, off_epochs, K=K)
    g2 = _train_hop2(g1, tr["q"], tr["seed_idx"], tr["gold_idx"], X, Xt, index, device, off_epochs)
    heads = (g1, g_hard, g_mix, g2)

    membership = _onehop_membership(engine)                                  # overlap (the winner)
    mem_idx = [sorted(membership.get(idx2id[i], {int(hard[i])})) for i in range(n)]
    C, _ = _centroids(engine, node_vecs, membership, npart)
    Cg = F.normalize(torch.tensor(C, dtype=torch.float32, device=device), dim=-1); D = C.shape[1]

    gp = {s: _gp_gpl(splits, s, membership) for s in ("train", "val", "test")}
    tr_rows = [(nd, gpu, gd) for (nd, _, gd), gpu in zip(splits["train"], gp["train"][0])]
    va_rows = [(nd, gpu, gd) for (nd, _, gd), gpu in zip(splits["val"], gp["val"][0])]
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, npart - 1)
    f_tr = _concat(sig["train"], ["q", "seed", "nbr"]); f_va = _concat(sig["val"], ["q", "seed", "nbr"])
    route_model = _train_router(f_tr, tr_rows, f_va, va_rows, Cg, D, device, tau, hn_k, epochs)

    Fkw = dict(route_model=route_model, C=C, Cg=Cg, mem_idx=mem_idx, npart=npart, Xt=Xt,
               index=index, device=device, K=K, topn=topn)
    F_tr, _ = _features_for_split(sig["train"], heads, **Fkw)
    F_va, _ = _features_for_split(sig["val"], heads, **Fkw)
    F_te, node_orders_te = _features_for_split(sig["test"], heads, **Fkw)

    gpl_te = gp["test"][1]
    # convert each NODE order to a PARTITION order (via votes) BEFORE fusing partition rankings
    vote_orders = {nm: np.argsort(-_vote(order, mem_idx, npart, topn), axis=1)
                   for nm, order in node_orders_te.items()}
    fixed = {}
    for name, porder in vote_orders.items():
        fixed[name] = _cov_pre(porder, gpl_te, npart)[0]
    fixed["fuse_dense+relhard"] = _cov_pre(
        _rrf_fuse([_ranks(vote_orders["vote_dense"]), _ranks(vote_orders["vote_relhard"])], [1.0, 1.0], k=npart),
        gpl_te, npart)[0]
    fixed["fuse_all_votes"] = _cov_pre(
        _rrf_fuse([_ranks(vote_orders[nm]) for nm in ("vote_dense", "vote_relhard", "vote_rel2hop", "vote_mlpT")],
                  [1.0] * 4, k=npart), gpl_te, npart)[0]

    return {"dataset": dataset, "npart": npart, "n_test": len(gpl_te), "D": F_te.shape[-1],
            "F_tr": F_tr, "gp_tr": gp["train"][0], "F_va": F_va, "gpl_va": gp["val"][1],
            "F_te": F_te, "gpl_te": gpl_te, "fixed": fixed}


def _cache_path(dataset):
    d = os.path.join("data", "ukb_storage", dataset, "results", "L1_select"); os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"upr_feat_{dataset}.npz")


def _save_cache(blk):
    np.savez_compressed(
        _cache_path(blk["dataset"]), F_tr=blk["F_tr"], F_va=blk["F_va"], F_te=blk["F_te"],
        npart=blk["npart"], gp_tr=np.array(json.dumps(blk["gp_tr"])),
        gpl_va=np.array(json.dumps(blk["gpl_va"])), gpl_te=np.array(json.dumps(blk["gpl_te"])))


def _load_cache(dataset):
    z = np.load(_cache_path(dataset), allow_pickle=False)
    return {"dataset": dataset, "npart": int(z["npart"]), "F_tr": z["F_tr"], "F_va": z["F_va"], "F_te": z["F_te"],
            "gp_tr": json.loads(str(z["gp_tr"])), "gpl_va": json.loads(str(z["gpl_va"])),
            "gpl_te": json.loads(str(z["gpl_te"]))}


def _eval_block(model, blk, device):
    with torch.no_grad():
        lg = model(torch.tensor(blk["F_te"], device=device)).cpu().numpy()
    fc, gt = _cov_pre(np.argsort(-lg, axis=1), blk["gpl_te"], blk["npart"])
    return {"fullcov": fc, "gt_recall": gt}


def run_solo(dataset, epochs, off_epochs, limit, K, topn, device):
    blk = _compute_dataset(dataset, epochs, off_epochs, limit, K, topn, device)
    model = _train_combiner([blk], blk["D"], device, epochs=60)
    learned = _eval_block(model, blk, device)
    _save_cache(blk)
    out = {"dataset": dataset, "npart": blk["npart"], "n_test": blk["n_test"], "budgets": KS, "features": FEATURES,
           "learned_perds": learned, "fixed": {k: {"fullcov": v} for k, v in blk["fixed"].items()}}
    d = os.path.join("data", "ukb_storage", dataset, "results", "L1_select"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"unified_ranker_{dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] learned_perds FullCov {learned['fullcov']} | fixed fuse_all {blk['fixed']['fuse_all_votes']}")
    return out


def run_universal(datasets, device, epochs=60):
    blks = [_load_cache(d) for d in datasets]
    D = blks[0]["F_tr"].shape[-1]
    model = _train_combiner(blks, D, device, epochs=epochs)
    summary = {}
    for b in blks:
        res = _eval_block(model, b, device)
        summary[b["dataset"]] = res["fullcov"]
        log.info(f"[UNIV eval {b['dataset']}] FullCov {res['fullcov']}")
        outp = os.path.join("data", "ukb_storage", b["dataset"], "results", "L1_select", f"unified_ranker_{b['dataset']}.json")
        if os.path.exists(outp):
            j = json.load(open(outp)); j["learned_univ"] = res
            json.dump(j, open(outp, "w"), indent=2)
    idx_dir = os.path.join("data", "ukb_storage", "_index"); os.makedirs(idx_dir, exist_ok=True)
    with open(os.path.join(idx_dir, "unified_universal.json"), "w", encoding="utf-8") as f:
        json.dump({"datasets": datasets, "learned_univ_fullcov": summary}, f, indent=2)
    return summary


def _rr_from_F(F, k0=K0):
    """Per-signal reciprocal-rank matrix (nq,npart): 1/(k0 + rank_of_partition_in_that_signal).
    Weighted RRF then reduces to a vectorized weighted sum over signals (fast; no Python loops)."""
    nq, npart, _ = F.shape; rows = np.arange(nq)[:, None]; ranks = np.arange(npart)[None, :]
    out = {}
    for i, s in enumerate(FEATURES):
        order = np.argsort(-F[:, :, i], axis=1)
        rank = np.empty((nq, npart), np.int32); rank[rows, order] = ranks
        out[s] = (1.0 / (k0 + rank)).astype(np.float32)
    return out


def _wrrf_order(rr, signals, w):
    score = None
    for s in signals:
        if w[s] <= 0:
            continue
        score = w[s] * rr[s] if score is None else score + w[s] * rr[s]
    if score is None:
        score = -np.zeros_like(next(iter(rr.values())))
    return np.argsort(-score, axis=1)


def _tune_universal_weights(va_blocks, signals, budget=20, grid=(0.0, 0.5, 1.0, 2.0), passes=2):
    """Coordinate-ascent ONE weight vector maximizing MEAN val FullCov@budget across datasets.
    va_blocks: [{rr, gpl, npart}] with rr precomputed (memory-light — F_va already freed)."""
    w = {s: 1.0 for s in signals}

    def mean_val(w):
        return float(np.mean([_cov_pre(_wrrf_order(v["rr"], signals, w), v["gpl"], v["npart"])[0][budget]
                              for v in va_blocks]))
    best = mean_val(w)
    for _ in range(passes):
        for s in signals:
            bw, bs = w[s], best
            for cand in grid:
                w[s] = cand; sc = mean_val(w)
                if sc > bs:
                    bs, bw = sc, cand
            w[s] = bw; best = bs
    return w, round(best, 2)


def run_analyze(datasets, device, epochs=60, with_mlp=False, val_cap=2000):
    """Memory-disciplined: load one dataset's arrays at a time, free before the next (avoids OOM
    from holding all 5 datasets' (nq x npart x D) feature tensors at once)."""
    votes = ["vote_dense", "vote_relhard", "vote_rel2hop", "vote_mlpT"]
    allsig = FEATURES
    # ---- tuning: per-dataset val reciprocal-rank blocks (subsampled), F_va freed immediately
    va_blocks = []
    for d in datasets:
        z = np.load(_cache_path(d), allow_pickle=False)
        Fva = z["F_va"]; gpl_va = json.loads(str(z["gpl_va"])); npart = int(z["npart"])
        if len(Fva) > val_cap:
            Fva = Fva[:val_cap]; gpl_va = gpl_va[:val_cap]
        va_blocks.append({"rr": _rr_from_F(Fva), "gpl": gpl_va, "npart": npart})
        del Fva, z
    w_votes, vv = _tune_universal_weights(va_blocks, votes)
    w_all, va = _tune_universal_weights(va_blocks, allsig)
    del va_blocks
    log.info(f"[UNIV weighted-RRF] votes w={w_votes} (val {vv}) | all w={w_all} (val {va})")

    mlp = None
    if with_mlp:
        blks = [_load_cache(d) for d in datasets]
        mlp = _train_combiner(blks, blks[0]["F_tr"].shape[-1], device, epochs=epochs); del blks

    # ---- eval on test, one dataset at a time (free arrays before next)
    table = {}
    for d in datasets:
        z = np.load(_cache_path(d), allow_pickle=False)
        Fte = z["F_te"]; gpl = json.loads(str(z["gpl_te"])); npart = int(z["npart"])
        rr = _rr_from_F(Fte)
        row = {}
        for s in allsig:
            row[s] = _cov_pre(np.argsort(-Fte[:, :, FEATURES.index(s)], axis=1), gpl, npart)[0]
        row["equal_rrf_votes"] = _cov_pre(_wrrf_order(rr, votes, {s: 1.0 for s in votes}), gpl, npart)[0]
        row["wrrf_votes"] = _cov_pre(_wrrf_order(rr, votes, w_votes), gpl, npart)[0]
        row["wrrf_all"] = _cov_pre(_wrrf_order(rr, allsig, w_all), gpl, npart)[0]
        if mlp is not None:
            with torch.no_grad():
                lg = mlp(torch.tensor(Fte, device=device)).cpu().numpy()
            row["learned_mlp_univ"] = _cov_pre(np.argsort(-lg, axis=1), gpl, npart)[0]
        table[d] = row
        log.info(f"[{d}] equal_rrf@20 {row['equal_rrf_votes'][20]} | wrrf_votes@20 {row['wrrf_votes'][20]} "
                 f"| wrrf_all@20 {row['wrrf_all'][20]}")
        del Fte, rr, z
    methods = allsig + ["equal_rrf_votes", "wrrf_votes", "wrrf_all"] + (["learned_mlp_univ"] if mlp is not None else [])
    means = {m: {k: round(float(np.mean([table[d][m][k] for d in datasets])), 2) for k in KS} for m in methods}
    out = {"datasets": datasets, "weights_votes": w_votes, "weights_all": w_all,
           "per_dataset": table, "mean_over_datasets": means}
    idx_dir = os.path.join("data", "ukb_storage", "_index"); os.makedirs(idx_dir, exist_ok=True)
    with open(os.path.join(idx_dir, "unified_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info("=== MEAN over datasets (partition FullCov) ===")
    for m in methods:
        log.info(f"  {m:20} @20={means[m][20]:6} @50={means[m][50]:6} @100={means[m][100]:6}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L1 unified dataset-agnostic partition ranker (everything fused).")
    p.add_argument("--datasets", nargs="+", default=["2wiki_clean", "musique_clean", "metaqa"])
    p.add_argument("--mode", choices=["solo", "universal", "analyze"], default="solo")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--off-epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=8000)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--topn", type=int, default=200)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.mode == "universal":
        log.info(f"===== UPR UNIVERSAL (pool {a.datasets}) =====")
        run_universal(a.datasets, device, epochs=a.epochs)
        return
    if a.mode == "analyze":
        log.info(f"===== UPR ANALYZE (weighted-RRF + universal MLP, {a.datasets}) =====")
        run_analyze(a.datasets, device, epochs=a.epochs)
        return
    for ds in a.datasets:
        log.info(f"===== L1 UNIFIED RANKER (solo): {ds.upper()} =====")
        try:
            run_solo(ds, a.epochs, a.off_epochs, a.limit, a.K, a.topn, device)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")


if __name__ == "__main__":
    main()
