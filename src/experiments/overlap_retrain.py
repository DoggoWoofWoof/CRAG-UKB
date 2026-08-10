"""
1-hop-overlap partition RETRAIN (Level-1 structural ablation).
==============================================================
User's approach: expand each partition to also contain the 1-hop graph
neighbours of its member nodes, REBUILD the partition centroids over those
(larger, overlapping) partitions, and RETRAIN the MLP router on them. A gold doc
then belongs to its own partition + every partition whose members border it, so
routing has more chances to cover it — at the cost of a bigger candidate pool
("explosion"), which we measure.

Controlled comparison in one run: `hard` (original single-membership partitions
+ original centroids) vs `overlap1` (1-hop-expanded partitions + rebuilt
centroids), both KL-trained identically, evaluated final-vs-final. Coverage is
computed per gold DOC (covered if ANY of its partitions is in top-K). Writes
results/overlap_ablation/{dataset}_overlap_retrain.json and per-epoch logs.
Self-contained (reuses kl_div_loss + TextPartitionMLP); no train_mlp changes.
"""
import os
import json
import csv
import random
import logging
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.alignment.mlp_encoder import TextPartitionMLP
from src.alignment.train_mlp import kl_div_loss
from src.alignment.coverage_losses import partition_coverage_loss
from src.evaluation.benchmark_partition_selection import COVERAGE_K_VALUES

log = logging.getLogger("experiments.overlap_retrain")

SPLIT_SEED, TRAIN_RATIO, VAL_RATIO = 42, 0.70, 0.20
INIT_SEED = 42   # pin MLP weight init for reproducible paired comparisons (audit fix)
TAU = {"metaqa": 0.01, "2wiki": 0.07, "musique": 0.05, "squad": 0.1, "squad_clean": 0.1,
       "2wiki_clean": 0.07, "musique_clean": 0.05, "hotpotqa_clean": 0.07}
# hn_k ~= npart-1 (all-negatives HNM). Set for the FINAL 100-docs/partition substrate:
# 2wiki_clean=658, metaqa=401, squad=190, musique_clean=136, hotpotqa_clean~665 partitions.
HNK = {"metaqa": 400, "2wiki": 149, "musique": 33, "squad": 189, "squad_clean": 189,
       "2wiki_clean": 657, "musique_clean": 135, "hotpotqa_clean": 660}


def _compute_loss(loss_name, proj, pids, Cg, tau, hn_k):
    """Dispatch the training/val loss. 'kl' = KL(+HNM) baseline; 'coverage' = the
    ported Jigsaw CVaR + FullCov@20-barrier loss (target_topk aligned to the
    headline metric so it gets its best shot on the overlapped partitions)."""
    if loss_name == "kl":
        return kl_div_loss(proj, pids, Cg, temperature=tau, hn_k=hn_k)
    if loss_name == "coverage":
        return partition_coverage_loss(proj, pids, Cg, temperature=tau, target_topk=20)
    raise ValueError(f"unknown loss {loss_name!r}")


def _reconstruct(index):
    return np.array([index.reconstruct(i) for i in range(index.ntotal)], dtype=np.float32)


def _hard_membership(engine):
    return {nid: {int(p)} for nid, p in engine.partition_map.items()}


def _onehop_membership(engine):
    """node -> {own partition} ∪ {partitions of its 1-hop (original) neighbours}."""
    metis = {nid: int(p) for nid, p in engine.partition_map.items()}
    mem = {nid: {p} for nid, p in metis.items()}
    for node in engine.nodes:
        for nb in node.neighbors:               # original dataset edges (node.neighbors)
            if nb in metis:
                mem[node.node_id].add(metis[nb])
    return mem


def _synhop_membership(engine):
    """node -> {own} ∪ {partitions of its SYNTHETIC (index-time kNN-bridge) neighbours}.
    Uses node.metadata['synthetic_neighbors'] — the semantic doc->doc edges added at
    index time, distinct from the original dataset edges in node.neighbors. A
    graph-native (doc-level) analogue of the centroid-kNN overlap."""
    metis = {nid: int(p) for nid, p in engine.partition_map.items()}
    mem = {nid: {p} for nid, p in metis.items()}
    for node in engine.nodes:
        for nb in node.metadata.get("synthetic_neighbors", ()):
            if nb in metis:
                mem[node.node_id].add(metis[nb])
    return mem


def _twohop_membership(engine):
    """node -> {own} ∪ {partitions of 1-hop} ∪ {partitions of 2-hop neighbours}."""
    metis = {nid: int(p) for nid, p in engine.partition_map.items()}
    nbrs = {node.node_id: list(node.neighbors) for node in engine.nodes}
    mem = {nid: {p} for nid, p in metis.items()}
    for node in engine.nodes:
        acc = mem[node.node_id]
        for nb in nbrs.get(node.node_id, ()):    # 1-hop
            if nb in metis:
                acc.add(metis[nb])
            for nb2 in nbrs.get(nb, ()):          # 2-hop
                if nb2 in metis:
                    acc.add(metis[nb2])
    return mem


def _knn_membership(engine, node_vecs, m):
    """node -> {own METIS partition} ∪ {partitions of its m nearest centroids}.
    A *semantic* overlap axis (embedding-kNN to centroids), orthogonal to graph
    hops — and unlike graph hops it's a continuous knob in m. Bootstraps the kNN
    assignment from the ORIGINAL frozen centroids; centroids are then rebuilt over
    the resulting membership by _centroids()."""
    import faiss
    cents = _reconstruct(engine.centroid_index)
    cpids = [int(p) for p in engine.centroid_pids]
    nv = node_vecs.copy(); faiss.normalize_L2(nv)
    cv = cents.copy(); faiss.normalize_L2(cv)
    metis = {nid: int(p) for nid, p in engine.partition_map.items()}
    mem = {nid: {p} for nid, p in metis.items()}
    if m > 0:
        sims = nv @ cv.T
        topm = np.argsort(-sims, axis=1)[:, :m]
        for i, node in enumerate(engine.nodes):
            if node.node_id in mem:
                mem[node.node_id].update(int(cpids[j]) for j in topm[i])
    return mem


def _extra_edge_membership(engine, edge_file):
    """node -> {own METIS partition} ∪ {partitions of its neighbours in an EXTERNAL
    edge file} (doc_id -> [neighbour doc_ids]). Used for NER (structural) / SPLADE
    (synthetic) edge atoms so they enter the pool-matched structure sweep exactly
    like overlap1/knn — a prerequisite for a valid champion determination."""
    import os
    metis = {nid: int(p) for nid, p in engine.partition_map.items()}
    mem = {nid: {p} for nid, p in metis.items()}
    if os.path.exists(edge_file):
        edges = json.load(open(edge_file, encoding="utf-8"))
        for src, nbrs in edges.items():
            if src in metis:
                for nb in nbrs:
                    if nb in metis:
                        mem[src].add(metis[nb])
    else:
        log.warning(f"extra-edge atom: {edge_file} missing — atom reduces to hard membership")
    return mem


def _centroids(engine, node_vecs, membership, npart):
    """Degree-weighted centroids over (possibly overlapping) membership + sizes."""
    import faiss
    acc = defaultdict(list)
    for i, node in enumerate(engine.nodes):
        w = float(len(node.neighbors)) + 1.0
        for p in membership.get(node.node_id, ()):
            acc[p].append((i, w))
    dim = node_vecs.shape[1]
    C = np.zeros((npart, dim), dtype=np.float32)
    sizes = np.zeros(npart, dtype=np.float64)
    for p, items in acc.items():
        idxs = [i for i, _ in items]
        ws = np.array([w for _, w in items], dtype=np.float64)
        C[p] = np.average(node_vecs[idxs], axis=0, weights=ws)
        sizes[p] = len(idxs)
    faiss.normalize_L2(C)
    return C, sizes


def _splits(engine, membership):
    """Return locked query splits and respect official metadata when available.

    Some clean development corpora contain only one official source split. Those
    still use the deterministic 70/20/10 fallback, but datasets such as MetaQA
    must not have their official train/dev/test boundaries shuffled together.
    """
    pairs = []
    for node in engine.all_nodes:
        if node.metadata.get("type") == "question":
            gp = set()
            golds = [nb for nb in node.neighbors if nb in membership]
            for nb in golds:
                gp |= membership[nb]
            if gp:
                split = str(node.metadata.get("split", "")).lower()
                pairs.append((node.node_id, node, sorted(gp), golds, split))
    pairs.sort(key=lambda x: x[0])

    official = {"train": [], "val": [], "test": []}
    aliases = {"train": "train", "dev": "val", "validation": "val", "val": "val", "test": "test"}
    for item in pairs:
        target = aliases.get(item[4])
        if target:
            official[target].append(item)
    if all(official.values()) and sum(map(len, official.values())) == len(pairs):
        return {
            split: [(nd, pd, gd) for _, nd, pd, gd, _ in items]
            for split, items in official.items()
        }

    random.Random(SPLIT_SEED).shuffle(pairs)
    n = len(pairs)
    tr, va = int(n * TRAIN_RATIO), int(n * TRAIN_RATIO) + int(n * VAL_RATIO)
    f = lambda s: [(nd, pd, gd) for _, nd, pd, gd, _ in s]
    return {"train": f(pairs[:tr]), "val": f(pairs[tr:va]), "test": f(pairs[va:])}


def _coverage(top_pids, gold_docs, membership, k):
    topk = set(top_pids[:k])
    cov = [g for g in gold_docs if membership.get(g, set()) & topk]
    full = 1.0 if gold_docs and len(cov) == len(gold_docs) else 0.0
    gtr = len(cov) / len(gold_docs) if gold_docs else 0.0
    return full, gtr


def _train(engine, C, splits, split_embs, device, tau, hn_k, epochs, logs_dir, name, loss_name="kl"):
    Cg = F.normalize(torch.tensor(C, dtype=torch.float32, device=device), dim=-1)
    D = C.shape[1]
    torch.manual_seed(INIT_SEED)      # reproducible weight init (audit fix for ~0.4pt run noise)
    model = TextPartitionMLP(input_dim=D, hidden_dim=512, output_dim=D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=7)

    tr, va = splits["train"], splits["val"]
    tr_e, va_e = split_embs["train"], split_embs["val"]
    bs = 64
    os.makedirs(logs_dir, exist_ok=True)
    hist_path = os.path.join(logs_dir, "history.csv")
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_full_coverage@20", "is_best"])

    def _val():
        model.eval()
        tot, nb = 0.0, 0
        fc = []
        with torch.no_grad():
            for i in range(0, len(va), bs):
                embs = torch.tensor(va_e[i:i + bs], dtype=torch.float32, device=device)
                proj = F.normalize(model(embs), dim=-1)
                loss = _compute_loss(loss_name, proj, [p for _, p, _ in va[i:i + bs]], Cg, tau, hn_k)
                tot += float(loss); nb += 1
                sims = proj @ Cg.T
                topk = torch.topk(sims, min(20, sims.shape[1]), dim=1).indices.cpu().tolist()
                for j, (_, _, golds) in enumerate(va[i:i + bs]):
                    if golds:
                        fc.append(_coverage(topk[j], golds, membership_ref[name], 20)[0])
        return tot / max(nb, 1), (float(np.mean(fc)) if fc else 0.0)

    best_val, best_state, final_state, best_ep = float("inf"), None, None, 0
    no_imp = 0
    for ep in range(epochs):
        model.train()
        order = list(range(len(tr))); random.Random(ep).shuffle(order)
        ttot, tnb = 0.0, 0
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            embs = torch.tensor(tr_e[idx], dtype=torch.float32, device=device)
            pids = [tr[i][1] for i in idx]
            proj = F.normalize(model(embs), dim=-1)
            loss = _compute_loss(loss_name, proj, pids, Cg, tau, hn_k)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            ttot += float(loss); tnb += 1
        vl, vfc = _val(); sched.step(vl)
        improved = vl < best_val
        if improved:
            best_val, best_ep = vl, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        with open(hist_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep + 1, round(ttot / max(tnb, 1), 6), round(vl, 6), round(vfc * 100, 2), improved])
        if no_imp >= 20:
            break
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model, best_state, final_state, Cg


membership_ref = {}   # name -> membership (used by _val for the coverage curve)


def _eval(model, state, Cg, test, test_e, membership, device):
    """Full per-doc metric suite from the complete partition ranking. A gold DOC is
    'covered@k' if ANY of its (possibly several) partitions ranks <= k — the
    overlap-correct semantics that reduces to partition-in-top-k for hard configs.
    Emits full_coverage/gt_recall/recall @K + mrr + weakest_positive_rank, and a
    per-query FullCov@50 0/1 vector ('_fc50_vec') for paired McNemar."""
    model.load_state_dict(state); model.eval()
    npart = Cg.shape[0]
    Ks = COVERAGE_K_VALUES
    sentinel = npart + 1
    fc = {k: [] for k in Ks}; gtr = {k: [] for k in Ks}; rec = {k: [] for k in Ks}
    weakest, mrr, fc50 = [], [], []
    with torch.no_grad():
        for i in range(0, len(test), 256):
            embs = torch.tensor(test_e[i:i + 256], dtype=torch.float32, device=device)
            proj = F.normalize(model(embs), dim=-1)
            order = torch.argsort(-(proj @ Cg.T), dim=1).cpu().tolist()   # full ranking
            for j, (_, _, golds) in enumerate(test[i:i + 256]):
                golds = [g for g in golds if g in membership]
                if not golds:
                    continue
                rank_of = {p: r + 1 for r, p in enumerate(order[j])}       # 1-indexed
                best = [min((rank_of.get(p, sentinel) for p in membership.get(d, ())), default=sentinel)
                        for d in golds]
                fh, wk = min(best), max(best)
                weakest.append(wk if wk <= npart else sentinel)
                mrr.append(1.0 / fh if fh <= npart else 0.0)
                for k in Ks:
                    cov = sum(1 for b in best if b <= k)
                    fc[k].append(1.0 if cov == len(best) else 0.0)
                    gtr[k].append(cov / len(best))
                    rec[k].append(1.0 if cov > 0 else 0.0)
                fc50.append(1 if all(b <= 50 for b in best) else 0)
    out = {}
    for k in Ks:
        out[f"full_coverage@{k}"] = round(float(np.mean(fc[k])) * 100, 2) if fc[k] else 0.0
        out[f"gt_recall@{k}"] = round(float(np.mean(gtr[k])) * 100, 2) if gtr[k] else 0.0
        out[f"recall@{k}"] = round(float(np.mean(rec[k])) * 100, 2) if rec[k] else 0.0
    out["mrr"] = round(float(np.mean(mrr)) * 100, 2) if mrr else 0.0
    out["weakest_positive_rank"] = round(float(np.mean(weakest)), 2) if weakest else 0.0
    out["n_test"] = len(fc50)
    out["_fc50_vec"] = fc50
    return out


def run_dataset(dataset, epochs=100, limit=0, device=None, configs=("hard", "overlap1"),
                losses=("kl",), out_suffix="", hn_k_override=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tau, hn_k = TAU.get(dataset, 0.07), HNK.get(dataset, 0)
    if hn_k_override is not None:
        hn_k = hn_k_override
    log.info(f"===== OVERLAP-RETRAIN: {dataset.upper()} (tau={tau:g}, hn_k={hn_k}) "
             f"configs={list(configs)} losses={list(losses)} =====")
    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    node_vecs = _reconstruct(engine.node_index)
    npart = max(int(p) for p in engine.partition_map.values()) + 1

    def _atom(name):                                      # build one membership atom by name
        if name == "hard":
            return _hard_membership(engine)
        if name == "overlap1":
            return _onehop_membership(engine)
        if name == "overlap2":
            return _twohop_membership(engine)
        if name == "syn1":
            return _synhop_membership(engine)
        if name.startswith("knn"):
            return _knn_membership(engine, node_vecs, int(name[3:]))
        if name == "ner":
            return _extra_edge_membership(engine, f"results/research/ner_edges_{dataset}.json")
        if name == "splade":
            return _extra_edge_membership(engine, f"results/research/splade_edges_{dataset}.json")
        raise ValueError(f"unknown membership atom {name!r}")

    def _build(cfg):                                      # cfg may be a '+'-union, e.g. overlap1+knn1
        atoms = [_atom(a) for a in cfg.split("+")]
        if len(atoms) == 1:
            return atoms[0]
        keys = set().union(*[set(a) for a in atoms])
        return {k: set().union(*[a.get(k, set()) for a in atoms]) for k in keys}

    all_membership = {cfg: _build(cfg) for cfg in configs}
    hard_sizes = np.array([len(engine.get_partition_nodes(p)) for p in range(npart)], dtype=np.float64)
    results = {}
    vecs = {}                                             # key -> per-query FullCov@20 vector (for McNemar)
    for cfg in configs:
        membership = all_membership[cfg]
        membership_ref[cfg] = membership
        C, sizes = _centroids(engine, node_vecs, membership, npart)
        splits = _splits(engine, membership)
        if limit:
            splits = {s: q[:limit] for s, q in splits.items()}
        if not splits["train"]:
            continue
        split_embs = {s: encoder.encode([q.content for q, _, _ in splits[s]]).astype("float32")
                      for s in splits if splits[s]}
        mem_per_doc = float(np.mean([len(membership.get(n.node_id, set())) for n in engine.nodes]))
        explosion = float(np.sum(sizes) / max(np.sum(hard_sizes), 1))

        for loss_name in losses:
            key = f"{cfg}__{loss_name}"                      # e.g. overlap1__coverage
            logs_dir = os.path.join("logs", dataset, f"overlap_retrain_{key}")
            model, best_state, final_state, Cg = _train(
                engine, C, splits, split_embs, device, tau, hn_k, epochs, logs_dir, cfg, loss_name)
            m_best = _eval(model, best_state, Cg, splits["test"], split_embs["test"], membership, device)
            m_final = _eval(model, final_state, Cg, splits["test"], split_embs["test"], membership, device)
            vecs[key] = m_final.pop("_fc50_vec"); m_best.pop("_fc50_vec", None)  # keep vectors out of JSON
            results[key] = {
                "config": cfg, "loss": loss_name, "hn_k": hn_k,
                "mean_memberships_per_doc": round(mem_per_doc, 3),
                "membership_explosion_x": round(explosion, 3),
                "best": m_best, "final": m_final,
            }
            log.info(f"  [{key}] mem/doc={mem_per_doc:.2f} explosion={explosion:.2f}x | "
                     f"final FCov@20={m_final['full_coverage@20']}% FCov@50={m_final['full_coverage@50']}% "
                     f"gtR@20={m_final['gt_recall@20']}% | best FCov@20={m_best['full_coverage@20']}%")

    # Paired McNemar at FullCov@50: every non-hard config vs the hard__<loss> baseline.
    from src.experiments.stats import paired
    mcnemar = {}
    for key in vecs:
        cfg_k, loss_k = results[key]["config"], results[key]["loss"]
        base = f"hard__{loss_k}"
        if cfg_k != "hard" and base in vecs:
            mcnemar[f"{key}_vs_{base}"] = paired(vecs[key], vecs[base])
    if len(losses) > 1:                                   # loss comparisons on the same config (e.g. coverage vs kl)
        for cfg in configs:
            if f"{cfg}__coverage" in vecs and f"{cfg}__kl" in vecs:
                mcnemar[f"{cfg}__coverage_vs_{cfg}__kl"] = paired(vecs[f"{cfg}__coverage"], vecs[f"{cfg}__kl"])

    out_dir = os.path.join("results", "overlap_ablation")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{dataset}_overlap_retrain{('_' + out_suffix) if out_suffix else ''}.json"
    with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "tau": tau, "hn_k": hn_k, "npart": npart,
                   "mcnemar_operating_k": 50,
                   "configs": list(configs), "losses": list(losses),
                   "results": results, "mcnemar": mcnemar}, f, indent=2)
    log.info(f"Saved results/overlap_ablation/{fname}  (+{len(mcnemar)} McNemar pairs)")
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description="1-hop-overlap partition retrain ablation.")
    p.add_argument("--datasets", nargs="+", default=["2wiki", "musique", "squad", "metaqa"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--configs", nargs="+", default=["hard", "overlap1"],
                   help="Membership configs: hard | overlap1 (graph 1-hop) | overlap2 (graph 2-hop) | "
                        "knn<m> (centroid-kNN overlap, e.g. knn1 knn2 knn3).")
    p.add_argument("--losses", nargs="+", default=["kl"], choices=["kl", "coverage"],
                   help="Loss(es) per config. Use 'kl coverage' for the overlap+KL vs overlap+Jigsaw comparison.")
    p.add_argument("--out_suffix", default="", help="Suffix for the output json (avoids overwriting prior runs).")
    p.add_argument("--hn_k", type=int, default=None, help="Override hard-negative count (for the HNM ablation).")
    a = p.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in a.datasets:
        run_dataset(ds, epochs=a.epochs, limit=a.limit, device=dev,
                    configs=tuple(a.configs), losses=tuple(a.losses), out_suffix=a.out_suffix,
                    hn_k_override=a.hn_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
