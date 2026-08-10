"""
L1 fusion experiments — cached orders + OVERLAP-gated fusion.
============================================================
The per-query weight-rule needs a signal that says "is rel redundant here?".
dense_top1_sim FAILED (hotpot 0.658 ~ 2wiki 0.641, but hotpot wants rel dropped and
2wiki wants it kept). The right signal is the INFERENCE-TIME OVERLAP between dense's
and rel's retrieved sets — |dense_topk ∩ rel_topk| / k — which measures redundancy vs
complementarity directly, with NO golds. High overlap (hotpot/squad) -> rel redundant
-> down-weight; low overlap (metaqa/musique) -> complementary -> keep. This is the
"overlap is the novelty" idea used as the fusion signal.

Engineering: trained orders (dense, rel_hard, rel_2hop + dense top-1 sim + gold) are
CACHED to scratch per dataset, so metaqa's slow heads train ONCE and every fusion
variant below is instant. Variants: equal_rrf | conf_gate (dense_top1_sim, the failed
baseline) | OVERLAP_gate. Reports recall@{50,100,200,500}, mean dense-rel overlap
(diagnostic: does it separate hotpot-high from metaqa-low?), and the hotpot delta.
Writes _index/l1_fusion_summary.json.
"""
import os
import gc
import json
import logging
import argparse

import numpy as np
import torch
import faiss

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership
from src.experiments.l1_ablate import _train_offset, _order, _ranks, _rrf_fuse, _recall, KS, MAXK
from src.experiments.l1_dynamic import _train_hop2

log = logging.getLogger("experiments.l1_fusion")
CACHE_DIR = "C:\\Users\\Swastik\\AppData\\Local\\Temp\\claude\\C--Users-Swastik-Desktop-CRAG\\4046bf12-ec21-473b-a77e-55b83a7e3989\\scratchpad\\l1_fusion_cache"


def _pad(lol, width=MAXK):
    a = np.full((len(lol), width), -1, np.int32)
    for i, r in enumerate(lol):
        r = list(r)[:width]
        a[i, :len(r)] = r
    return a


def build_or_load(ds, limit, epochs, device):
    os.makedirs(CACHE_DIR, exist_ok=True)
    npz = os.path.join(CACHE_DIR, f"{ds}_L{limit}.npz")
    gj = os.path.join(CACHE_DIR, f"{ds}_L{limit}_gold.json")
    if os.path.exists(npz) and os.path.exists(gj):
        z = np.load(npz); g = json.load(open(gj, encoding="utf-8"))
        log.info(f"[{ds}] loaded cached orders")
        return {k: z[k] for k in z.files} | {"gold_tr": g["tr"], "gold_te": g["te"]}

    engine = CoreEngine(source=ds); encoder = DenseEncoder()
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    d = X.shape[1]; index = faiss.IndexFlatIP(d); index.add(X); Xt = torch.tensor(X, device=device)
    id2idx = engine.node_id_to_idx
    splits = _splits(engine, _hard_membership(engine))
    if limit:
        splits = {s: qs[:limit] for s, qs in splits.items()}

    def prep(qs):
        q = encoder.encode([n.content for n, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
        _, seed = index.search(q, 1)
        gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]
        return q, seed[:, 0], gold
    q_tr, seed_tr, gold_tr = prep(splits["train"])
    q_te, seed_te, gold_te = prep(splits["test"])

    log.info(f"[{ds}] training g1,g_hard,g2 (n_tr={len(q_tr)}) [will cache]...")
    g1 = _train_offset("base", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g_hard = _train_offset("hard", q_tr, seed_tr, gold_tr, Xt, index, device, epochs)
    g2 = _train_hop2(g1, q_tr, seed_tr, gold_tr, X, Xt, index, device, epochs)

    def orders(q, seed):
        qt = torch.tensor(q, device=device); sv = Xt[[int(s) for s in seed]]
        with torch.no_grad():
            hp = g_hard(qt, sv).cpu().numpy(); h1 = g1(qt, sv).cpu().numpy()
        D, dord = index.search(np.ascontiguousarray(q.astype("float32")), MAXK)
        hard = _order(hp, index)
        hop1 = _order(h1, index); s1 = hop1[:, 0]
        with torch.no_grad():
            hop2 = _order(g2(qt, Xt[[int(s) for s in s1]]).cpu().numpy(), index)
        two = _pad(_rrf_fuse([_ranks(hop1), _ranks(hop2)], [1.0, 1.0]))
        return dord.astype(np.int32), D[:, 0].astype(np.float32), hard.astype(np.int32), two
    d_tr, conf_tr, h_tr, t_tr = orders(q_tr, seed_tr)
    d_te, conf_te, h_te, t_te = orders(q_te, seed_te)
    np.savez_compressed(npz, d_tr=d_tr, conf_tr=conf_tr, h_tr=h_tr, t_tr=t_tr,
                        d_te=d_te, conf_te=conf_te, h_te=h_te, t_te=t_te)
    json.dump({"tr": gold_tr, "te": gold_te}, open(gj, "w", encoding="utf-8"))
    log.info(f"[{ds}] cached orders -> {npz}")
    return {"d_tr": d_tr, "conf_tr": conf_tr, "h_tr": h_tr, "t_tr": t_tr,
            "d_te": d_te, "conf_te": conf_te, "h_te": h_te, "t_te": t_te,
            "gold_tr": gold_tr, "gold_te": gold_te}


def _overlap(dense, rel, k=20):
    return np.array([len(set(dense[i, :k].tolist()) & set(rel[i, :k].tolist())) / k for i in range(len(dense))])


def _gate(signal, a, b, lo=0.1, hi=1.2):
    return np.clip(a * (b - signal), lo, hi)                 # decreasing in signal


def _fit(signal_tr, dmap, hmap, tmap, gold_tr, nsub=2000):
    n = len(signal_tr)                                       # subsample train: the (a,b) grid
    if n > nsub:                                             # doesn't need all queries, and the
        idx = np.linspace(0, n - 1, nsub).astype(int)       # pure-Python RRF over 20k x 40 combos
        signal_tr = signal_tr[idx]                           # was the real bottleneck (minutes/ds)
        dmap = [dmap[i] for i in idx]; hmap = [hmap[i] for i in idx]
        tmap = [tmap[i] for i in idx]; gold_tr = [gold_tr[i] for i in idx]
    best, ab = -1, (3.0, 0.5)
    for a in (1.0, 2.0, 3.0, 5.0):
        for b in (0.2, 0.4, 0.6, 0.8, 0.9):
            w = _gate(signal_tr, a, b)
            r = _recall(_rrf_fuse([dmap, hmap, tmap], [np.ones_like(w), w, w]), gold_tr, budgets=[100])[100]
            if r > best:
                best, ab = r, (a, b)
    return ab


def run(ds, cache):
    d_tr, h_tr, t_tr, gold_tr = cache["d_tr"], cache["h_tr"], cache["t_tr"], cache["gold_tr"]
    d_te, h_te, t_te, gold_te = cache["d_te"], cache["h_te"], cache["t_te"], cache["gold_te"]
    dmap, hmap, tmap = _ranks(d_te), _ranks(h_te), _ranks(t_te)
    dmap_tr, hmap_tr, tmap_tr = _ranks(d_tr), _ranks(h_tr), _ranks(t_tr)

    conf_tr, conf_te = cache["conf_tr"], cache["conf_te"]
    ov_tr, ov_te = _overlap(d_tr, h_tr), _overlap(d_te, h_te)
    ac, bc = _fit(conf_tr, dmap_tr, hmap_tr, tmap_tr, gold_tr)
    ao, bo = _fit(ov_tr, dmap_tr, hmap_tr, tmap_tr, gold_tr)
    wc, wo = _gate(conf_te, ac, bc), _gate(ov_te, ao, bo)

    cfg = {
        "dense": _recall(d_te, gold_te),
        "equal_rrf": _recall(_rrf_fuse([dmap, hmap, tmap], [1.0, 1.0, 1.0]), gold_te),
        "conf_gate": _recall(_rrf_fuse([dmap, hmap, tmap], [np.ones_like(wc), wc, wc]), gold_te),
        "overlap_gate": _recall(_rrf_fuse([dmap, hmap, tmap], [np.ones_like(wo), wo, wo]), gold_te),
    }
    out = {"dataset": ds, "n_test": len([g for g in gold_te if g]), "budgets": KS, "configs": cfg,
           "mean_overlap": round(float(ov_te.mean()), 3), "dense_conf_mean": round(float(conf_te.mean()), 3),
           "overlap_gate_ab": [ao, bo], "overlap_relw_mean": round(float(wo.mean()), 3),
           "ovgate_vs_equal@100": round(cfg["overlap_gate"][100] - cfg["equal_rrf"][100], 2),
           "ovgate_vs_dense@100": round(cfg["overlap_gate"][100] - cfg["dense"][100], 2)}
    with open(os.path.join("data", "ukb_storage", ds, "results", "L1", "fusion.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{ds}] overlap~{out['mean_overlap']} relw~{out['overlap_relw_mean']} || dense {cfg['dense'][100]} | "
             f"equal {cfg['equal_rrf'][100]} | conf_gate {cfg['conf_gate'][100]} | OVERLAP_gate {cfg['overlap_gate'][100]} "
             f"(vs equal {out['ovgate_vs_equal@100']}, vs dense {out['ovgate_vs_dense@100']})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Overlap-gated L1 fusion (cached orders).")
    p.add_argument("--datasets", nargs="+",
                   default=["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean", "metaqa"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=20000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cpu")
    results = {}
    for ds in a.datasets:
        log.info(f"===== L1 FUSION (overlap-gate): {ds.upper()} =====")
        try:
            cache = build_or_load(ds, a.limit, a.epochs, device)
            results[ds] = run(ds, cache)
            del cache
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")
        gc.collect()
    if results:
        def m(k):
            return round(float(np.mean([r["configs"][k][100] for r in results.values()])), 2)
        summary = {"datasets": list(results), "dense_mean@100": m("dense"), "equal_rrf_mean@100": m("equal_rrf"),
                   "conf_gate_mean@100": m("conf_gate"), "overlap_gate_mean@100": m("overlap_gate"),
                   "per_dataset": {ds: {"overlap": r["mean_overlap"], "dense": r["configs"]["dense"][100],
                                        "equal_rrf": r["configs"]["equal_rrf"][100],
                                        "overlap_gate": r["configs"]["overlap_gate"][100],
                                        "ovgate_vs_equal": r["ovgate_vs_equal@100"]} for ds, r in results.items()}}
        summary["ovgate_vs_equal_mean@100"] = round(summary["overlap_gate_mean@100"] - summary["equal_rrf_mean@100"], 2)
        with open(os.path.join("data", "ukb_storage", "_index", "l1_fusion_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"FUSION SUMMARY: dense {summary['dense_mean@100']} | equal {summary['equal_rrf_mean@100']} | "
                 f"conf_gate {summary['conf_gate_mean@100']} | OVERLAP_gate {summary['overlap_gate_mean@100']} "
                 f"(+{summary['ovgate_vs_equal_mean@100']} vs equal)")


if __name__ == "__main__":
    main()
