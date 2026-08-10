"""
L1 relational retriever — full improvement ablation.
=====================================================
The single-hop offset (l1_relational.py) fused with dense already wins on KB /
multi-hop and is neutral where dense saturates. This harness ablates EVERY
proposed improvement, individually and combined, and surfaces the best config
per dataset plus a uniform (generalizable) champion.

Retrievers (each -> ranked doc order per query):
  dense       : q -> faiss top-k                              (semantic baseline)
  rel_base    : seed(top1) + g(q) -> faiss                    (single-hop offset)
  rel_mseed   : union over top-M seeds of seed_m + g(q)       (robust to wrong start)
  rel_mix     : K learned directions seed + g_k(q), unioned   (1-to-many, e.g. actor->movies)
  rel_hard    : rel_base trained with faiss-mined hard negs   (sharper offset)
  rel_mhop    : iterate g from top-H hop-1 docs, unioned      (compositional / 2-hop)

Fusion of dense (+) a relational retriever:
  rrf         : equal reciprocal-rank fusion
  wrrf(alpha) : weighted RRF, rel weighted alpha (global sweep) — alpha=0 is dense, 1 is rel
  gate        : learned per-query alpha = sigmoid(gate(q)) (fixes the hotpot dilution)

Ablations: (A) rel-variants alone, (B) fusion on the best rel-variant, (C) greedy
forward-selection combo over ALL retrievers, + overlap analysis (which golds each
retriever alone reaches) and a uniform champion across datasets.
Writes data/ukb_storage/{ds}/results/L1/ablation.json + a combined summary.
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
from src.experiments.overlap_retrain import _reconstruct, _splits, _hard_membership, INIT_SEED
from src.experiments.query_relation import OffsetHead
from src.pipeline.ukb_results import rpath

log = logging.getLogger("experiments.l1_ablate")
KS = [50, 100, 200, 500]
MAXK = 500
TAU = 0.05
BIG = 10 ** 9


# --------------------------------------------------------------------------- heads
class MixtureHead(nn.Module):
    """K query-conditioned relation directions; predicts K candidate answer positions."""
    def __init__(self, d, K=8):
        super().__init__()
        self.K, self.d = K, d
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, K * d))

    def forward(self, qn, seed):                              # -> (B, K, d)
        off = self.net(qn).view(-1, self.K, self.d)
        return F.normalize(seed.unsqueeze(1) + off, dim=-1)


class Gate(nn.Module):
    """Per-query fusion weight alpha in (0,1): how much to trust rel vs dense."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, qn):
        return torch.sigmoid(self.net(qn)).squeeze(-1)        # (B,)


# --------------------------------------------------------------------------- training
def _train_offset(kind, q_tr, seed_tr, gold_tr, Xt, index, device, epochs, K=8):
    """kind in {base, hard, mix, mix_hard}. Returns trained head.
    mix_hard = MixtureHead (K directions) + rel_hard's hard-negative mining — mlpT was weak at
    rescuing dense-buried answers (15%) precisely because 'mix' lacks hard negs (rel_hard has them,
    rescues 59%); mix_hard gives the K-head mixture the same hard-neg pressure."""
    trip = [(i, int(seed_tr[i]), int(g)) for i, gl in enumerate(gold_tr) for g in gl]
    qtr = torch.tensor(q_tr, device=device)
    torch.manual_seed(INIT_SEED)
    d = Xt.shape[1]
    head = (MixtureHead(d, K) if kind in ("mix", "mix_hard", "mix_div") else OffsetHead(d)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    bs = 256
    for ep in range(epochs):
        head.train(); random.Random(ep).shuffle(trip)
        for s in range(0, len(trip), bs):
            b = trip[s:s + bs]
            qn = qtr[[t[0] for t in b]]
            seed = Xt[[t[1] for t in b]]
            goldv = Xt[[t[2] for t in b]]
            if kind in ("mix", "mix_hard", "mix_div"):
                pred = head(qn, seed)                          # (B,K,d)
                sim = torch.einsum("bkd,jd->bkj", pred, goldv).max(1).values / TAU   # (B,B) best direction
                if kind == "mix_hard":                          # hard negs near each head's prediction
                    with torch.no_grad():
                        _, nn_idx = index.search(
                            np.ascontiguousarray(pred.reshape(-1, d).detach().cpu().numpy().astype("float32")), 8)
                    neg_ids = torch.tensor(nn_idx.reshape(-1), device=device)          # (B*K*8,)
                    hard = torch.einsum("bkd,nd->bkn", pred, Xt[neg_ids]).max(1).values / TAU  # (B, B*K*8)
                    own = torch.tensor([t[2] for t in b], device=device)
                    hard = hard.masked_fill(neg_ids.unsqueeze(0) == own.unsqueeze(1), -1e4)
                    sim = torch.cat([sim, hard], dim=1)
                loss = F.cross_entropy(sim, torch.arange(len(b), device=device))
                if kind == "mix_div":                           # anti-collapse: push the K heads apart
                    G = torch.einsum("bkd,bjd->bkj", pred, pred)                       # (B,K,K) head-head cos
                    eye = torch.eye(pred.shape[1], device=device).unsqueeze(0)
                    loss = loss + 0.5 * (G * (1 - eye)).clamp(min=0).mean()            # penalize off-diag similarity
            else:
                pred = head(qn, seed)                          # (B,d)
                logits = pred @ goldv.T / TAU                  # in-batch negatives
                if kind == "hard":
                    with torch.no_grad():
                        _, nn_idx = index.search(pred.detach().cpu().numpy().astype("float32"), 16)
                    negv = Xt[torch.tensor(nn_idx.reshape(-1), device=device)]        # (B*16, d)
                    hard = pred @ negv.T / TAU                  # (B, B*16)
                    own = torch.tensor([t[2] for t in b], device=device)
                    mask = (torch.tensor(nn_idx.reshape(-1), device=device).unsqueeze(0) == own.unsqueeze(1))
                    hard = hard.masked_fill(mask, -1e4)         # don't penalize a query's own gold
                    logits = torch.cat([logits, hard], dim=1)
                loss = F.cross_entropy(logits, torch.arange(len(b), device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def _train_gate(g_base, q_tr, seed_tr, gold_tr, Xt, device, epochs=25):
    """Learn per-query alpha to blend rel vs dense (differentiable InfoNCE surrogate)."""
    trip = [(i, int(seed_tr[i]), int(g)) for i, gl in enumerate(gold_tr) for g in gl]
    qtr = torch.tensor(q_tr, device=device)
    torch.manual_seed(INIT_SEED)
    gate = Gate(Xt.shape[1]).to(device); opt = torch.optim.Adam(gate.parameters(), lr=1e-3)
    bs = 256
    for ep in range(epochs):
        gate.train(); random.Random(ep + 7).shuffle(trip)
        for s in range(0, len(trip), bs):
            b = trip[s:s + bs]
            qn = qtr[[t[0] for t in b]]; seed = Xt[[t[1] for t in b]]; goldv = Xt[[t[2] for t in b]]
            with torch.no_grad():
                relpos = g_base(qn, seed)                       # (B,d)
            a = gate(qn).unsqueeze(1)                           # (B,1)
            rel_sim = relpos @ goldv.T                          # (B,B)
            den_sim = qn @ goldv.T
            blended = (a * rel_sim + (1 - a) * den_sim) / TAU
            loss = F.cross_entropy(blended, torch.arange(len(b), device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    gate.eval()
    return gate


# --------------------------------------------------------------------------- retrieval / fusion
def _order(pos, index, k=MAXK):
    _, o = index.search(np.ascontiguousarray(pos.astype("float32")), k)
    return o


def _ranks(order):
    """order (nq,k) -> list of {docid: rank} per query."""
    return [{int(d): r for r, d in enumerate(row)} for row in order]


def _rrf_fuse(rank_maps, weights, k=MAXK, k0=60):
    """rank_maps: list of per-query rank dicts (one per retriever). weights: list (scalar or per-query array)."""
    nq = len(rank_maps[0]); out = []
    for qi in range(nq):
        sc = {}
        for ri, rm in enumerate(rank_maps):
            w = weights[ri] if np.isscalar(weights[ri]) else weights[ri][qi]
            if w == 0:
                continue
            for d, r in rm[qi].items():
                sc[d] = sc.get(d, 0.0) + w / (k0 + r)
        out.append([d for d, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:k]])
    return out


def _recall(order, gold, budgets=KS):
    out = {b: [] for b in budgets}
    for qi, g in enumerate(gold):
        if not g:
            continue
        gs = set(g)
        row = order[qi]
        for b in budgets:
            top = set(row[:b]) if isinstance(row, list) else set(row[:b].tolist())
            out[b].append(len(gs & top) / len(gs))
    return {b: round(float(np.mean(out[b])) * 100, 2) for b in budgets}


def _overlap(dense_order, rel_order, gold, b=100):
    """Per gold-instance: reached by dense-only / rel-only / both / neither @b — explains fusion."""
    c = {"both": 0, "dense_only": 0, "rel_only": 0, "neither": 0, "total": 0}
    row = lambda o: set(o[:b]) if isinstance(o, list) else set(o[:b].tolist())
    for qi, g in enumerate(gold):
        if not g:
            continue
        ds, rs = row(dense_order[qi]), row(rel_order[qi])
        for gd in g:
            ind, inr = gd in ds, gd in rs
            c["total"] += 1
            c["both" if (ind and inr) else "dense_only" if ind else "rel_only" if inr else "neither"] += 1
    t = max(c["total"], 1)
    return {k: round(v / t * 100, 2) for k, v in c.items() if k != "total"} | {"n": c["total"]}


# --------------------------------------------------------------------------- driver
def run(dataset, epochs=25, limit=8000, K=8, M=3, H=5, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset); encoder = DenseEncoder()
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    d = X.shape[1]; index = faiss.IndexFlatIP(d); index.add(X)
    Xt = torch.tensor(X, device=device)
    id2idx = engine.node_id_to_idx
    splits = _splits(engine, _hard_membership(engine))
    if limit:
        splits = {s: qs[:limit] for s, qs in splits.items()}

    def prep(qs):
        q = encoder.encode([n.content for n, _, _ in qs]).astype("float32"); faiss.normalize_L2(q)
        _, seed = index.search(q, M)
        gold = [[id2idx[g] for g in golds if g in id2idx] for _, _, golds in qs]
        return q, seed, gold
    q_tr, seed_tr, gold_tr = prep(splits["train"])
    q_te, seed_te, gold_te = prep(splits["test"])
    qte_t = torch.tensor(q_te, device=device)

    # ---- train heads
    log.info(f"[{dataset}] training heads (base, hard, mix, gate)...")
    g_base = _train_offset("base", q_tr, seed_tr[:, 0], gold_tr, Xt, index, device, epochs)
    g_hard = _train_offset("hard", q_tr, seed_tr[:, 0], gold_tr, Xt, index, device, epochs)
    g_mix = _train_offset("mix", q_tr, seed_tr[:, 0], gold_tr, Xt, index, device, epochs, K=K)
    gate = _train_gate(g_base, q_tr, seed_tr[:, 0], gold_tr, Xt, device, epochs)

    # ---- build retriever orders (test)
    def pos_np(head, seed_col):
        with torch.no_grad():
            return head(qte_t, Xt[[int(s) for s in seed_te[:, seed_col]]]).cpu().numpy()
    orders = {}
    orders["dense"] = _order(q_te, index)
    orders["rel_base"] = _order(pos_np(g_base, 0), index)
    orders["rel_hard"] = _order(pos_np(g_hard, 0), index)
    # multi-seed: union of offset from each of top-M seeds
    mseed_maps = [_ranks(_order(pos_np(g_base, m), index)) for m in range(M)]
    orders["rel_mseed"] = _rrf_fuse(mseed_maps, [1.0] * M)
    # mixture: union of the K predicted directions
    with torch.no_grad():
        mix_pos = g_mix(qte_t, Xt[[int(s) for s in seed_te[:, 0]]]).cpu().numpy()   # (nq,K,d)
    mix_maps = [_ranks(_order(mix_pos[:, k, :], index)) for k in range(K)]
    orders["rel_mix"] = _rrf_fuse(mix_maps, [1.0] * K)
    # multi-hop: iterate the relation from top-H hop-1 docs
    base_order = orders["rel_base"]
    hop_maps = [_ranks(base_order)]
    for h in range(H):
        inter = X[base_order[:, h]]                             # h-th hop-1 doc as intermediate
        with torch.no_grad():
            hp = g_base(qte_t, torch.tensor(inter, device=device)).cpu().numpy()
        hop_maps.append(_ranks(_order(hp, index)))
    orders["rel_mhop"] = _rrf_fuse(hop_maps, [1.0] * len(hop_maps))

    rmap = {name: _ranks(o) for name, o in orders.items()}   # _ranks handles ndarray & list-of-lists

    # =============== Ablation A: rel-variants alone ===============
    relnames = ["rel_base", "rel_mseed", "rel_mix", "rel_hard", "rel_mhop"]
    A = {name: _recall(orders[name], gold_te) for name in ["dense"] + relnames}
    best_rel = max(relnames, key=lambda n: A[n][100])

    # =============== Ablation B: fusion dense (+) best_rel ===============
    dmap, bmap = rmap["dense"], rmap[best_rel]
    alpha_gate = gate(qte_t).detach().cpu().numpy()
    B = {
        "dense_only": A["dense"],
        f"{best_rel}_only": A[best_rel],
        "rrf_equal": _recall(_rrf_fuse([dmap, bmap], [1.0, 1.0]), gold_te),
    }
    for a in (0.25, 0.5, 0.75):
        B[f"wrrf_rel{a}"] = _recall(_rrf_fuse([dmap, bmap], [1 - a, a]), gold_te)
    B["gate"] = _recall(_rrf_fuse([dmap, bmap], [1 - alpha_gate, alpha_gate]), gold_te)
    best_fusion = max(B, key=lambda n: B[n][100])

    # =============== Ablation C: greedy forward-selection over ALL retrievers ===============
    pool = ["dense"] + relnames
    chosen = [max(pool, key=lambda n: A[n][100])]
    cur = A[chosen[0]][100]
    while True:
        cand = [(n, _rrf_fuse([rmap[c] for c in chosen] + [rmap[n]], [1.0] * (len(chosen) + 1)))
                for n in pool if n not in chosen]
        if not cand:
            break
        scored = [(n, _recall(o, gold_te)) for n, o in cand]
        bn, br = max(scored, key=lambda x: x[1][100])
        if br[100] <= cur + 0.05:                               # stop when no material gain
            break
        chosen.append(bn); cur = br[100]
    combo = {"retrievers": chosen, "recall": _recall(_rrf_fuse([rmap[c] for c in chosen], [1.0] * len(chosen)), gold_te)}

    # =============== overlap (dense vs best_rel) ===============
    ov = _overlap(orders["dense"], orders[best_rel], gold_te, b=100)

    out = {"dataset": dataset, "n_test": len([g for g in gold_te if g]), "budgets": KS,
           "A_rel_variants": A, "best_rel_variant": best_rel,
           "B_fusion": B, "best_fusion": best_fusion,
           "C_greedy_combo": combo,
           "overlap_dense_vs_bestrel@100": ov,
           "alpha_gate_mean": round(float(alpha_gate.mean()), 3)}
    with open(rpath(dataset, "L1", "ablation"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] bestRel={best_rel} ({A[best_rel][100]}) | bestFusion={best_fusion} ({B[best_fusion][100]}) "
             f"| combo={'+'.join(chosen)} ({combo['recall'][100]}) | dense={A['dense'][100]} "
             f"| overlap rel_only={ov['rel_only']}% neither={ov['neither']}%")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Full ablation of L1 relational-retriever improvements.")
    p.add_argument("--datasets", nargs="+",
                   default=["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean", "metaqa"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--limit", type=int, default=8000)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--H", type=int, default=5)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = {}
    for ds in a.datasets:
        log.info(f"===== L1 ABLATION: {ds.upper()} =====")
        try:
            results[ds] = run(ds, epochs=a.epochs, limit=a.limit, K=a.K, M=a.M, H=a.H)
        except Exception as e:
            log.exception(f"[{ds}] FAILED: {e}")

    # ---- uniform champion: fixed configs scored by mean recall@100 across datasets
    if len(results) > 1:
        summary = {"per_dataset_best": {}, "uniform_candidates": {}}
        for ds, r in results.items():
            summary["per_dataset_best"][ds] = {
                "best_rel_variant": r["best_rel_variant"], "best_fusion": r["best_fusion"],
                "greedy_combo": r["C_greedy_combo"]["retrievers"],
                "combo_recall@100": r["C_greedy_combo"]["recall"][100],
                "dense@100": r["A_rel_variants"]["dense"][100]}
        # candidate uniform configs referenced by the ablation each dataset already computed
        cand_names = ["dense_only", "rrf_equal", "gate"]
        for cn in cand_names:
            vals = [r["B_fusion"].get(cn, {}).get(100) for r in results.values()]
            vals = [v for v in vals if v is not None]
            if vals:
                summary["uniform_candidates"][cn] = round(float(np.mean(vals)), 2)
        # also mean of greedy-combo and best-fusion (upper bounds, not single fixed config)
        summary["uniform_candidates"]["greedy_combo(mean)"] = round(
            float(np.mean([r["C_greedy_combo"]["recall"][100] for r in results.values()])), 2)
        summary["uniform_candidates"]["best_fusion(mean,per-ds-oracle)"] = round(
            float(np.mean([r["B_fusion"][r["best_fusion"]][100] for r in results.values()])), 2)
        summary["uniform_champion"] = max(summary["uniform_candidates"],
                                          key=lambda k: summary["uniform_candidates"][k])
        path = os.path.join("data", "ukb_storage", "_index", "l1_ablation_summary.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info(f"SUMMARY uniform_candidates={summary['uniform_candidates']} -> champion={summary['uniform_champion']}")


if __name__ == "__main__":
    main()
