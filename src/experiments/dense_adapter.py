"""
Universal dense adapter — make the SAME frozen encoder better (no scaling, no LLM).
=====================================================================================
A small residual MLP A(x) = L2norm(x + g(x)) applied to BOTH query and document embeddings, trained
contrastively (InfoNCE + hard negatives) as ONE universal adapter across corpora. The gte-Qwen2-1.5B
encoder stays frozen; only the adapter (~a few M params) trains. Goal: reshape the frozen space so
plain dense retrieval improves — i.e. raise the dense FLOOR that the whole pipeline sits on.

Residual + near-zero init on the last layer => the adapter STARTS as the identity (adapted-dense == raw
dense at init), so training can only depart from raw dense if it helps the contrastive objective.

Reports adapted-dense vs raw-dense Recall@k on the eval corpora. Writes results/L2/dense_adapter_{subdir}.json.
"""
import os
import json
import logging
import argparse
import random as _random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

from src.experiments.l1_universal_head import _load
from src.experiments.l2_seed import _recall, KS
from src.experiments.e2e_pipeline import _merge_minrank

log = logging.getLogger(__name__)
INIT_SEED = 0

# training / eval corpora (cache-ready, no encoder load); hotpot held out -> zero-shot check
HEAD_MIX = ["musique_clean", "2wiki_clean"]
HPR_EVAL = ["musique_hpr_clean", "2wiki_hpr_clean", "hotpot_hpr_clean"]


class ResidualAdapter(nn.Module):
    """A(x) = L2norm(x + g(x)); g is a 2-layer MLP whose final layer is zero-init so A == identity at start."""
    def __init__(self, d, hidden=2048):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)   # start as identity

    def forward(self, x):
        return F.normalize(x + self.fc2(F.gelu(self.fc1(x))), dim=-1)


def _mine_hardnegs(qraw, Xraw, golds, k=10):
    """Per-query hard negatives = raw-dense top-k docs that are NOT golds (fixed pool from the frozen space)."""
    idx = faiss.IndexFlatIP(Xraw.shape[1]); idx.add(Xraw)
    _, I = idx.search(qraw, k + 8)
    negs = []
    for qi in range(len(qraw)):
        gs = set(golds[qi])
        negs.append([int(x) for x in I[qi] if int(x) not in gs][:k])
    return negs


def train_adapter(per_ds, device, epochs=30, bs=256, tau=0.05, n_hard=8):
    """ONE residual adapter, InfoNCE with in-batch + hard negatives, iterated over all training corpora.
    Memory-light: document embeddings stay on CPU (numpy); only per-batch gold/hard-neg rows and the
    (small) query set move to the GPU — so a 781k-doc KB corpus trains without full-corpus GPU residency."""
    d = per_ds[list(per_ds)[0]]["X"].shape[1]
    torch.manual_seed(INIT_SEED)
    A = ResidualAdapter(d).to(device)
    opt = torch.optim.AdamW(A.parameters(), lr=2e-4, weight_decay=1e-4)
    trips, Xcpu, Qt = {}, {}, {}
    for dsn, dd in per_ds.items():
        q, _, gold = dd["train"]
        negs = _mine_hardnegs(q.astype("float32"), dd["X"], gold, k=n_hard)
        trips[dsn] = [(qi, g, negs[qi]) for qi, gl in enumerate(gold) for g in gl]
        Xcpu[dsn] = dd["X"]                                           # docs stay on CPU
        Qt[dsn] = torch.tensor(dd["train"][0].astype("float32"), device=device)   # queries (small) on GPU
    for ep in range(epochs):
        A.train(); order = list(per_ds); _random.Random(ep).shuffle(order); tot = 0.0
        for dsn in order:
            trip = trips[dsn][:]; _random.Random(ep * 97 + len(dsn)).shuffle(trip)
            X = Xcpu[dsn]; Q = Qt[dsn]
            for s in range(0, len(trip), bs):
                b = trip[s:s + bs]
                qi = torch.tensor([t[0] for t in b], device=device)
                gi = np.array([t[1] for t in b])
                aq = A(Q[qi]); ag = A(torch.tensor(X[gi], device=device))         # gather golds from CPU
                logits = aq @ ag.T / tau                              # (B,B) in-batch: diagonal = positive
                labels = torch.arange(len(b), device=device)
                loss = F.cross_entropy(logits, labels)
                hard = [t[2] for t in b]
                if any(hard):
                    hn = np.array([h[0] if h else t[1] for h, t in zip(hard, b)])
                    ahn = A(torch.tensor(X[hn], device=device))                   # one hard neg per query
                    hard_logit = (aq * ahn).sum(-1, keepdim=True) / tau
                    loss = loss + F.cross_entropy(torch.cat([logits, hard_logit], 1), labels)
                opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        if ep % 5 == 0 or ep == epochs - 1:
            log.info("  [adapter] epoch %d loss %.4f", ep, tot)
    A.eval()
    return A


def _apply_adapter(A, Xnp, device, bs=16384):
    """Apply the adapter over a (possibly 781k-row) embedding matrix in chunks; returns adapted numpy."""
    out = []
    with torch.no_grad():
        for s in range(0, len(Xnp), bs):
            out.append(A(torch.tensor(np.ascontiguousarray(Xnp[s:s + bs]), device=device)).cpu().numpy())
    return np.concatenate(out, 0)


def _dense_order(qb, Xb, k=100):
    idx = faiss.IndexFlatIP(Xb.shape[1]); idx.add(np.ascontiguousarray(Xb))
    D, I = idx.search(np.ascontiguousarray(qb), min(k, Xb.shape[0]))
    return I, D                                                      # I = doc indices, D = cosine scores


def _combsum(sig_orders, sig_scores, weights=None):
    """Score-based fusion: rank by the WEIGHTED SUM of each doc's cosine across signals (0 where absent).
    weights (per signal) down-weight an unreliable signal so it can't drag the fusion — set them from each
    signal's TRAIN recall so a corpus where raw dense fails (KB) auto-suppresses raw. weights=None => equal."""
    w = weights if weights is not None else [1.0] * len(sig_orders)
    out = []
    for qi in range(len(sig_orders[0])):
        acc = {}
        for wi, I, D in zip(w, sig_orders, sig_scores):
            for doc, sc in zip(I[qi], D[qi]):
                doc = int(doc)
                if doc < 0:
                    continue
                acc[doc] = acc.get(doc, 0.0) + wi * float(sc)
        out.append([d for d, _ in sorted(acc.items(), key=lambda kv: -kv[1])])
    return out


def train_or_load(head_datasets, subdir, epochs, device):
    """ONE universal adapter, trained on head_datasets and cached (fingerprint = config + gold/encoder sample),
    so the pipeline reuses it instead of retraining. Returns the fitted ResidualAdapter."""
    import hashlib
    per_tr = {}
    for dsn in head_datasets:
        dd = _load(dsn, subdir, 8000, 3000, 1)
        per_tr[dsn] = {"X": dd["X"].astype("float32"), "train": dd["train"]}
    fp = hashlib.md5(f"adapter|e{epochs}".encode())
    for dsn in sorted(per_tr):
        fp.update(dsn.encode())
        fp.update(np.asarray([g for gl in per_tr[dsn]["train"][2] for g in gl], dtype=np.int64).tobytes())
        fp.update(np.ascontiguousarray(per_tr[dsn]["X"][:64]).tobytes())
    path = os.path.join("data", "ukb_storage", "_head_cache", f"adapter_{fp.hexdigest()[:16]}.pt")
    d = per_tr[list(per_tr)[0]]["X"].shape[1]
    A = ResidualAdapter(d).to(device)
    if os.path.exists(path):
        A.load_state_dict(torch.load(path, map_location=device)); A.eval()
        log.info("[adapter-cache] reused <- %s", os.path.basename(path))
        return A
    A = train_adapter(per_tr, device, epochs=epochs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(A.state_dict(), path); log.info("[adapter-cache] saved -> %s", os.path.basename(path))
    return A


def run(datasets=None, head_datasets=None, subdir="gte_qwen", epochs=30, device=None):
    import gc
    datasets = datasets or HPR_EVAL
    head_datasets = head_datasets or HEAD_MIX
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    A = train_or_load(head_datasets, subdir, epochs, device)        # cached -> reused across runs
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out = {}
    for dsn in datasets:
        dd = _load(dsn, subdir, 8000, 3000, 2000)
        X = dd["X"].astype("float32"); q = dd["test"][0].astype("float32"); gold = dd["test"][2]
        Iraw, Draw = _dense_order(q, X)
        aX = _apply_adapter(A, X, device); aq = _apply_adapter(A, q, device)
        Iadp, Dadp = _dense_order(aq, aX)
        # reliability weights from TRAIN recall@20 (train-only labels; no test leak) -> auto-suppress raw on KB
        qtr, _, gtr = dd["train"]
        Itr_r, _ = _dense_order(qtr.astype("float32"), X)
        Itr_a, _ = _dense_order(_apply_adapter(A, qtr.astype("float32"), device), aX)
        w_raw = max(_recall(Itr_r, gtr).get(20, 0) / 100, 1e-3); w_adp = max(_recall(Itr_a, gtr).get(20, 0) / 100, 1e-3)
        minrank = [_merge_minrank([Iraw[qi], Iadp[qi]]) for qi in range(len(gold))]     # rank-based union
        combsum = _combsum([Iraw, Iadp], [Draw, Dadp])                                  # equal-weight score fusion
        wcombsum = _combsum([Iraw, Iadp], [Draw, Dadp], weights=[w_raw, w_adp])         # reliability-weighted
        zs = " [zero-shot]" if dsn.split("_")[0] not in {h.split("_")[0] for h in head_datasets} else ""
        out[dsn] = {"raw_dense": _recall(Iraw, gold), "adapted_dense": _recall(Iadp, gold),
                    "fused_minrank": _recall(minrank, gold), "fused_combsum": _recall(combsum, gold),
                    "fused_wcombsum": _recall(wcombsum, gold), "w_raw": round(w_raw, 3), "w_adp": round(w_adp, 3),
                    "zero_shot": bool(zs)}
        rr, ad, mn, cs, wc = (out[dsn][k] for k in ("raw_dense", "adapted_dense", "fused_minrank",
                                                    "fused_combsum", "fused_wcombsum"))
        log.info("[adapter/%s]%s R@5 raw=%.1f adapted=%.1f | minrank=%.1f combsum=%.1f Wcombsum=%.1f (w_raw=%.2f w_adp=%.2f) | R@20 Wc=%.1f",
                 dsn, zs, rr.get(5, 0), ad.get(5, 0), mn.get(5, 0), cs.get(5, 0), wc.get(5, 0), w_raw, w_adp, wc.get(20, 0))
        del dd, X, aX; gc.collect()
    os.makedirs("results/L2", exist_ok=True)
    json.dump(out, open(f"results/L2/dense_adapter_{subdir}.json", "w"), indent=2)
    log.info("-> results/L2/dense_adapter_%s.json", subdir)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Universal residual dense adapter on the frozen encoder.")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--head-datasets", nargs="+", default=None)
    p.add_argument("--subdir", default="gte_qwen")
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, head_datasets=a.head_datasets, subdir=a.subdir, epochs=a.epochs)


if __name__ == "__main__":
    main()
