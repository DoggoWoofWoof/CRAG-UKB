"""Shared L2 E0-E3 library: canonical C/P50 corpus loaders, expert scorers, fixed fusions,
per-query listwise InfoNCE trainer (multi-positive correct), and the evaluation harness.

Invariants enforced everywhere:
  - SAME query splits, SAME P50 candidate universe, SAME positive labels, SAME evaluator.
  - Ranking metrics separate L1 pruning failure from L2 ranking failure via three denominators:
      ALL_EVAL_QUERIES            (queries expected to have gold; L1_ANY_FAIL counted as recall 0)
      COND_ANY_GOLD_IN_SCOPE      (queries with >=1 in-scope gold)  <- the honest L2-ranking denominator
      COND_ALL_GOLD_IN_SCOPE      (queries with ALL golds in scope) <- for multi-gold ALL@K
"""
import os, json, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse as sp

UKB = "data/ukb_storage"
CORP = "data/l2_corpus"
KS = [1, 5, 10, 20, 50, 100]
TAU = 0.05          # matches l1_ablate.TAU
RRF_K0 = 60         # matches the locked L1 partition-RRF k0
DEV = torch.device("cpu")


# ----------------------------------------------------------------- heads (exact arch)
class OffsetHead(nn.Module):
    def __init__(self, d=1536):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, d))
    def forward(self, qn, seed):                       # qn,seed:(B,d) normalized
        return F.normalize(seed + self.net(qn), dim=-1)


class MixtureHead(nn.Module):
    def __init__(self, d=1536, K=8):
        super().__init__(); self.K, self.d = K, d
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, K * d))
    def forward(self, qn, seed):                       # -> (B,K,d)
        off = self.net(qn).view(-1, self.K, self.d)
        return F.normalize(seed.unsqueeze(1) + off, dim=-1)


# ----------------------------------------------------------------- loaders
_EMB_CACHE = {}

def load_embeddings(ds):
    if ds in _EMB_CACHE:
        return _EMB_CACHE[ds]
    base = f"{UKB}/{ds}/gte_qwen"
    doc = np.load(f"{base}/nodes.npy").astype("float32")
    Dn = F.normalize(torch.from_numpy(doc), dim=1)
    q = np.load(f"{base}/queries_all.npy").astype("float32")
    Qn = F.normalize(torch.from_numpy(q), dim=1)
    dtop = np.load(f"{base}/dense_top200_all.npy")
    _EMB_CACHE[ds] = (Dn, Qn, dtop)
    return _EMB_CACHE[ds]


def load_split(ds, split):
    d = f"{CORP}/{ds}/{split}"
    out = dict(
        off=np.load(f"{d}/query_offsets.npy"),
        cand=np.load(f"{d}/cand_ids.npy"),
        labels=np.load(f"{d}/labels.npy"),
        dense_score=np.load(f"{d}/dense_score.npy").astype("float32"),
        splade_scope_score=np.load(f"{d}/splade_scope_score.npy").astype("float32"),
        splade_global_rank=np.load(f"{d}/splade_rank.npy"),
        meta=json.load(open(f"{d}/query_meta.json")),
        dir=d,
    )
    # train subset (all positives + tagged negatives), local indices into each query's scope
    if os.path.exists(f"{d}/train_sub_offsets.npy"):
        out["ts_off"] = np.load(f"{d}/train_sub_offsets.npy")
        out["ts_local"] = np.load(f"{d}/train_sub_local.npy")
    return out


def scope(S, qi):
    s, e = int(S["off"][qi]), int(S["off"][qi + 1])
    return s, e


# ----------------------------------------------------------------- expert per-candidate scores
def dense_scores(S, qi):
    s, e = scope(S, qi)
    return S["dense_score"][s:e]


def splade_scores(S, qi):
    s, e = scope(S, qi)
    return S["splade_scope_score"][s:e]


def head_scores(S, qi, head, Dn, Qn, dtop, kind):
    """OffsetHead/MixtureHead per-candidate score over the query scope."""
    s, e = scope(S, qi)
    cand = torch.from_numpy(S["cand"][s:e].astype("int64"))
    r = S["meta"][qi]["row_all"]
    seed_idx = int(dtop[r, 0])
    qn = Qn[r:r + 1]; seed = Dn[seed_idx:seed_idx + 1]
    with torch.no_grad():
        pred = head(qn, seed)               # offset (1,d) / mixture (1,K,d)
        C = Dn[cand]
        if pred.dim() == 2:
            sc = C @ pred[0]
        else:
            sc = (C @ pred[0].T).max(dim=1).values
    return sc.numpy()


def precompute_head_preds(S, head, Dn, Qn, dtop, batch=1024):
    """One batched forward over ALL queries → preds tensor (Nq,d) offset / (Nq,K,d) mixture.
    Removes the per-query MLP forward from eval loops (the regime bottleneck)."""
    nq = len(S["off"]) - 1
    rows = torch.tensor([S["meta"][qi]["row_all"] for qi in range(nq)], dtype=torch.long)
    seed_idx = torch.from_numpy(dtop[rows.numpy(), 0].astype("int64"))
    preds = []
    with torch.no_grad():
        for b in range(0, nq, batch):
            rb = rows[b:b + batch]; sb = seed_idx[b:b + batch]
            preds.append(head(Qn[rb], Dn[sb]))
    return torch.cat(preds, dim=0)


def head_scores_pre(S, qi, preds, Dn):
    """Per-candidate score using a precomputed pred row (see precompute_head_preds)."""
    s, e = scope(S, qi)
    cand = torch.from_numpy(S["cand"][s:e].astype("int64"))
    p = preds[qi]
    with torch.no_grad():
        C = Dn[cand]
        sc = C @ p if p.dim() == 1 else (C @ p.T).max(dim=1).values
    return sc.numpy()


def precompute_head_scope_scores(S, preds, Dn, block=256):
    """Flat per-candidate score array aligned to S['cand'] (like dense_score), via blocked
    doc-matmuls instead of a per-query gather+matmul. Identical math, ~30x faster on CPU."""
    nq = len(S["off"]) - 1
    off = S["off"]; cand = S["cand"]
    allsc = np.empty(len(cand), dtype=np.float32)
    DnT = Dn.t().contiguous()
    with torch.no_grad():
        for b in range(0, nq, block):
            pb = preds[b:min(b + block, nq)]
            if pb.dim() == 2:
                doc = (pb @ DnT)                                  # (Q, Ndoc)
            else:
                Q, K, d = pb.shape
                doc = (pb.reshape(Q * K, d) @ DnT).reshape(Q, K, -1).max(dim=1).values
            doc = doc.numpy()
            for j in range(doc.shape[0]):
                qi = b + j; s, e = off[qi], off[qi + 1]
                allsc[s:e] = doc[j][cand[s:e]]
    return allsc


# ----------------------------------------------------------------- fixed fusions (parameter-free)
def _ranks_from_scores(score):
    order = np.argsort(-score, kind="stable")
    rk = np.empty(len(order), dtype=np.int64); rk[order] = np.arange(len(order))
    return rk


def rrf_fuse(score_list, k0=RRF_K0):
    """Reciprocal-rank fusion over per-candidate expert scores. Scale-free. Higher = better."""
    fused = np.zeros(len(score_list[0]), dtype=np.float64)
    for sc in score_list:
        rk = _ranks_from_scores(sc)
        fused += 1.0 / (k0 + rk)
    return fused


def znorm_fuse(score_list):
    """Per-query z-normalized score sum. Documented explicit transformation."""
    fused = np.zeros(len(score_list[0]), dtype=np.float64)
    for sc in score_list:
        sc = sc.astype(np.float64); mu = sc.mean(); sd = sc.std()
        fused += (sc - mu) / (sd + 1e-8)
    return fused


# ----------------------------------------------------------------- evaluation harness
def _query_gold_local(S, qi):
    s, e = scope(S, qi)
    lab = S["labels"][s:e]
    return np.where(lab == 1)[0]


def eval_ranking(S, score_fn, collect_regime=False):
    """score_fn(qi)->np.array over scope (higher=better). Returns metrics dict with 3 denominators.
    Also returns per-query best_gold_rank array (0-idx, -1 if no in-scope gold) for regime raw material."""
    Nq = len(S["meta"])
    per_best = np.full(Nq, -1, dtype=np.int64)
    per_worst = np.full(Nq, -1, dtype=np.int64)
    # accumulators keyed by denominator
    acc = {k: {"rr": [], "recall": {K: [] for K in KS}, "any": {K: [] for K in KS},
               "allk": {K: [] for K in KS}, "best": [], "worst": []} for k in
           ("ALL_EVAL_QUERIES", "COND_ANY_GOLD_IN_SCOPE", "COND_ALL_GOLD_IN_SCOPE")}
    n_l1_fail = n_partial = n_allsucc = n_expected = 0
    for qi in range(Nq):
        m = S["meta"][qi]
        expected = m["N_GOLD_EXPECTED_INCORP"]
        status = m["L1_STATUS"]
        if status == "L1_ANY_FAIL": n_l1_fail += 1
        elif status == "L1_PARTIAL": n_partial += 1
        elif status == "L1_ALL_SUCCESS": n_allsucc += 1
        if expected and expected > 0:
            n_expected += 1
        gold = _query_gold_local(S, qi)
        ngold = len(gold)
        if ngold == 0:
            # L1 pruning failure: contributes to ALL_EVAL_QUERIES as a miss (recall 0), never to COND_*.
            if expected and expected > 0:
                a = acc["ALL_EVAL_QUERIES"]
                a["rr"].append(0.0)
                for K in KS:
                    a["recall"][K].append(0.0); a["any"][K].append(0.0); a["allk"][K].append(0.0)
            continue
        sc = score_fn(qi)
        rk = _ranks_from_scores(sc)
        gr = rk[gold]
        best = int(gr.min()); worst = int(gr.max())
        per_best[qi] = best; per_worst[qi] = worst
        rr = 1.0 / (best + 1)
        rec = {K: float((gr < K).sum()) / ngold for K in KS}
        anyk = {K: 1.0 if best < K else 0.0 for K in KS}
        allk = {K: 1.0 if worst < K else 0.0 for K in KS}
        # ALL_EVAL_QUERIES (end-to-end; includes queries that had gold in scope)
        for name in ("ALL_EVAL_QUERIES", "COND_ANY_GOLD_IN_SCOPE"):
            a = acc[name]
            a["rr"].append(rr); a["best"].append(best); a["worst"].append(worst)
            for K in KS:
                a["recall"][K].append(rec[K]); a["any"][K].append(anyk[K]); a["allk"][K].append(allk[K])
        if m["ALL_GOLD_PRESENT"]:
            a = acc["COND_ALL_GOLD_IN_SCOPE"]
            a["rr"].append(rr); a["best"].append(best); a["worst"].append(worst)
            for K in KS:
                a["recall"][K].append(rec[K]); a["any"][K].append(anyk[K]); a["allk"][K].append(allk[K])

    def agg(a):
        if not a["rr"]:
            return {"n": 0}
        out = {"n": len(a["rr"]), "MRR": float(np.mean(a["rr"])),
               "best_gold_rank_mean": float(np.mean(a["best"])) if a["best"] else None,
               "best_gold_rank_median": float(np.median(a["best"])) if a["best"] else None,
               "worst_gold_rank_mean_multigold": None}
        # worst-gold mean over multi-gold queries only
        multi = [w for w, b in zip(a["worst"], a["best"]) if True]
        wm = [w for w in a["worst"]]
        if wm:
            out["worst_gold_rank_mean"] = float(np.mean(wm))
        for K in KS:
            out[f"R@{K}"] = float(np.mean(a["recall"][K]))
            out[f"ANY@{K}"] = float(np.mean(a["any"][K]))
            out[f"ALL@{K}"] = float(np.mean(a["allk"][K]))
        return out

    res = {name: agg(a) for name, a in acc.items()}
    res["_counts"] = {"Nq": Nq, "N_expected_gold": n_expected, "L1_ANY_FAIL": n_l1_fail,
                      "L1_PARTIAL": n_partial, "L1_ALL_SUCCESS": n_allsucc}
    if collect_regime:
        return res, per_best, per_worst
    return res


# ----------------------------------------------------------------- batched listwise InfoNCE trainer
def build_train_tensors(ds, smoke_n=None, seed=1234, gmax=None, nneg=128):
    """Dense padded tensors over TRAIN queries (>=1 in-scope gold), from the CANONICAL train_sub.
    Multi-positive: every in-scope gold kept as a positive slot; negatives are the query's sampled
    non-gold train_sub candidates (=> ZERO false negatives). Returns dict of tensors + sampler stats."""
    Dn, Qn, dtop = load_embeddings(ds)
    S = load_split(ds, "train")
    ts_off, ts_local = S["ts_off"], S["ts_local"]
    rows, seeds, golds, negs = [], [], [], []
    neg_counts = []; pos_collisions = 0
    Nq = len(S["meta"]); order_q = list(range(Nq))
    if smoke_n:
        rng = np.random.default_rng(seed); order_q = sorted(rng.choice(Nq, size=min(smoke_n, Nq), replace=False).tolist())
    for qi in order_q:
        s, e = scope(S, qi)
        lab = S["labels"][s:e]; gl = np.where(lab == 1)[0]
        if len(gl) == 0:
            continue
        a, b = int(ts_off[qi]), int(ts_off[qi + 1])
        sub = ts_local[a:b]; nl = sub[lab[sub] == 0]
        pos_collisions += sum(1 for x in nl.tolist() if x in set(gl.tolist()))
        neg_counts.append(len(nl))
        cand = S["cand"][s:e].astype("int64")
        rows.append(S["meta"][qi]["row_all"]); seeds.append(int(dtop[S["meta"][qi]["row_all"], 0]))
        golds.append(cand[gl]); negs.append(cand[nl])
    Nt = len(rows)
    GM = gmax or max(len(g) for g in golds)
    gold_ids = np.full((Nt, GM), -1, np.int64); gmask = np.zeros((Nt, GM), bool)
    neg_ids = np.full((Nt, nneg), -1, np.int64); nmask = np.zeros((Nt, nneg), bool)
    for i, (g, n) in enumerate(zip(golds, negs)):
        gg = g[:GM]; gold_ids[i, :len(gg)] = gg; gmask[i, :len(gg)] = True
        nn_ = n[:nneg]; neg_ids[i, :len(nn_)] = nn_; nmask[i, :len(nn_)] = True
    return {
        "rows": torch.from_numpy(np.array(rows, np.int64)), "seeds": torch.from_numpy(np.array(seeds, np.int64)),
        "gold_ids": torch.from_numpy(gold_ids), "gmask": torch.from_numpy(gmask),
        "neg_ids": torch.from_numpy(neg_ids), "nmask": torch.from_numpy(nmask),
        "stat": {"queries": Nt, "mean_negs_per_query": float(np.mean(neg_counts)) if neg_counts else 0,
                 "positive_collisions": int(pos_collisions), "gmax": GM, "nneg": nneg},
    }


def _batch_loss(head, T, idx, Dn, Qn):
    r = T["rows"][idx]; sidx = T["seeds"][idx]
    qn = Qn[r]; seed = Dn[sidx]
    pred = head(qn, seed)                                  # (B,d) or (B,K,d)
    gid = T["gold_ids"][idx]; gmask = T["gmask"][idx]
    nid = T["neg_ids"][idx]; nmask = T["nmask"][idx]
    gemb = Dn[gid.clamp(min=0)]                            # (B,GM,d)
    nemb = Dn[nid.clamp(min=0)]                            # (B,Nneg,d)
    if pred.dim() == 2:
        pos = torch.einsum("bd,bgd->bg", pred, gemb)       # (B,GM)
        neg = torch.einsum("bd,bnd->bn", pred, nemb)       # (B,Nneg)
    else:
        pos = torch.einsum("bkd,bgd->bkg", pred, gemb).max(1).values   # (B,GM)
        neg = torch.einsum("bkd,bnd->bkn", pred, nemb).max(1).values   # (B,Nneg)
    neg = neg.masked_fill(~nmask, -1e4)
    B, GM = pos.shape
    logits = torch.cat([pos.unsqueeze(2), neg.unsqueeze(1).expand(B, GM, neg.shape[1])], dim=2) / TAU  # (B,GM,1+Nneg)
    logp0 = logits[..., 0] - torch.logsumexp(logits, dim=2)   # (B,GM) log-prob of the positive
    loss = -(logp0 * gmask).sum() / gmask.sum().clamp(min=1)
    return loss


def train_head(ds, kind, epochs=8, lr=1e-3, seed=1234, smoke_n=None, log=print, K=8,
               QB=256, val_select=False, base_experts=None, patience=3):
    """Batched per-query listwise InfoNCE. If val_select, do per-epoch VAL fused-RRF MRR checkpoint
    selection (base_experts = list of S->lambda scorers, head appended). Returns (head, tlog)."""
    torch.manual_seed(seed); np.random.seed(seed)
    Dn, Qn, dtop = load_embeddings(ds)
    T = build_train_tensors(ds, smoke_n=smoke_n, seed=seed)
    head = (MixtureHead(1536, K) if kind == "mixture" else OffsetHead(1536)).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Nt = T["stat"]["queries"]
    tlog = {**T["stat"], "epochs": epochs, "lr": lr, "tau": TAU, "seed": seed, "QB": QB,
            "kind": kind, "K": (K if kind == "mixture" else None), "loss_curve": [], "val_mrr_curve": [],
            "loss": "batched per-query listwise InfoNCE over canonical sampled negs; multi-positive (each in-scope gold a positive slot, negs exclude all golds)"}
    Sv = load_split(ds, "val") if val_select else None
    best = {"mrr": -1, "state": None, "epoch": -1}
    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.from_numpy(np.random.default_rng(seed * 100 + ep).permutation(Nt))
        ep_loss = 0.0; nb = 0
        for bs in range(0, Nt, QB):
            idx = perm[bs:bs + QB]
            opt.zero_grad()
            loss = _batch_loss(head, T, idx, Dn, Qn)
            loss.backward(); opt.step()
            ep_loss += float(loss.detach()); nb += 1
        head.eval()
        tlog["loss_curve"].append(ep_loss / max(1, nb))
        msg = f"    [{ds}/{kind}] ep{ep}/{epochs} loss={tlog['loss_curve'][-1]:.4f}"
        if val_select:
            hs = lambda qi, h=head: head_scores(Sv, qi, h, Dn, Qn, dtop, kind)
            fn = lambda qi: rrf_fuse([e(Sv)(qi) for e in base_experts] + [hs(qi)])
            mrr = eval_ranking(Sv, fn)["COND_ANY_GOLD_IN_SCOPE"]["MRR"]
            tlog["val_mrr_curve"].append(mrr)
            if mrr > best["mrr"] + 1e-5:
                best = {"mrr": mrr, "state": {k: v.clone() for k, v in head.state_dict().items()}, "epoch": ep}
            msg += f" VAL_fusedMRR={mrr:.4f} (best {best['mrr']:.4f}@{best['epoch']})"
        log(msg)
        if val_select and ep - best["epoch"] >= patience:
            log("    early stop"); break
    if val_select and best["state"] is not None:
        head.load_state_dict(best["state"]); tlog["best_epoch"] = best["epoch"]; tlog["best_val_mrr"] = best["mrr"]
    head.eval()
    return head, tlog
