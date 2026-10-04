"""Hybrid fusion — expressive flat candidate MLP ADVISED by a KL-trained soft expert router.

Findings: flat MLP is expressive (WebQSP 54) but collapses to the dominant signal (MetaQA offset -> 69, misses
+relation); the constrained KL convex-mixture is anti-collapse (MetaQA 77) but under-expressive (WebQSP 44);
BCE gates dead. Synthesis (advisor, not controller): a shared query-state encoder produces soft expert
weights alpha (trained by KL to a per-query expert-quality teacher pi* = softmax(-L_e/tau_r), L_e = each
expert's bounded-InfoNCE loss), and alpha is fed to the flat candidate MLP as FEATURES (raw scores + alpha +
alpha*scores + query-state) rather than forced as a convex sum. So the router advises; the MLP still learns
arbitrary interactions and can override a bad routing decision.

WINNER = full_fd2 (hybrid + consistency-preserving co-dropout): during training (p=0.2) mask BOTH an expert's
raw score s_e AND its advised score alpha_e*s_e fed to the MLP, keep alpha, KL supervises alpha normally. No
contradictory gradient (unlike original `full` which hard-masked the alpha logit -> collapse 37/60). 10-seed
joint benchmark WebQSP R@5 54.80 / MetaQA 87.70.

Modes: flat | flat_drop | hybrid | full (diagnosed collapse) | full_fd1 | full_fd2 (winner).

Two run kinds:
  benchmark  (--datasets ...)                : ONE joint checkpoint per mode, trained on the mixed datasets, eval each.
  transfer   (--train-datasets .. --eval-dataset X): train `mode` on train-datasets, ZERO-SHOT eval on X using
             the FROZEN train-set mu/sd (never recomputed on X) -> genuine transfer test. Also re-evals the
             train datasets to confirm no degradation.
Runs on cached features exported by kg_relsig (relsig_feats_{ds}.pt); no corpus reload.
"""
import logging
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
EXPERTS = [0, 1, 3, 5, 6]   # dense, offset, splade, relation, path  (columns in the 7-feature vector)
NE = len(EXPERTS)
EXPERT_NAMES = ["dense", "offset", "splade", "relation", "path"]     # names for EXPERTS columns (router alpha order)
FEAT_NAMES = ["dense", "offset", "prototype", "splade", "graph", "relation_raw", "path_raw"]  # all 7 raw columns
IDX_RELATION, IDX_PATH = 3, 4                      # positions of relation/path within EXPERTS
KS = (5, 20, 50)                                   # recall cutoffs
HS = (5, 50)                                       # hit cutoffs
METR = [f"R@{k}" for k in KS] + [f"Hit@{k}" for k in HS]
ROUTER_MODES = ("hybrid", "full", "full_fd1", "full_fd2")
DROP_MODES = ("flat_drop", "full", "full_fd1", "full_fd2")
# CHOSEN clipping hyperparameter (NOT a mathematically principled worst-case InfoNCE bound: real InfoNCE is
# unbounded as the positive falls arbitrarily far below the negatives). We clip the per-expert bounded-InfoNCE
# loss to this value so BOTH pathological cases map to the same "worst possible expert" penalty:
#   available + no scorable gold  -> MAX_EXPERT_LOSS   (pi*_e -> 0, but NOT excluded; exclusion is via qavail)
#   available + terrible gold rank-> clamp(loss, max=MAX_EXPERT_LOSS)
# Availability itself is decided ONLY by qavail (an unavailable expert is excluded from the teacher upstream).
MAX_EXPERT_LOSS = 8.0


from dataclasses import dataclass, replace as _dc_replace


# NOTE: PrepBatch is the Phase-2 single-semantics contract. It is defined here now but NOT yet wired into the
# training path (_prep/_train still use the legacy 6-tuple, pending the Phase-2 refactor). The Phase-1 audit
# (diag_substrate / diag_confidence) reads the saved substrate directly via _row_unpack and does not depend on it.
@dataclass
class PrepBatch:
    """One query's frozen substrate, with a SINGLE availability semantics used everywhere.
      raw    (n,ncol) raw expert scores               -- graph coverage / edge-path strength only
      mask   (n,ncol) authoritative per-candidate existence (1/0) -- the ONLY source of availability
      qavail (ncol,)  mask.any over candidates (1/0)  -- query-level per-expert availability (ALL experts)
      x_fuse (n,ncol) present-only z-score, missing->0 -- MLP / fusion inputs ONLY
      x_rank (n,ncol) present-only z-score, missing->-inf -- ranking / top-k / margins / InfoNCE ONLY
      y (n,), ng      golds; w/ds training metadata.
    Existence is NEVER inferred from a score value again."""
    raw: torch.Tensor
    mask: torch.Tensor
    qavail: torch.Tensor
    x_fuse: torch.Tensor
    x_rank: torch.Tensor
    y: torch.Tensor
    ng: int
    w: float = 1.0
    ds: str = ""


def _expert_loss(s, y, M=20):
    """Bounded InfoNCE over an expert's VALID candidates. `s` is an x_rank column (missing = -inf), so a
    candidate the expert cannot score can never be a false positive. An AVAILABLE expert that ranks no gold
    (all its gold scores are -inf) gets the worst bounded loss -> pi*->0, but it is NOT excluded here;
    exclusion happens only via qavail at the teacher. Clamped so a terrible-but-scorable gold can't exceed
    the can't-score-gold penalty."""
    g = s[y > 0]
    g = g[torch.isfinite(g)]                                # golds this expert can actually score
    if g.numel() == 0:
        return torch.tensor(MAX_EXPERT_LOSS, device=s.device)
    neg = s[y == 0]; neg = neg[torch.isfinite(neg)]
    if neg.numel() > M:
        neg = torch.topk(neg, M).values
    denom = torch.cat([g, neg]) if neg.numel() else g
    return (-(torch.logsumexp(g, 0) - torch.logsumexp(denom, 0))).clamp(max=MAX_EXPERT_LOSS)


def _qstate(xz):                                                    # query-state inputs from expert distributions
    feats = []
    for c in EXPERTS:
        s = xz[:, c]; p = F.softmax(s, 0); ent = -(p * (p + 1e-9).log()).sum()
        tp = torch.topk(s, min(2, len(s))).values
        feats += [ent, (tp[0] - tp[-1]) if len(tp) > 1 else tp[0] * 0]
    return torch.stack(feats)                                       # (2*NE,)


class QEnc(nn.Module):
    def __init__(s, nin, h=32):
        super().__init__(); s.m = nn.Sequential(nn.Linear(nin, 64), nn.GELU(), nn.Linear(64, h)); s.a = nn.Linear(h, NE)
    def forward(s, u):
        hq = s.m(u); return hq, s.a(hq)                            # query state + alpha logits


CONF_K = 5
UIN_CONF = 3 * NE + 2                                              # [margin, entropy, agree_others] per expert + avail(2)
STATE_CENTRAL = 3 * NE + 3 + 2 + 2 + NE + 1                        # margin+ent+agree(3NE) + cov(3) + edge_str(2) + path_str(2) + source_comp(NE) + dense_dominance(1)
UIN_CENTRAL = STATE_CENTRAL + 2                                    # + avail(2)


def _qstate_conf(xz, K=CONF_K):
    """Dataset-INDEPENDENT confidence router inputs: per-query STANDARDIZED margin + normalized entropy, plus
    each expert's rank-agreement with the other experts (scale-free). Deliberately NO standardized variance
    (~1 by construction) and NO reliance on raw max. These are the features the router must use to infer
    reliability without a dataset ID. Vectorized (no python-scalar syncs) -> cheap in the training loop."""
    n = xz.shape[0]; ln = float(np.log(max(n, 2)))
    sub = xz[:, EXPERTS]                                           # (n, NE)
    z = (sub - sub.mean(0)) / sub.std(0).clamp(min=1e-6)          # per-expert per-query standardize
    p = F.softmax(z, 0)
    ent = -(p * (p + 1e-9).log()).sum(0) / ln                     # (NE,) normalized entropy in [0,1]
    kk = min(2, n)
    tp = torch.topk(z, kk, dim=0).values
    marg = (tp[0] - tp[-1]) if n > 1 else torch.zeros(NE, device=xz.device)   # (NE,) top1-top2 in std units
    Kk = min(K, n)
    tk = torch.topk(sub, Kk, dim=0).indices                       # (Kk, NE) each expert's top-K candidate ids
    ov = torch.zeros(NE, NE, device=xz.device)
    for i in range(NE):
        for j in range(NE):
            if i != j:
                ov[i, j] = (tk[:, i].unsqueeze(1) == tk[:, j].unsqueeze(0)).any(1).float().sum() / Kk
    agree = ov.sum(1) / (NE - 1)                                   # (NE,) mean top-K overlap with the other experts
    return torch.cat([marg, ent, agree])                          # (3*NE,)


def _corrupt_easy(xz, dcol, sharpen=3.0, gen=None):
    """EASY corruption: permute the expert's candidate scores (ranking becomes wrong) then SHARPEN so it still
    looks confident. This expert now DISAGREES with the others -> detectable via low agreement. A weak lesson."""
    x = xz.clone(); c = EXPERTS[dcol]; col = x[:, c]
    perm = torch.randperm(col.shape[0], device=col.device, generator=gen)
    col = col[perm]
    m = col.mean(); col = m + (col - m) * sharpen
    x[:, c] = col
    return x


def _corrupt_plausible(xz, y, dcol, n_swap=3, sharpen=2.0, gen=None):
    """PLAUSIBLE corruption (mimics the Hotpot-offset failure): keep MOST of the expert's top-k (so it still
    AGREES with the consensus and looks reliable), but promote a few WRONG (non-gold) candidates to the very top
    and sharpen so it stays confident. Teacher InfoNCE marks it worse (golds pushed down). Forces the router to
    learn that a confident, consensus-looking expert can STILL be wrong -- confidence+agreement != reliability."""
    x = xz.clone(); c = EXPERTS[dcol]; s = x[:, c].clone(); n = s.shape[0]
    order = torch.argsort(-s)
    ycpu = y > 0
    band = order[3:min(n, 60)]                                    # mid-band candidates (plausible, not obviously top)
    wrong = band[~ycpu[band]]                                     # non-gold -> genuinely wrong to promote
    if wrong.numel() == 0:
        return x
    k = min(n_swap, wrong.numel())
    pick = wrong[torch.randperm(wrong.numel(), device=s.device, generator=gen)[:k]]
    smax = s.max(); step = s.std().clamp(min=1e-6)
    s[pick] = smax + step * (1.0 + torch.arange(k, device=s.device).float())   # promote wrongs ABOVE the current top
    m = s.mean(); s = m + (s - m) * sharpen                       # stay confident (high margin / low entropy)
    x[:, c] = s
    return x


class Cand(nn.Module):
    def __init__(s, nin):
        super().__init__(); s.m = nn.Sequential(nn.Linear(nin, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(s, x): return s.m(x).squeeze(-1)


def _qmet(ry, ng):                                                 # per-query metrics from ranked gold mask (top-50)
    d = {}
    for k in KS:
        d[f"R@{k}"] = float(ry[:k].sum()) / ng
    for k in HS:
        d[f"Hit@{k}"] = 1.0 if int(ry[:k].sum()) > 0 else 0.0
    return d


def _cand_input(xz, hq, alpha, mode, xz_dim):
    if mode in ("flat", "flat_drop"):
        return xz                                                  # raw normalized 7 features only
    n = xz.shape[0]
    aexp = xz[:, EXPERTS] * alpha[None, :]                         # alpha-weighted expert scores (n,NE)
    return torch.cat([xz, alpha[None, :].expand(n, -1), aexp, hq[None, :].expand(n, -1)], 1)


def _prep(data, datasets, split, mu, sd, DEV):                    # flatten to a list of (xz, y, ng, avail, w, ds)
    out = []
    for ds in datasets:
        for r in data[ds][split]:
            x, y, ng, a = _row_unpack(r)
            x = x.to(DEV); y = y.to(DEV); a = a.to(DEV)
            xz = ((x - mu) / sd) * a                              # masked z-score: missing -> 0 (present-mean), never the leaked sentinel
            avail = torch.stack([a[:, 5].amax(), a[:, 6].amax()])  # query-level relation/path availability from the stored mask
            w = 1.0                                                # disagreement weight: offset misses a gold relation/path finds
            if y.sum() > 0:
                orank = torch.empty(len(y), dtype=torch.long, device=DEV); orank[torch.argsort(-xz[:, 1])] = torch.arange(len(y), device=DEV)
                rrank = torch.empty(len(y), dtype=torch.long, device=DEV); rrank[torch.argsort(-xz[:, 5])] = torch.arange(len(y), device=DEV)
                prank = torch.empty(len(y), dtype=torch.long, device=DEV); prank[torch.argsort(-xz[:, 6])] = torch.arange(len(y), device=DEV)
                gm = y > 0
                if ((orank[gm] >= 5) & ((rrank[gm] < 5) | (prank[gm] < 5))).any():
                    w = 2.0
            out.append((xz, y, ng, avail, w, ds))
    return out


def _pooled_musd(data, datasets, DEV, split="train"):
    """Phase-2 JOINT-training helper: pooled per-column mu/sd over the union of `datasets` (`split` rows),
    computed with ONLY mask=True (present) entries so sentinel/missing values NEVER enter the statistics.
    This is the frozen train-only coordinate system Phase-2 will standardize against; dropout later changes
    availability, never these stats. Returns (mu, sd) on DEV. Defined + tested now; NOT wired into the current
    trainers (they still recompute their own mu/sd pending the Phase-2 refactor)."""
    xs, ms = [], []
    for ds in datasets:
        for r in data[ds][split]:
            x, _y, _ng, a = _row_unpack(r)
            xs.append(x.to(DEV)); ms.append(a.to(DEV).float())
    X = torch.cat(xs, 0); M = torch.cat(ms, 0)
    cnt = M.sum(0).clamp(min=1.0)                                 # present-count per column (never 0)
    mu = (X * M).sum(0) / cnt                                     # present-only mean
    sd = (((X - mu) ** 2 * M).sum(0) / cnt).sqrt().clamp(min=1e-6)  # present-only std
    return mu, sd


def _build(mode, xz_dim, uin, hq_dim, DEV):
    use_router = mode in ROUTER_MODES
    qenc = QEnc(uin, hq_dim).to(DEV) if use_router else None
    nin = xz_dim if mode in ("flat", "flat_drop") else (xz_dim + NE + NE + hq_dim)
    return qenc, Cand(nin).to(DEV)


def _train(mode, seed, TR, xz_dim, uin, DEV, epochs=12, lam=1.0, tau_r=1.0, drop=0.2, hq_dim=32):
    torch.manual_seed(seed); np.random.seed(seed)
    qenc, cand = _build(mode, xz_dim, uin, hq_dim, DEV)
    params = list(cand.parameters()) + (list(qenc.parameters()) if qenc else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    drop_on = mode in DROP_MODES; router = mode in ROUTER_MODES
    weights = np.array([w for *_, w, _ in TR]); order0 = np.arange(len(TR))
    for ep in range(epochs):
        order = np.random.choice(order0, size=len(TR), replace=True, p=weights / weights.sum())  # disagreement oversample
        for j in order:
            xz, y, ng, avail, w, ds = TR[j]
            dcol = np.random.randint(NE) if (drop_on and np.random.rand() < drop) else -1
            if router:
                avail_use = avail
                if mode == "full_fd1" and dcol >= 0:               # corrected v1: reflect masked expert in router input
                    avail_use = avail.clone()
                    if dcol == IDX_RELATION: avail_use[0] = 0.0
                    elif dcol == IDX_PATH: avail_use[1] = 0.0
                hq, alogit = qenc(torch.cat([_qstate(xz), avail_use]))
                if mode == "full" and dcol >= 0:                   # ORIGINAL (bad): hard-mask alpha -> fights KL
                    alogit = alogit.clone(); alogit[dcol] = -1e4
                alpha = F.softmax(alogit, 0)
            else:
                hq = torch.zeros(hq_dim, device=DEV); alpha = torch.zeros(NE, device=DEV)
            xin = _cand_input(xz, hq, alpha, mode, xz_dim)
            if dcol >= 0:                                           # feature-level dropout (leaves alpha/KL intact)
                if mode == "flat_drop":
                    xin = xin.clone(); xin[:, EXPERTS[dcol]] = 0.0
                elif mode in ("full_fd1", "full_fd2"):
                    xin = xin.clone(); xin[:, EXPERTS[dcol]] = 0.0          # mask raw expert score s_e
                    if mode == "full_fd2":
                        xin[:, xz_dim + NE + dcol] = 0.0                    # also mask alpha*s_e column
            S = cand(xin)
            loss = -(F.log_softmax(S, 0) * (y / y.sum())).sum()
            if router:
                with torch.no_grad():
                    Le = torch.stack([_expert_loss(xz[:, c], y) for c in EXPERTS])
                    pi = F.softmax(-Le / tau_r, 0)
                loss = loss + lam * (pi * ((pi + 1e-9).log() - F.log_softmax(alogit, 0))).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return qenc, cand


def _eval(mode, qenc, cand, TE, datasets, xz_dim, hq_dim, DEV):   # per-query ranked gold mask (top-50) + gold count
    recs = {ds: [] for ds in datasets}
    with torch.no_grad():
        for xz, y, ng, avail, w, dd in TE:
            if dd not in recs:
                continue
            if mode in ROUTER_MODES:
                hq, alogit = qenc(torch.cat([_qstate(xz), avail])); alpha = F.softmax(alogit, 0)
            else:
                hq = torch.zeros(hq_dim, device=DEV); alpha = torch.zeros(NE, device=DEV)
            S = cand(_cand_input(xz, hq, alpha, mode, xz_dim)).cpu().numpy()
            ry = y.cpu().numpy()[np.argsort(-S)][:50].astype(np.uint8)
            recs[dd].append((ry, int(ng)))
    return recs


def _aggregate(seed_recs, datasets):                             # -> ({ds:{metric:[mean,std]}}, preds{ds:{seed:[(mask,ng)]}})
    agg = {}; preds = {}
    for ds in datasets:
        per_seed = {m: [] for m in METR}; preds[ds] = {}
        for s, rec in enumerate(seed_recs):
            qms = [_qmet(ry, ng) for ry, ng in rec[ds]]
            for m in METR:
                per_seed[m].append(100.0 * float(np.mean([q[m] for q in qms])))
            preds[ds][s] = [(ry.tolist(), ng) for ry, ng in rec[ds]]
        agg[ds] = {m: [round(float(np.mean(per_seed[m])), 2), round(float(np.std(per_seed[m])), 2)] for m in METR}
    return agg, preds


def _signal_baselines(TE, DEV):
    """'Old CRAG' & per-signal baselines on the SAME frozen eval pool: rank each raw feature column alone,
    apply the identical top-50 R@k/Hit@k cutoffs. Ranking is deterministic (no seeds) -> std=0."""
    acc = {c: {m: [] for m in METR} for c in range(len(FEAT_NAMES))}
    for xz, y, ng, avail, w, dd in TE:
        yn = y.cpu().numpy()
        for c in range(len(FEAT_NAMES)):
            ry = yn[np.argsort(-xz[:, c].cpu().numpy())][:50].astype(np.uint8)
            q = _qmet(ry, ng)
            for m in METR:
                acc[c][m].append(100.0 * q[m])
    return {FEAT_NAMES[c]: {m: [round(float(np.mean(acc[c][m])), 2), 0.0] for m in METR}
            for c in range(len(FEAT_NAMES))}


def _mean_alpha(qenc, TE, DEV):
    """Router's mean soft expert weight over the eval queries (evidence of down/up-weighting per signal)."""
    accs = []
    with torch.no_grad():
        for xz, y, ng, avail, w, dd in TE:
            _, alogit = qenc(torch.cat([_qstate(xz), avail]))
            accs.append(F.softmax(alogit, 0).cpu().numpy())
    return np.mean(np.stack(accs), 0)


def _fusion_and_dense(mode, qenc, cand, TE, datasets, xz_dim, hq_dim, DEV):
    """Per query: (z_dense col-0, S_fusion MLP score, gold mask, ng). Lets us form S_final = z_dense + lam*S_fusion
    for any lambda WITHOUT re-inference -> cheap lambda sweep for dense-anchored residual fusion."""
    recs = {ds: [] for ds in datasets}
    with torch.no_grad():
        for xz, y, ng, avail, w, dd in TE:
            if dd not in recs:
                continue
            if mode in ROUTER_MODES:
                hq, alogit = qenc(torch.cat([_qstate(xz), avail])); alpha = F.softmax(alogit, 0)
            else:
                hq = torch.zeros(hq_dim, device=DEV); alpha = torch.zeros(NE, device=DEV)
            S = cand(_cand_input(xz, hq, alpha, mode, xz_dim)).cpu().numpy()
            recs[dd].append((xz[:, 0].cpu().numpy(), S, y.cpu().numpy(), int(ng)))
    return recs


def _metric_at_lambda(recs_ds, lam):
    """R@k/Hit@k for one dataset's queries under S_final = z_dense + lam*S_fusion."""
    acc = {m: [] for m in METR}
    for zd, S, y, ng in recs_ds:
        ry = y[np.argsort(-(zd + lam * S))][:50].astype(np.uint8)
        q = _qmet(ry, ng)
        for m in METR:
            acc[m].append(100.0 * q[m])
    return {m: float(np.mean(acc[m])) for m in METR}


def _dense_anchor(fd_train_seeds, fd_eval_seeds, train_datasets, eval_dataset, grid):
    """Dense-anchored residual fusion S_final = z_dense + lam*S_fusion. lambda* selected ONLY on the train datasets'
    pools (eval_dataset strictly held out; no eval labels touched). lam=0 -> pure dense; lam->inf -> pure full_fd2."""
    nseed = len(fd_train_seeds); curve = {}
    for lam in grid:
        tr_seed_means, hp_r5 = [], []
        for si in range(nseed):
            tr_seed_means.append(float(np.mean([_metric_at_lambda(fd_train_seeds[si][ds], lam)["R@5"] for ds in train_datasets])))
            hp_r5.append(_metric_at_lambda(fd_eval_seeds[si][eval_dataset], lam)["R@5"])
        curve[lam] = {"train_mean_R@5": round(float(np.mean(tr_seed_means)), 2), "held_out_R@5": round(float(np.mean(hp_r5)), 2)}
    lam_star = max(grid, key=lambda L: curve[L]["train_mean_R@5"])       # SELECTED ON TRAIN ONLY
    hp = {m: [] for m in METR}; tr_by_ds = {ds: [] for ds in train_datasets}
    for si in range(nseed):
        hm = _metric_at_lambda(fd_eval_seeds[si][eval_dataset], lam_star)
        for m in METR:
            hp[m].append(hm[m])
        for ds in train_datasets:
            tr_by_ds[ds].append(_metric_at_lambda(fd_train_seeds[si][ds], lam_star)["R@5"])
    return {"lambda_grid": list(grid), "lambda_star": lam_star, "selection": "argmax mean train-dataset R@5 (held-out dataset never used)",
            "sweep": {str(L): curve[L] for L in grid},
            "held_out_at_lambda_star": {m: [round(float(np.mean(hp[m])), 2), round(float(np.std(hp[m])), 2)] for m in METR},
            "train_R@5_at_lambda_star": {ds: round(float(np.mean(tr_by_ds[ds])), 2) for ds in train_datasets}}


def _load(datasets, DEV):
    return {ds: torch.load(f"results/L2/relsig_feats_{ds}.pt", map_location=DEV) for ds in datasets}


def _run(datasets=("webqsp", "metaqa"), epochs=12, seeds=10, lam=1.0, tau_r=1.0, drop=0.2, hq_dim=32,
         modes=("flat", "flat_drop", "hybrid"),
         out_path="results/L2/kg_hybrid.json", preds_path="results/L2/kg_hybrid_preds.pt"):
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _load(datasets, DEV)
    allx = torch.cat([x for ds in datasets for x, _, _ in data[ds]["train"]], 0)
    mu = allx.mean(0).to(DEV); sd = allx.std(0).clamp(min=1e-6).to(DEV)
    uin = 2 * NE + 2
    TR = _prep(data, datasets, "train", mu, sd, DEV); TE = _prep(data, datasets, "test", mu, sd, DEV)
    xz_dim = TR[0][0].shape[1]
    frozen = {"experts": EXPERTS, "epochs": epochs, "seeds": seeds, "lam": lam, "tau_r": tau_r, "drop": drop,
              "hq_dim": hq_dim, "infonce_topM": 20, "lr": 1e-3, "optimizer": "adam",
              "norm": "joint mu/sd over pooled train", "metric": "recall/hit@k over frozen cached test pool",
              "datasets": list(datasets)}
    out = {"_config": frozen, "_note": "ONE joint checkpoint per mode, trained on mixed %s (no dataset ID), eval each." % "+".join(datasets)}
    preds = {}
    for mode in modes:
        seed_recs = [_eval(mode, *_train(mode, s, TR, xz_dim, uin, DEV, epochs, lam, tau_r, drop, hq_dim),
                           TE, datasets, xz_dim, hq_dim, DEV) for s in range(seeds)]
        agg, pr = _aggregate(seed_recs, datasets)
        out[mode] = agg; preds[mode] = pr
        log.info("[kg-hybrid] %-10s %s", mode, {ds: {"R@5": agg[ds]["R@5"], "R@20": agg[ds]["R@20"]} for ds in datasets})
        json.dump(out, open(out_path, "w"), indent=2)             # incremental: never lose a completed mode
        torch.save(preds, preds_path)
    log.info("-> %s  (+ per-question preds -> %s)", out_path, preds_path)
    return out


def run_transfer(train_datasets=("webqsp", "metaqa"), eval_dataset="2wiki_clean", mode="full_fd2",
                 epochs=12, seeds=5, lam=1.0, tau_r=1.0, drop=0.2, hq_dim=32,
                 out_path=None, preds_path=None):
    """Train `mode` on train_datasets, ZERO-SHOT eval on eval_dataset with the FROZEN train-set mu/sd."""
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = out_path or f"results/L2/{eval_dataset}_zero_shot_{mode}.json"
    preds_path = preds_path or f"results/L2/{eval_dataset}_zero_shot_{mode}_preds.pt"
    dtr = _load(train_datasets, DEV)
    allx = torch.cat([x for ds in train_datasets for x, _, _ in dtr[ds]["train"]], 0)
    mu = allx.mean(0).to(DEV); sd = allx.std(0).clamp(min=1e-6).to(DEV)     # FROZEN normalization (train only)
    uin = 2 * NE + 2
    TR = _prep(dtr, train_datasets, "train", mu, sd, DEV)
    TE_train = _prep(dtr, train_datasets, "test", mu, sd, DEV)              # reference: no-degradation check
    deval = _load([eval_dataset], DEV)
    TE_eval = _prep(deval, [eval_dataset], "test", mu, sd, DEV)             # eval-set test, normalized by FROZEN mu/sd
    xz_dim = TR[0][0].shape[1]
    eval_recs = []; train_recs = []; alpha_seeds = []; fd_eval_seeds = []; fd_train_seeds = []
    for s in range(seeds):
        qenc, cand = _train(mode, s, TR, xz_dim, uin, DEV, epochs, lam, tau_r, drop, hq_dim)
        eval_recs.append(_eval(mode, qenc, cand, TE_eval, [eval_dataset], xz_dim, hq_dim, DEV))
        train_recs.append(_eval(mode, qenc, cand, TE_train, train_datasets, xz_dim, hq_dim, DEV))
        fd_eval_seeds.append(_fusion_and_dense(mode, qenc, cand, TE_eval, [eval_dataset], xz_dim, hq_dim, DEV))
        fd_train_seeds.append(_fusion_and_dense(mode, qenc, cand, TE_train, train_datasets, xz_dim, hq_dim, DEV))
        if mode in ROUTER_MODES:
            alpha_seeds.append(_mean_alpha(qenc, TE_eval, DEV))
    agg_eval, pr_eval = _aggregate(eval_recs, [eval_dataset])
    agg_train, _ = _aggregate(train_recs, train_datasets)
    baselines = _signal_baselines(TE_eval, DEV)                             # dense-only + per-signal reference, same pool
    LGRID = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    anchor = _dense_anchor(fd_train_seeds, fd_eval_seeds, train_datasets, eval_dataset, LGRID)  # S_final = z_dense + lam*S_fusion
    mean_alpha = ({EXPERT_NAMES[i]: round(float(np.mean([a[i] for a in alpha_seeds])), 4) for i in range(NE)}
                  if alpha_seeds else None)
    out = {"_config": {"kind": "leave_one_out_zero_shot_transfer", "mode": mode, "train_datasets": list(train_datasets),
                       "eval_dataset": eval_dataset, "epochs": epochs, "seeds": seeds,
                       "note": "MLP RETRAINED on train_datasets (eval_dataset held out); NOT a previously-frozen anchor checkpoint",
                       "norm": "FROZEN mu/sd from train_datasets (not recomputed on eval)"},
           "zero_shot": agg_eval, "train_ref": agg_train,
           "dense_baseline": {eval_dataset: baselines["dense"]},           # dense-only reference on the SAME eval pool (a signal floor, NOT "old CRAG" -- compare the historical CRAG number separately)
           "signal_baselines": {eval_dataset: baselines},                  # every raw signal alone (note: splade may be a wiring/alignment failure on some substrates -> not a SPLADE claim)
           "dense_anchor": {eval_dataset: anchor},                         # S_final = z_dense + lam*S_fusion; lam* chosen on TRAIN only, held-out eval never tuned
           "router_mean_alpha": {eval_dataset: mean_alpha}}                # leave-eval-out transfer router weights: should suppress edge/path
    json.dump(out, open(out_path, "w"), indent=2)
    torch.save({eval_dataset: pr_eval}, preds_path)
    log.info("[kg-hybrid:transfer] ZERO-SHOT %s on %s: %s", mode, eval_dataset,
             {eval_dataset: {"R@5": agg_eval[eval_dataset]["R@5"], "R@20": agg_eval[eval_dataset]["R@20"]}})
    log.info("[kg-hybrid:transfer] dense baseline (same pool) %s: %s", eval_dataset,
             {m: baselines["dense"][m][0] for m in METR})
    log.info("[kg-hybrid:transfer] leave-%s-out %s transfer: %s", eval_dataset, mode,
             {m: agg_eval[eval_dataset][m][0] for m in METR})
    if mean_alpha is not None:
        log.info("[kg-hybrid:transfer] router mean-alpha (edge/path suppressed?): %s", mean_alpha)
    log.info("[kg-hybrid:transfer] dense-anchor lam*=%s (train-selected) -> held-out %s R@5=%s (dense=%s, %s=%s)",
             anchor["lambda_star"], eval_dataset, anchor["held_out_at_lambda_star"]["R@5"][0],
             baselines["dense"]["R@5"][0], mode, agg_eval[eval_dataset]["R@5"][0])
    log.info("[kg-hybrid:transfer] dense-anchor train R@5 @lam* (keep gains?): %s", anchor["train_R@5_at_lambda_star"])
    log.info("[kg-hybrid:transfer] train-ref (no-degradation): %s", {d: agg_train[d]["R@5"] for d in train_datasets})
    log.info("-> %s", out_path)
    return out


def _conf_stats_query(s, valid, K=5):
    """Dataset-independent confidence signature over an expert's VALID candidates only. Standardization, margin,
    softmax entropy and the top-K set are all computed on the valid subset; entropy is normalized by
    log(#valid) -- the number of candidates THIS expert can actually score for THIS query -- never the pool size
    and never the number of experts. Returns (stats, top-K GLOBAL indices among valid). An expert with no valid
    candidate is unavailable for the query -> zeroed stats and an empty top-K (recall 0 naturally)."""
    vidx = np.where(valid)[0]; nv = len(vidx)
    if nv == 0:
        return {"margin": 0.0, "entropy": 0.0, "pmax": 0.0, "max_raw": 0.0, "n_valid": 0}, np.array([], dtype=int)
    sv = s[vidx]
    ss = (sv - sv.mean()) / (sv.std() + 1e-9)                       # per-query standardize over VALID only
    local = np.argsort(-ss)
    margin = float(ss[local[0]] - ss[local[1]]) if nv > 1 else 0.0  # top1-top2 gap in std units, valid candidates only
    pp = np.exp(ss - ss.max()); pp = pp / pp.sum()
    ent = float(-(pp * np.log(pp + 1e-12)).sum() / np.log(nv)) if nv > 1 else 0.0   # normalize by log(#valid candidates)
    topk = vidx[local[:K]]                                          # GLOBAL indices of this expert's top-K valid candidates
    return {"margin": margin, "entropy": ent, "pmax": float(pp.max()), "max_raw": float(sv.max()), "n_valid": int(nv)}, topk


def diag_confidence(datasets, out_path="results/L2/router_confidence_diag.json", K=5):
    """Per dataset, per expert: mean confidence signature + rank-agreement with dense and with the other experts,
    alongside the expert's true R@5. The gating question: does OFFSET on a held-out dataset look detectably
    different (in router-VISIBLE features: entropy/margin/agreement) from OFFSET on the training datasets?
    If features look the same but R@5 collapses -> confidence features alone cannot predict OOD unreliability."""
    DEV = torch.device("cpu")
    data = _load(datasets, DEV)
    res = {}
    for ds in datasets:
        agg = {fn: {"entropy": [], "margin": [], "pmax": [], "max_raw": [], "avail": [], "n_valid": [],
                    "agree_dense@5": [], "agree_others@5": [], "R@5": [], "conf_top1_correct": []} for fn in FEAT_NAMES}
        for r in data[ds]["test"]:
            x, y, ng, av = _row_unpack(r)
            xn = x.numpy(); yn = y.numpy(); avn = av.numpy()
            golds = set(np.where(yn > 0)[0].tolist())
            top = {}; topord = {}; stt = {}
            for ci, fn in enumerate(FEAT_NAMES):                                          # top-K over VALID candidates only (k_eff)
                stt[fn], tk = _conf_stats_query(xn[:, ci], avn[:, ci] > 0.5, K)
                topord[fn] = tk; top[fn] = set(tk.tolist())
            dset = top["dense"]
            for ci, fn in enumerate(FEAT_NAMES):
                a = agg[fn]
                a["avail"].append(float(avn[:, ci].max()))         # stored mask, not a score threshold
                for kk in ("entropy", "margin", "pmax", "max_raw", "n_valid"):
                    a[kk].append(stt[fn][kk])
                # agreement normalized by max achievable overlap of the two VALID top-K sets (k_eff), not a fixed K
                a["agree_dense@5"].append(len(top[fn] & dset) / max(min(len(top[fn]), len(dset)), 1))
                others = [top[o] for o in FEAT_NAMES if o != fn]                          # mean valid-top-K overlap with each other expert
                a["agree_others@5"].append(float(np.mean([len(top[fn] & o) / max(min(len(top[fn]), len(o)), 1) for o in others])))
                a["R@5"].append(len(golds & top[fn]) / ng)                                # golds among this expert's VALID top-K
                a["conf_top1_correct"].append(1.0 if (len(topord[fn]) > 0 and int(topord[fn][0]) in golds) else 0.0)   # expert's #1 valid candidate gold?
        res[ds] = {fn: {k: round(float(np.mean(v)), 4) for k, v in a.items()} for fn, a in agg.items()}
    import os as _os; _os.makedirs("results/L2", exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=2)
    # focused console summary for OFFSET (the failing expert) across datasets
    log.info("[diag] OFFSET confidence signature by dataset (entropy / margin / agree_dense@5 / agree_others@5 / R@5 / conf_top1_correct):")
    for ds in datasets:
        o = res[ds]["offset"]
        log.info("  %-16s ent=%.3f margin=%.3f agD=%.3f agO=%.3f R@5=%.3f top1ok=%.3f",
                 ds, o["entropy"], o["margin"], o["agree_dense@5"], o["agree_others@5"], o["R@5"], o["conf_top1_correct"])
    log.info("-> %s", out_path)
    return res


# ----------------------------------------------------------------------------------------------------------------
# PHASE 1 -- substrate audit. Validate every expert independently, per dataset, on the SAVED cached substrate
# (relsig_feats_{ds}.pt: raw x, y, ng, mu, sd, feat_names). No routing, no fusion. Answers: is each expert's
# candidate alignment/ordering sane, are missing-value sentinels handled, is normalization safe, what is each
# expert's STANDALONE R@k/Hit@k, and what is the graph/edge/path/splade coverage. Flags the known defects:
# constant/absent columns (e.g. SPLADE all-zero on Hotpot) and sentinels that leak as real scores after norm.
_MISSING_SENTINEL = {2: 0.0, 3: 0.0, 5: -1.0, 6: -1.0}   # prototype/splade: 0=absent; relation/path: -1.0=absent (dense/offset/graph always present)
# What a column's "availability" actually MEANS -- so the audit never conflates "a score exists" with "the
# candidate is structurally reachable". Only relation/path carry genuine per-candidate graph provenance;
# dense/offset/graph always have a score (an always-on prior -> availability=1.0 says nothing about reachability).
_AVAIL_SEMANTICS = {0: "always_present_prior", 1: "always_present_prior", 2: "entity_presence",
                    3: "dataset_level_splade_presence", 4: "always_present_prior",
                    5: "per_candidate_graph_provenance", 6: "per_candidate_graph_provenance"}


def _valid_mask(col, c):
    """Per-candidate validity for expert column c (raw scores). dense/offset/graph always valid;
    relation/path valid where raw>-0.5 (missing sentinel -1.0); splade/prototype valid where !=0 (0=absent)."""
    sent = _MISSING_SENTINEL.get(c)
    if sent is None:
        return np.ones(len(col), bool)
    return col > -0.5 if sent == -1.0 else col != 0.0


def _row_unpack(r):
    """(x, y, ng, avail) from a substrate row. New masked substrate stores a per-candidate availability
    mask as the 4th element -- the AUTHORITATIVE availability, used verbatim (never re-thresholded). Legacy
    3-tuple rows have no mask: derive it once from the raw missing sentinel so old dumps still load."""
    if len(r) >= 4:
        return r[0], r[1], r[2], r[3]
    x, y, ng = r
    a = torch.ones_like(x)
    for c in range(x.shape[1]):
        sent = _MISSING_SENTINEL.get(c)
        if sent == -1.0:
            a[:, c] = (x[:, c] > -0.5).float()
        elif sent == 0.0:
            a[:, c] = (x[:, c] != 0.0).float()
    return x, y, ng, a


def diag_substrate(datasets, out_path="results/L2/substrate_audit.json"):
    """Phase-1 audit distinguishing candidate coverage from QUERY availability and GOLD visibility. A structural
    expert can be valid on ~0.1% of *candidates* yet available on ~100% of *queries* (1 useful edge per 1000-cand
    pool) and cover most golds -- that is sparse-by-design, NOT broken. Genuine breakage = constant/absent column
    (e.g. SPLADE all-zero). Reports per expert: (1) candidate_valid_fraction, (2) query_available_fraction,
    (3) mean_valid_candidates_per_available_query, (4) gold_valid_fraction (golds-in-pool with a real score),
    (5) R@k on queries where the expert is available, (6) unconditional R@k."""
    DEV = torch.device("cpu")
    res = {}
    for ds in datasets:
        d = torch.load(f"results/L2/relsig_feats_{ds}.pt", map_location=DEV)
        mu = d["mu"].numpy(); sd = d["sd"].numpy(); FE = d["feat_names"]; te = d["test"]; ncol = len(FE)
        cvf = {c: [] for c in range(ncol)}; qav = {c: 0 for c in range(ncol)}; vpa = {c: [] for c in range(ncol)}
        gvalid = {c: 0 for c in range(ncol)}; gtot = 0
        r_cond = {c: {m: [] for m in METR} for c in range(ncol)}; r_unc = {c: {m: [] for m in METR} for c in range(ncol)}
        poolsz = []; pool_cov = []
        stored_mask = len(te[0]) >= 4 if te else False           # new masked substrate carries the authoritative avail mask
        for r in te:
            x, y, ng, a = _row_unpack(r)
            xn = x.numpy(); yn = y.numpy(); n = len(yn); gidx = np.where(yn > 0)[0]; an = a.numpy()
            poolsz.append(n); pool_cov.append(float(yn.sum()) / ng if ng else 0.0); gtot += len(gidx)
            for c in range(ncol):
                col = xn[:, c]
                valid = an[:, c] > 0.5 if stored_mask else _valid_mask(col, c)   # trust the stored mask, never re-threshold
                nv = int(valid.sum())
                cvf[c].append(nv / n)
                gvalid[c] += int(valid[gidx].sum())
                colr = np.where(valid, col, -np.inf)                    # x_rank semantics: a missing candidate can never rank;
                q = _qmet(yn[np.argsort(-colr)][:50].astype(np.uint8), ng)  # a gold with no valid score for this expert -> recall 0 naturally
                for m in METR:
                    r_unc[c][m].append(100.0 * q[m])
                if nv > 0:
                    qav[c] += 1; vpa[c].append(nv)
                    for m in METR:
                        r_cond[c][m].append(100.0 * q[m])
        allx = torch.cat([r[0] for r in te], 0).numpy(); nq = len(te)
        experts = {}
        for c in range(ncol):
            allc = allx[:, c]; sent = _MISSING_SENTINEL.get(c)
            norm_sent = float((sent - mu[c]) / sd[c]) if sent is not None else None
            experts[FE[c]] = {
                "availability_semantics": _AVAIL_SEMANTICS.get(c, "unknown"),   # "a score exists" != "reachable"; only relation/path = real provenance
                "candidate_valid_fraction": round(float(np.mean(cvf[c])), 4),
                "query_available_fraction": round(qav[c] / nq, 4),
                "mean_valid_candidates_per_available_query": round(float(np.mean(vpa[c])), 2) if vpa[c] else 0.0,
                "gold_valid_fraction": round(gvalid[c] / gtot, 4) if gtot else None,
                "R@k_available_queries": {m: round(float(np.mean(r_cond[c][m])), 2) if r_cond[c][m] else None for m in METR},
                "R@k_unconditional": {m: round(float(np.mean(r_unc[c][m])), 2) for m in METR},
                "constant_column": bool(allc.std() < 1e-9),
                "raw_min": round(float(allc.min()), 4), "raw_max": round(float(allc.max()), 4),
                "raw_std": round(float(allc.std()), 4),
                "missing_sentinel": sent, "frac_at_sentinel": round(float((allc == sent).mean()), 4) if sent is not None else None,
                "normalized_sentinel": round(norm_sent, 4) if norm_sent is not None else None,
                # design flaw ONLY where a consumer thresholds the NORMALIZED score for availability; all load-bearing
                # paths (_prep/router/diag) use the raw mask, so this is a latent risk, not a proven-live defect.
                "norm_sentinel_above_threshold": bool(norm_sent is not None and norm_sent > -0.5)}
        res[ds] = {"n_test_queries": nq, "pool_size_mean": round(float(np.mean(poolsz)), 1),
                   "pool_gold_coverage": round(float(np.mean(pool_cov)), 4),
                   "masked_substrate": bool(d.get("masked", False)),          # present-only norm + stored avail mask
                   "splade_available": d.get("splade_available"),             # dataset-level: real for all cands, or genuinely absent
                   "experts": experts}
        log.info("=== %s  (n=%d, pool~%d, gold-in-pool=%.3f) ===", ds, nq, np.mean(poolsz), np.mean(pool_cov))
        log.info("  %-13s %8s %8s %9s %8s | %8s %8s  %s", "expert", "cand_val", "q_avail", "val/avail_q",
                 "gold_val", "R@5|avail", "R@5|unc", "const")
        for c in range(ncol):
            e = experts[FE[c]]
            log.info("  %-13s %8.4f %8.4f %9.1f %8.4f | %8s %8.2f  %s", FE[c], e["candidate_valid_fraction"],
                     e["query_available_fraction"], e["mean_valid_candidates_per_available_query"], e["gold_valid_fraction"],
                     e["R@k_available_queries"]["R@5"], e["R@k_unconditional"]["R@5"], e["constant_column"])
    import os as _os; _os.makedirs("results/L2", exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=2)
    log.info("-> %s", out_path)
    return res


# ----------------------------------------------------------------------------------------------------------------
# Confidence-aware router (dataset-independent inputs) + synthetic confidently-wrong corruption.
# Mechanism = full_fd2 (hybrid MLP advised by KL-trained soft router, consistency-preserving co-dropout), but:
#   (a) router INPUT is _qstate_conf (per-query standardized margin/entropy + cross-expert agreement) -- no dataset ID;
#   (b) OPTIONAL training augmentation corrupts one expert to look confident-but-wrong so the teacher pi* marks it
#       unreliable -> the router must learn confidence(+consensus) != reliability.
# True LODO here is the EVAL protocol: train on the 4 datasets, Hotpot completely unseen. No epoch-rotation trick.
# Corruption hyperparams (corrupt_p/sharpen/n_swap) and the fallback beta are FIXED a priori or selected on the
# TRAIN datasets only -- Hotpot labels are never touched during model/hyperparameter selection.
ARM2CORR = {"conf_only": "none", "conf_easy": "easy", "conf_plausible": "plausible"}


def _train_conf(seed, TR, xz_dim, DEV, epochs=12, lam=1.0, tau_r=1.0, drop=0.2, hq_dim=32,
                corrupt="none", corrupt_p=0.3, sharpen=2.0, n_swap=3):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=DEV); gen.manual_seed(seed + 20260821)
    qenc = QEnc(UIN_CONF, hq_dim).to(DEV); cand = Cand(xz_dim + NE + NE + hq_dim).to(DEV)
    opt = torch.optim.Adam(list(cand.parameters()) + list(qenc.parameters()), lr=1e-3)
    weights = np.array([w for *_, w, _ in TR]); order0 = np.arange(len(TR))
    for ep in range(epochs):
        order = np.random.choice(order0, size=len(TR), replace=True, p=weights / weights.sum())
        for j in order:
            xz, y, ng, avail, w, ds = TR[j]
            if corrupt != "none" and np.random.rand() < corrupt_p:                    # confident-but-wrong expert
                # corrupt only an AVAILABLE expert: dense/offset/splade always on; relation/path from the raw-derived avail mask
                cols = [k for k in range(NE) if k not in (IDX_RELATION, IDX_PATH)] \
                    + [k for k, a in ((IDX_RELATION, avail[0]), (IDX_PATH, avail[1])) if float(a) > 0.5]
                dc = cols[np.random.randint(len(cols))]
                xz = (_corrupt_easy(xz, dc, sharpen, gen) if corrupt == "easy"
                      else _corrupt_plausible(xz, y, dc, n_swap, sharpen, gen))
            dcol = np.random.randint(NE) if np.random.rand() < drop else -1           # full_fd2 co-dropout
            hq, alogit = qenc(torch.cat([_qstate_conf(xz), avail])); alpha = F.softmax(alogit, 0)
            xin = _cand_input(xz, hq, alpha, "full_fd2", xz_dim)
            if dcol >= 0:
                xin = xin.clone(); xin[:, EXPERTS[dcol]] = 0.0; xin[:, xz_dim + NE + dcol] = 0.0
            S = cand(xin)
            loss = -(F.log_softmax(S, 0) * (y / y.sum())).sum()
            with torch.no_grad():                                                     # teacher sees the corrupted column
                Le = torch.stack([_expert_loss(xz[:, c], y) for c in EXPERTS])
                pi = F.softmax(-Le / tau_r, 0)
            loss = loss + lam * (pi * ((pi + 1e-9).log() - F.log_softmax(alogit, 0))).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return qenc, cand


def _capture_conf(qenc, cand, TE, datasets, xz_dim, hq_dim, DEV):
    """Per query: (z_dense col-0, S_fusion, router_conf, gold mask, ng). router_conf = 1 - normalized alpha-entropy
    (high -> router committed to one expert; low -> unsure). Enables the low-confidence dense fallback sweep."""
    recs = {ds: [] for ds in datasets}
    with torch.no_grad():
        for xz, y, ng, avail, w, dd in TE:
            if dd not in recs:
                continue
            hq, alogit = qenc(torch.cat([_qstate_conf(xz), avail])); alpha = F.softmax(alogit, 0)
            S = cand(_cand_input(xz, hq, alpha, "full_fd2", xz_dim)).cpu().numpy()
            rconf = float(1.0 - (-(alpha * (alpha + 1e-9).log()).sum() / np.log(NE)).item())
            recs[dd].append((xz[:, 0].cpu().numpy(), S, rconf, y.cpu().numpy(), int(ng)))
    return recs


def _mean_alpha_conf(qenc, TE, DEV):
    accs = []
    with torch.no_grad():
        for xz, y, ng, avail, w, dd in TE:
            _, alogit = qenc(torch.cat([_qstate_conf(xz), avail]))
            accs.append(F.softmax(alogit, 0).cpu().numpy())
    return np.mean(np.stack(accs), 0)


def _agg_from_capture(seed_caps, datasets):                       # rank by S_fusion -> top-50 gold mask -> _aggregate
    seed_recs = []
    for cap in seed_caps:
        rec = {ds: [] for ds in datasets}
        for ds in datasets:
            for zd, S, rconf, y, ng in cap[ds]:
                rec[ds].append((y[np.argsort(-S)][:50].astype(np.uint8), ng))
        seed_recs.append(rec)
    return _aggregate(seed_recs, datasets)


def _metric_fallback(cap_ds, beta):
    """S_final = z(S_fusion) + beta*(1-router_conf)*z(dense); per-query standardized so scales are comparable.
    beta=0 -> pure fusion. Dense weight grows only where the router is UNSURE -> a low-confidence dense fallback."""
    acc = {m: [] for m in METR}
    for zd, S, rconf, y, ng in cap_ds:
        Sz = (S - S.mean()) / (S.std() + 1e-9)
        Dz = (zd - zd.mean()) / (zd.std() + 1e-9)
        ry = y[np.argsort(-(Sz + beta * (1.0 - rconf) * Dz))][:50].astype(np.uint8)
        q = _qmet(ry, ng)
        for m in METR:
            acc[m].append(100.0 * q[m])
    return {m: float(np.mean(acc[m])) for m in METR}


def _conf_fallback(cap_train_seeds, cap_eval_seeds, train_datasets, eval_dataset, grid):
    nseed = len(cap_train_seeds); curve = {}
    for b in grid:
        trm, hp = [], []
        for si in range(nseed):
            trm.append(float(np.mean([_metric_fallback(cap_train_seeds[si][ds], b)["R@5"] for ds in train_datasets])))
            hp.append(_metric_fallback(cap_eval_seeds[si][eval_dataset], b)["R@5"])
        curve[b] = {"train_mean_R@5": round(float(np.mean(trm)), 2), "held_out_R@5": round(float(np.mean(hp)), 2)}
    b_star = max(grid, key=lambda B: curve[B]["train_mean_R@5"])              # SELECTED ON TRAIN ONLY
    hp = {m: [] for m in METR}; trd = {ds: [] for ds in train_datasets}
    for si in range(nseed):
        hm = _metric_fallback(cap_eval_seeds[si][eval_dataset], b_star)
        for m in METR:
            hp[m].append(hm[m])
        for ds in train_datasets:
            trd[ds].append(_metric_fallback(cap_train_seeds[si][ds], b_star)["R@5"])
    return {"beta_grid": list(grid), "beta_star": b_star, "selection": "argmax mean train-dataset R@5 (held-out never used)",
            "sweep": {str(B): curve[B] for B in grid},
            "held_out_at_beta_star": {m: [round(float(np.mean(hp[m])), 2), round(float(np.std(hp[m])), 2)] for m in METR},
            "train_R@5_at_beta_star": {ds: round(float(np.mean(trd[ds])), 2) for ds in train_datasets}}


def run_conf_transfer(train_datasets=("webqsp", "metaqa", "2wiki_clean", "musique_clean"),
                      eval_dataset="hotpotqa_clean", arms=("conf_only", "conf_easy", "conf_plausible"),
                      epochs=12, seeds=5, lam=1.0, tau_r=1.0, drop=0.2, hq_dim=32,
                      corrupt_p=0.3, sharpen=2.0, n_swap=3, out_path=None):
    """Confidence-aware router ablation, Hotpot strictly held out (true LODO eval). Arms: conf_only,
    conf_easy (shuffle+sharpen corruption), conf_plausible (consensus-preserving corruption). Each arm also
    reports an OPTIONAL low-confidence dense fallback (beta selected on TRAIN only). Key question: does plausible
    corruption teach 'confident and consensus-looking can still be wrong' -> recover Hotpot toward dense ~70
    while preserving the 4-dataset gains, with NO Hotpot-specific rule and NO Hotpot label touched in selection."""
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = out_path or f"results/L2/{eval_dataset}_conf_router.json"
    dtr = _load(train_datasets, DEV)
    allx = torch.cat([x for ds in train_datasets for x, _, _ in dtr[ds]["train"]], 0)
    mu = allx.mean(0).to(DEV); sd = allx.std(0).clamp(min=1e-6).to(DEV)      # FROZEN normalization (train only)
    TR = _prep(dtr, train_datasets, "train", mu, sd, DEV)
    TE_train = _prep(dtr, train_datasets, "test", mu, sd, DEV)
    deval = _load([eval_dataset], DEV)
    TE_eval = _prep(deval, [eval_dataset], "test", mu, sd, DEV)
    xz_dim = TR[0][0].shape[1]
    baselines = _signal_baselines(TE_eval, DEV)
    BGRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    out = {"_config": {"kind": "confidence_router_leave_one_out_transfer", "train_datasets": list(train_datasets),
                       "eval_dataset": eval_dataset, "arms": list(arms), "epochs": epochs, "seeds": seeds,
                       "router_input": "dataset-independent: per-query standardized margin/entropy + cross-expert agreement + avail",
                       "corruption": {"corrupt_p": corrupt_p, "sharpen": sharpen, "n_swap": n_swap,
                                      "note": "fixed a priori, NOT tuned on Hotpot"},
                       "selection": "corruption fixed a priori; fallback beta chosen on train datasets only; Hotpot labels untouched",
                       "norm": "FROZEN mu/sd from train_datasets"},
           "dense_baseline": {eval_dataset: baselines["dense"]}}
    for arm in arms:
        corr = ARM2CORR[arm]
        cap_eval, cap_train, alpha_seeds = [], [], []
        for s in range(seeds):
            qenc, cand = _train_conf(s, TR, xz_dim, DEV, epochs, lam, tau_r, drop, hq_dim,
                                     corr, corrupt_p, sharpen, n_swap)
            cap_eval.append(_capture_conf(qenc, cand, TE_eval, [eval_dataset], xz_dim, hq_dim, DEV))
            cap_train.append(_capture_conf(qenc, cand, TE_train, train_datasets, xz_dim, hq_dim, DEV))
            alpha_seeds.append(_mean_alpha_conf(qenc, TE_eval, DEV))
        agg_eval, _ = _agg_from_capture(cap_eval, [eval_dataset])
        agg_train, _ = _agg_from_capture(cap_train, train_datasets)
        fb = _conf_fallback(cap_train, cap_eval, train_datasets, eval_dataset, BGRID)
        mean_alpha = {EXPERT_NAMES[i]: round(float(np.mean([a[i] for a in alpha_seeds])), 4) for i in range(NE)}
        out[arm] = {"zero_shot": agg_eval, "train_ref": agg_train,
                    "router_mean_alpha": {eval_dataset: mean_alpha},
                    "low_conf_dense_fallback": {eval_dataset: fb}}
        json.dump(out, open(out_path, "w"), indent=2)                         # incremental: never lose an arm
        log.info("[conf] %-14s Hotpot R@5=%s (dense=%s) | train R@5=%s | alpha=%s | fallback beta*=%s -> R@5=%s",
                 arm, agg_eval[eval_dataset]["R@5"][0], baselines["dense"]["R@5"][0],
                 {d: agg_train[d]["R@5"][0] for d in train_datasets}, mean_alpha,
                 fb["beta_star"], fb["held_out_at_beta_star"]["R@5"][0])
    log.info("-> %s", out_path)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Centralized KL-supervised CONTROLLER (query + graph + expert state), not a confidence-only gate.
# alpha(q,G) = softmax(controller(state)); state is dataset-INDEPENDENT: per-expert standardized margin/entropy +
# cross-expert agreement + graph reachability/coverage + edge/path strength + candidate-source composition +
# dense-dominance (semantic-retrieval-strength proxy) + availability. NO dataset ID and (deliberately) NO raw
# query embedding -- a raw embedding is both absent from the cached substrate AND a dataset-ID leakage backdoor;
# dense-dominance/margin stand in for "semantic retrieval strength". Teacher pi* = softmax(-L_e/tau), L_KL = KL(pi*||alpha).
# alpha is an ADVISOR (features into the expressive candidate MLP), NEVER a constrained convex final scorer.
CENTRAL_MODES = ("central_kl", "central_kl_advisor")


def _qstate_central(xz, mu, sd, K=CONF_K):
    """Dataset-independent QUERY+GRAPH+EXPERT state for the centralized controller (STATE_CENTRAL,). Reachability
    and edge/path strength are computed on RAW columns (raw = xz*sd+mu) because the missing sentinel (-1.0) shifts
    above the -0.5 test under normalization -- on xz every candidate would falsely read as reachable."""
    n = xz.shape[0]; ln = float(np.log(max(n, 2)))
    sub = xz[:, EXPERTS]                                           # (n, NE)
    z = (sub - sub.mean(0)) / sub.std(0).clamp(min=1e-6)
    p = F.softmax(z, 0); ent = -(p * (p + 1e-9).log()).sum(0) / ln
    kk = min(2, n); tp = torch.topk(z, kk, dim=0).values
    marg = (tp[0] - tp[-1]) if n > 1 else torch.zeros(NE, device=xz.device)
    Kk = min(K, n); tk = torch.topk(sub, Kk, dim=0).indices        # (Kk, NE)
    ov = torch.zeros(NE, NE, device=xz.device)
    for i in range(NE):
        for j in range(NE):
            if i != j:
                ov[i, j] = (tk[:, i].unsqueeze(1) == tk[:, j].unsqueeze(0)).any(1).float().sum() / Kk
    agree = ov.sum(1) / (NE - 1)                                   # (NE,)
    raw = xz * sd + mu                                             # de-normalize to test the raw -1.0 sentinel
    rel = raw[:, 5]; pth = raw[:, 6]; gph = raw[:, 4]
    rel_m = rel > -0.5; pth_m = pth > -0.5                         # present (raw>-0.5) vs missing (raw==-1.0)
    cov = torch.stack([rel_m.float().mean(), pth_m.float().mean(), (gph > -0.5).float().mean()])   # reachable-candidate fractions
    def _str(col, m):
        return torch.stack([col[m].mean(), col[m].max()]) if bool(m.any()) else torch.zeros(2, device=xz.device)
    estr = _str(rel, rel_m); pstr = _str(pth, pth_m)              # edge / path strength (raw cosine) over reachable candidates
    uni = torch.unique(tk.reshape(-1)); win = z[uni].argmax(1)     # candidate-source composition over the union top-K set
    comp = torch.bincount(win, minlength=NE).float(); comp = comp / comp.sum().clamp(min=1)
    t1 = tp[0]; dens_dom = (t1[0] - (t1.sum() - t1[0]) / (NE - 1)).reshape(1)   # dense top1 vs mean structural top1 (std units)
    return torch.cat([marg, ent, agree, cov, estr, pstr, comp, dens_dom])       # (STATE_CENTRAL,)


def _reach_fracs(xz, mu, sd):
    """Raw-space edge/path reachable fractions for regime bucketing (same de-normalization as _qstate_central)."""
    raw = xz * sd + mu
    return float((raw[:, 5] > -0.5).float().mean()), float((raw[:, 6] > -0.5).float().mean())


def _qstate_router(mode, xz, mu, sd):
    if mode in CENTRAL_MODES:
        return _qstate_central(xz, mu, sd)
    if mode.startswith("conf"):
        return _qstate_conf(xz)
    return _qstate(xz)                                             # full_fd2 legacy (dataset-dependent raw entropy/margin)


def _uin_router(mode):
    return UIN_CENTRAL if mode in CENTRAL_MODES else (UIN_CONF if mode.startswith("conf") else 2 * NE + 2)


def _cand_input_uni(mode, xz, hq, alpha, xz_dim, mu, sd):
    x = _cand_input(xz, hq, alpha, "full_fd2", xz_dim)            # [s_d, alpha, alpha*s_d, hq]
    if mode == "central_kl_advisor":                             # also feed raw query/graph state to the MLP
        st = _qstate_central(xz, mu, sd)
        x = torch.cat([x, st[None, :].expand(xz.shape[0], -1)], 1)
    return x


def _train_uni(mode, seed, TR, xz_dim, mu, sd, DEV, epochs=12, lam=1.0, tau_r=1.0, drop=0.2, hq_dim=32):
    """Unified full_fd2-style trainer (co-dropout + KL advisor) for any router input; used by all ablation arms."""
    torch.manual_seed(seed); np.random.seed(seed)
    qenc = QEnc(_uin_router(mode), hq_dim).to(DEV)
    nin = xz_dim + NE + NE + hq_dim + (STATE_CENTRAL if mode == "central_kl_advisor" else 0)
    cand = Cand(nin).to(DEV)
    opt = torch.optim.Adam(list(cand.parameters()) + list(qenc.parameters()), lr=1e-3)
    weights = np.array([w for *_, w, _ in TR]); order0 = np.arange(len(TR))
    for ep in range(epochs):
        order = np.random.choice(order0, size=len(TR), replace=True, p=weights / weights.sum())
        for j in order:
            xz, y, ng, avail, w, ds = TR[j]
            dcol = np.random.randint(NE) if np.random.rand() < drop else -1
            hq, alogit = qenc(torch.cat([_qstate_router(mode, xz, mu, sd), avail])); alpha = F.softmax(alogit, 0)
            xin = _cand_input_uni(mode, xz, hq, alpha, xz_dim, mu, sd)
            if dcol >= 0:
                xin = xin.clone(); xin[:, EXPERTS[dcol]] = 0.0; xin[:, xz_dim + NE + dcol] = 0.0
            S = cand(xin)
            loss = -(F.log_softmax(S, 0) * (y / y.sum())).sum()
            with torch.no_grad():
                Le = torch.stack([_expert_loss(xz[:, c], y) for c in EXPERTS]); pi = F.softmax(-Le / tau_r, 0)
            loss = loss + lam * (pi * ((pi + 1e-9).log() - F.log_softmax(alogit, 0))).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return qenc, cand


def _capture_uni(mode, qenc, cand, TE, datasets, xz_dim, mu, sd, hq_dim, DEV):
    """fb[ds]=list of (z_dense,S_fusion,router_conf,gold,ng) for fallback; reg[ds]=list of (alpha, frac_edge, frac_path)."""
    fb = {ds: [] for ds in datasets}; reg = {ds: [] for ds in datasets}
    with torch.no_grad():
        for xz, y, ng, avail, w, dd in TE:
            if dd not in fb:
                continue
            hq, alogit = qenc(torch.cat([_qstate_router(mode, xz, mu, sd), avail])); alpha = F.softmax(alogit, 0)
            S = cand(_cand_input_uni(mode, xz, hq, alpha, xz_dim, mu, sd)).cpu().numpy()
            rconf = float(1.0 - (-(alpha * (alpha + 1e-9).log()).sum() / np.log(NE)).item())
            fb[dd].append((xz[:, 0].cpu().numpy(), S, rconf, y.cpu().numpy(), int(ng)))
            fe, fp = _reach_fracs(xz, mu, sd)
            reg[dd].append((alpha.cpu().numpy(), fe, fp))
    return fb, reg


def _alpha_regime(reg_seeds, ds):
    """Mean alpha overall and split by query regime: structural-evidence (edge/path reachable) vs semantic-only."""
    rows = [r for seed in reg_seeds for r in seed[ds]]
    def mean_a(pred):
        A = [a for a, fe, fp in rows if pred(fe, fp)]
        return ({EXPERT_NAMES[i]: round(float(np.mean([a[i] for a in A])), 4) for i in range(NE)}, len(A)) if A else (None, 0)
    ov, _ = mean_a(lambda fe, fp: True)
    st, ns = mean_a(lambda fe, fp: (fe + fp) > 0.05)
    sm, nm = mean_a(lambda fe, fp: (fe + fp) <= 0.05)
    return {"overall": ov, "structural_evidence": {"mean_alpha": st, "n": ns},
            "semantic_only": {"mean_alpha": sm, "n": nm}}


def run_central_transfer(train_datasets=("webqsp", "metaqa", "2wiki_clean", "musique_clean"),
                         eval_dataset="hotpotqa_clean",
                         arms=("conf_only", "full_fd2", "central_kl", "central_kl_advisor"),
                         epochs=12, seeds=5, lam=1.0, tau_r=1.0, drop=0.2, hq_dim=32, out_path=None):
    """Centralized-controller ablation, Hotpot strictly held out. Arms: conf_only (confidence-only ref),
    full_fd2 (dataset-dependent router ref), central_kl (rich query+graph+expert state controller),
    central_kl_advisor (also feeds raw graph/query state to the candidate MLP). Reports Hotpot R@k/Hit@k,
    dense floor, train-suite R@5, mean alpha, alpha by regime, and the train-selected low-conf dense fallback."""
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = out_path or f"results/L2/{eval_dataset}_central_router.json"
    dtr = _load(train_datasets, DEV)
    allx = torch.cat([x for ds in train_datasets for x, _, _ in dtr[ds]["train"]], 0)
    mu = allx.mean(0).to(DEV); sd = allx.std(0).clamp(min=1e-6).to(DEV)
    TR = _prep(dtr, train_datasets, "train", mu, sd, DEV)
    TE_train = _prep(dtr, train_datasets, "test", mu, sd, DEV)
    deval = _load([eval_dataset], DEV)
    TE_eval = _prep(deval, [eval_dataset], "test", mu, sd, DEV)
    xz_dim = TR[0][0].shape[1]
    baselines = _signal_baselines(TE_eval, DEV)
    BGRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    out = {"_config": {"kind": "centralized_controller_leave_one_out_transfer", "train_datasets": list(train_datasets),
                       "eval_dataset": eval_dataset, "arms": list(arms), "epochs": epochs, "seeds": seeds,
                       "controller_state": "dataset-independent: per-expert std margin/entropy + cross-expert agreement + "
                                           "graph reachability/coverage + edge/path strength + candidate-source composition + "
                                           "dense-dominance (semantic-strength proxy) + availability",
                       "omitted": "raw query embedding (absent from cached substrate AND a dataset-ID leakage backdoor); "
                                  "dense-dominance/margin used as the dataset-independent semantic-retrieval-strength signal",
                       "advisor": "alpha fed to expressive candidate MLP as features (NOT a convex final scorer)",
                       "selection": "fallback beta chosen on train datasets only; Hotpot labels untouched",
                       "norm": "FROZEN mu/sd from train_datasets"},
           "dense_baseline": {eval_dataset: baselines["dense"]}}
    for arm in arms:
        cap_e, cap_t, reg_e = [], [], []
        for s in range(seeds):
            qenc, cand = _train_uni(arm, s, TR, xz_dim, mu, sd, DEV, epochs, lam, tau_r, drop, hq_dim)
            fe, re_ = _capture_uni(arm, qenc, cand, TE_eval, [eval_dataset], xz_dim, mu, sd, hq_dim, DEV)
            ft, _rt = _capture_uni(arm, qenc, cand, TE_train, train_datasets, xz_dim, mu, sd, hq_dim, DEV)
            cap_e.append(fe); cap_t.append(ft); reg_e.append(re_)
        agg_e, _ = _agg_from_capture(cap_e, [eval_dataset])
        agg_t, _ = _agg_from_capture(cap_t, train_datasets)
        fb = _conf_fallback(cap_t, cap_e, train_datasets, eval_dataset, BGRID)
        regime = _alpha_regime(reg_e, eval_dataset)
        out[arm] = {"zero_shot": agg_e, "train_ref": agg_t,
                    "router_mean_alpha": {eval_dataset: regime["overall"]},
                    "alpha_by_regime": {eval_dataset: {k: regime[k] for k in ("structural_evidence", "semantic_only")}},
                    "low_conf_dense_fallback": {eval_dataset: fb}}
        json.dump(out, open(out_path, "w"), indent=2)                         # incremental: never lose an arm
        log.info("[central] %-18s Hotpot R@5=%s (dense=%s) | train R@5=%s | alpha=%s | fallback beta*=%s -> R@5=%s",
                 arm, agg_e[eval_dataset]["R@5"][0], baselines["dense"]["R@5"][0],
                 {d: agg_t[d]["R@5"][0] for d in train_datasets}, regime["overall"],
                 fb["beta_star"], fb["held_out_at_beta_star"]["R@5"][0])
    log.info("-> %s", out_path)
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["webqsp", "metaqa"])
    p.add_argument("--modes", nargs="+", default=["flat", "flat_drop", "hybrid"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="results/L2/kg_hybrid.json")
    p.add_argument("--preds", default="results/L2/kg_hybrid_preds.pt")
    # transfer (zero-shot) mode
    p.add_argument("--train-datasets", nargs="+", default=None)
    p.add_argument("--eval-dataset", default=None)
    p.add_argument("--transfer-mode", default="full_fd2")
    p.add_argument("--diag-confidence", action="store_true")     # per-expert confidence signature by dataset (OOD gating diagnostic)
    p.add_argument("--diag-substrate", action="store_true")       # Phase 1: per-dataset per-expert standalone metrics + coverage/sentinel audit
    p.add_argument("--conf-router", action="store_true")          # confidence-aware router ablation (conf_only/easy/plausible)
    p.add_argument("--central-router", action="store_true")        # centralized KL controller ablation (conf_only/full_fd2/central_kl/central_kl_advisor)
    p.add_argument("--arms", nargs="+", default=["conf_only", "conf_easy", "conf_plausible"])
    p.add_argument("--corrupt-p", type=float, default=0.3)
    p.add_argument("--sharpen", type=float, default=2.0)
    p.add_argument("--n-swap", type=int, default=3)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if a.diag_substrate:
        diag_substrate(tuple(a.datasets), out_path=a.out if a.out != "results/L2/kg_hybrid.json" else "results/L2/substrate_audit.json")
    elif a.diag_confidence:
        diag_confidence(tuple(a.datasets), out_path=a.out if a.out != "results/L2/kg_hybrid.json" else "results/L2/router_confidence_diag.json")
    elif a.conf_router:
        run_conf_transfer(train_datasets=tuple(a.train_datasets or ["webqsp", "metaqa", "2wiki_clean", "musique_clean"]),
                          eval_dataset=a.eval_dataset or "hotpotqa_clean", arms=tuple(a.arms),
                          epochs=a.epochs, seeds=a.seeds, corrupt_p=a.corrupt_p, sharpen=a.sharpen, n_swap=a.n_swap)
    elif a.central_router:
        _arms = tuple(a.arms) if a.arms != ["conf_only", "conf_easy", "conf_plausible"] else \
            ("conf_only", "full_fd2", "central_kl", "central_kl_advisor")
        run_central_transfer(train_datasets=tuple(a.train_datasets or ["webqsp", "metaqa", "2wiki_clean", "musique_clean"]),
                             eval_dataset=a.eval_dataset or "hotpotqa_clean", arms=_arms,
                             epochs=a.epochs, seeds=a.seeds)
    elif a.eval_dataset:
        run_transfer(train_datasets=tuple(a.train_datasets or ["webqsp", "metaqa"]), eval_dataset=a.eval_dataset,
                     mode=a.transfer_mode, epochs=a.epochs, seeds=a.seeds)
    else:
        _run(datasets=tuple(a.datasets), epochs=a.epochs, seeds=a.seeds, modes=tuple(a.modes),
             out_path=a.out, preds_path=a.preds)


if __name__ == "__main__":
    main()
