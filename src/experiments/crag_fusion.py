"""CRAG L2 fusion — Phase 2 (expert/structure dropout) + Phase 3 (centralized KL controller + bounded-InfoNCE
expert teacher + expressive candidate MLP), on the FROZEN canonical PrepBatch substrate.

Single semantics everywhere (see kg_hybrid.PrepBatch):
  * mask = authoritative per-candidate existence; qavail = mask.any over candidates (query-level, per expert).
  * x_fuse (missing->0) feeds the MLP/fusion; x_rank (missing->-inf) feeds ranking / margins / top-k / InfoNCE.
  * mu/sd are FROZEN train-only present-only (kg_hybrid._pooled_musd). Dropout edits the MASK first, then derives
    x_fuse/x_rank/qavail from it -- normalization is NEVER recomputed after dropout.

Controller: alpha(state) with UNAVAILABLE experts hard-masked to logit -inf (alpha_e=0). Teacher pi* = softmax(-L_e/tau)
over AVAILABLE experts only (L_e = bounded-InfoNCE expert quality, clamped to MAX_EXPERT_LOSS; an available expert
that ranks no gold gets the worst loss but is NOT excluded). Objective = ListNet candidate loss + lam*KL(pi*||alpha),
the KL summed only over available experts. alpha is an ADVISOR: the MLP scores candidates from
[s, alpha, alpha*s, state, availability] and can override the routing. (KL=who to trust, InfoNCE=which candidate,
dropout=what to do when experts vanish, MLP=how signals interact.)

  Phase 2 ablation = lam=0 (dropout-robust scorer, uniform-ish alpha).   Phase 3 = lam>0 (KL controller).
  run_lodo = true leave-one-dataset-out over the 5 datasets (held-out never used for tuning).
"""
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.experiments.kg_hybrid import (PrepBatch, EXPERTS, NE, EXPERT_NAMES, FEAT_NAMES, KS, HS, METR,
                                       _expert_loss, _row_unpack, _pooled_musd, _load, _qmet, _aggregate,
                                       QEnc, Cand, MAX_EXPERT_LOSS, _dc_replace)

log = logging.getLogger(__name__)
CONF_K = 5
STATE_DIM = 3 * NE                                  # per-expert [margin, entropy, cross-expert agreement]
UIN = STATE_DIM + NE                                # controller input: state + availability(NE)
CAND_IN = len(FEAT_NAMES) + 2 * NE + STATE_DIM + NE  # [s(7), alpha(NE), alpha*s(NE), state(3NE), avail(NE)]

# Phase-2 missing-feature regimes (FEAT_NAMES column indices to drop). "full" = no drop.
# dense=0 offset=1 prototype=2 splade=3 graph=4 relation=5 path=6
DROP_REGIMES = {
    "full":               [],
    "no_edge_path":       [5, 6],
    "no_graph_edge_path": [4, 5, 6],
    "no_splade":          [3],
    "no_offset":          [1],
    "dense_only":         [1, 2, 3, 4, 5, 6],
    "dense_splade":       [1, 2, 4, 5, 6],
    "structure_heavy":    [0, 1],                  # drop the always-on semantic priors -> force splade/graph/edge/path
}


def prepbatch(row, mu, sd, DEV, ds=""):
    """Build the canonical PrepBatch from a saved substrate row. mu/sd are the FROZEN present-only pooled stats."""
    x, y, ng, m = _row_unpack(row)
    x = x.to(DEV).float(); y = y.to(DEV).float(); m = m.to(DEV).float()
    z = (x - mu) / sd
    x_fuse = torch.where(m > 0.5, z, torch.zeros_like(z))                    # missing -> 0 (fusion)
    x_rank = torch.where(m > 0.5, z, torch.full_like(z, float("-inf")))      # missing -> -inf (ranking)
    qavail = (m.sum(0) > 0).float()                                          # query-level availability, ALL columns
    return PrepBatch(raw=x, mask=m, qavail=qavail, x_fuse=x_fuse, x_rank=x_rank, y=y, ng=int(ng), ds=ds)


def apply_drop(pb, drop_cols):
    """Phase-2 dropout: edit the MASK first, then DERIVE x_fuse/x_rank/qavail from the masked state.
    Frozen mu/sd are never recomputed -- the coordinate system is fixed; only availability changes."""
    if not drop_cols:
        return pb
    m = pb.mask.clone(); xf = pb.x_fuse.clone(); xr = pb.x_rank.clone(); qa = pb.qavail.clone()
    for c in drop_cols:
        m[:, c] = 0.0; xf[:, c] = 0.0; xr[:, c] = float("-inf"); qa[c] = 0.0
    return _dc_replace(pb, mask=m, x_fuse=xf, x_rank=xr, qavail=qa)


# ---- mask-aware query state (ONE definition; entropy normalized by log(#valid candidates for that expert)) ----
def _valid_z(pb):
    """Per-query per-expert standardization over VALID candidates only. Returns (z, cnt): z present-standardized
    with missing = -inf; cnt = #valid candidates per expert (for entropy normalization / k_eff)."""
    sub = pb.x_fuse[:, EXPERTS]; m = pb.mask[:, EXPERTS]
    cnt = m.sum(0)                                                  # (NE,)
    safe = cnt.clamp(min=1.0)
    mean = (sub * m).sum(0) / safe
    var = (((sub - mean) ** 2) * m).sum(0) / safe
    std = var.sqrt().clamp(min=1e-6)
    z = (sub - mean) / std
    z = torch.where(m > 0.5, z, torch.full_like(z, float("-inf")))
    return z, cnt


def _ent_marg(z, cnt):
    p = torch.nan_to_num(F.softmax(z, 0), nan=0.0)                  # all-(-inf) column -> 0
    ent_raw = -(p * torch.where(p > 0, (p + 1e-12).log(), torch.zeros_like(p))).sum(0)
    ln = torch.log(cnt.clamp(min=2.0))                             # normalize by log(#VALID candidates for this expert)
    ent = torch.where(cnt > 1, ent_raw / ln, torch.zeros_like(ent_raw))
    kk = min(2, z.shape[0]); tp = torch.topk(z, kk, dim=0).values
    marg = torch.where(cnt >= 2, tp[0] - tp[-1], torch.zeros(z.shape[1], device=z.device))
    marg = torch.nan_to_num(marg, neginf=0.0, posinf=0.0)
    return ent, marg


def _agree(z, cnt, K=CONF_K):
    """Cross-expert top-k overlap using k_eff = min(K, #valid) per expert; -inf entries can NEVER enter a top-k set
    (no fabricated agreement). Overlap normalized by the max achievable (min of the two valid top-k sizes)."""
    n, ne = z.shape; Kk = min(K, n)
    order = torch.argsort(z, dim=0, descending=True)               # (n, NE)
    keff = torch.minimum(cnt, torch.full_like(cnt, float(Kk)))     # (NE,)
    topk_idx = order[:Kk]                                          # (Kk, NE)
    ar = torch.arange(Kk, device=z.device).unsqueeze(1)           # (Kk,1)
    valid_rank = ar < keff.unsqueeze(0)                           # (Kk, NE): which of the top-Kk are within k_eff
    memb = torch.zeros(n, ne, device=z.device)
    for e in range(ne):
        sel = topk_idx[:, e][valid_rank[:, e]]
        memb[sel, e] = 1.0
    inter = memb.t() @ memb                                        # (NE,NE) intersection counts
    sizes = memb.sum(0)                                            # (NE,)
    agree = torch.zeros(ne, device=z.device)
    for i in range(ne):
        accs = []
        for j in range(ne):
            if i == j:
                continue
            denom = torch.minimum(sizes[i], sizes[j]).clamp(min=1.0)
            accs.append(inter[i, j] / denom)
        agree[i] = torch.stack(accs).mean() if accs else torch.zeros((), device=z.device)
    return agree


def state_of(pb, K=CONF_K):
    z, cnt = _valid_z(pb)
    ent, marg = _ent_marg(z, cnt)
    ag = _agree(z, cnt, K)
    return torch.cat([marg, ent, ag])                             # (3*NE,)


def _cand_input(pb, alpha, state):
    n = pb.x_fuse.shape[0]
    aexp = pb.x_fuse[:, EXPERTS] * alpha[None, :]
    avail = pb.qavail[EXPERTS]
    return torch.cat([pb.x_fuse, alpha[None, :].expand(n, -1), aexp,
                      state[None, :].expand(n, -1), avail[None, :].expand(n, -1)], 1)


def _controller(qenc, pb, state):
    """alpha with UNAVAILABLE experts hard-masked to -inf. Returns (alpha, alogit). Asserts >=1 expert available."""
    avail = pb.qavail[EXPERTS]
    assert bool(avail.any()), "no available expert for query (all-unavailable) -- would produce NaN alpha"
    _, alogit = qenc(torch.cat([state, avail]))
    alogit = torch.where(avail > 0.5, alogit, torch.full_like(alogit, float("-inf")))
    return F.softmax(alogit, 0), alogit


def _teacher(pb):
    """pi* = softmax(-L_e/tau) over AVAILABLE experts only; unavailable -> logit -inf (pi*_e=0). L_e is bounded
    InfoNCE from x_rank (post-drop): a dropped/absent expert scores no gold -> excluded; an available expert that
    can't rank a gold keeps the worst bounded loss (NOT excluded)."""
    avail = pb.qavail[EXPERTS]
    Le = torch.stack([_expert_loss(pb.x_rank[:, c], pb.y) for c in EXPERTS])
    return avail, Le


def train_crag(seed, TR, DEV, epochs=12, lam=1.0, tau=1.0, K=CONF_K, hq=32,
               regimes=None, p_drop=0.5):
    """Phase 2+3 trainer. p_drop = probability a training query is put in a (sampled) missing-feature regime;
    lam = KL controller weight (lam=0 -> Phase-2 dropout-only scorer). Frozen mu/sd already baked into TR."""
    torch.manual_seed(seed); np.random.seed(seed)
    regimes = regimes if regimes is not None else DROP_REGIMES
    rnames = list(regimes.keys())
    qenc = QEnc(UIN, hq).to(DEV); cand = Cand(CAND_IN).to(DEV)
    opt = torch.optim.Adam(list(cand.parameters()) + list(qenc.parameters()), lr=1e-3)
    for ep in range(epochs):
        for j in np.random.permutation(len(TR)):
            pb = TR[j]
            if p_drop > 0 and np.random.rand() < p_drop:
                pbc = apply_drop(pb, regimes[rnames[np.random.randint(len(rnames))]])
            else:
                pbc = pb
            if float(pbc.qavail[EXPERTS].sum()) == 0:               # regime removed every expert (shouldn't happen); skip
                continue
            state = state_of(pbc, K)
            alpha, alogit = _controller(qenc, pbc, state)
            S = cand(_cand_input(pbc, alpha, state))
            y = pb.y
            closs = -(F.log_softmax(S, 0) * (y / y.sum())).sum()    # ListNet candidate correctness
            if lam > 0:
                with torch.no_grad():
                    avail, Le = _teacher(pbc)
                    tlog = torch.where(avail > 0.5, -Le / tau, torch.full_like(Le, float("-inf")))
                    pi = torch.nan_to_num(F.softmax(tlog, 0), nan=0.0)
                la = F.log_softmax(alogit, 0)
                kl = (pi * ((pi + 1e-9).log() - la))
                kl = kl[avail > 0.5].sum()                          # only over available experts (no 0*-inf)
                loss = closs + lam * kl
            else:
                loss = closs
            opt.zero_grad(); loss.backward(); opt.step()
    return qenc, cand


# ---------------------------------------------------------------------------------------------------------------
# STEP 7: XGBoost controller (decision boundaries over the tabular query/expert state) vs the neural MLP controller.
# Both imitate the SAME bounded-InfoNCE teacher pi* and feed the SAME expressive candidate MLP scorer -- so the
# comparison isolates the controller family. XGBoost also yields gain-based feature importance (rule explainability).
STATE_FEATS = ([f"margin[{n}]" for n in EXPERT_NAMES] + [f"entropy[{n}]" for n in EXPERT_NAMES]
               + [f"agree[{n}]" for n in EXPERT_NAMES] + [f"avail[{n}]" for n in EXPERT_NAMES])   # 4*NE names


def collect_teacher(TR, K=CONF_K, regimes=None, p_drop=0.5, passes=2, seed=0):
    """Build the XGBoost training set: for each training query (with dropout regimes sampled, matching the scorer's
    experience) collect features [state(3NE), avail(NE)] and the teacher target pi* over AVAILABLE experts."""
    regimes = regimes if regimes is not None else DROP_REGIMES
    rnames = list(regimes.keys()); rng = np.random.RandomState(seed)
    X, Y = [], []
    for _ in range(passes):
        for pb in TR:
            pbc = apply_drop(pb, regimes[rnames[rng.randint(len(rnames))]]) if rng.rand() < p_drop else pb
            avail = pbc.qavail[EXPERTS]
            if float(avail.sum()) == 0:
                continue
            state = state_of(pbc, K)
            _, Le = _teacher(pbc)
            tlog = torch.where(avail > 0.5, -Le / 1.0, torch.full_like(Le, float("-inf")))
            pi = torch.nan_to_num(F.softmax(tlog, 0), nan=0.0)
            X.append(torch.cat([state, avail]).cpu().numpy())
            Y.append(pi.cpu().numpy())
    return np.stack(X), np.stack(Y)


def fit_xgb(X, Y):
    """One XGBRegressor per expert predicting pi*_e from [state, avail]. Returns (models, gain_importance)."""
    import xgboost as xgb
    models, imp = [], {}
    for e in range(NE):
        m = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4, verbosity=0)
        m.fit(X, Y[:, e])
        models.append(m)
        b = m.get_booster(); b.feature_names = STATE_FEATS
        imp[EXPERT_NAMES[e]] = {k: round(float(v), 2) for k, v in
                               sorted(b.get_score(importance_type="gain").items(), key=lambda kv: -kv[1])[:6]}
    return models, imp


def _xgb_alpha(models, pb, state):
    """alpha from XGBoost: predict pi*_e, hard-mask unavailable experts to 0, renormalize over available."""
    avail = pb.qavail[EXPERTS].cpu().numpy()
    feat = np.concatenate([state.cpu().numpy(), avail])[None, :]
    raw = np.array([float(m.predict(feat)[0]) for m in models])
    raw = np.clip(raw, 0.0, None) * avail
    s = raw.sum()
    a = (raw / s) if s > 1e-9 else (avail / max(avail.sum(), 1))
    return torch.tensor(a, dtype=torch.float32, device=pb.x_fuse.device)


def train_scorer_xgb(seed, TR, models, DEV, epochs=12, K=CONF_K, regimes=None, p_drop=0.5):
    """Train the candidate MLP with alpha supplied by the (fixed) XGBoost controller; same dropout as the KL-MLP."""
    torch.manual_seed(seed); np.random.seed(seed)
    regimes = regimes if regimes is not None else DROP_REGIMES
    rnames = list(regimes.keys())
    cand = Cand(CAND_IN).to(DEV)
    opt = torch.optim.Adam(cand.parameters(), lr=1e-3)
    for ep in range(epochs):
        for j in np.random.permutation(len(TR)):
            pb = TR[j]
            pbc = apply_drop(pb, regimes[rnames[np.random.randint(len(rnames))]]) if (p_drop > 0 and np.random.rand() < p_drop) else pb
            if float(pbc.qavail[EXPERTS].sum()) == 0:
                continue
            state = state_of(pbc, K)
            with torch.no_grad():
                alpha = _xgb_alpha(models, pbc, state)
            S = cand(_cand_input(pbc, alpha, state))
            loss = -(F.log_softmax(S, 0) * (pb.y / pb.y.sum())).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return cand


def eval_xgb(models, cand, TE, datasets, K=CONF_K):
    recs = {ds: [] for ds in datasets}
    with torch.no_grad():
        for pb in TE:
            if pb.ds not in recs:
                continue
            state = state_of(pb, K)
            alpha = _xgb_alpha(models, pb, state)
            S = cand(_cand_input(pb, alpha, state)).cpu().numpy()
            ry = pb.y.cpu().numpy()[np.argsort(-S)][:50].astype(np.uint8)
            recs[pb.ds].append((ry, pb.ng))
    return recs


def eval_crag(qenc, cand, TE, datasets, K=CONF_K):
    """Full-availability eval (no dropout): rank candidates by the MLP score -> top-50 gold mask + ng."""
    recs = {ds: [] for ds in datasets}
    with torch.no_grad():
        for pb in TE:
            if pb.ds not in recs:
                continue
            state = state_of(pb, K)
            alpha, _ = _controller(qenc, pb, state)
            S = cand(_cand_input(pb, alpha, state)).cpu().numpy()
            ry = pb.y.cpu().numpy()[np.argsort(-S)][:50].astype(np.uint8)
            recs[pb.ds].append((ry, pb.ng))
    return recs


def mean_alpha(qenc, TE, datasets, K=CONF_K):
    accs = []
    with torch.no_grad():
        for pb in TE:
            if pb.ds not in datasets:
                continue
            state = state_of(pb, K)
            alpha, _ = _controller(qenc, pb, state)
            accs.append(alpha.cpu().numpy())
    return {EXPERT_NAMES[i]: round(float(np.mean([a[i] for a in accs])), 4) for i in range(NE)} if accs else None


def _prep_split(data, datasets, split, mu, sd, DEV):
    return [prepbatch(r, mu, sd, DEV, ds) for ds in datasets for r in data[ds][split]]


def run_lodo(all_datasets=("webqsp", "metaqa", "2wiki_clean", "musique_clean", "hotpotqa_clean"),
             arms=("dropout_only", "kl_dropout"), epochs=12, seeds=5, lam=1.0, tau=1.0, p_drop=0.5,
             out_path="results/L2/crag_lodo.json"):
    """True leave-one-dataset-out over the 5 datasets. For each held-out dataset: FROZEN present-only pooled mu/sd
    from the 4 training datasets only; train each arm; zero-shot eval on the held-out set + re-eval train sets.
    arms: dropout_only (lam=0), kl_dropout (lam=lam). Held-out labels never touch training or selection."""
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _load(list(all_datasets), DEV)
    ARM = {"dropout_only": 0.0, "kl_dropout": lam}
    out = {"_config": {"kind": "true_LODO", "datasets": list(all_datasets), "arms": list(arms),
                       "epochs": epochs, "seeds": seeds, "lam": lam, "tau": tau, "p_drop": p_drop,
                       "regimes": {k: v for k, v in DROP_REGIMES.items()},
                       "norm": "FROZEN present-only pooled mu/sd from the 4 TRAIN datasets (held-out never used)",
                       "controller": "hard-masked alpha (unavail->-inf) + KL(pi*||alpha) over available experts",
                       "scorer": "MLP[s, alpha, alpha*s, state, availability]"}}
    for held in all_datasets:
        train_ds = [d for d in all_datasets if d != held]
        mu, sd = _pooled_musd(data, train_ds, DEV, "train")        # frozen, train-only, present-only
        TR = _prep_split(data, train_ds, "train", mu, sd, DEV)
        TE_held = _prep_split(data, [held], "test", mu, sd, DEV)
        TE_train = _prep_split(data, train_ds, "test", mu, sd, DEV)
        out[held] = {}
        for arm in arms:
            erecs, trecs, alphas = [], [], []
            imp = None
            if arm == "xgb_kl":                                    # STEP 7: XGBoost controller -> same MLP scorer
                X, Y = collect_teacher(TR, CONF_K, DROP_REGIMES, p_drop, passes=2, seed=0)
                models, imp = fit_xgb(X, Y)
                for s in range(seeds):
                    cand = train_scorer_xgb(s, TR, models, DEV, epochs, CONF_K, DROP_REGIMES, p_drop)
                    erecs.append(eval_xgb(models, cand, TE_held, [held]))
                    trecs.append(eval_xgb(models, cand, TE_train, train_ds))
            else:                                                  # neural controller (dropout_only / kl_dropout)
                for s in range(seeds):
                    qenc, cand = train_crag(s, TR, DEV, epochs, ARM[arm], tau, CONF_K, 32, DROP_REGIMES, p_drop)
                    erecs.append(eval_crag(qenc, cand, TE_held, [held]))
                    trecs.append(eval_crag(qenc, cand, TE_train, train_ds))
                    alphas.append(mean_alpha(qenc, TE_held, [held]))
            agg_e, _ = _aggregate(erecs, [held])
            agg_t, _ = _aggregate(trecs, train_ds)
            ma = {k: round(float(np.mean([a[k] for a in alphas])), 4) for k in alphas[0]} if alphas and alphas[0] else None
            out[held][arm] = {"held_out": agg_e[held], "train_ref": {d: agg_t[d]["R@5"] for d in train_ds},
                              "held_out_mean_alpha": ma, "xgb_gain_importance": imp}
            log.info("[LODO hold=%s] %-12s held R@5=%s R@20=%s | train R@5=%s | alpha=%s", held, arm,
                     agg_e[held]["R@5"][0], agg_e[held]["R@20"][0], {d: agg_t[d]["R@5"][0] for d in train_ds}, ma)
            if imp:
                log.info("[LODO hold=%s] xgb gain importance: %s", held, imp)
        json.dump(out, open(out_path, "w"), indent=2)              # incremental
    log.info("-> %s", out_path)
    return out


def run_indist(datasets=("webqsp", "metaqa", "2wiki_clean", "musique_clean", "hotpotqa_clean"),
               arms=("dropout_only", "kl_dropout"), epochs=12, seeds=5, lam=1.0, tau=1.0, p_drop=0.5,
               out_path="results/L2/crag_indist.json"):
    """IN-DISTRIBUTION counterpart to run_lodo: ONE universal model trained on the POOLED train of ALL `datasets`
    (every target IS in training), evaluated on each dataset's own test. Same architecture/recipe/dropout as LODO --
    the ONLY difference is the target is seen in training. This isolates domain-transfer (LODO) from capability:
    if in-dist reproduces old CRAG but LODO does not, the gap is pure transfer, not a lost/regressed substrate.
    FROZEN present-only pooled mu/sd from the pooled train of all `datasets`. No dataset ID is ever fed to the model."""
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = _load(list(datasets), DEV)
    ARM = {"dropout_only": 0.0, "kl_dropout": lam}
    mu, sd = _pooled_musd(data, list(datasets), DEV, "train")        # frozen, train-only, present-only, ALL datasets
    TR = _prep_split(data, list(datasets), "train", mu, sd, DEV)
    TE = _prep_split(data, list(datasets), "test", mu, sd, DEV)
    out = {"_config": {"kind": "in_distribution_universal", "datasets": list(datasets), "arms": list(arms),
                       "epochs": epochs, "seeds": seeds, "lam": lam, "tau": tau, "p_drop": p_drop,
                       "regimes": {k: v for k, v in DROP_REGIMES.items()},
                       "norm": "FROZEN present-only pooled mu/sd from the pooled train of ALL datasets (target INCLUDED)",
                       "note": "ONE universal checkpoint per arm, target-in-training; direct counterpart to run_lodo."},
           "results": {}}
    for arm in arms:
        recs, alphas, imp = [], [], None
        if arm == "xgb_kl":
            X, Y = collect_teacher(TR, CONF_K, DROP_REGIMES, p_drop, passes=2, seed=0)
            models, imp = fit_xgb(X, Y)
            for s in range(seeds):
                cand = train_scorer_xgb(s, TR, models, DEV, epochs, CONF_K, DROP_REGIMES, p_drop)
                recs.append(eval_xgb(models, cand, TE, list(datasets)))
        else:
            for s in range(seeds):
                qenc, cand = train_crag(s, TR, DEV, epochs, ARM[arm], tau, CONF_K, 32, DROP_REGIMES, p_drop)
                recs.append(eval_crag(qenc, cand, TE, list(datasets)))
                alphas.append({d: mean_alpha(qenc, TE, [d]) for d in datasets})
        agg, _ = _aggregate(recs, list(datasets))
        ma = None
        if alphas and alphas[0]:
            ma = {d: {k: round(float(np.mean([a[d][k] for a in alphas])), 4) for k in alphas[0][d]} for d in datasets}
        out["results"][arm] = {"per_dataset": {d: agg[d] for d in datasets},
                               "mean_alpha": ma, "xgb_gain_importance": imp}
        log.info("[IN-DIST] %-12s %s", arm, {d: agg[d]["R@5"][0] for d in datasets})
        json.dump(out, open(out_path, "w"), indent=2)
    log.info("-> %s", out_path)
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lodo", action="store_true")
    p.add_argument("--indist", action="store_true", help="in-distribution universal run (target included in training)")
    p.add_argument("--datasets", nargs="+", default=["webqsp", "metaqa", "2wiki_clean", "musique_clean", "hotpotqa_clean"])
    p.add_argument("--arms", nargs="+", default=["dropout_only", "kl_dropout"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--p-drop", type=float, default=0.5)
    p.add_argument("--out", default="results/L2/crag_lodo.json")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if a.indist:
        out = a.out if a.out != "results/L2/crag_lodo.json" else "results/L2/crag_indist.json"
        run_indist(tuple(a.datasets), tuple(a.arms), a.epochs, a.seeds, a.lam, p_drop=a.p_drop, out_path=out)
    elif a.lodo:
        run_lodo(tuple(a.datasets), tuple(a.arms), a.epochs, a.seeds, a.lam, p_drop=a.p_drop, out_path=a.out)


if __name__ == "__main__":
    main()
