"""CRAG L2 Phase-D: candidate-level LEARNING-TO-RANK (XGBRanker) + controller x scorer ablation + per-query oracle.

Motivation (Phase A/B/C): the alpha bottleneck (query-state -> alpha -> candidate MLP) may throw away useful
candidate-level interactions, and the fusion DILUTES the strong expert on splade-dominant sets (metaqa/hotpot fusion
< standalone splade) while offset is the unique-gold hero on webqsp. A direct grouped tree ranker can learn universal
evidence rules ("dense&splade agree + offset disagrees + offset has no unique support -> don't let offset demote")
from QUERY EVIDENCE, no dataset ID.

Arms (Section 1):  A KL->MLP   B XGBctrl->MLP   C KL->XGBRanker   D XGBctrl->XGBRanker   E Direct-XGBRanker(no alpha)
A/B reuse crag_fusion. Here we add the candidate feature builder, the XGBRanker (proper query grouping, one group =
one query, negatives subsampled for TRAIN only, FULL pool at EVAL), the per-query expert oracle, and the arms C/D/E.

Authoritative mask governs missingness everywhere; no sentinel inference. Frozen present-only mu/sd (never recomputed).
"""
import logging, json
import numpy as np
import torch
import torch.nn.functional as F

from src.experiments.kg_hybrid import (EXPERTS, NE, EXPERT_NAMES, FEAT_NAMES, KS, HS, _pooled_musd, _load,
                                        _qmet, _aggregate)
from src.experiments import crag_fusion as CF
from src.experiments.crag_fusion import prepbatch, apply_drop, state_of, _controller, DROP_REGIMES

log = logging.getLogger(__name__)
DENSE, OFFSET, SPLADE, RELATION, PATH = 0, 1, 2, 3, 4        # positions WITHIN EXPERTS (dense,offset,splade,relation,path)
TOPK = 5


# ------------------------------------------------------------------------------------------------------------------
# Candidate feature builder. Returns (n_cand, F) float32 + the parallel feature-name list (built once).
# ------------------------------------------------------------------------------------------------------------------
def _valid_percentile(xr):
    """Per-expert percentile rank of each candidate among VALID candidates (higher score -> higher pct). (n,NE).
    Missing candidates (xr=-inf) -> 0.0. xr is x_rank[:,EXPERTS] (missing -> -inf)."""
    n, ne = xr.shape
    out = np.zeros((n, ne), dtype=np.float32)
    for e in range(ne):
        col = xr[:, e]
        valid = np.isfinite(col)
        nv = int(valid.sum())
        if nv <= 1:
            out[valid, e] = 1.0
            continue
        order = np.argsort(col[valid])                        # ascending
        ranks = np.empty(nv); ranks[order] = np.arange(nv)
        out[np.where(valid)[0], e] = ranks / (nv - 1)         # 0..1, best=1
    return out


def _topk_member(xr, K):
    """Boolean (n,NE): is candidate in expert e's VALID top-K (by score)."""
    n, ne = xr.shape
    m = np.zeros((n, ne), dtype=np.float32)
    for e in range(ne):
        col = xr[:, e]
        valid = np.where(np.isfinite(col))[0]
        if len(valid) == 0:
            continue
        kk = min(K, len(valid))
        top = valid[np.argsort(-col[valid])[:kk]]
        m[top, e] = 1.0
    return m


def _pair_overlap(member, a, b):
    """|topK_a ∩ topK_b| / min(|topK_a|,|topK_b|) — scalar query-level agreement between experts a,b."""
    sa, sb = member[:, a], member[:, b]
    inter = float((sa * sb).sum()); na = float(sa.sum()); nb = float(sb.sum())
    denom = max(min(na, nb), 1.0)
    return inter / denom


_CAND_FEAT_NAMES = None
def cand_feat_names(with_alpha):
    base = ([f"score[{n}]" for n in EXPERT_NAMES] +
            [f"avail[{n}]" for n in EXPERT_NAMES] +
            [f"pct[{n}]" for n in EXPERT_NAMES] +
            [f"top5[{n}]" for n in EXPERT_NAMES] +
            ["both_dense_splade", "offset_only", "relation_only", "path_only", "dense_only", "splade_only"] +
            [f"q_margin[{n}]" for n in EXPERT_NAMES] +
            [f"q_entropy[{n}]" for n in EXPERT_NAMES] +
            [f"q_pmax[{n}]" for n in EXPERT_NAMES] +
            [f"q_validfrac[{n}]" for n in EXPERT_NAMES] +
            [f"q_avail[{n}]" for n in EXPERT_NAMES] +
            ["agree_dense_splade", "agree_dense_offset", "agree_splade_offset",
             "semantic_consensus", "offset_evidence_adv", "relation_coverage", "path_coverage"])
    if with_alpha:
        base += [f"alpha[{n}]" for n in EXPERT_NAMES] + [f"alpha_x_score[{n}]" for n in EXPERT_NAMES]
    return base


def cand_features(pb, K=TOPK, alpha=None):
    """(n_cand, F) candidate features. Query-level signals are repeated on every candidate of the query.
    alpha (torch (NE,)) optional -> appends alpha and alpha*score (controller arms). Direct arm passes alpha=None."""
    xf = pb.x_fuse[:, EXPERTS].cpu().numpy()                  # normalized, missing->0
    xr = pb.x_rank[:, EXPERTS].cpu().numpy()                  # missing->-inf
    mask = pb.mask[:, EXPERTS].cpu().numpy()
    n = xf.shape[0]
    pct = _valid_percentile(xr)
    member = _topk_member(xr, K)
    # candidate support flags
    d, o, s, r, p = member[:, DENSE], member[:, OFFSET], member[:, SPLADE], member[:, RELATION], member[:, PATH]
    both_ds = (d * s)
    offset_only = (o * (1 - d) * (1 - s))
    relation_only = (r * (1 - d) * (1 - s) * (1 - o))
    path_only = (p * (1 - d) * (1 - s) * (1 - o))
    dense_only = (d * (1 - s) * (1 - o))
    splade_only = (s * (1 - d) * (1 - o))
    # query-level state via crag_fusion (margin/entropy/agreement) + pmax + valid frac
    z, cnt = CF._valid_z(pb)                                  # torch
    ent, marg = CF._ent_marg(z, cnt)
    pmax = torch.nan_to_num(F.softmax(z, 0), nan=0.0).max(0).values
    marg = marg.cpu().numpy(); ent = ent.cpu().numpy(); pmax = pmax.cpu().numpy()
    validfrac = mask.mean(0)                                  # (NE,)
    qavail = (mask.sum(0) > 0).astype(np.float32)
    ag_ds = _pair_overlap(member, DENSE, SPLADE)
    ag_do = _pair_overlap(member, DENSE, OFFSET)
    ag_so = _pair_overlap(member, SPLADE, OFFSET)
    sem_consensus = ag_ds
    offset_adv = float(marg[OFFSET] - 0.5 * (marg[DENSE] + marg[SPLADE]))
    rel_cov = float(validfrac[RELATION]); path_cov = float(validfrac[PATH])
    qvec = np.concatenate([marg, ent, pmax, validfrac, qavail,
                           [ag_ds, ag_do, ag_so, sem_consensus, offset_adv, rel_cov, path_cov]]).astype(np.float32)
    cand_block = np.concatenate([xf, mask, pct, member,
                                 np.stack([both_ds, offset_only, relation_only, path_only, dense_only, splade_only], 1)], 1)
    Q = np.tile(qvec, (n, 1))
    feats = np.concatenate([cand_block, Q], 1)
    if alpha is not None:
        a = alpha.cpu().numpy().astype(np.float32)
        feats = np.concatenate([feats, np.tile(a, (n, 1)), xf * a[None, :]], 1)
    return feats.astype(np.float32)


# ------------------------------------------------------------------------------------------------------------------
# Per-query expert ORACLE (Section 6): best achievable R@k if we could pick the best single expert per query.
# ------------------------------------------------------------------------------------------------------------------
def _expert_r_at(col, mask_col, y, K):
    valid = mask_col > 0.5
    if valid.sum() == 0:
        return None
    s = np.where(valid, col, -np.inf)
    top = np.argsort(-s)[:K]
    ng = int(y.sum())
    return float(y[top].sum() / ng) if ng else 0.0


def oracle_diag(datasets, DEV, K=5):
    data = _load(list(datasets), DEV)
    mu, sd = _pooled_musd(data, list(datasets), DEV, "train")
    out = {}
    for ds in datasets:
        rows = [prepbatch(r, mu, sd, DEV, ds) for r in data[ds]["test"]]
        per_expert = {n: [] for n in EXPERT_NAMES}
        anchor, best_expert, pool_oracle = [], [], []
        for pb in rows:
            y = pb.y.cpu().numpy(); ng = int(y.sum())
            if ng == 0:
                continue
            xr = pb.x_rank[:, EXPERTS].cpu().numpy()
            mask = pb.mask[:, EXPERTS].cpu().numpy()
            rvals = []
            for ei, nm in enumerate(EXPERT_NAMES):
                r = _expert_r_at(xr[:, ei], mask[:, ei], y, K)
                if r is not None:
                    per_expert[nm].append(r)
                    rvals.append(r)
                else:
                    rvals.append(0.0)
            best_expert.append(max(rvals))
            # semantic anchor = z_dense + z_splade (both always available), rank-safe sum
            zc = pb.x_fuse[:, EXPERTS].cpu().numpy()
            sem = zc[:, DENSE] + zc[:, SPLADE]
            top = np.argsort(-sem)[:K]; anchor.append(float(y[top].sum() / ng))
            pool_oracle.append(1.0)                            # gold is in pool by construction of y; pool-oracle handled separately
        out[ds] = {
            "standalone_R@5": {n: round(100 * float(np.mean(v)), 2) for n, v in per_expert.items() if v},
            "semantic_anchor_R@5": round(100 * float(np.mean(anchor)), 2),
            "per_query_expert_oracle_R@5": round(100 * float(np.mean(best_expert)), 2),
            "n": len(best_expert),
        }
        log.info("[oracle %s] best-standalone=%s anchor=%.2f per-query-oracle=%.2f", ds,
                 max(out[ds]["standalone_R@5"].values()), out[ds]["semantic_anchor_R@5"], out[ds]["per_query_expert_oracle_R@5"])
    return out


# ------------------------------------------------------------------------------------------------------------------
# XGBRanker: proper query grouping. TRAIN subsamples negatives per query; EVAL uses the FULL pool.
# ------------------------------------------------------------------------------------------------------------------
def _train_rows(pb, alpha, K, neg_cap=150, rng=None):
    """Feature matrix + relevance for ONE training query, negatives subsampled (all golds + hard/random negs)."""
    feats = cand_features(pb, K, alpha)
    y = pb.y.cpu().numpy().astype(np.int32)
    pos = np.where(y > 0)[0]
    neg = np.where(y == 0)[0]
    if len(neg) > neg_cap:
        # hard negatives = top by max normalized expert score, plus random
        xf = pb.x_fuse[:, EXPERTS].cpu().numpy().max(1)
        hard = neg[np.argsort(-xf[neg])[:neg_cap // 2]]
        rest = rng.choice(neg, size=neg_cap - len(hard), replace=False)
        neg = np.concatenate([hard, rest])
    idx = np.concatenate([pos, neg])
    return feats[idx], y[idx]


def _alpha_for(arm, pb, state, qenc=None, models=None):
    if arm in ("C_kl_xgbrank",):
        with torch.no_grad():
            alpha, _ = _controller(qenc, pb, state)
        return alpha.detach()
    if arm in ("D_xgbctrl_xgbrank",):
        return CF._xgb_alpha(models, pb, state)
    return None                                                # E_direct_xgbrank


def fit_xgbranker(TR, arm, K=TOPK, neg_cap=150, seed=0, qenc=None, models=None,
                  n_estimators=300, max_depth=6, lr=0.1):
    import xgboost as xgb
    rng = np.random.RandomState(seed)
    Xs, ys, groups = [], [], []
    for pb in TR:
        state = state_of(pb, K) if arm != "E_direct_xgbrank" else None
        alpha = _alpha_for(arm, pb, state, qenc, models)
        fx, fy = _train_rows(pb, alpha, K, neg_cap, rng)
        if fy.sum() == 0:                                      # no gold scorable -> skip (nothing to rank toward)
            continue
        Xs.append(fx); ys.append(fy); groups.append(len(fy))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    r = xgb.XGBRanker(objective="rank:pairwise", n_estimators=n_estimators, max_depth=max_depth,
                      learning_rate=lr, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                      n_jobs=4, verbosity=0)
    r.fit(X, y, group=groups)
    r.get_booster().feature_names = cand_feat_names(arm != "E_direct_xgbrank")
    return r


def eval_xgbranker(ranker, TE, datasets, arm, K=TOPK, qenc=None, models=None):
    recs = {ds: [] for ds in datasets}
    for pb in TE:
        if pb.ds not in recs:
            continue
        state = state_of(pb, K) if arm != "E_direct_xgbrank" else None
        alpha = _alpha_for(arm, pb, state, qenc, models)
        X = cand_features(pb, K, alpha)
        S = ranker.predict(X)
        ry = pb.y.cpu().numpy()[np.argsort(-S)][:50].astype(np.uint8)
        recs[pb.ds].append((ry, pb.ng))
    return recs


def ranker_importance(ranker, topn=15):
    b = ranker.get_booster()
    return {k: round(float(v), 1) for k, v in sorted(b.get_score(importance_type="gain").items(),
                                                     key=lambda kv: -kv[1])[:topn]}


# ------------------------------------------------------------------------------------------------------------------
# STEP 2 driver: 5-arm controller x scorer ablation. In-dist (train on pooled `datasets`, eval each) or with a
# held-out sanity set. p_drop LOCKED at 0.2. Reports R@5 per dataset + tree-ranker gain importance.
# ------------------------------------------------------------------------------------------------------------------
ARMS_ALL = ["A_kl_mlp", "B_xgbctrl_mlp", "C_kl_xgbrank", "D_xgbctrl_xgbrank", "E_direct_xgbrank"]


def _prep(data, datasets, split, mu, sd, DEV):
    return [prepbatch(r, mu, sd, DEV, ds) for ds in datasets for r in data[ds][split]]


def run_arms(datasets=("webqsp", "metaqa"), arms=ARMS_ALL, epochs=12, seeds=3, p_drop=0.2, lam=1.0,
             neg_cap=150, out_path="results/L2/crag_ranker_arms.json", heldout=None):
    """Train each arm on the pooled train of `datasets` (in-dist), eval each dataset's test. If `heldout` given,
    also train-on-datasets / eval-on-heldout as a transfer sanity (heldout NEVER in training)."""
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ds = list(datasets) + ([heldout] if heldout else [])
    data = _load(all_ds, DEV)
    mu, sd = _pooled_musd(data, list(datasets), DEV, "train")     # frozen, train-only, present-only (heldout excluded)
    TR = _prep(data, list(datasets), "train", mu, sd, DEV)
    TE = _prep(data, list(datasets), "test", mu, sd, DEV)
    TE_h = _prep(data, [heldout], "test", mu, sd, DEV) if heldout else None
    out = {"_config": {"datasets": list(datasets), "heldout": heldout, "arms": list(arms), "epochs": epochs,
                       "seeds": seeds, "p_drop": p_drop, "lam": lam, "neg_cap": neg_cap,
                       "note": "in-dist (target in training) + optional transfer sanity on heldout (never trained)."},
           "oracle": oracle_diag(datasets, DEV, 5),
           "results": {}, "importance": {}}
    for arm in arms:
        indist, transfer, imp = [], [], None
        for s in range(seeds):
            if arm == "A_kl_mlp":
                qenc, cand = CF.train_crag(s, TR, DEV, epochs, lam, 1.0, CF.CONF_K, 32, DROP_REGIMES, p_drop)
                indist.append(CF.eval_crag(qenc, cand, TE, list(datasets)))
                if TE_h: transfer.append(CF.eval_crag(qenc, cand, TE_h, [heldout]))
            elif arm == "B_xgbctrl_mlp":
                X, Y = CF.collect_teacher(TR, CF.CONF_K, DROP_REGIMES, p_drop, 2, s)
                models, imp = CF.fit_xgb(X, Y)
                cand = CF.train_scorer_xgb(s, TR, models, DEV, epochs, CF.CONF_K, DROP_REGIMES, p_drop)
                indist.append(CF.eval_xgb(models, cand, TE, list(datasets)))
                if TE_h: transfer.append(CF.eval_xgb(models, cand, TE_h, [heldout]))
            elif arm == "C_kl_xgbrank":
                qenc, _ = CF.train_crag(s, TR, DEV, epochs, lam, 1.0, CF.CONF_K, 32, DROP_REGIMES, p_drop)
                rk = fit_xgbranker(TR, arm, TOPK, neg_cap, s, qenc=qenc)
                imp = ranker_importance(rk)
                indist.append(eval_xgbranker(rk, TE, list(datasets), arm, TOPK, qenc=qenc))
                if TE_h: transfer.append(eval_xgbranker(rk, TE_h, [heldout], arm, TOPK, qenc=qenc))
            elif arm == "D_xgbctrl_xgbrank":
                X, Y = CF.collect_teacher(TR, CF.CONF_K, DROP_REGIMES, p_drop, 2, s)
                models, _ = CF.fit_xgb(X, Y)
                rk = fit_xgbranker(TR, arm, TOPK, neg_cap, s, models=models)
                imp = ranker_importance(rk)
                indist.append(eval_xgbranker(rk, TE, list(datasets), arm, TOPK, models=models))
                if TE_h: transfer.append(eval_xgbranker(rk, TE_h, [heldout], arm, TOPK, models=models))
            elif arm == "E_direct_xgbrank":
                rk = fit_xgbranker(TR, arm, TOPK, neg_cap, s)
                imp = ranker_importance(rk)
                indist.append(eval_xgbranker(rk, TE, list(datasets), arm, TOPK))
                if TE_h: transfer.append(eval_xgbranker(rk, TE_h, [heldout], arm, TOPK))
        agg_i, _ = _aggregate(indist, list(datasets))
        out["results"][arm] = {"in_dist": {d: agg_i[d] for d in datasets}}
        if transfer:
            agg_t, _ = _aggregate(transfer, [heldout])
            out["results"][arm]["transfer_heldout"] = {heldout: agg_t[heldout]}
        if imp:
            out["importance"][arm] = imp
        log.info("[arm %s] in-dist R@5=%s%s", arm, {d: agg_i[d]["R@5"][0] for d in datasets},
                 f" | transfer {heldout} R@5={agg_t[heldout]['R@5'][0]}" if transfer else "")
        json.dump(out, open(out_path, "w"), indent=2)
    log.info("-> %s", out_path)
    return out


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--oracle", action="store_true", help="per-query expert oracle diagnostic only")
    p.add_argument("--arms", nargs="+", default=ARMS_ALL)
    p.add_argument("--datasets", nargs="+", default=["webqsp", "metaqa"])
    p.add_argument("--heldout", default=None, help="transfer-sanity dataset (never trained)")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--p-drop", type=float, default=0.2)
    p.add_argument("--neg-cap", type=int, default=150)
    p.add_argument("--out", default="results/L2/crag_ranker_arms.json")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.oracle:
        res = oracle_diag(a.datasets, DEV, 5)
        json.dump(res, open(a.out, "w"), indent=2)
        log.info("-> %s", a.out)
    else:
        run_arms(tuple(a.datasets), a.arms, a.epochs, a.seeds, a.p_drop, out_path=a.out, heldout=a.heldout)


if __name__ == "__main__":
    main()
