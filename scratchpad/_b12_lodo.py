"""G2 B1.2 — INVARIANCE REPAIR · TRUE 3-way LODO + FIRST DIAGNOSTIC.

Loads scratchpad/_b12/{metaqa,2wiki_clean,squad_clean}.npz. Compares SIX admission policies under identical
base-anchored residual admission (sc = fused_rrf_pct + beta*zscore_q(learned_raw), beta TRAIN-selected incl 0.0):
  BASE      fused-top50 (no model)
  S0        retrieval-relative (5)                                    -- candidate-local, identity-clean baseline
  S1old     S0 + B1.1 structural (10)  [MIXED: raw degree/support/ppr = QUERY_GRAPH_REGIME]
  S1new     S0 + INVARIANT structural (10)  [CANDIDATE_LOCAL only]
  S2old     S0 + old structural + B1.1 directional (17)  [n_struct_anchors = QGR]
  S2new     S0 + invariant structural + INVARIANT directional (16)  [CANDIDATE_LOCAL only]

FIRST DIAGNOSTIC ("why can S0 admit recovered golds?"): every recovered gold -> {A dense-top200 only,
B splade-top200 only, C both, D neither}; same decomposition for the subset admitted by the winning S0; plus
dense/splade rank-percentile distributions and RRF support for recovered+admitted vs recovered+rejected.

IDENTITY DIAGNOSTIC (analysis only): 3-way dataset classifier on each block; desideratum = invariant blocks
LOWER predictability than their old counterparts while admission utility is retained.

Linear first; MLP only on a ladder that already shows useful invariant transfer. No dataset id / gold / hop-
label / relation-type in any model feature. TEST untouched. STOP after B1.2 (no six-way).
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np

DSES = ["metaqa", "2wiki_clean", "squad_clean"]
TOP_POOL = 50; K0 = 60; TOP200 = 200
BETAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35]     # 0.0 = BASE null (select_beta may choose not to perturb); low end fine
# ladder tag -> list of stored blocks to concatenate (in order); fused col is always index 2 (inside S0)
LADDER = {"S0": ["S0"], "S1old": ["S0", "S1old"], "S1new": ["S0", "S1new"],
          "S2old": ["S0", "S1old", "S2old"], "S2new": ["S0", "S1new", "S2new"]}
FUSED_COL = 2
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
np.random.seed(0)
try:
    import torch; torch.manual_seed(0); HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def load(ds):
    z = np.load(f"scratchpad/_b12/{ds}.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["blocks"] = {b: d[b] for b in ["S0", "S1old", "S2old", "S1new", "S2new"]}
    return d


def feat(d, tag):
    return np.concatenate([d["blocks"][b] for b in LADDER[tag]], 1)


def per_query_idx(d):
    g = d["groups"]; st = np.concatenate([[0], np.cumsum(g)])
    for i in range(len(g)):
        yield i, int(st[i]), int(st[i + 1])


# ------------------------------------------------------------------ models
def fit_logreg(X, y, iters=400, lr=0.5, l2=1e-3):
    n, dd = X.shape; w = np.zeros(dd); b = 0.0
    pos = max(y.sum(), 1); wpos = (n - pos) / pos
    sw = np.where(y == 1, wpos, 1.0)
    for _ in range(iters):
        z = X @ w + b; p = 1 / (1 + np.exp(-z)); grad = (p - y) * sw
        w -= lr * (X.T @ grad / n + l2 * w); b -= lr * grad.mean()
    return w, b


class MLP:
    def __init__(self, din, h=64):
        self.net = torch.nn.Sequential(torch.nn.Linear(din, h), torch.nn.ReLU(),
                                       torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, 1))

    def fit(self, X, y, epochs=150, lr=1e-3):
        Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
        pos = max(y.sum(), 1); wpos = torch.tensor((len(y) - pos) / pos, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=wpos)
        for _ in range(epochs):
            opt.zero_grad(); out = self.net(Xt).squeeze(1); loss = lossf(out, yt); loss.backward(); opt.step()
        return self

    def score(self, X):
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()


# ------------------------------------------------------------------ query-balanced hard sampling
def hard_sample(d, tag, per_q=60):
    F = feat(d, tag); width = F.shape[1]; y = d["y"]; Xs = []; Ys = []
    for i, a, b in per_query_idx(d):
        f = F[a:b]; yy = y[a:b]; n = len(yy)
        fused = f[:, FUSED_COL]; order = np.argsort(-fused, kind="stable")
        sel = set(np.where(yy == 1)[0].tolist())
        sel |= set(order[:20].tolist())
        sel |= set(order[max(TOP_POOL - 10, 0):TOP_POOL + 10].tolist())
        if width >= 10:
            sel |= set(np.argsort(-f[:, 5:10].sum(1))[:10].tolist())     # structural block (either variant)
        if width >= 16:
            sel |= set(np.argsort(-f[:, 10])[:10].tolist())              # first directional feature (s_dir)
        pool = list(sel); rest = [j for j in range(n) if j not in sel]
        if rest:
            pool += list(np.random.choice(rest, size=min(len(rest), max(per_q - len(pool), 0)), replace=False))
        pool = pool[:per_q] if len(pool) > per_q else pool
        idx = np.array(pool, np.int64); Xs.append(f[idx]); Ys.append(yy[idx])
    return np.concatenate(Xs), np.concatenate(Ys)


# ------------------------------------------------------------------ admission metrics (base-anchored residual)
def _zq(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def admit(d, tag, score_fn, beta=None):
    F = feat(d, tag); y = d["y"]; recov = d["recov"]; qhop = d["qhop"]
    agg = dict(nq=0, ANY=0, RECn=0, RECd=0, ALL=0, rec_adm=0, rec_avail=0, orig_ret=0, orig_base=0,
               new_adm=0, old_evict=0, churn=[], per_hop={})
    for i, a, b in per_query_idx(d):
        yy = y[a:b]; ng = int(yy.sum())
        if ng == 0:
            continue
        f = F[a:b]; raw = score_fn(f)
        sc = raw if beta is None else (f[:, FUSED_COL] + beta * _zq(raw))
        top = set(np.argsort(-sc, kind="stable")[:TOP_POOL].tolist())
        base = set(np.argsort(-f[:, FUSED_COL], kind="stable")[:TOP_POOL].tolist())
        golds = set(np.where(yy == 1)[0].tolist())
        rec = set(np.where(recov[a:b] == 1)[0].tolist()) & golds
        gt = golds & top; gb = golds & base
        agg["nq"] += 1; agg["ANY"] += int(len(gt) >= 1); agg["RECn"] += len(gt); agg["RECd"] += ng
        agg["ALL"] += int(len(gt) == ng)
        agg["rec_adm"] += len(rec & top); agg["rec_avail"] += len(rec)
        agg["orig_ret"] += len(gb & top); agg["orig_base"] += len(gb)
        agg["new_adm"] += len(gt - gb); agg["old_evict"] += len(gb - gt)
        agg["churn"].append(len(top ^ base) / (2 * TOP_POOL))
        h = int(qhop[i]); ph = agg["per_hop"].setdefault(h, dict(nq=0, RECn=0, RECd=0, rec_adm=0, rec_avail=0,
                                                                 new_adm=0, old_evict=0, ALL=0))
        ph["nq"] += 1; ph["RECn"] += len(gt); ph["RECd"] += ng; ph["rec_adm"] += len(rec & top)
        ph["rec_avail"] += len(rec); ph["new_adm"] += len(gt - gb); ph["old_evict"] += len(gb - gt)
        ph["ALL"] += int(len(gt) == ng)
    return agg


def summ(agg):
    nq = max(agg["nq"], 1)
    o = {"nq": agg["nq"], "ANY@50": round(agg["ANY"] / nq, 4),
         "GOLD_RECALL@50": round(agg["RECn"] / max(agg["RECd"], 1), 4),
         "ALL@50": round(agg["ALL"] / nq, 4),
         "RECOVERED_GOLD_ADMISSION@50": round(agg["rec_adm"] / max(agg["rec_avail"], 1), 4),
         "recovered_golds_available": agg["rec_avail"], "recovered_golds_admitted": agg["rec_adm"],
         "ORIGINAL_GOLD_RETENTION@50": round(agg["orig_ret"] / max(agg["orig_base"], 1), 4),
         "NEW_GOLDS_ADMITTED": agg["new_adm"], "OLD_GOLDS_EVICTED": agg["old_evict"],
         "NET_GOLD_GAIN@50": agg["new_adm"] - agg["old_evict"],
         "POOL_CHURN_mean": round(float(np.mean(agg["churn"])) if agg["churn"] else 0.0, 4)}
    ph = {}
    for h, dd in sorted(agg["per_hop"].items()):
        ph[str(h)] = {"nq": dd["nq"], "GOLD_RECALL@50": round(dd["RECn"] / max(dd["RECd"], 1), 4),
                      "RECOVERED_GOLD_ADMISSION@50": round(dd["rec_adm"] / max(dd["rec_avail"], 1), 4),
                      "recovered_avail": dd["rec_avail"], "recovered_admitted": dd["rec_adm"],
                      "NET_GOLD_GAIN@50": dd["new_adm"] - dd["old_evict"]}
    o["PER_HOP"] = ph
    return o


def base_summary(d):
    return summ(admit(d, "S0", lambda F: F[:, FUSED_COL], beta=0.0))


# ------------------------------------------------------------------ FIRST DIAGNOSTIC
def first_diagnostic(d, s0_score_fn, s0_beta):
    """For every RECOVERED gold: dense/splade top-200 membership (A/B/C/D) + rank distributions.
    Repeat for the subset admitted by the winning S0. RRF-support: recovered+admitted vs recovered+rejected."""
    F = feat(d, "S0"); y = d["y"]; recov = d["recov"]
    dr = d["dense_rank"]; sr = d["splade_rank"]
    def cat(drk, srk):
        din = drk < TOP200; sin = srk < TOP200
        if din and sin: return "C_both"
        if din: return "A_dense_only"
        if sin: return "B_splade_only"
        return "D_neither"
    overall = {"A_dense_only": 0, "B_splade_only": 0, "C_both": 0, "D_neither": 0}
    admitted = {"A_dense_only": 0, "B_splade_only": 0, "C_both": 0, "D_neither": 0}
    adm_dense_pct = []; adm_splade_pct = []; adm_fused = []; adm_support = []
    rej_dense_pct = []; rej_splade_pct = []; rej_fused = []; rej_support = []
    n_rec = 0; n_adm = 0
    for i, a, b in per_query_idx(d):
        yy = y[a:b]
        if yy.sum() == 0:
            continue
        f = F[a:b]; raw = s0_score_fn(f)
        sc = f[:, FUSED_COL] + s0_beta * _zq(raw)
        top = set(np.argsort(-sc, kind="stable")[:TOP_POOL].tolist())
        rr = np.where(recov[a:b] == 1)[0]; rr = [j for j in rr if yy[j] == 1]
        drk = dr[a:b]; srk = sr[a:b]
        for j in rr:
            n_rec += 1; c = cat(int(drk[j]), int(srk[j])); overall[c] += 1
            # rank percentile: 1 - rank/200 (1=best); absent(200)->0
            dp = 1.0 - min(int(drk[j]), TOP200) / TOP200; sp = 1.0 - min(int(srk[j]), TOP200) / TOP200
            fused = float(f[j, FUSED_COL]); supp = float(f[j, 3])
            if j in top:
                n_adm += 1; admitted[c] += 1
                adm_dense_pct.append(dp); adm_splade_pct.append(sp); adm_fused.append(fused); adm_support.append(supp)
            else:
                rej_dense_pct.append(dp); rej_splade_pct.append(sp); rej_fused.append(fused); rej_support.append(supp)
    def ms(x):
        return {"n": len(x), "mean": round(float(np.mean(x)), 4) if x else None,
                "p50": round(float(np.percentile(x, 50)), 4) if x else None} if True else None
    def frac(dic):
        t = max(sum(dic.values()), 1); return {k: round(v / t, 4) for k, v in dic.items()}
    return {"n_recovered_golds": n_rec, "n_recovered_admitted_by_S0": n_adm,
            "overall_decomposition_count": overall, "overall_decomposition_frac": frac(overall),
            "S0_admitted_decomposition_count": admitted, "S0_admitted_decomposition_frac": frac(admitted),
            "recovered_admitted": {"dense_rank_pct": ms(adm_dense_pct), "splade_rank_pct": ms(adm_splade_pct),
                                   "fused_rrf_pct": ms(adm_fused), "retriever_support_frac": ms(adm_support)},
            "recovered_rejected": {"dense_rank_pct": ms(rej_dense_pct), "splade_rank_pct": ms(rej_splade_pct),
                                   "fused_rrf_pct": ms(rej_fused), "retriever_support_frac": ms(rej_support)},
            "interpretation_note": ("Fraction of recovered golds present in ordinary Dense/SPLADE top-200 (A/B/C) "
                                    "vs graph-only (D). D-heavy admits => geometry is genuine DISCOVERY beyond "
                                    "retrieval; A/B/C-heavy admits => geometry surfaces candidates that ordinary "
                                    "retrieval also ranks, i.e. discovery is aided but VERIFIED by retrieval evidence.")}


# ------------------------------------------------------------------ identity leak diagnostic (analysis only)
def identity_diag(DATA, tag):
    Xs = []; ys = []; per = 40000
    for k, ds in enumerate(DSES):
        X = feat(DATA[ds], tag); idx = np.random.permutation(len(X))[:per]
        Xs.append(X[idx]); ys.append(np.full(len(idx), k))
    X = np.concatenate(Xs); yv = np.concatenate(ys)
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xn = (X - mu) / sd
    n, dd = Xn.shape; W = np.zeros((dd, 3)); b = np.zeros(3)
    for _ in range(300):
        z = Xn @ W + b; z -= z.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
        Y = np.eye(3)[yv]; g = (p - Y) / n
        W -= 0.5 * (Xn.T @ g + 1e-3 * W); b -= 0.5 * g.sum(0)
    acc = float((np.argmax(Xn @ W + b, 1) == yv).mean())
    return {"accuracy": round(acc, 4), "chance": round(1 / 3, 4),
            "per_feature_importance": [round(float(x), 3) for x in np.abs(W).sum(1)]}


# ------------------------------------------------------------------ inference-safety audit
BANNED = ["gold", "label", "dataset", "corpus", "supporting", "relation_type", "bracket", "answer", "target"]


def audit(DATA):
    viol = []
    for ds in DSES:
        d = DATA[ds]
        cols = [str(c) for b in ["S0", "S1OLD", "S2OLD", "S1NEW", "S2NEW"] for c in d[b + "_COLS"]]
        for c in cols:
            for bad in BANNED:
                if bad in c.lower():
                    viol.append(f"{ds}: banned '{bad}' in '{c}'")
        if len(d["qhop"]) != len(d["groups"]):
            viol.append(f"{ds}: qhop per-candidate (len {len(d['qhop'])} != nq {len(d['groups'])})")
        # dense_rank/splade_rank are DIAGNOSTIC channels — must NOT be reachable as model features
        for tag in LADDER:
            if feat(d, tag).shape[1] not in (5, 10, 16, 17):
                viol.append(f"{ds}:{tag} unexpected width {feat(d,tag).shape[1]}")
    ok = len(viol) == 0
    log(f"INFERENCE_SAFETY_AUDIT PASS={ok} violations={viol}")
    assert ok, viol
    return {"PASS": ok, "violations": viol}


# ------------------------------------------------------------------ main LODO
def main():
    log(f"=== B1.2 INVARIANCE-REPAIR LODO  datasets={DSES} TOP_POOL={TOP_POOL} ===")
    DATA = {ds: load(ds) for ds in DSES}
    AUD = audit(DATA)
    for ds in DSES:
        d = DATA[ds]
        log(f"{ds}: q={len(d['groups'])} cands={len(d['y'])} recovered={int(d['recov'].sum())} "
            f"widths={{S0:{feat(d,'S0').shape[1]}, S1old:{feat(d,'S1old').shape[1]}, S1new:{feat(d,'S1new').shape[1]}, "
            f"S2old:{feat(d,'S2old').shape[1]}, S2new:{feat(d,'S2new').shape[1]}}}")

    def build_train(train_ds, tag):
        pX = []; pY = []
        for dd in train_ds:
            Xs, Ys = hard_sample(DATA[dd], tag); pX.append(Xs); pY.append(Ys)
        m = min(len(a) for a in pY)
        X = np.concatenate([a[np.random.permutation(len(a))[:m]] for a in pX])
        Y = np.concatenate([a[np.random.permutation(len(a))[:m]] for a in pY])
        return X, Y

    def select_beta(train_ds, tag, score_fn):
        best = (0.0, -1e9)
        for beta in BETAS:
            nets = [summ(admit(DATA[dd], tag, score_fn, beta=beta))["NET_GOLD_GAIN@50"] for dd in train_ds]
            mn = float(np.mean(nets))
            if mn > best[1]:
                best = (beta, mn)
        return best[0]

    results = {}; s0_models = {}
    for target in DSES:
        train_ds = [x for x in DSES if x != target]
        row = {"train": train_ds, "BASE_FUSED": base_summary(DATA[target])}
        for tag in LADDER:
            Xtr, Ytr = build_train(train_ds, tag)
            mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
            w, b = fit_logreg((Xtr - mu) / sd, Ytr)
            score_fn = (lambda W, B, MU, SD: (lambda F: ((F - MU) / SD) @ W + B))(w, b, mu, sd)
            beta = select_beta(train_ds, tag, score_fn)
            s = summ(admit(DATA[target], tag, score_fn, beta=beta)); s["beta_train_selected"] = beta
            row[tag + "_linear"] = s
            if tag == "S0":
                s0_models[target] = (score_fn, beta)
            log(f"  {target} <- {tag}-linear(b={beta}): REC_ADM={s['RECOVERED_GOLD_ADMISSION@50']} "
                f"R@50={s['GOLD_RECALL@50']} RETAIN={s['ORIGINAL_GOLD_RETENTION@50']} NET={s['NET_GOLD_GAIN@50']} "
                f"churn={s['POOL_CHURN_mean']}")
        # FIRST DIAGNOSTIC with the winning (LODO-trained) S0 for this target
        sfn, sb = s0_models[target]
        row["FIRST_DIAGNOSTIC_S0"] = first_diagnostic(DATA[target], sfn, sb)
        fd = row["FIRST_DIAGNOSTIC_S0"]
        log(f"  {target} FIRST_DIAG: recovered={fd['n_recovered_golds']} S0admitted={fd['n_recovered_admitted_by_S0']} "
            f"overall={fd['overall_decomposition_frac']} admitted={fd['S0_admitted_decomposition_frac']}")
        results[target] = row

    # MLP only on ladders where the invariant linear showed useful transfer (>=0 on all three).
    def all_nonneg(tag):
        return all(results[t][tag + "_linear"]["NET_GOLD_GAIN@50"] >= 0 for t in DSES)
    mlp_tags = [t for t in ["S1new", "S2new"] if all_nonneg(t)]
    if HAVE_TORCH and mlp_tags:
        for target in DSES:
            train_ds = [x for x in DSES if x != target]
            for tag in mlp_tags:
                Xtr, Ytr = build_train(train_ds, tag); Xtr = Xtr.astype(np.float32); Ytr = Ytr.astype(np.float32)
                mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
                mdl = MLP(feat(DATA[target], tag).shape[1]).fit((Xtr - mu) / sd, Ytr)
                score_fn = (lambda MU, SD, M: (lambda F: M.score(((F - MU) / SD).astype(np.float32))))(mu, sd, mdl)
                beta = select_beta(train_ds, tag, score_fn)
                s = summ(admit(DATA[target], tag, score_fn, beta=beta)); s["beta_train_selected"] = beta
                results[target]["MLP_" + tag] = s
                log(f"  {target} <- MLP({tag},b={beta}): REC_ADM={s['RECOVERED_GOLD_ADMISSION@50']} "
                    f"R@50={s['GOLD_RECALL@50']} RETAIN={s['ORIGINAL_GOLD_RETENTION@50']} NET={s['NET_GOLD_GAIN@50']}")
    else:
        log(f"MLP skipped (invariant linear not non-negative on all three): candidates={mlp_tags}")

    iddiag = {tag: identity_diag(DATA, tag) for tag in LADDER}
    log(f"IDENTITY_DIAG " + json.dumps({k: v["accuracy"] for k, v in iddiag.items()}))

    contracts = {ds: json.load(open(f"scratchpad/_b12/{ds}_stat.json"))["compute_contract"] for ds in DSES}
    out = {"phase": "G2 B1.2 invariance-repair LODO; TEST untouched; STOP before six-way",
           "datasets": DSES, "TOP_POOL": TOP_POOL, "ladder": list(LADDER.keys()),
           "feature_separation": {
               "CANDIDATE_LOCAL": ["S0(all)", "S1new(all)", "S2new(all)", "min_hop", "seed_conn/support", "s_dir",
                                   "dir_exp_rank", "exphop", "struct_dist", "dir_support_max", "geometry_added"],
               "QUERY_GRAPH_REGIME": ["S1old.struct_support_pct", "S1old.degree_pct", "S1old.ppr_pct",
                                      "S2old.n_struct_anchors_pct"]},
           "leakage_checks": {"target_excluded_from_train": True, "train_only_normalization": True,
                              "no_dataset_id_feature": True, "diagnostic_channels_not_model_features": True},
           "INFERENCE_SAFETY_AUDIT": AUD, "GLOBAL_COMPUTE_CONTRACT": contracts,
           "RESULTS": results, "IDENTITY_LEAK_DIAGNOSTIC": iddiag,
           "NEW_ENCODER_FORWARD_PASSES": 0, "TARGET_TEST_TOUCHED": "NO"}
    os.makedirs("results/GENERALIZATION", exist_ok=True)
    json.dump(out, open("results/GENERALIZATION/_g2_b12_invariance.json", "w"), indent=1, default=str)
    log("B12_LODO_DONE -> results/GENERALIZATION/_g2_b12_invariance.json")


if __name__ == "__main__":
    main()
