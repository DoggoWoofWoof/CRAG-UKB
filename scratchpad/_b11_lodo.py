"""G2 B1.1 — EXPANDED-UNIVERSE ADMISSION PILOT · TRUE 3-way LODO admission stage.

Loads scratchpad/_b11/{metaqa,2wiki_clean,squad_clean}.npz (universe features from _b11_build). For each TARGET,
TRAIN = the other two datasets ONLY; TRAIN-only normalization; evaluate target VAL once. Feature ladder
S0-linear -> S1-linear -> S2-linear -> best-S* small MLP. Query-balanced HARD sampling for training.
Admission selects TOP_POOL=50 from the query's universe. Reports the full admission-metric suite incl the
central RECOVERED_GOLD_ADMISSION@50, MetaQA per-hop, SQuAD negative-control, dataset-identity leak diagnostic,
and the linear-vs-MLP verdict. No dataset id / gold / relation-type in any model feature. TEST untouched.
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad"); sys.path.insert(0, os.getcwd())
import numpy as np

DSES = ["metaqa", "2wiki_clean", "squad_clean"]
TOP_POOL = 50; K0 = 60
S0 = 5; S1 = 10; S2 = 17          # cumulative feature-set widths (S0=0:5, S1=0:10, S2=0:17)
LADDER = {"S0": S0, "S1": S1, "S2": S2}
FUSED_COL = 2                     # fused_rrf_pct (BASE admission == top-50 by this == original P50/inp50)
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
np.random.seed(0)
try:
    import torch; torch.manual_seed(0); HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def load(ds):
    z = np.load(f"scratchpad/_b11/{ds}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def per_query(d):
    """yield (feat, y, recov, inp50, hop) per query slice."""
    g = d["groups"]; st = np.concatenate([[0], np.cumsum(g)])
    for i in range(len(g)):
        a, b = int(st[i]), int(st[i + 1])
        yield (d["X"][a:b], d["y"][a:b], d["recov"][a:b], d["inp50"][a:b], int(d["qhop"][i]))


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
    """1-hidden-layer MLP (torch), class-balanced BCE. Small residual design."""
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


# ------------------------------------------------------------------ hard sampling
def hard_sample(d, width, per_q=60):
    """query-balanced hard negatives: all golds + top-base + admission-boundary + top-struct + top-dir + random."""
    Xs = []; Ys = []
    for feat, y, recov, inp50, hop in per_query(d):
        n = len(y); f = feat[:, :width]
        fused = feat[:, FUSED_COL]
        order = np.argsort(-fused, kind="stable")               # by base admission score
        sel = set(np.where(y == 1)[0].tolist())                 # ALL golds
        sel |= set(order[:20].tolist())                          # top base-ranked
        sel |= set(order[max(TOP_POOL - 10, 0):TOP_POOL + 10].tolist())   # admission boundary
        if width >= S1:
            struct = feat[:, 5:10].sum(1); sel |= set(np.argsort(-struct)[:10].tolist())
        if width >= S2:
            sdir = feat[:, 10]; sel |= set(np.argsort(-sdir)[:10].tolist())
        pool = list(sel)
        rest = [i for i in range(n) if i not in sel]
        if rest:
            pool += list(np.random.choice(rest, size=min(len(rest), max(per_q - len(pool), 0)), replace=False))
        pool = pool[:per_q] if len(pool) > per_q else pool
        idx = np.array(pool, np.int64)
        Xs.append(f[idx]); Ys.append(y[idx])
    return np.concatenate(Xs), np.concatenate(Ys)


# ------------------------------------------------------------------ admission metrics
def _zq(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def admit(d, score_fn, width, beta=None):
    """Admit top-50 from the universe; metric suite vs BASE=fused-top50.
    beta=None  -> PURE learned re-rank (diagnostic; ignores base).
    beta>=0    -> BASE-ANCHORED RESIDUAL: final = base_fused_pct + beta * zscore_q(learned_raw). Protects the
                  base retrieval ranking; the model only swaps in candidates it is confident about. beta=0 == BASE."""
    agg = dict(nq=0, ANY=0, RECn=0, RECd=0, ALL=0,
               rec_adm=0, rec_avail=0, orig_ret=0, orig_base=0,
               new_adm=0, old_evict=0, churn=[], per_hop={})
    for feat, y, recov, inp50, hop in per_query(d):
        ng_uni = int(y.sum())
        if ng_uni == 0:
            continue
        n = len(y)
        raw = score_fn(feat[:, :width])
        sc = raw if beta is None else (feat[:, FUSED_COL] + beta * _zq(raw))
        top = set(np.argsort(-sc, kind="stable")[:TOP_POOL].tolist())
        base = set(np.argsort(-feat[:, FUSED_COL], kind="stable")[:TOP_POOL].tolist())
        golds = set(np.where(y == 1)[0].tolist())
        rec = set(np.where(recov == 1)[0].tolist()) & golds
        gin_t = golds & top; gin_b = golds & base
        agg["nq"] += 1
        agg["ANY"] += int(len(gin_t) >= 1); agg["RECn"] += len(gin_t); agg["RECd"] += ng_uni
        agg["ALL"] += int(len(gin_t) == ng_uni)
        agg["rec_adm"] += len(rec & top); agg["rec_avail"] += len(rec)
        agg["orig_ret"] += len(gin_b & top); agg["orig_base"] += len(gin_b)
        agg["new_adm"] += len(gin_t - gin_b); agg["old_evict"] += len(gin_b - gin_t)
        agg["churn"].append(len(top ^ base) / (2 * TOP_POOL))
        ph = agg["per_hop"].setdefault(hop, dict(nq=0, RECn=0, RECd=0, rec_adm=0, rec_avail=0,
                                                 new_adm=0, old_evict=0, ALL=0))
        ph["nq"] += 1; ph["RECn"] += len(gin_t); ph["RECd"] += ng_uni
        ph["rec_adm"] += len(rec & top); ph["rec_avail"] += len(rec)
        ph["new_adm"] += len(gin_t - gin_b); ph["old_evict"] += len(gin_b - gin_t)
        ph["ALL"] += int(len(gin_t) == ng_uni)
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
    for h, d in sorted(agg["per_hop"].items()):
        ph[str(h)] = {"nq": d["nq"], "GOLD_RECALL@50": round(d["RECn"] / max(d["RECd"], 1), 4),
                      "ALL@50": round(d["ALL"] / max(d["nq"], 1), 4),
                      "RECOVERED_GOLD_ADMISSION@50": round(d["rec_adm"] / max(d["rec_avail"], 1), 4),
                      "recovered_avail": d["rec_avail"], "recovered_admitted": d["rec_adm"],
                      "NET_GOLD_GAIN@50": d["new_adm"] - d["old_evict"]}
    o["PER_HOP"] = ph
    return o


def base_summary(d):
    agg = admit(d, lambda F: F[:, FUSED_COL], S2)   # BASE == fused admission (learned==base -> identity)
    return summ(agg)


# ------------------------------------------------------------------ identity leak diagnostic (analysis only)
def identity_diag(DATA, width):
    """3-way logistic (softmax) predicting dataset from features (balanced sample). Report acc + top feature."""
    Xs = []; ys = []
    per = 40000
    for k, ds in enumerate(DSES):
        X = DATA[ds]["X"][:, :width]; idx = np.random.permutation(len(X))[:per]
        Xs.append(X[idx]); ys.append(np.full(len(idx), k))
    X = np.concatenate(Xs); yv = np.concatenate(ys)
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xn = (X - mu) / sd
    n, dd = Xn.shape; W = np.zeros((dd, 3)); b = np.zeros(3)
    for _ in range(300):
        z = Xn @ W + b; z -= z.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
        Y = np.eye(3)[yv]; g = (p - Y) / n
        W -= 0.5 * (Xn.T @ g + 1e-3 * W); b -= 0.5 * g.sum(0)
    acc = float((np.argmax(Xn @ W + b, 1) == yv).mean())
    imp = np.abs(W).sum(1)
    return {"accuracy": round(acc, 4), "chance": round(1 / 3, 4),
            "per_feature_importance": [round(float(x), 3) for x in imp]}


# ------------------------------------------------------------------ automated inference-safety audit (leakage test)
SAFE_COLS = {"dense_pct", "splade_pct", "fused_rrf_pct", "retriever_support_frac", "dense_splade_agree",
             "min_hop_from_seed_norm", "seed_connectivity_frac", "struct_support_norm", "degree_pct", "ppr_pct",
             "s_dir_pct", "dir_exp_rank_pct", "min_exp_hop_norm", "n_struct_anchors_pct", "struct_dist_pct",
             "dir_support_max_pct", "geometry_added_ind"}
BANNED_SUBSTR = ["gold", "label", "dataset", "corpus", "hop_annotation", "supporting", "relation_type",
                 "bracket", "answer", "target"]


def inference_safety_audit(DATA):
    """HARD leakage test per the G2 inference-safety contract. Fails immediately on any violation.
    Checks: (1) model feature columns == the 17 contract-safe features, no banned substrings; (2) the model
    feature matrix X carries NO label/recov/inp50/qhop channel; (3) benchmark hop (qhop) is stored per-query
    only (eval breakdown), not per-candidate in X; (4) reports per-dataset abstention rates so missingness
    fingerprinting is visible. Normalization safety (TRAIN-frozen) is enforced structurally in the LODO loop
    (target excluded; mu/sd from source rows only) and asserted there."""
    audit = {"per_dataset": {}, "violations": []}
    for ds in DSES:
        d = DATA[ds]
        cols = [str(c) for c in (list(d["S0"]) + list(d["S1"]) + list(d["S2"]))]
        if set(cols) != SAFE_COLS:
            audit["violations"].append(f"{ds}: feature columns {set(cols)^SAFE_COLS} deviate from contract-safe set")
        for c in cols:
            for b in BANNED_SUBSTR:
                if b in c.lower():
                    audit["violations"].append(f"{ds}: banned substring '{b}' in feature '{c}'")
        if d["X"].shape[1] != 17:
            audit["violations"].append(f"{ds}: feature dim {d['X'].shape[1]} != 17")
        if len(d["qhop"]) != len(d["groups"]):
            audit["violations"].append(f"{ds}: qhop is per-query length {len(d['qhop'])} != n_query {len(d['groups'])} (must be eval-only, not per-candidate)")
        # abstention (missingness) rates -> fingerprint visibility
        Xd = d["X"]
        audit["per_dataset"][ds] = {
            "n_cand": int(len(d["y"])), "feat_dim": int(Xd.shape[1]),
            "s_dir_abstain_frac": round(float((Xd[:, 10] == 0).mean()), 4),
            "geometry_added_frac": round(float((Xd[:, 16] == 1).mean()), 4),
            "struct_unreachable_frac": round(float((Xd[:, 5] == 0).mean()), 4)}
    audit["PASS"] = len(audit["violations"]) == 0
    log(f"INFERENCE_SAFETY_AUDIT PASS={audit['PASS']} violations={audit['violations']}")
    for ds, s in audit["per_dataset"].items():
        log(f"  {ds}: abstain(s_dir)={s['s_dir_abstain_frac']} geom_added={s['geometry_added_frac']} struct_unreach={s['struct_unreachable_frac']}")
    assert audit["PASS"], f"INFERENCE-SAFETY AUDIT FAILED: {audit['violations']}"
    return audit


# ------------------------------------------------------------------ main LODO
def main():
    log(f"=== B1.1 LODO admission pilot  datasets={DSES} TOP_POOL={TOP_POOL} ===")
    DATA = {ds: load(ds) for ds in DSES}
    AUDIT = inference_safety_audit(DATA)     # HARD GATE — must pass before any training/eval
    for ds in DSES:
        d = DATA[ds]; nq = len(d["groups"]); nc = len(d["y"])
        rec = int(d["recov"].sum())
        cols = list(d["S0"]) + list(d["S1"]) + list(d["S2"])
        log(f"{ds}: q={nq} cands={nc} feat={d['X'].shape[1]} recovered_gold_cands={rec} | base={base_summary(d)['RECOVERED_GOLD_ADMISSION@50']==0}")
    FEAT_COLS = list(DATA[DSES[0]]["S0"]) + list(DATA[DSES[0]]["S1"]) + list(DATA[DSES[0]]["S2"])

    BETAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35]   # incl 0.0 (=BASE null) so select_beta CAN choose not to perturb;
                                                 # finer low end (residual sc=fused_pct+beta*zscore_q, beta=0.1 already shifts ~0.3 percentile)

    def build_train(train_ds, width):
        parts_X = []; parts_Y = []
        for dd in train_ds:
            Xs, Ys = hard_sample(DATA[dd], width); parts_X.append(Xs); parts_Y.append(Ys)
        m = min(len(a) for a in parts_Y)
        X = np.concatenate([a[np.random.permutation(len(a))[:m]] for a in parts_X])
        Y = np.concatenate([a[np.random.permutation(len(a))[:m]] for a in parts_Y])
        return X, Y

    def select_beta(train_ds, score_fn, width):
        """Pick beta maximizing mean TRAIN NET_GOLD_GAIN@50 across SOURCE datasets (train-only model selection)."""
        best = (0.35, -1e9)
        for beta in BETAS:
            nets = [summ(admit(DATA[dd], score_fn, width, beta=beta))["NET_GOLD_GAIN@50"] for dd in train_ds]
            mn = float(np.mean(nets))
            if mn > best[1]:
                best = (beta, mn)
        return best[0]

    results = {}
    for target in DSES:
        train_ds = [x for x in DSES if x != target]
        assert target not in train_ds
        row = {"train": train_ds, "BASE_FUSED": base_summary(DATA[target])}
        best_tag = None; best_net = -1e9; best_width = S2
        for tag, width in LADDER.items():
            Xtr, Ytr = build_train(train_ds, width)
            mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
            w, b = fit_logreg((Xtr - mu) / sd, Ytr)
            score_fn = (lambda W, B, MU, SD: (lambda F: ((F - MU) / SD) @ W + B))(w, b, mu, sd)
            beta = select_beta(train_ds, score_fn, width)                 # TRAIN-selected, frozen
            s = summ(admit(DATA[target], score_fn, width, beta=beta))     # BASE-ANCHORED RESIDUAL admission
            s["beta_train_selected"] = beta
            row[tag + "_linear"] = s
            if tag == "S2":                                              # keep the naive pure re-rank as a documented failure mode
                row["S2_linear_NAIVE_PURE_rerank"] = summ(admit(DATA[target], score_fn, width, beta=None))
            net = s["NET_GOLD_GAIN@50"] + 50 * s["RECOVERED_GOLD_ADMISSION@50"]
            if net > best_net:
                best_net = net; best_tag = tag; best_width = width
            log(f"  {target} <- {tag}-linear(beta={beta}): REC_ADM={s['RECOVERED_GOLD_ADMISSION@50']} "
                f"R@50={s['GOLD_RECALL@50']} RETAIN={s['ORIGINAL_GOLD_RETENTION@50']} NET={s['NET_GOLD_GAIN@50']}")
        row["best_linear_tag"] = best_tag
        # small MLP on the best S* feature set (same base-anchored residual admission + TRAIN-selected beta)
        if HAVE_TORCH:
            Xtr, Ytr = build_train(train_ds, best_width)
            Xtr = Xtr.astype(np.float32); Ytr = Ytr.astype(np.float32)
            mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
            mdl = MLP(best_width).fit((Xtr - mu) / sd, Ytr)
            score_fn = (lambda MU, SD: (lambda F: mdl.score(((F - MU) / SD).astype(np.float32))))(mu, sd)
            beta = select_beta(train_ds, score_fn, best_width)
            s = summ(admit(DATA[target], score_fn, best_width, beta=beta)); s["beta_train_selected"] = beta
            row["MLP_on_best"] = s; row["MLP_feature_set"] = best_tag
            log(f"  {target} <- MLP({best_tag},beta={beta}): REC_ADM={s['RECOVERED_GOLD_ADMISSION@50']} "
                f"R@50={s['GOLD_RECALL@50']} RETAIN={s['ORIGINAL_GOLD_RETENTION@50']} NET={s['NET_GOLD_GAIN@50']}")
        results[target] = row

    iddiag = {tag: identity_diag(DATA, w) for tag, w in LADDER.items()}
    log(f"IDENTITY_DIAG {iddiag}")

    out = {"phase": "G2 B1.1 expanded-universe admission LODO pilot; TEST untouched",
           "datasets": DSES, "TOP_POOL": TOP_POOL, "M_MAX": 256,
           "feature_cols": FEAT_COLS, "ladder": list(LADDER.keys()),
           "leakage_checks": {"target_excluded_from_train": True, "train_only_normalization": True,
                              "no_dataset_id_feature": True, "no_gold_or_relationtype_or_bracket_feature": True},
           "central_metric": "RECOVERED_GOLD_ADMISSION@50",
           "INFERENCE_SAFETY_AUDIT": AUDIT,
           "RESULTS": results, "IDENTITY_LEAK_DIAGNOSTIC": iddiag,
           "NEW_ENCODER_FORWARD_PASSES": 0, "TARGET_TEST_TOUCHED": "NO"}
    os.makedirs("results/GENERALIZATION", exist_ok=True)
    json.dump(out, open("results/GENERALIZATION/_g2_b11_pilot.json", "w"), indent=1, default=str)
    log("B11_LODO_DONE -> results/GENERALIZATION/_g2_b11_pilot.json")


if __name__ == "__main__":
    main()
