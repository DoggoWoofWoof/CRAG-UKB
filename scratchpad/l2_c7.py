"""C7 — ADAPTIVE POLICY DISTILLATION + CANDIDATE-AWARE RELATION RESIDUAL GATING. VAL only, no TEST.

Two independent tracks over the FROZEN 5 experts (no new experts, no Shapley aux, no dropout):

  Track AB  distill the TRAIN oracle-archetype structure into an inference-safe query policy
    C7a  HARD    : classifier -> argmax archetype -> its frozen weight vector
    C7b  SOFT    : classifier probs p_k(q) -> w(q) = Σ_k p_k(q)·AW[k]
    C7b2 UTILITY : regress per-archetype TRAIN NDCG utility u_k(q) -> softmax(u/τ) mixture

  Track C   freeze BASE4 (dense+splade+offset+mixture, equal RRF); learn ONLY the relation multiplier
    score(q,d) = BASE4(q,d) + alpha·r_relation(q,d),  alpha = 2·sigmoid(z)  (init z=0 ⇒ alpha=1 ⇒ == C0)
    C7c0 alpha=1 (PARITY == C0)   C7c1 alpha(q)   C7c2 alpha(q,d)
    Structural mask is automatic: r_relation=0 on mask=0, so alpha·0=0 (never fabricates relation evidence).

Training objective (Track C) = C6c FULL-P50 TRUE-RANK ΔNDCG@50 LambdaRank (no sampled-list). Archetype
labels/utilities come from TRAIN ONLY; VAL oracle is ceiling/analysis only.
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import l2_shapley as SH
import l2_controller as CT
import l2_c6 as C6

DS = CT.DS; CORP = CT.CORP; CACHE = CT.CACHE; UKB = CT.UKB; OUT = "results/L2/_ctrl"; K0 = SH.K0; N = 5; KS = (1, 5, 10, 20, 50)
SCALE = CT.SCALE
# FROZEN archetype set (identical to C6 / the +0.0395 ceiling) — order = (dense,splade,offset,mixture,relation)
ARCH = {"equal": [1, 1, 1, 1, 1], "up_relation": [1, 1, 1, 1, 2.5], "up_offmix": [1, 1, 1.6, 1.6, 1],
        "up_splade": [1, 1.6, 1, 1, 1], "up_dense": [1.6, 1, 1, 1, 1], "down_relation": [1, 1, 1, 1, 0.4]}
ANAMES = list(ARCH); AWMAT = np.array([ARCH[a] for a in ANAMES], np.float32)   # (6,5)


# ============================================================ generic full-scope VAL evaluator
def eval_generic(caches, weight_fn, want_alpha=False):
    """weight_fn(ds, qi, r, rk, mask, hqv, scal) -> (score(n,), alpha(n,) or None).
    Full-scope metrics under COND_ANY_GOLD_IN_SCOPE. Returns per-ds + POOLED + _perq (+ _alpha if want_alpha)."""
    res = {}; perq = {}; alpha_rows = {ds: {"all": [], "gold": [], "nongold": [], "relrank": [], "mask": []} for ds in DS}
    for ds in DS:
        c = caches[ds]; H = _hq(ds)
        rr = []; rec = {K: [] for K in KS}; anyk = {K: [] for K in KS}; allk = {K: [] for K in KS}; ndcg_sum = 0.0
        pq = {k: [] for k in ("qi", "ndcg", "mrr", "all10", "all50", "best")}
        nqv = len(c["q_off"]) - 1
        for qi in range(nqv):
            a, b = int(c["q_off"][qi]), int(c["q_off"][qi + 1])
            if b == a: continue
            r = torch.from_numpy(c["r"][a:b].astype(np.float32)); rk = torch.from_numpy(c["rk"][a:b].astype(np.float32))
            mask = torch.from_numpy(c["mask"][a:b]); hqv = torch.from_numpy(H[c["row_all"][qi]]); scal = torch.from_numpy(c["qscalar"][qi])
            with torch.no_grad():
                s, alpha = weight_fn(ds, qi, r, rk, mask, hqv, scal)
            s = s.numpy()
            ga, gb = int(c["gold_off"][qi]), int(c["gold_off"][qi + 1]); gold = c["gold_loc"][ga:gb]
            order = np.argsort(-s, kind="stable"); rankof = np.empty(len(s), np.int64); rankof[order] = np.arange(len(s))
            gr = rankof[gold]; best = int(gr.min()); worst = int(gr.max()); ng = len(gold)
            rr.append(1.0 / (best + 1))
            for K in KS:
                rec[K].append((gr < K).sum() / ng); anyk[K].append(1.0 if best < K else 0.0); allk[K].append(1.0 if worst < K else 0.0)
            dcg = np.sum([1.0 / np.log2(x + 2) for x in gr if x < 50]); idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(ng, 50))])
            nd = (dcg / idcg) if idcg > 0 else 0.0; ndcg_sum += nd
            pq["qi"].append(qi); pq["ndcg"].append(nd); pq["mrr"].append(1.0 / (best + 1))
            pq["all10"].append(1.0 if worst < 10 else 0.0); pq["all50"].append(1.0 if worst < 50 else 0.0); pq["best"].append(best)
            if want_alpha and alpha is not None:
                al = alpha.numpy(); mk = c["mask"][a:b].astype(bool); elig = np.where(mk)[0]
                if len(elig):
                    gset = set(gold.tolist())
                    for li in elig:
                        alpha_rows[ds]["all"].append(float(al[li])); alpha_rows[ds]["mask"].append(1)
                        (alpha_rows[ds]["gold"] if li in gset else alpha_rows[ds]["nongold"]).append(float(al[li]))
                        alpha_rows[ds]["relrank"].append(float(rk[li, 4].item()))
        d = dict(n=len(rr), MRR=float(np.mean(rr)), **{f"R@{K}": float(np.mean(rec[K])) for K in KS},
                 **{f"ANY@{K}": float(np.mean(anyk[K])) for K in KS}, **{f"ALL@{K}": float(np.mean(allk[K])) for K in KS},
                 **{"NDCG@50": ndcg_sum / len(rr)})
        res[ds] = d; perq[ds] = {k: np.array(v) for k, v in pq.items()}
    pooled = {}
    for k in ("MRR", "NDCG@50", *[f"R@{K}" for K in KS], *[f"ANY@{K}" for K in KS], *[f"ALL@{K}" for K in KS]):
        pooled[k] = sum(res[ds][k] * res[ds]["n"] for ds in DS) / sum(res[ds]["n"] for ds in DS)
    pooled["n"] = sum(res[ds]["n"] for ds in DS); res["POOLED"] = pooled; res["_perq"] = perq
    if want_alpha: res["_alpha"] = alpha_rows
    return res


_HQ = {}
def _hq(ds):
    if ds not in _HQ:
        q = np.load(f"{UKB}/{ds}/gte_qwen/queries_all.npy").astype(np.float32); _HQ[ds] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    return _HQ[ds]


# ============================================================ Track AB: archetype distillation
def _load_feat(ds, split):
    z = np.load(f"{OUT}/_c6_feat_{ds}_{split}.npz"); return z["feats"], z["ndcg"], z["labels"], z["qi"]


def fit_archetype_heads(seed=0):
    """Return (clf, reg, mu, sd) trained on pooled TRAIN. clf=6-way logits; reg=6 utility outputs."""
    Xtr = np.concatenate([_load_feat(ds, "train")[0] for ds in DS]); ytr = np.concatenate([_load_feat(ds, "train")[2] for ds in DS])
    NDtr = np.concatenate([_load_feat(ds, "train")[1] for ds in DS])
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6; Xn = (Xtr - mu) / sd
    torch.manual_seed(seed)
    clf = nn.Linear(Xn.shape[1], len(ANAMES)); reg = nn.Linear(Xn.shape[1], len(ANAMES))
    cnt = np.bincount(ytr, minlength=len(ANAMES)); cw = torch.from_numpy((cnt.sum() / (len(ANAMES) * np.clip(cnt, 1, None))).astype(np.float32))
    Xt = torch.from_numpy(Xn); yt = torch.from_numpy(ytr); NDt = torch.from_numpy(NDtr)
    oc = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    orr = torch.optim.Adam(reg.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(300):
        oc.zero_grad(); F.cross_entropy(clf(Xt), yt, weight=cw).backward(); oc.step()
        orr.zero_grad(); F.mse_loss(reg(Xt), NDt).backward(); orr.step()
    return clf, reg, mu, sd


def val_weight_vectors(clf, reg, mu, sd, mode, tau=0.05):
    """Per-ds dict of full (nq,5) weight arrays (default equal for skipped qi) from the chosen policy."""
    wq = {}
    for ds in DS:
        Xva, _, _, qiva = _load_feat(ds, "val")
        Xn = torch.from_numpy(((Xva - mu) / sd).astype(np.float32))
        nq = len(np.load(f"{CORP}/{ds}/val/query_offsets.npy")) - 1
        W = np.tile(AWMAT[ANAMES.index("equal")], (nq, 1)).astype(np.float32)
        with torch.no_grad():
            if mode == "hard":
                pred = clf(Xn).argmax(1).numpy(); Wsel = AWMAT[pred]
            elif mode == "soft":
                p = F.softmax(clf(Xn), 1).numpy(); Wsel = p @ AWMAT
            elif mode == "utility":
                u = reg(Xn).numpy(); p = _softmax(u / tau); Wsel = p @ AWMAT
        W[qiva] = Wsel.astype(np.float32); wq[ds] = W
    return wq


def _softmax(x):
    x = x - x.max(1, keepdims=True); e = np.exp(x); return e / e.sum(1, keepdims=True)


def archetype_weight_fn(wq):
    def fn(ds, qi, r, rk, mask, hqv, scal):
        w = torch.from_numpy(wq[ds][qi]); return (w.unsqueeze(0) * r).sum(1), None
    return fn


# ============================================================ Track C: relation residual gating
class AlphaQ(nn.Module):        # C7c1: alpha(q) = 2·sigmoid(z(q)), init z=0 -> alpha=1
    def __init__(self):
        super().__init__(); self.proj = nn.Linear(1536, 32)
        self.net = nn.Sequential(nn.Linear(32 + 3, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def alpha_q(self, hqv, qrel):    # hqv(1536,), qrel(3,) = [relany, log1p_relnc, disagree]
        z = self.net(torch.cat([self.proj(hqv), qrel]))
        return 2.0 * torch.sigmoid(z).squeeze(-1)


class AlphaQD(nn.Module):       # C7c2: alpha(q,d) = 2·sigmoid(z(q,d)), init -> 1
    def __init__(self):
        super().__init__(); self.proj = nn.Linear(1536, 32)
        self.net = nn.Sequential(nn.Linear(32 + 3 + 9, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def alpha_qd(self, hqv, qrel, cf):   # cf (n,9)
        q = torch.cat([self.proj(hqv), qrel])                       # (35,)
        x = torch.cat([q.unsqueeze(0).expand(cf.shape[0], -1), cf], 1)
        z = self.net(x); return 2.0 * torch.sigmoid(z).squeeze(-1)  # (n,)


def _qrel(scal):    # inference-safe query relation diagnostics: [relany, log1p(relnc?), disagree]
    # scal = [disagree, relany, relfrac, jac_om, jac_do]; use relany, relfrac(proxy for count), disagree
    return torch.tensor([float(scal[1]), float(scal[2]), float(scal[0])], dtype=torch.float32)


def _cand_feats_rel(r, rk, mask):    # (n,9): relation r+rank+mask + 4 base ranks + (dense-splade) rankdiff
    return torch.stack([r[:, 4], rk[:, 4], mask.float(), rk[:, 0], rk[:, 1], rk[:, 2], rk[:, 3],
                        rk[:, 0] - rk[:, 1], rk[:, 2] - rk[:, 3]], 1)


def relation_score_fn(model, kind):
    """score = base4 + alpha·r_relation.  kind in {c0,c1,c2}."""
    def fn(ds, qi, r, rk, mask, hqv, scal):
        base4 = r[:, :4].sum(1); rel = r[:, 4]
        if kind == "c0":
            alpha = torch.ones_like(rel)
        elif kind == "c1":
            alpha = model.alpha_q(hqv, _qrel(scal)).expand(r.shape[0])
        else:
            alpha = model.alpha_qd(hqv, _qrel(scal), _cand_feats_rel(r, rk, mask))
        return base4 + alpha * rel, alpha
    return fn


def train_alpha(kind, epochs=8, lr=1e-3, seed=0, balance_relsig=True, log=print):
    """Full-P50 true-rank ΔNDCG@50 LambdaRank; only the alpha network trains (BASE4 frozen equal RRF)."""
    torch.manual_seed(seed); np.random.seed(seed)
    raw = {ds: C6._load_raw(ds, "train") for ds in DS}
    meta = {ds: np.load(f"{CACHE}/{ds}_traindepth.npz") for ds in DS}
    va = {ds: CT.load_cache(ds, "val") for ds in DS}
    model = AlphaQ() if kind == "c1" else AlphaQD()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # query lists; optionally emphasise relation-signal-present queries (inference-observable only)
    qlists = {}
    for ds in DS:
        val = meta[ds]["valid"]; relany = raw[ds] and None
        qs = [qi for qi in range(len(raw[ds]["query_offsets"]) - 1) if val[qi]]
        qlists[ds] = qs
    relsig = {ds: meta[ds]["qscalar"][:, 1] > 0.5 for ds in DS}    # relation_any_signal_present per qi
    def sel_key(p): return (round(p["NDCG@50"], 6), round(p["ALL@10"], 6), round(p["ALL@50"], 6))
    p0 = eval_generic(va, relation_score_fn(model, kind))["POOLED"]
    best = sel_key(p0); best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_ep = -1
    tb_key = None; tb_state = None
    hist = [{"epoch": -1, "val_pooled": {k: round(p0[k], 4) for k in ("MRR", "NDCG@50", "R@5", "ALL@10", "ALL@50")}}]
    log(f"[C7{kind}] ep-1(init/C0) NDCG50={p0['NDCG@50']:.4f} MRR={p0['MRR']:.4f} ALL@10={p0['ALL@10']:.4f} ALL@50={p0['ALL@50']:.4f}")
    nbal = min(len(qlists[ds]) for ds in DS); t0 = time.time()
    for ep in range(epochs):
        rng = np.random.default_rng(seed + ep)
        eq = []
        for ds in DS:
            pool = qlists[ds]
            if balance_relsig:                        # oversample relation-signal-present (zero-gradient otherwise)
                sig = [q for q in pool if relsig[ds][q]]; nos = [q for q in pool if not relsig[ds][q]]
                take_sig = rng.choice(sig, min(len(sig), nbal // 2), replace=False) if sig else np.array([], int)
                rem = nbal - len(take_sig); take_no = rng.choice(nos, min(len(nos), rem), replace=False) if nos else np.array([], int)
                sub = np.concatenate([take_sig, take_no]).astype(int)
            else:
                sub = rng.choice(pool, size=nbal, replace=False)
            eq += [(ds, int(q)) for q in sub]
        rng.shuffle(eq)
        model.train(); opt.zero_grad(); acc = 0.0; step = 0; B = 32
        for j, (ds, qi) in enumerate(eq):
            R = raw[ds]; s0, e0 = int(R["query_offsets"][qi]), int(R["query_offsets"][qi + 1]); n = e0 - s0
            lab = R["labels"][s0:e0]; gold = np.where(lab == 1)[0]
            if len(gold) == 0: continue
            C = SH.contribs(R["dense_score"][s0:e0], R["splade_scope_score"][s0:e0], R["offset_score"][s0:e0],
                            R["mixture_score"][s0:e0], R["relation_qwen_score"][s0:e0], R["relation_mask"][s0:e0])
            rk = np.empty((N, n), np.float32)
            for i in range(4): rk[i] = (1.0 / C[i]) - K0
            ie = np.where(R["relation_mask"][s0:e0])[0]; rk[4] = n
            if len(ie): rk[4][ie] = (1.0 / C[4][ie]) - K0
            rt = torch.from_numpy(C.T.astype(np.float32)); rkt = torch.from_numpy((rk.T / n).astype(np.float32))
            maskt = torch.from_numpy((R["relation_mask"][s0:e0] > 0)); scal = meta[ds]["qscalar"][qi]
            hqv = torch.from_numpy(_hq(ds)[int(meta[ds]["row_all"][qi])])
            base4 = rt[:, :4].sum(1); rel = rt[:, 4]
            if kind == "c1":
                alpha = model.alpha_q(hqv, _qrel(scal)).expand(n)
            else:
                alpha = model.alpha_qd(hqv, _qrel(scal), _cand_feats_rel(rt, rkt, maskt))
            s = base4 + alpha * rel
            order = torch.argsort(-s.detach(), stable=True).numpy(); rankof = np.empty(n, np.int64); rankof[order] = np.arange(n)
            D = np.where(rankof < 50, 1.0 / np.log2(rankof + 2.0), 0.0)
            ng = len(gold); idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(ng, 50))])
            neg = np.where(lab == 0)[0]; negb = neg[rankof[neg] < 50]
            if len(negb) == 0 or idcg <= 0: continue
            Wm = torch.from_numpy((np.abs(D[gold][:, None] - D[negb][None, :]) / idcg).astype(np.float32))
            if float(Wm.sum()) <= 0: continue
            diff = s[torch.from_numpy(gold)].unsqueeze(1) - s[torch.from_numpy(negb)].unsqueeze(0)
            L = (Wm * F.softplus(-SCALE * diff)).sum() / Wm.sum()
            (L / B).backward(); acc += float(L.detach()); step += 1
            if (j + 1) % B == 0: opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        p = eval_generic(va, relation_score_fn(model, kind))["POOLED"]; key = sel_key(p)
        hist.append({"epoch": ep, "train_loss": acc / max(step, 1), "val_pooled": {k: round(p[k], 4) for k in ("MRR", "NDCG@50", "R@5", "ALL@10", "ALL@50")}})
        log(f"[C7{kind}] ep{ep} loss={acc/max(step,1):.4f} NDCG50={p['NDCG@50']:.4f} MRR={p['MRR']:.4f} R@5={p['R@5']:.4f} ALL@10={p['ALL@10']:.4f} ALL@50={p['ALL@50']:.4f} ({time.time()-t0:.0f}s)")
        if key > best: best = key; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_ep = ep
        if tb_key is None or key > tb_key: tb_key = key; tb_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    tb = (AlphaQ() if kind == "c1" else AlphaQD()); tb.load_state_dict(tb_state); tb.eval()
    trained_best = max((h for h in hist if h["epoch"] >= 0), key=lambda h: h["val_pooled"]["NDCG@50"], default=None)
    return model, {"kind": f"C7{kind}", "epochs": epochs, "lr": lr, "best_sel": best, "selected_epoch": best_ep,
                   "selected_is_init": best_ep == -1, "trained_best_epoch": trained_best["epoch"] if trained_best else None,
                   "trained_best_ndcg": trained_best["val_pooled"]["NDCG@50"] if trained_best else None, "history": hist}, tb
