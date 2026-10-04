"""C6 — FULL-SCOPE-AWARE CONTROLLER TRAINING (query-only gate).

Isolates whether the C0-C5 collapse was a TRAIN candidate-distribution / objective mismatch, NOT another
architecture search. Architecture is FROZEN to the C2 QueryGate (query embedding + inference-available query
diagnostics -> small MLP -> 5 normalized positive weights w_i(q) -> weighted frozen-expert RRF contributions).

Two controlled knobs over the OLD C2 (CT.train("C2")):
  Task 2  depth-aware negatives   : train candidates = ALL positives + retained hard negs + explicit
                                    C0-full-scope depth negatives (TOP 0-9 / MID 10-19 / DEEP 20-49 /
                                    NEARTAIL 50-99 / TAIL 100+), stratified, ~128-256 negs/query.
  Task 6  NDCG@50 LambdaRank loss : each (gold,neg) pair weighted by |ΔNDCG@50| of swapping them (K=50,
                                    all golds equal relevance). Bigger gradient to errors that move the
                                    actual selection metric; pairs entirely below rank 50 get ~0 weight.

  C6a = C2 arch + depth-aware negatives + RankNet      (isolates SAMPLING)
  C6b = C2 arch + depth-aware negatives + LambdaRank   (adds OBJECTIVE)
  C6c = optional full-P50-scope training (only if C6b fails and runtime is reasonable)

lambda_shapley = 0 and expert_dropout = 0 for C6a/C6b (do not mix into the causal test). Init-inclusive VAL
selection (primary NDCG@50, tie ALL@10/ALL@50, monitor MRR); record BOTH best-trained and selected epoch.
VAL is always the FULL P50 scope (CT val caches). No TEST.
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch, torch.nn.functional as F
import l2_shapley as SH
import l2_controller as CT

DS = CT.DS; CORP = CT.CORP; CACHE = CT.CACHE; EXPERTS = CT.EXPERTS; N = CT.N; K0 = SH.K0; SCALE = CT.SCALE
PROV_CODE = {"POSITIVE": 0, "DENSE_HARD": 1, "SPLADE_HARD": 2, "DENSE_SPLADE_DISAGREEMENT": 3,
             "HIGH_PARTITION_VOTE": 4, "SAME_PARTITION": 5, "RANDOM": 6,
             "C0_TOP_NEG": 7, "C0_MID_NEG": 8, "C0_DEEP_NEG": 9, "C0_NEARTAIL_NEG": 10, "C0_TAIL_NEG": 11}
CODE_PROV = {v: k for k, v in PROV_CODE.items()}
# depth strata on the C0 full-scope rank (0-indexed) and per-query quotas (emphasis on DEEP + NEARTAIL,
# the ranks that decide NDCG@50 / ALL@50). Retained hard negs are capped separately.
STRATA = [("C0_TOP_NEG", 0, 9, 16), ("C0_MID_NEG", 10, 19, 16), ("C0_DEEP_NEG", 20, 49, 48),
          ("C0_NEARTAIL_NEG", 50, 99, 32), ("C0_TAIL_NEG", 100, 10**9, 16)]
RETAIN_PROV = ("DENSE_HARD", "SPLADE_HARD", "DENSE_SPLADE_DISAGREEMENT")
RETAIN_CAP = 48


# ------------------------------------------------------------------ Task 2: depth-aware negative cache
def build_depth_cache(ds, log=print):
    """Write scratchpad/_ctrl_cache/{ds}_traindepth.npz in CT.build_cache schema (+ prov codes).
    Candidates per query = all positives + up to RETAIN_CAP retained hard negs + C0-depth-stratified negs."""
    d = f"{CORP}/{ds}/train"
    off = np.load(f"{d}/query_offsets.npy"); labels = np.load(f"{d}/labels.npy")
    dense = np.load(f"{d}/dense_score.npy"); splade = np.load(f"{d}/splade_scope_score.npy")
    offs = np.load(f"{d}/offset_score.npy"); mix = np.load(f"{d}/mixture_score.npy")
    rels = np.load(f"{d}/relation_qwen_score.npy"); relm = np.load(f"{d}/relation_mask.npy")
    M = np.load(f"{d}/expert_meta.npz")
    disagree = M["dense_splade_disagreement"].astype(np.float32); relany = M["relation_any_signal_present"].astype(np.float32)
    row_all = M["row_all"].astype(np.int64)
    src = json.load(open(f"{d}/train_sub_source.json")); tso = np.load(f"{d}/train_sub_offsets.npy"); tsl = np.load(f"{d}/train_sub_local.npy")
    nq = len(off) - 1
    r_rows, rk_rows, mask_rows, lab_rows, prov_rows = [], [], [], [], []
    q_off = [0]; qscalar = np.zeros((nq, 5), np.float32); valid = np.zeros(nq, bool)
    gold_off = [0]; gold_loc = []; t0 = time.time(); tot_neg = 0
    for qi in range(nq):
        s, e = int(off[qi]), int(off[qi + 1]); n = e - s
        lab = labels[s:e]; gl = np.where(lab == 1)[0]
        C = SH.contribs(dense[s:e], splade[s:e], offs[s:e], mix[s:e], rels[s:e], relm[s:e])  # (5,n)
        rk = np.empty((N, n), np.float32)
        for i in range(4): rk[i] = (1.0 / C[i]) - K0
        ie = np.where(relm[s:e])[0]; rk[4] = n
        if len(ie): rk[4][ie] = (1.0 / C[4][ie]) - K0
        relfrac = float(len(ie)) / n
        qscalar[qi] = [disagree[qi], relany[qi], relfrac, CT._jac_topk(rk[2], rk[3]), CT._jac_topk(rk[0], rk[2])]
        if len(gl) == 0:
            q_off.append(q_off[-1]); gold_off.append(gold_off[-1]); continue
        valid[qi] = True
        c0 = C.sum(0); order = np.argsort(-c0, kind="stable"); rankof = np.empty(n, np.int64); rankof[order] = np.arange(n)
        rng = np.random.default_rng(1000 + qi)
        chosen = {}                                    # local idx -> prov code (positives added first)
        for li in gl: chosen[int(li)] = 0              # POSITIVE
        # retained hard negs (from the canonical sub), capped
        a, b = int(tso[qi]), int(tso[qi + 1]); loc = tsl[a:b]; prov = src[a:b]
        provmap = {int(li): pv for li, pv in zip(loc, prov)}
        ret = [int(li) for li, pv in zip(loc, prov) if pv in RETAIN_PROV and lab[int(li)] == 0]
        if len(ret) > RETAIN_CAP: ret = list(rng.choice(ret, RETAIN_CAP, replace=False))
        for li in ret: chosen[li] = PROV_CODE[provmap[li]]   # record TRUE provenance of retained hard negs
        # C0-depth-stratified negatives (negatives only, not already chosen)
        neg_mask = (lab == 0)
        for name, lo, hi, quota in STRATA:
            band = np.where(neg_mask & (rankof >= lo) & (rankof <= hi))[0]
            band = np.array([x for x in band if int(x) not in chosen], np.int64)
            if len(band) == 0: continue
            take = band if len(band) <= quota else rng.choice(band, quota, replace=False)
            for li in take: chosen[int(li)] = PROV_CODE[name]
        sel = np.array(sorted(chosen.keys()), np.int64); pv = np.array([chosen[int(x)] for x in sel], np.int8)
        Csel = C[:, sel].T.astype(np.float16); rksel = (rk[:, sel].T / n).astype(np.float16)
        masksel = (relm[s:e][sel]).astype(bool); labsel = lab[sel].astype(np.int8)
        r_rows.append(Csel); rk_rows.append(rksel); mask_rows.append(masksel); lab_rows.append(labsel); prov_rows.append(pv)
        selpos = np.where(labsel == 1)[0]; gold_loc.append(selpos.astype(np.int64)); gold_off.append(gold_off[-1] + len(selpos))
        q_off.append(q_off[-1] + len(sel)); tot_neg += int((labsel == 0).sum())
        if (qi + 1) % 3000 == 0: log(f"  [{ds}/depth] {qi+1}/{nq} {time.time()-t0:.0f}s")
    out = dict(r=np.concatenate(r_rows), rk=np.concatenate(rk_rows), mask=np.concatenate(mask_rows),
               lab=np.concatenate(lab_rows), prov=np.concatenate(prov_rows),
               q_off=np.array(q_off, np.int64), qscalar=qscalar, valid=valid, row_all=row_all,
               gold_loc=np.concatenate(gold_loc), gold_off=np.array(gold_off, np.int64))
    np.savez(f"{CACHE}/{ds}_traindepth.npz", **out)
    nqv = int(valid.sum()); log(f"SAVED depth cache {ds} nq_valid={nqv} rows={len(out['r'])} avg_neg/q={tot_neg/nqv:.1f} {time.time()-t0:.0f}s")
    return out


def depth_provenance_summary():
    from collections import Counter
    summ = {}
    for ds in DS:
        z = np.load(f"{CACHE}/{ds}_traindepth.npz"); pv = z["prov"]; lab = z["lab"]
        negc = Counter(CODE_PROV[int(c)] for c in pv[lab == 0])
        q_off = z["q_off"]; nqv = int(z["valid"].sum())
        summ[ds] = {"n_queries_valid": nqv, "rows": int(len(pv)), "avg_cands_per_q": round(len(pv) / nqv, 1),
                    "avg_neg_per_q": round(int((lab == 0).sum()) / nqv, 1), "neg_provenance": dict(negc)}
    return summ


# ------------------------------------------------------------------ Task 6: NDCG@50 LambdaRank loss
def _lambda_weights(s_np, gold_idx, neg_idx, K=50):
    """|ΔNDCG@K| for each (gold,neg) swap, using CURRENT ranks within the candidate list. Detached numpy.
    All golds equal relevance; discount D(r)=1/log2(r+2) truncated to 0 beyond rank K; IDCG over all golds."""
    n = len(s_np); order = np.argsort(-s_np, kind="stable"); rankof = np.empty(n, np.int64); rankof[order] = np.arange(n)
    D = np.where(rankof < K, 1.0 / np.log2(rankof + 2.0), 0.0)     # (n,)
    ng = len(gold_idx); idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(ng, K))])
    if idcg <= 0: return None
    dg = D[gold_idx]; dn = D[neg_idx]
    return np.abs(dg[:, None] - dn[None, :]) / idcg               # (|G|,|neg|)


def loss_query_c6(model, r, rk, mask, hqv, scal, gold, neg, loss_kind):
    s, _ = CT.score_query(model, "C2", r, rk, mask, hqv, scal)
    diff = s[gold].unsqueeze(1) - s[neg].unsqueeze(0)             # (|G|,|neg|)
    if loss_kind == "ranknet":
        return F.softplus(-SCALE * diff).mean()
    if loss_kind == "lambdarank":
        w = _lambda_weights(s.detach().numpy(), gold.numpy(), neg.numpy(), K=50)
        if w is None: return None
        W = torch.from_numpy(w.astype(np.float32)); denom = W.sum()
        if float(denom) <= 0: return None                         # all pairs below rank 50 this step
        return (W * F.softplus(-SCALE * diff)).sum() / denom
    raise ValueError(loss_kind)


# ------------------------------------------------------------------ trainer (query-only, init-inclusive)
def load_named(ds, name):
    z = np.load(f"{CACHE}/{ds}_{name}.npz"); return {k: z[k] for k in z.files}


def train_c6(tag, loss_kind, train_name="traindepth", epochs=10, lr=3e-4, seed=0, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = {ds: load_named(ds, train_name) for ds in DS}
    va = {ds: CT.load_cache(ds, "val") for ds in DS}
    model = CT.QueryGate()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    qlists = {ds: [qi for qi in range(len(tr[ds]["q_off"]) - 1) if tr[ds]["q_off"][qi + 1] > tr[ds]["q_off"][qi]] for ds in DS}
    nbal = min(len(qlists[ds]) for ds in DS)
    def sel_key(p): return (round(p["NDCG@50"], 6), round(p["ALL@10"], 6), round(p["ALL@50"], 6))
    p0 = CT.evaluate(model, "C2", va)["POOLED"]
    best = sel_key(p0); best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_ep = -1
    tb_key = None; tb_state = None; tb_ep = None      # best TRAINED epoch (>=0), regardless of init floor
    hist = [{"epoch": -1, "train_loss": None, "val_pooled": {k: round(p0[k], 4) for k in ("MRR", "NDCG@50", "R@5", "ALL@10", "ALL@50")}}]
    log(f"[{tag}] ep-1(init/C0) NDCG50={p0['NDCG@50']:.4f} MRR={p0['MRR']:.4f} ALL@10={p0['ALL@10']:.4f} ALL@50={p0['ALL@50']:.4f}")
    t0 = time.time()
    for ep in range(epochs):
        rng = np.random.default_rng(seed + ep)
        epoch_q = [(ds, qi) for ds in DS for qi in rng.choice(qlists[ds], size=nbal, replace=False)]; rng.shuffle(epoch_q)
        model.train(); opt.zero_grad(); acc = 0.0; step = 0; B = 32
        for j, (ds, qi) in enumerate(epoch_q):
            c = tr[ds]; a, b = int(c["q_off"][qi]), int(c["q_off"][qi + 1])
            r = torch.from_numpy(c["r"][a:b].astype(np.float32)); rk = torch.from_numpy(c["rk"][a:b].astype(np.float32))
            mask = torch.from_numpy(c["mask"][a:b]); hqv = torch.from_numpy(CT.hq(ds)[c["row_all"][qi]]); scal = torch.from_numpy(c["qscalar"][qi])
            ga, gb = int(c["gold_off"][qi]), int(c["gold_off"][qi + 1]); gold = torch.from_numpy(c["gold_loc"][ga:gb])
            neg = torch.from_numpy(np.where(c["lab"][a:b] == 0)[0])
            if len(neg) == 0: continue
            L = loss_query_c6(model, r, rk, mask, hqv, scal, gold, neg, loss_kind)
            if L is None: continue
            (L / B).backward(); acc += float(L); step += 1
            if (j + 1) % B == 0: opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        p = CT.evaluate(model, "C2", va)["POOLED"]; key = sel_key(p)
        hist.append({"epoch": ep, "train_loss": acc / max(step, 1), "val_pooled": {k: round(p[k], 4) for k in ("MRR", "NDCG@50", "R@5", "ALL@10", "ALL@50")}})
        log(f"[{tag}] ep{ep} loss={acc/max(step,1):.4f} NDCG50={p['NDCG@50']:.4f} MRR={p['MRR']:.4f} R@5={p['R@5']:.4f} ALL@10={p['ALL@10']:.4f} ALL@50={p['ALL@50']:.4f} ({time.time()-t0:.0f}s)")
        if key > best: best = key; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_ep = ep
        if tb_key is None or key > tb_key: tb_key = key; tb_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; tb_ep = ep
    model.load_state_dict(best_state)
    trained_best = max((h for h in hist if h["epoch"] >= 0), key=lambda h: h["val_pooled"]["NDCG@50"], default=None)
    tb_model = CT.QueryGate(); tb_model.load_state_dict(tb_state); tb_model.eval()   # best TRAINED checkpoint
    return model, {"tag": tag, "loss_kind": loss_kind, "train_name": train_name, "epochs": epochs, "lr": lr,
                   "best_sel": best, "selected_epoch": best_ep, "selected_is_init": best_ep == -1,
                   "trained_best_epoch": trained_best["epoch"] if trained_best else None,
                   "trained_best_ndcg": trained_best["val_pooled"]["NDCG@50"] if trained_best else None,
                   "history": hist}, tb_model


# ------------------------------------------------------------------ Task 7: C6c FULL-P50-SCOPE training
# Only run if C6b fails. Ranks are the TRUE full-scope ranks (not a sampled sublist), so LambdaRank ΔNDCG@50
# is the exact deployment objective. Naturally sparse: only negatives with full-scope rank < 50 carry nonzero
# ΔNDCG@50 weight, so the pairwise loss is |G|×(≤50 boundary negs) per query — cheap. On-the-fly from raw
# arrays (no giant cache): per query compute contribs, full-scope scores (differentiable), detached ranks.
def _load_raw(ds, split):
    d = f"{CORP}/{ds}/{split}"
    R = {k: np.load(f"{d}/{k}.npy") for k in ("query_offsets", "labels", "dense_score",
         "splade_scope_score", "offset_score", "mixture_score", "relation_qwen_score", "relation_mask")}
    return R


def train_c6c(tag="C6c", epochs=8, lr=3e-4, seed=0, log=print):
    """Query-only gate trained directly on the FULL P50 scope with true-rank LambdaRank ΔNDCG@50."""
    torch.manual_seed(seed); np.random.seed(seed)
    raw = {ds: _load_raw(ds, "train") for ds in DS}
    meta = {ds: np.load(f"{CACHE}/{ds}_traindepth.npz") for ds in DS}   # qscalar/row_all/valid aligned by qi
    va = {ds: CT.load_cache(ds, "val") for ds in DS}
    model = CT.QueryGate(); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    qlists = {ds: [qi for qi in range(len(raw[ds]["query_offsets"]) - 1) if meta[ds]["valid"][qi]] for ds in DS}
    nbal = min(len(qlists[ds]) for ds in DS)
    def sel_key(p): return (round(p["NDCG@50"], 6), round(p["ALL@10"], 6), round(p["ALL@50"], 6))
    p0 = CT.evaluate(model, "C2", va)["POOLED"]
    best = sel_key(p0); best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_ep = -1
    tb_key = None; tb_state = None; tb_ep = None
    hist = [{"epoch": -1, "train_loss": None, "val_pooled": {k: round(p0[k], 4) for k in ("MRR", "NDCG@50", "R@5", "ALL@10", "ALL@50")}}]
    log(f"[{tag}] ep-1(init/C0) NDCG50={p0['NDCG@50']:.4f} MRR={p0['MRR']:.4f} ALL@10={p0['ALL@10']:.4f} ALL@50={p0['ALL@50']:.4f}")
    t0 = time.time()
    for ep in range(epochs):
        rng = np.random.default_rng(seed + ep)
        epoch_q = [(ds, qi) for ds in DS for qi in rng.choice(qlists[ds], size=nbal, replace=False)]; rng.shuffle(epoch_q)
        model.train(); opt.zero_grad(); acc = 0.0; step = 0; B = 32
        for j, (ds, qi) in enumerate(epoch_q):
            R = raw[ds]; s0, e0 = int(R["query_offsets"][qi]), int(R["query_offsets"][qi + 1]); n = e0 - s0
            lab = R["labels"][s0:e0]; gold = np.where(lab == 1)[0]
            if len(gold) == 0: continue
            C = SH.contribs(R["dense_score"][s0:e0], R["splade_scope_score"][s0:e0], R["offset_score"][s0:e0],
                            R["mixture_score"][s0:e0], R["relation_qwen_score"][s0:e0], R["relation_mask"][s0:e0])  # (5,n)
            rk = np.empty((N, n), np.float32)
            for i in range(4): rk[i] = (1.0 / C[i]) - K0
            ie = np.where(R["relation_mask"][s0:e0])[0]; rk[4] = n
            if len(ie): rk[4][ie] = (1.0 / C[4][ie]) - K0
            rt = torch.from_numpy(C.T.astype(np.float32)); rkt = torch.from_numpy((rk.T / n).astype(np.float32))
            maskt = torch.from_numpy((R["relation_mask"][s0:e0] > 0))
            hqv = torch.from_numpy(CT.hq(ds)[int(meta[ds]["row_all"][qi])]); scal = torch.from_numpy(meta[ds]["qscalar"][qi])
            w = model.weights_q(hqv.unsqueeze(0), scal.unsqueeze(0))[0]        # (5,) query-only
            s = (w.unsqueeze(0) * rt).sum(1)                                   # (n,) full-scope scores, differentiable
            # true full-scope ranks (detached) -> ΔNDCG@50 weights; boundary negatives (rank<50) only
            order = torch.argsort(-s.detach(), stable=True).numpy(); rankof = np.empty(n, np.int64); rankof[order] = np.arange(n)
            D = np.where(rankof < 50, 1.0 / np.log2(rankof + 2.0), 0.0)
            ng = len(gold); idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(ng, 50))])
            neg = np.where(lab == 0)[0]; negb = neg[rankof[neg] < 50]          # only negs that affect NDCG@50
            if len(negb) == 0 or idcg <= 0: continue
            Wm = torch.from_numpy((np.abs(D[gold][:, None] - D[negb][None, :]) / idcg).astype(np.float32))
            if float(Wm.sum()) <= 0: continue
            diff = s[torch.from_numpy(gold)].unsqueeze(1) - s[torch.from_numpy(negb)].unsqueeze(0)
            L = (Wm * F.softplus(-SCALE * diff)).sum() / Wm.sum()
            (L / B).backward(); acc += float(L.detach()); step += 1
            if (j + 1) % B == 0: opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        p = CT.evaluate(model, "C2", va)["POOLED"]; key = sel_key(p)
        hist.append({"epoch": ep, "train_loss": acc / max(step, 1), "val_pooled": {k: round(p[k], 4) for k in ("MRR", "NDCG@50", "R@5", "ALL@10", "ALL@50")}})
        log(f"[{tag}] ep{ep} loss={acc/max(step,1):.4f} NDCG50={p['NDCG@50']:.4f} MRR={p['MRR']:.4f} R@5={p['R@5']:.4f} ALL@10={p['ALL@10']:.4f} ALL@50={p['ALL@50']:.4f} ({time.time()-t0:.0f}s)")
        if key > best: best = key; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_ep = ep
        if tb_key is None or key > tb_key: tb_key = key; tb_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; tb_ep = ep
    model.load_state_dict(best_state)
    trained_best = max((h for h in hist if h["epoch"] >= 0), key=lambda h: h["val_pooled"]["NDCG@50"], default=None)
    tb_model = CT.QueryGate(); tb_model.load_state_dict(tb_state); tb_model.eval()
    return model, {"tag": tag, "loss_kind": "lambdarank_fullscope", "train_name": "full_p50", "epochs": epochs, "lr": lr,
                   "best_sel": best, "selected_epoch": best_ep, "selected_is_init": best_ep == -1,
                   "trained_best_epoch": trained_best["epoch"] if trained_best else None,
                   "trained_best_ndcg": trained_best["val_pooled"]["NDCG@50"] if trained_best else None,
                   "history": hist}, tb_model


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        for ds in DS: build_depth_cache(ds)
        print(json.dumps(depth_provenance_summary(), indent=1))
