"""L2 soft expert controller — C0..C4 controlled progression over the 5 FROZEN experts.

Canonical interface = RRF expert contribution r_i(q,d)=1/(K0+rank_i), scale-free, matches the Shapley
coalition value function. Relation ABSTAINS (r=0) on mask=0 — structural, never overridable by the gate.

  C0  fixed equal RRF                          score = Σ_i r_i
  C1  learned GLOBAL positive weights          score = Σ_i w_i r_i           (w shared across all q,d)
  C2  QUERY-ONLY gate                          score = Σ_i w_i(q) r_i
  C3  QUERY+CANDIDATE gate                      score = Σ_i w_i(q,d) r_i
  C4  C3 + positive-KL Shapley auxiliary (λ)

Weights are positive & sum-to-1 (softplus / Σ); init → equal 0.2 so C1@init == C0 ranking.
Primary loss = multi-positive listwise softmax on the canonical train_sub (all positives + hard negs);
no positive ever relabeled (POSITIVE_COLLISIONS=0). VAL evaluation ranks the FULL ~5000 P-scope.
MODEL SELECTION (declared pre-training): primary NDCG@50, tie-break ALL@10 then ALL@50, monitor MRR.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import l2_shapley as SH   # contribs(), K0, EXPERTS
import torch, torch.nn as nn, torch.nn.functional as F

DS = ("2wiki_clean", "musique_clean")
CORP = "data/l2_corpus"; UKB = "data/ukb_storage"; SHAP = "results/L2/_shapley"
CACHE = "scratchpad/_ctrl_cache"; os.makedirs(CACHE, exist_ok=True)
EXPERTS = SH.EXPERTS; N = 5; K0 = SH.K0
KS = (1, 5, 10, 20, 50)
SCALE = 40.0    # FIXED softmax temperature for the listwise loss. Softer than a learned runaway scale =>
                # gradient spreads over ALL positives + deeper ranks (targets multi-gold NDCG@50, not top-1 MRR).
torch.manual_seed(0); np.random.seed(0)


# ------------------------------------------------------------------ feature cache
def _jac_topk(rk_a, rk_b, k=10):
    A = set(np.where(rk_a < k)[0].tolist()); B = set(np.where(rk_b < k)[0].tolist())
    u = len(A | B); return (len(A & B) / u) if u else 0.0


def build_cache(ds, split):
    """Per-candidate r_i(5)+normrank(5)+mask, per-query 5 scalar diagnostics + row_all + gold offsets.
    train split -> only train_sub candidates; val split -> full scope."""
    d = f"{CORP}/{ds}/{split}"
    off = np.load(f"{d}/query_offsets.npy"); labels = np.load(f"{d}/labels.npy")
    dense = np.load(f"{d}/dense_score.npy"); splade = np.load(f"{d}/splade_scope_score.npy")
    offs = np.load(f"{d}/offset_score.npy"); mix = np.load(f"{d}/mixture_score.npy")
    rels = np.load(f"{d}/relation_qwen_score.npy"); relm = np.load(f"{d}/relation_mask.npy")
    M = np.load(f"{d}/expert_meta.npz")
    disagree = M["dense_splade_disagreement"].astype(np.float32)
    relany = M["relation_any_signal_present"].astype(np.float32)
    relnc = M["relation_num_candidates"].astype(np.float32)
    row_all = M["row_all"].astype(np.int64)
    nq = len(off) - 1
    is_train = (split == "train")
    if is_train:
        ts_off = np.load(f"{d}/train_sub_offsets.npy"); ts_local = np.load(f"{d}/train_sub_local.npy")

    r_rows, rk_rows, mask_rows, lab_rows = [], [], [], []
    q_off = [0]; qscalar = np.zeros((nq, 5), np.float32); valid = np.zeros(nq, bool)
    gold_off = [0]; gold_loc = []          # per-query in-sub (train) or in-scope (val) gold local idx
    t0 = time.time()
    for qi in range(nq):
        s, e = int(off[qi]), int(off[qi + 1]); n = e - s
        lab = labels[s:e]
        C = SH.contribs(dense[s:e], splade[s:e], offs[s:e], mix[s:e], rels[s:e], relm[s:e])  # (5,n) r_i
        # integer ranks per expert (relation: rank among eligible, else n = worst)
        rk = np.empty((N, n), np.float32)
        for i in range(4):
            rk[i] = (1.0 / C[i]) - K0
        ie = np.where(relm[s:e])[0]; rk[4] = n
        if len(ie):
            rk[4][ie] = (1.0 / C[4][ie]) - K0
        # query scalar diagnostics (inference-available, gold-free)
        relfrac = float(len(ie)) / n
        jac_om = _jac_topk(rk[2], rk[3]); jac_do = _jac_topk(rk[0], rk[2])
        qscalar[qi] = [disagree[qi], relany[qi], relfrac, jac_om, jac_do]
        # select rows
        if is_train:
            a, b = int(ts_off[qi]), int(ts_off[qi + 1]); sel = ts_local[a:b].astype(np.int64)
        else:
            sel = np.arange(n)
        gl_scope = np.where(lab == 1)[0]
        if len(gl_scope) == 0:                # L1_ANY_FAIL -> skip query entirely
            q_off.append(q_off[-1]); gold_off.append(gold_off[-1]); continue
        valid[qi] = True
        Csel = C[:, sel].T.astype(np.float16)        # (nsel,5)
        rksel = (rk[:, sel].T / n).astype(np.float16)  # normalized rank
        masksel = (relm[s:e][sel]).astype(bool)
        labsel = lab[sel].astype(np.int8)
        r_rows.append(Csel); rk_rows.append(rksel); mask_rows.append(masksel); lab_rows.append(labsel)
        q_off.append(q_off[-1] + len(sel))
        # gold local indices *within sel* (train: golds are in sub by construction; val: within scope=sel)
        if is_train:
            selpos = np.where(labsel == 1)[0]
        else:
            selpos = gl_scope
        gold_loc.append(selpos.astype(np.int64)); gold_off.append(gold_off[-1] + len(selpos))
        if (qi + 1) % 3000 == 0:
            print(f"  [{ds}/{split}] {qi+1}/{nq} {time.time()-t0:.0f}s", flush=True)
    out = dict(
        r=np.concatenate(r_rows), rk=np.concatenate(rk_rows),
        mask=np.concatenate(mask_rows), lab=np.concatenate(lab_rows),
        q_off=np.array(q_off, np.int64), qscalar=qscalar, valid=valid, row_all=row_all,
        gold_loc=np.concatenate(gold_loc), gold_off=np.array(gold_off, np.int64))
    np.savez(f"{CACHE}/{ds}_{split}.npz", **out)
    print(f"SAVED cache {ds}/{split} nq_valid={int(valid.sum())} rows={len(out['r'])} {time.time()-t0:.0f}s", flush=True)
    return out


def load_cache(ds, split):
    z = np.load(f"{CACHE}/{ds}_{split}.npz")
    return {k: z[k] for k in z.files}


_HQ = {}
def hq(ds):
    if ds not in _HQ:
        q = np.load(f"{UKB}/{ds}/gte_qwen/queries_all.npy").astype(np.float32)
        _HQ[ds] = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    return _HQ[ds]


# ------------------------------------------------------------------ models
class GlobalW(nn.Module):                       # C1
    def __init__(self): super().__init__(); self.z = nn.Parameter(torch.zeros(N))
    def weights(self, qf=None, cf=None, nrows=1):
        w = F.softplus(self.z); w = w / w.sum()
        return w.unsqueeze(0)                    # (1,5) broadcast


class QueryGate(nn.Module):                     # C2
    def __init__(self, hid=64):
        super().__init__(); self.proj = nn.Linear(1536, 64)
        self.net = nn.Sequential(nn.Linear(64 + 5, hid), nn.ReLU(), nn.Linear(hid, N))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)  # init => equal weights == C0
    def qfeat(self, hq_vec, scal):               # hq_vec (B,1536), scal (B,5)
        return torch.cat([self.proj(hq_vec), scal], 1)
    def weights_q(self, hq_vec, scal):
        z = self.net(self.qfeat(hq_vec, scal)); w = F.softplus(z); return w / w.sum(1, keepdim=True)  # (B,5)


class CandGate(nn.Module):                       # C3
    def __init__(self, hid=64):
        super().__init__(); self.proj = nn.Linear(1536, 64)
        self.net = nn.Sequential(nn.Linear(64 + 5 + 13, hid), nn.ReLU(), nn.Linear(hid, N))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)  # init => equal weights == C0
    def weights_qd(self, qf, cf):                 # qf (69,), cf (n,13) -> (n,5)
        x = torch.cat([qf.unsqueeze(0).expand(cf.shape[0], -1), cf], 1)
        z = self.net(x); w = F.softplus(z); return w / w.sum(1, keepdim=True)


def cand_feats(r, rk, mask):                      # torch tensors (n,5),(n,5),(n,)
    dmsp = (rk[:, 0] - rk[:, 1]).unsqueeze(1); omix = (rk[:, 2] - rk[:, 3]).unsqueeze(1)
    return torch.cat([r, rk, mask.float().unsqueeze(1), dmsp, omix], 1)   # (n,13)


# ------------------------------------------------------------------ per-query score
def score_query(model, kind, r, rk, mask, hqv, scal):
    """r,rk (n,5) f32; mask (n,) bool; hqv (1536,); scal (5,). Returns (score(n,), w_used(n,5) or (5,))."""
    if kind == "C1":
        w = model.weights()                       # (1,5)
        s = (w * r).sum(1); return s, w.expand(r.shape[0], -1)
    if kind == "C2":
        w = model.weights_q(hqv.unsqueeze(0), scal.unsqueeze(0))[0]   # (5,)
        s = (w.unsqueeze(0) * r).sum(1); return s, w.unsqueeze(0).expand(r.shape[0], -1)
    if kind in ("C3", "C4"):
        qf = torch.cat([model.proj(hqv.unsqueeze(0))[0], scal])       # (69,)
        cf = cand_feats(r, rk, mask)
        w = model.weights_qd(qf, cf)              # (n,5)
        s = (w * r).sum(1); return s, w
    raise ValueError(kind)


# ------------------------------------------------------------------ multi-positive ranking loss
LOSS_KIND = "ranknet"   # "ranknet" = multi-positive pairwise logistic (every gold-neg pair equal weight;
                        # multi-gold / deep-recall friendly, an NDCG-aligned surrogate). "listwise" = top-1-ish.
def loss_query(model, kind, r, rk, mask, hqv, scal, gold_idx, neg_idx=None):
    s, _ = score_query(model, kind, r, rk, mask, hqv, scal)
    if LOSS_KIND == "listwise":
        logit = SCALE * s
        return -(logit[gold_idx] - torch.logsumexp(logit, 0)).mean()
    # RankNet: mean over (gold, neg) pairs of softplus(-scale*(s_g - s_n)) => pushes EVERY gold above negs
    diff = s[gold_idx].unsqueeze(1) - s[neg_idx].unsqueeze(0)      # (|G|,|neg|)
    return F.softplus(-SCALE * diff).mean()


# ------------------------------------------------------------------ full-scope VAL evaluator
def evaluate(model, kind, caches, want_gate=False, collect=False):
    """Rank full scope per val query; metrics under COND_ANY_GOLD_IN_SCOPE. Returns per-dataset + pooled.
    collect=True also returns res['_perq'][ds] = dict of per-query arrays (ndcg,mrr,all10,all50,best,relw)."""
    model.eval() if isinstance(model, nn.Module) else None
    res = {}
    gate_acc = {e: [] for e in EXPERTS}; gate_ent = []; gate_dom = {e: 0 for e in EXPERTS}
    relgate_by_signal = {0: [], 1: []}; perq = {}
    with torch.no_grad():
        for ds in DS:
            c = caches[ds]; H = hq(ds)
            rr = []; rec = {K: [] for K in KS}; anyk = {K: [] for K in KS}; allk = {K: [] for K in KS}
            nqv = len(c["q_off"]) - 1; ndcg_sum = 0.0
            pq = {k: [] for k in ("qi", "ndcg", "mrr", "all10", "all50", "best", "relw", "meanw")} if collect else None
            for qi in range(nqv):
                a, b = int(c["q_off"][qi]), int(c["q_off"][qi + 1])
                if b == a: continue
                r = torch.from_numpy(c["r"][a:b].astype(np.float32))
                rk = torch.from_numpy(c["rk"][a:b].astype(np.float32))
                mask = torch.from_numpy(c["mask"][a:b])
                hqv = torch.from_numpy(H[c["row_all"][qi]])
                scal = torch.from_numpy(c["qscalar"][qi])
                s, w = score_query(model, kind, r, rk, mask, hqv, scal)
                ga, gb = int(c["gold_off"][qi]), int(c["gold_off"][qi + 1])
                gold = c["gold_loc"][ga:gb]
                order = torch.argsort(-s, stable=True).numpy()
                rankof = np.empty(len(s), np.int64); rankof[order] = np.arange(len(s))
                gr = rankof[gold]; best = int(gr.min()); worst = int(gr.max()); ng = len(gold)
                rr.append(1.0 / (best + 1))
                for K in KS:
                    rec[K].append((gr < K).sum() / ng); anyk[K].append(1.0 if best < K else 0.0); allk[K].append(1.0 if worst < K else 0.0)
                dcg = np.sum([1.0 / np.log2(x + 2) for x in gr if x < 50])
                idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(ng, 50))])
                nd = (dcg / idcg) if idcg > 0 else 0.0; ndcg_sum += nd
                wm = w.mean(0).numpy() if (want_gate or collect) else None
                if want_gate:
                    for i, e in enumerate(EXPERTS): gate_acc[e].append(float(wm[i]))
                    p = wm / (wm.sum() + 1e-9); gate_ent.append(float(-(p * np.log(p + 1e-12)).sum()))
                    gate_dom[EXPERTS[int(wm.argmax())]] += 1
                    relgate_by_signal[int(c["qscalar"][qi][1] > 0.5)].append(float(wm[4]))
                if collect:
                    pq["qi"].append(qi); pq["ndcg"].append(nd); pq["mrr"].append(1.0 / (best + 1))
                    pq["all10"].append(1.0 if worst < 10 else 0.0); pq["all50"].append(1.0 if worst < 50 else 0.0)
                    pq["best"].append(best); pq["relw"].append(float(wm[4])); pq["meanw"].append(wm.tolist())
            d = dict(n=len(rr), MRR=float(np.mean(rr)),
                     **{f"R@{K}": float(np.mean(rec[K])) for K in KS},
                     **{f"ANY@{K}": float(np.mean(anyk[K])) for K in KS},
                     **{f"ALL@{K}": float(np.mean(allk[K])) for K in KS},
                     **{"NDCG@50": ndcg_sum / len(rr)})
            res[ds] = d
            if collect: perq[ds] = {k: np.array(v) for k, v in pq.items()}
    if collect: res["_perq"] = perq
    # pooled (query-weighted)
    pooled = {}
    for k in ("MRR", "NDCG@50", *[f"R@{K}" for K in KS], *[f"ANY@{K}" for K in KS], *[f"ALL@{K}" for K in KS]):
        num = sum(res[ds][k] * res[ds]["n"] for ds in DS); den = sum(res[ds]["n"] for ds in DS)
        pooled[k] = num / den
    pooled["n"] = sum(res[ds]["n"] for ds in DS); res["POOLED"] = pooled
    if want_gate:
        res["_gate"] = {"mean_weight": {e: float(np.mean(gate_acc[e])) for e in EXPERTS},
                        "entropy_mean": float(np.mean(gate_ent)),
                        "dominant_freq": gate_dom,
                        "relation_weight_by_signal": {str(k): float(np.mean(v)) if v else None for k, v in relgate_by_signal.items()}}
    return res


# ------------------------------------------------------------------ training
def shapley_target(ds):
    """positive-normalized PHI_NDCG50 per train query (local index order), for the C4 KL auxiliary."""
    z = np.load(f"{SHAP}/{ds}_train.npz")
    phi = np.vstack([z[f"phi_ndcg50_{e}"] for e in EXPERTS]).T   # (nq,5) local order
    p = np.clip(phi, 0, None); s = p.sum(1, keepdims=True)
    t = np.where(s > 1e-9, p / np.clip(s, 1e-9, None), 0.2)      # uniform fallback if no positive mass
    return t.astype(np.float32)


def train(kind, epochs=12, lr=1e-3, lam=0.0, smoke=None, seed=0, sel_rule="ndcg", expdrop=0.0, log=print):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = {ds: load_cache(ds, "train") for ds in DS}
    va = {ds: load_cache(ds, "val") for ds in DS}
    tgt = {ds: shapley_target(ds) for ds in DS} if kind == "C4" else None
    model = {"C1": GlobalW, "C2": QueryGate, "C3": CandGate, "C4": CandGate}[kind]()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # balanced epoch: equal #queries from each dataset (subsample larger to smaller)
    qlists = {ds: [qi for qi in range(len(tr[ds]["q_off"]) - 1) if tr[ds]["q_off"][qi + 1] > tr[ds]["q_off"][qi]] for ds in DS}
    if smoke:
        for ds in DS: qlists[ds] = qlists[ds][:smoke]
    nbal = min(len(qlists[ds]) for ds in DS)
    best = None; best_state = None; hist = []; pos_collisions = 0; t0 = time.time()
    # INIT-INCLUSIVE selection: equal-weight init == C0. A learned model is max(C0, best trained epoch),
    # so "does training help?" is measured honestly and the controller can never underperform equal fusion.
    def sel_key(p):
        return (round(p["NDCG@50"], 6), round(p["ALL@10"], 6), round(p["ALL@50"], 6)) if sel_rule == "ndcg" else (round(p["MRR"], 6),)
    p0 = evaluate(model, kind, va)["POOLED"]
    best = sel_key(p0); best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_ep = -1
    hist.append({"epoch": -1, "train_loss": None, "val_pooled": {k: round(p0[k], 4) for k in ("MRR", "NDCG@50", "R@5", "ALL@10", "ALL@50")}})
    log(f"[{kind}] ep-1(init/C0) NDCG50={p0['NDCG@50']:.4f} MRR={p0['MRR']:.4f} ALL@10={p0['ALL@10']:.4f} ALL@50={p0['ALL@50']:.4f}")
    for ep in range(epochs):
        rng = np.random.default_rng(seed + ep)
        epoch_q = [(ds, qi) for ds in DS for qi in rng.choice(qlists[ds], size=nbal, replace=False)]
        rng.shuffle(epoch_q)
        model.train(); opt.zero_grad(); acc_loss = 0.0; step = 0; B = 32
        for j, (ds, qi) in enumerate(epoch_q):
            c = tr[ds]; a, b = int(c["q_off"][qi]), int(c["q_off"][qi + 1])
            r = torch.from_numpy(c["r"][a:b].astype(np.float32)); rk = torch.from_numpy(c["rk"][a:b].astype(np.float32))
            mask = torch.from_numpy(c["mask"][a:b]); hqv = torch.from_numpy(hq(ds)[c["row_all"][qi]]); scal = torch.from_numpy(c["qscalar"][qi])
            ga, gb = int(c["gold_off"][qi]), int(c["gold_off"][qi + 1]); gold = torch.from_numpy(c["gold_loc"][ga:gb])
            neg = torch.from_numpy(np.where(c["lab"][a:b] == 0)[0])
            if len(neg) == 0: continue
            if expdrop > 0:                          # C5: stochastic expert dropout (never drop all; keep >=2)
                keep = np.random.random(N) > expdrop
                if keep.sum() < 2: keep[np.random.choice(N, 2, replace=False)] = True
                r = r * torch.from_numpy(keep.astype(np.float32))   # zero dropped experts' contribution + feature
            L = loss_query(model, kind, r, rk, mask, hqv, scal, gold, neg)
            if kind == "C4" and lam > 0:
                _, w = score_query(model, kind, r, rk, mask, hqv, scal)
                gate_dist = w.mean(0); gate_dist = gate_dist / gate_dist.sum()
                t = torch.from_numpy(tgt[ds][qi])
                L = L + lam * (t * (torch.log(t + 1e-9) - torch.log(gate_dist + 1e-9))).sum()
            (L / B).backward(); acc_loss += float(L); step += 1
            if (j + 1) % B == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        val = evaluate(model, kind, va)
        p = val["POOLED"]; key = sel_key(p)
        hist.append({"epoch": ep, "train_loss": acc_loss / step, "val_pooled": {k: round(p[k], 4) for k in ("MRR", "NDCG@50", "R@5", "ALL@10", "ALL@50")}})
        log(f"[{kind}] ep{ep} loss={acc_loss/step:.4f} NDCG50={p['NDCG@50']:.4f} MRR={p['MRR']:.4f} R@5={p['R@5']:.4f} ALL@10={p['ALL@10']:.4f} ALL@50={p['ALL@50']:.4f} ({time.time()-t0:.0f}s)")
        if key > best:
            best = key; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best_ep = ep
    model.load_state_dict(best_state)
    return model, {"kind": kind, "epochs": epochs, "lr": lr, "lam": lam, "best_sel": best, "best_epoch": best_ep, "history": hist, "pos_collisions": pos_collisions}


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "cache":
        for ds in DS:
            for sp in ("train", "val"): build_cache(ds, sp)
    elif stage == "smoke":
        m, meta = train(sys.argv[2] if len(sys.argv) > 2 else "C1", epochs=3, smoke=250)
        print(json.dumps(meta["history"], indent=1))
