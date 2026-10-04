"""C8 — TOP-5 EVIDENCE OPTIMIZATION. VAL used ONCE at the end; design decisions on DEV_INNER (from TRAIN).
No TEST, no L1 change, no L3, no new experts, no unrestricted controller search.

Reuses the frozen 5 experts + the frozen C7 archetype set. Adds:
  - deterministic stratified TRAIN_INNER / DEV_INNER split (90/10)
  - top-5 metric suite (ANY@5, GOLD_RECALL@5, NDCG@5, ALL@5, ALL@5_FEASIBLE) + deep guardrails
  - per-archetype top-5 utilities (precomputed once) for oracle + distillation targets
  - C8a soft archetype distillation aimed at NDCG@5 (same classifier family as C7b, only the TRAIN label changes)
  - expert-level top5/top10 oracle + union analysis (fusion-vs-scoring diagnosis)
"""
import os, sys, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import l2_shapley as SH
import l2_c7 as C7          # AWMAT, ANAMES, ARCH, _hq
import l2_c6 as C6          # _load_raw

DS = ("2wiki_clean", "musique_clean"); CORP = "data/l2_corpus"; OUT = "results/L2/_ctrl"
K0 = SH.K0; N = 5
ANAMES = C7.ANAMES; AWMAT = C7.AWMAT.astype(np.float64)   # (6,5)
EXPERTS = ("dense", "splade", "offset", "mixture", "relation")
RANKBINS = [(0, 4), (5, 9), (10, 19), (20, 49), (50, 99), (100, 499), (500, 10**9)]
RBNAMES = ["0-4", "5-9", "10-19", "20-49", "50-99", "100-499", "500+"]


def rankbin(r):
    for i, (lo, hi) in enumerate(RANKBINS):
        if lo <= r <= hi:
            return i
    return len(RANKBINS) - 1


# ---------------------------------------------------------------- per-query contribs & metrics
def contribs_of(R, s, e):
    return SH.contribs(R["dense_score"][s:e], R["splade_scope_score"][s:e], R["offset_score"][s:e],
                       R["mixture_score"][s:e], R["relation_qwen_score"][s:e], R["relation_mask"][s:e])


def ranks_from_score(score):
    order = np.argsort(-score, kind="stable"); rankof = np.empty(len(score), np.int64); rankof[order] = np.arange(len(score))
    return rankof


def topk_metrics(score, gold):
    """Full top-5 suite + deep guardrails for one query. gold = local indices of in-scope golds."""
    rankof = ranks_from_score(score); gr = rankof[gold]; ng = len(gold)
    best = int(gr.min()); worst = int(gr.max())
    def ndcg(k):
        dcg = np.sum([1.0 / np.log2(x + 2) for x in gr if x < k]); idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(ng, k))])
        return (dcg / idcg) if idcg > 0 else 0.0
    g5 = int((gr < 5).sum())
    return {"any5": 1.0 if best < 5 else 0.0, "golds_in5": g5, "ng": ng, "recall5": g5 / ng,
            "ndcg5": ndcg(5), "ndcg50": ndcg(50), "mrr": 1.0 / (best + 1),
            "all5": 1.0 if worst < 5 else 0.0, "all10": 1.0 if worst < 10 else 0.0, "all50": 1.0 if worst < 50 else 0.0,
            "all5_feas": (1.0 if worst < 5 else 0.0) if ng <= 5 else None,
            "best": best, "worst": worst, "gr": gr}


def valid_queries(R):
    off = R["query_offsets"]; nq = len(off) - 1; out = []
    for qi in range(nq):
        s, e = int(off[qi]), int(off[qi + 1])
        if int(R["labels"][s:e].sum()) > 0: out.append(qi)
    return np.array(out, np.int64)


# ---------------------------------------------------------------- per-archetype top-k utilities (cache)
def precompute_arch(ds, split, log=print):
    fp = f"{OUT}/_c8_arch_{ds}_{split}.npz"
    if os.path.exists(fp): return dict(np.load(fp, allow_pickle=True))
    R = C6._load_raw(ds, split); off = R["query_offsets"]
    M = np.load(f"{CORP}/{ds}/{split}/expert_meta.npz")
    disagree = M["dense_splade_disagreement"].astype(np.float32); relany = M["relation_any_signal_present"].astype(np.float32)
    gcount = M["gold_count"].astype(np.int64)
    qis = valid_queries(R)
    nd5 = np.zeros((len(qis), 6), np.float32); nd50 = np.zeros((len(qis), 6), np.float32)
    rec5 = np.zeros((len(qis), 6), np.float32); a5f = np.full((len(qis), 6), np.nan, np.float32)
    ng = np.zeros(len(qis), np.int64); dis = np.zeros(len(qis), np.float32); rel = np.zeros(len(qis), np.float32)
    t0 = time.time()
    for r, qi in enumerate(qis):
        s, e = int(off[qi]), int(off[qi + 1]); gold = np.where(R["labels"][s:e] == 1)[0]
        C = contribs_of(R, s, e)
        for ai in range(6):
            sc = (AWMAT[ai][:, None] * C).sum(0); m = topk_metrics(sc, gold)
            nd5[r, ai] = m["ndcg5"]; nd50[r, ai] = m["ndcg50"]; rec5[r, ai] = m["recall5"]
            if m["all5_feas"] is not None: a5f[r, ai] = m["all5_feas"]
        ng[r] = len(gold); dis[r] = disagree[qi]; rel[r] = relany[qi]
        if (r + 1) % 4000 == 0: log(f"  [arch {ds}/{split}] {r+1}/{len(qis)} {time.time()-t0:.0f}s")
    out = dict(qi=qis, ndcg5=nd5, ndcg50=nd50, recall5=rec5, all5feas=a5f, ng=ng, disagree=dis, relany=rel,
               gold_count=gcount[qis])
    np.savez(fp, **out); log(f"ARCH {ds}/{split} nq={len(qis)} {time.time()-t0:.0f}s -> {fp}")
    return out


# ---------------------------------------------------------------- deterministic stratified inner/dev split
def _h(qi):  # stable integer hash in [0,1)
    x = (int(qi) * 2654435761 + 12345) & 0xFFFFFFFF
    x ^= (x >> 16); x = (x * 2246822519) & 0xFFFFFFFF; x ^= (x >> 13)
    return x / 0xFFFFFFFF


def build_inner_split(dev_frac=0.10, log=print):
    """Per dataset: stratify valid TRAIN queries by (gold-mult bucket, relany, disagree>median); take dev_frac
    of each stratum (lowest hash) into DEV_INNER, rest TRAIN_INNER. Deterministic. Saved to _c8_split_{ds}.npz."""
    res = {}
    for ds in DS:
        A = precompute_arch(ds, "train", log); qis = A["qi"]; ng = A["ng"]; rel = A["relany"]; dis = A["disagree"]
        med = np.median(dis)
        gb = np.clip(ng, 1, 3)   # 1,2,3+ multiplicity bucket
        strat = {}
        for r, qi in enumerate(qis):
            key = (int(gb[r]), int(rel[r] > 0.5), int(dis[r] > med)); strat.setdefault(key, []).append((int(qi), _h(qi)))
        dev = []; inner = []
        for key, lst in strat.items():
            lst.sort(key=lambda t: t[1]); k = int(round(len(lst) * dev_frac))
            dev += [q for q, _ in lst[:k]]; inner += [q for q, _ in lst[k:]]
        dev = np.array(sorted(dev), np.int64); inner = np.array(sorted(inner), np.int64)
        np.savez(f"{OUT}/_c8_split_{ds}.npz", inner=inner, dev=dev, dev_median_disagree=med)
        res[ds] = {"n_valid": int(len(qis)), "n_inner": int(len(inner)), "n_dev": int(len(dev)),
                   "n_strata": len(strat)}
        log(f"SPLIT {ds}: valid={len(qis)} inner={len(inner)} dev={len(dev)} strata={len(strat)}")
    return res


def load_split(ds):
    z = np.load(f"{OUT}/_c8_split_{ds}.npz"); return z["inner"], z["dev"]


# ---------------------------------------------------------------- inference-safe features (reuse C6 feats)
_FEAT = {}
def feat_map(ds, split):
    key = (ds, split)
    if key not in _FEAT:
        z = np.load(f"{OUT}/_c6_feat_{ds}_{split}.npz"); qi = z["qi"]
        _FEAT[key] = (z["feats"], {int(q): i for i, q in enumerate(qi)})
    return _FEAT[key]


def feats_for(ds, split, qi_list):
    F_, idx = feat_map(ds, split); rows = [idx[int(q)] for q in qi_list]; return F_[rows]


# ---------------------------------------------------------------- soft archetype head (C7b / C8a family)
def fit_soft_head(target, qi_by_ds, seed=0, epochs=300):
    """target in {'ndcg50'(C7b),'ndcg5'(C8a)}. Labels = argmax per-archetype utility on the given TRAIN qi subset.
    Same tiny class-balanced multinomial-logistic family as C7b. Returns (clf, mu, sd)."""
    Xs, ys = [], []
    for ds in DS:
        A = precompute_arch(ds, "train"); pos = {int(q): r for r, q in enumerate(A["qi"])}
        rows = [pos[int(q)] for q in qi_by_ds[ds]]
        U = A[target][rows]; Xs.append(feats_for(ds, "train", qi_by_ds[ds])); ys.append(U.argmax(1))
    X = np.concatenate(Xs).astype(np.float32); y = np.concatenate(ys).astype(np.int64)
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)); yt = torch.from_numpy(y)
    torch.manual_seed(seed); clf = nn.Linear(X.shape[1], 6)
    cnt = np.bincount(y, minlength=6); cw = torch.from_numpy((cnt.sum() / (6 * np.clip(cnt, 1, None))).astype(np.float32))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(); F.cross_entropy(clf(Xn), yt, weight=cw).backward(); opt.step()
    return clf, mu, sd, {"label_dist": {ANAMES[c]: int((y == c).sum()) for c in range(6)}}


def soft_weights(clf, mu, sd, ds, split, qi_list):
    X = feats_for(ds, split, qi_list); Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    with torch.no_grad(): p = F.softmax(clf(Xn), 1).numpy()
    return p @ AWMAT   # (nq,5)


def equal_weights(nq):
    return np.tile(AWMAT[ANAMES.index("equal")], (nq, 1))


# ---------------------------------------------------------------- policy evaluation (top-5 + deep suite)
AGG = ["any5", "recall5_micro", "recall5_macro", "ndcg5", "ndcg50", "mrr", "all5", "all10", "all50", "all5_feas"]
def eval_policy(ds, split, qi_list, W, want_perq=True, want_ranks=False):
    """W = (nq,5) weight rows aligned to qi_list. Returns aggregates + per-query arrays (+ per-gold ranks)."""
    R = C6._load_raw(ds, split); off = R["query_offsets"]
    acc = {k: [] for k in ("any5", "ndcg5", "ndcg50", "mrr", "all5", "all10", "all50")}
    g_in5 = 0; g_tot = 0; rec_macro = []; a5f = []
    pq = {k: [] for k in ("qi", "ndcg5", "ndcg50", "recall5", "any5", "mrr", "all5", "all10", "all50", "best")}
    goldranks = []   # list of (qi, gold_local, rank) if want_ranks
    for r, qi in enumerate(qi_list):
        s, e = int(off[qi]), int(off[qi + 1]); gold = np.where(R["labels"][s:e] == 1)[0]
        C = contribs_of(R, s, e); score = (W[r][:, None] * C).sum(0); m = topk_metrics(score, gold)
        for k in acc: acc[k].append(m[k])
        g_in5 += m["golds_in5"]; g_tot += m["ng"]; rec_macro.append(m["recall5"])
        if m["all5_feas"] is not None: a5f.append(m["all5_feas"])
        for k in ("any5", "ndcg5", "ndcg50", "mrr", "all5", "all10", "all50", "best"): pq[k].append(m[k])
        pq["recall5"].append(m["recall5"]); pq["qi"].append(int(qi))
        if want_ranks:
            for gl, rk in zip(gold, m["gr"]): goldranks.append((int(qi), int(gl), int(rk)))
    out = {"n": len(qi_list), "any5": float(np.mean(acc["any5"])), "recall5_micro": g_in5 / max(g_tot, 1),
           "recall5_macro": float(np.mean(rec_macro)), "ndcg5": float(np.mean(acc["ndcg5"])),
           "ndcg50": float(np.mean(acc["ndcg50"])), "mrr": float(np.mean(acc["mrr"])),
           "all5": float(np.mean(acc["all5"])), "all10": float(np.mean(acc["all10"])), "all50": float(np.mean(acc["all50"])),
           "all5_feas": float(np.mean(a5f)) if a5f else None, "n_all5_feas": len(a5f), "_gtot": g_tot}
    if want_perq: out["_perq"] = {k: np.array(v) for k, v in pq.items()}
    if want_ranks: out["_goldranks"] = goldranks
    return out


def pool(res_by_ds, keys=AGG):
    tot = sum(res_by_ds[ds]["n"] for ds in DS); o = {"n": tot}
    for k in keys:
        if k == "recall5_micro":   # re-derive micro from counts stored per-ds
            o[k] = sum(res_by_ds[ds]["recall5_micro"] * res_by_ds[ds]["_gtot"] for ds in DS) / sum(res_by_ds[ds]["_gtot"] for ds in DS)
        elif k == "all5_feas":
            num = sum((res_by_ds[ds]["all5_feas"] or 0) * res_by_ds[ds]["n_all5_feas"] for ds in DS)
            den = sum(res_by_ds[ds]["n_all5_feas"] for ds in DS); o[k] = num / den if den else None
        else:
            o[k] = sum(res_by_ds[ds][k] * res_by_ds[ds]["n"] for ds in DS) / tot
    return o


# ---------------------------------------------------------------- expert-level oracle & union (Task 2)
def expert_analysis(ds, split, qi_list, W_c7b):
    """Per-expert standalone top5/top10 coverage, single-expert-per-query oracle, and union-of-experts
    coverage — restricted also to golds that C7b MISSES from top5 (fusion-vs-scoring diagnosis)."""
    R = C6._load_raw(ds, split); off = R["query_offsets"]
    exp = {ex: {"any5": [], "rec5": [], "all5f": []} for ex in EXPERTS}
    orc = {"any5": [], "rec5": [], "all5f": []}
    uni5 = {"any": [], "rec": [], "all5f": []}; uni10 = {"any": [], "rec": [], "all5f": []}
    # C7b-missed golds: where do they live?
    missed_in_union5 = 0; missed_in_union10 = 0; missed_deep = 0; missed_tot = 0
    for r, qi in enumerate(qi_list):
        s, e = int(off[qi]), int(off[qi + 1]); gold = np.where(R["labels"][s:e] == 1)[0]; ng = len(gold); gset = set(gold.tolist())
        C = contribs_of(R, s, e); n = e - s
        # per-expert rankings
        top5 = {}; top10 = {}
        best_rec = -1; best_any = 0; best_all5f = None
        for ei, ex in enumerate(EXPERTS):
            ro = ranks_from_score(C[ei]); t5 = set(np.where(ro < 5)[0].tolist()); t10 = set(np.where(ro < 10)[0].tolist())
            top5[ex] = t5; top10[ex] = t10
            gin5 = len(gset & t5); rec = gin5 / ng; any5 = 1.0 if gin5 > 0 else 0.0
            all5f = (1.0 if gset <= t5 else 0.0) if ng <= 5 else None
            exp[ex]["any5"].append(any5); exp[ex]["rec5"].append(rec)
            if all5f is not None: exp[ex]["all5f"].append(all5f)
            if rec > best_rec: best_rec = rec
            best_any = max(best_any, any5)
            if all5f is not None: best_all5f = max(best_all5f or 0.0, all5f)
        orc["any5"].append(best_any); orc["rec5"].append(best_rec)
        if ng <= 5 and best_all5f is not None: orc["all5f"].append(best_all5f)
        # unions
        U5 = set().union(*top5.values()); U10 = set().union(*top10.values())
        gin_u5 = len(gset & U5); gin_u10 = len(gset & U10)
        uni5["any"].append(1.0 if gin_u5 > 0 else 0.0); uni5["rec"].append(gin_u5 / ng)
        uni10["any"].append(1.0 if gin_u10 > 0 else 0.0); uni10["rec"].append(gin_u10 / ng)
        if ng <= 5:
            uni5["all5f"].append(1.0 if gset <= U5 else 0.0); uni10["all5f"].append(1.0 if gset <= U10 else 0.0)
        # C7b top5 miss localization
        sc = (W_c7b[r][:, None] * C).sum(0); ro = ranks_from_score(sc); c7b_t5 = set(np.where(ro < 5)[0].tolist())
        for g in gold:
            if g in c7b_t5: continue
            missed_tot += 1
            if g in U5: missed_in_union5 += 1
            elif g in U10: missed_in_union10 += 1
            else: missed_deep += 1
    def mean(x): return float(np.mean(x)) if len(x) else None
    return {
        "per_expert": {ex: {"any5": mean(exp[ex]["any5"]), "recall5_macro": mean(exp[ex]["rec5"]),
                            "all5_feas": mean(exp[ex]["all5f"])} for ex in EXPERTS},
        "single_expert_oracle": {"any5": mean(orc["any5"]), "recall5_macro": mean(orc["rec5"]), "all5_feas": mean(orc["all5f"])},
        "union_top5": {"any5": mean(uni5["any"]), "recall5_macro": mean(uni5["rec"]), "all5_feas": mean(uni5["all5f"])},
        "union_top10": {"any5": mean(uni10["any"]), "recall5_macro": mean(uni10["rec"]), "all5_feas": mean(uni10["all5f"])},
        "c7b_missed_from_top5": {"total": missed_tot, "in_some_expert_top5": missed_in_union5,
                                 "in_some_expert_top10_only": missed_in_union10, "deep_under_all_experts": missed_deep,
                                 "frac_recoverable_by_fusion_top5": missed_in_union5 / max(missed_tot, 1),
                                 "frac_recoverable_by_fusion_top10": (missed_in_union5 + missed_in_union10) / max(missed_tot, 1)},
        "n": len(qi_list),
    }
