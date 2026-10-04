"""C6 Task 4 — is the oracle-winning weight archetype PREDICTABLE from inference-safe features?
Turn the fixed-archetype oracle selection into a diagnostic classification problem (NOT the final controller):
  features (gold-free): query embedding + 5 query diagnostics + per-expert top-k RRF summary stats
  label:  argmax-NDCG@50 archetype (same fixed set as the ceiling), computed on TRAIN (VAL labels ceiling-only)
Train a tiny multinomial-logistic classifier on TRAIN, evaluate on VAL. Report accuracy / macro-F1 /
balanced accuracy, but the DECISIVE metric is: applying the PREDICTED archetype to actual VAL retrieval,
what VAL NDCG@50 results (vs equal 0.8401 and oracle 0.8796)?
Forbidden features (not used): relation_gold_present, gold ranks, label-Shapley, gold_count, L1 gold-success."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import l2_shapley as SH
import l2_c6_audit as AUD   # ARCH/AW/ANAMES/ndcg50/load

DS = ("2wiki_clean", "musique_clean"); UKB = "data/ukb_storage"; OUT = "results/L2/_ctrl"
ANAMES = AUD.ANAMES; AW = AUD.AW; N = 5; K0 = SH.K0


def _hq(ds):
    q = np.load(f"{UKB}/{ds}/gte_qwen/queries_all.npy").astype(np.float32)
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)


def build_features(ds, split, log=print):
    A, d = AUD.load(ds, split); off = A["query_offsets"]; nq = len(off) - 1
    M = np.load(f"{d}/expert_meta.npz"); row_all = M["row_all"].astype(np.int64)
    disagree = M["dense_splade_disagreement"].astype(np.float32); relany = M["relation_any_signal_present"].astype(np.float32)
    relnc = M["relation_num_candidates"].astype(np.float32)
    H = _hq(ds)
    feats = []; ndcg_mat = []; labels = []; keep_qi = []; t0 = time.time()
    for qi in range(nq):
        s, e = int(off[qi]), int(off[qi + 1]); n = e - s
        lab = A["labels"][s:e]; gold = np.where(lab == 1)[0]
        if len(gold) == 0: continue
        C = SH.contribs(A["dense_score"][s:e], A["splade_scope_score"][s:e], A["offset_score"][s:e],
                        A["mixture_score"][s:e], A["relation_qwen_score"][s:e], A["relation_mask"][s:e])  # (5,n)
        c0 = C.sum(0); order = np.argsort(-c0, kind="stable")
        top = order[:20]
        # per-expert summary stats over the C0 top-20 (inference-safe): mean + max RRF contribution, win-share
        mean_top = C[:, top].mean(1); max_top = C.max(1)
        win = np.zeros(N)
        for t in top[:10]:
            win[int(np.argmax(C[:, t]))] += 1
        win = win / 10.0
        relfrac = float((A["relation_mask"][s:e] > 0).sum()) / n
        qdiag = np.array([disagree[qi], relany[qi], relfrac, np.log1p(relnc[qi]), np.log1p(n)], np.float32)
        f = np.concatenate([H[row_all[qi]], qdiag, mean_top, max_top, win]).astype(np.float32)  # 1536+5+15
        nd = np.array([AUD.ndcg50((AW[a][:, None] * C).sum(0), gold) for a in ANAMES], np.float32)
        feats.append(f); ndcg_mat.append(nd); labels.append(int(nd.argmax())); keep_qi.append(qi)
        if (qi + 1) % 4000 == 0: log(f"  [{ds}/{split}] feat {qi+1}/{nq} {time.time()-t0:.0f}s")
    out = dict(feats=np.stack(feats), ndcg=np.stack(ndcg_mat), labels=np.array(labels, np.int64), qi=np.array(keep_qi, np.int64))
    np.savez(f"{OUT}/_c6_feat_{ds}_{split}.npz", **out)
    log(f"FEAT {ds}/{split} nq={len(labels)} dim={out['feats'].shape[1]} {time.time()-t0:.0f}s")
    return out


def macro_f1(y, p, ncls):
    fs = []
    for c in range(ncls):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(fs))


def balanced_acc(y, p, ncls):
    recs = []
    for c in range(ncls):
        m = y == c
        if m.sum(): recs.append(np.mean(p[m] == c))
    return float(np.mean(recs))


def main():
    log = lambda *a: print(*a, flush=True)
    data = {}
    for ds in DS:
        for sp in ("train", "val"):
            fp = f"{OUT}/_c6_feat_{ds}_{sp}.npz"
            data[(ds, sp)] = {k: v for k, v in np.load(fp).items()} if os.path.exists(fp) else build_features(ds, sp, log)
    # pool datasets
    Xtr = np.concatenate([data[(ds, "train")]["feats"] for ds in DS]); ytr = np.concatenate([data[(ds, "train")]["labels"] for ds in DS])
    Xva = np.concatenate([data[(ds, "val")]["feats"] for ds in DS]); yva = np.concatenate([data[(ds, "val")]["labels"] for ds in DS])
    ndva = np.concatenate([data[(ds, "val")]["ndcg"] for ds in DS])   # (nqv,6)
    # standardize (fit on TRAIN)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    ncls = len(ANAMES)
    # class-balanced multinomial logistic (tiny: single linear layer)
    torch.manual_seed(0)
    clf = nn.Linear(Xtr.shape[1], ncls)
    cnt = np.bincount(ytr, minlength=ncls); cw = torch.from_numpy((cnt.sum() / (ncls * np.clip(cnt, 1, None))).astype(np.float32))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    for ep in range(300):
        opt.zero_grad(); logit = clf(Xt); loss = F.cross_entropy(logit, yt, weight=cw); loss.backward(); opt.step()
    with torch.no_grad():
        pva = clf(torch.from_numpy(Xva)).argmax(1).numpy()
    # classification metrics
    acc = float(np.mean(pva == yva)); mf1 = macro_f1(yva, pva, ncls); bacc = balanced_acc(yva, pva, ncls)
    # DECISIVE: retrieval NDCG@50 of predicted archetype vs equal / oracle
    eq_idx = ANAMES.index("equal")
    nd_pred = float(ndva[np.arange(len(pva)), pva].mean())
    nd_equal = float(ndva[:, eq_idx].mean()); nd_oracle = float(ndva.max(1).mean())
    # confusion of predictions
    from collections import Counter
    pred_dist = {ANAMES[c]: int((pva == c).sum()) for c in range(ncls)}
    true_dist = {ANAMES[c]: int((yva == c).sum()) for c in range(ncls)}
    # also: retrieval NDCG if we only trust confident non-equal predictions? report plain argmax (spec).
    res = {"features": "query_emb(1536)+qdiag(5:[disagree,relany,relfrac,log1p_relnc,log1p_n])+expert_stats(15:mean/max/win)",
           "classifier": "class-balanced multinomial logistic (single linear layer), trained on TRAIN",
           "n_train": int(len(ytr)), "n_val": int(len(yva)),
           "classification": {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4), "balanced_accuracy": round(bacc, 4)},
           "retrieval_ndcg50": {"equal_C0": round(nd_equal, 4), "predicted_archetype": round(nd_pred, 4),
                                "oracle_ceiling": round(nd_oracle, 4),
                                "predicted_minus_equal": round(nd_pred - nd_equal, 4),
                                "frac_of_oracle_headroom_captured": round((nd_pred - nd_equal) / (nd_oracle - nd_equal), 4) if nd_oracle > nd_equal else None},
           "val_pred_distribution": pred_dist, "val_true_oracle_distribution": true_dist}
    json.dump(res, open(f"{OUT}/_c6_predict.json", "w"), indent=1, default=str)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
