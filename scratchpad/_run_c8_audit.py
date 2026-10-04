"""C8 TOP-5 FAILURE AUDIT — Tasks 1,2,3,8. Audits the deployed C7b (full-TRAIN soft-archetype, U50) on
TRAIN_INNER / DEV_INNER / VAL. Writes results/L2/L2_TOP5_AUDIT.{md,json}. No TEST beyond VAL reporting."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np, l2_c8 as C8
from collections import Counter
ANAMES = C8.ANAMES; OUT = C8.OUT; log = lambda *a: print(*a, flush=True)


def c7b_full_train_head():
    """The deployed C7b: soft archetype head fit on FULL TRAIN with NDCG@50 argmax labels."""
    qi_by_ds = {ds: C8.precompute_arch(ds, "train")["qi"] for ds in C8.DS}
    return C8.fit_soft_head("ndcg50", qi_by_ds)


def failure_buckets(ds, split, qi_list, W):
    """Task1: per-query category (A/B/C/D/E) + per-gold C7b rank bins. Full L1 coverage in this corpus =>
    A/B only for degenerate gold_count=0 (already excluded from valid)."""
    import l2_c6 as C6
    R = C6._load_raw(ds, split); off = R["query_offsets"]
    cat = Counter(); bins = Counter(); ng_hist = Counter()
    for r, qi in enumerate(qi_list):
        s, e = int(off[qi]), int(off[qi + 1]); gold = np.where(R["labels"][s:e] == 1)[0]; ng = len(gold)
        C = C8.contribs_of(R, s, e); score = (W[r][:, None] * C).sum(0); m = C8.topk_metrics(score, gold)
        gr = m["gr"]; in5 = int((gr < 5).sum())
        if in5 == 0: cat["C_all_in_scope_none_top5"] += 1
        elif in5 < ng: cat["D_some_top5_others_below"] += 1
        else: cat["E_all_in_scope_top5"] += 1
        ng_hist[min(ng, 6)] += 1
        for x in gr: bins[C8.RBNAMES[C8.rankbin(int(x))]] += 1
    tot_g = sum(bins.values())
    return {"n_queries": len(qi_list), "category_counts": dict(cat),
            "category_frac": {k: round(v / len(qi_list), 4) for k, v in cat.items()},
            "gold_rank_bins": {b: bins.get(b, 0) for b in C8.RBNAMES},
            "gold_rank_bin_frac": {b: round(bins.get(b, 0) / max(tot_g, 1), 4) for b in C8.RBNAMES},
            "ng_hist": dict(ng_hist), "note": "L1 coverage complete by corpus construction; A/B empty (see top-level note)"}


def top50_oracle(ds, split, qi_list, W):
    """Task8: fraction of required golds inside C7b top50 (max pool for a top-50 reranker)."""
    import l2_c6 as C6
    R = C6._load_raw(ds, split); off = R["query_offsets"]
    g_tot = 0; in50 = 0; in50_not5 = 0; miss50 = 0; q_all50 = 0; q_all50_present = 0
    for r, qi in enumerate(qi_list):
        s, e = int(off[qi]), int(off[qi + 1]); gold = np.where(R["labels"][s:e] == 1)[0]; ng = len(gold)
        C = C8.contribs_of(R, s, e); score = (W[r][:, None] * C).sum(0); gr = C8.ranks_from_score(score)[gold]
        g_tot += ng; a = int((gr < 50).sum()); in50 += a; miss50 += ng - a; in50_not5 += int(((gr >= 5) & (gr < 50)).sum())
        if int(gr.max()) < 50: q_all50 += 1
    return {"n_queries": len(qi_list), "golds_total": g_tot, "golds_in_top50": in50, "golds_missing_top50": miss50,
            "frac_golds_in_top50": round(in50 / max(g_tot, 1), 4),
            "golds_top50_present_not_top5": in50_not5,
            "frac_present_not_top5_of_pool": round(in50_not5 / max(in50, 1), 4),
            "queries_all_golds_in_top50": q_all50}


def arch_top5_oracle(ds, qi_by_ds_train):
    """Task3: TRAIN-only oracle archetype behaviour under NDCG@5 vs the C7 NDCG@50 target; winner agreement."""
    A = C8.precompute_arch(ds, "train"); pos = {int(q): r for r, q in enumerate(A["qi"])}
    rows = [pos[int(q)] for q in qi_by_ds_train[ds]]
    nd5 = A["ndcg5"][rows]; nd50 = A["ndcg50"][rows]; rec5 = A["recall5"][rows]; a5f = A["all5feas"][rows]
    w5 = nd5.argmax(1); w50 = nd50.argmax(1)
    same = float((w5 == w50).mean())
    conf = np.zeros((6, 6), int)
    for a, b in zip(w50, w5): conf[a, b] += 1
    eqi = ANAMES.index("equal")
    with np.errstate(invalid="ignore"):
        a5f_mean = np.nanmean(a5f, 0)
    return {"n": len(rows), "same_winner_frac_ndcg5_vs_ndcg50": round(same, 4),
            "equal_ndcg5": round(float(nd5[:, eqi].mean()), 4), "oracle_ndcg5": round(float(nd5.max(1).mean()), 4),
            "oracle_ndcg5_gain": round(float(nd5.max(1).mean() - nd5[:, eqi].mean()), 4),
            "equal_recall5": round(float(rec5[:, eqi].mean()), 4), "oracle_recall5": round(float(rec5.max(1).mean()), 4),
            "arch_mean_ndcg5": {ANAMES[i]: round(float(nd5[:, i].mean()), 4) for i in range(6)},
            "arch_mean_recall5": {ANAMES[i]: round(float(rec5[:, i].mean()), 4) for i in range(6)},
            "arch_mean_all5feas": {ANAMES[i]: (round(float(a5f_mean[i]), 4) if not np.isnan(a5f_mean[i]) else None) for i in range(6)},
            "ndcg5_winner_counts": {ANAMES[i]: int((w5 == i).sum()) for i in range(6)},
            "ndcg50_winner_counts": {ANAMES[i]: int((w50 == i).sum()) for i in range(6)},
            "confusion_ndcg50_to_ndcg5": {ANAMES[i]: {ANAMES[j]: int(conf[i, j]) for j in range(6)} for i in range(6)}}


def eval_c7b(clf, mu, sd, split_map):
    """C7b top5 baseline on each of the 3 splits (per-ds + pooled)."""
    out = {}
    for name, per_ds in split_map.items():
        rb = {}
        for ds in C8.DS:
            qi = per_ds[ds]; W = C8.soft_weights(clf, mu, sd, ds, "train" if name != "val" else "val", qi)
            rb[ds] = C8.eval_policy(ds, "train" if name != "val" else "val", qi, W, want_perq=False)
        out[name] = {"per_ds": {ds: {k: round(rb[ds][k], 4) if rb[ds][k] is not None else None
                                     for k in C8.AGG} for ds in C8.DS},
                     "pooled": {k: (round(v, 4) if v is not None else None) for k, v in C8.pool(rb).items()}}
    return out


def main():
    clf, mu, sd, meta = c7b_full_train_head()
    inner = {ds: C8.load_split(ds)[0] for ds in C8.DS}; dev = {ds: C8.load_split(ds)[1] for ds in C8.DS}
    val = {ds: C8.precompute_arch(ds, "val")["qi"] for ds in C8.DS}
    split_map = {"train_inner": inner, "dev_inner": dev, "val": val}

    res = {"note": ("Within l2_corpus, gold_count == in-scope gold count for EVERY query (full L1 coverage by "
                    "construction; only a few degenerate gold_count=0 queries, already excluded from 'valid'). "
                    "Therefore Task-1 categories A (L1_ANY_FAIL) and B (L1_PARTIAL) are empty HERE — the pre-P50 "
                    "L1 scope-coverage failure population lives upstream and is out of C8 scope (Task 9). All "
                    "audited top-5 failures are the recoverable L2-ranking population (C/D/E)."),
           "c7b_head": {"label": "soft archetype, argmax-NDCG@50 labels, full-TRAIN fit", **meta}}

    res["C7B_TOP5_BASELINE"] = eval_c7b(clf, mu, sd, split_map)
    log("C7B baseline done"); log(json.dumps(res["C7B_TOP5_BASELINE"]["dev_inner"]["pooled"], indent=1))

    # Task1 failure buckets (all three splits)
    res["TOP5_FAILURE_BUCKETS"] = {}
    for name, per_ds in split_map.items():
        sp = "val" if name == "val" else "train"
        pd = {}
        for ds in C8.DS:
            qi = per_ds[ds]; W = C8.soft_weights(clf, mu, sd, ds, sp, qi); pd[ds] = failure_buckets(ds, sp, qi, W)
        res["TOP5_FAILURE_BUCKETS"][name] = pd
    log("Task1 buckets done")

    # Task2 expert oracle / union — on DEV_INNER (design split) + VAL (reporting)
    res["TOP5_EXPERT_ORACLE"] = {}
    for name in ("dev_inner", "val"):
        per_ds = split_map[name]; sp = "val" if name == "val" else "train"; pd = {}
        for ds in C8.DS:
            qi = per_ds[ds]; W = C8.soft_weights(clf, mu, sd, ds, sp, qi); pd[ds] = C8.expert_analysis(ds, sp, qi, W)
        res["TOP5_EXPERT_ORACLE"][name] = pd
    log("Task2 expert oracle done")

    # Task3 archetype top5 oracle (TRAIN_INNER only)
    res["TOP5_ARCHETYPE_ORACLE"] = {ds: arch_top5_oracle(ds, inner) for ds in C8.DS}
    log("Task3 archetype oracle done")

    # Task8 top50 oracle (DEV_INNER + VAL)
    res["TOP50_ORACLE"] = {}
    for name in ("dev_inner", "val"):
        per_ds = split_map[name]; sp = "val" if name == "val" else "train"; pd = {}
        for ds in C8.DS:
            qi = per_ds[ds]; W = C8.soft_weights(clf, mu, sd, ds, sp, qi); pd[ds] = top50_oracle(ds, sp, qi, W)
        res["TOP50_ORACLE"][name] = pd
    log("Task8 top50 oracle done")

    json.dump(res, open("results/L2/L2_TOP5_AUDIT.json", "w"), indent=1, default=str)
    log("AUDIT_DONE wrote results/L2/L2_TOP5_AUDIT.json")


if __name__ == "__main__":
    main()
