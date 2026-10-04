"""C9 prep: nested C9_TRAIN/C9_DEV split (85/15, deterministic stratified) INSIDE old TRAIN_INNER; keep old
DEV_INNER as secondary held-out. Write C9_FEATURE_MANIFEST.json (Task 5: EXISTS vs NEW audit)."""
import sys, os, json
sys.path.insert(0, "scratchpad")
import numpy as np, l2_c8 as C8, l2_c9 as C9
OUT = C8.OUT; log = lambda *a: print(*a, flush=True)


def nested_split(dev_frac=0.15):
    res = {}
    for ds in C8.DS:
        inner, _ = C8.load_split(ds)
        A = C8.precompute_arch(ds, "train"); pos = {int(q): r for r, q in enumerate(A["qi"])}
        ng = A["ng"]; rel = A["relany"]; dis = A["disagree"]; med = np.median(dis[[pos[int(q)] for q in inner]])
        strat = {}
        for qi in inner:
            r = pos[int(qi)]; key = (int(min(ng[r], 3)), int(rel[r] > 0.5), int(dis[r] > med))
            strat.setdefault(key, []).append((int(qi), C8._h(qi)))
        dev = []; tr = []
        for key, lst in strat.items():
            lst.sort(key=lambda t: t[1]); k = int(round(len(lst) * dev_frac))
            dev += [q for q, _ in lst[:k]]; tr += [q for q, _ in lst[k:]]
        c9tr = np.array(sorted(tr), np.int64); c9dev = np.array(sorted(dev), np.int64)
        np.savez(f"{OUT}/_c9_split_{ds}.npz", c9train=c9tr, c9dev=c9dev)
        res[ds] = {"c9train": int(len(c9tr)), "c9dev": int(len(c9dev)), "old_inner": int(len(inner))}
        log(f"C9SPLIT {ds}: c9train={len(c9tr)} c9dev={len(c9dev)} (of inner {len(inner)})")
    return res


def manifest():
    """Map every C9-requested feature to EXISTS (in the 28 C8c features) or NEW, with the implemented column."""
    exists = {n: n for n in C9.C8C28}
    M = {"C8c_existing_28": C9.C8C28,
         "requested_features": {
             # Task2 expert support
             "best_expert_rank": ("NEW", "best_rank5 (min over all 5 experts; min_rk4 was over 4)"),
             "second_best_expert_rank": ("NEW", "second_min_rank5"),
             "mean_expert_rank": ("NEW", "mean_rank5 (mean_rk4 was over 4)"),
             "median_expert_rank": ("NEW", "median_rank5"),
             "rank_std": ("NEW", "rank_std5"),
             "best_RRF_contribution": ("EXISTS", "max_contrib"),
             "second_best_contribution": ("NEW", "second_contrib"),
             "mean_contribution": ("NEW", "mean_contrib"),
             "std_contribution": ("NEW", "std_contrib"),
             "best_minus_second_gap": ("NEW", "best_minus_second_contrib"),
             "best_over_mean_ratio": ("NEW", "max_mean_ratio"),
             "N_EXPERT_TOP3": ("NEW", "n_top3"),
             "N_EXPERT_TOP5": ("NEW", "n_top5"),
             "N_EXPERT_TOP10": ("EXISTS", "votes_top10"),
             "N_EXPERT_TOP20": ("NEW", "n_top20"),
             "ARGMAX_EXPERT_onehot": ("NEW", "argmax_is_{dense,splade,offset,mixture,relation}"),
             # Task3 regime x candidate
             "w_i*r_i (5)": ("NEW", "wr_{dense,splade,offset,mixture,relation}"),
             "WEIGHT_OF_ARGMAX_EXPERT": ("NEW", "weight_of_argmax_expert"),
             "TRUSTED_MAX_CONTRIB": ("NEW", "trusted_max_contrib = max_contrib * weight_of_argmax_expert"),
             "REGIME_SUPPORT_SUM": ("EXISTS", "base_score == sum_i w_i*r_i (C7b fused score)"),
             "REGIME_SUPPORT_MAX": ("NEW", "regime_support_max"),
             "REGIME_SUPPORT_SECOND": ("NEW", "regime_support_second"),
             # Task4 consensus
             "TOP5_VOTE_COUNT": ("NEW", "n_top5 (alias)"),
             "TOP10_VOTE_COUNT": ("EXISTS", "votes_top10"),
             "TOP20_VOTE_COUNT": ("NEW", "n_top20 (alias)"),
             "MIN_EXPERT_RANK": ("NEW", "best_rank5 (alias)"),
             "SECOND_MIN_EXPERT_RANK": ("NEW", "second_min_rank5 (alias)"),
             "RANK_DISPERSION": ("NEW", "rank_std5 (alias)"),
             "CONTRIB_ENTROPY": ("NEW", "contrib_entropy"),
             "MAX_SECOND_RATIO": ("NEW", "max_second_ratio"),
             "MAX_MEAN_RATIO": ("NEW", "max_mean_ratio (alias of best_over_mean_ratio)"),
             # Task8 C8c residual (OOF)
             "C8C_SCORE": ("NEW", "c8c_oof_score (OOF on C9_TRAIN)"),
             "C8C_RANK": ("NEW", "c8c_oof_rank"),
             "C8C_RANK_MINUS_5": ("NEW", "c8c_rank_minus5"),
             "C8C_SCORE_MINUS_SCORE_AT_RANK5": ("NEW", "c8c_score_minus_rank5"),
             "C8C_SCORE_MINUS_SCORE_AT_RANK1": ("NEW", "c8c_score_minus_rank1"),
             "C8C_RANK_PERCENTILE_top50": ("NEW", "c8c_rank_percentile"),
         },
         "C9_final_feature_order": C9.C9_FEATNAMES,
         "n_features_total": len(C9.C9_FEATNAMES),
         "note": ("EXISTS = already in the 28 C8c features (not duplicated). NEW = genuinely new information added. "
                  "Aliases collapse duplicate requests to one column. OOF = out-of-fold C8c predictions used to "
                  "avoid stacking leakage (Task 9).")}
    return M


def main():
    sp = nested_split(); man = manifest()
    json.dump(man, open("results/L2/L2_C9_FEATURE_MANIFEST.json", "w"), indent=1)
    json.dump(sp, open(f"{OUT}/_c9_split_meta.json", "w"), indent=1)
    log(f"MANIFEST features={man['n_features_total']} (28 C8c + {len(C9.EXTRA)} extra + {len(C9.RESID)} resid)")
    log("C9_PREP_DONE " + json.dumps(sp))


if __name__ == "__main__":
    main()
