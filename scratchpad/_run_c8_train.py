"""C8a / C8b — soft archetype distillation aimed at top-5, selected on DEV_INNER. Tasks 4,5,6.
Same classifier family + same archetype set as C7b; ONLY the TRAIN utility target changes.
  C7b_inner : argmax NDCG@50   (baseline, refit on TRAIN_INNER for a clean DEV comparison)
  C8a       : argmax NDCG@5
  C8b       : argmax (lambda*NDCG@5 + (1-lambda)*NDCG@50), one declared lambda (0.65), only if C8a hurts deep recall
Selection PRIMARY NDCG@5, SECONDARY recall5/all5feas, GUARDRAILS ndcg50/mrr/all50. Writes _c8_train.json."""
import sys, os, json
sys.path.insert(0, "scratchpad")
import numpy as np, l2_c8 as C8
OUT = C8.OUT; log = lambda *a: print(*a, flush=True)
LAMBDA = 0.65
GUARD = {"ndcg50": 0.005, "mrr": 0.010, "all50": 0.010}   # material-harm thresholds (dev pooled drop)


def fit_head_target(target_name, inner):
    if target_name in ("ndcg5", "ndcg50"):
        return C8.fit_soft_head(target_name, inner)
    # mixture target: build combined utility array per ds, argmax -> labels via a temp field
    Xs, ys = [], []
    import torch, torch.nn as nn, torch.nn.functional as F
    for ds in C8.DS:
        A = C8.precompute_arch(ds, "train"); pos = {int(q): r for r, q in enumerate(A["qi"])}
        rows = [pos[int(q)] for q in inner[ds]]
        U = LAMBDA * A["ndcg5"][rows] + (1 - LAMBDA) * A["ndcg50"][rows]
        Xs.append(C8.feats_for(ds, "train", inner[ds])); ys.append(U.argmax(1))
    X = np.concatenate(Xs).astype(np.float32); y = np.concatenate(ys).astype(np.int64)
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32)); yt = torch.from_numpy(y)
    torch.manual_seed(0); clf = nn.Linear(X.shape[1], 6)
    cnt = np.bincount(y, minlength=6); cw = torch.from_numpy((cnt.sum() / (6 * np.clip(cnt, 1, None))).astype(np.float32))
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(300):
        opt.zero_grad(); F.cross_entropy(clf(Xn), yt, weight=cw).backward(); opt.step()
    return clf, mu, sd, {"label_dist": {C8.ANAMES[c]: int((y == c).sum()) for c in range(6)}, "lambda": LAMBDA}


def eval_on_dev(clf, mu, sd, dev, want_ranks=False):
    rb = {}; ranks = {}
    for ds in C8.DS:
        qi = dev[ds]; W = C8.soft_weights(clf, mu, sd, ds, "train", qi)
        r = C8.eval_policy(ds, "train", qi, W, want_perq=True, want_ranks=want_ranks); rb[ds] = r
        if want_ranks: ranks[ds] = {(q, gl): rk for q, gl, rk in r["_goldranks"]}
    pooled = C8.pool(rb)
    return rb, pooled, ranks


def rescue(ranks_a, ranks_b):
    """golds moved into top5 (b vs a) by source bin, and golds pushed out. a=baseline(C7b), b=candidate."""
    src_bins = ["5-9", "10-19", "20-49", "50-99", "100-499", "500+"]
    resc = {b: 0 for b in src_bins}; hurt = 0; net_in = 0
    for key, ra in ranks_a.items():
        rb = ranks_b.get(key)
        if rb is None: continue
        a_in = ra < 5; b_in = rb < 5
        if b_in and not a_in:
            net_in += 1; resc[C8.RBNAMES[C8.rankbin(int(ra))]] = resc.get(C8.RBNAMES[C8.rankbin(int(ra))], 0) + 1
        elif a_in and not b_in:
            hurt += 1
    return {"rescued_into_top5_by_source_bin": {b: resc.get(b, 0) for b in src_bins},
            "TOP5_GOLD_RESCUE": sum(resc.values()), "TOP5_GOLD_HURT": hurt,
            "NET_TOP5_GOLD_GAIN": sum(resc.values()) - hurt}


def slim(pooled):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pooled.items() if k != "n"}


def main():
    inner = {ds: C8.load_split(ds)[0] for ds in C8.DS}; dev = {ds: C8.load_split(ds)[1] for ds in C8.DS}
    heads = {}
    heads["C7b_inner"] = C8.fit_soft_head("ndcg50", inner)
    heads["C8a"] = C8.fit_soft_head("ndcg5", inner)

    rb0, p0, rk0 = eval_on_dev(*heads["C7b_inner"][:3], dev, want_ranks=True)
    rbA, pA, rkA = eval_on_dev(*heads["C8a"][:3], dev, want_ranks=True)
    log("C7b_inner dev pooled: " + json.dumps(slim(p0)))
    log("C8a       dev pooled: " + json.dumps(slim(pA)))

    def harmed(pc):
        return {g: round(p0[g] - pc[g], 4) for g in GUARD if (p0[g] - pc[g]) > GUARD[g]}
    a_harm = harmed(pA)
    res = {"lambda_c8b": LAMBDA, "guard_thresholds": GUARD,
           "label_dists": {"C7b_inner": heads["C7b_inner"][3], "C8a": heads["C8a"][3]},
           "dev": {"C7b_inner": {"pooled": slim(p0), "per_ds": {ds: slim(rb0[ds]) for ds in C8.DS}},
                   "C8a": {"pooled": slim(pA), "per_ds": {ds: slim(rbA[ds]) for ds in C8.DS}}},
           "delta_C8a_minus_C7b": {k: round(pA[k] - p0[k], 4) for k in C8.AGG if isinstance(pA[k], float)},
           "C8a_guardrail_harm": a_harm,
           "rescue_C8a_vs_C7b": {ds: rescue(rk0[ds], rkA[ds]) for ds in C8.DS}}
    # pooled rescue
    res["rescue_C8a_vs_C7b"]["pooled"] = {
        "TOP5_GOLD_RESCUE": sum(res["rescue_C8a_vs_C7b"][ds]["TOP5_GOLD_RESCUE"] for ds in C8.DS),
        "TOP5_GOLD_HURT": sum(res["rescue_C8a_vs_C7b"][ds]["TOP5_GOLD_HURT"] for ds in C8.DS),
        "NET_TOP5_GOLD_GAIN": sum(res["rescue_C8a_vs_C7b"][ds]["NET_TOP5_GOLD_GAIN"] for ds in C8.DS)}

    # decide whether C8b needed: C8a improves ndcg5 but harms a guardrail
    need_c8b = (pA["ndcg5"] > p0["ndcg5"]) and bool(a_harm)
    res["C8b_triggered"] = bool(need_c8b)
    selected = "C8a"
    if need_c8b or True:   # always compute C8b for completeness of the Pareto comparison
        heads["C8b"] = fit_head_target("mix", inner)
        rbB, pB, rkB = eval_on_dev(*heads["C8b"][:3], dev, want_ranks=True)
        log("C8b       dev pooled: " + json.dumps(slim(pB)))
        b_harm = harmed(pB)
        res["label_dists"]["C8b"] = heads["C8b"][3]
        res["dev"]["C8b"] = {"pooled": slim(pB), "per_ds": {ds: slim(rbB[ds]) for ds in C8.DS}}
        res["delta_C8b_minus_C7b"] = {k: round(pB[k] - p0[k], 4) for k in C8.AGG if isinstance(pB[k], float)}
        res["C8b_guardrail_harm"] = b_harm
        res["rescue_C8b_vs_C7b"] = {ds: rescue(rk0[ds], rkB[ds]) for ds in C8.DS}
        res["rescue_C8b_vs_C7b"]["pooled"] = {
            "TOP5_GOLD_RESCUE": sum(res["rescue_C8b_vs_C7b"][ds]["TOP5_GOLD_RESCUE"] for ds in C8.DS),
            "TOP5_GOLD_HURT": sum(res["rescue_C8b_vs_C7b"][ds]["TOP5_GOLD_HURT"] for ds in C8.DS),
            "NET_TOP5_GOLD_GAIN": sum(res["rescue_C8b_vs_C7b"][ds]["NET_TOP5_GOLD_GAIN"] for ds in C8.DS)}
        # selection: prefer highest ndcg5 among guardrail-passing; else best guardrail-respecting top5 gain
        cands = []
        for tag, pc, harm in (("C8a", pA, a_harm), ("C8b", pB, b_harm)):
            cands.append((tag, pc["ndcg5"], bool(harm), pc))
        passing = [c for c in cands if not c[2] and c[1] > p0["ndcg5"]]
        if passing:
            selected = max(passing, key=lambda c: c[1])[0]
        else:
            # none pass cleanly: pick the one with best ndcg5 that still keeps deep recall closest (min harm), if it beats C7b top5
            improving = [c for c in cands if c[1] > p0["ndcg5"]]
            selected = (max(improving, key=lambda c: c[1])[0] if improving else "C7b")
    res["SELECTED_ON_DEV"] = selected
    log(f"SELECTED_ON_DEV = {selected}")
    json.dump(res, open(f"{OUT}/_c8_train.json", "w"), indent=1, default=str)
    log("C8_TRAIN_DONE wrote results/L2/_ctrl/_c8_train.json")


if __name__ == "__main__":
    main()
