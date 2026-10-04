"""Track-1 EXPERT-SYMMETRIC adversarial source audit (source/train ONLY, parameter-free, NO training, NO test).

We do NOT single out offset. For EVERY expert e in {dense,offset,splade,relation,path} we inject cross-query
candidates that e itself scores highly (pooled-z, comparable across queries) as (assumed) NONGOLDS -> symmetric
"confident-but-wrong e" examples, balanced quota J per expert. Then we audit original vs adversarial pools:

 per expert:  P(top1 nongold), P(top5 has hard-neg above gold), mean gold rank_pct, standalone R@5,
              uncorroborated-high-conf rate, + mining-type usable-failure counts (A..E).
 per pool:    realistic oracle R@5 (any-expert top5), mean top1 disagreement.
 + exact 32-subset Shapley (phi + pairwise interaction) ORIGINAL vs ADVERSARIAL, two utilities.

The ONLY prior used is the generic principle: confident single-expert evidence may be wrong. No test, no test-tuned
thresholds. J and K_e are fixed a-priori (not chosen against any performance). Absence of failures is a diagnostic,
reported, never fabricated.
"""
import json, itertools, numpy as np, torch
from src.experiments.kg_hybrid import NE, EXPERT_NAMES, EXPERTS, _load, _pooled_musd
from src.experiments import crag_fusion as CF
from src.experiments import crag_gates as G
from scratchpad import shapley_audit as SH   # reuse _util_all_subsets / weights / SUBSETS / ALL

DEV = torch.device("cpu")
DS = ["webqsp", "metaqa"]
NAMES = [EXPERT_NAMES[e] for e in range(NE)]
ALL = list(range(NE))
J_PER_EXPERT = 8        # fixed a-priori injected confident-picks per expert per query
K_TOP = 20              # "high rank" band K_e (fixed a-priori)
K5 = 5


def _pbs(data, ds, mu, sd):
    return [CF.prepbatch(r, mu, sd, DEV, ds) for r in data[ds]["train"]]


def _bank_all(pbs, mu, sd):
    """Bank of ALL train candidate rows: raw[K,7], mask[K,7], pooled-z per-EXPERT score[K,NE], source-query id,
    plus specificity margin spec[K,NE] = z_e - max_{f!=e} z_f (an e-confident, others-quiet distractor scores high)."""
    raws, masks, qids = [], [], []
    for qi, pb in enumerate(pbs):
        raws.append(pb.raw); masks.append(pb.mask)
        qids.append(torch.full((pb.raw.shape[0],), qi, dtype=torch.long))
    raw = torch.cat(raws, 0); mask = torch.cat(masks, 0); qid = torch.cat(qids, 0)
    z = (raw - mu) / sd
    zc = torch.where(mask > 0.5, z, torch.full_like(z, float("-inf")))[:, EXPERTS]   # (K,NE) invalid=-inf
    zf = torch.where(torch.isfinite(zc), zc, torch.full_like(zc, -1e9))
    spec = torch.empty_like(zc)
    for e in range(NE):
        others = torch.cat([zf[:, :e], zf[:, e+1:]], 1).max(1).values
        spec[:, e] = zc[:, e] - others                 # high only when e alone is confident
    return raw, mask, qid, zc, spec


def adv_inject(pb, qi, bank, mu, sd, e_attack, j=J_PER_EXPERT):
    """SURGICAL expert-symmetric adversarial injection. Attack ONE expert e per query (rotated across queries for
    a balanced per-expert quota). Inject j cross-query nongolds that (1) outrank e's own best gold under e, and
    (2) are e-SPECIFIC (high spec_e = confident under e, quiet under everyone else) -> e is made confidently wrong
    while the gold stays reachable by corroborating experts (oracle preserved). Returns (aug_pb, n_injected)."""
    raw, mask, qid, zc, spec = bank
    T = G.cand_tensors(pb)
    ve = (T["valid"][:, e_attack] > 0.5)
    y = pb.y > 0.5
    gm = y & ve
    if int(gm.sum()) == 0:
        return pb, 0                                   # e has no valid gold here -> nothing to attack
    best_gold_ze = float(T["score"][gm, e_attack].max())   # e's z-score on its best gold
    col = zc[:, e_attack].clone()
    col[qid == qi] = float("-inf")                     # exclude same-query rows
    # candidates: outrank the gold under e (z_e slightly above), AND e-specific (spec>0), pick smallest margin above gold
    elig = torch.isfinite(col) & (col > best_gold_ze) & (spec[:, e_attack] > 0.0)
    if int(elig.sum()) == 0:
        return pb, 0
    margin = col - best_gold_ze
    margin[~elig] = float("inf")
    k = min(j, int(elig.sum()))
    idx = torch.topk(-margin, k).indices               # smallest positive margin = minimally sufficient distractors
    x = torch.cat([pb.raw, raw[idx]], 0); m = torch.cat([pb.mask, mask[idx]], 0)
    yy = torch.cat([pb.y, torch.zeros(k, device=pb.y.device)])
    return G.make_pb(x, yy, pb.ng, m, mu, sd, pb.ds), k


def per_expert_stats(pb):
    """Return dict of per-expert arrays for one query + pool-level scalars. Parameter-free."""
    T = G.cand_tensors(pb)
    rp = T["rank_pct"].cpu().numpy()          # (n,NE) higher=better, invalid=0
    valid = T["valid"].cpu().numpy() > 0.5
    y = pb.y.cpu().numpy() > 0.5
    n = rp.shape[0]
    if y.sum() == 0:
        return None
    pos = {}
    st = {"top1_nongold": np.zeros(NE), "hardneg_top5": np.zeros(NE), "gold_rank": np.zeros(NE),
          "sa_r5": np.zeros(NE), "uncorr_hi": np.zeros(NE),
          "A": np.zeros(NE), "B": np.zeros(NE), "C": np.zeros(NE), "D": np.zeros(NE)}
    top5_sets = []
    top20_sets = []
    top1 = np.full(NE, -1)
    for e in range(NE):
        r = rp[:, e]; ve = valid[:, e]
        order = np.argsort(-r, kind="stable")
        order = [i for i in order if ve[i]]          # only valid
        top5 = order[:K5]; top20 = order[:K_TOP]
        top5_sets.append(set(top5)); top20_sets.append(set(top20))
        top1[e] = order[0] if order else -1
        gmask = y & ve
        best_gr = r[gmask].max() if gmask.any() else 0.0
        st["gold_rank"][e] = float(r[gmask].mean()) if gmask.any() else 0.0
        st["sa_r5"][e] = float(y[top5].sum()) / max(pb.ng, 1)
        if top1[e] >= 0 and not y[top1[e]]:
            st["top1_nongold"][e] = 1.0
        # A: any nongold in top20_e ranked above e's best gold
        hard = [i for i in top20 if (not y[i]) and r[i] > best_gr + 1e-9]
        if hard: st["A"][e] = 1.0
        if any((not y[i]) and r[i] > best_gr + 1e-9 for i in top5): st["hardneg_top5"][e] = 1.0
    # B: e top1 nongold AND some other expert top1 gold
    for e in range(NE):
        if top1[e] >= 0 and not y[top1[e]]:
            if any(top1[f] >= 0 and y[top1[f]] for f in range(NE) if f != e):
                st["B"][e] = 1.0
    # C: e has a nongold in top5 that NO other expert has in top20  (uncorroborated confident-wrong)
    for e in range(NE):
        for i in top5_sets[e]:
            if not y[i] and all(i not in top20_sets[f] for f in range(NE) if f != e):
                st["C"][e] = 1.0; st["uncorr_hi"][e] = 1.0; break
    # D: false consensus — a nongold in >=2 experts' top5. Credit to each participating expert.
    from collections import Counter
    cnt = Counter()
    for e in range(NE):
        for i in top5_sets[e]:
            if not y[i]: cnt[i] += 1
    fc = {i for i, c in cnt.items() if c >= 2}
    for e in range(NE):
        if any(i in fc for i in top5_sets[e]): st["D"][e] = 1.0
    # pool-level
    any_top5 = set()
    for e in range(NE): any_top5 |= set(list(top5_sets[e]))
    oracle_r5 = float(sum(1 for i in any_top5 if y[i])) / max(pb.ng, 1)
    disc = np.mean([top1[e] != top1[f] for e, f in itertools.combinations(range(NE), 2)
                    if top1[e] >= 0 and top1[f] >= 0]) if n else 0.0
    return st, oracle_r5, float(disc)


def agg_pool(pbs):
    S = {k: [] for k in ["top1_nongold", "hardneg_top5", "gold_rank", "sa_r5", "uncorr_hi", "A", "B", "C", "D"]}
    orc, dis = [], []
    for pb in pbs:
        out = per_expert_stats(pb)
        if out is None: continue
        st, o, d = out
        for k in S: S[k].append(st[k])
        orc.append(o); dis.append(d)
    A = {k: np.mean(v, 0) for k, v in S.items()}
    counts = {k: np.sum(v, 0).astype(int) for k, v in S.items() if k in ("A", "B", "C", "D")}
    return A, counts, float(np.mean(orc)), float(np.mean(dis)), len(orc)


def shapley_pool(pbs):
    acc = {u: {"phi": [], "inter": {p: [] for p in itertools.combinations(ALL, 2)}} for u in ("rank", "R5")}
    nq = 0
    for pb in pbs:
        sh = SH.query_shapley(pb)
        if sh is None: continue
        nq += 1
        for u in ("rank", "R5"):
            acc[u]["phi"].append(sh[u]["phi"])
            for p in acc[u]["inter"]: acc[u]["inter"][p].append(sh[u]["inter"][p])
    res = {}
    for u in ("rank", "R5"):
        phi = np.mean(acc[u]["phi"], 0)
        inter = {f"{NAMES[e]}+{NAMES[f]}": round(float(np.mean(acc[u]["inter"][(e, f)])), 4)
                 for e, f in itertools.combinations(ALL, 2)}
        res[u] = {"phi": {NAMES[e]: round(float(phi[e]), 4) for e in ALL}, "interaction": inter}
    return res, nq


def main():
    data = _load(DS, DEV)
    out = {"_config": {"J_per_expert": J_PER_EXPERT, "K_top": K_TOP, "note": "expert-symmetric; y_injected=0; source-only"}}
    for ds in DS:
        mu, sd = _pooled_musd(data, [ds], DEV, "train")
        pbs = _pbs(data, ds, mu, sd)
        bank = _bank_all(pbs, mu, sd)
        # rotate the attacked expert across queries -> balanced per-expert quota; count successful attacks vs skips
        adv, added_tot, attempted, succeeded = [], np.zeros(NE), np.zeros(NE), np.zeros(NE)
        for qi, pb in enumerate(pbs):
            e_attack = qi % NE
            attempted[e_attack] += 1
            apb, added = adv_inject(pb, qi, bank, mu, sd, e_attack)
            adv.append(apb); added_tot[e_attack] += added
            if added > 0:
                succeeded[e_attack] += 1
        stO, cntO, orcO, disO, nqO = agg_pool(pbs)
        stA, cntA, orcA, disA, nqA = agg_pool(adv)
        shO, _ = shapley_pool(pbs)
        shA, _ = shapley_pool(adv)
        out[ds] = {"n_queries": nqO,
                   "attack_success": {NAMES[e]: {"attempted": int(attempted[e]), "succeeded": int(succeeded[e]),
                                                  "rate": round(float(succeeded[e]/max(attempted[e],1)), 3),
                                                  "injected": int(added_tot[e])} for e in ALL},
                   "injected_per_expert_total": {NAMES[e]: int(added_tot[e]) for e in ALL},
                   "orig": {"per_expert": {NAMES[e]: {k: round(float(stO[k][e]), 3) for k in stO} for e in ALL},
                            "oracle_r5": round(orcO, 3), "top1_disagreement": round(disO, 3),
                            "usable_counts": {t: {NAMES[e]: int(cntO[t][e]) for e in ALL} for t in cntO}},
                   "adv": {"per_expert": {NAMES[e]: {k: round(float(stA[k][e]), 3) for k in stA} for e in ALL},
                           "oracle_r5": round(orcA, 3), "top1_disagreement": round(disA, 3),
                           "usable_counts": {t: {NAMES[e]: int(cntA[t][e]) for e in ALL} for t in cntA}},
                   "shapley_orig": shO, "shapley_adv": shA}
        # console
        print(f"\n===== {ds}  (nq={nqO}) =====")
        print("attack success (attempted/succeeded/rate):",
              {NAMES[e]: f"{int(attempted[e])}/{int(succeeded[e])}/{succeeded[e]/max(attempted[e],1):.2f}" for e in ALL})
        for tag in ("orig", "adv"):
            b = out[ds][tag]
            print(f"-- {tag}: oracle_r5={b['oracle_r5']} top1_disagree={b['top1_disagreement']}")
            print(f"   {'expert':9s} {'t1_ng':>6s} {'hn@5':>6s} {'gold_r':>7s} {'sa_R5':>6s} {'uncor':>6s}")
            for e in ALL:
                s = b["per_expert"][NAMES[e]]
                print(f"   {NAMES[e]:9s} {s['top1_nongold']:>6.2f} {s['hardneg_top5']:>6.2f} {s['gold_rank']:>7.3f} {s['sa_r5']:>6.2f} {s['uncorr_hi']:>6.2f}")
            print("   usable counts A/B/C/D:", b["usable_counts"])
        print("-- Shapley R5  orig vs adv (phi):")
        for e in ALL:
            print(f"   {NAMES[e]:9s} {shO['R5']['phi'][NAMES[e]]:>8.4f}  ->  {shA['R5']['phi'][NAMES[e]]:>8.4f}")
    json.dump(out, open("results/L2/_adv_source_audit.json", "w"), indent=2)
    print("\n-> results/L2/_adv_source_audit.json")


if __name__ == "__main__":
    main()
