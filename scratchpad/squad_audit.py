"""SQuAD substrate audit S7-S10 + WebQSP/MetaQA/SQuAD evidence-regime comparison (parameter-free, NO training,
NO test-tuning; canonical test split used for the audit numbers themselves). Reuses the frozen Shapley machinery."""
import json, itertools, numpy as np, torch
from src.experiments.kg_hybrid import NE, EXPERT_NAMES, EXPERTS, _load, _pooled_musd
from src.experiments import crag_fusion as CF
from src.experiments import crag_gates as G
from scratchpad import shapley_audit as SH

DEV = torch.device("cpu")
NAMES = [EXPERT_NAMES[e] for e in range(NE)]
ALL = list(range(NE))
DSS = ["webqsp", "metaqa", "squad_clean"]


def _pbs(data, ds, mu, sd, split):
    return [CF.prepbatch(r, mu, sd, DEV, ds) for r in data[ds][split]]


def audit_ds(pbs):
    pool_sizes, golds_per_q, poolrecall = [], [], []
    # per expert accumulators
    avail = np.zeros(NE); r1 = np.zeros(NE); r5 = np.zeros(NE); r20 = np.zeros(NE)
    grank_mean, grank_med, t1ng = [[] for _ in ALL], [[] for _ in ALL], np.zeros(NE)
    cnt_expert = np.zeros(NE)
    coop1 = coop5 = coop20 = 0.0
    # dense vs splade contingency
    DENSE, SPLADE = 0, 2
    both_c = dcorr = scorr = both_w = disc = 0
    nq = 0
    for pb in pbs:
        T = G.cand_tensors(pb)
        rp = T["rank_pct"].cpu().numpy(); valid = T["valid"].cpu().numpy() > 0.5
        y = pb.y.cpu().numpy() > 0.5
        n = rp.shape[0]
        if y.sum() == 0:
            continue
        nq += 1
        pool_sizes.append(n); golds_per_q.append(int(pb.ng)); poolrecall.append(float(y.sum()) / max(pb.ng, 1))
        # coop baseline
        coop = G.coop_maxrank(T).cpu().numpy()
        co = np.argsort(-coop, kind="stable")
        coop1 += float(y[co[:1]].sum()) / pb.ng; coop5 += float(y[co[:5]].sum()) / pb.ng; coop20 += float(y[co[:20]].sum()) / pb.ng
        top1e = np.full(NE, -1)
        for e in range(NE):
            ve = valid[:, e]
            if ve.sum() == 0:
                continue
            avail[e] += 1
            r = rp[:, e]
            order = [i for i in np.argsort(-r, kind="stable") if ve[i]]
            top1e[e] = order[0] if order else -1
            r1[e] += float(y[order[:1]].sum()) / pb.ng
            r5[e] += float(y[order[:5]].sum()) / pb.ng
            r20[e] += float(y[order[:20]].sum()) / pb.ng
            gm = y & ve
            if gm.any():
                pos = {i: k for k, i in enumerate(order)}
                gr = [pos[i] for i in np.where(gm)[0] if i in pos]
                if gr:
                    grank_mean[e].append(float(np.mean(gr))); grank_med[e].append(float(np.median(gr)))
            if top1e[e] >= 0 and not y[top1e[e]]:
                t1ng[e] += 1
            cnt_expert[e] += 1
        # dense vs splade
        dc = top1e[DENSE] >= 0 and y[top1e[DENSE]]
        sc = top1e[SPLADE] >= 0 and y[top1e[SPLADE]]
        if top1e[DENSE] >= 0 and top1e[SPLADE] >= 0 and top1e[DENSE] != top1e[SPLADE]:
            disc += 1
        both_c += int(dc and sc); dcorr += int(dc and not sc); scorr += int(sc and not dc); both_w += int(not dc and not sc)
    def rate(x): return round(float(x) / max(nq, 1), 3)
    per_expert = {}
    for e in ALL:
        av = cnt_expert[e]
        per_expert[NAMES[e]] = {
            "avail_pct": round(float(avail[e]) / max(nq, 1), 3),
            "R@1": round(100 * float(r1[e]) / max(av, 1), 2),
            "R@5": round(100 * float(r5[e]) / max(av, 1), 2),
            "R@20": round(100 * float(r20[e]) / max(av, 1), 2),
            "mean_gold_rank": round(float(np.mean(grank_mean[e])), 2) if grank_mean[e] else None,
            "median_gold_rank": round(float(np.median(grank_med[e])), 2) if grank_med[e] else None,
            "top1_nongold": round(float(t1ng[e]) / max(av, 1), 3),
        }
    ps = np.array(pool_sizes)
    return {
        "n_queries": nq,
        "pool_size": {"mean": round(float(ps.mean()), 1), "median": float(np.median(ps)), "p95": float(np.percentile(ps, 95))},
        "golds_per_query": round(float(np.mean(golds_per_q)), 2),
        "candidate_pool_gold_recall": round(float(np.mean(poolrecall)), 3),
        "coop_baseline": {"R@1": round(100 * coop1 / max(nq, 1), 2), "R@5": round(100 * coop5 / max(nq, 1), 2),
                          "R@20": round(100 * coop20 / max(nq, 1), 2)},
        "per_expert": per_expert,
        "dense_vs_splade": {"disagree_top1": rate(disc), "both_correct": rate(both_c),
                            "dense_only_correct": rate(dcorr), "splade_only_correct": rate(scorr),
                            "both_wrong": rate(both_w)},
    }


def shapley_ds(pbs):
    acc = {u: {"phi": [], "cp": [], "cn": [], "inter": {p: [] for p in itertools.combinations(ALL, 2)}} for u in ("rank", "R5")}
    nq = 0
    for pb in pbs:
        sh = SH.query_shapley(pb)
        if sh is None:
            continue
        nq += 1
        for u in ("rank", "R5"):
            acc[u]["phi"].append(sh[u]["phi"]); acc[u]["cp"].append(sh[u]["cond_pos"]); acc[u]["cn"].append(sh[u]["cond_neg"])
            for p in acc[u]["inter"]:
                acc[u]["inter"][p].append(sh[u]["inter"][p])
    out = {}
    for u in ("rank", "R5"):
        phi = np.mean(acc[u]["phi"], 0); cp = np.mean(acc[u]["cp"], 0); cn = np.mean(acc[u]["cn"], 0)
        inter = {f"{NAMES[e]}+{NAMES[f]}": round(float(np.mean(acc[u]["inter"][(e, f)])), 4)
                 for e, f in itertools.combinations(ALL, 2)}
        out[u] = {"per_expert": {NAMES[e]: {"shapley": round(float(phi[e]), 4), "cond_pos": round(float(cp[e]), 3),
                                            "cond_neg": round(float(cn[e]), 3)} for e in ALL},
                  "interaction": inter}
    return out


def main():
    data = _load(DSS, DEV)
    out = {"_config": {"split": "test", "note": "parameter-free; canonical test split; NO training/tuning"}, "datasets": {}}
    for ds in DSS:
        mu, sd = _pooled_musd(data, [ds], DEV, "train")
        pbs = _pbs(data, ds, mu, sd, "test")
        aud = audit_ds(pbs)
        shp = shapley_ds(pbs)
        out["datasets"][ds] = {"audit": aud, "shapley": shp}
        print(f"\n===== {ds}  (nq={aud['n_queries']}) =====")
        print(f" pool mean/med/p95: {aud['pool_size']} | golds/q {aud['golds_per_query']} | pool-gold-recall {aud['candidate_pool_gold_recall']}")
        print(f" coop baseline: {aud['coop_baseline']}")
        print(f" {'expert':10s} {'avail':>6s} {'R@1':>6s} {'R@5':>6s} {'R@20':>6s} {'g_rank':>7s} {'t1_ng':>6s} {'phiR5':>8s}")
        for e in ALL:
            p = aud["per_expert"][NAMES[e]]; s = shp["R5"]["per_expert"][NAMES[e]]["shapley"]
            gr = p["mean_gold_rank"] if p["mean_gold_rank"] is not None else -1
            print(f" {NAMES[e]:10s} {p['avail_pct']:>6.2f} {p['R@1']:>6.1f} {p['R@5']:>6.1f} {p['R@20']:>6.1f} {gr:>7.1f} {p['top1_nongold']:>6.2f} {s:>8.4f}")
        print(" dense vs splade:", aud["dense_vs_splade"])
    json.dump(out, open("results/L2/_squad_regime_audit.json", "w"), indent=2)
    # S10 comparison table
    print("\n\n===== S10  EVIDENCE-REGIME COMPARISON (test) =====")
    hdr = f"{'':16s}" + "".join(f"{d.replace('_clean',''):>11s}" for d in DSS)
    print(hdr)
    for e in ALL:
        row = f"{NAMES[e]+' R@5':16s}" + "".join(f"{out['datasets'][d]['audit']['per_expert'][NAMES[e]]['R@5']:>11.1f}" for d in DSS)
        print(row)
    print(f"{'coop R@5':16s}" + "".join(f"{out['datasets'][d]['audit']['coop_baseline']['R@5']:>11.1f}" for d in DSS))
    print("-- Shapley phi (U_R5) --")
    for e in ALL:
        print(f"{NAMES[e]:16s}" + "".join(f"{out['datasets'][d]['shapley']['R5']['per_expert'][NAMES[e]]['shapley']:>11.4f}" for d in DSS))
    print("-- availability --")
    for e in ALL:
        print(f"{NAMES[e]:16s}" + "".join(f"{out['datasets'][d]['audit']['per_expert'][NAMES[e]]['avail_pct']:>11.2f}" for d in DSS))
    print(f"{'disagree top1':16s}" + "".join(f"{out['datasets'][d]['audit']['dense_vs_splade']['disagree_top1']:>11.2f}" for d in DSS))
    print("\n-> results/L2/_squad_regime_audit.json")


if __name__ == "__main__":
    main()
