"""Q1-Q5 query-regime profiling (TRAIN-ONLY, parameter-free Shapley, NO training, NO test).

Q1 per-query exact Shapley phi (U_R5 primary, U_rank retained) over available experts.
Q2 TRAIN-ONLY per-expert quantiles (Q25/Q50/Q75) -> regime thresholds (expert-symmetric; no fixed 0.7).
Q3 interpretable regime taxonomy w/ documented precedence + an unsupervised KMeans cross-check on normalized phi.
Q4 dataset x regime distribution table + totals.
Q5 dataset x regime sampler cells: report counts + proposed balanced weights (do NOT train here).

Shapley uses gold labels -> TRAIN-TIME profiling ONLY, never required at inference.
"""
import json, itertools, numpy as np, torch
from src.experiments.kg_hybrid import NE, EXPERT_NAMES, EXPERTS, _load, _pooled_musd
from src.experiments import crag_fusion as CF
from src.experiments import crag_gates as G
from scratchpad import shapley_audit as SH

DEV = torch.device("cpu")
DSS = ["webqsp", "metaqa", "squad_clean"]
NAMES = [EXPERT_NAMES[e] for e in range(NE)]              # dense,offset,splade,relation,path
DENSE, OFFSET, SPLADE, RELATION, PATH = 0, 1, 2, 3, 4
SEM = [DENSE, SPLADE]; REL = [OFFSET, RELATION, PATH]
REGIMES = ["hard", "missing_structure", "conflict", "single_rescue", "redundant", "mixed_coop", "semantic", "relational"]


def per_query(pbs):
    """Return list of dicts: phi(U_R5), interactions, availability, top1-disagreement per TRAIN query."""
    recs = []
    for pb in pbs:
        sh = SH.query_shapley(pb)
        if sh is None:
            continue
        phi = sh["R5"]["phi"]                            # (NE,)
        inter = sh["R5"]["inter"]                        # {(e,f): val}
        T = G.cand_tensors(pb)
        avail = (T["valid"].sum(0) > 0).cpu().numpy().astype(bool)   # per-expert present in pool
        rp = T["rank_pct"].cpu().numpy()
        top1 = [int(np.argmax(rp[:, e])) if avail[e] and rp[:, e].max() > 0 else -1 for e in range(NE)]
        recs.append({"phi": phi, "inter": inter, "avail": avail, "top1": top1})
    return recs


def label(rec, thr, neg_int):
    """Assign ONE regime via documented precedence. thr=(NE,) train-Q75 per expert; neg_int negative-interaction cutoff."""
    phi = rec["phi"]; av = rec["avail"]; inter = rec["inter"]; top1 = rec["top1"]
    useful = np.array([av[e] and phi[e] >= thr[e] for e in range(NE)])
    nu = int(useful.sum())
    sem_use = any(useful[e] for e in SEM)
    rel_use = any(useful[e] for e in REL)
    # 1 HARD: nobody reaches their own top-quartile credit
    if nu == 0:
        return "hard"
    # 2 MISSING-STRUCTURE: relation AND path unavailable while semantic carries evidence
    if (not av[RELATION]) and (not av[PATH]) and sem_use and not any(useful[e] for e in (OFFSET,)):
        return "missing_structure"
    # 3 CONFLICT: >=2 present experts point their #1 at different candidates AND no single dominant expert
    present_top1 = [top1[e] for e in range(NE) if top1[e] >= 0]
    disagree = len(set(present_top1)) >= 2
    dominant = phi.max() >= 2.0 * (np.sort(phi)[-2] if (phi > 0).sum() >= 2 else 0.0) and phi.max() > 0
    if disagree and nu >= 2 and not dominant:
        # only CONFLICT if the useful experts actually disagree on #1
        u_top1 = set(top1[e] for e in range(NE) if useful[e] and top1[e] >= 0)
        if len(u_top1) >= 2:
            return "conflict"
    # 4 SINGLE-EXPERT RESCUE
    if nu == 1:
        return "single_rescue"
    # 5 REDUNDANT: >=2 useful experts with a negative (substitutable) interaction among them
    uidx = [e for e in range(NE) if useful[e]]
    for e, f in itertools.combinations(uidx, 2):
        if inter.get((e, f), inter.get((f, e), 0.0)) <= neg_int:
            return "redundant"
    # 6 MIXED-COOPERATIVE
    if sem_use and rel_use:
        return "mixed_coop"
    # 7 SEMANTIC / 8 RELATIONAL
    if sem_use:
        return "semantic"
    return "relational"


def main():
    data = _load(DSS, DEV)
    profiles = {}
    for ds in DSS:
        mu, sd = _pooled_musd(data, [ds], DEV, "train")
        pbs = [CF.prepbatch(r, mu, sd, DEV, ds) for r in data[ds]["train"]]
        profiles[ds] = per_query(pbs)
        print(f"{ds}: {len(profiles[ds])} train queries profiled")

    # Q2 TRAIN-ONLY per-expert quantiles pooled across all training domains (source-only)
    allphi = np.array([r["phi"] for ds in DSS for r in profiles[ds]])       # (Nq, NE)
    q25 = np.percentile(allphi, 25, 0); q50 = np.percentile(allphi, 50, 0); q75 = np.percentile(allphi, 75, 0)
    thr = q75.copy()
    # negative-interaction cutoff = pooled 25th percentile of all interaction values (train-only)
    allint = np.array([v for ds in DSS for r in profiles[ds] for v in r["inter"].values()])
    neg_int = float(np.percentile(allint, 25))
    print("\nTRAIN quantiles per expert (phi U_R5):")
    for e in range(NE):
        print(f"  {NAMES[e]:9s} Q25={q25[e]:.3f} Q50={q50[e]:.3f} Q75={q75[e]:.3f}")
    print(f"  neg-interaction cutoff (Q25 of all I(e,f)) = {neg_int:.4f}")

    # Q3+Q4 labels + distribution
    dist = {ds: {r: 0 for r in REGIMES} for ds in DSS}
    labels = {ds: [] for ds in DSS}
    for ds in DSS:
        for rec in profiles[ds]:
            lb = label(rec, thr, neg_int)
            dist[ds][lb] += 1; labels[ds].append(lb)

    # KMeans cross-check on normalized phi (standardized train-only)
    km_info = {}
    try:
        from sklearn.cluster import KMeans
        mu_p = allphi.mean(0); sd_p = allphi.std(0) + 1e-9
        Z = (allphi - mu_p) / sd_p
        km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(Z)
        # cluster centroids in phi space (de-normalized) for interpretability
        cent = km.cluster_centers_ * sd_p + mu_p
        km_info = {"n_clusters": 6, "cluster_sizes": np.bincount(km.labels_).tolist(),
                   "centroids_phi": {f"c{i}": {NAMES[e]: round(float(cent[i, e]), 3) for e in range(NE)} for i in range(6)}}
    except Exception as ex:
        km_info = {"error": repr(ex)}

    # Q5 sampler cells: dataset x regime counts + proposed balanced weights
    cells = {(ds, r): dist[ds][r] for ds in DSS for r in REGIMES if dist[ds][r] > 0}
    n_cells = len(cells)
    OVERSAMPLE_CAP = 5.0        # a-priori cap: no cell weighted >5x its natural share
    total = sum(cells.values())
    # target = uniform over non-empty cells, but cap oversample factor vs natural
    tgt = {}
    for (ds, r), n in cells.items():
        nat = n / total
        uni = 1.0 / n_cells
        eff = min(uni, nat * OVERSAMPLE_CAP)             # cap oversampling of tiny cells
        tgt[(ds, r)] = eff
    zs = sum(tgt.values())
    tgt = {k: v / zs for k, v in tgt.items()}

    out = {"_config": {"quantiles": {NAMES[e]: {"Q25": round(float(q25[e]), 4), "Q50": round(float(q50[e]), 4),
                                                 "Q75": round(float(q75[e]), 4)} for e in range(NE)},
                       "neg_int_cutoff": round(neg_int, 4), "oversample_cap": OVERSAMPLE_CAP,
                       "precedence": REGIMES, "note": "TRAIN-only; Shapley train-time profiling; no test"},
           "distribution": dist,
           "totals_by_regime": {r: sum(dist[ds][r] for ds in DSS) for r in REGIMES},
           "totals_by_dataset": {ds: sum(dist[ds].values()) for ds in DSS},
           "kmeans_crosscheck": km_info,
           "sampler_cells": [{"dataset": ds, "regime": r, "n": n, "natural_frac": round(n / total, 4),
                              "target_frac": round(tgt[(ds, r)], 4),
                              "oversample_x": round(tgt[(ds, r)] / (n / total), 2)}
                             for (ds, r), n in sorted(cells.items(), key=lambda kv: -kv[1])]}
    json.dump(out, open("results/L2/_query_regime.json", "w"), indent=2)

    # console table
    print("\n===== Q4  dataset x regime distribution =====")
    hdr = f"{'regime':18s}" + "".join(f"{ds.replace('_clean',''):>9s}" for ds in DSS) + f"{'TOTAL':>8s}"
    print(hdr)
    for r in REGIMES:
        row = f"{r:18s}" + "".join(f"{dist[ds][r]:>9d}" for ds in DSS) + f"{out['totals_by_regime'][r]:>8d}"
        print(row)
    print(f"{'TOTAL':18s}" + "".join(f"{out['totals_by_dataset'][ds]:>9d}" for ds in DSS) + f"{total:>8d}")
    print("\nKMeans(6) cluster sizes:", km_info.get("cluster_sizes"))
    print("\n===== Q5 sampler cells (top by count) =====")
    for c in out["sampler_cells"][:20]:
        print(f"  {c['dataset']:12s} {c['regime']:16s} n={c['n']:5d} nat={c['natural_frac']:.3f} -> tgt={c['target_frac']:.3f} ({c['oversample_x']}x)")
    print("\n-> results/L2/_query_regime.json")


if __name__ == "__main__":
    main()
