"""Analyses 1-6 + regime discovery + controller-readiness gates over the exact Shapley npz."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
OUTD = "results/L2/_shapley"
EXPERTS = ("dense", "splade", "offset", "mixture", "relation")
UTILS = ("ndcg50", "mrr", "all10", "all50")
PILOTS = ("2wiki_clean", "musique_clean")


def load(ds, split):
    return np.load(f"{OUTD}/{ds}_{split}.npz")


def _phi_mat(z, u):
    """(nq,5) phi for utility u, restricted to valid queries -> returns (mat_valid, valid_idx)."""
    v = z["valid"]
    mat = np.vstack([z[f"phi_{u}_{e}"] for e in EXPERTS]).T   # (nq,5)
    return mat[v], np.where(v)[0], v


def analysis1_global(z):
    out = {}
    for u in UTILS:
        mat, _, _ = _phi_mat(z, u)
        st = {}
        for i, e in enumerate(EXPERTS):
            x = mat[:, i]
            st[e] = {"mean": round(float(x.mean()), 5), "median": round(float(np.median(x)), 5),
                     "std": round(float(x.std()), 5),
                     "p10": round(float(np.percentile(x, 10)), 5), "p25": round(float(np.percentile(x, 25)), 5),
                     "p75": round(float(np.percentile(x, 75)), 5), "p90": round(float(np.percentile(x, 90)), 5),
                     "frac_pos": round(float((x > 1e-6).mean()), 4), "frac_zero": round(float((np.abs(x) <= 1e-6).mean()), 4),
                     "frac_neg": round(float((x < -1e-6).mean()), 4)}
        out[u] = st
    return out


def analysis2_heterogeneity(z, u="ndcg50"):
    mat, _, _ = _phi_mat(z, u)
    nq = len(mat)
    top = mat.argmax(1)
    # second largest
    order = np.argsort(-mat, 1)
    second = order[:, 1]
    largest_counts = {e: int((top == i).sum()) for i, e in enumerate(EXPERTS)}
    second_counts = {e: int((second == i).sum()) for i, e in enumerate(EXPERTS)}
    neg_counts = {e: int((mat[:, i] < -1e-6).sum()) for i, e in enumerate(EXPERTS)}
    nearzero_counts = {e: int((np.abs(mat[:, i]) <= 1e-6).sum()) for i, e in enumerate(EXPERTS)}
    # positive-mass concentration (entropy over positive phi, normalized)
    pos = np.clip(mat, 0, None)
    psum = pos.sum(1, keepdims=True)
    ent = np.zeros(nq)
    hasp = psum[:, 0] > 1e-9
    p = pos[hasp] / psum[hasp]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent[hasp] = -(np.where(p > 0, p * np.log(p), 0)).sum(1) / np.log(5)   # normalized [0,1]
    # how many experts needed to reach 90% of positive mass
    sp = np.sort(pos, 1)[:, ::-1]; cum = np.cumsum(sp, 1) / np.clip(sp.sum(1, keepdims=True), 1e-9, None)
    k90 = (cum < 0.9).sum(1) + 1
    return {"n": int(nq),
            "largest_positive_contributor_counts": largest_counts,
            "second_contributor_counts": second_counts,
            "negative_contributor_counts": neg_counts,
            "near_zero_counts": nearzero_counts,
            "frac_single_expert_dominant(>=0.9 mass in 1)": round(float((k90 == 1).mean()), 4),
            "frac_need_2plus_experts(90% mass)": round(float((k90 >= 2).mean()), 4),
            "frac_need_3plus_experts(90% mass)": round(float((k90 >= 3).mean()), 4),
            "positive_mass_entropy_mean": round(float(ent[hasp].mean()), 4) if hasp.any() else None,
            "dominant_expert_distribution_note": "spread of largest-contributor across experts => query-dependent"}


def analysis3_relation(z):
    out = {}
    for u in ("ndcg50", "all50", "mrr"):
        pr = z[f"phi_{u}_relation"]; v = z["valid"]
        anysig = z["relation_any_signal_present"].astype(bool)
        goldp = z["relation_gold_present"].astype(bool)
        def stat(mask):
            m = v & mask
            x = pr[m]
            return {"n": int(m.sum()), "mean": round(float(x.mean()), 5) if m.sum() else None,
                    "frac_pos": round(float((x > 1e-6).mean()), 4) if m.sum() else None,
                    "maxabs": round(float(np.abs(x).max()), 6) if m.sum() else None}
        out[u] = {"signal_TRUE": stat(anysig), "signal_FALSE": stat(~anysig),
                  "gold_present_TRUE": stat(goldp), "gold_present_FALSE": stat(~goldp)}
    # sanity: no-signal -> phi_relation exactly 0
    v = z["valid"]; nosig = v & ~z["relation_any_signal_present"].astype(bool)
    out["SANITY_no_signal_relation_phi_maxabs"] = float(np.abs(np.vstack([z[f"phi_{u}_relation"] for u in UTILS])[:, nosig]).max()) if nosig.any() else 0.0
    return out


def analysis4_offset_mixture(z):
    v = z["valid"]; out = {}
    for e in ("offset", "mixture"):
        mrr = z[f"phi_mrr_{e}"][v]; ndcg = z[f"phi_ndcg50_{e}"][v]; all50 = z[f"phi_all50_{e}"][v]
        # queries where MRR contribution <=0 but deep-recovery contribution >0
        tradeoff = (mrr <= 1e-6) & (ndcg > 1e-6)
        tradeoff50 = (mrr <= 1e-6) & (all50 > 1e-6)
        out[e] = {"mean_phi_mrr": round(float(mrr.mean()), 5), "mean_phi_ndcg50": round(float(ndcg.mean()), 5),
                  "mean_phi_all50": round(float(all50.mean()), 5),
                  "frac_MRR<=0_but_nDCG50>0": round(float(tradeoff.mean()), 4),
                  "frac_MRR<=0_but_ALL50>0": round(float(tradeoff50.mean()), 4),
                  "frac_phi_mrr_negative": round(float((mrr < -1e-6).mean()), 4)}
    return out


def analysis5_disagreement(z, qtiles):
    v = z["valid"]; dsg = z["dense_splade_disagreement"]
    b = np.digitize(dsg, qtiles)                          # 0 low,1 med,2 high (train quantiles)
    out = {}
    for u in ("ndcg50",):
        per = {}
        for bi, name in ((0, "low"), (1, "med"), (2, "high")):
            m = v & (b == bi)
            per[name] = {"n": int(m.sum()),
                         **{e: round(float(z[f"phi_{u}_{e}"][m].mean()), 5) if m.sum() else None for e in EXPERTS}}
        out[u] = per
    return out


def analysis6_complementarity(z, u="ndcg50"):
    v = z["valid"]; mat = np.vstack([z[f"phi_{u}_{e}"][v] for e in EXPERTS])
    C = np.corrcoef(mat)
    pairs = {}
    for a in range(5):
        for b in range(a + 1, 5):
            pairs[f"{EXPERTS[a]}~{EXPERTS[b]}"] = round(float(C[a, b]), 3)
    return pairs


def hardness(z, u="ndcg50"):
    v = z["valid"]; vf = z[f"v_full_{u}"][v]
    mat = np.vstack([z[f"phi_{u}_{e}"][v] for e in EXPERTS]).T
    absmass = np.abs(mat).sum(1)
    lo_phi = absmass <= np.percentile(absmass, 25)
    hi_vf = vf >= np.percentile(vf, 75); lo_vf = vf <= np.percentile(vf, 25)
    return {"v_full_mean": round(float(vf.mean()), 4),
            "frac_lowphi_highvfull(easy/redundant)": round(float((lo_phi & hi_vf).mean()), 4),
            "frac_lowphi_lowvfull(genuinely_hard)": round(float((lo_phi & lo_vf).mean()), 4)}


def regime_discovery(ds, u="ndcg50", k=6):
    """Cluster TRAIN phi vectors (primary utility). Report centroids + sizes; label only if supported."""
    z = load(ds, "train"); v = z["valid"]
    mat = np.vstack([z[f"phi_{u}_{e}"][v] for e in EXPERTS]).T
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(mat)
        lab = km.labels_; cent = km.cluster_centers_
    except Exception as ex:
        return {"error": str(ex)}
    clusters = []
    for c in range(k):
        m = lab == c
        cen = {EXPERTS[i]: round(float(cent[c, i]), 4) for i in range(5)}
        dom = EXPERTS[int(np.argmax(cent[c]))]
        clusters.append({"id": c, "frac": round(float(m.mean()), 4), "n": int(m.sum()),
                         "centroid": cen, "dominant_expert": dom})
    clusters.sort(key=lambda x: -x["frac"])
    return {"utility": u, "k": k, "clusters": clusters}


def run_all():
    rep = {}
    # train-derived disagreement quantiles per dataset (do NOT use test)
    for ds in PILOTS:
        ztr = load(ds, "train")
        qt = np.quantile(ztr["dense_splade_disagreement"][ztr["valid"]], [0.3333, 0.6666])
        rep[ds] = {}
        for sp in ("train", "val"):
            z = load(ds, sp)
            rep[ds][sp] = {
                "n_valid": int(z["valid"].sum()), "n_total": int(len(z["valid"])),
                "n_L1_fail_or_novalid": int((~z["valid"]).sum()),
                "A1_global": analysis1_global(z),
                "A2_heterogeneity": analysis2_heterogeneity(z),
                "A3_relation": analysis3_relation(z),
                "A4_offset_mixture": analysis4_offset_mixture(z),
                "A5_disagreement": analysis5_disagreement(z, qt),
                "A6_complementarity": analysis6_complementarity(z),
                "hardness": hardness(z),
            }
        rep[ds]["regime_train"] = regime_discovery(ds)
    json.dump(rep, open("scratchpad/_shapley_analysis.json", "w"), indent=1, default=str)
    return rep


if __name__ == "__main__":
    r = run_all()
    print("analysis written scratchpad/_shapley_analysis.json")
