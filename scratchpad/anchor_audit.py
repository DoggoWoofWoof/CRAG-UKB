"""STEP 5A - safe-anchor audit + realistic parameter-free oracles on all 5 canonical substrates.
Parameter-free, deployed-convention ranking (missing -> -inf, rank FULL pool).
Answers: does a universal RANK-SAFE anchor (max-rank / RRF) avoid the equal-average dilution
(metaqa dense+splade avg 62.8 < splade 77) while keeping offset/relational strength on webqsp?
Run: python scratchpad/anchor_audit.py
"""
import json, numpy as np, torch

DS = ["webqsp", "metaqa", "2wiki_clean", "musique_clean", "hotpotqa_clean"]
DENSE, OFFSET, SPLADE, RELATION, PATH = 0, 1, 3, 5, 6
KS = [5, 20, 50]
RRF_K = 60


def _scores_missing_neg(col, mcol):
    """Deployed convention: valid candidates keep score, missing -> -inf."""
    return np.where(mcol > 0.5, col, -np.inf)


def _rankscore(col, mcol):
    """Normalized rank-score in [0,1], higher=better, over the FULL pool (missing sink to ~0)."""
    s = _scores_missing_neg(col, mcol)
    order = np.argsort(-s, kind="stable")      # best first
    n = len(s)
    rs = np.empty(n)
    rs[order] = 1.0 - np.arange(n) / n          # rank 0 -> 1.0, last -> ~0
    return rs


def _rrf(col, mcol):
    s = _scores_missing_neg(col, mcol)
    order = np.argsort(-s, kind="stable")
    n = len(s)
    pos = np.empty(n)
    pos[order] = np.arange(n)
    r = 1.0 / (RRF_K + pos)
    r[mcol <= 0.5] = 0.0                         # missing contributes nothing
    return r


def _recall_at(fused, y, ng, K):
    """Canonical CRAG metric (kg_hybrid._qmet): recall = golds in top-K / total gold count ng."""
    top = np.argsort(-fused, kind="stable")[:K]
    return float(y[top].sum()) / max(float(ng), 1.0)


def _hit5(fused, y, K=5):
    top = np.argsort(-fused, kind="stable")[:K]
    return float(y[top].sum() > 0)


def audit():
    out = {}
    for ds in DS:
        d = torch.load(f"results/L2/relsig_feats_{ds}.pt", map_location="cpu", weights_only=False)
        rows = d["test"]
        n = len(rows)
        # per-anchor per-K hit accumulators + per-query hit vectors for oracles
        anchors = ["dense", "splade", "offset", "maxrank_ds", "rrf_ds", "minrank_ds",
                   "maxrank_dso", "rrf_dso", "maxrank_dsOFF_rel"]
        acc = {a: {K: 0.0 for K in KS} for a in anchors}
        # per-query hit@5 for oracle modes
        hq = {k: np.zeros(n) for k in ["dense", "splade", "offset", "maxrank_ds",
                                        "relmode", "safesem"]}
        for i, (x, y, ng, m) in enumerate(rows):
            x = x.numpy(); y = y.numpy().astype(float); m = m.numpy()
            ng = float(ng) if np.ndim(ng) == 0 else float(np.asarray(ng).reshape(-1)[0])
            rs_d = _rankscore(x[:, DENSE], m[:, DENSE])
            rs_s = _rankscore(x[:, SPLADE], m[:, SPLADE])
            rs_o = _rankscore(x[:, OFFSET], m[:, OFFSET])
            rs_r = _rankscore(x[:, RELATION], m[:, RELATION])
            rrf_d = _rrf(x[:, DENSE], m[:, DENSE])
            rrf_s = _rrf(x[:, SPLADE], m[:, SPLADE])
            rrf_o = _rrf(x[:, OFFSET], m[:, OFFSET])
            sc = {
                "dense": _scores_missing_neg(x[:, DENSE], m[:, DENSE]),
                "splade": _scores_missing_neg(x[:, SPLADE], m[:, SPLADE]),
                "offset": _scores_missing_neg(x[:, OFFSET], m[:, OFFSET]),
                "maxrank_ds": np.maximum(rs_d, rs_s),
                "rrf_ds": rrf_d + rrf_s,
                "minrank_ds": np.minimum(rs_d, rs_s),
                "maxrank_dso": np.maximum(np.maximum(rs_d, rs_s), rs_o),
                "rrf_dso": rrf_d + rrf_s + rrf_o,
                "maxrank_dsOFF_rel": np.maximum(np.maximum(rs_d, rs_s), np.maximum(rs_o, rs_r)),
            }
            for a in anchors:
                for K in KS:
                    acc[a][K] += _recall_at(sc[a], y, ng, K)
            # per-query recall@5 for oracle upper bounds (max over modes, per query)
            hq["dense"][i] = _recall_at(sc["dense"], y, ng, 5)
            hq["splade"][i] = _recall_at(sc["splade"], y, ng, 5)
            hq["offset"][i] = _recall_at(sc["offset"], y, ng, 5)
            hq["maxrank_ds"][i] = _recall_at(sc["maxrank_ds"], y, ng, 5)
            hq["safesem"][i] = _recall_at(sc["maxrank_ds"], y, ng, 5)          # safe-semantic mode
            hq["relmode"][i] = _recall_at(np.maximum(rs_o, rs_r), y, ng, 5)    # relational mode (offset+relation)
        res = {a: {f"R@{K}": round(100 * acc[a][K] / n, 2) for K in KS} for a in anchors}
        # realistic parameter-free oracles (upper bounds a 2-mode selector could reach) @5
        oracle = {
            "oracle_dense_splade_offset@5": round(100 * np.maximum.reduce(
                [hq["dense"], hq["splade"], hq["offset"]]).mean(), 2),
            "oracle_splade_vs_offset@5": round(100 * np.maximum(hq["splade"], hq["offset"]).mean(), 2),
            "oracle_safesem_vs_relmode@5": round(100 * np.maximum(hq["safesem"], hq["relmode"]).mean(), 2),
            "oracle_maxrankDS_vs_offset@5": round(100 * np.maximum(hq["maxrank_ds"], hq["offset"]).mean(), 2),
        }
        out[ds] = {"n": n, "anchors": res, "oracles_paramfree": oracle}
        print(f"\n=== {ds} (n={n}) ===")
        for a in anchors:
            print(f"  {a:20s} R@5={res[a]['R@5']:6.2f}  R@20={res[a]['R@20']:6.2f}  R@50={res[a]['R@50']:6.2f}")
        for k, v in oracle.items():
            print(f"  [oracle] {k:34s} {v}")
    json.dump(out, open("results/L2/anchor_audit.json", "w"), indent=2)
    print("\n-> results/L2/anchor_audit.json")


if __name__ == "__main__":
    audit()
