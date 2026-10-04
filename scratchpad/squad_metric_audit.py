"""Close the S11 validation-metric question: rank-sensitive SQuAD metrics over the frozen A/H/I trajectory.

Checkpoints from the Modal S11 run are gone (in-process list). Training is DETERMINISTIC (seed 0, frozen confirm
config) so we regenerate ONCE, LOCALLY (no Modal spend), PERSIST checkpoints to disk, and score each on SQuAD with
R@1/R@5/MRR/NDCG@5/NDCG@10 (R@5 = known saturated control) + source webqsp/metaqa TEST R@5 + gate balance. Then
Spearman+Pearson(metric, source-test) retrospectively ONCE. No test-tuning; metrics fixed a-priori.
"""
import json, math, numpy as np, torch
from scipy.stats import spearmanr, pearsonr
from src.experiments.kg_hybrid import _load, _pooled_musd, EXPERTS, EXPERT_NAMES
from src.experiments import crag_fusion as CF
from src.experiments import crag_gates as G

DEV = torch.device("cpu")
SRC = ["webqsp", "metaqa"]
SQ = "squad_clean"


def _rich_agg(recs):
    """recs = {ds: [(ry_top50_goldmask, ng)]} -> {ds: R@1/R@5/MRR/NDCG@5/NDCG@10}."""
    out = {}
    for ds, rs in recs.items():
        if not rs:
            continue
        r1 = r5 = mrr = n5 = n10 = 0.0
        for ry, ng in rs:
            ry = np.asarray(ry, float)
            r1 += ry[:1].sum() / ng
            r5 += ry[:5].sum() / ng
            nz = np.nonzero(ry)[0]
            mrr += 1.0 / (nz[0] + 1) if len(nz) else 0.0
            for k, acc in ((5, "n5"), (10, "n10")):
                dcg = sum(ry[i] / math.log2(i + 2) for i in range(min(k, len(ry))))
                idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, int(ng))))
                v = dcg / idcg if idcg > 0 else 0.0
                if k == 5: n5 += v
                else: n10 += v
        m = len(rs)
        out[ds] = {"R@1": round(100 * r1 / m, 2), "R@5": round(100 * r5 / m, 2), "MRR": round(mrr / m, 4),
                   "NDCG@5": round(n5 / m, 4), "NDCG@10": round(n10 / m, 4)}
    return out


def _rich(arm, state, vset, ds):
    if arm == "A":
        qenc = G.QEnc(CF.UIN, G.HQ).to(DEV); cand = G.Cand(CF.CAND_IN).to(DEV)
        qenc.load_state_dict(state["qenc"]); cand.load_state_dict(state["cand"])
        recs = CF.eval_crag(qenc, cand, vset, [ds])
    else:
        model = G.GateModel(G.ARMS[arm]["use_int"]).to(DEV)
        model.load_state_dict(state["model"])
        recs = G.eval_gate(model, vset, [ds], arm)
    return _rich_agg(recs).get(ds, {})


def _gates(arm, state, vset, ds):
    if arm == "A":
        return {"offset": float("nan"), "splade": float("nan")}
    model = G.GateModel(G.ARMS[arm]["use_int"]).to(DEV)
    model.load_state_dict(state["model"])
    mg = G.mean_gates(model, vset, [ds])
    return {"offset": mg[ds]["offset"] if ds in mg else float("nan"),
            "splade": mg[ds]["splade"] if ds in mg else float("nan")}


def _corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return {"spearman": None, "pearson": None}
    return {"spearman": round(float(spearmanr(a[ok], b[ok]).correlation), 3),
            "pearson": round(float(pearsonr(a[ok], b[ok])[0]), 3)}


def main(max_steps=16000, val_every=500, arms=("I", "H", "A"), split_seed=1234, frac_val=0.15, p_drop=0.2,
         eval_cap=800):
    data = _load(SRC + [SQ], DEV)
    mu, sd = _pooled_musd(data, SRC, DEV, "train")
    splits = G.make_splits(data, SRC, frac_val, split_seed)
    TR_opt = G._prep_from_idx(data, SRC, splits, "opt", mu, sd)
    TE_val = G._prep_from_idx(data, SRC, splits, "val", mu, sd)
    # fixed a-priori eval caps (deterministic head) keep local scoring tractable; degradation signal preserved
    TE_test = [CF.prepbatch(r, mu, sd, DEV, ds) for ds in SRC for r in data[ds]["test"][:eval_cap]]
    refs = G._val_refs(TE_val, SRC)
    sq_test = [CF.prepbatch(r, mu, sd, DEV, SQ) for r in data[SQ]["test"][:eval_cap]]   # held-out domain, source mu/sd
    print(f"sizes: opt={len(TR_opt)} val={len(TE_val)} src_test={len(TE_test)} squad_test={len(sq_test)} (cap={eval_cap})")

    out = {"_config": {"kind": "squad_metric_audit_local", "regenerated": "deterministic seed0 frozen confirm config",
                       "squad_split": "test", "max_steps": max_steps, "note": "metrics fixed a-priori; test used once for corr"},
           "arms": {}}
    ckpt_store = {}
    for arm in arms:
        ck = []
        test_final, curve, meta = G._run_confirm_arm(0, arm, TR_opt, TE_val, TE_test, SRC, refs,
                                                     max_steps, val_every, p_drop, ckpt_sink=ck)
        ckpt_store[arm] = [{"step": c["step"], "state": c["state"]} for c in ck]
        steps = [c["step"] for c in ck]
        sq = {m: [] for m in ("R@1", "R@5", "MRR", "NDCG@5", "NDCG@10")}
        srct = {ds: [] for ds in SRC}
        gates_sq = {"offset": [], "splade": []}
        for c in ck:
            rm = _rich(arm, c["state"], sq_test, SQ)
            for m in sq:
                sq[m].append(rm.get(m, float("nan")))
            rt = _rich(arm, c["state"], TE_test, None) if False else None
            # source test R@5 per ds
            if arm == "A":
                qenc = G.QEnc(CF.UIN, G.HQ).to(DEV); cand = G.Cand(CF.CAND_IN).to(DEV)
                qenc.load_state_dict(c["state"]["qenc"]); cand.load_state_dict(c["state"]["cand"])
                rr = G._agg(CF.eval_crag(qenc, cand, TE_test, SRC))
            else:
                model = G.GateModel(G.ARMS[arm]["use_int"]).to(DEV); model.load_state_dict(c["state"]["model"])
                rr = G._agg(G.eval_gate(model, TE_test, SRC, arm))
            for ds in SRC:
                srct[ds].append(rr[ds]["R@5"] if ds in rr else float("nan"))
            g = _gates(arm, c["state"], sq_test, SQ)
            gates_sq["offset"].append(g["offset"]); gates_sq["splade"].append(g["splade"])
        # correlations: each SQuAD metric vs source-test degradation
        corr = {m: {ds: _corr(sq[m], srct[ds]) for ds in SRC} for m in sq}
        rng = {m: round(float(np.nanmax(sq[m]) - np.nanmin(sq[m])), 3) for m in sq}
        out["arms"][arm] = {"steps": steps, "best_step": meta["best_step"], "test_final": test_final,
                            "squad": sq, "squad_range": rng, "source_test": srct, "squad_gates": gates_sq,
                            "corr_metric_vs_sourcetest": corr}
        print(f"\n=== {arm} best_step={meta['best_step']} squad ranges {rng}")
        for ds in SRC:
            print(f"   corr vs {ds}_test:", {m: corr[m][ds] for m in sq})
    torch.save(ckpt_store, "results/L2/_squad_metric_ckpts.pt")
    json.dump(out, open("results/L2/_squad_metric_audit.json", "w"), indent=2)
    print("\n-> results/L2/_squad_metric_audit.json (+ _squad_metric_ckpts.pt)")


if __name__ == "__main__":
    import sys
    ms = int(sys.argv[1]) if len(sys.argv) > 1 else 16000
    ar = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("I", "H", "A")
    main(max_steps=ms, arms=ar)
