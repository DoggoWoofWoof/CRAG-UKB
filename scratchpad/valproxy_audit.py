"""Phase-1 VALIDATION-PROXY AUDIT (diagnosis only, NO tuning, NO test-driven construction).

Question: WHY is the random train-derived validation split saturated (MetaQA val ~96 vs test ~74)?
Compares the TRAIN-only validation slice (split_seed=1234 val indices) against the canonical TEST slice
on structural substrate statistics -- pool size, gold coverage, gold-rank, hard-negative presence,
single-expert-trivial fraction, expert disagreement. Metrics are for DIAGNOSIS; nothing here is tuned to
any test value. Everything is computed on the FROZEN canonical substrate.
"""
import json
import numpy as np
import torch

from src.experiments.kg_hybrid import EXPERTS, NE, EXPERT_NAMES, _load, _pooled_musd
from src.experiments import crag_fusion as CF
from src.experiments import crag_gates as G

DEV = torch.device("cpu")
DS = ["webqsp", "metaqa"]
K = 5


def per_query_stats(pb):
    n = pb.x_rank.shape[0]
    y = pb.y.cpu().numpy() > 0.5
    ng = pb.ng
    xr = pb.x_rank[:, EXPERTS].cpu().numpy()          # (n,NE) -inf missing
    valid = np.isfinite(xr)
    # per-expert rank position (0=best); missing -> large
    pos = np.full((n, NE), n, dtype=np.int64)
    for e in range(NE):
        v = np.where(valid[:, e])[0]
        if len(v):
            order = v[np.argsort(-xr[v, e], kind="stable")]
            pos[order, e] = np.arange(len(order))
    golds_in_pool = int(y.sum())
    best_gold_pos = None
    if golds_in_pool:
        best_gold_pos = float(np.mean(pos[y].min(1)))         # mean best-across-experts rank of gold
    # hard negatives: nongold candidate in top-5 of SOME expert
    top5_any = (pos < 5).any(1)
    hard_negs = int((top5_any & (~y)).sum())
    # single-expert-trivial: some expert's top-5 contains ALL in-pool golds
    trivial = False
    if golds_in_pool:
        for e in range(NE):
            top5e = set(np.where(pos[:, e] < 5)[0].tolist())
            if set(np.where(y)[0].tolist()).issubset(top5e):
                trivial = True; break
    # expert disagreement: #distinct top-1 candidates across available experts
    top1 = [int(np.argmin(pos[:, e])) for e in range(NE) if valid[:, e].any()]
    distinct_top1 = len(set(top1))
    # conflict: does offset's top-1 differ from splade's top-1 (the metaqa failure axis)?
    off_top = int(np.argmin(pos[:, G.OFFSET])) if valid[:, G.OFFSET].any() else -1
    spl_top = int(np.argmin(pos[:, G.SPLADE])) if valid[:, G.SPLADE].any() else -2
    offset_vs_splade_conflict = int(off_top != spl_top)
    # offset promotes a NONGOLD #1 while splade #1 IS gold (the exact I failure)
    offset_wrong_splade_right = int(off_top >= 0 and not y[off_top] and spl_top >= 0 and y[spl_top])
    return dict(n=n, ng=ng, golds_in_pool=golds_in_pool,
                gold_cov=(golds_in_pool / ng if ng else 0.0),
                best_gold_pos=best_gold_pos, hard_negs=hard_negs,
                trivial=int(trivial), distinct_top1=distinct_top1,
                offset_vs_splade_conflict=offset_vs_splade_conflict,
                offset_wrong_splade_right=offset_wrong_splade_right)


def agg(pbs):
    S = [per_query_stats(pb) for pb in pbs]
    def m(key):
        vals = [s[key] for s in S if s[key] is not None]
        return round(float(np.mean(vals)), 3) if vals else None
    o = G.anchors_and_oracles(pbs, list({pb.ds for pb in pbs}), klmlp=None)
    return dict(nq=len(S), pool_n=m("n"), golds_per_q=m("golds_in_pool"),
                gold_cov=m("gold_cov"), mean_best_gold_rank=m("best_gold_pos"),
                hard_negs_per_q=m("hard_negs"), frac_trivial=m("trivial"),
                distinct_top1=m("distinct_top1"),
                frac_offset_splade_conflict=m("offset_vs_splade_conflict"),
                frac_offset_wrong_splade_right=m("offset_wrong_splade_right"))


def main():
    data = _load(DS, DEV)
    mu, sd = _pooled_musd(data, DS, DEV, "train")
    splits = G.make_splits(data, DS, 0.15, 1234)
    out = {}
    for ds in DS:
        val_pbs = [CF.prepbatch(data[ds]["train"][i], mu, sd, DEV, ds) for i in splits[ds]["val"]]
        test_pbs = [CF.prepbatch(r, mu, sd, DEV, ds) for r in data[ds]["test"]]
        va = agg(val_pbs); te = agg(test_pbs)
        oa_v = G.anchors_and_oracles(val_pbs, [ds], klmlp=None)[ds]
        oa_t = G.anchors_and_oracles(test_pbs, [ds], klmlp=None)[ds]
        out[ds] = {"train_val": {**va, "dense_R5": oa_v["dense"], "splade_R5": oa_v["splade"],
                                 "offset_R5": oa_v["offset"], "coop_R5": oa_v["coop_maxrank"],
                                 "pool_oracle_R5": oa_v["oracle_dense_splade_offset"]},
                   "test":      {**te, "dense_R5": oa_t["dense"], "splade_R5": oa_t["splade"],
                                 "offset_R5": oa_t["offset"], "coop_R5": oa_t["coop_maxrank"],
                                 "pool_oracle_R5": oa_t["oracle_dense_splade_offset"]}}
    json.dump(out, open("results/L2/_valproxy_audit_phase1.json", "w"), indent=2)
    for ds in DS:
        print(f"\n===== {ds} =====")
        print(f"{'stat':32s} {'TRAIN-VAL':>12s} {'TEST':>12s}")
        for k in out[ds]["train_val"]:
            v = out[ds]["train_val"][k]; t = out[ds]["test"][k]
            print(f"{k:32s} {str(v):>12s} {str(t):>12s}")


if __name__ == "__main__":
    main()
