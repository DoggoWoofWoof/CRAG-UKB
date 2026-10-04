"""HARDENED-POOL exact Shapley (source/train only, parameter-free, NO training, NO test, NO tuning-to-outcome).

Recomputes the SAME 32-subset Shapley credit (shapley_audit definitions) on principled source-only harder pools,
to test whether a genuine retrieval-difficulty shift rebalances offset vs splade credit. Pools:

  normal   : full train candidate pool as-is.
  hardneg  : keep ALL golds + only NONGOLD candidates that rank in top-J(=10) of SOME expert (drop the easy tail);
             ranks recomputed within the reduced pool. Increases negative density among discriminative candidates.
  inject   : ADD K(=25) hard-negative distractors = golds of OTHER train queries (docs genuinely high-scoring for a
             DIFFERENT question -> plausible-but-wrong here); ranks recomputed over the augmented pool. This is the
             one transform that can make a strong-looking expert (offset) promote a WRONG candidate.
  conflict : slice = queries whose offset top-1 candidate != splade top-1 candidate (pool itself unchanged).

None of the pools is constructed using gold identity beyond "is/ isn't this query's gold", nor tuned to any test
number. ng (total gold count) is preserved; golds are never dropped.
"""
import json
import numpy as np
import torch

from src.experiments.kg_hybrid import EXPERTS, NE, EXPERT_NAMES, _load, _pooled_musd
from src.experiments import crag_fusion as CF
from src.experiments import crag_gates as G
from scratchpad.shapley_audit import SUBSETS, _sh_w, ALL, NAMES

DEV = torch.device("cpu")
DS = ["webqsp", "metaqa"]
MU = SD = None
J_TOP = 10
K_INJECT = 25
RNG = np.random.RandomState(7)


def rankpct_from_raw(x, mask):
    """Replicate crag_gates.cand_tensors rank_pct on arbitrary raw x (n,7) + mask (n,7). Returns rp (n,NE)."""
    z = (x - MU) / SD
    xr = np.where(mask > 0.5, z, -np.inf)[:, EXPERTS]         # (n,NE)
    valid = np.isfinite(xr)
    n = xr.shape[0]
    rp = np.zeros((n, NE))
    for e in range(NE):
        v = np.where(valid[:, e])[0]
        if len(v) == 0:
            continue
        order = v[np.argsort(-xr[v, e], kind="stable")]
        pos = np.full(n, n); pos[order] = np.arange(len(order))
        rp[:, e] = np.where(valid[:, e], 1.0 - pos / n, 0.0)
    return rp


def _util_all_subsets_np(rp, y, ng):
    n = rp.shape[0]; yb = y > 0.5; res = {}
    for S in SUBSETS:
        if not S:
            res[S] = (0.0, 0.0); continue
        fused = rp[:, list(S)].max(1)
        order = np.argsort(-fused, kind="stable")
        pos = np.empty(n, dtype=np.int64); pos[order] = np.arange(n)
        u_rank = float((1.0 - pos[yb] / n).mean()) if yb.any() else 0.0
        u_r5 = float(yb[order[:5]].sum()) / max(float(ng), 1.0)
        res[S] = (u_rank, u_r5)
    return res


def phi_of(rp, y, ng, ui):
    U = _util_all_subsets_np(rp, y, ng)
    phi = np.zeros(NE)
    for e in ALL:
        for S in SUBSETS:
            if e in S:
                continue
            phi[e] += _sh_w(len(S)) * (U[S | {e}][ui] - U[S][ui])
    return phi


def build_pools(pb, bank):
    x = pb.raw.cpu().numpy(); mask = pb.mask.cpu().numpy(); y = pb.y.cpu().numpy(); ng = pb.ng
    rp0 = rankpct_from_raw(x, mask)
    pools = {"normal": (rp0, y, ng)}
    # hardneg: golds + nongold in top-J of some expert
    top = np.zeros(x.shape[0], dtype=bool)
    for e in range(NE):
        col = rp0[:, e]
        if (col > 0).any():
            idx = np.argsort(-col, kind="stable")[:J_TOP]
            top[idx] = True
    keep = (y > 0.5) | top
    if keep.sum() >= 2:
        pools["hardneg"] = (rankpct_from_raw(x[keep], mask[keep]), y[keep], ng)
    # inject: add K other-query golds as hard negatives
    if len(bank):
        pick = RNG.randint(0, len(bank), size=min(K_INJECT, len(bank)))
        xr = np.concatenate([x] + [bank[i][0][None, :] for i in pick], 0)
        mr = np.concatenate([mask] + [bank[i][1][None, :] for i in pick], 0)
        yr = np.concatenate([y, np.zeros(len(pick))])
        pools["inject"] = (rankpct_from_raw(xr, mr), yr, ng)
    # conflict flag: offset top1 != splade top1
    o = rp0[:, G.OFFSET]; s = rp0[:, G.SPLADE]
    conflict = (o.max() > 0 and s.max() > 0 and int(np.argmax(o)) != int(np.argmax(s)))
    return pools, conflict


def main():
    global MU, SD
    data = _load(DS, DEV)
    mu, sd = _pooled_musd(data, DS, DEV, "train")
    MU = mu.cpu().numpy(); SD = sd.cpu().numpy()
    out = {}
    for ds in DS:
        pbs = [CF.prepbatch(r, mu, sd, DEV, ds) for r in data[ds]["train"]]
        bank = []
        for pb in pbs:
            x = pb.raw.cpu().numpy(); mask = pb.mask.cpu().numpy(); yb = pb.y.cpu().numpy() > 0.5
            for i in np.where(yb)[0]:
                bank.append((x[i], mask[i]))
        acc = {p: {"rank": [], "R5": []} for p in ("normal", "hardneg", "inject")}
        conf_acc = {"rank": [], "R5": []}
        for pb in pbs:
            if pb.y.sum() == 0:
                continue
            pools, conflict = build_pools(pb, bank)
            for p, (rp, y, ng) in pools.items():
                if p == "conflict":
                    continue
                if p in acc:
                    acc[p]["rank"].append(phi_of(rp, y, ng, 0))
                    acc[p]["R5"].append(phi_of(rp, y, ng, 1))
            if conflict:
                rp, y, ng = pools["normal"]
                conf_acc["rank"].append(phi_of(rp, y, ng, 0))
                conf_acc["R5"].append(phi_of(rp, y, ng, 1))
        res = {}
        for u, ui in (("rank", 0), ("R5", 1)):
            cols = {}
            for p in ("normal", "hardneg", "inject"):
                phi = np.mean(acc[p][u], 0)
                cols[p] = {NAMES[e]: round(float(phi[e]), 4) for e in ALL}
                cols[p]["offset/splade"] = round(float(phi[G.OFFSET] / (phi[G.SPLADE] + 1e-9)), 3)
            phi = np.mean(conf_acc[u], 0)
            cols["conflict"] = {NAMES[e]: round(float(phi[e]), 4) for e in ALL}
            cols["conflict"]["offset/splade"] = round(float(phi[G.OFFSET] / (phi[G.SPLADE] + 1e-9)), 3)
            cols["conflict"]["nq"] = len(conf_acc[u])
            res[u] = cols
        out[ds] = res
    json.dump(out, open("results/L2/_hardened_shapley.json", "w"), indent=2)
    for ds in DS:
        for u in ("rank", "R5"):
            print(f"\n===== {ds}  U_{u} =====")
            pools = ["normal", "hardneg", "inject", "conflict"]
            print(f"{'expert':14s}" + "".join(f"{p:>12s}" for p in pools))
            for e in ALL:
                print(f"{NAMES[e]:14s}" + "".join(f"{out[ds][u][p][NAMES[e]]:>12.4f}" for p in pools))
            print(f"{'offset/splade':14s}" + "".join(f"{out[ds][u][p]['offset/splade']:>12.3f}" for p in pools))
            print(f"(conflict nq={out[ds][u]['conflict']['nq']})")


if __name__ == "__main__":
    main()
