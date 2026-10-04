"""P2B — EXACT SHAPLEY contribution audit (source/train, parameter-free, NO training, NO test, NO tuning).

Fixes the leave-one-out under-crediting of redundant experts. 5 experts -> 2^5=32 subsets -> exact Shapley.

  U(S)      = utility of expert-set S's parameter-free max-rank fusion (two frozen defs: U_rank, U_R5). U({})=0.
  phi_e     = sum_{S subN\e} [|S|!(n-1-|S|)!/n!] (U(S+e)-U(S))            fair credit under redundancy.
  I(e,f)    = sum_{S subN\{e,f}} [|S|!(n-2-|S|)!/(n-1)!] * delta_ef(S),   Shapley interaction index
              delta_ef(S) = U(S+e+f) - U(S+e) - U(S+f) + U(S).
              I(e,f) > 0  TRUE SYNERGY (value only together);  I(e,f) < 0  REDUNDANCY (either alone suffices).
  cond_marg pos/neg frac = fraction of conditioning subsets S where (U(S+e)-U(S)) is >0 / <0 (magnitude>1e-3).

Two utilities:  U_rank = mean in-pool-gold rank-percentile of the fused score;
                U_R5   = Recall@5 = golds_in_top5 / ng of the fused score.
"""
import json
import math
import itertools
import numpy as np
import torch

from src.experiments.kg_hybrid import NE, EXPERT_NAMES, _load, _pooled_musd
from src.experiments import crag_fusion as CF
from src.experiments import crag_gates as G

DEV = torch.device("cpu")
DS = ["webqsp", "metaqa"]
NAMES = [EXPERT_NAMES[e] for e in range(NE)]                 # dense,offset,splade,relation,path
ALL = list(range(NE))
SUBSETS = [frozenset(c) for k in range(NE + 1) for c in itertools.combinations(ALL, k)]
FACT = [math.factorial(i) for i in range(NE + 1)]


def _sh_w(s, n=NE):
    return FACT[s] * FACT[n - 1 - s] / FACT[n]              # phi weight for |S|=s


def _int_w(s, n=NE):
    return FACT[s] * FACT[n - 2 - s] / FACT[n - 1]          # interaction weight for |S|=s (pair)


def _util_all_subsets(rp, y, ng):
    """Return {S: (U_rank, U_R5)} for all 32 subsets for one query. rp=(n,NE) rank_pct missing->0."""
    n = rp.shape[0]
    yb = y > 0.5
    ginpool = int(yb.sum())
    res = {}
    for S in SUBSETS:
        if not S:
            res[S] = (0.0, 0.0); continue
        cols = list(S)
        fused = rp[:, cols].max(1)
        order = np.argsort(-fused, kind="stable")
        # U_rank
        pos = np.empty(n, dtype=np.int64); pos[order] = np.arange(n)
        u_rank = float((1.0 - pos[yb] / n).mean()) if ginpool else 0.0
        # U_R5
        top5 = order[:5]
        u_r5 = float(yb[top5].sum()) / max(float(ng), 1.0)
        res[S] = (u_rank, u_r5)
    return res


def query_shapley(pb):
    T = G.cand_tensors(pb)
    rp = T["rank_pct"].cpu().numpy()
    y = pb.y.cpu().numpy()
    if y.sum() == 0:
        return None
    U = _util_all_subsets(rp, y, pb.ng)
    out = {}
    for ui, uname in enumerate(("rank", "R5")):
        phi = np.zeros(NE); cond_pos = np.zeros(NE); cond_neg = np.zeros(NE); cond_cnt = 0
        for e in ALL:
            for S in SUBSETS:
                if e in S:
                    continue
                marg = U[S | {e}][ui] - U[S][ui]
                phi[e] += _sh_w(len(S)) * marg
                if marg > 1e-3:
                    cond_pos[e] += 1
                elif marg < -1e-3:
                    cond_neg[e] += 1
            cond_cnt = len([S for S in SUBSETS if e not in S])
        inter = {}
        for e, f in itertools.combinations(ALL, 2):
            v = 0.0
            for S in SUBSETS:
                if e in S or f in S:
                    continue
                d = U[S | {e, f}][ui] - U[S | {e}][ui] - U[S | {f}][ui] + U[S][ui]
                v += _int_w(len(S)) * d
            inter[(e, f)] = v
        out[uname] = dict(phi=phi, cond_pos=cond_pos / max(cond_cnt, 1),
                          cond_neg=cond_neg / max(cond_cnt, 1), inter=inter)
    return out


def loo_delta(pb):
    T = G.cand_tensors(pb); rp = T["rank_pct"].cpu().numpy(); y = pb.y.cpu().numpy()
    if y.sum() == 0:
        return None
    U = _util_all_subsets(rp, y, pb.ng)
    full = frozenset(ALL)
    return {ui: np.array([U[full][i] - U[full - {e}][i] for e in ALL]) for i, ui in enumerate(("rank", "R5"))}


def main():
    data = _load(DS, DEV)
    mu, sd = _pooled_musd(data, DS, DEV, "train")
    out = {}
    for ds in DS:
        pbs = [CF.prepbatch(r, mu, sd, DEV, ds) for r in data[ds]["train"]]
        acc = {u: {"phi": [], "cond_pos": [], "cond_neg": [], "inter": {p: [] for p in itertools.combinations(ALL, 2)}}
               for u in ("rank", "R5")}
        loo = {u: [] for u in ("rank", "R5")}
        nq = 0
        for pb in pbs:
            sh = query_shapley(pb); ld = loo_delta(pb)
            if sh is None:
                continue
            nq += 1
            for u in ("rank", "R5"):
                acc[u]["phi"].append(sh[u]["phi"]); acc[u]["cond_pos"].append(sh[u]["cond_pos"])
                acc[u]["cond_neg"].append(sh[u]["cond_neg"])
                for p in acc[u]["inter"]:
                    acc[u]["inter"][p].append(sh[u]["inter"][p])
                loo[u].append(ld[u])
        res = {"nq": nq}
        for u in ("rank", "R5"):
            phi = np.mean(acc[u]["phi"], 0); cp = np.mean(acc[u]["cond_pos"], 0); cn = np.mean(acc[u]["cond_neg"], 0)
            lm = np.mean(loo[u], 0)
            inter = {f"{NAMES[e]}+{NAMES[f]}": round(float(np.mean(acc[u]["inter"][(e, f)])), 4)
                     for e, f in itertools.combinations(ALL, 2)}
            res[u] = {"per_expert": {NAMES[e]: {"loo_delta": round(float(lm[e]), 4), "shapley": round(float(phi[e]), 4),
                                                "cond_pos_frac": round(float(cp[e]), 3), "cond_neg_frac": round(float(cn[e]), 3)}
                                     for e in ALL},
                      "interaction": inter,
                      "top_synergy": dict(sorted(inter.items(), key=lambda kv: -kv[1])[:4]),
                      "top_redundancy": dict(sorted(inter.items(), key=lambda kv: kv[1])[:4])}
        out[ds] = res
    json.dump(out, open("results/L2/_shapley_audit_P2B.json", "w"), indent=2)
    for ds in DS:
        print(f"\n===== {ds} (nq={out[ds]['nq']}) =====")
        for u in ("rank", "R5"):
            print(f"-- U_{u} --")
            print(f"{'expert':10s} {'loo':>8s} {'shapley':>8s} {'cond+':>7s} {'cond-':>7s}")
            for e in ALL:
                s = out[ds][u]["per_expert"][NAMES[e]]
                print(f"{NAMES[e]:10s} {s['loo_delta']:>8.4f} {s['shapley']:>8.4f} {s['cond_pos_frac']:>7.2f} {s['cond_neg_frac']:>7.2f}")
            print("  synergy(+):", out[ds][u]["top_synergy"])
            print("  redundancy(-):", out[ds][u]["top_redundancy"])


if __name__ == "__main__":
    main()
