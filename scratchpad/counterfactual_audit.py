"""PARALLEL BRANCH P1/P2 — counterfactual / marginal-contribution target audit (source-only, parameter-free,
NO training, NO test). Decides whether marginal-contribution + synergy labels carry meaningful structure
before building the expert-token model (P3+).

Definitions (documented):
  fused score of an expert set S  = max over e in S of rank_pct_e(candidate)   (rank_pct missing->0; the
                                    same cooperative max-rank operator used as the frozen baseline).
  U(S) (utility)                  = mean over IN-POOL golds of the gold's rank_pct UNDER the fused score of S
                                    (rank_pct = 1 - pos/n; higher = golds sit nearer the fused top).
  Delta_e   (marginal)            = U(all5) - U(all5 \ {e})     >0 e adds evidence, ~0 redundant, <0 harmful.
  Delta_ef  (pair marginal)       = U(all5) - U(all5 \ {e,f}).
  Synergy(e,f)                    = Delta_ef - (Delta_e + Delta_f).
      Synergy > 0  : dropping the pair hurts MORE than the sum of dropping each alone -> the two COVER for
                     each other (complementary / redundant-substitutes: either alone suffices, both gone hurts).
      Synergy < 0  : the pair is a bundle whose value is already captured by each drop-one term (over-counted).
All rank features come from crag_gates.cand_tensors on the FROZEN substrate.
"""
import json
import itertools
import numpy as np
import torch

from src.experiments.kg_hybrid import NE, EXPERT_NAMES, _load, _pooled_musd
from src.experiments import crag_fusion as CF
from src.experiments import crag_gates as G

DEV = torch.device("cpu")
DS = ["webqsp", "metaqa"]
NAMES = [EXPERT_NAMES[e] for e in range(NE)]              # dense,offset,splade,relation,path (EXPERTS order)
ALL = list(range(NE))
EPS = 1e-6


def _fused_gold_util(rp, y, S):
    """U(S) for one query. rp=(n,NE) rank_pct (missing->0); y=bool gold; S=list of expert idx."""
    if not S or y.sum() == 0:
        return None
    fused = rp[:, S].max(1)                                # (n,)
    n = fused.shape[0]
    order = np.argsort(-fused, kind="stable")
    pos = np.empty(n, dtype=np.int64); pos[order] = np.arange(n)
    gold_rankpct = 1.0 - pos[y] / n
    return float(gold_rankpct.mean())


def query_counterfactuals(pb):
    T = G.cand_tensors(pb)
    rp = T["rank_pct"].cpu().numpy()                       # (n,NE) missing->0
    y = pb.y.cpu().numpy() > 0.5
    U_all = _fused_gold_util(rp, y, ALL)
    if U_all is None:
        return None
    delta = {}
    for e in ALL:
        U_me = _fused_gold_util(rp, y, [x for x in ALL if x != e])
        delta[e] = U_all - (U_me if U_me is not None else 0.0)
    dpair = {}
    for e, f in itertools.combinations(ALL, 2):
        U_ef = _fused_gold_util(rp, y, [x for x in ALL if x not in (e, f)])
        dpair[(e, f)] = U_all - (U_ef if U_ef is not None else 0.0)
    syn = {(e, f): dpair[(e, f)] - (delta[e] + delta[f]) for e, f in dpair}
    # strata flags (source-only, from rank features)
    valid = np.isfinite(pb.x_rank[:, [0, 1, 3, 5, 6]].cpu().numpy())    # EXPERTS cols availability per candidate
    def top1(e):
        col = rp[:, e]
        return int(np.argmax(col)) if col.max() > 0 else -1
    strata = {
        "dense_splade_agree": int(top1(G.DENSE) == top1(G.SPLADE) and top1(G.DENSE) >= 0),
        "offset_relation_agree": int(top1(G.OFFSET) == top1(G.RELATION) and top1(G.OFFSET) >= 0),
        "offset_only_gold": int((rp[y, G.OFFSET] > 0).any() and not (rp[y][:, [G.DENSE, G.SPLADE, G.RELATION]] > 0).any()) if y.sum() else 0,
        "relpath_avail": int(bool(pb.qavail[[5, 6]].sum() > 0)),
    }
    return delta, dpair, syn, strata


def main():
    data = _load(DS, DEV)
    mu, sd = _pooled_musd(data, DS, DEV, "train")
    out = {}
    for ds in DS:
        pbs = [CF.prepbatch(r, mu, sd, DEV, ds) for r in data[ds]["train"]]
        D = {e: [] for e in ALL}
        SYN = {p: [] for p in itertools.combinations(ALL, 2)}
        strat_delta = {}
        nq = 0
        for pb in pbs:
            r = query_counterfactuals(pb)
            if r is None:
                continue
            delta, dpair, syn, strata = r
            nq += 1
            for e in ALL:
                D[e].append(delta[e])
            for p in SYN:
                SYN[p].append(syn[p])
            for sk, sv in strata.items():
                if sv:
                    strat_delta.setdefault(sk, {e: [] for e in ALL})
                    for e in ALL:
                        strat_delta[sk][e].append(delta[e])
        summary = {}
        for e in ALL:
            arr = np.array(D[e])
            summary[NAMES[e]] = {"mean": round(float(arr.mean()), 4), "median": round(float(np.median(arr)), 4),
                                 "pos_frac": round(float((arr > 0.01).mean()), 3),
                                 "neg_frac": round(float((arr < -0.01).mean()), 3),
                                 "zero_frac": round(float((np.abs(arr) <= 0.01).mean()), 3)}
        syn_mean = {f"{NAMES[e]}+{NAMES[f]}": round(float(np.mean(SYN[(e, f)])), 4) for e, f in SYN}
        top_syn = dict(sorted(syn_mean.items(), key=lambda kv: -kv[1])[:5])
        bot_syn = dict(sorted(syn_mean.items(), key=lambda kv: kv[1])[:5])
        strat_out = {sk: {NAMES[e]: round(float(np.mean(v[e])), 4) for e in ALL} for sk, v in strat_delta.items()}
        out[ds] = {"nq": nq, "delta_e": summary, "top_synergy": top_syn, "bottom_synergy": bot_syn,
                   "strata_mean_delta": strat_out}
    json.dump(out, open("results/L2/_counterfactual_audit_P1P2.json", "w"), indent=2)
    for ds in DS:
        o = out[ds]
        print(f"\n===== {ds}  (nq={o['nq']}) =====")
        print(f"{'expert':10s} {'meanD':>8s} {'medD':>8s} {'pos%':>6s} {'neg%':>6s} {'zero%':>6s}")
        for e in ALL:
            s = o["delta_e"][NAMES[e]]
            print(f"{NAMES[e]:10s} {s['mean']:>8.4f} {s['median']:>8.4f} {s['pos_frac']:>6.2f} {s['neg_frac']:>6.2f} {s['zero_frac']:>6.2f}")
        print("top synergy:", o["top_synergy"])
        print("strata mean Delta:")
        for sk, v in o["strata_mean_delta"].items():
            print(f"  {sk:22s} {v}")


if __name__ == "__main__":
    main()
