"""Oracle ceiling of per-query weight adaptivity on VAL (uses val golds => CEILING ONLY, not a model).
For each query pick the weight archetype maximizing that query's NDCG@50; report pooled vs equal.
Also report an inference-feature 'heuristic' gate (relation-signal -> up_relation) as an honest lower bar."""
import sys; sys.path.insert(0, "scratchpad")
import numpy as np, torch
import l2_controller as CT

va = {ds: CT.load_cache(ds, "val") for ds in CT.DS}
ARCH = {"equal": [1, 1, 1, 1, 1], "up_relation": [1, 1, 1, 1, 2.5], "up_offmix": [1, 1, 1.6, 1.6, 1],
        "up_splade": [1, 1.6, 1, 1, 1], "up_dense": [1.6, 1, 1, 1, 1], "down_relation": [1, 1, 1, 1, 0.4]}
names = list(ARCH); W = {k: np.array(v, np.float32) for k, v in ARCH.items()}


def perq_ndcg(ds, wvec):
    c = va[ds]; out = []
    for qi in range(len(c["q_off"]) - 1):
        a, b = int(c["q_off"][qi]), int(c["q_off"][qi + 1])
        if b == a: continue
        r = c["r"][a:b].astype(np.float32)
        s = (r * wvec).sum(1)
        ga, gb = int(c["gold_off"][qi]), int(c["gold_off"][qi + 1]); gold = c["gold_loc"][ga:gb]
        order = np.argsort(-s, kind="stable"); rankof = np.empty(len(s), np.int64); rankof[order] = np.arange(len(s))
        gr = rankof[gold]; ng = len(gold)
        dcg = sum(1.0 / np.log2(x + 2) for x in gr if x < 50); idcg = sum(1.0 / np.log2(i + 2) for i in range(min(ng, 50)))
        out.append(dcg / idcg if idcg else 0.0)
    return np.array(out)


def relsignal(ds):
    c = va[ds]; sig = []
    for qi in range(len(c["q_off"]) - 1):
        a, b = int(c["q_off"][qi]), int(c["q_off"][qi + 1])
        if b == a: continue
        sig.append(c["qscalar"][qi][1] > 0.5)   # relation_any_signal_present
    return np.array(sig)


tab = {}
allnd = {ds: {nm: perq_ndcg(ds, W[nm]) for nm in names} for ds in CT.DS}
tot_eq = 0; tot_or = 0; tot_h = 0; n = 0
best_choice = {nm: 0 for nm in names}
for ds in CT.DS:
    nd = allnd[ds]; M = np.vstack([nd[nm] for nm in names])  # (A, nq)
    eq = nd["equal"]; orc = M.max(0); choice = M.argmax(0)
    sig = relsignal(ds)
    heur = np.where(sig, nd["up_relation"], nd["equal"])     # inference-feature heuristic (honest)
    tot_eq += eq.sum(); tot_or += orc.sum(); tot_h += heur.sum(); n += len(eq)
    for i, nm in enumerate(names): best_choice[nm] += int((choice == i).sum())
    tab[ds] = {"equal": round(float(eq.mean()), 4), "oracle_max": round(float(orc.mean()), 4),
               "heuristic_relsig": round(float(heur.mean()), 4),
               "oracle_gain": round(float(orc.mean() - eq.mean()), 4)}
tab["POOLED"] = {"equal": round(tot_eq / n, 4), "oracle_max": round(tot_or / n, 4),
                 "heuristic_relsig": round(tot_h / n, 4), "oracle_gain": round((tot_or - tot_eq) / n, 4)}
tab["oracle_archetype_choice_counts"] = best_choice
import json; print(json.dumps(tab, indent=1))
