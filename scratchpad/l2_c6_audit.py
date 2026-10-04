"""C6 Task 1 (negative depth distribution) + Task 3 (TRAIN & VAL weight-archetype oracle ceiling).
Both need the per-query FULL-SCOPE C0 RRF ranking, computed once per query via SH.contribs and reused.
Archetype set is FIXED (identical to the VAL +0.0395 ceiling) — NOT created from VAL. VAL golds used for
oracle CEILING only. No TEST touched."""
import sys, os, json, time
sys.path.insert(0, "scratchpad")
import numpy as np
import l2_shapley as SH
from collections import Counter

DS = ("2wiki_clean", "musique_clean")
CORP = "data/l2_corpus"; OUT = "results/L2/_ctrl"; os.makedirs(OUT, exist_ok=True)
N = 5; K0 = SH.K0
BUCKETS = [(0, 10), (11, 20), (21, 50), (51, 100), (101, 500), (501, 10**9)]
BNAMES = ["r0-10", "r11-20", "r21-50", "r51-100", "r101-500", "r>500"]
# SAME fixed archetypes used for the VAL +0.0395 ceiling (results/L2/_ctrl/_oracle.json)
ARCH = {"equal": [1, 1, 1, 1, 1], "up_relation": [1, 1, 1, 1, 2.5], "up_offmix": [1, 1, 1.6, 1.6, 1],
        "up_splade": [1, 1.6, 1, 1, 1], "up_dense": [1.6, 1, 1, 1, 1], "down_relation": [1, 1, 1, 1, 0.4]}
ANAMES = list(ARCH); AW = {k: np.array(v, np.float64) for k, v in ARCH.items()}


def bucket_of(rank):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= rank <= hi:
            return i
    return len(BUCKETS) - 1


def ndcg50(score, gold):
    order = np.argsort(-score, kind="stable"); rankof = np.empty(len(score), np.int64); rankof[order] = np.arange(len(score))
    gr = rankof[gold]; ng = len(gold)
    dcg = np.sum([1.0 / np.log2(x + 2) for x in gr if x < 50])
    idcg = np.sum([1.0 / np.log2(i + 2) for i in range(min(ng, 50))])
    return (dcg / idcg) if idcg > 0 else 0.0


def load(ds, split):
    d = f"{CORP}/{ds}/{split}"
    A = {k: np.load(f"{d}/{k}.npy") for k in ("query_offsets", "labels", "dense_score",
         "splade_scope_score", "offset_score", "mixture_score", "relation_qwen_score", "relation_mask")}
    return A, d


def run_split(ds, split, do_depth):
    A, d = load(ds, split)
    off = A["query_offsets"]; lab_all = A["labels"]; nq = len(off) - 1
    if do_depth:
        src = json.load(open(f"{d}/train_sub_source.json"))
        tso = np.load(f"{d}/train_sub_offsets.npy"); tsl = np.load(f"{d}/train_sub_local.npy")
        depth = {bn: Counter() for bn in BNAMES}          # bucket -> provenance counts (negatives only)
        depth_tot = Counter()
    orc_sum = {a: 0.0 for a in ANAMES}; best_sum = 0.0; nqv = 0; choice = Counter()
    t0 = time.time()
    for qi in range(nq):
        s, e = int(off[qi]), int(off[qi + 1]); n = e - s
        lab = lab_all[s:e]; gold = np.where(lab == 1)[0]
        if len(gold) == 0:
            continue
        C = SH.contribs(A["dense_score"][s:e], A["splade_scope_score"][s:e], A["offset_score"][s:e],
                        A["mixture_score"][s:e], A["relation_qwen_score"][s:e], A["relation_mask"][s:e])  # (5,n)
        # C0 full-scope ranking
        c0 = C.sum(0); order = np.argsort(-c0, kind="stable"); rankof = np.empty(n, np.int64); rankof[order] = np.arange(n)
        # ---- Task 1: depth of current train_sub NEGATIVES ----
        if do_depth:
            a, b = int(tso[qi]), int(tso[qi + 1]); loc = tsl[a:b]; prov = src[a:b]
            for li, pv in zip(loc, prov):
                if lab[li] == 1:
                    continue                              # skip positives
                bi = bucket_of(int(rankof[li])); depth[BNAMES[bi]][pv] += 1; depth_tot[BNAMES[bi]] += 1
        # ---- Task 3: per-query oracle over fixed archetypes (full scope) ----
        nqv += 1; best = -1.0; besta = None
        for aname in ANAMES:
            sc = (AW[aname][:, None] * C).sum(0)
            nd = ndcg50(sc, gold); orc_sum[aname] += nd
            if nd > best + 1e-12: best = nd; besta = aname
        best_sum += best; choice[besta] += 1
        if (qi + 1) % 3000 == 0:
            print(f"  [{ds}/{split}] {qi+1}/{nq} {time.time()-t0:.0f}s", flush=True)
    res = {"n_queries_valid": nqv,
           "equal_ndcg50": round(orc_sum["equal"] / nqv, 4),
           "oracle_ndcg50": round(best_sum / nqv, 4),
           "oracle_gain": round((best_sum - orc_sum["equal"]) / nqv, 4),
           "archetype_mean_ndcg50": {a: round(orc_sum[a] / nqv, 4) for a in ANAMES},
           "oracle_choice_counts": dict(choice)}
    if do_depth:
        res["neg_depth_totals"] = dict(depth_tot)
        tot = sum(depth_tot.values())
        res["neg_depth_proportion"] = {bn: round(depth_tot[bn] / tot, 4) for bn in BNAMES}
        res["neg_depth_by_provenance"] = {bn: dict(depth[bn]) for bn in BNAMES}
    print(f"DONE {ds}/{split} valid={nqv} equal={res['equal_ndcg50']} oracle={res['oracle_ndcg50']} (+{res['oracle_gain']}) {time.time()-t0:.0f}s", flush=True)
    return res


def main():
    out = {}
    for ds in DS:
        out[ds] = {"train": run_split(ds, "train", do_depth=True),
                   "val": run_split(ds, "val", do_depth=False)}
    # pooled oracle
    for split in ("train", "val"):
        nq = sum(out[ds][split]["n_queries_valid"] for ds in DS)
        eq = sum(out[ds][split]["equal_ndcg50"] * out[ds][split]["n_queries_valid"] for ds in DS) / nq
        orc = sum(out[ds][split]["oracle_ndcg50"] * out[ds][split]["n_queries_valid"] for ds in DS) / nq
        out.setdefault("POOLED", {})[split] = {"equal_ndcg50": round(eq, 4), "oracle_ndcg50": round(orc, 4),
                                               "oracle_gain": round(orc - eq, 4), "n_queries_valid": nq}
    json.dump(out, open(f"{OUT}/_c6_audit.json", "w"), indent=1, default=str)
    print("WROTE results/L2/_ctrl/_c6_audit.json")
    print("POOLED", json.dumps(out["POOLED"], indent=1))


if __name__ == "__main__":
    main()
