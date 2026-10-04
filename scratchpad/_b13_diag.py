"""G2 B1.3 — GRAPH-ONLY ADMISSION BRIDGE (diagnostic-only; NO training, NO rebuild, NO encoder, NO TEST).

Reuses scratchpad/_b12/{ds}.npz. Question: can the GRAPH_ONLY discoveries (recovered golds absent from BOTH
Dense and SPLADE top-200) be admitted through a small PARAMETER-FREE structural reserve, without a learned
directional L2? Gold labels are used for EVALUATION ONLY — never to build any ranking/score/reserve.

Outputs:
 (1) population partition recovered golds -> RETRIEVAL_VISIBLE vs GRAPH_ONLY, per target.
 (2) graph-only rank diagnostic: for each inference-safe candidate-generation signal, rank graph-only golds
     WITHIN the geometry-added candidate list; R@{4,8,16,32,64} + median/p25/p75 rank (diagnostic ceilings).
 (3) gold-vs-negative separability among graph-only added candidates: Cohen's d + AUC per signal per dataset,
     and a cross-dataset consistency check (MetaQA vs 2Wiki sign agreement).
 (4) fixed-reserve ceiling: TOP50 = top(50-R) by BASE fused-RRF  UNION  top-R parameter-free structural
     discovery, R in {0,4,8,16,32}; GRAPH_ONLY_GOLD_ADMITTED / RETRIEVAL_VISIBLE_GOLD_RETAINED /
     NET_GOLD_GAIN@50 / GOLD_RECALL@50 / ALL@50 / POOL_CHURN. R uniform across all datasets (no per-target tuning).
 (5) SQuAD negative control: retrieval-visible gold displaced per R.
"""
import os, sys, json, time
import numpy as np

DSES = ["metaqa", "2wiki_clean", "squad_clean"]
TOP_POOL = 50; TOP200 = 200
RESERVE_R = [0, 4, 8, 16, 32]
RK = [4, 8, 16, 32, 64]
T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:.0f}s]", *a, flush=True)
np.random.seed(0)

# inference-safe candidate-generation signals (column refs into stored blocks). Higher = stronger discovery.
# name -> (block, col). All are candidate-local / query-relative percentiles or bounded ratios.
SIGNALS = {
    "s_dir_pct": ("S2old", 0),                 # directional discovery score (geometry mechanism)
    "dir_support_max_pct": ("S2old", 5),       # best directional support
    "n_struct_anchors_pct": ("S2old", 3),      # number/fraction of independent anchors
    "min_exp_hop_norm": ("S2old", 2),          # minimum expansion hop (closer = higher)
    "seed_support_frac": ("S1new", 1),         # structural support fraction
    "ppr_share_pct": ("S1new", 2),             # query-local PPR share
    "degree_local_ratio": ("S1new", 3),        # candidate-local degree ratio
    "struct_vs_retrieval_adv": ("S1new", 4),   # candidate-local structural score advantage
    "dir_margin": ("S2new", 1),                # directional margin among traversed
}
DISCOVERY_SIGNAL = "s_dir_pct"   # canonical parameter-free reserve ordering = the L1 geometry discovery score


def load(ds):
    z = np.load(f"scratchpad/_b12/{ds}.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    d["fused"] = d["S0"][:, 2].astype(np.float64)
    d["graph_only"] = (d["dense_rank"] >= TOP200) & (d["splade_rank"] >= TOP200)
    d["retr_vis"] = ~d["graph_only"]
    d["added"] = d["S2old"][:, 6] == 1
    d["gold"] = d["y"] == 1
    d["recg"] = (d["recov"] == 1) & (d["y"] == 1)
    return d


def sig(d, name):
    b, c = SIGNALS[name]
    return d[b][:, c].astype(np.float64)


def per_query(d):
    g = d["groups"]; st = np.concatenate([[0], np.cumsum(g)])
    for i in range(len(g)):
        yield i, int(st[i]), int(st[i + 1])


# ------------------------------------------------------------------ (1) population
def population(d):
    recg = d["recg"]; go = d["graph_only"]
    n_rec = int(recg.sum()); n_go = int((recg & go).sum()); n_rv = int((recg & ~go).sum())
    return {"recovered_golds": n_rec, "GRAPH_ONLY": n_go, "RETRIEVAL_VISIBLE": n_rv,
            "graph_only_frac": round(n_go / max(n_rec, 1), 4),
            "graph_only_nongold_added_cands": int((d["added"] & (d["y"] == 0) & go).sum())}


# ------------------------------------------------------------------ (2) graph-only rank diagnostic
def rank_diag(d):
    """Per signal: rank of each GRAPH_ONLY gold among its query's GEOMETRY-ADDED candidates (desc by signal)."""
    out = {}
    for name in SIGNALS:
        s = sig(d, name); ranks = []
        for i, a, b in per_query(d):
            am = d["added"][a:b]
            if am.sum() == 0:
                continue
            gogold = (d["recg"][a:b]) & (d["graph_only"][a:b]) & am
            if gogold.sum() == 0:
                continue
            sv = s[a:b][am]; local = np.where(am)[0]
            order = np.argsort(-sv, kind="stable")             # positions within added, best first
            rankpos = np.empty(len(order), np.int64); rankpos[order] = np.arange(len(order))
            gidx_local = np.searchsorted(local, np.where(gogold)[0])
            ranks.extend(rankpos[gidx_local].tolist())
        ranks = np.array(ranks)
        if len(ranks) == 0:
            out[name] = {"n": 0}; continue
        out[name] = {"n": int(len(ranks)),
                     **{f"R@{k}": round(float((ranks < k).mean()), 4) for k in RK},
                     "median_rank": float(np.median(ranks)),
                     "p25_rank": float(np.percentile(ranks, 25)),
                     "p75_rank": float(np.percentile(ranks, 75))}
    return out


# ------------------------------------------------------------------ (3) separability
def separability(d):
    """Cohen's d + AUC (gold > non-gold) among GRAPH-ONLY GEOMETRY-ADDED candidates, per signal."""
    mask = d["added"] & d["graph_only"]
    goldm = mask & d["gold"]; negm = mask & (~d["gold"])
    ng, nn = int(goldm.sum()), int(negm.sum())
    out = {"n_graph_only_gold": ng, "n_graph_only_nongold": nn, "signals": {}}
    if ng == 0 or nn == 0:
        return out
    for name in SIGNALS:
        s = sig(d, name); g = s[goldm]; nz = s[negm]
        pooled = np.sqrt((g.var() + nz.var()) / 2) + 1e-9
        cohend = float((g.mean() - nz.mean()) / pooled)
        # AUC via rank-sum (subsample negatives for speed if huge)
        if nn > 20000:
            nz2 = nz[np.random.permutation(nn)[:20000]]
        else:
            nz2 = nz
        allv = np.concatenate([g, nz2]); r = allv.argsort().argsort() + 1
        auc = float((r[:len(g)].sum() - len(g) * (len(g) + 1) / 2) / (len(g) * len(nz2)))
        out["signals"][name] = {"cohen_d": round(cohend, 4), "auc": round(auc, 4),
                                "gold_mean": round(float(g.mean()), 4), "nongold_mean": round(float(nz.mean()), 4)}
    return out


# ------------------------------------------------------------------ (4)+(5) fixed-reserve ceiling
def reserve_ceiling(d, R, disc_name=DISCOVERY_SIGNAL):
    disc = sig(d, disc_name)
    agg = dict(nq=0, go_adm=0, go_avail=0, rv_ret=0, rv_base=0, new=0, evict=0,
               RECn=0, RECd=0, ALL=0, churn=[])
    for i, a, b in per_query(d):
        y = d["gold"][a:b]; ng = int(y.sum())
        if ng == 0:
            continue
        fused = d["fused"][a:b]; am = d["added"][a:b]; go = d["graph_only"][a:b]; dloc = disc[a:b]
        n = len(y)
        forder = np.argsort(-fused, kind="stable")
        base = forder[:TOP_POOL]
        if R == 0:
            top = base.copy()
        else:
            verify = [int(x) for x in forder[:TOP_POOL - R]]
            vset = set(verify)
            dorder = np.argsort(-dloc, kind="stable")
            top = verify
            for c in dorder:
                if len(top) >= TOP_POOL:
                    break
                if am[c] and c not in vset:
                    top.append(int(c)); vset.add(int(c))
            if len(top) < TOP_POOL:                      # fill remainder from fused (few added available)
                for c in forder:
                    if len(top) >= TOP_POOL:
                        break
                    if int(c) not in vset:
                        top.append(int(c)); vset.add(int(c))
            top = np.array(top[:TOP_POOL])
        tset = set(top.tolist()); bset = set(base.tolist())
        golds = set(np.where(y)[0].tolist())
        go_gold = set(np.where(d["recg"][a:b] & go)[0].tolist()) & golds
        rv_gold = golds - go_gold
        gt = golds & tset; gb = golds & bset
        agg["nq"] += 1
        agg["go_adm"] += len(go_gold & tset); agg["go_avail"] += len(go_gold)
        agg["rv_ret"] += len(rv_gold & tset); agg["rv_base"] += len(rv_gold)
        agg["new"] += len(gt - gb); agg["evict"] += len(gb - gt)
        agg["RECn"] += len(gt); agg["RECd"] += ng; agg["ALL"] += int(len(gt) == ng)
        agg["churn"].append(len(tset ^ bset) / (2 * TOP_POOL))
    nq = max(agg["nq"], 1)
    return {"R": R, "GRAPH_ONLY_GOLD_ADMITTED": agg["go_adm"], "graph_only_gold_avail": agg["go_avail"],
            "GRAPH_ONLY_ADMISSION_frac": round(agg["go_adm"] / max(agg["go_avail"], 1), 4),
            "RETRIEVAL_VISIBLE_GOLD_RETAINED": agg["rv_ret"], "retr_visible_gold_base": agg["rv_base"],
            "RETRIEVAL_VISIBLE_RETENTION": round(agg["rv_ret"] / max(agg["rv_base"], 1), 4),
            "retrieval_visible_gold_displaced": agg["rv_base"] - agg["rv_ret"],
            "NEW_GOLDS_ADMITTED": agg["new"], "OLD_GOLDS_EVICTED": agg["evict"],
            "NET_GOLD_GAIN@50": agg["new"] - agg["evict"],
            "GOLD_RECALL@50": round(agg["RECn"] / max(agg["RECd"], 1), 4),
            "ALL@50": round(agg["ALL"] / nq, 4),
            "POOL_CHURN_mean": round(float(np.mean(agg["churn"])), 4)}


def main():
    log("=== B1.3 GRAPH-ONLY ADMISSION BRIDGE (diagnostic-only) ===")
    DATA = {ds: load(ds) for ds in DSES}
    res = {"phase": "G2 B1.3 graph-only admission bridge; diagnostic-only; no training/rebuild/encoder/TEST",
           "reserve_R_grid": RESERVE_R, "discovery_signal": DISCOVERY_SIGNAL,
           "rank_diag_K": RK, "signals": list(SIGNALS.keys()), "per_dataset": {}}
    for ds in DSES:
        d = DATA[ds]
        pop = population(d); rd = rank_diag(d); sep = separability(d)
        ceil = [reserve_ceiling(d, R) for R in RESERVE_R]
        res["per_dataset"][ds] = {"population": pop, "graph_only_rank_diag": rd,
                                  "separability": sep, "fixed_reserve_ceiling": ceil}
        log(f"{ds}: pop={pop}")
        log(f"  rank_diag[{DISCOVERY_SIGNAL}]={rd[DISCOVERY_SIGNAL]}")
        best = max(sep["signals"].items(), key=lambda kv: abs(kv[1]["auc"] - 0.5)) if sep["signals"] else None
        log(f"  separability best-AUC signal={best[0] if best else None} {best[1] if best else ''}")
        for c in ceil:
            log(f"  reserve R={c['R']}: GO_adm={c['GRAPH_ONLY_GOLD_ADMITTED']}/{c['graph_only_gold_avail']} "
                f"RVret={c['RETRIEVAL_VISIBLE_RETENTION']} NET={c['NET_GOLD_GAIN@50']} "
                f"R@50={c['GOLD_RECALL@50']} ALL@50={c['ALL@50']} churn={c['POOL_CHURN_mean']}")

    # cross-dataset consistency of separability sign (MetaQA vs 2Wiki)
    consist = {}
    sm = DATA["metaqa"]; s2 = DATA["2wiki_clean"]
    sepm = res["per_dataset"]["metaqa"]["separability"]["signals"]
    sep2 = res["per_dataset"]["2wiki_clean"]["separability"]["signals"]
    for name in SIGNALS:
        if name in sepm and name in sep2:
            am = sepm[name]["auc"]; a2 = sep2[name]["auc"]
            consist[name] = {"metaqa_auc": am, "2wiki_auc": a2,
                             "both_separate_gold_up": bool(am > 0.5 and a2 > 0.5),
                             "min_auc": round(min(am, a2), 4)}
    res["separability_consistency_metaqa_vs_2wiki"] = consist
    ordered = sorted(consist.items(), key=lambda kv: -kv[1]["min_auc"])
    log("CONSISTENCY (signals separating gold-up on BOTH, by min AUC):")
    for name, v in ordered:
        log(f"  {name}: metaqa {v['metaqa_auc']} / 2wiki {v['2wiki_auc']} both_up={v['both_separate_gold_up']}")

    os.makedirs("results/GENERALIZATION", exist_ok=True)
    json.dump(res, open("results/GENERALIZATION/_g2_b13_bridge.json", "w"), indent=1, default=str)
    log("B13_DIAG_DONE -> results/GENERALIZATION/_g2_b13_bridge.json")


if __name__ == "__main__":
    main()
