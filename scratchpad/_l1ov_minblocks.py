"""PHASE 4 -- minimum blocks required by the gold/reference nodes, per hard partition.

The program asks for this number to be REPORTED and explicitly NOT used as the winning
objective: a partition that packs every query's required nodes into few blocks is easier to
cover, but the previous program already showed containment does not predict retrieval utility
(HotpotQA gained while every containment statistic got worse).  So this is a descriptive
statistic sitting next to the utility table, never an objective.

Gold is touched only here, after the partition exists -- no gold or query information enters
any build.

  python scratchpad/_l1ov_minblocks.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import numpy as np
import _l1ep_pu as PU

OUT = "results/GENERALIZATION/G2_L1_PARTITION_SEARCH/L1_UNIVERSAL_PARTITION_SEARCH"
PARTS = "scratchpad/_l1ep/parts"
DS = ["metaqa", "webqsp", "2wiki_clean", "musique_clean", "squad_clean", "hotpotqa_clean"]


def stats(hard, g, gptr, nq):
    need_blocks, need_nodes = [], []
    for qi in range(nq):
        req = g[gptr[qi]:gptr[qi + 1]]
        if len(req) == 0:
            continue
        need_nodes.append(len(req))
        need_blocks.append(len(np.unique(hard[np.asarray(req, np.int64)])))
    b = np.asarray(need_blocks, np.float64)
    n = np.asarray(need_nodes, np.float64)
    return {"queries": int(len(b)),
            "required_nodes_mean": round(float(n.mean()), 3),
            "min_blocks_mean": round(float(b.mean()), 3),
            "min_blocks_median": float(np.median(b)),
            "min_blocks_p90": float(np.percentile(b, 90)),
            "min_blocks_max": int(b.max()),
            "frac_single_block": round(float((b == 1).mean()), 4),
            "colocation_ratio": round(float(b.sum() / max(n.sum(), 1e-9)), 4)}


def main():
    out = {"NOTE": ("reported per the program and deliberately NOT used as the winning "
                    "objective.  On this matrix the association with utility is real "
                    "(see PREDICTIVENESS) but not sufficient: it cannot vary at all on a "
                    "one-gold-node corpus whose utility does vary, and the previous "
                    "program has a standing counter-example where every containment "
                    "statistic worsened while F6 ALL@50 gained significantly"),
           "DEFINITION": ("min_blocks = number of DISTINCT partitions the required nodes of one "
                          "query fall into; the fewest blocks any selector would have to pick"),
           "PER_CORPUS": {}}
    for ds in DS:
        g, gptr, rows, hops = PU.gold_rows(ds)
        tags = ["CURRENT"] + sorted(f[:-4].split("__", 1)[1] for f in os.listdir(PARTS)
                                    if f.startswith(ds + "__H") and f.endswith(".npy"))
        per = {}
        for tag in tags:
            if tag == "CURRENT":
                hard, npart = PU.load_assignment(ds, "CURRENT")
            else:
                fp = f"{PARTS}/{ds}__{tag}.npy"
                if not os.path.exists(fp):
                    continue
                hard = np.load(fp)
                npart = int(np.asarray(hard).max()) + 1
            hard = np.asarray(hard, np.int64)
            nq = len(gptr) - 1
            per[tag] = dict(stats(hard, g, gptr, nq), npart=int(npart))
            s = per[tag]
            print(f"{ds:16s} {tag:50s} mean {s['min_blocks_mean']:6.3f}  "
                  f"median {s['min_blocks_median']:4.1f}  max {s['min_blocks_max']:4d}  "
                  f"single-block {s['frac_single_block']:.1%}")
        out["PER_CORPUS"][ds] = per
    # Does containment PREDICT utility?  Reported, because the honest answer is "partly".
    import glob
    rep = {os.path.basename(f)[7:-5]: json.load(open(f))
           for f in glob.glob(f"{OUT}/hard_partitions/REPLAY_*.json")}
    xs, ys, cells = [], [], []
    for ds, per in out["PER_CORPUS"].items():
        R = rep.get(ds) or {}
        if "CURRENT" not in R or "CURRENT" not in per:
            continue
        b0, f0 = per["CURRENT"]["min_blocks_mean"], R["CURRENT"]["F6_ALL_P50"]
        for tag, st in per.items():
            if tag == "CURRENT" or tag not in R:
                continue
            db = round(st["min_blocks_mean"] - b0, 4)
            du = round(R[tag]["F6_ALL_P50"] - f0, 4)
            xs.append(db); ys.append(du)
            cells.append({"ds": ds, "tag": tag, "d_min_blocks": db, "d_F6_ALL_P50": du})
    if len(xs) > 2:
        from scipy.stats import spearmanr
        a, b = np.asarray(xs), np.asarray(ys)
        sr = spearmanr(a, b)
        deg = sorted({d for d in out["PER_CORPUS"]
                      if out["PER_CORPUS"][d].get("CURRENT", {}).get("min_blocks_max") == 1})
        out["PREDICTIVENESS"] = {
            "n_cells": len(xs),
            "pearson_r": round(float(np.corrcoef(a, b)[0, 1]), 4),
            "spearman_rho": round(float(sr.statistic), 4),
            "spearman_p": float(f"{sr.pvalue:.3g}"),
            "PEARSON_CAVEAT": ("Pearson is dominated by two MetaQA cells at d_min_blocks ~ -1.2; "
                               "Spearman is the honest summary"),
            "DEGENERATE_CORPORA": deg,
            "DEGENERATE_NOTE": ("these corpora have exactly one required node per query, so "
                                "min_blocks == 1 for EVERY partition and containment cannot "
                                "vary -- yet their utility does, which is why containment is "
                                "reported and not optimised"),
            "CELLS": sorted(cells, key=lambda c: (c["ds"], c["tag"]))}
        print(f"containment vs utility: spearman rho={out['PREDICTIVENESS']['spearman_rho']:+.3f} "
              f"p={out['PREDICTIVENESS']['spearman_p']:.3g} over {len(xs)} cells "
              f"(degenerate: {deg})")
    os.makedirs(f"{OUT}/hard_partitions", exist_ok=True)
    fp = f"{OUT}/hard_partitions/MIN_BLOCKS_REQUIRED.json"
    json.dump(out, open(fp, "w"), indent=1)
    print("wrote", fp)
    return out


if __name__ == "__main__":
    main()
