"""
Per-dataset L1 champion: POOL-MATCHED frontier over the trained ablations.
==========================================================================
Raw FullCov@K flatters overlap (bigger pool per partition) -- e.g. overlap2 "wins"
only via a 100-400x pool blowup. The fair comparison fixes the effective POOL
(docs the router hands downstream) and asks which (structure, loss) maximizes
FullCov + recall there. This is exactly what L2 cares about (minimize nodes), so
the L1 champion and the L2 budget share one axis.

Method: each config's membership explosion m = mean_memberships_per_doc means the
top-K routed partitions hold ~ K * n * m / npart docs. To hit a target pool
fraction f, read the config at K_equiv = f * npart / m partitions, interpolating
FullCov/gt_recall over the measured K grid. Rank configs by pool-matched FullCov
at each target f. Loss variants: KL (S2struct) + coverage (S2jigsaw).
Writes results/research/champion_{dataset}.json.
"""
import os
import json
import logging
import argparse

import numpy as np

log = logging.getLogger("experiments.champion")

K_GRID = [1, 3, 5, 10, 20, 50, 100, 200]
POOL_FRACS = [0.05, 0.10]


def _interp(kq, ks, ys):
    kq = min(max(kq, ks[0]), ks[-1])
    return float(np.interp(kq, ks, ys))


def _configs(ds):
    """Merge EVERY (config, loss) cell from ALL overlap_retrain sweep files for the
    dataset — S2struct/S2jigsaw/S5/Sner/… at legacy paths + migrated UKB store — so
    new atoms (NER/SPLADE) and losses auto-enter the champion without code changes."""
    import glob
    files = glob.glob(f"results/overlap_ablation/{ds}_overlap_retrain_*.json")
    files += glob.glob(f"data/ukb_storage/{ds}/results/L1/*.json")
    out = {}
    for p in files:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "results" not in d or "npart" not in d:
            continue
        npart = d["npart"]
        for r in d["results"].values():
            if not (isinstance(r, dict) and "config" in r and "loss" in r
                    and "best" in r and "mean_memberships_per_doc" in r):
                continue                                    # skip non-sweep files (hnm/champion/etc.)
            b = r["best"]
            out[f"{r['config']}__{r['loss']}"] = {
                "config": r["config"], "loss": r["loss"], "npart": npart,
                "mem_per_doc": r["mean_memberships_per_doc"],
                "fcov": [b.get(f"full_coverage@{k}", 0.0) for k in K_GRID],
                "gtr": [b.get(f"gt_recall@{k}", 0.0) for k in K_GRID],
            }
    return out


def run(dataset):
    cfgs = _configs(dataset)
    if not cfgs:
        log.warning(f"[{dataset}] no ablation results yet"); return None
    out = {"dataset": dataset, "k_grid": K_GRID, "pool_fracs": POOL_FRACS, "frontier": {}}
    for f in POOL_FRACS:
        ranked = []
        for name, c in cfgs.items():
            kq = f * c["npart"] / max(c["mem_per_doc"], 1e-6)
            ranked.append({
                "config": c["config"], "loss": c["loss"], "mem_per_doc": round(c["mem_per_doc"], 2),
                "k_equiv": round(kq, 2),
                "fullcov_pm": round(_interp(kq, K_GRID, c["fcov"]), 2),
                "gt_recall_pm": round(_interp(kq, K_GRID, c["gtr"]), 2),
            })
        ranked.sort(key=lambda x: (-x["fullcov_pm"], -x["gt_recall_pm"]))
        out["frontier"][f"pool_{int(f*100)}pct"] = ranked
    # champion = best pool-matched FullCov at 10% (tie-break recall), excluding absurd pool blowup
    top = out["frontier"]["pool_10pct"]
    champ = next((r for r in top if r["mem_per_doc"] <= 10), top[0])
    out["champion_10pct"] = champ
    out["status"] = ("PROVISIONAL — best of {structure(title/kNN) × loss(KL/coverage)} only. "
                     "NER/SPLADE edge atoms and relation-aware routing are NOT yet in this sweep; "
                     "the final champion is decided after the full L1 method space is explored.")
    from src.pipeline.ukb_results import rpath
    with open(rpath(dataset, "L1", "champion"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] champion@10%pool: {champ['config']}/{champ['loss']} "
             f"FullCov={champ['fullcov_pm']} gtR={champ['gt_recall_pm']} (mem/doc {champ['mem_per_doc']})")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Pool-matched per-dataset L1 champion.")
    p.add_argument("--datasets", nargs="+",
                   default=["2wiki_clean", "hotpotqa_clean", "metaqa", "musique_clean", "squad_clean"])
    a = p.parse_args(argv)
    for ds in a.datasets:
        run(ds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
