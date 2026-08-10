"""
UKB result store: co-locate results/benchmarks INSIDE each dataset's UKB.
=========================================================================
The UKB is the store for results too, organized by LEVEL x DATASET so it can be
queried directly:
    data/ukb_storage/{dataset}/results/{L1,L2,L3,cross}/{artifact}.json
plus a cross-dataset manifest at data/ukb_storage/_index/manifest.json
(dataset -> level -> artifact -> path + headline metrics).

This module is the single source of truth for result paths:
  rpath(dataset, level, artifact)  -> canonical new path (writers use this)
  resolve(dataset, level, artifact) -> new path if present else the legacy path
                                       (readers use this during the transition)
  classify(legacy_path)            -> (dataset, level, artifact) or None
  migrate(datasets)                -> move matching legacy files into the UKB
  build_index()                    -> (re)build the queryable manifest

Cutover plan: migrate done/unguarded datasets now; the rest (guarded by the
running campaign) are swept in once the jobs finish and the campaign scripts are
repointed here. Legacy fallback keeps everything readable meanwhile.
"""
import os
import re
import glob
import json
import shutil
import logging

log = logging.getLogger("pipeline.ukb_results")

UKB = "data/ukb_storage"
INDEX_DIR = os.path.join(UKB, "_index")
# longest names first so '2wiki_clean' matches before '2wiki'
DATASETS = ["2wiki_clean", "musique_clean", "hotpotqa_clean", "squad_clean",
            "metaqa", "2wiki", "musique", "squad"]

# legacy scattered dir -> (level, artifact-from-filename resolver)
# resolver returns the artifact basename (no dataset, no .json)
def _overlap_artifact(fn):
    for tok, art in [("S1loss", "loss_ablation"), ("S1nohnm", "hnm_legacy"),
                     ("S2struct", "structure_sweep"), ("S2jigsaw", "jigsaw_loss"),
                     ("hnm_sweep", "hnm_sweep")]:
        if tok in fn:
            return art
    return None

def _variant(fn, ds):
    """trailing config after the dataset, e.g. musique_clean_overlap1_knn1.json -> overlap1_knn1"""
    stem = fn[:-5] if fn.endswith(".json") else fn
    v = stem.replace(ds, "", 1).strip("_")
    return v or None

DIRMAP = {
    "overlap_ablation": ("L1", _overlap_artifact),
    "finetune_ablation": ("L1", lambda fn: "encoder_ft"),
    "adaptive_k": ("L1", lambda fn: "adaptive_k"),
    "multiproto": ("L1", "multiproto"),           # variant appended
    "query_decomp": ("L1", "query_decomp"),
    "gnn_ablation": ("L1", lambda fn: "gnn"),
    "level_1": ("L1", lambda fn: "bench"),
    "partition_ablation": ("L1", "partition"),     # variant-aware: {ds}.json + {ds}_trained_confirm.json
    "overlap_partsize": ("L1", "overlap_partsize"),
    "l3_recovery": ("L3", "recovery"),
    "pool_narrow": ("L3", "pool_narrow"),
}
# research/ files: prefix -> (level, artifact)
RESEARCH = {"champion_": ("L1", "champion"), "l1l3_": ("cross", "l1l3"),
            "reach_": ("L3", "reachability"), "ppr_": ("L3", "ppr"),
            "why_": ("L3", "why"), "l3_latency_": ("L3", "latency")}


def _ds_in(name):
    for d in DATASETS:
        if d in name:
            return d
    return None


def classify(path):
    """Map a legacy result path -> (dataset, level, artifact) or None."""
    parts = path.replace("\\", "/").split("/")
    fn = parts[-1]
    if not fn.endswith(".json") and not fn.endswith(".csv"):
        return None
    ds = _ds_in(fn)
    if ds is None:
        return None
    d = parts[-2] if len(parts) >= 2 else ""
    if d == "research":
        for pre, (lvl, art) in RESEARCH.items():
            if fn.startswith(pre):
                return ds, lvl, art
        return None
    if d in DIRMAP:
        lvl, resolver = DIRMAP[d]
        if callable(resolver):
            art = resolver(fn)
        else:                                       # dir with config variants
            v = _variant(fn, ds)
            art = f"{resolver}__{v}" if v else resolver
        if art:
            ext = ".csv" if fn.endswith(".csv") else ".json"
            return ds, lvl, art + ext
    return None


def rpath(dataset, level, artifact):
    """Canonical UKB result path (writers use this). artifact may include extension."""
    if not artifact.endswith((".json", ".csv")):
        artifact += ".json"
    d = os.path.join(UKB, dataset, "results", level)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, artifact)


def resolve(dataset, level, artifact, legacy_candidates=()):
    """Read path: prefer the UKB location, fall back to a legacy path during transition."""
    p = rpath(dataset, level, artifact)
    if os.path.exists(p):
        return p
    for c in legacy_candidates:
        if os.path.exists(c):
            return c
    return p


def migrate(datasets, move=True):
    """Move (or copy) every legacy result belonging to `datasets` into the UKB tree."""
    moved = []
    roots = ["results/overlap_ablation", "results/finetune_ablation", "results/adaptive_k",
             "results/multiproto", "results/query_decomp", "results/gnn_ablation",
             "results/level_1", "results/partition_ablation", "results/overlap_partsize",
             "results/l3_recovery", "results/pool_narrow", "results/research"]
    for root in roots:
        for path in glob.glob(os.path.join(root, "*")):
            if os.path.isdir(path):
                continue
            c = classify(path)
            if not c:
                continue
            ds, lvl, art = c
            if ds not in datasets:
                continue
            dst = rpath(ds, lvl, art)
            (shutil.move if move else shutil.copy2)(path, dst)
            moved.append((path, dst))
    for src, dst in moved:
        log.info(f"  {src}  ->  {dst}")
    log.info(f"migrated {len(moved)} files for {datasets}")
    return moved


def _headline(path):
    """Pull a few key metrics for the manifest so the index is queryable at a glance."""
    if not path.endswith(".json"):
        return {}
    try:
        d = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    if isinstance(d, dict):
        if "champion_10pct" in d:
            ch = d["champion_10pct"]
            out = {"champion": f"{ch.get('config')}/{ch.get('loss')}",
                   "fullcov": ch.get("fullcov_pm"), "gt_recall": ch.get("gt_recall_pm")}
        elif "reachability_pct" in d:
            out = {"reach_h2": d["reachability_pct"].get("h2"),
                   "frontier_h2%": d.get("frontier_pct_of_corpus", {}).get("h2")}
        elif "results" in d and isinstance(d["results"], dict):
            out = {"n_configs": len(d["results"])}
    return out


def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    manifest = {}
    for ds in sorted(set(DATASETS)):
        base = os.path.join(UKB, ds, "results")
        if not os.path.isdir(base):
            continue
        for lvl in sorted(os.listdir(base)):
            ld = os.path.join(base, lvl)
            if not os.path.isdir(ld):
                continue
            for fn in sorted(os.listdir(ld)):
                p = os.path.join(ld, fn)
                manifest.setdefault(ds, {}).setdefault(lvl, {})[fn] = {
                    "path": p.replace("\\", "/"), **_headline(p)}
    with open(os.path.join(INDEX_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    n = sum(len(v2) for v in manifest.values() for v2 in v.values())
    log.info(f"index: {n} artifacts across {len(manifest)} datasets -> {INDEX_DIR}/manifest.json")
    return manifest


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="UKB result store: migrate + index.")
    p.add_argument("--migrate", nargs="*", default=None, help="datasets to migrate into the UKB tree")
    p.add_argument("--copy", action="store_true", help="copy instead of move")
    p.add_argument("--index", action="store_true", help="(re)build the manifest")
    a = p.parse_args(argv)
    if a.migrate is not None:
        migrate(a.migrate or DATASETS, move=not a.copy)
    if a.index or a.migrate is not None:
        build_index()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
