"""
Single-source L1 partition-routing benchmark.
=============================================
ONE place that aggregates every canonical per-cell rerank result into ONE reproducible
table. A cell = (dataset x encoder). For each cell we read the frozen result file
    data/ukb_storage/{dataset}/results/L1_select/rerank100_{dataset}[_{subdir}].json
(produced by l1_rerank100, which runs the full offset stack: dense + rel_hard + rel_2hop
+ gated-mlpT, equal-RRF fused, sum/max votes over overlap membership, best-of).

Reproducibility is by construction:
  * query splits         -> overlap_retrain.SPLIT_SEED (locked train/val/test)
  * offset MLP init      -> overlap_retrain.INIT_SEED
  * encoder fine-tune    -> l1_finetune_encoder.FT_SEED (seeded shuffle + init)
  * frozen encodings     -> model-deterministic (BGE / gte-Qwen2, no training)

The benchmark ENFORCES the "proper UKB" property: for a given dataset every encoder must
report the SAME npart and n_test (same frozen substrate + same locked split); a mismatch is
flagged loudly rather than silently averaged.

Outputs (all in results/):  L1_benchmark.md  L1_benchmark.csv  L1_benchmark.json
Run:  python experiments.py run l1-benchmark            # aggregate what exists
      python experiments.py run l1-benchmark -- --run-missing   # + run absent cells (local CPU)
"""
import os
import csv
import json
import logging
import argparse
import datetime

log = logging.getLogger("experiments.l1_benchmark")

DATASETS = ["musique_clean", "2wiki_clean", "squad_clean", "metaqa", "hotpotqa_clean"]
# (column label, subdir or None for the default MiniLM index, kind)
ENCODERS = [
    ("MiniLM-L6",       None,         "frozen"),
    ("BGE-large",       "bge_large",  "frozen"),
    ("BGE-large-ft",    "ft_bge",     "finetuned"),
    ("gte-Qwen2-1.5B",  "gte_qwen",   "frozen"),
]
METRICS = ["eqrrf6+bestof", "eqrrf6+bestof3"]      # primary ranking + best-of-3 variant
PRIMARY = "eqrrf6+bestof"
K = "20"


def _result_path(dataset, subdir):
    tag = f"_{subdir}" if subdir else ""
    return os.path.join("data", "ukb_storage", dataset, "results", "L1_select",
                        f"rerank100_{dataset}{tag}.json")


def _read_cell(dataset, subdir):
    p = _result_path(dataset, subdir)
    if not os.path.exists(p):
        return None
    r = json.load(open(p, encoding="utf-8"))
    res = r.get("results", {})
    out = {"n_test": r.get("n_test"), "npart": r.get("npart"), "encoder": r.get("encoder"),
           "oracle@20": res.get("oracle", {}).get(K)}
    for m in METRICS:
        out[f"{m}@20"] = res.get(m, {}).get(K)
    return out


def _seeds():
    from src.experiments import overlap_retrain as ot
    from src.experiments import l1_finetune_encoder as ft
    return {"SPLIT_SEED": getattr(ot, "SPLIT_SEED", None),
            "INIT_SEED": getattr(ot, "INIT_SEED", None),
            "FT_SEED": getattr(ft, "FT_SEED", None)}


def _reproduce(dataset, subdir, kind):
    """Exact command(s) to regenerate this cell from the frozen substrate."""
    acct = "<acct>"
    if subdir is None:
        return [f"python experiments.py run l1-rerank100 --backend modal --cpu --account {acct} "
                f"-- --datasets {dataset}"]
    if kind == "finetuned":
        enc = (f"python experiments.py run l1-finetune-encoder --backend modal --gpu --account {acct} "
               f"-- --datasets {dataset} --base BAAI/bge-large-en-v1.5 --subdir {subdir} "
               f"--epochs 1 --batch 16 --max-seq 256")
    else:
        model = {"bge_large": "BAAI/bge-large-en-v1.5",
                 "gte_qwen": "Alibaba-NLP/gte-Qwen2-1.5B-instruct"}.get(subdir, "<model>")
        enc = (f"python experiments.py run reencode-ukb --backend modal --gpu --account {acct} "
               f"-- --datasets {dataset} --model {model} --subdir {subdir} --batch 16")
    rr = (f"python experiments.py run l1-rerank100 --backend modal --cpu --account {acct} "
          f"-- --datasets {dataset} --subdir {subdir}")
    return [enc, rr]


def run(datasets=None, run_missing=False, out_dir="results"):
    datasets = datasets or DATASETS
    os.makedirs(out_dir, exist_ok=True)
    grid, missing, warnings = {}, [], []

    for d in datasets:
        grid[d] = {}
        npseen, ntseen = {}, {}
        for label, subdir, kind in ENCODERS:
            cell = _read_cell(d, subdir)
            if cell is None:
                if run_missing:
                    log.info(f"[{d}/{label}] missing -> running rerank locally")
                    from src.experiments import l1_rerank100
                    try:
                        l1_rerank100.run(d, subdir=subdir)
                        cell = _read_cell(d, subdir)
                    except Exception as e:
                        log.warning(f"[{d}/{label}] rerank failed: {e}")
                if cell is None:
                    missing.append((d, label, subdir, kind))
                    grid[d][label] = None
                    continue
            grid[d][label] = cell
            if cell["npart"] is not None:
                npseen.setdefault(cell["npart"], []).append(label)
            if cell["n_test"] is not None:
                ntseen.setdefault(cell["n_test"], []).append(label)
            cov, orc = cell.get(f"{PRIMARY}@20"), cell.get("oracle@20")   # coverage can't beat its own ceiling
            if isinstance(cov, (int, float)) and isinstance(orc, (int, float)) and cov > orc + 0.01:
                warnings.append(f"{d}/{label}: coverage {cov} > oracle {orc} — STALE/buggy oracle, re-run this cell")
        # integrity: every encoder for a dataset must share npart + n_test (same UKB + split)
        if len(npseen) > 1:
            warnings.append(f"{d}: npart MISMATCH across encoders {npseen} — not the same substrate!")
        if len(ntseen) > 1:
            warnings.append(f"{d}: n_test MISMATCH across encoders {ntseen} — not the same eval split!")

    seeds = _seeds()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _write_json(out_dir, grid, missing, warnings, seeds, stamp, datasets)
    _write_csv(out_dir, grid, datasets)
    _write_md(out_dir, grid, missing, warnings, seeds, stamp, datasets)
    log.info(f"benchmark written to {out_dir}/L1_benchmark.{{md,csv,json}}  "
             f"({sum(1 for d in grid for l in grid[d] if grid[d][l])}/{len(datasets)*len(ENCODERS)} cells, "
             f"{len(missing)} missing, {len(warnings)} integrity warnings)")
    for w in warnings:
        log.warning("INTEGRITY: " + w)
    return grid


def _write_json(out_dir, grid, missing, warnings, seeds, stamp, datasets):
    payload = {"generated": stamp, "primary_metric": f"{PRIMARY}@{K}", "seeds": seeds,
               "datasets": datasets, "encoders": [e[0] for e in ENCODERS],
               "grid": grid, "missing": [f"{d}/{l}" for d, l, *_ in missing],
               "integrity_warnings": warnings}
    json.dump(payload, open(os.path.join(out_dir, "L1_benchmark.json"), "w", encoding="utf-8"), indent=2)


def _write_csv(out_dir, grid, datasets):
    with open(os.path.join(out_dir, "L1_benchmark.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "encoder", "kind", f"{PRIMARY}@20", "bestof3@20", "oracle@20", "n_test", "npart"])
        for d in datasets:
            for label, subdir, kind in ENCODERS:
                c = grid[d].get(label)
                if not c:
                    w.writerow([d, label, kind, "", "", "", "", ""]); continue
                w.writerow([d, label, kind, c.get(f"{PRIMARY}@20"), c.get("eqrrf6+bestof3@20"),
                            c.get("oracle@20"), c.get("n_test"), c.get("npart")])


def _fmt(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


def _write_md(out_dir, grid, missing, warnings, seeds, stamp, datasets):
    L = [f"# L1 Partition-Routing Benchmark", "",
         f"_Generated {stamp}. Primary metric: **{PRIMARY}@{K}** (L1 partition FullCov@20). "
         f"Oracle@20 is the greedy min-cover ceiling. Higher is better._", ""]
    L.append(f"Seeds — splits `{seeds['SPLIT_SEED']}`, offset-init `{seeds['INIT_SEED']}`, "
             f"fine-tune `{seeds['FT_SEED']}`.")
    L.append("")
    # main table: rows=datasets, cols=encoders (primary metric)
    heads = ["dataset"] + [e[0] for e in ENCODERS] + ["oracle"]
    L.append("| " + " | ".join(heads) + " |")
    L.append("|" + "|".join(["---"] * len(heads)) + "|")
    for d in datasets:
        cells = []
        orc = None
        for label, subdir, kind in ENCODERS:
            c = grid[d].get(label)
            if not c:
                cells.append("—"); continue
            v = c.get(f"{PRIMARY}@20")
            mark = " ✓" if isinstance(v, (int, float)) and v >= 95 else ""
            cells.append(f"{_fmt(v)}{mark}")
            orc = orc if orc is not None else c.get("oracle@20")
        L.append(f"| {d} | " + " | ".join(cells) + f" | {_fmt(orc)} |")
    L.append("")
    if warnings:
        L.append("## ⚠️ Integrity warnings")
        for w in warnings:
            L.append(f"- {w}")
        L.append("")
    else:
        L.append("_Integrity: every encoder shares npart + n_test per dataset (same frozen UKB + split). ✓_")
        L.append("")
    if missing:
        L.append("## Missing cells (run to complete)")
        for d, label, subdir, kind in missing:
            L.append(f"- **{d} / {label}**:")
            for cmd in _reproduce(d, subdir, kind):
                L.append(f"  - `{cmd}`")
        L.append("")
    open(os.path.join(out_dir, "L1_benchmark.md"), "w", encoding="utf-8").write("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description="Aggregate the L1 partition-routing benchmark into one table.")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--run-missing", action="store_true", help="run absent cells locally (CPU) before aggregating")
    p.add_argument("--out-dir", default="results")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, run_missing=a.run_missing, out_dir=a.out_dir)


if __name__ == "__main__":
    main()
