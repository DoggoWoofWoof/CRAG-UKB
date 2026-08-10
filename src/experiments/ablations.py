"""
Level-1 loss / temperature / HNM ablation bodies (backend-agnostic).
====================================================================
Reproduces the three Paper-1 ablations by reusing the tested primitives
(train_mlp.train + benchmark_partition_selection.benchmark) instead of the old
Modal-entangled run_loss_eval.py / run_temp_eval.py / run_hnm_eval.py. All three
train into checkpoints/{ds}/hnm_ablation/ with the canonical
alignment_mlp_{loss}_tau_{tau}_hnm_{hn_k}.pth naming and write comparison
JSON+CSV under results/{loss,temp,hnm}_ablation/.
"""
import os
import csv
import json
import logging
import argparse

import torch

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.alignment.mlp_encoder import TextPartitionMLP
from src.alignment.train_mlp import train as train_mlp
from src.evaluation.benchmark_partition_selection import _get_split_queries, benchmark

log = logging.getLogger("experiments.ablations")

TAU_CONFIG = {"metaqa": 0.01, "2wiki": 0.07, "musique": 0.05, "squad": 0.1}
TEMP_SWEEP = [0.01, 0.05, 0.07, 0.1, 0.2, 0.5]
LOSS_SWEEP = ["info_nce_single", "info_nce_multi", "kl_div", "bce"]


def get_hn_sweep(n_partitions):
    """Geometric quartile HNM sweep bounded by the partition count."""
    max_hn = max(1, n_partitions - 1)
    if max_hn <= 3:
        return [0, max_hn]
    sweep = {0, max(1, int(max_hn * 0.25)), max(1, int(max_hn * 0.5)),
             max(1, int(max_hn * 0.75)), max_hn}
    return sorted(sweep)


def _ckpt_path(ds, loss, tau, hn_k, limit=0):
    lim = f"_lim{limit}" if limit and limit > 0 else ""
    return f"checkpoints/{ds}/hnm_ablation/alignment_mlp_{loss}_tau_{tau:g}_hnm_{hn_k}{lim}.pth"


def _load_mlp(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    in_dim = int(ckpt.get("input_dim", sd["net.0.weight"].shape[1]))
    hidden = int(ckpt.get("hidden_dim", sd["net.0.weight"].shape[0]))
    out_dim = int(ckpt.get("output_dim", sd["net.3.weight"].shape[0]))
    m = TextPartitionMLP(input_dim=in_dim, hidden_dim=hidden, output_dim=out_dim).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m


def _run_ablation(out_name, configs_for, datasets, epochs, limit, device):
    """configs_for(ds, n_parts) -> list of (label, loss, tau, hn_k)."""
    out_dir = os.path.join("results", out_name)
    os.makedirs(out_dir, exist_ok=True)

    for ds in datasets:
        if ds not in TAU_CONFIG:
            log.warning(f"No tau config for {ds}; skipping.")
            continue
        log.info(f"===== {out_name.upper()}: {ds.upper()} =====")
        engine = CoreEngine(source=ds)
        encoder = DenseEncoder()
        splits = _get_split_queries(engine, dataset=ds)
        if not any(splits.values()):
            continue
        if limit:
            splits = {s: q[:limit] for s, q in splits.items()}
        split_embs = {s: encoder.encode([q.content for q, _ in qs]) for s, qs in splits.items() if qs}
        n_parts = len(set(int(p) for p in engine.partition_map.values()))

        all_results = {}
        for label, loss, tau, hn_k in configs_for(ds, n_parts):
            ckpt = _ckpt_path(ds, loss, tau, hn_k, limit)
            if not os.path.exists(ckpt):
                log.info(f"Training {label} -> {ckpt}")
                train_mlp(dataset_name=ds, loss_type=loss, tau=tau, hn_k=hn_k, epochs=epochs, limit=limit)
            if not os.path.exists(ckpt):
                log.warning(f"Checkpoint {ckpt} missing after training; skipping {label}.")
                continue
            model = _load_mlp(ckpt, device)
            all_results[label] = {}
            for s, queries in splits.items():
                if not queries:
                    continue
                res = benchmark(engine, encoder, "mlp", queries, model=model,
                                precomputed_embs=split_embs.get(s))
                res["method"] = label
                all_results[label][s] = res
            t = all_results[label].get("test", {})
            log.info(f"  [{label}] test R@1={t.get('recall@1', 0):.2f}% "
                     f"FCov@20={t.get('full_coverage@20', 0):.2f}% MRR={t.get('mrr', 0):.2f}%")

        with open(os.path.join(out_dir, f"comparison_{ds}_{out_name.split('_')[0]}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        rows = []
        for label, split_data in all_results.items():
            for s, metrics in split_data.items():
                row = {"dataset": ds, "method": label, "split": s}
                row.update({k: v for k, v in metrics.items() if k != "method"})
                rows.append(row)
        if rows:
            keys = []
            for r in rows:
                for kk in r:
                    if kk not in keys:
                        keys.append(kk)
            with open(os.path.join(out_dir, f"{ds}_{out_name}_results.csv"),
                      "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
        log.info(f"Saved results/{out_name}/{ds}_{out_name}_results.csv (+ json)")


def run_loss(datasets, epochs=100, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _run_ablation("loss_ablation",
                  lambda ds, n: [(f"mlp_{ls}", ls, TAU_CONFIG[ds], 0) for ls in LOSS_SWEEP],
                  datasets, epochs, limit, device)


def run_temp(datasets, epochs=100, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _run_ablation("temp_ablation",
                  lambda ds, n: [(f"mlp_kl_div_tau_{t:g}", "kl_div", t, 0) for t in TEMP_SWEEP]
                  + [(f"mlp_info_nce_multi_tau_{t:g}", "info_nce_multi", t, 0) for t in TEMP_SWEEP],
                  datasets, epochs, limit, device)


def run_hnm(datasets, epochs=100, limit=0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def cfgs(ds, n_parts):
        out = []
        for loss in ("info_nce_multi", "kl_div"):
            for hn_k in get_hn_sweep(n_parts):
                out.append((f"mlp_{loss}_hnm_{hn_k}", loss, TAU_CONFIG[ds], hn_k))
        return out

    _run_ablation("hnm_ablation", cfgs, datasets, epochs, limit, device)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Level-1 loss/temp/HNM ablation.")
    parser.add_argument("which", choices=["loss", "temp", "hnm"])
    parser.add_argument("--datasets", nargs="+", default=["squad", "metaqa", "musique", "2wiki"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    {"loss": run_loss, "temp": run_temp, "hnm": run_hnm}[args.which](
        args.datasets, epochs=args.epochs, limit=args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
