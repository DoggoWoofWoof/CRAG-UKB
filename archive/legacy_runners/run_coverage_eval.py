"""
Coverage-loss Level-1 ablation runner (local).
===============================================
Trains and evaluates the Jigsaw-style coverage-aware Level-1 routing loss
against the frozen best KL+HNM baseline, sweeping the coverage weight lambda.

For each dataset it:
  1. Keeps the selected best (tau, hn_k) FIXED (the frozen KL-HNM config).
  2. Ensures the baseline `kl_div` checkpoint exists (it already does on disk).
  3. Trains `coverage_kl` (default) for each lambda in the sweep (skips if the
     checkpoint already exists).
  4. Evaluates every model with the full metric suite — including the new
     FullCov@{1,3,5,10,20,50} sweep and weakest_positive_rank — on train/val/test.
  5. Runs a per-query paired McNemar test on FullCov@20 (test split) between the
     best coverage model and the KL baseline, so the improvement is significance-
     backed, not a vibes-level aggregate delta (Jigsaw methodology §4).

Outputs:
  results/coverage_ablation/{dataset}_coverage_ablation_results.csv
  results/coverage_ablation/comparison_{dataset}_coverage.json

This runner is intentionally Modal-free so it runs on any machine with the
data/ukb_storage indexes present. Use --datasets, --lambdas, --loss, --epochs,
and --limit to control scope. 2Wiki and MuSiQue are the recommended proof
datasets (SQuAD's 19 partitions make top-K routing degenerate).

Example:
    python run_coverage_eval.py --datasets 2wiki musique --epochs 100
    python run_coverage_eval.py --datasets 2wiki --lambdas 0.25 0.5 --limit 2000
"""
import os
import sys
import json
import csv
import math
import logging
import argparse
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.alignment.mlp_encoder import TextPartitionMLP
from src.alignment.train_mlp import train as train_mlp
from src.evaluation.benchmark_partition_selection import (
    _get_split_queries,
    compute_multi_gt_metrics,
    COVERAGE_K_VALUES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("coverage_eval")

# Frozen best Level-1 config (MLP + KL + HNM) per dataset. The coverage term is
# added on top of this base without re-tuning tau/hn_k (per the plan).
BEST_KL_CONFIG = {
    "squad":   {"tau": 0.1,  "hn_k": 18},
    "metaqa":  {"tau": 0.01, "hn_k": 0},
    "musique": {"tau": 0.05, "hn_k": 33},
    "2wiki":   {"tau": 0.07, "hn_k": 149},
}

DEFAULT_LAMBDAS = [0.1, 0.25, 0.5, 1.0]


def _ckpt_dir(dataset: str) -> str:
    return os.path.join("checkpoints", dataset, "hnm_ablation")


def _lim_suffix(limit: int) -> str:
    return f"_lim{limit}" if limit and limit > 0 else ""


def _kl_ckpt_path(dataset: str, tau: float, hn_k: int, limit: int = 0) -> str:
    return os.path.join(_ckpt_dir(dataset),
                        f"alignment_mlp_kl_div_tau_{tau:g}_hnm_{hn_k}{_lim_suffix(limit)}.pth")


def _coverage_ckpt_path(dataset: str, loss: str, tau: float, hn_k: int, lam: float, limit: int = 0) -> str:
    return os.path.join(_ckpt_dir(dataset),
                        f"alignment_mlp_{loss}_tau_{tau:g}_hnm_{hn_k}_lam_{lam:g}{_lim_suffix(limit)}.pth")


def _load_mlp(ckpt_path: str, device) -> TextPartitionMLP:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    in_dim = int(ckpt.get("input_dim", state_dict["net.0.weight"].shape[1]))
    hidden = int(ckpt.get("hidden_dim", state_dict["net.0.weight"].shape[0]))
    out_dim = int(ckpt.get("output_dim", state_dict["net.3.weight"].shape[0]))
    model = TextPartitionMLP(input_dim=in_dim, hidden_dim=hidden, output_dim=out_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _eval_per_query(engine, model, queries, embs, device):
    """Per-query metrics for the MLP routing path over the FULL partition ranking."""
    import faiss
    n_parts = len(set(int(p) for p in engine.partition_map.values()))
    retr_k = max(n_parts, max(COVERAGE_K_VALUES))
    per_query = []
    for i, (q_node, gt_pids) in enumerate(queries):
        qv = np.asarray(embs[i:i + 1], dtype="float32").copy()
        faiss.normalize_L2(qv)
        with torch.no_grad():
            projected = model(torch.tensor(qv, dtype=torch.float32).to(device)).cpu().numpy()
        results = engine.search_centroids(projected, k=retr_k)
        retrieved = [pid for pid, _ in results]
        per_query.append(compute_multi_gt_metrics(retrieved, gt_pids, num_partitions=n_parts))
    return per_query


def _aggregate(per_query):
    """Aggregate per-query metric dicts the same way benchmark() does."""
    agg = defaultdict(list)
    for m in per_query:
        for k, v in m.items():
            agg[k].append(v)
    summary = {}
    for key, vals in agg.items():
        if key == "num_gt":
            summary["avg_gt_partitions"] = round(float(np.mean(vals)), 2)
            summary["median_gt_partitions"] = round(float(np.median(vals)), 1)
        elif key in ("first_hit_pos", "weakest_positive_rank"):
            summary[f"avg_{key}"] = round(float(np.mean(vals)), 2)
            summary[f"median_{key}"] = round(float(np.median(vals)), 1)
        else:
            summary[key] = round(float(np.mean(vals)) * 100, 2)
    summary["total_queries"] = len(per_query)
    return summary


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for paired binary outcomes.

    b = #queries the baseline covered but the new model did not.
    c = #queries the new model covered but the baseline did not.

    Computed in log-space (lgamma + logsumexp) so it never overflows on large
    binomial coefficients nor underflows 0.5**n to 0.0 — the naive
    `2 * sum(comb(n,i)) * 0.5**n` crashes for n >~ 1026.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    log_half_n = n * math.log(0.5)
    log_terms = [
        (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)) + log_half_n
        for i in range(0, k + 1)
    ]
    m = max(log_terms)
    log_tail = m + math.log(sum(math.exp(t - m) for t in log_terms))
    p = 2.0 * math.exp(log_tail)
    return float(min(1.0, p))


def run_dataset(dataset: str, lambdas, loss: str, epochs: int, limit: int, device):
    if dataset not in BEST_KL_CONFIG:
        log.warning(f"No frozen KL config for {dataset}; skipping.")
        return None
    tau = BEST_KL_CONFIG[dataset]["tau"]
    hn_k = BEST_KL_CONFIG[dataset]["hn_k"]
    log.info(f"===== COVERAGE ABLATION: {dataset.upper()} (base tau={tau:g}, hn_k={hn_k}) =====")

    engine = CoreEngine(source=dataset)
    encoder = DenseEncoder()
    splits = _get_split_queries(engine, dataset=dataset)
    if not any(splits.values()):
        log.warning(f"No queries for {dataset}; skipping.")
        return None

    if limit:
        splits = {s: q[:limit] for s, q in splits.items()}

    split_embs = {}
    for s, queries in splits.items():
        if queries:
            split_embs[s] = encoder.encode([q.content for q, _ in queries])

    # ── 1. Baseline: frozen KL + HNM ──
    configs = {}  # label -> checkpoint path
    kl_path = _kl_ckpt_path(dataset, tau, hn_k, limit)
    if not os.path.exists(kl_path):
        log.info(f"Baseline KL checkpoint missing ({kl_path}); training it.")
        train_mlp(dataset_name=dataset, loss_type="kl_div", tau=tau, hn_k=hn_k,
                  epochs=epochs, limit=limit)
    configs["kl_div_baseline"] = kl_path

    # ── 2. Coverage sweep ──
    for lam in lambdas:
        cov_path = _coverage_ckpt_path(dataset, loss, tau, hn_k, lam, limit)
        if not os.path.exists(cov_path):
            log.info(f"Training {loss} lambda={lam:g} -> {cov_path}")
            train_mlp(dataset_name=dataset, loss_type=loss, tau=tau, hn_k=hn_k,
                      epochs=epochs, lambda_cov=lam, limit=limit)
        else:
            log.info(f"Found {cov_path}; skipping training.")
        configs[f"{loss}_lam_{lam:g}"] = cov_path

    # ── 3. Evaluate all configs on all splits ──
    all_results = {}
    per_query_test = {}  # label -> per-query metric dicts on test
    for label, path in configs.items():
        if not os.path.exists(path):
            log.warning(f"Checkpoint {path} missing after training; skipping {label}.")
            continue
        model = _load_mlp(path, device)
        all_results[label] = {}
        for s, queries in splits.items():
            if not queries:
                continue
            pq = _eval_per_query(engine, model, queries, split_embs[s], device)
            all_results[label][s] = _aggregate(pq)
            all_results[label][s]["method"] = label
            if s == "test":
                per_query_test[label] = pq
        t = all_results[label].get("test", {})
        log.info(
            f"  [{label}] test FCov@20={t.get('full_coverage@20', 0):.2f}% "
            f"FCov@50={t.get('full_coverage@50', 0):.2f}% "
            f"R@1={t.get('recall@1', 0):.2f}% "
            f"weakest_rank(med)={t.get('median_weakest_positive_rank', 0)}"
        )

    # ── 4. Significance: best coverage vs KL baseline on test FullCov@20 ──
    significance = {}
    if "kl_div_baseline" in per_query_test:
        base_pq = per_query_test["kl_div_baseline"]
        base_cov = [int(m["full_coverage@20"] >= 1.0) for m in base_pq]
        best_label, best_fcov = None, -1.0
        for label, pq in per_query_test.items():
            if label == "kl_div_baseline":
                continue
            fcov = float(np.mean([m["full_coverage@20"] for m in pq]))
            if fcov > best_fcov:
                best_fcov, best_label = fcov, label
        if best_label is not None:
            new_pq = per_query_test[best_label]
            n = min(len(base_cov), len(new_pq))
            new_cov = [int(new_pq[i]["full_coverage@20"] >= 1.0) for i in range(n)]
            b = sum(1 for i in range(n) if base_cov[i] == 1 and new_cov[i] == 0)
            c = sum(1 for i in range(n) if base_cov[i] == 0 and new_cov[i] == 1)
            p_value = mcnemar_exact(b, c)
            significance = {
                "baseline": "kl_div_baseline",
                "best_coverage": best_label,
                "test_full_coverage@20_baseline": round(float(np.mean(base_cov)) * 100, 2),
                "test_full_coverage@20_coverage": round(best_fcov * 100, 2),
                "discordant_baseline_only(b)": b,
                "discordant_coverage_only(c)": c,
                "mcnemar_p_value": p_value,
                "n_test": n,
            }
            log.info(
                f"  McNemar {best_label} vs kl_div_baseline: "
                f"FCov@20 {significance['test_full_coverage@20_baseline']}% -> "
                f"{significance['test_full_coverage@20_coverage']}% "
                f"(b={b}, c={c}, p={p_value:.3e})"
            )

    # ── 5. Persist ──
    out_dir = os.path.join("results", "coverage_ablation")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"comparison_{dataset}_coverage.json")
    payload = {
        "dataset": dataset,
        "base_config": {"loss": loss, "tau": tau, "hn_k": hn_k},
        "results": all_results,
        "significance": significance,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Saved {json_path}")

    # Flatten to CSV (dynamic columns so new metrics flow through automatically).
    rows = []
    for label, split_data in all_results.items():
        for split_name, metrics in split_data.items():
            row = {"dataset": dataset, "method": label, "split": split_name}
            row.update({k: v for k, v in metrics.items() if k != "method"})
            rows.append(row)
    if rows:
        keys = []
        for r in rows:
            for kk in r.keys():
                if kk not in keys:
                    keys.append(kk)
        csv_path = os.path.join(out_dir, f"{dataset}_coverage_ablation_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Saved {csv_path}")

    return payload


def main():
    parser = argparse.ArgumentParser(description="Coverage-loss Level-1 ablation (local).")
    parser.add_argument("--datasets", nargs="+", default=["2wiki", "musique"],
                        choices=list(BEST_KL_CONFIG.keys()))
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--loss", type=str, default="coverage_kl",
                        choices=["coverage_kl", "coverage_infonce", "coverage"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap queries per split for a fast smoke run, END-TO-END "
                             "(caps BOTH training and evaluation; 0 = full corpus). "
                             "Limited runs write isolated _lim{N} checkpoints so they "
                             "never overwrite production models.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | datasets={args.datasets} | lambdas={args.lambdas} | loss={args.loss}")

    for ds in args.datasets:
        run_dataset(ds, args.lambdas, args.loss, args.epochs, args.limit, device)

    log.info("Coverage ablation complete.")


if __name__ == "__main__":
    main()
