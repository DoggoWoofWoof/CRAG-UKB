"""
Coverage-loss Level-1 ablation body (backend-agnostic).
=======================================================
Trains the Jigsaw-style coverage loss over a lambda sweep vs the frozen KL+HNM
baseline, with FullCov@K / weakest_positive_rank and a paired exact McNemar test.
Pure compute on the local filesystem — the experiments runner provisions the
Modal/Lightning machine and calls main(). (Formerly run_coverage_eval.py.)
"""
import os
import json
import csv
import math
import logging
import argparse
from collections import defaultdict

import numpy as np
import torch

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.alignment.mlp_encoder import TextPartitionMLP
from src.alignment.train_mlp import train as train_mlp
from src.evaluation.benchmark_partition_selection import (
    _get_split_queries, compute_multi_gt_metrics, COVERAGE_K_VALUES,
)

log = logging.getLogger("experiments.coverage")

# Frozen best Level-1 config (MLP + KL + HNM) per dataset. The coverage term is
# added on top without re-tuning tau/hn_k.
BEST_KL_CONFIG = {
    "squad":   {"tau": 0.1,  "hn_k": 18},
    "metaqa":  {"tau": 0.01, "hn_k": 0},
    "musique": {"tau": 0.05, "hn_k": 33},
    "2wiki":   {"tau": 0.07, "hn_k": 149},
    # audited "_clean" substrates are the same datasets -> reuse each one's KL baseline hyperparameters
    "squad_clean":   {"tau": 0.1,  "hn_k": 18},
    "musique_clean": {"tau": 0.05, "hn_k": 33},
    "2wiki_clean":   {"tau": 0.07, "hn_k": 149},
}
DEFAULT_LAMBDAS = [0.1, 0.25, 0.5, 1.0]


def _ckpt_dir(dataset):
    return os.path.join("checkpoints", dataset, "hnm_ablation")


def _lim_suffix(limit):
    return f"_lim{limit}" if limit and limit > 0 else ""


def _kl_ckpt_path(dataset, tau, hn_k, limit=0, tag=""):
    tg = f"_{tag}" if tag else ""
    return os.path.join(_ckpt_dir(dataset),
                        f"alignment_mlp_kl_div_tau_{tau:g}_hnm_{hn_k}{tg}{_lim_suffix(limit)}.pth")


def _coverage_ckpt_path(dataset, loss, tau, hn_k, lam, limit=0):
    return os.path.join(_ckpt_dir(dataset),
                        f"alignment_mlp_{loss}_tau_{tau:g}_hnm_{hn_k}_lam_{lam:g}{_lim_suffix(limit)}.pth")


def _load_ckpt(ckpt_path, device):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=device)


def _build_mlp(ckpt, device, key="model_state_dict"):
    """Build a TextPartitionMLP from a checkpoint dict, selecting the best-val
    (`model_state_dict`) or final-epoch (`final_state_dict`) weights."""
    state_dict = ckpt.get(key) if isinstance(ckpt, dict) else None
    if state_dict is None:
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    in_dim = int(ckpt.get("input_dim", state_dict["net.0.weight"].shape[1]))
    hidden = int(ckpt.get("hidden_dim", state_dict["net.0.weight"].shape[0]))
    out_dim = int(ckpt.get("output_dim", state_dict["net.3.weight"].shape[0]))
    model = TextPartitionMLP(input_dim=in_dim, hidden_dim=hidden, output_dim=out_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _eval_per_query(engine, model, queries, embs, device):
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


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value in log-space (overflow/underflow safe)."""
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
    return float(min(1.0, 2.0 * math.exp(log_tail)))


def run_dataset(dataset, lambdas, loss, epochs, limit, device):
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

    configs = {}
    # Frozen production baseline (val-loss selected, old run) — kept as a REFERENCE
    # row so we can see the selection effect, but NOT the comparison baseline.
    frozen_path = _kl_ckpt_path(dataset, tau, hn_k, limit)
    if os.path.exists(frozen_path):
        configs["kl_div_frozen"] = frozen_path
    # Fair baseline: a FRESH KL trained with the exact same code/selection as the
    # coverage models (best+final+FC), so the ablation is controlled (final-vs-final,
    # differing ONLY in the coverage term). Non-clobbering `_covbase` tag.
    fair_path = _kl_ckpt_path(dataset, tau, hn_k, limit, tag="covbase")
    if not os.path.exists(fair_path):
        log.info(f"Training fair KL baseline -> {fair_path}")
        train_mlp(dataset_name=dataset, loss_type="kl_div", tau=tau, hn_k=hn_k,
                  epochs=epochs, limit=limit, tag="covbase")
    configs["kl_div_baseline"] = fair_path

    for lam in lambdas:
        cov_path = _coverage_ckpt_path(dataset, loss, tau, hn_k, lam, limit)
        if not os.path.exists(cov_path):
            log.info(f"Training {loss} lambda={lam:g} -> {cov_path}")
            train_mlp(dataset_name=dataset, loss_type=loss, tau=tau, hn_k=hn_k,
                      epochs=epochs, lambda_cov=lam, limit=limit)
        else:
            log.info(f"Found {cov_path}; skipping training.")
        configs[f"{loss}_lam_{lam:g}"] = cov_path

    out_dir = os.path.join("results", "coverage_ablation")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"comparison_{dataset}_coverage.json")
    csv_path = os.path.join(out_dir, f"{dataset}_coverage_ablation_results.csv")

    # Resume / merge: load any prior results for this dataset so a crash or a
    # re-run never discards completed configs — we add missing ones and keep
    # existing (never truncate-and-rewrite from empty).
    all_results = {}
    if os.path.exists(json_path):
        try:
            all_results = json.load(open(json_path, encoding="utf-8")).get("results", {}) or {}
            log.info(f"Resuming {dataset}: {len(all_results)} config(s) already on disk.")
        except Exception as e:
            log.warning(f"Could not read existing {json_path} ({e}); starting fresh.")

    def _save(sig):
        payload = {"dataset": dataset, "base_config": {"loss": loss, "tau": tau, "hn_k": hn_k},
                   "results": all_results, "significance": sig}
        tmp = json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, json_path)   # atomic — never leaves a half-written json
        rows = []
        for lbl, split_data in all_results.items():
            for split_name, metrics in split_data.items():
                r = {"dataset": dataset, "method": lbl, "split": split_name}
                r.update({k: v for k, v in metrics.items() if k != "method"})
                rows.append(r)
        if rows:
            keys = []
            for r in rows:
                for kk in r.keys():
                    if kk not in keys:
                        keys.append(kk)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

    per_query_test = {}
    for label, path in configs.items():
        if not os.path.exists(path):
            log.warning(f"Checkpoint {path} missing after training; skipping {label}.")
            continue
        ckpt = _load_ckpt(path, device)
        # Benchmark both the best-by-val checkpoint and (if present) the
        # final-epoch weights, as distinct method rows.
        variants = [(label, "model_state_dict")]
        if isinstance(ckpt, dict) and "final_state_dict" in ckpt:
            variants.append((f"{label}__final", "final_state_dict"))
        for vlabel, key in variants:
            if all_results.get(vlabel, {}).get("test"):
                log.info(f"  [{vlabel}] already on disk; skipping (resume).")
                continue
            model = _build_mlp(ckpt, device, key=key)
            all_results[vlabel] = {}
            for s, queries in splits.items():
                if not queries:
                    continue
                pq = _eval_per_query(engine, model, queries, split_embs[s], device)
                all_results[vlabel][s] = _aggregate(pq)
                all_results[vlabel][s]["method"] = vlabel
                if s == "test":
                    per_query_test[vlabel] = pq
            t = all_results[vlabel].get("test", {})
            log.info(f"  [{vlabel}] test FCov@20={t.get('full_coverage@20', 0):.2f}% "
                     f"FCov@50={t.get('full_coverage@50', 0):.2f}% R@1={t.get('recall@1', 0):.2f}% "
                     f"weakest_rank(med)={t.get('median_weakest_positive_rank', 0)}")
            _save({})   # incremental crash-safe save after each config variant

    # Significance: best baseline VARIANT (frozen/fresh, best-or-final) vs best
    # coverage VARIANT (best-or-final) — a fair best-config-vs-best-config McNemar.
    significance = {}
    base_keys = [k for k in per_query_test if k.startswith("kl_div")]
    cov_keys = [k for k in per_query_test if k.startswith(loss)]
    if base_keys and cov_keys:
        def _fc(k):
            return float(np.mean([m["full_coverage@20"] for m in per_query_test[k]]))
        base_best = max(base_keys, key=_fc)
        cov_best = max(cov_keys, key=_fc)
        bpq, cpq = per_query_test[base_best], per_query_test[cov_best]
        n = min(len(bpq), len(cpq))
        bcov = [int(bpq[i]["full_coverage@20"] >= 1.0) for i in range(n)]
        ccov = [int(cpq[i]["full_coverage@20"] >= 1.0) for i in range(n)]
        b = sum(1 for i in range(n) if bcov[i] == 1 and ccov[i] == 0)
        c = sum(1 for i in range(n) if bcov[i] == 0 and ccov[i] == 1)
        p_value = mcnemar_exact(b, c)
        significance = {
            "best_baseline": base_best, "best_coverage": cov_best,
            "baseline_FCov@20": round(_fc(base_best) * 100, 2),
            "coverage_FCov@20": round(_fc(cov_best) * 100, 2),
            "discordant_baseline_only(b)": b, "discordant_coverage_only(c)": c,
            "mcnemar_p_value": p_value, "n_test": n,
        }
        log.info(f"  McNemar {cov_best} vs {base_best}: "
                 f"{significance['baseline_FCov@20']}% -> {significance['coverage_FCov@20']}% "
                 f"(b={b}, c={c}, p={p_value:.3e})")

    _save(significance)
    log.info(f"Saved results/coverage_ablation/{dataset}_coverage_ablation_results.csv (+ json)")
    return {"dataset": dataset, "results": all_results, "significance": significance}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Coverage-loss Level-1 ablation.")
    parser.add_argument("--datasets", nargs="+", default=["2wiki", "musique"],
                        choices=list(BEST_KL_CONFIG.keys()))
    parser.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--loss", type=str, default="coverage_kl",
                        choices=["coverage_kl", "coverage_infonce", "coverage"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | datasets={args.datasets} | lambdas={args.lambdas} | loss={args.loss}")
    for ds in args.datasets:
        run_dataset(ds, args.lambdas, args.loss, args.epochs, args.limit, device)
    log.info("Coverage ablation complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
