"""Dataset-agnostic Level-1 document routing.

One shared router is trained across multiple independently indexed corpora. The
model never receives a dataset identifier: all learned parameters operate in
the common dense-encoder space. A pure dense position is included as an
immutable skip head, and a query-conditioned gate chooses between it and K
learned relational offsets. This lets the model preserve dense retrieval when
relational movement is unhelpful without selecting a per-dataset architecture.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.encoders import DenseEncoder
from src.core.engine import CoreEngine
from src.experiments.l1_optimize import (
    BUDGETS,
    ENCODER_NAME,
    MAX_CANDIDATES,
    _atomic_json_dump,
    _atomic_torch_save,
    _apply_spec,
    _candidate_batch,
    _cap,
    _coverage_objective,
    _document_manifest,
    _gate_balance,
    _head_diversity,
    _load_matching_checkpoint,
    _mean_metrics,
    _metrics,
    _partition_quota_order,
    _prepare_cached,
    _query_cache_valid,
    _selection_key,
    _split_manifest,
    _write_candidates,
)
from src.experiments.l1_mlp_transformer import _combined_logits
from src.experiments.overlap_retrain import (
    _centroids,
    _hard_membership,
    _reconstruct,
    _splits,
)
from src.experiments.stats import mcnemar_exact


log = logging.getLogger("experiments.l1_unified")


@dataclass(frozen=True)
class UnifiedConfig:
    relational_heads: int
    lambda_coverage: float
    lambda_diversity: float
    lambda_balance: float

    @property
    def total_heads(self) -> int:
        return self.relational_heads + 1

    @property
    def label(self) -> str:
        return (
            f"dense_plus_r{self.relational_heads}"
            f"_cov{self.lambda_coverage:g}"
            f"_div{self.lambda_diversity:g}"
            f"_bal{self.lambda_balance:g}"
        )


class DatasetAgnosticRouter(nn.Module):
    """Shared dense skip head, relational offsets, and query-conditioned gate."""

    def __init__(self, dimension: int, relational_heads: int, hidden: int = 512):
        super().__init__()
        self.dimension = dimension
        self.relational_heads = relational_heads
        self.trunk = nn.Sequential(nn.Linear(dimension, hidden), nn.ReLU())
        self.offsets = nn.Linear(hidden, relational_heads * dimension)
        self.gate = nn.Linear(hidden, relational_heads + 1)

    def forward(self, query: torch.Tensor, dense_seed: torch.Tensor):
        query = F.normalize(query, dim=-1)
        hidden = self.trunk(query)
        relational = F.normalize(
            dense_seed.unsqueeze(1)
            + self.offsets(hidden).view(
                -1,
                self.relational_heads,
                self.dimension,
            ),
            dim=-1,
        )
        positions = torch.cat((query.unsqueeze(1), relational), dim=1)
        weights = F.softmax(self.gate(hidden), dim=-1)
        return positions, weights


@dataclass
class DatasetContext:
    name: str
    fingerprint: str
    documents: np.ndarray
    document_tensor: torch.Tensor
    index: faiss.Index
    splits: Mapping[str, Sequence]
    prepared: Mapping[str, tuple]
    dense_orders: Mapping[str, List[List[int]]]
    document_ids: List[str]
    cache_dir: Path
    manifest: Mapping[str, object]
    centroids: np.ndarray
    docs_by_partition: List[List[int]]


def _data_fingerprint(engine, documents, splits, dataset):
    payload = {
        "version": 2,
        "dataset": dataset,
        "encoder": ENCODER_NAME,
        "document_manifest": _document_manifest(engine, documents),
        "splits": _split_manifest(splits),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return fingerprint, payload


def _prepare_dataset(
    dataset: str,
    *,
    limit: int,
    device: torch.device,
    encoder_holder: list,
) -> DatasetContext:
    engine = CoreEngine(source=dataset)
    documents = _reconstruct(engine.node_index).astype("float32")
    faiss.normalize_L2(documents)
    index = faiss.IndexFlatIP(documents.shape[1])
    index.add(documents)

    membership = _hard_membership(engine)
    splits = _splits(engine, membership)
    split_seed = {"train": 101, "val": 202, "test": 303}
    splits = {
        name: _cap(rows, limit, split_seed[name])
        for name, rows in splits.items()
    }
    if not all(splits.values()):
        raise RuntimeError(f"{dataset}: train/val/test split is incomplete")

    fingerprint, manifest = _data_fingerprint(
        engine,
        documents,
        splits,
        dataset,
    )
    cache_dir = (
        Path("data") / "ukb_storage" / dataset / "cache" / "L1" / fingerprint
    )
    cache_paths = {
        name: cache_dir / f"queries_{name}.npz" for name in splits
    }
    cache_ready = all(
        _query_cache_valid(
            cache_paths[name],
            splits[name],
            dense_seeds=10,
            dimension=documents.shape[1],
        )
        for name in splits
    )
    if not cache_ready and not encoder_holder:
        encoder_holder.append(DenseEncoder(ENCODER_NAME))
    encoder = encoder_holder[0] if encoder_holder else None
    prepared = {
        name: _prepare_cached(
            cache_paths[name],
            rows,
            encoder,
            index,
            engine.node_id_to_idx,
            dense_seeds=10,
        )
        for name, rows in splits.items()
    }
    dense_orders = {
        name: index.search(values[0], MAX_CANDIDATES)[1].tolist()
        for name, values in prepared.items()
    }
    num_partitions = max(int(pid) for pid in engine.partition_map.values()) + 1
    centroids, _ = _centroids(
        engine,
        documents,
        membership,
        num_partitions,
    )
    docs_by_partition = [[] for _ in range(num_partitions)]
    for document_index, node in enumerate(engine.nodes):
        partition = int(engine.partition_map[node.node_id])
        docs_by_partition[partition].append(document_index)
    context = DatasetContext(
        name=dataset,
        fingerprint=fingerprint,
        documents=documents,
        document_tensor=torch.tensor(documents, device=device),
        index=index,
        splits=splits,
        prepared=prepared,
        dense_orders=dense_orders,
        document_ids=[node.node_id for node in engine.nodes],
        cache_dir=cache_dir,
        manifest=manifest,
        centroids=centroids,
        docs_by_partition=docs_by_partition,
    )
    del engine
    gc.collect()
    return context


def _balanced_epoch_batches(
    eligible: Mapping[str, Sequence[int]],
    *,
    batch_size: int,
    seed: int,
    epoch: int,
) -> List[tuple[str, List[int]]]:
    """Oversample deterministically so every corpus contributes equal updates."""
    if not eligible or any(not values for values in eligible.values()):
        raise ValueError("Every training dataset needs at least one eligible query.")
    steps = max(math.ceil(len(values) / batch_size) for values in eligible.values())
    schedule = []
    for dataset, source_values in sorted(eligible.items()):
        rng = random.Random(f"{seed}:{epoch}:{dataset}")
        values = list(source_values)
        expanded = []
        while len(expanded) < steps * batch_size:
            cycle = values.copy()
            rng.shuffle(cycle)
            expanded.extend(cycle)
        for step in range(steps):
            start = step * batch_size
            schedule.append((dataset, expanded[start : start + batch_size]))
    random.Random(seed + epoch * 104729).shuffle(schedule)
    return schedule


def _retrieve(
    model,
    prepared,
    context: DatasetContext,
    device,
    k=MAX_CANDIDATES,
):
    queries, seeds, _, _ = prepared
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(queries), 256):
            query_batch = torch.tensor(queries[start : start + 256], device=device)
            seed_batch = context.document_tensor[
                [int(value) for value in seeds[start : start + 256, 0]]
            ]
            positions, weights = model(query_batch, seed_batch)
            flat = (
                positions.reshape(-1, positions.shape[-1])
                .cpu()
                .numpy()
                .astype("float32")
            )
            _, per_head = context.index.search(np.ascontiguousarray(flat), k)
            per_head = per_head.reshape(
                len(query_batch),
                positions.shape[1],
                k,
            )
            position_values = positions.cpu().numpy()
            weight_values = weights.cpu().numpy()
            for row in range(len(query_batch)):
                candidates = list(
                    dict.fromkeys(per_head[row].reshape(-1).tolist())
                )
                candidate_vectors = context.documents[candidates]
                scores = (
                    candidate_vectors @ position_values[row].T / 0.05
                    + np.log(weight_values[row] + 1e-9)
                )
                combined = np.logaddexp.reduce(scores, axis=1)
                ranked = np.argsort(-combined)[:k]
                output.append([int(candidates[index]) for index in ranked])
    return output


def _gate_statistics(model, prepared, context, device):
    queries, seeds, _, _ = prepared
    weights = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(queries), 512):
            query_batch = torch.tensor(queries[start : start + 512], device=device)
            seed_batch = context.document_tensor[
                [int(value) for value in seeds[start : start + 512, 0]]
            ]
            _, batch_weights = model(query_batch, seed_batch)
            weights.append(batch_weights.cpu().numpy())
    values = np.concatenate(weights, axis=0)
    entropy = -(values * np.log(values + 1e-9)).sum(axis=1)
    winners = np.argmax(values, axis=1)
    return {
        "mean_head_weights": [round(float(value), 6) for value in values.mean(0)],
        "dense_mean_weight": round(float(values[:, 0].mean()), 6),
        "dense_argmax_rate": round(float(np.mean(winners == 0)) * 100.0, 2),
        "mean_gate_entropy": round(float(entropy.mean()), 6),
    }


def _shared_partition_orders(model, prepared, context, device):
    """Route through corpus centroids using only the shared model positions."""
    queries, seeds, _, _ = prepared
    score_batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(queries), 256):
            query_batch = torch.tensor(queries[start : start + 256], device=device)
            seed_batch = context.document_tensor[
                [int(value) for value in seeds[start : start + 256, 0]]
            ]
            positions, weights = model(query_batch, seed_batch)
            position_values = positions.cpu().numpy()
            weight_values = weights.cpu().numpy()
            scores = np.einsum(
                "bhd,pd->bhp",
                position_values,
                context.centroids,
            )
            scores = scores / 0.05 + np.log(weight_values + 1e-9)[:, :, None]
            score_batches.append(np.logaddexp.reduce(scores, axis=1))
    partition_scores = np.concatenate(score_batches, axis=0)
    output = {}
    for partitions, quota in ((20, 5), (10, 10), (5, 20)):
        label = f"shared_partition_p{partitions}q{quota}"
        output[label] = _partition_quota_order(
            partition_scores,
            queries,
            context.documents,
            context.docs_by_partition,
            partitions,
            quota,
        )
    return output


def _unified_fusion_specs(partition_signals):
    """One validation-selected fusion policy shared by every corpus."""
    specs = [
        {"label": "dense", "weights": {"dense": 1.0}},
        {"label": "shared_router", "weights": {"shared_router": 1.0}},
    ]
    for relational_weight in (0.5, 1.0, 2.0):
        specs.append(
            {
                "label": f"rrf_dense+shared_r{relational_weight:g}",
                "weights": {
                    "dense": 1.0,
                    "shared_router": relational_weight,
                },
            }
        )
    for partition_signal in partition_signals:
        specs.append(
            {
                "label": partition_signal,
                "weights": {partition_signal: 1.0},
            }
        )
        for partition_weight in (0.5, 1.0, 2.0):
            specs.extend(
                [
                    {
                        "label": (
                            f"rrf_dense+{partition_signal}"
                            f"_p{partition_weight:g}"
                        ),
                        "weights": {
                            "dense": 1.0,
                            partition_signal: partition_weight,
                        },
                    },
                    {
                        "label": (
                            f"rrf_shared+{partition_signal}"
                            f"_p{partition_weight:g}"
                        ),
                        "weights": {
                            "shared_router": 1.0,
                            partition_signal: partition_weight,
                        },
                    },
                    {
                        "label": (
                            f"rrf_dense+shared+{partition_signal}"
                            f"_p{partition_weight:g}"
                        ),
                        "weights": {
                            "dense": 1.0,
                            "shared_router": 1.0,
                            partition_signal: partition_weight,
                        },
                    },
                ]
            )
    return specs


def _macro_metrics(per_dataset: Mapping[str, Mapping]) -> dict:
    return _mean_metrics(list(per_dataset.values()))


def _budget_robust_key(metrics):
    fullcov = [metrics["fullcov"][str(budget)] for budget in BUDGETS]
    recall = [metrics["recall"][str(budget)] for budget in BUDGETS]
    return (
        float(np.mean(fullcov)),
        float(np.mean(recall)),
        metrics["fullcov"]["100"],
        -metrics["weakest_positive_rank"],
    )


def _budget_selection_key(metrics, budget):
    key = str(budget)
    return (
        metrics["fullcov"][key],
        metrics["recall"][key],
        metrics["hit_rate"][key],
        -metrics["weakest_positive_rank"],
    )


def _assemble_budget_metrics(per_budget, include_vectors=False):
    primary = per_budget[MAX_CANDIDATES]
    output = {
        group: {
            str(budget): per_budget[budget][group][str(budget)]
            for budget in BUDGETS
        }
        for group in ("recall", "fullcov", "hit_rate")
    }
    output["weakest_positive_rank"] = primary["weakest_positive_rank"]
    output["n_queries"] = primary["n_queries"]
    if include_vectors:
        output["_fullcov_vectors"] = {
            str(budget): per_budget[budget]["_fullcov_vectors"][str(budget)]
            for budget in BUDGETS
        }
    return output


def _without_private_metrics(metrics: Mapping) -> dict:
    return {
        key: value for key, value in metrics.items() if not key.startswith("_")
    }


def _train_shared(
    config: UnifiedConfig,
    contexts: Mapping[str, DatasetContext],
    train_datasets: Sequence[str],
    selection_datasets: Sequence[str],
    *,
    device,
    seed,
    epochs,
    hard_negative_k,
    eval_every,
    patience,
    batch_size,
):
    dimension = next(iter(contexts.values())).documents.shape[1]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = DatasetAgnosticRouter(
        dimension,
        config.relational_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_tensors = {
        dataset: torch.tensor(
            contexts[dataset].prepared["train"][0],
            device=device,
        )
        for dataset in train_datasets
    }
    eligible = {
        dataset: [
            index
            for index, positives in enumerate(
                contexts[dataset].prepared["train"][2]
            )
            if positives
        ]
        for dataset in train_datasets
    }

    best_state = None
    best_macro = None
    best_per_dataset = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = {dataset: [] for dataset in train_datasets}
        schedule = _balanced_epoch_batches(
            eligible,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
        )
        for dataset, batch in schedule:
            context = contexts[dataset]
            prepared = context.prepared["train"]
            query_batch = train_tensors[dataset][batch]
            seed_batch = context.document_tensor[
                [int(prepared[1][index, 0]) for index in batch]
            ]
            positions, weights = model(query_batch, seed_batch)
            candidate_vectors, positive_mask = _candidate_batch(
                [prepared[2][index] for index in batch],
                positions,
                context.index,
                context.document_tensor,
                device,
                hard_negative_k,
            )
            logits = _combined_logits(positions, weights, candidate_vectors)
            kl, coverage = _coverage_objective(logits, positive_mask)
            loss = kl + config.lambda_coverage * coverage
            loss = loss + config.lambda_diversity * _head_diversity(
                positions[:, 1:],
                seed_batch,
            )
            loss = loss + config.lambda_balance * _gate_balance(weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses[dataset].append(float(loss.detach().cpu()))

        if epoch % eval_every != 0 and epoch != epochs:
            continue
        per_dataset = {}
        for dataset in selection_datasets:
            context = contexts[dataset]
            order = _retrieve(
                model,
                context.prepared["val"],
                context,
                device,
            )
            per_dataset[dataset] = _metrics(
                order,
                context.prepared["val"][2],
            )
        macro = _macro_metrics(per_dataset)
        history.append(
            {
                "epoch": epoch,
                "train_loss_by_dataset": {
                    dataset: round(float(np.mean(values)), 6)
                    for dataset, values in losses.items()
                },
                "validation_macro": macro,
                "validation_by_dataset": per_dataset,
            }
        )
        if best_macro is None or _selection_key(macro) > _selection_key(best_macro):
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_macro = macro
            best_per_dataset = per_dataset
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        log.info(
            "[unified] %s seed=%d epoch=%d macro val FCov@100=%.2f",
            config.label,
            seed,
            epoch,
            macro["fullcov"]["100"],
        )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Unified router training produced no validation state.")
    model.load_state_dict(best_state)
    model.eval()
    return (
        model,
        best_state,
        best_macro,
        best_per_dataset,
        best_epoch,
        history,
    )


def _training_signature(
    contexts,
    train_datasets,
    selection_datasets,
    config,
    *,
    seed,
    epochs,
    hard_negative_k,
    eval_every,
    patience,
    batch_size,
):
    payload = {
        "version": 1,
        "datasets": {
            dataset: contexts[dataset].fingerprint
            for dataset in sorted(contexts)
        },
        "train_datasets": sorted(train_datasets),
        "selection_datasets": sorted(selection_datasets),
        "config": asdict(config),
        "seed": seed,
        "epochs": epochs,
        "hard_negative_k": hard_negative_k,
        "eval_every": eval_every,
        "patience": patience,
        "batch_size": batch_size,
        "sampling": "equal_updates_per_dataset_with_deterministic_oversampling",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _local_upper_bound(dataset: str, local_run_id: str):
    path = (
        Path("data")
        / "ukb_storage"
        / dataset
        / "results"
        / "L1"
        / local_run_id
        / "summary.json"
    )
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {
        "run_id": local_run_id,
        "selected_model": payload.get("selected_model"),
        "selected_fusion": payload.get("selected_fusion"),
        "test_summary": payload.get("test_summary"),
    }


def run(
    datasets,
    run_id,
    *,
    holdout_dataset=None,
    epochs=40,
    limit=15000,
    relational_heads=(1, 4, 8),
    coverage_lambdas=(0.0, 0.25),
    seeds=(42,),
    hard_negative_k=32,
    eval_every=5,
    patience=3,
    batch_size=128,
    local_run_id="l1opt_v1",
    device=None,
):
    datasets = list(dict.fromkeys(datasets))
    if holdout_dataset and holdout_dataset not in datasets:
        raise ValueError("--holdout-dataset must also appear in --datasets.")
    train_datasets = [
        dataset for dataset in datasets if dataset != holdout_dataset
    ]
    if not train_datasets:
        raise ValueError("Unified training needs at least one non-held-out dataset.")
    selection_datasets = train_datasets
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    contexts = {}
    encoder_holder = []
    for dataset in datasets:
        log.info("[unified] preparing %s", dataset)
        contexts[dataset] = _prepare_dataset(
            dataset,
            limit=limit,
            device=device,
            encoder_holder=encoder_holder,
        )
    dimensions = {context.documents.shape[1] for context in contexts.values()}
    if len(dimensions) != 1:
        raise RuntimeError(
            "Dataset-agnostic training requires one shared encoder dimension."
        )

    configs = []
    for heads in relational_heads:
        for coverage in coverage_lambdas:
            configs.append(
                UnifiedConfig(
                    relational_heads=heads,
                    lambda_coverage=coverage,
                    lambda_diversity=0.05 if heads > 1 else 0.0,
                    lambda_balance=0.01,
                )
            )
    shared_root = (
        Path("data") / "ukb_storage" / "_shared"
    )
    checkpoint_dir = shared_root / "checkpoints" / "L1" / run_id
    result_dir = shared_root / "results" / "L1" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trained = {}
    validation_models = {}
    validation_by_dataset = {}
    for config in configs:
        validation_models[config.label] = []
        validation_by_dataset[config.label] = {}
        for seed in seeds:
            signature = _training_signature(
                contexts,
                train_datasets,
                selection_datasets,
                config,
                seed=seed,
                epochs=epochs,
                hard_negative_k=hard_negative_k,
                eval_every=eval_every,
                patience=patience,
                batch_size=batch_size,
            )
            checkpoint_path = (
                checkpoint_dir / f"model_{config.label}_seed{seed}.pth"
            )
            cached = _load_matching_checkpoint(
                checkpoint_path,
                signature,
                device,
            )
            if cached is None:
                log.info("[unified] training %s seed=%d", config.label, seed)
                (
                    model,
                    state,
                    macro,
                    per_dataset,
                    best_epoch,
                    history,
                ) = _train_shared(
                    config,
                    contexts,
                    train_datasets,
                    selection_datasets,
                    device=device,
                    seed=seed,
                    epochs=epochs,
                    hard_negative_k=hard_negative_k,
                    eval_every=eval_every,
                    patience=patience,
                    batch_size=batch_size,
                )
                _atomic_torch_save(
                    {
                        "training_signature": signature,
                        "model_config": asdict(config),
                        "seed": seed,
                        "model_state_dict": state,
                        "validation_macro": macro,
                        "validation_by_dataset": per_dataset,
                        "best_epoch": best_epoch,
                        "history": history,
                        "embedding_dim": next(iter(dimensions)),
                        "datasets": {
                            dataset: contexts[dataset].fingerprint
                            for dataset in datasets
                        },
                        "holdout_dataset": holdout_dataset,
                    },
                    checkpoint_path,
                )
            else:
                log.info("[unified] reusing checkpoint %s", checkpoint_path)
                model = DatasetAgnosticRouter(
                    cached["embedding_dim"],
                    config.relational_heads,
                ).to(device)
                model.load_state_dict(cached["model_state_dict"])
                model.eval()
                state = cached["model_state_dict"]
                macro = cached["validation_macro"]
                per_dataset = cached["validation_by_dataset"]
                best_epoch = cached["best_epoch"]
                history = cached["history"]
            validation_models[config.label].append(macro)
            validation_by_dataset[config.label][str(seed)] = per_dataset
            trained[(config.label, seed)] = {
                "model": model,
                "state": state,
                "best_epoch": best_epoch,
                "history": history,
            }

    validation_summary = {
        label: _mean_metrics(values)
        for label, values in validation_models.items()
    }
    selected_label = max(
        validation_summary,
        key=lambda label: _budget_robust_key(validation_summary[label]),
    )
    selected_config = next(
        config for config in configs if config.label == selected_label
    )
    log.info("[unified] validation-selected model: %s", selected_label)

    signal_orders = {}
    for dataset, context in contexts.items():
        signal_orders[dataset] = {}
        for seed in seeds:
            model = trained[(selected_label, seed)]["model"]
            signal_orders[dataset][seed] = {}
            for split in ("val", "test"):
                direct = _retrieve(
                    model,
                    context.prepared[split],
                    context,
                    device,
                )
                signal_orders[dataset][seed][split] = {
                    "dense": context.dense_orders[split],
                    "shared_router": direct,
                    **_shared_partition_orders(
                        model,
                        context.prepared[split],
                        context,
                        device,
                    ),
                }
    partition_signals = tuple(
        sorted(
            name
            for name in signal_orders[datasets[0]][seeds[0]]["val"]
            if name.startswith("shared_partition_")
        )
    )
    fusion_specs = _unified_fusion_specs(partition_signals)
    validation_fusions = {}
    for spec in fusion_specs:
        metrics = []
        for dataset, context in contexts.items():
            if dataset not in selection_datasets:
                continue
            for seed in seeds:
                order = _apply_spec(
                    spec,
                    signal_orders[dataset][seed]["val"],
                )
                metrics.append(
                    _metrics(order, context.prepared["val"][2])
                )
        validation_fusions[spec["label"]] = _mean_metrics(metrics)
    selected_fusions = {
        str(budget): max(
            fusion_specs,
            key=lambda spec: _budget_selection_key(
                validation_fusions[spec["label"]],
                budget,
            ),
        )
        for budget in BUDGETS
    }
    for budget in BUDGETS:
        log.info(
            "[unified] validation-selected global fusion at K=%d: %s",
            budget,
            selected_fusions[str(budget)]["label"],
        )

    test_by_dataset = {}
    dense_by_dataset = {}
    gate_by_dataset = {}
    significance = {}
    selected_orders = {}
    for dataset, context in contexts.items():
        test_by_dataset[dataset] = {}
        gate_by_dataset[dataset] = {}
        dense = _metrics(
            context.dense_orders["test"],
            context.prepared["test"][2],
            include_vectors=True,
        )
        dense_by_dataset[dataset] = {
            key: value for key, value in dense.items() if not key.startswith("_")
        }
        selected_orders[dataset] = {}
        for seed in seeds:
            model = trained[(selected_label, seed)]["model"]
            orders = {
                budget: _apply_spec(
                    selected_fusions[str(budget)],
                    signal_orders[dataset][seed]["test"],
                )
                for budget in BUDGETS
            }
            selected_orders[dataset][seed] = orders
            per_budget_metrics = {
                budget: _metrics(
                    orders[budget],
                    context.prepared["test"][2],
                    include_vectors=True,
                )
                for budget in BUDGETS
            }
            metrics = _assemble_budget_metrics(
                per_budget_metrics,
                include_vectors=True,
            )
            test_by_dataset[dataset][str(seed)] = metrics
            gate_by_dataset[dataset][str(seed)] = _gate_statistics(
                model,
                context.prepared["test"],
                context,
                device,
            )
        canonical_seed = seeds[0]
        significance[dataset] = {}
        for budget in BUDGETS:
            dense_vector = dense["_fullcov_vectors"][str(budget)]
            unified_vector = test_by_dataset[dataset][str(canonical_seed)][
                "_fullcov_vectors"
            ][str(budget)]
            dense_only = sum(
                int(base and not treatment)
                for base, treatment in zip(dense_vector, unified_vector)
            )
            unified_only = sum(
                int(treatment and not base)
                for base, treatment in zip(dense_vector, unified_vector)
            )
            significance[dataset][str(budget)] = {
                "comparison": "unified vs dense",
                "budget": budget,
                "selected_fusion": selected_fusions[str(budget)]["label"],
                "dense_only": dense_only,
                "unified_only": unified_only,
                "mcnemar_p_value": mcnemar_exact(dense_only, unified_only),
            }

    test_summary_by_dataset = {
        dataset: _mean_metrics(list(seed_metrics.values()))
        for dataset, seed_metrics in test_by_dataset.items()
    }
    test_macro = _macro_metrics(test_summary_by_dataset)
    dense_macro = _macro_metrics(dense_by_dataset)
    canonical_seed = seeds[0]
    for dataset, context in contexts.items():
        dataset_result = (
            Path("data")
            / "ukb_storage"
            / dataset
            / "results"
            / "L1"
            / f"{run_id}_unified"
        )
        for budget in BUDGETS:
            budget_order = [
                order[:budget]
                for order in selected_orders[dataset][canonical_seed][budget]
            ]
            _write_candidates(
                dataset_result / f"candidates_test_k{budget}.jsonl",
                context.prepared["test"][3],
                budget_order,
                context.prepared["test"][2],
                context.document_ids,
                canonical_seed,
                selected_fusions[str(budget)]["label"],
            )
            if budget == MAX_CANDIDATES:
                _write_candidates(
                    dataset_result / "candidates_test.jsonl",
                    context.prepared["test"][3],
                    budget_order,
                    context.prepared["test"][2],
                    context.document_ids,
                    canonical_seed,
                    selected_fusions[str(budget)]["label"],
                )
        _atomic_json_dump(
            {
                "run_id": run_id,
                "shared_checkpoint": str(checkpoint_dir / f"seed_{canonical_seed}.pth"),
                "dataset": dataset,
                "dataset_identity_input": False,
                "holdout_dataset": holdout_dataset,
                "selected_model": selected_label,
                "selected_fusions_by_budget": selected_fusions,
                "canonical_seed": canonical_seed,
                "test_summary": test_summary_by_dataset[dataset],
                "dense_baseline": dense_by_dataset[dataset],
                "gate_statistics": gate_by_dataset[dataset][str(canonical_seed)],
                "significance_vs_dense": significance[dataset],
            },
            dataset_result / "summary.json",
        )

    for seed in seeds:
        _atomic_torch_save(
            {
                "run_id": run_id,
                "seed": seed,
                "datasets": datasets,
                "train_datasets": train_datasets,
                "holdout_dataset": holdout_dataset,
                "dataset_identity_input": False,
                "model_config": asdict(selected_config),
                "fusions_by_budget": selected_fusions,
                "model_state_dict": trained[(selected_label, seed)]["state"],
                "embedding_dim": next(iter(dimensions)),
                "encoder": ENCODER_NAME,
                "fingerprints": {
                    dataset: contexts[dataset].fingerprint
                    for dataset in datasets
                },
            },
            checkpoint_dir / f"seed_{seed}.pth",
        )

    local_upper_bounds = {
        dataset: _local_upper_bound(dataset, local_run_id)
        for dataset in datasets
    }
    result = {
        "run_id": run_id,
        "device": str(device),
        "protocol": {
            "objective": "one shared dataset-agnostic checkpoint",
            "dataset_identity_input": False,
            "encoder": ENCODER_NAME,
            "datasets": datasets,
            "train_datasets": train_datasets,
            "selection_datasets": selection_datasets,
            "holdout_dataset": holdout_dataset,
            "selection_split": "macro_average_validation_on_train_datasets",
            "model_selection": "mean FullCov over K=20/50/100",
            "fusion_selection": (
                "one corpus-independent policy selected separately at each K"
            ),
            "test_used_after_selection": True,
            "limit_per_split": limit,
            "balanced_sampling": (
                "equal optimizer updates per dataset with deterministic "
                "oversampling; no pooled-size weighting"
            ),
            "budgets": list(BUDGETS),
            "fingerprints": {
                dataset: contexts[dataset].fingerprint for dataset in datasets
            },
            "cache_manifests": {
                dataset: contexts[dataset].manifest for dataset in datasets
            },
        },
        "architecture": {
            "dense_skip_head": True,
            "query_conditioned_gate": True,
            "relational_heads": selected_config.relational_heads,
            "total_heads": selected_config.total_heads,
            "dataset_specific_parameters": 0,
            "ranking": (
                "weighted soft-OR over pure dense query and relational "
                "seed-offset positions, plus optional fixed-budget routing "
                "through per-corpus immutable centroids"
            ),
            "partition_router_parameters": 0,
        },
        "training": {
            "epochs_max": epochs,
            "eval_every": eval_every,
            "patience": patience,
            "hard_negative_k": hard_negative_k,
            "batch_size": batch_size,
            "seeds": list(seeds),
            "loss": (
                "multi-positive KL + weakest-positive top100 coverage barrier "
                "+ relational-head diversity + gate balance"
            ),
        },
        "model_candidates": [
            asdict(config)
            | {
                "label": config.label,
                "total_heads": config.total_heads,
            }
            for config in configs
        ],
        "validation_models_macro": validation_summary,
        "validation_models_by_dataset": validation_by_dataset,
        "fusion_candidates": fusion_specs,
        "validation_fusions_macro": validation_fusions,
        "selected_fusions_by_budget": selected_fusions,
        "selected_model": asdict(selected_config)
        | {
            "label": selected_label,
            "total_heads": selected_config.total_heads,
        },
        "test_by_dataset_and_seed": {
            dataset: {
                seed: _without_private_metrics(metrics)
                for seed, metrics in seed_metrics.items()
            }
            for dataset, seed_metrics in test_by_dataset.items()
        },
        "test_summary_by_dataset": test_summary_by_dataset,
        "test_macro": test_macro,
        "dense_by_dataset": dense_by_dataset,
        "dense_macro": dense_macro,
        "gate_by_dataset_and_seed": gate_by_dataset,
        "significance_vs_dense": significance,
        "local_model_upper_bounds": local_upper_bounds,
        "acceptance_gate": {
            "target": (
                "unified FullCov@100 within 2 percentage points of each "
                "validation-selected per-dataset upper bound"
            ),
            "status": (
                "pending_local_runs"
                if any(value is None for value in local_upper_bounds.values())
                else "ready_to_evaluate"
            ),
        },
        "checkpoint_dir": str(checkpoint_dir),
    }
    _atomic_json_dump(result, result_dir / "summary.json")
    _atomic_json_dump(
        {
            f"{label}__seed{seed}": trained[(label, seed)]["history"]
            for label in validation_summary
            for seed in seeds
        },
        result_dir / "training_history.json",
    )
    log.info(
        "[unified] TEST macro FCov@100 %.2f; dense %.2f; R@100 %.2f",
        test_macro["fullcov"]["100"],
        dense_macro["fullcov"]["100"],
        test_macro["recall"]["100"],
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train one dataset-agnostic Level-1 dense/relational router."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "2wiki_clean",
            "musique_clean",
            "hotpotqa_clean",
            "squad_clean",
            "metaqa",
        ],
    )
    parser.add_argument("--run-id", default="l1_unified_v1")
    parser.add_argument("--holdout-dataset")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--limit", type=int, default=15000)
    parser.add_argument(
        "--relational-heads",
        type=int,
        nargs="+",
        default=[1, 4, 8],
    )
    parser.add_argument(
        "--coverage-lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.25],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--hard-negative-k", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--local-run-id", default="l1opt_v1")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(
        datasets=args.datasets,
        run_id=args.run_id,
        holdout_dataset=args.holdout_dataset,
        epochs=args.epochs,
        limit=args.limit,
        relational_heads=tuple(args.relational_heads),
        coverage_lambdas=tuple(args.coverage_lambdas),
        seeds=tuple(args.seeds),
        hard_negative_k=args.hard_negative_k,
        eval_every=args.eval_every,
        patience=args.patience,
        batch_size=args.batch_size,
        local_run_id=args.local_run_id,
    )


if __name__ == "__main__":
    main()
