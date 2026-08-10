"""Validation-locked Level-1 document candidate optimization.

This is the publication-oriented successor to the exploratory ``l1_*`` scripts:

* model and fusion choices are made on validation queries only;
* every gold document for a query is handled as a positive in the same loss;
* MetaQA's official train/dev/test split is preserved by ``_splits``;
* the final test evaluation is run only after the model/fusion configuration is
  frozen; and
* the selected top-100 candidates and checkpoint are persisted for Level 2.

The retriever is a lightweight query-conditioned ensemble of relational offsets.
It predicts K document-space offsets from the query and combines the heads with
a learned soft-OR. The objective combines multi-positive KL with a differentiable
FullCov@K barrier, plus small head-diversity and gate-balance regularizers.
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
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from src.core.encoders import DenseEncoder
from src.core.engine import CoreEngine
from src.alignment.train_mlp import kl_div_loss
from src.experiments.l1_ablate import _ranks, _rrf_fuse
from src.experiments.l1_mlp_transformer import MLPTransformer, _combined_logits
from src.experiments.multisignal_route import MultiRouter
from src.experiments.overlap_retrain import (
    HNK,
    TAU,
    _centroids,
    _hard_membership,
    _reconstruct,
    _splits,
)
from src.experiments.stats import mcnemar_exact

log = logging.getLogger("experiments.l1_optimize")

BUDGETS = (20, 50, 100)
MAX_CANDIDATES = max(BUDGETS)
ENCODER_NAME = "multi-qa-MiniLM-L6-cos-v1"


@dataclass(frozen=True)
class ModelConfig:
    heads: int
    lambda_coverage: float
    lambda_diversity: float
    lambda_balance: float

    @property
    def label(self) -> str:
        return (
            f"k{self.heads}_cov{self.lambda_coverage:g}"
            f"_div{self.lambda_diversity:g}_bal{self.lambda_balance:g}"
        )


def _cap(items: Sequence, limit: int, seed: int) -> list:
    """Take a deterministic, order-unbiased subset instead of a prefix."""
    if not limit or len(items) <= limit:
        return list(items)
    chosen = sorted(random.Random(seed).sample(range(len(items)), limit))
    return [items[i] for i in chosen]


def _split_manifest(splits: Dict[str, Sequence]) -> dict:
    out = {}
    for name, rows in splits.items():
        digest = hashlib.sha256()
        for node, positive_partitions, gold_ids in rows:
            record = {
                "query_id": node.node_id,
                "content": node.content,
                "positive_partitions": sorted(map(int, positive_partitions)),
                "gold_ids": sorted(map(str, gold_ids)),
                "source_split": node.metadata.get("split"),
            }
            digest.update(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            digest.update(b"\n")
        out[name] = {"count": len(rows), "sha256": digest.hexdigest()}
    return out


def _document_manifest(engine, documents):
    """Fingerprint every input that can change a cached document ranking."""
    digest = hashlib.sha256()
    for node in engine.nodes:
        record = {
            "node_id": node.node_id,
            "content": node.content,
            "partition": int(engine.partition_map[node.node_id]),
        }
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    vectors = np.ascontiguousarray(documents)
    digest.update(memoryview(vectors).cast("B"))
    return {
        "count": len(engine.nodes),
        "dimension": int(documents.shape[1]),
        "sha256": digest.hexdigest(),
    }


def _query_cache_valid(cache_path, rows, dense_seeds, dimension):
    if not cache_path.exists():
        return False
    expected_ids = [row[0].node_id for row in rows]
    try:
        with np.load(cache_path, allow_pickle=True) as cached:
            cached_ids = cached["query_ids"].tolist()
            return (
                int(cached["dense_seeds"]) == dense_seeds
                and cached_ids == expected_ids
                and cached["query_vectors"].shape == (len(rows), dimension)
                and cached["seeds"].shape == (len(rows), dense_seeds)
                and len(cached["gold"]) == len(rows)
            )
    except (KeyError, OSError, TypeError, ValueError, EOFError):
        log.warning("Ignoring invalid UKB query cache: %s", cache_path)
        return False


def _prepare(
    rows: Sequence,
    encoder: DenseEncoder,
    index,
    id_to_idx: Dict[str, int],
    dense_seeds: int,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]], List[str]]:
    texts = [node.content for node, _, _ in rows]
    query_vectors = encoder.encode(texts).astype("float32")
    faiss.normalize_L2(query_vectors)
    _, seeds = index.search(query_vectors, dense_seeds)
    gold = [[id_to_idx[g] for g in golds if g in id_to_idx] for _, _, golds in rows]
    query_ids = [node.node_id for node, _, _ in rows]
    return query_vectors, seeds, gold, query_ids


def _prepare_cached(cache_path, rows, encoder, index, id_to_idx, dense_seeds):
    if _query_cache_valid(cache_path, rows, dense_seeds, index.d):
        with np.load(cache_path, allow_pickle=True) as cached:
            log.info("Reusing UKB query cache: %s", cache_path)
            return (
                cached["query_vectors"].astype("float32"),
                cached["seeds"].astype(np.int64),
                [list(map(int, values)) for values in cached["gold"]],
                cached["query_ids"].tolist(),
            )
    if encoder is None:
        raise RuntimeError(f"Missing UKB query cache {cache_path} and no encoder was loaded")
    prepared = _prepare(rows, encoder, index, id_to_idx, dense_seeds)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            query_vectors=prepared[0],
            seeds=prepared[1].astype(np.int32),
            gold=np.asarray(prepared[2], dtype=object),
            query_ids=np.asarray(prepared[3], dtype=object),
            dense_seeds=np.asarray(dense_seeds),
        )
    os.replace(tmp_path, cache_path)
    _commit_remote_artifacts(cache_path)
    return prepared


def _load_matching_checkpoint(path, training_signature, device):
    if not path.exists():
        return None
    try:
        candidate = torch.load(path, map_location=device, weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError, TypeError):
        log.warning("Ignoring incomplete UKB checkpoint: %s", path)
        return None
    if not isinstance(candidate, dict):
        return None
    if candidate.get("training_signature") != training_signature:
        return None
    return candidate


def _commit_remote_artifacts(path):
    """Make completed cache/checkpoint files survive Modal worker preemption."""
    try:
        from src.experiments.backends import commit_persistent_storage

        if commit_persistent_storage():
            log.info("Committed recoverable artifact to remote storage: %s", path)
    except Exception as exc:  # The final task-level commit remains a fallback.
        log.warning("Could not commit remote artifact %s: %s", path, exc)


def _atomic_torch_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        _commit_remote_artifacts(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_json_dump(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, path)
        _commit_remote_artifacts(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _candidate_batch(gold_sets, positions, index, document_tensor, device, hard_negative_k):
    candidate_ids = set()
    for gold in gold_sets:
        candidate_ids.update(gold)
    if hard_negative_k:
        flat = positions.detach().reshape(-1, positions.shape[-1]).cpu().numpy().astype("float32")
        _, neighbors = index.search(np.ascontiguousarray(flat), hard_negative_k)
        candidate_ids.update(int(x) for x in neighbors.reshape(-1) if x >= 0)
    candidates = sorted(candidate_ids)
    lookup = {doc_id: column for column, doc_id in enumerate(candidates)}
    vectors = document_tensor[torch.tensor(candidates, device=device)]
    positive_mask = torch.zeros(
        len(gold_sets), len(candidates), dtype=torch.bool, device=device
    )
    for row, gold in enumerate(gold_sets):
        for doc_id in gold:
            positive_mask[row, lookup[doc_id]] = True
    return vectors, positive_mask


def _coverage_objective(logits, positive_mask, target_topk=MAX_CANDIDATES):
    """Multi-positive KL plus weakest-positive top-K and hardest-negative barriers."""
    log_probs = F.log_softmax(logits, dim=1)
    target = positive_mask.float()
    target = target / target.sum(1, keepdim=True).clamp(min=1)
    kl = -(target * log_probs).sum(1).mean()

    barrier_terms = []
    margin_terms = []
    for row_logits, row_mask in zip(logits, positive_mask):
        positives = row_logits[row_mask]
        negatives = row_logits[~row_mask]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        kth = min(max(1, target_topk - positives.numel() + 1), negatives.numel())
        threshold = torch.topk(negatives, kth).values[-1]
        barrier_terms.append(F.softplus(threshold - positives).max())
        margin_terms.append(F.softplus(negatives.max() - positives).max())

    if not barrier_terms:
        return kl, logits.new_tensor(0.0)
    coverage = torch.stack(barrier_terms).mean() + 0.25 * torch.stack(margin_terms).mean()
    return kl, coverage


def _head_diversity(positions, seed_vectors):
    if positions.shape[1] <= 1:
        return positions.new_tensor(0.0)
    offsets = F.normalize(positions - seed_vectors.unsqueeze(1), dim=-1)
    gram = torch.einsum("bkd,bjd->bkj", offsets, offsets).abs()
    heads = positions.shape[1]
    return ((gram.sum((1, 2)) - heads) / (heads * (heads - 1))).mean()


def _gate_balance(weights):
    if weights.shape[1] <= 1:
        return weights.new_tensor(0.0)
    mean_weights = weights.mean(0).clamp(min=1e-9)
    return (mean_weights * torch.log(mean_weights * weights.shape[1])).sum()


def _retrieve_model(model, queries, seeds, documents, document_tensor, index, device, k):
    """Retrieve with one dense seed per query and a learned soft-OR over heads."""
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(queries), 256):
            q_batch = torch.tensor(queries[start : start + 256], device=device)
            seed_batch = document_tensor[
                [int(x) for x in seeds[start : start + 256]]
            ]
            positions, weights = model(q_batch, seed_batch)
            flat = positions.reshape(-1, positions.shape[-1]).cpu().numpy().astype("float32")
            _, per_head = index.search(np.ascontiguousarray(flat), k)
            per_head = per_head.reshape(len(q_batch), positions.shape[1], k)
            pos_np = positions.cpu().numpy()
            weight_np = weights.cpu().numpy()
            for row in range(len(q_batch)):
                candidates = list(dict.fromkeys(per_head[row].reshape(-1).tolist()))
                candidate_vectors = documents[candidates]
                scores = (
                    candidate_vectors @ pos_np[row].T / 0.05
                    + np.log(weight_np[row] + 1e-9)
                )
                combined = np.logaddexp.reduce(scores, axis=1)
                ranked = np.argsort(-combined)[:k]
                output.append([int(candidates[i]) for i in ranked])
    return output


def _multi_seed_order(model, queries, seeds, documents, document_tensor, index, device, count):
    orders = [
        _retrieve_model(
            model, queries, seeds[:, seed_col], documents, document_tensor, index,
            device, MAX_CANDIDATES
        )
        for seed_col in range(count)
    ]
    if len(orders) == 1:
        return orders[0]
    return _rrf_fuse([_ranks(order) for order in orders], [1.0] * len(orders), k=MAX_CANDIDATES)


def _metrics(order, gold, budgets=BUDGETS, include_vectors=False):
    recall = {k: [] for k in budgets}
    full_coverage = {k: [] for k in budgets}
    hit = {k: [] for k in budgets}
    weakest = []
    vectors = {k: [] for k in budgets}
    for ranked, positives in zip(order, gold):
        if not positives:
            continue
        ranks = {int(doc): rank + 1 for rank, doc in enumerate(ranked)}
        positive_set = set(positives)
        positive_ranks = [ranks.get(doc, MAX_CANDIDATES + 1) for doc in positive_set]
        weakest.append(max(positive_ranks))
        for k in budgets:
            retrieved = set(ranked[:k])
            overlap = len(positive_set & retrieved)
            recall[k].append(overlap / len(positive_set))
            covered = float(overlap == len(positive_set))
            full_coverage[k].append(covered)
            hit[k].append(float(overlap > 0))
            vectors[k].append(int(covered))
    result = {
        "recall": {str(k): round(float(np.mean(recall[k])) * 100, 2) for k in budgets},
        "fullcov": {
            str(k): round(float(np.mean(full_coverage[k])) * 100, 2) for k in budgets
        },
        "hit_rate": {str(k): round(float(np.mean(hit[k])) * 100, 2) for k in budgets},
        "weakest_positive_rank": round(float(np.mean(weakest)), 2),
        "n_queries": len(weakest),
    }
    if include_vectors:
        result["_fullcov_vectors"] = {str(k): vectors[k] for k in budgets}
    return result


def _selection_key(metrics):
    return (
        metrics["fullcov"][str(MAX_CANDIDATES)],
        metrics["recall"][str(MAX_CANDIDATES)],
        metrics["fullcov"]["50"],
        -metrics["weakest_positive_rank"],
    )


def _train(
    config,
    train_data,
    val_data,
    documents,
    document_tensor,
    index,
    device,
    seed,
    epochs,
    hard_negative_k,
    eval_every,
    patience,
):
    q_train, seed_train, gold_train, _ = train_data
    q_val, seed_val, gold_val, _ = val_data
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = MLPTransformer(documents.shape[1], config.heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_tensor = torch.tensor(q_train, device=device)
    eligible = [i for i, positives in enumerate(gold_train) if positives]
    best_state = None
    best_metrics = None
    best_epoch = 0
    stale_rounds = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        random.Random(seed + epoch).shuffle(eligible)
        epoch_loss = []
        for start in range(0, len(eligible), 128):
            batch = eligible[start : start + 128]
            query_batch = train_tensor[batch]
            seed_batch = document_tensor[[int(seed_train[i, 0]) for i in batch]]
            positions, weights = model(query_batch, seed_batch)
            candidate_vectors, positive_mask = _candidate_batch(
                [gold_train[i] for i in batch],
                positions,
                index,
                document_tensor,
                device,
                hard_negative_k,
            )
            logits = _combined_logits(positions, weights, candidate_vectors)
            kl, coverage = _coverage_objective(logits, positive_mask)
            loss = kl + config.lambda_coverage * coverage
            loss = loss + config.lambda_diversity * _head_diversity(positions, seed_batch)
            loss = loss + config.lambda_balance * _gate_balance(weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach().cpu()))

        if epoch % eval_every != 0 and epoch != epochs:
            continue
        val_order = _retrieve_model(
            model, q_val, seed_val[:, 0], documents, document_tensor, index,
            device, MAX_CANDIDATES
        )
        val_metrics = _metrics(val_order, gold_val)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(np.mean(epoch_loss)), 6),
                "val": val_metrics,
            }
        )
        if best_metrics is None or _selection_key(val_metrics) > _selection_key(best_metrics):
            best_metrics = val_metrics
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_rounds = 0
        else:
            stale_rounds += 1
        if stale_rounds >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_state, best_metrics, best_epoch, history


def _bm25_order(engine, rows, k):
    id_to_idx = engine.node_id_to_idx
    output = []
    for node, _, _ in rows:
        docs = engine.search_lexical(node.content, k=k)
        output.append([id_to_idx[doc.node_id] for doc in docs if doc.node_id in id_to_idx][:k])
    return output


def _bm25_order_cached(cache_path, engine, rows, k):
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == (len(rows), k):
            log.info("Reusing UKB BM25 cache: %s", cache_path)
            return cached.astype(np.int64).tolist()
    order = _bm25_order(engine, rows, k)
    matrix = np.full((len(order), k), -1, dtype=np.int32)
    for row, values in enumerate(order):
        matrix[row, : len(values)] = values
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, matrix)
    os.replace(tmp_path, cache_path)
    return order


def _partition_features(prepared, documents, signal_set):
    queries, seeds, _, _ = prepared
    pieces = [queries]
    if signal_set == "q+seed+nbr":
        pieces.extend([documents[seeds[:, 0]], documents[seeds[:, :10]].mean(axis=1)])
    return np.concatenate(pieces, axis=1).astype("float32")


def _partition_metrics(scores, rows, budget=20):
    ranked = np.argsort(-scores, axis=1)
    full_coverage = []
    recall = []
    for order, (_, positive_partitions, _) in zip(ranked, rows):
        positives = set(int(pid) for pid in positive_partitions)
        selected = set(int(pid) for pid in order[:budget])
        overlap = len(positives & selected)
        full_coverage.append(float(overlap == len(positives)))
        recall.append(overlap / len(positives))
    return {
        "fullcov@20": round(float(np.mean(full_coverage)) * 100, 2),
        "recall@20": round(float(np.mean(recall)) * 100, 2),
    }


def _train_partition_router(
    train_features,
    train_rows,
    val_features,
    val_rows,
    centroids,
    device,
    tau,
    hard_negative_k,
    epochs,
    seed=42,
):
    torch.manual_seed(seed)
    model = MultiRouter(train_features.shape[1], centroids.shape[1]).to(device)
    centroid_tensor = F.normalize(torch.tensor(centroids, device=device), dim=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_state = None
    best_metrics = None
    best_epoch = 0
    stale = 0
    order = list(range(len(train_rows)))
    for epoch in range(1, epochs + 1):
        model.train()
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), 64):
            batch = order[start : start + 64]
            projected = model(torch.tensor(train_features[batch], device=device))
            loss = kl_div_loss(
                projected,
                [train_rows[i][1] for i in batch],
                centroid_tensor,
                temperature=tau,
                hn_k=hard_negative_k,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch % 5 != 0 and epoch != epochs:
            continue
        model.eval()
        with torch.no_grad():
            projected = model(torch.tensor(val_features, device=device))
            scores = (projected @ centroid_tensor.T).cpu().numpy()
        metrics = _partition_metrics(scores, val_rows)
        key = (metrics["fullcov@20"], metrics["recall@20"])
        best_key = (
            (best_metrics["fullcov@20"], best_metrics["recall@20"])
            if best_metrics is not None
            else (-1.0, -1.0)
        )
        if key > best_key:
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_metrics = metrics
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= 3:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_state, best_metrics, best_epoch, centroid_tensor


def _partition_quota_order(scores, queries, documents, docs_by_partition, partitions, quota):
    """Interleave dense-ranked documents from the top routed partitions."""
    output = []
    ranked_partitions = np.argsort(-scores, axis=1)[:, :partitions]
    for row, top_partitions in enumerate(ranked_partitions):
        per_partition = []
        for partition_id in top_partitions:
            candidates = docs_by_partition[int(partition_id)]
            if not candidates:
                per_partition.append([])
                continue
            candidate_array = np.asarray(candidates, dtype=np.int64)
            similarities = documents[candidate_array] @ queries[row]
            best = np.argsort(-similarities)[:quota]
            per_partition.append([int(candidate_array[i]) for i in best])
        interleaved = []
        for rank in range(quota):
            for partition_docs in per_partition:
                if rank < len(partition_docs):
                    interleaved.append(partition_docs[rank])
        output.append(interleaved[:MAX_CANDIDATES])
    return output


def _fusion_specs(include_bm25, partition_signals=()):
    specs = [
        {"label": "dense", "weights": {"dense": 1.0}},
        {"label": "offset_seed1", "weights": {"offset_seed1": 1.0}},
        {"label": "offset_seed3", "weights": {"offset_seed3": 1.0}},
    ]
    for signal in ("offset_seed1", "offset_seed3"):
        for relation_weight in (0.5, 1.0, 2.0, 4.0):
            specs.append(
                {
                    "label": f"rrf_dense+{signal}_r{relation_weight:g}",
                    "weights": {"dense": 1.0, signal: relation_weight},
                }
            )
    if include_bm25:
        specs.append({"label": "bm25", "weights": {"bm25": 1.0}})
        for relation_weight in (1.0, 2.0, 4.0):
            for lexical_weight in (0.5, 1.0, 2.0):
                specs.append(
                    {
                        "label": (
                            f"rrf_dense+offset_seed3+bm25"
                            f"_r{relation_weight:g}_b{lexical_weight:g}"
                        ),
                        "weights": {
                            "dense": 1.0,
                            "offset_seed3": relation_weight,
                            "bm25": lexical_weight,
                        },
                    }
                )
    for partition_signal in partition_signals:
        specs.append(
            {"label": partition_signal, "weights": {partition_signal: 1.0}}
        )
        for partition_weight in (0.5, 1.0, 2.0):
            specs.append(
                {
                    "label": f"rrf_dense+{partition_signal}_p{partition_weight:g}",
                    "weights": {"dense": 1.0, partition_signal: partition_weight},
                }
            )
            for relation_weight in (1.0, 2.0):
                specs.append(
                    {
                        "label": (
                            f"rrf_dense+offset_seed3+{partition_signal}"
                            f"_r{relation_weight:g}_p{partition_weight:g}"
                        ),
                        "weights": {
                            "dense": 1.0,
                            "offset_seed3": relation_weight,
                            partition_signal: partition_weight,
                        },
                    }
                )
        if include_bm25:
            specs.append(
                {
                    "label": f"rrf_all+{partition_signal}",
                    "weights": {
                        "dense": 1.0,
                        "offset_seed3": 2.0,
                        "bm25": 1.0,
                        partition_signal: 1.0,
                    },
                }
            )
    return specs


def _apply_spec(spec, signal_orders):
    names = list(spec["weights"])
    return _rrf_fuse(
        [_ranks(signal_orders[name]) for name in names],
        [spec["weights"][name] for name in names],
        k=MAX_CANDIDATES,
    )


def _mean_metrics(items):
    clean = []
    for metrics in items:
        clean.append({k: v for k, v in metrics.items() if not k.startswith("_")})
    result = {}
    for group in ("recall", "fullcov", "hit_rate"):
        result[group] = {}
        for budget in map(str, BUDGETS):
            values = [item[group][budget] for item in clean]
            result[group][budget] = round(float(np.mean(values)), 2)
            result.setdefault(f"{group}_std", {})[budget] = round(float(np.std(values)), 2)
    for key in ("weakest_positive_rank", "n_queries"):
        values = [item[key] for item in clean]
        result[key] = round(float(np.mean(values)), 2)
        if key != "n_queries":
            result[f"{key}_std"] = round(float(np.std(values)), 2)
    return result


def _write_candidates(path, query_ids, order, gold, document_ids, seed, method):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for query_id, ranked, positives in zip(query_ids, order, gold):
                row = {
                    "query_id": query_id,
                    "method": method,
                    "seed": seed,
                    "candidate_doc_indices": [int(x) for x in ranked[:MAX_CANDIDATES]],
                    "candidate_doc_ids": [
                        document_ids[int(x)] for x in ranked[:MAX_CANDIDATES]
                    ],
                    "gold_doc_indices": [int(x) for x in positives],
                    "gold_doc_ids": [document_ids[int(x)] for x in positives],
                }
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def run(
    dataset,
    run_id,
    epochs=40,
    limit=15000,
    heads=(1, 4, 8),
    coverage_lambdas=(0.0, 0.25),
    seeds=(42,),
    hard_negative_k=32,
    eval_every=5,
    patience=3,
    include_bm25=True,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = CoreEngine(source=dataset)
    documents = _reconstruct(engine.node_index).astype("float32")
    faiss.normalize_L2(documents)
    index = faiss.IndexFlatIP(documents.shape[1])
    index.add(documents)
    document_tensor = torch.tensor(documents, device=device)

    membership = _hard_membership(engine)
    splits = _splits(engine, membership)
    split_seed = {"train": 101, "val": 202, "test": 303}
    splits = {
        name: _cap(rows, limit, split_seed[name])
        for name, rows in splits.items()
    }
    if not all(splits.values()):
        raise RuntimeError(f"{dataset}: train/val/test split is incomplete")

    manifest = _split_manifest(splits)
    fingerprint_payload = {
        "version": 2,
        "dataset": dataset,
        "encoder": ENCODER_NAME,
        "document_manifest": _document_manifest(engine, documents),
        "splits": manifest,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = (
        Path("data") / "ukb_storage" / dataset / "cache" / "L1" / fingerprint
    )
    query_cache_paths = {
        name: cache_dir / f"queries_{name}.npz" for name in splits
    }
    encoder = None
    if not all(
        _query_cache_valid(
            query_cache_paths[name],
            splits[name],
            dense_seeds=10,
            dimension=documents.shape[1],
        )
        for name in splits
    ):
        encoder = DenseEncoder(ENCODER_NAME)
    prepared = {
        name: _prepare_cached(
            query_cache_paths[name],
            rows,
            encoder,
            index,
            engine.node_id_to_idx,
            dense_seeds=10,
        )
        for name, rows in splits.items()
    }
    dense_orders = {
        name: index.search(data[0], MAX_CANDIDATES)[1].tolist()
        for name, data in prepared.items()
    }

    num_partitions = max(int(pid) for pid in engine.partition_map.values()) + 1
    centroids, _ = _centroids(engine, documents, membership, num_partitions)
    docs_by_partition = [[] for _ in range(num_partitions)]
    for doc_index, node in enumerate(engine.nodes):
        docs_by_partition[int(engine.partition_map[node.node_id])].append(doc_index)
    checkpoint_dir = (
        Path("data")
        / "ukb_storage"
        / dataset
        / "checkpoints"
        / "L1"
        / run_id
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    partition_routers = {seed: {} for seed in seeds}
    partition_orders = {
        seed: {"val": {}, "test": {}}
        for seed in seeds
    }
    for seed in seeds:
        for signal_set in ("q", "q+seed+nbr"):
            train_features = _partition_features(
                prepared["train"], documents, signal_set
            )
            val_features = _partition_features(prepared["val"], documents, signal_set)
            partition_checkpoint = checkpoint_dir / (
                f"partition_{signal_set.replace('+', '_')}_seed{seed}.pth"
            )
            partition_signature = hashlib.sha256(
                json.dumps(
                    {
                        "version": 2,
                        "data": fingerprint,
                        "signal_set": signal_set,
                        "seed": seed,
                        "epochs": epochs,
                        "tau": TAU.get(dataset, 0.07),
                        "hard_negative_k": min(
                            HNK.get(dataset, num_partitions - 1),
                            num_partitions - 1,
                        ),
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:16]
            cached_router = _load_matching_checkpoint(
                partition_checkpoint, partition_signature, device
            )
            if cached_router is None:
                model, state, metrics, best_epoch, centroid_tensor = (
                    _train_partition_router(
                        train_features,
                        splits["train"],
                        val_features,
                        splits["val"],
                        centroids,
                        device,
                        TAU.get(dataset, 0.07),
                        min(
                            HNK.get(dataset, num_partitions - 1),
                            num_partitions - 1,
                        ),
                        epochs,
                        seed=seed,
                    )
                )
                _atomic_torch_save(
                    {
                        "fingerprint": fingerprint,
                        "training_signature": partition_signature,
                        "signal_set": signal_set,
                        "seed": seed,
                        "model_state_dict": state,
                        "validation": metrics,
                        "best_epoch": best_epoch,
                        "input_dim": train_features.shape[1],
                        "output_dim": documents.shape[1],
                    },
                    partition_checkpoint,
                )
            else:
                log.info(
                    "Reusing UKB partition-router checkpoint: %s",
                    partition_checkpoint,
                )
                model = MultiRouter(
                    cached_router["input_dim"], cached_router["output_dim"]
                ).to(device)
                model.load_state_dict(cached_router["model_state_dict"])
                model.eval()
                state = cached_router["model_state_dict"]
                metrics = cached_router["validation"]
                best_epoch = cached_router["best_epoch"]
                centroid_tensor = F.normalize(
                    torch.tensor(centroids, device=device), dim=-1
                )
            partition_routers[seed][signal_set] = {
                "model": model,
                "state": state,
                "validation": metrics,
                "best_epoch": best_epoch,
            }
            for split_name in ("val", "test"):
                features = _partition_features(
                    prepared[split_name], documents, signal_set
                )
                with torch.no_grad():
                    projected = model(torch.tensor(features, device=device))
                    scores = (projected @ centroid_tensor.T).cpu().numpy()
                for partitions, quota in ((20, 5), (10, 10), (5, 20)):
                    label = (
                        f"partition_{signal_set.replace('+', '_')}"
                        f"_p{partitions}q{quota}"
                    )
                    partition_orders[seed][split_name][label] = (
                        _partition_quota_order(
                            scores,
                            prepared[split_name][0],
                            documents,
                            docs_by_partition,
                            partitions,
                            quota,
                        )
                    )
            log.info(
                "[%s] partition router %s seed=%d best_epoch=%d val FCov@20=%.2f",
                dataset,
                signal_set,
                seed,
                best_epoch,
                metrics["fullcov@20"],
            )

    configs = []
    for head_count in heads:
        for lambda_coverage in coverage_lambdas:
            regularized = head_count > 1
            configs.append(
                ModelConfig(
                    heads=head_count,
                    lambda_coverage=lambda_coverage,
                    lambda_diversity=0.05 if regularized else 0.0,
                    lambda_balance=0.01 if regularized else 0.0,
                )
            )

    trained = {}
    validation_models = {}
    for config in configs:
        validation_models[config.label] = []
        for seed in seeds:
            model_checkpoint = checkpoint_dir / f"model_{config.label}_seed{seed}.pth"
            model_signature = hashlib.sha256(
                json.dumps(
                    {
                        "version": 1,
                        "data": fingerprint,
                        "config": asdict(config),
                        "seed": seed,
                        "epochs": epochs,
                        "hard_negative_k": hard_negative_k,
                        "eval_every": eval_every,
                        "patience": patience,
                    },
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:16]
            cached_model = _load_matching_checkpoint(
                model_checkpoint, model_signature, device
            )
            if cached_model is None:
                log.info("[%s] training %s seed=%d", dataset, config.label, seed)
                model, state, metrics, best_epoch, history = _train(
                    config,
                    prepared["train"],
                    prepared["val"],
                    documents,
                    document_tensor,
                    index,
                    device,
                    seed,
                    epochs,
                    hard_negative_k,
                    eval_every,
                    patience,
                )
                _atomic_torch_save(
                    {
                        "fingerprint": fingerprint,
                        "training_signature": model_signature,
                        "model_config": asdict(config),
                        "seed": seed,
                        "model_state_dict": state,
                        "validation": metrics,
                        "best_epoch": best_epoch,
                        "history": history,
                        "embedding_dim": documents.shape[1],
                    },
                    model_checkpoint,
                )
            else:
                log.info("Reusing UKB offset checkpoint: %s", model_checkpoint)
                model = MLPTransformer(
                    cached_model["embedding_dim"], config.heads
                ).to(device)
                model.load_state_dict(cached_model["model_state_dict"])
                model.eval()
                state = cached_model["model_state_dict"]
                metrics = cached_model["validation"]
                best_epoch = cached_model["best_epoch"]
                history = cached_model["history"]
            validation_models[config.label].append(metrics)
            trained[(config.label, seed)] = {
                "model": model,
                "state": state,
                "best_epoch": best_epoch,
                "history": history,
            }
            log.info(
                "[%s] %s seed=%d best_epoch=%d val FCov@100=%.2f",
                dataset, config.label, seed, best_epoch, metrics["fullcov"]["100"],
            )

    validation_summary = {
        label: _mean_metrics(metrics)
        for label, metrics in validation_models.items()
    }
    selected_label = max(
        validation_summary,
        key=lambda label: _selection_key(validation_summary[label]),
    )
    selected_config = next(config for config in configs if config.label == selected_label)
    log.info("[%s] validation-selected model: %s", dataset, selected_label)

    bm25_orders = {}
    if include_bm25:
        bm25_orders = {
            name: _bm25_order_cached(
                cache_dir / f"bm25_{name}_{MAX_CANDIDATES}.npy",
                engine,
                splits[name],
                MAX_CANDIDATES,
            )
            for name in ("val", "test")
        }

    per_seed_signals = {}
    for seed in seeds:
        model = trained[(selected_label, seed)]["model"]
        per_seed_signals[seed] = {}
        for split_name in ("val", "test"):
            queries, dense_seeds, _, _ = prepared[split_name]
            signals = {
                "dense": dense_orders[split_name],
                "offset_seed1": _multi_seed_order(
                    model, queries, dense_seeds, documents, document_tensor, index,
                    device, 1
                ),
                "offset_seed3": _multi_seed_order(
                    model, queries, dense_seeds, documents, document_tensor, index,
                    device, min(3, dense_seeds.shape[1])
                ),
            }
            if include_bm25:
                signals["bm25"] = bm25_orders[split_name]
            signals.update(partition_orders[seed][split_name])
            per_seed_signals[seed][split_name] = signals

    partition_signal_names = tuple(
        sorted(partition_orders[seeds[0]]["val"])
    )
    specs = _fusion_specs(include_bm25, partition_signal_names)
    validation_fusions = {}
    for spec in specs:
        metrics = []
        for seed in seeds:
            order = _apply_spec(spec, per_seed_signals[seed]["val"])
            metrics.append(_metrics(order, prepared["val"][2]))
        validation_fusions[spec["label"]] = _mean_metrics(metrics)
    selected_spec = max(
        specs, key=lambda spec: _selection_key(validation_fusions[spec["label"]])
    )
    log.info("[%s] validation-selected fusion: %s", dataset, selected_spec["label"])

    test_by_seed = {}
    selected_orders = {}
    for seed in seeds:
        order = _apply_spec(selected_spec, per_seed_signals[seed]["test"])
        selected_orders[seed] = order
        test_by_seed[str(seed)] = _metrics(
            order, prepared["test"][2], include_vectors=True
        )
    test_summary = _mean_metrics(list(test_by_seed.values()))

    canonical_seed = seeds[0]
    dense_test = _metrics(
        dense_orders["test"], prepared["test"][2], include_vectors=True
    )
    selected_vector = test_by_seed[str(canonical_seed)]["_fullcov_vectors"]["100"]
    dense_vector = dense_test["_fullcov_vectors"]["100"]
    baseline_only = sum(1 for a, b in zip(dense_vector, selected_vector) if a and not b)
    selected_only = sum(1 for a, b in zip(dense_vector, selected_vector) if b and not a)
    significance = {
        "comparison": f"{selected_spec['label']} vs dense",
        "budget": 100,
        "seed": canonical_seed,
        "dense_only": baseline_only,
        "selected_only": selected_only,
        "mcnemar_p_value": mcnemar_exact(baseline_only, selected_only),
    }

    for seed in seeds:
        checkpoint = {
            "dataset": dataset,
            "run_id": run_id,
            "seed": seed,
            "model_config": asdict(selected_config),
            "fusion": selected_spec,
            "model_state_dict": trained[(selected_label, seed)]["state"],
            "partition_router_state_dicts": {
                signal_set: router["state"]
                for signal_set, router in partition_routers[seed].items()
            },
            "best_epoch": trained[(selected_label, seed)]["best_epoch"],
            "embedding_dim": documents.shape[1],
            "fingerprint": fingerprint,
        }
        _atomic_torch_save(checkpoint, checkpoint_dir / f"seed_{seed}.pth")

    result_dir = Path("data") / "ukb_storage" / dataset / "results" / "L1" / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_candidates(
        result_dir / "candidates_test.jsonl",
        prepared["test"][3],
        selected_orders[canonical_seed],
        prepared["test"][2],
        [node.node_id for node in engine.nodes],
        canonical_seed,
        selected_spec["label"],
    )
    output = {
        "dataset": dataset,
        "run_id": run_id,
        "device": str(device),
        "protocol": {
            "selection_split": "val",
            "test_used_after_selection": True,
            "limit_per_split": limit,
            "split_manifest": _split_manifest(splits),
            "cache_fingerprint": fingerprint,
            "cache_manifest": fingerprint_payload,
            "cache_dir": str(cache_dir),
            "budgets": list(BUDGETS),
        },
        "training": {
            "epochs_max": epochs,
            "eval_every": eval_every,
            "patience": patience,
            "hard_negative_k": hard_negative_k,
            "seeds": list(seeds),
            "loss": (
                "uniform_multi_positive_kl + lambda_coverage * "
                "(weakest_positive_top100_barrier + "
                "0.25 * hardest_negative_margin)"
            ),
        },
        "model_candidates": [asdict(config) | {"label": config.label} for config in configs],
        "validation_models": validation_summary,
        "selected_model": asdict(selected_config) | {"label": selected_label},
        "partition_routers": {
            str(seed): {
                signal_set: {
                    "validation": router["validation"],
                    "best_epoch": router["best_epoch"],
                }
                for signal_set, router in partition_routers[seed].items()
            }
            for seed in seeds
        },
        "validation_fusions": validation_fusions,
        "selected_fusion": selected_spec,
        "test_by_seed": test_by_seed,
        "test_summary": test_summary,
        "dense_test": {k: v for k, v in dense_test.items() if not k.startswith("_")},
        "significance": significance,
        "candidate_file": str(result_dir / "candidates_test.jsonl"),
        "checkpoint_dir": str(checkpoint_dir),
    }
    _atomic_json_dump(output, result_dir / "summary.json")
    _atomic_json_dump(
        {
            f"{label}__seed{seed}": trained[(label, seed)]["history"]
            for label in validation_summary
            for seed in seeds
        },
        result_dir / "training_history.json",
    )

    log.info(
        "[%s] TEST %s FCov@100 %.2f +/- %.2f; dense %.2f; R@100 %.2f",
        dataset,
        selected_spec["label"],
        test_summary["fullcov"]["100"],
        test_summary["fullcov_std"]["100"],
        dense_test["fullcov"]["100"],
        test_summary["recall"]["100"],
    )
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validation-locked Level-1 relational candidate optimization."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["2wiki_clean", "musique_clean", "metaqa"],
    )
    parser.add_argument("--run-id", default="l1opt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--limit", type=int, default=15000)
    parser.add_argument("--heads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--coverage-lambdas", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--hard-negative-k", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-bm25", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for dataset in args.datasets:
        results[dataset] = run(
            dataset=dataset,
            run_id=args.run_id,
            epochs=args.epochs,
            limit=args.limit,
            heads=tuple(args.heads),
            coverage_lambdas=tuple(args.coverage_lambdas),
            seeds=tuple(args.seeds),
            hard_negative_k=args.hard_negative_k,
            eval_every=args.eval_every,
            patience=args.patience,
            include_bm25=not args.no_bm25,
            device=device,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    index_dir = Path("data") / "ukb_storage" / "_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / f"{args.run_id}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                dataset: {
                    "selected_model": result["selected_model"],
                    "selected_fusion": result["selected_fusion"],
                    "test_summary": result["test_summary"],
                    "dense_test": result["dense_test"],
                    "significance": result["significance"],
                }
                for dataset, result in results.items()
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
