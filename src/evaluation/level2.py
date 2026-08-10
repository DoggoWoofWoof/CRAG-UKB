"""
Level-2 reranking benchmark body (backend-agnostic).
=====================================================
Lifted verbatim (compute-wise) from the old run_level2_eval.py Modal function,
with the Modal scaffolding removed: no @app.function, no volume symlinks, and
`volume.commit()` replaced by an optional `checkpoint_cb`. All encoding /
reranking / metric logic is unchanged.

Loads a frozen best Level-1 MLP, selects Top-K partitions, pools their docs into
one candidate pool per query, and reranks that identical pool with each method
(no_rerank, bm25, faiss_dense, colbert, splade). Writes
results/level_2/{ds}_level_2_reranking.{json,csv}. colbert/splade need cached
doc embeddings (data/ukb_storage/{ds}/{colbert_token_embs,splade_doc_embs}.pkl);
they are pre-encoded on first run (GPU) and cached.
"""
import os
import csv
import json
import time
import random
import pickle
import logging
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from src.core.engine import CoreEngine
from src.core.encoders import DenseEncoder
from src.alignment.mlp_encoder import TextPartitionMLP

log = logging.getLogger("experiments.level2")

SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TOP_K_PARTITIONS = 20
K_VALUES = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
COLBERT_DOC_BATCH_SIZE = 32
ALL_METHODS = ["no_rerank", "bm25", "faiss_dense", "colbert", "splade"]

BEST_MODELS = {
    "squad":   "checkpoints/squad/hnm_ablation/alignment_mlp_kl_div_tau_0.1_hnm_18.pth",
    "2wiki":   "checkpoints/2wiki/hnm_ablation/alignment_mlp_kl_div_tau_0.07_hnm_149.pth",
    "metaqa":  "checkpoints/metaqa/hnm_ablation/alignment_mlp_kl_div_tau_0.01_hnm_0.pth",
    "musique": "checkpoints/musique/hnm_ablation/alignment_mlp_kl_div_tau_0.05_hnm_33.pth",
}


def run_level2(datasets, methods=None, checkpoint_cb=None):
    """Run the Level-2 reranking benchmark for each dataset.

    Args:
        datasets: list of dataset names.
        methods: subset of ALL_METHODS (default all).
        checkpoint_cb: optional callable invoked after each incremental save
                       (the Modal backend passes volume.commit; local passes None).
    """
    import faiss
    reranker_methods = methods or list(ALL_METHODS)
    checkpoint_cb = checkpoint_cb or (lambda: None)
    all_exported_results = {}

    for ds in datasets:
        log.info(f"========== LEVEL 2 RERANKING BENCHMARK: {ds.upper()} ==========")
        logging.getLogger('src.core.engine').setLevel(logging.WARNING)
        engine = CoreEngine(source=ds)
        encoder = DenseEncoder()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = BEST_MODELS.get(ds, f"checkpoints/{ds}/alignment_mlp.pth")
        if not os.path.exists(ckpt_path):
            log.warning(f"MLP checkpoint not found: {ckpt_path}. Skipping {ds}.")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"]
        input_dim = state_dict["net.0.weight"].shape[1]
        hidden_dim = ckpt.get("hidden_dim", 256)
        output_dim = state_dict["net.3.weight"].shape[0] if "net.3.weight" in state_dict else input_dim
        mlp = TextPartitionMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
        mlp.load_state_dict(state_dict)
        mlp.eval()
        log.info(f"Loaded MLP: {input_dim}->{hidden_dim}->{output_dim}")

        # ── Build splits (identical logic to Level-1) ──
        all_pairs = []
        for node in engine.all_nodes:
            if node.metadata.get("type") == "question":
                gt_doc_ids, gt_pids = [], []
                for neighbor_id in node.neighbors:
                    pid = engine.partition_map.get(neighbor_id)
                    if pid is not None:
                        gt_pids.append(int(pid))
                        gt_doc_ids.append(neighbor_id)
                if gt_pids:
                    all_pairs.append((node.node_id, node, list(set(gt_pids)), gt_doc_ids))
        if not all_pairs:
            log.warning(f"No questions found for {ds}. Skipping.")
            continue
        all_pairs.sort(key=lambda p: p[0])
        random.Random(SPLIT_SEED).shuffle(all_pairs)
        n = len(all_pairs)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)
        splits = {
            "train": [(node, pids, doc_ids) for _, node, pids, doc_ids in all_pairs[:train_end]],
            "val":   [(node, pids, doc_ids) for _, node, pids, doc_ids in all_pairs[train_end:val_end]],
            "test":  [(node, pids, doc_ids) for _, node, pids, doc_ids in all_pairs[val_end:]],
        }
        log.info(f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

        split_embs = {}
        for split_name, queries in splits.items():
            if queries:
                split_embs[split_name] = encoder.encode([q.content for q, _, _ in queries])

        # ── Pre-encode ColBERT per-token doc embeddings (cached) ──
        colbert_cache_path = f"data/ukb_storage/{ds}/colbert_token_embs.pkl"
        colbert_doc_embs, colbert_gpu_tensor, colbert_id_to_idx = {}, None, {}
        colbert_doc_lengths_tensor = None
        if "colbert" in reranker_methods:
            if os.path.exists(colbert_cache_path):
                log.info(f"Loading cached ColBERT token embeddings from {colbert_cache_path}...")
                with open(colbert_cache_path, "rb") as f:
                    colbert_doc_embs = pickle.load(f)
            else:
                log.info("Pre-encoding ColBERT per-token doc embeddings (one-time cost)...")
                from colbert.infra import ColBERTConfig
                from colbert.modeling.checkpoint import Checkpoint
                colbert_ckpt = Checkpoint("colbert-ir/colbertv2.0",
                                          colbert_config=ColBERTConfig(doc_maxlen=256, query_maxlen=64))
                doc_nodes = [n for n in engine.all_nodes if n.metadata.get("type") != "question"]
                for batch_start in tqdm(range(0, len(doc_nodes), COLBERT_DOC_BATCH_SIZE),
                                        desc="ColBERT doc encoding"):
                    batch = doc_nodes[batch_start:batch_start + COLBERT_DOC_BATCH_SIZE]
                    with torch.no_grad():
                        embs = colbert_ckpt.docFromText([n.content[:512] for n in batch]).cpu().numpy()
                    for i, node in enumerate(batch):
                        doc_emb = embs[i]
                        norms = np.linalg.norm(doc_emb, axis=1)
                        real_len = max(1, int((norms > 1e-8).sum()))
                        colbert_doc_embs[node.node_id] = doc_emb[:real_len]
                os.makedirs(os.path.dirname(colbert_cache_path), exist_ok=True)
                with open(colbert_cache_path, "wb") as f:
                    pickle.dump(colbert_doc_embs, f)
                checkpoint_cb()
                del colbert_ckpt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            log.info("Building padded ColBERT GPU tensor for fast MaxSim...")
            node_ids_ordered = list(colbert_doc_embs.keys())
            max_td = max(e.shape[0] for e in colbert_doc_embs.values())
            dim_cb = next(iter(colbert_doc_embs.values())).shape[1]
            colbert_id_to_idx = {nid: i for i, nid in enumerate(node_ids_ordered)}
            colbert_doc_lengths = np.zeros(len(node_ids_ordered), dtype=np.int32)
            padded_np = np.zeros((len(node_ids_ordered), max_td, dim_cb), dtype=np.float32)
            for i, nid in enumerate(node_ids_ordered):
                emb = colbert_doc_embs[nid]
                padded_np[i, :emb.shape[0], :] = emb
                colbert_doc_lengths[i] = emb.shape[0]
            colbert_gpu_tensor = torch.from_numpy(padded_np).to(device)
            colbert_doc_lengths_tensor = torch.from_numpy(colbert_doc_lengths).to(device)
            del padded_np

        # ── Pre-encode SPLADE (cached) ──
        splade_cache_path = f"data/ukb_storage/{ds}/splade_doc_embs.pkl"
        splade_matrix, splade_id_to_idx = None, {}
        if "splade" in reranker_methods:
            if os.path.exists(splade_cache_path):
                log.info(f"Loading cached SPLADE sparse matrix from {splade_cache_path}...")
                with open(splade_cache_path, "rb") as f:
                    splade_data = pickle.load(f)
                    splade_matrix = splade_data["matrix"]
                    splade_id_to_idx = splade_data["id_to_idx"]
            else:
                log.info("Pre-encoding SPLADE sparse vectors (one-time cost)...")
                import scipy.sparse
                from transformers import AutoModelForMaskedLM, AutoTokenizer
                splade_tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
                splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil").to(device)
                splade_model.eval()
                doc_nodes = [n for n in engine.all_nodes if n.metadata.get("type") != "question"]
                splade_id_to_idx = {n.node_id: i for i, n in enumerate(doc_nodes)}
                rows, cols, data = [], [], []
                for batch_start in tqdm(range(0, len(doc_nodes), 64), desc="SPLADE doc encoding"):
                    batch = doc_nodes[batch_start:batch_start + 64]
                    inputs = splade_tokenizer([n.content[:1024] for n in batch], return_tensors="pt",
                                              padding=True, truncation=True, max_length=256).to(device)
                    with torch.no_grad():
                        logits = splade_model(**inputs).logits
                        relu_log = torch.log(1 + torch.relu(logits))
                        mask = inputs.attention_mask.unsqueeze(-1)
                        sparse_embs = torch.max(relu_log * mask, dim=1).values.cpu()
                    del logits, relu_log, mask, inputs
                    for i, sparse_vec in enumerate(sparse_embs):
                        nz = sparse_vec.nonzero(as_tuple=True)[0]
                        rows.extend([batch_start + i] * len(nz))
                        cols.extend(nz.numpy())
                        data.extend(sparse_vec[nz].numpy())
                vocab_size = splade_model.config.vocab_size
                splade_matrix = scipy.sparse.csr_matrix((data, (rows, cols)),
                                                        shape=(len(doc_nodes), vocab_size), dtype=np.float32)
                os.makedirs(os.path.dirname(splade_cache_path), exist_ok=True)
                with open(splade_cache_path, "wb") as f:
                    pickle.dump({"matrix": splade_matrix, "id_to_idx": splade_id_to_idx}, f)
                checkpoint_cb()
                del splade_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # ── Incremental save setup ──
        json_path = f"results/level_2/{ds}_level_2_reranking.json"
        csv_path = f"results/level_2/{ds}_level_2_reranking.csv"
        os.makedirs("results/level_2", exist_ok=True)
        all_results = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
                log.info(f"Resuming {ds}: {len(all_results)} methods found.")
            except Exception as e:
                log.warning(f"Could not load existing {json_path}: {e}")

        def safe_convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        csv_columns = ["dataset", "method", "split", "total_queries", "avg_pool_size"]
        for k in K_VALUES:
            csv_columns.extend([f"recall@{k}", f"gt_recall@{k}", f"precision@{k}", f"f1@{k}",
                                f"ndcg@{k}", f"full_coverage@{k}"])
        csv_columns.extend(["mrr", "avg_gt_docs", "min_gt_docs", "max_gt_docs", "median_gt_docs",
                            "std_gt_docs", "avg_first_hit_pos", "median_first_hit_pos",
                            "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
                            "avg_l1_latency_ms", "avg_l2_latency_ms", "p95_l2_latency_ms"])

        def _save_incremental():
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4, default=safe_convert)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
                writer.writeheader()
                for m_key, splits_d in all_results.items():
                    for s_key, metrics in splits_d.items():
                        if isinstance(metrics, dict) and "recall@1" in metrics:
                            row = {"dataset": ds, "method": m_key, "split": s_key}
                            row.update(metrics)
                            writer.writerow(row)
            checkpoint_cb()

        # ── Pre-extract all node vectors ──
        log.info("Pre-extracting all node vectors from FAISS index into numpy matrix...")
        n_vectors = engine.node_index.ntotal
        all_node_vecs = np.zeros((n_vectors, engine.node_index.d), dtype=np.float32)
        for i in range(n_vectors):
            all_node_vecs[i] = engine.node_index.reconstruct(i)

        # ── Pre-compute L1 pool per split ──
        split_l1_cache = {}
        for split_name in ["train", "val", "test"]:
            if all(split_name in all_results.get(m, {}) and "recall@1" in all_results.get(m, {}).get(split_name, {})
                   for m in reranker_methods):
                log.info(f"All methods done for [{split_name}], skipping L1 pre-computation.")
                continue
            queries = splits[split_name]
            query_embs = split_embs.get(split_name)
            if not queries or query_embs is None:
                continue
            log.info(f"Pre-computing L1 for [{split_name}] ({len(queries)} queries)...")
            l1_data = []
            for idx, (q_node, gt_pids, gt_doc_ids) in tqdm(enumerate(queries),
                                                           desc=f"L1 [{split_name}]", total=len(queries), leave=False):
                t0 = time.time()
                query_vector = query_embs[idx:idx + 1].astype("float32").copy()
                faiss.normalize_L2(query_vector)
                with torch.no_grad():
                    projected = mlp(torch.tensor(query_vector, dtype=torch.float32).to(device)).cpu().numpy()
                partition_ids = [pid for pid, _ in engine.search_centroids(projected, k=TOP_K_PARTITIONS)]
                t_l1 = time.time() - t0
                pool = []
                for pid in partition_ids:
                    pool.extend(engine.get_partition_nodes(pid))
                pool_idx_list, pool_valid = [], []
                for node in pool:
                    ni = engine.node_id_to_idx.get(node.node_id)
                    if ni is not None:
                        pool_idx_list.append(int(ni))
                        pool_valid.append(node)
                pool_fi = np.array(pool_idx_list, dtype=np.int64) if pool_idx_list else np.array([], dtype=np.int64)
                l1_data.append((query_vector, pool, pool_fi, pool_valid, t_l1, len(pool)))
            split_l1_cache[split_name] = l1_data

        # ── Rerank + metrics ──
        for method in reranker_methods:
            log.info(f"--> Benchmarking reranker: {method}")
            method_results = all_results.setdefault(method, {})
            for split_name in ["train", "val", "test"]:
                if split_name in method_results and "recall@1" in method_results[split_name]:
                    log.info(f"Skipping {method} [{split_name}] (already completed).")
                    continue
                if split_name not in split_l1_cache:
                    continue
                queries = splits[split_name]
                embs = split_embs.get(split_name)
                l1_data = split_l1_cache[split_name]
                if not queries or embs is None:
                    continue

                all_metrics = defaultdict(list)
                latencies, l1_latencies, l2_latencies, pool_sizes = [], [], [], []
                for idx, (q_node, gt_pids, gt_doc_ids) in tqdm(enumerate(queries),
                                                               desc=f"L2 {method} [{split_name}]",
                                                               total=len(queries), leave=False):
                    t0 = time.time()
                    query_vector, pool, pool_faiss_indices, pool_valid_nodes, t_l1, pool_sz = l1_data[idx]
                    pool_sizes.append(pool_sz)
                    if not pool:
                        latencies.append(time.time() - t0); l1_latencies.append(t_l1); l2_latencies.append(0.0)
                        for k in K_VALUES:
                            for m in ("recall", "gt_recall", "precision", "f1", "ndcg", "full_coverage"):
                                all_metrics[f"{m}@{k}"].append(0.0)
                        all_metrics["mrr"].append(0.0); all_metrics["first_hit_pos"].append(0)
                        all_metrics["num_gt"].append(float(len(set(gt_doc_ids))))
                        continue

                    t_l2_start = time.time()
                    if method == "splade":
                        if not hasattr(engine, '_splade_tokenizer'):
                            from transformers import AutoModelForMaskedLM, AutoTokenizer
                            engine._splade_tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
                            engine._splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil").to(device)
                            engine._splade_model.eval()
                        inputs = engine._splade_tokenizer([q_node.content], return_tensors="pt",
                                                          padding=True, truncation=True, max_length=64).to(device)
                        with torch.no_grad():
                            logits = engine._splade_model(**inputs).logits
                            relu_log = torch.log(1 + torch.relu(logits))
                            mask = inputs.attention_mask.unsqueeze(-1)
                            q_sparse = torch.max(relu_log * mask, dim=1).values.cpu().numpy()[0]
                        pool_indices, valid_nodes = [], []
                        for node in pool:
                            cb_idx = splade_id_to_idx.get(node.node_id)
                            if cb_idx is not None:
                                pool_indices.append(cb_idx); valid_nodes.append(node)
                        if valid_nodes:
                            import scipy.sparse
                            scores = splade_matrix[pool_indices].dot(
                                scipy.sparse.csr_matrix(q_sparse).T).toarray().flatten()
                            top_idx = np.argsort(-scores)[:min(max(K_VALUES), len(scores))]
                            retrieved_ids = [valid_nodes[i].node_id for i in top_idx]
                        else:
                            retrieved_ids = []
                    elif method == "faiss_dense":
                        if len(pool_faiss_indices) > 0:
                            node_vecs = all_node_vecs[pool_faiss_indices]
                            qv_flat = query_vector.flatten()
                            scores = np.dot(node_vecs, qv_flat) / (
                                (np.linalg.norm(node_vecs, axis=1) + 1e-8) * (np.linalg.norm(qv_flat) + 1e-8))
                            top_idx = np.argsort(-scores)[:min(max(K_VALUES), len(scores))]
                            retrieved_ids = [pool_valid_nodes[i].node_id for i in top_idx]
                        else:
                            retrieved_ids = []
                    elif method == "bm25":
                        tokenized_query = q_node.content.lower().split()
                        if len(pool_faiss_indices) > 0:
                            pool_bm25 = np.array(engine.bm25.get_batch_scores(tokenized_query, pool_faiss_indices.tolist()))
                            top_idx = np.argsort(-pool_bm25)[:min(max(K_VALUES), len(pool_bm25))]
                            retrieved_ids = [pool_valid_nodes[i].node_id for i in top_idx]
                        else:
                            retrieved_ids = []
                    elif method == "colbert":
                        if not hasattr(engine, '_colbert_query_ckpt'):
                            from colbert.infra import ColBERTConfig
                            from colbert.modeling.checkpoint import Checkpoint
                            engine._colbert_query_ckpt = Checkpoint(
                                "colbert-ir/colbertv2.0",
                                colbert_config=ColBERTConfig(doc_maxlen=256, query_maxlen=64))
                        with torch.no_grad():
                            q_emb_gpu = engine._colbert_query_ckpt.queryFromText([q_node.content])[0]
                            pool_indices, valid_nodes = [], []
                            for node in pool:
                                cb_idx = colbert_id_to_idx.get(node.node_id)
                                if cb_idx is not None:
                                    pool_indices.append(cb_idx); valid_nodes.append(node)
                            if valid_nodes:
                                idx_tensor = torch.tensor(pool_indices, dtype=torch.long, device=device)
                                pool_embs = colbert_gpu_tensor[idx_tensor]
                                sims = torch.einsum('qd,ntd->nqt', q_emb_gpu, pool_embs)
                                pool_lengths = colbert_doc_lengths_tensor[idx_tensor]
                                td_range = torch.arange(pool_embs.shape[1], device=device).unsqueeze(0)
                                pad_mask = td_range >= pool_lengths.unsqueeze(1)
                                sims.masked_fill_(pad_mask.unsqueeze(1), float('-inf'))
                                scores = sims.max(dim=2).values.sum(dim=1)
                                top_idx = torch.argsort(scores, descending=True)[:min(max(K_VALUES), len(scores))]
                                retrieved_ids = [valid_nodes[i].node_id for i in top_idx.cpu().tolist()]
                            else:
                                retrieved_ids = []
                    elif method == "no_rerank":
                        retrieved_ids = [n.node_id for n in pool[:max(K_VALUES)]]
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    t_l2 = time.time() - t_l2_start
                    latencies.append(time.time() - t0); l1_latencies.append(t_l1); l2_latencies.append(t_l2)

                    gt_set = set(gt_doc_ids)
                    num_gt = len(gt_set)
                    for k in K_VALUES:
                        top_k = retrieved_ids[:k]
                        hits = len(gt_set & set(top_k))
                        all_metrics[f"recall@{k}"].append(1.0 if hits > 0 else 0.0)
                        all_metrics[f"gt_recall@{k}"].append(hits / num_gt if num_gt > 0 else 0.0)
                        all_metrics[f"precision@{k}"].append(hits / k)
                        p = hits / k
                        r = hits / num_gt if num_gt > 0 else 0.0
                        all_metrics[f"f1@{k}"].append((2 * p * r / (p + r)) if (p + r) > 0 else 0.0)
                        dcg = sum(1.0 / np.log2(i + 2) for i, nid in enumerate(top_k) if nid in gt_set)
                        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_gt, k)))
                        all_metrics[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)
                    mrr = 0.0
                    for i, nid in enumerate(retrieved_ids):
                        if nid in gt_set:
                            mrr = 1.0 / (i + 1); break
                    all_metrics["mrr"].append(mrr)
                    for k in K_VALUES:
                        all_metrics[f"full_coverage@{k}"].append(
                            1.0 if gt_set.issubset(set(retrieved_ids[:k])) else 0.0)
                    first_hit = 0
                    for i, nid in enumerate(retrieved_ids):
                        if nid in gt_set:
                            first_hit = i + 1; break
                    all_metrics["first_hit_pos"].append(first_hit)
                    all_metrics["num_gt"].append(float(num_gt))

                def get_summary(metrics_dict, lat_list, l1_list, l2_list, pool_list, n_queries):
                    summary = {}
                    if n_queries == 0:
                        return summary
                    for key, vals in metrics_dict.items():
                        if key == "num_gt":
                            summary["avg_gt_docs"] = round(float(np.mean(vals)), 2)
                            summary["min_gt_docs"] = int(np.min(vals))
                            summary["max_gt_docs"] = int(np.max(vals))
                            summary["median_gt_docs"] = round(float(np.median(vals)), 1)
                            summary["std_gt_docs"] = round(float(np.std(vals)), 2)
                        elif key == "first_hit_pos":
                            summary["avg_first_hit_pos"] = round(float(np.mean(vals)), 2)
                            summary["median_first_hit_pos"] = round(float(np.median(vals)), 1)
                        else:
                            summary[key] = round(float(np.mean(vals)) * 100, 2)
                    summary["avg_latency_ms"] = round(float(np.mean(lat_list)) * 1000, 2)
                    summary["p50_latency_ms"] = round(float(np.percentile(lat_list, 50)) * 1000, 2)
                    summary["p95_latency_ms"] = round(float(np.percentile(lat_list, 95)) * 1000, 2)
                    summary["p99_latency_ms"] = round(float(np.percentile(lat_list, 99)) * 1000, 2)
                    summary["avg_l1_latency_ms"] = round(float(np.mean(l1_list)) * 1000, 2)
                    summary["avg_l2_latency_ms"] = round(float(np.mean(l2_list)) * 1000, 2)
                    summary["p95_l2_latency_ms"] = round(float(np.percentile(l2_list, 95)) * 1000, 2)
                    summary["avg_pool_size"] = round(float(np.mean(pool_list)), 1)
                    summary["total_queries"] = n_queries
                    summary["method"] = method
                    return summary

                summary = get_summary(all_metrics, latencies, l1_latencies, l2_latencies, pool_sizes, len(queries))
                method_results[split_name] = summary
                log.info(f"  [{split_name}] R@1={summary.get('recall@1', 0):.1f}% "
                         f"FCov@20={summary.get('full_coverage@20', 0):.1f}% MRR={summary.get('mrr', 0):.1f}% "
                         f"Pool={summary.get('avg_pool_size', 0):.0f}")

                if ds == "metaqa":
                    hop_indices = defaultdict(list)
                    for i, (q_node, _, _) in enumerate(queries):
                        hop_indices[str(q_node.metadata.get("hop", "unknown"))].append(i)
                    for hop in sorted(hop_indices.keys()):
                        idxs = hop_indices[hop]
                        h_summary = get_summary(
                            {k: [v[i] for i in idxs] for k, v in all_metrics.items()},
                            [latencies[i] for i in idxs], [l1_latencies[i] for i in idxs],
                            [l2_latencies[i] for i in idxs], [pool_sizes[i] for i in idxs], len(idxs))
                        method_results[f"{split_name}_hop{hop}"] = h_summary

                all_results[method] = method_results
                _save_incremental()

        all_exported_results[ds] = all_results
    return all_exported_results
