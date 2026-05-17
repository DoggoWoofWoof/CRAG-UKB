import os
import sys
import logging
import json
import csv
import time
import random
import torch
import numpy as np
import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil
import subprocess
from modal import App, Image, Volume

# 1. Define the Global Modal App (Reuses the cache of crag-benchmark)
app = modal.App("crag-benchmark")

# 2. Define the Volume for persistent data
volume = modal.Volume.from_name("crag-data-volume", create_if_missing=True)

# 3. Define the Image environment
image = (
    modal.Image.micromamba(python_version="3.11")
    .env({"CONDA_OVERRIDE_CUDA": "12.1", "CUDA_HOME": "/opt/conda", "TORCH_CUDA_ARCH_LIST": "8.6"})
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("torch==2.2.1", "numpy<2.0")
    .pip_install(
        "torch-geometric==2.5.2",
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.18",
        find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html"
    )
    .pip_install(
        "networkx==3.2.1",
        "rank_bm25",
        "spacy",
        "pyyaml",
        "pandas",
        "tqdm",
        "sentence-transformers<3.0",
        "transformers==4.47.1",
        "scipy"
    )
    .pip_install("colbert-ai>=0.2.19", extra_options="--no-deps")
    .pip_install("ragatouille==0.0.9", "langchain<0.2")
    .run_commands("pip uninstall -y faiss-cpu faiss-gpu")
    .pip_install("faiss-gpu-cu12==1.8.0.1")
    .micromamba_install(
        "pymetis=2022.1",
        "pytorch-cuda=12.1",
        "cuda-nvcc",
        "cuda-cudart-dev",
        channels=["conda-forge", "pytorch", "nvidia"]
    )
    .run_commands("python -m spacy download en_core_web_sm")
    .add_local_dir("src", remote_path="/root/CRAG/src")
    .add_local_dir("configs", remote_path="/root/CRAG/configs")
)


# ═══════════════════════════════════════════════════════════════════
# Sync Helpers (Copied securely to prevent remote import faults)
# ═══════════════════════════════════════════════════════════════════

def sync_tree(src_dir: str, dst_dir: str, logger=None, prefer_newer: bool = True):
    if not os.path.exists(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            sync_tree(s, d, logger=logger, prefer_newer=prefer_newer)
            continue
        should_copy = not os.path.exists(d)
        if not should_copy:
            try:
                if os.path.getsize(s) != os.path.getsize(d):
                    should_copy = True
                elif prefer_newer and int(os.path.getmtime(s)) >= int(os.path.getmtime(d)):
                    should_copy = True
            except OSError:
                should_copy = True
        if should_copy:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(s, d)


def safe_replace_with_symlink(local_path: str, target_path: str, logger=None):
    if os.path.islink(local_path):
        try:
            if os.readlink(local_path) == target_path:
                return
            os.unlink(local_path)
        except OSError:
            pass
    elif os.path.exists(local_path):
        if os.path.isdir(local_path): shutil.rmtree(local_path)
        else: os.remove(local_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    os.symlink(target_path, local_path)


def setup_local_to_volume_symlink(local_path: str, remote_path: str, logger=None):
    if os.path.exists(local_path) and not os.path.islink(local_path):
        sync_tree(local_path, remote_path, logger=logger, prefer_newer=True)
    safe_replace_with_symlink(local_path, remote_path, logger=logger)


def sync_data_from_volume(datasets_to_run: list):
    print("\n📦 Syncing Level 2 reranking results from Modal Volume...")
    modal_bin = [sys.executable, "-m", "modal"]
    sync_targets = []
    for d in datasets_to_run:
        sync_targets.extend([
            f"results/level_2/{d}_level_2_reranking.json",
            f"results/level_2/{d}_level_2_reranking.csv",
            f"data/ukb_storage/{d}/colbert_token_embs.pkl",
            f"data/ukb_storage/{d}/splade_doc_embs.pkl"
        ])
    try:
        os.makedirs("results/level_2", exist_ok=True)
        for target in sync_targets:
            print(f"  - Syncing {target}...")
            local_dest = f"./{target}"
            os.makedirs(os.path.dirname(local_dest), exist_ok=True)
            subprocess.run(
                modal_bin + ["volume", "get", "crag-data-volume", target, os.path.dirname(local_dest), "--force"],
                check=False, capture_output=True
            )
        print("✅ Sync complete.")
    except Exception as e:
        print(f"⚠️ Sync failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# Level 2 Benchmark Configuration
# ═══════════════════════════════════════════════════════════════════

SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

K_VALUES = [1, 3, 5, 10, 20, 50, 100, 200, 500, 1000]

RERANKER_METHODS = ["no_rerank", "bm25", "faiss_dense", "colbert", "splade"]

TOP_K_PARTITIONS = 20

# Cross-encoder is O(n) transformer forward passes per query.
# Pre-filter with FAISS dense to top-K candidates before cross-encoding.
CROSS_ENCODER_PREFILTER_K = 100

# ColBERT per-token encoding batch size (for pre-encoding all docs)
COLBERT_DOC_BATCH_SIZE = 256


# ═══════════════════════════════════════════════════════════════════
# Modal Remote Execution Function (The GPU Heavy Lifter)
# ═══════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/CRAG/storage": volume},
    timeout=86400,
)
def run_cloud_level2(datasets: list):
    log.info("🚀 Container Booted. Configuring remote workspace for Level 2 Eval...")

    os.chdir("/root/CRAG")
    if "/root/CRAG" not in sys.path:
        sys.path.append("/root/CRAG")

    storage_root = "/root/CRAG/storage"
    project_root = "/root/CRAG"

    for sd in ["data", "checkpoints", "results"]:
        remote_sd = os.path.join(storage_root, sd)
        local_sd = os.path.join(project_root, sd)
        os.makedirs(remote_sd, exist_ok=True)
        setup_local_to_volume_symlink(local_sd, remote_sd, logger=log)

    volume.commit()

    # Lazy imports inside container
    import json
    import pickle
    import faiss
    from tqdm import tqdm
    from collections import defaultdict
    from src.core.engine import CoreEngine
    from src.core.encoders import DenseEncoder
    from src.alignment.mlp_encoder import TextPartitionMLP
    from src.evaluation.benchmark_partition_selection import (
        _get_split_queries, compute_multi_gt_metrics,
        _print_recall_table, _print_detailed_metrics,
        _print_per_method_summary, _print_overall_summary,
    )

    all_exported_results = {}

    for ds in datasets:
        log.info(f"========== LEVEL 2 RERANKING BENCHMARK: {ds.upper()} ==========")

        # ── Load Engine + Encoder ─────────────────────────────────
        logging.getLogger('src.core.engine').setLevel(logging.WARNING)
        engine = CoreEngine(source=ds)
        encoder = DenseEncoder()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Load MLP Checkpoint (Best Phase 1 Model) ──────────────
        best_models = {
            "squad":   "checkpoints/squad/hnm_ablation/alignment_mlp_kl_div_tau_0.1_hnm_18.pth",
            "2wiki":   "checkpoints/2wiki/hnm_ablation/alignment_mlp_kl_div_tau_0.07_hnm_149.pth",
            "metaqa":  "checkpoints/metaqa/hnm_ablation/alignment_mlp_kl_div_tau_0.01_hnm_0.pth",
            "musique": "checkpoints/musique/hnm_ablation/alignment_mlp_kl_div_tau_0.05_hnm_33.pth",
        }
        
        ckpt_path = best_models.get(ds, f"checkpoints/{ds}/alignment_mlp.pth")
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
        log.info(f"Loaded MLP: {input_dim}→{hidden_dim}→{output_dim}")

        # ── Build Splits (identical logic to Phase 1) ─────────────
        all_pairs = []
        for node in engine.all_nodes:
            if node.metadata.get("type") == "question":
                gt_doc_ids = []
                gt_pids = []
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
        rng = random.Random(SPLIT_SEED)
        rng.shuffle(all_pairs)

        n = len(all_pairs)
        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        splits = {
            "train": [(node, pids, doc_ids) for _, node, pids, doc_ids in all_pairs[:train_end]],
            "val":   [(node, pids, doc_ids) for _, node, pids, doc_ids in all_pairs[train_end:val_end]],
            "test":  [(node, pids, doc_ids) for _, node, pids, doc_ids in all_pairs[val_end:]],
        }
        log.info(f"Splits: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

        # ── Pre-encode queries ────────────────────────────────────
        split_embs = {}
        for split_name, queries in splits.items():
            if queries:
                texts = [q.content for q, _, _ in queries]
                log.info(f"Batch encoding {len(texts)} '{split_name}' queries...")
                split_embs[split_name] = encoder.encode(texts)

        # ── Pre-encode ColBERT per-token doc embeddings (cached to disk) ──
        colbert_cache_path = f"data/ukb_storage/{ds}/colbert_token_embs.pkl"
        colbert_doc_embs = {}  # node_id -> np.ndarray (Td, 128)
        colbert_gpu_tensor = None  # Will hold (N_docs, max_Td, 128) on GPU
        colbert_id_to_idx = {}    # node_id -> index into gpu tensor

        if "colbert" in RERANKER_METHODS:
            if os.path.exists(colbert_cache_path):
                log.info(f"Loading cached ColBERT token embeddings from {colbert_cache_path}...")
                with open(colbert_cache_path, "rb") as f:
                    colbert_doc_embs = pickle.load(f)
                log.info(f"  Loaded {len(colbert_doc_embs)} cached doc embeddings.")
            else:
                log.info("Pre-encoding ColBERT per-token doc embeddings (one-time cost)...")
                from colbert.infra import ColBERTConfig
                from colbert.modeling.checkpoint import Checkpoint

                colbert_ckpt = Checkpoint(
                    "colbert-ir/colbertv2.0",
                    colbert_config=ColBERTConfig(doc_maxlen=256, query_maxlen=64)
                )

                # Get all doc nodes (non-question)
                doc_nodes = [n for n in engine.all_nodes if n.metadata.get("type") != "question"]
                log.info(f"  Encoding {len(doc_nodes)} doc nodes in batches of {COLBERT_DOC_BATCH_SIZE}...")

                for batch_start in tqdm(
                    range(0, len(doc_nodes), COLBERT_DOC_BATCH_SIZE),
                    desc="ColBERT doc encoding",
                    total=(len(doc_nodes) + COLBERT_DOC_BATCH_SIZE - 1) // COLBERT_DOC_BATCH_SIZE
                ):
                    batch = doc_nodes[batch_start:batch_start + COLBERT_DOC_BATCH_SIZE]
                    texts_batch = [n.content[:512] for n in batch]
                    # docFromText returns (batch, max_tokens, 128)
                    with torch.no_grad():
                        embs = colbert_ckpt.docFromText(texts_batch).cpu().numpy()
                    for i, node in enumerate(batch):
                        doc_emb = embs[i]  # (max_Td, 128)
                        # Strip trailing zero-padded rows to save memory
                        norms = np.linalg.norm(doc_emb, axis=1)
                        real_len = max(1, int((norms > 1e-8).sum()))
                        colbert_doc_embs[node.node_id] = doc_emb[:real_len]  # (real_Td, 128)

                # Cache to disk for future runs
                os.makedirs(os.path.dirname(colbert_cache_path), exist_ok=True)
                with open(colbert_cache_path, "wb") as f:
                    pickle.dump(colbert_doc_embs, f)
                log.info(f"  Cached {len(colbert_doc_embs)} ColBERT embeddings to {colbert_cache_path}")
                volume.commit()

                del colbert_ckpt
                torch.cuda.empty_cache()

            # Pre-build padded GPU tensor for ALL doc nodes (one-time per dataset)
            # This moves the 40B FLOP MaxSim computation from CPU→GPU (31 TFLOPS)
            log.info("Building padded ColBERT GPU tensor for fast MaxSim...")
            node_ids_ordered = list(colbert_doc_embs.keys())
            max_td = max(e.shape[0] for e in colbert_doc_embs.values())
            dim_cb = next(iter(colbert_doc_embs.values())).shape[1]  # 128
            colbert_id_to_idx = {nid: i for i, nid in enumerate(node_ids_ordered)}

            # Build padded numpy array + length tracking for proper masking
            colbert_doc_lengths = np.zeros(len(node_ids_ordered), dtype=np.int32)
            padded_np = np.zeros((len(node_ids_ordered), max_td, dim_cb), dtype=np.float32)
            for i, nid in enumerate(node_ids_ordered):
                emb = colbert_doc_embs[nid]
                padded_np[i, :emb.shape[0], :] = emb
                colbert_doc_lengths[i] = emb.shape[0]
            colbert_gpu_tensor = torch.from_numpy(padded_np).to(device)  # (N, max_Td, 128) on GPU
            colbert_doc_lengths_tensor = torch.from_numpy(colbert_doc_lengths).to(device)
            del padded_np
            log.info(f"  ColBERT GPU tensor: {colbert_gpu_tensor.shape} on {device}")

        # ── Pre-encode SPLADE (cached to disk) ──
        splade_cache_path = f"data/ukb_storage/{ds}/splade_doc_embs.pkl"
        splade_matrix = None
        splade_id_to_idx = {}
        
        if "splade" in RERANKER_METHODS:
            if os.path.exists(splade_cache_path):
                log.info(f"Loading cached SPLADE sparse matrix from {splade_cache_path}...")
                with open(splade_cache_path, "rb") as f:
                    splade_data = pickle.load(f)
                    splade_matrix = splade_data["matrix"]
                    splade_id_to_idx = splade_data["id_to_idx"]
                log.info(f"  Loaded SPLADE matrix of shape {splade_matrix.shape}")
            else:
                log.info("Pre-encoding SPLADE sparse vectors (one-time cost)...")
                import scipy.sparse
                from transformers import AutoModelForMaskedLM, AutoTokenizer
                
                splade_tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
                splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil").to(device)
                splade_model.eval()

                doc_nodes = [n for n in engine.all_nodes if n.metadata.get("type") != "question"]
                node_ids_ordered = [n.node_id for n in doc_nodes]
                splade_id_to_idx = {nid: i for i, nid in enumerate(node_ids_ordered)}
                
                rows, cols, data = [], [], []
                
                SPLADE_BATCH = 64
                for batch_start in tqdm(
                    range(0, len(doc_nodes), SPLADE_BATCH),
                    desc="SPLADE doc encoding"
                ):
                    batch = doc_nodes[batch_start:batch_start + SPLADE_BATCH]
                    texts = [n.content[:1024] for n in batch]
                    inputs = splade_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
                    with torch.no_grad():
                        logits = splade_model(**inputs).logits
                        relu_log = torch.log(1 + torch.relu(logits))
                        mask = inputs.attention_mask.unsqueeze(-1)
                        # Max pool over seq_len -> (B, vocab_size)
                        sparse_embs = torch.max(relu_log * mask, dim=1).values.cpu()
                        
                    del logits, relu_log, mask, inputs
                    
                    for i, sparse_vec in enumerate(sparse_embs):
                        nonzero_idx = sparse_vec.nonzero(as_tuple=True)[0]
                        weights = sparse_vec[nonzero_idx]
                        
                        doc_idx = batch_start + i
                        rows.extend([doc_idx] * len(nonzero_idx))
                        cols.extend(nonzero_idx.numpy())
                        data.extend(weights.numpy())
                
                vocab_size = splade_model.config.vocab_size
                splade_matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(doc_nodes), vocab_size), dtype=np.float32)
                
                os.makedirs(os.path.dirname(splade_cache_path), exist_ok=True)
                with open(splade_cache_path, "wb") as f:
                    pickle.dump({"matrix": splade_matrix, "id_to_idx": splade_id_to_idx}, f)
                log.info(f"  Cached SPLADE matrix to {splade_cache_path}")
                volume.commit()

                del splade_model
                torch.cuda.empty_cache()

        # ── Setup incremental saving ──
        json_path = f"results/level_2/{ds}_level_2_reranking.json"
        csv_path = f"results/level_2/{ds}_level_2_reranking.csv"
        
        os.makedirs("results/level_2", exist_ok=True)
        all_results = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    all_results = json.load(f)
                log.info(f"Resuming {ds} from existing checkpoint: {len(all_results)} methods found.")
            except Exception as e:
                log.warning(f"Could not load existing {json_path}: {e}")

        def safe_convert(obj):
            if isinstance(obj, (np.integer, np.floating)): return obj.item()
            elif isinstance(obj, np.ndarray): return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        # Build CSV columns dynamically based on K_VALUES
        csv_columns = ["dataset", "method", "split", "total_queries", "avg_pool_size"]
        for k in K_VALUES:
            csv_columns.extend([f"recall@{k}", f"gt_recall@{k}", f"precision@{k}", f"f1@{k}", f"ndcg@{k}", f"full_coverage@{k}"])
        csv_columns.extend([
            "mrr",
            "avg_gt_docs", "min_gt_docs", "max_gt_docs", "median_gt_docs", "std_gt_docs",
            "avg_first_hit_pos", "median_first_hit_pos",
            "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
            "avg_l1_latency_ms", "avg_l2_latency_ms", "p95_l2_latency_ms"
        ])

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
            volume.commit()

        # ── Run Level 2 Benchmark ─────────────────────────────────


        # ── Pre-extract ALL node vectors into matrix (eliminates 19k reconstruct calls) ──
        log.info("Pre-extracting all node vectors from FAISS index into numpy matrix...")
        n_vectors = engine.node_index.ntotal
        vec_dim = engine.node_index.d
        all_node_vecs = np.zeros((n_vectors, vec_dim), dtype=np.float32)
        for i in range(n_vectors):
            all_node_vecs[i] = engine.node_index.reconstruct(i)
        log.info(f"  Pre-extracted {n_vectors} vectors (shape {all_node_vecs.shape})")

        # ── Pre-compute L1 results for each split (avoid 4x redundant MLP+centroid) ──
        split_l1_cache = {}  # split_name -> list of (partition_ids, pool, pool_faiss_indices, pool_valid_nodes, t_l1)

        for split_name in ["train", "val", "test"]:
            # Check if ALL methods already completed for this split — if so, skip L1
            all_done = all(
                split_name in all_results.get(m, {}) and "recall@1" in all_results.get(m, {}).get(split_name, {})
                for m in RERANKER_METHODS
            )
            if all_done:
                log.info(f"All methods done for [{split_name}], skipping L1 pre-computation.")
                continue

            queries = splits[split_name]
            query_embs = split_embs.get(split_name)
            if not queries or query_embs is None:
                continue

            log.info(f"Pre-computing L1 (MLP partition selection) for [{split_name}] ({len(queries)} queries)...")
            l1_data = []

            for idx, (q_node, gt_pids, gt_doc_ids) in tqdm(
                enumerate(queries),
                desc=f"L1 precompute [{split_name}]",
                total=len(queries),
                leave=False
            ):
                t0 = time.time()

                # Level 1: MLP partition selection
                query_vector = query_embs[idx:idx+1].astype("float32").copy()
                faiss.normalize_L2(query_vector)

                with torch.no_grad():
                    qv = torch.tensor(query_vector, dtype=torch.float32).to(device)
                    projected = mlp(qv).cpu().numpy()

                results = engine.search_centroids(projected, k=TOP_K_PARTITIONS)
                partition_ids = [pid for pid, _ in results]
                t_l1 = time.time() - t0

                # Pool construction
                pool = []
                for pid in partition_ids:
                    pool.extend(engine.get_partition_nodes(pid))

                # Pre-compute pool indices
                pool_idx_list = []
                pool_valid = []
                for node in pool:
                    ni = engine.node_id_to_idx.get(node.node_id)
                    if ni is not None:
                        pool_idx_list.append(int(ni))
                        pool_valid.append(node)
                pool_fi = np.array(pool_idx_list, dtype=np.int64) if pool_idx_list else np.array([], dtype=np.int64)

                l1_data.append((query_vector, pool, pool_fi, pool_valid, t_l1, len(pool)))

            split_l1_cache[split_name] = l1_data
            log.info(f"  L1 precomputed for [{split_name}]: {len(l1_data)} queries cached.")

        # ── Run Level 2 Benchmark (L2 only — L1 is pre-cached) ──────

        for method in RERANKER_METHODS:
            log.info(f"--→ Benchmarking reranker: {method}")
            if method not in all_results:
                all_results[method] = {}
            method_results = all_results[method]

            for split_name in ["train", "val", "test"]:
                # RESUME LOGIC:
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
                latencies = []
                l1_latencies = []
                l2_latencies = []
                pool_sizes = []

                for idx, (q_node, gt_pids, gt_doc_ids) in tqdm(
                    enumerate(queries),
                    desc=f"L2 {method} [{split_name}]",
                    total=len(queries),
                    leave=False
                ):
                    t0 = time.time()

                    # Retrieve pre-computed L1 data
                    query_vector, pool, pool_faiss_indices, pool_valid_nodes, t_l1, pool_sz = l1_data[idx]
                    pool_sizes.append(pool_sz)

                    if not pool:
                        latencies.append(time.time() - t0)
                        l1_latencies.append(t_l1)
                        l2_latencies.append(0.0)
                        for k in K_VALUES:
                            all_metrics[f"recall@{k}"].append(0.0)
                            all_metrics[f"gt_recall@{k}"].append(0.0)
                            all_metrics[f"precision@{k}"].append(0.0)
                            all_metrics[f"f1@{k}"].append(0.0)
                            all_metrics[f"ndcg@{k}"].append(0.0)
                        all_metrics["mrr"].append(0.0)
                        for k in K_VALUES:
                            all_metrics[f"full_coverage@{k}"].append(0.0)
                        all_metrics["first_hit_pos"].append(0)
                        all_metrics["num_gt"].append(float(len(set(gt_doc_ids))))
                        continue

                    t_l2_start = time.time()

                    # Level 2: Rerank
                    if method == "splade":
                        if not hasattr(engine, '_splade_tokenizer'):
                            from transformers import AutoModelForMaskedLM, AutoTokenizer
                            engine._splade_tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
                            engine._splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil").to(device)
                            engine._splade_model.eval()
                            log.info("SPLADE query encoder loaded.")
                            
                        # Encode query
                        inputs = engine._splade_tokenizer([q_node.content], return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
                        with torch.no_grad():
                            logits = engine._splade_model(**inputs).logits
                            relu_log = torch.log(1 + torch.relu(logits))
                            mask = inputs.attention_mask.unsqueeze(-1)
                            q_sparse = torch.max(relu_log * mask, dim=1).values.cpu().numpy()[0] # (vocab_size,)
                        
                        pool_indices = []
                        valid_nodes = []
                        for node in pool:
                            cb_idx = splade_id_to_idx.get(node.node_id)
                            if cb_idx is not None:
                                pool_indices.append(cb_idx)
                                valid_nodes.append(node)
                                
                        if valid_nodes:
                            # Extract pre-computed dense matrix for pool
                            import scipy.sparse
                            pool_submatrix = splade_matrix[pool_indices] # (P, vocab_size)
                            q_sparse_csr = scipy.sparse.csr_matrix(q_sparse) # (1, vocab_size)
                            # Dot product
                            scores = pool_submatrix.dot(q_sparse_csr.T).toarray().flatten()
                            
                            top_count = min(max(K_VALUES), len(scores))
                            top_idx = np.argsort(-scores)[:top_count]
                            retrieved_ids = [valid_nodes[i].node_id for i in top_idx]
                        else:
                            retrieved_ids = []
                    elif method == "faiss_dense":
                        if len(pool_faiss_indices) > 0:
                            # Use pre-extracted vectors (no reconstruct calls)
                            node_vecs = all_node_vecs[pool_faiss_indices]
                            qv_flat = query_vector.flatten()
                            qv_norm = np.linalg.norm(qv_flat) + 1e-8
                            norms = np.linalg.norm(node_vecs, axis=1) + 1e-8
                            scores = np.dot(node_vecs, qv_flat) / (norms * qv_norm)
                            top_count = min(max(K_VALUES), len(scores))
                            top_idx = np.argsort(-scores)[:top_count]
                            retrieved_ids = [pool_valid_nodes[i].node_id for i in top_idx]
                        else:
                            retrieved_ids = []

                    elif method == "bm25":
                        # Vectorized BM25 on pool only
                        tokenized_query = q_node.content.lower().split()
                        if len(pool_faiss_indices) > 0:
                            pool_bm25 = np.array(engine.bm25.get_batch_scores(tokenized_query, pool_faiss_indices.tolist()))
                            top_count = min(max(K_VALUES), len(pool_bm25))
                            top_idx = np.argsort(-pool_bm25)[:top_count]
                            retrieved_ids = [pool_valid_nodes[i].node_id for i in top_idx]
                        else:
                            retrieved_ids = []

                    elif method == "colbert":
                        # ColBERT late-interaction: GPU-accelerated MaxSim
                        # Lazy-load the ColBERT query encoder once
                        if not hasattr(engine, '_colbert_query_ckpt'):
                            from colbert.infra import ColBERTConfig
                            from colbert.modeling.checkpoint import Checkpoint
                            engine._colbert_query_ckpt = Checkpoint(
                                "colbert-ir/colbertv2.0",
                                colbert_config=ColBERTConfig(doc_maxlen=256, query_maxlen=64)
                            )
                            log.info("ColBERT query encoder loaded.")

                        with torch.no_grad():
                            # Query encoding on GPU → keep on GPU (don't move to CPU)
                            q_emb_gpu = engine._colbert_query_ckpt.queryFromText(
                                [q_node.content]
                            )[0]  # (Tq, 128) on GPU

                            # Gather pool indices into the pre-built GPU tensor
                            pool_indices = []
                            valid_nodes = []
                            for node in pool:
                                cb_idx = colbert_id_to_idx.get(node.node_id)
                                if cb_idx is not None:
                                    pool_indices.append(cb_idx)
                                    valid_nodes.append(node)

                            if valid_nodes:
                                # Index into GPU tensor: (pool_size, max_Td, 128)
                                idx_tensor = torch.tensor(pool_indices, dtype=torch.long, device=device)
                                pool_embs = colbert_gpu_tensor[idx_tensor]  # (P, Td, 128) on GPU

                                # GPU MaxSim: einsum + max + sum — entire 40B FLOP op on A10G
                                sims = torch.einsum('qd,ntd->nqt', q_emb_gpu, pool_embs)  # (P, Tq, Td)

                                # Mask padded positions with -inf so they never win the max
                                pool_lengths = colbert_doc_lengths_tensor[idx_tensor]  # (P,)
                                max_td_pool = pool_embs.shape[1]
                                td_range = torch.arange(max_td_pool, device=device).unsqueeze(0)  # (1, Td)
                                pad_mask = td_range >= pool_lengths.unsqueeze(1)  # (P, Td)
                                sims.masked_fill_(pad_mask.unsqueeze(1), float('-inf'))  # (P, Tq, Td)

                                scores = sims.max(dim=2).values.sum(dim=1)  # (P,)

                                top_count = min(max(K_VALUES), len(scores))
                                top_idx = torch.argsort(scores, descending=True)[:top_count]
                                retrieved_ids = [valid_nodes[i] .node_id for i in top_idx.cpu().tolist()]
                            else:
                                retrieved_ids = []

                    elif method == "no_rerank":
                        retrieved_ids = [n.node_id for n in pool[:max(K_VALUES)]]

                    else:
                        raise ValueError(f"Unknown method: {method}")

                    t_l2 = time.time() - t_l2_start  # Level 2 latency only
                    latency = time.time() - t0
                    latencies.append(latency)
                    l1_latencies.append(t_l1)
                    l2_latencies.append(t_l2)

                    # Chunk-level metrics (full Phase 1 suite)
                    gt_set = set(gt_doc_ids)
                    num_gt = len(gt_set)
                    for k in K_VALUES:
                        top_k = retrieved_ids[:k]
                        top_k_set = set(top_k)
                        hits = len(gt_set & top_k_set)

                        # Recall@K: did we find at least one GT doc?
                        all_metrics[f"recall@{k}"].append(1.0 if hits > 0 else 0.0)
                        # GT Recall@K: what fraction of all GT docs did we find?
                        all_metrics[f"gt_recall@{k}"].append(hits / num_gt if num_gt > 0 else 0.0)
                        all_metrics[f"precision@{k}"].append(hits / k)
                        p = hits / k
                        r = hits / num_gt if num_gt > 0 else 0.0
                        all_metrics[f"f1@{k}"].append((2 * p * r / (p + r)) if (p + r) > 0 else 0.0)
                        dcg = sum(1.0 / np.log2(i + 2) for i, nid in enumerate(top_k) if nid in gt_set)
                        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_gt, k)))
                        all_metrics[f"ndcg@{k}"].append(dcg / idcg if idcg > 0 else 0.0)

                    # MRR
                    mrr = 0.0
                    for i, nid in enumerate(retrieved_ids):
                        if nid in gt_set:
                            mrr = 1.0 / (i + 1)
                            break
                    all_metrics["mrr"].append(mrr)

                    # Full Coverage@K: are ALL GT docs in the top K?
                    for k in K_VALUES:
                        all_metrics[f"full_coverage@{k}"].append(
                            1.0 if gt_set.issubset(set(retrieved_ids[:k])) else 0.0
                        )

                    # First Hit Position
                    first_hit = 0
                    for i, nid in enumerate(retrieved_ids):
                        if nid in gt_set:
                            first_hit = i + 1
                            break
                    all_metrics["first_hit_pos"].append(first_hit)

                    all_metrics["num_gt"].append(float(num_gt))

                # Aggregate
                summary = {}
                for key, vals in all_metrics.items():
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

                # Total latency breakdown
                summary["avg_latency_ms"] = round(float(np.mean(latencies)) * 1000, 2)
                summary["p50_latency_ms"] = round(float(np.percentile(latencies, 50)) * 1000, 2)
                summary["p95_latency_ms"] = round(float(np.percentile(latencies, 95)) * 1000, 2)
                summary["p99_latency_ms"] = round(float(np.percentile(latencies, 99)) * 1000, 2)
                # L1 vs L2 latency split
                summary["avg_l1_latency_ms"] = round(float(np.mean(l1_latencies)) * 1000, 2)
                summary["avg_l2_latency_ms"] = round(float(np.mean(l2_latencies)) * 1000, 2)
                summary["p95_l2_latency_ms"] = round(float(np.percentile(l2_latencies, 95)) * 1000, 2)
                summary["avg_pool_size"] = round(float(np.mean(pool_sizes)), 1)
                summary["total_queries"] = len(queries)
                summary["method"] = method

                method_results[split_name] = summary
                log.info(
                    f"  [{split_name}] R@1={summary['recall@1']:.1f}% R@5={summary.get('recall@5',0):.1f}% "
                    f"GTR@10={summary.get('gt_recall@10',0):.1f}% MRR={summary['mrr']:.1f}% "
                    f"FCov@20={summary.get('full_coverage@20',0):.1f}% FCov@100={summary.get('full_coverage@100',0):.1f}% "
                    f"L1={summary['avg_l1_latency_ms']:.1f}ms L2={summary['avg_l2_latency_ms']:.1f}ms "
                    f"Pool={summary['avg_pool_size']:.0f}"
                )

                # MetaQA per-hop breakdown
                if ds == "metaqa":
                    from collections import defaultdict as dd2
                    hops = dd2(list)
                    hop_emb_map = dd2(list)
                    for i, (q_node, gt_p, gt_d) in enumerate(queries):
                        hop = q_node.metadata.get("hop", "unknown")
                        hops[hop].append((q_node, gt_p, gt_d))
                        hop_emb_map[hop].append(embs[i])

                    for hop in sorted(hops.keys()):
                        log.info(f"    ↳ Hop {hop}: {len(hops[hop])} queries (per-hop eval logged)")

                all_results[method] = method_results
                _save_incremental()

        # ── Print Summary Tables ──────────────────────────────────
        W = 130
        log.info(f"\n{'═' * W}")
        log.info(f"  LEVEL 2 RERANKING BENCHMARK — {ds.upper()}")
        log.info(f"  Top-{TOP_K_PARTITIONS} partitions → Evaluate at K={K_VALUES}")
        log.info(f"{'═' * W}")
        # Key K values to display in summary table (full data in CSV)
        display_ks = [1, 5, 10, 20, 100, 1000]
        r_hdr = ' '.join([f"{'R@'+str(k):>6}" for k in display_ks])
        gtr_hdr = ' '.join([f"{'GTR@'+str(k):>7}" for k in display_ks])
        header = (
            f"  {'Method':<17} {'Split':<6} │ {r_hdr} │ "
            f"{'MRR':>5} │ {'L1':>5} {'L2':>6} {'Tot':>6} {'Pool':>5}"
        )
        log.info(header)
        log.info(f"  {'─' * (W - 2)}")

        for method, splits_data in all_results.items():
            for split_name in ["train", "val", "test"]:
                m = splits_data.get(split_name, {})
                if not m:
                    continue
                r_vals = ' '.join([f"{m.get(f'recall@{k}',0):>5.1f}%" for k in display_ks])
                log.info(
                    f"  {method:<17} {split_name:<6} │ {r_vals} │ "
                    f"{m.get('mrr',0):>4.1f}% │ "
                    f"{m.get('avg_l1_latency_ms',0):>4.1f}ms {m.get('avg_l2_latency_ms',0):>5.1f}ms "
                    f"{m.get('avg_latency_ms',0):>5.1f}ms {m.get('avg_pool_size',0):>5.0f}"
                )
            log.info(f"  {'─' * (W - 2)}")
        log.info(f"{'═' * W}")

        # Detailed metrics table (test only)
        log.info(f"\n{'═' * W}")
        log.info(f"  DETAILED METRICS (Test Split)")
        log.info(f"{'═' * W}")
        detail_ks = [1, 5, 10, 20, 100]
        p_hdr = ' '.join([f"{'P@'+str(k):>6}" for k in detail_ks])
        f1_hdr = ' '.join([f"{'F1@'+str(k):>6}" for k in detail_ks])
        ndcg_hdr = ' '.join([f"{'NDCG@'+str(k):>7}" for k in detail_ks])
        log.info(
            f"  {'Method':<17} │ {p_hdr} │ {ndcg_hdr} │ "
            f"{'AvgHit':>6} {'MedHit':>6}"
        )
        log.info(f"  {'─' * (W - 2)}")
        for method, splits_data in all_results.items():
            m = splits_data.get("test", {})
            if not m: continue
            p_vals = ' '.join([f"{m.get(f'precision@{k}',0):>5.1f}%" for k in detail_ks])
            ndcg_vals = ' '.join([f"{m.get(f'ndcg@{k}',0):>6.1f}%" for k in detail_ks])
            log.info(
                f"  {method:<17} │ {p_vals} │ {ndcg_vals} │ "
                f"{m.get('avg_first_hit_pos',0):>5.2f} {m.get('median_first_hit_pos',0):>5.1f}"
            )
        log.info(f"{'═' * W}")

        # Final save (incremental already saved per-split, this is the final commit)
        _save_incremental()
        log.info(f"Final CSV/JSON exported for {ds}")

        all_exported_results[ds] = all_results
        
        # ── Force CUDA Memory Reset Before Next Dataset ──
        log.info(f"Clearing RAM/VRAM cache for {ds} to prevent memory leaks...")
        try:
            if "colbert_gpu_tensor" in locals(): del colbert_gpu_tensor
            if "colbert_doc_lengths_tensor" in locals(): del colbert_doc_lengths_tensor
            if "colbert_doc_embs" in locals(): del colbert_doc_embs
            if "splade_matrix" in locals(): del splade_matrix
            if "engine" in locals(): del engine
            if "encoder" in locals(): del encoder
            if "mlp" in locals(): del mlp
            if "all_node_vecs" in locals(): del all_node_vecs
        except Exception:
            pass
        
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    return all_exported_results


# ═══════════════════════════════════════════════════════════════════
# Local Execution Entry Point
# ═══════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def level2_main(dataset: str = "squad"):
    print(f"🚀 Pushing C-RAG Level 2 Reranking Benchmark to Modal cloud (A10G GPU). Dataset={dataset}...")
    ds_list = ["squad", "metaqa", "musique", "2wiki"] if dataset == "all" else [dataset]

    # Upward Data Synchronization
    print("\n⏳ Ensuring your local database is completely synced UP to the Modal Volume prior to launch...")

    for ds in ds_list:
        local_dir = f"data/ukb_storage/{ds}"
        if os.path.exists(local_dir):
            print(f"  ⬆️ Pushing local {local_dir} -> Cloud Volume /data/ukb_storage/{ds}")
            subprocess.run([sys.executable, "-m", "modal", "volume", "put", "crag-data-volume", local_dir, f"data/ukb_storage/{ds}"], capture_output=True, check=False)

        ckpt_dir = f"checkpoints/{ds}/hnm_ablation"
        if os.path.exists(ckpt_dir):
            files = os.listdir(ckpt_dir)
            if any(f.endswith('.pth') for f in files):
                print(f"  ⬆️ Pushing local checkpoints {ckpt_dir} -> Cloud Volume /checkpoints/{ds}/hnm_ablation")
                subprocess.run([sys.executable, "-m", "modal", "volume", "put", "crag-data-volume", ckpt_dir, f"checkpoints/{ds}/hnm_ablation"], capture_output=True, check=False)

    if os.path.exists("data/processed"):
        print(f"  ⬆️ Pushing local data/processed -> Cloud Volume /data/processed")
        subprocess.run([sys.executable, "-m", "modal", "volume", "put", "crag-data-volume", "data/processed", "data/processed"], capture_output=True, check=False)

    print("✅ Upward Volume Synchronization complete.\n")

    # Trigger cloud execution
    results = run_cloud_level2.remote(datasets=ds_list)
    print("✅ Cloud Level 2 Benchmark Completed.")

    # Sync data locally
    sync_data_from_volume(ds_list)
    print("\n✅ Level 2 reranking results successfully mounted to local directories.")


if __name__ == "__main__":
    import subprocess
    import re

    log_filename = "pipeline_level2_reranking.log"
    cmd = [sys.executable, "-m", "modal", "run", "run_level2_eval.py"]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    dataset = "Specified from args" if "--dataset" in sys.argv else "SQuAD (default)"

    ignore_patterns = [
        r"Creating objects", r"Creating mount", r"Uploaded", r"Finalizing index",
        r"Creating function", r"Created objects", r"Initializing...",
        r"Running app", r"Worker assigned", r"Loading images",
        r"Running \(\d+/\d+ containers active\)",
        r"Created mount", r"Created function",
        r"Mounting .+", r"Connecting from Modal", r"keyboard interrupt",
        r"0/1 \[00:00", r"1/1 \[00:00", r"0/2 \[00:00", r"2/2 \[00:00",
        r"Batches:.*\b0/1\b", r"Batches:.*\b1/1\b", r"Batches:   0%\|", r"Batches: 100%\|"
    ]
    spinner_start = r"^[|/\\-]\s"

    print(f"🚀 Launching Level 2 Reranking Pipeline for {dataset}...")
    print(f"📋 Logging clean output to {log_filename}")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    with open(log_filename, "w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1, env=env
        )
        url_printed = False

        while True:
            raw_line = process.stdout.readline()
            if not raw_line:
                if process.poll() is not None: break
                continue

            clean = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', raw_line)
            segments = clean.split('\r')
            line = ''
            for seg in reversed(segments):
                if seg.strip():
                    line = seg.strip()
                    break

            if not line: continue
            if "View app at" in line or "modal.com" in line:
                if not url_printed:
                    f.write(line + '\n')
                    f.flush()
                    print(f"🔗 {line}")
                    url_printed = True
                continue

            if any(re.search(p, line) for p in ignore_patterns): continue
            if re.match(spinner_start, line): continue

            f.write(line + '\n')
            f.flush()
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))

    process.wait()
    print("\n✅ Level 2 Reranking Pipeline completed.")
