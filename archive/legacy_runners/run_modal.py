import modal
from modal import App, Image, Volume
import os
import sys
import json
import logging
import traceback
import shutil
import subprocess

# 1. Define the Global Modal App
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
        "transformers<5.0.0"
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

# Add checkpoints and MetaQA data if they exist locally
if os.path.exists("checkpoints"):
    image = image.add_local_dir("checkpoints", remote_path="/root/CRAG/checkpoints")

if os.path.exists("data/raw/metaqa"):
    image = image.add_local_dir("data/raw/metaqa", remote_path="/root/CRAG/metaqa_temp")


# ═══════════════════════════════════════════════════════════════════
# Sync Helpers
# ═══════════════════════════════════════════════════════════════════

def sync_tree(src_dir: str, dst_dir: str, logger=None, prefer_newer: bool = True):
    """
    Recursively sync src_dir -> dst_dir.

    Rules:
    - Copy if destination is missing
    - Copy if file size differs
    - Copy if source mtime is newer/equal than destination mtime (when prefer_newer=True)
    """
    if not os.path.exists(src_dir):
        if logger:
            logger.info(f"Source path does not exist, skipping sync: {src_dir}")
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
                src_size = os.path.getsize(s)
                dst_size = os.path.getsize(d)
                src_mtime = int(os.path.getmtime(s))
                dst_mtime = int(os.path.getmtime(d))

                if src_size != dst_size:
                    should_copy = True
                elif prefer_newer and src_mtime >= dst_mtime:
                    should_copy = True
            except OSError:
                should_copy = True

        if should_copy:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            if logger:
                logger.info(f"  Syncing {s} -> {d}")
            shutil.copy2(s, d)


def safe_replace_with_symlink(local_path: str, target_path: str, logger=None):
    """
    Replace local_path with a symlink to target_path.
    Assumes any required local->target sync has already happened.
    """
    if os.path.islink(local_path):
        try:
            current = os.readlink(local_path)
            if current == target_path:
                return
            os.unlink(local_path)
        except OSError:
            pass
    elif os.path.exists(local_path):
        if os.path.isdir(local_path):
            shutil.rmtree(local_path)
        else:
            os.remove(local_path)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    os.symlink(target_path, local_path)
    if logger:
        logger.info(f"Symlinked {local_path} -> {target_path}")


def setup_local_to_volume_symlink(local_path: str, remote_path: str, logger=None):
    """
    Sync local -> remote first, then replace local path with a symlink to remote.
    Local wins at startup.
    """
    if os.path.exists(local_path) and not os.path.islink(local_path):
        if logger:
            logger.info(f"Syncing local {local_path} -> volume {remote_path} (local overwrites cloud)")
        sync_tree(local_path, remote_path, logger=logger, prefer_newer=True)

    safe_replace_with_symlink(local_path, remote_path, logger=logger)


def sync_metaqa_temp_into_volume(temp_path: str, metaqa_volume_path: str, logger=None):
    """
    If MetaQA temp files were staged locally, sync them into persistent storage.
    """
    if os.path.exists(temp_path):
        if logger:
            logger.info(f"Syncing MetaQA temp data {temp_path} -> {metaqa_volume_path}")
        sync_tree(temp_path, metaqa_volume_path, logger=logger, prefer_newer=True)


def sync_data_from_volume(datasets_to_run: list):
    """
    Sync cloud-managed artifacts from Modal Volume to local filesystem.
    Cloud overwrites local.
    """
    print("\n📦 Syncing trained models and results from Modal Volume...")

    modal_bin = [sys.executable, "-m", "modal"]

    # Sync targets explicitly tied to datasets being run
    sync_targets = []
    for d in datasets_to_run:
        sync_targets.extend([
            f"checkpoints/{d}",
            f"results/level_1/comparison_{d}.json",
            f"results/level_1/{d}_level_1_benchmark_results.csv",
            f"data/ukb_storage/{d}",
            f"data/raw/{d}",
        ])
    
    # We always sync processed as it's shared master nodes
    sync_targets.append("data/processed")

    try:
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        for target in sync_targets:
            # check if remote path exists before dumping
            print(f"  - Syncing {target}...")
            
            # Use --force so it overwrites existing versions of *this specific* subfolder/file 
            # without replacing the parent folder which holds other datasets.
            local_dest = f"./{target}"
            os.makedirs(os.path.dirname(local_dest), exist_ok=True)
            
            subprocess.run(
                modal_bin + ["volume", "get", "crag-data-volume", target, os.path.dirname(local_dest), "--force"],
                check=False,
                capture_output=True # hide errors if remote file doesn't exist
            )

        print("✅ Sync complete.")
    except Exception as e:
        print(f"⚠️ Sync failed: {e}")


def _dataset_has_indices(dataset_name: str) -> bool:
    required = [
        f"data/ukb_storage/{dataset_name}/nodes.index",
        f"data/ukb_storage/{dataset_name}/centroids.index",
        f"data/ukb_storage/{dataset_name}/partition_map.json",
        f"data/ukb_storage/{dataset_name}/graph.pt",
    ]
    return all(os.path.exists(p) for p in required)


def _dataset_has_result(dataset_name: str) -> bool:
    return os.path.exists(f"results/level_1/comparison_{dataset_name}.json")


# 4. Define the Cloud Execution Logic
@app.function(
    image=image,
    volumes={"/root/CRAG/storage": volume},
    gpu="A10G",
    cpu=8.0,
    timeout=72000
)
def run_cloud_pipeline(fresh: bool = False, dataset: str = "all"):
    os.chdir("/root/CRAG")
    if "/root/CRAG" not in sys.path:
        sys.path.append("/root/CRAG")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("modal_runner")

    try:
        import faiss
        import torch
        logger.info(
            f"GPU Diagnostic: Torch CUDA={torch.cuda.is_available()}, "
            f"FAISS GPUs={faiss.get_num_gpus() if hasattr(faiss, 'get_num_gpus') else 'N/A'}"
        )
    except Exception as e:
        logger.warning(f"GPU Diagnostic Failed: {e}")

    storage_root = "/root/CRAG/storage"
    project_root = "/root/CRAG"

    dataset_choices = ["squad", "musique", "2wiki", "metaqa"]
    datasets_to_run = dataset_choices if dataset == "all" else [dataset]

    # Ensure the mounted volume is what the run sees first.
    subdirs = ["data", "checkpoints", "results"]
    for sd in subdirs:
        remote_sd = os.path.join(storage_root, sd)
        local_sd = os.path.join(project_root, sd)

        if fresh and os.path.exists(remote_sd):
            logger.info(f"Clearing remote storage dir for {datasets_to_run} inside {remote_sd}")
            if sd in ["checkpoints", "results"]:
                for d in datasets_to_run:
                    target_sd = os.path.join(remote_sd, d)
                    if os.path.exists(target_sd):
                        shutil.rmtree(target_sd)

            if sd == "data":
                for d in datasets_to_run:
                    target_sd = os.path.join(remote_sd, "ukb_storage", d)
                    if os.path.exists(target_sd):
                        shutil.rmtree(target_sd)

                if dataset == "all":
                    processed_dir = os.path.join(remote_sd, "processed")
                    if os.path.exists(processed_dir):
                        shutil.rmtree(processed_dir)

        os.makedirs(remote_sd, exist_ok=True)
        setup_local_to_volume_symlink(local_sd, remote_sd, logger=logger)

    metaqa_temp = os.path.join(project_root, "metaqa_temp")
    metaqa_volume = os.path.join(project_root, "data", "raw", "metaqa")
    sync_metaqa_temp_into_volume(metaqa_temp, metaqa_volume, logger=logger)

    volume.commit()

    from src.pipeline.loaders import build_unified_dataset
    from src.core.indexers import build_all
    from src.alignment.train_alignment import train as train_alignment
    from src.evaluation.benchmark_partition_selection import run_benchmark as run_partition_selection_benchmark

    all_results = {}

    try:
        master_nodes_path = "data/processed/master_nodes.json"
        master_exists = os.path.exists(master_nodes_path)

        logger.info(
            f"Checking for master nodes in cloud-mounted storage: "
            f"{os.path.abspath(master_nodes_path)} (Exists: {master_exists})"
        )

        all_indices_exist = master_exists and all(_dataset_has_indices(ds) for ds in datasets_to_run)

        if not all_indices_exist or fresh:
            logger.info(f"--- 1. BUILDING UNIFIED DATABASE ({', '.join(datasets_to_run)}) ---")
            build_unified_dataset()
            volume.commit()

            logger.info("--- 2. BUILDING FAISS/BM25 INDICES ---")
            build_all(target_datasets=datasets_to_run, force_rebuild=fresh)
            volume.commit()
        else:
            logger.info("--- 1. REUSING CLOUD DATABASE/INDICES (already present in Modal Volume) ---")

        # Add mlp_topo here
        model_types = ["mlp", "mlp_topo", "gin", "gcn", "sage"]

        for target_dataset in datasets_to_run:
            for m_type in model_types:
                ckpt_dir = f"checkpoints/{target_dataset}"
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"alignment_{m_type}.pth")

                if os.path.exists(ckpt_path) and not fresh:
                    logger.info(
                        f"--- 3. REUSING CLOUD CHECKPOINT: {m_type.upper()} for "
                        f"{target_dataset.upper()} at {ckpt_path} ---"
                    )
                    continue

                logger.info(
                    f"--- 3. TRAINING {m_type.upper()} ALIGNMENT: "
                    f"{target_dataset.upper()} (100 Epochs) ---"
                )
                train_alignment(
                    model_type=m_type,
                    dataset_name=target_dataset,
                    output_path=ckpt_path,
                    epochs=100
                )
                volume.commit()

            result_path = f"results/level_1/comparison_{target_dataset}.json"
            if os.path.exists(result_path) and not fresh:
                logger.info(
                    f"--- 4. REUSING CLOUD BENCHMARK RESULT: {result_path} ---"
                )
                with open(result_path, "r") as f:
                    all_results[target_dataset] = json.load(f)
                continue

            logger.info(f"--- 4. COMPARING RECALL (Level 1 & 2): {target_dataset} ---")
            metrics = run_partition_selection_benchmark(dataset=target_dataset)

            if metrics:
                all_results[target_dataset] = metrics
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                with open(result_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                volume.commit()

        volume.commit()
        logger.info("--- 5. ALL BENCHMARKS COMPLETED SUCCESSFULLY ---")
        return all_results

    except Exception as e:
        logger.error(f"❌ Cloud Execution Error: {e}")
        traceback.print_exc()
        try:
            volume.commit()
        except Exception:
            pass
        return {"error": str(e)}


# 5. Define the Local Entry Point
@app.local_entrypoint()
def main(fresh: bool = False, dataset: str = "all"):
    print(f"🚀 Pushing C-RAG benchmark to Modal cloud. Dataset={dataset}...")

    dataset_choices = ["squad", "musique", "2wiki", "metaqa"]
    datasets_to_run = dataset_choices if dataset == "all" else [dataset]

    if fresh:
        print(f"🧹 Fresh start requested! Clearing local stale cached directories for dataset '{dataset}'...")
        for base_dir in ["checkpoints", "results", "data/ukb_storage"]:
            for d in datasets_to_run:
                target_dir = os.path.join(base_dir, d)
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                    print(f"   Deleted {target_dir}")

        if dataset == "all" and os.path.exists("data/processed"):
            shutil.rmtree("data/processed")
            print("   Deleted data/processed")

        print("🧹 Modal persistent volume dataset cache will also be cleared!")

    try:
        results = run_cloud_pipeline.remote(fresh=fresh, dataset=dataset)

        if results and "error" not in results:
            print("\n" + "=" * 50)
            print("🎯 FINAL RECALL METRICS (Level 1 & 2)")
            print("=" * 50)
            for ds, metrics in results.items():
                if isinstance(metrics, dict):
                    print(f"\n[{ds.upper()}]")
                    for k, v in metrics.items():
                        print(f"  {k}: {v}")
            print("=" * 50)

        sync_data_from_volume(datasets_to_run)

    except Exception as e:
        print(f"❌ Execution Failure: {e}")
        traceback.print_exc()