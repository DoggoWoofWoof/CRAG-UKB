import os
import sys
import logging
import argparse
from tqdm import tqdm
import json
import torch
import numpy as np
import pickle
import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil
import subprocess
from modal import App, Image, Volume

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

# ═══════════════════════════════════════════════════════════════════
# Sync Helpers
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
    print("\n📦 Syncing trained HNM ablation metrics and results from Modal Volume...")
    modal_bin = [sys.executable, "-m", "modal"]
    sync_targets = []
    for d in datasets_to_run:
        sync_targets.extend([
            f"checkpoints/{d}/hnm_ablation",
            f"results/hnm_ablation/{d}_hnm_ablation_results.csv",
            f"results/hnm_ablation/comparison_{d}_hnm.json"
        ])
    try:
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("data", exist_ok=True)

        for target in sync_targets:
            print(f"  - Syncing {target}...")
            local_dest = f"./{target}"
            os.makedirs(os.path.dirname(local_dest), exist_ok=True)
            
            subprocess.run(
                modal_bin + ["volume", "get", "crag-data-volume", target, os.path.dirname(local_dest), "--force"],
                check=False,
                capture_output=True 
            )

        print("✅ Sync complete.")
    except Exception as e:
        print(f"⚠️ Sync failed: {e}")

# ═══════════════════════════════════════════════════════════════════
# Hard Negative Logic
# ═══════════════════════════════════════════════════════════════════



def get_hn_sweep(n_partitions: int) -> list[int]:
    """
    Dynamically derive meaningful hn_k geometric quartile sweeps strictly bound natively 
    by this dataset's physical partition boundary regardless of array scale.
    """
    max_hn = max(1, n_partitions - 1)
    
    if max_hn <= 3:
        return [0, max_hn]
        
    # Dynamically map [0, 25%, 50%, 75%, 100%] topological bounds securely
    sweep = {
        0, 
        max(1, int(max_hn * 0.25)),
        max(1, int(max_hn * 0.50)),
        max(1, int(max_hn * 0.75)),
        max_hn
    }
    
    return sorted(list(sweep))

LOSS = "info_nce_multi"
TAU_CONFIG = {
    "metaqa": 0.01,
    "2wiki": 0.07,
    "musique": 0.05,
    "squad": 0.1,
}

# ═══════════════════════════════════════════════════════════════════
# Modal Remote Execution Function (The GPU Heavy Lifter)
# ═══════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/root/CRAG/storage": volume},
    timeout=86400
)
def run_cloud_ablation(datasets: list):
    log.info("🚀 Container Booted. Configuring remote workspace...")
    import faiss
    if getattr(faiss, 'StandardGpuResources', None) is None:
        log.warning("WARNING: FAISS GPU is not available inside Modal execution!")
        
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
    
    from src.alignment.train_mlp import train as train_ablation
    from src.alignment.mlp_encoder import TextPartitionMLP
    from src.evaluation.benchmark_partition_selection import CoreEngine, DenseEncoder, _get_split_queries, benchmark

    all_exported_results = {}

    for ds in datasets:
        log.info(f"========== NT-XENT ABLATION TARGET: {ds.upper()} ==========")
        if ds not in TAU_CONFIG:
            log.warning(f"Configuration missing for {ds}. Skipping...")
            continue
            
        import faiss
        centroid_index = faiss.read_index(f"data/ukb_storage/{ds}/centroids.index")
        n_partitions = centroid_index.ntotal
        hn_sweep = get_hn_sweep(n_partitions)
        log.info(f"{ds.upper()}: Executing {n_partitions} partitions tracking exactly → sweeping hn_k = {hn_sweep}")
        
        # Load dataset queries and encoder outside loops as data is completely invariant organically
        log.info("--> Loading Evaluation Engine...")
        logging.getLogger('src.core.engine').setLevel(logging.WARNING)
        
        engine = CoreEngine(source=ds)
        encoder = DenseEncoder()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        split_queries = _get_split_queries(engine, dataset=ds)
        if not split_queries:
            log.warning(f"No split queries found for {ds}. Skipping eval.")
            continue
            
        split_embs = {}
        for split_name, queries in split_queries.items():
            if queries:
                texts = [q.content for q, _ in queries]
                split_embs[split_name] = encoder.encode(texts)

        all_results = {}

        for target_loss in ["info_nce_multi", "kl_div"]:
            target_tau = TAU_CONFIG[ds]
            
            # 1. Train all Hard Negative limits sequentially
            for hn_k in hn_sweep:
                log.info(f"--> Ensuring {target_loss} | tau={target_tau:g} | hn_k={hn_k} model is statically trained...")
                ckpt_path = f"checkpoints/{ds}/hnm_ablation/alignment_mlp_{target_loss}_tau_{target_tau:g}_hnm_{hn_k}.pth"
                if not os.path.exists(ckpt_path):
                    log.info(f"Model missing. Triggering training engine for hn_k={hn_k}...")
                    train_ablation(dataset_name=ds, loss_type=target_loss, tau=target_tau, hn_k=hn_k, epochs=100) 
                    volume.commit()  
                else:
                    log.info(f"Checkpoint found: {ckpt_path}. Skipping training.")

            # 2. Evaluate all boundaries inherently side-by-side
            for hn_k in hn_sweep:
                ckpt_path = f"checkpoints/{ds}/hnm_ablation/alignment_mlp_{target_loss}_tau_{target_tau:g}_hnm_{hn_k}.pth"
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                
                embed_dim = ckpt.get("input_dim", 384)
                hidden_dim = ckpt.get("hidden_dim", 512)
                
                model = TextPartitionMLP(input_dim=embed_dim, hidden_dim=hidden_dim, output_dim=embed_dim).to(device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                
                method_label = f"mlp_{target_loss}_hnm_{hn_k}"
                all_results[method_label] = {}
                
                for split_name, queries in split_queries.items():
                    if not queries: continue
                    
                    embs = split_embs[split_name]
                    res = benchmark(
                        engine=engine,
                        encoder=encoder,
                        method="mlp", 
                        queries=queries,
                        k=20,
                        model=model,
                        precomputed_embs=split_embs[split_name]
                    )
                    res["method"] = method_label
                    all_results[method_label][split_name] = res

                if ds == "metaqa":
                    from collections import defaultdict
                    import numpy as np
                    hops = defaultdict(list)
                    hop_embs = defaultdict(list)
                    
                    test_queries = split_queries.get("test", [])
                    test_embs = split_embs.get("test")
                    
                    for i, (q_node, gt) in enumerate(test_queries):
                        hop = q_node.metadata.get("hop", "unknown")
                        hops[hop].append((q_node, gt))
                        if test_embs is not None:
                            hop_embs[hop].append(test_embs[i])

                    for hop in sorted(hops.keys()):
                        h_queries = hops[hop]
                        h_embs = np.array(hop_embs[hop]) if hop_embs else None
                        h_res = benchmark(
                            engine=engine,
                            encoder=encoder,
                            method="mlp",
                            queries=h_queries,
                            k=20,
                            model=model,
                            precomputed_embs=h_embs
                        )
                        h_res["method"] = method_label
                        all_results[method_label][f"{split_name}_hop{hop}"] = h_res
                    
        # Export Grid inside Cloud
        from src.evaluation.benchmark_partition_selection import (
            _export_csv, _print_recall_table, _print_detailed_metrics, 
            _print_per_method_summary, _print_overall_summary
        )
        
        csv_path = f"results/hnm_ablation/{ds}_hnm_ablation_results.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # Flatten inherently manually to ensure hn_k column organically maps
        flattened = []
        for method, split_data in all_results.items():
            for split, metrics in split_data.items():
                row = {
                    "dataset": ds,
                    "method": method,
                    "hn_k": int(method.split("_hnm_")[1]),
                    "split": split
                }
                if "metrics" in metrics:
                    row.update(metrics["metrics"])
                else:
                    for k, v in metrics.items():
                        if k not in ["queries_evaluated", "method", "dataset"]:
                            row[k] = v
                flattened.append(row)

        import csv
        if flattened:
            keys = list(flattened[0].keys())
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(flattened)
        log.info(f"Saved custom HNM CSV explicitly to {csv_path}")
        
        json_path = f"results/hnm_ablation/comparison_{ds}_hnm.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=4)
            
        all_exported_results[ds] = all_results
        volume.commit()

    return all_exported_results

# ═══════════════════════════════════════════════════════════════════
# Local Execution Entry Point (Interacts securely with python sys.args)
# ═══════════════════════════════════════════════════════════════════
# Execution Orchestrator
# ═══════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def ablation_main(dataset: str = "2wiki"):
    print(f"🚀 Pushing C-RAG HNM tests to Modal cloud (A10G GPU). Dataset={dataset}...")
    ds_list = ["squad", "metaqa", "musique", "2wiki"] if dataset == "all" else [dataset]
    
    # User constraint: Sync local datasets UP to the modal volume aggressively prior to remote launch
    print("\n⏳ Ensuring your local database is completely synced UP to the Modal Volume prior to launch...")
    
    # Upward Data Synchronization
    for ds in ds_list:
        local_dir = f"data/ukb_storage/{ds}"
        if os.path.exists(local_dir):
            print(f"  ⬆️ Pushing local {local_dir} -> Cloud Volume /data/ukb_storage/{ds}")
            subprocess.run([sys.executable, "-m", "modal", "volume", "put", "crag-data-volume", local_dir, f"data/ukb_storage/{ds}"], capture_output=True, check=False)
            
        ckpt_dir = f"checkpoints/{ds}/hnm_ablation"
        if os.path.exists(ckpt_dir):
            files = os.listdir(ckpt_dir)
            if any(f.endswith('.pth') for f in files):
                print(f"  ⬆️ Pushing local checkpoints {ckpt_dir} -> Cloud Volume /{ckpt_dir}")
                subprocess.run([sys.executable, "-m", "modal", "volume", "put", "crag-data-volume", ckpt_dir, f"checkpoints/{ds}/hnm_ablation"], capture_output=True, check=False)
            
    if os.path.exists("data/processed"):
        print(f"  ⬆️ Pushing local data/processed -> Cloud Volume /data/processed")
        subprocess.run([sys.executable, "-m", "modal", "volume", "put", "crag-data-volume", "data/processed", "data/processed"], capture_output=True, check=False)
        
    print("✅ Upward Volume Synchronization complete.\n")

    # Trigger cloud execution
    results = run_cloud_ablation.remote(datasets=ds_list)
    print("✅ Cloud Ablation Completed.")
    
    # Sync data locally so the new ablations show up exactly on your hard drive 
    sync_data_from_volume(ds_list)
    print("\n✅ Sequenced ablation successfully mounted to local directories.")

if __name__ == "__main__":
    # If run natively via `python run_hnm_eval.py`, act as the clean pipeline wrapper
    import subprocess
    import re
    import sys
    import os

    log_filename = "pipeline_hnm_ablation.log"
    cmd = [sys.executable, "-m", "modal", "run", "run_hnm_eval.py"]
    if len(sys.argv) > 1:
        cmd.extend(sys.argv[1:])

    dataset = "Specified from args" if "--dataset" in sys.argv else "SQuAD, MuSiQue, 2Wiki, MetaQA"

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

    print(f"🚀 Launching NT-Xent Pipeline for {dataset}...")
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
    print("\n✅ Ablation Pipeline completed.")
