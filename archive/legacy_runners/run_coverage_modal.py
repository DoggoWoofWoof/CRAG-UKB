"""
Modal cloud launcher for the coverage-loss Level-1 sweep.
=========================================================
Runs the same logic as `run_coverage_eval.py` (train coverage_kl over a lambda
sweep vs the frozen KL+HNM baseline, with FullCov@K / weakest_positive_rank and
a paired McNemar test) on a Modal A10G GPU, syncing data/checkpoints up and the
coverage results back down.

Usage (from repo root, with modal configured):
    python run_coverage_modal.py                                   # 2wiki+musique, full sweep
    modal run run_coverage_modal.py --datasets 2wiki --lambdas 0.5 --epochs 5 --limit 500
    modal run run_coverage_modal.py --datasets 2wiki,musique --lambdas 0.1,0.25,0.5,1.0

Only needs the frozen KL+HNM checkpoints (already on disk / the Modal volume) and
the per-dataset ukb_storage indexes. The coverage loss uses centroids only — no
ColBERT/SPLADE embeddings required.
"""
import os
import sys
import shutil
import subprocess
import logging

import modal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

app = modal.App("crag-coverage")
volume = modal.Volume.from_name("crag-data-volume", create_if_missing=True)

# Reuse the proven environment from run_hnm_eval.py (torch + PyG + faiss-gpu +
# sentence-transformers + pymetis). ColBERT/SPLADE are not needed for the
# coverage sweep, but keeping one image definition avoids drift.
image = (
    modal.Image.micromamba(python_version="3.11")
    .env({"CONDA_OVERRIDE_CUDA": "12.1", "CUDA_HOME": "/opt/conda", "TORCH_CUDA_ARCH_LIST": "8.6"})
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install("torch==2.2.1", "numpy<2.0")
    .pip_install(
        "torch-geometric==2.5.2",
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.18",
        find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html",
    )
    .pip_install(
        "networkx==3.2.1", "rank_bm25", "spacy", "pyyaml", "pandas", "tqdm",
        "sentence-transformers<3.0", "transformers<5.0.0",
    )
    .run_commands("pip uninstall -y faiss-cpu faiss-gpu")
    .pip_install("faiss-gpu-cu12==1.8.0.1")
    .micromamba_install(
        "pymetis=2022.1", "pytorch-cuda=12.1", "cuda-nvcc", "cuda-cudart-dev",
        channels=["conda-forge", "pytorch", "nvidia"],
    )
    .add_local_dir("src", remote_path="/root/CRAG/src")
    .add_local_dir("configs", remote_path="/root/CRAG/configs")
    .add_local_file("run_coverage_eval.py", remote_path="/root/CRAG/run_coverage_eval.py")
)


# ── Volume sync helpers (mirrors run_hnm_eval.py) ──
def sync_tree(src_dir, dst_dir, prefer_newer=True):
    if not os.path.exists(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        s, d = os.path.join(src_dir, item), os.path.join(dst_dir, item)
        if os.path.isdir(s):
            sync_tree(s, d, prefer_newer=prefer_newer)
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


def safe_replace_with_symlink(local_path, target_path):
    if os.path.islink(local_path):
        try:
            if os.readlink(local_path) == target_path:
                return
            os.unlink(local_path)
        except OSError:
            pass
    elif os.path.exists(local_path):
        shutil.rmtree(local_path) if os.path.isdir(local_path) else os.remove(local_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    os.symlink(target_path, local_path)


def setup_local_to_volume_symlink(local_path, remote_path):
    if os.path.exists(local_path) and not os.path.islink(local_path):
        sync_tree(local_path, remote_path)
    safe_replace_with_symlink(local_path, remote_path)


@app.function(image=image, gpu="A10G", volumes={"/root/CRAG/storage": volume}, timeout=86400)
def run_cloud_coverage(datasets, lambdas, loss, epochs, limit):
    import torch
    os.chdir("/root/CRAG")
    if "/root/CRAG" not in sys.path:
        sys.path.append("/root/CRAG")

    storage_root, project_root = "/root/CRAG/storage", "/root/CRAG"
    for sd in ["data", "checkpoints", "results"]:
        os.makedirs(os.path.join(storage_root, sd), exist_ok=True)
        setup_local_to_volume_symlink(os.path.join(project_root, sd), os.path.join(storage_root, sd))

    import run_coverage_eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Coverage sweep on {device}: datasets={datasets} lambdas={lambdas} loss={loss}")

    out = {}
    for ds in datasets:
        payload = run_coverage_eval.run_dataset(ds, lambdas, loss, epochs, limit, device)
        out[ds] = (payload or {}).get("significance", {})
        volume.commit()
    return out


def _sync_down(datasets):
    print("\n\U0001F4E6 Syncing coverage results + checkpoints back from the Modal Volume...")
    modal_bin = [sys.executable, "-m", "modal"]
    targets = ["results/coverage_ablation"]
    for d in datasets:
        targets.append(f"checkpoints/{d}/hnm_ablation")
    for target in targets:
        os.makedirs(os.path.dirname(f"./{target}"), exist_ok=True)
        subprocess.run(
            modal_bin + ["volume", "get", "crag-data-volume", target, os.path.dirname(f"./{target}"), "--force"],
            check=False, capture_output=True,
        )
    print("✅ Sync complete.")


@app.local_entrypoint()
def coverage_main(datasets: str = "2wiki,musique",
                  lambdas: str = "0.1,0.25,0.5,1.0",
                  loss: str = "coverage_kl",
                  epochs: int = 100,
                  limit: int = 0):
    ds_list = [d.strip() for d in datasets.split(",") if d.strip()]
    lam_list = [float(x) for x in lambdas.split(",") if x.strip()]
    print(f"\U0001F680 Coverage sweep -> Modal (A10G). datasets={ds_list} lambdas={lam_list} "
          f"loss={loss} epochs={epochs} limit={limit}")

    # Push data + baseline checkpoints up to the volume.
    for ds in ds_list:
        for local_dir in (f"data/ukb_storage/{ds}", f"checkpoints/{ds}/hnm_ablation"):
            if os.path.exists(local_dir):
                print(f"  ⬆️  {local_dir} -> volume")
                subprocess.run(
                    [sys.executable, "-m", "modal", "volume", "put", "crag-data-volume",
                     local_dir, local_dir],
                    capture_output=True, check=False,
                )
    if os.path.exists("data/processed"):
        subprocess.run(
            [sys.executable, "-m", "modal", "volume", "put", "crag-data-volume",
             "data/processed", "data/processed"],
            capture_output=True, check=False,
        )

    sig = run_cloud_coverage.remote(datasets=ds_list, lambdas=lam_list, loss=loss,
                                    epochs=epochs, limit=limit)
    print("✅ Cloud coverage sweep complete. Significance summary:")
    for ds, s in (sig or {}).items():
        print(f"  {ds}: {s}")
    _sync_down(ds_list)
    print("✅ Results in results/coverage_ablation/.")
