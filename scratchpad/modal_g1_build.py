"""G1 target-substrate build on Modal GPU. Runs scratchpad/_g1_build_target.py for one target dataset,
assembling the frozen 5-expert C11a interface (Stages A-E), then commits the volume.

Run:  MODAL_PROFILE=<name> modal run scratchpad/modal_g1_build.py --ds squad_clean --splits train,val --train-cap 25000
Resumable: every stage skips when its output already exists on the volume, so re-running continues.
"""
import modal
app = modal.App("crag-g1-build")
volume = modal.Volume.from_name("crag-data-volume", create_if_missing=True)
image = (
    modal.Image.micromamba(python_version="3.11")
    .env({"CONDA_OVERRIDE_CUDA": "12.1", "CUDA_HOME": "/opt/conda", "TORCH_CUDA_ARCH_LIST": "8.6", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    .apt_install("git", "build-essential")
    .pip_install("torch==2.2.1", "numpy<2.0", "scipy", "sentence-transformers<3.0", "transformers==4.44.2", "faiss-gpu-cu12==1.8.0.1", "rank_bm25", "networkx", "pyyaml", "pandas", "tqdm", "xgboost", "joblib", "scikit-learn")
    .pip_install("torch-geometric==2.5.2", "torch-scatter==2.1.2", "torch-sparse==0.6.18", find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html")
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1%2Bcu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
    .add_local_dir("src", remote_path="/root/CRAG/src")
    .add_local_dir("configs", remote_path="/root/CRAG/configs")
    .add_local_file("experiments.py", remote_path="/root/CRAG/experiments.py")
    .add_local_dir("scratchpad", remote_path="/root/CRAG/scratchpad")
)


@app.function(image=image, gpu="A10G", volumes={"/root/CRAG/storage": volume}, timeout=86400)
def run(ds: str, splits: str, train_cap: int):
    import os, sys, shutil, subprocess
    os.chdir("/root/CRAG")
    if "/root/CRAG" not in sys.path:
        sys.path.insert(0, "/root/CRAG")
    for sd in ["data", "checkpoints", "results"]:
        remote = os.path.join("/root/CRAG/storage", sd); local = os.path.join("/root/CRAG", sd)
        os.makedirs(remote, exist_ok=True)
        if os.path.islink(local):
            continue
        if os.path.exists(local):
            for item in os.listdir(local):
                s = os.path.join(local, item); d = os.path.join(remote, item)
                if not os.path.exists(d):
                    (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s, d)
            shutil.rmtree(local) if os.path.isdir(local) else os.remove(local)
        os.symlink(remote, local)
    os.environ["HF_HOME"] = "data/ukb_storage/_models/huggingface"
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = "data/ukb_storage/_models/sentence_transformers"
    os.environ["TORCH_HOME"] = "data/ukb_storage/_models/torch"
    os.environ["G1_DS"] = ds; os.environ["G1_SPLITS"] = splits; os.environ["G1_TRAIN_CAP"] = str(train_cap)
    print(f"REMOTE_FUNCTION_STARTED ds={ds} splits={splits} train_cap={train_cap}", flush=True)
    import torch
    print(f"cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}", flush=True)
    sys.path.insert(0, "scratchpad")
    import _g1_build_target as G           # reads G1_* env at import; module-level DS/SPLITS/TRAIN_CAP
    man = {"dataset": ds, "splits": G.SPLITS, "train_cap": train_cap}
    for name, fn in G.STAGES:               # commit the volume after EVERY stage -> cheap resume / account-switch
        man[name] = fn()
        volume.commit()
        print(f"STAGE_COMMITTED {name}", flush=True)
    G.finalize(man)
    print("G1_BUILD_RC=0", flush=True)
    volume.commit()
    print("VOLUME_COMMITTED", flush=True)


@app.local_entrypoint()
def main(ds: str = "squad_clean", splits: str = "train,val", train_cap: int = 25000):
    run.remote(ds, splits, train_cap)
