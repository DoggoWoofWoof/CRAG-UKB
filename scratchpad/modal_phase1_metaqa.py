import modal
app = modal.App("crag-phase1-metaqa")
volume = modal.Volume.from_name("crag-data-volume", create_if_missing=True)
image = (
    modal.Image.micromamba(python_version="3.11")
    .env({"CONDA_OVERRIDE_CUDA": "12.1", "CUDA_HOME": "/opt/conda", "TORCH_CUDA_ARCH_LIST": "8.6", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    .apt_install("git","build-essential")
    .pip_install("torch==2.2.1","numpy<2.0","scipy","sentence-transformers<3.0","transformers==4.44.2","faiss-gpu-cu12==1.8.0.1","rank_bm25","networkx","pyyaml","pandas","tqdm")
    .pip_install("torch-geometric==2.5.2","torch-scatter==2.1.2","torch-sparse==0.6.18", find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html")
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1%2Bcu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
    .add_local_dir("src", remote_path="/root/CRAG/src")
    .add_local_dir("configs", remote_path="/root/CRAG/configs")
    .add_local_file("experiments.py", remote_path="/root/CRAG/experiments.py")
    .add_local_dir("scratchpad", remote_path="/root/CRAG/scratchpad")
)
@app.function(image=image, gpu="A10G", volumes={"/root/CRAG/storage": volume}, timeout=86400)
def run():
    import os, sys, subprocess
    os.chdir("/root/CRAG")
    if "/root/CRAG" not in sys.path:
        sys.path.insert(0, "/root/CRAG")
    import shutil
    for sd in ["data","checkpoints","results"]:
        remote=os.path.join("/root/CRAG/storage", sd)
        local=os.path.join("/root/CRAG", sd)
        import os as _os
        _os.makedirs(remote, exist_ok=True)
        if os.path.islink(local):
            continue
        if os.path.exists(local):
            for item in os.listdir(local):
                s=os.path.join(local,item); d=os.path.join(remote,item)
                if not os.path.exists(d):
                    (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s,d)
            shutil.rmtree(local) if os.path.isdir(local) else os.remove(local)
        os.symlink(remote, local)
    os.environ["HF_HOME"]="data/ukb_storage/_models/huggingface"
    os.environ["SENTENCE_TRANSFORMERS_HOME"]="data/ukb_storage/_models/sentence_transformers"
    os.environ["TORCH_HOME"]="data/ukb_storage/_models/torch"
    print("REMOTE_FUNCTION_STARTED")
    import torch
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name={torch.cuda.get_device_name(0)}")
    print("REMOTE_ROOT=/root/CRAG")
    print(f"DATASET=metaqa")
    import subprocess, logging
    logging.basicConfig(level=logging.INFO)
    ret=subprocess.call(["python","scratchpad/phase1_l1_allq.py","--datasets","metaqa"])
    print(f"metaqa done {ret}")
    volume.commit()

@app.local_entrypoint()
def main():
    run.remote()
