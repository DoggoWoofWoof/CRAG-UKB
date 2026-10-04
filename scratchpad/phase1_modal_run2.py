import modal, os, sys
app = modal.App("crag-phase1-l1")
volume = modal.Volume.from_name("crag-data-volume", create_if_missing=True)

def _base_image(modal):
    return (
        modal.Image.micromamba(python_version="3.11")
        .env({"CONDA_OVERRIDE_CUDA": "12.1", "CUDA_HOME": "/opt/conda", "TORCH_CUDA_ARCH_LIST": "8.6"})
        .apt_install("git","build-essential","ninja-build")
        .pip_install("torch==2.2.1","numpy<2.0")
        .pip_install("torch-geometric==2.5.2","torch-scatter==2.1.2","torch-sparse==0.6.18", find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html")
        .pip_install("networkx==3.2.1","rank_bm25","spacy","pyyaml","pandas","tqdm","scipy","sentence-transformers<3.0","transformers==4.44.2")
        .pip_install("faiss-gpu-cu12==1.8.0.1")
        .micromamba_install("pymetis=2022.1","pytorch-cuda=12.1","cuda-nvcc","cuda-cudart-dev", channels=["conda-forge","pytorch","nvidia"])
        .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1%2Bcu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
        .run_commands("python -m spacy download en_core_web_sm")
    )

image = _base_image(modal).pip_install("xgboost==2.1.1").add_local_dir("src", remote_path="/root/CRAG/src").add_local_dir("configs", remote_path="/root/CRAG/configs").add_local_file("experiments.py", remote_path="/root/CRAG/experiments.py").add_local_dir("scratchpad", remote_path="/root/CRAG/scratchpad")

@app.function(image=image, gpu="A10G", volumes={"/root/CRAG/storage": volume}, timeout=86400)
def run(datasets: str):
    import os, sys, subprocess, shutil
    os.chdir("/root/CRAG")
    if "/root/CRAG" not in sys.path:
        sys.path.append("/root/CRAG")
    # symlink storage
    storage_root="/root/CRAG/storage"
    for sd in ["data","checkpoints","results"]:
        remote=os.path.join(storage_root, sd)
        local=os.path.join("/root/CRAG", sd)
        os.makedirs(remote, exist_ok=True)
        if os.path.islink(local):
            continue
        if os.path.exists(local):
            for item in os.listdir(local):
                s=os.path.join(local,item); d=os.path.join(remote,item)
                if not os.path.exists(d):
                    (shutil.copytree if os.path.isdir(s) else shutil.copy2)(s,d)
            shutil.rmtree(local) if os.path.isdir(local) else os.remove(local)
        os.symlink(remote, local)
    # ensure model cache on storage
    os.makedirs("data/ukb_storage/_models/huggingface", exist_ok=True)
    os.environ["HF_HOME"]="data/ukb_storage/_models/huggingface"
    os.environ["SENTENCE_TRANSFORMERS_HOME"]="data/ukb_storage/_models/sentence_transformers"
    os.environ["TORCH_HOME"]="data/ukb_storage/_models/torch"
    ds_list=datasets.split(",")
    cmd=["python","scratchpad/phase1_l1_allq.py","--datasets"]+ds_list
    print("CMD",cmd)
    import logging
    logging.basicConfig(level=logging.INFO)
    ret=subprocess.call(cmd)
    print("ret",ret)
    volume.commit()

@app.local_entrypoint()
def main(datasets: str):
    run.remote(datasets)
