import modal
app = modal.App("crag-smoke3")
volume = modal.Volume.from_name("crag-data-volume", create_if_missing=True)
image = modal.Image.micromamba(python_version="3.11").env({"CONDA_OVERRIDE_CUDA": "12.1", "CUDA_HOME": "/opt/conda", "TORCH_CUDA_ARCH_LIST": "8.6"}).apt_install("git","build-essential").pip_install("torch==2.2.1","numpy<2.0").pip_install("faiss-gpu-cu12==1.8.0.1").add_local_dir("src", remote_path="/root/CRAG/src").add_local_dir("configs", remote_path="/root/CRAG/configs").add_local_file("experiments.py", remote_path="/root/CRAG/experiments.py").add_local_dir("scratchpad", remote_path="/root/CRAG/scratchpad")
# Use simple base without pymetis etc for smoke

@app.function(image=image, gpu="A10G", volumes={"/root/CRAG/storage": volume}, timeout=600)
def smoke():
    import os, sys, torch
    # Need to ensure src is importable
    if "/root/CRAG" not in sys.path:
        sys.path.insert(0, "/root/CRAG")
    os.chdir("/root/CRAG")
    # symlink storage
    import shutil
    for sd in ["data","checkpoints","results"]:
        remote=os.path.join("/root/CRAG/storage", sd)
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
        if not os.path.exists(local):
            os.symlink(remote, local)
    print("REMOTE_FUNCTION_STARTED")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name={torch.cuda.get_device_name(0)}")
    print(f"REMOTE_ROOT=/root/CRAG")
    print(f"volume storage path=/root/CRAG/storage exists={os.path.exists('/root/CRAG/storage')}")
    print(f"data exists={os.path.exists('data/ukb_storage/2wiki_clean/gte_qwen/nodes.npy')}")
    volume.commit()
    print("SMOKE PASS")

@app.local_entrypoint()
def main():
    smoke.remote()
