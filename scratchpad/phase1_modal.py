import modal
app = modal.App("crag-phase1-l1")
image = modal.Image.micromamba(python_version="3.11").apt_install("git","build-essential").pip_install("torch==2.2.1","numpy<2.0","faiss-cpu","scipy","sentence-transformers<3.0","transformers==4.44.2","torch-geometric==2.5.2", find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html").pip_install("faiss-gpu-cu12==1.8.0.1").run_commands("pip uninstall -y faiss-cpu faiss-gpu").pip_install("faiss-gpu-cu12==1.8.0.1").add_local_dir("src", remote_path="/root/CRAG/src").add_local_dir("data", remote_path="/root/CRAG/data", copy=True)
# Actually data is large, use volume instead
volume = modal.Volume.from_name("crag-data-volume", create_if_missing=True)

@app.function(image=image, gpu="A10G", volumes={"/root/CRAG/data": volume}, timeout=86400)
def run_phase1(datasets: list):
    import os, sys
    sys.path.insert(0, "/root/CRAG")
    os.chdir("/root/CRAG")
    import subprocess
    cmd = ["python", "scratchpad/phase1_l1_allq.py", "--datasets"] + datasets
    print("Running", cmd)
    import logging
    logging.basicConfig(level=logging.INFO)
    ret = subprocess.call(cmd)
    print("done", ret)
    volume.commit()

@app.local_entrypoint()
def main(datasets: list):
    run_phase1.remote(datasets)
