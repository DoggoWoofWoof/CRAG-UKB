import modal
from src.experiments.backends import _modal_base_image, MODAL_VOLUME, REMOTE_ROOT

app = modal.App("crag-phase1-l1-clean")
volume = modal.Volume.from_name(MODAL_VOLUME, create_if_missing=True)
image = _modal_base_image(modal).pip_install("xgboost==2.1.1").add_local_dir("src", remote_path=f"{REMOTE_ROOT}/src").add_local_dir("configs", remote_path=f"{REMOTE_ROOT}/configs").add_local_file("experiments.py", remote_path=f"{REMOTE_ROOT}/experiments.py").add_local_dir("scratchpad", remote_path=f"{REMOTE_ROOT}/scratchpad")

@app.function(image=image, gpu="A10G", volumes={f"{REMOTE_ROOT}/storage": volume}, timeout=86400)
def run(datasets: str):
    import os, sys, subprocess, shutil
    os.chdir(REMOTE_ROOT)
    if REMOTE_ROOT not in sys.path:
        sys.path.append(REMOTE_ROOT)
    # symlink
    import src.experiments.backends as be
    be._symlink_storage(None)
    # ensure cache dirs
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
