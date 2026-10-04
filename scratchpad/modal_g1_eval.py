"""G1 transfer EVAL on Modal — runs scratchpad/_g1_eval_target.py where the TARGET substrate already lives
(so we never pull the ~7GB substrate down). The frozen source backbone is shipped as a tiny artifact bundle
(results/GENERALIZATION/_g1_backbone/, uploaded to the volume) and loaded via G1_BACKBONE_DIR — no dev substrate
needed remotely. CPU only (no new encoder passes; qv/dv come from cached nodes/queries).

Run:  MODAL_PROFILE=<name> modal run scratchpad/modal_g1_eval.py --tgt squad_clean
Result: results/GENERALIZATION/_g1_eval_<tgt>.json committed to the volume; pull it down after.
"""
import modal
app = modal.App("crag-g1-eval")
volume = modal.Volume.from_name("crag-data-volume", create_if_missing=True)
image = (
    modal.Image.micromamba(python_version="3.11")
    .env({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    .apt_install("git", "build-essential")
    .pip_install("torch==2.2.1", "numpy<2.0", "scipy", "pandas", "tqdm", "xgboost", "joblib", "scikit-learn")
    .add_local_dir("src", remote_path="/root/CRAG/src")
    .add_local_dir("configs", remote_path="/root/CRAG/configs")
    .add_local_file("experiments.py", remote_path="/root/CRAG/experiments.py")
    .add_local_dir("scratchpad", remote_path="/root/CRAG/scratchpad")
)


@app.function(image=image, volumes={"/root/CRAG/storage": volume}, cpu=8.0, memory=32768, timeout=86400)
def run(tgt: str, eval_module: str = "_g1_eval_target"):
    import os, sys, shutil
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
    os.environ["G1_TGT"] = tgt
    os.environ["G1_BACKBONE_DIR"] = "results/GENERALIZATION/_g1_backbone"
    print(f"REMOTE_EVAL_STARTED tgt={tgt}", flush=True)
    assert os.path.exists("results/GENERALIZATION/_g1_backbone/base_full.joblib"), "backbone bundle missing on volume"
    assert os.path.exists(f"data/l2_corpus/{tgt}/val/cand_ids.npy"), "target substrate missing on volume"
    sys.path.insert(0, "scratchpad")
    import importlib
    EV = importlib.import_module(eval_module)
    EV.main()
    volume.commit()
    print("VOLUME_COMMITTED", flush=True)


@app.local_entrypoint()
def main(tgt: str = "squad_clean", eval_module: str = "_g1_eval_target"):
    run.remote(tgt, eval_module)
