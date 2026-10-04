"""Q2 eval on Modal (crm) — runs scratchpad/_q2_eval.py where the MetaQA val substrate (197M-row P50 corpus,
frozen backbone bundle, gte_qwen embeddings, master_nodes, kb.txt, partition_map) already lives. CPU only
(no new encoder passes: dense=cos, offset/mixture=frozen universal heads, splade=conservative, relation=abstain).

Run (SMOKE first):  MODAL_PROFILE=<crm> modal run scratchpad/modal_q2.py --m 0,32 --subset 300
     (FULL):        MODAL_PROFILE=<crm> modal run scratchpad/modal_q2.py --m 0,32,64,128,256 --subset 0
Result: results/GENERALIZATION/_q2_metaqa*.json committed to the volume; pull it down after.
"""
import modal
app = modal.App("crag-q2-eval")
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


@app.function(image=image, volumes={"/root/CRAG/storage": volume}, cpu=8.0, memory=65536, timeout=86400)
def run(m: str, subset: int, out: str, oracle: str = "0"):
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
    os.environ["Q2_M"] = m; os.environ["Q2_SUBSET"] = str(subset); os.environ["Q2_OUT"] = out
    os.environ["Q2_ORACLE"] = oracle
    os.environ["Q2_BACKBONE_DIR"] = "results/GENERALIZATION/_g1_backbone"
    print(f"REMOTE_Q2_STARTED m={m} subset={subset} out={out} oracle={oracle}", flush=True)
    for req in ["results/GENERALIZATION/_g1_backbone/C11_models.joblib",
                f"data/l2_corpus/metaqa/val/cand_ids.npy",
                "data/original/metaqa/kb.txt",
                "scratchpad/ablation_qwen/metaqa/variant_C/partition_map.json",
                "data/processed/master_nodes_metaqa.json",
                "data/ukb_storage/metaqa/gte_qwen/nodes.npy",
                "results/L2/_heads/universal_offset_src_gteqwen.pt"]:
        assert os.path.exists(req), f"MISSING PREREQ: {req}"
    sys.path.insert(0, "scratchpad")
    import importlib
    EV = importlib.import_module("_q2_eval")
    EV.main()
    volume.commit()
    print("VOLUME_COMMITTED", flush=True)


@app.local_entrypoint()
def main(m: str = "0,32,64,128,256", subset: int = 0, out: str = "results/GENERALIZATION/_q2_metaqa.json",
         oracle: str = "0"):
    run.remote(m, subset, out, oracle)
