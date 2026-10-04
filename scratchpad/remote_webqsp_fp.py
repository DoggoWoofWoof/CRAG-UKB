import modal
import os, json, hashlib

app = modal.App("crag-webqsp-fp")
vol = modal.Volume.from_name("crag-data-volume", create_if_missing=False)
# Use base image with numpy only, then run _load which will use faiss via pip faiss-cpu, torch via existing env?
# For fingerprint, we need torch and faiss, so use crag base image via backends helper
from src.experiments.backends import _modal_base_image
import modal as _modal_sdk
image = _modal_base_image(_modal_sdk).pip_install("faiss-cpu")

@app.function(volumes={"/root/CRAG/storage": vol}, image=image, gpu="A10G", timeout=600)
def extract():
    import os, json, hashlib, numpy as np, torch, faiss
    REMOTE_ROOT = "/root/CRAG/storage"
    # Load webqsp X via mmap
    import pathlib
    # Use the same logic as _load but minimal
    # Load master to get train split
    import json as js
    master_path = os.path.join(REMOTE_ROOT, "data/processed/master_nodes_webqsp.json")
    # Also check if fallback master_nodes.json is used? For webqsp, it's master_nodes_webqsp.json
    # Load nodes
    nodes_path = os.path.join(REMOTE_ROOT, "data/ukb_storage/webqsp/gte_qwen/nodes.npy")
    queries_train_path = os.path.join(REMOTE_ROOT, "data/ukb_storage/webqsp/gte_qwen/queries_train.npy")
    queries_test_path = os.path.join(REMOTE_ROOT, "data/ukb_storage/webqsp/gte_qwen/queries_test.npy")
    # For fingerprint, we need train seeds/golds and Xt64
    # Replicate _load logic for train
    # Instead of using CoreEngine, directly load master and compute splits as in _splits
    # But easier: use CoreEngine + _load with minimal overhead, but we want to avoid SPLADE/graph
    # Let's try to use the lightweight approach: load X via mmap, load queries, compute splits via master
    # For now, try to import the production _load but it will load graph etc; we can still do it on GPU host with enough RAM
    import sys
    sys.path.append("/root/CRAG")
    os.chdir("/root/CRAG")
    # Need to ensure src is on path
    from src.experiments.l1_universal_head import _load
    # Use _load with limit 8000 tr_cap 3000 te_cap 1 as production
    dd = _load("webqsp", "gte_qwen", 8000, 3000, 1)
    import torch as th
    Xt64 = np.ascontiguousarray(th.tensor(dd["X"][:64], device=th.device("cpu")).numpy())  # already normalized? dd["X"] is normalized
    # Actually dd["X"] is already normalized via load_docs_and_encoder
    # So Xt64 is normalized
    seeds = np.asarray(dd["train"][1])
    golds_flat = np.asarray([g for gl in dd["train"][2] for g in gl], dtype=np.int64)
    result = {
        "seeds_shape": list(seeds.shape),
        "seeds_dtype": str(seeds.dtype),
        "seeds_hash": hashlib.sha256(seeds.tobytes()).hexdigest()[:16],
        "seeds_len": len(seeds),
        "golds_shape": list(golds_flat.shape),
        "golds_dtype": str(golds_flat.dtype),
        "golds_hash": hashlib.sha256(golds_flat.tobytes()).hexdigest()[:16],
        "golds_len": len(golds_flat),
        "Xt64_shape": list(Xt64.shape),
        "Xt64_dtype": str(Xt64.dtype),
        "Xt64_hash": hashlib.sha256(Xt64.tobytes()).hexdigest()[:16],
        "X_shape": list(dd["X"].shape),
        "train_len": len(seeds),
        "master_path": master_path,
        "master_size": os.path.getsize(master_path) if os.path.exists(master_path) else -1,
        "nodes_size": os.path.getsize(nodes_path) if os.path.exists(nodes_path) else -1,
    }
    # Also compute raw first64 hash for comparison
    import numpy as npl
    arr_raw = np.load(nodes_path, mmap_mode="r")
    raw64 = np.ascontiguousarray(arr_raw[:64].astype(np.float32))
    result["Xt64_raw_hash"] = hashlib.sha256(raw64.tobytes()).hexdigest()[:16]
    # Save to volume for later
    out_path = os.path.join(REMOTE_ROOT, "scratchpad/fingerprint_inputs_remote_webqsp.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    # Also save actual arrays to npz for offline recompute
    npz_path = os.path.join(REMOTE_ROOT, "scratchpad/fingerprint_inputs_remote_D.npz")
    np.savez_compressed(npz_path, webqsp_seeds=seeds, webqsp_golds=golds_flat, webqsp_Xt64=Xt64)
    return result

@app.local_entrypoint()
def main():
    import json
    result = extract.remote()
    print(json.dumps(result, indent=2))
    # Save locally
    with open("scratchpad/remote_webqsp_fp.json", "w") as f:
        json.dump(result, f, indent=2)
    print("saved scratchpad/remote_webqsp_fp.json")
    # Pull the npz
    import subprocess, sys
    # Use modal volume get via CLI? The function already saved to volume, we can pull via volume get
    # Instead, just print that remote npz is at scratchpad/fingerprint_inputs_remote_D.npz on volume
    # We will pull it via modal volume get in next step
