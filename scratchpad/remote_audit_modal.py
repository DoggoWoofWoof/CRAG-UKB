import modal
import os, json, hashlib, pathlib

app = modal.App("crag-audit")
vol = modal.Volume.from_name("crag-data-volume", create_if_missing=False)
image = modal.Image.debian_slim().pip_install("numpy")

@app.function(volumes={"/root/CRAG/storage": vol}, image=image, cpu=1, timeout=600)
def audit():
    import os, json, hashlib, pathlib
    import numpy as np
    REMOTE_ROOT = "/root/CRAG/storage"
    # datasets to audit
    datasets = ["2wiki_clean","hotpotqa_clean","metaqa","musique_clean","squad_clean","webqsp"]
    result = {}
    for ds in datasets:
        ds_result = {}
        # Master
        for master_path in [f"data/processed/master_nodes_{ds}.json", "data/processed/master_nodes.json"]:
            full = os.path.join(REMOTE_ROOT, master_path)
            if os.path.exists(full):
                size = os.path.getsize(full)
                # hash first 1MiB and maybe full small file
                h = hashlib.sha256()
                with open(full, "rb") as f:
                    h.update(f.read(1048576))
                ds_result[master_path] = {
                    "exists": True,
                    "size": size,
                    "sha_first1M": h.hexdigest()[:16],
                    "mtime": os.path.getmtime(full)
                }
                # For small master, also full hash if <50MB
                if size < 50*1024*1024:
                    h2 = hashlib.sha256(open(full,"rb").read()).hexdigest()[:16]
                    ds_result[master_path]["sha_full"] = h2
                break
        # Nodes
        for nodes_path in [f"data/ukb_storage/{ds}/gte_qwen/nodes.npy", f"data/ukb_storage/{ds}/nodes.npy"]:
            full = os.path.join(REMOTE_ROOT, nodes_path)
            if os.path.exists(full):
                size = os.path.getsize(full)
                # mmap to get shape/dtype without loading all
                try:
                    arr = np.load(full, mmap_mode="r")
                    shape = list(arr.shape)
                    dtype = str(arr.dtype)
                    # first 64 rows hash
                    # arr is (N,1536) float32, take first 64
                    first64 = np.ascontiguousarray(arr[:64].astype(np.float32)) if arr.dtype != np.float32 else np.ascontiguousarray(arr[:64])
                    h64 = hashlib.sha256(first64.tobytes()).hexdigest()[:16]
                    # first 1MiB hash
                    h1m = hashlib.sha256(open(full,"rb").read(1048576)).hexdigest()[:16]
                except Exception as e:
                    shape = dtype = h64 = h1m = f"error {e}"
                ds_result[nodes_path] = {
                    "exists": True,
                    "size": size,
                    "shape": shape,
                    "dtype": dtype,
                    "sha_first64": h64,
                    "sha_first1M": h1m,
                    "mtime": os.path.getmtime(full)
                }
                break
        # Queries
        for qtype in ["queries_train.npy","queries_test.npy","queries_val.npy","queries_train.npy","queries_test.npy"]:
            for base in [f"data/ukb_storage/{ds}/gte_qwen/{qtype}", f"data/ukb_storage/{ds}/{qtype}"]:
                full = os.path.join(REMOTE_ROOT, base)
                if os.path.exists(full):
                    size = os.path.getsize(full)
                    try:
                        arr = np.load(full, mmap_mode="r")
                        shape = list(arr.shape) if hasattr(arr, 'shape') else "unknown"
                        dtype = str(arr.dtype) if hasattr(arr, 'dtype') else "unknown"
                        h1m = hashlib.sha256(open(full,"rb").read(1048576)).hexdigest()[:16]
                        hfull = hashlib.sha256(open(full,"rb").read()).hexdigest()[:16] if size < 20*1024*1024 else "too_large"
                    except Exception as e:
                        shape = dtype = h1m = f"error {e}"
                        hfull = "error"
                    ds_result[base] = {
                        "exists": True,
                        "size": size,
                        "shape": shape,
                        "dtype": dtype,
                        "sha_first1M": h1m,
                        "sha_full": hfull,
                        "mtime": os.path.getmtime(full)
                    }
        # meta.json
        meta_path = f"data/ukb_storage/{ds}/gte_qwen/meta.json"
        full = os.path.join(REMOTE_ROOT, meta_path)
        if os.path.exists(full):
            size = os.path.getsize(full)
            h = hashlib.sha256(open(full,"rb").read()).hexdigest()[:16]
            ds_result[meta_path] = {"exists": True, "size": size, "sha_full": h, "mtime": os.path.getmtime(full)}
        result[ds] = ds_result
    # head cache
    head_dir = os.path.join(REMOTE_ROOT, "data/ukb_storage/_head_cache")
    heads = {}
    if os.path.exists(head_dir):
        for fname in os.listdir(head_dir):
            if fname.startswith("head_") and fname.endswith(".pt"):
                full = os.path.join(head_dir, fname)
                size = os.path.getsize(full)
                # hash first 1M and maybe full if small
                h1m = hashlib.sha256(open(full,"rb").read(1048576)).hexdigest()[:16]
                # for small heads 6MB, full hash is okay
                if size < 30*1024*1024:
                    hfull = hashlib.sha256(open(full,"rb").read()).hexdigest()[:16]
                else:
                    hfull = "large"
                heads[fname] = {"size": size, "sha_first1M": h1m, "sha_full": hfull, "mtime": os.path.getmtime(full)}
    result["_head_cache"] = heads
    # Write to volume storage for pulling?
    out_path = os.path.join(REMOTE_ROOT, "scratchpad/d_remote_artifact_audit.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result

@app.local_entrypoint()
def main():
    import json
    result = audit.remote()
    print(json.dumps(result, indent=2))
    # also save locally for reference
    with open("scratchpad/d_remote_artifact_audit.json", "w") as f:
        json.dump(result, f, indent=2)
    print("saved locally scratchpad/d_remote_artifact_audit.json")
