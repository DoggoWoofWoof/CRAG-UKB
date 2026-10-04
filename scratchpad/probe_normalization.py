import modal

vol = modal.Volume.from_name("crag-data-volume", create_if_missing=False)
# Use a minimal image that matches D's faiss/torch versions for normalization
# D's base image is micromamba python 3.11 with torch 2.2.1, faiss-gpu-cu12 1.8.0.1
# For this probe we use debian_slim with faiss-cpu which should be byte-identical for L2 normalization (per-row)
# Add src for any imports if needed (not needed for this probe which only uses numpy/faiss)
image = modal.Image.debian_slim().pip_install("numpy", "faiss-cpu").add_local_dir("src", remote_path="/root/CRAG/src").add_local_dir("configs", remote_path="/root/CRAG/configs").add_local_file("experiments.py", remote_path="/root/CRAG/experiments.py")

app = modal.App("crag-probe-norm")

@app.function(volumes={"/root/CRAG/storage": vol}, image=image, cpu=2, memory=8192, timeout=300)
def probe():
    import os, sys, hashlib, numpy as np, faiss
    import platform
    REMOTE_ROOT = "/root/CRAG/storage"
    result = {}
    # versions
    result["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "torch": "not_installed",
    }
    print(f"versions {result['versions']}")
    for ds in ["2wiki_clean", "webqsp"]:
        nodes_path = os.path.join(REMOTE_ROOT, f"data/ukb_storage/{ds}/gte_qwen/nodes.npy")
        print(f"probing {ds} {nodes_path}")
        # mmap only first 64
        arr = np.load(nodes_path, mmap_mode="r")
        print(f"  shape {arr.shape} dtype {arr.dtype}")
        raw64 = np.ascontiguousarray(arr[:64].astype(np.float32))
        h_raw = hashlib.sha256(raw64.tobytes()).hexdigest()[:16]
        print(f"  raw64 hash {h_raw}")
        norm64 = raw64.copy()
        faiss.normalize_L2(norm64)
        h_norm = hashlib.sha256(norm64.tobytes()).hexdigest()[:16]
        print(f"  norm64 hash {h_norm}")
        result[ds] = {
            "raw64_hash": h_raw,
            "norm64_hash": h_norm,
            "raw64_shape": list(raw64.shape),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
        # Also check file first1M hash for reference
        with open(nodes_path, "rb") as f:
            h1m = hashlib.sha256(f.read(1048576)).hexdigest()[:16]
        result[ds]["first1M_hash"] = h1m
    return result

@app.local_entrypoint()
def main():
    import json
    result = probe.remote()
    print(json.dumps(result, indent=2))
    with open("scratchpad/probe_normalization.json", "w") as f:
        json.dump(result, f, indent=2)
    print("saved scratchpad/probe_normalization.json")
