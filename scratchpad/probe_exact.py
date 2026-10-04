import modal

vol = modal.Volume.from_name("crag-data-volume", create_if_missing=False)

def _base_image(modal):
    return (
        modal.Image.micromamba(python_version="3.11")
        .env({"CONDA_OVERRIDE_CUDA": "12.1", "CUDA_HOME": "/opt/conda", "TORCH_CUDA_ARCH_LIST": "8.6"})
        .apt_install("git", "build-essential", "ninja-build")
        .pip_install("torch==2.2.1", "numpy<2.0")
        .pip_install("torch-geometric==2.5.2", "torch-scatter==2.1.2", "torch-sparse==0.6.18",
                     find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html")
        .pip_install("networkx==3.2.1", "rank_bm25", "spacy", "pyyaml", "pandas", "tqdm",
                     "scipy", "sentence-transformers<3.0", "transformers==4.44.2")
        .pip_install("colbert-ai>=0.2.19", extra_options="--no-deps")
        .pip_install("ragatouille==0.0.9", "langchain<0.2")
        .run_commands("pip uninstall -y faiss-cpu faiss-gpu")
        .pip_install("faiss-gpu-cu12==1.8.0.1")
        .micromamba_install("pymetis=2022.1", "pytorch-cuda=12.1", "cuda-nvcc", "cuda-cudart-dev",
                            channels=["conda-forge", "pytorch", "nvidia"])
        .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/"
                     "flash_attn-2.5.9.post1%2Bcu122torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
        .run_commands("python -m spacy download en_core_web_sm")
    )

image = _base_image(modal).add_local_dir("src", remote_path="/root/CRAG/src").add_local_dir("configs", remote_path="/root/CRAG/configs").add_local_file("experiments.py", remote_path="/root/CRAG/experiments.py")

app = modal.App("crag-probe-exact")

@app.function(volumes={"/root/CRAG/storage": vol}, image=image, cpu=2, memory=8192, timeout=300, gpu="A10G")
def probe():
    import os, hashlib, numpy as np, faiss, torch, platform
    REMOTE_ROOT = "/root/CRAG/storage"
    result = {}
    result["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "torch": torch.__version__,
    }
    print(f"versions {result['versions']}")
    for ds in ["2wiki_clean", "webqsp"]:
        nodes_path = os.path.join(REMOTE_ROOT, f"data/ukb_storage/{ds}/gte_qwen/nodes.npy")
        arr = np.load(nodes_path, mmap_mode="r")
        raw64 = np.ascontiguousarray(arr[:64].astype(np.float32))
        h_raw = hashlib.sha256(raw64.tobytes()).hexdigest()[:16]
        norm64 = raw64.copy()
        faiss.normalize_L2(norm64)
        h_norm = hashlib.sha256(norm64.tobytes()).hexdigest()[:16]
        result[ds] = {"raw64": h_raw, "norm64": h_norm, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        print(f"{ds} raw {h_raw} norm {h_norm}")
        # Also test full matrix normalized then slice
        # For 2wiki, we can test whole matrix normalize then slice vs slice normalize
        # But for webqsp large, skip full
        if ds == "2wiki_clean":
            X_full = np.load(nodes_path).astype(np.float32)
            faiss.normalize_L2(X_full)
            h_full_slice = hashlib.sha256(np.ascontiguousarray(X_full[:64]).tobytes()).hexdigest()[:16]
            print(f"  full norm slice {h_full_slice} matches slice norm {h_norm == h_full_slice}")
            result[ds]["full_norm_slice"] = h_full_slice
    return result

@app.local_entrypoint()
def main():
    import json
    result = probe.remote()
    print(json.dumps(result, indent=2))
    with open("scratchpad/probe_exact.json", "w") as f:
        json.dump(result, f, indent=2)
