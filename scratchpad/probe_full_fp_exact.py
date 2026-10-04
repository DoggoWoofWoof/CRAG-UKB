import modal

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

vol = modal.Volume.from_name("crag-data-volume", create_if_missing=False)
image = _base_image(modal).add_local_dir("src", remote_path="/root/CRAG/src").add_local_dir("configs", remote_path="/root/CRAG/configs").add_local_file("experiments.py", remote_path="/root/CRAG/experiments.py")

app = modal.App("crag-probe-full-fp")

@app.function(volumes={"/root/CRAG/storage": vol}, image=image, cpu=4, memory=32768, timeout=1200, gpu="A10G")
def extract():
    import os, sys
    os.chdir("/root/CRAG")
    sys.path.append("/root/CRAG")
    # Setup storage symlinks as in production
    from src.experiments.backends import _symlink_storage
    _symlink_storage(None)
    import hashlib, numpy as np, torch, gc, faiss, platform
    from src.experiments.l1_universal_head import _load
    head_datasets = sorted(["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"])
    subdir="gte_qwen"
    limit=8000; tr_cap=3000; te_cap=1
    result = {}
    per_dataset = {}
    for d in head_datasets:
        print(f"loading {d}...")
        dd = _load(d, subdir, limit, tr_cap, te_cap)
        # Use production Xt as torch tensor
        Xt = torch.tensor(dd["X"], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # For fingerprint, need Xt[:64] detached cpu
        Xt64 = np.ascontiguousarray(Xt[:64].detach().cpu().numpy())
        seeds = np.asarray(dd["train"][1])
        golds_flat = np.asarray([g for gl in dd["train"][2] for g in gl], dtype=np.int64)
        per_dataset[d] = {"seeds": seeds, "golds": golds_flat, "Xt64": Xt64, "X_shape": dd["X"].shape}
        result[d] = {
            "seeds_hash": hashlib.sha256(seeds.tobytes()).hexdigest()[:16],
            "seeds_len": len(seeds),
            "golds_hash": hashlib.sha256(golds_flat.tobytes()).hexdigest()[:16],
            "golds_len": len(golds_flat),
            "Xt64_hash": hashlib.sha256(Xt64.tobytes()).hexdigest()[:16],
            "X_shape": list(dd["X"].shape),
        }
        print(f"  {d} seeds {result[d]['seeds_hash']} golds {result[d]['golds_hash']} Xt64 {result[d]['Xt64_hash']}")
        del dd, Xt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Compute fingerprints
    for kind in ["hard","mix_hard"]:
        fp = hashlib.md5(f"{kind}|K8|e15|d1536".encode())
        for d in head_datasets:
            fp.update(d.encode())
            fp.update(np.asarray(per_dataset[d]["seeds"]).tobytes())
            fp.update(np.asarray(per_dataset[d]["golds"], dtype=np.int64).tobytes())
            fp.update(np.ascontiguousarray(per_dataset[d]["Xt64"]).tobytes())
        h = fp.hexdigest()
        print(f"{kind} {h} file head_{h[:16]}.pt")
        result[f"fp_{kind}"] = h
        result[f"fp_{kind}_file"] = f"head_{h[:16]}.pt"
    result["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "torch": torch.__version__,
    }
    print(f"versions {result['versions']}")
    # Save to volume
    import json, os as _os
    out_path = "/root/CRAG/storage/scratchpad/probe_full_fp_exact.json"
    _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result

@app.local_entrypoint()
def main():
    import json
    result = extract.remote()
    print(json.dumps(result, indent=2))
    with open("scratchpad/probe_full_fp_exact.json", "w") as f:
        json.dump(result, f, indent=2)
    print("saved scratchpad/probe_full_fp_exact.json")
