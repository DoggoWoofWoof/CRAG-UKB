import modal
vol = modal.Volume.from_name("crag-data-volume", create_if_missing=False)
image = modal.Image.debian_slim().pip_install("numpy", "faiss-cpu")

app = modal.App("crag-remote-all-fp")

@app.function(volumes={"/root/CRAG/storage": vol}, image=image, cpu=4, memory=16384, timeout=1200)
def extract_all():
    import os, sys
    os.chdir("/root/CRAG")
    sys.path.append("/root/CRAG")
    import hashlib, numpy as np, gc
    from src.experiments.l1_universal_head import _load
    head_datasets = sorted(["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"])
    subdir="gte_qwen"
    limit=8000; tr_cap=3000; te_cap=1
    result = {}
    for d in head_datasets:
        print(f"loading {d}...")
        dd = _load(d, subdir, limit, tr_cap, te_cap)
        # Use np path for Xt64
        Xt64 = np.ascontiguousarray(dd["X"][:64].astype(np.float32))
        # Normalize as production does? dd["X"] is already normalized, so just take first 64
        # But to be safe, normalize again via faiss
        import faiss
        # dd["X"] is already normalized, but we need to ensure Xt64 is normalized as in production: per_ds Xt is torch tensor of X, which is normalized, so Xt64 is normalized
        # Our dd["X"] is normalized, so Xt64 is normalized
        seeds = np.asarray(dd["train"][1])
        golds_flat = np.asarray([g for gl in dd["train"][2] for g in gl], dtype=np.int64)
        result[d] = {
            "seeds_hash": hashlib.sha256(seeds.tobytes()).hexdigest()[:16],
            "seeds_len": len(seeds),
            "golds_hash": hashlib.sha256(golds_flat.tobytes()).hexdigest()[:16],
            "golds_len": len(golds_flat),
            "Xt64_hash": hashlib.sha256(Xt64.tobytes()).hexdigest()[:16],
            "Xt64_shape": list(Xt64.shape),
            "X_shape": list(dd["X"].shape),
        }
        # Save actual arrays for recompute
        # To avoid large memory, save to npz incrementally
        print(f"  {d} seeds {result[d]['seeds_hash']} golds {result[d]['golds_hash']} Xt64 {result[d]['Xt64_hash']}")
        del dd
        gc.collect()
    # Save result json to volume
    import json
    out_path = "/root/CRAG/storage/scratchpad/remote_all_fp.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    # Also save actual arrays to npz for offline
    # Need to reload to get arrays again? We can just save from result? We need actual bytes, not just hashes
    # For now, just return result, and we will do second pass to save npz if needed
    return result

@app.local_entrypoint()
def main():
    import json
    result = extract_all.remote()
    print(json.dumps(result, indent=2))
    with open("scratchpad/remote_all_fp.json", "w") as f:
        json.dump(result, f, indent=2)
    print("saved scratchpad/remote_all_fp.json")
