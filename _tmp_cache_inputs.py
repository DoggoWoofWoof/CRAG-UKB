import numpy as np, hashlib, os, json, gc
from src.experiments.l1_universal_head import _load
import torch

head_datasets_sorted = sorted(["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"])
subdir="gte_qwen"
limit=8000; tr_cap=3000; te_cap=1

cache_path="scratchpad/fingerprint_inputs_current.npz"
os.makedirs("scratchpad", exist_ok=True)

data_to_save={}
# Also save provenance
provenance={}
for d in head_datasets_sorted:
    print(f"loading {d} for cache...")
    dd=_load(d, subdir, limit, tr_cap, te_cap)
    # Use np path for Xt64 to avoid torch copy
    Xt64 = np.ascontiguousarray(dd["X"][:64].astype(np.float32))
    # Seeds and golds
    seeds = np.asarray(dd["train"][1])
    golds_flat = np.asarray([g for gl in dd["train"][2] for g in gl], dtype=np.int64)
    data_to_save[f"{d}_seeds"] = seeds
    data_to_save[f"{d}_golds"] = golds_flat
    data_to_save[f"{d}_Xt64"] = Xt64
    # provenance
    provenance[d] = {
        "X_shape": dd["X"].shape,
        "train_len": len(seeds),
        "golds_len": len(golds_flat),
        "seeds_hash": hashlib.sha256(seeds.tobytes()).hexdigest()[:16],
        "golds_hash": hashlib.sha256(golds_flat.tobytes()).hexdigest()[:16],
        "Xt64_hash": hashlib.sha256(Xt64.tobytes()).hexdigest()[:16],
        "Xt64_shape": Xt64.shape,
        "Xt64_dtype": str(Xt64.dtype)
    }
    print(f"  saved {d} seeds {seeds.shape} golds {golds_flat.shape} Xt64 {Xt64.shape}")
    del dd
    gc.collect()

# Save
np.savez_compressed(cache_path, **data_to_save)
print(f"saved cache {cache_path} keys {list(data_to_save.keys())}")

# Save provenance json
with open("scratchpad/fingerprint_inputs_current_provenance.json","w") as f:
    json.dump({
        "head_datasets_sorted": head_datasets_sorted,
        "subdir": subdir,
        "limit": limit,
        "tr_cap": tr_cap,
        "te_cap": te_cap,
        "K": 8,
        "epochs": 15,
        "dim": 1536,
        "per_dataset": provenance,
        "code_commit": os.popen("git rev-parse HEAD").read().strip(),
        "master_hashes": {d: hashlib.sha256(open(f"data/processed/master_nodes_{d}.json","rb").read(1048576)).hexdigest()[:16] if os.path.exists(f"data/processed/master_nodes_{d}.json") else "missing" for d in head_datasets_sorted},
        "nodes_hashes": {d: hashlib.sha256(open(f"data/ukb_storage/{d}/gte_qwen/nodes.npy","rb").read(1048576)).hexdigest()[:16] if os.path.exists(f"data/ukb_storage/{d}/gte_qwen/nodes.npy") else "missing" for d in head_datasets_sorted}
    }, f, indent=2)
print("provenance saved")
# Verify equivalence of torch vs np for one dataset
print("verify torch vs np equivalence for last dataset webqsp already done, skip")
