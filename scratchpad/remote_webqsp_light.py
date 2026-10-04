import modal
vol = modal.Volume.from_name("crag-data-volume", create_if_missing=False)
image = modal.Image.debian_slim().pip_install("numpy", "faiss-cpu")

app = modal.App("crag-webqsp-light")

@app.function(volumes={"/root/CRAG/storage": vol}, image=image, cpu=4, memory=16384, timeout=600)
def extract():
    import os, json, hashlib, numpy as np, faiss
    REMOTE_ROOT = "/root/CRAG/storage"
    # Paths
    nodes_path = os.path.join(REMOTE_ROOT, "data/ukb_storage/webqsp/gte_qwen/nodes.npy")
    master_path = os.path.join(REMOTE_ROOT, "data/processed/master_nodes_webqsp.json")
    queries_train_path = os.path.join(REMOTE_ROOT, "data/ukb_storage/webqsp/gte_qwen/queries_train.npy")
    # Load X via mmap
    X = np.load(nodes_path, mmap_mode="r")
    print(f"X shape {X.shape} dtype {X.dtype}")
    # First 64 normalized hash
    # Need to normalize as production does: faiss.normalize_L2
    # For fingerprint, Xt64 is normalized X[:64]
    # Load first 64 and normalize
    X64 = np.ascontiguousarray(X[:64].astype(np.float32))
    faiss.normalize_L2(X64)
    h_Xt64 = hashlib.sha256(X64.tobytes()).hexdigest()[:16]
    print(f"Xt64 normalized hash {h_Xt64}")
    # For seeds, need queries_train
    q_train = np.load(queries_train_path, mmap_mode="r")
    print(f"q_train shape {q_train.shape}")
    # Normalize queries as well? In production, q is from eq which is already normalized via encoder (gte). The X and q are both normalized via faiss.normalize_L2 in load_docs_and_encoder.
    # For fingerprint seeds, seed = argmax(q @ X.T)
    # So we need to normalize both
    q_train_norm = np.ascontiguousarray(q_train.astype(np.float32))
    faiss.normalize_L2(q_train_norm)
    # Build faiss index for X (normalized)
    # For seeds, we need to find top1 for each train query (1104)
    # Use faiss
    X_norm = np.ascontiguousarray(X.astype(np.float32))
    faiss.normalize_L2(X_norm)
    print("building index...")
    index = faiss.IndexFlatIP(X_norm.shape[1])
    # For large X, adding all at once may be heavy but okay (4.8GB)
    # Use batch addition?
    index.add(X_norm)
    print(f"index ntotal {index.ntotal}")
    # Search top1 for train queries
    D, I = index.search(q_train_norm, 1)
    seeds = I.ravel().astype(np.int64)
    print(f"seeds shape {seeds.shape} hash {hashlib.sha256(seeds.tobytes()).hexdigest()[:16]} first3 {seeds[:3]}")
    # Golds: need master file to get train golds
    # Master file is 350MB, load and parse
    print("loading master...")
    import json as js
    # For memory, stream?
    with open(master_path, "r", encoding="utf-8") as f:
        data = js.load(f)
    print(f"master loaded {len(data)} nodes")
    # Build id2idx for docs
    docs = [n for n in data if n["metadata"].get("type") != "question"]
    id2idx = {n["node_id"]: i for i,n in enumerate(docs)}
    qs = [n for n in data if n["metadata"].get("type") == "question"]
    print(f"docs {len(docs)} qs {len(qs)}")
    # Build pairs as in _splits: pairs = list of (node_id, node, golds) where golds = neighbors in id2idx
    # But for fingerprint, we need train split as per _splits logic: fallback random 70/20/10 sorted by node_id then shuffle seed 42
    import random
    pairs = []
    for n in qs:
        golds = [nb for nb in n["neighbors"] if nb in id2idx]
        if golds:
            pairs.append((n["node_id"], n, golds))
    print(f"pairs with gold {len(pairs)}")
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    random.Random(42).shuffle(pairs_sorted)
    n = len(pairs_sorted)
    tr = int(n*0.70)
    va = tr + int(n*0.20)
    train_pairs = pairs_sorted[:tr]
    print(f"train_pairs {len(train_pairs)} tr {tr} va {va} n {n}")
    # For fingerprint, we need seeds and golds for train split only (1104)
    # But our seeds computed via faiss are for queries_train.npy which is already the train queries in order of train_pairs?
    # Need to verify order: queries_train.npy order corresponds to train_pairs order (sorted then shuffled)
    # Our seeds computed via q_train order should match train_pairs order
    # For golds, we need flattened golds for train_pairs
    golds_flat_list = []
    for _, _, golds in train_pairs:
        for g in golds:
            golds_flat_list.append(id2idx[g])
    golds_flat = np.asarray(golds_flat_list, dtype=np.int64)
    print(f"golds_flat len {len(golds_flat)} hash {hashlib.sha256(golds_flat.tobytes()).hexdigest()[:16]}")
    # Also need to ensure seeds order matches train_pairs order
    # Seeds we computed are for q_train (1104) in order of train_pairs
    # So seeds array should be length 1104, matching train_pairs
    # Our seeds computed via faiss are for q_train (1104) - correct
    # Now we have all three components for webqsp
    result = {
        "seeds_hash": hashlib.sha256(seeds.tobytes()).hexdigest()[:16],
        "seeds_len": len(seeds),
        "seeds_shape": list(seeds.shape),
        "seeds_dtype": str(seeds.dtype),
        "golds_hash": hashlib.sha256(golds_flat.tobytes()).hexdigest()[:16],
        "golds_len": len(golds_flat),
        "Xt64_hash": h_Xt64,
        "Xt64_shape": list(X64.shape),
        "X_shape": list(X.shape),
        "train_len": len(train_pairs),
    }
    # Save to volume
    out_path = os.path.join(REMOTE_ROOT, "scratchpad/remote_webqsp_light.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        js.dump(result, f, indent=2)
    # Also save arrays for offline recompute
    npz_path = os.path.join(REMOTE_ROOT, "scratchpad/fingerprint_remote_webqsp_light.npz")
    np.savez_compressed(npz_path, seeds=seeds, golds=golds_flat, Xt64=X64)
    return result

@app.local_entrypoint()
def main():
    import json
    result = extract.remote()
    print(json.dumps(result, indent=2))
    with open("scratchpad/remote_webqsp_light.json", "w") as f:
        json.dump(result, f, indent=2)
    print("saved scratchpad/remote_webqsp_light.json")
