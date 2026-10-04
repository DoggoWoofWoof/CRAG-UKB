import modal
vol = modal.Volume.from_name("crag-data-volume", create_if_missing=False)
image = modal.Image.debian_slim().pip_install("numpy", "faiss-cpu")

app = modal.App("crag-remote-all-light")

@app.function(volumes={"/root/CRAG/storage": vol}, image=image, cpu=4, memory=16384, timeout=1200)
def extract_all():
    import os, json, hashlib, numpy as np, faiss, random, gc
    REMOTE_ROOT = "/root/CRAG/storage"
    datasets = ["2wiki_clean","hotpotqa_clean","metaqa","musique_clean","squad_clean","webqsp"]
    result = {}
    for ds in sorted(datasets):
        print(f"processing {ds}...")
        # Master path
        master_candidates = [f"data/processed/master_nodes_{ds}.json", "data/processed/master_nodes.json"]
        master_path = None
        for cand in master_candidates:
            full = os.path.join(REMOTE_ROOT, cand)
            if os.path.exists(full):
                master_path = full
                break
        if not master_path:
            result[ds] = {"error": "no master"}
            continue
        # Load X
        x_candidates = [f"data/ukb_storage/{ds}/gte_qwen/nodes.npy", f"data/ukb_storage/{ds}/nodes.npy"]
        x_path = None
        for cand in x_candidates:
            full = os.path.join(REMOTE_ROOT, cand)
            if os.path.exists(full):
                x_path = full
                break
        if not x_path:
            result[ds] = {"error": "no X"}
            continue
        # Load queries_train
        q_candidates = [f"data/ukb_storage/{ds}/gte_qwen/queries_train.npy", f"data/ukb_storage/{ds}/queries_train.npy"]
        q_path = None
        for cand in q_candidates:
            full = os.path.join(REMOTE_ROOT, cand)
            if os.path.exists(full):
                q_path = full
                break
        if not q_path:
            # try alternative: queries_train may be in gte_qwen
            result[ds] = {"error": "no q_train"}
            continue
        # Load master json (stream for large)
        # For memory, use json.load (may be large for webqsp 350MB, but okay with 16GB)
        print(f"  loading master {master_path}...")
        with open(master_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = [n for n in data if n["metadata"].get("type") != "question"]
        id2idx = {n["node_id"]: i for i,n in enumerate(docs)}
        qs = [n for n in data if n["metadata"].get("type") == "question"]
        print(f"  docs {len(docs)} qs {len(qs)}")
        # Build pairs
        pairs = []
        for n in qs:
            golds = [nb for nb in n["neighbors"] if nb in id2idx]
            if golds:
                pairs.append((n["node_id"], n, golds))
        pairs_sorted = sorted(pairs, key=lambda x: x[0])
        random.Random(42).shuffle(pairs_sorted)
        n = len(pairs_sorted)
        # Apply production caps: tr_cap=3000, limit=8000
        tr_cap = 3000
        limit = 8000
        caps_train = tr_cap or limit
        # In production, caps = {"train": tr_cap or limit, "test": te_cap or limit}
        # For head training, te_cap=1, but train cap is 3000
        tr = min(int(n*0.70), caps_train)
        train_pairs = pairs_sorted[:tr]
        print(f"  train_pairs {len(train_pairs)} (n {n} 70% {int(n*0.70)} capped {caps_train})")
        # Load X and q for fingerprint
        print(f"  loading X {x_path}...")
        X = np.load(x_path, mmap_mode="r")
        X_shape = list(X.shape)
        # Xt64: first 64 rows, normalized
        # X is already normalized in production, but to be safe, normalize the slice
        Xt64_raw = np.ascontiguousarray(X[:64].astype(np.float32))
        # Normalize
        faiss.normalize_L2(Xt64_raw)
        h_Xt64 = hashlib.sha256(Xt64_raw.tobytes()).hexdigest()[:16]
        # For seeds, need q_train
        print(f"  loading q {q_path}...")
        q_train = np.load(q_path, mmap_mode="r")
        # q_train is already normalized, but normalize again
        q_train_norm = np.ascontiguousarray(q_train[:len(train_pairs)].astype(np.float32))
        if q_train_norm.shape[0] != len(train_pairs):
            print(f"  WARNING q_train {q_train_norm.shape} vs train_pairs {len(train_pairs)} - truncating")
            q_train_norm = q_train_norm[:len(train_pairs)]
        faiss.normalize_L2(q_train_norm)
        # Build index for X (need normalized X)
        # For seeds, we need full X normalized
        # Use mmap X and normalize in chunks? For now, load full X as float32 and normalize (may be heavy for webqsp 4.8GB)
        # Instead, use faiss index with X_norm
        print(f"  building index for {ds} X {X_shape}...")
        # For large X, we need to handle memory: X is mmap, we can add in chunks
        dim = X.shape[1]
        index = faiss.IndexFlatIP(dim)
        # Add in chunks to avoid large copy
        # Normalize X in chunks and add
        # For simplicity, use numpy for small datasets, and chunked for large
        if X_shape[0] > 100000:
            # chunked
            chunk = 50000
            for start in range(0, X_shape[0], chunk):
                end = min(start+chunk, X_shape[0])
                chunk_arr = np.ascontiguousarray(X[start:end].astype(np.float32))
                faiss.normalize_L2(chunk_arr)
                index.add(chunk_arr)
                print(f"    added {start}:{end}")
        else:
            X_norm = np.ascontiguousarray(X.astype(np.float32))
            faiss.normalize_L2(X_norm)
            index.add(X_norm)
        print(f"  index ntotal {index.ntotal}")
        # Search top1 for train queries
        # Use batch search
        seeds = []
        # Search in batches of 500 to avoid large memory
        bs = 500
        for start in range(0, q_train_norm.shape[0], bs):
            end = min(start+bs, q_train_norm.shape[0])
            D, I = index.search(q_train_norm[start:end], 1)
            seeds.extend(I.ravel().tolist())
        seeds = np.asarray(seeds, dtype=np.int64)
        h_seeds = hashlib.sha256(seeds.tobytes()).hexdigest()[:16]
        # Golds flat
        golds_flat_list = []
        for _, _, golds in train_pairs:
            for g in golds:
                golds_flat_list.append(id2idx[g])
        golds_flat = np.asarray(golds_flat_list, dtype=np.int64)
        h_golds = hashlib.sha256(golds_flat.tobytes()).hexdigest()[:16]
        print(f"  seeds {h_seeds} len {len(seeds)} golds {h_golds} len {len(golds_flat)} Xt64 {h_Xt64}")
        result[ds] = {
            "seeds_hash": h_seeds,
            "seeds_len": len(seeds),
            "golds_hash": h_golds,
            "golds_len": len(golds_flat),
            "Xt64_hash": h_Xt64,
            "Xt64_shape": [64, 1536],
            "X_shape": X_shape,
            "train_len": len(train_pairs),
        }
        # Save arrays to volume for later recompute (optional)
        # To avoid large, we can save to npz per dataset
        out_npz = f"/root/CRAG/storage/scratchpad/fp_{ds}_remote.npz"
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)
        # Use local seeds/golds/Xt64 already computed
        np.savez_compressed(out_npz, seeds=seeds, golds=golds_flat, Xt64=Xt64_raw)
        del X, q_train, q_train_norm, index, seeds, golds_flat, Xt64_raw, data, docs, id2idx, qs, pairs, pairs_sorted
        gc.collect()
    # Save result
    out_json = "/root/CRAG/storage/scratchpad/remote_all_light.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    return result

@app.local_entrypoint()
def main():
    import json
    result = extract_all.remote()
    print(json.dumps(result, indent=2))
    with open("scratchpad/remote_all_light.json", "w") as f:
        json.dump(result, f, indent=2)
    print("saved scratchpad/remote_all_light.json")
