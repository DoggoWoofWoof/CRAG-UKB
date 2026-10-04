import os, json, hashlib, glob, pathlib
base = pathlib.Path("data/ukb_storage/2wiki_clean")
checks = []
def exists(p):
    return pathlib.Path(p).exists()
def size(p):
    try: return pathlib.Path(p).stat().st_size
    except: return -1
# Core artifacts
artifacts = {
    "graph.pt": base/"graph.pt",
    "partition_map.json": base/"partition_map.json",
    "centroids.index": base/"centroids.index",
    "centroid_pids.json": base/"centroid_pids.json",
    "nodes.index": base/"nodes.index",
    "ner_edges_w_df25.pkl": base/"ner_edges_w_df25.pkl",
    "splade_doc_embs.pkl": base/"splade_doc_embs.pkl",
    "gte_qwen/nodes.npy": base/"gte_qwen/nodes.npy",
    "gte_qwen/queries.npy": base/"gte_qwen/queries.npy",
    "gte_qwen/nodes_ids.json": base/"gte_qwen/nodes_ids.json",
    "gte_qwen/queries_ids.json": base/"gte_qwen/queries_ids.json",
    "head hard": pathlib.Path("data/ukb_storage/_head_cache/head_06a9fd3a3e39b3d0.pt"),
    "head mix": pathlib.Path("data/ukb_storage/_head_cache/head_32404bf9b65a2d95.pt"),
}
for k,p in artifacts.items():
    print(f"{k:30} exists={exists(p)} size={size(p)} mtime={p.stat().st_mtime if exists(p) else 'MISSING'}")
# Also check canonical 2wiki
for ds in ["2wiki","2wiki_clean","squad","squad_clean","hotpotqa","hotpotqa_clean","musique","musique_clean","metaqa","webqsp"]:
    for sub in [f"data/ukb_storage/{ds}/graph.pt", f"data/ukb_storage/{ds}/partition_map.json"]:
        if exists(sub):
            print(f"FOUND {sub} size {size(sub)}")
# Check partition_map content
pm = base/"partition_map.json"
if pm.exists():
    import json
    d=json.load(open(pm))
    print(f"partition_map keys {list(d.keys())[:3]} n_parts len {len(d)} sample {list(d.items())[:2]}")
# Check graph.pt quickly
g = base/"graph.pt"
if g.exists():
    print(f"graph.pt size {g.stat().st_size}")
    # try torch load small?
# Check gte_qwen manifest
manifest = base/"gte_qwen/manifest.json"
if manifest.exists():
    import json
    print(json.load(open(manifest)))
else:
    print("no manifest.json at gte_qwen")
# Check nodes.npy shape via header
import numpy as np
for p in [base/"gte_qwen/nodes.npy", base/"gte_qwen/queries.npy"]:
    if p.exists():
        try:
            arr=np.lib.format.open_memmap(str(p), mode='r')
            print(f"{p} shape {arr.shape} dtype {arr.dtype}")
        except Exception as e:
            print(f"{p} error {e}")
# Check ner_edges
ner = base/"ner_edges_w_df25.pkl"
if ner.exists():
    import pickle, hashlib
    print(f"ner size {ner.stat().st_size} sha {hashlib.sha256(open(ner,'rb').read()).hexdigest()[:8]}")
# Check heads
for h in ["head_06a9fd3a3e39b3d0.pt","head_32404bf9b65a2d95.pt"]:
    p=pathlib.Path(f"data/ukb_storage/_head_cache/{h}")
    if p.exists():
        print(f"{h} size {p.stat().st_size}")
