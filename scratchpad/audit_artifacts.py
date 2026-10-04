import os, json, hashlib, pathlib

datasets=["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"]
base="data"
def sha256_file(p, head=1024*1024):
    if not os.path.exists(p): return None
    h=hashlib.sha256()
    with open(p,"rb") as f:
        while True:
            b=f.read(8192)
            if not b: break
            h.update(b)
    return h.hexdigest()

def size(p):
    return os.path.getsize(p) if os.path.exists(p) else 0

for ds in datasets:
    print(f"\n=== {ds} ===")
    # master nodes
    for m in [f"data/processed/master_nodes_{ds}.json", "data/processed/master_nodes.json"]:
        if os.path.exists(m):
            print(f"master {m} size {size(m)} sha {sha256_file(m)[:16] if size(m)<200000000 else 'large'}")
            break
    # gte_qwen nodes
    p=f"data/ukb_storage/{ds}/gte_qwen/nodes.npy"
    print(f"gte_qwen nodes {p} exists {os.path.exists(p)} size {size(p)}")
    if os.path.exists(p):
        print(f"  sha head {sha256_file(p)[:16]}")
    # query caches
    for split in ["train","test","val"]:
        q=f"data/ukb_storage/{ds}/gte_qwen/queries_{split}.npy"
        if os.path.exists(q):
            print(f"  query {split} {q} size {size(q)}")
    # splade doc
    for pp in [f"data/ukb_storage/{ds}/splade_doc_embs.pkl", f"data/ukb_storage/{ds}/gte_qwen/splade_doc_embs.pkl"]:
        if os.path.exists(pp):
            print(f"splade doc {pp} size {size(pp)} sha {sha256_file(pp)[:16]}")
    # graph
    for pp in [f"data/ukb_storage/{ds}/gte_qwen/graph.pt", f"data/ukb_storage/{ds}/graph.pt"]:
        if os.path.exists(pp):
            print(f"graph {pp} size {size(pp)}")
            break
    # partition
    for pp in [f"data/ukb_storage/{ds}/gte_qwen/partition_map.json", f"data/ukb_storage/{ds}/partition_map.json"]:
        if os.path.exists(pp):
            print(f"partition {pp} size {size(pp)} npart {len(json.load(open(pp))) if os.path.getsize(pp)<5000000 else 'large'}")
            break
    for pp in [f"data/ukb_storage/{ds}/gte_qwen/centroids.index", f"data/ukb_storage/{ds}/centroids.index"]:
        if os.path.exists(pp):
            print(f"centroids {pp} size {size(pp)}")
            break
    for pp in [f"data/ukb_storage/{ds}/gte_qwen/ner_edges_w_df25.pkl", f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl"]:
        if os.path.exists(pp):
            print(f"ner {pp} size {size(pp)} sha {sha256_file(pp)[:16]}")
            break
    else:
        # check any ner
        import glob
        cand=glob.glob(f"data/ukb_storage/{ds}/*ner*")
        if cand:
            print(f"ner cand {cand}")
        # also check canonical
        cand2=glob.glob(f"data/canonical/{ds.split('_')[0]}/ner*")
        if cand2:
            print(f"canonical ner {cand2}")

# heads
print("\n=== heads ===")
import glob
for p in sorted(glob.glob("data/ukb_storage/_head_cache/head_*.pt")):
    print(p, size(p), hashlib.sha256(open(p,"rb").read(1024*1024)).hexdigest()[:16] if os.path.exists(p) else "")

# D results
print("\n=== D results ===")
for p in sorted(glob.glob("results/L2/*.json")):
    print(p, size(p))

# signals
for p in sorted(glob.glob("results/L2/signals_*.npz"))[:5]:
    print(p, size(p))
