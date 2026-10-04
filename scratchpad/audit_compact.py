import os, json, hashlib, glob, pathlib

datasets=["musique_clean","2wiki_clean","squad_clean","metaqa","hotpotqa_clean","webqsp"]
def sha(p):
    if not os.path.exists(p): return "MISSING"
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda: f.read(8192), b""):
            h.update(b)
    return h.hexdigest()[:16]

def size(p):
    return os.path.getsize(p) if os.path.exists(p) else 0

# Check local
for ds in datasets:
    print(f"\n{ds}")
    # MASTER
    for m in [f"data/processed/master_nodes_{ds}.json","data/processed/master_nodes.json"]:
        if os.path.exists(m):
            print(f"MASTER {m} {size(m)} {sha(m)}")
            break
    # GTE_DOCS
    p=f"data/ukb_storage/{ds}/gte_qwen/nodes.npy"
    print(f"GTE_DOCS {p} {size(p)} {sha(p) if size(p)<500000000 else 'large'}")
    # GTE queries
    for split in ["train","test"]:
        q=f"data/ukb_storage/{ds}/gte_qwen/queries_{split}.npy"
        if os.path.exists(q):
            print(f"GTE_QUERY_{split.upper()} {q} {size(q)}")
    # SPLADE
    p=f"data/ukb_storage/{ds}/splade_doc_embs.pkl"
    print(f"SPLADE_DOCS {p} {size(p)} {sha(p) if size(p)<100000000 else 'large'}")
    # STRUCT
    p=f"data/ukb_storage/{ds}/gte_qwen/graph.pt"
    print(f"STRUCT_GRAPH {p} {size(p)}")
    # PARTITION
    p=f"data/ukb_storage/{ds}/gte_qwen/partition_map.json"
    print(f"PARTITION_MAP {p} {size(p)} {sha(p)[:16] if os.path.exists(p) else 'MISSING'}")
    p=f"data/ukb_storage/{ds}/gte_qwen/centroids.index"
    print(f"CENTROIDS {p} {size(p)}")
    # NER
    p=f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl"
    if not os.path.exists(p):
        p=f"data/ukb_storage/{ds}/gte_qwen/ner_edges_w_df25.pkl"
    print(f"NER_DF25 {p} {size(p)} {sha(p)[:16] if os.path.exists(p) else 'MISSING'}")
    if not os.path.exists(p):
        # check volume via local alternative
        import glob as gl
        cand=gl.glob(f"data/ukb_storage/{ds}/*ner*")
        print(f"  cand {cand}")
# Heads
print("\nHEADS")
for p in sorted(glob.glob("data/ukb_storage/_head_cache/head_*.pt"))[:4]:
    print(p, size(p), sha(p))

# Now compare to account1 via modal volume ls --json
print("\nREMOTE account1 check via volume ls --json")
import subprocess, sys, json as js
from src.experiments import credentials
cred=credentials.load_pool('modal')[1]
cred.activate()
# For each dataset, check NER existence on remote
for ds in datasets:
    for pp in [f"data/ukb_storage/{ds}/ner_edges_w_df25.pkl", f"data/ukb_storage/{ds}/gte_qwen/ner_edges_w_df25.pkl"]:
        r=subprocess.run([sys.executable,'-m','modal','volume','ls','crag-data-volume',pp,'--json'], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20)
        if r.returncode==0 and r.stdout.strip():
            try:
                j=js.loads(r.stdout)
                print(f"REMOTE NER {ds} {pp} EXISTS {j[0]['Size'] if j else 'empty'}")
            except:
                print(f"REMOTE NER {ds} {pp} raw {r.stdout[:200]}")
            break
    else:
        print(f"REMOTE NER {ds} MISSING_BOTH")
