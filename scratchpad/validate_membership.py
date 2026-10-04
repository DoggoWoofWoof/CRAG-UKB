import sys, os
sys.path.insert(0, os.getcwd())
import json, pickle, numpy as np, torch, faiss, random
from src.core.engine import CoreEngine
from src.experiments.l1_universal_head import _load
from src.experiments.overlap_retrain import _onehop_membership
from pathlib import Path

DATASET="2wiki_clean"
SUBDIR="gte_qwen"

# Load canonical data for A onehop
data = _load(DATASET, SUBDIR, 8000, 3000, 200)
mem_canonical = data["mem_idx"]
hard_canonical = data["hard"]
print(f"canonical mem_idx mean {sum(len(x) for x in mem_canonical)/len(mem_canonical):.2f} max {max(len(x) for x in mem_canonical)} sample {mem_canonical[0][:5]}")

# Engine onehop
eng = CoreEngine(source=DATASET, index_subdir=SUBDIR)
mem_onehop = _onehop_membership(eng)
idx2id = {v:k for k,v in eng.node_id_to_idx.items()}
mem_engine = [sorted(mem_onehop.get(idx2id[i], {int(hard_canonical[i])})) for i in range(len(eng.nodes))]
print(f"engine onehop mean {sum(len(x) for x in mem_engine)/len(mem_engine):.2f} max {max(len(x) for x in mem_engine)}")
mism = sum(1 for i in range(len(mem_canonical)) if set(mem_canonical[i]) != set(mem_engine[i]))
print(f"A_MEM_IDX_RECONSTRUCTION_PARITY mism {mism} / {len(mem_canonical)} {'PASS' if mism==0 else 'FAIL'}")
print(f"total canonical elements {sum(len(x) for x in mem_canonical)} reconstructed {sum(len(x) for x in mem_engine)}")

# Test helper for variant B/C: structural + NER
def load_ner_adj(dataset, doc_id_to_idx):
    pkl=f"data/ukb_storage/{dataset}/ner_edges_w_df25.pkl"
    A=pickle.load(open(pkl,"rb"))
    N=len(doc_id_to_idx)
    # Build adj from CSR
    coo=A.tocoo()
    adj=[set() for _ in range(N)]
    for r,c in zip(coo.row, coo.col):
        if r!=c:
            adj[r].add(int(c))
    return adj

# Build helper that mimics _onehop but with variant NER
def build_mem_idx_variant(dataset, partition_map, doc_id_to_idx, hard_array, include_ner=False):
    # structural adj from engine nodes
    eng_tmp = CoreEngine(source=dataset, index_subdir=SUBDIR)  # for structural neighbors, but we will use same eng as canonical? For variant, structural is same
    # Use canonical engine's nodes for structural
    structural_adj = [set() for _ in range(len(doc_id_to_idx))]
    for i, node in enumerate(eng_tmp.nodes):
        for nid in node.neighbors:
            j = doc_id_to_idx.get(nid)
            if j is not None and j!=i:
                structural_adj[i].add(j)
                structural_adj[j].add(i)  # ensure undirected
    # NER adj if needed
    ner_adj = None
    if include_ner:
        ner_adj = load_ner_adj(dataset, doc_id_to_idx)
        # Merge
        for i in range(len(structural_adj)):
            structural_adj[i].update(ner_adj[i])
    # Now onehop: for each doc, mem = {hard[i]} ∪ {hard[nb] for nb in structural_adj[i]}
    N = len(hard_array)
    mem_idx = []
    for i in range(N):
        s = {int(hard_array[i])}
        for nb in structural_adj[i]:
            s.add(int(hard_array[nb]))
        mem_idx.append(sorted(s))
    return mem_idx

# Test A via helper (structural only)
# Need hard_variant for A: hard_canonical
doc_id_to_idx = data["id2idx"]
hard_A = hard_canonical
mem_A_helper = build_mem_idx_variant(DATASET, None, doc_id_to_idx, hard_A, include_ner=False)
print(f"helper A mean {sum(len(x) for x in mem_A_helper)/len(mem_A_helper):.2f} mism vs canonical {sum(1 for i in range(len(mem_canonical)) if set(mem_canonical[i]) != set(mem_A_helper[i]))}")

# For B/C, need hard_B/C from partition maps
for variant in ["B","C"]:
    pm_path = f"scratchpad/ablation/2wiki_clean/variant_{variant}/partition_map.json"
    pm = json.load(open(pm_path))
    pm = {k:int(v) for k,v in pm.items()}
    N = len(doc_id_to_idx)
    hard_variant = np.full(N, -1, dtype=np.int64)
    for nid, pid in pm.items():
        if nid in doc_id_to_idx:
            hard_variant[doc_id_to_idx[nid]] = pid
    assert np.all(hard_variant>=0)
    include_ner = variant in ("B","C")
    mem_variant = build_mem_idx_variant(DATASET, pm, doc_id_to_idx, hard_variant, include_ner=include_ner)
    print(f"variant {variant} hard mean part {N/len(set(hard_variant)):.1f} mem mean {sum(len(x) for x in mem_variant)/len(mem_variant):.2f} median {np.median([len(x) for x in mem_variant]):.1f} p95 {np.percentile([len(x) for x in mem_variant],95):.1f} max {max(len(x) for x in mem_variant)} isolated {sum(1 for x in mem_variant if len(x)==1)}")
    # Also check partition stats
    npart = max(hard_variant)+1
    from collections import Counter
    counts = Counter(hard_variant)
    print(f" {variant} npart {npart} counts mean {np.mean(list(counts.values())):.1f}")

# Check L1 effective docs
from src.experiments.l1_rerank100 import _feats
# Simulate L1 vote: dense_order 100 vs 500
# For quick, use dummy order 100
print("L1_EFFECTIVE_DOCS_PER_QUERY check: _feats slices order[:topn] where topn=200, so 100 order => effective 100")
# Demonstrate
order_dummy = np.random.randint(0, 65865, size=(2,100))
S,M = _feats(order_dummy, mem_canonical, 658, topn=200)
print(f"S shape {S.shape} M shape {M.shape} (should be 2x658) - topn 200 on 100 order uses all 100")
order_dummy500 = np.random.randint(0, 65865, size=(2,500))
S2,M2 = _feats(order_dummy500, mem_canonical, 658, topn=200)
print(f"500 order truncated to 200 => uses 200, not 500, so L1 vote K=100 + topn200 => 100 docs, K=500+topn200=>200 docs difference!")

# SPLADE reuse check
print("SPLADEreuse: _splade_query_vecs caches per dataset, _splade_scoped_order calls it each time but will hit cache if file exists")
import os
print(f"splade_q_test.pkl exists {os.path.exists('data/ukb_storage/2wiki_clean/splade_q_test.pkl')}")
# Check _splade_scoped_order re-encodes: it calls _splade_query_vecs which checks cache file and reuses if shape >= len(texts)
# So second call will reuse, so encodings per job =1 if cache exists
print("SPLADE_ENCODINGS_PER_JOB should be 1 if cache exists, else 1 per first call + 0 thereafter")

# P100 filter parity check
print("P100 filter parity: need to test direct P50 vs P100 filter")
from src.experiments.l2_seed import _scoped_order
from src.experiments.e2e_pipeline import _merge_minrank
# Use small test
import torch
device=torch.device("cpu")
# Load heads and data for quick test (reuse previous data)
from src.experiments.query_relation import OffsetHead
from src.experiments.l1_ablate import MixtureHead
sd_hard=torch.load("data/ukb_storage/_head_cache/head_06a9fd3a3e39b3d0.pt", map_location="cpu")
hard_head=OffsetHead(1536); hard_head.load_state_dict(sd_hard); hard_head.eval()
sd_mix=torch.load("data/ukb_storage/_head_cache/head_32404bf9b65a2d95.pt", map_location="cpu")
mix_head=MixtureHead(1536,K=8); mix_head.load_state_dict(sd_mix); mix_head.eval()
X=data["X"]
qte=data["test"][0][:5]
ste=data["test"][1][:5]
gte=data["test"][2][:5]
texts=data["test_texts"][:5]
hard_t=torch.tensor(hard_canonical, device=device)
X_t=torch.tensor(X, device=device)
with torch.no_grad():
    qt=torch.tensor(qte, device=device)
    sv=torch.tensor(X[ste], device=device)
    pos_dense=qt
    pos_hard=hard_head(qt,sv)
    pos_mix=mix_head(qt,sv)
# Need partition voting for P100 and P50
import faiss
idx=faiss.IndexFlatIP(1536); idx.add(X)
_, dense_order100 = idx.search(qte, 100)
S,M = _feats(dense_order100, mem_canonical, 658, topn=200)
votes = _rr(S)+_rr(M)
part_rank = np.argsort(-votes, axis=1)
topP100=[set(int(pid) for pid in part_rank[qi,:100]) for qi in range(len(qte))]
topP50=[set(int(pid) for pid in part_rank[qi,:50]) for qi in range(len(qte))]
# Direct P50
od_dense50,_ = _scoped_order(pos_dense.cpu(), X_t, hard_t, topP50, False, device, k=500)
# P100 then filter
od_dense100,_ = _scoped_order(pos_dense.cpu(), X_t, hard_t, topP100, False, device, k=500)
# Filter P100 to P50: keep only docs whose partition in topP50
# For each query, filter od_dense100 row to keep docs where hard[doc] in topP50[qi], then take top 500
filtered=[]
for qi in range(len(qte)):
    row=od_dense100[qi]
    keep=[int(doc) for doc in row if doc>=0 and hard_canonical[int(doc)] in topP50[qi]]
    # pad
    keep = keep[:500] + [-1]*(500-len(keep))
    filtered.append(keep)
filtered=np.array(filtered, dtype=np.int64)
print("direct P50 first row", od_dense50[0][:10])
print("filtered P50 first row", filtered[0][:10])
print("parity", np.array_equal(od_dense50, filtered))

