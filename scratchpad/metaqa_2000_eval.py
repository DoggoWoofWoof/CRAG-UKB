#!/usr/bin/env python
import os, sys
sys.path.insert(0, os.path.abspath("."))
import json, hashlib, numpy as np, torch, faiss
from src.core.engine import CoreEngine
from src.experiments.overlap_retrain import _splits, _hard_membership
from src.experiments.l1_universal_head import _load, _train_universal
from src.experiments.l1_rerank100 import _feats, _rr

# Load historical 2000 IDs
hist=json.load(open('data/canonical/metaqa/splits/historical_2000_ids.json'))
hist_ids=hist['ids']
print(f"hist 2000 hash {hist['hash']} n {len(hist_ids)}")

# Load metaqa via _load with 8000,3000,2000 to get same n as historical? But we want to evaluate on exact IDs, so we need to load full test and slice
# Instead we will use _splits to get test in order and take first 2000
eng=CoreEngine(source='metaqa', index_subdir='gte_qwen')
print(f"master {eng.master_nodes_path}")
sp=_splits(eng, _hard_membership(eng))
test=sp['test']
test_ids=[nd.node_id for nd,_,_ in test]
assert test_ids[:2000]==hist_ids, "mismatch"
print("verified first 2000 match")

# Need to evaluate current model on those 2000
# Load data via _load with large caps to get full, but we will slice to first 2000 after
# Use _load to get X, mem_idx, etc and also qte etc
from src.experiments.l1_universal_head import _load as load2
device=torch.device('cpu')
# For head training, need per_ds for 6 datasets to get heads (reuse cache)
head_datasets=['musique_clean','2wiki_clean','squad_clean','metaqa','hotpotqa_clean','webqsp']
per_ds={}
for d in head_datasets:
    dd=load2(d, 'gte_qwen', 8000, 3000, 1)
    idx=faiss.IndexFlatIP(dd['X'].shape[1])
    idx.add(dd['X'])
    per_ds[d]={'train': dd['train'], 'Xt': torch.tensor(dd['X'], device=device), 'index': idx}
    print(f"loaded head {d} {dd['X'].shape}")

# Train/load heads (should hit cache)
heads={}
for kind in ['hard','mix_hard']:
    h=_train_universal(kind, per_ds, device, epochs=15, K=8)
    heads[kind]=h
    print(f"head {kind} ready")

# Now load metaqa eval data full
data=load2('metaqa','gte_qwen',8000,3000,1000000)  # uncapped full
print(f"metaqa full test {len(data['test'][2])} train {len(data['train'][0])}")
# Slice to historical 2000
qte, ste, gte=data['test']
# Need to find indices of historical 2000 in this data's test ordering
# data['test'] order is same as sp['test'] order? _load's test is sp['test'][:caps] where caps is 1M -> all 39093 in same order
# So first 2000 are hist
qte2000=qte[:2000]
ste2000=ste[:2000]
gte2000=gte[:2000]
# Also need splade, etc
X=data['X']
npart=data['npart']
mem_idx=data['mem_idx']
hard=data['hard']
print(f"npart {npart} X {X.shape} q {qte2000.shape}")

# Build faiss for dense top100 voting (for _topP) and for seed etc
# For scope0, topP empty, so voting not needed, but we still need dense order via _scoped_order full corpus
# Use same logic as e2e_pipeline level2_order
import torch
Xt=torch.tensor(X, device=device)
hard_t=torch.tensor(hard, device=device)
# Need faiss for topP but scope0 empty
from src.experiments.l2_seed import _topP, _scoped_order, _splade_scoped_order, _recall, MAXK
# Create per_d
idx2=faiss.IndexFlatIP(X.shape[1]); idx2.add(X)
per_d={'Xt':Xt, 'faiss': idx2}
# Compute dense order for topP (not used for scope0 but for completeness)
_, I = idx2.search(qte2000, 100)
topP=[set()]*len(qte2000)  # scope0

# Compute positions
with torch.no_grad():
    qt=torch.tensor(qte2000, device=device)
    sv=Xt[torch.tensor(ste2000, device=device)]
    pos={"dense":qt, "rel_hard":heads['hard'](qt, sv), "mlpT":heads['mix_hard'](qt, sv)}

# Retrieve orders
from src.experiments.l2_seed import _scoped_order
# Use same device and batch
is_mix={'dense':False,'rel_hard':False,'mlpT':True}
orders={}
for m in pos:
    o,_=_scoped_order(pos[m].cpu(), Xt, hard_t, topP, is_mix[m], device, bs=64, k=500)
    orders[m]=o
    print(f"{m} order {o.shape}")

# SPLADE
splade_data=data.get('splade')
if splade_data is not None:
    from src.experiments.l2_seed import _splade_scoped_order
    # Need test_texts
    te_texts=data['test_texts'][:2000]
    # _splade_scoped_order expects splade tuple, texts, hard, topP
    so=_splade_scoped_order(splade_data, te_texts, hard, topP, dataset='metaqa', k=500)
    orders['splade']=so
    print(f"splade order {so.shape}")

# Compute per-signal recall R@2,5,20,50
KS=[2,5,20,50]
def recall(order, golds):
    out={k:0 for k in KS}
    n=0
    for qi,g in enumerate(golds):
        if not g: continue
        n+=1
        gs=set(g)
        row=order[qi]
        for k in KS:
            found=gs & set(int(x) for x in row[:k] if x>=0)
            out[k]+=len(found)/len(g)
    return {k: round(100*out[k]/max(n,1),2) for k in KS}

for sig in ['dense','rel_hard','mlpT','splade']:
    if sig in orders:
        r=recall(orders[sig], gte2000)
        print(f"{sig} R@ {r}")

# D_L2 minrank fusion of 4
from src.experiments.l2_seed import _bestof
# Need to convert orders to list of arrays for _bestof
# _bestof expects list of orders
orders_list=[orders[k] for k in ['dense','rel_hard','mlpT'] if k in orders]
if 'splade' in orders:
    orders_list.append(orders['splade'])
# _bestof does best rank fusion
from src.experiments.l2_seed import _bestof as bf
# But bf expects orders as np arrays
# Use same as e2e: perq _merge_minrank
# Simplify: use _bestof from l2_seed
# Need to handle nq
# Create array of orders
# Use e2e _merge_minrank logic: we can import
from src.experiments.e2e_pipeline import _merge_minrank
nq=len(gte2000)
l2_min=[_merge_minrank([orders[k][qi] for k in orders]) for qi in range(nq)]
# Actually _merge_minrank takes list of per-query orders (each is array of doc ids)
# So we do:
l2_min2=[]
for qi in range(nq):
    docs=[orders[k][qi] for k in orders]
    l2_min2.append(_merge_minrank(docs))
# Convert to array for recall
# _recall expects list of lists
from src.experiments.l2_seed import _recall as rec2
r_l2=rec2(l2_min2, gte2000)
print(f"D_L2 minrank R@ {r_l2}")

# D_E2E needs L3: we need to do NER graph etc. For now just report D_L2 and note D_E2E requires graph
# We can compute L3 if not too heavy: need CoreEngine graph
from src.core.engine import CoreEngine as CE
eng2=CoreEngine(source='metaqa', index_subdir='gte_qwen')
from src.experiments.l1l3_recall import _graph
n=X.shape[0]
id2idx=eng2.node_id_to_idx
_, A_str, _ = _graph(eng2, n, id2idx, sources=("struct",))
from src.pipeline.ner_edges import build_ner_edges
A_ner=build_ner_edges('metaqa', [eng2.nodes[i].content for i in range(len(eng2.nodes))], n)
import scipy.sparse as sp
def _transition(A):
    A=A.tocsr().astype(np.float32)
    deg=np.asarray(A.sum(1)).ravel(); deg[deg==0]=1.0
    return sp.diags(1.0/deg) @ A
P=_transition((A_str+A_ner).tocsr())
from src.experiments.e2e_pipeline import _ner_compose
composed=[_ner_compose(l2_min2[qi], P, n, n_seed=2) for qi in range(nq)]
r_e2e=rec2(composed, gte2000)
print(f"D_E2E L2_plus_nerL3 R@ {r_e2e}")

# Save results
out={
 'historical_ids_hash': hist['hash'],
 'n': 2000,
 'dense': recall(orders['dense'], gte2000),
 'rel_hard': recall(orders['rel_hard'], gte2000),
 'mlpT': recall(orders['mlpT'], gte2000),
 'splade': recall(orders.get('splade', np.zeros((2000,500))), gte2000) if 'splade' in orders else {},
 'D_L2': r_l2,
 'D_E2E': r_e2e,
 'K':8,'MAXK':500,'scope':0
}
json.dump(out, open('scratchpad/metaqa_2000_repro.json','w'), indent=2)
print("saved to scratchpad/metaqa_2000_repro.json")
