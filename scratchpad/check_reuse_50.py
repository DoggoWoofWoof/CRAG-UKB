import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, torch, faiss, json, hashlib, pickle
from src.core.engine import CoreEngine
from src.experiments.l1_universal_head import _load
from src.experiments.l1_rerank100 import _feats, _rr
from src.experiments.l2_seed import _scoped_order, _splade_scoped_order, _recall
from src.experiments.e2e_pipeline import _merge_minrank
from src.experiments.query_relation import OffsetHead
from src.experiments.l1_ablate import MixtureHead

DATASET="2wiki_clean"
SUBDIR="gte_qwen"
device=torch.device("cpu")
data=_load(DATASET, SUBDIR, 8000, 3000, 50)
X=data["X"]; qte=data["test"][0]; ste=data["test"][1]; gte=data["test"][2]; texts=data["test_texts"]
hard=data["hard"]; mem_idx=data["mem_idx"]; npart=data["npart"]
print(f"50q loaded qte {qte.shape}")

# Heads
sd_hard=torch.load("data/ukb_storage/_head_cache/head_06a9fd3a3e39b3d0.pt", map_location="cpu")
hard_head=OffsetHead(1536); hard_head.load_state_dict(sd_hard); hard_head.eval().to(device)
sd_mix=torch.load("data/ukb_storage/_head_cache/head_32404bf9b65a2d95.pt", map_location="cpu")
mix_head=MixtureHead(1536,K=8); mix_head.load_state_dict(sd_mix); mix_head.eval().to(device)
X_t=torch.tensor(X, device=device)
hard_t=torch.tensor(hard, device=device)
with torch.no_grad():
    qt=torch.tensor(qte, device=device)
    sv=torch.tensor(X[ste], device=device)
    pos_dense=qt
    pos_hard=hard_head(qt,sv)
    pos_mix=mix_head(qt,sv)
# SPLADE
splade=data["splade"]
from src.experiments.l2_seed import _splade_query_vecs
splade_q=None
if splade:
    splade_q=_splade_query_vecs(splade[0], texts, dataset=DATASET)
    print(f"splade_q {splade_q.shape}")

# FAISS
idx=faiss.IndexFlatIP(1536); idx.add(X)
_, dense_order100 = idx.search(qte, 100)
S,M=_feats(dense_order100, mem_idx, npart, topn=200)
votes=_rr(S)+_rr(M)
part_rank=np.argsort(-votes, axis=1)
topP100=[set(int(pid) for pid in part_rank[qi,:100]) for qi in range(len(qte))]
topP50=[set(int(pid) for pid in part_rank[qi,:50]) for qi in range(len(qte))]

def direct(topP):
    orders={}
    od,_=_scoped_order(pos_dense.cpu(), X_t, hard_t, topP, False, device, k=500)
    orders["dense"]=od
    od,_=_scoped_order(pos_hard.cpu(), X_t, hard_t, topP, False, device, k=500)
    orders["rel_hard"]=od
    od,_=_scoped_order(pos_mix.cpu(), X_t, hard_t, topP, True, device, k=500)
    orders["mlpT"]=od
    if splade:
        od=_splade_scoped_order(splade, texts, hard, topP, dataset=DATASET, k=500)
        orders["splade"]=od
    sigs=[orders[k] for k in ["dense","rel_hard","mlpT"] if k in orders]
    if "splade" in orders:
        sigs.append(orders["splade"])
    l2=[_merge_minrank([s[qi] for s in sigs]) for qi in range(len(qte))]
    l2_arr=np.full((len(qte),500), -1, dtype=np.int64)
    for qi,lst in enumerate(l2):
        arr=np.array(lst[:500], dtype=np.int64)
        l2_arr[qi,:len(arr)]=arr
    orders["D_L2"]=l2_arr
    return orders

# Direct P50
direct50=direct(topP50)
# Full P100 then filter: compute full P100 scores with full pool (not top500 truncated) then filter to P50
# For each expert, compute full ordering for P100 pool (all docs in P100)
def full_orders_for_P100():
    harr=np.array(hard)
    full={}
    for name, pos in [("dense",pos_dense),("rel_hard",pos_hard),("mlpT",pos_mix)]:
        is_mix=(name=="mlpT")
        full_list=[]
        for qi in range(len(qte)):
            pool=np.where(np.isin(harr, list(topP100[qi])))[0]
            if is_mix:
                p=pos[qi:qi+1].to(device)
                sim=torch.einsum("bkd,nd->bkn", p, X_t).max(1).values[0]
                sim_pool=sim[torch.tensor(pool, device=device)]
                order_idx=torch.argsort(sim_pool, descending=True).cpu().numpy()
                order=pool[order_idx]
                full_list.append(order)
            else:
                p=pos[qi].to(device)
                sim=p @ X_t.T
                sim_pool=sim[torch.tensor(pool, device=device)]
                order_idx=torch.argsort(sim_pool, descending=True).cpu().numpy()
                order=pool[order_idx]
                full_list.append(order)
        full[name]=full_list
    # Splade
    if splade:
        doc_matrix=splade[1]
        full_list=[]
        for qi in range(len(qte)):
            pool=np.where(np.isin(harr, list(topP100[qi])))[0]
            qvec=splade_q[qi]
            scores=np.asarray(doc_matrix.dot(qvec.T).todense()).ravel()
            pool_scores=scores[pool]
            order_idx=np.argsort(-pool_scores)
            order=pool[order_idx]
            full_list.append(order)
        full["splade"]=full_list
    # D_L2 full via merge of full lists
    # For each query, D_L2 full order is merge of full expert orders
    full_D=[]
    for qi in range(len(qte)):
        sigs=[full[name][qi] for name in ["dense","rel_hard","mlpT"] if name in full]
        if "splade" in full:
            sigs.append(full["splade"][qi])
        # Need to convert each full order (which is pool docs sorted) to list, then merge
        # But _merge_minrank expects lists of doc ids (full pool sorted)
        merged=_merge_minrank(sigs)
        full_D.append(np.array(merged[:500], dtype=np.int64) if len(merged)>=500 else np.array(merged + [-1]*(500-len(merged)), dtype=np.int64))
    full["D_L2"]=np.array(full_D)
    return full

full100=full_orders_for_P100()
# Now derive P50 by filtering full100
derived={}
for name in ["dense","rel_hard","mlpT","splade"]:
    if name not in full100:
        continue
    full_list=full100[name]
    harr=np.array(hard)
    derived_list=[]
    for qi in range(len(qte)):
        # Filter full_list[qi] (which is P100 pool sorted) to keep docs where hard in topP50[qi]
        keep=[int(doc) for doc in full_list[qi] if harr[int(doc)] in topP50[qi]]
        keep=keep[:500]+[-1]*(500-len(keep))
        derived_list.append(keep)
    derived[name]=np.array(derived_list)

# For D_L2, need to derive similarly but from full100 D_L2? Actually D_L2 derived from expert full orders, then filtered? Simpler: compute D_L2 derived as merge of derived expert orders
derived_D=[]
for qi in range(len(qte)):
    sigs=[derived[name][qi] for name in ["dense","rel_hard","mlpT"] if name in derived]
    if "splade" in derived:
        sigs.append(derived["splade"][qi])
    merged=_merge_minrank(sigs)
    derived_D.append(np.array(merged[:500] + [-1]*(500-len(merged)) if len(merged)<500 else merged[:500], dtype=np.int64))
derived["D_L2"]=np.array(derived_D)

# Compare
for name in ["dense","rel_hard","mlpT","splade","D_L2"]:
    if name not in direct50 or name not in derived:
        continue
    eq=np.array_equal(direct50[name], derived[name])
    diff=np.sum(direct50[name]!=derived[name])
    rec_direct=_recall(direct50[name], gte)
    rec_derived=_recall(derived[name], gte)
    print(f"{name} equal {eq} diff {diff} rec_direct R5 {rec_direct[5]} rec_derived {rec_derived[5]} hit {rec_direct['hit5']} vs {rec_derived['hit5']}")
    if not eq:
        # Check tie handling: find first diff position
        for qi in range(len(qte)):
            if not np.array_equal(direct50[name][qi], derived[name][qi]):
                print(f"  first diff qi {qi} direct {direct50[name][qi][:5]} derived {derived[name][qi][:5]}")
                break
print("FULL_SCORE_REUSE_METRIC_PARITY check done")
# Overall metric parity
all_eq=all(_recall(direct50[k], gte)[5]==_recall(derived[k], gte)[5] for k in ["dense","rel_hard","mlpT","splade","D_L2"] if k in derived)
print(f"FULL_SCORE_REUSE_METRIC_PARITY = {'PASS' if all_eq else 'FAIL'}")
