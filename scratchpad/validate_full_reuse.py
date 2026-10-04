import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, torch, faiss, pickle, json, hashlib
from src.core.engine import CoreEngine
from src.experiments.l1_universal_head import _load
from src.experiments.l1_rerank100 import _feats, _rr
from src.experiments.l2_seed import _scoped_order, _splade_scoped_order, _recall, MAXK
from src.experiments.e2e_pipeline import _merge_minrank
from src.experiments.query_relation import OffsetHead
from src.experiments.l1_ablate import MixtureHead

DATASET="2wiki_clean"
SUBDIR="gte_qwen"
device=torch.device("cpu")
# Load 5 queries
data=_load(DATASET, SUBDIR, 8000, 3000, 5)
X=data["X"]; qte=data["test"][0]; ste=data["test"][1]; gte=data["test"][2]; texts=data["test_texts"]
hard=data["hard"]; mem_idx=data["mem_idx"]; npart=data["npart"]
print(f"qte {qte.shape} hard max {hard.max()} npart {npart} mean mem {sum(len(x) for x in mem_idx)/len(mem_idx):.2f}")
# FAISS for L1
idx=faiss.IndexFlatIP(1536); idx.add(X)
_, dense_order100 = idx.search(qte, 100)
S,M=_feats(dense_order100, mem_idx, npart, topn=200)
votes=_rr(S)+_rr(M)
part_rank=np.argsort(-votes, axis=1)
topP100=[set(int(pid) for pid in part_rank[qi,:100]) for qi in range(5)]
topP50=[set(int(pid) for pid in part_rank[qi,:50]) for qi in range(5)]
topP20=[set(int(pid) for pid in part_rank[qi,:20]) for qi in range(5)]
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
# Load splade
splade=data["splade"]
splade_q=None
if splade:
    from src.experiments.l2_seed import _splade_query_vecs
    splade_q=_splade_query_vecs(splade[0], texts, dataset=DATASET)
    print(f"splade_q {splade_q.shape}")

def direct(Ptop):
    orders={}
    od,_=_scoped_order(pos_dense.cpu(), X_t, hard_t, Ptop, False, device, k=500)
    orders["dense"]=od
    od,_=_scoped_order(pos_hard.cpu(), X_t, hard_t, Ptop, False, device, k=500)
    orders["rel_hard"]=od
    od,_=_scoped_order(pos_mix.cpu(), X_t, hard_t, Ptop, True, device, k=500)
    orders["mlpT"]=od
    if splade:
        od=_splade_scoped_order(splade, texts, hard, Ptop, dataset=DATASET, k=500)
        orders["splade"]=od
    # D_L2
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

def full_reuse():
    # For each expert, compute scores for ALL docs in P100 scope, retain doc,partition,score
    # Then derive P100/P50/P20 by filtering
    # Implement for dense as example: score = pos_dense @ X_t.T  (full) then mask
    # For efficiency, we can do per query: compute full scores for P100 pool, sort, then filter
    # We'll test for dense, rel_hard, mlpT, splade
    nq=len(qte)
    # Precompute hard array
    harr=np.array(hard)
    # For each expert, compute full P100 scores
    results={}
    for name, pos in [("dense",pos_dense),("rel_hard",pos_hard),("mlpT",pos_mix)]:
        is_mix=(name=="mlpT")
        # Compute full scores for P100 pool per query, keep all
        all_orders_P100=[]
        all_scores_P100=[]
        for qi in range(nq):
            # P100 pool docs
            pool = np.where(np.isin(harr, list(topP100[qi])))[0]
            # Score
            if is_mix:
                # pos shape (nq,K,d) -> max over K
                p = pos[qi:qi+1].to(device)  # (1,K,d)
                # Compute sim = einsum bkd,nd -> bkn
                sim = torch.einsum("bkd,nd->bkn", p, X_t).max(1).values[0]  # (N,)
                # Only pool
                sim_pool = sim[torch.tensor(pool, device=device)]
                # Get top500 among pool by score
                k = min(500, len(pool))
                vals, loc = torch.topk(sim_pool, k)
                order = pool[loc.cpu().numpy()]
                # Pad
                order_padded = np.full(500, -1, dtype=np.int64)
                order_padded[:k]=order
                all_orders_P100.append(order_padded)
            else:
                p = pos[qi].to(device)  # (d,)
                sim = p @ X_t.T  # (N,)
                sim_pool = sim[torch.tensor(pool, device=device)]
                k = min(500, len(pool))
                vals, loc = torch.topk(sim_pool, k)
                order = pool[loc.cpu().numpy()]
                order_padded = np.full(500, -1, dtype=np.int64)
                order_padded[:k]=order
                all_orders_P100.append(order_padded)
        all_orders_P100=np.array(all_orders_P100)
        results[name+"_P100"]=all_orders_P100
        # Derive P50 by filtering P100 scored pool's full ordering? But we only have top500 of P100, not full pool
        # To be exact, we need full pool ordering, not top500 truncated
        # So this test with top500 truncated will FAIL as before
        # Let's instead compute full pool ordering (all pool docs sorted) then take top500
        # For P50, pool is subset of P100, so we can filter full P100 pool ordering
        # Need full ordering for P100 (all pool docs sorted)
        # Let's compute full ordering for P100 (not truncated)
        full_orders_P100=[]
        for qi in range(nq):
            pool = np.where(np.isin(harr, list(topP100[qi])))[0]
            if is_mix:
                p = pos[qi:qi+1].to(device)
                sim = torch.einsum("bkd,nd->bkn", p, X_t).max(1).values[0]
                sim_pool = sim[torch.tensor(pool, device=device)]
                # Sort all pool by sim
                order_idx = torch.argsort(sim_pool, descending=True).cpu().numpy()
                order = pool[order_idx]
                full_orders_P100.append(order)
            else:
                p = pos[qi].to(device)
                sim = p @ X_t.T
                sim_pool = sim[torch.tensor(pool, device=device)]
                order_idx = torch.argsort(sim_pool, descending=True).cpu().numpy()
                order = pool[order_idx]
                full_orders_P100.append(order)
        # Now derive P50/P20 by filtering full_orders_P100
        for P, topP in [(50, topP50), (20, topP20)]:
            derived=[]
            for qi in range(nq):
                full = full_orders_P100[qi]
                # Filter to keep docs where hard[doc] in topP[qi]
                keep = [int(doc) for doc in full if harr[int(doc)] in topP[qi]]
                keep = keep[:500] + [-1]*(500-len(keep))
                derived.append(keep)
            derived=np.array(derived)
            results[name+f"_P{P}_derived"]=derived
            # Compare to direct
            direct_orders = direct(topP)[name]
            eq = np.array_equal(derived, direct_orders)
            print(f"{name} P{P} derived vs direct equal {eq} diff {np.sum(derived!=direct_orders)}")
            if not eq:
                # Check if metrics same despite tie differences?
                # Compute R@ for gte
                # For this 5q test, check recall
                from src.experiments.l2_seed import _recall
                rec_direct=_recall(direct_orders, gte[:len(qte)])
                rec_derived=_recall(derived, gte[:len(qte)])
                print(f" rec direct {rec_direct} derived {rec_derived}")
    return results

print("=== DIRECT vs FULL_REUSE ===")
# Test direct vs derived for 5q
for name in ["dense","rel_hard","mlpT"]:
    is_mix=(name=="mlpT")
    pos = pos_mix if is_mix else (pos_dense if name=="dense" else pos_hard)
    # Direct P50
    d50=direct(topP50)[name]
    # Full reuse derived already tested above, but we need to run full_reuse for each
    pass

full_reuse()
print("done")
