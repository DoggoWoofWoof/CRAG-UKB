import os, json, hashlib, numpy as np, pathlib

# Tiny synthetic test for ranking persistence
# Create synthetic data for 5 queries, 10 candidates, 4 experts
np.random.seed(0)
nq=5
nc=10
# Create synthetic rankings
query_ids=[f"q_{i}" for i in range(nq)]
# Dense topK etc
dense=np.array([np.random.permutation(nc) for _ in range(nq)], dtype=np.int32)
rel_hard=np.array([np.random.permutation(nc) for _ in range(nq)], dtype=np.int32)
mlpT=np.array([np.random.permutation(nc) for _ in range(nq)], dtype=np.int32)
splade=np.array([np.random.permutation(nc) for _ in range(nq)], dtype=np.int32)
# D_L2 as minrank fusion
def merge_minrank(orders):
    # orders is list of arrays per expert
    best={}
    for order in orders:
        for r, doc in enumerate(order):
            if doc not in best or r < best[doc]:
                best[doc]=r
    return [d for d,_ in sorted(best.items(), key=lambda kv: kv[1])]

D_L2=np.array([merge_minrank([dense[i], rel_hard[i], mlpT[i], splade[i]]) for i in range(nq)], dtype=np.int32)
# Gold IDs
gold_ids=[ [np.random.randint(0,nc)] for _ in range(nq)]
# Save
os.makedirs("scratchpad/ranking_test", exist_ok=True)
# Save query_ids separately
with open("scratchpad/ranking_test/ids.json","w") as f:
    json.dump(query_ids, f)
# Save rankings as npz
np.savez_compressed("scratchpad/ranking_test/rankings.npz",
    dense=dense, rel_hard=rel_hard, mlpT=mlpT, splade=splade, D_L2=D_L2,
    gold=np.array(gold_ids, dtype=object))

print("saved synthetic rankings", dense.shape)

# Reload and compute R@2/5 offline
data=np.load("scratchpad/ranking_test/rankings.npz", allow_pickle=True)
ids=json.load(open("scratchpad/ranking_test/ids.json"))
# Compute in-memory before save for comparison
def recall(order, golds, k):
    hit=0
    for qi,g in enumerate(golds):
        if set(g) & set(order[qi][:k]):
            hit+=1
    return 100*hit/len(golds)

# In-memory
for name in ["dense","rel_hard","mlpT","splade","D_L2"]:
    arr = {"dense":dense,"rel_hard":rel_hard,"mlpT":mlpT,"splade":splade,"D_L2":D_L2}[name]
    r2=recall(arr, gold_ids, 2)
    r5=recall(arr, gold_ids, 5)
    print(f"in-mem {name} R@2 {r2:.1f} R@5 {r5:.1f}")

# Reloaded
for name in ["dense","rel_hard","mlpT","splade","D_L2"]:
    arr=data[name]
    r2=recall(arr, gold_ids, 2)
    r5=recall(arr, gold_ids, 5)
    print(f"reload {name} R@2 {r2:.1f} R@5 {r5:.1f}")

# Assert match
for name in ["dense","rel_hard","mlpT","splade","D_L2"]:
    arr1={"dense":dense,"rel_hard":rel_hard,"mlpT":mlpT,"splade":splade,"D_L2":D_L2}[name]
    arr2=data[name]
    assert np.array_equal(arr1, arr2), f"mismatch {name}"
print("PASS ranking persistence exact")
