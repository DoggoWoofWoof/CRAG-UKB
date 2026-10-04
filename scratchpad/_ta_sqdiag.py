import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, _ta_prepartition as TA
DS=os.environ.get("DSQ","squad_clean")
hard, mem, npart, adj, deg, id2row = TA.load_topology(DS)
mem_ptr, mem_flat = mem
adj_ptr, adj_idx = adj
n=len(hard)
# variant A: own partition only
mp_own=np.arange(n+1,dtype=np.int64); mf_own=hard.astype(np.int32)
# variant B: own + UNDIRECTED 1-hop
lists=[]
for i in range(n):
    s={int(hard[i])}
    for j in adj_idx[adj_ptr[i]:adj_ptr[i+1]]: s.add(int(hard[j]))
    lists.append(sorted(s))
ml=np.array([len(x) for x in lists],np.int64); mp_und=np.zeros(n+1,np.int64); mp_und[1:]=np.cumsum(ml)
mf_und=np.concatenate([np.array(x,np.int32) for x in lists])
B=f"data/ukb_storage/{DS}/gte_qwen/"
d=f"data/l2_corpus/{DS}/val"
off=np.load(f"{d}/query_offsets.npy"); pid=np.load(f"{d}/part_id.npy"); qm=json.load(open(f"{d}/query_meta.json"))
dA=np.load(B+"dense_top200_all.npy",mmap_mode="r"); sA=np.load(B+"splade_top200_all.npy",mmap_mode="r")
N=60; rows=[int(m["row_all"]) for m in qm[:N]]
stored=[set(int(x) for x in np.unique(pid[int(off[q]):int(off[q+1])])) for q in range(N)]
print(f"{DS}: stored |sel| mean = {np.mean([len(s) for s in stored]):.2f}  (P_MAIN should be 50)")
for mname,(mpx,mfx) in [("own_only",(mp_own,mf_own)),("dir_1hop",(mem_ptr,mem_flat)),("und_1hop",(mp_und,mf_und))]:
    for K in (50,100,200):
        dl=[np.asarray(dA[i][:K],np.int64) for i in rows]; sl=[np.asarray(sA[i][:K],np.int64) for i in rows]
        rk=TA.rrf_partitions([TA.partition_ranking(dl,(mpx,mfx),npart),TA.partition_ranking(sl,(mpx,mfx),npart)],npart)
        ov=np.mean([len(stored[q]&set(int(x) for x in rk[q][:50]))/max(len(stored[q]),1) for q in range(N)])
        exact=sum(stored[q]==set(int(x) for x in rk[q][:50]) for q in range(N))
        print(f"  mem={mname:9s} K={K:3d}  set-overlap={ov:.4f}  exact={exact}/{N}")
# dense-only / splade-only controls at the canonical setting
for nm,ls in [("DENSE_ONLY",[dA]),("SPLADE_ONLY",[sA])]:
    l=[np.asarray(ls[0][i][:100],np.int64) for i in rows]
    rk=TA.rrf_partitions([TA.partition_ranking(l,mem,npart)],npart)
    ov=np.mean([len(stored[q]&set(int(x) for x in rk[q][:50]))/max(len(stored[q]),1) for q in range(N)])
    print(f"  {nm:11s} K=100  set-overlap={ov:.4f}")
