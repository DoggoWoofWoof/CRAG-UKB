import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, _ta_prepartition as TA
DS="squad_clean"
hard, mem, npart, _a, _d, id2row = TA.load_topology(DS)
mem_ptr, mem_flat = mem
B=f"data/ukb_storage/{DS}/gte_qwen/"
d=f"data/l2_corpus/{DS}/val"
off=np.load(f"{d}/query_offsets.npy"); pid=np.load(f"{d}/part_id.npy"); qm=json.load(open(f"{d}/query_meta.json"))
dA=np.load(B+"dense_top200_all.npy",mmap_mode="r"); sA=np.load(B+"splade_top200_all.npy",mmap_mode="r")
N=60; rows=[int(m["row_all"]) for m in qm[:N]]
dl=[np.asarray(dA[i][:100],np.int64) for i in rows]; sl=[np.asarray(sA[i][:100],np.int64) for i in rows]
PRd=TA.partition_ranking(dl,mem,npart); PRs=TA.partition_ranking(sl,mem,npart)
mine=TA.rrf_partitions([PRd,PRs],npart)
def votes(lists):
    S=np.zeros((N,npart),np.float32); M=np.zeros((N,npart),np.float32)
    for qi in range(N):
        for r,nd in enumerate(lists[qi]):
            nd=int(nd)
            if nd<0 or nd>=len(hard): continue
            w=1.0/(60+r); ps=mem_flat[mem_ptr[nd]:mem_ptr[nd+1]]
            S[qi,ps]+=w; np.maximum.at(M[qi],ps,w)
    return S,M
Sd,Md=votes(dl); Ss,Ms=votes(sl)
tied=0; untied=0; ex=None
for q in range(N):
    stored=set(int(x) for x in np.unique(pid[int(off[q]):int(off[q+1])]))
    mn=set(int(x) for x in mine[q][:50])
    onlyS=stored-mn; onlyM=mn-stored
    for a in onlyS:
        # is there a partition in onlyM with IDENTICAL dense AND splade vote pair?
        hit=any(Sd[q,a]==Sd[q,b] and Md[q,a]==Md[q,b] and Ss[q,a]==Ss[q,b] and Ms[q,a]==Ms[q,b] for b in onlyM)
        tied+=hit; untied+=(not hit)
        if not hit and ex is None:
            b=list(onlyM)[0] if onlyM else None
            ex=(q,a,b,float(Sd[q,a]),float(Sd[q,b]) if b is not None else None)
print(f"squad differing partitions: {tied} have an EXACT vote-tie counterpart, {untied} do not")
print(f"  -> tie-break non-determinism explains {tied/max(tied+untied,1):.1%} of the disagreement")
if ex: print(f"  first untied example: q={ex[0]} stored_only_p={ex[1]} mine_only_p={ex[2]} Sd={ex[3]:.6g} vs {ex[4]}")
