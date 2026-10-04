import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, _ta_prepartition as TA
for DS in ("squad_clean","2wiki_clean","musique_clean","metaqa"):
    hard, mem, npart, _a, _d, id2row = TA.load_topology(DS)
    B=f"data/ukb_storage/{DS}/gte_qwen/"
    j=json.load(open(B+"query_ids_all.json")); si=j["split_indices"]
    rows=(si.get("val") or si["all"])[:100]
    dA=np.load(B+"dense_top200_all.npy",mmap_mode="r")
    mem_ptr, mem_flat = mem
    S=np.zeros((len(rows),npart),np.float32); M=np.zeros((len(rows),npart),np.float32)
    for qi,i in enumerate(rows):
        for r,nd in enumerate(np.asarray(dA[i][:100],np.int64)):
            nd=int(nd)
            if nd<0 or nd>=len(hard): continue
            w=1.0/(60+r); ps=mem_flat[mem_ptr[nd]:mem_ptr[nd+1]]
            S[qi,ps]+=w; np.maximum.at(M[qi],ps,w)
    # how many partitions share a vote value with another (ties the unstable argsort must break)
    tie_frac=[]; nz=[]
    for qi in range(len(rows)):
        v=S[qi]; u,c=np.unique(v,return_counts=True)
        tie_frac.append(float((c[c>1].sum())/npart)); nz.append(float((v>0).mean()))
    # and: how many DISTINCT vote levels exist in the contested band around rank 50
    print(f"{DS:15s} npart={npart:5d}  mem_idx_mean={np.mean(np.diff(mem_ptr)):6.2f}  "
          f"frac_partitions_in_a_TIE={np.mean(tie_frac):.4f}  frac_with_any_vote={np.mean(nz):.4f}")
