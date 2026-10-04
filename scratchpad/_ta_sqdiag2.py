import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import numpy as np, _ta_prepartition as TA
for DS in ("squad_clean","2wiki_clean"):
    hard, mem, npart, adj, deg, id2row = TA.load_topology(DS)
    B=f"data/ukb_storage/{DS}/gte_qwen/"; d=f"data/l2_corpus/{DS}/val"
    off=np.load(f"{d}/query_offsets.npy"); pid=np.load(f"{d}/part_id.npy")
    cid=np.load(f"{d}/cand_ids.npy",mmap_mode="r"); dr=np.load(f"{d}/dense_rank.npy",mmap_mode="r")
    qm=json.load(open(f"{d}/query_meta.json"))
    dA=np.load(B+"dense_top200_all.npy",mmap_mode="r"); sA=np.load(B+"splade_top200_all.npy",mmap_mode="r")
    N=60; hit1=hit5=s1=0; rankagree=0; rn=0
    for q in range(N):
        r=int(qm[q]["row_all"]); s,e=int(off[q]),int(off[q+1])
        sel=set(int(x) for x in np.unique(pid[s:e]))
        dt=[int(x) for x in dA[r][:5]]; st=[int(x) for x in sA[r][:5]]
        hit1+=int(hard[dt[0]] in sel); hit5+=sum(hard[x] in sel for x in dt)/5.0
        s1+=int(hard[st[0]] in sel)
        # does the corpus's own stored dense_rank reproduce the current dense array?
        cands=np.asarray(cid[s:e]); ranks=np.asarray(dr[s:e])
        top=cands[np.argsort(ranks,kind="stable")[:20]]
        cur=[int(x) for x in dA[r][:200]]; curset={v:i for i,v in enumerate(cur)}
        inb=[curset.get(int(x),9999) for x in top]
        rankagree+=sum(1 for v in inb if v<20)/20.0; rn+=1
    print(f"{DS}: current dense-top1 partition in STORED scope {hit1}/{N}  dense-top5 frac {hit5/N:.3f}  splade-top1 {s1}/{N}")
    print(f"   stored best-dense-ranked 20 cands that are in CURRENT dense top-20: {rankagree/rn:.3f}")
