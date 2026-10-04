import os, sys
sys.path.insert(0, os.path.abspath("."))
import json, numpy as np, faiss
from src.core.engine import CoreEngine
from src.experiments.l1_universal_head import _load
from src.experiments.l1_rerank100 import _feats, _rr
Ps=[1,3,5,10,20,50,100,200]
ds="hotpotqa_clean"; te_cap=500
print(f"[hotpot] loading {ds} te_cap {te_cap}")
data=_load(ds, "gte_qwen", limit=8000, tr_cap=3000, te_cap=te_cap)
X=data["X"]; npart=data["npart"]; mem_idx=data["mem_idx"]; hard=data["hard"]
qte, ste, gte=data["test"]
print(f"  nq {len(gte)} npart {npart} X {X.shape}")
idx=faiss.IndexFlatIP(X.shape[1]); idx.add(X)
_, I = idx.search(qte, 100)
S,M=_feats(I, mem_idx, npart, topn=200)
votes=_rr(S)+_rr(M)
ranking=np.argsort(-votes, axis=1)
results={}
for P in Ps:
    eff=min(P,npart); sat=P>npart
    topP=[set(ranking[qi,:eff]) for qi in range(len(ranking))]
    full=anyc=allc=0; frac=0; nq=0
    for qi,gg in enumerate(gte):
        if not gg: continue
        nq+=1
        hits=[1 if set(mem_idx[g]) & topP[qi] else 0 for g in gg]
        f=sum(hits)/len(hits) if hits else 0
        frac+=f
        if f==1.0: full+=1; allc+=1
        if f>0: anyc+=1
    res={"P":P,"effective_P":eff,"saturated":sat,"FullCov":round(100*full/max(1,nq),2),"AnyCov":round(100*anyc/max(1,nq),2),"AllCov":round(100*allc/max(1,nq),2),"mean_gold_fraction":round(100*frac/max(1,nq),2),"nq":nq,"npart":npart}
    results[str(P)]=res
    print(f" P{P} Full{res['FullCov']}")
json.dump(results, open("scratchpad/partition_curves/hotpotqa_clean_A.json","w"), indent=2)
# merge
import glob, pathlib
out={}
for p in glob.glob("scratchpad/partition_curves/*_A.json"):
    name=pathlib.Path(p).name.replace("_A.json","")
    out[name]=json.load(open(p))
json.dump(out, open("results/L1/partition_curves_A.json","w"), indent=2)
print("-> done 6", list(out.keys()))
