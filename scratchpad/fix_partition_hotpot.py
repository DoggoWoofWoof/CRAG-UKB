import os, sys
sys.path.insert(0, os.path.abspath("."))
import json, numpy as np, faiss, logging
from src.core.engine import CoreEngine
from src.experiments.l1_universal_head import _load
from src.experiments.l1_rerank100 import _feats, _rr

logging.basicConfig(level=logging.INFO)
Ps=[1,3,5,10,20,50,100,200]

def process(ds, te_cap):
    print(f"[fix] {ds} te_cap={te_cap}")
    try:
        data=_load(ds, "gte_qwen", limit=8000, tr_cap=3000, te_cap=te_cap)
        X=data["X"]; npart=data["npart"]; mem_idx=data["mem_idx"]; hard=data["hard"]
        qte, ste, gte=data["test"]
        print(f"  nq={len(gte)} npart={npart} X {X.shape}")
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
                if f==1.0:
                    full+=1; allc+=1
                if f>0: anyc+=1
            res={"P":P,"effective_P":eff,"saturated":sat,"FullCov":round(100*full/max(1,nq),2),"AnyCov":round(100*anyc/max(1,nq),2),"AllCov":round(100*allc/max(1,nq),2),"mean_gold_fraction":round(100*frac/max(1,nq),2),"nq":nq,"npart":npart}
            results[str(P)]=res
            print(f"    P{P} Full{res['FullCov']} Any{res['AnyCov']}")
        json.dump(results, open(f"scratchpad/partition_curves/{ds}_A.json","w"), indent=2)
        return results
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error":str(e)}

# Try squad with 2000 (cache hit)
for ds, cap in [("squad_clean",2000),("metaqa",2000),("hotpotqa_clean",500)]:
    process(ds, cap)

# Merge
import glob
out={}
for p in glob.glob("scratchpad/partition_curves/*_A.json"):
    ds=p.split("\\")[-1].split("_A.json")[0] if "\\" in p else p.split("/")[-1].split("_A.json")[0]
    # fallback
    import pathlib
    ds=pathlib.Path(p).name.replace("_A.json","")
    out[ds]=json.load(open(p))
# Also include previous
for ds in ["2wiki_clean","musique_clean","webqsp"]:
    if ds not in out:
        try: out[ds]=json.load(open(f"scratchpad/partition_curves/{ds}_A.json"))
        except: pass
json.dump(out, open("results/L1/partition_curves_A.json","w"), indent=2)
print("-> merged 6 datasets", list(out.keys()))
