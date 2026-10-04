import sys,time,json,numpy as np,os; sys.path.insert(0,'scratchpad'); sys.path.insert(0,'.')
import l2_shapley as SH
os.makedirs(SH.OUTD,exist_ok=True)
summ={}
for ds in SH.PILOTS:
    for sp in SH.SPLITS:
        out=SH.compute_split(ds,sp,verbose=True)
        np.savez(f"{SH.OUTD}/{ds}_{sp}.npz",**{k:v for k,v in out.items() if not k.startswith('_')})
        summ[f"{ds}/{sp}"]={'n':out['_n_computed'],'sec':round(out['_runtime_s'],1)}
        print(f"SAVED {ds}/{sp} n={out['_n_computed']} {out['_runtime_s']:.0f}s",flush=True)
json.dump(summ,open('scratchpad/_shapley_full_summary.json','w'),indent=1)
print("SHAPLEY_FULL_ALL_DONE",json.dumps(summ),flush=True)
