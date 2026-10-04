import sys, time; sys.path.insert(0,'scratchpad'); sys.path.insert(0,'.')
import l2_relation_qwen as Q
t0=time.time()
for ds in Q.PILOTS:
    for sp in Q.SPLITS:
        n=0
        while True:
            r=Q.build_sparse_edges(ds,sp,budget_s=110)
            n+=1
            if r is not None:
                print(f"COMPLETE {ds}/{sp} after {n} chunk(s) t={time.time()-t0:.0f}s", flush=True); break
print("ALL_EDGES_COMPLETE t=%.0fs"%(time.time()-t0), flush=True)
