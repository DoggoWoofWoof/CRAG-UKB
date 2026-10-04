import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.getcwd())
import _l1cg_core as CG
T0 = time.time()
for ds in CG.DSETS:
    t = time.time()
    S = CG.substrate(ds)
    CG.part_graph(ds, print)
    CG.proposals(ds, S, print)
    print(f"[{time.time()-T0:7.1f}s] {ds} done in {time.time()-t:.1f}s", flush=True)
