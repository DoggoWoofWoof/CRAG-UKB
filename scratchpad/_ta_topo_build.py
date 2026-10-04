import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import _ta_prepartition as TA
t0 = time.time()
ds = sys.argv[1]
hard, mem, npart, (ap, ai), deg, id2row = TA.load_topology(ds, lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True))
print(f"DONE {ds} docs={len(hard)} npart={npart} undirected_edges={int(ap[-1])//2} "
      f"deg_mean={deg.mean():.2f} deg_max={deg.max()} isolated={int((deg==0).sum())} "
      f"({time.time()-t0:.0f}s)", flush=True)
