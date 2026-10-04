"""Local equivalent of the WORKER string in modal_hyper.py -- same algorithm, same preset, same
seed, run directly under WSL against files on the local disk instead of a Modal volume.

No 900s heartbeat exists locally, so there is no GIL-vs-watchdog reason to fork a child process
here; this file IS the child process, invoked once per (corpus, representation) by the caller.

  python3 scratchpad/_l1hu_local_worker.py <src.npz> <out.npy> <stats.json> [threads]
"""
import sys, json, time, resource
import numpy as np
import mtkahypar

src, out, stats_fp = sys.argv[1], sys.argv[2], sys.argv[3]
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 12

t0 = time.time()
z = np.load(src)
N, k, EPS = int(z['N'][0]), int(z['k'][0]), 0.03
eptr = z['eptr'].astype(np.int64)
eidx = z['eidx'].astype(np.int64)
ew = z['ew'].astype(np.int64) if 'ew' in z.files else None

init = mtkahypar.initialize(threads)
ctx = init.context_from_preset(mtkahypar.PresetType.DETERMINISTIC_QUALITY)
ctx.set_partitioning_parameters(k, EPS, mtkahypar.Objective.KM1)
mtkahypar.set_seed(0)
try:
    ctx.logging = False
except Exception:
    pass

ne = len(eptr) - 1
he = [eidx[eptr[t]:eptr[t + 1]].tolist() for t in range(ne)]
st = {'hyperedges': ne, 'pins': int(len(eidx)), 'N': N, 'k': k, 'threads': threads,
     'ran_on': 'local_wsl'}
if ew is None:
    hg = init.create_hypergraph(ctx, N, ne, he)
    st['edge_weighted'] = False
else:
    hg = init.create_hypergraph(ctx, N, ne, he, [1] * N, ew.tolist())
    st['edge_weighted'] = True
    st['ew_min'] = int(ew.min())
    st['ew_max'] = int(ew.max())
    rb = [int(hg.edge_weight(t)) for t in range(min(1000, ne))]
    st['ew_readback_exact'] = bool(rb == ew[:len(rb)].tolist())
    assert st['ew_readback_exact'], 'edge weights did not survive create_hypergraph'

part = hg.partition(ctx)
mem = np.array([part.block_id(i) for i in range(N)], np.int64)
for fld, fn in (('objective_km1', 'km1'), ('imbalance', 'imbalance'), ('objective_cut', 'cut')):
    try:
        st[fld] = float(getattr(part, fn)())
    except Exception:
        pass
st['peak_rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
st['wall_seconds'] = round(time.time() - t0, 1)
np.save(out, mem)
json.dump(st, open(stats_fp, 'w'), indent=1)
print(json.dumps(st, indent=1))
