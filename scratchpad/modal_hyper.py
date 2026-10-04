"""UNIVERSAL TRUE-HYPERGRAPH PHASE -- Mt-KaHyPar on Modal CPU, one representation per call.

The partitioner is FIXED here on purpose.  The previous program established that swapping the
algorithm changes nothing (H3_MTKAHYPAR_GRAPH x G0_TOPOLOGY_C is null on all six corpora) while
changing the REPRESENTATION moves corpora.  So this app varies only the hypergraph that goes in:

  H0_FIXED_CAP        the previous H4 build, reproduced bit-for-bit  (cap 25, drop over cap)
  H1_FULL_WEIGHTED    every hyperedge, no size cap, w(e) = 1/(|e|-1)
  H2_Q90/Q95/Q99      ONE universal percentile rule; the numeric cap follows the corpus
  H3_B05/B10/B20      cap = c * target_block_size, ONE universal c
  H4_SPLIT_PRESERVE   oversized groups deterministically split, NO pin discarded

Everything else is held: mtkahypar version, DETERMINISTIC_QUALITY preset, KM1, k = production
npart, epsilon = 0.03, seed 0, same node universe.

Mt-KaHyPar runs in a CHILD PROCESS.  Its Python binding holds the GIL for the whole of
hg.partition(ctx), and the hyperedge list construction above it holds it too, so an in-process
call longer than Modal's 900 s runner heartbeat gets killed as unresponsive.  subprocess.run
releases the GIL; that is the only reason this indirection exists and it cannot change a
partition.

NO GPU is requested anywhere in this file.

  MODAL_PROFILE=<name> modal run scratchpad/modal_hyper.py::apisig
  MODAL_PROFILE=<name> modal run scratchpad/modal_hyper.py::matrix --dss metaqa --tags H1_FULL_WEIGHTED__SK
"""
import modal

app = modal.App("crag-hypergraph-cpu")
vol = modal.Volume.from_name("crag-partition", create_if_missing=True)

# Lean image: mtkahypar from pip, nothing else.  The earlier spec inherited a from-source KaHIP
# compile from modal_partition.py so that the cached layer would be reused; that cache lived in a
# workspace that ran out of credit mid-phase, and this phase never calls KaHIP, so carrying a
# ~15 minute compile into a fresh workspace would be paid-for dead weight.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libtbb-dev", "libhwloc-dev")
    .pip_install("numpy<2.0")
    .run_commands("pip install mtkahypar")
)


@app.function(image=image, cpu=4.0, memory=8192, timeout=900, volumes={"/vol": vol})
def apisig():
    """Pin the create_hypergraph signature BEFORE relying on weighted hyperedges.

    The binding is pybind11, so inspect.signature does not work -- the real signature lives in
    __doc__.  Acceptance of a 6-argument call proves nothing on its own: a binding that silently
    dropped the weights would look identical.  So this also (a) reads the weights back out of the
    built hypergraph and (b) runs a DISCRIMINATING partition -- a path 0-1-2-3 cut into two blocks
    of two, where the cheapest cut is the middle edge when weights are equal and the two outer
    edges when the middle edge is expensive.  If the returned block ids do not flip, the weights
    are not reaching the objective and every weighted build in this phase is invalid.
    """
    import json, os
    out = {}
    try:
        import mtkahypar
        out["version"] = getattr(mtkahypar, "__version__", "installed")
        for nm in ("create_hypergraph", "create_graph"):
            out[nm + "_doc"] = (getattr(mtkahypar.Initializer, nm).__doc__ or "")[:1500]
        out["Hypergraph_methods"] = [m for m in dir(mtkahypar.Hypergraph)
                                     if not m.startswith("_")]
        init = mtkahypar.initialize(2)

        def ctx_of(k, eps):
            c = init.context_from_preset(mtkahypar.PresetType.DETERMINISTIC_QUALITY)
            c.set_partitioning_parameters(k, eps, mtkahypar.Objective.KM1)
            try:
                c.logging = False
            except Exception:
                pass
            return c

        # (1) acceptance + weight read-back
        c = ctx_of(2, 0.03)
        he = [[0, 1, 2], [2, 3]]
        for label, args in (("unweighted", (c, 4, 2, he)),
                            ("node+edge_weights", (c, 4, 2, he, [1, 1, 1, 1], [7, 3]))):
            try:
                hg = init.create_hypergraph(*args)
                p = hg.partition(c)
                out["ok_" + label] = [int(p.block_id(i)) for i in range(4)]
                try:
                    out["readback_" + label] = [int(hg.edge_weight(i)) for i in range(2)]
                except Exception as e:
                    out["readback_" + label] = "ERR %s: %s" % (type(e).__name__, e)
            except Exception as e:
                out["ok_" + label] = "ERR %s: %s" % (type(e).__name__, e)

        # (2) discriminating test: path 0-1-2-3, k=2, blocks of exactly 2
        path = [[0, 1], [1, 2], [2, 3]]
        for label, ew in (("equal_1_1_1", [1, 1, 1]), ("heavy_middle_1_100_1", [1, 100, 1])):
            cc = ctx_of(2, 0.001)
            mtkahypar.set_seed(0)
            hg = init.create_hypergraph(cc, 4, 3, path, [1, 1, 1, 1], ew)
            p = hg.partition(cc)
            blk = [int(p.block_id(i)) for i in range(4)]
            out["discriminate_" + label] = {"blocks": blk, "km1": float(p.km1()),
                                            "same_block_1_2": blk[1] == blk[2]}
        d1 = out.get("discriminate_equal_1_1_1", {})
        d2 = out.get("discriminate_heavy_middle_1_100_1", {})
        out["EDGE_WEIGHTS_AFFECT_OBJECTIVE"] = bool(
            d1.get("same_block_1_2") is False and d2.get("same_block_1_2") is True)
    except Exception as e:
        out["fatal"] = "ERR %s: %s" % (type(e).__name__, e)
    os.makedirs("/vol/hmeta", exist_ok=True)
    json.dump(out, open("/vol/hmeta/MTKAHYPAR_API.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(out, indent=1))
    return out


WORKER = r"""
import sys, json, resource
import numpy as np
import mtkahypar

cfg = json.load(open(sys.argv[1]))
z = np.load(cfg['src'])
N, k, EPS = int(z['N'][0]), int(z['k'][0]), cfg['EPS']
eptr = z['eptr'].astype(np.int64)
eidx = z['eidx'].astype(np.int64)
ew = z['ew'].astype(np.int64) if 'ew' in z.files else None

init = mtkahypar.initialize(cfg['threads'])
ctx = init.context_from_preset(mtkahypar.PresetType.DETERMINISTIC_QUALITY)
ctx.set_partitioning_parameters(k, EPS, mtkahypar.Objective.KM1)
mtkahypar.set_seed(0)
try:
    ctx.logging = False
except Exception:
    pass

ne = len(eptr) - 1
he = [eidx[eptr[t]:eptr[t + 1]].tolist() for t in range(ne)]
st = {'hyperedges': ne, 'pins': int(len(eidx)), 'N': N, 'k': k}
if ew is None:
    hg = init.create_hypergraph(ctx, N, ne, he)
    st['edge_weighted'] = False
else:
    hg = init.create_hypergraph(ctx, N, ne, he, [1] * N, ew.tolist())
    st['edge_weighted'] = True
    st['ew_min'] = int(ew.min()); st['ew_max'] = int(ew.max())
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
st['child_peak_rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
np.save(cfg['out'], mem)
json.dump(st, open(cfg['stats'], 'w'))
"""


SMALL = ("metaqa", "musique_clean", "2wiki_clean", "squad_clean")


def _run(ds: str, tag: str):
    """Partition ONE exported hypergraph.  Input and output both live on the volume."""
    import os, sys, json, time, resource, subprocess
    import numpy as np

    t0 = time.time()
    src = "/vol/hgraphs/%s__%s.npz" % (ds, tag)
    meta = {"ds": ds, "tag": tag, "preset": "DETERMINISTIC_QUALITY", "objective": "KM1",
            "epsilon": 0.03, "seed": 0, "runs_in_child_process": True}
    if not os.path.exists(src):
        meta.update(status="MISSING_INPUT", src=src)
        return meta

    wp = "/tmp/hyper_worker.py"
    open(wp, "w").write(WORKER)
    cfg = {"src": src, "EPS": 0.03, "threads": int(os.environ.get("MTK_THREADS", "16")),
           "out": "/tmp/mem_%s_%s.npy" % (ds, tag), "stats": "/tmp/st_%s_%s.json" % (ds, tag)}
    cp = "/tmp/cfg_%s_%s.json" % (ds, tag)
    json.dump(cfg, open(cp, "w"))
    r = subprocess.run([sys.executable, wp, cp], capture_output=True, text=True)
    meta["returncode"] = r.returncode
    meta["stderr_tail"] = (r.stderr or "")[-2000:]
    os.makedirs("/vol/hmeta", exist_ok=True)
    os.makedirs("/vol/hparts", exist_ok=True)
    if r.returncode != 0 or not os.path.exists(cfg["out"]):
        meta["status"] = "FAILED"
        meta["wall_seconds"] = round(time.time() - t0, 1)
        json.dump(meta, open("/vol/hmeta/%s__%s.json" % (ds, tag), "w"), indent=1)
        vol.commit()
        return meta

    mem = np.load(cfg["out"])
    meta.update(json.load(open(cfg["stats"])))
    k = int(meta["k"])
    sizes = np.bincount(mem, minlength=k)
    nz = sizes[sizes > 0]
    meta.update({
        "status": "OK",
        "wall_seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                 resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
                             / 1024, 1),
        "blocks_used": int((sizes > 0).sum()),
        "size_min": int(nz.min()), "size_max": int(sizes.max()),
        "size_median": float(np.median(nz)), "size_mean": float(sizes.mean()),
        "size_p90": float(np.percentile(nz, 90)), "size_p99": float(np.percentile(nz, 99)),
        "max_over_mean": round(float(sizes.max() / max(sizes.mean(), 1e-9)), 4),
        "size_cv": round(float(nz.std() / max(nz.mean(), 1e-9)), 4),
    })
    np.save("/vol/hparts/%s__%s.npy" % (ds, tag), mem)
    json.dump(meta, open("/vol/hmeta/%s__%s.json" % (ds, tag), "w"), indent=1)
    vol.commit()
    return meta


# Resources follow the corpus, not the experiment: the four small corpora never exceeded a few GB.
# The large H4_SPLIT_PRESERVE build measured on webqsp needed far more than pin count alone would
# suggest -- 125.0 GB actual peak against a 64 GB request it was allowed to burst past (8.19M pins).
# hotpotqa_clean has 10.22M pins (1.25x webqsp's); linear scaling alone puts it near 156 GB, and the
# split/duplication construction may not scale linearly, so this is raised to 256 GB for real margin
# rather than gambling a ~90 minute run on a repeat of that undersized guess.
@app.function(image=image, cpu=8.0, memory=16384, timeout=43200, volumes={"/vol": vol})
def partition_sm(ds: str, tag: str):
    return _run(ds, tag)


@app.function(image=image, cpu=16.0, memory=262144, timeout=43200, volumes={"/vol": vol})
def partition_lg(ds: str, tag: str):
    return _run(ds, tag)


# webqsp's own H4_SPLIT_PRESERVE__SK build (same N=781,485, k=7,814 -- famset doesn't change node
# universe or k) already measured 125.0 GB actual peak at 8.19M pins (see partition_lg's comment).
# SKN adds NER hyperedges, raising pins to 10.90M (1.33x SK's).  hotpotqa_clean's SK build has
# slightly MORE pins (10.22M) but 1.54x FEWER nodes/blocks than webqsp, and still measured 245.6 GB
# actual against partition_lg's 256 GB request -- 94% utilized, no real margin left.  webqsp+SKN
# combines webqsp's larger N/k with more pins than that already-tight hotpot run, so reusing
# partition_lg unmodified risks a repeat OOM after a multi-hour job.  Sized up rather than gambling.
# 409600 MiB was rejected outright by Modal ("must be between 128 and 344064 MiB" for this
# workspace) -- 344064 MiB (336 GB) is the actual platform ceiling here, used as-is below.
@app.function(image=image, cpu=16.0, memory=344064, timeout=43200, volumes={"/vol": vol})
def partition_xl(ds: str, tag: str):
    return _run(ds, tag)


@app.local_entrypoint()
def one_xl(ds: str, tag: str):
    """Single job on the XL (400 GB) tier -- for webqsp/hotpotqa_clean SKN, not covered by matrix's
    SMALL/large-only routing.  MODAL_PROFILE=<name> modal run scratchpad/modal_hyper.py::one_xl --ds webqsp --tag H4_SPLIT_PRESERVE__SKN"""
    import json
    r = partition_xl.remote(ds, tag)
    print(json.dumps(r, indent=1))


@app.local_entrypoint()
def matrix(dss: str, tags: str):
    """Fan (corpus x representation) out in parallel, routing each corpus to its size class."""
    import json
    keep = ("ds", "tag", "status", "wall_seconds", "peak_rss_mb", "hyperedges", "pins",
            "edge_weighted", "ew_readback_exact", "blocks_used", "size_min", "size_max",
            "max_over_mean", "size_cv", "objective_km1", "imbalance", "returncode")
    pairs = [(d.strip(), t.strip()) for d in dss.split(",") for t in tags.split(",")]
    sm = [p for p in pairs if p[0] in SMALL]
    lg = [p for p in pairs if p[0] not in SMALL]
    print("launching %d small + %d large hypergraph partition jobs" % (len(sm), len(lg)))
    hs = [(partition_sm, sm), (partition_lg, lg)]
    for fn, args in hs:
        if not args:
            continue
        for r in fn.starmap(args, order_outputs=False, return_exceptions=True,
                            wrap_returned_exceptions=False):
            if isinstance(r, Exception):
                print("EXC %s: %s" % (type(r).__name__, str(r)[:300]))
            else:
                print(json.dumps({k: r.get(k) for k in keep}))
