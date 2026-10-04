"""PHASE 0/1/3 -- official balanced partitioners on Modal CPU.

The previous program had to substitute P2_METIS_STRONG and a clique-expansion surrogate for
KaHIP and KaHyPar because neither has a win32 / CPython-3.13 wheel.  That was an environment
limit, not an algorithmic one, so the real implementations run here on Linux CPU.

  H1_KAHIP_STRONG            KaFFPa strong
  H2_KAHIP_CONNECTED         KaFFPa strong + connected-block mode
  H3_MTKAHYPAR_GRAPH         Mt-KaHyPar, graph partitioning
  H4_MTKAHYPAR_TRUE_HYPERGRAPH   Mt-KaHyPar on a genuine hypergraph (km1), the thing the
                             clique expansion was only ever approximating

NO GPU is requested anywhere in this file.  The image compiles KaHIP once and caches it in the
layer, so nothing is rebuilt per corpus.

  MODAL_PROFILE=<name> modal run scratchpad/modal_partition.py::run --ds metaqa --methods H1_KAHIP_STRONG,H3_MTKAHYPAR_GRAPH
  MODAL_PROFILE=<name> modal run scratchpad/modal_partition.py::probe
"""
import modal

app = modal.App("crag-partition-cpu")
vol = modal.Volume.from_name("crag-partition", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "cmake", "libtbb-dev", "libhwloc-dev",
                 "libboost-program-options-dev", "python3-dev", "scons", "wget", "ca-certificates",
                 # KaHIP's CMakeLists does find_package(MPI) unconditionally; without it
                 # no Makefile is generated and deploy/ ends up holding only the header
                 "libopenmpi-dev", "openmpi-bin")
    .pip_install("numpy<2.0", "scipy")
    # KaHIP from source: the pip wheel does not ship the kaffpa binary, and --connected_blocks
    # (H2) is a binary flag.  Pinned by recording the commit we actually built.
    .run_commands(
        "git clone --depth 1 https://github.com/KaHIP/KaHIP /opt/KaHIP",
        "cd /opt/KaHIP && git rev-parse HEAD > /opt/KAHIP_COMMIT",
        # KaHIP defaults to -march=native.  Modal builds and runs on different CPUs, so a
        # natively-tuned binary dies with SIGILL (rc -4) at run time.  Pin a portable baseline.
        "cd /opt/KaHIP && grep -rl 'march=native' --include=CMakeLists.txt . | "
        "xargs -r sed -i 's/-march=native/-march=x86-64-v2/g'",
        "cd /opt/KaHIP && ./compile_withcmake.sh",
        # prove the binary actually RUNS on this CPU, not merely that it linked
        "cd /tmp && echo '4 4' > t.graph && echo '2 3' >> t.graph && echo '1 3' >> t.graph "
        "&& echo '1 2 4' >> t.graph && echo '3' >> t.graph "
        "&& /opt/KaHIP/deploy/kaffpa t.graph --k=2 --preconfiguration=strong "
        "--output_filename=t.part && cat t.part",
    )
    # Mt-KaHyPar: official python interface, true hypergraph partitioning with km1.
    .run_commands("pip install mtkahypar || pip install mt-kahypar || true")
    # no local dir is mounted: every input and output lives on the volume, and mounting the
    # live scratchpad made the build race the running sweep's log writes.
)


@app.function(image=image, cpu=8.0, memory=32768, timeout=3600, volumes={"/vol": vol})
def probe():
    """PHASE 0 -- record exactly what got built, before trusting any partition it produces."""
    import os, subprocess, json, multiprocessing
    out = {"cpu_count": multiprocessing.cpu_count()}
    try:
        out["KAHIP_COMMIT"] = open("/opt/KAHIP_COMMIT").read().strip()
    except Exception as e:
        out["KAHIP_COMMIT"] = f"ERR {e}"
    deploy = "/opt/KaHIP/deploy"
    out["kahip_deploy"] = sorted(os.listdir(deploy)) if os.path.isdir(deploy) else "MISSING"
    for exe in ("kaffpa", "graphchecker"):
        p = os.path.join(deploy, exe)
        out[f"{exe}_exists"] = os.path.exists(p)
        if os.path.exists(p):
            r = subprocess.run([p, "--help"], capture_output=True, text=True)
            txt = (r.stdout or "") + (r.stderr or "")
            out[f"{exe}_has_connected_blocks"] = "connected_blocks" in txt
            out[f"{exe}_preconfigurations"] = [w for w in ("strong", "eco", "fast", "fastsocial")
                                               if w in txt]
    try:
        import mtkahypar
        out["mtkahypar"] = getattr(mtkahypar, "__version__", "installed")
        out["mtkahypar_attrs"] = [a for a in dir(mtkahypar) if not a.startswith("_")][:40]
    except Exception as e:
        out["mtkahypar"] = f"ERR {e}"
    try:
        out["mem_total_kb"] = int([l for l in open("/proc/meminfo")
                                   if "MemTotal" in l][0].split()[1])
    except Exception:
        pass
    print(json.dumps(out, indent=1))
    return out


@app.function(image=image, cpu=16.0, memory=65536, timeout=21600, volumes={"/vol": vol})
def partition(ds: str, method: str, graph: str = "G0_TOPOLOGY_C"):
    """Run ONE partitioner on ONE exported graph.  Inputs and outputs both live on the volume."""
    import os, json, time, resource, subprocess
    import numpy as np
    src = f"/vol/graphs/{ds}__{graph}.npz"
    assert os.path.exists(src), f"missing {src} -- upload the exported graph first"
    z = np.load(src)
    ptr, idx = z["ptr"], z["idx"]
    k = int(z["k"][0])
    N = len(ptr) - 1
    os.makedirs("/vol/parts", exist_ok=True)
    os.makedirs("/vol/meta", exist_ok=True)
    dst = f"/vol/parts/{ds}__{graph}__{method}.npy"
    # production used pymetis ufactor=30 == 1 + 30/1000 == 3% allowed imbalance
    EPS = 0.03
    meta = {"ds": ds, "method": method, "graph": graph, "N": N, "k": k,
            "edges_directed": int(len(idx)), "imbalance_tolerance": EPS,
            "matches_production_ufactor": 30}
    t0 = time.time()

    if method.startswith("H1_KAHIP") or method.startswith("H2_KAHIP"):
        # KaHIP metis-format input: first line "N M", then one line of 1-based neighbours per node
        gp = "/tmp/g.graph"
        with open(gp, "w") as f:
            f.write(f"{N} {len(idx)//2}\n")
            for v in range(N):
                nb = idx[ptr[v]:ptr[v + 1]]
                f.write(" ".join(map(str, (nb + 1).tolist())) + "\n")
        out = "/tmp/part.txt"
        cmd = ["/opt/KaHIP/deploy/kaffpa", gp, f"--k={k}", "--preconfiguration=strong",
               f"--imbalance={EPS*100:g}", "--seed=0", f"--output_filename={out}"]
        if method.startswith("H2_"):
            cmd.append("--connected_blocks")
        r = subprocess.run(cmd, capture_output=True, text=True)
        meta["cmd"] = " ".join(cmd)
        meta["stdout_tail"] = (r.stdout or "")[-2000:]
        meta["stderr_tail"] = (r.stderr or "")[-2000:]
        meta["returncode"] = r.returncode
        if r.returncode != 0 or not os.path.exists(out):
            meta["status"] = "FAILED"
            json.dump(meta, open(f"/vol/meta/{ds}__{graph}__{method}.json", "w"), indent=1)
            vol.commit()
            return meta
        mem = np.loadtxt(out, dtype=np.int64)
    else:
        import mtkahypar
        # mtkahypar 1.6 API: initialize(threads) -> Initializer, which builds contexts and graphs.
        # DETERMINISTIC_QUALITY is the fixed high-quality preset the program asks for; it is the
        # same preset for every corpus and every representation.  km1 is the hypergraph objective.
        init = mtkahypar.initialize(int(os.environ.get("MTK_THREADS", "16")))
        ctx = init.context_from_preset(mtkahypar.PresetType.DETERMINISTIC_QUALITY)
        ctx.set_partitioning_parameters(k, EPS, mtkahypar.Objective.KM1)
        mtkahypar.set_seed(0)
        try:
            ctx.logging = False
        except Exception:
            pass
        meta["preset"] = "DETERMINISTIC_QUALITY"
        meta["objective"] = "KM1"
        meta["epsilon"] = EPS
        if method.startswith("H4_"):
            hp = f"/vol/graphs/{ds}__{graph}__hyper.npz"
            assert os.path.exists(hp), f"missing hypergraph {hp}"
            h = np.load(hp)
            eptr, eidx = h["eptr"].astype(np.int64), h["eidx"].astype(np.int64)
            hyperedges = [eidx[eptr[t]:eptr[t + 1]].tolist() for t in range(len(eptr) - 1)]
            meta["hyperedges"] = len(hyperedges)
            meta["pins"] = int(len(eidx))
            hg = init.create_hypergraph(ctx, N, len(hyperedges), hyperedges)
        else:
            # upper triangle only: create_graph wants each undirected edge once
            esrc = np.repeat(np.arange(N, dtype=np.int64), np.diff(ptr))
            edst = idx.astype(np.int64)
            up = esrc < edst
            edges = list(zip(esrc[up].tolist(), edst[up].tolist()))
            meta["edges_undirected"] = len(edges)
            if "w" in z:
                w = z["w"].astype(np.int64)[up]
                hg = init.create_graph(ctx, N, len(edges), edges,
                                       [1] * N, w.tolist())
                meta["weighted"] = True
            else:
                hg = init.create_graph(ctx, N, len(edges), edges)
                meta["weighted"] = False
        part = hg.partition(ctx)
        mem = np.array([part.block_id(i) for i in range(N)], np.int64)
        for fld, fn in (("objective_km1", "km1"), ("imbalance", "imbalance"),
                        ("objective_cut", "cut")):
            try:
                meta[fld] = float(getattr(part, fn)())
            except Exception:
                pass

    sizes = np.bincount(mem, minlength=k)
    meta.update({
        "status": "OK", "wall_seconds": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "blocks_used": int((sizes > 0).sum()),
        "size_min": int(sizes.min()), "size_max": int(sizes.max()),
        "size_mean": float(sizes.mean()),
        "size_cv": round(float(sizes.std() / max(sizes.mean(), 1e-9)), 4),
        "max_over_mean": round(float(sizes.max() / max(sizes.mean(), 1e-9)), 4)})
    np.save(dst, mem.astype(np.int32))
    json.dump(meta, open(f"/vol/meta/{ds}__{graph}__{method}.json", "w"), indent=1)
    vol.commit()
    print(json.dumps(meta, indent=1))
    return meta


@app.local_entrypoint()
def run(ds: str = "metaqa", methods: str = "H1_KAHIP_STRONG", graph: str = "G0_TOPOLOGY_C"):
    for m in methods.split(","):
        print(partition.remote(ds, m.strip(), graph))


@app.local_entrypoint()
def matrix(dss: str, cells: str):
    """Fan the (partitioner x representation) matrix out in parallel.

    cells are METHOD:GRAPH pairs, e.g. H1_KAHIP_STRONG:G0_TOPOLOGY_C,H4_...:G2_TRUE_HYPERGRAPH
    """
    import json
    args = [(d.strip(), c.split(":")[0], c.split(":")[1])
            for d in dss.split(",") for c in cells.split(",")]
    print(f"launching {len(args)} partition jobs")
    for r in partition.starmap(args, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            print(f"EXC {type(r).__name__}: {str(r)[:300]}")
        else:
            print(json.dumps({k: r.get(k) for k in
                              ("ds", "method", "graph", "status", "wall_seconds", "peak_rss_mb",
                               "blocks_used", "size_min", "size_max", "max_over_mean",
                               "objective_km1", "returncode")}))


@app.function(image=image, cpu=16.0, memory=32768, timeout=3600, volumes={"/vol": vol})
def diag():
    """One-off: find out WHY compile_withcmake.sh produced no binaries, and pin the mtkahypar API.

    The image step piped the build through `tail`, which masked its exit code -- so the failure
    was invisible.  This reproduces it with the error visible.
    """
    import subprocess, json, inspect
    out = {}
    r = subprocess.run("cd /opt/KaHIP && bash ./compile_withcmake.sh", shell=True,
                       capture_output=True, text=True)
    out["rc"] = r.returncode
    out["tail"] = ((r.stdout or "") + (r.stderr or ""))[-4000:]
    r2 = subprocess.run("ls /opt/KaHIP/deploy; ls /opt/KaHIP/build 2>/dev/null | head",
                        shell=True, capture_output=True, text=True)
    out["deploy_after"] = r2.stdout
    import mtkahypar
    def _try(k, fn):
        try:
            out[k] = fn()
        except Exception as e:
            out[k] = f"ERR {type(e).__name__}: {e}"
    # pybind builtins have no inspect.signature -- their __doc__ carries the real signature
    _try("initialize_doc", lambda: (mtkahypar.initialize.__doc__ or "")[:400])
    _try("set_seed_doc", lambda: (mtkahypar.set_seed.__doc__ or "")[:200])
    _try("Context_init_doc", lambda: (mtkahypar.Context.__init__.__doc__ or "")[:400])
    _try("Context_methods", lambda: [m for m in dir(mtkahypar.Context)
                                     if not m.startswith("_")])
    _try("Graph_init_doc", lambda: (mtkahypar.Graph.__init__.__doc__ or "")[:900])
    _try("Graph_methods", lambda: [m for m in dir(mtkahypar.Graph) if not m.startswith("_")])
    _try("Hypergraph_init_doc", lambda: (mtkahypar.Hypergraph.__init__.__doc__ or "")[:900])
    _try("PresetType", lambda: [m for m in dir(mtkahypar.PresetType) if not m.startswith("_")])
    _try("Objective", lambda: [m for m in dir(mtkahypar.Objective) if not m.startswith("_")])
    _try("PartitionedHypergraph_methods",
         lambda: [m for m in dir(mtkahypar.PartitionedHypergraph) if not m.startswith("_")])
    _try("Initializer_methods", lambda: [m for m in dir(mtkahypar.Initializer)
                                         if not m.startswith("_")])
    for nm in ("create_graph", "create_hypergraph", "context_from_preset"):
        _try(f"Initializer.{nm}", lambda nm=nm: (getattr(mtkahypar.Initializer, nm).__doc__ or "")[:700])
    _try("Graph.partition", lambda: (mtkahypar.Graph.partition.__doc__ or "")[:400])
    _try("Hypergraph.partition", lambda: (mtkahypar.Hypergraph.partition.__doc__ or "")[:400])
    _try("Context.set_partitioning_parameters",
         lambda: (mtkahypar.Context.set_partitioning_parameters.__doc__ or "")[:400])
    print(json.dumps(out, indent=1))
    return out


@app.local_entrypoint()
def rundiag():
    diag.remote()
