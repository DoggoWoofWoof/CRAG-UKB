"""Exact top-1000 retrieval caches for hotpotqa and 2wiki, on Modal.

WHY REMOTE AT ALL

  Measured on this machine, the local builder runs at 8.354e6 query-document pairs/s (metaqa
  dense, 407,513 x 43,234 in 2,109 s).  The three missing caches are 1.228e10 / 5.508e11 /
  1.154e12 pairs, i.e. webqsp 0.8 h, hotpotqa 36.6 h and 2wiki 76.7 h for both models --
  114 hours.  webqsp was built locally in 22 minutes.  These two are not.

WHAT IS AND IS NOT SHIPPED

  Nothing of the corpus.  The doc stores are on crag-data-volume already.  Only the bundle
  moves -- 1.94 GB: map.npz, a full shard census, the flat patch arrays, and the query matrix
  resolved locally.  See scratchpad/make_cache_bundle.py.

THE THREE THINGS THAT MUST MATCH THE LOCAL BUILDER, OR THESE ARE A DIFFERENT ARTIFACT WEARING
THE SAME FILENAME

  1  RANKING OVER POSITIONS, NOT DISTINCT ROWS.  The pointer index is many-to-one -- 12,581
     duplicate positions in hotpotqa, 7,122 in 2wiki -- because token-identical encoder inputs
     share one vector.  The corpus is RESOLVED once as distinct rows (cheap) and GATHERED to
     positions through pos2u before ranking, so every duplicate position competes on its own.
     Ranking distinct rows would return one representative per tie group and produce a
     top-1000 that no local run reproduces.

  2  THE TIE-BREAK.  build_retrieval_cache.order_key packs the float32 score's order-preserving
     unsigned form into the high 32 bits and 0xFFFFFFFF - position into the low 32, so equal
     scores resolve to the smaller canonical position.  torch has no usable uint64, so the same
     key is built SIGNED: the float32 bit pattern is made signed-comparable by
     `i if i >= 0 else i ^ 0x7FFFFFFF` (in int32, no widening), then key = mono32 * 2**32 + low.
     Algebraically mono_unsigned == mono32 + 2**31 in BOTH branches, so this key is exactly
     order_key's minus 2**63 -- a strictly increasing map, hence the same selection in the same
     order.  It is unpacked by the local builder's own unpack() after 2**63 is added back.

  3  THE SCORE DTYPE, which is where a GPU port silently goes wrong.  The local dense builder
     upcasts BOTH operands to float32 (build_retrieval_cache.py:94,101) and gets a float32
     score.  An fp16 matmul returning fp16 would round scores to ~5e-4 relative, manufacturing
     ties the local builder does not have and moving top-1000 membership at the boundary.  So
     the GEMM here is float32-in / float32-out on TF32 tensor cores.  TF32 carries an 8-bit
     exponent and a 10-bit mantissa and every stored value is an fp16 value, so the operands
     are represented EXACTLY -- identical numbers to numpy's, accumulated in fp32.  The only
     residual difference is summation order over dim=1536: the same class of difference as
     numpy on another BLAS, and the one the builder's own docstring already concedes when it
     withdraws its earlier claim of byte-reproducibility.

     splade needs no such argument.  It runs the local builder's scipy path unchanged, on CPU,
     with numpy and scipy pinned to this machine's versions.

  Blocking does not enter the result.  A row of the score matrix depends on one query only, and
  the merge keeps the k largest keys over the whole corpus, so DOC_BLOCK, the query chunk size
  and the splade query sharding are all free parameters.  Only dim is fixed.

THE PREFILTER, AND WHY IT IS NOT AN APPROXIMATION

  A naive per-block topk pushes all 5.5e11 / 1.15e12 (query, position) keys through a radix
  select -- several passes over 8 bytes each, which costs more than the GEMM.  Instead: after
  the first block `best` is sorted descending, so best[q, k-1] is the EXACT k-th largest key so
  far, and its score is a valid admission threshold.  An item with a strictly smaller score can
  never place, because key = mono32 * 2**32 + low with 0 <= low < 2**32, so a mono32 smaller by
  one already makes the key strictly smaller whatever the position.  An item with an EQUAL
  score can place, on position, so the threshold is `>=` and not `>`.  Later blocks therefore
  rank only the survivors, and the int64 key is never materialised for the ~99.99% of a block
  that cannot place.

  That is a proof, so it is also CHECKED: --check-blocks recomputes the first N post-threshold
  blocks the naive way from a saved copy of `best` and asserts bit-identical output.

VERIFICATION IS NOT A SEPARATE STAGE

  Every base shard is sha256'd AS IT IS READ during resolve, against the local census, and so
  are the bundle's own files.  One mismatch aborts before any scoring.  This is not paranoia:
  21 of 281 dense shards on a Modal volume were once the right size and the wrong bytes,
  because `modal volume put` skips on name -- and `volume ls` under-lists, so a count does not
  settle it either.  Hashing during resolve costs no extra remote I/O.

Run:
  modal run scratchpad/modal_retrieval_cache.py::census
  modal run scratchpad/modal_retrieval_cache.py --ds hotpotqa --model dense
  modal run scratchpad/modal_retrieval_cache.py --ds 2wiki --model splade --qshards 24
"""

import os

import modal

app = modal.App("crag-retrieval-cache")
vol = modal.Volume.from_name("crag-data-volume", create_if_missing=True)
bundle = modal.Volume.from_name("crag-cache-bundle", create_if_missing=True)
out = modal.Volume.from_name("crag-retrieval-out", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy==2.3.2", "scipy==1.16.1", "torch==2.8.0"))
VOLS = {"/root/vol": vol, "/root/bundle": bundle, "/root/out": out}

# TF32 throughput is what sets the dense wall clock; override if the class is unavailable.
# CRAG_GPU=none declares dense_run WITHOUT a GPU.  Modal validates every function in an app at
# deploy time, so an app that merely mentions an H100 is rejected outright by a workspace with
# no payment method -- which would block the splade CPU stages too, for a function that run
# never invokes.
GPU = os.environ.get("CRAG_GPU", "H100")
GPU_SPEC = None if GPU.strip().lower() in ("", "none") else GPU

K = 1000
DOC_BLOCK_DENSE = 262144
DOC_BLOCK_SPLADE = 250000       # the local builder's DOC_BLOCK
KEY_BUDGET = 500_000_000        # the local builder's KEY_BUDGET
TWO32 = 1 << 32
M32 = 0xFFFFFFFF
JOBS = [("hotpotqa", "dense"), ("hotpotqa", "splade"),
        ("2wiki", "dense"), ("2wiki", "splade")]


# ---------------------------------------------------------------- shared helpers

def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def order_key(scores, positions):
    """build_retrieval_cache.order_key, verbatim."""
    import numpy as np
    u = np.ascontiguousarray(scores, dtype=np.float32).view(np.uint32)
    mono = np.where((u >> 31) == 0, u | np.uint32(0x80000000), ~u)
    return (mono.astype(np.uint64) << np.uint64(32)) | \
           (np.uint64(0xFFFFFFFF) - positions.astype(np.uint64))


def unpack(keys):
    """build_retrieval_cache.unpack, verbatim."""
    import numpy as np
    mono = (keys >> np.uint64(32)).astype(np.uint32)
    u = np.where((mono >> 31) == 1, mono & np.uint32(0x7FFFFFFF), ~mono)
    s = u.view(np.float32)
    pos = (np.uint64(0xFFFFFFFF) - (keys & np.uint64(0xFFFFFFFF))).astype(np.int32)
    return s, pos


def take_top(keys, k):
    """build_retrieval_cache.take_top, verbatim."""
    import numpy as np
    if keys.shape[1] <= k:
        o = np.argsort(-keys, axis=1, kind="stable")
        return np.take_along_axis(keys, o, 1)
    p = np.argpartition(keys, keys.shape[1] - k, axis=1)[:, keys.shape[1] - k:]
    c = np.take_along_axis(keys, p, 1)
    o = np.argsort(-c, axis=1, kind="stable")
    return np.take_along_axis(c, o, 1)


def _bundle_files(ds, model):
    """(path, expected_sha256) for every bundle file the remote side will read."""
    import json
    B = "/root/bundle/%s_%s" % (ds, model)
    meta = json.load(open("%s/meta.json" % B))
    want = [("%s/map.npz" % B, meta["map_sha256"]),
            ("%s/%s" % (B, meta["queries"]["file"]), meta["queries"]["sha256"])]
    ext = "npy" if model == "dense" else "npz"
    for st in meta["stores"]:
        if st["i"] and st.get("flat"):
            want.append(("%s/store_%d.%s" % (B, st["i"], ext), st["sha256"]))
    return meta, B, want


def _csr(z, rows):
    import numpy as np
    import scipy.sparse as sp
    a = sp.csr_matrix((z["data"], z["indices"], z["indptr"]),
                      shape=(int(z["shape"][0]), int(z["shape"][1])))
    return a[np.asarray(rows)]


def _load_bundle(ds, model, verify_bundle=True):
    """Resolve the corpus to distinct rows, hashing every byte of provenance on the way."""
    import json
    import os as _os
    import time

    import numpy as np

    meta, B, want = _bundle_files(ds, model)
    if verify_bundle:
        for p, w in want:
            g = _sha(p)
            if g != w:
                raise SystemExit("BUNDLE MISMATCH %s: remote %s != local %s" % (p, g, w))
        print("[%s/%s] bundle verified: %d files" % (ds, model, len(want)), flush=True)

    store0 = json.load(open("%s/store0.json" % B))
    if not store0.get("census_is_complete"):
        raise SystemExit("bundle census is not complete -- refusing")
    m = np.load("%s/map.npz" % B)
    u_src = m["u_src"].astype(np.int64)
    u_row = m["u_row"].astype(np.int64)
    pos2u = m["pos2u"]
    nu, n_pos = u_src.size, pos2u.size
    ss = int(store0["shard_size"])
    cens = store0["local_shard_sha256"]
    print("[%s/%s] positions=%d distinct=%d dup=%d" % (ds, model, n_pos, nu, n_pos - nu),
          flush=True)

    t = time.time()
    sel0 = np.nonzero(u_src == 0)[0]
    rows0 = u_row[sel0]
    shard_of = rows0 // ss
    verified = 0
    X = None
    parts, order = [], []

    for kk in np.unique(shard_of):
        p = _os.path.join(store0["dir"], store0["pattern"] % int(kk))
        if not _os.path.isfile(p):
            raise SystemExit("MISSING remote base shard %s" % p)
        got = _sha(p)
        wnt = cens.get(str(int(kk)))
        if wnt is None:
            raise SystemExit("shard %d is not in the local census" % int(kk))
        if got != wnt:
            raise SystemExit("SHARD MISMATCH %s: remote %s != local %s" % (p, got, wnt))
        verified += 1
        mk = np.nonzero(shard_of == kk)[0]
        off = rows0[mk] - int(kk) * ss
        z = np.load(p)
        if model == "dense":
            if X is None:
                X = np.zeros((nu, z.shape[1]), dtype=np.float16)
            X[sel0[mk]] = z[off]
        else:
            parts.append(_csr(z, off))
            order.append(sel0[mk])
        del z

    ext = "npy" if model == "dense" else "npz"
    for st in meta["stores"]:
        if not st["i"] or not st.get("flat"):
            continue
        sel = np.nonzero(u_src == st["i"])[0]
        if not sel.size:
            continue
        p = "%s/store_%d.%s" % (B, st["i"], ext)
        if model == "dense":
            a = np.load(p, mmap_mode="r")
            X[sel] = a[u_row[sel]]
            del a
        else:
            parts.append(_csr(np.load(p), u_row[sel]))
            order.append(sel)

    if model == "splade":
        import scipy.sparse as sp
        X = sp.vstack(parts, format="csr")
        idx = np.concatenate(order)
        inv = np.empty(nu, dtype=np.int64)
        inv[idx] = np.arange(idx.size)
        X = X[inv]
        del parts, order
        if X.shape[0] != nu:
            raise SystemExit("resolved %d rows != %d distinct" % (X.shape[0], nu))
        print("[%s/%s] resolved %d x %d csr nnz=%d (%.1f GB) shards=%d empty_rows=%d %.0fs"
              % (ds, model, nu, X.shape[1], X.nnz,
                 (X.data.nbytes + X.indices.nbytes) / 1e9, verified,
                 int((np.diff(X.indptr) == 0).sum()), time.time() - t), flush=True)
    else:
        nz = 0
        for a0 in range(0, nu, 1 << 19):           # chunked: np.abs(X) whole would double 18 GB
            blk = X[a0:a0 + (1 << 19)]
            nz += int((np.abs(blk.astype(np.float32)).sum(axis=1) == 0).sum())
        print("[%s/%s] resolved %d x %d fp16 (%.1f GB) zero_rows=%d shards=%d %.0fs"
              % (ds, model, nu, X.shape[1], X.nbytes / 1e9, nz, verified, time.time() - t),
              flush=True)
        if nz:
            raise SystemExit("%d all-zero rows after resolution -- refusing" % nz)

    return meta, X, pos2u, n_pos, B


def _emit(ds, model, best_keys, nq, k, secs, extra, subdir=None, name=None):
    import json
    import os as _os

    import numpy as np

    ids = np.empty((nq, k), dtype=np.int32)
    scf = np.empty((nq, k), dtype=np.float16)
    for a in range(0, nq, 20000):
        s_, i_ = unpack(best_keys[a:a + 20000])
        ids[a:a + 20000] = i_.astype(np.int32)
        scf[a:a + 20000] = s_.astype(np.float16)
    if int(ids.min()) < 0:
        raise SystemExit("negative canonical id in the output -- a padded key survived")
    od = "/root/out/%s%s" % (ds, "/" + subdir if subdir else "")
    _os.makedirs(od, exist_ok=True)
    p = "%s/%s" % (od, name or "%s_top%d.npz" % (model, k))
    np.savez(p, ids=ids, scores=scf)
    r = {"n_queries": int(nq), "K": int(k), "seconds": round(secs, 1),
         "score_min": float(scf.min()), "score_max": float(scf.max()),
         "top1_score_median": float(np.median(scf[:, 0])),
         "bytes": _os.path.getsize(p)}
    r.update(extra)
    json.dump(r, open(p[:-4] + ".meta.json", "w"), indent=1)
    out.commit()
    print("[%s/%s] WROTE %s  %.3f GB  %.0fs  top1_median=%.4f"
          % (ds, model, p, r["bytes"] / 1e9, secs, r["top1_score_median"]), flush=True)
    return r


# ---------------------------------------------------------------- stage: census

@app.function(image=image, volumes=VOLS, cpu=4.0, memory=16384, timeout=7200)
def census_one(ds: str, model: str):
    """Hash every remote shard the job will read against the local census. No scoring."""
    import json
    import os as _os

    meta, B, want = _bundle_files(ds, model)
    rep = {"dataset": ds, "model": model, "bundle_files": []}
    for p, w in want:
        rep["bundle_files"].append({"file": _os.path.basename(p),
                                    "ok": _os.path.isfile(p) and _sha(p) == w})

    store0 = json.load(open("%s/store0.json" % B))
    need = int(store0["max_base_shard_needed"])
    listed = _os.listdir(store0["dir"]) if _os.path.isdir(store0["dir"]) else []
    miss, bad, ok = [], [], 0
    for k in range(need + 1):
        p = _os.path.join(store0["dir"], store0["pattern"] % k)
        if not _os.path.isfile(p):
            miss.append(k)
        elif _sha(p) != store0["local_shard_sha256"][str(k)]:
            bad.append(k)
        else:
            ok += 1
    bundle_ok = all(f["ok"] for f in rep["bundle_files"])
    rep["shards"] = {"needed": need + 1, "present_and_correct": ok, "missing": miss,
                     "wrong_bytes": bad, "listdir_count": len(listed)}
    rep["verdict"] = "CLEAN" if not miss and not bad and bundle_ok else "DEFECTIVE"
    print("[%s/%s] shards needed=%d ok=%d missing=%d wrong=%d listdir=%d bundle_ok=%s  %s"
          % (ds, model, need + 1, ok, len(miss), len(bad), len(listed), bundle_ok,
             rep["verdict"]), flush=True)
    if miss:
        print("   missing: %s%s" % (miss[:40], " ..." if len(miss) > 40 else ""), flush=True)
    if bad:
        print("   wrong bytes: %s%s" % (bad[:40], " ..." if len(bad) > 40 else ""), flush=True)
    return rep


@app.local_entrypoint()
def census():
    import json
    res = list(census_one.starmap(JOBS))
    print("")
    for r in res:
        print("%-9s %-7s %-9s shards ok %d/%d  missing %d  wrong %d"
              % (r["dataset"], r["model"], r["verdict"], r["shards"]["present_and_correct"],
                 r["shards"]["needed"], len(r["shards"]["missing"]),
                 len(r["shards"]["wrong_bytes"])))
    print("ALL CLEAN: %s" % ("YES" if all(r["verdict"] == "CLEAN" for r in res) else "NO"))
    open("scratchpad/_remote_census.json", "w").write(json.dumps(res, indent=1))


# ---------------------------------------------------------------- stage: dense (GPU)

@app.function(image=image, volumes=VOLS, gpu=GPU_SPEC, cpu=8.0, memory=131072,
              timeout=86400)
def dense_run(ds: str, check_blocks: int = 2):
    import time

    import numpy as np
    import torch

    meta, X, pos2u, n_pos, B = _load_bundle(ds, "dense")
    qm = np.load("%s/queries.npy" % B)
    nq = qm.shape[0]
    k = min(K, n_pos)

    torch.backends.cuda.matmul.allow_tf32 = True     # fp32 in/out, exact fp16 operands
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    print("[%s/dense] gpu=%s  queries=%d  k=%d" % (ds, torch.cuda.get_device_name(0), nq, k),
          flush=True)
    t0 = time.time()

    Xg = torch.from_numpy(X).to(dev)                 # fp16 resident, gathered per block
    del X
    Q = torch.from_numpy(np.ascontiguousarray(qm, dtype=np.float32)).to(dev)
    del qm
    p2u = torch.from_numpy(pos2u.astype(np.int64)).to(dev)
    best = torch.empty((nq, k), dtype=torch.int64, device=dev)
    checks = 0

    def mono(s):
        """float32 -> signed int32 preserving float order, without widening."""
        i = s.view(torch.int32)
        return torch.where(i >= 0, i, i ^ 0x7FFFFFFF)

    nblk = (n_pos + DOC_BLOCK_DENSE - 1) // DOC_BLOCK_DENSE
    for bi, b0 in enumerate(range(0, n_pos, DOC_BLOCK_DENSE)):
        b1 = min(b0 + DOC_BLOCK_DENSE, n_pos)
        Bn = b1 - b0
        dm = Xg[p2u[b0:b1]].to(torch.float32).T                       # (dim, Bn) fp32 view
        low = M32 - torch.arange(b0, b1, dtype=torch.int64, device=dev)
        qb = max(64, min(nq, int(6.0e9 // (Bn * 16))))
        for q0 in range(0, nq, qb):
            q1 = min(q0 + qb, nq)
            ms = mono(Q[q0:q1] @ dm)                                  # fp32 TF32 GEMM
            do_check = bi and checks < check_blocks
            bprev = best[q0:q1].clone() if do_check else None

            if bi == 0:
                key = ms.to(torch.int64) * TWO32 + low.unsqueeze(0)
                if key.shape[1] < k:
                    raise SystemExit("first document block yielded %d < K=%d columns"
                                     % (key.shape[1], k))
                best[q0:q1] = torch.topk(key, k, dim=1, sorted=True).values
                del key
            else:
                thr = torch.div(best[q0:q1, k - 1:k], TWO32,
                                rounding_mode="floor").to(torch.int32)
                idx = torch.nonzero(ms >= thr, as_tuple=False)
                if idx.numel():
                    r, c = idx[:, 0], idx[:, 1]
                    cnt = torch.bincount(r, minlength=q1 - q0)
                    mmax = int(cnt.max())
                    if mmax > 4 * k:                                  # pathological: rank all
                        key = ms.to(torch.int64) * TWO32 + low.unsqueeze(0)
                        cand = torch.topk(key, min(k, Bn), dim=1, sorted=True).values
                        del key
                    else:
                        base = torch.cumsum(cnt, 0) - cnt
                        within = torch.arange(r.numel(), device=dev) - base[r]
                        cand = torch.full((q1 - q0, mmax), -(1 << 62),
                                          dtype=torch.int64, device=dev)
                        cand[r, within] = ms[r, c].to(torch.int64) * TWO32 + low[c]
                    best[q0:q1] = torch.topk(torch.cat([best[q0:q1], cand], 1), k,
                                             dim=1, sorted=True).values
                    del cand
                del idx

            if do_check:
                key = ms.to(torch.int64) * TWO32 + low.unsqueeze(0)
                ref = torch.topk(key, min(k, Bn), dim=1, sorted=True).values
                naive = torch.topk(torch.cat([bprev, ref], 1), k, dim=1, sorted=True).values
                if not torch.equal(naive, best[q0:q1]):
                    d = int((naive != best[q0:q1]).sum())
                    raise SystemExit("PREFILTER DISAGREES with naive topk at block %d rows "
                                     "%d..%d: %d of %d entries differ"
                                     % (bi, q0, q1, d, naive.numel()))
                del key, ref, naive, bprev
                checks += 1
                print("  [%s] prefilter == naive topk, block %d rows %d..%d (%d entries)"
                      % (ds, bi, q0, q1, k * (q1 - q0)), flush=True)
            del ms
        del dm, low
        torch.cuda.empty_cache()
        if bi % 4 == 0 or b1 == n_pos:
            print("  [%s] block %d/%d docs %d/%d qb=%d %.0fs"
                  % (ds, bi + 1, nblk, b1, n_pos, qb, time.time() - t0), flush=True)

    keys = best.cpu().numpy().astype(np.uint64) + np.uint64(1 << 63)
    del best
    return _emit(ds, "dense", keys, nq, k, time.time() - t0,
                 {"n_docs": int(n_pos), "n_distinct": int(meta["n_distinct"]),
                  "duplicate_positions": int(meta["n_duplicate_positions"]),
                  "backend": "modal %s, float32 TF32 GEMM" % GPU,
                  "prefilter": "score >= exact k-th-best score; %d blocks checked against "
                               "naive topk, bit-identical" % checks})


# ---------------------------------------------------------------- stage: splade (CPU)

@app.function(image=image, volumes=VOLS, cpu=8.0, memory=65536, timeout=86400)
def splade_run(ds: str, qshard: int = 0, n_qshards: int = 1):
    """The local builder's scipy path, unchanged, over a slice of the QUERIES.

    Query sharding is not an approximation: a row of the score matrix depends on one query
    only, and the merge keeps the k largest keys over the whole corpus, so a shard's rows are
    exactly what a single container would have written for those queries.
    """
    import time

    import numpy as np
    import scipy.sparse as sp

    meta, X, pos2u, n_pos, B = _load_bundle(ds, "splade")
    Qall = sp.load_npz("%s/queries.npz" % B).tocsr().astype(np.float32)
    nq_all = Qall.shape[0]
    lo = nq_all * qshard // n_qshards
    hi = nq_all * (qshard + 1) // n_qshards
    qm = Qall[lo:hi]
    del Qall
    nq = hi - lo
    k = min(K, n_pos)
    t0 = time.time()
    print("[%s/splade] shard %d/%d queries %d..%d (%d)"
          % (ds, qshard, n_qshards, lo, hi, nq), flush=True)

    best = np.empty((nq, k), dtype=np.uint64)
    first = True
    for b0 in range(0, n_pos, DOC_BLOCK_SPLADE):
        b1 = min(b0 + DOC_BLOCK_SPLADE, n_pos)
        Bn = b1 - b0
        pos = np.arange(b0, b1, dtype=np.int64)
        dm = X[pos2u[b0:b1]].tocsr().astype(np.float32).T.tocsc()
        qb = max(64, min(nq, int(KEY_BUDGET // (max(Bn, 1) * 8))))
        for q0 in range(0, nq, qb):
            q1 = min(q0 + qb, nq)
            s = np.asarray((qm[q0:q1] @ dm).todense(), dtype=np.float32)
            kk = order_key(s, np.broadcast_to(pos, (q1 - q0, Bn)))
            top = take_top(kk, min(k, Bn))
            del s, kk
            if first:
                if top.shape[1] != k:
                    raise SystemExit("first document block yielded %d < K=%d columns"
                                     % (top.shape[1], k))
                best[q0:q1] = top
            else:
                best[q0:q1] = take_top(np.concatenate([best[q0:q1], top], axis=1), k)
            del top
        first = False
        del dm
        print("  [%s.%d] docs %d/%d qb=%d %.0fs"
              % (ds, qshard, b1, n_pos, qb, time.time() - t0), flush=True)

    return _emit(ds, "splade", best, nq, k, time.time() - t0,
                 {"n_docs": int(n_pos), "qshard": qshard, "n_qshards": n_qshards,
                  "lo": int(lo), "hi": int(hi),
                  "backend": "modal cpu, scipy path identical to the local builder"},
                 subdir="_splade_shards",
                 name="part_%04d_of_%04d.npz" % (qshard, n_qshards))


@app.function(image=image, volumes=VOLS, cpu=4.0, memory=65536, timeout=7200)
def splade_merge(ds: str, n_qshards: int):
    import glob
    import json
    import os as _os

    import numpy as np

    od = "/root/out/%s" % ds
    parts = sorted(glob.glob("%s/_splade_shards/part_*_of_%04d.npz" % (od, n_qshards)))
    if len(parts) != n_qshards:
        raise SystemExit("have %d of %d parts -- refusing to merge a partial cache"
                         % (len(parts), n_qshards))
    metas = [json.load(open(p[:-4] + ".meta.json")) for p in parts]
    nq = max(m["hi"] for m in metas)
    k = int(np.load(parts[0])["ids"].shape[1])
    ids = np.empty((nq, k), dtype=np.int32)
    scf = np.empty((nq, k), dtype=np.float16)
    seen = np.zeros(nq, dtype=bool)
    for p, m in zip(parts, metas):
        z = np.load(p)
        if z["ids"].shape[0] != m["hi"] - m["lo"]:
            raise SystemExit("%s has %d rows but claims %d"
                             % (p, z["ids"].shape[0], m["hi"] - m["lo"]))
        ids[m["lo"]:m["hi"]] = z["ids"]
        scf[m["lo"]:m["hi"]] = z["scores"]
        seen[m["lo"]:m["hi"]] = True
    if not seen.all():
        raise SystemExit("%d query rows were never written" % int((~seen).sum()))
    o = "%s/splade_top%d.npz" % (od, k)
    np.savez(o, ids=ids, scores=scf)
    json.dump({"n_queries": int(nq), "K": int(k), "merged_from": n_qshards,
               "seconds": round(sum(m["seconds"] for m in metas), 1),
               "wall_seconds_slowest_shard": round(max(m["seconds"] for m in metas), 1),
               "score_min": float(scf.min()), "score_max": float(scf.max()),
               "top1_score_median": float(np.median(scf[:, 0])),
               "n_docs": metas[0]["n_docs"], "bytes": _os.path.getsize(o),
               "backend": "modal cpu x%d, scipy path identical to the local builder"
                          % n_qshards},
              open("%s/splade_top%d.meta.json" % (od, k), "w"), indent=1)
    out.commit()
    print("[%s/splade] MERGED %s  %.3f GB from %d parts  top1_median=%.4f"
          % (ds, o, _os.path.getsize(o) / 1e9, n_qshards, float(np.median(scf[:, 0]))),
          flush=True)
    return {"file": o, "n_queries": int(nq)}


@app.local_entrypoint()
def shards(ds: str = "hotpotqa", qshards: int = 24, only: str = ""):
    """Run a NAMED SUBSET of the splade query shards.

    The first workspace was disabled with 18 of hotpotqa's 24 parts written, so the remaining
    six have to be run without redoing the eighteen.  lo/hi are a pure function of
    (qshard, n_qshards), so a part produced in one workspace and a part produced in another
    cover exactly the query rows their names claim, and merge without overlap or gap.
    """
    idx = [int(x) for x in only.split(",") if x != ""] or list(range(qshards))
    print("running %s splade shards %s of %d" % (ds, idx, qshards))
    res = list(splade_run.starmap([(ds, i, qshards) for i in idx]))
    print("shards done: %d of %d requested" % (len(res), len(idx)))
    for r in res:
        print("   part %04d rows %d..%d  %.0fs" % (r["qshard"], r["lo"], r["hi"],
                                                   r["seconds"]))


@app.local_entrypoint()
def main(ds: str = "hotpotqa", model: str = "dense", qshards: int = 24):
    if model == "dense":
        print(dense_run.remote(ds))
    else:
        res = list(splade_run.starmap([(ds, i, qshards) for i in range(qshards)]))
        print("shards done: %d of %d" % (len(res), qshards))
        print(splade_merge.remote(ds, qshards))
