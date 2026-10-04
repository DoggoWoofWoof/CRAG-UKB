# -*- coding: utf-8 -*-
"""Exact k=3 semantic kNN over the RESOLVED canonical vectors, on GPU.

WHY NOT modal_w0_knn_sharded.py
    That script reads data/canonical/<tree>/encodings/dense/ straight off the volume. For
    2wiki and hotpotqa the canonical vector for a row is NOT what is in that tree: 87,764
    2wiki rows come from the REV2 textualization patch and 63,107 more from the dense-repair
    patch (hotpotqa: 1,688 + 47,691). The Aug-28 checkpoints still on the volume were computed
    before either patch existed, so they are a kNN over ~2.5% wrong 2wiki vectors. This script
    reconstructs exactly what pointer_resolver.CanonicalEmbeddings returns and recomputes.

MATH CONTRACT -- unchanged from the frozen metaqa/squad/musique kNN
    fp16 storage, fp32 normalize, fp32 accumulate, TF32 OFF, exhaustive, no ANN,
    top-SEARCH_K=4 including self, drop self, keep k=3, undirected, weight = cosine.

SPACE
    Similarity runs over DISTINCT source rows, not canonical positions: 30.9% of the webqsp
    nodes share an encoder row with another node (identical text) and in position space those
    ties would take all three neighbour slots at cosine 1.0. Every canonical position then
    inherits the neighbours of its source row, each mapped to the lowest canonical position of
    that neighbour, so deduplicated nodes get edges too.

USAGE
    MODAL_PROFILE=<acct> modal run --detach scratchpad/modal_canonical_knn.py \
        --ds 2wiki --stage verify
    ... --ds 2wiki --stage compute --qshards 0-45
    ... --ds 2wiki --stage merge
"""
import modal

app = modal.App("crag-canonical-knn")
vol = modal.Volume.from_name("crag-data-volume", create_if_missing=True)
enc = modal.Volume.from_name("crag-webqsp-enc", create_if_missing=True)
bundle = modal.Volume.from_name("crag-knn-bundle", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.2.1", "numpy<2.0"))

QSHARD_ROWS = 65536
DOC_BLOCK = 131072
QMICRO = 4096
SEARCH_K = 4
# Headroom left free on the accelerator for the fp32 working set: one DOC_BLOCK of the corpus
# cast to fp32 (0.80 GB), the query-by-block tile (2.15 GB), the topk workspace behind it and
# the CUDA context. DOC_BLOCK and QMICRO are the values the squad parity run reproduced the
# frozen kNN with, so they are not tuning knobs -- the accelerator is sized to them instead.


def headroom(qmicro, dim):
    """Bytes to leave free on the card for the fp32 working set of one doc block.

    Db is DOC_BLOCK x dim fp32; the score tile is qmicro x DOC_BLOCK fp32 and topk wants
    room of the same order behind it; the rest is the CUDA context and fragmentation. The
    2wiki corpus is 17.1 GiB and an A10G reports about 22.2 GiB, so at qmicro 4096 the tile
    alone pushes it over and the corpus would stream once per query shard.
    """
    return (DOC_BLOCK * dim * 4) + (3 * qmicro * DOC_BLOCK * 4) + (3 * (1024 ** 3) // 2)

VOLS = {"/root/vol": vol, "/root/enc": enc, "/root/bundle": bundle}


def _paths(ds):
    return "/root/bundle/%s" % ds, "/root/bundle/%s/_ckpt" % ds


def _work(ds, stage, qshards, qmicro=QMICRO):
    import hashlib
    import json
    import os
    import time

    import numpy as np

    B, CK = _paths(ds)
    meta = json.load(open("%s/meta.json" % B))
    store0 = json.load(open("%s/store0.json" % B))
    dim = meta["dim"]
    m = np.load("%s/map.npz" % B)
    u_src, u_row = m["u_src"].astype(np.int64), m["u_row"].astype(np.int64)
    nu = u_src.size
    n_qshards = (nu + QSHARD_ROWS - 1) // QSHARD_ROWS
    os.makedirs(CK, exist_ok=True)

    def base_shard(k):
        return os.path.join(store0["dir"], store0["pattern"] % k)

    # ---------------- verify: prove the base store on the volume is the local one --------
    if stage == "verify":
        ss = store0["shard_size"]
        want = sorted(set(int(x) for x in store0["verify_shards"]))
        out = {}
        for k in want:
            p = base_shard(k)
            if not os.path.exists(p):
                out[str(k)] = {"MISSING": p}
                continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for c in iter(lambda: f.read(1 << 22), b""):
                    h.update(c)
            a = np.load(p, mmap_mode="r")
            out[str(k)] = {"sha256": h.hexdigest(), "shape": list(a.shape),
                           "dtype": str(a.dtype), "bytes": os.path.getsize(p)}
        need = int(u_row[u_src == 0].max()) // ss
        out["_max_base_shard_needed"] = need
        out["_base_shards_present"] = int(sum(os.path.exists(base_shard(k))
                                              for k in range(need + 1)))
        out["_shard_size"] = ss
        json.dump(out, open("%s/VERIFY_BASE.json" % B, "w"), indent=1)
        bundle.commit()
        print("[%s] VERIFY %s" % (ds, json.dumps(out, indent=1)), flush=True)
        return 0

    # ---------------- resolve: rebuild the canonical distinct-row matrix -----------------
    def resolve():
        t = time.time()
        X = np.zeros((nu, dim), dtype=np.float16)
        ss = store0["shard_size"]
        sel = np.nonzero(u_src == 0)[0]
        rows = u_row[sel]
        sh = rows // ss
        for k in np.unique(sh):
            mk = np.nonzero(sh == k)[0]
            # read the shard whole. Nearly every row of every shard is wanted, and mmap over a
            # network-backed volume turns that into thousands of scattered page faults per file
            # instead of one sequential read.
            a = np.load(base_shard(int(k)))
            X[sel[mk]] = a[rows[mk] - int(k) * ss]
            del a
        for st in meta["stores"]:
            if st["i"] == 0 or not st.get("flat"):
                continue
            sel = np.nonzero(u_src == st["i"])[0]
            if not sel.size:
                continue
            a = np.load("%s/store_%d.npy" % (B, st["i"]), mmap_mode="r")
            X[sel] = a[u_row[sel]]
        nz = int((np.abs(X).sum(axis=1) == 0).sum())
        print("[%s] resolved %d x %d fp16 (%.1f GB) zero-rows=%d %.0fs"
              % (ds, nu, dim, X.nbytes / 1e9, nz, time.time() - t), flush=True)
        if nz:
            raise SystemExit("[%s] %d all-zero rows after resolution -- refusing" % (ds, nz))
        return X

    if stage == "resolve_fingerprint":
        X = resolve()
        # the sample rows are chosen LOCALLY and shipped, so the comparison never depends on
        # two numpy versions drawing the same stream
        pick = np.load("%s/fp_rows.npy" % B).astype(np.int64)
        fp = hashlib.sha256(np.ascontiguousarray(X[pick]).tobytes()).hexdigest()
        r = {"n_distinct": int(nu), "dim": int(dim), "n_sample": int(pick.size),
             "sample_sha256": fp, "first_rows": pick.tolist()[:8]}
        json.dump(r, open("%s/RESOLVE_FINGERPRINT.json" % B, "w"), indent=1)
        bundle.commit()
        print("[%s] FINGERPRINT n=%d sample=%d sha=%s" % (ds, nu, pick.size, fp), flush=True)
        return 0

    # ---------------- merge: lift distinct-row neighbours to canonical positions ---------
    if stage == "merge":
        pf = "%s/COMPUTE_PARAMS.json" % CK
        if os.path.exists(pf):
            cp = json.load(open(pf))
            qmicro, qm_src = int(cp["qmicro"]), "read from the compute stage"
        else:
            qm_src = ("passed to this merge; the compute run predates COMPUTE_PARAMS.json "
                      "and its logs record the value it ran with")
        done = [s for s in range(n_qshards) if os.path.exists("%s/q%05d.done" % (CK, s))]
        if len(done) != n_qshards:
            miss = sorted(set(range(n_qshards)) - set(done))
            print("[%s] MERGE ABORT %d/%d shards missing: %s"
                  % (ds, len(miss), n_qshards, miss[:10]), flush=True)
            return 1
        nbr = np.full((nu, SEARCH_K), -1, dtype=np.int64)
        sc = np.zeros((nu, SEARCH_K), dtype=np.float32)
        for s in range(n_qshards):
            d = np.load("%s/q%05d.npz" % (CK, s))
            r0 = int(d["row_start"])
            nbr[r0:r0 + d["nbr"].shape[0]] = d["nbr"]
            sc[r0:r0 + d["nbr"].shape[0]] = d["score"]
        pos2u = m["pos2u"].astype(np.int64)
        first = m["first_pos"].astype(np.int64)
        npos = pos2u.size
        pos = np.arange(npos, dtype=np.int64)
        src_l, dst_l, w_l = [], [], []
        for c in range(SEARCH_K):
            v = nbr[pos2u, c]
            w = sc[pos2u, c]
            keep = (v >= 0) & (v != pos2u)
            j = first[v[keep]]
            i = pos[keep]
            ok = i != j
            src_l.append(i[ok])
            dst_l.append(j[ok])
            w_l.append(w[keep][ok])
        i = np.concatenate(src_l)
        j = np.concatenate(dst_l)
        w = np.concatenate(w_l)
        a = np.minimum(i, j)
        b = np.maximum(i, j)
        key = (a << 32) | b
        order = np.lexsort((-w, key))            # highest cosine wins a duplicated pair
        key, a, b, w = key[order], a[order], b[order], w[order]
        keep = np.ones(key.size, dtype=bool)
        keep[1:] = key[1:] != key[:-1]
        a, b, w = a[keep], b[keep], w[keep]
        outp = "%s/knn_canonical_edges.npz" % B
        np.savez_compressed(outp, src=a.astype(np.int32), dst=b.astype(np.int32),
                            weight=w.astype(np.float32))
        h = hashlib.sha256(open(outp, "rb").read()).hexdigest()
        man = {"dataset": ds, "edge_family": "knn", "edge_subtype": "semantic_knn_k3",
               "method": "EXHAUSTIVE_SHARDED_FP32_COMPUTE over RESOLVED canonical vectors",
               "flags": {"FP16_STORAGE": "YES", "FP16_REDUCED_COMPUTE": "NO",
                         "FP32_COMPUTE": "YES", "TF32": "OFF", "ANN": "NO",
                         "EXHAUSTIVE": "YES"},
               "exact": True, "ann": False, "search_k": SEARCH_K, "n_neighbors": 3,
               "n_canonical_positions": int(npos), "n_distinct_source_rows": int(nu),
               "n_duplicate_positions": int(meta["n_duplicate_positions"]),
               "n_edges": int(a.size), "directed": False,
               "endpoint_space": "canonical positions (nodes.jsonl line numbers)",
               "similarity_space": "distinct source rows; each position inherits the "
                                   "neighbours of its source row, mapped to the lowest "
                                   "canonical position of each neighbour",
               "vector_source": "pointer-resolved (PHASE_C + REV2_PATCH + DENSE_REPAIR)",
               "n_qshards": n_qshards, "qshard_rows": QSHARD_ROWS,
               "doc_block": DOC_BLOCK, "qmicro": qmicro,
               "qmicro_provenance": qm_src,
               "qmicro_note": "batching only -- topk over dim=1 is per row; asserted "
                              "IDENTICAL against qmicro=%d by the qmicro_selftest stage"
                              % QMICRO,
               "edges_npz_sha256": h}
        json.dump(man, open("%s/knn_manifest.json" % B, "w"), indent=2)
        bundle.commit()
        print("[%s] MERGE DONE edges=%d sha=%s" % (ds, a.size, h[:16]), flush=True)
        return 0

    # ---------------- compute ------------------------------------------------------------
    import torch
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    t0 = time.time()

    def parse(spec):
        if spec in (None, "all"):
            return list(range(n_qshards))
        out = []
        for p in str(spec).split(","):
            if "-" in p:
                x, y = p.split("-")
                out += list(range(int(x), int(y) + 1))
            else:
                out.append(int(p))
        return [s for s in out if 0 <= s < n_qshards]

    sel = parse(qshards)
    todo = [s for s in sel if not os.path.exists("%s/q%05d.done" % (CK, s))]
    if stage != "qmicro_selftest" and not todo:
        print("[%s] nothing to do (%d shards already done)" % (ds, n_qshards), flush=True)
        return 0
    Xc = torch.from_numpy(resolve())
    # 2wiki is 18.4 GB of fp16 corpus and hotpotqa 16.0 GB. Streaming that across PCIe once
    # per query shard would move 1.7 TB over the run; resident it moves once. Decide from the
    # card actually assigned rather than a compiled-in ceiling.
    vram = torch.cuda.get_device_properties(0).total_memory
    on_gpu = (nu * dim * 2) < (vram - headroom(qmicro, dim))
    Xg = Xc.to(dev) if on_gpu else None
    nrm = torch.nn.functional.normalize
    print("[%s] nu=%d qshards=%d todo=%d corpus=%.1fGB vram=%.1fGB corpus_on_gpu=%s %.0fs"
          % (ds, nu, n_qshards, len(todo), nu * dim * 2 / 2**30, vram / 2**30,
             on_gpu, time.time() - t0), flush=True)

    def search_shard(sh, qm_size):
        r0 = sh * QSHARD_ROWS
        r1 = min(r0 + QSHARD_ROWS, nu)
        q = r1 - r0
        best_sc = torch.full((q, SEARCH_K), -1e30, device=dev)
        best_id = torch.full((q, SEARCH_K), -1, dtype=torch.long, device=dev)
        for b in range(0, nu, DOC_BLOCK):
            e = min(b + DOC_BLOCK, nu)
            Db = nrm((Xg[b:e] if on_gpu else Xc[b:e].to(dev)).float(), dim=1)
            k = min(SEARCH_K, e - b)
            for qm in range(0, q, qm_size):
                qe = min(qm + qm_size, q)
                Qs = nrm((Xg[r0 + qm:r0 + qe] if on_gpu
                          else Xc[r0 + qm:r0 + qe].to(dev)).float(), dim=1)
                tile = Qs @ Db.T
                v, jj = torch.topk(tile, k, dim=1)
                cs = torch.cat([best_sc[qm:qe], v], 1)
                ci = torch.cat([best_id[qm:qe], jj + b], 1)
                nsc, o = torch.topk(cs, SEARCH_K, dim=1)
                best_sc[qm:qe] = nsc
                best_id[qm:qe] = torch.gather(ci, 1, o)
                del Qs, tile
            del Db
        return best_id.cpu().numpy(), best_sc.cpu().numpy(), r0, r1

    if stage == "qmicro_selftest":
        s0 = sel[0]                      # a finished shard is the point: recompute and compare
        outs = {}
        for qm_try in (QMICRO, qmicro):
            outs[qm_try] = search_shard(s0, qm_try)[0]
            print("  [%s] selftest qshard %d qmicro=%d chk=%s %.0fs"
                  % (ds, s0, qm_try, hashlib.sha256(outs[qm_try].tobytes()).hexdigest()[:16],
                     time.time() - t0), flush=True)
        same = all(np.array_equal(v, outs[QMICRO]) for v in outs.values())
        print("[%s] QMICRO_SELFTEST %s" % (ds, "IDENTICAL" if same else "DIVERGENT"),
              flush=True)
        return 0 if same else 1

    json.dump({"qmicro": qmicro, "doc_block": DOC_BLOCK, "qshard_rows": QSHARD_ROWS,
               "search_k": SEARCH_K}, open("%s/COMPUTE_PARAMS.json" % CK, "w"))
    for s in todo:
        nb, sco, r0, r1 = search_shard(s, qmicro)
        tmp = "%s/q%05d.tmp.npz" % (CK, s)
        np.savez(tmp, nbr=nb, score=sco, row_start=r0, row_end=r1)
        os.replace(tmp, "%s/q%05d.npz" % (CK, s))
        chk = hashlib.sha256(nb.tobytes()).hexdigest()[:16]
        open("%s/q%05d.done" % (CK, s), "w").write(chk)
        bundle.commit()
        print("  [%s] qshard %d/%d rows[%d:%d) chk=%s %.0fs"
              % (ds, s, n_qshards, r0, r1, chk, time.time() - t0), flush=True)
    print("[%s] COMPUTE done %.0fs" % (ds, time.time() - t0), flush=True)
    return 0


# verify / resolve_fingerprint / merge never touch the GPU, and merge for 2wiki wants more RAM
# than it wants cores. Booking an A10G for them was burning the accelerator on file hashing.
@app.function(image=image, volumes=VOLS, cpu=8.0, memory=131072, timeout=86400)
def cpu_run(ds: str, stage: str, qshards: str = "all", qmicro: int = QMICRO):
    return _work(ds, stage, qshards, qmicro)


@app.function(image=image, volumes=VOLS, gpu="A10G", cpu=8.0, memory=131072, timeout=86400)
def gpu_run(ds: str, stage: str = "compute", qshards: str = "all", qmicro: int = QMICRO):
    return _work(ds, stage, qshards, qmicro)



@app.local_entrypoint()
def main(ds: str = "webqsp", stage: str = "compute", qshards: str = "all",
         qmicro: int = QMICRO):
    (gpu_run if stage in ("compute", "qmicro_selftest") else cpu_run).remote(
        ds, stage, qshards, qmicro)
