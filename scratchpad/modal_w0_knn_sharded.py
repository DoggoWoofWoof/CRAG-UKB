"""W0 Task 9 — SHARDED exact Qwen k=3 kNN for >1M corpora (WebQSP/Hotpot/universe).

Canonical method WORLD_EXACT_IP_METHOD = EXHAUSTIVE_SHARDED_FP32_COMPUTE:
  FP16_STORAGE=YES, FP16_REDUCED_COMPUTE=NO, FP32_COMPUTE=YES, ANN=NO, EXHAUSTIVE=YES.

Frozen-C kNN semantics (src/core/indexers.py:109-161): L2-normalized Qwen doc
vectors, GLOBAL exact search, top-4 including self, drop self, keep exact k=3
neighbours, undirected unweighted semantic edges. No IVF / HNSW / ANN.

Two-dimensional blocking (never materializes N×N):
  corpus embeddings held fp16 on GPU when they fit (n*dim*2 < CORPUS_GPU_MAX), else
  fp16 on CPU streamed per doc-block. For each QUERY SHARD (row range):
    running top-4 (scores+global ids); for each DOC BLOCK: fp32 score tile
    Qs_fp32 @ Db_fp32.T -> local top-4 -> merge into running top-4.
  Checkpoint per query shard -> resume (a Modal interruption resumes from the last
  completed query shard, never restarts the corpus). Final pass merges all shard
  checkpoints -> graph_knn.tsv (drop self, keep 3) + knn_manifest.json.

Run one account over a query-shard range (independent shards are distributable):
  MODAL_PROFILE=<acct> modal run --detach scratchpad/modal_w0_knn_sharded.py \
      --ds webqsp --qshards 0-19            # this account does shards 0..19
  ... then --stage merge  (after ALL shards done, on any one account) writes the tsv.
"""
import modal

app = modal.App("crag-w0-knn-sharded")
volume = modal.Volume.from_name("crag-data-volume", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.2.1", "numpy<2.0"))

QSHARD_ROWS = 65536          # query rows per checkpointed shard
DOC_BLOCK = 131072           # doc rows per score tile
CORPUS_GPU_MAX = 14 * (1024**3)   # hold fp16 corpus on GPU below this; else stream doc blocks
                                  # from CPU (doc-outer loop => each block loaded once/shard, cheap).
                                  # 5-6M corpora (16-18GB fp16) stream to keep A10G headroom for tiles.
SEARCH_K = 4


def _ckpt_dir(ds): return f"storage/data/canonical/{ds}/_knn_ckpt"


@app.function(image=image, volumes={"/root/CRAG/storage": volume}, gpu="A10G",
              cpu=8.0, memory=131072, timeout=86400)
def run(ds: str, qshards: str = "all", stage: str = "compute",
        qshard_rows: int = QSHARD_ROWS, doc_block: int = DOC_BLOCK):
    import os, json, time, hashlib, numpy as np, torch
    os.chdir("/root/CRAG")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # FP16_REDUCED_COMPUTE=NO
    torch.backends.cuda.matmul.allow_tf32 = False   # true fp32 accumulate (FP32_COMPUTE=YES), NOT TF32
    torch.backends.cudnn.allow_tf32 = False
    DOCS = f"storage/data/canonical/{ds}/encodings/dense/docs"
    rm = json.load(open(f"{DOCS}/retrieval_manifest.json"))
    n = rm["n_items"]; dim = rm["dim"]
    ck = _ckpt_dir(ds); os.makedirs(ck, exist_ok=True)
    n_qshards = (n + qshard_rows - 1) // qshard_rows

    def load_ids():
        ids = []
        for sh in rm["shards"]:
            ids += json.load(open(f"{DOCS}/ids_{sh['shard_id']:05d}.json"))
        assert len(ids) == n
        return ids

    # ---------- MERGE stage: assemble edges from completed shard checkpoints ----------
    if stage == "merge":
        done = [s for s in range(n_qshards) if os.path.exists(f"{ck}/qshard_{s:05d}.done")]
        if len(done) != n_qshards:
            miss = [s for s in range(n_qshards) if s not in set(done)]
            print(f"[{ds}] MERGE ABORT: {len(miss)}/{n_qshards} query shards incomplete: {miss[:8]}...", flush=True)
            return 1
        ids = load_ids()
        seen = set(); edges = []
        for s in range(n_qshards):
            d = np.load(f"{ck}/qshard_{s:05d}.npz")
            nbr = d["nbr"]; sc = d["score"]; r0 = int(d["row_start"])
            for r in range(nbr.shape[0]):
                i = r0 + r
                for c in range(SEARCH_K):
                    j = int(nbr[r, c])
                    if j == i or j < 0:
                        continue
                    a, b = (i, j) if i < j else (j, i)
                    if (a, b) in seen:
                        continue
                    seen.add((a, b)); edges.append((a, b, float(sc[r, c])))
        outp = f"storage/data/canonical/{ds}/graph_knn.tsv"; tmp = outp + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for a, b, w in edges:
                f.write(f"{ids[a]}\t{ids[b]}\t{w:.6f}\n")
        os.replace(tmp, outp)
        h = hashlib.sha256(open(outp, "rb").read()).hexdigest()
        man = dict(dataset=ds, edge_family="qwen_knn", edge_subtype="semantic_knn_k3",
                   method="EXHAUSTIVE_SHARDED_FP32_COMPUTE",
                   flags=dict(FP16_STORAGE="YES", FP16_REDUCED_COMPUTE="NO", FP32_COMPUTE="YES", ANN="NO", EXHAUSTIVE="YES"),
                   algorithm="sharded query×doc exact IP (fp32 compute/fp16 storage, L2-normalized) — faithful to indexers.py:109-161",
                   exact=True, ann=False, search_k=SEARCH_K, n_neighbors=3, n_nodes=n,
                   n_edges=len(edges), n_qshards=n_qshards, qshard_rows=qshard_rows,
                   directed=False, edge_schema="src_id\\tdst_id\\tcosine", graph_knn_tsv_sha256=h)
        json.dump(man, open(f"storage/data/canonical/{ds}/knn_manifest.json", "w"), indent=2)
        volume.commit()
        print(f"[{ds}] MERGE DONE n_edges={len(edges)} sha={h[:16]} -> {outp}", flush=True)
        return 0

    # ---------- PARITY stage: sharded merge vs single-shot reference ----------
    if stage == "parity":
        assert torch.cuda.is_available()
        dev = "cuda"; TIE = 1e-3
        nrm = torch.nn.functional.normalize
        Xcpu = torch.empty((n, dim), dtype=torch.float16); row = 0
        for sh in rm["shards"]:
            x = np.load(f"{DOCS}/shard_{sh['shard_id']:05d}.npy")
            Xcpu[row:row + x.shape[0]] = torch.from_numpy(x); row += x.shape[0]
        Xg = Xcpu.to(dev)                                  # raw fp16 storage; normalize in fp32 below
        Xn = nrm(Xg.float(), dim=1)                        # full fp32-normalized (small corpus only)
        rng = np.random.default_rng(0)
        qsel = np.sort(rng.choice(n, size=min(512, n), replace=False))
        Qs = Xn[qsel]
        # reference: single-shot full fp32 top-4
        ref = Qs @ Xn.T
        rv, ri = torch.topk(ref, SEARCH_K, dim=1)
        # sharded: running top-4 over small doc blocks (forces the merge path)
        small = max(1024, n // 17)
        bsc = torch.full((Qs.shape[0], SEARCH_K), -1e30, device=dev)
        bid = torch.full((Qs.shape[0], SEARCH_K), -1, dtype=torch.long, device=dev)
        nblk = 0
        for b in range(0, n, small):
            e = min(b + small, n); nblk += 1
            tile = Qs @ Xn[b:e].T
            k = min(SEARCH_K, e - b)
            v, j = torch.topk(tile, k, dim=1)
            cs = torch.cat([bsc, v], 1); ci = torch.cat([bid, (j + b)], 1)
            bsc, o = torch.topk(cs, SEARCH_K, dim=1); bid = torch.gather(ci, 1, o)
        rv, ri, bsc, bid = rv.cpu().numpy(), ri.cpu().numpy(), bsc.cpu().numpy(), bid.cpu().numpy()
        max_score_diff = float(np.abs(np.sort(rv, 1) - np.sort(bsc, 1)).max())
        top1 = float((ri[:, 0] == bid[:, 0]).mean())
        set_ok = nontie = 0
        for r in range(ri.shape[0]):
            if set(ri[r, :3].tolist()) == set(bid[r, :3].tolist()):
                set_ok += 1
            else:  # count only real (non-tie) disagreements
                a = {int(i): float(s) for i, s in zip(ri[r], rv[r])}
                bmap = {int(i): float(s) for i, s in zip(bid[r], bsc[r])}
                for i in set(a) ^ set(bmap):
                    sc = a.get(i, bmap.get(i))
                    peer = max(list(bmap.values()) if i in a else list(a.values()))
                    if abs(sc - peer) > TIE:
                        nontie += 1
        top3_set = set_ok / ri.shape[0]
        rep = dict(ds=ds, n=n, sample=int(ri.shape[0]), doc_blocks=nblk, block_size=small,
                   max_score_diff=max_score_diff, top1_agree=top1, top3_set_agree=top3_set,
                   real_nontie_disagreements=int(nontie), TIE_EPS=TIE,
                   PARITY_PASS=bool(max_score_diff < TIE and nontie == 0 and top1 == 1.0))
        os.makedirs("storage/data/canonical/_w0_parity", exist_ok=True)
        json.dump(rep, open(f"storage/data/canonical/_w0_parity/sharded_knn_{ds}.json", "w"), indent=2)
        volume.commit()
        print(f"[PARITY {ds}] blocks={nblk} top1={top1} top3set={top3_set} "
              f"maxΔ={max_score_diff:.2e} nontie={nontie} PASS={rep['PARITY_PASS']}", flush=True)
        return 0

    # ---------- COMPUTE stage: process a set of query shards ----------
    assert torch.cuda.is_available()
    dev = "cuda"; t0 = time.time()
    # load corpus fp16 (CPU), place on GPU if it fits
    Xcpu = torch.empty((n, dim), dtype=torch.float16)
    row = 0
    for sh in rm["shards"]:
        x = np.load(f"{DOCS}/shard_{sh['shard_id']:05d}.npy")
        Xcpu[row:row + x.shape[0]] = torch.from_numpy(x); row += x.shape[0]
    assert row == n
    # Storage is RAW fp16 (FP16_STORAGE=YES). Normalization happens in fp32 at compute
    # time (never re-rounded to fp16) — bit-faithful to faiss.normalize_L2 over the
    # fp32-upcast fp16 vectors, i.e. to indexers.py / the tractable-4 reference graphs.
    on_gpu = (n * dim * 2) < CORPUS_GPU_MAX
    Xg = Xcpu.to(dev) if on_gpu else None
    nrm = torch.nn.functional.normalize
    print(f"[{ds}] n={n} dim={dim} qshards={n_qshards} corpus_on_gpu={on_gpu} "
          f"(fp16 {n*dim*2/1e9:.1f}GB) {time.time()-t0:.0f}s", flush=True)

    def parse(spec):
        if spec in (None, "all"):
            return list(range(n_qshards))
        out = []
        for p in str(spec).split(","):
            if "-" in p:
                a, b = p.split("-"); out += list(range(int(a), int(b) + 1))
            else:
                out.append(int(p))
        return [s for s in out if 0 <= s < n_qshards]

    QMICRO = 4096   # query rows per matmul — bounds tile mem (QMICRO×doc_block×4B)
    for s in parse(qshards):
        donef = f"{ck}/qshard_{s:05d}.done"
        if os.path.exists(donef):
            print(f"  qshard {s} skip (done)", flush=True); continue
        r0 = s * qshard_rows; r1 = min(r0 + qshard_rows, n)
        q = r1 - r0
        best_sc = torch.full((q, SEARCH_K), -1e30, device=dev)
        best_id = torch.full((q, SEARCH_K), -1, dtype=torch.long, device=dev)
        # doc-outer / query-inner: each doc block is fp32-normalized exactly once per shard
        for b in range(0, n, doc_block):
            e = min(b + doc_block, n)
            Db = nrm((Xg[b:e] if on_gpu else Xcpu[b:e].to(dev)).float(), dim=1)  # fp32 normalize+compute
            k = min(SEARCH_K, e - b)
            for qm in range(0, q, QMICRO):                        # micro-batch queries (bounds tile mem)
                qe = min(qm + QMICRO, q)
                Qs = nrm((Xg[r0 + qm:r0 + qe] if on_gpu else Xcpu[r0 + qm:r0 + qe].to(dev)).float(), dim=1)
                tile = Qs @ Db.T                                  # [qm, blk] fp32 exact
                v, j = torch.topk(tile, k, dim=1)
                cs = torch.cat([best_sc[qm:qe], v], 1); ci = torch.cat([best_id[qm:qe], j + b], 1)
                nsc, o = torch.topk(cs, SEARCH_K, dim=1)
                best_sc[qm:qe] = nsc; best_id[qm:qe] = torch.gather(ci, 1, o)
                del Qs, tile
            del Db
        nbr = best_id.cpu().numpy(); sc = best_sc.cpu().numpy()
        tmp = f"{ck}/qshard_{s:05d}.tmp.npz"   # must end in .npz (savez appends it otherwise)
        np.savez(tmp, nbr=nbr, score=sc, row_start=r0, row_end=r1)
        os.replace(tmp, f"{ck}/qshard_{s:05d}.npz")
        chk = hashlib.sha256(nbr.tobytes()).hexdigest()[:16]
        open(donef, "w").write(json.dumps(dict(row_start=r0, row_end=r1, rows=q, nbr_sha16=chk)))
        volume.commit()
        print(f"  qshard {s}/{n_qshards} rows[{r0}:{r1}) chk={chk} {time.time()-t0:.0f}s", flush=True)
    print(f"[{ds}] COMPUTE shards done {time.time()-t0:.0f}s", flush=True)
    return 0


@app.local_entrypoint()
def main(ds: str = "webqsp", qshards: str = "all", stage: str = "compute",
         qshard_rows: int = QSHARD_ROWS, doc_block: int = DOC_BLOCK):
    run.remote(ds, qshards, stage, qshard_rows, doc_block)
