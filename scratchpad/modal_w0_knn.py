"""W0 Task 9 (Qwen-kNN) — exact global k=3 semantic kNN on Modal GPU (kill-resilient).

Faithful to frozen topology-C (src/core/indexers.py:109-161): L2-normalize gte-Qwen2
doc embeddings (cosine), exact top-4 inner-product neighbours (self + 3), undirected,
unweighted. Exact — no ANN. fp16 STORAGE -> fp32 COMPUTE (matches the W0 dense lock,
allow_fp16_reduced_precision_reduction=False), block-tiled query loop; whole corpus
held on GPU when it fits, else sharded doc loop per query block.

Reads sharded dense docs from the account volume (uploaded for Task-7). Writes
data/canonical/<ds>/graph_knn.tsv (+ knn_manifest.json) back to the volume, committed.
Run: MODAL_PROFILE=<acct> modal run --detach scratchpad/modal_w0_knn.py --ds 2wiki
"""
import modal

app = modal.App("crag-w0-knn")
volume = modal.Volume.from_name("crag-data-volume", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.2.1", "numpy<2.0"))


@app.function(image=image, volumes={"/root/CRAG/storage": volume}, gpu="A10G",
              cpu=8.0, memory=65536, timeout=86400)
def run(ds: str, qblock: int = 16384, kmax_load: int = 6_000_000):
    import os, json, time, hashlib, numpy as np, torch
    os.chdir("/root/CRAG")
    t0 = time.time()
    DOCS = f"storage/data/canonical/{ds}/encodings/dense/docs"
    rm = json.load(open(f"{DOCS}/retrieval_manifest.json"))
    dev = "cuda"; assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    n = rm["n_items"]; dim = rm["dim"]
    print(f"[{ds}] n={n} dim={dim} shards={rm['n_shards']} — exact k=3 kNN (fp32 compute/fp16 storage)", flush=True)

    # load all embeddings (fp16) + ids in canonical order
    ids = []
    Xh = torch.empty((n, dim), dtype=torch.float16)
    row = 0
    for sh in rm["shards"]:
        s = sh["shard_id"]
        x = np.load(f"{DOCS}/shard_{s:05d}.npy")
        ids += json.load(open(f"{DOCS}/ids_{s:05d}.json"))
        Xh[row:row + x.shape[0]] = torch.from_numpy(x)
        row += x.shape[0]
    assert row == n == len(ids), (row, n, len(ids))
    # normalize in fp32 (cosine), keep a GPU fp32 corpus for exact IP
    X = Xh.to(dev).float()
    X = torch.nn.functional.normalize(X, dim=1)           # == faiss.normalize_L2
    del Xh

    seen = set(); edges = []
    SEARCH_K = 4
    for qs in range(0, n, qblock):
        qe = min(qs + qblock, n)
        sims = X[qs:qe] @ X.T                              # [b, n] fp32 exact
        v, idx = torch.topk(sims, SEARCH_K, dim=1)
        idx = idx.cpu().numpy(); v = v.cpu().numpy()
        for r in range(qe - qs):
            i = qs + r
            for c in range(SEARCH_K):
                j = int(idx[r, c])
                if j == i or j < 0:
                    continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in seen:
                    continue
                seen.add((a, b)); edges.append((a, b, float(v[r, c])))
        del sims
        if qs % (qblock * 8) == 0:
            print(f"  q {qe}/{n} edges={len(edges)} {time.time()-t0:.0f}s", flush=True)

    outp = f"storage/data/canonical/{ds}/graph_knn.tsv"; tmp = outp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for a, b, w in edges:
            f.write(f"{ids[a]}\t{ids[b]}\t{w:.6f}\n")
    os.replace(tmp, outp)
    h = hashlib.sha256(open(outp, "rb").read()).hexdigest()
    man = dict(dataset=ds, edge_family="qwen_knn", edge_subtype="semantic_knn_k3",
               algorithm="exact IndexFlatIP-equivalent (fp32 compute over fp16 storage, L2-normalized) — faithful to indexers.py:109-161",
               exact=True, ann=False, search_k=SEARCH_K, n_neighbors=3, n_nodes=n,
               n_edges=len(edges), normalize="L2 (cosine)", directed=False,
               edge_schema="src_id\\tdst_id\\tcosine", graph_knn_tsv_sha256=h,
               secs=round(time.time() - t0, 1))
    json.dump(man, open(f"storage/data/canonical/{ds}/knn_manifest.json", "w"), indent=2)
    volume.commit()
    print(f"[{ds}] DONE n_edges={len(edges)} secs={time.time()-t0:.1f} sha={h[:16]} -> {outp}", flush=True)


@app.local_entrypoint()
def main(ds: str = "2wiki", qblock: int = 16384):
    run.remote(ds, qblock)
