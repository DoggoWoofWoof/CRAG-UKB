"""W0 Task 9 (Qwen-kNN family) — exact global k=3 semantic kNN at WORLD scale.

Faithful reproduction of the FROZEN topology-C kNN (src/core/indexers.py:109-161):
  vectors = gte-Qwen2 doc embeddings (canonical order) -> faiss.normalize_L2 (cosine)
  -> faiss.IndexFlatIP (EXACT, no ANN) -> search(vectors, 4) -> take neighbors [1:4]
  (top-3 non-self), undirected, unweighted adjacency. Endpoints = canonical_doc_id.

This is the exact all-pairs kNN the frozen builder GPU-accelerates; faiss-cpu
IndexFlatIP returns the SAME exact neighbors. For the tractable corpora it runs
locally in seconds–minutes with NO Modal cost. The cosine sim is recorded as a 3rd
column for provenance; C uses these edges as unweighted (matching frozen graph.pt,
which carries no edge_weight).

Writes: data/canonical/<ds>/graph_knn.tsv  (src_id\tdst_id\tcosine)
        data/canonical/<ds>/knn_manifest.json
Usage:  python scratchpad/_w0_world_knn.py <ds> [<ds> ...]
"""
import os, sys, json, time, hashlib
import numpy as np

K_NBR = 3          # frozen: top-4 minus self -> 3 neighbors
SEARCH_K = 4

def load_dense(ds):
    base = f"data/canonical/{ds}/encodings/dense/docs"
    mf = json.load(open(f"{base}/manifest.json", encoding="utf-8"))
    ns = mf["n_shards"]
    embs, ids = [], []
    for s in range(ns):
        embs.append(np.load(f"{base}/shard_{s:05d}.npy"))
        ids += json.load(open(f"{base}/ids_{s:05d}.json", encoding="utf-8"))
    X = np.concatenate(embs, 0).astype("float32")   # fp16 storage -> fp32 compute (matches dense parity lock)
    assert X.shape[0] == mf["n_items"] == len(ids), (X.shape, mf["n_items"], len(ids))
    return X, ids, mf

def build(ds):
    import faiss
    t0 = time.time()
    X, ids, mf = load_dense(ds)
    n, dim = X.shape
    print(f"[{ds}] n={n} dim={dim} — exact IndexFlatIP kNN (k={K_NBR})", flush=True)
    faiss.normalize_L2(X)                            # cosine, exactly as indexers.py
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    D, I = index.search(X, SEARCH_K)                 # exact top-4 (self + 3)
    seen = set()
    outp = f"data/canonical/{ds}/graph_knn.tsv"; tmp = outp + ".tmp"
    n_edges = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for i in range(n):
            for r in range(1, SEARCH_K):             # skip self at col 0
                j = int(I[i, r])
                if j < 0 or j == i:
                    continue
                a, b = (i, j) if i < j else (j, i)   # canonicalize orientation (undirected)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                f.write(f"{ids[a]}\t{ids[b]}\t{float(D[i, r]):.6f}\n")
                n_edges += 1
    os.replace(tmp, outp)
    h = hashlib.sha256()
    with open(outp, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    man = dict(dataset=ds, edge_family="qwen_knn", edge_subtype="semantic_knn_k3",
               algorithm="src/core/indexers.py build_pyg_graph global kNN (exact faiss.IndexFlatIP, k=3) — faithful reproduction",
               encoder=mf.get("encoder"), source_sha256=mf.get("source_sha256"),
               normalize="faiss.normalize_L2 (cosine)", exact=True, ann=False,
               search_k=SEARCH_K, n_neighbors=K_NBR, n_nodes=n, n_edges=n_edges,
               edge_schema="src_id\\tdst_id\\tcosine", weight_usage="unweighted in C (cosine recorded for provenance)",
               directed=False, graph_knn_tsv_sha256=h.hexdigest(), secs=round(time.time() - t0, 1))
    json.dump(man, open(f"data/canonical/{ds}/knn_manifest.json", "w"), indent=2)
    print(f"[{ds}] DONE n_edges={n_edges} secs={time.time()-t0:.1f} sha={h.hexdigest()[:16]} -> {outp}", flush=True)

if __name__ == "__main__":
    for ds in sys.argv[1:]:
        build(ds)
