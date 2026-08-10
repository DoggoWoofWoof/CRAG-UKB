"""
Probe: can graph relations be DERIVED as semantic offsets? (king - man + woman = queen)
========================================================================================
NOT about replacing the graph — about teaching the representation to bridge
semantically-distant-but-related nodes, so the L1 MLP stops flailing on KB (where
query and answer are far in cosine and PPR has to rescue it).

Idea: relational edges give pairs (A --rel--> B). The offset delta = v_B - v_A is
the "relation direction" (like the gender offset king->queen). We have no relation
labels, so DISCOVER relation types by clustering the deltas (faiss K-means); each
centroid r_k is one learned relation direction. Then test whether adding r_k to a
node's vector recovers its relational neighbours that plain kNN misses.

Reports per dataset:
  edge_cosine       : mean cos(v_A, v_B) over relational edges = the king<->queen gap
                      (low on KB = far apart = where plain semantics fails)
  recover_plain@M   : % edges where B is in the top-M nearest of A (no offset)
  recover_offset@M  : % where B is in top-M of (A + best-of-K r_k)   [try-all-offsets]
  recover_oracle@M  : % where B is in top-M of (A + r_{k*}), k* = cluster of the true delta
If offset >> plain on metaqa, relations are learnable-as-offsets -> fold into the
router (auxiliary link-prediction loss / offset-augmented routing) to cut PPR reliance.
Writes results/research/rel_probe_{dataset}.json.
"""
import os
import json
import logging
import argparse

import numpy as np
import faiss

from src.core.engine import CoreEngine
from src.experiments.overlap_retrain import _reconstruct

log = logging.getLogger("experiments.l3_relation_probe")


def _rel_edges(engine, id2idx):
    """directed relational edges (node.neighbors ∩ docs) — the structural/KB edges,
    NOT synthetic kNN (whose delta is ~0 by construction)."""
    docset = set(id2idx)
    a, b = [], []
    for node in engine.nodes:
        i = id2idx[node.node_id]
        for nb in node.neighbors:
            if nb in docset and nb != node.node_id:
                a.append(i); b.append(id2idx[nb])
    return np.array(a), np.array(b)


def run(dataset, K=16, edge_sample=20000, test_sample=2000, Ms=(10, 100, 500), seed_pad=0):
    engine = CoreEngine(source=dataset)
    X = _reconstruct(engine.node_index).astype("float32"); faiss.normalize_L2(X)
    n, d = X.shape
    id2idx = engine.node_id_to_idx
    A, B = _rel_edges(engine, id2idx)
    if len(A) == 0:
        log.warning(f"[{dataset}] no relational edges"); return None
    index = faiss.IndexFlatIP(d); index.add(X)

    # the king<->queen gap: how far apart related nodes are in cosine
    edge_cos = float(np.mean(np.sum(X[A] * X[B], axis=1)))

    # deltas -> discover relation directions by clustering
    rng = np.random.RandomState(0)
    if len(A) > edge_sample:
        sel = rng.choice(len(A), edge_sample, replace=False)
    else:
        sel = np.arange(len(A))
    deltas = (X[B[sel]] - X[A[sel]]).astype("float32")
    km = faiss.Kmeans(d, K, niter=20, seed=0, verbose=False)
    km.train(deltas)
    R = km.centroids.reshape(K, d).astype("float32")            # learned relation offsets
    _, dcl = km.index.search(deltas, 1)                          # cluster id per delta (for oracle)

    # test on a held-out edge sample: can we recover B from A (plain vs offset)?
    tsel = rng.choice(len(A), min(test_sample, len(A)), replace=False)
    aT, bT = A[tsel], B[tsel]
    _, deltacl = km.index.search((X[bT] - X[aT]).astype("float32"), 1)
    deltacl = deltacl.ravel()
    maxM = max(Ms)

    def _hits(query_vecs, targets):
        """for each row, is target in the top-maxM neighbors? return rank (or maxM+1)."""
        _, I = index.search(query_vecs.astype("float32"), maxM + 1)
        ranks = np.full(len(targets), maxM + 1)
        for i, t in enumerate(targets):
            row = I[i]
            pos = np.where(row == t)[0]
            if len(pos):
                ranks[i] = pos[0]
        return ranks

    plain = _hits(X[aT], bT)
    oracle = _hits(X[aT] + R[deltacl], bT)
    # try-all-offsets: best rank of B across A + r_k for all k
    best = np.full(len(bT), maxM + 1)
    for k in range(K):
        r = _hits(X[aT] + R[k], bT)
        best = np.minimum(best, r)

    def pct(ranks, m): return round(float(np.mean(ranks < m)) * 100, 1)
    out = {"dataset": dataset, "n_docs": n, "n_rel_edges": int(len(A)), "K": K,
           "edge_cosine_mean": round(edge_cos, 3),
           "recover_plain": {f"@{m}": pct(plain, m) for m in Ms},
           "recover_offset_bestK": {f"@{m}": pct(best, m) for m in Ms},
           "recover_offset_oracle": {f"@{m}": pct(oracle, m) for m in Ms},
           "median_rank": {"plain": int(np.median(plain)), "oracle": int(np.median(oracle)),
                           "bestK": int(np.median(best))}}
    os.makedirs("results/research", exist_ok=True)
    with open(f"results/research/rel_probe_{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log.info(f"[{dataset}] edge_cos={edge_cos:.3f} | plain {out['recover_plain']} | "
             f"oracle {out['recover_offset_oracle']} | bestK {out['recover_offset_bestK']}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Can graph relations be derived as semantic offsets?")
    p.add_argument("--datasets", nargs="+", default=["metaqa", "2wiki_clean", "musique_clean"])
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--test_sample", type=int, default=2000)
    a = p.parse_args(argv)
    for ds in a.datasets:
        log.info(f"===== RELATION-OFFSET PROBE: {ds.upper()} =====")
        run(ds, K=a.K, test_sample=a.test_sample)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
