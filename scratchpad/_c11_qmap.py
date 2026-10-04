"""Recover the exact corpus_qi -> queries_train row permutation for the cached gte_qwen query embeddings.
dense_score is affine in cosine(q_true, node), so the true query row has correlation ~1.0 between its cosine
profile (over a query's candidates) and dense_score. Match by top-K-dense-candidate correlation; cache to npz.
This is representation bookkeeping only — no new encoder forward passes."""
import sys, os, time
sys.path.insert(0, "scratchpad")
import numpy as np
import l2_c8 as C8
OUT = C8.OUT; DS = C8.DS; K = 24; log = lambda *a: print(*a, flush=True); T0 = time.time()


def recover(ds):
    off = np.load(f"data/l2_corpus/{ds}/train/query_offsets.npy"); Nq = len(off) - 1
    cand = np.load(f"data/l2_corpus/{ds}/train/cand_ids.npy", mmap_mode="r")
    dsc = np.load(f"data/l2_corpus/{ds}/train/dense_score.npy", mmap_mode="r")
    nodes = np.load(f"data/ukb_storage/{ds}/gte_qwen/nodes.npy")            # (Nn,1536) in RAM
    qtr = np.load(f"data/ukb_storage/{ds}/gte_qwen/queries_train.npy").astype(np.float32)  # (Nq,1536)
    qtr_n = qtr - qtr.mean(1, keepdims=True)                                 # for correlation
    qmap = np.full(Nq, -1, np.int64); corrs = np.zeros(Nq, np.float32); taken = np.zeros(len(qtr), bool)
    for qi in range(Nq):
        s, e = int(off[qi]), int(off[qi + 1]); n = e - s
        d = np.asarray(dsc[s:e], np.float32); top = np.argsort(-d)[:min(K, n)]
        ids = np.asarray(cand[s + top], np.int64); E = nodes[ids].astype(np.float32)   # (k,1536)
        S = qtr @ E.T                                                        # (Nq,k) cosine
        dc = d[top] - d[top].mean(); Sc = S - S.mean(1, keepdims=True)
        num = Sc @ dc; den = np.sqrt((Sc ** 2).sum(1)) * (np.sqrt((dc ** 2).sum()) + 1e-12) + 1e-12
        corr = num / den
        r = int(np.argmax(corr)); qmap[qi] = r; corrs[qi] = corr[r]; taken[r] = True
        if qi % 2000 == 0: log(f"[{time.time()-T0:.0f}s] {ds} qi={qi} corr={corr[r]:.4f}")
    bij = len(set(qmap.tolist())) == Nq
    log(f"[{time.time()-T0:.0f}s] {ds}: min_corr={corrs.min():.4f} mean_corr={corrs.mean():.4f} bijective={bij} "
        f"n_low={(corrs<0.999).sum()}")
    np.savez(f"{OUT}/_c11_qmap_{ds}.npz", qmap=qmap, corr=corrs)
    return corrs.min(), bij


def main():
    for ds in DS:
        recover(ds)
    log("C11_QMAP_DONE")


if __name__ == "__main__":
    main()
