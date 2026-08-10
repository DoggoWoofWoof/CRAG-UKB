"""
Standalone L2 memory microbench (clean process, real torch peak).
==================================================================
Quantifies the routed-pool memory saving with a MEASURED GPU peak that is NOT confounded by the
multi-dataset eval harness. It loads only the frozen embeddings + partition assignments + test
queries (via ``_load``; heads are never trained, the SPLADE model is never invoked), routes each
query to its top-K partitions, then measures peak GPU memory for two scoring paths, one query at a
time (the deployment unit):

  FULL   : corpus matrix X resident on GPU, score  q @ X^T  over all N docs    -> O(N) working set
  GATHER : X stays on CPU; fetch ONLY the routed-pool rows  X[pool]  to GPU    -> O(pool) working set

Because the pool is a near-constant ~scope*|partition| docs regardless of N, GATHER's working set is
O(1) in corpus size. It also checks that GATHER returns the SAME top-k as the MASKED-FULL path
(``topk_mismatch`` == 0), so the saving is not bought with a different result. This is the concrete,
runnable backing for the memory claim (vs the analytical N/P in l2_seed's _pool_stats).

Writes results/L2/mem_bench_{subdir}.json.
"""
import os
import json
import logging
import argparse

import numpy as np

log = logging.getLogger(__name__)


def _peak_mb(device):
    import torch
    return torch.cuda.max_memory_allocated() / 1e6 if str(device) == "cuda" else 0.0


def run(datasets=None, subdir="gte_qwen", scope_topk=50, sample=64, te_cap=2000):
    import torch
    from src.experiments.l1_universal_head import _load
    from src.experiments.l2_seed import _topP, MAXK

    datasets = datasets or ["hotpotqa_clean"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results/L2", exist_ok=True)
    out = {}
    for d in datasets:
        data = _load(d, subdir, 2000, 1, te_cap)               # tr_cap=1 -> skip train encode; X returned on CPU
        X = np.ascontiguousarray(data["X"], dtype="float32"); hard = np.asarray(data["hard"])
        mem_idx = data["mem_idx"]; npart = data["npart"]
        qte, _, gte = data["test"]
        n, dim = X.shape
        S = min(sample, len(qte))
        qt = torch.tensor(qte, device=device)

        # ---- routing: dense doc order (batched) -> overlap-voting -> top-K partitions ----
        Xd = torch.tensor(X, device=device)
        I = np.empty((len(qte), MAXK), dtype=np.int64)
        with torch.no_grad():
            for s in range(0, len(qte), 256):
                I[s:s + 256] = torch.topk(qt[s:s + 256] @ Xd.T, min(MAXK, n), dim=1).indices.cpu().numpy()
        topP = _topP(I, mem_idx, npart, scope_topk)
        allow_np = [np.isin(hard, np.fromiter(topP[qi], int)) for qi in range(S)]  # pool membership per sampled query
        pool_sizes = np.array([int(a.sum()) for a in allow_np], dtype=np.int64)

        # ---- FULL leg: corpus resident, one query at a time; record peak BEFORE any masking work ----
        masked_full = []
        with torch.no_grad():
            for qi in range(S):
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                sim = qt[qi:qi + 1] @ Xd.T                      # (1, n) — X resident is the whole point
                _ = torch.topk(sim, min(MAXK, n), dim=1)
                peak_full = _peak_mb(device) if qi == 0 else max(peak_full, _peak_mb(device))
                allow = torch.from_numpy(allow_np[qi]).to(device)   # correctness bookkeeping (peak already read)
                sm = torch.where(allow, sim[0], torch.full_like(sim[0], -1e9))
                masked_full.append(torch.topk(sm, min(MAXK, n)).indices.cpu().numpy())
                del sim, allow, sm
        del Xd
        if device == "cuda":
            torch.cuda.empty_cache()

        # ---- GATHER leg: X on CPU, fetch ONLY routed-pool rows; record peak; check top-k parity ----
        Xc = torch.from_numpy(X)                                # pinned-ish CPU tensor (host store stand-in)
        mism = 0
        with torch.no_grad():
            for qi in range(S):
                rows = np.where(allow_np[qi])[0]
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                Xg = Xc[torch.from_numpy(rows)].to(device)      # ONLY the pool (P x d) crosses to GPU
                sim = qt[qi:qi + 1] @ Xg.T
                loc = torch.topk(sim, min(MAXK, len(rows)), dim=1).indices.cpu().numpy()[0]
                peak_gather = _peak_mb(device) if qi == 0 else max(peak_gather, _peak_mb(device))
                del Xg, sim
                gather_glob = rows[loc]                          # map pool-local -> global doc ids
                mf = masked_full[qi]; mf = mf[mf >= 0]
                k = min(len(gather_glob), len(mf), 20)           # compare the usable top-20
                if set(int(x) for x in gather_glob[:k]) != set(int(x) for x in mf[:k]):
                    mism += 1

        mean_pool = float(pool_sizes.mean())
        rec = {
            "corpus_N": int(n), "dim": int(dim), "scope_topk": scope_topk, "sample": S,
            "mean_pool": round(mean_pool, 1), "max_pool": int(pool_sizes.max()),
            "emb_full_MB": round(n * dim * 4 / 1e6, 1), "emb_pool_MB": round(mean_pool * dim * 4 / 1e6, 2),
            "emb_reduction": round(n / max(1.0, mean_pool), 2),
            "peak_full_MB": round(peak_full, 1), "peak_gather_MB": round(peak_gather, 1),
            "peak_reduction": round(peak_full / max(1e-6, peak_gather), 2) if device == "cuda" else None,
            "topk_mismatch": int(mism),                          # 0 == gather identical to masked-full (correct)
        }
        out[d] = rec
        log.info("[mem-bench/%s] N=%d mean_pool=%.0f | emb %.0f->%.1f MB (%.1fx) | "
                 "peak %.0f->%.0f MB (%.1fx) | topk_mismatch=%d",
                 d, n, mean_pool, rec["emb_full_MB"], rec["emb_pool_MB"], rec["emb_reduction"],
                 rec["peak_full_MB"], rec["peak_gather_MB"], rec["peak_reduction"] or 0.0, mism)
        import gc
        del X, Xc, qt, masked_full
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    path = f"results/L2/mem_bench_{subdir}.json"
    json.dump(out, open(path, "w"), indent=2)
    log.info("-> %s", path)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L2 memory microbench: full-resident vs routed-pool-gather peak GPU.")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--subdir", default="gte_qwen")
    p.add_argument("--scope-topk", type=int, default=50)
    p.add_argument("--sample", type=int, default=64)
    p.add_argument("--te-cap", type=int, default=2000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, subdir=a.subdir, scope_topk=a.scope_topk, sample=a.sample, te_cap=a.te_cap)


if __name__ == "__main__":
    main()
