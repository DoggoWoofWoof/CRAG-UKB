"""
L1 candidate-generator ablation — the defense of the partition structure.
=========================================================================
The skeptic's question: "to shrink the L2 rerank pool, why the partition + overlap-voting graph —
why not just dense top-N?" This measures, at MATCHED per-query budget P (P = the query's partition-pool
size), the gold coverage of two candidate generators over the SAME number of docs:

  PARTITION  : docs in the top-K voted partitions (dense top-200 vote for their partition + 1-hop graph
               neighbors' partitions) — a graph-structured neighborhood pool.
  DENSE-topP : the P docs of highest query-doc cosine — the obvious content-similarity pool.

If PARTITION recall >> DENSE-topP recall, the graph pool recovers golds dense ranks far down (multi-hop
bridge entities at dense rank 1000s) that a same-size dense pool never sees -> the partition structure
earns its place on RECALL, not just memory/latency. If they're equal, the structure is not justified.
Reports fraction-of-golds recall and hit (>=1 gold) for both, plus the gap. Writes results/L2/candgen_{subdir}.json.
"""
import os
import json
import logging
import argparse

import numpy as np

log = logging.getLogger(__name__)


def run(datasets=None, subdir="gte_qwen", scopes=(3, 6, 15, 50), te_cap=2000):
    import torch
    from src.experiments.l1_universal_head import _load
    from src.experiments.l2_seed import _topP, MAXK

    datasets = datasets or ["musique_clean", "2wiki_clean", "squad_clean", "metaqa", "hotpotqa_clean"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results/L2", exist_ok=True)
    out = {}
    for d in datasets:
        data = _load(d, subdir, 2000, 1, te_cap)
        X = np.ascontiguousarray(data["X"], dtype="float32"); hard = np.asarray(data["hard"])
        mem_idx = data["mem_idx"]; npart = data["npart"]
        qte, _, gte = data["test"]
        n = X.shape[0]
        Xd = torch.tensor(X, device=device)
        qt = torch.tensor(qte, device=device)

        # dense full sims cached per query (reused across all scopes); + gold dense ranks
        I = np.empty((len(qte), min(MAXK, n)), dtype=np.int64)
        drank = []
        with torch.no_grad():
            for s in range(0, len(qte), 256):
                I[s:s + 256] = torch.topk(qt[s:s + 256] @ Xd.T, min(MAXK, n), dim=1).indices.cpu().numpy()
        out[d] = {"corpus_N": int(n), "by_scope": {}}
        for scope_topk in scopes:                                  # BUDGET SWEEP: does the partition edge grow as the pool shrinks?
            topP = _topP(I, mem_idx, npart, scope_topk)
            pc, ph, dc, dh, psz = [], [], [], [], []
            with torch.no_grad():
                for qi in range(len(qte)):
                    g = gte[qi]
                    if not g:
                        continue
                    gs = set(int(x) for x in g)
                    allow = np.isin(hard, np.fromiter(topP[qi], int))
                    P = int(allow.sum()); psz.append(P)
                    if P == 0:
                        pc.append(0.0); ph.append(0.0); dc.append(0.0); dh.append(0.0); continue
                    pool = set(np.where(allow)[0].tolist())        # PARTITION pool (graph-structured)
                    pc.append(len(gs & pool) / len(gs)); ph.append(1.0 if (gs & pool) else 0.0)
                    sim = (qt[qi:qi + 1] @ Xd.T)[0]                # DENSE-topP pool at the SAME budget P
                    dense_top = set(torch.topk(sim, min(P, n)).indices.cpu().numpy().tolist())
                    dc.append(len(gs & dense_top) / len(gs)); dh.append(1.0 if (gs & dense_top) else 0.0)
                    if scope_topk == scopes[-1]:                   # gold dense ranks once (largest scope)
                        gidx = torch.tensor(sorted(gs), device=device)
                        drank.extend(int(r) for r in (sim.unsqueeze(0) > sim[gidx].unsqueeze(1)).sum(dim=1).cpu().numpy())
            rec = {
                "mean_pool": round(float(np.mean(psz)), 1),
                "partition_recall": round(100 * float(np.mean(pc)), 2),
                "dense_topP_recall": round(100 * float(np.mean(dc)), 2),
                "recall_gap": round(100 * float(np.mean(pc) - np.mean(dc)), 2),
                "partition_hit": round(100 * float(np.mean(ph)), 2),
                "dense_topP_hit": round(100 * float(np.mean(dh)), 2),
                "hit_gap": round(100 * float(np.mean(ph) - np.mean(dh)), 2),
            }
            out[d]["by_scope"][str(scope_topk)] = rec
            log.info("[candgen/%s scope=%d] pool=%.0f | PART R=%.1f hit=%.1f | DENSE R=%.1f hit=%.1f | GAP R=%+.1f hit=%+.1f",
                     d, scope_topk, rec["mean_pool"], rec["partition_recall"], rec["partition_hit"],
                     rec["dense_topP_recall"], rec["dense_topP_hit"], rec["recall_gap"], rec["hit_gap"])
        out[d]["median_gold_dense_rank"] = int(np.median(drank)) if drank else -1
        del Xd, qt
        import gc; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    path = f"results/L2/candgen_sweep_{subdir}.json"
    json.dump(out, open(path, "w"), indent=2)
    log.info("-> %s", path)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="L1 candidate-generator ablation: partition pool vs dense top-P at matched budget.")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--subdir", default="gte_qwen")
    p.add_argument("--scopes", type=int, nargs="+", default=[3, 6, 15, 50], help="partition-budget sweep (pool grows with scope)")
    p.add_argument("--te-cap", type=int, default=2000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, subdir=a.subdir, scopes=tuple(a.scopes), te_cap=a.te_cap)


if __name__ == "__main__":
    main()
