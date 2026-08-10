"""
Does OVERLAP voting (the graph signal) help L1 partition routing?
=================================================================
L1 routing = retrieved docs vote for partitions; overlap means a doc votes for its OWN partition AND
its 1-hop graph-neighbours' partitions (the graph signal smuggled into routing). This isolates whether
that graph signal is worth anything at PARTITION granularity, by comparing the SAME voting with two
memberships:

  overlap    : mem_idx = own partition UNION 1-hop graph-neighbour partitions   (current mechanism)
  no-overlap : each doc votes ONLY for its own hard partition                    (graph removed)

Pooling/scoring rule is identical (a gold is 'covered' if its hard partition is in the top-K voted set);
only the vote membership differs. If overlap ~= no-overlap, the graph adds ~nothing to routing -> it is
WASTED at partition granularity, and only pays off at node granularity (traversal, see l3_graphlift).
Writes results/L2/overlap_test_{subdir}.json.
"""
import os
import json
import logging
import argparse

import numpy as np

log = logging.getLogger(__name__)


def run(datasets=None, subdir="gte_qwen", scopes=(5, 20, 50), te_cap=2000):
    import torch
    from src.experiments.l1_universal_head import _load, _votes, _dense_order

    datasets = datasets or ["musique_clean", "2wiki_clean", "squad_clean", "metaqa", "hotpotqa_clean"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("results/L2", exist_ok=True)
    out = {}
    for d in datasets:
        data = _load(d, subdir, 2000, 1, te_cap)
        X = data["X"]; hard = np.asarray(data["hard"]); mem_idx = data["mem_idx"]; npart = data["npart"]
        qte, _, gte = data["test"]
        mem_overlap = mem_idx                                       # own + 1-hop neighbour partitions
        mem_hardonly = [[int(hard[i])] for i in range(len(hard))]   # own partition only (graph removed)
        avg_memb = round(float(np.mean([len(m) for m in mem_overlap])), 2)

        do = _dense_order(qte, X, device)                          # dense doc order (the voters)
        v_ov = _votes(do, mem_overlap, npart)
        v_no = _votes(do, mem_hardonly, npart)

        def _metrics(votes, scope):
            topP = [set(np.argsort(-votes[qi])[:scope].tolist()) for qi in range(len(qte))]
            cov_h, cov_m, hit = [], [], []
            for qi, gg in enumerate(gte):
                if not gg:
                    continue
                # HARD coverage (retrieval-honest: gold's single pooling partition) vs MEMBERSHIP coverage
                # (champion _fullcov: gold covered if ANY of its overlapping buckets is selected)
                cov_h.append(sum(1 for g in gg if int(hard[g]) in topP[qi]) / len(gg))
                cov_m.append(sum(1 for g in gg if any(p in topP[qi] for p in mem_overlap[g])) / len(gg))
                hit.append(1.0 if any(int(hard[g]) in topP[qi] for g in gg) else 0.0)
            return (round(100 * float(np.mean(cov_h)), 2), round(100 * float(np.mean(cov_m)), 2),
                    round(100 * float(np.mean(hit)), 2))

        out[d] = {"corpus_N": int(X.shape[0]), "npart": int(npart), "avg_membership": avg_memb, "by_scope": {}}
        for scope in scopes:
            o_h, o_m, _ = _metrics(v_ov, scope)                    # overlap voting: hard-cov, membership-cov
            n_h, n_m, _ = _metrics(v_no, scope)                    # no-overlap voting
            out[d]["by_scope"][str(scope)] = {
                # champion-style comparison (membership coverage = gold covered if ANY of its buckets selected)
                "overlap_membcov": o_m, "nooverlap_membcov": n_m, "memb_gap": round(o_m - n_m, 2),
                # retrieval-honest comparison (hard coverage = gold's single pooling partition)
                "overlap_hardcov": o_h, "nooverlap_hardcov": n_h, "hard_gap": round(o_h - n_h, 2),
                # bucket-inflation: same overlap selection, membership scoring MINUS hard scoring
                "bucket_inflation": round(o_m - o_h, 2),
            }
            log.info("[overlap/%s sc=%d] MEMB-cov(champion) ov=%.1f no=%.1f (Δ%+.1f) | HARD-cov(honest) "
                     "ov=%.1f no=%.1f (Δ%+.1f) | bucket-inflation(o_m-o_h)=%+.1f [memb=%.1f]",
                     d, scope, o_m, n_m, o_m - n_m, o_h, n_h, o_h - n_h, o_m - o_h, avg_memb)
        import gc; gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    path = f"results/L2/overlap_test_{subdir}.json"
    json.dump(out, open(path, "w"), indent=2)
    log.info("-> %s", path)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Overlap vs no-overlap voting: does the graph help L1 partition routing?")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--subdir", default="gte_qwen")
    p.add_argument("--scopes", type=int, nargs="+", default=[5, 20, 50])
    p.add_argument("--te-cap", type=int, default=2000)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run(datasets=a.datasets, subdir=a.subdir, scopes=tuple(a.scopes), te_cap=a.te_cap)


if __name__ == "__main__":
    main()
