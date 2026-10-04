# -*- coding: utf-8 -*-
"""
Why does the DENSE cache fail an independent recompute when SPLADE passes it 8/8?

Two explanations fit the pattern and they have opposite consequences:

  A  BENIGN. The cached ids are right and the ORDER is ambiguous at the last bit. The builder
     computes a (queries x 1536) @ (1536 x block) matrix-matrix product; the verifier computes
     (docs x 1536) @ (1536,) matrix-vector. BLAS sums those in different orders, so two
     documents whose true scores are equal -- common here, because the pointer index is
     many-to-one and token-identical texts SHARE a vector -- can come out bit-equal in one
     computation and one ulp apart in the other. Sparse splade dots have a fixed accumulation
     order, which is exactly why splade does not show it.

  B  REAL. The cache holds the wrong documents.

These are not distinguishable by counting mismatching queries, which is all the gate did. They
ARE distinguishable by looking at the SCORES of the documents that disagree:

  * under A the disagreeing documents have equal-to-float-noise scores, and the cached set and
    the reference set differ only where the ranking is genuinely a coin flip;
  * under B some cached document is materially worse than one the reference chose, and the
    cache is missing real top-K mass.

So this measures the score gap at every disagreement, and arbitrates with a float64 recompute,
which has ~11 more bits than either float32 path and therefore breaks near-ties the way the
true value would.
"""
import io
import json
import os
import sys
import time

import numpy as np

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "data/final_canonical")
from pointer_resolver import CanonicalEmbeddings  # noqa: E402

ROOT = "data/final_canonical"
OUT = ROOT + "/DENSE_CACHE_ORDER_DIAGNOSIS.json"
N_SAMPLE = 8


def main():
    t0 = time.time()
    rec = {"RECORD": "DENSE_CACHE_ORDER_DIAGNOSIS",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "datasets": {}}
    worst_gap = 0.0
    worst_setdiff = 0
    total_q = 0
    all_tail = []

    for ds in ["metaqa", "musique", "squad"]:
        p = "%s/%s/retrieval_cache/dense_top1000.npz" % (ROOT, ds)
        if not os.path.exists(p):
            continue
        z = np.load(p)
        ids = z["ids"]
        E = CanonicalEmbeddings(ds, "dense", "docs", root=ROOT)
        Q = CanonicalEmbeddings(ds, "dense", "queries", root=ROOT)
        nd = len(E)
        docs32 = np.asarray(E.gather(np.arange(nd)), dtype=np.float32)
        docs64 = docs32.astype(np.float64)

        rng = np.random.RandomState(12345)          # the SAME sample the gate drew
        qs = np.unique(rng.randint(0, len(Q), size=N_SAMPLE * 3))[:N_SAMPLE]
        per = []
        for qi in qs:
            qv = np.asarray(Q.get(int(qi)), dtype=np.float32).ravel()
            s32 = (docs32 @ qv).astype(np.float32)
            s64 = docs64 @ qv.astype(np.float64)
            pos = np.arange(nd, dtype=np.int64)
            k = ids.shape[1]
            ref32 = np.lexsort((pos, -s32))[:k]
            ref64 = np.lexsort((pos, -s64))[:k]
            got = ids[qi].astype(np.int64)

            same_seq32 = bool(np.array_equal(ref32, got))
            same_set32 = int(len(set(ref32.tolist()) ^ set(got.tolist())) // 2)
            same_set64 = int(len(set(ref64.tolist()) ^ set(got.tolist())) // 2)

            # the score consequence: what does the cached selection cost against the float64
            # arbiter? if the cache picked equally-scoring documents this is exactly 0.
            gap = float(s64[ref64].sum() - s64[got].sum())
            # where do the sequence disagreements sit? a rank-1000 boundary effect is a
            # different animal from a rank-5 one.
            d = np.nonzero(ref32 != got)[0]
            first_bad = int(d[0]) if d.size else -1
            # are the disagreeing ranks actually ties?
            tie_like = True
            maxdelta = 0.0
            for r in d[:200]:
                a, b = float(s64[ref32[r]]), float(s64[got[r]])
                maxdelta = max(maxdelta, abs(a - b))
                if abs(a - b) > 1e-6:
                    tie_like = False
            per.append({"query_row": int(qi), "exact_sequence_match_f32": same_seq32,
                        "set_diff_vs_f32_reference": same_set32,
                        "set_diff_vs_f64_reference": same_set64,
                        "first_disagreeing_rank": first_bad,
                        "n_disagreeing_ranks": int(d.size),
                        "score_mass_lost_vs_f64_arbiter": gap,
                        "max_score_delta_at_disagreements": maxdelta,
                        "all_disagreements_are_ties": tie_like})
            worst_gap = max(worst_gap, abs(gap))
            worst_setdiff = max(worst_setdiff, same_set64)
            total_q += 1
            if first_bad >= 0:
                all_tail.append(first_bad)
            print("  %-8s q=%-7d seq32=%-5s setdiff32=%-4d setdiff64=%-4d firstbad=%-5d "
                  "maxdelta=%.3e masslost=%.3e"
                  % (ds, qi, same_seq32, same_set32, same_set64, first_bad, maxdelta, gap),
                  flush=True)
        rec["datasets"][ds] = {"n_docs": nd, "K": int(ids.shape[1]), "queries": per}
        del docs32, docs64

    rec["SUMMARY"] = {
        "queries_examined": total_q,
        "max_set_difference_vs_float64_arbiter": worst_setdiff,
        "max_score_mass_lost_vs_float64_arbiter": worst_gap,
        "median_first_disagreeing_rank": (int(np.median(all_tail)) if all_tail else None),
        "min_first_disagreeing_rank": (int(min(all_tail)) if all_tail else None)}
    rec["VERDICT"] = ("BENIGN_ORDER_AMBIGUITY" if worst_setdiff == 0 and worst_gap < 1e-3
                      else "REAL_CONTENT_DIFFERENCE")
    rec["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nmax set difference vs float64 arbiter : %d" % worst_setdiff)
    print("max score mass lost vs float64 arbiter: %.6e" % worst_gap)
    print("first disagreeing rank: min %s median %s"
          % (rec["SUMMARY"]["min_first_disagreeing_rank"],
             rec["SUMMARY"]["median_first_disagreeing_rank"]))
    print("VERDICT: %s   %.0fs" % (rec["VERDICT"], rec["elapsed_s"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
