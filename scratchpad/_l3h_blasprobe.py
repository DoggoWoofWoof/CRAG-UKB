"""BLAS probe for the host L3 lane (stage L3_HOST): the FLAT dense score is the float32 product Qu[j0:j1] @ Ec.T (OpenBLAS sgemm),
and OpenBLAS partitions that product by its thread count, so the float sums -- and near-tied FLAT ranks -- depend on the number of
BLAS threads.  The laptop's records were computed with its default, 12 threads (Haswell kernel, OpenBLAS 0.3.30).  This probe
computes the product of the first L1_DEV population batch (200 rows) with the first node block, as the harnesses do, and records the
BLAS configuration and the sha of the product under the process's own settings and, when threadpoolctl is present, per thread count.

usage: python scratchpad/_l3h_blasprobe.py <label>  -> results/L3_HOST/BLAS_PROBE__<label>.json (write-once)
"""
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import _l1d_lib as D  # noqa: E402

THREADS = (1, 2, 4, 8, 12, 16, 32)


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    label = sys.argv[1]
    out = os.path.join(D.REPO, "results", "L3_HOST", "BLAS_PROBE__%s.json" % label)
    assert not os.path.exists(out), "write-once"
    try:
        import threadpoolctl
        info = threadpoolctl.threadpool_info()
    except Exception as e:
        threadpoolctl, info = None, "threadpoolctl unavailable: %s: %s" % (type(e).__name__, e)
    env = {k: v for k, v in sorted(os.environ.items()) if any(s in k.upper() for s in ("THREAD", "OPENBLAS", "OMP_", "MKL_"))}
    cd = D.AD.CanonicalDataset("metaqa")
    pop = D.Population(cd, None)
    rows = pop.rows[:200]
    Qu = D.unit_queries(cd, rows)
    a, b_, blk = next(iter(D.chunk_iter(cd.node_embeddings)))
    Ec = np.ascontiguousarray(np.asarray(blk, np.float32))
    ref = D.dense_products(Qu, Ec, 0, len(rows))
    rec = {"stage": "L3_HOST", "probe": "BLAS thread dependence of the FLAT dense product", "label": label,
           "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "blas": info, "env": env,
           "numpy": np.__version__, "platform": D.platform_record(), "host": D.host_state(),
           "inputs": {"rows": "the first 200 rows of the L1_DEV population", "Qu_sha256": h(Qu), "node_block": [int(a), int(b_)],
                      "Ec_sha256": h(Ec)},
           "product_default": {"shape": list(ref.shape), "sha256": h(ref)},
           "code": {"path": "scratchpad/_l3h_blasprobe.py", "sha256": D.sha_file(os.path.abspath(__file__))}}
    if threadpoolctl is not None:
        per = {}
        for t in THREADS:
            with threadpoolctl.threadpool_limits(limits=t, user_api="blas"):
                r = D.dense_products(Qu, Ec, 0, len(rows))
            per[str(t)] = {"sha256": h(r), "equals_default": bool((r == ref).all()), "n_diff_vs_default": int((r != ref).sum()),
                           "max_abs_diff_vs_default": float(np.abs(r - ref).max())}
        rec["product_per_thread_count"] = per
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    D.log("BLAS probe %s: default product sha %s -> %s" % (label, rec["product_default"]["sha256"][:16], D.rel(out)))


if __name__ == "__main__":
    main()
