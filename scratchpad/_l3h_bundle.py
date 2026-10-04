"""Population bundle for the host L3 lane (stage L3_HOST).

The L3 harnesses read the frozen top-1000 retrieval caches (data/final_canonical/<ds>/retrieval_cache/{dense,splade}_top1000.npz,
2.45 GB each for MetaQA) only row-wise for the L1_DEV population (the FLAT agreement gate and the dense top-1 fallback seed).
The host link is ~1 MB/s ("move code, not data"), so the host never receives the caches: this module copies the population rows of
every per-query member of both caches (ids and scores at depth 1000, exactly as stored) into one bundle, written once, with the shas of
the source caches (recomputed, and checked against the dataset manifest) and of the population record.

The bundle is a derived development artifact (results/L3_HOST/), never a canonical file; nothing under data/ is written.
The host runner (_l3h_run.py) serves CanonicalDataset.cache(model) from it and refuses any row outside it.

usage: python scratchpad/_l3h_bundle.py BUILD <ds>  -> results/L3_HOST/bundle_<ds>__L1DEV.{npz,json}
"""
import json
import os
import sys
import time

import numpy as np

import _l1d_lib as D

OUTB = os.path.join(D.REPO, "results", "L3_HOST")


def build(ds):
    t0 = time.time()
    os.makedirs(OUTB, exist_ok=True)
    fz = os.path.join(OUTB, "bundle_%s__L1DEV.npz" % ds)
    fj = os.path.join(OUTB, "bundle_%s__L1DEV.json" % ds)
    assert not os.path.exists(fz) and not os.path.exists(fj), "write-once: the bundle of %s exists" % ds
    cd = D.AD.CanonicalDataset(ds)
    pop = D.Population(cd, None)
    rows = np.asarray(pop.rows, np.int64)
    nq_all = len(cd.query_ids)
    arrays = {"rows": rows}
    src = {}
    for mdl in ("dense", "splade"):
        path = os.path.join(cd.dir, "retrieval_cache", "%s_top1000.npz" % mdl)
        c = cd.cache(mdl)
        members = {}
        for k in c.files:
            a = c[k]
            if getattr(a, "ndim", 0) >= 1 and a.shape[0] == nq_all:
                arrays["%s__%s" % (mdl, k)] = np.ascontiguousarray(np.asarray(a[rows]))
                members[k] = {"per_row": True, "dtype": str(a.dtype), "shape_full": list(a.shape)}
            else:
                arrays["%s__%s" % (mdl, k)] = np.asarray(a)
                members[k] = {"per_row": False, "dtype": str(a.dtype), "shape_full": list(np.shape(a))}
        sha = D.sha_file(path)
        man = cd.manifest.get("retrieval_cache", {}).get(mdl, {}).get("sha256")
        assert man is None or man == sha, "the %s cache differs from its manifest sha" % mdl
        src[mdl] = {"path": D.rel(path), "bytes": os.path.getsize(path), "sha256": sha, "manifest_sha256": man, "members": members}
        log_("  %s: %s" % (mdl, json.dumps(members)))
    np.savez_compressed(fz, **arrays)
    rec = {"stage": "L3_HOST", "what": "population bundle: the L1_DEV population rows of both top-1000 retrieval caches, exactly as stored",
           "dataset": ds, "n_rows": int(len(rows)), "n_queries_in_cache": nq_all, "population": pop.record,
           "rows_sha256": D.sha_text(",".join(str(int(r)) for r in rows)), "sources": src,
           "code": {"path": "scratchpad/_l3h_bundle.py", "sha256": D.sha_file(os.path.abspath(__file__)),
                    "lib": {"path": "scratchpad/_l1d_lib.py", "sha256": D.sha_file(os.path.join(D.HERE, "_l1d_lib.py"))}},
           "npz": {"path": D.rel(fz), "bytes": os.path.getsize(fz), "sha256": D.sha_file(fz),
                   "arrays": {k: [str(v.dtype), list(v.shape)] for k, v in arrays.items()}},
           "platform": D.platform_record(), "seconds": round(time.time() - t0, 1)}
    with open(fj, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    log_("bundle %s: %d rows -> %s (%d bytes, sha %s)" % (ds, len(rows), D.rel(fz), rec["npz"]["bytes"], rec["npz"]["sha256"][:16]))


def log_(s):
    D.log(s)


def main():
    assert len(sys.argv) == 3 and sys.argv[1] == "BUILD", __doc__
    build(sys.argv[2])


if __name__ == "__main__":
    main()
