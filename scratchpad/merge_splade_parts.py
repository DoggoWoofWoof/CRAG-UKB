"""Merge query-sharded splade parts into one cache, locally.

WHY LOCALLY AND NOT ON MODAL

  The parts are split across two Modal workspaces -- eighteen were written before the first
  workspace was disabled and the rest in a second one -- and volumes do not cross workspaces.
  The merge is pure concatenation in query order, so where it runs is immaterial; what matters
  is that it REFUSES a partial cache rather than writing one with holes.

  A part's [lo, hi) is a pure function of (qshard, n_qshards), so parts made in different
  workspaces still tile the query axis exactly.  That is checked here rather than assumed: the
  merge asserts every part's row count equals hi - lo, that the union covers every query row
  once, and that the total equals the dataset's canonical query count.
"""
import glob
import io
import json
import os
import sys

import numpy as np

FC = "data/final_canonical"


def _rows_per_backend(metas):
    """How many query rows each backend produced. A mixed artifact should say so in rows."""
    out = {}
    for m in metas:
        b = m.get("backend", "unrecorded")
        out[b] = out.get(b, 0) + int(m["hi"]) - int(m["lo"])
    return out


def merge(ds, n_qshards):
    d = "scratchpad/_splade_parts/%s" % ds
    parts = sorted(glob.glob("%s/part_*_of_%04d.npz" % (d, n_qshards)))
    if len(parts) != n_qshards:
        have = sorted(int(os.path.basename(p)[5:9]) for p in parts)
        missing = [i for i in range(n_qshards) if i not in have]
        raise SystemExit("%s: have %d of %d parts, missing %s -- refusing to merge a partial "
                         "cache" % (ds, len(parts), n_qshards, missing))
    nq_expected = len(json.load(io.open(
        "%s/%s/queries/pointer_index/query_ids.json" % (FC, ds), encoding="utf-8")))
    metas = [json.load(io.open(p[:-4] + ".meta.json", encoding="utf-8")) for p in parts]
    nq = max(m["hi"] for m in metas)
    if nq != nq_expected:
        raise SystemExit("%s: parts cover %d rows, %d canonical queries" % (ds, nq, nq_expected))
    k = int(np.load(parts[0])["ids"].shape[1])
    ids = np.empty((nq, k), dtype=np.int32)
    scf = np.empty((nq, k), dtype=np.float16)
    seen = np.zeros(nq, dtype=bool)
    for p, m in zip(parts, metas):
        z = np.load(p)
        if z["ids"].shape[0] != m["hi"] - m["lo"]:
            raise SystemExit("%s has %d rows but claims %d"
                             % (p, z["ids"].shape[0], m["hi"] - m["lo"]))
        if seen[m["lo"]:m["hi"]].any():
            raise SystemExit("%s overlaps a part already written" % p)
        ids[m["lo"]:m["hi"]] = z["ids"]
        scf[m["lo"]:m["hi"]] = z["scores"]
        seen[m["lo"]:m["hi"]] = True
    if not seen.all():
        raise SystemExit("%s: %d query rows never written" % (ds, int((~seen).sum())))
    od = "%s/%s/retrieval_cache" % (FC, ds)
    os.makedirs(od, exist_ok=True)
    o = "%s/splade_top%d.npz" % (od, k)
    np.savez(o, ids=ids, scores=scf)
    rec = {"n_queries": int(nq), "K": int(k), "merged_from": n_qshards,
           "seconds_summed_over_shards": round(sum(m["seconds"] for m in metas), 1),
           "slowest_shard_seconds": round(max(m["seconds"] for m in metas), 1),
           "score_min": float(scf.min()), "score_max": float(scf.max()),
           "top1_score_median": float(np.median(scf[:, 0])),
           "n_docs": metas[0]["n_docs"], "bytes": os.path.getsize(o),
           "merged_locally": True,
           # Read off the parts, never asserted. 2wiki ended up mixed: 13 shards on Modal
           # before the workspace was disabled, the remaining 11 locally after every Modal
           # workspace hit its spend limit. A merged artifact that claimed one backend for
           # rows built by two would be false, and it is exactly the kind of false that no
           # later check would catch, because the bytes are fine.
           "backend_by_part": {int(m["qshard"]): m.get("backend", "unrecorded")
                               for m in metas},
           "backend_row_counts": _rows_per_backend(metas),
           "backend": "MIXED -- see backend_by_part" if len(
               {m.get("backend", "") for m in metas}) > 1
               else (metas[0].get("backend", "unrecorded"))}
    json.dump(rec, io.open("%s/splade_top%d.meta.json" % (od, k), "w", encoding="utf-8"),
              indent=1)
    print("%s: MERGED %s  %.3f GB  %d queries  top1_median=%.4f"
          % (ds, o, rec["bytes"] / 1e9, nq, rec["top1_score_median"]))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        ds, n = a.split(":")
        merge(ds, int(n))
