"""FBX_SCALE Track A -- the long tail of Freebase name lengths (why the NER EXTRACT needs a length rule): counts of names above character thresholds, the longest names' positions.

  python -u scratchpad/_fbx_name_tail.py <procs>      # read-only over the node shards' `name` column; prints one JSON line; writes results/FREEBASE_SCALE/ner/FBX_NAME_TAIL__v1.json (write-once)
"""
import io
import json
import os
import sys
import time

import numpy as np

import _fbx_ner_build as NB

THR = [1000, 10000, 50000, 100000, 250000, 500000, 1000000]
OUT = os.path.join(NB.RECS, "FBX_NAME_TAIL__v1.json")


def unit_tail(u):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    s, g, r0, n = u
    tb = pq.ParquetFile(os.path.join(NB.TREE, "nodes", "shard_%05d.parquet" % s)).read_row_group(g, columns=["position", "name"])
    ln = pc.utf8_length(tb.column("name").fill_null("")).to_numpy(zero_copy_only=False).astype(np.int64)
    pos = tb.column("position").to_numpy()
    top = np.argsort(-ln)[:5]
    return {"n": int(n), "over": [int((ln > t).sum()) for t in THR], "max": int(ln.max()), "top": [(int(ln[i]), int(pos[i])) for i in top], "chars": int(ln.sum())}


def main(procs):
    assert not os.path.exists(OUT), "write-once"
    us = NB.units()
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(procs) as pool:
        rs = pool.map(unit_tail, us, chunksize=4)
    over = np.sum([r["over"] for r in rs], axis=0).tolist()
    top = sorted((x for r in rs for x in r["top"]), reverse=True)[:20]
    rec = {"RECORD": "FBX_NAME_TAIL", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "units": len(rs), "rows": int(sum(r["n"] for r in rs)), "total_chars": int(sum(r["chars"] for r in rs)),
           "thresholds_chars": THR, "names_longer_than": over, "max_chars": int(max(r["max"] for r in rs)), "longest_20_(chars, position)": top, "seconds": round(time.time() - t0, 1),
           "code_sha256": NB.sha256(os.path.abspath(__file__))}
    NB.wj(OUT, rec)
    print(json.dumps({k: rec[k] for k in ("rows", "thresholds_chars", "names_longer_than", "max_chars", "seconds")}))


if __name__ == "__main__":
    main(int(sys.argv[1]))
