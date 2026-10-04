"""Peek helper (read-only): parquet schemas of RoG webqsp/cwq train+validation only (never test); original WebQSP.train.json keys."""
import os, json, glob
import psutil
import pyarrow.parquet as pq
psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for ds in ("webqsp", "cwq"):
    d = os.path.join(ROOT, "data", "original", ds, "rog_" + ds)
    for pat in ("validation-*", "train-00000-*"):
        fs = sorted(glob.glob(os.path.join(d, pat + ".parquet")))
        f = fs[0]
        pf = pq.ParquetFile(f)
        print(ds, os.path.basename(f), pf.metadata.num_rows, pf.metadata.num_row_groups)
        print(pf.schema_arrow)
        cols = [c for c in pf.schema_arrow.names if c != "graph"]
        rg = pf.read_row_group(0, columns=cols)
        row = rg.slice(0, 1).to_pylist()[0]
        print(json.dumps(row, default=str)[:1500])
        print("---")
print(open(os.path.join(ROOT, "data", "original", "cwq", "rog_cwq", "README.md"), encoding="utf-8").read())
p = os.path.join(ROOT, "data", "original", "webqsp", "WebQSP", "data", "WebQSP.train.json")
j = json.load(open(p, encoding="utf-8"))
print(j.keys(), len(j["Questions"]))
print(json.dumps(j["Questions"][0])[:2500])
print(open(os.path.join(ROOT, "data", "original", "cwq", "SOURCE_PROVENANCE.json"), encoding="utf-8").read()[:3500])
