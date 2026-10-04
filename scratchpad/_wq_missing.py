import os, sys, glob
R = "data/final_canonical/webqsp"
host = {}
for ln in open("data/_cache/webqsp_host_listing.tsv", encoding="utf-8"):
    ln = ln.rstrip("\n")
    if "\t" not in ln: continue
    s, p = ln.split("\t", 1)
    host[p] = int(s)
inputs = []
for sub in ("embeddings", "graph", "retrieval_cache", "v1"):
    for dp, dn, fn in os.walk(os.path.join(R, sub)):
        for f in fn:
            inputs.append(os.path.relpath(os.path.join(dp, f), R).replace("\\", "/"))
for f in ("queries/query_ids.json", "queries/train.jsonl"):
    inputs.append(f)
tot = 0; miss = []; done = 0; donesz = 0
for p in sorted(inputs):
    sz = os.path.getsize(os.path.join(R, p))
    if host.get(p) == sz:
        done += 1; donesz += sz
    else:
        miss.append((p, sz)); tot += sz
print("inputs", len(inputs), "on host complete", done, "%.2f GB" % (donesz/1e9), "missing", len(miss), "%.2f GB" % (tot/1e9))
with open("data/_cache/webqsp_missing.txt", "w", encoding="utf-8") as f:
    for p, sz in miss: f.write("%d\t%s\n" % (sz, p))
import collections
c = collections.Counter(p.split("/")[0] + "/" + (p.split("/")[1] if p.count("/") > 1 else "") for p, _ in miss)
print(c)
