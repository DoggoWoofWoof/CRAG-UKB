import json, os, collections, sys
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "freebase_scale", "ner", "ent")
c = collections.Counter(); ex = {}
for f in sorted(os.listdir(root)):
    if f.endswith(".json"):
        r = json.load(open(os.path.join(root, f)))
        k = r["code_sha256"][:12]
        c[k] += 1
        ex.setdefault(k, []).append((f, r.get("utc")))
for k, n in c.items():
    print(k, n, ex[k][0], ex[k][-1], min(u for _, u in ex[k]), max(u for _, u in ex[k]))
