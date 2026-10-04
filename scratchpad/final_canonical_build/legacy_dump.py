"""Phase 0/6 helper: stream the LEGACY master_nodes_{ds}.json (the node table the frozen G2 caches
index into, via _ta_prepartition.load_topology) and write compact per-dataset tables:

  data/final_canonical/_work/legacy/{ds}/docs.jsonl       node_id,title,sha_exact,sha_ws,sha_nfc_ws,len,n_nb,[content]
  data/final_canonical/_work/legacy/{ds}/questions.jsonl  node_id,content,neighbors(golds),metadata
  data/final_canonical/_work/legacy/{ds}/summary.json

Read-only over data/processed; streaming (peak RSS << 1 GB). Usage: python legacy_dump.py [ds ...]
"""
import sys, os, json, hashlib, unicodedata, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jsonstream import iter_json_array

ROOT = "data/final_canonical/_work/legacy"
KEEP_CONTENT = {"2wiki_clean", "musique_clean", "squad_clean"}   # metaqa content is a verbalized triple bag, not the doc text


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def ws(s):
    return " ".join(s.split())


def dump(ds):
    t0 = time.time()
    path = f"data/processed/master_nodes_{ds}.json"
    out = f"{ROOT}/{ds}"; os.makedirs(out, exist_ok=True)
    nd = nq = 0; types = {}; srcs = {}; sample_doc = None
    with open(f"{out}/docs.jsonl", "w", encoding="utf-8") as fd, \
         open(f"{out}/questions.jsonl", "w", encoding="utf-8") as fq:
        for n in iter_json_array(path):
            md = n.get("metadata", {})
            t = md.get("type"); types[t] = types.get(t, 0) + 1
            srcs[md.get("source")] = srcs.get(md.get("source"), 0) + 1
            if t == "question":
                nq += 1
                fq.write(json.dumps({"node_id": n["node_id"], "content": n["content"],
                                     "neighbors": n.get("neighbors", []), "metadata": md}, ensure_ascii=False) + "\n")
            else:
                c = n["content"]
                rec = {"node_id": n["node_id"], "title": md.get("title", ""), "sha_exact": sha(c),
                       "sha_ws": sha(ws(c)), "sha_nfc_ws": sha(ws(unicodedata.normalize("NFC", c))),
                       "len": len(c), "n_nb": len(n.get("neighbors", []))}
                if ds in KEEP_CONTENT:
                    rec["content"] = c
                if sample_doc is None:
                    sample_doc = {k: (v if k != "content" else v[:200]) for k, v in rec.items()}
                    sample_doc["metadata_keys"] = sorted(md.keys())
                fd.write(json.dumps(rec, ensure_ascii=False) + "\n")
                nd += 1
    summ = {"dataset": ds, "master_path": path, "master_bytes": os.path.getsize(path),
            "n_docs": nd, "n_questions": nq, "types": types, "sources": srcs, "sample_doc": sample_doc,
            "seconds": round(time.time() - t0, 1)}
    json.dump(summ, open(f"{out}/summary.json", "w"), indent=2)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["musique_clean", "squad_clean", "2wiki_clean", "metaqa"]):
        dump(ds)
