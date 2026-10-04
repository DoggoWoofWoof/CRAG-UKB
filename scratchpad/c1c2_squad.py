"""Phase C1+C2 — SQuAD v2.0. Doc = unique context paragraph (dedup by text). Gold = the context the
question was written against. answerability = not is_impossible. No public test -> train+dev only.
Writes data/canonical/squad/{documents,queries}.jsonl + manifests."""
import json, hashlib, os
BASE="data/original/squad/v2.0"
OUTD="data/canonical/squad"; os.makedirs(OUTD, exist_ok=True)
def sha(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()
FILES={"train":"train-v2.0.json","dev":"dev-v2.0.json"}

# ---- C1: unique contexts across splits ----
text2id={}; docs=[]
for split,fn in FILES.items():
    d=json.load(open(f"{BASE}/{fn}",encoding="utf-8"))
    for art in d["data"]:
        for p in art["paragraphs"]:
            ctx=p["context"]; h=sha(ctx)
            if h in text2id: continue
            did=f"squad_{h[:16]}"; text2id[h]=did
            docs.append({"dataset":"squad","canonical_doc_id":did,"original_source_id":h[:16],
                "title":art["title"],"text":ctx,"text_sha256":h,
                "source_file":"contexts (union train+dev)","provenance":"official squad2 context paragraph"})
with open(f"{OUTD}/documents.jsonl","w",encoding="utf-8") as f:
    for d in docs: f.write(json.dumps(d,ensure_ascii=False)+"\n")
docsha=hashlib.sha256(open(f"{OUTD}/documents.jsonl","rb").read()).hexdigest()
json.dump({"dataset":"squad","doc_unit":"context paragraph (text-hash dedup)","n_docs":len(docs),
    "unique_ids":len(text2id),"unique_text_hashes":len(text2id),"documents_jsonl_sha256":docsha,
    "provenance":"union of official train+dev contexts (no public test)"},
    open(f"{OUTD}/document_manifest.json","w"),indent=2)
print(f"squad C1: n_docs={len(docs)}")

# ---- C2 ----
qsum={}; open(f"{OUTD}/queries.jsonl","w").close()
for split,fn in FILES.items():
    d=json.load(open(f"{BASE}/{fn}",encoding="utf-8"))
    recs=[]; goldmap=0; nimp=0
    idx=0
    for art in d["data"]:
        for p in art["paragraphs"]:
            did=text2id[sha(p["context"])]
            for qa in p["qas"]:
                imp=qa.get("is_impossible",False)
                if imp: nimp+=1
                ans=list(dict.fromkeys([a["text"] for a in qa.get("answers",[])]))
                gold=[did]  # context is always the gold doc (answerable or not)
                goldmap+=1
                recs.append({"dataset":"squad","query_id":qa["id"],"official_split":split,"question":qa["question"],
                    "answers":ans,"gold_doc_ids":gold,"hop":None,"question_type":None,
                    "answerability":(not imp),"original_index":idx,"source_file":fn}); idx+=1
    with open(f"{OUTD}/queries.jsonl","a",encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    qsum[split]={"n":len(recs),"gold_mapped":goldmap,"impossible":nimp,"answerable":len(recs)-nimp}
    print(f"squad {split}: n={len(recs)} gold_mapped={goldmap} impossible={nimp} answerable={len(recs)-nimp}")
qsha=hashlib.sha256(open(f"{OUTD}/queries.jsonl","rb").read()).hexdigest()
json.dump({"dataset":"squad","splits":qsum,"total":sum(v["n"] for v in qsum.values()),
    "queries_jsonl_sha256":qsha,"expected":{"train":130319,"dev":11873},
    "gold_policy":"question's own context paragraph = gold doc; answerability=not is_impossible"},
    open(f"{OUTD}/query_manifest.json","w"),indent=2)
print("squad DONE")
