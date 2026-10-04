"""Phase C1+C2 — MuSiQue-Ans. Doc = paragraph (title+paragraph_text), corpus = union over splits deduped
by text hash. Gold = is_supporting paragraphs. hop from id prefix. test gold hidden.
Writes data/canonical/musique/{documents,queries}.{jsonl} + manifests."""
import json, hashlib, os
BASE="data/original/musique/v1.0"
OUTD="data/canonical/musique"; os.makedirs(OUTD, exist_ok=True)
def sha(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()
FILES={"train":"musique_ans_v1.0_train.jsonl","dev":"musique_ans_v1.0_dev.jsonl","test":"musique_ans_v1.0_test.jsonl"}

# ---- C1 ----
text2id={}; docs=[]
for split,fn in FILES.items():
    for line in open(f"{BASE}/{fn}",encoding="utf-8"):
        m=json.loads(line)
        for p in m["paragraphs"]:
            txt=p["paragraph_text"]; h=sha(txt)
            if h in text2id: continue
            did=f"musique_{h[:16]}"; text2id[h]=did
            docs.append({"dataset":"musique","canonical_doc_id":did,"original_source_id":h[:16],
                "title":p["title"],"text":txt,"text_sha256":h,
                "source_file":"paragraphs (union of splits)","provenance":"official musique paragraph"})
with open(f"{OUTD}/documents.jsonl","w",encoding="utf-8") as f:
    for d in docs: f.write(json.dumps(d,ensure_ascii=False)+"\n")
docsha=hashlib.sha256(open(f"{OUTD}/documents.jsonl","rb").read()).hexdigest()
json.dump({"dataset":"musique","doc_unit":"paragraph (text-hash dedup)","n_docs":len(docs),
    "unique_ids":len(text2id),"unique_text_hashes":len(text2id),"documents_jsonl_sha256":docsha,
    "provenance":"union of official train/dev/test paragraphs"},open(f"{OUTD}/document_manifest.json","w"),indent=2)
print(f"musique C1: n_docs={len(docs)}")

# ---- C2 ----
qsum={}; open(f"{OUTD}/queries.jsonl","w").close()
for split,fn in FILES.items():
    recs=[]; goldmap=0; allmap=0; zero=0
    for i,line in enumerate(open(f"{BASE}/{fn}",encoding="utf-8")):
        m=json.loads(line)
        sup=[p["paragraph_text"] for p in m["paragraphs"] if p.get("is_supporting")]
        gold=[text2id[sha(t)] for t in sup if sha(t) in text2id]
        if gold: goldmap+=1
        if sup and len(gold)==len(sup): allmap+=1
        if sup and not gold: zero+=1
        hop=m["id"].split("hop")[0]+"hop" if "hop" in m["id"] else None
        recs.append({"dataset":"musique","query_id":m["id"],"official_split":split,"question":m["question"],
            "answers":([m["answer"]]+m.get("answer_aliases",[])) if m.get("answer") else [],
            "gold_doc_ids":gold,"hop":hop,"question_type":hop,
            "answerability":m.get("answerable"),"original_index":i,"source_file":fn})
    with open(f"{OUTD}/queries.jsonl","a",encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    qsum[split]={"n":len(recs),"gold_mapped":goldmap,"all_gold_mapped":allmap,"zero_gold_with_sup":zero}
    print(f"musique {split}: n={len(recs)} gold_mapped={goldmap} all_gold_mapped={allmap} zero_gold_with_sup={zero}")
qsha=hashlib.sha256(open(f"{OUTD}/queries.jsonl","rb").read()).hexdigest()
json.dump({"dataset":"musique","splits":qsum,"total":sum(v["n"] for v in qsum.values()),
    "queries_jsonl_sha256":qsha,"expected":{"train":19938,"dev":2417,"test":2459},
    "gold_policy":"is_supporting paragraphs -> canonical doc ids; test gold hidden"},
    open(f"{OUTD}/query_manifest.json","w"),indent=2)
print("musique DONE")
