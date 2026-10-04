"""Phase C1+C2 — 2WikiMultiHopQA. Doc = context paragraph (title + sentences), corpus = union over
train/dev/test contexts deduped by title. Gold = supporting_facts titles. Queries uncapped, test gold hidden.
Writes data/canonical/2wiki/{documents.jsonl,document_manifest.json,queries.jsonl,query_manifest.json}."""
import json, hashlib, os
BASE="data/original/2wiki/v1.0_ids_april2021"
OUTD="data/canonical/2wiki"; os.makedirs(OUTD, exist_ok=True)
def sha(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()
SPLITS={"train":"train.json","dev":"dev.json","test":"test.json"}

# ---- C1: corpus from union of contexts, dedup by title ----
title2id={}; docs=[]; collide_difftext=0
for split,fn in SPLITS.items():
    d=json.load(open(f"{BASE}/{fn}",encoding="utf-8"))
    for q in d:
        for entry in q["context"]:
            title=entry[0]; text=" ".join(entry[1])
            if title in title2id:
                if title2id[title][1]!=sha(text): collide_difftext+=1
                continue
            did=f"2wiki_{sha(title)[:16]}"
            title2id[title]=(did,sha(text))
            docs.append({"dataset":"2wiki","canonical_doc_id":did,"original_source_id":title,
                "title":title,"text":text,"text_sha256":sha(text),
                "source_file":"context paragraphs (union of splits)","provenance":"official 2wiki context paragraph"})
with open(f"{OUTD}/documents.jsonl","w",encoding="utf-8") as f:
    for d in docs: f.write(json.dumps(d,ensure_ascii=False)+"\n")
docsha=hashlib.sha256(open(f"{OUTD}/documents.jsonl","rb").read()).hexdigest()
uniq_h=len({d["text_sha256"] for d in docs})
json.dump({"dataset":"2wiki","doc_unit":"context paragraph (title-level dedup)","n_docs":len(docs),
    "unique_ids":len(title2id),"unique_text_hashes":uniq_h,"title_collisions_difftext":collide_difftext,
    "documents_jsonl_sha256":docsha,"provenance":"union of official train/dev/test context paragraphs"},
    open(f"{OUTD}/document_manifest.json","w"),indent=2)
print(f"2wiki C1: n_docs={len(docs)} uniq_titles={len(title2id)} uniq_text={uniq_h} title_collide_difftext={collide_difftext}")

# ---- C2: queries ----
name2id={t:v[0] for t,v in title2id.items()}
qsum={}
open(f"{OUTD}/queries.jsonl","w").close()
for split,fn in SPLITS.items():
    d=json.load(open(f"{BASE}/{fn}",encoding="utf-8"))
    recs=[]; goldmap=0; zero=0; allmap=0
    for i,q in enumerate(d):
        sf_titles=list(dict.fromkeys([s[0] for s in q.get("supporting_facts",[])]))
        gold=[name2id[t] for t in sf_titles if t in name2id]
        if gold: goldmap+=1
        if sf_titles and len(gold)==len(sf_titles): allmap+=1
        if not gold and sf_titles: zero+=1
        recs.append({"dataset":"2wiki","query_id":q["_id"],"official_split":split,"question":q["question"],
            "answers":[q["answer"]] if q.get("answer") else [],"gold_doc_ids":gold,
            "gold_titles":sf_titles,"question_type":q.get("type"),"hop":None,
            "answerability":None,"original_index":i,"source_file":fn})
    with open(f"{OUTD}/queries.jsonl","a",encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    qsum[split]={"n":len(recs),"gold_mapped":goldmap,"all_gold_mapped":allmap,"zero_gold_with_sf":zero}
    print(f"2wiki {split}: n={len(recs)} gold_mapped={goldmap} all_gold_mapped={allmap} zero_gold_with_sf={zero}")
qsha=hashlib.sha256(open(f"{OUTD}/queries.jsonl","rb").read()).hexdigest()
json.dump({"dataset":"2wiki","splits":qsum,"total":sum(v["n"] for v in qsum.values()),
    "queries_jsonl_sha256":qsha,"expected":{"train":167454,"dev":12576,"test":12576},
    "gold_policy":"supporting_facts titles -> canonical doc ids; test gold hidden"},
    open(f"{OUTD}/query_manifest.json","w"),indent=2)
print("2wiki DONE")
