"""Phase C1+C2 — WebQSP. Two-layer provenance:
 C1 docs = Freebase ENTITIES (nodes of the per-question RoG subgraphs, all splits) -> structural_derived_from_freebase.
           This is NOT part of the official Microsoft WebQSP release; provenance recorded explicitly.
 C2 queries = OFFICIAL Microsoft WebQSP.train/test.json (3098/1639), answerability + parsing eligibility preserved.
           gold = answer entity names -> entity doc ids (RoG a_entity cross-check by id).
Writes data/canonical/webqsp/{documents,queries}.jsonl + manifests."""
import json, hashlib, os, glob
import pandas as pd
OUTD="data/canonical/webqsp"; os.makedirs(OUTD, exist_ok=True)
def sha(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()

# ---- C1: entity corpus from RoG subgraphs (train+val+test) ----
ROG=sorted(glob.glob("data/original/webqsp/rog_webqsp/*.parquet"))
name2id={}; docs=[]
rog_gold={}   # id -> list of a_entity names
rog_topic={}  # id -> list of q_entity names
def aslist(x):
    if x is None: return []
    try: return list(x)
    except TypeError: return [x]
def add(name):
    if not name or name in name2id: return
    did=f"webqsp_ent_{sha(name)[:16]}"; name2id[name]=did
    docs.append({"dataset":"webqsp","canonical_doc_id":did,"original_source_id":name,"title":name,
        "text":name,"text_sha256":sha(name),"source_file":"RoG-webqsp subgraph nodes",
        "provenance":"structural_derived_from_freebase (RoG-webqsp per-question subgraph node; NOT official MS WebQSP release)"})
for f in ROG:
    df=pd.read_parquet(f)
    for _,r in df.iterrows():
        for tr in aslist(r["graph"]):
            add(tr[0]); add(tr[2])
        for a in aslist(r["a_entity"]): add(a)
        for q in aslist(r["q_entity"]): add(q)
        rog_gold[r["id"]]=aslist(r["a_entity"])
        rog_topic[r["id"]]=aslist(r["q_entity"])
with open(f"{OUTD}/documents.jsonl","w",encoding="utf-8") as fo:
    for d in docs: fo.write(json.dumps(d,ensure_ascii=False)+"\n")
docsha=hashlib.sha256(open(f"{OUTD}/documents.jsonl","rb").read()).hexdigest()
json.dump({"dataset":"webqsp","doc_unit":"Freebase entity (RoG subgraph node)","n_docs":len(docs),
    "unique_ids":len(name2id),"unique_text_hashes":len({d['text_sha256'] for d in docs}),
    "documents_jsonl_sha256":docsha,"rog_questions_with_subgraph":len(rog_gold),
    "provenance":"structural_derived_from_freebase — RoG-webqsp subgraphs, NOT official Microsoft WebQSP graph",
    "old_substrate_docs":781000},open(f"{OUTD}/document_manifest.json","w"),indent=2)
print(f"webqsp C1: n_entities={len(docs)} rog_qs_with_subgraph={len(rog_gold)}")

# ---- C2: official questions 3098/1639 ----
def ans_names(parses):
    names=[]
    for p in parses or []:
        for a in p.get("Answers",[]):
            nm=a.get("EntityName") or a.get("AnswerArgument")
            if nm: names.append(nm)
    return list(dict.fromkeys(names))
def topic_names(parses):
    names=[]
    for p in parses or []:
        if p.get("TopicEntityName"): names.append(p["TopicEntityName"])
    return list(dict.fromkeys(names))

OFF={"train":"data/original/webqsp/WebQSP/data/WebQSP.train.json",
     "test":"data/original/webqsp/WebQSP/data/WebQSP.test.json"}
qsum={}; open(f"{OUTD}/queries.jsonl","w").close()
for split,fn in OFF.items():
    d=json.load(open(fn,encoding="utf-8"))["Questions"]
    recs=[]; goldmap=0; zero=0; parsed=0; in_rog=0; used_rog=0
    for i,q in enumerate(d):
        qid=q["QuestionId"]
        gnames=ans_names(q["Parses"])
        answerable=len(gnames)>0
        if answerable: parsed+=1
        # prefer RoG a_entity (guaranteed in entity set) when the id has a subgraph
        if qid in rog_gold and rog_gold[qid]:
            gnames_src=rog_gold[qid]; used_rog+=1
        else:
            gnames_src=gnames
        if qid in rog_gold: in_rog+=1
        gold=[name2id[n] for n in gnames_src if n in name2id]
        if gold: goldmap+=1
        if answerable and not gold: zero+=1
        topics=topic_names(q["Parses"]) or rog_topic.get(qid,[])
        recs.append({"dataset":"webqsp","query_id":qid,"official_split":split,
            "question":q.get("RawQuestion") or q.get("ProcessedQuestion"),"answers":gnames,
            "gold_doc_ids":gold,"gold_entity_names":gnames_src,"topic_entities":topics,
            "answerability":answerable,"parsing_eligible":(qid in rog_gold),
            "hop":None,"question_type":None,"original_index":i,"source_file":fn})
    with open(f"{OUTD}/queries.jsonl","a",encoding="utf-8") as fo:
        for r in recs: fo.write(json.dumps(r,ensure_ascii=False)+"\n")
    qsum[split]={"n":len(recs),"answerable":parsed,"gold_mapped":goldmap,
        "zero_gold_answerable":zero,"has_rog_subgraph":in_rog,"used_rog_gold":used_rog}
    print(f"webqsp {split}: n={len(recs)} answerable={parsed} gold_mapped={goldmap} "
          f"zero_gold_ans={zero} has_rog_subgraph={in_rog}")
qsha=hashlib.sha256(open(f"{OUTD}/queries.jsonl","rb").read()).hexdigest()
json.dump({"dataset":"webqsp","splits":qsum,"total":sum(v["n"] for v in qsum.values()),
    "queries_jsonl_sha256":qsha,"expected_official":{"train":3098,"test":1639},
    "rog_answerable":{"train":2826,"val":246,"test":1628},
    "gold_policy":"answer entity names -> entity doc ids; prefer RoG a_entity where subgraph exists",
    "note":"official MS questions are canonical; RoG subgraphs are structural_derived_from_freebase (not official graph)"},
    open(f"{OUTD}/query_manifest.json","w"),indent=2)
print("webqsp DONE")
