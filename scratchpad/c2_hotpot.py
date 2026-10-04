"""Phase C2 — HotpotQA FULLWIKI queries. train=distractor train (shared questions), dev=fullwiki val,
test=fullwiki test (gold hidden). Gold = supporting_facts titles -> canonical doc ids (title index over 5.23M).
Writes data/canonical/hotpotqa/{queries.jsonl,query_manifest.json}."""
import json, hashlib, os, glob, time
import pandas as pd
OUTD="data/canonical/hotpotqa"
t0=time.time()

# title -> canonical_doc_id index over 5.23M abstracts
title2id={}
for line in open(f"{OUTD}/documents.jsonl",encoding="utf-8"):
    d=json.loads(line); title2id[d["title"]]=d["canonical_doc_id"]
print(f"title index: {len(title2id)} titles, {time.time()-t0:.0f}s")

def aslist(x):
    if x is None: return []
    try: return list(x)
    except TypeError: return [x]

SRC={"train":sorted(glob.glob("data/original/hotpotqa/hf_distractor/train-*.parquet")),
     "dev":["data/original/hotpotqa/hf_fullwiki/validation-00000-of-00001.parquet"],
     "test":["data/original/hotpotqa/hf_fullwiki/test-00000-of-00001.parquet"]}
QSRC={"train":"hotpot_train_v1.1","dev":"hotpot_dev_fullwiki_v1.1","test":"hotpot_test_fullwiki_v1.1"}
GOLD_AVAIL={"train":True,"dev":True,"test":False}
qsum={}; open(f"{OUTD}/queries.jsonl","w").close()
for split,files in SRC.items():
    recs=[]; goldmap=0; allmap=0; partial=0; zero=0; idx=0; ga=GOLD_AVAIL[split]
    for f in files:
        df=pd.read_parquet(f)
        for _,r in df.iterrows():
            sf=r["supporting_facts"]; titles=list(dict.fromkeys(aslist(sf["title"]))) if isinstance(sf,dict) else []
            gold=[title2id[t] for t in titles if t in title2id]
            if ga:  # only meaningful where gold is visible
                if gold: goldmap+=1
                if titles and len(gold)==len(titles): allmap+=1
                elif gold: partial+=1
                if titles and not gold: zero+=1
            ans=r.get("answer"); ans=[ans] if (ans and str(ans) not in ("None","")) else []
            recs.append({"dataset":"hotpotqa","query_id":r["id"],"official_split":split,
                "question_source":QSRC[split],"retrieval_setting":"fullwiki",
                "question":r["question"],"answers":ans,"gold_doc_ids":gold,"gold_titles":titles,
                "gold_available":ga,"question_type":r.get("type"),"level":r.get("level"),"hop":None,
                "answerability":None,"original_index":idx,"source_file":f}); idx+=1
    with open(f"{OUTD}/queries.jsonl","a",encoding="utf-8") as fo:
        for rr in recs: fo.write(json.dumps(rr,ensure_ascii=False)+"\n")
    qsum[split]={"n":len(recs),"question_source":QSRC[split],"gold_available":ga,
        "all_sf_titles_mapped":allmap,"partial_mapped":partial,"zero_mapped":zero,
        "title_norm_failures_not_dropped":zero}
    print(f"hotpot {split}: n={len(recs)} gold_available={ga} all_mapped={allmap} partial={partial} zero={zero}")
qsha=hashlib.sha256(open(f"{OUTD}/queries.jsonl","rb").read()).hexdigest()
json.dump({"dataset":"hotpotqa","question_source":"hotpot_train_v1.1 (official Hotpot QA)","retrieval_setting":"fullwiki",
    "splits":qsum,"total":sum(v["n"] for v in qsum.values()),
    "queries_jsonl_sha256":qsha,"expected":{"train":90447,"dev":7405,"test":7405},
    "gold_coverage_report":{k:{"n":qsum[k]["n"],"all_sf_titles_mapped":qsum[k]["all_sf_titles_mapped"],
        "partial":qsum[k]["partial_mapped"],"zero":qsum[k]["zero_mapped"]} for k in qsum},
    "gold_policy":"supporting_facts titles -> canonical FullWiki doc ids; title-norm failures NOT silently dropped (counted as zero)",
    "test_policy":"hidden gold (gold_available=false); LOCKED final eval — no Shapley/regime/design/checkpoint use",
    "note":"official Hotpot train questions (hotpot_train_v1.1); retrieval corpus/setting=FullWiki (NOT distractor)"},
    open(f"{OUTD}/query_manifest.json","w"),indent=2)
print(f"hotpot C2 DONE, {time.time()-t0:.0f}s")
