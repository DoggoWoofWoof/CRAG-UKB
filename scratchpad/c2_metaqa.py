"""Phase C2 — canonical query manifests for MetaQA, PER HOP (1hop/2hop/3hop), uncapped.
Gold = answer entities mapped to canonical doc IDs; topic entity (in [brackets]) recorded too.
Writes data/canonical/metaqa_{h}hop/queries.jsonl + query_manifest.json. Read-only over official raw."""
import json, hashlib, os, re

DOCS="data/canonical/metaqa/documents.jsonl"
name2id={}
for line in open(DOCS, encoding="utf-8"):
    d=json.loads(line); name2id[d["title"]]=d["canonical_doc_id"]

TOPIC=re.compile(r"\[([^\]]+)\]")
def qid(split,hop,idx,q): return "mq_"+hashlib.sha256(f"metaqa|{hop}|{split}|{idx}|{q}".encode()).hexdigest()[:16]

summary={}
for hop in ("1","2","3"):
    for split in ("train","dev","test"):
        qf=f"data/original/metaqa/{hop}-hop/vanilla/qa_{split}.txt"
        tf=f"data/original/metaqa/{hop}-hop/qa_{split}_qtype.txt"
        qtypes=[l.rstrip("\n") for l in open(tf,encoding="utf-8")] if os.path.exists(tf) else None
        outd=f"data/canonical/metaqa_{hop}hop"; os.makedirs(outd, exist_ok=True)
        recs=[]; n_goldmap=0; n_allmap=0; n_zero=0; n_topic_map=0
        for i,line in enumerate(open(qf,encoding="utf-8")):
            line=line.rstrip("\n")
            if not line: continue
            q,ans = line.split("\t",1)
            answers=[a for a in ans.split("|") if a]
            gold_ids=[name2id[a] for a in answers if a in name2id]
            topic_m=TOPIC.search(q); topic=topic_m.group(1) if topic_m else None
            topic_id=name2id.get(topic.title()) or name2id.get(topic) if topic else None
            if topic_id: n_topic_map+=1
            if gold_ids: n_goldmap+=1
            if gold_ids and len(gold_ids)==len(answers): n_allmap+=1
            if not gold_ids: n_zero+=1
            recs.append({"dataset":"metaqa","hop":f"{hop}hop","query_id":qid(split,hop,i,q),
                "official_split":split,"question":q,"answers":answers,"gold_doc_ids":gold_ids,
                "topic_entity":topic,"topic_doc_id":topic_id,
                "question_type":qtypes[i] if qtypes and i<len(qtypes) else None,
                "original_index":i,"source_file":qf})
        with open(f"{outd}/queries.jsonl","a",encoding="utf-8") as f:  # append per split
            for r in recs: f.write(json.dumps(r,ensure_ascii=False)+"\n")
        summary[f"{hop}hop/{split}"]={"n":len(recs),"gold_mapped":n_goldmap,"all_gold_mapped":n_allmap,
            "zero_gold":n_zero,"topic_mapped":n_topic_map}
        print(f"metaqa {hop}hop {split}: n={len(recs)} gold_mapped={n_goldmap} zero_gold={n_zero} topic_mapped={n_topic_map}")

# freeze per-hop manifests (all splits concatenated in queries.jsonl)
for hop in ("1","2","3"):
    outd=f"data/canonical/metaqa_{hop}hop"
    fsha=hashlib.sha256(open(f"{outd}/queries.jsonl","rb").read()).hexdigest()
    hs={k:v for k,v in summary.items() if k.startswith(f"{hop}hop")}
    tot=sum(v["n"] for v in hs.values())
    man={"dataset":f"metaqa_{hop}hop","splits":hs,"total_queries":tot,
         "queries_jsonl_sha256":fsha,"gold_policy":"answer entities -> canonical doc ids",
         "expected":{"1":{"train":96106,"dev":9992,"test":9947},"2":{"train":118980,"dev":14872,"test":14872},
                     "3":{"train":114196,"dev":14274,"test":14274}}[hop]}
    json.dump(man, open(f"{outd}/query_manifest.json","w"), indent=2)
print("\n-> data/canonical/metaqa_{1,2,3}hop/{queries.jsonl,query_manifest.json}")
