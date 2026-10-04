"""Phase C4 — NER 'shares-entity' edges (edge_family=ner) for text datasets (2wiki/musique/squad).
spaCy en_core_web_sm NER only; entity postings; keep entities with 2<=df<=25 (df-weighted 1/df to beat hubs);
connect all doc pairs sharing such an entity. Writes data/canonical/<ds>/graph_ner.tsv + ner_manifest.json.
KB datasets (metaqa/webqsp) skip NER (docs are bare entity names -> degenerate). Hotpot deferred (5.23M, native hyperlinks suffice)."""
import json, hashlib, os, sys, time
from collections import defaultdict
import spacy

DF_MIN, DF_MAX = 2, 25
def build(ds):
    t0=time.time(); OUTD=f"data/canonical/{ds}"
    nlp=spacy.load("en_core_web_sm", disable=["tagger","parser","lemmatizer","attribute_ruler"])
    ids=[]; texts=[]
    for line in open(f"{OUTD}/documents.jsonl",encoding="utf-8"):
        d=json.loads(line); ids.append(d["canonical_doc_id"]); texts.append(d["text"][:2000])
    print(f"{ds}: {len(ids)} docs, running NER...",flush=True)
    postings=defaultdict(set)  # entity_norm -> set(doc_idx)
    for i,doc in enumerate(nlp.pipe(texts, batch_size=256)):
        seen=set()
        for e in doc.ents:
            if e.label_ in ("CARDINAL","ORDINAL","PERCENT","QUANTITY","MONEY","DATE","TIME"): continue
            key=e.text.strip().lower()
            if len(key)<3 or key in seen: continue
            seen.add(key); postings[key].add(i)
        if (i+1)%50000==0: print(f"  {ds} {i+1}/{len(ids)}, {time.time()-t0:.0f}s",flush=True)
    OUT=f"{OUTD}/graph_ner.tsv"; seen_e=set(); n_edges=0; n_ent_used=0
    fout=open(OUT,"w",encoding="utf-8")
    for ent,dset in postings.items():
        df=len(dset)
        if df<DF_MIN or df>DF_MAX: continue
        n_ent_used+=1; w=1.0/df; dl=sorted(dset)
        for a in range(len(dl)):
            for b in range(a+1,len(dl)):
                s,t=ids[dl[a]],ids[dl[b]]
                key=(s,t)
                if key in seen_e: continue
                seen_e.add(key); n_edges+=1
                fout.write(f"{s}\t{t}\t{w:.4f}\n")
    fout.close()
    fsha=hashlib.sha256(open(OUT,"rb").read()).hexdigest()
    man={"dataset":ds,"edge_family":"ner","edge_subtype":"shares_entity",
         "provenance":"spaCy en_core_web_sm NER shared-entity, df-weighted 1/df",
         "df_min":DF_MIN,"df_max":DF_MAX,"n_entities_total":len(postings),"n_entities_used":n_ent_used,
         "n_edges":n_edges,"weighted":True,"edge_schema":"src_id\\tdst_id\\tweight(1/df)",
         "graph_ner_tsv_sha256":fsha,"directed":False,"secs":round(time.time()-t0,1)}
    json.dump(man,open(f"{OUTD}/ner_manifest.json","w"),indent=2)
    print(f"{ds} C4 NER: entities_used={n_ent_used} n_edges={n_edges} secs={man['secs']}",flush=True)

if __name__=="__main__":
    for ds in (sys.argv[1:] or ["squad","musique","2wiki"]): build(ds)
    print("DONE_NER")
