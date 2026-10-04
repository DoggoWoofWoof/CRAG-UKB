"""Phase C4 — canonical NER edges reproducing the ESTABLISHED CRAG algorithm EXACTLY by calling
src.pipeline.ner_edges.build_ner_edges (spaCy en_core_web_sm, 9-label whitelist, Sigma 1/df over shared
entities, 2<=df<=25, full text). We only serialize its symmetric CSR to canonical tsv (src<dst, weight).
Text datasets only (2wiki/musique/squad); KB datasets (metaqa/webqsp) get ner_available=false (see c4_kb_mask).
Hotpot (5.23M) handled by a separate sharded builder. Writes data/canonical/<ds>/{graph_ner.tsv,ner_manifest.json}."""
import json, hashlib, os, sys, time
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.abspath("."))
from src.pipeline.ner_edges import build_ner_edges, ENT_LABELS

def build(ds):
    t0=time.time(); OUTD=f"data/canonical/{ds}"
    ids=[]; texts=[]
    for line in open(f"{OUTD}/documents.jsonl",encoding="utf-8"):
        d=json.loads(line); ids.append(d["canonical_doc_id"]); texts.append(d["text"])
    n=len(ids); print(f"{ds}: {n} docs -> established build_ner_edges ...",flush=True)
    A=build_ner_edges(None, texts, n, maxdf=25, weighted=True)  # dataset=None -> no cache read/write; fresh on canonical texts
    Au=sp.triu(A, k=1).tocoo()                                  # symmetric -> keep upper triangle once
    OUT=f"{OUTD}/graph_ner.tsv"
    with open(OUT,"w",encoding="utf-8") as f:
        for a,b,v in zip(Au.row, Au.col, Au.data):
            f.write(f"{ids[a]}\t{ids[b]}\t{v:.6f}\n")
    fsha=hashlib.sha256(open(OUT,"rb").read()).hexdigest()
    man={"dataset":ds,"edge_family":"ner","edge_subtype":"shares_entity","ner_available":True,
         "provenance":"spaCy en_core_web_sm NER over canonical documents.jsonl; shared named-entity links, df-weighted 1/df (established CRAG NER, no LLM)",
         "algorithm":"src/pipeline/ner_edges.build_ner_edges (established CRAG NER, reproduced exactly)",
         "ner_model":"spacy en_core_web_sm (disable parser,lemmatizer)","ent_labels":sorted(ENT_LABELS),
         "df_min":2,"df_max":25,"weight":"sum 1/df over shared entities","min_ent_char":">2 (len(e.text)>2)",
         "entity_norm":"e.text.lower().strip()","full_text":True,"n_edges":int(Au.nnz),
         "edge_schema":"src_id\\tdst_id\\tweight","graph_ner_tsv_sha256":fsha,"directed":False,
         "secs":round(time.time()-t0,1)}
    json.dump(man,open(f"{OUTD}/ner_manifest.json","w"),indent=2)
    print(f"{ds} C4 NER (established): n_edges={int(Au.nnz)} secs={man['secs']}",flush=True)

if __name__=="__main__":
    for ds in (sys.argv[1:] or ["squad","musique","2wiki"]): build(ds)
    print("DONE_NER_CANONICAL")
