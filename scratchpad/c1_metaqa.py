"""Phase C1 — canonical document manifest for MetaQA.
Doc unit = official KB entity (from kb_entity_dict.txt, corroborated by kb.txt). Deterministic IDs,
text = entity name (pure official representation; verbalization is a downstream encoding choice).
Writes data/canonical/metaqa/documents.jsonl + document_manifest.json. Read-only over official raw."""
import json, hashlib, os
from collections import Counter

SRC_DICT="data/original/metaqa/entity/kb_entity_dict.txt"
SRC_KB="data/original/metaqa/kb.txt"
OUTD="data/canonical/metaqa"; os.makedirs(OUTD, exist_ok=True)

def sha(t): return hashlib.sha256(t.encode("utf-8")).hexdigest()

# entities in kb.txt (to confirm dict covers the KB)
kb_ents=set()
for line in open(SRC_KB, encoding="utf-8"):
    p=line.rstrip("\n").split("|")
    if len(p)==3: kb_ents.add(p[0]); kb_ents.add(p[2])

docs=[]; ids=set(); texts=Counter(); titles=Counter(); empty=0
for line in open(SRC_DICT, encoding="utf-8"):
    line=line.rstrip("\n")
    if not line: continue
    idx, name = line.split("\t", 1)
    did=f"metaqa_ent_{idx}"
    text=name
    if not text.strip(): empty+=1
    d={"dataset":"metaqa","canonical_doc_id":did,"original_source_id":idx,"title":name,"text":text,
       "text_sha256":sha(text),"source_file":"data/original/metaqa/entity/kb_entity_dict.txt",
       "source_index":int(idx),"provenance":"official kb_entity_dict entity node"}
    docs.append(d); ids.add(did); texts[text]+=1; titles[name]+=1

# verification
dict_names=set(titles)
missing_from_dict=len(kb_ents - dict_names)     # KB entities not in dict (should be 0)
name2ids={}
for d in docs: name2ids.setdefault(d["title"],[]).append(d["canonical_doc_id"])
dup_titles={k:v for k,v in titles.items() if v>1}
uniq_hashes=len({d["text_sha256"] for d in docs})

with open(f"{OUTD}/documents.jsonl","w",encoding="utf-8") as f:
    for d in docs: f.write(json.dumps(d, ensure_ascii=False)+"\n")

# freeze hash = sha256 of the documents.jsonl
docs_file_sha=hashlib.sha256(open(f"{OUTD}/documents.jsonl","rb").read()).hexdigest()
man={"dataset":"metaqa","doc_unit":"KB entity (name)","n_docs":len(docs),
     "unique_ids":len(ids),"unique_text_hashes":uniq_hashes,"duplicate_texts":sum(v-1 for v in texts.values() if v>1),
     "empty_docs":empty,"title_collisions":len(dup_titles),"title_collision_examples":dict(list(dup_titles.items())[:5]),
     "kb_entities":len(kb_ents),"kb_entities_missing_from_dict":missing_from_dict,
     "old_substrate_docs":40151,"delta_vs_old":len(docs)-40151,
     "source_files":[SRC_DICT,SRC_KB],"documents_jsonl_sha256":docs_file_sha,
     "text_policy":"text = entity name (official); verbalization deferred to encoding stage",
     "reuse_old_embeddings":False,"reuse_reason":"doc-set differs from old (43234 vs 40151) -> re-embed"}
json.dump(man, open(f"{OUTD}/document_manifest.json","w"), indent=2)
print(json.dumps({k:man[k] for k in ("n_docs","unique_ids","unique_text_hashes","duplicate_texts","empty_docs",
      "title_collisions","kb_entities","kb_entities_missing_from_dict","delta_vs_old","documents_jsonl_sha256")}, indent=2))
print("-> data/canonical/metaqa/{documents.jsonl,document_manifest.json}")
