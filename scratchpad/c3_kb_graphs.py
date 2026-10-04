"""Phase C3 — structural graphs for the KB datasets.
 MetaQA : structural_native from kb.txt triples (h|r|t).
 WebQSP : structural_derived_from_freebase from RoG subgraph triples (NOT official MS graph).
Edges stored compact: src_id \t dst_id \t relation (edge_family+provenance uniform per file, in manifest).
Writes data/canonical/<ds>/graph_structural.tsv + graph_manifest.json."""
import json, hashlib, os, glob
import pandas as pd

def load_name2id(ds):
    m={}
    for line in open(f"data/canonical/{ds}/documents.jsonl",encoding="utf-8"):
        d=json.loads(line); m[d["title"]]=d["canonical_doc_id"]
    return m

def freeze(ds, edges_iter, family, provenance, src_files):
    outd=f"data/canonical/{ds}"; out=f"{outd}/graph_structural.tsv"
    seen=set(); n=0; dropped=0; rels=set()
    with open(out,"w",encoding="utf-8") as f:
        for s,r,t in edges_iter:
            if s is None or t is None: dropped+=1; continue
            key=(s,t,r)
            if key in seen: continue
            seen.add(key); rels.add(r); n+=1
            f.write(f"{s}\t{t}\t{r}\n")
    fsha=hashlib.sha256(open(out,"rb").read()).hexdigest()
    man={"dataset":ds,"edge_family":family,"provenance":provenance,"source_files":src_files,
         "n_edges":n,"n_relations":len(rels),"unmapped_endpoint_edges_dropped":dropped,
         "edge_schema":"src_id\\tdst_id\\trelation","graph_tsv_sha256":fsha,"directed":True}
    json.dump(man,open(f"{outd}/graph_manifest.json","w"),indent=2)
    print(f"{ds} C3: n_edges={n} n_relations={len(rels)} unmapped_dropped={dropped}")

# ---- MetaQA ----
n2i=load_name2id("metaqa")
def metaqa_edges():
    for line in open("data/original/metaqa/kb.txt",encoding="utf-8"):
        p=line.rstrip("\n").split("|")
        if len(p)!=3: continue
        yield (n2i.get(p[0]), p[1], n2i.get(p[2]))
freeze("metaqa", metaqa_edges(), "structural_native",
       "official MetaQA kb.txt entity-relation triples", ["data/original/metaqa/kb.txt"])

# ---- WebQSP ----
w2i=load_name2id("webqsp")
def webqsp_edges():
    for f in sorted(glob.glob("data/original/webqsp/rog_webqsp/*.parquet")):
        df=pd.read_parquet(f, columns=["graph"])
        for g in df["graph"]:
            for tr in list(g):
                yield (w2i.get(tr[0]), tr[1], w2i.get(tr[2]))
freeze("webqsp", webqsp_edges(), "structural_derived_from_freebase",
       "RoG-webqsp per-question Freebase subgraph triples; NOT official Microsoft WebQSP graph",
       sorted(glob.glob("data/original/webqsp/rog_webqsp/*.parquet")))
print("KB graphs DONE")
