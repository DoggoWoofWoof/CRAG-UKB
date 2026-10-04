"""Phase C3 — 2Wiki structural_native graph from official para_with_hyperlink.jsonl (7GB, streamed from zip).
Keep only edges internal to our canonical 2wiki corpus (both endpoints are corpus titles).
Edge = (src_id, dst_id, 'hyperlink'). Writes data/canonical/2wiki/graph_structural.tsv + graph_manifest.json."""
import json, hashlib, os, zipfile, time
ZIP="data/original/2wiki/v1.0_ids_april2021/para_with_hyperlink.zip"
OUTD="data/canonical/2wiki"; OUT=f"{OUTD}/graph_structural.tsv"

# corpus title -> id
title2id={}
for line in open(f"{OUTD}/documents.jsonl",encoding="utf-8"):
    d=json.loads(line); title2id[d["title"]]=d["canonical_doc_id"]
print(f"corpus titles: {len(title2id)}")

t0=time.time(); n_rec=0; n_src_in=0; n_edges=0; seen=set()
z=zipfile.ZipFile(ZIP)
fout=open(OUT,"w",encoding="utf-8")
with z.open("para_with_hyperlink.jsonl") as f:
    for raw in f:
        n_rec+=1
        try: r=json.loads(raw)
        except Exception: continue
        src=title2id.get(r.get("title"))
        if src is None:
            if n_rec % 1000000==0: print(f"  ...{n_rec} recs, {n_edges} edges, {time.time()-t0:.0f}s",flush=True)
            continue
        n_src_in+=1
        for m in r.get("mentions",[]):
            ref=m.get("ref_url")
            if not ref: continue
            tgt=title2id.get(ref.replace("_"," "))
            if tgt is None or tgt==src: continue
            key=(src,tgt)
            if key in seen: continue
            seen.add(key); n_edges+=1
            fout.write(f"{src}\t{tgt}\thyperlink\n")
        if n_rec % 1000000==0: print(f"  ...{n_rec} recs, {n_edges} edges, {time.time()-t0:.0f}s",flush=True)
fout.close()
fsha=hashlib.sha256(open(OUT,"rb").read()).hexdigest()
man={"dataset":"2wiki","edge_family":"structural_native",
     "provenance":"official 2wiki para_with_hyperlink.jsonl Wikipedia hyperlinks (internal to corpus)",
     "source_files":[ZIP],"n_edges":n_edges,"n_relations":1,"edge_subtype":"hyperlink",
     "corpus_titles":len(title2id),"src_records_in_corpus":n_src_in,"total_records_scanned":n_rec,
     "edge_schema":"src_id\\tdst_id\\thyperlink","graph_tsv_sha256":fsha,"directed":True,"secs":round(time.time()-t0,1)}
json.dump(man,open(f"{OUTD}/graph_manifest.json","w"),indent=2)
print(f"2wiki C3: n_edges={n_edges} src_in_corpus={n_src_in}/{n_rec} secs={man['secs']}")
print("DONE_2WIKI_GRAPH")
