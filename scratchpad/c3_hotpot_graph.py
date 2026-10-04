"""Phase C3 — HotpotQA fullwiki structural_native graph from official abstracts hyperlinks (text_with_links).
Streams tar->bz2->json; edge (src_curid -> tgt_curid) where tgt href-title resolves to a corpus title.
Memory-safe: title->int32 index; unique edges via packed int64 set. Writes graph_structural.tsv + graph_manifest.json."""
import tarfile, bz2, json, re, os, time, hashlib
from urllib.parse import unquote
TB="data/original/hotpotqa/fullwiki_corpus/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
OUTD="data/canonical/hotpotqa"; OUT=f"{OUTD}/graph_structural.tsv"
AHREF=re.compile(r'<a href="([^"]*)">')

t0=time.time()
# title -> (int idx, curid) from canonical docs
title2int={}; int2curid=[]
for line in open(f"{OUTD}/documents.jsonl",encoding="utf-8"):
    d=json.loads(line)
    title2int[d["title"]]=len(int2curid); int2curid.append(d["original_source_id"])
N=len(int2curid); print(f"title index {N}, {time.time()-t0:.0f}s",flush=True)

def flat(t):
    if isinstance(t,str):
        for m in AHREF.finditer(t): yield m.group(1)
    elif isinstance(t,list):
        for x in t: yield from flat(x)

seen=set(); n_edges=0; n_art=0; drop=0
fout=open(OUT,"w",encoding="utf-8"); tar=tarfile.open(TB,"r:bz2"); nm=0
for member in tar:
    if not member.isfile(): continue
    nm+=1
    try: text=bz2.decompress(tar.extractfile(member).read()).decode("utf-8")
    except Exception: continue
    for line in text.splitlines():
        line=line.strip()
        if not line: continue
        r=json.loads(line); n_art+=1
        s=title2int.get(r.get("title"))
        if s is None: continue
        for href in flat(r.get("text_with_links") or []):
            tt=unquote(href).replace("_"," ").strip()
            t=title2int.get(tt)
            if t is None or t==s: continue
            key=(s<<23)^t if N< (1<<23) else s*N+t
            key=s*N+t
            if key in seen: continue
            seen.add(key); n_edges+=1
            fout.write(f"{int2curid[s]}\t{int2curid[t]}\thyperlink\n")
    if nm % 5000==0: print(f"  ...{nm} members, {n_art} arts, {n_edges} edges, {time.time()-t0:.0f}s",flush=True)
fout.close()
fsha=hashlib.sha256(open(OUT,"rb").read()).hexdigest()
man={"dataset":"hotpotqa","edge_family":"structural_native","edge_subtype":"hyperlink",
     "provenance":"official enwiki-20171001 withlinks-abstracts intra-corpus hyperlinks",
     "source_files":[TB],"n_edges":n_edges,"n_relations":1,"corpus_titles":N,
     "edge_schema":"src_curid\\tdst_curid\\thyperlink","graph_tsv_sha256":fsha,"directed":True,
     "secs":round(time.time()-t0,1)}
json.dump(man,open(f"{OUTD}/graph_manifest.json","w"),indent=2)
print(f"hotpot C3: n_edges={n_edges} secs={man['secs']}"); print("DONE_HOTPOT_GRAPH")
