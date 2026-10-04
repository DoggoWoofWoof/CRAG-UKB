"""Phase C3 — structural_derived (title-mention) graphs for MuSiQue + SQuAD (no native structure).
Edge doc_A -> doc_B iff A's text contains B's title as a phrase. Efficiency: index each title by its
most-selective token; verify full title (word-boundary, case-insensitive) at trigger positions.
Writes data/canonical/<ds>/graph_structural.tsv + graph_manifest.json."""
import json, hashlib, os, re, time
from collections import defaultdict

STOP=set("the a an of and or to in on at for with by from as is are was were be this that these those "
         "he she it they we you i his her its their our your".split())
WORD=re.compile(r"[A-Za-z0-9]+")

def build(ds, min_title_len=4):
    t0=time.time(); OUTD=f"data/canonical/{ds}"
    docs=[];
    for line in open(f"{OUTD}/documents.jsonl",encoding="utf-8"):
        d=json.loads(line); docs.append((d["canonical_doc_id"],d["title"],d["text"]))
    # index: selective token (longest, non-stop) -> [(title_lower, docid, title_wordset)]
    tokidx=defaultdict(list); title_pat={}
    for did,title,_ in docs:
        tl=title.strip()
        if len(tl)<min_title_len: continue
        toks=[w.lower() for w in WORD.findall(tl)]
        cand=[w for w in toks if w not in STOP and len(w)>2] or toks
        if not cand: continue
        key=max(cand,key=len)
        tokidx[key].append((tl.lower(), did))
        title_pat[did]=re.compile(r"\b"+re.escape(tl.lower())+r"\b")
    print(f"{ds}: {len(docs)} docs, {len(tokidx)} index tokens, {time.time()-t0:.0f}s",flush=True)
    OUT=f"{OUTD}/graph_structural.tsv"; seen=set(); n_edges=0
    fout=open(OUT,"w",encoding="utf-8")
    for i,(did,_,text) in enumerate(docs):
        tl=text.lower(); toks=set(WORD.findall(tl))
        emitted=set()
        for tok in toks:
            for (cand_l,cand_did) in tokidx.get(tok,()):
                if cand_did==did or cand_did in emitted: continue
                if title_pat[cand_did].search(tl):
                    emitted.add(cand_did)
                    key=(did,cand_did)
                    if key in seen: continue
                    seen.add(key); n_edges+=1
                    fout.write(f"{did}\t{cand_did}\ttitle_mention\n")
        if (i+1)%20000==0: print(f"  {ds} {i+1}/{len(docs)} docs, {n_edges} edges, {time.time()-t0:.0f}s",flush=True)
    fout.close()
    fsha=hashlib.sha256(open(OUT,"rb").read()).hexdigest()
    man={"dataset":ds,"edge_family":"structural_derived","edge_subtype":"title_mention",
         "provenance":"derived: doc text contains another doc's title as a word-bounded phrase (no native graph available)",
         "source_files":[f"{OUTD}/documents.jsonl"],"n_edges":n_edges,"n_relations":1,
         "edge_schema":"src_id\\tdst_id\\ttitle_mention","graph_tsv_sha256":fsha,"directed":True,
         "min_title_len":min_title_len,"secs":round(time.time()-t0,1)}
    json.dump(man,open(f"{OUTD}/graph_manifest.json","w"),indent=2)
    print(f"{ds} C3 derived: n_edges={n_edges} secs={man['secs']}",flush=True)

if __name__=="__main__":
    build("musique"); build("squad"); print("DONE_DERIVED")
