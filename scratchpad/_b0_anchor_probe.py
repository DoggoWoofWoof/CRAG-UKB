import os, re, json, sys, time
sys.path.insert(0,"scratchpad")
import numpy as np
import l2_relation as RL
_BRK=re.compile(r"\[(.+?)\]")
T0=time.time()
def probe(ds, n=400):
    M=RL.load_master(ds)
    qt=M["q_text"]
    # val query ids from query_ids_all if present, else all question nodes
    try:
        j=json.load(open(f"data/ukb_storage/{ds}/gte_qwen/query_ids_all.json"))
        si=j.get("split_indices",{}); ids=j["ids"]
        val_ids=[ids[i] for i in si.get("val", si.get("test", range(len(ids))))]
    except Exception:
        val_ids=list(qt.keys())
    val_ids=[q for q in val_ids if q in qt][:n]
    t2r=M["t2r"]
    nb=nt=nany=0
    for qid in val_ids:
        q=qt[qid]
        brk=[b for b in _BRK.findall(q) if b.strip()]
        br_hit=any((("metaqa_ent_"+b.strip().lower()) in M["docrow"]) or (b.strip() in t2r) for b in brk)
        tops=RL.find_topics(M,q)
        nb+=int(br_hit); nt+=int(len(tops)>0); nany+=int(br_hit or len(tops)>0)
    N=max(len(val_ids),1)
    return dict(ds=ds, n=len(val_ids), bracket=round(nb/N,3), title_match=round(nt/N,3), any_anchor=round(nany/N,3))
for ds in ["metaqa","2wiki_clean","musique_clean","squad_clean"]:
    try:
        print(json.dumps(probe(ds)), f"[{time.time()-T0:.0f}s]", flush=True)
    except Exception as e:
        print(ds,"ERR",repr(e)[:150], flush=True)
