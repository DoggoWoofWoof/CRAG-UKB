import os, re, json, sys, time
sys.path.insert(0,"scratchpad")
import l2_relation as RL
_BRK=re.compile(r"\[(.+?)\]")
T0=time.time()
def probe(ds, n=400):
    M=RL.load_master(ds); qt=M["q_text"]; t2r=M["t2r"]
    try:
        j=json.load(open(f"data/ukb_storage/{ds}/gte_qwen/query_ids_all.json"))
        si=j.get("split_indices",{}); ids=j["ids"]
        val_ids=[ids[i] for i in si.get("val", si.get("test", range(len(ids))))]
    except Exception:
        val_ids=list(qt.keys())
    val_ids=[q for q in val_ids if q in qt][:n]
    nt=nb=0
    for qid in val_ids:
        q=qt[qid]
        brk=[b for b in _BRK.findall(q) if b.strip()]
        nb+=int(any(b.strip() in t2r for b in brk))
        nt+=int(len(RL.find_topics(M,q))>0)
    N=max(len(val_ids),1)
    return dict(ds=ds,n=len(val_ids),bracket=round(nb/N,3),title_match=round(nt/N,3))
ds=sys.argv[1]
print(json.dumps(probe(ds)), f"[{time.time()-T0:.0f}s]", flush=True)
