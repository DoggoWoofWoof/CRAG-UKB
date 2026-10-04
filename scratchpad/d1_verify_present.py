"""Verify the already-present official releases in data/original (metaqa, musique-ans, squad v2).
Counts, ID uniqueness, cross-split disjointness, answerable/unanswerable, SHA256. Read-only verify."""
import json, hashlib, os, glob

def sha256(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def wc(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f: n += 1
    return n

OUT = "results/data_audit"; os.makedirs(OUT, exist_ok=True)
rep = {}

# ---- METAQA ----
mq = {"kb_txt_triples": wc("data/original/metaqa/kb.txt"), "hops": {}}
for h in ("1","2","3"):
    d = {}
    for s in ("train","dev","test"):
        p = f"data/original/metaqa/{h}-hop/vanilla/qa_{s}.txt"
        d[s] = wc(p) if os.path.exists(p) else None
    mq["hops"][f"{h}hop"] = d
mq["expected"] = {"1hop":{"train":96106,"dev":9992,"test":9947},"2hop":{"train":118980,"dev":14872,"test":14872},"3hop":{"train":114196,"dev":14274,"test":14274}}
mq["match"] = all(mq["hops"][f"{h}hop"][s]==mq["expected"][f"{h}hop"][s] for h in ("1","2","3") for s in ("train","dev","test"))
mq["kb_sha256"] = sha256("data/original/metaqa/kb.txt")
rep["metaqa"] = mq

# ---- MUSIQUE-Ans ----
def jsonl_ids(path, key="id"):
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                ids.append(json.loads(line).get(key))
    return ids
mus = {}
tr = jsonl_ids("data/original/musique/v1.0/musique_ans_v1.0_train.jsonl")
dv = jsonl_ids("data/original/musique/v1.0/musique_ans_v1.0_dev.jsonl")
te = jsonl_ids("data/original/musique/v1.0/musique_ans_v1.0_test.jsonl")
mus["counts"] = {"train":len(tr),"dev":len(dv),"test":len(te)}
mus["expected"] = {"train":19938,"dev":2417,"test":2459}
mus["match"] = mus["counts"]==mus["expected"]
mus["dup_ids"] = {"train":len(tr)-len(set(tr)),"dev":len(dv)-len(set(dv)),"test":len(te)-len(set(te))}
sT,sD,sE=set(tr),set(dv),set(te)
mus["leakage"] = {"train_dev":len(sT&sD),"train_test":len(sT&sE),"dev_test":len(sD&sE)}
# hop derivable from id prefix like '2hop__...'
from collections import Counter
mus["train_hop_prefix"] = dict(Counter(str(i).split("__")[0] for i in tr).most_common())
rep["musique_ans"] = mus

# ---- SQUAD v2 ----
def squad_stats(path):
    d = json.load(open(path, encoding="utf-8"))
    ver = d.get("version"); nq=0; nimp=0; nans=0; ids=[]; arts=len(d["data"])
    for a in d["data"]:
        for p in a["paragraphs"]:
            for qa in p["qas"]:
                nq+=1; ids.append(qa["id"])
                if qa.get("is_impossible"): nimp+=1
                else: nans+=1
    return {"version":ver,"articles":arts,"questions":nq,"answerable":nans,"unanswerable":nimp,
            "dup_ids":len(ids)-len(set(ids)),"ids":ids}
sq={}
for split,p in (("train","data/original/squad/v2.0/train-v2.0.json"),("dev","data/original/squad/v2.0/dev-v2.0.json")):
    s=squad_stats(p); ids=s.pop("ids"); sq[split]=s; sq[split+"_idset"]=ids
tr_ids=set(sq.pop("train_idset")); dv_ids=set(sq.pop("dev_idset"))
sq["leakage_train_dev_ids"]=len(tr_ids&dv_ids)
sq["expected"]={"train":130319,"dev":11873}
sq["match"]=(sq["train"]["questions"]==130319 and sq["dev"]["questions"]==11873)
rep["squad_v2"]=sq

json.dump(rep, open(f"{OUT}/verify_present.json","w"), indent=2)
print("METAQA match:", mq["match"], mq["hops"])
print("MUSIQUE match:", mus["match"], mus["counts"], "leak:", mus["leakage"], "dup:", mus["dup_ids"])
print("  musique train hop prefixes:", mus["train_hop_prefix"])
print("SQUAD match:", sq["match"], {k:sq[k] for k in ("train","dev")}, "id-leak:", sq["leakage_train_dev_ids"])
print("-> verify_present.json")
