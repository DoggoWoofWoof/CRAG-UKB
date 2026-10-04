"""Read-only: join the 786-row WebQSP KB-L3 development population (results/L3_DEV/l3w_population_webqsp__v1.json, query ids WebQTrn-*) to the NSM release by id and count how many
map (topic entities + answers) onto FREEBASE canonical positions.   PYTHONHASHSEED=0 python scratchpad/_fbq_nsm_join786.py
Reads NSM train_simple/dev_simple only (WebQTrn ids); no TEST row, no sealed split-B id is looked up."""
import json, os
import numpy as np
assert os.environ.get("PYTHONHASHSEED") == "0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSM = os.path.join(ROOT, "data", "final_canonical", "webqsp", "_acquisition", "nsm", "extracted", "webqsp", "webqsp")
FB = os.path.join(ROOT, "data", "final_canonical", "freebase")
pop = json.load(open(os.path.join(ROOT, "results", "L3_DEV", "l3w_population_webqsp__v1.json")))
want = set(pop["query_ids"])
ents = [l.rstrip("\n") for l in open(os.path.join(NSM, "entities.txt"), encoding="utf-8")]
uid = np.load(os.path.join(FB, "nodes", "node_uid.npy"), mmap_mode="r")
cache = {}
def pos(m):
    if m not in cache:
        u = hash(m.encode("utf-8")); i = int(np.searchsorted(uid, u))
        cache[m] = i if i < len(uid) and int(uid[i]) == u else -1
    return cache[m]
found = {}
for fn in ("train_simple.json", "dev_simple.json"):
    for l in open(os.path.join(NSM, fn), encoding="utf-8"):
        i0 = l.find('"id": "'); qid = l[i0 + 7:l.find('"', i0 + 7)]
        if qid in want and qid not in found:
            d = json.loads(l)
            tm = [ents[i] for i in d["entities"]]
            am = [a["kb_id"] for a in d["answers"] if isinstance(a["kb_id"], str) and a["kb_id"][:2] in ("m.", "g.")]
            nonmid = len(d["answers"]) - len(am)
            found[qid] = (d["question"], [pos(m) for m in tm], [pos(m) for m in am], nonmid)
n = len(want); f = len(found)
tp_all = sum(1 for v in found.values() if v[1] and all(p >= 0 for p in v[1]))
an_all = sum(1 for v in found.values() if v[2] and all(p >= 0 for p in v[2]) and v[3] == 0)
an_any = sum(1 for v in found.values() if any(p >= 0 for p in v[2]))
both = sum(1 for v in found.values() if v[1] and all(p >= 0 for p in v[1]) and v[2] and all(p >= 0 for p in v[2]) and v[3] == 0)
na = sum(len(v[2]) for v in found.values()); nah = sum(sum(p >= 0 for p in v[2]) for v in found.values())
nt = sum(len(v[1]) for v in found.values()); nth = sum(sum(p >= 0 for p in v[1]) for v in found.values())
print(json.dumps({"population_rows": n, "joined_to_NSM_by_id": f, "topic_entities": nt, "topic_resolved": nth, "answer_mids": na, "answer_mids_resolved": nah,
                  "questions_all_topic_resolved": tp_all, "questions_all_answers_resolved(no literal answers)": an_all, "questions_any_answer_resolved": an_any,
                  "questions_fully_mappable": both}, indent=1))
