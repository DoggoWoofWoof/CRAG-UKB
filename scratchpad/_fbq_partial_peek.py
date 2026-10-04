"""Read-only peek at the LOCAL original WebQSP.train.partial.json (train only; the test.partial file is NOT opened)."""
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "original", "webqsp", "WebQSP", "data")
j = json.load(open(os.path.join(D, "WebQSP.train.partial.json"), encoding="utf-8"))
qs = j["Questions"]
full = json.load(open(os.path.join(D, "WebQSP.train.json"), encoding="utf-8"))["Questions"]
full_ids = {q["QuestionId"] for q in full}
ids = [q["QuestionId"] for q in qs]
print("partial questions", len(qs), "unique ids", len(set(ids)), "ids also in full train", len(set(ids) & full_ids), "id prefixes", collections.Counter(i.split("-")[0] for i in ids))
print("first ids", ids[:8], "max Trn index", max(int(i.split("-")[1]) for i in ids))
c = collections.Counter()
for q in qs:
    ps = q.get("Parses") or []
    c["n_parses=%d" % min(len(ps), 3)] += 1
    for p in ps[:1]:
        c["topic_mid_present" if p.get("TopicEntityMid") else "topic_mid_missing"] += 1
        c["has_chain" if p.get("InferentialChain") else "no_chain"] += 1
        c["has_sparql" if p.get("Sparql") else "no_sparql"] += 1
        a = p.get("Answers") or []
        c["has_answers" if a else "no_answers"] += 1
        c["answer_entity_mid" if any(x.get("AnswerType") == "Entity" for x in a) else "no_entity_answer"] += 1
        c["quality=" + str((p.get("AnnotatorComment") or {}).get("ParseQuality"))] += 1
print(dict(c))
print(json.dumps(qs[0])[:1500])
print(json.dumps(qs[5])[:1500])
