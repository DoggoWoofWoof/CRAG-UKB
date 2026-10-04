"""Read-only probe: can WebQSP questions be mapped onto the FREEBASE canonical positions through the NSM release's MIDs?
  PYTHONHASHSEED=0 python scratchpad/_fbq_nsm_probe.py [split]     (split = dev_simple | train_simple | test_simple; default dev_simple)
Chain: question -> NSM `entities` (indices into entities.txt = topic-entity MIDs) / `answers[].kb_id` (answer MIDs) -> uid = hash(mid utf-8) -> searchsorted(node_uid) -> position, kind.
Only counts; no held-out population is read (dev_simple only by default)."""
import json
import os
import sys

import numpy as np

assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NSM = os.path.join(ROOT, "data", "final_canonical", "webqsp", "_acquisition", "nsm", "extracted", "webqsp", "webqsp")
FB = os.path.join(ROOT, "data", "final_canonical", "freebase")
KINDS = ["ENTITY_MID", "CVT_MEDIATOR", "SCHEMA_TYPE", "SCHEMA_PROPERTY", "SCHEMA_OTHER", "EXTERNAL_URI", "LITERAL"]


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "dev_simple"
    ents = [l.rstrip("\n") for l in open(os.path.join(NSM, "entities.txt"), encoding="utf-8")]
    uid = np.load(os.path.join(FB, "nodes", "node_uid.npy"), mmap_mode="r")
    kind = np.load(os.path.join(FB, "nodes", "kind.npy"), mmap_mode="r")
    qs = [json.loads(l) for l in open(os.path.join(NSM, split + ".json"), encoding="utf-8")]
    cache = {}

    def pos(mid):
        if mid in cache:
            return cache[mid]
        u = hash(mid.encode("utf-8"))
        i = int(np.searchsorted(uid, u))
        r = i if i < len(uid) and int(uid[i]) == u else -1
        cache[mid] = r
        return r

    n = len(qs)
    topic_all = topic_any = ans_all = ans_any = both = 0
    topic_n = topic_hit = ans_n = ans_hit = ans_nonmid = 0
    kinds_topic, kinds_ans = {}, {}
    examples = []
    for q in qs:
        tm = [ents[i] for i in q["entities"]]
        am = []
        for a in q["answers"]:
            k = a["kb_id"]
            if isinstance(k, str) and k[:2] in ("m.", "g."):
                am.append(k)
            else:
                ans_nonmid += 1
        tp = [pos(m) for m in tm]
        ap = [pos(m) for m in am]
        topic_n += len(tp)
        topic_hit += sum(p >= 0 for p in tp)
        ans_n += len(ap)
        ans_hit += sum(p >= 0 for p in ap)
        for p in tp:
            if p >= 0:
                kinds_topic[KINDS[int(kind[p])]] = kinds_topic.get(KINDS[int(kind[p])], 0) + 1
        for p in ap:
            if p >= 0:
                kinds_ans[KINDS[int(kind[p])]] = kinds_ans.get(KINDS[int(kind[p])], 0) + 1
        ta = bool(tp) and all(p >= 0 for p in tp)
        aa = bool(ap) and all(p >= 0 for p in ap)
        topic_all += ta
        topic_any += any(p >= 0 for p in tp)
        ans_all += aa
        ans_any += any(p >= 0 for p in ap)
        both += ta and aa
        if len(examples) < 3:
            examples.append({"id": q["id"], "q": q["question"], "topic": list(zip(tm, tp)), "answers": list(zip(am, ap))[:3]})
    rec = {"split": split, "questions": n, "topic_entities": topic_n, "topic_resolved": topic_hit, "answers": ans_n, "answers_resolved": ans_hit,
           "answers_non_mid_values": ans_nonmid, "questions_all_topic_resolved": topic_all, "questions_any_topic_resolved": topic_any,
           "questions_all_answers_resolved": ans_all, "questions_any_answer_resolved": ans_any, "questions_topic_and_answers_all_resolved": both,
           "kind_of_resolved_topic": kinds_topic, "kind_of_resolved_answers": kinds_ans, "examples": examples}
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
