"""Supplementary check (read-only on FBQ_SET__v1.parquet): is the CWQ id prefix really the WebQSP parent?  Compares the observed topic-MID / answer overlap
between a CWQ row and its id-parent against a random-parent baseline (parents permuted among the same CWQ rows, seed 0).
Writes results/FREEBASE_SCALE/FBQ_SET__v1__PARENT_CHECK.json (write-once).  No test file, no split-B id is touched.
  python scratchpad/_fbq_parent_check.py"""
import collections
import hashlib
import json
import os
import time

import numpy as np
import psutil
import pyarrow.parquet as pq

psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PQ = os.path.join(ROOT, "results", "FREEBASE_SCALE", "FBQ_SET__v1.parquet")
OUT = os.path.join(ROOT, "results", "FREEBASE_SCALE", "FBQ_SET__v1__PARENT_CHECK.json")
assert not os.path.exists(OUT)
t0 = time.time()
t = pq.read_table(PQ, columns=["source", "split", "qid", "topic_mids", "answer_values", "cwq_parent_id", "cwq_parent_kind", "eval_flag", "parent_eval_flag", "n_cwq_children"]).to_pydict()
n = len(t["qid"])
wq = {t["qid"][i]: i for i in range(n) if t["source"][i] == "nsm_webqsp"}
cw = [i for i in range(n) if t["source"][i] == "nsm_cwq"]
cw_trn = [i for i in cw if t["cwq_parent_kind"][i] == "WebQTrn" and t["cwq_parent_id"][i] in wq]
rng = np.random.RandomState(0)
perm = rng.permutation(len(cw_trn))
par = [wq[t["cwq_parent_id"][i]] for i in cw_trn]


def ov(a, b):
    return bool(set(a) & set(b))


obs_top = sum(ov(t["topic_mids"][i], t["topic_mids"][p]) for i, p in zip(cw_trn, par))
rnd_top = sum(ov(t["topic_mids"][i], t["topic_mids"][par[j]]) for i, j in zip(cw_trn, perm))
# any-shared-MID (topic or answer of child vs topic or answer of parent), a looser structural link
def allm(i):
    return set(t["topic_mids"][i]) | set(t["answer_values"][i])
obs_any = sum(bool(allm(i) & allm(p)) for i, p in zip(cw_trn, par))
rnd_any = sum(bool(allm(i) & allm(par[j])) for i, j in zip(cw_trn, perm))
kinds = collections.Counter()
by_kind_parents = {"WebQTrn": set(), "WebQTest": set()}
for i in cw:
    by_kind_parents[t["cwq_parent_kind"][i]].add(t["cwq_parent_id"][i])
by_split_kind = collections.Counter((t["split"][i], t["cwq_parent_kind"][i]) for i in cw)
# CWQ rows with parent in the 786 dev population / in NSM dev, per CWQ split
by_parent_flag = collections.Counter((t["split"][i], t["parent_eval_flag"][i]) for i in cw)
# parents that have children in BOTH cwq train and cwq dev
ch = collections.defaultdict(set)
for i in cw:
    ch[t["cwq_parent_id"][i]].add(t["split"][i])
both = sum(1 for v in ch.values() if len(v) == 2)
# WebQSP rows (train/dev) per eval_flag that have >=1 CWQ child
wq_children = collections.Counter()
for q, i in wq.items():
    if t["n_cwq_children"][i] > 0:
        wq_children[t["eval_flag"][i]] += 1
rec = {
    "RECORD": "FBQ_SET_PARENT_CHECK", "version": "v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "question": "Does the CWQ id prefix encode the WebQSP parent question?  Observed overlap of the CWQ row's topic MIDs with the id-parent's topic MIDs vs a random-parent baseline (permuted, seed 0).",
    "cwq_rows_with_WebQTrn_parent_present_in_nsm_webqsp": len(cw_trn),
    "topic_mid_overlap_observed": {"hits": obs_top, "rate": round(obs_top / len(cw_trn), 6)},
    "topic_mid_overlap_random_parent_baseline": {"hits": rnd_top, "rate": round(rnd_top / len(cw_trn), 6)},
    "any_shared_mid_(topic_or_answer)_observed": {"hits": obs_any, "rate": round(obs_any / len(cw_trn), 6)},
    "any_shared_mid_(topic_or_answer)_random_parent_baseline": {"hits": rnd_any, "rate": round(rnd_any / len(cw_trn), 6)},
    "distinct_parents_by_kind": {k: len(v) for k, v in by_kind_parents.items()},
    "cwq_rows_by_split_and_parent_kind": {"%s/%s" % k: v for k, v in sorted(by_split_kind.items())},
    "cwq_rows_by_split_and_parent_eval_flag": {"%s/%s" % k: v for k, v in sorted(by_parent_flag.items())},
    "parents_with_children_in_both_cwq_train_and_dev": both,
    "webqsp_rows_with_at_least_one_cwq_child_by_eval_flag": dict(wq_children),
    "reading": "see the record's `topic_mid_overlap_observed` vs the baseline: a large gap = the id prefix is a real parent link (even though a CWQ rewrite often swaps the topic entity so the overlap is far from 100%). WebQTest parents are only counted from the id prefix (not verifiable without reading the sealed WebQSP test).",
    "code": {"scratchpad/_fbq_parent_check.py": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()},
    "input": {"results/FREEBASE_SCALE/FBQ_SET__v1.parquet": hashlib.sha256(open(PQ, "rb").read()).hexdigest()},
    "seconds": round(time.time() - t0, 1),
}
with open(OUT, "x", encoding="utf-8", newline="\n") as fh:
    json.dump(rec, fh, indent=1)
    fh.write("\n")
print(json.dumps(rec, indent=1))
