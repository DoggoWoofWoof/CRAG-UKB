"""L3 DEVELOPMENT (WebQSP KB-L3 transfer) -- POPULATION: the development rows, written once.
(User ruling 2026-09-30, REPORT 43.6: "WebQSP KB-L3 transfer first, with a development population of all scoreable non-held-out WebQSP
dev rows (only sealed split B, TEST and malformed or unscoreable rows excluded; no random 2,000-row sample)".)

Reading of the ruling (the same carve as L1_COVPART section 30, which the L1 lane already fixed and pinned):
  eval split      train_holdout: 1,549 rows (the cap EVAL_CAP = 2000 does not bind, so there is no random sample: every row is a
                  candidate)
  unscoreable     rows without a gold node (query index gold count 0): 46 dropped -> 1,503 scoreable rows
  sealed split B  parity of sha1(query_id)[:8] is 1: 717 rows, not read here (only their ids and gold COUNTS enter the parity
                  carve; no text, no gold node, no score)
  TEST            1,639 rows in the 'test' split, never touched
  development     the 786 parity-0 rows (split A) = "all scoreable non-held-out dev rows of the eval split"
The 1,549 train rows outside the train_holdout carve are the L2-training side of the legacy layout; they are not part of this
population (they have no L1 cache and were never a development population).  If the user reads the ruling as including them, the
population is re-declared in a new record; this one is not edited.

The rule is exactly _l1c_flath2.py lines 868-878 (cd.eval_rows(), the replay-cache shuffle with np.random.seed(0), first EVAL_CAP,
sorted, drop rows without gold, parity of sha1(query_id)[:8]); the record is checked against L1_COVPART's pinned POP constants
(row_query_ids_sha256 faa1d372..., n 1,503, n_A 786, n_B 717) and against the rows of results/L1_COVPART/flath2_G_webqsp.npz, the
run that first used split A.

Usage: python scratchpad/_l3w_pop.py            -> results/L3_DEV/l3w_population_webqsp__v1.json (write-once)
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from src.l1_canonical import adapter as AD  # noqa: E402

OUT = os.path.join(REPO, "results", "L3_DEV", "l3w_population_webqsp__v1.json")
FLAT_G = os.path.join(REPO, "results", "L1_COVPART", "flath2_G_webqsp.npz")
EVAL_CAP = AD.CONTRACT["EVAL_CAP"]
WANT = {"row_query_ids_sha256": "faa1d372294cc62f116549bdce087bb0d3f8ce533dce8bfd814c56e1c42e7a2e", "n": 1503, "n_A": 786, "n_B": 717}


def sha_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def population(cd):
    """(rows_A, POPV): the carve of _l1c_flath2.py, unchanged."""
    ev = [int(x) for x in cd.eval_rows()]
    z = np.load(cd._query_index_path(), allow_pickle=False)
    gold_ptr = z["gold_ptr"]
    n_eval = len(ev)
    np.random.seed(0)
    np.random.shuffle(ev)
    pop = sorted(ev[:EVAL_CAP])
    ng_all = np.diff(gold_ptr)[np.asarray(pop, np.int64)]
    pop = [r for r, n in zip(pop, ng_all) if n > 0]
    qids = [cd.query_ids[r] for r in pop]
    par = np.array([int(hashlib.sha1(q.encode("utf-8")).hexdigest()[:8], 16) & 1 for q in qids], np.int8)
    rows_A = np.asarray([r for r, p in zip(pop, par) if p == 0], np.int64)
    POPV = {"eval_split": cd.eval_split, "n_eval_rows": n_eval, "eval_cap": int(EVAL_CAP), "cap_binds": bool(n_eval > EVAL_CAP),
            "n_without_gold_dropped": int(len(ev[:EVAL_CAP]) - len(pop)), "row_query_ids_sha256": sha_text(",".join(qids)),
            "rows_sha256": sha_text(",".join(str(r) for r in pop)), "n": len(pop), "n_A": int(len(rows_A)), "n_B": int((par == 1).sum())}
    for k in WANT:
        assert POPV[k] == WANT[k], "population %s differs from the L1_COVPART pin: %s" % (k, POPV[k])
    return rows_A, POPV


def main():
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    t0 = time.time()
    cd = AD.CanonicalDataset("webqsp")
    rows_A, POPV = population(cd)
    zg = np.load(FLAT_G)
    assert (zg["rows"] == rows_A).all(), "split A differs from the rows of flath2_G_webqsp.npz"
    qids_A = [cd.query_ids[int(r)] for r in rows_A]
    assert not any(int(r) >= cd.split_ranges["test"][0] for r in rows_A), "a TEST row"
    z = np.load(cd._query_index_path(), allow_pickle=False)
    ng = np.diff(z["gold_ptr"])[rows_A]
    assert (ng >= 1).all()
    rec = {
        "RECORD": "L3W_POPULATION", "dataset": "webqsp", "stage": "L3 development: WebQSP KB-L3 transfer",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ruling (verbatim, REPORT 43.6)": "WebQSP KB-L3 transfer first, with a development population of all scoreable non-held-out WebQSP dev rows "
                                          "(only sealed split B, TEST and malformed or unscoreable rows excluded; no random 2,000-row sample)",
        "reading": __doc__,
        "carve": POPV,
        "rows": [int(r) for r in rows_A], "n_rows": int(len(rows_A)), "query_ids_sha256": sha_text(",".join(qids_A)),
        "rows_A_sha256": sha_text(",".join(str(int(r)) for r in rows_A)), "query_ids": qids_A,
        "gold_nodes": {"total": int(ng.sum()), "min": int(ng.min()), "median": float(np.median(ng)), "max": int(ng.max()),
                       "by_count": {b: int(((ng >= lo) & (ng <= hi)).sum()) for b, lo, hi in
                                    (("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5-10", 5, 10), ("11+", 11, 1 << 40))}},
        "matches": {"L1_COVPART POP constants": WANT, "flath2_G_webqsp.npz rows": {"sha256": sha_file(FLAT_G), "equal": True}},
        "dataset_pins": {"DATASET_json_RECORD_SHA256": cd.record_sha, "query_index.npz": sha_file(cd._query_index_path())},
        "not_read": ["split B (717 rows): ids and gold counts entered the parity carve only", "TEST (1,639 rows)",
                     "the 1,549 train rows outside train_holdout"],
        "code": {"path": "scratchpad/_l3w_pop.py", "sha256": sha_file(os.path.abspath(__file__))},
        "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, ensure_ascii=False))
    print("wrote %s (%d rows, %d gold nodes) sha256 %s" % (OUT, len(rows_A), int(ng.sum()), sha_file(OUT)[:16]))


if __name__ == "__main__":
    main()
