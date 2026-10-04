"""Check 7 (CVT coverage) plus the entity2id semantics question check 2 left open.

    python scratchpad/final_canonical_build/webqsp_v1/v3_check7_cvt.py

Check 4 found the deficit is 18.20% missing ENTITIES against only 5.23% missing edges between
entities IDIR does know. So the question is which entities. V1 classified its MID-labelled nodes and
96.1% of them are CVT_MEDIATOR, so this is where a CVT answer would show up.

Two things measured here.

1. WHAT entity2id MEANS. It carries 101,917,083 ids against a published entity count of 59,894,890
   for this variant. If it is a global object table rather than this variant's node set, then
   "present in entity2id" is weaker than "present in the graph", and check 2's coverage numbers are
   optimistic. Settled by counting the distinct ids that actually appear in the delivered triples.

2. COVERAGE BY NODE KIND, at both strengths: known to entity2id, and incident to at least one
   backbone edge. V1's node_kind is the only classification either side has.
"""
import json, os, time
import numpy as np
import pyarrow.parquet as pq
from pyarrow import csv as pacsv

IDIR = "data/final_canonical/freebase_v3/_acquisition/idir/extracted/idirlab-freebases"
BB = f"{IDIR}/FB+CVT-REV"
PROBE = "data/final_canonical/freebase_v3/probe"
OUT = "data/final_canonical/freebase_v3/V3_CHECK7_CVT_COVERAGE.json"


def main():
    t0 = time.time()
    t = pq.read_table(f"{PROBE}/v1_mid_node_kind.parquet")
    v1_mid = t.column("mid").to_pylist()
    v1_kind = t.column("node_kind").to_pylist()
    nsm_mids = set(pq.read_table(f"{PROBE}/nsm_all_mids.parquet").column(0).to_pylist())
    ans = set(pq.read_table(f"{PROBE}/nsm_answer_mids.parquet").column(0).to_pylist())
    top = set(pq.read_table(f"{PROBE}/nsm_topic_mids.parquet").column(0).to_pylist())
    want = {"/" + m.replace(".", "/", 1): m for m in nsm_mids}

    found = {}
    max_id = 0
    with open(f"{BB}/entity2id.txt", encoding="utf-8") as fh:
        for line in fh:
            c = line.rfind(",")
            if c < 0:
                continue
            i = int(line[c + 1:])
            if i > max_id:
                max_id = i
            m = want.get(line[:c])
            if m is not None:
                found[m] = i
    print(f"[entity2id] max_id={max_id:,} hits={len(found):,} t={time.time()-t0:.0f}s", flush=True)

    # ---- which ids actually appear in the delivered triples ----
    seen_arr = np.zeros(max_id + 1, dtype=bool)
    n_trip = 0
    for split in ("train", "test", "valid"):
        tb = pacsv.read_csv(f"{BB}/{split}.txt",
                            read_options=pacsv.ReadOptions(autogenerate_column_names=True),
                            convert_options=pacsv.ConvertOptions(
                                column_types={"f0": "int32", "f1": "int32", "f2": "int32"}))
        s = tb.column(0).to_numpy()
        o = tb.column(2).to_numpy()
        n_trip += len(s)
        seen_arr[s] = True
        seen_arr[o] = True
        del tb, s, o
        print(f"[{split}] cumulative {n_trip:,} t={time.time()-t0:.0f}s", flush=True)
    used = int(seen_arr.sum())

    entity_semantics = {
        "entity2id_ids": max_id + 1,
        "distinct_ids_actually_appearing_in_delivered_triples": used,
        "published_entity_count": 59894890,
        "delivered_triples_counted": n_trip,
        "published_triple_count": 134213735,
        "triple_count_delta": 134213735 - n_trip,
        "VERDICT": None,
    }
    close = abs(used - 59894890) / 59894890 < 0.02
    entity_semantics["VERDICT"] = (
        "entity2id.txt is a GLOBAL object table, not this variant's node set. The distinct ids "
        "appearing in the delivered triples land at the published entity count, while the file "
        "itself is 1.7x larger. Membership in entity2id is therefore NOT membership in the graph, "
        "and check 2's coverage figures are an upper bound."
        if close else
        "entity2id line count and used-id count do NOT reconcile to the published entity count. "
        "Recorded as an unexplained discrepancy; not used downstream and not guessed at.")
    print(json.dumps(entity_semantics, indent=1), flush=True)

    # ---- coverage by node kind, at both strengths ----
    by_kind = {}
    for m, k in zip(v1_mid, v1_kind):
        d = by_kind.setdefault(k, {"n": 0, "in_entity2id": 0, "has_backbone_edge": 0})
        d["n"] += 1
        i = found.get(m)
        if i is not None:
            d["in_entity2id"] += 1
            if seen_arr[i]:
                d["has_backbone_edge"] += 1
    for k, d in by_kind.items():
        d["in_entity2id_pct"] = round(100 * d["in_entity2id"] / d["n"], 3)
        d["has_backbone_edge_pct"] = round(100 * d["has_backbone_edge"] / d["n"], 3)

    def strength(sel, label):
        n = len(sel)
        a = sum(1 for m in sel if m in found)
        b = sum(1 for m in sel if m in found and seen_arr[found[m]])
        return {"set": label, "n": n, "in_entity2id": a, "in_entity2id_pct": round(100 * a / n, 3),
                "has_backbone_edge": b, "has_backbone_edge_pct": round(100 * b / n, 3)}

    doc = {
        "schema": "V3_CHECK7_CVT_COVERAGE/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ENTITY2ID_SEMANTICS": entity_semantics,
        "CHECK_7_COVERAGE_BY_V1_NODE_KIND": by_kind,
        "COVERAGE_BY_ROLE": [strength(nsm_mids, "all NSM mids"),
                             strength(ans, "gold answer mids"),
                             strength(top, "question topic mids")],
        "READING_GUIDE": "two strengths are reported because they answer different questions. "
                         "in_entity2id says IDIR has heard of the identifier. has_backbone_edge "
                         "says the node is actually reachable in FB+CVT-REV. Only the second is a "
                         "graph property, and only the second bounds what CRAG could traverse.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps(by_kind, indent=1))
    print(json.dumps(doc["COVERAGE_BY_ROLE"], indent=1))


if __name__ == "__main__":
    main()
