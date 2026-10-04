"""
QUERY POINTER INDEX
===================
Gives every dataset ONE canonical query view over the frozen query embeddings, without
re-encoding and without copying vectors.

Canonical query order is deterministic and documented: splits are concatenated in
SPLIT_ORDER (train, dev, validation, test -- whichever exist), and within a split the
order is the file order of queries/<split>.jsonl. query_ids.json records that order
explicitly so no consumer has to re-derive it.

Four of the five datasets already share a query-id scheme with their Phase-C tree, so the
bridge there is identity-by-id and this script VERIFIES it rather than assuming it.

MetaQA is the exception and the reason this script exists. Its query embeddings were
produced in three per-hop trees (metaqa_1hop 116,045 / metaqa_2hop 148,724 /
metaqa_3hop 142,744 = 407,513) under a different id scheme (mq_<hash>), while
canonical_v1 ids are metaqa:<hop>:<split>:<index>. The join key is
(hop, official_split, original_index) and it was proven a bijection: 407,513 rows, no
duplicate keys, all 9 (hop, split) cells equal, 0 missing, 0 question-text mismatches.
This script re-proves that on every run instead of trusting the earlier probe.

Emits per dataset per model:
    queries/pointer_index/<model>.npz  src(int8), row(int32)
    queries/pointer_index/query_ids.json
Resolver is identical to the document pointer index:
    shard = row // shard_size, offset = row % shard_size
inside data/canonical/<tree_for_src>/encodings/<model>/queries/.
"""
import json, io, os, sys, time
from array import array
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = "data/final_canonical"
SPLIT_ORDER = ["train", "dev", "validation", "test"]

# dataset -> ordered list of Phase-C trees holding its query encodings.
# The list position IS the src code stored in the pointer array.
TREES = {"metaqa": ["metaqa_1hop", "metaqa_2hop", "metaqa_3hop"],
         "2wiki": ["2wiki"], "musique": ["musique"],
         "hotpotqa": ["hotpotqa"], "squad": ["squad"]}


def fc_queries(ds):
    """Canonical query rows in canonical order: SPLIT_ORDER, then file order."""
    for sp in SPLIT_ORDER:
        p = "%s/%s/queries/%s.jsonl" % (ROOT, ds, sp)
        if not os.path.exists(p):
            continue
        with io.open(p, encoding="utf-8") as f:
            for ln in f:
                yield sp, json.loads(ln)


def phase_c(tree):
    """query_id -> row, plus row -> question, for one Phase-C tree."""
    pos, q = {}, []
    with io.open("data/canonical/%s/queries.jsonl" % tree, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            o = json.loads(ln)
            pos[o["query_id"]] = i
            q.append(o)
    return pos, q


def build(ds):
    t0 = time.time()
    trees = TREES[ds]
    pos, rows = {}, {}
    for t in trees:
        pos[t], rows[t] = phase_c(t)

    # metaqa joins on (hop, official_split, original_index); everything else on query_id
    keyed = {}
    if ds == "metaqa":
        for ti, t in enumerate(trees):
            for i, o in enumerate(rows[t]):
                k = (o["hop"], o["official_split"], o["original_index"])
                if k in keyed:
                    raise SystemExit("DUPLICATE JOIN KEY %s %s" % (ds, k))
                keyed[k] = (ti, i, o["question"])

    src, row, qids = array("b"), array("i"), []
    n_q = n_mismatch = 0
    for sp, o in fc_queries(ds):
        qid = o["query_id"]
        qids.append(qid)
        n_q += 1
        if ds == "metaqa":
            _, hp, spl, ix = qid.split(":")
            k = (hp, spl, int(ix))
            if k not in keyed:
                raise SystemExit("QUERY NOT IN PHASE_C %s %s" % (ds, qid))
            ti, i, question = keyed[k]
            # the join is only trustworthy if the joined rows carry the same question
            if question != o["question"]:
                n_mismatch += 1
            src.append(ti)
            row.append(i)
        else:
            if qid not in pos[trees[0]]:
                raise SystemExit("QUERY ID NOT IN PHASE_C %s %s" % (ds, qid))
            i = pos[trees[0]][qid]
            if rows[trees[0]][i].get("question") != o.get("question"):
                n_mismatch += 1
            src.append(0)
            row.append(i)
    if n_mismatch:
        raise SystemExit("QUESTION TEXT MISMATCH %s n=%d" % (ds, n_mismatch))

    # every Phase-C query row must be pointed to exactly once: no orphans, no double use
    total_pc = sum(len(rows[t]) for t in trees)
    if total_pc != n_q:
        raise SystemExit("PHASE_C QUERY ROWS %d != CANONICAL QUERIES %d (%s)"
                         % (total_pc, n_q, ds))
    s = np.frombuffer(src, dtype=np.int8)
    r = np.frombuffer(row, dtype=np.int32)
    seen = set(zip(s.tolist(), r.tolist()))
    if len(seen) != n_q:
        raise SystemExit("QUERY POINTERS NOT INJECTIVE %s %d/%d" % (ds, len(seen), n_q))

    d = "%s/%s/queries/pointer_index" % (ROOT, ds)
    os.makedirs(d, exist_ok=True)
    out = {"n_queries": n_q, "trees": trees, "join": (
        "(hop, official_split, original_index)" if ds == "metaqa" else "query_id identity"),
        "question_text_verified": True, "split_order": SPLIT_ORDER}
    for m in ("dense", "splade"):
        sizes, counts = [], []
        for t in trees:
            j = json.load(io.open(
                "data/canonical/%s/encodings/%s/queries/index.json" % (t, m), encoding="utf-8"))
            sizes.append(j["shard_size"])
            counts.append(sum(int(sh["rows"]) for sh in j["shards"]))
        for ti, t in enumerate(trees):
            mx = r[s == ti]
            if mx.size and int(mx.max()) >= counts[ti]:
                raise SystemExit("QUERY ROW OUT OF RANGE %s %s %s" % (ds, m, t))
            if int((s == ti).sum()) != counts[ti]:
                raise SystemExit("TREE ROW COUNT MISMATCH %s %s %s %d!=%d"
                                 % (ds, m, t, int((s == ti).sum()), counts[ti]))
        np.savez(os.path.join(d, "%s.npz" % m), src=s, row=r)
        out[m] = {"shard_size_per_tree": sizes, "n_items_per_tree": counts}
    json.dump(qids, io.open(os.path.join(d, "query_ids.json"), "w", encoding="utf-8"))
    out["seconds"] = round(time.time() - t0, 1)
    print("== %-9s queries=%-7d trees=%-42s join=%s"
          % (ds, n_q, ",".join(trees), out["join"]))
    return out


if __name__ == "__main__":
    man = {}
    for ds in (sys.argv[1:] or ["metaqa", "2wiki", "musique", "hotpotqa", "squad"]):
        man[ds] = build(ds)
    p = os.path.join(ROOT, "POINTER_INDEX.json")
    old = json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else {}
    old.setdefault("queries", {}).update(man)
    old.setdefault("resolver", {})["QUERIES"] = (
        "queries/pointer_index/<model>.npz ; position i == entry i of query_ids.json ; "
        "src indexes the trees list; shard = row // shard_size, offset = row % shard_size "
        "inside data/canonical/<tree>/encodings/<model>/queries/")
    json.dump(old, io.open(p, "w", encoding="utf-8"), indent=2)
    print("\nwrote", p)
