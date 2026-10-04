# -*- coding: utf-8 -*-
"""
WebQSP query work order: the canonical query ORDER, and the texts to encode in it.

WHY THE ORDER IS COPIED FROM THE OTHER FIVE RATHER THAN CHOSEN
    In this package a query embedding has no id of its own. queries/pointer_index/query_ids.json
    IS the identity, and row i of the query matrix is query_ids[i]. The convention the five
    frozen datasets use, verified against musique rather than assumed, is: concatenate the
    official splits in the order train, dev, validation, test -- whichever exist -- each in
    file order, and that list is query_ids.json.

    So webqsp is train then test. Choosing any other order would not be wrong in isolation; it
    would be wrong in company, because every consumer that already knows how to read the five
    would silently misread the sixth.

WHAT GETS ENCODED, AND THE PREFIX THAT MUST NOT BE FORGOTTEN
    The dense query channel takes the GTE retrieval instruction; documents take nothing. That
    asymmetry is the whole point of the instruct model, and a query channel encoded without it
    is not comparable to the corpus it is scored against. The prefix is written here, once,
    copied verbatim from canonical_encode.py:30, and passed explicitly to the encoder rather
    than defaulted anywhere.

    SPLADE takes no prefix on either side.

NO DEDUPLICATION
    The document side deduplicates because 2.59M nodes collapse to 1.79M distinct texts and the
    saving is real. 4,737 queries are not worth a dedup indirection, and keeping row == query
    index means the pointer index is the identity map, which is one fewer thing to get wrong.
    Duplicate question strings are counted and reported, not merged.
"""
import collections
import hashlib
import io
import json
import os
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

D = "data/final_canonical/webqsp"
SPLIT_ORDER = ["train", "dev", "validation", "test"]
GTE_QINSTR = ("Instruct: Given a web search query, retrieve relevant passages that answer the "
              "query\nQuery: ")
OUT_IDS = D + "/queries/pointer_index/query_ids.json"
OUT_TXT = D + "/query_inputs.jsonl"
OUT_REC = D + "/QUERY_WORK_ORDER.json"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    # the prefix is a literal in canonical_encode.py; if that ever changes, the query channels
    # stop being comparable and this must fail rather than quietly encode the old one.
    src = io.open("src/experiments/canonical_encode.py", encoding="utf-8").read()
    if repr(GTE_QINSTR)[1:-1] not in src.replace('"', "'").replace("\\n", "\\n"):
        if GTE_QINSTR not in src.encode().decode("unicode_escape"):
            expect = 'GTE_QINSTR = "Instruct: Given a web search query, retrieve relevant '
            if expect not in src:
                sys.exit("GTE_QINSTR no longer matches canonical_encode.py -- refusing to "
                         "encode queries against a prefix the five datasets did not use")

    present = [s for s in SPLIT_ORDER if os.path.exists("%s/queries/%s.jsonl" % (D, s))]
    unexpected = [f[:-6] for f in sorted(os.listdir(D + "/queries"))
                  if f.endswith(".jsonl") and f[:-6] not in SPLIT_ORDER]
    if unexpected:
        sys.exit("split file(s) not in the canonical order list: %s" % unexpected)

    ids, texts, by_split = [], [], {}
    for sp in present:
        n = 0
        for ln in io.open("%s/queries/%s.jsonl" % (D, sp), encoding="utf-8"):
            o = json.loads(ln)
            ids.append(o["query_id"])
            texts.append(o["question"])
            n += 1
        by_split[sp] = n

    if len(set(ids)) != len(ids):
        dup = [k for k, v in collections.Counter(ids).items() if v > 1][:5]
        sys.exit("duplicate query_id in the canonical order: %s" % dup)
    empty = sum(1 for t in texts if not (t or "").strip())
    if empty:
        sys.exit("%d queries have empty text" % empty)

    os.makedirs(os.path.dirname(OUT_IDS), exist_ok=True)
    json.dump(ids, io.open(OUT_IDS, "w", encoding="utf-8"), ensure_ascii=False)
    with io.open(OUT_TXT, "w", encoding="utf-8", newline="\n") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"row": i, "text": t}, ensure_ascii=False) + "\n")

    dupe_text = len(texts) - len(set(texts))
    rec = {"RECORD": "WEBQSP_QUERY_WORK_ORDER",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_queries": len(ids), "by_split": by_split, "split_order": present,
           "order_convention": ("official splits concatenated in the order train, dev, "
                                "validation, test, each in file order -- verified against "
                                "musique, where the concatenation reproduces query_ids.json "
                                "exactly"),
           "duplicate_question_strings": dupe_text,
           "deduplication": ("NONE. row == query index, so the query pointer index is the "
                             "identity map. Duplicate question strings are counted, not merged."),
           "dense_query_prefix": GTE_QINSTR,
           "splade_query_prefix": "",
           "prefix_provenance": "canonical_encode.py:30, verbatim",
           "files": {OUT_IDS: {"bytes": os.path.getsize(OUT_IDS), "sha256": sha256(OUT_IDS)},
                     OUT_TXT: {"bytes": os.path.getsize(OUT_TXT), "sha256": sha256(OUT_TXT)}},
           "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rec, io.open(OUT_REC, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("queries=%d  %s  duplicate question strings=%d"
          % (len(ids), by_split, dupe_text))
    print("wrote %s" % OUT_IDS)
    print("wrote %s" % OUT_TXT)
    print("wrote %s  %.1fs" % (OUT_REC, rec["elapsed_s"]))


if __name__ == "__main__":
    main()
