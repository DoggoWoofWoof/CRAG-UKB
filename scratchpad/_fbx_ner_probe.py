"""FBX_SCALE Track A -- how expensive are NER 'shares-entity' edges on the Freebase names?  A read-only throughput / yield probe; no edge is written into any tree.

The recipe is the frozen one (src/pipeline/ner_edges.build_ner_edges): spaCy en_core_web_sm with parser and lemmatizer disabled, the nine-label whitelist, len(entity text) > 2, entity key =
e.text.lower().strip(), one document per node name, edges among the nodes sharing an entity with 2 <= df <= 25, weight sum 1/df.  Positions of the Freebase tree are ranks of a hash of the node
id, so a contiguous run of rows is a uniform random sample of the 301,977,131 nodes; the probe reads one such run (read-only) and reports rates, not edge counts of the full graph.

  python scratchpad/_fbx_ner_probe.py <shard_index> <n_names> [<batch>]
"""
import io
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.pipeline.ner_edges import ENT_LABELS, _nlp  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE = os.path.join(REPO, "data", "final_canonical", "freebase")
OUT = os.path.join(REPO, "results", "FREEBASE_SCALE", "FBX_NER_PROBE__v1.json")


def main(shard, n, batch):
    assert not os.path.exists(OUT), "write-once: %s exists" % OUT
    pf = pq.ParquetFile(os.path.join(TREE, "nodes", "shard_%05d.parquet" % shard))
    names, kinds = [], []
    for rb in pf.iter_batches(batch_size=65536, columns=["name", "kind"]):
        names += rb.column("name").to_pylist()
        kinds += rb.column("kind").to_pylist()
        if len(names) >= n:
            break
    names, kinds = names[:n], kinds[:n]
    nlp = _nlp()
    t = time.time()
    ent_docs, n_with, n_ent, chars = {}, 0, 0, 0
    for i, doc in enumerate(nlp.pipe([str(x) for x in names], batch_size=batch)):
        got = False
        for e in doc.ents:
            if e.label_ in ENT_LABELS and len(e.text) > 2:
                ent_docs.setdefault(e.text.lower().strip(), set()).add(i)
                n_ent += 1
                got = True
        n_with += got
    sec = time.time() - t
    chars = int(sum(len(str(x)) for x in names))
    df = np.array([len(v) for v in ent_docs.values()], np.int64)
    kc = np.bincount(np.asarray(kinds, np.int64) if len(kinds) else np.zeros(0, np.int64))
    pairs_in_sample = int(sum(d * (d - 1) // 2 for d in df if 2 <= d <= 25))
    rec = {"RECORD": "FBX_NER_PROBE", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tree": "data/final_canonical/freebase (read-only)", "shard": shard, "names": len(names),
           "recipe": "spaCy en_core_web_sm (parser, lemmatizer disabled), ENT_LABELS whitelist, len > 2, lower().strip(); batch %d, 1 process, 1 thread-equivalent" % batch,
           "seconds": round(sec, 2), "names_per_second_1_process": round(len(names) / sec, 1), "chars_per_name_mean": round(chars / max(len(names), 1), 2),
           "names_with_an_entity": int(n_with), "entity_mentions": int(n_ent), "distinct_entity_keys": int(len(ent_docs)), "kind_counts_in_sample": [int(x) for x in kc],
           "entity_df_in_sample": {"df1": int((df == 1).sum()), "df2_25": int(((df >= 2) & (df <= 25)).sum()), "df_gt_25": int((df > 25).sum())},
           "sample_pairs_with_2<=df<=25": pairs_in_sample,
           "note": "df and pairs are counted INSIDE the sample: an entity's full-graph df is about df_sample / (sample / 301,977,131) for a frequent entity and unknowable for a rare one, so no full-graph edge "
                   "count is inferred here; the probe measures per-name cost (throughput) and mention yield only",
           "projection": {"full_graph_names": 301977131, "single_process_hours": round(301977131 / (len(names) / sec) / 3600.0, 1),
                          "hours_at_8_processes": round(301977131 / (8 * len(names) / sec) / 3600.0, 2), "hours_at_24_processes": round(301977131 / (24 * len(names) / sec) / 3600.0, 2)}}
    with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    main(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else 256)
