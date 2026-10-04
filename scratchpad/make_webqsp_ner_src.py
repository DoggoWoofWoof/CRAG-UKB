# -*- coding: utf-8 -*-
"""Materialise webqsp's NER input shards from the canonical nodes.jsonl.

webqsp_rog_v1 has no encodings/_src/docs tree (it was encoded from a different staging path),
so build_ner_sharded.py has nothing to read. The canonical text IS nodes.jsonl's "text" field --
the NAME_ONLY column that ships -- so write it out in the same {id, text} shard format the
builder expects, with node_id as the id so edges map straight back to canonical positions.
"""
import hashlib
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = "data/final_canonical/webqsp/nodes.jsonl"
OUT = "data/_family_v1/webqsp/encodings/_src/docs"
SHARD = 40000

os.makedirs(OUT, exist_ok=True)
h = hashlib.sha256()
n = 0
sid = 0
fh = None
blank = 0
for ln in io.open(SRC, encoding="utf-8"):
    r = json.loads(ln)
    if n % SHARD == 0:
        if fh:
            fh.close()
        fh = io.open("%s/shard_%05d.jsonl" % (OUT, sid), "w", encoding="utf-8", newline=chr(10))
        sid += 1
    t = r["text"]
    if not t.strip():
        blank += 1
    fh.write(json.dumps({"id": r["node_id"], "text": t}, ensure_ascii=False) + chr(10))
    h.update((r["node_id"] + chr(1) + t + chr(10)).encode("utf-8"))
    n += 1
if fh:
    fh.close()
json.dump({"source_sha256": h.hexdigest(), "shard_size": SHARD, "n_shards": sid, "n_items": n,
           "source": SRC, "text_field": "text", "id_field": "node_id",
           "note": "generated for NER; webqsp_rog_v1 has no encodings/_src/docs of its own"},
          io.open(OUT + "/_srcmeta.json", "w", encoding="utf-8"), indent=1)
print("webqsp NER src: %d items, %d shards, %d blank-text, sha %s" % (n, sid, blank, h.hexdigest()[:16]))
