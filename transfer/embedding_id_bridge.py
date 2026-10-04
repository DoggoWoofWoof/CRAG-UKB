# -*- coding: utf-8 -*-
"""Map the Aug-23 encoding row ids to canonical_v1 node_ids.

The encodings under data/canonical/<enc_dir>/encodings/ were computed before the canonical_v1
node_id scheme existed. They are still 100% reusable: the row sets are identical, only the id
STRINGS differ. For three corpora the map is a pure string rewrite; for musique and squad the
legacy id hashed the TEXT ALONE while canonical hashes (title, text), so those two need a real
lookup table -- emitted here, and small (117,534 + 20,233 rows).

Measured coverage (see TRANSFER_MANIFEST.md):
  2wiki    5,989,847 / 5,989,847   hotpotqa 5,233,329 / 5,233,329   metaqa 43,234 / 43,234
  musique    117,534 / 117,534     squad       20,233 / 20,233
"""
import os, sys, io, json, glob

ENC_DIR = {"2wiki": "2wiki_universe", "hotpotqa": "hotpotqa", "metaqa": "metaqa",
           "musique": "musique", "squad": "squad"}

# pure rewrites: legacy encoding row id -> canonical_v1 node_id
REWRITE = {
    "2wiki":    lambda x: "2wiki:c" + x.split(":", 1)[1],      # 2wu:17888798   -> 2wiki:c17888798
    "hotpotqa": lambda x: "hotpotqa:c" + x.split("_", 1)[1],   # hotpot_7533751 -> hotpotqa:c7533751
    "metaqa":   lambda x: "metaqa:e%05d" % int(x.rsplit("_", 1)[1]),  # metaqa_ent_0 -> metaqa:e00000
}
# musique/squad: no rewrite exists; join on exact text (the encoder input was text only)
TEXT_JOIN = ("musique", "squad")


def encoding_row_ids(ds, model="dense"):
    """Row ids in encoder output order, i.e. row i of shard N is ids[i] of that shard."""
    out = []
    for f in sorted(glob.glob("data/canonical/%s/encodings/%s/docs/ids_*.json" % (ENC_DIR[ds], model))):
        out.extend(json.load(io.open(f, encoding="utf-8")))
    return out


def build(ds, out_path=None):
    ids = encoding_row_ids(ds)
    if ds in REWRITE:
        m = {i: REWRITE[ds](i) for i in ids}
    else:
        by_text = {}
        with io.open("data/final_canonical/%s/nodes.jsonl" % ds, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                by_text.setdefault(r["text"], []).append(r["node_id"])
        src = {}
        for f in sorted(glob.glob("data/canonical/%s/encodings/_src/docs/shard_*.jsonl" % ENC_DIR[ds])):
            with io.open(f, encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    src[r["id"]] = r["text"]
        # one legacy row can serve SEVERAL canonical nodes: musique has 1 text carried by 2 titles,
        # and both canonical nodes legitimately share that row's vector because the encoder input
        # was the text alone. The value is therefore a LIST, not a scalar.
        m = {i: by_text.get(src.get(i, "\x00"), []) for i in ids}
    if out_path:
        with io.open(out_path, "w", encoding="utf-8") as f:
            json.dump({"dataset": ds, "encoding_dir": ENC_DIR[ds],
                       "rule": "string rewrite" if ds in REWRITE else "exact text join",
                       "n_rows": len(ids), "map": m}, f, ensure_ascii=False)
    return m


if __name__ == "__main__":
    os.makedirs("transfer/id_bridge", exist_ok=True)
    for ds in TEXT_JOIN:                      # only these two need a materialised table
        p = "transfer/id_bridge/%s_legacy_to_canonical.json" % ds
        m = build(ds, p)
        n_multi = sum(1 for v in m.values() if isinstance(v, list) and len(v) > 1)
        n_none = sum(1 for v in m.values() if isinstance(v, list) and not v)
        print("%-9s rows=%d  unmatched=%d  one_row_serves_2_nodes=%d  -> %s"
              % (ds, len(m), n_none, n_multi, p))
    print("2wiki / hotpotqa / metaqa need no table -- use REWRITE[ds] directly.")
