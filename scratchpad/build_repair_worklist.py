"""
DENSE REPAIR WORKLIST
=====================
Extracts the exact encoder input for every dense row the integrity scan found damaged, so
those rows can be re-encoded and installed as an additional pointer source.

Only dense is affected. The SPLADE tree scanned clean on every channel, which fits the
mechanism: SPLADE shards are small compressed archives, dense shards are 122 MB flat files
and 2wiki_universe dense alone is 17.6 GB.

The encoder contract is read off src/experiments/canonical_encode.py rather than restated:
    docs    -> documents.jsonl, id canonical_doc_id, text field "text",  no prefix
    queries -> queries.jsonl,   id query_id,         text field "question", GTE_QINSTR prefix
    missing/None text encodes as the empty string (r.get(txf) or "")

Row alignment is not assumed. For every extracted row the id in the source jsonl must equal
the id at the same position in the channel ids_*.json, or the run aborts: if those two ever
disagreed, the whole pointer index would be resolving to the wrong vectors.
"""
import glob, io, json, os, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CANON = "data/canonical"
OUT = "scratchpad/repair_worklist"
GTE_QINSTR = ("Instruct: Given a web search query, retrieve relevant passages that answer "
              "the query\nQuery: ")
SPEC = {"docs": ("documents.jsonl", "canonical_doc_id", "text", ""),
        "queries": ("queries.jsonl", "query_id", "question", GTE_QINSTR)}


def ids_for(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "ids_*.json"))):
        out += json.load(io.open(f, encoding="utf-8"))
    return out


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    scan = json.load(io.open("scratchpad/encoding_integrity.json", encoding="utf-8"))
    man = {}
    for key, r in sorted(scan["BY_CHANNEL"].items()):
        tree, kind, model = key.split("/")
        if model != "dense" or not r.get("bad_rows"):
            continue
        wl = json.load(io.open(r["worklist"], encoding="utf-8"))
        bad = set(wl["bad_rows"])
        src, idf, txf, pre = SPEC[kind]
        ids = ids_for(os.path.join(CANON, tree, "encodings", "dense", kind))
        if len(ids) != r["rows"]:
            sys.exit("ids length %d != rows %d for %s" % (len(ids), r["rows"], key))
        outp = os.path.join(OUT, "%s__%s.jsonl" % (tree, kind))
        n = 0
        with io.open(os.path.join(CANON, tree, src), encoding="utf-8") as f, \
                io.open(outp, "w", encoding="utf-8") as g:
            for i, ln in enumerate(f):
                if i not in bad:
                    continue
                o = json.loads(ln)
                if o[idf] != ids[i]:
                    sys.exit("ROW ALIGNMENT BROKEN %s row %d: source=%s ids=%s"
                             % (key, i, o[idf], ids[i]))
                g.write(json.dumps({"row": i, "id": o[idf],
                                    "text": (o.get(txf) or "")}, ensure_ascii=False) + "\n")
                n += 1
        if n != len(bad):
            sys.exit("extracted %d of %d bad rows for %s" % (n, len(bad), key))
        man[key] = {"tree": tree, "kind": kind, "n": n, "file": outp, "prefix": pre,
                    "source": os.path.join(CANON, tree, src),
                    "id_field": idf, "text_field": txf,
                    "row_alignment_verified": True}
        print("  %-34s n=%-7d -> %s" % (key, n, outp), flush=True)
    tot = sum(v["n"] for v in man.values())
    json.dump({"RECORD": "DENSE_REPAIR_WORKLIST", "TOTAL_ROWS": tot,
               "encoder": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
               "contract": "fp16, normalize_embeddings=True, max_seq_length=32768, "
                           "docs no prefix, queries GTE_QINSTR prefix, no flash_attn",
               "channels": man, "elapsed_s": round(time.time() - t0, 1)},
              io.open("scratchpad/REPAIR_WORKLIST.json", "w", encoding="utf-8"), indent=1)
    print("\nTOTAL %s rows  %.1fs" % (format(tot, ","), time.time() - t0))


if __name__ == "__main__":
    main()
