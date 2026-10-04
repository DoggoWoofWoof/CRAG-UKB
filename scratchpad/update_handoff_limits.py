# -*- coding: utf-8 -*-
"""
Two jobs.

1. Section 8 lists six limitations, all from the five-dataset era. LOCKED_6_OF_6 records seven
   more for dataset 6. A reader who reads only the handoff -- which is most readers -- would
   never see them. Fold them in, condensed, each pointing at where the full text lives.

2. Re-wrap the three paragraphs the 6/6 updater emitted as single long lines. Everything else
   in the document wraps near 95 columns; these did not, which is visible in any editor and in
   a diff.
"""
import io
import sys

H = "data/final_canonical/HANDOFF.md"
s = io.open(H, encoding="utf-8").read()

REPS = [
    # ------------------------------------------------------------- section 8
    ("6. **Retrieval caches are partial.** See below.",
     "6. **Retrieval caches are partial.** metaqa, squad and musique only. See §9.\n"
     "\n"
     "Dataset 6 adds seven of its own. The full text of each is in\n"
     "`LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json` → `KNOWN_LIMITATIONS`; condensed:\n"
     "\n"
     "7. **WebQSP carries the structural family only** — no NER (no document prose to run it\n"
     "   over) and no kNN (kNN needs embeddings, which the graph build deliberately did not\n"
     "   depend on).\n"
     "8. **WebQSP SPLADE is attention-masked and the five's mostly is not.** Ranks fuse; raw\n"
     "   scores do not compare across that boundary. §5.\n"
     "9. **WebQSP gold covers 62.42% of references / 97.74% of answerable queries.** The\n"
     "   topic-entity control at 99.92% is what shows this is the RoG graph's contents and not a\n"
     "   broken join. 106 queries have gold that is genuinely absent. §9.\n"
     "10. **WebQSP entity text is `NAME_ONLY`.** `NAME_PLUS_FACTS` is materialised and available\n"
     "    as the ablation arm; it costs 2.554× the tokens. A column choice, recorded as an open\n"
     "    axis rather than settled by default.\n"
     "11. **WebQSP shards at 12,000 rows, the five at 40,000.** The resolver reads `shard_size`\n"
     "    per store, so this is transparent to every consumer.\n"
     "12. **WebQSP has no retrieval cache**, for the same throughput reason as hotpotqa and\n"
     "    2wiki. Caches are an accelerator, never a dependency.\n"
     "13. **WebQSP is the RoG WebQSP+CWQ subgraph and must never be called full Freebase.** The\n"
     "    canonical Freebase universe is a separate frozen artifact of 301,977,131 nodes (§10);\n"
     "    these 2,592,894 nodes are not it."),

    # ------------------------------------------------------ re-wrap, §9 encoder
    ("**The encoder ran, and here is what landed.** Deduplication does most of the work: "
     "2,592,894 nodes carry only 1,791,533 distinct texts, which the pointer index expands back "
     "to per-node vectors. Encoding the distinct texts locally would have cost ~34 days for "
     "dense (1.0e17 FLOP at 33.8 GFLOPS, no CUDA), so it ran on a Modal A10G.",
     "**The encoder ran, and here is what landed.** Deduplication does most of the work:\n"
     "2,592,894 nodes carry only 1,791,533 distinct texts, which the pointer index expands back\n"
     "to per-node vectors. Encoding those texts locally would have cost ~34 days for dense\n"
     "(1.0e17 FLOP at 33.8 GFLOPS, no CUDA), so it ran on a Modal A10G."),

    ("Shards are **12,000 rows**, not the 40,000 the five use. The Modal parts *are* the shards; "
     "re-sharding would have written a second 5.5 GB copy on a volume with ~12 GB free in order "
     "to change a number `pointer_resolver` reads per-store from `POINTER_INDEX.json` and that "
     "cannot affect a single vector.",
     "Shards are **12,000 rows**, not the 40,000 the five use. The Modal parts *are* the shards;\n"
     "re-sharding would have written a second 5.5 GB copy on a volume with ~12 GB free, in order\n"
     "to change a number `pointer_resolver` reads per-store from `POINTER_INDEX.json` and that\n"
     "cannot affect a single vector.\n"
     "\n"
     "Those four stores were then content-hashed and scanned row by row — all 3,592,540 rows,\n"
     "**0 bad** (`WEBQSP_ENCODING_SHARD_HASHES.json`). Do not use the `webqsp/*` keys in\n"
     "`ENCODING_SHARD_HASHES.json` for this; they describe a superseded tree. See §3b.8."),

    ("The dense **query** prefix was compared by value — not by eye — against "
     "`src/experiments/canonical_encode.py`, the script that encoded the five, and matches "
     "(`sha256 df4b2898bf22e00b…`, length 92). Queries encoded without that prefix would be the "
     "right shape, dtype and norm, and silently incomparable; SPLADE takes no prefix and the "
     "Modal script raises if handed one.",
     "The dense **query** prefix was compared by value — not by eye — against\n"
     "`src/experiments/canonical_encode.py`, the script that encoded the five, and matches\n"
     "(`sha256 df4b2898bf22e00b…`, length 92). Queries encoded without that prefix would be the\n"
     "right shape, dtype and norm, and silently incomparable; SPLADE takes no prefix and the\n"
     "Modal script raises if handed one."),

    # the §9 heading still promises only "what is not done"
    ("## 9. What is not done",
     "## 9. What is not done, and dataset 6 in full"),
]

for old, new in REPS:
    if old not in s:
        sys.exit("anchor not found: %r" % old[:70])
    s = s.replace(old, new, 1)

io.open(H, "w", encoding="utf-8", newline=chr(10)).write(s)
long_lines = [i + 1 for i, ln in enumerate(s.split(chr(10)))
              if len(ln) > 110 and not ln.lstrip().startswith("|")]
print("limits folded in, paragraphs rewrapped")
print("remaining non-table lines over 110 chars: %s" % (long_lines or "none"))
