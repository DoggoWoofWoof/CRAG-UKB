# -*- coding: utf-8 -*-
"""
Add the eighth misread and register the new records in the layout tree.

The eighth is the one found last and is the sharpest of the set: ENCODING_SHARD_HASHES.json
has four keys that begin "webqsp/" and they describe a DIFFERENT corpus. Someone verifying
dataset 6 against them reads a DAMAGED verdict for a tree dataset 6 does not use, on files it
never touches, and concludes the sixth dataset is corrupt. It isn't — it was scanned row by row
and every one of its 3,592,540 rows is clean.
"""
import io
import json
import sys

ROOT = "data/final_canonical"
H = ROOT + "/HANDOFF.md"
we = json.load(io.open(ROOT + "/WEBQSP_ENCODING_SHARD_HASHES.json", encoding="utf-8"))
rows = sum(v["rows_scanned"] for v in we["channels"].values())

s = io.open(H, encoding="utf-8").read()

REPS = [
    ("## 3b. Seven ways this package gets misread",
     "## 3b. Eight ways this package gets misread"),

    ("7. **Running a builder without `PYTHONHASHSEED=0`.** Every script refuses rather than\n"
     "   producing an artifact that will not reproduce — but only because they check. If you write\n"
     "   your own, check it too.",
     "7. **Running a builder without `PYTHONHASHSEED=0`.** Every script refuses rather than\n"
     "   producing an artifact that will not reproduce — but only because they check. If you write\n"
     "   your own, check it too.\n"
     "\n"
     "8. **Verifying dataset 6's vectors against `ENCODING_SHARD_HASHES.json`.** That record has\n"
     "   four keys beginning `webqsp/`, and every one of them describes\n"
     "   `data/canonical/webqsp/encodings/…` — the **superseded** Phase-C webqsp tree from an\n"
     "   earlier KB build, which it honestly marks `row_integrity: DAMAGED`, 12,814 damaged rows.\n"
     "   Dataset 6 does not use one byte of that tree; it resolves into\n"
     "   `data/canonical/webqsp_rog_v1/encodings/…`. Verify it against\n"
     "   `WEBQSP_ENCODING_SHARD_HASHES.json`, which fingerprints the real stores and reports the\n"
     "   exhaustive row scan of all %s of them: **0 bad rows**. The stale record was left\n"
     "   unedited on purpose — `LOCKED_5_OF_5` hashes it, so amending it would break that freeze." % format(rows, ",")),

    ("  _WEBQSP_ACCEPTANCE_GATE.json              dataset 6's acceptance criteria and their verdicts\n",
     "  _WEBQSP_ACCEPTANCE_GATE.json              dataset 6's acceptance criteria and their verdicts\n"
     "  WEBQSP_ENCODING_SHARD_HASHES.json         dataset 6's vector shards: hashes + exhaustive scan\n"
     "                                            (NOT the webqsp/* keys in ENCODING_SHARD_HASHES,\n"
     "                                            which describe a superseded tree — see 3b.8)\n"
     "  FROZEN_ARTIFACT_RECHECK.json              every frozen artifact re-hashed from disk\n"
     "  DIVERGED_ARTIFACT_EXPLANATION.json        the two that moved, and the proof they only grew\n"
     "  HANDOFF_CLAIMS_CHECK.json                 this document, checked against the data\n"),
]

for old, new in REPS:
    if old not in s:
        sys.exit("anchor not found: %r" % old[:70])
    s = s.replace(old, new, 1)

io.open(H, "w", encoding="utf-8", newline=chr(10)).write(s)
print("eighth misread added; %d layout entries registered" % 5)
