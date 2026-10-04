"""Recompute the per-variant size consistency with the unwrapped sizes, closing FINDING_2.

Appends. FINDING_2's original entry stays exactly as written: it is the record of what the size
fields alone supported before the streams were read, and of the fact that it declined to guess a
cause. The correction sits beside it, not over it.
"""
import json, os, time

REC = "data/final_canonical/freebase_v3/V3_EXTRACTION_RECORD.json"

PUBLISHED = {"FB+CVT-REV": 134213735, "FB-CVT-REV": 125124274,
             "FB+CVT+REV": 244112599, "FB-CVT+REV": 238981274}

# central-directory values, with the three measured true sizes substituted in. Only train.txt
# wrapped in the +REV variants; test/valid were already correct and CRC-verified.
TRUE = {
    "FB+CVT-REV": {"train": 2527440120, "test": 139216099, "valid": 139207173},
    "FB-CVT-REV": {"train": 2234320200, "test": 124133619, "valid": 124127911},
    "FB+CVT+REV": {"train": 4781675728, "test": 289953152, "valid": 289947137},
    "FB-CVT+REV": {"train": 4306501986, "test": 239253246, "valid": 239263837},
}

doc = json.load(open(REC, encoding="utf-8"))
per = {}
for v, n in PUBLISHED.items():
    b = sum(TRUE[v].values())
    per[v] = {"published_triples": n, "true_triple_file_bytes": b,
              "bytes_per_triple": round(b / n, 2),
              "train_declared": None, "train_true": TRUE[v]["train"]}
per["FB+CVT+REV"]["train_declared"] = 486708432
per["FB-CVT+REV"]["train_declared"] = 11534690
per["FB+CVT-REV"]["train_declared"] = 2527440120
per["FB-CVT-REV"]["train_declared"] = 2234320200

vals = [p["bytes_per_triple"] for p in per.values()]
doc["FINDING_2_CLOSED_BY_FINDING_4"] = {
    "status": "CLOSED. The inconsistency was an artifact of the wrapped size fields, not of the "
              "packaging.",
    "recomputed_with_true_sizes": per,
    "bytes_per_triple_range": [min(vals), max(vals)],
    "reading": "all four variants now land in a %.1f-%.1f bytes/triple band. Before the wrap was "
               "undone, FB+CVT+REV read as 4.4 and FB-CVT+REV as 2.1, which is what made them look "
               "truncated. Each of the two train.txt streams was inflated to its own deflate EOF "
               "and its CRC matched the central directory exactly, so the true sizes are verified "
               "rather than inferred." % (min(vals), max(vals)),
    "what_this_changes_for_CRAG": "nothing about the backbone, which was always one of the two "
                                  "consistent variants. It does mean FB+CVT+REV is usable if the "
                                  "reverse-orientation question ever sends us there, which "
                                  "FINDING_2 had implicitly ruled out.",
    "method_note": "the original FINDING_2 entry is retained verbatim above. It recorded what the "
                   "size fields supported at the time and explicitly declined to assert a cause "
                   "('do not guess; it does not block the backbone'). That restraint is why this "
                   "closes cleanly instead of contradicting a claim.",
    "appended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
tmp = REC + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(doc, fh, indent=2)
os.replace(tmp, REC)
print(json.dumps({k: v["bytes_per_triple"] for k, v in per.items()}, indent=1))
print("range", min(vals), max(vals))
