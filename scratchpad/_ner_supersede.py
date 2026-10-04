"""FBX_SCALE Track A: make the Freebase NER family single-software.

FINALIZE refused (assert: units extracted with different software): 48 units (s000g0 ... s005g7, extracted 2026-09-30 17:55-18:20 UTC) carry extract code_sha256 653b0410..., the other 556 carry 8e096647...
(the script was patched at 18:34 UTC the same day: the MAXCHARS rule, which only cuts a name longer than spaCy's max_length of 1,000,000 characters; nothing else in the extraction path).
This script SUPERSEDES (moves, never deletes) the 48 old units and everything derived from them (group buckets, pair files, aggregated ranges) into <ner>/_history/, so that EXTRACT redoes exactly those 48 with the
current code and GROUP / AGG / FINALIZE rebuild the family.  The old aggregated-range hashes stay in the moved agg_r*.json: when the rebuilt ner_raw files hash identically the two extraction versions are
byte-equivalent end to end (a free check, reported either way).

  python -u scratchpad/_ner_supersede.py [DRY]          # DRY: list what would move
"""
import glob
import io
import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "data", "freebase_scale", "ner")
RECS = os.path.join(REPO, "results", "FREEBASE_SCALE", "ner")
OLD = "653b0410497b99341535650f7aa12ef66b45e19752c38e7831aa436229cebb37"
NEW = "8e096647f9fb4fef3241d363638672e685603b680b10ace7b9175dea23883c54"
HIST = os.path.join(ROOT, "_history", "extract_653b0410")


def main(dry):
    ent = os.path.join(ROOT, "ent")
    old, new = [], []
    for f in sorted(os.listdir(ent)):
        if f.endswith(".json"):
            r = json.load(io.open(os.path.join(ent, f), encoding="utf-8"))
            (old if r["code_sha256"] == OLD else new).append(f[:-5])
            assert r["code_sha256"] in (OLD, NEW), "unexpected extract code hash %s in %s" % (r["code_sha256"], f)
    assert len(old) == 48 and len(new) == 556, (len(old), len(new))
    unit_files = [os.path.join(ent, nm + s) for nm in old for s in (".hi.npy", ".lo.npy", ".pos.npy", ".off.npy", ".json")]
    assert all(os.path.exists(p) for p in unit_files)
    down_dirs = [os.path.join(ROOT, d) for d in ("pairs", "ner_raw") if os.path.isdir(os.path.join(ROOT, d))]
    down_recs = sorted(glob.glob(os.path.join(RECS, "group_b*.json")) + glob.glob(os.path.join(RECS, "agg_r*.json")))
    print("old-code units %d, current-code units %d; unit files %d; derived dirs %s; derived records %d" % (len(old), len(new), len(unit_files), [os.path.basename(d) for d in down_dirs], len(down_recs)))
    if dry:
        return
    assert not os.path.exists(HIST), "already superseded"
    os.makedirs(os.path.join(HIST, "ent"))
    os.makedirs(os.path.join(HIST, "records"))
    moved = []
    for p in unit_files:
        shutil.move(p, os.path.join(HIST, "ent", os.path.basename(p)))
        moved.append(os.path.relpath(p, REPO).replace("\\", "/"))
    for d in down_dirs:
        shutil.move(d, os.path.join(HIST, os.path.basename(d)))
        moved.append(os.path.relpath(d, REPO).replace("\\", "/") + "/")
    for p in down_recs:
        shutil.move(p, os.path.join(HIST, "records", os.path.basename(p)))
        moved.append(os.path.relpath(p, REPO).replace("\\", "/"))
    rec = {"RECORD": "FBX_NER_SUPERSEDE", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "reason": "FINALIZE assert: units extracted with different software (extract code_sha256 653b0410... for 48 units, 8e096647... for 556)",
           "old_extract_code_sha256": OLD, "current_extract_code_sha256": NEW, "superseded_units": old, "moved_to": os.path.relpath(HIST, REPO).replace("\\", "/"), "moved_count": len(moved),
           "known_difference": "the MAXCHARS rule added to extract_names at 2026-09-30 18:34 UTC (a name longer than 1,000,000 characters is cut; shorter names untouched); old units were extracted 17:55-18:20 UTC",
           "check_after_rebuild": "the rebuilt ner_raw/r*.{key,w}.npy sha256 vs the superseded agg_r*.json (in _history/records): equal => the two extraction versions are byte-equivalent end to end"}
    out = os.path.join(RECS, "FBX_NER_SUPERSEDE__653b0410__v1.json")
    assert not os.path.exists(out)
    with io.open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1))
    print("superseded %d units; moved %d paths to %s" % (len(old), len(moved), HIST))


if __name__ == "__main__":
    main("DRY" in sys.argv[1:])
