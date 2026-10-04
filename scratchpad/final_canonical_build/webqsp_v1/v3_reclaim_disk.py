"""Reclaim disk for the raw mirror by deleting extraction products the oracle now supersedes.

    python scratchpad/final_canonical_build/webqsp_v1/v3_reclaim_disk.py [--commit]

Dry-run by default. Nothing is removed without --commit.

STANDING RULE AND ITS OVERRIDE. The project rule is supersede-don't-delete: move superseded things
into a clearly named subdirectory rather than removing them. The user has overridden it for this
specific class -- "delete expendable extracted variants/temporary repair products" -- and the
override is sound here for a reason worth stating: these files are not history, they are a cache.
Every one is re-derivable in minutes from idirlab-freebases.zip, whose bytes and MD5 match Zenodo
exactly and whose repaired offsets are cached. Deleting a verified-regenerable cache loses no
record. The ZIP itself is NOT deleted, precisely because it is what makes the rest reversible.

PRECONDITIONS, all checked before anything is removed:
  1. the ZIP is present and its byte count still matches the value check 1 verified
  2. recovered_offsets.json is present and covers every entry being deleted
  3. every oracle Parquet opens and its row count matches the source it distilled
Any failure aborts with nothing removed.
"""
import json, os, sys, time
import pyarrow.parquet as pq

ROOT = "data/final_canonical/freebase_v3/_acquisition/idir"
ZIP = f"{ROOT}/idirlab-freebases.zip"
SRC = f"{ROOT}/extracted/idirlab-freebases"
ORACLE = f"{ROOT}/oracle"
OFF = f"{ROOT}/recovered_offsets.json"
QUAR = f"{ROOT}/_quarantine_failed_crc"
REC = "data/final_canonical/freebase_v3/V3_DISK_RECLAMATION.json"

ZIP_BYTES = 14148416296

# (path, why it goes, what replaces it)
DELETE = [
    (f"{SRC}/Metadata/object_types.csv", "distilled", "oracle/mid_type.parquet + schema_type.parquet"),
    (f"{SRC}/Metadata/object_names.csv", "distilled", "oracle/mid_name.parquet + schema_name.parquet"),
    (f"{SRC}/Metadata/entities_id_label.csv", "distilled", "oracle/entity_label.parquet"),
    (f"{SRC}/FB+CVT-REV/entity2id.txt", "distilled", "oracle/backbone_entity2id.parquet"),
    (f"{SRC}/FB+CVT-REV/train.txt", "distilled", "oracle/backbone_triples.parquet"),
    (f"{SRC}/FB+CVT-REV/test.txt", "distilled", "oracle/backbone_triples.parquet"),
    (f"{SRC}/FB+CVT-REV/valid.txt", "distilled", "oracle/backbone_triples.parquet"),
    (f"{SRC}/FB+CVT+REV/entity2id.txt", "purpose served", "V3_WHY_CVTS_DROPPED.json"),
    (f"{SRC}/FB+CVT+REV/train.txt", "purpose served", "V3_WHY_CVTS_DROPPED.json"),
    (f"{SRC}/FB+CVT+REV/test.txt", "purpose served", "V3_WHY_CVTS_DROPPED.json"),
    (f"{SRC}/FB+CVT+REV/valid.txt", "purpose served", "V3_WHY_CVTS_DROPPED.json"),
    (f"{QUAR}/object_types.csv.PARTIAL_CRC_FAILED", "superseded repair product",
     "the full CRC-verified extraction, FINDING_4"),
]

# row counts each oracle table must reproduce, from the sources they distilled
EXPECT = {
    "backbone_entity2id.parquet": 101917083,
    "backbone_triples.parquet": 134142730,
    "backbone_relation2id.parquet": 2641,
    "entity_label.parquet": 47176593,
    "mid_name.parquet": 72537217,
    "mid_type.parquet": 266262349,
}

KEEP_NOTE = {
    "idirlab-freebases.zip": "provenance anchor AND the compressed store every deletion here is "
                             "reversible from. Never delete.",
    "recovered_offsets.json": "the repair key. Without it re-extraction costs an 81-second "
                              "signature scan of 14 GB; with it, nothing.",
    "oracle/*.parquet": "IDIR's frozen role, in the form that role actually needs.",
    "Metadata/object_ids.csv": "162 MB, already small",
    "Metadata/{properties,types,domains}_id_label.csv": "under 1 MB total; the TypeSystem keys",
    "TypeSystem/freebase_endtypes.csv": "48 KB; property type signatures",
    "*/relation2id.txt": "all four variants, under 1 MB total; the reverse-split evidence",
}


def main():
    commit = "--commit" in sys.argv
    fail = []

    if not os.path.exists(ZIP):
        fail.append("ZIP missing")
    elif os.path.getsize(ZIP) != ZIP_BYTES:
        fail.append(f"ZIP size {os.path.getsize(ZIP)} != verified {ZIP_BYTES}")
    if not os.path.exists(OFF):
        fail.append("recovered_offsets.json missing")
    else:
        off = json.load(open(OFF, encoding="utf-8"))
        for p, _, _ in DELETE:
            if p.startswith(SRC):
                rel = "idirlab-freebases/" + os.path.relpath(p, SRC).replace(os.sep, "/")
                if rel not in off:
                    fail.append(f"no cached offset for {rel}")

    checked = {}
    for f, want in EXPECT.items():
        p = f"{ORACLE}/{f}"
        if not os.path.exists(p):
            fail.append(f"oracle missing {f}")
            continue
        try:
            got = pq.read_metadata(p).num_rows
        except Exception as ex:
            fail.append(f"oracle unreadable {f}: {ex}")
            continue
        checked[f] = {"rows": got, "expected": want, "match": got == want}
        if got != want:
            fail.append(f"row mismatch {f}: {got} != {want}")

    plan, total = [], 0
    for p, why, repl in DELETE:
        if os.path.exists(p):
            b = os.path.getsize(p)
            total += b
            plan.append({"path": p, "bytes": b, "gb": round(b / 1e9, 3),
                         "reason": why, "superseded_by": repl})
        else:
            plan.append({"path": p, "bytes": 0, "reason": why, "already_absent": True})

    free_before = __import__("shutil").disk_usage(".").free
    print(f"preconditions: {'FAILED' if fail else 'PASSED'}")
    for f in fail:
        print("  !", f)
    print(f"would delete {len(plan)} paths, {total/1e9:.2f} GB")
    for x in plan:
        if x.get("bytes"):
            print(f"  {x['gb']:>7.3f} GB  {x['reason']:<26} {x['path']}")
    print(f"free now {free_before/1e9:.1f} GB -> would be {(free_before+total)/1e9:.1f} GB")

    if fail:
        print("\nABORT: preconditions failed, nothing removed.")
        return
    if not commit:
        print("\nDRY RUN. re-run with --commit to delete.")
        return

    removed = []
    for x in plan:
        if x.get("already_absent"):
            continue
        os.remove(x["path"])
        removed.append(x)
    for d in (f"{SRC}/FB+CVT+REV", QUAR):
        try:
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
        except OSError:
            pass

    free_after = __import__("shutil").disk_usage(".").free
    doc = {
        "schema": "V3_DISK_RECLAMATION/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "why": "the raw mirror needs headroom for source + edge shards + sort spills + canonical "
               "outputs coexisting. 46 GB free was too tight for a 31.3 GB source.",
        "STANDING_RULE_OVERRIDE": {
            "the_rule": "supersede-don't-delete: move superseded artifacts into a clearly named "
                        "subdirectory rather than removing them.",
            "overridden_by": "the user, explicitly, for extraction products and temporary repair "
                             "products.",
            "why_it_is_sound_here": "these files are a cache, not history. Every one is re-derivable "
                                    "from idirlab-freebases.zip, which is retained and whose bytes "
                                    "and MD5 match Zenodo exactly. Deleting a verified-regenerable "
                                    "cache loses no record. Nothing that constitutes evidence -- no "
                                    "measurement, no check record, no hash -- is touched.",
            "what_was_NOT_deleted": KEEP_NOTE,
        },
        "PRECONDITIONS_CHECKED": {
            "zip_present_and_size_matches_check_1": True,
            "zip_bytes": ZIP_BYTES,
            "offsets_cached_for_every_deleted_entry": True,
            "oracle_row_counts_match_sources": checked,
        },
        "deleted": removed,
        "deleted_bytes": sum(x["bytes"] for x in removed),
        "deleted_gb": round(sum(x["bytes"] for x in removed) / 1e9, 2),
        "free_before_gb": round(free_before / 1e9, 1),
        "free_after_gb": round(free_after / 1e9, 1),
        "REGENERATION": {
            "how": "python scratchpad/final_canonical_build/webqsp_v1/v3_extract_true_size.py with "
                   "the WRITE list set to the wanted entries. It reads recovered_offsets.json, "
                   "inflates each stream to its own deflate EOF and CRC-verifies against the "
                   "central directory.",
            "cost": "object_types.csv 66s; the FB+CVT+REV set about 62s; everything else faster.",
            "caveat": "the distillation logged small unparsed residues -- 794 lines of "
                      "entities_id_label.csv and 677 of object_names.csv, 0 of object_types.csv, "
                      "out of 386M lines total. If those ever need inspecting, re-extract; the "
                      "Parquet does not contain them.",
        },
        "elapsed_note": "oracle distillation took 1138s and is recorded in "
                        "V3_ORACLE_DISTILLATION.json.",
    }
    tmp = REC + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REC)
    print(f"\ndeleted {doc['deleted_gb']} GB; free {doc['free_before_gb']} -> {doc['free_after_gb']} GB")


if __name__ == "__main__":
    main()
