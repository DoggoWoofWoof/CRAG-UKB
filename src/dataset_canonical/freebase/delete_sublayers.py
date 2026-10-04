# -*- coding: utf-8 -*-
"""Delete the superseded freebase_v3 sub-layer DATA files (the served tree data/final_canonical/freebase/ carries their
content: names with provenance, semantic text, edges, metadata), keep every record / report (*.json) in place, and log it.

Authorisation (user, 2026-09-13): "delete the superseded freebase_v3 sub-layers and free up space and completely finish the
freebase so that we can freeze it completely".

Two phases, so nothing is removed on the strength of a listing:
    python src/dataset_canonical/freebase/delete_sublayers.py --census   # hash every candidate file, match against the V3
                                                                         # records' pins, write the log with status PLANNED
    python src/dataset_canonical/freebase/delete_sublayers.py --apply    # re-hash each file, delete only on an exact match
                                                                         # with the census, rewrite the log with status APPLIED
Log: data/final_canonical/_history/logs/FREEBASE_V3_SUBLAYER_DELETION.json (self-hashed record).
Not touched, by the standing rules: _acquisition/ (raw mirror, IDIR zip, FACC1, every recovery source), canonical/ (the frozen
V3 graph the served tree was built from), inference_admissibility_v1/, and every *.json anywhere under freebase_v3/.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import re
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
FC = "data/final_canonical"
FV = FC + "/freebase_v3"
LOG = FC + "/_history/logs/FREEBASE_V3_SUBLAYER_DELETION.json"
LAYERS = {
    "overlay_v1": "CRAG_FREEBASE_RESOLUTION_OVERLAY_V1 (262 parquet): every node's resolved display name + source; carried into "
                  "freebase/nodes/ as name / name_kind / name_source (OVERLAY_MANIFEST.json kept)",
    "inferred_name_v1": "the inferred-name layer (33 parquet, LOCKED_FREEBASE_INFERENCE_V1): carried as name_kind INFERRED",
    "inference_overlay_v1": "the inference overlay (32 parquet, LOCKED_FREEBASE_INFERENCE_V1): carried as name_rule / floor reasons",
    "semantic_v1": "K=6 role text per node (262 parquet): carried as freebase/nodes/ column semantic_text",
    "pass_a": "PASS A per-member extraction shards (200 .gz; the 201 *_meta.json reports kept)",
    "pass_b": "PASS B mediator / literal tables (1 parquet + 1 npy; the 200 per-member reports kept)",
    "pass_c": "PASS C reclamation array (1 npy)",
    "pass_d": "PASS D canonical-edge bucket sample (1 parquet; reverse_map.json kept)",
    "probe": "source-structure probe tables (7 parquet)",
    "pass_a_probe": "PASS A probe tables (10 parquet)",
}
KEEP_SUFFIX = (".json",)


def sha_file(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(bs), b""):
            h.update(chunk)
    return h.hexdigest()


def self_hash(rec):
    """the repo convention (freeze_canonical.record_hash): json.dumps(indent=1) without the digest fields, CRLF and LF"""
    sys.path.insert(0, os.path.join(ROOT, "src", "dataset_canonical"))
    from freeze_canonical import record_hash
    rec["RECORD_SHA256"] = record_hash(rec)
    rec["RECORD_SHA256_LF"] = record_hash(rec, crlf=False)
    return rec


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def known_pins():
    """every sha256 any V3 record pins, with the record that pins it"""
    pins = {}
    recs = sorted(glob.glob(FV + "/*.json")) + sorted(glob.glob(FV + "/*/*.json")) + sorted(glob.glob(FV + "/*/*/*.json"))
    rx = re.compile(r'"(?:sha256|sha256_of_bytes|file_sha256)"\s*:\s*"([0-9a-f]{64})"')
    for r in recs:
        try:
            s = io.open(r, encoding="utf-8").read()
        except Exception:
            continue
        for h in rx.findall(s):
            pins.setdefault(h, os.path.relpath(r, FV).replace(os.sep, "/"))
    return pins


def candidates():
    out = []
    for L in LAYERS:
        d = FV + "/" + L
        if not os.path.isdir(d):
            continue
        for r, _, fs in os.walk(d):
            for x in sorted(fs):
                p = os.path.join(r, x).replace(os.sep, "/")
                if x.lower().endswith(KEEP_SUFFIX):
                    continue
                out.append((L, p))
    return out


def census():
    t0 = time.time()
    pins = known_pins()
    files, by_layer = [], {}
    cand = candidates()
    for i, (L, p) in enumerate(cand):
        b = os.path.getsize(p)
        h = sha_file(p)
        files.append({"layer": L, "file": p, "bytes": b, "sha256": h, "pinned_by": pins.get(h)})
        a = by_layer.setdefault(L, {"files": 0, "bytes": 0, "pinned": 0, "kept_records": 0})
        a["files"] += 1
        a["bytes"] += b
        a["pinned"] += 1 if h in pins else 0
        if (i + 1) % 100 == 0:
            print("  hashed %d / %d (%.0fs)" % (i + 1, len(cand), time.time() - t0))
            sys.stdout.flush()
    for L in LAYERS:
        d = FV + "/" + L
        if os.path.isdir(d):
            by_layer.setdefault(L, {"files": 0, "bytes": 0, "pinned": 0, "kept_records": 0})
            by_layer[L]["kept_records"] = sum(1 for r, _, fs in os.walk(d) for x in fs if x.lower().endswith(KEEP_SUFFIX))
    rec = {
        "RECORD": "FREEBASE_V3_SUBLAYER_DELETION", "status": "PLANNED", "created_utc": utc(),
        "authorisation": "user, 2026-09-13: 'delete the superseded freebase_v3 sub-layers and free up space and completely finish the "
                         "freebase so that we can freeze it completely and go for edge creation or embedding or partitioning'",
        "what": "data files of the freebase_v3 sub-layers whose content the served tree data/final_canonical/freebase/ carries "
                "(DATASET.json d3b9324b..., verified 155/0; CANONICAL_FREEZE eaab0b8f..., verified 679/0). Every *.json record or report "
                "under those directories is kept in place; nothing under _acquisition/, canonical/ or inference_admissibility_v1/ is touched.",
        "layers": {L: dict(LAYERS_WHAT=LAYERS[L], **by_layer.get(L, {})) for L in LAYERS},
        "files_total": len(files), "bytes_total": int(sum(f["bytes"] for f in files)),
        "pinned_total": int(sum(1 for f in files if f["pinned_by"])),
        "pins_searched_in": "every *.json under freebase_v3/ (three levels): sha256 / sha256_of_bytes / file_sha256 fields",
        "not_deleted": ["_acquisition/ (raw mirror 30 GB, IDIR zip 14 GB + oracle/metadata, FACC1, the published-dump recovery sources)",
                        "canonical/ (36 GB; the frozen V3 graph, CANONICAL_MANIFEST.json pinned by freebase/DATASET.json source_layer)",
                        "inference_admissibility_v1/ (3.4 MB)", "every *.json under freebase_v3/"],
        "files": files, "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    io.open(LOG, "w", encoding="utf-8", newline="\n").write(json.dumps(self_hash(rec), ensure_ascii=False, indent=1))
    print("census: %d files, %.2f GB, %d pinned by a V3 record; log %s (%.0fs)" % (
        len(files), rec["bytes_total"] / 1e9, rec["pinned_total"], LOG, rec["seconds"]))
    for L, a in rec["layers"].items():
        print("  %-22s files %4d  %7.2f GB  pinned %4d  records kept %d" % (
            L, a.get("files", 0), a.get("bytes", 0) / 1e9, a.get("pinned", 0), a.get("kept_records", 0)))


def apply():
    t0 = time.time()
    rec = json.load(io.open(LOG, encoding="utf-8"))
    assert rec["status"] == "PLANNED", rec["status"]
    deleted, mismatched, missing = [], [], []
    for i, f in enumerate(rec["files"]):
        p = f["file"]
        if not os.path.exists(p):
            missing.append(p)
            continue
        if os.path.getsize(p) != f["bytes"] or sha_file(p) != f["sha256"]:
            mismatched.append(p)
            continue
        os.remove(p)
        deleted.append(p)
        if (i + 1) % 100 == 0:
            print("  %d / %d (%.0fs)" % (i + 1, len(rec["files"]), time.time() - t0))
            sys.stdout.flush()
    left = {}
    for L in LAYERS:
        d = FV + "/" + L
        if os.path.isdir(d):
            fs = [os.path.join(r, x).replace(os.sep, "/") for r, _, xs in os.walk(d) for x in xs]
            left[L] = {"files_left": len(fs), "non_record_files_left": [x for x in fs if not x.lower().endswith(KEEP_SUFFIX)]}
    dset = set(deleted)
    rec["status"] = "APPLIED" if not mismatched and not missing else "PARTIAL"
    rec["applied_utc"] = utc()
    rec["deleted"] = len(deleted)
    rec["bytes_deleted"] = int(sum(f["bytes"] for f in rec["files"] if f["file"] in dset))
    rec["mismatched_not_deleted"] = mismatched
    rec["missing_at_apply"] = missing
    rec["left_in_place"] = left
    rec["apply_seconds"] = round(time.time() - t0, 1)
    io.open(LOG, "w", encoding="utf-8", newline="\n").write(json.dumps(self_hash(rec), ensure_ascii=False, indent=1))
    print("%s: deleted %d files / %.2f GB; mismatched %d; missing %d (%.0fs)" % (
        rec["status"], len(deleted), rec["bytes_deleted"] / 1e9, len(mismatched), len(missing), rec["apply_seconds"]))
    for L, a in left.items():
        print("  %-22s left %d files, non-record files left %d" % (L, a["files_left"], len(a["non_record_files_left"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.census:
        census()
    elif a.apply:
        apply()
    else:
        ap.error("--census or --apply")
