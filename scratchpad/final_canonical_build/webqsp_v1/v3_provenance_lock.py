"""Provenance-lock the Internet Archive mirror BEFORE downloading it.

    python scratchpad/final_canonical_build/webqsp_v1/v3_provenance_lock.py

Records the expected digests, size and custody facts first, so verification after transfer is a
check against a value fixed in advance rather than a value read off the thing being checked.

The honest limitation, recorded rather than buried: this is a THIRD-PARTY CUSTODY copy. Google's
own endpoint returns 403 AccessDenied, so there is no upstream digest to compare against. The
archive.org MD5/SHA1 attest to what the Internet Archive has held since 2015-06-30; they cannot
attest that it equals what Google published. What CAN corroborate that is IDIR: an independent 2023
preprocessing of the same dump. If our raw build reproduces IDIR's 134,142,730 retained triples at
the 99.998% agreement rate check 4 measured, that is strong independent evidence the mirror is the
dataset it claims to be. Provenance here is a chain plus a cross-check, not a signature.
"""
import json, os, subprocess, time

OUT = "data/final_canonical/freebase_v3/V3_RAW_MIRROR_PROVENANCE_LOCK.json"
META = "scratchpad/final_canonical_build/webqsp_v1/_ia_metadata.json"
URL = "https://archive.org/download/freebase-rdf-latest/freebase-rdf-latest.gz"

meta = json.load(open(META, encoding="utf-8"))
files = {f["name"]: f for f in meta.get("files", [])}
tgt = files["freebase-rdf-latest.gz"]

head = subprocess.run(["curl", "-sIL", "--max-time", "90", URL],
                      capture_output=True, text=True).stdout
hdr = {}
for line in head.splitlines():
    if ":" in line:
        k, v = line.split(":", 1)
        k = k.strip().lower()
        if k in ("content-length", "accept-ranges", "etag", "last-modified", "content-type"):
            hdr[k] = v.strip()

doc = {
    "schema": "V3_RAW_MIRROR_PROVENANCE_LOCK/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": "LOCKED BEFORE TRANSFER",

    "TARGET": {
        "url": URL,
        "identifier": meta["metadata"]["identifier"],
        "archive_org_server": meta.get("server"),
        "archive_org_dir": meta.get("dir"),
        "size_bytes": int(tgt["size"]),
        "size_gb": round(int(tgt["size"]) / 1e9, 2),
        "md5_expected": tgt["md5"],
        "sha1_expected": tgt["sha1"],
        "crc32_expected": tgt["crc32"],
        "source_mtime_epoch": int(tgt["mtime"]),
        "source_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(tgt["mtime"]))),
    },

    "LIVE_HEAD_PROBE": {
        **hdr,
        "content_length_matches_metadata":
            hdr.get("content-length") == tgt["size"],
        "range_requests_supported": hdr.get("accept-ranges") == "bytes",
        "range_probe": "HTTP 206 for bytes 0-63; payload begins 1f8b08 (gzip, deflate). Verified "
                       "before the lock was written, so the download can resume after "
                       "interruption rather than restarting 31.3 GB.",
    },

    "CUSTODY": {
        "uploader": meta["metadata"].get("uploader"),
        "addeddate": meta["metadata"].get("addeddate"),
        "chain": "Google published freebase-rdf-latest.gz; the Internet Archive item was created "
                 "2015-06-30 and has held it since. Our copy will come from the Internet Archive.",
        "LIMITATION": "third-party custody. Google's own endpoint returns 403 AccessDenied "
                      "(verified, recorded in V3_SOURCE_ASSESSMENT.json), so there is NO upstream "
                      "digest to verify against. The MD5/SHA1 below attest to what the Internet "
                      "Archive holds, not that it equals what Google published.",
        "MITIGATION": "IDIR is an independent 2023 preprocessing of the same dump. Check 4 measured "
                      "99.998% exact structural agreement on retained predicates with zero "
                      "reverse-only cases. If the raw build reproduces IDIR's 134,142,730 triples "
                      "at that rate, the mirror is corroborated by a party that never saw our copy.",
        "this_is_a_chain_plus_a_cross_check": "not a signature. Recorded as such so no later "
                                              "document can cite this lock as proof of authenticity "
                                              "it does not provide.",
    },

    "OTHER_FILES_IN_THE_ITEM": {
        n: {"size": f.get("size"), "md5": f.get("md5")}
        for n, f in files.items() if n != "freebase-rdf-latest.gz"
    },
    "NOTE_ON_OTHER_FILES": "fb2w.nt.gz (22 MB) is the Freebase-to-Wikidata mapping and "
                           "deleted_freebase.tar.gz (2.2 GB) covers deleted entities. Neither is "
                           "needed for the contract as frozen. Recorded so a later need does not "
                           "require re-discovering the item.",

    "VERIFICATION_PLAN": {
        "on_completion": ["byte count == 31,305,093,084",
                          "md5 == 38d8e9cb7a12634e46a0c52c4ee8dd52",
                          "sha1 == d1950f3c966932358a79a02cd1d56ae3165b26f4"],
        "if_any_fails": "do not proceed to PASS A. A digest mismatch means the delivered bytes are "
                        "not the published artifact and every downstream number would inherit that. "
                        "Same rule check 1 applied to the IDIR package.",
        "gzip_integrity": "digest match proves transfer fidelity. Stream-decodability is proven "
                          "separately by PASS A reaching EOF, which is the same standard FINDING_4 "
                          "used on object_types.csv.",
    },
    "DISK_POLICY": "the source is never expanded to its documented ~250 GB uncompressed form. "
                   "Every pass streams from gzip. Contract clause SOURCE_EXPANSION.",
}

tmp = OUT + ".tmp"
with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(doc, fh, indent=2)
os.replace(tmp, OUT)
print(json.dumps({"TARGET": doc["TARGET"], "LIVE_HEAD_PROBE": doc["LIVE_HEAD_PROBE"]}, indent=1))
