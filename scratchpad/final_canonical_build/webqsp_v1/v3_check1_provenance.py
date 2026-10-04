"""Check 1: byte provenance, then list the archive without extracting it.

    python scratchpad/final_canonical_build/webqsp_v1/v3_check1_provenance.py

Run once the download reports complete. Verifies the delivered bytes against Zenodo's published
MD5, records a SHA256 of our own, and enumerates the archive so the selective-extraction decision
is made against a real manifest rather than an assumption about layout.

Extracts NOTHING. Disk headroom is the reason the extraction must be selective, so the listing has
to come first.
"""
import hashlib, json, os, time, zipfile
from collections import defaultdict

ZIP = "data/final_canonical/freebase_v3/_acquisition/idir/idirlab-freebases.zip"
OUT = "data/final_canonical/freebase_v3/V3_ACQUISITION_RECORD.json"

EXPECTED_BYTES = 14148416296
EXPECTED_MD5 = "170689b7aad9f029566a4deb36605b01"
URL = "https://zenodo.org/api/records/7909511/files/idirlab-freebases.zip/content"
RECORD = "https://zenodo.org/records/7909511"


def digest(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    n = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 22)
            if not b:
                break
            n += len(b)
            md5.update(b)
            sha.update(b)
    return md5.hexdigest(), sha.hexdigest(), n


def main():
    t0 = time.time()
    size = os.path.getsize(ZIP)
    print(f"[size] {size:,} expected {EXPECTED_BYTES:,}", flush=True)
    md5, sha, n = digest(ZIP)
    print(f"[digest] md5={md5} t={time.time()-t0:.0f}s", flush=True)

    ok_bytes = size == EXPECTED_BYTES
    ok_md5 = md5 == EXPECTED_MD5

    doc = {
        "schema": "V3_ACQUISITION_RECORD/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "check": "1 of 7 -- byte provenance",
        "source_record": RECORD,
        "source_url": URL,
        "local_path": ZIP,
        "bytes_delivered": size,
        "bytes_expected": EXPECTED_BYTES,
        "bytes_match": ok_bytes,
        "md5_delivered": md5,
        "md5_published_by_zenodo": EXPECTED_MD5,
        "md5_match": ok_md5,
        "sha256_delivered": sha,
        "sha256_note": "Zenodo publishes MD5 only. SHA256 is recorded by us so later builds can "
                       "verify against a modern digest without re-fetching 14 GB.",
        "CHECK_1_PASSED": bool(ok_bytes and ok_md5),
        "nothing_extracted": True,
    }

    if not doc["CHECK_1_PASSED"]:
        doc["VERDICT"] = ("FAILED. Do not extract and do not proceed to checks 2-7: a byte or "
                          "digest mismatch means the delivered archive is not the published "
                          "artifact, and every downstream number would inherit that.")
        tmp = OUT + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=2)
        os.replace(tmp, OUT)
        print(json.dumps(doc, indent=1))
        return

    # ---- listing only ----
    with zipfile.ZipFile(ZIP) as z:
        infos = z.infolist()
    top = defaultdict(lambda: {"entries": 0, "compressed": 0, "uncompressed": 0})
    total_u = total_c = 0
    for i in infos:
        head = i.filename.split("/")[0] if "/" in i.filename else "(root)"
        t = top[head]
        t["entries"] += 1
        t["compressed"] += i.compress_size
        t["uncompressed"] += i.file_size
        total_u += i.file_size
        total_c += i.compress_size

    free = None
    try:
        st = os.statvfs(os.path.dirname(ZIP))
        free = st.f_bavail * st.f_frsize
    except (AttributeError, OSError):
        import shutil as _sh
        free = _sh.disk_usage(os.path.dirname(ZIP)).free

    doc["ARCHIVE_LISTING"] = {
        "entries": len(infos),
        "total_uncompressed_bytes": total_u,
        "total_uncompressed_gb": round(total_u / 1e9, 2),
        "total_compressed_bytes": total_c,
        "top_level": {k: {**v, "uncompressed_gb": round(v["uncompressed"] / 1e9, 2)}
                      for k, v in sorted(top.items(), key=lambda kv: -kv[1]["uncompressed"])},
        "largest_entries": [
            {"name": i.filename, "uncompressed_gb": round(i.file_size / 1e9, 3)}
            for i in sorted(infos, key=lambda x: -x.file_size)[:30]
        ],
    }
    doc["DISK_DECISION"] = {
        "free_bytes_now": free,
        "free_gb_now": round(free / 1e9, 1),
        "full_extraction_would_need_gb": round(total_u / 1e9, 2),
        "full_extraction_fits": bool(free is not None and total_u < free),
        "policy": "extract SELECTIVELY regardless of whether it fits: FB+CVT-REV, the "
                  "mapping/support files, the reverse-property metadata, and the name/type/property "
                  "maps. The other three variants are not needed for checks 2-7 and the raw mirror "
                  "stays untouched.",
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({k: doc[k] for k in ("CHECK_1_PASSED", "md5_match", "bytes_match",
                                          "sha256_delivered", "DISK_DECISION")}, indent=1))
    print(json.dumps(doc["ARCHIVE_LISTING"]["top_level"], indent=1))


if __name__ == "__main__":
    main()
