"""Verify the downloaded raw mirror against the digests locked BEFORE transfer.

    python scratchpad/final_canonical_build/webqsp_v1/v3_verify_raw_mirror.py

Compares against V3_RAW_MIRROR_PROVENANCE_LOCK.json, which was written before a byte was fetched.
The .part suffix is only dropped on success, so a failed or partial transfer can never be mistaken
for a verified artifact by anything downstream.

Same standard check 1 applied to the IDIR package: a digest mismatch means the delivered bytes are
not the published artifact, and every downstream number would inherit that. On failure this records
the failure and stops; it does not repair, retry or rationalise.
"""
import hashlib, json, os, time, zlib

RAW = "data/final_canonical/freebase_v3/_acquisition/raw"
PART = f"{RAW}/freebase-rdf-latest.gz.part"
FINAL = f"{RAW}/freebase-rdf-latest.gz"
LOCK = "data/final_canonical/freebase_v3/V3_RAW_MIRROR_PROVENANCE_LOCK.json"
OUT = "data/final_canonical/freebase_v3/V3_RAW_MIRROR_ACQUISITION.json"


def main():
    t0 = time.time()
    lock = json.load(open(LOCK, encoding="utf-8"))["TARGET"]
    src = FINAL if os.path.exists(FINAL) else PART
    if not os.path.exists(src):
        raise SystemExit(f"no download found at {PART}")

    size = os.path.getsize(src)
    print(f"[size] {size:,} expected {lock['size_bytes']:,} "
          f"match={size == lock['size_bytes']}", flush=True)

    md5, sha1, sha256 = hashlib.md5(), hashlib.sha1(), hashlib.sha256()
    crc = 0
    done = 0
    with open(src, "rb") as fh:
        while True:
            b = fh.read(1 << 24)
            if not b:
                break
            md5.update(b); sha1.update(b); sha256.update(b)
            crc = zlib.crc32(b, crc)
            done += len(b)
            if done % (1 << 32) < (1 << 24):
                print(f"  hashed {done/1e9:.1f} GB t={time.time()-t0:.0f}s", flush=True)

    ok_size = size == lock["size_bytes"]
    ok_md5 = md5.hexdigest() == lock["md5_expected"]
    ok_sha1 = sha1.hexdigest() == lock["sha1_expected"]
    crc_hex = f"{crc & 0xffffffff:08x}"
    ok_crc = crc_hex == lock["crc32_expected"]
    passed = ok_size and ok_md5 and ok_sha1 and ok_crc

    doc = {
        "schema": "V3_RAW_MIRROR_ACQUISITION/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_url": lock["url"],
        "bytes_delivered": size,
        "bytes_expected": lock["size_bytes"],
        "bytes_match": ok_size,
        "md5_delivered": md5.hexdigest(),
        "md5_expected": lock["md5_expected"],
        "md5_match": ok_md5,
        "sha1_delivered": sha1.hexdigest(),
        "sha1_expected": lock["sha1_expected"],
        "sha1_match": ok_sha1,
        "crc32_delivered": crc_hex,
        "crc32_expected": lock["crc32_expected"],
        "crc32_match": ok_crc,
        "sha256_delivered": sha256.hexdigest(),
        "sha256_note": "recorded by us. The Internet Archive publishes MD5/SHA1/CRC32 only; a "
                       "modern digest lets later builds verify without re-fetching 31.3 GB.",
        "VERIFIED": passed,
        "verified_against": "digests locked in V3_RAW_MIRROR_PROVENANCE_LOCK.json before transfer "
                            "began, not read off the delivered file.",
        "custody_caveat_still_applies": "these digests prove transfer fidelity from the Internet "
                                        "Archive. They do not prove the IA copy equals what Google "
                                        "published; Google's endpoint is 403. The corroboration is "
                                        "IDIR, per the lock's MITIGATION clause -- and it is not "
                                        "discharged until PASS B reproduces IDIR's retained triples.",
        "elapsed_s": round(time.time() - t0, 1),
    }
    if passed:
        os.replace(src, FINAL)
        doc["local_path"] = FINAL
        doc["renamed_from_part"] = True
        doc["NEXT"] = ("PASS A -- stream schema + metadata from gzip. The source is never "
                       "expanded to ~250 GB uncompressed (contract clause SOURCE_EXPANSION).")
    else:
        doc["local_path"] = src
        doc["renamed_from_part"] = False
        doc["VERDICT"] = ("FAILED. The .part suffix is retained so nothing downstream can mistake "
                          "this for a verified artifact. Do not start PASS A. Either resume the "
                          "transfer (curl -C - against the same .part) or re-fetch; do not proceed "
                          "on unverified bytes.")

    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({k: doc[k] for k in ("bytes_match", "md5_match", "sha1_match",
                                          "crc32_match", "VERIFIED",
                                          "sha256_delivered", "local_path")}, indent=1))


if __name__ == "__main__":
    main()
