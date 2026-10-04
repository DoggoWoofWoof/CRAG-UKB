"""Recover the IDIR archive's true local-header offsets, then extract selectively.

    python scratchpad/final_canonical_build/webqsp_v1/v3_repair_and_extract.py

WHY THIS IS NEEDED. The delivered bytes are correct: size and MD5 both match Zenodo's published
values exactly. The archive's CENTRAL DIRECTORY is what is broken -- its local-header offsets point
past the end of the file (max 16,766,275,907 in a 14,148,416,296-byte file). Info-ZIP diagnoses
"12884901888 extra bytes", exactly 3 x 2^32, i.e. a 32-bit offset overflow in whatever wrote the
archive. The inflation is NOT a single constant: subtracting 12 GiB fixes only 25 of 92 entries,
because entries wrapped a different number of times depending on where they sit.

THE RULE, stated generally and not per-file: the compressed payload is intact and every local file
header is present, so true offsets are recovered by scanning the whole file for PK\\x03\\x04
signatures and keeping those whose parsed filename appears in the central directory's name set.
Requiring a filename match makes a false positive inside compressed data effectively impossible.
Sizes, CRCs and compression methods still come from the central directory, which is otherwise sound.

Nothing is guessed and nothing is special-cased. Every extracted file is CRC-verified by zipfile on
read, so a wrong offset cannot pass silently.
"""
import json, os, struct, time, zipfile

ROOT = "data/final_canonical/freebase_v3/_acquisition/idir"
ZIP = f"{ROOT}/idirlab-freebases.zip"
DST = f"{ROOT}/extracted"
REC = "data/final_canonical/freebase_v3/V3_EXTRACTION_RECORD.json"
ACQ = "data/final_canonical/freebase_v3/V3_ACQUISITION_RECORD.json"

WANT = [
    "idirlab-freebases/FB+CVT-REV/train.txt",
    "idirlab-freebases/FB+CVT-REV/test.txt",
    "idirlab-freebases/FB+CVT-REV/valid.txt",
    "idirlab-freebases/FB+CVT-REV/entity2id.txt",
    "idirlab-freebases/FB+CVT-REV/relation2id.txt",
    "idirlab-freebases/FB+CVT+REV/relation2id.txt",
    "idirlab-freebases/FB-CVT-REV/relation2id.txt",
    "idirlab-freebases/FB-CVT+REV/relation2id.txt",
    "idirlab-freebases/Metadata/object_names.csv",
    "idirlab-freebases/Metadata/object_types.csv",
    "idirlab-freebases/Metadata/object_ids.csv",
    "idirlab-freebases/Metadata/entities_id_label.csv",
    "idirlab-freebases/Metadata/properties_id_label.csv",
    "idirlab-freebases/Metadata/types_id_label.csv",
    "idirlab-freebases/Metadata/domains_id_label.csv",
    "idirlab-freebases/TypeSystem/freebase_endtypes.csv",
]

SIG = b"PK\x03\x04"
CHUNK = 1 << 24


def scan_offsets(path, names):
    """-> {filename: true_header_offset}, by signature scan validated against `names`."""
    found = {}
    dup = []
    size = os.path.getsize(path)
    name_bytes = {n.encode("utf-8"): n for n in names}
    maxname = max(len(b) for b in name_bytes) if name_bytes else 0
    tail = b""
    base = 0
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(CHUNK)
            if not buf:
                break
            data = tail + buf
            start = 0
            while True:
                i = data.find(SIG, start)
                if i < 0:
                    break
                abs_off = base - len(tail) + i
                # need 30-byte header + filename
                if i + 30 + maxname > len(data) and len(buf) == CHUNK:
                    break  # resolve in next window
                hdr = data[i:i + 30]
                if len(hdr) == 30:
                    nlen = struct.unpack("<H", hdr[26:28])[0]
                    nm = data[i + 30:i + 30 + nlen]
                    if nm in name_bytes:
                        fn = name_bytes[nm]
                        if fn in found:
                            dup.append((fn, found[fn], abs_off))
                        else:
                            found[fn] = abs_off
                start = i + 1
            keep = 30 + maxname + 4
            tail = data[-keep:] if len(data) > keep else data
            base += len(buf)
            if base % (1 << 30) < CHUNK:
                print(f"  scanned {base/1e9:.1f} GB, found {len(found)}", flush=True)
    return found, dup, size


def main():
    t0 = time.time()
    os.makedirs(DST, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    infos = {i.filename: i for i in z.infolist()}
    reported = {n: i.header_offset for n, i in infos.items()}

    print(f"[scan] {len(infos)} central-directory entries", flush=True)
    found, dup, size = scan_offsets(ZIP, set(infos))
    print(f"[scan] recovered {len(found)}/{len(infos)} offsets t={time.time()-t0:.0f}s", flush=True)

    deltas = {n: reported[n] - found[n] for n in found}
    dist = {}
    for n, d in deltas.items():
        k = d / 2**32
        dist[str(k)] = dist.get(str(k), 0) + 1

    for n, o in found.items():
        infos[n].header_offset = o

    got, failed = [], []
    for w in WANT:
        if w not in found:
            failed.append({"name": w, "reason": "no local header recovered by scan"})
            continue
        try:
            src = z.open(infos[w])           # CRC-verified on read
            out = os.path.join(DST, w)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            n = 0
            with src, open(out, "wb") as fh:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    fh.write(b)
                    n += len(b)
            got.append({"name": w, "bytes": n,
                        "matches_central_directory_size": n == infos[w].file_size})
            print(f"[ok] {w} {n/1e6:.1f} MB t={time.time()-t0:.0f}s", flush=True)
        except Exception as ex:
            failed.append({"name": w, "reason": f"{type(ex).__name__}: {ex}"})
            print(f"[FAIL] {w} {type(ex).__name__}: {ex}", flush=True)

    doc = json.load(open(REC, encoding="utf-8")) if os.path.exists(REC) else {}
    doc.update({
        "schema": "V3_EXTRACTION_RECORD/v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "archive": ZIP,
        "archive_sha256": json.load(open(ACQ, encoding="utf-8"))["sha256_delivered"],
        "destination": DST,

        "FINDING_3_ARCHIVE_CENTRAL_DIRECTORY_IS_CORRUPT": {
            "statement": "the published Zenodo archive cannot be extracted by a conforming reader. "
                         "Our copy is byte-perfect -- size and MD5 both match the published values "
                         "-- so the defect is in the artifact as published, not in the transfer.",
            "evidence": {
                "file_size": size,
                "max_reported_header_offset": max(reported.values()),
                "offset_exceeds_file_by": max(reported.values()) - size,
                "infozip_diagnosis": "12884901888 extra bytes at beginning or within zipfile",
                "that_constant_is": "3 x 2^32 exactly (12 GiB), a 32-bit offset overflow",
                "python_zipfile_error": "BadZipFile: Truncated file header, on every entry",
                "uniform_constant_does_NOT_fix_it": "subtracting 12 GiB validates only 25 of 92 "
                                                    "entries; entries wrapped a differing number "
                                                    "of times by position",
                "delta_distribution_in_multiples_of_2^32": dist,
            },
            "repair_rule": "ignore central-directory offsets; recover true local-header offsets by "
                           "scanning the file for PK\\x03\\x04 and accepting only those whose "
                           "parsed filename is in the central directory's name set. Sizes, CRCs "
                           "and methods still come from the central directory. Every read is "
                           "CRC-verified, so a wrong offset cannot pass silently.",
            "is_a_general_rule_not_a_special_case": True,
            "duplicate_name_offsets": dup,
            "offsets_recovered": len(found),
            "offsets_expected": len(infos),
        },

        "extracted": got,
        "extracted_n": len(got),
        "extracted_gb": round(sum(g["bytes"] for g in got) / 1e9, 2),
        "all_sizes_match_central_directory": all(g["matches_central_directory_size"] for g in got),
        "failed": failed,
        "elapsed_s": round(time.time() - t0, 1),
    })
    tmp = REC + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REC)
    print(json.dumps({k: doc[k] for k in ("extracted_n", "extracted_gb",
                                          "all_sizes_match_central_directory", "failed")}, indent=1))


if __name__ == "__main__":
    main()
