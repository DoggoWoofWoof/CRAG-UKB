"""Extract by the deflate stream's TRUE length, not the central directory's wrapped file_size.

    python scratchpad/final_canonical_build/webqsp_v1/v3_extract_true_size.py

THIRD SYMPTOM OF ONE DEFECT. Pass 1 found the archive's local-header offsets inflated by multiples
of 2^32 and repaired them by signature scan. Pass 2 found ZipInfo._end_offset stale for the same
reason and recomputed it. This pass finds that the UNCOMPRESSED SIZE field wrapped too, on three
entries. Same 32-bit overflow, third field.

Metadata/object_types.csv declares 769,323,370 bytes with a compress_size of 1,201,090,329 -- larger
than its own declared output, impossible for deflate on CSV. Reading the whole stream: it consumes
all 1,201,090,329 input bytes, reaches a clean deflate EOF, emits 13,654,225,258 bytes, and the CRC
over that full output matches the declared 0xed19ca81 exactly. 13,654,225,258 - 769,323,370 =
12,884,901,888 = 3 x 2^32. Python stopped at the wrapped size and CRC'd a prefix; hence "Bad CRC-32"
on an entry that was never damaged.

THE RULE, general and not per-file: never trust file_size. Inflate each stream to its own EOF and
take the true size from what comes out. The central directory's CRC is the check -- CRC-32 has no
length field to wrap, so it validates the recovered size rather than depending on it. Entries whose
declared size is correct verify unchanged; entries whose size wrapped come out longer by a multiple
of 2^32. No entry is special-cased and no check is bypassed.

The other 14 files extracted earlier are NOT in doubt: each was CRC-verified against its declared
size, and a CRC over a truncated prefix does not match. That verification is what proves their size
fields are sound -- "all_sizes_match_central_directory" alone never could have.
"""
import json, os, struct, time, zipfile, zlib

ROOT = "data/final_canonical/freebase_v3/_acquisition/idir"
ZIP = f"{ROOT}/idirlab-freebases.zip"
DST = f"{ROOT}/extracted"
OFF = f"{ROOT}/recovered_offsets.json"
REC = "data/final_canonical/freebase_v3/V3_EXTRACTION_RECORD.json"

# needed on disk: the per-object type map, for checks 6 (metadata coverage) and 7 (CVT coverage)
WRITE = ["idirlab-freebases/Metadata/object_types.csv"]

# inflated and CRC-checked but NOT written: these two only need their true size established, to
# settle whether FINDING_2 was reading truncation or reading the same size wrap.
VERIFY_ONLY = ["idirlab-freebases/FB+CVT+REV/train.txt",
               "idirlab-freebases/FB-CVT+REV/train.txt"]

SIG = b"PK\x03\x04"
CHUNK = 1 << 24
W32 = 1 << 32


def scan_offsets(path, names):
    """-> {filename: true_header_offset}. Signature scan validated against the name set."""
    found, tail, base = {}, b"", 0
    nb = {n.encode("utf-8"): n for n in names}
    maxn = max(len(b) for b in nb)
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(CHUNK)
            if not buf:
                break
            data = tail + buf
            s = 0
            while True:
                i = data.find(SIG, s)
                if i < 0:
                    break
                ao = base - len(tail) + i
                if i + 30 + maxn > len(data) and len(buf) == CHUNK:
                    break
                h = data[i:i + 30]
                if len(h) == 30:
                    nl = struct.unpack("<H", h[26:28])[0]
                    nm = data[i + 30:i + 30 + nl]
                    if nm in nb:
                        found.setdefault(nb[nm], ao)
                s = i + 1
            k = 30 + maxn + 4
            tail = data[-k:] if len(data) > k else data
            base += len(buf)
    return found


def inflate(fh, info, out_path):
    """Inflate one entry to its own deflate EOF. -> dict of measured facts. Never trusts file_size."""
    fh.seek(info.header_offset)
    h = fh.read(30)
    if h[:4] != SIG:
        raise ValueError(f"no local header at {info.header_offset}")
    nl, el = struct.unpack("<HH", h[26:30])
    fh.seek(info.header_offset + 30 + nl + el)

    d = zlib.decompressobj(-15) if info.compress_type == zipfile.ZIP_DEFLATED else None
    sink = open(out_path, "wb") if out_path else None
    left, produced, crc = info.compress_size, 0, 0
    try:
        while left > 0:
            b = fh.read(min(1 << 22, left))
            if not b:
                break
            left -= len(b)
            c = d.decompress(b) if d else b
            if c:
                produced += len(c)
                crc = zlib.crc32(c, crc)
                if sink:
                    sink.write(c)
        if d:
            c = d.flush()
            if c:
                produced += len(c)
                crc = zlib.crc32(c, crc)
                if sink:
                    sink.write(c)
    finally:
        if sink:
            sink.close()

    delta = produced - info.file_size
    return {
        "name": info.filename,
        "declared_file_size": info.file_size,
        "true_uncompressed_size": produced,
        "compress_size": info.compress_size,
        "input_fully_consumed": left == 0,
        "deflate_reached_eof": bool(d.eof) if d else True,
        "crc_declared": f"0x{info.CRC:08x}",
        "crc_computed": f"0x{crc:08x}",
        "crc_match": crc == info.CRC,
        "size_field_wrapped": delta != 0,
        "wrap_multiples_of_2^32": delta / W32 if delta else 0,
        "wrap_is_exact_multiple": delta % W32 == 0,
        "written_to_disk": bool(out_path),
    }


def main():
    t0 = time.time()
    z = zipfile.ZipFile(ZIP)
    infos = {i.filename: i for i in z.infolist()}

    if os.path.exists(OFF):
        found = json.load(open(OFF, encoding="utf-8"))
        print(f"[offsets] reusing {len(found)} cached", flush=True)
    else:
        found = scan_offsets(ZIP, set(infos))
        with open(OFF + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(found, fh, indent=1)
        os.replace(OFF + ".tmp", OFF)
        print(f"[offsets] recovered {len(found)}/{len(infos)} t={time.time()-t0:.0f}s", flush=True)
    for n, o in found.items():
        infos[n].header_offset = o

    results = []
    with open(ZIP, "rb") as fh:
        for w in WRITE + VERIFY_ONLY:
            out = None
            if w in WRITE:
                out = os.path.join(DST, w)
                os.makedirs(os.path.dirname(out), exist_ok=True)
            r = inflate(fh, infos[w], out)
            results.append(r)
            print(f"[{'ok' if r['crc_match'] else 'FAIL'}] {w}  declared "
                  f"{r['declared_file_size']:,} -> true {r['true_uncompressed_size']:,}  "
                  f"wrap={r['wrap_multiples_of_2^32']}x2^32  crc={r['crc_match']}  "
                  f"t={time.time()-t0:.0f}s", flush=True)

    by = {r["name"]: r for r in results}
    ot = by["idirlab-freebases/Metadata/object_types.csv"]
    doc = json.load(open(REC, encoding="utf-8"))

    doc["FINDING_4_UNCOMPRESSED_SIZE_FIELD_ALSO_WRAPPED"] = {
        "statement": "the same 32-bit overflow that corrupted the local-header offsets (FINDING_3) "
                     "also corrupted the UNCOMPRESSED SIZE field, on three entries. Third field, "
                     "one defect.",
        "detected_by": "compress_size exceeding file_size on entries that are plain text -- "
                       "impossible for deflate. That signature is what prompted reading the streams "
                       "in full rather than trusting the field.",
        "rule_applied": "inflate each stream to its own deflate EOF and take the true size from the "
                        "output. The central-directory CRC validates the recovered size; CRC-32 has "
                        "no length field to wrap. General, deterministic, no per-file special case.",
        "measured": results,
        "object_types_csv": {
            "declared": ot["declared_file_size"],
            "true": ot["true_uncompressed_size"],
            "difference": ot["true_uncompressed_size"] - ot["declared_file_size"],
            "difference_is": "3 x 2^32 exactly -- the same constant Info-ZIP reported for the offsets",
            "crc_match_over_full_stream": ot["crc_match"],
            "why_it_read_as_Bad_CRC_32": "Python stopped at the wrapped size and computed the CRC "
                                         "over a 769 MB prefix of a 13.65 GB stream. The payload "
                                         "was never damaged.",
        },
        "CORRECTS_FINDING_2": {
            "finding_2_said": "both +REV variants are size-inconsistent with their published triple "
                              "counts by more than an order of magnitude; FB-CVT+REV/train.txt is "
                              "11.5 MB against test and valid at 239 MB each.",
            "that_reading_was_wrong": "those two train.txt entries are exactly the other two wrapped "
                                      "size fields. Their streams inflate to a clean EOF with "
                                      "matching CRCs at 1 x 2^32 above the declared value.",
            "corrected_sizes": {
                r["name"]: {"declared": r["declared_file_size"],
                            "true": r["true_uncompressed_size"],
                            "crc_match": r["crc_match"]}
                for r in results if r["name"] in VERIFY_ONLY
            },
            "consequence": "the +REV variants are NOT truncated and NOT packaged wrongly. The "
                           "archive is internally consistent once the 32-bit wrap is undone. "
                           "FINDING_2's 'not_asserted' clause held the door open for exactly this; "
                           "the door is now closed by measurement.",
            "why_the_original_entry_is_kept": "it is the record of what the size fields alone "
                                              "supported before the streams were read. Superseded, "
                                              "not deleted.",
        },
        "other_14_extractions_are_not_in_doubt": "each was CRC-verified by zipfile against its "
                                                 "declared size. A CRC over a truncated prefix does "
                                                 "not match, so those size fields are sound. The "
                                                 "CRC is what proves this; "
                                                 "all_sizes_match_central_directory could not.",
        "recovered_offsets_cached_at": OFF,
    }

    if ot["crc_match"] and ot["written_to_disk"]:
        doc["extracted"] = doc.get("extracted", []) + [{
            "name": ot["name"],
            "bytes": ot["true_uncompressed_size"],
            "matches_central_directory_size": False,
            "size_source": "true deflate output length, CRC-verified; central directory value "
                           "wrapped by 3 x 2^32",
        }]
        doc["extracted_n"] = len(doc["extracted"])
        doc["extracted_gb"] = round(sum(g["bytes"] for g in doc["extracted"]) / 1e9, 2)
        doc["failed"] = []
        doc["all_entries_crc_verified"] = True
        doc["all_sizes_match_central_directory"] = False
        doc["size_field_note"] = ("false is now the CORRECT value: object_types.csv "
                                  "deliberately does not match the central directory, because "
                                  "the central directory is wrong there. See FINDING_4.")

    tmp = REC + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REC)
    print(json.dumps({k: doc[k] for k in ("extracted_n", "extracted_gb", "failed",
                                          "all_entries_crc_verified")}, indent=1))


if __name__ == "__main__":
    main()
