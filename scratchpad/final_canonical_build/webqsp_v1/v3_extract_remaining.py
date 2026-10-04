"""Extract the two entries the first repair pass failed on.

    python scratchpad/final_canonical_build/webqsp_v1/v3_extract_remaining.py

Root cause of both failures: Python 3.12+ precomputes ZipInfo._end_offset from the central
directory's entry ordering, as a zip-bomb guard. This archive's central-directory offsets are
corrupt, so those bounds are meaningless. The first pass patched header_offset but left _end_offset
stale, which truncated the read -- surfacing as "Overlapped entries" on one file and as "Bad CRC-32"
on the other. One cause, two symptoms.

Fix: recompute _end_offset from the RECOVERED ordering, so the guard still bounds each entry by its
true successor rather than by a corrupt value. CRC verification stays on; nothing is bypassed.
"""
import json, os, struct, time, zipfile

ROOT = "data/final_canonical/freebase_v3/_acquisition/idir"
ZIP = f"{ROOT}/idirlab-freebases.zip"
DST = f"{ROOT}/extracted"
REC = "data/final_canonical/freebase_v3/V3_EXTRACTION_RECORD.json"

WANT = [
    "idirlab-freebases/Metadata/object_names.csv",
    "idirlab-freebases/Metadata/object_types.csv",
]
SIG = b"PK\x03\x04"
CHUNK = 1 << 24


def scan(path, names):
    found = {}
    nb = {n.encode("utf-8"): n for n in names}
    maxn = max(len(b) for b in nb)
    tail, base = b"", 0
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


def main():
    t0 = time.time()
    size = os.path.getsize(ZIP)
    z = zipfile.ZipFile(ZIP)
    infos = {i.filename: i for i in z.infolist()}
    found = scan(ZIP, set(infos))
    order = sorted(found.items(), key=lambda kv: kv[1])
    print(f"[scan] {len(found)}/{len(infos)} t={time.time()-t0:.0f}s", flush=True)

    # true successor bound per entry, from the recovered layout
    end_of = {}
    for idx, (n, o) in enumerate(order):
        end_of[n] = order[idx + 1][1] if idx + 1 < len(order) else size
    for n, o in found.items():
        infos[n].header_offset = o
        if hasattr(infos[n], "_end_offset"):
            infos[n]._end_offset = end_of[n]

    got, failed = [], []
    for w in WANT:
        try:
            out = os.path.join(DST, w)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            n = 0
            with z.open(infos[w]) as src, open(out, "wb") as fh:
                while True:
                    b = src.read(1 << 22)
                    if not b:
                        break
                    fh.write(b)
                    n += len(b)
            ok = n == infos[w].file_size
            got.append({"name": w, "bytes": n, "matches_central_directory_size": ok})
            print(f"[ok] {w} {n/1e6:.1f} MB size_match={ok} t={time.time()-t0:.0f}s", flush=True)
        except Exception as ex:
            failed.append({"name": w, "reason": f"{type(ex).__name__}: {ex}"})
            print(f"[FAIL] {w} {type(ex).__name__}: {ex}", flush=True)

    doc = json.load(open(REC, encoding="utf-8"))
    doc["SECOND_PASS_END_OFFSET_FIX"] = {
        "root_cause": "ZipInfo._end_offset is precomputed by Python 3.12+ from central-directory "
                      "ordering as a zip-bomb guard. Those offsets are corrupt in this archive, so "
                      "the bound was meaningless and truncated the read. Pass 1 patched "
                      "header_offset but not _end_offset.",
        "symptoms_explained": {"object_names.csv": "BadZipFile: Overlapped entries",
                               "object_types.csv": "BadZipFile: Bad CRC-32"},
        "fix": "recompute _end_offset from the recovered layout so each entry is bounded by its "
               "true successor. The guard remains active and CRC verification is untouched.",
        "recovered": got,
        "still_failed": failed,
    }
    doc["extracted"] = doc.get("extracted", []) + got
    doc["extracted_n"] = len(doc["extracted"])
    doc["extracted_gb"] = round(sum(g["bytes"] for g in doc["extracted"]) / 1e9, 2)
    doc["failed"] = failed
    doc["all_sizes_match_central_directory"] = all(
        g.get("matches_central_directory_size", True) for g in doc["extracted"])
    tmp = REC + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REC)
    print(json.dumps({k: doc[k] for k in ("extracted_n", "extracted_gb",
                                          "all_sizes_match_central_directory", "failed")}, indent=1))


if __name__ == "__main__":
    main()
