"""STEP S -- SERIALIZATION HARD BLOCKER.

The prior audit wrote the union endpoint list with

    s.replace("\\\\", "\\\\\\\\").replace("\\n", "\\\\n").replace("\\r", "\\\\r")

and read it back with

    x.replace("\\\\r", "\\r").replace("\\\\n", "\\n").replace("\\\\\\\\", "\\\\")     # audit_webqsp_E_reuse.py:80-81

The ENCODER is a well-formed prefix escape; the DECODER is a naive ordered replace and is
WRONG for any string containing a literal backslash followed by 'n' or 'r'.  This script

  1. decodes the file with a strict left-to-right scanner,
  2. proves losslessness by re-encoding and comparing bytes to the file,
  3. counts how many strings the OLD decoder corrupts (the reported "5"),
  4. RE-DERIVES the endpoint set independently from the 29 official parquet shards and
     asserts set equality (endpoint_set_before == endpoint_set_after),
  5. re-serializes losslessly as JSONL and asserts decode(encode(x)) == x for every endpoint.

Nothing is normalized and nothing is repaired.  Writes only to _audit/.
"""
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

A = "C:/Users/Swastik/Desktop/CRAG/scratchpad/final_canonical_build/_audit"
WEBQSP_GLOB = "C:/Users/Swastik/Desktop/CRAG/data/original/webqsp/rog_webqsp/*.parquet"
CWQ_GLOB = "C:/Users/Swastik/Desktop/CRAG/data/original/cwq/rog_cwq/*.parquet"
EXPECT_N = 2_592_894


def old_encode(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def old_decode_BUGGY(x):
    """Verbatim copy of audit_webqsp_E_reuse.py:80-81 -- kept only to quantify its damage."""
    return x.replace("\\r", "\r").replace("\\n", "\n").replace("\\\\", "\\")


def strict_decode(x):
    """Correct inverse of old_encode: single left-to-right pass, no re-scanning."""
    if "\\" not in x:
        return x
    out = []
    i, n = 0, len(x)
    while i < n:
        c = x[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 >= n:
            raise ValueError("dangling escape")
        d = x[i + 1]
        if d == "\\":
            out.append("\\")
        elif d == "n":
            out.append("\n")
        elif d == "r":
            out.append("\r")
        else:
            raise ValueError(f"unknown escape \\{d!r}")
        i += 2
    return "".join(out)


def read_lines(path):
    """Read exactly the logical lines the writer produced (universal newlines, as written)."""
    with open(path, encoding="utf-8") as fh:
        return fh.read().split("\n")[:-1]   # trailing '' after final newline


def scan_endpoints(paths, tag):
    """Independent re-derivation of the distinct graph endpoint set, streamed."""
    seen = set()
    t0 = time.time()
    for pi, p in enumerate(paths):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(batch_size=200, columns=["graph"]):
            lvl1 = b.column("graph").flatten()          # ListArray<string>, one per triple
            if len(lvl1) == 0:
                continue
            wn = pc.list_value_length(lvl1).to_numpy(zero_copy_only=False)
            if (wn != 3).any():
                import pyarrow as pa
                lvl1 = lvl1.take(pa.array(np.flatnonzero(wn == 3)))
                if len(lvl1) == 0:
                    continue
            flat = lvl1.flatten()
            enc = flat.dictionary_encode()
            codes = enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
            vocab = enc.dictionary.to_pylist()
            pos = np.arange(len(codes)) % 3
            for i in np.unique(codes[pos != 1]).tolist():
                seen.add(vocab[i])
        print(f"  [{tag}] {pi+1}/{len(paths)} {os.path.basename(p)[:30]:32s} "
              f"endpoints={len(seen):,} {time.time()-t0:.0f}s", flush=True)
    return seen


def main():
    res = {"_generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    raw = read_lines(A + "/union_entities.txt")
    print(f"[S] raw lines = {len(raw):,}", flush=True)
    assert len(raw) == EXPECT_N, f"line count {len(raw)} != {EXPECT_N}"

    dec = [strict_decode(x) for x in raw]

    # (2) losslessness of the FILE: re-encode must reproduce every line byte-for-byte
    bad_encode = [i for i, (s, x) in enumerate(zip(dec, raw)) if old_encode(s) != x]
    # injectivity: the decoded set must still have EXPECT_N distinct members
    dset = set(dec)
    print(f"[S] re-encode mismatches = {len(bad_encode)}   distinct decoded = {len(dset):,}", flush=True)

    # (3) how many strings does the OLD decoder corrupt?
    old_bad = [i for i, x in enumerate(raw) if old_decode_BUGGY(x) != dec[i]]
    res["OLD_DECODER"] = {
        "code": "x.replace('\\\\r','\\r').replace('\\\\n','\\n').replace('\\\\\\\\','\\\\')",
        "site": "scratchpad/final_canonical_build/audit_webqsp_E_reuse.py:80-81",
        "ROUNDTRIP_FAILURES_before_fix": len(old_bad),
        "failing_examples": [{"line_no": i + 1,
                              "file_bytes": raw[i],
                              "correct": dec[i],
                              "old_decoder_gave": old_decode_BUGGY(raw[i])} for i in old_bad[:20]],
    }
    print(f"[S] OLD decoder failures = {len(old_bad)}", flush=True)
    for i in old_bad[:10]:
        print(f"      line {i+1}: file={raw[i]!r} correct={dec[i]!r} old={old_decode_BUGGY(raw[i])!r}", flush=True)

    # (5) lossless JSONL re-serialization
    jl_fail = 0
    h = hashlib.sha256()
    with open(A + "/union_entities.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for s in dec:
            line = json.dumps(s, ensure_ascii=False)
            if json.loads(line) != s:
                jl_fail += 1
            fh.write(line + "\n")
            h.update(line.encode("utf-8"))
            h.update(b"\n")
    print(f"[S] JSONL round-trip failures = {jl_fail}", flush=True)

    # re-read the JSONL from disk and compare, order-sensitively
    reread = []
    with open(A + "/union_entities.jsonl", encoding="utf-8") as fh:
        for line in fh:
            reread.append(json.loads(line))
    disk_ok = (reread == dec)
    print(f"[S] JSONL disk re-read identical = {disk_ok}", flush=True)

    res["NEW_SERIALIZATION"] = {
        "format": "JSONL -- one json.dumps(endpoint, ensure_ascii=False) per line, LF, utf-8",
        "path": "scratchpad/final_canonical_build/_audit/union_entities.jsonl",
        "N": len(dec),
        "DISTINCT_N": len(dset),
        "ROUNDTRIP_FAILURES": jl_fail,
        "JSONL_DISK_REREAD_IDENTICAL": bool(disk_ok),
        "OLD_FILE_REENCODE_MISMATCHES": len(bad_encode),
        "ENTITY_ORDER_SHA256": h.hexdigest(),
    }

    # (4) independent re-derivation from source
    wq = sorted(glob.glob(WEBQSP_GLOB))
    cq = sorted(glob.glob(CWQ_GLOB))
    assert len(wq) == 5 and len(cq) == 24, (len(wq), len(cq))
    src = scan_endpoints(wq, "webqsp")
    src |= scan_endpoints(cq, "cwq")
    same = (src == dset)
    res["SOURCE_REDERIVATION"] = {
        "_what": "distinct graph endpoints re-derived from the 29 official parquet shards, "
                 "independent of the escaped temp file",
        "SOURCE_ENDPOINT_N": len(src),
        "DECODED_FILE_ENDPOINT_N": len(dset),
        "ENDPOINT_SET_BEFORE_EQ_AFTER": bool(same),
        "only_in_source": sorted(src - dset)[:20],
        "only_in_file": sorted(dset - src)[:20],
    }
    print(f"[S] source endpoints={len(src):,}  file endpoints={len(dset):,}  EQUAL={same}", flush=True)

    res["VERDICT"] = {
        "ROUNDTRIP_FAILURES": jl_fail,
        "ENDPOINT_SET_BEFORE_EQ_AFTER": bool(same),
        "CLEARED": bool(jl_fail == 0 and same and len(bad_encode) == 0 and disk_ok and len(dset) == EXPECT_N),
        "_root_cause": "DECODER bug, not encoder loss. The on-disk escape is a well-formed prefix escape; "
                       "the naive ordered .replace() inverse corrupts any endpoint containing a literal "
                       "backslash immediately followed by 'n' or 'r'. No string was normalized or repaired.",
    }
    with open(A + "/S_serialization.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("WROTE " + A + "/S_serialization.json", flush=True)
    return 0 if res["VERDICT"]["CLEARED"] else 3


if __name__ == "__main__":
    sys.exit(main())
