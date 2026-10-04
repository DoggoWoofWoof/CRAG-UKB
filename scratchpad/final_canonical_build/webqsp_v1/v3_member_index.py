"""Index the independently-decompressible gzip members of the raw mirror.

    python scratchpad/final_canonical_build/webqsp_v1/v3_member_index.py [--scan-gb N]

WHY THIS EXISTS. Every full pass over the source was budgeted as a serial decompression: gzip.exe
sustains ~99 MB/s decompressed, so ~1.25 h per pass, and PASSES A/B/C/D would pay it four times.
But the mirror is not one gzip stream -- it is a concatenation of ~207 complete gzip members of
~151 MB each, and a member is a self-contained deflate stream. Any member can therefore be
decompressed without touching the ones before it, which turns every pass from serial into
embarrassingly parallel across the machine's 12 cores.

WHAT IS VERIFIED, NOT ASSUMED. The 3-byte gzip magic occurs by chance roughly every 16 MB of
compressed data, so ~1,900 false hits are expected across 31.3 GB. A candidate is accepted only if
a real decompressor consumes it AND the output looks like the N-Triples the source is supposed to
contain. Header-field plausibility alone is used to skip obvious non-headers cheaply; it never
accepts anything on its own.

LINE ALIGNMENT IS THE LOAD-BEARING PROPERTY. Parallel workers can only be given whole members if
members break on line boundaries. If a member ended mid-triple, the next would begin mid-triple and
its first byte would not be '<'. Every accepted member is checked to start with '<', and one member
is additionally decompressed IN FULL to confirm it ends with a newline -- inferring alignment from
the neighbours alone would be an argument, not a measurement.
"""
import hashlib
import json
import os
import sys
import time
import zlib

RAW = r"data\final_canonical\freebase_v3\_acquisition\raw\freebase-rdf-latest.gz"
OUT = r"data\final_canonical\freebase_v3\V3_GZIP_MEMBER_INDEX.json"
TOTAL = 31_305_093_084
MAGIC = bytes([0x1F, 0x8B, 0x08])
BLK = 1 << 24


def verify(fh, off, keep_pos):
    """Decompress a probe at `off`. Returns the first bytes of output, or None if not a member."""
    fh.seek(off)
    probe = fh.read(1 << 20)
    fh.seek(keep_pos)
    try:
        out = zlib.decompressobj(31).decompress(probe, 1 << 20)
    except Exception:
        return None
    if len(out) > 4096 and out[:1] == b"<" and out.count(b"\t") > 100:
        return out[:200]
    return None


def main():
    scan = TOTAL
    if "--scan-gb" in sys.argv:
        scan = int(float(sys.argv[sys.argv.index("--scan-gb") + 1]) * 1e9)

    t0 = time.time()
    starts = [0]
    firsts = {}
    prev = b""
    pos = 0
    candidates = 0
    with open(RAW, "rb") as fh:
        while pos < scan:
            blk = fh.read(BLK)
            if not blk:
                break
            buf = prev + blk
            base = pos - len(prev)
            i = 0
            while True:
                i = buf.find(MAGIC, i)
                if i < 0:
                    break
                off = base + i
                if off > 0:
                    hdr = buf[i:i + 10]
                    if len(hdr) == 10 and hdr[3] in (0, 8) and hdr[9] in (0, 3, 255):
                        candidates += 1
                        head = verify(fh, off, pos + len(blk))
                        if head is not None:
                            starts.append(off)
                            firsts[off] = head.split(b"\t", 1)[0].decode("utf-8", "replace")
                i += 1
            prev = buf[-3:]
            pos += len(blk)
            if pos % (1 << 30) < BLK:
                print(f"  {pos/1e9:5.1f} GB scanned  {len(starts):4d} members  "
                      f"{pos/(time.time()-t0)/1e6:4.0f} MB/s", flush=True)

    starts = sorted(set(starts))
    sizes = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    if scan >= TOTAL:
        sizes.append(TOTAL - starts[-1])

    # Full decompression of one interior member: proves a member both starts AND ends on a line
    # boundary, which is what makes whole-member work units safe.
    align = {}
    if len(starts) > 3:
        k = 2
        lo, hi = starts[k], starts[k + 1]
        with open(RAW, "rb") as fh:
            fh.seek(lo)
            raw = fh.read(hi - lo)
        d = zlib.decompressobj(31)
        out = d.decompress(raw)
        out += d.flush()
        align = {
            "member_checked": k,
            "compressed_bytes": hi - lo,
            "decompressed_bytes": len(out),
            "expansion_ratio": round(len(out) / (hi - lo), 3),
            "starts_with_lt": out[:1] == b"<",
            "ends_with_newline": out[-1:] == b"\n",
            "unused_data_after_member": len(d.unused_data),
            "lines": out.count(b"\n"),
            "first_line": out[:out.find(b"\n")].decode("utf-8", "replace")[:160],
            "last_line": out[out.rfind(b"\n", 0, len(out) - 1) + 1:-1].decode("utf-8", "replace")[:160],
        }

    est_decomp = None
    if align.get("expansion_ratio"):
        est_decomp = int(TOTAL * align["expansion_ratio"])

    doc = {
        "schema": "V3_GZIP_MEMBER_INDEX/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": RAW,
        "source_bytes": TOTAL,
        "partial_scan": scan < TOTAL,
        "scanned_bytes": pos,
        "elapsed_s": round(time.time() - t0, 1),
        "MEMBERS_N": len(starts),
        "magic_candidates_tested": candidates,
        "false_candidates_rejected": candidates - (len(starts) - 1),
        "member_size_compressed": {
            "min": min(sizes) if sizes else None,
            "max": max(sizes) if sizes else None,
            "mean": int(sum(sizes) / len(sizes)) if sizes else None,
        },
        "LINE_ALIGNMENT": align,
        "estimated_decompressed_bytes": est_decomp,
        "PARALLELISM": {
            "verdict": "MEMBER_PARALLEL_DECOMPRESSION_AVAILABLE" if len(starts) > 1 else
                       "SINGLE_MEMBER_SERIAL_ONLY",
            "reading": "each member is a complete deflate stream, so a worker can be handed a "
                       "(start, end) byte range and decompress it with no dependency on any other "
                       "worker. Passes A/B/C/D all become parallel.",
            "caveat_for_contiguity": "a subject block may span a member boundary. Workers must "
                                     "report the first and last subject of their member so the "
                                     "stitcher can decide whether two adjacent blocks are one "
                                     "block. Ignoring this would report false reopenings (or hide "
                                     "true ones) exactly at the 206 seams.",
        },
        "member_starts": starts,
        "member_first_subject": [firsts.get(s) for s in starts],
        "index_sha256": hashlib.sha256(
            json.dumps(starts).encode()).hexdigest(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)

    print(json.dumps({k: doc[k] for k in ("MEMBERS_N", "magic_candidates_tested",
                                          "false_candidates_rejected",
                                          "member_size_compressed",
                                          "estimated_decompressed_bytes")}, indent=1))
    print(json.dumps(doc["LINE_ALIGNMENT"], indent=1))
    print("VERDICT:", doc["PARALLELISM"]["verdict"])


if __name__ == "__main__":
    main()
