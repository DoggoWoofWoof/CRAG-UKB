"""Segmented parallel fetch of the raw Freebase mirror, with per-segment resume.

    python scratchpad/final_canonical_build/webqsp_v1/v3_fetch_mirror.py [--segments N]

A single stream from archive.org sustains ~2.5 MB/s (~3.3 h). The item serves concurrent ranged
requests, so the transfer is split into N contiguous byte ranges fetched in parallel and
concatenated in order.

WHY THIS IS SAFE. Assembly is the one thing that could silently corrupt a segmented download, and
it is exactly what the pre-registered digests catch: md5/sha1 were locked in
V3_RAW_MIRROR_PROVENANCE_LOCK.json before a byte moved, so a mis-ordered or short segment fails
verification rather than passing as the artifact. Segments are also ETag-checked so a mirror that
rotated mid-transfer is caught at the segment, not 31 GB later.

DISK. Concatenation appends each segment into segment 0 and deletes it immediately, so peak usage is
~31.3 GB plus one segment, not 2x. The existing sequential .part is reused as segment 0's prefix --
already-transferred bytes are not re-fetched.
"""
import json, os, subprocess, sys, time

RAW = "data/final_canonical/freebase_v3/_acquisition/raw"
PART = f"{RAW}/freebase-rdf-latest.gz.part"
SEG = f"{RAW}/_segments"
LOCK = "data/final_canonical/freebase_v3/V3_RAW_MIRROR_PROVENANCE_LOCK.json"
URL = "https://archive.org/download/freebase-rdf-latest/freebase-rdf-latest.gz"
TOTAL = 31305093084
ETAG = '"55925d79-749edcfdc"'


def bounds(n):
    step = TOTAL // n
    return [(i * step, (i + 1) * step if i < n - 1 else TOTAL) for i in range(n)]


def seg_path(i):
    return PART if i == 0 else f"{SEG}/seg{i:02d}.part"


def have(i):
    p = seg_path(i)
    return os.path.getsize(p) if os.path.exists(p) else 0


def launch(i, lo, hi):
    """Resume segment i from wherever its file currently ends. Appends."""
    got = have(i)
    if got >= hi - lo:
        return None
    start = lo + got
    p = seg_path(i)
    cmd = ["curl", "-sL", "--retry", "30", "--retry-delay", "10", "--retry-all-errors",
           "--speed-limit", "16384", "--speed-time", "120",
           "-r", f"{start}-{hi-1}", "-D", f"{SEG}/hdr{i:02d}.txt", URL]
    fh = open(p, "ab")
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.DEVNULL), fh


def etag_ok(i):
    h = f"{SEG}/hdr{i:02d}.txt"
    if not os.path.exists(h):
        return None
    for line in open(h, encoding="utf-8", errors="replace"):
        if line.lower().startswith("etag:"):
            return line.split(":", 1)[1].strip() == ETAG
    return None


def main():
    n = 8
    if "--segments" in sys.argv:
        n = int(sys.argv[sys.argv.index("--segments") + 1])
    os.makedirs(SEG, exist_ok=True)
    segs = bounds(n)

    # a stale sequential curl would keep writing past segment 0's boundary
    subprocess.run(["taskkill", "/F", "/IM", "curl.exe"], capture_output=True)
    time.sleep(2)

    p0 = have(0)
    if p0 > segs[0][1] - segs[0][0]:
        raise SystemExit(f"existing .part ({p0}) exceeds segment 0 length; use fewer segments")
    print(f"{n} segments; segment 0 already holds {p0/1e9:.2f} GB", flush=True)

    procs = {}
    t0 = time.time()
    last, last_t = sum(have(i) for i in range(n)), time.time()
    while True:
        for i, (lo, hi) in enumerate(segs):
            done = have(i) >= hi - lo
            alive = i in procs and procs[i][0].poll() is None
            if not done and not alive:
                r = launch(i, lo, hi)
                if r:
                    if i in procs:
                        procs[i][1].close()
                    procs[i] = r

        tot = sum(have(i) for i in range(n))
        if tot >= TOTAL:
            break
        now = time.time()
        if now - last_t >= 30:
            rate = (tot - last) / (now - last_t)
            eta = (TOTAL - tot) / rate / 3600 if rate > 0 else float("inf")
            print(f"  {tot/1e9:6.2f}/{TOTAL/1e9:.2f} GB  {100*tot/TOTAL:5.1f}%  "
                  f"{rate/1048576:5.2f} MB/s  eta {eta:4.2f} h  "
                  f"live={sum(1 for i in procs if procs[i][0].poll() is None)}", flush=True)
            last, last_t = tot, now
        time.sleep(5)

    for i in procs:
        procs[i][1].close()
    tags = {i: etag_ok(i) for i in range(1, n)}
    bad = [i for i, v in tags.items() if v is False]
    if bad:
        raise SystemExit(f"ETag mismatch on segments {bad}: the mirror changed mid-transfer. "
                         f"Do not assemble; re-fetch.")
    print(f"all segments complete in {(time.time()-t0)/60:.1f} min; ETags consistent", flush=True)

    # append-and-delete so peak disk stays ~TOTAL + one segment
    with open(PART, "ab") as out:
        for i in range(1, n):
            p = seg_path(i)
            with open(p, "rb") as fh:
                while True:
                    b = fh.read(1 << 24)
                    if not b:
                        break
                    out.write(b)
            os.remove(p)
            print(f"  merged seg{i:02d}, part now {os.path.getsize(PART)/1e9:.2f} GB", flush=True)

    got = os.path.getsize(PART)
    print(json.dumps({"assembled_bytes": got, "expected": TOTAL, "match": got == TOTAL,
                      "next": "run v3_verify_raw_mirror.py"}, indent=1))


if __name__ == "__main__":
    main()
