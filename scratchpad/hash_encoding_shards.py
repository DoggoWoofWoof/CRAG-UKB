"""
SHARD CONTENT HASHES
====================
The direct lesson of the dense corruption.

Every encoding manifest in this repo records n_items, rows_covered and source_sha256 -- the
hash of the INPUT jsonl -- and then declares "complete": true on row count alone. None of
them records anything about the bytes that were actually written. That is exactly why
142,633 dense rows could sit all-zero on disk since the original encode, survive every
manifest check, and silently underlie earlier experiments: a lost writeback changes the
file's CONTENT without changing its LENGTH or its row count, so a length-and-count manifest
cannot see it.

This walks every shard of every encoding channel and records:

    sha256      of the file bytes
    size        in bytes
    mtime_utc   as written by the filesystem

with a per-channel roll-up hash over the ordered per-shard digests, so one value fingerprints
a whole channel.

WHAT THIS DOES AND DOES NOT PROMISE
    It fingerprints what is on disk NOW. Ten of the fifteen dense channels are still
    physically damaged -- the repair was installed non-destructively, as a pointer redirect,
    so the corrupt shards were deliberately left untouched. Their hashes therefore record a
    damaged file, honestly, and the row-level verdict from the integrity scan is carried
    alongside each channel so the two are never confused. From here on any further change to
    a shard is detectable; nothing about a matching hash says the bytes were ever correct.
"""
import hashlib
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CANON = "data/canonical"
OUT = "data/final_canonical/ENCODING_SHARD_HASHES.json"
BUF = 8 << 20


def sha256_file(p):
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(BUF)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def channels():
    for tree in sorted(os.listdir(CANON)):
        d = os.path.join(CANON, tree, "encodings")
        if not os.path.isdir(d):
            continue
        for model in ("dense", "splade"):
            for kind in ("docs", "queries"):
                c = os.path.join(d, model, kind)
                if os.path.isdir(c):
                    yield tree, model, kind, c


def main():
    t0 = time.time()
    only = set(sys.argv[1:])
    scan = {}
    p = "scratchpad/encoding_integrity.json"
    if os.path.exists(p):
        scan = json.load(io.open(p, encoding="utf-8")).get("BY_CHANNEL", {})

    out = {"RECORD": "CANONICAL_ENCODING_SHARD_HASHES_V1",
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "WHY": ("manifests recorded row counts and the hash of the INPUT jsonl, never the "
                   "bytes written; a lost writeback changes content without changing length "
                   "or row count, so 142,633 all-zero dense rows passed every existing check"),
           "algorithm": "sha256 per shard file; channel_hash = sha256 of the ordered "
                        "'<name> <sha256>\\n' lines",
           "caveat": ("fingerprints the CURRENT bytes. The dense repair was installed as a "
                      "pointer redirect and did NOT rewrite the damaged shards, so a channel "
                      "marked row_integrity=DAMAGED is hashed in its damaged state on "
                      "purpose. A matching hash proves no change since now, never that the "
                      "bytes were correct."),
           "channels": {}}

    tot_bytes = 0
    for tree, model, kind, d in channels():
        key = "%s/%s/%s" % (tree, kind, model)
        if only and tree not in only:
            continue
        shards = sorted(f for f in os.listdir(d)
                        if f.startswith("shard_") and f.endswith((".npy", ".npz")))
        rows = []
        agg = hashlib.sha256()
        cb = 0
        for f in shards:
            fp = os.path.join(d, f)
            hx, n = sha256_file(fp)
            cb += n
            agg.update(("%s %s\n" % (f, hx)).encode("ascii"))
            rows.append({"file": f, "sha256": hx, "bytes": n,
                         "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                    time.gmtime(os.path.getmtime(fp)))})
        s = scan.get(key, {})
        out["channels"][key] = {
            "dir": d.replace("\\", "/"), "n_shards": len(shards), "bytes": cb,
            "channel_hash": agg.hexdigest(),
            "row_integrity": ("DAMAGED" if s.get("bad_rows") else
                              ("CLEAN" if s else "NOT_SCANNED")),
            "damaged_rows": int(s.get("bad_rows", 0)),
            "shards": rows}
        tot_bytes += cb
        print("  %-34s shards=%-4d %7.2f GB  %s  %s"
              % (key, len(shards), cb / 1e9, agg.hexdigest()[:16],
                 out["channels"][key]["row_integrity"]), flush=True)

    out["total_bytes"] = tot_bytes
    out["n_channels"] = len(out["channels"])
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n%d channels, %.1f GB, %.1fs -> %s"
          % (out["n_channels"], tot_bytes / 1e9, out["elapsed_s"], OUT))


if __name__ == "__main__":
    main()
