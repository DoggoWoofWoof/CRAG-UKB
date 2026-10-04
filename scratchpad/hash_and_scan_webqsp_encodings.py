# -*- coding: utf-8 -*-
"""
Content-fingerprint and exhaustively integrity-scan dataset 6's ACTUAL vector stores.

THE GAP THIS CLOSES
    ENCODING_SHARD_HASHES.json fingerprints 30 channels / 52.5 GB. Four of those are keyed
    "webqsp/..." and they point at `data/canonical/webqsp/encodings/...` -- the SUPERSEDED
    Phase-C webqsp tree from the earlier KB build, which dataset 6 does not use and which that
    record marks row_integrity=DAMAGED with 12,814 damaged rows.

    Dataset 6 resolves into `data/canonical/webqsp_rog_v1/encodings/...`. Those four stores --
    5.95 GB, every vector the sixth dataset has -- were never fingerprinted and never
    exhaustively scanned. scan_encoding_integrity.py's TREES list predates them.

    So the exact check that caught 142,633 silently-zero dense rows in the five had never been
    run against the sixth. A reader verifying webqsp against ENCODING_SHARD_HASHES.json would
    be checking the wrong directory and would see a DAMAGED verdict that belongs to a tree the
    package does not use.

WHAT THIS DOES
    Reuses the two existing implementations rather than restating them, so the criteria cannot
    drift: sha256_file from hash_encoding_shards, scan_dense/scan_splade from
    scan_encoding_integrity. Same tolerance, same definition of a bad row, same channel_hash
    construction.

    Writes a NEW record. ENCODING_SHARD_HASHES.json is hashed by LOCKED_5_OF_5 and is not
    touched -- amending it would break that freeze.
"""
import hashlib
import importlib.util
import io
import json
import os
import sys
import time

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0")
# both imported modules rewrap sys.stdout at module scope. Keep a reference to whatever
# is there so the wrapper being replaced is not garbage-collected -- its __del__ closes
# the underlying buffer, which is what made the first run die with "I/O operation on
# closed file" before a single byte was hashed.
_KEEP_STDOUT = [sys.stdout]

TREE = "data/canonical/webqsp_rog_v1/encodings"
OUT = "data/final_canonical/WEBQSP_ENCODING_SHARD_HASHES.json"
SUPERSEDED = "data/canonical/webqsp/encodings"


def _mod(name, path):
    """Import a sibling script without running its __main__ block."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


HASHER = _mod("_hasher", "scratchpad/hash_encoding_shards.py")
_KEEP_STDOUT.append(sys.stdout)
SCAN = _mod("_scan", "scratchpad/scan_encoding_integrity.py")
_KEEP_STDOUT.append(sys.stdout)


def main():
    t0 = time.time()
    six = json.load(io.open("data/final_canonical/LOCKED_6_OF_6_BENCHMARK_SUBSTRATES.json",
                            encoding="utf-8"))
    declared = six["webqsp"]["encoding_channels"]

    chans, fails = {}, []
    tot_bytes = 0
    for model in ("dense", "splade"):
        for kind in ("docs", "queries"):
            d = "%s/%s/%s" % (TREE, model, kind)
            key = "%s/%s" % (kind, model)
            if not os.path.isdir(d):
                fails.append("store directory missing: %s" % d)
                continue

            shards = sorted(f for f in os.listdir(d)
                            if f.startswith("shard_") and f.endswith((".npy", ".npz")))
            rows, agg, cb = [], hashlib.sha256(), 0
            for f in shards:
                fp = os.path.join(d, f)
                hx, n = HASHER.sha256_file(fp)
                cb += n
                agg.update(("%s %s\n" % (f, hx)).encode("ascii"))
                rows.append({"file": f, "sha256": hx, "bytes": n,
                             "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                        time.gmtime(os.path.getmtime(fp)))})

            # exhaustive, every row -- the same criteria that found the Phase-C corruption
            if model == "dense":
                n, bz, bt = SCAN.scan_dense(d)
                ez = 0
            else:
                n, bz, ez = SCAN.scan_splade(d)
                bt = []
            bad = sorted(set(bz) | set(bt))

            chans[key] = {
                "dir": d, "n_shards": len(shards), "bytes": cb,
                "channel_hash": agg.hexdigest(),
                "rows_scanned": n, "zero_rows": len(bz), "torn_rows": len(bt),
                "explicit_zero_entries": ez, "bad_rows": len(bad),
                "row_integrity": "DAMAGED" if bad else "CLEAN",
                "bad_row_ids": bad[:1000],
                "shards": rows}
            tot_bytes += cb

            dec = declared.get(key)
            if dec is None:
                fails.append("%s is on disk but LOCKED_6_OF_6 declares no such channel" % key)
            else:
                if n != dec["rows"]:
                    fails.append("%s: scanned %d rows, LOCKED_6_OF_6 declares %d"
                                 % (key, n, dec["rows"]))
                if cb != dec["bytes"]:
                    fails.append("%s: %d bytes on disk, LOCKED_6_OF_6 declares %d"
                                 % (key, cb, dec["bytes"]))
                if dec["store"].rstrip("/") != d:
                    fails.append("%s: LOCKED_6_OF_6 store is %s, scanned %s"
                                 % (key, dec["store"], d))
            if bad:
                fails.append("%s: %d bad rows -- dataset 6's vectors are NOT clean" % (key, len(bad)))
            print("  %-16s shards=%-4d %7.2f GB  rows=%-9d bad=%-6d %s  %s"
                  % (key, len(shards), cb / 1e9, n, len(bad),
                     chans[key]["row_integrity"], agg.hexdigest()[:16]), flush=True)

    for key in declared:
        if key not in chans:
            fails.append("LOCKED_6_OF_6 declares channel %s that was not found on disk" % key)

    rec = {
        "RECORD": "WEBQSP_ENCODING_SHARD_HASHES",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "WHY": ("dataset 6's vector stores were never content-hashed and never exhaustively "
                "integrity-scanned. ENCODING_SHARD_HASHES.json's webqsp/* entries describe a "
                "DIFFERENT, superseded tree."),
        "THE_TRAP": {
            "record": "data/final_canonical/ENCODING_SHARD_HASHES.json",
            "its_webqsp_keys_point_at": SUPERSEDED,
            "which_is": ("the superseded Phase-C webqsp tree from the earlier KB build. It is "
                         "marked row_integrity=DAMAGED there, honestly, and dataset 6 does not "
                         "use a single byte of it."),
            "dataset_6_actually_uses": TREE,
            "so": ("do not verify dataset 6 against ENCODING_SHARD_HASHES.json. Use this "
                   "record. ENCODING_SHARD_HASHES.json is hashed by LOCKED_5_OF_5 and was "
                   "deliberately NOT amended -- editing it would break that freeze.")},
        "algorithm": ("identical to hash_encoding_shards.py and scan_encoding_integrity.py -- "
                      "those modules are imported, not reimplemented, so the criteria cannot "
                      "drift: dense bad = L2 zero or |L2-1| > %g; splade bad = empty row; "
                      "channel_hash = sha256 of the ordered '<name> <sha256>\\n' lines"
                      % SCAN.TOL),
        "channels": chans,
        "total_bytes": tot_bytes,
        "n_channels": len(chans),
        "failures": fails,
        "VERDICT": "ALL_PASS" if not fails else "FAIL",
        "elapsed_s": round(time.time() - t0, 1)}
    json.dump(rec, io.open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n%d channels, %.2f GB" % (len(chans), tot_bytes / 1e9))
    print("VERDICT %s  (%d failures)  %.1fs" % (rec["VERDICT"], len(fails), rec["elapsed_s"]))
    for f in fails[:20]:
        print("  FAIL %s" % f)
    print("wrote %s" % OUT)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
