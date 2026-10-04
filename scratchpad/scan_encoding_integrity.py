"""
PHASE-C ENCODING INTEGRITY SCAN
===============================
Finds every row in the frozen Phase-C encoding tree whose stored vector is not a valid
embedding, and writes the exact re-encode worklist.

WHY THIS EXISTS
A pointer-index verification found dense rows with L2 norm 0.0. Three independent read
paths (mmap, full read, raw bytes) agree, so the zeros are on disk. In metaqa_1hop the
damaged span runs from byte 8,458,240 to exactly 16,777,216 (16 MiB) -- it starts 4096-byte
aligned in the MIDDLE of row 2753 and ends on a 16 MiB boundary, leaving torn vectors at
both edges (norms 0.5651 and 0.8329) and all-zero rows between. Encoder or batching faults
align to row and batch boundaries; 4 KiB pages and a 16 MiB mark are filesystem extents.
So this is lost writeback, not a bad encode -- consistent with a 98%-full volume.

No existing manifest could have caught it: manifest.json records n_items, rows_covered and
a source_sha256 (a hash of the INPUT jsonl), never a hash of the shard contents, and
declares "complete": true on row count alone.

WHAT COUNTS AS BAD
  dense  : L2 norm is 0 (never written), or differs from 1 by more than 1e-2 (torn write).
           The encoder contract is normalize_embeddings=True, so every intact row is 1.0
           up to float16 rounding, which the observed clean rows bound at 1.000497.
  splade : an all-zero row. SPLADE is log(1+relu(x)) max-pooled, so a real row is
           non-negative and, for any non-empty input, has at least one non-zero term.
           Zeroed CSR data also shows up as explicit zero-valued entries, which are
           counted separately because scipy stores them without changing indptr.

Emits scratchpad/encoding_integrity.json and, per affected channel, the row ids to redo.
"""
import glob, io, json, os, sys, time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CANON = "data/canonical"
OUT = "scratchpad/encoding_integrity"
TOL = 1e-2
TREES = ["metaqa", "metaqa_1hop", "metaqa_2hop", "metaqa_3hop", "2wiki", "2wiki_universe",
         "musique", "hotpotqa", "squad", "webqsp"]


def ids_for(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "ids_*.json"))):
        out += json.load(io.open(f, encoding="utf-8"))
    return out


def scan_dense(d):
    bad_zero, bad_torn, n = [], [], 0
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npy"))):
        a = np.load(f, mmap_mode="r")
        for i in range(0, a.shape[0], 20000):
            b = np.asarray(a[i:i + 20000], dtype=np.float32)
            nr = np.linalg.norm(b, axis=1)
            z = np.nonzero(nr == 0)[0]
            t = np.nonzero((nr > 0) & (np.abs(nr - 1.0) > TOL))[0]
            bad_zero += (z + n + i).tolist()
            bad_torn += (t + n + i).tolist()
        n += a.shape[0]
    return n, bad_zero, bad_torn


def scan_splade(d):
    from scipy.sparse import csr_matrix
    bad_zero, explicit_zero, n = [], 0, 0
    for f in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
        z = np.load(f)
        M = csr_matrix((z["data"], z["indices"], z["indptr"]),
                       shape=(int(z["shape"][0]), int(z["shape"][1])))
        empty = np.nonzero(np.diff(M.indptr) == 0)[0]
        bad_zero += (empty + n).tolist()
        explicit_zero += int((M.data == 0).sum())
        n += M.shape[0]
    return n, bad_zero, explicit_zero


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rep = {}
    for tree in TREES:
        for kind in ("docs", "queries"):
            for model in ("dense", "splade"):
                d = os.path.join(CANON, tree, "encodings", model, kind)
                if not os.path.isdir(d):
                    continue
                key = "%s/%s/%s" % (tree, kind, model)
                if model == "dense":
                    n, bz, bt = scan_dense(d)
                    ez = 0
                else:
                    n, bz, ez = scan_splade(d)
                    bt = []
                bad = sorted(set(bz) | set(bt))
                r = {"rows": n, "zero_rows": len(bz), "torn_rows": len(bt),
                     "explicit_zero_entries": ez, "bad_rows": len(bad),
                     "bad_pct": round(100.0 * len(bad) / n, 4) if n else 0.0}
                if bad:
                    ids = ids_for(d)
                    if len(ids) != n:
                        r["IDS_LENGTH_MISMATCH"] = len(ids)
                    r["worklist"] = "%s/%s.json" % (OUT, key.replace("/", "__"))
                    json.dump({"tree": tree, "kind": kind, "model": model,
                               "rows": n, "bad_rows": bad,
                               "bad_ids": [ids[i] for i in bad] if len(ids) == n else None},
                              io.open(r["worklist"], "w", encoding="utf-8"))
                    # contiguity: real corruption comes in runs, a bad encode would not
                    a = np.asarray(bad)
                    runs = 1 + int((np.diff(a) != 1).sum()) if a.size else 0
                    r["contiguous_runs"] = runs
                rep[key] = r
                print("  %-34s rows=%-9d bad=%-7d %s"
                      % (key, n, len(bad), "runs=%d" % r.get("contiguous_runs", 0) if bad else ""),
                      flush=True)
    tot = sum(v["bad_rows"] for v in rep.values())
    out = {"RECORD": "PHASE_C_ENCODING_INTEGRITY_SCAN",
           "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "TOTAL_BAD_ROWS": tot, "TOLERANCE_L2": TOL,
           "mechanism": ("lost writeback, not a bad encode: damaged spans are 4096-byte "
                         "aligned and end on filesystem extent boundaries, tearing vectors "
                         "mid-row at both edges"),
           "detectability": ("no manifest could have caught this: manifest.json records "
                             "n_items, rows_covered and a hash of the SOURCE jsonl, never a "
                             "hash of shard contents"),
           "BY_CHANNEL": rep, "elapsed_s": round(time.time() - t0, 1)}
    json.dump(out, io.open("scratchpad/encoding_integrity.json", "w", encoding="utf-8"), indent=1)
    print("\nTOTAL BAD ROWS %s   %.1fs" % (format(tot, ","), time.time() - t0))
