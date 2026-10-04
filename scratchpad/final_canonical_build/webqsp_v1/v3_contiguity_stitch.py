"""Global subject contiguity, stitched from the per-member opener shards PASS A emitted.

    python scratchpad/final_canonical_build/webqsp_v1/v3_contiguity_stitch.py

Contract v2 holds SOURCE_SUBJECT_CONTIGUOUS as PROVISIONAL and requires three numbers over the
whole verified source before PASS B may skip subject-side external sorting:

    SUBJECT_BLOCKS_N   DISTINCT_SUBJECTS_N   REOPENED_SUBJECT_BLOCKS_N

Only REOPENED_SUBJECT_BLOCKS_N == 0 opens that door.

THE SEAM IS THE WHOLE DIFFICULTY. Workers see one gzip member each, so a subject whose triples
straddle a member boundary is opened twice -- once at the end of member k, once at the start of
member k+1 -- by two workers neither of which can see the other. Those are ONE block, not two, and
counting them naively would manufacture up to 199 false reopenings and wrongly report the source as
non-contiguous. Every seam is therefore checked: if last_subject[k] == first_subject[k+1] the two
blocks are merged and the duplicate opener is suppressed before duplicate detection runs.

WHY NOT JUST TEST SORTEDNESS. A strictly increasing opener sequence would prove contiguity in O(1)
memory, and an 8 GB probe came within 4 violations of it -- but the source is a sharded export of
200 members, each internally sorted across the whole alphabet, so it is globally unsorted BY
CONSTRUCTION and sortedness is simply the wrong invariant. Duplicates are the right one, and are
measured directly here rather than inferred from order.

EXACTNESS. Openers are hashed to 64 bits so ~10^8 of them fit in memory. A collision would INFLATE
the reopened count, never hide one, and every duplicate hash is then resolved against the real
strings in a second read of the opener shards -- cheap, because those shards are ~0.5 GB compressed
and re-reading them costs nothing like re-decompressing the 31.3 GB source.
"""
import glob
import gzip
import json
import os
import time

import numpy as np

SHARDS = r"data\final_canonical\freebase_v3\pass_a\shards"
IDX = r"data\final_canonical\freebase_v3\V3_GZIP_MEMBER_INDEX.json"
OUT = r"data\final_canonical\freebase_v3\V3_SUBJECT_CONTIGUITY.json"
CHUNK = 8_000_000


def opener_files():
    fs = sorted(glob.glob(os.path.join(SHARDS, "m*_openers.tsv.gz")))
    return fs


def iter_openers(path, skip_first=None):
    """Yield opener lines (bytes). If skip_first is given and equals the first opener, drop it:
    that opener is the continuation of a block already counted in the previous member."""
    first = True
    with gzip.open(path, "rb") as fh:
        for line in fh:
            s = line.rstrip(b"\n")
            if not s:
                continue
            if first:
                first = False
                if skip_first is not None and s == skip_first:
                    continue
            yield s


def main():
    t0 = time.time()
    metas = {}
    for p in sorted(glob.glob(os.path.join(SHARDS, "m*_meta.json"))):
        m = json.load(open(p, encoding="utf-8"))
        metas[m["member"]] = m
    if not metas:
        raise SystemExit("no member metadata; run v3_pass_a_par.py first")
    ks = sorted(metas)
    idx = json.load(open(IDX, encoding="utf-8"))
    if len(ks) != idx["MEMBERS_N"]:
        raise SystemExit(f"have {len(ks)} member shards but the index lists {idx['MEMBERS_N']}; "
                         f"a partial PASS A cannot produce a global contiguity verdict")

    # --- seams ---
    seam_merges = 0
    seams = []
    for a, b in zip(ks, ks[1:]):
        la = metas[a]["last_subject"]
        fb = metas[b]["first_subject"]
        if la is not None and la == fb:
            seam_merges += 1
            seams.append({"between": [a, b], "subject": la})
    blocks_raw = sum(metas[k]["blocks"] for k in ks)
    blocks = blocks_raw - seam_merges

    # --- pass 1: hash every opener, suppressing the merged seam duplicates ---
    files = opener_files()
    if len(files) != len(ks):
        raise SystemExit(f"{len(files)} opener shards vs {len(ks)} members")
    parts = []
    buf = np.empty(CHUNK, dtype=np.int64)
    n = 0
    total = 0
    nonbracketed = 0
    for i, k in enumerate(ks):
        skip = metas[ks[i - 1]]["last_subject"].encode() if (
            i > 0 and metas[ks[i - 1]]["last_subject"] is not None) else None
        for s in iter_openers(files[i], skip_first=skip):
            t = s[1:-1] if (s[:1] == b"<" and s[-1:] == b">") else s
            if t is s:
                nonbracketed += 1
            buf[n] = hash(t)
            n += 1
            total += 1
            if n == CHUNK:
                parts.append(buf[:n].copy())
                n = 0
        print(f"  member {k:4d}  {total/1e6:8.1f}M openers  t={time.time()-t0:6.0f}s", flush=True)
    if n:
        parts.append(buf[:n].copy())
    hs = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    del parts, buf

    # Sort IN PLACE and compare neighbours, rather than np.unique(return_counts=True). np.unique
    # would allocate a sorted copy plus full-length values/counts arrays -- roughly 2.5 GB at ~10^8
    # openers, against a machine with about that much free. Sorting in place and taking one boolean
    # neighbour-comparison keeps the peak near the array itself, and the duplicated VALUES are few
    # enough to materialise. Multiplicities are then recovered exactly by binary search over the
    # sorted array, which is O(log n) per duplicate group instead of O(n).
    hs.sort()
    eq = hs[1:] == hs[:-1]
    n_dup_pairs = int(np.count_nonzero(eq))
    dup_hashes = np.unique(hs[1:][eq]) if n_dup_pairs else np.empty(0, dtype=np.int64)
    distinct_by_hash = int(hs.size - n_dup_pairs)
    if dup_hashes.size:
        lo = np.searchsorted(hs, dup_hashes, side="left")
        hi = np.searchsorted(hs, dup_hashes, side="right")
        occ = dict(zip(dup_hashes.tolist(), (hi - lo).tolist()))
    else:
        occ = {}
    del eq, hs

    # --- pass 2: exact resolution of every colliding hash ---
    reopened_exact = None
    collisions = 0
    resolved = []
    if dup_hashes.size:
        want = set(int(x) for x in dup_hashes)
        groups = {}
        for i, k in enumerate(ks):
            skip = metas[ks[i - 1]]["last_subject"].encode() if (
                i > 0 and metas[ks[i - 1]]["last_subject"] is not None) else None
            for s in iter_openers(files[i], skip_first=skip):
                t = s[1:-1] if (s[:1] == b"<" and s[-1:] == b">") else s
                hv = hash(t)
                if hv in want:
                    groups.setdefault(hv, set()).add(t)
        true_dups = 0
        for hv, strs in groups.items():
            o = occ[hv]
            if len(strs) == 1:
                true_dups += o - 1
                if len(resolved) < 12:
                    resolved.append({"subject": next(iter(strs)).decode("utf-8", "replace"),
                                     "opener_occurrences": o})
            else:
                collisions += 1
                true_dups += o - len(strs)
        reopened_exact = true_dups

    reopened = reopened_exact if reopened_exact is not None else 0
    distinct = blocks - reopened
    contiguous = reopened == 0

    doc = {
        "schema": "V3_SUBJECT_CONTIGUITY/v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": "V3_CANONICAL_CONTRACT_V2.json SUBJECT_CONTIGUITY_IS_PROVISIONAL",
        "derived_from": "per-member opener shards emitted by v3_pass_a_par.py (one pass over the "
                        "source, shared with PASS A)",
        "members": len(ks),
        "elapsed_s": round(time.time() - t0, 1),

        "SUBJECT_BLOCKS_N": blocks,
        "DISTINCT_SUBJECTS_N": distinct,
        "REOPENED_SUBJECT_BLOCKS_N": reopened,
        "SOURCE_SUBJECT_CONTIGUOUS": bool(contiguous),

        "SEAMS": {
            "member_boundaries": len(ks) - 1,
            "blocks_merged_across_seams": seam_merges,
            "blocks_before_seam_merge": blocks_raw,
            "examples": seams[:12],
            "why": "a subject straddling a member boundary is opened once by each of two workers. "
                   "Those are one block; without this correction each such seam would be reported "
                   "as a reopening and the source would look non-contiguous purely because of how "
                   "it was read.",
        },
        "DUPLICATE_RESOLUTION": {
            "hash_duplicate_groups": int(dup_hashes.size),
            "distinct_by_hash": distinct_by_hash,
            "confirmed_true_duplicates": reopened_exact,
            "hash_collisions_found": collisions,
            "examples": resolved,
            "method": "64-bit hash, then an exact second read of the opener shards for every "
                      "colliding hash. A collision inflates the count rather than hiding a "
                      "reopening, and is removed by comparing the real strings.",
        },
        "ORDER": {
            "nonbracketed_subjects": nonbracketed,
            "note": "order is deliberately NOT the test. The source is 200 internally sorted "
                    "members, so it is globally unsorted by construction; contiguity is decided by "
                    "duplicate openers alone.",
        },
        "GATE": {
            "rule": "PASS B may skip subject-side external sorting only if "
                    "REOPENED_SUBJECT_BLOCKS_N == 0 over the whole verified source.",
            "verdict": ("SUBJECT-SIDE SORT NOT REQUIRED" if contiguous else
                        "SUBJECT-SIDE SORT REQUIRED"),
            "if_required": "the build still works; it costs what contract v1 budgeted. Only the "
                           "amount of external sorting changes, never correctness.",
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)

    print(json.dumps({k2: doc[k2] for k2 in ("SUBJECT_BLOCKS_N", "DISTINCT_SUBJECTS_N",
                                             "REOPENED_SUBJECT_BLOCKS_N",
                                             "SOURCE_SUBJECT_CONTIGUOUS")}, indent=1))
    print(json.dumps(doc["SEAMS"], indent=1)[:800])
    print("VERDICT:", doc["GATE"]["verdict"])


if __name__ == "__main__":
    main()
