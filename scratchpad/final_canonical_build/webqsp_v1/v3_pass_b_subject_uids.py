"""Mint node UIDs for every distinct subject, and prove the 64-bit UID space is collision-free.

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/v3_pass_b_subject_uids.py

WHY THIS RUNS BEFORE PASS B. PASS B is about to write on the order of 2e9 edges whose endpoints are
64-bit UIDs, onto a disk with ~40 GB free. If two distinct nodes ever share a UID, every one of
those edges is ambiguous and the pass has to be re-run under a different salt. That risk is worth
about a gigabyte and four minutes to retire in advance rather than discover afterwards.

The test is exact, not statistical, because contiguity already gave us the answer to compare
against: DISTINCT_SUBJECTS_N is known to be 121,616,633. Hash all of them, sort, and count distinct
UIDs. If the two numbers agree there is no collision anywhere in the subject universe -- not "few
enough to ignore", none. If they disagree the colliding strings are recovered and named.

WHAT IS HASHED. The namespace-stripped identifier, not the bracketed URI: '<http://rdf.freebase.com
/ns/m.0zgmtqc>' mints from 'm.0zgmtqc'. The prefix is constant across the whole dump, so it carries
no identity, and stripping it keeps the UID tied to the released endpoint itself as the contract
requires. Non-namespace URIs keep their full form, since for those the host IS part of the identity.

DETERMINISM. Python's hash() is randomised per process unless PYTHONHASHSEED is fixed, which would
make two workers disagree about the same node. The seed is required here and asserted, not assumed;
the interpreter version is recorded because siphash is an interpreter detail, and PASS C re-checks
UID/string injectivity over the whole node universe regardless.

THE ARRAY IS ALSO A MEMBERSHIP INDEX. PASS B memory-maps the sorted output so a worker can ask, for
a batch of object UIDs at a time, which of them are NOT subjects anywhere in the dump. Only those
need their strings written out; the rest are already named by this file. That is what keeps PASS B's
node dictionary from being a second copy of the edge stream.
"""
import glob
import gzip
import json
import os
import sys
import time

import numpy as np

SHARDS = r"data\final_canonical\freebase_v3\pass_a\shards"
CONTIG = r"data\final_canonical\freebase_v3\V3_SUBJECT_CONTIGUITY.json"
OUTDIR = r"data\final_canonical\freebase_v3\pass_b"
UIDS = os.path.join(OUTDIR, "subject_uids.npy")
REPORT = r"data\final_canonical\freebase_v3\V3_NODE_UID_SPACE.json"

NS = b"<http://rdf.freebase.com/ns/"
NSL = len(NS)


def node_id(term):
    """Bracketed URI -> the identifier the UID is minted from."""
    if term[:NSL] == NS:
        return term[NSL:-1]
    if term[:1] == b"<" and term[-1:] == b">":
        return term[1:-1]
    return term


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 is required: hash() is randomised per process otherwise, "
                         "and PASS B's workers would mint different UIDs for the same node.")
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)
    contig = json.load(open(CONTIG, encoding="utf-8"))
    expect = contig["DISTINCT_SUBJECTS_N"]

    metas = {}
    for p in sorted(glob.glob(os.path.join(SHARDS, "m*_meta.json"))):
        m = json.load(open(p, encoding="utf-8"))
        metas[m["member"]] = m
    ks = sorted(metas)
    files = sorted(glob.glob(os.path.join(SHARDS, "m*_openers.tsv.gz")))
    if len(files) != len(ks):
        raise SystemExit(f"{len(files)} opener shards vs {len(ks)} members")

    # Preallocated once. Growing a list of 1.2e8 Python ints would cost ~4 GB; the array costs 973
    # MB and is the artifact we want anyway.
    a = np.empty(expect + 1024, dtype=np.int64)
    n = 0
    nonbracketed = 0
    for i, k in enumerate(ks):
        # Same seam suppression the contiguity pass used: if a subject ends member k-1 and opens
        # member k it is ONE block, and its opener must be counted once. Measured 0 here, but the
        # check stays because that is a property of this source, not a guarantee about it.
        prev = metas[ks[i - 1]]["last_subject"] if i > 0 else None
        skip = prev.encode() if prev is not None else None
        first = True
        with gzip.open(files[i], "rb") as fh:
            for line in fh:
                s = line.rstrip(b"\n")
                if not s:
                    continue
                if first:
                    first = False
                    if skip is not None and s == skip:
                        continue
                t = node_id(s)
                if t is s:
                    nonbracketed += 1
                if n >= a.size:
                    raise SystemExit(f"more openers than DISTINCT_SUBJECTS_N ({expect}); the "
                                     f"contiguity record and the shards disagree")
                a[n] = hash(t)
                n += 1
        if (k + 1) % 25 == 0:
            print(f"  member {k:4d}  {n/1e6:8.2f}M subjects  t={time.time()-t0:5.0f}s", flush=True)

    a = a[:n]
    if n != expect:
        raise SystemExit(f"hashed {n} openers but contiguity recorded {expect} distinct subjects")

    a.sort()
    eq = a[1:] == a[:-1]
    dup_pairs = int(np.count_nonzero(eq))
    distinct = n - dup_pairs

    colliding = []
    if dup_pairs:
        # Recover the actual strings behind every duplicated UID. A duplicate here is a genuine
        # collision, not a repeated subject: contiguity already established every subject is
        # emitted exactly once.
        want = set(np.unique(a[1:][eq]).tolist())
        found = {}
        for i, k in enumerate(ks):
            prev = metas[ks[i - 1]]["last_subject"] if i > 0 else None
            skip = prev.encode() if prev is not None else None
            first = True
            with gzip.open(files[i], "rb") as fh:
                for line in fh:
                    s = line.rstrip(b"\n")
                    if not s:
                        continue
                    if first:
                        first = False
                        if skip is not None and s == skip:
                            continue
                    t = node_id(s)
                    h = hash(t)
                    if h in want:
                        found.setdefault(h, set()).add(t)
        colliding = [{"uid": h, "strings": sorted(x.decode("utf-8", "replace") for x in v)}
                     for h, v in found.items() if len(v) > 1]

    np.save(UIDS, a)
    doc = {
        "schema": "V3_NODE_UID_SPACE/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "UID_SCHEME": {
            "function": "CPython hash() over the namespace-stripped identifier, as UTF-8 bytes",
            "width_bits": 64,
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "python": sys.version.split()[0],
            "note": "siphash is an interpreter detail, so the interpreter version is part of the "
                    "build provenance. PASS C re-checks UID -> string injectivity across the whole "
                    "node universe, so a changed hash surfaces as a mismatch, never as silent drift.",
        },
        "SUBJECTS": {
            "distinct_subjects_expected": expect,
            "subjects_hashed": n,
            "distinct_uids": distinct,
            "colliding_uid_groups": len(colliding),
            "COLLISION_FREE": distinct == n,
            "nonbracketed_openers": nonbracketed,
            "examples": colliding[:10],
            "reading": "distinct_uids == subjects_hashed means the 64-bit UID space separates every "
                       "subject in Freebase exactly. It is a measurement over all 121.6M of them, "
                       "not a birthday-bound argument.",
        },
        "ARTIFACT": {
            "path": UIDS,
            "dtype": "int64",
            "sorted": True,
            "bytes": os.path.getsize(UIDS),
            "purpose": "PASS B memory-maps this to test, per batch, which object UIDs are not "
                       "subjects anywhere; only those need their strings emitted.",
        },
    }
    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps({"SUBJECTS": doc["SUBJECTS"], "ARTIFACT": doc["ARTIFACT"]}, indent=1))


if __name__ == "__main__":
    main()
