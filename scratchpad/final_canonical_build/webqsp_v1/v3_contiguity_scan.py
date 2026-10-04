"""Global subject-contiguity scan over the whole verified source.

    python scratchpad/final_canonical_build/webqsp_v1/v3_contiguity_scan.py [--limit-gb N] [--tag T]

Contract v2 marks SOURCE_SUBJECT_CONTIGUOUS as PROVISIONAL and requires three numbers over the
whole source before PASS B may skip subject-side external sorting:

    SUBJECT_BLOCKS_N
    DISTINCT_SUBJECTS_N
    REOPENED_SUBJECT_BLOCKS_N

Only REOPENED_SUBJECT_BLOCKS_N == 0 opens that door.

WHY THIS DOES NOT JUST CHECK SORTEDNESS. A strictly increasing opener sequence would prove
contiguity in O(1) memory, and an 8 GB probe came within 4 violations of exactly that. Four is not
zero, so the cheap proof is unavailable and this scan measures duplicates directly instead of
inferring them from order. Order is still recorded, because WHERE the sequence breaks is diagnostic.

METHOD. `cut -f1 | uniq` reduces the stream to block openers at C speed -- roughly 2.5M openers per
8 GB, so about 78M for the full source. Each is hashed to 64 bits and the sorted array is scanned
for duplicates. A 64-bit hash over 78M items has a ~1.6e-4 chance of a spurious collision, which
would inflate the reopened count rather than hide one, so any duplicate found is then resolved
EXACTLY by a second targeted scan that collects the actual strings behind the colliding hashes.
Cheap in the common case, exact in the end, and it never reports contiguity that was not observed.
"""
import hashlib, json, os, subprocess, sys, time
import numpy as np

RAW = "data/final_canonical/freebase_v3/_acquisition/raw/freebase-rdf-latest.gz"
PART = RAW + ".part"
OUT = "data/final_canonical/freebase_v3/V3_SUBJECT_CONTIGUITY.json"
CHUNK = 8_000_000


# emit ONLY block openers. `cut -f1 | uniq` would be equivalent but pushes every subject of every
# line through the pipe -- about 90 GB for the full source -- where this pushes only the ~3.5 GB of
# openers. Same result, an order of magnitude less pipe traffic. Referencing $0 rather than $1 also
# keeps awk from splitting fields it never uses.
AWK_OPENERS = '{i=index($0,"\\t"); s=substr($0,1,i-1); if (s!=p) {print s; p=s}}'


def openers(src, limit):
    """Yield block-opening subject lines (bytes, no newline)."""
    gz = subprocess.Popen(["gzip", "-dc", src], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, bufsize=1 << 22)
    aw = subprocess.Popen(["awk", AWK_OPENERS], stdin=gz.stdout, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, bufsize=1 << 22,
                          env={**os.environ, "LC_ALL": "C"})
    gz.stdout.close()
    read = 0
    truncated = False
    try:
        for line in aw.stdout:
            read += len(line)
            if limit and read > limit:
                truncated = True
                break
            yield line.rstrip(b"\n")
    finally:
        if truncated:
            for q in (aw, gz):
                try:
                    q.kill()
                except Exception:
                    pass
        else:
            # drain both so gzip's exit status is meaningful. A nonzero code means the stream did
            # not decompress to EOF, and the block counts below would be silently short.
            try:
                aw.wait(timeout=120)
                gz.wait(timeout=120)
            except Exception:
                pass
        openers.last_rc = (gz.returncode, aw.returncode)


def ident(b):
    """Compare on the identifier, never the bracketed URI. '>' is 0x3E and digits start at 0x30,
    so '<...m.010l_>' compares greater than '<...m.010l_0k1>' and the bracketed form reports a
    false violation on every prefix pair in the file."""
    return b[1:-1] if b[:1] == b"<" and b[-1:] == b">" else b


def h64(b):
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little", signed=True)


def main():
    t0 = time.time()
    limit = None
    tag = ""
    if "--limit-gb" in sys.argv:
        limit = int(float(sys.argv[sys.argv.index("--limit-gb") + 1]) * 1e9)
    if "--tag" in sys.argv:
        tag = "_" + sys.argv[sys.argv.index("--tag") + 1]
    src = RAW if os.path.exists(RAW) else PART
    out = OUT if not tag else OUT.replace(".json", f"{tag}.json")
    if src == PART and not tag:
        raise SystemExit("refusing an untagged scan of an unverified .part file")

    blocks = 0
    viol = 0
    examples = []
    prev = None
    parts = []
    buf = np.empty(CHUNK, dtype=np.int64)
    k = 0
    nonbracketed = 0
    run_sizes = []
    run_start = 0

    for line in openers(src, limit):
        t = ident(line)
        if line is t:
            nonbracketed += 1
        blocks += 1
        if prev is not None and not (t > prev):
            viol += 1
            run_sizes.append(blocks - 1 - run_start)
            run_start = blocks - 1
            if len(examples) < 12:
                examples.append({"prev": prev.decode("utf-8", "replace"),
                                 "next": t.decode("utf-8", "replace"),
                                 "at_block": blocks})
        prev = t
        buf[k] = h64(t)
        k += 1
        if k == CHUNK:
            parts.append(buf[:k].copy())
            k = 0
            print(f"  {blocks/1e6:.1f}M openers  t={time.time()-t0:.0f}s", flush=True)
    if k:
        parts.append(buf[:k].copy())
    run_sizes.append(blocks - run_start)

    rc = getattr(openers, "last_rc", (None, None))
    if not limit and rc[0] not in (0, None):
        raise SystemExit(f"gzip exited {rc[0]}: the source did not decompress to EOF. The block "
                         f"counts would be short and must not be reported as a contiguity result.")

    hs = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    del parts, buf
    # one O(n log n) pass gives both the duplicate hashes AND their occurrence counts. Looking each
    # count up with (hs == hv).sum() instead would be O(n) per group -- fine for a handful of
    # groups, ruinous for thousands, which is exactly the case that matters here.
    uniq_h, cnts = np.unique(hs, return_counts=True)
    dup_sel = cnts > 1
    dup_hashes = uniq_h[dup_sel]
    occ_by_hash = dict(zip(dup_hashes.tolist(), cnts[dup_sel].tolist()))
    distinct_by_hash = int(uniq_h.size)
    del uniq_h, cnts, hs

    # exact resolution: a 64-bit collision would inflate this count, so any duplicate is confirmed
    # against the real strings rather than trusted.
    reopened_exact = None
    collisions = 0
    resolved = []
    if dup_hashes.size:
        want = set(int(x) for x in dup_hashes)
        groups = {}
        for line in openers(src, limit):
            t = ident(line)
            hv = h64(t)
            if hv in want:
                groups.setdefault(hv, set()).add(t)
        true_dups = 0
        for hv, strs in groups.items():
            occurrences = occ_by_hash[hv]
            if len(strs) == 1:
                true_dups += occurrences - 1
                if len(resolved) < 12:
                    resolved.append({"subject": next(iter(strs)).decode("utf-8", "replace"),
                                     "opener_occurrences": occurrences})
            else:
                collisions += 1
                true_dups += occurrences - len(strs)
        reopened_exact = true_dups

    reopened = reopened_exact if reopened_exact is not None else 0
    distinct = blocks - reopened
    contiguous = reopened == 0

    doc = {
        "schema": "V3_SUBJECT_CONTIGUITY/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": "V3_CANONICAL_CONTRACT_V2.json SUBJECT_CONTIGUITY_IS_PROVISIONAL",
        "source": src,
        "partial_run": bool(limit),
        "limit_gb": (limit / 1e9) if limit else None,
        "elapsed_s": round(time.time() - t0, 1),
        "gzip_returncode": rc[0],
        "STREAM_REACHED_EOF": rc[0] == 0,

        "SUBJECT_BLOCKS_N": blocks,
        "DISTINCT_SUBJECTS_N": distinct,
        "REOPENED_SUBJECT_BLOCKS_N": reopened,
        "SOURCE_SUBJECT_CONTIGUOUS": bool(contiguous) if not limit else "PROVISIONAL (partial run)",

        "ORDER": {
            "sort_violations": viol,
            "examples": examples,
            "note": "order is recorded because where the sequence breaks is diagnostic, but "
                    "contiguity is decided by duplicate openers, not by order. A file can be "
                    "unsorted and still perfectly contiguous.",
            "comparison_rule": "on the identifier between '<' and '>', never the bracketed URI.",
            "nonbracketed_subjects": nonbracketed,
        },
        "RUN_STRUCTURE": {
            "sorted_runs": len(run_sizes),
            "run_opener_counts": run_sizes if len(run_sizes) <= 200 else
                                 run_sizes[:100] + ["...truncated..."] + run_sizes[-20:],
            "reading": "the source is not one sorted stream but a concatenation of internally "
                       "sorted runs, each spanning the whole alphabet with near-identical "
                       "composition. That is the signature of a subject-partitioned sharded "
                       "export. It means global sortedness is the WRONG invariant to test for -- "
                       "the right one is whether any subject appears in more than one run, which "
                       "is what REOPENED_SUBJECT_BLOCKS_N measures.",
            "why_uniform_run_sizes_matter": "if the export partitioned by subject, each subject "
                                            "lands in exactly one run and contiguity holds despite "
                                            "the file never being globally sorted. If it "
                                            "partitioned by triple, subjects split across runs and "
                                            "the subject side needs the same external sort the "
                                            "object side needs. The duplicate count decides which.",
        },
        "DUPLICATE_RESOLUTION": {
            "hash_duplicate_groups": int(dup_hashes.size),
            "distinct_by_hash": distinct_by_hash,
            "confirmed_true_duplicates": reopened_exact,
            "hash_collisions_found": collisions,
            "examples": resolved,
            "method": "64-bit blake2b, then an exact second scan over every colliding hash. A "
                      "collision inflates the count rather than hiding a reopening, and is then "
                      "removed by comparing the real strings.",
        },
        "GATE": {
            "rule": "PASS B may skip subject-side external sorting only if "
                    "REOPENED_SUBJECT_BLOCKS_N == 0 over the whole verified source.",
            "verdict": ("SUBJECT-SIDE SORT NOT REQUIRED" if contiguous and not limit else
                        "PARTIAL RUN -- NOT A VERDICT" if limit else
                        "SUBJECT-SIDE SORT REQUIRED"),
            "if_required": "the build still works; it costs what contract v1 budgeted. Nothing "
                           "about correctness changes, only the amount of external sorting.",
        },
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, out)
    print(json.dumps({k: doc[k] for k in ("SUBJECT_BLOCKS_N", "DISTINCT_SUBJECTS_N",
                                          "REOPENED_SUBJECT_BLOCKS_N",
                                          "SOURCE_SUBJECT_CONTIGUOUS")}, indent=1))
    print(json.dumps(doc["ORDER"]["examples"][:6], indent=1))
    print("VERDICT:", doc["GATE"]["verdict"])


if __name__ == "__main__":
    main()
