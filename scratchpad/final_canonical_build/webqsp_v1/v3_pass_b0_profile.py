"""PASS B0: profile the structural (non-metadata) portion of the source on a member sample.

    python scratchpad/final_canonical_build/webqsp_v1/v3_pass_b0_profile.py [--members 4]

WHY A PROFILING PASS EXISTS AT ALL. PASS B has to commit, before it runs for half an hour over
409 GB, to an on-disk encoding for ~2.2e9 edges on a disk with ~45 GB free. The three numbers that
decide that encoding -- how many structural edges there are, how their objects split between
Freebase URIs / external URIs / literals, and how many DISTINCT objects a member contains -- are
currently unknown, and each one changes the design:

  * distinct objects per member decides whether object dictionaries can be deduplicated in memory
    inside a worker or must be hash-partitioned to disk for PASS C to fold;
  * the literal share decides whether literal_uid minting is a side note or the dominant cost;
  * the predicate histogram is the relations table PASS D needs anyway, and it is what tells us
    which reverse pairs (type.object.type / type.type.instance and friends) actually carry volume.

Guessing these and being wrong costs a full re-run. Measuring them costs one pass over ~2% of the
source. The sample is spread across the file rather than taken from the front, because members are
internally sorted over the whole alphabet and the front of the file is not representative of it.

DISTINCT COUNTING IS NUMPY, NOT A PYTHON SET. A member holds on the order of 10^7 object positions;
a Python set of that many ints costs several hundred MB per worker and would not survive four
workers on this machine. Hashes are accumulated into int64 chunks and reduced with np.unique, which
peaks at about twice the array rather than ~8x.
"""
import collections
import json
import multiprocessing as mp
import os
import sys
import time
import zlib

import numpy as np

IDX = r"data\final_canonical\freebase_v3\V3_GZIP_MEMBER_INDEX.json"
OUT = r"data\final_canonical\freebase_v3\V3_PASS_B0_PROFILE.json"

NS = b"<http://rdf.freebase.com/ns/"
NSL = len(NS)
RDFS = b"<http://www.w3.org/2000/01/rdf-schema#"
SYN = b"<http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# Exactly the predicates PASS A consumed. Anything here is metadata and is NOT a structural edge;
# everything else is. Kept as a literal set of full predicate terms so the test is one dict lookup
# on the raw bytes, with no prefix logic to get subtly wrong.
META_PREDS = frozenset([
    NS + b"type.object.name>", RDFS + b"label>", NS + b"common.topic.alias>",
    NS + b"common.topic.description>", NS + b"type.object.type>", SYN + b"type>",
    NS + b"type.object.key>", NS + b"type.property.reverse_property>",
    NS + b"type.property.master_property>", NS + b"type.property.schema>",
    NS + b"type.property.expected_type>", NS + b"type.property.unique>",
    NS + b"type.property.delegated>", RDFS + b"domain>", RDFS + b"range>",
    NS + b"freebase.type_hints.mediator>", NS + b"freebase.type_hints.included_types>",
    NS + b"freebase.type_hints.enumeration>", NS + b"freebase.type_hints.deprecated>",
])

CHUNK = 1 << 22


def member_lines(fh, lo, hi):
    """Yield decompressed blocks that always end on a line boundary.

    Both sides are streamed: a member averages 1.9 GB decompressed and the largest is ~5.4 GB, so
    neither the compressed range nor its expansion may be read whole."""
    d = zlib.decompressobj(31)
    carry = b""
    remaining = hi - lo
    fh.seek(lo)
    while remaining > 0:
        blk = fh.read(min(CHUNK, remaining))
        if not blk:
            break
        remaining -= len(blk)
        out = d.decompress(blk)
        if not out:
            continue
        buf = carry + out
        j = buf.rfind(b"\n")
        if j < 0:
            carry = buf
            continue
        carry = buf[j + 1:]
        yield buf[:j]
    tail = d.flush()
    if tail:
        buf = carry + tail
        j = buf.rfind(b"\n")
        if j >= 0:
            carry = buf[j + 1:]
            yield buf[:j]
        else:
            carry = buf
    if carry.strip():
        yield carry


def split_literal(o):
    """Same terminator-dispatched split PASS A used, so both passes agree on literal identity."""
    if o[-1:] == b'"':
        return o[1:-1], b"", b""
    if o[-1:] == b">":
        i = o.rfind(b'"^^<')
        if i > 0:
            return o[1:i], b"", o[i + 4:-1]
        return None
    i = o.rfind(b'"@')
    if i > 0:
        return o[1:i], o[i + 2:], b""
    return None


class Uniq:
    """Accumulate int64 hashes in chunks and reduce with np.unique only at the end."""

    def __init__(self):
        self.parts = []
        self.buf = np.empty(1 << 21, dtype=np.int64)
        self.n = 0
        self.total = 0

    def add(self, h):
        if self.n == self.buf.size:
            self.parts.append(np.unique(self.buf))
            self.n = 0
        self.buf[self.n] = h
        self.n += 1
        self.total += 1

    def distinct(self):
        if self.n:
            self.parts.append(np.unique(self.buf[:self.n]))
            self.n = 0
        if not self.parts:
            return 0
        a = np.unique(np.concatenate(self.parts))
        self.parts = [a]
        return int(a.size)


def work(job):
    k, lo, hi, src = job
    t0 = time.time()
    preds = collections.Counter()
    kinds = collections.Counter()
    dtypes = collections.Counter()
    langs = collections.Counter()
    ext_hosts = collections.Counter()
    obj = Uniq()
    lit = Uniq()
    subj = Uniq()
    lines = struct = meta = malformed = 0
    obj_bytes = lit_bytes = 0
    samples = []

    with open(src, "rb") as fh:
        for block in member_lines(fh, lo, hi):
            for line in block.split(b"\n"):
                if not line:
                    continue
                lines += 1
                try:
                    s, p, rest = line.split(b"\t", 2)
                except ValueError:
                    malformed += 1
                    continue
                # Splitting only twice and stripping the terminator, rather than requiring exactly
                # four fields, keeps a literal that contains a raw tab intact instead of rejecting
                # the line -- and still validates the line ending.
                if rest[-2:] != b"\t.":
                    malformed += 1
                    continue
                o = rest[:-2]
                if p in META_PREDS:
                    meta += 1
                    continue
                struct += 1
                preds[p] += 1
                subj.add(hash(s))
                c = o[:1]
                if c == b"<":
                    if o[:NSL] == NS:
                        kinds["uri_ns"] += 1
                        obj.add(hash(o))
                        obj_bytes += len(o) - NSL
                    else:
                        kinds["uri_ext"] += 1
                        obj.add(hash(o))
                        obj_bytes += len(o)
                        j = o.find(b"/", 8)
                        ext_hosts[o[1:j] if j > 0 else o[:40]] += 1
                elif c == b'"':
                    t = split_literal(o)
                    if t is None:
                        kinds["literal_unparsed"] += 1
                        if len(samples) < 20:
                            samples.append(repr(line[:200]))
                        continue
                    lex, lang, dt = t
                    kinds["literal"] += 1
                    if lang:
                        langs[lang] += 1
                    if dt:
                        dtypes[dt] += 1
                    lit.add(hash(o))
                    lit_bytes += len(o)
                else:
                    kinds["other"] += 1
                    if len(samples) < 20:
                        samples.append(repr(line[:200]))

    return {
        "member": k, "elapsed_s": round(time.time() - t0, 1),
        "lines": lines, "structural": struct, "metadata": meta, "malformed": malformed,
        "kinds": {kk.decode() if isinstance(kk, bytes) else kk: v for kk, v in kinds.items()},
        "distinct_structural_subjects": subj.distinct(),
        "distinct_objects": obj.distinct(), "object_positions": obj.total,
        "distinct_literals": lit.distinct(), "literal_positions": lit.total,
        "object_id_bytes": obj_bytes, "literal_term_bytes": lit_bytes,
        "preds": {kk.decode("utf-8", "replace"): v for kk, v in preds.items()},
        "dtypes": {kk.decode("utf-8", "replace"): v for kk, v in dtypes.most_common(60)},
        "langs": {kk.decode("utf-8", "replace"): v for kk, v in langs.most_common(60)},
        "ext_hosts": {kk.decode("utf-8", "replace"): v for kk, v in ext_hosts.most_common(40)},
        "samples": samples,
    }


def main():
    nm = 4
    if "--members" in sys.argv:
        nm = int(sys.argv[sys.argv.index("--members") + 1])
    idx = json.load(open(IDX, encoding="utf-8"))
    starts = idx["member_starts"]
    src = idx["source"]
    total = idx["source_bytes"]
    ends = starts[1:] + [total]
    # Spread the sample across the file. Members are each sorted over the whole alphabet, so the
    # front of the file is not a representative slice of the whole.
    picks = [int(round((i + 0.5) * len(starts) / nm)) for i in range(nm)]
    picks = sorted(set(min(max(p, 0), len(starts) - 1) for p in picks))
    jobs = [(k, starts[k], ends[k], src) for k in picks]
    print(f"profiling members {picks} of {len(starts)}", flush=True)

    t0 = time.time()
    res = []
    with mp.Pool(len(jobs)) as pool:
        for r in pool.imap_unordered(work, jobs):
            res.append(r)
            print(f"  member {r['member']:4d}  {r['lines']/1e6:7.2f}M lines  "
                  f"{r['structural']/1e6:7.2f}M structural  {r['elapsed_s']:5.0f}s "
                  f"({len(res)}/{len(jobs)})", flush=True)

    n = len(starts)
    lines = sum(r["lines"] for r in res)
    struct = sum(r["structural"] for r in res)
    meta = sum(r["metadata"] for r in res)
    scale = n / len(res)
    kinds = collections.Counter()
    preds = collections.Counter()
    for r in res:
        kinds.update(r["kinds"])
        preds.update(r["preds"])

    obj_pos = sum(r["object_positions"] for r in res)
    obj_dist = sum(r["distinct_objects"] for r in res)
    lit_pos = sum(r["literal_positions"] for r in res)
    lit_dist = sum(r["distinct_literals"] for r in res)

    # Encoding budget. An edge row is src(int64) + rel(int64) + dst(int64) + kind(int8). The src
    # column is constant across a subject block and dictionary+RLE encodes to almost nothing, and
    # rel has only a few thousand distinct values, so dst is the column that actually costs. ~9-10
    # bytes/edge is the working estimate this pass exists to confirm or refute.
    est_edges = int(struct * scale)
    doc = {
        "schema": "V3_PASS_B0_PROFILE/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "members_sampled": [r["member"] for r in res],
        "members_total": n,
        "sample_fraction": round(len(res) / n, 5),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "SAMPLE": {
            "lines": lines, "structural": struct, "metadata": meta,
            "malformed": sum(r["malformed"] for r in res),
            "structural_fraction": round(struct / lines, 6) if lines else None,
            "object_kinds": dict(kinds),
            "distinct_objects_per_member_max": max(r["distinct_objects"] for r in res),
            "distinct_literals_per_member_max": max(r["distinct_literals"] for r in res),
            "object_dedup_ratio": round(obj_dist / obj_pos, 5) if obj_pos else None,
            "literal_dedup_ratio": round(lit_dist / lit_pos, 5) if lit_pos else None,
        },
        "EXTRAPOLATED_TO_FULL_SOURCE": {
            "note": "linear scaling from the sampled members; an estimate, not a measurement, and "
                    "superseded by PASS B's own totals.",
            "structural_edges": est_edges,
            "metadata_rows": int(meta * scale),
            "pass_a_rows_measured": 815908863,
            "literal_positions": int(lit_pos * scale),
            "edge_bytes_at_10B_per_edge_GB": round(est_edges * 10 / 1e9, 1),
            "edge_bytes_at_20B_per_edge_GB": round(est_edges * 20 / 1e9, 1),
        },
        "PREDICATES": {
            "distinct_in_sample": len(preds),
            "top_60": dict(preds.most_common(60)),
        },
        "DATATYPES": dict(collections.Counter(
            {k2: v for r in res for k2, v in r["dtypes"].items()}).most_common(40)),
        "EXT_HOSTS": dict(collections.Counter(
            {k2: v for r in res for k2, v in r["ext_hosts"].items()}).most_common(30)),
        "UNPARSED_SAMPLES": [s for r in res for s in r["samples"]][:20],
        "PER_MEMBER": [{k2: r[k2] for k2 in
                        ("member", "lines", "structural", "metadata", "malformed",
                         "distinct_objects", "object_positions", "distinct_literals",
                         "literal_positions", "distinct_structural_subjects", "elapsed_s")}
                       for r in sorted(res, key=lambda x: x["member"])],
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, OUT)
    print(json.dumps({k2: doc[k2] for k2 in
                      ("SAMPLE", "EXTRAPOLATED_TO_FULL_SOURCE")}, indent=1))
    print("distinct predicates in sample:", len(preds))
    for p, c in preds.most_common(25):
        print(f"  {c:>12,}  {p}")


if __name__ == "__main__":
    main()
