"""PASS A stage 1, member-parallel: schema/metadata extraction AND subject-block openers.

    python scratchpad/final_canonical_build/webqsp_v1/v3_pass_a_par.py [--workers N] [--members K]

TWO CONTRACT JOBS, ONE READ OF THE SOURCE. Contract v2 wants PASS A (names, aliases, types,
mediator declarations, reverse_property pairs, schema metadata) and a global subject-contiguity
measurement (SUBJECT_BLOCKS_N / DISTINCT_SUBJECTS_N / REOPENED_SUBJECT_BLOCKS_N). They were two
separate full passes. Contiguity needs every line's subject and PASS A needs a filtered slice of
the same lines, so one pass produces both.

WHY THIS IS PARALLEL AT ALL. The mirror is a concatenation of 200 complete gzip members, each a
self-contained deflate stream (V3_GZIP_MEMBER_INDEX.json, where line alignment is verified by
decompressing one member in full rather than inferred). A worker takes one member's byte range and
decompresses it with no dependency on any other worker.

WHY NOT gzip.exe HERE. One gzip.exe over a native pipe is the fastest SERIAL reader measured
(99 MB/s decompressed vs 46 MB/s for in-process zlib), but gzip consumes a stream, so handing it a
single member means pushing 151 MB through a pipe inbound as well as outbound. In-process zlib
takes a byte range directly, and N workers at 46 MB/s beat one at 99. Never route the decompressed
stream through a Git Bash pipe: measured at 4.4 MB/s compressed against 99 MB/s for a native pipe,
which is what made the first PASS A look like a Python problem when it was a pipe problem.

MEMORY. A member averages 1.92 GB decompressed and the largest is ~5.4 GB, so nothing here reads a
member whole. Both the compressed and decompressed sides are chunked and a worker holds one block
of each; peak per worker is tens of MB, which is what makes N workers fit in ~5 GB.

LITERALS ARE SPLIT BUT NOT UNESCAPED. Stage 1 records (lexical, lang, datatype) with the lexical
form still RDF-escaped, because unescaping correctly is a single left-to-right scan and belongs in
stage 2, where it runs over a filtered column instead of every row.

THE REDUNDANCY DECISIONS ARE NOT MADE HERE. rdfs:label vs type.object.name, and rdf:type vs
type.object.type, are per-subject set comparisons. Doing them in stage 1 would require a subject's
rows to land in one worker, which is exactly what a member seam breaks. Stage 2 does them as global
joins on subject, correct no matter which worker saw which row.
"""
import json
import multiprocessing as mp
import os
import sys
import time
import zlib

RAW = r"data\final_canonical\freebase_v3\_acquisition\raw\freebase-rdf-latest.gz"
IDX = r"data\final_canonical\freebase_v3\V3_GZIP_MEMBER_INDEX.json"
SHARDS = r"data\final_canonical\freebase_v3\pass_a\shards"

NS = b"<http://rdf.freebase.com/ns/"
NSL = len(NS)
RDFS = b"<http://www.w3.org/2000/01/rdf-schema#"
SYN = b"<http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# predicate -> (table, kind). One dict lookup per line; the ~70% of lines that miss stop there.
PRED = {
    NS + b"type.object.name>": ("name", "lit"),
    RDFS + b"label>": ("label", "lit"),
    NS + b"common.topic.alias>": ("alias", "lit"),
    NS + b"common.topic.description>": ("description", "lit"),
    NS + b"type.object.type>": ("type", "uri"),
    SYN + b"type>": ("rdf_type", "rdftype"),
    NS + b"type.object.key>": ("key", "lit1"),
    NS + b"type.property.reverse_property>": ("reverse_property", "uri"),
    NS + b"type.property.master_property>": ("master_property", "uri"),
    NS + b"type.property.schema>": ("property_schema", "kv:schema"),
    NS + b"type.property.expected_type>": ("property_schema", "kv:expected_type"),
    NS + b"type.property.unique>": ("property_schema", "kv:unique"),
    NS + b"type.property.delegated>": ("property_schema", "kv:delegated"),
    RDFS + b"domain>": ("property_schema", "kv:rdfs_domain"),
    RDFS + b"range>": ("property_schema", "kv:rdfs_range"),
    NS + b"freebase.type_hints.mediator>": ("type_hints", "kv:mediator"),
    NS + b"freebase.type_hints.included_types>": ("type_hints", "kv:included_types"),
    NS + b"freebase.type_hints.enumeration>": ("type_hints", "kv:enumeration"),
    NS + b"freebase.type_hints.deprecated>": ("type_hints", "kv:deprecated"),
}
TABLES = ("name", "label", "alias", "description", "type", "rdf_type", "key",
          "reverse_property", "master_property", "property_schema", "type_hints")


def split_literal(o):
    """b'"lex"@en' / b'"lex"^^<iri>' / b'"lex"'  ->  (lexical, lang, datatype), still escaped.

    Dispatched on the terminator so a quote inside the lexical form is never mistaken for the
    closing quote: a plain literal ends with '"', a typed one with '>', a tagged one with its tag.
    Returns None rather than guessing when the term is none of the three."""
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


class GzOut:
    """A gzip-compressed shard writer.

    Shards are compressed on the way out, not written raw. PASS A emits on the order of 10^9 rows;
    raw that is ~35 GB of TSV, and holding it alongside the Parquet it converts into would break a
    60 GB disk margin. zlib level 1 costs little against work that is already decompression-bound,
    and it is what the contract's disk discipline asks for: stream to compressed shards, convert,
    and delete a shard only once its conversion is validated."""

    __slots__ = ("f", "c", "buf", "n", "raw")

    def __init__(self, path):
        self.f = open(path, "wb", buffering=1 << 20)
        self.c = zlib.compressobj(1, zlib.DEFLATED, 31)
        self.buf = []
        self.n = 0
        self.raw = 0

    def write(self, b):
        self.buf.append(b)
        self.n += len(b)
        if self.n >= (1 << 22):
            self.raw += self.n
            self.f.write(self.c.compress(b"".join(self.buf)))
            self.buf = []
            self.n = 0

    def close(self):
        if self.buf:
            self.raw += self.n
            self.f.write(self.c.compress(b"".join(self.buf)))
        self.f.write(self.c.flush())
        self.f.close()


def member_lines(fh, lo, hi, state):
    """Yield decompressed chunks that always end on a line boundary.

    Never splits a line across yields, so the consumer can split on newline without a carry of its
    own. state[] collects what the stitcher needs: decompressed size, trailing partial line (empty
    for a line-aligned member) and any bytes left after the deflate stream ends."""
    d = zlib.decompressobj(31)
    carry = b""
    remaining = hi - lo
    fh.seek(lo)
    while remaining > 0:
        blk = fh.read(min(1 << 22, remaining))
        if not blk:
            break
        remaining -= len(blk)
        out = d.decompress(blk)
        if not out:
            continue
        state["dec"] += len(out)
        buf = carry + out
        j = buf.rfind(b"\n")
        if j < 0:
            carry = buf
            continue
        carry = buf[j + 1:]
        yield buf[:j]
    rest = d.flush()
    if rest:
        state["dec"] += len(rest)
        buf = carry + rest
        j = buf.rfind(b"\n")
        if j >= 0:
            carry = buf[j + 1:]
            yield buf[:j]
        else:
            carry = buf
    state["unused"] = len(d.unused_data)
    state["tail"] = carry          # empty iff the member ended exactly on a newline


def work(job):
    k, lo, hi = job
    os.makedirs(SHARDS, exist_ok=True)
    out_f = {t: GzOut(os.path.join(SHARDS, f"m{k:04d}_{t}.tsv.gz")) for t in TABLES}
    f_open = GzOut(os.path.join(SHARDS, f"m{k:04d}_openers.tsv.gz"))
    st = {"lines": 0, "blocks": 0, "rows": 0, "subject_not_ns": 0, "type_nonuri": 0,
          "name_nonliteral": 0, "literal_parse_failed": 0, "short_lines": 0}
    langs, dtypes = {}, {}
    first_subject = last_subject = None
    prev = None
    state = {"dec": 0, "unused": 0, "tail": b""}
    t0 = time.time()

    with open(RAW, "rb") as fh:
        for chunk in member_lines(fh, lo, hi, state):
            for ln in chunk.split(b"\n"):
                if not ln:
                    continue
                st["lines"] += 1
                a = ln.find(b"\t")
                if a < 0:
                    st["short_lines"] += 1
                    continue
                b = ln.find(b"\t", a + 1)
                if b < 0:
                    st["short_lines"] += 1
                    continue

                s = ln[:a]
                if s != prev:
                    f_open.write(s)
                    f_open.write(b"\n")
                    prev = s
                    st["blocks"] += 1
                    if first_subject is None:
                        first_subject = s
                    last_subject = s

                pk = PRED.get(ln[a + 1:b])
                if pk is None:
                    continue
                if not s.startswith(NS):
                    st["subject_not_ns"] += 1
                    continue
                su = s[NSL:-1]
                c = ln.find(b"\t", b + 1)
                o = ln[b + 1:c] if c > 0 else ln[b + 1:]
                tbl, kind = pk

                if kind == "lit" or kind == "lit1":
                    if o[:1] != b'"':
                        if tbl == "name":
                            st["name_nonliteral"] += 1
                        continue
                    L = split_literal(o)
                    if L is None:
                        st["literal_parse_failed"] += 1
                        continue
                    lex, lang, dt = L
                    if lang:
                        langs[lang] = langs.get(lang, 0) + 1
                    if dt:
                        dtypes[dt] = dtypes.get(dt, 0) + 1
                    if kind == "lit1":
                        out_f[tbl].write(su + b"\t" + lex + b"\n")
                    else:
                        out_f[tbl].write(su + b"\t" + lex + b"\t" + lang + b"\n")
                elif kind == "uri":
                    if not o.startswith(NS):
                        if tbl == "type":
                            st["type_nonuri"] += 1
                        continue
                    out_f[tbl].write(su + b"\t" + o[NSL:-1] + b"\n")
                elif kind == "rdftype":
                    # in-ns rdf:type objects MAY duplicate type.object.type; stage 2 decides that
                    # per subject. Out-of-ns is the RDF/OWL vocabulary layer and is always kept.
                    if o.startswith(NS):
                        out_f[tbl].write(su + b"\t" + o[NSL:-1] + b"\t1\n")
                    else:
                        out_f[tbl].write(su + b"\t" + (o[1:-1] if o[-1:] == b">" else o) + b"\t0\n")
                else:
                    key = kind[3:].encode()
                    if o[:1] == b'"':
                        L = split_literal(o)
                        if L is None:
                            st["literal_parse_failed"] += 1
                            continue
                        v = L[0]
                    else:
                        v = o[NSL:-1] if o.startswith(NS) else (o[1:-1] if o[-1:] == b">" else o)
                    out_f[tbl].write(su + b"\t" + key + b"\t" + v + b"\n")
                st["rows"] += 1

    raw_out = {t: out_f[t].raw for t in TABLES}
    for f in out_f.values():
        f.close()
    f_open.close()

    meta = {
        "member": k, "lo": lo, "hi": hi,
        "decompressed_bytes": state["dec"],
        "ends_with_newline": state["tail"] == b"",
        "unused_data_after_member": state["unused"],
        "straddle_tail": state["tail"].decode("utf-8", "replace") if state["tail"] else None,
        "first_subject": first_subject.decode("utf-8", "replace") if first_subject else None,
        "last_subject": last_subject.decode("utf-8", "replace") if last_subject else None,
        "langs": {k2.decode("utf-8", "replace"): v for k2, v in langs.items()},
        "dtypes": {k2.decode("utf-8", "replace"): v for k2, v in dtypes.items()},
        "raw_shard_bytes": raw_out,
        "elapsed_s": round(time.time() - t0, 1), **st,
    }
    with open(os.path.join(SHARDS, f"m{k:04d}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


def main():
    workers = 8
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    limit = None
    if "--members" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--members") + 1])

    idx = json.load(open(IDX, encoding="utf-8"))
    if idx.get("partial_scan"):
        raise SystemExit("member index is from a partial scan; PASS A needs the full index")
    starts = idx["member_starts"]
    total = idx["source_bytes"]
    bounds = [(i, starts[i], starts[i + 1] if i + 1 < len(starts) else total)
              for i in range(len(starts))]
    if limit:
        bounds = bounds[:limit]
    os.makedirs(SHARDS, exist_ok=True)

    t0 = time.time()
    metas = []
    print(f"{len(bounds)} members, {workers} workers", flush=True)
    with mp.Pool(workers) as pool:
        for m in pool.imap_unordered(work, bounds):
            metas.append(m)
            el = time.time() - t0
            gb = sum(x["decompressed_bytes"] for x in metas) / 1e9
            done = len(metas)
            print(f"  {done:4d}/{len(bounds)}  {gb:7.1f} GB decomp  {gb*1000/el:6.0f} MB/s  "
                  f"{sum(x['lines'] for x in metas)/1e6:8.1f}M lines  "
                  f"{sum(x['rows'] for x in metas)/1e6:7.1f}M rows  "
                  f"t={el/60:5.1f}m  eta={el/done*(len(bounds)-done)/60:5.1f}m", flush=True)

    metas.sort(key=lambda m: m["member"])
    with open(os.path.join(SHARDS, "_all_meta.json"), "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=1)
    summary = {
        "members": len(metas),
        "lines": sum(m["lines"] for m in metas),
        "rows": sum(m["rows"] for m in metas),
        "blocks_within_members": sum(m["blocks"] for m in metas),
        "decompressed_bytes": sum(m["decompressed_bytes"] for m in metas),
        "members_not_newline_terminated": sum(0 if m["ends_with_newline"] else 1 for m in metas),
        "members_with_unused_data": sum(1 for m in metas if m["unused_data_after_member"]),
        "short_lines": sum(m["short_lines"] for m in metas),
        "literal_parse_failed": sum(m["literal_parse_failed"] for m in metas),
        "subject_not_ns": sum(m["subject_not_ns"] for m in metas),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
