"""PASS B: the structural edge stream. One member-parallel pass over the raw dump.

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/v3_pass_b.py [--workers N]

PASS B emits graph facts and nothing else: src_uid | rel_uid | dst_uid | dst_kind, for every triple
whose predicate PASS A did not already consume as names, types, aliases, keys or schema plumbing.
That split is the whole point -- 3.0e9 RDF statements are not 3.0e9 edges, and treating them as if
they were is what turns a graph build into an unusable one.

WHAT COUNTS AS AN EDGE. Everything that is not one of PASS A's nineteen metadata predicates. That
deliberately includes type.type.instance, which is the declared reverse of type.object.type and so
restates the type assertions PASS A already holds. PASS B does not make that judgement: it is a
faithful projection of the source, and PASS D -- which owns the frozen reverse-property map -- is
where a relation whose canonical direction is metadata gets reclassified. Baking the decision in
here would mean the extraction pass silently disagreed with the source it is supposed to mirror.

UID MINTING. src and dst are 64-bit UIDs over the namespace-stripped identifier; literals mint from
(lexical_form, datatype, language) exactly as the contract requires, never from the visible string
alone, so that "5"^^xsd:int and "5"@en and plain "5" stay three distinct nodes. The subject UID
space was already proven collision-free over all 121,616,633 subjects before this pass was allowed
to run, and PASS C re-checks injectivity across the whole node universe including objects.

NODE DICTIONARIES WITHOUT A SECOND EDGE STREAM. Writing (uid, string) for all ~2e9 object positions
would cost more than the edges do. Instead the sorted subject-UID array is memory-mapped and each
batch of object UIDs is tested against it vectorised: a UID that is a subject somewhere is already
named by the openers, so only the misses need writing. Repeat suppression on those uses a
direct-mapped cache rather than a growing set -- a slot match is an exact UID match, so the cache
can only ever cause a node to be written twice (PASS C folds duplicates), never to be dropped. That
keeps a worker's memory flat regardless of how many distinct objects a member turns out to hold.

SOURCE ORDER IS KEPT. Rows are written in the order they occur, which means src is constant across
a subject block and Parquet's RLE makes that column nearly free. Sorting would cost more than the
column does.
"""
import collections
import json
import multiprocessing as mp
import os
import sys
import time
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

IDX = r"data\final_canonical\freebase_v3\V3_GZIP_MEMBER_INDEX.json"
UIDS = r"data\final_canonical\freebase_v3\pass_b\subject_uids.npy"
OUTDIR = r"data\final_canonical\freebase_v3\pass_b"
EDGES = os.path.join(OUTDIR, "edges")
DICTS = os.path.join(OUTDIR, "objdict")
LITS = os.path.join(OUTDIR, "literals")
METAD = os.path.join(OUTDIR, "meta")
REPORT = r"data\final_canonical\freebase_v3\V3_PASS_B_REPORT.json"

NS = b"<http://rdf.freebase.com/ns/"
NSL = len(NS)
RDFS = b"<http://www.w3.org/2000/01/rdf-schema#"
SYN = b"<http://www.w3.org/1999/02/22-rdf-syntax-ns#"

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

K_NS, K_EXT, K_LIT = 0, 1, 2
CHUNK = 1 << 22
FLUSH = 1 << 19          # edge rows per Parquet row group; also bounds the pending
                         # object-identifier list, which is what actually costs memory here
CACHE_BITS = 22          # 4.2M slots, 33 MB, per dictionary


def member_lines(fh, lo, hi):
    """Yield decompressed blocks that always end on a line boundary; both sides streamed."""
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
    """Terminator-dispatched, identical to PASS A, so both passes agree on literal identity."""
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
    """Level-1 gzip shard writer. Dictionaries are written compressed on the way out; holding them
    raw alongside the edges they accompany is exactly the pattern the disk discipline forbids."""

    __slots__ = ("f", "c", "buf", "n", "raw", "rows")

    def __init__(self, path):
        self.f = open(path, "wb")
        self.c = zlib.compressobj(1, zlib.DEFLATED, 31)
        self.buf = []
        self.n = 0
        self.raw = 0
        self.rows = 0

    def write(self, b):
        self.buf.append(b)
        self.n += len(b)
        self.rows += 1
        if self.n >= (1 << 22):
            self.raw += self.n
            self.f.write(self.c.compress(b"".join(self.buf)))
            self.buf = []
            self.n = 0

    def close(self):
        self.raw += self.n
        if self.buf:
            self.f.write(self.c.compress(b"".join(self.buf)))
        self.f.write(self.c.flush())
        self.f.close()


def work(job):
    k, lo, hi, src, nsub = job
    t0 = time.time()
    subj_uids = np.load(UIDS, mmap_mode="r")
    if subj_uids.size != nsub:
        raise SystemExit(f"subject UID array has {subj_uids.size} entries, expected {nsub}")

    ew = None
    epath = os.path.join(EDGES, f"m{k:04d}.parquet")
    dw = GzOut(os.path.join(DICTS, f"m{k:04d}.tsv.gz"))
    lw = GzOut(os.path.join(LITS, f"m{k:04d}.tsv.gz"))

    # Direct-mapped repeat suppressors. A hit is an exact UID equality, so these can only cause a
    # node to be emitted more than once (PASS C folds that), never suppress a distinct node.
    mask = (1 << CACHE_BITS) - 1
    dcache = np.zeros(1 << CACHE_BITS, dtype=np.int64)
    lcache = np.zeros(1 << CACHE_BITS, dtype=np.int64)
    # A UID of exactly 0 is indistinguishable from an empty slot in a zero-filled cache, so such a
    # node would be suppressed on every occurrence and never written at all. hash() can return 0,
    # and the loss would be silent here -- it would only surface much later as a dangling edge
    # endpoint -- so the two zero cases are tracked out of band rather than left to chance.
    zero_seen = [False, False]

    S = np.empty(FLUSH, dtype=np.int64)
    R = np.empty(FLUSH, dtype=np.int64)
    D = np.empty(FLUSH, dtype=np.int64)
    KD = np.empty(FLUSH, dtype=np.int8)
    # Object identifiers for the pending batch, so the subject-membership test runs vectorised on
    # a whole row group at a time instead of once per row.
    pend = [None] * FLUSH
    n = 0

    rel_counts = collections.Counter()
    kinds = collections.Counter()
    lines = struct = meta = malformed = 0
    lit_unparsed = obj_other = 0
    pred_not_ns = 0
    samples = []
    edges_written = 0
    dict_rows = lit_rows = 0
    subject_hits = 0

    def flush():
        nonlocal ew, n, edges_written, dict_rows, subject_hits
        if n == 0:
            return
        d = D[:n]
        kd = KD[:n]
        # Which of this batch's non-literal objects are subjects somewhere in the dump? searchsorted
        # over the memory-mapped array is one vectorised binary search for the whole row group.
        isuri = kd != K_LIT
        if isuri.any():
            q = d[isuri]
            pos = np.searchsorted(subj_uids, q)
            np.clip(pos, 0, subj_uids.size - 1, out=pos)
            hit = subj_uids[pos] == q
            subject_hits += int(hit.sum())
            miss_idx = np.flatnonzero(isuri)[~hit]
        else:
            miss_idx = np.empty(0, dtype=np.int64)
        for i in miss_idx.tolist():
            u = int(D[i])
            slot = u & mask
            if u == 0:
                if zero_seen[0]:
                    continue
                zero_seen[0] = True
            elif dcache[slot] == u:
                continue
            else:
                dcache[slot] = u
            dw.write(b"%d\t%d\t%s\n" % (u, KD[i], pend[i]))
            dict_rows += 1
        tb = pa.Table.from_arrays(
            [pa.array(S[:n]), pa.array(R[:n]), pa.array(d), pa.array(kd)],
            names=["src", "rel", "dst", "dst_kind"])
        if ew is None:
            ew = pq.ParquetWriter(epath, tb.schema, compression="zstd")
        ew.write_table(tb)
        edges_written += n
        n = 0

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
                # Split twice and strip the terminator rather than demanding four fields: a literal
                # containing a raw tab stays intact instead of being rejected, and the line ending
                # is still validated.
                if rest[-2:] != b"\t.":
                    malformed += 1
                    continue
                if p in META_PREDS:
                    meta += 1
                    continue
                o = rest[:-2]
                struct += 1

                sid = s[NSL:-1] if s[:NSL] == NS else s[1:-1]
                if p[:NSL] == NS:
                    pid = p[NSL:-1]
                else:
                    pid = p[1:-1]
                    pred_not_ns += 1
                rel_counts[pid] += 1

                c = o[:1]
                if c == b"<":
                    if o[:NSL] == NS:
                        oid = o[NSL:-1]
                        kind = K_NS
                    else:
                        oid = o[1:-1]
                        kind = K_EXT
                    kinds[kind] += 1
                    du = hash(oid)
                elif c == b'"':
                    t = split_literal(o)
                    if t is None:
                        lit_unparsed += 1
                        if len(samples) < 20:
                            samples.append(repr(line[:200]))
                        continue
                    lex, lang, dt = t
                    # Literal identity is (lexical form, datatype, language), never the visible
                    # string: this is what keeps "5", "5"@en and "5"^^xsd:int three distinct nodes.
                    # The joined form is unambiguous because N-Triples escapes a literal tab.
                    oid = lex + b"\t" + dt + b"\t" + lang
                    kind = K_LIT
                    kinds[kind] += 1
                    du = hash(oid)
                    slot = du & mask
                    if du == 0:
                        if not zero_seen[1]:
                            zero_seen[1] = True
                            lw.write(b"%d\t%s\n" % (du, o))
                            lit_rows += 1
                    elif lcache[slot] != du:
                        lcache[slot] = du
                        lw.write(b"%d\t%s\n" % (du, o))
                        lit_rows += 1
                else:
                    obj_other += 1
                    if len(samples) < 20:
                        samples.append(repr(line[:200]))
                    continue

                S[n] = hash(sid)
                R[n] = hash(pid)
                D[n] = du
                KD[n] = kind
                pend[n] = oid
                n += 1
                if n == FLUSH:
                    flush()

    flush()
    if ew is not None:
        ew.close()
    dw.close()
    lw.close()

    m = {"member": k, "elapsed_s": round(time.time() - t0, 1),
         "lines": lines, "structural": struct, "metadata": meta, "malformed": malformed,
         "edges_written": edges_written, "literal_unparsed": lit_unparsed,
         "object_blank_or_other": obj_other, "predicate_not_in_ns": pred_not_ns,
         "kinds": {"uri_ns": kinds[K_NS], "uri_ext": kinds[K_EXT], "literal": kinds[K_LIT]},
         "objdict_rows": dict_rows, "literal_rows": lit_rows,
         "object_uris_that_are_subjects": subject_hits,
         "zero_uid_seen": zero_seen,
         "samples": samples,
         "rel_counts": {kk.decode("utf-8", "replace"): v for kk, v in rel_counts.items()}}
    with open(os.path.join(METAD, f"m{k:04d}.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(m, fh)
    m.pop("rel_counts")
    m.pop("samples")
    return m


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 is required: without it each worker mints different "
                         "UIDs for the same node and the edge stream is incoherent.")
    workers = 7
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    only = None
    if "--members" in sys.argv:
        only = [int(x) for x in sys.argv[sys.argv.index("--members") + 1].split(",")]

    for d in (EDGES, DICTS, LITS, METAD):
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(UIDS):
        raise SystemExit("subject_uids.npy missing; run v3_pass_b_subject_uids.py first")
    nsub = int(np.load(UIDS, mmap_mode="r").size)

    idx = json.load(open(IDX, encoding="utf-8"))
    starts = idx["member_starts"]
    ends = starts[1:] + [idx["source_bytes"]]
    src = idx["source"]
    ks = only if only is not None else list(range(len(starts)))
    jobs = [(k, starts[k], ends[k], src, nsub) for k in ks]
    print(f"PASS B: {len(jobs)} members, {workers} workers, {nsub:,} subject UIDs", flush=True)

    t0 = time.time()
    res = []
    with mp.Pool(min(workers, len(jobs))) as pool:
        for r in pool.imap_unordered(work, jobs):
            res.append(r)
            print(f"  m{r['member']:04d}  {r['lines']/1e6:6.2f}M lines  "
                  f"{r['edges_written']/1e6:6.2f}M edges  dict {r['objdict_rows']/1e6:5.2f}M  "
                  f"lit {r['literal_rows']/1e6:6.2f}M  {r['elapsed_s']:5.0f}s "
                  f"({len(res)}/{len(jobs)})", flush=True)

    rel = collections.Counter()
    for r in sorted(res, key=lambda x: x["member"]):
        mm = json.load(open(os.path.join(METAD, f"m{r['member']:04d}.json"), encoding="utf-8"))
        rel.update(mm["rel_counts"])

    def tot(f):
        return sum(r[f] for r in res)

    edges = tot("edges_written")
    # Relations are keyed by the same 64-bit UID the edge stream carries, so no remapping pass is
    # needed after the merge. Injectivity is checked here rather than assumed: two distinct
    # predicate strings sharing a UID would make every edge on either of them ambiguous.
    ruids = {p: hash(p.encode()) for p in rel}
    byuid = collections.defaultdict(list)
    for p, u in ruids.items():
        byuid[u].append(p)
    rel_collisions = {u: v for u, v in byuid.items() if len(v) > 1}

    rt = pa.Table.from_arrays(
        [pa.array([ruids[p] for p in sorted(rel)], type=pa.int64()),
         pa.array(sorted(rel), type=pa.string()),
         pa.array([rel[p] for p in sorted(rel)], type=pa.int64())],
        names=["rel_uid", "relation", "edge_count"])
    pq.write_table(rt, os.path.join(OUTDIR, "relations_raw.parquet"), compression="zstd")

    doc = {
        "schema": "V3_PASS_B_REPORT/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "members": len(res),
        "workers": workers,
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "LINES": {
            "lines_read": tot("lines"),
            "metadata_lines_skipped": tot("metadata"),
            "structural_lines": tot("structural"),
            "malformed_lines": tot("malformed"),
            "stage1_lines_recorded": 3008314716,
            "line_count_matches_stage1": tot("lines") == 3008314716,
        },
        "EDGES": {
            "edges_written": edges,
            "object_kinds": {
                "uri_ns": sum(r["kinds"]["uri_ns"] for r in res),
                "uri_ext": sum(r["kinds"]["uri_ext"] for r in res),
                "literal": sum(r["kinds"]["literal"] for r in res)},
            "dropped_literal_unparsed": tot("literal_unparsed"),
            "dropped_object_blank_or_other": tot("object_blank_or_other"),
            "accounts_for_all_structural_lines":
                edges + tot("literal_unparsed") + tot("object_blank_or_other") == tot("structural"),
            "predicate_not_in_namespace": tot("predicate_not_in_ns"),
        },
        "NODE_DICTIONARIES": {
            "object_uris_already_subjects": tot("object_uris_that_are_subjects"),
            "objdict_rows_written": tot("objdict_rows"),
            "literal_rows_written": tot("literal_rows"),
            "reading": "object_uris_already_subjects are objects whose UID is in the frozen subject "
                       "array, so the openers already name them and nothing was written. "
                       "objdict_rows are the object-only nodes. Both dictionary files may repeat a "
                       "node: the repeat suppressor is a direct-mapped cache, exact on a hit and "
                       "merely wasteful on a miss, and PASS C folds the duplicates.",
        },
        "RELATIONS": {
            "distinct_relations": len(rel),
            "uid_collisions": {str(u): v for u, v in rel_collisions.items()},
            "COLLISION_FREE": not rel_collisions,
            "top_40": dict(rel.most_common(40)),
        },
        "PER_MEMBER": sorted(res, key=lambda x: x["member"]),
    }
    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps({k2: doc[k2] for k2 in
                      ("LINES", "EDGES", "NODE_DICTIONARIES")}, indent=1))
    print("distinct relations:", len(rel), " collision-free:", not rel_collisions)


if __name__ == "__main__":
    main()
