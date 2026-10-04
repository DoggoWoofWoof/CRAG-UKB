"""PASS C: the node universe. External union of every subject, every object and every literal.

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/v3_pass_c.py [--parts P]

This is the pass that answers the question the whole build exists to answer: how many nodes does
Freebase actually have, by kind. Every earlier number -- the ~60.4M lower bound included -- was a
count of something narrower than the graph: entities that survived a filter, or entities that
happened to carry a name. The union below is over the raw source, so nothing is excluded for being
unnamed, untyped, mediator-shaped or literal.

THREE SOURCES, THREE DIFFERENT GUARANTEES, SO THREE DIFFERENT TREATMENTS.

  SUBJECTS come from the openers and are already known distinct (contiguity established it) and
  already proven collision-free in the 64-bit UID space over all 121,616,633 of them. They are
  streamed straight into the node table; deduplicating them again would only cost time.

  OBJECT-ONLY URIS come from PASS B, which wrote a node only when its UID was absent from the
  subject array. That test was exact, but the repeat suppressor in front of it was a direct-mapped
  cache, so the same node may appear many times across members. These need folding.

  LITERALS were suppressed by the same kind of cache and need the same folding. Their identity is
  (lexical form, datatype, language) as minted in PASS B, never the visible string, so "5", "5"@en
  and "5"^^xsd:int remain three nodes here exactly as they were three nodes there.

WHY PARTITION RATHER THAN SORT. Folding needs all occurrences of a UID in one place, not in order.
Hash-partitioning on the low bits of the UID puts them there in one streaming pass, and each part is
then small enough to fold in memory. An external sort would order data nobody needs ordered.

INJECTIVITY IS CHECKED, NOT ASSUMED. Inside every part, a UID that maps to two different strings is
a hash collision, and a collision would make every edge on that node ambiguous. The check costs
nothing here because the strings are already grouped, and it is the reason the 64-bit UID scheme is
defensible rather than merely convenient.

CVT CLASSIFICATION IS BY DECLARATION. A node is a CVT_MEDIATOR if and only if one of its
type.object.type values is a type declared with freebase.type_hints.mediator. No degree, naming or
shape heuristic participates -- those are how a build ends up with an entity count it cannot defend.
"""
import collections
import glob
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PA = r"data\final_canonical\freebase_v3\pass_a"
PB = r"data\final_canonical\freebase_v3\pass_b"
PC = r"data\final_canonical\freebase_v3\pass_c"
BUCKETS = os.path.join(PC, "_buckets")
NODES = os.path.join(PC, "nodes")
FREEZE = r"data\final_canonical\freebase_v3\V3_PASS_A_SCHEMA_FREEZE.json"
CONTIG = r"data\final_canonical\freebase_v3\V3_SUBJECT_CONTIGUITY.json"
REPORT = r"data\final_canonical\freebase_v3\V3_PASS_C_NODE_UNIVERSE.json"

NS = b"<http://rdf.freebase.com/ns/"
NSL = len(NS)

# Node kinds, in the order the contract asks them to be reported. LITERAL and CVT_MEDIATOR are
# tested before ENTITY_MID because a CVT also carries an MID: the more specific claim wins.
LITERAL, CVT_MEDIATOR, ENTITY_MID, SCHEMA_TYPE, SCHEMA_PROPERTY, SCHEMA_OTHER, \
    EXTERNAL_URI, OTHER = range(8)
KIND_NAME = {LITERAL: "LITERAL", CVT_MEDIATOR: "CVT_MEDIATOR", ENTITY_MID: "ENTITY_MID",
             SCHEMA_TYPE: "SCHEMA_TYPE", SCHEMA_PROPERTY: "SCHEMA_PROPERTY",
             SCHEMA_OTHER: "SCHEMA_OTHER", EXTERNAL_URI: "EXTERNAL_URI", OTHER: "OTHER"}


def classify_uri(sid):
    """Classify a namespace-stripped Freebase identifier from its own form.

    MIDs are 'm.' (topics) and 'g.' (machine ids). Everything else in the namespace is a schema
    path, whose segment count says what it names: two is a type, three a property, and the deeper
    ones are the user/base namespaces Freebase let contributors create."""
    if sid[:2] in ("m.", "g.") and len(sid) > 2:
        return ENTITY_MID
    if "." in sid:
        n = sid.count(".") + 1
        if n == 2:
            return SCHEMA_TYPE
        if n == 3:
            return SCHEMA_PROPERTY
        return SCHEMA_OTHER
    return OTHER


def partition_one(job):
    """Stream one gz shard into P bucket files keyed by the low bits of the UID."""
    src, kind, parts, tag = job
    outs = [gzip.open(os.path.join(BUCKETS, f"{tag}_{i:03d}.tsv.gz"), "ab", compresslevel=1)
            for i in range(parts)]
    buf = [[] for _ in range(parts)]
    mask = parts - 1
    n = 0
    with gzip.open(src, "rb") as fh:
        for line in fh:
            if not line.rstrip(b"\n"):
                continue
            u = int(line[:line.index(b"\t")])
            b = u & mask
            buf[b].append(line)
            n += 1
            if len(buf[b]) >= 20000:
                outs[b].write(b"".join(buf[b]))
                buf[b] = []
    for i in range(parts):
        if buf[i]:
            outs[i].write(b"".join(buf[i]))
        outs[i].close()
    return src, kind, n


def fold_part(job):
    """Fold one bucket: dedup by UID, verify UID -> string is injective, emit a node table part."""
    i, is_lit, subj_path, nsub = job
    tag = "lit" if is_lit else "obj"
    files = sorted(glob.glob(os.path.join(BUCKETS, f"{tag}*_{i:03d}.tsv.gz")))
    seen = {}
    collisions = []
    rows = 0
    for p in files:
        with gzip.open(p, "rb") as fh:
            for line in fh:
                line = line.rstrip(b"\n")
                if not line:
                    continue
                rows += 1
                if is_lit:
                    j = line.index(b"\t")
                    u = int(line[:j])
                    s = line[j + 1:]
                    k = LITERAL
                else:
                    j = line.index(b"\t")
                    j2 = line.index(b"\t", j + 1)
                    u = int(line[:j])
                    k = EXTERNAL_URI if line[j + 1:j2] == b"1" else None
                    s = line[j2 + 1:]
                prev = seen.get(u)
                if prev is None:
                    seen[u] = (s, k)
                elif prev[0] != s:
                    if len(collisions) < 20:
                        collisions.append({"uid": u,
                                           "a": prev[0].decode("utf-8", "replace")[:120],
                                           "b": s.decode("utf-8", "replace")[:120]})
    # A node written here must not also be a subject: PASS B only wrote misses, so any overlap
    # means the membership test and this fold disagree, which would double-count the universe.
    uids = np.fromiter(seen.keys(), dtype=np.int64, count=len(seen))
    overlap = 0
    if uids.size and not is_lit:
        subj = np.load(subj_path, mmap_mode="r")
        pos = np.searchsorted(subj, uids)
        np.clip(pos, 0, subj.size - 1, out=pos)
        overlap = int((subj[pos] == uids).sum())
    strs, kinds = [], []
    for u in uids.tolist():
        s, k = seen[u]
        strs.append(s.decode("utf-8", "replace"))
        kinds.append(k if k is not None else classify_uri(strs[-1]))
    tb = pa.Table.from_arrays(
        [pa.array(uids), pa.array(strs, type=pa.string()),
         pa.array(kinds, type=pa.int8())],
        names=["node_uid", "node_id", "kind"])
    pq.write_table(tb, os.path.join(NODES, f"{tag}_{i:03d}.parquet"), compression="zstd")
    return {"part": i, "is_lit": is_lit, "rows_read": rows, "distinct": int(uids.size),
            "collisions": collisions, "subject_overlap": overlap}


def node_id(term):
    """Bracketed URI -> the identifier the UID is minted from. Byte-identical to the function that
    minted subject_uids.npy in PASS B; if these two ever drift, every subject UID here misses its
    counterpart there and nothing downstream joins."""
    if term[:NSL] == NS:
        return term[NSL:-1]
    if term[:1] == b"<" and term[-1:] == b">":
        return term[1:-1]
    return term


def subject_nodes(job):
    """Emit node rows for one run of members' openers. Subjects are already distinct."""
    wid, members, cvt_path = job
    shards = os.path.join(PA, "shards")
    metas = {}
    for k in members:
        metas[k] = json.load(open(os.path.join(shards, f"m{k:04d}_meta.json"), encoding="utf-8"))
    cvt = np.load(cvt_path, mmap_mode="r") if os.path.exists(cvt_path) else None
    uu, ss, kk = [], [], []
    w = None
    out = os.path.join(NODES, f"subj_{wid:03d}.parquet")
    n = 0
    counts = collections.Counter()

    def flush():
        nonlocal uu, ss, kk, w, n
        if not uu:
            return
        u = np.array(uu, dtype=np.int64)
        k = np.array(kk, dtype=np.int8)
        if cvt is not None and cvt.size:
            pos = np.searchsorted(cvt, u)
            np.clip(pos, 0, cvt.size - 1, out=pos)
            ismed = cvt[pos] == u
            k[ismed] = CVT_MEDIATOR
        for v in k.tolist():
            counts[v] += 1
        tb = pa.Table.from_arrays([pa.array(u), pa.array(ss, type=pa.string()), pa.array(k)],
                                  names=["node_uid", "node_id", "kind"])
        if w is None:
            w = pq.ParquetWriter(out, tb.schema, compression="zstd")
        w.write_table(tb)
        n += tb.num_rows
        uu, ss, kk = [], [], []

    for k in members:
        prev_meta_path = os.path.join(shards, f"m{k-1:04d}_meta.json")
        prev = None
        if k > 0 and os.path.exists(prev_meta_path):
            prev = json.load(open(prev_meta_path, encoding="utf-8"))["last_subject"]
        skip = prev.encode() if prev is not None else None
        first = True
        with gzip.open(os.path.join(shards, f"m{k:04d}_openers.tsv.gz"), "rb") as fh:
            for line in fh:
                s = line.rstrip(b"\n")
                if not s:
                    continue
                if first:
                    first = False
                    if skip is not None and s == skip:
                        continue
                sid = node_id(s)
                t = sid.decode("utf-8", "replace")
                uu.append(hash(sid))
                ss.append(t)
                kk.append(classify_uri(t))
                if len(uu) >= (1 << 19):
                    flush()
    flush()
    if w is not None:
        w.close()
    return {"worker": wid, "members": len(members), "rows": n,
            "kinds": {int(a): b for a, b in counts.items()}}


def build_cvt_set():
    """UIDs of every subject carrying a declared mediator type. Sorted, for searchsorted.

    THE NAME SPACE IS THE WHOLE PROBLEM HERE. freebase.type_hints.mediator is declared ON A MID:
    the raw source says <ns/m.0101csmw> freebase.type_hints.mediator "true". But type.object.type
    names its object by SCHEMA PATH, never by MID -- measured over all 254,946,431 rows as 0 MID /
    254,946,431 path. So matching the declared MIDs against the type column directly returns zero
    rows, and returns them silently: no error, no warning, just a graph with no CVTs in a source
    that is full of them. PASS A's freeze resolved 1,649 of the 2,211 declared MIDs to their paths
    through two independent joins that agreed 8,866/8,866, and those paths are what is matched here.

    The 562 that did not resolve are carried into the report rather than dropped quietly: they are a
    known, bounded under-count of the mediator declaration, not an unknown one.
    """
    fr = json.load(open(FREEZE, encoding="utf-8"))
    dec = fr["DECLARATIONS"]
    med = set(dec["MEDIATOR_TYPE_PATHS"])
    stats = {"declared_mediator_mids": len(dec["MEDIATOR_TYPE_MIDS"]),
             "resolved_to_type_paths": len(med),
             "unresolved_mids": len(dec["MEDIATOR_MIDS_UNRESOLVED"])}
    if not med:
        raise SystemExit("no mediator type paths in the freeze; refusing to build a CVT-free graph")

    medset = pa.array(sorted(med), type=pa.string())
    pf = pq.ParquetFile(os.path.join(PA, "type.parquet"))
    out = []
    matched_rows = 0
    paths_hit = set()
    for batch in pf.iter_batches(batch_size=1 << 20, columns=["subject", "type"]):
        m = pc.is_in(batch.column(1), value_set=medset)
        if not pc.any(m).as_py():
            continue
        subs = pc.filter(batch.column(0), m)
        matched_rows += len(subs)
        paths_hit.update(pc.unique(pc.filter(batch.column(1), m)).to_pylist())
        # type.parquet stores the namespace-stripped identifier already, so the UID is minted from
        # the stored value as-is. Prefixing it back would mint a UID for a string that exists
        # nowhere else in the build.
        out.append(np.fromiter((hash(x.encode()) for x in subs.to_pylist()),
                               dtype=np.int64, count=len(subs)))
    stats["type_object_type_rows_matched"] = matched_rows
    stats["mediator_paths_with_at_least_one_instance"] = len(paths_hit)
    stats["mediator_paths_never_instantiated"] = len(med) - len(paths_hit)
    if not out:
        raise SystemExit("mediator type paths matched zero type.object.type rows -- this is the "
                         "name-space failure the freeze exists to prevent, not an empty result")
    a = np.unique(np.concatenate(out))
    stats["cvt_nodes"] = int(a.size)

    # Every subject of a type.object.type triple is by construction a subject block opener, so each
    # CVT UID must be present in the subject universe. If the two sides encoded the identifier even
    # slightly differently, this drops well below 100% and says so.
    sp = os.path.join(PB, "subject_uids.npy")
    if os.path.exists(sp):
        su = np.load(sp, mmap_mode="r")
        pos = np.searchsorted(su, a)
        np.clip(pos, 0, su.size - 1, out=pos)
        hit = int((su[pos] == a).sum())
        stats["cvt_uids_found_in_subject_universe"] = hit
        stats["CVT_UID_ENCODING_AGREES"] = hit == a.size
        if hit != a.size:
            raise SystemExit(f"only {hit}/{a.size} CVT UIDs appear in the subject universe; the "
                             f"identifier encodings disagree between PASS A and PASS B")
    return a, stats


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 is required so UIDs match those PASS B minted.")
    # 128 rather than 64: the fold holds one Python dict per part, and at 64 parts the literal
    # buckets put roughly 2.4 GB across seven workers on a machine with about 4.5 GB free. Halving
    # the part size halves that, and the only cost is more bucket files.
    parts = 128
    if "--parts" in sys.argv:
        parts = int(sys.argv[sys.argv.index("--parts") + 1])
    if parts & (parts - 1):
        raise SystemExit("--parts must be a power of two: bucketing masks the low bits.")
    workers = 7
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    for d in (PC, BUCKETS, NODES):
        os.makedirs(d, exist_ok=True)
    t0 = time.time()

    subj_path = os.path.join(PB, "subject_uids.npy")
    nsub = int(np.load(subj_path, mmap_mode="r").size)

    # ---- CVT declaration set ----
    print("building CVT set from declared mediator types", flush=True)
    cvt, cvt_stats = build_cvt_set()
    cvt_path = os.path.join(PC, "cvt_uids.npy")
    np.save(cvt_path, cvt)
    print(f"  {cvt_stats['resolved_to_type_paths']} mediator type paths -> {cvt.size:,} CVT nodes "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- partition object dictionaries and literals ----
    jobs = [(p, "obj", parts, "obj" + os.path.basename(p)[1:5])
            for p in sorted(glob.glob(os.path.join(PB, "objdict", "m*.tsv.gz")))]
    jobs += [(p, "lit", parts, "lit" + os.path.basename(p)[1:5])
             for p in sorted(glob.glob(os.path.join(PB, "literals", "m*.tsv.gz")))]
    print(f"partitioning {len(jobs)} shards into {parts} buckets", flush=True)
    read = collections.Counter()
    done = 0
    with mp.Pool(workers) as pool:
        for src, kind, n in pool.imap_unordered(partition_one, jobs):
            read[kind] += n
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}  t={time.time()-t0:.0f}s", flush=True)

    # ---- fold ----
    print("folding buckets", flush=True)
    fj = [(i, False, subj_path, nsub) for i in range(parts)]
    fj += [(i, True, subj_path, nsub) for i in range(parts)]
    folds = []
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(fold_part, fj):
            folds.append(r)
            if len(folds) % 32 == 0:
                print(f"  {len(folds)}/{len(fj)}  t={time.time()-t0:.0f}s", flush=True)

    # ---- subject nodes ----
    print("emitting subject nodes", flush=True)
    contig = json.load(open(CONTIG, encoding="utf-8"))
    ks = sorted(int(os.path.basename(p)[1:5])
                for p in glob.glob(os.path.join(PA, "shards", "m*_openers.tsv.gz")))
    chunks = [[] for _ in range(workers)]
    for i, k in enumerate(ks):
        chunks[i % workers].append(k)
    sres = []
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(subject_nodes,
                                     [(w, c, cvt_path) for w, c in enumerate(chunks) if c]):
            sres.append(r)
            print(f"  worker {r['worker']} {r['rows']:,} subject nodes", flush=True)

    obj_distinct = sum(f["distinct"] for f in folds if not f["is_lit"])
    lit_distinct = sum(f["distinct"] for f in folds if f["is_lit"])
    subj_rows = sum(r["rows"] for r in sres)
    collisions = [c for f in folds for c in f["collisions"]]
    overlap = sum(f["subject_overlap"] for f in folds)

    kinds = collections.Counter()
    for r in sres:
        for a, b in r["kinds"].items():
            kinds[int(a)] += b
    for p in glob.glob(os.path.join(NODES, "obj_*.parquet")) + \
            glob.glob(os.path.join(NODES, "lit_*.parquet")):
        t = pq.read_table(p, columns=["kind"])
        for a, b in collections.Counter(t.column(0).to_pylist()).items():
            kinds[int(a)] += b

    total = subj_rows + obj_distinct + lit_distinct
    doc = {
        "schema": "V3_PASS_C_NODE_UNIVERSE/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "parts": parts,
        "NODE_COUNTS": {KIND_NAME[k]: v for k, v in sorted(kinds.items())},
        "TOTAL_NODES": total,
        "SOURCES": {
            "subjects": subj_rows,
            "subjects_expected": contig["DISTINCT_SUBJECTS_N"],
            "subjects_match": subj_rows == contig["DISTINCT_SUBJECTS_N"],
            "object_only_uris_distinct": obj_distinct,
            "object_dictionary_rows_read": read["obj"],
            "literals_distinct": lit_distinct,
            "literal_rows_read": read["lit"],
            "reading": "rows_read exceeds distinct because PASS B's repeat suppressor is a "
                       "direct-mapped cache: it can emit a node twice but never drop one. The "
                       "difference is the cache's miss rate, not lost data.",
        },
        "INTEGRITY": {
            "uid_string_collisions": len(collisions),
            "UID_INJECTIVE": not collisions,
            "object_nodes_that_are_also_subjects": overlap,
            "OBJECT_SUBJECT_DISJOINT": overlap == 0,
            "examples": collisions[:10],
            "reading": "UID_INJECTIVE false would mean two distinct nodes share a 64-bit UID, which "
                       "makes every edge on either ambiguous; the remedy is to re-run PASS B under "
                       "a salt, not to merge the nodes. OBJECT_SUBJECT_DISJOINT false would mean "
                       "the node universe double counts.",
        },
        "CVT": dict(cvt_stats, **{
            "authority": "freebase.type_hints.mediator = true, frozen in "
                         "V3_PASS_A_SCHEMA_FREEZE.json. No shape or degree heuristic participates.",
            "name_space": "the declaration is on a MID; type.object.type names types by path only. "
                          "The freeze's MID->path resolution is what makes the two joinable, and "
                          "the unresolved MIDs are a bounded under-count stated here rather than "
                          "hidden in a zero.",
        }),
        "ANSWERS": {
            "ENTITY_MID": kinds.get(ENTITY_MID, 0),
            "CVT_MEDIATOR": kinds.get(CVT_MEDIATOR, 0),
            "LITERAL": kinds.get(LITERAL, 0),
            "OTHER": sum(v for k, v in kinds.items()
                         if k not in (ENTITY_MID, CVT_MEDIATOR, LITERAL)),
            "TOTAL_NODES": total,
        },
    }
    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps({k: doc[k] for k in
                      ("NODE_COUNTS", "TOTAL_NODES", "SOURCES", "INTEGRITY", "CVT", "ANSWERS")},
                     indent=1))


if __name__ == "__main__":
    main()
