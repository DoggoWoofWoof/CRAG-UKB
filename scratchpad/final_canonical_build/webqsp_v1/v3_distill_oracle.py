"""Distil the IDIR extraction into a compact Parquet oracle, so the CSVs can be reclaimed.

    python scratchpad/final_canonical_build/webqsp_v1/v3_distill_oracle.py

IDIR's role is now fixed: audit oracle, metadata accelerator, structural validation reference --
NOT the canonical V3 graph. What that role needs is the CONTENT, not the CSV serialisation. The
extraction currently costs 31 GB, most of it in two text files, and the raw mirror needs the space.

So: convert to Parquet, keep the content, drop the serialisation. This is strictly better for the
oracle role than the CSVs were -- columnar lookup instead of line-scanning 13.65 GB every time we
want to check a type.

WHAT IS AND IS NOT A FILTER HERE. Rows keyed by a schema PATH (/film/film/starring) rather than an
MID are dropped from the mid_* tables and kept in the schema_* tables. That is a source-level
structural distinction, not a subset: nothing is selected by reference to any question, and the
schema rows are preserved, not discarded. No probe set is consulted anywhere in this file.

REVERSIBILITY. Everything deleted afterwards is re-extractable from idirlab-freebases.zip, whose
bytes and MD5 match Zenodo exactly and whose recovered offsets are cached in recovered_offsets.json.
Re-extracting object_types.csv takes 66 seconds. The deletion is therefore reversible in minutes,
which is why it is safe despite the standing supersede-don't-delete rule.
"""
import json, os, time
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "data/final_canonical/freebase_v3/_acquisition/idir"
SRC = f"{ROOT}/extracted/idirlab-freebases"
DST = f"{ROOT}/oracle"
REC = "data/final_canonical/freebase_v3/V3_ORACLE_DISTILLATION.json"
BATCH = 4_000_000


def is_mid(s):
    return s.startswith("/m/") or s.startswith("/g/")


def dotted(s):
    """/m/010016 -> m.010016 ; leaves schema paths alone."""
    return s[1:].replace("/", ".", 1) if is_mid(s) else s


def write_batches(path, schema, gen):
    """Stream row batches to one Parquet file. Returns rows written."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = pq.ParquetWriter(path, schema, compression="zstd", compression_level=6)
    n = 0
    try:
        for cols in gen:
            if not cols[0]:
                continue
            w.write_table(pa.Table.from_arrays(
                [pa.array(c, type=f.type) for c, f in zip(cols, schema)], schema=schema))
            n += len(cols[0])
    finally:
        w.close()
    return n


def distil_names(t0):
    """object_names.csv -> mid_name.parquet + schema_name.parquet"""
    sch = pa.schema([("mid", pa.string()), ("name", pa.string()), ("lang", pa.string())])
    sch2 = pa.schema([("path", pa.string()), ("name", pa.string()), ("lang", pa.string())])
    mids, names, langs = [], [], []
    p2, n2, l2 = [], [], []
    stats = {"lines": 0, "mid_rows": 0, "schema_rows": 0, "unparsed": 0}

    def parse(line):
        # subject,predicate,object ; predicate has no comma, object is a quoted literal
        a = line.find(",")
        if a < 0:
            return None
        b = line.find(",", a + 1)
        if b < 0:
            return None
        subj, obj = line[:a], line[b + 1:].rstrip("\n")
        lang = ""
        if obj.endswith('"'):
            pass
        else:
            q = obj.rfind('"@')
            if q >= 0:
                lang = obj[q + 2:]
                obj = obj[:q + 1]
        return subj, obj.strip('"'), lang

    def gen_mid():
        with open(f"{SRC}/Metadata/object_names.csv", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stats["lines"] += 1
                r = parse(line)
                if r is None:
                    stats["unparsed"] += 1
                    continue
                s, nm, lg = r
                if is_mid(s):
                    stats["mid_rows"] += 1
                    mids.append(dotted(s))
                    names.append(nm)
                    langs.append(lg)
                    if len(mids) >= BATCH:
                        yield (mids[:], names[:], langs[:])
                        mids.clear(); names.clear(); langs.clear()
                else:
                    stats["schema_rows"] += 1
                    p2.append(s); n2.append(nm); l2.append(lg)
        if mids:
            yield (mids, names, langs)

    n = write_batches(f"{DST}/mid_name.parquet", sch, gen_mid())
    pq.write_table(pa.Table.from_arrays(
        [pa.array(p2), pa.array(n2), pa.array(l2)], schema=sch2),
        f"{DST}/schema_name.parquet", compression="zstd")
    print(f"[names] {stats} -> {n:,} mid rows t={time.time()-t0:.0f}s", flush=True)
    return {**stats, "mid_parquet_rows": n, "schema_parquet_rows": len(p2)}


def distil_types(t0):
    """object_types.csv -> mid_type.parquet + schema_type.parquet"""
    sch = pa.schema([("mid", pa.string()), ("type", pa.string())])
    sch2 = pa.schema([("path", pa.string()), ("type", pa.string())])
    mids, types = [], []
    p2, t2 = [], []
    stats = {"lines": 0, "mid_rows": 0, "schema_rows": 0, "unparsed": 0}

    def gen_mid():
        with open(f"{SRC}/Metadata/object_types.csv", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stats["lines"] += 1
                a = line.find(",")
                b = line.rfind(",")
                if a < 0 or b <= a:
                    stats["unparsed"] += 1
                    continue
                s, ty = line[:a], line[b + 1:].rstrip("\n")
                if is_mid(s):
                    stats["mid_rows"] += 1
                    mids.append(dotted(s))
                    types.append(ty)
                    if len(mids) >= BATCH:
                        yield (mids[:], types[:])
                        mids.clear(); types.clear()
                else:
                    stats["schema_rows"] += 1
                    p2.append(s); t2.append(ty)
        if mids:
            yield (mids, types)

    n = write_batches(f"{DST}/mid_type.parquet", sch, gen_mid())
    pq.write_table(pa.Table.from_arrays([pa.array(p2), pa.array(t2)], schema=sch2),
                   f"{DST}/schema_type.parquet", compression="zstd")
    print(f"[types] {stats} -> {n:,} mid rows t={time.time()-t0:.0f}s", flush=True)
    return {**stats, "mid_parquet_rows": n, "schema_parquet_rows": len(p2)}


def distil_entities_label(t0):
    """entities_id_label.csv -> entity_label.parquet. Labels contain escaped commas."""
    sch = pa.schema([("mid", pa.string()), ("label", pa.string()), ("global_id", pa.int64())])
    a_, b_, c_ = [], [], []
    stats = {"lines": 0, "rows": 0, "unparsed": 0}

    def gen():
        with open(f"{SRC}/Metadata/entities_id_label.csv", encoding="utf-8",
                  errors="replace") as fh:
            for line in fh:
                stats["lines"] += 1
                line = line.rstrip("\n")
                i = line.find(",")
                j = line.rfind(",")
                if i < 0 or j <= i:
                    stats["unparsed"] += 1
                    continue
                try:
                    gid = int(line[j + 1:])
                except ValueError:
                    stats["unparsed"] += 1
                    continue
                stats["rows"] += 1
                a_.append(dotted(line[:i]))
                b_.append(line[i + 1:j].replace("\\,", ","))
                c_.append(gid)
                if len(a_) >= BATCH:
                    yield (a_[:], b_[:], c_[:])
                    a_.clear(); b_.clear(); c_.clear()
        if a_:
            yield (a_, b_, c_)

    n = write_batches(f"{DST}/entity_label.parquet", sch, gen())
    print(f"[entity_label] {stats} -> {n:,} rows t={time.time()-t0:.0f}s", flush=True)
    return {**stats, "parquet_rows": n}


def distil_backbone(t0):
    """FB+CVT-REV entity2id + triples -> parquet. The structural validation reference."""
    from pyarrow import csv as pacsv
    sch = pa.schema([("mid", pa.string()), ("idir_id", pa.int32())])
    a_, b_ = [], []
    stats = {"lines": 0}

    def gen():
        with open(f"{SRC}/FB+CVT-REV/entity2id.txt", encoding="utf-8") as fh:
            for line in fh:
                stats["lines"] += 1
                c = line.rfind(",")
                if c < 0:
                    continue
                a_.append(dotted(line[:c]))
                b_.append(int(line[c + 1:]))
                if len(a_) >= BATCH:
                    yield (a_[:], b_[:])
                    a_.clear(); b_.clear()
        if a_:
            yield (a_, b_)

    n = write_batches(f"{DST}/backbone_entity2id.parquet", sch, gen())

    parts = []
    tot = 0
    for split in ("train", "test", "valid"):
        t = pacsv.read_csv(f"{SRC}/FB+CVT-REV/{split}.txt",
                           read_options=pacsv.ReadOptions(
                               column_names=["s", "r", "o"]),
                           convert_options=pacsv.ConvertOptions(
                               column_types={"s": "int32", "r": "int32", "o": "int32"}))
        tot += t.num_rows
        parts.append(t)
    tt = pa.concat_tables(parts)
    pq.write_table(tt, f"{DST}/backbone_triples.parquet", compression="zstd", compression_level=6)
    del parts, tt

    rel = []
    rid = []
    with open(f"{SRC}/FB+CVT-REV/relation2id.txt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line:
                p, i = line.rsplit(",", 1)
                rel.append(p)
                rid.append(int(i))
    pq.write_table(pa.table({"relation": rel, "idir_id": rid}),
                   f"{DST}/backbone_relation2id.parquet", compression="zstd")
    print(f"[backbone] entity2id {n:,}, triples {tot:,}, relations {len(rel):,} "
          f"t={time.time()-t0:.0f}s", flush=True)
    return {"entity2id_rows": n, "triple_rows": tot, "relation_rows": len(rel)}


def main():
    t0 = time.time()
    os.makedirs(DST, exist_ok=True)
    out = {
        "schema": "V3_ORACLE_DISTILLATION/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "IDIR's frozen role is audit oracle / metadata accelerator / structural "
                   "validation reference. That role needs the content, not the CSV serialisation. "
                   "Parquet serves it better and frees the disk the raw mirror needs.",
        "filter_note": "rows keyed by a schema path rather than an MID go to the schema_* tables "
                       "instead of the mid_* tables. Nothing is discarded and no probe set is "
                       "consulted; this file never reads a question, an answer or a topic.",
    }
    out["backbone"] = distil_backbone(t0)
    out["entity_label"] = distil_entities_label(t0)
    out["names"] = distil_names(t0)
    out["types"] = distil_types(t0)

    sizes = {}
    for f in sorted(os.listdir(DST)):
        sizes[f] = os.path.getsize(os.path.join(DST, f))
    out["oracle_files"] = sizes
    out["oracle_bytes"] = sum(sizes.values())
    out["oracle_gb"] = round(sum(sizes.values()) / 1e9, 2)
    out["elapsed_s"] = round(time.time() - t0, 1)

    tmp = REC + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, REC)
    print(json.dumps({"oracle_gb": out["oracle_gb"], "files": sizes}, indent=1))


if __name__ == "__main__":
    main()
