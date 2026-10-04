"""PASS A stage 2, member-parallel: gz shards -> validated Parquet + the two redundancy decisions.

    python scratchpad/final_canonical_build/webqsp_v1/v3_pass_a_stage2_par.py [--workers N]

WHY THIS MAY BE PARALLEL AT ALL. Both redundancy decisions (rdfs:label vs type.object.name,
rdf:type vs type.object.type) are per-SUBJECT set comparisons, so they are exact only if a worker
sees every row of every subject it judges. Two measured facts make that true member by member:
REOPENED_SUBJECT_BLOCKS_N = 0 (no subject appears in two places) and blocks_merged_across_seams = 0
(no subject straddles a member boundary). Both are read back from the contiguity record and enforced
at startup rather than assumed, because the failure mode is silent: a label row would be filed as
residue while the matching name row sat in a member owned by another worker.

WHY UNESCAPING IS NOT A CHAIN OF replace_substring CALLS. RDF escapes compose: the source text
backslash-backslash-t is an escaped BACKSLASH followed by a literal 't'. Substituting the escaped
backslash first rewrites it to backslash-t, which the next substitution turns into a TAB, silently
merging two distinct literals into one identity. Unescaping is therefore a single left-to-right
scan, applied only to rows that contain a backslash at all (a vectorised test), so the common case
stays fast and the rare case stays correct.

MEMORY IS THE DESIGN CONSTRAINT. About 4.5 GB is free and the largest per-member table is ~1.3M
rows, so a worker never holds two large structures at once: pass-through tables are read, written
and released one at a time, and each comparison side is a set of joined keys built by streaming the
shard rather than a dict of sets over materialised columns. Joining fields with a TAB is safe for
exactly the reason the shards are readable as TSV at all -- N-Triples escapes a literal tab as two
characters, so no field can contain a raw one.
"""
import glob
import gzip
import json
import multiprocessing as mp
import os
import re
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SHARDS = r"data\final_canonical\freebase_v3\pass_a\shards"
OUTDIR = r"data\final_canonical\freebase_v3\pass_a"
PARTS = os.path.join(OUTDIR, "_parts")
CONTIG = r"data\final_canonical\freebase_v3\V3_SUBJECT_CONTIGUITY.json"
REPORT = r"data\final_canonical\freebase_v3\V3_PASS_A_SCHEMA_METADATA.json"

BS = chr(92)
# BS+BS is a regex-escaped backslash: it MATCHES one literal backslash. A single BS would escape the
# '(' that follows and the pattern would not compile at all.
_ESC = re.compile(
    BS + BS + "(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|[tbnrf" + BS + BS + "'" + chr(34) + "])")
_SIMPLE = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f",
           chr(34): chr(34), "'": "'", BS: BS}

COLS = {
    "name": ["subject", "lexical", "lang"],
    "label": ["subject", "lexical", "lang"],
    "alias": ["subject", "lexical", "lang"],
    "description": ["subject", "lexical", "lang"],
    "type": ["subject", "type"],
    "rdf_type": ["subject", "type", "in_ns"],
    "key": ["subject", "key"],
    "reverse_property": ["subject", "object"],
    "master_property": ["subject", "object"],
    "property_schema": ["subject", "field", "value"],
    "type_hints": ["subject", "field", "value"],
    "label_residue": ["subject", "lexical", "lang"],
    "rdf_type_residue": ["subject", "type"],
}
# Tables copied through unchanged. 'label' and 'rdf_type' are absent on purpose: each is split by a
# decision rather than copied, so neither has a pass-through form.
PLAIN = ("name", "alias", "description", "type", "key", "reverse_property",
         "master_property", "property_schema", "type_hints")
OUT_TABLES = PLAIN + ("rdf_type", "label_residue", "rdf_type_residue")
# Columns holding RDF literal text, i.e. the ones that must be unescaped. URIs and the in_ns flag
# are not literals and stay byte-identical to the source.
LITERAL_COLS = {"name": ["lexical"], "alias": ["lexical"], "description": ["lexical"],
                "key": ["key"], "property_schema": ["value"], "type_hints": ["value"],
                "label_residue": ["lexical"]}


def _sub(m):
    g = m.group(1)
    return chr(int(g[1:], 16)) if g[0] in "uU" else _SIMPLE[g]


def unescape(col):
    if len(col) == 0:
        return col
    needs = pc.match_substring(col, BS)
    if not pc.any(needs).as_py():
        return col
    vals = col.to_pylist()
    flags = needs.to_pylist()
    for i, f in enumerate(flags):
        if f and vals[i] is not None:
            vals[i] = _ESC.sub(_sub, vals[i])
    return pa.array(vals, type=pa.string())


def stream_shard(path, ncol, badbox):
    """Yield field tuples from a gz TSV shard without materialising it.

    Deliberately NOT pyarrow.csv: shards hold raw RDF-escaped text in which a double quote is
    ordinary data and every CSV reader would treat it as a quoting character. Splitting on tab is
    unambiguous because N-Triples escapes a literal tab as two characters and stage 1 left the
    escaped form untouched. Wrong-arity rows are counted, never coerced into shape."""
    if not os.path.exists(path):
        return
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="\n") as fh:
        for line in fh:
            if line.endswith("\n"):
                line = line[:-1]
            if not line:
                continue
            f = line.split("\t")
            if len(f) != ncol:
                badbox[0] += 1
                continue
            yield f


def read_shard(path, names, badbox):
    n = len(names)
    cols = [[] for _ in names]
    for f in stream_shard(path, n, badbox):
        for i in range(n):
            cols[i].append(f[i])
    return cols


class PartWriter:
    __slots__ = ("w", "path", "n")

    def __init__(self, path):
        self.w = None
        self.path = path
        self.n = 0

    def write(self, table, cols, names):
        if not cols[0]:
            return
        arrs = []
        lit = LITERAL_COLS.get(table, ())
        for i, nm in enumerate(names):
            a = pa.array(cols[i], type=pa.string())
            cols[i] = None                      # release the Python list as we convert
            if nm in lit:
                a = unescape(a)
            arrs.append(a)
        tb = pa.Table.from_arrays(arrs, names=names)
        del arrs
        if self.w is None:
            self.w = pq.ParquetWriter(self.path, tb.schema, compression="zstd")
        self.w.write_table(tb)
        self.n += tb.num_rows

    def close(self):
        if self.w is not None:
            self.w.close()


def work(job):
    wid, members = job
    os.makedirs(PARTS, exist_ok=True)
    W = {t: PartWriter(os.path.join(PARTS, f"{t}.w{wid:02d}.parquet")) for t in OUT_TABLES}
    st = {"label_rows": 0, "label_redundant": 0, "label_residue": 0,
          "rdf_type_rows": 0, "rdf_type_in_ns": 0, "rdf_type_ns_dropped_as_redundant": 0,
          "rdf_type_residue": 0, "rdf_type_out_of_ns_kept": 0,
          "subjects_with_label_residue": 0, "subjects_with_rdf_type_residue": 0}
    bad = {}
    t0 = time.time()

    def path(k, t):
        return os.path.join(SHARDS, f"m{k:04d}_{t}.tsv.gz")

    for k in members:
        # ---- pass-through tables, one at a time so peak memory is one table, not the member ----
        for tbl in PLAIN:
            bb = [0]
            cols = read_shard(path(k, tbl), COLS[tbl], bb)
            bad[tbl] = bad.get(tbl, 0) + bb[0]
            W[tbl].write(tbl, cols, COLS[tbl])
            del cols

        # ---- rdfs:label vs type.object.name ----
        # The name side is re-streamed from disk rather than kept from the write above: that shard
        # is ~5 MB compressed, and holding its columns alive across the write would double peak use.
        bb = [0]
        name_keys = set()
        for s, lex, lang in stream_shard(path(k, "name"), 3, bb):
            name_keys.add(s + "\t" + lex + "\t" + lang)
        bb = [0]
        res = [[], [], []]
        seen = set()
        for s, lex, lang in stream_shard(path(k, "label"), 3, bb):
            st["label_rows"] += 1
            if s + "\t" + lex + "\t" + lang in name_keys:
                st["label_redundant"] += 1
            else:
                res[0].append(s)
                res[1].append(lex)
                res[2].append(lang)
                st["label_residue"] += 1
                seen.add(s)
        bad["label"] = bad.get("label", 0) + bb[0]
        st["subjects_with_label_residue"] += len(seen)
        del name_keys, seen
        W["label_residue"].write("label_residue", res, COLS["label_residue"])
        del res

        # ---- rdf:type vs type.object.type ----
        bb = [0]
        type_keys = set()
        for s, ty in stream_shard(path(k, "type"), 2, bb):
            type_keys.add(s + "\t" + ty)
        bb = [0]
        keep = [[], [], []]
        resid = [[], []]
        seen = set()
        for s, ty, in_ns in stream_shard(path(k, "rdf_type"), 3, bb):
            st["rdf_type_rows"] += 1
            if in_ns == "0":
                # Out-of-namespace rdf:type is the RDF/OWL vocabulary layer, not a Freebase type
                # assertion. type.object.type never carries it, so it can never be redundant.
                st["rdf_type_out_of_ns_kept"] += 1
                keep[0].append(s)
                keep[1].append(ty)
                keep[2].append(in_ns)
                continue
            st["rdf_type_in_ns"] += 1
            if s + "\t" + ty in type_keys:
                st["rdf_type_ns_dropped_as_redundant"] += 1
            else:
                resid[0].append(s)
                resid[1].append(ty)
                st["rdf_type_residue"] += 1
                seen.add(s)
        bad["rdf_type"] = bad.get("rdf_type", 0) + bb[0]
        st["subjects_with_rdf_type_residue"] += len(seen)
        del type_keys, seen
        W["rdf_type"].write("rdf_type", keep, COLS["rdf_type"])
        W["rdf_type_residue"].write("rdf_type_residue", resid, COLS["rdf_type_residue"])
        del keep, resid

    counts = {t: W[t].n for t in OUT_TABLES}
    for w in W.values():
        w.close()
    return {"worker": wid, "members": len(members), "counts": counts, "stats": st,
            "bad": bad, "elapsed_s": round(time.time() - t0, 1)}


def merge(table, nworkers):
    """Concatenate the per-worker parts into one Parquet file by streaming row groups.

    Never materialises a table, so a multi-GB output costs bounded memory. Parts are deleted only
    once the merged file's row count matches the sum of the parts' -- the row-count precondition the
    disk-discipline rule sets before any merge input may be removed."""
    files = [os.path.join(PARTS, f"{table}.w{w:02d}.parquet") for w in range(nworkers)]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        # A table that legitimately has zero rows still gets a file, with its declared schema. An
        # absent artifact is indistinguishable from a failed one to everything downstream, and the
        # validator rightly refuses to call a missing file a passing empty table.
        out = os.path.join(OUTDIR, f"{table}.parquet")
        if not os.path.exists(out):
            pq.write_table(
                pa.table({n: pa.array([], type=pa.string()) for n in COLS[table]}),
                out, compression="zstd")
        return 0, True
    expected = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    out = os.path.join(OUTDIR, f"{table}.parquet")
    w = None
    n = 0
    for f in files:
        pf = pq.ParquetFile(f)
        if w is None:
            w = pq.ParquetWriter(out, pf.schema_arrow, compression="zstd")
        for i in range(pf.num_row_groups):
            tb = pf.read_row_group(i)
            w.write_table(tb)
            n += tb.num_rows
            del tb
        del pf
    if w is not None:
        w.close()
    ok = (n == expected) and (pq.ParquetFile(out).metadata.num_rows == n)
    if ok:
        for f in files:
            os.remove(f)
    return n, ok


def main():
    workers = 7
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    contig = json.load(open(CONTIG, encoding="utf-8"))
    if not contig.get("SOURCE_SUBJECT_CONTIGUOUS"):
        raise SystemExit(f"REOPENED_SUBJECT_BLOCKS_N = {contig.get('REOPENED_SUBJECT_BLOCKS_N')}: "
                         "a per-member comparison would judge subjects on partial row sets.")
    if contig["SEAMS"]["blocks_merged_across_seams"] != 0:
        raise SystemExit("subjects straddle member seams; per-member comparison is not exact here.")

    metas = {}
    for p in sorted(glob.glob(os.path.join(SHARDS, "m*_meta.json"))):
        m = json.load(open(p, encoding="utf-8"))
        metas[m["member"]] = m
    ks = sorted(metas)
    print(f"{len(ks)} members, {workers} workers; contiguity ESTABLISHED, 0 seam subjects",
          flush=True)

    chunks = [[] for _ in range(workers)]
    for i, k in enumerate(ks):
        chunks[i % workers].append(k)
    jobs = [(w, c) for w, c in enumerate(chunks) if c]

    t0 = time.time()
    results = []
    with mp.Pool(len(jobs)) as pool:
        for r in pool.imap_unordered(work, jobs):
            results.append(r)
            print(f"  worker {r['worker']:2d} done  {r['members']} members  "
                  f"{sum(r['counts'].values())/1e6:8.1f}M rows  {r['elapsed_s']/60:5.1f}m  "
                  f"({len(results)}/{len(jobs)})", flush=True)

    print(f"parallel phase {(time.time()-t0)/60:.1f}m; merging parts", flush=True)
    counts, merge_ok = {}, {}
    for t in OUT_TABLES:
        counts[t], merge_ok[t] = merge(t, workers)
        print(f"  merged {t:20s} {counts[t]:>13,} rows  ok={merge_ok[t]}", flush=True)
    try:
        os.rmdir(PARTS)
    except OSError:
        pass

    stats, bad = {}, {}
    for r in results:
        for k2, v in r["stats"].items():
            stats[k2] = stats.get(k2, 0) + v
        for k2, v in r["bad"].items():
            bad[k2] = bad.get(k2, 0) + v

    langs, dtypes = {}, {}
    for k in ks:
        for d, src in ((langs, "langs"), (dtypes, "dtypes")):
            for kk, vv in metas[k][src].items():
                d[kk] = d.get(kk, 0) + vv

    stage1_rows = sum(metas[k]["rows"] for k in ks)
    # Row reconciliation. Stage 1 counted one row per emitted shard line; every one must be
    # accounted for here. 'label' is split into redundant + residue and 'rdf_type' into kept +
    # dropped + residue, so this identity is what proves nothing was lost between the stages --
    # the precondition the disk-discipline rule sets before the shards may be deleted.
    accounted = sum(counts[t] for t in PLAIN) + stats["label_rows"] + stats["rdf_type_rows"]
    doc = {
        "schema": "V3_PASS_A_SCHEMA_METADATA/v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": "V3_CANONICAL_CONTRACT_V3.json PASS_A_STATUS",
        "members": len(ks),
        "workers": len(jobs),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "TABLES": counts,
        "MERGE_ROWCOUNT_VERIFIED": merge_ok,
        "malformed_shard_rows": bad,
        "ROW_RECONCILIATION": {
            "stage1_rows_emitted": stage1_rows,
            "stage2_rows_accounted": accounted,
            "difference": stage1_rows - accounted,
            "balanced": stage1_rows == accounted,
            "identity": "sum(pass-through tables) + label_rows + rdf_type_rows == stage 1 rows, "
                        "where label_rows = redundant + residue and rdf_type_rows = "
                        "out-of-ns kept + dropped-as-redundant + residue",
            "rdf_type_kept_equals_out_of_ns":
                counts["rdf_type"] == stats["rdf_type_out_of_ns_kept"],
            "label_split_closes": stats["label_rows"] == stats["label_redundant"] + stats["label_residue"],
            "rdf_type_split_closes": stats["rdf_type_rows"] == (
                stats["rdf_type_out_of_ns_kept"] + stats["rdf_type_ns_dropped_as_redundant"]
                + stats["rdf_type_residue"]),
        },
        "REDUNDANCY": {
            **stats,
            "label_redundant_fraction": round(stats["label_redundant"] / stats["label_rows"], 6)
            if stats["label_rows"] else None,
            "rdf_type_in_ns_redundant_fraction":
                round(stats["rdf_type_ns_dropped_as_redundant"] / stats["rdf_type_in_ns"], 6)
                if stats["rdf_type_in_ns"] else None,
            "reading": "a label row is dropped only where it equals a type.object.name row of the "
                       "SAME subject on all three of (subject, lexical form, language tag); every "
                       "label row that is not equal survives in label_residue, so the discard is "
                       "provable rather than assumed. rdf:type inside the Freebase namespace drops "
                       "only where type.object.type already carries that exact type for that "
                       "subject; anything else lands in rdf_type_residue. rdf:type outside the "
                       "namespace is the RDF/OWL vocabulary layer and is always kept.",
            "contract_note": "contract v2 forbids collapsing rdf:type into type.object.type "
                             "wholesale; the two were measured non-equivalent. This records the "
                             "difference instead of assuming it away.",
        },
        "LANGUAGES_N": len(langs),
        "LANGUAGES_TOP": dict(sorted(langs.items(), key=lambda x: -x[1])[:40]),
        "DATATYPES_N": len(dtypes),
        "DATATYPES_TOP": dict(sorted(dtypes.items(), key=lambda x: -x[1])[:40]),
    }
    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps({"TABLES": counts, "MERGE_ROWCOUNT_VERIFIED": merge_ok,
                      "ROW_RECONCILIATION": doc["ROW_RECONCILIATION"],
                      "REDUNDANCY": stats, "malformed_shard_rows": bad,
                      "LANGUAGES_N": len(langs), "DATATYPES_N": len(dtypes)}, indent=1))


if __name__ == "__main__":
    main()
