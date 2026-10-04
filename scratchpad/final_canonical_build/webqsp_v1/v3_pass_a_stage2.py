"""PASS A stage 2: per-member gz shards -> validated Parquet, with the redundancy decisions.

    python scratchpad/final_canonical_build/webqsp_v1/v3_pass_a_stage2.py [--members K]

WHY UNESCAPING IS NOT A CHAIN OF replace_substring CALLS. It is tempting to run the handful of RDF
escape substitutions vectorised and be done. That is WRONG, and wrong in exactly the
identity-collapsing direction the literal-identity rule exists to prevent. RDF escapes compose: the
source text  backslash backslash t  is an escaped BACKSLASH followed by a literal 't'. Replacing
the two-character escaped-backslash first rewrites it into backslash-t, which the next substitution
then turns into a TAB. Two distinct source literals silently become one, and nothing downstream can
tell. Unescaping must be a single left-to-right scan.

So: rows containing no backslash at all -- the overwhelming majority -- are passed through untouched
by one vectorised test, and only the remainder pay for a correct scalar scan. Fast because of the
filter, correct because the slow path is a real single-pass decoder.

WHY THE COMPARISONS RUN PER MEMBER. rdfs:label vs type.object.name and rdf:type vs
type.object.type are per-subject set comparisons. Given SOURCE_SUBJECT_CONTIGUOUS, every subject's
rows live in one member (or, for at most 199 subjects, in two members that meet at a seam), so each
member can be compared exactly in memory -- tens of MB -- instead of joining ~10^8-row tables
globally. Seam subjects are handled explicitly rather than hoped about. The precondition is READ,
not assumed: this refuses to run if contiguity has not been established.

NOTHING IS DROPPED SILENTLY. A label row that does not match a name row lands in label_residue; an
in-namespace rdf:type row not covered by type.object.type lands in rdf_type_residue. Out-of-namespace
rdf:type is the RDF/OWL vocabulary layer and is always kept in rdf_type.
"""
import glob
import gzip
import json
import os
import re
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SHARDS = r"data\final_canonical\freebase_v3\pass_a\shards"
OUTDIR = r"data\final_canonical\freebase_v3\pass_a"
CONTIG = r"data\final_canonical\freebase_v3\V3_SUBJECT_CONTIGUITY.json"
REPORT = r"data\final_canonical\freebase_v3\V3_PASS_A_SCHEMA_METADATA.json"

BS = chr(92)
# one alternation, one scan. \uXXXX and \UXXXXXXXX before the single-character escapes so a
# longest-match is taken; the leading backslash is consumed by the pattern itself, so an escaped
# backslash cannot be re-read as the start of another escape.
# BS+BS is a regex-escaped backslash, i.e. it MATCHES one literal backslash. A single BS here would
# escape the '(' that follows and the pattern would not compile at all.
_ESC = re.compile(
    BS + BS + "(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|[tbnrf" + BS + BS + "'" + chr(34) + "])")
_SIMPLE = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f",
           chr(34): chr(34), "'": "'", BS: BS}

SCHEMAS = {
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
}
LITERAL_COLS = {"name": ["lexical"], "label": ["lexical"], "alias": ["lexical"],
                "description": ["lexical"], "key": ["key"],
                "property_schema": ["value"], "type_hints": ["value"]}


def _sub(m):
    g = m.group(1)
    return chr(int(g[1:], 16)) if g[0] in "uU" else _SIMPLE[g]


def unescape(col):
    """pyarrow StringArray -> StringArray. Single-pass decode, applied only where needed."""
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


def read_shard(path, names):
    """Parse a gz TSV shard into columns of str.

    Deliberately NOT pyarrow.csv: the shards hold raw RDF-escaped text where a double quote is
    ordinary data, and every CSV reader wants to treat it as a quoting character. Splitting on tab
    is unambiguous here because N-Triples escapes a literal tab as two characters, which stage 1
    left untouched (verified: a literal containing an escaped tab keeps exactly 3 field separators).
    """
    ncol = len(names)
    cols = [[] for _ in names]
    bad = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="\n") as fh:
        for line in fh:
            if line.endswith("\n"):
                line = line[:-1]
            if not line:
                continue
            f = line.split("\t")
            if len(f) != ncol:
                bad += 1
                continue
            for i in range(ncol):
                cols[i].append(f[i])
    return cols, bad


def main():
    limit = None
    if "--members" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--members") + 1])

    if not os.path.exists(CONTIG):
        raise SystemExit("V3_SUBJECT_CONTIGUITY.json missing: run v3_contiguity_stitch.py first. "
                         "The per-member comparison is only exact if subjects do not reopen.")
    contig = json.load(open(CONTIG, encoding="utf-8"))
    if not contig.get("SOURCE_SUBJECT_CONTIGUOUS"):
        raise SystemExit(
            f"REOPENED_SUBJECT_BLOCKS_N = {contig.get('REOPENED_SUBJECT_BLOCKS_N')}: subjects "
            f"reopen, so a per-member comparison would compare an incomplete row set for those "
            f"subjects. Stage 2 needs the global-join variant; refusing to emit a wrong answer.")

    metas = {}
    for p in sorted(glob.glob(os.path.join(SHARDS, "m*_meta.json"))):
        m = json.load(open(p, encoding="utf-8"))
        metas[m["member"]] = m
    ks = sorted(metas)
    if limit:
        ks = ks[:limit]

    # Subjects that straddle a member seam have rows in TWO members, so neither member alone holds
    # the full row set and comparing them in either member could mark a label row as residue while
    # its matching name row sits in the other. They are excluded from per-member comparison
    # entirely, accumulated across the whole run, and compared once at the end. At most one subject
    # per seam, so this is a set of <=199.
    seam_subjects = set()
    for a, b in zip(ks, ks[1:]):
        la, fb = metas[a]["last_subject"], metas[b]["first_subject"]
        if la is not None and la == fb:
            seam_subjects.add(la)

    os.makedirs(OUTDIR, exist_ok=True)
    writers = {}
    counts = {}
    bad_rows = {}
    t0 = time.time()
    stats = {"label_rows": 0, "label_redundant": 0, "label_residue": 0,
             "rdf_type_rows": 0, "rdf_type_in_ns": 0, "rdf_type_ns_dropped_as_redundant": 0,
             "rdf_type_residue": 0, "rdf_type_out_of_ns_kept": 0,
             "subjects_with_differing_label_sets": 0,
             "subjects_with_differing_rdf_type_sets": 0}

    def emit(table, cols, names):
        if not cols[0]:
            return
        arrs = []
        for i, nm in enumerate(names):
            a = pa.array(cols[i], type=pa.string())
            if nm in LITERAL_COLS.get(table, []):
                a = unescape(a)
            arrs.append(a)
        tb = pa.Table.from_arrays(arrs, names=names)
        if table not in writers:
            path = os.path.join(OUTDIR, f"{table}.parquet")
            writers[table] = pq.ParquetWriter(path, tb.schema, compression="zstd")
        writers[table].write_table(tb)
        counts[table] = counts.get(table, 0) + tb.num_rows

    # accumulators for the deferred seam subjects (bounded by the number of seams)
    seam_name = {}       # subject -> set of (lexical, lang)
    seam_type = {}       # subject -> set of type
    seam_label = []      # (subject, lexical, lang)
    seam_rdft = []       # (subject, type, in_ns)

    for k in ks:
        # ---- plain tables ----
        for tbl in ("alias", "description", "key", "reverse_property", "master_property",
                    "property_schema", "type_hints", "name", "type"):
            p = os.path.join(SHARDS, f"m{k:04d}_{tbl}.tsv.gz")
            if not os.path.exists(p):
                continue
            cols, bad = read_shard(p, SCHEMAS[tbl])
            bad_rows[tbl] = bad_rows.get(tbl, 0) + bad
            emit(tbl, cols, SCHEMAS[tbl])

        # ---- name/label comparison ----
        ncols, _ = read_shard(os.path.join(SHARDS, f"m{k:04d}_name.tsv.gz"), SCHEMAS["name"])
        name_set = {}
        for s, lex, lang in zip(*ncols):
            if s in seam_subjects:
                seam_name.setdefault(s, set()).add((lex, lang))
            else:
                name_set.setdefault(s, set()).add((lex, lang))

        lp = os.path.join(SHARDS, f"m{k:04d}_label.tsv.gz")
        lcols, bad = read_shard(lp, SCHEMAS["label"])
        bad_rows["label"] = bad_rows.get("label", 0) + bad
        res = [[], [], []]
        diff_subj = set()
        for s, lex, lang in zip(*lcols):
            if s in seam_subjects:
                seam_label.append((s, lex, lang))
                continue
            stats["label_rows"] += 1
            if (lex, lang) in name_set.get(s, ()):
                stats["label_redundant"] += 1
            else:
                res[0].append(s); res[1].append(lex); res[2].append(lang)
                stats["label_residue"] += 1
                diff_subj.add(s)
        stats["subjects_with_differing_label_sets"] += len(diff_subj)
        emit("label_residue", res, SCHEMAS["label"])

        # ---- rdf_type / type comparison ----
        tcols, _ = read_shard(os.path.join(SHARDS, f"m{k:04d}_type.tsv.gz"), SCHEMAS["type"])
        type_set = {}
        for s, ty in zip(*tcols):
            if s in seam_subjects:
                seam_type.setdefault(s, set()).add(ty)
            else:
                type_set.setdefault(s, set()).add(ty)

        rcols, bad = read_shard(os.path.join(SHARDS, f"m{k:04d}_rdf_type.tsv.gz"),
                                SCHEMAS["rdf_type"])
        bad_rows["rdf_type"] = bad_rows.get("rdf_type", 0) + bad
        keep = [[], [], []]
        resid = [[], []]
        diff_t = set()
        for s, ty, in_ns in zip(*rcols):
            if s in seam_subjects:
                seam_rdft.append((s, ty, in_ns))
                continue
            stats["rdf_type_rows"] += 1
            if in_ns == "0":
                stats["rdf_type_out_of_ns_kept"] += 1
                keep[0].append(s); keep[1].append(ty); keep[2].append(in_ns)
                continue
            stats["rdf_type_in_ns"] += 1
            if ty in type_set.get(s, ()):
                stats["rdf_type_ns_dropped_as_redundant"] += 1
            else:
                resid[0].append(s); resid[1].append(ty)
                stats["rdf_type_residue"] += 1
                diff_t.add(s)
        stats["subjects_with_differing_rdf_type_sets"] += len(diff_t)
        emit("rdf_type", keep, SCHEMAS["rdf_type"])
        emit("rdf_type_residue", resid, ["subject", "type"])

        if (k + 1) % 10 == 0 or k == ks[-1]:
            el = time.time() - t0
            print(f"  member {k:4d}/{ks[-1]}  t={el/60:5.1f}m  "
                  f"eta={el/(ks.index(k)+1)*(len(ks)-ks.index(k)-1)/60:5.1f}m  "
                  f"rows={sum(counts.values())/1e6:8.1f}M", flush=True)

    # ---- deferred seam subjects, now with their COMPLETE row sets from both members ----
    res = [[], [], []]
    diff_subj = set()
    for s, lex, lang in seam_label:
        stats["label_rows"] += 1
        if (lex, lang) in seam_name.get(s, ()):
            stats["label_redundant"] += 1
        else:
            res[0].append(s); res[1].append(lex); res[2].append(lang)
            stats["label_residue"] += 1
            diff_subj.add(s)
    stats["subjects_with_differing_label_sets"] += len(diff_subj)
    emit("label_residue", res, SCHEMAS["label"])

    keep = [[], [], []]
    resid = [[], []]
    diff_t = set()
    for s, ty, in_ns in seam_rdft:
        stats["rdf_type_rows"] += 1
        if in_ns == "0":
            stats["rdf_type_out_of_ns_kept"] += 1
            keep[0].append(s); keep[1].append(ty); keep[2].append(in_ns)
            continue
        stats["rdf_type_in_ns"] += 1
        if ty in seam_type.get(s, ()):
            stats["rdf_type_ns_dropped_as_redundant"] += 1
        else:
            resid[0].append(s); resid[1].append(ty)
            stats["rdf_type_residue"] += 1
            diff_t.add(s)
    stats["subjects_with_differing_rdf_type_sets"] += len(diff_t)
    emit("rdf_type", keep, SCHEMAS["rdf_type"])
    emit("rdf_type_residue", resid, ["subject", "type"])
    stats["seam_subjects_deferred"] = len(seam_subjects)
    stats["seam_label_rows"] = len(seam_label)
    stats["seam_rdf_type_rows"] = len(seam_rdft)

    for w in writers.values():
        w.close()

    langs, dtypes = {}, {}
    for k in ks:
        for d, src in ((langs, "langs"), (dtypes, "dtypes")):
            for kk, vv in metas[k][src].items():
                d[kk] = d.get(kk, 0) + vv

    doc = {
        "schema": "V3_PASS_A_SCHEMA_METADATA/v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": "V3_CANONICAL_CONTRACT_V2.json PIPELINE_V2 PASS_A",
        "members": len(ks),
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "TABLES": counts,
        "malformed_shard_rows": bad_rows,
        "REDUNDANCY": {
            **stats,
            "reading": "label rows equal to a type.object.name row of the same subject are dropped "
                       "as redundant; every label row that is NOT equal is kept in label_residue, "
                       "so the discard is provable rather than assumed. rdf:type inside the "
                       "Freebase namespace is dropped only when type.object.type already carries "
                       "it for that subject; anything else lands in rdf_type_residue. rdf:type "
                       "outside the namespace is the RDF/OWL vocabulary layer and is always kept.",
            "contract_note": "contract v2 forbids collapsing rdf:type into type.object.type "
                             "wholesale; it was measured non-equivalent. This records the "
                             "difference rather than assuming it away.",
        },
        "LANGUAGES": dict(sorted(langs.items(), key=lambda x: -x[1])[:60]),
        "LANGUAGES_N": len(langs),
        "DATATYPES": dict(sorted(dtypes.items(), key=lambda x: -x[1])[:60]),
        "DATATYPES_N": len(dtypes),
    }
    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps({"TABLES": counts, "REDUNDANCY": stats,
                      "LANGUAGES_N": len(langs), "DATATYPES_N": len(dtypes)}, indent=1))


if __name__ == "__main__":
    main()
