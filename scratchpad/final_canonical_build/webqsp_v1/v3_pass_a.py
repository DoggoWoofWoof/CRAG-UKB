"""PASS A -- schema + metadata extraction from the raw Freebase mirror.

    python scratchpad/final_canonical_build/webqsp_v1/v3_pass_a.py [--limit-gb N] [--tag NAME]

First of the passes that touch the source. Emits the reverse-property map, the mediator
declarations, the property schema, and the name/alias/description/type/key metadata. PASS B consumes
these; it cannot classify mediators or canonicalise orientation without them.

WHAT THIS PASS UNLOCKS THAT IDIR COULD NOT PROVIDE.
  * type.property.reverse_property -- absent from the IDIR archive (FINDING_1), which is why
    INVARIANT_2 could only be bounded (25 predicates / 17,930 triples) instead of computed.
  * freebase.type_hints.mediator -- the authoritative CVT declaration. Check 6 measured that IDIR's
    types do NOT separate CVTs from entities (68.081% of CVT type mass sits on types entities also
    carry, against a 5% pre-registered threshold). That was a limit of typing as evidence, not a
    fact about Freebase: the dump declares mediator status directly.

REDUNDANCY IS MEASURED, NOT ASSUMED (contract v2, PASS_A clause).
  * rdfs:label is dropped only where it is identical to type.object.name FOR THAT SUBJECT. Any
    differing label is written to label_residue.parquet. A prefix sample showed 0 differing out of
    507,837 subjects; this pass re-establishes that over the whole source instead of inheriting it.
  * rdf:type is NOT collapsed into type.object.type -- they were measured non-equivalent. Objects
    outside the freebase ns/ namespace are kept unconditionally (the RDF/OWL vocabulary layer).
    An ns/-object row is dropped only when the same assertion is present as type.object.type for
    that subject; anything unmatched goes to rdf_type_residue.parquet and is counted.
  Subject contiguity makes both comparisons exact in O(1) memory rather than sampled.

STREAMING, NOT EXPANSION. The source is never written out uncompressed (contract clause
SOURCE_EXPANSION). `gzip -dc` feeds `grep -F` at ~120 MB/s measured, and only matching lines reach
Python. grep is a PREFILTER ONLY -- it matches a pattern anywhere on the line, including in object
position, so every line is re-checked against the exact predicate field here.

NO QUERY CONDITIONING. This pass never reads a question, an answer, or a topic entity. It selects by
predicate, which is a source-level property of the dump.
"""
import json, os, subprocess, sys, time
from collections import Counter
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

RAW = "data/final_canonical/freebase_v3/_acquisition/raw/freebase-rdf-latest.gz"
PART = RAW + ".part"
DST = "data/final_canonical/freebase_v3/pass_a"
REC = "data/final_canonical/freebase_v3/V3_PASS_A_SCHEMA_METADATA.json"
BATCH = 2_000_000

NS = b"<http://rdf.freebase.com/ns/"
RDFS = b"<http://www.w3.org/2000/01/rdf-schema#"
RDFSYN = b"<http://www.w3.org/1999/02/22-rdf-syntax-ns#"

P_NAME = NS + b"type.object.name>"
P_LABEL = RDFS + b"label>"
P_TYPE = NS + b"type.object.type>"
P_RDFTYPE = RDFSYN + b"type>"
P_ALIAS = NS + b"common.topic.alias>"
P_DESC = NS + b"common.topic.description>"
P_REV = NS + b"type.property.reverse_property>"
P_MASTER = NS + b"type.property.master_property>"
P_KEY = NS + b"type.object.key>"

P_SCHEMA = {
    NS + b"type.property.schema>": "schema",
    NS + b"type.property.expected_type>": "expected_type",
    NS + b"type.property.unique>": "unique",
    NS + b"type.property.delegated>": "delegated",
    RDFS + b"domain>": "rdfs_domain",
    RDFS + b"range>": "rdfs_range",
}
P_HINT = {
    NS + b"freebase.type_hints.mediator>": "mediator",
    NS + b"freebase.type_hints.included_types>": "included_types",
    NS + b"freebase.type_hints.enumeration>": "enumeration",
    NS + b"freebase.type_hints.deprecated>": "deprecated",
}

GREP_PATTERNS = [P_NAME, P_LABEL, P_TYPE, P_RDFTYPE, P_ALIAS, P_DESC, P_REV, P_MASTER, P_KEY] \
    + list(P_SCHEMA) + list(P_HINT)

ESC = ((b"\\\\", b"\x00"), (b'\\"', b'"'), (b"\\n", b"\n"), (b"\\r", b"\r"), (b"\\t", b"\t"))


def uri(b):
    """<http://rdf.freebase.com/ns/m.0abc> -> 'm.0abc'. None for any other namespace."""
    if b.startswith(NS) and b.endswith(b">"):
        return b[len(NS):-1].decode("utf-8", "replace")
    return None


def unescape(b):
    for a, c in ESC:
        b = b.replace(a, c)
    s = b.replace(b"\x00", b"\\").decode("utf-8", "replace")
    if "\\u" in s or "\\U" in s:
        try:
            s = s.encode("latin-1", "backslashreplace").decode("unicode_escape")
        except Exception:
            pass
    return s


def literal(b):
    """-> (lexical, lang, datatype). Datatype is kept; the contract forbids silent coercion."""
    if not b.startswith(b'"'):
        return None
    if b.endswith(b">"):
        i = b.rfind(b'"^^<')
        if i > 0:
            return unescape(b[1:i]), "", b[i + 4:-1].decode("utf-8", "replace")
    i = b.rfind(b'"@')
    if i > 0 and b'"' not in b[i + 2:]:
        return unescape(b[1:i]), b[i + 2:].decode("utf-8", "replace"), ""
    if b.endswith(b'"'):
        return unescape(b[1:-1]), "", ""
    return unescape(b.strip(b'"')), "", ""


class Tbl:
    """Streaming Parquet writer over a fixed all-string schema."""

    def __init__(self, path, cols):
        self.sch = pa.schema([(c, pa.string()) for c in cols])
        self.path = path
        self.buf = [[] for _ in cols]
        self.w = None
        self.n = 0

    def add(self, *vals):
        for b, v in zip(self.buf, vals):
            b.append(v)
        if len(self.buf[0]) >= BATCH:
            self.flush()

    def flush(self):
        if not self.buf[0]:
            return
        if self.w is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.w = pq.ParquetWriter(self.path, self.sch, compression="zstd",
                                      compression_level=6)
        self.w.write_table(pa.Table.from_arrays(
            [pa.array(b, type=pa.string()) for b in self.buf], schema=self.sch))
        self.n += len(self.buf[0])
        for b in self.buf:
            b.clear()

    def close(self):
        self.flush()
        if self.w:
            self.w.close()
        return self.n


def main():
    t0 = time.time()
    limit = None
    tag = ""
    if "--limit-gb" in sys.argv:
        limit = int(float(sys.argv[sys.argv.index("--limit-gb") + 1]) * 1e9)
    if "--tag" in sys.argv:
        tag = "_" + sys.argv[sys.argv.index("--tag") + 1]
    src = RAW if os.path.exists(RAW) else PART
    if src == PART and not tag:
        raise SystemExit("refusing to run an untagged PASS A against an unverified .part file. "
                         "The mirror must pass the acquisition gate first.")
    dst = DST + tag
    rec = REC if not tag else REC.replace(".json", f"{tag}.json")

    t = {n: Tbl(f"{dst}/{n}.parquet", c) for n, c in {
        "name": ("uid", "name", "lang"),
        "label_residue": ("uid", "label", "lang"),
        "alias": ("uid", "alias", "lang"),
        "description": ("uid", "text", "lang"),
        "type": ("uid", "type"),
        "rdf_type": ("uid", "type"),
        "rdf_type_residue": ("uid", "type"),
        "key": ("uid", "value"),
        "property_schema": ("prop", "attr", "value"),
        "reverse_property": ("prop", "reverse_prop"),
        "master_property": ("dependent", "master"),
        "type_hints": ("subject", "hint", "value"),
    }.items()}

    st = {"lines_in": 0, "rows_emitted": 0, "prefilter_false_positives": 0,
          "name_nonliteral": 0, "type_nonuri": 0, "subject_not_ns": 0,
          "label_rows_seen": 0, "label_rows_dropped_as_identical": 0,
          "rdf_type_rows_seen": 0, "rdf_type_ns_dropped_as_redundant": 0}

    cmp_ = {"subjects_with_name_and_label": 0, "label_sets_identical": 0,
            "label_sets_differing": 0, "label_only_no_name": 0, "name_only_no_label": 0,
            "label_examples": [],
            "subjects_with_both_type_predicates": 0, "type_sets_identical": 0,
            "type_sets_differing": 0, "rdf_type_examples": []}

    langs = Counter()
    dtypes = Counter()

    # per-subject buffers, valid because the source is subject-grouped
    cur_raw = None   # raw subject bytes, for block-change detection
    cur = None       # decoded uid, for emission
    b_name, b_label, b_type, b_rdft = [], [], [], []

    def close_subject():
        if cur is None:
            return
        su = cur
        names = {(v, lg) for v, lg in b_name}
        labels = {(v, lg) for v, lg in b_label}
        for v, lg in b_name:
            t["name"].add(su, v, lg)
        st["label_rows_seen"] += len(b_label)
        if names and labels:
            cmp_["subjects_with_name_and_label"] += 1
            if names == labels:
                cmp_["label_sets_identical"] += 1
            else:
                cmp_["label_sets_differing"] += 1
                if len(cmp_["label_examples"]) < 8:
                    cmp_["label_examples"].append(
                        {"uid": su, "label_only": sorted(labels - names)[:3],
                         "name_only": sorted(names - labels)[:3]})
        elif labels:
            cmp_["label_only_no_name"] += 1
        elif names:
            cmp_["name_only_no_label"] += 1
        for v, lg in b_label:
            if (v, lg) in names:
                st["label_rows_dropped_as_identical"] += 1
            else:
                t["label_residue"].add(su, v, lg)

        tset = set(b_type)
        for ty in b_type:
            t["type"].add(su, ty)
        st["rdf_type_rows_seen"] += len(b_rdft)
        ns_rdft = {ty for ty, in_ns in b_rdft if in_ns}
        if tset and ns_rdft:
            cmp_["subjects_with_both_type_predicates"] += 1
            if tset == ns_rdft:
                cmp_["type_sets_identical"] += 1
            else:
                cmp_["type_sets_differing"] += 1
                if len(cmp_["rdf_type_examples"]) < 8:
                    cmp_["rdf_type_examples"].append(
                        {"uid": su, "rdf_type_only": sorted(ns_rdft - tset)[:3],
                         "object_type_only": sorted(tset - ns_rdft)[:3]})
        for ty, in_ns in b_rdft:
            if not in_ns:
                t["rdf_type"].add(su, ty)          # RDF/OWL vocabulary layer, always kept
            elif ty in tset:
                st["rdf_type_ns_dropped_as_redundant"] += 1
            else:
                t["rdf_type_residue"].add(su, ty)  # unmatched: kept, never silently dropped

    gz = subprocess.Popen(["gzip", "-dc", src], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, bufsize=1 << 22)
    args = ["grep", "-F"]
    for p in GREP_PATTERNS:
        args += ["-e", p.decode()]
    gp = subprocess.Popen(args, stdin=gz.stdout, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, bufsize=1 << 22)
    gz.stdout.close()

    read = 0
    last_report = time.time()
    for line in gp.stdout:
        st["lines_in"] += 1
        read += len(line)
        if limit and read > limit:
            break
        p = line.rstrip(b"\n").split(b"\t")
        if len(p) < 3:
            continue
        s, pr, o = p[0], p[1], p[2]

        if s != cur_raw:
            close_subject()
            cur_raw = s
            cur = uri(s)
            if cur is None:
                st["subject_not_ns"] += 1
            b_name, b_label, b_type, b_rdft = [], [], [], []
        if cur is None:
            continue

        if pr == P_NAME:
            L = literal(o)
            if L is None:
                st["name_nonliteral"] += 1
            else:
                b_name.append((L[0], L[1]))
                langs[L[1]] += 1
        elif pr == P_LABEL:
            L = literal(o)
            if L is not None:
                b_label.append((L[0], L[1]))
        elif pr == P_TYPE:
            ou = uri(o)
            if ou is None:
                st["type_nonuri"] += 1
            else:
                b_type.append(ou)
        elif pr == P_RDFTYPE:
            if o.startswith(NS):
                b_rdft.append((o[len(NS):-1].decode("utf-8", "replace"), True))
            else:
                b_rdft.append((o[1:-1].decode("utf-8", "replace")
                               if o.endswith(b">") else o.decode("utf-8", "replace"), False))
        elif pr == P_ALIAS:
            L = literal(o)
            if L:
                t["alias"].add(cur, L[0], L[1])
                langs[L[1]] += 1
        elif pr == P_DESC:
            L = literal(o)
            if L:
                t["description"].add(cur, L[0], L[1])
                langs[L[1]] += 1
        elif pr == P_REV:
            ou = uri(o)
            if ou:
                t["reverse_property"].add(cur, ou)
        elif pr == P_MASTER:
            ou = uri(o)
            if ou:
                t["master_property"].add(cur, ou)
        elif pr == P_KEY:
            L = literal(o)
            if L:
                t["key"].add(cur, L[0])
        elif pr in P_SCHEMA:
            ou = uri(o)
            if ou is None:
                L = literal(o)
                if L:
                    ou, dt = L[0], L[2]
                    if dt:
                        dtypes[dt] += 1
                else:
                    ou = o.decode("utf-8", "replace")
            t["property_schema"].add(cur, P_SCHEMA[pr], ou)
        elif pr in P_HINT:
            ou = uri(o)
            if ou is None:
                L = literal(o)
                ou = L[0] if L else o.decode("utf-8", "replace")
            t["type_hints"].add(cur, P_HINT[pr], ou)
        else:
            st["prefilter_false_positives"] += 1

        if st["lines_in"] % 2_000_000 == 0:
            now = time.time()
            if now - last_report >= 60:
                el = now - t0
                print(f"  {st['lines_in']/1e6:8.1f}M lines  {read/1e9:6.2f} GB matched  "
                      f"{read/1e6/el:5.1f} MB/s  t={el/60:5.1f}m  "
                      f"rows={sum(tb.n + len(tb.buf[0]) for tb in t.values())/1e6:.1f}M",
                      flush=True)
                last_report = now

    close_subject()
    # A truncated decompression would produce plausible-but-incomplete outputs, so gzip's exit
    # status is checked rather than discarded. Reaching EOF with rc 0 is what proves the gzip
    # stream is intact end to end -- the digests proved the BYTES, not the stream.
    gp.stdout.close()
    try:
        gp.wait(timeout=120)
        gz.wait(timeout=120)
    except Exception:
        pass
    rc_gzip, rc_grep = gz.returncode, gp.returncode
    if not limit and rc_gzip not in (0, None):
        raise SystemExit(f"gzip exited {rc_gzip}: the source did not decompress to EOF. "
                         f"PASS A output is incomplete and must not be used.")

    counts = {n: tb.close() for n, tb in t.items()}
    st["rows_emitted"] = sum(counts.values())
    sizes = {f: os.path.getsize(f"{dst}/{f}") for f in sorted(os.listdir(dst))} \
        if os.path.isdir(dst) else {}

    med_true = med_false = 0
    mediators = []
    hp = f"{dst}/type_hints.parquet"
    if os.path.exists(hp):
        th = pq.read_table(hp)
        m = th.filter(pc.equal(th["hint"], "mediator"))
        subs = m.column("subject").to_pylist()
        vals = m.column("value").to_pylist()
        med_true = sum(1 for v in vals if v == "true")
        med_false = sum(1 for v in vals if v == "false")
        mediators = sorted({s for s, v in zip(subs, vals) if v == "true"})

    out = {
        "schema": "V3_PASS_A_SCHEMA_METADATA/v2",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract": "V3_CANONICAL_CONTRACT_V2.json PIPELINE_V2.PASS_A",
        "source": src,
        "partial_run": bool(limit),
        "limit_gb": (limit / 1e9) if limit else None,
        "elapsed_s": round(time.time() - t0, 1),
        "gzip_returncode": rc_gzip,
        "grep_returncode": rc_grep,
        "STREAM_REACHED_EOF": rc_gzip == 0,
        "counts": counts,
        "parse_stats": st,
        "output_bytes": sizes,
        "output_gb": round(sum(sizes.values()) / 1e9, 3),
        "language_histogram_top": dict(langs.most_common(25)),
        "distinct_languages": len(langs),
        "datatype_histogram": dict(dtypes.most_common(25)),

        "REVERSE_PROPERTY_MAP": {
            "pairs": counts.get("reverse_property", 0),
            "master_property_pairs": counts.get("master_property", 0),
            "why_it_matters": "absent from the IDIR archive (FINDING_1). INVARIANT_2 could only be "
                              "bounded at 25 predicates / 17,930 triples because the mapping was "
                              "unavailable. PASS D now canonicalises orientation from the source "
                              "schema rather than from an inference about IDIR's anti-join.",
            "cross_check": "reverse_property and master_property are independent expressions of "
                           "the same pairing. Disagreement between them is a signal worth reading, "
                           "not an error to suppress.",
        },
        "MEDIATOR_DECLARATION": {
            "declared_true": med_true,
            "declared_false": med_false,
            "distinct_mediator_types": len(mediators),
            "why_it_matters": "check 6 measured that IDIR's types do NOT separate CVTs from "
                              "entities: 68.081% of CVT type mass sits on types entities also "
                              "carry, against a 5% pre-registered threshold. The dump declares "
                              "mediator status directly, so V3 reads the classification instead of "
                              "inferring it. node_kind_source records which of the two produced "
                              "any given label.",
        },
        "LABEL_VS_NAME": {
            **{k: v for k, v in cmp_.items() if k.startswith(("subjects_with_name", "label_"))},
            "residue_rows_kept": counts.get("label_residue", 0),
            "rows_dropped_as_identical": st["label_rows_dropped_as_identical"],
            "method": "exact per-subject set comparison over the whole source, enabled by subject "
                      "contiguity. Not a sample.",
            "reading": "differing == 0 means rdfs:label carried nothing beyond type.object.name "
                       "and dropping the identical rows loses nothing. Any differing row is in "
                       "label_residue.parquet, so the discard is provable rather than asserted.",
        },
        "RDF_TYPE_VS_OBJECT_TYPE": {
            **{k: v for k, v in cmp_.items() if k.startswith(("subjects_with_both", "type_sets",
                                                              "rdf_type_"))},
            "rdf_vocabulary_rows_kept": counts.get("rdf_type", 0),
            "ns_residue_rows_kept": counts.get("rdf_type_residue", 0),
            "ns_rows_dropped_as_redundant": st["rdf_type_ns_dropped_as_redundant"],
            "the_rule": "objects outside the freebase ns/ namespace are kept unconditionally. An "
                        "ns/-object row is dropped ONLY when the identical assertion exists as "
                        "type.object.type for the same subject. Nothing is collapsed unmeasured.",
        },
        "GREP_IS_A_PREFILTER_ONLY": "grep -F matches anywhere on the line, including in object "
                                    "position. Every line is re-checked against the exact "
                                    "predicate field here. prefilter_false_positives is the count "
                                    "grep let through and this pass rejected.",
        "NO_QUERY_CONDITIONING": "selection is by predicate, a source-level property. No question, "
                                 "answer or topic entity is read anywhere in this pass.",
    }
    if mediators:
        out["MEDIATOR_DECLARATION"]["sample_mediator_types"] = mediators[:40]
    if cmp_["label_examples"]:
        out["LABEL_VS_NAME"]["examples"] = cmp_["label_examples"]
    if cmp_["rdf_type_examples"]:
        out["RDF_TYPE_VS_OBJECT_TYPE"]["examples"] = cmp_["rdf_type_examples"]

    os.makedirs(os.path.dirname(rec), exist_ok=True)
    tmp = rec + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, rec)
    print(json.dumps({"counts": counts, "elapsed_s": out["elapsed_s"],
                      "output_gb": out["output_gb"], "mediator_true": med_true,
                      "reverse_pairs": counts.get("reverse_property", 0),
                      "label_differing": cmp_["label_sets_differing"],
                      "rdf_type_sets_differing": cmp_["type_sets_differing"],
                      "rdf_type_residue": counts.get("rdf_type_residue", 0)}, indent=1))


if __name__ == "__main__":
    main()
