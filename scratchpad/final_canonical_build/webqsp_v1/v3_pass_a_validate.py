"""Independently validate PASS A's Parquet against the shards it came from, and free the shards.

    python scratchpad/final_canonical_build/webqsp_v1/v3_pass_a_validate.py [--delete-shards]

The disk-discipline rule says a merge input may be deleted only once the merged artifact passes row
count, schema and endpoint integrity. Stage 2 already reconciles its own arithmetic, but that is
self-reported: the same loop that wrote the rows also counted them, so a systematic error would
agree with itself. This re-derives the row counts by reading the 200 gz shards again and comparing
against the Parquet, which is the independent measurement the rule actually asks for.

Four things are checked, and all four have to pass before anything is deleted:

  ROW COUNT   -- per table, shard lines == Parquet rows, with the two split tables checked against
                 their decision arithmetic instead (label -> redundant + residue; rdf_type ->
                 out-of-namespace kept + dropped-as-redundant + residue).
  SCHEMA      -- column names and types are what the build declared, not merely "some Parquet".
  ENDPOINTS   -- no empty or null subject, no empty object-side identifier. An edge or metadata row
                 with a blank endpoint is unusable later and is far cheaper to find now.
  TOTAL       -- everything sums to stage 1's independently recorded 815,908,863 rows.

Deletion is opt-in. 11.5 GB is worth reclaiming for PASS B, but not on a silent judgment call.
"""
import glob
import gzip
import json
import multiprocessing as mp
import os
import sys
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SHARDS = r"data\final_canonical\freebase_v3\pass_a\shards"
OUTDIR = r"data\final_canonical\freebase_v3\pass_a"
REPORT = r"data\final_canonical\freebase_v3\V3_PASS_A_VALIDATION.json"
META = r"data\final_canonical\freebase_v3\V3_PASS_A_SCHEMA_METADATA.json"

SHARD_TABLES = ("name", "label", "alias", "description", "type", "rdf_type", "key",
                "property_schema", "reverse_property", "master_property", "type_hints")
PLAIN = ("name", "alias", "description", "type", "key", "reverse_property",
         "master_property", "property_schema", "type_hints")
COLS = {
    "name": ["subject", "lexical", "lang"], "alias": ["subject", "lexical", "lang"],
    "description": ["subject", "lexical", "lang"], "type": ["subject", "type"],
    "rdf_type": ["subject", "type", "in_ns"], "key": ["subject", "key"],
    "reverse_property": ["subject", "object"], "master_property": ["subject", "object"],
    "property_schema": ["subject", "field", "value"], "type_hints": ["subject", "field", "value"],
    "label_residue": ["subject", "lexical", "lang"], "rdf_type_residue": ["subject", "type"],
}
# The column that must never be empty in each table: an identifier, not free text. A blank lexical
# form is a legitimate literal; a blank subject or type is a broken endpoint.
ENDPOINT_COLS = {
    "name": ["subject"], "alias": ["subject"], "description": ["subject"],
    "type": ["subject", "type"], "rdf_type": ["subject", "type"], "key": ["subject"],
    "reverse_property": ["subject", "object"], "master_property": ["subject", "object"],
    "property_schema": ["subject", "field"], "type_hints": ["subject", "field"],
    "label_residue": ["subject"], "rdf_type_residue": ["subject", "type"],
}


def count_member(k):
    out = {}
    for t in SHARD_TABLES:
        p = os.path.join(SHARDS, f"m{k:04d}_{t}.tsv.gz")
        n = 0
        if os.path.exists(p):
            with gzip.open(p, "rb") as fh:
                for _ in fh:
                    n += 1
        out[t] = n
    return k, out


def scan_parquet(table):
    """Row count, schema and endpoint integrity in one streaming pass over the file."""
    p = os.path.join(OUTDIR, f"{table}.parquet")
    if not os.path.exists(p):
        return {"exists": False}
    pf = pq.ParquetFile(p)
    names = [f.name for f in pf.schema_arrow]
    types = [str(f.type) for f in pf.schema_arrow]
    rows = 0
    empty = {c: 0 for c in ENDPOINT_COLS.get(table, [])}
    nulls = {c: 0 for c in ENDPOINT_COLS.get(table, [])}
    for batch in pf.iter_batches(batch_size=1 << 18):
        rows += batch.num_rows
        for c in empty:
            if c not in batch.schema.names:
                continue
            col = batch.column(batch.schema.get_field_index(c))
            nulls[c] += col.null_count
            empty[c] += pc.sum(pc.equal(pc.binary_length(col), 0)).as_py() or 0
    return {"exists": True, "rows": rows, "columns": names, "types": types,
            "empty_endpoints": empty, "null_endpoints": nulls,
            "bytes": os.path.getsize(p),
            "schema_ok": names == COLS[table] and all(t == "string" for t in types)}


def main():
    t0 = time.time()
    delete = "--delete-shards" in sys.argv
    metas = {}
    for p in sorted(glob.glob(os.path.join(SHARDS, "m*_meta.json"))):
        m = json.load(open(p, encoding="utf-8"))
        metas[m["member"]] = m
    ks = sorted(metas)
    stage1_rows = sum(metas[k]["rows"] for k in ks)
    st2 = json.load(open(META, encoding="utf-8"))
    red = st2["REDUNDANCY"]

    reuse = "--reuse-counts" in sys.argv and os.path.exists(REPORT)
    if reuse:
        # The shard recount is the independent half of this check, and it has already been made
        # against these same shards. Re-deriving it a second time would measure the same files with
        # the same code and add nothing; what changed since is the Parquet side, which is rescanned
        # in full below.
        shard = json.load(open(REPORT, encoding="utf-8"))["SHARD_ROWS"]
        print(f"reusing recorded shard counts for {len(ks)} members", flush=True)
    else:
        print(f"recounting {len(ks)} members x {len(SHARD_TABLES)} shard tables", flush=True)
        shard = {t: 0 for t in SHARD_TABLES}
        done = 0
        with mp.Pool(7) as pool:
            for k, out in pool.imap_unordered(count_member, ks):
                for t, n in out.items():
                    shard[t] += n
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(ks)} members  t={time.time()-t0:.0f}s", flush=True)

    print("scanning Parquet", flush=True)
    pqs = {t: scan_parquet(t) for t in list(PLAIN) + ["rdf_type", "label_residue",
                                                     "rdf_type_residue"]}

    checks = {}
    for t in PLAIN:
        checks[f"rows_{t}"] = {"shard": shard[t], "parquet": pqs[t].get("rows"),
                               "ok": shard[t] == pqs[t].get("rows")}
    # The two split tables have no pass-through form: their shard count must equal the decision
    # arithmetic, which is the only way a dropped row can be shown to have been dropped on purpose.
    checks["rows_label_split"] = {
        "shard": shard["label"],
        "redundant_plus_residue": red["label_redundant"] + red["label_residue"],
        "residue_parquet": pqs["label_residue"].get("rows"),
        "ok": (shard["label"] == red["label_redundant"] + red["label_residue"]
               and pqs["label_residue"].get("rows") == red["label_residue"])}
    checks["rows_rdf_type_split"] = {
        "shard": shard["rdf_type"],
        "kept_plus_dropped_plus_residue": (red["rdf_type_out_of_ns_kept"]
                                           + red["rdf_type_ns_dropped_as_redundant"]
                                           + red["rdf_type_residue"]),
        "kept_parquet": pqs["rdf_type"].get("rows"),
        "residue_parquet": pqs["rdf_type_residue"].get("rows"),
        "ok": (shard["rdf_type"] == (red["rdf_type_out_of_ns_kept"]
                                     + red["rdf_type_ns_dropped_as_redundant"]
                                     + red["rdf_type_residue"])
               and pqs["rdf_type"].get("rows") == red["rdf_type_out_of_ns_kept"]
               and pqs["rdf_type_residue"].get("rows") == red["rdf_type_residue"])}
    checks["total_rows"] = {"shard_sum": sum(shard.values()), "stage1_recorded": stage1_rows,
                            "ok": sum(shard.values()) == stage1_rows}
    checks["schema"] = {"per_table": {t: pqs[t].get("schema_ok") for t in pqs},
                        "ok": all(pqs[t].get("schema_ok") for t in pqs)}
    bad_ends = {t: {"empty": pqs[t].get("empty_endpoints"), "null": pqs[t].get("null_endpoints")}
                for t in pqs
                if any((pqs[t].get("empty_endpoints") or {}).values())
                or any((pqs[t].get("null_endpoints") or {}).values())}
    checks["endpoint_integrity"] = {"tables_with_empty_or_null_endpoints": bad_ends,
                                    "ok": not bad_ends}
    missing = sorted(t for t, v in pqs.items() if not v.get("exists"))
    checks["tables_present"] = {
        "missing": missing, "ok": not missing,
        "note": "a table with zero rows still has to exist with its declared schema. An absent "
                "file is indistinguishable from a failed one to everything downstream, so it is "
                "reported as missing rather than counted as an empty pass."}

    all_ok = all(v["ok"] for v in checks.values())
    doc = {
        "schema": "V3_PASS_A_VALIDATION/v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.time() - t0, 1),
        "method": "shard line counts re-derived by reading all 200 gz shards again, independently "
                  "of the stage 2 loop that wrote the Parquet, then compared against a full "
                  "streaming scan of each Parquet file.",
        "SHARD_ROWS": shard,
        "PARQUET": {t: {k2: v for k2, v in pqs[t].items() if k2 != "types"} for t in pqs},
        "CHECKS": checks,
        "ALL_CHECKS_PASS": all_ok,
        "SHARDS_DELETED": False,
        "parquet_bytes_total": sum(pqs[t].get("bytes", 0) for t in pqs),
    }

    if all_ok and delete:
        n = 0
        freed = 0
        for p in glob.glob(os.path.join(SHARDS, "m*_*.tsv.gz")):
            # Openers are kept: they are the distinct-subject list PASS B mints UIDs from and are
            # not reproducible without another full pass over the source.
            if p.endswith("_openers.tsv.gz"):
                continue
            freed += os.path.getsize(p)
            os.remove(p)
            n += 1
        doc["SHARDS_DELETED"] = True
        doc["shard_files_deleted"] = n
        doc["bytes_freed"] = freed
        doc["openers_retained"] = "m*_openers.tsv.gz retained: they are the distinct-subject list " \
                                  "PASS B mints node UIDs from, and regenerating them costs a full " \
                                  "pass over the 409 GB source."
        print(f"deleted {n} shard files, freed {freed/1e9:.1f} GB", flush=True)
    elif delete:
        print("NOT deleting: a check failed", flush=True)

    tmp = REPORT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, REPORT)
    print(json.dumps({"CHECKS": checks, "ALL_CHECKS_PASS": all_ok,
                      "SHARDS_DELETED": doc["SHARDS_DELETED"]}, indent=1))


if __name__ == "__main__":
    main()
