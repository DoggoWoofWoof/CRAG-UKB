"""Where did the 2010 recovery land?  Cross-tab it against the frozen overlay's own classes.

    PYTHONHASHSEED=0 python .../fb2010_breakdown.py

The join count alone (167,487 residue / 162,597 hunt) does not say WHICH residue the 2010 dump
reached.  This does, on three axes that already exist in semantic_kind_v2_1:
    recovery_class        what the overlay guessed the node is
    set_named_rate_band   the band the user ordered spend by (HIGH/MID/LOW/UNTYPED/NEAR_ZERO)
    nameless_grade        whether the overlay had already called the node nameless
The third axis is the one that can falsify something.  If the 2010 dump names nodes the overlay
graded *_NAMELESS, the grading is wrong and must be revised: a name published in Freebase's own
2010 dump outranks an inference drawn from the 2015 dump's silence.

Also reports how many recovered rows carry a 2010 type list, a 2010 /en/ key and a Wikipedia
numeric page id, because those are separate recovery rungs delivered by the same file.

All counting is done with pyarrow.compute value_counts on dictionary-encoded columns; nothing is
converted to Python objects per row.  (The first version of this script used to_pylist() and would
have taken about an hour on 69.8M rows x 3 columns.)

OUTPUT  V3_FB2010_BREAKDOWN.json  (a new record; nothing frozen is touched)
"""
import sys, io, os, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
t0 = time.time()

T = pa.concat_tables([pq.read_table(fp, columns=["node_uid", "en_key_2010",
                                                 "wikipedia_en_page_id", "types_2010", "in_hunt"])
                      for fp in sorted(glob.glob(f"{ACQ}/fb2010_names/*.parquet"))])
uid = np.sort(T["node_uid"].to_numpy())
print(f"2010 recoveries on disk: {T.num_rows:,} rows, {len(np.unique(uid)):,} distinct nodes "
      f"({time.time()-t0:.0f}s)", flush=True)


def nonempty(c):
    return int(pc.sum(pc.cast(pc.greater(pc.binary_length(T[c]), 0), pa.int64())).as_py() or 0)


n_types, n_wp, n_key = nonempty("types_2010"), nonempty("wikipedia_en_page_id"), nonempty("en_key_2010")
n_hunt = int(pc.sum(pc.cast(T["in_hunt"], pa.int64())).as_py() or 0)

# top 2010 types among the recovered nodes -- what the 2015 graph forgot these nodes were
tc = collections.Counter()
for s in T["types_2010"].to_pylist():
    if s:
        tc.update(t for t in s.split(",") if t)

# ---------------------------------------------------------------- overlay cross-tab
AX = ["recovery_class", "set_named_rate_band", "nameless_grade"]
pf = pq.ParquetFile(f"{ACQ}/semantic_kind_v2_1.parquet")
have = [c for c in AX if c in set(pf.schema_arrow.names)]
print(f"axes present: {have}", flush=True)
ct = {c: collections.Counter() for c in have}      # recovered by the 2010 dump
pop = {c: collections.Counter() for c in have}     # whole class, for the reach rate
seen = 0


def count_into(counter, col):
    col = pc.cast(col, pa.string()) if pa.types.is_dictionary(col.type) else col
    vc = pc.value_counts(col)
    for st in vc:
        counter[str(st["values"].as_py())] += int(st["counts"].as_py())


for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g, columns=["node_uid"] + have)
    u = t["node_uid"].to_numpy()
    p = np.clip(np.searchsorted(uid, u), 0, len(uid) - 1)
    hit = uid[p] == u
    seen += int(hit.sum())
    hm = pa.array(hit)
    for c in have:
        count_into(pop[c], t[c])
        if hit.any():
            count_into(ct[c], pc.filter(t[c], hm))
    if g % 10 == 0:
        print(f"  rg {g}/{pf.metadata.num_row_groups} matched {seen:,} ({time.time()-t0:.0f}s)",
              flush=True)


def tab(c):
    return {k: {"named_by_2010_dump": v, "class_population": pop[c][k],
                "reach_pct": round(100.0 * v / max(1, pop[c][k]), 3)}
            for k, v in sorted(ct[c].items(), key=lambda x: -x[1])}


nameless_hits = {k: v["named_by_2010_dump"] for k, v in (tab("nameless_grade").items()
                 if "nameless_grade" in have else []) if "NAMELESS" in k.upper()}
rec = {"schema": "FB2010_BREAKDOWN/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "new record; frozen overlay read-only, locked provenance order untouched.",
       "recoveries_on_disk": int(T.num_rows),
       "matched_in_overlay": seen,
       "in_hunt": n_hunt,
       "extra_rungs_delivered_by_the_same_file": {
           "rows_with_a_2010_type_list": n_types,
           "rows_with_a_2010_en_key": n_key,
           "rows_with_a_wikipedia_numeric_page_id": n_wp},
       "top_30_types_the_2015_graph_no_longer_states": [
           {"type_2010": k, "n": v} for k, v in tc.most_common(30)],
       "by_recovery_class": tab("recovery_class") if "recovery_class" in have else None,
       "by_set_named_rate_band": tab("set_named_rate_band") if "set_named_rate_band" in have else None,
       "by_nameless_grade": tab("nameless_grade") if "nameless_grade" in have else None,
       "NAMELESS_GRADES_CONTRADICTED": nameless_hits,
       "FALSIFICATION_CHECK": "any non-zero count in NAMELESS_GRADES_CONTRADICTED is an overlay "
                              "grading error: Freebase's own 2010 dump published a name for a node "
                              "the overlay called nameless. The 2010 name wins; the grade must be "
                              "revised in a new overlay record, never by editing the frozen one.",
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FB2010_BREAKDOWN.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps(rec, indent=1, ensure_ascii=False))
