"""The nodes the overlay graded *_NAMELESS that Freebase's own 2010 dump names.

    PYTHONHASHSEED=0 python .../nameless_contradictions.py

WHY THIS IS ITS OWN RECORD
  The nameless grades are the terminal claim of the whole campaign: a node graded nameless is a node
  we assert never had a name.  V3_FB2010_BREAKDOWN found 4 nodes graded EMPIRICALLY_NAMELESS for
  which the 2010 dump publishes a /type/object/name.  Four is small, but the direction is what
  matters: an empirical grade is an inference from the 2015 dump's silence, and a name published by
  Freebase itself in 2010 outranks it.  Each one is listed here with its 2010 name, key, types and
  Wikipedia page id so the contradiction is auditable rather than a count.

  This is exactly why the three grades are kept apart: SOURCE_DECLARED_NAMELESS could not be
  contradicted this way, EMPIRICALLY_NAMELESS can, and this is the evidence that it can.

Nothing is edited.  The frozen overlay keeps its grades; correcting them is a new overlay record.
OUTPUT  V3_NAMELESS_CONTRADICTIONS.json
"""
import sys, io, os, json, glob, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
t0 = time.time()

T = pa.concat_tables([pq.read_table(fp) for fp in sorted(glob.glob(f"{ACQ}/fb2010_names/*.parquet"))])
uid = T["node_uid"].to_numpy()
order = np.argsort(uid)
suid = uid[order]

GRADES = pa.array(["SOURCE_DECLARED_NAMELESS", "EMPIRICALLY_NAMELESS", "INFERRED_NAMELESS"])
pf = pq.ParquetFile(f"{ACQ}/semantic_kind_v2_1.parquet")
have = [c for c in ("nameless_grade", "recovery_class", "set_named_rate_band")
        if c in set(pf.schema_arrow.names)]
bad_uid, bad_grade = [], []
for g in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(g, columns=["node_uid"] + have)
    ng = t["nameless_grade"]
    ng = pc.cast(ng, pa.string()) if pa.types.is_dictionary(ng.type) else ng
    # NOT_NAMELESS contains "NAMELESS" as a substring, so the grades are named explicitly.
    m = pc.is_in(ng, value_set=GRADES)
    if not pc.any(m).as_py():
        continue
    sub_u = pc.filter(t["node_uid"], m).to_numpy()
    sub_g = pc.filter(ng, m).to_pylist()
    p = np.clip(np.searchsorted(suid, sub_u), 0, len(suid) - 1)
    hit = suid[p] == sub_u
    for i in np.flatnonzero(hit).tolist():
        bad_uid.append(int(sub_u[i])); bad_grade.append(sub_g[i])
print(f"contradicted nodes: {len(bad_uid)} ({time.time()-t0:.0f}s)", flush=True)

pos = {int(u): i for i, u in enumerate(uid.tolist())}
items = []
for u, gr in zip(bad_uid, bad_grade):
    i = pos[u]
    items.append({"node_uid": u, "node_id": T["node_id"][i].as_py(),
                  "graded": gr,
                  "name_2010": T["display_name"][i].as_py(),
                  "en_key_2010": T["en_key_2010"][i].as_py(),
                  "types_2010": T["types_2010"][i].as_py(),
                  "wikipedia_en_page_id": T["wikipedia_en_page_id"][i].as_py(),
                  "description_2010": (T["description_2010"][i].as_py() or "")[:300]})
items.sort(key=lambda d: d["node_id"])

rec = {"schema": "NAMELESS_CONTRADICTIONS/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "new record. The frozen overlay is NOT edited; its grades stand until a new "
                      "overlay record with its own hash revises them.",
       "source_of_the_contradiction": "freebase-simple-topic-dump.tsv.bz2, release 2010-07-16, "
                                      "checksum-verified (V3_FB2010_ACQUISITION_SIMPLE.json)",
       "n_contradicted": len(items),
       "WHAT_IT_MEANS": "an EMPIRICALLY_NAMELESS grade is an inference from the 2015 dump's silence. "
                        "A /type/object/name published by Freebase in 2010 is a source assertion and "
                        "outranks it. SOURCE_DECLARED_NAMELESS is not contradictable this way, which "
                        "is the reason the grades are kept apart rather than merged.",
       "items": items,
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_NAMELESS_CONTRADICTIONS.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps(rec, indent=1, ensure_ascii=False))
