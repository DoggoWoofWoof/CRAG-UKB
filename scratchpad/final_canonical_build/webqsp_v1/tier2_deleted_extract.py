"""TIER 2 -- one streaming pass over Google's deleted-triples dump, serving both targets.

Format (8 comma-separated fields, only `object` may itself contain commas):
    created_ts, creator, deleted_ts, deletor, subject, predicate, object, lang

Two extractions in the same pass, because the archive is a single gzip stream and cannot be
re-read cheaply:
  A) every row whose SUBJECT is one of the 562 unresolved mediator MIDs -- tiny, kept in memory
  B) every /type/object/name row in the whole dump -- the candidate pool for the 19,387,189
     unnamed ENTITY_MIDs, written to parquet in batches and joined afterwards
Plus a global predicate histogram, which is free here and tells us what this dump can ever offer.
"""
import tarfile, json, time, os
from collections import Counter
import pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
SRC = f"{V3}/_acquisition/raw/deleted_freebase.tar.gz"
OUT = f"{V3}/_acquisition/deleted_names.parquet"

frz = json.load(open(f"{V3}/V3_PASS_A_SCHEMA_FREEZE.json", encoding="utf-8"))
UNRES = set(frz["DECLARATIONS"]["MEDIATOR_MIDS_UNRESOLVED"])          # 'm.0101csmw' form
UNRES_SLASH = {"/m/" + m[2:] for m in UNRES}                          # '/m/0101csmw' form
print(f"targeting {len(UNRES_SLASH)} mediator MIDs", flush=True)

NAME_PRED = "/type/object/name"
schema = pa.schema([("subject", pa.string()), ("object", pa.string()), ("lang", pa.string())])
writer = pq.ParquetWriter(OUT, schema, compression="zstd")

pred_hist = Counter()
mediator_rows = []
buf_s, buf_o, buf_l = [], [], []
lines = kept_names = bad = 0
t0 = time.time()

def flush():
    global buf_s, buf_o, buf_l
    if buf_s:
        writer.write_table(pa.table({"subject": buf_s, "object": buf_o, "lang": buf_l},
                                    schema=schema))
        buf_s, buf_o, buf_l = [], [], []

tf = tarfile.open(SRC, "r|gz")
for m in tf:
    if not m.isfile():
        continue
    f = tf.extractfile(m)
    for raw in f:
        lines += 1
        try:
            s = raw.decode("utf-8", "replace")
        except Exception:
            bad += 1; continue
        p = s.rstrip("\n").split(",", 6)
        if len(p) < 7:
            bad += 1; continue
        subj, pred, rest = p[4], p[5], p[6]
        if "," in rest:
            obj, lang = rest.rsplit(",", 1)
        else:
            obj, lang = rest, ""
        pred_hist[pred] += 1
        if pred == NAME_PRED:
            kept_names += 1
            buf_s.append(subj); buf_o.append(obj); buf_l.append(lang)
            if len(buf_s) >= 500_000:
                flush()
        if subj in UNRES_SLASH:
            mediator_rows.append({"subject": subj, "predicate": pred, "object": obj, "lang": lang,
                                  "created_ts": p[0], "creator": p[1],
                                  "deleted_ts": p[2], "deletor": p[3]})
    print(f"  {m.name}  lines={lines:,}  names={kept_names:,}  med={len(mediator_rows):,}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
flush(); writer.close(); tf.close()

print(f"\ntotal lines read      : {lines:,}")
print(f"unparsable lines      : {bad:,}")
print(f"/type/object/name rows: {kept_names:,}")
print(f"rows on the 562 MIDs  : {len(mediator_rows):,}")
print(f"distinct predicates   : {len(pred_hist):,}")
print("\ntop predicates in the deleted dump:")
for p, n in pred_hist.most_common(15):
    print(f"   {n:>10,}  {p}")

json.dump({
    "schema": "TIER2_DELETED_EXTRACT/v1",
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source": {"artifact": "deleted_freebase.tar.gz", "members": 20,
               "md5": "3a2b903862ea9d7d79f106a3821c3b02",
               "published_triple_count": 63036271,
               "coverage": "one-time dump of triples deleted through March 2013"},
    "lines_read": lines,
    "unparsable_lines": bad,
    "matches_published_count": lines == 63036271,
    "name_rows_extracted": kept_names,
    "rows_on_unresolved_mediators": len(mediator_rows),
    "distinct_predicates": len(pred_hist),
    "TOP_PREDICATES": dict(pred_hist.most_common(60)),
    "MEDIATOR_ROWS": mediator_rows[:2000],
    "names_parquet": OUT,
}, open(f"{V3}/V3_TIER2_DELETED_EXTRACT.json", "w", encoding="utf-8"), indent=1)
print("\nwrote V3_TIER2_DELETED_EXTRACT.json and", OUT)
