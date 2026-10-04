"""Stage 3: go to the deleted-triples dump for the 562 unresolved mediator MIDs. Append-only.

Stage 1B recovered paths for 239 of the 562 from keys in the frozen metadata and found all 239
have zero instances. Stage 2 showed none of the 562 appears as a raw MID in type.object.type.
What remains open is the 323 with NO key in the frozen metadata: their type paths are unknown, so
we cannot check whether one of the 6,698 unmapped instantiated paths from stage 1C belongs to
them.

deleted_freebase.tar.gz is Google's one-time dump of triples deleted through March 2013 -- exactly
where a key that no longer exists in the final snapshot would be. This pass keeps EVERY predicate
for the 562 (not just names, which is all the tier-2 extract persisted), so type and key rows are
available this time.

Changes nothing: 301,977,131 nodes / 2,062,430,072 edges either way.
"""
import sys, io, os, json, tarfile, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

V3 = "data/final_canonical/freebase_v3"
TAR = f"{V3}/_acquisition/raw/deleted_freebase.tar.gz"
frz = json.load(io.open(f"{V3}/V3_PASS_A_SCHEMA_FREEZE.json", encoding="utf-8"))
dec = frz["DECLARATIONS"]
unres = dec["MEDIATOR_MIDS_UNRESOLVED"]
unres = sorted(unres.keys() if isinstance(unres, dict) else unres)
b1b = json.load(io.open(f"{V3}/V3_MEDIATOR_RESOLUTION_STAGE1B.json", encoding="utf-8"))
have_path = set(b1b.get("MID_TO_PATH_FROM_KEY", {}))
need = [m for m in unres if m not in have_path]
print(f"unresolved: {len(unres)}   already have a path: {len(have_path)}   still need: {len(need)}")

# the dump writes MIDs as /m/xxx; our node_ids are m.xxx
want = set()
for m in unres:
    want.add("/" + m.replace(".", "/", 1))
needset = {"/" + m.replace(".", "/", 1) for m in need}

t0 = time.time()
rows = collections.defaultdict(list)
lines = 0
tf = tarfile.open(TAR, "r:gz")
for member in tf:
    if not member.isfile():
        continue
    f = tf.extractfile(member)
    if f is None:
        continue
    for raw in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
        lines += 1
        # created_ts,creator,deleted_ts,deleter,SUBJECT,PREDICATE,OBJECT,lang -- the object may
        # itself contain commas, so bound the split at 6 and take lang off the tail, exactly as
        # the tier-2 extract does. Testing only field 4 is one set lookup per line, not seven.
        parts = raw.rstrip("\n").split(",", 6)
        if len(parts) < 7:
            continue
        subj = parts[4]
        if subj not in want:
            continue
        obj, _, lang = parts[6].rpartition(",")
        rows[subj].append((parts[5], obj or parts[6], lang))
        if lines % 10_000_000 == 0:
            print(f"  {lines:,} lines  hits={sum(len(v) for v in rows.values()):,} "
                  f"({time.time()-t0:.0f}s)", flush=True)
tf.close()

hits = sum(len(v) for v in rows.values())
print(f"\nlines read: {lines:,}   rows touching one of the 562: {hits:,}   "
      f"MIDs touched: {len(rows):,}  ({time.time()-t0:.0f}s)")

# what predicates showed up, and did any yield a type path?
pred = collections.Counter()
paths = {}
for mid, rs in rows.items():
    m = mid.lstrip("/").replace("/", ".", 1)
    for predicate, obj, lang in rs:
        pred[predicate] += 1
        # a type MID's key IS its path, e.g. /film/performance
        if predicate == "/type/object/key":
            cand = obj.strip().strip('"')
            if cand.startswith("/") and cand.count("/") >= 2:
                paths.setdefault(m, set()).add(cand)

print(f"\npredicates seen on the 562: {dict(pred.most_common(10))}")
print(f"MIDs with a key-derived path from the deleted dump: {len(paths):,}")
newly = {m: sorted(v) for m, v in paths.items() if m in set(need)}
print(f"  of which are among the {len(need)} still needed: {len(newly):,}")
for m, v in list(newly.items())[:10]:
    print(f"    {m} -> {v}")

rec = {"schema": "MEDIATOR_RESOLUTION_STAGE3/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("does not modify canonical/. 301,977,131 nodes / 2,062,430,072 edges "
                       "regardless of the result."),
       "QUESTION": ("can the deleted-triples dump supply type paths for the 323 unresolved "
                    "mediator MIDs that have no key in the frozen metadata?"),
       "source": {"artifact": "deleted_freebase.tar.gz",
                  "note": "Google's one-time dump of triples deleted through March 2013"},
       "lines_read": lines,
       "unresolved_mediator_mids": len(unres),
       "already_had_a_path_from_frozen_keys": len(have_path),
       "still_needing_a_path": len(need),
       "rows_touching_an_unresolved_mediator": hits,
       "distinct_mids_touched": len(rows),
       "predicates_seen": dict(pred.most_common(30)),
       "mids_with_a_key_derived_path": len(paths),
       "NEWLY_RESOLVED": newly,
       "newly_resolved_count": len(newly),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_MEDIATOR_RESOLUTION_STAGE3.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\nwrote V3_MEDIATOR_RESOLUTION_STAGE3.json  ({time.time()-t0:.0f}s)")
