"""Is FREEBASE_CURRENT_NAME empty BY CONSTRUCTION, or empty by OMISSION?

    PYTHONHASHSEED=0 python .../current_name_gate.py

THE QUESTION
  FREEBASE_CURRENT_NAME sits at the TOP of the locked provenance order and the cascade prints it as
  "+0 (no source yet)".  That reads as pending work, but the 2015 snapshot IS the source of record,
  so the tier should be empty by construction: the residue is the set of nodes the snapshot does not
  name.  "Should be" is not a count, and the campaign rule is that every artifact is invalid until
  counts establish otherwise.

WHY IT IS WORTH A PASS RATHER THAN AN ASSUMPTION
  name.parquet holds 68,362,456 rows in 240 languages -- 44,997,002 English and 23,365,454 NOT.  If
  the residue had been computed as "no ENGLISH name" rather than "no name in any language", then
  every residue node carrying, say, only a German or Japanese /type/object/name would be sitting in
  the hunt with its ORIGINAL name already on local disk, in the top tier, at zero acquisition cost.
  That would be the largest single finding of this campaign.  If instead the count is zero, the
  residue definition is confirmed airtight against the frozen metadata and the top tier is honestly
  closed rather than merely unattempted.  Either answer is worth having; only one of them is free.

WHAT IS COUNTED
  Every subject in name.parquet and alias.parquet is hashed to a node_uid and looked up in the
  69,777,967-node residue.  Hits are reported whole: by language, by hunt membership, with samples.
  A hit in name.parquet is a FREEBASE_CURRENT_NAME (is_original_name true, is_current_snapshot_name
  TRUE -- the only tier that can say that).  A hit in alias.parquet is a FREEBASE_ALIAS.

  node_uid = hash(node_id) under PYTHONHASHSEED=0; the subject column is the node_id STRING, exactly
  as in key.parquet, so it is hashed per row group.

NOTHING IS APPLIED.  Read-only over the frozen graph; a new record with its own hash.

OUTPUT
  _acquisition/current_name_hits/part_*.parquet   only if the count is non-zero
  V3_CURRENT_NAME_GATE.json
"""
import sys, io, os, json, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
OUT = f"{ACQ}/current_name_hits"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

U = np.load(f"{ACQ}/_unresolved_population.npz")["U"]
sk = pq.read_table(f"{ACQ}/semantic_kind_v2_1.parquet", columns=["node_uid", "recovery_class"])
rc = sk["recovery_class"]
rc = pc.cast(rc, pa.string()) if pa.types.is_dictionary(rc.type) else rc
HUNTC = pa.array(["LIKELY_REAL_ENTITY", "UNTYPED_CANDIDATE", "LIKELY_PLACE", "LIKELY_MEDIA_ENTITY",
                  "LIKELY_CREATIVE_WORK", "LIKELY_PERSON"])
H = np.sort(pc.filter(sk["node_uid"], pc.is_in(rc, value_set=HUNTC)).to_numpy())
del sk, rc
hunt_mask = np.zeros(len(U), bool)
p = np.clip(np.searchsorted(U, H), 0, len(U) - 1)
hunt_mask[p[U[p] == H]] = True
print(f"residue {len(U):,}  hunt {len(H):,}  ({time.time()-t0:.0f}s)", flush=True)

report = {}
part = 0
for tag, fname in (("name", "name.parquet"), ("alias", "alias.parquet")):
    pf = pq.ParquetFile(f"{V3}/canonical/metadata/{fname}")
    NRG = pf.metadata.num_row_groups
    hits_by_lang = collections.Counter()
    n_rows = n_hit = n_hunt = 0
    hit_nodes = []
    samples = []
    buf = {"uid": [], "lex": [], "lang": [], "hunt": []}
    for g in range(NRG):
        t = pf.read_row_group(g, columns=["subject", "lexical", "lang"])
        n_rows += t.num_rows
        sub = t["subject"]
        sub = pc.cast(sub, pa.string()) if pa.types.is_dictionary(sub.type) else sub
        sub = sub.to_pylist()
        subj = np.fromiter((hash(x) for x in sub), np.int64, t.num_rows)
        pos = np.clip(np.searchsorted(U, subj), 0, len(U) - 1)
        hit = U[pos] == subj
        k = int(hit.sum())
        if k:
            n_hit += k
            lex = t["lexical"]
            lex = pc.cast(lex, pa.string()) if pa.types.is_dictionary(lex.type) else lex
            lg = t["lang"]
            lg = pc.cast(lg, pa.string()) if pa.types.is_dictionary(lg.type) else lg
            lex = lex.to_pylist(); lg = lg.to_pylist()
            for i in np.flatnonzero(hit).tolist():
                inh = bool(hunt_mask[pos[i]])
                hits_by_lang[lg[i]] += 1
                n_hunt += inh
                hit_nodes.append(int(subj[i]))
                buf["uid"].append(int(subj[i])); buf["lex"].append(lex[i][:400])
                buf["lang"].append(lg[i]); buf["hunt"].append(inh)
                if len(samples) < 25:
                    samples.append({"node_id": sub[i], "lexical": lex[i][:120],
                                    "lang": lg[i], "in_hunt": inh})
            if len(buf["uid"]) >= 400000:
                pq.write_table(pa.table({"node_uid": pa.array(buf["uid"], pa.int64()),
                                         "display_name": pa.array(buf["lex"]),
                                         "language": pa.array(buf["lang"]),
                                         "in_hunt": pa.array(buf["hunt"], pa.bool_()),
                                         "source": pa.array(["FREEBASE_CURRENT_NAME"
                                                             if tag == "name" else
                                                             "FREEBASE_ALIAS"] * len(buf["uid"]))}),
                               f"{OUT}/{tag}_part_{part:05d}.parquet", compression="zstd")
                part += 1
                buf = {"uid": [], "lex": [], "lang": [], "hunt": []}
        if g % 25 == 0:
            print(f"  {tag} rg {g}/{NRG}  rows {n_rows:,}  residue hits {n_hit:,}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if buf["uid"]:
        pq.write_table(pa.table({"node_uid": pa.array(buf["uid"], pa.int64()),
                                 "display_name": pa.array(buf["lex"]),
                                 "language": pa.array(buf["lang"]),
                                 "in_hunt": pa.array(buf["hunt"], pa.bool_()),
                                 "source": pa.array(["FREEBASE_CURRENT_NAME" if tag == "name"
                                                     else "FREEBASE_ALIAS"] * len(buf["uid"]))}),
                       f"{OUT}/{tag}_part_{part:05d}.parquet", compression="zstd")
        part += 1
    report[tag] = {"file": fname, "rows_scanned": n_rows,
                   "rows_whose_subject_is_an_UNNAMED_RESIDUE_node": n_hit,
                   "distinct_such_nodes": int(len(np.unique(np.array(hit_nodes, np.int64)))
                                              if hit_nodes else 0),
                   "of_those_rows_in_the_hunt": n_hunt,
                   "by_language": dict(hits_by_lang.most_common(40)),
                   "samples": samples}
    print(f"{tag}: {n_rows:,} rows, {n_hit:,} on residue nodes ({time.time()-t0:.0f}s)", flush=True)

# A whitespace-only /type/object/name is not a name. Counting one as a hit once flipped this
# verdict to "FREE NAMES FOUND" on the strength of a single space, so the test strips first.
nh = report["name"]["rows_whose_subject_is_an_UNNAMED_RESIDUE_node"]
nblank = sum(1 for smp in report["name"]["samples"] if not smp["lexical"].strip())
verdict = ("EMPTY_BY_CONSTRUCTION" if nh - nblank <= 0 else "EMPTY_BY_OMISSION -- NAMES FOUND")
rec = {"schema": "CURRENT_NAME_GATE/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "read-only over the frozen graph; new record only, nothing reclassified.",
       "QUESTION": "is the FREEBASE_CURRENT_NAME tier empty because the snapshot genuinely does not "
                   "name these nodes, or because the residue was computed against English only?",
       "VERDICT": verdict,
       "WHY_IT_MATTERED": "name.parquet holds 68,362,456 rows in 240 languages, 23,365,454 of them "
                          "NOT English. An English-only residue definition would have left original "
                          "non-English names sitting unclaimed in the top tier at zero cost.",
       "residue_population": int(len(U)),
       "hunt_population": int(len(H)),
       "results": report,
       "IF_NONZERO_THESE_ARE": "FREEBASE_CURRENT_NAME: is_original_name true AND "
                               "is_current_snapshot_name TRUE -- the only tier that can assert both.",
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_CURRENT_NAME_GATE.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in rec.items() if k != "results"}, indent=1, ensure_ascii=False))
for tag in ("name", "alias"):
    r = report[tag]
    print(f"\n{tag}: {r['rows_whose_subject_is_an_UNNAMED_RESIDUE_node']:,} residue rows / "
          f"{r['distinct_such_nodes']:,} distinct nodes / {r['of_those_rows_in_the_hunt']:,} in hunt")
    if r["by_language"]:
        print("  by language:", dict(list(r["by_language"].items())[:12]))
    for smp in r["samples"][:10]:
        print(f"    {smp['node_id']:<16} {smp['lexical'][:60]!r:<62} [{smp['lang']}] "
              f"hunt={smp['in_hunt']}")
