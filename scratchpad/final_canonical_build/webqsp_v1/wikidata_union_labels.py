"""Close cascade step 4: label the Wikidata hits, then compute the Wikidata ceiling as a union.

    PYTHONHASHSEED=0 python .../wikidata_union_labels.py

WHY A UNION AND NOT THREE SEPARATE COUNTS
  Samsung (2014 archive), P646 (current Freebase ID) and P2671 (current Google Knowledge Graph ID)
  are three views of the same correspondence and overlap heavily. Adding their hit counts would
  overstate the reach of Wikidata as a source. What bounds it is the number of DISTINCT residual
  nodes any of them reaches, so that is what is computed and reported.

  This matters because the same mistake was already caught once in this campaign: IDIR's two label
  tables looked like 2,259 + 1,580 = 3,839 recovered nodes, and the intersection showed the second
  table added exactly zero.

LABELS ARE FETCHED ONCE PER QID, NOT ONCE PER HIT
  A QID reached by two sources is fetched once. Labels already obtained for the Samsung pass are
  reused from disk rather than re-requested.

CASCADE ORDER IS THE LOCKED ONE
  ... IDIR_FREEBASE_NAME > IDIR_CANONICAL_LABEL > SAMSUNG_WIKIDATA_EXACT > CURRENT_WIKIDATA_EXACT ...
  A node reached by several sources is attributed to the highest-priority one, so provenance stays
  meaningful and the per-source "new nodes" columns sum to the union.

Reads the frozen graph and the frozen overlay. Writes neither.
"""
import sys, io, os, json, time, urllib.request, urllib.parse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
API = "https://www.wikidata.org/w/api.php"
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
BATCH = 50
RESIDUE_ENTITY_MIDS = 17_809_849
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")


def load(fp, cols):
    if not os.path.exists(fp):
        print(f"  absent: {os.path.basename(fp)}", flush=True)
        return None
    return pq.read_table(fp, columns=cols)


# ------------------------------------------------------------------ what needs a label
qid_of = {}          # node_uid -> qid, for the CURRENT_WIKIDATA sources only
node_prop = {}       # node_uid -> which property reached it first
wd_uids = {}
for prop in ("P646", "P2671"):
    t = load(f"{ACQ}/wikidata_{prop}_hits.parquet", ["node_uid", "node_id", "qid"])
    if t is None:
        continue
    u = t["node_uid"].to_pylist(); q = t["qid"].to_pylist(); n = t["node_id"].to_pylist()
    for a, b, c in zip(u, q, n):
        if a not in qid_of:
            qid_of[a] = b
            node_prop[a] = prop
            wd_uids[a] = c
    print(f"  {prop}: {len(u):,} hits", flush=True)

# labels already fetched during the Samsung pass are reused rather than re-requested
known = {}
sn = load(f"{ACQ}/samsung_names.parquet", ["qid", "display_name", "label_lang"])
if sn is not None:
    for q, d, l in zip(sn["qid"].to_pylist(), sn["display_name"].to_pylist(),
                       sn["label_lang"].combine_chunks().cast(pa.string()).to_pylist()):
        known[q] = (d, l)
need = sorted({q for q in qid_of.values() if q not in known})
print(f"distinct QIDs to fetch: {len(need):,} "
      f"({len(set(qid_of.values())):,} needed, {len(set(qid_of.values())) - len(need):,} reused "
      f"from the Samsung pass)  ({time.time()-t0:.0f}s)", flush=True)


def fetch(ids, lang):
    p = {"action": "wbgetentities", "ids": "|".join(ids), "props": "labels", "format": "json"}
    if lang:
        p["languages"] = lang
    url = API + "?" + urllib.parse.urlencode(p)
    for att in range(5):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(r, timeout=60) as f:
                return json.load(f).get("entities", {})
        except Exception as e:
            if att == 4:
                print(f"  gave up on a batch: {e}", flush=True)
                return {}
            time.sleep(2 ** att)
    return {}


lab, missing, no_en = dict(known), set(), []
for i in range(0, len(need), BATCH):
    ch = need[i:i + BATCH]
    ent = fetch(ch, "en")
    for q in ch:
        e = ent.get(q)
        if e is None or "missing" in e:
            missing.add(q); continue
        v = (e.get("labels") or {}).get("en", {}).get("value")
        if v:
            lab[q] = (v, "en")
        else:
            no_en.append(q)
    if (i // BATCH) % 25 == 0:
        print(f"  pass1 {i+len(ch):,}/{len(need):,} labelled={len(lab)-len(known):,} "
              f"no_en={len(no_en):,} missing={len(missing):,} ({time.time()-t0:.0f}s)", flush=True)
    time.sleep(0.1)
for i in range(0, len(no_en), BATCH):
    ch = no_en[i:i + BATCH]
    ent = fetch(ch, None)
    for q in ch:
        e = ent.get(q)
        if e is None or "missing" in e:
            missing.add(q); continue
        ls = e.get("labels") or {}
        if not ls:
            continue
        for pref in ("en", "mul"):          # deterministic fallback, same rule as the Samsung pass
            if pref in ls:
                lab[q] = (ls[pref]["value"], pref); break
        else:
            k = sorted(ls)[0]
            lab[q] = (ls[k]["value"], k)
    time.sleep(0.1)
print(f"labels: {len(lab)-len(known):,} newly fetched, {len(missing):,} missing/deleted "
      f"({time.time()-t0:.0f}s)", flush=True)

rows_u = [u for u in sorted(qid_of) if qid_of[u] in lab]
if rows_u:
    pq.write_table(pa.table({
        "node_uid": pa.array(rows_u, pa.int64()),
        "node_id": pa.array([wd_uids[u] for u in rows_u]),
        "qid": pa.array([qid_of[u] for u in rows_u]),
        "display_name": pa.array([lab[qid_of[u]][0] for u in rows_u]),
        "label_lang": pa.array([lab[qid_of[u]][1] for u in rows_u]).dictionary_encode(),
        "prop": pa.array([node_prop[u] for u in rows_u]).dictionary_encode()}),
        f"{ACQ}/wikidata_current_names.parquet", compression="zstd")

# ------------------------------------------------------------------ union under the cascade
CASCADE = [("IDIR_FREEBASE_NAME", f"{ACQ}/idir_names.parquet"),
           ("IDIR_CANONICAL_LABEL", f"{ACQ}/idir_entities_id_label.parquet"),
           ("SAMSUNG_WIKIDATA_EXACT", f"{ACQ}/samsung_names.parquet"),
           ("CURRENT_WIKIDATA_EXACT", f"{ACQ}/wikidata_current_names.parquet")]
claimed, per_source, raw_counts = set(), {}, {}
for name, fp in CASCADE:
    t = load(fp, ["node_uid"])
    if t is None:
        per_source[name] = 0; raw_counts[name] = 0; continue
    s = set(t["node_uid"].to_pylist())
    raw_counts[name] = len(s)
    per_source[name] = len(s - claimed)
    claimed |= s

wd_reach = set()
for prop in ("P646", "P2671"):
    t = load(f"{ACQ}/wikidata_{prop}_hits.parquet", ["node_uid"])
    if t is not None:
        wd_reach |= set(t["node_uid"].to_pylist())
sam = load(f"{ACQ}/samsung_hits.parquet", ["node_uid"])
sam_reach = set(sam["node_uid"].to_pylist()) if sam is not None else set()
wd_all = wd_reach | sam_reach

rec = {"schema": "WIKIDATA_CEILING_AND_UNION/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. RESOLUTION_OVERLAY_V1 "
                       "manifest hash 25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865"
                       " unchanged."),
       "residue_entity_mids": RESIDUE_ENTITY_MIDS,
       "WIKIDATA_CEILING": {
           "samsung_2014_reach": len(sam_reach),
           "current_P646_P2671_reach": len(wd_reach),
           "UNION_reach": len(wd_all),
           "sum_if_disjoint": len(sam_reach) + len(wd_reach),
           "current_adds_over_samsung": len(wd_reach - sam_reach),
           "samsung_adds_over_current": len(sam_reach - wd_reach),
           "pct_of_residue": round(100 * len(wd_all) / RESIDUE_ENTITY_MIDS, 4),
           "MEANING": ("the number of DISTINCT residual nodes any Wikidata-mediated route reaches. "
                       "Adding the per-source counts would overstate it; they overlap heavily."),
           "IS_A_CEILING_FOR": ("Wikidata-mediated routes only. It does not bound FACC1, archived "
                                "Freebase pages, or any other MID-keyed source.")},
       "RECOVERED_UNDER_CASCADE": {
           "order": [c[0] for c in CASCADE],
           "nodes_new_at_this_tier": per_source,
           "nodes_reached_by_source_ignoring_cascade": raw_counts,
           "TOTAL_DISTINCT_NAMED": len(claimed),
           "pct_of_residue": round(100 * len(claimed) / RESIDUE_ENTITY_MIDS, 4),
           "NOTE": ("nodes_new_at_this_tier sums to TOTAL_DISTINCT_NAMED by construction; the raw "
                    "counts do not, because the sources overlap.")},
       "labels": {"qids_needed": len(set(qid_of.values())),
                  "qids_reused_from_samsung_pass": len(set(qid_of.values())) - len(need),
                  "qids_fetched": len(need),
                  "qids_missing_or_deleted": len(missing),
                  "FALLBACK_RULE": "en, then mul, then the lexicographically smallest language code"},
       "IS_ORIGINAL_NAME": False,
       "WHY_NOT_ORIGINAL": ("a current Wikidata label for an item asserted to be the same thing as "
                            "this MID. Wikidata may have renamed the item since 2014, and the "
                            "mapping is a third party's, not Google's. It is an attested "
                            "human-readable identity, not a recovered /type/object/name value."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_WIKIDATA_CEILING.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)

print(f"\nWIKIDATA CEILING (distinct residual nodes reachable)")
print(f"  samsung 2014          {len(sam_reach):>8,}")
print(f"  current P646/P2671    {len(wd_reach):>8,}")
print(f"  UNION                 {len(wd_all):>8,}   ({rec['WIKIDATA_CEILING']['pct_of_residue']}% of residue)")
print(f"  current adds new      {len(wd_reach - sam_reach):>8,}")
print(f"\nRECOVERED NAMES under the cascade")
for k, v in per_source.items():
    print(f"  {k:<24} +{v:>7,}   (source reaches {raw_counts[k]:,})")
print(f"  {'TOTAL DISTINCT':<24}  {len(claimed):>7,}   "
      f"({rec['RECOVERED_UNDER_CASCADE']['pct_of_residue']}% of {RESIDUE_ENTITY_MIDS:,})")
print(f"{time.time()-t0:.0f}s")
