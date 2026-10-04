"""Cascade step 3: resolve QIDs to labels -- ONLY for the MIDs that Samsung actually hit.

    PYTHONHASHSEED=0 python .../samsung_qid_labels.py

The point of doing it this way round is cost. A full Wikidata label dump is tens of GB; the
intersection it would be used for is 6,245 items. So SAMSUNG_HIT_N is measured first (it was), and
only those QIDs are fetched, 50 at a time, from the live API.

TWO PASSES, BECAUSE ASKING FOR EVERY LANGUAGE IS WASTEFUL AND ASKING FOR ONE IS LOSSY
  Pass 1 requests languages=en, which keeps payloads tiny. Pass 2 re-requests only the items that
  came back without an English label, this time with no language filter, and picks a fallback
  deterministically: en, then mul, then the lexicographically smallest language code. So an item
  labelled only in Japanese still yields a name, and the choice is reproducible.

WHAT A RECOVERED NAME HERE IS
  A current Wikidata label for an item that Samsung asserted is the same thing as this MID. It is
  NOT the original /type/object/name value -- Wikidata may have renamed the item since 2014, and
  the sameAs assertion is Samsung's, not Google's. Provenance is SAMSUNG_WIKIDATA_EXACT and
  is_original_name is false.

Reads the frozen graph and the frozen overlay. Writes neither.
"""
import sys, io, os, json, time, urllib.request, urllib.parse, urllib.error
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
HITS = f"{V3}/_acquisition/samsung_hits.parquet"
API = "https://www.wikidata.org/w/api.php"
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
BATCH = 50
t0 = time.time()

t = pq.read_table(HITS)
uids = t["node_uid"].to_pylist()
nids = t["node_id"].to_pylist()
qids = t["qid"].to_pylist()
kinds = t["kind"].cast(pa.string()).to_pylist()
uniq = sorted(set(qids), key=lambda q: int(q[1:]) if q[1:].isdigit() else 0)
print(f"samsung hits: {len(qids):,} rows, {len(uniq):,} distinct QIDs", flush=True)


def fetch(ids, lang_filter):
    q = {"action": "wbgetentities", "ids": "|".join(ids), "props": "labels", "format": "json"}
    if lang_filter:
        q["languages"] = lang_filter
    url = API + "?" + urllib.parse.urlencode(q)
    for attempt in range(5):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(r, timeout=60) as f:
                return json.load(f).get("entities", {})
        except Exception as e:
            if attempt == 4:
                print(f"  give up on a batch after 5 tries: {e}", flush=True)
                return {}
            time.sleep(2 ** attempt)
    return {}


lab = {}
missing = set()
no_en = []
for i in range(0, len(uniq), BATCH):
    ch = uniq[i:i + BATCH]
    ent = fetch(ch, "en")
    for q in ch:
        e = ent.get(q)
        if e is None or "missing" in e:
            missing.add(q)
            continue
        v = (e.get("labels") or {}).get("en", {}).get("value")
        if v:
            lab[q] = (v, "en")
        else:
            no_en.append(q)
    if (i // BATCH) % 25 == 0:
        print(f"  pass1 {i+len(ch):,}/{len(uniq):,} labelled={len(lab):,} "
              f"no_en={len(no_en):,} missing={len(missing):,} ({time.time()-t0:.0f}s)", flush=True)
    time.sleep(0.1)
print(f"pass1 done: labelled={len(lab):,} no_en={len(no_en):,} missing={len(missing):,}"
      f"  ({time.time()-t0:.0f}s)", flush=True)

for i in range(0, len(no_en), BATCH):
    ch = no_en[i:i + BATCH]
    ent = fetch(ch, None)
    for q in ch:
        e = ent.get(q)
        if e is None or "missing" in e:
            missing.add(q)
            continue
        labs = e.get("labels") or {}
        if not labs:
            continue
        # deterministic fallback: en, then mul, then smallest language code
        for pref in ("en", "mul"):
            if pref in labs:
                lab[q] = (labs[pref]["value"], pref)
                break
        else:
            k = sorted(labs)[0]
            lab[q] = (labs[k]["value"], k)
    time.sleep(0.1)
print(f"pass2 done: labelled={len(lab):,} missing={len(missing):,}  ({time.time()-t0:.0f}s)",
      flush=True)

rows_u, rows_n, rows_q, rows_l, rows_lang, rows_k = [], [], [], [], [], []
for u, n, q, kd in zip(uids, nids, qids, kinds):
    if q in lab:
        rows_u.append(u); rows_n.append(n); rows_q.append(q)
        rows_l.append(lab[q][0]); rows_lang.append(lab[q][1]); rows_k.append(kd)
if rows_u:
    pq.write_table(pa.table({
        "node_uid": pa.array(rows_u, pa.int64()),
        "node_id": pa.array(rows_n),
        "qid": pa.array(rows_q),
        "display_name": pa.array(rows_l),
        "label_lang": pa.array(rows_lang).dictionary_encode(),
        "kind": pa.array(rows_k).dictionary_encode()}),
        f"{V3}/_acquisition/samsung_names.parquet", compression="zstd")

by_lang = {}
for _, lg in lab.values():
    by_lang[lg] = by_lang.get(lg, 0) + 1
rec = {"schema": "SAMSUNG_QID_LABELS/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. 301,977,131 nodes / "
                       "2,062,430,072 edges unchanged; RESOLUTION_OVERLAY_V1 manifest hash "
                       "25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865 unchanged."),
       "METHOD": ("labels fetched from the live Wikidata API for the 6,245 intersecting QIDs only. "
                  "No bulk label dump was downloaded: the intersection was measured first, and it "
                  "is small enough that 125 batched API calls are cheaper than any dump."),
       "distinct_qids_requested": len(uniq),
       "qids_labelled": len(lab),
       "qids_missing_or_deleted": len(missing),
       "nodes_named": len(rows_u),
       "labels_by_language": dict(sorted(by_lang.items(), key=lambda x: -x[1])[:15]),
       "FALLBACK_RULE": ("en, then mul, then the lexicographically smallest language code -- "
                         "deterministic, so the artifact is reproducible."),
       "PROVENANCE": "SAMSUNG_WIKIDATA_EXACT",
       "IS_ORIGINAL_NAME": False,
       "WHY_NOT_ORIGINAL": ("this is a CURRENT Wikidata label for an item Samsung asserted is the "
                            "same thing as this MID. Wikidata may have renamed the item since 2014, "
                            "and the sameAs assertion is Samsung's, not Google's. It is an attested "
                            "human-readable identity, not a recovered /type/object/name value."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_SAMSUNG_QID_LABELS.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\nnodes named from Samsung: {len(rows_u):,}")
print(f"  qids labelled {len(lab):,} / {len(uniq):,}   missing {len(missing):,}")
print(f"  languages: {rec['labels_by_language']}")
print(f"{time.time()-t0:.0f}s")
