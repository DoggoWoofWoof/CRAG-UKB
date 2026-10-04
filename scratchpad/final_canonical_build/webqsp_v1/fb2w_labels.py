"""The fb2w route into Wikidata: size it was measured, labels it never got.

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/fb2w_labels.py

WHY THIS PASS EXISTS
  V3_TIER3_FB2W_COVERAGE measured that fb2w reaches 20,024 unnamed nodes and then stopped, with an
  explicit caveat: "a Wikidata Q-id is an identifier, not a label ... turning that into human-
  readable text needs a second source, which is a separate acquisition."  That acquisition was never
  done.  The later Wikidata label pass fetched QIDs for the P646/P2671 route ONLY, so fb2w has been
  sitting on disk as a sized-but-unconverted source ever since.

  The JOIN_CONTROL retrofit is what surfaced it: fb2w's positive control is 2,095,507 of 2,099,583
  ids in graph (99.81%) and 18,614 on the residue, against a Wikidata-mediated union of 11,707 from
  the two routes actually labelled.  A third exact route reaching more nodes than the other two
  combined is not something to leave unconverted while the exact-source sweep is still open.

WHY IT IS A ROUTE AND NOT A NEW TIER
  The order is locked by V3_NAME_PROVENANCE_ORDER_V2.  fb2w's PROVENANCE is unusually good -- Google
  published it in November 2013 and it shipped from the same Internet Archive item as the main dump
  -- but the LABEL it yields is today's Wikidata label, not a Freebase /type/object/name.  Authority
  over the mapping and authority over the string are different things, and only the second one sets
  the tier.  So these rows enter the existing CURRENT_WIKIDATA_EXACT tier with is_original_name
  false, carrying route="fb2w" so the three Wikidata routes stay separable in the evidence.
  That placement is BY INFERENCE from the locked order, not a ruling, and is recorded as such.

  The label rule is the one already used by the Samsung and P646 passes, unchanged: en, then mul,
  then the lexicographically smallest language code.  No name matching anywhere in this pass -- the
  only key is an identifier Google itself asserted.
"""
import sys, io, os, re, json, time, gzip, hashlib, urllib.request, urllib.parse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
API = "https://www.wikidata.org/w/api.php"
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
BATCH = 50
t0 = time.time()

z = np.load(f"{ACQ}/_unresolved_population.npz")
RES = z["U"]
MID = re.compile(r"^m\.[0-9a-z_]{2,}$")
QID = re.compile(r"Q[0-9]+$")

# ------------------------------------------------------------------ read the mapping
pairs, n_lines, bad = {}, 0, 0
with gzip.open(f"{ACQ}/raw/fb2w.nt.gz", "rt", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if line.startswith("#") or not line.strip():
            continue
        n_lines += 1
        f = line.split("\t")
        if len(f) < 3:
            bad += 1
            continue
        a, b = f[0].find("ns/"), f[2].find("entity/")
        if a < 0 or b < 0:
            bad += 1
            continue
        mid = f[0][a + 3:].rstrip(">").replace("/", ".")
        qid = f[2][b + 7:].split(">")[0]
        if MID.match(mid) and QID.match(qid):
            pairs.setdefault(mid, qid)
        else:
            bad += 1
print(f"fb2w: {n_lines:,} links, {len(pairs):,} distinct mids, {bad:,} unparsed "
      f"({time.time()-t0:.0f}s)", flush=True)
if not pairs:
    sys.exit("POSITIVE CONTROL FAILED: parsed 0 mid->qid pairs. Refusing to write a zero.")

mids = sorted(pairs)
uids = np.fromiter((hash(m) for m in mids), np.int64, len(mids))
order = np.argsort(uids, kind="stable")
su = uids[order]
i = np.searchsorted(RES, su)
hit = np.take(RES, np.minimum(i, len(RES) - 1)) == su
sel = order[hit]
hit_mids = [mids[k] for k in sel]
hit_uids = uids[sel]
print(f"on residue: {len(hit_mids):,} nodes  ({time.time()-t0:.0f}s)", flush=True)
if not hit_mids:
    sys.exit("POSITIVE CONTROL FAILED: 0 fb2w mids on the residue. Check the id normalisation.")

# ------------------------------------------------------------------ reuse every label on disk
known = {}
for fp, qc, dc, lc in (("samsung_names.parquet", "qid", "display_name", "label_lang"),
                       ("wikidata_current_names.parquet", "qid", "display_name", "label_lang")):
    p = f"{ACQ}/{fp}"
    if not os.path.exists(p):
        continue
    t = pq.read_table(p, columns=[qc, dc, lc])
    for q, d, l in zip(t[qc].to_pylist(), t[dc].to_pylist(),
                       t[lc].combine_chunks().cast(pa.string()).to_pylist()):
        known.setdefault(q, (d, l))
want = {pairs[m] for m in hit_mids}
need = sorted(want - set(known))
print(f"QIDs: {len(want):,} needed, {len(want)-len(need):,} already on disk, {len(need):,} to fetch",
      flush=True)


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
        for pref in ("en", "mul"):
            if pref in ls:
                lab[q] = (ls[pref]["value"], pref); break
        else:
            k = sorted(ls)[0]
            lab[q] = (ls[k]["value"], k)
    time.sleep(0.1)
newly = len(lab) - len(known)
print(f"labels: {newly:,} newly fetched, {len(missing):,} missing/deleted ({time.time()-t0:.0f}s)",
      flush=True)
if need and newly == 0 and not missing:
    sys.exit("POSITIVE CONTROL FAILED: asked for labels and got neither a label nor a 'missing' "
             "for any QID. That is an API/transport failure, not a result. Refusing to write.")

rows = [(int(u), m, pairs[m]) for u, m in zip(hit_uids, hit_mids) if pairs[m] in lab]
rows.sort()
pq.write_table(pa.table({
    "node_uid": pa.array([r[0] for r in rows], pa.int64()),
    "node_id": pa.array([r[1] for r in rows]),
    "qid": pa.array([r[2] for r in rows]),
    "display_name": pa.array([lab[r[2]][0] for r in rows]),
    "label_lang": pa.array([lab[r[2]][1] for r in rows]).dictionary_encode(),
    "route": pa.array(["fb2w"] * len(rows)).dictionary_encode()}),
    f"{ACQ}/fb2w_names.parquet", compression="zstd")

# what does this add over the routes already labelled?
prev = set()
for fp in ("samsung_names.parquet", "wikidata_current_names.parquet"):
    p = f"{ACQ}/{fp}"
    if os.path.exists(p):
        prev |= set(pq.read_table(p, columns=["node_uid"])["node_uid"].to_pylist())
new_over = len({r[0] for r in rows} - prev)

byl = {}
for r in rows:
    byl[lab[r[2]][1]] = byl.get(lab[r[2]][1], 0) + 1

rec = {
 "RECORD": "V3_FB2W_LABELS",
 "WHAT": "the fb2w Freebase->Wikidata mapping, converted from identifiers to labels. Sized in "
         "V3_TIER3_FB2W_COVERAGE 2026-09-07 and left unconverted there by an explicit caveat; the "
         "JOIN_CONTROL retrofit surfaced that it reaches more residue nodes than the two Wikidata "
         "routes that WERE labelled, combined.",
 "SOURCE": {"artifact": "raw/fb2w.nt.gz", "published_by": "Google, November 2013, CC0",
            "same_IA_item_as_the_main_dump": True},
 "JOIN_CONTROL": {"SOURCE_ROWS": n_lines, "VALID_ID_ROWS": len(pairs),
                  "IN_GRAPH_POSITIVE_CONTROL_N": 2095507, "RESIDUE_HIT_N": len(hit_mids),
                  "JOIN_NORMALIZATION": "subject IRI tail after 'ns/', '/'->'.'; object tail after "
                                        "'entity/'. fb2w is ALREADY dot-form -- assuming slash-form "
                                        "here scored a false clean zero once and is why the "
                                        "extractor now accepts both spellings.",
                  "unparsed_lines": bad,
                  "control_source": "V3_JOIN_CONTROL_RETROFIT (NON_CIRCULAR_RAW_SIDE)"},
 "LABELS": {"qids_needed": len(want), "qids_reused_from_disk": len(want) - len(need),
            "qids_fetched": newly, "qids_missing_or_deleted": len(missing),
            "FALLBACK_RULE": "en, then mul, then the lexicographically smallest language code",
            "by_language": dict(sorted(byl.items(), key=lambda kv: -kv[1])[:12])},
 "RESULT": {"rows_written": len(rows),
            "nodes_NEW_over_samsung_and_P646_P2671": new_over,
            "nodes_already_reached_by_another_wikidata_route": len(rows) - new_over},
 "TIER_PLACEMENT": {
    "tier": "CURRENT_WIKIDATA_EXACT", "is_original_name": False, "route": "fb2w",
    "REASONING": "authority over the MAPPING and authority over the STRING are different things, "
                 "and only the second sets the tier. Google published the mapping, but the label is "
                 "today's Wikidata label, not a Freebase /type/object/name. So this is a third "
                 "route into an existing tier, not a new tier, and the locked V2 order is unchanged.",
    "STATUS": "PLACED_BY_INFERENCE_NOT_RULED"},
 "NO_NAME_MATCHING": "every row was obtained by an identifier Google asserted. No fuzzy or "
                     "name-based matching occurs anywhere in this pass.",
 "elapsed_s": round(time.time() - t0, 1)}
b = json.dumps(rec, indent=1, ensure_ascii=False).encode()
rec["record_sha256"] = hashlib.sha256(b).hexdigest()
io.open(f"{V3}/V3_FB2W_LABELS.json", "w", encoding="utf-8").write(
    json.dumps(rec, indent=1, ensure_ascii=False))
print(f"\nfb2w_names.parquet: {len(rows):,} rows, {new_over:,} NEW over the labelled routes")
print(f"wrote V3_FB2W_LABELS.json  sha {rec['record_sha256'][:16]}  ({rec['elapsed_s']}s)")
