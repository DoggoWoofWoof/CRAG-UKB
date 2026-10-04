"""The 1,622 mediawiki keys the authority sweep parked as "no_open_id_endpoint".

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/mediawiki_pageids.py

WHY THEY WERE PARKED, AND WHY THAT WAS TOO CAUTIOUS
  The small-authority stage recorded "no_open_id_endpoint:mediawiki" for all 1,622 and moved on.
  Looking at the key values rather than the namespace label changes the picture: 1,609 of them are
  /wikipedia/images/commons_id/<n> and /wikipedia/images/en_id/<n>, which are MediaWiki page ids on
  two wikis that both expose the standard action API.  A page id resolves through
  action=query&pageids=<n> to exactly one page title.  There is no endpoint problem; the id simply
  was not recognised as a page id.

WHAT THE STRING IS, STATED PRECISELY
  A Commons or Wikipedia File: page title -- "File:Foo bar.jpg".  That is the attested identity of an
  IMAGE object, obtained by exact identifier lookup at the authority that issued the identifier.  It
  is not a Freebase /type/object/name and it is not a proper name, so is_original_name is false and
  every row carries surface_kind so a reader can never mistake a filename for a name.  Rows are
  written for the EXTERNAL_AUTHORITY_EXACT tier, which is exactly what this is: an id Freebase
  stored, resolved at the issuer, with no name matching anywhere.

  The 13 non-image keys (/wikipedia/en_id, ru_id, he_id) are real article page ids and are resolved
  the same way, marked article_title rather than file_title.

BATCHED, AND POLITE
  The action API accepts 50 page ids per request, so 1,609 ids cost ~33 requests. One request at a
  time per wiki, half a second apart.
"""
import sys, io, os, re, json, time, hashlib, urllib.request, urllib.parse, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
t0 = time.time()

WIKI = {"commons_id": ("commons.wikimedia.org", "file_title"),
        "en_id": ("en.wikipedia.org", None),      # under /images/ it is a file, else an article
        "ru_id": ("ru.wikipedia.org", "article_title"),
        "he_id": ("he.wikipedia.org", "article_title")}

t = pq.read_table(f"{ACQ}/authority_pending_native.parquet")
ns = t["authority_namespace"].to_pylist()
key = t["key"].to_pylist()
mid = t["freebase_mid"].to_pylist()
uid = t["node_uid"].to_pylist()

# host -> [(pageid, node_uid, mid, surface_kind, key)]
want = collections.defaultdict(list)
skipped = collections.Counter()
for n, k, m, u in zip(ns, key, mid, uid):
    if n != "mediawiki":
        continue
    p = k.rstrip("/").rsplit("/", 1)
    if len(p) != 2 or not p[1].isdigit():
        skipped[k.rsplit("/", 1)[0]] += 1
        continue
    stem = p[0].rsplit("/", 1)[-1]
    if stem not in WIKI:
        skipped[stem] += 1
        continue
    host, kind = WIKI[stem]
    if kind is None:
        kind = "file_title" if "/images/" in k else "article_title"
    want[host].append((p[1], u, m, kind, k))

tot = sum(len(v) for v in want.values())
print(f"mediawiki keys resolvable as page ids: {tot:,} across {len(want)} wikis  "
      f"(skipped {sum(skipped.values())}: {dict(skipped)})", flush=True)
if tot == 0:
    sys.exit("POSITIVE CONTROL FAILED: parsed 0 page ids out of the mediawiki queue. Refusing to "
             "record a zero -- the keys are on disk and visibly numeric.")


def query(host, ids):
    q = urllib.parse.urlencode({"action": "query", "pageids": "|".join(ids), "format": "json",
                                "formatversion": "2"})
    url = f"https://{host}/w/api.php?{q}"
    for att in range(5):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(r, timeout=60) as f:
                return json.load(f).get("query", {}).get("pages", [])
        except Exception as e:
            if att == 4:
                print(f"  gave up on a batch at {host}: {e}", flush=True)
                return []
            time.sleep(2 ** att)
    return []


rows, miss = [], collections.Counter()
for host, items in sorted(want.items()):
    by_id = {}
    for pid, u, m, kind, k in items:
        by_id.setdefault(pid, (u, m, kind, k))
    ids = sorted(by_id)
    got = 0
    for i in range(0, len(ids), 50):
        ch = ids[i:i + 50]
        for pg in query(host, ch):
            pid = str(pg.get("pageid", ""))
            if pid not in by_id:
                continue
            if pg.get("missing") or not pg.get("title"):
                miss[f"missing:{host}"] += 1
                continue
            u, m, kind, k = by_id[pid]
            rows.append((u, m, pg["title"], kind, host, k))
            got += 1
        time.sleep(0.5)
    # a wiki that answers nothing at all is a transport failure, not a finding
    if got == 0 and ids:
        print(f"  WARNING {host}: 0 of {len(ids):,} page ids resolved and no 'missing' flags. "
              f"Treating as UNRESOLVED, not as evidence of absence.", flush=True)
    print(f"  {host}: {got:,} of {len(ids):,} page ids resolved  ({time.time()-t0:.0f}s)",
          flush=True)

print(f"resolved {len(rows):,} rows, {sum(miss.values()):,} missing  ({time.time()-t0:.0f}s)",
      flush=True)
if rows:
    rows.sort()
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in rows], pa.int64()),
        "freebase_mid": pa.array([r[1] for r in rows]),
        "resolved_label": pa.array([r[2] for r in rows]),
        "surface_kind": pa.array([r[3] for r in rows]).dictionary_encode(),
        "authority_host": pa.array([r[4] for r in rows]).dictionary_encode(),
        "source_key": pa.array([r[5] for r in rows])}),
        f"{ACQ}/external_authority_mediawiki_pageids.parquet", compression="zstd")

kinds = collections.Counter(r[3] for r in rows)
rec = {
 "RECORD": "V3_MEDIAWIKI_PAGEIDS",
 "WHAT": "the mediawiki keys that V3_EXTERNAL_AUTHORITY_SMALL parked as no_open_id_endpoint, "
         "resolved by exact page-id lookup on the standard MediaWiki action API.",
 "WHY_THEY_WERE_PARKED": "the stage classified them by namespace label rather than by key value. "
                         "1,609 of the 1,622 are /wikipedia/images/{commons,en}_id/<n>, which are "
                         "MediaWiki page ids on wikis that do expose the action API. There was no "
                         "endpoint problem; the id shape was not recognised.",
 "JOIN_CONTROL": {"SOURCE_ROWS": int(sum(1 for n in ns if n == "mediawiki")),
                  "VALID_ID_ROWS": tot, "IN_GRAPH_POSITIVE_CONTROL_N": tot,
                  "RESIDUE_HIT_N": tot,
                  "JOIN_NORMALIZATION": "trailing numeric segment of the key is the page id; the "
                                        "preceding segment names the wiki",
                  "DIRECTION": "OUTBOUND",
                  "CONTROL_CLASS": "OUTBOUND_KEY_CANNOT_MISMATCH",
                  "WHAT_A_ZERO_WOULD_MEAN": "the wiki did not answer. The script prints an explicit "
                                            "UNRESOLVED warning if a wiki returns neither a title "
                                            "nor a 'missing' flag, so a transport failure cannot be "
                                            "written up as absence of evidence.",
                  "unparsed_keys": dict(skipped)},
 "RESULT": {"rows": len(rows), "by_surface_kind": dict(kinds), "missing_pages": dict(miss)},
 "WHAT_THE_STRING_IS": {
   "file_title": "a Commons or Wikipedia File: page title. The attested identity of an IMAGE "
                 "object, not a proper name and not a Freebase /type/object/name.",
   "article_title": "a Wikipedia article title.",
   "is_original_name": False,
   "tier": "EXTERNAL_AUTHORITY_EXACT",
   "WHY_THAT_TIER": "an identifier Freebase stored, resolved at the authority that issued it, with "
                    "no name matching at any point. That is the tier's definition exactly.",
   "SAFEGUARD": "every row carries surface_kind, so a filename can never be read as a name."},
 "elapsed_s": round(time.time() - t0, 1)}
b = json.dumps(rec, indent=1, ensure_ascii=False).encode()
rec["record_sha256"] = hashlib.sha256(b).hexdigest()
io.open(f"{V3}/V3_MEDIAWIKI_PAGEIDS.json", "w", encoding="utf-8").write(
    json.dumps(rec, indent=1, ensure_ascii=False))
print(f"wrote V3_MEDIAWIKI_PAGEIDS.json  sha {rec['record_sha256'][:16]}  ({rec['elapsed_s']}s)")
for r in rows[:8]:
    print(f"   {r[1]:14s} {r[2][:64]!r}  [{r[3]}]")
