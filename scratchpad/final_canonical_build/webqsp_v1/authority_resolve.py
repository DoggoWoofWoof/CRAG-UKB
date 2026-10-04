"""EXTERNAL_AUTHORITY_EXACT: resolve the residue's authority ids by exact identifier lookup only.

    PYTHONHASHSEED=0 python .../authority_resolve.py

THE RULE FOR THIS TIER
  Exact identifier lookup, and nothing else. No search-by-name, no fuzzy matching, no "closest
  title". An identifier either resolves or it does not; a near miss is a miss. That is what makes
  this tier as strong a recovery as we can get, and it is why a low yield here is a fact about the
  identifiers rather than a reason to loosen the rule.

THREE ROUTES, IN THIS ORDER, ALL OF THEM EXACT
  1. DECODE      Some keys already contain the answer. /wikipedia/en/Sky_99.5 and
                 /wikipedia/ja_title/易培基 are page titles held in the key itself; recovering them
                 is an unescape, not a lookup, and it needs no network at all.
  2. WIKIDATA    Wikidata carries an exact-identifier property for most of these authorities
                 (P434 MusicBrainz artist, P345 IMDb, P648 Open Library, P244 LCCN, ...). One
                 QLever query per chunk with a VALUES clause matches our ids against those
                 properties server-side. This is still exact identifier lookup -- Wikidata is the
                 index, not the matcher.
  3. NATIVE      Whatever Wikidata does not carry goes to the issuing authority's own id endpoint
                 (MusicBrainz /ws/2/<entity>/<mbid>, MediaWiki pageids=, Open Library). Handled by
                 a companion pass so the rate-limited work is separable from the bulk work.

WHAT IS DELIBERATELY NOT RESOLVED
  2,026 MusicBrainz keys have the shape <uuid>::<uuid>::<n>. That is an artist-credit position
  joining two entities, not an entity id, and no lookup service accepts it. They are recorded as
  INTERNAL_RECORD rather than counted as failures, because failing to resolve them is correct.

Reads the frozen graph and the frozen overlay. Writes neither.
"""
import sys, io, os, re, json, time, urllib.request, urllib.parse, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
QLEVER = "https://qlever.dev/api/wikidata"   # qlever.cs.uni-freiburg.de 308s here, and urllib will not re-POST across a 308
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
ESC = re.compile(r"\$([0-9A-Fa-f]{4})")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
CHUNK = 400
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

dec = lambda s: ESC.sub(lambda m: chr(int(m.group(1), 16)), s)

# namespace -> the Wikidata properties that hold THIS authority's identifier, exactly
PROPS = {
    "musicbrainz": ["P434", "P435", "P436", "P966", "P982", "P1004", "P1330", "P1407",
                    "P4404", "P5813", "P1004"],
    "imdb": ["P345"],
    "discogs": ["P1953", "P1954", "P2206", "P1955", "P2168"],
    "thetvdb": ["P4835"],
    "openlibrary": ["P648"],
    "us": ["P244"],                  # /authority/us/gov/loc/na/<lccn>
    "gnis": ["P590"],
    "iata": ["P238"],
    "isfdb": ["P1274", "P1234"],
    "netflix": ["P1874"],
    "giantbomb": ["P5247"],
    "rovi": ["P1728", "P1729"],
    "stanford": [],                  # SearchWorks control numbers: no Wikidata property
    "google": [], "myspace": [], "tvrage": [],
}


def ident(ns, key):
    """The bare identifier a lookup service would accept, or None if the key is not one."""
    p = key.split("/")[3:]
    if ns == "musicbrainz":
        if any("::" in x for x in p):
            return None                      # artist-credit position, not an entity id
        m = UUID.search(key)
        return m.group(0) if m else None
    if ns == "us":
        return p[-1] if p else None          # .../gov/loc/na/n84032581
    if ns in ("imdb", "discogs", "thetvdb", "openlibrary", "isfdb", "giantbomb", "rovi",
              "netflix", "stanford"):
        return p[-1] if p else None
    return p[-1] if p else None


t = pq.read_table(f"{ACQ}/residue_authority_ids.parquet")
rows = list(zip(t["node_uid"].to_pylist(), t["node_id"].to_pylist(), t["key"].to_pylist(),
                t["bucket"].combine_chunks().cast(pa.string()).to_pylist()))
print(f"authority keys: {len(rows):,} over {len({r[0] for r in rows}):,} nodes", flush=True)

out = []                     # the EXTERNAL_AUTHORITY_EXACT rows
decoded = []                 # WIKIPEDIA_TITLE rows, recovered without any lookup
deferred = collections.defaultdict(list)   # route 3: native id endpoints
skipped = collections.Counter()
by_ns = collections.defaultdict(list)

for u, nid, key, bucket in rows:
    kind, ns = bucket.split(":", 1)
    d = dec(key)
    if kind == "wikipedia":
        # /wikipedia/en/Sky_99.5 and /wikipedia/ja_title/... hold the title outright.
        # /wikipedia/en_id/46384040 and /wikipedia/images/... hold a numeric id and need MediaWiki.
        if ns.endswith("_id") or ns == "images":
            deferred["mediawiki"].append((u, nid, ns, d))
        else:
            lang = ns.replace("_title", "")
            title = d.split("/", 3)[-1].replace("_", " ").strip()
            if title:
                decoded.append((u, nid, f"wikipedia:{lang}", d.split("/", 3)[-1], title,
                                "wikipedia-article",
                                f"https://{lang}.wikipedia.org/wiki/"
                                + urllib.parse.quote(d.split('/', 3)[-1])))
        continue
    i = ident(ns, key)
    if i is None:
        skipped["musicbrainz_composite" if ns == "musicbrainz" else f"unparsable:{ns}"] += 1
        continue
    by_ns[ns].append((u, nid, i, key))

print(f"decoded without any lookup: {len(decoded):,}   "
      f"deferred to native endpoints: {sum(len(v) for v in deferred.values()):,}   "
      f"not an entity id: {dict(skipped)}", flush=True)


def qlever(q, attempts=4):
    """POST, not GET: a chunk of 400 uuids in a query string is a 414 from this endpoint."""
    for a in range(attempts):
        try:
            r = urllib.request.Request(QLEVER, data=q.encode("utf-8"),
                                       headers={"Accept": "text/tab-separated-values",
                                                "Content-Type": "application/sparql-query",
                                                "User-Agent": UA})
            with urllib.request.urlopen(r, timeout=300) as f:
                return f.read().decode("utf-8", "replace")
        except Exception as e:
            if a == attempts - 1:
                print(f"    qlever gave up: {type(e).__name__} {e}", flush=True)
                return ""
            time.sleep(2 ** a)
    return ""


Q = chr(34)
unq = lambda v: v[v.find(Q) + 1:v.rfind(Q)] if 0 <= v.find(Q) < v.rfind(Q) else v
resolved_ids = set()
for ns, items in sorted(by_ns.items()):
    props = PROPS.get(ns, [])
    if not props:
        deferred[ns].extend(items)
        continue
    uniq = sorted({i for (_, _, i, _) in items})
    pv = " ".join("wdt:" + p for p in dict.fromkeys(props))
    hit = {}
    for c in range(0, len(uniq), CHUNK):
        ch = uniq[c:c + CHUNK]
        vals = " ".join(Q + x.replace(Q, "") + Q for x in ch)
        q = ("PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
             "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
             "PREFIX wdtn: <http://www.wikidata.org/prop/direct-normalized/> "
             f"SELECT ?id ?item ?label WHERE {{ VALUES ?id {{ {vals} }} "
             f"VALUES ?p {{ {pv} }} ?item ?p ?id . "
             'OPTIONAL { ?item rdfs:label ?label . FILTER(lang(?label)="en") } }')
        d = qlever(q)
        for ln in d.split("\n")[1:]:
            f = ln.rstrip("\r").split("\t")
            if len(f) < 3 or not f[0]:
                continue
            iid, item, lab = unq(f[0]), f[1].rsplit("/", 1)[-1].rstrip(">"), unq(f[2])
            if lab and iid not in hit:
                hit[iid] = (item, lab)
        print(f"  {ns}: {min(c+CHUNK, len(uniq)):,}/{len(uniq):,} matched={len(hit):,} "
              f"({time.time()-t0:.0f}s)", flush=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for (u, nid, i, key) in items:
        if i in hit:
            qid, lab = hit[i]
            out.append((u, nid, ns, i, lab, f"wikidata:{qid}",
                        f"https://www.wikidata.org/wiki/{qid}", now))
            resolved_ids.add((u, i))
        else:
            deferred[ns].append((u, nid, i, key))

print(f"\nWikidata exact-id route resolved {len(out):,} rows  ({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ write what we have
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
allrows = [(u, n, ns, i, lab, et, su, now) for (u, n, ns, i, lab, et, su, _) in out] + \
          [(u, n, ns, i, lab, et, su, now) for (u, n, ns, i, lab, et, su) in decoded]
if allrows:
    allrows.sort()
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in allrows], pa.int64()),
        "freebase_mid": pa.array([r[1] for r in allrows]),
        "authority_namespace": pa.array([r[2] for r in allrows]).dictionary_encode(),
        "authority_id": pa.array([r[3] for r in allrows]),
        "resolved_label": pa.array([r[4] for r in allrows]),
        "resolved_entity_type": pa.array([r[5] for r in allrows]),
        "source_url": pa.array([r[6] for r in allrows]),
        "retrieved_at": pa.array([r[7] for r in allrows])}),
        f"{ACQ}/external_authority_names.parquet", compression="zstd")

pend = [(u, n, ns, i if isinstance(i, str) else i, k)
        for ns, v in deferred.items() for (u, n, i, k) in v]
if pend:
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in pend], pa.int64()),
        "freebase_mid": pa.array([r[1] for r in pend]),
        "authority_namespace": pa.array([r[2] for r in pend]).dictionary_encode(),
        "authority_id": pa.array([str(r[3]) for r in pend]),
        "key": pa.array([str(r[4]) for r in pend])}),
        f"{ACQ}/authority_pending_native.parquet", compression="zstd")

rec = {"schema": "EXTERNAL_AUTHORITY_EXACT/v1",
       "generated_utc": now,
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. RESOLUTION_OVERLAY_V1 "
                       "manifest hash 25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865"
                       " unchanged."),
       "RULE": "exact identifier lookup only; no search-by-name, no fuzzy matching at any stage",
       "input_keys": len(rows),
       "input_nodes": len({r[0] for r in rows}),
       "resolved_rows": len(allrows),
       "resolved_nodes": len({r[0] for r in allrows}),
       "by_route": {"decoded_from_key_no_lookup": len(decoded),
                    "wikidata_exact_identifier": len(out)},
       "pending_native_endpoint": len(pend),
       "pending_by_namespace": {k: len(v) for k, v in sorted(deferred.items())},
       "NOT_AN_ENTITY_ID": {"counts": dict(skipped),
                            "WHY": ("<uuid>::<uuid>::<n> is a MusicBrainz artist-credit position "
                                    "joining two entities. No lookup service accepts it because it "
                                    "does not denote an entity. These are INTERNAL_RECORD, and not "
                                    "counted as resolution failures.")},
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_EXTERNAL_AUTHORITY.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)

print(f"\nresolved {len(allrows):,} rows over {len({r[0] for r in allrows}):,} nodes")
print(f"  decoded from key (no lookup): {len(decoded):,}")
print(f"  wikidata exact identifier:    {len(out):,}")
print(f"  pending native endpoints:     {len(pend):,}  {rec['pending_by_namespace']}")
for r in allrows[:10]:
    print(f"   {r[1]:<14} {r[2]:<14} {r[3][:34]:<36} {r[4][:40]!r}")
print(f"{time.time()-t0:.0f}s")
