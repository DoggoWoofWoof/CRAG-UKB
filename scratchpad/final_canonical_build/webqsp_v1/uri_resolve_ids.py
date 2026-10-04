"""The 85,133 /uri/ references, part 1: the ones that carry an exact identifier.

    PYTHONHASHSEED=0 python .../uri_resolve_ids.py

WHY THIS RUNS BEFORE ANY PAGE FETCHING
  A probe of the top hosts showed the live web is largely shut to this cohort: Amazon serves a bot
  page with the title "Amazon.com", Discogs and artic.edu answer 403, and collections.sfmoma.org and
  every citysearch subdomain no longer resolve in DNS at all. Fetching pages is therefore the
  expensive and least reliable route, not the first one.

  But most of these URLs are identifiers in disguise. /exec/obidos/ASIN/B000051JTT and
  /gp/product/B000008405 are both an ASIN; discogs.com/release/91982 is a Discogs release id;
  artic.edu/.../artwork/27134 is an Art Institute artwork id. Those can be matched exactly against
  the authority properties Wikidata already indexes, in seconds, with no scraping and no guessing.

WHICH TIER THIS PRODUCES, AND WHY NOT AN URL TIER
  A row recovered here was NOT recovered by retrieving the page, so calling it
  EXTERNAL_URL_LIVE_EXACT would misdescribe it. The identifier came out of the URL and was then
  resolved exactly against the issuing authority's index, which is precisely what
  EXTERNAL_AUTHORITY_EXACT means; the host is recorded as the namespace. EXTERNAL_URL_LIVE_EXACT and
  EXTERNAL_URL_ARCHIVE_EXACT are reserved for rows actually obtained from a page, live or archived.

Reads the frozen graph and the frozen overlay. Writes neither.
"""
import sys, io, os, re, json, time, urllib.request, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
QLEVER = "https://qlever.dev/api/wikidata"
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
CHUNK = 400
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

# (name, regex on the url, wikidata properties holding that authority's identifier)
PAT = [
    ("amazon_asin",  r"amazon\.[a-z.]+/(?:.*?/)?(?:ASIN|dp|gp/product)/([0-9A-Z]{10})", ["P5749"]),
    ("discogs_rel",  r"discogs\.com/release/(\d+)",                    ["P2206"]),
    ("discogs_mas",  r"discogs\.com/master/(\d+)",                     ["P1954"]),
    ("discogs_art",  r"discogs\.com/artist/(\d+)",                     ["P1953"]),
    ("discogs_lab",  r"discogs\.com/label/(\d+)",                      ["P1955"]),
    ("artic",        r"artic\.edu/.*?/artwork/(\d+)",                  ["P4610"]),
    ("metalarch",    r"metal-archives\.com/bands/[^/]+/(\d+)",         ["P1952"]),
    ("newadvent",    r"newadvent\.org/cathen/(\w+)\.htm",              ["P3241"]),
    ("imdb_title",   r"imdb\.com/title/(tt\d+)",                       ["P345"]),
    ("imdb_name",    r"imdb\.com/name/(nm\d+)",                        ["P345"]),
    ("allmusic",     r"allmusic\.com/.*?((?:mn|mw|mc)\d{10})",         ["P1728", "P1729"]),
    ("gutenberg",    r"gutenberg\.org/.*?(?:etext|ebooks)/(\d+)",      ["P2034"]),
    ("musicbrainz",  r"musicbrainz\.org/\w+/([0-9a-f-]{36})",
                     ["P434", "P435", "P436", "P4404", "P5813", "P966"]),
    ("openlibrary",  r"openlibrary\.org/[bw]/(OL\d+[AMW])",            ["P648"]),
    ("nndb",         r"nndb\.com/people/(\d+)",                        []),
]

t = pq.read_table(f"{ACQ}/residue_uris.parquet")
recs = list(zip(t["node_uid"].to_pylist(), t["node_id"].to_pylist(), t["url"].to_pylist(),
                t["host"].combine_chunks().cast(pa.string()).to_pylist()))
print(f"urls: {len(recs):,}", flush=True)

by_pat = collections.defaultdict(list)
unmatched = []
for u, nid, url, host in recs:
    for name, rx, props in PAT:
        m = re.search(rx, url, re.I)
        if m:
            by_pat[name].append((u, nid, m.group(1), url, host))
            break
    else:
        unmatched.append((u, nid, url, host))
print(f"carry an identifier: {sum(len(v) for v in by_pat.values()):,}   "
      f"need a page fetch: {len(unmatched):,}", flush=True)
for k, v in sorted(by_pat.items(), key=lambda x: -len(x[1])):
    print(f"    {len(v):>7,}  {k}", flush=True)


def qlever(q, attempts=4):
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
PROPS = {n: p for n, r, p in PAT}
out, unresolved = [], []
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
for name, items in sorted(by_pat.items(), key=lambda x: -len(x[1])):
    props = PROPS[name]
    if not props:
        unresolved.extend([(u, nid, url, host) for (u, nid, i, url, host) in items])
        continue
    uniq = sorted({i for (_, _, i, _, _) in items})
    pv = " ".join("wdt:" + p for p in dict.fromkeys(props))
    hit = {}
    for c in range(0, len(uniq), CHUNK):
        ch = uniq[c:c + CHUNK]
        vals = " ".join(Q + x.replace(Q, "") + Q for x in ch)
        q = ("PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
             "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
             f"SELECT ?id ?item ?label WHERE {{ VALUES ?id {{ {vals} }} "
             f"VALUES ?p {{ {pv} }} ?item ?p ?id . "
             'OPTIONAL { ?item rdfs:label ?label . FILTER(lang(?label)="en") } }')
        for ln in qlever(q).split("\n")[1:]:
            f = ln.rstrip("\r").split("\t")
            if len(f) < 3 or not f[0]:
                continue
            iid, item, lab = unq(f[0]), f[1].rsplit("/", 1)[-1].rstrip(">"), unq(f[2])
            if lab and iid not in hit:
                hit[iid] = (item, lab)
    print(f"  {name}: {len(uniq):,} distinct ids, matched {len(hit):,} "
          f"({time.time()-t0:.0f}s)", flush=True)
    for (u, nid, i, url, host) in items:
        if i in hit:
            qid, lab = hit[i]
            out.append((u, nid, host, i, lab, f"wikidata:{qid}", url, now))
        else:
            unresolved.append((u, nid, url, host))

if out:
    out.sort()
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in out], pa.int64()),
        "freebase_mid": pa.array([r[1] for r in out]),
        "authority_namespace": pa.array([r[2] for r in out]).dictionary_encode(),
        "authority_id": pa.array([r[3] for r in out]),
        "resolved_label": pa.array([r[4] for r in out]),
        "resolved_entity_type": pa.array([r[5] for r in out]),
        "source_url": pa.array([r[6] for r in out]),
        "retrieved_at": pa.array([r[7] for r in out])}),
        f"{ACQ}/uri_authority_names.parquet", compression="zstd")

need = unresolved + unmatched
if need:
    need.sort()
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in need], pa.int64()),
        "node_id": pa.array([r[1] for r in need]),
        "url": pa.array([r[2] for r in need]),
        "host": pa.array([r[3] for r in need]).dictionary_encode()}),
        f"{ACQ}/uri_pending_fetch.parquet", compression="zstd")

hostc = collections.Counter(r[3] for r in need)
rec = {"schema": "URI_IDENTIFIER_RESOLUTION/v1",
       "generated_utc": now,
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. RESOLUTION_OVERLAY_V1 "
                       "manifest hash 25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865"
                       " unchanged."),
       "RULE": "exact identifier lookup only; no search-by-name at any stage",
       "input_urls": len(recs),
       "urls_carrying_an_identifier": sum(len(v) for v in by_pat.values()),
       "resolved_rows": len(out),
       "resolved_nodes": len({r[0] for r in out}),
       "TIER": "EXTERNAL_AUTHORITY_EXACT",
       "WHY_NOT_A_URL_TIER": ("nothing here was obtained by retrieving a page. The identifier was "
                              "extracted from the URL and resolved against the authority index, so "
                              "the URL tiers are reserved for rows actually fetched from a page."),
       "pending_page_fetch": len(need),
       "pending_top_hosts": dict(hostc.most_common(20)),
       "LIVE_WEB_PROBE": {
           "amazon.com": "HTTP 200 but the title is 'Amazon.com' -- a bot-block page, unusable",
           "discogs.com": "403", "artic.edu": "403", "metal-archives.com": "403",
           "collections.sfmoma.org": "DNS does not resolve", "citysearch.com": "DNS does not resolve",
           "digitalcity.com": "HTTP 200, parked, no title element",
           "newadvent.org": "live and usable, title 'CATHOLIC ENCYCLOPEDIA: Aachen'",
           "CONSEQUENCE": ("the pending cohort is dominated by hosts that are dead or closed to "
                           "automated access, so it is an archive problem, not a live-fetch one.")},
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_URI_IDENTIFIERS.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)

print(f"\nresolved {len(out):,} rows over {len({r[0] for r in out}):,} nodes")
print(f"pending page fetch: {len(need):,}")
for r in out[:12]:
    print(f"   {r[1]:<14} {r[2]:<22} {r[3][:14]:<16} {r[4][:44]!r}")
print(f"{time.time()-t0:.0f}s")
