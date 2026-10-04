"""Download the Wayback CDX index once per host instead of once per URL.

    PYTHONHASHSEED=0 python .../uri_cdx_index.py [max_pages] [min_urls]

WHY
  The first wayback pass asked archive.org one CDX question per URL and measured 6.6 s/item. For
  83,324 URLs that is over six days, and it asks archive.org 83,324 questions to which it already
  holds one answer. The CDX server will return a whole host, or a whole path prefix, in a handful of
  paged requests, and the pending URLs are extremely regular:

      collections.sfmoma.org/OBJ100972.htm            6,259 urls, whole domain = 4 pages
      sanfrancisco.citysearch.com/E/V/SFOCA/...       1,271 urls
      accessarizona.com/auto_docs/dining/35684.html     944 urls
      digitalcity.com/orangecounty/dining/venue.dci?vid=14040

  So: size each candidate with showNumPages (one cheap request), download the ones that fit, and
  leave the rest to the per-URL path. amazon.com/gp/product/ alone is 13,027 pages -- about 195
  million captures -- and is deliberately NOT indexed here; it stays a per-URL cohort.

WHAT THIS DOES NOT DO
  It does not fetch any snapshot and it does not decide any name. It builds a local answer to the
  question "is this exact URL archived, and when", so the fetch stage spends its requests only on
  URLs that are actually there.

Writes only under _acquisition/cdx_index/.
"""
import sys, io, os, json, time, glob, collections, urllib.request, urllib.parse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
OUT = f"{ACQ}/cdx_index"
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
CDX = "https://web.archive.org/cdx/search/cdx"
MAX_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MIN_URLS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
t0 = time.time()
os.makedirs(OUT, exist_ok=True)

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")


def norm(u):
    """Both sides of the join go through this. Wayback stores 'http://host:80/path'; we hold
    'http://www.host/path'. Scheme, www., the default port and a trailing slash are the only
    differences that show up, and none of them changes which document is meant."""
    u = u.split("://", 1)[-1].lower()
    if u.startswith("www."):
        u = u[4:]
    u = u.replace(":80/", "/", 1).replace(":443/", "/", 1)
    return u.rstrip("/")


def req(url, timeout=180, attempts=4):
    for a in range(attempts):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(r, timeout=timeout) as f:
                return f.read()
        except Exception as e:
            if a == attempts - 1:
                return None
            time.sleep(3 * (a + 1))
    return None


def npages(target, mtype):
    b = req(f"{CDX}?url={urllib.parse.quote(target, safe='')}&matchType={mtype}"
            f"&showNumPages=true", timeout=90)
    time.sleep(0.8)
    try:
        return int((b or b"").decode().strip())
    except Exception:
        return None


t = pq.read_table(f"{ACQ}/uri_pending_fetch.parquet")
urls = t["url"].to_pylist()
hosts = t["host"].combine_chunks().cast(pa.string()).to_pylist()
byh = collections.defaultdict(list)
for u, h in zip(urls, hosts):
    byh[h].append(u)
big = sorted([h for h in byh if len(byh[h]) >= MIN_URLS], key=lambda h: -len(byh[h]))
print(f"{len(urls):,} urls over {len(byh):,} hosts; {len(big)} hosts have >= {MIN_URLS} urls "
      f"({sum(len(byh[h]) for h in big):,} urls)", flush=True)

manifest = json.load(io.open(f"{OUT}/_manifest.json", encoding="utf-8")) \
    if os.path.exists(f"{OUT}/_manifest.json") else {}

for h in big:
    safe = h.replace(":", "_")
    if h in manifest and manifest[h].get("status") in ("indexed", "too_large", "empty"):
        continue
    n = len(byh[h])
    # candidate 1: the whole host. candidate 2..: two-segment path prefixes, which is where these
    # sites put their record directories (/E/V/, /auto_docs/dining/, /exec/obidos/ASIN/).
    plan, pages_host = None, npages(h, "domain")
    if pages_host is not None and pages_host <= MAX_PAGES:
        plan = [(h, "domain", pages_host)]
    else:
        pref = collections.Counter()
        for u in byh[h]:
            p = norm(u).split("?", 1)[0].split("/")
            if len(p) >= 3:
                pref["/".join(p[:3]) + "/"] += 1
            elif len(p) == 2:
                pref["/".join(p[:2])] += 1
        cand = [p for p, c in pref.most_common(6) if c >= max(20, 0.05 * n)]
        plan, tot = [], 0
        for p in cand:
            pg = npages(p, "prefix")
            if pg is None:
                continue
            if pg <= MAX_PAGES and tot + pg <= MAX_PAGES * 2:
                plan.append((p, "prefix", pg))
                tot += pg
        if not plan:
            manifest[h] = {"status": "too_large", "urls": n, "domain_pages": pages_host,
                           "prefix_pages": {p: npages(p, "prefix") for p in cand[:2]}}
            print(f"  {h}: too large (domain {pages_host} pages) -> per-URL cohort", flush=True)
            continue

    rows = []
    for target, mtype, pg in plan:
        for page in range(max(pg, 1)):
            b = req(f"{CDX}?url={urllib.parse.quote(target, safe='')}&matchType={mtype}"
                    f"&output=json&fl=original,timestamp&filter=statuscode:200"
                    f"&collapse=urlkey&page={page}")
            time.sleep(1.5)
            if not b:
                continue
            try:
                d = json.loads(b.decode("utf-8", "replace") or "[]")
            except Exception:
                continue
            for r in d[1:]:
                if len(r) >= 2:
                    rows.append((norm(r[0]), r[1]))
    if not rows:
        manifest[h] = {"status": "empty", "urls": n, "plan": [list(x) for x in plan]}
        print(f"  {h}: {n:,} urls, index returned 0 captures", flush=True)
        json.dump(manifest, io.open(f"{OUT}/_manifest.json", "w", encoding="utf-8"), indent=1)
        continue
    rows.sort()
    pq.write_table(pa.table({"url_norm": pa.array([r[0] for r in rows]),
                             "timestamp": pa.array([r[1] for r in rows])}),
                   f"{OUT}/{safe}.parquet", compression="zstd")
    have = {r[0] for r in rows}
    hit = sum(1 for u in byh[h] if norm(u) in have)
    manifest[h] = {"status": "indexed", "urls": n, "captures": len(rows),
                   "pending_urls_archived": hit,
                   "hit_rate": round(hit / n, 4), "plan": [list(x) for x in plan]}
    print(f"  {h}: {n:,} urls, {len(rows):,} captures indexed, {hit:,} of our urls archived "
          f"({hit/n:.1%})  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(manifest, io.open(f"{OUT}/_manifest.json", "w", encoding="utf-8"), indent=1)

json.dump(manifest, io.open(f"{OUT}/_manifest.json", "w", encoding="utf-8"), indent=1)
ind = {k: v for k, v in manifest.items() if v.get("status") == "indexed"}
print(f"\nindexed {len(ind)} hosts; {sum(v['pending_urls_archived'] for v in ind.values()):,} of "
      f"{sum(v['urls'] for v in ind.values()):,} pending urls have a capture")
print(f"{time.time()-t0:.0f}s")
