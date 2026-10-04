"""The 85,133 /uri/ references, part 2: retrieve identity from the page, live or archived.

    PYTHONHASHSEED=0 python .../uri_fetch.py discogs     # 11,741 ids, exact API, ~25 req/min
    PYTHONHASHSEED=0 python .../uri_fetch.py wbindex     # dead hosts, via the local CDX index
    PYTHONHASHSEED=0 python .../uri_fetch.py wayback     # dead hosts, one CDX request per url
    PYTHONHASHSEED=0 python .../uri_fetch.py live        # hosts that still answer

ACCEPTANCE ORDER, AS SPECIFIED
    schema.org name/headline  >  OpenGraph title  >  domain-specific canonical  >  <title>
  and the archived equivalent of the same ladder when the page is only in the Wayback Machine. The
  rung that produced the value is recorded per row, so a row sourced from a bare <title> is never
  mistaken for one sourced from structured data.

SUFFIX STRIPPING IS WHITELISTED, NOT GUESSED
  Only deterministic, established site templates are stripped -- " - IMDb", " | Discogs",
  " - Wikipedia". The rule is a fixed table keyed by host; nothing is trimmed heuristically, because
  a title that merely looks like it has a suffix may simply have a long name.

TWO TIERS, KEPT APART
  EXTERNAL_URL_LIVE_EXACT     the page answered now, at the exact URL Freebase stored
  EXTERNAL_URL_ARCHIVE_EXACT  the page is gone and the value came from a dated snapshot of that
                              exact URL. The snapshot timestamp is retained, because a 2007 title
                              is evidence about 2007 and should be readable as such.

RESUMABLE BY CONSTRUCTION
  The Discogs stage alone is roughly eight hours at the rate the service asks for. Progress is
  appended to a shard every few hundred rows and already-done ids are skipped on restart, so an
  interruption costs one shard, not the run.

Writes only under _acquisition/. Touches neither the frozen graph nor the frozen overlay.
"""
import sys, io, os, re, json, glob, time, gzip, html, urllib.request, urllib.parse, urllib.error
import collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
           "Chrome/124.0 Safari/537.36")
STAGE = sys.argv[1] if len(sys.argv) > 1 else "discogs"
OUT = f"{ACQ}/uri_{STAGE}_hits"
FLUSH = 200
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
os.makedirs(OUT, exist_ok=True)

# host -> suffix that this site's template appends to every page title. Fixed table, not a heuristic.
SUFFIX = {"discogs.com": [" | Discogs"], "imdb.com": [" - IMDb"],
          "metal-archives.com": [" - Encyclopaedia Metallum: The Metal Archives"],
          "artic.edu": [" | The Art Institute of Chicago"],
          "newadvent.org": [], "en.wikipedia.org": [" - Wikipedia"]}

OG = re.compile(r'<meta[^>]+(?:property|name)=["\']og:title["\'][^>]*content=["\']([^"\']+)', re.I)
OG2 = re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\']og:title', re.I)
TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
LD = re.compile(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', re.I | re.S)
TAG = re.compile(r"<[^>]+>")


def strip_suffix(t, host):
    for s in SUFFIX.get(host, []):
        if t.endswith(s):
            return t[: -len(s)].strip()
    return t


def identity(body, host):
    """The acceptance ladder. Returns (value, which_rung) or (None, None)."""
    for m in LD.finditer(body):
        try:
            d = json.loads(m.group(1).strip())
        except Exception:
            continue
        for o in (d if isinstance(d, list) else [d]):
            if isinstance(o, dict):
                v = o.get("name") or o.get("headline")
                if isinstance(v, str) and v.strip():
                    return strip_suffix(html.unescape(v.strip()), host), "schema.org"
    m = OG.search(body) or OG2.search(body)
    if m and m.group(1).strip():
        return strip_suffix(html.unescape(m.group(1).strip()), host), "opengraph"
    m = TITLE.search(body)
    if m:
        v = html.unescape(TAG.sub("", m.group(1))).strip()
        v = re.sub(r"\s+", " ", v)
        if v:
            return strip_suffix(v, host), "html_title"
    return None, None


def fetch(url, headers=None, timeout=40, cap=500000):
    h = {"User-Agent": UA, "Accept-Encoding": "gzip"}
    if headers:
        h.update(headers)
    r = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(r, timeout=timeout) as f:
        b = f.read(cap)
        if f.headers.get("Content-Encoding") == "gzip":
            try:
                b = gzip.decompress(b)
            except Exception:
                pass
        return f.status, f.url, b.decode("utf-8", "replace")


done = set()
for fp in glob.glob(f"{OUT}/*.parquet"):
    try:
        done |= set(pq.read_table(fp, columns=["node_uid"])["node_uid"].to_pylist())
    except Exception:
        pass
seen_fail = set()
fp_fail = f"{ACQ}/uri_{STAGE}_missed.json"
if os.path.exists(fp_fail):
    try:
        seen_fail = set(json.load(io.open(fp_fail, encoding="utf-8")))
    except Exception:
        pass
print(f"already done: {len(done):,} rows, {len(seen_fail):,} recorded misses", flush=True)

t = pq.read_table(f"{ACQ}/uri_pending_fetch.parquet")
PEND = [r for r in zip(t["node_uid"].to_pylist(), t["node_id"].to_pylist(), t["url"].to_pylist(),
                       t["host"].combine_chunks().cast(pa.string()).to_pylist())
        if r[0] not in done and r[0] not in seen_fail]
print(f"pending for stage {STAGE}: {len(PEND):,}", flush=True)

rows, miss = [], collections.Counter()
part = len(glob.glob(f"{OUT}/*.parquet"))
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def flush():
    global part, rows
    if not rows:
        return
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in rows], pa.int64()),
        "freebase_mid": pa.array([r[1] for r in rows]),
        "url": pa.array([r[2] for r in rows]),
        "host": pa.array([r[3] for r in rows]),
        "resolved_label": pa.array([r[4] for r in rows]),
        "identity_rung": pa.array([r[5] for r in rows]),
        "snapshot_timestamp": pa.array([r[6] for r in rows]),
        "tier": pa.array([r[7] for r in rows]),
        "retrieved_at": pa.array([r[8] for r in rows])}),
        f"{OUT}/part_{part:05d}.parquet", compression="zstd")
    part += 1
    rows = []


# ARTIST AND LABEL ARE NAME SLUGS, NOT IDS. Every discogs URL Freebase stored uses the OLD
# scheme, where /artist/<seg> and /label/<seg> carry the NAME: /artist/3LW, /artist/2+Mental+(2),
# /artist/1200+Mics. Bands whose names are numbers -- 112, 707, 1927, 35007, 08001, 3-D, 4-4-2 --
# made an unanchored (\d+) match a FRAGMENT of the name and send it to
# api.discogs.com/artists/<n>, which returned a completely unrelated artist. That is why 26 nodes
# all came back "Mr. James Barth & A.D." (every /artist/2... collapsed to artists/2) and 13 came
# back "Josh Wink" (/artist/3...). All 168 artist rows already written are invalid; see
# V3_DISCOGS_ARTIST_SLUG_DEFECT. Only /release/<id> and /master/<id> are numeric in this scheme,
# and the digits are anchored so a name fragment can never be read as an id.
DISCOGS = re.compile(r"discogs\.com/(release|master)/(\d+)(?=[-/?#]|$)", re.I)
API = {"release": "releases", "master": "masters"}

if STAGE == "discogs":
    items = [r for r in PEND if r[3] == "discogs.com" and DISCOGS.search(r[2])]
    print(f"discogs items: {len(items):,}  (~{len(items)*2.6/3600:.1f}h at the requested rate)",
          flush=True)
    for n, (u, mid, url, host) in enumerate(items):
        m = DISCOGS.search(url)
        kind, rid = m.group(1).lower(), m.group(2)
        try:
            s, _, b = fetch(f"https://api.discogs.com/{API[kind]}/{rid}",
                            {"Accept": "application/json"}, timeout=45)
            d = json.loads(b)
            lab = d.get("title") or d.get("name")
            if lab:
                rows.append((u, mid, url, host, lab.strip(), "discogs_api", "",
                             "EXTERNAL_URL_LIVE_EXACT", now()))
            else:
                miss["no_title_field"] += 1
                seen_fail.add(u)
        except urllib.error.HTTPError as e:
            miss[f"http_{e.code}"] += 1
            if e.code in (404, 403, 401):
                seen_fail.add(u)
            if e.code == 429:
                time.sleep(60)
        except Exception as e:
            miss[type(e).__name__] += 1
        time.sleep(2.6)                      # Discogs asks unauthenticated clients to stay <25/min
        if len(rows) >= FLUSH:
            flush()
            json.dump(sorted(seen_fail), io.open(fp_fail, "w", encoding="utf-8"))
            print(f"  {n+1:,}/{len(items):,} kept={part*FLUSH:,} miss={sum(miss.values()):,} "
                  f"({time.time()-t0:.0f}s)", flush=True)

elif STAGE == "wayback":
    HOSTS = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    items = [r for r in PEND if (HOSTS is None or r[3] in HOSTS)]
    print(f"wayback items: {len(items):,}", flush=True)
    for n, (u, mid, url, host) in enumerate(items):
        ts = None
        try:
            q = ("https://web.archive.org/cdx/search/cdx?url=" + urllib.parse.quote(url, safe="")
                 + "&output=json&fl=timestamp&filter=statuscode:200&limit=1")
            s, _, b = fetch(q, timeout=45, cap=60000)
            d = json.loads(b or "[]")
            ts = d[1][0] if len(d) > 1 else None
        except Exception as e:
            miss["cdx_" + type(e).__name__] += 1
        if not ts:
            miss["no_snapshot"] += 1
            seen_fail.add(u)
        else:
            try:
                s, _, b = fetch(f"https://web.archive.org/web/{ts}id_/{url}",
                                {"User-Agent": BROWSER}, timeout=60)
                lab, rung = identity(b, host)
                if lab:
                    rows.append((u, mid, url, host, lab, rung, ts,
                                 "EXTERNAL_URL_ARCHIVE_EXACT", now()))
                else:
                    miss["snapshot_no_identity"] += 1
                    seen_fail.add(u)
            except Exception as e:
                miss["snap_" + type(e).__name__] += 1
        time.sleep(1.2)
        if len(rows) >= FLUSH or (n % 500 == 0 and n):
            flush()
            json.dump(sorted(seen_fail), io.open(fp_fail, "w", encoding="utf-8"))
            print(f"  {n+1:,}/{len(items):,} miss={dict(miss)} ({time.time()-t0:.0f}s)", flush=True)

elif STAGE == "wbindex":
    # THE INDEXED ARCHIVE PASS. uri_cdx_index.py has already downloaded, per host, the Wayback CDX
    # index that says which exact URLs were captured and when. Consulting it locally removes the CDX
    # request from the loop entirely: the per-URL pass measured 6.6 s/item because it asked
    # archive.org one question per URL, and this asks it none. What is left is the snapshot fetch,
    # and only for URLs the index says are actually there -- digitalcity.com, for instance, turned
    # out to have 2 of its 7,980 URLs archived, which the index established in 41 seconds.
    HOSTS = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None

    def unorm(u):
        u = u.split("://", 1)[-1].lower()
        if u.startswith("www."):
            u = u[4:]
        u = u.replace(":80/", "/", 1).replace(":443/", "/", 1)
        return u.rstrip("/")

    idx = {}
    for fp in sorted(glob.glob(f"{ACQ}/cdx_index/*.parquet")):
        d = pq.read_table(fp)
        for a, b in zip(d["url_norm"].to_pylist(), d["timestamp"].to_pylist()):
            if a not in idx:
                idx[a] = b                  # the index is collapsed per urlkey: earliest capture
    print(f"cdx index: {len(idx):,} archived urls known", flush=True)
    # "not in the index" means two different things and they must not be reported as one. A URL on
    # a host that WAS indexed and is absent from that index is genuinely not archived. A URL on a
    # host with no index yet -- amazon.com's prefix alone is 13,027 CDX pages -- is simply unknown.
    mf = json.load(io.open(f"{ACQ}/cdx_index/_manifest.json", encoding="utf-8"))         if os.path.exists(f"{ACQ}/cdx_index/_manifest.json") else {}
    indexed = {h for h, v in mf.items() if v.get("status") in ("indexed", "empty")}
    cand = [r for r in PEND if (HOSTS is None or r[3] in HOSTS)]
    items = [r for r in cand if unorm(r[2]) in idx]
    absent = [r for r in cand if unorm(r[2]) not in idx]
    miss["host_indexed_url_not_archived"] = sum(1 for r in absent if r[3] in indexed)
    miss["host_not_indexed_yet"] = sum(1 for r in absent if r[3] not in indexed)
    print(f"wbindex: {len(items):,} of {len(cand):,} pending urls have a capture; "
          f"{miss['host_indexed_url_not_archived']:,} are on an indexed host and genuinely not "
          f"archived; {miss['host_not_indexed_yet']:,} are on a host with no index yet",
          flush=True)
    for n, (u, mid, url, host) in enumerate(items):
        ts = idx[unorm(url)]
        try:
            s, _, b = fetch(f"https://web.archive.org/web/{ts}id_/{url}",
                            {"User-Agent": BROWSER}, timeout=60)
            lab, rung = identity(b, host)
            if lab and lab.lower() not in (host, host.replace("www.", "")):
                rows.append((u, mid, url, host, lab, rung, ts,
                             "EXTERNAL_URL_ARCHIVE_EXACT", now()))
            else:
                # a capture that exists but carries no usable identity is a miss, not a name
                miss["snapshot_no_identity"] += 1
                seen_fail.add(u)
        except Exception as e:
            miss["snap_" + type(e).__name__] += 1
        time.sleep(1.2)
        if len(rows) >= FLUSH or (n % 500 == 0 and n):
            flush()
            json.dump(sorted(seen_fail), io.open(fp_fail, "w", encoding="utf-8"))
            print(f"  {n+1:,}/{len(items):,} kept={part*FLUSH:,} miss={dict(miss)} "
                  f"({time.time()-t0:.0f}s)", flush=True)

elif STAGE == "live":
    HOSTS = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
    items = [r for r in PEND if (HOSTS is None or r[3] in HOSTS)]
    print(f"live items: {len(items):,}", flush=True)
    for n, (u, mid, url, host) in enumerate(items):
        try:
            s, fin, b = fetch(url, {"User-Agent": BROWSER}, timeout=35)
            lab, rung = identity(b, host)
            if lab and lab.lower() not in (host, host.replace("www.", "")):
                rows.append((u, mid, url, host, lab, rung, "",
                             "EXTERNAL_URL_LIVE_EXACT", now()))
            else:
                miss["no_identity"] += 1
                seen_fail.add(u)
        except urllib.error.HTTPError as e:
            miss[f"http_{e.code}"] += 1
            seen_fail.add(u)
        except Exception as e:
            miss[type(e).__name__] += 1
        time.sleep(0.8)
        if len(rows) >= FLUSH or (n % 500 == 0 and n):
            flush()
            json.dump(sorted(seen_fail), io.open(fp_fail, "w", encoding="utf-8"))
            print(f"  {n+1:,}/{len(items):,} miss={sum(miss.values()):,} ({time.time()-t0:.0f}s)",
                  flush=True)
else:
    sys.exit(f"unknown stage {STAGE!r}")

flush()
json.dump(sorted(seen_fail), io.open(fp_fail, "w", encoding="utf-8"))
tot = 0
for fp in glob.glob(f"{OUT}/*.parquet"):
    tot += pq.ParquetFile(fp).metadata.num_rows
rec = {"schema": f"URI_FETCH_{STAGE.upper()}/v1",
       "generated_utc": now(),
       "APPEND_ONLY": "writes only under _acquisition/; frozen artifacts untouched.",
       "stage": STAGE,
       "rows_total_including_earlier_runs": tot,
       "misses_this_run": dict(miss),
       "ACCEPTANCE_ORDER": ["schema.org name/headline", "opengraph title", "html <title>"],
       "SUFFIX_RULE": ("whitelisted per host only; nothing is trimmed heuristically because a title "
                       "that looks suffixed may simply have a long name"),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_URI_FETCH_{STAGE.upper()}.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\n{STAGE}: {tot:,} rows total   misses this run {dict(miss)}")
print(f"{time.time()-t0:.0f}s")
