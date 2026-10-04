"""The URL cohort, resolved per URL the way it was specified: live, then structured, then archived.

    PYTHONHASHSEED=0 python .../uri_resolve.py tail                  # hosts with < 100 urls
    PYTHONHASHSEED=0 python .../uri_resolve.py hosts amazon.com,...  # named hosts
    PYTHONHASHSEED=0 python .../uri_resolve.py authority             # the 16 dead-authority rows
    PYTHONHASHSEED=0 python .../uri_resolve.py index                 # hosts with a local CDX index
    PYTHONHASHSEED=0 python .../uri_resolve.py validate              # re-check every hit on disk

THE FLOW, PER URL
    exact URL  ->  live redirect chain  ->  structured page identity  ->  historical snapshot if dead
  1. LIVE     the exact URL Freebase stored is fetched, redirects followed, the final URL recorded.
              A redirect that leaves the page behind is not the page: if the chain lands on another
              site and the URL's own identifier token does not survive, or on the site's root, or
              on an error/search/login page, the live answer is rejected and the URL is treated as
              dead. A page whose only identity is the site's own name (a parked domain, Amazon's
              bot page titled "Amazon.com", "Myspace"), or a server default / interstitial title,
              is not a name either and falls through.
  2. ARCHIVE  the Wayback availability API is asked for the capture closest to 2010-01-01 -- the
              middle of Freebase's life, so a title contemporaneous with the key -- and the raw
              snapshot (the id_ form) is read through the same identity ladder.
  In 'index' mode the availability question is answered from the CDX index uri_cdx_index.py
  already downloaded, and archive.org is asked only for the snapshot.

ACCEPTANCE ORDER, AS SPECIFIED
    schema.org name/headline  >  OpenGraph title  >  <title>
  with the rung recorded per row. Prefix and suffix stripping is a fixed per-host whitelist of
  established site templates ("CATHOLIC ENCYCLOPEDIA: ", " - IMDb", Amazon's ": Amazon.com: Books"
  category tail); nothing is trimmed on a guess, and nothing is ever inferred from surrounding text.

VALIDATION -- THE PART THAT ACTUALLY BIT
  The first authority run accepted eight dead Google+ profiles that all redirect to one unrelated
  "Google Workspace Updates" landing page. Two rules now catch that class: the redirect rule above,
  applied at fetch time, and a post-run pass that quarantines every live row whose final URL is
  shared with another original URL (two pages cannot both be one page). 'validate' re-applies all
  the rules to every row on disk, rewrites this stage's parts without the failures (so they leave
  the done-set and are retried through the archive route), and appends the failures with their
  reason to uri_quarantine.parquet, which the cascade must exclude at fold-in.

MISSES ARE NOT ALL FINAL
  A URL with no snapshot, a snapshot with no identity, or an archive 404 is exhausted and goes to
  uri_resolve_missed.json (part of the done-set). A timeout, connection reset, 429 or 5xx is a
  transient failure recorded in uri_resolve_transient.json, which the done-set ignores, so the URL
  is simply attempted again on the next run.

WHY THREADS, AND WHY A SHARED PACER
  The serial indexed pass measured 5.1 s per URL, almost all of it waiting on a single socket.
  Three workers run concurrently, but every request to archive.org goes through one pacer that
  enforces a minimum gap regardless of which worker asks. Origin hosts are unpaced because the tail
  spreads 14K URLs across 8.5K of them; a host that has failed live five times in a row is not
  asked again this run.

RESUMABLE, AND IT KNOWS WHAT THE OTHER STAGES DID
  Done-sets are read from every uri_*_hits/ directory and every uri_*_missed.json, so a node
  resolved or exhausted by an earlier stage is never fetched twice.

Writes only under _acquisition/. Touches neither the frozen graph nor the frozen overlay.
"""
import sys, io, os, re, json, glob, time, gzip, zlib, html, threading, collections
import urllib.request, urllib.parse, urllib.error
from concurrent.futures import ThreadPoolExecutor
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow as pa, pyarrow.parquet as pq
import requests

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
OUT = f"{ACQ}/uri_resolve_hits"
QUAR = f"{ACQ}/uri_quarantine.parquet"
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
           "Chrome/124.0 Safari/537.36")
MODE = sys.argv[1] if len(sys.argv) > 1 else "tail"
ARG = sys.argv[2] if len(sys.argv) > 2 else ""
WORKERS = 3
ARCHIVE_GAP = 0.75                  # seconds between any two requests to archive.org, globally
FLUSH = 200
GZIP_MAGIC = bytes([0x1F, 0x8B])
t0 = time.time()
os.makedirs(OUT, exist_ok=True)

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

# ---------------------------------------------------------------- the identity ladder
SUFFIX = {"discogs.com": [" | Discogs"], "imdb.com": [" - IMDb"],
          "metal-archives.com": [" - Encyclopaedia Metallum: The Metal Archives"],
          "artic.edu": [" | The Art Institute of Chicago"], "en.wikipedia.org": [" - Wikipedia"],
          "allmusic.com": [" | AllMusic"], "sfgate.com": [" - SFGate"],
          "plus.google.com": [" - Google+"], "myspace.com": [" on Myspace", " | Myspace"]}
PREFIX = {"newadvent.org": ["CATHOLIC ENCYCLOPEDIA: "],
          "amazon.com": ["Amazon.com: "], "amazon.co.uk": ["Amazon.co.uk: "],
          "amazon.de": ["Amazon.de: "], "amazon.fr": ["Amazon.fr: "],
          "amazon.co.jp": ["Amazon.co.jp: "]}
# Amazon's older template puts the site and the store category at the END of the title:
#   "Harry Potter and the Sorcerer's Stone: J.K. Rowling: Amazon.com: Books"
AMAZON_TAIL = re.compile(r"\s*:\s*Amazon\.(?:com|co\.uk|de|fr|co\.jp)\s*:\s*[^:]{1,40}$", re.I)
OG = re.compile(r'<meta[^>]+(?:property|name)=["\']og:title["\'][^>]*content=["\']([^"\']+)', re.I)
OG2 = re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\']og:title', re.I)
TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
LD = re.compile(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', re.I | re.S)
TAG = re.compile(r"<[^>]+>")
# server defaults, parking pages and interstitials: deterministic non-identities
GENERIC = {"index of /", "welcome to nginx!", "apache2 ubuntu default page: it works", "it works!",
           "iis windows server", "iis7", "default web site page", "under construction",
           "coming soon", "site unavailable", "account suspended", "domain suspended",
           "this site can't be reached", "suspended", "web hosting", "just a moment...",
           "attention required! | cloudflare", "access denied", "forbidden", "403 forbidden",
           "404", "404 not found", "not found", "page not found", "error", "error 404",
           "object moved", "document moved", "moved", "redirecting", "redirect", "loading",
           "please wait", "sign in", "log in", "login", "are you a robot?", "robot check",
           "captcha", "503 service unavailable", "service unavailable", "bad gateway",
           "502 bad gateway", "domain for sale", "this domain is for sale", "buy this domain",
           "untitled document", "untitled page", "home", "homepage", "home page", "welcome",
           "index", "main page", "search", "search results", "new page 1", "amazon.com",
           "myspace", "facebook", "twitter", "youtube", "google+", "google",
           # placeholders a dead profile page is served with
           "default name", "default", "unknown", "n/a", "none", "null", "undefined", "no title",
           "(no title)", "title", "name", "profile", "user", "user profile", "my profile"}
GENERIC_SUB = ("is for sale", "for sale!", "domain name", "web hosting", "hugedomains", "godaddy",
               "sedo", "parked", "buy this domain", "this website is for sale", "namecheap",
               "just a moment", "cloudflare", "captcha", "are you a robot", "access denied",
               "site not found", "page not found", "no longer available", "has been suspended")
FIN_BAD = ("/error", "404", "notfound", "not-found", "/search", "login", "signin", "sign-in",
           "/suspended", "expired", "for-sale", "forsale")


def reg_domain(h):
    h = (h or "").lower()
    h = h[4:] if h.startswith("www.") else h
    p = h.split(".")
    if len(p) >= 3 and p[-2] in ("co", "com", "org", "net", "ac", "gov", "edu", "or", "ne") \
            and len(p[-1]) == 2:
        return ".".join(p[-3:])
    return ".".join(p[-2:]) if len(p) >= 2 else h


def strip_template(t, host):
    h = host[4:] if host.startswith("www.") else host
    for s in SUFFIX.get(h, []) + SUFFIX.get(reg_domain(h), []):
        if t.endswith(s):
            t = t[: -len(s)].strip()
    for p in PREFIX.get(h, []) + PREFIX.get(reg_domain(h), []):
        if t.startswith(p):
            t = t[len(p):].strip()
    if reg_domain(h).startswith("amazon."):
        t = AMAZON_TAIL.sub("", t).strip()
    return t


def identity(body, host):
    """-> (value, rung) or (None, None). Structured data first, bare <title> last."""
    for m in LD.finditer(body):
        try:
            d = json.loads(m.group(1).strip())
        except Exception:
            continue
        for o in (d if isinstance(d, list) else [d]):
            if isinstance(o, dict):
                v = o.get("name") or o.get("headline")
                if isinstance(v, str) and v.strip():
                    return strip_template(html.unescape(v.strip()), host), "schema.org"
    m = OG.search(body) or OG2.search(body)
    if m and m.group(1).strip():
        return strip_template(html.unescape(m.group(1).strip()), host), "opengraph"
    m = TITLE.search(body)
    if m:
        v = re.sub(r"\s+", " ", html.unescape(TAG.sub("", m.group(1)))).strip()
        if v:
            return strip_template(v, host), "html_title"
    return None, None


def usable(lab, host):
    """A label that is just the site's own name, a server default or an interstitial is not a page."""
    if not lab:
        return False
    l = re.sub(r"\s+", " ", lab.lower()).strip(" .|-:")
    if len(l) < 2:
        return False
    h = host.lower()
    h = h[4:] if h.startswith("www.") else h
    rd = reg_domain(h)
    brand = rd.split(".")[0]
    if l in GENERIC or l in {h, rd, brand, h.split(".")[0], "www." + rd, rd + ".com"}:
        return False
    if rd in l or ("www." + rd) in l:
        return False                              # the site talking about itself, not a page
    return not any(s in l for s in GENERIC_SUB)


def id_token(url):
    p = urllib.parse.urlsplit(url)
    toks = [t for t in re.split(r"[/?=&#.,;:+]", p.path + "?" + p.query) if len(t) >= 4]
    return toks[-1].lower() if toks else ""


def redirect_ok(url, fin):
    """Did the live chain stay on the page? Same URL, or same site off the root and off error/search
    pages, or another site where the URL's own identifier token survives."""
    if not fin or unorm(fin) == unorm(url):
        return True
    fs = urllib.parse.urlsplit(fin)
    fpath = (fs.path or "").lower()
    if any(b in fpath for b in FIN_BAD) or any(b in (fs.query or "").lower() for b in FIN_BAD):
        return False
    if reg_domain(fs.hostname) == reg_domain(urllib.parse.urlsplit(url).hostname):
        p = fpath.strip("/")
        return bool(p) and p not in ("index.html", "index.htm", "index.php", "index.asp", "home",
                                     "default.aspx", "default.asp", "default.htm", "main")
    tok = id_token(url)
    return bool(tok) and tok in fin.lower()


TRANSIENT = ("URLError", "timeout", "Timeout", "RemoteDisconnected", "ConnectionReset",
             "ConnectionError", "ChunkedEncodingError", "ProtocolError", "IncompleteRead",
             "http_429", "http_5", "avail_", "OSError", "socket")


def is_transient(why):
    return any(t in why for t in TRANSIENT)


# ---------------------------------------------------------------- transport
_alock, _alast = threading.Lock(), [0.0]


def pace_archive():
    with _alock:
        d = ARCHIVE_GAP - (time.time() - _alast[0])
        if d > 0:
            time.sleep(d)
        _alast[0] = time.time()


_tls = threading.local()


def sess():
    """One requests.Session per worker thread: archive.org limits connection churn per IP (new TCP
    connections were being SYN-dropped at ~1/s across our jobs), so connections are kept alive."""
    s = getattr(_tls, "s", None)
    if s is None:
        s = _tls.s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})
    return s


def inflate(raw, enc):
    """Decompress what we have. Reading only the first `cap` bytes truncates the gzip stream, and the
    stock decoder raises DecodeError on that; a decompressobj just returns the prefix it could."""
    if "gzip" in enc.lower() or "deflate" in enc.lower() or raw[:2] == GZIP_MAGIC:
        try:
            return zlib.decompressobj(zlib.MAX_WBITS | 32).decompress(raw)
        except Exception:
            return raw
    return raw


def fetch(url, headers=None, timeout=30, cap=400000):
    """-> (status, final_url, text). Raises urllib.error.HTTPError on 4xx/5xx so the callers'
    accounting is unchanged."""
    r = sess().get(url, headers=headers or {}, timeout=timeout, stream=True, allow_redirects=True)
    try:
        if r.status_code >= 400:
            raise urllib.error.HTTPError(url, r.status_code, r.reason, r.headers, None)
        b = inflate(r.raw.read(cap, decode_content=False), r.headers.get("Content-Encoding", ""))
    finally:
        r.close()
    try:
        return r.status_code, r.url, b.decode(r.encoding or "utf-8", "replace")
    except LookupError:
        return r.status_code, r.url, b.decode("utf-8", "replace")


def unorm(u):
    u = u.split("://", 1)[-1].lower()
    if u.startswith("www."):
        u = u[4:]
    return u.replace(":80/", "/", 1).replace(":443/", "/", 1).rstrip("/")


now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
COLS = ["node_uid", "freebase_mid", "url", "host", "resolved_label", "identity_rung",
        "snapshot_timestamp", "tier", "retrieved_at", "final_url"]


def to_table(rs):
    return pa.table({"node_uid": pa.array([r[0] for r in rs], pa.int64()),
                     **{c: pa.array([r[i] for r in rs]) for i, c in enumerate(COLS) if i}})


# ---------------------------------------------------------------- validate: every hit on disk
def check_row(r, collapsed):
    """-> reason or None. r is a dict with the COLS keys."""
    lab, host, url, fin, tier = r["resolved_label"], r["host"], r["url"], r.get("final_url") or "", r["tier"]
    if not usable(lab, host):
        return "label_is_site_or_generic"
    if tier == "EXTERNAL_URL_LIVE_EXACT":
        if not redirect_ok(url, fin):
            return "live_redirected_away"
        if fin and unorm(fin) in collapsed:
            return "redirect_collapsed_with_other_url"
    return None


def validate():
    quar = []
    stages = {"uri_resolve": True, "uri_wbindex": False, "uri_live": False}   # True = rewrite parts
    for stage, rewrite in stages.items():
        files = sorted(glob.glob(f"{ACQ}/{stage}_hits/*.parquet"))
        if not files:
            continue
        rows = []
        for fp in files:
            for r in pq.read_table(fp).to_pylist():
                r.setdefault("final_url", "")
                rows.append(r)
        # a final URL shared by two DIFFERENT original URLs identifies neither
        by_fin = collections.defaultdict(set)
        for r in rows:
            if r["tier"] == "EXTERNAL_URL_LIVE_EXACT" and r["final_url"]:
                by_fin[unorm(r["final_url"])].add(unorm(r["url"]))
        collapsed = {k for k, v in by_fin.items() if len(v) >= 2}
        good, bad = [], []
        for r in rows:
            why = check_row(r, collapsed)
            (bad if why else good).append((r, why))
        print(f"validate {stage}: {len(rows):,} rows, {len(bad):,} quarantined "
              f"{dict(collections.Counter(w for _, w in bad))}", flush=True)
        for r, why in bad:
            quar.append({"node_uid": r["node_uid"], "stage": stage, "url": r["url"],
                         "resolved_label": r["resolved_label"], "final_url": r["final_url"],
                         "reason": why, "quarantined_at": now()})
        if rewrite and bad:
            for fp in files:
                os.remove(fp)
            keep = [tuple(r[c] for c in COLS) for r, _ in good]
            for i in range(0, len(keep), 5000):
                pq.write_table(to_table(keep[i:i + 5000]), f"{OUT}/part_{i//5000:05d}.parquet",
                               compression="zstd")
            print(f"  rewrote {stage}_hits: {len(keep):,} rows kept", flush=True)
    if quar:
        old = pq.read_table(QUAR).to_pylist() if os.path.exists(QUAR) else []
        have = {(q["node_uid"], q["stage"]) for q in old}
        new = [q for q in quar if (q["node_uid"], q["stage"]) not in have]
        allq = old + new
        pq.write_table(pa.table({k: pa.array([q[k] for q in allq],
                                             pa.int64() if k == "node_uid" else pa.string())
                                 for k in ("node_uid", "stage", "url", "resolved_label",
                                           "final_url", "reason", "quarantined_at")}),
                       QUAR, compression="zstd")
        print(f"quarantine: {len(allq):,} rows on disk ({len(new):,} new)", flush=True)


if MODE == "validate":
    validate()
    print(f"{time.time()-t0:.0f}s")
    sys.exit(0)

# ---------------------------------------------------------------- what is already done
done = set()
for fp in glob.glob(f"{ACQ}/uri_*_hits/*.parquet"):
    try:
        done |= set(pq.read_table(fp, columns=["node_uid"])["node_uid"].to_pylist())
    except Exception:
        pass
for fp in glob.glob(f"{ACQ}/uri_*_missed.json"):
    try:
        v = json.load(io.open(fp, encoding="utf-8"))
        done |= set(v if isinstance(v, list) else [int(k) for k in v])
    except Exception:
        pass
print(f"already resolved or exhausted by any stage: {len(done):,}", flush=True)

# ---------------------------------------------------------------- the cohort
if MODE == "authority":
    t = pq.read_table(f"{ACQ}/authority_pending_archive.parquet")
    PEND = [(u, m, url, urllib.parse.urlsplit(url).hostname or "") for u, m, url in
            zip(t["node_uid"].to_pylist(), t["freebase_mid"].to_pylist(), t["url"].to_pylist())]
else:
    t = pq.read_table(f"{ACQ}/uri_pending_fetch.parquet")
    PEND = list(zip(t["node_uid"].to_pylist(), t["node_id"].to_pylist(), t["url"].to_pylist(),
                    t["host"].combine_chunks().cast(pa.string()).to_pylist()))
hc = collections.Counter(r[3] for r in PEND)
IDX = {}
if MODE == "index":
    for fp in sorted(glob.glob(f"{ACQ}/cdx_index/*.parquet")):
        d = pq.read_table(fp)
        for a, b in zip(d["url_norm"].to_pylist(), d["timestamp"].to_pylist()):
            IDX.setdefault(a, b)
    PEND = [r for r in PEND if unorm(r[2]) in IDX]
elif MODE == "tail":
    PEND = [r for r in PEND if hc[r[3]] < 100]
elif MODE == "hosts":
    hs = set(ARG.split(","))
    PEND = [r for r in PEND if r[3] in hs]
PEND = [r for r in PEND if r[0] not in done]
print(f"mode {MODE}: {len(PEND):,} urls over {len({r[3] for r in PEND}):,} hosts", flush=True)

# ---------------------------------------------------------------- resolution
rows, miss = [], collections.Counter()
failed, transient = {}, {}
dead_live = collections.Counter()
lock = threading.Lock()
part = len(glob.glob(f"{OUT}/*.parquet"))
SKIP_LIVE = {"amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.co.jp",
             "digitalcity.com", "collections.sfmoma.org"} | {h for h in hc if "citysearch" in h}
if MODE == "index":
    SKIP_LIVE |= set(hc)                     # indexed hosts are the dead ones by construction


def flush_locked():
    global part, rows
    if rows:
        pq.write_table(to_table(rows), f"{OUT}/part_{part:05d}.parquet", compression="zstd")
        part += 1
        rows = []
    json.dump(failed, io.open(f"{ACQ}/uri_resolve_missed.json", "w", encoding="utf-8"))
    json.dump(transient, io.open(f"{ACQ}/uri_resolve_transient.json", "w", encoding="utf-8"))


def resolve(item):
    u, mid, url, host = item
    # 1. live
    if host not in SKIP_LIVE and dead_live[host] < 5:
        try:
            s, fin, b = fetch(url, {"User-Agent": BROWSER}, timeout=25)
            lab, rung = identity(b, host)
            if not redirect_ok(url, fin):
                dead_live[host] += 1
                why = "live_redirected_away"
            elif usable(lab, host):
                dead_live[host] = 0
                return (u, mid, url, host, lab, rung, "", "EXTERNAL_URL_LIVE_EXACT", now(), fin), \
                    None
            else:
                dead_live[host] += 1
                why = "live_no_identity"
        except urllib.error.HTTPError as e:
            dead_live[host] += 1
            why = f"live_http_{e.code}"
        except Exception as e:
            dead_live[host] += 1
            why = "live_" + type(e).__name__
    else:
        why = "live_skipped"
    # 2. archive
    ts = IDX.get(unorm(url)) if MODE == "index" else None
    if ts is None and MODE != "index":
        try:
            pace_archive()
            s, _, b = fetch("https://archive.org/wayback/available?url="
                            + urllib.parse.quote(url, safe="") + "&timestamp=20100101",
                            timeout=45, cap=20000)
            c = (json.loads(b).get("archived_snapshots") or {}).get("closest") or {}
            ts = c.get("timestamp") if str(c.get("status", "")).startswith("2") else None
        except Exception as e:
            return None, f"{why}|avail_{type(e).__name__}"
    if not ts:
        return None, f"{why}|no_snapshot"
    try:
        pace_archive()
        s, _, b = fetch(f"https://web.archive.org/web/{ts}id_/{url}", {"User-Agent": BROWSER},
                        timeout=60)
        lab, rung = identity(b, host)
        if usable(lab, host):
            return (u, mid, url, host, lab, rung, ts, "EXTERNAL_URL_ARCHIVE_EXACT", now(), ""), \
                None
        return None, f"{why}|snapshot_no_identity"
    except urllib.error.HTTPError as e:
        return None, f"{why}|snapshot_http_{e.code}"
    except Exception as e:
        return None, f"{why}|snapshot_{type(e).__name__}"


n_done = [0]


def work(item):
    row, why = resolve(item)
    with lock:
        n_done[0] += 1
        if row:
            rows.append(row)
        else:
            (transient if is_transient(why.split("|")[-1]) else failed)[str(item[0])] = why
            miss[why.split("|")[-1]] += 1
        if len(rows) >= FLUSH or n_done[0] % 500 == 0:
            flush_locked()
            print(f"  {n_done[0]:,}/{len(PEND):,} kept={part*FLUSH + len(rows):,} "
                  f"miss={dict(miss.most_common(6))} ({time.time()-t0:.0f}s)", flush=True)


with ThreadPoolExecutor(WORKERS) as ex:
    list(ex.map(work, PEND))
with lock:
    flush_locked()
validate()

kept = sum(pq.read_table(fp).num_rows for fp in glob.glob(f"{OUT}/*.parquet"))
rec = {"schema": f"URI_RESOLVE_{MODE.upper()}/v2", "generated_utc": now(),
       "APPEND_ONLY": "writes only under _acquisition/; frozen artifacts untouched.",
       "mode": MODE, "arg": ARG, "workers": WORKERS, "archive_gap_s": ARCHIVE_GAP,
       "FLOW": "exact URL -> live redirect chain -> structured page identity -> snapshot if dead",
       "SNAPSHOT_PREFERENCE": "closest capture to 2010-01-01, contemporaneous with the key",
       "REJECTIONS": "redirect off the page, site-name/generic/interstitial titles, collapsed redirects (see validate)",
       "urls_attempted": len(PEND), "resolved_rows_all_runs_after_validation": kept,
       "misses_this_run": dict(miss), "transient_pending_retry": len(transient),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_URI_RESOLVE_{MODE.upper()}.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\n{MODE}: attempted {len(PEND):,}; hits on disk {kept:,}; misses {dict(miss)}; "
      f"transient {len(transient):,}")
print(f"{time.time()-t0:.0f}s")
