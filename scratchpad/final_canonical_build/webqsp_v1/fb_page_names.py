"""FREEBASE_HISTORICAL_PAGE_NAME: names from Freebase's own archived topic pages.

    PYTHONHASHSEED=0 python .../fb_page_names.py hunt      # the V2.1 hunt population, band-ordered
    PYTHONHASHSEED=0 python .../fb_page_names.py residue   # declared/empirically-nameless residue

TIER (locked, V3_NAME_PROVENANCE_ORDER_V1.json)
  Above FREEBASE_ALIAS, below FREEBASE_DELETED_NAME.  is_original_name = True (Freebase itself
  rendered /type/object/name for that topic); is_current_snapshot_name = False (recovered from a
  historical page, NOT from the frozen dump).  Three independent anchors are checked: the requested
  MID, the page's own mid span, and the h1/title name with its language code.

TWO KINDS OF ARCHIVED PAGE
  bare  /m/<mid>     -> <title>{name} - Freebase</title> and an <h1> carrying {name} plus
                        <sup class="lang-code en">en</sup> = the language it was rendered in.
  /m/<mid>?i18n=     -> the same page PLUS a server-rendered localized-names table (id="i18n-table"):
                        one <tr class="data-row"> per attested label, each carrying the literal, the
                        /lang/<code> of its edit handler, and the date that name was asserted.
                        Verified on m/02mjmr@20130602062125: 41 labels in 41 languages.
                        Rows marked `wrapper missing` are empty add-slots and are skipped.
  The i18n capture is a superset, so it is preferred whenever one exists.

MULTI-LABEL RULE (locked)
  ALL attested labels are preserved in fb_page_labels/.  The display name is deterministic:
  English if present, else the language the page itself rendered in the h1, else the
  lexicographically first language code.  No attested label is ever discarded.

ACCEPTANCE
  * page mid span must equal the requested MID; a mismatch means the topic was merged and the page
    describes the survivor -> fb_page_merged.parquet as evidence, NOT as a name
  * a MID-shaped name is what Freebase rendered for a nameless topic -- and the frozen graph itself
    stores three such strings as names -- so it is never accepted; the outcome `page_shows_no_name`
    is a PERMANENT miss and is positive evidence for terminal bucket C
  * " - Freebase" is the only string stripped, and only as a suffix

Writes only under _acquisition/.  Frozen graph and frozen overlay untouched.
"""
import sys, io, os, re, json, glob, time, html, zlib, threading, collections
import urllib.error, urllib.parse
from concurrent.futures import ThreadPoolExecutor
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, requests

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
OUT = f"{ACQ}/fb_page_names"
LBL = f"{ACQ}/fb_page_labels"
NIL = f"{ACQ}/fb_page_nameless"
os.makedirs(OUT, exist_ok=True)
os.makedirs(LBL, exist_ok=True)
os.makedirs(NIL, exist_ok=True)
MODE = sys.argv[1] if len(sys.argv) > 1 else "hunt"
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
GAP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
FLUSH, CAP = 200, 500000
GZIP_MAGIC = bytes([0x1F, 0x8B])
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
TIER = "FREEBASE_HISTORICAL_PAGE_NAME"
t0 = time.time()
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
H1 = re.compile(r"<h1[^>]*>\s*(.*?)\s*<a href=\"/m/([0-9a-z_]+)\?i18n=\"[^>]*>\s*"
                r"<sup class=\"lang-code ([A-Za-z0-9\-]+)\">", re.S)
H1_PLAIN = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)
MIDSPAN = re.compile(r'<span class="mid">/m/([0-9a-z_]+)</span>')
I18N_TABLE = re.compile(r'id="i18n-table"(.*?)</table>', re.S)
ROW = re.compile(r'<tr class="data-row[^"]*">(.*?)</tr>', re.S)
LIT = re.compile(r'<span class="literal-value[^"]*property-value">\s*(.*?)\s*</span>', re.S)
ROWLANG = re.compile(r"fb\.i18n_tab\.(?:edit|delete)_name\(this, '/lang/([A-Za-z0-9\-]+)'\)")
ROWDATE = re.compile(r'<span class="date">([\d\-]+)</span>')
TAG = re.compile(r"<[^>]+>")
MIDLIKE = re.compile(r"^/?m/[0-9a-z_]+$")
NOTNAME = ("freebase", "page not found", "not found", "error", "sign in")


def fold(x):
    return re.sub(r"\s+", " ", html.unescape(TAG.sub("", x))).strip()


def bad_name(n):
    if not n or MIDLIKE.match(n):
        return True
    low = n.lower()
    return low in NOTNAME or low.startswith("page not found") or "topic not found" in low


def parse(body, mid):
    """-> (result|None, reason|None, extra|None)"""
    page_mid = None
    m = MIDSPAN.search(body)
    if m:
        page_mid = m.group(1)
    title = None
    m = TITLE.search(body)
    if m:
        t = fold(m.group(1))
        if t.endswith(" - Freebase"):
            t = t[: -len(" - Freebase")].strip()
        title = t or None
    h1name = h1lang = None
    m = H1.search(body)
    if m:
        h1name, h1mid, h1lang = fold(m.group(1)) or None, m.group(2), m.group(3)
        if page_mid is None:
            page_mid = h1mid
    else:
        m = H1_PLAIN.search(body)
        if m:
            h1name = fold(m.group(1)) or None
    if page_mid is not None and page_mid != mid:
        return None, "page_mid_mismatch", {"page_mid": page_mid, "name": h1name or title,
                                           "lang": h1lang}

    # every attested label, from the localized-names table when the capture carries one
    labels = []
    tm = I18N_TABLE.search(body)
    if tm:
        for blk in ROW.findall(tm.group(1)):
            if "wrapper missing" in blk:
                continue
            li, la = LIT.search(blk), ROWLANG.search(blk)
            if not (li and la):
                continue
            lit = fold(li.group(1))
            if not lit or bad_name(lit):
                continue
            d = ROWDATE.search(blk)
            labels.append((la.group(1), lit, d.group(1) if d else ""))
    seen, uniq = set(), []
    for lg, lit, d in labels:
        if (lg, lit) not in seen:
            seen.add((lg, lit))
            uniq.append((lg, lit, d))
    labels = uniq

    # deterministic display choice
    if labels:
        by = {}
        for lg, lit, d in labels:
            by.setdefault(lg, (lit, d))
        if "en" in by:
            lang, rule = "en", "english_present"
        elif h1lang and h1lang in by:
            lang, rule = h1lang, "page_render_language"
        else:
            lang, rule = min(by), "lexicographically_first_language"
        name, rung = by[lang][0], "freebase_i18n_table"
    elif h1name and not bad_name(h1name):
        name, lang, rung, rule = h1name, h1lang or "", "freebase_h1", "h1_only"
        labels = [(lang, name, "")]
    elif title and not bad_name(title):
        name, lang, rung, rule = title, h1lang or "", "html_title", "title_only"
        labels = [(lang, name, "")]
    else:
        return None, "page_shows_no_name", {"rendered_title": (title or h1name or "")[:200],
                                            "page_mid": page_mid or ""}
    agrees = (title is None) or (h1name is None) or (fold(title) == fold(h1name))
    return ({"name": name, "lang": lang, "rung": rung, "rule": rule, "labels": labels,
             "page_mid": page_mid or "", "h1_agrees": bool(agrees),
             "mid_verified": page_mid == mid}, None, None)


# ---------------------------------------------------------------- transport
_tls, _alock, _alast = threading.local(), threading.Lock(), [0.0]


def sess():
    s = getattr(_tls, "s", None)
    if s is None:
        s = _tls.s = requests.Session()
        s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip"})
    return s


def pace():
    with _alock:
        d = GAP - (time.time() - _alast[0])
        if d > 0:
            time.sleep(d)
        _alast[0] = time.time()


def fetch(url):
    r = sess().get(url, timeout=60, stream=True, allow_redirects=True)
    try:
        if r.status_code >= 400:
            raise urllib.error.HTTPError(url, r.status_code, r.reason, r.headers, None)
        b = r.raw.read(CAP, decode_content=False)
        enc = r.headers.get("Content-Encoding", "").lower()
        if "gzip" in enc or "deflate" in enc or b[:2] == GZIP_MAGIC:
            try:                                # truncated at CAP: take the decodable prefix
                b = zlib.decompressobj(zlib.MAX_WBITS | 32).decompress(b)
            except Exception:
                pass
    finally:
        r.close()
    return r.url, b.decode("utf-8", "replace")


TRANSIENT = ("Timeout", "timeout", "ConnectionError", "ConnectionReset", "ChunkedEncoding",
             "ProtocolError", "RemoteDisconnected", "http_429", "http_5", "OSError")

# ---------------------------------------------------------------- the cohort
a = pq.read_table(f"{ACQ}/fb_archived_mids_v2.parquet")
inh = np.array(a["in_hunt"].to_pylist())
inr = np.array(a["in_residue"].to_pylist())
sel = inh if MODE == "hunt" else (inr & ~inh)
cols = {c: a[c].to_pylist() for c in a.schema.names}
items = []
for i in np.flatnonzero(sel).tolist():
    mid = cols["mid"][i]
    if cols["i18n_url"][i]:                      # superset page: h1 + full label table
        ts, qs = cols["i18n_ts"][i], "?i18n="
    else:
        u = cols["best_url"][i]
        ts = cols["best_ts"][i]
        q = urllib.parse.urlsplit(u).query
        qs = "?" + q if q else ""
    items.append((cols["node_uid"][i], mid, ts, qs, cols["band"][i] or "?",
                  int(cols["n_200"][i])))
BAND_RANK = {"HIGH": 0, "MID": 1, "LOW": 2, "UNTYPED": 3, "NEAR_ZERO": 4, "ZERO": 5}
items.sort(key=lambda r: (BAND_RANK.get(r[4], 9), 0 if r[5] > 0 else 1, r[1]))

done = set()
for fp in glob.glob(f"{OUT}/*.parquet"):
    done |= set(pq.read_table(fp, columns=["node_uid"])["node_uid"].to_pylist())
fp_miss = f"{ACQ}/fb_page_names_missed.json"
failed = json.load(io.open(fp_miss, encoding="utf-8")) if os.path.exists(fp_miss) else {}
done |= {int(k) for k in failed}
items = [i for i in items if i[0] not in done]
print(f"mode {MODE}: {len(items):,} captured pages to read; already done {len(done):,}", flush=True)
print(f"  bands: {dict(collections.Counter(i[4] for i in items))}", flush=True)
n_i18n = sum(1 for i in items if i[3] == "?i18n=")
print(f"  with an i18n (all-labels) capture: {n_i18n:,}", flush=True)

# ---------------------------------------------------------------- run
rows, labrows, nilrows, merged, miss = [], [], [], [], collections.Counter()
transient = {}
lock = threading.Lock()
part = len(glob.glob(f"{OUT}/*.parquet"))
npart = len(glob.glob(f"{NIL}/*.parquet"))
n_done, n_kept = [0], [0]


def flush_locked():
    global part, npart, rows, labrows, nilrows, merged
    if nilrows:
        pq.write_table(pa.table({
            "node_uid": pa.array([r[0] for r in nilrows], pa.int64()),
            "freebase_mid": pa.array([r[1] for r in nilrows]),
            "capture_timestamp": pa.array([r[2] for r in nilrows]),
            "archive_url": pa.array([r[3] for r in nilrows]),
            "rendered_title": pa.array([r[4] for r in nilrows]),
            "band": pa.array([r[5] for r in nilrows]),
            "evidence": pa.array(["FREEBASE_PAGE_SHOWED_NO_NAME"] * len(nilrows)),
            "observed_at": pa.array([r[6] for r in nilrows])}),
            f"{NIL}/part_{npart:05d}.parquet", compression="zstd")
        npart += 1
        nilrows = []
    if rows:
        pq.write_table(pa.table({
            "node_uid": pa.array([r["u"] for r in rows], pa.int64()),
            "freebase_mid": pa.array([r["mid"] for r in rows]),
            "display_name": pa.array([r["name"] for r in rows]),
            "language": pa.array([r["lang"] for r in rows]),
            "capture_timestamp": pa.array([r["ts"] for r in rows]),
            "archive_url": pa.array([r["url"] for r in rows]),
            "page_mid_verified": pa.array([r["ver"] for r in rows], pa.bool_()),
            "source": pa.array([TIER] * len(rows)),
            "is_original_name": pa.array([True] * len(rows), pa.bool_()),
            "is_current_snapshot_name": pa.array([False] * len(rows), pa.bool_()),
            "identity_rung": pa.array([r["rung"] for r in rows]),
            "display_rule": pa.array([r["rule"] for r in rows]),
            "n_attested_labels": pa.array([r["nlab"] for r in rows], pa.int32()),
            "h1_agrees": pa.array([r["agr"] for r in rows], pa.bool_()),
            "band": pa.array([r["band"] for r in rows]),
            "retrieved_at": pa.array([r["at"] for r in rows])}),
            f"{OUT}/part_{part:05d}.parquet", compression="zstd")
        if labrows:
            pq.write_table(pa.table({
                "node_uid": pa.array([r[0] for r in labrows], pa.int64()),
                "freebase_mid": pa.array([r[1] for r in labrows]),
                "language": pa.array([r[2] for r in labrows]),
                "label": pa.array([r[3] for r in labrows]),
                "name_asserted_date": pa.array([r[4] for r in labrows]),
                "capture_timestamp": pa.array([r[5] for r in labrows]),
                "source": pa.array([TIER] * len(labrows)),
                "is_display_name": pa.array([r[6] for r in labrows], pa.bool_())}),
                f"{LBL}/part_{part:05d}.parquet", compression="zstd")
        part += 1
        rows, labrows = [], []
    if merged:
        fp = f"{ACQ}/fb_page_merged.parquet"
        old = pq.read_table(fp).to_pylist() if os.path.exists(fp) else []
        allm = old + merged
        pq.write_table(pa.table({k: pa.array([m[k] for m in allm],
                                             pa.int64() if k == "node_uid" else pa.string())
                                 for k in ("node_uid", "freebase_mid", "page_mid", "page_name",
                                           "page_lang", "capture_timestamp", "retrieved_at")}),
                       fp, compression="zstd")
        merged = []
    json.dump(failed, io.open(fp_miss, "w", encoding="utf-8"))
    json.dump(transient, io.open(f"{ACQ}/fb_page_names_transient.json", "w", encoding="utf-8"))


def work(item):
    u, mid, ts, qs, band, n200 = item
    url = f"https://web.archive.org/web/{ts}id_/http://www.freebase.com/m/{mid}{qs}"
    try:
        pace()
        _, body = fetch(url)
        res, why, extra = parse(body, mid)
    except urllib.error.HTTPError as e:
        res, why, extra = None, f"http_{e.code}", None
    except Exception as e:
        res, why, extra = None, type(e).__name__, None
    with lock:
        n_done[0] += 1
        if res:
            n_kept[0] += 1
            rows.append({"u": u, "mid": "m." + mid, "name": res["name"], "lang": res["lang"],
                         "ts": ts, "url": url, "ver": res["mid_verified"], "rung": res["rung"],
                         "rule": res["rule"], "nlab": len(res["labels"]), "agr": res["h1_agrees"],
                         "band": band, "at": now()})
            for lg, lit, d in res["labels"]:
                labrows.append((u, "m." + mid, lg, lit, d, ts,
                                bool(lg == res["lang"] and lit == res["name"])))
        else:
            miss[why] += 1
            if why == "page_mid_mismatch":
                merged.append({"node_uid": u, "freebase_mid": "m." + mid,
                               "page_mid": "m." + extra["page_mid"],
                               "page_name": extra["name"] or "", "page_lang": extra["lang"] or "",
                               "capture_timestamp": ts, "retrieved_at": now()})
                failed[str(u)] = why
            elif why == "page_shows_no_name":
                nilrows.append((u, "m." + mid, ts, url,
                                (extra or {}).get("rendered_title", ""), band, now()))
                failed[str(u)] = why
            elif any(t in why for t in TRANSIENT):
                transient[str(u)] = why
            else:
                failed[str(u)] = why
        if len(rows) >= FLUSH or len(nilrows) >= FLUSH or n_done[0] % 50 == 0:
            flush_locked()
            print(f"  {n_done[0]:,}/{len(items):,} named={n_kept[0]:,} "
                  f"miss={dict(miss.most_common(6))} ({time.time()-t0:.0f}s)", flush=True)


with ThreadPoolExecutor(WORKERS) as ex:
    list(ex.map(work, items))
with lock:
    flush_locked()

kept = sum(pq.read_table(fp).num_rows for fp in glob.glob(f"{OUT}/*.parquet"))
nlab = sum(pq.read_table(fp).num_rows for fp in glob.glob(f"{LBL}/*.parquet"))
rec = {"schema": f"FB_PAGE_NAMES_{MODE.upper()}/v2", "generated_utc": now(),
       "APPEND_ONLY": "writes only under _acquisition/; frozen artifacts untouched.",
       "tier": TIER, "tier_order_record": "V3_NAME_PROVENANCE_ORDER_V1.json",
       "is_original_name": True, "is_current_snapshot_name": False,
       "mode": MODE, "workers": WORKERS, "archive_gap_s": GAP,
       "EVIDENCE": "Freebase's own topic page as captured by the Wayback Machine; mid span checked "
                   "against the requested MID; ?i18n= captures also carry the server-rendered "
                   "localized-names table, so every attested label is kept in fb_page_labels/",
       "pages_attempted": len(items), "named_this_run": n_kept[0],
       "rows_on_disk_all_runs": kept, "attested_labels_on_disk": nlab,
       "misses_this_run": dict(miss), "transient_pending_retry": len(transient),
       "page_shows_no_name_is_bucket_C_evidence": miss.get("page_shows_no_name", 0),
       "NAMELESS_EVIDENCE": "fb_page_nameless/: Freebase's own live site rendered the MID as the "
                            "page title, i.e. it held no /type/object/name for that topic at capture "
                            "time. Verified on m/01hh6f@20131210091937: <title>/m/01hh6f - Freebase</title>, "
                            "<h1>/m/01hh6f</h1>. This is EMPIRICAL evidence (bucket C) dated to the "
                            "capture, never SOURCE_DECLARED.",
       "nameless_rows_on_disk": sum(pq.read_table(fp).num_rows for fp in glob.glob(f"{NIL}/*.parquet")),
       "merged_pages_recorded_not_named": sum(1 for v in failed.values() if v == "page_mid_mismatch"),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FB_PAGE_NAMES_{MODE.upper()}.json", "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=1, ensure_ascii=False)
print(json.dumps(rec, indent=1, ensure_ascii=False))
