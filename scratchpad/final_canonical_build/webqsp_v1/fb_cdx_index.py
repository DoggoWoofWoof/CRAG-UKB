"""Historical Freebase topic pages: which residue MIDs does the Wayback Machine hold a page for?

    PYTHONHASHSEED=0 python .../fb_cdx_index.py

WHY THIS RUNG COMES FIRST IN THE HISTORICAL CHAIN
  http://www.freebase.com/m/<mid> was the topic page of exactly that MID, rendered by Freebase itself
  from the live graph, with the topic's name as the page title. A capture of it is Freebase's own
  statement of the name at capture time -- no external source, no name matching, no inference. The
  user's chain is: remaining MID -> historical Freebase page archives -> Wikipedia historical
  identifiers -> Common Crawl -> domain authority datasets -> archived KB mappings. This is step one.

WHAT IT DOES
  The CDX index for the prefix www.freebase.com/m/ is 231 pages (sized with showNumPages). Each page
  is downloaded once, collapsed per urlkey (earliest capture per distinct URL), and saved as-is under
  _acquisition/fb_cdx/ so the run is resumable page by page. Rows are then filtered to bare topic
  URLs -- scheme, optional www., optional :80, /m/<mid>, nothing else -- with status 200, the MID is
  hashed the way node_uid is (hash("m." + mid) under PYTHONHASHSEED=0) and looked up in the residue
  and in the V2 hunt population. Query variants (?props=, ?filter=...) are counted but not used: a
  filtered view of the topic is not the topic page.

OUTPUT (all under _acquisition/, append-only)
  fb_cdx/m_page_NNN.parquet          raw page rows: original, timestamp, statuscode
  fb_archived_mids.parquet           mid, node_uid, timestamp, statuscode, in_residue, in_hunt
  ../V3_FB_ARCHIVE_INDEX.json        counts

The snapshot fetch itself (title extraction, "<Name> - Freebase" template) is a separate stage; this
one only establishes, cheaply, the exact population the expensive stage should be spent on.
"""
import sys, io, os, re, json, time, glob, urllib.request, urllib.parse, urllib.error
import requests
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
OUT = f"{ACQ}/fb_cdx"
os.makedirs(OUT, exist_ok=True)
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
CDX = "https://web.archive.org/cdx/search/cdx"
TARGET = "www.freebase.com/m/"
GAP = 1.5
t0 = time.time()


S = requests.Session()
S.headers.update({"User-Agent": UA})


def req(url, timeout=120):
    """One kept-alive connection: archive.org drops new TCP connections when a client churns them."""
    r = S.get(url, timeout=timeout)
    if r.status_code >= 400:
        raise urllib.error.HTTPError(url, r.status_code, r.reason, None, None)
    return r.text


npages = int(req(f"{CDX}?url={urllib.parse.quote(TARGET, safe='')}&matchType=prefix&showNumPages=true").strip())
print(f"{TARGET}: {npages} CDX pages", flush=True)
last = 0.0
for page in range(npages):
    fp = f"{OUT}/m_page_{page:03d}.parquet"
    if os.path.exists(fp):
        continue
    for attempt in range(6):
        d = GAP - (time.time() - last)
        if d > 0:
            time.sleep(d)
        last = time.time()
        try:
            body = req(f"{CDX}?url={urllib.parse.quote(TARGET, safe='')}&matchType=prefix"
                       f"&fl=original,timestamp,statuscode&collapse=urlkey&page={page}")
            break
        except Exception as e:
            print(f"  page {page} attempt {attempt}: {type(e).__name__}", flush=True)
            time.sleep(10 * (attempt + 1))
    else:
        sys.exit(f"page {page} failed six times; rerun to resume")
    rows = [l.split(" ") for l in body.split("\n") if l.strip()]
    rows = [r for r in rows if len(r) == 3]
    pq.write_table(pa.table({"original": pa.array([r[0] for r in rows]),
                             "timestamp": pa.array([r[1] for r in rows]),
                             "statuscode": pa.array([r[2] for r in rows])}), fp, compression="zstd")
    if page % 10 == 0:
        print(f"  page {page}/{npages}: {len(rows):,} rows ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- filter to bare topic URLs
BARE = re.compile(r"^https?://(?:www\.)?freebase\.com(?::80)?/m/([0-9a-z_]+)/?$")
mids, ts, sc = [], [], []
n_rows = n_query = 0
for fp in sorted(glob.glob(f"{OUT}/m_page_*.parquet")):
    t = pq.read_table(fp)
    for o, a, b in zip(t["original"].to_pylist(), t["timestamp"].to_pylist(), t["statuscode"].to_pylist()):
        n_rows += 1
        m = BARE.match(o)
        if not m:
            n_query += ("?" in o)
            continue
        mids.append(m.group(1)); ts.append(a); sc.append(b)
print(f"rows {n_rows:,}; bare topic urls {len(mids):,}; query variants {n_query:,}", flush=True)

uid = np.fromiter((hash("m." + m) for m in mids), np.int64, len(mids))
U = np.load(f"{ACQ}/_unresolved_population.npz")["U"]
pos = np.searchsorted(U, uid); pos[pos >= len(U)] = 0
in_res = U[pos] == uid
# the V2 hunt population: everything not declared/measured nameless and not yet named
sk = pq.read_table(f"{ACQ}/semantic_kind_v2.parquet")
rc_col = next(c for c in sk.column_names if "recovery" in c)
rc = sk[rc_col].combine_chunks()
if pa.types.is_dictionary(rc.type):
    rc = rc.cast(pa.string())
HUNT = {"LIKELY_REAL_ENTITY", "LIKELY_CREATIVE_WORK", "LIKELY_MEDIA_ENTITY", "LIKELY_PLACE",
        "LIKELY_PERSON", "UNTYPED_CANDIDATE", "EXTERNAL_RESOURCE"}
mask = pc.is_in(rc, value_set=pa.array(sorted(HUNT)))
H = np.sort(pc.filter(sk["node_uid"], mask).to_numpy())
pos = np.searchsorted(H, uid); pos[pos >= len(H)] = 0
in_hunt = H[pos] == uid
ok = np.array([s == "200" for s in sc])
pq.write_table(pa.table({"mid": pa.array(mids), "node_uid": pa.array(uid, pa.int64()),
                         "timestamp": pa.array(ts), "statuscode": pa.array(sc),
                         "in_residue": pa.array(in_res), "in_hunt": pa.array(in_hunt)}),
               f"{ACQ}/fb_archived_mids.parquet", compression="zstd")
rec = {"schema": "FB_ARCHIVE_INDEX/v1", "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "target_prefix": TARGET, "cdx_pages": npages, "cdx_rows_collapsed_per_url": n_rows,
       "bare_topic_urls": len(mids), "query_variant_urls_ignored": n_query,
       "status_200": int(ok.sum()),
       "in_residue_69M": int(in_res.sum()), "in_residue_status_200": int((in_res & ok).sum()),
       "in_hunt_population_V2": int(in_hunt.sum()), "in_hunt_status_200": int((in_hunt & ok).sum()),
       "hunt_population_size": int(len(H)),
       "NOTE": "earliest capture per URL (collapse=urlkey); a 200 here means Freebase served that topic page at that time",
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FB_ARCHIVE_INDEX.json", "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=1)
print(json.dumps(rec, indent=1))
