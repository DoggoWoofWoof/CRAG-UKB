"""Is there any Freebase dump between 2010-07-16 and freebase-rdf-latest?  Ask the archive.

    PYTHONHASHSEED=0 python .../fb_dump_search.py

WHY IT MATTERS MORE THAN ANYTHING ELSE IN THE QUEUE
  The 2010-07-16 topic dump alone recovered 167,487 residue names -- more than the entire prior
  campaign -- because a published Freebase dump is an exact source and covers everything at once.
  Google published quarterly RDF dumps through 2013 and 2014.  A 2013 or 2014 release would sit far
  closer to the 2015 snapshot than 2010 does, so it would cover far more of the residue.  If one
  survives anywhere on the Internet Archive it is worth more than every remaining scraping job put
  together, so this asks the archive directly instead of assuming the earlier negative result.

  Two independent questions are asked, because they fail differently:
    1. advancedsearch  -- is there an ITEM whose metadata mentions a Freebase dump?
    2. per-item file listing -- for every hit, what files does it actually hold, and how big?
  An item can exist with no usable file, and a file can exist in an item whose title says nothing.

Read-only.  Writes one record, downloads nothing.
OUTPUT  V3_FB_DUMP_SEARCH.json
"""
import sys, io, os, json, time, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import requests

V3 = "data/final_canonical/freebase_v3"
t0 = time.time()
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (research; CRAG canonical Freebase build)"})

QUERIES = ["freebase dump", "freebase rdf", "freebase datadump", "freebase quadruples",
           "title:(freebase)", "freebase-rdf", "freebase triples", "freebase data"]
SEEN = {}
for q in QUERIES:
    try:
        r = S.get("https://archive.org/advancedsearch.php",
                  params={"q": q, "fl[]": ["identifier", "title", "date", "publicdate", "item_size"],
                          "rows": 100, "page": 1, "output": "json"}, timeout=90)
        docs = r.json().get("response", {}).get("docs", [])
    except Exception as e:
        print(f"  query {q!r} failed: {type(e).__name__}", flush=True)
        continue
    for d in docs:
        SEEN.setdefault(d["identifier"], d)
    print(f"  {q!r}: {len(docs)} hits (total distinct {len(SEEN)})  ({time.time()-t0:.0f}s)", flush=True)

INTERESTING = re.compile(r"(rdf|quadruple|datadump|dump|triple|tsv|nt|gz|bz2)", re.I)
items = []
for ident, d in sorted(SEEN.items()):
    try:
        meta = S.get(f"https://archive.org/metadata/{ident}", timeout=90).json()
    except Exception as e:
        print(f"  metadata {ident} failed: {type(e).__name__}", flush=True)
        continue
    files = [{"name": f.get("name"), "size": int(f.get("size", 0) or 0), "md5": f.get("md5")}
             for f in meta.get("files", [])
             if INTERESTING.search(f.get("name", "")) and int(f.get("size", 0) or 0) > 10_000_000]
    if not files:
        continue
    files.sort(key=lambda f: -f["size"])
    items.append({"identifier": ident, "title": str(d.get("title", ""))[:160],
                  "date": d.get("date") or d.get("publicdate"),
                  "n_big_files": len(files),
                  "files": files[:12]})
    print(f"  {ident:<46} {str(d.get('date'))[:10]:<12} "
          f"{sum(f['size'] for f in files)/1e9:8.2f} GB  {len(files)} files", flush=True)

items.sort(key=lambda i: -sum(f["size"] for f in i["files"]))
rec = {"schema": "FB_DUMP_SEARCH/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "read-only survey; nothing downloaded, nothing frozen touched.",
       "QUESTION": "does any Freebase dump exist between 2010-07-16 and freebase-rdf-latest (2015)?",
       "queries": QUERIES,
       "distinct_items_seen": len(SEEN),
       "items_with_a_large_dump_like_file": len(items),
       "items": items,
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FB_DUMP_SEARCH.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\n{len(items)} items carry a dump-like file >10 MB; record written ({time.time()-t0:.0f}s)")
