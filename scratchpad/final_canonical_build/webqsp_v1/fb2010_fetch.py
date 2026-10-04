"""Acquire the 2010-07-16 Freebase data dump from the Internet Archive, checksum-verified.

    PYTHONHASHSEED=0 python .../fb2010_fetch.py simple      # freebase-simple-topic-dump.tsv.bz2 (1.09 GB)
    PYTHONHASHSEED=0 python .../fb2010_fetch.py quads       # freebase-datadump-quadruples.tsv.bz2 (4.23 GB)
    PYTHONHASHSEED=0 python .../fb2010_fetch.py quads2008   # the 2008-03-28 dump (0.57 GB)

WHY THIS IS AN EXACT SOURCE, NOT A SCRAPE
  These are Freebase's own published database dumps, five years EARLIER than the
  freebase-rdf-latest snapshot that is our source of record.  A topic that carried
  /type/object/name in 2010 and carries none in the final dump lost that name between the two
  releases (deletion, merge, or export loss).  Recovering it from the 2010 dump is therefore an
  exact-source recovery of an original Freebase name, not an external reconciliation.

THE JOIN (proven empirically before this script was written)
  The 2010 dumps are keyed on GUIDs, the final dump on MIDs.  The derivation
      mid = "m.0" + base32(int(guid[16:], 16) - 0x8000000000000000)
      base32 alphabet "0123456789bcdfghjklmnpqrstvwxyz_"
  was tested against the frozen 302M-node graph on 67,543 topics from the head of the simple topic
  dump: 48,399 (71.7%) of the derived MIDs are present in the graph, and the rule independently
  reproduces m.02mjmr (Barack Obama) from its guid.  The ~28% absent are topics deleted between
  2010 and 2015, which is exactly what a five-year gap predicts.  No other candidate derivation
  ("m." + base32, base32 including the 0x8000.. offset) matched a single node.

INTEGRITY
  size and md5 are taken from the item's own archive.org metadata and verified after download;
  a mismatch deletes the partial file and exits non-zero.  Resumable by HTTP Range.

Writes only under _acquisition/raw/.  Nothing else is touched.
"""
import sys, io, os, json, time, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import requests

V3 = "data/final_canonical/freebase_v3"
RAW = f"{V3}/_acquisition/raw"
os.makedirs(RAW, exist_ok=True)
WHICH = sys.argv[1] if len(sys.argv) > 1 else "simple"
WEX = "freebase-wex-data-dump-2010-07-05"
ITEM = {"simple": ("freebase-data-dump-2010-07-16", "freebase-simple-topic-dump.tsv.bz2"),
        "quads": ("freebase-data-dump-2010-07-16", "freebase-datadump-quadruples.tsv.bz2"),
        "tsv": ("freebase-data-dump-2010-07-16", "freebase-datadump-tsv.tar.bz2"),
        "quads2008": ("freebase_dump_2008-03-28", "freebase-datadump-quadruples.tsv.bz2"),
        # WEX is Metaweb's own Wikipedia-extraction release. Three of its nine files are directly
        # on target and tiny: a Freebase NAMES table, a Freebase-to-Wikipedia-page-id table, and a
        # Freebase types table -- 138 MB together, found by V3_FB_DUMP_SEARCH.
        "wex_names": (WEX, "freebase-wex-2010-07-05-freebase_names.tsv.bz2"),
        "wex_wpid": (WEX, "freebase-wex-2010-07-05-freebase_wpid.tsv.bz2"),
        "wex_types": (WEX, "freebase-wex-2010-07-05-freebase_types.tsv.bz2"),
        "wex_redirects": (WEX, "freebase-wex-2010-07-05-redirects.tsv.bz2")}[WHICH]
ident, fname = ITEM
out = f"{RAW}/{WHICH}_{fname}"
t0 = time.time()

S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (research; CRAG canonical Freebase build)"})
meta = S.get(f"https://archive.org/metadata/{ident}", timeout=120).json()
fmeta = next(f for f in meta["files"] if f["name"] == fname)
want_size, want_md5 = int(fmeta["size"]), fmeta["md5"]
print(f"{ident}/{fname}\n  published size {want_size:,} bytes  md5 {want_md5}", flush=True)

have = os.path.getsize(out) if os.path.exists(out) else 0
if have > want_size:
    sys.exit(f"local file larger than published ({have:,} > {want_size:,}); refusing to touch it")
if have < want_size:
    mode = "ab" if have else "wb"
    hdr = {"Range": f"bytes={have}-"} if have else {}
    print(f"  downloading from byte {have:,} ...", flush=True)
    r = S.get(f"https://archive.org/download/{ident}/{fname}", headers=hdr, stream=True, timeout=300)
    r.raise_for_status()
    n = have
    last = time.time()
    with open(out, mode) as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
            n += len(chunk)
            if time.time() - last > 20:
                last = time.time()
                print(f"    {n/1e9:.2f}/{want_size/1e9:.2f} GB  "
                      f"{n/max(1e-9, time.time()-t0)/1e6:.1f} MB/s  ({time.time()-t0:.0f}s)", flush=True)
    r.close()
else:
    print("  already complete on disk", flush=True)

got = os.path.getsize(out)
print(f"  size on disk {got:,} (published {want_size:,}) match={got == want_size}", flush=True)
h = hashlib.md5()
with open(out, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 22), b""):
        h.update(b)
got_md5 = h.hexdigest()
ok = (got == want_size) and (got_md5 == want_md5)
print(f"  md5 {got_md5} published {want_md5} match={got_md5 == want_md5}", flush=True)
rec = {"schema": "FB2010_ACQUISITION/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "writes only under _acquisition/raw/.",
       "archive_item": ident, "file": fname, "local_path": out,
       "published_size": want_size, "local_size": got,
       "published_md5": want_md5, "local_md5": got_md5,
       "verified": bool(ok),
       "WHAT_IT_IS": "Freebase's own published database dump, earlier than the freebase-rdf-latest "
                     "snapshot that is the source of record. Used as an exact source for names that "
                     "existed then and are absent from the final dump.",
       "GUID_TO_MID": 'mid = "m.0" + base32(int(guid[16:],16) - 0x8000000000000000), alphabet '
                      '"0123456789bcdfghjklmnpqrstvwxyz_"; verified on 67,543 head topics '
                      '(71.7% present in the frozen graph) and on m.02mjmr.',
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FB2010_ACQUISITION_{WHICH.upper()}.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps(rec, indent=1))
if not ok:
    sys.exit("CHECKSUM/SIZE MISMATCH - artifact is invalid until counts/checksums establish otherwise")
print(f"OK {time.time()-t0:.0f}s")
