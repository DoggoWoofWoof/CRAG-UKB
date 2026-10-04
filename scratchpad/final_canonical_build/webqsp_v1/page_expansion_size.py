"""Size the archived-Freebase-page pool beyond the current hunt list. Decision support, no crawling.

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/page_expansion_size.py

WHY
  The running hunt crawls 6,540 MIDs drawn from the hunt population.  V3_FB_ARCHIVE_INDEX_V2 records
  38,295 RESIDUE mids with a capture of some kind, so most of the archived pool is not being
  crawled.  Archived Freebase topic pages are a HISTORICAL_PAGE source -- one of the three members
  of the FREEBASE_HISTORICAL_ASSERTION class -- so this is part of the exact-source sweep, not
  generic web recovery, and it is the last part of that sweep with any headroom left.

  What it costs is the question.  The hunt's own numbers are the only honest basis for projecting
  yield, and they are unflattering: most captures render a page that shows no name at all.  So this
  pass measures the pool and projects from the observed rate.  It crawls nothing.

WHY 200-ONLY IS THE RIGHT DENOMINATOR
  A capture with a non-200 status is a redirect or an error page and cannot carry a topic name.  The
  index groups every URL variant per MID, so a MID counts as reachable if ANY of its captures is a
  200.  That is the pool an expanded hunt could actually draw from.
"""
import sys, io, os, re, json, glob, time, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow.parquet as pq

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
t0 = time.time()
RES = np.load(f"{ACQ}/_unresolved_population.npz")["U"]
# THE DOT/SLASH TRAP, THIRD OCCURRENCE. The archived URLs are BOTH spellings and carry an explicit
# port: 'http://www.freebase.com:80/m/0abc' and 'http://www.freebase.com:80/m.01vzz1c'. A pattern
# anchored on 'freebase.com/' with a slash-form mid silently dropped most of the index -- it scored
# 280,819 mids against the 441,454 the archive index recorded, and 9,212 residue hits against
# 38,295. Both spellings, and an optional port, are matched here.
MIDPAT = re.compile(r"freebase\.com(?::\d+)?/(m[./][0-9a-z_]{2,})")

any_cap, ok_cap = set(), set()
files = sorted(glob.glob(f"{ACQ}/fb_cdx/*.parquet"))
for i, p in enumerate(files):
    t = pq.read_table(p, columns=["original", "statuscode"])
    for u, s in zip(t["original"].to_pylist(), t["statuscode"].to_pylist()):
        m = MIDPAT.search(u or "")
        if not m:
            continue
        g = m.group(1)
        mid = g[0] + "." + g[2:]
        any_cap.add(mid)
        if str(s) == "200":
            ok_cap.add(mid)
    if (i + 1) % 60 == 0:
        print(f"  cdx {i+1}/{len(files)}  any={len(any_cap):,} ok={len(ok_cap):,} "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"cdx: {len(any_cap):,} distinct mids, {len(ok_cap):,} with a 200 ({time.time()-t0:.0f}s)",
      flush=True)
if not any_cap:
    sys.exit("POSITIVE CONTROL FAILED: parsed 0 mids out of the cdx index. Refusing to size a zero.")


def on_residue(s):
    if not s:
        return set()
    lst = sorted(s)
    u = np.fromiter((hash(x) for x in lst), np.int64, len(lst))
    o = np.argsort(u, kind="stable")
    su = u[o]
    j = np.searchsorted(RES, su)
    hit = np.take(RES, np.minimum(j, len(RES) - 1)) == su
    return {lst[k] for k in o[hit]}


res_any, res_ok = on_residue(any_cap), on_residue(ok_cap)

# what the running hunt already covers
done = set()
for p in sorted(glob.glob(f"{ACQ}/fb_page_labels/*.parquet")):
    done |= set(pq.read_table(p, columns=["freebase_mid"])["freebase_mid"].to_pylist())
for p in (f"{ACQ}/fb_page_names_missed.json",):
    if os.path.exists(p):
        try:
            d = json.load(io.open(p, encoding="utf-8"))
            done |= set(d) if isinstance(d, dict) else set()
        except Exception:
            pass

# already named by the cascade -- a page can only ADD a name to a node that lacks one
named = set(pq.read_table(f"{ACQ}/cascade_names.parquet",
                          columns=["node_id"])["node_id"].to_pylist())
pool = res_ok - named
print(f"residue mids with any capture {len(res_any):,}   with a 200 {len(res_ok):,}", flush=True)
print(f"of those, still unnamed by the cascade: {len(pool):,}", flush=True)
print(f"already attempted by the running hunt (labelled or recorded as missed): {len(done):,}",
      flush=True)

RATE, SEC = 125 / 4050, 8224 / 4050            # observed by the running hunt
# SET DIFFERENCE, NOT COUNT SUBTRACTION. `done` is not a subset of `pool` -- it includes mids that
# are already named or that never had a 200 -- so len(pool) - len(done) is not the remainder and can
# even go negative. This is the same mistake the WEX record warns about two hours old, so it is
# spelled out rather than quietly fixed.
todo = pool - done
rem = len(todo)
rec = {
 "RECORD": "V3_PAGE_EXPANSION_SIZING",
 "WHAT": "how much headroom is left in the archived-Freebase-page source after the running hunt. "
         "Nothing was crawled; this pass reads the CDX index already on disk.",
 "WHY_IT_IS_IN_SCOPE": "archived Freebase topic pages are a HISTORICAL_PAGE source, one of the "
                       "three members of the FREEBASE_HISTORICAL_ASSERTION class. This is the "
                       "exact-source sweep, not generic web recovery.",
 "POOL": {"distinct_mids_in_cdx": len(any_cap), "with_a_200_capture": len(ok_cap),
          "on_the_residue_any_capture": len(res_any), "on_the_residue_with_a_200": len(res_ok),
          "of_those_still_unnamed_by_the_cascade": len(pool),
          "already_attempted_by_the_running_hunt": len(done),
          "NOT_YET_ATTEMPTED": rem},
 "WHY_200_ONLY": "a non-200 capture is a redirect or an error page and cannot carry a topic name. "
                 "A mid counts if ANY of its url variants captured 200.",
 "PROJECTION_FROM_THE_RUNNING_HUNT": {
    "observed": "125 names from 4,050 pages in 8,224s",
    "yield_rate_pct": round(RATE * 100, 2),
    "seconds_per_page": round(SEC, 2),
    "projected_new_names": int(rem * RATE),
    "projected_hours": round(rem * SEC / 3600, 1),
    "DOMINANT_MISS": "page_shows_no_name -- the capture exists and renders, but Freebase served no "
                     "name on it. That is the same finding the dump sources give: these objects "
                     "were nameless in the live system too."},
 "RECOMMENDATION_IS_NOT_A_DECISION": "this is a costing, not an approval. It is a large crawl for a "
                                     "small yield, it is the last headroom in the exact-source "
                                     "sweep, and whether to spend it is the user's call.",
 "elapsed_s": round(time.time() - t0, 1)}
b = json.dumps(rec, indent=1, ensure_ascii=False).encode()
rec["record_sha256"] = hashlib.sha256(b).hexdigest()
io.open(f"{V3}/V3_PAGE_EXPANSION_SIZING.json", "w", encoding="utf-8").write(
    json.dumps(rec, indent=1, ensure_ascii=False))
print(f"\nNOT YET ATTEMPTED: {rem:,} mids -> projected {int(rem*RATE):,} names in "
      f"{rem*SEC/3600:.1f} h at the observed rate")
print(f"wrote V3_PAGE_EXPANSION_SIZING.json  sha {rec['record_sha256'][:16]}")
