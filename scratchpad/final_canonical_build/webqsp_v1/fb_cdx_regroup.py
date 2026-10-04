"""Re-group the 231 downloaded Freebase CDX pages BY MID instead of by URL.

    PYTHONHASHSEED=0 python .../fb_cdx_regroup.py

WHY
  fb_cdx_index.py kept only bare  http://www.freebase.com/m/<mid>  URLs (401,645) and discarded
  514,512 "query variants".  Those variants are not other pages -- they are other VIEWS of the same
  topic:  ?i18n= (the localized-names view, which lists every language label),  ?links= , ?props= ,
  ?writes= , ?filter=... .  Every one of them renders the same <h1> topic name, and ?i18n= renders
  ALL attested labels.  So the recoverable MID set is the union over URL variants, not the bare set.

  collapse=urlkey gave the EARLIEST capture per distinct URL, so each variant is an independent shot
  at a status-200 capture of the same MID.

OUTPUT (append-only; does not touch fb_archived_mids.parquet)
  fb_archived_mids_v2.parquet   mid, node_uid, in_residue, in_hunt, band, n_caps, n_200,
                                best_url, best_ts, best_status, i18n_url, i18n_ts
  V3_FB_ARCHIVE_INDEX_V2.json
"""
import sys, io, os, re, json, time, glob, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
t0 = time.time()

MID = re.compile(r"^https?://(?:www\.)?freebase\.com(?::80)?/m/([0-9a-z_]+?)/?(?:\?(.*))?$", re.I)

# rank of a capture for "which page do we fetch first": bare 200 wins, then i18n 200, then any 200
def rank(qs, status):
    ok = 0 if status == "200" else 4
    if qs is None:      v = 0        # bare topic page
    elif qs == "i18n=": v = 1        # localized names view
    elif qs == "":      v = 0
    else:               v = 2
    return ok * 8 + v

best = {}          # mid -> (rank, url, ts, status)
i18n = {}          # mid -> (ts, url)   any i18n capture, 200 preferred
caps = collections.Counter()
c200 = collections.Counter()
files = sorted(glob.glob(f"{ACQ}/fb_cdx/m_page_*.parquet"))
print(f"cdx pages on disk: {len(files)}", flush=True)
nrows = 0
for i, fp in enumerate(files):
    t = pq.read_table(fp)
    for url, ts, st in zip(t["original"].to_pylist(), t["timestamp"].to_pylist(), t["statuscode"].to_pylist()):
        nrows += 1
        m = MID.match(url or "")
        if not m:
            continue
        mid, qs = m.group(1), m.group(2)
        caps[mid] += 1
        if st == "200":
            c200[mid] += 1
        r = rank(qs, st)
        cur = best.get(mid)
        if cur is None or r < cur[0]:
            best[mid] = (r, url, ts, st)
        if qs == "i18n=":
            cur = i18n.get(mid)
            if cur is None or (st == "200" and cur[2] != "200"):
                i18n[mid] = (ts, url, st)
    if i % 40 == 0:
        print(f"  page {i}/{len(files)} rows {nrows:,} mids {len(best):,} ({time.time()-t0:.0f}s)", flush=True)
print(f"rows {nrows:,}; distinct MIDs {len(best):,}; MIDs with >=1 status-200 capture {len(c200):,}", flush=True)

mids = sorted(best)
uid = np.fromiter((hash("m." + m) for m in mids), np.int64, len(mids))

U = np.load(f"{ACQ}/_unresolved_population.npz")["U"]
p = np.searchsorted(U, uid); p[p >= len(U)] = 0
in_res = U[p] == uid

sk = pq.read_table(f"{ACQ}/semantic_kind_v2_1.parquet",
                   columns=["node_uid", "recovery_class", "set_named_rate_band"]) \
     if "set_named_rate_band" in pq.ParquetFile(f"{ACQ}/semantic_kind_v2_1.parquet").schema.names else \
     pq.read_table(f"{ACQ}/semantic_kind_v2_1.parquet")
cols = sk.schema.names
bandcol = "set_named_rate_band" if "set_named_rate_band" in cols else ("band" if "band" in cols else None)
HUNT = {"LIKELY_REAL_ENTITY", "UNTYPED_CANDIDATE", "LIKELY_PLACE", "LIKELY_MEDIA_ENTITY",
        "LIKELY_CREATIVE_WORK", "LIKELY_PERSON"}
rc = sk["recovery_class"]
rc = pc.cast(rc, pa.string()) if pa.types.is_dictionary(rc.type) else rc
hmask = pc.is_in(rc, value_set=pa.array(sorted(HUNT)))
hu = pc.filter(sk["node_uid"], hmask).to_numpy()
hb = (pc.filter(sk[bandcol], hmask) if bandcol else None)
hb = (pc.cast(hb, pa.string()) if (hb is not None and pa.types.is_dictionary(hb.type)) else hb)
hb = hb.to_pylist() if hb is not None else ["?"] * len(hu)
o = np.argsort(hu); hu = hu[o]; hb = [hb[i] for i in o.tolist()]
p2 = np.searchsorted(hu, uid); p2[p2 >= len(hu)] = 0
in_hunt = hu[p2] == uid
band = [hb[int(j)] if k else "" for j, k in zip(p2, in_hunt.tolist())]

tab = pa.table({
    "mid": pa.array(mids), "node_uid": pa.array(uid),
    "in_residue": pa.array(in_res), "in_hunt": pa.array(in_hunt), "band": pa.array(band),
    "n_caps": pa.array([caps[m] for m in mids], pa.int32()),
    "n_200": pa.array([c200.get(m, 0) for m in mids], pa.int32()),
    "best_url": pa.array([best[m][1] for m in mids]),
    "best_ts": pa.array([best[m][2] for m in mids]),
    "best_status": pa.array([best[m][3] for m in mids]),
    "i18n_url": pa.array([i18n[m][1] if m in i18n else None for m in mids]),
    "i18n_ts": pa.array([i18n[m][0] if m in i18n else None for m in mids]),
})
pq.write_table(tab, f"{ACQ}/fb_archived_mids_v2.parquet", compression="zstd")

hb_cnt = collections.Counter(b for b, k in zip(band, in_hunt.tolist()) if k)
h200 = int(sum(1 for m, k in zip(mids, in_hunt.tolist()) if k and c200.get(m, 0) > 0))
rec = {"schema": "FB_ARCHIVE_INDEX/v2",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "SUPERSEDES_NOTHING": "V3_FB_ARCHIVE_INDEX.json (bare-URL-only view) left as written; new record.",
       "CHANGE": "grouped by MID over ALL url variants (?i18n=, ?links=, ?props=, ?writes=, ?filter=), "
                 "which are alternate views of the same topic page, not other pages",
       "cdx_rows_scanned": nrows, "distinct_mids": len(best),
       "mids_with_200_capture": len(c200),
       "mids_with_i18n_capture": len(i18n),
       "in_residue_69M": int(in_res.sum()),
       "in_hunt_V2_1": int(in_hunt.sum()), "in_hunt_with_200": h200,
       "hunt_by_band": dict(hb_cnt.most_common()),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{ACQ}/V3_FB_ARCHIVE_INDEX_V2.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps(rec, indent=1, ensure_ascii=False), flush=True)
