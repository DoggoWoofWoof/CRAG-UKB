"""Join Metaweb's WEX 2010-07-05 release onto the frozen graph's unresolved residue.

    PYTHONHASHSEED=0 python .../wex_join.py

WHAT WEX IS
  WEX is Metaweb's own Wikipedia-extraction release, published eleven days before the 2010-07-16
  data dump.  Nine files; three of them are directly on target and tiny (138 MB together), and were
  found by asking the Internet Archive rather than assumed:
      freebase_names.tsv.bz2   guid <TAB> name              -- a Freebase NAME table, nothing else
      freebase_wpid.tsv.bz2    guid <TAB> wikipedia page id -- identity by NUMBER, never by name
      freebase_types.tsv.bz2   guid <TAB> /type/path        -- one row per type, repeated guids
  Same 32-hex guid form as the data dump, so the same verified derivation applies:
      mid = "m.0" + base32(int(guid[16:],16) - 0x8000000000000000)

WHY RUN IT WHEN THE 2010-07-16 DUMP IS ALREADY JOINED
  It is a DIFFERENT extraction on a DIFFERENT date from a DIFFERENT pipeline.  Overlap is expected
  to be large and is not assumed to be total: the campaign has already been burned once by adding
  per-source reaches that turned out to intersect completely, so the marginal gain over
  fb2010_names is measured here, not inferred.  Every row reports whether the 2010-07-16 dump
  already had that node.

TIER
  FREEBASE_HISTORICAL_DUMP_NAME, source_release 2010-07-05, source_rung "wex".  Same provisional
  placement question as the 2010-07-16 dump; the locked order is not edited by this script.

OUTPUT (append-only)
  _acquisition/wex_names/part_00000.parquet    residue nodes WEX names
  _acquisition/wex_struct/part_*.parquet       WEX types and wikipedia page ids for residue nodes
  V3_WEX_JOIN.json
"""
import sys, io, os, bz2, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
RAW = f"{ACQ}/raw"
OUT = f"{ACQ}/wex_names"
STR = f"{ACQ}/wex_struct"
for d in (OUT, STR):
    os.makedirs(d, exist_ok=True)
TIER = "FREEBASE_HISTORICAL_DUMP_NAME"
RELEASE = "2010-07-05"
AB = "0123456789bcdfghjklmnpqrstvwxyz_"
OFF = 0x8000000000000000
PREFIX = "9202a8c04000641f"
SRC = {"name": f"{RAW}/wex_names_freebase-wex-2010-07-05-freebase_names.tsv.bz2",
       "wikipedia_page_id": f"{RAW}/wex_wpid_freebase-wex-2010-07-05-freebase_wpid.tsv.bz2",
       "type": f"{RAW}/wex_types_freebase-wex-2010-07-05-freebase_types.tsv.bz2"}
t0 = time.time()
for k, v in SRC.items():
    if not os.path.exists(v):
        sys.exit(f"missing {v} -- run fb2010_fetch.py wex_* first")


def enc(n):
    s = ""
    while n:
        s = AB[n & 31] + s
        n >>= 5
    return s or "0"


U = np.load(f"{ACQ}/_unresolved_population.npz")["U"]
sk = pq.read_table(f"{ACQ}/semantic_kind_v2_1.parquet", columns=["node_uid", "recovery_class"])
rc = sk["recovery_class"]
rc = pc.cast(rc, pa.string()) if pa.types.is_dictionary(rc.type) else rc
HUNTC = pa.array(["LIKELY_REAL_ENTITY", "UNTYPED_CANDIDATE", "LIKELY_PLACE", "LIKELY_MEDIA_ENTITY",
                  "LIKELY_CREATIVE_WORK", "LIKELY_PERSON"])
H = np.sort(pc.filter(sk["node_uid"], pc.is_in(rc, value_set=HUNTC)).to_numpy())
del sk, rc
# what the 2010-07-16 dump already reached, so the MARGINAL gain is measured and not inferred
prev = np.sort(pa.concat_tables(
    [pq.read_table(fp, columns=["node_uid"]) for fp in
     sorted(glob.glob(f"{ACQ}/fb2010_names/*.parquet"))])["node_uid"].to_numpy()) \
    if glob.glob(f"{ACQ}/fb2010_names/*.parquet") else np.zeros(0, np.int64)
print(f"residue {len(U):,}  hunt {len(H):,}  already named by 2010-07-16 {len(prev):,} "
      f"({time.time()-t0:.0f}s)", flush=True)


def member(arr, sorted_ref):
    if len(sorted_ref) == 0:
        return np.zeros(len(arr), bool)
    p = np.clip(np.searchsorted(sorted_ref, arr), 0, len(sorted_ref) - 1)
    return sorted_ref[p] == arr


counts = {}
name_rows = {"uid": [], "mid": [], "val": [], "hunt": [], "new": []}
spart = len(glob.glob(f"{STR}/*.parquet"))
struct = {"uid": [], "mid": [], "kind": [], "val": [], "hunt": []}


def sflush():
    global spart, struct
    if not struct["uid"]:
        return
    pq.write_table(pa.table({
        "node_uid": pa.array(struct["uid"], pa.int64()), "node_id": pa.array(struct["mid"]),
        "fact_kind": pa.array(struct["kind"]), "value_2010_wex": pa.array(struct["val"]),
        "in_hunt": pa.array(struct["hunt"], pa.bool_())}),
        f"{STR}/part_{spart:05d}.parquet", compression="zstd")
    spart += 1
    struct = {"uid": [], "mid": [], "kind": [], "val": [], "hunt": []}


for kind, path in SRC.items():
    n_lines = n_bad = n_hit = 0
    bu, bm, bv = [], [], []

    def drain():
        global n_hit
        if not bu:
            return
        arr = np.array(bu, np.int64)
        inres = member(arr, U)
        inhunt = member(arr, H) & inres
        isnew = inres & ~member(arr, prev)
        for i in np.flatnonzero(inres).tolist():
            if kind == "name":
                name_rows["uid"].append(bu[i]); name_rows["mid"].append(bm[i])
                name_rows["val"].append(bv[i]); name_rows["hunt"].append(bool(inhunt[i]))
                name_rows["new"].append(bool(isnew[i]))
            else:
                struct["uid"].append(bu[i]); struct["mid"].append(bm[i])
                struct["kind"].append(kind); struct["val"].append(bv[i])
                struct["hunt"].append(bool(inhunt[i]))
        n_hit += int(inres.sum())
        bu.clear(); bm.clear(); bv.clear()
        if len(struct["uid"]) >= 500000:
            sflush()

    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n_lines += 1
            f = line.rstrip("\n").split("\t")
            if len(f) < 2 or not f[1]:
                continue
            h = f[0].strip()
            if len(h) != 32 or not h.startswith(PREFIX):
                n_bad += 1
                continue
            mid = "m.0" + enc(int(h[16:], 16) - OFF)
            bu.append(hash(mid)); bm.append(mid); bv.append(f[1])
            if len(bu) >= (1 << 18):
                drain()
    drain()
    counts[kind] = {"lines": n_lines, "guid_rejected": n_bad, "residue_hits": n_hit}
    print(f"  {kind:<18} lines {n_lines:>10,}  residue hits {n_hit:>9,}  ({time.time()-t0:.0f}s)",
          flush=True)
sflush()

# one row per node for the name tier (WEX has one name per guid, but be explicit about it)
seen = {}
for u, m, v, hu, nw in zip(name_rows["uid"], name_rows["mid"], name_rows["val"],
                           name_rows["hunt"], name_rows["new"]):
    seen.setdefault(u, (m, v, hu, nw))
rows = sorted((u,) + t for u, t in seen.items())
if rows:
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in rows], pa.int64()),
        "node_id": pa.array([r[1] for r in rows]),
        "display_name": pa.array([r[2] for r in rows]),
        "in_hunt": pa.array([r[3] for r in rows], pa.bool_()),
        "new_vs_2010_07_16": pa.array([r[4] for r in rows], pa.bool_()),
        "language": pa.array(["en"] * len(rows)),
        "source": pa.array([TIER] * len(rows)),
        "source_release": pa.array([RELEASE] * len(rows)),
        "source_rung": pa.array(["wex"] * len(rows)),
        "is_original_name": pa.array([True] * len(rows), pa.bool_()),
        "is_current_snapshot_name": pa.array([False] * len(rows), pa.bool_())}),
        f"{OUT}/part_00000.parquet", compression="zstd")

n_hunt = sum(1 for r in rows if r[3])
n_new = sum(1 for r in rows if r[4])
rec = {"schema": "WEX_JOIN/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "writes only under _acquisition/; frozen graph and frozen overlay untouched.",
       "source": "freebase-wex-data-dump-2010-07-05, files freebase_names / freebase_wpid / "
                 "freebase_types (checksum-verified, see V3_FB2010_ACQUISITION_WEX_*.json)",
       "tier": TIER, "source_release": RELEASE, "source_rung": "wex",
       "TIER_PLACEMENT_IS_PROVISIONAL": "same open question as the 2010-07-16 dump; the locked "
                                        "provenance order is NOT edited here.",
       "per_file": counts,
       "residue_nodes_named": len(rows), "HUNT_nodes_named": n_hunt,
       "MARGINAL_over_2010_07_16": n_new,
       "WHY_MARGINAL_IS_REPORTED": "per-source reaches are not additive and this campaign has "
                                   "already seen two label tables whose union added exactly zero. "
                                   "new_vs_2010_07_16 is computed per row, not inferred.",
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_WEX_JOIN.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps(rec, indent=1, ensure_ascii=False))
for r in [r for r in rows if r[4]][:20]:
    print(f"  NEW {r[1]:<15} {r[2][:56]!r}")
