"""Names from the 2010-07-16 per-type TSV export -- the one 2010 file that covers NON-topics.

    PYTHONHASHSEED=0 python .../fb2010_tsv_join.py

WHY THIS FILE IS DIFFERENT FROM THE OTHER TWO 2010 SOURCES
  freebase-simple-topic-dump lists /common/topic objects only.  WEX lists Wikipedia-linked objects
  only.  Both therefore miss the mediators, editions, tracks, performances and versions that make up
  most of the residue.  freebase-datadump-tsv.tar.bz2 is one TSV PER FREEBASE TYPE, and the type
  list includes exactly those non-topic types:
      data/cvg/game_performance.tsv, data/cvg/musical_game_song_relationship.tsv,
      data/cvg/game_version.tsv, ...
  Every member has a header row, and the first two columns are `name` and `id`, with id shaped
  /guid/<32hex>.  So this file states a name for typed objects that were never topics.

  The member path is itself the type: data/cvg/computer_videogame.tsv -> /cvg/computer_videogame.
  That is recorded alongside the name, giving a second independent 2010 type source (the quadruples
  give the first) for nodes the frozen 2015 graph types not at all.

STREAMED, NEVER EXPANDED
  1.35 GB compressed and about 8 GB expanded, against 21 GB free.  tarfile in stream mode ("r|")
  reads members sequentially straight out of the bz2 stream; nothing is written to disk and no
  member is held whole in memory.

THE JOIN
  mid = "m.0" + base32(int(guid[16:],16) - 0x8000000000000000), the same derivation used for the
  other 2010 sources, shown injective over 3,038,202 real guids with zero collisions
  (V3_WEX_JOIN.json / GUID_TO_MID_INJECTIVITY).

TIER
  FREEBASE_HISTORICAL_DUMP_NAME, source_release 2010-07-16, source_rung "per_type_tsv".  Same
  provisional placement question as the other 2010 rungs; the locked order is not edited here.

OUTPUT (append-only)
  _acquisition/fb2010_tsv_names/part_00000.parquet   one row per residue node
  _acquisition/fb2010_tsv_types/part_*.parquet       (node, type) pairs read off the member paths
  V3_FB2010_TSV_JOIN.json
"""
import sys, io, os, bz2, json, glob, time, tarfile, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
SRC = f"{ACQ}/raw/tsv_freebase-datadump-tsv.tar.bz2"
OUT = f"{ACQ}/fb2010_tsv_names"
TYP = f"{ACQ}/fb2010_tsv_types"
for d in (OUT, TYP):
    os.makedirs(d, exist_ok=True)
TIER = "FREEBASE_HISTORICAL_DUMP_NAME"
RELEASE = "2010-07-16"
RUNG = "per_type_tsv"
AB = "0123456789bcdfghjklmnpqrstvwxyz_"
OFF = 0x8000000000000000
PREFIX = "9202a8c04000641f"
NAME_CAP = 400
t0 = time.time()
if not os.path.exists(SRC):
    sys.exit(f"missing {SRC} -- run fb2010_fetch.py tsv first")


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
prevpaths = (sorted(glob.glob(f"{ACQ}/fb2010_names/*.parquet")) +
             sorted(glob.glob(f"{ACQ}/wex_names/*.parquet")) +
             sorted(glob.glob(f"{ACQ}/fb2010_quad_names/*.parquet")))
prev = (np.sort(pa.concat_tables([pq.read_table(p, columns=["node_uid"]) for p in prevpaths])
                ["node_uid"].to_numpy()) if prevpaths else np.zeros(0, np.int64))
print(f"residue {len(U):,}  hunt {len(H):,}  already named by other 2010 rungs {len(prev):,} "
      f"({time.time()-t0:.0f}s)", flush=True)


def member(arr, ref):
    if len(ref) == 0:
        return np.zeros(len(arr), bool)
    p = np.clip(np.searchsorted(ref, arr), 0, len(ref) - 1)
    return ref[p] == arr


best = {}                       # node_uid -> (mid, name, first type seen, in_hunt, is_new)
tpart = len(glob.glob(f"{TYP}/*.parquet"))
trow = {"uid": [], "mid": [], "type": []}
n_members = n_rows = n_hits = n_notype = 0
per_type = collections.Counter()
bu, bm, bn, bt = [], [], [], []


def tflush():
    global tpart, trow
    if not trow["uid"]:
        return
    pq.write_table(pa.table({"node_uid": pa.array(trow["uid"], pa.int64()),
                             "node_id": pa.array(trow["mid"]),
                             "type_2010": pa.array(trow["type"])}),
                   f"{TYP}/part_{tpart:05d}.parquet", compression="zstd")
    tpart += 1
    trow = {"uid": [], "mid": [], "type": []}


def drain():
    global n_hits
    if not bu:
        return
    arr = np.array(bu, np.int64)
    inres = member(arr, U)
    inhunt = member(arr, H) & inres
    isnew = inres & ~member(arr, prev)
    for i in np.flatnonzero(inres).tolist():
        u = bu[i]
        if u not in best:
            best[u] = (bm[i], bn[i], bt[i], bool(inhunt[i]), bool(isnew[i]))
            per_type[bt[i]] += 1
        trow["uid"].append(u); trow["mid"].append(bm[i]); trow["type"].append(bt[i])
    n_hits += int(inres.sum())
    bu.clear(); bm.clear(); bn.clear(); bt.clear()
    if len(trow["uid"]) >= 500000:
        tflush()


fh = bz2.open(SRC, "rb")
tf = tarfile.open(fileobj=fh, mode="r|")
for m in tf:
    if not m.isfile() or not m.name.endswith(".tsv"):
        continue
    ftype = "/" + m.name[len("data/"):-len(".tsv")] if m.name.startswith("data/") else m.name
    ex = tf.extractfile(m)
    if ex is None:
        continue
    n_members += 1
    head = ex.readline().decode("utf-8", "replace").rstrip("\n").split("\t")
    try:
        i_name, i_id = head.index("name"), head.index("id")
    except ValueError:
        n_notype += 1
        continue
    for raw in ex:
        n_rows += 1
        f = raw.decode("utf-8", "replace").rstrip("\n").split("\t")
        if len(f) <= max(i_name, i_id):
            continue
        nm, gid = f[i_name], f[i_id]
        if not nm or not gid:
            continue
        h = gid.rsplit("/", 1)[-1]
        if len(h) != 32 or not h.startswith(PREFIX):
            continue
        mid = "m.0" + enc(int(h[16:], 16) - OFF)
        bu.append(hash(mid)); bm.append(mid); bn.append(nm[:NAME_CAP]); bt.append(ftype)
        if len(bu) >= (1 << 18):
            drain()
    if n_members % 200 == 0:
        drain()
        print(f"  {n_members} members  {n_rows:,} rows  residue nodes {len(best):,}  "
              f"({time.time()-t0:.0f}s)", flush=True)
drain()
tflush()
tf.close()
fh.close()
print(f"members {n_members}  rows {n_rows:,}  residue hits {n_hits:,}  "
      f"distinct residue nodes {len(best):,}  ({time.time()-t0:.0f}s)", flush=True)

rows = sorted((u,) + v for u, v in best.items())
if rows:
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in rows], pa.int64()),
        "node_id": pa.array([r[1] for r in rows]),
        "display_name": pa.array([r[2] for r in rows]),
        "type_2010": pa.array([r[3] for r in rows]),
        "in_hunt": pa.array([r[4] for r in rows], pa.bool_()),
        "new_vs_other_2010_rungs": pa.array([r[5] for r in rows], pa.bool_()),
        "language": pa.array(["en"] * len(rows)),
        "source": pa.array([TIER] * len(rows)),
        "source_release": pa.array([RELEASE] * len(rows)),
        "source_rung": pa.array([RUNG] * len(rows)),
        "is_original_name": pa.array([True] * len(rows), pa.bool_()),
        "is_current_snapshot_name": pa.array([False] * len(rows), pa.bool_())}),
        f"{OUT}/part_00000.parquet", compression="zstd")

n_hunt = sum(1 for r in rows if r[4])
n_new = sum(1 for r in rows if r[5])
rec = {"schema": "FB2010_TSV_JOIN/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "writes only under _acquisition/; frozen graph and frozen overlay untouched.",
       "source": "freebase-data-dump-2010-07-16 / freebase-datadump-tsv.tar.bz2 "
                 "(checksum-verified, see V3_FB2010_ACQUISITION_TSV.json)",
       "tier": TIER, "source_release": RELEASE, "source_rung": RUNG,
       "TIER_PLACEMENT_IS_PROVISIONAL": "same open question as the other 2010 rungs; the locked "
                                        "provenance order is NOT edited here.",
       "WHY_THIS_FILE": "one TSV per Freebase type, so it names NON-topic objects (mediators, "
                        "versions, performances) that the topic dump and WEX cannot reach.",
       "members_read": n_members, "members_without_name_and_id_columns": n_notype,
       "data_rows_read": n_rows, "residue_row_hits": n_hits,
       "residue_nodes_named": len(rows), "HUNT_nodes_named": n_hunt,
       "MARGINAL_over_other_2010_rungs": n_new,
       "WHY_MARGINAL_IS_REPORTED": "per-source reaches are not additive; WEX looked like 40,305 new "
                                   "names and was worth 602 once intersected.",
       "top_40_types_by_residue_nodes_named": [{"type_2010": k, "n": v}
                                               for k, v in per_type.most_common(40)],
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FB2010_TSV_JOIN.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in rec.items() if k != "top_40_types_by_residue_nodes_named"},
                 indent=1, ensure_ascii=False))
print("\ntop types by residue nodes named:")
for d in rec["top_40_types_by_residue_nodes_named"][:25]:
    print(f"  {d['type_2010'][:60]:<60} {d['n']:>8,}")
print("\nsample NEW names:")
for r in [r for r in rows if r[5]][:20]:
    print(f"  {r[1]:<15} {r[2][:52]!r:<54} {r[3][:34]}")
