"""Join the 2010-07-16 Freebase simple topic dump onto the frozen graph's unresolved residue.

    PYTHONHASHSEED=0 python .../fb2010_join.py

WHAT THIS RECOVERS
  A name that Freebase published in its own July 2010 dump for a topic that carries no
  /type/object/name in the freebase-rdf-latest snapshot.  The name existed; it was lost between the
  two releases (deletion, merge, or export loss).  Recovering it is an exact-source recovery of an
  original Freebase name -- Freebase's own published data, not a page render and not an external
  reconciliation.

TIER  (PROVISIONAL - needs a ruling, see V3_FB2010_TIER_QUESTION in the output record)
  FREEBASE_HISTORICAL_DUMP_NAME, placed provisionally immediately ABOVE
  FREEBASE_HISTORICAL_PAGE_NAME and below FREEBASE_DELETED_NAME, because it is the same kind of
  evidence as the archived page (original, historical, not current-snapshot) but obtained from
  Freebase's own published dump rather than from parsing an HTML capture.
      is_original_name         = True     (Freebase asserted it)
      is_current_snapshot_name = False    (it is not in our frozen dump)
      source_release           = 2010-07-16
  Nothing in the locked order is edited by this script; the placement is recorded, not applied.

THE JOIN
  mid = "m.0" + base32(int(guid[16:], 16) - 0x8000000000000000)
  alphabet "0123456789bcdfghjklmnpqrstvwxyz_".  Verified before this script was written on 67,543
  head topics (71.7% present in the frozen 302M-node graph) and independently on m.02mjmr.
  Any guid that does not have the 9202a8c04000641f prefix, or whose derived MID is not a node of
  the frozen graph, is counted and dropped -- never coerced.

MEMORY / DISK
  The .bz2 is decompressed as a stream and never written out (the standing constraint: the source
  is never expanded on disk).  Rows are flushed to parquet parts every FLUSH_ROWS.

THE DUMP HAS SIX COLUMNS, NOT TWO
  guid | name | /en/ key | /wikipedia/en_id/<pageid> | types | description
  All of them are captured, because three of them are independent identity evidence:
    col2  the human-readable /en/ key Freebase itself minted for the topic
    col3  the Wikipedia NUMERIC PAGE ID -- this is the "Wikipedia historical identifiers" rung of
          the recovery chain, delivered by the same file; it resolves to an article without any
          name matching (https://en.wikipedia.org/?curid=<id>)
    col5  Freebase's own description blurb (capped at DESC_CAP chars on disk)

OUTPUT (append-only)
  _acquisition/fb2010_names/part_*.parquet      residue nodes that had a 2010 name
  V3_FB2010_JOIN.json
"""
import sys, io, os, bz2, json, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
SRC = f"{ACQ}/raw/simple_freebase-simple-topic-dump.tsv.bz2"
OUT = f"{ACQ}/fb2010_names"
os.makedirs(OUT, exist_ok=True)
TIER = "FREEBASE_HISTORICAL_DUMP_NAME"
RELEASE = "2010-07-16"
FLUSH_ROWS = 400000
DESC_CAP = 4000
NULL = chr(92) + "N"
AB = "0123456789bcdfghjklmnpqrstvwxyz_"
OFF = 0x8000000000000000
PREFIX = "9202a8c04000641f"
t0 = time.time()

if not os.path.exists(SRC):
    sys.exit(f"missing {SRC} -- run fb2010_fetch.py simple first")


def enc(n):
    if n == 0:
        return "0"
    s = ""
    while n:
        s = AB[n & 31] + s
        n >>= 5
    return s


# ---------------------------------------------------------------- targets
U = np.load(f"{ACQ}/_unresolved_population.npz")["U"]
print(f"residue population: {len(U):,} ({time.time()-t0:.0f}s)", flush=True)
sk = pq.read_table(f"{ACQ}/semantic_kind_v2_1.parquet", columns=["node_uid", "recovery_class"])
rc = sk["recovery_class"]
rc = pc.cast(rc, pa.string()) if pa.types.is_dictionary(rc.type) else rc
HUNT = pa.array(["LIKELY_REAL_ENTITY", "UNTYPED_CANDIDATE", "LIKELY_PLACE", "LIKELY_MEDIA_ENTITY",
                 "LIKELY_CREATIVE_WORK", "LIKELY_PERSON"])
H = np.sort(pc.filter(sk["node_uid"], pc.is_in(rc, value_set=HUNT)).to_numpy())
del sk, rc
print(f"hunt population: {len(H):,} ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- stream
n_lines = n_guid_ok = n_named = 0
bad_prefix = 0
part = len([p for p in os.listdir(OUT) if p.endswith(".parquet")])
kept_res = kept_hunt = 0
samples = []
kept = {"uid": [], "mid": [], "name": [], "enkey": [], "wpid": [], "types": [],
        "desc": [], "hunt": []}


def flush():
    global part
    if not kept["uid"]:
        return
    pq.write_table(pa.table({
        "node_uid": pa.array(kept["uid"], pa.int64()),
        "node_id": pa.array(kept["mid"]),
        "display_name": pa.array(kept["name"]),
        "en_key_2010": pa.array(kept["enkey"]),
        "wikipedia_en_page_id": pa.array(kept["wpid"]),
        "types_2010": pa.array(kept["types"]),
        "description_2010": pa.array(kept["desc"]),
        "in_hunt": pa.array(kept["hunt"], pa.bool_()),
        "source": pa.array([TIER] * len(kept["uid"])),
        "source_release": pa.array([RELEASE] * len(kept["uid"])),
        "is_original_name": pa.array([True] * len(kept["uid"]), pa.bool_()),
        "is_current_snapshot_name": pa.array([False] * len(kept["uid"]), pa.bool_())}),
        f"{OUT}/part_{part:05d}.parquet", compression="zstd")
    part += 1
    for k in kept:
        kept[k] = []


BATCH = 1 << 18
b_uid, b_mid, b_name, b_types = [], [], [], []
b_enkey, b_wpid, b_desc = [], [], []


def drain():
    global kept_res, kept_hunt, n_named
    if not b_uid:
        return
    arr = np.array(b_uid, np.int64)
    p = np.clip(np.searchsorted(U, arr), 0, len(U) - 1)
    inres = U[p] == arr
    q = np.clip(np.searchsorted(H, arr), 0, len(H) - 1)
    inhunt = (H[q] == arr) & inres
    idx = np.flatnonzero(inres)
    kept_res += len(idx)
    kept_hunt += int(inhunt.sum())
    for i in idx.tolist():
        kept["uid"].append(b_uid[i]); kept["mid"].append(b_mid[i])
        kept["name"].append(b_name[i]); kept["enkey"].append(b_enkey[i])
        kept["wpid"].append(b_wpid[i]); kept["types"].append(b_types[i])
        kept["desc"].append(b_desc[i]); kept["hunt"].append(bool(inhunt[i]))
        if len(samples) < 40 and inhunt[i]:
            samples.append((b_mid[i], b_name[i], b_enkey[i], b_wpid[i], b_types[i][:50]))
    b_uid.clear(); b_mid.clear(); b_name.clear(); b_types.clear()
    b_enkey.clear(); b_wpid.clear(); b_desc.clear()
    if len(kept["uid"]) >= FLUSH_ROWS:
        flush()


with bz2.open(SRC, "rt", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        n_lines += 1
        f = line.rstrip("\n").split("\t")
        if len(f) < 2:
            continue
        h = f[0].rsplit("/", 1)[-1]
        if len(h) != 32 or not h.startswith(PREFIX):
            bad_prefix += 1
            continue
        v = int(h[16:], 16)
        mid = "m.0" + enc(v - OFF if v >= OFF else v)
        n_guid_ok += 1
        name = f[1]
        if not name or name == NULL:
            continue
        n_named += 1
        b_uid.append(hash(mid)); b_mid.append(mid); b_name.append(name)
        col = lambda i: (f[i] if len(f) > i and f[i] != NULL else "")
        b_enkey.append(col(2))
        w = col(3)
        b_wpid.append(w.rsplit("/", 1)[-1] if w.startswith("/wikipedia/en_id/") else "")
        b_types.append(col(4))
        b_desc.append(col(5)[:DESC_CAP])
        if len(b_uid) >= BATCH:
            drain()
        if n_lines % 2000000 == 0:
            print(f"  {n_lines:,} lines  named {n_named:,}  residue-hits {kept_res:,}  "
                  f"hunt-hits {kept_hunt:,}  ({time.time()-t0:.0f}s)", flush=True)
drain()
flush()

rec = {"schema": "FB2010_JOIN/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "writes only under _acquisition/; frozen graph and frozen overlay untouched.",
       "source": "freebase-data-dump-2010-07-16 / freebase-simple-topic-dump.tsv.bz2 "
                 "(checksum-verified, see V3_FB2010_ACQUISITION_SIMPLE.json)",
       "tier": TIER, "source_release": RELEASE,
       "TIER_PLACEMENT_IS_PROVISIONAL": "recorded, not applied: proposed immediately above "
                                        "FREEBASE_HISTORICAL_PAGE_NAME and below FREEBASE_DELETED_NAME. "
                                        "V3_NAME_PROVENANCE_ORDER_V1.json is NOT edited; a ruling "
                                        "would be a new record with its own hash.",
       "is_original_name": True, "is_current_snapshot_name": False,
       "lines_read": n_lines, "guid_derivable": n_guid_ok, "guid_prefix_rejected": bad_prefix,
       "topics_with_a_2010_name": n_named,
       "names_for_residue_nodes": kept_res,
       "names_for_HUNT_nodes": kept_hunt,
       "residue_population": int(len(U)), "hunt_population": int(len(H)),
       "GUID_TO_MID": 'mid = "m.0" + base32(int(guid[16:],16) - 0x8000000000000000)',
       "samples_hunt": [{"mid": m, "name_2010": n, "en_key_2010": k,
                         "wikipedia_en_page_id": w, "types_2010": t}
                        for m, n, k, w, t in samples[:25]],
       "WIKIPEDIA_RUNG": "wikipedia_en_page_id is Freebase's stored /wikipedia/en_id key: a NUMERIC "
                         "page id, so the article is reached by identifier, never by name matching. "
                         "It is captured here as evidence, not resolved by this script.",
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FB2010_JOIN.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in rec.items() if k != "samples_hunt"}, indent=1))
print("\nsample recovered HUNT names:")
for m, n, k, w, t in samples[:25]:
    print(f"  {m:<15} {n[:40]!r:<42} key={k[:26]:<26} wp={w:<9} {t[:40]}")
print(f"{time.time()-t0:.0f}s")
