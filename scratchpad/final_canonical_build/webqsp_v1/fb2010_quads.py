"""Names (all languages) and aliases from a Freebase QUADRUPLES dump.

    PYTHONHASHSEED=0 python .../fb2010_quads.py [2010|2008]

TWO RELEASES, ONE PARSER
  2010-07-16 (4.23 GB) and 2008-03-28 (0.56 GB) share the format exactly:
      subject <TAB> predicate <TAB> object <TAB> literal
  They differ only in what the subject can be.  The 2008 file mixes SCHEMA objects, whose subject is
  a path like /american_football/football_coach, with INSTANCES, whose subject is /guid/<32hex>;
  measured on 600,000 lines, name quads split 40,465 path to 61,457 guid.  Path subjects are Freebase
  schema, not graph nodes, and the guid-prefix test already drops them without coercion.

WHY THE QUADS AND NOT ONLY THE TOPIC DUMP
  freebase-simple-topic-dump lists TOPICS only -- 13,310,836 of them -- and gives exactly one name
  per topic, English-biased.  The quadruples dump is the whole 2010 database: every object, every
  property.  Two things follow:
    * objects that were never /common/topic (mediators, editions, tracks, documents) can carry
      /type/object/name here and are invisible to the topic dump.  Most of our residue is exactly
      that kind of object.
    * names come with their /lang/<code>, so the locked multi-label rule applies properly:
      keep every attested label, choose the display deterministically.
  Verified on the first 712,687 quads: 65,866 are /type/object/name, shaped
      /guid/<32hex> \\t /type/object/name \\t /lang/en \\t <the name>
  and /lang/de, /lang/fr ... appear alongside /lang/en.

THE JOIN
  mid = "m.0" + base32(int(guid[16:],16) - 0x8000000000000000), alphabet
  "0123456789bcdfghjklmnpqrstvwxyz_".  Same derivation as fb2010_join.py, verified there against
  the frozen graph (71.7% of 2010 head topics present; reproduces m.02mjmr).

TWO PASSES, BOUNDED MEMORY
  pass 1 streams the .bz2 (never expanded on disk) and writes every residue-matching name/alias
         quad straight out to parts -- no grouping, so memory is O(batch)
  pass 2 reads the parts back, groups by node, applies the display rule
         (English, else the lexicographically first language code) and writes the tier table

TYPES AND KEYS COME FREE IN THE SAME STREAM
  Measured on a 2,500,000-line sample of the real file, the predicate mix is
      /type/object/type 582,406   /type/object/key 381,436   /type/object/name 230,370
  so the quads dump also states, for every 2010 object, the types it carried and the keys it was
  minted under.  That is the source-attested answer to the question "what ARE the 5,832,113
  UNTYPED_CANDIDATE nodes", for the subset the 2010 dump covers -- the frozen 2015 graph states no
  type for them at all.  Capturing it costs nothing here and would cost a second 4.23 GB stream
  later, so it is captured.  It goes to its own evidence table and is NOT applied: the frozen
  node_kind is not touched and no band is recomputed by this script.

OUTPUT (append-only)
  _acquisition/fb2010_quad_labels/part_*.parquet   every attested label: node_uid, lang, label, kind
  _acquisition/fb2010_quad_names/part_00000.parquet the display name per node (the tier table)
  _acquisition/fb2010_quad_struct/part_*.parquet   2010 types and keys for residue nodes (evidence)
  V3_FB2010_QUADS.json
"""
import sys, io, os, bz2, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
WHICH = sys.argv[1] if len(sys.argv) > 1 else "2010"
if WHICH not in ("2010", "2008"):
    sys.exit("usage: fb2010_quads.py [2010|2008]")
SRCF = {"2010": "quads_freebase-datadump-quadruples.tsv.bz2",
        "2008": "quads2008_freebase-datadump-quadruples.tsv.bz2"}[WHICH]
SFX = "" if WHICH == "2010" else "_2008"
SRC = f"{ACQ}/raw/{SRCF}"
LBL = f"{ACQ}/fb2010_quad_labels{SFX}"
OUT = f"{ACQ}/fb2010_quad_names{SFX}"
STR = f"{ACQ}/fb2010_quad_struct{SFX}"
for _d in (LBL, OUT, STR):
    os.makedirs(_d, exist_ok=True)
TIER = "FREEBASE_HISTORICAL_DUMP_NAME"
RELEASE = {"2010": "2010-07-16", "2008": "2008-03-28"}[WHICH]
AB = "0123456789bcdfghjklmnpqrstvwxyz_"
OFF = 0x8000000000000000
PREFIX = "9202a8c04000641f"
NAME_P = "/type/object/name"
ALIAS_P = "/common/topic/alias"
TYPE_P = "/type/object/type"
KEY_P = "/type/object/key"
KIND = {NAME_P: "name", ALIAS_P: "alias", TYPE_P: "type", KEY_P: "key"}
BATCH = 1 << 18
t0 = time.time()

if not os.path.exists(SRC):
    sys.exit(f"missing {SRC} -- run fb2010_fetch.py quads first")


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
print(f"residue {len(U):,}  hunt {len(H):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- pass 1
part = len(glob.glob(f"{LBL}/*.parquet"))
spart = len(glob.glob(f"{STR}/*.parquet"))
b_uid, b_mid, b_lang, b_lab, b_kind = [], [], [], [], []
k_uid, k_mid, k_lang, k_lab, k_kind, k_hunt = [], [], [], [], [], []
s_uid, s_mid, s_kind, s_val, s_hunt = [], [], [], [], []
n_lines = n_kept = n_struct = 0
seen_p = collections.Counter()


def sflush():
    global spart, s_uid, s_mid, s_kind, s_val, s_hunt
    if not s_uid:
        return
    pq.write_table(pa.table({
        "node_uid": pa.array(s_uid, pa.int64()), "node_id": pa.array(s_mid),
        "fact_kind": pa.array(s_kind), "value_2010": pa.array(s_val),
        "in_hunt": pa.array(s_hunt, pa.bool_())}),
        f"{STR}/part_{spart:05d}.parquet", compression="zstd")
    spart += 1
    s_uid, s_mid, s_kind, s_val, s_hunt = [], [], [], [], []


def flush():
    global part, k_uid, k_mid, k_lang, k_lab, k_kind, k_hunt
    if not k_uid:
        return
    pq.write_table(pa.table({
        "node_uid": pa.array(k_uid, pa.int64()), "node_id": pa.array(k_mid),
        "language": pa.array(k_lang), "label": pa.array(k_lab),
        "label_kind": pa.array(k_kind), "in_hunt": pa.array(k_hunt, pa.bool_())}),
        f"{LBL}/part_{part:05d}.parquet", compression="zstd")
    part += 1
    k_uid, k_mid, k_lang, k_lab, k_kind, k_hunt = [], [], [], [], [], []


def drain():
    global n_kept, n_struct
    if not b_uid:
        return
    arr = np.array(b_uid, np.int64)
    p = np.clip(np.searchsorted(U, arr), 0, len(U) - 1)
    inres = U[p] == arr
    q = np.clip(np.searchsorted(H, arr), 0, len(H) - 1)
    inhunt = (H[q] == arr) & inres
    for i in np.flatnonzero(inres).tolist():
        kd = b_kind[i]
        if kd == "name" or kd == "alias":
            k_uid.append(b_uid[i]); k_mid.append(b_mid[i]); k_lang.append(b_lang[i])
            k_lab.append(b_lab[i]); k_kind.append(kd); k_hunt.append(bool(inhunt[i]))
            n_kept += 1
        else:
            s_uid.append(b_uid[i]); s_mid.append(b_mid[i]); s_kind.append(kd)
            s_val.append(b_lab[i]); s_hunt.append(bool(inhunt[i]))
            n_struct += 1
    b_uid.clear(); b_mid.clear(); b_lang.clear(); b_lab.clear(); b_kind.clear()
    if len(k_uid) >= 500000:
        flush()
    if len(s_uid) >= 500000:
        sflush()


with bz2.open(SRC, "rt", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        n_lines += 1
        f = line.rstrip("\n").split("\t")
        if len(f) < 4:
            continue
        if n_lines % 20000000 == 0:
            # NOTE: this check must sit BEFORE the predicate filter. Placed after it, the `continue`
            # for non-matching predicates skips it and line 20,000,000 is almost never one of the
            # four predicates, so the job runs silently for hours.
            print(f"  {n_lines:,} quads  name {seen_p['name']:,} alias {seen_p['alias']:,} "
                  f"type {seen_p['type']:,} key {seen_p['key']:,}  residue labels {n_kept:,} "
                  f"struct {n_struct:,}  ({time.time()-t0:.0f}s)", flush=True)
        kind = KIND.get(f[1])
        if kind is None:
            continue
        seen_p[kind] += 1
        # name/alias carry the string in col 4; type/key name their object in col 3
        lab = f[3] if (kind == "name" or kind == "alias") else (f[2] or f[3])
        if not lab:
            continue
        h = f[0].rsplit("/", 1)[-1]
        if len(h) != 32 or not h.startswith(PREFIX):
            continue
        mid = "m.0" + enc(int(h[16:], 16) - OFF)
        lang = f[2].rsplit("/", 1)[-1] if f[2].startswith("/lang/") else ""
        b_uid.append(hash(mid)); b_mid.append(mid); b_lang.append(lang)
        b_lab.append(lab); b_kind.append(kind)
        if len(b_uid) >= BATCH:
            drain()
drain()
flush()
sflush()
print(f"pass 1 done: {n_lines:,} quads, {n_kept:,} residue labels, {n_struct:,} residue "
      f"type/key facts ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- pass 2: display rule
by = collections.defaultdict(list)
mid_of, hunt_of = {}, {}
for fp in sorted(glob.glob(f"{LBL}/*.parquet")):
    t = pq.read_table(fp)
    for u, m, lg, lb, kd, hu in zip(t["node_uid"].to_pylist(), t["node_id"].to_pylist(),
                                    t["language"].to_pylist(), t["label"].to_pylist(),
                                    t["label_kind"].to_pylist(), t["in_hunt"].to_pylist()):
        if kd != "name":
            continue
        by[u].append((lg, lb))
        mid_of[u] = m
        hunt_of[u] = hu
rows = []
for u, labs in by.items():
    d = {}
    for lg, lb in labs:
        d.setdefault(lg or "und", lb)
    if "en" in d:
        lang, rule = "en", "english_present"
    else:
        lang, rule = min(d), "lexicographically_first_language"
    rows.append((u, mid_of[u], d[lang], lang, rule, len(d), hunt_of[u]))
rows.sort()
if rows:
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in rows], pa.int64()),
        "node_id": pa.array([r[1] for r in rows]),
        "display_name": pa.array([r[2] for r in rows]),
        "language": pa.array([r[3] for r in rows]),
        "display_rule": pa.array([r[4] for r in rows]),
        "n_attested_labels": pa.array([r[5] for r in rows], pa.int32()),
        "in_hunt": pa.array([r[6] for r in rows], pa.bool_()),
        "source": pa.array([TIER] * len(rows)),
        "source_release": pa.array([RELEASE] * len(rows)),
        "source_rung": pa.array(["quadruples"] * len(rows)),
        "is_original_name": pa.array([True] * len(rows), pa.bool_()),
        "is_current_snapshot_name": pa.array([False] * len(rows), pa.bool_())}),
        f"{OUT}/part_00000.parquet", compression="zstd")

n_hunt = sum(1 for r in rows if r[6])
multi = sum(1 for r in rows if r[5] > 1)
rec = {"schema": "FB2010_QUADS/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": "writes only under _acquisition/; frozen artifacts untouched.",
       "source": f"{SRCF} at release {RELEASE} "
                 "(checksum-verified, see V3_FB2010_ACQUISITION_QUADS.json)",
       "tier": TIER, "source_release": RELEASE, "source_rung": "quadruples",
       "TIER_PLACEMENT_IS_PROVISIONAL": "same as fb2010_join: proposed above "
                                        "FREEBASE_HISTORICAL_PAGE_NAME, not applied to the locked order.",
       "quads_read": n_lines,
       "quads_by_predicate": {k: int(v) for k, v in seen_p.items()},
       "residue_label_rows": n_kept,
       "residue_type_and_key_facts": n_struct,
       "TYPES_AND_KEYS_ARE_EVIDENCE_ONLY": "fb2010_quad_struct/ records the types and keys the 2010 "
                                           "dump states for residue nodes. The frozen node_kind is "
                                           "NOT touched and no band is recomputed here; applying it "
                                           "would be a new overlay record with its own hash.",
       "residue_nodes_named": len(rows), "HUNT_nodes_named": n_hunt,
       "nodes_with_more_than_one_language": multi,
       "MULTI_LABEL_RULE": "every attested label kept in fb2010_quad_labels/; display = English if "
                           "present, else the lexicographically first language code.",
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_FB2010_QUADS{SFX.upper()}.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(json.dumps(rec, indent=1, ensure_ascii=False))
for r in rows[:20]:
    print(f"  {r[1]:<15} {r[2][:50]!r:<52} [{r[3]}] labels={r[5]} hunt={r[6]}")
print(f"{time.time()-t0:.0f}s")
