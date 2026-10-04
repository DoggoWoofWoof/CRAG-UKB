"""The 3,186 MusicBrainz MBIDs, resolved by exact identifier lookup at MusicBrainz.

    PYTHONHASHSEED=0 python .../authority_musicbrainz.py

WHY THIS IS A SEPARATE SCRIPT FROM authority_native.py
  The stage inside that file spent an hour on under a hundred ids and the cause was measured, not
  guessed: MusicBrainz answers 503 to a request that arrives less than a second after the previous
  one, and the generic retry helper retried after 1s, then 2s, then 4s. Every retry arrived inside
  the window that caused the 503, so a single rate-limit turned into six endpoint attempts x four
  retries, and the job paced itself into the floor. Three things fix it and all three are structural
  rather than parameter tweaks, so they get their own file:

    1. ONE PACER FOR EVERY REQUEST. Nothing in this file may talk to musicbrainz.org except through
       pace(); it blocks until 1.15 s have passed since the previous request, whatever the caller.
       A 503 is then a real error rather than a self-inflicted one, and is given 5 s.
    2. THE ENTITY TYPE COSTS ONE REQUEST, NOT SIX. /ws/2 needs the entity type in the path and 1,360
       of these MBIDs are stored bare. https://musicbrainz.org/mbid/<uuid> 302-redirects to
       /<entity>/<uuid>, so the server is asked once where the id lives instead of being asked six
       times whether it lives here. Still an exact id lookup; nothing is searched for by name.
    3. IT CHECKPOINTS. Types and resolved rows are written every 100 ids, so an interruption costs
       a hundred lookups rather than the whole run.

WHAT COUNTS AS A MISS
  A 404 from /ws/2/<type>/<mbid> is recorded as a 404. It is never retried as a search, and no name
  is taken from anywhere but the record that id addresses.

Writes only under _acquisition/. Touches neither the frozen graph nor the frozen overlay.
"""
import sys, io, os, json, glob, time, collections, urllib.request, urllib.error
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
OUT = f"{ACQ}/mb_hits"
TYPEMAP = f"{ACQ}/mb_entity_types.json"
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
MIN_GAP = 1.15                      # MusicBrainz asks for <= 1 request/second, with headroom
t0 = time.time()
os.makedirs(OUT, exist_ok=True)

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

_last = [0.0]


def pace():
    d = MIN_GAP - (time.time() - _last[0])
    if d > 0:
        time.sleep(d)
    _last[0] = time.time()


class _NoRedir(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, hdrs, newurl):
        raise urllib.error.HTTPError(req.full_url, code, newurl, hdrs, fp)


_op = urllib.request.build_opener(_NoRedir)


def mb_json(url, attempts=4):
    """-> parsed json, or None. 404 is a miss and returns immediately; 503 means we were too fast
    and is given five seconds, never the sub-second backoff that caused it."""
    for a in range(attempts):
        pace()
        try:
            r = urllib.request.Request(url, headers={"User-Agent": UA,
                                                     "Accept": "application/json"})
            with urllib.request.urlopen(r, timeout=45) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            time.sleep(5 if e.code == 503 else 2)
        except Exception:
            time.sleep(2)
    return None


def mb_type(mbid):
    """The entity type, from MusicBrainz's own redirect for that exact id."""
    pace()
    try:
        _op.open(urllib.request.Request(f"https://musicbrainz.org/mbid/{mbid}",
                                        headers={"User-Agent": UA}), timeout=30)
        return None                                  # 200 means it did not redirect
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            seg = str(e.reason).rstrip("/").split("/")
            return seg[-2] if len(seg) >= 2 and seg[-1] == mbid else None
        return None
    except Exception:
        return None


# ---------------------------------------------------------------- the ids and what we know of them
t = pq.read_table(f"{ACQ}/authority_pending_native.parquet")
items = [(u, m, a, k) for u, m, ns, a, k in
         zip(t["node_uid"].to_pylist(), t["freebase_mid"].to_pylist(),
             t["authority_namespace"].combine_chunks().cast(pa.string()).to_pylist(),
             t["authority_id"].to_pylist(), t["key"].to_pylist()) if ns == "musicbrainz"]
print(f"musicbrainz rows: {len(items):,}  distinct mbids: {len({i[2] for i in items}):,}",
      flush=True)

MB2FB = [("/music/artist", "artist"), ("/music/musical_group", "artist"),
         ("/music/recording", "recording"), ("/music/track", "recording"),
         ("/music/release", "release"), ("/music/album", "release-group"),
         ("/music/composition", "work"), ("/music/record_label", "label"),
         ("/music/label", "label")]
KEYTYPE = {"artist", "recording", "work", "release", "release_group", "release-group", "label"}

etype = json.load(io.open(TYPEMAP, encoding="utf-8")) if os.path.exists(TYPEMAP) else {}
src = collections.Counter()
for u, m, aid, key in items:
    if aid in etype:
        continue
    p = key.split("/")
    if len(p) >= 5 and p[3] in KEYTYPE:
        etype[aid] = p[3].replace("_", "-")
        src["from_key"] += 1
print(f"  entity type from the key: {src['from_key']:,}", flush=True)

need = sorted({aid for _, _, aid, _ in items if aid not in etype})
if need:
    byuid = {}
    for u, _, aid, _ in items:
        byuid.setdefault(aid, u)
    want = {byuid[a] for a in need}
    fbt = collections.defaultdict(set)
    for fp in sorted(glob.glob(f"{ACQ}/residue_types/*.parquet")):
        tt = pq.read_table(fp, columns=["node_uid", "type"])
        for a, b in zip(tt["node_uid"].to_pylist(), tt["type"].to_pylist()):
            if a in want:
                fbt[a].add(b)
    for aid in need:
        ts = fbt.get(byuid[aid], set())
        for fb, mb in MB2FB:
            k = fb[1:].replace("/", ".")
            if any(x == k or x.startswith(k + ".") for x in ts):
                etype[aid] = mb
                src["from_graph_types"] += 1
                break
    json.dump(etype, io.open(TYPEMAP, "w", encoding="utf-8"))
    print(f"  entity type from our own graph's types: {src['from_graph_types']:,}", flush=True)

need = sorted({aid for _, _, aid, _ in items if aid not in etype})
print(f"  still untyped, resolving by redirect: {len(need):,}  "
      f"(~{len(need)*MIN_GAP/60:.0f} min)", flush=True)
for n, aid in enumerate(need):
    et = mb_type(aid)
    if et:
        etype[aid] = et
        src["from_mbid_redirect"] += 1
    else:
        etype[aid] = ""                              # recorded, so a restart does not re-ask
        src["redirect_gave_nothing"] += 1
    if n % 100 == 0:
        json.dump(etype, io.open(TYPEMAP, "w", encoding="utf-8"))
        print(f"    type {n:,}/{len(need):,} ({time.time()-t0:.0f}s)", flush=True)
json.dump(etype, io.open(TYPEMAP, "w", encoding="utf-8"))
print(f"  entity type source: {dict(src)}", flush=True)

# ---------------------------------------------------------------- one /ws/2 lookup per distinct id
label = {}
for fp in sorted(glob.glob(f"{OUT}/*.parquet")):
    d = pq.read_table(fp)
    for a, l, e in zip(d["authority_id"].to_pylist(), d["resolved_label"].to_pylist(),
                       d["resolved_entity_type"].to_pylist()):
        label[a] = (l, e)
done = set(json.load(io.open(f"{ACQ}/mb_missed.json", encoding="utf-8"))) \
    if os.path.exists(f"{ACQ}/mb_missed.json") else set()
print(f"  already resolved: {len(label):,}   already missed: {len(done):,}", flush=True)

todo = [a for a in sorted({i[2] for i in items}) if a not in label and a not in done]
print(f"  to look up: {len(todo):,}  (~{len(todo)*MIN_GAP/60:.0f} min)", flush=True)
rows, part = [], len(glob.glob(f"{OUT}/*.parquet"))
now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
bymid = collections.defaultdict(list)
for u, m, aid, _ in items:
    bymid[aid].append((u, m))


def flush():
    global part, rows
    if not rows:
        return
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in rows], pa.int64()),
        "freebase_mid": pa.array([r[1] for r in rows]),
        "authority_namespace": pa.array(["musicbrainz"] * len(rows)).dictionary_encode(),
        "authority_id": pa.array([r[2] for r in rows]),
        "resolved_label": pa.array([r[3] for r in rows]),
        "resolved_entity_type": pa.array([r[4] for r in rows]),
        "source_url": pa.array([r[5] for r in rows]),
        "retrieved_at": pa.array([r[6] for r in rows])}),
        f"{OUT}/part_{part:05d}.parquet", compression="zstd")
    part += 1
    rows = []


miss = collections.Counter()
for n, aid in enumerate(todo):
    et = etype.get(aid) or ""
    if not et:
        miss["no_entity_type"] += 1
        done.add(aid)
        continue
    d = mb_json(f"https://musicbrainz.org/ws/2/{et}/{aid}?fmt=json")
    lab = (d or {}).get("name") or (d or {}).get("title")
    if lab:
        label[aid] = (lab, et)
        for u, m in bymid[aid]:
            rows.append((u, m, aid, lab, et, f"https://musicbrainz.org/{et}/{aid}", now()))
    else:
        miss["ws2_404"] += 1
        done.add(aid)
    if len(rows) >= 200 or (n % 100 == 0 and n):
        flush()
        json.dump(sorted(done), io.open(f"{ACQ}/mb_missed.json", "w", encoding="utf-8"))
        print(f"    {n:,}/{len(todo):,} resolved={len(label):,} miss={dict(miss)} "
              f"({time.time()-t0:.0f}s)", flush=True)
flush()
json.dump(sorted(done), io.open(f"{ACQ}/mb_missed.json", "w", encoding="utf-8"))

# ---------------------------------------------------------------- consolidate
allr = []
for fp in sorted(glob.glob(f"{OUT}/*.parquet")):
    allr.append(pq.read_table(fp))
if allr:
    tb = pa.concat_tables(allr, promote_options="default")
    pq.write_table(tb, f"{ACQ}/external_authority_musicbrainz.parquet", compression="zstd")
    nodes = len(set(tb["node_uid"].to_pylist()))
else:
    nodes = 0
rec = {"schema": "EXTERNAL_AUTHORITY_MUSICBRAINZ/v1",
       "generated_utc": now(),
       "APPEND_ONLY": "writes only under _acquisition/; frozen artifacts untouched.",
       "RULE": "exact identifier lookup only; a 404 is recorded, never retried as a name search",
       "PACING": f"one request per {MIN_GAP}s, globally, with 5s after a 503",
       "input_rows": len(items),
       "distinct_mbids": len({i[2] for i in items}),
       "entity_type_source": dict(src),
       "resolved_mbids": len(label),
       "resolved_nodes": nodes,
       "misses": dict(miss),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_EXTERNAL_AUTHORITY_MUSICBRAINZ.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\nmusicbrainz: {len(label):,} mbids resolved over {nodes:,} nodes; misses {dict(miss)}")
for a, (l, e) in list(label.items())[:10]:
    print(f"   {a}  {e:<14} {l[:50]!r}")
print(f"{time.time()-t0:.0f}s")
