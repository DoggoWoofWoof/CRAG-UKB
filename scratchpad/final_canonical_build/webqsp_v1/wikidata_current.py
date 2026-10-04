"""Cascade step 4: CURRENT Wikidata P646 / P2671 against the unresolved residue.

    PYTHONHASHSEED=0 python .../wikidata_current.py P646
    PYTHONHASHSEED=0 python .../wikidata_current.py P2671

WHY THIS IS NOT REDUNDANT WITH SAMSUNG
  Samsung's archive is a 2014 snapshot holding 3,890,343 sameAs pairs. Wikidata today holds about
  4.47M P646 statements and 8.18M P2671 statements, so both are strictly larger, and eleven years of
  curation sit between them. P2671 in particular did not exist in 2014. The overlap with Samsung is
  expected to be heavy; it is measured rather than assumed, and the union is computed explicitly.

WHY QLEVER AND NOT THE OFFICIAL query.wikidata.org
  This was measured, not preferred. WDQS answers COUNT over the whole property in under a second but
  returns 504 on any query that has to MATERIALISE the rows: ORDER BY paging, STRSTARTS prefix
  chunks and index-range filters all failed at 60s, on both properties, including chunks as narrow
  as a three-character prefix. QLever returned all 4,470,489 P646 rows in one query in 85s. So the
  prefix-partition scheme WDQS would have needed is not merely slower, it is unavailable.

  QLever serves a periodic dump; WDQS is the live store. Their totals therefore differ slightly
  (4,470,489 vs 4,470,353 for P646 at the time of writing). Both are recorded. The difference is a
  snapshot date, not a disagreement, and a rerun can be compared against either.

WHAT THIS STEP PRODUCES
  MID -> QID. A QID is not a name. Labels are fetched in a separate step, and only for the MIDs that
  actually hit, so no bulk Wikidata label dump is downloaded on speculation.

Reads the frozen graph and the frozen overlay. Writes neither.
"""
import sys, io, os, json, time, urllib.request, urllib.parse, urllib.error
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
CACHE = f"{V3}/_acquisition/_unresolved_population.npz"
RAW = f"{V3}/_acquisition/wikidata_raw"
QLEVER = "https://qlever.cs.uni-freiburg.de/api/wikidata"
WDQS = "https://query.wikidata.org/sparql"
PFX = "PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
UA = "CRAG-freebase-name-recovery/1.0 (academic research; contact swastik9895@gmail.com)"
KIND = ["ENTITY_MID", "CVT_MEDIATOR", "OTHER"]
CHUNK = 1_000_000
Q = chr(34)
PROP = sys.argv[1] if len(sys.argv) > 1 else "P646"
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
os.makedirs(RAW, exist_ok=True)


def scalar(ep, q, timeout=180):
    """One small answer. Used only for the two independent statement totals."""
    u = ep + "?" + urllib.parse.urlencode({"query": q})
    r = urllib.request.Request(u, headers={"User-Agent": UA,
                                           "Accept": "text/tab-separated-values"})
    with urllib.request.urlopen(r, timeout=timeout) as f:
        body = f.read().decode("utf-8", "replace")
    return body.split("\n")[1].strip().strip(Q)


def unquote(v):
    """QLever TSV renders a literal as "value" and may append ^^<datatype>."""
    a = v.find(Q)
    b = v.rfind(Q)
    return v[a + 1:b] if 0 <= a < b else v


# ---------------------------------------------------------------------------- population
z = np.load(CACHE)
U, KC = z["U"], z["KC"]
NU = len(U)
print(f"{PROP}: unresolved population {NU:,} "
      f"(ENTITY_MID {int((KC==0).sum()):,}, CVT {int((KC==1).sum()):,})", flush=True)

TOTAL = int(scalar(QLEVER, PFX + "SELECT (COUNT(*) AS ?n) WHERE { ?i wdt:%s ?f }" % PROP))
try:
    LIVE = int(scalar(WDQS, "SELECT (COUNT(*) AS ?n) WHERE { ?i wdt:%s ?f }" % PROP))
except Exception as e:
    LIVE = None
    print(f"  live WDQS total unavailable ({type(e).__name__}); recording QLever only", flush=True)
print(f"{PROP}: QLever {TOTAL:,} statements; live WDQS {LIVE if LIVE is None else format(LIVE, ',')}"
      f"  ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------------------- bulk fetch + probe
hits = {}
got = 0
seen = 0
skipped = 0
part = 0
b_nid, b_qid = [], []


def flush_batch():
    """Batch the membership test and the parquet spill together.

    A scalar searchsorted per row costs ~78us against an array this size and would dominate the
    whole pass; one vectorised lookup per million rows does not."""
    global part
    if not b_nid:
        return
    u = np.fromiter(map(hash, b_nid), dtype=np.int64, count=len(b_nid))
    j = np.searchsorted(U, u)
    np.clip(j, 0, NU - 1, out=j)
    for i in np.flatnonzero(U[j] == u).tolist():
        hits.setdefault(int(u[i]), (b_nid[i], b_qid[i]))
    pq.write_table(pa.table({"node_id": pa.array(b_nid), "qid": pa.array(b_qid)}),
                   f"{RAW}/{PROP}_{part:03d}.parquet", compression="zstd")
    part += 1


url = QLEVER + "?" + urllib.parse.urlencode(
    {"query": PFX + "SELECT ?i ?f WHERE { ?i wdt:%s ?f }" % PROP})
req = urllib.request.Request(url, headers={"User-Agent": UA,
                                           "Accept": "text/tab-separated-values"})
with urllib.request.urlopen(req, timeout=1800) as resp:
    for ln, raw in enumerate(io.TextIOWrapper(resp, encoding="utf-8", errors="replace")):
        if ln == 0:
            continue                      # header row: ?i\t?f
        a, _, b = raw.rstrip("\r\n").partition("\t")
        if not b:
            continue
        seen += 1
        fb = unquote(b)
        if len(fb) < 4 or fb[0] != "/":
            skipped += 1                  # retrieved and counted, but not MID-shaped: never a hit
            continue
        got += 1
        b_nid.append(fb[1:].replace("/", ".", 1))
        b_qid.append(a.rsplit("/", 1)[-1].rstrip(">"))
        if len(b_nid) >= CHUNK:
            flush_batch()
            b_nid, b_qid = [], []
            print(f"  rows={seen:,}/{TOTAL:,} hits={len(hits):,} ({time.time()-t0:.0f}s)", flush=True)
flush_batch()
print(f"retrieved {seen:,} of {TOTAL:,} ({got:,} MID-shaped, {skipped:,} not)   hits {len(hits):,}  ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------------------- record
FETCH_COMPLETE = seen == TOTAL
by = {}
if hits:
    ks = sorted(hits)
    kc = KC[np.searchsorted(U, np.array(ks, dtype=np.int64))]
    by = {KIND[i]: int((kc == i).sum()) for i in range(3)}
    pq.write_table(pa.table({
        "node_uid": pa.array(ks, pa.int64()),
        "node_id": pa.array([hits[k][0] for k in ks]),
        "qid": pa.array([hits[k][1] for k in ks]),
        "prop": pa.array([PROP] * len(ks)).dictionary_encode(),
        "kind": pa.array([KIND[c] for c in kc.tolist()]).dictionary_encode()}),
        f"{V3}/_acquisition/wikidata_{PROP}_hits.parquet", compression="zstd")

n_ent = int((KC == 0).sum())
rec = {"schema": "WIKIDATA_CURRENT_INTERSECT/v1", "property": PROP,
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. 301,977,131 nodes / "
                       "2,062,430,072 edges unchanged; RESOLUTION_OVERLAY_V1 manifest hash "
                       "25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865 unchanged."),
       "source": {"endpoint": QLEVER,
                  "why_not_wdqs": ("measured, not preferred: query.wikidata.org answers COUNT over "
                                   "the whole property in <1s but returns 504/502 on every query "
                                   "that materialises rows -- ORDER BY paging, STRSTARTS prefix "
                                   "chunks and index-range filters all failed at 60s on both "
                                   "properties, including 3-character prefixes."),
                  "qlever_statement_total": TOTAL,
                  "wdqs_live_statement_total": LIVE,
                  "SNAPSHOT_NOTE": ("QLever serves a periodic Wikidata dump; WDQS is the live "
                                    "store. Any difference between the two totals is a snapshot "
                                    "date, not a disagreement. Both are recorded so a rerun is "
                                    "comparable against either.")},
       "statements_retrieved": seen, "mid_shaped_values": got, "non_mid_shaped_values": skipped,
       "FETCH_COMPLETE": FETCH_COMPLETE,
       "unresolved_nodes_probed": int(NU), "unresolved_entity_mids": n_ent,
       "HIT_N": len(hits), "hits_by_kind": by,
       "entity_mid_hit_pct": round(100 * by.get("ENTITY_MID", 0) / n_ent, 4) if n_ent else 0,
       "WHAT_A_HIT_IS": ("a MID -> QID correspondence against a node that currently has no attested "
                         "name. A QID is not a name; labels are a separate step, run only for these "
                         "QIDs."),
       "PROVENANCE_IF_USED": "CURRENT_WIKIDATA_EXACT",
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_WIKIDATA_{PROP}_INTERSECT.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\n{PROP}_HIT_N = {len(hits):,}   retrieved {seen:,}/{TOTAL:,} COMPLETE={FETCH_COMPLETE}")
for k, v in by.items():
    print(f"   {v:>12,}  {k}")
print(f"ENTITY_MID hit rate: {rec['entity_mid_hit_pct']}% of {n_ent:,}")
print(f"{time.time()-t0:.0f}s")
