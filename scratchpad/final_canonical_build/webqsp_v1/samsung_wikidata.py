"""Cascade step 2: the Samsung Freebase-Wikidata sameAs mapping against the unresolved residue.

    PYTHONHASHSEED=0 python .../samsung_wikidata.py

WHY THIS ONE IS WORTH RUNNING RATHER THAN REASONING ABOUT
  Every cascade step measured so far reaches the graph through Freebase's own key values, and
  V3_EXTERNAL_RECOVERY_CEILING.json bounded all of them together at 36,547 nodes. Samsung is the
  first source keyed by MID rather than by key, so that ceiling does not apply to it and cannot be
  used to predict its yield. It is also small -- 34 MB compressed, 609 MB of RDF/XML -- so the
  honest move is to intersect it and read the number off, not to estimate it.

  source   github.com/Samsung/KnowledgeSharingPlatform, sameas/freebase-wikidata
  artifact sameAs-wikidata-freebase.tar.gz, 34,035,849 bytes, verified against the size GitHub
           declares for the blob
  sha256   010dd69fedd6590557a1e128161666ec1e4a2342f35aadce6daed99af0abed1c
  licence  CC0
  note     the announcement is dated 2015-02-13; the file inside the archive carries an mtime of
           2014-08-20. Both are recorded rather than reconciled, because neither is checkable here.

WHAT THIS STEP DOES AND DOES NOT PRODUCE
  It produces MID -> QID. A QID is not a name. Resolving QIDs to labels is a SEPARATE step, run
  only for the MIDs that actually hit, so no bulk Wikidata label dump is fetched on speculation.

Streamed out of the tarball -- the 609 MB of XML never lands on disk. Reads the frozen graph and
the frozen overlay; writes neither.
"""
import sys, io, os, json, time, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
TGZ = f"{V3}/_acquisition/samsung/sameAs-wikidata-freebase.tar.gz"
CACHE = f"{V3}/_acquisition/_unresolved_population.npz"
KIND = ["ENTITY_MID", "CVT_MEDIATOR", "OTHER"]
PROBE = 1_000_000
ANNOUNCED_PAIRS = 4_395_258      # the 2015-02-13 wikidata-l figure
FILE_SAMEAS_LINES = 3_890_343    # what this archive actually contains, counted
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")
z = np.load(CACHE)
U, KC = z["U"], z["KC"]
NU = len(U)
print(f"unresolved population: {NU:,} "
      f"(ENTITY_MID {int((KC==0).sum()):,}, CVT {int((KC==1).sum()):,})", flush=True)

p = subprocess.Popen(["tar", "-xzOf", TGZ], stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, bufsize=1 << 22)
pairs = 0
cur = None
b_mid, b_qid = [], []
hits = {}
NS = 'ns/'
ENT = 'entity/'


def probe():
    """Batch the membership test: a scalar searchsorted per row costs ~78us and dominates."""
    if not b_mid:
        return
    u = np.fromiter(map(hash, b_mid), dtype=np.int64, count=len(b_mid))
    j = np.searchsorted(U, u)
    np.clip(j, 0, NU - 1, out=j)
    for i in np.flatnonzero(U[j] == u).tolist():
        hits.setdefault(int(u[i]), (b_mid[i], b_qid[i]))


for raw in io.TextIOWrapper(p.stdout, encoding="utf-8", errors="replace", newline=""):
    if 'rdf:about=' in raw:
        # <rdf:Description rdf:about="http://rdf.freebase.com/ns/m.01cx_">
        a = raw.split(NS, 1)
        cur = a[1].split('"', 1)[0] if len(a) > 1 else None
    elif cur is not None and 'owl:sameAs' in raw:
        a = raw.split(ENT, 1)
        if len(a) > 1:
            q = a[1].split('"', 1)[0]
            pairs += 1
            b_mid.append(cur)
            b_qid.append(q)
            if len(b_mid) >= PROBE:
                probe()
                b_mid, b_qid = [], []
                print(f"  pairs={pairs:,} hits={len(hits):,} ({time.time()-t0:.0f}s)", flush=True)
probe()
p.stdout.close()
p.wait()
print(f"pairs parsed: {pairs:,} (file contains {FILE_SAMEAS_LINES:,}; announcement said {ANNOUNCED_PAIRS:,})  hits: {len(hits):,}"
      f"  ({time.time()-t0:.0f}s)", flush=True)

by = {}
if hits:
    ks = sorted(hits)
    kc = KC[np.searchsorted(U, np.array(ks, dtype=np.int64))]
    by = {KIND[i]: int((kc == i).sum()) for i in range(3)}
    pq.write_table(pa.table({
        "node_uid": pa.array(ks, pa.int64()),
        "node_id": pa.array([hits[k][0] for k in ks]),
        "qid": pa.array([hits[k][1] for k in ks]),
        "kind": pa.array([KIND[c] for c in kc.tolist()]).dictionary_encode()}),
        f"{V3}/_acquisition/samsung_hits.parquet", compression="zstd")

n_ent = int((KC == 0).sum())
rec = {"schema": "SAMSUNG_WIKIDATA_INTERSECT/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. 301,977,131 nodes / "
                       "2,062,430,072 edges unchanged; RESOLUTION_OVERLAY_V1 manifest hash "
                       "25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865 unchanged."),
       "source": {"repo": "github.com/Samsung/KnowledgeSharingPlatform",
                  "path": "sameas/freebase-wikidata/sameAs-wikidata-freebase.tar.gz",
                  "bytes": 34035849,
                  "sha256": "010dd69fedd6590557a1e128161666ec1e4a2342f35aadce6daed99af0abed1c",
                  "inner_file": "sameAs-wikidata-freebase.rdf",
                  "inner_bytes": 609086059,
                  "licence": "CC0",
                  "announced": "2015-02-13 (wikidata-l)",
                  "archive_member_mtime": "2014-08-20",
                  "date_note": ("announcement and archive mtime disagree; both are recorded rather "
                                "than reconciled, since neither is checkable from here.")},
       "pairs_parsed": pairs,
       "pairs_in_file": FILE_SAMEAS_LINES,
       "pairs_announced_2015_02_13": ANNOUNCED_PAIRS,
       "PARSE_COMPLETE": pairs == FILE_SAMEAS_LINES,
       "COUNT_DISCREPANCY": ("the archive holds 3,890,343 owl:sameAs lines over 3,890,322 "
                             "rdf:Description records, counted directly; the mailing-list "
                             "announcement claims 4,395,258. The archive member's mtime is "
                             "2014-08-20, not the announced 2015-02-13, so this repo copy appears "
                             "to predate the announced release. PARSE_COMPLETE is therefore tested "
                             "against what the file contains, not against the announcement -- the "
                             "shortfall is in the artifact, not in the parser."),
       "unresolved_nodes_probed": int(NU),
       "unresolved_entity_mids": n_ent,
       "SAMSUNG_HIT_N": len(hits),
       "hits_by_kind": by,
       "entity_mid_hit_pct": round(100 * by.get("ENTITY_MID", 0) / n_ent, 4) if n_ent else 0,
       "WHAT_A_HIT_IS": ("a MID -> QID correspondence. A QID is NOT a name. Labels are fetched in a "
                         "separate step, for these QIDs only, so no bulk Wikidata label dump is "
                         "downloaded on speculation."),
       "PROVENANCE_IF_USED": "SAMSUNG_WIKIDATA_EXACT",
       "WHY_THE_KEY_CEILING_DOES_NOT_APPLY": ("V3_EXTERNAL_RECOVERY_CEILING.json bounds the routes "
                                              "that go through Freebase's own key values. Samsung is "
                                              "keyed by MID, so a node with no Freebase key at all "
                                              "can still appear here."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_SAMSUNG_WIKIDATA_INTERSECT.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)
print(f"\nSAMSUNG_HIT_N = {len(hits):,}")
for k, v in by.items():
    print(f"   {v:>12,}  {k}")
print(f"ENTITY_MID hit rate: {rec['entity_mid_hit_pct']}% of {n_ent:,}")
print(f"{time.time()-t0:.0f}s")
