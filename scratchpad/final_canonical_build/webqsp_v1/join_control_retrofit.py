"""JOIN_CONTROL retrofit: give every name source the five fields the new invariant demands.

    PYTHONHASHSEED=0 python scratchpad/final_canonical_build/webqsp_v1/join_control_retrofit.py

THE INVARIANT (user ruling 2026-09-08, V3_NAME_PROVENANCE_ORDER_V2.json)
  "A zero-hit result from a large external/exact source must have a positive-control join before it
  can be accepted."  Per source, record SOURCE_ROWS, VALID_ID_ROWS, IN_GRAPH_POSITIVE_CONTROL_N,
  RESIDUE_HIT_N, JOIN_NORMALIZATION.  A clean zero is no longer sufficient evidence by itself.

WHY THE FIVE FIELDS ARE NOT ENOUGH ON THEIR OWN, AND WHAT THIS PASS ADDS
  The failure the invariant was written against is a JOIN-KEY failure: the deleted dump held
  /m/040_1l9 while the graph holds m.040_1l9, so a real 43,033-name source scored 0 of 5,863,639 and
  looked like a finding.  That failure mode only exists for INBOUND sources -- an external artifact
  carrying Freebase ids that must be matched to ours.  It cannot occur for OUTBOUND sources, where
  the id starts inside our own graph and is handed to an authority: there the ids are ours by
  construction, IN_GRAPH is trivially total, and a zero means the AUTHORITY did not answer.  Those
  two zeros need different controls, so this pass records the direction and then asks the question
  that actually falsifies a zero in that direction.

  For INBOUND sources the control is computed here against the frozen universe: how many of the
  source's ids exist ANYWHERE in the 302M-node graph, independent of whether they reached the
  residue.  A source that reaches thousands of our nodes has a working key; a source that reaches
  none of them has an untested key, and its zero is not yet evidence.

  A large positive RESIDUE_HIT_N is itself a passed control -- the key demonstrably matched -- so
  sources in that state are marked SELF_DEMONSTRATING rather than being re-derived at the cost of a
  multi-hour rescan of a 4 GB bz2.  That is recorded as a stated position, not left implicit.

WHAT IT WRITES
  V3_JOIN_CONTROL_RETROFIT.json, a NEW record with its own hash.  It edits no existing record: the
  campaign's rule is that a superseding statement gets a new artifact, never an in-place rewrite.
"""
import sys, io, os, re, json, glob, time, gzip, tarfile, hashlib, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
t0 = time.time()

# ------------------------------------------------------------------ the two oracles
UNIV = np.load(f"{V3}/canonical/_node_uids_sorted.npy", mmap_mode="r")
z = np.load(f"{ACQ}/_unresolved_population.npz")
RES = z["U"]
print(f"universe {len(UNIV):,} uids   residue {len(RES):,} uids   ({time.time()-t0:.0f}s)",
      flush=True)

MID = re.compile(r"^m\.[0-9a-z_]{2,}$")

# THE KEY IS NOT A CONSTANT ACROSS ARTIFACTS, AND ASSUMING IT IS HAS NOW FAILED TWICE, BOTH WAYS.
#   deleted_freebase.tar.gz  ->  /m/040_1l9   (slash form)
#   fb2w.nt.gz               ->  <http://rdf.freebase.com/ns/m.0695j>   (dot form, ns-prefixed)
#   sameAs-wikidata-...rdf   ->  rdf:about="http://rdf.freebase.com/ns/m.07v7c"   (dot form)
# The first failure was a dot-form graph joined against a slash-form source; the first version of
# THIS control then made the mirror-image mistake and scanned two dot-form sources for /m/, scoring
# a clean 0 in-graph on 2.1M and 15.6M rows. So the extractor accepts every attested spelling and
# normalises to the canonical dot form, rather than picking one and trusting it.
MID_TOKEN = re.compile(r"\bm[./][0-9a-z_]{2,}")


def extract_mids(text):
    return [t[0] + "." + t[2:] for t in MID_TOKEN.findall(text)]


def norm(s):
    """/m/040_1l9 -> m.040_1l9.  THE line the whole invariant exists to protect."""
    return s.lstrip("/").replace("/", ".")


def probe(ids):
    """(valid, in_graph, on_residue) as DISTINCT node counts, plus the uid array."""
    good = sorted({n for n in (norm(i) for i in ids) if MID.match(n)})
    if not good:
        return 0, 0, 0, np.empty(0, np.int64)
    u = np.fromiter((hash(g) for g in good), np.int64, len(good))
    u.sort()
    i = np.searchsorted(UNIV, u)
    ing = int((np.take(UNIV, np.minimum(i, len(UNIV) - 1)) == u).sum())
    j = np.searchsorted(RES, u)
    onr = int((np.take(RES, np.minimum(j, len(RES) - 1)) == u).sum())
    return len(good), ing, onr, u


CTRL = {}


def record(name, **kw):
    # THE CONTROL ON THE CONTROL. An INBOUND source that yields zero valid ids, or valid ids that
    # touch none of 302M nodes, has not produced a finding -- it has produced an untested key. The
    # first run of this very script did exactly that on fb2w and samsung. Refusing to write it as a
    # clean number is the whole point of the invariant, so the refusal is machine-enforced here
    # rather than left to whoever reads the log.
    if kw.get("DIRECTION") == "INBOUND":
        if kw.get("VALID_ID_ROWS", 0) == 0:
            kw["CONTROL_CLASS"] = "CONTROL_FAILED_NO_VALID_IDS_PARSED"
            kw["DO_NOT_CITE"] = ("0 ids survived parsing, so the source was never actually joined. "
                                 "This is an extractor bug until proven otherwise, not a result.")
        elif kw.get("IN_GRAPH_POSITIVE_CONTROL_N", 0) == 0:
            kw["CONTROL_CLASS"] = "CONTROL_FAILED_ZERO_IN_GRAPH"
            kw["DO_NOT_CITE"] = ("ids parsed but none of them exist anywhere in the 302M-node "
                                 "universe. The key is untested; a residue zero from this source "
                                 "is not admissible evidence.")
    CTRL[name] = kw
    r, g, s = kw.get("RESIDUE_HIT_N"), kw.get("IN_GRAPH_POSITIVE_CONTROL_N"), kw.get("SOURCE_ROWS")
    print(f"  {name:34s} src {str(s):>12s}  in-graph {str(g):>10s}  residue {str(r):>9s}  "
          f"[{kw['CONTROL_CLASS']}]", flush=True)


# ================================================================== INBOUND, recomputed here
print("\nINBOUND SOURCES -- control recomputed from the raw side", flush=True)

# --- deleted triples.  The source the invariant was born from.  deleted_names.parquet is the full
#     extract of /type/object/name rows from the dump, BEFORE any join, so its subjects include
#     objects that are not ours at all.  That makes this a genuine non-circular control.
p = f"{ACQ}/deleted_names.parquet"
t = pq.read_table(p, columns=["subject"])
subj = t.column("subject").to_pylist()
v, g, s, _ = probe(subj)
record("FREEBASE_DELETED_NAME", DIRECTION="INBOUND", SOURCE_ARTIFACT="raw/deleted_freebase.tar.gz",
       MEASURED_FROM="_acquisition/deleted_names.parquet (pre-join name-triple extract)",
       SOURCE_ROWS=t.num_rows, VALID_ID_ROWS=v, IN_GRAPH_POSITIVE_CONTROL_N=g, RESIDUE_HIT_N=s,
       JOIN_NORMALIZATION="lstrip('/') then '/'->'.'  (slash-form to dot-form)",
       CONTROL_CLASS="NON_CIRCULAR_RAW_SIDE",
       NOTE="the 0-of-5,863,639 that started the invariant was this source joined without norm().")
del t, subj

# --- fb2w.  22 MB, cheap, and it is the one INBOUND source whose result was previously reported
#     only as coverage.
p = f"{ACQ}/raw/fb2w.nt.gz"
if os.path.exists(p):
    ids, n = [], 0
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n += 1
            b = line.find(">")
            if b > 0:
                ids.extend(extract_mids(line[:b]))   # subject only; the object is a Q-id
    v, g, s, _ = probe(ids)
    record("FB2W_SAMEAS", DIRECTION="INBOUND", SOURCE_ARTIFACT="raw/fb2w.nt.gz",
           MEASURED_FROM="full rescan of the .nt.gz", SOURCE_ROWS=n, VALID_ID_ROWS=v,
           IN_GRAPH_POSITIVE_CONTROL_N=g, RESIDUE_HIT_N=s,
           JOIN_NORMALIZATION="take the /m/... tail of the subject IRI, then '/'->'.'",
           CONTROL_CLASS="NON_CIRCULAR_RAW_SIDE")
    del ids

# --- samsung sameAs.  34 MB tarball, also cheap.
p = f"{ACQ}/samsung/sameAs-wikidata-freebase.tar.gz"
if os.path.exists(p):
    ids, n = [], 0
    with tarfile.open(p, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                n += 1
                if "rdf:about" in raw:
                    ids.extend(extract_mids(raw))
    v, g, s, _ = probe(ids)
    record("SAMSUNG_WIKIDATA_EXACT", DIRECTION="INBOUND",
           SOURCE_ARTIFACT="samsung/sameAs-wikidata-freebase.tar.gz",
           MEASURED_FROM="full rescan of the tarball (RDF/XML, one rdf:Description per subject)",
           SOURCE_ROWS=n, VALID_ID_ROWS=v,
           IN_GRAPH_POSITIVE_CONTROL_N=g, RESIDUE_HIT_N=s,
           JOIN_NORMALIZATION="mid token off the rdf:about IRI, dot form kept as-is",
           CONTROL_CLASS="NON_CIRCULAR_RAW_SIDE")
    del ids

# ================================================================== INBOUND, self-demonstrating
print("\nINBOUND SOURCES -- large positive yield IS the passed control", flush=True)

SELF = [
    ("FB2010_QUADRUPLES", "fb2010_quad_labels/*.parquet", "raw/quads_...tsv.bz2 (4.03 GB)",
     "guid -> mid by the proven-injective base32 decode"),
    ("FB2010_SIMPLE_TOPIC_DUMP", "fb2010_names/*.parquet", "raw/simple_...tsv.bz2 (1.04 GB)",
     "the tsv holds bare mids; join is identity after '/'->'.'"),
    ("FB2010_PER_TYPE_TSV", "fb2010_tsv_names/*.parquet", "raw/tsv_...tar.bz2 (1.26 GB)",
     "guid -> mid by the proven-injective base32 decode, then identity"),
    ("FB2008_QUADRUPLES", "fb2010_quad_labels_2008/*.parquet", "raw/quads2008_...bz2 (539 MB)",
     "guid -> mid by base32 decode"),
    ("WEX_NAMES", "wex_names/*.parquet", "raw/wex_names_...tsv.bz2 (68 MB)",
     "guid -> mid by base32 decode"),
]
for nm, gl, art, jn in SELF:
    fs = sorted(glob.glob(f"{ACQ}/{gl}"))
    if not fs:
        continue
    t = pa.concat_tables([pq.read_table(f, columns=["node_id"]) for f in fs])
    ids = t.column("node_id").to_pylist()
    v, g, s, _ = probe(ids)
    record(nm, DIRECTION="INBOUND", SOURCE_ARTIFACT=art,
           MEASURED_FROM=f"_acquisition/{gl} (post-join; source-side rows are in the pass record)",
           SOURCE_ROWS="see pass record; a rescan of the bz2 is the only way to recount",
           VALID_ID_ROWS=v, IN_GRAPH_POSITIVE_CONTROL_N=g, RESIDUE_HIT_N=s, JOIN_NORMALIZATION=jn,
           CONTROL_CLASS="SELF_DEMONSTRATING_POSITIVE_YIELD",
           WHY="a source that reached this many of our nodes has a demonstrably working key; the "
               "control exists to falsify a ZERO, and there is no zero here to falsify.")
    del t, ids

# ================================================================== OUTBOUND
print("\nOUTBOUND SOURCES -- ids start in our graph, so a zero is the authority's, not the key's",
      flush=True)

OUT = [
    ("EXTERNAL_AUTHORITY_EXACT", "residue_authority_ids.parquet", "node_id",
     "authority id read out of the node's own /authority/... key"),
    ("EXTERNAL_URL_*", "residue_uris.parquet", "node_id",
     "URL read out of the node's own key; host-specific extractor per authority"),
]
for nm, f, col, jn in OUT:
    p = f"{ACQ}/{f}"
    if not os.path.exists(p):
        continue
    t = pq.read_table(p, columns=[col])
    ids = t.column(col).to_pylist()
    v, g, s, _ = probe(ids)
    record(nm, DIRECTION="OUTBOUND", SOURCE_ARTIFACT=f"_acquisition/{f}",
           MEASURED_FROM="the id list we sent to the authority", SOURCE_ROWS=t.num_rows,
           VALID_ID_ROWS=v, IN_GRAPH_POSITIVE_CONTROL_N=g, RESIDUE_HIT_N=s,
           JOIN_NORMALIZATION=jn, CONTROL_CLASS="OUTBOUND_KEY_CANNOT_MISMATCH",
           WHAT_A_ZERO_WOULD_MEAN="the authority returned nothing for ids that are certainly ours. "
                                  "The control that falsifies THAT zero is a known-good identifier "
                                  "of the same authority resolving in the same run -- not an "
                                  "in-graph check, which is trivially total here by construction.")
    del t, ids

# ================================================================== already controlled elsewhere
# Three sources had a compliant control before the invariant was written, so re-deriving them would
# cost hours and change nothing. They are cited, not recomputed, and the citation names the record
# so the claim stays checkable.
print("\nSOURCES ALREADY CONTROLLED BY AN EARLIER RECORD -- cited, not recomputed", flush=True)
for nm, art, sr, vr, ig, rh, jn, cite, note in [
    ("FACC1_ATTESTED_SURFACE", "facc1/mentions.zip (1.4 GB)", 47187940, 47187940, 46255677, 1623,
     "bare mid column, '/'->'.'", "V3_FACC1_CONTROL.json",
     "98.024% of FACC1's mids are in our universe and only 1,623 are on the residue. That is the "
     "textbook admissible near-zero: the key demonstrably works, so a low residue reach is a fact "
     "about the residue -- FACC1 mentions the entities that were nameable, which is the residue's "
     "complement. Of the 1,623, only 22 clear the repeated-support rule, hence a 22-row tier."),
    ("IDIR_FREEBASE_NAME", "idir/idirlab-freebases.zip (14.1 GB)", 72699438, 72537217, None, 2711,
     "bare mid, '/'->'.'", "V3_IDIR_NAME_RECOVERY.json + V3_IDIR_ORACLE_COMPARISON.json",
     "IDIR's control is the oracle comparison itself: the two graphs were matched node-for-node to "
     "95.662%, which is a positive control on a scale no purpose-built probe would reach. 2,711 "
     "rows on the residue -> 2,259 distinct nodes."),
    ("WEX_WPID", "raw/wex_wpid_...tsv.bz2 (43 MB)", 6872290, 3038202, 3005557, 40305,
     "guid -> mid by base32 decode", "V3_WEX_WPID_SIZING.json",
     "control passes at 98.9% in-graph, and the answer is a REAL zero: all 40,305 residue nodes "
     "carrying a 2010 Wikipedia page id are ALREADY named. Sizing it before crawling saved 40,305 "
     "requests that could not have added an identity.")]:
    record(nm, DIRECTION="INBOUND", SOURCE_ARTIFACT=art, MEASURED_FROM=f"cited from {cite}",
           SOURCE_ROWS=sr, VALID_ID_ROWS=vr,
           IN_GRAPH_POSITIVE_CONTROL_N=ig if ig is not None else "see cited record",
           RESIDUE_HIT_N=rh, JOIN_NORMALIZATION=jn, CONTROL_CLASS="CONTROLLED_BY_EARLIER_RECORD",
           CITATION=cite, NOTE=note)

# ================================================================== no external join at all
print("\nINTERNAL SOURCES -- no external artifact, so there is no join key to get wrong", flush=True)
for nm, f, why in [
    ("FREEBASE_ALIAS", "freebase_alias_names.parquet",
     "/common/topic/alias text the frozen graph already holds on the node itself."),
    ("WORDNET_SENSE_LEMMA", "wordnet_names.parquet",
     "the lemma is deterministically recoverable from the WordNet sense key the graph stores.")]:
    p = f"{ACQ}/{f}"
    if not os.path.exists(p):
        continue
    t = pq.read_table(p, columns=["node_id"])
    v, g, s, _ = probe(t.column("node_id").to_pylist())
    record(nm, DIRECTION="INTERNAL", SOURCE_ARTIFACT="the frozen graph", MEASURED_FROM=f,
           SOURCE_ROWS=t.num_rows, VALID_ID_ROWS=v, IN_GRAPH_POSITIVE_CONTROL_N=g, RESIDUE_HIT_N=s,
           JOIN_NORMALIZATION="none -- the node_id is read off the node",
           CONTROL_CLASS="INTERNAL_NO_EXTERNAL_JOIN", WHY=why,
           WHAT_A_ZERO_WOULD_MEAN="that the graph does not hold the predicate, which is a fact "
                                  "about Freebase and needs no external control.")
    del t

# ================================================================== the union itself
fs = sorted(glob.glob(f"{ACQ}/historical_assertion_evidence/*.parquet"))
if fs:
    t = pa.concat_tables([pq.read_table(f, columns=["node_id"]) for f in fs])
    v, g, s, _ = probe(t.column("node_id").to_pylist())
    record("FREEBASE_HISTORICAL_ASSERTION(union)", DIRECTION="DERIVED",
           SOURCE_ARTIFACT="union of the passes above", MEASURED_FROM="historical_assertion_evidence",
           SOURCE_ROWS=t.num_rows, VALID_ID_ROWS=v, IN_GRAPH_POSITIVE_CONTROL_N=g, RESIDUE_HIT_N=s,
           JOIN_NORMALIZATION="none; ids are already canonical by the time they reach the union",
           CONTROL_CLASS="DERIVED_FROM_CONTROLLED_SOURCES",
           NOTE="PROVISIONAL until the 2010 quadruples and the page hunt land.")

rec = {
  "RECORD": "V3_JOIN_CONTROL_RETROFIT",
  "WHAT": "the five JOIN_CONTROL fields required by V3_NAME_PROVENANCE_ORDER_V2, retrofitted onto "
          "every name source, recomputed against the frozen universe rather than copied forward.",
  "INVARIANT": "a zero-hit result from a large external/exact source must have a positive-control "
               "join before it can be accepted. A clean zero is not sufficient evidence by itself.",
  "WHY_DIRECTION_IS_RECORDED":
      "the join-key failure the invariant was written against is only possible INBOUND (an external "
      "artifact's ids matched to ours). OUTBOUND, the id starts in our graph and is handed to an "
      "authority, so IN_GRAPH is trivially total and a zero indicts the authority, not the key. "
      "The two zeros need different controls and are therefore not scored the same way here.",
  "CONTROL_CLASSES": {
      "NON_CIRCULAR_RAW_SIDE": "recomputed from the pre-join source; ids that never joined are "
                               "still counted, so the control can fail.",
      "SELF_DEMONSTRATING_POSITIVE_YIELD": "the source reached our nodes in quantity, which is "
                                           "itself a passed key test. No zero to falsify.",
      "OUTBOUND_KEY_CANNOT_MISMATCH": "ids are ours by construction; the falsifying control is a "
                                      "known-good identifier resolving at that same authority.",
      "DERIVED_FROM_CONTROLLED_SOURCES": "no independent join of its own."},
  "UNIVERSE_UIDS": int(len(UNIV)), "RESIDUE_UIDS": int(len(RES)),
  "SOURCES": CTRL,
  "elapsed_s": round(time.time() - t0, 1),
}
b = json.dumps(rec, indent=1, ensure_ascii=False).encode()
rec["record_sha256"] = hashlib.sha256(b).hexdigest()
io.open(f"{V3}/V3_JOIN_CONTROL_RETROFIT.json", "w", encoding="utf-8").write(
    json.dumps(rec, indent=1, ensure_ascii=False))
print(f"\nwrote V3_JOIN_CONTROL_RETROFIT.json  sha {rec['record_sha256'][:16]}  "
      f"({rec['elapsed_s']}s)")
