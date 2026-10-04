"""What the residue IS, from the keys the graph already holds -- and what those keys still name.

    PYTHONHASHSEED=0 python .../residue_key_census.py

WHY THIS COMES BEFORE ANY FURTHER EXTERNAL ACQUISITION
  Five external sources have been worked through and they collectively name 13,788 of 17,809,849
  residual entity MIDs. Before going after a sixth, the obvious question is what the graph already
  says about these nodes. It turns out to say a great deal: 2.68M of them carry a /type/object/key,
  and those keys are not opaque. They record which bulk load created the node and, in one large
  namespace, they contain a human-readable lemma outright.

TWO DIFFERENT KINDS OF ANSWER COME OUT OF THE SAME SCAN
  RECOVERY   a key whose text contains an attested human-readable identity, or an external
             authority id that can be resolved to one.
  EVIDENCE   a key that instead demonstrates the node never had a name: the MusicBrainz NGS loader
             wrote "..._cvt_id_<uuid>::<uuid>", the freeq loader wrote "dummy_cvt_/m/x/m/y", and
             others encode a pair of MIDs plus a date. Those are reified relationships. This is
             stronger than our own CVT_MEDIATOR label, which is a structural inference drawn from
             relation types; here the loader that created the node says so in the node's own key.

  The second kind is what the stopping criterion needs. The goal was to keep going until what
  remains "consists only of objects that demonstrably never had names -- primarily CVTs/internal
  records", and a declaration in the source is the demonstration.

WHAT IS AND IS NOT CLAIMED FOR THE WORDNET LEMMAS
  /user/jamie/wordnet/sensekey/'spaghetti_squash%1:20:00::' is a WordNet sense key in WordNet's own
  documented format; the substring before % is the lemma. Extracting it recovers a human-readable
  identity for a word sense. It is NOT the /type/object/name value Freebase held, and the namespace
  is a user namespace rather than a Freebase authority, so is_original_name stays false and the
  provenance records the namespace explicitly.

Reads the frozen graph and the frozen overlay. Writes neither.
"""
import sys, io, os, re, json, glob, time, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
RESIDUE_ENTITY_MIDS = 17_809_849
ESC = re.compile(r"\$([0-9A-Fa-f]{4})")
WN = re.compile(r"^/user/jamie/wordnet/sensekey/'(.+?)%")
VAR = re.compile(r"_var_(.*)$")
HEX = re.compile(r"^[0-9a-f]{6,}\|?_?$")
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")


def dec(s):
    """Freebase key escaping: $002F -> '/'. Documented and reversible, so this recovers."""
    return ESC.sub(lambda m: chr(int(m.group(1), 16)), s)


def classify(k, d):
    """One key -> one bucket. Buckets are either RECOVERABLE, INTERNAL or UNKNOWN downstream."""
    if k.startswith("/user/jamie/wordnet/sensekey"):
        return "wordnet_sensekey" if WN.match(d) else "wordnet_unparsed"
    if k.startswith("/authority/"):
        return "authority:" + k.split("/")[2]
    if k.startswith("/wikipedia/"):
        return "wikipedia:" + k.split("/")[2]
    if k.startswith("/uri/"):
        return "uri_reference"          # a URL, not a name: kept whole, not split into per-URL buckets
    if k.startswith("/dataworld/freeq"):
        g = VAR.search(d)
        if not g:
            return "freeq_other"
        v = g.group(1)
        if v.startswith("dummy_cvt"):
            return "freeq_declared_dummy_cvt"
        if "_cvt_id_" in v or v.startswith("mb_ngs:"):
            return "freeq_musicbrainz_cvt"
        if v.startswith("/m/") or v.startswith("/g/"):
            return "freeq_mid_pair"
        if HEX.match(v):
            return "freeq_hex_blob"
        if re.fullmatch(r"[\d.]+", v):
            return "freeq_literal_value"
        return "freeq_other_text"
    if k.startswith("/base/medicaldrugs"):
        return "base_medicaldrugs_load"
    if k.startswith("/base/fbontology"):
        return "base_fbontology_metaschema"
    if k.startswith("/base/wsjtopics"):
        return "base_wsjtopics_articleid"
    if k.startswith("/user/"):
        return "other_user_namespace"
    return "other:" + "/".join(k.split("/")[:3])


# a bucket is one of these three; anything unlisted counts as UNKNOWN and is reported as such
RECOVERABLE = {"wordnet_sensekey"}
# freeq var values look tempting -- some decode to '1,000,000 Night (ラスベガスのテーマ)' -- but they
# are lossy at BOTH ends. '_var_pringfield (medley): ... / Gracie Fi' is missing the leading S of
# Springfield and is cut off at a key-length limit; '_var_kit #1' is Skit #1. A truncated string is
# not the name, so this bucket is refused rather than mined.
REFUSED = {"freeq_other_text"}
INTERNAL = {"freeq_declared_dummy_cvt", "freeq_musicbrainz_cvt", "freeq_mid_pair",
            "freeq_hex_blob", "freeq_literal_value", "base_medicaldrugs_load",
            "base_fbontology_metaschema", "base_wsjtopics_articleid"}

z = np.load(f"{ACQ}/_unresolved_population.npz")
U, KC = z["U"], z["KC"]
NU = len(U)

src = sorted(glob.glob(f"{ACQ}/residue_keys/*.parquet"))
if not src:
    src = [f"{ACQ}/unnamed_keys.parquet"]
    print("NOTE: complete residue_keys/ not present; falling back to the unnamed_keys extract, "
          "which is a subset and will UNDERCOUNT.", flush=True)
print(f"reading {len(src)} key file(s)  ({time.time()-t0:.0f}s)", flush=True)

bucket = collections.Counter()
node_bucket = {}                 # node_uid -> best bucket seen (recoverable wins)
wn_rows = {}
auth_rows = collections.defaultdict(list)
for fp in src:
    t = pq.read_table(fp)
    cols = t.schema.names
    ids = t["node_id"].to_pylist()
    ks = t["key"].to_pylist()
    if "node_uid" in cols:
        us = t["node_uid"].to_pylist()
    else:
        us = [hash(x) for x in ids]
    for nid, k, u in zip(ids, ks, us):
        j = int(np.searchsorted(U, u))
        if j >= NU or U[j] != u or KC[j] != 0:      # ENTITY_MID residue only
            continue
        d = dec(k)
        b = classify(k, d)
        bucket[b] += 1
        prev = node_bucket.get(u)
        if prev is None or (b in RECOVERABLE and prev not in RECOVERABLE):
            node_bucket[u] = b
        if b == "wordnet_sensekey":
            wn_rows.setdefault(u, (nid, WN.match(d).group(1).replace("_", " ").strip()))
        elif b.startswith("authority:") or b.startswith("wikipedia:"):
            auth_rows[b].append((u, nid, d))

print(f"residue ENTITY_MID nodes with >=1 key: {len(node_bucket):,}  "
      f"({100*len(node_bucket)/RESIDUE_ENTITY_MIDS:.2f}% of the entity residue)  "
      f"({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ recovered: wordnet lemmas
if wn_rows:
    rows = sorted(wn_rows.items())
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in rows], pa.int64()),
        "node_id": pa.array([r[1][0] for r in rows]),
        "display_name": pa.array([r[1][1] for r in rows])}),
        f"{ACQ}/wordnet_names.parquet", compression="zstd")

# ------------------------------------------------------------------ carried forward: authority ids
flat = [(u, n, d, b) for b, v in auth_rows.items() for (u, n, d) in v]
if flat:
    flat.sort()
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in flat], pa.int64()),
        "node_id": pa.array([r[1] for r in flat]),
        "key": pa.array([r[2] for r in flat]),
        "bucket": pa.array([r[3] for r in flat]).dictionary_encode()}),
        f"{ACQ}/residue_authority_ids.parquet", compression="zstd")

nodes_by_class = collections.Counter()
for u, b in node_bucket.items():
    nodes_by_class["RECOVERABLE" if b in RECOVERABLE else
                   "REFUSED_LOSSY" if b in REFUSED else
                   "URI_REFERENCE" if b == "uri_reference" else
                   "INTERNAL_DECLARED" if b in INTERNAL else
                   "EXTERNAL_ID" if b.startswith(("authority:", "wikipedia:")) else
                   "UNKNOWN"] += 1

rec = {"schema": "RESIDUE_KEY_CENSUS/v1",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. RESOLUTION_OVERLAY_V1 "
                       "manifest hash 25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865"
                       " unchanged."),
       "SOURCE_COMPLETE": bool(glob.glob(f"{ACQ}/residue_keys/*.parquet")),
       "residue_entity_mids": RESIDUE_ENTITY_MIDS,
       "entity_residue_nodes_with_a_key": len(node_bucket),
       "pct_of_entity_residue": round(100 * len(node_bucket) / RESIDUE_ENTITY_MIDS, 4),
       "nodes_by_class": dict(nodes_by_class),
       "key_rows_by_bucket": dict(bucket.most_common()),
       "RECOVERED": {"wordnet_sensekey_lemmas": len(wn_rows),
                     "PROVENANCE": "WORDNET_SENSE_LEMMA",
                     "IS_ORIGINAL_NAME": False,
                     "WHY_NOT_ORIGINAL": ("the lemma is extracted from a WordNet sense key held in "
                                          "a /user/ namespace, not from /type/object/name. WordNet's "
                                          "sense-key format is documented and the lemma is the "
                                          "substring before '%', so the extraction is a decode "
                                          "rather than a guess -- but the string is WordNet's, not "
                                          "Freebase's.")},
       "CARRIED_FORWARD": {"external_authority_ids": len(flat),
                           "namespaces": sorted({b for b in auth_rows}),
                           "NOTE": ("these resolve to names only by querying the issuing authority. "
                                    "They are extracted here and left unresolved; resolving them is "
                                    "the EXTERNAL_ID_EXACT cascade tier.")},
       "REFUSED": {"freeq_other_text": bucket.get("freeq_other_text", 0),
                   "WHY": ("freeq var values decode to plausible titles but are lossy at both ends: "
                           "'_var_pringfield (medley): ... / Gracie Fi' is missing the leading S of "
                           "Springfield and is cut at a key-length limit, and '_var_kit #1' is "
                           "Skit #1. A truncated string is not the name."),
                   "REVISIT_IF": "the freeq job schemas are ever recovered, which would say what "
                                 "each var actually denotes"},
       "URI_REFERENCES": {"nodes": None,
                          "WHAT": ("/uri/ keys hold a URL for the node -- amazon ASINs, discogs "
                                   "releases, sfmoma and artic collection objects, and defunct "
                                   "citysearch/digitalcity pages. A URL is not a name, but it is a "
                                   "documented pointer to a page that carried one, so these feed "
                                   "the ARCHIVED_WEB tier rather than being counted here."),
                          "NOTE": "count is filled from nodes_by_class['URI_REFERENCE']"},
       "EVIDENCE_OF_NEVER_NAMED": {
           "WHAT": ("keys written by the loader that created the node, which identify it as a "
                    "reified relationship or a load artifact rather than an entity."),
           "freeq_musicbrainz_cvt": bucket.get("freeq_musicbrainz_cvt", 0),
           "freeq_declared_dummy_cvt": bucket.get("freeq_declared_dummy_cvt", 0),
           "freeq_mid_pair": bucket.get("freeq_mid_pair", 0),
           "freeq_hex_blob": bucket.get("freeq_hex_blob", 0),
           "WHY_THIS_IS_STRONGER_THAN_OUR_LABEL": (
               "our CVT_MEDIATOR label is a structural inference from relation types. These nodes "
               "are classified ENTITY_MID by that inference, yet the loader's own key says "
               "'_cvt_id_' or 'dummy_cvt_' or encodes a MID pair plus a date. The source declares "
               "what the inference had to guess, and it disagrees with the inference."),
           "CONSEQUENCE": ("a share of the 17.8M 'entity' residue is internal records, so the "
                           "denominator for name recovery is smaller than 17.8M. How much smaller "
                           "is reported above and is NOT folded into any coverage percentage here.")},
       "elapsed_s": round(time.time() - t0, 1)}
rec["URI_REFERENCES"]["nodes"] = nodes_by_class.get("URI_REFERENCE", 0)
with io.open(f"{V3}/V3_RESIDUE_KEY_CENSUS.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)

print(f"\nnodes by class: {dict(nodes_by_class)}")
print(f"WordNet lemmas recovered: {len(wn_rows):,}")
print(f"external authority ids carried forward: {len(flat):,}")
print("\nkey rows by bucket:")
for b, c in bucket.most_common(20):
    print(f"  {c:>10,}  {b}")
print(f"{time.time()-t0:.0f}s")
