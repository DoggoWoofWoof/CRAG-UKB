"""The cascade union: one consolidated name table across every source acquired so far.

    PYTHONHASHSEED=0 python .../cascade_union.py

WHY THIS EXISTS AS ITS OWN PASS
  Each acquisition step reported its own reach, and those numbers cannot be added. It was already
  demonstrated once in this campaign that they cannot: IDIR's two label tables looked like
  2,259 + 1,580 = 3,839 recovered nodes, and the intersection showed the second added exactly zero.
  So the total is recomputed here from the tables themselves, every time a source lands, rather than
  carried forward as arithmetic on previous totals.

WHAT IT PRODUCES
  cascade_names.parquet -- node_uid, node_id, display_name, source, is_original_name.
  One row per node, attributed to the HIGHEST-priority source that reached it. This is the table
  that a semantic overlay would consume; it is written under _acquisition/ and nothing downstream
  is built from it here.

THE TWO TIERS THAT WERE PROVISIONAL ARE NOW RULED ON
  FREEBASE_ALIAS is a PERMANENT EXACT-SOURCE TIER. /common/topic/alias is text Freebase itself
  attached to the MID, so it belongs above every third-party mapping and carries is_original_name
  true, with the provenance keeping it distinguishable from /type/object/name.
  WORDNET_SENSE_LEMMA is SOURCE-DERIVED EXACT, explicitly NOT an original /type/object/name: the
  lemma is deterministically recoverable from the WordNet sense key the graph already stores, so it
  needs no external source, but the value is WordNet's lemma and the namespace is /user/jamie/.
  is_original_name stays false and the tier name says where the string came from.

THE URL TIERS ARE KEPT APART
  EXTERNAL_URL_LIVE_EXACT and EXTERNAL_URL_ARCHIVE_EXACT are separate tiers, not one ARCHIVED_WEB
  bucket, because "the page still exists and says this" and "the page is gone and a 2004 capture
  says this" are different evidential claims and the paper has to be able to report them apart.

Reads the frozen graph and the frozen overlay. Writes neither.
"""
import sys, io, os, glob, json, time, collections, html
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import re, unicodedata
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

V3 = "data/final_canonical/freebase_v3"
ACQ = f"{V3}/_acquisition"
RESIDUE_ENTITY_MIDS = 17_809_849
t0 = time.time()

if os.environ.get("PYTHONHASHSEED") != "0":
    sys.exit("refusing to run without PYTHONHASHSEED=0: node_uid is hash(node_id)")

# (tier, [file globs], id column, name column, is_original_name).
# A tier with an empty file list has no source acquired yet and is reported as absent, not skipped.
CASCADE = [
    # ORDER IS LOCKED by V3_NAME_PROVENANCE_ORDER_V2.json (user ruling 2026-09-08),
    # sha c4a2a85e461084a459650cfe084293211429cba7217e9590cff7dd8604eec0e7.
    # V1 is superseded, not edited, and is retained as the record of what was locked before.
    # Earlier tier wins the display name; every tier is still retained as evidence.
    ("FREEBASE_CURRENT_NAME",     [],                                "node_id", None, True),
    # V2 RULING: FREEBASE_HISTORICAL_DUMP_NAME, FREEBASE_DELETED_NAME and
    # FREEBASE_HISTORICAL_PAGE_NAME are NOT three ranks. They are one class of ORIGINAL Freebase
    # assertions of /type/object/name, differing only in the artifact that preserved them -- a
    # published snapshot, a deletion log, a rendered page. Ranking one artifact as intrinsically
    # more authoritative than another is not justified, so they collapse here and the choice
    # between them is made by TIME, in historical_assertion_union.py:
    #     english if present -> else lexicographically first language -> then latest
    #     last_attested_ts (valid_until for deleted, snapshot_ts for a dump, capture_ts for a page)
    # That matters: on the current evidence 53,804 nodes get a DIFFERENT display name under
    # chronology than under any rigid source order. Source establishes authenticity; time decides
    # which historical name is shown.
    # A deleted name carries assertion_status = DELETED. It is NOT penalised as non-original -- it
    # was an original Freebase name -- the flag exists only so a later surviving assertion wins.
    ("FREEBASE_HISTORICAL_ASSERTION", ["historical_assertion_display/*.parquet"],
     "node_id", "display_name", True),
    ("FREEBASE_ALIAS",            ["freebase_alias_names.parquet"],
     "node_id", "display_name", True),
    ("WORDNET_SENSE_LEMMA",       ["wordnet_names.parquet"],         "node_id", "display_name",
     False),
    ("IDIR_FREEBASE_NAME",        ["idir_names.parquet"],            "node_id", "display_name",
     False),
    ("IDIR_CANONICAL_LABEL",      ["idir_entities_id_label.parquet"], "node_id", "value", False),
    ("SAMSUNG_WIKIDATA_EXACT",    ["samsung_names.parquet"],         "node_id", "display_name",
     False),
    # THREE ROUTES, ONE TIER. Samsung's 2014 sameAs archive, current Wikidata P646/P2671, and
    # fb2w -- the Freebase-to-Wikidata mapping Google itself published in November 2013, from the
    # same Internet Archive item as the main dump. fb2w's provenance is the best of the three, but
    # provenance over the MAPPING is not provenance over the STRING: the label is still today's
    # Wikidata label, not a Freebase /type/object/name, so it sits in this tier and not higher.
    # fb2w was sized on 2026-09-07 and left unlabelled behind an explicit caveat; the JOIN_CONTROL
    # retrofit found it reaching 18,614 residue nodes against 11,573 for the two routes that had
    # been labelled. Converting it added 11,707 nodes no other Wikidata route reaches, and the two
    # older routes still reach 4,750 that fb2w misses -- so all three are kept, not consolidated.
    ("CURRENT_WIKIDATA_EXACT",    ["wikidata_current_names.parquet", "fb2w_names.parquet"],
     "node_id", "display_name", False),
    ("WIKIPEDIA_TITLE",           [],                                "node_id", None, False),
    # EXTERNAL_AUTHORITY_EXACT is the locked EXTERNAL_ID_EXACT tier under the name the campaign now
    # uses for it. Every row was obtained by looking up an identifier Freebase stored, at the
    # authority that issued it. No row here came from matching a name.
    # external_authority_mediawiki_pageids: 951 rows the small-authority stage had parked as
    # "no_open_id_endpoint:mediawiki". They were page ids on wikis that do expose the action API;
    # the namespace label was classified, not the key value. Mostly Commons File: titles, which are
    # the attested identity of an IMAGE object -- every row carries surface_kind so a filename can
    # never be read as a proper name.
    ("EXTERNAL_AUTHORITY_EXACT",  ["external_authority_names.parquet",
                                   "external_authority_mediawiki.parquet",
                                   "external_authority_mediawiki_pageids.parquet",
                                   "external_authority_musicbrainz.parquet",
                                   "external_authority_small.parquet",
                                   "uri_authority_names.parquet"],
     "freebase_mid", "resolved_label", False),
    ("DBPEDIA_EXACT",             [],                                "node_id", None, False),
    ("FACC1_ATTESTED_SURFACE",    ["facc1_names.parquet"],           "node_id", "display_name",
     False),
    ("EXTERNAL_URL_LIVE_EXACT",   ["@uri_live"],                     "freebase_mid",
     "resolved_label", False),
    ("EXTERNAL_URL_ARCHIVE_EXACT", ["@uri_archive"],                 "freebase_mid",
     "resolved_label", False),
]

# A rung that proves the identity came from an authority's index rather than from a retrieved page.
# The discogs stage of uri_fetch extracts the release id out of the URL and calls
# api.discogs.com/releases/<id>; no page is fetched, so those rows are EXTERNAL_AUTHORITY_EXACT and
# are moved there regardless of the tier string the shard was written with.
RUNG_IS_AUTHORITY = {"discogs_api"}

z = np.load(f"{ACQ}/_unresolved_population.npz")
U, KC = z["U"], z["KC"]
NU = len(U)
print(f"residue population {NU:,}  (ENTITY_MID {int((KC==0).sum()):,})  ({time.time()-t0:.0f}s)",
      flush=True)


def norm(x):
    """Fold to the comparison form: accents, case and punctuation are not disagreements."""
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", x)).strip()


# ---------------------------------------------------------------- the per-row-tier URL shards
URLROWS = collections.defaultdict(list)          # tier -> [(node_uid, mid, label)]
# Rows uri_resolve.py's validator has quarantined (redirect-away landing pages, site-name titles,
# collapsed redirects) are excluded here by (stage, node_uid); they are not names.
QUAR = set()
if os.path.exists(f"{ACQ}/uri_quarantine.parquet"):
    q = pq.read_table(f"{ACQ}/uri_quarantine.parquet")
    QUAR = set(zip(q["stage"].to_pylist(), q["node_uid"].to_pylist()))
# DISCOGS ARTIST/LABEL URLS ARE NAME SLUGS, NOT IDS -- see V3_DISCOGS_ARTIST_SLUG_DEFECT.
# uri_fetch's discogs pattern was unanchored: r"discogs\.com/(release|master|artist|label)/(\d+)".
# Every discogs URL Freebase stored uses the OLD scheme, where /artist/<seg> and /label/<seg> carry
# the artist's NAME. Bands whose names begin with digits -- 112, 707, 1927, 35007, 08001, 3-D, 4-4-2
# -- had a FRAGMENT of the name matched as a numeric id and looked up at
# api.discogs.com/artists/<n>, which returned a completely unrelated artist. Every /artist/2... came
# back "Mr. James Barth & A.D." (26 nodes) and every /artist/3... came back "Josh Wink" (13). All
# 168 artist rows on disk are invalid, including the 38 whose slug happens to be all digits -- those
# are band names too, not ids. The rejection is by URL shape because it is exact: there is no
# identifier in those URLs to recover, so no row from them can ever be admissible.
DISCOGS_NAME_SLUG = re.compile(r"discogs\.com/(artist|label)/", re.I)

# A SITE TITLE IS NOT AN ENTITY NAME -- see V3_URL_TIER_SITE_TITLES. When a deep listing URL dies,
# the host serves its front page and the extractor reads THAT page's title, so one string lands on
# many unrelated nodes. Three such strings are present, each verified against its own rows:
#   'CitySearch - Your local city guide'  100 rows, 100 DISTINCT live listing URLs across 5 city
#                                         subdomains. A site tagline, not the name of 100 businesses.
#   'Welcome to Ocnow!'                    12 rows, all ocnow.com restaurant pages. Same pattern.
#   'Iowa General Assembly - ():'           5 rows. The site's member template with the name and
#                                         party slots EMPTY -- there is no name in the string at all.
# This is an exact list, not a heuristic, and it is deliberately not a rule about shared labels:
# 32 nodes legitimately share 'Burger King' and 70 legitimately share an SFMOMA portfolio title.
# Sharing is a screen; only these three were shown to carry no entity content.
SITE_TITLE_NOT_A_NAME = {"CitySearch - Your local city guide",
                         "Welcome to Ocnow!",
                         "Iowa General Assembly - ():"}

n_quar = 0
n_slug = 0
n_site = 0
shards = sorted(glob.glob(f"{ACQ}/uri_*_hits/*.parquet"))
for fp in shards:
    stage = os.path.basename(os.path.dirname(fp))[: -len("_hits")]
    t = pq.read_table(fp)
    _url = t["url"].to_pylist() if "url" in t.schema.names else [""] * t.num_rows
    for u, m, lab, rung, tier, uu in zip(t["node_uid"].to_pylist(), t["freebase_mid"].to_pylist(),
                                         t["resolved_label"].to_pylist(),
                                         t["identity_rung"].to_pylist(), t["tier"].to_pylist(),
                                         _url):
        if (stage, u) in QUAR:
            n_quar += 1
            continue
        if uu and DISCOGS_NAME_SLUG.search(uu):
            n_slug += 1
            continue
        if (lab or "").strip() in SITE_TITLE_NOT_A_NAME:
            n_site += 1
            continue
        lab = re.sub(r"\s+", " ", html.unescape(lab or "")).strip()   # '&amp;' -> '&'
        if not lab:
            continue
        eff = "EXTERNAL_AUTHORITY_EXACT" if rung in RUNG_IS_AUTHORITY else tier
        URLROWS[eff].append((u, m, lab))
if shards:
    print(f"url shards: {len(shards)} files ({n_quar:,} quarantined, {n_slug:,} discogs "
          f"name-slug, {n_site:,} site-title rows excluded) -> " +
          ", ".join(f"{k} {len(v):,}" for k, v in sorted(URLROWS.items())), flush=True)
SPECIAL = {"@uri_live": "EXTERNAL_URL_LIVE_EXACT", "@uri_archive": "EXTERNAL_URL_ARCHIVE_EXACT"}


def load(files, idcol, namecol, tier):
    """-> (node_uids, node_ids, names). Missing files are simply absent, not an error: a tier whose
    acquisition has not run yet must report zero rather than abort the union."""
    uu, nn, vv = [], [], []
    for fn in files:
        if fn in SPECIAL:
            src = URLROWS.get(SPECIAL[fn], [])
            if tier == "EXTERNAL_AUTHORITY_EXACT":
                src = []
            for a, b, c in src:
                uu.append(a), nn.append(b), vv.append(c)
            continue
        # a tier may be a single table or a directory of parts (fb2010_names/*.parquet)
        paths = sorted(glob.glob(f"{ACQ}/{fn}")) if "*" in fn else (
            [f"{ACQ}/{fn}"] if os.path.exists(f"{ACQ}/{fn}") else [])
        for fp in paths:
            t = pq.read_table(fp, columns=["node_uid", idcol, namecol])
            uu += t["node_uid"].to_pylist()
            nn += t[idcol].to_pylist()
            vv += t[namecol].combine_chunks().cast(pa.string()).to_pylist()
    if tier == "EXTERNAL_AUTHORITY_EXACT":
        for a, b, c in URLROWS.get("EXTERNAL_AUTHORITY_EXACT", []):
            uu.append(a), nn.append(b), vv.append(c)
    return uu, nn, vv


claimed = {}                     # node_uid -> (node_id, name, tier, is_original)
per_tier, reach, absent, offres, corrob, nfiles = {}, {}, [], {}, {}, {}
for tier, files, idcol, namecol, orig in CASCADE:
    uu, nn, vv = ([], [], []) if namecol is None else load(files, idcol, namecol, tier)
    nfiles[tier] = len(uu)
    if not uu:
        absent.append(tier)
        per_tier[tier] = 0
        continue
    new, off, seen = 0, 0, set()
    agree = {"exact": 0, "substring": 0, "different": 0}
    for a, b, c in zip(uu, nn, vv):
        if not c:
            continue
        seen.add(a)
        # a source may name nodes that are NOT in the residue; those are not recoveries here
        j = int(np.searchsorted(U, a))
        if j >= NU or U[j] != a:
            off += 1
            continue
        if a in claimed:
            # A source that adds no NEW node is not thereby worthless: where it fires on a node a
            # higher tier already named, the two were derived independently, so their agreement is
            # evidence about the higher tier. Disagreement is not automatically an error either --
            # an alternate-language or colloquial title is a different string for the same thing.
            x, y = norm(claimed[a][1]), norm(c)
            agree["exact" if x == y else
                  "substring" if x and y and (x in y or y in x) else "different"] += 1
            continue
        claimed[a] = (b, c, tier, orig)
        new += 1
    corrob[tier] = agree
    per_tier[tier] = new
    reach[tier] = len(seen)
    offres[tier] = off
    print(f"  {tier:<28} +{new:>7,} new   (source rows {len(uu):,} reach {len(seen):,}, "
          f"{off:,} outside the residue)", flush=True)

rows = sorted(claimed.items())
kc = KC[np.searchsorted(U, np.fromiter((r[0] for r in rows), np.int64, len(rows)))] if rows else \
     np.array([], np.int8)
KIND = ["ENTITY_MID", "CVT_MEDIATOR", "OTHER"]
by_kind = {KIND[i]: int((kc == i).sum()) for i in range(3)}

if rows:
    pq.write_table(pa.table({
        "node_uid": pa.array([r[0] for r in rows], pa.int64()),
        "node_id": pa.array([r[1][0] for r in rows]),
        "display_name": pa.array([r[1][1] for r in rows]),
        "source": pa.array([r[1][2] for r in rows]).dictionary_encode(),
        "is_original_name": pa.array([r[1][3] for r in rows], pa.bool_()),
        "node_kind": pa.array([KIND[i] for i in kc.tolist()]).dictionary_encode()}),
        f"{ACQ}/cascade_names.parquet", compression="zstd")

n_ent = by_kind["ENTITY_MID"]
rec = {"schema": "CASCADE_UNION/v2",
       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "APPEND_ONLY": ("reads canonical/ and overlay_v1/, writes neither. RESOLUTION_OVERLAY_V1 "
                       "manifest hash 25b734fe9acf2ca74814cf9f3444757636f19100305daa28b3ec19a2fb27d865"
                       " unchanged."),
       "residue_population": int(NU),
       "residue_entity_mids": RESIDUE_ENTITY_MIDS,
       "CASCADE_ORDER": [c[0] for c in CASCADE],
       "TIER_RULINGS": {
           "FREEBASE_ALIAS": {
               "status": "PERMANENT_EXACT_SOURCE_TIER",
               "why": ("/common/topic/alias is text Freebase itself attached to the MID, so it "
                       "outranks every third-party mapping and sits directly below the Freebase "
                       "name tiers."),
               "is_original_name": True,
               "CAVEAT": ("the alias property, NOT /type/object/name. The tier name and the "
                          "provenance keep the two distinguishable in every row.")},
           "WORDNET_SENSE_LEMMA": {
               "status": "SOURCE_DERIVED_EXACT",
               "NOT": "an original /type/object/name",
               "why": ("the lemma is deterministically recoverable from the WordNet sense key "
                       "already stored in the frozen graph, so the tier needs no external source "
                       "at all and is placed above EXTERNAL_AUTHORITY_EXACT, which does."),
               "is_original_name": False,
               "CAVEAT": ("the value is WordNet's lemma and the namespace is /user/jamie/, a "
                          "user-contributed namespace, not a Freebase authority.")},
           "EXTERNAL_AUTHORITY_EXACT": {
               "status": "the locked EXTERNAL_ID_EXACT tier, renamed",
               "RULE": "exact identifier lookup only; no search-by-name at any stage",
               "ROWS_MOVED_HERE_FROM_A_URL_TIER": len(URLROWS.get("EXTERNAL_AUTHORITY_EXACT", [])),
               "why_moved": ("the discogs stage extracts the release id from the URL and calls the "
                             "Discogs API; no page is retrieved, so those rows are an authority "
                             "resolution and calling them a URL recovery would misdescribe them.")},
           "EXTERNAL_URL_LIVE_EXACT_vs_ARCHIVE": (
               "kept as two tiers rather than one ARCHIVED_WEB bucket. LIVE outranks ARCHIVE "
               "because it is a direct retrieval of the resource the URL names; the live stage "
               "refuses a title equal to the bare host, so a parked domain cannot outrank a real "
               "archived page.")},
       "TIERS_WITH_NO_SOURCE_YET": absent,
       "source_rows_read": nfiles,
       "nodes_new_at_this_tier": per_tier,
       "source_reach_ignoring_cascade": reach,
       "source_rows_outside_the_residue": offres,
       "CORROBORATION": {
           "counts": corrob,
           "MEANING": ("where a source fires on a node an earlier tier already named, the two names "
                       "were derived independently. Agreement is evidence for the earlier tier; it "
                       "is not a recovery and is not counted as one."),
           "NORMALISATION": "NFKD, ASCII-fold, lowercase, punctuation to space -- casing and "
                            "accents are not disagreements",
           "DIFFERENT_IS_NOT_WRONG": ("an alternate-language or colloquial title is a different "
                                      "string for the same entity, so 'different' is a bucket to "
                                      "inspect, not an error count.")},
       "TOTAL_DISTINCT_NAMED": len(claimed),
       "by_node_kind": by_kind,
       "pct_of_residue_population": round(100 * len(claimed) / NU, 6),
       "pct_of_residue_entity_mids": round(100 * n_ent / RESIDUE_ENTITY_MIDS, 6),
       "WHY_RECOMPUTED_NOT_ADDED": ("per-source reaches overlap and cannot be summed. IDIR's two "
                                    "label tables looked like 2,259 + 1,580 and the second added "
                                    "exactly zero; the total is therefore recomputed from the "
                                    "tables every time a source lands."),
       "ORIGINAL_NAME_ROWS": sum(1 for r in rows if r[1][3]),
       "EMPTY_TIERS_ARE_NOT_ALL_ALIKE": {
           "WIKIPEDIA_TITLE": "REFUTED, not pending. V3_KEY_NAMESPACE_CENSUS censused all "
                              "143,981,520 /type/object/key rows against the unnamed residue: "
                              "/wikipedia/en reaches 1 node, /wikipedia/en_title 1, every Wikipedia "
                              "TITLE namespace together about 15. Do not commission a decoder. "
                              "Unrelated and still live: the 56,285 wikipedia_en_page_id values "
                              "captured from ARCHIVED PAGES, resolvable by ?curid=.",
           "FREEBASE_CURRENT_NAME": "genuinely pending, not refuted.",
           "FREEBASE_DELETED_NAME": "genuinely pending, not refuted.",
           "DBPEDIA_EXACT": "genuinely pending, not refuted."},
       "WHAT_THIS_IS_NOT": ("not a claim that these are the /type/object/name values Freebase held. "
                            "Only rows with is_original_name true would be that, and no tier "
                            "producing them has a source acquired for the residue."),
       "elapsed_s": round(time.time() - t0, 1)}
with io.open(f"{V3}/V3_CASCADE_UNION.json", "w", encoding="utf-8") as f:
    json.dump(rec, f, indent=1, ensure_ascii=False)

print(f"\nCASCADE UNION")
# WIKIPEDIA_TITLE is the REFUTED case: V3_KEY_NAMESPACE_CENSUS censused all 143,981,520
# /type/object/key rows against the unnamed residue and /wikipedia/en reaches 1 node,
# /wikipedia/en_title 1, every Wikipedia TITLE namespace together about 15.  A node holding a
# Wikipedia title key kept its /type/object/name, so the unnamed residue does not carry them.
CLOSED = {"WIKIPEDIA_TITLE": "(REFUTED: ~15 residue nodes hold any Wikipedia title key"
                             " -- V3_KEY_NAMESPACE_CENSUS)"}
for tier, _, _, _, _ in CASCADE:
    mark = ("  " + CLOSED[tier]) if tier in CLOSED else (
        "  (no source yet)" if tier in absent else "")
    print(f"  {tier:<28} +{per_tier[tier]:>7,}{mark}")
print(f"  {'TOTAL DISTINCT NAMED':<28}  {len(claimed):>7,}   by kind {by_kind}")
print(f"  ENTITY_MID coverage       {rec['pct_of_residue_entity_mids']}% of {RESIDUE_ENTITY_MIDS:,}")
print(f"{time.time()-t0:.0f}s")
